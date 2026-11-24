"""Native AOT absorbed MLA kernels for Hy4-flash (32-head DSA and 64-head SWA-512).

Hy4-flash uses absorbed MLA with a 576-dim KV record (512 normalized latent +
64 interleaved-RoPE k_rope), identical in layout to Kimi/GLM (1152 B BF16 or
656 B FP8-DS), so it directly reuses ``DenseMlaForwardKernel`` + ``DenseMlaMergeKernel``:

- DSA layers (12): 32 heads (16/GPU TP2), ``page_size=1`` selected-token sparse gather
  over up to 2048 physical slots, ``softmax_scale = 256^-0.5 = 1/16``.
- SWA layers (30): 64 heads (32/GPU TP2), ``window_size=512``, ``page_size=64``
  circular ring, ``softmax_scale = 256^-0.5 = 1/16``.
- Exact F32 per-head learnable sinks: since causal attention always has >=1 visible
  key, ``O_sink = O_no_sink * sigmoid(lse_e - sink[h])`` is algebraically exact and
  folds directly into ``W_UV`` + ``sigmoid(linear_gate)`` with zero extra HBM pass.

Linear projections (``q_a``, ``kv_a``, ``q_b``, ``linear_gate``, ``o_proj``) remain
caller-invoked so BF16, EXL3 K8/K6 MUL1, and block-FP8 views share one attention ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32, fabs_f32, fmax_f32
from b12x.attention.dense_mla._forward import DenseMlaForwardKernel
from b12x.attention.dense_mla._layout import choose_layout, make_smem_layout
from b12x.attention.dense_mla._merge import DenseMlaMergeKernel
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, _bf16, _block_sum, _fp8_scale, _reduction_storage, _rsqrt, _warp_max
from .kimi_mla import _ContextPartialMerge

__all__ = [
    "Hy4MlaGeometry",
    "compile_hy4_mla_attention_aot",
    "compile_hy4_mla_dsa_partial_aot",
    "compile_hy4_mla_gate_uv_aot",
    "compile_hy4_mla_q_absorb_aot",
    "compile_hy4_mla_rank_norm_pack_aot",
    "compile_hy4_mla_sink_gate_uv_aot",
    "hy4_attention_layout",
    "hy4_dsa_partial_scratch_bytes",
]


@dataclass(frozen=True)
class Hy4MlaGeometry:
    heads: int = 32
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_dim: int = 192
    qk_rope_dim: int = 64
    v_head_dim: int = 256
    norm_eps: float = 1e-5
    page_rows: int = 64
    window_size: int | None = None
    kv_cache: str = "bf16"
    mrope_section: tuple = (12, 10, 10)

    def __post_init__(self) -> None:
        if self.kv_cache not in ("bf16", "fp8"):
            raise ValueError("Hy4 MLA kv_cache must be 'bf16' or 'fp8'")
        if self.heads not in (16, 32, 64):
            raise ValueError(f"Hy4 MLA heads must be 16, 32, or 64; got {self.heads}")
        if (self.q_lora_rank, self.kv_lora_rank, self.qk_nope_dim, self.qk_rope_dim,
                self.v_head_dim, self.page_rows) != (1024, 512, 192, 64, 256, 64):
            raise ValueError("unsupported Hy4 MLA dimensions")
        if self.window_size not in (None, 512):
            raise ValueError("Hy4 MLA window_size must be None (DSA) or 512 (SWA)")
        _validate_mrope(self.mrope_section, self.qk_rope_dim // 2)

    @property
    def record_bytes(self) -> int:
        return 656 if self.kv_cache == "fp8" else 1152

    @property
    def softmax_scale(self) -> float:
        return 256.0 ** -0.5


def _validate_mrope(section, pairs):
    if (not isinstance(section, tuple) or len(section) != 3
            or any(not isinstance(s, int) or s < 0 for s in section) or sum(section) != pairs):
        raise ValueError(f"Hy4 mrope_section must be three non-negative ints summing to {pairs}")


@cute.jit
def _mrope_position(positions: cute.Pointer, row: Int64, pair: Int64, s1: cutlass.Constexpr, s2: cutlass.Constexpr):
    """Interleaved M-RoPE: pair p reads H when p%3==1 and p<3*s1, W when p%3==2
    and p<3*s2, otherwise T, from ``positions[row, 3]``. Equal axes (text rows)
    give exactly the scalar-position angle."""
    axis = Int64(0)
    remainder = pair % Int64(3)
    if remainder == Int64(1):
        if pair < Int64(3 * s1):
            axis = Int64(1)
    if remainder == Int64(2):
        if pair < Int64(3 * s2):
            axis = Int64(2)
    return Float32(positions[row * Int64(3) + axis])


def _align(n: int) -> int:
    return (int(n) + 1023) // 1024 * 1024


@cute.jit
def _offset(n: Int64) -> Int64:
    return (n + Int64(1023)) // Int64(1024) * Int64(1024)


def hy4_dense_mla_layout(g: Hy4MlaGeometry, rows: int, *, mode: str):
    return choose_layout(
        mode=mode,
        fp8=False,
        qk_dim=576,
        value_dim=512,
        single_request=True,
        windowed=g.window_size is not None,
        query_rows=rows,
    )


def hy4_attention_layout(g: Hy4MlaGeometry, rows: int, *, max_context: int, mode: str) -> dict:
    if mode not in ("decode", "extend"):
        raise ValueError("mode must be 'decode' or 'extend'")
    chunk = hy4_dense_mla_layout(g, rows, mode=mode).candidates_per_chunk
    chunks_total = (int(max_context) + chunk - 1) // chunk
    splits = min(16, chunks_total) if mode == "decode" else 1
    chunks = (chunks_total + splits - 1) // splits
    part = _align(rows * g.heads * splits * 512 * 2) if splits > 1 else 0
    lse = _align(rows * g.heads * splits * 4) if splits > 1 else 0
    return dict(
        num_splits=splits,
        chunks_per_split=chunks,
        partial_output_bytes=part,
        partial_lse_bytes=lse,
        scratch_bytes=part + lse,
    )


def hy4_dsa_partial_scratch_bytes(
    rows: int,
    *,
    head_count: int = 16,
    num_splits: int = 16,
    row_chunk_size: int = 64,
) -> int:
    chunk_rows = min(max(int(rows), 1), int(row_chunk_size))
    n = chunk_rows * int(head_count) * int(num_splits)
    return _align(n * 512 * 2) + _align(n * 4)


class _Hy4RankNormPack:
    """RMSNorm q_a (1024) + kv_a (512), interleaved RoPE-64 on k_rope, write KV cache."""

    def __init__(self, g: Hy4MlaGeometry):
        self.g = g

    @cute.jit
    def __call__(
        self,
        q_a: cute.Pointer,
        kv_a: cute.Pointer,
        q_norm: cute.Pointer,
        kv_norm: cute.Pointer,
        slots: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        q_resid: cute.Pointer,
        kv_cache: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(q_a, kv_a, q_norm, kv_norm, slots, positions, inv_freq, q_resid, kv_cache).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        q_a: cute.Pointer,
        kv_a: cute.Pointer,
        q_norm: cute.Pointer,
        kv_norm: cute.Pointer,
        slots: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        q_resid: cute.Pointer,
        cache: cute.Pointer,
    ):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, 8))
        sums = storage.sums.get_tensor(cute.make_layout((2, 8), stride=(8, 1)))

        q = cute.make_rmem_tensor((4,), Float32)
        kv = cute.make_rmem_tensor((2,), Float32)
        qs, ks = Float32(0.0), Float32(0.0)
        for j in cutlass.range_constexpr(4):
            d = Int64(j * 256) + Int64(tid)
            q[j] = Float32(q_a[row * Int64(1024) + d])
            qs += q[j] * q[j]
        for j in cutlass.range_constexpr(2):
            d = Int64(j * 256) + Int64(tid)
            kv[j] = Float32(kv_a[row * Int64(576) + d])
            ks += kv[j] * kv[j]

        qi = _rsqrt(_block_sum(qs, sums, Int32(0), 8) / Float32(1024.0) + Float32(self.g.norm_eps))
        ki = _rsqrt(_block_sum(ks, sums, Int32(1), 8) / Float32(512.0) + Float32(self.g.norm_eps))

        for j in cutlass.range_constexpr(4):
            d = Int64(j * 256) + Int64(tid)
            q_resid[row * Int64(1024) + d] = (q[j] * qi * Float32(q_norm[d])).to(BFloat16)

        slot = Int64(slots[row])
        if slot >= Int64(0):
            if cutlass.const_expr(self.g.kv_cache == "fp8"):
                record = Int64(cache.toint()) + slot * Int64(self.g.record_bytes)
                if tid < Int32(128):
                    normalized = cute.make_rmem_tensor((4,), Float32)
                    local = Float32(0.0)
                    for j in cutlass.range_constexpr(4):
                        d = Int64(4) * Int64(tid) + Int64(j)
                        value = Float32(kv_a[row * Int64(576) + d])
                        normalized[j] = _bf16(value * ki * Float32(kv_norm[d]))
                        local = fmax_f32(local, fabs_f32(normalized[j]))
                    scale = _fp8_scale(_warp_max(local))
                    words = cute.make_ptr(Uint32, record, cute.AddressSpace.gmem, assumed_align=4)
                    words[tid] = cvt_f32x4_to_e4m3x4(
                        div_rn_f32(normalized[0], scale), div_rn_f32(normalized[1], scale),
                        div_rn_f32(normalized[2], scale), div_rn_f32(normalized[3], scale),
                    )
                    if tid % Int32(32) == Int32(0):
                        scales = cute.make_ptr(Float32, record + Int64(512), cute.AddressSpace.gmem, assumed_align=4)
                        scales[tid // Int32(32)] = scale
                if tid < Int32(32):
                    pair = Int64(tid)
                    angle = _mrope_position(positions, row, pair, self.g.mrope_section[1], self.g.mrope_section[2]) * Float32(inv_freq[pair])
                    c = cute.math.cos(angle, fastmath=False)
                    s = cute.math.sin(angle, fastmath=False)
                    x0 = Float32(kv_a[row * Int64(576) + Int64(512) + pair * Int64(2)])
                    x1 = Float32(kv_a[row * Int64(576) + Int64(512) + pair * Int64(2) + Int64(1)])
                    tail = cute.make_ptr(BFloat16, record + Int64(528), cute.AddressSpace.gmem, assumed_align=2)
                    tail[pair * Int64(2)] = (x0 * c - x1 * s).to(BFloat16)
                    tail[pair * Int64(2) + Int64(1)] = (x1 * c + x0 * s).to(BFloat16)
            else:
                for j in cutlass.range_constexpr(2):
                    d = Int64(j * 256) + Int64(tid)
                    cache[slot * Int64(576) + d] = (kv[j] * ki * Float32(kv_norm[d])).to(BFloat16)
                if tid < Int32(32):
                    pair = Int64(tid)
                    angle = _mrope_position(positions, row, pair, self.g.mrope_section[1], self.g.mrope_section[2]) * Float32(inv_freq[pair])
                    c = cute.math.cos(angle, fastmath=False)
                    s = cute.math.sin(angle, fastmath=False)
                    x0 = Float32(kv_a[row * Int64(576) + Int64(512) + pair * Int64(2)])
                    x1 = Float32(kv_a[row * Int64(576) + Int64(512) + pair * Int64(2) + Int64(1)])
                    cache[slot * Int64(576) + Int64(512) + pair * Int64(2)] = (x0 * c - x1 * s).to(BFloat16)
                    cache[slot * Int64(576) + Int64(512) + pair * Int64(2) + Int64(1)] = (x1 * c + x0 * s).to(BFloat16)


class _Hy4QueryTailRope:
    """Interleaved M-RoPE-64 on q_rope [rows, H, 192..256] -> query [rows, H, 512..576]."""

    def __init__(self, heads: int, mrope_section: tuple = (12, 10, 10)):
        self.heads = heads
        self.mrope_section = mrope_section

    @cute.jit
    def __call__(
        self,
        q_raw: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        query: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(q_raw, positions, inv_freq, query).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, q_raw: cute.Pointer, positions: cute.Pointer, inv_freq: cute.Pointer, query: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        pair = Int64(cute.arch.thread_idx()[0])
        angle = _mrope_position(positions, Int64(row), pair, self.mrope_section[1], self.mrope_section[2]) * Float32(inv_freq[pair])
        c = cute.math.cos(angle, fastmath=False)
        s = cute.math.sin(angle, fastmath=False)
        h = Int64(row) * Int64(self.heads) + Int64(head)
        src = h * Int64(256) + Int64(192) + pair * Int64(2)
        dst = h * Int64(576) + Int64(512) + pair * Int64(2)
        x0 = Float32(q_raw[src])
        x1 = Float32(q_raw[src + Int64(1)])
        query[dst] = (x0 * c - x1 * s).to(BFloat16)
        query[dst + Int64(1)] = (x1 * c + x0 * s).to(BFloat16)


class _Hy4QAbsorb:
    def __init__(self, g: Hy4MlaGeometry):
        self.uk = BatchedBf16Gemm(
            n=512, k=192, batch=g.heads,
            a_row=g.heads * 256, a_batch=256,
            o_row=g.heads * 576, o_batch=576,
        )
        self.tail = _Hy4QueryTailRope(g.heads, g.mrope_section)

    @cute.jit
    def __call__(
        self,
        q_raw: cute.Pointer,
        w_uk: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        query: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.uk(q_raw, w_uk, query, rows, stream)
        self.tail(q_raw, positions, inv_freq, query, rows, stream)


class _Hy4SinkGateMul:
    """Fold exact F32 sink rescaling sigmoid(lse_e - sink[h]) * sigmoid(gate) into UV output."""

    def __init__(self, heads: int, v_dim: int = 256):
        self.heads = heads
        self.v_dim = v_dim

    @cute.jit
    def __call__(
        self,
        uv: cute.Pointer,
        gate: cute.Pointer,
        lse: cute.Pointer,
        sink: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(uv, gate, lse, sink, out).launch(
            grid=(rows, self.heads, 1), block=(self.v_dim, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, uv: cute.Pointer, gate: cute.Pointer, lse: cute.Pointer, sink: cute.Pointer, out: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        d = Int64(cute.arch.thread_idx()[0])
        rh = row * Int64(self.heads) + head
        idx = rh * Int64(self.v_dim) + d
        lse_val = Float32(lse[rh])
        sink_val = Float32(sink[head])
        # Exact causal sink weight: 1 / (1 + exp(sink - lse_e))
        sink_scale = Float32(1.0) / (Float32(1.0) + cute.math.exp(sink_val - lse_val, fastmath=False))
        g_val = Float32(gate[idx])
        g_sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-g_val, fastmath=False))
        out[idx] = (Float32(uv[idx]) * sink_scale * g_sig).to(BFloat16)


class _Hy4SinkGateUv:
    def __init__(self, g: Hy4MlaGeometry):
        self.uv = BatchedBf16Gemm(
            n=256, k=512, batch=g.heads,
            a_row=g.heads * 512, a_batch=512,
            o_row=g.heads * 256, o_batch=256,
        )
        self.mul = _Hy4SinkGateMul(g.heads, 256)

    @cute.jit
    def __call__(
        self,
        attn: cute.Pointer,
        w_uv: cute.Pointer,
        gate: cute.Pointer,
        lse: cute.Pointer,
        sink: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.uv(attn, w_uv, out, rows, stream)
        self.mul(out, gate, lse, sink, out, rows, stream)


class _Hy4GateMul:
    """Apply sigmoid(gate) to UV output when sink was already folded by lse_combine2."""

    def __init__(self, heads: int, v_dim: int = 256):
        self.heads = heads
        self.v_dim = v_dim

    @cute.jit
    def __call__(
        self,
        uv: cute.Pointer,
        gate: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(uv, gate, out).launch(
            grid=(rows, self.heads, 1), block=(self.v_dim, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, uv: cute.Pointer, gate: cute.Pointer, out: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        d = Int64(cute.arch.thread_idx()[0])
        rh = row * Int64(self.heads) + head
        idx = rh * Int64(self.v_dim) + d
        g_val = Float32(gate[idx])
        g_sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-g_val, fastmath=False))
        out[idx] = (Float32(uv[idx]) * g_sig).to(BFloat16)


class _Hy4GateUv:
    def __init__(self, g: Hy4MlaGeometry):
        self.uv = BatchedBf16Gemm(
            n=256, k=512, batch=g.heads,
            a_row=g.heads * 512, a_batch=512,
            o_row=g.heads * 256, o_batch=256,
        )
        self.mul = _Hy4GateMul(g.heads, 256)

    @cute.jit
    def __call__(
        self,
        attn: cute.Pointer,
        w_uv: cute.Pointer,
        gate: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.uv(attn, w_uv, out, rows, stream)
        self.mul(out, gate, out, rows, stream)


class _Hy4DsaPartial:
    """Sink-free DSA partial over owned page_size=1 slots with base-2 FP32 LSE.

    Reads a [head_begin, head_begin + head_count) slice from the full 32-head query
    and writes compact [rows, head_count, 512] BF16 partial output plus base-2 FP32
    LSE [rows, head_count]. Empty rows (cache_seqlens[r] == 0) emit zero and -inf.
    """

    def __init__(
        self,
        g: Hy4MlaGeometry,
        *,
        head_begin: int,
        head_count: int,
        num_splits: int,
        heads_per_tile: int = 8,
    ):
        self.g = g
        self.begin = int(head_begin)
        self.heads = int(head_count)
        self.splits = int(num_splits)
        self.smem = make_smem_layout(query_tile=1, fp8=False, heads_per_tile=heads_per_tile)
        chunks_total = (2048 + self.smem.candidates_per_chunk - 1) // self.smem.candidates_per_chunk
        self.chunks_per_split = (chunks_total + self.splits - 1) // self.splits
        self.forward = DenseMlaForwardKernel(
            layout=self.smem,
            page_size=1,
            num_heads=self.heads,
            num_splits=self.splits,
            chunks_per_split=self.chunks_per_split,
            query_tile=1,
            fp8=False,
            qk_dim=576,
            value_dim=512,
            window_size=None,
            noncausal_block=False,
            ds_mla_record=(g.kv_cache == "fp8"),
            dynamic_split_chunks=True,
            singleton_requests=True,
        )
        self.merge = _ContextPartialMerge(self.heads, self.splits)

    @cute.jit
    def run_partial(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Pointer,
        cache_seqlens: cute.Pointer,
        cu_seqlens_q: cute.Pointer,
        attn: cute.Pointer,
        lse: cute.Pointer,
        scratch: cute.Pointer,
        rows: Int32,
        page_table_width: Int64,
        active_splits: Int32,
        stream: cuda.CUstream,
    ):
        g, h, s = self.g, self.heads, self.splits
        # DenseMlaForwardKernel only writes base-2 partial_lse when active_splits > 1;
        # clamp to [2, s] so single-split natural-log final output can never leak onto the wire.
        splits = active_splits if active_splits >= Int32(2) else Int32(2)
        if splits > Int32(s):
            splits = Int32(s)
        q = cute.make_tensor(
            cute.recast_ptr(query + Int64(self.begin * 576), dtype=cutlass.Uint8),
            cute.make_layout((1,)),
        )
        kv = cute.make_tensor(cute.recast_ptr(kv_cache, dtype=cutlass.Uint8), cute.make_layout((1,)))
        table = cute.make_tensor(page_table, cute.make_layout((Int64(rows) * page_table_width,)))
        lengths = cute.make_tensor(cache_seqlens, cute.make_layout((rows,)))
        cu = cute.make_tensor(cu_seqlens_q, cute.make_layout((rows + Int32(1),)))
        out = cute.make_tensor(attn, cute.make_layout((rows, h, 512), stride=(h * 512, 512, 1)))
        lo = cute.make_tensor(lse, cute.make_layout((rows, h), stride=(h, 1)))
        po = cute.make_tensor(
            cute.recast_ptr(scratch, dtype=BFloat16),
            cute.make_layout((rows, h, s, 512), stride=(h * s * 512, s * 512, 512, 1)),
        )
        pl_ptr = cute.make_ptr(
            Float32,
            Int64(scratch.toint()) + _offset(Int64(rows) * Int64(h * s * 512 * 2)),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        pl = cute.make_tensor(pl_ptr, cute.make_layout((rows, h, s), stride=(h * s, s, 1)))
        self.forward(
            q, kv, table, lengths, cu, out, lo, po, pl, lo, lo,
            Float32(g.softmax_scale * math.log2(math.e)),
            Int64(g.heads * 576 * 2),
            Int64(576 * 2),
            Int64(g.record_bytes),
            Int64(g.record_bytes),
            page_table_width,
            rows,
            rows,
            splits,
            stream,
        )
        self.merge(po, pl, attn, lse, rows, splits, stream)


class _Hy4DsaPartialChunks(_Hy4DsaPartial):
    def __init__(
        self,
        g: Hy4MlaGeometry,
        *,
        max_rows: int,
        head_begin: int,
        head_count: int,
        num_splits: int,
        row_chunk_size: int,
        heads_per_tile: int = 8,
    ):
        self.row_chunk_size = min(int(max_rows), int(row_chunk_size))
        super().__init__(
            g,
            head_begin=head_begin,
            head_count=head_count,
            num_splits=num_splits,
            heads_per_tile=heads_per_tile,
        )

    def scratch_bytes(self, rows: int) -> int:
        return hy4_dsa_partial_scratch_bytes(
            rows,
            head_count=self.heads,
            num_splits=self.splits,
            row_chunk_size=self.row_chunk_size,
        )

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Pointer,
        cache_seqlens: cute.Pointer,
        cu_seqlens_q: cute.Pointer,
        attn: cute.Pointer,
        lse: cute.Pointer,
        scratch: cute.Pointer,
        rows: Int32,
        page_table_width: Int64,
        active_splits: Int32,
        stream: cuda.CUstream,
    ):
        for first in cutlass.range(0, rows, self.row_chunk_size):
            count = min(Int32(self.row_chunk_size), rows - first)
            row = Int64(first)
            self.run_partial(
                query + row * Int64(self.g.heads * 576),
                kv_cache,
                page_table + row * page_table_width,
                cache_seqlens + row,
                cu_seqlens_q,
                attn + row * Int64(self.heads * 512),
                lse + row * Int64(self.heads),
                scratch,
                count,
                page_table_width,
                active_splits,
                stream,
            )


class _Hy4Attention:
    def __init__(self, g: Hy4MlaGeometry, rows: int, max_context: int, mode: str, *, page_size: int):
        self.g = g
        self.page_size = int(page_size)
        self.layout = hy4_attention_layout(g, rows, max_context=max_context, mode=mode)
        self.splits = self.layout["num_splits"]
        self.smem = hy4_dense_mla_layout(g, rows, mode=mode)
        self.forward = DenseMlaForwardKernel(
            layout=self.smem,
            page_size=self.page_size,
            num_heads=g.heads,
            num_splits=self.splits,
            chunks_per_split=self.layout["chunks_per_split"],
            query_tile=self.smem.query_tile,
            fp8=False,
            qk_dim=576,
            value_dim=512,
            window_size=g.window_size,
            noncausal_block=False,
            ds_mla_record=g.kv_cache == "fp8",
            dynamic_split_chunks=True,
            singleton_requests=(self.page_size == 1),
        )
        self.merge = DenseMlaMergeKernel(num_splits=self.splits, value_dim=512) if self.splits > 1 else None

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Pointer,
        cache_seqlens: cute.Pointer,
        cu_seqlens_q: cute.Pointer,
        attn: cute.Pointer,
        lse: cute.Pointer,
        scratch: cute.Pointer,
        rows: Int32,
        batch: Int32,
        page_table_width: Int64,
        active_splits: Int32,
        stream: cuda.CUstream,
    ):
        g = self.g
        q = cute.make_tensor(cute.recast_ptr(query, dtype=cutlass.Uint8), cute.make_layout((1,)))
        kv = cute.make_tensor(cute.recast_ptr(kv_cache, dtype=cutlass.Uint8), cute.make_layout((1,)))
        table = cute.make_tensor(page_table, cute.make_layout((Int64(batch) * page_table_width,)))
        lengths = cute.make_tensor(cache_seqlens, cute.make_layout((batch,)))
        cu = cute.make_tensor(cu_seqlens_q, cute.make_layout((batch + Int32(1),)))
        out = cute.make_tensor(attn, cute.make_layout((rows, g.heads, 512), stride=(g.heads * 512, 512, 1)))
        lse_out = cute.make_tensor(lse, cute.make_layout((rows, g.heads), stride=(g.heads, 1)))
        if cutlass.const_expr(self.splits > 1):
            po = cute.make_tensor(
                cute.recast_ptr(scratch, dtype=BFloat16),
                cute.make_layout(
                    (rows, g.heads, self.splits, 512),
                    stride=(g.heads * self.splits * 512, self.splits * 512, 512, 1),
                ),
            )
            pl_ptr = cute.make_ptr(
                Float32,
                Int64(scratch.toint()) + _offset(Int64(rows) * Int64(g.heads * self.splits * 512 * 2)),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            pl = cute.make_tensor(
                pl_ptr,
                cute.make_layout((rows, g.heads, self.splits), stride=(g.heads * self.splits, self.splits, 1)),
            )
        else:
            po, pl = out, lse_out
        scale_log2 = Float32(g.softmax_scale * math.log2(math.e))
        self.forward(
            q, kv, table, lengths, cu, out, lse_out, po, pl, lse_out, lse_out,
            scale_log2, Int64(g.heads * 576 * 2), Int64(576 * 2),
            Int64(self.page_size * g.record_bytes), Int64(g.record_bytes),
            page_table_width, rows, batch, active_splits, stream,
        )
        if cutlass.const_expr(self.splits > 1):
            if active_splits > Int32(1):
                self.merge(po, pl, out, lse_out, active_splits, stream)


def compile_hy4_mla_rank_norm_pack_aot(g: Hy4MlaGeometry):
    operands = (
        Operand("q_a", torch.bfloat16, "[rows,1024]"),
        Operand("kv_a", torch.bfloat16, "[rows,576]"),
        Operand("q_norm", torch.bfloat16, "[1024]"),
        Operand("kv_norm", torch.bfloat16, "[512]"),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("positions", torch.int64, "[rows,3]", align=8, note="M-RoPE (T,H,W) rotary positions"),
        Operand("inv_freq", torch.float32, "[32]", align=4),
        Operand("q_resid", torch.bfloat16, "[rows,1024]", "out"),
        Operand(
            "kv_cache",
            torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
            "[pages,64,656]" if g.kv_cache == "fp8" else "[pages,64,576]",
            "inout",
        ),
    )
    return compile_program(
        _Hy4RankNormPack(g),
        name=f"hy4_mla_rank_norm_pack_{g.kv_cache}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g.kv_cache, g.norm_eps, "rank_norm_pack", g.mrope_section),
        geometry={"kv_cache": g.kv_cache, "norm_eps": g.norm_eps, "record_bytes": g.record_bytes,
                  "mrope_section": list(g.mrope_section)},
        doc=__doc__,
    )


def compile_hy4_mla_q_absorb_aot(g: Hy4MlaGeometry):
    operands = (
        Operand("q_raw", torch.bfloat16, f"[rows,{g.heads},256]"),
        Operand("w_uk", torch.bfloat16, f"[{g.heads},512,192]"),
        Operand("positions", torch.int64, "[rows,3]", align=8, note="M-RoPE (T,H,W) rotary positions"),
        Operand("inv_freq", torch.float32, "[32]", align=4),
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]", "out"),
    )
    return compile_program(
        _Hy4QAbsorb(g),
        name=f"hy4_mla_q_absorb_h{g.heads}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g.heads, "q_absorb", g.mrope_section),
        geometry={"heads": g.heads, "qk_nope_dim": 192, "qk_rope_dim": 64,
                  "mrope_section": list(g.mrope_section)},
        doc=__doc__,
    )


def compile_hy4_mla_sink_gate_uv_aot(g: Hy4MlaGeometry):
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]"),
        Operand("w_uv", torch.bfloat16, f"[{g.heads},256,512]"),
        Operand("gate", torch.bfloat16, f"[rows,{g.heads},256]"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", align=4),
        Operand("sink", torch.float32, f"[{g.heads}]", align=4),
        Operand("out", torch.bfloat16, f"[rows,{g.heads},256]", "out"),
    )
    return compile_program(
        _Hy4SinkGateUv(g),
        name=f"hy4_mla_sink_gate_uv_h{g.heads}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g.heads, "sink_gate_uv"),
        geometry={"heads": g.heads, "v_head_dim": 256, "kv_lora_rank": 512},
        doc=__doc__,
    )


def compile_hy4_mla_gate_uv_aot(g: Hy4MlaGeometry):
    operands = (
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]"),
        Operand("w_uv", torch.bfloat16, f"[{g.heads},256,512]"),
        Operand("gate", torch.bfloat16, f"[rows,{g.heads},256]"),
        Operand("out", torch.bfloat16, f"[rows,{g.heads},256]", "out"),
    )
    return compile_program(
        _Hy4GateUv(g),
        name=f"hy4_mla_gate_uv_h{g.heads}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g.heads, "gate_uv"),
        geometry={"heads": g.heads, "v_head_dim": 256, "kv_lora_rank": 512},
        doc=__doc__,
    )


def compile_hy4_mla_dsa_partial_aot(
    g: Hy4MlaGeometry,
    *,
    max_rows: int = 4096,
    head_begin: int = 0,
    head_count: int = 16,
    num_splits: int = 16,
    row_chunk_size: int = 64,
):
    """Sink-free Hy4 DSA ContextSplit partial + base-2 FP32 LSE over owned page_size=1 slots.

    For every DSA layer on 2 GPUs, each rank launches both ``head_begin=0`` and ``head_begin=16``
    over its own parity-owned pages (32 heads total over local context), keeps its own 16-head
    partial, and pushes the peer's 16-head partial before ``lse_combine2(heads=16, has_sink=True)``.
    ``active_splits`` is clamped to ``[2, num_splits]`` at launch so base-2 partial LSE is always
    emitted. Empty rows (``cache_seqlens[r] == 0``) produce zero output and ``-inf`` LSE.
    """
    if g.heads != 32 or g.window_size is not None:
        raise ValueError("Hy4 DSA partial requires 32-head full-context DSA geometry")
    if head_begin not in (0, 16) or head_count != 16:
        raise ValueError("Hy4 DSA partial requires head_begin in (0, 16) and head_count == 16")
    if num_splits != 16:
        raise ValueError("Hy4 DSA partial requires num_splits == 16")
    if not 1 <= int(max_rows) <= 4096 or not 1 <= int(row_chunk_size) <= 4096:
        raise ValueError("Hy4 DSA partial requires 1 <= max_rows, row_chunk_size <= 4096")
    launch = _Hy4DsaPartialChunks(
        g,
        max_rows=int(max_rows),
        head_begin=int(head_begin),
        head_count=int(head_count),
        num_splits=int(num_splits),
        row_chunk_size=int(row_chunk_size),
    )
    operands = (
        Operand("query", torch.bfloat16, "[rows,32,576]"),
        Operand(
            "kv_cache",
            torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
            "[pool_tokens,656]" if g.kv_cache == "fp8" else "[pool_tokens,576]",
        ),
        Operand("page_table", torch.int32, "[rows,page_table_width]", align=4),
        Operand("cache_seqlens", torch.int32, "[rows]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[rows+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{head_count},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{head_count}]", "out", align=4),
        Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch,
        name=f"hy4_mla_dsa_partial_{g.kv_cache}_h{head_begin}",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
        key=(
            g,
            int(max_rows),
            int(head_begin),
            int(head_count),
            int(num_splits),
            launch.row_chunk_size,
            launch.smem.query_tile,
            launch.smem.kv_stages,
            launch.smem.heads_per_tile,
            launch.smem.candidates_per_chunk,
            launch.smem.math_warps_per_query,
            "dsa_partial_base2",
        ),
        geometry={
            **vars(g),
            "head_begin": int(head_begin),
            "head_count": int(head_count),
            "heads_per_tile": launch.smem.heads_per_tile,
            "num_splits": int(num_splits),
            "chunks_per_split": launch.chunks_per_split,
            "max_rows": int(max_rows),
            "topk": 2048,
            "page_size": 1,
            "row_chunk_size": launch.row_chunk_size,
            "lse_base": 2,
        },
        scratch={"scratch": launch.scratch_bytes},
        doc=compile_hy4_mla_dsa_partial_aot.__doc__,
    )


def compile_hy4_mla_attention_aot(g: Hy4MlaGeometry, *, max_rows: int = 64, max_context: int = 2048,
                                  mode: str = "decode", page_size: int | None = None):
    ps = (64 if g.window_size is not None else 1) if page_size is None else int(page_size)
    if ps == 1 and mode != "decode":
        raise ValueError("DSA page_size=1 selected-token gather uses mode='decode' (singleton requests)")
    launch = _Hy4Attention(g, int(max_rows), int(max_context), mode, page_size=ps)
    tag = f"swa{g.window_size}" if g.window_size is not None else f"dsa_p{ps}"
    operands = (
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
        Operand(
            "kv_cache",
            torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
            "[pool_tokens,656]" if g.kv_cache == "fp8" else "[pool_tokens,576]",
        ),
        Operand("page_table", torch.int32, "[batch,page_table_width]", align=4),
        Operand("cache_seqlens", torch.int32, "[batch]", align=4),
        Operand("cu_seqlens_q", torch.int32, "[batch+1]", align=4),
        Operand("attn", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
        Operand("lse", torch.float32, f"[rows,{g.heads}]", "out", align=4),
        Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch,
        name=f"hy4_mla_{tag}_{mode}_h{g.heads}_{g.kv_cache}",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("batch"), Scalar("page_table_width", "int64"), Scalar("active_splits")),
        key=(g, int(max_rows), int(max_context), mode, ps),
        geometry={**vars(g), **launch.layout, "mode": mode, "page_size": ps, "max_rows": int(max_rows),
                  "max_context": int(max_context), "lse_base": "e"},
        scratch={"scratch": lambda rows: hy4_attention_layout(g, rows, max_context=max_context, mode=mode)["scratch_bytes"]},
        doc=__doc__,
    )
