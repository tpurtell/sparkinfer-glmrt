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
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    cp_async_bulk_g2s_mbar,
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_i32,
    ld_shared_u32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m16n16x1_trans_b8,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v2_u32,
    st_shared_f32,
    st_shared_i32,
    st_shared_u32,
    st_shared_v4_f32,
    st_shared_v4_u32,
    store_bf16x2,
)
from b12x._lib.utils import runtime_num_sm
from b12x.attention.dense_mla._forward import DenseMlaForwardKernel, _exit_thread
from b12x.attention.dense_mla._layout import choose_layout, make_smem_layout
from b12x.attention.dense_mla._math import exp2_approx, log2_approx
from b12x.attention.dense_mla._merge import DenseMlaMergeKernel
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo, _ld_stream
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import BatchedBf16Gemm, _bf16, _block_sum, _fp8_scale, _reduction_storage, _rsqrt, _warp_max, _warp_sum
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


_LN2 = 0.6931471805599453
_LOG2E = 1.4426950408889634
_INV_TWO_PI = 0.15915494309189535
_TWO_PI_HI = 6.25
_TWO_PI_MID = 0.032958984375
_TWO_PI_LO = 0.00022632280888501555
_FP8_RECORD_BYTES = 656


@dsl_user_op
def _rint_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "cvt.rni.f32.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cos_approx_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "cos.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sin_approx_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "sin.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _fast_cos_sin(angle: Float32):
    """3-term Cody-Waite range reduction + hardware cos/sin.approx (STACK: 0, <5e-7 err to 262K)."""
    k = _rint_f32(angle * Float32(_INV_TWO_PI))
    r = _fma_rn_f32(-k, Float32(_TWO_PI_HI), angle)
    r = _fma_rn_f32(-k, Float32(_TWO_PI_MID), r)
    r = _fma_rn_f32(-k, Float32(_TWO_PI_LO), r)
    return _cos_approx_f32(r), _sin_approx_f32(r)


@cute.jit
def _fast_sigmoid(x: Float32) -> Float32:
    """Hardware-exp2 sigmoid (STACK: 0, ~1e-7 accuracy)."""
    return Float32(1.0) / (Float32(1.0) + exp2_approx(-x * Float32(_LOG2E)))


@cute.jit
def _warp_sum_i32(v: Int32) -> Int32:
    for shift in cutlass.range_constexpr(5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << shift)
    return v


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
        if self.q_lora_rank not in (1024, 2048):
            raise ValueError(f"Hy4 MLA q_lora_rank must be 1024 or 2048; got {self.q_lora_rank}")
        if (self.kv_lora_rank, self.qk_nope_dim, self.qk_rope_dim,
                self.v_head_dim, self.page_rows) != (512, 192, 64, 256, 64):
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


def _is_tp1(g: Hy4MlaGeometry) -> bool:
    # Hy4-flash (q_lora_rank=1024): TP1 has 32 DSA heads and 64 SWA heads; keep TP2 routes unchanged.
    # Hy4 (q_lora_rank=2048): fast fused Q-absorb, sink-gate-UV, and FP8 MLA on both TP1 (64H) and TP2 (32H).
    if g.q_lora_rank == 2048:
        return g.heads in (32, 64)
    return g.heads == (32 if g.window_size is None else 64)


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
    """RMSNorm q_a (1024 or 2048) + kv_a (512), interleaved RoPE-64 on k_rope, write KV cache."""

    def __init__(self, g: Hy4MlaGeometry):
        self.g = g
        self.q_rank = int(g.q_lora_rank)
        self.q_iters = self.q_rank // 256

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

        q = cute.make_rmem_tensor((self.q_iters,), Float32)
        kv = cute.make_rmem_tensor((2,), Float32)
        qs, ks = Float32(0.0), Float32(0.0)
        for j in cutlass.range_constexpr(self.q_iters):
            d = Int64(j * 256) + Int64(tid)
            q[j] = Float32(q_a[row * Int64(self.q_rank) + d])
            qs += q[j] * q[j]
        for j in cutlass.range_constexpr(2):
            d = Int64(j * 256) + Int64(tid)
            kv[j] = Float32(kv_a[row * Int64(576) + d])
            ks += kv[j] * kv[j]

        qi = _rsqrt(_block_sum(qs, sums, Int32(0), 8) / Float32(float(self.q_rank)) + Float32(self.g.norm_eps))
        ki = _rsqrt(_block_sum(ks, sums, Int32(1), 8) / Float32(512.0) + Float32(self.g.norm_eps))

        for j in cutlass.range_constexpr(self.q_iters):
            d = Int64(j * 256) + Int64(tid)
            q_resid[row * Int64(self.q_rank) + d] = (q[j] * qi * Float32(q_norm[d])).to(BFloat16)

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


# ---------------------------------------------------------------------------
# Fused Q-absorb: q_nope @ W_UK^T + interleaved tail-64 RoPE in ONE launch
# ---------------------------------------------------------------------------


@cute.jit
def _apply_tail_rope_rows(
    q_raw_base: Int64,
    query_base: Int64,
    positions: cute.Pointer,
    inv_freq: cute.Pointer,
    head: Int64,
    heads: cutlass.Constexpr,
    row_start: Int32,
    row_end: Int32,
    warp_idx: Int32,
    lane: Int32,
    num_warps: cutlass.Constexpr,
    mrope_section: cutlass.Constexpr,
):
    """Rotate the query tail across warps using the shared three-axis position ABI."""
    freq = Float32(inv_freq[Int64(lane)])
    r = row_start + warp_idx
    while r < row_end:
        angle = _mrope_position(positions, Int64(r), Int64(lane), mrope_section[1], mrope_section[2]) * freq
        c, s = _fast_cos_sin(angle)
        h = Int64(r) * Int64(heads) + head
        src = q_raw_base + (h * Int64(256) + Int64(192) + Int64(lane) * Int64(2)) * Int64(2)
        dst = query_base + (h * Int64(576) + Int64(512) + Int64(lane) * Int64(2)) * Int64(2)
        packed = ld_global_nc_u32(src)
        x0 = _bf16_lo(packed)
        x1 = _bf16_hi(packed)
        st_global_u32(dst, pack_f32x2_to_bfloat2(x0 * c - x1 * s, x1 * c + x0 * s))
        r = r + Int32(num_warps)


class _Hy4QAbsorbGemvFused:
    """Coalesced 128B-cache-line streaming BF16 GEMV + STACK:0 fused RoPE for rows <= 4.

    128 threads (4 warps) = 16 output columns/CTA (8 lanes/col = 128B contiguous cache line,
    all 3 streaming weight loads issued up front). Grid ``(32, heads, 1)`` launches 1024/2048 CTAs.
    """

    def __init__(self, heads: int, mrope_section: tuple):
        self.heads = int(heads)
        self.mrope_section = mrope_section

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "q_s": cute.struct.Align[cute.struct.MemRange[Uint32, 4 * 96], 16],
        }
        return cute.struct(Storage)

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
        self.kernel(q_raw, w_uk, positions, inv_freq, query, rows).launch(
            grid=(32, self.heads, 1), block=(128, 1, 1), stream=stream
        )

    @cute.jit
    def _accum_rows(
        self,
        acc: cute.Tensor,
        w0: cute.Tensor,
        w1: cute.Tensor,
        w2: cute.Tensor,
        q_s_addr: Int32,
        k_lane: Int32,
        live: cutlass.Constexpr,
    ):
        lane_off = k_lane * Int32(16)
        for m in cutlass.range_constexpr(live):
            row_off = Int32(m * 384) + lane_off
            a = acc[m]
            q0 = ld_shared_v4_u32(q_s_addr + row_off)
            for j in cutlass.range_constexpr(4):
                a = a + w0[2 * j] * _bf16_lo(q0[j]) + w0[2 * j + 1] * _bf16_hi(q0[j])
            q1 = ld_shared_v4_u32(q_s_addr + row_off + Int32(128))
            for j in cutlass.range_constexpr(4):
                a = a + w1[2 * j] * _bf16_lo(q1[j]) + w1[2 * j + 1] * _bf16_hi(q1[j])
            q2 = ld_shared_v4_u32(q_s_addr + row_off + Int32(256))
            for j in cutlass.range_constexpr(4):
                a = a + w2[2 * j] * _bf16_lo(q2[j]) + w2[2 * j + 1] * _bf16_hi(q2[j])
            acc[m] = a

    @cute.kernel
    def kernel(
        self,
        q_raw: cute.Pointer,
        w_uk: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        query: cute.Pointer,
        rows: Int32,
    ):
        n_tile = Int32(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp_idx = tid >> Int32(5)
        lane = tid & Int32(31)
        sub_col = tid >> Int32(3)      # 0..15
        k_lane = tid & Int32(7)        # 0..7 (128B contiguous line)

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        q_s_addr = shared_ptr_to_u32(st.q_s.data_ptr())

        q_raw_base = Int64(q_raw.toint())
        query_base = Int64(query.toint())
        total_vecs = rows * Int32(24)
        v_idx = tid
        while v_idx < total_vecs:
            r = v_idx // Int32(24)
            c16 = v_idx - r * Int32(24)
            src = q_raw_base + ((Int64(r) * Int64(self.heads) + head) * Int64(256) * Int64(2)) + Int64(c16) * Int64(16)
            u0, u1, u2, u3 = ld_global_nc_v4_u32(src)
            st_shared_v4_u32(q_s_addr + r * Int32(384) + c16 * Int32(16), u0, u1, u2, u3)
            v_idx = v_idx + Int32(128)
        cute.arch.sync_threads()

        col = Int64(n_tile * Int32(16) + sub_col)
        w_addr = Int64(w_uk.toint()) + (head * Int64(512) + col) * Int64(384) + Int64(k_lane) * Int64(16)
        wv0 = _ld_stream(w_addr)
        wv1 = _ld_stream(w_addr + Int64(128))
        wv2 = _ld_stream(w_addr + Int64(256))
        w0 = cute.make_rmem_tensor((8,), Float32)
        w1 = cute.make_rmem_tensor((8,), Float32)
        w2 = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(4):
            w0[2 * j] = _bf16_lo(wv0[j])
            w0[2 * j + 1] = _bf16_hi(wv0[j])
            w1[2 * j] = _bf16_lo(wv1[j])
            w1[2 * j + 1] = _bf16_hi(wv1[j])
            w2[2 * j] = _bf16_lo(wv2[j])
            w2[2 * j + 1] = _bf16_hi(wv2[j])

        acc = cute.make_rmem_tensor((4,), Float32)
        for m in cutlass.range_constexpr(4):
            acc[m] = Float32(0.0)

        if rows == Int32(1):
            self._accum_rows(acc, w0, w1, w2, q_s_addr, k_lane, 1)
        elif rows == Int32(2):
            self._accum_rows(acc, w0, w1, w2, q_s_addr, k_lane, 2)
        elif rows == Int32(3):
            self._accum_rows(acc, w0, w1, w2, q_s_addr, k_lane, 3)
        else:
            self._accum_rows(acc, w0, w1, w2, q_s_addr, k_lane, 4)

        for m in cutlass.range_constexpr(4):
            if Int32(m) < rows:
                v = acc[m]
                v = v + cute.arch.shuffle_sync_bfly(v, offset=4)
                v = v + cute.arch.shuffle_sync_bfly(v, offset=2)
                v = v + cute.arch.shuffle_sync_bfly(v, offset=1)
                v_next = cute.arch.shuffle_sync(v, (lane & Int32(24)) ^ Int32(8))
                if k_lane == Int32(0) and (sub_col & Int32(1)) == Int32(0):
                    dst = query_base + ((Int64(m) * Int64(self.heads) + head) * Int64(576) + col) * Int64(2)
                    st_global_u32(dst, pack_f32x2_to_bfloat2(v, v_next))

        if n_tile == Int32(0):
            _apply_tail_rope_rows(
                q_raw_base, query_base, positions, inv_freq, head, self.heads,
                Int32(0), rows, warp_idx, lane, 4, self.mrope_section,
            )


class _Hy4QAbsorbCoalescedMma:
    """100%-coalesced 128B-cache-line Tensor-Core BF16 GEMM + STACK:0 RoPE for rows 5..64.

    128 threads (4 warps) own 32 output columns (1024 CTAs for H=64, 512 for H=32).
    All 6 contiguous 128B-line weight loads are issued back-to-back into padded SMEM
    (STS.128, zero bank conflicts on ldmatrix_m8n8x2_b16), then reused across all M-tiles.
    """

    def __init__(self, heads: int, num_m_tiles: int, mrope_section: tuple):
        self.heads = int(heads)
        self.mrope_section = mrope_section
        self.num_m_tiles = int(num_m_tiles)
        self.tile_m = 16 * self.num_m_tiles
        self.warps_per_cta = 4
        self.num_threads = 128
        self.cols_per_cta = 32
        self.n_tiles = 512 // self.cols_per_cta  # 16 -> 1024 CTAs (H=64), 512 (H=32)
        self.row_stride_bytes = 400              # 192 * 2 + 16 pad (100 % 32 == 4 banks -> 0 conflicts)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "sA": cute.struct.Align[cute.struct.MemRange[Uint8, self.tile_m * self.row_stride_bytes], 128],
            "sB": cute.struct.Align[cute.struct.MemRange[Uint8, self.cols_per_cta * self.row_stride_bytes], 128],
        }
        return cute.struct(Storage)

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
        self.kernel(q_raw, w_uk, positions, inv_freq, query, rows).launch(
            grid=(self.n_tiles, self.heads, 1), block=(self.num_threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        q_raw: cute.Pointer,
        w_uk: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        query: cute.Pointer,
        rows: Int32,
    ):
        n_tile = Int32(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp_idx = tid >> Int32(5)
        lane = tid & Int32(31)
        gid = lane >> Int32(2)
        ltid = lane & Int32(3)

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        sA_addr = shared_ptr_to_u32(st.sA.data_ptr())
        sB_addr = shared_ptr_to_u32(st.sB.data_ptr())

        q_raw_base = Int64(q_raw.toint())
        query_base = Int64(query.toint())

        # Issue ALL 6 coalesced 128B-line weight loads (32 cols x 384 B) up front!
        c_loc = tid >> Int32(3)   # 0..15
        k_vec = tid & Int32(7)    # 0..7 -> 128B contiguous cache line
        col0 = Int64(n_tile * Int32(32) + c_loc)
        w_ptr0 = Int64(w_uk.toint()) + (head * Int64(512) + col0) * Int64(384) + Int64(k_vec) * Int64(16)
        w_ptr1 = w_ptr0 + Int64(16 * 384)
        b00 = _ld_stream(w_ptr0)
        b01 = _ld_stream(w_ptr0 + Int64(128))
        b02 = _ld_stream(w_ptr0 + Int64(256))
        b10 = _ld_stream(w_ptr1)
        b11 = _ld_stream(w_ptr1 + Int64(128))
        b12 = _ld_stream(w_ptr1 + Int64(256))

        # Stage q_raw[0..rows-1, head, 0:192] directly with STS.128 while weights are in flight.
        total_vecs = rows * Int32(24)
        v_idx = tid
        while v_idx < total_vecs:
            r = v_idx // Int32(24)
            c16 = v_idx - r * Int32(24)
            src = q_raw_base + ((Int64(r) * Int64(self.heads) + head) * Int64(256) * Int64(2)) + Int64(c16) * Int64(16)
            u0, u1, u2, u3 = ld_global_nc_v4_u32(src)
            st_shared_v4_u32(sA_addr + r * Int32(self.row_stride_bytes) + c16 * Int32(16), u0, u1, u2, u3)
            v_idx = v_idx + Int32(128)

        sb_dst0 = sB_addr + c_loc * Int32(self.row_stride_bytes) + k_vec * Int32(16)
        sb_dst1 = sb_dst0 + Int32(16 * self.row_stride_bytes)
        st_shared_v4_u32(sb_dst0, b00[0], b00[1], b00[2], b00[3])
        st_shared_v4_u32(sb_dst0 + Int32(128), b01[0], b01[1], b01[2], b01[3])
        st_shared_v4_u32(sb_dst0 + Int32(256), b02[0], b02[1], b02[2], b02[3])
        st_shared_v4_u32(sb_dst1, b10[0], b10[1], b10[2], b10[3])
        st_shared_v4_u32(sb_dst1 + Int32(128), b11[0], b11[1], b11[2], b11[3])
        st_shared_v4_u32(sb_dst1 + Int32(256), b12[0], b12[1], b12[2], b12[3])
        cute.arch.sync_threads()

        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col_byte = (lane >> Int32(4)) * Int32(16)
        b_row = warp_idx * Int32(8) + (lane & Int32(7))
        b_col_byte = ((lane >> Int32(3)) & Int32(1)) * Int32(16)
        sA_lane_base = sA_addr + a_row * Int32(self.row_stride_bytes) + a_col_byte
        sB_lane_base = sB_addr + b_row * Int32(self.row_stride_bytes) + b_col_byte

        acc = cute.make_rmem_tensor((self.num_m_tiles * 4,), Float32)
        for i in cutlass.range_constexpr(self.num_m_tiles * 4):
            acc[i] = Float32(0.0)

        for ks in cutlass.range_constexpr(12):
            b0, b1 = ldmatrix_m8n8x2_b16(sB_lane_base + Int32(ks * 32))
            for mt in cutlass.range_constexpr(self.num_m_tiles):
                bi: cutlass.Constexpr = mt * 4
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    sA_lane_base + Int32(mt * 16 * self.row_stride_bytes + ks * 32)
                )
                d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                    acc[bi + 0], acc[bi + 1], acc[bi + 2], acc[bi + 3],
                    a0, a1, a2, a3, b0, b1,
                )
                acc[bi + 0] = d0
                acc[bi + 1] = d1
                acc[bi + 2] = d2
                acc[bi + 3] = d3

        out_col = Int64(n_tile * Int32(32) + warp_idx * Int32(8) + ltid * Int32(2))
        for mt in cutlass.range_constexpr(self.num_m_tiles):
            bi: cutlass.Constexpr = mt * 4
            r0 = Int32(mt * 16) + gid
            r1 = r0 + Int32(8)
            if r0 < rows:
                dst0 = query_base + ((Int64(r0) * Int64(self.heads) + head) * Int64(576) + out_col) * Int64(2)
                st_global_u32(dst0, pack_f32x2_to_bfloat2(acc[bi + 0], acc[bi + 1]))
            if r1 < rows:
                dst1 = query_base + ((Int64(r1) * Int64(self.heads) + head) * Int64(576) + out_col) * Int64(2)
                st_global_u32(dst1, pack_f32x2_to_bfloat2(acc[bi + 2], acc[bi + 3]))

        if n_tile == Int32(0):
            _apply_tail_rope_rows(
                q_raw_base, query_base, positions, inv_freq, head, self.heads,
                Int32(0), rows, warp_idx, lane, 4, self.mrope_section,
            )


class _Hy4QAbsorbMmaFused:
    """Row-tiled TMA Tensor-Core BF16 batched GEMM + fused tail-64 RoPE on n_tile==0."""

    tile_k = 64
    buffer_align_bytes = 1024

    def __init__(self, *, heads: int, warp_m: int, warp_n: int, tile_n: int, mrope_section: tuple, num_stages: int = 3):
        self.heads = int(heads)
        self.mrope_section = mrope_section
        self.n = 512
        self.k = 192
        self.a_row = self.heads * 256
        self.a_batch = 256
        self.o_row = self.heads * 576
        self.o_batch = 576
        self.warp_m = int(warp_m)
        self.warp_n = int(warp_n)
        self.num_compute_warps = self.warp_m * self.warp_n
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        self.tile_m = 16 * self.warp_m
        self.tile_n = int(tile_n)
        self.num_stages = int(num_stages)
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = self.n // self.tile_n

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.warp_m, self.warp_n, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _smem_layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, self.tile_k),
            BFloat16,
        )
        s_a = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k, self.num_stages), order=(0, 1, 2))
        return s_a, s_b

    def _storage(self, s_a, s_b):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sA": cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(s_a)], self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(s_b)], self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

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
        a_t = cute.make_tensor(q_raw, cute.make_layout((rows, self.k, self.heads), stride=(self.a_row, 1, self.a_batch)))
        w_t = cute.make_tensor(w_uk, cute.make_layout((self.n, self.k, self.heads), stride=(self.k, 1, self.n * self.k)))
        o_t = cute.make_tensor(query, cute.make_layout((rows, self.n, self.heads), stride=(self.o_row, 1, self.o_batch)))
        s_a, s_b = self._smem_layouts()
        tiled_mma = self._tiled_mma()
        storage = self._storage(s_a, s_b)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1,
        )
        tma_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), w_t, cute.slice_(s_b, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1,
        )
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(
            tma_tensor_a, tma_tensor_b, q_raw, positions, inv_freq, o_t,
            tma_a, tma_b, s_a, s_b, tiled_mma, storage, rows,
        ).launch(grid=(grid_m, self.n_tiles, self.heads), block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        q_raw_ptr: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        output: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        s_a_layout: cute.ComposedLayout,
        s_b_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        tma_bytes = (self.tile_m + self.tile_n) * self.tile_k * BFloat16.width // 8
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None, None))
        g_b = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2), cute.group_modes(g_a, 0, 2))
        t_bs, t_bg = cpasync.tma_partition(tma_atom_b, 0, cta_layout, cute.group_modes(s_b, 0, 2), cute.group_modes(g_b, 0, 2))
        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            thr_mma = tiled_mma.get_slice(tidx)
            t_csa = thr_mma.partition_A(s_a)
            t_csb = thr_mma.partition_B(s_b)
            t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
            t_crb = thr_mma.make_fragment_B(t_csb[None, None, None, 0])
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc.fill(0.0)
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16),
                tiled_mma,
            ).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16),
                tiled_mma,
            ).get_slice(tidx)
            t_ssa = copy_a.partition_S(s_a)
            t_ssb = copy_b.partition_S(s_b)
            for _k in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.consumer_wait(consumer_state)
                shared_a = t_ssa[None, None, None, consumer_state.index]
                shared_b = t_ssb[None, None, None, consumer_state.index]
                target_a = copy_a.retile(t_cra)
                target_b = copy_b.retile(t_crb)
                cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                cute.copy(copy_b, shared_b[None, None, 0], target_b[None, None, 0])
                for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                    if kk < cute.size(shared_a.shape[2]) - 1:
                        cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                        cute.copy(copy_b, shared_b[None, None, kk + 1], target_b[None, None, kk + 1])
                    cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
                cute.arch.fence_view_async_shared()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            base = Int64(output.iterator.toint())
            for pair in cutlass.range_constexpr(cute.size(acc) // 2):
                coord = coordinates[2 * pair]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens:
                    at = (Int64(token) * Int64(self.o_row) + Int64(batch) * Int64(self.o_batch) + Int64(column)) * Int64(2)
                    st_global_u32(base + at, pack_f32x2_to_bfloat2(acc[2 * pair], acc[2 * pair + 1]))
            if Int32(n_tile) == Int32(0):
                row_start = Int32(m_tile) * Int32(self.tile_m)
                row_end = row_start + Int32(self.tile_m)
                if row_end > num_tokens:
                    row_end = num_tokens
                _apply_tail_rope_rows(
                    Int64(q_raw_ptr.toint()), base, positions, inv_freq, Int64(batch), self.heads,
                    row_start, row_end, Int32(warp_idx), Int32(lane), self.num_compute_warps, self.mrope_section,
                )
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a, t_ag[(None, m_tile, k_tile, batch)], t_as[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b, t_bg[(None, n_tile, k_tile, batch)], t_bs[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class _Hy4Tp1QAbsorb:
    def __init__(self, g: Hy4MlaGeometry):
        h = g.heads
        self.gemv = _Hy4QAbsorbGemvFused(h, g.mrope_section)
        self.mma16 = _Hy4QAbsorbCoalescedMma(heads=h, num_m_tiles=1, mrope_section=g.mrope_section)
        self.mma32 = _Hy4QAbsorbCoalescedMma(heads=h, num_m_tiles=2, mrope_section=g.mrope_section)
        self.mma64 = _Hy4QAbsorbCoalescedMma(heads=h, num_m_tiles=4, mrope_section=g.mrope_section)
        self.mma_prefill = _Hy4QAbsorbMmaFused(
            heads=h, warp_m=4, warp_n=1, tile_n=128, mrope_section=g.mrope_section,
        )

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
        if rows > Int32(0):
            if rows <= Int32(4):
                self.gemv(q_raw, w_uk, positions, inv_freq, query, rows, stream)
            elif rows <= Int32(16):
                self.mma16(q_raw, w_uk, positions, inv_freq, query, rows, stream)
            elif rows <= Int32(32):
                self.mma32(q_raw, w_uk, positions, inv_freq, query, rows, stream)
            elif rows <= Int32(64):
                self.mma64(q_raw, w_uk, positions, inv_freq, query, rows, stream)
            else:
                self.mma_prefill(q_raw, w_uk, positions, inv_freq, query, rows, stream)


# ---------------------------------------------------------------------------
# Fused Sink-Gate-UV: attn @ W_UV^T * sigmoid(lse - sink) * sigmoid(gate)
# ---------------------------------------------------------------------------


class _Hy4SinkGateUvGemvFused:
    """Coalesced 512B/warp streaming BF16 GEMV + STACK:0 fused sink & gate for rows <= 4.

    256 threads (8 warps) = 8 output columns/CTA (32 lanes/col = 512B contiguous = 4 full
    128B cache lines per load, both loads issued up front). Grid ``(32, heads, 1)`` = 1024/2048 CTAs.
    """

    def __init__(self, heads: int):
        self.heads = int(heads)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "a_s": cute.struct.Align[cute.struct.MemRange[Uint32, 4 * 256], 16],
            "sink_s": cute.struct.Align[cute.struct.MemRange[Float32, 4], 16],
            "out_s": cute.struct.Align[cute.struct.MemRange[Float32, 32], 16],
        }
        return cute.struct(Storage)

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
        self.kernel(attn, w_uv, gate, lse, sink, out, rows).launch(
            grid=(32, self.heads, 1), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def _accum_rows(
        self,
        acc: cute.Tensor,
        w0: cute.Tensor,
        w1: cute.Tensor,
        a_s_addr: Int32,
        lane: Int32,
        live: cutlass.Constexpr,
    ):
        lane_off = lane * Int32(16)
        for m in cutlass.range_constexpr(live):
            row_off = Int32(m * 1024) + lane_off
            a = acc[m]
            x0 = ld_shared_v4_u32(a_s_addr + row_off)
            for j in cutlass.range_constexpr(4):
                a = a + w0[2 * j] * _bf16_lo(x0[j]) + w0[2 * j + 1] * _bf16_hi(x0[j])
            x1 = ld_shared_v4_u32(a_s_addr + row_off + Int32(512))
            for j in cutlass.range_constexpr(4):
                a = a + w1[2 * j] * _bf16_lo(x1[j]) + w1[2 * j + 1] * _bf16_hi(x1[j])
            acc[m] = a

    @cute.kernel
    def kernel(
        self,
        attn: cute.Pointer,
        w_uv: cute.Pointer,
        gate: cute.Pointer,
        lse: cute.Pointer,
        sink: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
    ):
        n_tile = Int32(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp_idx = tid >> Int32(5)
        lane = tid & Int32(31)

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        a_s_addr = shared_ptr_to_u32(st.a_s.data_ptr())
        sink_s = st.sink_s.get_tensor(cute.make_layout((4,)))
        out_s = st.out_s.get_tensor(cute.make_layout((4, 8), stride=(8, 1)))

        attn_base = Int64(attn.toint())
        total_vecs = rows * Int32(64)
        v_idx = tid
        while v_idx < total_vecs:
            r = v_idx >> Int32(6)
            c16 = v_idx & Int32(63)
            src = attn_base + ((Int64(r) * Int64(self.heads) + head) * Int64(1024)) + Int64(c16) * Int64(16)
            u0, u1, u2, u3 = ld_global_nc_v4_u32(src)
            st_shared_v4_u32(a_s_addr + r * Int32(1024) + c16 * Int32(16), u0, u1, u2, u3)
            v_idx = v_idx + Int32(256)
        if tid < rows:
            rh = Int64(tid) * Int64(self.heads) + head
            sink_s[tid] = _fast_sigmoid(Float32(lse[rh]) - Float32(sink[head]))
        cute.arch.sync_threads()

        col = Int64(n_tile * Int32(8) + warp_idx)
        w_addr = Int64(w_uv.toint()) + (head * Int64(256) + col) * Int64(1024) + Int64(lane) * Int64(16)
        wv0 = _ld_stream(w_addr)
        wv1 = _ld_stream(w_addr + Int64(512))
        w0 = cute.make_rmem_tensor((8,), Float32)
        w1 = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(4):
            w0[2 * j] = _bf16_lo(wv0[j])
            w0[2 * j + 1] = _bf16_hi(wv0[j])
            w1[2 * j] = _bf16_lo(wv1[j])
            w1[2 * j + 1] = _bf16_hi(wv1[j])

        acc = cute.make_rmem_tensor((4,), Float32)
        for m in cutlass.range_constexpr(4):
            acc[m] = Float32(0.0)

        if rows == Int32(1):
            self._accum_rows(acc, w0, w1, a_s_addr, lane, 1)
        elif rows == Int32(2):
            self._accum_rows(acc, w0, w1, a_s_addr, lane, 2)
        elif rows == Int32(3):
            self._accum_rows(acc, w0, w1, a_s_addr, lane, 3)
        else:
            self._accum_rows(acc, w0, w1, a_s_addr, lane, 4)

        for m in cutlass.range_constexpr(4):
            if Int32(m) < rows:
                v = _warp_sum(acc[m])
                if lane == Int32(0):
                    out_s[m, warp_idx] = v
        cute.arch.sync_threads()

        if tid < rows * Int32(4):
            m = tid >> Int32(2)
            c2 = (tid & Int32(3)) * Int32(2)
            col0 = Int64(n_tile * Int32(8) + c2)
            byte_off = ((Int64(m) * Int64(self.heads) + head) * Int64(256) + col0) * Int64(2)
            g_packed = ld_global_nc_u32(Int64(gate.toint()) + byte_off)
            sc = sink_s[m]
            s0 = _fast_sigmoid(_bf16_lo(g_packed))
            s1 = _fast_sigmoid(_bf16_hi(g_packed))
            st_global_u32(
                Int64(out.toint()) + byte_off,
                pack_f32x2_to_bfloat2(out_s[m, c2] * sc * s0, out_s[m, c2 + Int32(1)] * sc * s1),
            )


class _Hy4SinkGateUvCoalescedMma:
    """100%-coalesced 256B-line software-pipelined TC BF16 GEMM + fused sink & gate for rows 5..64.

    128 threads (4 warps: 2 col-tiles x 2 K-halves) own 16 output columns -> 1024 CTAs (H=64),
    512 CTAs (H=32). Double-buffered sB[2] + software-pipelined 256B contiguous `_ld_stream` loads
    overlap DRAM weight fetches 100% with m16n8k16 Tensor Core math (13-26 KB SMEM -> 3-7 CTAs/SM).
    """

    k_tile = 128
    k_tiles = 4
    k_tile_vecs = 16       # 128 BF16 / 8 = 16 x 16B vectors per row
    row_stride_bytes = 272  # 128 * 2 + 16 pad (68 % 32 == 4 banks -> 0 conflicts on ldmatrix)
    sb_stage_bytes = 16 * 272  # 4,352 B

    def __init__(self, heads: int, num_m_tiles: int):
        self.heads = int(heads)
        self.num_m_tiles = int(num_m_tiles)
        self.tile_m = 16 * self.num_m_tiles
        self.num_threads = 128
        self.cols_per_cta = 16
        self.n_tiles = 256 // self.cols_per_cta  # 16 -> 1024 CTAs (H=64), 512 (H=32)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "sA": cute.struct.Align[cute.struct.MemRange[Uint8, self.tile_m * self.row_stride_bytes], 128],
            "sB": cute.struct.Align[cute.struct.MemRange[Uint8, 2 * self.sb_stage_bytes], 128],
            "sink_s": cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16],
        }
        return cute.struct(Storage)

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
        self.kernel(attn, w_uv, gate, lse, sink, out, rows).launch(
            grid=(self.n_tiles, self.heads, 1), block=(self.num_threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        attn: cute.Pointer,
        w_uv: cute.Pointer,
        gate: cute.Pointer,
        lse: cute.Pointer,
        sink: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
    ):
        n_tile = Int32(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp_idx = tid >> Int32(5)
        lane = tid & Int32(31)
        gid = lane >> Int32(2)
        ltid = lane & Int32(3)
        w_n = warp_idx & Int32(1)   # 0 -> cols 0..7, 1 -> cols 8..15
        w_k = warp_idx >> Int32(1)  # 0 -> k 0..63, 1 -> k 64..127

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        sA_addr = shared_ptr_to_u32(st.sA.data_ptr())
        sB_base = shared_ptr_to_u32(st.sB.data_ptr())
        sink_s = st.sink_s.get_tensor(cute.make_layout((self.tile_m,)))

        if tid < rows:
            rh = Int64(tid) * Int64(self.heads) + head
            sink_s[tid] = _fast_sigmoid(Float32(lse[rh]) - Float32(sink[head]))

        attn_base = Int64(attn.toint())
        # 128 threads = 8 cols (c8) x 16 lanes (256B contiguous = 2 full 128B cache lines!).
        c8 = tid >> Int32(4)    # 0..7
        k16 = tid & Int32(15)   # 0..15
        col0 = Int64(n_tile * Int32(16) + c8)
        w_ptr0 = Int64(w_uv.toint()) + (head * Int64(256) + col0) * Int64(1024) + Int64(k16) * Int64(16)
        w_ptr1 = w_ptr0 + Int64(8 * 1024)
        sb_off0 = c8 * Int32(self.row_stride_bytes) + k16 * Int32(16)
        sb_off1 = sb_off0 + Int32(8 * self.row_stride_bytes)

        # Prime kt = 0: issue 256B coalesced weight loads + stage sA(kt=0).
        bw0 = _ld_stream(w_ptr0)
        bw1 = _ld_stream(w_ptr1)
        total_vecs = rows * Int32(self.k_tile_vecs)
        v_idx = tid
        while v_idx < total_vecs:
            r = v_idx >> Int32(4)
            c16 = v_idx & Int32(15)
            src = attn_base + ((Int64(r) * Int64(self.heads) + head) * Int64(1024)) + Int64(c16) * Int64(16)
            u0, u1, u2, u3 = ld_global_nc_v4_u32(src)
            st_shared_v4_u32(sA_addr + r * Int32(self.row_stride_bytes) + c16 * Int32(16), u0, u1, u2, u3)
            v_idx = v_idx + Int32(128)
        st_shared_v4_u32(sB_base + sb_off0, bw0[0], bw0[1], bw0[2], bw0[3])
        st_shared_v4_u32(sB_base + sb_off1, bw1[0], bw1[1], bw1[2], bw1[3])
        cute.arch.sync_threads()

        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col_byte = (lane >> Int32(4)) * Int32(16)
        b_row = w_n * Int32(8) + (lane & Int32(7))
        b_col_byte = ((lane >> Int32(3)) & Int32(1)) * Int32(16)
        wk_byte_off = w_k * Int32(128)
        sA_lane_base = sA_addr + a_row * Int32(self.row_stride_bytes) + wk_byte_off + a_col_byte
        sB_lane_off = b_row * Int32(self.row_stride_bytes) + wk_byte_off + b_col_byte

        acc = cute.make_rmem_tensor((self.num_m_tiles * 4,), Float32)
        for i in cutlass.range_constexpr(self.num_m_tiles * 4):
            acc[i] = Float32(0.0)

        for kt in cutlass.range_constexpr(4):
            cur_sb = sB_base + Int32((kt & 1) * self.sb_stage_bytes)
            # Prefetch next K-tile's 256B coalesced weights into registers while computing kt!
            if cutlass.const_expr(kt < 3):
                nxt_off: cutlass.Constexpr = (kt + 1) * 256
                bw0 = _ld_stream(w_ptr0 + Int64(nxt_off))
                bw1 = _ld_stream(w_ptr1 + Int64(nxt_off))

            for ks in cutlass.range_constexpr(4):
                b0, b1 = ldmatrix_m8n8x2_b16(cur_sb + sB_lane_off + Int32(ks * 32))
                for mt in cutlass.range_constexpr(self.num_m_tiles):
                    bi: cutlass.Constexpr = mt * 4
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        sA_lane_base + Int32(mt * 16 * self.row_stride_bytes + ks * 32)
                    )
                    d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                        acc[bi + 0], acc[bi + 1], acc[bi + 2], acc[bi + 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[bi + 0] = d0
                    acc[bi + 1] = d1
                    acc[bi + 2] = d2
                    acc[bi + 3] = d3

            if cutlass.const_expr(kt < 3):
                nxt_sb = sB_base + Int32(((kt + 1) & 1) * self.sb_stage_bytes)
                st_shared_v4_u32(nxt_sb + sb_off0, bw0[0], bw0[1], bw0[2], bw0[3])
                st_shared_v4_u32(nxt_sb + sb_off1, bw1[0], bw1[1], bw1[2], bw1[3])
                cute.arch.sync_threads()
                nxt_k_byte: cutlass.Constexpr = (kt + 1) * 256
                v_idx = tid
                while v_idx < total_vecs:
                    r = v_idx >> Int32(4)
                    c16 = v_idx & Int32(15)
                    src = (
                        attn_base
                        + ((Int64(r) * Int64(self.heads) + head) * Int64(1024))
                        + Int64(nxt_k_byte)
                        + Int64(c16) * Int64(16)
                    )
                    u0, u1, u2, u3 = ld_global_nc_v4_u32(src)
                    st_shared_v4_u32(sA_addr + r * Int32(self.row_stride_bytes) + c16 * Int32(16), u0, u1, u2, u3)
                    v_idx = v_idx + Int32(128)
                cute.arch.sync_threads()

        # Fold K-half w_k==1 into w_k==0 inside dead sB (4096 B max <= 8704 B) in full FP32.
        cute.arch.sync_threads()
        if w_k == Int32(1):
            for mt in cutlass.range_constexpr(self.num_m_tiles):
                bi: cutlass.Constexpr = mt * 4
                red_off = ((Int32(mt * 2) + w_n) * Int32(32) + lane) * Int32(16)
                st_shared_v4_f32(sB_base + red_off, acc[bi + 0], acc[bi + 1], acc[bi + 2], acc[bi + 3])
        cute.arch.sync_threads()

        if w_k == Int32(0):
            gate_base = Int64(gate.toint())
            out_base = Int64(out.toint())
            out_col = Int64(n_tile * Int32(16) + w_n * Int32(8) + ltid * Int32(2))
            for mt in cutlass.range_constexpr(self.num_m_tiles):
                bi: cutlass.Constexpr = mt * 4
                red_off = ((Int32(mt * 2) + w_n) * Int32(32) + lane) * Int32(16)
                p0, p1, p2, p3 = ld_shared_v4_f32(sB_base + red_off)
                r0 = Int32(mt * 16) + gid
                r1 = r0 + Int32(8)
                if r0 < rows:
                    at0 = ((Int64(r0) * Int64(self.heads) + head) * Int64(256) + out_col) * Int64(2)
                    g_p = ld_global_nc_u32(gate_base + at0)
                    sc = sink_s[r0]
                    s0 = _fast_sigmoid(_bf16_lo(g_p))
                    s1 = _fast_sigmoid(_bf16_hi(g_p))
                    st_global_u32(
                        out_base + at0,
                        pack_f32x2_to_bfloat2((acc[bi + 0] + p0) * sc * s0, (acc[bi + 1] + p1) * sc * s1),
                    )
                if r1 < rows:
                    at1 = ((Int64(r1) * Int64(self.heads) + head) * Int64(256) + out_col) * Int64(2)
                    g_p = ld_global_nc_u32(gate_base + at1)
                    sc = sink_s[r1]
                    s0 = _fast_sigmoid(_bf16_lo(g_p))
                    s1 = _fast_sigmoid(_bf16_hi(g_p))
                    st_global_u32(
                        out_base + at1,
                        pack_f32x2_to_bfloat2((acc[bi + 2] + p2) * sc * s0, (acc[bi + 3] + p3) * sc * s1),
                    )


class _Hy4SinkGateUvMmaFused:
    """Row-tiled TMA Tensor-Core BF16 batched GEMM + fused sink & gate epilogue."""

    tile_k = 64
    buffer_align_bytes = 1024

    def __init__(self, *, heads: int, warp_m: int, warp_n: int, tile_n: int, num_stages: int = 3):
        self.heads = int(heads)
        self.n = 256
        self.k = 512
        self.a_row = self.heads * 512
        self.a_batch = 512
        self.o_row = self.heads * 256
        self.o_batch = 256
        self.warp_m = int(warp_m)
        self.warp_n = int(warp_n)
        self.num_compute_warps = self.warp_m * self.warp_n
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        self.tile_m = 16 * self.warp_m
        self.tile_n = int(tile_n)
        self.num_stages = int(num_stages)
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = self.n // self.tile_n

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.warp_m, self.warp_n, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _smem_layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, self.tile_k),
            BFloat16,
        )
        s_a = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k, self.num_stages), order=(0, 1, 2))
        return s_a, s_b

    def _storage(self, s_a, s_b):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sink_s": cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m], 16],
            "sA": cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(s_a)], self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(s_b)], self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

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
        a_t = cute.make_tensor(attn, cute.make_layout((rows, self.k, self.heads), stride=(self.a_row, 1, self.a_batch)))
        w_t = cute.make_tensor(w_uv, cute.make_layout((self.n, self.k, self.heads), stride=(self.k, 1, self.n * self.k)))
        o_t = cute.make_tensor(out, cute.make_layout((rows, self.n, self.heads), stride=(self.o_row, 1, self.o_batch)))
        s_a, s_b = self._smem_layouts()
        tiled_mma = self._tiled_mma()
        storage = self._storage(s_a, s_b)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1,
        )
        tma_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), w_t, cute.slice_(s_b, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1,
        )
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(
            tma_tensor_a, tma_tensor_b, gate, lse, sink, o_t,
            tma_a, tma_b, s_a, s_b, tiled_mma, storage, rows,
        ).launch(grid=(grid_m, self.n_tiles, self.heads), block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        gate: cute.Pointer,
        lse: cute.Pointer,
        sink: cute.Pointer,
        output: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        s_a_layout: cute.ComposedLayout,
        s_b_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        sink_s = storage.sink_s.get_tensor(cute.make_layout((self.tile_m,)))
        tma_bytes = (self.tile_m + self.tile_n) * self.tile_k * BFloat16.width // 8
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None, None))
        g_b = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2), cute.group_modes(g_a, 0, 2))
        t_bs, t_bg = cpasync.tma_partition(tma_atom_b, 0, cta_layout, cute.group_modes(s_b, 0, 2), cute.group_modes(g_b, 0, 2))
        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            thr_mma = tiled_mma.get_slice(tidx)
            t_csa = thr_mma.partition_A(s_a)
            t_csb = thr_mma.partition_B(s_b)
            t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
            t_crb = thr_mma.make_fragment_B(t_csb[None, None, None, 0])
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc.fill(0.0)
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16),
                tiled_mma,
            ).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16),
                tiled_mma,
            ).get_slice(tidx)
            t_ssa = copy_a.partition_S(s_a)
            t_ssb = copy_b.partition_S(s_b)
            for _k in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.consumer_wait(consumer_state)
                shared_a = t_ssa[None, None, None, consumer_state.index]
                shared_b = t_ssb[None, None, None, consumer_state.index]
                target_a = copy_a.retile(t_cra)
                target_b = copy_b.retile(t_crb)
                cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                cute.copy(copy_b, shared_b[None, None, 0], target_b[None, None, 0])
                for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                    if kk < cute.size(shared_a.shape[2]) - 1:
                        cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                        cute.copy(copy_b, shared_b[None, None, kk + 1], target_b[None, None, kk + 1])
                    cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
                cute.arch.fence_view_async_shared()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            if Int32(tidx) < Int32(self.tile_m):
                tok_s = Int32(m_tile) * Int32(self.tile_m) + Int32(tidx)
                sc_val = Float32(0.0)
                if tok_s < num_tokens:
                    rh = Int64(tok_s) * Int64(self.heads) + Int64(batch)
                    lse_val = Float32(lse[rh])
                    sink_val = Float32(sink[Int64(batch)])
                    sc_val = _fast_sigmoid(lse_val - sink_val)
                sink_s[Int32(tidx)] = sc_val
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * self.num_compute_warps)

            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            out_base = Int64(output.iterator.toint())
            gate_base = Int64(gate.toint())
            for pair in cutlass.range_constexpr(cute.size(acc) // 2):
                coord = coordinates[2 * pair]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens:
                    at = (Int64(token) * Int64(self.o_row) + Int64(batch) * Int64(self.o_batch) + Int64(column)) * Int64(2)
                    g_packed = ld_global_nc_u32(gate_base + at)
                    g0 = _bf16_lo(g_packed)
                    g1 = _bf16_hi(g_packed)
                    sc = sink_s[coord[0]]
                    s0 = _fast_sigmoid(g0)
                    s1 = _fast_sigmoid(g1)
                    st_global_u32(
                        out_base + at,
                        pack_f32x2_to_bfloat2(acc[2 * pair] * sc * s0, acc[2 * pair + 1] * sc * s1),
                    )
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a, t_ag[(None, m_tile, k_tile, batch)], t_as[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b, t_bg[(None, n_tile, k_tile, batch)], t_bs[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class _Hy4Tp1SinkGateUv:
    def __init__(self, g: Hy4MlaGeometry):
        h = g.heads
        self.gemv = _Hy4SinkGateUvGemvFused(h)
        self.mma16 = _Hy4SinkGateUvCoalescedMma(heads=h, num_m_tiles=1)
        self.mma32 = _Hy4SinkGateUvCoalescedMma(heads=h, num_m_tiles=2)
        self.mma64 = _Hy4SinkGateUvCoalescedMma(heads=h, num_m_tiles=4)
        self.mma_prefill = _Hy4SinkGateUvMmaFused(heads=h, warp_m=4, warp_n=1, tile_n=64)

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
        if rows > Int32(0):
            if rows <= Int32(4):
                self.gemv(attn, w_uv, gate, lse, sink, out, rows, stream)
            elif rows <= Int32(16):
                self.mma16(attn, w_uv, gate, lse, sink, out, rows, stream)
            elif rows <= Int32(32):
                self.mma32(attn, w_uv, gate, lse, sink, out, rows, stream)
            elif rows <= Int32(64):
                self.mma64(attn, w_uv, gate, lse, sink, out, rows, stream)
            else:
                self.mma_prefill(attn, w_uv, gate, lse, sink, out, rows, stream)


# ---------------------------------------------------------------------------
# Fast double-buffered raw-FP8-DS MLA Attention for Hy4-flash (hpb=16 / hpb=32)
# ---------------------------------------------------------------------------


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


@cute.jit
def _zero_smem_528(dst_addr: Int32):
    """Zero the 512 E4M3 + 16 FP32-scale bytes of one masked KV record in SMEM."""
    z = Uint32(0)
    for i in cutlass.range_constexpr(33):
        st_shared_v4_u32(dst_addr + Int32(i * 16), z, z, z, z)


@cute.jit
def _resolve_row_bounds(
    query_row: Int32,
    batch: Int32,
    cache_seqlens: cute.Tensor,
    cu_seqlens_q: cute.Tensor,
    *,
    singleton_requests: cutlass.Constexpr,
    window_size: cutlass.Constexpr,
):
    request = Int32(0)
    query_begin = Int32(0)
    query_end = Int32(1)
    if cutlass.const_expr(singleton_requests):
        request = query_row
        query_begin = query_row
        query_end = query_row + Int32(1)
    else:
        lower = Int32(0)
        upper = batch
        while lower + Int32(1) < upper:
            middle = (lower + upper) // Int32(2)
            if query_row >= Int32(cu_seqlens_q[middle]):
                lower = middle
            else:
                upper = middle
        request = lower
        query_begin = Int32(cu_seqlens_q[request])
        query_end = Int32(cu_seqlens_q[request + Int32(1)])
    query_length = query_end - query_begin
    cache_length = Int32(cache_seqlens[request])
    local_query = query_row - query_begin
    visible_end = cache_length - query_length + local_query + Int32(1)
    if visible_end < Int32(0):
        visible_end = Int32(0)
    if visible_end > cache_length:
        visible_end = cache_length
    visible_begin = Int32(0)
    if cutlass.const_expr(window_size is not None):
        visible_begin = visible_end - Int32(window_size)
        if visible_begin < Int32(0):
            visible_begin = Int32(0)
    return request, visible_begin, visible_end


@cute.jit
def _io_gather_fp8_chunk(
    cache_base: Int64,
    page_table: cute.Tensor,
    row_table_offset: Int64,
    kv_dst_addr: Int32,
    tok_valid_addr: Int32,
    full_mbar_ptr,
    token_begin: Int32,
    visible_begin: Int32,
    visible_end: Int32,
    io_lane: Int32,
    *,
    bi: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
):
    """Gather one BI-token chunk of raw 656B FP8-DS records with cp.async.bulk.

    Invalid / masked tokens issue zero global traffic and have their 528B NoPE+scale
    SMEM bytes zeroed, eliminating any NaN risk from uninitialized or poisoned slots.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    if cutlass.const_expr(page_size == 64):
        # All BI (32 or 64) tokens in [token_begin, token_begin + BI) lie in one 64-token page.
        valid_lo = visible_begin - token_begin
        if valid_lo < Int32(0):
            valid_lo = Int32(0)
        if valid_lo > Int32(bi):
            valid_lo = Int32(bi)
        valid_hi = visible_end - token_begin
        if valid_hi < Int32(0):
            valid_hi = Int32(0)
        if valid_hi > Int32(bi):
            valid_hi = Int32(bi)
        valid_cnt = valid_hi - valid_lo
        if valid_cnt < Int32(0):
            valid_cnt = Int32(0)

        for rep in cutlass.range_constexpr(bi // 32):
            e = Int32(rep * 32) + io_lane
            is_v = Int32(1) if (e >= valid_lo and e < valid_hi) else Int32(0)
            st_shared_i32(tok_valid_addr + e * Int32(4), is_v)
            if is_v == Int32(0):
                _zero_smem_528(kv_dst_addr + e * Int32(_FP8_RECORD_BYTES))

        cute.arch.sync_warp()
        cute.arch.fence_acq_rel_cta()
        if io_lane == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_mbar_ptr, valid_cnt * Int32(_FP8_RECORD_BYTES)
            )
            if valid_cnt > Int32(0):
                logical_page = token_begin // Int32(64)
                in_page = token_begin - logical_page * Int32(64)
                phys_page = Int64(Int32(page_table[row_table_offset + Int64(logical_page)]))
                page_src = (
                    cache_base
                    + phys_page * Int64(64 * _FP8_RECORD_BYTES)
                    + Int64(in_page + valid_lo) * Int64(_FP8_RECORD_BYTES)
                )
                cp_async_bulk_g2s_mbar(
                    kv_dst_addr + valid_lo * Int32(_FP8_RECORD_BYTES),
                    page_src,
                    valid_cnt * Int32(_FP8_RECORD_BYTES),
                    full_mbar_addr,
                )
    else:
        # DSA page_size=1: per-token selected physical slot gather.
        slots_r = cute.make_rmem_tensor((bi // 32,), Int32)
        local_valid = Int32(0)
        for rep in cutlass.range_constexpr(bi // 32):
            e = Int32(rep * 32) + io_lane
            tok = token_begin + e
            slot = Int32(-1)
            if tok >= visible_begin and tok < visible_end:
                slot = Int32(page_table[row_table_offset + Int64(tok)])
            slots_r[rep] = slot
            is_v = Int32(1) if slot >= Int32(0) else Int32(0)
            st_shared_i32(tok_valid_addr + e * Int32(4), is_v)
            if is_v != Int32(0):
                local_valid = local_valid + Int32(1)
            else:
                _zero_smem_528(kv_dst_addr + e * Int32(_FP8_RECORD_BYTES))

        total_valid = _warp_sum_i32(local_valid)
        cute.arch.sync_warp()
        cute.arch.fence_acq_rel_cta()
        if io_lane == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_mbar_ptr, total_valid * Int32(_FP8_RECORD_BYTES)
            )
        for rep in cutlass.range_constexpr(bi // 32):
            slot = slots_r[rep]
            if slot >= Int32(0):
                e = Int32(rep * 32) + io_lane
                src = cache_base + Int64(slot) * Int64(_FP8_RECORD_BYTES)
                cp_async_bulk_g2s_mbar(
                    kv_dst_addr + e * Int32(_FP8_RECORD_BYTES),
                    src,
                    Int32(_FP8_RECORD_BYTES),
                    full_mbar_addr,
                )


class _Hy4Fp8MlaForward:
    """Double-buffered raw-656B FP8-DS MLA kernel with permuted-Q v4 K loads and sm_p[4] native-FP8 PV.

    Supports two compile-time specializations (both bi=32, 8 math + 1 IO warp = 288 threads):
    - ``hpb=16, bi=32`` (69.4 KB SMEM, 2-way K-split 18-vs-18 QK, sm_p[4] native fp8x4 PV)
    - ``hpb=32, bi=32`` (91.4 KB SMEM, 2 head groups/CTA, sm_p[4] native fp8x4 PV)
    """

    def __init__(
        self,
        *,
        num_heads: int,
        hpb: int,
        bi: int,
        page_size: int,
        window_size: int | None,
        max_splits: int,
    ):
        self.num_heads = int(num_heads)
        self.hpb = int(hpb)
        self.bi = int(bi)
        self.head_tiles = self.num_heads // self.hpb
        self.page_size = int(page_size)
        self.window_size = None if window_size is None else int(window_size)
        self.singleton_requests = (self.page_size == 1)
        self.max_splits = int(max_splits)
        self.n_hg = self.hpb // 16  # 1 or 2
        self.q_nope_smem_stride = 520  # 512 + 8 pad (1040 B -> 260 % 32 == 4 banks, 0 conflicts)
        self.q_rope_stride = 72
        self.sm_p_stride = self.bi + 8  # 40 (bi=32)
        self.sm_p_group_elems = self.hpb * self.sm_p_stride
        self.sm_p_group_bytes = self.sm_p_group_elems * 2

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "kv": cute.struct.Align[cute.struct.MemRange[Uint8, 2 * self.bi * _FP8_RECORD_BYTES], 128],
            "q_nope": cute.struct.Align[cute.struct.MemRange[BFloat16, self.hpb * self.q_nope_smem_stride], 128],
            "q_rope": cute.struct.Align[cute.struct.MemRange[BFloat16, self.hpb * self.q_rope_stride], 128],
            "sm_p": cute.struct.Align[cute.struct.MemRange[BFloat16, 4 * self.sm_p_group_elems], 128],
            "qk_part": cute.struct.Align[cute.struct.MemRange[Float32, 4 * 32 * 4 if self.n_hg == 1 else 4], 16],
            "reduce": cute.struct.Align[cute.struct.MemRange[Float32, 8 * 8 * 4], 16],
            "tok_valid": cute.struct.Align[cute.struct.MemRange[Int32, 2 * self.bi], 16],
            "mbar": cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        sm_scale_log2: Float32,
        page_table_width: Int64,
        rows: Int32,
        batch: Int32,
        eff_splits: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            query, kv_cache, page_table, cache_seqlens, cu_seqlens_q,
            output, final_lse, partial_output, partial_lse,
            sm_scale_log2, page_table_width, batch, eff_splits,
        ).launch(
            grid=(rows, self.head_tiles, eff_splits),
            block=(288, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _write_lse_empty(
        self,
        query_row: Int32,
        head_base: Int32,
        split: Int32,
        eff_splits: Int32,
        tid: Int32,
        final_lse: cute.Tensor,
        partial_lse: cute.Tensor,
    ):
        if tid < Int32(self.hpb):
            if cutlass.const_expr(self.max_splits > 1):
                if eff_splits > Int32(1):
                    partial_lse[query_row, head_base + tid, split] = Float32(-Float32.inf)
                else:
                    final_lse[query_row, head_base + tid] = Float32(-Float32.inf)
            else:
                final_lse[query_row, head_base + tid] = Float32(-Float32.inf)

    @cute.jit
    def _write_epilogue(
        self,
        acc,
        global_max,
        global_sum,
        out_o: cute.Tensor,
        out_lse: cute.Tensor,
        query_row: Int32,
        head_base: Int32,
        split: Int32,
        warp_id: Int32,
        lane: Int32,
        *,
        has_splits: cutlass.Constexpr,
    ):
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        for hg in cutlass.range_constexpr(self.n_hg):
            m0 = global_max[hg * 2 + 0]
            m1 = global_max[hg * 2 + 1]
            s0 = global_sum[hg * 2 + 0]
            s1 = global_sum[hg * 2 + 1]
            inv0 = Float32(1.0) / s0 if s0 > Float32(0.0) else Float32(0.0)
            inv1 = Float32(1.0) / s1 if s1 > Float32(0.0) else Float32(0.0)
            h0 = head_base + Int32(hg * 16) + gid
            h1 = h0 + Int32(8)
            for nt in cutlass.range_constexpr(8):
                d0 = warp_id * Int32(64) + Int32(nt * 8) + tid * Int32(2)
                base_idx: cutlass.Constexpr = (hg * 8 + nt) * 4
                if cutlass.const_expr(has_splits):
                    off0 = cute.crd2idx((query_row, h0, split, d0), out_o.layout)
                    off1 = cute.crd2idx((query_row, h1, split, d0), out_o.layout)
                else:
                    off0 = cute.crd2idx((query_row, h0, d0), out_o.layout)
                    off1 = cute.crd2idx((query_row, h1, d0), out_o.layout)
                store_bf16x2(get_ptr_as_int64(out_o, off0), acc[base_idx + 0] * inv0, acc[base_idx + 1] * inv0)
                store_bf16x2(get_ptr_as_int64(out_o, off1), acc[base_idx + 2] * inv1, acc[base_idx + 3] * inv1)
            if warp_id == Int32(0) and tid == Int32(0):
                lse0 = (log2_approx(s0) + m0) if s0 > Float32(0.0) else Float32(-Float32.inf)
                lse1 = (log2_approx(s1) + m1) if s1 > Float32(0.0) else Float32(-Float32.inf)
                if cutlass.const_expr(not has_splits):
                    if s0 > Float32(0.0):
                        lse0 = lse0 * Float32(_LN2)
                    if s1 > Float32(0.0):
                        lse1 = lse1 * Float32(_LN2)
                    out_lse[query_row, h0] = lse0
                    out_lse[query_row, h1] = lse1
                else:
                    out_lse[query_row, h0, split] = lse0
                    out_lse[query_row, h1, split] = lse1

    @cute.kernel
    def kernel(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        sm_scale_log2: Float32,
        page_table_width: Int64,
        batch: Int32,
        eff_splits: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        query_row = Int32(cute.arch.block_idx()[0])
        head_tile = Int32(cute.arch.block_idx()[1])
        split = Int32(cute.arch.block_idx()[2])
        head_base = head_tile * Int32(self.hpb)

        request, visible_begin, visible_end = _resolve_row_bounds(
            query_row, batch, cache_seqlens, cu_seqlens_q,
            singleton_requests=self.singleton_requests, window_size=self.window_size,
        )
        first_valid_chunk = visible_begin // Int32(self.bi)
        valid_chunks = (visible_end + Int32(self.bi - 1)) // Int32(self.bi)
        total_chunks = valid_chunks - first_valid_chunk
        if total_chunks < Int32(0):
            total_chunks = Int32(0)

        chunks_per_split = (total_chunks + eff_splits - Int32(1)) // eff_splits
        split_first_chunk = first_valid_chunk + split * chunks_per_split
        split_last_chunk = split_first_chunk + chunks_per_split
        if split_last_chunk > valid_chunks:
            split_last_chunk = valid_chunks
        active_chunks = split_last_chunk - split_first_chunk
        if active_chunks < Int32(0):
            active_chunks = Int32(0)

        if active_chunks == Int32(0):
            self._write_lse_empty(query_row, head_base, split, eff_splits, tid, final_lse, partial_lse)
            _exit_thread()

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        kv_base_addr = shared_ptr_to_u32(st.kv.data_ptr())
        q_nope_addr = shared_ptr_to_u32(st.q_nope.data_ptr())
        q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
        sm_p_addr = shared_ptr_to_u32(st.sm_p.data_ptr())
        qk_part_addr = shared_ptr_to_u32(st.qk_part.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        tok_valid_base_addr = shared_ptr_to_u32(st.tok_valid.data_ptr())
        mbar_base = st.mbar.data_ptr()

        if tid == Int32(0):
            cute.arch.mbarrier_init(mbar_base + Int32(0), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(1), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(2), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(3), Int32(1))
        cute.arch.barrier()

        kv_stage_bytes = Int32(self.bi * _FP8_RECORD_BYTES)
        tok_stage_bytes = Int32(self.bi * 4)

        if warp_id == Int32(8):
            cache_base = Int64(kv_cache.toint())
            row_table_offset = Int64(request) * page_table_width
            prod_idx = Int32(0)
            prod_phase = Int32(1)
            for local_chunk in cutlass.range(active_chunks, unroll=1):
                chunk = split_first_chunk + Int32(local_chunk)
                token_begin = chunk * Int32(self.bi)
                cute.arch.mbarrier_wait(mbar_base + Int32(2) + prod_idx, phase=prod_phase)
                _io_gather_fp8_chunk(
                    cache_base, page_table, row_table_offset,
                    kv_base_addr + prod_idx * kv_stage_bytes,
                    tok_valid_base_addr + prod_idx * tok_stage_bytes,
                    mbar_base + prod_idx,
                    token_begin, visible_begin, visible_end, lane,
                    bi=self.bi, page_size=self.page_size,
                )
                prod_idx = prod_idx ^ Int32(1)
                if prod_idx == Int32(0):
                    prod_phase = prod_phase ^ Int32(1)
        else:
            gid = lane >> Int32(2)
            ltid = lane & Int32(3)
            q_gmem_base = Int64(query.toint()) + (Int64(query_row) * Int64(self.num_heads) + Int64(head_base)) * Int64(576 * 2)

            # S0: Stage permuted Q-NoPE (matching v4_u32 K order within each 128-dim block) + Q-RoPE.
            total_nope_vecs: cutlass.Constexpr = self.hpb * 64
            idx_q = tid
            while idx_q < Int32(total_nope_vecs):
                h_loc = idx_q >> Int32(6)
                v_loc = idx_q & Int32(63)
                blk = v_loc >> Int32(4)
                in_blk = v_loc & Int32(15)
                l = in_blk >> Int32(2)
                seg = in_blk & Int32(3)
                src_q = q_gmem_base + Int64(h_loc) * Int64(576 * 2) + Int64(v_loc * 16)
                u0, u1, u2, u3 = ld_global_nc_v4_u32(src_q)
                tile0 = q_nope_addr + h_loc * Int32(self.q_nope_smem_stride * 2) + (blk * Int32(8) + seg * Int32(2)) * Int32(32)
                st_shared_u32(tile0 + l * Int32(4), u0)
                st_shared_u32(tile0 + Int32(16) + l * Int32(4), u1)
                st_shared_u32(tile0 + Int32(32) + l * Int32(4), u2)
                st_shared_u32(tile0 + Int32(48) + l * Int32(4), u3)
                idx_q = idx_q + Int32(256)

            total_rope_vecs: cutlass.Constexpr = self.hpb * 8
            idx_r = tid
            while idx_r < Int32(total_rope_vecs):
                h_loc = idx_r >> Int32(3)
                v_loc = idx_r & Int32(7)
                src_r = q_gmem_base + Int64(h_loc) * Int64(576 * 2) + Int64(512 * 2 + v_loc * 16)
                dst_r = q_rope_addr + h_loc * Int32(self.q_rope_stride * 2) + v_loc * Int32(16)
                u0, u1, u2, u3 = ld_global_nc_v4_u32(src_r)
                st_shared_v4_u32(dst_r, u0, u1, u2, u3)
                idx_r = idx_r + Int32(256)
            cute.arch.barrier(barrier_id=3, number_of_threads=256)

            acc_slots: cutlass.Constexpr = self.n_hg * 8 * 4
            acc_frag = cute.make_rmem_tensor((acc_slots,), Float32)
            gmax_frag = cute.make_rmem_tensor((self.n_hg * 2,), Float32)
            gsum_frag = cute.make_rmem_tensor((self.n_hg * 2,), Float32)
            for i in cutlass.range_constexpr(acc_slots):
                acc_frag[i] = Float32(0.0)
            for i in cutlass.range_constexpr(self.n_hg * 2):
                gmax_frag[i] = Float32(-1.0e30)
                gsum_frag[i] = Float32(0.0)

            a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            a_col_byte = (lane >> Int32(4)) * Int32(16)

            qk_group = warp_id >> Int32(2)       # hpb=16: K-half (0/1); hpb=32: head-group (0/1)
            qk_sub_warp = warp_id & Int32(3)     # 0..3 -> 32 candidates (8/warp)
            warp_first_cand = qk_sub_warp * Int32(8)
            q_nope_hg_addr = (
                q_nope_addr + qk_group * Int32(16 * self.q_nope_smem_stride * 2)
                if cutlass.const_expr(self.n_hg == 2) else q_nope_addr
            )
            q_rope_hg_addr = (
                q_rope_addr + qk_group * Int32(16 * self.q_rope_stride * 2)
                if cutlass.const_expr(self.n_hg == 2) else q_rope_addr
            )

            l16 = lane & Int32(15)
            v_perm_row = (l16 & Int32(1)) | ((l16 & Int32(12)) >> Int32(1)) | ((l16 & Int32(2)) << Int32(2))
            p_ld_off = ((lane & Int32(15)) * Int32(self.sm_p_stride) + (lane >> Int32(4)) * Int32(8)) * Int32(2)
            sm_p_g_addr = sm_p_addr + (warp_id >> Int32(1)) * Int32(self.sm_p_group_bytes)

            cons_idx = Int32(0)
            cons_phase = Int32(0)
            for _local_chunk in cutlass.range(active_chunks, unroll=1):
                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                kv_buf = kv_base_addr + cons_idx * kv_stage_bytes
                tok_buf = tok_valid_base_addr + cons_idx * tok_stage_bytes

                # S1 + S2: QK-NoPE (v4_u32 zero-bank-conflict K loads + fp8x4) + QK-RoPE.
                qk0 = Float32(0.0)
                qk1 = Float32(0.0)
                qk2 = Float32(0.0)
                qk3 = Float32(0.0)
                c0 = warp_first_cand + ltid * Int32(2)
                c1 = c0 + Int32(1)
                cand_row_addr = kv_buf + (warp_first_cand + gid) * Int32(_FP8_RECORD_BYTES)
                k_words = cute.make_rmem_tensor((8,), Uint32)

                if cutlass.const_expr(self.n_hg == 1):
                    # hpb=16, bi=32: 2-way K-split (warps 0..3: blk 0..1 + rope 0..1; warps 4..7: blk 2..3 + rope 2..3).
                    blk_start = qk_group * Int32(2)
                    for blk_i in cutlass.range_constexpr(2):
                        blk = blk_start + Int32(blk_i)
                        kb = cand_row_addr + blk * Int32(128) + ltid * Int32(32)
                        u0, u1, u2, u3 = ld_shared_v4_u32(kb)
                        u4, u5, u6, u7 = ld_shared_v4_u32(kb + Int32(16))
                        k_words[0] = u0
                        k_words[1] = u1
                        k_words[2] = u2
                        k_words[3] = u3
                        k_words[4] = u4
                        k_words[5] = u5
                        k_words[6] = u6
                        k_words[7] = u7
                        t0 = Float32(0.0)
                        t1 = Float32(0.0)
                        t2 = Float32(0.0)
                        t3 = Float32(0.0)
                        q_blk_base = q_nope_addr + a_row * Int32(self.q_nope_smem_stride * 2) + blk * Int32(256) + a_col_byte
                        for ks in cutlass.range_constexpr(8):
                            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(k_words[ks])
                            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_blk_base + Int32(ks * 32))
                            t0, t1, t2, t3 = mma_m16n8k16_f32_bf16(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1)
                        sc_off = Int32(512) + blk * Int32(4)
                        ks0 = ld_shared_f32(kv_buf + c0 * Int32(_FP8_RECORD_BYTES) + sc_off)
                        ks1 = ld_shared_f32(kv_buf + c1 * Int32(_FP8_RECORD_BYTES) + sc_off)
                        qk0 = qk0 + t0 * ks0
                        qk1 = qk1 + t1 * ks1
                        qk2 = qk2 + t2 * ks0
                        qk3 = qk3 + t3 * ks1

                    rope_row_addr = cand_row_addr + Int32(528)
                    rope_start = qk_group * Int32(2)
                    q_rope_base = q_rope_addr + a_row * Int32(self.q_rope_stride * 2) + a_col_byte
                    for rks_i in cutlass.range_constexpr(2):
                        rks = rope_start + Int32(rks_i)
                        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_rope_base + rks * Int32(32))
                        b0 = ld_shared_u32(rope_row_addr + rks * Int32(32) + ltid * Int32(4))
                        b1 = ld_shared_u32(rope_row_addr + rks * Int32(32) + Int32(16) + ltid * Int32(4))
                        qk0, qk1, qk2, qk3 = mma_m16n8k16_f32_bf16(qk0, qk1, qk2, qk3, a0, a1, a2, a3, b0, b1)

                    if qk_group == Int32(1):
                        st_shared_v4_f32(qk_part_addr + (qk_sub_warp * Int32(32) + lane) * Int32(16), qk0, qk1, qk2, qk3)
                    cute.arch.barrier(barrier_id=3, number_of_threads=256)

                    lmax0 = Float32(-1.0e30)
                    lmax1 = Float32(-1.0e30)
                    p0 = Float32(0.0)
                    p1 = Float32(0.0)
                    p2 = Float32(0.0)
                    p3 = Float32(0.0)
                    if qk_group == Int32(0):
                        pq0, pq1, pq2, pq3 = ld_shared_v4_f32(qk_part_addr + (qk_sub_warp * Int32(32) + lane) * Int32(16))
                        qk0 = qk0 + pq0
                        qk1 = qk1 + pq1
                        qk2 = qk2 + pq2
                        qk3 = qk3 + pq3
                        v_c0 = ld_shared_i32(tok_buf + c0 * Int32(4))
                        v_c1 = ld_shared_i32(tok_buf + c1 * Int32(4))
                        qk0 = (qk0 * sm_scale_log2) if v_c0 != Int32(0) else Float32(-1.0e30)
                        qk2 = (qk2 * sm_scale_log2) if v_c0 != Int32(0) else Float32(-1.0e30)
                        qk1 = (qk1 * sm_scale_log2) if v_c1 != Int32(0) else Float32(-1.0e30)
                        qk3 = (qk3 * sm_scale_log2) if v_c1 != Int32(0) else Float32(-1.0e30)

                        lmax0 = fmax_f32(qk0, qk1)
                        lmax1 = fmax_f32(qk2, qk3)
                        lmax0 = fmax_f32(lmax0, cute.arch.shuffle_sync_bfly(lmax0, offset=2))
                        lmax1 = fmax_f32(lmax1, cute.arch.shuffle_sync_bfly(lmax1, offset=2))
                        lmax0 = fmax_f32(lmax0, cute.arch.shuffle_sync_bfly(lmax0, offset=1))
                        lmax1 = fmax_f32(lmax1, cute.arch.shuffle_sync_bfly(lmax1, offset=1))

                        p0 = exp2_approx(qk0 - lmax0) if (v_c0 != Int32(0) and lmax0 > Float32(-1.0e29)) else Float32(0.0)
                        p1 = exp2_approx(qk1 - lmax0) if (v_c1 != Int32(0) and lmax0 > Float32(-1.0e29)) else Float32(0.0)
                        p2 = exp2_approx(qk2 - lmax1) if (v_c0 != Int32(0) and lmax1 > Float32(-1.0e29)) else Float32(0.0)
                        p3 = exp2_approx(qk3 - lmax1) if (v_c1 != Int32(0) and lmax1 > Float32(-1.0e29)) else Float32(0.0)
                        lsum0 = p0 + p1
                        lsum1 = p2 + p3
                        lsum0 = lsum0 + cute.arch.shuffle_sync_bfly(lsum0, offset=2)
                        lsum1 = lsum1 + cute.arch.shuffle_sync_bfly(lsum1, offset=2)
                        lsum0 = lsum0 + cute.arch.shuffle_sync_bfly(lsum0, offset=1)
                        lsum1 = lsum1 + cute.arch.shuffle_sync_bfly(lsum1, offset=1)
                        if ltid == Int32(0):
                            st_shared_v4_f32(
                                reduce_addr + (qk_sub_warp * Int32(8) + gid) * Int32(16),
                                lmax0, lsum0, lmax1, lsum1,
                            )
                    cute.arch.barrier(barrier_id=3, number_of_threads=256)

                    # S4: All 8 warps reduce 4 candidate-warps and rescale acc_frag.
                    wm0 = cute.make_rmem_tensor((4,), Float32)
                    ws0 = cute.make_rmem_tensor((4,), Float32)
                    wm1 = cute.make_rmem_tensor((4,), Float32)
                    ws1 = cute.make_rmem_tensor((4,), Float32)
                    bmax0 = Float32(-1.0e30)
                    bmax1 = Float32(-1.0e30)
                    for w_i in cutlass.range_constexpr(4):
                        m0v, s0v, m1v, s1v = ld_shared_v4_f32(reduce_addr + (Int32(w_i * 8) + gid) * Int32(16))
                        wm0[w_i] = m0v
                        ws0[w_i] = s0v
                        wm1[w_i] = m1v
                        ws1[w_i] = s1v
                        bmax0 = fmax_f32(bmax0, m0v)
                        bmax1 = fmax_f32(bmax1, m1v)
                    bsum0 = Float32(0.0)
                    bsum1 = Float32(0.0)
                    for w_i in cutlass.range_constexpr(4):
                        if wm0[w_i] > Float32(-1.0e29):
                            bsum0 = bsum0 + ws0[w_i] * exp2_approx(wm0[w_i] - bmax0)
                        if wm1[w_i] > Float32(-1.0e29):
                            bsum1 = bsum1 + ws1[w_i] * exp2_approx(wm1[w_i] - bmax1)
                    old_m0 = gmax_frag[0]
                    old_m1 = gmax_frag[1]
                    new_m0 = fmax_f32(old_m0, bmax0)
                    new_m1 = fmax_f32(old_m1, bmax1)
                    alpha0 = exp2_approx(old_m0 - new_m0) if old_m0 > Float32(-1.0e29) else Float32(0.0)
                    alpha1 = exp2_approx(old_m1 - new_m1) if old_m1 > Float32(-1.0e29) else Float32(0.0)
                    bscale0 = exp2_approx(bmax0 - new_m0) if bmax0 > Float32(-1.0e29) else Float32(0.0)
                    bscale1 = exp2_approx(bmax1 - new_m1) if bmax1 > Float32(-1.0e29) else Float32(0.0)
                    for nt in cutlass.range_constexpr(8):
                        bi_idx: cutlass.Constexpr = nt * 4
                        acc_frag[bi_idx + 0] = acc_frag[bi_idx + 0] * alpha0
                        acc_frag[bi_idx + 1] = acc_frag[bi_idx + 1] * alpha0
                        acc_frag[bi_idx + 2] = acc_frag[bi_idx + 2] * alpha1
                        acc_frag[bi_idx + 3] = acc_frag[bi_idx + 3] * alpha1
                    gsum_frag[0] = gsum_frag[0] * alpha0 + bsum0 * bscale0
                    gsum_frag[1] = gsum_frag[1] * alpha1 + bsum1 * bscale1
                    gmax_frag[0] = new_m0
                    gmax_frag[1] = new_m1

                    # S5: Warps 0..3 write scale-premultiplied BF16 probabilities into sm_p[4].
                    if qk_group == Int32(0):
                        wr0 = exp2_approx(lmax0 - new_m0) if lmax0 > Float32(-1.0e29) else Float32(0.0)
                        wr1 = exp2_approx(lmax1 - new_m1) if lmax1 > Float32(-1.0e29) else Float32(0.0)
                        p0w = p0 * wr0
                        p1w = p1 * wr0
                        p2w = p2 * wr1
                        p3w = p3 * wr1
                        sc0_0, sc0_1, sc0_2, sc0_3 = ld_shared_v4_f32(kv_buf + c0 * Int32(_FP8_RECORD_BYTES) + Int32(512))
                        sc1_0, sc1_1, sc1_2, sc1_3 = ld_shared_v4_f32(kv_buf + c1 * Int32(_FP8_RECORD_BYTES) + Int32(512))
                        p_off0 = (gid * Int32(self.sm_p_stride) + c0) * Int32(2)
                        p_off1 = ((gid + Int32(8)) * Int32(self.sm_p_stride) + c0) * Int32(2)
                        st_shared_u32(sm_p_addr + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_0, p1w * sc1_0))
                        st_shared_u32(sm_p_addr + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_0, p3w * sc1_0))
                        st_shared_u32(sm_p_addr + Int32(self.sm_p_group_bytes) + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_1, p1w * sc1_1))
                        st_shared_u32(sm_p_addr + Int32(self.sm_p_group_bytes) + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_1, p3w * sc1_1))
                        st_shared_u32(sm_p_addr + Int32(2 * self.sm_p_group_bytes) + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_2, p1w * sc1_2))
                        st_shared_u32(sm_p_addr + Int32(2 * self.sm_p_group_bytes) + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_2, p3w * sc1_2))
                        st_shared_u32(sm_p_addr + Int32(3 * self.sm_p_group_bytes) + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_3, p1w * sc1_3))
                        st_shared_u32(sm_p_addr + Int32(3 * self.sm_p_group_bytes) + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_3, p3w * sc1_3))
                    cute.arch.barrier(barrier_id=3, number_of_threads=256)
                else:
                    # hpb=32, bi=32: warps 0..3 -> hg=0, warps 4..7 -> hg=1.
                    q_nope_row_base = q_nope_hg_addr + a_row * Int32(self.q_nope_smem_stride * 2) + a_col_byte
                    for blk in cutlass.range_constexpr(4):
                        kb = cand_row_addr + Int32(blk * 128) + ltid * Int32(32)
                        u0, u1, u2, u3 = ld_shared_v4_u32(kb)
                        u4, u5, u6, u7 = ld_shared_v4_u32(kb + Int32(16))
                        k_words[0] = u0
                        k_words[1] = u1
                        k_words[2] = u2
                        k_words[3] = u3
                        k_words[4] = u4
                        k_words[5] = u5
                        k_words[6] = u6
                        k_words[7] = u7
                        t0 = Float32(0.0)
                        t1 = Float32(0.0)
                        t2 = Float32(0.0)
                        t3 = Float32(0.0)
                        for ks in cutlass.range_constexpr(8):
                            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(k_words[ks])
                            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_nope_row_base + Int32((blk * 8 + ks) * 32))
                            t0, t1, t2, t3 = mma_m16n8k16_f32_bf16(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1)
                        sc_off: cutlass.Constexpr = 512 + blk * 4
                        ks0 = ld_shared_f32(kv_buf + c0 * Int32(_FP8_RECORD_BYTES) + Int32(sc_off))
                        ks1 = ld_shared_f32(kv_buf + c1 * Int32(_FP8_RECORD_BYTES) + Int32(sc_off))
                        qk0 = qk0 + t0 * ks0
                        qk1 = qk1 + t1 * ks1
                        qk2 = qk2 + t2 * ks0
                        qk3 = qk3 + t3 * ks1

                    rope_row_addr = cand_row_addr + Int32(528)
                    q_rope_row_base = q_rope_hg_addr + a_row * Int32(self.q_rope_stride * 2) + a_col_byte
                    for ks in cutlass.range_constexpr(4):
                        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_rope_row_base + Int32(ks * 32))
                        b0 = ld_shared_u32(rope_row_addr + Int32(ks * 32) + ltid * Int32(4))
                        b1 = ld_shared_u32(rope_row_addr + Int32(ks * 32 + 16) + ltid * Int32(4))
                        qk0, qk1, qk2, qk3 = mma_m16n8k16_f32_bf16(qk0, qk1, qk2, qk3, a0, a1, a2, a3, b0, b1)

                    v_c0 = ld_shared_i32(tok_buf + c0 * Int32(4))
                    v_c1 = ld_shared_i32(tok_buf + c1 * Int32(4))
                    qk0 = (qk0 * sm_scale_log2) if v_c0 != Int32(0) else Float32(-1.0e30)
                    qk2 = (qk2 * sm_scale_log2) if v_c0 != Int32(0) else Float32(-1.0e30)
                    qk1 = (qk1 * sm_scale_log2) if v_c1 != Int32(0) else Float32(-1.0e30)
                    qk3 = (qk3 * sm_scale_log2) if v_c1 != Int32(0) else Float32(-1.0e30)

                    lmax0 = fmax_f32(qk0, qk1)
                    lmax1 = fmax_f32(qk2, qk3)
                    lmax0 = fmax_f32(lmax0, cute.arch.shuffle_sync_bfly(lmax0, offset=2))
                    lmax1 = fmax_f32(lmax1, cute.arch.shuffle_sync_bfly(lmax1, offset=2))
                    lmax0 = fmax_f32(lmax0, cute.arch.shuffle_sync_bfly(lmax0, offset=1))
                    lmax1 = fmax_f32(lmax1, cute.arch.shuffle_sync_bfly(lmax1, offset=1))

                    p0 = exp2_approx(qk0 - lmax0) if (v_c0 != Int32(0) and lmax0 > Float32(-1.0e29)) else Float32(0.0)
                    p1 = exp2_approx(qk1 - lmax0) if (v_c1 != Int32(0) and lmax0 > Float32(-1.0e29)) else Float32(0.0)
                    p2 = exp2_approx(qk2 - lmax1) if (v_c0 != Int32(0) and lmax1 > Float32(-1.0e29)) else Float32(0.0)
                    p3 = exp2_approx(qk3 - lmax1) if (v_c1 != Int32(0) and lmax1 > Float32(-1.0e29)) else Float32(0.0)
                    lsum0 = p0 + p1
                    lsum1 = p2 + p3
                    lsum0 = lsum0 + cute.arch.shuffle_sync_bfly(lsum0, offset=2)
                    lsum1 = lsum1 + cute.arch.shuffle_sync_bfly(lsum1, offset=2)
                    lsum0 = lsum0 + cute.arch.shuffle_sync_bfly(lsum0, offset=1)
                    lsum1 = lsum1 + cute.arch.shuffle_sync_bfly(lsum1, offset=1)

                    if ltid == Int32(0):
                        st_shared_v4_f32(
                            reduce_addr + (warp_id * Int32(8) + gid) * Int32(16),
                            lmax0, lsum0, lmax1, lsum1,
                        )
                    cute.arch.barrier(barrier_id=3, number_of_threads=256)

                    my_new_max0 = Float32(-1.0e30)
                    my_new_max1 = Float32(-1.0e30)
                    for hg in cutlass.range_constexpr(2):
                        wm0 = cute.make_rmem_tensor((4,), Float32)
                        ws0 = cute.make_rmem_tensor((4,), Float32)
                        wm1 = cute.make_rmem_tensor((4,), Float32)
                        ws1 = cute.make_rmem_tensor((4,), Float32)
                        bmax0 = Float32(-1.0e30)
                        bmax1 = Float32(-1.0e30)
                        for w_i in cutlass.range_constexpr(4):
                            w_slot: cutlass.Constexpr = hg * 4 + w_i
                            m0v, s0v, m1v, s1v = ld_shared_v4_f32(
                                reduce_addr + (Int32(w_slot * 8) + gid) * Int32(16)
                            )
                            wm0[w_i] = m0v
                            ws0[w_i] = s0v
                            wm1[w_i] = m1v
                            ws1[w_i] = s1v
                            bmax0 = fmax_f32(bmax0, m0v)
                            bmax1 = fmax_f32(bmax1, m1v)
                        bsum0 = Float32(0.0)
                        bsum1 = Float32(0.0)
                        for w_i in cutlass.range_constexpr(4):
                            if wm0[w_i] > Float32(-1.0e29):
                                bsum0 = bsum0 + ws0[w_i] * exp2_approx(wm0[w_i] - bmax0)
                            if wm1[w_i] > Float32(-1.0e29):
                                bsum1 = bsum1 + ws1[w_i] * exp2_approx(wm1[w_i] - bmax1)
                        old_m0 = gmax_frag[hg * 2 + 0]
                        old_m1 = gmax_frag[hg * 2 + 1]
                        new_m0 = fmax_f32(old_m0, bmax0)
                        new_m1 = fmax_f32(old_m1, bmax1)
                        alpha0 = exp2_approx(old_m0 - new_m0) if old_m0 > Float32(-1.0e29) else Float32(0.0)
                        alpha1 = exp2_approx(old_m1 - new_m1) if old_m1 > Float32(-1.0e29) else Float32(0.0)
                        bscale0 = exp2_approx(bmax0 - new_m0) if bmax0 > Float32(-1.0e29) else Float32(0.0)
                        bscale1 = exp2_approx(bmax1 - new_m1) if bmax1 > Float32(-1.0e29) else Float32(0.0)
                        for nt in cutlass.range_constexpr(8):
                            bi_idx: cutlass.Constexpr = (hg * 8 + nt) * 4
                            acc_frag[bi_idx + 0] = acc_frag[bi_idx + 0] * alpha0
                            acc_frag[bi_idx + 1] = acc_frag[bi_idx + 1] * alpha0
                            acc_frag[bi_idx + 2] = acc_frag[bi_idx + 2] * alpha1
                            acc_frag[bi_idx + 3] = acc_frag[bi_idx + 3] * alpha1
                        gsum_frag[hg * 2 + 0] = gsum_frag[hg * 2 + 0] * alpha0 + bsum0 * bscale0
                        gsum_frag[hg * 2 + 1] = gsum_frag[hg * 2 + 1] * alpha1 + bsum1 * bscale1
                        gmax_frag[hg * 2 + 0] = new_m0
                        gmax_frag[hg * 2 + 1] = new_m1
                        if qk_group == Int32(hg):
                            my_new_max0 = new_m0
                            my_new_max1 = new_m1

                    wr0 = exp2_approx(lmax0 - my_new_max0) if lmax0 > Float32(-1.0e29) else Float32(0.0)
                    wr1 = exp2_approx(lmax1 - my_new_max1) if lmax1 > Float32(-1.0e29) else Float32(0.0)
                    p0w = p0 * wr0
                    p1w = p1 * wr0
                    p2w = p2 * wr1
                    p3w = p3 * wr1
                    sc0_0, sc0_1, sc0_2, sc0_3 = ld_shared_v4_f32(kv_buf + c0 * Int32(_FP8_RECORD_BYTES) + Int32(512))
                    sc1_0, sc1_1, sc1_2, sc1_3 = ld_shared_v4_f32(kv_buf + c1 * Int32(_FP8_RECORD_BYTES) + Int32(512))
                    p_row0 = qk_group * Int32(16) + gid
                    p_off0 = (p_row0 * Int32(self.sm_p_stride) + c0) * Int32(2)
                    p_off1 = ((p_row0 + Int32(8)) * Int32(self.sm_p_stride) + c0) * Int32(2)
                    st_shared_u32(sm_p_addr + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_0, p1w * sc1_0))
                    st_shared_u32(sm_p_addr + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_0, p3w * sc1_0))
                    st_shared_u32(sm_p_addr + Int32(self.sm_p_group_bytes) + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_1, p1w * sc1_1))
                    st_shared_u32(sm_p_addr + Int32(self.sm_p_group_bytes) + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_1, p3w * sc1_1))
                    st_shared_u32(sm_p_addr + Int32(2 * self.sm_p_group_bytes) + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_2, p1w * sc1_2))
                    st_shared_u32(sm_p_addr + Int32(2 * self.sm_p_group_bytes) + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_2, p3w * sc1_2))
                    st_shared_u32(sm_p_addr + Int32(3 * self.sm_p_group_bytes) + p_off0, pack_f32x2_to_bfloat2(p0w * sc0_3, p1w * sc1_3))
                    st_shared_u32(sm_p_addr + Int32(3 * self.sm_p_group_bytes) + p_off1, pack_f32x2_to_bfloat2(p2w * sc0_3, p3w * sc1_3))
                    cute.arch.barrier(barrier_id=3, number_of_threads=256)

                # S6: PV via ldmatrix_m16n16x1_trans_b8 + 2-inst native fp8x4 widen + m16n8k16 BF16 MMA.
                vbase = kv_buf + v_perm_row * Int32(_FP8_RECORD_BYTES) + warp_id * Int32(64)
                pv_k_steps: cutlass.Constexpr = self.bi // 16
                for ks in cutlass.range_constexpr(pv_k_steps):
                    pa0, pa1, pa2, pa3 = ldmatrix_m8n8x4_b16(sm_p_g_addr + p_ld_off + Int32(ks * 32))
                    if cutlass.const_expr(self.n_hg == 2):
                        pb0, pb1, pb2, pb3 = ldmatrix_m8n8x4_b16(
                            sm_p_g_addr + Int32(16 * self.sm_p_stride * 2) + p_ld_off + Int32(ks * 32)
                        )
                    next_code = Uint32(0)
                    for nt in cutlass.range_constexpr(8):
                        code = Uint32(0)
                        if cutlass.const_expr(nt % 2 == 0):
                            code, next_code = ldmatrix_m16n16x1_trans_b8(
                                vbase + Int32(ks * 16 * _FP8_RECORD_BYTES + (nt // 2) * 16)
                            )
                        else:
                            code = next_code
                        vb0, vb1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(code)
                        bi0: cutlass.Constexpr = nt * 4
                        d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                            acc_frag[bi0 + 0], acc_frag[bi0 + 1], acc_frag[bi0 + 2], acc_frag[bi0 + 3],
                            pa0, pa1, pa2, pa3, vb0, vb1,
                        )
                        acc_frag[bi0 + 0] = d0
                        acc_frag[bi0 + 1] = d1
                        acc_frag[bi0 + 2] = d2
                        acc_frag[bi0 + 3] = d3
                        if cutlass.const_expr(self.n_hg == 2):
                            bi1: cutlass.Constexpr = (8 + nt) * 4
                            e0, e1, e2, e3 = mma_m16n8k16_f32_bf16(
                                acc_frag[bi1 + 0], acc_frag[bi1 + 1], acc_frag[bi1 + 2], acc_frag[bi1 + 3],
                                pb0, pb1, pb2, pb3, vb0, vb1,
                            )
                            acc_frag[bi1 + 0] = e0
                            acc_frag[bi1 + 1] = e1
                            acc_frag[bi1 + 2] = e2
                            acc_frag[bi1 + 3] = e3

                cute.arch.barrier(barrier_id=3, number_of_threads=256)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(2) + cons_idx)
                cons_idx = cons_idx ^ Int32(1)
                if cons_idx == Int32(0):
                    cons_phase = cons_phase ^ Int32(1)

            # S7: Epilogue.
            if cutlass.const_expr(self.max_splits > 1):
                if eff_splits > Int32(1):
                    self._write_epilogue(
                        acc_frag, gmax_frag, gsum_frag, partial_output, partial_lse,
                        query_row, head_base, split, warp_id, lane, has_splits=True,
                    )
                else:
                    self._write_epilogue(
                        acc_frag, gmax_frag, gsum_frag, output, final_lse,
                        query_row, head_base, split, warp_id, lane, has_splits=False,
                    )
            else:
                self._write_epilogue(
                    acc_frag, gmax_frag, gsum_frag, output, final_lse,
                    query_row, head_base, split, warp_id, lane, has_splits=False,
                )


@cute.jit
def _io_gather_fp8_bi32_no_tokbuf(
    cache_base: Int64,
    page_table: cute.Tensor,
    row_table_offset: Int64,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    token_begin: Int32,
    visible_begin: Int32,
    visible_end: Int32,
    io_lane: Int32,
):
    """Gather one 32-token half-page SWA chunk of raw 656B FP8-DS records (no tok_valid SMEM)."""
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    valid_lo = visible_begin - token_begin
    if valid_lo < Int32(0):
        valid_lo = Int32(0)
    if valid_lo > Int32(32):
        valid_lo = Int32(32)
    valid_hi = visible_end - token_begin
    if valid_hi < Int32(0):
        valid_hi = Int32(0)
    if valid_hi > Int32(32):
        valid_hi = Int32(32)
    valid_cnt = valid_hi - valid_lo
    if valid_cnt < Int32(0):
        valid_cnt = Int32(0)

    if io_lane < valid_lo or io_lane >= valid_hi:
        _zero_smem_528(kv_dst_addr + io_lane * Int32(_FP8_RECORD_BYTES))

    cute.arch.sync_warp()
    cute.arch.fence_acq_rel_cta()
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr, valid_cnt * Int32(_FP8_RECORD_BYTES)
        )
        if valid_cnt > Int32(0):
            logical_page = token_begin // Int32(64)
            in_page = token_begin - logical_page * Int32(64)
            phys_page = Int64(Int32(page_table[row_table_offset + Int64(logical_page)]))
            page_src = (
                cache_base
                + phys_page * Int64(64 * _FP8_RECORD_BYTES)
                + Int64(in_page + valid_lo) * Int64(_FP8_RECORD_BYTES)
            )
            cp_async_bulk_g2s_mbar(
                kv_dst_addr + valid_lo * Int32(_FP8_RECORD_BYTES),
                page_src,
                valid_cnt * Int32(_FP8_RECORD_BYTES),
                full_mbar_addr,
            )


class _Hy4Swa2WgForward:
    """True double-buffered 2-WG (16 math + 1 IO = 544 threads) raw-656B FP8-DS SWA-512 kernel.

    ``hpb=8, bi=32, 4 KV stages`` (99.9 KB SMEM): WG0 double-buffers ``kv[0, 2]`` (even chunks)
    and WG1 double-buffers ``kv[1, 3]`` (odd chunks) with 100% IO overlap, 2-way K-split 18-vs-18 QK,
    XOR-swizzled SMEM ``q_nope`` + ``q_rope`` (zero global LDGs in loop), dedicated ``reduce``,
    XOR-swizzled ``sm_p[2][4]`` native fp8x4 PV, and full-FP32 in-SMEM merge inside dead ``kv[0]``.
    """

    hpb = 8
    bi = 32
    sm_p_group_elems = 8 * 32
    sm_p_group_bytes = sm_p_group_elems * 2    # 512 B
    sm_p_wg_bytes = 4 * sm_p_group_bytes       # 2,048 B
    qk_part_wg_bytes = 4 * 32 * 2 * 4          # 1,024 B (4 cand-warps x 32 lanes x 2 floats)
    reduce_wg_bytes = 4 * 8 * 2 * 4            # 256 B (4 cand-warps x 8 heads x 2 floats)

    def __init__(self, *, num_heads: int, page_size: int, window_size: int | None):
        self.num_heads = int(num_heads)
        self.head_tiles = self.num_heads // self.hpb
        self.page_size = int(page_size)
        self.window_size = None if window_size is None else int(window_size)
        self.singleton_requests = (self.page_size == 1)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "kv": cute.struct.Align[cute.struct.MemRange[Uint8, 4 * self.bi * _FP8_RECORD_BYTES], 128],
            "q_nope": cute.struct.Align[cute.struct.MemRange[BFloat16, self.hpb * 512], 128],
            "q_rope": cute.struct.Align[cute.struct.MemRange[BFloat16, self.hpb * 64], 128],
            "sm_p": cute.struct.Align[cute.struct.MemRange[BFloat16, 2 * 4 * self.sm_p_group_elems], 128],
            "qk_part": cute.struct.Align[cute.struct.MemRange[Float32, 2 * 4 * 32 * 2], 16],
            "reduce": cute.struct.Align[cute.struct.MemRange[Float32, 2 * 4 * 8 * 2], 16],
            "mbar": cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 8], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        sm_scale_log2: Float32,
        page_table_width: Int64,
        rows: Int32,
        batch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            query, kv_cache, page_table, cache_seqlens, cu_seqlens_q,
            output, final_lse, sm_scale_log2, page_table_width, batch,
        ).launch(
            grid=(rows, self.head_tiles, 1),
            block=(544, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _run_wg_loop(
        self,
        wg_chunks: Int32,
        first_wg_chunk: Int32,
        visible_begin: Int32,
        visible_end: Int32,
        kv_base_addr: Int32,
        q_nope_addr: Int32,
        q_rope_addr: Int32,
        sm_p_wg_addr: Int32,
        qk_part_wg_addr: Int32,
        reduce_wg_addr: Int32,
        mbar_base,
        sm_scale_log2: Float32,
        w_in_wg: Int32,
        wg_tid: Int32,
        lane: Int32,
        gid: Int32,
        ltid: Int32,
        acc_frag,
        *,
        slot_bias: cutlass.Constexpr,
        barrier_id: cutlass.Constexpr,
    ):
        gmax0 = Float32(-1.0e30)
        gsum0 = Float32(0.0)
        qk_half = w_in_wg >> Int32(2)         # 0: blk 0..1 + rope 0..1; 1: blk 2..3 + rope 2..3
        qk_sub_warp = w_in_wg & Int32(3)      # 0..3 -> 32 candidates (8/warp)
        warp_first_cand = qk_sub_warp * Int32(8)
        c0 = warp_first_cand + ltid * Int32(2)
        c1 = c0 + Int32(1)
        cand_row_off = (warp_first_cand + gid) * Int32(_FP8_RECORD_BYTES)

        a_row = lane & Int32(7)
        a_half = (lane >> Int32(3)) & Int32(1)
        l16 = lane & Int32(15)
        v_perm_row = (l16 & Int32(1)) | ((l16 & Int32(12)) >> Int32(1)) | ((l16 & Int32(2)) << Int32(2))
        sm_p_g_addr = sm_p_wg_addr + (w_in_wg >> Int32(1)) * Int32(self.sm_p_group_bytes)
        v_lane_off = v_perm_row * Int32(_FP8_RECORD_BYTES) + w_in_wg * Int32(64)
        p_write_off = gid * Int32(64) + (qk_sub_warp ^ (gid >> Int32(1))) * Int32(16) + ltid * Int32(4)
        qk_part_off = (qk_sub_warp * Int32(32) + lane) * Int32(8)
        kv_stage_bytes = Int32(self.bi * _FP8_RECORD_BYTES)

        k_half = cute.make_rmem_tensor((4,), Uint32)
        cons_phase = Int32(0)
        for it in cutlass.range(wg_chunks, unroll=1):
            sub = Int32(it) & Int32(1)
            slot = Int32(slot_bias) + sub * Int32(2)
            cute.arch.mbarrier_wait(mbar_base + slot, phase=cons_phase)
            kv_buf = kv_base_addr + slot * kv_stage_bytes
            cand_row_addr = kv_buf + cand_row_off
            tok_base = (first_wg_chunk + Int32(it) * Int32(2)) * Int32(32)

            # S1 + S2: 2-way K-split QK (18 MMAs per warp, XOR-swizzled SMEM q_nope + q_rope, 0 global LDGs).
            qk0 = Float32(0.0)
            qk1 = Float32(0.0)
            blk_start = qk_half * Int32(2)
            for blk_i in cutlass.range(2, unroll=1):
                blk = blk_start + Int32(blk_i)
                t0 = Float32(0.0)
                t1 = Float32(0.0)
                for hk in cutlass.range_constexpr(2):
                    kb = cand_row_addr + blk * Int32(128) + Int32(hk * 16) + ltid * Int32(32)
                    u0, u1, u2, u3 = ld_shared_v4_u32(kb)
                    k_half[0] = u0
                    k_half[1] = u1
                    k_half[2] = u2
                    k_half[3] = u3
                    for j in cutlass.range_constexpr(4):
                        ks: cutlass.Constexpr = hk * 4 + j
                        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(k_half[j])
                        swiz_ch = (Int32(ks * 2) + a_half) ^ a_row
                        a0, a2 = ldmatrix_m8n8x2_b16(
                            q_nope_addr + (blk * Int32(8) + a_row) * Int32(256) + swiz_ch * Int32(16)
                        )
                        t0, t1, _, _ = mma_m16n8k16_f32_bf16(
                            t0, t1, Float32(0.0), Float32(0.0),
                            a0, Uint32(0), a2, Uint32(0), b0, b1,
                        )
                sc_off = Int32(512) + blk * Int32(4)
                ks0 = ld_shared_f32(kv_buf + c0 * Int32(_FP8_RECORD_BYTES) + sc_off)
                ks1 = ld_shared_f32(kv_buf + c1 * Int32(_FP8_RECORD_BYTES) + sc_off)
                qk0 = qk0 + t0 * ks0
                qk1 = qk1 + t1 * ks1

            rope_row_addr = cand_row_addr + Int32(528)
            rope_start = qk_half * Int32(2)
            for rks_i in cutlass.range_constexpr(2):
                rks = rope_start + Int32(rks_i)
                swiz_r = (rks * Int32(2) + a_half) ^ a_row
                a0, a2 = ldmatrix_m8n8x2_b16(q_rope_addr + a_row * Int32(128) + swiz_r * Int32(16))
                b0 = ld_shared_u32(rope_row_addr + rks * Int32(32) + ltid * Int32(4))
                b1 = ld_shared_u32(rope_row_addr + rks * Int32(32) + Int32(16) + ltid * Int32(4))
                qk0, qk1, _, _ = mma_m16n8k16_f32_bf16(
                    qk0, qk1, Float32(0.0), Float32(0.0),
                    a0, Uint32(0), a2, Uint32(0), b0, b1,
                )

            if qk_half == Int32(1):
                st_shared_f32(qk_part_wg_addr + qk_part_off, qk0)
                st_shared_f32(qk_part_wg_addr + qk_part_off + Int32(4), qk1)
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=256)

            lmax0 = Float32(-1.0e30)
            p0 = Float32(0.0)
            p1 = Float32(0.0)
            if qk_half == Int32(0):
                qk0 = qk0 + ld_shared_f32(qk_part_wg_addr + qk_part_off)
                qk1 = qk1 + ld_shared_f32(qk_part_wg_addr + qk_part_off + Int32(4))
                t_c0 = tok_base + c0
                t_c1 = tok_base + c1
                v_c0 = Int32(1) if (t_c0 >= visible_begin and t_c0 < visible_end) else Int32(0)
                v_c1 = Int32(1) if (t_c1 >= visible_begin and t_c1 < visible_end) else Int32(0)
                qk0 = (qk0 * sm_scale_log2) if v_c0 != Int32(0) else Float32(-1.0e30)
                qk1 = (qk1 * sm_scale_log2) if v_c1 != Int32(0) else Float32(-1.0e30)

                lmax0 = fmax_f32(qk0, qk1)
                lmax0 = fmax_f32(lmax0, cute.arch.shuffle_sync_bfly(lmax0, offset=2))
                lmax0 = fmax_f32(lmax0, cute.arch.shuffle_sync_bfly(lmax0, offset=1))

                p0 = exp2_approx(qk0 - lmax0) if (v_c0 != Int32(0) and lmax0 > Float32(-1.0e29)) else Float32(0.0)
                p1 = exp2_approx(qk1 - lmax0) if (v_c1 != Int32(0) and lmax0 > Float32(-1.0e29)) else Float32(0.0)
                lsum0 = p0 + p1
                lsum0 = lsum0 + cute.arch.shuffle_sync_bfly(lsum0, offset=2)
                lsum0 = lsum0 + cute.arch.shuffle_sync_bfly(lsum0, offset=1)

                if ltid == Int32(0):
                    red_off = (qk_sub_warp * Int32(8) + gid) * Int32(8)
                    st_shared_f32(reduce_wg_addr + red_off, lmax0)
                    st_shared_f32(reduce_wg_addr + red_off + Int32(4), lsum0)
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=256)

            # S4: All 8 warps in this WG reduce 4 candidate-warps from dedicated reduce_wg_addr.
            bmax0 = Float32(-1.0e30)
            for w_i in cutlass.range_constexpr(4):
                mv = ld_shared_f32(reduce_wg_addr + (Int32(w_i * 8) + gid) * Int32(8))
                bmax0 = fmax_f32(bmax0, mv)
            bsum0 = Float32(0.0)
            for w_i in cutlass.range_constexpr(4):
                r_off = (Int32(w_i * 8) + gid) * Int32(8)
                mv = ld_shared_f32(reduce_wg_addr + r_off)
                sv = ld_shared_f32(reduce_wg_addr + r_off + Int32(4))
                if mv > Float32(-1.0e29):
                    bsum0 = bsum0 + sv * exp2_approx(mv - bmax0)

            old_m0 = gmax0
            new_m0 = fmax_f32(old_m0, bmax0)
            alpha0 = exp2_approx(old_m0 - new_m0) if old_m0 > Float32(-1.0e29) else Float32(0.0)
            bscale0 = exp2_approx(bmax0 - new_m0) if bmax0 > Float32(-1.0e29) else Float32(0.0)
            for nt in cutlass.range_constexpr(8):
                acc_frag[nt * 2 + 0] = acc_frag[nt * 2 + 0] * alpha0
                acc_frag[nt * 2 + 1] = acc_frag[nt * 2 + 1] * alpha0
            gsum0 = gsum0 * alpha0 + bsum0 * bscale0
            gmax0 = new_m0

            # S5: Warps 0..3 write scale-premultiplied BF16 probabilities into XOR-swizzled sm_p[4].
            if qk_half == Int32(0):
                wr0 = exp2_approx(lmax0 - new_m0) if lmax0 > Float32(-1.0e29) else Float32(0.0)
                p0w = p0 * wr0
                p1w = p1 * wr0
                sc0_0, sc0_1, sc0_2, sc0_3 = ld_shared_v4_f32(kv_buf + c0 * Int32(_FP8_RECORD_BYTES) + Int32(512))
                sc1_0, sc1_1, sc1_2, sc1_3 = ld_shared_v4_f32(kv_buf + c1 * Int32(_FP8_RECORD_BYTES) + Int32(512))
                st_shared_u32(sm_p_wg_addr + p_write_off, pack_f32x2_to_bfloat2(p0w * sc0_0, p1w * sc1_0))
                st_shared_u32(sm_p_wg_addr + Int32(self.sm_p_group_bytes) + p_write_off, pack_f32x2_to_bfloat2(p0w * sc0_1, p1w * sc1_1))
                st_shared_u32(sm_p_wg_addr + Int32(2 * self.sm_p_group_bytes) + p_write_off, pack_f32x2_to_bfloat2(p0w * sc0_2, p1w * sc1_2))
                st_shared_u32(sm_p_wg_addr + Int32(3 * self.sm_p_group_bytes) + p_write_off, pack_f32x2_to_bfloat2(p0w * sc0_3, p1w * sc1_3))
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=256)

            # S6: PV via XOR-swizzled ldmatrix_x2 + ldmatrix_m16n16x1_trans_b8 + 2-inst native fp8x4 widen (ks=0..1).
            vbase = kv_buf + v_lane_off
            for ks in cutlass.range(2, unroll=1):
                swiz_c = (ks * Int32(2) + a_half) ^ (a_row >> Int32(1))
                pa0, pa2 = ldmatrix_m8n8x2_b16(sm_p_g_addr + a_row * Int32(64) + swiz_c * Int32(16))
                next_code = Uint32(0)
                v_ks_base = vbase + ks * Int32(16 * _FP8_RECORD_BYTES)
                for nt in cutlass.range_constexpr(8):
                    code = Uint32(0)
                    if cutlass.const_expr(nt % 2 == 0):
                        code, next_code = ldmatrix_m16n16x1_trans_b8(v_ks_base + Int32((nt // 2) * 16))
                    else:
                        code = next_code
                    vb0, vb1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(code)
                    d0, d1, _, _ = mma_m16n8k16_f32_bf16(
                        acc_frag[nt * 2 + 0], acc_frag[nt * 2 + 1], Float32(0.0), Float32(0.0),
                        pa0, Uint32(0), pa2, Uint32(0), vb0, vb1,
                    )
                    acc_frag[nt * 2 + 0] = d0
                    acc_frag[nt * 2 + 1] = d1

            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=256)
            if wg_tid == Int32(0):
                cute.arch.mbarrier_arrive(mbar_base + Int32(4) + slot)
            if sub == Int32(1):
                cons_phase = cons_phase ^ Int32(1)
        return gmax0, gsum0

    @cute.kernel
    def kernel(
        self,
        query: cute.Pointer,
        kv_cache: cute.Pointer,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        sm_scale_log2: Float32,
        page_table_width: Int64,
        batch: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        query_row = Int32(cute.arch.block_idx()[0])
        head_tile = Int32(cute.arch.block_idx()[1])
        head_base = head_tile * Int32(self.hpb)

        request, visible_begin, visible_end = _resolve_row_bounds(
            query_row, batch, cache_seqlens, cu_seqlens_q,
            singleton_requests=self.singleton_requests, window_size=self.window_size,
        )
        first_valid_chunk = visible_begin // Int32(self.bi)
        valid_chunks = (visible_end + Int32(self.bi - 1)) // Int32(self.bi)
        active_chunks = valid_chunks - first_valid_chunk
        if active_chunks < Int32(0):
            active_chunks = Int32(0)

        if active_chunks == Int32(0):
            if tid < Int32(self.hpb):
                final_lse[query_row, head_base + tid] = Float32(-Float32.inf)
            _exit_thread()

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        kv_base_addr = shared_ptr_to_u32(st.kv.data_ptr())
        q_nope_addr = shared_ptr_to_u32(st.q_nope.data_ptr())
        q_rope_addr = shared_ptr_to_u32(st.q_rope.data_ptr())
        sm_p_addr = shared_ptr_to_u32(st.sm_p.data_ptr())
        qk_part_addr = shared_ptr_to_u32(st.qk_part.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        mbar_base = st.mbar.data_ptr()

        if tid == Int32(0):
            for mb_i in cutlass.range_constexpr(8):
                cute.arch.mbarrier_init(mbar_base + Int32(mb_i), Int32(1))
        cute.arch.barrier()

        kv_stage_bytes = Int32(self.bi * _FP8_RECORD_BYTES)

        if warp_id == Int32(16):
            cache_base = Int64(kv_cache.toint())
            row_table_offset = Int64(request) * page_table_width
            prod_slot = Int32(0)
            prod_phase = Int32(1)
            for local_chunk in cutlass.range(active_chunks, unroll=1):
                chunk = first_valid_chunk + Int32(local_chunk)
                token_begin = chunk * Int32(self.bi)
                cute.arch.mbarrier_wait(mbar_base + Int32(4) + prod_slot, phase=prod_phase)
                _io_gather_fp8_bi32_no_tokbuf(
                    cache_base, page_table, row_table_offset,
                    kv_base_addr + prod_slot * kv_stage_bytes,
                    mbar_base + prod_slot,
                    token_begin, visible_begin, visible_end, lane,
                )
                prod_slot = (prod_slot + Int32(1)) & Int32(3)
                if prod_slot == Int32(0):
                    prod_phase = prod_phase ^ Int32(1)
        else:
            gid = lane >> Int32(2)
            ltid = lane & Int32(3)
            wg = warp_id >> Int32(3)       # 0 or 1
            w_in_wg = warp_id & Int32(7)   # 0..7
            wg_tid = tid & Int32(255)
            q_gmem_base = Int64(query.toint()) + (Int64(query_row) * Int64(self.num_heads) + Int64(head_base)) * Int64(576 * 2)

            # S0: Stage all 512 Q-NoPE dims (1 vec/thread) + 64 Q-RoPE dims into XOR-swizzled SMEM.
            h_loc = tid >> Int32(6)          # 0..7
            v_loc = tid & Int32(63)          # 0..63 (blk*16 + l*4 + seg)
            blk = v_loc >> Int32(4)          # 0..3
            in_blk = v_loc & Int32(15)
            l = in_blk >> Int32(2)           # 0..3
            seg = in_blk & Int32(3)          # 0..3
            src_q = q_gmem_base + Int64(h_loc) * Int64(576 * 2) + Int64(v_loc * 16)
            u0, u1, u2, u3 = ld_global_nc_v4_u32(src_q)
            row_base = q_nope_addr + (blk * Int32(8) + h_loc) * Int32(256)
            ch0 = seg * Int32(4)
            st_shared_u32(row_base + ((ch0 + Int32(0)) ^ h_loc) * Int32(16) + l * Int32(4), u0)
            st_shared_u32(row_base + ((ch0 + Int32(1)) ^ h_loc) * Int32(16) + l * Int32(4), u1)
            st_shared_u32(row_base + ((ch0 + Int32(2)) ^ h_loc) * Int32(16) + l * Int32(4), u2)
            st_shared_u32(row_base + ((ch0 + Int32(3)) ^ h_loc) * Int32(16) + l * Int32(4), u3)

            if tid < Int32(64):
                rh = tid >> Int32(3)
                rc = tid & Int32(7)
                r0, r1, r2, r3 = ld_global_nc_v4_u32(q_gmem_base + Int64(rh) * Int64(576 * 2) + Int64(1024 + rc * 16))
                st_shared_v4_u32(q_rope_addr + rh * Int32(128) + (rc ^ rh) * Int32(16), r0, r1, r2, r3)
            cute.arch.barrier(barrier_id=5, number_of_threads=512)

            acc_frag = cute.make_rmem_tensor((16,), Float32)
            for i in cutlass.range_constexpr(16):
                acc_frag[i] = Float32(0.0)

            gmax0 = Float32(-1.0e30)
            gsum0 = Float32(0.0)
            if wg == Int32(0):
                wg0_chunks = (active_chunks + Int32(1)) >> Int32(1)
                gmax0, gsum0 = self._run_wg_loop(
                    wg0_chunks,
                    first_valid_chunk,
                    visible_begin, visible_end,
                    kv_base_addr,
                    q_nope_addr, q_rope_addr,
                    sm_p_addr, qk_part_addr, reduce_addr,
                    mbar_base,
                    sm_scale_log2, w_in_wg, wg_tid, lane, gid, ltid, acc_frag,
                    slot_bias=0, barrier_id=3,
                )
            else:
                wg1_chunks = active_chunks >> Int32(1)
                gmax0, gsum0 = self._run_wg_loop(
                    wg1_chunks,
                    first_valid_chunk + Int32(1),
                    visible_begin, visible_end,
                    kv_base_addr,
                    q_nope_addr, q_rope_addr,
                    sm_p_addr + Int32(self.sm_p_wg_bytes),
                    qk_part_addr + Int32(self.qk_part_wg_bytes),
                    reduce_addr + Int32(self.reduce_wg_bytes),
                    mbar_base,
                    sm_scale_log2, w_in_wg, wg_tid, lane, gid, ltid, acc_frag,
                    slot_bias=1, barrier_id=4,
                )

            # In-SMEM full-FP32 merge inside dead kv[0] (16,384 B acc + 64 B max/sum <= 41,984 B).
            cute.arch.barrier(barrier_id=5, number_of_threads=512)
            if wg == Int32(1):
                for nt in cutlass.range_constexpr(8):
                    d0 = w_in_wg * Int32(64) + Int32(nt * 8) + ltid * Int32(2)
                    st_shared_f32(kv_base_addr + (gid * Int32(512) + d0) * Int32(4), acc_frag[nt * 2 + 0])
                    st_shared_f32(kv_base_addr + (gid * Int32(512) + d0 + Int32(1)) * Int32(4), acc_frag[nt * 2 + 1])
                if w_in_wg == Int32(0) and ltid == Int32(0):
                    st_shared_f32(kv_base_addr + Int32(16384) + gid * Int32(8), gmax0)
                    st_shared_f32(kv_base_addr + Int32(16384) + gid * Int32(8) + Int32(4), gsum0)
            cute.arch.barrier(barrier_id=5, number_of_threads=512)

            if wg == Int32(0):
                m1 = ld_shared_f32(kv_base_addr + Int32(16384) + gid * Int32(8))
                s1 = ld_shared_f32(kv_base_addr + Int32(16384) + gid * Int32(8) + Int32(4))
                m_fin = fmax_f32(gmax0, m1)
                w0 = exp2_approx(gmax0 - m_fin) if gmax0 > Float32(-1.0e29) else Float32(0.0)
                w1 = exp2_approx(m1 - m_fin) if m1 > Float32(-1.0e29) else Float32(0.0)
                s_fin = gsum0 * w0 + s1 * w1
                inv = Float32(1.0) / s_fin if s_fin > Float32(0.0) else Float32(0.0)
                h0 = head_base + gid
                for nt in cutlass.range_constexpr(8):
                    d0 = w_in_wg * Int32(64) + Int32(nt * 8) + ltid * Int32(2)
                    v1_0 = ld_shared_f32(kv_base_addr + (gid * Int32(512) + d0) * Int32(4))
                    v1_1 = ld_shared_f32(kv_base_addr + (gid * Int32(512) + d0 + Int32(1)) * Int32(4))
                    o0 = (acc_frag[nt * 2 + 0] * w0 + v1_0 * w1) * inv
                    o1 = (acc_frag[nt * 2 + 1] * w0 + v1_1 * w1) * inv
                    off0 = cute.crd2idx((query_row, h0, d0), output.layout)
                    store_bf16x2(get_ptr_as_int64(output, off0), o0, o1)
                if w_in_wg == Int32(0) and ltid == Int32(0):
                    lse_val = (
                        (log2_approx(s_fin) + m_fin) * Float32(_LN2)
                        if s_fin > Float32(0.0) else Float32(-Float32.inf)
                    )
                    final_lse[query_row, h0] = lse_val


class _Hy4MergeKernel:
    """Fast split-K reduction for Hy4 MLA partials (streaming nc loads, natural-log LSE)."""

    def __init__(self, num_splits: int, value_dim: int = 512):
        self.num_splits = int(num_splits)
        self.value_dim = int(value_dim)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "weight": cute.struct.Align[cute.struct.MemRange[Float32, self.num_splits], 16],
            "inverse": cute.struct.Align[cute.struct.MemRange[Float32, 1], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        active_splits: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(partial_output, partial_lse, output, final_lse, active_splits).launch(
            grid=(output.shape[0], output.shape[1], 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        active_splits: Int32,
    ):
        thread = Int32(cute.arch.thread_idx()[0])
        lane = thread & Int32(31)
        warp_id = thread >> Int32(5)
        query = Int32(cute.arch.block_idx()[0])
        head = Int32(cute.arch.block_idx()[1])
        dimension = thread * Int32(4)

        smem = utils.SmemAllocator()
        st = smem.allocate(self._storage())
        split_weight = st.weight.get_tensor(cute.make_layout((self.num_splits,), stride=(1,)))
        inverse_storage = st.inverse.get_tensor(cute.make_layout((1,), stride=(1,)))

        if warp_id == Int32(0):
            merged_max = Float32(-Float32.inf)
            if lane < active_splits:
                merged_max = Float32(partial_lse[query, head, lane])
            for offset in (16, 8, 4, 2, 1):
                other_max = cute.arch.shuffle_sync_bfly(merged_max, offset=offset)
                if other_max > merged_max:
                    merged_max = other_max

            merged_sum = Float32(0.0)
            if lane < active_splits:
                part_lse = Float32(partial_lse[query, head, lane])
                weight = exp2_approx(part_lse - merged_max) if part_lse != Float32(-Float32.inf) else Float32(0.0)
                split_weight[lane] = weight
                merged_sum = weight
            for offset in (16, 8, 4, 2, 1):
                merged_sum = merged_sum + cute.arch.shuffle_sync_bfly(merged_sum, offset=offset)

            if lane == Int32(0):
                inverse = Float32(1.0) / merged_sum if merged_sum > Float32(0.0) else Float32(0.0)
                lse_val = (merged_max + log2_approx(merged_sum)) * Float32(_LN2) if merged_sum > Float32(0.0) else Float32(-Float32.inf)
                inverse_storage[0] = inverse
                final_lse[query, head] = lse_val

        cute.arch.barrier()

        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        head_base_off = (Int64(query) * Int64(partial_output.shape[1]) + Int64(head)) * Int64(self.num_splits)
        split = Int32(0)
        while split < active_splits:
            w = Float32(split_weight[split])
            if w > Float32(0.0):
                partial_offset = (head_base_off + Int64(split)) * Int64(self.value_dim) + Int64(dimension)
                packed0, packed1 = ld_global_nc_v2_u32(get_ptr_as_int64(partial_output, partial_offset))
                v0, v1 = bfloat2_to_float2_scaled(packed0, w)
                v2, v3 = bfloat2_to_float2_scaled(packed1, w)
                acc0 = acc0 + v0
                acc1 = acc1 + v1
                acc2 = acc2 + v2
                acc3 = acc3 + v3
            split = split + Int32(1)

        inverse = Float32(inverse_storage[0])
        out_off = cute.crd2idx((query, head, dimension), output.layout)
        st_global_v2_u32(
            get_ptr_as_int64(output, out_off),
            pack_f32x2_to_bfloat2(acc0 * inverse, acc1 * inverse),
            pack_f32x2_to_bfloat2(acc2 * inverse, acc3 * inverse),
        )


class _Hy4Attention:
    def __init__(self, g: Hy4MlaGeometry, rows: int, max_context: int, mode: str, *, page_size: int):
        self.g = g
        self.page_size = int(page_size)
        self.max_context = int(max_context)
        self.layout = hy4_attention_layout(g, rows, max_context=max_context, mode=mode)
        self.splits = self.layout["num_splits"]
        self.use_fast_fp8 = g.kv_cache == "fp8" and _is_tp1(g)
        if self.use_fast_fp8:
            self.has_h32 = (g.heads >= 32 and g.heads % 32 == 0)
            self.has_swa_2wg = (self.page_size == 64 and self.splits == 1 and g.heads % 8 == 0)
            self.fwd_h16 = _Hy4Fp8MlaForward(
                num_heads=g.heads, hpb=16, bi=32, page_size=self.page_size,
                window_size=g.window_size, max_splits=self.splits,
            )
            self.fwd_h32 = (
                _Hy4Fp8MlaForward(
                    num_heads=g.heads, hpb=32, bi=32, page_size=self.page_size,
                    window_size=g.window_size, max_splits=self.splits,
                )
                if self.has_h32 else None
            )
            self.fwd_swa_2wg = (
                _Hy4Swa2WgForward(
                    num_heads=g.heads, page_size=self.page_size, window_size=g.window_size,
                )
                if self.has_swa_2wg else None
            )
            self.merge = _Hy4MergeKernel(num_splits=self.splits, value_dim=512) if self.splits > 1 else None
        else:
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

        if cutlass.const_expr(self.use_fast_fp8):
            if rows > Int32(0):
                if cutlass.const_expr(self.page_size == 1):
                    # DSA selected-token gather: dynamic split-K + hpb=16/32 dispatch from runtime SM count.
                    sm_count = runtime_num_sm()
                    h16_tiles: cutlass.Constexpr = g.heads // 16
                    if cutlass.const_expr(self.has_h32):
                        h32_tiles: cutlass.Constexpr = g.heads // 32
                        if rows * Int32(h16_tiles * self.splits) <= sm_count:
                            eff = Int32(self.splits)
                            self.fwd_h16(
                                query, kv_cache, table, lengths, cu, out, lse_out, po, pl,
                                scale_log2, page_table_width, rows, batch, eff, stream,
                            )
                            if cutlass.const_expr(self.splits > 1):
                                if eff > Int32(1):
                                    self.merge(po, pl, out, lse_out, eff, stream)
                        else:
                            max_chunks_32: cutlass.Constexpr = (self.max_context + 31) // 32
                            max_s = sm_count // (rows * Int32(h32_tiles))
                            if max_s > Int32(self.splits):
                                max_s = Int32(self.splits)
                            if max_s < Int32(1):
                                max_s = Int32(1)
                            cps = (Int32(max_chunks_32) + max_s - Int32(1)) // max_s
                            eff = (Int32(max_chunks_32) + cps - Int32(1)) // cps
                            if eff > Int32(self.splits):
                                eff = Int32(self.splits)
                            if eff < Int32(1):
                                eff = Int32(1)
                            self.fwd_h32(
                                query, kv_cache, table, lengths, cu, out, lse_out, po, pl,
                                scale_log2, page_table_width, rows, batch, eff, stream,
                            )
                            if cutlass.const_expr(self.splits > 1):
                                if eff > Int32(1):
                                    self.merge(po, pl, out, lse_out, eff, stream)
                    else:
                        max_chunks_32: cutlass.Constexpr = (self.max_context + 31) // 32
                        max_s = sm_count // (rows * Int32(h16_tiles))
                        if max_s > Int32(self.splits):
                            max_s = Int32(self.splits)
                        if max_s < Int32(1):
                            max_s = Int32(1)
                        cps = (Int32(max_chunks_32) + max_s - Int32(1)) // max_s
                        eff = (Int32(max_chunks_32) + cps - Int32(1)) // cps
                        if eff > Int32(self.splits):
                            eff = Int32(self.splits)
                        if eff < Int32(1):
                            eff = Int32(1)
                        self.fwd_h16(
                            query, kv_cache, table, lengths, cu, out, lse_out, po, pl,
                            scale_log2, page_table_width, rows, batch, eff, stream,
                        )
                        if cutlass.const_expr(self.splits > 1):
                            if eff > Int32(1):
                                self.merge(po, pl, out, lse_out, eff, stream)
                else:
                    # SWA-512 (page_size=64): decode (splits>1) or extend (splits=1).
                    h8_tiles: cutlass.Constexpr = g.heads // 8
                    h16_tiles: cutlass.Constexpr = g.heads // 16
                    if cutlass.const_expr(self.splits > 1):
                        # swa_decode: split-K for tiny rows (1..4), single-pass above.
                        eff = Int32(1)
                        if rows == Int32(1):
                            eff = Int32(self.splits)
                        elif rows <= Int32(4):
                            eff = Int32(4) if cutlass.const_expr(self.splits >= 4) else Int32(self.splits)
                        self.fwd_h16(
                            query, kv_cache, table, lengths, cu, out, lse_out, po, pl,
                            scale_log2, page_table_width, rows, batch, eff, stream,
                        )
                        if eff > Int32(1):
                            self.merge(po, pl, out, lse_out, eff, stream)
                    else:
                        sm_count = runtime_num_sm()
                        # swa_extend: 2-WG hpb=8 (1 wave up to sm_count/8 rows, e.g. rows=2..16 on SM120),
                        # hpb=16 (up to sm_count/4 rows, e.g. rows=32), hpb=32 above (rows=64 and prefill).
                        if cutlass.const_expr(self.has_swa_2wg) and rows * Int32(h8_tiles) <= sm_count:
                            self.fwd_swa_2wg(
                                query, kv_cache, table, lengths, cu, out, lse_out,
                                scale_log2, page_table_width, rows, batch, stream,
                            )
                        elif cutlass.const_expr(self.has_h32) and rows * Int32(h16_tiles) > sm_count:
                            self.fwd_h32(
                                query, kv_cache, table, lengths, cu, out, lse_out, po, pl,
                                scale_log2, page_table_width, rows, batch, Int32(1), stream,
                            )
                        else:
                            self.fwd_h16(
                                query, kv_cache, table, lengths, cu, out, lse_out, po, pl,
                                scale_log2, page_table_width, rows, batch, Int32(1), stream,
                            )
        else:
            q = cute.make_tensor(cute.recast_ptr(query, dtype=Uint8), cute.make_layout((1,)))
            kv = cute.make_tensor(cute.recast_ptr(kv_cache, dtype=Uint8), cute.make_layout((1,)))
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
        Operand("q_a", torch.bfloat16, f"[rows,{g.q_lora_rank}]"),
        Operand("kv_a", torch.bfloat16, "[rows,576]"),
        Operand("q_norm", torch.bfloat16, f"[{g.q_lora_rank}]"),
        Operand("kv_norm", torch.bfloat16, "[512]"),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("positions", torch.int64, "[rows,3]", align=8, note="M-RoPE (T,H,W) rotary positions"),
        Operand("inv_freq", torch.float32, "[32]", align=4),
        Operand("q_resid", torch.bfloat16, f"[rows,{g.q_lora_rank}]", "out"),
        Operand(
            "kv_cache",
            torch.uint8 if g.kv_cache == "fp8" else torch.bfloat16,
            "[pages,64,656]" if g.kv_cache == "fp8" else "[pages,64,576]",
            "inout",
        ),
    )
    key = (
        (g.kv_cache, g.norm_eps, "rank_norm_pack", g.mrope_section)
        if g.q_lora_rank == 1024
        else (g.q_lora_rank, g.kv_cache, g.norm_eps, "rank_norm_pack", g.mrope_section)
    )
    geometry = {
        "kv_cache": g.kv_cache,
        "norm_eps": g.norm_eps,
        "record_bytes": g.record_bytes,
        "mrope_section": list(g.mrope_section),
        **({"q_lora_rank": g.q_lora_rank} if g.q_lora_rank != 1024 else {}),
    }
    return compile_program(
        _Hy4RankNormPack(g),
        name=f"hy4_mla_rank_norm_pack_{g.kv_cache}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=key,
        geometry=geometry,
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
    fast = _is_tp1(g)
    key = (
        (g.heads, "q_absorb", g.mrope_section)
        if g.q_lora_rank == 1024
        else (g.heads, "q_absorb", g.mrope_section, fast)
    )
    return compile_program(
        _Hy4Tp1QAbsorb(g) if fast else _Hy4QAbsorb(g),
        name=f"hy4_mla_q_absorb_h{g.heads}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=key,
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
    fast = _is_tp1(g)
    key = (
        (g.heads, "sink_gate_uv")
        if g.q_lora_rank == 1024
        else (g.heads, "sink_gate_uv", fast)
    )
    return compile_program(
        _Hy4Tp1SinkGateUv(g) if fast else _Hy4SinkGateUv(g),
        name=f"hy4_mla_sink_gate_uv_h{g.heads}",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=key,
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
    if g.heads not in (32, 64) or g.window_size is not None:
        raise ValueError("Hy4 DSA partial requires 32-head or 64-head full-context DSA geometry")
    half = g.heads // 2
    if head_begin not in (0, half) or head_count != half:
        raise ValueError(f"Hy4 DSA partial requires head_begin in (0, {half}) and head_count == {half}")
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
        Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
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
