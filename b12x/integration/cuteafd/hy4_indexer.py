"""Native AOT 16-head compressed indexer kernels for Hy4-flash DSA layers.

Matches ``python/reference/families/hy4_flash/model.py`` (``index_projections`` and
``recall_groups``):

- 16 indexer heads, head_dim = 128, interleaved tail-64 RoPE (dims 64..128),
  compress_ratio = 4 (16 groups per 64-token KV page), group_topk = 512
  (top-511 prior complete groups + current group -> up to 2048 physical KV slots).
- Group cache layout: 132 bytes per group (128 E4M3 codes + 4B F32 per-group scale),
  16 groups per page = 2112 bytes per page (matching ``execution.rs::INDEX_GROUP_RECORD_BYTES``).

Four composable AOT programs:
1. ``compile_hy4_indexer_query_rope_aot``:
   Interleaved tail-64 RoPE on ``q_raw [rows, 16, 128]`` + scale ``w_raw [rows, 16]`` by
   ``16^-0.5 * 128^-0.5``.
2. ``compile_hy4_indexer_pool_group_aot``:
   Exact F32 gated 4-token pooling (``key_value_projection``, ``key_gate_projection`` +
   ``intra_group_ape``, softmax over the 4 tokens, F32 ``key_norm`` RMSNorm with eps=1e-5,
   interleaved tail-64 RoPE at ``group_pos = 4*g``) -> pack E4M3+F32 into ``group_cache``.
3. ``compile_hy4_indexer_score_groups_aot``:
   Scores ``sum_h weights[r, h] * relu(dot(q[r, h], group_key[g]))`` across all causal
   complete groups ``0 <= g < group_ids[r]`` addressed through ``page_table``.
4. ``compile_hy4_indexer_expand_slots_aot``:
   Expands recalled group ids ``[rows, k]`` (top-511 + current group) by 4 into causal
   physical KV token slots ``[rows, 2048]`` + valid count ``candidate_lengths [rows]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    red_add_shared_i32,
    shared_ptr_to_u32,
)
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _fp8_scale
from .context_split import _pair_before, _sort_pairs

__all__ = [
    "HY4_FLASH_INDEXER",
    "Hy4IndexerGeometry",
    "compile_hy4_indexer_expand_slots_aot",
    "compile_hy4_indexer_merge_expand_owned_aot",
    "compile_hy4_indexer_pool_group_aot",
    "compile_hy4_indexer_query_rope_aot",
    "compile_hy4_indexer_score_groups_aot",
    "compile_hy4_indexer_scored_topk_aot",
    "compile_hy4_indexer_topk_expand_aot",
]

_GROUP_RECORD_BYTES = 132  # 128 E4M3 + 4B F32 scale
_GROUPS_PER_PAGE = 16      # 64 tokens / compress_ratio 4
_PAGE_BYTES = _GROUPS_PER_PAGE * _GROUP_RECORD_BYTES  # 2112


from .hy4_mla import _mrope_position, _validate_mrope


@dataclass(frozen=True)
class Hy4IndexerGeometry:
    heads: int = 16
    head_dim: int = 128
    rope_dim: int = 64
    compress_ratio: int = 4
    group_topk: int = 512
    candidate_topk: int = 2048
    page_rows: int = 64
    norm_eps: float = 1e-5
    mrope_section: tuple = (12, 10, 10)

    def __post_init__(self) -> None:
        if (self.heads, self.head_dim, self.rope_dim, self.compress_ratio,
                self.group_topk, self.candidate_topk, self.page_rows) != (16, 128, 64, 4, 512, 2048, 64):
            raise ValueError("unsupported Hy4 compressed indexer geometry")
        _validate_mrope(self.mrope_section, self.rope_dim // 2)

    @property
    def weight_scale(self) -> float:
        return (self.heads ** -0.5) * (self.head_dim ** -0.5)

    @property
    def page_bytes(self) -> int:
        return _PAGE_BYTES


HY4_FLASH_INDEXER = Hy4IndexerGeometry()




@cute.jit
def _warp_sum(v: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        v = Float32(v + cute.arch.shuffle_sync_bfly(v, offset=1 << shift))
    return v


@cute.jit
def _warp_max(v: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(v, offset=1 << shift)
        v = fmax_f32(v, other)
    return v


@cute.jit
def _decode_e4m3fn(byte_val: Uint8) -> Float32:
    """Exact IEEE F32 decode of OCP E4M3FN byte."""
    u = Int32(byte_val)
    sign = (u >> Int32(7)) & Int32(1)
    exp = (u >> Int32(3)) & Int32(0xF)
    mant = u & Int32(0x7)
    val = Float32(0.0)
    if exp == Int32(0):
        # Subnormal: mant * 2^-9
        val = Float32(mant) * Float32(1.0 / 512.0)
    else:
        # Normal: (1 + mant/8) * 2^(exp - 7)
        val = (Float32(1.0) + Float32(mant) * Float32(0.125)) * cute.math.exp2(Float32(exp - Int32(7)), fastmath=False)
    return -val if sign != Int32(0) else val


class _Hy4IndexerQueryRope:
    """Interleaved tail-64 RoPE on [rows, 16, 128] + weight scaling [rows, 16]."""

    def __init__(self, g: Hy4IndexerGeometry):
        self.heads = g.heads
        self.weight_scale = float(g.weight_scale)
        self.mrope_section = g.mrope_section

    @cute.jit
    def __call__(
        self,
        q_raw: cute.Pointer,
        w_raw: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        q_out: cute.Pointer,
        w_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(q_raw, w_raw, positions, inv_freq, q_out, w_out).launch(
            grid=(rows, self.heads, 1), block=(64, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        q_raw: cute.Pointer,
        w_raw: cute.Pointer,
        positions: cute.Pointer,
        inv_freq: cute.Pointer,
        q_out: cute.Pointer,
        w_out: cute.Pointer,
    ):
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        tid = Int64(cute.arch.thread_idx()[0])
        base = (row * Int64(self.heads) + head) * Int64(128)
        # Leading 64 dims pass through unchanged.
        q_out[base + tid] = q_raw[base + tid]
        # Tail 64 dims rotated in 32 interleaved pairs by threads 0..31.
        if tid < Int64(32):
            angle = _mrope_position(positions, row, tid, self.mrope_section[1], self.mrope_section[2]) * Float32(inv_freq[tid])
            c = cute.math.cos(angle, fastmath=False)
            s = cute.math.sin(angle, fastmath=False)
            off = base + Int64(64) + tid * Int64(2)
            x0 = Float32(q_raw[off])
            x1 = Float32(q_raw[off + Int64(1)])
            q_out[off] = (x0 * c - x1 * s).to(BFloat16)
            q_out[off + Int64(1)] = (x1 * c + x0 * s).to(BFloat16)
        if tid == Int64(0):
            rh = row * Int64(self.heads) + head
            w_out[rh] = Float32(w_raw[rh]) * Float32(self.weight_scale)


class _Hy4IndexerPoolGroup:
    """Exact F32 4-token gated pooling + RMSNorm + tail-64 RoPE -> E4M3+F32 group cache.

    One CTA of 128 threads per complete 4-token group: thread ``d`` computes output dim ``d``
    for all 4 tokens, softmax over the 4 tokens, F32 RMSNorm across 128 dims, tail-64 RoPE,
    and packs the 132-byte E4M3+F32 group record at ``group_cache[group_slots[g]]``.
    """

    def __init__(self, g: Hy4IndexerGeometry):
        self.eps = float(g.norm_eps)
        self.mrope_section = g.mrope_section

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "tok": cute.struct.Align[cute.struct.MemRange[Float32, 4 * 128], 16],
            "pooled": cute.struct.Align[cute.struct.MemRange[Float32, 128], 16],
            "warp_buf": cute.struct.Align[cute.struct.MemRange[Float32, 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        tokens: cute.Pointer,
        w_kv: cute.Pointer,
        w_kg: cute.Pointer,
        ape: cute.Pointer,
        key_norm: cute.Pointer,
        group_positions: cute.Pointer,
        group_slots: cute.Pointer,
        inv_freq: cute.Pointer,
        group_cache: cute.Pointer,
        groups: Int32,
        stream: cuda.CUstream,
    ):
        g = Int64(groups)
        self.kernel(
            cute.make_tensor(tokens, cute.make_layout((g, 4, 128), stride=(512, 128, 1))),
            cute.make_tensor(w_kv, cute.make_layout((128, 128), stride=(128, 1))),
            cute.make_tensor(w_kg, cute.make_layout((128, 128), stride=(128, 1))),
            cute.make_tensor(ape, cute.make_layout((4, 128), stride=(128, 1))),
            cute.make_tensor(key_norm, cute.make_layout((128,))),
            group_positions,
            group_slots,
            inv_freq,
            group_cache,
        ).launch(grid=(groups, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tokens: cute.Tensor,
        w_kv: cute.Tensor,
        w_kg: cute.Tensor,
        ape: cute.Tensor,
        key_norm: cute.Tensor,
        group_positions: cute.Pointer,
        group_slots: cute.Pointer,
        inv_freq: cute.Pointer,
        group_cache: cute.Pointer,
    ):
        gid = Int64(cute.arch.block_idx()[0])
        slot = Int64(group_slots[gid])
        d = Int32(cute.arch.thread_idx()[0])
        lane = d % Int32(32)
        warp = d // Int32(32)
        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage())
        tok = st.tok.get_tensor(cute.make_layout((4, 128), stride=(128, 1)))
        pooled_s = st.pooled.get_tensor(cute.make_layout((128,)))
        warp_buf = st.warp_buf.get_tensor(cute.make_layout((4,)))

        for t in cutlass.range_constexpr(4):
            tok[t, d] = Float32(tokens[gid, t, d])
        cute.arch.sync_threads()

        vals = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        gates = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for t in cutlass.range_constexpr(4):
            v_acc = Float32(0.0)
            g_acc = Float32(0.0)
            for k in cutlass.range(128, unroll_full=True):
                tk = tok[t, k]
                v_acc = v_acc + tk * Float32(w_kv[d, k])
                g_acc = g_acc + tk * Float32(w_kg[d, k])
            vals[t] = v_acc
            gates[t] = g_acc + Float32(ape[t, d])

        # Softmax over the 4 tokens in the group for dim d.
        m = fmax_f32(fmax_f32(gates[0], gates[1]), fmax_f32(gates[2], gates[3]))
        e0 = cute.math.exp(gates[0] - m, fastmath=False)
        e1 = cute.math.exp(gates[1] - m, fastmath=False)
        e2 = cute.math.exp(gates[2] - m, fastmath=False)
        e3 = cute.math.exp(gates[3] - m, fastmath=False)
        denom = e0 + e1 + e2 + e3
        p = (vals[0] * e0 + vals[1] * e1 + vals[2] * e2 + vals[3] * e3) / denom

        # F32 RMSNorm across 128 dims.
        ss = _warp_sum(p * p)
        if lane == Int32(0):
            warp_buf[warp] = ss
        cute.arch.sync_threads()
        total_ss = warp_buf[0] + warp_buf[1] + warp_buf[2] + warp_buf[3]
        inv_rms = cute.math.rsqrt(total_ss / Float32(128.0) + Float32(self.eps))
        pooled_s[d] = p * inv_rms * Float32(key_norm[d])
        cute.arch.sync_threads()

        # Interleaved tail-64 RoPE on dims 64..128 in place in shared memory.
        if d < Int32(32):
            pair = Int64(d)
            angle = _mrope_position(group_positions, gid, pair, self.mrope_section[1], self.mrope_section[2]) * Float32(inv_freq[pair])
            c = cute.math.cos(angle, fastmath=False)
            s = cute.math.sin(angle, fastmath=False)
            i0 = Int32(64) + d * Int32(2)
            x0 = pooled_s[i0]
            x1 = pooled_s[i0 + Int32(1)]
            pooled_s[i0] = x0 * c - x1 * s
            pooled_s[i0 + Int32(1)] = x1 * c + x0 * s
        cute.arch.sync_threads()

        # Quantize 128 F32 values -> 128 E4M3 + 1 F32 scale (132 B record).
        val = pooled_s[d]
        wmax = _warp_max(fabs_f32(val))
        if lane == Int32(0):
            warp_buf[warp] = wmax
        cute.arch.sync_threads()
        amax = fmax_f32(fmax_f32(warp_buf[0], warp_buf[1]), fmax_f32(warp_buf[2], warp_buf[3]))
        scale = _fp8_scale(amax)
        if slot >= Int64(0):
            page = slot // Int64(_GROUPS_PER_PAGE)
            local = slot % Int64(_GROUPS_PER_PAGE)
            rec_addr = Int64(group_cache.toint()) + page * Int64(_PAGE_BYTES) + local * Int64(_GROUP_RECORD_BYTES)
            if d < Int32(32):
                words = cute.make_ptr(Uint32, rec_addr, cute.AddressSpace.gmem, assumed_align=4)
                i0 = d * Int32(4)
                words[d] = cvt_f32x4_to_e4m3x4(
                    div_rn_f32(pooled_s[i0], scale),
                    div_rn_f32(pooled_s[i0 + Int32(1)], scale),
                    div_rn_f32(pooled_s[i0 + Int32(2)], scale),
                    div_rn_f32(pooled_s[i0 + Int32(3)], scale),
                )
            if d == Int32(0):
                s_ptr = cute.make_ptr(Float32, rec_addr + Int64(128), cute.AddressSpace.gmem, assumed_align=4)
                s_ptr[0] = scale


class _Hy4IndexerScoreGroups:
    """Scores causal complete groups for each query row.

    Grid ``(groups_tiles, rows, 1)`` where each CTA has 128 threads (4 warps) and scores
    4 consecutive groups (1 warp per group, 32 lanes x 4 dims = 128 dims across all 16 heads).
    """

    def __init__(self, g: Hy4IndexerGeometry):
        self.heads = g.heads

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "q": cute.struct.Align[cute.struct.MemRange[Float32, 16 * 128], 16],
            "w": cute.struct.Align[cute.struct.MemRange[Float32, 16], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        q_rot: cute.Pointer,
        weights: cute.Pointer,
        group_cache: cute.Pointer,
        page_table: cute.Pointer,
        group_limits: cute.Pointer,
        scores_out: cute.Pointer,
        rows: Int32,
        max_groups: Int32,
        table_stride: Int32,
        stream: cuda.CUstream,
    ):
        tiles = (max_groups + Int32(3)) // Int32(4)
        self.kernel(
            q_rot, weights, group_cache, page_table, group_limits, scores_out, max_groups, table_stride
        ).launch(grid=(tiles, rows, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q_rot: cute.Pointer,
        weights: cute.Pointer,
        group_cache: cute.Pointer,
        page_table: cute.Pointer,
        group_limits: cute.Pointer,
        scores_out: cute.Pointer,
        max_groups: Int32,
        table_stride: Int32,
    ):
        tile = Int32(cute.arch.block_idx()[0])
        row = Int64(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid % Int32(32)

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage())
        q_s = st.q.get_tensor(cute.make_layout((16, 128), stride=(128, 1)))
        w_s = st.w.get_tensor(cute.make_layout((16,)))

        q_base = row * Int64(16 * 128)
        for h in cutlass.range_constexpr(16):
            q_s[h, tid] = Float32(q_rot[q_base + Int64(h * 128) + Int64(tid)])
        if tid < Int32(16):
            w_s[tid] = Float32(weights[row * Int64(16) + Int64(tid)])
        cute.arch.sync_threads()

        gid = tile * Int32(4) + warp
        if gid < max_groups:
            limit = Int32(group_limits[row])
            if gid >= limit:
                if lane == Int32(0):
                    scores_out[row * Int64(max_groups) + Int64(gid)] = Float32(-Float32.inf)
            else:
                page_idx = Int64(gid // Int32(_GROUPS_PER_PAGE))
                local = Int64(gid % Int32(_GROUPS_PER_PAGE))
                phys_page = Int64(page_table[row * Int64(table_stride) + page_idx])
                rec_addr = Int64(group_cache.toint()) + phys_page * Int64(_PAGE_BYTES) + local * Int64(_GROUP_RECORD_BYTES)
                codes = cute.make_ptr(Uint8, rec_addr, cute.AddressSpace.gmem, assumed_align=4)
                scale = Float32(cute.make_ptr(Float32, rec_addr + Int64(128), cute.AddressSpace.gmem, assumed_align=4)[0])

                k0 = _decode_e4m3fn(codes[Int64(lane * Int32(4))]) * scale
                k1 = _decode_e4m3fn(codes[Int64(lane * Int32(4) + Int32(1))]) * scale
                k2 = _decode_e4m3fn(codes[Int64(lane * Int32(4) + Int32(2))]) * scale
                k3 = _decode_e4m3fn(codes[Int64(lane * Int32(4) + Int32(3))]) * scale

                total = Float32(0.0)
                d0 = lane * Int32(4)
                for h in cutlass.range_constexpr(16):
                    dot = (
                        q_s[h, d0] * k0
                        + q_s[h, d0 + Int32(1)] * k1
                        + q_s[h, d0 + Int32(2)] * k2
                        + q_s[h, d0 + Int32(3)] * k3
                    )
                    dot = _warp_sum(dot)
                    if dot > Float32(0.0):
                        total = total + w_s[h] * dot
                if lane == Int32(0):
                    scores_out[row * Int64(max_groups) + Int64(gid)] = total


class _Hy4IndexerExpandSlots:
    """Expand recalled group ids [rows, num_recalled] + current group into causal physical KV slots."""

    def __init__(self, g: Hy4IndexerGeometry):
        self.candidate_topk = g.candidate_topk
        self.page_rows = g.page_rows

    @cute.jit
    def __call__(
        self,
        recalled_groups: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        slots_out: cute.Pointer,
        lengths_out: cute.Pointer,
        rows: Int32,
        num_recalled: Int32,
        table_stride: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            recalled_groups, positions, page_table, slots_out, lengths_out, num_recalled, table_stride
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        recalled_groups: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        slots_out: cute.Pointer,
        lengths_out: cute.Pointer,
        num_recalled: Int32,
        table_stride: Int32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        pos = Int32(positions[row])
        seq_len = pos + Int32(1)
        out_base = row * Int64(self.candidate_topk)
        pt_base = row * Int64(table_stride)

        if seq_len <= Int32(self.candidate_topk):
            # Every causal token 0..=pos fits inside the 2048 candidate budget.
            idx = tid
            while idx < Int32(self.candidate_topk):
                if idx < seq_len:
                    page = Int64(page_table[pt_base + Int64(idx // Int32(self.page_rows))])
                    slots_out[out_base + Int64(idx)] = Int32(page * Int64(self.page_rows) + Int64(idx % Int32(self.page_rows)))
                else:
                    slots_out[out_base + Int64(idx)] = Int32(-1)
                idx = idx + Int32(256)
            if tid == Int32(0):
                lengths_out[row] = seq_len
        else:
            # Long sequence: expand num_recalled prior groups (<=511) + current group (pos // 4).
            cur_group = pos // Int32(4)
            cur_valid = (pos % Int32(4)) + Int32(1)
            total_valid = num_recalled * Int32(4) + cur_valid
            if total_valid > Int32(self.candidate_topk):
                total_valid = Int32(self.candidate_topk)
            rec_base = row * Int64(num_recalled)
            idx = tid
            while idx < Int32(self.candidate_topk):
                if idx < num_recalled * Int32(4):
                    g_i = idx // Int32(4)
                    sub = idx % Int32(4)
                    gid = Int32(recalled_groups[rec_base + Int64(g_i)])
                    tok = gid * Int32(4) + sub
                    page = Int64(page_table[pt_base + Int64(tok // Int32(self.page_rows))])
                    slots_out[out_base + Int64(idx)] = Int32(page * Int64(self.page_rows) + Int64(tok % Int32(self.page_rows)))
                elif idx < total_valid:
                    sub = idx - num_recalled * Int32(4)
                    tok = cur_group * Int32(4) + sub
                    page = Int64(page_table[pt_base + Int64(tok // Int32(self.page_rows))])
                    slots_out[out_base + Int64(idx)] = Int32(page * Int64(self.page_rows) + Int64(tok % Int32(self.page_rows)))
                else:
                    slots_out[out_base + Int64(idx)] = Int32(-1)
                idx = idx + Int32(256)
            if tid == Int32(0):
                lengths_out[row] = total_valid


def compile_hy4_indexer_query_rope_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("q_raw", torch.bfloat16, "[rows,16,128]"),
        Operand("w_raw", torch.bfloat16, "[rows,16]"),
        Operand("positions", torch.int64, "[rows,3]", align=8, note="M-RoPE (T,H,W) rotary positions"),
        Operand("inv_freq", torch.float32, "[32]", align=4),
        Operand("q_out", torch.bfloat16, "[rows,16,128]", "out"),
        Operand("w_out", torch.float32, "[rows,16]", "out", align=4),
    )
    return compile_program(
        _Hy4IndexerQueryRope(g),
        name="hy4_indexer_query_rope",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "query_rope"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_indexer_pool_group_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("tokens", torch.bfloat16, "[groups,4,128]"),
        Operand("w_kv", torch.float32, "[128,128]"),
        Operand("w_kg", torch.float32, "[128,128]"),
        Operand("ape", torch.float32, "[4,128]"),
        Operand("key_norm", torch.float32, "[128]"),
        Operand("group_positions", torch.int64, "[groups,3]", align=8,
                note="M-RoPE (T,H,W) of each group's first token"),
        Operand("group_slots", torch.int64, "[groups]", align=8),
        Operand("inv_freq", torch.float32, "[32]", align=4),
        Operand("group_cache", torch.uint8, f"[pages,{_PAGE_BYTES}]", "inout"),
    )
    return compile_program(
        _Hy4IndexerPoolGroup(g),
        name="hy4_indexer_pool_group",
        operands=operands,
        scalars=(Scalar("groups"),),
        key=(g, "pool_group"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_indexer_score_groups_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("q_rot", torch.bfloat16, "[rows,16,128]"),
        Operand("weights", torch.float32, "[rows,16]", align=4),
        Operand("group_cache", torch.uint8, f"[pages,{_PAGE_BYTES}]"),
        Operand("page_table", torch.int32, "[rows,table_stride]", align=4),
        Operand("group_limits", torch.int32, "[rows]", align=4),
        Operand("scores_out", torch.float32, "[rows,max_groups]", "out", align=4),
    )
    return compile_program(
        _Hy4IndexerScoreGroups(g),
        name="hy4_indexer_score_groups",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("max_groups"), Scalar("table_stride")),
        key=(g, "score_groups"),
        geometry=vars(g),
        doc=__doc__,
    )


@dsl_user_op
def _float_as_uint32(val: Float32, *, loc=None, ip=None) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [Float32(val).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _desc_key(val: Float32) -> Uint32:
    """Order-preserving uint32 key for descending F32 sort (larger score -> smaller key)."""
    u = _float_as_uint32(val)
    if u == Uint32(0x80000000):
        u = Uint32(0)
    abs_u = u & Uint32(0x7FFFFFFF)
    key = Uint32(0xFFFFFFFF)
    if abs_u <= Uint32(0x7F800000):
        if (u & Uint32(0x80000000)) == Uint32(0):
            key = u ^ Uint32(0x7FFFFFFF)
        else:
            key = u
    return key


class _Hy4IndexerTopkExpand:
    """Zero-scratch deterministic GPU radix top-511 group select + physical KV slot expansion.

    For ``pos + 1 <= 2048`` every causal token ``0..=pos`` is emitted directly through
    ``page_table`` without reading ``scores``. For ``pos + 1 > 2048`` an exact 4-pass 8-bit
    radix select finds the top 511 prior complete groups in ``[0, pos // 4)`` (breaking ties
    deterministically by lowest group id) and expands them plus the current group into
    ``slots_out [rows, 2048]`` and ``lengths_out [rows]``.
    """

    def __init__(self, g: Hy4IndexerGeometry):
        self.group_topk = g.group_topk
        self.prior_k = g.group_topk - 1  # 511
        self.candidate_topk = g.candidate_topk
        self.page_rows = g.page_rows

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "hist": cute.struct.Align[cute.struct.MemRange[Int32, 256], 16],
            "excl": cute.struct.Align[cute.struct.MemRange[Int32, 256], 16],
            "warp_tot": cute.struct.Align[cute.struct.MemRange[Int32, 8], 16],
            "out_groups": cute.struct.Align[cute.struct.MemRange[Int32, 512], 16],
            "meta": cute.struct.Align[cute.struct.MemRange[Int32, 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        scores: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        slots_out: cute.Pointer,
        lengths_out: cute.Pointer,
        rows: Int32,
        score_stride: Int32,
        table_stride: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            scores, positions, page_table, slots_out, lengths_out, score_stride, table_stride
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        scores: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        slots_out: cute.Pointer,
        lengths_out: cute.Pointer,
        score_stride: Int32,
        table_stride: Int32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid % Int32(32)

        pos = Int32(positions[row])
        seq_len = pos + Int32(1)
        out_base = row * Int64(self.candidate_topk)
        pt_base = row * Int64(table_stride)

        if seq_len <= Int32(self.candidate_topk):
            idx = tid
            while idx < Int32(self.candidate_topk):
                if idx < seq_len:
                    page = Int64(page_table[pt_base + Int64(idx // Int32(self.page_rows))])
                    slots_out[out_base + Int64(idx)] = Int32(
                        page * Int64(self.page_rows) + Int64(idx % Int32(self.page_rows))
                    )
                else:
                    slots_out[out_base + Int64(idx)] = Int32(-1)
                idx = idx + Int32(256)
            if tid == Int32(0):
                lengths_out[row] = seq_len
        else:
            smem = cutlass_utils.SmemAllocator()
            st = smem.allocate(self._storage())
            hist = st.hist.get_tensor(cute.make_layout((256,)))
            excl = st.excl.get_tensor(cute.make_layout((256,)))
            warp_tot = st.warp_tot.get_tensor(cute.make_layout((8,)))
            out_groups = st.out_groups.get_tensor(cute.make_layout((512,)))
            meta = st.meta.get_tensor(cute.make_layout((4,)))
            hist_addr = shared_ptr_to_u32(hist.iterator)

            cur_group = pos // Int32(4)
            limit = cur_group
            score_base = row * Int64(score_stride)

            prefix = Uint32(0)
            mask = Uint32(0)
            rem = Int32(self.prior_k)

            for pass_idx in cutlass.range_constexpr(4):
                shift = 24 - pass_idx * 8
                hist[tid] = Int32(0)
                cute.arch.sync_threads()
                gid = tid
                while gid < limit:
                    k = _desc_key(Float32(scores[score_base + Int64(gid)]))
                    if (k & mask) == prefix:
                        bucket = Int32((k >> Uint32(shift)) & Uint32(0xFF))
                        red_add_shared_i32(hist_addr + bucket * Int32(4), Int32(1))
                    gid = gid + Int32(256)
                cute.arch.sync_threads()

                v = hist[tid]
                s = v
                for off in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(s, offset=1 << off)
                    if lane >= Int32(1 << off):
                        s = s + up
                if lane == Int32(31):
                    warp_tot[warp] = s
                cute.arch.sync_threads()
                if warp == Int32(0):
                    wt = warp_tot[lane] if lane < Int32(8) else Int32(0)
                    for off in cutlass.range_constexpr(3):
                        up = cute.arch.shuffle_sync_up(wt, offset=1 << off)
                        if lane >= Int32(1 << off):
                            wt = wt + up
                    if lane < Int32(8):
                        warp_tot[lane] = wt
                cute.arch.sync_threads()
                warp_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
                excl_val = warp_off + s - v
                excl[tid] = excl_val
                if excl_val < rem and excl_val + v >= rem:
                    meta[0] = tid
                    meta[1] = rem - excl_val
                cute.arch.sync_threads()
                chosen = Uint32(meta[0])
                prefix = prefix | (chosen << Uint32(shift))
                mask = mask | (Uint32(0xFF) << Uint32(shift))
                rem = meta[1]

            # Compact strictly better groups in logical gid order, then the
            # lowest gids tied at the cutoff. Keep physical COW slot numbering
            # out of the order so the consumer's reduction is repeatable.
            thresh = prefix
            eq_needed = rem
            eq_base = Int32(self.prior_k) - eq_needed
            if tid == Int32(0):
                meta[2] = Int32(0)  # strict_taken: running count of strict-better gids
                meta[3] = Int32(0)  # eq_taken
            cute.arch.sync_threads()

            # Each scan packs two 9-bit counts into one Int32: strict in the
            # low field, ties in the next.  Per warp each field is <= 32 and
            # across the eight warps <= 256, so the fields never carry.
            scan_base = Int32(512)
            base = Int32(0)
            while base < limit:
                gid = base + tid
                is_strict = Int32(0)
                is_eq = Int32(0)
                if gid < limit:
                    k = _desc_key(Float32(scores[score_base + Int64(gid)]))
                    if k < thresh:
                        is_strict = Int32(1)
                    elif k == thresh:
                        is_eq = Int32(1)
                strict_start = meta[2]
                eq_start = meta[3]
                s = is_strict + is_eq * scan_base
                for off in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(s, offset=1 << off)
                    if lane >= Int32(1 << off):
                        s = s + up
                if lane == Int32(31):
                    warp_tot[warp] = s
                cute.arch.sync_threads()
                if warp == Int32(0):
                    wt = warp_tot[lane] if lane < Int32(8) else Int32(0)
                    for off in cutlass.range_constexpr(3):
                        up = cute.arch.shuffle_sync_up(wt, offset=1 << off)
                        if lane >= Int32(1 << off):
                            wt = wt + up
                    if lane < Int32(8):
                        warp_tot[lane] = wt
                cute.arch.sync_threads()
                packed_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
                strict_off = packed_off % scan_base
                eq_off = (packed_off // scan_base) % scan_base
                if is_strict == Int32(1):
                    local_strict = strict_off + (s % scan_base) - Int32(1)
                    out_groups[strict_start + local_strict] = gid
                if is_eq == Int32(1):
                    local_eq = eq_off + ((s // scan_base) % scan_base) - Int32(1)
                    rank_eq = eq_start + local_eq
                    if rank_eq < eq_needed:
                        out_groups[eq_base + rank_eq] = gid
                if tid == Int32(255):
                    meta[2] = strict_start + warp_tot[7] % scan_base
                    meta[3] = eq_start + (warp_tot[7] // scan_base) % scan_base
                cute.arch.sync_threads()
                base = base + Int32(256)

            # Expand top-511 prior groups + current group into physical KV slots.
            cur_valid = (pos % Int32(4)) + Int32(1)
            prior_tokens = Int32(self.prior_k * 4)
            total_valid = prior_tokens + cur_valid
            idx = tid
            while idx < Int32(self.candidate_topk):
                if idx < prior_tokens:
                    g_i = idx // Int32(4)
                    sub = idx % Int32(4)
                    gid = out_groups[g_i]
                    tok = gid * Int32(4) + sub
                    page = Int64(page_table[pt_base + Int64(tok // Int32(self.page_rows))])
                    slots_out[out_base + Int64(idx)] = Int32(
                        page * Int64(self.page_rows) + Int64(tok % Int32(self.page_rows))
                    )
                elif idx < total_valid:
                    sub = idx - prior_tokens
                    tok = cur_group * Int32(4) + sub
                    page = Int64(page_table[pt_base + Int64(tok // Int32(self.page_rows))])
                    slots_out[out_base + Int64(idx)] = Int32(
                        page * Int64(self.page_rows) + Int64(tok % Int32(self.page_rows))
                    )
                else:
                    slots_out[out_base + Int64(idx)] = Int32(-1)
                idx = idx + Int32(256)
            if tid == Int32(0):
                lengths_out[row] = total_valid


@cute.jit
def _norm_score(val: Float32) -> Float32:
    """Canonicalize -0.0 to +0.0 so score outputs and comparisons stay deterministic."""
    return Float32(0.0) if val == Float32(0.0) else val


class _Hy4IndexerScoredTopk:
    """Per-shard deterministic top-511 scored candidate selector in global logical group coordinates.

    For ``pos + 1 <= 2048`` or ``local_limits[row] <= 0`` emits ``(-inf, -1)`` without
    reading ``scores``. Otherwise selects up to 511 local groups (radix select when
    ``limit > 511``), maps each local group id ``lg`` to its global logical group id
    ``((lg // 16) * page_stride + page_offset) * 16 + (lg % 16)``, pads slot 511 (and any
    missing slots) with ``(-inf, -1)``, and sorts all 512 pairs in total
    ``(score desc, global gid asc)`` order for bitonic merge across ranks.
    """

    def __init__(self, g: Hy4IndexerGeometry):
        self.group_topk = g.group_topk      # 512
        self.prior_k = g.group_topk - 1     # 511
        self.candidate_topk = g.candidate_topk

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "hist": cute.struct.Align[cute.struct.MemRange[Int32, 256], 16],
            "excl": cute.struct.Align[cute.struct.MemRange[Int32, 256], 16],
            "warp_tot": cute.struct.Align[cute.struct.MemRange[Int32, 8], 16],
            "sv": cute.struct.Align[cute.struct.MemRange[Float32, 512], 16],
            "si": cute.struct.Align[cute.struct.MemRange[Int32, 512], 16],
            "meta": cute.struct.Align[cute.struct.MemRange[Int32, 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        scores: cute.Pointer,
        positions: cute.Pointer,
        local_limits: cute.Pointer,
        out_scores: cute.Pointer,
        out_indices: cute.Pointer,
        rows: Int32,
        score_stride: Int32,
        page_stride: Int32,
        page_offset: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            scores, positions, local_limits, out_scores, out_indices,
            score_stride, page_stride, page_offset,
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        scores: cute.Pointer,
        positions: cute.Pointer,
        local_limits: cute.Pointer,
        out_scores: cute.Pointer,
        out_indices: cute.Pointer,
        score_stride: Int32,
        page_stride: Int32,
        page_offset: Int32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid % Int32(32)

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage())
        hist = st.hist.get_tensor(cute.make_layout((256,)))
        excl = st.excl.get_tensor(cute.make_layout((256,)))
        warp_tot = st.warp_tot.get_tensor(cute.make_layout((8,)))
        sv = st.sv.get_tensor(cute.make_layout((512,)))
        si = st.si.get_tensor(cute.make_layout((512,)))
        meta = st.meta.get_tensor(cute.make_layout((4,)))
        hist_addr = shared_ptr_to_u32(hist.iterator)

        pos = Int32(positions[row])
        seq_len = pos + Int32(1)
        limit = Int32(local_limits[row])
        out_base = row * Int64(self.group_topk)
        score_base = row * Int64(score_stride)

        if seq_len <= Int32(self.candidate_topk) or limit <= Int32(0):
            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                out_scores[out_base + Int64(i)] = Float32(float("-inf"))
                out_indices[out_base + Int64(i)] = Int32(-1)
        elif limit <= Int32(self.prior_k):
            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                if i < limit:
                    val = _norm_score(Float32(scores[score_base + Int64(i)]))
                    gid = ((i // Int32(_GROUPS_PER_PAGE)) * page_stride + page_offset) * Int32(_GROUPS_PER_PAGE) + (
                        i % Int32(_GROUPS_PER_PAGE)
                    )
                    sv[i] = val
                    si[i] = gid
                else:
                    sv[i] = Float32(float("-inf"))
                    si[i] = Int32(-1)
            cute.arch.sync_threads()
            _sort_pairs(sv, si, 512, True)
            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                out_scores[out_base + Int64(i)] = sv[i]
                out_indices[out_base + Int64(i)] = si[i]
        else:
            prefix = Uint32(0)
            mask = Uint32(0)
            rem = Int32(self.prior_k)

            for pass_idx in cutlass.range_constexpr(4):
                shift = 24 - pass_idx * 8
                hist[tid] = Int32(0)
                cute.arch.sync_threads()
                lg = tid
                while lg < limit:
                    k = _desc_key(Float32(scores[score_base + Int64(lg)]))
                    if (k & mask) == prefix:
                        bucket = Int32((k >> Uint32(shift)) & Uint32(0xFF))
                        red_add_shared_i32(hist_addr + bucket * Int32(4), Int32(1))
                    lg = lg + Int32(256)
                cute.arch.sync_threads()

                v = hist[tid]
                s = v
                for off in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(s, offset=1 << off)
                    if lane >= Int32(1 << off):
                        s = s + up
                if lane == Int32(31):
                    warp_tot[warp] = s
                cute.arch.sync_threads()
                if warp == Int32(0):
                    wt = warp_tot[lane] if lane < Int32(8) else Int32(0)
                    for off in cutlass.range_constexpr(3):
                        up = cute.arch.shuffle_sync_up(wt, offset=1 << off)
                        if lane >= Int32(1 << off):
                            wt = wt + up
                    if lane < Int32(8):
                        warp_tot[lane] = wt
                cute.arch.sync_threads()
                warp_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
                excl_val = warp_off + s - v
                excl[tid] = excl_val
                if excl_val < rem and excl_val + v >= rem:
                    meta[0] = tid
                    meta[1] = rem - excl_val
                cute.arch.sync_threads()
                chosen = Uint32(meta[0])
                prefix = prefix | (chosen << Uint32(shift))
                mask = mask | (Uint32(0xFF) << Uint32(shift))
                rem = meta[1]

            thresh = prefix
            eq_needed = rem
            eq_base = Int32(self.prior_k) - eq_needed
            if tid == Int32(0):
                meta[2] = Int32(0)
                meta[3] = Int32(0)
                sv[511] = Float32(float("-inf"))
                si[511] = Int32(-1)
            cute.arch.sync_threads()

            scan_base = Int32(512)
            base = Int32(0)
            while base < limit:
                lg = base + tid
                is_strict = Int32(0)
                is_eq = Int32(0)
                val = Float32(float("-inf"))
                if lg < limit:
                    val = _norm_score(Float32(scores[score_base + Int64(lg)]))
                    k = _desc_key(val)
                    if k < thresh:
                        is_strict = Int32(1)
                    elif k == thresh:
                        is_eq = Int32(1)
                strict_start = meta[2]
                eq_start = meta[3]
                s = is_strict + is_eq * scan_base
                for off in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(s, offset=1 << off)
                    if lane >= Int32(1 << off):
                        s = s + up
                if lane == Int32(31):
                    warp_tot[warp] = s
                cute.arch.sync_threads()
                if warp == Int32(0):
                    wt = warp_tot[lane] if lane < Int32(8) else Int32(0)
                    for off in cutlass.range_constexpr(3):
                        up = cute.arch.shuffle_sync_up(wt, offset=1 << off)
                        if lane >= Int32(1 << off):
                            wt = wt + up
                    if lane < Int32(8):
                        warp_tot[lane] = wt
                cute.arch.sync_threads()
                packed_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
                strict_off = packed_off % scan_base
                eq_off = (packed_off // scan_base) % scan_base
                if is_strict == Int32(1):
                    local_strict = strict_off + (s % scan_base) - Int32(1)
                    dst = strict_start + local_strict
                    gid = ((lg // Int32(_GROUPS_PER_PAGE)) * page_stride + page_offset) * Int32(_GROUPS_PER_PAGE) + (
                        lg % Int32(_GROUPS_PER_PAGE)
                    )
                    sv[dst] = val
                    si[dst] = gid
                if is_eq == Int32(1):
                    local_eq = eq_off + ((s // scan_base) % scan_base) - Int32(1)
                    rank_eq = eq_start + local_eq
                    if rank_eq < eq_needed:
                        dst = eq_base + rank_eq
                        gid = ((lg // Int32(_GROUPS_PER_PAGE)) * page_stride + page_offset) * Int32(
                            _GROUPS_PER_PAGE
                        ) + (lg % Int32(_GROUPS_PER_PAGE))
                        sv[dst] = val
                        si[dst] = gid
                if tid == Int32(255):
                    meta[2] = strict_start + warp_tot[7] % scan_base
                    meta[3] = eq_start + (warp_tot[7] // scan_base) % scan_base
                cute.arch.sync_threads()
                base = base + Int32(256)

            _sort_pairs(sv, si, 512, True)
            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                out_scores[out_base + Int64(i)] = sv[i]
                out_indices[out_base + Int64(i)] = si[i]


class _Hy4IndexerMergeExpandOwned:
    """Bitonic-merge two shards' top-511 scored group candidates and expand owned physical slots.

    For ``pos + 1 <= 2048`` never reads ``scores``/``indices``; emits every causal token
    in ``[0, pos]`` whose logical page parity matches ``rank`` into compact ``local_slots``
    and ``local_lengths``. For ``pos + 1 > 2048`` merges GPU0 ``[0..512)`` and GPU1
    ``[512..1024)`` into the exact global top-511 prior groups, filters to groups owned by
    ``rank`` in ascending logical group order, and expands them plus ``cur_group = pos // 4``
    (if owned) into compact ``local_slots`` + ``local_lengths``.

    When ``write_draft != 0``, also populates sparse global-position ``draft_slots [rows, 2048]``
    with this rank's physical token slots at their exact global recall positions and ``-1`` for
    peer-owned tokens, enabling zero-sync dual-rank DFly2 recall gather.
    """

    def __init__(self, g: Hy4IndexerGeometry):
        self.group_topk = g.group_topk      # 512
        self.prior_k = g.group_topk - 1     # 511
        self.candidate_topk = g.candidate_topk
        self.page_rows = g.page_rows        # 64

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "sv": cute.struct.Align[cute.struct.MemRange[Float32, 512], 16],
            "si": cute.struct.Align[cute.struct.MemRange[Int32, 512], 16],
            "owned": cute.struct.Align[cute.struct.MemRange[Int32, 512], 16],
            "meta": cute.struct.Align[cute.struct.MemRange[Int32, 2], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        scores: cute.Pointer,
        indices: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        local_slots: cute.Pointer,
        local_lengths: cute.Pointer,
        draft_slots: cute.Pointer,
        rows: Int32,
        table_stride: Int32,
        rank: Int32,
        write_draft: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            scores, indices, positions, page_table, local_slots, local_lengths, draft_slots,
            table_stride, rank, write_draft,
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        scores: cute.Pointer,
        indices: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        local_slots: cute.Pointer,
        local_lengths: cute.Pointer,
        draft_slots: cute.Pointer,
        table_stride: Int32,
        rank: Int32,
        write_draft: Int32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage())
        sv = st.sv.get_tensor(cute.make_layout((512,)))
        si = st.si.get_tensor(cute.make_layout((512,)))
        owned = st.owned.get_tensor(cute.make_layout((512,)))
        meta = st.meta.get_tensor(cute.make_layout((2,)))

        pos = Int32(positions[row])
        seq_len = pos + Int32(1)
        out_base = row * Int64(self.candidate_topk)
        pt_base = row * Int64(table_stride)

        if seq_len <= Int32(self.candidate_topk):
            full_pairs = seq_len // Int32(128)
            pair_rem = seq_len % Int32(128)
            extra = Int32(0)
            if rank == Int32(0):
                extra = pair_rem if pair_rem < Int32(64) else Int32(64)
            else:
                extra = (pair_rem - Int32(64)) if pair_rem > Int32(64) else Int32(0)
            owned_len = full_pairs * Int32(64) + extra

            idx = tid
            while idx < Int32(self.candidate_topk):
                if idx < owned_len:
                    local_page = idx // Int32(64)
                    off = idx % Int32(64)
                    phys_page = Int64(page_table[pt_base + Int64(local_page)])
                    local_slots[out_base + Int64(idx)] = Int32(phys_page * Int64(64) + Int64(off))
                else:
                    local_slots[out_base + Int64(idx)] = Int32(-1)
                idx = idx + Int32(256)
            if tid == Int32(0):
                local_lengths[row] = owned_len

            if write_draft != Int32(0):
                idx = tid
                while idx < Int32(self.candidate_topk):
                    slot_val = Int32(-1)
                    if idx < seq_len:
                        logical_page = idx // Int32(64)
                        if logical_page % Int32(2) == rank:
                            phys_page = Int64(page_table[pt_base + Int64(logical_page // Int32(2))])
                            slot_val = Int32(phys_page * Int64(64) + Int64(idx % Int32(64)))
                    draft_slots[out_base + Int64(idx)] = slot_val
                    idx = idx + Int32(256)
        else:
            # Bitonic compare-exchange: A[0..512) + reverse(B[512..1024)) -> exact global top-512.
            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                a = row * Int64(1024) + Int64(i)
                b = row * Int64(1024) + Int64(1023) - Int64(i)
                ai, av = Int32(indices[a]), Float32(scores[a])
                bi, bv = Int32(indices[b]), Float32(scores[b])
                si[i], sv[i] = bi, bv
                if _pair_before(av, ai, bv, bi):
                    si[i], sv[i] = ai, av
            cute.arch.sync_threads()
            _sort_pairs(sv, si, 512, True, True)

            # Filter exact global top-511 prior groups (indices 0..510) to this rank's parity.
            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                gid = Int32(si[i]) if i < Int32(self.prior_k) else Int32(-1)
                if gid >= Int32(0) and ((gid // Int32(_GROUPS_PER_PAGE)) % Int32(2) != rank):
                    gid = Int32(-1)
                owned[i] = gid
            if tid == Int32(0):
                meta[0] = Int32(0)
            cute.arch.sync_threads()
            _sort_pairs(sv, owned, 512, False)

            for slot in cutlass.range_constexpr(2):
                i = tid + Int32(slot * 256)
                g_cur = Int32(owned[i])
                if g_cur >= Int32(0):
                    if i == Int32(511):
                        meta[0] = Int32(512)
                    elif Int32(owned[i + Int32(1)]) < Int32(0):
                        meta[0] = i + Int32(1)
            cute.arch.sync_threads()
            owned_groups = Int32(meta[0])

            cur_group = pos // Int32(4)
            cur_valid_all = (pos % Int32(4)) + Int32(1)
            cur_owned = Int32(1) if ((cur_group // Int32(_GROUPS_PER_PAGE)) % Int32(2) == rank) else Int32(0)
            cur_valid = cur_valid_all if cur_owned != Int32(0) else Int32(0)
            prior_tokens = owned_groups * Int32(4)
            total_local = prior_tokens + cur_valid

            idx = tid
            while idx < Int32(self.candidate_topk):
                if idx < prior_tokens:
                    g_i = idx // Int32(4)
                    sub = idx % Int32(4)
                    gid = Int32(owned[g_i])
                    local_page = gid // Int32(32)
                    tok_in_page = (gid % Int32(_GROUPS_PER_PAGE)) * Int32(4) + sub
                    phys_page = Int64(page_table[pt_base + Int64(local_page)])
                    local_slots[out_base + Int64(idx)] = Int32(phys_page * Int64(64) + Int64(tok_in_page))
                elif idx < total_local:
                    sub = idx - prior_tokens
                    local_page = cur_group // Int32(32)
                    tok_in_page = (cur_group % Int32(_GROUPS_PER_PAGE)) * Int32(4) + sub
                    phys_page = Int64(page_table[pt_base + Int64(local_page)])
                    local_slots[out_base + Int64(idx)] = Int32(phys_page * Int64(64) + Int64(tok_in_page))
                else:
                    local_slots[out_base + Int64(idx)] = Int32(-1)
                idx = idx + Int32(256)
            if tid == Int32(0):
                local_lengths[row] = total_local

            if write_draft != Int32(0):
                prior_all = Int32(self.prior_k * 4)
                total_all = prior_all + cur_valid_all
                idx = tid
                while idx < Int32(self.candidate_topk):
                    slot_val = Int32(-1)
                    if idx < prior_all:
                        g_i = idx // Int32(4)
                        sub = idx % Int32(4)
                        gid = Int32(si[g_i])
                        if gid >= Int32(0) and ((gid // Int32(_GROUPS_PER_PAGE)) % Int32(2) == rank):
                            local_page = gid // Int32(32)
                            tok_in_page = (gid % Int32(_GROUPS_PER_PAGE)) * Int32(4) + sub
                            phys_page = Int64(page_table[pt_base + Int64(local_page)])
                            slot_val = Int32(phys_page * Int64(64) + Int64(tok_in_page))
                    elif idx < total_all:
                        sub = idx - prior_all
                        if cur_owned != Int32(0):
                            local_page = cur_group // Int32(32)
                            tok_in_page = (cur_group % Int32(_GROUPS_PER_PAGE)) * Int32(4) + sub
                            phys_page = Int64(page_table[pt_base + Int64(local_page)])
                            slot_val = Int32(phys_page * Int64(64) + Int64(tok_in_page))
                    draft_slots[out_base + Int64(idx)] = slot_val
                    idx = idx + Int32(256)


def compile_hy4_indexer_scored_topk_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("scores", torch.float32, "[rows,score_stride]", align=4),
        Operand("positions", torch.int32, "[rows]", align=4),
        Operand("local_limits", torch.int32, "[rows]", align=4),
        Operand("out_scores", torch.float32, f"[rows,{g.group_topk}]", "out", align=4),
        Operand("out_indices", torch.int32, f"[rows,{g.group_topk}]", "out", align=4),
    )
    return compile_program(
        _Hy4IndexerScoredTopk(g),
        name="hy4_indexer_scored_topk",
        operands=operands,
        scalars=(
            Scalar("rows"),
            Scalar("score_stride"),
            Scalar("page_stride"),
            Scalar("page_offset"),
        ),
        key=(g, "scored_topk"),
        geometry={**vars(g), "prior_k": g.group_topk - 1},
        doc=_Hy4IndexerScoredTopk.__doc__,
    )


def compile_hy4_indexer_merge_expand_owned_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("scores", torch.float32, f"[rows,{2 * g.group_topk}]", align=4),
        Operand("indices", torch.int32, f"[rows,{2 * g.group_topk}]", align=4),
        Operand("positions", torch.int32, "[rows]", align=4),
        Operand("page_table", torch.int32, "[rows,table_stride]", align=4),
        Operand("local_slots", torch.int32, f"[rows,{g.candidate_topk}]", "out", align=4),
        Operand("local_lengths", torch.int32, "[rows]", "out", align=4),
        Operand("draft_slots", torch.int32, f"[rows,{g.candidate_topk}]", "out", align=4),
    )
    return compile_program(
        _Hy4IndexerMergeExpandOwned(g),
        name="hy4_indexer_merge_expand_owned",
        operands=operands,
        scalars=(
            Scalar("rows"),
            Scalar("table_stride"),
            Scalar("rank"),
            Scalar("write_draft"),
        ),
        key=(g, "merge_expand_owned"),
        geometry={**vars(g), "prior_k": g.group_topk - 1},
        doc=_Hy4IndexerMergeExpandOwned.__doc__,
    )


def compile_hy4_indexer_topk_expand_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("scores", torch.float32, "[rows,score_stride]", align=4),
        Operand("positions", torch.int32, "[rows]", align=4),
        Operand("page_table", torch.int32, "[rows,table_stride]", align=4),
        Operand("slots_out", torch.int32, f"[rows,{g.candidate_topk}]", "out", align=4),
        Operand("lengths_out", torch.int32, "[rows]", "out", align=4),
    )
    return compile_program(
        _Hy4IndexerTopkExpand(g),
        name="hy4_indexer_topk_expand",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("score_stride"), Scalar("table_stride")),
        key=(g, "topk_expand"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_indexer_expand_slots_aot(g: Hy4IndexerGeometry = HY4_FLASH_INDEXER):
    operands = (
        Operand("recalled_groups", torch.int32, "[rows,num_recalled]", align=4),
        Operand("positions", torch.int32, "[rows]", align=4),
        Operand("page_table", torch.int32, "[rows,table_stride]", align=4),
        Operand("slots_out", torch.int32, f"[rows,{g.candidate_topk}]", "out", align=4),
        Operand("lengths_out", torch.int32, "[rows]", "out", align=4),
    )
    return compile_program(
        _Hy4IndexerExpandSlots(g),
        name="hy4_indexer_expand_slots",
        operands=operands,
        scalars=(Scalar("rows"), Scalar("num_recalled"), Scalar("table_stride")),
        key=(g, "expand_slots"),
        geometry=vars(g),
        doc=__doc__,
    )
