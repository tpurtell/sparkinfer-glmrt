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
    atomic_add_shared_i32,
    cp_async_bulk_g2s_mbar,
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    ld_global_nc_f32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_i32,
    ld_shared_u32,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    pack_f32x2_to_bfloat2,
    red_add_shared_i32,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_v2_f32,
    st_global_v2_u32,
    st_global_v4_u32,
    st_shared_u32,
    st_shared_v4_u32,
    u32_as_f32,
)
from b12x._lib.utils import runtime_num_sm
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _fp8_scale
from .context_split import _pair_before, _sort_pairs
from .hy4_mla import _mrope_position, _validate_mrope

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


class _Hy4IndexerQueryRope:
    """Interleaved tail-64 RoPE on [rows, 16, 128] + weight scaling [rows, 16].

    One 256-thread CTA per row computes cos/sin once in shared memory (instead
    of 16x per row) and copies/rotates all 16 heads with coalesced 64-bit ops.
    """

    def __init__(self, g: Hy4IndexerGeometry):
        self.heads = g.heads
        self.weight_scale = float(g.weight_scale)
        self.mrope_section = g.mrope_section

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "cos_s": cute.struct.Align[cute.struct.MemRange[Float32, 32], 16],
            "sin_s": cute.struct.Align[cute.struct.MemRange[Float32, 32], 16],
        }
        return cute.struct(Storage)

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
        if rows > Int32(0):
            self.kernel(q_raw, w_raw, positions, inv_freq, q_out, w_out).launch(
                grid=(rows, 1, 1), block=(256, 1, 1), stream=stream
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
        tid = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage())
        cos_s = st.cos_s.get_tensor(cute.make_layout((32,)))
        sin_s = st.sin_s.get_tensor(cute.make_layout((32,)))

        if tid < Int32(32):
            pair = Int64(tid)
            angle = _mrope_position(positions, row, pair, self.mrope_section[1], self.mrope_section[2]) * Float32(inv_freq[pair])
            cos_s[tid] = cute.math.cos(angle, fastmath=False)
            sin_s[tid] = cute.math.sin(angle, fastmath=False)
        if tid < Int32(self.heads):
            rh = row * Int64(self.heads) + Int64(tid)
            w_out[rh] = Float32(w_raw[rh]) * Float32(self.weight_scale)
        cute.arch.sync_threads()

        # 16 heads x 64 pass-through BF16 = 1024 BF16 = 256 x 4 BF16 (8B each).
        # 16 heads x 32 RoPE pairs = 512 pairs = 256 x 2 pairs (8B each).
        head = Int64(tid >> Int32(4))        # 0..15
        sub = Int64(tid & Int32(15))         # 0..15 (4 BF16 elems = 2 pairs)
        row_q_in = Int64(q_raw.toint()) + (row * Int64(self.heads) + head) * Int64(256)
        row_q_out = Int64(q_out.toint()) + (row * Int64(self.heads) + head) * Int64(256)

        # Leading 64 dims: 4 BF16 per thread via 8B load/store.
        p0, p1 = ld_global_nc_v2_u32(row_q_in + sub * Int64(8))
        st_global_v2_u32(row_q_out + sub * Int64(8), p0, p1)

        # Tail 64 dims: 2 interleaved pairs per thread via 8B load/store.
        r0, r1 = ld_global_nc_v2_u32(row_q_in + Int64(128) + sub * Int64(8))
        pair0 = Int32(sub * Int64(2))
        pair1 = pair0 + Int32(1)
        c0 = cos_s[pair0]
        s0 = sin_s[pair0]
        c1 = cos_s[pair1]
        s1 = sin_s[pair1]
        x0 = _bf16_lo(r0)
        x1 = _bf16_hi(r0)
        x2 = _bf16_lo(r1)
        x3 = _bf16_hi(r1)
        o0 = pack_f32x2_to_bfloat2(x0 * c0 - x1 * s0, x1 * c0 + x0 * s0)
        o1 = pack_f32x2_to_bfloat2(x2 * c1 - x3 * s1, x3 * c1 + x2 * s1)
        st_global_v2_u32(row_q_out + Int64(128) + sub * Int64(8), o0, o1)


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


@cute.jit
def _load_q_permuted_row(
    q_base: Int64,
    w_base: Int64,
    row: Int64,
    gid: Int32,
    ltid: Int32,
    q_regs: cute.Tensor,
    q_off: cutlass.Constexpr,
):
    """Load one row's 16x128 BF16 Q in bank-conflict-free MMA order (dim = 32*ltid + 4*ks)."""
    row_q = q_base + row * Int64(16 * 128 * 2)
    lo_head = row_q + Int64(gid) * Int64(256) + Int64(ltid) * Int64(64)
    hi_head = row_q + Int64(gid + Int32(8)) * Int64(256) + Int64(ltid) * Int64(64)
    for seg in cutlass.range_constexpr(4):
        u0, u1, u2, u3 = ld_global_nc_v4_u32(lo_head + Int64(seg * 16))
        v0, v1, v2, v3 = ld_global_nc_v4_u32(hi_head + Int64(seg * 16))
        ks0: cutlass.Constexpr = seg * 2
        ks1: cutlass.Constexpr = seg * 2 + 1
        q_regs[q_off + ks0 * 4 + 0] = u0
        q_regs[q_off + ks0 * 4 + 1] = v0
        q_regs[q_off + ks0 * 4 + 2] = u1
        q_regs[q_off + ks0 * 4 + 3] = v1
        q_regs[q_off + ks1 * 4 + 0] = u2
        q_regs[q_off + ks1 * 4 + 1] = v2
        q_regs[q_off + ks1 * 4 + 2] = u3
        q_regs[q_off + ks1 * 4 + 3] = v3
    row_w = w_base + row * Int64(16 * 4)
    w0 = ld_global_nc_f32(row_w + Int64(gid) * Int64(4))
    w1 = ld_global_nc_f32(row_w + Int64(gid + Int32(8)) * Int64(4))
    return w0, w1


@cute.jit
def _stage_group_page(
    cache_base: Int64,
    phys_page: Int64,
    smem_dst: Int32,
    tid: Int32,
):
    """Coalesced 16B copy of one 2112B (132 x 16B) indexer group page into shared memory."""
    src = cache_base + phys_page * Int64(_PAGE_BYTES)
    v0, v1, v2, v3 = ld_global_nc_v4_u32(src + Int64(tid) * Int64(16))
    st_shared_v4_u32(smem_dst + tid * Int32(16), v0, v1, v2, v3)
    if tid < Int32(4):
        idx = Int32(128) + tid
        u0, u1, u2, u3 = ld_global_nc_v4_u32(src + Int64(idx) * Int64(16))
        st_shared_v4_u32(smem_dst + idx * Int32(16), u0, u1, u2, u3)


@cute.jit
def _write_score_pair(
    scores_base: Int64,
    row: Int64,
    max_groups: Int32,
    limit: Int32,
    g0: Int32,
    r0: Float32,
    r1: Float32,
):
    g1 = g0 + Int32(1)
    if g0 < max_groups:
        v0 = r0 if g0 < limit else Float32(-Float32.inf)
        addr0 = scores_base + (row * Int64(max_groups) + Int64(g0)) * Int64(4)
        if g1 < max_groups:
            v1 = r1 if g1 < limit else Float32(-Float32.inf)
            if (addr0 & Int64(7)) == Int64(0):
                st_global_v2_f32(addr0, v0, v1)
            else:
                st_global_f32(addr0, v0)
                st_global_f32(addr0 + Int64(4), v1)
        else:
            st_global_f32(addr0, v0)


class _Hy4IndexerScoreGroups:
    """Double-buffered async Tensor-Core (m16n8k16 BF16) group scorer.

    - ``rows == 1``: 4 math warps + 1 IO warp (160 threads), register-resident Q,
      double-buffered 2112B ``cp.async.bulk`` page gather.
    - ``rows >= 2``: 4 math warps + 1 IO warp (160 threads), SMEM-resident permuted Q
      for up to 8 rows (43.9 KB total SMEM -> 2 CTAs/SM, ~64 regs/thread), double-buffered
      ``cp.async.bulk`` page gather, 8x shared-page K reuse via ``ldmatrix_m8n8x4_b16``,
      and exact per-row fallback when page tables diverge.
    """

    q_smem_stride = 136  # 128 + 8 BF16 pad (272 B -> 68 % 32 == 4 banks, 0 conflicts on ldmatrix)
    q_smem_row_bytes = 272
    q_smem_token_bytes = 16 * 272  # 4,352 B per row (34,816 B for 8 rows)

    def __init__(self, g: Hy4IndexerGeometry):
        self.heads = g.heads

    def _storage_1row(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "pages": cute.struct.Align[cute.struct.MemRange[Uint8, 2 * 2 * _PAGE_BYTES], 128],
            "mbar": cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 4], 16],
        }
        return cute.struct(Storage)

    def _storage_8row(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "q_s": cute.struct.Align[cute.struct.MemRange[BFloat16, 8 * 16 * self.q_smem_stride], 128],
            "pages": cute.struct.Align[cute.struct.MemRange[Uint8, 2 * 2 * _PAGE_BYTES], 128],
            "w_s": cute.struct.Align[cute.struct.MemRange[Float32, 8 * 16], 16],
            "lims_s": cute.struct.Align[cute.struct.MemRange[Int32, 8], 16],
            "stage_shared": cute.struct.Align[cute.struct.MemRange[Int32, 2], 16],
            "mbar": cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 4], 16],
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
        if rows > Int32(0) and max_groups > Int32(0):
            sm_count = runtime_num_sm()
            total_pages = (max_groups + Int32(_GROUPS_PER_PAGE - 1)) // Int32(_GROUPS_PER_PAGE)
            total_pairs = (total_pages + Int32(1)) // Int32(2)
            if rows == Int32(1):
                target_ctas = sm_count * Int32(4)
                grid_y = total_pairs
                if grid_y > target_ctas:
                    grid_y = target_ctas
                self.kernel_1row(
                    q_rot, weights, group_cache, page_table, group_limits, scores_out,
                    max_groups, table_stride, total_pages, total_pairs,
                ).launch(grid=(Int32(1), grid_y, 1), block=(160, 1, 1), stream=stream)
            else:
                row_tiles = (rows + Int32(7)) // Int32(8)
                target_ctas = sm_count * Int32(2)
                grid_y = (target_ctas + row_tiles - Int32(1)) // row_tiles
                if grid_y > total_pairs:
                    grid_y = total_pairs
                if grid_y < Int32(1):
                    grid_y = Int32(1)
                self.kernel_8row(
                    q_rot, weights, group_cache, page_table, group_limits, scores_out,
                    rows, max_groups, table_stride, total_pages, total_pairs,
                ).launch(grid=(row_tiles, grid_y, 1), block=(160, 1, 1), stream=stream)

    @cute.kernel
    def kernel_1row(
        self,
        q_rot: cute.Pointer,
        weights: cute.Pointer,
        group_cache: cute.Pointer,
        page_table: cute.Pointer,
        group_limits: cute.Pointer,
        scores_out: cute.Pointer,
        max_groups: Int32,
        table_stride: Int32,
        total_pages: Int32,
        total_pairs: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        grid_y = Int32(cute.arch.grid_dim()[1])

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage_1row())
        pages_base = shared_ptr_to_u32(st.pages.data_ptr())
        mbar_base = st.mbar.data_ptr()

        if tid == Int32(0):
            cute.arch.mbarrier_init(mbar_base + Int32(0), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(1), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(2), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(3), Int32(1))
        cute.arch.barrier()

        limit = Int32(group_limits[0])
        stage_bytes = Int32(2 * _PAGE_BYTES)

        if warp == Int32(4):
            cache_base = Int64(group_cache.toint())
            prod_idx = Int32(0)
            prod_phase = Int32(1)
            pair_idx = Int32(cute.arch.block_idx()[1])
            while pair_idx < total_pairs:
                cute.arch.mbarrier_wait(mbar_base + Int32(2) + prod_idx, phase=prod_phase)
                if lane == Int32(0):
                    p0 = pair_idx * Int32(2)
                    p1 = p0 + Int32(1)
                    has0 = Int32(1) if (p0 < total_pages and p0 * Int32(_GROUPS_PER_PAGE) < limit) else Int32(0)
                    has1 = Int32(1) if (p1 < total_pages and p1 * Int32(_GROUPS_PER_PAGE) < limit) else Int32(0)
                    full_ptr = mbar_base + prod_idx
                    full_addr = shared_ptr_to_u32(full_ptr)
                    cute.arch.mbarrier_arrive_and_expect_tx(full_ptr, (has0 + has1) * Int32(_PAGE_BYTES))
                    dst_base = pages_base + prod_idx * stage_bytes
                    if has0 != Int32(0):
                        phys0 = Int64(page_table[Int64(p0)])
                        cp_async_bulk_g2s_mbar(dst_base, cache_base + phys0 * Int64(_PAGE_BYTES), Int32(_PAGE_BYTES), full_addr)
                    if has1 != Int32(0):
                        phys1 = Int64(page_table[Int64(p1)])
                        cp_async_bulk_g2s_mbar(dst_base + Int32(_PAGE_BYTES), cache_base + phys1 * Int64(_PAGE_BYTES), Int32(_PAGE_BYTES), full_addr)
                prod_idx = prod_idx ^ Int32(1)
                if prod_idx == Int32(0):
                    prod_phase = prod_phase ^ Int32(1)
                pair_idx = pair_idx + grid_y
        else:
            gid = lane >> Int32(2)
            ltid = lane & Int32(3)
            p_sub = warp >> Int32(1)
            half = warp & Int32(1)

            q_regs = cute.make_rmem_tensor((32,), Uint32)
            w0, w1 = _load_q_permuted_row(
                Int64(q_rot.toint()), Int64(weights.toint()), Int64(0), gid, ltid, q_regs, 0
            )
            scores_base = Int64(scores_out.toint())
            cons_idx = Int32(0)
            cons_phase = Int32(0)
            pair_idx = Int32(cute.arch.block_idx()[1])
            while pair_idx < total_pairs:
                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                pages_addr = pages_base + cons_idx * stage_bytes
                p_idx = pair_idx * Int32(2) + p_sub
                if p_idx < total_pages:
                    g_base = p_idx * Int32(_GROUPS_PER_PAGE) + half * Int32(8)
                    if g_base >= limit:
                        if gid == Int32(0):
                            _write_score_pair(
                                scores_base, Int64(0), max_groups, limit,
                                g_base + ltid * Int32(2), Float32(-Float32.inf), Float32(-Float32.inf),
                            )
                    else:
                        rec_addr = (
                            pages_addr
                            + p_sub * Int32(_PAGE_BYTES)
                            + (half * Int32(8) + gid) * Int32(_GROUP_RECORD_BYTES)
                        )
                        scale = ld_shared_f32(rec_addr + Int32(128))
                        d0 = Float32(0.0)
                        d1 = Float32(0.0)
                        d2 = Float32(0.0)
                        d3 = Float32(0.0)
                        k_lane_off = ltid * Int32(32)
                        for ks in cutlass.range_constexpr(8):
                            word = ld_shared_u32(rec_addr + k_lane_off + Int32(ks * 4))
                            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(word)
                            d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                                d0, d1, d2, d3,
                                q_regs[ks * 4 + 0], q_regs[ks * 4 + 1],
                                q_regs[ks * 4 + 2], q_regs[ks * 4 + 3],
                                b0, b1,
                            )
                        s0 = cute.arch.shuffle_sync(scale, ltid * Int32(8))
                        s1 = cute.arch.shuffle_sync(scale, ltid * Int32(8) + Int32(4))
                        r0 = (
                            fmax_f32(d0 * s0, Float32(0.0)) * w0
                            + fmax_f32(d2 * s0, Float32(0.0)) * w1
                        )
                        r1 = (
                            fmax_f32(d1 * s1, Float32(0.0)) * w0
                            + fmax_f32(d3 * s1, Float32(0.0)) * w1
                        )
                        for off in cutlass.range_constexpr(3):
                            sh: cutlass.Constexpr = 4 << off
                            r0 = r0 + cute.arch.shuffle_sync_bfly(r0, offset=sh)
                            r1 = r1 + cute.arch.shuffle_sync_bfly(r1, offset=sh)
                        if gid == Int32(0):
                            _write_score_pair(
                                scores_base, Int64(0), max_groups, limit,
                                g_base + ltid * Int32(2), r0, r1,
                            )
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(2) + cons_idx)
                cons_idx = cons_idx ^ Int32(1)
                if cons_idx == Int32(0):
                    cons_phase = cons_phase ^ Int32(1)
                pair_idx = pair_idx + grid_y

    @cute.kernel
    def kernel_8row(
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
        total_pages: Int32,
        total_pairs: Int32,
    ):
        row0 = Int32(cute.arch.block_idx()[0]) * Int32(8)
        live_rows = rows - row0
        if live_rows > Int32(8):
            live_rows = Int32(8)
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        grid_y = Int32(cute.arch.grid_dim()[1])

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage_8row())
        q_s_addr = shared_ptr_to_u32(st.q_s.data_ptr())
        pages_base = shared_ptr_to_u32(st.pages.data_ptr())
        w_s = st.w_s.get_tensor(cute.make_layout((8, 16), stride=(16, 1)))
        lims_s = st.lims_s.get_tensor(cute.make_layout((8,)))
        stage_shared = st.stage_shared.get_tensor(cute.make_layout((2,)))
        mbar_base = st.mbar.data_ptr()

        if tid == Int32(0):
            cute.arch.mbarrier_init(mbar_base + Int32(0), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(1), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(2), Int32(1))
            cute.arch.mbarrier_init(mbar_base + Int32(3), Int32(1))
        if tid < Int32(8):
            lims_s[tid] = Int32(group_limits[Int64(row0 + tid)]) if tid < live_rows else Int32(0)
        cute.arch.barrier()

        max_lim = Int32(0)
        for r_i in cutlass.range_constexpr(8):
            lv = lims_s[r_i]
            if lv > max_lim:
                max_lim = lv

        cache_base = Int64(group_cache.toint())
        ts = Int64(table_stride)
        stage_bytes = Int32(2 * _PAGE_BYTES)

        if warp == Int32(4):
            prod_idx = Int32(0)
            prod_phase = Int32(1)
            pair_idx = Int32(cute.arch.block_idx()[1])
            while pair_idx < total_pairs:
                cute.arch.mbarrier_wait(mbar_base + Int32(2) + prod_idx, phase=prod_phase)
                if lane == Int32(0):
                    p0 = pair_idx * Int32(2)
                    p1 = p0 + Int32(1)
                    shared = Int32(1)
                    ref0 = Int64(-1)
                    ref1 = Int64(-1)
                    if p0 < total_pages and p0 * Int32(_GROUPS_PER_PAGE) < max_lim:
                        for r_off in cutlass.range_constexpr(8):
                            if Int32(r_off) < live_rows and p0 * Int32(_GROUPS_PER_PAGE) < lims_s[r_off]:
                                ph = Int64(page_table[Int64(row0 + Int32(r_off)) * ts + Int64(p0)])
                                if ref0 < Int64(0):
                                    ref0 = ph
                                elif ph != ref0:
                                    shared = Int32(0)
                    if p1 < total_pages and p1 * Int32(_GROUPS_PER_PAGE) < max_lim:
                        for r_off in cutlass.range_constexpr(8):
                            if Int32(r_off) < live_rows and p1 * Int32(_GROUPS_PER_PAGE) < lims_s[r_off]:
                                ph = Int64(page_table[Int64(row0 + Int32(r_off)) * ts + Int64(p1)])
                                if ref1 < Int64(0):
                                    ref1 = ph
                                elif ph != ref1:
                                    shared = Int32(0)
                    stage_shared[prod_idx] = shared
                    cute.arch.fence_acq_rel_cta()
                    full_ptr = mbar_base + prod_idx
                    full_addr = shared_ptr_to_u32(full_ptr)
                    if shared == Int32(1):
                        has0 = Int32(1) if ref0 >= Int64(0) else Int32(0)
                        has1 = Int32(1) if ref1 >= Int64(0) else Int32(0)
                        cute.arch.mbarrier_arrive_and_expect_tx(full_ptr, (has0 + has1) * Int32(_PAGE_BYTES))
                        dst_base = pages_base + prod_idx * stage_bytes
                        if ref0 >= Int64(0):
                            cp_async_bulk_g2s_mbar(dst_base, cache_base + ref0 * Int64(_PAGE_BYTES), Int32(_PAGE_BYTES), full_addr)
                        if ref1 >= Int64(0):
                            cp_async_bulk_g2s_mbar(dst_base + Int32(_PAGE_BYTES), cache_base + ref1 * Int64(_PAGE_BYTES), Int32(_PAGE_BYTES), full_addr)
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(full_ptr, Int32(0))
                prod_idx = prod_idx ^ Int32(1)
                if prod_idx == Int32(0):
                    prod_phase = prod_phase ^ Int32(1)
                pair_idx = pair_idx + grid_y
        else:
            # 4 Math warps (128 threads): stage permuted Q [live_rows, 16, 136] + weights in SMEM
            # concurrently while IO warp prefetches the first 2 page pairs!
            q_base = Int64(q_rot.toint())
            w_base = Int64(weights.toint())
            scores_base = Int64(scores_out.toint())

            total_q_vecs = live_rows * Int32(256)  # 16 heads * 16 x 16B vectors per row
            v_idx = tid
            while v_idx < total_q_vecs:
                r_off = v_idx >> Int32(8)
                in_row = v_idx & Int32(255)
                h_loc = in_row >> Int32(4)
                in_h = in_row & Int32(15)
                l = in_h >> Int32(2)
                seg = in_h & Int32(3)
                src_q = q_base + ((Int64(row0 + r_off) * Int64(16) + Int64(h_loc)) * Int64(256)) + Int64(in_h * 16)
                u0, u1, u2, u3 = ld_global_nc_v4_u32(src_q)
                tile0 = q_s_addr + r_off * Int32(self.q_smem_token_bytes) + h_loc * Int32(self.q_smem_row_bytes) + seg * Int32(64)
                st_shared_u32(tile0 + l * Int32(4), u0)
                st_shared_u32(tile0 + Int32(16) + l * Int32(4), u1)
                st_shared_u32(tile0 + Int32(32) + l * Int32(4), u2)
                st_shared_u32(tile0 + Int32(48) + l * Int32(4), u3)
                v_idx = v_idx + Int32(128)

            if tid < live_rows * Int32(16):
                wr_off = tid >> Int32(4)
                wh_loc = tid & Int32(15)
                w_s[wr_off, wh_loc] = ld_global_nc_f32(w_base + (Int64(row0 + wr_off) * Int64(16) + Int64(wh_loc)) * Int64(4))
            cute.arch.barrier(barrier_id=3, number_of_threads=128)

            gid = lane >> Int32(2)
            ltid = lane & Int32(3)
            p_sub = warp >> Int32(1)
            half = warp & Int32(1)
            a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            a_col_byte = (lane >> Int32(4)) * Int32(16)
            q_lane_off = a_row * Int32(self.q_smem_row_bytes) + a_col_byte
            k_lane_off = ltid * Int32(32)

            acc = cute.make_rmem_tensor((32,), Float32)
            cons_idx = Int32(0)
            cons_phase = Int32(0)
            pair_idx = Int32(cute.arch.block_idx()[1])
            while pair_idx < total_pairs:
                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                pages_addr = pages_base + cons_idx * stage_bytes
                is_shared = stage_shared[cons_idx]

                if is_shared == Int32(1):
                    p_idx = pair_idx * Int32(2) + p_sub
                    if p_idx < total_pages:
                        g_base = p_idx * Int32(_GROUPS_PER_PAGE) + half * Int32(8)
                        if g_base >= max_lim:
                            if gid == Int32(0):
                                for r_off in cutlass.range_constexpr(8):
                                    if Int32(r_off) < live_rows:
                                        _write_score_pair(
                                            scores_base, Int64(row0 + Int32(r_off)), max_groups, lims_s[r_off],
                                            g_base + ltid * Int32(2), Float32(-Float32.inf), Float32(-Float32.inf),
                                        )
                        else:
                            rec_addr = (
                                pages_addr
                                + p_sub * Int32(_PAGE_BYTES)
                                + (half * Int32(8) + gid) * Int32(_GROUP_RECORD_BYTES)
                            )
                            scale = ld_shared_f32(rec_addr + Int32(128))
                            for idx_a in cutlass.range_constexpr(32):
                                acc[idx_a] = Float32(0.0)
                            for ks in cutlass.range_constexpr(8):
                                word = ld_shared_u32(rec_addr + k_lane_off + Int32(ks * 4))
                                b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(word)
                                ks_off = q_lane_off + Int32(ks * 32)
                                for r_off in cutlass.range_constexpr(8):
                                    if Int32(r_off) < live_rows:
                                        a_off: cutlass.Constexpr = r_off * 4
                                        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                                            q_s_addr + Int32(r_off * self.q_smem_token_bytes) + ks_off
                                        )
                                        d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                                            acc[a_off + 0], acc[a_off + 1],
                                            acc[a_off + 2], acc[a_off + 3],
                                            a0, a1, a2, a3, b0, b1,
                                        )
                                        acc[a_off + 0] = d0
                                        acc[a_off + 1] = d1
                                        acc[a_off + 2] = d2
                                        acc[a_off + 3] = d3
                            s0 = cute.arch.shuffle_sync(scale, ltid * Int32(8))
                            s1 = cute.arch.shuffle_sync(scale, ltid * Int32(8) + Int32(4))
                            for r_off in cutlass.range_constexpr(8):
                                if Int32(r_off) < live_rows:
                                    a_off: cutlass.Constexpr = r_off * 4
                                    w0_v = w_s[r_off, gid]
                                    w1_v = w_s[r_off, gid + Int32(8)]
                                    r0 = (
                                        fmax_f32(acc[a_off + 0] * s0, Float32(0.0)) * w0_v
                                        + fmax_f32(acc[a_off + 2] * s0, Float32(0.0)) * w1_v
                                    )
                                    r1 = (
                                        fmax_f32(acc[a_off + 1] * s1, Float32(0.0)) * w0_v
                                        + fmax_f32(acc[a_off + 3] * s1, Float32(0.0)) * w1_v
                                    )
                                    for off in cutlass.range_constexpr(3):
                                        sh: cutlass.Constexpr = 4 << off
                                        r0 = r0 + cute.arch.shuffle_sync_bfly(r0, offset=sh)
                                        r1 = r1 + cute.arch.shuffle_sync_bfly(r1, offset=sh)
                                    if gid == Int32(0):
                                        _write_score_pair(
                                            scores_base, Int64(row0 + Int32(r_off)), max_groups, lims_s[r_off],
                                            g_base + ltid * Int32(2), r0, r1,
                                        )
                else:
                    # Divergent page tables: math warps stage and score each row independently in pages[cons_idx].
                    p0 = pair_idx * Int32(2)
                    p1 = p0 + Int32(1)
                    for r_off in cutlass.range_constexpr(8):
                        if Int32(r_off) < live_rows:
                            cur_row = Int64(row0 + Int32(r_off))
                            cur_lim = lims_s[r_off]
                            if p0 < total_pages and p0 * Int32(_GROUPS_PER_PAGE) < cur_lim:
                                ph0 = Int64(page_table[cur_row * ts + Int64(p0)])
                                _stage_group_page(cache_base, ph0, pages_addr, tid)
                            if p1 < total_pages and p1 * Int32(_GROUPS_PER_PAGE) < cur_lim:
                                ph1 = Int64(page_table[cur_row * ts + Int64(p1)])
                                _stage_group_page(cache_base, ph1, pages_addr + Int32(_PAGE_BYTES), tid)
                            cute.arch.barrier(barrier_id=3, number_of_threads=128)
                            p_idx = pair_idx * Int32(2) + p_sub
                            if p_idx < total_pages:
                                g_base = p_idx * Int32(_GROUPS_PER_PAGE) + half * Int32(8)
                                if g_base >= cur_lim:
                                    if gid == Int32(0):
                                        _write_score_pair(
                                            scores_base, cur_row, max_groups, cur_lim,
                                            g_base + ltid * Int32(2), Float32(-Float32.inf), Float32(-Float32.inf),
                                        )
                                else:
                                    rec_addr = (
                                        pages_addr
                                        + p_sub * Int32(_PAGE_BYTES)
                                        + (half * Int32(8) + gid) * Int32(_GROUP_RECORD_BYTES)
                                    )
                                    scale = ld_shared_f32(rec_addr + Int32(128))
                                    d0 = Float32(0.0)
                                    d1 = Float32(0.0)
                                    d2 = Float32(0.0)
                                    d3 = Float32(0.0)
                                    q_row_base = q_s_addr + Int32(r_off * self.q_smem_token_bytes) + q_lane_off
                                    for ks in cutlass.range_constexpr(8):
                                        word = ld_shared_u32(rec_addr + k_lane_off + Int32(ks * 4))
                                        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(word)
                                        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_row_base + Int32(ks * 32))
                                        d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                                            d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,
                                        )
                                    s0 = cute.arch.shuffle_sync(scale, ltid * Int32(8))
                                    s1 = cute.arch.shuffle_sync(scale, ltid * Int32(8) + Int32(4))
                                    w0_v = w_s[r_off, gid]
                                    w1_v = w_s[r_off, gid + Int32(8)]
                                    r0 = (
                                        fmax_f32(d0 * s0, Float32(0.0)) * w0_v
                                        + fmax_f32(d2 * s0, Float32(0.0)) * w1_v
                                    )
                                    r1 = (
                                        fmax_f32(d1 * s1, Float32(0.0)) * w0_v
                                        + fmax_f32(d3 * s1, Float32(0.0)) * w1_v
                                    )
                                    for off in cutlass.range_constexpr(3):
                                        sh: cutlass.Constexpr = 4 << off
                                        r0 = r0 + cute.arch.shuffle_sync_bfly(r0, offset=sh)
                                        r1 = r1 + cute.arch.shuffle_sync_bfly(r1, offset=sh)
                                    if gid == Int32(0):
                                        _write_score_pair(
                                            scores_base, cur_row, max_groups, cur_lim,
                                            g_base + ltid * Int32(2), r0, r1,
                                        )
                            cute.arch.barrier(barrier_id=3, number_of_threads=128)

                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + Int32(2) + cons_idx)
                cons_idx = cons_idx ^ Int32(1)
                if cons_idx == Int32(0):
                    cons_phase = cons_phase ^ Int32(1)
                pair_idx = pair_idx + grid_y


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


_TOPK_THREADS = 1024
_TOPK_WARPS = 32
_TOPK_CAND_CAP = 8192


@cute.jit
def _store_slots_v4(
    slots_out: cute.Pointer,
    out_row_addr: Int64,
    out_base: Int64,
    slot_idx: Int32,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    aligned16: Int32,
):
    if aligned16 != Int32(0):
        st_global_v4_u32(
            out_row_addr + Int64(slot_idx) * Int64(4),
            Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3),
        )
    else:
        slots_out[out_base + Int64(slot_idx)] = v0
        slots_out[out_base + Int64(slot_idx + Int32(1))] = v1
        slots_out[out_base + Int64(slot_idx + Int32(2))] = v2
        slots_out[out_base + Int64(slot_idx + Int32(3))] = v3


@cute.jit
def _scan_hist_256(
    hist_addr: Int32,
    warp_tot: cute.Tensor,
    meta: cute.Tensor,
    tid: Int32,
    warp: Int32,
    lane: Int32,
    rem: Int32,
):
    """Exclusive prefix sum over hist[0..255] and select the bucket containing rank ``rem``."""
    if tid < Int32(256):
        v = ld_shared_i32(hist_addr + tid * Int32(4))
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
    if tid < Int32(256):
        v = ld_shared_i32(hist_addr + tid * Int32(4))
        s = v
        for off in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(s, offset=1 << off)
            if lane >= Int32(1 << off):
                s = s + up
        warp_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
        excl_val = warp_off + s - v
        if excl_val < rem and excl_val + v >= rem:
            meta[0] = tid
            meta[1] = rem - excl_val
            meta[2] = v
    cute.arch.sync_threads()


@cute.jit
def _collect_topk_ordered(
    scores: cute.Pointer,
    score_base: Int64,
    limit: Int32,
    thresh: Uint32,
    prior_k: Int32,
    eq_needed: Int32,
    out_groups: cute.Tensor,
    warp_tot: cute.Tensor,
    meta: cute.Tensor,
    tid: Int32,
    warp: Int32,
    lane: Int32,
):
    # Keep the master's two gid-ordered runs: strictly better groups, then
    # the lowest gids at the cutoff. The cutoff itself comes from the fast radix path.
    eq_base = prior_k - eq_needed
    if tid == Int32(0):
        meta[3] = Int32(0)  # strict_taken
        meta[4] = Int32(0)  # eq_taken (may exceed eq_needed in the last tile)
    cute.arch.sync_threads()

    # Two 11-bit counts in one scan; neither can exceed the 1024-thread tile.
    base = Int32(0)
    while base < limit and (meta[3] < eq_base or meta[4] < eq_needed):
        gid = base + tid
        is_strict = Int32(0)
        is_eq = Int32(0)
        if gid < limit:
            key = _desc_key(Float32(scores[score_base + Int64(gid)]))
            if key < thresh:
                is_strict = Int32(1)
            elif key == thresh:
                is_eq = Int32(1)
        strict_start = meta[3]
        eq_start = meta[4]
        s = is_strict + (is_eq << Int32(11))
        for off in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(s, offset=1 << off)
            if lane >= Int32(1 << off):
                s = s + up
        if lane == Int32(31):
            warp_tot[warp] = s
        cute.arch.sync_threads()
        if warp == Int32(0):
            wt = warp_tot[lane]
            for off in cutlass.range_constexpr(5):
                up = cute.arch.shuffle_sync_up(wt, offset=1 << off)
                if lane >= Int32(1 << off):
                    wt = wt + up
            warp_tot[lane] = wt
        cute.arch.sync_threads()
        packed_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
        prefix = packed_off + s
        if is_strict != Int32(0):
            out_groups[strict_start + (prefix & Int32(2047)) - Int32(1)] = gid
        if is_eq != Int32(0):
            rank_eq = eq_start + (prefix >> Int32(11)) - Int32(1)
            if rank_eq < eq_needed:
                out_groups[eq_base + rank_eq] = gid
        if tid == Int32(_TOPK_THREADS - 1):
            meta[3] = strict_start + (prefix & Int32(2047))
            meta[4] = eq_start + (prefix >> Int32(11))
        cute.arch.sync_threads()
        base = base + Int32(_TOPK_THREADS)


class _Hy4IndexerTopkExpand:
    """Zero-scratch deterministic GPU radix top-511 group select + physical KV slot expansion.

    The 1024-thread CTA finds the cutoff with a 10-bit coarse histogram and
    up to 8192 buffered keys from its cutoff bucket. It falls back to global
    refinement on overflow. The final scan preserves logical gid order within
    the strict and tied runs, before vectorized physical slot expansion.
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
            "hist": cute.struct.Align[cute.struct.MemRange[Int32, 1024], 16],
            "cand_key": cute.struct.Align[cute.struct.MemRange[Uint32, _TOPK_CAND_CAP], 16],
            "out_groups": cute.struct.Align[cute.struct.MemRange[Int32, 512], 16],
            "warp_tot": cute.struct.Align[cute.struct.MemRange[Int32, _TOPK_WARPS], 16],
            "meta": cute.struct.Align[cute.struct.MemRange[Int32, 8], 16],
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
        if rows > Int32(0):
            self.kernel(
                scores, positions, page_table, slots_out, lengths_out, score_stride, table_stride
            ).launch(grid=(rows, 1, 1), block=(_TOPK_THREADS, 1, 1), stream=stream)

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
        warp = tid >> Int32(5)
        lane = tid & Int32(31)

        pos = Int32(positions[row])
        seq_len = pos + Int32(1)
        out_base = row * Int64(self.candidate_topk)
        out_row_addr = Int64(slots_out.toint()) + out_base * Int64(4)
        aligned16 = Int32(1) if (out_row_addr & Int64(15)) == Int64(0) else Int32(0)
        pt_base = row * Int64(table_stride)

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(self._storage())
        hist = st.hist.get_tensor(cute.make_layout((1024,)))
        cand_key = st.cand_key.get_tensor(cute.make_layout((_TOPK_CAND_CAP,)))
        out_groups = st.out_groups.get_tensor(cute.make_layout((512,)))
        warp_tot = st.warp_tot.get_tensor(cute.make_layout((_TOPK_WARPS,)))
        meta = st.meta.get_tensor(cute.make_layout((8,)))
        hist_addr = shared_ptr_to_u32(hist.iterator)
        meta_addr = shared_ptr_to_u32(meta.iterator)

        if seq_len <= Int32(self.candidate_topk):
            # Stage <= 32 page_table entries once in SMEM, then emit 512 x 4 slots in 1 store/thread.
            if tid < Int32(32):
                if tid * Int32(64) < seq_len:
                    hist[tid] = Int32(page_table[pt_base + Int64(tid)])
                else:
                    hist[tid] = Int32(0)
            cute.arch.sync_threads()
            if tid < Int32(512):
                idx0 = tid * Int32(4)
                page = Int64(hist[tid >> Int32(4)])
                s0 = Int32(page * Int64(64) + Int64(idx0 & Int32(63)))
                v0 = s0 if idx0 < seq_len else Int32(-1)
                v1 = s0 + Int32(1) if idx0 + Int32(1) < seq_len else Int32(-1)
                v2 = s0 + Int32(2) if idx0 + Int32(2) < seq_len else Int32(-1)
                v3 = s0 + Int32(3) if idx0 + Int32(3) < seq_len else Int32(-1)
                _store_slots_v4(slots_out, out_row_addr, out_base, idx0, v0, v1, v2, v3, aligned16)
            if tid == Int32(0):
                lengths_out[row] = seq_len
        else:
            cur_group = pos // Int32(4)
            limit = cur_group
            score_base = row * Int64(score_stride)
            row_score_addr = Int64(scores.toint()) + score_base * Int64(4)
            score_aligned16 = Int32(1) if (row_score_addr & Int64(15)) == Int64(0) else Int32(0)
            limit_vec = limit & ~Int32(3)

            # Pass 1: 10-bit coarse histogram (1024 bins, top bits 22..31).
            hist[tid] = Int32(0)
            cute.arch.sync_threads()
            if score_aligned16 != Int32(0):
                g4 = tid * Int32(4)
                while g4 < limit_vec:
                    u0, u1, u2, u3 = ld_global_nc_v4_u32(row_score_addr + Int64(g4) * Int64(4))
                    b0 = Int32(_desc_key(u32_as_f32(u0)) >> Uint32(22))
                    b1 = Int32(_desc_key(u32_as_f32(u1)) >> Uint32(22))
                    b2 = Int32(_desc_key(u32_as_f32(u2)) >> Uint32(22))
                    b3 = Int32(_desc_key(u32_as_f32(u3)) >> Uint32(22))
                    red_add_shared_i32(hist_addr + b0 * Int32(4), Int32(1))
                    red_add_shared_i32(hist_addr + b1 * Int32(4), Int32(1))
                    red_add_shared_i32(hist_addr + b2 * Int32(4), Int32(1))
                    red_add_shared_i32(hist_addr + b3 * Int32(4), Int32(1))
                    g4 = g4 + Int32(_TOPK_THREADS * 4)
                if tid < limit - limit_vec:
                    gid_t = limit_vec + tid
                    bt = Int32(_desc_key(Float32(scores[score_base + Int64(gid_t)])) >> Uint32(22))
                    red_add_shared_i32(hist_addr + bt * Int32(4), Int32(1))
            else:
                gid = tid
                while gid < limit:
                    b = Int32(_desc_key(Float32(scores[score_base + Int64(gid)])) >> Uint32(22))
                    red_add_shared_i32(hist_addr + b * Int32(4), Int32(1))
                    gid = gid + Int32(_TOPK_THREADS)
            cute.arch.sync_threads()

            # 1024-thread exclusive prefix sum over hist[0..1023].
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
                wt = warp_tot[lane]
                for off in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(wt, offset=1 << off)
                    if lane >= Int32(1 << off):
                        wt = wt + up
                warp_tot[lane] = wt
            cute.arch.sync_threads()
            warp_off = warp_tot[warp - Int32(1)] if warp > Int32(0) else Int32(0)
            excl_val = warp_off + s - v
            if excl_val < Int32(self.prior_k) and excl_val + v >= Int32(self.prior_k):
                meta[0] = tid
                meta[1] = Int32(self.prior_k) - excl_val
                meta[2] = v
            if tid == Int32(0):
                meta[4] = Int32(0)  # count of buffered keys
            cute.arch.sync_threads()

            b0_sel = Uint32(meta[0])
            rem0 = Int32(meta[1])
            b0_count = Int32(meta[2])

            buffered = b0_count <= Int32(_TOPK_CAND_CAP)
            if buffered:
                # Pass 2: buffer only keys in the cutoff bucket. Insertion
                # order does not affect the histogram; the final scan emits gids.
                if score_aligned16 != Int32(0):
                    g4 = tid * Int32(4)
                    while g4 < limit_vec:
                        u0, u1, u2, u3 = ld_global_nc_v4_u32(row_score_addr + Int64(g4) * Int64(4))
                        ks_vec = cute.make_rmem_tensor((4,), Uint32)
                        ks_vec[0] = _desc_key(u32_as_f32(u0))
                        ks_vec[1] = _desc_key(u32_as_f32(u1))
                        ks_vec[2] = _desc_key(u32_as_f32(u2))
                        ks_vec[3] = _desc_key(u32_as_f32(u3))
                        for sub in cutlass.range_constexpr(4):
                            kv = ks_vec[sub]
                            if kv >> Uint32(22) == b0_sel:
                                p_c = atomic_add_shared_i32(meta_addr + Int32(16), Int32(1))
                                cand_key[p_c] = kv
                        g4 = g4 + Int32(_TOPK_THREADS * 4)
                    if tid < limit - limit_vec:
                        gid = limit_vec + tid
                        kv = _desc_key(Float32(scores[score_base + Int64(gid)]))
                        if kv >> Uint32(22) == b0_sel:
                            p_c = atomic_add_shared_i32(meta_addr + Int32(16), Int32(1))
                            cand_key[p_c] = kv
                else:
                    gid = tid
                    while gid < limit:
                        kv = _desc_key(Float32(scores[score_base + Int64(gid)]))
                        if kv >> Uint32(22) == b0_sel:
                            p_c = atomic_add_shared_i32(meta_addr + Int32(16), Int32(1))
                            cand_key[p_c] = kv
                        gid = gid + Int32(_TOPK_THREADS)
                cute.arch.sync_threads()

            prefix = b0_sel << Uint32(22)
            mask = Uint32(0xFFC00000)
            rem = rem0
            # Refine the remaining 22 bits (8 + 8 + 6). Scan global scores
            # instead of the buffer when the coarse bucket overflows.
            for p_i in cutlass.range_constexpr(3):
                sh_bits: cutlass.Constexpr = 14 if p_i == 0 else (6 if p_i == 1 else 0)
                b_mask: cutlass.Constexpr = 0xFF if p_i < 2 else 0x3F
                if tid < Int32(256):
                    hist[tid] = Int32(0)
                cute.arch.sync_threads()
                i = tid
                end = b0_count if buffered else limit
                while i < end:
                    kv = Uint32(0)
                    if buffered:
                        kv = cand_key[i]
                    else:
                        kv = _desc_key(Float32(scores[score_base + Int64(i)]))
                    if (kv & mask) == prefix:
                        bk = Int32((kv >> Uint32(sh_bits)) & Uint32(b_mask))
                        red_add_shared_i32(hist_addr + bk * Int32(4), Int32(1))
                    i = i + Int32(_TOPK_THREADS)
                cute.arch.sync_threads()
                _scan_hist_256(hist_addr, warp_tot, meta, tid, warp, lane, rem)
                chosen = Uint32(meta[0])
                prefix = prefix | (chosen << Uint32(sh_bits))
                mask = mask | (Uint32(b_mask) << Uint32(sh_bits))
                rem = meta[1]

            _collect_topk_ordered(
                scores, score_base, limit, prefix, Int32(self.prior_k), rem,
                out_groups, warp_tot, meta, tid, warp, lane,
            )

            # Expand top-511 prior groups + current group into physical KV slots (1 lookup + 1 v4 store per group).
            cur_valid = (pos & Int32(3)) + Int32(1)
            if tid < Int32(512):
                if tid < Int32(self.prior_k):
                    g_id = out_groups[tid]
                    page = Int64(page_table[pt_base + Int64(g_id >> Int32(4))])
                    s0 = Int32(page * Int64(64) + Int64((g_id & Int32(15)) * Int32(4)))
                    _store_slots_v4(
                        slots_out, out_row_addr, out_base, tid * Int32(4),
                        s0, s0 + Int32(1), s0 + Int32(2), s0 + Int32(3), aligned16,
                    )
                else:
                    page = Int64(page_table[pt_base + Int64(cur_group >> Int32(4))])
                    s0 = Int32(page * Int64(64) + Int64((cur_group & Int32(15)) * Int32(4)))
                    v0 = s0
                    v1 = s0 + Int32(1) if cur_valid > Int32(1) else Int32(-1)
                    v2 = s0 + Int32(2) if cur_valid > Int32(2) else Int32(-1)
                    v3 = s0 + Int32(3) if cur_valid > Int32(3) else Int32(-1)
                    _store_slots_v4(
                        slots_out, out_row_addr, out_base, Int32(self.prior_k * 4),
                        v0, v1, v2, v3, aligned16,
                    )
                    lengths_out[row] = Int32(self.prior_k * 4) + cur_valid


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
