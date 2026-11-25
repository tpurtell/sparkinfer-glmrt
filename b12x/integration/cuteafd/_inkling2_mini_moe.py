"""Inkling 2 Mini specialized prefill routed-MoE GEMM kernels (NVFP4 W4A4 & MXFP8 W8A8).

Geometry-specialized for Inkling 2 Mini TP1 (hidden=2304, slice=1152, experts=128, top_k=8,
SiTU gate_cap=4, up_cap=25, route weight before FC2) at prefill capacities (max_rows >= 256).
Every other model family and capacity remains byte-identical to stock.

Round 2 architecture (strictly K-order preserving -> bit-identical to stock):
1. Expert-stationary multi-chunk scheduling (up to 4 chunks = 512 rows per CTA) with O(1)
   4-bucket Longest-Group-First (LJF) ordering over E=128 experts:
   - Reads each expert's weights from DRAM strictly once for experts with <= 512 rows
     (~95%+ of experts at R=4096, ~100% at R=256..3000), reusing weights from L2 across
     chunks 1..3 and reusing preloaded coalesced weight scales from SMEM across all chunks.
   - Launches heaviest 4-chunk groups first (Wave 0) and lightest 1-chunk groups last,
     eliminating tail-wave stragglers under skewed routing while keeping grid.x=18 fast for
     18x activation L2 sharing.
2. True warp specialization (8 MMA warps + 1 dedicated TMA DMA warp = 288 threads,
   setmaxnreg 232/40) on NVFP4 GateUp, NVFP4 Down, and MXFP8 Down:
   - Zero __syncthreads() in the K-loop (async full/empty mbarrier ring with 8-warp arrivals).
   - DMA warp streams continuously across chunk boundaries without draining on epilogues.
   - Only 64 accumulator regs/thread (vs 128 in 2-chunk fusion), enabling full ldmatrix
     fragment double-buffering with zero register spills.
3. Coalesced SMEM scale staging & zero-SMEM register epilogue (NVFP4 GateUp):
   - Preloads s1/s3 (18.9 KB) and s2 (9.7 KB / 4.6 KB) coalesced along K once per CTA with
     odd word strides (0 SMEM bank conflicts, 0 global scale loads in K-loop).
   - NVFP4 GateUp streams activation scales xs via 2D TMA ((32, 16) boxes) and assigns
     (32 gate + 32 up) columns per warp so SiTU x gamma and 16-col amax butterfly reduction
     run directly in registers with 0 epilogue SMEM and 0 epilogue barriers.
   - NVFP4 Down and MXFP8 Down preload per-chunk activation scales coalesced into SMEM and
     stage coalesced 16B BF16 stores through a dedicated 8.7 KB 32-row epilogue buffer
     using MMA-only named barriers (barrier_id=1, 256 threads).
4. MXFP8 GateUp: BM=128 wide-N (64 gate + 64 up) expert-looping kernel with w_ring=3 TMA,
   a_ring=2 cp.async (16B cp_async4 on 16B-aligned even rows), hoisted weight scale loads
   (4x fewer LDS), double-buffered ldmatrix fragments, and overlapped next-chunk W TMA
   during the K32 quantization epilogue.
5. Launched through @cute.kernel methods on the exact stock classes (StreamNvfp4GateUpDyn,
   StreamNvfp4LinearDyn, StreamMxfp8GateUp, StreamMxfp8Down) so CUPTI kernel attribution
   scores GEMM-only durations and splits gate/up vs down cleanly.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.activation import gated_activation_value, situ_vendor
from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    ld_shared_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    max_abs_32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_u8,
    st_global_u32,
    st_global_v4_u32,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import META_HEAD, _i32_at, _round_bf16
from ._fp8_moe_stream import STREAM_TILE_M, _u8_weight
from ._fp8_weights import _ld_f32
from ._mxfp4_down_a8_plan import mxfp8_down_row_bytes

__all__ = [
    "MINI_EXPERTS",
    "MINI_HIDDEN",
    "MINI_INTER",
    "MINI_MAX_CHUNKS",
    "mini_max_groups",
    "mini_mxfp8_down_body",
    "mini_mxfp8_down_layouts",
    "mini_mxfp8_down_storage",
    "mini_mxfp8_gate_up_body",
    "mini_mxfp8_gate_up_layout",
    "mini_mxfp8_gate_up_storage",
    "mini_nvfp4_down_body",
    "mini_nvfp4_down_layouts",
    "mini_nvfp4_down_storage",
    "mini_nvfp4_gate_up_body",
    "mini_nvfp4_gate_up_layouts",
    "mini_nvfp4_gate_up_storage",
    "mini_u8_strided",
    "mini_u8_weight",
]

MINI_HIDDEN = 2304
MINI_INTER = 1152
MINI_EXPERTS = 128
MINI_MAX_CHUNKS = 4
MINI_MMA_THREADS = 256
MINI_WS_THREADS = 288
MINI_MMA_REGS = 232
MINI_DMA_REGS = 40


def mini_u8_weight(pointer: cute.Pointer, rows, cols: int) -> cute.Tensor:
    return _u8_weight(pointer, rows, cols)


def mini_u8_strided(pointer: cute.Pointer, rows, cols: int, row_stride: int) -> cute.Tensor:
    u8 = cute.make_ptr(cutlass.Uint8, Int64(pointer.toint()), cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(u8, cute.make_layout((rows, cols), stride=(row_stride, 1)))


@cute.jit
def mini_max_groups(experts: cutlass.Constexpr, pairs: Int32) -> Int32:
    """Upper bound on sum_e ceil(counts[e] / 512) for any routing of ``pairs`` pairs."""
    groups = pairs
    if groups > Int32(experts):
        groups = Int32(experts)
    return (pairs + Int32(511)) // Int32(512) + groups


@cute.jit
def _resolve_group_ljf_128(meta: cute.Pointer, group_id: Int32, tidx: Int32,
                           warp_id: Int32, lane: Int32, smem_base: Int32):
    """Map ``group_id`` to ``(expert, tile0, first0, count_rem, n_chunks)`` in LJF order.

    Bucket 0: full 4-chunk groups (512 rows, or final partial 4th chunk).
    Bucket 1: 3-chunk remainder groups.
    Bucket 2: 2-chunk remainder groups.
    Bucket 3: 1-chunk remainder groups.

    Uses 100 bytes at ``smem_base`` before TMA begins (safe for 256 or 288 threads).
    Returns ``expert = -1`` when ``group_id >= total_groups``.
    """
    if tidx == Int32(0):
        st_shared_u32(smem_base + Int32(80), Uint32(0xFFFFFFFF))
    n = Int32(0)
    tiles_e = Int32(0)
    full4 = Int32(0)
    b3 = Int32(0)
    b2 = Int32(0)
    b1 = Int32(0)
    if tidx < Int32(128):
        n = _i32_at(Int64(meta.toint()) + Int64(4 * META_HEAD), tidx)
        tiles_e = (n + Int32(127)) // Int32(128)
        full4 = tiles_e >> Int32(2)
        rem = tiles_e & Int32(3)
        b3 = cutlass.select_(rem == Int32(3), Int32(1), Int32(0))
        b2 = cutlass.select_(rem == Int32(2), Int32(1), Int32(0))
        b1 = cutlass.select_(rem == Int32(1), Int32(1), Int32(0))
    s4, s3, s2, s1, st = full4, b3, b2, b1, tiles_e
    for stage in cutlass.range_constexpr(5):
        off = 1 << stage
        u4 = Int32(cute.arch.shuffle_sync_up(s4, offset=off, mask_and_clamp=0))
        u3 = Int32(cute.arch.shuffle_sync_up(s3, offset=off, mask_and_clamp=0))
        u2 = Int32(cute.arch.shuffle_sync_up(s2, offset=off, mask_and_clamp=0))
        u1 = Int32(cute.arch.shuffle_sync_up(s1, offset=off, mask_and_clamp=0))
        ut = Int32(cute.arch.shuffle_sync_up(st, offset=off, mask_and_clamp=0))
        if lane >= Int32(off):
            s4 = s4 + u4
            s3 = s3 + u3
            s2 = s2 + u2
            s1 = s1 + u1
            st = st + ut
    if (warp_id < Int32(4)) & (lane == Int32(31)):
        st_shared_u32(smem_base + warp_id * Int32(4), Uint32(s4))
        st_shared_u32(smem_base + Int32(16) + warp_id * Int32(4), Uint32(s3))
        st_shared_u32(smem_base + Int32(32) + warp_id * Int32(4), Uint32(s2))
        st_shared_u32(smem_base + Int32(48) + warp_id * Int32(4), Uint32(s1))
        st_shared_u32(smem_base + Int32(64) + warp_id * Int32(4), Uint32(st))
    cute.arch.sync_threads()
    if warp_id == Int32(0):
        w4, w3, w2, w1, wt = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
        if lane < Int32(4):
            w4 = Int32(ld_shared_u32(smem_base + lane * Int32(4)))
            w3 = Int32(ld_shared_u32(smem_base + Int32(16) + lane * Int32(4)))
            w2 = Int32(ld_shared_u32(smem_base + Int32(32) + lane * Int32(4)))
            w1 = Int32(ld_shared_u32(smem_base + Int32(48) + lane * Int32(4)))
            wt = Int32(ld_shared_u32(smem_base + Int32(64) + lane * Int32(4)))
        for stage in cutlass.range_constexpr(2):
            off = 1 << stage
            u4 = Int32(cute.arch.shuffle_sync_up(w4, offset=off, mask_and_clamp=0))
            u3 = Int32(cute.arch.shuffle_sync_up(w3, offset=off, mask_and_clamp=0))
            u2 = Int32(cute.arch.shuffle_sync_up(w2, offset=off, mask_and_clamp=0))
            u1 = Int32(cute.arch.shuffle_sync_up(w1, offset=off, mask_and_clamp=0))
            ut = Int32(cute.arch.shuffle_sync_up(wt, offset=off, mask_and_clamp=0))
            if lane >= Int32(off):
                w4 = w4 + u4
                w3 = w3 + u3
                w2 = w2 + u2
                w1 = w1 + u1
                wt = wt + ut
        if lane < Int32(4):
            st_shared_u32(smem_base + lane * Int32(4), Uint32(w4))
            st_shared_u32(smem_base + Int32(16) + lane * Int32(4), Uint32(w3))
            st_shared_u32(smem_base + Int32(32) + lane * Int32(4), Uint32(w2))
            st_shared_u32(smem_base + Int32(48) + lane * Int32(4), Uint32(w1))
            st_shared_u32(smem_base + Int32(64) + lane * Int32(4), Uint32(wt))
    cute.arch.sync_threads()
    if tidx < Int32(128):
        tot4 = Int32(ld_shared_u32(smem_base + Int32(12)))
        tot3 = Int32(ld_shared_u32(smem_base + Int32(28)))
        tot2 = Int32(ld_shared_u32(smem_base + Int32(44)))
        tot1 = Int32(ld_shared_u32(smem_base + Int32(60)))
        base4, base3, base2, base1, baset = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
        if warp_id > Int32(0):
            wm1 = (warp_id - Int32(1)) * Int32(4)
            base4 = Int32(ld_shared_u32(smem_base + wm1))
            base3 = Int32(ld_shared_u32(smem_base + Int32(16) + wm1))
            base2 = Int32(ld_shared_u32(smem_base + Int32(32) + wm1))
            base1 = Int32(ld_shared_u32(smem_base + Int32(48) + wm1))
            baset = Int32(ld_shared_u32(smem_base + Int32(64) + wm1))
        first_tile = baset + st - tiles_e
        hit = Int32(0)
        tile0 = Int32(0)
        row_start = Int32(0)
        count_rem = Int32(0)
        n_chunks = Int32(0)
        if group_id < tot4:
            end4 = base4 + s4
            start4 = end4 - full4
            if (group_id >= start4) & (group_id < end4):
                g_in_e = group_id - start4
                tile0 = first_tile + g_in_e * Int32(4)
                row_start = g_in_e * Int32(512)
                count_rem = n - row_start
                if count_rem > Int32(512):
                    count_rem = Int32(512)
                n_chunks = Int32(4)
                hit = Int32(1)
        elif group_id < tot4 + tot3:
            g3 = group_id - tot4
            start3 = base3 + s3 - b3
            if (b3 != Int32(0)) & (g3 == start3):
                tile0 = first_tile + full4 * Int32(4)
                row_start = full4 * Int32(512)
                count_rem = n - row_start
                n_chunks = Int32(3)
                hit = Int32(1)
        elif group_id < tot4 + tot3 + tot2:
            g2 = group_id - tot4 - tot3
            start2 = base2 + s2 - b2
            if (b2 != Int32(0)) & (g2 == start2):
                tile0 = first_tile + full4 * Int32(4)
                row_start = full4 * Int32(512)
                count_rem = n - row_start
                n_chunks = Int32(2)
                hit = Int32(1)
        elif group_id < tot4 + tot3 + tot2 + tot1:
            g1 = group_id - tot4 - tot3 - tot2
            start1 = base1 + s1 - b1
            if (b1 != Int32(0)) & (g1 == start1):
                tile0 = first_tile + full4 * Int32(4)
                row_start = full4 * Int32(512)
                count_rem = n - row_start
                n_chunks = Int32(1)
                hit = Int32(1)
        if hit != Int32(0):
            offset_e = _i32_at(Int64(meta.toint()) + Int64(4 * (META_HEAD + 128)), tidx)
            st_shared_u32(smem_base + Int32(80), Uint32(tidx))
            st_shared_u32(smem_base + Int32(84), Uint32(tile0))
            st_shared_u32(smem_base + Int32(88), Uint32(offset_e + row_start))
            st_shared_u32(smem_base + Int32(92), Uint32(count_rem))
            st_shared_u32(smem_base + Int32(96), Uint32(n_chunks))
    cute.arch.sync_threads()
    e = Int32(ld_shared_u32(smem_base + Int32(80)))
    tile0 = Int32(ld_shared_u32(smem_base + Int32(84)))
    first0 = Int32(ld_shared_u32(smem_base + Int32(88)))
    count_rem = Int32(ld_shared_u32(smem_base + Int32(92)))
    n_chunks = Int32(ld_shared_u32(smem_base + Int32(96)))
    return e, tile0, first0, count_rem, n_chunks


# ===========================================================================
# 1. NVFP4 W4A4 GateUp (Warp-Specialized 8 MMA + 1 DMA, Zero-SMEM Epilogue)
# ===========================================================================

_NV4_GU_STAGES = 2
_NV4_GU_HALF_N = 64
_NV4_GU_K_BYTES = 128
_NV4_GU_K_BLOCKS = 9
_NV4_GU_BOX_M = 32
_NV4_GU_HALF_BYTES = _NV4_GU_HALF_N * _NV4_GU_K_BYTES      # 8192
_NV4_GU_W_STAGE_BYTES = 2 * _NV4_GU_HALF_BYTES             # 16384
_NV4_GU_A_BOX_BYTES = _NV4_GU_BOX_M * _NV4_GU_K_BYTES      # 4096
_NV4_GU_A_STAGE_BYTES = STREAM_TILE_M * _NV4_GU_K_BYTES    # 16384
_NV4_GU_SFA_BOX_BYTES = _NV4_GU_BOX_M * 16                 # 512
_NV4_GU_SFA_STAGE_BYTES = STREAM_TILE_M * 16               # 2048
_NV4_GU_SC_COLS = MINI_HIDDEN // 16                        # 144 (36 u32 words)
_NV4_GU_SC_STRIDE = _NV4_GU_SC_COLS + 4                    # 148 (37 u32 words -> odd -> 0 conflicts)


def mini_nvfp4_gate_up_layouts():
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, _NV4_GU_K_BYTES),
        cutlass.Uint8)
    w_layout = cute.tile_to_shape(atom, (_NV4_GU_HALF_N, _NV4_GU_K_BYTES, 2 * _NV4_GU_STAGES), order=(0, 1, 2))
    a_layout = cute.tile_to_shape(atom, (_NV4_GU_BOX_M, _NV4_GU_K_BYTES, 4 * _NV4_GU_STAGES), order=(0, 1, 2))
    sfa_layout = cute.make_layout((_NV4_GU_BOX_M, 16, 4 * _NV4_GU_STAGES), stride=(16, 1, _NV4_GU_SFA_BOX_BYTES))
    return w_layout, a_layout, sfa_layout


def mini_nvfp4_gate_up_storage(w_layout, a_layout, sfa_layout):
    class MiniNvfp4GateUpStorage:
        pass

    MiniNvfp4GateUpStorage.__annotations__ = {
        "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
        "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(a_layout)], 1024],
        "sfa": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(sfa_layout)], 1024],
        "sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * _NV4_GU_HALF_N * _NV4_GU_SC_STRIDE], 16],
        "mbar": cute.struct.MemRange[cutlass.Int64, 2 * _NV4_GU_STAGES],
    }
    return cute.struct(MiniNvfp4GateUpStorage)


@cute.jit
def _nv4_gu_mma_k256(a_base: Int32, w_base: Int32, sfa_base: Int32, sc_base: Int32,
                     k_blk: Int32, acc: cute.Tensor, warp_m: Int32, warp_n: Int32,
                     q: Int32, c: Int32, mat: Int32, mrow: Int32):
    """Warp tile 32 x (32 gate + 32 up) over K=256 with double-buffered ldmatrix fragments."""
    r_sfa0 = warp_m * Int32(32) + q + Int32(8) * (c & Int32(1))
    r_sfa1 = r_sfa0 + Int32(16)
    arow0 = warp_m * Int32(32) + Int32(8) * (mat & Int32(1)) + mrow
    arow1 = arow0 + Int32(16)
    bcol_base = warp_n * Int32(32) + Int32(8) * (mat >> Int32(1)) + mrow
    sc_row_base = warp_n * Int32(32) + q
    k_sc_base = k_blk * Int32(16)

    # Prologue: load step=0 A fragments into registers.
    apiece0 = ((mat >> Int32(1)) ^ mrow) * Int32(16)
    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + arow0 * Int32(_NV4_GU_K_BYTES) + apiece0)
    a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(a_base + arow1 * Int32(_NV4_GU_K_BYTES) + apiece0)

    for step in cutlass.range_constexpr(4):
        sfa0 = ld_shared_u32(sfa_base + r_sfa0 * Int32(16) + Int32(4 * step))
        sfa1 = ld_shared_u32(sfa_base + r_sfa1 * Int32(16) + Int32(4 * step))
        k_sc = k_sc_base + Int32(4 * step)
        bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ mrow) * Int32(16)

        # Preload (side=0, pair=0) B fragments and its 2 SFB words.
        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + bcol_base * Int32(_NV4_GU_K_BYTES) + bpiece)
        sfb0 = ld_shared_u32(sc_base + sc_row_base * Int32(_NV4_GU_SC_STRIDE) + k_sc)
        sfb1 = ld_shared_u32(sc_base + (sc_row_base + Int32(8)) * Int32(_NV4_GU_SC_STRIDE) + k_sc)

        for idx in cutlass.range_constexpr(4):
            # Issue next B ldmatrix + SFB loads BEFORE current 4 OMMAs for full overlap.
            nb0, nb1, nb2, nb3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            nsfb0, nsfb1 = Uint32(0), Uint32(0)
            if const_expr(idx < 3):
                nxt_side = (idx + 1) // 2
                nxt_pair = (idx + 1) % 2
                nb_col = bcol_base + Int32(16 * nxt_pair)
                nb0, nb1, nb2, nb3 = ldmatrix_m8n8x4_b16(
                    w_base + Int32(nxt_side * _NV4_GU_HALF_BYTES) + nb_col * Int32(_NV4_GU_K_BYTES) + bpiece)
                nsc_r = Int32(nxt_side * _NV4_GU_HALF_N + 16 * nxt_pair) + sc_row_base
                nsfb0 = ld_shared_u32(sc_base + nsc_r * Int32(_NV4_GU_SC_STRIDE) + k_sc)
                nsfb1 = ld_shared_u32(sc_base + (nsc_r + Int32(8)) * Int32(_NV4_GU_SC_STRIDE) + k_sc)
            elif const_expr(step < 3):
                napiece = ((Int32(2 * (step + 1)) + (mat >> Int32(1))) ^ mrow) * Int32(16)
                na0, na1, na2, na3 = ldmatrix_m8n8x4_b16(a_base + arow0 * Int32(_NV4_GU_K_BYTES) + napiece)
                na4, na5, na6, na7 = ldmatrix_m8n8x4_b16(a_base + arow1 * Int32(_NV4_GU_K_BYTES) + napiece)

            nt0 = 2 * idx
            nt1 = nt0 + 1
            f00 = 4 * nt0
            f10 = 4 * (8 + nt0)
            f01 = 4 * nt1
            f11 = 4 * (8 + nt1)
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3], a0, a1, a2, a3, b0, b1, sfa0, sfb0)
            acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3], a4, a5, a6, a7, b0, b1, sfa1, sfb0)
            acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3], a0, a1, a2, a3, b2, b3, sfa0, sfb1)
            acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3], a4, a5, a6, a7, b2, b3, sfa1, sfb1)
            acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3] = d0, d1, d2, d3

            if const_expr(idx < 3):
                b0, b1, b2, b3 = nb0, nb1, nb2, nb3
                sfb0, sfb1 = nsfb0, nsfb1
            elif const_expr(step < 3):
                a0, a1, a2, a3 = na0, na1, na2, na3
                a4, a5, a6, a7 = na4, na5, na6, na7


@cute.jit
def mini_nvfp4_gate_up_body(
    meta: cute.Pointer, act_t: cute.Tensor, sfa_t: cute.Tensor,
    weight_g: cute.Tensor, s1: cute.Pointer, weight_u: cute.Tensor, s3: cute.Pointer,
    row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
    act_amax_part: cute.Pointer, tma_a: cute.CopyAtom, tma_sfa: cute.CopyAtom,
    tma_g: cute.CopyAtom, tma_u: cute.CopyAtom,
    w_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout, sfa_layout: cute.Layout,
    storage, amax_offset: cutlass.Constexpr, denominator_offset: cutlass.Constexpr,
    gate_cap: cutlass.Constexpr, up_cap: cutlass.Constexpr,
):
    n_blk = Int32(cute.arch.block_idx()[0])
    group_id = Int32(cute.arch.block_idx()[1])
    tidx = Int32(cute.arch.thread_idx()[0])
    warp_id = cute.arch.make_warp_uniform(tidx // Int32(32))
    lane = tidx % Int32(32)

    # Extract all SMEM pointers/tensors BEFORE any dynamic warp-dispatch branch.
    sw_addr = shared_ptr_to_u32(storage.w.data_ptr())
    sa_addr = shared_ptr_to_u32(storage.a.data_ptr())
    ssfa_addr = shared_ptr_to_u32(storage.sfa.data_ptr())
    ssc_addr = shared_ptr_to_u32(storage.sc.data_ptr())
    mbar = storage.mbar.data_ptr()
    full_mbar = mbar
    empty_mbar = mbar + _NV4_GU_STAGES
    s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
    s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
    s_sfa = storage.sfa.get_tensor(sfa_layout)
    t_ws, t_wg = cpasync.tma_partition(
        tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
        cute.group_modes(cute.local_tile(weight_g, (_NV4_GU_HALF_N, _NV4_GU_K_BYTES), (None, None)), 0, 2))
    _, t_wu = cpasync.tma_partition(
        tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
        cute.group_modes(cute.local_tile(weight_u, (_NV4_GU_HALF_N, _NV4_GU_K_BYTES), (None, None)), 0, 2))
    t_as, t_ag = cpasync.tma_partition(
        tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
        cute.group_modes(cute.local_tile(act_t, (_NV4_GU_BOX_M, _NV4_GU_K_BYTES), (None, None)), 0, 2))
    t_sfas, t_sfag = cpasync.tma_partition(
        tma_sfa, 0, cute.make_layout(1), cute.group_modes(s_sfa, 0, 2),
        cute.group_modes(cute.local_tile(sfa_t, (_NV4_GU_BOX_M, 16), (None, None)), 0, 2))

    e, tile0, first0, count_rem, n_chunks = _resolve_group_ljf_128(meta, group_id, tidx, warp_id, lane, sw_addr)
    if e >= Int32(0):
        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_g)
            cpasync.prefetch_descriptor(tma_u)
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            for s in cutlass.range_constexpr(_NV4_GU_STAGES):
                cute.arch.mbarrier_init(full_mbar + s, 1)
                cute.arch.mbarrier_init(empty_mbar + s, 8)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        n0 = n_blk * Int32(_NV4_GU_HALF_N)
        w_tile = e * Int32(MINI_INTER // _NV4_GU_HALF_N) + n_blk

        if warp_id == Int32(8):
            cute.arch.setmaxregister_decrease(MINI_DMA_REGS)
            step_idx = Int32(0)
            for ci in cutlass.range(n_chunks, unroll=1):
                tile_c = tile0 + ci
                live_c = count_rem - ci * Int32(STREAM_TILE_M)
                if live_c > Int32(STREAM_TILE_M):
                    live_c = Int32(STREAM_TILE_M)
                live_boxes = (live_c + Int32(31)) // Int32(32)
                tx_bytes = Int32(_NV4_GU_W_STAGE_BYTES) + live_boxes * Int32(_NV4_GU_A_BOX_BYTES + _NV4_GU_SFA_BOX_BYTES)
                base_row_tile = tile_c * Int32(4)
                for k_blk in cutlass.range(_NV4_GU_K_BLOCKS, unroll=1):
                    slot = step_idx % Int32(_NV4_GU_STAGES)
                    if step_idx >= Int32(_NV4_GU_STAGES):
                        e_phase = ((step_idx - Int32(_NV4_GU_STAGES)) // Int32(_NV4_GU_STAGES)) & Int32(1)
                        cute.arch.mbarrier_wait(empty_mbar + slot, e_phase)
                    barrier = full_mbar + slot
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_bytes)
                    cute.copy(tma_g, t_wg[(None, w_tile, k_blk)], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
                    cute.copy(tma_u, t_wu[(None, w_tile, k_blk)], t_ws[(None, slot * Int32(2) + Int32(1))], tma_bar_ptr=barrier)
                    base_slot = slot * Int32(4)
                    for b in cutlass.range_constexpr(4):
                        if Int32(b) < live_boxes:
                            cute.copy(tma_a, t_ag[(None, base_row_tile + Int32(b), k_blk)],
                                      t_as[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)
                            cute.copy(tma_sfa, t_sfag[(None, base_row_tile + Int32(b), k_blk)],
                                      t_sfas[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)
                    step_idx = step_idx + Int32(1)
        else:
            cute.arch.setmaxregister_increase(MINI_MMA_REGS)
            # Preload all 128 weight scale rows (64 gate + 64 up) x 36 u32 words coalesced ONCE per CTA,
            # overlapping directly with the DMA warp's initial k=0,1 TMA pulls.
            sc_words = _NV4_GU_SC_COLS // 4  # 36
            total_sc = 2 * _NV4_GU_HALF_N * sc_words  # 4608 = 18 * 256
            s1_base = Int64(s1.toint())
            s3_base = Int64(s3.toint())
            e_n0_off = Int64(e) * Int64(MINI_INTER) + Int64(n0)
            for qsc in cutlass.range_constexpr(total_sc // MINI_MMA_THREADS):
                u = tidx + Int32(qsc * MINI_MMA_THREADS)
                r = u // Int32(sc_words)
                cw = u % Int32(sc_words)
                base_s = cutlass.select_(r < Int32(_NV4_GU_HALF_N), s1_base, s3_base)
                col_s = cutlass.select_(r < Int32(_NV4_GU_HALF_N), r, r - Int32(_NV4_GU_HALF_N))
                cp_async_u32_shared_global(
                    ssc_addr + r * Int32(_NV4_GU_SC_STRIDE) + cw * Int32(4),
                    base_s + (e_n0_off + Int64(col_s)) * Int64(_NV4_GU_SC_COLS) + Int64(cw) * Int64(4))
            cute.arch.cp_async_commit_group()

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            q = lane // Int32(4)
            c = lane % Int32(4)
            mat = lane // Int32(8)
            mrow = lane % Int32(8)

            w_amax_g = _ld_f32(s1_base + Int64(amax_offset) + Int64(e) * Int64(4))
            rcp_denom_g = div_rn_f32(Float32(1.0), Float32(1536.0) * _ld_f32(s1_base + Int64(denominator_offset)))
            w_amax_u = _ld_f32(s3_base + Int64(amax_offset) + Int64(e) * Int64(4))
            rcp_denom_u = div_rn_f32(Float32(1.0), Float32(1536.0) * _ld_f32(s3_base + Int64(denominator_offset)))

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)

            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            gamma_base = Int64(gamma.toint())
            row_amax_base = Int64(row_amax.toint())
            act_bf16_base = Int64(act_bf16.toint())
            act_part_base = Int64(act_amax_part.toint())

            step_idx = Int32(0)
            for ci in cutlass.range(n_chunks, unroll=1):
                tile_c = tile0 + ci
                first_c = first0 + ci * Int32(STREAM_TILE_M)
                live_c = count_rem - ci * Int32(STREAM_TILE_M)
                if live_c > Int32(STREAM_TILE_M):
                    live_c = Int32(STREAM_TILE_M)
                active = warp_m * Int32(32) < live_c
                for v in cutlass.range_constexpr(64):
                    acc[v] = Float32(0.0)
                for k_blk in cutlass.range(_NV4_GU_K_BLOCKS, unroll=1):
                    slot = step_idx % Int32(_NV4_GU_STAGES)
                    f_phase = (step_idx // Int32(_NV4_GU_STAGES)) & Int32(1)
                    cute.arch.mbarrier_wait(full_mbar + slot, f_phase)
                    if active:
                        _nv4_gu_mma_k256(
                            sa_addr + slot * Int32(_NV4_GU_A_STAGE_BYTES),
                            sw_addr + slot * Int32(_NV4_GU_W_STAGE_BYTES),
                            ssfa_addr + slot * Int32(_NV4_GU_SFA_STAGE_BYTES),
                            ssc_addr, k_blk, acc, warp_m, warp_n, q, c, mat, mrow)
                        cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        cute.arch.mbarrier_arrive(empty_mbar + slot)
                    step_idx = step_idx + Int32(1)

                # Zero-SMEM, zero-barrier register epilogue: exact SiTU x gamma + 4-lane amax butterfly.
                if active:
                    amax_t = row_amax_base + Int64(tile_c) * Int64(STREAM_TILE_M) * Int64(4)
                    for mt in cutlass.range_constexpr(2):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + q
                            in_row = row < live_c
                            fg_scale = Float32(0.0)
                            fu_scale = Float32(0.0)
                            g_row = Float32(0.0)
                            if in_row:
                                amax_r = _ld_f32(amax_t + Int64(row) * Int64(4))
                                fg_scale = (amax_r * w_amax_g) * rcp_denom_g
                                fu_scale = (amax_r * w_amax_u) * rcp_denom_u
                                g_row = _ld_f32(gamma_base + Int64(first_c + row) * Int64(4))
                            row_i = Int64(tile_c) * Int64(STREAM_TILE_M) + Int64(row)
                            for grp_w in cutlass.range_constexpr(2):
                                nt0 = 2 * grp_w
                                nt1 = nt0 + 1
                                fg0 = 4 * (8 * mt + nt0) + 2 * half
                                fu0 = 4 * (8 * mt + nt0 + 4) + 2 * half
                                fg1 = 4 * (8 * mt + nt1) + 2 * half
                                fu1 = 4 * (8 * mt + nt1 + 4) + 2 * half
                                local = Float32(0.0)
                                if in_row:
                                    v0 = situ_vendor(_round_bf16(acc[fg0] * fg_scale),
                                                     _round_bf16(acc[fu0] * fu_scale),
                                                     g_row, gate_cap=gate_cap, up_cap=up_cap)
                                    v1 = situ_vendor(_round_bf16(acc[fg0 + 1] * fg_scale),
                                                     _round_bf16(acc[fu0 + 1] * fu_scale),
                                                     g_row, gate_cap=gate_cap, up_cap=up_cap)
                                    v2 = situ_vendor(_round_bf16(acc[fg1] * fg_scale),
                                                     _round_bf16(acc[fu1] * fu_scale),
                                                     g_row, gate_cap=gate_cap, up_cap=up_cap)
                                    v3 = situ_vendor(_round_bf16(acc[fg1 + 1] * fg_scale),
                                                     _round_bf16(acc[fu1 + 1] * fu_scale),
                                                     g_row, gate_cap=gate_cap, up_cap=up_cap)
                                    local = fmax_f32(
                                        fmax_f32(fabs_f32(v0), fabs_f32(v1)),
                                        fmax_f32(fabs_f32(v2), fabs_f32(v3)),
                                    )
                                    col_base = n0 + warp_n * Int32(32) + Int32(16 * grp_w) + Int32(2) * c
                                    out_base = act_bf16_base + (row_i * Int64(MINI_INTER) + Int64(col_base)) * Int64(2)
                                    st_global_u32(out_base, pack_f32x2_to_bfloat2(v0, v1))
                                    st_global_u32(out_base + Int64(16), pack_f32x2_to_bfloat2(v2, v3))
                                # Convergent 4-lane quad reduction (all 4 lanes c=0..3 share identical row).
                                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=1)))
                                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=2)))
                                if in_row & (c == Int32(0)):
                                    st_global_f32(
                                        act_part_base
                                        + (row_i * Int64(MINI_INTER // 16)
                                           + Int64(n0 // Int32(16) + warp_n * Int32(2) + Int32(grp_w))) * Int64(4),
                                        local,
                                    )


# ===========================================================================
# 2. NVFP4 W4A4 Down (Warp-Specialized 8 MMA + 1 DMA, 4-Stage TMA, 8.7KB Epi)
# ===========================================================================

_NV4_DN_STAGES = 4
_NV4_DN_TILE_N = 128
_NV4_DN_K_BYTES = 64
_NV4_DN_K_BLOCKS = 9
_NV4_DN_BOX_M = 32
_NV4_DN_W_STAGE_BYTES = _NV4_DN_TILE_N * _NV4_DN_K_BYTES   # 8192
_NV4_DN_A_BOX_BYTES = _NV4_DN_BOX_M * _NV4_DN_K_BYTES      # 2048
_NV4_DN_A_STAGE_BYTES = STREAM_TILE_M * _NV4_DN_K_BYTES    # 8192
_NV4_DN_SC_COLS = MINI_INTER // 16                         # 72 (18 u32 words)
_NV4_DN_SC_STRIDE = _NV4_DN_SC_COLS + 4                    # 76 (19 u32 words -> odd -> 0 conflicts)
_NV4_DN_OUT_STRIDE = _NV4_DN_TILE_N * 2 + 16               # 272 B (32 rows = 8704 B)


def mini_nvfp4_down_layouts():
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, _NV4_DN_K_BYTES),
        cutlass.Uint8)
    w_layout = cute.tile_to_shape(atom, (_NV4_DN_TILE_N, _NV4_DN_K_BYTES, _NV4_DN_STAGES), order=(0, 1, 2))
    a_layout = cute.tile_to_shape(atom, (_NV4_DN_BOX_M, _NV4_DN_K_BYTES, 4 * _NV4_DN_STAGES), order=(0, 1, 2))
    return w_layout, a_layout


def mini_nvfp4_down_storage(w_layout, a_layout):
    class MiniNvfp4DownStorage:
        pass

    MiniNvfp4DownStorage.__annotations__ = {
        "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
        "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(a_layout)], 1024],
        "sb": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, _NV4_DN_TILE_N * _NV4_DN_SC_STRIDE], 16],
        "sa_sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, STREAM_TILE_M * _NV4_DN_SC_STRIDE], 16],
        "epi": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 32 * _NV4_DN_OUT_STRIDE], 16],
        "mbar": cute.struct.MemRange[cutlass.Int64, 2 * _NV4_DN_STAGES],
    }
    return cute.struct(MiniNvfp4DownStorage)


@cute.jit
def _nv4_dn_mma_k128(a_base: Int32, w_base: Int32, sa_sc_base: Int32, sb_base: Int32,
                     k_blk: Int32, acc: cute.Tensor, warp_m: Int32, warp_n: Int32,
                     q: Int32, c: Int32, mat: Int32, mrow: Int32):
    """Warp tile 32 x 64 over K=128 (64B swizzle) with double-buffered ldmatrix fragments."""
    swz64 = mrow >> Int32(1)
    r_sfa0 = warp_m * Int32(32) + q + Int32(8) * (c & Int32(1))
    r_sfa1 = r_sfa0 + Int32(16)
    arow0 = warp_m * Int32(32) + Int32(8) * (mat & Int32(1)) + mrow
    arow1 = arow0 + Int32(16)
    bcol_base = warp_n * Int32(64) + Int32(8) * (mat >> Int32(1)) + mrow
    sb_row_base = warp_n * Int32(64) + q
    k_sc_base = k_blk * Int32(8)

    apiece0 = ((mat >> Int32(1)) ^ swz64) * Int32(16)
    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + arow0 * Int32(_NV4_DN_K_BYTES) + apiece0)
    a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(a_base + arow1 * Int32(_NV4_DN_K_BYTES) + apiece0)

    for step in cutlass.range_constexpr(2):
        k_sc = k_sc_base + Int32(4 * step)
        sfa0 = ld_shared_u32(sa_sc_base + r_sfa0 * Int32(_NV4_DN_SC_STRIDE) + k_sc)
        sfa1 = ld_shared_u32(sa_sc_base + r_sfa1 * Int32(_NV4_DN_SC_STRIDE) + k_sc)
        bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ swz64) * Int32(16)

        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + bcol_base * Int32(_NV4_DN_K_BYTES) + bpiece)
        sfb0 = ld_shared_u32(sb_base + sb_row_base * Int32(_NV4_DN_SC_STRIDE) + k_sc)
        sfb1 = ld_shared_u32(sb_base + (sb_row_base + Int32(8)) * Int32(_NV4_DN_SC_STRIDE) + k_sc)

        for pair in cutlass.range_constexpr(4):
            nb0, nb1, nb2, nb3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            nsfb0, nsfb1 = Uint32(0), Uint32(0)
            if const_expr(pair < 3):
                nb_col = bcol_base + Int32(16 * (pair + 1))
                nb0, nb1, nb2, nb3 = ldmatrix_m8n8x4_b16(w_base + nb_col * Int32(_NV4_DN_K_BYTES) + bpiece)
                nsb_r = sb_row_base + Int32(16 * (pair + 1))
                nsfb0 = ld_shared_u32(sb_base + nsb_r * Int32(_NV4_DN_SC_STRIDE) + k_sc)
                nsfb1 = ld_shared_u32(sb_base + (nsb_r + Int32(8)) * Int32(_NV4_DN_SC_STRIDE) + k_sc)
            elif const_expr(step == 0):
                napiece = ((Int32(2) + (mat >> Int32(1))) ^ swz64) * Int32(16)
                na0, na1, na2, na3 = ldmatrix_m8n8x4_b16(a_base + arow0 * Int32(_NV4_DN_K_BYTES) + napiece)
                na4, na5, na6, na7 = ldmatrix_m8n8x4_b16(a_base + arow1 * Int32(_NV4_DN_K_BYTES) + napiece)

            nt0 = 2 * pair
            nt1 = nt0 + 1
            f00 = 4 * nt0
            f10 = 4 * (8 + nt0)
            f01 = 4 * nt1
            f11 = 4 * (8 + nt1)
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3], a0, a1, a2, a3, b0, b1, sfa0, sfb0)
            acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3], a4, a5, a6, a7, b0, b1, sfa1, sfb0)
            acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3], a0, a1, a2, a3, b2, b3, sfa0, sfb1)
            acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3], a4, a5, a6, a7, b2, b3, sfa1, sfb1)
            acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3] = d0, d1, d2, d3

            if const_expr(pair < 3):
                b0, b1, b2, b3 = nb0, nb1, nb2, nb3
                sfb0, sfb1 = nsfb0, nsfb1
            elif const_expr(step == 0):
                a0, a1, a2, a3 = na0, na1, na2, na3
                a4, a5, a6, a7 = na4, na5, na6, na7


@cute.jit
def mini_nvfp4_down_body(
    a_s: cute.Pointer, meta: cute.Pointer, act_t: cute.Tensor,
    weight_t: cute.Tensor, s2: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer,
    tma_a: cute.CopyAtom, tma_w: cute.CopyAtom,
    w_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout,
    storage, amax_offset: cutlass.Constexpr, denominator_offset: cutlass.Constexpr,
):
    n_blk = Int32(cute.arch.block_idx()[0])
    group_id = Int32(cute.arch.block_idx()[1])
    tidx = Int32(cute.arch.thread_idx()[0])
    warp_id = cute.arch.make_warp_uniform(tidx // Int32(32))
    lane = tidx % Int32(32)

    sw_addr = shared_ptr_to_u32(storage.w.data_ptr())
    sa_addr = shared_ptr_to_u32(storage.a.data_ptr())
    ssb_addr = shared_ptr_to_u32(storage.sb.data_ptr())
    ssa_sc_addr = shared_ptr_to_u32(storage.sa_sc.data_ptr())
    sepi_addr = shared_ptr_to_u32(storage.epi.data_ptr())
    mbar = storage.mbar.data_ptr()
    full_mbar = mbar
    empty_mbar = mbar + _NV4_DN_STAGES
    s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
    s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
    t_ws, t_wg = cpasync.tma_partition(
        tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
        cute.group_modes(cute.local_tile(weight_t, (_NV4_DN_TILE_N, _NV4_DN_K_BYTES), (None, None)), 0, 2))
    t_as, t_ag = cpasync.tma_partition(
        tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
        cute.group_modes(cute.local_tile(act_t, (_NV4_DN_BOX_M, _NV4_DN_K_BYTES), (None, None)), 0, 2))

    e, tile0, first0, count_rem, n_chunks = _resolve_group_ljf_128(meta, group_id, tidx, warp_id, lane, sw_addr)
    if e >= Int32(0):
        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_w)
            cpasync.prefetch_descriptor(tma_a)
            for s in cutlass.range_constexpr(_NV4_DN_STAGES):
                cute.arch.mbarrier_init(full_mbar + s, 1)
                cute.arch.mbarrier_init(empty_mbar + s, 8)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        n0 = n_blk * Int32(_NV4_DN_TILE_N)
        w_tile = e * Int32(MINI_HIDDEN // _NV4_DN_TILE_N) + n_blk

        if warp_id == Int32(8):
            cute.arch.setmaxregister_decrease(MINI_DMA_REGS)
            step_idx = Int32(0)
            for ci in cutlass.range(n_chunks, unroll=1):
                tile_c = tile0 + ci
                live_c = count_rem - ci * Int32(STREAM_TILE_M)
                if live_c > Int32(STREAM_TILE_M):
                    live_c = Int32(STREAM_TILE_M)
                live_boxes = (live_c + Int32(31)) // Int32(32)
                tx_bytes = Int32(_NV4_DN_W_STAGE_BYTES) + live_boxes * Int32(_NV4_DN_A_BOX_BYTES)
                base_row_tile = tile_c * Int32(4)
                for k_blk in cutlass.range(_NV4_DN_K_BLOCKS, unroll=1):
                    slot = step_idx % Int32(_NV4_DN_STAGES)
                    if step_idx >= Int32(_NV4_DN_STAGES):
                        e_phase = ((step_idx - Int32(_NV4_DN_STAGES)) // Int32(_NV4_DN_STAGES)) & Int32(1)
                        cute.arch.mbarrier_wait(empty_mbar + slot, e_phase)
                    barrier = full_mbar + slot
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_bytes)
                    cute.copy(tma_w, t_wg[(None, w_tile, k_blk)], t_ws[(None, slot)], tma_bar_ptr=barrier)
                    base_slot = slot * Int32(4)
                    for b in cutlass.range_constexpr(4):
                        if Int32(b) < live_boxes:
                            cute.copy(tma_a, t_ag[(None, base_row_tile + Int32(b), k_blk)],
                                      t_as[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)
                    step_idx = step_idx + Int32(1)
        else:
            cute.arch.setmaxregister_increase(MINI_MMA_REGS)
            # Preload s2 (128 rows x 18 u32 words = 2304 words = 9 * 256) coalesced ONCE per CTA.
            sc_words = _NV4_DN_SC_COLS // 4  # 18
            s2_base = Int64(s2.toint())
            s2_tile_base = s2_base + (Int64(e) * Int64(MINI_HIDDEN) + Int64(n0)) * Int64(_NV4_DN_SC_COLS)
            for qsb in cutlass.range_constexpr(_NV4_DN_TILE_N * sc_words // MINI_MMA_THREADS):
                u = tidx + Int32(qsb * MINI_MMA_THREADS)
                r = u // Int32(sc_words)
                cw = u % Int32(sc_words)
                cp_async_u32_shared_global(
                    ssb_addr + r * Int32(_NV4_DN_SC_STRIDE) + cw * Int32(4),
                    s2_tile_base + Int64(r) * Int64(_NV4_DN_SC_COLS) + Int64(cw) * Int64(4))
            cute.arch.cp_async_commit_group()

            w_amax = _ld_f32(s2_base + Int64(amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(Float32(1.0), Float32(1536.0) * _ld_f32(s2_base + Int64(denominator_offset)))

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            q = lane // Int32(4)
            c = lane % Int32(4)
            mat = lane // Int32(8)
            mrow = lane % Int32(8)

            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            as_base = Int64(a_s.toint())
            row_amax_base = Int64(row_amax.toint())
            y_base = Int64(y.toint())

            step_idx = Int32(0)
            for ci in cutlass.range(n_chunks, unroll=1):
                tile_c = tile0 + ci
                first_c = first0 + ci * Int32(STREAM_TILE_M)
                live_c = count_rem - ci * Int32(STREAM_TILE_M)
                if live_c > Int32(STREAM_TILE_M):
                    live_c = Int32(STREAM_TILE_M)
                active = warp_m * Int32(32) < live_c

                # Preload chunk's a_s scales (live_c x 18 u32 words) coalesced into SMEM.
                as_tile_base = as_base + Int64(tile_c) * Int64(STREAM_TILE_M) * Int64(_NV4_DN_SC_COLS)
                for qsa in cutlass.range_constexpr(STREAM_TILE_M * sc_words // MINI_MMA_THREADS):
                    u = tidx + Int32(qsa * MINI_MMA_THREADS)
                    r = u // Int32(sc_words)
                    cw = u % Int32(sc_words)
                    if r < live_c:
                        cp_async_u32_shared_global(
                            ssa_sc_addr + r * Int32(_NV4_DN_SC_STRIDE) + cw * Int32(4),
                            as_tile_base + Int64(r) * Int64(_NV4_DN_SC_COLS) + Int64(cw) * Int64(4))
                cute.arch.cp_async_commit_group()
                for v in cutlass.range_constexpr(64):
                    acc[v] = Float32(0.0)
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)

                for k_blk in cutlass.range(_NV4_DN_K_BLOCKS, unroll=1):
                    slot = step_idx % Int32(_NV4_DN_STAGES)
                    f_phase = (step_idx // Int32(_NV4_DN_STAGES)) & Int32(1)
                    cute.arch.mbarrier_wait(full_mbar + slot, f_phase)
                    if active:
                        _nv4_dn_mma_k128(
                            sa_addr + slot * Int32(_NV4_DN_A_STAGE_BYTES),
                            sw_addr + slot * Int32(_NV4_DN_W_STAGE_BYTES),
                            ssa_sc_addr, ssb_addr, k_blk, acc, warp_m, warp_n, q, c, mat, mrow)
                        cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        cute.arch.mbarrier_arrive(empty_mbar + slot)
                    step_idx = step_idx + Int32(1)

                # Coalesced 16B BF16 epilogue via 32-row dedicated SMEM buffer (4 passes, MMA-only barrier).
                amax_t = row_amax_base + Int64(tile_c) * Int64(STREAM_TILE_M) * Int64(4)
                segments = _NV4_DN_TILE_N * 2 // 16  # 16 segments of 16B per row
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + q
                        if row < live_c:
                            factor = (_ld_f32(amax_t + Int64(row) * Int64(4)) * w_amax) * rcp_denom
                            srow = warp_m * Int32(8) + q
                            for nt in cutlass.range_constexpr(8):
                                f = 4 * (8 * mt + nt) + 2 * half
                                col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * c
                                st_shared_u32(sepi_addr + srow * Int32(_NV4_DN_OUT_STRIDE) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
                        cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)
                        # 32 rows x 16 segments = 512 segments / 256 threads = 2 coalesced 16B stores/thread.
                        for qit in cutlass.range_constexpr(32 * segments // MINI_MMA_THREADS):
                            ci_s = tidx + Int32(MINI_MMA_THREADS * qit)
                            srow_o = ci_s // Int32(segments)
                            seg = ci_s % Int32(segments)
                            wm_o = srow_o // Int32(8)
                            q_o = srow_o % Int32(8)
                            row_o = wm_o * Int32(32) + Int32(16 * mt + 8 * half) + q_o
                            if row_o < live_c:
                                v0, v1, v2, v3 = ld_shared_v4_u32(
                                    sepi_addr + srow_o * Int32(_NV4_DN_OUT_STRIDE) + seg * Int32(16))
                                st_global_v4_u32(
                                    y_base + ((Int64(first_c) + Int64(row_o)) * Int64(MINI_HIDDEN) + Int64(n0)) * Int64(2)
                                    + Int64(seg) * Int64(16),
                                    v0, v1, v2, v3,
                                )
                        cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)


# ===========================================================================
# 3. MXFP8 W8A8 Down (Warp-Specialized 8 MMA + 1 DMA, 2-Stage TMA, 8.7KB Epi)
# ===========================================================================

_MX8_DN_STAGES = 2
_MX8_DN_TILE_N = 128
_MX8_DN_K_BLOCK = 128
_MX8_DN_BLOCKS = MINI_INTER // _MX8_DN_K_BLOCK             # 9
_MX8_DN_BOX_M = 32
_MX8_DN_ROW_BYTES = mxfp8_down_row_bytes(MINI_INTER)       # 1200 B
_MX8_DN_SC_WORDS = _MX8_DN_BLOCKS                          # 9 (odd -> 0 bank conflicts!)
_MX8_DN_SC_STRIDE = _MX8_DN_SC_WORDS * 4                   # 36 B
_MX8_DN_W_STAGE_BYTES = _MX8_DN_TILE_N * _MX8_DN_K_BLOCK   # 16384
_MX8_DN_A_BOX_BYTES = _MX8_DN_BOX_M * _MX8_DN_K_BLOCK      # 4096
_MX8_DN_A_STAGE_BYTES = STREAM_TILE_M * _MX8_DN_K_BLOCK    # 16384
_MX8_DN_OUT_STRIDE = _MX8_DN_TILE_N * 2 + 16               # 272 B


def mini_mxfp8_down_layouts():
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, _MX8_DN_K_BLOCK),
        cutlass.Uint8)
    w_layout = cute.tile_to_shape(atom, (_MX8_DN_TILE_N, _MX8_DN_K_BLOCK, _MX8_DN_STAGES), order=(0, 1, 2))
    a_layout = cute.tile_to_shape(atom, (_MX8_DN_BOX_M, _MX8_DN_K_BLOCK, 4 * _MX8_DN_STAGES), order=(0, 1, 2))
    return w_layout, a_layout


def mini_mxfp8_down_storage(w_layout, a_layout):
    class MiniMxfp8DownStorage:
        pass

    MiniMxfp8DownStorage.__annotations__ = {
        "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
        "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(a_layout)], 1024],
        "sb": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, _MX8_DN_TILE_N * _MX8_DN_SC_STRIDE], 16],
        "sa_sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, STREAM_TILE_M * _MX8_DN_SC_STRIDE], 16],
        "epi": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 32 * _MX8_DN_OUT_STRIDE], 16],
        "mbar": cute.struct.MemRange[cutlass.Int64, 2 * _MX8_DN_STAGES],
    }
    return cute.struct(MiniMxfp8DownStorage)


@cute.jit
def _mx8_dn_mma_k128(block: Int32, a_base: Int32, w_base: Int32, sa_sc_base: Int32, sb_base: Int32,
                     acc: cute.Tensor, warp_m: Int32, warp_n: Int32, lane: Int32, g: Int32):
    """Warp tile 32 x 64 over K=128 with hoisted scales and double-buffered ldmatrix fragments."""
    srow0 = warp_m * Int32(32) + g + (lane & Int32(1)) * Int32(8)
    asc0 = ld_shared_u32(sa_sc_base + srow0 * Int32(_MX8_DN_SC_STRIDE) + block * Int32(4))
    asc1 = ld_shared_u32(sa_sc_base + (srow0 + Int32(16)) * Int32(_MX8_DN_SC_STRIDE) + block * Int32(4))
    # Hoist all 8 B scale words ONCE per K=128 block (4x fewer LDS than per-ks reload).
    bsc = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
    for nt in cutlass.range_constexpr(8):
        col = warp_n * Int32(64) + Int32(8 * nt) + g
        bsc[nt] = ld_shared_u32(sb_base + col * Int32(_MX8_DN_SC_STRIDE) + block * Int32(4))

    row0 = warp_m * Int32(32) + (lane % Int32(16))
    row0_swz = row0 % Int32(8)
    row1 = row0 + Int32(16)
    nrow_base = warp_n * Int32(64) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
    nrow_swz = nrow_base % Int32(8)

    chunk_a0 = lane // Int32(16)
    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + row0 * Int32(_MX8_DN_K_BLOCK) + ((chunk_a0 ^ row0_swz) * Int32(16)))
    a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(a_base + row1 * Int32(_MX8_DN_K_BLOCK) + ((chunk_a0 ^ row0_swz) * Int32(16)))

    for ks in cutlass.range_constexpr(4):
        chunk_b = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
        b_swz = (chunk_b ^ nrow_swz) * Int32(16)
        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + nrow_base * Int32(_MX8_DN_K_BLOCK) + b_swz)

        for np in cutlass.range_constexpr(4):
            nb0, nb1, nb2, nb3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if const_expr(np < 3):
                nb0, nb1, nb2, nb3 = ldmatrix_m8n8x4_b16(
                    w_base + (nrow_base + Int32(16 * (np + 1))) * Int32(_MX8_DN_K_BLOCK) + b_swz)
            elif const_expr(ks < 3):
                nchunk_a = Int32(2 * (ks + 1)) + lane // Int32(16)
                na_swz = (nchunk_a ^ row0_swz) * Int32(16)
                na0, na1, na2, na3 = ldmatrix_m8n8x4_b16(a_base + row0 * Int32(_MX8_DN_K_BLOCK) + na_swz)
                na4, na5, na6, na7 = ldmatrix_m8n8x4_b16(a_base + row1 * Int32(_MX8_DN_K_BLOCK) + na_swz)

            nt0 = 2 * np
            nt1 = nt0 + 1
            f00 = 4 * nt0
            f10 = 4 * (8 + nt0)
            f01 = 4 * nt1
            f11 = 4 * (8 + nt1)
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3],
                a0, a1, a2, a3, b0, b1, asc0, bsc[nt0], bid_a=ks, bid_b=ks)
            acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3],
                a4, a5, a6, a7, b0, b1, asc1, bsc[nt0], bid_a=ks, bid_b=ks)
            acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3],
                a0, a1, a2, a3, b2, b3, asc0, bsc[nt1], bid_a=ks, bid_b=ks)
            acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3],
                a4, a5, a6, a7, b2, b3, asc1, bsc[nt1], bid_a=ks, bid_b=ks)
            acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3] = d0, d1, d2, d3

            if const_expr(np < 3):
                b0, b1, b2, b3 = nb0, nb1, nb2, nb3
            elif const_expr(ks < 3):
                a0, a1, a2, a3 = na0, na1, na2, na3
                a4, a5, a6, a7 = na4, na5, na6, na7


@cute.jit
def mini_mxfp8_down_body(
    act: cute.Pointer, meta: cute.Pointer, act_t: cute.Tensor,
    weight_t: cute.Tensor, s2: cute.Pointer, y: cute.Pointer,
    tma_a: cute.CopyAtom, tma_w: cute.CopyAtom,
    w_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout, storage,
):
    n_blk = Int32(cute.arch.block_idx()[0])
    group_id = Int32(cute.arch.block_idx()[1])
    tidx = Int32(cute.arch.thread_idx()[0])
    warp_id = cute.arch.make_warp_uniform(tidx // Int32(32))
    lane = tidx % Int32(32)

    sw_addr = shared_ptr_to_u32(storage.w.data_ptr())
    sa_addr = shared_ptr_to_u32(storage.a.data_ptr())
    ssb_addr = shared_ptr_to_u32(storage.sb.data_ptr())
    ssa_sc_addr = shared_ptr_to_u32(storage.sa_sc.data_ptr())
    sepi_addr = shared_ptr_to_u32(storage.epi.data_ptr())
    mbar = storage.mbar.data_ptr()
    full_mbar = mbar
    empty_mbar = mbar + _MX8_DN_STAGES
    s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
    s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
    t_ws, t_wg = cpasync.tma_partition(
        tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
        cute.group_modes(cute.local_tile(weight_t, (_MX8_DN_TILE_N, _MX8_DN_K_BLOCK), (None, None)), 0, 2))
    t_as, t_ag = cpasync.tma_partition(
        tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
        cute.group_modes(cute.local_tile(act_t, (_MX8_DN_BOX_M, _MX8_DN_K_BLOCK), (None, None)), 0, 2))

    e, tile0, first0, count_rem, n_chunks = _resolve_group_ljf_128(meta, group_id, tidx, warp_id, lane, sw_addr)
    if e >= Int32(0):
        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_w)
            cpasync.prefetch_descriptor(tma_a)
            for s in cutlass.range_constexpr(_MX8_DN_STAGES):
                cute.arch.mbarrier_init(full_mbar + s, 1)
                cute.arch.mbarrier_init(empty_mbar + s, 8)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        n0 = n_blk * Int32(_MX8_DN_TILE_N)
        w_tile = e * Int32(MINI_HIDDEN // _MX8_DN_TILE_N) + n_blk

        if warp_id == Int32(8):
            cute.arch.setmaxregister_decrease(MINI_DMA_REGS)
            step_idx = Int32(0)
            for ci in cutlass.range(n_chunks, unroll=1):
                tile_c = tile0 + ci
                live_c = count_rem - ci * Int32(STREAM_TILE_M)
                if live_c > Int32(STREAM_TILE_M):
                    live_c = Int32(STREAM_TILE_M)
                live_boxes = (live_c + Int32(31)) // Int32(32)
                tx_bytes = Int32(_MX8_DN_W_STAGE_BYTES) + live_boxes * Int32(_MX8_DN_A_BOX_BYTES)
                base_row_tile = tile_c * Int32(4)
                for block in cutlass.range(_MX8_DN_BLOCKS, unroll=1):
                    slot = step_idx % Int32(_MX8_DN_STAGES)
                    if step_idx >= Int32(_MX8_DN_STAGES):
                        e_phase = ((step_idx - Int32(_MX8_DN_STAGES)) // Int32(_MX8_DN_STAGES)) & Int32(1)
                        cute.arch.mbarrier_wait(empty_mbar + slot, e_phase)
                    barrier = full_mbar + slot
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_bytes)
                    cute.copy(tma_w, t_wg[(None, w_tile, block)], t_ws[(None, slot)], tma_bar_ptr=barrier)
                    base_slot = slot * Int32(4)
                    for b in cutlass.range_constexpr(4):
                        if Int32(b) < live_boxes:
                            cute.copy(tma_a, t_ag[(None, base_row_tile + Int32(b), block)],
                                      t_as[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)
                    step_idx = step_idx + Int32(1)
        else:
            cute.arch.setmaxregister_increase(MINI_MMA_REGS)
            # Preload s2 (128 x 9 u32 words = 1152 words) coalesced ONCE per CTA.
            total_sb = _MX8_DN_TILE_N * _MX8_DN_SC_WORDS  # 1152
            s2_tile_base = Int64(s2.toint()) + (Int64(e) * Int64(MINI_HIDDEN) + Int64(n0)) * Int64(_MX8_DN_SC_STRIDE)
            for qsb in cutlass.range_constexpr((total_sb + MINI_MMA_THREADS - 1) // MINI_MMA_THREADS):
                u = tidx + Int32(qsb * MINI_MMA_THREADS)
                if u < Int32(total_sb):
                    r = u // Int32(_MX8_DN_SC_WORDS)
                    cw = u % Int32(_MX8_DN_SC_WORDS)
                    cp_async_u32_shared_global(
                        ssb_addr + r * Int32(_MX8_DN_SC_STRIDE) + cw * Int32(4),
                        s2_tile_base + Int64(r) * Int64(_MX8_DN_SC_STRIDE) + Int64(cw) * Int64(4))
            cute.arch.cp_async_commit_group()

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            g = lane // Int32(4)
            j = lane % Int32(4)

            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            act_base = Int64(act.toint())
            y_base = Int64(y.toint())

            step_idx = Int32(0)
            for ci in cutlass.range(n_chunks, unroll=1):
                tile_c = tile0 + ci
                first_c = first0 + ci * Int32(STREAM_TILE_M)
                live_c = count_rem - ci * Int32(STREAM_TILE_M)
                if live_c > Int32(STREAM_TILE_M):
                    live_c = Int32(STREAM_TILE_M)
                active = warp_m * Int32(32) < live_c

                # Preload chunk's act UE8M0 scales (live_c x 9 u32 words) coalesced into SMEM.
                act_sc_tile = act_base + Int64(tile_c) * Int64(STREAM_TILE_M) * Int64(_MX8_DN_ROW_BYTES) + Int64(MINI_INTER)
                for qsa in cutlass.range_constexpr((total_sb + MINI_MMA_THREADS - 1) // MINI_MMA_THREADS):
                    u = tidx + Int32(qsa * MINI_MMA_THREADS)
                    if u < Int32(total_sb):
                        r = u // Int32(_MX8_DN_SC_WORDS)
                        cw = u % Int32(_MX8_DN_SC_WORDS)
                        if r < live_c:
                            cp_async_u32_shared_global(
                                ssa_sc_addr + r * Int32(_MX8_DN_SC_STRIDE) + cw * Int32(4),
                                act_sc_tile + Int64(r) * Int64(_MX8_DN_ROW_BYTES) + Int64(cw) * Int64(4))
                cute.arch.cp_async_commit_group()
                for v in cutlass.range_constexpr(64):
                    acc[v] = Float32(0.0)
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)

                for block in cutlass.range(_MX8_DN_BLOCKS, unroll=1):
                    slot = step_idx % Int32(_MX8_DN_STAGES)
                    f_phase = (step_idx // Int32(_MX8_DN_STAGES)) & Int32(1)
                    cute.arch.mbarrier_wait(full_mbar + slot, f_phase)
                    if active:
                        _mx8_dn_mma_k128(
                            block, sa_addr + slot * Int32(_MX8_DN_A_STAGE_BYTES),
                            sw_addr + slot * Int32(_MX8_DN_W_STAGE_BYTES),
                            ssa_sc_addr, ssb_addr, acc, warp_m, warp_n, lane, g)
                        cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        cute.arch.mbarrier_arrive(empty_mbar + slot)
                    step_idx = step_idx + Int32(1)

                # Coalesced 16B BF16 epilogue via 32-row dedicated SMEM buffer (4 passes, MMA-only barrier).
                segments = _MX8_DN_TILE_N * 2 // 16
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live_c:
                            srow = warp_m * Int32(8) + g
                            for nt in cutlass.range_constexpr(8):
                                f = 4 * (8 * mt + nt) + 2 * half
                                col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                                st_shared_u32(sepi_addr + srow * Int32(_MX8_DN_OUT_STRIDE) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
                        cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)
                        for qit in cutlass.range_constexpr(32 * segments // MINI_MMA_THREADS):
                            ci_s = tidx + Int32(MINI_MMA_THREADS * qit)
                            srow_o = ci_s // Int32(segments)
                            seg = ci_s % Int32(segments)
                            wm_o = srow_o // Int32(8)
                            q_o = srow_o % Int32(8)
                            row_o = wm_o * Int32(32) + Int32(16 * mt + 8 * half) + q_o
                            if row_o < live_c:
                                v0, v1, v2, v3 = ld_shared_v4_u32(
                                    sepi_addr + srow_o * Int32(_MX8_DN_OUT_STRIDE) + seg * Int32(16))
                                st_global_v4_u32(
                                    y_base + ((Int64(first_c) + Int64(row_o)) * Int64(MINI_HIDDEN) + Int64(n0)) * Int64(2)
                                    + Int64(seg) * Int64(16),
                                    v0, v1, v2, v3,
                                )
                        cute.arch.barrier(barrier_id=1, number_of_threads=MINI_MMA_THREADS)


# ===========================================================================
# 4. MXFP8 W8A8 GateUp (BM=128 Wide-N Expert-Looping, w_ring=3, a_ring=2)
# ===========================================================================

_MX8_GU_COLS = 64
_MX8_GU_K_STEP = 128
_MX8_GU_K_STEPS = MINI_HIDDEN // _MX8_GU_K_STEP            # 18
_MX8_GU_W_RING = 3
_MX8_GU_A_RING = 2
_MX8_GU_ROW_BYTES = MINI_HIDDEN + MINI_HIDDEN // 32        # 2376 B (8 mod 16)
_MX8_GU_HALF_BYTES = _MX8_GU_COLS * _MX8_GU_K_STEP         # 8192 B
_MX8_GU_A_BYTES = STREAM_TILE_M * _MX8_GU_K_STEP           # 16384 B
_MX8_GU_S_BYTES = STREAM_TILE_M * 4                        # 512 B
_MX8_GU_SC_COLS = MINI_HIDDEN // 32                        # 72 B (18 u32 words)
_MX8_GU_SC_STRIDE = _MX8_GU_SC_COLS + 4                    # 76 B (19 u32 words -> odd -> 0 conflicts)
_MX8_GU_ACT_ROW_BYTES = mxfp8_down_row_bytes(MINI_INTER)   # 1200 B
_MX8_GU_ACT_STRIDE = _MX8_GU_COLS * 2 + 16                 # 144 B (128 * 144 = 18432 B <= 32768 B)


def mini_mxfp8_gate_up_layout():
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, _MX8_GU_K_STEP),
        cutlass.Uint8)
    return cute.tile_to_shape(atom, (_MX8_GU_COLS, _MX8_GU_K_STEP, 2 * _MX8_GU_W_RING), order=(0, 1, 2))


def mini_mxfp8_gate_up_storage(w_layout):
    class MiniMxfp8GateUpStorage:
        pass

    MiniMxfp8GateUpStorage.__annotations__ = {
        "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
        "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, _MX8_GU_A_RING * _MX8_GU_A_BYTES], 1024],
        "s": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, _MX8_GU_A_RING * _MX8_GU_S_BYTES], 16],
        "sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * _MX8_GU_COLS * _MX8_GU_SC_STRIDE], 16],
        "mbar": cute.struct.MemRange[cutlass.Int64, _MX8_GU_W_RING],
    }
    return cute.struct(MiniMxfp8GateUpStorage)


@cute.jit
def _mx8_gu_issue_w(w_seq: Int32, k_step: Int32, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile: Int32):
    slot = w_seq % Int32(_MX8_GU_W_RING)
    barrier = mbar + slot
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * _MX8_GU_HALF_BYTES)
    cute.copy(tma_g, t_wg[(None, w_tile, k_step)], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
    cute.copy(tma_u, t_wu[(None, w_tile, k_step)], t_ws[(None, slot * Int32(2) + Int32(1))], tma_bar_ptr=barrier)


@cute.jit
def _mx8_gu_load_a(k_step: Int32, stage: Int32, sa: Int32, ss: Int32,
                   src: cute.Tensor, s_src: Int64, ch: Int32, r_base: Int32, tidx: Int32):
    k_off = Int64(k_step) * Int64(_MX8_GU_K_STEP)
    a_stage = sa + stage * Int32(_MX8_GU_A_BYTES)
    for q in cutlass.range_constexpr(4):
        addr = src[q]
        if addr >= Int64(0):
            row = r_base + Int32(32 * q)
            dst = a_stage + row * Int32(_MX8_GU_K_STEP) + ((ch ^ (row % Int32(8))) * Int32(16))
            p = addr + k_off
            if (p & Int64(8)) == Int64(0):
                cp_async4_shared_global(dst, p)
            else:
                cp_async_u64_shared_global(dst, p)
                cp_async_u64_shared_global(dst + Int32(8), p + Int64(8))
    if s_src >= Int64(0):
        cp_async_u32_shared_global(ss + stage * Int32(_MX8_GU_S_BYTES) + tidx * Int32(4),
                                   s_src + Int64(k_step) * Int64(4))


@cute.jit
def _mx8_gu_mma_k128(k_step: Int32, a_stage: Int32, w_slot: Int32, sa: Int32, sw: Int32,
                     ss: Int32, sc: Int32, warp_m: Int32, warp_n: Int32, lane: Int32, g: Int32,
                     acc: cute.Tensor):
    """Warp tile 32 x (32 gate + 32 up) with hoisted wsc and double-buffered ldmatrix fragments."""
    a_base = sa + a_stage * Int32(_MX8_GU_A_BYTES)
    w_base = sw + w_slot * Int32(2 * _MX8_GU_HALF_BYTES)
    s_base = ss + a_stage * Int32(_MX8_GU_S_BYTES)
    srow0 = warp_m * Int32(32) + g + (lane & Int32(1)) * Int32(8)
    asc0 = ld_shared_u32(s_base + srow0 * Int32(4))
    asc1 = ld_shared_u32(s_base + (srow0 + Int32(16)) * Int32(4))

    # Hoist all 8 weight scale words ONCE per K=128 step (4x fewer LDS than per-ks reload).
    wsc = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
    for side in cutlass.range_constexpr(2):
        for nt_s in cutlass.range_constexpr(4):
            col = Int32(side * _MX8_GU_COLS + 8 * nt_s) + warp_n * Int32(32) + g
            wsc[4 * side + nt_s] = ld_shared_u32(sc + col * Int32(_MX8_GU_SC_STRIDE) + k_step * Int32(4))

    row0 = warp_m * Int32(32) + (lane % Int32(16))
    row0_swz = row0 % Int32(8)
    row1 = row0 + Int32(16)
    nrow_base = warp_n * Int32(32) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
    nrow_swz = nrow_base % Int32(8)

    chunk_a0 = lane // Int32(16)
    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + row0 * Int32(_MX8_GU_K_STEP) + ((chunk_a0 ^ row0_swz) * Int32(16)))
    a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(a_base + row1 * Int32(_MX8_GU_K_STEP) + ((chunk_a0 ^ row0_swz) * Int32(16)))

    for ks in cutlass.range_constexpr(4):
        chunk_b = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
        b_swz = (chunk_b ^ nrow_swz) * Int32(16)
        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + nrow_base * Int32(_MX8_GU_K_STEP) + b_swz)

        for idx in cutlass.range_constexpr(4):
            nb0, nb1, nb2, nb3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if const_expr(idx < 3):
                nxt_side = (idx + 1) // 2
                nxt_np = (idx + 1) % 2
                nb0, nb1, nb2, nb3 = ldmatrix_m8n8x4_b16(
                    w_base + Int32(nxt_side * _MX8_GU_HALF_BYTES)
                    + (nrow_base + Int32(16 * nxt_np)) * Int32(_MX8_GU_K_STEP) + b_swz)
            elif const_expr(ks < 3):
                nchunk_a = Int32(2 * (ks + 1)) + lane // Int32(16)
                na_swz = (nchunk_a ^ row0_swz) * Int32(16)
                na0, na1, na2, na3 = ldmatrix_m8n8x4_b16(a_base + row0 * Int32(_MX8_GU_K_STEP) + na_swz)
                na4, na5, na6, na7 = ldmatrix_m8n8x4_b16(a_base + row1 * Int32(_MX8_GU_K_STEP) + na_swz)

            nt0 = 2 * idx
            nt1 = nt0 + 1
            f00 = 4 * nt0
            f10 = 4 * (8 + nt0)
            f01 = 4 * nt1
            f11 = 4 * (8 + nt1)
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3],
                a0, a1, a2, a3, b0, b1, asc0, wsc[nt0], bid_a=ks, bid_b=ks)
            acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3],
                a4, a5, a6, a7, b0, b1, asc1, wsc[nt0], bid_a=ks, bid_b=ks)
            acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3],
                a0, a1, a2, a3, b2, b3, asc0, wsc[nt1], bid_a=ks, bid_b=ks)
            acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3] = d0, d1, d2, d3
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3],
                a4, a5, a6, a7, b2, b3, asc1, wsc[nt1], bid_a=ks, bid_b=ks)
            acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3] = d0, d1, d2, d3

            if const_expr(idx < 3):
                b0, b1, b2, b3 = nb0, nb1, nb2, nb3
            elif const_expr(ks < 3):
                a0, a1, a2, a3 = na0, na1, na2, na3
                a4, a5, a6, a7 = na4, na5, na6, na7


@cute.jit
def mini_mxfp8_gate_up_body(
    x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
    weight_g: cute.Tensor, weight_u: cute.Tensor, s1: cute.Pointer, s3: cute.Pointer,
    act: cute.Pointer, tma_g: cute.CopyAtom, tma_u: cute.CopyAtom,
    w_layout: cute.ComposedLayout, storage, grouped_gamma: cute.Pointer,
    gate_cap: cutlass.Constexpr, up_cap: cutlass.Constexpr,
):
    n_blk = Int32(cute.arch.block_idx()[0])
    group_id = Int32(cute.arch.block_idx()[1])
    tidx = Int32(cute.arch.thread_idx()[0])
    warp_id = tidx // Int32(32)
    lane = tidx % Int32(32)

    sw_addr = shared_ptr_to_u32(storage.w.data_ptr())
    sa_addr = shared_ptr_to_u32(storage.a.data_ptr())
    ss_addr = shared_ptr_to_u32(storage.s.data_ptr())
    ssc_addr = shared_ptr_to_u32(storage.sc.data_ptr())
    mbar = storage.mbar.data_ptr()
    s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)

    e, tile0, first0, count_rem, n_chunks = _resolve_group_ljf_128(meta, group_id, tidx, warp_id, lane, sw_addr)
    if e >= Int32(0):
        n0 = n_blk * Int32(_MX8_GU_COLS)
        w_tile = e * Int32(MINI_INTER // _MX8_GU_COLS) + n_blk
        t_ws, t_wg = cpasync.tma_partition(
            tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
            cute.group_modes(cute.local_tile(weight_g, (_MX8_GU_COLS, _MX8_GU_K_STEP), (None, None)), 0, 2))
        _, t_wu = cpasync.tma_partition(
            tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
            cute.group_modes(cute.local_tile(weight_u, (_MX8_GU_COLS, _MX8_GU_K_STEP), (None, None)), 0, 2))
        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_g)
            cpasync.prefetch_descriptor(tma_u)
            for slot in cutlass.range_constexpr(_MX8_GU_W_RING):
                cute.arch.mbarrier_init(mbar + slot, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Preload all 128 weight scale rows (64 gate + 64 up) x 18 u32 words coalesced ONCE per CTA.
        sc_words = _MX8_GU_SC_COLS // 4  # 18
        total_sc = 2 * _MX8_GU_COLS * sc_words  # 2304 = 9 * 256
        s1_base = Int64(s1.toint())
        s3_base = Int64(s3.toint())
        e_n0_off = Int64(e) * Int64(MINI_INTER) + Int64(n0)
        for qsc in cutlass.range_constexpr(total_sc // MINI_MMA_THREADS):
            u = tidx + Int32(qsc * MINI_MMA_THREADS)
            r = u // Int32(sc_words)
            cw = u % Int32(sc_words)
            base_s = cutlass.select_(r < Int32(_MX8_GU_COLS), s1_base, s3_base)
            col_s = cutlass.select_(r < Int32(_MX8_GU_COLS), r, r - Int32(_MX8_GU_COLS))
            cp_async_u32_shared_global(
                ssc_addr + r * Int32(_MX8_GU_SC_STRIDE) + cw * Int32(4),
                base_s + (e_n0_off + Int64(col_s)) * Int64(_MX8_GU_SC_COLS) + Int64(cw) * Int64(4))

        # Kick off chunk 0's initial W TMA pulls (steps 0, 1) immediately.
        if warp_id == Int32(0):
            for pre in cutlass.range_constexpr(_MX8_GU_W_RING - 1):
                _mx8_gu_issue_w(Int32(pre), Int32(pre), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)

        ch = tidx % Int32(8)
        r_base = tidx // Int32(8)
        x_base = Int64(x.toint())
        pr_base = Int64(pair_row.toint())
        gamma_base = Int64(grouped_gamma.toint())
        act_base = Int64(act.toint())
        warp_m = warp_id // Int32(2)
        warp_n = warp_id % Int32(2)
        g = lane // Int32(4)
        j = lane % Int32(4)

        src = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
        acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
        total_w_steps = n_chunks * Int32(_MX8_GU_K_STEPS)
        w_seq = Int32(0)

        for ci in cutlass.range(n_chunks, unroll=1):
            tile_c = tile0 + ci
            first_c = first0 + ci * Int32(STREAM_TILE_M)
            live_c = count_rem - ci * Int32(STREAM_TILE_M)
            if live_c > Int32(STREAM_TILE_M):
                live_c = Int32(STREAM_TILE_M)
            active = warp_m * Int32(32) < live_c

            for q in cutlass.range_constexpr(4):
                row = r_base + Int32(32 * q)
                src[q] = Int64(-1)
                if row < live_c:
                    source = _i32_at(pr_base, first_c + row)
                    src[q] = x_base + Int64(source) * Int64(_MX8_GU_ROW_BYTES) + Int64(ch) * Int64(16)
            s_src = Int64(-1)
            if tidx < live_c:
                source = _i32_at(pr_base, first_c + tidx)
                s_src = x_base + Int64(source) * Int64(_MX8_GU_ROW_BYTES) + Int64(MINI_HIDDEN)

            # A[0] joins sc cp.async group on ci=0; starts fresh on ci>0 after epilogue sync.
            _mx8_gu_load_a(Int32(0), Int32(0), sa_addr, ss_addr, src, s_src, ch, r_base, tidx)
            cute.arch.cp_async_commit_group()
            for v in cutlass.range_constexpr(64):
                acc[v] = Float32(0.0)

            for k_step in cutlass.range(_MX8_GU_K_STEPS, unroll=1):
                w_slot = w_seq % Int32(_MX8_GU_W_RING)
                w_phase = (w_seq // Int32(_MX8_GU_W_RING)) & Int32(1)
                a_stage = k_step % Int32(_MX8_GU_A_RING)
                cute.arch.cp_async_wait_group(0)
                cute.arch.mbarrier_wait(mbar + w_slot, w_phase)
                cute.arch.sync_threads()
                if k_step + Int32(1) < Int32(_MX8_GU_K_STEPS):
                    _mx8_gu_load_a(k_step + Int32(1), (k_step + Int32(1)) % Int32(_MX8_GU_A_RING),
                                   sa_addr, ss_addr, src, s_src, ch, r_base, tidx)
                    cute.arch.cp_async_commit_group()
                if warp_id == Int32(0):
                    nxt_w = w_seq + Int32(_MX8_GU_W_RING - 1)
                    if nxt_w < total_w_steps:
                        _mx8_gu_issue_w(nxt_w, nxt_w % Int32(_MX8_GU_K_STEPS),
                                        tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                if active:
                    _mx8_gu_mma_k128(k_step, a_stage, w_slot, sa_addr, sw_addr, ss_addr, ssc_addr,
                                     warp_m, warp_n, lane, g, acc)
                    cute.arch.fence_view_async_shared()
                w_seq = w_seq + Int32(1)

            # Note: W ring already has chunk ci+1's k=0,1 in flight while sa is drained for epilogue!
            cute.arch.sync_threads()
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live_c:
                            g_val = _ld_f32(gamma_base + Int64(first_c + row) * Int64(4))
                            for nt in cutlass.range_constexpr(4):
                                fg = 4 * (8 * mt + nt) + 2 * half
                                fu = 4 * (8 * mt + nt + 4) + 2 * half
                                col = warp_n * Int32(32) + Int32(8 * nt) + Int32(2) * j
                                v0 = _round_bf16(
                                    gated_activation_value(_round_bf16(acc[fg]), _round_bf16(acc[fu]),
                                                           is_situ=True, situ_beta=gate_cap,
                                                           situ_linear_beta=up_cap) * g_val)
                                v1 = _round_bf16(
                                    gated_activation_value(_round_bf16(acc[fg + 1]), _round_bf16(acc[fu + 1]),
                                                           is_situ=True, situ_beta=gate_cap,
                                                           situ_linear_beta=up_cap) * g_val)
                                st_shared_u32(sa_addr + row * Int32(_MX8_GU_ACT_STRIDE) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(v0, v1))
            cute.arch.sync_threads()
            # All 256 threads quantize 128 rows x 2 K32 blocks = 256 blocks.
            q_row = tidx // Int32(2)
            q_blk = tidx % Int32(2)
            if q_row < live_c:
                values = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
                base_smem = sa_addr + q_row * Int32(_MX8_GU_ACT_STRIDE) + q_blk * Int32(64)
                for q in cutlass.range_constexpr(4):
                    words = ld_shared_v4_u32(base_smem + Int32(16 * q))
                    for word in cutlass.range_constexpr(4):
                        values[8 * q + 2 * word] = _bf16_lo(words[word])
                        values[8 * q + 2 * word + 1] = _bf16_hi(words[word])
                payload, scale = quantize_block_fp8_mx(values, max_abs_32(values))
                at = act_base + (Int64(tile_c) * Int64(STREAM_TILE_M) + Int64(q_row)) * Int64(_MX8_GU_ACT_ROW_BYTES)
                col_off = Int64(n0) + Int64(q_blk) * Int64(32)
                for half in cutlass.range_constexpr(2):
                    st_global_v4_u32(at + col_off + Int64(16 * half),
                                     payload[4 * half], payload[4 * half + 1],
                                     payload[4 * half + 2], payload[4 * half + 3])
                st_global_u8(at + Int64(MINI_INTER) + Int64(n0 // Int32(32)) + Int64(q_blk), cutlass.Uint8(scale))
            if ci + Int32(1) < n_chunks:
                cute.arch.sync_threads()
