"""Inkling 2 Mini specialized prefill routed-MoE GEMM kernels (NVFP4 W4A4 & MXFP8 W8A8).

Geometry-specialized for Inkling 2 Mini TP1 (hidden=2304, slice=1152, experts=128, top_k=8,
SiTU gate_cap=4, up_cap=25, route weight before FC2) at prefill capacities (max_rows >= 256).
Every other model family and capacity remains byte-identical to stock.

Design pillars (all K-order preserving -> bit-identical to stock):
1. Expert-stationary 2-chunk pairing (BM = up to 256 rows = 2 consecutive 128-row chunks of the
   same expert per CTA, resolved in O(1) via a 4-warp prefix scan over counts[0..127] in meta):
   - Cuts DRAM/L2 weight reads and SMEM weight loads in half when experts have >128 rows
     (R = 2048, 2752, 3000, 4096), while skipping chunk 1 cleanly when count <= 128.
   - Doubles weight arithmetic intensity and MMA-to-LDSM ratio.
2. Full 2D TMA with 32-row live-bounded activation boxes ((32, 128) or (32, 64)) and deep rings:
   - Eliminates dead-padding DRAM pulls on tail chunks and small-R calls (R = 256, 1000).
   - NVFP4 GateUp: 3-stage W TMA (128B swizzle, K=256) + 3-stage A TMA (128B swizzle, 32-row boxes),
     scales loaded directly into registers via 4B-aligned ld.global.nc.u32 + warp shuffle (0 SMEM).
   - NVFP4 Down: 4-stage W TMA (64B swizzle, K=128) + 4-stage A TMA (64B swizzle, 32-row boxes),
     scales loaded directly into registers + coalesced 16B BF16 epilogue through SMEM.
   - MXFP8 GateUp: doubled N tile (cols=64: 64 gate + 64 up = 128 N, halving grid.x from 36 to 18
     and halving scattered x re-reads), 3-stage W TMA (128B swizzle) + 2-stage cp.async A ring +
     coalesced preloaded weight scales + all-256-thread K32 quantization epilogue.
   - MXFP8 Down: 3-stage W TMA (128B swizzle) + 2-stage A TMA (32-row boxes, stride 1200) +
     coalesced preloaded s2 scales + 2-chunk weight reuse.
3. Single unified launch per projection (eliminating the serialized small/wide split and
   27,648-CTA empty small_down launch at prefill row counts).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.activation import gated_activation_value, situ_vendor
from b12x._lib.intrinsics import (
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    ld_global_nc_u32,
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
    st_global_v4_u32,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import META_HEAD, _i32_at, _round_bf16
from ._fp8_moe_stream import STREAM_TILE_M, _u8_weight
from ._fp8_weights import _ld_f32
from ._mxfp4_down_a8_plan import mxfp4_scale_row_bytes, mxfp8_down_row_bytes
from ._nvfp4_moe_dynamic import _shfl_idx_u32, nvfp4_amax_offset, nvfp4_denominator_offset

__all__ = [
    "MiniMxfp8Down",
    "MiniMxfp8GateUp",
    "MiniNvfp4Down",
    "MiniNvfp4GateUp",
    "mini_max_pairs",
]


def _u8_strided(pointer: cute.Pointer, rows, cols: int, row_stride: int) -> cute.Tensor:
    u8 = cute.make_ptr(cutlass.Uint8, Int64(pointer.toint()), cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(u8, cute.make_layout((rows, cols), stride=(row_stride, 1)))


@cute.jit
def mini_max_pairs(experts: cutlass.Constexpr, pairs: Int32) -> Int32:
    """Upper bound on sum_e ceil(counts[e] / 256) for any routing of ``pairs`` pairs."""
    groups = pairs
    if groups > Int32(experts):
        groups = Int32(experts)
    return (pairs + Int32(255)) // Int32(256) + groups


@cute.jit
def _resolve_pair_128(meta: cute.Pointer, pair_id: Int32, tidx: Int32, warp_id: Int32, lane: Int32,
                      smem_base: Int32):
    """Map ``pair_id`` to ``(expert, tile0, first0, live0, live1)`` over E=128 experts.

    Uses 52 bytes at ``smem_base`` before TMA begins (caller must sync_threads before
    overwriting ``smem_base``). Returns ``expert = -1`` when ``pair_id >= total_pairs``.
    """
    if tidx == Int32(0):
        st_shared_u32(smem_base + Int32(32), Uint32(0xFFFFFFFF))
    n = Int32(0)
    tiles_e = Int32(0)
    pairs_e = Int32(0)
    if tidx < Int32(128):
        n = _i32_at(Int64(meta.toint()) + Int64(4 * META_HEAD), tidx)
        tiles_e = (n + Int32(127)) // Int32(128)
        pairs_e = (n + Int32(255)) // Int32(256)
    scan_p = pairs_e
    scan_t = tiles_e
    for stage in cutlass.range_constexpr(5):
        off = 1 << stage
        up_p = Int32(cute.arch.shuffle_sync_up(scan_p, offset=off, mask_and_clamp=0))
        up_t = Int32(cute.arch.shuffle_sync_up(scan_t, offset=off, mask_and_clamp=0))
        if lane >= Int32(off):
            scan_p = scan_p + up_p
            scan_t = scan_t + up_t
    if (warp_id < Int32(4)) & (lane == Int32(31)):
        st_shared_u32(smem_base + warp_id * Int32(4), Uint32(scan_p))
        st_shared_u32(smem_base + Int32(16) + warp_id * Int32(4), Uint32(scan_t))
    cute.arch.sync_threads()
    if warp_id == Int32(0):
        wp = Int32(0)
        wt = Int32(0)
        if lane < Int32(4):
            wp = Int32(ld_shared_u32(smem_base + lane * Int32(4)))
            wt = Int32(ld_shared_u32(smem_base + Int32(16) + lane * Int32(4)))
        for stage in cutlass.range_constexpr(2):
            off = 1 << stage
            up_p = Int32(cute.arch.shuffle_sync_up(wp, offset=off, mask_and_clamp=0))
            up_t = Int32(cute.arch.shuffle_sync_up(wt, offset=off, mask_and_clamp=0))
            if lane >= Int32(off):
                wp = wp + up_p
                wt = wt + up_t
        if lane < Int32(4):
            st_shared_u32(smem_base + lane * Int32(4), Uint32(wp))
            st_shared_u32(smem_base + Int32(16) + lane * Int32(4), Uint32(wt))
    cute.arch.sync_threads()
    if tidx < Int32(128):
        base_p = Int32(0)
        base_t = Int32(0)
        if warp_id > Int32(0):
            base_p = Int32(ld_shared_u32(smem_base + (warp_id - Int32(1)) * Int32(4)))
            base_t = Int32(ld_shared_u32(smem_base + Int32(16) + (warp_id - Int32(1)) * Int32(4)))
        end_p = base_p + scan_p
        start_p = end_p - pairs_e
        if (pair_id >= start_p) & (pair_id < end_p):
            p_in_e = pair_id - start_p
            first_tile = base_t + scan_t - tiles_e
            tile0 = first_tile + p_in_e * Int32(2)
            row_start = p_in_e * Int32(256)
            rem = n - row_start
            live0 = rem
            if live0 > Int32(128):
                live0 = Int32(128)
            live1 = rem - Int32(128)
            if live1 < Int32(0):
                live1 = Int32(0)
            if live1 > Int32(128):
                live1 = Int32(128)
            offset_e = _i32_at(Int64(meta.toint()) + Int64(4 * (META_HEAD + 128)), tidx)
            first0 = offset_e + row_start
            st_shared_u32(smem_base + Int32(32), Uint32(tidx))
            st_shared_u32(smem_base + Int32(36), Uint32(tile0))
            st_shared_u32(smem_base + Int32(40), Uint32(first0))
            st_shared_u32(smem_base + Int32(44), Uint32(live0))
            st_shared_u32(smem_base + Int32(48), Uint32(live1))
    cute.arch.sync_threads()
    e = Int32(ld_shared_u32(smem_base + Int32(32)))
    tile0 = Int32(ld_shared_u32(smem_base + Int32(36)))
    first0 = Int32(ld_shared_u32(smem_base + Int32(40)))
    live0 = Int32(ld_shared_u32(smem_base + Int32(44)))
    live1 = Int32(ld_shared_u32(smem_base + Int32(48)))
    return e, tile0, first0, live0, live1


# ===========================================================================
# NVFP4 W4A4 Kernels (Inkling 2 Mini: H=2304, I=1152, E=128)
# ===========================================================================


class MiniNvfp4GateUp:
    """2D-TMA 2-chunk weight-stationary NVFP4 W4A4 GateUp + SiTU x gamma.

    Grid ``(18, max_pairs)`` = ``(I // 64, max_pairs)``, 256 threads (4 M-warps x 2 N-warps).
    - ``w_ring = 3`` 128B-swizzled TMA boxes ``(64, 128)`` for w1 and w3 (49,152 B).
    - ``a_ring = 3`` 128B-swizzled TMA boxes ``(32, 128)`` x 4 for xq (49,152 B), issuing only
      ``ceil(live / 32)`` boxes so tail chunks never pull dead padding from DRAM.
    - ``s1``/``s3`` and ``xs`` loaded directly into registers via naturally 4B-aligned
      ``ld_global_nc_u32`` + quad shuffle (0 SMEM, preserving align=4 ABI).
    - Total SMEM: 49,152 + 49,152 + 48 = 98,352 B <= 101,376 B.
    """

    threads = 256
    half_n = 64
    k_block = 256
    k_bytes = 128
    w_ring = 3
    a_ring = 3
    box_m = 32
    x_stride = 2 * 64 * 2 + 16  # 272 B (64 gate + 64 up BF16 + 16 B pad)

    def __init__(self, *, inter: int = 1152, hidden: int = 2304, experts: int = 128,
                 gate_cap: float = 4.0, up_cap: float = 25.0):
        self.inter, self.hidden, self.experts = int(inter), int(hidden), int(experts)
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        if self.inter != 1152 or self.hidden != 2304 or self.experts != 128:
            raise ValueError("MiniNvfp4GateUp is specialized for Inkling 2 Mini (H=2304, I=1152, E=128)")
        self.row_bytes = self.hidden // 2      # 1152 B
        self.scale_bytes = self.hidden // 16   # 144 B
        self.k_blocks = self.hidden // self.k_block  # 9
        self.half_bytes = self.half_n * self.k_bytes  # 8192 B
        self.box_bytes = self.box_m * self.k_bytes    # 4096 B
        self.a_stage_bytes = STREAM_TILE_M * self.k_bytes  # 16384 B
        self.amax_offset = nvfp4_amax_offset(self.experts, self.inter, self.hidden)
        self.denominator_offset = nvfp4_denominator_offset(self.experts, self.inter, self.hidden)

    def key(self) -> tuple:
        return ("mini_nvfp4_gate_up", 1, self.inter, self.hidden, self.experts,
                self.w_ring, self.a_ring, self.gate_cap, self.up_cap)

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_bytes),
            cutlass.Uint8)
        w_layout = cute.tile_to_shape(atom, (self.half_n, self.k_bytes, 2 * self.w_ring), order=(0, 1, 2))
        a_layout = cute.tile_to_shape(atom, (self.box_m, self.k_bytes, 4 * self.a_ring), order=(0, 1, 2))
        return w_layout, a_layout

    def _storage(self, w_layout, a_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(a_layout)], 1024],
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring + self.a_ring],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer,
                 w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                 row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
                 act_amax_part: cute.Pointer, max_tiles: Int32, max_pairs: Int32,
                 stream: cuda.CUstream):
        w_layout, a_layout = self._layouts()
        rows_w = self.experts * self.inter
        w_box = cute.slice_(w_layout, (None, None, 0))
        a_box = cute.slice_(a_layout, (None, None, 0))
        tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w1, rows_w, self.row_bytes), w_box,
            (self.half_n, self.k_bytes), num_multicast=1)
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w3, rows_w, self.row_bytes), w_box,
            (self.half_n, self.k_bytes), num_multicast=1)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(aq, max_tiles * Int32(STREAM_TILE_M), self.row_bytes),
            a_box, (self.box_m, self.k_bytes), num_multicast=1)
        self.kernel(
            a_s, meta, tma_tensor_a, tma_tensor_g, s1, tma_tensor_u, s3,
            row_amax, gamma, act_bf16, act_amax_part,
            tma_a, tma_g, tma_u, w_layout, a_layout, self._storage(w_layout, a_layout),
        ).launch(
            grid=(self.inter // self.half_n, max_pairs, 1),
            block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1,
        )

    @cute.jit
    def _issue_w(self, k_blk: Int32, tma_g, tma_u, t_wg, t_wu, t_ws, w_mbar, w_tile: Int32):
        slot = k_blk % Int32(self.w_ring)
        barrier = w_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * self.half_bytes)
        cute.copy(tma_g, t_wg[(None, w_tile, k_blk)], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
        cute.copy(tma_u, t_wu[(None, w_tile, k_blk)], t_ws[(None, slot * Int32(2) + Int32(1))], tma_bar_ptr=barrier)

    @cute.jit
    def _issue_a(self, a_step: Int32, tile: Int32, live_boxes: Int32, k_blk: Int32,
                 tma_a, t_ag, t_as, a_mbar):
        slot = a_step % Int32(self.a_ring)
        barrier = a_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, live_boxes * Int32(self.box_bytes))
        base_row_tile = tile * Int32(4)
        base_slot = slot * Int32(4)
        for b in cutlass.range_constexpr(4):
            if Int32(b) < live_boxes:
                cute.copy(tma_a, t_ag[(None, base_row_tile + Int32(b), k_blk)],
                          t_as[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)

    @cute.jit
    def _load_sfb(self, s_half_base: Int64, k_blk: Int32, c: Int32, sfb: cute.Tensor):
        k_off = Int64(k_blk) * Int64(16) + Int64(c) * Int64(4)
        for nt in cutlass.range_constexpr(8):
            sfb[nt] = ld_global_nc_u32(s_half_base + Int64(8 * nt) * Int64(self.scale_bytes) + k_off)

    @cute.jit
    def _mma_k256(self, a_base: Int32, w_base: Int32,
                  sfa_lo0: Uint32, sfa_hi0: Uint32, sfa_lo1: Uint32, sfa_hi1: Uint32,
                  sfb: cute.Tensor, acc: cute.Tensor, warp_m: Int32,
                  c: Int32, mat: Int32, mrow: Int32, quad_base: Int32):
        is_odd_c = (c & Int32(1)) != Int32(0)
        for step in cutlass.range_constexpr(4):
            apiece = ((Int32(2 * step) + (mat >> Int32(1))) ^ mrow) * Int32(16)
            bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ mrow) * Int32(16)
            step_lane = quad_base + Int32(step)
            sfa0 = Uint32(cutlass.select_(is_odd_c, _shfl_idx_u32(sfa_hi0, step_lane), _shfl_idx_u32(sfa_lo0, step_lane)))
            sfa1 = Uint32(cutlass.select_(is_odd_c, _shfl_idx_u32(sfa_hi1, step_lane), _shfl_idx_u32(sfa_lo1, step_lane)))
            arow0 = warp_m * Int32(32) + Int32(8) * (mat & Int32(1)) + mrow
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + arow0 * Int32(self.k_bytes) + apiece)
            a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(a_base + (arow0 + Int32(16)) * Int32(self.k_bytes) + apiece)
            for pair in cutlass.range_constexpr(4):
                bcol = Int32(16 * pair) + Int32(8) * (mat >> Int32(1)) + mrow
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + bcol * Int32(self.k_bytes) + bpiece)
                sfb0 = _shfl_idx_u32(sfb[2 * pair], step_lane)
                sfb1 = _shfl_idx_u32(sfb[2 * pair + 1], step_lane)
                f00 = 4 * (2 * pair)
                f10 = 4 * (8 + 2 * pair)
                f01 = 4 * (2 * pair + 1)
                f11 = 4 * (8 + 2 * pair + 1)
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

    @cute.jit
    def _load_sfa(self, as_row0_base: Int64, k_blk: Int32, c: Int32):
        k_off = Int64(k_blk) * Int64(16) + Int64(c) * Int64(4)
        sfa_lo0 = ld_global_nc_u32(as_row0_base + k_off)
        sfa_hi0 = ld_global_nc_u32(as_row0_base + Int64(8 * self.scale_bytes) + k_off)
        sfa_lo1 = ld_global_nc_u32(as_row0_base + Int64(16 * self.scale_bytes) + k_off)
        sfa_hi1 = ld_global_nc_u32(as_row0_base + Int64(24 * self.scale_bytes) + k_off)
        return sfa_lo0, sfa_hi0, sfa_lo1, sfa_hi1

    @cute.jit
    def _epilogue_chunk(self, xch_smem: Int32, acc: cute.Tensor, tile: Int32, first: Int32, live: Int32,
                        n0: Int32, w_amax: Float32, rcp_denom: Float32, row_amax: cute.Pointer,
                        gamma: cute.Pointer, act_bf16: cute.Pointer, act_amax_part: cute.Pointer,
                        tidx: Int32, warp_m: Int32, warp_n: Int32, q: Int32, c: Int32):
        amax_t = Int64(row_amax.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(4)
        if warp_m * Int32(32) < live:
            for mt in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + q
                    factor = Float32(0.0)
                    if row < live:
                        factor = (_ld_f32(amax_t + Int64(row) * Int64(4)) * w_amax) * rcp_denom
                    for nt in cutlass.range_constexpr(8):
                        f = 4 * (8 * mt + nt) + 2 * half
                        col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * c
                        st_shared_u32(xch_smem + row * Int32(self.x_stride) + col * Int32(2),
                                      pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
        cute.arch.sync_threads()
        gamma_t = Int64(gamma.toint())
        # 128 rows x 4 groups (of 16 cols) = 512 groups -> 2 iterations across 256 threads.
        for it in cutlass.range(2, unroll=1):
            gi = tidx + it * Int32(self.threads)
            row = gi // Int32(4)
            grp = gi % Int32(4)
            if row < live:
                g_row = _ld_f32(gamma_t + Int64(first + row) * Int64(4))
                at = xch_smem + row * Int32(self.x_stride) + grp * Int32(32)
                i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(row)
                out_at = Int64(act_bf16.toint()) + (i * Int64(self.inter) + Int64(n0) + Int64(grp) * Int64(16)) * Int64(2)
                local = Float32(0.0)
                for half in cutlass.range_constexpr(2):
                    gw = ld_shared_v4_u32(at + Int32(16 * half))
                    uw = ld_shared_v4_u32(at + Int32(self.half_n * 2 + 16 * half))
                    packed = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                    for qq in cutlass.range_constexpr(4):
                        v0 = situ_vendor(_bf16_lo(gw[qq]), _bf16_lo(uw[qq]), g_row,
                                         gate_cap=self.gate_cap, up_cap=self.up_cap)
                        v1 = situ_vendor(_bf16_hi(gw[qq]), _bf16_hi(uw[qq]), g_row,
                                         gate_cap=self.gate_cap, up_cap=self.up_cap)
                        local = fmax_f32(local, fmax_f32(fabs_f32(v0), fabs_f32(v1)))
                        packed[qq] = pack_f32x2_to_bfloat2(v0, v1)
                    st_global_v4_u32(out_at + Int64(16 * half), packed[0], packed[1], packed[2], packed[3])
                st_global_f32(
                    Int64(act_amax_part.toint())
                    + (i * Int64(self.inter // 16) + Int64(n0 // Int32(16)) + Int64(grp)) * Int64(4),
                    local,
                )

    @cute.kernel
    def kernel(self, a_s: cute.Pointer, meta: cute.Pointer, act_t: cute.Tensor,
               weight_g: cute.Tensor, s1: cute.Pointer, weight_u: cute.Tensor, s3: cute.Pointer,
               row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
               act_amax_part: cute.Pointer, tma_a: cute.CopyAtom, tma_g: cute.CopyAtom,
               tma_u: cute.CopyAtom, w_layout: cute.ComposedLayout, a_layout: cute.ComposedLayout,
               Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        pair_id = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sw = shared_ptr_to_u32(storage.w.data_ptr())
        sa = shared_ptr_to_u32(storage.a.data_ptr())

        e, tile0, first0, live0, live1 = _resolve_pair_128(meta, pair_id, tidx, warp_id, lane, sw)
        if e >= Int32(0):
            mbar = storage.mbar.data_ptr()
            w_mbar = mbar
            a_mbar = mbar + self.w_ring
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            t_ws, t_wg = cpasync.tma_partition(
                tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_g, (self.half_n, self.k_bytes), (None, None)), 0, 2))
            _, t_wu = cpasync.tma_partition(
                tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_u, (self.half_n, self.k_bytes), (None, None)), 0, 2))
            t_as, t_ag = cpasync.tma_partition(
                tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
                cute.group_modes(cute.local_tile(act_t, (self.box_m, self.k_bytes), (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_g)
                cpasync.prefetch_descriptor(tma_u)
                cpasync.prefetch_descriptor(tma_a)
                for s in cutlass.range_constexpr(self.w_ring + self.a_ring):
                    cute.arch.mbarrier_init(mbar + s, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            n0 = n_blk * Int32(self.half_n)
            w_tile = e * Int32(self.inter // self.half_n) + n_blk
            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            q = lane // Int32(4)
            c = lane % Int32(4)
            mat = lane // Int32(8)
            mrow = lane % Int32(8)
            quad_base = q * Int32(4)

            sp = cutlass.select_(warp_n == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
            s_half_base = sp + (Int64(e) * Int64(self.inter) + Int64(n0 + q)) * Int64(self.scale_bytes)
            w_amax = _ld_f32(sp + Int64(self.amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(
                Float32(1.0),
                Float32(1536.0) * _ld_f32(sp + Int64(self.denominator_offset)),
            )

            live0_boxes = (live0 + Int32(31)) // Int32(32)
            as0_row0 = Int64(a_s.toint()) + (Int64(tile0) * Int64(STREAM_TILE_M)
                                             + Int64(warp_m * Int32(32) + q)) * Int64(self.scale_bytes)
            active0 = warp_m * Int32(32) < live0

            acc0 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc0[v] = Float32(0.0)
            sfb_cur = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            sfb_next = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)

            if live1 > Int32(0):
                tile1 = tile0 + Int32(1)
                live1_boxes = (live1 + Int32(31)) // Int32(32)
                as1_row0 = Int64(a_s.toint()) + (Int64(tile1) * Int64(STREAM_TILE_M)
                                                 + Int64(warp_m * Int32(32) + q)) * Int64(self.scale_bytes)
                active1 = warp_m * Int32(32) < live1
                acc1 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(64):
                    acc1[v] = Float32(0.0)

                # Prologue: W[0..w_ring-2], A[0..a_ring-2], plus prefetch sfb[0] while TMA flies.
                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._issue_w(Int32(pre), tma_g, tma_u, t_wg, t_wu, t_ws, w_mbar, w_tile)
                    self._issue_a(Int32(0), tile0, live0_boxes, Int32(0), tma_a, t_ag, t_as, a_mbar)
                    if const_expr(self.a_ring > 2):
                        self._issue_a(Int32(1), tile1, live1_boxes, Int32(0), tma_a, t_ag, t_as, a_mbar)
                self._load_sfb(s_half_base, Int32(0), c, sfb_cur)

                total_a = Int32(2 * self.k_blocks)
                for k_blk in cutlass.range(self.k_blocks, unroll=1):
                    w_slot = k_blk % Int32(self.w_ring)
                    a0 = k_blk * Int32(2)
                    a0_slot = a0 % Int32(self.a_ring)
                    sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1 = self._load_sfa(as0_row0, k_blk, c)
                    cute.arch.mbarrier_wait(w_mbar + w_slot, (k_blk // Int32(self.w_ring)) & Int32(1))
                    cute.arch.mbarrier_wait(a_mbar + a0_slot, (a0 // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if k_blk + Int32(self.w_ring - 1) < Int32(self.k_blocks):
                            self._issue_w(k_blk + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, w_mbar, w_tile)
                        nxt0 = a0 + Int32(self.a_ring - 1)
                        if nxt0 < total_a:
                            nxt0_t = cutlass.select_((nxt0 & Int32(1)) == Int32(0), tile0, tile1)
                            nxt0_b = cutlass.select_((nxt0 & Int32(1)) == Int32(0), live0_boxes, live1_boxes)
                            self._issue_a(nxt0, nxt0_t, nxt0_b, nxt0 >> Int32(1), tma_a, t_ag, t_as, a_mbar)
                    if k_blk + Int32(1) < Int32(self.k_blocks):
                        self._load_sfb(s_half_base, k_blk + Int32(1), c, sfb_next)
                    w_base = sw + (w_slot * Int32(2) + warp_n) * Int32(self.half_bytes)
                    if active0:
                        self._mma_k256(sa + a0_slot * Int32(self.a_stage_bytes), w_base,
                                       sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1,
                                       sfb_cur, acc0, warp_m, c, mat, mrow, quad_base)

                    a1 = a0 + Int32(1)
                    a1_slot = a1 % Int32(self.a_ring)
                    sfa1_lo0, sfa1_hi0, sfa1_lo1, sfa1_hi1 = self._load_sfa(as1_row0, k_blk, c)
                    cute.arch.mbarrier_wait(a_mbar + a1_slot, (a1 // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        nxt1 = a1 + Int32(self.a_ring - 1)
                        if nxt1 < total_a:
                            nxt1_t = cutlass.select_((nxt1 & Int32(1)) == Int32(0), tile0, tile1)
                            nxt1_b = cutlass.select_((nxt1 & Int32(1)) == Int32(0), live0_boxes, live1_boxes)
                            self._issue_a(nxt1, nxt1_t, nxt1_b, nxt1 >> Int32(1), tma_a, t_ag, t_as, a_mbar)
                    if active1:
                        self._mma_k256(sa + a1_slot * Int32(self.a_stage_bytes), w_base,
                                       sfa1_lo0, sfa1_hi0, sfa1_lo1, sfa1_hi1,
                                       sfb_cur, acc1, warp_m, c, mat, mrow, quad_base)
                    for nt in cutlass.range_constexpr(8):
                        sfb_cur[nt] = sfb_next[nt]

                cute.arch.sync_threads()
                self._epilogue_chunk(sw, acc1, tile1, first0 + Int32(STREAM_TILE_M), live1, n0, w_amax, rcp_denom,
                                     row_amax, gamma, act_bf16, act_amax_part, tidx, warp_m, warp_n, q, c)
            else:
                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._issue_w(Int32(pre), tma_g, tma_u, t_wg, t_wu, t_ws, w_mbar, w_tile)
                    for pre in cutlass.range_constexpr(self.a_ring - 1):
                        self._issue_a(Int32(pre), tile0, live0_boxes, Int32(pre), tma_a, t_ag, t_as, a_mbar)
                if active0:
                    self._load_sfb(s_half_base, Int32(0), c, sfb_cur)

                for k_blk in cutlass.range(self.k_blocks, unroll=1):
                    w_slot = k_blk % Int32(self.w_ring)
                    a_slot = k_blk % Int32(self.a_ring)
                    sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if active0:
                        sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1 = self._load_sfa(as0_row0, k_blk, c)
                    cute.arch.mbarrier_wait(w_mbar + w_slot, (k_blk // Int32(self.w_ring)) & Int32(1))
                    cute.arch.mbarrier_wait(a_mbar + a_slot, (k_blk // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if k_blk + Int32(self.w_ring - 1) < Int32(self.k_blocks):
                            self._issue_w(k_blk + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, w_mbar, w_tile)
                        if k_blk + Int32(self.a_ring - 1) < Int32(self.k_blocks):
                            self._issue_a(k_blk + Int32(self.a_ring - 1), tile0, live0_boxes,
                                          k_blk + Int32(self.a_ring - 1), tma_a, t_ag, t_as, a_mbar)
                    if active0:
                        if k_blk + Int32(1) < Int32(self.k_blocks):
                            self._load_sfb(s_half_base, k_blk + Int32(1), c, sfb_next)
                        w_base = sw + (w_slot * Int32(2) + warp_n) * Int32(self.half_bytes)
                        self._mma_k256(sa + a_slot * Int32(self.a_stage_bytes), w_base,
                                       sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1,
                                       sfb_cur, acc0, warp_m, c, mat, mrow, quad_base)
                        for nt in cutlass.range_constexpr(8):
                            sfb_cur[nt] = sfb_next[nt]

            cute.arch.sync_threads()
            self._epilogue_chunk(sw, acc0, tile0, first0, live0, n0, w_amax, rcp_denom,
                                 row_amax, gamma, act_bf16, act_amax_part, tidx, warp_m, warp_n, q, c)


class MiniNvfp4Down:
    """2D-TMA 2-chunk weight-stationary NVFP4 W4A4 Down projection + coalesced BF16 epilogue.

    Grid ``(18, max_pairs)`` = ``(H // 128, max_pairs)``, 256 threads (4 M-warps x 2 N-warps).
    - ``w_ring = 5`` 64B-swizzled TMA boxes ``(128, 64)`` for w2 (40,960 B, also holds the
      34,816 B coalesced BF16 epilogue tile strictly within storage.w).
    - ``a_ring = 4`` 64B-swizzled TMA boxes ``(32, 64)`` x 4 for aq (32,768 B), issuing only
      ``ceil(live / 32)`` boxes.
    - ``s2`` and ``a_s`` loaded directly into registers via 4B-aligned ``ld_global_nc_u32`` +
      quad shuffle (0 SMEM).
    - Coalesced 16B ``st_global_v4_u32`` epilogue through drained SMEM.
    - Total SMEM: 40,960 + 32,768 + 72 = 73,800 B <= 101,376 B.
    """

    threads = 256
    tile_n = 128
    k_block = 128
    k_bytes = 64
    w_ring = 5
    a_ring = 4
    box_m = 32
    out_stride = 128 * 2 + 16  # 272 B

    def __init__(self, *, hidden: int = 2304, inter: int = 1152, experts: int = 128):
        self.hidden, self.inter, self.experts = int(hidden), int(inter), int(experts)
        if self.hidden != 2304 or self.inter != 1152 or self.experts != 128:
            raise ValueError("MiniNvfp4Down is specialized for Inkling 2 Mini (H=2304, I=1152, E=128)")
        self.row_bytes = self.inter // 2      # 576 B = 9 * 64 B
        self.scale_bytes = self.inter // 16   # 72 B = 9 * 8 B
        self.k_blocks = self.inter // self.k_block  # 9
        self.w_stage_bytes = self.tile_n * self.k_bytes  # 8192 B
        self.box_bytes = self.box_m * self.k_bytes       # 2048 B
        self.a_stage_bytes = STREAM_TILE_M * self.k_bytes  # 8192 B
        self.amax_offset = nvfp4_amax_offset(self.experts, self.hidden, self.inter)
        self.denominator_offset = nvfp4_denominator_offset(self.experts, self.hidden, self.inter)

    def key(self) -> tuple:
        return ("mini_nvfp4_down", 1, self.hidden, self.inter, self.experts, self.w_ring, self.a_ring)

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_bytes),
            cutlass.Uint8)
        w_layout = cute.tile_to_shape(atom, (self.tile_n, self.k_bytes, self.w_ring), order=(0, 1, 2))
        a_layout = cute.tile_to_shape(atom, (self.box_m, self.k_bytes, 4 * self.a_ring), order=(0, 1, 2))
        return w_layout, a_layout

    def _storage(self, w_layout, a_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(a_layout)], 1024],
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring + self.a_ring],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer,
                 w2: cute.Pointer, s2: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer,
                 max_tiles: Int32, max_pairs: Int32, stream: cuda.CUstream):
        w_layout, a_layout = self._layouts()
        rows_w = self.experts * self.hidden
        w_box = cute.slice_(w_layout, (None, None, 0))
        a_box = cute.slice_(a_layout, (None, None, 0))
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w2, rows_w, self.row_bytes), w_box,
            (self.tile_n, self.k_bytes), num_multicast=1)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(aq, max_tiles * Int32(STREAM_TILE_M), self.row_bytes),
            a_box, (self.box_m, self.k_bytes), num_multicast=1)
        self.kernel(
            a_s, meta, tma_tensor_a, tma_tensor_w, s2, y, row_amax,
            tma_a, tma_w, w_layout, a_layout, self._storage(w_layout, a_layout),
        ).launch(
            grid=(self.hidden // self.tile_n, max_pairs, 1),
            block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1,
        )

    @cute.jit
    def _issue_w(self, k_blk: Int32, tma_w, t_wg, t_ws, w_mbar, w_tile: Int32):
        slot = k_blk % Int32(self.w_ring)
        barrier = w_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.w_stage_bytes)
        cute.copy(tma_w, t_wg[(None, w_tile, k_blk)], t_ws[(None, slot)], tma_bar_ptr=barrier)

    @cute.jit
    def _issue_a(self, a_step: Int32, tile: Int32, live_boxes: Int32, k_blk: Int32,
                 tma_a, t_ag, t_as, a_mbar):
        slot = a_step % Int32(self.a_ring)
        barrier = a_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, live_boxes * Int32(self.box_bytes))
        base_row_tile = tile * Int32(4)
        base_slot = slot * Int32(4)
        for b in cutlass.range_constexpr(4):
            if Int32(b) < live_boxes:
                cute.copy(tma_a, t_ag[(None, base_row_tile + Int32(b), k_blk)],
                          t_as[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)

    @cute.jit
    def _load_sfb(self, s_row0_base: Int64, k_blk: Int32, c: Int32, sfb: cute.Tensor):
        k_off = Int64(k_blk) * Int64(8) + Int64(c & Int32(1)) * Int64(4)
        for nt in cutlass.range_constexpr(8):
            sfb[nt] = ld_global_nc_u32(s_row0_base + Int64(8 * nt) * Int64(self.scale_bytes) + k_off)

    @cute.jit
    def _load_sfa(self, as_row0_base: Int64, k_blk: Int32, c: Int32):
        k_off = Int64(k_blk) * Int64(8) + Int64(c & Int32(1)) * Int64(4)
        sfa_lo0 = ld_global_nc_u32(as_row0_base + k_off)
        sfa_hi0 = ld_global_nc_u32(as_row0_base + Int64(8 * self.scale_bytes) + k_off)
        sfa_lo1 = ld_global_nc_u32(as_row0_base + Int64(16 * self.scale_bytes) + k_off)
        sfa_hi1 = ld_global_nc_u32(as_row0_base + Int64(24 * self.scale_bytes) + k_off)
        return sfa_lo0, sfa_hi0, sfa_lo1, sfa_hi1

    @cute.jit
    def _mma_k128(self, a_base: Int32, w_base: Int32,
                  sfa_lo0: Uint32, sfa_hi0: Uint32, sfa_lo1: Uint32, sfa_hi1: Uint32,
                  sfb: cute.Tensor, acc: cute.Tensor, warp_m: Int32, warp_n: Int32,
                  c: Int32, mat: Int32, mrow: Int32, quad_base: Int32):
        is_odd_c = (c & Int32(1)) != Int32(0)
        swz64 = mrow >> Int32(1)
        for step in cutlass.range_constexpr(2):
            apiece = ((Int32(2 * step) + (mat >> Int32(1))) ^ swz64) * Int32(16)
            bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ swz64) * Int32(16)
            step_lane = quad_base + Int32(step)
            sfa0 = Uint32(cutlass.select_(is_odd_c, _shfl_idx_u32(sfa_hi0, step_lane), _shfl_idx_u32(sfa_lo0, step_lane)))
            sfa1 = Uint32(cutlass.select_(is_odd_c, _shfl_idx_u32(sfa_hi1, step_lane), _shfl_idx_u32(sfa_lo1, step_lane)))
            arow0 = warp_m * Int32(32) + Int32(8) * (mat & Int32(1)) + mrow
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + arow0 * Int32(self.k_bytes) + apiece)
            a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(a_base + (arow0 + Int32(16)) * Int32(self.k_bytes) + apiece)
            for pair in cutlass.range_constexpr(4):
                bcol = warp_n * Int32(64) + Int32(16 * pair) + Int32(8) * (mat >> Int32(1)) + mrow
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + bcol * Int32(self.k_bytes) + bpiece)
                sfb0 = _shfl_idx_u32(sfb[2 * pair], step_lane)
                sfb1 = _shfl_idx_u32(sfb[2 * pair + 1], step_lane)
                f00 = 4 * (2 * pair)
                f10 = 4 * (8 + 2 * pair)
                f01 = 4 * (2 * pair + 1)
                f11 = 4 * (8 + 2 * pair + 1)
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

    @cute.jit
    def _epilogue_chunk(self, out_smem: Int32, acc: cute.Tensor, tile: Int32, first: Int32, live: Int32,
                        n0: Int32, w_amax: Float32, rcp_denom: Float32, row_amax: cute.Pointer,
                        y: cute.Pointer, tidx: Int32, warp_m: Int32, warp_n: Int32, q: Int32, c: Int32):
        amax_t = Int64(row_amax.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(4)
        if warp_m * Int32(32) < live:
            for mt in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + q
                    if row < live:
                        factor = (_ld_f32(amax_t + Int64(row) * Int64(4)) * w_amax) * rcp_denom
                        for nt in cutlass.range_constexpr(8):
                            f = 4 * (8 * mt + nt) + 2 * half
                            col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * c
                            st_shared_u32(out_smem + row * Int32(self.out_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
        cute.arch.sync_threads()
        segments = self.tile_n * 2 // 16  # 16 segments of 16B per row
        for qit in cutlass.range_constexpr(STREAM_TILE_M * segments // self.threads):
            ci = tidx + Int32(self.threads * qit)
            row = ci // Int32(segments)
            seg = ci % Int32(segments)
            if row < live:
                v0, v1, v2, v3 = ld_shared_v4_u32(out_smem + row * Int32(self.out_stride) + seg * Int32(16))
                st_global_v4_u32(
                    Int64(y.toint()) + ((Int64(first) + Int64(row)) * Int64(self.hidden) + Int64(n0)) * Int64(2)
                    + Int64(seg) * Int64(16),
                    v0, v1, v2, v3,
                )

    @cute.kernel
    def kernel(self, a_s: cute.Pointer, meta: cute.Pointer, act_t: cute.Tensor,
               weight_t: cute.Tensor, s2: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer,
               tma_a: cute.CopyAtom, tma_w: cute.CopyAtom, w_layout: cute.ComposedLayout,
               a_layout: cute.ComposedLayout, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        pair_id = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sw = shared_ptr_to_u32(storage.w.data_ptr())
        sa = shared_ptr_to_u32(storage.a.data_ptr())

        e, tile0, first0, live0, live1 = _resolve_pair_128(meta, pair_id, tidx, warp_id, lane, sw)
        if e >= Int32(0):
            mbar = storage.mbar.data_ptr()
            w_mbar = mbar
            a_mbar = mbar + self.w_ring
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            t_ws, t_wg = cpasync.tma_partition(
                tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_t, (self.tile_n, self.k_bytes), (None, None)), 0, 2))
            t_as, t_ag = cpasync.tma_partition(
                tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
                cute.group_modes(cute.local_tile(act_t, (self.box_m, self.k_bytes), (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_w)
                cpasync.prefetch_descriptor(tma_a)
                for s in cutlass.range_constexpr(self.w_ring + self.a_ring):
                    cute.arch.mbarrier_init(mbar + s, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            n0 = n_blk * Int32(self.tile_n)
            w_tile = e * Int32(self.hidden // self.tile_n) + n_blk
            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            q = lane // Int32(4)
            c = lane % Int32(4)
            mat = lane // Int32(8)
            mrow = lane % Int32(8)
            quad_base = q * Int32(4)

            s_row0_base = Int64(s2.toint()) + (Int64(e) * Int64(self.hidden)
                                               + Int64(n0 + warp_n * Int32(64) + q)) * Int64(self.scale_bytes)
            w_amax = _ld_f32(Int64(s2.toint()) + Int64(self.amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(
                Float32(1.0),
                Float32(1536.0) * _ld_f32(Int64(s2.toint()) + Int64(self.denominator_offset)),
            )

            live0_boxes = (live0 + Int32(31)) // Int32(32)
            as0_row0 = Int64(a_s.toint()) + (Int64(tile0) * Int64(STREAM_TILE_M)
                                             + Int64(warp_m * Int32(32) + q)) * Int64(self.scale_bytes)
            active0 = warp_m * Int32(32) < live0

            acc0 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc0[v] = Float32(0.0)
            sfb_cur = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            sfb_next = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)

            if live1 > Int32(0):
                tile1 = tile0 + Int32(1)
                live1_boxes = (live1 + Int32(31)) // Int32(32)
                as1_row0 = Int64(a_s.toint()) + (Int64(tile1) * Int64(STREAM_TILE_M)
                                                 + Int64(warp_m * Int32(32) + q)) * Int64(self.scale_bytes)
                active1 = warp_m * Int32(32) < live1
                acc1 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(64):
                    acc1[v] = Float32(0.0)

                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._issue_w(Int32(pre), tma_w, t_wg, t_ws, w_mbar, w_tile)
                    for pre in cutlass.range_constexpr(self.a_ring - 1):
                        pt = tile0 if (pre % 2 == 0) else tile1
                        pb = live0_boxes if (pre % 2 == 0) else live1_boxes
                        self._issue_a(Int32(pre), pt, pb, Int32(pre // 2), tma_a, t_ag, t_as, a_mbar)
                self._load_sfb(s_row0_base, Int32(0), c, sfb_cur)

                total_a = Int32(2 * self.k_blocks)
                for k_blk in cutlass.range(self.k_blocks, unroll=1):
                    w_slot = k_blk % Int32(self.w_ring)
                    a0 = k_blk * Int32(2)
                    a0_slot = a0 % Int32(self.a_ring)
                    sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1 = self._load_sfa(as0_row0, k_blk, c)
                    cute.arch.mbarrier_wait(w_mbar + w_slot, (k_blk // Int32(self.w_ring)) & Int32(1))
                    cute.arch.mbarrier_wait(a_mbar + a0_slot, (a0 // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if k_blk + Int32(self.w_ring - 1) < Int32(self.k_blocks):
                            self._issue_w(k_blk + Int32(self.w_ring - 1), tma_w, t_wg, t_ws, w_mbar, w_tile)
                        nxt0 = a0 + Int32(self.a_ring - 1)
                        if nxt0 < total_a:
                            nxt0_t = cutlass.select_((nxt0 & Int32(1)) == Int32(0), tile0, tile1)
                            nxt0_b = cutlass.select_((nxt0 & Int32(1)) == Int32(0), live0_boxes, live1_boxes)
                            self._issue_a(nxt0, nxt0_t, nxt0_b, nxt0 >> Int32(1), tma_a, t_ag, t_as, a_mbar)
                    if k_blk + Int32(1) < Int32(self.k_blocks):
                        self._load_sfb(s_row0_base, k_blk + Int32(1), c, sfb_next)
                    w_base = sw + w_slot * Int32(self.w_stage_bytes)
                    if active0:
                        self._mma_k128(sa + a0_slot * Int32(self.a_stage_bytes), w_base,
                                       sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1,
                                       sfb_cur, acc0, warp_m, warp_n, c, mat, mrow, quad_base)

                    a1 = a0 + Int32(1)
                    a1_slot = a1 % Int32(self.a_ring)
                    sfa1_lo0, sfa1_hi0, sfa1_lo1, sfa1_hi1 = self._load_sfa(as1_row0, k_blk, c)
                    cute.arch.mbarrier_wait(a_mbar + a1_slot, (a1 // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        nxt1 = a1 + Int32(self.a_ring - 1)
                        if nxt1 < total_a:
                            nxt1_t = cutlass.select_((nxt1 & Int32(1)) == Int32(0), tile0, tile1)
                            nxt1_b = cutlass.select_((nxt1 & Int32(1)) == Int32(0), live0_boxes, live1_boxes)
                            self._issue_a(nxt1, nxt1_t, nxt1_b, nxt1 >> Int32(1), tma_a, t_ag, t_as, a_mbar)
                    if active1:
                        self._mma_k128(sa + a1_slot * Int32(self.a_stage_bytes), w_base,
                                       sfa1_lo0, sfa1_hi0, sfa1_lo1, sfa1_hi1,
                                       sfb_cur, acc1, warp_m, warp_n, c, mat, mrow, quad_base)
                    for nt in cutlass.range_constexpr(8):
                        sfb_cur[nt] = sfb_next[nt]

                cute.arch.sync_threads()
                self._epilogue_chunk(sw, acc1, tile1, first0 + Int32(STREAM_TILE_M), live1, n0, w_amax, rcp_denom,
                                     row_amax, y, tidx, warp_m, warp_n, q, c)
            else:
                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._issue_w(Int32(pre), tma_w, t_wg, t_ws, w_mbar, w_tile)
                    for pre in cutlass.range_constexpr(self.a_ring - 1):
                        self._issue_a(Int32(pre), tile0, live0_boxes, Int32(pre), tma_a, t_ag, t_as, a_mbar)
                if active0:
                    self._load_sfb(s_row0_base, Int32(0), c, sfb_cur)

                for k_blk in cutlass.range(self.k_blocks, unroll=1):
                    w_slot = k_blk % Int32(self.w_ring)
                    a_slot = k_blk % Int32(self.a_ring)
                    sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if active0:
                        sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1 = self._load_sfa(as0_row0, k_blk, c)
                    cute.arch.mbarrier_wait(w_mbar + w_slot, (k_blk // Int32(self.w_ring)) & Int32(1))
                    cute.arch.mbarrier_wait(a_mbar + a_slot, (k_blk // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if k_blk + Int32(self.w_ring - 1) < Int32(self.k_blocks):
                            self._issue_w(k_blk + Int32(self.w_ring - 1), tma_w, t_wg, t_ws, w_mbar, w_tile)
                        if k_blk + Int32(self.a_ring - 1) < Int32(self.k_blocks):
                            self._issue_a(k_blk + Int32(self.a_ring - 1), tile0, live0_boxes,
                                          k_blk + Int32(self.a_ring - 1), tma_a, t_ag, t_as, a_mbar)
                    if active0:
                        if k_blk + Int32(1) < Int32(self.k_blocks):
                            self._load_sfb(s_row0_base, k_blk + Int32(1), c, sfb_next)
                        w_base = sw + w_slot * Int32(self.w_stage_bytes)
                        self._mma_k128(sa + a_slot * Int32(self.a_stage_bytes), w_base,
                                       sfa0_lo0, sfa0_hi0, sfa0_lo1, sfa0_hi1,
                                       sfb_cur, acc0, warp_m, warp_n, c, mat, mrow, quad_base)
                        for nt in cutlass.range_constexpr(8):
                            sfb_cur[nt] = sfb_next[nt]

            cute.arch.sync_threads()
            self._epilogue_chunk(sw, acc0, tile0, first0, live0, n0, w_amax, rcp_denom,
                                 row_amax, y, tidx, warp_m, warp_n, q, c)


# ===========================================================================
# MXFP8 W8A8 Kernels (Inkling 2 Mini: H=2304, I=1152, E=128)
# ===========================================================================


class MiniMxfp8GateUp:
    """Wide-N (64 gate + 64 up) 2-chunk weight-stationary MXFP8 W8A8 GateUp + SiTU + MXFP8 quant.

    Grid ``(18, max_pairs)`` = ``(I // 64, max_pairs)``, 256 threads (4 M-warps x 2 N-warps).
    - Doubles N per CTA from stock's 32 to 64 (halving grid.x from 36 to 18 and halving x reads).
    - Pairs 2 chunks of the same expert per CTA (halving weight DRAM/L2 reads at R=2048..4096).
    - ``w_ring = 3`` 128B-swizzled TMA boxes ``(64, 128)`` for w1 and w3 (49,152 B).
    - ``a_ring = 2`` XOR-swizzled cp.async A ring (32,768 B) + A scales (1,024 B).
    - Preloaded coalesced weight scales ``sc`` (128 x 76 B = 9,728 B, odd word stride -> 0 conflicts).
    - Total SMEM: 49,152 + 32,768 + 1,024 + 9,728 + 24 = 92,696 B <= 101,376 B.
    - All 256 threads quantize 128 rows x 2 K32 blocks in the epilogue (2x stock epilogue width).
    """

    threads = 256
    cols = 64
    k_step = 128
    w_ring = 3
    a_ring = 2

    def __init__(self, *, inter: int = 1152, hidden: int = 2304, experts: int = 128,
                 gate_cap: float = 4.0, up_cap: float = 25.0):
        self.inter, self.hidden, self.experts = int(inter), int(hidden), int(experts)
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        if self.inter != 1152 or self.hidden != 2304 or self.experts != 128:
            raise ValueError("MiniMxfp8GateUp is specialized for Inkling 2 Mini (H=2304, I=1152, E=128)")
        self.row_bytes = self.hidden + self.hidden // 32  # 2376 B (8 mod 16)
        self.k_steps = self.hidden // self.k_step         # 18
        self.half_bytes = self.cols * self.k_step         # 8192 B
        self.a_bytes = STREAM_TILE_M * self.k_step        # 16384 B
        self.s_bytes = STREAM_TILE_M * 4                  # 512 B
        self.scale_cols = self.hidden // 32               # 72 B (18 u32 words)
        self.sc_stride = self.scale_cols + 4              # 76 B (19 u32 words -> odd -> 0 conflicts)
        self.act_row_bytes = mxfp8_down_row_bytes(self.inter)  # 1200 B
        self.act_stride = self.cols * 2 + 16              # 144 B (128 * 144 = 18,432 B)

    def key(self) -> tuple:
        return ("mini_mxfp8_gate_up", 1, self.inter, self.hidden, self.experts,
                self.cols, self.w_ring, self.a_ring, self.gate_cap, self.up_cap)

    def _w_layout(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_step),
            cutlass.Uint8)
        return cute.tile_to_shape(atom, (self.cols, self.k_step, 2 * self.w_ring), order=(0, 1, 2))

    def _storage(self, w_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.a_bytes], 1024],
            "s": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.s_bytes], 16],
            "sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.cols * self.sc_stride], 16],
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
                 w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                 act: cute.Pointer, max_pairs: Int32, stream: cuda.CUstream,
                 grouped_gamma: cute.Pointer):
        w_layout = self._w_layout()
        rows_w = self.experts * self.inter
        box = cute.slice_(w_layout, (None, None, 0))
        tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w1, rows_w, self.hidden), box,
            (self.cols, self.k_step), num_multicast=1)
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w3, rows_w, self.hidden), box,
            (self.cols, self.k_step), num_multicast=1)
        self.kernel(
            x, pair_row, meta, tma_tensor_g, tma_tensor_u, s1, s3, act,
            tma_g, tma_u, w_layout, self._storage(w_layout), grouped_gamma,
        ).launch(
            grid=(self.inter // self.cols, max_pairs, 1),
            block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1,
        )

    @cute.jit
    def _issue_w(self, step: Int32, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile: Int32):
        slot = step % Int32(self.w_ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * self.half_bytes)
        cute.copy(tma_g, t_wg[(None, w_tile, step)], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
        cute.copy(tma_u, t_wu[(None, w_tile, step)], t_ws[(None, slot * Int32(2) + Int32(1))], tma_bar_ptr=barrier)

    @cute.jit
    def _load_a_slot(self, step: Int32, stage: Int32, sa: Int32, ss: Int32,
                     src: cute.Tensor, s_src: Int64, ch: Int32, r_base: Int32, tidx: Int32):
        k_off = Int64(step) * Int64(self.k_step)
        a_stage = sa + stage * Int32(self.a_bytes)
        for q in cutlass.range_constexpr(4):
            if src[q] >= Int64(0):
                row = r_base + Int32(32 * q)
                dst = a_stage + row * Int32(self.k_step) + ((ch ^ (row % Int32(8))) * Int32(16))
                cp_async_u64_shared_global(dst, src[q] + k_off)
                cp_async_u64_shared_global(dst + Int32(8), src[q] + k_off + Int64(8))
        if s_src >= Int64(0):
            cp_async_u32_shared_global(ss + stage * Int32(self.s_bytes) + tidx * Int32(4),
                                       s_src + Int64(step) * Int64(4))

    @cute.jit
    def _mma_k128(self, step: Int32, a_stage: Int32, w_slot: Int32, sa: Int32, sw: Int32,
                  ss: Int32, sc: Int32, warp_m: Int32, warp_n: Int32, lane: Int32, g: Int32,
                  acc: cute.Tensor):
        a_base = sa + a_stage * Int32(self.a_bytes)
        w_base = sw + w_slot * Int32(2 * self.half_bytes)
        s_base = ss + a_stage * Int32(self.s_bytes)
        asc0 = ld_shared_u32(s_base + (warp_m * Int32(32) + g + (lane & Int32(1)) * Int32(8)) * Int32(4))
        asc1 = ld_shared_u32(s_base + (warp_m * Int32(32) + Int32(16) + g + (lane & Int32(1)) * Int32(8)) * Int32(4))
        for ks in cutlass.range_constexpr(4):
            row0 = warp_m * Int32(32) + (lane % Int32(16))
            chunk_a = Int32(2 * ks) + lane // Int32(16)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                a_base + row0 * Int32(self.k_step) + ((chunk_a ^ (row0 % Int32(8))) * Int32(16)))
            a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(
                a_base + (row0 + Int32(16)) * Int32(self.k_step) + ((chunk_a ^ (row0 % Int32(8))) * Int32(16)))
            chunk_b = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
            for side in cutlass.range_constexpr(2):
                for np in cutlass.range_constexpr(2):
                    nrow = warp_n * Int32(32) + Int32(16 * np) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                    b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                        w_base + Int32(side * self.half_bytes) + nrow * Int32(self.k_step)
                        + ((chunk_b ^ (nrow % Int32(8))) * Int32(16)))
                    col0 = Int32(side * self.cols + 16 * np) + warp_n * Int32(32) + g
                    wsc0 = ld_shared_u32(sc + col0 * Int32(self.sc_stride) + step * Int32(4))
                    wsc1 = ld_shared_u32(sc + (col0 + Int32(8)) * Int32(self.sc_stride) + step * Int32(4))
                    nt0 = 4 * side + 2 * np
                    nt1 = nt0 + 1
                    f00 = 4 * nt0
                    f10 = 4 * (8 + nt0)
                    f01 = 4 * nt1
                    f11 = 4 * (8 + nt1)
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3],
                        a0, a1, a2, a3, b0, b1, asc0, wsc0, bid_a=ks, bid_b=ks)
                    acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3],
                        a4, a5, a6, a7, b0, b1, asc1, wsc0, bid_a=ks, bid_b=ks)
                    acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3],
                        a0, a1, a2, a3, b2, b3, asc0, wsc1, bid_a=ks, bid_b=ks)
                    acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3],
                        a4, a5, a6, a7, b2, b3, asc1, wsc1, bid_a=ks, bid_b=ks)
                    acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3] = d0, d1, d2, d3

    @cute.jit
    def _activation(self, gate: Float32, up: Float32, gamma: Float32) -> Float32:
        value = gated_activation_value(_round_bf16(gate), _round_bf16(up), is_situ=True,
                                       situ_beta=self.gate_cap, situ_linear_beta=self.up_cap)
        return _round_bf16(value * gamma)

    @cute.jit
    def _epilogue_chunk(self, sa: Int32, acc: cute.Tensor, tile: Int32, first: Int32, live: Int32,
                        n0: Int32, grouped_gamma: cute.Pointer, act: cute.Pointer,
                        tidx: Int32, warp_m: Int32, warp_n: Int32, g: Int32, j: Int32):
        gamma_t = cute.make_ptr(Float32, Int64(grouped_gamma.toint()), cute.AddressSpace.gmem, assumed_align=4)
        if warp_m * Int32(32) < live:
            for mt in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                    if row < live:
                        gamma = Float32(gamma_t[first + row])
                        for nt in cutlass.range_constexpr(4):
                            fg = 4 * (8 * mt + nt) + 2 * half
                            fu = 4 * (8 * mt + nt + 4) + 2 * half
                            col = warp_n * Int32(32) + Int32(8 * nt) + Int32(2) * j
                            v0 = self._activation(acc[fg], acc[fu], gamma)
                            v1 = self._activation(acc[fg + 1], acc[fu + 1], gamma)
                            st_shared_u32(sa + row * Int32(self.act_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(v0, v1))
        cute.arch.sync_threads()
        # All 256 threads quantize 128 rows x 2 K32 blocks = 256 blocks.
        q_row = tidx // Int32(2)
        q_blk = tidx % Int32(2)
        if q_row < live:
            values = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            base_smem = sa + q_row * Int32(self.act_stride) + q_blk * Int32(64)
            for q in cutlass.range_constexpr(4):
                words = ld_shared_v4_u32(base_smem + Int32(16 * q))
                for word in cutlass.range_constexpr(4):
                    values[8 * q + 2 * word] = _bf16_lo(words[word])
                    values[8 * q + 2 * word + 1] = _bf16_hi(words[word])
            payload, scale = quantize_block_fp8_mx(values, max_abs_32(values))
            at = Int64(act.toint()) + (Int64(tile) * Int64(STREAM_TILE_M) + Int64(q_row)) * Int64(self.act_row_bytes)
            col_off = Int64(n0) + Int64(q_blk) * Int64(32)
            for half in cutlass.range_constexpr(2):
                st_global_v4_u32(at + col_off + Int64(16 * half),
                                 payload[4 * half], payload[4 * half + 1],
                                 payload[4 * half + 2], payload[4 * half + 3])
            st_global_u8(at + Int64(self.inter) + Int64(n0 // Int32(32)) + Int64(q_blk), cutlass.Uint8(scale))

    @cute.kernel
    def kernel(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
               weight_g: cute.Tensor, weight_u: cute.Tensor, s1: cute.Pointer, s3: cute.Pointer,
               act: cute.Pointer, tma_g: cute.CopyAtom, tma_u: cute.CopyAtom,
               w_layout: cute.ComposedLayout, Storage: cutlass.Constexpr,
               grouped_gamma: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        pair_id = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sw = shared_ptr_to_u32(storage.w.data_ptr())
        sa = shared_ptr_to_u32(storage.a.data_ptr())
        ss = shared_ptr_to_u32(storage.s.data_ptr())
        sc = shared_ptr_to_u32(storage.sc.data_ptr())

        e, tile0, first0, live0, live1 = _resolve_pair_128(meta, pair_id, tidx, warp_id, lane, sw)
        if e >= Int32(0):
            mbar = storage.mbar.data_ptr()
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            n0 = n_blk * Int32(self.cols)
            w_tile = e * Int32(self.inter // self.cols) + n_blk
            t_ws, t_wg = cpasync.tma_partition(
                tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_g, (self.cols, self.k_step), (None, None)), 0, 2))
            _, t_wu = cpasync.tma_partition(
                tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_u, (self.cols, self.k_step), (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_g)
                cpasync.prefetch_descriptor(tma_u)
                for slot in cutlass.range_constexpr(self.w_ring):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # Preload all 128 weight scale rows (64 gate + 64 up) coalesced along K (18 words/row).
            scale_words = self.scale_cols // 4  # 18
            total_sc = 2 * self.cols * scale_words  # 2304 = 9 * 256
            for qsc in cutlass.range_constexpr((total_sc + self.threads - 1) // self.threads):
                u = tidx + Int32(qsc * self.threads)
                if u < Int32(total_sc):
                    r = u // Int32(scale_words)
                    c_w = u % Int32(scale_words)
                    base = Int64(s1.toint())
                    col = r
                    if r >= Int32(self.cols):
                        base = Int64(s3.toint())
                        col = r - Int32(self.cols)
                    cp_async_u32_shared_global(
                        sc + r * Int32(self.sc_stride) + c_w * Int32(4),
                        base + (Int64(e) * Int64(self.inter) + Int64(n0 + col)) * Int64(self.scale_cols)
                        + Int64(c_w) * Int64(4))

            ch = tidx % Int32(8)
            r_base = tidx // Int32(8)
            x_base = Int64(x.toint())
            pr_base = Int64(pair_row.toint())
            src0 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
            for q in cutlass.range_constexpr(4):
                row = r_base + Int32(32 * q)
                src0[q] = Int64(-1)
                if row < live0:
                    source = _i32_at(pr_base, first0 + row)
                    src0[q] = x_base + Int64(source) * Int64(self.row_bytes) + Int64(ch) * Int64(16)
            s_src0 = Int64(-1)
            if tidx < live0:
                source = _i32_at(pr_base, first0 + tidx)
                s_src0 = x_base + Int64(source) * Int64(self.row_bytes) + Int64(self.hidden)

            # Prologue: A0[0] joins the sc cp.async group; W[0..w_ring-2] via TMA.
            self._load_a_slot(Int32(0), Int32(0), sa, ss, src0, s_src0, ch, r_base, tidx)
            cute.arch.cp_async_commit_group()
            if warp_id == Int32(0):
                for pre in cutlass.range_constexpr(self.w_ring - 1):
                    self._issue_w(Int32(pre), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active0 = warp_m * Int32(32) < live0
            g = lane // Int32(4)
            j = lane % Int32(4)

            acc0 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc0[v] = Float32(0.0)

            if live1 > Int32(0):
                first1 = first0 + Int32(STREAM_TILE_M)
                src1 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
                for q in cutlass.range_constexpr(4):
                    row = r_base + Int32(32 * q)
                    src1[q] = Int64(-1)
                    if row < live1:
                        source = _i32_at(pr_base, first1 + row)
                        src1[q] = x_base + Int64(source) * Int64(self.row_bytes) + Int64(ch) * Int64(16)
                s_src1 = Int64(-1)
                if tidx < live1:
                    source = _i32_at(pr_base, first1 + tidx)
                    s_src1 = x_base + Int64(source) * Int64(self.row_bytes) + Int64(self.hidden)
                active1 = warp_m * Int32(32) < live1
                acc1 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(64):
                    acc1[v] = Float32(0.0)

                for step in cutlass.range(self.k_steps, unroll=1):
                    w_slot = step % Int32(self.w_ring)
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.mbarrier_wait(mbar + w_slot, (step // Int32(self.w_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    self._load_a_slot(step, Int32(1), sa, ss, src1, s_src1, ch, r_base, tidx)
                    cute.arch.cp_async_commit_group()
                    if warp_id == Int32(0):
                        if step + Int32(self.w_ring - 1) < Int32(self.k_steps):
                            self._issue_w(step + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                    if active0:
                        self._mma_k128(step, Int32(0), w_slot, sa, sw, ss, sc, warp_m, warp_n, lane, g, acc0)

                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    if step + Int32(1) < Int32(self.k_steps):
                        self._load_a_slot(step + Int32(1), Int32(0), sa, ss, src0, s_src0, ch, r_base, tidx)
                        cute.arch.cp_async_commit_group()
                    if active1:
                        self._mma_k128(step, Int32(1), w_slot, sa, sw, ss, sc, warp_m, warp_n, lane, g, acc1)

                cute.arch.sync_threads()
                self._epilogue_chunk(sa, acc1, tile0 + Int32(1), first1, live1, n0, grouped_gamma, act,
                                     tidx, warp_m, warp_n, g, j)
            else:
                for step in cutlass.range(self.k_steps, unroll=1):
                    w_slot = step % Int32(self.w_ring)
                    a_stage = step % Int32(self.a_ring)
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.mbarrier_wait(mbar + w_slot, (step // Int32(self.w_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if step + Int32(1) < Int32(self.k_steps):
                        self._load_a_slot(step + Int32(1), (step + Int32(1)) % Int32(self.a_ring),
                                          sa, ss, src0, s_src0, ch, r_base, tidx)
                        cute.arch.cp_async_commit_group()
                    if warp_id == Int32(0):
                        if step + Int32(self.w_ring - 1) < Int32(self.k_steps):
                            self._issue_w(step + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                    if active0:
                        self._mma_k128(step, a_stage, w_slot, sa, sw, ss, sc, warp_m, warp_n, lane, g, acc0)

            cute.arch.sync_threads()
            self._epilogue_chunk(sa, acc0, tile0, first0, live0, n0, grouped_gamma, act, tidx, warp_m, warp_n, g, j)


class MiniMxfp8Down:
    """2D-TMA 2-chunk weight-stationary MXFP8 W8A8 Down projection.

    Grid ``(18, max_pairs)`` = ``(H // 128, max_pairs)``, 256 threads (4 M-warps x 2 N-warps).
    - ``w_ring = 3`` 128B-swizzled TMA boxes ``(128, 128)`` for w2 (49,152 B).
    - ``a_ring = 2`` 128B-swizzled TMA boxes ``(32, 128)`` x 4 for act (stride 1200, 32,768 B),
      issuing only ``ceil(live / 32)`` boxes.
    - Coalesced preloaded ``sb`` (128 x 36 B = 4,608 B, 9 words/row is odd -> 0 conflicts).
    - Act scales loaded directly via ``ld_global_nc_u32`` (0 SMEM, 0 cp.async in K-loop).
    - Total SMEM: 49,152 + 32,768 + 4,608 + 40 = 86,568 B <= 101,376 B.
    """

    threads = 256
    tile_n = 128
    k_block = 128
    w_ring = 3
    a_ring = 2
    box_m = 32
    out_stride = 128 * 2 + 16  # 272 B

    def __init__(self, *, hidden: int = 2304, inter: int = 1152, experts: int = 128):
        self.hidden, self.inter, self.experts = int(hidden), int(inter), int(experts)
        if self.hidden != 2304 or self.inter != 1152 or self.experts != 128:
            raise ValueError("MiniMxfp8Down is specialized for Inkling 2 Mini (H=2304, I=1152, E=128)")
        self.blocks = self.inter // self.k_block  # 9
        self.row_bytes = mxfp8_down_row_bytes(self.inter)      # 1200 B (multiple of 16)
        self.scale_row_bytes = mxfp4_scale_row_bytes(self.inter)  # 36 B (9 u32 words -> odd!)
        self.sb_words = self.blocks                            # 9
        self.sb_stride = self.sb_words * 4                     # 36 B (odd word stride -> 0 conflicts)
        self.w_stage_bytes = self.tile_n * self.k_block        # 16384 B
        self.box_bytes = self.box_m * self.k_block             # 4096 B
        self.a_stage_bytes = STREAM_TILE_M * self.k_block      # 16384 B

    def key(self) -> tuple:
        return ("mini_mxfp8_down", 1, self.hidden, self.inter, self.experts, self.w_ring, self.a_ring)

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_block),
            cutlass.Uint8)
        w_layout = cute.tile_to_shape(atom, (self.tile_n, self.k_block, self.w_ring), order=(0, 1, 2))
        a_layout = cute.tile_to_shape(atom, (self.box_m, self.k_block, 4 * self.a_ring), order=(0, 1, 2))
        return w_layout, a_layout

    def _storage(self, w_layout, a_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(a_layout)], 1024],
            "sb": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.tile_n * self.sb_stride], 16],
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring + self.a_ring],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, act: cute.Pointer, meta: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, max_pairs: Int32, stream: cuda.CUstream):
        w_layout, a_layout = self._layouts()
        w_box = cute.slice_(w_layout, (None, None, 0))
        a_box = cute.slice_(a_layout, (None, None, 0))
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            _u8_weight(w2, self.experts * self.hidden, self.inter), w_box,
            (self.tile_n, self.k_block), num_multicast=1)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            _u8_strided(act, max_tiles * Int32(STREAM_TILE_M), self.inter, self.row_bytes), a_box,
            (self.box_m, self.k_block), num_multicast=1)
        self.kernel(
            act, meta, tma_tensor_a, tma_tensor_w, s2, y,
            tma_a, tma_w, w_layout, a_layout, self._storage(w_layout, a_layout),
        ).launch(
            grid=(self.hidden // self.tile_n, max_pairs, 1),
            block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1,
        )

    @cute.jit
    def _issue_w(self, block: Int32, tma_w, t_wg, t_ws, w_mbar, w_tile: Int32):
        slot = block % Int32(self.w_ring)
        barrier = w_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.w_stage_bytes)
        cute.copy(tma_w, t_wg[(None, w_tile, block)], t_ws[(None, slot)], tma_bar_ptr=barrier)

    @cute.jit
    def _issue_a(self, a_step: Int32, tile: Int32, live_boxes: Int32, block: Int32,
                 tma_a, t_ag, t_as, a_mbar):
        slot = a_step % Int32(self.a_ring)
        barrier = a_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, live_boxes * Int32(self.box_bytes))
        base_row_tile = tile * Int32(4)
        base_slot = slot * Int32(4)
        for b in cutlass.range_constexpr(4):
            if Int32(b) < live_boxes:
                cute.copy(tma_a, t_ag[(None, base_row_tile + Int32(b), block)],
                          t_as[(None, base_slot + Int32(b))], tma_bar_ptr=barrier)

    @cute.jit
    def _mma_k128(self, block: Int32, a_base: Int32, w_base: Int32, asc0: Uint32, asc1: Uint32, ssb: Int32,
                  acc: cute.Tensor, warp_m: Int32, warp_n: Int32, lane: Int32, g: Int32):
        for ks in cutlass.range_constexpr(4):
            row0 = warp_m * Int32(32) + (lane % Int32(16))
            chunk_a = Int32(2 * ks) + lane // Int32(16)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                a_base + row0 * Int32(self.k_block) + ((chunk_a ^ (row0 % Int32(8))) * Int32(16)))
            a4, a5, a6, a7 = ldmatrix_m8n8x4_b16(
                a_base + (row0 + Int32(16)) * Int32(self.k_block) + ((chunk_a ^ (row0 % Int32(8))) * Int32(16)))
            chunk_b = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
            for np in cutlass.range_constexpr(4):
                nrow = warp_n * Int32(64) + Int32(16 * np) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                    w_base + nrow * Int32(self.k_block) + ((chunk_b ^ (nrow % Int32(8))) * Int32(16)))
                col0 = warp_n * Int32(64) + Int32(16 * np) + g
                bsc0 = ld_shared_u32(ssb + col0 * Int32(self.sb_stride) + block * Int32(4))
                bsc1 = ld_shared_u32(ssb + (col0 + Int32(8)) * Int32(self.sb_stride) + block * Int32(4))
                nt0 = 2 * np
                nt1 = nt0 + 1
                f00 = 4 * nt0
                f10 = 4 * (8 + nt0)
                f01 = 4 * nt1
                f11 = 4 * (8 + nt1)
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3],
                    a0, a1, a2, a3, b0, b1, asc0, bsc0, bid_a=ks, bid_b=ks)
                acc[f00], acc[f00 + 1], acc[f00 + 2], acc[f00 + 3] = d0, d1, d2, d3
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3],
                    a4, a5, a6, a7, b0, b1, asc1, bsc0, bid_a=ks, bid_b=ks)
                acc[f10], acc[f10 + 1], acc[f10 + 2], acc[f10 + 3] = d0, d1, d2, d3
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3],
                    a0, a1, a2, a3, b2, b3, asc0, bsc1, bid_a=ks, bid_b=ks)
                acc[f01], acc[f01 + 1], acc[f01 + 2], acc[f01 + 3] = d0, d1, d2, d3
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3],
                    a4, a5, a6, a7, b2, b3, asc1, bsc1, bid_a=ks, bid_b=ks)
                acc[f11], acc[f11 + 1], acc[f11 + 2], acc[f11 + 3] = d0, d1, d2, d3

    @cute.jit
    def _epilogue_chunk(self, out_smem: Int32, acc: cute.Tensor, first: Int32, live: Int32,
                        n0: Int32, y: cute.Pointer, tidx: Int32, warp_m: Int32, warp_n: Int32,
                        g: Int32, j: Int32):
        if warp_m * Int32(32) < live:
            for mt in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                    if row < live:
                        for nt in cutlass.range_constexpr(8):
                            f = 4 * (8 * mt + nt) + 2 * half
                            col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                            st_shared_u32(out_smem + row * Int32(self.out_stride) + col * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
        cute.arch.sync_threads()
        segments = self.tile_n * 2 // 16
        for qit in cutlass.range_constexpr(STREAM_TILE_M * segments // self.threads):
            ci = tidx + Int32(self.threads * qit)
            row = ci // Int32(segments)
            seg = ci % Int32(segments)
            if row < live:
                v0, v1, v2, v3 = ld_shared_v4_u32(out_smem + row * Int32(self.out_stride) + seg * Int32(16))
                st_global_v4_u32(
                    Int64(y.toint()) + ((Int64(first) + Int64(row)) * Int64(self.hidden) + Int64(n0)) * Int64(2)
                    + Int64(seg) * Int64(16),
                    v0, v1, v2, v3,
                )

    @cute.kernel
    def kernel(self, act: cute.Pointer, meta: cute.Pointer, act_t: cute.Tensor,
               weight_t: cute.Tensor, s2: cute.Pointer, y: cute.Pointer,
               tma_a: cute.CopyAtom, tma_w: cute.CopyAtom, w_layout: cute.ComposedLayout,
               a_layout: cute.ComposedLayout, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        pair_id = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sw = shared_ptr_to_u32(storage.w.data_ptr())
        sa = shared_ptr_to_u32(storage.a.data_ptr())
        ssb = shared_ptr_to_u32(storage.sb.data_ptr())

        e, tile0, first0, live0, live1 = _resolve_pair_128(meta, pair_id, tidx, warp_id, lane, sw)
        if e >= Int32(0):
            mbar = storage.mbar.data_ptr()
            w_mbar = mbar
            a_mbar = mbar + self.w_ring
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            s_a = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            t_ws, t_wg = cpasync.tma_partition(
                tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_t, (self.tile_n, self.k_block), (None, None)), 0, 2))
            t_as, t_ag = cpasync.tma_partition(
                tma_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2),
                cute.group_modes(cute.local_tile(act_t, (self.box_m, self.k_block), (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_w)
                cpasync.prefetch_descriptor(tma_a)
                for s in cutlass.range_constexpr(self.w_ring + self.a_ring):
                    cute.arch.mbarrier_init(mbar + s, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            n0 = n_blk * Int32(self.tile_n)
            w_tile = e * Int32(self.hidden // self.tile_n) + n_blk
            s_rows = Int64(s2.toint()) + (Int64(e) * Int64(self.hidden) + Int64(n0)) * Int64(self.scale_row_bytes)
            total_sb = self.tile_n * self.sb_words  # 1152
            for qsb in cutlass.range_constexpr((total_sb + self.threads - 1) // self.threads):
                u = tidx + Int32(qsb * self.threads)
                if u < Int32(total_sb):
                    sb_r = u // Int32(self.sb_words)
                    sb_c = u % Int32(self.sb_words)
                    cp_async_u32_shared_global(
                        ssb + sb_r * Int32(self.sb_stride) + sb_c * Int32(4),
                        s_rows + Int64(sb_r) * Int64(self.scale_row_bytes) + Int64(sb_c) * Int64(4))
            cute.arch.cp_async_commit_group()

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            g = lane // Int32(4)
            j = lane % Int32(4)
            live0_boxes = (live0 + Int32(31)) // Int32(32)
            srow_in_chunk = warp_m * Int32(32) + g + (lane & Int32(1)) * Int32(8)
            as0_row0 = Int64(act.toint()) + (Int64(tile0) * Int64(STREAM_TILE_M)
                                             + Int64(srow_in_chunk)) * Int64(self.row_bytes) + Int64(self.inter)
            active0 = warp_m * Int32(32) < live0

            acc0 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc0[v] = Float32(0.0)

            if live1 > Int32(0):
                tile1 = tile0 + Int32(1)
                live1_boxes = (live1 + Int32(31)) // Int32(32)
                as1_row0 = Int64(act.toint()) + (Int64(tile1) * Int64(STREAM_TILE_M)
                                                 + Int64(srow_in_chunk)) * Int64(self.row_bytes) + Int64(self.inter)
                active1 = warp_m * Int32(32) < live1
                acc1 = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(64):
                    acc1[v] = Float32(0.0)

                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._issue_w(Int32(pre), tma_w, t_wg, t_ws, w_mbar, w_tile)
                    self._issue_a(Int32(0), tile0, live0_boxes, Int32(0), tma_a, t_ag, t_as, a_mbar)
                cute.arch.cp_async_wait_group(0)

                total_a = Int32(2 * self.blocks)
                for block in cutlass.range(self.blocks, unroll=1):
                    w_slot = block % Int32(self.w_ring)
                    a0 = block * Int32(2)
                    a0_slot = a0 % Int32(self.a_ring)
                    asc0_0 = ld_global_nc_u32(as0_row0 + Int64(block) * Int64(4))
                    asc0_1 = ld_global_nc_u32(as0_row0 + Int64(16 * self.row_bytes) + Int64(block) * Int64(4))
                    cute.arch.mbarrier_wait(w_mbar + w_slot, (block // Int32(self.w_ring)) & Int32(1))
                    cute.arch.mbarrier_wait(a_mbar + a0_slot, (a0 // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if block + Int32(self.w_ring - 1) < Int32(self.blocks):
                            self._issue_w(block + Int32(self.w_ring - 1), tma_w, t_wg, t_ws, w_mbar, w_tile)
                        nxt0 = a0 + Int32(self.a_ring - 1)
                        if nxt0 < total_a:
                            nxt0_t = cutlass.select_((nxt0 & Int32(1)) == Int32(0), tile0, tile1)
                            nxt0_b = cutlass.select_((nxt0 & Int32(1)) == Int32(0), live0_boxes, live1_boxes)
                            self._issue_a(nxt0, nxt0_t, nxt0_b, nxt0 >> Int32(1), tma_a, t_ag, t_as, a_mbar)
                    w_base = sw + w_slot * Int32(self.w_stage_bytes)
                    if active0:
                        self._mma_k128(block, sa + a0_slot * Int32(self.a_stage_bytes), w_base,
                                       asc0_0, asc0_1, ssb, acc0, warp_m, warp_n, lane, g)

                    a1 = a0 + Int32(1)
                    a1_slot = a1 % Int32(self.a_ring)
                    asc1_0 = ld_global_nc_u32(as1_row0 + Int64(block) * Int64(4))
                    asc1_1 = ld_global_nc_u32(as1_row0 + Int64(16 * self.row_bytes) + Int64(block) * Int64(4))
                    cute.arch.mbarrier_wait(a_mbar + a1_slot, (a1 // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        nxt1 = a1 + Int32(self.a_ring - 1)
                        if nxt1 < total_a:
                            nxt1_t = cutlass.select_((nxt1 & Int32(1)) == Int32(0), tile0, tile1)
                            nxt1_b = cutlass.select_((nxt1 & Int32(1)) == Int32(0), live0_boxes, live1_boxes)
                            self._issue_a(nxt1, nxt1_t, nxt1_b, nxt1 >> Int32(1), tma_a, t_ag, t_as, a_mbar)
                    if active1:
                        self._mma_k128(block, sa + a1_slot * Int32(self.a_stage_bytes), w_base,
                                       asc1_0, asc1_1, ssb, acc1, warp_m, warp_n, lane, g)

                cute.arch.sync_threads()
                self._epilogue_chunk(sw, acc1, first0 + Int32(STREAM_TILE_M), live1, n0, y,
                                     tidx, warp_m, warp_n, g, j)
            else:
                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._issue_w(Int32(pre), tma_w, t_wg, t_ws, w_mbar, w_tile)
                    self._issue_a(Int32(0), tile0, live0_boxes, Int32(0), tma_a, t_ag, t_as, a_mbar)
                cute.arch.cp_async_wait_group(0)

                for block in cutlass.range(self.blocks, unroll=1):
                    w_slot = block % Int32(self.w_ring)
                    a_slot = block % Int32(self.a_ring)
                    asc0_0, asc0_1 = Uint32(0), Uint32(0)
                    if active0:
                        asc0_0 = ld_global_nc_u32(as0_row0 + Int64(block) * Int64(4))
                        asc0_1 = ld_global_nc_u32(as0_row0 + Int64(16 * self.row_bytes) + Int64(block) * Int64(4))
                    cute.arch.mbarrier_wait(w_mbar + w_slot, (block // Int32(self.w_ring)) & Int32(1))
                    cute.arch.mbarrier_wait(a_mbar + a_slot, (block // Int32(self.a_ring)) & Int32(1))
                    cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if block + Int32(self.w_ring - 1) < Int32(self.blocks):
                            self._issue_w(block + Int32(self.w_ring - 1), tma_w, t_wg, t_ws, w_mbar, w_tile)
                        if block + Int32(self.a_ring - 1) < Int32(self.blocks):
                            self._issue_a(block + Int32(self.a_ring - 1), tile0, live0_boxes,
                                          block + Int32(self.a_ring - 1), tma_a, t_ag, t_as, a_mbar)
                    if active0:
                        w_base = sw + w_slot * Int32(self.w_stage_bytes)
                        self._mma_k128(block, sa + a_slot * Int32(self.a_stage_bytes), w_base,
                                       asc0_0, asc0_1, ssb, acc0, warp_m, warp_n, lane, g)

            cute.arch.sync_threads()
            self._epilogue_chunk(sw, acc0, first0, live0, n0, y, tidx, warp_m, warp_n, g, j)
