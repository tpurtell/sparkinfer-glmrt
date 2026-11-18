"""Double-buffered K256 fetch slabs for the SM121 K2 M8 cooperative GEMMs.

A slab is four of the original K64 compute tiles, not a new reduction tile.
Each warp still accumulates the same two K16s per K64 in the same order.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async4_shared_global_pred_l2hint,
    create_l2_evict_first_policy,
    get_ptr_as_int64,
    ld_shared_u32,
    ldmatrix_m8n8x2_b16,
    packed_dequant_trellis_to_half2x4,
)
from .kernel import _SCALAR_ACC_FRAGMENT_WIDTH
from .mixed_trellis import W4A16MixedTrellisKernel


class W4A16MixedTrellisK2Slab(W4A16MixedTrellisKernel):
    # Per bank: K256 x N256 x 2 bits followed by M8 x K256 x FP16.
    B_BYTES = 16384
    BANK_BYTES = B_BYTES + 4096

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep the original shared route metadata live across the main loop;
        # fold/store may reuse the slabs only after the final CTA rendezvous.
        self.slab_offset = self.tier0.fc1.sh_b_off * 16
        assert self.slab_offset == self.tier0.fc2.sh_b_off * 16
        self.shared_words = max(self.shared_words, (self.slab_offset + 2 * self.BANK_BYTES) // 4)

    @property
    def __cache_key__(self):
        return super().__cache_key__ + ("k2_fetch_slab_v1",)

    def _dispatch_bound(self, is_fc1, combined_expert, gemm, compact_weight, *args):
        if gemm.trellis_bits == 2:
            self._slab_tile(gemm, *args)
        else:
            super()._dispatch_bound(is_fc1, combined_expert, gemm, compact_weight, *args)

    @cute.jit
    def _fetch(self, dst: Int32, b_src: Int64, a_src: Int64, slab: Int32,
               tid: Int32, row_stride: cutlass.Constexpr, policy):
        # Every warp fetches one whole 512-byte sector run. Adjacent pairs of
        # warps cover an N256 slice of a physical K16 row.
        for i in cutlass.range_constexpr(4):
            chunk = tid + Int32(i * 256)
            k16 = slab * Int32(16) + (chunk >> Int32(6))
            src = b_src + (k16 * Int32(row_stride) + (chunk & Int32(63)) * Int32(16)).to(Int64)
            cp_async4_shared_global_pred_l2hint(dst + chunk * Int32(16), src, Int32(1), policy)
        row = tid >> Int32(5)
        col = tid & Int32(31)
        cp_async4_shared_global(dst + Int32(self.B_BYTES) +
                                (row * Int32(32) + (col ^ (row & Int32(7)))) * Int32(16),
                                a_src + slab.to(Int64) * Int64(512))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _slab_tile(self, gemm, a_flat, a_alt_flat, b_flat, c_flat, scales_flat,
                   global_scale, packed, topk_weights, c_tmp, locks, lut_addr,
                   smem_base, tid, route_block, expert, output_n_tile, reduce_k_tile,
                   reduce_tile_count, reduce_slice_count, reduce_slice_idx, lock_slot, active_m):
        scale = global_scale[expert].to(cutlass.Float32)
        valid = gemm._read_moe_block_data(packed, topk_weights, smem_base, tid,
                                          route_block, scale, active_m)
        # Both rotated FC1 operands and activated FC2 rows are route-major.
        # Padding lanes read route zero, never an uninitialized shared route
        # (the m1 prologue writes just slot zero). Their outputs are not stored.
        row = tid >> Int32(5)
        route = Uint32(0)
        if row < valid:
            route = ld_shared_u32(smem_base + Int32(16 * gemm.sh_route_off) + row * Int32(4))
        a_off = route.to(Int64) * Int64(gemm.size_k) + (tid & Int32(31)).to(Int64) * Int64(8)
        a_src = get_ptr_as_int64(a_flat, a_off)
        n16 = gemm.size_n // 16
        local_n = output_n_tile
        plane = Int64(0)
        if cutlass.const_expr(gemm.dual_a):
            n16 = gemm.size_n // 32
            if output_n_tile >= Int32(gemm.n_tiles // 2):
                plane = Int64(cute.size(b_flat)) // Int64(2)
                local_n = output_n_tile - Int32(gemm.n_tiles // 2)
                a_src = get_ptr_as_int64(a_alt_flat, a_off)
        # K2 is 16 u32 per N16/K16 tile. Projection planes use the runtime
        # physical slot count, not the descriptor's logical expert count.
        b_src = get_ptr_as_int64(b_flat, plane + expert.to(Int64) * Int64(gemm.size_k * n16)
                                 + local_n.to(Int64) * Int64(256))
        base = smem_base + Int32(self.slab_offset)
        policy = create_l2_evict_first_policy()
        self._fetch(base, b_src, a_src, Int32(0), tid, n16 * 64, policy)
        acc = [cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
               for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)]
        for i in cutlass.range_constexpr(16 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc[i].fill(0.0)
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        a = cute.make_rmem_tensor((2,), Uint32)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        a_row = lane & Int32(7)
        word = lane >> Int32(1)
        slab = Int32(0)
        while slab < Int32(gemm.size_k // 256):
            bank = base + (slab & Int32(1)) * Int32(self.BANK_BYTES)
            # Publish this slab and retire all readers of the other bank
            # before its reuse. Its refill then overlaps this slab's MMAs.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if slab + Int32(1) < Int32(gemm.size_k // 256):
                self._fetch(base + ((slab + Int32(1)) & Int32(1)) * Int32(self.BANK_BYTES),
                            b_src, a_src, slab + Int32(1), tid, n16 * 64, policy)
            for kt in cutlass.range_constexpr(4):
                for kk in cutlass.range_constexpr(2):
                    k16 = Int32(4 * kt + kk) + (warp >> Int32(2)) * Int32(2)
                    col = k16 * Int32(2) + ((lane >> Int32(3)) & Int32(1))
                    a[0], a[1] = ldmatrix_m8n8x2_b16(bank + Int32(self.B_BYTES) +
                        (a_row * Int32(32) + (col ^ a_row)) * Int32(16))
                    b_row = bank + k16 * Int32(1024) + (warp & Int32(3)) * Int32(256)
                    for jj in cutlass.range_constexpr(4):
                        tile = b_row + Int32(64 * jj)
                        cur = ld_shared_u32(tile + word * Int32(4))
                        prev = ld_shared_u32(tile + ((word - Int32(1)) & Int32(15)) * Int32(4))
                        # Eight K2 states span only 30 bits; one aligned
                        # window supplies both halves of the stock decoder.
                        shift = ((lane + Int32(1)) & Int32(1)) * Int32(16)
                        win = gemm._trellis_funnel256(prev, cur, shift)
                        frag[0, 0], frag[0, 1], frag[1, 0], frag[1, 1] = (
                            packed_dequant_trellis_to_half2x4(win, win >> Uint32(8), 2))
                        gemm._mma_accumulate_m8(acc, jj, a, frag)
            slab += Int32(1)
        cute.arch.sync_threads()
        gemm._fold_cta_partials_m8(acc, smem_base, tid)
        gemm._store_tile_m8(acc, c_flat, smem_base, tid, output_n_tile, valid, scale)
