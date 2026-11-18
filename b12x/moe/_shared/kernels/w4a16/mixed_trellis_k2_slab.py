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
    st_shared_i32,
    ldmatrix_m8n8x2_b16,
    packed_dequant_trellis_to_half2x4,
)
from .kernel import _SCALAR_ACC_FRAGMENT_WIDTH
from .mixed_trellis import W4A16MixedTrellisKernel


class W4A16MixedTrellisK2Slab(W4A16MixedTrellisKernel):
    # Per bank: K256 x N256 x 2 bits followed by M8 x K256 x FP16.
    pair_k2_route_blocks = True
    B_BYTES = 16384
    BANK_BYTES = B_BYTES + 4096

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep the original shared route metadata live across the main loop;
        # fold/store may reuse the slabs only after the final CTA rendezvous.
        self.slab_offset = self.tier0.fc1.sh_b_off * 16
        assert self.slab_offset == self.tier0.fc2.sh_b_off * 16
        bank_bytes = self.BANK_BYTES + (4096 if self.size_m != 1 else 0)
        self.shared_words = max(self.shared_words, (self.slab_offset + 2 * bank_bytes) // 4)

    @property
    def __cache_key__(self):
        return super().__cache_key__ + ("k2_m8_pair_slab_v2",)

    def _dispatch_bound(self, is_fc1, combined_expert, gemm, compact_weight, *args):
        if gemm.trellis_bits == 2:
            self._slab_tile(is_fc1, combined_expert, *args)
        else:
            super()._dispatch_bound(is_fc1, combined_expert, gemm, compact_weight, *args)

    @cute.jit
    def _fetch(self, dst: Int32, b_src: Int64, a_src: Int64, a_peer: Int64,
               slab: Int32, tid: Int32, row_stride: cutlass.Constexpr, policy,
               paired: cutlass.Constexpr):
        # Every warp fetches one whole 512-byte sector run. Adjacent pairs of
        # warps cover an N256 slice of a physical K16 row.
        for i in cutlass.range_constexpr(4):
            chunk = tid + Int32(i * 256)
            k16 = slab * Int32(16) + (chunk >> Int32(6))
            src = b_src + (k16 * Int32(row_stride) + (chunk & Int32(63)) * Int32(16)).to(Int64)
            cp_async4_shared_global_pred_l2hint(dst + chunk * Int32(16), src, Int32(1), policy)
        row = tid >> Int32(5)
        col = tid & Int32(31)
        a_dst = dst + Int32(self.B_BYTES) + (row * Int32(32) + (col ^ row)) * Int32(16)
        cp_async4_shared_global(a_dst, a_src + slab.to(Int64) * Int64(512))
        if cutlass.const_expr(paired):
            cp_async4_shared_global(a_dst + Int32(4096), a_peer + slab.to(Int64) * Int64(512))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _routes8(self, packed, block, live, smem, tid):
        # Resolve the eight rows once per warp, not via a CTA copy/wait/count
        # sequence. Each warp stages one row; padding borrows the first row,
        # but is never stored. Lane 0..7 of warp 0 publish the epilogue IDs.
        lane = tid & Int32(31)
        idx = packed[block * Int32(8) + (lane & Int32(7))].to(Int32)
        valid = cute.arch.warp_redux_sync(((lane < Int32(8)) & (idx < live)).to(Int32), "add")
        if tid < Int32(8):
            st_shared_i32(smem + tid * Int32(4), idx)
        first = cute.arch.shuffle_sync(idx, Int32(0))
        safe = idx
        if idx >= live:
            safe = first
        return cute.arch.shuffle_sync(safe, tid >> Int32(5)), valid

    @cute.jit
    def _slab_tile(self, is_fc1: cutlass.Constexpr, combined, a_flat, a_alt_flat, b_flat,
                   c_flat, scales_flat, global_scale, packed, topk_weights, c_tmp, locks,
                   lut_addr, smem, tid, block, expert, n_tile, reduce_k_tile,
                   reduce_tile_count, reduce_slice_count, reduce_slice_idx, lock_slot, active_m):
        adjacent = Int32(0)
        if cutlass.const_expr(self.size_m != 1):
            ids, count = self._k2_route_blocks
            # Pair even/odd neighbors only when both blocks belong to this
            # expert. Descriptor and scale are then identical for this N tile.
            peer = block ^ Int32(1)
            if peer < (count[Int32(0)].to(Int32) >> Int32(3)):
                adjacent = (ids[peer].to(Int32) == combined).to(Int32)
        if adjacent == Int32(0):
            self._tile_body(is_fc1, False, a_flat, a_alt_flat, b_flat, c_flat,
                            global_scale, packed, smem, tid, block, expert, n_tile, active_m)
        elif cutlass.const_expr(self.size_m != 1):
            if (block & Int32(1)) == Int32(0):
                self._tile_body(is_fc1, True, a_flat, a_alt_flat, b_flat, c_flat,
                                global_scale, packed, smem, tid, block, expert, n_tile, active_m)

    @cute.jit
    def _tile_body(self, is_fc1: cutlass.Constexpr, paired: cutlass.Constexpr, a_flat,
                   a_alt_flat, b_flat, c_flat, global_scale, packed, smem_base, tid,
                   block, expert, output_n_tile, active_m):
        gemm = self.tier0.fc2
        if cutlass.const_expr(is_fc1):
            gemm = self.tier0.fc1
        scale = global_scale[expert].to(cutlass.Float32)
        live = active_m * Int32(gemm.top_k)
        row, valid = self._routes8(packed, block, live, smem_base, tid)
        peer_row = Int32(0)
        peer_valid = Int32(0)
        if cutlass.const_expr(paired):
            peer_row, peer_valid = self._routes8(packed, block + Int32(1), live,
                                                  smem_base + Int32(32), tid)
        n16 = gemm.size_n // 16
        local_n = output_n_tile
        plane = Int64(0)
        a_base = get_ptr_as_int64(a_flat, Int32(0))
        if cutlass.const_expr(is_fc1):
            n16 = gemm.size_n // 32
            if output_n_tile >= Int32(gemm.n_tiles // 2):
                plane = Int64(cute.size(b_flat)) // Int64(2)
                local_n = output_n_tile - Int32(gemm.n_tiles // 2)
                a_base = get_ptr_as_int64(a_alt_flat, Int32(0))
        a_base += ((tid & Int32(31)) * Int32(16)).to(Int64)
        a_src = a_base + row.to(Int64) * Int64(2 * gemm.size_k)
        a_peer = a_base + peer_row.to(Int64) * Int64(2 * gemm.size_k)
        b_src = get_ptr_as_int64(b_flat, plane + expert.to(Int64) * Int64(gemm.size_k * n16)
                                 + local_n.to(Int64) * Int64(256))
        base = smem_base + Int32(self.slab_offset)
        policy = create_l2_evict_first_policy()
        self._fetch(base, b_src, a_src, a_peer, Int32(0), tid, n16 * 64, policy, paired)
        acc = [cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
               for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)]
        peer_acc = [cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                    for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)]
        for i in cutlass.range_constexpr(16 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc[i].fill(0.0)
            peer_acc[i].fill(0.0)
        slabs = gemm.size_k // 256
        stride = self.BANK_BYTES + (4096 if paired else 0)
        # Stage a compact runtime loop; the GEMM metadata is local to the
        # consume helper, never a loop-carried value. One barrier per slab.
        for slab in cutlass.range(slabs, unroll=1):
            bank = base + (slab & Int32(1)) * Int32(stride)
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if slab + Int32(1) < Int32(slabs):
                self._fetch(base + ((slab + Int32(1)) & Int32(1)) * Int32(stride),
                            b_src, a_src, a_peer, slab + Int32(1), tid, n16 * 64, policy, paired)
            self._consume(is_fc1, paired, bank, tid, acc, peer_acc)
        cute.arch.sync_threads()
        gemm._fold_cta_partials_m8(acc, smem_base, tid)
        gemm._store_tile_m8(acc, c_flat, smem_base, tid, output_n_tile, valid, scale)
        if cutlass.const_expr(paired):
            # Shift the epilogue base to the second published route vector.
            # Its fold scratch overlaps the first, now completely drained.
            peer_smem = smem_base + Int32(32)
            gemm._fold_cta_partials_m8(peer_acc, peer_smem, tid)
            gemm._store_tile_m8(peer_acc, c_flat, peer_smem, tid, output_n_tile, peer_valid, scale)

    @cute.jit
    def _consume(self, is_fc1: cutlass.Constexpr, paired: cutlass.Constexpr,
                 bank: Int32, tid: Int32, acc, peer_acc):
        gemm = self.tier0.fc2
        if cutlass.const_expr(is_fc1):
            gemm = self.tier0.fc1
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        a = cute.make_rmem_tensor((2,), Uint32)
        ap = cute.make_rmem_tensor((2,), Uint32)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        a_row = lane & Int32(7)
        word = lane >> Int32(1)
        for kt in cutlass.range_constexpr(4):
            for kk in cutlass.range_constexpr(2):
                k16 = Int32(4 * kt + kk) + (warp >> Int32(2)) * Int32(2)
                col = k16 * Int32(2) + ((lane >> Int32(3)) & Int32(1))
                addr = bank + Int32(self.B_BYTES) + (a_row * Int32(32) + (col ^ a_row)) * Int32(16)
                a[0], a[1] = ldmatrix_m8n8x2_b16(addr)
                if cutlass.const_expr(paired):
                    ap[0], ap[1] = ldmatrix_m8n8x2_b16(addr + Int32(4096))
                b_row = bank + k16 * Int32(1024) + (warp & Int32(3)) * Int32(256)
                for jj in cutlass.range_constexpr(4):
                    tile = b_row + Int32(64 * jj)
                    cur = ld_shared_u32(tile + word * Int32(4))
                    prev = ld_shared_u32(tile + ((word - Int32(1)) & Int32(15)) * Int32(4))
                    shift = ((lane + Int32(1)) & Int32(1)) * Int32(16)
                    win = gemm._trellis_funnel256(prev, cur, shift)
                    frag[0, 0], frag[0, 1], frag[1, 0], frag[1, 1] = (
                        packed_dequant_trellis_to_half2x4(win, win >> Uint32(8), 2))
                    gemm._mma_accumulate_m8(acc, jj, a, frag)
                    if cutlass.const_expr(paired):
                        gemm._mma_accumulate_m8(peer_acc, jj, ap, frag)
