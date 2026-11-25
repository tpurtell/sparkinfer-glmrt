"""Specialized K8 MCG BF16 decode kernels for GLM 5.3 Flash (exact rows 1..64).

Only the ten GLM 5.3 Flash standalone K8 MCG BF16 projections reach this module.
All other trellis shapes, bitrates, codebooks, dtypes, and grouped paths stay on
the unchanged stock paths in ``_native.py``.

Every route preserves the exact K128 block-scale FMA order within each K1024
split and the exact ascending split sum order, producing bit-identical FP32
outputs to stock with smooth, cliff-free row curves across R = 1..64.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    ld_global_nc_v4_u32,
    ld_shared_u32,
    shared_ptr_to_u32,
    st_global_v4_f32,
    st_shared_v4_u32,
    u32_as_f32,
    weight_stream_policy,
)
from b12x.moe._shared.kernels.w4a16.exlr8 import _side_value


_GLMF_K8_SHAPES = frozenset({
    (4096, 4096, 1),  # kda_qkv_tp2
    (4096, 4096, 2),  # kda_o_tp2
    (4096, 1536, 1),  # q_a
    (4096, 512, 1),   # kv_a
    (1536, 8192, 1),  # q_b_tp2
    (8192, 4096, 2),  # o_tp2
    (4096, 1024, 1),  # shared_gate_up_tp2
    (1024, 4096, 2),  # shared_down_tp2
    (4096, 6144, 1),  # dense_gate_up_tp2
    (6144, 4096, 2),  # dense_down_tp2
})


def is_glmf_k8_mcg(g) -> bool:
    return (
        g["bits"] == 8
        and g["codebook"] == "mcg"
        and g.get("input_dtype") == "bf16"
        and g.get("groups", 1) == 1
        and g.get("rotation_count") == 1
        and not g.get("compact_head_shard", False)
        and (g["input_features"], g["output_features"], g["sharded_side"]) in _GLMF_K8_SHAPES
    )


def glmf_k8_route_thresholds(g) -> tuple[int, ...]:
    edges = (8, 16, 32, 48)
    cap = min(64, g["max_rows"])
    return tuple(edge for edge in edges if 0 < edge < cap)


def _route_for_shape(k: int, n: int, splits: int):
    """Per-bucket route specs for (M8, M16, M32, M48, M64) plus ``stock_narrow`` flag.

    Each entry is either:
      ("stock",)                        - delegate R<=8 to stock _decode_k8
      ("opt", fuse: bool, b_sub: int)   - N16-per-warp direct-L1 kernel
      ("smem4",)                        - SMEM-staged A, 4 warps x N16 (N64), b_sub=1, split-K
    """
    if splits == 1:
        # shared_down_tp2 (1024x4096): 256 fused CTAs, zero reduce, proven 0.662x.
        return (("opt", True, 4),) * 5, False
    if splits == 4 and n <= 1536:
        # shared_gate_up_tp2 (4096x1024), q_a (4096x1536), kv_a (4096x512): proven 0.587-0.720x.
        return (("opt", True, 4), ("opt", True, 4), ("opt", False, 4), ("opt", False, 4), ("opt", False, 4)), False
    if splits == 2 and n == 8192:
        # q_b_tp2 (1536x8192): fused 512 CTAs for R<=32, N64 SMEM-A for R>=33, proven 0.787x.
        return (("opt", True, 2), ("opt", True, 2), ("opt", True, 2), ("smem4",), ("smem4",)), False
    if splits == 4 and n == 4096:
        # kda_qkv_tp2 / kda_o_tp2 (4096x4096): fused b_sub=2 for R<=8, N64 SMEM-A for all R=9..64.
        return (("opt", True, 2), ("smem4",), ("smem4",), ("smem4",), ("smem4",)), False
    # dense_gate_up_tp2 (4096x6144), dense_down_tp2 (6144x4096), o_tp2 (8192x4096):
    # stock _decode_k8 at R<=8 (zero regression risk), N64 SMEM-A across all R=9..64.
    return (("stock",), ("smem4",), ("smem4",), ("smem4",), ("smem4",)), True


class GlmfK8Decode:
    """Cliff-free K8 MCG BF16 decode kernels for rows 1..64 on GLM 5.3 Flash shapes."""

    def __init__(self, g):
        self.k = g["input_features"]
        self.n = g["output_features"]
        self.splits = g["splits"]
        self.input_units = g["sharded_side"] == 2
        self.routes, self.stock_narrow = _route_for_shape(self.k, self.n, self.splits)

    def _smem_storage(self, num_f32: int):
        class Storage:
            pass

        Storage.__annotations__ = {
            "smem": cute.struct.Align[cute.struct.MemRange[Float32, max(16, num_f32)], 16],
        }
        return cute.struct(Storage)

    def _a_storage(self, m_rows: int):
        class Storage:
            pass

        Storage.__annotations__ = {
            "x": cute.struct.Align[cute.struct.MemRange[Uint32, 16 * (m_rows + 1) * 4], 16],
        }
        return cute.struct(Storage)

    @cute.kernel
    def multiply_smem64(
        self,
        inputs: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        partial: cute.Tensor,
        rows: Int32,
        rotation_id: Int32,
        active: cutlass.Constexpr,
    ):
        """SMEM-staged A kernel with 4 warps x N16 (N64 per CTA) and K8 v2+shfl prefetch.

        ``active in (1, 2, 3, 4)`` serves M16/M32/M48/M64. All 4 warps share the same K128
        block and read 4 contiguous ``n16_sub`` weight tiles (1 KB sequential per k16).
        """
        from ._native import (
            _decode_k8_words_shfl,
            _ld_l2_v2_off_pf,
            fma_rn_f32,
            st_global_v2_f32,
        )

        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        c, r_grp = tid & Int32(15), tid >> Int32(4)

        M = 16 * active
        M_PAD = M + 1
        load_iters = 2 * active
        guarantee_iters = 2 * (active - 1)
        prefetch_k16 = 8 if active <= 2 else 4

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._a_storage(M))
        sx = shared_ptr_to_u32(storage.x.data_ptr())
        sx_store = sx + (c * Int32(M_PAD) + r_grp) * Int32(16)
        sx_read = sx + lane * Int32(4)

        x = inputs[rotation_id]
        w_base = payload.iterator.toint() + lane.to(Int64) * Int64(8)
        prev_lane = (lane + Int32(31)) & Int32(31)
        policy = weight_stream_policy(Int32(1))

        n_tiles = self.n // 64
        tiles = Int32(n_tiles * self.splits)
        lane_r = lane >> Int32(2)

        while task < tiles:
            if cutlass.const_expr(self.input_units):
                tile_n = task % Int32(n_tiles)
                split = task // Int32(n_tiles)
            else:
                split = task % Int32(self.splits)
                tile_n = task // Int32(self.splits)

            n16_global = tile_n * Int32(4) + warp
            nb = n16_global >> Int32(3)
            n16_sub = n16_global & Int32(7)

            total = cute.make_rmem_tensor((active, 2, 4), Float32)
            total.fill(Float32(0.0))

            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    tile_kb = nb * Int32(self.k // 16 * 8) + kb * Int32(64) + n16_sub
                    if cutlass.const_expr(self.input_units):
                        tile_kb = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + n16_sub
                    w_kb = w_base + tile_kb.to(Int64) * Int64(256)

                    # Issue initial weight prefetch BEFORE staging A + sync_threads to hide DRAM latency.
                    u0_buf = cute.make_rmem_tensor((prefetch_k16,), Uint32)
                    u1_buf = cute.make_rmem_tensor((prefetch_k16,), Uint32)
                    for ki in cutlass.range_constexpr(prefetch_k16):
                        u0, u1 = _ld_l2_v2_off_pf(w_kb, ki * 2048, policy)
                        u0_buf[ki], u1_buf[ki] = u0, u1

                    k_col = kb * Int32(128) + c * Int32(8)
                    for s in cutlass.range_constexpr(load_iters):
                        row = r_grp + Int32(s * 8)
                        v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if cutlass.const_expr(s < guarantee_iters):
                            v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2)
                            )
                        else:
                            if row < rows:
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    x + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2)
                                )
                        st_shared_v4_u32(sx_store + Int32(s * 8 * 16), v0, v1, v2, v3)
                    cute.arch.sync_threads()

                    acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                    acc.fill(Float32(0.0))
                    for ko in cutlass.range_constexpr(8 // prefetch_k16):
                        k_base = ko * prefetch_k16
                        if cutlass.const_expr(ko > 0):
                            for ki in cutlass.range_constexpr(prefetch_k16):
                                u0, u1 = _ld_l2_v2_off_pf(w_kb, (k_base + ki) * 2048, policy)
                                u0_buf[ki], u1_buf[ki] = u0, u1
                        for ki in cutlass.range_constexpr(prefetch_k16):
                            k16 = k_base + ki
                            b0, b1, b2, b3 = _decode_k8_words_shfl(
                                u0_buf[ki], u1_buf[ki], prev_lane, False, False, True
                            )
                            for mb in cutlass.range_constexpr(active):
                                a0 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                a1 = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                a2 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                a3 = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                    acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                    a0, a1, a2, a3, b0, b1,
                                )
                                acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                    acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                    a0, a1, a2, a3, b2, b3,
                                )
                                acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    for mb in cutlass.range_constexpr(active):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                total[mb, n, v] = fma_rn_f32(acc[mb, n, v], scale, total[mb, n, v])
                    cute.arch.sync_threads()

            col_base = n16_global * Int32(16) + (lane & Int32(3)) * Int32(2)
            p_base = partial.iterator.toint()
            for mb in cutlass.range_constexpr(active):
                row0 = lane_r + Int32(mb * 16)
                row1 = row0 + Int32(8)
                p_off = (
                    (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n)
                    + col_base.to(Int64)
                ) * Int64(4)
                p_addr = p_base + p_off
                if row0 < rows:
                    for n in cutlass.range_constexpr(2):
                        st_global_v2_f32(p_addr + Int64(n * 32), total[mb, n, 0], total[mb, n, 1])
                if row1 < rows:
                    p_addr1 = p_addr + Int64(8 * self.splits * self.n * 4)
                    for n in cutlass.range_constexpr(2):
                        st_global_v2_f32(p_addr1 + Int64(n * 32), total[mb, n, 2], total[mb, n, 3])
            task += step

    @cute.kernel
    def multiply_narrow(
        self,
        inputs: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        rotation_id: Int32,
        fuse_splits: cutlass.Constexpr,
        b_sub: cutlass.Constexpr,
    ):
        from ._native import (
            _decode_k8_words_shfl,
            _ld_l2_v2_off_pf,
            fma_rn_f32,
            ld_global_nc_u32_offset,
            ld_shared_v2_f32_offset,
            st_global_v2_f32,
            st_shared_v2_f32_offset,
        )

        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        sub = warp % Int32(b_sub) if cutlass.const_expr(b_sub > 1) else Int32(0)
        split_local = warp // Int32(b_sub) if cutlass.const_expr(b_sub > 1) else warp

        blocks_per_sub = 8 // b_sub
        n_groups = self.splits if fuse_splits else 1
        phase_a_slots = n_groups * (8 - blocks_per_sub) if b_sub > 1 else 0
        phase_b_slots = (self.splits - 1) if (fuse_splits and self.splits > 1) else 0
        total_f32 = (phase_a_slots + phase_b_slots) * 128
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._smem_storage(total_f32))
        smem_lane = shared_ptr_to_u32(storage.smem.data_ptr()) + lane * Int32(8)

        x = inputs[rotation_id]
        w_base = payload.iterator.toint() + lane.to(Int64) * Int64(8)
        prev_lane = (lane + Int32(31)) & Int32(31)
        policy = weight_stream_policy(Int32(1))

        row0 = lane >> Int32(2)
        r0c = min(row0, rows - Int32(1))
        x0_row = x + (r0c.to(Int64) * Int64(self.k) + ((lane & Int32(3)) * Int32(2)).to(Int64)) * Int64(2)

        n_tiles = self.n // 16
        tiles = Int32(n_tiles) if cutlass.const_expr(fuse_splits) else Int32(n_tiles * self.splits)

        while task < tiles:
            if cutlass.const_expr(fuse_splits):
                n16_global = task
                split = split_local
            else:
                if cutlass.const_expr(self.input_units):
                    n16_global = task % Int32(n_tiles)
                    split = task // Int32(n_tiles)
                else:
                    split = task % Int32(self.splits)
                    n16_global = task // Int32(self.splits)

            nb = n16_global >> Int32(3)
            n16_sub = n16_global & Int32(7)

            total = cute.make_rmem_tensor((2, 2), Float32)
            total.fill(Float32(0.0))

            for b_local in cutlass.range(blocks_per_sub, unroll=1):
                block = Int32(sub * blocks_per_sub) + b_local
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((2, 2), Float32)
                    acc.fill(Float32(0.0))
                    x0_kb = x0_row + kb.to(Int64) * Int64(256)
                    tile_kb = nb * Int32(self.k // 16 * 8) + kb * Int32(64) + n16_sub
                    if cutlass.const_expr(self.input_units):
                        tile_kb = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + n16_sub
                    w_kb = w_base + tile_kb.to(Int64) * Int64(256)

                    u0_buf = cute.make_rmem_tensor((8,), Uint32)
                    u1_buf = cute.make_rmem_tensor((8,), Uint32)
                    for k16 in cutlass.range_constexpr(8):
                        u0, u1 = _ld_l2_v2_off_pf(w_kb, k16 * 2048, policy)
                        u0_buf[k16], u1_buf[k16] = u0, u1
                    for k16 in cutlass.range_constexpr(8):
                        a0 = ld_global_nc_u32_offset(x0_kb, k16 * 32)
                        a2 = ld_global_nc_u32_offset(x0_kb, k16 * 32 + 16)
                        b0, b1, b2, b3 = _decode_k8_words_shfl(
                            u0_buf[k16], u1_buf[k16], prev_lane, False, False, True
                        )
                        d0, d1, _, _ = bf16_mma_m16n8k16_f32(
                            acc[0, 0], acc[0, 1], Float32(0.0), Float32(0.0),
                            a0, Uint32(0), a2, Uint32(0), b0, b1,
                        )
                        acc[0, 0], acc[0, 1] = d0, d1
                        d0, d1, _, _ = bf16_mma_m16n8k16_f32(
                            acc[1, 0], acc[1, 1], Float32(0.0), Float32(0.0),
                            a0, Uint32(0), a2, Uint32(0), b2, b3,
                        )
                        acc[1, 0], acc[1, 1] = d0, d1

                    if cutlass.const_expr(b_sub == 1):
                        scale = _side_value(scales.iterator.toint(), kb)
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(2):
                                total[n, v] = fma_rn_f32(acc[n, v], scale, total[n, v])
                    else:
                        if sub == Int32(0):
                            scale = _side_value(scales.iterator.toint(), kb)
                            for n in cutlass.range_constexpr(2):
                                for v in cutlass.range_constexpr(2):
                                    total[n, v] = fma_rn_f32(acc[n, v], scale, total[n, v])
                        else:
                            b_rem_dyn = (sub - Int32(1)) * Int32(blocks_per_sub) + b_local
                            slot_addr = smem_lane + (
                                split_local * Int32(8 - blocks_per_sub) + b_rem_dyn
                            ) * Int32(512)
                            for n in cutlass.range_constexpr(2):
                                st_shared_v2_f32_offset(slot_addr, n * 256, acc[n, 0], acc[n, 1])

            if cutlass.const_expr(b_sub > 1):
                cute.arch.sync_threads()
                if sub == Int32(0):
                    group_base = smem_lane + split_local * Int32((8 - blocks_per_sub) * 512)
                    for b_rem in cutlass.range_constexpr(8 - blocks_per_sub):
                        kb_rem = split * Int32(8) + Int32(blocks_per_sub + b_rem)
                        if cutlass.const_expr(self.k % 1024 == 0):
                            scale_rem = _side_value(scales.iterator.toint(), kb_rem)
                            for n in cutlass.range_constexpr(2):
                                t0, t1 = ld_shared_v2_f32_offset(group_base, b_rem * 512 + n * 256)
                                total[n, 0] = fma_rn_f32(t0, scale_rem, total[n, 0])
                                total[n, 1] = fma_rn_f32(t1, scale_rem, total[n, 1])
                        else:
                            if kb_rem < Int32(self.k // 128):
                                scale_rem = _side_value(scales.iterator.toint(), kb_rem)
                                for n in cutlass.range_constexpr(2):
                                    t0, t1 = ld_shared_v2_f32_offset(group_base, b_rem * 512 + n * 256)
                                    total[n, 0] = fma_rn_f32(t0, scale_rem, total[n, 0])
                                    total[n, 1] = fma_rn_f32(t1, scale_rem, total[n, 1])

            col_base = n16_global * Int32(16) + (lane & Int32(3)) * Int32(2)
            if cutlass.const_expr(not fuse_splits):
                if sub == Int32(0) and row0 < rows:
                    p_off = (
                        (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n)
                        + col_base.to(Int64)
                    ) * Int64(4)
                    p_addr = partial.iterator.toint() + p_off
                    for n in cutlass.range_constexpr(2):
                        st_global_v2_f32(p_addr + Int64(n * 32), total[n, 0], total[n, 1])
            elif cutlass.const_expr(self.splits == 1):
                if sub == Int32(0) and row0 < rows:
                    o_off = (row0.to(Int64) * Int64(self.n) + col_base.to(Int64)) * Int64(4)
                    o_addr = output.iterator.toint() + o_off
                    for n in cutlass.range_constexpr(2):
                        st_global_v2_f32(
                            o_addr + Int64(n * 32), Float32(0.0) + total[n, 0], Float32(0.0) + total[n, 1]
                        )
            else:
                if sub == Int32(0) and split_local > Int32(0):
                    pb_addr = smem_lane + Int32(phase_a_slots * 512) + (split_local - Int32(1)) * Int32(512)
                    for n in cutlass.range_constexpr(2):
                        st_shared_v2_f32_offset(pb_addr, n * 256, total[n, 0], total[n, 1])
                cute.arch.sync_threads()
                if sub == Int32(0) and split_local == Int32(0) and row0 < rows:
                    pb_base = smem_lane + Int32(phase_a_slots * 512)
                    o_off = (row0.to(Int64) * Int64(self.n) + col_base.to(Int64)) * Int64(4)
                    o_addr = output.iterator.toint() + o_off
                    for n in cutlass.range_constexpr(2):
                        s0 = Float32(0.0) + total[n, 0]
                        s1 = Float32(0.0) + total[n, 1]
                        for sp in cutlass.range_constexpr(self.splits - 1):
                            p0, p1 = ld_shared_v2_f32_offset(pb_base, sp * 512 + n * 256)
                            s0 = s0 + p0
                            s1 = s1 + p1
                        st_global_v2_f32(o_addr + Int64(n * 32), s0, s1)

            if task + step < tiles:
                cute.arch.sync_threads()
            task += step

    @cute.kernel
    def multiply_wide(
        self,
        inputs: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        rotation_id: Int32,
        active: cutlass.Constexpr,
        fuse_splits: cutlass.Constexpr,
        b_sub: cutlass.Constexpr,
    ):
        from ._native import (
            _decode_k8_words_shfl,
            _ld_l2_v2_off_pf,
            fma_rn_f32,
            ld_global_nc_u32_offset,
            ld_shared_v2_f32_offset,
            st_global_v2_f32,
            st_shared_v2_f32_offset,
        )

        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        sub = warp % Int32(b_sub) if cutlass.const_expr(b_sub > 1) else Int32(0)
        split_local = warp // Int32(b_sub) if cutlass.const_expr(b_sub > 1) else warp

        blocks_per_sub = 8 // b_sub
        n_groups = self.splits if fuse_splits else 1
        phase_a_slots = n_groups * (8 - blocks_per_sub) if b_sub > 1 else 0
        phase_b_slots = (self.splits - 1) if (fuse_splits and self.splits > 1) else 0
        slot_bytes = active * 1024
        total_f32 = (phase_a_slots + phase_b_slots) * active * 256
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._smem_storage(total_f32))
        smem_lane = shared_ptr_to_u32(storage.smem.data_ptr()) + lane * Int32(8)

        x = inputs[rotation_id]
        w_base = payload.iterator.toint() + lane.to(Int64) * Int64(8)
        prev_lane = (lane + Int32(31)) & Int32(31)
        policy = weight_stream_policy(Int32(1))

        lane_r = lane >> Int32(2)
        lane_k_bytes = ((lane & Int32(3)) * Int32(2)).to(Int64) * Int64(2)
        row_bytes = Int64(self.k * 2)
        x_base = x + lane_r.to(Int64) * row_bytes + lane_k_bytes
        r0_last = min(Int32((active - 1) * 16) + lane_r, rows - Int32(1))
        r1_last = min(Int32((active - 1) * 16 + 8) + lane_r, rows - Int32(1))
        x_last0 = x + r0_last.to(Int64) * row_bytes + lane_k_bytes
        x_last1 = x + r1_last.to(Int64) * row_bytes + lane_k_bytes

        prefetch_k16 = 8 if active <= 2 else 4
        n_tiles = self.n // 16
        tiles = Int32(n_tiles) if cutlass.const_expr(fuse_splits) else Int32(n_tiles * self.splits)

        while task < tiles:
            if cutlass.const_expr(fuse_splits):
                n16_global = task
                split = split_local
            else:
                if cutlass.const_expr(self.input_units):
                    n16_global = task % Int32(n_tiles)
                    split = task // Int32(n_tiles)
                else:
                    split = task % Int32(self.splits)
                    n16_global = task // Int32(self.splits)

            nb = n16_global >> Int32(3)
            n16_sub = n16_global & Int32(7)

            total = cute.make_rmem_tensor((active, 2, 4), Float32)
            total.fill(Float32(0.0))

            for b_local in cutlass.range(blocks_per_sub, unroll=1):
                block = Int32(sub * blocks_per_sub) + b_local
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                    acc.fill(Float32(0.0))
                    kb_bytes = kb.to(Int64) * Int64(256)
                    x_base_kb = x_base + kb_bytes
                    x_last0_kb = x_last0 + kb_bytes
                    x_last1_kb = x_last1 + kb_bytes
                    tile_kb = nb * Int32(self.k // 16 * 8) + kb * Int32(64) + n16_sub
                    if cutlass.const_expr(self.input_units):
                        tile_kb = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + n16_sub
                    w_kb = w_base + tile_kb.to(Int64) * Int64(256)

                    for ko in cutlass.range_constexpr(8 // prefetch_k16):
                        k_base = ko * prefetch_k16
                        u0_buf = cute.make_rmem_tensor((prefetch_k16,), Uint32)
                        u1_buf = cute.make_rmem_tensor((prefetch_k16,), Uint32)
                        for ki in cutlass.range_constexpr(prefetch_k16):
                            u0, u1 = _ld_l2_v2_off_pf(w_kb, (k_base + ki) * 2048, policy)
                            u0_buf[ki], u1_buf[ki] = u0, u1

                        for ki in cutlass.range_constexpr(prefetch_k16):
                            k_off = (k_base + ki) * 32
                            if cutlass.const_expr(active == 1):
                                a0 = ld_global_nc_u32_offset(x_last0_kb, k_off)
                                a2 = ld_global_nc_u32_offset(x_last0_kb, k_off + 16)
                                a1 = ld_global_nc_u32_offset(x_last1_kb, k_off)
                                a3 = ld_global_nc_u32_offset(x_last1_kb, k_off + 16)
                            else:
                                a0 = ld_global_nc_u32_offset(x_base_kb, k_off)
                                a2 = ld_global_nc_u32_offset(x_base_kb, k_off + 16)
                                a1 = ld_global_nc_u32_offset(x_base_kb + Int64(8 * self.k * 2), k_off)
                                a3 = ld_global_nc_u32_offset(x_base_kb + Int64(8 * self.k * 2), k_off + 16)

                            b0, b1, b2, b3 = _decode_k8_words_shfl(
                                u0_buf[ki], u1_buf[ki], prev_lane, False, False, True
                            )

                            for mb in cutlass.range_constexpr(active):
                                na0, na1, na2, na3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                                if cutlass.const_expr(mb + 1 < active):
                                    if cutlass.const_expr(mb + 1 == active - 1):
                                        na0 = ld_global_nc_u32_offset(x_last0_kb, k_off)
                                        na2 = ld_global_nc_u32_offset(x_last0_kb, k_off + 16)
                                        na1 = ld_global_nc_u32_offset(x_last1_kb, k_off)
                                        na3 = ld_global_nc_u32_offset(x_last1_kb, k_off + 16)
                                    else:
                                        p0 = x_base_kb + Int64((mb + 1) * 16 * self.k * 2)
                                        p1 = x_base_kb + Int64(((mb + 1) * 16 + 8) * self.k * 2)
                                        na0 = ld_global_nc_u32_offset(p0, k_off)
                                        na2 = ld_global_nc_u32_offset(p0, k_off + 16)
                                        na1 = ld_global_nc_u32_offset(p1, k_off)
                                        na3 = ld_global_nc_u32_offset(p1, k_off + 16)

                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                    acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3],
                                    a0, a1, a2, a3, b0, b1,
                                )
                                acc[mb, 0, 0], acc[mb, 0, 1], acc[mb, 0, 2], acc[mb, 0, 3] = d0, d1, d2, d3
                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                    acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3],
                                    a0, a1, a2, a3, b2, b3,
                                )
                                acc[mb, 1, 0], acc[mb, 1, 1], acc[mb, 1, 2], acc[mb, 1, 3] = d0, d1, d2, d3
                                if cutlass.const_expr(mb + 1 < active):
                                    a0, a1, a2, a3 = na0, na1, na2, na3

                    if cutlass.const_expr(b_sub == 1):
                        scale = _side_value(scales.iterator.toint(), kb)
                        for mb in cutlass.range_constexpr(active):
                            for n in cutlass.range_constexpr(2):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, n, v] = fma_rn_f32(acc[mb, n, v], scale, total[mb, n, v])
                    else:
                        if sub == Int32(0):
                            scale = _side_value(scales.iterator.toint(), kb)
                            for mb in cutlass.range_constexpr(active):
                                for n in cutlass.range_constexpr(2):
                                    for v in cutlass.range_constexpr(4):
                                        total[mb, n, v] = fma_rn_f32(acc[mb, n, v], scale, total[mb, n, v])
                        else:
                            b_rem_dyn = (sub - Int32(1)) * Int32(blocks_per_sub) + b_local
                            slot_addr = smem_lane + (
                                split_local * Int32(8 - blocks_per_sub) + b_rem_dyn
                            ) * Int32(slot_bytes)
                            for mb in cutlass.range_constexpr(active):
                                for n in cutlass.range_constexpr(2):
                                    st_shared_v2_f32_offset(
                                        slot_addr, mb * 1024 + n * 256, acc[mb, n, 0], acc[mb, n, 1]
                                    )
                                    st_shared_v2_f32_offset(
                                        slot_addr, mb * 1024 + n * 256 + 512, acc[mb, n, 2], acc[mb, n, 3]
                                    )

            if cutlass.const_expr(b_sub > 1):
                cute.arch.sync_threads()
                if sub == Int32(0):
                    group_base = smem_lane + split_local * Int32((8 - blocks_per_sub) * slot_bytes)
                    for b_rem in cutlass.range_constexpr(8 - blocks_per_sub):
                        kb_rem = split * Int32(8) + Int32(blocks_per_sub + b_rem)
                        if cutlass.const_expr(self.k % 1024 == 0):
                            scale_rem = _side_value(scales.iterator.toint(), kb_rem)
                            for mb in cutlass.range_constexpr(active):
                                for n in cutlass.range_constexpr(2):
                                    off = b_rem * slot_bytes + mb * 1024 + n * 256
                                    t0, t1 = ld_shared_v2_f32_offset(group_base, off)
                                    t2, t3 = ld_shared_v2_f32_offset(group_base, off + 512)
                                    total[mb, n, 0] = fma_rn_f32(t0, scale_rem, total[mb, n, 0])
                                    total[mb, n, 1] = fma_rn_f32(t1, scale_rem, total[mb, n, 1])
                                    total[mb, n, 2] = fma_rn_f32(t2, scale_rem, total[mb, n, 2])
                                    total[mb, n, 3] = fma_rn_f32(t3, scale_rem, total[mb, n, 3])
                        else:
                            if kb_rem < Int32(self.k // 128):
                                scale_rem = _side_value(scales.iterator.toint(), kb_rem)
                                for mb in cutlass.range_constexpr(active):
                                    for n in cutlass.range_constexpr(2):
                                        off = b_rem * slot_bytes + mb * 1024 + n * 256
                                        t0, t1 = ld_shared_v2_f32_offset(group_base, off)
                                        t2, t3 = ld_shared_v2_f32_offset(group_base, off + 512)
                                        total[mb, n, 0] = fma_rn_f32(t0, scale_rem, total[mb, n, 0])
                                        total[mb, n, 1] = fma_rn_f32(t1, scale_rem, total[mb, n, 1])
                                        total[mb, n, 2] = fma_rn_f32(t2, scale_rem, total[mb, n, 2])
                                        total[mb, n, 3] = fma_rn_f32(t3, scale_rem, total[mb, n, 3])

            col_base = n16_global * Int32(16) + (lane & Int32(3)) * Int32(2)
            if cutlass.const_expr(not fuse_splits):
                if sub == Int32(0):
                    p_base = partial.iterator.toint()
                    for mb in cutlass.range_constexpr(active):
                        row0 = lane_r + Int32(mb * 16)
                        row1 = row0 + Int32(8)
                        p_off = (
                            (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n)
                            + col_base.to(Int64)
                        ) * Int64(4)
                        p_addr = p_base + p_off
                        if row0 < rows:
                            for n in cutlass.range_constexpr(2):
                                st_global_v2_f32(p_addr + Int64(n * 32), total[mb, n, 0], total[mb, n, 1])
                        if row1 < rows:
                            p_addr1 = p_addr + Int64(8 * self.splits * self.n * 4)
                            for n in cutlass.range_constexpr(2):
                                st_global_v2_f32(p_addr1 + Int64(n * 32), total[mb, n, 2], total[mb, n, 3])
            elif cutlass.const_expr(self.splits == 1):
                if sub == Int32(0):
                    o_base = output.iterator.toint()
                    for mb in cutlass.range_constexpr(active):
                        row0 = lane_r + Int32(mb * 16)
                        row1 = row0 + Int32(8)
                        o_off = (row0.to(Int64) * Int64(self.n) + col_base.to(Int64)) * Int64(4)
                        o_addr = o_base + o_off
                        if row0 < rows:
                            for n in cutlass.range_constexpr(2):
                                st_global_v2_f32(
                                    o_addr + Int64(n * 32),
                                    Float32(0.0) + total[mb, n, 0],
                                    Float32(0.0) + total[mb, n, 1],
                                )
                        if row1 < rows:
                            o_addr1 = o_addr + Int64(8 * self.n * 4)
                            for n in cutlass.range_constexpr(2):
                                st_global_v2_f32(
                                    o_addr1 + Int64(n * 32),
                                    Float32(0.0) + total[mb, n, 2],
                                    Float32(0.0) + total[mb, n, 3],
                                )
            else:
                if sub == Int32(0) and split_local > Int32(0):
                    pb_addr = smem_lane + Int32(phase_a_slots * slot_bytes) + (
                        split_local - Int32(1)
                    ) * Int32(slot_bytes)
                    for mb in cutlass.range_constexpr(active):
                        for n in cutlass.range_constexpr(2):
                            st_shared_v2_f32_offset(
                                pb_addr, mb * 1024 + n * 256, total[mb, n, 0], total[mb, n, 1]
                            )
                            st_shared_v2_f32_offset(
                                pb_addr, mb * 1024 + n * 256 + 512, total[mb, n, 2], total[mb, n, 3]
                            )
                cute.arch.sync_threads()
                if sub == Int32(0) and split_local == Int32(0):
                    pb_base = smem_lane + Int32(phase_a_slots * slot_bytes)
                    o_base = output.iterator.toint()
                    for mb in cutlass.range_constexpr(active):
                        row0 = lane_r + Int32(mb * 16)
                        row1 = row0 + Int32(8)
                        o_off = (row0.to(Int64) * Int64(self.n) + col_base.to(Int64)) * Int64(4)
                        o_addr = o_base + o_off
                        if row0 < rows:
                            for n in cutlass.range_constexpr(2):
                                s0 = Float32(0.0) + total[mb, n, 0]
                                s1 = Float32(0.0) + total[mb, n, 1]
                                for sp in cutlass.range_constexpr(self.splits - 1):
                                    p0, p1 = ld_shared_v2_f32_offset(
                                        pb_base, sp * slot_bytes + mb * 1024 + n * 256
                                    )
                                    s0 = s0 + p0
                                    s1 = s1 + p1
                                st_global_v2_f32(o_addr + Int64(n * 32), s0, s1)
                        if row1 < rows:
                            o_addr1 = o_addr + Int64(8 * self.n * 4)
                            for n in cutlass.range_constexpr(2):
                                s2 = Float32(0.0) + total[mb, n, 2]
                                s3 = Float32(0.0) + total[mb, n, 3]
                                for sp in cutlass.range_constexpr(self.splits - 1):
                                    p2, p3 = ld_shared_v2_f32_offset(
                                        pb_base, sp * slot_bytes + mb * 1024 + n * 256 + 512
                                    )
                                    s2 = s2 + p2
                                    s3 = s3 + p3
                                st_global_v2_f32(o_addr1 + Int64(n * 32), s2, s3)

            if task + step < tiles:
                cute.arch.sync_threads()
            task += step

    @cute.kernel
    def reduce_128(self, partial: cute.Tensor, output: cute.Tensor, rows: Int32):
        """One warp per 128 output columns (v4 coalesced, exact split sum order)."""
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        n_blocks = Int32(self.n // 128)
        p_ptr = partial.iterator.toint()
        o_ptr = output.iterator.toint()
        while task < rows * n_blocks:
            row = task // n_blocks
            col = (task % n_blocks) * Int32(128) + Int32(tid) * Int32(4)
            t0, t1, t2, t3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            row_base = p_ptr + (row.to(Int64) * Int64(self.splits * self.n) + col.to(Int64)) * Int64(4)
            for split in cutlass.range_constexpr(self.splits):
                v0, v1, v2, v3 = ld_global_nc_v4_u32(row_base + Int64(split * self.n * 4))
                t0 = t0 + u32_as_f32(v0)
                t1 = t1 + u32_as_f32(v1)
                t2 = t2 + u32_as_f32(v2)
                t3 = t3 + u32_as_f32(v3)
            out_addr = o_ptr + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
            st_global_v4_f32(out_addr, t0, t1, t2, t3)
            task += step

    @cute.kernel
    def reduce_512(self, partial: cute.Tensor, output: cute.Tensor, rows: Int32):
        """Four warps per 512 output columns (high warps/SM when rows * N/128 > cta_limit)."""
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        n_blocks = Int32(self.n // 512)
        p_ptr = partial.iterator.toint()
        o_ptr = output.iterator.toint()
        while task < rows * n_blocks:
            row = task // n_blocks
            col = (task % n_blocks) * Int32(512) + Int32(tid) * Int32(4)
            t0, t1, t2, t3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            row_base = p_ptr + (row.to(Int64) * Int64(self.splits * self.n) + col.to(Int64)) * Int64(4)
            for split in cutlass.range_constexpr(self.splits):
                v0, v1, v2, v3 = ld_global_nc_v4_u32(row_base + Int64(split * self.n * 4))
                t0 = t0 + u32_as_f32(v0)
                t1 = t1 + u32_as_f32(v1)
                t2 = t2 + u32_as_f32(v2)
                t3 = t3 + u32_as_f32(v3)
            out_addr = o_ptr + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
            st_global_v4_f32(out_addr, t0, t1, t2, t3)
            task += step

    @cute.jit
    def _launch_reduce(
        self,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
    ):
        blocks_128 = rows * Int32(self.n // 128)
        if blocks_128 <= cta_limit:
            self.reduce_128(partial, output, rows).launch(
                grid=(blocks_128, 1, 1), block=(32, 1, 1), stream=stream
            )
        else:
            blocks_512 = rows * Int32(self.n // 512)
            self.reduce_512(partial, output, rows).launch(
                grid=(min(blocks_512, cta_limit), 1, 1), block=(128, 1, 1), stream=stream
            )

    @cute.jit
    def _launch_bucket(
        self,
        inputs: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        rotation_id: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
        active: cutlass.Constexpr,
        route: cutlass.Constexpr,
    ):
        if cutlass.const_expr(route[0] == "smem4"):
            tiles = Int32((self.n // 64) * self.splits)
            min_bmp = 3 if active >= 3 else 2
            self.multiply_smem64(
                inputs, payload, scales, partial, rows, rotation_id, active
            ).launch(
                grid=(min(tiles, cta_limit), 1, 1),
                block=(128, 1, 1),
                min_blocks_per_mp=min_bmp,
                stream=stream,
            )
            if cutlass.const_expr(self.splits > 1):
                self._launch_reduce(partial, output, rows, cta_limit, stream)
        elif cutlass.const_expr(route[0] == "opt"):
            fuse, b_sub = route[1], route[2]
            n_tiles = self.n // 16
            tiles = Int32(n_tiles) if cutlass.const_expr(fuse) else Int32(n_tiles * self.splits)
            warps = (self.splits if fuse else 1) * b_sub
            if cutlass.const_expr(active == 0):
                min_bmp = 2 if warps <= 4 else 1
                self.multiply_narrow(
                    inputs, payload, scales, partial, output, rows, rotation_id, fuse, b_sub
                ).launch(
                    grid=(min(tiles, cta_limit), 1, 1),
                    block=(32 * warps, 1, 1),
                    min_blocks_per_mp=min_bmp,
                    stream=stream,
                )
            else:
                blocks_per_sub = 8 // b_sub
                n_groups = self.splits if fuse else 1
                smem_slots = (n_groups * (8 - blocks_per_sub) if b_sub > 1 else 0) + (
                    (self.splits - 1) if (fuse and self.splits > 1) else 0
                )
                smem_bytes = smem_slots * active * 1024
                min_bmp = 3 if (warps <= 4 and active >= 3 and smem_bytes <= 28 * 1024) else (2 if warps <= 4 else 1)
                self.multiply_wide(
                    inputs, payload, scales, partial, output, rows, rotation_id, active, fuse, b_sub
                ).launch(
                    grid=(min(tiles, cta_limit), 1, 1),
                    block=(32 * warps, 1, 1),
                    min_blocks_per_mp=min_bmp,
                    stream=stream,
                )
            if cutlass.const_expr((not fuse) and self.splits > 1):
                self._launch_reduce(partial, output, rows, cta_limit, stream)

    @cute.jit
    def __call__(
        self,
        inputs: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        rotation_id: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.stock_narrow):
            if rows <= Int32(16):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    1, self.routes[1],
                )
            elif rows <= Int32(32):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    2, self.routes[2],
                )
            elif rows <= Int32(48):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    3, self.routes[3],
                )
            else:
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    4, self.routes[4],
                )
        else:
            if rows <= Int32(8):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    0, self.routes[0],
                )
            elif rows <= Int32(16):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    1, self.routes[1],
                )
            elif rows <= Int32(32):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    2, self.routes[2],
                )
            elif rows <= Int32(48):
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    3, self.routes[3],
                )
            else:
                self._launch_bucket(
                    inputs, payload, scales, partial, output, rows, rotation_id, cta_limit, stream,
                    4, self.routes[4],
                )
