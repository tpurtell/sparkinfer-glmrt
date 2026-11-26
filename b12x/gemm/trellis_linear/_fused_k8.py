"""Fused H128 ingress + K8 trellis decode + fused split-K/H128 egress kernels.

Eliminates standalone ``trellis_output`` launches, intermediate F32
``accumulator`` traffic, and ``input_block_scales`` setup/reads around
``NativeTrellisLinear`` K8 decode across all shared EXL3 families
(Hy4, Hy4-flash, GLM, GLM-flash, Kimi, Muse, ML4, Step-5, Inkling 2) for live
decode/verify rows ``1..=129`` (``TRELLIS_PARTIAL_ROWS``).

Execution pipeline inside one unified AOT program:
1. ``ingress_bf16`` (25 regs, zero ``input_block_scales``): vectorized
   ``BF16(H128(F32(x) * F32(suh)))`` written once into the tail of the caller's
   ``scratch`` buffer so every GEMM CTA reads L2-hot rotated BF16 with zero
   per-CTA H128 recomputation.
2. K8 decode multiply (zero ``input_block_scales`` loads):
   - Narrow/medium shapes (``(N/64)*splits <= 256`` at ``R <= 16``, and
     ``(N/64)*splits <= 72`` across all ``R = 1..129``): tiled N16-per-CTA K8
     decode with 4-way K128 parallelism (128 threads/CTA), keeping all SMs
     saturated even on narrow ``kv_a``/``rel`` projections with zero
     ``R=8->12`` route transitions.
   - Wide shapes: high-occupancy SMEM-staged N64/N128 K8 decode
     (M16/M32/[M48 for >=768 tiles]/M64), with split-ni N128 on vocabulary
     heads (``N > 65536``) to halve CTA count at M48/M64.
3. ``reduce_fused_egress`` (40 regs): ascending split-K F32 reduction fused
   directly with output ``BF16/F32(H128(acc) * F32(svh))`` in the same warp
   (or raw F32 when ``output_dtype == "raw_f32"`` for TP2 K-split side=2
   reduction prior to a single egress). Zero ``accumulator`` global write/read
   and zero standalone ``trellis_output`` launch.

Bitwise contract:
Every route ``1..=129`` preserves identical ``_mul`` + ``_w4a8_had128_quad`` +
``bf16_mma_m16n8k16_f32`` + ascending K128 ``fma_rn_f32(acc, 1.0, total)`` +
ascending split-K sum order, producing outputs 100% bit-identical across live
rows and bit-identical to the unfused 3-program chain.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    f16x2_to_f32x2,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ld_shared_u32,
    shared_ptr_to_u32,
    st_global_v2_f32,
    st_global_v2_u32,
    st_global_v4_f32,
    st_shared_v4_u32,
    u32_as_f32,
    weight_stream_policy,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from b12x.integration.cuteafd.pointwise import _mul
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


FUSED_DECODE_MAX_ROWS = 129


def fused_scratch_bytes(rows: int, k: int, n: int) -> int:
    """Single scratch extent: partial[rows, splits, N] F32 + rotated[rows, K] BF16."""
    splits = (k + 1023) // 1024
    # N % 128 == 0 => partial_bytes is a multiple of 512 B (16-byte aligned).
    return rows * splits * n * 4 + rows * k * 2


@dsl_user_op
def _pack_f32x2_to_bf16x2_rn(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Round two F32 values to packed BF16x2 (RNE, matching scalar .to(BFloat16))."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class FusedK8TrellisDecode:
    """Fused H128-ingress + K8 decode + split-K/H128-egress engine (rows 1..=129)."""

    def __init__(self, g: dict):
        self.k = int(g["input_features"])
        self.n = int(g["output_features"])
        self.bits = int(g["bits"])
        self.mul1 = g["codebook"] == "mul1"
        self.input_units = int(g["sharded_side"]) == 2
        self.max_rows = int(g["max_rows"])
        self.splits = int(g["splits"])
        self.output_dtype = str(g["output_dtype"])
        self.raw_f32 = self.output_dtype == "raw_f32"
        self.out_bf16 = self.output_dtype == "bf16"
        self.n_tiles_64 = self.n // 64
        self.n_tiles_128 = self.n // 128

        # Narrow/medium shapes ((N/64)*splits <= 256): tiled N16-per-CTA with
        # b_sub=4 (128 threads/CTA) saturates all SMs at R <= 16 with zero R=8->12 cliff.
        self.use_n16_low = (self.n_tiles_64 * self.splits) <= 256
        # Very narrow shapes ((N/64)*splits <= 72, e.g. glmf_kv_a, hy4_kv_a_padded,
        # mla_kv_a_padded, step5_kv_tp2, inkling_rel_tp2) keep tiled N16-per-CTA
        # across all R=1..129 to prevent SM underfill and eliminate R=16->24 cliffs.
        self.use_n16_all = (self.n_tiles_64 * self.splits) <= 72
        self.n16_b_sub = 4

        # N128-per-CTA (4 warps x N32) halves SMEM-A traffic for wide shapes at M16/M32
        # when N128*splits >= 384; M64 uses N64 (REG<=156) for body shapes and N128
        # split-ni for vocabulary heads (N > 65536) to halve CTA count.
        self.use_smem128_mid = (self.n_tiles_128 * self.splits) >= 384
        self.use_m48 = (self.n_tiles_64 * self.splits) >= 768
        self.is_vocab_head = self.n > 65536

    @property
    def exact_row_cap(self) -> int:
        return self.max_rows

    def _smem_f32_storage(self, num_f32: int):
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
    def ingress_bf16(
        self,
        x: cute.Tensor,
        suh: cute.Tensor,
        scratch: cute.Tensor,
        rows: Int32,
    ):
        """Vectorized H128(F32(x) * F32(suh)) -> BF16 into scratch tail (no block_scales)."""
        tid, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        grid_dim, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        task = Int32(block_idx) * Int32(4) + warp
        step = Int32(grid_dim) * Int32(4)

        k_blocks = Int32(self.k // 128)
        total_tasks = rows * k_blocks
        x_ptr = x.iterator.toint()
        suh_ptr = suh.iterator.toint()
        rot_ptr = scratch.iterator.toint() + rows.to(Int64) * Int64(self.splits * self.n * 4)

        while task < total_tasks:
            row = task // k_blocks
            kb = task % k_blocks
            col = kb * Int32(128) + lane * Int32(4)
            row_off = (row.to(Int64) * Int64(self.k) + col.to(Int64)) * Int64(2)
            x01, x23 = ld_global_nc_v2_u32(x_ptr + row_off)
            sh01, sh23 = ld_global_nc_v2_u32(suh_ptr + col.to(Int64) * Int64(2))
            s0, s1 = f16x2_to_f32x2(sh01)
            s2, s3 = f16x2_to_f32x2(sh23)
            v0 = _mul(_bf16_lo(x01), s0)
            v1 = _mul(_bf16_hi(x01), s1)
            v2 = _mul(_bf16_lo(x23), s2)
            v3 = _mul(_bf16_hi(x23), s3)
            h0, h1, h2, h3 = _w4a8_had128_quad(v0, v1, v2, v3, lane)
            st_global_v2_u32(
                rot_ptr + row_off,
                _pack_f32x2_to_bf16x2_rn(h0, h1),
                _pack_f32x2_to_bf16x2_rn(h2, h3),
            )
            task += step

    @cute.kernel
    def multiply_rot_n16_m16(
        self,
        payload: cute.Tensor,
        scratch: cute.Tensor,
        rows: Int32,
        b_sub: cutlass.Constexpr,
    ):
        """Tiled N16-per-CTA K8 decode (16-row groups) reading L2-hot rotated BF16 from scratch tail."""
        from ._native import (
            _decode_k8_words_shfl,
            _ld_l2_v2_off_pf,
            fma_rn_f32,
            ld_global_nc_u32_offset,
            ld_shared_v2_f32_offset,
            st_shared_v2_f32_offset,
        )

        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        sub = warp

        blocks_per_sub = 8 // b_sub
        phase_a_slots = 8 - blocks_per_sub if b_sub > 1 else 0
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._smem_f32_storage(phase_a_slots * 256))
        smem_lane = shared_ptr_to_u32(storage.smem.data_ptr()) + lane * Int32(8)

        rot_ptr = scratch.iterator.toint() + rows.to(Int64) * Int64(self.splits * self.n * 4)
        w_base = payload.iterator.toint() + lane.to(Int64) * Int64(8)
        prev_lane = (lane + Int32(31)) & Int32(31)

        n_tiles = self.n // 16
        base_tiles = Int32(n_tiles * self.splits)
        m_groups = (rows + Int32(15)) // Int32(16)
        total_tiles = m_groups * base_tiles
        lane_r = lane >> Int32(2)
        k_lane_off = ((lane & Int32(3)) * Int32(2)).to(Int64)

        while task < total_tiles:
            group = task // base_tiles
            tile = task % base_tiles
            policy = weight_stream_policy((group + Int32(1) == m_groups).to(Int32))

            row0 = group * Int32(16) + lane_r
            row1 = row0 + Int32(8)
            r0c = min(row0, rows - Int32(1))
            r1c = min(row1, rows - Int32(1))
            x0_row = rot_ptr + (r0c.to(Int64) * Int64(self.k) + k_lane_off) * Int64(2)
            x1_row = rot_ptr + (r1c.to(Int64) * Int64(self.k) + k_lane_off) * Int64(2)

            if cutlass.const_expr(self.input_units):
                n16_global = tile % Int32(n_tiles)
                split = tile // Int32(n_tiles)
            else:
                split = tile % Int32(self.splits)
                n16_global = tile // Int32(self.splits)

            nb = n16_global >> Int32(3)
            n16_sub = n16_global & Int32(7)

            total = cute.make_rmem_tensor((2, 4), Float32)
            total.fill(Float32(0.0))

            for b_local in cutlass.range(blocks_per_sub, unroll=1):
                block = Int32(sub * blocks_per_sub) + b_local
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    x0_kb = x0_row + kb.to(Int64) * Int64(256)
                    x1_kb = x1_row + kb.to(Int64) * Int64(256)
                    tile_kb = nb * Int32(self.k // 16 * 8) + kb * Int32(64) + n16_sub
                    if cutlass.const_expr(self.input_units):
                        tile_kb = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + n16_sub
                    w_kb = w_base + tile_kb.to(Int64) * Int64(256)

                    u0_buf = cute.make_rmem_tensor((8,), Uint32)
                    u1_buf = cute.make_rmem_tensor((8,), Uint32)
                    for k16 in cutlass.range_constexpr(8):
                        u0, u1 = _ld_l2_v2_off_pf(w_kb, k16 * 2048, policy)
                        u0_buf[k16], u1_buf[k16] = u0, u1

                    acc = cute.make_rmem_tensor((2, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        a0 = ld_global_nc_u32_offset(x0_kb, k16 * 32)
                        a2 = ld_global_nc_u32_offset(x0_kb, k16 * 32 + 16)
                        a1 = ld_global_nc_u32_offset(x1_kb, k16 * 32)
                        a3 = ld_global_nc_u32_offset(x1_kb, k16 * 32 + 16)
                        b0, b1, b2, b3 = _decode_k8_words_shfl(
                            u0_buf[k16], u1_buf[k16], prev_lane, False, self.mul1, True
                        )
                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                            acc[0, 0], acc[0, 1], acc[0, 2], acc[0, 3],
                            a0, a1, a2, a3, b0, b1,
                        )
                        acc[0, 0], acc[0, 1], acc[0, 2], acc[0, 3] = d0, d1, d2, d3
                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                            acc[1, 0], acc[1, 1], acc[1, 2], acc[1, 3],
                            a0, a1, a2, a3, b2, b3,
                        )
                        acc[1, 0], acc[1, 1], acc[1, 2], acc[1, 3] = d0, d1, d2, d3

                    if sub == Int32(0):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                total[n, v] = fma_rn_f32(acc[n, v], Float32(1.0), total[n, v])
                    else:
                        b_rem_dyn = (sub - Int32(1)) * Int32(blocks_per_sub) + b_local
                        slot_addr = smem_lane + b_rem_dyn * Int32(1024)
                        for n in cutlass.range_constexpr(2):
                            st_shared_v2_f32_offset(slot_addr, n * 256, acc[n, 0], acc[n, 1])
                            st_shared_v2_f32_offset(slot_addr, n * 256 + 512, acc[n, 2], acc[n, 3])

            if cutlass.const_expr(b_sub > 1):
                cute.arch.sync_threads()
                if sub == Int32(0):
                    for b_rem in cutlass.range_constexpr(8 - blocks_per_sub):
                        kb_rem = split * Int32(8) + Int32(blocks_per_sub + b_rem)
                        if cutlass.const_expr(self.k % 1024 == 0):
                            for n in cutlass.range_constexpr(2):
                                t0, t1 = ld_shared_v2_f32_offset(smem_lane, b_rem * 1024 + n * 256)
                                t2, t3 = ld_shared_v2_f32_offset(smem_lane, b_rem * 1024 + n * 256 + 512)
                                total[n, 0] = fma_rn_f32(t0, Float32(1.0), total[n, 0])
                                total[n, 1] = fma_rn_f32(t1, Float32(1.0), total[n, 1])
                                total[n, 2] = fma_rn_f32(t2, Float32(1.0), total[n, 2])
                                total[n, 3] = fma_rn_f32(t3, Float32(1.0), total[n, 3])
                        else:
                            if kb_rem < Int32(self.k // 128):
                                for n in cutlass.range_constexpr(2):
                                    t0, t1 = ld_shared_v2_f32_offset(smem_lane, b_rem * 1024 + n * 256)
                                    t2, t3 = ld_shared_v2_f32_offset(smem_lane, b_rem * 1024 + n * 256 + 512)
                                    total[n, 0] = fma_rn_f32(t0, Float32(1.0), total[n, 0])
                                    total[n, 1] = fma_rn_f32(t1, Float32(1.0), total[n, 1])
                                    total[n, 2] = fma_rn_f32(t2, Float32(1.0), total[n, 2])
                                    total[n, 3] = fma_rn_f32(t3, Float32(1.0), total[n, 3])

            if sub == Int32(0):
                col_base = n16_global * Int32(16) + (lane & Int32(3)) * Int32(2)
                p_off = (
                    (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n)
                    + col_base.to(Int64)
                ) * Int64(4)
                p_addr = scratch.iterator.toint() + p_off
                if row0 < rows:
                    for n in cutlass.range_constexpr(2):
                        st_global_v2_f32(p_addr + Int64(n * 32), total[n, 0], total[n, 1])
                if row1 < rows:
                    p_addr1 = p_addr + Int64(8 * self.splits * self.n * 4)
                    for n in cutlass.range_constexpr(2):
                        st_global_v2_f32(p_addr1 + Int64(n * 32), total[n, 2], total[n, 3])
            if task + step < total_tiles:
                cute.arch.sync_threads()
            task += step

    @cute.kernel
    def multiply_rot_smem(
        self,
        payload: cute.Tensor,
        scratch: cute.Tensor,
        rows: Int32,
        active: cutlass.Constexpr,
        n16_per_warp: cutlass.Constexpr,
    ):
        """High-occupancy SMEM-staged K8 decode reading L2-hot rotated BF16 from scratch tail."""
        from ._native import (
            _decode_k8_words_shfl,
            _ld_l2_v2_off_pf,
            fma_rn_f32,
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
        guarantee_iters = 2 * (active - 1) if active < 4 else 0
        n_cta = 64 * n16_per_warp
        n_tiles = self.n // n_cta
        base_tiles = Int32(n_tiles * self.splits)
        total_tiles = base_tiles if cutlass.const_expr(active < 4) else ((rows + Int32(M - 1)) // Int32(M)) * base_tiles

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._a_storage(M))
        sx = shared_ptr_to_u32(storage.x.data_ptr())
        sx_store = sx + (c * Int32(M_PAD) + r_grp) * Int32(16)
        sx_read = sx + lane * Int32(4)

        rot_ptr = scratch.iterator.toint() + rows.to(Int64) * Int64(self.splits * self.n * 4)
        w_base = payload.iterator.toint() + lane.to(Int64) * Int64(8)
        prev_lane = (lane + Int32(31)) & Int32(31)
        lane_r = lane >> Int32(2)

        prefetch_k16 = 8 if active <= 2 else 4

        while task < total_tiles:
            group = Int32(0) if cutlass.const_expr(active < 4) else task // base_tiles
            tile = task if cutlass.const_expr(active < 4) else task % base_tiles
            last_reader = Int32(1)
            if cutlass.const_expr(active == 4):
                last_reader = (group + Int32(1) == (rows + Int32(M - 1)) // Int32(M)).to(Int32)
            policy = weight_stream_policy(last_reader)

            if cutlass.const_expr(self.input_units):
                tile_n = tile % Int32(n_tiles)
                split = tile // Int32(n_tiles)
            else:
                split = tile % Int32(self.splits)
                tile_n = tile // Int32(self.splits)

            n16_base = tile_n * Int32(4 * n16_per_warp) + warp * Int32(n16_per_warp)
            nb = n16_base >> Int32(3)
            n16_sub = n16_base & Int32(7)
            row_group_base = Int32(0) if cutlass.const_expr(active < 4) else group * Int32(M)

            total = cute.make_rmem_tensor((active, 2 * n16_per_warp, 4), Float32)
            total.fill(Float32(0.0))

            for block in cutlass.range(8, unroll=1):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    tile_kb = nb * Int32(self.k // 16 * 8) + kb * Int32(64) + n16_sub
                    if cutlass.const_expr(self.input_units):
                        tile_kb = kb * Int32(self.n // 16 * 8) + nb * Int32(64) + n16_sub
                    w_kb = w_base + tile_kb.to(Int64) * Int64(256)

                    k_col = kb * Int32(128) + c * Int32(8)
                    if cutlass.const_expr(n16_per_warp == 1):
                        u0_buf = cute.make_rmem_tensor((prefetch_k16,), Uint32)
                        u1_buf = cute.make_rmem_tensor((prefetch_k16,), Uint32)
                        for ki in cutlass.range_constexpr(prefetch_k16):
                            u0, u1 = _ld_l2_v2_off_pf(w_kb, ki * 2048, policy)
                            u0_buf[ki], u1_buf[ki] = u0, u1
                        for s in cutlass.range_constexpr(load_iters):
                            row = row_group_base + r_grp + Int32(s * 8)
                            v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                            if cutlass.const_expr(s < guarantee_iters):
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    rot_ptr + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2)
                                )
                            else:
                                if row < rows:
                                    v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                        rot_ptr + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2)
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
                                    u0_buf[ki], u1_buf[ki], prev_lane, False, self.mul1, True
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
                        for mb in cutlass.range_constexpr(active):
                            for n in cutlass.range_constexpr(2):
                                for v in cutlass.range_constexpr(4):
                                    total[mb, n, v] = fma_rn_f32(acc[mb, n, v], Float32(1.0), total[mb, n, v])
                    else:
                        for s in cutlass.range_constexpr(load_iters):
                            row = row_group_base + r_grp + Int32(s * 8)
                            v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                            if cutlass.const_expr(s < guarantee_iters):
                                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                    rot_ptr + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2)
                                )
                            else:
                                if row < rows:
                                    v0, v1, v2, v3 = ld_global_nc_v4_u32(
                                        rot_ptr + (row.to(Int64) * Int64(self.k) + k_col.to(Int64)) * Int64(2)
                                    )
                            st_shared_v4_u32(sx_store + Int32(s * 8 * 16), v0, v1, v2, v3)
                        cute.arch.sync_threads()

                        if cutlass.const_expr(active <= 2):
                            acc = cute.make_rmem_tensor((active, 4, 4), Float32)
                            acc.fill(Float32(0.0))
                            u0_buf = cute.make_rmem_tensor((8, 2), Uint32)
                            u1_buf = cute.make_rmem_tensor((8, 2), Uint32)
                            for k16 in cutlass.range_constexpr(8):
                                for ni in cutlass.range_constexpr(2):
                                    u0, u1 = _ld_l2_v2_off_pf(w_kb, (k16 * 8 + ni) * 256, policy)
                                    u0_buf[k16, ni], u1_buf[k16, ni] = u0, u1
                            for k16 in cutlass.range_constexpr(8):
                                a = cute.make_rmem_tensor((active, 4), Uint32)
                                for mb in cutlass.range_constexpr(active):
                                    a[mb, 0] = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16) * 16))
                                    a[mb, 1] = ld_shared_u32(sx_read + Int32(((2 * k16) * M_PAD + mb * 16 + 8) * 16))
                                    a[mb, 2] = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16) * 16))
                                    a[mb, 3] = ld_shared_u32(sx_read + Int32(((2 * k16 + 1) * M_PAD + mb * 16 + 8) * 16))
                                for ni in cutlass.range_constexpr(2):
                                    b0, b1, b2, b3 = _decode_k8_words_shfl(
                                        u0_buf[k16, ni], u1_buf[k16, ni], prev_lane, False, self.mul1, True
                                    )
                                    for mb in cutlass.range_constexpr(active):
                                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                            acc[mb, 2 * ni, 0], acc[mb, 2 * ni, 1],
                                            acc[mb, 2 * ni, 2], acc[mb, 2 * ni, 3],
                                            a[mb, 0], a[mb, 1], a[mb, 2], a[mb, 3], b0, b1,
                                        )
                                        acc[mb, 2 * ni, 0], acc[mb, 2 * ni, 1] = d0, d1
                                        acc[mb, 2 * ni, 2], acc[mb, 2 * ni, 3] = d2, d3
                                        d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                            acc[mb, 2 * ni + 1, 0], acc[mb, 2 * ni + 1, 1],
                                            acc[mb, 2 * ni + 1, 2], acc[mb, 2 * ni + 1, 3],
                                            a[mb, 0], a[mb, 1], a[mb, 2], a[mb, 3], b2, b3,
                                        )
                                        acc[mb, 2 * ni + 1, 0], acc[mb, 2 * ni + 1, 1] = d0, d1
                                        acc[mb, 2 * ni + 1, 2], acc[mb, 2 * ni + 1, 3] = d2, d3
                            for mb in cutlass.range_constexpr(active):
                                for n in cutlass.range_constexpr(4):
                                    for v in cutlass.range_constexpr(4):
                                        total[mb, n, v] = fma_rn_f32(acc[mb, n, v], Float32(1.0), total[mb, n, v])
                        else:
                            for ni in cutlass.range_constexpr(2):
                                acc = cute.make_rmem_tensor((active, 2, 4), Float32)
                                acc.fill(Float32(0.0))
                                for ko in cutlass.range(2, unroll=1):
                                    sx_k = sx_read + ko * Int32(8 * M_PAD * 16)
                                    w_ko = w_kb + (ko * Int32(32) + Int32(ni)).to(Int64) * Int64(256)
                                    u0_buf = cute.make_rmem_tensor((4,), Uint32)
                                    u1_buf = cute.make_rmem_tensor((4,), Uint32)
                                    for ki in cutlass.range_constexpr(4):
                                        u0, u1 = _ld_l2_v2_off_pf(w_ko, ki * 2048, policy)
                                        u0_buf[ki], u1_buf[ki] = u0, u1
                                    for ki in cutlass.range_constexpr(4):
                                        b0, b1, b2, b3 = _decode_k8_words_shfl(
                                            u0_buf[ki], u1_buf[ki], prev_lane, False, self.mul1, True
                                        )
                                        for mb in cutlass.range_constexpr(active):
                                            a0 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16) * 16))
                                            a1 = ld_shared_u32(sx_k + Int32(((2 * ki) * M_PAD + mb * 16 + 8) * 16))
                                            a2 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16) * 16))
                                            a3 = ld_shared_u32(sx_k + Int32(((2 * ki + 1) * M_PAD + mb * 16 + 8) * 16))
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
                                for mb in cutlass.range_constexpr(active):
                                    for v in cutlass.range_constexpr(4):
                                        total[mb, 2 * ni, v] = fma_rn_f32(acc[mb, 0, v], Float32(1.0), total[mb, 2 * ni, v])
                                        total[mb, 2 * ni + 1, v] = fma_rn_f32(acc[mb, 1, v], Float32(1.0), total[mb, 2 * ni + 1, v])
                    cute.arch.sync_threads()

            col_base = n16_base * Int32(16) + (lane & Int32(3)) * Int32(2)
            p_base = scratch.iterator.toint()
            for mb in cutlass.range_constexpr(active):
                row0 = row_group_base + lane_r + Int32(mb * 16)
                row1 = row0 + Int32(8)
                p_off = (
                    (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n)
                    + col_base.to(Int64)
                ) * Int64(4)
                p_addr = p_base + p_off
                if row0 < rows:
                    for n in cutlass.range_constexpr(2 * n16_per_warp):
                        st_global_v2_f32(p_addr + Int64(n * 32), total[mb, n, 0], total[mb, n, 1])
                if row1 < rows:
                    p_addr1 = p_addr + Int64(8 * self.splits * self.n * 4)
                    for n in cutlass.range_constexpr(2 * n16_per_warp):
                        st_global_v2_f32(p_addr1 + Int64(n * 32), total[mb, n, 2], total[mb, n, 3])
            task += step

    @cute.kernel
    def reduce_fused_egress(
        self,
        scratch: cute.Tensor,
        svh: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
    ):
        """Ascending split-K F32 sum fused with output H128 * svh -> BF16/F32 (or raw F32)."""
        tid, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        grid_dim, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        task = Int32(block_idx) * Int32(4) + warp
        step = Int32(grid_dim) * Int32(4)

        n_blocks = Int32(self.n // 128)
        total_tasks = rows * n_blocks
        p_ptr = scratch.iterator.toint()
        svh_ptr = svh.iterator.toint()
        o_ptr = output.iterator.toint()

        while task < total_tasks:
            row = task // n_blocks
            nb = task % n_blocks
            col = nb * Int32(128) + lane * Int32(4)
            t0, t1, t2, t3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            row_base = p_ptr + (row.to(Int64) * Int64(self.splits * self.n) + col.to(Int64)) * Int64(4)
            for split in cutlass.range_constexpr(self.splits):
                v0, v1, v2, v3 = ld_global_nc_v4_u32(row_base + Int64(split * self.n * 4))
                t0 = t0 + u32_as_f32(v0)
                t1 = t1 + u32_as_f32(v1)
                t2 = t2 + u32_as_f32(v2)
                t3 = t3 + u32_as_f32(v3)
            if cutlass.const_expr(self.raw_f32):
                out_addr = o_ptr + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
                st_global_v4_f32(out_addr, t0, t1, t2, t3)
            else:
                h0, h1, h2, h3 = _w4a8_had128_quad(t0, t1, t2, t3, lane)
                sv01, sv23 = ld_global_nc_v2_u32(svh_ptr + col.to(Int64) * Int64(2))
                s0, s1 = f16x2_to_f32x2(sv01)
                s2, s3 = f16x2_to_f32x2(sv23)
                h0, h1, h2, h3 = h0 * s0, h1 * s1, h2 * s2, h3 * s3
                if cutlass.const_expr(self.out_bf16):
                    out_addr = o_ptr + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(2)
                    st_global_v2_u32(
                        out_addr,
                        _pack_f32x2_to_bf16x2_rn(h0, h1),
                        _pack_f32x2_to_bf16x2_rn(h2, h3),
                    )
                else:
                    out_addr = o_ptr + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
                    st_global_v4_f32(out_addr, h0, h1, h2, h3)
            task += step

    @cute.jit
    def _launch_ingress(
        self,
        x: cute.Tensor,
        suh: cute.Tensor,
        scratch: cute.Tensor,
        rows: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
    ):
        total_tasks = rows * Int32(self.k // 128)
        self.ingress_bf16(x, suh, scratch, rows).launch(
            grid=(min((total_tasks + Int32(3)) // Int32(4), cta_limit), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _launch_rot_smem(
        self,
        payload: cute.Tensor,
        scratch: cute.Tensor,
        rows: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
        active: cutlass.Constexpr,
        n16_per_warp: cutlass.Constexpr,
    ):
        n_tiles = self.n // (64 * n16_per_warp)
        base_tiles = Int32(n_tiles * self.splits)
        M = 16 * active
        tiles = base_tiles if cutlass.const_expr(active < 4) else ((rows + Int32(M - 1)) // Int32(M)) * base_tiles
        self.multiply_rot_smem(
            payload, scratch, rows, active, n16_per_warp
        ).launch(
            grid=(min(tiles, cta_limit), 1, 1),
            block=(128, 1, 1),
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _launch_reduce(
        self,
        scratch: cute.Tensor,
        svh: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
    ):
        total_tasks = rows * Int32(self.n // 128)
        self.reduce_fused_egress(scratch, svh, output, rows).launch(
            grid=(min((total_tasks + Int32(3)) // Int32(4), cta_limit), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        payload: cute.Tensor,
        suh: cute.Tensor,
        svh: cute.Tensor,
        scratch: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        cta_limit: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
            self._launch_ingress(x, suh, scratch, rows, cta_limit, stream)
            n16_mid = 2 if self.use_smem128_mid else 1
            if cutlass.const_expr(self.use_n16_all):
                tiles = ((rows + Int32(15)) // Int32(16)) * Int32((self.n // 16) * self.splits)
                self.multiply_rot_n16_m16(payload, scratch, rows, self.n16_b_sub).launch(
                    grid=(min(tiles, cta_limit), 1, 1),
                    block=(32 * self.n16_b_sub, 1, 1),
                    min_blocks_per_mp=2,
                    stream=stream,
                )
            elif rows <= Int32(16):
                if cutlass.const_expr(self.use_n16_low):
                    tiles = Int32((self.n // 16) * self.splits)
                    self.multiply_rot_n16_m16(payload, scratch, rows, self.n16_b_sub).launch(
                        grid=(min(tiles, cta_limit), 1, 1),
                        block=(32 * self.n16_b_sub, 1, 1),
                        min_blocks_per_mp=2,
                        stream=stream,
                    )
                else:
                    self._launch_rot_smem(payload, scratch, rows, cta_limit, stream, 1, n16_mid)
            elif rows <= Int32(32):
                self._launch_rot_smem(payload, scratch, rows, cta_limit, stream, 2, n16_mid)
            elif rows <= Int32(48):
                n16_hi = 2 if self.is_vocab_head else 1
                if cutlass.const_expr(self.use_m48):
                    self._launch_rot_smem(payload, scratch, rows, cta_limit, stream, 3, n16_hi)
                else:
                    self._launch_rot_smem(payload, scratch, rows, cta_limit, stream, 4, 1)
            else:
                n16_hi = 2 if self.is_vocab_head else 1
                self._launch_rot_smem(payload, scratch, rows, cta_limit, stream, 4, n16_hi)
            self._launch_reduce(scratch, svh, output, rows, cta_limit, stream)
