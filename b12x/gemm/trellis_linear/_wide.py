"""Shared-ABI wide trellis GEMM with cooperative, exact F16 decode reuse.

A 64x64 CTA decodes each K128 slab once. Shared memory holds the native MMA
B-register layout, so the four consumer warps need no transpose or re-rounding.
The K128 and split reduction order is identical to the register-only core.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import f16_mma_m16n8k16_f32, ld_global_nc_u32
from b12x.moe._shared.kernels.w4a16.exlr8 import _side_value
from ._native import NativeTrellisLinear, decode_tile


class WideTrellisLinear(NativeTrellisLinear):
    @cute.jit
    def __call__(self, inputs: cute.Tensor, payload: cute.Tensor,
                 scales: cute.Tensor, partial: cute.Tensor, output: cute.Tensor,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations) and cta_limit > Int32(0):
            self.multiply_wide(inputs, payload, scales, partial, rows, rotation_id).launch(
                grid=(min(((rows + Int32(63)) // Int32(64)) * Int32(self.n // 64 * self.splits), cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=2, stream=stream)
            self.reduce(partial, output, rows).launch(
                grid=(min(rows * Int32(self.n // 128), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)

    @cute.kernel
    def multiply_wide(self, inputs: cute.Tensor, payload: cute.Tensor,
                      scales: cute.Tensor, partial: cute.Tensor,
                      rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        allocator = cutlass.utils.SmemAllocator()
        weights = allocator.allocate_tensor(element_type=Uint32,
            layout=cute.make_layout((32, 4, 4, 8), stride=(1, 32, 128, 512)), byte_alignment=128)
        x = inputs[rotation_id]
        while task < ((rows + Int32(63)) // Int32(64)) * Int32(self.n // 64 * self.splits):
            split = task % Int32(self.splits)
            nb = (task // Int32(self.splits)) % Int32(self.n // 64)
            group = task // Int32(self.splits * (self.n // 64))
            row0 = group * Int32(64) + warp * Int32(16) + lane // Int32(4)
            row1 = row0 + Int32(8)
            total = cute.make_rmem_tensor((8, 4), Float32)
            total.fill(Float32(0.0))
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.k // 128):
                    for k16 in cutlass.range(8):
                        tile = (nb // Int32(2)) * Int32(self.k // 16 * 8) + (kb * Int32(8) + k16) * Int32(8) + (nb % Int32(2)) * Int32(4) + warp
                        if cutlass.const_expr(self.input_units):
                            tile = kb * Int32(self.n // 16 * 8) + (nb // Int32(2)) * Int32(64) + k16 * Int32(8) + (nb % Int32(2)) * Int32(4) + warp
                        b0, b1, b2, b3 = decode_tile(payload.iterator.toint(), tile, lane, self.bits, self.e4m3, self.mul1)
                        weights[lane, 0, warp, k16] = b0
                        weights[lane, 1, warp, k16] = b1
                        weights[lane, 2, warp, k16] = b2
                        weights[lane, 3, warp, k16] = b3
                    cute.arch.sync_threads()
                    acc = cute.make_rmem_tensor((8, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range(8):
                        k = kb * Int32(128) + k16 * Int32(16) + (lane & Int32(3)) * Int32(2)
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        if row0 < rows:
                            addr = x + (row0.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a0, a2 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        if row1 < rows:
                            addr = x + (row1.to(Int64) * Int64(self.k) + k.to(Int64)) * Int64(2)
                            a1, a3 = ld_global_nc_u32(addr), ld_global_nc_u32(addr + Int64(16))
                        for n16 in cutlass.range_constexpr(4):
                            b0, b1 = weights[lane, 0, n16, k16], weights[lane, 1, n16, k16]
                            b2, b3 = weights[lane, 2, n16, k16], weights[lane, 3, n16, k16]
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3] = d0, d1, d2, d3
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    for n in cutlass.range_constexpr(8):
                        for v in cutlass.range_constexpr(4):
                            total[n, v] = total[n, v] + acc[n, v] * scale
                    cute.arch.sync_threads()
            for n in cutlass.range_constexpr(8):
                col = nb * Int32(64) + Int32(n*8) + (lane & Int32(3)) * Int32(2)
                base = (row0.to(Int64) * Int64(self.splits) + split.to(Int64)) * Int64(self.n) + col.to(Int64)
                if row0 < rows:
                    partial[base], partial[base + Int64(1)] = total[n, 0], total[n, 1]
                if row1 < rows:
                    base += Int64(8 * self.splits * self.n)
                    partial[base], partial[base + Int64(1)] = total[n, 2], total[n, 3]
            task += step
