"""Experimental K4/E4M3 dense prefill with shared MXFP8 activation records.

The 256x128 CTA and four-stage K64 ring follow kindlingai/gr8 (MIT,
85da0d3), engine/kernels/bind_prefill.cuh k_gemm_tr. The CuTe implementation
uses the fork's trellis decoder and scale-aware F32 output contract. Records
follow the T1 layout: 512 fragment bytes and 16 per-row E8M0 scale bytes.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    broadcast_f32_to_half2, cp_async4_shared_global_pred,
    habs2, half2_mul, hmax2, hmax_to_f32, ld_global_nc_v2_u32,
    ld_shared_u32, ld_shared_u8, ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16, mxfp8_mma_m16n8k32_f32_e4m3,
    packed_dequant_trellis_to_half2x4, pow2_ceil_ue8m0,
    shared_ptr_to_u32, st_global_u8, st_global_v2_f32, st_global_v4_u32,
    st_shared_u16, ue8m0_to_output_scale,
)
from b12x.gemm.native_tile_linear import _fma
from b12x.moe._shared.kernels.w4a16.exlr8 import _side_value

RECORD_BYTES = 528


@dsl_user_op
def _pack_half2(lo: Uint32, hi: Uint32, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 a, b; cvt.rn.satfinite.e4m3x2.f16x2 a, $1; "
        "cvt.rn.satfinite.e4m3x2.f16x2 b, $2; mov.b32 $0, {a, b}; }",
        "=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


class WideTrellisMxfp8:
    """Two launches, fixed geometry, no materialized weight copy or live-row key."""

    def __init__(self, geometry, *, row_tiles=16, stages=4):
        self.k, self.n = geometry["input_features"], geometry["output_features"]
        if geometry["bits"] != 4 or geometry["codebook"] != "e4m3":
            raise ValueError("wide MXFP8 prefill requires K4/E4M3 trellis")
        if self.k % 128 or self.n % 128 or row_tiles != 16 or stages not in (3, 4):
            raise ValueError("wide prefill requires 128-aligned geometry, 16 row tiles, 3/4 stages")
        self.side2 = geometry["sharded_side"] == 2
        self.max_rows, self.rotations = geometry["max_rows"], geometry["rotation_count"]
        self.rb, self.stages = row_tiles, stages
        self.warps = row_tiles
        self.a_bytes = 2 * row_tiles * RECORD_BYTES
        self.raw_bytes = 4096
        self.stage_bytes = self.a_bytes + self.raw_bytes
        self.dec_bytes = 8192
        self.quant_warps = 8

    @property
    def scratch_bytes(self):
        return ((self.max_rows + 15) // 16) * (self.k // 32) * RECORD_BYTES

    @cute.kernel
    def quantize(self, inputs: cute.Tensor, records: cute.Tensor,
                 rows: Int32, rotation: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        task = Int32(task) * Int32(8) + warp
        count = ((rows + Int32(15)) // Int32(16)) * Int32(self.k // 32)
        while task < count:
            rb, kb = task // Int32(self.k // 32), task % Int32(self.k // 32)
            r = rb * Int32(16) + (lane >> Int32(2))
            c = kb * Int32(32) + (lane & Int32(3)) * Int32(4)
            x0 = inputs[rotation] + min(r, rows - Int32(1)).to(Int64) * Int64(2 * self.k)
            x1 = inputs[rotation] + min(r + Int32(8), rows - Int32(1)).to(Int64) * Int64(2 * self.k)
            f00, f01 = ld_global_nc_v2_u32(x0 + c.to(Int64) * Int64(2))
            f10, f11 = ld_global_nc_v2_u32(x1 + c.to(Int64) * Int64(2))
            f20, f21 = ld_global_nc_v2_u32(x0 + c.to(Int64) * Int64(2) + Int64(32))
            f30, f31 = ld_global_nc_v2_u32(x1 + c.to(Int64) * Int64(2) + Int64(32))
            m0 = hmax_to_f32(hmax2(hmax2(habs2(f00), habs2(f01)), hmax2(habs2(f20), habs2(f21))))
            m1 = hmax_to_f32(hmax2(hmax2(habs2(f10), habs2(f11)), hmax2(habs2(f30), habs2(f31))))
            m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=1))
            m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=1))
            m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=2))
            m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=2))
            _, code0 = pow2_ceil_ue8m0(m0 * Float32(1.0 / 448.0))
            _, code1 = pow2_ceil_ue8m0(m1 * Float32(1.0 / 448.0))
            h0 = broadcast_f32_to_half2(ue8m0_to_output_scale(code0))
            h1 = broadcast_f32_to_half2(ue8m0_to_output_scale(code1))
            w0 = _pack_half2(half2_mul(f00, h0), half2_mul(f01, h0))
            w1 = _pack_half2(half2_mul(f10, h1), half2_mul(f11, h1))
            w2 = _pack_half2(half2_mul(f20, h0), half2_mul(f21, h0))
            w3 = _pack_half2(half2_mul(f30, h1), half2_mul(f31, h1))
            rec = records.iterator.toint() + task.to(Int64) * Int64(RECORD_BYTES)
            st_global_v4_u32(rec + lane.to(Int64) * Int64(16), w0, w1, w2, w3)
            if (lane & Int32(3)) == Int32(0):
                st_global_u8(rec + Int64(512) + (lane >> Int32(2)).to(Int64), Uint8(code0))
                st_global_u8(rec + Int64(520) + (lane >> Int32(2)).to(Int64), Uint8(code1))
            task += Int32(grid) * Int32(8)

    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "buf": cute.struct.Align[cute.struct.MemRange[Uint8,
                self.stages * self.stage_bytes + 2 * self.dec_bytes], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def _issue(self, base: Int32, payload: cute.Tensor, records: cute.Tensor,
               nblock: Int32, rb0: Int32, stage: Int32, rows: Int32, tid: Int32):
        target = base + (stage % Int32(self.stages)) * Int32(self.stage_bytes)
        for i in cutlass.range_constexpr((2 * self.rb * 33 + self.warps * 32 - 1) // (self.warps * 32)):
            u = tid + Int32(i * self.warps * 32)
            if u < Int32(2 * self.rb * 33):
                k32 = u // Int32(self.rb * 33)
                rb = (u // Int32(33)) % Int32(self.rb)
                chunk = u % Int32(33)
                rec = ((rb0 + rb).to(Int64) * Int64(self.k // 32)
                       + (stage * Int32(2) + k32).to(Int64)) * Int64(RECORD_BYTES)
                pred = Int32(0)
                if (rb0 + rb) * Int32(16) < rows:
                    pred = Int32(1)
                cp_async4_shared_global_pred(target + u * Int32(16),
                    records.iterator.toint() + rec + chunk.to(Int64) * Int64(16), pred)
        for i in cutlass.range_constexpr(256 // (self.warps * 32) + 1):
            u = tid + Int32(i * self.warps * 32)
            if u < Int32(256):
                tile, chunk = u >> Int32(3), u & Int32(7)
                k16 = stage * Int32(4) + (tile >> Int32(3))
                slot = tile & Int32(7)
                if cutlass.const_expr(self.side2):
                    index = (k16 // Int32(8)) * Int32(self.n // 16 * 8) + nblock * Int32(64) + (k16 % Int32(8)) * Int32(8) + slot
                else:
                    index = nblock * Int32(self.k // 16 * 8) + k16 * Int32(8) + slot
                cp_async4_shared_global_pred(target + Int32(self.a_bytes) + u * Int32(16),
                    payload.iterator.toint() + index.to(Int64) * Int64(128) + chunk.to(Int64) * Int64(16), Int32(1))

    @cute.jit
    def _decode(self, base: Int32, stage: Int32, lane: Int32, warp: Int32):
        raw = base + (stage % Int32(self.stages)) * Int32(self.stage_bytes) + Int32(self.a_bytes)
        dst = base + Int32(self.stages * self.stage_bytes) + (stage & Int32(1)) * Int32(self.dec_bytes)
        for i in cutlass.range_constexpr(32 // self.warps):
            tile = warp + Int32(i * self.warps)
            w = ld_shared_u32(raw + tile * Int32(128) + lane * Int32(4))
            prev = cute.arch.shuffle_sync(w, (lane + Int32(31)) & Int32(31))
            b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(w, (prev << Uint32(16)) | (w >> Uint32(16)), 4)
            n = (tile & Int32(7)) * Int32(16) + (lane >> Int32(2))
            k16 = tile >> Int32(3)
            kpair = (lane & Int32(3)) * Int32(2)
            e0, e1 = _pack_half2(b0, b1), _pack_half2(b2, b3)
            off = dst + n * Int32(64) + ((k16 ^ ((n >> Int32(1)) & Int32(3))) << Int32(4)) + kpair
            st_shared_u16(off, e0)
            st_shared_u16(off + Int32(8), e0 >> Uint32(16))
            st_shared_u16(off + Int32(512), e1)
            st_shared_u16(off + Int32(520), e1 >> Uint32(16))

    @cute.kernel
    def gemm(self, payload: cute.Tensor, scales: cute.Tensor, records: cute.Tensor,
             output: cute.Tensor, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        wm, wn = warp // Int32(4), warp % Int32(4)
        smem = utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        base = shared_ptr_to_u32(storage.buf.data_ptr())
        mtiles = (rows + Int32(self.rb * 16 - 1)) // Int32(self.rb * 16)
        count = mtiles * Int32(self.n // 128)
        stages = Int32(self.k // 64)
        while task < count:
            rb0 = (Int32(task) % mtiles) * Int32(self.rb)
            nb = Int32(task) // mtiles
            acc = cute.make_rmem_tensor((4, 4, 4), Float32)
            acc.fill(Float32(0))
            for p in cutlass.range_constexpr(self.stages - 1):
                if Int32(p) < stages:
                    self._issue(base, payload, records, nb, rb0, Int32(p), rows, tid)
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            self._decode(base, Int32(0), lane, warp)
            for stage in cutlass.range(self.k // 64, unroll=1):
                cute.arch.cp_async_wait_group(self.stages - 3)
                cute.arch.sync_threads()
                if stage + Int32(self.stages - 1) < stages:
                    self._issue(base, payload, records, nb, rb0, stage + Int32(self.stages - 1), rows, tid)
                cute.arch.cp_async_commit_group()
                if stage + Int32(1) < stages:
                    self._decode(base, stage + Int32(1), lane, warp)
                a = base + (stage % Int32(self.stages)) * Int32(self.stage_bytes)
                b = base + Int32(self.stages * self.stage_bytes) + (stage & Int32(1)) * Int32(self.dec_bytes)
                scale = _side_value(scales.iterator.toint(), stage // Int32(2))
                for k32 in cutlass.range_constexpr(2):
                    bf = cute.make_rmem_tensor((8,), Uint32)
                    for pair in cutlass.range_constexpr(2):
                        n = wn * Int32(32) + Int32(pair * 16) + (lane >> Int32(4)) * Int32(8) + (lane & Int32(7))
                        chunk = Int32(2 * k32) + ((lane >> Int32(3)) & Int32(1))
                        v = ldmatrix_m8n8x4_b16(b + n * Int32(64) + ((chunk ^ ((n >> Int32(1)) & Int32(3))) << Int32(4)))
                        for j in cutlass.range_constexpr(4):
                            bf[pair * 4 + j] = v[j]
                    for m in cutlass.range_constexpr(4):
                        rec = a + (Int32(k32 * self.rb) + wm * Int32(4) + Int32(m)) * Int32(RECORD_BYTES)
                        a0, a1, a2, a3 = ld_shared_v4_u32(rec + lane * Int32(16))
                        sfa = Uint32(ld_shared_u8(rec + Int32(512) + (lane >> Int32(2)) + (lane & Int32(1)) * Int32(8)))
                        for n in cutlass.range_constexpr(4):
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                Float32(0), Float32(0), Float32(0), Float32(0),
                                a0, a1, a2, a3, bf[n * 2], bf[n * 2 + 1], sfa, Uint32(127))
                            acc[m, n, 0] = _fma(d0, scale, acc[m, n, 0])
                            acc[m, n, 1] = _fma(d1, scale, acc[m, n, 1])
                            acc[m, n, 2] = _fma(d2, scale, acc[m, n, 2])
                            acc[m, n, 3] = _fma(d3, scale, acc[m, n, 3])
            cute.arch.cp_async_wait_group(0)
            for m in cutlass.range_constexpr(4):
                row = (rb0 + wm * Int32(4) + Int32(m)) * Int32(16) + (lane >> Int32(2))
                for n in cutlass.range_constexpr(4):
                    col = nb * Int32(128) + wn * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                    address = output.iterator.toint() + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
                    if row < rows:
                        st_global_v2_f32(address, acc[m, n, 0], acc[m, n, 1])
                    if row + Int32(8) < rows:
                        st_global_v2_f32(address + Int64(self.n * 32), acc[m, n, 2], acc[m, n, 3])
            cute.arch.sync_threads()
            task += grid

    @cute.jit
    def __call__(self, inputs: cute.Pointer, payload: cute.Pointer, scales: cute.Pointer,
                 scratch: cute.Pointer, output: cute.Pointer, rows: Int32,
                 rotation: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(self.max_rows) and rotation >= Int32(0) and rotation < Int32(self.rotations) and cta_limit > Int32(0):
            layout = cute.make_layout(1, stride=1)
            records = cute.make_tensor(scratch, layout)
            self.quantize(cute.make_tensor(inputs, layout), records, rows, rotation).launch(
                grid=(min(((rows + Int32(15)) // Int32(16) * Int32(self.k // 32) + Int32(7)) // Int32(8), cta_limit), 1, 1),
                block=(256, 1, 1), stream=stream)
            self.gemm(cute.make_tensor(payload, layout), cute.make_tensor(scales, layout), records,
                      cute.make_tensor(output, layout), rows).launch(
                grid=(min((rows + Int32(self.rb * 16 - 1)) // Int32(self.rb * 16) * Int32(self.n // 128), cta_limit), 1, 1),
                block=(self.warps * 32, 1, 1), stream=stream, min_blocks_per_mp=1)
