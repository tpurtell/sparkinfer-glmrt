"""Register-streaming NVFP4 decode over adjacent K256/N8 weight-scale records."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32
from b12x._lib.intrinsics import (ld_global_v4_u32, nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2, shared_ptr_to_u32, st_shared_u32, ld_shared_v4_u32)
from ._fp8_moe_kernels import _i32_at, _ld_v2_u32
from ._fp8_moe_stream import _chunk, STREAM_TILE_M
from ._fp8_weights import _ld_f32
from ._nvfp4_moe_a4 import _quantize_row16, _swiglu16, nvfp4_input_scale_offset


class DecodeNvfp4GateUp:
    threads = 128
    half_n = 16
    x_stride = 80

    def __init__(self, inter, hidden, experts, limit):
        self.inter, self.k, self.experts, self.limit = inter, hidden, experts, limit
        self.alpha = experts * inter * hidden // 16
        self.inscale = self.alpha + 4 * experts
        self.down_in = nvfp4_input_scale_offset(experts, hidden, inter)

    def key(self):
        return ('decode_nvfp4_adjacent', 1, self.inter, self.k, self.experts, self.limit)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer,
                 act: cute.Pointer, act_s: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(aq, a_s, meta, w1, s1, w3, s3, s2, act, act_s).launch(
            grid=(self.inter // 16, max_tiles, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer,
               s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer,
               act: cute.Pointer, act_s: cute.Pointer):
        tile = Int32(cute.arch.block_idx()[1])
        nb = Int32(cute.arch.block_idx()[0])
        smem = utils.SmemAllocator()
        exchange = smem.allocate_tensor(Uint32, cute.make_layout((320,)), byte_alignment=16)
        ring = shared_ptr_to_u32(exchange.iterator)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if live <= Int32(16):
                tid = Int32(cute.arch.thread_idx()[0])
                warp, lane = tid // Int32(32), tid % Int32(32)
                q, c = lane // Int32(4), lane % Int32(4)
                projection = warp // Int32(2)
                col = nb * Int32(16) + (warp % Int32(2)) * Int32(8)
                wp = cutlass.select_(projection == Int32(0), Int64(w1.toint()), Int64(w3.toint()))
                sp = cutlass.select_(projection == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
                factor = _ld_f32(sp + Int64(self.alpha) + Int64(e) * Int64(4)) * _ld_f32(
                    sp + Int64(self.inscale) + Int64(e) * Int64(4))
                wb = wp + (Int64(e) * Int64(self.inter // 8) + Int64(col // Int32(8))) * Int64(self.k // 256 * 1152)
                ab = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M * (self.k // 2))
                sb = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M * (self.k // 16))
                d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
                for block in cutlass.range(self.k // 256, unroll=1):
                    base = wb + Int64(block) * Int64(1152)
                    wv = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
                    s = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                    v0, v1, v2, v3 = ld_global_v4_u32(base + Int64(lane) * Int64(32))
                    wv[0], wv[1], wv[2], wv[3] = v0, v1, v2, v3
                    v0, v1, v2, v3 = ld_global_v4_u32(base + Int64(lane) * Int64(32) + Int64(16))
                    wv[4], wv[5], wv[6], wv[7] = v0, v1, v2, v3
                    v0, v1, v2, v3 = ld_global_v4_u32(base + Int64(1024) + Int64(q) * Int64(16))
                    s[0], s[1], s[2], s[3] = v0, v1, v2, v3
                    for step in cutlass.range_constexpr(4):
                        off = Int64(block) * Int64(128) + Int64(step * 32) + Int64(c) * Int64(4)
                        # K halves are 16 bytes apart, not adjacent words.
                        ap0 = cute.make_ptr(Uint32, ab + Int64(q) * Int64(self.k // 2) + off, cute.AddressSpace.gmem, assumed_align=4)
                        ap1 = cute.make_ptr(Uint32, ab + Int64(q + Int32(8)) * Int64(self.k // 2) + off, cute.AddressSpace.gmem, assumed_align=4)
                        asp = cute.make_ptr(Uint32, sb + Int64(q + Int32(8) * (c & Int32(1))) * Int64(self.k // 16)
                            + Int64(block) * Int64(16) + Int64(step * 4), cute.AddressSpace.gmem, assumed_align=4)
                        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(d0, d1, d2, d3,
                            ap0[0], ap1[0], ap0[4], ap1[4], wv[2 * step], wv[2 * step + 1], asp[0], s[step])
                for half in cutlass.range_constexpr(2):
                    row = q + Int32(8 * half)
                    at = ring + row * Int32(80) + (projection * Int32(16) + warp % Int32(2) * Int32(8) + c * Int32(2)) * Int32(2)
                    if cutlass.const_expr(half == 0):
                        st_shared_u32(at, pack_f32x2_to_bfloat2(d0 * factor, d1 * factor))
                    else:
                        st_shared_u32(at, pack_f32x2_to_bfloat2(d2 * factor, d3 * factor))
                cute.arch.sync_threads()
                if tid < live:
                    gw = ld_shared_v4_u32(ring + tid * Int32(80))
                    uw = ld_shared_v4_u32(ring + tid * Int32(80) + Int32(32))
                    values = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
                    _swiglu16(gw, uw, values, 0, self.limit)
                    gw = ld_shared_v4_u32(ring + tid * Int32(80) + Int32(16))
                    uw = ld_shared_v4_u32(ring + tid * Int32(80) + Int32(48))
                    _swiglu16(gw, uw, values, 8, self.limit)
                    gs = Float32(1) / _ld_f32(Int64(s2.toint()) + Int64(self.down_in) + Int64(e) * Int64(4))
                    rr = Int64(tile) * Int64(STREAM_TILE_M) + Int64(tid)
                    _quantize_row16(values, gs, Int64(act.toint()) + rr * Int64(self.inter // 2) + Int64(nb) * Int64(8),
                        Int64(act_s.toint()) + rr * Int64(self.inter // 16) + Int64(nb))
