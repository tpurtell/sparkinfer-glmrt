"""Full-rotation top-k sum with the combined reply codec in its epilogue."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_ptr

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16,
                                 div_rn_f32, f16x2_to_f32x2, get_ptr_as_int64,
                                 ld_global_nc_v2_u32)
from b12x._lib.utils import current_cuda_stream
from b12x.integration.cuteafd.pointwise import _view
from .kernel import W4A16TopKSumKernel
from .expert_partial import _fp4_pair, fp8_scale


class ReplySum(W4A16TopKSumKernel):
    def __init__(self, *, reply_format, **options):
        super().__init__(**options)
        if not self.full_rotation or not self.full_rotation_output_bf16 or self.coupled_hadamard:
            raise ValueError("reply sum requires uncoupled full-rotation BF16 output")
        if reply_format not in (1, 2):
            raise ValueError("reply format must be FP8=1 or NVFP4=2")
        self.reply_format = reply_format

    @cute.jit
    def __call__(self, fc2, output, weights, ids, expert_map, svh,
                 experts: Int32, route_experts: Int32, rows: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(fc2, output, weights, ids, expert_map, svh,
                        experts, route_experts, rows).launch(
                grid=(rows * Int32(self.hidden_size // 128), 1, 1),
                block=(256, 1, 1), stream=stream)

    @cute.jit
    def encode(self, out, row: Int64, col: Int64, lane: Int32,
               acc0: Float32, acc1: Float32, acc2: Float32, acc3: Float32):
        # Quantize the exact BF16 value the unfused sum would have stored.
        values = cute.make_rmem_tensor((4,), Float32)
        values[0], values[1] = Float32(BFloat16(acc0)), Float32(BFloat16(acc1))
        values[2], values[3] = Float32(BFloat16(acc2)), Float32(BFloat16(acc3))
        amax = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            amax = cutlass.max(amax, cute.math.abs(values[j]))
        for shift in cutlass.range_constexpr(2):
            amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift))
        if cutlass.const_expr(self.reply_format == 1):
            amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=4))
        scale, code = Float32(1.0), Uint32(0)
        h = Int64(self.hidden_size)
        if cutlass.const_expr(self.reply_format == 2):
            if amax > Float32(0.0):
                code = cvt_f32_to_e4m3(div_rn_f32(amax, Float32(6.0)))
                scale = cvt_e4m3_to_f32_via_f16(code)
                if scale < Float32(2.0 ** -9):
                    scale, code = Float32(2.0 ** -9), Uint32(1)
            base = row * (h // Int64(2) + h // Int64(16))
            for j in cutlass.range_constexpr(2):
                packed = _fp4_pair(div_rn_f32(values[j * 2], scale),
                                   div_rn_f32(values[j * 2 + 1], scale))
                if amax == Float32(0.0):
                    packed = Uint32(0)
                out[base + col // Int64(2) + Int64(j)] = Uint8(packed)
            if lane % Int32(4) == Int32(0):
                out[base + h // Int64(2) + col // Int64(16)] = Uint8(code)
        else:
            if amax > Float32(0.0):
                scale, code = fp8_scale(amax)
            base = row * (h + h // Int64(32))
            for j in cutlass.range_constexpr(4):
                value = cvt_f32_to_e4m3(div_rn_f32(values[j], scale))
                if amax == Float32(0.0):
                    value = Uint32(0)
                out[base + col + Int64(j)] = Uint8(value)
            if lane % Int32(8) == Int32(0):
                out[base + h + col // Int64(32)] = Uint8(code)

    @cute.kernel
    def kernel(self, fc2, output, weights, ids, expert_map, svh,
               experts: Int32, route_experts: Int32, rows: Int32):
        fc2, output, weights, ids, expert_map, svh = (
            _view(p) for p in (fc2, output, weights, ids, expert_map, svh))
        tid = Int32(cute.arch.thread_idx()[0])
        lane, warp = tid & Int32(31), tid >> Int32(5)
        unit = Int64(cute.arch.block_idx()[0])
        nblk = Int64(self.hidden_size // 128)
        token, blk = unit // nblk, unit % nblk
        col0 = blk * Int64(128) + Int64(lane) * Int64(4)
        route_values_ptr = cute.arch.alloc_smem(Float32, self.topk * 128)
        route_values = cute.make_tensor(route_values_ptr, cute.make_layout(self.topk * 128))
        route_weights_ptr = cute.arch.alloc_smem(Float32, self.topk)
        route_weights = cute.make_tensor(route_weights_ptr, cute.make_layout(self.topk))
        for group in cutlass.range_constexpr((self.topk + self.route_warps - 1) // self.route_warps):
            route = warp + Int32(group * self.route_warps)
            valid = Int32(0)
            expert = Int32(-1)
            row = token * Int64(self.topk) + Int64(route)
            if route < Int32(self.topk):
                raw = Int32(ids[row])
                expert = raw
                if cutlass.const_expr(self.use_expert_map):
                    expert = Int32(-1)
                    if raw >= Int32(0) and raw < route_experts:
                        expert = Int32(expert_map[raw])
                if expert >= Int32(0) and expert < experts:
                    valid = Int32(1)
            hs0, hs1, hs2, hs3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            if valid != Int32(0):
                base = row * Int64(self.hidden_size) + col0
                sbase = Int64(expert) * Int64(self.hidden_size) + col0
                if cutlass.const_expr(self.broadcast_svh):
                    sbase = col0
                p01, p23 = ld_global_nc_v2_u32(get_ptr_as_int64(fc2, base))
                sp01, sp23 = ld_global_nc_v2_u32(get_ptr_as_int64(svh, sbase))
                v0, v1 = f16x2_to_f32x2(p01)
                v2, v3 = f16x2_to_f32x2(p23)
                h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
                s0, s1 = f16x2_to_f32x2(sp01)
                s2, s3 = f16x2_to_f32x2(sp23)
                hs0, hs1, hs2, hs3 = h0 * s0, h1 * s1, h2 * s2, h3 * s3
            at = route * Int32(128) + lane * Int32(4)
            if route < Int32(self.topk):
                route_values[at], route_values[at + 1] = hs0, hs1
                route_values[at + 2], route_values[at + 3] = hs2, hs3
                if lane == Int32(0):
                    weight = Float32(0.0)
                    if valid != Int32(0):
                        weight = Float32(weights[row])
                    route_weights[route] = weight
        cute.arch.sync_threads()
        if warp == Int32(0):
            acc0, acc1, acc2, acc3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            for route in cutlass.range_constexpr(self.topk):
                at = Int32(route * 128) + lane * Int32(4)
                weight = route_weights[Int32(route)]
                acc0 += route_values[at] * weight
                acc1 += route_values[at + 1] * weight
                acc2 += route_values[at + 2] * weight
                acc3 += route_values[at + 3] * weight
            self.encode(output, token, col0, lane, acc0, acc1, acc2, acc3)


def compile_reply_sum(*, reply_format, m, topk, hidden_size, element_dtype,
                      full_rotation, full_rotation_output_dtype, num_experts,
                      route_num_experts, route_ids_dtype, use_expert_map,
                      broadcast_svh=False, coupled_hadamard=False):
    import torch
    if route_ids_dtype != torch.int32:
        raise ValueError("reply sum requires int32 route IDs")
    kernel = ReplySum(reply_format=reply_format, topk=topk, hidden_size=hidden_size,
                      element_dtype=element_dtype, full_rotation=full_rotation,
                      full_rotation_output_dtype=full_rotation_output_dtype,
                      num_experts=num_experts, route_num_experts=route_num_experts,
                      use_expert_map=use_expert_map, broadcast_svh=broadcast_svh,
                      coupled_hadamard=coupled_hadamard)

    def ptr(dtype, address, align=16):
        return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)

    key = (topk, hidden_size, reply_format, use_expert_map, broadcast_svh)
    compiled = b12x_compile(kernel, ptr(Float16, 16), ptr(Uint8, 16), ptr(Float32, 4, 4),
                           ptr(Int32, 4, 4), ptr(Int32, 4, 4), ptr(Float16, 16),
                           Int32(num_experts), Int32(route_num_experts), Int32(1),
                           current_cuda_stream(),
                           compile_spec=KernelCompileSpec.from_key("moe.w4a16.reply_sum", 1, key))

    def launch(fc2, out, weights, ids, expert_map, svh, rows, stream):
        compiled(ptr(Float16, fc2.data_ptr()), ptr(Uint8, out.data_ptr()),
                 ptr(Float32, weights.data_ptr(), 4), ptr(Int32, ids.data_ptr(), 4),
                 ptr(Int32, expert_map.data_ptr(), 4), ptr(Float16, svh.data_ptr()),
                 Int32(num_experts), Int32(route_num_experts), Int32(rows), stream)

    launch.compiled = compiled
    return launch
