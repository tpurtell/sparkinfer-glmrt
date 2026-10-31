"""Muse TP rank reduction and per-route post-expert normalization."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import cvt_e4m3_to_f32_via_f16, div_rn_f32
from b12x.moe._shared.kernels.w4a16.expert_partial import decode_fp4, partial_row_bytes
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _bf16, _warp_sum
from .pointwise import _add, _mul, _sqrt_rn, _view


@dsl_user_op
def _ue8m0_scale(code, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Uint32(code).ir_value()],
        "{ .reg .b32 bits; .reg .pred zero; shl.b32 bits, $1, 23; "
        "setp.eq.u32 zero, $1, 0; @zero mov.b32 bits, 4194304; mov.b32 $0, bits; }",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


class TpCombine:
    def __init__(self, width, topk, format):
        self.width, self.topk, self.format = width, topk, format
        self.stride = partial_row_bytes(width, format)

    @cute.jit
    def __call__(self, data: cute.Pointer, row_map: cute.Pointer, weights: cute.Pointer,
                 norm: cute.Pointer, out: cute.Pointer, ranks: Int32, region_rows: Int32,
                 rows: Int32, stream: cuda.CUstream):
        if region_rows > Int32(0):
            self.kernel(data, row_map, weights, norm, out, ranks, region_rows, rows).launch(
                grid=(region_rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def partial(self, data: cute.Tensor, base: Int64, col: Int64):
        value = Float32(0.0)
        if cutlass.const_expr(self.format == "bf16"):
            value = Float32(data[base + col])
        elif cutlass.const_expr(self.format == "nvfp4"):
            code = Uint32(data[base + col // Int64(2)])
            code = (code >> (Uint32(col % Int64(2)) * Uint32(4))) & Uint32(15)
            scale = cvt_e4m3_to_f32_via_f16(Uint32(data[base + Int64(self.width // 2) + col // Int64(16)]))
            value = _mul(decode_fp4(code), scale)
        else:
            code = Uint32(data[base + col])
            scale = _ue8m0_scale(Uint32(data[base + Int64(self.width) + col // Int64(32)]))
            value = _mul(cvt_e4m3_to_f32_via_f16(code), scale)
        return value

    @cute.kernel
    def kernel(self, data: cute.Pointer, row_map: cute.Pointer, weights: cute.Pointer,
               norm: cute.Pointer, out: cute.Pointer, ranks: Int32, region_rows: Int32, rows: Int32):
        data, row_map, weights, norm, out = (_view(p) for p in (data, row_map, weights, norm, out))
        region = Int64(cute.arch.block_idx()[0])
        row = Int64(row_map[region])
        tid = Int32(cute.arch.thread_idx()[0])
        warp, lane = tid // Int32(32), tid % Int32(32)
        values = cute.make_rmem_tensor((self.width // 256,), Float32)
        accumulator = cute.make_rmem_tensor((self.width // 256,), Float32)
        accumulator.fill(0.0)
        totals = cutlass.utils.SmemAllocator().allocate_tensor(Float32, cute.make_layout((8,)))
        if row < Int64(rows):
            for route in cutlass.range_constexpr(self.topk):
                squares = Float32(0.0)
                for slab in cutlass.range_constexpr(self.width // 1024):
                    for j in cutlass.range_constexpr(4):
                        col = Int64(slab * 1024) + Int64(warp) * Int64(128) + Int64(lane) * Int64(4) + Int64(j)
                        summed = Float32(0.0)
                        for rank in cutlass.range(ranks, unroll=1):
                            stride = self.width if self.format == "bf16" else self.stride
                            base = ((Int64(rank) * Int64(region_rows) + region) * Int64(self.topk) + Int64(route)) * Int64(stride)
                            summed = _add(summed, self.partial(data, base, col))
                        value = _bf16(_mul(_bf16(summed), _bf16(_add(Float32(1.0), Float32(norm[col])))))
                        values[slab * 4 + j] = value
                        squares = _add(squares, _mul(value, value))
                squares = _warp_sum(squares)
                if lane == Int32(0):
                    totals[warp] = squares
                cute.arch.sync_threads()
                total = Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    total = _add(total, Float32(totals[w]))
                inv = div_rn_f32(Float32(1.0), _sqrt_rn(_add(div_rn_f32(total, Float32(self.width)), Float32(1e-8))))
                weight = Float32(weights[row * Int64(self.topk) + Int64(route)])
                for item in cutlass.range_constexpr(self.width // 256):
                    weighted = _bf16(_mul(_bf16(_mul(values[item], inv)), weight))
                    accumulator[item] = _bf16(_add(accumulator[item], weighted))
                cute.arch.sync_threads()
            for slab in cutlass.range_constexpr(self.width // 1024):
                for j in cutlass.range_constexpr(4):
                    col = Int64(slab * 1024) + Int64(warp) * Int64(128) + Int64(lane) * Int64(4) + Int64(j)
                    out[row * Int64(self.width) + col] = BFloat16(accumulator[slab * 4 + j])


def compile_tp_combine_aot(*, format, width=4096, topk=8):
    stride = partial_row_bytes(width, format)
    if width % 1024 or topk <= 0:
        raise ValueError("Muse TP combine requires H1024 width and positive topk")
    return compile_program(TpCombine(width, topk, format), name=f"muse_tp_combine_{format}",
        operands=(Operand("data", torch.bfloat16 if format == "bf16" else torch.uint8,
                          f"[ranks,region_rows,{topk},{width if format == 'bf16' else stride}]"),
                  Operand("row_map", torch.uint32, "[region_rows]", align=4),
                  Operand("weights", torch.float32, f"[rows,{topk}]"),
                  Operand("norm", torch.bfloat16, f"[{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=tuple(Scalar(n) for n in ("ranks", "region_rows", "rows")),
        key=(width, topk, format), geometry=dict(width=width, topk=topk, format=format, eps=1e-8))
