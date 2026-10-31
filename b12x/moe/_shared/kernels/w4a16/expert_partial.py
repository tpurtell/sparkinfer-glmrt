"""Unweighted EXL3 TP replies, compacted by a request-stable region row map.

The existing weighted-sum kernel is not changed. This epilogue replaces it
only when a worker selects per-route partial replies.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16,
                                 div_rn_f32)
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd.pointwise import _view
from .kernel import W4A16TopKSumKernel


FORMATS = ("bf16", "nvfp4", "fp8")


def partial_row_bytes(width, format):
    if width <= 0 or width % 128 or format not in FORMATS:
        raise ValueError("expert partials require a positive H128 width and bf16/nvfp4/fp8")
    stride = {"bf16": width * 2, "nvfp4": width // 2 + width // 16,
              "fp8": width + width // 32}[format]
    if stride % 16:
        raise ValueError(f"expert partial {format} width {width} has a non-16-byte-aligned record ({stride} bytes)")
    return stride


@dsl_user_op
def _fp4_pair(lo, hi, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(T.i32(),
        [Float32(lo).ir_value(), Float32(hi).ir_value()],
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        "=r,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fp8_scale(amax, *, loc=None, ip=None):
    # Compare amax's significand to 448 / 256 exactly. Division/log2 can
    # underflow or round across a power-of-two boundary before the ceil.
    result = llvm.inline_asm(llvm.StructType.get_literal([T.f32(), T.i32()]),
        [Float32(amax).ir_value()], """
        { .reg .pred tiny, above; .reg .f32 x; .reg .b32 bits, mant;
          .reg .s32 exponent, adjust;
          mov.f32 x, $2;
          setp.lt.f32 tiny, x, 0f00800000;
          @tiny mul.rn.f32 x, x, 0f4B800000;
          selp.s32 adjust, 24, 0, tiny;
          mov.b32 bits, x;
          bfe.u32 exponent, bits, 23, 8;
          sub.s32 exponent, exponent, 135;
          sub.s32 exponent, exponent, adjust;
          and.b32 mant, bits, 8388607;
          setp.gt.u32 above, mant, 6291456;
          @above add.s32 exponent, exponent, 1;
          max.s32 exponent, exponent, -127;
          min.s32 exponent, exponent, 127;
          add.s32 $1, exponent, 127;
          shl.b32 bits, $1, 23;
          setp.eq.u32 tiny, $1, 0;
          @tiny mov.b32 bits, 4194304;
          mov.b32 $0, bits;
        }""", "=f,=r,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return (Float32(llvm.extractvalue(T.f32(), result, [0])),
            Uint32(llvm.extractvalue(T.i32(), result, [1])))


@cute.jit
def decode_fp4(code: Uint32):
    magnitude = code & Uint32(7)
    value = Float32(0.0)
    for i in cutlass.range_constexpr(8):
        if magnitude == Uint32(i):
            value = Float32((0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[i])
    if (code & Uint32(8)) != Uint32(0):
        value = -value
    return value


class ExpertPartial:
    def __init__(self, width, topk, format, rotated=True):
        self.width, self.topk, self.format = width, topk, format
        self.rotated = rotated
        self.stride = partial_row_bytes(width, format)
        self.hadamard = W4A16TopKSumKernel(topk=topk, hidden_size=width,
            element_dtype="fp16", full_rotation=True, num_experts=1)

    @cute.jit
    def __call__(self, fc2: cute.Pointer, svh: cute.Pointer, ids: cute.Pointer,
                 expert_map: cute.Pointer, row_map: cute.Pointer, out: cute.Pointer,
                 region_rows: Int32, rows: Int32, local_experts: Int32, experts: Int32,
                 stream: cuda.CUstream):
        if region_rows > Int32(0):
            self.kernel(fc2, svh, ids, expert_map, row_map, out, rows, local_experts, experts).launch(
                grid=(region_rows, self.topk, self.width // 128), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, fc2: cute.Pointer, svh: cute.Pointer, ids: cute.Pointer,
               expert_map: cute.Pointer, row_map: cute.Pointer, out: cute.Pointer,
               rows: Int32, local_experts: Int32, experts: Int32):
        fc2, svh, ids, expert_map, row_map, out = (_view(p) for p in (fc2, svh, ids, expert_map, row_map, out))
        region, route, slab = cute.arch.block_idx()
        lane = Int32(cute.arch.thread_idx()[0])
        row = Int64(row_map[Int64(region)])
        col = Int64(slab) * Int64(128) + Int64(lane) * Int64(4)
        values = cute.make_rmem_tensor((4,), Float32)
        values.fill(0.0)
        if row < Int64(rows):
            if cutlass.const_expr(self.rotated):
                raw = Int32(ids[row * Int64(self.topk) + Int64(route)])
                expert = Int32(-1)
                if raw >= Int32(0) and raw < experts:
                    expert = Int32(expert_map[Int64(raw)])
                if expert >= Int32(0) and expert < local_experts:
                    base = (row * Int64(self.topk) + Int64(route)) * Int64(self.width) + col
                    v0, v1, v2, v3 = self.hadamard._had128_quad(
                        Float32(fc2[base]), Float32(fc2[base + Int64(1)]),
                        Float32(fc2[base + Int64(2)]), Float32(fc2[base + Int64(3)]), lane)
                    values[0], values[1], values[2], values[3] = v0, v1, v2, v3
                    for j in cutlass.range_constexpr(4):
                        values[j] = values[j] * Float32(svh[Int64(expert) * Int64(self.width) + col + Int64(j)])
            else:
                for j in cutlass.range_constexpr(4):
                    values[j] = Float32(fc2[(row * Int64(self.topk) + Int64(route)) * Int64(self.width) + col + Int64(j)])
            base = (Int64(region) * Int64(self.topk) + Int64(route)) * Int64(self.stride)
            if cutlass.const_expr(self.format == "bf16"):
                base = (Int64(region) * Int64(self.topk) + Int64(route)) * Int64(self.width)
                for j in cutlass.range_constexpr(4):
                    out[base + col + Int64(j)] = BFloat16(values[j])
            else:
                amax = Float32(0.0)
                for j in cutlass.range_constexpr(4):
                    amax = cutlass.max(amax, cute.math.abs(values[j]))
                for shift in cutlass.range_constexpr(2 if self.format == "nvfp4" else 3):
                    amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift))
                scale, code = Float32(1.0), Uint32(0)
                if cutlass.const_expr(self.format == "nvfp4"):
                    if amax > Float32(0.0):
                        code = cvt_f32_to_e4m3(div_rn_f32(amax, Float32(6.0)))
                        scale = cvt_e4m3_to_f32_via_f16(code)
                        if scale < Float32(2.0 ** -9):
                            scale, code = Float32(2.0 ** -9), Uint32(1)
                    for j in cutlass.range_constexpr(2):
                        packed = _fp4_pair(div_rn_f32(values[j * 2], scale),
                                           div_rn_f32(values[j * 2 + 1], scale))
                        if amax == Float32(0.0):
                            packed = Uint32(0)
                        out[base + col // Int64(2) + Int64(j)] = Uint8(packed)
                    if lane % Int32(4) == Int32(0):
                        out[base + Int64(self.width // 2) + col // Int64(16)] = Uint8(code)
                else:
                    if amax > Float32(0.0):
                        scale, code = fp8_scale(amax)
                    for j in cutlass.range_constexpr(4):
                        value = cvt_f32_to_e4m3(div_rn_f32(values[j], scale))
                        if amax == Float32(0.0):
                            value = Uint32(0)
                        out[base + col + Int64(j)] = Uint8(value)
                    if lane % Int32(8) == Int32(0):
                        out[base + Int64(self.width) + col // Int64(32)] = Uint8(code)


def compile_expert_partial_aot(*, width=4096, topk=8, format, rotated=True):
    stride = partial_row_bytes(width, format)
    if topk <= 0:
        raise ValueError("expert partials require positive topk")
    return compile_program(ExpertPartial(width, topk, format, rotated), name=f"expert_partial_{format}",
        operands=(Operand("fc2", torch.float16 if rotated else torch.float32, f"[rows,{topk},{width}]"),
                  Operand("svh", torch.float16, f"[local_experts,{width}]"),
                  Operand("ids", torch.int32, f"[rows,{topk}]", align=4),
                  Operand("expert_map", torch.int32, "[experts]", align=4),
                  Operand("row_map", torch.uint32, "[region_rows]", align=4),
                  Operand("out", torch.bfloat16 if format == "bf16" else torch.uint8,
                          f"[region_rows,{topk},{width if format == 'bf16' else stride}]", "out")),
        scalars=tuple(Scalar(n) for n in ("region_rows", "rows", "local_experts", "experts")),
        key=(width, topk, format, rotated),
        geometry=dict(width=width, topk=topk, format=format, rotated=rotated, row_bytes=stride))
