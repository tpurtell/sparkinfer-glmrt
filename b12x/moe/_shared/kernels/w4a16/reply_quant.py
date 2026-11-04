"""Block codecs shared by the standalone reply kernel and the fused epilogue.

The combined Spark expert reply is quantized from the BF16-rounded top-k sum.
``ExpertReply`` reads that sum back from memory; the fused top-k-sum epilogue
still has it in registers.  Both call the primitives below so the wire bytes
are identical (FP8 E4M3 K32, or NVFP4 E2M1 K16 with an E4M3 block scale).
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16, div_rn_f32


FORMAT_BF16 = 0
FORMAT_FP8 = 1
FORMAT_NVFP4 = 2


def reply_wire_bytes(width: int, format: int) -> int:
    """Bytes of one row of the compressed reply wire."""
    if format == FORMAT_FP8:
        return width + width // 32
    if format == FORMAT_NVFP4:
        return width // 2 + width // 16
    raise ValueError(f"unsupported compressed reply format {format}")


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
def store_compressed_reply(
    output_flat: cute.Tensor,
    row: Int32,
    col: Int32,
    lane: Int32,
    width: Int32,
    format: cutlass.Constexpr,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
):
    """Quantize one lane's four consecutive columns of a row and store them.

    ``col`` is the first column of this lane's quad; the lane layout is the
    standalone reply codec's (32 threads, four consecutive columns each), so
    eight lanes cover one FP8 K32 block and four cover one NVFP4 K16 block.
    The block scale is reduced with the same butterfly offsets, and the value
    is quantized from the BF16-rounded sum, matching the codec's read-back.
    """
    values = cute.make_rmem_tensor((4,), Float32)
    values[0] = Float32(BFloat16(v0))
    values[1] = Float32(BFloat16(v1))
    values[2] = Float32(BFloat16(v2))
    values[3] = Float32(BFloat16(v3))
    amax = Float32(0.0)
    for j in cutlass.range_constexpr(4):
        amax = cutlass.max(amax, cute.math.abs(values[j]))
    for shift in cutlass.range_constexpr(2):
        amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift))
    if cutlass.const_expr(format == FORMAT_FP8):
        amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax, offset=4))
        scale, code = Float32(1.0), Uint32(0)
        if amax > Float32(0.0):
            scale, code = fp8_scale(amax)
        base = Int64(row) * Int64(width + width // Int32(32))
        for j in cutlass.range_constexpr(4):
            value = cvt_f32_to_e4m3(div_rn_f32(values[j], scale))
            if amax == Float32(0.0):
                value = Uint32(0)
            output_flat[base + Int64(col) + Int64(j)] = Uint8(value)
        if lane % Int32(8) == Int32(0):
            output_flat[base + Int64(width) + Int64(col // Int32(32))] = Uint8(code)
    else:
        code, scale = Uint32(0), Float32(1.0)
        if amax > Float32(0.0):
            code = cvt_f32_to_e4m3(div_rn_f32(amax, Float32(6.0)))
            scale = cvt_e4m3_to_f32_via_f16(code)
            if scale < Float32(2.0 ** -9):
                scale, code = Float32(2.0 ** -9), Uint32(1)
        base = Int64(row) * Int64(width // Int32(2) + width // Int32(16))
        for j in cutlass.range_constexpr(2):
            packed = _fp4_pair(div_rn_f32(values[j * 2], scale),
                               div_rn_f32(values[j * 2 + 1], scale))
            if amax == Float32(0.0):
                packed = Uint32(0)
            output_flat[base + Int64(col // Int32(2)) + Int64(j)] = Uint8(packed)
        if lane % Int32(4) == Int32(0):
            output_flat[base + Int64(width // Int32(2)) + Int64(col // Int32(16))] = Uint8(code)
