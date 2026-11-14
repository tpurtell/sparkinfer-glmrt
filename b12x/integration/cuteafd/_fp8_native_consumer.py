"""Bounded native FP8 K64 consumer shared by thin and wide decode GEMV."""
from cutlass import Float32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def native_k64(c0, c1, c2, c3, w0, w1, w2, w3,
               a00, a01, a02, a03, a10, a11, a12, a13,
               a20, a21, a22, a23, a30, a31, a32, a33,
               scale_even, scale_odd, *, loc=None, ip=None):
    """Four zero-seeded MMAs, then the base kernel's contracted FP32 scale/add.

    Keep weights packed until their MMA consumes them. The assembler sees only
    one widened weight word and one four-float partial live at a time, rather
    than a complete K64 of widened operands around a separately scheduled fold.
    """
    inputs = [Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)]
    inputs += [Uint32(v).ir_value(loc=loc, ip=ip) for v in (
        w0, w1, w2, w3, a00, a01, a02, a03, a10, a11, a12, a13,
        a20, a21, a22, a23, a30, a31, a32, a33)]
    inputs += [Float32(v).ir_value(loc=loc, ip=ip) for v in (scale_even, scale_odd)]
    asm = ["{", ".reg .b16 e0, e1;", ".reg .b32 b0, b1;",
           ".reg .f32 p0, p1, p2, p3;",
           "mov.f32 p0, 0f00000000;", "mov.f32 p1, 0f00000000;",
           "mov.f32 p2, 0f00000000;", "mov.f32 p3, 0f00000000;"]
    for t in range(4):
        a = 12 + 4 * t
        asm += [f"mov.b32 {{e0, e1}}, ${8 + t};",
                "cvt.rn.bf16x2.e4m3x2 b0, e0;",
                "cvt.rn.bf16x2.e4m3x2 b1, e1;",
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                f"{{p0, p1, p2, p3}}, {{${a}, ${a+1}, ${a+2}, ${a+3}}}, "
                "{b0, b1}, {p0, p1, p2, p3};"]
    for v in range(4):
        asm.append(f"fma.rn.f32 p{v}, p{v}, ${28 + v % 2}, ${4 + v};")
    # Consume every input before writing outputs: LLVM may alias an output
    # with a scale or accumulator input under the ordinary '=f' constraint.
    for v in range(4):
        asm.append(f"mov.f32 ${v}, p{v};")
    asm.append("}")
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4), inputs, "\n".join(asm),
        ",".join(["=f"] * 4 + ["f"] * 4 + ["r"] * 20 + ["f"] * 2),
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [v], loc=loc, ip=ip)) for v in range(4))
