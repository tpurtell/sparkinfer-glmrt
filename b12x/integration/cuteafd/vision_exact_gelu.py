"""ATen erf-GELU operation order, separate from legacy media GELU."""
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _sub, _mul, _fma, _view


@dsl_user_op
def _ex2_ftz(value: Float32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [value.ir_value(loc=loc, ip=ip)],
        "ex2.approx.ftz.f32 $0, $1;", "=f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def _libdevice_erff(value):
    # CUDA13.0.88 libdevice __nv_erff, matching the ATen cu130 wheel's SASS.
    # CUDA13.2 changed coefficients and threshold; don't inherit host tooling.
    absolute = cute.math.abs(value)
    argument = _mul(value, value)
    c0, c1 = Float32(8.483494457323104e-05), Float32(-0.0008213091641664505)
    c2, c3 = Float32(0.005213488824665546), Float32(-0.026868773624300957)
    c4, c5, c6 = Float32(0.11284004896879196), Float32(-0.37612664699554443), Float32(0.12837915122509003)
    large = absolute >= Float32(1.002959966659546)
    if large:
        argument = absolute
        c0, c1 = Float32(0.00011219871521461755), Float32(-0.0013275252422317863)
        c2, c3 = Float32(0.00839653518050909), Float32(-0.04024658352136612)
        c4, c5, c6 = Float32(0.15950430929660797), Float32(0.9129176735877991), Float32(0.6290600299835205)
    polynomial = _fma(c0, argument, c1)
    polynomial = _fma(polynomial, argument, c2)
    polynomial = _fma(polynomial, argument, c3)
    polynomial = _fma(polynomial, argument, c4)
    polynomial = _fma(polynomial, argument, c5)
    polynomial = _fma(polynomial, argument, c6)
    factor = value
    if large:
        factor = -absolute
    result = _fma(polynomial, factor, factor)
    if large:
        result = _sub(Float32(1.0), _ex2_ftz(result))
        result = cute.math.copysign(result, value)
    return result


class ExactGelu:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.width)
        self.kernel(x, out, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, count: Int64):
        x, out = _view(x), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            value = Float32(x[at])
            argument = _mul(value, Float32(0.7071067811865476))
            activation = _libdevice_erff(argument)
            scaled = _mul(value, Float32(0.5))
            out[at] = _mul(scaled, _add(Float32(1.0), activation)).to(BFloat16)


def compile_exact_vision_gelu_aot(*, width):
    if width <= 0:
        raise ValueError("exact vision GELU width must be positive")
    return compile_program(ExactGelu(width), name="vision_exact_gelu",
        operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,),
        geometry=dict(width=width, activation="aten_erf_gelu"))
