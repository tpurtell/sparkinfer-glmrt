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
    # CUDA13.2 libdevice __nv_erff polynomial and branch, explicit RN FMAs.
    # This avoids MLIR math.erf selecting a different libdevice approximation.
    absolute = cute.math.abs(value)
    argument = _mul(value, value)
    c0, c1 = Float32(6.749860040144995e-05), Float32(-0.0007647646707482636)
    c2, c3 = Float32(0.005143017042428255), Float32(-0.026827244088053703)
    c4, c5, c6 = Float32(0.11282869428396225), Float32(-0.3761255741119385), Float32(0.12837916612625122)
    large = absolute >= Float32(1.003758430480957)
    if large:
        argument = absolute
        c0, c1 = Float32(6.7848181060981e-05), Float32(-0.0009299388038925827)
        c2, c3 = Float32(0.006954224780201912), Float32(-0.03752800449728966)
        c4, c5, c6 = Float32(0.1566874235868454), Float32(0.9144431948661804), Float32(0.6287218928337097)
    polynomial = _fma(c0, argument, c1)
    polynomial = _fma(polynomial, argument, c2)
    polynomial = _fma(polynomial, argument, c3)
    polynomial = _fma(polynomial, argument, c4)
    polynomial = _fma(polynomial, argument, c5)
    polynomial = _fma(polynomial, argument, c6)
    result = _fma(polynomial, value, value)
    if large:
        result = _sub(Float32(1.0), _ex2_ftz(-cute.math.abs(result)))
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
