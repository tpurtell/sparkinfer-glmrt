"""ATen erf-GELU operation order, separate from legacy media GELU."""
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _mul, _view


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
            activation = cute.math.erf(argument, fastmath=False)
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
