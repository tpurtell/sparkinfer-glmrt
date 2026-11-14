"""Inkling per-head L2 clipping, optional checkpoint scale and global log scale.

Exactly the #1915 producer order: v=x.float()*scale.float() (when present),
v*=min(1, threshold/(norm(v)+eps)), then multiply global q/r by
1+alpha*ln(max(1,(position+1)/floor)). K has neither scale nor log scaling.
One warp per head, four warps/CTA. BF16 or F32 I/O, BF16 learned scales;
positions I64, live rows runtime. No state, scratch, allocation or host work.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
import torch

from b12x._lib.intrinsics import div_rn_f32, fmax_f32, fmin_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _warp_sum


class _Clip:
    def __init__(self, heads, dim, threshold, eps, scaled, floor, alpha):
        self.heads, self.dim = heads, dim
        self.threshold, self.eps = float(threshold), float(eps)
        self.scaled, self.floor, self.alpha = scaled, floor, float(alpha)

    @cute.jit
    def __call__(self, x: cute.Pointer, scale: cute.Pointer, positions: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(x, scale, positions, out, rows).launch(
            grid=((rows * self.heads + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, scale: cute.Pointer, positions: cute.Pointer,
               out: cute.Pointer, rows: Int32):
        tid = Int64(cute.arch.thread_idx()[0])
        head_row = Int64(cute.arch.block_idx()[0]) * Int64(4) + tid // Int64(32)
        lane = tid % Int64(32)
        if head_row < Int64(rows) * Int64(self.heads):
            head, row = head_row % Int64(self.heads), head_row // Int64(self.heads)
            values = cute.make_rmem_tensor(((self.dim + 31) // 32,), Float32)
            square = Float32(0)
            for j in cutlass.range_constexpr((self.dim + 31) // 32):
                col = lane + Int64(j * 32)
                value = Float32(0)
                if col < Int64(self.dim):
                    value = Float32(x[head_row * Int64(self.dim) + col])
                    if cutlass.const_expr(self.scaled):
                        value = value * Float32(scale[head * Int64(self.dim) + col])
                values[j] = value
                square = square + value * value
            norm = cute.math.sqrt(_warp_sum(square), fastmath=False)
            factor = fmin_f32(Float32(1), div_rn_f32(Float32(self.threshold), norm + Float32(self.eps)))
            tau = Float32(1)
            if cutlass.const_expr(self.floor > 0):
                ratio = div_rn_f32((positions[row] + Int64(1)).to(Float32), Float32(self.floor))
                tau = Float32(1) + Float32(self.alpha) * cute.math.log(fmax_f32(Float32(1), ratio), fastmath=False)
            for j in cutlass.range_constexpr((self.dim + 31) // 32):
                col = lane + Int64(j * 32)
                if col < Int64(self.dim):
                    out[head_row * Int64(self.dim) + col] = (values[j] * factor) * tau


def compile_inkling2_clip_aot(*, heads: int, dim: int, threshold: float, eps: float = 1e-6,
                             scaled: bool = True, log_floor: int = 0, log_alpha: float = 0.,
                             max_rows: int = 4096, dtype=torch.bfloat16):
    import math
    if (type(heads) is not int or heads <= 0 or dim not in (16, 128)
            or not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(eps) or eps <= 0
            or type(log_floor) is not int or log_floor < 0 or not math.isfinite(log_alpha) or log_alpha < 0
            or type(max_rows) is not int or not 1 <= max_rows <= (2**31 - 1) // heads
            or dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("invalid Inkling clip geometry/scales/capacity/dtype")
    return compile_program(
        _Clip(heads, dim, threshold, eps, scaled, log_floor, log_alpha), name="inkling2_clip",
        operands=(Operand("x", dtype, f"[rows,{heads},{dim}]"),
                  Operand("scale", torch.bfloat16, f"[{heads},{dim}]"),
                  Operand("positions", torch.int64, "[rows]", align=8),
                  Operand("out", dtype, f"[rows,{heads},{dim}]", "out")),
        scalars=(Scalar("rows"),),
        key=(heads, dim, threshold, eps, scaled, log_floor, log_alpha, max_rows, dtype),
        geometry=dict(heads=heads, dim=dim, threshold=threshold, eps=eps, scaled=scaled,
                      log_floor=log_floor, log_alpha=log_alpha, max_rows=max_rows), doc=__doc__)
