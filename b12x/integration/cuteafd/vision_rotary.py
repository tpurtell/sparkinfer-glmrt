"""Native BF16 vision QKV rotary boundaries; tables and row counts are runtime data."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _sub, _mul, _fma, _view


class VisionQkvRotary:
    def __init__(self, heads, dim, rotated, adjacent):
        self.heads, self.dim, self.rotated, self.adjacent = heads, dim, rotated, adjacent

    @cute.jit
    def __call__(self, qkv: cute.Pointer, cos: cute.Pointer, sin: cute.Pointer,
                 q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.heads * self.dim)
        self.kernel(qkv, cos, sin, q, k, v, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkv: cute.Pointer, cos: cute.Pointer, sin: cute.Pointer,
               q: cute.Pointer, k: cute.Pointer, v: cute.Pointer, count: Int64):
        qkv, cos, sin, q, k, v = _view(qkv), _view(cos), _view(sin), _view(q), _view(k), _view(v)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            width = Int64(self.heads * self.dim)
            row, column = at // width, at % width
            local = column % Int64(self.dim)
            base = row * Int64(3) * width + column
            v[at] = qkv[base + Int64(2) * width]
            if local < Int64(self.rotated):
                if cutlass.const_expr(self.adjacent):
                    pair = column - local + (local ^ Int64(1))
                    table = row * Int64(self.rotated // 2) + local // Int64(2)
                    negative = local % Int64(2) == Int64(0)
                else:
                    half = Int64(self.rotated // 2)
                    pair = column - local + (local + half) % Int64(self.rotated)
                    table = row * Int64(self.rotated // 2) + local % half
                    negative = local < half
                c, s = Float32(cos[table]), Float32(sin[table])
                for offset, out in ((Int64(0), q), (width, k)):
                    value = Float32(qkv[base + offset])
                    partner = Float32(qkv[row * Int64(3) * width + pair + offset])
                    second = _mul(partner, s)
                    if cutlass.const_expr(self.adjacent):
                        # Installed ATen complex64 mul SASS: FMUL(b,d)/FMUL(a,d),
                        # then FFMA(a,c,-bd)/FFMA(b,c,ad), one BF16 cast afterward.
                        if negative:
                            second = -second
                        result = _fma(value, c, second)
                    else:
                        first = _mul(value, c)
                        result = _add(first, second)
                        if negative:
                            result = _sub(first, second)
                    out[at] = result.to(BFloat16)
            else:
                q[at], k[at] = qkv[base], qkv[base + width]


def compile_vision_qkv_rotary_aot(*, heads, dim, rotated, adjacent=False):
    """Input [rows,3,heads,dim]; F32 tables have [rows,rotated/2] values.

    Adjacent-pair complex rotation is MoonViT. Split-half partial rotation is
    MiniMax (78 of 80 dims); its two unrotated tail values are copied exactly.
    Outputs must not alias the fused source. No live quantity enters the key.
    """
    if heads <= 0 or dim <= 0 or rotated <= 0 or rotated > dim or rotated % 2:
        raise ValueError("invalid vision QKV rotary geometry")
    if adjacent and rotated != dim:
        raise ValueError("adjacent-pair vision rotary requires the complete head")
    return compile_program(VisionQkvRotary(heads, dim, rotated, bool(adjacent)),
        name="vision_qkv_rotary", operands=(
            Operand("qkv", torch.bfloat16, f"[rows,3,{heads},{dim}]"),
            Operand("cos", torch.float32, f"[rows,{rotated // 2}]"),
            Operand("sin", torch.float32, f"[rows,{rotated // 2}]"),
            Operand("q", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
            Operand("k", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
            Operand("v", torch.bfloat16, f"[rows,{heads},{dim}]", "out")),
        scalars=(Scalar("rows"),), key=(heads, dim, rotated, bool(adjacent)),
        geometry=dict(heads=heads, dim=dim, rotated=rotated, adjacent=bool(adjacent)))
