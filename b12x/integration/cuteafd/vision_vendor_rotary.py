"""SGLang fused-FMA vision RoPE; distinct from ATen complex64 arithmetic."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import Operand, Scalar, compile_program
from .pointwise import _fma, _mul, _view


class VendorVisionQkvRotary:
    def __init__(self, heads, dim):
        self.heads, self.dim = heads, dim

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
            pair = column - local + (local ^ Int64(1))
            table = row * Int64(self.dim // 2) + local // Int64(2)
            c, s = Float32(cos[table]), Float32(sin[table])
            v[at] = qkv[base + Int64(2) * width]
            for offset, out in ((Int64(0), q), (width, k)):
                value = Float32(qkv[base + offset])
                partner = Float32(qkv[row * Int64(3) * width + pair + offset])
                first = _mul(value, c)
                if local % Int64(2) == Int64(0):
                    partner = -partner
                # Vendor vision_rope.py rounds value*cos, then fuses partner*sin
                # with it. ATen complex64 instead fuses value*cos; keep distinct.
                out[at] = _fma(partner, s, first).to(BFloat16)


def compile_vendor_vision_qkv_rotary_aot(*, heads, dim):
    """SGLang a9bf8c76 fused complex RoPE, BF16 [rows,3,heads,dim]."""
    if heads <= 0 or dim <= 0 or dim % 2:
        raise ValueError("invalid vendor vision rotary geometry")
    return compile_program(VendorVisionQkvRotary(heads, dim),
        name="vision_vendor_qkv_rotary", operands=(
            Operand("qkv", torch.bfloat16, f"[rows,3,{heads},{dim}]"),
            Operand("cos", torch.float32, f"[rows,{dim // 2}]"),
            Operand("sin", torch.float32, f"[rows,{dim // 2}]"),
            Operand("q", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
            Operand("k", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
            Operand("v", torch.bfloat16, f"[rows,{heads},{dim}]", "out")),
        scalars=(Scalar("rows"),), key=(heads, dim),
        geometry=dict(heads=heads, dim=dim, arithmetic="sglang_fused_fma"))
