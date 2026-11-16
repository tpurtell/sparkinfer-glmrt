"""Perception-Encoder BF16 boundaries; all image extents are runtime data.

Inkling's layer scale rounds its product before the residual add. Step-5 uses
quick GELU and two dense 3x3 stride-2 convolutions; convolution lowering writes
caller-owned im2col scratch for the existing BF16 projection implementation.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import Operand, Scalar, compile_program
from b12x._lib.intrinsics import div_rn_f32
from .pointwise import _add, _fma, _mul, _sub, _view


class ScaledResidual:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, x: cute.Pointer, delta: cute.Pointer, gamma: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.width)
        self.kernel(x, delta, gamma, out, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, delta: cute.Pointer, gamma: cute.Pointer,
               out: cute.Pointer, count: Int64):
        x, delta, gamma, out = _view(x), _view(delta), _view(gamma), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            product = _mul(Float32(delta[at]), Float32(gamma[at % Int64(self.width)]))
            rounded = Float32(product.to(BFloat16))
            out[at] = _add(Float32(x[at]), rounded).to(BFloat16)


class QuickGelu:
    def __init__(self, width, bf16_intermediates):
        self.width, self.bf16_intermediates = width, bf16_intermediates

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
            scaled = _mul(value, Float32(1.702))
            if cutlass.const_expr(self.bf16_intermediates):
                scaled = Float32(scaled.to(BFloat16))
            denominator = _add(Float32(1.0), cute.math.exp(-scaled, fastmath=False))
            if cutlass.const_expr(self.bf16_intermediates):
                sigmoid = Float32(div_rn_f32(Float32(1.0),denominator).to(BFloat16))
                out[at] = _mul(value,sigmoid).to(BFloat16)
            else:
                out[at] = div_rn_f32(value,denominator).to(BFloat16)


class Rotary2D:
    def __init__(self, heads, dim, class_token):
        self.heads, self.dim, self.class_token = heads, dim, class_token

    @cute.jit
    def __call__(self, qkv: cute.Pointer, inv_freq: cute.Pointer, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 rows: Int32, grid_width: Int32, stream: cuda.CUstream):
        count = Int64(rows) * Int64(self.heads * self.dim)
        self.kernel(qkv, inv_freq, q, k, v, count, grid_width).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkv: cute.Pointer, inv_freq: cute.Pointer, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
               count: Int64, grid_width: Int32):
        qkv, inv_freq, q, k, v = _view(qkv), _view(inv_freq), _view(q), _view(k), _view(v)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            row = at // Int64(self.heads * self.dim)
            col = at % Int64(self.heads * self.dim)
            dim = col % Int64(self.dim)
            patch = row - Int64(self.class_token)
            position = patch % Int64(grid_width) + Int64(self.class_token)
            if dim >= Int64(self.dim // 2):
                position = patch // Int64(grid_width) + Int64(self.class_token)
            if cutlass.const_expr(self.class_token):
                if row == Int64(0):
                    position = Int64(0)
            pair_index = (dim % Int64(self.dim // 2)) // Int64(2)
            angle = _mul(Float32(position), Float32(inv_freq[pair_index]))
            cosine = cute.math.cos(angle, fastmath=False)
            sine = cute.math.sin(angle, fastmath=False)
            pair = col + Int64(1)
            sign = Float32(-1.0)
            if dim % Int64(2) != Int64(0):
                pair = col - Int64(1)
                sign = Float32(1.0)
            base = row * Int64(3 * self.heads * self.dim)
            for offset, out in ((0, q), (self.heads * self.dim, k)):
                value = Float32(qkv[base + Int64(offset) + col])
                partner = Float32(qkv[base + Int64(offset) + pair])
                b = _mul(_mul(sign, partner), sine)
                if cutlass.const_expr(self.class_token):
                    # Inkling's pinned Triton _rotate_pair explicitly uses FMA.
                    out[at] = _fma(value,cosine,b).to(BFloat16)
                else:
                    out[at] = _add(_mul(value,cosine),b).to(BFloat16)
            v[at] = qkv[base + Int64(2 * self.heads * self.dim) + col]


class Conv3Stride2Lower:
    def __init__(self, channels):
        self.channels = channels

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, height: Int32, width: Int32,
                 stream: cuda.CUstream):
        oh, ow = (height + Int32(1)) // Int32(2), (width + Int32(1)) // Int32(2)
        count = Int64(oh) * Int64(ow) * Int64(self.channels * 9)
        self.kernel(x, out, height, width, ow, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, height: Int32, width: Int32,
               out_width: Int32, count: Int64):
        x, out = _view(x), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            col = at % Int64(self.channels * 9)
            row = at // Int64(self.channels * 9)
            y = (row // Int64(out_width)) * Int64(2) - Int64(1) + (col % Int64(9)) // Int64(3)
            w = (row % Int64(out_width)) * Int64(2) - Int64(1) + col % Int64(3)
            value = BFloat16(0.0)
            if y >= Int64(0) and y < Int64(height) and w >= Int64(0) and w < Int64(width):
                value = x[(y * Int64(width) + w) * Int64(self.channels) + col // Int64(9)]
            out[at] = value


class Merge2Program:
    """Checked Python launch facade; native callers validate before the raw ABI."""
    def __init__(self, program):
        self.program = program

    def __getattr__(self, name):
        return getattr(self.program,name)

    def launch(self, *tensors, scalars, stream=None):
        if len(scalars) != 2 or min(scalars) <= 0 or any(value%2 for value in scalars):
            raise ValueError("PE merge2 requires positive even height and width")
        return self.program.launch(*tensors,scalars=scalars,stream=stream)


class Merge2:
    def __init__(self, width, class_token):
        self.width, self.class_token = width, class_token

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, height: Int32, width: Int32,
                 stream: cuda.CUstream):
        count = Int64(height) * Int64(width) * Int64(self.width)
        self.kernel(x, out, width, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, width: Int32, count: Int64):
        x, out = _view(x), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            row = at // Int64(4 * self.width)
            col = at % Int64(4 * self.width)
            cell, channel = col // Int64(self.width), col % Int64(self.width)
            y = row // Int64(width // Int32(2)) * Int64(2) + cell // Int64(2)
            w = row % Int64(width // Int32(2)) * Int64(2) + cell % Int64(2)
            src = (y * Int64(width) + w + Int64(self.class_token)) * Int64(self.width) + channel
            out[at] = x[src]


def compile_scaled_residual_aot(*, width):
    if width <= 0:
        raise ValueError("layer-scale width must be positive")
    return compile_program(ScaledResidual(width), name="perception_scaled_residual", operands=(
        Operand("x", torch.bfloat16, f"[rows,{width}]"),
        Operand("delta", torch.bfloat16, f"[rows,{width}]"),
        Operand("gamma", torch.bfloat16, f"[{width}]"),
        Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,), geometry=dict(width=width, bf16_product=True))


def compile_quick_gelu_aot(*, width, bf16_intermediates=False):
    """Fused vendor CUDA math by default; explicit eager-Torch rounding variant."""
    if width <= 0:
        raise ValueError("quick-GELU width must be positive")
    rounded = bool(bf16_intermediates)
    return compile_program(QuickGelu(width, rounded), name="perception_quick_gelu", operands=(
        Operand("x", torch.bfloat16, f"[rows,{width}]"),
        Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width,rounded),
        geometry=dict(width=width, activation="quick_gelu", bf16_intermediates=rounded),version=2)


def compile_rotary_2d_aot(*, heads, dim, class_token=False):
    if min(heads, dim) <= 0 or dim % 4:
        raise ValueError("PE rotary requires positive heads and dimension divisible by four")
    return compile_program(Rotary2D(heads, dim, bool(class_token)), name="perception_rotary_2d", operands=(
        Operand("qkv", torch.bfloat16, f"[rows,{3*heads*dim}]"),
        Operand("inv_freq", torch.float32, f"[{dim//4}]"),
        Operand("q", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
        Operand("k", torch.bfloat16, f"[rows,{heads},{dim}]", "out"),
        Operand("v", torch.bfloat16, f"[rows,{heads},{dim}]", "out")),
        scalars=(Scalar("rows"), Scalar("grid_width")), key=(heads,dim,bool(class_token)),
        geometry=dict(heads=heads,dim=dim,class_token=bool(class_token),pairing="adjacent",axes="x_y"),version=2)


def compile_conv3_stride2_lower_aot(*, channels):
    if channels <= 0:
        raise ValueError("convolution channels must be positive")
    return compile_program(Conv3Stride2Lower(channels), name="perception_conv3_stride2_lower", operands=(
        Operand("x", torch.bfloat16, f"[height,width,{channels}]"),
        Operand("out", torch.bfloat16, f"[ceil(height/2)*ceil(width/2),{channels*9}]", "out")),
        scalars=(Scalar("height"), Scalar("width")), key=(channels,),
        geometry=dict(channels=channels,kernel=3,stride=2,padding=1,layout="channel_kernel"))


def compile_merge2_aot(*, width, class_token=False):
    if width <= 0:
        raise ValueError("merge width must be positive")
    return Merge2Program(compile_program(Merge2(width, bool(class_token)), name="perception_merge2", operands=(
        Operand("x", torch.bfloat16, f"[height*width+{int(class_token)},{width}]"),
        Operand("out", torch.bfloat16, f"[height*width/4,{width*4}]", "out")),
        scalars=(Scalar("height"), Scalar("width")), key=(width,bool(class_token)),
        geometry=dict(width=width,class_token=bool(class_token),layout="spatial_channel")))
