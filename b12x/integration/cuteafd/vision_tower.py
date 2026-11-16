"""Checkpoint-native still-image tower primitives; grid sizes are launch data."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _sub, _fma, _view


class BilinearPosition:
    """Match the official compiled MoonViT divided_fixed interpolation.

    BF16 table loads promote to FP32. Half-pixel coordinates and width-first,
    height-second lerps use FP32 FMA contraction, followed by one BF16 store.
    The caller computes this once per image grid, never once per block.
    """
    def __init__(self, channels, table_height, table_width):
        self.channels = channels
        self.table_height, self.table_width = table_height, table_width

    @cute.jit
    def __call__(self, table: cute.Pointer, out: cute.Pointer,
                 height: Int32, width: Int32, stream: cuda.CUstream):
        count = Int64(height) * Int64(width) * Int64(self.channels)
        self.kernel(table, out, height, width, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, table: cute.Pointer, out: cute.Pointer,
               height: Int32, width: Int32, count: Int64):
        table, out = _view(table), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            row, channel = at // Int64(self.channels), at % Int64(self.channels)
            y, x = row // Int64(width), row % Int64(width)
            sy = _fma(_add(Float32(y), Float32(0.5)),
                      Float32(self.table_height) / Float32(height), Float32(-0.5))
            sx = _fma(_add(Float32(x), Float32(0.5)),
                      Float32(self.table_width) / Float32(width), Float32(-0.5))
            sy = cute.math.max(sy, Float32(0.0))
            sx = cute.math.max(sx, Float32(0.0))
            y0, x0 = Int64(sy), Int64(sx)
            y1 = cute.math.min(y0 + Int64(1), Int64(self.table_height - 1))
            x1 = cute.math.min(x0 + Int64(1), Int64(self.table_width - 1))
            wy = cute.math.min(cute.math.max(_sub(sy, Float32(y0)), Float32(0.0)), Float32(1.0))
            wx = cute.math.min(cute.math.max(_sub(sx, Float32(x0)), Float32(0.0)), Float32(1.0))
            a = Float32(table[(y0 * Int64(self.table_width) + x0) * Int64(self.channels) + channel])
            b = Float32(table[(y0 * Int64(self.table_width) + x1) * Int64(self.channels) + channel])
            c = Float32(table[(y1 * Int64(self.table_width) + x0) * Int64(self.channels) + channel])
            d = Float32(table[(y1 * Int64(self.table_width) + x1) * Int64(self.channels) + channel])
            top = _fma(_sub(b, a), wx, a)
            bottom = _fma(_sub(d, c), wx, c)
            out[at] = _fma(_sub(bottom, top), wy, top).to(BFloat16)


def compile_bilinear_position_aot(*, channels=1024, table_height=64, table_width=64):
    """Compile static learned-table geometry; output H/W never enter the key."""
    if min(channels, table_height, table_width) <= 0:
        raise ValueError("position table geometry must be positive")
    return compile_program(BilinearPosition(channels, table_height, table_width),
        name="vision_tower_bilinear_position", operands=(
            Operand("table", torch.bfloat16, f"[{table_height},{table_width},{channels}]"),
            Operand("out", torch.bfloat16, f"[height,width,{channels}]", "out")),
        scalars=(Scalar("height"), Scalar("width")),
        key=(channels, table_height, table_width),
        geometry=dict(channels=channels, table_height=table_height, table_width=table_width,
                      interpolation="compiled_half_pixel_width_then_height_fma"))
