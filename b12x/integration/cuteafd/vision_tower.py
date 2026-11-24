"""Checkpoint-native still-image tower primitives; grid sizes are launch data."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ._common import Operand, Scalar, compile_program
from .pointwise import _add, _sub, _mul, _fma, _view


@dsl_user_op
def _div_full(a: Float32, b: Float32, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip)], "div.full.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def _axis(coordinate: Int64, length: Int32, table_length: int):
    scale = _div_full(Float32(table_length), Float32(length))
    position = _fma(_add(Float32(coordinate), Float32(0.5)), scale, Float32(-0.5))
    position = cute.math.max(position, Float32(0.0))
    low = Int64(position)
    high = cute.math.min(low + Int64(1), Int64(table_length - 1))
    weight = cute.math.min(cute.math.max(_sub(position, Float32(low)), Float32(0.0)), Float32(1.0))
    return scale, position, low, high, weight


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
            _, _, y0, y1, wy = _axis(y, height, self.table_height)
            _, _, x0, x1, wx = _axis(x, width, self.table_width)
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


class PositionAxisProbe:
    """Probe the same axis helper used by the production interpolation kernel."""
    def __init__(self, table_length):
        self.table_length = table_length

    @cute.jit
    def __call__(self, floats: cute.Pointer, indices: cute.Pointer,
                 length: Int32, stream: cuda.CUstream):
        self.kernel(floats, indices, length).launch(
            grid=((length + Int32(255)) // Int32(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, floats: cute.Pointer, indices: cute.Pointer, length: Int32):
        floats, indices = _view(floats), _view(indices)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < Int64(length):
            scale, position, low, high, weight = _axis(at, length, self.table_length)
            floats[at * Int64(3)] = scale
            floats[at * Int64(3) + Int64(1)] = position
            floats[at * Int64(3) + Int64(2)] = weight
            indices[at * Int64(2)] = low
            indices[at * Int64(2) + Int64(1)] = high


def compile_position_axis_probe_aot(*, table_length=64):
    if table_length <= 0:
        raise ValueError("position axis table length must be positive")
    return compile_program(PositionAxisProbe(table_length), name="vision_tower_axis_probe",
        operands=(Operand("floats", torch.float32, "[length,3]", "out"),
                  Operand("indices", torch.int64, "[length,2]", "out")),
        scalars=(Scalar("length"),), key=(table_length,), geometry=dict(table_length=table_length))


class ScaledBilinearPosition:
    """PyTorch CUDA ``upsample_bilinear2d`` with an explicit ``scale_factor``.

    Hy4's SigLIP2 tower resizes its learned 128x128 grid with
    ``scale_factor=((h+0.1)/128, (w+0.1)/128)`` and ``align_corners=False``.
    The caller passes ATen's ``area_pixel_compute_scale`` result,
    ``float(1/scale_factor)``, per axis. The kernel mirrors ATen's NHWC frame:
    source index ``rscale*(dst+0.5)-0.5`` clamped at 0, truncating cast,
    ``+1`` neighbour only inside the table, then ``h0*(w0*a+w1*b)+h1*(w0*c+w1*d)``
    on F32-widened BF16 values, one BF16 store (the caller's ``.bfloat16()``).
    """
    def __init__(self, channels, table_height, table_width, contraction="aten"):
        self.channels = channels
        self.table_height, self.table_width = table_height, table_width
        self.contraction = contraction

    @cute.jit
    def __call__(self, table: cute.Pointer, out: cute.Pointer, height: Int32, width: Int32,
                 scale_h: Float32, scale_w: Float32, stream: cuda.CUstream):
        count = Int64(height) * Int64(width) * Int64(self.channels)
        self.kernel(table, out, width, scale_h, scale_w, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def _source(self, rscale: Float32, dst: Int64, length: int):
        src = _fma(rscale, _add(Float32(dst), Float32(0.5)), Float32(-0.5))
        if cutlass.const_expr(self.contraction == "unfused"):
            src = _sub(_mul(rscale, _add(Float32(dst), Float32(0.5))), Float32(0.5))
        src = cute.math.max(src, Float32(0.0))
        low = Int64(src)
        step = cute.math.min(low + Int64(1), Int64(length - 1)) - low
        lam1 = _sub(src, Float32(low))
        lam0 = _sub(Float32(1.0), lam1)
        return low, step, lam0, lam1

    @cute.kernel
    def kernel(self, table: cute.Pointer, out: cute.Pointer, width: Int32,
               scale_h: Float32, scale_w: Float32, count: Int64):
        table, out = _view(table), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            channel = at % Int64(self.channels)
            pixel = at // Int64(self.channels)
            y, x = pixel // Int64(width), pixel % Int64(width)
            y0, dy, h0, h1 = self._source(scale_h, y, self.table_height)
            x0, dx, w0, w1 = self._source(scale_w, x, self.table_width)
            row = Int64(self.table_width * self.channels)
            base = (y0 * Int64(self.table_width) + x0) * Int64(self.channels) + channel
            a = Float32(table[base])
            b = Float32(table[base + dx * Int64(self.channels)])
            c = Float32(table[base + dy * row])
            d = Float32(table[base + dy * row + dx * Int64(self.channels)])
            if cutlass.const_expr(self.contraction == "unfused"):
                top = _add(_mul(w0, a), _mul(w1, b))
                bottom = _add(_mul(w0, c), _mul(w1, d))
                value = _add(_mul(h0, top), _mul(h1, bottom))
            else:
                # nvcc -fmad contraction of ATen's expression: the left product
                # of each sum folds into an FMA with the right product.
                top = _fma(w0, a, _mul(w1, b))
                bottom = _fma(w0, c, _mul(w1, d))
                value = _fma(h0, top, _mul(h1, bottom))
            out[at] = value.to(BFloat16)


def compile_scaled_bilinear_position_aot(*, channels=1152, table_height=128, table_width=128,
                                         contraction="aten"):
    """Static table geometry only; output H/W and both scales are launch data.

    ``table`` is the learned grid without its class row (callers pass the row-1
    offset), BF16 ``[table_height, table_width, channels]``.
    """
    if min(channels, table_height, table_width) <= 0:
        raise ValueError("position table geometry must be positive")
    if contraction not in ("aten", "unfused"):
        raise ValueError("unknown bilinear contraction")
    key = (channels, table_height, table_width) + ((contraction,) if contraction != "aten" else ())
    return compile_program(ScaledBilinearPosition(channels, table_height, table_width, contraction),
        name="vision_tower_scaled_bilinear_position", operands=(
            Operand("table", torch.bfloat16, f"[{table_height},{table_width},{channels}]"),
            Operand("out", torch.bfloat16, f"[height*width,{channels}]", "out")),
        scalars=(Scalar("height"), Scalar("width"), Scalar("scale_h", "float32"), Scalar("scale_w", "float32")),
        key=key,
        geometry=dict(channels=channels, table_height=table_height, table_width=table_width,
                      interpolation="aten_upsample_bilinear2d_scale_factor", contraction=contraction))


class RowMarkers:
    """Append one checkpoint marker row after every merged grid row."""
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, x: cute.Pointer, marker: cute.Pointer, out: cute.Pointer,
                 height: Int32, width: Int32, stream: cuda.CUstream):
        count = Int64(height) * Int64(width + Int32(1)) * Int64(self.width)
        self.kernel(x, marker, out, width, count).launch(
            grid=((count + Int64(255)) // Int64(256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, marker: cute.Pointer, out: cute.Pointer, width: Int32, count: Int64):
        x, marker, out = _view(x), _view(marker), _view(out)
        at = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if at < count:
            channel = at % Int64(self.width)
            slot = at // Int64(self.width)
            columns = Int64(width) + Int64(1)
            row, column = slot // columns, slot % columns
            if column == Int64(width):
                out[at] = marker[channel]
            else:
                out[at] = x[(row * Int64(width) + column) * Int64(self.width) + channel]


def compile_row_markers_aot(*, width):
    """[h*w,width] merged rows to [h*(w+1),width] with a marker closing each row."""
    if width <= 0:
        raise ValueError("row marker width must be positive")
    return compile_program(RowMarkers(width), name="vision_tower_row_markers", operands=(
            Operand("x", torch.bfloat16, f"[height*width,{width}]"),
            Operand("marker", torch.bfloat16, f"[{width}]"),
            Operand("out", torch.bfloat16, f"[height*(width+1),{width}]", "out")),
        scalars=(Scalar("height"), Scalar("width")), key=(width,), geometry=dict(width=width))
