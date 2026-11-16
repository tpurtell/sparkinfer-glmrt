"""F32 convolution prefill with explicit lag/slot pitches and no replay sink.

The history pointer is pre-offset to the desired layer and field. Packed
segment metadata follows the original convolution contract, with exclusive
slots and host-validated first/slots arrays. All row counts are runtime.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import Operand, Scalar, compile_program
from .inkling2_convolution_replay import _ConvolutionReplay


class _ConvolutionStrided(_ConvolutionReplay):
    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, dynamic: cute.Pointer,
                 history: cute.Pointer, first: cute.Pointer, slots: cute.Pointer,
                 out: cute.Pointer, rows: Int32, history_slot_stride_elements: Int64,
                 history_lag_stride_elements: Int64, stream: cuda.CUstream):
        if (rows > 0 and rows <= 4096 and history_lag_stride_elements >= self.width
                and history_slot_stride_elements >= (self.taps - 1) * history_lag_stride_elements):
            # Reuse the exact producer with the sink removed at compile time.
            self.convolve(x, weight, dynamic, history, first, slots, out, rows,
                          history_slot_stride_elements, history_lag_stride_elements,
                          out, Int64(0), False).launch(
                grid=(rows, (self.width + self.threads - 1) // self.threads, 1),
                block=(self.threads, 1, 1), stream=stream)
            self.commit(x, history, first, slots, rows, history_slot_stride_elements,
                        history_lag_stride_elements).launch(
                grid=(rows, (self.width + self.threads - 1) // self.threads, 1),
                block=(self.threads, 1, 1), stream=stream)


def compile_inkling2_convolution_strided_aot(*, width: int, taps: int = 4,
                                           heads: int = 1, dynamic: bool = False):
    """ABI: x, weight, dynamic, history, first, slots, out, rows:i32,
    history_slot_stride_elements:i64, history_lag_stride_elements:i64, stream.
    """
    if (type(width) is not int or width <= 0 or width > 11264
            or type(taps) is not int or taps != 4
            or type(heads) is not int or heads <= 0 or width % heads):
        raise ValueError("invalid strided convolution geometry")
    kind = "dynamic" if dynamic else "static"
    return compile_program(_ConvolutionStrided(width, taps, heads, dynamic),
        name=f"inkling2_{kind}_convolution_strided", operands=(
            Operand("x", torch.float32, f"[rows,{width}]"),
            Operand("weight", torch.bfloat16, f"[{taps},{width}]" if dynamic else f"[{width},{taps}]"),
            Operand("dynamic", torch.float32, f"[rows,{heads},{taps}]"),
            Operand("history", torch.float32, f"[slots,{taps-1},{width}]", "inout"),
            Operand("first", torch.int32, "[rows]", align=4),
            Operand("slots", torch.int64, "[rows]", align=8),
            Operand("out", torch.float32, f"[rows,{width}]", "out"),
        ), scalars=(Scalar("rows"), Scalar("history_slot_stride_elements", "int64"),
                    Scalar("history_lag_stride_elements", "int64")),
        key=(width, taps, heads, kind),
        geometry={"width": width, "taps": taps, "heads": heads,
                  "dynamic": bool(dynamic), "max_rows": 4096, "history": "newest_first"},
        doc=__doc__)
