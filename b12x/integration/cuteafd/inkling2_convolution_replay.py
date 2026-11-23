"""Optional F32 convolution producer with an exact replay sink.

History and replay pointers are already offset to the desired layer/field.
Their lag/slot/row pitches are runtime element counts, supporting padded
lag-major banks and TP widths. Sources, outputs, replay and history are disjoint.
The existing producer module and exports are deliberately unchanged.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from ._common import Operand, Scalar, compile_program
from .inkling2_convolution import _mul_rn


class _ConvolutionReplay:
    threads = 256

    def __init__(self, width, taps, heads, dynamic, max_rows=64):
        self.width, self.taps, self.heads = int(width), int(taps), int(heads)
        self.dynamic = bool(dynamic)
        self.max_rows = int(max_rows)
        self.factor = (self.taps * (2 if self.dynamic else 1) + 1) ** -0.5

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, dynamic: cute.Pointer,
                 history: cute.Pointer, first: cute.Pointer, slots: cute.Pointer,
                 out: cute.Pointer, replay: cute.Pointer, rows: Int32, history_slot_stride_elements: Int64, history_lag_stride_elements: Int64,
                 replay_row_stride_elements: Int64, stream: cuda.CUstream):
        if (rows > 0 and rows <= self.max_rows and history_lag_stride_elements >= self.width
                and history_slot_stride_elements >= (self.taps - 1) * history_lag_stride_elements
                and replay_row_stride_elements >= self.width):
            self.convolve(x, weight, dynamic, history, first, slots, out, rows,
                          history_slot_stride_elements, history_lag_stride_elements,
                          replay, replay_row_stride_elements).launch(
                grid=(rows, (self.width + self.threads - 1) // self.threads, 1),
                block=(self.threads, 1, 1), stream=stream)
            self.commit(x, history, first, slots, rows, history_slot_stride_elements,
                        history_lag_stride_elements).launch(
                grid=(rows, (self.width + self.threads - 1) // self.threads, 1),
                block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def convolve(self, x: cute.Pointer, weight: cute.Pointer, dynamic: cute.Pointer,
                 history: cute.Pointer, first: cute.Pointer, slots: cute.Pointer,
                 out: cute.Pointer, rows: Int32, history_slot_stride_elements: Int64, history_lag_stride_elements: Int64,
                 replay: cute.Pointer, replay_row_stride_elements: Int64,
                 replay_sink: cutlass.Constexpr = True):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if row < Int64(rows) and col < Int64(self.width):
            if cutlass.const_expr(replay_sink):
                # Capture the source bits before workspace reuse; absent in prefill.
                cute.recast_ptr(replay, dtype=Uint32)[row * replay_row_stride_elements + col] = (
                    cute.recast_ptr(x, dtype=Uint32)[row * Int64(self.width) + col])
            begin = Int64(first[row])
            slot = Int64(slots[row])
            acc = Float32(0)
            for lag in cutlass.range_constexpr(self.taps):
                value = Float32(0)
                if row - Int64(lag) >= begin:
                    value = Float32(x[(row - Int64(lag)) * Int64(self.width) + col])
                else:
                    prior = Int64(lag) - (row - begin) - Int64(1)
                    value = Float32(history[slot * history_slot_stride_elements + prior * history_lag_stride_elements + col])
                tap = Float32(0)
                if cutlass.const_expr(self.dynamic):
                    head = col // Int64(self.width // self.heads)
                    tap = Float32(weight[Int64(lag * self.width) + col]) + Float32(
                        dynamic[(row * Int64(self.heads) + head) * Int64(self.taps) + Int64(lag)])
                else:
                    tap = Float32(weight[col * Int64(self.taps) + Int64(self.taps - 1 - lag)])
                acc = acc + _mul_rn(value, tap)
            out[row * Int64(self.width) + col] = (acc + Float32(x[row * Int64(self.width) + col])) * Float32(self.factor)

    @cute.kernel
    def commit(self, x: cute.Pointer, history: cute.Pointer, first: cute.Pointer,
               slots: cute.Pointer, rows: Int32, history_slot_stride_elements: Int64, history_lag_stride_elements: Int64):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if row < Int64(rows) and col < Int64(self.width):
            last = row + Int64(1) == Int64(rows)
            if row + Int64(1) < Int64(rows):
                last = Int64(first[row + Int64(1)]) != Int64(first[row])
            if last:
                begin, slot = Int64(first[row]), Int64(slots[row])
                # Descending lags preserve old inputs when a chunk is shorter
                # than the history: each shifted read precedes its overwrite.
                for rev in cutlass.range_constexpr(self.taps - 1):
                    lag = self.taps - 2 - rev
                    value = Float32(0)
                    if row - Int64(lag) >= begin:
                        value = Float32(x[(row - Int64(lag)) * Int64(self.width) + col])
                    else:
                        prior = Int64(lag) - (row - begin) - Int64(1)
                        value = Float32(history[slot * history_slot_stride_elements + prior * history_lag_stride_elements + col])
                    history[slot * history_slot_stride_elements + Int64(lag) * history_lag_stride_elements + col] = value



def compile_inkling2_convolution_replay_aot(*, width: int, taps: int = 4,
                                          heads: int = 1, dynamic: bool = False,
                                          max_rows: int = 64):
    """F32 producer; capture source rows in the same convolve launch.

    ABI: x, weight, dynamic, history, first, slots, out, replay, rows:i32,
    history_slot_stride_elements:i64, history_lag_stride_elements:i64,
    replay_row_stride_elements:i64, stream.
    """
    if (type(width) is not int or width <= 0 or width > 11264
            or type(taps) is not int or taps != 4
            or type(heads) is not int or heads <= 0 or width % heads
            or type(max_rows) is not int or max_rows <= 0 or max_rows > 256):
        raise ValueError("invalid FIR replay convolution geometry")
    kind = "dynamic" if dynamic else "static"
    key = (width, taps, heads, kind) if max_rows == 64 else (width, taps, heads, kind, max_rows)
    return compile_program(_ConvolutionReplay(width, taps, heads, dynamic, max_rows=max_rows),
        name=f"inkling2_{kind}_convolution_replay", operands=(
            Operand("x", torch.float32, f"[rows,{width}]"),
            Operand("weight", torch.bfloat16, f"[{taps},{width}]" if dynamic else f"[{width},{taps}]"),
            Operand("dynamic", torch.float32, f"[rows,{heads},{taps}]"),
            Operand("history", torch.float32, f"[slots,{taps-1},{width}]", "inout"),
            Operand("first", torch.int32, "[rows]", align=4),
            Operand("slots", torch.int64, "[rows]", align=8),
            Operand("out", torch.float32, f"[rows,{width}]", "out"),
            Operand("replay", torch.float32, f"[{max_rows},{width}]", "out"),
        ), scalars=(Scalar("rows"), Scalar("history_slot_stride_elements", "int64"),
                    Scalar("history_lag_stride_elements", "int64"),
                    Scalar("replay_row_stride_elements", "int64")),
        key=key,
        geometry={"width": width, "taps": taps, "heads": heads,
                  "dynamic": bool(dynamic), "max_rows": max_rows, "history": "newest_first"},
        doc=__doc__)
