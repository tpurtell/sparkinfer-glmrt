"""Exact accepted-prefix commit for 48-layer, newest-first lag-3 FIR banks.

Pointers are disjoint F32 backings, but payload moves use U32 exclusively.
The host validates unique slots and packed segments before launch. Invalid
rows (slot/start/count) are independently skipped without touching live.
All strides are runtime element counts; neither TP widths nor layer padding
are compiled in. Replay has capacity 64 rows and accepted counts are 0..8.
"""
from __future__ import annotations

import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program


class _HistoryCommit:
    @cute.jit
    def __call__(self, checkpoint: cute.Pointer, replay: cute.Pointer,
                 table: cute.Pointer, live: cute.Pointer, sequences: Int32,
                 slots: Int32, channel_width: Int32,
                 history_lag_stride_elements: Int64,
                 history_layer_stride_elements: Int64,
                 history_slot_stride_elements: Int64,
                 replay_row_stride_elements: Int64,
                 replay_layer_stride_elements: Int64, stream: cuda.CUstream):
        if (sequences > 0 and sequences <= 64 and slots > 0
                and channel_width > 0 and channel_width <= 11264
                and history_lag_stride_elements >= Int64(channel_width)
                and history_layer_stride_elements >= 3 * history_lag_stride_elements
                and history_slot_stride_elements >= 48 * history_layer_stride_elements
                and replay_row_stride_elements >= Int64(channel_width)
                and replay_layer_stride_elements >= 64 * replay_row_stride_elements):
            self.kernel(checkpoint, replay, table, live, sequences, slots, channel_width,
                        history_lag_stride_elements, history_layer_stride_elements,
                        history_slot_stride_elements, replay_row_stride_elements,
                        replay_layer_stride_elements).launch(
                grid=((channel_width + 255) // 256, 48 * 3, sequences),
                block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, checkpoint: cute.Pointer, replay: cute.Pointer,
               table: cute.Pointer, live: cute.Pointer, sequences: Int32,
               slots: Int32, channel_width: Int32, lag_stride: Int64,
               layer_stride: Int64, slot_stride: Int64,
               replay_row_stride: Int64, replay_layer_stride: Int64):
        tile, layer_lag, seq = cute.arch.block_idx()
        col = Int64(tile) * 256 + Int64(cute.arch.thread_idx()[0])
        slot = Int64(table[Int64(seq)])
        first = Int64(table[Int64(sequences) + Int64(seq)])
        keep = Int64(table[2 * Int64(sequences) + Int64(seq)])
        if (col < Int64(channel_width) and slot >= 0 and slot < Int64(slots)
                and first >= 0 and first <= 64 and keep >= 0 and keep <= 8
                and first + keep <= 64):
            layer, lag = Int64(layer_lag) // 3, Int64(layer_lag) % 3
            base = slot * slot_stride + layer * layer_stride + col
            source = cute.recast_ptr(checkpoint, dtype=Uint32)
            value = Uint32(0)
            if keep > lag:
                source = cute.recast_ptr(replay, dtype=Uint32)
                value = Uint32(source[layer * replay_layer_stride + (first + keep - 1 - lag) * replay_row_stride + col])
            else:
                value = Uint32(source[base + (lag - keep) * lag_stride])
            cute.recast_ptr(live, dtype=Uint32)[base + lag * lag_stride] = value


def compile_fir_history_commit_aot():
    """Compile once; widths, counts and all backing pitches are runtime scalars."""
    return compile_program(_HistoryCommit(), name="fir_history_commit", operands=(
        Operand("checkpoint", torch.float32, "[slots,48,3,channel_width]"),
        Operand("replay", torch.float32, "[48,64,channel_width]"),
        Operand("table", torch.int32, "[3,sequences]", align=4),
        Operand("live", torch.float32, "[slots,48,3,channel_width]", "out"),
    ), scalars=(Scalar("sequences"), Scalar("slots"), Scalar("channel_width"),
        Scalar("history_lag_stride_elements", "int64"),
        Scalar("history_layer_stride_elements", "int64"),
        Scalar("history_slot_stride_elements", "int64"),
        Scalar("replay_row_stride_elements", "int64"),
        Scalar("replay_layer_stride_elements", "int64")),
        key=(), geometry={"layers": 48, "lags": 3, "max_width": 11264,
                          "max_rows": 64, "max_sequences": 64, "max_accepted": 8},
        doc=__doc__)
