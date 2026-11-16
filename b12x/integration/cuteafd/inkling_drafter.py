"""Inkling 2's three allocation-free MTP companion programs.

Pinned math: WoosukKwon/vllm d664cd1e, inkling2/ops/dspark.py and
inkling2/dspark.py. All arrays are contiguous, all outputs are disjoint from
inputs. The caller normalizes anchor embeddings and precomputes BF16 alpha
from FP32 softmax. Linear projections and biases belong to the existing
linear producers, not the recurrent gate. Compile before graph capture.
Live row counts never enter compilation keys; out-of-range calls do nothing.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cuda.bindings import driver as cuda

from b12x._lib.intrinsics import (
    block_reduce, cvt_f32_to_bf16_bits, div_rn_f32, u32_as_f32, warp_reduce,
)
from ._common import Operand, Scalar, compile_program

_HIDDEN = 4608
_TAPS = 8
_RANK = 256
_THREADS = 256


def _add(a, b):
    return a + b


@cute.jit
def _round_bf16(value: Float32) -> Float32:
    # An explicit conversion instruction keeps every Torch BF16 op boundary.
    return u32_as_f32(cvt_f32_to_bf16_bits(value) << 16)


class _Mix:
    @cute.jit
    def __call__(self, features: cute.Pointer, alpha: cute.Pointer,
                 gamma: cute.Pointer, mixed: cute.Pointer, rows: Int32,
                 eps: Float32, stream: cuda.CUstream):
        if rows > 0 and rows <= 4096:
            self.kernel(features, alpha, gamma, mixed, eps).launch(
                grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, features: cute.Pointer, alpha: cute.Pointer,
               gamma: cute.Pointer, mixed: cute.Pointer, eps: Float32):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        base = Int64(row) * Int64(_TAPS * _HIDDEN)
        values = cute.make_rmem_tensor(cute.make_layout(18), Float32)
        squares = Float32(0)
        for item in cutlass.range_constexpr(18):
            col = Int32(thread) + Int32(item * _THREADS)
            value = Float32(0)
            for tap in cutlass.range_constexpr(_TAPS):
                value += (Float32(features[base + Int64(tap * _HIDDEN) + Int64(col)])
                          * Float32(alpha[tap]))
            values[item] = value
            squares += value * value
        allocator = cutlass.utils.SmemAllocator()
        scratch = allocator.allocate_tensor(Float32, cute.make_layout((1, 8)), byte_alignment=16)
        inverse = allocator.allocate_tensor(Float32, cute.make_layout(1), byte_alignment=4)
        total = block_reduce(warp_reduce(squares, _add), _add, scratch, Float32(0))
        if thread == 0:
            inverse[0] = cute.math.rsqrt(total / Float32(_HIDDEN) + eps, fastmath=True)
        cute.arch.sync_threads()
        inv = inverse[0]
        for item in cutlass.range_constexpr(18):
            col = Int32(thread) + Int32(item * _THREADS)
            mixed[Int64(row) * Int64(_HIDDEN) + Int64(col)] = BFloat16(
                values[item] * inv * Float32(gamma[col]))


class _MaskInit:
    @cute.jit
    def __call__(self, normalized_anchors: cute.Pointer, mask: cute.Pointer,
                 block: cute.Pointer, sequences: Int32, stream: cuda.CUstream):
        if sequences > 0 and sequences <= 64:
            self.kernel(normalized_anchors, mask, block).launch(
                grid=(sequences, 8, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, normalized_anchors: cute.Pointer, mask: cute.Pointer,
               block: cute.Pointer):
        seq, pos, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        for item in cutlass.range_constexpr(18):
            col = Int64(thread) + Int64(item * _THREADS)
            value = mask[col]
            if pos == 0:
                value = normalized_anchors[Int64(seq) * Int64(_HIDDEN) + col]
            block[(Int64(seq) * Int64(8) + Int64(pos)) * Int64(_HIDDEN) + col] = value


class _Gate:
    @cute.jit
    def __call__(self, raw: cute.Pointer, state: cute.Pointer,
                 new_state: cute.Pointer, latent: cute.Pointer,
                 sequences: Int32, stream: cuda.CUstream):
        if sequences > 0 and sequences <= 64:
            self.kernel(raw, state, new_state, latent).launch(
                grid=(sequences, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, raw: cute.Pointer, state: cute.Pointer,
               new_state: cute.Pointer, latent: cute.Pointer):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        offset = Int64(row) * Int64(3 * _RANK) + Int64(thread)
        gate = _round_bf16(Float32(raw[offset]))
        candidate = _round_bf16(Float32(raw[offset + _RANK]))
        out = _round_bf16(Float32(raw[offset + 2 * _RANK]))
        gate = _round_bf16(div_rn_f32(Float32(1), Float32(1) + cute.math.exp(-gate, fastmath=False)))
        candidate = _round_bf16(cute.math.tanh(candidate, fastmath=False))
        index = Int64(row) * Int64(_RANK) + Int64(thread)
        keep = _round_bf16(gate * Float32(state[index]))
        update = _round_bf16(_round_bf16(Float32(1) - gate) * candidate)
        new_state[index] = BFloat16(keep + update)
        latent[index] = BFloat16(cute.math.tanh(out, fastmath=False))


def compile_inkling_dflare_mix_aot():
    """(features, alpha, gamma, mixed, rows:i32, eps:f32, stream)."""
    return compile_program(_Mix(), name="inkling_dflare_mix", operands=(
        Operand("features", torch.bfloat16, "[rows,8,4608]"),
        Operand("alpha", torch.bfloat16, "[8]"),
        Operand("gamma", torch.bfloat16, "[4608]"),
        Operand("mixed", torch.bfloat16, "[rows,4608]", "out"),
    ), scalars=(Scalar("rows"), Scalar("eps", "float32")),
        key=(_HIDDEN, _TAPS), geometry={"hidden": _HIDDEN, "taps": _TAPS}, doc=__doc__)


def compile_inkling_mask_init_aot():
    """(normalized_anchors, mask, block, sequences:i32, stream)."""
    return compile_program(_MaskInit(), name="inkling_mask_init", operands=(
        Operand("normalized_anchors", torch.bfloat16, "[sequences,4608]"),
        Operand("mask", torch.bfloat16, "[4608]"),
        Operand("block", torch.bfloat16, "[sequences,8,4608]", "out"),
    ), scalars=(Scalar("sequences"),), key=(_HIDDEN, 8),
        geometry={"hidden": _HIDDEN, "block": 8}, doc=__doc__)


def compile_inkling_recurrent_gate_aot():
    """(raw, state, new_state, latent, sequences:i32, stream)."""
    return compile_program(_Gate(), name="inkling_recurrent_gate", operands=(
        Operand("raw", torch.float32, "[sequences,768]"),
        Operand("state", torch.bfloat16, "[sequences,256]"),
        Operand("new_state", torch.bfloat16, "[sequences,256]", "out"),
        Operand("latent", torch.bfloat16, "[sequences,256]", "out"),
    ), scalars=(Scalar("sequences"),), key=(_RANK,),
        geometry={"rank": _RANK}, doc=__doc__)
