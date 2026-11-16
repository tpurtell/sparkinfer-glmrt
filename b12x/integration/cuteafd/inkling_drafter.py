"""Inkling 2's allocation-free MTP companion programs.

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
from cutlass import BFloat16, Float32, Float64, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
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


@dsl_user_op
def _narrow_double(value: Float64, *, loc=None, ip=None) -> Float32:
    """Direct FP64 -> BF16 RNE: FP32 as an intermediate can double-round."""
    return Float32(llvm.inline_asm(T.f32(), [Float64(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 b; cvt.rn.bf16.f64 b, $1; cvt.f32.bf16 $0, b; }", "=f,d",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


class _HeadFold:
    @cute.jit
    def __call__(self, lm_head: cute.Pointer, hidden_out: cute.Pointer,
                 hidden_bias: cute.Pointer, folded_weight: cute.Pointer,
                 folded_bias: cute.Pointer, first_row: Int32, rows: Int32,
                 stream: cuda.CUstream):
        if rows > 0 and rows <= 4096 and first_row >= 0 and first_row <= 201024 - rows:
            self.kernel(lm_head, hidden_out, hidden_bias, folded_weight, folded_bias,
                        first_row, rows).launch(grid=((rows + 3) // 4, 1, 1),
                                               block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, lm_head: cute.Pointer, hidden_out: cute.Pointer,
               hidden_bias: cute.Pointer, folded_weight: cute.Pointer,
               folded_bias: cute.Pointer, first_row: Int32, rows: Int32):
        tile, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        allocator = cutlass.utils.SmemAllocator()
        head = allocator.allocate_tensor(BFloat16, cute.make_layout((4, 64)), byte_alignment=16)
        sums = cute.make_rmem_tensor(cute.make_layout(4), Float64)
        for r in cutlass.range_constexpr(4):
            sums[r] = Float64(0)
        bias_sum = Float64(0)
        # A 4x64 head tile is shared across all rank columns. FP64 products
        # of BF16 inputs are exact; the only output narrowing is after K.
        for k_tile in range(72):
            local_r = Int32(thread) // 64
            k = Int32(k_tile) * 64 + Int32(thread) % 64
            relative = Int32(tile) * 4 + local_r
            value = BFloat16(0)
            if relative < rows:
                value = lm_head[(Int64(first_row) + Int64(relative)) * 4608 + Int64(k)]
            head[local_r, Int32(thread) % 64] = value
            cute.arch.sync_threads()
            for inner in cutlass.range_constexpr(64):
                k_index = Int64(k_tile) * 64 + Int64(inner)
                weight = Float64(hidden_out[k_index * 256 + Int64(thread)])
                for r in cutlass.range_constexpr(4):
                    sums[r] += Float64(head[r, inner]) * weight
                if thread < 4:
                    bias_sum += Float64(head[Int32(thread), inner]) * Float64(hidden_bias[k_index])
            cute.arch.sync_threads()
        for r in cutlass.range_constexpr(4):
            relative = Int32(tile) * 4 + r
            if relative < rows:
                absolute = Int64(first_row) + Int64(relative)
                folded_weight[absolute * 256 + Int64(thread)] = BFloat16(_narrow_double(sums[r]))
        if thread < 4:
            relative = Int32(tile) * 4 + Int32(thread)
            if relative < rows:
                folded_bias[Int64(first_row) + Int64(relative)] = BFloat16(_narrow_double(bias_sum))


class _LogitFinish:
    @cute.jit
    def __call__(self, base: cute.Pointer, correction: cute.Pointer,
                 output: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if rows > 0 and rows <= 448:
            self.kernel(base, correction, output, rows).launch(
                grid=((rows * 200058 + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, base: cute.Pointer, correction: cute.Pointer,
               output: cute.Pointer, rows: Int32):
        tile, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        index = Int64(tile) * 256 + Int64(thread)
        if index < Int64(rows) * 200058:
            source = (index // 200058) * 201024 + index % 200058
            value = _round_bf16(Float32(base[source]) + Float32(correction[source]))
            output[index] = BFloat16(Float32(30) * cute.math.tanh(
                div_rn_f32(value, Float32(30)), fastmath=False))


def compile_inkling_head_fold_aot():
    """FP64 startup fold, absolute output rows; zero device scratch, 512B/CTA SMEM."""
    return compile_program(_HeadFold(), name="inkling_head_fold", operands=(
        Operand("lm_head", torch.bfloat16, "[201024,4608]"),
        Operand("hidden_out", torch.bfloat16, "[4608,256]"),
        Operand("hidden_bias", torch.bfloat16, "[4608]"),
        Operand("folded_weight", torch.bfloat16, "[201024,256]", "out"),
        Operand("folded_bias", torch.bfloat16, "[201024]", "out"),
    ), scalars=(Scalar("first_row"), Scalar("rows")), key=(201024, 4608, 256),
        geometry={"vocab": 201024, "hidden": 4608, "rank": 256}, doc=__doc__)


def compile_inkling_logit_finish_aot():
    """BF16 sum before FP32 softcap30; output uses logical 200058 pitch."""
    return compile_program(_LogitFinish(), name="inkling_logit_finish", operands=(
        Operand("base", torch.bfloat16, "[rows,201024]"),
        Operand("correction", torch.bfloat16, "[rows,201024]"),
        Operand("output", torch.bfloat16, "[rows,200058]", "out", align=2),
    ), scalars=(Scalar("rows"),), key=(201024, 200058, 30),
        geometry={"vocab": 201024, "valid_vocab": 200058, "softcap": 30}, doc=__doc__)
