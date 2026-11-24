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
    block_reduce, cvt_f32_to_bf16_bits, cvt_f32_to_e4m3, div_rn_f32,
    fabs_f32, fmax_f32, u32_as_f32, warp_reduce,
)
from ._common import Operand, Scalar, compile_program
from .pointwise import _mul, _add as add_rn, _sub

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
    def __init__(self, hidden: int = _HIDDEN, taps: int = _TAPS):
        self.hidden = int(hidden)
        self.taps = int(taps)
        self.items = self.hidden // _THREADS

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
        base = Int64(row) * Int64(self.taps * self.hidden)
        values = cute.make_rmem_tensor(cute.make_layout(self.items), Float32)
        squares = Float32(0)
        for item in cutlass.range_constexpr(self.items):
            col = Int32(thread) + Int32(item * _THREADS)
            value = Float32(0)
            for tap in cutlass.range_constexpr(self.taps):
                value += (Float32(features[base + Int64(tap * self.hidden) + Int64(col)])
                          * Float32(alpha[tap]))
            values[item] = value
            squares += value * value
        allocator = cutlass.utils.SmemAllocator()
        scratch = allocator.allocate_tensor(Float32, cute.make_layout((1, 8)), byte_alignment=16)
        inverse = allocator.allocate_tensor(Float32, cute.make_layout(1), byte_alignment=4)
        total = block_reduce(warp_reduce(squares, _add), _add, scratch, Float32(0))
        if thread == 0:
            inverse[0] = cute.math.rsqrt(total / Float32(self.hidden) + eps, fastmath=True)
        cute.arch.sync_threads()
        inv = inverse[0]
        for item in cutlass.range_constexpr(self.items):
            col = Int32(thread) + Int32(item * _THREADS)
            mixed[Int64(row) * Int64(self.hidden) + Int64(col)] = BFloat16(
                values[item] * inv * Float32(gamma[col]))


class _MaskInit:
    def __init__(self, hidden: int = _HIDDEN):
        self.hidden = int(hidden)
        self.items = self.hidden // _THREADS

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
        for item in cutlass.range_constexpr(self.items):
            col = Int64(thread) + Int64(item * _THREADS)
            value = mask[col]
            if pos == 0:
                value = normalized_anchors[Int64(seq) * Int64(self.hidden) + col]
            block[(Int64(seq) * Int64(8) + Int64(pos)) * Int64(self.hidden) + col] = value


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


def compile_inkling_dflare_mix_aot(*, hidden: int = _HIDDEN, taps: int = _TAPS):
    """(features, alpha, gamma, mixed, rows:i32, eps:f32, stream)."""
    if type(hidden) is not int or hidden <= 0 or hidden % _THREADS or type(taps) is not int or taps <= 0:
        raise ValueError("Inkling dflare_mix requires hidden % 256 == 0 and positive taps")
    return compile_program(_Mix(hidden, taps), name="inkling_dflare_mix", operands=(
        Operand("features", torch.bfloat16, f"[rows,{taps},{hidden}]"),
        Operand("alpha", torch.bfloat16, f"[{taps}]"),
        Operand("gamma", torch.bfloat16, f"[{hidden}]"),
        Operand("mixed", torch.bfloat16, f"[rows,{hidden}]", "out"),
    ), scalars=(Scalar("rows"), Scalar("eps", "float32")),
        key=(hidden, taps), geometry={"hidden": hidden, "taps": taps}, doc=__doc__)


def compile_inkling_mask_init_aot(*, hidden: int = _HIDDEN):
    """(normalized_anchors, mask, block, sequences:i32, stream)."""
    if type(hidden) is not int or hidden <= 0 or hidden % _THREADS:
        raise ValueError("Inkling mask_init requires hidden % 256 == 0")
    return compile_program(_MaskInit(hidden), name="inkling_mask_init", operands=(
        Operand("normalized_anchors", torch.bfloat16, f"[sequences,{hidden}]"),
        Operand("mask", torch.bfloat16, f"[{hidden}]"),
        Operand("block", torch.bfloat16, f"[sequences,8,{hidden}]", "out"),
    ), scalars=(Scalar("sequences"),), key=(hidden, 8),
        geometry={"hidden": hidden, "block": 8}, doc=__doc__)


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
    """Match vendor Torch's FP64 -> FP32 -> BF16 cast, including double rounds."""
    return Float32(llvm.inline_asm(T.f32(), [Float64(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 f; .reg .b16 b; cvt.rn.f32.f64 f, $1; "
        "cvt.rn.bf16.f32 b, f; cvt.f32.bf16 $0, b; }", "=f,d",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


class _HeadFold:
    def __init__(self, hidden: int = _HIDDEN):
        self.hidden = int(hidden)
        self.k_tiles = self.hidden // 64

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
        for k_tile in range(self.k_tiles):
            local_r = Int32(thread) // 64
            k = Int32(k_tile) * 64 + Int32(thread) % 64
            relative = Int32(tile) * 4 + local_r
            value = BFloat16(0)
            if relative < rows:
                value = lm_head[(Int64(first_row) + Int64(relative)) * Int64(self.hidden) + Int64(k)]
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


def compile_inkling_head_fold_aot(*, hidden: int = _HIDDEN):
    """FP64 startup fold, absolute output rows; zero device scratch, 512B/CTA SMEM."""
    if type(hidden) is not int or hidden <= 0 or hidden % 64:
        raise ValueError("Inkling head_fold requires hidden % 64 == 0")
    return compile_program(_HeadFold(hidden), name="inkling_head_fold", operands=(
        Operand("lm_head", torch.bfloat16, f"[201024,{hidden}]"),
        Operand("hidden_out", torch.bfloat16, f"[{hidden},256]"),
        Operand("hidden_bias", torch.bfloat16, f"[{hidden}]"),
        Operand("folded_weight", torch.bfloat16, "[201024,256]", "out"),
        Operand("folded_bias", torch.bfloat16, "[201024]", "out"),
    ), scalars=(Scalar("first_row"), Scalar("rows")), key=(201024, hidden, 256),
        geometry={"vocab": 201024, "hidden": hidden, "rank": 256}, doc=__doc__)


def compile_inkling_logit_finish_aot():
    """BF16 sum before FP32 softcap30; output uses logical 200058 pitch."""
    return compile_program(_LogitFinish(), name="inkling_logit_finish", operands=(
        Operand("base", torch.bfloat16, "[rows,201024]"),
        Operand("correction", torch.bfloat16, "[rows,201024]"),
        Operand("output", torch.bfloat16, "[rows,200058]", "out", align=2),
    ), scalars=(Scalar("rows"),), key=(201024, 200058, 30),
        geometry={"vocab": 201024, "valid_vocab": 200058, "softcap": 30}, doc=__doc__)


class _ResidualNorm:
    def __init__(self, hidden: int = _HIDDEN):
        self.hidden = int(hidden)
        self.items = self.hidden // 256

    @cute.jit
    def __call__(self, residual: cute.Pointer, delta: cute.Pointer,
                 gamma: cute.Pointer, residual_out: cute.Pointer,
                 norm_out: cute.Pointer, rows: Int32, eps: Float32,
                 stream: cuda.CUstream):
        if rows > 0 and rows <= 4096:
            self.kernel(residual, delta, gamma, residual_out, norm_out, eps).launch(
                grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, residual: cute.Pointer, delta: cute.Pointer,
               gamma: cute.Pointer, residual_out: cute.Pointer,
               norm_out: cute.Pointer, eps: Float32):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        values = cute.make_rmem_tensor(cute.make_layout(self.items), Float32)
        squares = Float32(0)
        for item in cutlass.range_constexpr(self.items):
            column = Int64(thread) + Int64(item * 256)
            index = Int64(row) * Int64(self.hidden) + column
            value = Float32(residual[index]) + Float32(delta[index])
            values[item] = value
            residual_out[index] = BFloat16(value)
            squares += value * value
        allocator = cutlass.utils.SmemAllocator()
        scratch = allocator.allocate_tensor(Float32, cute.make_layout((1, 8)), byte_alignment=16)
        inverse = allocator.allocate_tensor(Float32, cute.make_layout(1), byte_alignment=4)
        total = block_reduce(warp_reduce(squares, _add), _add, scratch, Float32(0))
        if thread == 0:
            inverse[0] = cute.math.rsqrt(total / Float32(self.hidden) + eps, fastmath=True)
        cute.arch.sync_threads()
        for item in cutlass.range_constexpr(self.items):
            column = Int64(thread) + Int64(item * 256)
            norm_out[Int64(row) * Int64(self.hidden) + column] = BFloat16(
                values[item] * inverse[0] * Float32(gamma[column]))


class _QKNormRope:
    def __init__(self, q_heads, kv_heads, normalize=True, rotate=True):
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.heads = q_heads + 2 * kv_heads
        self.normalize, self.rotate = normalize, rotate

    @cute.jit
    def __call__(self, qkv: cute.Pointer, q_gamma: cute.Pointer,
                 k_gamma: cute.Pointer, positions: cute.Pointer,
                 inv_freq: cute.Pointer, q_out: cute.Pointer,
                 k_out: cute.Pointer, v_out: cute.Pointer, rows: Int32,
                 eps: Float32, stream: cuda.CUstream):
        if rows > 0 and rows <= 4096:
            self.kernel(qkv, q_gamma, k_gamma, positions, inv_freq,
                        q_out, k_out, v_out, eps).launch(
                grid=(rows, self.q_heads + self.kv_heads, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, qkv: cute.Pointer, q_gamma: cute.Pointer,
               k_gamma: cute.Pointer, positions: cute.Pointer,
               inv_freq: cute.Pointer, q_out: cute.Pointer,
               k_out: cute.Pointer, v_out: cute.Pointer, eps: Float32):
        row, head, _ = cute.arch.block_idx()
        column, _, _ = cute.arch.thread_idx()
        base = (Int64(row) * Int64(self.heads) + Int64(head)) * 128
        value = Float32(qkv[base + Int64(column)])
        allocator = cutlass.utils.SmemAllocator()
        scratch = allocator.allocate_tensor(Float32, cute.make_layout((1, 4)), byte_alignment=16)
        normalized = allocator.allocate_tensor(Float32, cute.make_layout(128), byte_alignment=16)
        inverse = allocator.allocate_tensor(Float32, cute.make_layout(1), byte_alignment=4)
        total = block_reduce(warp_reduce(value * value, _add), _add, scratch, Float32(0))
        if column == 0:
            inverse[0] = div_rn_f32(Float32(1), cute.math.sqrt(total / Float32(128) + eps, fastmath=False))
        cute.arch.sync_threads()
        gamma = Float32(k_gamma[column])
        if head < self.q_heads:
            gamma = Float32(q_gamma[column])
        if cutlass.const_expr(self.normalize):
            normalized[column] = _round_bf16(value * inverse[0] * gamma)
        else:
            normalized[column] = value
        cute.arch.sync_threads()
        half_column = Int32(column) % 64
        angle = Float32(positions[row]) * Float32(inv_freq[half_column])
        cosine = cute.math.cos(angle, fastmath=False)
        sine = cute.math.sin(angle, fastmath=False)
        first = normalized[half_column]
        second = normalized[half_column + 64]
        result = _sub(_mul(first, cosine), _mul(second, sine))
        if column >= 64:
            result = add_rn(_mul(second, cosine), _mul(first, sine))
        if cutlass.const_expr(not self.rotate):
            result = normalized[column]
        if head < self.q_heads:
            q_out[(Int64(row) * Int64(self.q_heads) + Int64(head)) * 128 + Int64(column)] = BFloat16(result)
        else:
            kv_head = Int32(head) - self.q_heads
            offset = (Int64(row) * Int64(self.kv_heads) + Int64(kv_head)) * 128 + Int64(column)
            k_out[offset] = BFloat16(result)
            source = (Int64(row) * Int64(self.heads) + Int64(self.q_heads + self.kv_heads) + Int64(kv_head)) * 128 + Int64(column)
            v_out[offset] = qkv[source]


def compile_inkling_residual_norm_aot(*, hidden: int = _HIDDEN):
    if type(hidden) is not int or hidden <= 0 or hidden % 256:
        raise ValueError("Inkling residual_norm requires hidden % 256 == 0")
    return compile_program(_ResidualNorm(hidden), name="inkling_residual_norm", operands=(
        Operand("residual", torch.bfloat16, f"[rows,{hidden}]"),
        Operand("delta", torch.bfloat16, f"[rows,{hidden}]"),
        Operand("gamma", torch.bfloat16, f"[{hidden}]"),
        Operand("residual_out", torch.bfloat16, f"[rows,{hidden}]", "out"),
        Operand("norm_out", torch.bfloat16, f"[rows,{hidden}]", "out"),
    ), scalars=(Scalar("rows"), Scalar("eps", "float32")), key=(hidden,),
        geometry={"hidden": hidden}, doc=__doc__)


def compile_inkling_qk_norm_rope_aot(*, q_heads=48, kv_heads=8):
    if (q_heads, kv_heads) not in ((48, 8), (24, 8), (24, 4), (0, 8), (0, 4)):
        raise ValueError("Inkling Q/K heads require Small/Mini TP1, TP2 or context-only geometry")
    return compile_program(_QKNormRope(q_heads, kv_heads), name="inkling_qk_norm_rope", operands=(
        Operand("qkv", torch.bfloat16, f"[rows,{(q_heads+2*kv_heads)*128}]"),
        Operand("q_gamma", torch.bfloat16, "[128]"),
        Operand("k_gamma", torch.bfloat16, "[128]"),
        Operand("positions", torch.int64, "[rows]"),
        Operand("inv_freq", torch.float32, "[64]"),
        Operand("q_out", torch.bfloat16, f"[rows,{q_heads},128]", "out"),
        Operand("k_out", torch.bfloat16, f"[rows,{kv_heads},128]", "out"),
        Operand("v_out", torch.bfloat16, f"[rows,{kv_heads},128]", "out"),
    ), scalars=(Scalar("rows"), Scalar("eps", "float32")), key=(q_heads, kv_heads, 128),
        geometry={"q_heads": q_heads, "kv_heads": kv_heads, "head_dim": 128}, doc=__doc__)


class _SiluMul:
    def __init__(self, intermediate):
        self.intermediate = intermediate

    @cute.jit
    def __call__(self, gate_up: cute.Pointer, activated: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows > 0 and rows <= 4096:
            self.kernel(gate_up, activated, rows).launch(
                grid=((rows * self.intermediate + 255) // 256, 1, 1),
                block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, gate_up: cute.Pointer, activated: cute.Pointer, rows: Int32):
        tile, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        index = Int64(tile) * 256 + Int64(thread)
        if index < Int64(rows) * Int64(self.intermediate):
            source = (index // self.intermediate) * Int64(2 * self.intermediate) + index % self.intermediate
            gate = Float32(gate_up[source])
            up = Float32(gate_up[source + self.intermediate])
            silu = div_rn_f32(gate, Float32(1) + cute.math.exp(-gate, fastmath=False))
            activated[index] = BFloat16(silu * up)


def compile_inkling_silu_mul_aot(*, intermediate=18432):
    if intermediate not in (18432, 9216):
        raise ValueError("Inkling SiLU intermediate must be TP1=18432 or TP2=9216")
    return compile_program(_SiluMul(intermediate), name="inkling_silu_mul", operands=(
        Operand("gate_up", torch.bfloat16, f"[rows,{2*intermediate}]"),
        Operand("activated", torch.bfloat16, f"[rows,{intermediate}]", "out"),
    ), scalars=(Scalar("rows"),), key=(intermediate,),
        geometry={"intermediate": intermediate}, doc=__doc__)


class _BiasAdd:
    def __init__(self, width, bf16_output):
        self.width, self.bf16_output = width, bf16_output

    @cute.jit
    def __call__(self, x: cute.Pointer, bias: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows > 0 and rows <= 448:
            self.kernel(x, bias, out, rows).launch(
                grid=((rows * self.width + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, bias: cute.Pointer, out: cute.Pointer, rows: Int32):
        tile, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        index = Int64(tile) * 256 + Int64(thread)
        if index < Int64(rows) * Int64(self.width):
            value = Float32(x[index]) + Float32(bias[index % self.width])
            if cutlass.const_expr(self.bf16_output):
                out[index] = BFloat16(value)
            else:
                out[index] = value


def compile_inkling_bias_add_aot(*, width=768):
    if width not in (768, 201024):
        raise ValueError("Inkling bias add width must be 768 or 201024")
    dtype = torch.bfloat16 if width == 201024 else torch.float32
    return compile_program(_BiasAdd(width, width == 201024), name="inkling_bias_add", operands=(
        Operand("x", torch.float32, f"[rows,{width}]"),
        Operand("bias", torch.bfloat16, f"[{width}]"),
        Operand("out", dtype, f"[rows,{width}]", "out"),
    ), scalars=(Scalar("rows"),), key=(width, str(dtype)),
        geometry={"width": width, "output_dtype": str(dtype)}, doc=__doc__)


class _KVStore:
    def __init__(self, kv_heads, fp8):
        self.kv_heads, self.fp8 = kv_heads, fp8
        self.stride = (kv_heads * 132 + 15) // 16 * 16

    @cute.jit
    def __call__(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
                 k_plane: cute.Pointer, v_plane: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        if rows > 0 and rows <= 4096:
            self.kernel(k, v, slots, k_plane, v_plane).launch(
                grid=(rows, self.kv_heads, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
               k_plane: cute.Pointer, v_plane: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        slot = Int64(slots[row])
        if slot >= 0:
            for plane in cutlass.range_constexpr(2):
                source, destination = k, k_plane
                if cutlass.const_expr(plane == 1):
                    source, destination = v, v_plane
                values = cute.make_rmem_tensor(cute.make_layout(4), Float32)
                maximum = Float32(0)
                for item in cutlass.range_constexpr(4):
                    column = Int64(lane) + Int64(item * 32)
                    index = (Int64(row) * Int64(self.kv_heads) + Int64(head)) * 128 + column
                    values[item] = Float32(source[index])
                    maximum = fmax_f32(maximum, fabs_f32(values[item]))
                if cutlass.const_expr(self.fp8):
                    maximum = warp_reduce(maximum, fmax_f32)
                    scale = Float32(1)
                    if maximum > Float32(0):
                        scale = div_rn_f32(maximum, Float32(448))
                    inverse = div_rn_f32(Float32(1), scale)
                    record = destination + slot * Int64(self.stride)
                    scales = cute.recast_ptr(record + self.kv_heads * 128, dtype=Float32)
                    if lane == 0:
                        scales[head] = scale
                    for item in cutlass.range_constexpr(4):
                        column = Int64(lane) + Int64(item * 32)
                        record[Int64(head) * 128 + column] = cutlass.Uint8(
                            cvt_f32_to_e4m3(values[item] * inverse))
                    if head == 0 and lane < self.stride - self.kv_heads * 132:
                        record[self.kv_heads * 132 + lane] = cutlass.Uint8(0)
                else:
                    for item in cutlass.range_constexpr(4):
                        column = Int64(lane) + Int64(item * 32)
                        destination[(slot * Int64(self.kv_heads) + Int64(head)) * 128 + column] = BFloat16(values[item])


def compile_inkling_kv_store_aot(*, kv_heads=8, kv_format="fp8"):
    if kv_heads not in (8, 4) or kv_format not in ("fp8", "bf16"):
        raise ValueError("Inkling KV store requires 8/4 heads and fp8/bf16")
    fp8 = kv_format == "fp8"
    dtype = torch.uint8 if fp8 else torch.bfloat16
    return compile_program(_KVStore(kv_heads, fp8), name="inkling_kv_store", operands=(
        Operand("k", torch.bfloat16, f"[rows,{kv_heads},128]"),
        Operand("v", torch.bfloat16, f"[rows,{kv_heads},128]"),
        Operand("slots", torch.int64, "[rows]"),
        Operand("k_plane", dtype, "[slots,plane_stride]", "out"),
        Operand("v_plane", dtype, "[slots,plane_stride]", "out"),
    ), scalars=(Scalar("rows"),), key=(kv_heads, kv_format),
        geometry={"kv_heads": kv_heads, "kv_format": kv_format,
                  "plane_bytes": (kv_heads*132+15)//16*16 if fp8 else kv_heads*256}, doc=__doc__)
