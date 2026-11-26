"""Inkling 2 Mini dedicated fused per-visit glue programs (SM120/SM121).

Dedicated Mini-only programs that fuse the per-visit non-GEMM glue chains while
preserving stock checkpoint arithmetic, reduction order, and BF16 rounding
boundaries bit-for-bit across all row counts 1..64 (and 1..4096 for row-parallel
norms/residuals/router):

1. ``compile_inkling2_mini_fast_norm_aot``: weighted H=2304 F32->BF16 RMSNorm
   preloading all BF16 weights before any output store to eliminate pointer-alias
   serialization stalls while preserving stock 128-thread, 4-warp reduction order.
2. ``compile_inkling2_mini_residual_norm_aot``: fused AttnRes (q=1 or q=3, c<=5)
   + weighted H=2304 RMSNorm -> BF16, preserving exact 256-thread AttnRes and
   128-thread Norm reduction orders via a 9 KiB shared-memory handoff.
3. ``compile_inkling2_mini_loop_blend_norm_aot``: fast pass>0 controller blend
   + weighted RMSNorm (slots=1 or 3, optional broadcast dest/weight) preloading
   weights and parallelizing F32 tanh across 768 threads while preserving exact
   256-thread, 8-warp reduction order.
4. ``compile_inkling2_mini_residual_pair_norm_aot``: fused pass>0 MLP AttnRes
   (q=1, c<=5) + LoopPairNorm (slots=1) in one launch.
5. ``compile_inkling2_mini_attn_cross_pair_norm_aot``: fused pass>0 attention
   AttnRes (q=1 or 3, c<=5) + 4-query cross-pass AttnRes (c=2) + LoopPairNorm
   (slots=3) in one launch.
6. ``compile_inkling2_mini_attn_prep_aot``: fused decode/verify (rows 1..64)
   Q clip + R clip & zero-pad to K=64 + K BF16->sconv->commit->BF16->clip->pack
   + V BF16->sconv->commit->BF16->norm->pack in one launch (replay or strided,
   local BF16 or global BF16/FP8).
7. ``compile_inkling2_mini_post_attn_conv_aot``: fused decode/verify (rows 1..64)
   BF16 wo_ud output -> attn_sconv -> commit -> BF16 round -> block-start copy or
   F32 residual add -> residual_arena write in one launch.
8. ``compile_inkling2_mini_post_mlp_conv_residual_aot``: fused decode/verify
   (rows 1..64) BF16 expert output -> mlp_sconv -> commit -> BF16 round -> F32
   residual add -> BF16 round -> partial & residual_arena write in one launch.
9. ``compile_inkling2_mini_fast_router_aot``: single-warp (32-thread, zero
   barrier/smem) E=128 top-8 sigmoid+bias router with unbiased log-sigmoid
   softmax weights scaled by 8.0.

All programs are zero-scratch, graph-capture safe, and never bake in device SM
counts. Existing shared Inkling 2 / Mini programs are untouched.
"""
from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
)
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import (
    _bf16,
    _block_sum,
    _fp8_scale,
    _reduction_storage,
    _rsqrt,
    _warp_max,
    _warp_sum,
)
from .inkling2_convolution import _mul_rn

HIDDEN = 2304
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
KV_WIDTH = KV_HEADS * HEAD_DIM
REL_DIM = 16
REL_PADDED = 64
TAPS = 4


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Force IEEE add.rn.f32 so +0.0 + (-0.0) normalizes to +0.0 like widen_zero."""
    return Float32(llvm.inline_asm(
        T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def _mixed_reduction_storage(slots: int, warps: int, width: int):
    class Storage:
        pass

    Storage.__annotations__ = {
        "sums": cute.struct.Align[cute.struct.MemRange[Float32, slots * warps], 16],
        "mixed": cute.struct.Align[cute.struct.MemRange[Float32, width], 16],
    }
    return cute.struct(Storage)


def _attn_prep_storage(fp8: bool):
    class Storage:
        pass

    if fp8:
        Storage.__annotations__ = {
            "sums": cute.struct.Align[cute.struct.MemRange[Float32, 4], 16],
            "kv": cute.struct.Align[cute.struct.MemRange[BFloat16, HEAD_DIM], 16],
        }
    else:
        Storage.__annotations__ = {
            "sums": cute.struct.Align[cute.struct.MemRange[Float32, 4], 16],
        }
    return cute.struct(Storage)


class _MiniFastNorm:
    threads = 128

    def __init__(self, width: int, eps: float):
        self.width = int(width)
        self.eps = float(eps)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(x, weight, out).launch(
                grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int64(cute.arch.thread_idx()[0])
        base = row * Int64(self.width)
        storage = utils.SmemAllocator().allocate(_reduction_storage(1, self.threads // 32))
        sums = storage.sums.get_tensor(cute.make_layout((1, self.threads // 32)))
        values = cute.make_rmem_tensor((self.width // self.threads,), Float32)
        weights = cute.make_rmem_tensor((self.width // self.threads,), Float32)
        square = Float32(0)
        for j in cutlass.range_constexpr(self.width // self.threads):
            col = tid + Int64(j * self.threads)
            value = Float32(x[base + col])
            weights[j] = Float32(weight[col])
            values[j] = value
            square = square + value * value
        total = _block_sum(square, sums, Int32(0), self.threads // 32)
        inv = _rsqrt(div_rn_f32(total, Float32(self.width)) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.width // self.threads):
            col = tid + Int64(j * self.threads)
            out[base + col] = BFloat16((values[j] * inv) * weights[j])


class _MiniResidualNorm:
    threads = 256

    def __init__(self, hidden: int, queries: int, maximum: int, eps: float):
        self.hidden = int(hidden)
        self.queries = int(queries)
        self.maximum = int(maximum)
        self.eps = float(eps)
        self.qscale = hidden ** -0.5

    @cute.jit
    def __call__(self, arena: cute.Pointer, query: cute.Pointer, weight: cute.Pointer,
                 out: cute.Pointer, rows: Int32, arena_rows: Int32, candidates: Int32,
                 stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(arena, query, weight, out, rows, arena_rows, candidates).launch(
                grid=(rows, self.queries, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, arena: cute.Pointer, query: cute.Pointer, weight: cute.Pointer,
               out: cute.Pointer, rows: Int32, arena_rows: Int32, candidates: Int32):
        row = Int64(cute.arch.block_idx()[0])
        q = Int64(cute.arch.block_idx()[1])
        tid = Int64(cute.arch.thread_idx()[0])
        tid32 = Int32(cute.arch.thread_idx()[0])
        stride = Int64(arena_rows) * Int64(self.hidden)
        base = row * Int64(self.hidden)
        storage = utils.SmemAllocator().allocate(_mixed_reduction_storage(2, self.threads // 32, self.hidden))
        sums = storage.sums.get_tensor(
            cute.make_layout((2, self.threads // 32), stride=(self.threads // 32, 1)))
        mixed_s = storage.mixed.get_tensor(cute.make_layout((self.hidden,)))
        values = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        q_weights = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        n_weights = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        mixed = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        scores = cute.make_rmem_tensor((self.maximum,), Float32)
        inverse = cute.make_rmem_tensor((self.maximum,), Float32)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            q_weights[j] = Float32(query[q * Int64(self.hidden) + col]) * Float32(self.qscale)
            n_weights[j] = Float32(weight[q * Int64(self.hidden) + col])
            mixed[j] = Float32(0)
        maximum = Float32(-float("inf"))
        for c in cutlass.range_constexpr(self.maximum):
            scores[c], inverse[c] = Float32(0), Float32(0)
            if Int32(c) < candidates:
                square = Float32(0)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    col = tid + Int64(j * self.threads)
                    value = Float32(arena[Int64(c) * stride + base + col])
                    values[j] = value
                    square = square + value * value
                total = _block_sum(square, sums, Int32(0), self.threads // 32)
                inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + Float32(self.eps))
                inverse[c] = inv
                dot = Float32(0)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    dot = dot + (values[j] * inv) * q_weights[j]
                scores[c] = _block_sum(dot, sums, Int32(1), self.threads // 32)
                maximum = fmax_f32(maximum, scores[c])
        denominator = Float32(0)
        for c in cutlass.range_constexpr(self.maximum):
            if Int32(c) < candidates:
                scores[c] = cute.math.exp(scores[c] - maximum, fastmath=False)
                denominator = denominator + scores[c]
        for c in cutlass.range_constexpr(self.maximum):
            if Int32(c) < candidates:
                probability = div_rn_f32(scores[c], denominator)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    col = tid + Int64(j * self.threads)
                    value = Float32(arena[Int64(c) * stride + base + col])
                    mixed[j] = mixed[j] + (value * inverse[c]) * probability
        # Exact 128-thread, 4-warp RMSNorm reduction matching stock _Norm bit-for-bit.
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            mixed_s[tid + Int64(j * self.threads)] = mixed[j]
        cute.arch.sync_threads()
        if tid32 < Int32(128):
            nsq = Float32(0)
            for j in cutlass.range_constexpr(self.hidden // 128):
                v = mixed_s[tid + Int64(j * 128)]
                nsq = nsq + v * v
            nsq = _warp_sum(nsq)
            if tid32 % Int32(32) == Int32(0):
                sums[0, tid32 // Int32(32)] = nsq
        cute.arch.sync_threads()
        ntotal = Float32(0.0)
        for src in cutlass.range_constexpr(4):
            ntotal = ntotal + sums[0, src]
        ninv = _rsqrt(div_rn_f32(ntotal, Float32(self.hidden)) + Float32(self.eps))
        out_base = (q * Int64(rows) + row) * Int64(self.hidden)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            out[out_base + col] = BFloat16((mixed[j] * ninv) * n_weights[j])


class _MiniLoopBlendNorm:
    threads = 768

    def __init__(self, hidden: int, slots: int, broadcast_dest: bool,
                 broadcast_weight: bool, eps: float):
        self.hidden = int(hidden)
        self.slots = int(slots)
        self.broadcast_dest = bool(broadcast_dest)
        self.broadcast_weight = bool(broadcast_weight)
        self.eps = float(eps)

    @cute.jit
    def __call__(self, dest: cute.Pointer, msg: cute.Pointer, forget_gain: cute.Pointer,
                 weight: cute.Pointer, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(dest, msg, forget_gain, weight, out, rows).launch(
                grid=(rows, self.slots, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, dest: cute.Pointer, msg: cute.Pointer, forget_gain: cute.Pointer,
               weight: cute.Pointer, out: cute.Pointer, rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        slot = Int64(cute.arch.block_idx()[1])
        tid = Int64(cute.arch.thread_idx()[0])
        tid32 = Int32(cute.arch.thread_idx()[0])
        dest_slot = Int64(0) if cutlass.const_expr(self.broadcast_dest) else slot
        w_slot = Int64(0) if cutlass.const_expr(self.broadcast_weight) else slot
        dest_base = (dest_slot * Int64(rows) + row) * Int64(self.hidden)
        msg_base = (slot * Int64(rows) + row) * Int64(self.hidden)
        fg_base = (slot * Int64(rows) + row) * Int64(2 * self.hidden)
        w_base = w_slot * Int64(self.hidden)
        storage = utils.SmemAllocator().allocate(_mixed_reduction_storage(1, 8, self.hidden))
        sums = storage.sums.get_tensor(cute.make_layout((1, 8), stride=(8, 1)))
        blended_s = storage.mixed.get_tensor(cute.make_layout((self.hidden,)))
        values = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        w_vals = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        # Preload weights before any stores and overlap with tanh compute.
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            w_vals[j] = Float32(weight[w_base + col])
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            forget = cute.math.tanh(Float32(forget_gain[fg_base + col]), fastmath=False)
            gain = cute.math.tanh(Float32(forget_gain[fg_base + Int64(self.hidden) + col]), fastmath=False)
            dv = Float32(dest[dest_base + col])
            mv = Float32(msg[msg_base + col])
            blended = (Float32(1.0) - forget) * dv + gain * mv
            values[j] = blended
            blended_s[col] = blended
        cute.arch.sync_threads()
        # Exact 256-thread, 8-warp reduction order matching stock _LoopBlendNorm.
        if tid32 < Int32(256):
            sq = Float32(0)
            for j in cutlass.range_constexpr(self.hidden // 256):
                v = blended_s[tid + Int64(j * 256)]
                sq = sq + v * v
            sq = _warp_sum(sq)
            if tid32 % Int32(32) == Int32(0):
                sums[0, tid32 // Int32(32)] = sq
        cute.arch.sync_threads()
        total = Float32(0.0)
        for src in cutlass.range_constexpr(8):
            total = total + sums[0, src]
        inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            out[msg_base + col] = BFloat16((values[j] * inv) * w_vals[j])


class _MiniResidualPairNorm:
    threads = 256

    def __init__(self, hidden: int, maximum: int, eps: float):
        self.hidden = int(hidden)
        self.maximum = int(maximum)
        self.eps = float(eps)
        self.qscale = hidden ** -0.5

    @cute.jit
    def __call__(self, arena: cute.Pointer, query: cute.Pointer, msg: cute.Pointer,
                 mixed_out: cute.Pointer, paired_out: cute.Pointer,
                 rows: Int32, arena_rows: Int32, candidates: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(arena, query, msg, mixed_out, paired_out, rows, arena_rows, candidates).launch(
                grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, arena: cute.Pointer, query: cute.Pointer, msg: cute.Pointer,
               mixed_out: cute.Pointer, paired_out: cute.Pointer,
               rows: Int32, arena_rows: Int32, candidates: Int32):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int64(cute.arch.thread_idx()[0])
        stride = Int64(arena_rows) * Int64(self.hidden)
        base = row * Int64(self.hidden)
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, self.threads // 32))
        sums = storage.sums.get_tensor(
            cute.make_layout((2, self.threads // 32), stride=(self.threads // 32, 1)))
        values = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        q_weights = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        mixed = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        m_vals = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        scores = cute.make_rmem_tensor((self.maximum,), Float32)
        inverse = cute.make_rmem_tensor((self.maximum,), Float32)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            q_weights[j] = Float32(query[col]) * Float32(self.qscale)
            m_vals[j] = Float32(msg[base + col])
            mixed[j] = Float32(0)
        maximum = Float32(-float("inf"))
        for c in cutlass.range_constexpr(self.maximum):
            scores[c], inverse[c] = Float32(0), Float32(0)
            if Int32(c) < candidates:
                square = Float32(0)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    col = tid + Int64(j * self.threads)
                    value = Float32(arena[Int64(c) * stride + base + col])
                    values[j] = value
                    square = square + value * value
                total = _block_sum(square, sums, Int32(0), self.threads // 32)
                inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + Float32(self.eps))
                inverse[c] = inv
                dot = Float32(0)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    dot = dot + (values[j] * inv) * q_weights[j]
                scores[c] = _block_sum(dot, sums, Int32(1), self.threads // 32)
                maximum = fmax_f32(maximum, scores[c])
        denominator = Float32(0)
        for c in cutlass.range_constexpr(self.maximum):
            if Int32(c) < candidates:
                scores[c] = cute.math.exp(scores[c] - maximum, fastmath=False)
                denominator = denominator + scores[c]
        for c in cutlass.range_constexpr(self.maximum):
            if Int32(c) < candidates:
                probability = div_rn_f32(scores[c], denominator)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    col = tid + Int64(j * self.threads)
                    value = Float32(arena[Int64(c) * stride + base + col])
                    mixed[j] = mixed[j] + (value * inverse[c]) * probability
        d_sq = Float32(0)
        m_sq = Float32(0)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            dv = mixed[j]
            mv = m_vals[j]
            mixed_out[base + col] = dv
            d_sq = d_sq + dv * dv
            m_sq = m_sq + mv * mv
        d_tot = _block_sum(d_sq, sums, Int32(0), self.threads // 32)
        m_tot = _block_sum(m_sq, sums, Int32(1), self.threads // 32)
        d_inv = _rsqrt(div_rn_f32(d_tot, Float32(self.hidden)) + Float32(self.eps))
        m_inv = _rsqrt(div_rn_f32(m_tot, Float32(self.hidden)) + Float32(self.eps))
        out_base = row * Int64(2 * self.hidden)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            paired_out[out_base + col] = BFloat16(mixed[j] * d_inv)
            paired_out[out_base + Int64(self.hidden) + col] = BFloat16(m_vals[j] * m_inv)


class _MiniAttnCrossPairNorm:
    threads = 256

    def __init__(self, hidden: int, broadcast_dest: bool, maximum: int, eps: float):
        self.hidden = int(hidden)
        self.broadcast_dest = bool(broadcast_dest)
        self.maximum = int(maximum)
        self.eps = float(eps)
        self.qscale = hidden ** -0.5

    @cute.jit
    def __call__(self, arena: cute.Pointer, attn_query: cute.Pointer,
                 prev_arena: cute.Pointer, cross_query: cute.Pointer,
                 dest_out: cute.Pointer, msg_out: cute.Pointer, paired_out: cute.Pointer,
                 rows: Int32, arena_rows: Int32, candidates: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(arena, attn_query, prev_arena, cross_query,
                        dest_out, msg_out, paired_out, rows, arena_rows, candidates).launch(
                grid=(rows, 4, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, arena: cute.Pointer, attn_query: cute.Pointer,
               prev_arena: cute.Pointer, cross_query: cute.Pointer,
               dest_out: cute.Pointer, msg_out: cute.Pointer, paired_out: cute.Pointer,
               rows: Int32, arena_rows: Int32, candidates: Int32):
        row = Int64(cute.arch.block_idx()[0])
        slot = Int64(cute.arch.block_idx()[1])
        tid = Int64(cute.arch.thread_idx()[0])
        stride = Int64(arena_rows) * Int64(self.hidden)
        base = row * Int64(self.hidden)
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, self.threads // 32))
        sums = storage.sums.get_tensor(
            cute.make_layout((2, self.threads // 32), stride=(self.threads // 32, 1)))
        values = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        q_weights = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        msg_vals = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        scores2 = cute.make_rmem_tensor((2,), Float32)
        inv2 = cute.make_rmem_tensor((2,), Float32)

        # 1. Cross-pass message retrieval (4 queries, 2 candidates in prev_arena).
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            q_weights[j] = Float32(cross_query[slot * Int64(self.hidden) + col]) * Float32(self.qscale)
            msg_vals[j] = Float32(0)
        max2 = Float32(-float("inf"))
        for c in cutlass.range_constexpr(2):
            sq = Float32(0)
            for j in cutlass.range_constexpr(self.hidden // self.threads):
                col = tid + Int64(j * self.threads)
                val = Float32(prev_arena[Int64(c) * stride + base + col])
                values[j] = val
                sq = sq + val * val
            tot = _block_sum(sq, sums, Int32(0), self.threads // 32)
            inv = _rsqrt(div_rn_f32(tot, Float32(self.hidden)) + Float32(self.eps))
            inv2[c] = inv
            dot = Float32(0)
            for j in cutlass.range_constexpr(self.hidden // self.threads):
                dot = dot + (values[j] * inv) * q_weights[j]
            scores2[c] = _block_sum(dot, sums, Int32(1), self.threads // 32)
            max2 = fmax_f32(max2, scores2[c])
        den2 = Float32(0)
        for c in cutlass.range_constexpr(2):
            scores2[c] = cute.math.exp(scores2[c] - max2, fastmath=False)
            den2 = den2 + scores2[c]
        for c in cutlass.range_constexpr(2):
            prob = div_rn_f32(scores2[c], den2)
            for j in cutlass.range_constexpr(self.hidden // self.threads):
                col = tid + Int64(j * self.threads)
                val = Float32(prev_arena[Int64(c) * stride + base + col])
                msg_vals[j] = msg_vals[j] + (val * inv2[c]) * prob
        msg_base = (slot * Int64(rows) + row) * Int64(self.hidden)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            msg_out[msg_base + col] = msg_vals[j]

        # 2. Slots 0..2 also compute attention AttnRes (dest) + LoopPairNorm -> paired_out.
        if slot < Int64(3):
            dest_slot = Int64(0) if cutlass.const_expr(self.broadcast_dest) else slot
            dest_vals = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
            scores = cute.make_rmem_tensor((self.maximum,), Float32)
            inverse = cute.make_rmem_tensor((self.maximum,), Float32)
            for j in cutlass.range_constexpr(self.hidden // self.threads):
                col = tid + Int64(j * self.threads)
                q_weights[j] = Float32(attn_query[dest_slot * Int64(self.hidden) + col]) * Float32(self.qscale)
                dest_vals[j] = Float32(0)
            maximum = Float32(-float("inf"))
            for c in cutlass.range_constexpr(self.maximum):
                scores[c], inverse[c] = Float32(0), Float32(0)
                if Int32(c) < candidates:
                    sq = Float32(0)
                    for j in cutlass.range_constexpr(self.hidden // self.threads):
                        col = tid + Int64(j * self.threads)
                        val = Float32(arena[Int64(c) * stride + base + col])
                        values[j] = val
                        sq = sq + val * val
                    tot = _block_sum(sq, sums, Int32(0), self.threads // 32)
                    inv = _rsqrt(div_rn_f32(tot, Float32(self.hidden)) + Float32(self.eps))
                    inverse[c] = inv
                    dot = Float32(0)
                    for j in cutlass.range_constexpr(self.hidden // self.threads):
                        dot = dot + (values[j] * inv) * q_weights[j]
                    scores[c] = _block_sum(dot, sums, Int32(1), self.threads // 32)
                    maximum = fmax_f32(maximum, scores[c])
            den = Float32(0)
            for c in cutlass.range_constexpr(self.maximum):
                if Int32(c) < candidates:
                    scores[c] = cute.math.exp(scores[c] - maximum, fastmath=False)
                    den = den + scores[c]
            for c in cutlass.range_constexpr(self.maximum):
                if Int32(c) < candidates:
                    prob = div_rn_f32(scores[c], den)
                    for j in cutlass.range_constexpr(self.hidden // self.threads):
                        col = tid + Int64(j * self.threads)
                        val = Float32(arena[Int64(c) * stride + base + col])
                        dest_vals[j] = dest_vals[j] + (val * inverse[c]) * prob
            write_dest = True if cutlass.const_expr(not self.broadcast_dest) else (slot == Int64(0))
            if write_dest:
                dest_base = (dest_slot * Int64(rows) + row) * Int64(self.hidden)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    col = tid + Int64(j * self.threads)
                    dest_out[dest_base + col] = dest_vals[j]
            d_sq = Float32(0)
            m_sq = Float32(0)
            for j in cutlass.range_constexpr(self.hidden // self.threads):
                dv = dest_vals[j]
                mv = msg_vals[j]
                d_sq = d_sq + dv * dv
                m_sq = m_sq + mv * mv
            d_tot = _block_sum(d_sq, sums, Int32(0), self.threads // 32)
            m_tot = _block_sum(m_sq, sums, Int32(1), self.threads // 32)
            d_inv = _rsqrt(div_rn_f32(d_tot, Float32(self.hidden)) + Float32(self.eps))
            m_inv = _rsqrt(div_rn_f32(m_tot, Float32(self.hidden)) + Float32(self.eps))
            out_base = (slot * Int64(rows) + row) * Int64(2 * self.hidden)
            for j in cutlass.range_constexpr(self.hidden // self.threads):
                col = tid + Int64(j * self.threads)
                paired_out[out_base + col] = BFloat16(dest_vals[j] * d_inv)
                paired_out[out_base + Int64(self.hidden) + col] = BFloat16(msg_vals[j] * m_inv)


class _MiniAttnPrep:
    threads = 128

    def __init__(self, local: bool, fp8: bool, replay: bool, max_rows: int = 64):
        self.local = bool(local)
        self.dynamic = not self.local
        self.fp8 = bool(fp8)
        self.replay = bool(replay)
        self.max_rows = int(max_rows)
        self.conv_factor = (TAPS * (2 if self.dynamic else 1) + 1) ** -0.5
        self.qk_threshold = HEAD_DIM ** 0.5 * 2.0
        self.r_threshold = 8.0
        self.eps = 1e-6
        self.log_floor = 0 if self.local else 8192
        self.log_alpha = 0.0 if self.local else 0.1
        self.scale_offset = 2 * KV_HEADS * HEAD_DIM
        self.record_bytes = self.scale_offset + 2 * KV_HEADS * 4

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer, q_scale: cute.Pointer,
        r: cute.Pointer, r_scale: cute.Pointer,
        positions: cute.Pointer,
        q_out: cute.Pointer, rel_input_out: cute.Pointer,
        k_bf16: cute.Pointer, k_weight: cute.Pointer, k_dynamic: cute.Pointer, k_history: cute.Pointer,
        v_bf16: cute.Pointer, v_weight: cute.Pointer, v_dynamic: cute.Pointer, v_history: cute.Pointer,
        first: cute.Pointer, conv_slots: cute.Pointer, cache_slots: cute.Pointer, cache: cute.Pointer,
        k_replay: cute.Pointer, v_replay: cute.Pointer,
        rows: Int32, history_slot_stride: Int64, history_lag_stride: Int64, replay_row_stride: Int64,
        stream: cuda.CUstream,
    ):
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and history_lag_stride >= Int64(KV_WIDTH)
                and history_slot_stride >= Int64(TAPS - 1) * history_lag_stride
                and (not self.replay or replay_row_stride >= Int64(KV_WIDTH))):
            qr_ctas = (rows * Int32(HEADS) + Int32(3)) // Int32(4)
            kv_ctas = Int32(16) * rows
            total_ctas = kv_ctas + qr_ctas * Int32(2)
            self.kernel(
                q, q_scale, r, r_scale, positions, q_out, rel_input_out,
                k_bf16, k_weight, k_dynamic, k_history,
                v_bf16, v_weight, v_dynamic, v_history,
                first, conv_slots, cache_slots, cache,
                k_replay, v_replay,
                rows, qr_ctas, history_slot_stride, history_lag_stride, replay_row_stride,
            ).launch(grid=(total_ctas, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer, q_scale: cute.Pointer,
        r: cute.Pointer, r_scale: cute.Pointer,
        positions: cute.Pointer,
        q_out: cute.Pointer, rel_input_out: cute.Pointer,
        k_bf16: cute.Pointer, k_weight: cute.Pointer, k_dynamic: cute.Pointer, k_history: cute.Pointer,
        v_bf16: cute.Pointer, v_weight: cute.Pointer, v_dynamic: cute.Pointer, v_history: cute.Pointer,
        first: cute.Pointer, conv_slots: cute.Pointer, cache_slots: cute.Pointer, cache: cute.Pointer,
        k_replay: cute.Pointer, v_replay: cute.Pointer,
        rows: Int32, qr_ctas: Int32, history_slot_stride: Int64, history_lag_stride: Int64,
        replay_row_stride: Int64,
    ):
        bid = Int32(cute.arch.block_idx()[0])
        tid32 = Int32(cute.arch.thread_idx()[0])
        tid = Int64(tid32)
        storage = utils.SmemAllocator().allocate(_attn_prep_storage(self.fp8))
        sums = storage.sums.get_tensor(cute.make_layout((1, 4)))
        if cutlass.const_expr(self.fp8):
            kv_s = storage.kv.get_tensor(cute.make_layout((HEAD_DIM,)))
        k_ctas = Int32(8) * rows
        kv_ctas = Int32(16) * rows

        if bid < k_ctas:
            # K branch: one CTA per (segment_start_row, head); non-start CTAs exit immediately.
            seg_start = bid // Int32(8)
            if Int32(first[Int64(seg_start)]) == seg_start and tid32 < Int32(32):
                head = Int64(bid % Int32(8))
                lane = tid
                sw = cute.make_rmem_tensor((4, TAPS), Float32)
                h0 = cute.make_rmem_tensor((4,), Float32)
                h1 = cute.make_rmem_tensor((4,), Float32)
                h2 = cute.make_rmem_tensor((4,), Float32)
                vals = cute.make_rmem_tensor((4,), Float32)
                cslot = Int64(conv_slots[Int64(seg_start)])
                for j in cutlass.range_constexpr(4):
                    col = head * Int64(HEAD_DIM) + lane + Int64(j * 32)
                    hbase = cslot * history_slot_stride + col
                    h0[j] = Float32(k_history[hbase])
                    h1[j] = Float32(k_history[hbase + history_lag_stride])
                    h2[j] = Float32(k_history[hbase + Int64(2) * history_lag_stride])
                    for lag in cutlass.range_constexpr(TAPS):
                        if cutlass.const_expr(self.dynamic):
                            sw[j, lag] = Float32(k_weight[Int64(lag * KV_WIDTH) + col])
                        else:
                            sw[j, lag] = Float32(k_weight[col * Int64(TAPS) + Int64(TAPS - 1 - lag)])
                row_i = seg_start
                while row_i < rows and Int32(first[Int64(row_i)]) == seg_start:
                    row = Int64(row_i)
                    dyn0 = Float32(0)
                    dyn1 = Float32(0)
                    dyn2 = Float32(0)
                    dyn3 = Float32(0)
                    if cutlass.const_expr(self.dynamic):
                        dbase = (row * Int64(KV_HEADS) + head) * Int64(TAPS)
                        dyn0 = Float32(k_dynamic[dbase])
                        dyn1 = Float32(k_dynamic[dbase + Int64(1)])
                        dyn2 = Float32(k_dynamic[dbase + Int64(2)])
                        dyn3 = Float32(k_dynamic[dbase + Int64(3)])
                    square = Float32(0)
                    for j in cutlass.range_constexpr(4):
                        col = head * Int64(HEAD_DIM) + lane + Int64(j * 32)
                        cur = _add_rn(Float32(0.0), Float32(k_bf16[row * Int64(KV_WIDTH) + col]))
                        if cutlass.const_expr(self.replay):
                            k_replay[row * replay_row_stride + col] = cur
                        t0 = sw[j, 0] + dyn0 if cutlass.const_expr(self.dynamic) else sw[j, 0]
                        t1 = sw[j, 1] + dyn1 if cutlass.const_expr(self.dynamic) else sw[j, 1]
                        t2 = sw[j, 2] + dyn2 if cutlass.const_expr(self.dynamic) else sw[j, 2]
                        t3 = sw[j, 3] + dyn3 if cutlass.const_expr(self.dynamic) else sw[j, 3]
                        acc = Float32(0)
                        acc = acc + _mul_rn(cur, t0)
                        acc = acc + _mul_rn(h0[j], t1)
                        acc = acc + _mul_rn(h1[j], t2)
                        acc = acc + _mul_rn(h2[j], t3)
                        y = (acc + cur) * Float32(self.conv_factor)
                        kv_val = _bf16(y)
                        h2[j] = h1[j]
                        h1[j] = h0[j]
                        h0[j] = cur
                        vals[j] = kv_val
                        square = square + kv_val * kv_val
                    norm = cute.math.sqrt(_warp_sum(square), fastmath=False)
                    factor = fmin_f32(Float32(1), div_rn_f32(Float32(self.qk_threshold), norm + Float32(self.eps)))
                    kslot = Int64(cache_slots[row])
                    if cutlass.const_expr(not self.fp8):
                        if kslot >= Int64(0):
                            cbase = kslot * Int64(2 * KV_WIDTH)
                            for j in cutlass.range_constexpr(4):
                                col = head * Int64(HEAD_DIM) + lane + Int64(j * 32)
                                cache[cbase + col] = BFloat16((vals[j] * factor) * Float32(1))
                    else:
                        for j in cutlass.range_constexpr(4):
                            kv_s[lane + Int64(j * 32)] = BFloat16((vals[j] * factor) * Float32(1))
                        cute.arch.sync_warp()
                        pvals = cute.make_rmem_tensor((4,), Float32)
                        pmax = Float32(0.0)
                        for j in cutlass.range_constexpr(4):
                            pv = Float32(kv_s[lane * Int64(4) + Int64(j)])
                            pvals[j] = pv
                            pmax = cutlass.max(pmax, cute.math.abs(pv))
                        pscale = _fp8_scale(_warp_max(pmax))
                        pinv = div_rn_f32(Float32(1.0), pscale)
                        packed = cvt_f32x4_to_e4m3x4(
                            pvals[0] * pinv, pvals[1] * pinv, pvals[2] * pinv, pvals[3] * pinv)
                        if kslot >= Int64(0):
                            caddr = Int64(cache.toint()) + kslot * Int64(self.record_bytes)
                            dst = cute.make_ptr(
                                Uint32, caddr + head * Int64(HEAD_DIM) + lane * Int64(4),
                                cute.AddressSpace.gmem, assumed_align=4)
                            dst[0] = packed
                            if lane == Int64(0):
                                sc = cute.make_ptr(
                                    Float32, caddr + Int64(self.scale_offset),
                                    cute.AddressSpace.gmem, assumed_align=4)
                                sc[head] = pscale
                        cute.arch.sync_warp()
                    row_i = row_i + Int32(1)
                for j in cutlass.range_constexpr(4):
                    col = head * Int64(HEAD_DIM) + lane + Int64(j * 32)
                    hbase = cslot * history_slot_stride + col
                    k_history[hbase] = h0[j]
                    k_history[hbase + history_lag_stride] = h1[j]
                    k_history[hbase + Int64(2) * history_lag_stride] = h2[j]

        elif bid < kv_ctas:
            # V branch: one CTA per (segment_start_row, head), all 128 threads match _Norm(V).
            vbid = bid - k_ctas
            seg_start = vbid // Int32(8)
            if Int32(first[Int64(seg_start)]) == seg_start:
                head = Int64(vbid % Int32(8))
                col = head * Int64(HEAD_DIM) + tid
                cslot = Int64(conv_slots[Int64(seg_start)])
                hbase = cslot * history_slot_stride + col
                h0 = Float32(v_history[hbase])
                h1 = Float32(v_history[hbase + history_lag_stride])
                h2 = Float32(v_history[hbase + Int64(2) * history_lag_stride])
                if cutlass.const_expr(self.dynamic):
                    w0 = Float32(v_weight[col])
                    w1 = Float32(v_weight[Int64(KV_WIDTH) + col])
                    w2 = Float32(v_weight[Int64(2 * KV_WIDTH) + col])
                    w3 = Float32(v_weight[Int64(3 * KV_WIDTH) + col])
                else:
                    w0 = Float32(v_weight[col * Int64(TAPS) + Int64(3)])
                    w1 = Float32(v_weight[col * Int64(TAPS) + Int64(2)])
                    w2 = Float32(v_weight[col * Int64(TAPS) + Int64(1)])
                    w3 = Float32(v_weight[col * Int64(TAPS)])
                row_i = seg_start
                while row_i < rows and Int32(first[Int64(row_i)]) == seg_start:
                    row = Int64(row_i)
                    cur = _add_rn(Float32(0.0), Float32(v_bf16[row * Int64(KV_WIDTH) + col]))
                    if cutlass.const_expr(self.replay):
                        v_replay[row * replay_row_stride + col] = cur
                    if cutlass.const_expr(self.dynamic):
                        dbase = (row * Int64(KV_HEADS) + head) * Int64(TAPS)
                        t0 = w0 + Float32(v_dynamic[dbase])
                        t1 = w1 + Float32(v_dynamic[dbase + Int64(1)])
                        t2 = w2 + Float32(v_dynamic[dbase + Int64(2)])
                        t3 = w3 + Float32(v_dynamic[dbase + Int64(3)])
                    else:
                        t0, t1, t2, t3 = w0, w1, w2, w3
                    acc = Float32(0)
                    acc = acc + _mul_rn(cur, t0)
                    acc = acc + _mul_rn(h0, t1)
                    acc = acc + _mul_rn(h1, t2)
                    acc = acc + _mul_rn(h2, t3)
                    y = (acc + cur) * Float32(self.conv_factor)
                    v_val = _add_rn(Float32(0.0), _bf16(y))
                    h2 = h1
                    h1 = h0
                    h0 = cur
                    total = _block_sum(v_val * v_val, sums, Int32(0), 4)
                    inv = _rsqrt(div_rn_f32(total, Float32(HEAD_DIM)) + Float32(self.eps))
                    vn = BFloat16(v_val * inv)
                    vslot = Int64(cache_slots[row])
                    if cutlass.const_expr(not self.fp8):
                        if vslot >= Int64(0):
                            cache[vslot * Int64(2 * KV_WIDTH) + Int64(KV_WIDTH) + col] = vn
                    else:
                        kv_s[tid] = vn
                        cute.arch.sync_threads()
                        if tid32 < Int32(32):
                            pvals = cute.make_rmem_tensor((4,), Float32)
                            pmax = Float32(0.0)
                            for j in cutlass.range_constexpr(4):
                                pv = Float32(kv_s[tid * Int64(4) + Int64(j)])
                                pvals[j] = pv
                                pmax = cutlass.max(pmax, cute.math.abs(pv))
                            pscale = _fp8_scale(_warp_max(pmax))
                            pinv = div_rn_f32(Float32(1.0), pscale)
                            packed = cvt_f32x4_to_e4m3x4(
                                pvals[0] * pinv, pvals[1] * pinv, pvals[2] * pinv, pvals[3] * pinv)
                            if vslot >= Int64(0):
                                caddr = Int64(cache.toint()) + vslot * Int64(self.record_bytes)
                                dst = cute.make_ptr(
                                    Uint32, caddr + Int64(KV_WIDTH) + head * Int64(HEAD_DIM) + tid * Int64(4),
                                    cute.AddressSpace.gmem, assumed_align=4)
                                dst[0] = packed
                                if tid32 == Int32(0):
                                    sc = cute.make_ptr(
                                        Float32, caddr + Int64(self.scale_offset),
                                        cute.AddressSpace.gmem, assumed_align=4)
                                    sc[Int64(KV_HEADS) + head] = pscale
                        cute.arch.sync_threads()
                    row_i = row_i + Int32(1)
                v_history[hbase] = h0
                v_history[hbase + history_lag_stride] = h1
                v_history[hbase + Int64(2) * history_lag_stride] = h2

        elif bid < kv_ctas + qr_ctas:
            # Q clip: 4 warps/CTA, 1 warp per (row, head), exact stock _Clip(Q).
            q_bid = Int64(bid - kv_ctas)
            head_row = q_bid * Int64(4) + tid // Int64(32)
            lane = tid % Int64(32)
            if head_row < Int64(rows) * Int64(HEADS):
                head, row = head_row % Int64(HEADS), head_row // Int64(HEADS)
                values = cute.make_rmem_tensor((4,), Float32)
                square = Float32(0)
                for j in cutlass.range_constexpr(4):
                    col = lane + Int64(j * 32)
                    val = Float32(q[head_row * Int64(HEAD_DIM) + col]) * Float32(q_scale[head * Int64(HEAD_DIM) + col])
                    values[j] = val
                    square = square + val * val
                norm = cute.math.sqrt(_warp_sum(square), fastmath=False)
                factor = fmin_f32(Float32(1), div_rn_f32(Float32(self.qk_threshold), norm + Float32(self.eps)))
                tau = Float32(1)
                if cutlass.const_expr(self.log_floor > 0):
                    ratio = div_rn_f32((positions[row] + Int64(1)).to(Float32), Float32(self.log_floor))
                    tau = Float32(1) + Float32(self.log_alpha) * cute.math.log(
                        fmax_f32(Float32(1), ratio), fastmath=False)
                for j in cutlass.range_constexpr(4):
                    col = lane + Int64(j * 32)
                    q_out[head_row * Int64(HEAD_DIM) + col] = BFloat16((values[j] * factor) * tau)

        else:
            # R clip + zero-pad to K=64 into rel_input_out directly (eliminates r_clipped + copy_rows).
            r_bid = Int64(bid - kv_ctas - qr_ctas)
            head_row = r_bid * Int64(4) + tid // Int64(32)
            lane = tid % Int64(32)
            if head_row < Int64(rows) * Int64(HEADS):
                head, row = head_row % Int64(HEADS), head_row // Int64(HEADS)
                val = Float32(0)
                if lane < Int64(REL_DIM):
                    val = Float32(r[head_row * Int64(REL_DIM) + lane]) * Float32(r_scale[head * Int64(REL_DIM) + lane])
                norm = cute.math.sqrt(_warp_sum(val * val), fastmath=False)
                factor = fmin_f32(Float32(1), div_rn_f32(Float32(self.r_threshold), norm + Float32(self.eps)))
                tau = Float32(1)
                if cutlass.const_expr(self.log_floor > 0):
                    ratio = div_rn_f32((positions[row] + Int64(1)).to(Float32), Float32(self.log_floor))
                    tau = Float32(1) + Float32(self.log_alpha) * cute.math.log(
                        fmax_f32(Float32(1), ratio), fastmath=False)
                out_base = head_row * Int64(REL_PADDED)
                if lane < Int64(REL_DIM):
                    rel_input_out[out_base + lane] = BFloat16((val * factor) * tau)
                rel_input_out[out_base + Int64(16) + lane] = BFloat16(0.0)
                if lane < Int64(16):
                    rel_input_out[out_base + Int64(48) + lane] = BFloat16(0.0)


class _MiniPostAttnConv:
    threads = 128

    def __init__(self, starts_block: bool, replay: bool, max_rows: int = 64):
        self.starts_block = bool(starts_block)
        self.replay = bool(replay)
        self.max_rows = int(max_rows)
        self.factor = (TAPS + 1) ** -0.5

    @cute.jit
    def __call__(
        self,
        x_bf16: cute.Pointer, weight: cute.Pointer, history: cute.Pointer,
        first: cute.Pointer, slots: cute.Pointer,
        partial: cute.Pointer, arena: cute.Pointer, out_base: cute.Pointer,
        replay: cute.Pointer,
        rows: Int32, arena_rows: Int32, mlp_cand_idx: Int32,
        history_slot_stride: Int64, history_lag_stride: Int64, replay_row_stride: Int64,
        stream: cuda.CUstream,
    ):
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and history_lag_stride >= Int64(HIDDEN)
                and history_slot_stride >= Int64(TAPS - 1) * history_lag_stride
                and (not self.replay or replay_row_stride >= Int64(HIDDEN))):
            self.kernel(
                x_bf16, weight, history, first, slots, partial, arena, out_base, replay,
                rows, arena_rows, mlp_cand_idx, history_slot_stride, history_lag_stride,
                replay_row_stride,
            ).launch(grid=(HIDDEN // self.threads, rows, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x_bf16: cute.Pointer, weight: cute.Pointer, history: cute.Pointer,
        first: cute.Pointer, slots: cute.Pointer,
        partial: cute.Pointer, arena: cute.Pointer, out_base: cute.Pointer,
        replay: cute.Pointer,
        rows: Int32, arena_rows: Int32, mlp_cand_idx: Int32,
        history_slot_stride: Int64, history_lag_stride: Int64, replay_row_stride: Int64,
    ):
        seg_start = Int32(cute.arch.block_idx()[1])
        if Int32(first[Int64(seg_start)]) == seg_start:
            col = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
            w0 = Float32(weight[col * Int64(TAPS) + Int64(3)])
            w1 = Float32(weight[col * Int64(TAPS) + Int64(2)])
            w2 = Float32(weight[col * Int64(TAPS) + Int64(1)])
            w3 = Float32(weight[col * Int64(TAPS)])
            arena_stride = Int64(arena_rows) * Int64(HIDDEN)
            mlp_slot_off = Int64(mlp_cand_idx) * arena_stride
            attn_slot_off = Int64(mlp_cand_idx - Int32(1)) * arena_stride
            cslot = Int64(slots[Int64(seg_start)])
            hbase = cslot * history_slot_stride + col
            h0 = Float32(history[hbase])
            h1 = Float32(history[hbase + history_lag_stride])
            h2 = Float32(history[hbase + Int64(2) * history_lag_stride])
            row_i = seg_start
            while row_i < rows and Int32(first[Int64(row_i)]) == seg_start:
                row = Int64(row_i)
                idx = row * Int64(HIDDEN) + col
                cur = _add_rn(Float32(0.0), Float32(x_bf16[idx]))
                if cutlass.const_expr(self.replay):
                    replay[row * replay_row_stride + col] = cur
                acc = Float32(0)
                acc = acc + _mul_rn(cur, w0)
                acc = acc + _mul_rn(h0, w1)
                acc = acc + _mul_rn(h1, w2)
                acc = acc + _mul_rn(h2, w3)
                y = (acc + cur) * Float32(self.factor)
                attn = _add_rn(Float32(0.0), _bf16(y))
                h2 = h1
                h1 = h0
                h0 = cur
                p_val = Float32(partial[idx])
                if cutlass.const_expr(self.starts_block):
                    out_base[idx] = attn
                    arena[attn_slot_off + idx] = p_val
                    arena[mlp_slot_off + idx] = attn
                else:
                    br = p_val + attn
                    out_base[idx] = br
                    arena[mlp_slot_off + idx] = br
                row_i = row_i + Int32(1)
            history[hbase] = h0
            history[hbase + history_lag_stride] = h1
            history[hbase + Int64(2) * history_lag_stride] = h2


class _MiniPostMlpConvResidual:
    threads = 128

    def __init__(self, replay: bool, max_rows: int = 64):
        self.replay = bool(replay)
        self.max_rows = int(max_rows)
        self.factor = (TAPS + 1) ** -0.5

    @cute.jit
    def __call__(
        self,
        x_bf16: cute.Pointer, weight: cute.Pointer, history: cute.Pointer,
        first: cute.Pointer, slots: cute.Pointer,
        base: cute.Pointer, partial: cute.Pointer, arena: cute.Pointer,
        replay: cute.Pointer,
        rows: Int32, arena_rows: Int32, mlp_cand_idx: Int32,
        history_slot_stride: Int64, history_lag_stride: Int64, replay_row_stride: Int64,
        stream: cuda.CUstream,
    ):
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and history_lag_stride >= Int64(HIDDEN)
                and history_slot_stride >= Int64(TAPS - 1) * history_lag_stride
                and (not self.replay or replay_row_stride >= Int64(HIDDEN))):
            self.kernel(
                x_bf16, weight, history, first, slots, base, partial, arena, replay,
                rows, arena_rows, mlp_cand_idx, history_slot_stride, history_lag_stride,
                replay_row_stride,
            ).launch(grid=(HIDDEN // self.threads, rows, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x_bf16: cute.Pointer, weight: cute.Pointer, history: cute.Pointer,
        first: cute.Pointer, slots: cute.Pointer,
        base: cute.Pointer, partial: cute.Pointer, arena: cute.Pointer,
        replay: cute.Pointer,
        rows: Int32, arena_rows: Int32, mlp_cand_idx: Int32,
        history_slot_stride: Int64, history_lag_stride: Int64, replay_row_stride: Int64,
    ):
        seg_start = Int32(cute.arch.block_idx()[1])
        if Int32(first[Int64(seg_start)]) == seg_start:
            col = Int64(cute.arch.block_idx()[0]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
            w0 = Float32(weight[col * Int64(TAPS) + Int64(3)])
            w1 = Float32(weight[col * Int64(TAPS) + Int64(2)])
            w2 = Float32(weight[col * Int64(TAPS) + Int64(1)])
            w3 = Float32(weight[col * Int64(TAPS)])
            arena_stride = Int64(arena_rows) * Int64(HIDDEN)
            mlp_slot_off = Int64(mlp_cand_idx) * arena_stride
            cslot = Int64(slots[Int64(seg_start)])
            hbase = cslot * history_slot_stride + col
            h0 = Float32(history[hbase])
            h1 = Float32(history[hbase + history_lag_stride])
            h2 = Float32(history[hbase + Int64(2) * history_lag_stride])
            row_i = seg_start
            while row_i < rows and Int32(first[Int64(row_i)]) == seg_start:
                row = Int64(row_i)
                idx = row * Int64(HIDDEN) + col
                cur = _add_rn(Float32(0.0), Float32(x_bf16[idx]))
                if cutlass.const_expr(self.replay):
                    replay[row * replay_row_stride + col] = cur
                acc = Float32(0)
                acc = acc + _mul_rn(cur, w0)
                acc = acc + _mul_rn(h0, w1)
                acc = acc + _mul_rn(h1, w2)
                acc = acc + _mul_rn(h2, w3)
                y = (acc + cur) * Float32(self.factor)
                mlp_val = _add_rn(Float32(0.0), _bf16(y))
                h2 = h1
                h1 = h0
                h0 = cur
                out_val = _add_rn(Float32(0.0), _bf16(Float32(base[idx]) + mlp_val))
                partial[idx] = out_val
                arena[mlp_slot_off + idx] = out_val
                row_i = row_i + Int32(1)
            history[hbase] = h0
            history[hbase + history_lag_stride] = h1
            history[hbase + Int64(2) * history_lag_stride] = h2


class _MiniFastRouter:
    """Single-warp E=128 top-8 router (4 experts/lane, zero sync_threads, zero smem)."""
    threads = 32

    def __init__(self, experts: int = 128, topk: int = 8, route_scale: float = 8.0):
        self.experts = int(experts)
        self.topk = int(topk)
        self.route_scale = float(route_scale)

    @cute.jit
    def __call__(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer,
                 weights: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(logits, bias, ids, weights).launch(
                grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer,
               weights: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        lane = Int32(cute.arch.thread_idx()[0])
        slots = self.experts // 32
        scores = cute.make_rmem_tensor((slots,), Float32)
        raw = cute.make_rmem_tensor((slots,), Float32)
        selected = cute.make_rmem_tensor((self.topk,), Float32)
        for j in cutlass.range_constexpr(slots):
            expert = lane + Int32(j * 32)
            value = Float32(logits[row * Int64(self.experts) + Int64(expert)])
            score = div_rn_f32(Float32(1), Float32(1) + cute.math.exp(-value, fastmath=False)) + Float32(bias[expert])
            scores[j] = score
            raw[j] = value
        maximum = Float32(-float("inf"))
        for choice in cutlass.range_constexpr(self.topk):
            best = Float32(-float("inf"))
            index = Int32(2147483647)
            for j in cutlass.range_constexpr(slots):
                candidate = lane + Int32(j * 32)
                if scores[j] > best or (scores[j] == best and candidate < index):
                    best, index = scores[j], candidate
            for shift in cutlass.range_constexpr(5):
                other = Float32(cute.arch.shuffle_sync_bfly(best, offset=1 << shift))
                other_id = Int32(cute.arch.shuffle_sync_bfly(index, offset=1 << shift))
                if other > best or (other == best and other_id < index):
                    best, index = other, other_id
            value = Float32(0)
            for j in cutlass.range_constexpr(slots):
                if lane + Int32(j * 32) == index:
                    value = raw[j]
                    scores[j] = Float32(-float("inf"))
            for shift in cutlass.range_constexpr(5):
                value = value + Float32(cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
            logsigmoid = fmin_f32(value, Float32(0)) - cute.math.log(
                Float32(1) + cute.math.exp(-fabs_f32(value), fastmath=False), fastmath=False)
            selected[choice] = logsigmoid
            maximum = fmax_f32(maximum, logsigmoid)
            if lane == Int32(0):
                ids[row * Int64(self.topk) + Int64(choice)] = index
        denominator = Float32(0)
        for choice in cutlass.range_constexpr(self.topk):
            selected[choice] = cute.math.exp(selected[choice] - maximum, fastmath=False)
            denominator = denominator + selected[choice]
        if lane == Int32(0):
            for choice in cutlass.range_constexpr(self.topk):
                weights[row * Int64(self.topk) + Int64(choice)] = (
                    div_rn_f32(selected[choice], denominator) * Float32(self.route_scale)
                )


def compile_inkling2_mini_fast_norm_aot(*, width: int = HIDDEN, eps: float = 1e-6, max_rows: int = 4096):
    if (type(width) is not int or width != HIDDEN or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling 2 Mini fast norm geometry")
    return compile_program(
        _MiniFastNorm(width, eps), name="inkling2_mini_fast_norm",
        operands=(
            Operand("x", torch.float32, f"[rows,{width}]"),
            Operand("weight", torch.bfloat16, f"[{width}]"),
            Operand("out", torch.bfloat16, f"[rows,{width}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(width, eps, max_rows),
        geometry=dict(width=width, eps=eps, output_dtype="torch.bfloat16", max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_mini_residual_norm_aot(
    *, hidden: int = HIDDEN, queries: int = 1, max_candidates: int = 5,
    eps: float = 1e-6, max_rows: int = 4096,
):
    if (type(hidden) is not int or hidden != HIDDEN or queries not in (1, 3)
            or max_candidates != 5 or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling 2 Mini residual_norm geometry")
    return compile_program(
        _MiniResidualNorm(hidden, queries, max_candidates, eps),
        name="inkling2_mini_residual_norm",
        operands=(
            Operand("arena", torch.float32, f"[{max_candidates},arena_rows,{hidden}]"),
            Operand("query", torch.float32, f"[{queries},{hidden}]"),
            Operand("weight", torch.bfloat16, f"[{queries},{hidden}]"),
            Operand("out", torch.bfloat16, f"[{queries},rows,{hidden}]", "out"),
        ),
        scalars=(Scalar("rows"), Scalar("arena_rows"), Scalar("candidates")),
        key=(hidden, queries, max_candidates, eps, max_rows),
        geometry=dict(hidden=hidden, queries=queries, max_candidates=max_candidates,
                      eps=eps, output_dtype="torch.bfloat16", max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_mini_loop_blend_norm_aot(
    *, hidden: int = HIDDEN, slots: int = 1, broadcast_dest: bool = False,
    broadcast_weight: bool = False, eps: float = 1e-6, max_rows: int = 4096,
):
    if (type(hidden) is not int or hidden != HIDDEN or slots not in (1, 3)
            or (slots == 1 and (broadcast_dest or broadcast_weight))
            or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling 2 Mini fast loop_blend_norm geometry")
    dest_slots = 1 if broadcast_dest else slots
    w_slots = 1 if broadcast_weight else slots
    return compile_program(
        _MiniLoopBlendNorm(hidden, slots, broadcast_dest, broadcast_weight, eps),
        name="inkling2_mini_loop_blend_norm_fast",
        operands=(
            Operand("dest", torch.float32, f"[{dest_slots},rows,{hidden}]"),
            Operand("msg", torch.float32, f"[{slots},rows,{hidden}]"),
            Operand("forget_gain", torch.float32, f"[{slots},rows,{2 * hidden}]"),
            Operand("weight", torch.bfloat16, f"[{w_slots},{hidden}]"),
            Operand("out", torch.bfloat16, f"[{slots},rows,{hidden}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(hidden, slots, broadcast_dest, broadcast_weight, eps, max_rows),
        geometry=dict(hidden=hidden, slots=slots, broadcast_dest=broadcast_dest,
                      broadcast_weight=broadcast_weight, eps=eps, max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_mini_residual_pair_norm_aot(
    *, hidden: int = HIDDEN, max_candidates: int = 5, eps: float = 1e-6, max_rows: int = 4096,
):
    if (type(hidden) is not int or hidden != HIDDEN or max_candidates != 5
            or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling 2 Mini residual_pair_norm geometry")
    return compile_program(
        _MiniResidualPairNorm(hidden, max_candidates, eps),
        name="inkling2_mini_residual_pair_norm",
        operands=(
            Operand("arena", torch.float32, f"[{max_candidates},arena_rows,{hidden}]"),
            Operand("query", torch.float32, f"[1,{hidden}]"),
            Operand("msg", torch.float32, f"[rows,{hidden}]"),
            Operand("mixed_out", torch.float32, f"[rows,{hidden}]", "out"),
            Operand("paired_out", torch.bfloat16, f"[rows,{2 * hidden}]", "out"),
        ),
        scalars=(Scalar("rows"), Scalar("arena_rows"), Scalar("candidates")),
        key=(hidden, max_candidates, eps, max_rows),
        geometry=dict(hidden=hidden, slots=1, max_candidates=max_candidates, eps=eps, max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_mini_attn_cross_pair_norm_aot(
    *, hidden: int = HIDDEN, broadcast_dest: bool = False, max_candidates: int = 5,
    eps: float = 1e-6, max_rows: int = 4096,
):
    if (type(hidden) is not int or hidden != HIDDEN or max_candidates != 5
            or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling 2 Mini attn_cross_pair_norm geometry")
    dest_slots = 1 if broadcast_dest else 3
    return compile_program(
        _MiniAttnCrossPairNorm(hidden, broadcast_dest, max_candidates, eps),
        name="inkling2_mini_attn_cross_pair_norm",
        operands=(
            Operand("arena", torch.float32, f"[{max_candidates},arena_rows,{hidden}]"),
            Operand("attn_query", torch.float32, f"[{dest_slots},{hidden}]"),
            Operand("prev_arena", torch.float32, f"[2,arena_rows,{hidden}]"),
            Operand("cross_query", torch.float32, f"[4,{hidden}]"),
            Operand("dest_out", torch.float32, f"[{dest_slots},rows,{hidden}]", "out"),
            Operand("msg_out", torch.float32, f"[4,rows,{hidden}]", "out"),
            Operand("paired_out", torch.bfloat16, f"[3,rows,{2 * hidden}]", "out"),
        ),
        scalars=(Scalar("rows"), Scalar("arena_rows"), Scalar("candidates")),
        key=(hidden, broadcast_dest, max_candidates, eps, max_rows),
        geometry=dict(hidden=hidden, broadcast_dest=broadcast_dest,
                      max_candidates=max_candidates, eps=eps, max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_mini_attn_prep_aot(
    *, local: bool, fp8: bool = False, replay: bool = False, max_rows: int = 64,
):
    if (local and fp8) or type(max_rows) is not int or not 1 <= max_rows <= 64:
        raise ValueError("invalid Inkling 2 Mini attn_prep geometry")
    w_shape = f"[{KV_WIDTH},{TAPS}]" if local else f"[{TAPS},{KV_WIDTH}]"
    cache_dtype = torch.uint8 if fp8 else torch.bfloat16
    cache_shape = f"[slots,{2 * KV_WIDTH + 2 * KV_HEADS * 4}]" if fp8 else f"[slots,{2 * KV_WIDTH}]"
    return compile_program(
        _MiniAttnPrep(local=local, fp8=fp8, replay=replay, max_rows=max_rows),
        name=f"inkling2_mini_attn_prep_{'local' if local else 'global'}_{'fp8' if fp8 else 'bf16'}_{'replay' if replay else 'strided'}",
        operands=(
            Operand("q", torch.bfloat16, f"[rows,{HEADS},{HEAD_DIM}]"),
            Operand("q_scale", torch.bfloat16, f"[{HEADS},{HEAD_DIM}]"),
            Operand("r", torch.bfloat16, f"[rows,{HEADS},{REL_DIM}]"),
            Operand("r_scale", torch.bfloat16, f"[{HEADS},{REL_DIM}]"),
            Operand("positions", torch.int64, "[rows]", align=8),
            Operand("q_out", torch.bfloat16, f"[rows,{HEADS},{HEAD_DIM}]", "out"),
            Operand("rel_input_out", torch.bfloat16, f"[rows,{HEADS},{REL_PADDED}]", "out"),
            Operand("k_bf16", torch.bfloat16, f"[rows,{KV_WIDTH}]"),
            Operand("k_weight", torch.bfloat16, w_shape),
            Operand("k_dynamic", torch.float32, f"[rows,{KV_HEADS},{TAPS}]"),
            Operand("k_history", torch.float32, f"[slots,{TAPS - 1},{KV_WIDTH}]", "inout"),
            Operand("v_bf16", torch.bfloat16, f"[rows,{KV_WIDTH}]"),
            Operand("v_weight", torch.bfloat16, w_shape),
            Operand("v_dynamic", torch.float32, f"[rows,{KV_HEADS},{TAPS}]"),
            Operand("v_history", torch.float32, f"[slots,{TAPS - 1},{KV_WIDTH}]", "inout"),
            Operand("first", torch.int32, "[rows]", align=4),
            Operand("conv_slots", torch.int64, "[rows]", align=8),
            Operand("cache_slots", torch.int64, "[rows]", align=8),
            Operand("cache", cache_dtype, cache_shape, "inout"),
            Operand("k_replay", torch.float32, f"[{max_rows},{KV_WIDTH}]", "out"),
            Operand("v_replay", torch.float32, f"[{max_rows},{KV_WIDTH}]", "out"),
        ),
        scalars=(
            Scalar("rows"),
            Scalar("history_slot_stride", "int64"),
            Scalar("history_lag_stride", "int64"),
            Scalar("replay_row_stride", "int64"),
        ),
        key=(bool(local), bool(fp8), bool(replay), max_rows),
        geometry=dict(
            local=bool(local), fp8=bool(fp8), replay=bool(replay),
            heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            rel_dim=REL_DIM, rel_padded=REL_PADDED, taps=TAPS, max_rows=max_rows,
        ),
        doc=__doc__,
    )


def compile_inkling2_mini_post_attn_conv_aot(
    *, starts_block: bool, replay: bool = False, max_rows: int = 64,
):
    if type(max_rows) is not int or not 1 <= max_rows <= 64:
        raise ValueError("invalid Inkling 2 Mini post_attn_conv geometry")
    return compile_program(
        _MiniPostAttnConv(starts_block=starts_block, replay=replay, max_rows=max_rows),
        name=f"inkling2_mini_post_attn_conv_{'start' if starts_block else 'cont'}_{'replay' if replay else 'strided'}",
        operands=(
            Operand("x_bf16", torch.bfloat16, f"[rows,{HIDDEN}]"),
            Operand("weight", torch.bfloat16, f"[{HIDDEN},{TAPS}]"),
            Operand("history", torch.float32, f"[slots,{TAPS - 1},{HIDDEN}]", "inout"),
            Operand("first", torch.int32, "[rows]", align=4),
            Operand("slots", torch.int64, "[rows]", align=8),
            Operand("partial", torch.float32, f"[rows,{HIDDEN}]"),
            Operand("arena", torch.float32, f"[5,arena_rows,{HIDDEN}]", "inout"),
            Operand("out_base", torch.float32, f"[rows,{HIDDEN}]", "out"),
            Operand("replay", torch.float32, f"[{max_rows},{HIDDEN}]", "out"),
        ),
        scalars=(
            Scalar("rows"),
            Scalar("arena_rows"),
            Scalar("mlp_cand_idx"),
            Scalar("history_slot_stride", "int64"),
            Scalar("history_lag_stride", "int64"),
            Scalar("replay_row_stride", "int64"),
        ),
        key=(bool(starts_block), bool(replay), max_rows),
        geometry=dict(
            hidden=HIDDEN, taps=TAPS, starts_block=bool(starts_block),
            replay=bool(replay), max_rows=max_rows,
        ),
        doc=__doc__,
    )


def compile_inkling2_mini_post_mlp_conv_residual_aot(*, replay: bool = False, max_rows: int = 64):
    if type(max_rows) is not int or not 1 <= max_rows <= 64:
        raise ValueError("invalid Inkling 2 Mini post_mlp_conv_residual geometry")
    return compile_program(
        _MiniPostMlpConvResidual(replay=replay, max_rows=max_rows),
        name=f"inkling2_mini_post_mlp_conv_residual_{'replay' if replay else 'strided'}",
        operands=(
            Operand("x_bf16", torch.bfloat16, f"[rows,{HIDDEN}]"),
            Operand("weight", torch.bfloat16, f"[{HIDDEN},{TAPS}]"),
            Operand("history", torch.float32, f"[slots,{TAPS - 1},{HIDDEN}]", "inout"),
            Operand("first", torch.int32, "[rows]", align=4),
            Operand("slots", torch.int64, "[rows]", align=8),
            Operand("base", torch.float32, f"[rows,{HIDDEN}]"),
            Operand("partial", torch.float32, f"[rows,{HIDDEN}]", "out"),
            Operand("arena", torch.float32, f"[5,arena_rows,{HIDDEN}]", "inout"),
            Operand("replay", torch.float32, f"[{max_rows},{HIDDEN}]", "out"),
        ),
        scalars=(
            Scalar("rows"),
            Scalar("arena_rows"),
            Scalar("mlp_cand_idx"),
            Scalar("history_slot_stride", "int64"),
            Scalar("history_lag_stride", "int64"),
            Scalar("replay_row_stride", "int64"),
        ),
        key=(bool(replay), max_rows),
        geometry=dict(hidden=HIDDEN, taps=TAPS, replay=bool(replay), max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_mini_fast_router_aot(
    *, experts: int = 128, topk: int = 8, route_scale: float = 8.0, max_rows: int = 4096,
):
    if (experts != 128 or topk != 8 or not math.isfinite(route_scale) or route_scale <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling 2 Mini fast router geometry")
    return compile_program(
        _MiniFastRouter(experts, topk, route_scale),
        name="inkling2_mini_fast_router",
        operands=(
            Operand("logits", torch.float32, f"[rows,{experts}]"),
            Operand("bias", torch.float32, f"[{experts}]"),
            Operand("ids", torch.int32, f"[rows,{topk}]", "out"),
            Operand("weights", torch.float32, f"[rows,{topk}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(experts, topk, route_scale, max_rows),
        geometry=dict(experts=experts, topk=topk, route_scale=route_scale, max_rows=max_rows),
        doc=__doc__,
    )
