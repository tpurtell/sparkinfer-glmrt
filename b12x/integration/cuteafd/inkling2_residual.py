"""Inkling normalized-value attention residuals, independent of prefix mutation.

#1915 attn_res: every candidate is RMS-normalized in F32; the F32 query is
scaled by hidden**-0.5; a softmax over candidate dot products weights those
normalized values. No BF16 intermediate and no post-mix layer norm. Supports
one (local/MLP/head) or three (global Q/K/V) queries. The caller appends a
completed six-layer partial block and updates partial state outside this pure
mix. Existing Kimi raw-value/BF16/update semantics are deliberately unchanged.

arena [max_candidates,arena_rows,hidden] F32; candidates/rows/arena_rows are
runtime. Optional addend applies only to the final candidate, before its norm.
queries [queries,hidden] F32; out [queries,rows,hidden] F32. No scratch or state
writes, safe to replay with changed row/candidate counts and stable owners.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64
import torch

from b12x._lib.intrinsics import div_rn_f32, fmax_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _block_sum, _reduction_storage, _rsqrt


class _Residual:
    threads = 256

    def __init__(self, hidden, queries, maximum, addend, eps):
        self.hidden, self.queries, self.maximum = hidden, queries, maximum
        self.addend, self.eps = addend, float(eps)
        self.qscale = hidden ** -.5

    @cute.jit
    def __call__(self, arena: cute.Pointer, addend: cute.Pointer, query: cute.Pointer,
                 out: cute.Pointer, rows: Int32, arena_rows: Int32, candidates: Int32,
                 stream: cuda.CUstream):
        self.kernel(arena, addend, query, out, rows, arena_rows, candidates).launch(
            grid=(rows, self.queries, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, arena: cute.Pointer, addend: cute.Pointer, query: cute.Pointer,
               out: cute.Pointer, rows: Int32, arena_rows: Int32, candidates: Int32):
        row = Int64(cute.arch.block_idx()[0])
        q = Int64(cute.arch.block_idx()[1])
        tid = Int64(cute.arch.thread_idx()[0])
        stride = Int64(arena_rows) * Int64(self.hidden)
        base = row * Int64(self.hidden)
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, self.threads // 32))
        sums = storage.sums.get_tensor(cute.make_layout((2, self.threads // 32), stride=(self.threads // 32, 1)))
        values = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        weights = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        mixed = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        scores = cute.make_rmem_tensor((self.maximum,), Float32)
        inverse = cute.make_rmem_tensor((self.maximum,), Float32)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            weights[j] = Float32(query[q * Int64(self.hidden) + col]) * Float32(self.qscale)
            mixed[j] = Float32(0)
        maximum = Float32(-float("inf"))
        for c in cutlass.range_constexpr(self.maximum):
            scores[c], inverse[c] = Float32(0), Float32(0)
            if Int32(c) < candidates:
                square = Float32(0)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    col = tid + Int64(j * self.threads)
                    value = Float32(arena[Int64(c) * stride + base + col])
                    if cutlass.const_expr(self.addend):
                        if Int32(c) == candidates - Int32(1):
                            value = value + Float32(addend[base + col])
                    values[j] = value
                    square = square + value * value
                total = _block_sum(square, sums, Int32(0), self.threads // 32)
                inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + Float32(self.eps))
                inverse[c] = inv
                dot = Float32(0)
                for j in cutlass.range_constexpr(self.hidden // self.threads):
                    dot = dot + (values[j] * inv) * weights[j]
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
                    if cutlass.const_expr(self.addend):
                        if Int32(c) == candidates - Int32(1):
                            value = value + Float32(addend[base + col])
                    mixed[j] = mixed[j] + (value * inverse[c]) * probability
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            out[(q * Int64(rows) + row) * Int64(self.hidden) + col] = mixed[j]


def compile_inkling2_residual_aot(*, hidden: int = 4608, queries: int = 1,
                                 max_candidates: int = 9, addend: bool = False,
                                 eps: float = 1e-6, max_rows: int = 4096):
    import math
    if (type(hidden) is not int or hidden <= 0 or hidden % 256
            or queries not in (1, 3) or not 1 <= max_candidates <= 9
            or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling residual geometry/capacity")
    return compile_program(
        _Residual(hidden, queries, max_candidates, addend, eps), name="inkling2_residual",
        operands=(Operand("arena", torch.float32, f"[{max_candidates},arena_rows,{hidden}]"),
                  Operand("addend", torch.float32, f"[rows,{hidden}]"),
                  Operand("query", torch.float32, f"[{queries},{hidden}]"),
                  Operand("out", torch.float32, f"[{queries},rows,{hidden}]", "out")),
        scalars=(Scalar("rows"), Scalar("arena_rows"), Scalar("candidates")),
        key=(hidden, queries, max_candidates, addend, eps, max_rows),
        geometry=dict(hidden=hidden, queries=queries, max_candidates=max_candidates,
                      addend=addend, eps=eps, max_rows=max_rows, normalize_values=True), doc=__doc__)
