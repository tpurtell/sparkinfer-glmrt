"""Compact raw-logit statistics for greedy relaxed speculative acceptance.

``compile_relaxed_acceptance_stats_f32_aot`` exports the raw-pointer ABI::

    (logits, proposals, stats, argmax_ids, argmax_scores, status,
     rows: int32, vocab: int32, row_stride: int64, stream)

All pointers are device-resident and four-byte aligned. ``logits`` is F32
with contiguous columns and ``row_stride`` measured in F32 elements; all
other buffers are dense. ``stats[rows, 3]`` contains ``[p_draft, p_argmax, H]``
with natural-log entropy. ``argmax_ids`` and ``status`` are U32 vectors;
``argmax_scores`` is an F32 vector containing the raw winning logit.
The smallest token ID wins an argmax tie. No logits are changed or downloaded.

Status: 0 VALID, 1 SKIPPED, 2 NONFINITE, 3 BAD_PROPOSAL. A UINT32_MAX
proposal requests checked argmax only: the softmax/entropy scan is skipped,
statistics are zero, and the winning ID/raw score are still returned. This
supports exact and bonus rows mixed with relaxed rows in one terminal call.
NONFINITE rejects any NaN or infinity, including negative infinity used for
grammar masking; the caller must reject masked rows before this raw-logit ABI.
NONFINITE writes zero stats, UINT32_MAX ID and NaN score. BAD_PROPOSAL writes
zero stats, UINT32_MAX ID and zero score without reading logits, before
finiteness checking. Nonfinite rows fail even when their proposal is sentinel.
The caller validates rows >= 0, vocab > 0 and row_stride >= vocab, supplies
nonoverlapping output buffers, and skips the launch when rows == 0.

No scratch, allocations, graph capture, host synchronization, or live-size
specializations are required. Compile/export once before requests; launch
against resident logits on their producing stream.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _warp_sum


UINT32_MAX = 0xFFFFFFFF
VALID = 0
SKIPPED = 1
NONFINITE = 2
BAD_PROPOSAL = 3


@cute.jit
def _view(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


class RelaxedAcceptanceStatsF32:
    @cute.jit
    def __call__(self, logits: cute.Pointer, proposals: cute.Pointer, stats: cute.Pointer,
                 argmax_ids: cute.Pointer, argmax_scores: cute.Pointer, status: cute.Pointer,
                 rows: Int32, vocab: Int32, row_stride: Int64, stream: cuda.CUstream):
        self.kernel(logits, proposals, stats, argmax_ids, argmax_scores, status, vocab, row_stride).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, proposals: cute.Pointer, stats: cute.Pointer,
               argmax_ids: cute.Pointer, argmax_scores: cute.Pointer, status: cute.Pointer,
               vocab: Int32, row_stride: Int64):
        logits, proposals, stats = _view(logits), _view(proposals), _view(stats)
        argmax_ids, argmax_scores, status = _view(argmax_ids), _view(argmax_scores), _view(status)
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        lane, warp = tid % Int32(32), tid // Int32(32)
        smem = cutlass.utils.SmemAllocator()
        maxima = smem.allocate_tensor(Float32, cute.make_layout((8,)))
        indices = smem.allocate_tensor(Uint32, cute.make_layout((8,)))
        invalids = smem.allocate_tensor(Int32, cute.make_layout((8,)))
        masses = smem.allocate_tensor(Float32, cute.make_layout((8,)))
        moments = smem.allocate_tensor(Float32, cute.make_layout((8,)))
        proposal = Uint32(proposals[row])
        code = Uint32(VALID)
        best_id = Uint32(UINT32_MAX)
        p_draft, p_max, entropy = Float32(0.0), Float32(0.0), Float32(0.0)
        raw_score = Float32(0.0)
        if (proposal != Uint32(UINT32_MAX)) & (proposal >= Uint32(vocab)):
            code = Uint32(BAD_PROPOSAL)
        else:
            base = row * row_stride
            best = Float32(float("-inf"))
            invalid = Int32(0)
            for col in cutlass.range(Int64(tid), Int64(vocab), Int64(256)):
                value = Float32(logits[base + col])
                if (value != value) | (value == Float32(float("inf"))) | (value == Float32(float("-inf"))):
                    invalid = Int32(1)
                if (value > best) | ((value == best) & (Uint32(col) < best_id)):
                    best = Float32(value)
                    best_id = Uint32(col)
            for shift in cutlass.range_constexpr(5):
                other = Float32(cute.arch.shuffle_sync_bfly(best, offset=1 << shift))
                other_id = Uint32(cute.arch.shuffle_sync_bfly(best_id, offset=1 << shift))
                invalid = invalid | Int32(cute.arch.shuffle_sync_bfly(invalid, offset=1 << shift))
                if (other > best) | ((other == best) & (other_id < best_id)):
                    best = Float32(other)
                    best_id = Uint32(other_id)
            if lane == Int32(0):
                maxima[warp] = best
                indices[warp] = best_id
                invalids[warp] = invalid
            cute.arch.sync_threads()
            for w in cutlass.range_constexpr(8):
                other, other_id = Float32(maxima[w]), Uint32(indices[w])
                invalid = invalid | Int32(invalids[w])
                if (other > best) | ((other == best) & (other_id < best_id)):
                    best = Float32(other)
                    best_id = Uint32(other_id)
            raw_score = best
            if invalid != Int32(0):
                code = Uint32(NONFINITE)
                best_id = Uint32(UINT32_MAX)
                raw_score = Float32(float("nan"))
            elif proposal == Uint32(UINT32_MAX):
                code = Uint32(SKIPPED)
            else:
                mass, moment = Float32(0.0), Float32(0.0)
                for col in cutlass.range(Int64(tid), Int64(vocab), Int64(256)):
                    delta = Float32(Float32(logits[base + col]) - best)
                    weight = cute.math.exp(delta, fastmath=False)
                    mass = Float32(mass + weight)
                    # Finite F32 operands can subtract to -inf. Its weight is
                    # zero, so omit 0 * -inf rather than poisoning entropy.
                    if weight > Float32(0.0):
                        moment = Float32(moment + Float32(weight * delta))
                mass, moment = _warp_sum(mass), _warp_sum(moment)
                if lane == Int32(0):
                    masses[warp], moments[warp] = mass, moment
                cute.arch.sync_threads()
                if tid == Int32(0):
                    mass, moment = Float32(0.0), Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        mass = Float32(mass + Float32(masses[w]))
                        moment = Float32(moment + Float32(moments[w]))
                    p_max = div_rn_f32(Float32(1.0), mass)
                    delta = Float32(Float32(logits[base + Int64(proposal)]) - best)
                    p_draft = div_rn_f32(cute.math.exp(delta, fastmath=False), mass)
                    entropy = Float32(cute.math.log(mass, fastmath=False) - div_rn_f32(moment, mass))
        if tid == Int32(0):
            stats[row * Int64(3)] = p_draft
            stats[row * Int64(3) + Int64(1)] = p_max
            stats[row * Int64(3) + Int64(2)] = entropy
            argmax_ids[row] = best_id
            argmax_scores[row] = raw_score
            status[row] = code


def compile_relaxed_acceptance_stats_f32_aot():
    """Compile one geometry-independent compact-statistics program, without scratch."""
    return compile_program(
        RelaxedAcceptanceStatsF32(), name="relaxed_acceptance_stats_f32",
        operands=(
            Operand("logits", torch.float32, "[rows,row_stride]", align=4),
            Operand("proposals", torch.uint32, "[rows]", align=4),
            Operand("stats", torch.float32, "[rows,3]", "out", align=4),
            Operand("argmax_ids", torch.uint32, "[rows]", "out", align=4),
            Operand("argmax_scores", torch.float32, "[rows]", "out", align=4),
            Operand("status", torch.uint32, "[rows]", "out", align=4),
        ),
        scalars=(Scalar("rows"), Scalar("vocab"), Scalar("row_stride", "int64")),
        key=(), geometry=dict(block_threads=256, stats_width=3, skip_proposal=UINT32_MAX),
        doc=__doc__,
    )
