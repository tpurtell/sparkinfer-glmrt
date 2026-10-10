"""Token-sharded MLA building blocks; no peer exchange or engine policy.

Candidates are FP32 scores and int32 *global logical* indices, ordered by
(score descending, logical index ascending), padded with (-inf, -1). Index
pages contain 64 keys for all three families (GLM Flash keys are pools of
four tokens). A compact shard table enumerates every second logical page.
Partials are normalized BF16 outputs with FP32 base-2 LSE, without a sink.
V4's natural-log per-head sink is added exactly once by lse_combine2.

All quantities that change per request are runtime scalars. Buffers belong
to the caller; launches neither allocate nor resolve kernels.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64

from b12x.attention._shared.cute.ops import LOG2_E
from ._common import GLM53, GLM53_FLASH, FLASH, Operand, Scalar, compile_program

__all__ = ["compile_scored_index_topk_aot", "compile_dsa_candidate_merge_aot",
           "compile_sparse_mla_partial_aot", "compile_lse_combine2_aot",
           "compile_paged_staging_gather_aot", "sparse_mla_partial_split_plan"]


@cute.jit
def _pair_before(av: Float32, a: Int32, bv: Float32, b: Int32):
    return (a >= Int32(0)) and ((b < Int32(0)) or (av > bv) or ((av == bv) and (a < b)))


@cute.jit
def _sort_pairs(values: cute.Tensor, indices: cute.Tensor,
                size: cutlass.Constexpr, by_score: cutlass.Constexpr):
    """Bitonic order: warp shuffles for low strides, shared exchange otherwise."""
    tid = Int32(cute.arch.thread_idx()[0])
    rv = cute.make_rmem_tensor((size // 256,), Float32)
    ri = cute.make_rmem_tensor((size // 256,), Int32)
    nv = cute.make_rmem_tensor((size // 256,), Float32)
    ni = cute.make_rmem_tensor((size // 256,), Int32)
    for slot in cutlass.range_constexpr(size // 256):
        rv[slot], ri[slot] = values[tid + Int32(slot * 256)], indices[tid + Int32(slot * 256)]
    for level in cutlass.range_constexpr(1, size.bit_length()):
        for step in cutlass.range_constexpr(level - 1, -1, -1):
            if cutlass.const_expr(5 <= step < 8):
                for slot in cutlass.range_constexpr(size // 256):
                    values[tid + Int32(slot * 256)], indices[tid + Int32(slot * 256)] = rv[slot], ri[slot]
                cute.arch.sync_threads()
            for slot in cutlass.range_constexpr(size // 256):
                i = tid + Int32(slot * 256)
                a, av = ri[slot], rv[slot]
                if cutlass.const_expr(step < 5):
                    b = cute.arch.shuffle_sync_bfly(a, offset=1 << step)
                    bv = cute.arch.shuffle_sync_bfly(av, offset=1 << step)
                elif cutlass.const_expr(step >= 8):
                    partner = slot ^ (1 << (step - 8))
                    b, bv = ri[partner], rv[partner]
                else:
                    j = i ^ Int32(1 << step)
                    b, bv = Int32(indices[j]), Float32(values[j])
                before = (a >= Int32(0)) and ((b < Int32(0)) or (a < b))
                if cutlass.const_expr(by_score):
                    before = (a >= Int32(0)) and ((b < Int32(0)) or
                             (av > bv) or ((av == bv) and (a < b)))
                take_first = ((i & Int32(1 << level)) == Int32(0)) == ((i & Int32(1 << step)) == Int32(0))
                ni[slot], nv[slot] = a, av
                if before != take_first:
                    ni[slot], nv[slot] = b, bv
            for slot in cutlass.range_constexpr(size // 256):
                ri[slot], rv[slot] = ni[slot], nv[slot]
            if cutlass.const_expr(5 <= step < 8):
                cute.arch.sync_threads()
    for slot in cutlass.range_constexpr(size // 256):
        values[tid + Int32(slot * 256)], indices[tid + Int32(slot * 256)] = rv[slot], ri[slot]
    cute.arch.sync_threads()


@cute.jit
def _sort_indices(indices: cute.Tensor, size: cutlass.Constexpr):
    """Ascending valid indices; high strides exchange thread-local registers."""
    tid = Int32(cute.arch.thread_idx()[0])
    ri = cute.make_rmem_tensor((size // 256,), Int32)
    ni = cute.make_rmem_tensor((size // 256,), Int32)
    for slot in cutlass.range_constexpr(size // 256):
        idx = Int32(indices[tid + Int32(slot * 256)])
        ri[slot] = idx if idx >= Int32(0) else Int32(2147483647)
    for level in cutlass.range_constexpr(1, size.bit_length()):
        for step in cutlass.range_constexpr(level - 1, -1, -1):
            if cutlass.const_expr(5 <= step < 8):
                for slot in cutlass.range_constexpr(size // 256):
                    indices[tid + Int32(slot * 256)] = ri[slot]
                cute.arch.sync_threads()
            for slot in cutlass.range_constexpr(size // 256):
                i = tid + Int32(slot * 256)
                a = ri[slot]
                if cutlass.const_expr(step < 5):
                    b = cute.arch.shuffle_sync_bfly(a, offset=1 << step)
                elif cutlass.const_expr(step >= 8):
                    b = ri[slot ^ (1 << (step - 8))]
                else:
                    b = Int32(indices[i ^ Int32(1 << step)])
                take_first = ((i & Int32(1 << level)) == Int32(0)) == ((i & Int32(1 << step)) == Int32(0))
                if take_first:
                    ni[slot] = min(a, b)
                else:
                    ni[slot] = max(a, b)
            for slot in cutlass.range_constexpr(size // 256):
                ri[slot] = ni[slot]
            if cutlass.const_expr(5 <= step < 8):
                cute.arch.sync_threads()
    for slot in cutlass.range_constexpr(size // 256):
        idx = ri[slot]
        indices[tid + Int32(slot * 256)] = idx if idx != Int32(2147483647) else Int32(-1)
    cute.arch.sync_threads()


class _ScoreOrder:
    def __init__(self, k: int):
        self.k = int(k)

    @cute.jit
    def __call__(self, scores: cute.Pointer, indices: cute.Pointer,
                 out_scores: cute.Pointer, out_indices: cute.Pointer,
                 rows: Int32, page_stride: Int32, page_offset: Int32,
                 stream: cuda.CUstream):
        self.kernel(scores, indices, out_scores, out_indices, page_stride,
                    page_offset).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, scores: cute.Pointer, indices: cute.Pointer,
               out_scores: cute.Pointer, out_indices: cute.Pointer,
               page_stride: Int32, page_offset: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        row = Int64(cute.arch.block_idx()[0])
        smem = utils.SmemAllocator()
        sv = smem.allocate_tensor(Float32, cute.make_layout((self.k,)), 16)
        si = smem.allocate_tensor(Int32, cute.make_layout((self.k,)), 16)
        for slot in cutlass.range_constexpr(self.k // 256):
            i = tid + Int32(slot * 256)
            idx = Int32(indices[row * Int64(self.k) + Int64(i)])
            if idx >= Int32(0):
                idx = ((idx // Int32(64)) * page_stride + page_offset) * Int32(64) + idx % Int32(64)
            si[i] = idx
            sv[i] = Float32(scores[row * Int64(self.k) + Int64(i)]) if idx >= Int32(0) else Float32(float("-inf"))
        cute.arch.sync_threads()
        _sort_pairs(sv, si, self.k, True)
        for slot in cutlass.range_constexpr(self.k // 256):
            i = tid + Int32(slot * 256)
            out_indices[row * Int64(self.k) + Int64(i)] = si[i]
            out_scores[row * Int64(self.k) + Int64(i)] = sv[i]


class _ScoredIndex:
    def __init__(self, g, max_rows: int, max_pages: int, mode: str):
        from .dsv4_indexer import _TopK
        self.topk = _TopK(g, max_rows, max_pages, mode, heads=g.index_heads, scored_logical=True)
        self.order = _ScoreOrder(self.topk.topk)

    @cute.jit
    def __call__(self, q: cute.Pointer, weights: cute.Pointer, cache: cute.Pointer,
                 table: cute.Pointer, lengths: cute.Pointer, indices: cute.Pointer,
                 scores: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_width: Int32, table_stride: Int32, page_stride: Int32,
                 page_offset: Int32, stream: cuda.CUstream):
        self.topk(q, weights, cache, table, lengths, indices, scratch,
                  rows, table_width, table_stride, stream)
        values = cute.make_ptr(Float32, Int64(scratch.toint()) + Int64(self.topk.layout.topk_values_offset_bytes),
                               cute.AddressSpace.gmem, assumed_align=16)
        self.order(values, indices, scores, indices, rows, page_stride, page_offset, stream)


def compile_scored_index_topk_aot(g=GLM53, *, max_rows: int, max_pages: int, mode="decode"):
    """ABI: q FP8 [R,I,128], weights f32 [R,I], cache u8 [P,8448],
    table i32 [R,stride] (prefill: shared [width]), lengths i32 [R] in compact
    shard coordinates, indices i32 [R,K] out, scores f32 [R,K] out, scratch u8.
    Scalars: rows, table_width, table_stride, page_stride, page_offset.
    Use (2,rank) for an interleaved half table and (1,0) for a full table.
    GLM Flash K=512 pools; V4 K=512/1024; GLM K=2048.
    """
    from .glmf import _PoolTopK
    if g is GLM53_FLASH or hasattr(g, "index_kpool"):
        g = _PoolTopK(g.index_topk // g.index_kpool, g.index_heads)
    if not hasattr(g, "index_heads"):
        g = _PoolTopK(g.index_topk, 64)
    if mode not in ("decode", "prefill") or min(max_rows, max_pages) <= 0:
        raise ValueError("positive capacities and mode=decode|prefill required")
    launch = _ScoredIndex(g, int(max_rows), int(max_pages), mode)
    k, h = g.index_topk, g.index_heads
    operands = (Operand("q", torch.float8_e4m3fn, f"[rows,{h},128]"),
                Operand("weights", torch.float32, f"[rows,{h}]"),
                Operand("cache", torch.uint8, "[pages,8448]"),
                Operand("table", torch.int32, "[rows,table_stride]", align=4),
                Operand("lengths", torch.int32, "[rows]", align=4),
                Operand("indices", torch.int32, f"[rows,{k}]", "out", align=4),
                Operand("scores", torch.float32, f"[rows,{k}]", "out", align=4),
                Operand("scratch", torch.uint8, "[scratch_bytes]", "scratch"))
    return compile_program(launch, name="scored_index_topk", operands=operands,
                           scalars=tuple(Scalar(x) for x in ("rows", "table_width", "table_stride", "page_stride", "page_offset")),
                           key=(launch.topk.key(), "scored-logical", mode),
                           geometry={"topk": k, "heads": h, "route": launch.topk.route,
                                     "stream_scorer_ctas": int(launch.topk.layout.stream_scorer_ctas)},
                           scratch={"scratch": lambda rows: launch.topk.layout.nbytes}, doc=compile_scored_index_topk_aot.__doc__)


def compile_dsa_candidate_merge_aot(*, topk: int, unit_rows: int = 64):
    """Merge disjoint shards' candidates: scores f32 and indices i32 [R,2K]
    (GPU0 first, GPU1 second), each K run already in total score order;
    out_scores/out_indices [R,K] in total score
    order; compact table [R,stride], local_slots [R,K] in ascending logical
    order (-1 padded), local_lengths [R]. Scalars: rows, table_stride, rank.
    unit_rows is 64 index records (including GLM Flash pools and V4 C4).
    """
    if topk not in (512, 1024, 2048) or unit_rows != 64:
        raise ValueError("topk=512|1024|2048 and unit_rows=64 required")
    launch = _CandidateMerge(topk, unit_rows)
    operands = (Operand("scores", torch.float32, f"[rows,{2*topk}]", align=4),
                Operand("indices", torch.int32, f"[rows,{2*topk}]", align=4),
                Operand("out_scores", torch.float32, f"[rows,{topk}]", "out", align=4),
                Operand("out_indices", torch.int32, f"[rows,{topk}]", "out", align=4),
                Operand("table", torch.int32, "[rows,table_stride]", align=4),
                Operand("local_slots", torch.int32, f"[rows,{topk}]", "out", align=4),
                Operand("local_lengths", torch.int32, "[rows]", "out", align=4))
    return compile_program(launch, name="dsa_candidate_merge", operands=operands,
                           scalars=(Scalar("rows"), Scalar("table_stride"), Scalar("rank")),
                           key=(topk, unit_rows), geometry={"topk": topk, "unit_rows": unit_rows},
                           doc=compile_dsa_candidate_merge_aot.__doc__)


class _CandidateMerge:
    def __init__(self, k, unit_rows):
        self.k, self.unit_rows = int(k), int(unit_rows)

    @cute.jit
    def __call__(self, scores: cute.Pointer, indices: cute.Pointer,
                 out_scores: cute.Pointer, out_indices: cute.Pointer,
                 table: cute.Pointer, local_slots: cute.Pointer, local_lengths: cute.Pointer,
                 rows: Int32, table_stride: Int32, rank: Int32, stream: cuda.CUstream):
        self.kernel(scores, indices, out_scores, out_indices, table, local_slots,
                    local_lengths, table_stride, rank).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, scores: cute.Pointer, indices: cute.Pointer,
               out_scores: cute.Pointer, out_indices: cute.Pointer,
               table: cute.Pointer, local_slots: cute.Pointer, local_lengths: cute.Pointer,
               table_stride: Int32, rank: Int32):
        tid, row = Int32(cute.arch.thread_idx()[0]), Int64(cute.arch.block_idx()[0])
        smem = utils.SmemAllocator()
        sv = smem.allocate_tensor(Float32, cute.make_layout((2*self.k,)), 16)
        si = smem.allocate_tensor(Int32, cute.make_layout((2*self.k,)), 16)
        for slot in cutlass.range_constexpr(2*self.k//256):
            i = tid + Int32(slot*256)
            si[i], sv[i] = indices[row*Int64(2*self.k)+Int64(i)], scores[row*Int64(2*self.k)+Int64(i)]
        cute.arch.sync_threads()
        picked = cute.make_rmem_tensor((self.k//256,), Int32)
        # One co-rank search per thread, then a contiguous register-sized run.
        start = tid * Int32(self.k//256)
        lo, hi = Int32(0), start
        while lo < hi:
            a = (lo+hi)//Int32(2)
            b = start-a
            advance = False
            if (b > Int32(0)) and (a < Int32(self.k)):
                advance = _pair_before(sv[a], si[a], sv[Int32(self.k)+b-Int32(1)], si[Int32(self.k)+b-Int32(1)])
            if advance:
                lo = a+Int32(1)
            else:
                hi = a
        a, b = lo, start-lo
        for slot in cutlass.range_constexpr(self.k//256):
            pos = start + Int32(slot)
            src = a
            if a >= Int32(self.k):
                src = Int32(self.k)+b
            elif b < Int32(self.k):
                if not _pair_before(sv[a], si[a], sv[Int32(self.k)+b], si[Int32(self.k)+b]):
                    src = Int32(self.k)+b
            idx, val = si[src], sv[src]
            if src < Int32(self.k):
                a += Int32(1)
            else:
                b += Int32(1)
            out_indices[row*Int64(self.k)+Int64(pos)] = idx
            out_scores[row*Int64(self.k)+Int64(pos)] = val if idx >= Int32(0) else Float32(float("-inf"))
            if (idx < Int32(0)) or ((idx//Int32(self.unit_rows))%Int32(2) != rank):
                idx = Int32(-1)
            picked[slot] = idx
        cute.arch.sync_threads()
        for slot in cutlass.range_constexpr(self.k//256):
            si[start+Int32(slot)] = picked[slot]
        cute.arch.sync_threads()
        _sort_indices(si, self.k)
        for slot in cutlass.range_constexpr(self.k//256):
            i = tid+Int32(slot*256)
            idx = Int32(si[i])
            physical = Int32(-1)
            if idx >= Int32(0):
                page = Int32(table[row*Int64(table_stride)+Int64(idx//Int32(self.unit_rows*2))])
                physical = page*Int32(self.unit_rows)+idx%Int32(self.unit_rows)
            local_slots[row*Int64(self.k)+Int64(i)] = physical
            if (i == Int32(0)) and (idx < Int32(0)):
                local_lengths[row] = Int32(0)
            if idx >= Int32(0):
                if i == Int32(self.k-1):
                    local_lengths[row] = Int32(self.k)
                elif Int32(si[i+Int32(1)]) < Int32(0):
                    local_lengths[row] = i+Int32(1)


class _Combine:
    def __init__(self, heads: int, sink: bool):
        self.heads, self.sink = int(heads), bool(sink)

    @cute.jit
    def __call__(self, o0: cute.Pointer, l0: cute.Pointer, o1: cute.Pointer,
                 l1: cute.Pointer, sink: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(o0, l0, o1, l1, sink, out).launch(
            grid=(rows, self.heads, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, o0: cute.Pointer, l0: cute.Pointer, o1: cute.Pointer,
               l1: cute.Pointer, sink: cute.Pointer, out: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        tid = Int32(cute.arch.thread_idx()[0])
        lh = Int64(row) * Int64(self.heads) + Int64(head)
        a, b = Float32(l0[lh]), Float32(l1[lh])
        m = cute.math.max(a, b)
        if cutlass.const_expr(self.sink):
            m = cute.math.max(m, Float32(sink[head]) * Float32(LOG2_E))
        wa, wb, ws = Float32(0), Float32(0), Float32(0)
        if a > Float32(float("-inf")):
            wa = cute.math.exp2(a - m, fastmath=True)
        if b > Float32(float("-inf")):
            wb = cute.math.exp2(b - m, fastmath=True)
        if cutlass.const_expr(self.sink):
            s = Float32(sink[head]) * Float32(LOG2_E)
            if s > Float32(float("-inf")):
                ws = cute.math.exp2(s - m, fastmath=True)
        denom = wa + wb + ws
        for i in cutlass.range_constexpr(2):
            off = lh * Int64(512) + Int64(tid + Int32(i * 256))
            v = Float32(0)
            # Do not read undefined/poisoned partials for empty shards.
            if wa > Float32(0):
                v += wa * Float32(o0[off])
            if wb > Float32(0):
                v += wb * Float32(o1[off])
            if denom > Float32(0):
                v = v / denom
            out[off] = v.to(cutlass.BFloat16)


def compile_lse_combine2_aot(*, heads: int, has_sink: bool = False):
    """o0 BF16 [R,H,512], l0 f32 [R,H], o1/l1 same, sink f32 [H]
    (ignored without has_sink), out BF16 [R,H,512]; scalar rows. LSE base2,
    sink natural-log units. Fixed GPU0 then GPU1 order; both empty -> zero.
    """
    if heads <= 0:
        raise ValueError("heads must be positive")
    operands = tuple(Operand(name, dtype, shape, role, align=align) for name, dtype, shape, role, align in (
        ("o0", torch.bfloat16, f"[rows,{heads},512]", "in", 16),
        ("l0", torch.float32, f"[rows,{heads}]", "in", 4),
        ("o1", torch.bfloat16, f"[rows,{heads},512]", "in", 16),
        ("l1", torch.float32, f"[rows,{heads}]", "in", 4),
        ("sink", torch.float32, f"[{heads}]", "in", 4),
        ("out", torch.bfloat16, f"[rows,{heads},512]", "out", 16)))
    return compile_program(_Combine(heads, has_sink), name="lse_combine2", operands=operands,
                           scalars=(Scalar("rows"),), key=(heads, has_sink),
                           geometry={"heads": heads, "has_sink": has_sink}, doc=compile_lse_combine2_aot.__doc__)


class _Gather:
    def __init__(self, row_bytes: int, page_rows: int, page_bytes: int):
        self.row_bytes, self.page_rows, self.page_bytes = row_bytes, page_rows, page_bytes

    @cute.jit
    def __call__(self, pool: cute.Pointer, table: cute.Pointer, staging: cute.Pointer,
                 rows: Int32, page_stride: Int32, page_offset: Int32, stream: cuda.CUstream):
        pages = (rows + Int32(self.page_rows - 1)) // Int32(self.page_rows)
        self.kernel(pool, table, staging, page_stride, page_offset).launch(
            grid=(pages, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, pool: cute.Pointer, table: cute.Pointer, staging: cute.Pointer,
               page_stride: Int32, page_offset: Int32):
        col = Int64(cute.arch.block_idx()[0])
        page = Int64(table[col])
        logical_page = col * Int64(page_stride) + Int64(page_offset)
        i = Int32(cute.arch.thread_idx()[0])
        if page >= Int64(0):
            src = cute.make_ptr(cutlass.Uint128, Int64(pool.toint()) + page*Int64(self.page_bytes), cute.AddressSpace.gmem, assumed_align=16)
            dst = cute.make_ptr(cutlass.Uint128, Int64(staging.toint()) + logical_page*Int64(self.page_bytes), cute.AddressSpace.gmem, assumed_align=16)
            while i < Int32(self.page_bytes//16):
                cute.arch.store(dst+Int64(i), cute.arch.load(src+Int64(i), cutlass.Uint128))
                i += Int32(256)


class _PartialLSE:
    def __init__(self, heads: int, splits: int):
        self.heads, self.splits = heads, splits

    @cute.jit
    def __call__(self, partial_lse: cute.Tensor, out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(partial_lse, out, rows).launch(
            grid=((rows * Int32(self.heads) + Int32(255)) // Int32(256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, partial_lse: cute.Tensor, out: cute.Pointer, rows: Int32):
        i = Int32(cute.arch.block_idx()[0]) * Int32(256) + Int32(cute.arch.thread_idx()[0])
        if i < rows * Int32(self.heads):
            row, head = i // Int32(self.heads), i % Int32(self.heads)
            m = Float32(float("-inf"))
            for s in cutlass.range_constexpr(self.splits):
                m = cute.math.max(m, Float32(partial_lse[row, head, s]))
            d = Float32(0)
            if m > Float32(float("-inf")):
                for s in cutlass.range_constexpr(self.splits):
                    d += cute.math.exp2(Float32(partial_lse[row, head, s]) - m, fastmath=True)
                m += cute.math.log2(d, fastmath=True)
            out[Int64(i)] = m


def sparse_mla_partial_split_plan(g=GLM53, *, max_rows: int, head_count: int, sm_count: int) -> int:
    """Load-time per-device numerics/launch choice; never a live-request policy."""
    from b12x.attention._shared.mla.kernel import plan_unified_decode_splits
    from ._common import DSV4Geometry
    from .glm_sparse_mla import _topk
    width = 128 + g.index_topk if isinstance(g, DSV4Geometry) else _topk(g)
    return plan_unified_decode_splits(topk=width, max_chunks=(width+63)//64,
                                     num_tokens=max_rows, h_blocks=head_count//16,
                                     sm_count=sm_count)[1]


class _Partial:
    def __init__(self, g, max_rows: int, begin: int, heads: int, num_splits: int | None):
        from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel, plan_unified_decode_splits
        from b12x.attention._shared.mla.merge import SparseMLASplitDecodeMergeKernel
        from b12x.attention._shared.mla.smem import make_smem_layout
        from ._common import DSV4Geometry
        from .glm_sparse_mla import _traits, _topk
        from .dsv4_sparse_mla import _traits as v4_traits
        self.v4 = isinstance(g, DSV4Geometry)
        self.begin, self.heads, self.total_heads = begin, heads, g.heads
        self.qk = 512 if self.v4 else g.latent_dim
        self.topk = 128 if self.v4 else _topk(g)
        self.extra_topk = g.index_topk if self.v4 else 0
        self.page_rows = 256 if self.v4 else g.page_rows
        self.record_bytes = 584 if self.v4 else g.record_bytes
        self.page_bytes = 149760 if self.v4 else g.kv_page_bytes
        self.extra_page_bytes = 37440 if self.v4 else 0
        self.scale = (512 ** -0.5) if self.v4 else g.softmax_scale
        traits = v4_traits() if self.v4 else _traits(g)
        hpb = traits.hpb
        if heads % hpb:
            raise ValueError(f"head_count must be divisible by {hpb}")
        width = self.topk + self.extra_topk
        _, self.splits, chunks = plan_unified_decode_splits(
            topk=width, max_chunks=(width + 63) // 64, num_tokens=max_rows,
            h_blocks=heads // hpb, forced_num_splits=num_splits,
            sm_count=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
        n, s = heads, self.splits
        self.decode = UnifiedDecodeKernel(
            traits, make_smem_layout(traits), self.page_rows, chunks,
            h_blocks=n // hpb, num_splits=s, num_heads=n, q_head_dim=self.qk,
            topk=self.topk, extra_topk=self.extra_topk,
            q_stride=(g.heads * self.qk, self.qk, 1), swa_indices_stride0=self.topk,
            extra_indices_stride0=self.extra_topk if self.v4 else self.topk,
            mid_out_stride=(n*s*512, s*512, 512, 1), mid_lse_stride=(n*s, s, 1),
            has_extra=self.v4, pbs_extra=64 if self.v4 else 1,
            valid_hpb=hpb, head_block_offset=0, per_token_len=True,
            native_glm_h8=False, native_dsv4_h8=False, native_dsv4_h16=False,
            native_dsv41_fp8=False, vector_q=True)
        self.merge = SparseMLASplitDecodeMergeKernel(static_num_chunks=s)
        self.lse = _PartialLSE(n, s)

    def scratch_bytes(self, rows):
        size = max(rows, 1) * self.heads * self.splits
        return (size * 512 * 2 + 1023) // 1024 * 1024 + size * 4

    @cute.jit
    def __call__(self, q: cute.Pointer, cache: cute.Pointer, indices: cute.Pointer,
                 lengths: cute.Pointer, extra_cache: cute.Pointer, extra_indices: cute.Pointer,
                 extra_lengths: cute.Pointer, out: cute.Pointer, lse: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m, n, s = Int64(rows), self.heads, self.splits
        base = Int64(scratch.toint())
        po = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, base, cute.AddressSpace.gmem, assumed_align=16),
                              cute.make_layout((m,n,s,512), stride=(n*s*512,s*512,512,1)))
        off = (m * Int64(n*s*512*2) + Int64(1023)) // Int64(1024) * Int64(1024)
        pl = cute.make_tensor(cute.make_ptr(Float32, base+off, cute.AddressSpace.gmem, assumed_align=16),
                              cute.make_layout((m,n,s), stride=(n*s,s,1)))
        qt = cute.make_tensor(q + Int64(self.begin*self.qk),
                              cute.make_layout((m,n,self.qk), stride=(self.total_heads*self.qk,self.qk,1)))
        kv = cute.make_tensor(cache, cute.make_layout((1,)))
        ix = cute.make_tensor(indices, cute.make_layout((m,self.topk), stride=(self.topk,1)))
        ln = cute.make_tensor(lengths, cute.make_layout((m,)))
        scale = Float32(self.scale * LOG2_E)
        if cutlass.const_expr(self.v4):
            self.decode.call_extra_pertok(qt, kv, ix, po, pl, scale, Float32(1), ln,
                Int64(self.page_bytes),
                cute.make_tensor(extra_cache, cute.make_layout((1,))),
                cute.make_tensor(extra_indices, cute.make_layout((m,self.extra_topk), stride=(self.extra_topk,1))),
                cute.make_tensor(extra_lengths, cute.make_layout((m,))),
                Int32(2), Int64(self.extra_page_bytes), rows, stream)
        else:
            self.decode.call_pertok(qt, kv, ix, po, pl, scale, Float32(1), ln,
                                   Int64(self.page_bytes), rows, stream)
        self.merge(po, pl, ln,
                   cute.make_tensor(out, cute.make_layout((m,n,512), stride=(n*512,512,1))), stream)
        self.lse(pl, lse, rows, stream)


def compile_sparse_mla_partial_aot(g=GLM53, *, max_rows: int, head_begin: int = 0,
                                   head_count: int | None = None, num_splits: int | None = None):
    """Normalized partial + base2 LSE, no sink. ABI: q BF16 [R,H,D],
    cache u8 [P,page_bytes], indices i32 [R,K], lengths i32 [R], extra_cache,
    extra_indices, extra_lengths (ignored for GLM), out BF16 [R,N,512],
    lse f32 [R,N], scratch u8; scalar rows. [head_begin,head_begin+N) reads
    the full query row and emits compact outputs. V4 main is its 128-window
    list (256 records/page), extra is C4 (64 records/page, K=512/1024).
    Empty lists produce zero output and -inf LSE. Caller supplies shard slots.
    Export explicit num_splits variants for each supported device, choose once
    at load via sparse_mla_partial_split_plan; never choose by live rows.
    """
    head_count = g.heads - head_begin if head_count is None else int(head_count)
    if max_rows <= 0 or head_begin < 0 or head_count <= 0 or head_begin + head_count > g.heads:
        raise ValueError("positive capacity and an in-bounds head range required")
    if num_splits is not None and num_splits <= 0:
        raise ValueError("num_splits must be positive")
    launch = _Partial(g, int(max_rows), int(head_begin), head_count, num_splits)
    operands = (Operand("q", torch.bfloat16, f"[rows,{g.heads},{launch.qk}]"),
                Operand("cache", torch.uint8, f"[pages,{launch.page_bytes}]"),
                Operand("indices", torch.int32, f"[rows,{launch.topk}]", align=4),
                Operand("lengths", torch.int32, "[rows]", align=4),
                Operand("extra_cache", torch.uint8, "[pages_c,37440]"),
                Operand("extra_indices", torch.int32, f"[rows,{launch.extra_topk}]", align=4),
                Operand("extra_lengths", torch.int32, "[rows]", align=4),
                Operand("out", torch.bfloat16, f"[rows,{head_count},512]", "out"),
                Operand("lse", torch.float32, f"[rows,{head_count}]", "out", align=4),
                Operand("scratch", torch.uint8, "[scratch_bytes]", "scratch"))
    return compile_program(launch, name="sparse_mla_partial", operands=operands,
                           scalars=(Scalar("rows"),),
                           key=(g.heads, launch.qk, launch.topk, launch.extra_topk,
                                head_begin, head_count, max_rows, launch.splits, launch.record_bytes),
                           geometry={"heads": g.heads, "head_begin": head_begin, "head_count": head_count,
                                     "record_bytes": launch.record_bytes, "num_splits": launch.splits},
                           scratch={"scratch": launch.scratch_bytes}, doc=compile_sparse_mla_partial_aot.__doc__)


def compile_paged_staging_gather_aot(*, row_bytes: int, page_rows: int, page_bytes: int | None = None):
    """Byte copy: pool u8 [P,page_bytes], compact table i32 [ceil(rows/page_rows)],
    staging u8 [logical_pages,page_bytes]. Scalars rows (local live rows),
    page_stride, page_offset: (2,rank) interleaves shards, (1,0) full gather.
    Copies whole pages including footer scales/padding and the allocated tail
    page: V4 and indexer use planar page payload + scales, GLM uses records.
    No conversion; 64-bit pool/staging offsets; caller allocates full pages.
    """
    page_bytes = page_rows * row_bytes if page_bytes is None else int(page_bytes)
    if min(row_bytes, page_rows) <= 0 or page_bytes < page_rows * row_bytes or page_bytes % 16:
        raise ValueError("positive geometry and a sufficient 16-byte-aligned page stride required")
    operands = (Operand("pool", torch.uint8, f"[pages,{page_bytes}]"),
                Operand("table", torch.int32, "[pages_local]", align=4),
                Operand("staging", torch.uint8, f"[pages_logical,{page_bytes}]", "out"))
    return compile_program(_Gather(row_bytes, page_rows, page_bytes), name="paged_staging_gather",
                           operands=operands, scalars=(Scalar("rows"), Scalar("page_stride"), Scalar("page_offset")),
                           key=(row_bytes, page_rows, page_bytes),
                           geometry={"row_bytes": row_bytes, "page_rows": page_rows, "page_bytes": page_bytes},
                           doc=compile_paged_staging_gather_aot.__doc__)
