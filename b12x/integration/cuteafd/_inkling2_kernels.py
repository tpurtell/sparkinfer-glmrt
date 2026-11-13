"""Inkling 2 relative-position GQA attention on MiMo's online GQA structure.

Arithmetic follows the cuteafd Inkling 2 reference (``python/reference/
families/inkling2/model.py::relative_attention``, #1915; the same rule as the
pinned vLLM ``d664cd1e`` ``tests/models/inkling2/test_attention.py::
_ref_attention``):

    scores[q, k] = (q . k + bias[q, head, q_pos - k_pos] / scale) * scale
    p = exp(scores - M) / (sum_k exp(scores - M) + exp(sink[q, head] - M))
    out = p @ v

with ``scale = head_dim ** -0.5``, the bias defined for distances ``0 <= d <
extent`` and zero beyond, keys masked to ``k_pos <= q_pos`` and, on sliding-
window layers, ``k_pos > q_pos - window`` (``extent == window`` there; global
layers use the checkpoint's ``rel_extent``). The sink is one unscaled logit
per (token, head). The output gate (``sigmoid(gate) * out``), the L2 clips and
the per-token log-scaling of q and r run outside this tile.

What changes against :class:`MimoGqaAttention`: head dims 128/128; a bias
gathered per (M row, key column) from ``rel`` (BF16 ``[rows, heads, extent]``)
and added to the raw FP32 accumulator as ``bias / scale`` before the online
softmax applies ``scale``; the sink read per (row, head) from an F32 ``[rows,
heads]`` tensor instead of per head. The split merge reads the same per-row
sink. Ring/paged key addressing, cp.async staging, the MMA shapes and the
base-2 FP32 online softmax with BF16 probabilities into the PV MMA are
MiMo's, inherited unchanged (``_resolve``, ``_load_bf16``, ``_layouts``,
``_tiled_mma_*``). Only BF16 records, one K/V tile staged at a time.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import warp

from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import warp_mma_gemm
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention.paged._selected_forward import _cp_async_load_128b_zfill

from ._mimo_kernels import LOG2_E, MAX_SPLITS, MimoGqaAttention, MimoSplitMerge, _ceil

__all__ = ["InklingGqaAttention", "InklingSplitMerge"]


class InklingGqaAttention(MimoGqaAttention):
    """MiMo's GQA tile with Inkling's distance bias and per-(token, head) sinks.

    ``extent`` is the bias table's last axis (local layers: the window; global
    layers: ``rel_extent``). The kernel entry takes ``rel`` (BF16 ``[rows,
    heads, extent]``) and ``sinks`` (F32 ``[rows, heads]``) where MiMo takes
    its per-head sink pointer; both are indexed by the live row (``row0 +
    token``) and the global head id. ``direct`` writes BF16 output with the
    sink in the denominator; otherwise normalized FP32 split partials and LSE
    without the sink, which :class:`InklingSplitMerge` adds once.
    """

    def __init__(self, *, heads: int, kv_heads: int, tokens: int, window: int, paged: bool, extent: int,
                 softmax_scale: float, direct: bool = True, page_rows: int = 64, ring_rows: int = 2048,
                 kv_warps: int = 4):
        super().__init__(heads=heads, kv_heads=kv_heads, tokens=tokens, window=window, paged=paged, sink=True,
                         direct=direct, softmax_scale=softmax_scale, page_rows=page_rows, ring_rows=ring_rows,
                         head=128, v_head=128, kv_warps=kv_warps)
        self.extent = int(extent)
        if self.extent <= 0:
            raise ValueError("the relative bias extent must be positive")
        if self.window > 0 and self.extent != self.window:
            raise ValueError("a sliding-window layer's bias extent is its window")
        # The online softmax multiplies the raw accumulator by scale, so the
        # bias enters the accumulator as bias / scale (the reference's order).
        self.bias_scale = 1.0 / float(softmax_scale)

    def key(self) -> tuple:
        return (*super().key(), "inkling2", self.extent)

    @cute.jit
    def __call__(self, q: cute.Pointer, cache: cute.Pointer, kv_step: cute.Pointer, positions: cute.Pointer,
                 page_table: cute.Pointer, ring_slots: cute.Pointer, seq_first: cute.Pointer,
                 rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, partial_out: cute.Pointer,
                 partial_lse: cute.Pointer, rows: Int32, table_stride: Int32, splits: Int32, stream: cuda.CUstream):
        q_layout, k_layout, v_layout, p_layout = self._layouts()
        self.kernel(q, cache, kv_step, positions, page_table, ring_slots, seq_first, rel, sinks, out, partial_out,
                    partial_lse, rows, table_stride, splits, q_layout, k_layout, v_layout, p_layout,
                    self._tiled_mma_qk(), self._tiled_mma_pv()).launch(
            grid=((rows + Int32(self.tokens - 1)) // Int32(self.tokens), self.kv_heads, splits),
            block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _biased_mask(self, token_pos: cute.Tensor, key_pos: cute.Tensor, s_p: cute.Tensor, s_scale: cute.Tensor,
                     thread: Int32, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, r_q: cute.Tensor,
                     r_k: cute.Tensor, r_p: cute.Tensor, r_v: cute.Tensor, c_q: cute.TiledCopy, c_k: cute.TiledCopy,
                     c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_q: cute.Tensor, cs_k: cute.Tensor,
                     cs_p: cute.Tensor, cs_v: cute.Tensor, acc_o: cute.Tensor, softmax: Softmax,
                     is_first: cutlass.Constexpr, kv_head: Int32, rel: cute.Pointer, row0: Int64, rows: Int32):
        """QK over the staged tile: add the gathered bias, mask, online softmax, PV accumulate."""
        if thread < Int32(self.qk_warps * 32):
            acc_s = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc_s.fill(0.0)
            warp_mma_gemm(tiled_mma_qk, acc_s, r_q, r_k, cs_q, cs_k, c_q, c_k, A_in_regs=not is_first)
            s_mn = layout_utils.reshape_acc_to_mn(acc_s)
            coords = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))))
            rel_t = cute.make_tensor(rel, cute.make_layout((Int64(rows) * Int64(self.heads * self.extent),)))
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                m_row = coords[m, 0][0]
                t_local = m_row // Int32(self.group)
                head = Int64(kv_head * Int32(self.group) + m_row % Int32(self.group))
                p = Int64(-1)
                if t_local < Int32(self.tokens):
                    p = Int64(token_pos[t_local])
                # This (live row, head)'s bias row; padded rows read row 0 and are masked below.
                rel_row = cutlass.select_(
                    p >= Int64(0), ((row0 + Int64(t_local)) * Int64(self.heads) + head) * Int64(self.extent),
                    Int64(0))
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    kp = Int64(key_pos[0, coords[m, n][1]])
                    masked = (p < Int64(0)) | (kp < Int64(0)) | (kp > p)
                    if const_expr(self.window > 0):
                        masked = masked | (kp <= p - Int64(self.window))
                    distance = p - kp
                    biased = (distance >= Int64(0)) & (distance < Int64(self.extent))
                    # Branchless gather: the index is clamped into the row, the value selected
                    # (masked pairs may add a bias; they become -inf right after).
                    index = cutlass.select_(distance < Int64(0), Int64(0), distance)
                    index = cutlass.select_(index >= Int64(self.extent), Int64(self.extent - 1), index)
                    bias = Float32(rel_t[rel_row + index]) * Float32(self.bias_scale)
                    s_mn[m, n] = s_mn[m, n] + cutlass.select_(biased, bias, Float32(0.0))
                    if masked:
                        s_mn[m, n] = -Float32.inf
            row_scale = softmax.online_softmax(acc_s, is_first=is_first, check_inf=True)
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    c = coords[m, n]
                    s_p[c[0], c[1]] = BFloat16(s_mn[m, n])
                if coords[m, 0][1] == Int32(0):
                    s_scale[coords[m, 0][0]] = row_scale[m]
        cute.arch.sync_threads()

        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
                scale = Float32(s_scale[o_coords[m, 0][0]])
                o_mn[m, None].store(o_mn[m, None].load() * scale)
        warp_mma_gemm(tiled_mma_pv, acc_o, r_p, r_v, cs_p, cs_v, c_p, c_v)
        cute.arch.sync_threads()

    @cute.jit
    def _biased_tile(self, tile: Int32, lo: Int64, hi: Int64, row0: Int64, kv_head: Int32, cache: cute.Pointer,
                     kv_step: cute.Pointer, page_table: cute.Pointer, table_base: Int64, seq_start: Int64,
                     step_first: Int64, ring_base: Int64, token_pos: cute.Tensor, key_addr: cute.Tensor,
                     key_pos: cute.Tensor, s_k: cute.Tensor, s_v: cute.Tensor, s_p: cute.Tensor,
                     s_scale: cute.Tensor, thread: Int32, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma,
                     r_q: cute.Tensor, r_k: cute.Tensor, r_p: cute.Tensor, r_v: cute.Tensor, c_q: cute.TiledCopy,
                     c_k: cute.TiledCopy, c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_q: cute.Tensor,
                     cs_k: cute.Tensor, cs_p: cute.Tensor, cs_v: cute.Tensor, acc_o: cute.Tensor, softmax: Softmax,
                     is_first: cutlass.Constexpr, rel: cute.Pointer, rows: Int32):
        """One BF16 key tile: resolve addresses, stage K/V, then the biased QK/softmax/PV step."""
        self._resolve(tile, Int32(0), lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                      ring_base, key_addr, key_pos, thread)
        cute.arch.sync_threads()
        self._load_bf16(key_addr, kv_head, cache, s_k, s_v, buf=Int32(0))
        cute.arch.sync_threads()
        self._biased_mask(token_pos, key_pos, s_p, s_scale, thread, tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v,
                          c_q, c_k, c_p, c_v, cs_q, cs_k, cs_p, cs_v, acc_o, softmax, is_first, kv_head, rel, row0,
                          rows)

    @cute.kernel
    def kernel(self, q: cute.Pointer, cache: cute.Pointer, kv_step: cute.Pointer, positions: cute.Pointer,
               page_table: cute.Pointer, ring_slots: cute.Pointer, seq_first: cute.Pointer, rel: cute.Pointer,
               sinks: cute.Pointer, out: cute.Pointer, partial_out: cute.Pointer, partial_lse: cute.Pointer,
               rows: Int32, table_stride: Int32, splits: Int32, q_layout: cute.ComposedLayout,
               k_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout,
               tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma):
        block, kv_head_idx, split_idx = cute.arch.block_idx()
        thread = Int32(cute.arch.thread_idx()[0])
        kv_head = Int32(kv_head_idx)
        split = Int32(split_idx)
        row0 = Int64(block) * Int64(self.tokens)
        live = Int32(rows) - Int32(row0)
        if live > Int32(self.tokens):
            live = Int32(self.tokens)

        allocator = cutlass.utils.SmemAllocator()
        s_q = allocator.allocate_tensor(element_type=BFloat16, layout=q_layout, byte_alignment=1024)
        s_k = allocator.allocate_tensor(element_type=BFloat16, layout=k_layout, byte_alignment=1024)
        s_v = allocator.allocate_tensor(element_type=BFloat16, layout=v_layout, byte_alignment=1024)
        s_p = allocator.allocate_tensor(element_type=BFloat16, layout=p_layout, byte_alignment=1024)
        s_scale = allocator.allocate_tensor(element_type=Float32, layout=cute.make_layout((self.tile_m,)),
                                            byte_alignment=16)
        # Slot 0 only (the inherited _resolve/_load_bf16 take a slot index); the (2, tile_n) shape is theirs.
        key_addr = allocator.allocate_tensor(element_type=Int64, layout=cute.make_layout(
            (2, self.tile_n), stride=(self.tile_n, 1)), byte_alignment=16)
        key_pos = allocator.allocate_tensor(element_type=Int64, layout=cute.make_layout(
            (2, self.tile_n), stride=(self.tile_n, 1)), byte_alignment=16)
        token_pos = allocator.allocate_tensor(element_type=Int64, layout=cute.make_layout((self.tokens,)),
                                              byte_alignment=16)

        pos_t = cute.make_tensor(positions, cute.make_layout((Int64(rows),)))
        if thread < Int32(self.tokens):
            p = Int64(-1)
            if thread < live:
                p = Int64(pos_t[row0 + Int64(thread)])
            token_pos[thread] = p
        # Query tile: M row m = (token m / group, head kv_head * group + m % group).
        for it in cutlass.range_constexpr(_ceil(self.tile_m * self.head // 8, self.threads)):
            linear = thread + Int32(it * self.threads)
            if linear < Int32(self.tile_m * self.head // 8):
                m = linear // Int32(self.head // 8)
                dim = (linear % Int32(self.head // 8)) * Int32(8)
                t_local = m // Int32(self.group)
                valid = (t_local < live) & (m < Int32(self.tokens * self.group))
                source = Int64(q.toint()) + (((row0 + Int64(t_local)) * Int64(self.heads) + Int64(
                    kv_head * Int32(self.group) + m % Int32(self.group))) * Int64(self.head) + Int64(dim)) * Int64(2)
                source = cutlass.select_(valid, source, Int64(q.toint()))
                _cp_async_load_128b_zfill(shared_ptr_to_u32(s_q.iterator + cute.crd2idx((m, dim), s_q.layout)),
                                          source, cutlass.select_(valid, Int32(16), Int32(0)))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Key range and addressing bases of this CTA's sequence.
        p_first = Int64(token_pos[0])
        p_last = Int64(token_pos[0])
        for t in cutlass.range_constexpr(self.tokens):
            if Int32(t) < live:
                p_last = Int64(token_pos[t])
        hi = p_last
        lo = Int64(0)
        if const_expr(self.window > 0):
            lo = p_first - Int64(self.window - 1)
            if lo < Int64(0):
                lo = Int64(0)
        table_base = Int64(0)
        seq_start = Int64(0)
        step_first = Int64(0)
        ring_base = Int64(0)
        if const_expr(self.paged):
            table_base = row0 * Int64(table_stride)
        else:
            first_t = cute.make_tensor(seq_first, cute.make_layout((Int64(rows),)))
            ring_t = cute.make_tensor(ring_slots, cute.make_layout((Int64(rows),)))
            step_first = Int64(first_t[row0])
            seq_start = Int64(pos_t[step_first])
            ring_base = Int64(ring_t[row0]) - p_first % Int64(self.ring_rows)

        qk_thread = thread % Int32(self.qk_warps * 32)
        thr_qk = tiled_mma_qk.get_slice(qk_thread)
        thr_pv = tiled_mma_pv.get_slice(thread)
        r_q = thr_qk.make_fragment_A(thr_qk.partition_A(s_q))
        r_k = thr_qk.make_fragment_B(thr_qk.partition_B(s_k))
        s_vt = layout_utils.transpose_view(s_v)
        r_p = thr_pv.make_fragment_A(thr_pv.partition_A(s_p))
        r_v = thr_pv.make_fragment_B(thr_pv.partition_B(s_vt))
        atom_qk = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
        atom_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
        c_q = cute.make_tiled_copy_A(atom_qk, tiled_mma_qk).get_slice(qk_thread)
        c_k = cute.make_tiled_copy_B(atom_qk, tiled_mma_qk).get_slice(qk_thread)
        c_p = cute.make_tiled_copy_A(atom_qk, tiled_mma_pv).get_slice(thread)
        c_v = cute.make_tiled_copy_B(atom_v, tiled_mma_pv).get_slice(thread)
        cs_q = c_q.partition_S(s_q)
        cs_k = c_k.partition_S(s_k)
        cs_p = c_p.partition_S(s_p)
        cs_v = c_v.partition_S(s_vt)
        acc_o = cute.make_rmem_tensor(thr_pv.partition_shape_C((self.tile_m, self.v_head)), Float32)
        acc_o.fill(0.0)
        s_layout = layout_utils.convert_layout_acc_mn(
            thr_qk.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))).layout)
        softmax = Softmax.create(Float32(self.softmax_scale * LOG2_E), cute.size(s_layout.shape[0]), arch=120)
        softmax.reset()

        tile_count = Int32((hi - lo + Int64(self.tile_n)) // Int64(self.tile_n))
        if live <= Int32(0):
            tile_count = Int32(0)
        my_tiles = (tile_count - split + splits - Int32(1)) // splits
        if my_tiles > Int32(0):
            self._biased_tile(split, lo, hi, row0, kv_head, cache, kv_step, page_table, table_base, seq_start,
                              step_first, ring_base, token_pos, key_addr, key_pos, s_k, s_v, s_p, s_scale, thread,
                              tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q, cs_k, cs_p,
                              cs_v, acc_o, softmax, True, rel, rows)
            for local in cutlass.range(my_tiles - Int32(1), unroll=1):
                tile = split + (Int32(local) + Int32(1)) * splits
                self._biased_tile(tile, lo, hi, row0, kv_head, cache, kv_step, page_table, table_base, seq_start,
                                  step_first, ring_base, token_pos, key_addr, key_pos, s_k, s_v, s_p, s_scale,
                                  thread, tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q,
                                  cs_k, cs_p, cs_v, acc_o, softmax, False, rel, rows)

        # Final scale per M row with its (row, head) sink (direct), or the LSE for split partials.
        if thread < Int32(self.qk_warps * 32):
            coords = layout_utils.reshape_acc_to_mn(
                thr_qk.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))))
            if const_expr(self.direct):
                sink_t = cute.make_tensor(sinks, cute.make_layout((Int64(rows) * Int64(self.heads),)))
                sink_vals = cute.make_rmem_tensor(cute.size(s_layout.shape[0]), Float32)
                for m in cutlass.range_constexpr(cute.size(s_layout.shape[0])):
                    m_row = coords[m, 0][0]
                    t_local = m_row // Int32(self.group)
                    head = Int64(kv_head * Int32(self.group) + m_row % Int32(self.group))
                    valid = (t_local < live) & (m_row < Int32(self.tokens * self.group))
                    sink_row = cutlass.select_(valid, (row0 + Int64(t_local)) * Int64(self.heads) + head, Int64(0))
                    sink_vals[m] = Float32(sink_t[sink_row])
                scale = softmax.finalize(sink_val=sink_vals)
            else:
                scale = softmax.finalize()
            if my_tiles <= Int32(0):
                for m in cutlass.range_constexpr(cute.size(s_layout.shape[0])):
                    scale[m] = Float32(0.0)
                    softmax.row_sum[m] = -Float32.inf
            if coords[0, 0][1] == Int32(0):
                for m in cutlass.range_constexpr(cute.size(softmax.row_sum)):
                    m_row = coords[m, 0][0]
                    s_scale[m_row] = scale[m]
                    if const_expr(not self.direct):
                        lse_token = m_row // Int32(self.group)
                        if (lse_token < live) & (m_row < Int32(self.tokens * self.group)):
                            lse_head = kv_head * Int32(self.group) + m_row % Int32(self.group)
                            lse_t = cute.make_tensor(partial_lse, cute.make_layout((Int64(rows) * Int64(
                                MAX_SPLITS * self.heads),)))
                            lse_t[((row0 + Int64(lse_token)) * Int64(splits) + Int64(split)) * Int64(self.heads)
                                  + Int64(lse_head)] = Float32(softmax.row_sum[m])
        cute.arch.sync_threads()

        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            thr_pv.partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
            o_row = o_coords[m, 0][0]
            o_scale = Float32(s_scale[o_row])
            o_token = o_row // Int32(self.group)
            if (o_token < live) & (o_row < Int32(self.tokens * self.group)):
                o_head = Int64(kv_head * Int32(self.group) + o_row % Int32(self.group))
                row = row0 + Int64(o_token)
                for n in cutlass.range_constexpr(cute.size(o_mn.shape[1])):
                    dim = Int64(o_coords[m, n][1])
                    value = Float32(o_mn[m, n]) * o_scale
                    if const_expr(self.direct):
                        o_t = cute.make_tensor(out, cute.make_layout((Int64(rows) * Int64(self.heads * self.v_head),)))
                        o_t[(row * Int64(self.heads) + o_head) * Int64(self.v_head) + dim] = BFloat16(value)
                    else:
                        po = cute.make_tensor(partial_out, cute.make_layout((Int64(rows) * Int64(
                            MAX_SPLITS * self.heads * self.v_head),)))
                        po[((row * Int64(splits) + Int64(split)) * Int64(self.heads) + o_head) * Int64(self.v_head)
                           + dim] = value


class InklingSplitMerge(MimoSplitMerge):
    """MiMo's split merge with the sink read per (row, head) from F32 ``[rows, heads]``.

    ``out = sum_s w_s o_s / (sum_s w_s + exp(sink - M))``, ``w_s = exp(lse_s -
    M)``, ``M`` the max over the live splits and the sink. One CTA per (head,
    row), one thread per value dim.
    """

    def __init__(self, *, heads: int, v_head: int = 128):
        super().__init__(heads=heads, v_head=v_head, sink=True, empty_rows_zero=False)

    @cute.kernel
    def kernel(self, partial_out: cute.Pointer, partial_lse: cute.Pointer, sinks: cute.Pointer,
               out: cute.Pointer, splits: Int32):
        head_i, row_i, _ = cute.arch.block_idx()
        dim = Int64(cute.arch.thread_idx()[0])
        head, row = Int64(head_i), Int64(row_i)
        lse = cute.make_tensor(partial_lse, cute.make_layout((Int64(1) << Int64(40),)))
        po = cute.make_tensor(partial_out, cute.make_layout((Int64(1) << Int64(40),)))
        sink_t = cute.make_tensor(sinks, cute.make_layout((Int64(1) << Int64(40),)))
        sink = Float32(sink_t[row * Int64(self.heads) + head])
        top = sink
        for s in cutlass.range(splits, unroll=1):
            value = Float32(lse[(row * Int64(splits) + Int64(s)) * Int64(self.heads) + head])
            if value > top:
                top = value
        total = Float32(0.0)
        acc = Float32(0.0)
        for s in cutlass.range(splits, unroll=1):
            value = Float32(lse[(row * Int64(splits) + Int64(s)) * Int64(self.heads) + head])
            if value > -Float32.inf:
                w = cute.math.exp2((value - top) * Float32(LOG2_E), fastmath=False)
                total = total + w
                acc = acc + w * Float32(po[((row * Int64(splits) + Int64(s)) * Int64(self.heads) + head)
                                           * Int64(self.v_head) + dim])
        total = total + cute.math.exp2((sink - top) * Float32(LOG2_E), fastmath=False)
        o = cute.make_tensor(out, cute.make_layout((Int64(1) << Int64(40),)))
        o[(row * Int64(self.heads) + head) * Int64(self.v_head) + dim] = BFloat16(acc / total)
