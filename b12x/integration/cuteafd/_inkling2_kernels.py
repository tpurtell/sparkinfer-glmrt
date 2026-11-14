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
MiMo's, inherited unchanged (``_resolve``, ``_layouts``, ``_tiled_mma_*``).
Prefill double-buffers K, V and the staged bias slice across the full QK/PV
compute step (aliasing ``s_q`` onto the second K stage once ``r_q`` is loaded
into registers), splits ``(K + rel)`` and ``V`` into separate ``cp.async``
commit groups, and specializes interior unmasked tiles with immediate-offset
shared-memory bias loads.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import ld_shared_bf16_to_f32, shared_ptr_to_u32
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import warp_mma_gemm
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention.paged._selected_forward import _cp_async_load_128b_zfill

from ._mimo_kernels import LOG2_E, MAX_SPLITS, MimoGqaAttention, MimoSplitMerge, _ceil

__all__ = ["InklingGqaAttention", "InklingSplitMerge"]


@dsl_user_op
def _ld_shared_bf16_f32_off(addr: Int32, byte_offset: int = 0, *, loc=None, ip=None) -> Float32:
    """Load one BF16 from shared memory at ``addr + byte_offset`` and widen to FP32."""
    offset = int(byte_offset)
    address = f"[$1+{offset}]" if offset else "[$1]"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            f"{{ .reg .b16 tmp; ld.shared.b16 tmp, {address}; cvt.f32.bf16 $0, tmp; }}",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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
        # Bias staging per M row: the tile's tile_n distances start at an 8-aligned
        # distance, so up to tile_n + 7 elements, rounded to whole 16-byte vectors.
        self.rel_stage = _ceil(self.tile_n + 7, 8) * 8
        self.rel_vectors = self.rel_stage // 8
        if self.extent % 8:
            raise ValueError("the bias extent must be a multiple of 8 (16-byte staging vectors)")
        # Prefill (tile_m == tile_n == 64) already runs at 1 CTA/SM and aliases s_q onto s_k1
        # once r_q is preloaded, fitting full 2-stage K/V/rel pipelining in 92.3 KiB SMEM.
        # Decode (tile_m == 16) keeps single-stage K/V/rel (42.3 KiB) to preserve 2 CTAs/SM.
        self.pipe_kv = self.tile_m == self.tile_n

    def key(self) -> tuple:
        return (*super().key(), "inkling2", self.extent)

    @cute.jit
    def __call__(self, q: cute.Pointer, cache: cute.Pointer, kv_step: cute.Pointer, positions: cute.Pointer,
                 page_table: cute.Pointer, ring_slots: cute.Pointer, seq_first: cute.Pointer,
                 rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, partial_out: cute.Pointer,
                 partial_lse: cute.Pointer, rows: Int32, table_stride: Int32, splits: Int32, stream: cuda.CUstream):
        q_layout, k_layout, v_layout, p_layout = self._layouts()
        if const_expr(self.pipe_kv):
            q_layout = k_layout
        self.kernel(q, cache, kv_step, positions, page_table, ring_slots, seq_first, rel, sinks, out, partial_out,
                    partial_lse, rows, table_stride, splits, q_layout, k_layout, v_layout, p_layout,
                    self._tiled_mma_qk(), self._tiled_mma_pv()).launch(
            grid=((rows + Int32(self.tokens - 1)) // Int32(self.tokens), self.kv_heads, splits),
            block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _stage_tile(self, tile: Int32, buf: Int32, lo: Int64, p_min: Int64, row0: Int64, kv_head: Int32,
                    cache: cute.Pointer, rel: cute.Pointer, live: Int32, token_pos: cute.Tensor,
                    key_addr: cute.Tensor, s_k: cute.Tensor, s_v: cute.Tensor, s_rel: cute.Tensor, thread: Int32):
        """Issue two ``cp.async`` commit groups for ``tile`` from ``key_addr[buf]``:
        group 1 stages ``K`` and (when any distance can fall below ``extent``) ``rel``;
        group 2 stages ``V``."""
        k_vectors = self.head // 8
        cache_fallback = Int64(cache.toint())
        for it_k in cutlass.range_constexpr(self.tile_n * k_vectors // self.threads):
            linear_k = thread + Int32(it_k * self.threads)
            token = linear_k // Int32(k_vectors)
            dim = (linear_k % Int32(k_vectors)) * Int32(8)
            base_addr = Int64(key_addr[buf, token])
            valid_k = base_addr >= Int64(0)
            src_k = cutlass.select_(
                valid_k, base_addr + (Int64(kv_head) * Int64(self.head) + Int64(dim)) * Int64(2), cache_fallback
            )
            _cp_async_load_128b_zfill(
                shared_ptr_to_u32(s_k.iterator + cute.crd2idx((token, dim), s_k.layout)),
                src_k,
                cutlass.select_(valid_k, Int32(16), Int32(0)),
            )

        k0 = lo + Int64(tile) * Int64(self.tile_n)
        skip_rel = Int32(0)
        if const_expr(self.window == 0):
            skip_rel = cutlass.select_(
                (live > Int32(0)) & (k0 >= Int64(0))
                & (k0 + Int64(self.tile_n - 1) <= p_min - Int64(self.extent)),
                Int32(1),
                Int32(0),
            )
        if skip_rel == Int32(0):
            rel_fallback = Int64(rel.toint())
            s_rel_base = shared_ptr_to_u32(s_rel.iterator)
            for it_r in cutlass.range_constexpr(_ceil(self.tile_m * self.rel_vectors, self.threads)):
                linear_r = thread + Int32(it_r * self.threads)
                if linear_r < Int32(self.tile_m * self.rel_vectors):
                    m = linear_r // Int32(self.rel_vectors)
                    vec = linear_r % Int32(self.rel_vectors)
                    t_local = m // Int32(self.group)
                    head = Int64(kv_head * Int32(self.group) + m % Int32(self.group))
                    valid_rel = (t_local < live) & (m < Int32(self.tokens * self.group))
                    p = Int64(0)
                    if valid_rel:
                        p = Int64(token_pos[t_local])
                    base_d = (Int32(p - k0) - Int32(self.tile_n - 1)) & ~Int32(7)
                    distance = base_d + vec * Int32(8)
                    valid_rel = valid_rel & (distance >= Int32(0)) & (distance < Int32(self.extent))
                    src_rel = rel_fallback + (
                        (((row0 + Int64(t_local)) * Int64(self.heads) + head) * Int64(self.extent))
                        + Int64(distance)
                    ) * Int64(2)
                    src_rel = cutlass.select_(valid_rel, src_rel, rel_fallback)
                    _cp_async_load_128b_zfill(
                        s_rel_base + (m * Int32(self.rel_stage) + vec * Int32(8)) * Int32(2),
                        src_rel,
                        cutlass.select_(valid_rel, Int32(16), Int32(0)),
                    )
        cute.arch.cp_async_commit_group()

        v_vectors = self.v_head // 8
        v0 = self.kv_heads * self.head
        for it_v in cutlass.range_constexpr(self.tile_n * v_vectors // self.threads):
            linear_v = thread + Int32(it_v * self.threads)
            token_v = linear_v // Int32(v_vectors)
            dim_v = (linear_v % Int32(v_vectors)) * Int32(8)
            base_addr_v = Int64(key_addr[buf, token_v])
            valid_v = base_addr_v >= Int64(0)
            src_v = cutlass.select_(
                valid_v,
                base_addr_v + (Int64(v0) + Int64(kv_head) * Int64(self.v_head) + Int64(dim_v)) * Int64(2),
                cache_fallback,
            )
            _cp_async_load_128b_zfill(
                shared_ptr_to_u32(s_v.iterator + cute.crd2idx((token_v, dim_v), s_v.layout)),
                src_v,
                cutlass.select_(valid_v, Int32(16), Int32(0)),
            )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _qk_bias_softmax(self, buf: Int32, k0: Int64, p_min: Int64, p_max: Int64, live: Int32,
                         token_pos: cute.Tensor, key_pos: cute.Tensor, s_p: cute.Tensor, s_scale: cute.Tensor,
                         s_rel: cute.Tensor, thread: Int32, tiled_mma_qk: cute.TiledMma, r_q: cute.Tensor,
                         r_k: cute.Tensor, c_q: cute.TiledCopy, c_k: cute.TiledCopy, cs_q: cute.Tensor,
                         cs_k: cute.Tensor, softmax: Softmax, is_first: cutlass.Constexpr):
        """QK MMA (from register-resident ``r_q``), relative-position bias, causal/window mask, and
        online softmax into ``s_p`` / ``s_scale``."""
        if thread < Int32(self.qk_warps * 32):
            thr_qk = tiled_mma_qk.get_slice(thread)
            acc_s = cute.make_rmem_tensor(thr_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc_s.fill(0.0)
            warp_mma_gemm(tiled_mma_qk, acc_s, r_q, r_k, cs_q, cs_k, c_q, c_k, A_in_regs=True)
            s_mn = layout_utils.reshape_acc_to_mn(acc_s)
            ident = cute.make_identity_tensor((self.tile_m, self.tile_n))
            coords = layout_utils.reshape_acc_to_mn(thr_qk.partition_C(ident))
            t0_coords = layout_utils.reshape_acc_to_mn(tiled_mma_qk.get_slice(0).partition_C(ident))
            thr_col_offset = Int32(coords[0, 0][1])
            s_rel_base = shared_ptr_to_u32(s_rel.iterator)
            bias_scale = Float32(self.bias_scale)

            # By two's-complement identity, d0 - ((d0 - (tile_n - 1)) & ~7) == (tile_n - 1) + rem
            # with rem = (d0 - (tile_n - 1)) & 7 in [0, 7], so slot = (tile_n - 1) - col + rem is
            # unconditionally in [0, tile_n + 6] subset [0, rel_stage) for every column col.
            use_imm = all(
                isinstance(t0_coords[0, j][1], int) and not isinstance(t0_coords[0, j][1], bool)
                for j in range(cute.size(s_mn.shape[1]))
            )
            max_t0_col = (
                max(int(t0_coords[0, j][1]) for j in range(cute.size(s_mn.shape[1])))
                if use_imm
                else 0
            )
            max_thr_col = (self.tile_n - 1) - max_t0_col

            full_biased = (
                (live > Int32(0))
                & (k0 >= Int64(0))
                & (k0 + Int64(self.tile_n - 1) <= p_min)
                & (p_max - k0 < Int64(self.extent))
            )
            if full_biased:
                for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                    m_row = Int32(coords[m, 0][0])
                    t_local = m_row // Int32(self.group)
                    if (t_local < live) & (m_row < Int32(self.tokens * self.group)):
                        p = Int64(token_pos[t_local])
                        d0 = Int32(p - k0)
                        rem = (d0 - Int32(self.tile_n - 1)) & Int32(7)
                        if const_expr(use_imm):
                            row_addr = s_rel_base + (
                                m_row * Int32(self.rel_stage) + rem + Int32(max_thr_col) - thr_col_offset
                            ) * Int32(2)
                            for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                                bias = (
                                    _ld_shared_bf16_f32_off(row_addr, 2 * (max_t0_col - int(t0_coords[0, n][1])))
                                    * bias_scale
                                )
                                s_mn[m, n] = s_mn[m, n] + bias
                        else:
                            row_addr = s_rel_base + (
                                m_row * Int32(self.rel_stage) + Int32(self.tile_n - 1) + rem - thr_col_offset
                            ) * Int32(2)
                            for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                                bias = (
                                    ld_shared_bf16_to_f32(row_addr - Int32(2) * Int32(t0_coords[0, n][1]))
                                    * bias_scale
                                )
                                s_mn[m, n] = s_mn[m, n] + bias
                    else:
                        for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                            s_mn[m, n] = -Float32.inf
            else:
                is_zero_bias = Int32(0)
                if const_expr(self.window == 0):
                    is_zero_bias = cutlass.select_(
                        (live > Int32(0))
                        & (k0 >= Int64(0))
                        & (k0 + Int64(self.tile_n - 1) <= p_min - Int64(self.extent)),
                        Int32(1),
                        Int32(0),
                    )
                if is_zero_bias != Int32(0):
                    for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                        m_row = Int32(coords[m, 0][0])
                        t_local = m_row // Int32(self.group)
                        if (t_local >= live) | (m_row >= Int32(self.tokens * self.group)):
                            for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                                s_mn[m, n] = -Float32.inf
                else:
                    for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                        m_row = Int32(coords[m, 0][0])
                        t_local = m_row // Int32(self.group)
                        p = Int64(-1)
                        if (t_local < live) & (m_row < Int32(self.tokens * self.group)):
                            p = Int64(token_pos[t_local])
                        if p >= Int64(0):
                            d0 = Int32(p - k0)
                            rem = (d0 - Int32(self.tile_n - 1)) & Int32(7)
                            if const_expr(use_imm):
                                row_addr = s_rel_base + (
                                    m_row * Int32(self.rel_stage) + rem + Int32(max_thr_col) - thr_col_offset
                                ) * Int32(2)
                                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                                    col = Int32(coords[m, n][1])
                                    kp = Int64(key_pos[buf, col])
                                    masked = (kp < Int64(0)) | (kp > p)
                                    if const_expr(self.window > 0):
                                        masked = masked | (kp <= p - Int64(self.window))
                                    distance = d0 - col
                                    biased = (distance >= Int32(0)) & (distance < Int32(self.extent))
                                    bias = (
                                        _ld_shared_bf16_f32_off(
                                            row_addr, 2 * (max_t0_col - int(t0_coords[0, n][1]))
                                        )
                                        * bias_scale
                                    )
                                    val = s_mn[m, n] + cutlass.select_(biased, bias, Float32(0.0))
                                    s_mn[m, n] = cutlass.select_(masked, -Float32.inf, val)
                            else:
                                row_addr = s_rel_base + (
                                    m_row * Int32(self.rel_stage) + Int32(self.tile_n - 1) + rem - thr_col_offset
                                ) * Int32(2)
                                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                                    col = Int32(coords[m, n][1])
                                    kp = Int64(key_pos[buf, col])
                                    masked = (kp < Int64(0)) | (kp > p)
                                    if const_expr(self.window > 0):
                                        masked = masked | (kp <= p - Int64(self.window))
                                    distance = d0 - col
                                    biased = (distance >= Int32(0)) & (distance < Int32(self.extent))
                                    bias = (
                                        ld_shared_bf16_to_f32(row_addr - Int32(2) * Int32(t0_coords[0, n][1]))
                                        * bias_scale
                                    )
                                    val = s_mn[m, n] + cutlass.select_(biased, bias, Float32(0.0))
                                    s_mn[m, n] = cutlass.select_(masked, -Float32.inf, val)
                        else:
                            for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                                s_mn[m, n] = -Float32.inf

            row_scale = softmax.online_softmax(acc_s, is_first=is_first, check_inf=True)
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    c = coords[m, n]
                    s_p[c[0], c[1]] = BFloat16(s_mn[m, n])
                if coords[m, 0][1] == Int32(0):
                    s_scale[coords[m, 0][0]] = row_scale[m]

    @cute.jit
    def _pv_step(self, s_scale: cute.Tensor, thread: Int32, tiled_mma_pv: cute.TiledMma, r_p: cute.Tensor,
                 r_v: cute.Tensor, c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_p: cute.Tensor,
                 cs_v: cute.Tensor, acc_o: cute.Tensor, is_first: cutlass.Constexpr):
        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
                scale = Float32(s_scale[o_coords[m, 0][0]])
                o_mn[m, None].store(o_mn[m, None].load() * scale)
        warp_mma_gemm(tiled_mma_pv, acc_o, r_p, r_v, cs_p, cs_v, c_p, c_v)

    @cute.jit
    def _pipe_step(self, idx: Int32, my_tiles: Int32, split: Int32, splits: Int32, lo: Int64, hi: Int64,
                   p_min: Int64, p_max: Int64, row0: Int64, kv_head: Int32, cache: cute.Pointer,
                   kv_step: cute.Pointer, page_table: cute.Pointer, table_base: Int64, seq_start: Int64,
                   step_first: Int64, ring_base: Int64, rel: cute.Pointer, live: Int32, token_pos: cute.Tensor,
                   key_addr: cute.Tensor, key_pos: cute.Tensor, s_k_next: cute.Tensor, s_v_next: cute.Tensor,
                   s_rel_next: cute.Tensor, s_p: cute.Tensor, s_scale: cute.Tensor, s_rel_cur: cute.Tensor,
                   thread: Int32, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, r_q: cute.Tensor,
                   r_k: cute.Tensor, r_p: cute.Tensor, r_v: cute.Tensor, c_q: cute.TiledCopy, c_k: cute.TiledCopy,
                   c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_q: cute.Tensor, cs_k_cur: cute.Tensor,
                   cs_p: cute.Tensor, cs_v_cur: cute.Tensor, acc_o: cute.Tensor, softmax: Softmax,
                   cur_buf: cutlass.Constexpr, is_first: cutlass.Constexpr):
        """One 2-stage pipelined tile step: issue ``(K+rel, V)`` of ``idx + 1`` before ``QK_idx``,
        drain ``V_idx`` before ``PV_idx``, resolve ``idx + 2`` during ``PV_idx``, and drain
        ``(K+rel)_{idx+1}`` after ``PV_idx``."""
        next_buf = 1 - cur_buf
        has_next = (idx + Int32(1)) < my_tiles
        has_next2 = (idx + Int32(2)) < my_tiles
        tile_cur = split + idx * splits
        if has_next:
            self._stage_tile(tile_cur + splits, Int32(next_buf), lo, p_min, row0, kv_head, cache, rel, live,
                             token_pos, key_addr, s_k_next, s_v_next, s_rel_next, thread)
        k0 = lo + Int64(tile_cur) * Int64(self.tile_n)
        self._qk_bias_softmax(Int32(cur_buf), k0, p_min, p_max, live, token_pos, key_pos, s_p, s_scale,
                              s_rel_cur, thread, tiled_mma_qk, r_q, r_k, c_q, c_k, cs_q, cs_k_cur, softmax,
                              is_first)
        if has_next:
            cute.arch.cp_async_wait_group(2)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if has_next2:
            self._resolve(tile_cur + Int32(2) * splits, Int32(cur_buf), lo, hi, cache, kv_step, page_table,
                          table_base, seq_start, step_first, ring_base, key_addr, key_pos, thread)
        self._pv_step(s_scale, thread, tiled_mma_pv, r_p, r_v, c_p, c_v, cs_p, cs_v_cur, acc_o, is_first)
        if has_next:
            cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

    @cute.jit
    def _single_step(self, idx: Int32, my_tiles: Int32, split: Int32, splits: Int32, buf: Int32, lo: Int64,
                     hi: Int64, p_min: Int64, p_max: Int64, row0: Int64, kv_head: Int32, cache: cute.Pointer,
                     kv_step: cute.Pointer, page_table: cute.Pointer, table_base: Int64, seq_start: Int64,
                     step_first: Int64, ring_base: Int64, rel: cute.Pointer, live: Int32, token_pos: cute.Tensor,
                     key_addr: cute.Tensor, key_pos: cute.Tensor, s_k: cute.Tensor, s_v: cute.Tensor,
                     s_p: cute.Tensor, s_scale: cute.Tensor, s_rel: cute.Tensor, thread: Int32,
                     tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, r_q: cute.Tensor, r_k: cute.Tensor,
                     r_p: cute.Tensor, r_v: cute.Tensor, c_q: cute.TiledCopy, c_k: cute.TiledCopy,
                     c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_q: cute.Tensor, cs_k: cute.Tensor,
                     cs_p: cute.Tensor, cs_v: cute.Tensor, acc_o: cute.Tensor, softmax: Softmax,
                     is_first: cutlass.Constexpr):
        """Decode single-stage K/V/rel step: overlap ``V`` transfer with ``QK + bias + softmax`` and
        overlap next-tile address resolution with ``PV``."""
        tile_cur = split + idx * splits
        has_next = (idx + Int32(1)) < my_tiles
        self._stage_tile(tile_cur, buf, lo, p_min, row0, kv_head, cache, rel, live, token_pos, key_addr,
                         s_k, s_v, s_rel, thread)
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()
        k0 = lo + Int64(tile_cur) * Int64(self.tile_n)
        self._qk_bias_softmax(buf, k0, p_min, p_max, live, token_pos, key_pos, s_p, s_scale, s_rel, thread,
                              tiled_mma_qk, r_q, r_k, c_q, c_k, cs_q, cs_k, softmax, is_first)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if has_next:
            self._resolve(tile_cur + splits, Int32(1) - buf, lo, hi, cache, kv_step, page_table, table_base,
                          seq_start, step_first, ring_base, key_addr, key_pos, thread)
        self._pv_step(s_scale, thread, tiled_mma_pv, r_p, r_v, c_p, c_v, cs_p, cs_v, acc_o, is_first)
        cute.arch.sync_threads()

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
        s_k0 = allocator.allocate_tensor(element_type=BFloat16, layout=k_layout, byte_alignment=1024)
        s_v0 = allocator.allocate_tensor(element_type=BFloat16, layout=v_layout, byte_alignment=1024)
        if const_expr(self.pipe_kv):
            s_k1 = allocator.allocate_tensor(element_type=BFloat16, layout=k_layout, byte_alignment=1024)
            s_v1 = allocator.allocate_tensor(element_type=BFloat16, layout=v_layout, byte_alignment=1024)
            s_q = s_k1
        else:
            s_k1 = s_k0
            s_v1 = s_v0
            s_q = allocator.allocate_tensor(element_type=BFloat16, layout=q_layout, byte_alignment=1024)
        s_p = allocator.allocate_tensor(element_type=BFloat16, layout=p_layout, byte_alignment=1024)
        rel_layout = cute.make_layout((self.tile_m, self.rel_stage), stride=(self.rel_stage, 1))
        s_rel0 = allocator.allocate_tensor(element_type=BFloat16, layout=rel_layout, byte_alignment=16)
        s_rel1 = s_rel0
        if const_expr(self.pipe_kv):
            s_rel1 = allocator.allocate_tensor(element_type=BFloat16, layout=rel_layout, byte_alignment=16)
        s_scale = allocator.allocate_tensor(element_type=Float32, layout=cute.make_layout((self.tile_m,)),
                                            byte_alignment=16)
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
        p_last = p_first
        p_min = p_first
        p_max = p_first
        for t in cutlass.range_constexpr(self.tokens):
            if Int32(t) < live:
                pt = Int64(token_pos[t])
                p_last = pt
                p_min = cutlass.select_(pt < p_min, pt, p_min)
                p_max = cutlass.select_(pt > p_max, pt, p_max)
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
        r_k = thr_qk.make_fragment_B(thr_qk.partition_B(s_k0))
        s_vt0 = layout_utils.transpose_view(s_v0)
        r_p = thr_pv.make_fragment_A(thr_pv.partition_A(s_p))
        r_v = thr_pv.make_fragment_B(thr_pv.partition_B(s_vt0))
        atom_qk = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
        atom_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16)
        c_q = cute.make_tiled_copy_A(atom_qk, tiled_mma_qk).get_slice(qk_thread)
        c_k = cute.make_tiled_copy_B(atom_qk, tiled_mma_qk).get_slice(qk_thread)
        c_p = cute.make_tiled_copy_A(atom_qk, tiled_mma_pv).get_slice(thread)
        c_v = cute.make_tiled_copy_B(atom_v, tiled_mma_pv).get_slice(thread)
        cs_q = c_q.partition_S(s_q)
        cs_k0 = c_k.partition_S(s_k0)
        cs_p = c_p.partition_S(s_p)
        cs_v0 = c_v.partition_S(s_vt0)
        cs_k1, cs_v1 = cs_k0, cs_v0
        if const_expr(self.pipe_kv):
            cs_k1 = c_k.partition_S(s_k1)
            cs_v1 = c_v.partition_S(layout_utils.transpose_view(s_v1))
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

        # Preload r_q from s_q and resolve tile 0 concurrently; the barrier frees s_q (= s_k1).
        if thread < Int32(self.qk_warps * 32):
            r_q_copy = c_q.retile(r_q)
            for k in cutlass.range_constexpr(cute.size(cs_q.shape[2])):
                cute.copy(c_q, cs_q[None, None, k], r_q_copy[None, None, k])
        if my_tiles > Int32(0):
            self._resolve(split, Int32(0), lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                          ring_base, key_addr, key_pos, thread)
        cute.arch.sync_threads()

        if const_expr(self.pipe_kv):
            if my_tiles > Int32(0):
                self._stage_tile(split, Int32(0), lo, p_min, row0, kv_head, cache, rel, live, token_pos,
                                 key_addr, s_k0, s_v0, s_rel0, thread)
                if my_tiles > Int32(1):
                    self._resolve(split + splits, Int32(1), lo, hi, cache, kv_step, page_table, table_base,
                                  seq_start, step_first, ring_base, key_addr, key_pos, thread)
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()

                self._pipe_step(Int32(0), my_tiles, split, splits, lo, hi, p_min, p_max, row0, kv_head, cache,
                                kv_step, page_table, table_base, seq_start, step_first, ring_base, rel, live,
                                token_pos, key_addr, key_pos, s_k1, s_v1, s_rel1, s_p, s_scale, s_rel0, thread,
                                tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q, cs_k0,
                                cs_p, cs_v0, acc_o, softmax, cur_buf=0, is_first=True)
                pairs = (my_tiles - Int32(1)) // Int32(2)
                for pair in cutlass.range(pairs, unroll=1):
                    idx1 = Int32(1) + Int32(pair) * Int32(2)
                    self._pipe_step(idx1, my_tiles, split, splits, lo, hi, p_min, p_max, row0, kv_head, cache,
                                    kv_step, page_table, table_base, seq_start, step_first, ring_base, rel, live,
                                    token_pos, key_addr, key_pos, s_k0, s_v0, s_rel0, s_p, s_scale, s_rel1, thread,
                                    tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q,
                                    cs_k1, cs_p, cs_v1, acc_o, softmax, cur_buf=1, is_first=False)
                    self._pipe_step(idx1 + Int32(1), my_tiles, split, splits, lo, hi, p_min, p_max, row0, kv_head,
                                    cache, kv_step, page_table, table_base, seq_start, step_first, ring_base, rel,
                                    live, token_pos, key_addr, key_pos, s_k1, s_v1, s_rel1, s_p, s_scale, s_rel0,
                                    thread, tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v,
                                    cs_q, cs_k0, cs_p, cs_v0, acc_o, softmax, cur_buf=0, is_first=False)
                if (my_tiles - Int32(1)) % Int32(2) != Int32(0):
                    self._pipe_step(my_tiles - Int32(1), my_tiles, split, splits, lo, hi, p_min, p_max, row0,
                                    kv_head, cache, kv_step, page_table, table_base, seq_start, step_first,
                                    ring_base, rel, live, token_pos, key_addr, key_pos, s_k0, s_v0, s_rel0, s_p,
                                    s_scale, s_rel1, thread, tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q,
                                    c_k, c_p, c_v, cs_q, cs_k1, cs_p, cs_v1, acc_o, softmax, cur_buf=1,
                                    is_first=False)
        else:
            if my_tiles > Int32(0):
                self._single_step(Int32(0), my_tiles, split, splits, Int32(0), lo, hi, p_min, p_max, row0, kv_head,
                                  cache, kv_step, page_table, table_base, seq_start, step_first, ring_base, rel,
                                  live, token_pos, key_addr, key_pos, s_k0, s_v0, s_p, s_scale, s_rel0, thread,
                                  tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v, c_q, c_k, c_p, c_v, cs_q, cs_k0,
                                  cs_p, cs_v0, acc_o, softmax, is_first=True)
                for local in cutlass.range(my_tiles - Int32(1), unroll=1):
                    idx = Int32(local) + Int32(1)
                    self._single_step(idx, my_tiles, split, splits, idx % Int32(2), lo, hi, p_min, p_max, row0,
                                      kv_head, cache, kv_step, page_table, table_base, seq_start, step_first,
                                      ring_base, rel, live, token_pos, key_addr, key_pos, s_k0, s_v0, s_p,
                                      s_scale, s_rel0, thread, tiled_mma_qk, tiled_mma_pv, r_q, r_k, r_p, r_v,
                                      c_q, c_k, c_p, c_v, cs_q, cs_k0, cs_p, cs_v0, acc_o, softmax,
                                      is_first=False)

        # Final scale per M row with its (row, head) sink (direct), or the LSE for split partials.
        if thread < Int32(self.qk_warps * 32):
            coords = layout_utils.reshape_acc_to_mn(
                thr_qk.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))))
            if const_expr(self.direct):
                sink_t = cute.make_tensor(sinks, cute.make_layout((Int64(rows) * Int64(self.heads),)))
                sink_vals = cute.make_rmem_tensor(cute.size(s_layout.shape[0]), Float32)
                for m in cutlass.range_constexpr(cute.size(s_layout.shape[0])):
                    sink_m = coords[m, 0][0]
                    sink_t_local = sink_m // Int32(self.group)
                    sink_head = Int64(kv_head * Int32(self.group) + sink_m % Int32(self.group))
                    sink_valid = (sink_t_local < live) & (sink_m < Int32(self.tokens * self.group))
                    sink_row = cutlass.select_(
                        sink_valid, (row0 + Int64(sink_t_local)) * Int64(self.heads) + sink_head, Int64(0))
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
