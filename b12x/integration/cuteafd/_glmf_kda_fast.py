"""Fast bit-exact KDA decode core for GLM 5.3 Flash TP2 (``g2``, max_rows <= 64).

Only ``compile_glmf_kda_core_aot(g2, max_rows=64, decode=True)`` (FP32 state,
non-projected, low-rank gate, 32 heads, hidden 4096) reaches this module.
Kimi KDA and GLM Flash prefill KDA stay on the unchanged stock paths.

Every floating-point operation preserves stock's exact rounding points and
reduction order so ``out[:rows]``, ``replay``, ``state``, and ``conv_state`` are
byte-identical to stock in both verify (``spec != 0``) and plain (``spec == 0``)
modes:

1. ``fab`` (288x4096 BF16 MMA) uses narrow/deep TMA tiles (18..72 CTAs vs 5).
2. ``_KdaPack`` is eliminated: ``fg``, ``conv_prep``, and ``conv_state_plain``
   read ``q``, ``k``, ``v``, and ``fab`` directly.
3. ``fg`` (batched 2x4096x128 BF16 MMA) uses ``tile_n=32, num_stages=2`` (256
   CTAs vs 64) with row-matched ``compute_warps`` (1/2/4 for R<=16/32/64).
4. ``GlmfKdaConvPrep`` fuses causal 4-tap conv + SiLU + BF16 rounding + L2 norm
   of q/k + decay + beta + replay writes across all ``(row, head)`` pairs in
   parallel, removing 32x redundant and 64x serial transcendental math from the
   recurrent loop.
5. ``GlmfKdaRecurrentFast`` maps 1 v-row per warp (512 CTAs, 4096 concurrent
   warps across all SMs) with vectorized 16-byte state loads/stores and
   double-buffered precomputed inputs.
6. ``GlmfKdaGatedNormFast`` launches 1 warp per ``(row, head)`` with vectorized
   BF16 loads/stores.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    div_rn_f32,
    ld_global_nc_f32,
    ld_global_v2_u32,
    ld_global_v4_f32,
    pack_f32x2_to_bfloat2,
    st_global_v2_u32,
    st_global_v4_f32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._common import GLMFGeometry
from ._glm_kernels import BatchedBf16Gemm, _bf16, _rsqrt, _warp_sum, glm_projection
from ._glmf_kernels import (
    CONV_TAPS,
    KDA_HEAD,
    REPLAY_ROWS,
    _conv_state,
    _replay_beta,
    _replay_rows,
    _shift_conv_state,
    _sigmoid,
    kda_replay_layout,
)
from .glmf import _align, _ptr, kda_scratch_bytes


def is_glmf_kda_decode(g, max_rows: int, decode: bool, state_dtype: str, projected: bool) -> bool:
    return (
        isinstance(g, GLMFGeometry)
        and bool(decode)
        and int(max_rows) <= 64
        and state_dtype == "float32"
        and not bool(projected)
        and g.kda_heads == 32
        and g.hidden == 4096
        and g.kda_head_dim == 128
    )


class GlmfKdaConvPrep:
    """Fused conv + SiLU + L2-norm(q, k) + decay + beta + replay writes.

    One warp per ``(row, head)``; lane ``l`` owns channels ``4l .. 4l+3``.
    In verify mode (``spec != 0``), writes ``k``, ``decay``, ``v``, ``beta`` and
    raw ``q|k|v`` directly to ``replay``, and ``q`` to ``q_prep``.
    In plain mode (``spec == 0``), writes ``q``, ``k``, ``decay``, ``v``, ``beta``
    to scratch (``replay`` untouched).
    """

    def __init__(self, *, heads: int, lower_bound: float, fab_stride: int, fg_stride: int,
                 eps: float = 1.0e-6, replay_rows: int = REPLAY_ROWS):
        self.heads = int(heads)
        self.width = self.heads * KDA_HEAD
        self.channels = 3 * self.width
        self.lower_bound = float(lower_bound)
        self.fab_stride = int(fab_stride)
        self.fg_stride = int(fg_stride)
        self.eps = float(eps)
        self.scale = KDA_HEAD ** -0.5
        self.replay_rows = int(replay_rows)
        self.beta_off, self.proj_off, _ = kda_replay_layout(self.heads, self.channels, self.replay_rows)

    @cute.jit
    def __call__(
        self,
        q_in: cute.Pointer,
        k_in: cute.Pointer,
        v_in: cute.Pointer,
        fab: cute.Pointer,
        g_raw: cute.Pointer,
        conv_w: cute.Pointer,
        a_log: cute.Pointer,
        dt_bias: cute.Pointer,
        conv_state: cute.Pointer,
        slots: cute.Pointer,
        seq_first: cute.Pointer,
        q_prep: cute.Pointer,
        k_prep: cute.Pointer,
        decay_prep: cute.Pointer,
        v_prep: cute.Pointer,
        beta_prep: cute.Pointer,
        replay: cute.Pointer,
        spec: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        m = Int64(rows)
        d = self.width
        c = self.channels
        self.kernel(
            q_in,
            k_in,
            v_in,
            cute.make_tensor(fab, cute.make_layout((m, self.fab_stride), stride=(self.fab_stride, 1))),
            g_raw,
            cute.make_tensor(conv_w, cute.make_layout((c, CONV_TAPS), stride=(CONV_TAPS, 1))),
            cute.make_tensor(a_log, cute.make_layout((self.heads,))),
            cute.make_tensor(dt_bias, cute.make_layout((d,))),
            conv_state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(seq_first, cute.make_layout((m,))),
            q_prep,
            k_prep,
            decay_prep,
            v_prep,
            cute.make_tensor(beta_prep, cute.make_layout((m, self.heads), stride=(self.heads, 1))),
            replay,
            spec,
        ).launch(grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def _conv_silu_4(
        self,
        src_base: Int64,
        conv_w: cute.Tensor,
        state: cute.Tensor,
        first: Int64,
        rel: Int64,
        slot: Int64,
        part_off: cutlass.Constexpr,
        col_k0: Int64,
        replay_proj_row: Int64,
        spec: Int32,
    ) -> tuple[Float32, Float32, Float32, Float32]:
        d = Int64(self.width)
        ch0 = Int64(part_off) + col_k0
        acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(4):
            acc[i] = Float32(0.0)
        for j in cutlass.range_constexpr(CONV_TAPS):
            s = rel - Int64(CONV_TAPS - 1 - j)
            x0, x1, x2, x3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            if s >= Int64(0):
                addr = src_base + ((first + s) * d + col_k0) * Int64(2)
                w0, w1 = ld_global_v2_u32(addr)
                if cutlass.const_expr(j == CONV_TAPS - 1):
                    if spec != Int32(0):
                        st_global_v2_u32(replay_proj_row + ch0 * Int64(2), w0, w1)
                x0, x1 = _bf16_lo(w0), _bf16_hi(w0)
                x2, x3 = _bf16_lo(w1), _bf16_hi(w1)
            elif slot >= Int64(0):
                st_r = Int64(CONV_TAPS - 1) + s
                x0 = Float32(state[st_r, ch0 + Int64(0)])
                x1 = Float32(state[st_r, ch0 + Int64(1)])
                x2 = Float32(state[st_r, ch0 + Int64(2)])
                x3 = Float32(state[st_r, ch0 + Int64(3)])
            acc[0] = acc[0] + Float32(conv_w[ch0 + Int64(0), j]) * x0
            acc[1] = acc[1] + Float32(conv_w[ch0 + Int64(1), j]) * x1
            acc[2] = acc[2] + Float32(conv_w[ch0 + Int64(2), j]) * x2
            acc[3] = acc[3] + Float32(conv_w[ch0 + Int64(3), j]) * x3
        s0 = div_rn_f32(acc[0], Float32(1.0) + cute.math.exp(-acc[0], fastmath=False))
        s1 = div_rn_f32(acc[1], Float32(1.0) + cute.math.exp(-acc[1], fastmath=False))
        s2 = div_rn_f32(acc[2], Float32(1.0) + cute.math.exp(-acc[2], fastmath=False))
        s3 = div_rn_f32(acc[3], Float32(1.0) + cute.math.exp(-acc[3], fastmath=False))
        p0 = pack_f32x2_to_bfloat2(s0, s1)
        p1 = pack_f32x2_to_bfloat2(s2, s3)
        return _bf16_lo(p0), _bf16_hi(p0), _bf16_lo(p1), _bf16_hi(p1)

    @cute.kernel
    def kernel(
        self,
        q_in: cute.Pointer,
        k_in: cute.Pointer,
        v_in: cute.Pointer,
        fab: cute.Tensor,
        g_raw: cute.Pointer,
        conv_w: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        conv_state: cute.Pointer,
        slots: cute.Tensor,
        seq_first: cute.Tensor,
        q_prep: cute.Pointer,
        k_prep: cute.Pointer,
        decay_prep: cute.Pointer,
        v_prep: cute.Pointer,
        beta_prep: cute.Tensor,
        replay_ptr: cute.Pointer,
        spec: Int32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        k0 = Int64(lane) * Int64(4)
        col = head * Int64(KDA_HEAD)
        col_k0 = col + k0

        first = Int64(seq_first[row])
        slot = Int64(slots[row])
        rel = row - first
        state = _conv_state(conv_state, slot, self.channels)

        r_base = Int64(replay_ptr.toint())
        replay_proj_row = r_base + Int64(self.proj_off) + row * Int64(self.channels * 2)

        q0, q1, q2, q3 = self._conv_silu_4(
            Int64(q_in.toint()), conv_w, state, first, rel, slot, 0, col_k0, replay_proj_row, spec
        )
        k0v, k1v, k2v, k3v = self._conv_silu_4(
            Int64(k_in.toint()), conv_w, state, first, rel, slot, self.width, col_k0, replay_proj_row, spec
        )
        v0, v1, v2, v3 = self._conv_silu_4(
            Int64(v_in.toint()), conv_w, state, first, rel, slot, 2 * self.width, col_k0, replay_proj_row, spec
        )

        # Exact same reduction order as stock GlmfKdaRecurrent lines 651-666.
        q_sq = Float32(0.0)
        q_sq = q_sq + q0 * q0
        q_sq = q_sq + q1 * q1
        q_sq = q_sq + q2 * q2
        q_sq = q_sq + q3 * q3

        k_sq = Float32(0.0)
        k_sq = k_sq + k0v * k0v
        k_sq = k_sq + k1v * k1v
        k_sq = k_sq + k2v * k2v
        k_sq = k_sq + k3v * k3v

        rate = cute.math.exp(Float32(a_log[head]), fastmath=False)
        gw0, gw1 = ld_global_v2_u32(
            Int64(g_raw.toint()) + (row * Int64(self.fg_stride) + col_k0) * Int64(2)
        )
        g0, g1 = _bf16_lo(gw0), _bf16_hi(gw0)
        g2, g3 = _bf16_lo(gw1), _bf16_hi(gw1)

        d0 = cute.math.exp(
            Float32(self.lower_bound) * _sigmoid(rate * (g0 + Float32(dt_bias[col_k0 + Int64(0)]))),
            fastmath=False,
        )
        d1 = cute.math.exp(
            Float32(self.lower_bound) * _sigmoid(rate * (g1 + Float32(dt_bias[col_k0 + Int64(1)]))),
            fastmath=False,
        )
        d2 = cute.math.exp(
            Float32(self.lower_bound) * _sigmoid(rate * (g2 + Float32(dt_bias[col_k0 + Int64(2)]))),
            fastmath=False,
        )
        d3 = cute.math.exp(
            Float32(self.lower_bound) * _sigmoid(rate * (g3 + Float32(dt_bias[col_k0 + Int64(3)]))),
            fastmath=False,
        )

        q_norm = cute.math.sqrt(_warp_sum(q_sq) + Float32(self.eps), fastmath=False)
        k_norm = cute.math.sqrt(_warp_sum(k_sq) + Float32(self.eps), fastmath=False)

        scale = Float32(self.scale)
        q0 = div_rn_f32(q0, q_norm) * scale
        q1 = div_rn_f32(q1, q_norm) * scale
        q2 = div_rn_f32(q2, q_norm) * scale
        q3 = div_rn_f32(q3, q_norm) * scale

        k0v = div_rn_f32(k0v, k_norm)
        k1v = div_rn_f32(k1v, k_norm)
        k2v = div_rn_f32(k2v, k_norm)
        k3v = div_rn_f32(k3v, k_norm)

        beta = _bf16(_sigmoid(Float32(fab[row, Int64(2 * KDA_HEAD) + head])))

        elem_off_f32 = (row * Int64(self.width) + col_k0) * Int64(4)
        st_global_v4_f32(Int64(q_prep.toint()) + elem_off_f32, q0, q1, q2, q3)

        if spec != Int32(0):
            # replay[row, head, p, k0..k0+3] stride = (heads * 3 * 128, 3 * 128, 128, 1)
            rh_base = r_base + ((row * Int64(self.heads) + head) * Int64(3 * KDA_HEAD) + k0) * Int64(4)
            st_global_v4_f32(rh_base, k0v, k1v, k2v, k3v)
            st_global_v4_f32(rh_base + Int64(KDA_HEAD * 4), d0, d1, d2, d3)
            st_global_v4_f32(rh_base + Int64(2 * KDA_HEAD * 4), v0, v1, v2, v3)
            if lane == Int32(0):
                replay_beta = _replay_beta(r_base + Int64(self.beta_off), self.heads, self.replay_rows)
                replay_beta[row, head] = beta
        else:
            st_global_v4_f32(Int64(k_prep.toint()) + elem_off_f32, k0v, k1v, k2v, k3v)
            st_global_v4_f32(Int64(decay_prep.toint()) + elem_off_f32, d0, d1, d2, d3)
            v_addr = Int64(v_prep.toint()) + (row * Int64(self.width) + col_k0) * Int64(2)
            st_global_v2_u32(v_addr, pack_f32x2_to_bfloat2(v0, v1), pack_f32x2_to_bfloat2(v2, v3))
            if lane == Int32(0):
                beta_prep[row, head] = beta


class GlmfKdaConvStatePlain:
    """Plain-mode (``spec == 0``) conv-state update reading ``q``, ``k``, ``v`` directly."""

    threads = 256

    def __init__(self, *, width: int):
        self.width = int(width)
        self.channels = 3 * self.width

    @cute.jit
    def __call__(
        self,
        q_in: cute.Pointer,
        k_in: cute.Pointer,
        v_in: cute.Pointer,
        conv_state: cute.Pointer,
        slots: cute.Pointer,
        seq_first: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        m = Int64(rows)
        d = self.width
        self.kernel(
            cute.make_tensor(q_in, cute.make_layout((m, d), stride=(d, 1))),
            cute.make_tensor(k_in, cute.make_layout((m, d), stride=(d, 1))),
            cute.make_tensor(v_in, cute.make_layout((m, d), stride=(d, 1))),
            conv_state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(seq_first, cute.make_layout((m,))),
            rows,
        ).launch(grid=(rows, self.channels // self.threads, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        v_in: cute.Tensor,
        conv_state: cute.Pointer,
        slots: cute.Tensor,
        seq_first: cute.Tensor,
        rows: Int32,
    ):
        row = Int64(cute.arch.block_idx()[0])
        ch = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        first = Int64(seq_first[row])
        last = row + Int64(1) == Int64(rows)
        if not last:
            last = Int64(seq_first[row + Int64(1)]) != first
        slot = Int64(slots[row])
        if last and slot >= Int64(0):
            state = _conv_state(conv_state, slot, self.channels)
            n = row - first + Int64(1)
            d = Int64(self.width)
            if ch < d:
                kept = cute.make_rmem_tensor(cute.make_layout((CONV_TAPS - 1,), stride=(1,)), BFloat16)
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    s = n - Int64(CONV_TAPS - 1 - i)
                    if s >= Int64(0):
                        kept[i] = q_in[first + s, ch]
                    else:
                        kept[i] = state[Int64(CONV_TAPS - 1) + s, ch]
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    state[i, ch] = kept[i]
            elif ch < Int64(2) * d:
                c_loc = ch - d
                kept = cute.make_rmem_tensor(cute.make_layout((CONV_TAPS - 1,), stride=(1,)), BFloat16)
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    s = n - Int64(CONV_TAPS - 1 - i)
                    if s >= Int64(0):
                        kept[i] = k_in[first + s, c_loc]
                    else:
                        kept[i] = state[Int64(CONV_TAPS - 1) + s, ch]
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    state[i, ch] = kept[i]
            else:
                c_loc = ch - Int64(2) * d
                kept = cute.make_rmem_tensor(cute.make_layout((CONV_TAPS - 1,), stride=(1,)), BFloat16)
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    s = n - Int64(CONV_TAPS - 1 - i)
                    if s >= Int64(0):
                        kept[i] = v_in[first + s, c_loc]
                    else:
                        kept[i] = state[Int64(CONV_TAPS - 1) + s, ch]
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    state[i, ch] = kept[i]


class GlmfKdaRecurrentFast:
    """High-occupancy KDA recurrence over precomputed FP32 (q, k, decay, v, beta).

    Grid ``(heads, 128 // 8, 1) = (32, 16, 1) = 512`` CTAs, 256 threads (8 warps),
    1 v-row per warp -> all 4096 ``(head, v_row)`` warps execute concurrently.
    Vectorized 16-byte FP32 state/input loads and double-buffered row prefetching.
    """

    threads = 256
    v_block = 8

    def __init__(self, *, heads: int, replay_rows: int = REPLAY_ROWS):
        self.heads = int(heads)
        self.width = self.heads * KDA_HEAD
        self.replay_rows = int(replay_rows)
        self.beta_off, _, _ = kda_replay_layout(self.heads, 3 * self.width, self.replay_rows)

    @cute.jit
    def __call__(
        self,
        q_prep: cute.Pointer,
        k_prep: cute.Pointer,
        decay_prep: cute.Pointer,
        v_prep: cute.Pointer,
        beta_prep: cute.Pointer,
        state: cute.Pointer,
        slots: cute.Pointer,
        out: cute.Pointer,
        replay: cute.Pointer,
        spec: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        m = Int64(rows)
        d = self.width
        self.kernel(
            q_prep,
            k_prep,
            decay_prep,
            cute.make_tensor(v_prep, cute.make_layout((m, d), stride=(d, 1))),
            cute.make_tensor(beta_prep, cute.make_layout((m, self.heads), stride=(self.heads, 1))),
            state,
            cute.make_tensor(slots, cute.make_layout((m,))),
            cute.make_tensor(out, cute.make_layout((m, d), stride=(d, 1))),
            replay,
            spec,
            rows,
        ).launch(
            grid=(self.heads, KDA_HEAD // self.v_block, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _state_row_addr(self, state: cute.Pointer, slot: Int64, head: Int64, v_row: Int64, k0: Int64) -> Int64:
        elem = ((slot * Int64(self.heads) + head) * Int64(KDA_HEAD) + v_row) * Int64(KDA_HEAD) + k0
        return Int64(state.toint()) + elem * Int64(4)

    @cute.kernel
    def kernel(
        self,
        q_prep: cute.Pointer,
        k_prep: cute.Pointer,
        decay_prep: cute.Pointer,
        v_prep: cute.Tensor,
        beta_prep: cute.Tensor,
        state: cute.Pointer,
        slots: cute.Tensor,
        out: cute.Tensor,
        replay_ptr: cute.Pointer,
        spec: Int32,
        rows: Int32,
    ):
        head = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx >> Int32(5)
        lane = tidx & Int32(31)
        v_row = Int64(cute.arch.block_idx()[1]) * Int64(self.v_block) + Int64(warp_id)
        k0 = Int64(lane) * Int64(4)
        col = head * Int64(KDA_HEAD)
        col_k0 = col + k0

        s0, s1, s2, s3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        q_base = Int64(q_prep.toint()) + col_k0 * Int64(4)
        q_row_stride = Int64(self.width * 4)

        if spec != Int32(0):
            # Verify path: read k, decay, v, beta from L2-hot replay; state is read-only.
            r_base = Int64(replay_ptr.toint())
            r_row_stride = Int64(self.heads * 3 * KDA_HEAD * 4)
            r_head_k = r_base + (head * Int64(3 * KDA_HEAD) + k0) * Int64(4)
            r_head_d = r_head_k + Int64(KDA_HEAD * 4)
            r_head_v = r_base + (head * Int64(3 * KDA_HEAD) + Int64(2 * KDA_HEAD) + v_row) * Int64(4)
            r_beta_base = r_base + Int64(self.beta_off) + head * Int64(4)
            r_beta_stride = Int64(self.heads * 4)

            current = Int64(-1)
            if rows > Int32(0):
                slot0 = Int64(slots[Int64(0)])
                if slot0 >= Int64(0):
                    s0, s1, s2, s3 = ld_global_v4_f32(self._state_row_addr(state, slot0, head, v_row, k0))
                current = slot0

                q0, q1, q2, q3 = ld_global_v4_f32(q_base)
                k0v, k1v, k2v, k3v = ld_global_v4_f32(r_head_k)
                d0, d1, d2, d3 = ld_global_v4_f32(r_head_d)
                v_cur = ld_global_nc_f32(r_head_v)
                b_cur = ld_global_nc_f32(r_beta_base)

                for t in cutlass.range(rows, unroll=1):
                    row = Int64(t)
                    if t > Int32(0):
                        slot = Int64(slots[row])
                        if slot != current:
                            if slot >= Int64(0):
                                s0, s1, s2, s3 = ld_global_v4_f32(
                                    self._state_row_addr(state, slot, head, v_row, k0)
                                )
                            else:
                                s0, s1, s2, s3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                            current = slot

                    # Software-pipeline next row's loads ahead of warp reductions.
                    nq0, nq1, nq2, nq3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                    nk0, nk1, nk2, nk3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                    nd0, nd1, nd2, nd3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                    nv, nb = Float32(0.0), Float32(0.0)
                    if t + Int32(1) < rows:
                        nr = row + Int64(1)
                        nq0, nq1, nq2, nq3 = ld_global_v4_f32(q_base + nr * q_row_stride)
                        nk0, nk1, nk2, nk3 = ld_global_v4_f32(r_head_k + nr * r_row_stride)
                        nd0, nd1, nd2, nd3 = ld_global_v4_f32(r_head_d + nr * r_row_stride)
                        nv = ld_global_nc_f32(r_head_v + nr * r_row_stride)
                        nb = ld_global_nc_f32(r_beta_base + nr * r_beta_stride)

                    s0 = s0 * d0
                    memory = Float32(0.0) + s0 * k0v
                    s1 = s1 * d1
                    memory = memory + s1 * k1v
                    s2 = s2 * d2
                    memory = memory + s2 * k2v
                    s3 = s3 * d3
                    memory = memory + s3 * k3v

                    delta = (v_cur - _warp_sum(memory)) * b_cur

                    s0 = s0 + k0v * delta
                    o = Float32(0.0) + s0 * q0
                    s1 = s1 + k1v * delta
                    o = o + s1 * q1
                    s2 = s2 + k2v * delta
                    o = o + s2 * q2
                    s3 = s3 + k3v * delta
                    o = o + s3 * q3

                    o = _warp_sum(o)
                    if lane == Int32(0):
                        out[row, col + v_row] = o.to(BFloat16)

                    if t + Int32(1) < rows:
                        q0, q1, q2, q3 = nq0, nq1, nq2, nq3
                        k0v, k1v, k2v, k3v = nk0, nk1, nk2, nk3
                        d0, d1, d2, d3 = nd0, nd1, nd2, nd3
                        v_cur, b_cur = nv, nb
        else:
            # Plain path: read k, decay, v, beta from scratch; commit state on transitions and exit.
            k_base = Int64(k_prep.toint()) + col_k0 * Int64(4)
            d_base = Int64(decay_prep.toint()) + col_k0 * Int64(4)
            current = Int64(-1)
            for t in cutlass.range(rows, unroll=1):
                row = Int64(t)
                slot = Int64(slots[row])
                if slot != current:
                    if current >= Int64(0):
                        st_global_v4_f32(self._state_row_addr(state, current, head, v_row, k0), s0, s1, s2, s3)
                    if slot >= Int64(0):
                        s0, s1, s2, s3 = ld_global_v4_f32(self._state_row_addr(state, slot, head, v_row, k0))
                    else:
                        s0, s1, s2, s3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                    current = slot

                q0, q1, q2, q3 = ld_global_v4_f32(q_base + row * q_row_stride)
                k0v, k1v, k2v, k3v = ld_global_v4_f32(k_base + row * q_row_stride)
                d0, d1, d2, d3 = ld_global_v4_f32(d_base + row * q_row_stride)
                v_cur = Float32(v_prep[row, col + v_row])
                b_cur = Float32(beta_prep[row, head])

                s0 = s0 * d0
                memory = Float32(0.0) + s0 * k0v
                s1 = s1 * d1
                memory = memory + s1 * k1v
                s2 = s2 * d2
                memory = memory + s2 * k2v
                s3 = s3 * d3
                memory = memory + s3 * k3v

                delta = (v_cur - _warp_sum(memory)) * b_cur

                s0 = s0 + k0v * delta
                o = Float32(0.0) + s0 * q0
                s1 = s1 + k1v * delta
                o = o + s1 * q1
                s2 = s2 + k2v * delta
                o = o + s2 * q2
                s3 = s3 + k3v * delta
                o = o + s3 * q3

                o = _warp_sum(o)
                if lane == Int32(0):
                    out[row, col + v_row] = o.to(BFloat16)

            if current >= Int64(0):
                st_global_v4_f32(self._state_row_addr(state, current, head, v_row, k0), s0, s1, s2, s3)


class GlmfKdaGatedNormFast:
    """Bit-exact ``GlmfKdaGatedNorm`` with 1 warp per ``(row, head)`` and vectorized v2 IO."""

    def __init__(self, *, heads: int, eps: float, gate_stride: int):
        self.heads = int(heads)
        self.width = self.heads * KDA_HEAD
        self.eps = float(eps)
        self.gate_stride = int(gate_stride)

    @cute.jit
    def __call__(
        self,
        o: cute.Pointer,
        gate: cute.Pointer,
        weight: cute.Pointer,
        y: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(o, gate, weight, y).launch(
            grid=(rows, self.heads, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, o: cute.Pointer, gate: cute.Pointer, weight: cute.Pointer, y: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        lane = Int64(cute.arch.thread_idx()[0])
        d_off = lane * Int64(4)
        col = head * Int64(KDA_HEAD) + d_off

        ow0, ow1 = ld_global_v2_u32(Int64(o.toint()) + (row * Int64(self.width) + col) * Int64(2))
        x0, x1 = _bf16_lo(ow0), _bf16_hi(ow0)
        x2, x3 = _bf16_lo(ow1), _bf16_hi(ow1)

        square = Float32(0.0)
        square = square + x0 * x0
        square = square + x1 * x1
        square = square + x2 * x2
        square = square + x3 * x3
        inv = _rsqrt(_warp_sum(square) / Float32(KDA_HEAD) + Float32(self.eps))

        gw0, gw1 = ld_global_v2_u32(Int64(gate.toint()) + (row * Int64(self.gate_stride) + col) * Int64(2))
        g0, g1 = _bf16_lo(gw0), _bf16_hi(gw0)
        g2, g3 = _bf16_lo(gw1), _bf16_hi(gw1)

        ww0, ww1 = ld_global_v2_u32(Int64(weight.toint()) + d_off * Int64(2))
        w0, w1 = _bf16_lo(ww0), _bf16_hi(ww0)
        w2, w3 = _bf16_lo(ww1), _bf16_hi(ww1)

        y0 = w0 * (x0 * inv) * _sigmoid(g0)
        y1 = w1 * (x1 * inv) * _sigmoid(g1)
        y2 = w2 * (x2 * inv) * _sigmoid(g2)
        y3 = w3 * (x3 * inv) * _sigmoid(g3)

        st_global_v2_u32(
            Int64(y.toint()) + (row * Int64(self.width) + col) * Int64(2),
            pack_f32x2_to_bfloat2(y0, y1),
            pack_f32x2_to_bfloat2(y2, y3),
        )


class _CoreStub:
    """Exposes the minimal metadata ``compile_kda_core_aot`` inspects on ``launch.core``."""

    def __init__(self, max_rows: int):
        self.replay_rows = max(int(max_rows), REPLAY_ROWS)


class _GlmfKdaCoreFast:
    """Drop-in replacement for ``_KdaCore`` specialized for GLM 5.3 Flash TP2 decode."""

    def __init__(self, g: GLMFGeometry, max_rows: int, decode: bool, state_dtype: str, state_rounding: str):
        self.g = g
        self.d = g.kda_width
        self.heads = g.kda_heads
        self.aux_width = 2 * g.kda_head_dim + g.kda_heads  # 288
        self.core = _CoreStub(max_rows)
        self.fab_offset = _align(kda_scratch_bytes(g, max_rows, 0))

        # Narrow, deep TMA tiles for (288, 4096): 18..72 CTAs vs stock's 5 CTAs, 100% bit-exact.
        self.fab = glm_projection(
            self.aux_width,
            g.hidden,
            tiles_by_rows=((64, 1, 2, 16, 16, 10),),
        )

        # Batched f_b(f_a) | g_b(g_a) reading directly from fab (no _KdaPack).
        # tile_n=32 -> 256 CTAs (vs stock's 64), num_stages=2 (k_tiles=2), row-matched compute_warps.
        fg_kw = dict(
            n=self.d,
            k=g.kda_head_dim,
            batch=2,
            a_row=self.aux_width,
            a_batch=g.kda_head_dim,
            o_row=2 * self.d,
            o_batch=self.d,
            tile_n=32,
            num_stages=2,
        )
        self.fg_16 = BatchedBf16Gemm(compute_warps=1, **fg_kw)
        self.fg_32 = BatchedBf16Gemm(compute_warps=2, **fg_kw)
        self.fg_64 = BatchedBf16Gemm(compute_warps=4, **fg_kw)

        self.conv_prep = GlmfKdaConvPrep(
            heads=self.heads,
            lower_bound=g.gate_lower_bound,
            fab_stride=self.aux_width,
            fg_stride=2 * self.d,
            replay_rows=self.core.replay_rows,
        )
        self.conv_state_plain = GlmfKdaConvStatePlain(width=self.d)
        self.recurrent = GlmfKdaRecurrentFast(heads=self.heads, replay_rows=self.core.replay_rows)
        self.norm = GlmfKdaGatedNormFast(heads=self.heads, eps=g.norm_eps, gate_stride=2 * self.d)

        # Fixed disjoint 1024-byte-aligned scratch layout inside [0, fab_offset):
        # fg (1.0 MiB) | q_prep (1.0) | k_prep (1.0) | decay_prep (1.0) | v_prep (0.5) | o (0.5) | beta (8 KiB)
        # Total = 5,251,072 B <= fab_offset (5,280,768 B at max_rows=64).
        cap = int(max_rows)
        self.fg_off = 0
        self.q_prep_off = self.fg_off + _align(cap * 2 * self.d * 2)
        self.k_prep_off = self.q_prep_off + _align(cap * self.d * 4)
        self.decay_prep_off = self.k_prep_off + _align(cap * self.d * 4)
        self.v_prep_off = self.decay_prep_off + _align(cap * self.d * 4)
        self.o_off = self.v_prep_off + _align(cap * self.d * 2)
        self.beta_prep_off = self.o_off + _align(cap * self.d * 2)
        end = self.beta_prep_off + _align(cap * self.heads * 4)
        if end > self.fab_offset:
            raise ValueError(f"fast KDA scratch layout ({end} B) exceeds fab_offset ({self.fab_offset} B)")

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        gate: cute.Pointer,
        w_fab: cute.Pointer,
        w_f: cute.Pointer,
        conv_w: cute.Pointer,
        a_log: cute.Pointer,
        dt_bias: cute.Pointer,
        o_norm: cute.Pointer,
        conv_state: cute.Pointer,
        state: cute.Pointer,
        slots: cute.Pointer,
        seq_first: cute.Pointer,
        out: cute.Pointer,
        replay: cute.Pointer,
        scratch: cute.Pointer,
        rows: Int32,
        spec: Int32,
        stream: cuda.CUstream,
    ):
        base = Int64(scratch.toint())
        fab = _ptr(BFloat16, base + Int64(self.fab_offset))
        fg = _ptr(BFloat16, base + Int64(self.fg_off))
        q_prep = _ptr(Float32, base + Int64(self.q_prep_off))
        k_prep = _ptr(Float32, base + Int64(self.k_prep_off))
        decay_prep = _ptr(Float32, base + Int64(self.decay_prep_off))
        v_prep = _ptr(BFloat16, base + Int64(self.v_prep_off))
        o_buf = _ptr(BFloat16, base + Int64(self.o_off))
        beta_prep = _ptr(Float32, base + Int64(self.beta_prep_off))

        self.fab(x, w_fab, fab, rows, stream)
        if rows <= Int32(16):
            self.fg_16(fab, w_f, fg, rows, stream)
        elif rows <= Int32(32):
            self.fg_32(fab, w_f, fg, rows, stream)
        else:
            self.fg_64(fab, w_f, fg, rows, stream)

        self.conv_prep(
            q, k, v, fab, fg, conv_w, a_log, dt_bias, conv_state, slots, seq_first,
            q_prep, k_prep, decay_prep, v_prep, beta_prep, replay, spec, rows, stream,
        )
        if spec == Int32(0):
            self.conv_state_plain(q, k, v, conv_state, slots, seq_first, rows, stream)

        self.recurrent(
            q_prep, k_prep, decay_prep, v_prep, beta_prep, state, slots, o_buf, replay, spec, rows, stream
        )
        gate_out = _ptr(BFloat16, base + Int64(self.fg_off) + Int64(self.d * 2))
        self.norm(o_buf, gate_out, o_norm, out, rows, stream)
