"""Sequence-aligned M=128 double-buffered SM120 prefill attention kernels for Inkling 2 Mini.

Serves only ``INKLING2_MINI`` (24 Q / 8 KV heads, group 3, head_dim 128,
window 1024, rel_extent 1024, ring_rows 1088, page_rows 64) on the two scored
prefill routes:

- :class:`InklingMiniGlobalFp8Prefill`: global causal FP8 KV cache
  (``inkling2_mini/global_attention_prefill_fp8_v2_m4096``).
- :class:`InklingMiniLocalBf16Prefill`: local sliding-window BF16 KV cache
  (``inkling2_mini/local_attention_prefill_bf16_v2_m4096``).

Design summary (Round 3):
- **Sequence-aligned M=128 pairing with unchanged ``rows_per_cta = 21`` ABI:**
  Launches ``grid = (tiles, 8, 1)`` in reverse tile order. Each CTA checks
  ``((tile_rows[tile_idx] - seq_first[row0]) // 21) & 1``: odd sequence-local
  tiles exit immediately, while even sequence-local tiles process up to 42
  same-sequence tokens (126 live M-rows) clamped at ``seq_first``. Because
  pairing depends solely on each sequence's own ``seq_first``, packed batches
  (including odd-tile packs like ``4x1024``) are bit-identical to standalone
  runs and row-count invariance holds by construction.
- **256 threads (8 active warps/CTA, 1 CTA/SM, 0 idle warps, up to 255 regs):**
  Halves KV L2 reads and FP8 K/V widening work per SM compared to M=64, while
  keeping all 8 warps (2 warps per SM sub-partition) busy on QK MMA, relative
  bias, online softmax, and register-sourced BF16 PV MMA (no ``sQ``, no ``s_p``,
  no ``cp.async.bulk`` / ``UTMA`` serialization).
- **True 2-stage swizzled SMEM double-buffering (83.9 KiB ≤ 101.4 KiB):**
  2-stage 16 KiB ``sK`` + 2-stage 16 KiB ``sV`` (``Swizzle<3,3,3>``, 0 bank
  conflicts) + 18 KiB ``s_rel`` (144 B stride, 0 bank conflicts).
  - **Local BF16:** 2-barrier-per-tile double-buffered ``cp.async`` (``LDGSTS``)
    pipeline for ``K``, ``V``, and ``rel`` (4 + 4 + 5 ``LDGSTS``/thread),
    overlapping ``K[t+1] + rel[t+1] + V[t+1]`` with ``PV[t]`` and ``V[t+1]``
    with ``QK[t+1]``, with zero register spills and L2-retained ``rel`` boundary
    lines.
  - **Global FP8:** 1-barrier-per-tile double-buffered Phase 1 loop beyond
    ``rel_extent`` with only 4 K + 4 V FP8 chunks widened per thread via SM120's
    exact ``_e4m3x8_scaled_bf16_sm120`` (matching stock's K/V rounding
    bit-for-bit across all 254 finite E4M3 values) and ``cp.async`` ``rel`` in
    Phase 2.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    fmax_f32,
    ld_global_nc_f32,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_trans_b16,
    mma_m16n8k16_f32_bf16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_v4_u32,
)
from b12x.attention.dense_mla._math import exp2_approx
from b12x.attention.paged._selected_forward import _cp_async_load_128b_zfill

from ._inkling2_kernels import _ld_shared_bf16_f32_off
from ._mimo_kernels import LOG2_E

__all__ = ["InklingMiniGlobalFp8Prefill", "InklingMiniLocalBf16Prefill"]

_HEADS = 24
_KV_HEADS = 8
_GROUP = 3
_HEAD_DIM = 128
_EXTENT = 1024
_WINDOW = 1024
_PAGE_ROWS = 64
_RING_ROWS = 1088

_ABI_TOKENS = 21
_MAX_TOKENS = 42  # 2 * 21 -> 126 live M-rows inside M=128
_TILE_M = 128
_TILE_N = 64
_LIVE_M = _MAX_TOKENS * _GROUP  # 126
_THREADS = 256  # 8 active warps/CTA, 1 CTA/SM, up to 255 regs/thread

# Relative bias staging: 64 distances + up to 7 alignment slop -> 72 BF16 = 144 B/row.
_REL_STAGE = 72
_REL_VECS = _REL_STAGE // 8  # 9 x 16B vectors per M-row
_REL_ROW_BYTES = _REL_STAGE * 2  # 144 B (16 B aligned, +4 banks/row -> 0 bank conflicts)
_REL_TILE_BYTES = _LIVE_M * _REL_ROW_BYTES  # 18,144 B
_REL_TOTAL_VECS = _LIVE_M * _REL_VECS  # 1,134 vectors -> 5 iterations across 256 threads
_REL_ITERS = (_REL_TOTAL_VECS + _THREADS - 1) // _THREADS  # 5

# 2-stage swizzled BF16 planes: 64 x 128 BF16 = 16,384 B/stage (Swizzle<3,3,3>: chunk ^ (row & 7)).
_BF16_PLANE_BYTES = _TILE_N * _HEAD_DIM * 2  # 16,384 B
_BF16_RECORD_BYTES = 4096

# Global FP8 layout: 2112 B record ([K: 8x128B | V: 8x128B | k_scale: 8x4B | v_scale: 8x4B]).
_FP8_RECORD_BYTES = 2112
_FP8_SCALE_OFFSET = 2048


@dsl_user_op
def _rcp_approx_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx_f32(val: Float32, src_lane: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip), Int32(src_lane).ir_value(loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x8_scaled_bf16_sm120(lo: Uint32, hi: Uint32, scale: Float32, *, loc=None, ip=None):
    """Exact ``bf16(f32(e4m3) * f32(scale))`` for 8 packed E4M3 bytes using SM120 direct cvt."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(lo).ir_value(loc=loc, ip=ip),
            Uint32(hi).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 e01, e23, e45, e67;
            .reg .b32 b01, b23, b45, b67;
            .reg .b32 u0, u1, u2, u3, u4, u5, u6, u7;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            mov.b32 {e01, e23}, $4;
            mov.b32 {e45, e67}, $5;
            cvt.rn.bf16x2.e4m3x2 b01, e01;
            cvt.rn.bf16x2.e4m3x2 b23, e23;
            cvt.rn.bf16x2.e4m3x2 b45, e45;
            cvt.rn.bf16x2.e4m3x2 b67, e67;
            shl.b32 u0, b01, 16;
            and.b32 u1, b01, 0xffff0000;
            shl.b32 u2, b23, 16;
            and.b32 u3, b23, 0xffff0000;
            shl.b32 u4, b45, 16;
            and.b32 u5, b45, 0xffff0000;
            shl.b32 u6, b67, 16;
            and.b32 u7, b67, 0xffff0000;
            mov.b32 f0, u0;
            mov.b32 f1, u1;
            mov.b32 f2, u2;
            mov.b32 f3, u3;
            mov.b32 f4, u4;
            mov.b32 f5, u5;
            mov.b32 f6, u6;
            mov.b32 f7, u7;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
            cvt.rn.bf16x2.f32 $2, f5, f4;
            cvt.rn.bf16x2.f32 $3, f7, f6;
        }
        """,
        "=r,=r,=r,=r,r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@cute.jit
def _qk_bf16_swiz(
    r_q0: cute.Tensor,
    r_q1: cute.Tensor,
    r_q2: cute.Tensor,
    r_q3: cute.Tensor,
    acc_s: cute.Tensor,
    sk_stage_addr: Int32,
    b_row: Int32,
    b_chunk: Int32,
):
    """Compute 16x64 raw FP32 ``Q @ K^T`` from Swizzle<3,3,3> 16 KiB BF16 ``sK``."""
    s_reg = [Float32(0.0) for _ in range(32)]
    for slab in cutlass.range_constexpr(4):
        swiz_c = (Int32(slab * 4) + b_chunk) ^ b_row
        base_addr = sk_stage_addr + b_row * Int32(256) + swiz_c * Int32(16)
        k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(base_addr)
        kb0 = slab * 2
        kb1 = slab * 2 + 1
        for nt in cutlass.range_constexpr(8):
            nk0, nk1, nk2, nk3 = k0, k1, k2, k3
            if const_expr(nt < 7):
                nk0, nk1, nk2, nk3 = ldmatrix_m8n8x4_b16(base_addr + Int32((nt + 1) * 2048))
            (
                s_reg[nt * 4],
                s_reg[nt * 4 + 1],
                s_reg[nt * 4 + 2],
                s_reg[nt * 4 + 3],
            ) = mma_m16n8k16_f32_bf16(
                s_reg[nt * 4],
                s_reg[nt * 4 + 1],
                s_reg[nt * 4 + 2],
                s_reg[nt * 4 + 3],
                r_q0[kb0],
                r_q1[kb0],
                r_q2[kb0],
                r_q3[kb0],
                k0,
                k1,
            )
            (
                s_reg[nt * 4],
                s_reg[nt * 4 + 1],
                s_reg[nt * 4 + 2],
                s_reg[nt * 4 + 3],
            ) = mma_m16n8k16_f32_bf16(
                s_reg[nt * 4],
                s_reg[nt * 4 + 1],
                s_reg[nt * 4 + 2],
                s_reg[nt * 4 + 3],
                r_q0[kb1],
                r_q1[kb1],
                r_q2[kb1],
                r_q3[kb1],
                k2,
                k3,
            )
            k0, k1, k2, k3 = nk0, nk1, nk2, nk3
    for i in cutlass.range_constexpr(32):
        acc_s[i] = s_reg[i]


@cute.jit
def _pv_bf16_rs_swiz(
    p_u32: cute.Tensor,
    acc_o: cute.Tensor,
    sv_stage_addr: Int32,
    v_row: Int32,
    v_row_lo: Int32,
    v_chunk: Int32,
):
    """Register-sourced BF16 ``P @ V`` from Swizzle<3,3,3> 16 KiB BF16 ``sV``."""
    o_reg = [acc_o[i] for i in range(64)]
    for ks in cutlass.range_constexpr(4):
        a0 = p_u32[ks * 4]
        a1 = p_u32[ks * 4 + 1]
        a2 = p_u32[ks * 4 + 2]
        a3 = p_u32[ks * 4 + 3]
        v_ks_row = sv_stage_addr + (Int32(ks * 16) + v_row) * Int32(256)
        swiz0 = v_chunk ^ v_row_lo
        v0, v1, v2, v3 = ldmatrix_m8n8x4_trans_b16(v_ks_row + swiz0 * Int32(16))
        for blk in cutlass.range_constexpr(8):
            nv0, nv1, nv2, nv3 = v0, v1, v2, v3
            if const_expr(blk < 7):
                nswiz = (Int32((blk + 1) * 2) + v_chunk) ^ v_row_lo
                nv0, nv1, nv2, nv3 = ldmatrix_m8n8x4_trans_b16(v_ks_row + nswiz * Int32(16))
            n0 = blk * 2
            o_reg[n0 * 4], o_reg[n0 * 4 + 1], o_reg[n0 * 4 + 2], o_reg[n0 * 4 + 3] = mma_m16n8k16_f32_bf16(
                o_reg[n0 * 4], o_reg[n0 * 4 + 1], o_reg[n0 * 4 + 2], o_reg[n0 * 4 + 3],
                a0, a1, a2, a3, v0, v1,
            )
            n1 = blk * 2 + 1
            o_reg[n1 * 4], o_reg[n1 * 4 + 1], o_reg[n1 * 4 + 2], o_reg[n1 * 4 + 3] = mma_m16n8k16_f32_bf16(
                o_reg[n1 * 4], o_reg[n1 * 4 + 1], o_reg[n1 * 4 + 2], o_reg[n1 * 4 + 3],
                a0, a1, a2, a3, v2, v3,
            )
            v0, v1, v2, v3 = nv0, nv1, nv2, nv3
    for i in cutlass.range_constexpr(64):
        acc_o[i] = o_reg[i]


@cute.jit
def _online_softmax_bf16(
    acc_s: cute.Tensor,
    acc_o: cute.Tensor,
    p_u32: cute.Tensor,
    m0_max: Float32,
    m1_max: Float32,
    l0_sum: Float32,
    l1_sum: Float32,
    scale_log2: Float32,
):
    """Online softmax (matching stock raw-score * scale_log2 order), rescale ``acc_o``, pack BF16 ``p_u32``."""
    rm0 = fmax_f32(acc_s[0], acc_s[1])
    rm1 = fmax_f32(acc_s[2], acc_s[3])
    for nt in cutlass.range_constexpr(1, 8):
        rm0 = fmax_f32(rm0, fmax_f32(acc_s[nt * 4], acc_s[nt * 4 + 1]))
        rm1 = fmax_f32(rm1, fmax_f32(acc_s[nt * 4 + 2], acc_s[nt * 4 + 3]))
    for off in (1, 2):
        rm0 = fmax_f32(rm0, cute.arch.shuffle_sync_bfly(rm0, offset=off))
        rm1 = fmax_f32(rm1, cute.arch.shuffle_sync_bfly(rm1, offset=off))

    rm0_log2 = cutlass.select_(rm0 == -Float32.inf, -Float32.inf, rm0 * scale_log2)
    rm1_log2 = cutlass.select_(rm1 == -Float32.inf, -Float32.inf, rm1 * scale_log2)
    new_m0 = fmax_f32(m0_max, rm0_log2)
    new_m1 = fmax_f32(m1_max, rm1_log2)
    safe_m0 = cutlass.select_(new_m0 == -Float32.inf, Float32(0.0), new_m0)
    safe_m1 = cutlass.select_(new_m1 == -Float32.inf, Float32(0.0), new_m1)

    alpha0 = exp2_approx(m0_max - safe_m0)
    alpha1 = exp2_approx(m1_max - safe_m1)
    l0_sum = l0_sum * alpha0
    l1_sum = l1_sum * alpha1
    if cute.arch.vote_any_sync(alpha0 != Float32(1.0)):
        for nt in cutlass.range_constexpr(16):
            acc_o[nt * 4] = acc_o[nt * 4] * alpha0
            acc_o[nt * 4 + 1] = acc_o[nt * 4 + 1] * alpha0
    if cute.arch.vote_any_sync(alpha1 != Float32(1.0)):
        for nt in cutlass.range_constexpr(16):
            acc_o[nt * 4 + 2] = acc_o[nt * 4 + 2] * alpha1
            acc_o[nt * 4 + 3] = acc_o[nt * 4 + 3] * alpha1

    m0_max = new_m0
    m1_max = new_m1

    s0_add = Float32(0.0)
    s1_add = Float32(0.0)
    for ks in cutlass.range_constexpr(4):
        nt0 = ks * 2
        nt1 = ks * 2 + 1
        p00 = exp2_approx(acc_s[nt0 * 4] * scale_log2 - safe_m0)
        p01 = exp2_approx(acc_s[nt0 * 4 + 1] * scale_log2 - safe_m0)
        p10 = exp2_approx(acc_s[nt0 * 4 + 2] * scale_log2 - safe_m1)
        p11 = exp2_approx(acc_s[nt0 * 4 + 3] * scale_log2 - safe_m1)
        q00 = exp2_approx(acc_s[nt1 * 4] * scale_log2 - safe_m0)
        q01 = exp2_approx(acc_s[nt1 * 4 + 1] * scale_log2 - safe_m0)
        q10 = exp2_approx(acc_s[nt1 * 4 + 2] * scale_log2 - safe_m1)
        q11 = exp2_approx(acc_s[nt1 * 4 + 3] * scale_log2 - safe_m1)
        s0_add = s0_add + (p00 + p01) + (q00 + q01)
        s1_add = s1_add + (p10 + p11) + (q10 + q11)
        p_u32[ks * 4] = pack_f32x2_to_bfloat2(p00, p01)
        p_u32[ks * 4 + 1] = pack_f32x2_to_bfloat2(p10, p11)
        p_u32[ks * 4 + 2] = pack_f32x2_to_bfloat2(q00, q01)
        p_u32[ks * 4 + 3] = pack_f32x2_to_bfloat2(q10, q11)
    l0_sum = l0_sum + s0_add
    l1_sum = l1_sum + s1_add
    return m0_max, m1_max, l0_sum, l1_sum


@cute.jit
def _cp_async_rel_tile(
    tidx: Int32,
    row0: Int64,
    live: Int32,
    kv_head: Int32,
    k0: Int64,
    rel_ptr: Int64,
    s_rel_base: Int32,
    token_pos: cute.Tensor,
):
    """Stage up to 126x72 BF16 relative biases via 5x 16-byte ``cp.async`` per thread (zero regs)."""
    for it in cutlass.range_constexpr(_REL_ITERS):
        lin = tidx + Int32(it * _THREADS)
        if lin < Int32(_REL_TOTAL_VECS):
            rm = lin // Int32(_REL_VECS)
            rv = lin - rm * Int32(_REL_VECS)
            rtl = rm // Int32(_GROUP)
            valid = rtl < live
            p = Int64(0)
            if valid:
                p = Int64(token_pos[rtl])
            base_d = (Int32(p - k0) - Int32(_TILE_N - 1)) & ~Int32(7)
            dist = base_d + rv * Int32(8)
            valid = valid & (dist >= Int32(0)) & (dist < Int32(_EXTENT))
            rhd = kv_head * Int32(_GROUP) + (rm - rtl * Int32(_GROUP))
            src = (
                rel_ptr
                + (((row0 + Int64(rtl)) * Int64(_HEADS) + Int64(rhd)) * Int64(_EXTENT) + Int64(dist))
                * Int64(2)
            )
            src = cutlass.select_(valid, src, rel_ptr)
            _cp_async_load_128b_zfill(
                s_rel_base + rm * Int32(_REL_ROW_BYTES) + rv * Int32(16),
                src,
                cutlass.select_(valid, Int32(16), Int32(0)),
            )


@cute.jit
def _apply_bias_and_mask(
    acc_s: cute.Tensor,
    k0: Int64,
    hi: Int64,
    p_min: Int64,
    p_max: Int64,
    valid0: Int32,
    valid1: Int32,
    pos0: Int64,
    pos1: Int64,
    rel_base_m0: Int32,
    rel_base_m1: Int32,
    tid: Int32,
    bias_scale: Float32,
    is_local: cutlass.Constexpr,
):
    """Add staged relative bias (as ``bias / scale``) and apply causal / sliding-window mask."""
    full_biased = (k0 + Int64(_TILE_N - 1) <= p_min) & (p_max - k0 < Int64(_EXTENT))
    if full_biased:
        if valid0 != Int32(0):
            rem0 = (Int32(pos0 - k0) - Int32(_TILE_N - 1)) & Int32(7)
            raddr0 = rel_base_m0 + rem0 * Int32(2)
            for nt in cutlass.range_constexpr(8):
                b0 = _ld_shared_bf16_f32_off(raddr0, 2 * (57 - nt * 8)) * bias_scale
                b1 = _ld_shared_bf16_f32_off(raddr0, 2 * (56 - nt * 8)) * bias_scale
                acc_s[nt * 4] = acc_s[nt * 4] + b0
                acc_s[nt * 4 + 1] = acc_s[nt * 4 + 1] + b1
        else:
            for nt in cutlass.range_constexpr(8):
                acc_s[nt * 4] = -Float32.inf
                acc_s[nt * 4 + 1] = -Float32.inf
        if valid1 != Int32(0):
            rem1 = (Int32(pos1 - k0) - Int32(_TILE_N - 1)) & Int32(7)
            raddr1 = rel_base_m1 + rem1 * Int32(2)
            for nt in cutlass.range_constexpr(8):
                b0 = _ld_shared_bf16_f32_off(raddr1, 2 * (57 - nt * 8)) * bias_scale
                b1 = _ld_shared_bf16_f32_off(raddr1, 2 * (56 - nt * 8)) * bias_scale
                acc_s[nt * 4 + 2] = acc_s[nt * 4 + 2] + b0
                acc_s[nt * 4 + 3] = acc_s[nt * 4 + 3] + b1
        else:
            for nt in cutlass.range_constexpr(8):
                acc_s[nt * 4 + 2] = -Float32.inf
                acc_s[nt * 4 + 3] = -Float32.inf
    else:
        if valid0 != Int32(0):
            d0_0 = Int32(pos0 - k0)
            rem0 = (d0_0 - Int32(_TILE_N - 1)) & Int32(7)
            raddr0 = rel_base_m0 + rem0 * Int32(2)
            for nt in cutlass.range_constexpr(8):
                col0 = Int32(nt * 8) + tid * Int32(2)
                col1 = col0 + Int32(1)
                dist0 = d0_0 - col0
                dist1 = d0_0 - col1
                b0 = _ld_shared_bf16_f32_off(raddr0, 2 * (57 - nt * 8)) * bias_scale
                b1 = _ld_shared_bf16_f32_off(raddr0, 2 * (56 - nt * 8)) * bias_scale
                if const_expr(is_local):
                    ok0 = (dist0 >= Int32(0)) & (dist0 < Int32(_WINDOW)) & ((k0 + Int64(col0)) <= hi)
                    ok1 = (dist1 >= Int32(0)) & (dist1 < Int32(_WINDOW)) & ((k0 + Int64(col1)) <= hi)
                    acc_s[nt * 4] = cutlass.select_(ok0, acc_s[nt * 4] + b0, -Float32.inf)
                    acc_s[nt * 4 + 1] = cutlass.select_(ok1, acc_s[nt * 4 + 1] + b1, -Float32.inf)
                else:
                    v0 = acc_s[nt * 4] + cutlass.select_((dist0 >= Int32(0)) & (dist0 < Int32(_EXTENT)), b0, Float32(0.0))
                    v1 = acc_s[nt * 4 + 1] + cutlass.select_((dist1 >= Int32(0)) & (dist1 < Int32(_EXTENT)), b1, Float32(0.0))
                    acc_s[nt * 4] = cutlass.select_((dist0 < Int32(0)) | ((k0 + Int64(col0)) > hi), -Float32.inf, v0)
                    acc_s[nt * 4 + 1] = cutlass.select_((dist1 < Int32(0)) | ((k0 + Int64(col1)) > hi), -Float32.inf, v1)
        else:
            for nt in cutlass.range_constexpr(8):
                acc_s[nt * 4] = -Float32.inf
                acc_s[nt * 4 + 1] = -Float32.inf
        if valid1 != Int32(0):
            d0_1 = Int32(pos1 - k0)
            rem1 = (d0_1 - Int32(_TILE_N - 1)) & Int32(7)
            raddr1 = rel_base_m1 + rem1 * Int32(2)
            for nt in cutlass.range_constexpr(8):
                col0 = Int32(nt * 8) + tid * Int32(2)
                col1 = col0 + Int32(1)
                dist0 = d0_1 - col0
                dist1 = d0_1 - col1
                b0 = _ld_shared_bf16_f32_off(raddr1, 2 * (57 - nt * 8)) * bias_scale
                b1 = _ld_shared_bf16_f32_off(raddr1, 2 * (56 - nt * 8)) * bias_scale
                if const_expr(is_local):
                    ok0 = (dist0 >= Int32(0)) & (dist0 < Int32(_WINDOW)) & ((k0 + Int64(col0)) <= hi)
                    ok1 = (dist1 >= Int32(0)) & (dist1 < Int32(_WINDOW)) & ((k0 + Int64(col1)) <= hi)
                    acc_s[nt * 4 + 2] = cutlass.select_(ok0, acc_s[nt * 4 + 2] + b0, -Float32.inf)
                    acc_s[nt * 4 + 3] = cutlass.select_(ok1, acc_s[nt * 4 + 3] + b1, -Float32.inf)
                else:
                    v0 = acc_s[nt * 4 + 2] + cutlass.select_((dist0 >= Int32(0)) & (dist0 < Int32(_EXTENT)), b0, Float32(0.0))
                    v1 = acc_s[nt * 4 + 3] + cutlass.select_((dist1 >= Int32(0)) & (dist1 < Int32(_EXTENT)), b1, Float32(0.0))
                    acc_s[nt * 4 + 2] = cutlass.select_((dist0 < Int32(0)) | ((k0 + Int64(col0)) > hi), -Float32.inf, v0)
                    acc_s[nt * 4 + 3] = cutlass.select_((dist1 < Int32(0)) | ((k0 + Int64(col1)) > hi), -Float32.inf, v1)
        else:
            for nt in cutlass.range_constexpr(8):
                acc_s[nt * 4 + 2] = -Float32.inf
                acc_s[nt * 4 + 3] = -Float32.inf


@cute.jit
def _finalize_and_store_output(
    acc_o: cute.Tensor,
    m0_max: Float32,
    m1_max: Float32,
    l0_sum: Float32,
    l1_sum: Float32,
    valid0: Int32,
    valid1: Int32,
    row0: Int64,
    t_local0: Int32,
    t_local1: Int32,
    head0: Int32,
    head1: Int32,
    sinks: cute.Pointer,
    out: cute.Pointer,
    tid: Int32,
):
    """Reduce row sums across 4 lanes, add per-(row, head) sink, normalize and store BF16 ``out``."""
    for off in (1, 2):
        l0_sum = l0_sum + cute.arch.shuffle_sync_bfly(l0_sum, offset=off)
        l1_sum = l1_sum + cute.arch.shuffle_sync_bfly(l1_sum, offset=off)

    inv0 = Float32(0.0)
    if valid0 != Int32(0):
        sink_idx0 = (row0 + Int64(t_local0)) * Int64(_HEADS) + Int64(head0)
        sink0_log2 = ld_global_nc_f32(Int64(sinks.toint()) + sink_idx0 * Int64(4)) * Float32(LOG2_E)
        top0 = fmax_f32(m0_max, sink0_log2)
        res0 = exp2_approx(m0_max - top0)
        denom0 = l0_sum * res0 + exp2_approx(sink0_log2 - top0)
        bad0 = (denom0 == Float32(0.0)) | (denom0 != denom0)
        inv0 = cutlass.select_(
            bad0,
            Float32(0.0),
            res0 * _rcp_approx_f32(cutlass.select_(bad0, Float32(1.0), denom0)),
        )

    inv1 = Float32(0.0)
    if valid1 != Int32(0):
        sink_idx1 = (row0 + Int64(t_local1)) * Int64(_HEADS) + Int64(head1)
        sink1_log2 = ld_global_nc_f32(Int64(sinks.toint()) + sink_idx1 * Int64(4)) * Float32(LOG2_E)
        top1 = fmax_f32(m1_max, sink1_log2)
        res1 = exp2_approx(m1_max - top1)
        denom1 = l1_sum * res1 + exp2_approx(sink1_log2 - top1)
        bad1 = (denom1 == Float32(0.0)) | (denom1 != denom1)
        inv1 = cutlass.select_(
            bad1,
            Float32(0.0),
            res1 * _rcp_approx_f32(cutlass.select_(bad1, Float32(1.0), denom1)),
        )

    if valid0 != Int32(0):
        out_base0 = (
            Int64(out.toint())
            + (((row0 + Int64(t_local0)) * Int64(_HEADS) + Int64(head0)) * Int64(_HEAD_DIM) + Int64(tid * Int32(2)))
            * Int64(2)
        )
        for nt in cutlass.range_constexpr(16):
            packed = pack_f32x2_to_bfloat2(acc_o[nt * 4] * inv0, acc_o[nt * 4 + 1] * inv0)
            st_global_u32(out_base0 + Int64(nt * 16), packed)

    if valid1 != Int32(0):
        out_base1 = (
            Int64(out.toint())
            + (((row0 + Int64(t_local1)) * Int64(_HEADS) + Int64(head1)) * Int64(_HEAD_DIM) + Int64(tid * Int32(2)))
            * Int64(2)
        )
        for nt in cutlass.range_constexpr(16):
            packed = pack_f32x2_to_bfloat2(acc_o[nt * 4 + 2] * inv1, acc_o[nt * 4 + 3] * inv1)
            st_global_u32(out_base1 + Int64(nt * 16), packed)


@cute.jit
def _load_q_regs(
    q: cute.Pointer,
    row0: Int64,
    t_local0: Int32,
    t_local1: Int32,
    head0: Int32,
    head1: Int32,
    valid0: Int32,
    valid1: Int32,
    tid: Int32,
    r_q0: cute.Tensor,
    r_q1: cute.Tensor,
    r_q2: cute.Tensor,
    r_q3: cute.Tensor,
):
    """Load 16x128 BF16 query tile directly into MMA A-fragment registers (no ``sQ``)."""
    q_ptr = Int64(q.toint())
    q_addr0 = q_ptr + ((row0 + Int64(t_local0)) * Int64(_HEADS) + Int64(head0)) * Int64(256) + Int64(tid * Int32(4))
    q_addr1 = q_ptr + ((row0 + Int64(t_local1)) * Int64(_HEADS) + Int64(head1)) * Int64(256) + Int64(tid * Int32(4))
    r_q0.fill(Uint32(0))
    r_q1.fill(Uint32(0))
    r_q2.fill(Uint32(0))
    r_q3.fill(Uint32(0))
    if valid0 != Int32(0):
        for blk in cutlass.range_constexpr(8):
            r_q0[blk] = ld_global_nc_u32(q_addr0 + Int64(blk * 32))
            r_q2[blk] = ld_global_nc_u32(q_addr0 + Int64(blk * 32 + 16))
    if valid1 != Int32(0):
        for blk in cutlass.range_constexpr(8):
            r_q1[blk] = ld_global_nc_u32(q_addr1 + Int64(blk * 32))
            r_q3[blk] = ld_global_nc_u32(q_addr1 + Int64(blk * 32 + 16))


@cute.jit
def _prefetch_fp8_plane_256(
    page_base: Int64,
    k0: Int64,
    hi: Int64,
    plane_off: Int64,
    sc_off: Int64,
    tok0: Int32,
    chunk: Int32,
    raw_lo: cute.Tensor,
    raw_hi: cute.Tensor,
    sc_arr: cute.Tensor,
):
    """256-thread coalesced load of 64x128 E4M3 plane (4 chunks/thread) + 1-per-16-lanes scale."""
    plane_col = plane_off + Int64(chunk * Int32(8))
    for it in cutlass.range_constexpr(4):
        tok = tok0 + Int32(it * 16)
        lo = Uint32(0)
        hi_u = Uint32(0)
        sc = Float32(0.0)
        if (k0 + Int64(tok)) <= hi:
            rec = page_base + Int64(tok) * Int64(_FP8_RECORD_BYTES)
            lo, hi_u = ld_global_nc_v2_u32(rec + plane_col)
            if chunk == Int32(0):
                sc = ld_global_nc_f32(rec + sc_off)
        raw_lo[it] = lo
        raw_hi[it] = hi_u
        sc_arr[it] = sc


@cute.jit
def _widen_store_fp8_plane_256(
    raw_lo: cute.Tensor,
    raw_hi: cute.Tensor,
    sc_arr: cute.Tensor,
    s_dst_base: Int32,
    tok0: Int32,
    swiz_col: Int32,
    half_leader: Int32,
):
    """256-thread widen (4 chunks/thread) of exact ``bf16(f32(e4m3)*f32(scale))`` into Swizzle<3,3,3> SMEM."""
    dst_base = s_dst_base + tok0 * Int32(256) + swiz_col
    for it in cutlass.range_constexpr(4):
        sc = _shfl_idx_f32(sc_arr[it], half_leader)
        w0, w1, w2, w3 = _e4m3x8_scaled_bf16_sm120(raw_lo[it], raw_hi[it], sc)
        st_shared_v4_u32(dst_base + Int32(it * 4096), w0, w1, w2, w3)


@cute.jit
def _cp_async_local_plane_256(
    k0: Int64,
    hi: Int64,
    seq_start: Int64,
    step_first: Int64,
    ring_base: Int64,
    step_ptr: Int64,
    ring_ptr: Int64,
    plane_off: Int64,
    tok0: Int32,
    chunk: Int32,
    swiz_col: Int32,
    s_dst_base: Int32,
):
    """Issue 4x 16-byte ``cp.async`` loads per thread (256 threads) into Swizzle<3,3,3> BF16 SMEM."""
    dst_base = s_dst_base + tok0 * Int32(256) + swiz_col
    col_off = plane_off + Int64(chunk * Int32(16))
    if k0 >= seq_start:
        step_k0 = step_ptr + (step_first + k0 - seq_start) * Int64(_BF16_RECORD_BYTES) + col_off
        for it in cutlass.range_constexpr(4):
            tok = tok0 + Int32(it * 16)
            valid = (k0 + Int64(tok)) <= hi
            src = cutlass.select_(
                valid,
                step_k0 + Int64(tok) * Int64(_BF16_RECORD_BYTES),
                step_ptr,
            )
            _cp_async_load_128b_zfill(
                dst_base + Int32(it * 4096),
                src,
                cutlass.select_(valid, Int32(16), Int32(0)),
            )
    else:
        k_mod = k0 % Int64(_RING_ROWS)
        for it in cutlass.range_constexpr(4):
            tok = tok0 + Int32(it * 16)
            kp = k0 + Int64(tok)
            valid = kp <= hi
            src = ring_ptr
            if kp >= seq_start:
                src = step_ptr + (step_first + kp - seq_start) * Int64(_BF16_RECORD_BYTES) + col_off
            else:
                slot = k_mod + Int64(tok)
                if slot >= Int64(_RING_ROWS):
                    slot = slot - Int64(_RING_ROWS)
                src = ring_ptr + (ring_base + slot) * Int64(_BF16_RECORD_BYTES) + col_off
            src = cutlass.select_(valid, src, ring_ptr)
            _cp_async_load_128b_zfill(
                dst_base + Int32(it * 4096),
                src,
                cutlass.select_(valid, Int32(16), Int32(0)),
            )


def _make_storage_2stage_cls():
    class Storage:
        pass

    Storage.__annotations__ = {
        "sk": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * _BF16_PLANE_BYTES], 128],
        "sv": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * _BF16_PLANE_BYTES], 128],
        "s_rel": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, _REL_TILE_BYTES], 128],
        "token_pos": cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 48], 16],
    }
    return cute.struct(Storage)


# ============================================================================
# Global Causal FP8 Prefill (INKLING2_MINI)
# ============================================================================


class InklingMiniGlobalFp8Prefill:
    """Sequence-aligned M=128 double-buffered global FP8 prefill for Inkling 2 Mini."""

    tokens = _ABI_TOKENS

    def __init__(self, *, softmax_scale: float):
        self.softmax_scale = float(softmax_scale)
        self.scale_log2 = float(softmax_scale) * LOG2_E
        self.bias_scale = 1.0 / float(softmax_scale)

    def key(self) -> tuple:
        return ("inkling2_mini_global_fp8_prefill_m128_v4", self.softmax_scale)

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        cache: cute.Pointer,
        kv_step: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        table_indices: cute.Pointer,
        tile_rows: cute.Pointer,
        ring_slots: cute.Pointer,
        seq_first: cute.Pointer,
        rel: cute.Pointer,
        sinks: cute.Pointer,
        out: cute.Pointer,
        partial_out: cute.Pointer,
        partial_lse: cute.Pointer,
        rows: Int32,
        table_stride: Int32,
        tiles: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ):
        if tiles > Int32(0):
            self.kernel(
                q, cache, positions, page_table, table_indices, tile_rows, seq_first,
                rel, sinks, out, rows, table_stride, tiles,
            ).launch(
                grid=(tiles, _KV_HEADS, 1),
                block=[_THREADS, 1, 1],
                min_blocks_per_mp=1,
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        cache: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        table_indices: cute.Pointer,
        tile_rows: cute.Pointer,
        seq_first: cute.Pointer,
        rel: cute.Pointer,
        sinks: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        table_stride: Int32,
        tiles: Int32,
    ):
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_make_storage_2stage_cls())

        tile_idx = tiles - Int32(1) - Int32(cute.arch.block_idx()[0])
        row0 = Int64(tile_rows[Int64(tile_idx)])
        sf0 = Int32(seq_first[row0])
        # Sequence-aligned pairing: odd tiles within each sequence exit immediately.
        if (((Int32(row0) - sf0) // Int32(_ABI_TOKENS)) & Int32(1)) == Int32(0):
            kv_head = Int32(cute.arch.block_idx()[1])
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx >> Int32(5)
            lane = tidx & Int32(31)

            sk_base = shared_ptr_to_u32(storage.sk.data_ptr())
            sv_base = shared_ptr_to_u32(storage.sv.data_ptr())
            s_rel_base = shared_ptr_to_u32(storage.s_rel.data_ptr())
            token_pos = storage.token_pos.get_tensor(cute.make_layout((48,)))

            live = rows - Int32(row0)
            if live > Int32(_MAX_TOKENS):
                live = Int32(_MAX_TOKENS)
            for t in cutlass.range_constexpr(1, _MAX_TOKENS):
                if Int32(t) < live:
                    if Int32(seq_first[row0 + Int64(t)]) != sf0:
                        live = Int32(t)

            if tidx < Int32(_MAX_TOKENS):
                p_val = Int64(-1)
                if tidx < live:
                    p_val = Int64(positions[row0 + Int64(tidx)])
                token_pos[tidx] = p_val
            cute.arch.sync_threads()

            p_first = Int64(token_pos[0])
            p_min = p_first
            p_max = p_first
            for t in cutlass.range_constexpr(1, _MAX_TOKENS):
                if Int32(t) < live:
                    pt = Int64(token_pos[t])
                    p_min = cutlass.select_(pt < p_min, pt, p_min)
                    p_max = cutlass.select_(pt > p_max, pt, p_max)
            hi = p_max
            tile_count = Int32((hi + Int64(_TILE_N)) // Int64(_TILE_N))
            if live <= Int32(0):
                tile_count = Int32(0)
            first_rel_tile = Int32(0)
            if p_min >= Int64(_EXTENT + _TILE_N - 1):
                first_rel_tile = Int32((p_min - Int64(_EXTENT + _TILE_N - 1)) // Int64(_TILE_N)) + Int32(1)
            if first_rel_tile > tile_count:
                first_rel_tile = tile_count
            table_base = Int64(table_indices[row0]) * Int64(table_stride)

            gid = lane >> Int32(2)
            tid = lane & Int32(3)
            m0 = warp_id * Int32(16) + gid
            m1 = m0 + Int32(8)
            t_local0 = m0 // Int32(_GROUP)
            head0 = kv_head * Int32(_GROUP) + (m0 - t_local0 * Int32(_GROUP))
            t_local1 = m1 // Int32(_GROUP)
            head1 = kv_head * Int32(_GROUP) + (m1 - t_local1 * Int32(_GROUP))
            live_m = live * Int32(_GROUP)
            valid0 = cutlass.select_(m0 < live_m, Int32(1), Int32(0))
            valid1 = cutlass.select_(m1 < live_m, Int32(1), Int32(0))
            pos0 = Int64(-1)
            pos1 = Int64(-1)
            if valid0 != Int32(0):
                pos0 = Int64(token_pos[t_local0])
            if valid1 != Int32(0):
                pos1 = Int64(token_pos[t_local1])

            b_row = lane & Int32(7)
            b_chunk = lane >> Int32(3)
            v_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            v_row_lo = lane & Int32(7)
            v_chunk = lane >> Int32(4)

            tok0 = tidx >> Int32(4)
            chunk = tidx & Int32(15)
            swiz_col = (chunk ^ (tok0 & Int32(7))) * Int32(16)
            half_leader = lane & Int32(16)

            cache_base = Int64(cache.toint())
            rel_ptr = Int64(rel.toint())
            k_plane_off = Int64(kv_head) * Int64(_HEAD_DIM)
            v_plane_off = Int64(_KV_HEADS * _HEAD_DIM) + k_plane_off
            k_sc_off = Int64(_FP8_SCALE_OFFSET) + Int64(kv_head) * Int64(4)
            v_sc_off = Int64(_FP8_SCALE_OFFSET + _KV_HEADS * 4) + Int64(kv_head) * Int64(4)
            scale_log2 = Float32(self.scale_log2)
            bias_scale = Float32(self.bias_scale)
            rel_base_m0 = s_rel_base + (m0 * Int32(_REL_STAGE) + Int32(6) - tid * Int32(2)) * Int32(2)
            rel_base_m1 = s_rel_base + (m1 * Int32(_REL_STAGE) + Int32(6) - tid * Int32(2)) * Int32(2)

            r_q0 = cute.make_rmem_tensor((8,), Uint32)
            r_q1 = cute.make_rmem_tensor((8,), Uint32)
            r_q2 = cute.make_rmem_tensor((8,), Uint32)
            r_q3 = cute.make_rmem_tensor((8,), Uint32)
            _load_q_regs(q, row0, t_local0, t_local1, head0, head1, valid0, valid1, tid, r_q0, r_q1, r_q2, r_q3)

            acc_o = cute.make_rmem_tensor((64,), Float32)
            acc_s = cute.make_rmem_tensor((32,), Float32)
            p_u32 = cute.make_rmem_tensor((16,), Uint32)
            k_lo = cute.make_rmem_tensor((4,), Uint32)
            k_hi = cute.make_rmem_tensor((4,), Uint32)
            k_sc = cute.make_rmem_tensor((4,), Float32)
            v_lo = cute.make_rmem_tensor((4,), Uint32)
            v_hi = cute.make_rmem_tensor((4,), Uint32)
            v_sc = cute.make_rmem_tensor((4,), Float32)

            acc_o.fill(Float32(0.0))
            m0_max = -Float32.inf
            m1_max = -Float32.inf
            l0_sum = Float32(0.0)
            l1_sum = Float32(0.0)

            # Prologue: stage K[0] + V[0] (+ rel[0] if Phase 2 starts at 0) into stage 0,
            # and launch non-blocking prefetch of K[1] + V[1] into registers.
            if tile_count > Int32(0):
                page0 = Int64(Int32(page_table[table_base]))
                pb0 = cache_base + page0 * Int64(_PAGE_ROWS * _FP8_RECORD_BYTES)
                _prefetch_fp8_plane_256(pb0, Int64(0), hi, k_plane_off, k_sc_off, tok0, chunk, k_lo, k_hi, k_sc)
                _prefetch_fp8_plane_256(pb0, Int64(0), hi, v_plane_off, v_sc_off, tok0, chunk, v_lo, v_hi, v_sc)
                if first_rel_tile == Int32(0):
                    _cp_async_rel_tile(tidx, row0, live, kv_head, Int64(0), rel_ptr, s_rel_base, token_pos)
                    cute.arch.cp_async_commit_group()
                _widen_store_fp8_plane_256(k_lo, k_hi, k_sc, sk_base, tok0, swiz_col, half_leader)
                _widen_store_fp8_plane_256(v_lo, v_hi, v_sc, sv_base, tok0, swiz_col, half_leader)
                if tile_count > Int32(1):
                    page1 = Int64(Int32(page_table[table_base + Int64(1)]))
                    pb1 = cache_base + page1 * Int64(_PAGE_ROWS * _FP8_RECORD_BYTES)
                    _prefetch_fp8_plane_256(pb1, Int64(_TILE_N), hi, k_plane_off, k_sc_off, tok0, chunk, k_lo, k_hi, k_sc)
                    _prefetch_fp8_plane_256(pb1, Int64(_TILE_N), hi, v_plane_off, v_sc_off, tok0, chunk, v_lo, v_hi, v_sc)
                if first_rel_tile == Int32(0):
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

            # ---- Phase 1: Zero-bias, unmasked 1-barrier double-buffered loop [0, first_rel_tile) ----
            for tile in cutlass.range(first_rel_tile, unroll=1):
                stage = tile & Int32(1)
                nxt_stage = stage ^ Int32(1)
                sk_cur = sk_base + stage * Int32(_BF16_PLANE_BYTES)
                sv_cur = sv_base + stage * Int32(_BF16_PLANE_BYTES)
                sk_nxt = sk_base + nxt_stage * Int32(_BF16_PLANE_BYTES)
                sv_nxt = sv_base + nxt_stage * Int32(_BF16_PLANE_BYTES)

                _qk_bf16_swiz(r_q0, r_q1, r_q2, r_q3, acc_s, sk_cur, b_row, b_chunk)
                if valid0 == Int32(0):
                    for nt in cutlass.range_constexpr(8):
                        acc_s[nt * 4] = -Float32.inf
                        acc_s[nt * 4 + 1] = -Float32.inf
                if valid1 == Int32(0):
                    for nt in cutlass.range_constexpr(8):
                        acc_s[nt * 4 + 2] = -Float32.inf
                        acc_s[nt * 4 + 3] = -Float32.inf
                m0_max, m1_max, l0_sum, l1_sum = _online_softmax_bf16(
                    acc_s, acc_o, p_u32, m0_max, m1_max, l0_sum, l1_sum, scale_log2
                )
                _pv_bf16_rs_swiz(p_u32, acc_o, sv_cur, v_row, v_row_lo, v_chunk)

                next_tile = tile + Int32(1)
                if next_tile < tile_count:
                    _widen_store_fp8_plane_256(k_lo, k_hi, k_sc, sk_nxt, tok0, swiz_col, half_leader)
                    _widen_store_fp8_plane_256(v_lo, v_hi, v_sc, sv_nxt, tok0, swiz_col, half_leader)
                    cute.arch.sync_threads()
                    next2 = tile + Int32(2)
                    if next2 < tile_count:
                        page_n2 = Int64(Int32(page_table[table_base + Int64(next2)]))
                        pb_n2 = cache_base + page_n2 * Int64(_PAGE_ROWS * _FP8_RECORD_BYTES)
                        k0_n2 = Int64(next2) * Int64(_TILE_N)
                        _prefetch_fp8_plane_256(pb_n2, k0_n2, hi, k_plane_off, k_sc_off, tok0, chunk, k_lo, k_hi, k_sc)
                        _prefetch_fp8_plane_256(pb_n2, k0_n2, hi, v_plane_off, v_sc_off, tok0, chunk, v_lo, v_hi, v_sc)

            # Transition Phase 1 -> Phase 2: stage rel[first_rel_tile] once.
            if (first_rel_tile > Int32(0)) & (first_rel_tile < tile_count):
                k0_first_rel = Int64(first_rel_tile) * Int64(_TILE_N)
                _cp_async_rel_tile(tidx, row0, live, kv_head, k0_first_rel, rel_ptr, s_rel_base, token_pos)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

            # ---- Phase 2: Biased & causal-diagonal tiles [first_rel_tile, tile_count) ----
            for tile_off in cutlass.range(tile_count - first_rel_tile, unroll=1):
                tile = first_rel_tile + Int32(tile_off)
                k0 = Int64(tile) * Int64(_TILE_N)
                stage = tile & Int32(1)
                nxt_stage = stage ^ Int32(1)
                sk_cur = sk_base + stage * Int32(_BF16_PLANE_BYTES)
                sv_cur = sv_base + stage * Int32(_BF16_PLANE_BYTES)
                sk_nxt = sk_base + nxt_stage * Int32(_BF16_PLANE_BYTES)
                sv_nxt = sv_base + nxt_stage * Int32(_BF16_PLANE_BYTES)

                _qk_bf16_swiz(r_q0, r_q1, r_q2, r_q3, acc_s, sk_cur, b_row, b_chunk)
                _apply_bias_and_mask(
                    acc_s, k0, hi, p_min, p_max, valid0, valid1, pos0, pos1,
                    rel_base_m0, rel_base_m1, tid, bias_scale, False,
                )
                m0_max, m1_max, l0_sum, l1_sum = _online_softmax_bf16(
                    acc_s, acc_o, p_u32, m0_max, m1_max, l0_sum, l1_sum, scale_log2
                )
                cute.arch.sync_threads()

                next_tile = tile + Int32(1)
                if next_tile < tile_count:
                    _cp_async_rel_tile(
                        tidx, row0, live, kv_head, Int64(next_tile) * Int64(_TILE_N), rel_ptr, s_rel_base, token_pos
                    )
                    cute.arch.cp_async_commit_group()

                _pv_bf16_rs_swiz(p_u32, acc_o, sv_cur, v_row, v_row_lo, v_chunk)

                if next_tile < tile_count:
                    _widen_store_fp8_plane_256(k_lo, k_hi, k_sc, sk_nxt, tok0, swiz_col, half_leader)
                    _widen_store_fp8_plane_256(v_lo, v_hi, v_sc, sv_nxt, tok0, swiz_col, half_leader)
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    next2 = tile + Int32(2)
                    if next2 < tile_count:
                        page_n2 = Int64(Int32(page_table[table_base + Int64(next2)]))
                        pb_n2 = cache_base + page_n2 * Int64(_PAGE_ROWS * _FP8_RECORD_BYTES)
                        k0_n2 = Int64(next2) * Int64(_TILE_N)
                        _prefetch_fp8_plane_256(pb_n2, k0_n2, hi, k_plane_off, k_sc_off, tok0, chunk, k_lo, k_hi, k_sc)
                        _prefetch_fp8_plane_256(pb_n2, k0_n2, hi, v_plane_off, v_sc_off, tok0, chunk, v_lo, v_hi, v_sc)

            _finalize_and_store_output(
                acc_o, m0_max, m1_max, l0_sum, l1_sum,
                valid0, valid1, row0, t_local0, t_local1, head0, head1,
                sinks, out, tid,
            )


# ============================================================================
# Local Sliding-Window BF16 Prefill (INKLING2_MINI)
# ============================================================================


class InklingMiniLocalBf16Prefill:
    """Sequence-aligned M=128 2-barrier double-buffered local BF16 prefill for Inkling 2 Mini."""

    tokens = _ABI_TOKENS

    def __init__(self, *, softmax_scale: float):
        self.softmax_scale = float(softmax_scale)
        self.scale_log2 = float(softmax_scale) * LOG2_E
        self.bias_scale = 1.0 / float(softmax_scale)

    def key(self) -> tuple:
        return ("inkling2_mini_local_bf16_prefill_m128_v3", self.softmax_scale)

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        cache: cute.Pointer,
        kv_step: cute.Pointer,
        positions: cute.Pointer,
        page_table: cute.Pointer,
        table_indices: cute.Pointer,
        tile_rows: cute.Pointer,
        ring_slots: cute.Pointer,
        seq_first: cute.Pointer,
        rel: cute.Pointer,
        sinks: cute.Pointer,
        out: cute.Pointer,
        partial_out: cute.Pointer,
        partial_lse: cute.Pointer,
        rows: Int32,
        table_stride: Int32,
        tiles: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ):
        if tiles > Int32(0):
            self.kernel(
                q, cache, kv_step, positions, tile_rows, ring_slots, seq_first,
                rel, sinks, out, rows, tiles,
            ).launch(
                grid=(tiles, _KV_HEADS, 1),
                block=[_THREADS, 1, 1],
                min_blocks_per_mp=1,
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        ring: cute.Pointer,
        kv_step: cute.Pointer,
        positions: cute.Pointer,
        tile_rows: cute.Pointer,
        ring_slots: cute.Pointer,
        seq_first: cute.Pointer,
        rel: cute.Pointer,
        sinks: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        tiles: Int32,
    ):
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_make_storage_2stage_cls())

        tile_idx = tiles - Int32(1) - Int32(cute.arch.block_idx()[0])
        row0 = Int64(tile_rows[Int64(tile_idx)])
        sf0 = Int32(seq_first[row0])
        # Sequence-aligned pairing: odd tiles within each sequence exit immediately.
        if (((Int32(row0) - sf0) // Int32(_ABI_TOKENS)) & Int32(1)) == Int32(0):
            kv_head = Int32(cute.arch.block_idx()[1])
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx >> Int32(5)
            lane = tidx & Int32(31)

            sk_base = shared_ptr_to_u32(storage.sk.data_ptr())
            sv_base = shared_ptr_to_u32(storage.sv.data_ptr())
            s_rel_base = shared_ptr_to_u32(storage.s_rel.data_ptr())
            token_pos = storage.token_pos.get_tensor(cute.make_layout((48,)))

            live = rows - Int32(row0)
            if live > Int32(_MAX_TOKENS):
                live = Int32(_MAX_TOKENS)
            for t in cutlass.range_constexpr(1, _MAX_TOKENS):
                if Int32(t) < live:
                    if Int32(seq_first[row0 + Int64(t)]) != sf0:
                        live = Int32(t)

            if tidx < Int32(_MAX_TOKENS):
                p_val = Int64(-1)
                if tidx < live:
                    p_val = Int64(positions[row0 + Int64(tidx)])
                token_pos[tidx] = p_val
            cute.arch.sync_threads()

            p_first = Int64(token_pos[0])
            p_min = p_first
            p_max = p_first
            for t in cutlass.range_constexpr(1, _MAX_TOKENS):
                if Int32(t) < live:
                    pt = Int64(token_pos[t])
                    p_min = cutlass.select_(pt < p_min, pt, p_min)
                    p_max = cutlass.select_(pt > p_max, pt, p_max)
            hi = p_max
            lo = p_first - Int64(_WINDOW - 1)
            if lo < Int64(0):
                lo = Int64(0)
            tile_count = Int32((hi - lo + Int64(_TILE_N)) // Int64(_TILE_N))
            if live <= Int32(0):
                tile_count = Int32(0)

            step_first = Int64(sf0)
            seq_start = Int64(positions[step_first])
            ring_base = Int64(ring_slots[row0]) - p_first % Int64(_RING_ROWS)

            gid = lane >> Int32(2)
            tid = lane & Int32(3)
            m0 = warp_id * Int32(16) + gid
            m1 = m0 + Int32(8)
            t_local0 = m0 // Int32(_GROUP)
            head0 = kv_head * Int32(_GROUP) + (m0 - t_local0 * Int32(_GROUP))
            t_local1 = m1 // Int32(_GROUP)
            head1 = kv_head * Int32(_GROUP) + (m1 - t_local1 * Int32(_GROUP))
            live_m = live * Int32(_GROUP)
            valid0 = cutlass.select_(m0 < live_m, Int32(1), Int32(0))
            valid1 = cutlass.select_(m1 < live_m, Int32(1), Int32(0))
            pos0 = Int64(-1)
            pos1 = Int64(-1)
            if valid0 != Int32(0):
                pos0 = Int64(token_pos[t_local0])
            if valid1 != Int32(0):
                pos1 = Int64(token_pos[t_local1])

            b_row = lane & Int32(7)
            b_chunk = lane >> Int32(3)
            v_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            v_row_lo = lane & Int32(7)
            v_chunk = lane >> Int32(4)

            tok0 = tidx >> Int32(4)
            chunk = tidx & Int32(15)
            swiz_col = (chunk ^ (tok0 & Int32(7))) * Int32(16)

            step_ptr = Int64(kv_step.toint())
            ring_ptr = Int64(ring.toint())
            rel_ptr = Int64(rel.toint())
            k_plane_off = Int64(kv_head) * Int64(_HEAD_DIM * 2)
            v_plane_off = Int64(_KV_HEADS * _HEAD_DIM * 2) + k_plane_off
            scale_log2 = Float32(self.scale_log2)
            bias_scale = Float32(self.bias_scale)
            rel_base_m0 = s_rel_base + (m0 * Int32(_REL_STAGE) + Int32(6) - tid * Int32(2)) * Int32(2)
            rel_base_m1 = s_rel_base + (m1 * Int32(_REL_STAGE) + Int32(6) - tid * Int32(2)) * Int32(2)

            r_q0 = cute.make_rmem_tensor((8,), Uint32)
            r_q1 = cute.make_rmem_tensor((8,), Uint32)
            r_q2 = cute.make_rmem_tensor((8,), Uint32)
            r_q3 = cute.make_rmem_tensor((8,), Uint32)
            _load_q_regs(q, row0, t_local0, t_local1, head0, head1, valid0, valid1, tid, r_q0, r_q1, r_q2, r_q3)

            acc_o = cute.make_rmem_tensor((64,), Float32)
            acc_s = cute.make_rmem_tensor((32,), Float32)
            p_u32 = cute.make_rmem_tensor((16,), Uint32)

            acc_o.fill(Float32(0.0))
            m0_max = -Float32.inf
            m1_max = -Float32.inf
            l0_sum = Float32(0.0)
            l1_sum = Float32(0.0)

            # Prologue: group 0 = K[0] + rel[0], group 1 = V[0]; wait_group(1) leaves V[0] in flight.
            if tile_count > Int32(0):
                _cp_async_local_plane_256(
                    lo, hi, seq_start, step_first, ring_base, step_ptr, ring_ptr,
                    k_plane_off, tok0, chunk, swiz_col, sk_base,
                )
                _cp_async_rel_tile(tidx, row0, live, kv_head, lo, rel_ptr, s_rel_base, token_pos)
                cute.arch.cp_async_commit_group()
                _cp_async_local_plane_256(
                    lo, hi, seq_start, step_first, ring_base, step_ptr, ring_ptr,
                    v_plane_off, tok0, chunk, swiz_col, sv_base,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()

            # 2-barrier double-buffered loop: QK[t] overlaps V[t]; PV[t] overlaps K[t+1] + rel[t+1] + V[t+1].
            for tile in cutlass.range(tile_count, unroll=1):
                k0 = lo + Int64(tile) * Int64(_TILE_N)
                stage = tile & Int32(1)
                nxt_stage = stage ^ Int32(1)
                sk_cur = sk_base + stage * Int32(_BF16_PLANE_BYTES)
                sv_cur = sv_base + stage * Int32(_BF16_PLANE_BYTES)
                sk_nxt = sk_base + nxt_stage * Int32(_BF16_PLANE_BYTES)
                sv_nxt = sv_base + nxt_stage * Int32(_BF16_PLANE_BYTES)

                _qk_bf16_swiz(r_q0, r_q1, r_q2, r_q3, acc_s, sk_cur, b_row, b_chunk)
                _apply_bias_and_mask(
                    acc_s, k0, hi, p_min, p_max, valid0, valid1, pos0, pos1,
                    rel_base_m0, rel_base_m1, tid, bias_scale, True,
                )
                m0_max, m1_max, l0_sum, l1_sum = _online_softmax_bf16(
                    acc_s, acc_o, p_u32, m0_max, m1_max, l0_sum, l1_sum, scale_log2
                )
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                next_tile = tile + Int32(1)
                has_next = next_tile < tile_count
                if has_next:
                    k0_next = lo + Int64(next_tile) * Int64(_TILE_N)
                    _cp_async_local_plane_256(
                        k0_next, hi, seq_start, step_first, ring_base, step_ptr, ring_ptr,
                        k_plane_off, tok0, chunk, swiz_col, sk_nxt,
                    )
                    _cp_async_rel_tile(tidx, row0, live, kv_head, k0_next, rel_ptr, s_rel_base, token_pos)
                    cute.arch.cp_async_commit_group()
                    _cp_async_local_plane_256(
                        k0_next, hi, seq_start, step_first, ring_base, step_ptr, ring_ptr,
                        v_plane_off, tok0, chunk, swiz_col, sv_nxt,
                    )
                    cute.arch.cp_async_commit_group()

                _pv_bf16_rs_swiz(p_u32, acc_o, sv_cur, v_row, v_row_lo, v_chunk)

                if has_next:
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()

            _finalize_and_store_output(
                acc_o, m0_max, m1_max, l0_sum, l1_sum,
                valid0, valid1, row0, t_local0, t_local1, head0, head1,
                sinks, out, tid,
            )
