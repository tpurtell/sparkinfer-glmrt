"""Native AOT identity hyper-connection (iHC) and terminal hc_head programs for Hy4-flash.

Hy4-flash uses 4-stream identity hyper-connections without a 4x4 stream-mixing
matrix or Sinkhorn normalization:

    flat = residual.reshape(T, 4 * H)                                   # BF16 -> F32
    rms  = rsqrt(mean(flat^2) + rms_eps)                                # rms_eps = 1e-5
    mix  = linear(flat, hc_fn) * rms                                    # [T, 8] F32
    pre  = sigmoid(mix[:, 0:4] * hc_scale[0] + hc_base[0:4]) + hc_eps   # hc_eps = 1e-6
    post = hc_magnitude * sigmoid(mix[:, 4:8] * hc_scale[1] + hc_base[4:8]) + hc_eps
    collapsed = sum_l pre[l] * residual[:, l, :]                        # BF16
    y = rms_norm(collapsed, layer_norm_w, rms_eps)                      # BF16

Post update:
    residual_out[:, l, h] = bf16(post[l] * delta[h] + residual[:, l, h])

Terminal hc_head (reuses ``MhcHeadKernel`` at H=3072, rms_eps=1e-5, hc_eps=1e-6):
    mix = linear(flat, hc_head_fn) * rsqrt(mean(flat^2) + rms_eps)      # [T, 4] F32
    pre = sigmoid(mix * hc_head_scale[0] + hc_head_base) + hc_eps
    out = rms_norm(sum_l pre[l] * residual[:, l, :], final_norm_w, rms_eps)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    create_l2_evict_first_policy,
    div_rn_f32,
    ld_global_nc_f32,
    ld_global_nc_v4_u32,
    ld_global_v4_u32,
    pack_f32x2_to_bfloat2,
    rsqrt_approx_ftz_f32,
    st_global_f32,
    st_global_v4_u32,
)
from b12x.norm.mhc._head_cute import MhcHeadKernel
from ._common import Operand, Scalar, compile_program
from .pointwise import _Hy4Bf16RmsNorm3072

__all__ = [
    "DUEL_D_CHAIN",
    "DUEL_D_PROGRAMS",
    "HY4_FLASH_IHC",
    "HY4_IHC",
    "Hy4IhcGeometry",
    "compile_hy4_hc_head_aot",
    "compile_hy4_ihc_norm_post_aot",
    "compile_hy4_ihc_norm_post_pre_aot",
    "compile_hy4_ihc_post_aot",
    "compile_hy4_ihc_post_pre_aot",
    "compile_hy4_ihc_pre_aot",
]

_THREADS = 256
_REDUCTIONS = 9  # sum(r^2) + 8 fn dot products
_FUSED_1TOK_MAX_ROWS = 6
_BM2_MAX_ROWS = 8
_DECODE_MAX_ROWS = 64
_TILE_H = 64  # 3072 / 64 = 48 tiles; 8 elems/lane -> 128-bit v4.u32 coalesced I/O
_BM2_BLOCK_M = 2
_MED_BLOCK_M = 4
_PREFILL_BLOCK_M = 32
_PREFILL_CHUNK_ROWS_6144 = 512  # 512 * 4 * 6144 * 2 = 24 MiB -> residual_out stays L2-resident across partial->finalize
_FINALIZE_THREADS = 384


@dataclass(frozen=True)
class Hy4IhcGeometry:
    hidden: int = 3072
    hc_mult: int = 4
    rms_eps: float = 1e-5
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0

    def __post_init__(self) -> None:
        if self.hc_mult != 4:
            raise ValueError("Hy4 iHC requires hc_mult=4")
        if self.hidden <= 0 or self.hidden % _THREADS != 0:
            raise ValueError(f"Hy4 iHC hidden {self.hidden} must divide by {_THREADS}")
        if not (math.isfinite(self.rms_eps) and self.rms_eps > 0):
            raise ValueError("Hy4 iHC rms_eps must be finite and positive")
        if not (math.isfinite(self.hc_eps) and self.hc_eps >= 0):
            raise ValueError("Hy4 iHC hc_eps must be finite and non-negative")
        if not (math.isfinite(self.hc_magnitude) and self.hc_magnitude > 0):
            raise ValueError("Hy4 iHC hc_magnitude must be finite and positive")


HY4_FLASH_IHC = Hy4IhcGeometry()
HY4_IHC = Hy4IhcGeometry(hidden=6144)


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_l1na_v4_f32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x f32) via non-coherent L1-no-allocate path for immutable weights."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_nc_l1na_v4_f32_l2hint(
    base_ptr: Int64, cache_policy: Uint64, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """128-bit non-coherent weight load with L1 no-allocate and L2 cache policy."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip), Uint64(cache_policy).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.f32 {$0, $1, $2, $3}, [$4], $5;",
        "=f,=f,=f,=f,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_v4_u32_l2hint(
    base_ptr: Int64, cache_policy: Uint64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Coherent 128-bit global load with L2 cache policy, strictly ordered after griddepcontrol.wait."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip), Uint64(cache_policy).ir_value(loc=loc, ip=ip)],
        "ld.global.L2::cache_hint.v4.u32 {$0, $1, $2, $3}, [$4], $5;",
        "=r,=r,=r,=r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_f32_ordered(base_ptr: Int64, *, loc=None, ip=None) -> Float32:
    """Coherent f32 global load with side effects so it never hoists across griddepcontrol.wait."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_lo(word: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $1, 16;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_hi(word: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0xFFFF0000;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


# Keep the original Hy4 (H=6144) kernels separate from the Flash routes.
class _Hy4IhcPreKernel:
    """Single-CTA-per-token iHC pre + fused RMSNorm (optionally preceded by post)."""

    def __init__(self, g: Hy4IhcGeometry, *, fuse_post: bool = False):
        self.hidden = g.hidden
        self.per_thread = g.hidden // _THREADS
        self.rms_eps = float(g.rms_eps)
        self.hc_eps = float(g.hc_eps)
        self.hc_magnitude = float(g.hc_magnitude)
        self.fuse_post = bool(fuse_post)
        self.warps = _THREADS // 32

    def _storage(self):
        warps = self.warps

        class Storage:
            pass

        Storage.__annotations__ = {
            "sums": cute.struct.Align[cute.struct.MemRange[Float32, _REDUCTIONS * warps], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        fn: cute.Pointer,
        scale: cute.Pointer,
        base: cute.Pointer,
        norm: cute.Pointer,
        residual_out: cute.Pointer,
        post_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        h = self.hidden
        m = Int64(rows)
        dt = cute.make_tensor(delta, cute.make_layout((m, h), stride=(h, 1)))
        r = cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        pp = cute.make_tensor(prev_post, cute.make_layout((m, 4), stride=(4, 1)))
        w = cute.make_tensor(fn, cute.make_layout((8, 4 * h), stride=(4 * h, 1)))
        sc = cute.make_tensor(scale, cute.make_layout((2,)))
        bs = cute.make_tensor(base, cute.make_layout((8,)))
        nw = cute.make_tensor(norm, cute.make_layout((h,)))
        ro = cute.make_tensor(residual_out, cute.make_layout((m, 4, h), stride=(4 * h, h, 1)))
        po = cute.make_tensor(post_out, cute.make_layout((m, 4), stride=(4, 1)))
        yo = cute.make_tensor(y_out, cute.make_layout((m, h), stride=(h, 1)))
        self.kernel(dt, r, pp, w, sc, bs, nw, ro, po, yo).launch(
            grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        delta: cute.Tensor,
        r: cute.Tensor,
        prev_post: cute.Tensor,
        w: cute.Tensor,
        sc: cute.Tensor,
        bs: cute.Tensor,
        nw: cute.Tensor,
        r_out: cute.Tensor,
        post_out: cute.Tensor,
        y_out: cute.Tensor,
    ):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sums = storage.sums.get_tensor(
            cute.make_layout((_REDUCTIONS, self.warps), stride=(self.warps, 1))
        )

        pp0, pp1, pp2, pp3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        if const_expr(self.fuse_post):
            pp0 = Float32(prev_post[token, 0])
            pp1 = Float32(prev_post[token, 1])
            pp2 = Float32(prev_post[token, 2])
            pp3 = Float32(prev_post[token, 3])

        acc = cute.make_rmem_tensor(cute.make_layout((_REDUCTIONS,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(_REDUCTIONS):
            acc[i] = Float32(0.0)

        s0 = cute.make_rmem_tensor(cute.make_layout((self.per_thread,), stride=(1,)), Float32)
        s1 = cute.make_rmem_tensor(cute.make_layout((self.per_thread,), stride=(1,)), Float32)
        s2 = cute.make_rmem_tensor(cute.make_layout((self.per_thread,), stride=(1,)), Float32)
        s3 = cute.make_rmem_tensor(cute.make_layout((self.per_thread,), stride=(1,)), Float32)

        hidden = Int64(self.hidden)
        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _THREADS) + Int64(tidx)
            r0 = Float32(r[token, 0, hh])
            r1 = Float32(r[token, 1, hh])
            r2 = Float32(r[token, 2, hh])
            r3 = Float32(r[token, 3, hh])
            if const_expr(self.fuse_post):
                d = Float32(delta[token, hh])
                b0 = (pp0 * d + r0).to(BFloat16)
                b1 = (pp1 * d + r1).to(BFloat16)
                b2 = (pp2 * d + r2).to(BFloat16)
                b3 = (pp3 * d + r3).to(BFloat16)
                r_out[token, 0, hh] = b0
                r_out[token, 1, hh] = b1
                r_out[token, 2, hh] = b2
                r_out[token, 3, hh] = b3
                r0, r1, r2, r3 = Float32(b0), Float32(b1), Float32(b2), Float32(b3)
            s0[j], s1[j], s2[j], s3[j] = r0, r1, r2, r3
            acc[0] = acc[0] + (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3)
            for mix in cutlass.range_constexpr(8):
                acc[1 + mix] = acc[1 + mix] + (
                    Float32(w[mix, hh]) * r0
                    + Float32(w[mix, hidden + hh]) * r1
                    + Float32(w[mix, Int64(2) * hidden + hh]) * r2
                    + Float32(w[mix, Int64(3) * hidden + hh]) * r3
                )

        for i in cutlass.range_constexpr(_REDUCTIONS):
            acc[i] = _warp_sum(acc[i])
        if lane == Int32(0):
            for i in cutlass.range_constexpr(_REDUCTIONS):
                sums[i, warp] = acc[i]
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(_REDUCTIONS):
            total = Float32(0.0)
            for src in cutlass.range_constexpr(self.warps):
                total = total + sums[i, src]
            acc[i] = total
        cute.arch.sync_threads()

        inv_rms = cute.math.rsqrt(acc[0] / Float32(4 * self.hidden) + Float32(self.rms_eps))
        s_pre = Float32(sc[0])
        s_post = Float32(sc[1])
        pre = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for l in cutlass.range_constexpr(4):
            logit_pre = acc[1 + l] * inv_rms * s_pre + Float32(bs[l])
            pre[l] = Float32(1.0) / (Float32(1.0) + cute.math.exp(-logit_pre, fastmath=False)) + Float32(self.hc_eps)
            if tidx == Int32(0):
                logit_post = acc[5 + l] * inv_rms * s_post + Float32(bs[4 + l])
                post_val = Float32(self.hc_magnitude) / (
                    Float32(1.0) + cute.math.exp(-logit_post, fastmath=False)
                ) + Float32(self.hc_eps)
                post_out[token, l] = post_val

        kept = cute.make_rmem_tensor(cute.make_layout((self.per_thread,), stride=(1,)), Float32)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(self.per_thread):
            val = pre[0] * s0[j] + pre[1] * s1[j] + pre[2] * s2[j] + pre[3] * s3[j]
            kept[j] = Float32(val.to(BFloat16))
            square = square + kept[j] * kept[j]
        square = _warp_sum(square)
        if lane == Int32(0):
            sums[0, warp] = square
        cute.arch.sync_threads()
        total = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total = total + sums[0, src]
        out_inv_rms = cute.math.rsqrt(total / Float32(self.hidden) + Float32(self.rms_eps))
        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _THREADS) + Int64(tidx)
            y_out[token, hh] = (kept[j] * out_inv_rms * Float32(nw[hh])).to(BFloat16)


_Hy4IhcPreKernel6144 = _Hy4IhcPreKernel


class _Hy4IhcPreStandalone:
    def __init__(self, g: Hy4IhcGeometry):
        self.inner = _Hy4IhcPreKernel6144(g, fuse_post=False)

    @cute.jit
    def __call__(
        self,
        residual: cute.Pointer,
        fn: cute.Pointer,
        scale: cute.Pointer,
        base: cute.Pointer,
        norm: cute.Pointer,
        post_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.inner(residual, residual, post_out, fn, scale, base, norm, residual, post_out, y_out, rows, stream)


class _Hy4IhcPostKernel:
    def __init__(self, g: Hy4IhcGeometry):
        self.hidden = g.hidden
        self.per_thread = g.hidden // _THREADS

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        h = self.hidden
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(delta, cute.make_layout((m, h), stride=(h, 1))),
            cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1))),
            cute.make_tensor(prev_post, cute.make_layout((m, 4), stride=(4, 1))),
            cute.make_tensor(out, cute.make_layout((m, 4, h), stride=(4 * h, h, 1))),
        ).launch(grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, delta: cute.Tensor, r: cute.Tensor, prev_post: cute.Tensor, out: cute.Tensor):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        p0 = Float32(prev_post[token, 0])
        p1 = Float32(prev_post[token, 1])
        p2 = Float32(prev_post[token, 2])
        p3 = Float32(prev_post[token, 3])
        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _THREADS) + Int64(tidx)
            d = Float32(delta[token, hh])
            out[token, 0, hh] = (p0 * d + Float32(r[token, 0, hh])).to(BFloat16)
            out[token, 1, hh] = (p1 * d + Float32(r[token, 1, hh])).to(BFloat16)
            out[token, 2, hh] = (p2 * d + Float32(r[token, 2, hh])).to(BFloat16)
            out[token, 3, hh] = (p3 * d + Float32(r[token, 3, hh])).to(BFloat16)


_Hy4IhcPreStandalone6144 = _Hy4IhcPreStandalone
_Hy4IhcPostKernel6144 = _Hy4IhcPostKernel


class _Hy4IhcTile64PartialKernel:
    """Unified 48-tile (TILE_H=64) zero-SMEM, zero-barrier 128-bit vectorized partial kernel.

    Each warp owns 1 fn mix across all 4 streams (lanes 0..7=s0, 8..15=s1, 16..23=s2,
    24..31=s3; 8 contiguous elements per lane -> 1x ld/st_global_v4_u32 for r/delta/r_out,
    2x ld_global_nc_l1na_v4_f32 for fn -> 8 fn regs/thread).

    All regimes are purely geometry-determined (zero compile-time SM/L2 reads) and stay
    at <= 40 regs/thread (6 CTAs/SM = 100% SM occupancy, zero spills):
      - row 1:       mix_groups=4 (64t,  192 CTAs), block_m=1, PDL=True, fn evict-first.
      - row 2:       mix_groups=2 (128t, 192 CTAs), block_m=1, PDL=True.
      - rows 3..6:   mix_groups=1 (256t, 144..288 CTAs), block_m=1, PDL=True.
      - rows 7..8:   mix_groups=1 (256t, 192 CTAs), block_m=2, PDL=True.
      - rows 9..64:  mix_groups=1 (256t, 144..768 CTAs), block_m=4, PDL=True.
      - rows > 64:   mix_groups=1 (256t, 48*ceil(rows/32) CTAs), block_m=32, PDL=False,
                     r/delta L2 evict-first on fuse_post so residual_out stays in L2.

    Writes transposed FP32 partials [9, 48] (1,728 B/row <= 6,144 B/row) into y_out.
    """

    def __init__(
        self,
        g: Hy4IhcGeometry,
        *,
        fuse_post: bool,
        mix_groups: int,
        block_m: int,
        use_pdl: bool,
        fn_evict_first: bool = False,
        r_evict_first: bool = False,
    ):
        self.hidden = g.hidden
        self.fuse_post = bool(fuse_post)
        self.mix_groups = int(mix_groups)
        self.block_m = int(block_m)
        self.use_pdl = bool(use_pdl)
        self.fn_evict_first = bool(fn_evict_first)
        self.r_evict_first = bool(r_evict_first)
        self.num_tiles = g.hidden // _TILE_H
        self.warps = 8 // self.mix_groups
        self.threads = self.warps * 32

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        fn: cute.Pointer,
        residual_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(self.block_m == 1):
            grid_z = rows
        else:
            grid_z = (rows + Int32(self.block_m - 1)) // Int32(self.block_m)
        self.kernel(
            Int64(delta.toint()),
            Int64(residual.toint()),
            Int64(prev_post.toint()),
            Int64(fn.toint()),
            Int64(residual_out.toint()),
            Int64(y_out.toint()),
            rows,
        ).launch(
            grid=(self.num_tiles, self.mix_groups, grid_z),
            block=(self.threads, 1, 1),
            use_pdl=self.use_pdl,
            stream=stream,
        )

    @cute.jit
    def _reduce_token(
        self,
        tok: Int32,
        ru0: Uint32,
        ru1: Uint32,
        ru2: Uint32,
        ru3: Uint32,
        du0: Uint32,
        du1: Uint32,
        du2: Uint32,
        du3: Uint32,
        pp: Float32,
        w0: Float32,
        w1: Float32,
        w2: Float32,
        w3: Float32,
        w4: Float32,
        w5: Float32,
        w6: Float32,
        w7: Float32,
        ro_addr: Int64,
        y_addr: Int64,
        fn_col: Int64,
        tile_h: Int32,
        group: Int32,
        warp: Int32,
        lane: Int32,
        mix: Int32,
    ):
        h = self.hidden
        r0, r1 = _bf16_lo(ru0), _bf16_hi(ru0)
        r2, r3 = _bf16_lo(ru1), _bf16_hi(ru1)
        r4, r5 = _bf16_lo(ru2), _bf16_hi(ru2)
        r6, r7 = _bf16_lo(ru3), _bf16_hi(ru3)
        if const_expr(self.fuse_post):
            ou0 = pack_f32x2_to_bfloat2(pp * _bf16_lo(du0) + r0, pp * _bf16_hi(du0) + r1)
            ou1 = pack_f32x2_to_bfloat2(pp * _bf16_lo(du1) + r2, pp * _bf16_hi(du1) + r3)
            ou2 = pack_f32x2_to_bfloat2(pp * _bf16_lo(du2) + r4, pp * _bf16_hi(du2) + r5)
            ou3 = pack_f32x2_to_bfloat2(pp * _bf16_lo(du3) + r6, pp * _bf16_hi(du3) + r7)
            if group == Int32(0):
                if warp == Int32(0):
                    ro_tok = ro_addr + (Int64(tok) * Int64(4 * h) + fn_col) * Int64(2)
                    st_global_v4_u32(ro_tok, ou0, ou1, ou2, ou3)
            r0, r1 = _bf16_lo(ou0), _bf16_hi(ou0)
            r2, r3 = _bf16_lo(ou1), _bf16_hi(ou1)
            r4, r5 = _bf16_lo(ou2), _bf16_hi(ou2)
            r6, r7 = _bf16_lo(ou3), _bf16_hi(ou3)

        dot = ((w0 * r0 + w1 * r1) + (w2 * r2 + w3 * r3)) + (
            (w4 * r4 + w5 * r5) + (w6 * r6 + w7 * r7)
        )
        dot = _warp_sum(dot)
        y_tok = y_addr + Int64(tok) * Int64(h * 2)
        nt = Int64(self.num_tiles)
        if lane == Int32(0):
            st_global_f32(y_tok + (Int64(1 + mix) * nt + Int64(tile_h)) * Int64(4), dot)
        if group == Int32(0):
            if warp == Int32(0):
                sq = ((r0 * r0 + r1 * r1) + (r2 * r2 + r3 * r3)) + (
                    (r4 * r4 + r5 * r5) + (r6 * r6 + r7 * r7)
                )
                sq = _warp_sum(sq)
                if lane == Int32(0):
                    st_global_f32(y_tok + Int64(tile_h) * Int64(4), sq)

    @cute.kernel
    def kernel(
        self,
        delta_addr: Int64,
        r_addr: Int64,
        prev_post_addr: Int64,
        fn_addr: Int64,
        ro_addr: Int64,
        y_addr: Int64,
        rows: Int32,
    ):
        tile_h = Int32(cute.arch.block_idx()[0])
        group = Int32(cute.arch.block_idx()[1])
        row_block = Int32(cute.arch.block_idx()[2])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx >> Int32(5)
        lane = tidx & Int32(31)
        stream_id = lane >> Int32(3)
        sub = lane & Int32(7)

        h = self.hidden
        h_off = Int64(tile_h) * Int64(_TILE_H) + Int64(sub) * Int64(8)
        fn_col = Int64(stream_id) * Int64(h) + h_off
        mix = group * Int32(self.warps) + warp

        # Preload immutable fn weights before griddepcontrol.wait().
        fn_ptr = fn_addr + (Int64(mix) * Int64(4 * h) + fn_col) * Int64(4)
        if const_expr(self.fn_evict_first):
            evict_fn = create_l2_evict_first_policy()
            w0, w1, w2, w3 = _ld_global_nc_l1na_v4_f32_l2hint(fn_ptr, evict_fn)
            w4, w5, w6, w7 = _ld_global_nc_l1na_v4_f32_l2hint(fn_ptr + Int64(16), evict_fn)
        else:
            w0, w1, w2, w3 = _ld_global_nc_l1na_v4_f32(fn_ptr)
            w4, w5, w6, w7 = _ld_global_nc_l1na_v4_f32(fn_ptr + Int64(16))

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        evict_r = Uint64(0)
        if const_expr(self.r_evict_first):
            evict_r = create_l2_evict_first_policy()

        if const_expr(self.block_m == 1):
            t0 = Int64(row_block)
            r0_ptr = r_addr + (t0 * Int64(4 * h) + fn_col) * Int64(2)
            r0_u0, r0_u1, r0_u2, r0_u3 = ld_global_v4_u32(r0_ptr)
            d0_u0, d0_u1, d0_u2, d0_u3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            pp0 = Float32(0.0)
            if const_expr(self.fuse_post):
                d0_u0, d0_u1, d0_u2, d0_u3 = ld_global_v4_u32(
                    delta_addr + (t0 * Int64(h) + h_off) * Int64(2)
                )
                pp0 = _ld_global_f32_ordered(
                    prev_post_addr + (t0 * Int64(4) + Int64(stream_id)) * Int64(4)
                )
            self._reduce_token(
                row_block, r0_u0, r0_u1, r0_u2, r0_u3, d0_u0, d0_u1, d0_u2, d0_u3, pp0,
                w0, w1, w2, w3, w4, w5, w6, w7, ro_addr, y_addr, fn_col, tile_h, group, warp, lane, mix,
            )
        else:
            tok_start = row_block * Int32(self.block_m)
            tok_end = tok_start + Int32(self.block_m)
            if tok_end > rows:
                tok_end = rows
            tok = tok_start
            while tok < tok_end:
                t0 = Int64(tok)
                r0_ptr = r_addr + (t0 * Int64(4 * h) + fn_col) * Int64(2)
                if const_expr(self.r_evict_first):
                    r0_u0, r0_u1, r0_u2, r0_u3 = _ld_global_v4_u32_l2hint(r0_ptr, evict_r)
                else:
                    r0_u0, r0_u1, r0_u2, r0_u3 = ld_global_v4_u32(r0_ptr)
                d0_u0, d0_u1, d0_u2, d0_u3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                pp0 = Float32(0.0)
                if const_expr(self.fuse_post):
                    d0_ptr = delta_addr + (t0 * Int64(h) + h_off) * Int64(2)
                    if const_expr(self.r_evict_first):
                        d0_u0, d0_u1, d0_u2, d0_u3 = _ld_global_v4_u32_l2hint(d0_ptr, evict_r)
                    else:
                        d0_u0, d0_u1, d0_u2, d0_u3 = ld_global_v4_u32(d0_ptr)
                    pp0 = _ld_global_f32_ordered(
                        prev_post_addr + (t0 * Int64(4) + Int64(stream_id)) * Int64(4)
                    )
                self._reduce_token(
                    tok, r0_u0, r0_u1, r0_u2, r0_u3, d0_u0, d0_u1, d0_u2, d0_u3, pp0,
                    w0, w1, w2, w3, w4, w5, w6, w7, ro_addr, y_addr, fn_col, tile_h, group, warp, lane, mix,
                )
                tok = tok + Int32(1)


class _Hy4IhcNormPostPreOneTokPartialKernel:
    """1-token-per-CTA fused RMSNorm(delta) + iHC post + partial reduction for rows 1..6.

    Parallelizes across rows via grid_z = rows (no serial token loop!) while scaling
    mix_groups down as rows grows so delta_raw L2 traffic stays ~0.86-1.73 MB total
    and each CTA executes only ONE sync_threads():
      - row 1:     mix_groups=4 (64t,  192 CTAs, 6 v4.u32 delta loads/thread, fn evict-first)
      - row 2:     mix_groups=2 (128t, 192 CTAs, 3 v4.u32 delta loads/thread)
      - rows 3..6: mix_groups=1 (256t, 144..288 CTAs, 1-2 v4.u32 delta loads/thread)
    """

    def __init__(self, g: Hy4IhcGeometry, *, mix_groups: int, fn_evict_first: bool = False):
        self.hidden = g.hidden
        self.rms_eps = float(g.rms_eps)
        self.num_tiles = g.hidden // _TILE_H  # 48
        self.mix_groups = int(mix_groups)
        self.fn_evict_first = bool(fn_evict_first)
        self.warps = 8 // self.mix_groups
        self.threads = self.warps * 32

    def _storage(self):
        warps = self.warps

        class NormPostPreStorage:
            pass

        NormPostPreStorage.__annotations__ = {
            "d_sums": cute.struct.Align[cute.struct.MemRange[Float32, warps], 16],
        }
        return cute.struct(NormPostPreStorage)

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        delta_norm_w: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        fn: cute.Pointer,
        residual_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            Int64(delta.toint()),
            Int64(delta_norm_w.toint()),
            Int64(residual.toint()),
            Int64(prev_post.toint()),
            Int64(fn.toint()),
            Int64(residual_out.toint()),
            Int64(y_out.toint()),
        ).launch(
            grid=(self.num_tiles, self.mix_groups, rows),
            block=(self.threads, 1, 1),
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        delta_addr: Int64,
        dnw_addr: Int64,
        r_addr: Int64,
        prev_post_addr: Int64,
        fn_addr: Int64,
        ro_addr: Int64,
        y_addr: Int64,
    ):
        tile_h = Int32(cute.arch.block_idx()[0])
        group = Int32(cute.arch.block_idx()[1])
        t64 = Int64(cute.arch.block_idx()[2])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx >> Int32(5)
        lane = tidx & Int32(31)
        stream_id = lane >> Int32(3)
        sub = lane & Int32(7)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        s_d = storage.d_sums.get_tensor(cute.make_layout((self.warps,), stride=(1,)))

        h = self.hidden
        h_off = Int64(tile_h) * Int64(_TILE_H) + Int64(sub) * Int64(8)
        fn_col = Int64(stream_id) * Int64(h) + h_off
        mix = group * Int32(self.warps) + warp

        # Preload immutable weights (fn + norm_ppn) before griddepcontrol.wait().
        fn_ptr = fn_addr + (Int64(mix) * Int64(4 * h) + fn_col) * Int64(4)
        if const_expr(self.fn_evict_first):
            evict_fn = create_l2_evict_first_policy()
            w0, w1, w2, w3 = _ld_global_nc_l1na_v4_f32_l2hint(fn_ptr, evict_fn)
            w4, w5, w6, w7 = _ld_global_nc_l1na_v4_f32_l2hint(fn_ptr + Int64(16), evict_fn)
        else:
            w0, w1, w2, w3 = _ld_global_nc_l1na_v4_f32(fn_ptr)
            w4, w5, w6, w7 = _ld_global_nc_l1na_v4_f32(fn_ptr + Int64(16))
        nw0, nw1, nw2, nw3 = ld_global_nc_v4_u32(dnw_addr + h_off * Int64(2))

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        d_row = delta_addr + t64 * Int64(h * 2)

        # Issue residual + prev_post loads early to overlap with delta RMS reduction.
        ru0, ru1, ru2, ru3 = ld_global_v4_u32(r_addr + (t64 * Int64(4 * h) + fn_col) * Int64(2))
        pp = _ld_global_f32_ordered(prev_post_addr + (t64 * Int64(4) + Int64(stream_id)) * Int64(4))

        # Full 3072-wide delta RMS reduction (384 v4.u32 vectors = 6 KB, 100% coalesced).
        sq_d = Float32(0.0)
        if const_expr(self.threads == 256):
            du_a, du_b, du_c, du_d = ld_global_v4_u32(d_row + Int64(tidx) * Int64(16))
            v0, v1 = _bf16_lo(du_a), _bf16_hi(du_a)
            v2, v3 = _bf16_lo(du_b), _bf16_hi(du_b)
            v4, v5 = _bf16_lo(du_c), _bf16_hi(du_c)
            v6, v7 = _bf16_lo(du_d), _bf16_hi(du_d)
            sq_d = ((v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3)) + (
                (v4 * v4 + v5 * v5) + (v6 * v6 + v7 * v7)
            )
            if warp < Int32(4):
                du_a, du_b, du_c, du_d = ld_global_v4_u32(
                    d_row + (Int64(256) + Int64(tidx)) * Int64(16)
                )
                v0, v1 = _bf16_lo(du_a), _bf16_hi(du_a)
                v2, v3 = _bf16_lo(du_b), _bf16_hi(du_b)
                v4, v5 = _bf16_lo(du_c), _bf16_hi(du_c)
                v6, v7 = _bf16_lo(du_d), _bf16_hi(du_d)
                sq_d = sq_d + (
                    ((v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3))
                    + ((v4 * v4 + v5 * v5) + (v6 * v6 + v7 * v7))
                )
        else:
            iters = (self.hidden // 8) // self.threads  # 6 for 64t, 3 for 128t
            for k in cutlass.range_constexpr(iters):
                du_a, du_b, du_c, du_d = ld_global_v4_u32(
                    d_row + (Int64(k * self.threads) + Int64(tidx)) * Int64(16)
                )
                v0, v1 = _bf16_lo(du_a), _bf16_hi(du_a)
                v2, v3 = _bf16_lo(du_b), _bf16_hi(du_b)
                v4, v5 = _bf16_lo(du_c), _bf16_hi(du_c)
                v6, v7 = _bf16_lo(du_d), _bf16_hi(du_d)
                sq_d = sq_d + (
                    ((v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3))
                    + ((v4 * v4 + v5 * v5) + (v6 * v6 + v7 * v7))
                )
        sq_d = _warp_sum(sq_d)
        if lane == Int32(0):
            s_d[warp] = sq_d

        # Tile delta slice hits L1 (just loaded above) while waiting on the single barrier.
        du0, du1, du2, du3 = ld_global_v4_u32(d_row + h_off * Int64(2))
        cute.arch.sync_threads()

        total_d = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total_d = _add_rn(total_d, s_d[src])
        inv_d = rsqrt_approx_ftz_f32(
            _add_rn(div_rn_f32(total_d, Float32(h)), Float32(self.rms_eps))
        )

        # Exact BF16 rounding of dn = rms_norm(delta, norm_ppn) before residual post update.
        dnu0 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(_bf16_lo(du0), inv_d), _bf16_lo(nw0)),
            _mul_rn(_mul_rn(_bf16_hi(du0), inv_d), _bf16_hi(nw0)),
        )
        dnu1 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(_bf16_lo(du1), inv_d), _bf16_lo(nw1)),
            _mul_rn(_mul_rn(_bf16_hi(du1), inv_d), _bf16_hi(nw1)),
        )
        dnu2 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(_bf16_lo(du2), inv_d), _bf16_lo(nw2)),
            _mul_rn(_mul_rn(_bf16_hi(du2), inv_d), _bf16_hi(nw2)),
        )
        dnu3 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(_bf16_lo(du3), inv_d), _bf16_lo(nw3)),
            _mul_rn(_mul_rn(_bf16_hi(du3), inv_d), _bf16_hi(nw3)),
        )

        ou0 = pack_f32x2_to_bfloat2(
            pp * _bf16_lo(dnu0) + _bf16_lo(ru0), pp * _bf16_hi(dnu0) + _bf16_hi(ru0)
        )
        ou1 = pack_f32x2_to_bfloat2(
            pp * _bf16_lo(dnu1) + _bf16_lo(ru1), pp * _bf16_hi(dnu1) + _bf16_hi(ru1)
        )
        ou2 = pack_f32x2_to_bfloat2(
            pp * _bf16_lo(dnu2) + _bf16_lo(ru2), pp * _bf16_hi(dnu2) + _bf16_hi(ru2)
        )
        ou3 = pack_f32x2_to_bfloat2(
            pp * _bf16_lo(dnu3) + _bf16_lo(ru3), pp * _bf16_hi(dnu3) + _bf16_hi(ru3)
        )
        if group == Int32(0):
            if warp == Int32(0):
                st_global_v4_u32(
                    ro_addr + (t64 * Int64(4 * h) + fn_col) * Int64(2), ou0, ou1, ou2, ou3
                )

        r0, r1 = _bf16_lo(ou0), _bf16_hi(ou0)
        r2, r3 = _bf16_lo(ou1), _bf16_hi(ou1)
        r4, r5 = _bf16_lo(ou2), _bf16_hi(ou2)
        r6, r7 = _bf16_lo(ou3), _bf16_hi(ou3)

        dot = _warp_sum(
            ((w0 * r0 + w1 * r1) + (w2 * r2 + w3 * r3))
            + ((w4 * r4 + w5 * r5) + (w6 * r6 + w7 * r7))
        )
        y_tok = y_addr + t64 * Int64(h * 2)
        nt = Int64(self.num_tiles)
        if lane == Int32(0):
            st_global_f32(y_tok + (Int64(1 + mix) * nt + Int64(tile_h)) * Int64(4), dot)
        if group == Int32(0):
            if warp == Int32(0):
                sq = _warp_sum(
                    ((r0 * r0 + r1 * r1) + (r2 * r2 + r3 * r3))
                    + ((r4 * r4 + r5 * r5) + (r6 * r6 + r7 * r7))
                )
                if lane == Int32(0):
                    st_global_f32(y_tok + Int64(tile_h) * Int64(4), sq)


class _Hy4IhcFinalizeKernel:
    """Single-CTA-per-token vectorized finalize: reduce tiled partials + collapse + RMSNorm.

    Each thread owns 8 contiguous BF16 elements (1 x ld/st_global_v4_u32):
    384 threads for H=3072 (48 tiles) and 768 threads for H=6144 (96 tiles).
    Preloads immutable weights (norm, scale, base) before griddepcontrol_wait(), reduces
    transposed [9, num_tiles] partials from y_out in parallel across warps 0..8,
    computes pre[0..3] and post[0..3] in parallel at 1-exp latency, rounds collapsed to BF16
    before RMSNorm (exact checkpoint numerics), and overwrites y_out after sync_threads().
    """

    def __init__(self, g: Hy4IhcGeometry, *, num_tiles: int, use_pdl: bool, r_evict_first: bool = False):
        self.hidden = g.hidden
        self.rms_eps = float(g.rms_eps)
        self.hc_eps = float(g.hc_eps)
        self.hc_magnitude = float(g.hc_magnitude)
        self.num_tiles = int(num_tiles)
        self.use_pdl = bool(use_pdl)
        self.r_evict_first = bool(r_evict_first)
        self.threads = g.hidden // 8
        self.warps = self.threads // 32

    def _storage(self):
        warps = self.warps

        class FinalizeStorage:
            pass

        FinalizeStorage.__annotations__ = {
            "totals": cute.struct.Align[cute.struct.MemRange[Float32, _REDUCTIONS], 16],
            "norm_sums": cute.struct.Align[cute.struct.MemRange[Float32, warps], 16],
        }
        return cute.struct(FinalizeStorage)

    @cute.jit
    def __call__(
        self,
        src_r: cute.Pointer,
        scale: cute.Pointer,
        base: cute.Pointer,
        norm: cute.Pointer,
        post_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            Int64(src_r.toint()),
            Int64(scale.toint()),
            Int64(base.toint()),
            Int64(norm.toint()),
            Int64(post_out.toint()),
            Int64(y_out.toint()),
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            use_pdl=self.use_pdl,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src_r_addr: Int64,
        scale_addr: Int64,
        base_addr: Int64,
        norm_addr: Int64,
        post_out_addr: Int64,
        y_addr: Int64,
    ):
        tok = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx >> Int32(5)
        lane = tidx & Int32(31)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        s_totals = storage.totals.get_tensor(cute.make_layout((_REDUCTIONS,), stride=(1,)))
        s_norm = storage.norm_sums.get_tensor(cute.make_layout((self.warps,), stride=(1,)))

        h = self.hidden
        h_off = Int64(tidx) * Int64(8)

        # Load immutable weights before griddepcontrol_wait() to hide cold DRAM latency.
        nw0, nw1, nw2, nw3 = ld_global_nc_v4_u32(norm_addr + h_off * Int64(2))
        s_pre = ld_global_nc_f32(scale_addr)
        s_post = ld_global_nc_f32(scale_addr + Int64(4))
        b_lane = Float32(0.0)
        if lane < Int32(8):
            b_lane = ld_global_nc_f32(base_addr + Int64(lane) * Int64(4))

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        y_tok = y_addr + tok * Int64(h * 2)
        if warp < Int32(_REDUCTIONS):
            if const_expr(self.num_tiles == 48):
                p_base = y_tok + (Int64(warp) * Int64(48) + Int64(lane)) * Int64(4)
                v = _ld_global_f32_ordered(p_base)
                if lane < Int32(16):
                    v = v + _ld_global_f32_ordered(p_base + Int64(32 * 4))
                v = _warp_sum(v)
            elif const_expr(self.num_tiles == 96):
                p_base = y_tok + (Int64(warp) * Int64(96) + Int64(lane)) * Int64(4)
                v = (
                    _ld_global_f32_ordered(p_base)
                    + _ld_global_f32_ordered(p_base + Int64(32 * 4))
                    + _ld_global_f32_ordered(p_base + Int64(64 * 4))
                )
                v = _warp_sum(v)
            else:
                v = Float32(0.0)
                chunks = (self.num_tiles + 31) // 32
                for c in cutlass.range_constexpr(chunks):
                    idx = Int32(c * 32) + lane
                    if idx < Int32(self.num_tiles):
                        v = v + _ld_global_f32_ordered(
                            y_tok + (Int64(warp) * Int64(self.num_tiles) + Int64(idx)) * Int64(4)
                        )
                v = _warp_sum(v)
            if lane == Int32(0):
                s_totals[warp] = v

        # Issue all 4 residual stream 128-bit loads in parallel before the first barrier.
        r_base = src_r_addr + (tok * Int64(4 * h) + h_off) * Int64(2)
        if const_expr(self.r_evict_first):
            evict_r = create_l2_evict_first_policy()
            r0_0, r0_1, r0_2, r0_3 = _ld_global_v4_u32_l2hint(r_base, evict_r)
            r1_0, r1_1, r1_2, r1_3 = _ld_global_v4_u32_l2hint(r_base + Int64(h * 2), evict_r)
            r2_0, r2_1, r2_2, r2_3 = _ld_global_v4_u32_l2hint(r_base + Int64(2 * h * 2), evict_r)
            r3_0, r3_1, r3_2, r3_3 = _ld_global_v4_u32_l2hint(r_base + Int64(3 * h * 2), evict_r)
        else:
            r0_0, r0_1, r0_2, r0_3 = ld_global_v4_u32(r_base)
            r1_0, r1_1, r1_2, r1_3 = ld_global_v4_u32(r_base + Int64(h * 2))
            r2_0, r2_1, r2_2, r2_3 = ld_global_v4_u32(r_base + Int64(2 * h * 2))
            r3_0, r3_1, r3_2, r3_3 = ld_global_v4_u32(r_base + Int64(3 * h * 2))

        # Every warp has finished reading y_out partials and s_totals[0..8] is ready.
        cute.arch.sync_threads()

        inv_rms = cute.math.rsqrt(
            s_totals[Int32(0)] / Float32(4 * h) + Float32(self.rms_eps)
        )
        my_pre = Float32(0.0)
        if lane < Int32(4):
            logit_pre = s_totals[Int32(1) + lane] * inv_rms * s_pre + b_lane
            my_pre = (
                Float32(1.0)
                / (Float32(1.0) + cute.math.exp(-logit_pre, fastmath=False))
                + Float32(self.hc_eps)
            )
        if warp == Int32(0):
            if lane >= Int32(4):
                if lane < Int32(8):
                    logit_post = s_totals[Int32(1) + lane] * inv_rms * s_post + b_lane
                    post_val = (
                        Float32(self.hc_magnitude)
                        / (Float32(1.0) + cute.math.exp(-logit_post, fastmath=False))
                        + Float32(self.hc_eps)
                    )
                    st_global_f32(
                        post_out_addr + (tok * Int64(4) + Int64(lane - Int32(4))) * Int64(4),
                        post_val,
                    )

        p0 = Float32(cute.arch.shuffle_sync(my_pre, Int32(0)))
        p1 = Float32(cute.arch.shuffle_sync(my_pre, Int32(1)))
        p2 = Float32(cute.arch.shuffle_sync(my_pre, Int32(2)))
        p3 = Float32(cute.arch.shuffle_sync(my_pre, Int32(3)))

        # Collapse 8 elements with exact BF16 rounding before RMSNorm.
        c0 = pack_f32x2_to_bfloat2(
            p0 * _bf16_lo(r0_0) + p1 * _bf16_lo(r1_0) + p2 * _bf16_lo(r2_0) + p3 * _bf16_lo(r3_0),
            p0 * _bf16_hi(r0_0) + p1 * _bf16_hi(r1_0) + p2 * _bf16_hi(r2_0) + p3 * _bf16_hi(r3_0),
        )
        c1 = pack_f32x2_to_bfloat2(
            p0 * _bf16_lo(r0_1) + p1 * _bf16_lo(r1_1) + p2 * _bf16_lo(r2_1) + p3 * _bf16_lo(r3_1),
            p0 * _bf16_hi(r0_1) + p1 * _bf16_hi(r1_1) + p2 * _bf16_hi(r2_1) + p3 * _bf16_hi(r3_1),
        )
        c2 = pack_f32x2_to_bfloat2(
            p0 * _bf16_lo(r0_2) + p1 * _bf16_lo(r1_2) + p2 * _bf16_lo(r2_2) + p3 * _bf16_lo(r3_2),
            p0 * _bf16_hi(r0_2) + p1 * _bf16_hi(r1_2) + p2 * _bf16_hi(r2_2) + p3 * _bf16_hi(r3_2),
        )
        c3 = pack_f32x2_to_bfloat2(
            p0 * _bf16_lo(r0_3) + p1 * _bf16_lo(r1_3) + p2 * _bf16_lo(r2_3) + p3 * _bf16_lo(r3_3),
            p0 * _bf16_hi(r0_3) + p1 * _bf16_hi(r1_3) + p2 * _bf16_hi(r2_3) + p3 * _bf16_hi(r3_3),
        )
        k0, k1 = _bf16_lo(c0), _bf16_hi(c0)
        k2, k3 = _bf16_lo(c1), _bf16_hi(c1)
        k4, k5 = _bf16_lo(c2), _bf16_hi(c2)
        k6, k7 = _bf16_lo(c3), _bf16_hi(c3)

        sq = ((k0 * k0 + k1 * k1) + (k2 * k2 + k3 * k3)) + (
            (k4 * k4 + k5 * k5) + (k6 * k6 + k7 * k7)
        )
        sq = _warp_sum(sq)
        if lane == Int32(0):
            s_norm[warp] = sq
        cute.arch.sync_threads()

        total = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total = total + s_norm[src]
        out_inv_rms = cute.math.rsqrt(total / Float32(h) + Float32(self.rms_eps))

        y0 = pack_f32x2_to_bfloat2(k0 * out_inv_rms * _bf16_lo(nw0), k1 * out_inv_rms * _bf16_hi(nw0))
        y1 = pack_f32x2_to_bfloat2(k2 * out_inv_rms * _bf16_lo(nw1), k3 * out_inv_rms * _bf16_hi(nw1))
        y2 = pack_f32x2_to_bfloat2(k4 * out_inv_rms * _bf16_lo(nw2), k5 * out_inv_rms * _bf16_hi(nw2))
        y3 = pack_f32x2_to_bfloat2(k6 * out_inv_rms * _bf16_lo(nw3), k7 * out_inv_rms * _bf16_hi(nw3))
        st_global_v4_u32(y_tok + h_off * Int64(2), y0, y1, y2, y3)


class _Hy4IhcPreKernel:
    """Monotonic 48-tile split-K + 384-thread finalize for ihc_pre / ihc_post_pre."""

    def __init__(self, g: Hy4IhcGeometry, *, fuse_post: bool = False):
        self.hidden = g.hidden
        self.fuse_post = bool(fuse_post)
        num_tiles = g.hidden // _TILE_H
        self.r1_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=fuse_post, mix_groups=4, block_m=1, use_pdl=True, fn_evict_first=True
        )
        self.r2_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=fuse_post, mix_groups=2, block_m=1, use_pdl=True
        )
        self.r36_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=fuse_post, mix_groups=1, block_m=1, use_pdl=True
        )
        self.bm2_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=fuse_post, mix_groups=1, block_m=_BM2_BLOCK_M, use_pdl=True
        )
        self.med_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=fuse_post, mix_groups=1, block_m=_MED_BLOCK_M, use_pdl=True
        )
        self.prefill_partial = _Hy4IhcTile64PartialKernel(
            g,
            fuse_post=fuse_post,
            mix_groups=1,
            block_m=_PREFILL_BLOCK_M,
            use_pdl=False,
            r_evict_first=fuse_post,
        )
        self.decode_finalize = _Hy4IhcFinalizeKernel(g, num_tiles=num_tiles, use_pdl=True)
        self.prefill_finalize = _Hy4IhcFinalizeKernel(
            g, num_tiles=num_tiles, use_pdl=False, r_evict_first=(g.hidden == 6144)
        )

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        fn: cute.Pointer,
        scale: cute.Pointer,
        base: cute.Pointer,
        norm: cute.Pointer,
        residual_out: cute.Pointer,
        post_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0):
            src_r = residual_out if const_expr(self.fuse_post) else residual
            if rows <= Int32(_DECODE_MAX_ROWS):
                if rows == Int32(1):
                    self.r1_partial(delta, residual, prev_post, fn, residual_out, y_out, rows, stream)
                elif rows == Int32(2):
                    self.r2_partial(delta, residual, prev_post, fn, residual_out, y_out, rows, stream)
                elif rows <= Int32(_FUSED_1TOK_MAX_ROWS):
                    self.r36_partial(delta, residual, prev_post, fn, residual_out, y_out, rows, stream)
                elif rows <= Int32(_BM2_MAX_ROWS):
                    self.bm2_partial(delta, residual, prev_post, fn, residual_out, y_out, rows, stream)
                else:
                    self.med_partial(delta, residual, prev_post, fn, residual_out, y_out, rows, stream)
                self.decode_finalize(src_r, scale, base, norm, post_out, y_out, rows, stream)
            else:
                if const_expr(self.hidden == 6144):
                    # Chunk wide H=6144 launches at 512 rows (24 MiB/chunk) so residual /
                    # residual_out stays L2-resident between partial and finalize, avoiding
                    # the 2x DRAM read cliff once rows * 48 KiB exceeds L2 capacity.
                    h = Int64(self.hidden)
                    num_chunks = (rows + Int32(_PREFILL_CHUNK_ROWS_6144 - 1)) // Int32(_PREFILL_CHUNK_ROWS_6144)
                    per_chunk = (rows + num_chunks - Int32(1)) // num_chunks
                    chunk_cap = ((per_chunk + Int32(31)) // Int32(32)) * Int32(32)
                    offset = Int32(0)
                    while offset < rows:
                        rem = rows - offset
                        c_rows = chunk_cap if rem > chunk_cap else rem
                        off64 = Int64(offset)
                        d_ptr = delta + off64 * h
                        r_ptr = residual + off64 * Int64(4) * h
                        pp_ptr = prev_post + off64 * Int64(4)
                        ro_ptr = residual_out + off64 * Int64(4) * h
                        po_ptr = post_out + off64 * Int64(4)
                        y_ptr = y_out + off64 * h
                        src_c = ro_ptr if const_expr(self.fuse_post) else r_ptr
                        self.prefill_partial(d_ptr, r_ptr, pp_ptr, fn, ro_ptr, y_ptr, c_rows, stream)
                        self.prefill_finalize(src_c, scale, base, norm, po_ptr, y_ptr, c_rows, stream)
                        offset = offset + c_rows
                else:
                    self.prefill_partial(delta, residual, prev_post, fn, residual_out, y_out, rows, stream)
                    self.prefill_finalize(src_r, scale, base, norm, post_out, y_out, rows, stream)


class _Hy4IhcPreStandalone:
    def __init__(self, g: Hy4IhcGeometry):
        self.inner = _Hy4IhcPreKernel(g, fuse_post=False)

    @cute.jit
    def __call__(
        self,
        residual: cute.Pointer,
        fn: cute.Pointer,
        scale: cute.Pointer,
        base: cute.Pointer,
        norm: cute.Pointer,
        post_out: cute.Pointer,
        y_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.inner(residual, residual, post_out, fn, scale, base, norm, residual, post_out, y_out, rows, stream)


class _Hy4IhcNormPostPreKernel:
    """Fused Steps 3+4: RMSNorm(delta, delta_norm_w) -> iHC post_pre.

    Monotonic row schedule (zero rows 2..8 inversion):
      - row 1:     2-kernel fused 48x4x1 (192 CTAs, 1 tok/CTA, scratch untouched).
      - row 2:     2-kernel fused 48x2x2 (192 CTAs, 1 tok/CTA, scratch untouched).
      - rows 3..6: 2-kernel fused 48x1xR (144..288 CTAs, 1 tok/CTA, scratch untouched).
      - rows 7..8: 3-kernel PDL rms_norm_3072 -> scratch -> 48x1x4 BM=2 partial -> finalize.
      - rows 9..64: 3-kernel PDL rms_norm_3072 -> scratch -> 48x1xceil(R/4) BM=4 partial -> finalize.
      - rows > 64: 3-kernel rms_norm_3072 -> scratch -> 48x1xceil(R/32) BM=32 partial -> finalize.
    """

    def __init__(self, g: Hy4IhcGeometry):
        num_tiles = g.hidden // _TILE_H
        self.r1_partial = _Hy4IhcNormPostPreOneTokPartialKernel(
            g, mix_groups=4, fn_evict_first=True
        )
        self.r2_partial = _Hy4IhcNormPostPreOneTokPartialKernel(g, mix_groups=2)
        self.r36_partial = _Hy4IhcNormPostPreOneTokPartialKernel(g, mix_groups=1)
        self.decode_norm = _Hy4Bf16RmsNorm3072(g.rms_eps, use_pdl=True)
        self.prefill_norm = _Hy4Bf16RmsNorm3072(g.rms_eps, use_pdl=False)
        self.bm2_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=True, mix_groups=1, block_m=_BM2_BLOCK_M, use_pdl=True
        )
        self.med_partial = _Hy4IhcTile64PartialKernel(
            g, fuse_post=True, mix_groups=1, block_m=_MED_BLOCK_M, use_pdl=True
        )
        self.prefill_partial = _Hy4IhcTile64PartialKernel(
            g,
            fuse_post=True,
            mix_groups=1,
            block_m=_PREFILL_BLOCK_M,
            use_pdl=False,
            r_evict_first=True,
        )
        self.decode_finalize = _Hy4IhcFinalizeKernel(g, num_tiles=num_tiles, use_pdl=True)
        self.prefill_finalize = _Hy4IhcFinalizeKernel(
            g, num_tiles=num_tiles, use_pdl=False, r_evict_first=(g.hidden == 6144)
        )

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        delta_norm_w: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        fn: cute.Pointer,
        scale: cute.Pointer,
        base: cute.Pointer,
        norm: cute.Pointer,
        residual_out: cute.Pointer,
        post_out: cute.Pointer,
        y_out: cute.Pointer,
        scratch: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0):
            if rows <= Int32(_DECODE_MAX_ROWS):
                if rows == Int32(1):
                    self.r1_partial(
                        delta, delta_norm_w, residual, prev_post, fn, residual_out, y_out, rows, stream
                    )
                elif rows == Int32(2):
                    self.r2_partial(
                        delta, delta_norm_w, residual, prev_post, fn, residual_out, y_out, rows, stream
                    )
                elif rows <= Int32(_FUSED_1TOK_MAX_ROWS):
                    self.r36_partial(
                        delta, delta_norm_w, residual, prev_post, fn, residual_out, y_out, rows, stream
                    )
                else:
                    self.decode_norm(delta, delta_norm_w, scratch, rows, stream)
                    if rows <= Int32(_BM2_MAX_ROWS):
                        self.bm2_partial(
                            scratch, residual, prev_post, fn, residual_out, y_out, rows, stream
                        )
                    else:
                        self.med_partial(
                            scratch, residual, prev_post, fn, residual_out, y_out, rows, stream
                        )
                self.decode_finalize(residual_out, scale, base, norm, post_out, y_out, rows, stream)
            else:
                self.prefill_norm(delta, delta_norm_w, scratch, rows, stream)
                self.prefill_partial(scratch, residual, prev_post, fn, residual_out, y_out, rows, stream)
                self.prefill_finalize(residual_out, scale, base, norm, post_out, y_out, rows, stream)


class _Hy4IhcNormPostImpl:
    """Single-kernel fused Steps 6+7: RMSNorm(delta, delta_norm_w) + iHC post residual update."""

    def __init__(self, g: Hy4IhcGeometry, *, use_pdl: bool, r_evict_first: bool = False):
        self.hidden = g.hidden
        self.rms_eps = float(g.rms_eps)
        self.use_pdl = bool(use_pdl)
        self.r_evict_first = bool(r_evict_first)
        self.warps = _FINALIZE_THREADS // 32

    def _storage(self):
        warps = self.warps

        class NormPostStorage:
            pass

        NormPostStorage.__annotations__ = {
            "sums": cute.struct.Align[cute.struct.MemRange[Float32, warps], 16],
        }
        return cute.struct(NormPostStorage)

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        delta_norm_w: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            Int64(delta.toint()),
            Int64(delta_norm_w.toint()),
            Int64(residual.toint()),
            Int64(prev_post.toint()),
            Int64(out.toint()),
        ).launch(
            grid=(rows, 1, 1),
            block=(_FINALIZE_THREADS, 1, 1),
            use_pdl=self.use_pdl,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        delta_addr: Int64,
        dnw_addr: Int64,
        r_addr: Int64,
        prev_post_addr: Int64,
        out_addr: Int64,
    ):
        tok = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx >> Int32(5)
        lane = tidx & Int32(31)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sums = storage.sums.get_tensor(cute.make_layout((self.warps,), stride=(1,)))

        h = self.hidden
        h_off = Int64(tidx) * Int64(8)

        # Preload immutable norm weight before griddepcontrol.wait().
        nw0, nw1, nw2, nw3 = ld_global_nc_v4_u32(dnw_addr + h_off * Int64(2))

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        du0, du1, du2, du3 = ld_global_v4_u32(delta_addr + (tok * Int64(h) + h_off) * Int64(2))
        d0, d1 = _bf16_lo(du0), _bf16_hi(du0)
        d2, d3 = _bf16_lo(du1), _bf16_hi(du1)
        d4, d5 = _bf16_lo(du2), _bf16_hi(du2)
        d6, d7 = _bf16_lo(du3), _bf16_hi(du3)

        pp_base = prev_post_addr + tok * Int64(16)
        p0 = _ld_global_f32_ordered(pp_base)
        p1 = _ld_global_f32_ordered(pp_base + Int64(4))
        p2 = _ld_global_f32_ordered(pp_base + Int64(8))
        p3 = _ld_global_f32_ordered(pp_base + Int64(12))

        sq = ((d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3)) + (
            (d4 * d4 + d5 * d5) + (d6 * d6 + d7 * d7)
        )
        sq = _warp_sum(sq)
        if lane == Int32(0):
            sums[warp] = sq

        # Overlap residual stream loads with the 12-warp RMS reduction barrier.
        r_base = r_addr + (tok * Int64(4 * h) + h_off) * Int64(2)
        stride_b = Int64(h * 2)
        if const_expr(self.r_evict_first):
            evict_r = create_l2_evict_first_policy()
            r0_0, r0_1, r0_2, r0_3 = _ld_global_v4_u32_l2hint(r_base, evict_r)
            r1_0, r1_1, r1_2, r1_3 = _ld_global_v4_u32_l2hint(r_base + stride_b, evict_r)
            r2_0, r2_1, r2_2, r2_3 = _ld_global_v4_u32_l2hint(r_base + Int64(2) * stride_b, evict_r)
            r3_0, r3_1, r3_2, r3_3 = _ld_global_v4_u32_l2hint(r_base + Int64(3) * stride_b, evict_r)
        else:
            r0_0, r0_1, r0_2, r0_3 = ld_global_v4_u32(r_base)
            r1_0, r1_1, r1_2, r1_3 = ld_global_v4_u32(r_base + stride_b)
            r2_0, r2_1, r2_2, r2_3 = ld_global_v4_u32(r_base + Int64(2) * stride_b)
            r3_0, r3_1, r3_2, r3_3 = ld_global_v4_u32(r_base + Int64(3) * stride_b)

        cute.arch.sync_threads()

        total = Float32(0.0)
        for src in cutlass.range_constexpr(self.warps):
            total = _add_rn(total, sums[src])
        inv = rsqrt_approx_ftz_f32(
            _add_rn(div_rn_f32(total, Float32(h)), Float32(self.rms_eps))
        )

        # Exact BF16 rounding of delta_norm before residual post update.
        dnu0 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(d0, inv), _bf16_lo(nw0)), _mul_rn(_mul_rn(d1, inv), _bf16_hi(nw0))
        )
        dnu1 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(d2, inv), _bf16_lo(nw1)), _mul_rn(_mul_rn(d3, inv), _bf16_hi(nw1))
        )
        dnu2 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(d4, inv), _bf16_lo(nw2)), _mul_rn(_mul_rn(d5, inv), _bf16_hi(nw2))
        )
        dnu3 = pack_f32x2_to_bfloat2(
            _mul_rn(_mul_rn(d6, inv), _bf16_lo(nw3)), _mul_rn(_mul_rn(d7, inv), _bf16_hi(nw3))
        )
        dn0, dn1 = _bf16_lo(dnu0), _bf16_hi(dnu0)
        dn2, dn3 = _bf16_lo(dnu1), _bf16_hi(dnu1)
        dn4, dn5 = _bf16_lo(dnu2), _bf16_hi(dnu2)
        dn6, dn7 = _bf16_lo(dnu3), _bf16_hi(dnu3)

        o_base = out_addr + (tok * Int64(4 * h) + h_off) * Int64(2)
        st_global_v4_u32(
            o_base,
            pack_f32x2_to_bfloat2(p0 * dn0 + _bf16_lo(r0_0), p0 * dn1 + _bf16_hi(r0_0)),
            pack_f32x2_to_bfloat2(p0 * dn2 + _bf16_lo(r0_1), p0 * dn3 + _bf16_hi(r0_1)),
            pack_f32x2_to_bfloat2(p0 * dn4 + _bf16_lo(r0_2), p0 * dn5 + _bf16_hi(r0_2)),
            pack_f32x2_to_bfloat2(p0 * dn6 + _bf16_lo(r0_3), p0 * dn7 + _bf16_hi(r0_3)),
        )
        st_global_v4_u32(
            o_base + stride_b,
            pack_f32x2_to_bfloat2(p1 * dn0 + _bf16_lo(r1_0), p1 * dn1 + _bf16_hi(r1_0)),
            pack_f32x2_to_bfloat2(p1 * dn2 + _bf16_lo(r1_1), p1 * dn3 + _bf16_hi(r1_1)),
            pack_f32x2_to_bfloat2(p1 * dn4 + _bf16_lo(r1_2), p1 * dn5 + _bf16_hi(r1_2)),
            pack_f32x2_to_bfloat2(p1 * dn6 + _bf16_lo(r1_3), p1 * dn7 + _bf16_hi(r1_3)),
        )
        st_global_v4_u32(
            o_base + Int64(2) * stride_b,
            pack_f32x2_to_bfloat2(p2 * dn0 + _bf16_lo(r2_0), p2 * dn1 + _bf16_hi(r2_0)),
            pack_f32x2_to_bfloat2(p2 * dn2 + _bf16_lo(r2_1), p2 * dn3 + _bf16_hi(r2_1)),
            pack_f32x2_to_bfloat2(p2 * dn4 + _bf16_lo(r2_2), p2 * dn5 + _bf16_hi(r2_2)),
            pack_f32x2_to_bfloat2(p2 * dn6 + _bf16_lo(r2_3), p2 * dn7 + _bf16_hi(r2_3)),
        )
        st_global_v4_u32(
            o_base + Int64(3) * stride_b,
            pack_f32x2_to_bfloat2(p3 * dn0 + _bf16_lo(r3_0), p3 * dn1 + _bf16_hi(r3_0)),
            pack_f32x2_to_bfloat2(p3 * dn2 + _bf16_lo(r3_1), p3 * dn3 + _bf16_hi(r3_1)),
            pack_f32x2_to_bfloat2(p3 * dn4 + _bf16_lo(r3_2), p3 * dn5 + _bf16_hi(r3_2)),
            pack_f32x2_to_bfloat2(p3 * dn6 + _bf16_lo(r3_3), p3 * dn7 + _bf16_hi(r3_3)),
        )


class _Hy4IhcNormPostKernel:
    def __init__(self, g: Hy4IhcGeometry):
        self.decode = _Hy4IhcNormPostImpl(g, use_pdl=True, r_evict_first=False)
        self.prefill = _Hy4IhcNormPostImpl(g, use_pdl=False, r_evict_first=True)

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        delta_norm_w: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0):
            if rows <= Int32(_DECODE_MAX_ROWS):
                self.decode(delta, delta_norm_w, residual, prev_post, out, rows, stream)
            else:
                self.prefill(delta, delta_norm_w, residual, prev_post, out, rows, stream)


class _Hy4IhcPostTiledKernel:
    """Multi-CTA 128-bit vectorized iHC post update: out = bf16(prev_post * delta + r)."""

    def __init__(self, g: Hy4IhcGeometry, *, threads: int, use_pdl: bool):
        self.hidden = g.hidden
        self.threads = int(threads)
        self.block_elems = self.threads * 8
        self.tiles = g.hidden // self.block_elems
        self.use_pdl = bool(use_pdl)

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            Int64(delta.toint()),
            Int64(residual.toint()),
            Int64(prev_post.toint()),
            Int64(out.toint()),
        ).launch(
            grid=(self.tiles, rows, 1),
            block=(self.threads, 1, 1),
            use_pdl=self.use_pdl,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, delta_addr: Int64, r_addr: Int64, prev_post_addr: Int64, out_addr: Int64):
        tile = Int64(cute.arch.block_idx()[0])
        tok = Int64(cute.arch.block_idx()[1])
        tidx = Int64(cute.arch.thread_idx()[0])

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        h = self.hidden
        h_off = tile * Int64(self.block_elems) + tidx * Int64(8)

        pp_base = prev_post_addr + tok * Int64(16)
        p0 = _ld_global_f32_ordered(pp_base)
        p1 = _ld_global_f32_ordered(pp_base + Int64(4))
        p2 = _ld_global_f32_ordered(pp_base + Int64(8))
        p3 = _ld_global_f32_ordered(pp_base + Int64(12))

        du0, du1, du2, du3 = ld_global_v4_u32(delta_addr + (tok * Int64(h) + h_off) * Int64(2))
        d0, d1 = _bf16_lo(du0), _bf16_hi(du0)
        d2, d3 = _bf16_lo(du1), _bf16_hi(du1)
        d4, d5 = _bf16_lo(du2), _bf16_hi(du2)
        d6, d7 = _bf16_lo(du3), _bf16_hi(du3)

        r_base = r_addr + (tok * Int64(4 * h) + h_off) * Int64(2)
        o_base = out_addr + (tok * Int64(4 * h) + h_off) * Int64(2)
        stride_b = Int64(h * 2)

        for s, ps in ((0, p0), (1, p1), (2, p2), (3, p3)):
            off = Int64(s) * stride_b
            ru0, ru1, ru2, ru3 = ld_global_v4_u32(r_base + off)
            st_global_v4_u32(
                o_base + off,
                pack_f32x2_to_bfloat2(ps * d0 + _bf16_lo(ru0), ps * d1 + _bf16_hi(ru0)),
                pack_f32x2_to_bfloat2(ps * d2 + _bf16_lo(ru1), ps * d3 + _bf16_hi(ru1)),
                pack_f32x2_to_bfloat2(ps * d4 + _bf16_lo(ru2), ps * d5 + _bf16_hi(ru2)),
                pack_f32x2_to_bfloat2(ps * d6 + _bf16_lo(ru3), ps * d7 + _bf16_hi(ru3)),
            )


class _Hy4IhcPostKernel:
    def __init__(self, g: Hy4IhcGeometry):
        self.decode = _Hy4IhcPostTiledKernel(g, threads=32, use_pdl=True)
        self.prefill = _Hy4IhcPostTiledKernel(g, threads=128, use_pdl=False)

    @cute.jit
    def __call__(
        self,
        delta: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0):
            if rows <= Int32(_DECODE_MAX_ROWS):
                self.decode(delta, residual, prev_post, out, rows, stream)
            else:
                self.prefill(delta, residual, prev_post, out, rows, stream)


def _supports_tiled_ihc(h: int) -> bool:
    return h in (3072, 6144)


def compile_hy4_ihc_pre_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC):
    h = g.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("fn", torch.float32, f"[8,{4 * h}]"),
        Operand("scale", torch.float32, "[2]", align=4),
        Operand("base", torch.float32, "[8]", align=4),
        Operand("norm", torch.bfloat16, f"[{h}]"),
        Operand("post_out", torch.float32, "[rows,4]", "out"),
        Operand("y_out", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    return compile_program(
        (_Hy4IhcPreStandalone(g) if _supports_tiled_ihc(h) else _Hy4IhcPreStandalone6144(g)),
        name="hy4_ihc_pre",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "pre"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_ihc_post_pre_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC):
    h = g.hidden
    operands = (
        Operand("delta", torch.bfloat16, f"[rows,{h}]"),
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("prev_post", torch.float32, "[rows,4]", align=4),
        Operand("fn", torch.float32, f"[8,{4 * h}]"),
        Operand("scale", torch.float32, "[2]", align=4),
        Operand("base", torch.float32, "[8]", align=4),
        Operand("norm", torch.bfloat16, f"[{h}]"),
        Operand("residual_out", torch.bfloat16, f"[rows,4,{h}]", "out"),
        Operand("post_out", torch.float32, "[rows,4]", "out"),
        Operand("y_out", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    return compile_program(
        (_Hy4IhcPreKernel(g, fuse_post=True) if _supports_tiled_ihc(h) else _Hy4IhcPreKernel6144(g, fuse_post=True)),
        name="hy4_ihc_post_pre",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "post_pre"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_ihc_post_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC):
    h = g.hidden
    operands = (
        Operand("delta", torch.bfloat16, f"[rows,{h}]"),
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("prev_post", torch.float32, "[rows,4]", align=4),
        Operand("out", torch.bfloat16, f"[rows,4,{h}]", "out"),
    )
    return compile_program(
        (_Hy4IhcPostKernel(g) if _supports_tiled_ihc(h) else _Hy4IhcPostKernel6144(g)),
        name="hy4_ihc_post",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "post"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_ihc_norm_post_pre_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC):
    """Fused post-projection RMSNorm(delta) + iHC post_pre."""
    h = g.hidden
    operands = (
        Operand("delta", torch.bfloat16, f"[rows,{h}]"),
        Operand("delta_norm_w", torch.bfloat16, f"[{h}]"),
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("prev_post", torch.float32, "[rows,4]", align=4),
        Operand("fn", torch.float32, f"[8,{4 * h}]"),
        Operand("scale", torch.float32, "[2]", align=4),
        Operand("base", torch.float32, "[8]", align=4),
        Operand("norm", torch.bfloat16, f"[{h}]"),
        Operand("residual_out", torch.bfloat16, f"[rows,4,{h}]", "out"),
        Operand("post_out", torch.float32, "[rows,4]", "out"),
        Operand("y_out", torch.bfloat16, f"[rows,{h}]", "out"),
        Operand("scratch", torch.bfloat16, f"[rows,{h}]", "scratch"),
    )
    return compile_program(
        _Hy4IhcNormPostPreKernel(g),
        name="hy4_ihc_norm_post_pre",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "norm_post_pre"),
        geometry=vars(g),
        scratch={"scratch": lambda rows: int(rows) * h * 2},
        doc=__doc__,
    )


def compile_hy4_ihc_norm_post_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC):
    """Fused FFN output RMSNorm(delta) + iHC post residual update."""
    h = g.hidden
    operands = (
        Operand("delta", torch.bfloat16, f"[rows,{h}]"),
        Operand("delta_norm_w", torch.bfloat16, f"[{h}]"),
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("prev_post", torch.float32, "[rows,4]", align=4),
        Operand("out", torch.bfloat16, f"[rows,4,{h}]", "out"),
    )
    return compile_program(
        _Hy4IhcNormPostKernel(g),
        name="hy4_ihc_norm_post",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "norm_post"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_hc_head_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC, *, store_collapsed: bool = False):
    """Terminal hc_head collapse + final RMSNorm (reuses MhcHeadKernel at H=3072)."""
    h = g.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("fn", torch.float32, f"[4,{4 * h}]"),
        Operand("scale", torch.float32, "[1]", align=4),
        Operand("base", torch.float32, "[4]", align=4),
        Operand("norm", torch.bfloat16, f"[{h}]"),
        Operand("collapsed", torch.bfloat16, f"[rows,{h}]", "out",
                note="written only when compiled with store_collapsed=True"),
        Operand("out", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    kernel = MhcHeadKernel(
        hidden_size=h, rms_eps=g.rms_eps, hc_eps=g.hc_eps,
        norm_eps=g.rms_eps, store_collapsed=store_collapsed,
    )
    return compile_program(
        kernel,
        name="hy4_hc_head",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(h, g.rms_eps, g.hc_eps, store_collapsed),
        geometry={"hidden": h, "rms_eps": g.rms_eps, "hc_eps": g.hc_eps, "store_collapsed": store_collapsed},
        doc=__doc__,
    )


_DUEL_PROG_CACHE: dict[tuple[str, int], object] = {}


def _build_duel_program(name: str):
    dev = torch.cuda.current_device()
    cache_key = (name, dev)
    if name == "ihc_norm_post_pre":
        prog = compile_hy4_ihc_norm_post_pre_aot(HY4_FLASH_IHC)
    elif name == "ihc_norm_post":
        prog = compile_hy4_ihc_norm_post_aot(HY4_FLASH_IHC)
    else:
        raise KeyError(name)
    _DUEL_PROG_CACHE[cache_key] = prog
    return prog


def _get_duel_program(progs, name: str):
    if isinstance(progs, Mapping) and name in progs:
        return progs[name]
    dev = torch.cuda.current_device()
    prog = _DUEL_PROG_CACHE.get((name, dev))
    if prog is None:
        prog = _build_duel_program(name)
    return prog


def _buf(container, name: str):
    return container[name] if isinstance(container, Mapping) else getattr(container, name)


DUEL_D_PROGRAMS = {
    "ihc_norm_post_pre": lambda: _build_duel_program("ihc_norm_post_pre"),
    "ihc_norm_post": lambda: _build_duel_program("ihc_norm_post"),
}


def DUEL_D_CHAIN(progs, I, B, rows: int):
    """Per-round fused Hy4-flash norm + iHC chain.

    Replaces the 5-step per-layer stock sequence (ihc_pre -> norm_attn -> ihc_post_pre
    -> norm_ffn -> ihc_post) with a 3-step fused sequence:
      1. ihc_pre           (42x): residual -> post_pre, y_pre
      2. ihc_norm_post_pre (42x): delta_raw + norm_ppn + residual + prev_post -> r_mid, post_mid, y_mid
                                  (uses dn_attn as scratch for rows > 6; untouched for rows <= 6)
      3. ihc_norm_post     (42x): delta_raw + norm_ffn + r_mid + post_mid -> r_out (skips dn_ffn)
    plus standalone norm_128 (12x): idx_raw + norm_128 -> k128.
    """
    p_pre = progs["ihc_pre"]
    p_npp = _get_duel_program(progs, "ihc_norm_post_pre")
    p_np = _get_duel_program(progs, "ihc_norm_post")
    p_128 = progs["rms_norm_128"]

    residual = _buf(I, "residual")
    delta_raw = _buf(I, "delta_raw")
    idx_raw = _buf(I, "idx_raw")
    fn = _buf(I, "fn")
    fn2 = _buf(I, "fn2")
    scale = _buf(I, "scale")
    scale2 = _buf(I, "scale2")
    base = _buf(I, "base")
    base2 = _buf(I, "base2")
    norm_in = _buf(I, "norm_in")
    norm_post = _buf(I, "norm_post")
    norm_ppn = _buf(I, "norm_ppn")
    norm_ffn = _buf(I, "norm_ffn")
    norm_128 = _buf(I, "norm_128")
    prev_post = _buf(I, "prev_post")

    y_pre = _buf(B, "y_pre")
    post_pre = _buf(B, "post_pre")
    dn_attn = _buf(B, "dn_attn")
    r_mid = _buf(B, "r_mid")
    post_mid = _buf(B, "post_mid")
    y_mid = _buf(B, "y_mid")
    r_out = _buf(B, "r_out")
    k128 = _buf(B, "k128")

    return {
        "ihc_pre": (
            lambda: p_pre.launch(
                residual, fn, scale, base, norm_in, post_pre, y_pre, scalars=(rows,)
            ),
            42,
        ),
        "ihc_norm_post_pre": (
            lambda: p_npp.launch(
                delta_raw, norm_ppn, residual, prev_post, fn2, scale2, base2, norm_post,
                r_mid, post_mid, y_mid, dn_attn, scalars=(rows,),
            ),
            42,
        ),
        "ihc_norm_post": (
            lambda: p_np.launch(
                delta_raw, norm_ffn, r_mid, post_mid, r_out, scalars=(rows,)
            ),
            42,
        ),
        "norm_128": (
            lambda: p_128.launch(
                idx_raw, norm_128, None, None, None, None, k128, scalars=(rows,)
            ),
            12,
        ),
    }
