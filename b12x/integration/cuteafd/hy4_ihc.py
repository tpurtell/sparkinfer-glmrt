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

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr

from b12x.norm.mhc._head_cute import MhcHeadKernel
from ._common import Operand, Scalar, compile_program

__all__ = [
    "HY4_FLASH_IHC",
    "Hy4IhcGeometry",
    "compile_hy4_hc_head_aot",
    "compile_hy4_ihc_post_aot",
    "compile_hy4_ihc_post_pre_aot",
    "compile_hy4_ihc_pre_aot",
]

_THREADS = 256
_REDUCTIONS = 9  # sum(r^2) + 8 fn dot products


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


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


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


def compile_hy4_ihc_pre_aot(g: Hy4IhcGeometry = HY4_FLASH_IHC):
    h = g.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("fn", torch.float32, f"[8,{4 * h}]"),
        Operand("scale", torch.float32, "[2]", align=4),
        Operand("base", torch.float32, "[8]", align=4),
        Operand("norm", torch.bfloat16, f"[{h}]"),
        Operand("post", torch.float32, "[rows,4]", "out"),
        Operand("y", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    return compile_program(
        _Hy4IhcPreStandalone(g),
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
        Operand("post", torch.float32, "[rows,4]", "out"),
        Operand("y", torch.bfloat16, f"[rows,{h}]", "out"),
    )
    return compile_program(
        _Hy4IhcPreKernel(g, fuse_post=True),
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
        _Hy4IhcPostKernel(g),
        name="hy4_ihc_post",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "post"),
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
