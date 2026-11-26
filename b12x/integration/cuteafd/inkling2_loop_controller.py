"""Inkling 2 Mini looped cross-pass controller fused boundary programs.

Implements the non-GEMM boundaries of Inkling 2 Mini's core-layer cross-pass
controller (``controllers.{attn,k,v,mlp}``) with zero scratch and live row
counts as runtime scalars:

* ``compile_inkling2_loop_pair_norm_aot``: unweighted F32 RMSNorm of both
  ``dest`` (current-pass input) and ``msg`` (previous-pass cross-pass message),
  concatenated into BF16 ``[slots, rows, 2 * hidden]`` for ``down_weight``.
* ``compile_inkling2_loop_blend_norm_aot``: F32 ``tanh`` on stacked
  ``[forget; gain]`` logits, F32 gated blend
  ``(1 - tanh(forget)) * dest + tanh(gain) * msg``, followed immediately by the
  sublayer's weighted F32 RMSNorm to BF16 ``[slots, rows, hidden]``.

``slots`` is 1 (MLP) or 3 (attention Q/K/V). When ``broadcast_dest=True``
(local core layers, where ``attn_res`` produces one shared input before the
three distinct ``attn``/``k``/``v`` controllers), ``dest`` has shape
``[1, rows, hidden]``; similarly ``broadcast_weight=True`` shares one
``attn_norm.weight`` across all three slots.
"""
from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64
import torch

from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _block_sum, _reduction_storage, _rsqrt


class _LoopPairNorm:
    threads = 256

    def __init__(self, hidden: int, slots: int, broadcast_dest: bool, eps: float):
        self.hidden = hidden
        self.slots = slots
        self.broadcast_dest = broadcast_dest
        self.eps = float(eps)

    @cute.jit
    def __call__(self, dest: cute.Pointer, msg: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows > Int32(0):
            self.kernel(dest, msg, out, rows).launch(
                grid=(rows, self.slots, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, dest: cute.Pointer, msg: cute.Pointer, out: cute.Pointer, rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        slot = Int64(cute.arch.block_idx()[1])
        tid = Int64(cute.arch.thread_idx()[0])
        dest_slot = Int64(0) if cutlass.const_expr(self.broadcast_dest) else slot
        dest_base = (dest_slot * Int64(rows) + row) * Int64(self.hidden)
        msg_base = (slot * Int64(rows) + row) * Int64(self.hidden)
        out_base = (slot * Int64(rows) + row) * Int64(2 * self.hidden)
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, self.threads // 32))
        sums = storage.sums.get_tensor(
            cute.make_layout((2, self.threads // 32), stride=(self.threads // 32, 1)))
        d_vals = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        m_vals = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        d_sq = Float32(0)
        m_sq = Float32(0)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            dv = Float32(dest[dest_base + col])
            mv = Float32(msg[msg_base + col])
            d_vals[j] = dv
            m_vals[j] = mv
            d_sq = d_sq + dv * dv
            m_sq = m_sq + mv * mv
        d_tot = _block_sum(d_sq, sums, Int32(0), self.threads // 32)
        m_tot = _block_sum(m_sq, sums, Int32(1), self.threads // 32)
        d_inv = _rsqrt(div_rn_f32(d_tot, Float32(self.hidden)) + Float32(self.eps))
        m_inv = _rsqrt(div_rn_f32(m_tot, Float32(self.hidden)) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            out[out_base + col] = BFloat16(d_vals[j] * d_inv)
            out[out_base + Int64(self.hidden) + col] = BFloat16(m_vals[j] * m_inv)


class _LoopBlendNorm:
    threads = 256

    def __init__(self, hidden: int, slots: int, broadcast_dest: bool,
                 broadcast_weight: bool, eps: float):
        self.hidden = hidden
        self.slots = slots
        self.broadcast_dest = broadcast_dest
        self.broadcast_weight = broadcast_weight
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
        dest_slot = Int64(0) if cutlass.const_expr(self.broadcast_dest) else slot
        w_slot = Int64(0) if cutlass.const_expr(self.broadcast_weight) else slot
        dest_base = (dest_slot * Int64(rows) + row) * Int64(self.hidden)
        msg_base = (slot * Int64(rows) + row) * Int64(self.hidden)
        fg_base = (slot * Int64(rows) + row) * Int64(2 * self.hidden)
        w_base = w_slot * Int64(self.hidden)
        storage = utils.SmemAllocator().allocate(_reduction_storage(1, self.threads // 32))
        sums = storage.sums.get_tensor(
            cute.make_layout((1, self.threads // 32), stride=(self.threads // 32, 1)))
        values = cute.make_rmem_tensor((self.hidden // self.threads,), Float32)
        square = Float32(0)
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            forget = cute.math.tanh(Float32(forget_gain[fg_base + col]), fastmath=False)
            gain = cute.math.tanh(Float32(forget_gain[fg_base + Int64(self.hidden) + col]), fastmath=False)
            dv = Float32(dest[dest_base + col])
            mv = Float32(msg[msg_base + col])
            blended = (Float32(1.0) - forget) * dv + gain * mv
            values[j] = blended
            square = square + blended * blended
        total = _block_sum(square, sums, Int32(0), self.threads // 32)
        inv = _rsqrt(div_rn_f32(total, Float32(self.hidden)) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.hidden // self.threads):
            col = tid + Int64(j * self.threads)
            out[msg_base + col] = BFloat16(values[j] * inv * Float32(weight[w_base + col]))


def _validate_common(hidden: int, slots: int, eps: float, max_rows: int) -> None:
    if (type(hidden) is not int or hidden <= 0 or hidden % 256
            or type(slots) is not int or slots not in (1, 3)
            or not math.isfinite(eps) or eps <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1):
        raise ValueError("invalid Inkling loop controller geometry/capacity")


def compile_inkling2_loop_pair_norm_aot(
    *, hidden: int = 2304, slots: int = 1, broadcast_dest: bool = False,
    eps: float = 1e-6, max_rows: int = 4096,
):
    """Unweighted F32 RMSNorm of (dest, msg) concatenated into BF16 [slots, rows, 2*hidden]."""
    _validate_common(hidden, slots, eps, max_rows)
    if broadcast_dest and slots == 1:
        raise ValueError("broadcast_dest requires slots > 1")
    dest_slots = 1 if broadcast_dest else slots
    return compile_program(
        _LoopPairNorm(hidden, slots, broadcast_dest, eps),
        name="inkling2_loop_pair_norm",
        operands=(
            Operand("dest", torch.float32, f"[{dest_slots},rows,{hidden}]"),
            Operand("msg", torch.float32, f"[{slots},rows,{hidden}]"),
            Operand("out", torch.bfloat16, f"[{slots},rows,{2 * hidden}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(hidden, slots, broadcast_dest, eps, max_rows),
        geometry=dict(hidden=hidden, slots=slots, broadcast_dest=broadcast_dest,
                      eps=eps, max_rows=max_rows),
        doc=__doc__,
    )


def compile_inkling2_loop_blend_norm_aot(
    *, hidden: int = 2304, slots: int = 1, broadcast_dest: bool = False,
    broadcast_weight: bool = False, eps: float = 1e-6, max_rows: int = 4096,
):
    """Fused F32 tanh-gated blend (1-tanh(f))*dest + tanh(g)*msg + weighted RMSNorm -> BF16."""
    _validate_common(hidden, slots, eps, max_rows)
    if (broadcast_dest or broadcast_weight) and slots == 1:
        raise ValueError("broadcast flags require slots > 1")
    dest_slots = 1 if broadcast_dest else slots
    w_slots = 1 if broadcast_weight else slots
    return compile_program(
        _LoopBlendNorm(hidden, slots, broadcast_dest, broadcast_weight, eps),
        name="inkling2_loop_blend_norm",
        operands=(
            Operand("dest", torch.float32, f"[{dest_slots},rows,{hidden}]"),
            Operand("msg", torch.float32, f"[{slots},rows,{hidden}]"),
            Operand("forget_gain", torch.float32, f"[{slots},rows,{2 * hidden}]",
                    note="stacked [forget; gain] F32 pre-tanh logits per row"),
            Operand("weight", torch.bfloat16, f"[{w_slots},{hidden}]"),
            Operand("out", torch.bfloat16, f"[{slots},rows,{hidden}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(hidden, slots, broadcast_dest, broadcast_weight, eps, max_rows),
        geometry=dict(hidden=hidden, slots=slots, broadcast_dest=broadcast_dest,
                      broadcast_weight=broadcast_weight, eps=eps, max_rows=max_rows),
        doc=__doc__,
    )


class _ControllerDownSplitK:
    """Fixed eight-way K partition, then increasing-part FP32 fold.

    Both row routes run precisely the same K16 MMA segments and split fold.
    The partial plane is caller-owned and uses live-row stride.
    """
    n = 64
    k = 4608
    threads = 128

    def __init__(self, *, splits=8, decode_tile=(1, 1, 16, 8, 4, 64), decode_limit=16):
        from b12x.gemm.bf16_gemv._prefill import Bf16PrefillKernel
        self.splits = splits
        self.decode_limit = decode_limit
        wm, wn, tm, tn, stages, tk = decode_tile
        self.decode = Bf16PrefillKernel(self.n, self.k, warp_layout=(wm, wn), tile_m=tm,
                                        tile_n=tn, tile_k=tk, num_stages=stages,
                                        compensated=False, split_k=splits)
        self.prefill = Bf16PrefillKernel(self.n, self.k, compute_warps=4, tile_n=64,
                                         num_stages=4, compensated=False, split_k=splits)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
                 partial: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        xt = cute.make_tensor(x, cute.make_layout((rows, self.k), stride=(self.k, 1)))
        wt = cute.make_tensor(w, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
        pt = cute.make_tensor(partial, cute.make_layout((rows * self.splits, self.n), stride=(self.n, 1)))
        if rows <= Int32(self.decode_limit):
            self.decode(xt, wt, pt, rows, stream)
        else:
            self.prefill(xt, wt, pt, rows, stream)
        self.fold(partial, out, rows).launch(
            grid=((rows * self.n + self.threads - 1) // self.threads, 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def fold(self, partial: cute.Pointer, out: cute.Pointer, rows: Int32):
        i = Int64(cute.arch.block_idx()[0]) * self.threads + Int64(cute.arch.thread_idx()[0])
        stride = Int64(rows) * self.n
        if i < stride:
            acc = Float32(partial[i])
            for part in cutlass.range_constexpr(1, self.splits):
                acc = acc + Float32(partial[i + Int64(part) * stride])
            out[i] = BFloat16(acc)


def compile_inkling2_mini_controller_down_aot(*, max_rows: int = 4096):
    """Target-only down projection. BF16 rank, fixed K partition at every width."""
    if not 1 <= max_rows <= 4096:
        raise ValueError("Mini controller row capacity must be in 1..4096")
    launch = _ControllerDownSplitK()
    return compile_program(
        launch, name="inkling2_mini_controller_down", operands=(
            Operand("x", torch.bfloat16, "[rows,4608]"),
            Operand("w", torch.bfloat16, "[64,4608]"),
            Operand("out", torch.bfloat16, "[rows,64]", "out"),
            Operand("partial", torch.float32, "[8,rows,64]", "out"),
        ), scalars=(Scalar("rows"),), key=(max_rows, 8, 16),
        geometry=dict(width=4608, outputs=64, output_dtype="torch.bfloat16",
                      reduction="split8_ordered_k16_fold", splits=8),
        scratch={"partial": lambda rows: 8 * rows * 64 * 4},
    )


def compile_inkling2_mini_controller_up_aot(*, max_rows: int = 4096):
    """Target-only forget/gain projection, preserving the original K16 order."""
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    if not 1 <= max_rows <= 4096:
        raise ValueError("Mini controller row capacity must be in 1..4096")
    tiles = ((16, 1, 4, 16, 64, 6),)
    launch = TmaBf16Projection(4608, 64, out_dtype=Float32, tiles_by_rows=tiles)
    return compile_program(
        launch, name="inkling2_mini_controller_up", operands=(
            Operand("x", torch.bfloat16, "[rows,64]"),
            Operand("w", torch.bfloat16, "[4608,64]"),
            Operand("out", torch.float32, "[rows,4608]", "out"),
        ), scalars=(Scalar("rows"),), key=(max_rows, launch.key()),
        geometry=dict(width=64, outputs=4608, output_dtype="torch.float32",
                      reduction="ordered_k16", tiles_by_rows=tiles),
    )
