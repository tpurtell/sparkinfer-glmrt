"""Native AOT GPU Engram kernels for Hy4-flash (layers 1, 11, 21, 31).

Matches ``python/reference/families/hy4_flash/model.py::engram`` exactly,
including ``nn.RMSNorm``'s default ``eps = finfo(float32).eps`` (1.1920929e-7),
NOT ``text_config.rms_norm_eps`` (1e-5):

1. ``compile_hy4_engram_head_norm_aot``:
   Per-head RMSNorm across 16 Engram heads of width 96 into ``[rows, 1536]``.

2. ``compile_hy4_engram_gate_add_aot``:
   After caller projects ``value_proj(emb) + bias`` (`[rows, 3072]`) and
   4x ``key_projs[j](emb) + bias`` (`[rows, 4, 3072]`):
     - ``v_norm = unified_pre_gamma * rms_norm_unweighted(value, f32_eps)``
     - For each stream ``j in 0..4``:
         ``key_j   = rms_norm(keys[:, j], norm_key[j], f32_eps)``
         ``query_j = rms_norm(residual[:, j], norm_query[j], f32_eps)``
         ``g       = sum_h(key_j * query_j) * H^(-0.5)``
         ``gate_j  = sigmoid(sqrt(max(|g|, 1e-6)) * sign(g))``
         If ``masked[row] == 0``:
           ``residual[:, j, h] += (gate_j + unified_per_hc_scales[j, h]) * v_norm[h]``
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

from ._common import Operand, Scalar, compile_program

__all__ = [
    "HY4_FLASH_ENGRAM",
    "Hy4EngramGeometry",
    "compile_hy4_engram_gate_add_aot",
    "compile_hy4_engram_head_norm_aot",
]

_F32_EPS = float(torch.finfo(torch.float32).eps)
_GATE_THREADS = 256


@dataclass(frozen=True)
class Hy4EngramGeometry:
    hidden: int = 3072
    hc_mult: int = 4
    heads: int = 16
    head_dim: int = 96
    eps: float = _F32_EPS

    def __post_init__(self) -> None:
        if self.hc_mult != 4:
            raise ValueError("Hy4 Engram requires hc_mult=4")
        if self.hidden <= 0 or self.hidden % _GATE_THREADS != 0:
            raise ValueError(f"Hy4 Engram hidden {self.hidden} must divide by {_GATE_THREADS}")
        if self.heads <= 0 or self.head_dim != 96:
            raise ValueError("Hy4 Engram requires head_dim=96")


HY4_FLASH_ENGRAM = Hy4EngramGeometry()


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


class _Hy4EngramHeadNormKernel:
    """One 32-thread warp per (row, head) normalizing 96 elements (3 per lane)."""

    def __init__(self, g: Hy4EngramGeometry):
        self.heads = g.heads
        self.head_dim = g.head_dim
        self.eps = float(g.eps)

    @cute.jit
    def __call__(
        self,
        raw: cute.Pointer,
        weights: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        m = Int64(rows)
        h, d = self.heads, self.head_dim
        self.kernel(
            cute.make_tensor(raw, cute.make_layout((m, h, d), stride=(h * d, d, 1))),
            cute.make_tensor(weights, cute.make_layout((h, d), stride=(d, 1))),
            cute.make_tensor(out, cute.make_layout((m, h, d), stride=(h * d, d, 1))),
        ).launch(grid=(rows, h, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, raw: cute.Tensor, weights: cute.Tensor, out: cute.Tensor):
        row = Int64(cute.arch.block_idx()[0])
        head = Int64(cute.arch.block_idx()[1])
        lane = Int64(cute.arch.thread_idx()[0])
        x0 = Float32(raw[row, head, lane])
        x1 = Float32(raw[row, head, lane + Int64(32)])
        x2 = Float32(raw[row, head, lane + Int64(64)])
        ss = _warp_sum(x0 * x0 + x1 * x1 + x2 * x2)
        inv = cute.math.rsqrt(ss / Float32(96.0) + Float32(self.eps))
        out[row, head, lane] = (x0 * inv * Float32(weights[head, lane])).to(BFloat16)
        out[row, head, lane + Int64(32)] = (x1 * inv * Float32(weights[head, lane + Int64(32)])).to(BFloat16)
        out[row, head, lane + Int64(64)] = (x2 * inv * Float32(weights[head, lane + Int64(64)])).to(BFloat16)


class _Hy4EngramGateAddKernel:
    """Fused 4-stream Engram gate + value norm + residual add, one CTA per token."""

    def __init__(self, g: Hy4EngramGeometry):
        self.hidden = g.hidden
        self.per_thread = g.hidden // _GATE_THREADS
        self.eps = float(g.eps)
        self.inv_sqrt_h = float(g.hidden ** -0.5)
        self.warps = _GATE_THREADS // 32

    def _storage(self):
        warps = self.warps

        class Storage:
            pass

        # 9 reductions in pass 1: sum(v^2), 4x sum(k_j^2), 4x sum(q_j^2)
        Storage.__annotations__ = {
            "sums": cute.struct.Align[cute.struct.MemRange[Float32, 9 * warps], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        residual: cute.Pointer,
        keys: cute.Pointer,
        value: cute.Pointer,
        norm_key: cute.Pointer,
        norm_query: cute.Pointer,
        pre_gamma: cute.Pointer,
        hc_scales: cute.Pointer,
        masked: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        h = self.hidden
        m = Int64(rows)
        self.kernel(
            cute.make_tensor(residual, cute.make_layout((m, 4, h), stride=(4 * h, h, 1))),
            cute.make_tensor(keys, cute.make_layout((m, 4, h), stride=(4 * h, h, 1))),
            cute.make_tensor(value, cute.make_layout((m, h), stride=(h, 1))),
            cute.make_tensor(norm_key, cute.make_layout((4, h), stride=(h, 1))),
            cute.make_tensor(norm_query, cute.make_layout((4, h), stride=(h, 1))),
            cute.make_tensor(pre_gamma, cute.make_layout((h,))),
            cute.make_tensor(hc_scales, cute.make_layout((4, h), stride=(h, 1))),
            cute.make_tensor(masked, cute.make_layout((m,))),
        ).launch(grid=(rows, 1, 1), block=(_GATE_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        residual: cute.Tensor,
        keys: cute.Tensor,
        value: cute.Tensor,
        norm_key: cute.Tensor,
        norm_query: cute.Tensor,
        pre_gamma: cute.Tensor,
        hc_scales: cute.Tensor,
        masked: cute.Tensor,
    ):
        row = Int64(cute.arch.block_idx()[0])
        is_masked = masked[row] != Uint8(0)
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sums = storage.sums.get_tensor(cute.make_layout((9, self.warps), stride=(self.warps, 1)))

        acc = cute.make_rmem_tensor(cute.make_layout((9,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(9):
            acc[i] = Float32(0.0)

        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _GATE_THREADS) + Int64(tidx)
            v = Float32(value[row, hh])
            acc[0] = acc[0] + v * v
            for s in cutlass.range_constexpr(4):
                kv = Float32(keys[row, s, hh])
                qv = Float32(residual[row, s, hh])
                acc[1 + s] = acc[1 + s] + kv * kv
                acc[5 + s] = acc[5 + s] + qv * qv

        for i in cutlass.range_constexpr(9):
            acc[i] = _warp_sum(acc[i])
        if lane == Int32(0):
            for i in cutlass.range_constexpr(9):
                sums[i, warp] = acc[i]
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(9):
            total = Float32(0.0)
            for src in cutlass.range_constexpr(self.warps):
                total = total + sums[i, src]
            acc[i] = cute.math.rsqrt(total / Float32(self.hidden) + Float32(self.eps))
        cute.arch.sync_threads()

        inv_v = acc[0]
        dot = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for s in cutlass.range_constexpr(4):
            dot[s] = Float32(0.0)

        for j in cutlass.range_constexpr(self.per_thread):
            hh = Int64(j * _GATE_THREADS) + Int64(tidx)
            for s in cutlass.range_constexpr(4):
                kn = Float32(keys[row, s, hh]) * acc[1 + s] * Float32(norm_key[s, hh])
                qn = Float32(residual[row, s, hh]) * acc[5 + s] * Float32(norm_query[s, hh])
                dot[s] = dot[s] + kn * qn

        for s in cutlass.range_constexpr(4):
            dot[s] = _warp_sum(dot[s])
        if lane == Int32(0):
            for s in cutlass.range_constexpr(4):
                sums[s, warp] = dot[s]
        cute.arch.sync_threads()
        gate = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
        for s in cutlass.range_constexpr(4):
            g = Float32(0.0)
            for src in cutlass.range_constexpr(self.warps):
                g = g + sums[s, src]
            g = g * Float32(self.inv_sqrt_h)
            mag = g if g >= Float32(0.0) else -g
            clamped = mag if mag > Float32(1e-6) else Float32(1e-6)
            signed_sqrt = cute.math.sqrt(clamped, fastmath=False)
            if g < Float32(0.0):
                signed_sqrt = -signed_sqrt
            elif g == Float32(0.0):
                signed_sqrt = Float32(0.0)
            gate[s] = Float32(1.0) / (Float32(1.0) + cute.math.exp(-signed_sqrt, fastmath=False))

        if not is_masked:
            for j in cutlass.range_constexpr(self.per_thread):
                hh = Int64(j * _GATE_THREADS) + Int64(tidx)
                vn = Float32(pre_gamma[hh]) * (Float32(value[row, hh]) * inv_v)
                for s in cutlass.range_constexpr(4):
                    delta = (gate[s] + Float32(hc_scales[s, hh])) * vn
                    residual[row, s, hh] = (Float32(residual[row, s, hh]) + delta).to(BFloat16)


def compile_hy4_engram_head_norm_aot(g: Hy4EngramGeometry = HY4_FLASH_ENGRAM):
    operands = (
        Operand("raw", torch.bfloat16, f"[rows,{g.heads},{g.head_dim}]"),
        Operand("weights", torch.bfloat16, f"[{g.heads},{g.head_dim}]"),
        Operand("out", torch.bfloat16, f"[rows,{g.heads * g.head_dim}]", "out"),
    )
    return compile_program(
        _Hy4EngramHeadNormKernel(g),
        name="hy4_engram_head_norm",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "head_norm"),
        geometry=vars(g),
        doc=__doc__,
    )


def compile_hy4_engram_gate_add_aot(g: Hy4EngramGeometry = HY4_FLASH_ENGRAM):
    h = g.hidden
    operands = (
        Operand("residual", torch.bfloat16, f"[rows,4,{h}]", "inout"),
        Operand("keys", torch.bfloat16, f"[rows,4,{h}]"),
        Operand("value", torch.bfloat16, f"[rows,{h}]"),
        Operand("norm_key", torch.bfloat16, f"[4,{h}]"),
        Operand("norm_query", torch.bfloat16, f"[4,{h}]"),
        Operand("pre_gamma", torch.bfloat16, f"[{h}]"),
        Operand("hc_scales", torch.bfloat16, f"[4,{h}]"),
        Operand("masked", torch.uint8, "[rows]", align=1),
    )
    return compile_program(
        _Hy4EngramGateAddKernel(g),
        name="hy4_engram_gate_add",
        operands=operands,
        scalars=(Scalar("rows"),),
        key=(g, "gate_add"),
        geometry=vars(g),
        doc=__doc__,
    )
