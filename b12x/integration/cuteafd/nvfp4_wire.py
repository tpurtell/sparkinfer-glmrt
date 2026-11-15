"""Exact BF16 -> NVFP4 expert-input wire rows, using the expert quantizers.

Each uint8 row is packed E2M1 [H/2], E4M3 [H/16], F32 amax, then 12 zero
bytes. Calibrated mode requires one identical raw input scale for every routed
expert of the layer (nonuniform expert scales are unsupported). Dynamic mode
uses the checkpoint's per-token fouroversix quantization and stores its amax.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from b12x._lib.intrinsics import st_global_f32, st_global_u32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from ._common import AotProgram, Operand, Scalar, compile_program
from ._fp8_moe_kernels import _ceil, _ld_v2_u32
from ._nvfp4_moe_a4 import _input_global_scale, _quantize_row16
from ._nvfp4_moe_dynamic import _bf16_row_amax, _quantize_row16_dyn, _row_factors


class _WireRows:
    threads = 128

    def __init__(self, hidden: int, mode: str, scale_convention: str):
        self.hidden = hidden
        self.dynamic = mode == "dynamic"
        self.divisor = scale_convention == "divisor"
        self.row_bytes = nvfp4_wire_row_bytes(hidden)

    @cute.jit
    def __call__(self, src: cute.Pointer, dst: cute.Pointer, rows: Int32,
                 input_scale: Float32, stream: cuda.CUstream):
        self.kernel(src, dst, input_scale).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, dst: cute.Pointer, input_scale: Float32):
        row = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        source = Int64(src.toint()) + row * Int64(self.hidden * 2)
        target = Int64(dst.toint()) + row * Int64(self.row_bytes)
        tail = target + Int64(self.hidden // 2 + self.hidden // 16)
        row_amax = Float32(0.0)
        if const_expr(self.dynamic):
            smem = cutlass_utils.SmemAllocator()
            red = smem.allocate_tensor(element_type=Float32, layout=cute.make_layout((self.threads,)), byte_alignment=16)
            row_amax = _bf16_row_amax(source, self.hidden, self.threads, tidx, red)
            encoder, decoder, outer = _row_factors(row_amax)
        else:
            gs = _input_global_scale(input_scale, self.divisor)
        if tidx == Int32(0):
            st_global_f32(tail, row_amax)
        if tidx < Int32(3):
            st_global_u32(tail + Int64(4) + Int64(tidx) * Int64(4), Uint32(0))
        groups = self.hidden // 16
        for it in cutlass.range_constexpr(_ceil(groups, self.threads)):
            grp = Int32(it * self.threads) + tidx
            if grp < Int32(groups):
                values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                for half in cutlass.range_constexpr(2):
                    at = source + (Int64(grp) * Int64(2) + Int64(half)) * Int64(16)
                    lo, hi = _ld_v2_u32(at)
                    lo2, hi2 = _ld_v2_u32(at + Int64(8))
                    words = (lo, hi, lo2, hi2)
                    for q in cutlass.range_constexpr(4):
                        values[8 * half + 2 * q] = _bf16_lo(words[q])
                        values[8 * half + 2 * q + 1] = _bf16_hi(words[q])
                packed_at = target + Int64(grp) * Int64(8)
                scale_at = target + Int64(self.hidden // 2) + Int64(grp)
                if const_expr(self.dynamic):
                    _quantize_row16_dyn(values, encoder, decoder, outer, packed_at, scale_at)
                else:
                    _quantize_row16(values, gs, packed_at, scale_at)


class _DynamicWireRows(_WireRows):
    @cute.jit
    def __call__(self, src: cute.Pointer, dst: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(src, dst, Float32(1.0)).launch(
            grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)


def nvfp4_wire_row_bytes(hidden: int) -> int:
    """16-byte aligned row stride, including amax and reserved zero bytes."""
    if int(hidden) <= 0 or int(hidden) % 256:
        raise ValueError("NVFP4 wire hidden must be positive and a multiple of 256")
    return int(hidden) // 2 + int(hidden) // 16 + 16


def compile_nvfp4_wire_rows_aot(*, hidden: int, max_rows: int, mode: str,
                                scale_convention: str = "multiplier") -> AotProgram:
    """Compile once for planned capacity; live rows and raw scale are runtime scalars.

    Calibrated: ``(src, dst, rows: Int32, input_scale: Float32, stream)``.
    Dynamic: ``(src, dst, rows: Int32, stream)``. No scratch operands.
    A scalar scale cannot represent nonuniform per-expert input scales: callers
    must validate layer uniformity before selecting this wire format.
    """
    hidden, max_rows = int(hidden), int(max_rows)
    stride = nvfp4_wire_row_bytes(hidden)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if mode not in ("calibrated", "dynamic"):
        raise ValueError("mode must be calibrated or dynamic")
    if scale_convention not in ("multiplier", "divisor"):
        raise ValueError("scale_convention must be multiplier or divisor")
    dynamic = mode == "dynamic"
    launch = (_DynamicWireRows if dynamic else _WireRows)(hidden, mode, scale_convention)
    return compile_program(
        launch, name=f"nvfp4_wire_rows_{mode}",
        operands=(Operand("src", torch.bfloat16, f"[rows,{hidden}]"),
                  Operand("dst", torch.uint8, f"[rows,{stride}]", "out")),
        scalars=(Scalar("rows"),) + (() if dynamic else (Scalar("input_scale", "float32"),)),
        key=(hidden, max_rows, mode, scale_convention),
        geometry=dict(hidden=hidden, max_rows=max_rows, mode=mode, scale_convention=scale_convention,
                      row_bytes=stride, uniform_expert_input_scale_required=not dynamic),
        doc=__doc__,
    )
