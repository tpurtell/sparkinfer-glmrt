"""Shared native trellis ABI v1, independent of model family and checkpoint.

All views are contiguous. ``rotated_inputs`` is an I64 device pointer table;
entry ``rotation_id`` addresses an F16 or BF16 [rows,K] matrix, selected by the
static ``input_dtype``, with K-element row stride.
The caller owns H128/sign rotations, including private id 0, and output H128
and channel scales. ``input_block_scales`` multiplies each K128 FP32 accumulator.
Output and partials stay F32. No allocation, atomics or live-row specialization.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda

from ._common import Operand, Scalar, compile_program

ABI_VERSION = 1
CTAS_PER_SM = 4
POINTERS = ("rotated_inputs", "payload", "input_block_scales", "partial", "output")
SCALARS = ("rows", "rotation_id", "cta_limit")


def geometry(input_features, output_features, *, bits=4, codebook="e4m3",
             sharded_side=1, rotation_count=4, max_rows=4096, groups=1, input_dtype="f16",
             compact_head_shard=False):
    values = (input_features, output_features, rotation_count, max_rows)
    if any(type(v) is not int or not 0 < v <= 2**31-1 for v in values):
        raise ValueError("trellis geometry must contain positive int32 dimensions")
    if input_features % 128 or output_features % 128:
        raise ValueError("trellis input/output dimensions must be multiples of 128")
    if type(bits) is not int or bits not in range(1, 9) or codebook not in ("e4m3", "f16", "mcg", "mul1") or sharded_side not in (1, 2) or type(groups) is not int or groups not in (1, 4) or rotation_count < groups:
        raise ValueError("trellis requires K1--K8, mcg/mul1/e4m3/f16, side 1/2 and groups 1/4")
    if input_dtype not in ("f16", "bf16"):
        raise ValueError("trellis input dtype must be f16 or bf16")
    if compact_head_shard and (input_features != 4608 or bits not in (6, 8)
            or codebook != "mcg" or input_dtype != "bf16" or sharded_side != 1 or groups != 1):
        raise ValueError("compact head shards require output-sharded Inkling BF16 K6/K8 MCG geometry")
    splits = (input_features + 1023) // 1024
    tasks = (max_rows + 15) // 16 * (output_features // 128) * splits * groups
    if max(tasks, max_rows * (output_features // 128) * groups, input_features * output_features // 256) > (2**31-1)//2:
        raise ValueError("trellis task/tile indexing exceeds signed int32")
    partial = max_rows * splits * output_features * 4 * groups
    if partial > 2**63-1:
        raise ValueError("trellis partials exceed signed int64 addressing")
    return dict(abi_version=ABI_VERSION, input_features=input_features,
                output_features=output_features, bits=bits, codebook=codebook,
                sharded_side=sharded_side, rotation_count=rotation_count,
                max_rows=max_rows, split_k_blocks=8, splits=splits, input_dtype=input_dtype,
                ctas_per_sm=CTAS_PER_SM, **({"groups": groups} if groups != 1 else {}),
                **({"compact_head_shard": True} if compact_head_shard else {}))


class _Pointers:
    def __init__(self, g):
        from b12x.gemm.trellis_linear._native import NativeTrellisLinear
        self.core = NativeTrellisLinear(g)

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, payload: cute.Pointer,
                 input_block_scales: cute.Pointer, partial: cute.Pointer,
                 output: cute.Pointer, rows: cutlass.Int32,
                 rotation_id: cutlass.Int32, cta_limit: cutlass.Int32,
                 stream: cuda.CUstream):
        self.core(cute.make_tensor(rotated_inputs, cute.make_layout(1, stride=1)),
                  cute.make_tensor(payload, cute.make_layout(1, stride=1)),
                  cute.make_tensor(input_block_scales, cute.make_layout(1, stride=1)),
                  cute.make_tensor(partial, cute.make_layout(1, stride=1)),
                  cute.make_tensor(output, cute.make_layout(1, stride=1)),
                  rows, rotation_id, cta_limit, stream)


GROUP_NAMES = ("q", "k", "v", "g")


class _GroupedPointers:
    def __init__(self, g):
        from b12x.gemm.trellis_linear._native import NativeTrellisLinear
        self.core = NativeTrellisLinear(g)

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer,
                 q_payload: cute.Pointer, k_payload: cute.Pointer,
                 v_payload: cute.Pointer, g_payload: cute.Pointer,
                 q_input_block_scales: cute.Pointer, k_input_block_scales: cute.Pointer,
                 v_input_block_scales: cute.Pointer, g_input_block_scales: cute.Pointer,
                 partial: cute.Pointer,
                 q_output: cute.Pointer, k_output: cute.Pointer,
                 v_output: cute.Pointer, g_output: cute.Pointer,
                 rows: cutlass.Int32, rotation_id: cutlass.Int32,
                 cta_limit: cutlass.Int32, stream: cuda.CUstream):
        layout = cute.make_layout(1, stride=1)
        self.core.call_grouped(
            cute.make_tensor(rotated_inputs, layout),
            (cute.make_tensor(q_payload, layout), cute.make_tensor(k_payload, layout),
             cute.make_tensor(v_payload, layout), cute.make_tensor(g_payload, layout)),
            (cute.make_tensor(q_input_block_scales, layout), cute.make_tensor(k_input_block_scales, layout),
             cute.make_tensor(v_input_block_scales, layout), cute.make_tensor(g_input_block_scales, layout)),
            cute.make_tensor(partial, layout),
            (cute.make_tensor(q_output, layout), cute.make_tensor(k_output, layout),
             cute.make_tensor(v_output, layout), cute.make_tensor(g_output, layout)),
            rows, rotation_id, cta_limit, stream)


def compile_trellis_linear_aot(*, input_features, output_features, bits=4,
                               codebook="e4m3", sharded_side=1,
                               rotation_count=4, max_rows=4096, groups=1, input_dtype="f16",
                               compact_head_shard=False):
    g = geometry(input_features, output_features, bits=bits, codebook=codebook,
                 sharded_side=sharded_side, rotation_count=rotation_count,
                 max_rows=max_rows, groups=groups, input_dtype=input_dtype,
                 compact_head_shard=compact_head_shard)
    k, n, s = g["input_features"], g["output_features"], g["splits"]
    if groups == 1:
        operands = (
            Operand("rotated_inputs", torch.int64, f"[{rotation_count}]"),
            Operand("payload", torch.uint8, f"[{k*n*bits//8}]"),
            Operand("input_block_scales", torch.float16, f"[{k//128}]"),
            Operand("partial", torch.float32, f"[rows,{s},{n}]", "scratch"),
            Operand("output", torch.float32, f"[rows,{n}]", "out"),
        )
        pointers = _Pointers(g)
    else:
        operands = (
            Operand("rotated_inputs", torch.int64, f"[{rotation_count}]"),
            *(Operand(f"{p}_payload", torch.uint8, f"[{k*n*bits//8}]") for p in GROUP_NAMES),
            *(Operand(f"{p}_input_block_scales", torch.float16, f"[{k//128}]") for p in GROUP_NAMES),
            Operand("partial", torch.float32, f"[rows,{s},{groups*n}]", "scratch"),
            *(Operand(f"{p}_output", torch.float32, f"[rows,{n}]", "out") for p in GROUP_NAMES),
        )
        pointers = _GroupedPointers(g)
    from b12x.gemm.trellis_linear._native import decode_route_thresholds
    routes = list(decode_route_thresholds(g))
    return compile_program(pointers, name="trellis_linear", operands=operands,
                           scalars=tuple(Scalar(name) for name in SCALARS),
                           key=tuple(g.values()), version=4,
                           geometry={**g, "exact_row_cap": pointers.core.exact_row_cap,
                                     "decode_route_thresholds": routes},
                           scratch={"partial": lambda rows: groups*rows*s*n*4}, doc=__doc__)


FUSED_SCALARS = ("rows", "cta_limit")


def fused_geometry(input_features, output_features, *, bits=8, codebook="mul1",
                   sharded_side=1, max_rows=64, input_dtype="bf16", output_dtype="bf16"):
    from b12x.gemm.trellis_linear._fused_k8 import FUSED_DECODE_MAX_ROWS, fused_scratch_bytes

    values = (input_features, output_features, max_rows)
    if any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in values):
        raise ValueError("fused trellis geometry must contain positive int32 dimensions")
    if max_rows > FUSED_DECODE_MAX_ROWS:
        raise ValueError(f"fused trellis decode supports max_rows <= {FUSED_DECODE_MAX_ROWS}")
    if input_features % 128 or output_features % 128:
        raise ValueError("fused trellis input/output dimensions must be multiples of 128")
    if bits != 8 or codebook not in ("mcg", "mul1") or sharded_side not in (1, 2):
        raise ValueError("fused trellis decode requires K8, mcg/mul1, and sharded_side 1/2")
    if input_dtype != "bf16" or output_dtype not in ("bf16", "f32", "raw_f32"):
        raise ValueError("fused trellis decode requires input_dtype='bf16' and output_dtype in ('bf16', 'f32', 'raw_f32')")
    splits = (input_features + 1023) // 1024
    scratch = fused_scratch_bytes(max_rows, input_features, output_features)
    if scratch > 2**63 - 1:
        raise ValueError("fused trellis scratch exceeds signed int64 addressing")
    return dict(
        abi_version=ABI_VERSION,
        fused_transform=True,
        input_features=input_features,
        output_features=output_features,
        bits=bits,
        codebook=codebook,
        sharded_side=sharded_side,
        max_rows=max_rows,
        split_k_blocks=8,
        splits=splits,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        ctas_per_sm=CTAS_PER_SM,
    )


class _FusedPointers:
    def __init__(self, g):
        from b12x.gemm.trellis_linear._fused_k8 import FusedK8TrellisDecode
        self.core = FusedK8TrellisDecode(g)

    @cute.jit
    def __call__(self, x: cute.Pointer, payload: cute.Pointer,
                 suh: cute.Pointer, svh: cute.Pointer,
                 scratch: cute.Pointer, output: cute.Pointer,
                 rows: cutlass.Int32, cta_limit: cutlass.Int32,
                 stream: cuda.CUstream):
        layout = cute.make_layout(1, stride=1)
        self.core(
            cute.make_tensor(x, layout),
            cute.make_tensor(payload, layout),
            cute.make_tensor(suh, layout),
            cute.make_tensor(svh, layout),
            cute.make_tensor(scratch, layout),
            cute.make_tensor(output, layout),
            rows, cta_limit, stream,
        )


class _FusedRawPointers:
    def __init__(self, g):
        from b12x.gemm.trellis_linear._fused_k8 import FusedK8TrellisDecode
        self.core = FusedK8TrellisDecode(g)

    @cute.jit
    def __call__(self, x: cute.Pointer, payload: cute.Pointer,
                 suh: cute.Pointer, scratch: cute.Pointer, output: cute.Pointer,
                 rows: cutlass.Int32, cta_limit: cutlass.Int32,
                 stream: cuda.CUstream):
        layout = cute.make_layout(1, stride=1)
        self.core(
            cute.make_tensor(x, layout),
            cute.make_tensor(payload, layout),
            cute.make_tensor(suh, layout),
            cute.make_tensor(suh, layout),  # constexpr-unused when output_dtype == "raw_f32"
            cute.make_tensor(scratch, layout),
            cute.make_tensor(output, layout),
            rows, cta_limit, stream,
        )


def compile_trellis_linear_fused_aot(*, input_features, output_features, bits=8,
                                     codebook="mul1", sharded_side=1, max_rows=64,
                                     input_dtype="bf16", output_dtype="bf16"):
    """Compile fused H128-ingress + K8 decode + fused split-K/H128-egress program."""
    from b12x.gemm.trellis_linear._fused_k8 import fused_scratch_bytes

    g = fused_geometry(
        input_features, output_features, bits=bits, codebook=codebook,
        sharded_side=sharded_side, max_rows=max_rows,
        input_dtype=input_dtype, output_dtype=output_dtype,
    )
    k, n = g["input_features"], g["output_features"]
    scratch_bytes_cap = fused_scratch_bytes(max_rows, k, n)
    if output_dtype == "raw_f32":
        operands = (
            Operand("x", torch.bfloat16, f"[rows,{k}]"),
            Operand("payload", torch.uint8, f"[{k*n*bits//8}]"),
            Operand("suh", torch.float16, f"[{k}]"),
            Operand("scratch", torch.uint8, f"[{scratch_bytes_cap}]", "scratch"),
            Operand("output", torch.float32, f"[rows,{n}]", "out"),
        )
        pointers = _FusedRawPointers(g)
    else:
        out_dt = torch.bfloat16 if output_dtype == "bf16" else torch.float32
        operands = (
            Operand("x", torch.bfloat16, f"[rows,{k}]"),
            Operand("payload", torch.uint8, f"[{k*n*bits//8}]"),
            Operand("suh", torch.float16, f"[{k}]"),
            Operand("svh", torch.float16, f"[{n}]"),
            Operand("scratch", torch.uint8, f"[{scratch_bytes_cap}]", "scratch"),
            Operand("output", out_dt, f"[rows,{n}]", "out"),
        )
        pointers = _FusedPointers(g)
    return compile_program(
        pointers, name="trellis_linear_fused", operands=operands,
        scalars=tuple(Scalar(name) for name in FUSED_SCALARS),
        key=tuple(g.values()), version=1,
        geometry={**g, "exact_row_cap": pointers.core.exact_row_cap,
                  "decode_route_thresholds": []},
        scratch={"scratch": lambda rows: fused_scratch_bytes(rows, k, n)}, doc=__doc__,
    )
