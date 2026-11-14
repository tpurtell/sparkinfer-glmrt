"""Native dense A8 prefill with caller-owned reusable MXFP8 records.

Quantize each distinct (input, rotation) once, then reuse its records for every
projection sharing that input. The GEMM never reads the original activations.
Records have ceil(rows/16) * (K/32) * 528 bytes; capacity is planned, rows is live.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32
import torch
from ._common import Operand, Scalar, compile_program


def geometry(input_features, output_features=128, *, encoding="trellis",
             sharded_side=1, max_rows=4096, rotation_count=4, input_dtype="f16"):
    values = (input_features, output_features, max_rows, rotation_count)
    if any(type(v) is not int or not 0 < v < 2**31 for v in values):
        raise ValueError("MX8 dense geometry requires positive int32 dimensions")
    if input_features % 128 or output_features % 32:
        raise ValueError("MX8 dense requires K128 and N32 alignment")
    if encoding not in ("trellis", "mxfp8"):
        raise ValueError("unsupported MX8 dense encoding")
    if sharded_side not in (1, 2) or input_dtype not in ("f16", "bf16"):
        raise ValueError("unsupported MX8 dense shard side or activation dtype")
    if encoding == "trellis" and output_features % 128:
        raise ValueError("trellis storage requires N128 alignment")
    if ((max_rows+15)//16)*(input_features//32) > (2**31-1)//8:
        raise ValueError("MX8 record indexing exceeds int32")
    return dict(input_features=input_features, output_features=output_features,
                encoding=encoding, sharded_side=sharded_side, max_rows=max_rows,
                rotation_count=rotation_count, input_dtype=input_dtype,
                bits=4, codebook="e4m3", abi_version=2)


def record_bytes(rows, input_features):
    return ((rows+15)//16)*(input_features//32)*528


def compile_dense_mxfp8_quant_aot(*, input_features, max_rows=4096,
                                 rotation_count=4, input_dtype="f16"):
    from b12x.gemm.dense_mxfp8 import TrellisMxfp8Quant
    g = geometry(input_features, max_rows=max_rows, rotation_count=rotation_count,
                 input_dtype=input_dtype)
    return compile_program(TrellisMxfp8Quant(g), name="dense_mxfp8_quant", operands=(
        Operand("rotated_inputs", torch.int64, f"[{rotation_count}]"),
        Operand("records", torch.uint8, "[record_bytes]", "out")),
        scalars=(Scalar("rows"), Scalar("rotation_id"), Scalar("cta_limit")),
        key=tuple(g.values()), geometry={**g, "exact_row_cap": 2**31 - 1})


# SM120 cold-L2 M1/2/4/8/16/32/64 curve, exact native group32 routes.
# Entries are (K, N): largest decode M; unmeasured shapes keep prefill.
NATIVE_DECODE_ROWS = {
    (6144, 8192): 32, (6144, 512): 64, (6144, 128): 64,
    (8192, 6144): 32, (6144, 3072): 64, (3072, 6144): 32,
    (6144, 12288): 32, (12288, 6144): 32, (6144, 4096): 64,
    (6144, 256): 64, (4096, 6144): 64, (6144, 1536): 64,
    (1536, 6144): 32, (6144, 6144): 32,
}


class RoutedNativeMxfp8:
    def __init__(self, g):
        from b12x.gemm.dense_mxfp8 import NativeMxfp8Decode, TrellisPrefillMxfp8
        self.decode = NativeMxfp8Decode(g)
        self.prefill = TrellisPrefillMxfp8(g)
        self.decode_rows = NATIVE_DECODE_ROWS.get((g["input_features"], g["output_features"]), 0)
        self.scratch_bytes = self.prefill.scratch_bytes

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, payload: cute.Pointer,
                 scales: cute.Pointer, records: cute.Pointer, output: cute.Pointer,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32, stream: cuda.CUstream):
        if rows <= Int32(self.decode_rows):
            self.decode(rotated_inputs, payload, scales, records, output,
                        rows, rotation_id, cta_limit, stream)
        else:
            self.prefill(rotated_inputs, payload, scales, records, output,
                         rows, rotation_id, cta_limit, stream)


def compile_dense_mxfp8_aot(*, input_features, output_features, encoding="trellis",
                          sharded_side=1, max_rows=4096, rotation_count=4,
                          input_dtype="f16", decode=False):
    from b12x.gemm.dense_mxfp8 import NativeMxfp8Decode, TrellisPrefillMxfp8
    g = geometry(input_features, output_features, encoding=encoding,
                 sharded_side=sharded_side, max_rows=max_rows,
                 rotation_count=rotation_count, input_dtype=input_dtype)
    k, n = input_features, output_features
    scale_dtype = torch.uint8 if encoding == "mxfp8" else torch.float16
    scale_count = k//128 if encoding == "trellis" else n*k//32
    if decode not in (False, True, "auto"):
        raise ValueError("native MXFP8 decode mode must be false, true or auto")
    if decode and encoding != "mxfp8":
        raise ValueError("native MXFP8 decode requires group32 weights")
    core = (RoutedNativeMxfp8(g) if decode == "auto" else
            NativeMxfp8Decode(g) if decode else TrellisPrefillMxfp8(g))
    if decode == "auto" and torch.cuda.get_device_capability()[1] != 0:
        core.decode_rows = 0
    return compile_program(core, name="dense_mxfp8", operands=(
        Operand("rotated_inputs", torch.int64, f"[{rotation_count}]"),
        Operand("payload", torch.uint8, f"[{k*n//(1 if encoding == 'mxfp8' else 2)}]"),
        Operand("scales", scale_dtype, f"[{scale_count}]"),
        Operand("records", torch.uint8, "[record_bytes]"),
        Operand("output", torch.float32, f"[rows,{n}]", "out")),
        scalars=(Scalar("rows"), Scalar("rotation_id"), Scalar("cta_limit")),
        key=(*g.values(), decode), geometry={**g, "exact_row_cap": 2**31 - 1})
