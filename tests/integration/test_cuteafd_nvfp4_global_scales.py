"""Checkpoint-native divisor globals: real MiniMax slices and reciprocal regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_nvfp4_moe_aot import dequant


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_divisor_reciprocal_round_trip_is_not_identity():
    # MiniMax layer 3, expert 0, w1.weight_global_scale at revision 54e4e69d.
    checkpoint = struct.unpack("<f", bytes.fromhex("c54e6c46"))[0]
    old = _f32(1.0 / _f32(1.0 / checkpoint))
    assert old != checkpoint
    assert struct.pack("<f", old).hex() == "c44e6c46"


def _tensor(snapshot, index, name):
    with (snapshot / index[name]).open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(size))[name]
        start, end = header["data_offsets"]
        file.seek(8 + size + start)
        return header, file.read(end - start)


@pytest.mark.parametrize("projection", ["w1", "w2"])
def test_real_minimax_divisor_slice(projection):
    """Run the production A4 linear on a small real expert slice vs F64 dequant."""
    require_b12x()
    selected = os.environ.get("B12X_MINIMAX_SNAPSHOT")
    if not selected:
        pytest.skip("set B12X_MINIMAX_SNAPSHOT to the official MiniMax NVFP4 snapshot")
    snapshot = Path(selected)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    stem = f"language_model.model.layers.3.block_sparse_moe.experts.0.{projection}"
    wh, wb = _tensor(snapshot, index, stem + ".weight_packed")
    sh, sb = _tensor(snapshot, index, stem + ".weight_scale")
    _, global_bytes = _tensor(snapshot, index, stem + ".weight_global_scale")
    _, input_bytes = _tensor(snapshot, index, stem + ".input_global_scale")
    weight_global = struct.unpack("<f", global_bytes)[0]
    input_global = struct.unpack("<f", input_bytes)[0]
    n, k = 128, 256
    w = torch.frombuffer(bytearray(wb), dtype=torch.uint8).reshape(wh["shape"])[:n, :k // 2].contiguous()
    s = torch.frombuffer(bytearray(sb), dtype=torch.uint8).reshape(sh["shape"])[:n, :k // 16].contiguous()
    weights = w.cuda()
    scales = torch.cat([s.flatten(), torch.tensor(list(global_bytes + input_bytes), dtype=torch.uint8)]).cuda()
    # A4 values/scales are exactly one: isolate weight-global application from
    # quantizer boundaries and nonlinearities. All products/sums are exact here.
    aq = torch.full((128, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    a_s = torch.full((128, k // 16), 0x38, dtype=torch.uint8, device="cuda")
    from b12x.integration.cuteafd._fp8_moe_kernels import META_HEAD
    from b12x.integration.cuteafd._nvfp4_moe_a4 import StreamNvfp4LinearA4
    from b12x.integration.cuteafd._common import Operand, compile_program

    meta = torch.zeros(META_HEAD + 4, dtype=torch.int32)
    meta[1], meta[META_HEAD] = 1, 1
    meta = meta.cuda()
    kernel = StreamNvfp4LinearA4(n=n, k=k, experts=1, scale_convention="divisor")
    program = compile_program(kernel, name="nvfp4_real_divisor_slice", operands=(
        Operand("meta", torch.int32, "[meta]"), Operand("aq", torch.uint8, "[128,K/2]"),
        Operand("a_s", torch.uint8, "[128,K/16]"), Operand("w", torch.uint8, "[N,K/2]"),
        Operand("s", torch.uint8, "[scales]"), Operand("y", torch.bfloat16, "[1,N]", "out")),
        scalars=(), key=kernel.key())
    out = torch.full((1, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    program.launch(meta, aq, a_s, weights, scales, out)
    # The raw GPU operand has the checkpoint's bits, not a reciprocal round trip.
    assert bytes(scales[-8:].cpu().tolist()) == global_bytes + input_bytes
    expected = (dequant(w, s).double().sum(-1) / float(weight_global) / float(input_global)).bfloat16()
    torch.testing.assert_close(out[0].cpu(), expected, rtol=0, atol=0)
    assert torch.isfinite(out).all() and torch.count_nonzero(out)
    old_effective = _f32(1.0 / _f32(1.0 / weight_global))
    print(f"{stem}: raw_global={weight_global.hex()} old_effective={old_effective.hex()} "
          f"changed={old_effective != weight_global}; F64 dequant -> production A4 linear exact {n} outputs")
