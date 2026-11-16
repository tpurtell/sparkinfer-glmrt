"""Combined cache ABI oracles through exported C wrappers, never JIT launch."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import types
import sys

import pytest
import torch

from ..conftest import require_b12x
from ..moe.segment_aot import ExportedProgram


def _baseline():
    root = Path(os.environ["CUTEAFD_DEV_BASELINE_SOURCE"]) / "third_party/sparkinfer/b12x/integration/cuteafd"
    package = "b12x.integration.cuteafd"
    spec = importlib.util.spec_from_file_location(package + "._inkling2_baseline", root / "_inkling2_kernels.py")
    kernels = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernels
    spec.loader.exec_module(kernels)
    module = types.ModuleType(package + ".inkling2_attention_baseline")
    sys.modules[module.__name__] = module
    module.__package__ = package
    source = (root / "inkling2_attention.py").read_text().replace(
        "from ._inkling2_kernels import InklingGqaAttention, InklingSplitMerge", "")
    module.__dict__.update(InklingGqaAttention=kernels.InklingGqaAttention,
                           InklingSplitMerge=kernels.InklingSplitMerge)
    exec(compile(source, str(root / "inkling2_attention.py"), "exec"), module.__dict__)
    return module.compile_inkling2_attention_aot


@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("route", ["decode", "prefill"])
@pytest.mark.parametrize("kv_format", ["bf16", "fp8"])
def test_exported_global_cache(tmp_path, rows, route, kv_format):
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_SMALL as g
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot

    torch.manual_seed(921 + rows)
    # Two different-length histories and deliberately non-tile-aligned segments.
    counts = [rows] if rows == 1 else [1, rows - 1]
    context = 4111
    stride = (context + 63) // 64
    page_count = len(counts) * stride
    tables = torch.randperm(page_count, device="cuda").int().reshape(len(counts), stride)
    k = torch.randn(page_count * 64, 8, 128, device="cuda")
    v = torch.randn_like(k)
    record_bytes = 2112 if kv_format == "fp8" else 4096
    cache = torch.zeros(page_count * 64, record_bytes, dtype=torch.uint8, device="cuda")
    pack = compile_inkling2_kv_pack_aot(kv_format=kv_format, max_rows=16384)
    pack_aot = ExportedProgram(pack, tmp_path, "pack")
    slots = torch.arange(page_count * 64, device="cuda")
    pack_aot.launch(k, v, slots, cache, scalars=(len(slots),))
    reference_cache = torch.cat([k.reshape(-1, 1024), v.reshape(-1, 1024)], dim=1).bfloat16()
    if kv_format == "fp8":
        scales = cache[:, 2048:].contiguous().view(torch.float32).reshape(-1, 2, 8)
        for plane, source in enumerate([k, v]):
            expected_scale = source.abs().amax(dim=-1) / 448
            expected_scale = torch.where(expected_scale > 0, expected_scale, 1.0)
            torch.testing.assert_close(scales[:, plane], expected_scale, atol=0, rtol=0)
            expected = (source * expected_scale.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn)
            payload = cache[:, plane * 1024:(plane + 1) * 1024].contiguous().view(torch.float8_e4m3fn)
            torch.testing.assert_close(payload.view(torch.uint8).reshape_as(expected.view(torch.uint8)),
                                       expected.view(torch.uint8), atol=0, rtol=0)
            reference_cache[:, plane * 1024:(plane + 1) * 1024] = (
                payload.float().reshape(-1, 8, 128) * expected_scale.unsqueeze(-1)).reshape(-1, 1024).bfloat16()
    else:
        assert torch.equal(cache, reference_cache.view(torch.uint8))
    positions, first, indices, tiles = [], [], [], []
    start = 0
    for member, count in enumerate(counts):
        positions.extend(range(context - count, context))
        first.extend([start] * count)
        indices.extend([member] * count)
        tiles.extend(range(start, start + count, 1 if route == "decode" else 10))
        start += count
    device = lambda values, dtype: torch.tensor(values, dtype=dtype, device="cuda")
    positions = device(positions, torch.int64)
    first = device(first, torch.int32)
    indices = device(indices, torch.int32)
    tile_rows = device(tiles, torch.int32)
    q = torch.randn(rows, 48, 128, device="cuda").bfloat16()
    rel = (torch.randn(rows, 48, 1024, device="cuda") * 2).bfloat16()
    sinks = torch.randn(rows, 48, device="cuda") + 4
    out = torch.empty_like(q)
    scratch = torch.empty(64 * 32 * 48 * 516, dtype=torch.uint8, device="cuda")
    splits = 1 if route == "prefill" else 8
    program = compile_inkling2_attention_aot(g, kind="global", route=route,
                                            max_rows=64 if route == "decode" else 4096, kv_format=kv_format)
    candidate = ExportedProgram(program, tmp_path, "attention")
    scalars = (rows, stride, len(tiles)) + ((splits,) if route == "decode" else ())
    candidate.launch(q, cache, positions, tables, indices, first, tile_rows, rel, sinks, out, scratch, scalars=scalars)
    baseline_program = _baseline()(g, kind="global", route=route,
                                  max_rows=64 if route == "decode" else 4096)
    baseline = ExportedProgram(baseline_program, tmp_path, "baseline")
    expected = torch.empty_like(q)
    start = 0
    for member, count in enumerate(counts):
        table = tables[member].repeat(count, 1) if route == "decode" else tables[member]
        old_scalars = (count, stride if route == "decode" else 0) + ((splits,) if route == "decode" else ())
        baseline.launch(q[start:start + count], reference_cache, positions[start:start + count], table,
                        rel[start:start + count], sinks[start:start + count], expected[start:start + count],
                        scratch, scalars=old_scalars)
        start += count
    torch.cuda.synchronize()
    # FP8 also matches exactly: baseline reads the same dequantized BF16 operands.
    assert torch.equal(out, expected), (out.float() - expected.float()).abs().max().item()
