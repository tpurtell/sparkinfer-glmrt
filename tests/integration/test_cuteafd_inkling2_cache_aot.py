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


def _baseline_module(name):
    root = Path(os.environ["CUTEAFD_DEV_BASELINE_SOURCE"]) / "third_party/sparkinfer/b12x/integration/cuteafd"
    package = "b12x.integration.cuteafd"
    if name != "inkling2_attention":
        spec = importlib.util.spec_from_file_location(package + "." + name + "_baseline", root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
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
    return module


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
            # Torch lowers float32 division by a constant to reciprocal multiply;
            # native pack deliberately uses correctly rounded div.rn.f32.
            expected_scale = (source.abs().amax(dim=-1).double() / 448).float()
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
    baseline_program = _baseline_module("inkling2_attention").compile_inkling2_attention_aot(g, kind="global", route=route,
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
    # Independent dequantized-KV reference; verify causal/bias/sink addressing
    # rather than relying solely on the old exported attention implementation.
    start = 0
    for member, count in enumerate(counts):
        physical = tables[member].long().repeat_interleave(64)*64 + torch.arange(
            stride*64, device="cuda") % 64
        records = reference_cache[physical[:context]].float()
        keys = records[:, :1024].reshape(context, 8, 128).repeat_interleave(6, dim=1)
        values = records[:, 1024:].reshape(context, 8, 128).repeat_interleave(6, dim=1)
        for row in range(start, start+count):
            pos = int(positions[row])
            score = torch.einsum("hd,khd->hk", q[row].float(), keys[:pos+1]) * (128**-0.5)
            distance = pos-torch.arange(pos+1, device="cuda")
            bias = rel[row].float()[:, distance.clamp_max(1023)]
            score += torch.where(distance[None] < 1024, bias, 0.0)
            probability = torch.softmax(torch.cat([score, sinks[row,:,None]], dim=1), dim=1)[:,:-1]
            reference = torch.einsum("hk,khd->hd", probability, values[:pos+1])
            cosine = torch.nn.functional.cosine_similarity(out[row].float().flatten(), reference.flatten(), dim=0)
            assert float(cosine) >= 0.9999
        start += count


@pytest.mark.parametrize("route", ["decode", "prefill"])
@pytest.mark.parametrize("rows", [1, 16, 64])
def test_exported_local_cache(tmp_path, route, rows):
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_SMALL as g
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    torch.manual_seed(718 + rows)
    counts = [rows] if rows == 1 else [1, rows-1]
    stride_records = g.ring_rows + 16
    ring = torch.randn((len(counts)*stride_records, 2048), device="cuda").bfloat16()
    original = ring.clone()
    records = torch.randn(rows, 2048, device="cuda").bfloat16()
    q = torch.randn(rows, 48, 128, device="cuda").bfloat16()
    rel = torch.randn(rows, 48, 1024, device="cuda").bfloat16()
    sinks = torch.randn(rows, 48, device="cuda") + 4
    first, slots, tiles, positions = [], [], [], []
    start = 0
    for member, count in enumerate(counts):
        first.extend([start]*count)
        positions.extend(range(1103, 1103+count))
        slots.extend(member*stride_records + p % g.ring_rows for p in range(1103, 1103+count))
        tiles.extend(range(start, start+count, 1 if route == "decode" else 10))
        start += count
    device = lambda values, dtype: torch.tensor(values, device="cuda", dtype=dtype)
    first = device(first, torch.int32)
    slots = device(slots, torch.int64)
    positions = device(positions, torch.int64)
    tile_rows = device(tiles, torch.int32)
    out, expected = torch.empty_like(q), torch.empty_like(q)
    scratch = torch.empty(1024, device="cuda", dtype=torch.uint8)
    capacity = 64 if route == "decode" else 4096
    candidate = ExportedProgram(compile_inkling2_attention_aot(g, kind="local", route=route,
                                                             max_rows=capacity), tmp_path, "local")
    candidate.launch(q, records, ring, positions, slots, first, tile_rows, rel, sinks, out, scratch,
                     scalars=(rows, len(tiles)))
    baseline = ExportedProgram(_baseline_module("inkling2_attention").compile_inkling2_attention_aot(
        g, kind="local", route=route, max_rows=capacity), tmp_path, "old_local")
    start = 0
    for member, count in enumerate(counts):
        baseline.launch(q[start:start+count], records[start:start+count], original, positions[start:start+count],
                        slots[start:start+count], torch.zeros(count, device="cuda", dtype=torch.int32),
                        rel[start:start+count], sinks[start:start+count], expected[start:start+count], scratch,
                        scalars=(count,))
        start += count
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(ring, original)


@pytest.mark.parametrize("width,dynamic", [(1024, False), (1024, True), (4608, False)])
@pytest.mark.parametrize("rows", [1, 16, 64])
def test_exported_convolution_history_stride(tmp_path, width, dynamic, rows):
    require_b12x()
    from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot

    torch.manual_seed(882 + rows)
    counts = [rows] if rows == 1 else [1, rows-1]
    stride = 3*width + 4096
    history = torch.randn((len(counts)*stride,), device="cuda")
    initial = history.clone()
    x = torch.randn(rows, width, device="cuda")
    weights = torch.randn((4, width) if dynamic else (width, 4), device="cuda").bfloat16()
    coefficients = torch.randn(rows, 8, 4, device="cuda")
    first, slots = [], []
    start = 0
    for member, count in enumerate(counts):
        first.extend([start]*count)
        slots.extend([member]*count)
        start += count
    first = torch.tensor(first, device="cuda", dtype=torch.int32)
    slots = torch.tensor(slots, device="cuda", dtype=torch.int64)
    out, expected = torch.empty_like(x), torch.empty_like(x)
    params = dict(width=width, heads=8 if dynamic else 1, dynamic=dynamic, dtype=torch.float32)
    candidate = ExportedProgram(compile_inkling2_convolution_aot(**params), tmp_path, "conv")
    candidate.launch(x, weights, coefficients, history, first, slots, out, scalars=(rows,stride))
    baseline = ExportedProgram(_baseline_module("inkling2_convolution").compile_inkling2_convolution_aot(
        **params), tmp_path, "old_conv")
    start = 0
    for member, count in enumerate(counts):
        baseline.launch(x[start:start+count], weights, coefficients[start:start+count],
                        initial[member*stride:member*stride+3*width],
                        torch.zeros(count, device="cuda", dtype=torch.int32),
                        torch.zeros(count, device="cuda", dtype=torch.int64), expected[start:start+count],
                        scalars=(count,))
        start += count
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(history, initial)


@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exported_fp8_pack_scatter_zero_and_skipped_slots(tmp_path, rows, dtype):
    require_b12x()
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot

    torch.manual_seed(693 + rows)
    k = torch.randn(rows, 8, 128, device="cuda").to(dtype)
    v = torch.randn_like(k)
    k[0].zero_()
    v[0].zero_()
    slots = torch.randperm(rows+5, device="cuda")[:rows].long()
    if rows > 1:
        slots[-1] = -1
    cache = torch.full((rows+5, 2112), 0x5A, device="cuda", dtype=torch.uint8)
    expected = cache.clone()
    scale = [(source.float().abs().amax(dim=-1).double()/448).float() for source in (k,v)]
    scale = [torch.where(values > 0, values, 1.0) for values in scale]
    for row in range(rows):
        slot = int(slots[row])
        if slot < 0:
            continue
        for plane, source in enumerate((k,v)):
            payload = (source[row].float()*scale[plane][row].reciprocal()[:,None]).to(torch.float8_e4m3fn)
            expected[slot, plane*1024:(plane+1)*1024] = payload.view(torch.uint8).flatten()
            expected[slot, 2048+plane*32:2048+(plane+1)*32] = scale[plane][row].view(torch.uint8)
    program = compile_inkling2_kv_pack_aot(input_dtype=dtype, kv_format="fp8")
    ExportedProgram(program, tmp_path, "scatter").launch(k,v,slots,cache,scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(cache, expected)


@pytest.mark.parametrize("route", ["decode", "prefill"])
@pytest.mark.parametrize("kind,kv_format", [("local", "bf16"), ("global", "bf16"), ("global", "fp8")])
def test_exported_half_heads_match_whole_cache_and_attention(tmp_path, route, kind, kv_format):
    from dataclasses import replace
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot, record_bytes
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot

    torch.manual_seed(803)
    rows, capacity = 17, 64 if route == "decode" else 4096
    # Cross a global page boundary and cross the sliding ring's modulus.
    context = 1152
    positions = torch.arange(1081, 1081 + rows, device="cuda", dtype=torch.int64)
    tables = torch.arange(context // 64, device="cuda", dtype=torch.int32)
    first = torch.zeros(rows, device="cuda", dtype=torch.int32)
    tiles = torch.arange(0, rows, 1 if route == "decode" else 10, device="cuda", dtype=torch.int32)
    q = torch.randn(rows, 48, 128, device="cuda").bfloat16()
    rel = torch.randn(rows, 48, 1024, device="cuda").bfloat16()
    sinks = torch.rand(rows, 48, device="cuda", dtype=torch.float32)
    key = torch.randn(context if kind == "global" else 1088 + rows, 8, 128, device="cuda")
    value = torch.randn_like(key)
    outputs, caches = [], []
    for index, (k_start, k_end) in enumerate(((0, 8), (0, 4), (4, 8))):
        heads = k_end - k_start
        g = replace(INKLING2_SMALL, heads=heads * 6, kv_heads=heads)
        q_window = slice(k_start * 6, k_end * 6)
        k, v = [tensor[:, k_start:k_end].contiguous() for tensor in (key, value)]
        slots = torch.arange(len(k), device="cuda", dtype=torch.int64)
        cache = torch.empty((len(k), record_bytes(g, kv_format)), device="cuda", dtype=torch.uint8)
        pack = compile_inkling2_kv_pack_aot(heads=heads, kv_format=kv_format)
        ExportedProgram(pack, tmp_path, f"head_pack_{index}").launch(k, v, slots, cache, scalars=(len(k),))
        local_q, local_rel, local_sinks = [tensor[:, q_window].contiguous() for tensor in (q, rel, sinks)]
        out = torch.empty_like(local_q)
        scratch = torch.empty(64 * 32 * g.heads * 516, dtype=torch.uint8, device="cuda")
        prog = compile_inkling2_attention_aot(g, kind=kind, route=route, max_rows=capacity, kv_format=kv_format)
        launch = ExportedProgram(prog, tmp_path, f"head_attention_{index}")
        if kind == "global":
            scalars = (rows, len(tables), len(tiles)) + ((8,) if route == "decode" else ())
            launch.launch(local_q, cache, positions, tables, first, first, tiles, local_rel, local_sinks, out, scratch, scalars=scalars)
        else:
            # Cache positions before 1081 are read from the ring; live positions
            # come from the step records, then commit back into that same ring.
            ring, step = cache[:1088], cache[1088:]
            launch.launch(local_q, step, ring, positions, positions % 1088, first, tiles, local_rel,
                          local_sinks, out, scratch, scalars=(rows, len(tiles)))
        outputs.append(out)
        caches.append(cache)
    torch.cuda.synchronize()
    assert torch.equal(outputs[0], torch.cat(outputs[1:], dim=1))
    # Records are K and V planes; FP8 appends a scale plane for each.
    planes = (1024, 1024, 32, 32) if kv_format == "fp8" else (2048, 2048)
    whole = caches[0].split(planes, dim=1)
    halves = [cache.split(tuple(width // 2 for width in planes), dim=1) for cache in caches[1:]]
    assert all(torch.equal(full, torch.cat(pair, dim=1)) for full, pair in zip(whole, zip(*halves)))
