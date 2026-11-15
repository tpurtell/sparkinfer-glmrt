"""Exact expert-input wire oracles, cross-architecture receipts and warm timings.

Set B12X_NVFP4_WIRE_RECEIPTS to a writable directory to save the common BF16
inputs, packed row bytes and timing JSON. Run the same test on SM120/SM121 and
compare the saved *wire.pt tensors byte for byte.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

LIVE_ROWS = (1, 2, 16, 17, 127, 128, 129, 2048, 4096, 8192)


def _inputs(hidden, capacity):
    gen = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randn((capacity, hidden), generator=gen).bfloat16()
    # Every live count includes zero, tiny/subnormal blocks and huge magnitudes.
    x[0] = 0
    x[0, :16] = torch.linspace(-0.01, 0.01, 16).bfloat16()
    x[0, 16:32] = torch.linspace(-1e-20, 1e-20, 16).bfloat16()
    x[0, 32:48] = torch.linspace(-1e4, 1e4, 16).bfloat16()
    if capacity > 1:
        x[1] = 0
    return x


def _weights(g, raw_scale):
    gen = torch.Generator(device="cuda").manual_seed(91)
    e, h, i = g.experts, g.hidden, g.slice

    def operand(n, k):
        w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
        s = torch.full((e * n * (k // 16),), 8, dtype=torch.uint8, device="cuda")
        if g.dynamic_nvfp4:
            tail = torch.cat((torch.full((e,), 0.01, device="cuda"), torch.tensor([2688.0], device="cuda")))
        else:
            alpha = 1000.0 if g.scale_convention == "divisor" else 0.001
            tail = torch.cat((torch.full((e,), alpha, device="cuda"), torch.full((e,), raw_scale, device="cuda")))
        return w, torch.cat((s, tail.view(torch.uint8)))

    return (*operand(i, h), *operand(i, h), *operand(h, i))


def _warm_us(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            fn()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 25


def _scratch_rows(g, scratch, rows, capacity):
    from b12x.integration.cuteafd._fp8_moe_kernels import META_HEAD, meta_words
    from b12x.integration.cuteafd._fp8_moe_stream import stream_max_tiles
    from b12x.integration.cuteafd.fp8_moe import _align

    pairs = rows * g.top_k
    tiles = stream_max_tiles(g.experts, pairs)
    row_at = _align(4 * meta_words(g.experts, stream_max_tiles(g.experts, capacity * g.top_k)))
    xq_at = row_at + 2 * _align(pairs * 4)
    xs_at = xq_at + _align(tiles * 128 * g.hidden // 2)
    xa_at = xs_at + _align(tiles * 128 * g.hidden // 16)
    meta = scratch[:row_at].view(torch.int32).cpu()
    count = int(meta[1])
    pair_row = scratch[row_at:row_at + pairs * 4].view(torch.int32).cpu()
    padded, sources = [], []
    for tile in range(count):
        code = int(meta[META_HEAD + 3 * g.experts + tile])
        e, row0 = code & 0xffff, (code >> 16) * 128
        live = min(128, int(meta[META_HEAD + e]) - row0)
        first = int(meta[META_HEAD + g.experts + e]) + row0
        padded.extend(range(tile * 128, tile * 128 + live))
        sources.extend(pair_row[first:first + live].tolist())
    dest = torch.tensor(padded, device="cuda", dtype=torch.int64)
    src = torch.tensor(sources, device="cuda", dtype=torch.int64)
    q = scratch[xq_at:xq_at + tiles * 128 * g.hidden // 2].view(-1, g.hidden // 2)[dest]
    s = scratch[xs_at:xs_at + tiles * 128 * g.hidden // 16].view(-1, g.hidden // 16)[dest]
    amax = (scratch[xa_at:xa_at + tiles * 128 * 4].view(torch.float32)[dest]
            if g.dynamic_nvfp4 else None)
    return src, q, s, amax


@pytest.mark.parametrize("name,tp,raw_scale", [
    ("minimax_nvfp4a4", 1, 1.0), ("minimax_nvfp4a4", 3, 0.37),
    ("inkling2_nvfp4", 1, 1.0), ("inkling2_nvfp4", 4, 1.0),
])
def test_wire_program_exact(name, tp, raw_scale):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    from b12x.integration.cuteafd.nvfp4_wire import compile_nvfp4_wire_rows_aot, nvfp4_wire_row_bytes

    g = GEOMETRIES[name].with_tp(tp)
    capacity = 8192
    mode = "dynamic" if g.dynamic_nvfp4 else "calibrated"
    quant = compile_nvfp4_wire_rows_aot(hidden=g.hidden, max_rows=capacity, mode=mode,
                                       scale_convention=g.scale_convention)
    bf16 = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    wire = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, input_kind="nvfp4")
    addressed = (None if g.dynamic_nvfp4 else compile_fp8_moe_aot(
        g, route="auto", max_rows=capacity, input_kind="nvfp4", row_addresses=True))
    assert bf16.scratch_bytes(capacity) == wire.scratch_bytes(capacity)
    assert fp8_moe_scratch_bytes(g, "auto", capacity, False) == fp8_moe_scratch_bytes(
        g, "auto", capacity, input_kind="nvfp4")
    cpu_x = _inputs(g.hidden, capacity)
    x = cpu_x.cuda()
    packed = torch.full((capacity, nvfp4_wire_row_bytes(g.hidden)), 0xa5, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(712)
    ids = torch.randint(0, g.experts, (capacity, g.top_k), device="cuda", generator=gen, dtype=torch.int32)
    # Duplicate expert IDs and concentrated chunks cover both <=16 and >16 paths.
    ids[:17] = 0
    gamma = torch.rand((capacity, g.top_k), device="cuda", generator=gen)
    weights = _weights(g, raw_scale)
    out_a, out_b = (torch.empty_like(x) for _ in range(2))
    scratch_a, scratch_b = (torch.empty(bf16.scratch_bytes(capacity)["scratch"], dtype=torch.uint8, device="cuda") for _ in range(2))
    if addressed is not None:
        assert addressed.scratch_bytes(capacity) == wire.scratch_bytes(capacity)
        out_addressed = torch.empty_like(x)
        # Reverse the physical row storage: the address table must not be ignored.
        retained = packed.clone()
        order = torch.arange(capacity - 1, -1, -1, device="cuda", dtype=torch.int64)
        addresses = retained.data_ptr() + order * retained.stride(0)
        output_addresses = out_addressed.data_ptr() + torch.arange(capacity, device="cuda", dtype=torch.int64) * g.hidden * 2
    timing = []
    with kernel_resolution_guard("NVFP4 wire live rows reuse fixed planned capacity"):
        # Zero-row quantization must not touch the destination.
        quant.launch(x, packed, scalars=(0,) if g.dynamic_nvfp4 else (0, raw_scale))
        assert bool((packed == 0xa5).all())
        for rows in LIVE_ROWS:
            scalars = (rows,) if g.dynamic_nvfp4 else (rows, raw_scale)
            qrun = lambda: quant.launch(x, packed, scalars=scalars)
            arun = lambda: bf16.launch(x, ids, gamma, *weights, out_a, scratch_a, scalars=(rows,))
            brun = lambda: wire.launch(packed, ids, gamma, *weights, out_b, scratch_b, scalars=(rows,))
            qrun()
            arun()
            brun()
            torch.cuda.synchronize()
            assert torch.equal(out_a[:rows].view(torch.uint8), out_b[:rows].view(torch.uint8)), (name, tp, rows)
            assert torch.isfinite(out_a[:rows]).all()
            assert out_a[:rows].count_nonzero() > 0
            if addressed is not None:
                retained[order[:rows]] = packed[:rows]
                addressed.launch(addresses, ids, gamma, *weights, output_addresses, scratch_b, scalars=(rows,))
                torch.cuda.synchronize()
                assert torch.equal(out_a[:rows].view(torch.uint8), out_addressed[:rows].view(torch.uint8))
            src, q, s, amax = _scratch_rows(g, scratch_a, rows, capacity)
            assert torch.equal(q, packed[src, :g.hidden // 2]), (name, tp, rows, "packed")
            assert torch.equal(s, packed[src, g.hidden // 2:g.hidden // 2 + g.hidden // 16]), (name, tp, rows, "scales")
            src_b, qb, sb, amax_b = _scratch_rows(g, scratch_b, rows, capacity)
            # MoePrep atomics may permute rows inside each expert between launches.
            assert torch.equal(qb, packed[src_b, :g.hidden // 2])
            assert torch.equal(sb, packed[src_b, g.hidden // 2:g.hidden // 2 + g.hidden // 16])
            tail = packed[:rows, g.hidden // 2 + g.hidden // 16:]
            assert bool((tail[:, 4:] == 0).all())
            if g.dynamic_nvfp4:
                assert torch.equal(amax.view(torch.uint8), tail[:, :4].contiguous().view(torch.float32).flatten()[src].view(torch.uint8))
                assert torch.equal(amax_b.view(torch.uint8), tail[:, :4].contiguous().view(torch.float32).flatten()[src_b].view(torch.uint8))
            else:
                assert bool((tail == 0).all())
            if rows in (1, 16, 2048, 8192):
                point = dict(rows=rows, quant_us=_warm_us(qrun), bf16_us=_warm_us(arun), nvfp4_us=_warm_us(brun))
                timing.append(point)
                print(name, tp, raw_scale, point, flush=True)
    receipt_dir = os.environ.get("B12X_NVFP4_WIRE_RECEIPTS")
    if receipt_dir:
        root = Path(receipt_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{name}-tp{tp}-scale{raw_scale}"
        torch.save(cpu_x, root / f"{stem}-input.pt")
        torch.save(packed.cpu(), root / f"{stem}-wire.pt")
        (root / f"{stem}-timings.json").write_text(json.dumps(dict(
            device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            geometry=name, tp=tp, raw_scale=raw_scale, points=timing), indent=2))


def test_wire_validation_and_legacy_keys():
    require_b12x()
    from b12x.integration.cuteafd._nvfp4_moe_a4 import ChunkRowsA4
    from b12x.integration.cuteafd._nvfp4_moe_dynamic import ChunkRowsDyn
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, _resolve, fp8_moe_reply_layout, fp8_moe_scratch_bytes
    from b12x.integration.cuteafd.nvfp4_wire import compile_nvfp4_wire_rows_aot, nvfp4_wire_row_bytes

    assert nvfp4_wire_row_bytes(6144) == 3472 and nvfp4_wire_row_bytes(4608) == 2608
    for name in ("mimo", "mimop", "inkling2_mxfp8", "step5_bf16"):
        with pytest.raises(ValueError, match="NVFP4 wire input requires"):
            fp8_moe_scratch_bytes(GEOMETRIES[name], "auto", 17, input_kind="nvfp4")
    for name in ("minimax_nvfp4a4", "inkling2_nvfp4"):
        g = GEOMETRIES[name]
        assert _resolve("auto", 8192, weights=g.kind, input_kind="nvfp4") == "stream"
        assert fp8_moe_reply_layout(g, route="auto", rows=17, max_rows=8192, wire=False) == fp8_moe_reply_layout(
            g, route="auto", rows=17, max_rows=8192, input_kind="nvfp4")
    for wire in (False, True):
        a = ChunkRowsA4(k=6144, wire=wire, experts=128, scale_rows=1024)
        assert a.key() == ("chunk_rows_a4", 2, 6144, wire, 128, a.in_offset)
    assert ChunkRowsDyn(k=4608, experts=384).key() == ("chunk_rows_dyn", 2, 4608, 384)
    for kwargs in (dict(mode="unknown"), dict(mode="calibrated", scale_convention="unknown")):
        with pytest.raises(ValueError):
            compile_nvfp4_wire_rows_aot(hidden=6144, max_rows=8192, **kwargs)
    assert "nonuniform" in compile_nvfp4_wire_rows_aot.__doc__


@pytest.mark.parametrize("mode,convention", [("calibrated", "multiplier"), ("calibrated", "divisor"), ("dynamic", "multiplier")])
def test_wire_exported_abi(tmp_path, mode, convention):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.nvfp4_wire import compile_nvfp4_wire_rows_aot

    with exportable_compilation():
        program = compile_nvfp4_wire_rows_aot(hidden=6144, max_rows=8192, mode=mode, scale_convention=convention)
    stem = f"nvfp4_wire_{mode}_{convention}"
    program.export_to_c(str(tmp_path), stem, stem)
    checked = validate_exported_header(program, tmp_path / (stem + ".h"), stem)
    assert checked["argument_count"] == (4 if mode == "dynamic" else 5)
    assert program.scratch_bytes(8192) == {}
