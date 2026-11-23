"""Opt-in SM120 timing of the Hy4 selector alone, on the same scores in both arms.

Set HY4_INDEXER_STOCK to the pinned stock hy4_indexer.py and run pytest -s.
Score generation, compilation, capture and correctness checks are untimed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import pytest

torch = pytest.importorskip("torch")

from b12x.integration.cuteafd.hy4_indexer import compile_hy4_indexer_topk_expand_aot
from tests._reference.helpers import require_b12x

# 40: standard 8K-with-tail fixture; 2688: 8192 modulo a 2752-row unit.
ROWS = (1, 8, 40, 64, 2048, 2688, 2752, 4096)
CONTEXT = 32768


def _stock_program(path, monkeypatch):
    # Import just the stock kernel, sharing the candidate's compiler and helpers.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"b12x.integration.cuteafd._hy4_indexer_stock_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    compile_program = module.compile_program

    def compile_stock(launch, *, key, **kwargs):
        # Explicit-spec memory caches do not include the Python target.
        return compile_program(launch, key=(*key, digest), **kwargs)

    monkeypatch.setattr(module, "compile_program", compile_stock)
    return module.compile_hy4_indexer_topk_expand_aot(), digest


def _capture(work, stream):
    with torch.cuda.stream(stream):
        for _ in range(3):
            work()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        work()
    return graph


def _measure_us(graph, replays):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / replays


@pytest.mark.skipif(not os.environ.get("HY4_INDEXER_STOCK"), reason="opt-in Hy4 selector timing")
def test_hy4_indexer_topk_row_band(monkeypatch):
    device = require_b12x()
    assert torch.cuda.get_device_capability(device) == (12, 0), "Hy4 selector ships on SM120 only"
    stock, digest = _stock_program(Path(os.environ["HY4_INDEXER_STOCK"]), monkeypatch)
    stable = compile_hy4_indexer_topk_expand_aot()
    print(json.dumps({"stock_source_sha256": digest, "device": torch.cuda.get_device_name(device),
                      "sms": torch.cuda.get_device_properties(device).multi_processor_count,
                      "context": CONTEXT, "rows": ROWS}), flush=True)
    for rows in ROWS:
        rng = torch.Generator(device=device).manual_seed(12032768 + rows)
        groups, pages = CONTEXT // 4, CONTEXT // 64
        scores = torch.randn((rows, groups), device=device, generator=rng)
        positions = torch.arange(CONTEXT - rows, CONTEXT, dtype=torch.int32, device=device)
        pages_in_order = torch.randperm(pages, dtype=torch.int32, device=device, generator=rng)
        table = pages_in_order.expand(rows, pages).contiguous()
        output = {
            arm: (torch.full((rows, 2048), -2, dtype=torch.int32, device=device),
                  torch.full((rows,), -1, dtype=torch.int32, device=device))
            for arm in ("stock", "stable")
        }
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graphs = {
            arm: _capture(lambda: program.launch(scores, positions, table, *output[arm],
                                                 scalars=(rows, groups, pages)), stream)
            for arm, program in (("stock", stock), ("stable", stable))
        }
        # Each arm warms every round; alternate the measurement order on one GPU.
        samples = {arm: [] for arm in graphs}
        for round_id in range(8):
            order = ("stock", "stable") if round_id % 2 == 0 else ("stable", "stock")
            for arm in order:
                for _ in range(3):
                    graphs[arm].replay()
                samples[arm].append(_measure_us(graphs[arm], replays=16))
        base_slots, base_lengths = output["stock"]
        stable_slots, stable_lengths = output["stable"]
        assert torch.equal(base_lengths, 2044 + positions.remainder(4) + 1)
        assert torch.equal(base_lengths, stable_lengths)
        assert torch.equal(base_slots.sort(dim=-1).values, stable_slots.sort(dim=-1).values)
        prior = stable_slots.clone()
        stable_slots.fill_(-2)
        graphs["stable"].replay()
        assert torch.equal(stable_slots, prior), f"unstable ordered bytes at rows={rows}"
        medians = {arm: statistics.median(us) for arm, us in samples.items()}
        print(json.dumps({"rows": rows, "us": medians,
                          "us_per_row": {arm: us / rows for arm, us in medians.items()},
                          "paired_ratio": statistics.median(c / b for b, c in zip(samples["stock"], samples["stable"])),
                          "samples_us": samples}), flush=True)
