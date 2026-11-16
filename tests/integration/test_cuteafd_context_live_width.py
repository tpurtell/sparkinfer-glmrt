"""Frozen-capacity Context selector vs exact-live launches, including graph growth."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize('rows', [1, 16, 64])
def test_context_selector_graph_grows_from_empty_without_changing_capacity(rows):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.kimi_dsa import KimiDsaGeometry
    from b12x.integration.cuteafd.context_split import compile_scored_index_topk_aot
    from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference

    capacity = 8192
    with exportable_compilation():
        program = compile_scored_index_topk_aot(KimiDsaGeometry(), max_rows=64,
                                                max_pages=capacity, mode='decode')
    device = torch.device('cuda')
    generator = torch.Generator().manual_seed(9241 + rows)
    # Include a supertile crossing so dead carry folds and live folds both run.
    live_pages = [0, 1, 10, 20, 65, 130, 513, 1025]
    maximum = max(live_pages)
    keys = torch.randn((maximum * 64, 128), generator=generator).to(device)
    cache = pack_index_k_cache_reference(keys)
    table = torch.full((rows, capacity), -1, dtype=torch.int32, device=device)
    table[:, :maximum] = torch.arange(maximum, device=device, dtype=torch.int32)
    q = torch.randn((rows, 32, 128), generator=generator).to(device).to(torch.float8_e4m3fn)
    weights = (torch.randn((rows, 32), generator=generator) * .02).to(device)
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    indices = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    scores = torch.empty((rows, 2048), dtype=torch.float32, device=device)
    scratch = torch.empty(program.scratch_bytes(rows)['scratch'], dtype=torch.uint8, device=device)

    def launch(width, out_indices=indices, out_scores=scores):
        program.launch(q, weights, cache, table, lengths, out_indices, out_scores, scratch,
                       scalars=(rows, width, capacity, 2, 1))

    launch(capacity)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(capacity)
    expected_indices = torch.empty_like(indices)
    expected_scores = torch.empty_like(scores)
    curve = []
    for pages in live_pages:
        ends = torch.arange(rows, device=device, dtype=torch.int32)
        lengths.copy_((pages * 64 - ends).clamp(min=0))
        launch(max(1, pages), expected_indices, expected_scores)
        expected_i, expected_s = expected_indices.clone(), expected_scores.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(indices, expected_i), (rows, pages, 'indices')
        assert torch.equal(scores.view(torch.uint8), expected_s.view(torch.uint8)), (rows, pages, 'scores')
        for row in range(rows):
            assert int((indices[row] >= 0).sum()) == min(int(lengths[row]), 2048)
        if pages in (10, 20, 65, 130, 513, 1025):
            exact_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(exact_graph):
                launch(pages)
            for _ in range(3):
                graph.replay()
                exact_graph.replay()
            def measure(call):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    call()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1000 / 20
            exact_us = measure(exact_graph.replay)
            capacity_us = measure(graph.replay)
            curve.append(dict(rows=rows, pages=pages, exact_live_us=exact_us,
                              admitted_us=capacity_us, admitted_over_exact=capacity_us / exact_us))
    print('CONTEXT_LIVE_WIDTH_CURVE=' + json.dumps(curve), flush=True)
    if output := os.environ.get('CUTEAFD_CONTEXT_WIDTH_RECEIPTS'):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / f'rows-{rows}.json').write_text(json.dumps(curve, indent=2) + '\n')
