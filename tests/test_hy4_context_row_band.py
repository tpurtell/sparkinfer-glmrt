"""Opt-in SM120 row-band timing for Hy4 ContextSplit kernels.

Set HY4_CONTEXT_BENCH=1 and run pytest -s tests/test_hy4_context_row_band.py.
Covers the policy live-row band (2048/2688/2752/4096) plus real decode/prefill tails
(1/8/16/40/64) on SM120. Hy4 attention ships on RTX SM120 only (SM121 is not a target).
"""
from __future__ import annotations

import json
import os
import statistics

import pytest

torch = pytest.importorskip("torch")

from b12x.integration.cuteafd.context_split import compile_lse_combine2_aot
from b12x.integration.cuteafd.hy4_indexer import (
    HY4_FLASH_INDEXER,
    compile_hy4_indexer_merge_expand_owned_aot,
    compile_hy4_indexer_scored_topk_aot,
)
from b12x.integration.cuteafd.hy4_mla import (
    Hy4MlaGeometry,
    compile_hy4_mla_dsa_partial_aot,
    compile_hy4_mla_gate_uv_aot,
)
from tests._reference.helpers import require_b12x

ROWS = (1, 8, 16, 40, 64, 2048, 2688, 2752, 4096)
CONTEXT = 32768


def _capture(work, stream):
    with torch.cuda.stream(stream):
        for _ in range(3):
            work()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        work()
    return graph


def _measure_us(graph, replays: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _dsa_partial_splits(rows: int, sm_count: int) -> int:
    # 16 heads / 8 heads_per_tile = 2 head tiles per row; max 16 splits for 1024 owned tokens (16 chunks of 64).
    wave = (sm_count + rows * 2 - 1) // (rows * 2)
    return max(2, min(16, wave))


@pytest.mark.skipif(not os.environ.get("HY4_CONTEXT_BENCH"), reason="opt-in Hy4 ContextSplit SM120 row-band timing")
def test_hy4_context_split_row_band():
    device = require_b12x()
    assert torch.cuda.get_device_capability(device) == (12, 0), "Hy4 attention ships on SM120 only"
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    g_idx = HY4_FLASH_INDEXER
    g32_bf16 = Hy4MlaGeometry(heads=32, window_size=None, kv_cache="bf16")
    g32_fp8 = Hy4MlaGeometry(heads=32, window_size=None, kv_cache="fp8")
    g16 = Hy4MlaGeometry(heads=16, window_size=None, kv_cache="bf16")

    p_bf16 = compile_hy4_mla_dsa_partial_aot(g32_bf16, max_rows=4096, head_begin=0, head_count=16, row_chunk_size=64)
    p_fp8 = compile_hy4_mla_dsa_partial_aot(g32_fp8, max_rows=4096, head_begin=0, head_count=16, row_chunk_size=64)
    scored_topk = compile_hy4_indexer_scored_topk_aot(g_idx)
    merge_owned = compile_hy4_indexer_merge_expand_owned_aot(g_idx)
    comb = compile_lse_combine2_aot(heads=16, has_sink=True)
    gate_uv = compile_hy4_mla_gate_uv_aot(g16)

    print(
        json.dumps({"device": props.name, "sms": sm_count, "context": CONTEXT, "rows": ROWS}),
        flush=True,
    )

    # Pre-allocate shared caches (1024 owned tokens = 16 local pages for partial; 32K context = 256 local pages for indexer).
    kv_bf16 = torch.randn(32, 64, 576, dtype=torch.bfloat16, device=device)
    kv_fp8 = torch.randint(0, 127, (32, 64, 656), dtype=torch.uint8, device=device)
    w_uv = torch.randn(16, 256, 512, dtype=torch.bfloat16, device=device) * 0.04
    sink = torch.randn(16, dtype=torch.float32, device=device) * 0.5
    scratch = torch.empty(p_bf16.scratch_bytes(4096)["scratch"], dtype=torch.uint8, device=device)

    half_pages = (CONTEXT // 64) // 2
    half_groups = half_pages * 16
    results = []

    for rows in ROWS:
        rng = torch.Generator(device=device).manual_seed(1202604 + rows)
        replays = 16 if rows <= 64 else 6
        splits = _dsa_partial_splits(rows, sm_count)

        # Partial inputs (1024 owned tokens per row = half of 2048 topk).
        query = torch.randn(rows, 32, 576, dtype=torch.bfloat16, device=device, generator=rng) * 0.2
        pt_tok = torch.arange(1024, dtype=torch.int32, device=device).unsqueeze(0).expand(rows, 1024).contiguous()
        seqlens = torch.full((rows,), 1024, dtype=torch.int32, device=device)
        cu = torch.arange(rows + 1, dtype=torch.int32, device=device)
        o0 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=device)
        l0 = torch.empty(rows, 16, dtype=torch.float32, device=device)
        o1 = torch.randn(rows, 16, 512, dtype=torch.bfloat16, device=device, generator=rng)
        l1 = torch.randn(rows, 16, dtype=torch.float32, device=device, generator=rng)
        c_out = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=device)
        gate = torch.randn(rows, 16, 256, dtype=torch.bfloat16, device=device, generator=rng)
        uv_out = torch.empty(rows, 16, 256, dtype=torch.bfloat16, device=device)

        # Indexer inputs (32K context -> 4096 local groups per rank).
        scores_loc = torch.randn(rows, half_groups, dtype=torch.float32, device=device, generator=rng)
        positions = torch.arange(CONTEXT - rows, CONTEXT, dtype=torch.int32, device=device)
        limits_loc = torch.full((rows,), half_groups, dtype=torch.int32, device=device)
        cand_s_loc = torch.empty(rows, 512, dtype=torch.float32, device=device)
        cand_i_loc = torch.empty(rows, 512, dtype=torch.int32, device=device)
        scored_topk.launch(scores_loc, positions, limits_loc, cand_s_loc, cand_i_loc, scalars=(rows, half_groups, 2, 0))
        cand_s_peer = torch.empty(rows, 512, dtype=torch.float32, device=device)
        cand_i_peer = torch.empty(rows, 512, dtype=torch.int32, device=device)
        scored_topk.launch(scores_loc, positions, limits_loc, cand_s_peer, cand_i_peer, scalars=(rows, half_groups, 2, 1))
        cand_s = torch.cat([cand_s_loc, cand_s_peer], dim=1).contiguous()
        cand_i = torch.cat([cand_i_loc, cand_i_peer], dim=1).contiguous()
        pt_idx = torch.randperm(half_pages, dtype=torch.int32, device=device, generator=rng).expand(rows, half_pages).contiguous()
        loc_slots = torch.empty(rows, 2048, dtype=torch.int32, device=device)
        loc_lens = torch.empty(rows, dtype=torch.int32, device=device)
        draft_slots = torch.empty(rows, 2048, dtype=torch.int32, device=device)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))

        arms = {
            "partial_bf16": lambda: p_bf16.launch(query, kv_bf16, pt_tok, seqlens, cu, o0, l0, scratch, scalars=(rows, 1024, splits)),
            "partial_fp8": lambda: p_fp8.launch(query, kv_fp8, pt_tok, seqlens, cu, o0, l0, scratch, scalars=(rows, 1024, splits)),
            "scored_topk": lambda: scored_topk.launch(scores_loc, positions, limits_loc, cand_s_loc, cand_i_loc, scalars=(rows, half_groups, 2, 0)),
            "merge_owned": lambda: merge_owned.launch(cand_s, cand_i, positions, pt_idx, loc_slots, loc_lens, draft_slots, scalars=(rows, half_pages, 0, 0)),
            "combine_gate_uv": lambda: (
                comb.launch(o0, l0, o1, l1, sink, c_out, scalars=(rows,)),
                gate_uv.launch(c_out, w_uv, gate, uv_out, scalars=(rows,)),
            ),
        }
        graphs = {k: _capture(fn, stream) for k, fn in arms.items()}
        medians = {}
        for name, graph in graphs.items():
            samples = []
            for _ in range(5):
                for _ in range(2):
                    graph.replay()
                samples.append(_measure_us(graph, replays=replays))
            medians[name] = statistics.median(samples)

        row_entry = {
            "rows": rows,
            "splits": splits,
            "us": {k: round(v, 2) for k, v in medians.items()},
            "us_per_row": {k: round(v / rows, 4) for k, v in medians.items()},
        }
        results.append(row_entry)
        print(json.dumps(row_entry), flush=True)

    # Gate: within the high-row live band (2048, 2688, 2752, 4096), verify no >10% per-row collapse.
    band = [r for r in results if r["rows"] in (2048, 2688, 2752, 4096)]
    for key in ("partial_bf16", "partial_fp8", "scored_topk", "merge_owned", "combine_gate_uv"):
        vals = [r["us_per_row"][key] for r in band]
        best = min(vals)
        worst = max(vals)
        assert worst <= best * 1.10, f"{key} high-row band collapse: {dict(zip([r['rows'] for r in band], vals))}"
