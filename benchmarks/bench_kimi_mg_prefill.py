#!/usr/bin/env python3
"""Fork-only Kimi MG oracle and interleaved warmed graph row curve.

Baselines are the shipping FP8 dense extend and split16/chunk64 sparse exports.
The MG oracle uses #305's existing output/LSE bounds, not byte identity between
algorithms. Same-row cross-width checks are byte exact within each MG variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def records(tokens, seed=7321):
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
    torch.manual_seed(seed)
    raw = torch.randn(tokens, 576, device="cuda").bfloat16()
    cache = pack_mla_kv_cache_reference(raw[:, :512], raw[:, 512:]).reshape(tokens, 656)
    # Baseline widens each grouped value to BF16 before QK/PV. The independent
    # reference follows that native computation rather than treating FP8 as BF16.
    from b12x.attention._shared.mla.reference import unpack_mla_kv_cache_reference
    values = unpack_mla_kv_cache_reference(cache).squeeze(1).bfloat16().float()
    return cache, values


def capture(fn):
    from b12x._lib.runtime_control import kernel_resolution_guard
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("Kimi MG precompiled graph"), torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def comparison(actual, expected, *, lse=False):
    if lse:
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=.035, atol=.012)
    finite = torch.isfinite(expected)
    return float((actual.float()[finite] - expected.float()[finite]).abs().max()) if finite.any() else 0.


def oracle(args, report):
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_dense_mla_aot
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_sparse_mla_aot
    from b12x.integration.cuteafd.kimi_mla_prefill_mg import compile_kimi_mg_prefill_aot
    results = []
    capacity = 65
    cache, values = records(4096)
    for heads in args.heads:
        g = KimiMlaGeometry(heads=heads, kv_cache="fp8")
        base = compile_kimi_sparse_mla_aot(g, max_rows=capacity, row_chunk_size=64)
        candidate = compile_kimi_mg_prefill_aot(g, max_rows=capacity, max_context=4096,
                                              sparse=True, bf16_compute=args.bf16_compute)
        q = torch.randn(capacity, heads, 576, device="cuda").bfloat16()
        # Every ragged final 64-key tile is covered (0..65, plus distant edges).
        edges = list(range(66)) + [127, 128, 129, 777, 2047, 2048]
        out = torch.full((capacity, heads, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
        lse = torch.full((capacity, heads), float("nan"), device="cuda")
        bo, bl = torch.empty_like(out), torch.empty_like(lse)
        scratch = torch.empty(base.scratch_bytes(capacity)["scratch"], device="cuda", dtype=torch.uint8)
        cu = torch.arange(capacity + 1, device="cuda", dtype=torch.int32)
        max_o, max_l = 0., 0.
        for start in range(0, len(edges), capacity):
            active = edges[start:start + capacity]
            n = len(active)
            lengths = torch.zeros(capacity, device="cuda", dtype=torch.int32)
            lengths[:n] = torch.tensor(active, device="cuda", dtype=torch.int32)
            indices = torch.full((capacity, 2048), -1, device="cuda", dtype=torch.int32)
            for row, length in enumerate(active):
                indices[row, :length] = torch.randperm(4095, device="cuda")[:length].sort().values.int() + 1
            fn = lambda: candidate.launch(q, cache, indices, lengths, out, lse, None, scalars=(n,))
            graph = capture(fn)
            graph.replay()
            base.launch(q, cache, indices, lengths, cu, bo, bl, scratch,
                        scalars=(n, n, 2048, 16))
            torch.cuda.synchronize()
            for row, length in enumerate(active):
                if length:
                    kv = values[indices[row, :length].long()]
                    score = q[row].float() @ kv.T * g.softmax_scale
                    expected = score.softmax(-1) @ kv[:, :512]
                    natural = score.logsumexp(-1)
                    max_o = max(max_o, comparison(out[row], expected))
                    max_l = max(max_l, comparison(lse[row] * torch.log(torch.tensor(2.)), natural, lse=True))
                    comparison(bo[row], expected)
                    comparison(bl[row], natural, lse=True)
                    comparison(out[row], bo[row])
                    comparison(lse[row] * torch.log(torch.tensor(2.)), bl[row], lse=True)
                else:
                    assert torch.count_nonzero(out[row]) == 0
                    assert torch.isneginf(lse[row]).all()
            assert torch.isnan(out[n:]).all() and torch.isnan(lse[n:]).all()
            # Invalid selected slots stage record zero. Poison verifies masking,
            # not merely the benign zero-record case.
            saved_out, saved_lse = out[:n].clone(), lse[:n].clone()
            saved_record = cache[0].clone()
            cache[0] = 255
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(saved_out.view(torch.uint8), out[:n].view(torch.uint8))
            assert torch.equal(saved_lse.view(torch.uint8), lse[:n].view(torch.uint8))
            cache[0] = saved_record
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
        # Identical input at different positions and live widths: no row grouping
        # or live request width may alter that row's numerical state.
        indices = torch.arange(1, 2049, device="cuda", dtype=torch.int32)[None].repeat(capacity, 1)
        lengths = torch.full((capacity,), 2048, device="cuda", dtype=torch.int32)
        q[:] = q[0].clone()
        canonical = None
        for n in (1, 6, 16, 64, 65):
            fn = lambda: candidate.launch(q, cache, indices, lengths, out, lse, None, scalars=(n,))
            graph = capture(fn)
            graph.replay()
            torch.cuda.synchronize()
            now = out[:n].view(torch.uint8), lse[:n].view(torch.uint8)
            if canonical is None:
                canonical = now[0][0].clone(), now[1][0].clone()
            for got, expected in zip(now, canonical):
                assert torch.equal(got, expected[None].expand_as(got))
        results.append(dict(route="sparse", heads=heads, edges=edges, output_max_abs=max_o,
                            lse_max_abs=max_l, row_invariant=True, poison=True, graph=True))
        # Causal dense extend, randomized page table, partial page and future poison.
        dense = compile_kimi_mg_prefill_aot(g, max_rows=capacity, max_context=4096,
                                          bf16_compute=args.bf16_compute)
        dense_base = compile_kimi_dense_mla_aot(g, max_rows=capacity, max_context=4096, mode="extend")
        work = torch.empty(dense.scratch_bytes(capacity)["scratch"], device="cuda", dtype=torch.uint8)
        table = torch.randperm(64, device="cuda").int()
        logical = torch.arange(4096, device="cuda")
        physical = table[logical // 64].long() * 64 + logical % 64
        ordered = torch.empty_like(cache)
        ordered[physical] = cache
        for n, prefix in ((1, 0), (17, 0), (65, 127), (65, 2048)):
            length = torch.tensor([prefix + n], device="cuda", dtype=torch.int32)
            offsets = torch.tensor([0, n], device="cuda", dtype=torch.int32)
            fn = lambda: dense.launch(q, ordered, table, length, out, lse, work, scalars=(n,))
            graph = capture(fn)
            graph.replay()
            dense_base.launch(q, ordered, table, length, offsets, bo, bl, None,
                              scalars=(n, 1, 64, 1))
            torch.cuda.synchronize()
            for row in (0, n - 1):
                kv = values[:prefix + row + 1]
                score = q[row].float() @ kv.T * g.softmax_scale
                comparison(out[row], score.softmax(-1) @ kv[:, :512])
                comparison(lse[row] * torch.log(torch.tensor(2.)), score.logsumexp(-1), lse=True)
                comparison(out[row], bo[row])
                comparison(lse[row] * torch.log(torch.tensor(2.)), bl[row], lse=True)
            before = out[:n].clone(), lse[:n].clone()
            old = ordered[physical[prefix+n:]].clone()
            ordered[physical[prefix+n:]] = 255
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:n].view(torch.uint8), before[0].view(torch.uint8))
            assert torch.equal(lse[:n].view(torch.uint8), before[1].view(torch.uint8))
            ordered[physical[prefix+n:]] = old
        results.append(dict(route="dense", heads=heads, causal=True, poison=True, graph=True))
        print(json.dumps(results[-2:]), flush=True)
    report["oracle"] = results
    return results


def curve(args, report):
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry, compile_kimi_dense_mla_aot
    from b12x.integration.cuteafd.kimi_dsa import compile_kimi_sparse_mla_aot
    from b12x.integration.cuteafd.kimi_mla_prefill_mg import compile_kimi_mg_prefill_aot
    results = report.setdefault("curve", [])
    capacity = max(args.rows)
    for heads in args.heads:
        g = KimiMlaGeometry(heads=heads, kv_cache="fp8")
        for route in args.routes:
            sparse = route == "sparse"
            max_context = max(4096, max(args.prefixes) + capacity)
            mg = compile_kimi_mg_prefill_aot(g, max_rows=capacity, max_context=max_context,
                                           sparse=sparse, bf16_compute=args.bf16_compute)
            base = (compile_kimi_sparse_mla_aot(g, max_rows=capacity, row_chunk_size=64) if sparse else
                    compile_kimi_dense_mla_aot(g, max_rows=capacity, max_context=max_context, mode="extend"))
            cache, _ = records(8192 if sparse else -(-max_context // 64) * 64)
            q = (torch.randn(capacity, heads, 576, device="cuda") * .25).bfloat16()
            out = [torch.empty(capacity, heads, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            lse = [torch.empty(capacity, heads, device="cuda") for _ in range(2)]
            bs = torch.empty(base.scratch_bytes(capacity).get("scratch", 0), device="cuda", dtype=torch.uint8)
            ms = torch.empty(mg.scratch_bytes(capacity)["scratch"], device="cuda", dtype=torch.uint8)
            if sparse:
                table = torch.stack([torch.randperm(8192, device="cuda")[:2048].sort().values.int()
                                     for _ in range(capacity)])
                cu = torch.arange(capacity + 1, device="cuda", dtype=torch.int32)
            else:
                table = torch.randperm(cache.shape[0] // 64, device="cuda").int()
            for rows in args.rows:
                for prefix in ([0] if sparse else args.prefixes):
                    lengths = torch.full((capacity if sparse else 1,), 2048 if sparse else prefix + rows,
                                         device="cuda", dtype=torch.int32)
                    offsets = cu if sparse else torch.tensor([0, rows], device="cuda", dtype=torch.int32)
                    def baseline():
                        base.launch(q, cache, table, lengths, offsets, out[0], lse[0], bs,
                            scalars=(rows, rows if sparse else 1, 2048 if sparse else table.numel(),
                                     16 if sparse else 1))
                    def candidate():
                        mg.launch(q, cache, table, lengths, out[1], lse[1], ms, scalars=(rows,))
                    graphs = [capture(baseline), capture(candidate)]
                    comparison(out[1][:rows], out[0][:rows])
                    comparison(lse[1][:rows] * torch.log(torch.tensor(2.)), lse[0][:rows], lse=True)
                    assert torch.isfinite(out[1][:rows]).all() and out[1][:rows].abs().max() > 0
                    before = torch.cuda.memory_allocated()
                    samples = [[], []]
                    for repeat in range(args.samples):
                        for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            start.record()
                            for _ in range(args.iterations):
                                graphs[arm].replay()
                            end.record()
                            end.synchronize()
                            samples[arm].append(start.elapsed_time(end) / args.iterations)
                    assert torch.cuda.memory_allocated() == before
                    medians = [statistics.median(x) for x in samples]
                    cell = dict(route=route, heads=heads, rows=rows, prefix=prefix,
                        baseline_ms=medians[0], mg_ms=medians[1], speedup=medians[0]/medians[1],
                        raw_ms=samples, baseline_scratch_bytes=bs.numel(), mg_scratch_bytes=ms.numel(),
                        mg_lse_bytes=capacity*heads*4, allocation_stable=True)
                    results.append(cell)
                    print(json.dumps(cell), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, nargs="+", default=[48, 96], choices=[48, 96])
    p.add_argument("--rows", type=int, nargs="+", default=[2048, 2752, 4096, 2673, 65])
    p.add_argument("--prefixes", type=int, nargs="+", default=[0, 8192, 32768, 65536])
    p.add_argument("--routes", nargs="+", default=["sparse", "dense"], choices=["sparse", "dense"])
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--curve-only", action="store_true")
    p.add_argument("--bf16-compute", action="store_true")
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties()
    report = dict(command=sys.argv, device=props.name, uuid=str(props.uuid),
        sm_count=props.multi_processor_count, bf16_compute=args.bf16_compute,
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__), ROOT / "b12x/integration/cuteafd/kimi_mla_prefill_mg.py"]})
    try:
        if not args.curve_only:
            oracle(args, report)
        if not args.oracle_only:
            curve(args, report)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
