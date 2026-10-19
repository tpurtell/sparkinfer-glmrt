#!/usr/bin/env python3
"""Same-session, cold-L2 MXFP4 decode comparison against a frozen core module.

Routes come from checkpoint router weights on generated BF16 inputs, not from
serving traffic. Expert weights are format-valid deterministic synthetic bytes.
No model-quality or end-to-end throughput claim follows from this benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch


def control_class(path):
    name = "b12x.integration.cuteafd._mxfp4_decode_control"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    class Control(module.GroupedMxfp4Gemv):
        def key(self):
            return ("frozen_control", digest, super().key())

    return Control


def compile_arm(g, capacity, core):
    from b12x.integration.cuteafd import fp8_moe

    previous = fp8_moe.GroupedMxfp4Gemv
    try:
        fp8_moe.GroupedMxfp4Gemv = core
        return fp8_moe.compile_fp8_moe_aot(g, route="decode", max_rows=capacity)
    finally:
        fp8_moe.GroupedMxfp4Gemv = previous


def packed_weights(g, generator):
    h, i, e = g.hidden, g.slice, g.experts
    tensors = []
    for shape in ((e, i, h // 2), (e, i, h // 32), (e, i, h // 2),
                  (e, i, h // 32), (e, h, i // 2), (e, h, i // 32)):
        scale = len(tensors) % 2
        tensors.append(torch.randint(118 if scale else 0, 124 if scale else 256,
                                     shape, device="cuda", dtype=torch.uint8, generator=generator))
    return tuple(tensors)


def router_rows(snapshot, layer, x):
    from safetensors import safe_open

    config = json.loads((snapshot / "config.json").read_text())
    assert config["n_group"] == config["topk_group"] == 1
    assert config["scoring_func"] == "sigmoid" and config["topk_method"] == "noaux_tc"
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{layer}.mlp.gate."
    tensors = []
    for suffix in ("weight", "e_score_correction_bias"):
        name = prefix + suffix
        with safe_open(snapshot / index[name], framework="pt", device="cpu") as reader:
            tensors.append(reader.get_tensor(name).cuda().float())
    scores = (x.float() @ tensors[0].T).sigmoid()
    ids = (scores + tensors[1]).topk(config["num_experts_per_tok"], dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if config["norm_topk_prob"]:
        weights /= weights.sum(-1, keepdim=True) + 1e-20
    weights *= config.get("routed_scaling_factor") or 1.0
    return ids.int().contiguous(), weights.contiguous()


def run_pair(args):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, fp8_moe_scratch_bytes
    from b12x.integration.cuteafd._mxfp4_moe_kernels import GroupedMxfp4Gemv
    from b12x._lib.utils import get_hardware_info
    from bench_cuteafd_mxfp4_down_a8 import wire_rows

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.10)
    g = GEOMETRIES["mimop"].with_tp(6)
    generator = torch.Generator(device="cuda").manual_seed(500)
    widths = [int(value) for value in args.rows.split(",")]
    if not widths or min(widths) < 1 or max(widths) > 128:
        raise ValueError("rows must be in 1..128")
    capacity = 128
    x = torch.randn(capacity, g.hidden, device="cuda", generator=generator).bfloat16()
    source, _ = wire_rows(x)
    ids, mix = router_rows(args.snapshot, args.layer, x)
    weights = packed_weights(g, generator)
    cores = {"master": control_class(args.control_core), "candidate": GroupedMxfp4Gemv}
    capacities = {rows: next(cap for cap in (1, 16, 80, 256) if cap >= rows) for rows in widths}
    programs = {(arm, cap): compile_arm(g, cap, core)
                for cap in sorted(set(capacities.values())) for arm, core in cores.items()}
    buffers = {arm: (torch.empty_like(x), torch.empty(fp8_moe_scratch_bytes(g, "decode", capacity),
                                                    device="cuda", dtype=torch.uint8)) for arm in cores}
    l2_bytes = int(get_hardware_info().get_l2_cache_size_in_bytes())
    assert l2_bytes > 0
    flush = torch.zeros(4 * l2_bytes, device="cuda", dtype=torch.uint8)
    result = {"control_sha256": hashlib.sha256(args.control_core.read_bytes()).hexdigest(),
              "routing": "checkpoint-router-derived; generated BF16 inputs; not observed serving traffic",
              "layer": args.layer, "geometry": {"hidden": g.hidden, "slice": g.slice,
              "experts": g.experts, "top_k": g.top_k}, "capacity": capacity,
              "device": torch.cuda.get_device_name(),
              "sms": torch.cuda.get_device_properties(0).multi_processor_count,
              "l2_bytes": l2_bytes, "rows": []}
    for rows in widths:
        def launch(arm):
            out, scratch = buffers[arm]
            programs[arm, capacities[rows]].launch(source, ids, mix, *weights, out, scratch, scalars=(rows,))

        graphs = {}
        for arm in cores:
            launch(arm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(arm)
            graph.replay()
            graphs[arm] = graph
        torch.cuda.synchronize()
        a, b = (buffers[arm][0][:rows] for arm in cores)
        assert torch.isfinite(a).all() and a.count_nonzero(), "invalid baseline output"
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"master bytes differ at {rows} rows"
        if args.profile_only:
            torch.cuda.cudart().cudaProfilerStart()
            launch(args.profile_arm)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            continue
        times = {arm: [] for arm in cores}
        for repeat in range(args.iters):
            order = tuple(cores) if repeat % 2 == 0 else tuple(reversed(cores))
            for arm in order:
                flush.add_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                times[arm].append(start.elapsed_time(end) * 1000)
        record = {"rows": rows, "capacity": capacities[rows], "unique_experts": ids[:rows].unique().numel(),
                  "max_rows_per_expert": int(torch.bincount(ids[:rows].flatten().long()).max()),
                  "exact": True, "samples_us": times}
        record.update({arm + "_us": statistics.median(values) for arm, values in times.items()})
        record["speedup"] = record["master_us"] / record["candidate_us"]
        result["rows"].append(record)
        print(json.dumps(record), flush=True)
        del graphs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-core", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--rows", default="1,4,8,16,43,64,128")
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-arm", choices=("master", "candidate"), default="master")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.iters <= 50:
        parser.error("iters must be in 1..50")
    run_pair(args)


if __name__ == "__main__":
    main()
