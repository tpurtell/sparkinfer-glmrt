#!/usr/bin/env python3
"""Cold-L2 SM121 benchmark: MiMo V2.6 Pro TP6 MXFP4 routed experts (candidate vs master control).

Both arms start from the same FP8 K32 wire rows (E4M3 + UE8M0 per 32) and end
with BF16 response rows on the TP6 slice (hidden=6144, slice=384, experts=384,
top_k=8). Every replay streams distinct weight copies spanning >= 2x L2 (even
at rows=1 where 8 active experts read 30.1 MB > L2, two full 1.44 GB layer
copies stream >= 60.2 MB > 2x L2). Master (control) and candidate are compiled
and timed in the same session with interleaved rounds, and bitwise equality is
checked at every row count before timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import types

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

BENCH_ROWS = (1, 4, 8, 16, 43, 64, 128)
PACKAGE_CAPACITIES = (1, 16, 80, 256)

STAGES = {
    "prep": ("MoePrep",),
    "gather": ("GatherRows",),
    "gate_up_swiglu": ("GroupedMxfp4Gemv",),
    "swiglu": ("MoeSwiGLU",),
    "combine": ("MoeCombine",),
}


def capacity_for_rows(rows: int) -> int:
    for cap in PACKAGE_CAPACITIES:
        if rows <= cap:
            return cap
    return max(PACKAGE_CAPACITIES)


def make_wire(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, h = x.shape
    blocks = x.float().reshape(rows, h // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    e4m3 = (blocks / scale).to(torch.float8_e4m3fn)
    wire = torch.empty((rows, h + h // 32), dtype=torch.uint8, device=x.device)
    wire[:, :h] = e4m3.reshape(rows, h).view(torch.uint8)
    wire[:, h:] = (exponent.reshape(rows, h // 32) + 127).to(torch.uint8)
    exact = (e4m3.float() * scale).reshape(rows, h).bfloat16()
    return wire, exact


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def timing(graph, replays: int) -> float:
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / replays)
    return statistics.median(values)


def stage_profile(graph, directory: Path) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", dir=directory) as stream:
        prof.export_chrome_trace(stream.name)
        events = json.loads(Path(stream.name).read_text())["traceEvents"]
    kernels = sorted([e for e in events if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    prep_us = sum(e["dur"] for e in kernels if "MoePrep" in e.get("name", ""))
    gather_us = sum(e["dur"] for e in kernels if "GatherRows" in e.get("name", ""))
    swiglu_us = sum(e["dur"] for e in kernels if "MoeSwiGLU" in e.get("name", ""))
    combine_us = sum(e["dur"] for e in kernels if "MoeCombine" in e.get("name", ""))
    gemvs = [e for e in kernels if "GroupedMxfp4Gemv" in e.get("name", "")]
    gate_up_us = sum(e["dur"] for idx, e in enumerate(gemvs) if idx % 2 == 0)
    down_us = sum(e["dur"] for idx, e in enumerate(gemvs) if idx % 2 == 1)
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) if kernels else 0.0
    total_k = sum(e["dur"] for e in kernels)
    return {
        "prep_us": prep_us,
        "gather_us": gather_us,
        "gate_up_us": gate_up_us,
        "swiglu_us": swiglu_us,
        "down_us": down_us,
        "combine_us": combine_us,
        "kernel_sum_us": total_k,
        "launch_gaps_us": max(0.0, span - total_k),
        "profiled_span_us": span,
        "kernels": len(kernels),
    }


def load_control_modules(control_dir: Path):
    """Load origin/master's _fp8_moe_kernels, _mxfp4_moe_kernels, and fp8_moe
    from snapshot files under an isolated subpackage so both arms coexist."""
    import b12x.integration.cuteafd as base_pkg

    pkg_name = "b12x.integration.cuteafd._ctrl_master"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = list(base_pkg.__path__)
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        for sub in ("_common", "_fp8_weights", "_mxfp4_down_a8_plan", "_fp8_moe_stream",
                    "_mxfp4_moe_stream", "_mxfp4_moe_a8", "_nvfp4_moe_kernels", "_nvfp4_moe_a4"):
            orig = importlib.import_module(f"b12x.integration.cuteafd.{sub}")
            sys.modules[f"{pkg_name}.{sub}"] = orig

    digests = []
    for mod_name, filename in (
        ("_fp8_moe_kernels", "_fp8_moe_kernels.py"),
        ("_mxfp4_moe_kernels", "_mxfp4_moe_kernels.py"),
        ("fp8_moe", "fp8_moe.py"),
    ):
        path = control_dir / filename
        source = path.read_bytes()
        digests.append(hashlib.sha256(source).hexdigest())
        full_name = f"{pkg_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = pkg_name
        sys.modules[full_name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
    ctrl_fp8_moe = sys.modules[f"{pkg_name}.fp8_moe"]
    return ctrl_fp8_moe, tuple(digests)


def make_weight_copies(g, l2_bytes: int, real: int = 352) -> list[tuple[torch.Tensor, ...]]:
    """Allocate distinct MXFP4 layer copies so every replay streams >= 2x L2."""
    h, i, e = g.hidden, g.slice, g.experts
    # Even at rows=1 (top_k=8 experts), one wave reads 8 * 3.76 MB = 30.1 MB.
    min_wave_bytes = g.top_k * (2 * i * (h // 2 + h // 32) + h * (i // 2 + i // 32))
    copies = max(2, -(-2 * l2_bytes // min_wave_bytes))
    layers = []
    for c in range(copies):
        gen = torch.Generator(device="cuda").manual_seed(1000 + c)
        w1 = torch.randint(0, 256, (e, i, h // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
        s1 = torch.randint(118, 124, (e, i, h // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
        w3 = torch.randint(0, 256, (e, i, h // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
        s3 = torch.randint(118, 124, (e, i, h // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
        w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
        s2 = torch.randint(118, 124, (e, h, i // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
        for w in (w1, s1, w3, s3):
            w[:, real:] = 0
        w2[:, :, real // 2:] = 0
        s2[:, :, real // 32:] = 0
        layers.append((w1, s1, w3, s3, w2, s2))
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,4,8,16,43,64,128")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved timing rounds; arm order rotates each round, medians reported")
    parser.add_argument("--control-dir", type=Path, required=True,
                        help="directory containing master's _fp8_moe_kernels.py, _mxfp4_moe_kernels.py, fp8_moe.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true",
                        help="one launch per arm inside cudaProfilerStart/Stop (for ncu); no timing")
    args = parser.parse_args()
    if not args.profile and args.output is None:
        parser.error("--output is required unless --profile")
    if torch.cuda.get_device_capability() != (12, 1):
        parser.error("this performance gate requires SM121")

    props = torch.cuda.get_device_properties(0)
    sm_count = int(props.multi_processor_count)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 24 << 20)
    if not args.profile:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    g = GEOMETRIES["mimop"].with_tp(6)
    ctrl_mod, ctrl_digests = load_control_modules(args.control_dir)
    ctrl_g = ctrl_mod.GEOMETRIES["mimop"].with_tp(6)

    row_list = [int(r) for r in args.rows.split(",")]
    needed_caps = sorted({capacity_for_rows(r) for r in row_list})
    cand_progs = {cap: compile_fp8_moe_aot(g, route="auto", max_rows=cap, wire=True) for cap in needed_caps}
    ctrl_progs = {cap: ctrl_mod.compile_fp8_moe_aot(ctrl_g, route="auto", max_rows=cap, wire=True)
                  for cap in needed_caps}

    layers = make_weight_copies(g, l2_bytes)
    num_layers = len(layers)
    expert_bytes = 2 * g.slice * (g.hidden // 2 + g.hidden // 32) + g.hidden * (g.slice // 2 + g.slice // 32)

    records = []
    for rows in row_list:
        cap = capacity_for_rows(rows)
        cand_prog = cand_progs[cap]
        ctrl_prog = ctrl_progs[cap]
        cand_scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", cap, True), dtype=torch.uint8, device="cuda")
        ctrl_scratch = torch.empty(ctrl_mod.fp8_moe_scratch_bytes(ctrl_g, "auto", cap, True),
                                   dtype=torch.uint8, device="cuda")

        # Distinct inputs and routing per layer copy so each layer streams distinct experts.
        layer_inputs = []
        active_experts_per_layer = []
        for idx in range(num_layers):
            gen = torch.Generator(device="cuda").manual_seed(5000 + rows * 31 + idx)
            x = torch.randn((rows, g.hidden), device="cuda", generator=gen).bfloat16()
            wire, _ = make_wire(x)
            scores = torch.rand((rows, g.experts), device="cuda", generator=gen)
            ids = scores.topk(g.top_k, -1).indices.int().contiguous()
            weights = torch.rand((rows, g.top_k), device="cuda", generator=gen).contiguous()
            weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
            active_experts_per_layer.append(int(ids.unique().numel()))
            layer_inputs.append((wire, ids, weights))

        cand_out = torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda")
        ctrl_out = torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda")

        # Verify bitwise identity on every layer copy before benchmarking.
        for idx in range(num_layers):
            wire, ids, weights = layer_inputs[idx]
            w = layers[idx]
            cand_out.fill_(float("nan"))
            ctrl_out.fill_(float("nan"))
            cand_prog.launch(wire, ids, weights, *w, cand_out, cand_scratch, scalars=(rows,))
            ctrl_prog.launch(wire, ids, weights, *w, ctrl_out, ctrl_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            if not torch.equal(cand_out.view(torch.int16), ctrl_out.view(torch.int16)):
                diff = int((cand_out.view(torch.int16) != ctrl_out.view(torch.int16)).sum())
                raise RuntimeError(f"Bitwise mismatch at rows={rows} layer={idx}: {diff} elements differ")

        def run_candidate():
            for idx in range(num_layers):
                wire, ids, weights = layer_inputs[idx]
                cand_prog.launch(wire, ids, weights, *layers[idx], cand_out, cand_scratch, scalars=(rows,))

        def run_control():
            for idx in range(num_layers):
                wire, ids, weights = layer_inputs[idx]
                ctrl_prog.launch(wire, ids, weights, *layers[idx], ctrl_out, ctrl_scratch, scalars=(rows,))

        if args.profile:
            for _ in range(2):
                run_control()
                run_candidate()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            wire, ids, weights = layer_inputs[0]
            ctrl_prog.launch(wire, ids, weights, *layers[0], ctrl_out, ctrl_scratch, scalars=(rows,))
            cand_prog.launch(wire, ids, weights, *layers[0], cand_out, cand_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"profiled rows={rows} cap={cap}", flush=True)
            continue

        cand_graph = capture(run_candidate)
        ctrl_graph = capture(run_control)
        cand_graph.replay()
        ctrl_graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(cand_out.view(torch.int16), ctrl_out.view(torch.int16)):
            raise RuntimeError(f"Graph replay bitwise mismatch at rows={rows}")

        arms = {"control": ctrl_graph, "candidate": cand_graph}
        names = list(arms)
        rounds = {n: [] for n in names}
        for r in range(args.repeats):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                rounds[n].append(timing(arms[n], args.replays) / num_layers)
        med = {n: statistics.median(v) for n, v in rounds.items()}
        ctrl_us = med["control"]
        cand_us = med["candidate"]
        reduction_pct = 100.0 * (ctrl_us - cand_us) / ctrl_us
        speedup = ctrl_us / cand_us
        avg_active_experts = sum(active_experts_per_layer) / num_layers
        wave_weight_bytes = int(round(avg_active_experts * expert_bytes))
        replay_streamed_bytes = sum(e * expert_bytes for e in active_experts_per_layer)

        cand_stages = {
            k: (v / num_layers if k.endswith("_us") else v // num_layers)
            for k, v in stage_profile(cand_graph, args.output.parent).items()
        }
        ctrl_stages = {
            k: (v / num_layers if k.endswith("_us") else v // num_layers)
            for k, v in stage_profile(ctrl_graph, args.output.parent).items()
        }

        record = {
            "rows": rows,
            "capacity": cap,
            "layer_copies": num_layers,
            "avg_active_experts": avg_active_experts,
            "wave_weight_bytes": wave_weight_bytes,
            "replay_streamed_bytes": replay_streamed_bytes,
            "replay_over_l2": replay_streamed_bytes / l2_bytes,
            "control_us": ctrl_us,
            "candidate_us": cand_us,
            "reduction_percent": reduction_pct,
            "speedup": speedup,
            "control_gbps": wave_weight_bytes / ctrl_us / 1000.0,
            "candidate_gbps": wave_weight_bytes / cand_us / 1000.0,
            "bitwise_equal": True,
            "rounds_us": rounds,
            "control_stages": ctrl_stages,
            "candidate_stages": cand_stages,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    if args.profile:
        return

    result = {
        "schema": 1,
        "kind": "mxfp4_spark_experts_tp6_ab",
        "device": props.name,
        "sm_count": sm_count,
        "l2_bytes": l2_bytes,
        "geometry": {
            "name": g.name,
            "hidden": g.hidden,
            "slice": g.slice,
            "real_slice": 352,
            "experts": g.experts,
            "top_k": g.top_k,
            "tp": g.tp,
        },
        "control_sha256": list(ctrl_digests),
        "conditions": (
            "MiMo V2.6 Pro TP6 MXFP4 routed experts, cold L2 (distinct weight copies >= 2x L2 per replay), "
            "CUDA graph replay, interleaved master (control) vs candidate rounds in the same session"
        ),
        "rows": records,
    }
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
