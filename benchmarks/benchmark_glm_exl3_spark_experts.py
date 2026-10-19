#!/usr/bin/env python3
"""Cold-L2 SM121 benchmark: GLM 5.3 EXL3 K4/K5 TP6 routed experts (candidate vs master control).

Both arms run on the critical 384-wide TP6 Spark rank slice of GLM 5.3
(hidden=6144, intermediate=384, experts=256, top_k=8, unclamped SwiGLU,
tier0=192 K4 experts, tier1=64 K5 experts) at the Spark worker's capacities
(m1 for 1 row, m80 for 2..80 rows, m256 for 81..256 rows). Every replay cycles
through distinct 256-expert layer copies spanning >= 2x L2 (even at rows=1 where
8 active experts read >= 28.3 MB > L2, two full layer copies stream >= 56.6 MB
> 2x L2). Master (control) and candidate are compiled and timed in the same
session with interleaved rounds, and bitwise equality is verified across every
layer copy and row count before timing.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    bind_mixed_trellis,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    run_bound_mixed_trellis,
)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

HIDDEN = 6144
INTERMEDIATE = 384
EXPERTS = 256
TOPK = 8
TIER0_IDS = tuple(range(0, 192))   # K4 (75% of experts)
TIER1_IDS = tuple(range(192, 256)) # K5 (25% of experts)
TIER0_BITS = 4
TIER1_BITS = 5
DEFAULT_TILE = (64, 128, 64, 128)
BENCH_ROWS = (1, 4, 8, 16, 43, 64, 128)
PACKAGE_CAPACITIES = (1, 80, 256)


def capacity_for_rows(rows: int) -> int:
    for cap in PACKAGE_CAPACITIES:
        if rows <= cap:
            return cap
    return max(PACKAGE_CAPACITIES)


def expert_bytes(bits: int) -> int:
    """Trellis weight bytes of one expert slice (gate, up, down) plus rotation scales."""
    trellis = 3 * HIDDEN * INTERMEDIATE * bits // 8
    rotations = (3 * HIDDEN + 3 * INTERMEDIATE) * 2
    return trellis + rotations


def load_control_modules(control_dir: Path):
    """Load origin/master's w4a16/{route_pack.py, kernel.py, mixed_trellis.py}
    under an isolated subpackage with distinct b12x_compile cache keys."""
    import b12x.moe._shared.kernels.w4a16 as base_pkg
    import b12x.moe._shared.kernels.w4a16.route_pack as cand_rp
    import b12x.moe._shared.kernels.w4a16.kernel as cand_k

    w4a16_dir = control_dir / "w4a16" if (control_dir / "w4a16").is_dir() else control_dir
    pkg_name = "b12x.moe._shared.kernels._ctrl_w4a16"
    if pkg_name in sys.modules and hasattr(sys.modules[pkg_name], "_ctrl_loaded"):
        return sys.modules[f"{pkg_name}.mixed_trellis"], sys.modules[pkg_name]._ctrl_digests

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = list(base_pkg.__path__)
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    for sub in ("host", "ws_a8", "mixed_trellis_ws"):
        orig = importlib.import_module(f"b12x.moe._shared.kernels.w4a16.{sub}")
        sys.modules[f"{pkg_name}.{sub}"] = orig

    digests = []
    try:
        for mod_name, filename in (
            ("route_pack", "route_pack.py"),
            ("kernel", "kernel.py"),
            ("mixed_trellis", "mixed_trellis.py"),
        ):
            path = w4a16_dir / filename
            source = path.read_bytes()
            digests.append(hashlib.sha256(source).hexdigest())
            full_name = f"{pkg_name}.{mod_name}"
            spec = importlib.util.spec_from_file_location(full_name, path)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = pkg_name
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            if mod_name == "route_pack":
                sys.modules["b12x.moe._shared.kernels.w4a16.route_pack"] = module
            elif mod_name == "kernel":
                sys.modules["b12x.moe._shared.kernels.w4a16.kernel"] = module
    finally:
        sys.modules["b12x.moe._shared.kernels.w4a16.route_pack"] = cand_rp
        sys.modules["b12x.moe._shared.kernels.w4a16.kernel"] = cand_k

    ctrl_k = sys.modules[f"{pkg_name}.kernel"]
    ctrl_mt = sys.modules[f"{pkg_name}.mixed_trellis"]
    digest_tuple = tuple(digests)
    orig_compile = ctrl_k.b12x_compile

    def _ctrl_compile(spec, *args, **kwargs):
        spec = dataclasses.replace(spec, cache_key=("ctrl_w4a16", *digest_tuple, *spec.cache_key))
        return orig_compile(spec, *args, **kwargs)

    ctrl_k.b12x_compile = _ctrl_compile
    ctrl_mt.b12x_compile = _ctrl_compile
    pkg._ctrl_loaded = True
    pkg._ctrl_digests = digest_tuple
    return ctrl_mt, digest_tuple


def make_layer(seed: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)).to(torch.float16)

    tiers = []
    for offset, (ids, bits) in enumerate(((TIER0_IDS, TIER0_BITS), (TIER1_IDS, TIER1_BITS))):
        experts = len(ids)
        tiers.append(prepare_trellis256_moe_weights(
            hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_experts=experts,
            activation="silu", fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=seed + offset, params_dtype=torch.float16,
            w13_layout="trellis_t256_proj", trellis_bits=bits, codebook="mcg",
            gate_suh=scales((experts, HIDDEN)), up_suh=scales((experts, HIDDEN)),
            intermediate_rotations=scales((experts, 3 * INTERMEDIATE)),
            down_svh=scales((experts, HIDDEN)), tile_config=DEFAULT_TILE,
        ))
    global_to_combined, descriptor = build_tiered_maps(TIER0_IDS, TIER1_IDS, device=device)
    return tiers[0], tiers[1], global_to_combined, descriptor, combine_trellis_rotations(*tiers)


def make_weight_copies(l2_bytes: int, device: torch.device):
    min_wave_bytes = TOPK * expert_bytes(TIER0_BITS)
    copies = max(2, -(-2 * l2_bytes // min_wave_bytes))
    return [make_layer(1000 * (c + 1), device) for c in range(copies)]


def compile_arm(mod, capacity: int, sms: int, max_shared_mem: int, device: torch.device, layers):
    slots = route_pack_capacity(capacity * TOPK, 8, EXPERTS, topk=TOPK)[1]
    launch = mod.compile_mixed_trellis(
        size_m=capacity, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        tier0_num_experts=len(TIER0_IDS), tier1_num_experts=len(TIER1_IDS), top_k=TOPK,
        route_num_experts=EXPERTS, max_m_blocks=(slots + 7) // 8, sms=sms,
        max_shared_mem=max_shared_mem, force_tile_config=DEFAULT_TILE,
        tier0_bits=TIER0_BITS, tier1_bits=TIER1_BITS, trellis_codebook="mcg",
        swiglu_limit=None, moe_block_size=8, rotation_input_dtype="bf16",
        full_rotation_output_dtype="bf16", route_ids_dtype=torch.int32,
    )
    buffers = mod.make_mixed_trellis_buffers(launch, device=device, sms=sms)
    layer_buffers = [dataclasses.replace(buffers, output=torch.empty_like(buffers.output)) for _ in layers]
    bindings = [mod.bind_mixed_trellis(t0, t1, g2c, desc, rot, launch) for t0, t1, g2c, desc, rot in layers]
    return launch, buffers, layer_buffers, bindings


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
    route_pack_us = sum(
        e["dur"] for e in kernels
        if "pack_topk_routes" in e.get("name", "") or "_w4a16_" in e.get("name", "")
    )
    core_us = sum(e["dur"] for e in kernels if "W4A16MixedTrellis" in e.get("name", ""))
    topk_sum_us = sum(e["dur"] for e in kernels if "W4A16TopKSum" in e.get("name", ""))
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) if kernels else 0.0
    total_k = sum(e["dur"] for e in kernels)
    return {
        "route_pack_us": route_pack_us,
        "core_us": core_us,
        "topk_sum_us": topk_sum_us,
        "kernel_sum_us": total_k,
        "launch_gaps_us": max(0.0, span - total_k),
        "profiled_span_us": span,
        "kernels": len(kernels),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,4,8,16,43,64,128")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved timing rounds; arm order rotates each round, medians reported")
    parser.add_argument("--control-dir", type=Path, required=True,
                        help="directory containing master's w4a16/{route_pack.py,kernel.py,mixed_trellis.py}")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true",
                        help="one launch per arm inside cudaProfilerStart/Stop (for ncu); no timing")
    args = parser.parse_args()
    if not args.profile and args.output is None:
        parser.error("--output is required unless --profile")
    if torch.cuda.get_device_capability() != (12, 1):
        parser.error("this performance gate requires SM121")

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sm_count = int(props.multi_processor_count)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 24 << 20)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    if not args.profile:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    import b12x.moe._shared.kernels.w4a16.mixed_trellis as cand_mod
    ctrl_mod, ctrl_digests = load_control_modules(args.control_dir)

    layers = make_weight_copies(l2_bytes, device)
    num_layers = len(layers)

    row_list = [int(r) for r in args.rows.split(",")]
    needed_caps = sorted({capacity_for_rows(r) for r in row_list})
    cand_arms = {cap: compile_arm(cand_mod, cap, sm_count, max_shared_mem, device, layers) for cap in needed_caps}
    ctrl_arms = {cap: compile_arm(ctrl_mod, cap, sm_count, max_shared_mem, device, layers) for cap in needed_caps}

    tier0_set = set(TIER0_IDS)
    records = []
    for rows in row_list:
        cap = capacity_for_rows(rows)
        cand_launch, _, cand_layer_buffers, cand_bindings = cand_arms[cap]
        ctrl_launch, _, ctrl_layer_buffers, ctrl_bindings = ctrl_arms[cap]

        layer_inputs = []
        active_experts_per_layer = []
        wave_bytes_per_layer = []
        for idx in range(num_layers):
            gen = torch.Generator(device=device).manual_seed(6000 + rows * 37 + idx)
            x = (torch.randn((rows, HIDDEN), device=device, generator=gen) * 0.25).bfloat16().contiguous()
            scores = torch.rand((rows, EXPERTS), device=device, generator=gen)
            ids = scores.topk(TOPK, -1).indices.to(torch.int32).contiguous()
            weights = torch.rand((rows, TOPK), device=device, generator=gen).contiguous()
            weights = (2.5 * weights / weights.sum(-1, keepdim=True)).to(torch.float32).contiguous()
            distinct = ids.unique().tolist()
            active_experts_per_layer.append(len(distinct))
            wave_bytes_per_layer.append(sum(expert_bytes(TIER0_BITS if e in tier0_set else TIER1_BITS) for e in distinct))
            layer_inputs.append((x, weights, ids))

        # Verify bitwise identity on every layer copy before timing.
        for idx in range(num_layers):
            x, weights, ids = layer_inputs[idx]
            cand_layer_buffers[idx].output.fill_(float("nan"))
            ctrl_layer_buffers[idx].output.fill_(float("nan"))
            cand_out = run_bound_mixed_trellis(x, weights, ids, cand_bindings[idx], cand_layer_buffers[idx])
            ctrl_out = ctrl_mod.run_bound_mixed_trellis(x, weights, ids, ctrl_bindings[idx], ctrl_layer_buffers[idx])
            torch.cuda.synchronize()
            if not torch.equal(cand_out.view(torch.int16), ctrl_out.view(torch.int16)):
                diff = int((cand_out.view(torch.int16) != ctrl_out.view(torch.int16)).sum())
                raise RuntimeError(f"GLM EXL3 bitwise mismatch at rows={rows} layer={idx}: {diff} elements differ")

        def run_candidate():
            for idx in range(num_layers):
                x, weights, ids = layer_inputs[idx]
                run_bound_mixed_trellis(x, weights, ids, cand_bindings[idx], cand_layer_buffers[idx])

        def run_control():
            for idx in range(num_layers):
                x, weights, ids = layer_inputs[idx]
                ctrl_mod.run_bound_mixed_trellis(x, weights, ids, ctrl_bindings[idx], ctrl_layer_buffers[idx])

        if args.profile:
            for _ in range(2):
                run_control()
                run_candidate()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            x, weights, ids = layer_inputs[0]
            ctrl_mod.run_bound_mixed_trellis(x, weights, ids, ctrl_bindings[0], ctrl_layer_buffers[0])
            run_bound_mixed_trellis(x, weights, ids, cand_bindings[0], cand_layer_buffers[0])
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"profiled rows={rows} cap={cap}", flush=True)
            continue

        cand_graph = capture(run_candidate)
        ctrl_graph = capture(run_control)
        for idx in range(num_layers):
            cand_layer_buffers[idx].output.fill_(float("nan"))
            ctrl_layer_buffers[idx].output.fill_(float("nan"))
        cand_graph.replay()
        ctrl_graph.replay()
        torch.cuda.synchronize()
        for idx in range(num_layers):
            c_out = cand_layer_buffers[idx].output[:rows]
            m_out = ctrl_layer_buffers[idx].output[:rows]
            if not torch.equal(c_out.view(torch.int16), m_out.view(torch.int16)):
                raise RuntimeError(f"GLM EXL3 graph replay bitwise mismatch at rows={rows} layer={idx}")

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
        wave_weight_bytes = int(round(sum(wave_bytes_per_layer) / num_layers))
        replay_streamed_bytes = sum(wave_bytes_per_layer)

        cand_stages = {
            k: (v / num_layers if k.endswith("_us") else v // num_layers)
            for k, v in stage_profile(cand_graph, args.output.parent).items()
        }
        ctrl_stages = {
            k: (v / num_layers if k.endswith("_us") else v // num_layers)
            for k, v in stage_profile(ctrl_graph, args.output.parent).items()
        }

        rec = {
            "rows": rows,
            "capacity": cap,
            "avg_active_experts": round(avg_active_experts, 1),
            "wave_weight_mb": round(wave_weight_bytes / (1024 * 1024), 2),
            "replay_streamed_mb": round(replay_streamed_bytes / (1024 * 1024), 2),
            "replay_over_l2": round(replay_streamed_bytes / l2_bytes, 2),
            "control_us": round(ctrl_us, 2),
            "candidate_us": round(cand_us, 2),
            "control_rounds_us": [round(v, 2) for v in rounds["control"]],
            "candidate_rounds_us": [round(v, 2) for v in rounds["candidate"]],
            "speedup": round(speedup, 4),
            "reduction_pct": round(reduction_pct, 2),
            "bitwise_equal": True,
            "control_blocks_per_sm": ctrl_launch.blocks_per_sm,
            "candidate_blocks_per_sm": cand_launch.blocks_per_sm,
            "control_stages": {k: round(v, 2) if isinstance(v, float) else v for k, v in ctrl_stages.items()},
            "candidate_stages": {k: round(v, 2) if isinstance(v, float) else v for k, v in cand_stages.items()},
        }
        records.append(rec)
        print(
            f"rows={rows:3d} cap={cap:3d} experts={avg_active_experts:5.1f} "
            f"ctrl={ctrl_us:7.2f}us (b/sm={ctrl_launch.blocks_per_sm}) "
            f"cand={cand_us:7.2f}us (b/sm={cand_launch.blocks_per_sm}) "
            f"diff={reduction_pct:+6.2f}% ({speedup:.3f}x) "
            f"[pack {ctrl_stages['route_pack_us']:.1f}->{cand_stages['route_pack_us']:.1f} "
            f"core {ctrl_stages['core_us']:.1f}->{cand_stages['core_us']:.1f} "
            f"sum {ctrl_stages['topk_sum_us']:.1f}->{cand_stages['topk_sum_us']:.1f}]",
            flush=True,
        )

    if args.profile:
        return

    payload = {
        "device": props.name,
        "sm_count": sm_count,
        "l2_bytes": l2_bytes,
        "num_cold_l2_layers": num_layers,
        "geometry": {
            "name": "glm",
            "hidden": HIDDEN,
            "tp": 6,
            "slice": INTERMEDIATE,
            "experts": EXPERTS,
            "top_k": TOPK,
            "tier0_bits": TIER0_BITS,
            "tier1_bits": TIER1_BITS,
            "tile": list(DEFAULT_TILE),
        },
        "control_sha256": list(ctrl_digests),
        "replays": args.replays,
        "repeats": args.repeats,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
