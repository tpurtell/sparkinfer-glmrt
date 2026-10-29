#!/usr/bin/env python3
"""Frozen Kimi RTX duel adapter for benchmark_mixed_trellis (no kernel code)."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

PIN = "26a1e5072fb707864845b88db65ddf1e80c5c872"
CAPACITIES = (1024, 2048, 2752, 4096)
PROFILES = {"rtx-tp2": 1536, "rtx-tp1": 3072}
LAYERS = (13, 92)
FROZEN_FILES = ("benchmarks/bench_kimi_ws_prefill.py",
                "benchmarks/benchmark_mixed_trellis.py",
                "benchmarks/kimi_trellis_stage_frozen.py",
                "benchmarks/benchmark_exlr8_a16.py",
                "tests/test_kimi_ws_prefill_contract.py")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def oracle_arm(variant):
    return dict(mode="ws" if variant.get("warp_specialized") else "cooperative",
                rotation="token" if variant.get("token_major_rotation") else "route",
                route_block=variant.get("route_block", 8), tile=tuple(variant["tile"]),
                ctas=variant["blocks_per_sm"], dynamic=int(variant.get("ws_dynamic_tiles", False)))


def cell_key(profile, capacity, layer, distribution):
    return f"{profile}/m{capacity}/layer{layer}/{distribution}"


def probe_rows(capacity):
    return sorted(set(range(1, 129)) | {capacity // 2 - 1, capacity // 2,
                                      capacity // 2 + 1, capacity - 1})


def validate_manifest(manifest):
    if manifest["sparkinfer_revision"] != PIN or manifest["compute"] != [12, 0]:
        raise ValueError("oracle package pin/architecture mismatch")
    cells = {}
    for profile, width in PROFILES.items():
        for capacity in CAPACITIES:
            name = f"{profile}/m{capacity}"
            variant, = [v for v in manifest["variants"] if v["directory"] == name]
            if (variant["intermediate"], variant["experts"], variant["top_k"], variant["bits"]) != (width, 896, 16, [2, 3]):
                raise ValueError("wrong Kimi geometry")
            cells[name] = oracle_arm(variant)
    return cells


def check_frozen(source, duel):
    record = load(duel / "oracle/ORACLE.json")
    for name, expected in record["harness"].items():
        if sha(source / name) != expected:
            raise ValueError(f"frozen harness differs: {name}")
    for name, expected in record["data"].items():
        if sha(duel / name) != expected:
            raise ValueError(f"frozen data differs: {name}")
    if source.resolve() == (duel / "oracle/oracle-tree").resolve():
        for name, expected in record["kernel_sources"].items():
            if sha(source / name) != expected:
                raise ValueError(f"oracle source differs: {name}")
    return record


def prepare_layer(stage, layer, width, device):
    import torch
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import (ProjectionTrellisTierWeights,
                                           prepare_projection_native_trellis_weights)
    from benchmarks import kimi_trellis_stage_frozen as S
    with safe_open(str(stage / f"layer{layer}.safetensors"), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    full = 3072
    tensors["w13"] = tensors["w13"][:, :, :width // 16].contiguous()
    tensors["w2"] = tensors["w2"][:, :width // 16].contiguous()
    tensors["intermediate_rotations"] = torch.cat([
        tensors["intermediate_rotations"][:, n * full:n * full + width]
        for n in range(3)], dim=1).contiguous()
    tensors = {k: v.to(device) for k, v in tensors.items()}
    members = tuple(range(896))
    tier = ProjectionTrellisTierWeights(2, tensors["w13"], tensors["w2"], members, members, members)
    empty = ProjectionTrellisTierWeights(3,
        torch.empty((0, 224, width // 16, 48), dtype=torch.int16, device=device),
        torch.empty((0, width // 16, 224, 48), dtype=torch.int16, device=device), (), (), ())
    prepared = prepare_projection_native_trellis_weights((tier, empty),
        gate_suh=tensors["gate_suh"], up_suh=tensors["up_suh"],
        intermediate_rotations=tensors["intermediate_rotations"], down_svh=tensors["down_svh"],
        activation="situ", params_dtype=torch.bfloat16, num_experts=896,
        hidden_size=3584, intermediate_size=width)
    tiers = []
    for bits, t in zip((2, 3), prepared.tiers):
        updates = {}
        if t.w13.numel() == 0:
            updates["w13"] = torch.zeros(S.empty_tier_plane_elements(bits, 3584, width), dtype=torch.int32, device=device)
        if t.w2.numel() == 0:
            updates["w2"] = torch.zeros(S.empty_tier_plane_elements(bits, 3584, width), dtype=torch.int32, device=device)
            updates["w2_global_scale"] = torch.ones(1, dtype=torch.float32, device=device)
        tiers.append(replace(t, **updates))
    return (*tiers, prepared.global_to_combined, prepared.descriptor_map, prepared.rotations)


def validate_baseline_write(record, baseline, config):
    if baseline and (config or "baselines/rtx/digests.json" in record["data"]):
        raise ValueError("baseline writes require an unsealed oracle and no candidate config")


def run(args):
    source, duel = args.source.resolve(), args.duel.resolve()
    record = check_frozen(source, duel)
    validate_baseline_write(record, args.baseline, args.config)
    if args.baseline or not args.config:
        for name, expected in record["kernel_sources"].items():
            if sha(source / name) != expected:
                raise ValueError(f"oracle source differs: {name}")
    sys.path.insert(0, str(source))
    import torch
    from unittest.mock import patch
    from benchmarks import benchmark_mixed_trellis as B
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mix
    from benchmarks import kimi_trellis_stage_frozen as S
    fixtures = load(duel / "fixtures/fixtures.json")
    stage = load(duel / "stage/manifest.json")
    S.verify_stage_manifest(duel / "stage", stage)
    cells = validate_manifest(load(duel / "provenance/coordinator-manifest.json"))
    config = load(args.config) if args.config else {}
    if set(config) - set(cells):
        raise ValueError("candidate config names an unknown cell")
    frozen = load(duel / "baselines/rtx/digests.json") if not args.baseline else None
    compile_original, bind_original = B.coordinator_compile_options, mix.bind_mixed_trellis
    digests, outputs = {}, []
    for profile in args.profiles.split(","):
        if profile not in PROFILES:
            raise ValueError("unknown RTX profile")
        layer_index = [0]
        def staged_layer(width, count, seed, device, family):
            layer = LAYERS[layer_index[0]]
            layer_index[0] += 1
            return prepare_layer(duel / "stage", layer, width, device)
        def compile_options(profile, capacity, arm, props, family="kimik3", compact=False):
            key = f"{profile}/m{capacity}"
            selected = config.get(key, cells[key])
            if args.config and selected.get("mode") != "ws":
                raise ValueError("all candidate cells must use warp specialization")
            return compile_original(profile, capacity, selected, props, family, compact)
        def bind(*values, **kwargs):
            kwargs.update(gate_experts=(896, 0), up_experts=(896, 0))
            return bind_original(*values, **kwargs)
        parsed = B.benchmark_parser().parse_args([
            "--prefill-family", "kimik3", "--role", "coordinator", "--profile", profile,
            "--capacities", args.capacities, "--layers", "2", "--rounds", str(args.rounds),
            "--replays", str(args.replays), "--warmup", "2", "--memory-cap-gib", "64",
            "--seed", str(fixtures["seed"]), "--route-distributions", "random,realistic",
            "--routes", str(duel / "fixtures/hot-routes.json"), "--coordinator-route-layer", "13",
            "--output", str(args.out.with_name(args.out.stem + "-" + profile + ".json"))])
        parsed.duel_probe_rows = probe_rows
        if args.check_only:
            parsed.correctness_only = True
        with patch.object(B, "_make_coordinator_layer", staged_layer), \
             patch.object(B, "coordinator_compile_options", compile_options), \
             patch.object(mix, "bind_mixed_trellis", bind):
            B._run_coordinator_benchmark(parsed, torch.device("cuda", 0), torch.cuda.get_device_properties(0))
        result = load(parsed.output)
        result["conditions"] = result["conditions"].replace("synthetic projection-major weights", "real Kimi layers 13/92; frozen full-width stage sliced on CPU")
        for row in result["rows"]:
            key = cell_key(profile, row["capacity"], LAYERS[row["weight_layer"]], row["distribution"])
            prefix = {k: v["coop"]["sha256"] for k, v in row["prefix"].items()}
            digests[key] = {"full": row["output_sha256"], "prefix": prefix}
            if not all(v["coop"]["finite"] and v["coop"]["max_abs_gap"] == 0
                       and v["coop"]["output_tail_untouched"] for v in row["prefix"].values()):
                raise ValueError(f"row invariance/tail guard failed: {key}")
            if frozen is not None and digests[key] != frozen[key]:
                raise ValueError(f"frozen oracle digest mismatch: {key}")
        outputs.append(result)
    telemetry = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,power.draw,power.limit,clocks.sm,temperature.gpu",
                                "--format=csv"], capture_output=True, text=True, timeout=15)
    report = dict(schema="cuteafd.kimi-ws-prefill.v1", exact=True, results=outputs,
                  telemetry=telemetry.stdout, telemetry_stderr=telemetry.stderr,
                  frozen_oracle=record["fork_commit"], config=config,
                  oracle_record_sha256=sha(duel / "oracle/ORACLE.json"),
                  source=str(source),
                  digests=digests, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(0))
    save(args.out, report)
    if args.baseline:
        save(duel / "baselines/rtx/digests.json", digests)


def validate_session(spec):
    names = [a["name"] for a in spec["arms"]]
    if len(names) != len(set(names)) or "oracle" not in names:
        raise ValueError("session needs unique names and oracle")
    oracle, = [a for a in spec["arms"] if a["name"] == "oracle"]
    if oracle.get("config") or Path(oracle["source"]).resolve() != (Path(spec["duel"]) / "oracle/oracle-tree").resolve():
        raise ValueError("session oracle must be the frozen config-less pin tree")
    if any(a["name"] != "oracle" and not a.get("config") for a in spec["arms"]):
        raise ValueError("every entrant must declare its WS config")
    return names


def session(args):
    spec = load(args.spec)
    names = validate_session(spec)
    records = []
    for round_index in range(spec.get("rounds", 3)):
        arms = spec["arms"][round_index % len(names):] + spec["arms"][:round_index % len(names)]
        for arm in arms:
            out = args.out.parent / f"{args.out.stem}-r{round_index}-{arm['name']}.json"
            command = [sys.executable, str(Path(arm["source"]) / FROZEN_FILES[0]), "run",
                       "--source", arm["source"], "--duel", spec["duel"], "--out", str(out),
                       "--rounds", "1", "--replays", str(spec.get("replays", 5))]
            if arm.get("config"):
                command += ["--config", arm["config"]]
            subprocess.run(command, check=True, timeout=spec.get("arm_timeout", 1800))
            records.append(dict(round=round_index, arm=arm["name"], report=load(out)))
    samples = {}
    for record in records:
        for result in record["report"]["results"]:
            for row in result["rows"]:
                key = cell_key(result["profile"], row["capacity"], LAYERS[row["weight_layer"]], row["distribution"])
                samples.setdefault(key, {}).setdefault(record["arm"], []).append(row["median_us"]["coop"])
    summary = {}
    for key, arms in samples.items():
        base = statistics.median(arms["oracle"])
        summary[key] = {name: dict(median_us=statistics.median(v), min_us=min(v), max_us=max(v),
                                   speedup=base / statistics.median(v)) for name, v in arms.items()}
    verdicts = {}
    for name in names:
        if name == "oracle":
            continue
        ratios = [arms[name]["speedup"] for arms in summary.values()]
        verdicts[name] = dict(all_cells_at_least_oracle=all(v >= 1.0 for v in ratios),
                              worst_speedup=min(ratios),
                              note="No weighted-average escape: every fixture/cell gates. Ties require main's spread review.")
    save(args.out, dict(exact=True, records=records, summary=summary, verdicts=verdicts,
                        conditions="rotating fresh-process arms, one shared GPU lease; core+combine only"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--duel", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--config", type=Path)
    r.add_argument("--baseline", action="store_true")
    r.add_argument("--check-only", action="store_true")
    r.add_argument("--profiles", default=",".join(PROFILES))
    r.add_argument("--capacities", default=",".join(map(str, CAPACITIES)))
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--replays", type=int, default=5)
    s = sub.add_parser("session")
    s.add_argument("--spec", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args) if args.command == "run" else session(args)


if __name__ == "__main__":
    main()
