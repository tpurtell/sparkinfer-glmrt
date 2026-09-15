#!/usr/bin/env python3
"""Compare planned A4 and native-weight A16 MoE decode on one NVFP4 checkpoint.

Each precision must pass its existing GPU oracle before either arm is timed.
Both arms share packed weights and block scales, use fixed-capacity scratch,
and replay CUDA graphs with kernel resolution frozen. Timings exclude loading,
preparation, compilation, routing generation, and optional L2 eviction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from b12x._lib.runtime_control import kernel_resolution_guard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from b12x.moe import fused_moe
from b12x.moe._shared.kernels.reference import moe_reference_nvfp4
from b12x.preparation.types import require_prepared
from benchmarks.benchmark_moe import (
    MODEL_PROFILES,
    bench_events,
    build_model_spec,
    compare_to_reference,
    get_quant_mode_params,
    load_expert_weights,
    make_oracle_reference,
    make_profile_routed_inputs,
)
from benchmarks.benchmark_w4a16_nvfp4_layouts import (
    _check,
    _gpu_snapshot,
    _sha256,
    _weight_hashes,
)
from benchmarks.common import make_l2_flush_fn, resolve_l2_flush_bytes
from benchmarks.moe_checkpoint_snapshot import load_snapshot, save_snapshot, tensor_digest
from benchmarks.moe_preparation import prepared_call, request_for_capacity, scratch_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="glm52")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--cache", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    snapshots = parser.add_mutually_exclusive_group()
    snapshots.add_argument("--input-snapshot", type=Path)
    snapshots.add_argument("--export-input-snapshot", type=Path)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        parser.error("set CUDA_VISIBLE_DEVICES to the single assigned GPU")
    if min(*args.batch_sizes, args.warmup, args.iterations, args.rounds) <= 0:
        parser.error("batch sizes, warmup, iterations, and rounds must be positive")
    if args.capacity < max(args.batch_sizes):
        parser.error("capacity must cover every batch size")
    profile = MODEL_PROFILES[args.model_profile]
    if profile.shape is not None or profile.default_activation != "silu":
        parser.error("select a checkpoint profile with SiLU activation")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    spec = build_model_spec(args.model_path, profile, tp_size_override=args.tp_size)
    report = {
        "command": [sys.executable, *sys.argv],
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "worktree": str(ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("B12X_", "CUTE_DSL_", "CUDA_VISIBLE_DEVICES"))
        },
        "versions": {
            name: metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl", "triton")
        },
        "source_sha256": {
            str(path): _sha256(ROOT / path)
            for path in (
                Path(__file__).relative_to(ROOT),
                Path("benchmarks/benchmark_w4a16_nvfp4_layouts.py"),
                Path("benchmarks/benchmark_moe.py"),
                Path("benchmarks/common.py"),
                Path("benchmarks/moe_checkpoint_snapshot.py"),
                Path("b12x/moe/fused_moe/_impl.py"),
                Path("b12x/moe/_shared/kernels/w4a16/kernel.py"),
                Path("b12x/moe/_shared/kernels/w4a16/prepare.py"),
                Path("b12x/moe/_shared/kernels/micro.py"),
                Path("b12x/moe/_shared/kernels/dynamic.py"),
                Path("b12x/moe/_shared/kernels/reference.py"),
                Path("b12x/_lib/intrinsics.py"),
            )
        },
        "checkpoint": str(args.model_path.resolve()),
        "checkpoint_metadata_sha256": {
            name: _sha256(args.model_path / name)
            for name in ("config.json", "model.safetensors.index.json")
        },
        "layer": args.layer_idx,
        "shape": asdict(spec),
        "routing": profile.default_routing,
        "seed": 42,
        "capacity": args.capacity,
        "cache": args.cache,
        "l2_flush_bytes": (
            resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cache != "warm" else 0
        ),
        "timing": {
            "cold": "per-replay CUDA events; L2 eviction excluded",
            "warm": "CUDA events around consecutive graph replays",
        },
        "ratio": "A16 latency / A4 latency; lower is better for A16",
        "qualified": False,
        "cases": [],
    }

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Loading {args.model_path}, layer={args.layer_idx}, {spec}", flush=True)
    snapshot_metadata = {
        "checkpoint_metadata_sha256": report["checkpoint_metadata_sha256"],
        "shape": asdict(spec), "layer": args.layer_idx,
        "routing": profile.default_routing, "seed": 42,
    }
    if args.input_snapshot:
        weights, inputs, digests = load_snapshot(
            args.input_snapshot, device=device, expected_metadata=snapshot_metadata,
        )
        if weights.spec != spec or weights.layer_idx != args.layer_idx:
            raise ValueError("snapshot shape or layer does not match the benchmark")
        if any(m not in inputs for m in args.batch_sizes):
            raise ValueError("snapshot must contain every requested token count")
        report["input_snapshot_sha256"] = _sha256(args.input_snapshot)
        report["snapshot_tensor_digests"] = digests
    else:
        weights = load_expert_weights(
            args.model_path, layer_idx=args.layer_idx,
            checkpoint_family=profile.checkpoint_family, spec=spec, activation="silu",
        )
        inputs = {
            m: make_profile_routed_inputs(profile, weights, spec, m, 42, device)
            for m in args.batch_sizes
        }
        if args.export_input_snapshot:
            save_snapshot(
                args.export_input_snapshot, weights=weights, inputs=inputs,
                metadata=snapshot_metadata,
            )
            report["input_snapshot_sha256"] = _sha256(args.export_input_snapshot)
    if weights.source_format != "modelopt_nvfp4":
        raise ValueError(f"expected ModelOpt NVFP4, got {weights.source_format}")
    raw_params = get_quant_mode_params(weights, "shared", "w4a16")
    # Capacity is an independently specialized prepared variant even when it
    # is not one of the reported exact-M oracle cases.
    if args.capacity not in inputs:
        inputs[args.capacity] = make_profile_routed_inputs(
            profile, weights, spec, args.capacity, 42, device,
        )
    a4_params = get_quant_mode_params(weights, "shared", "nvfp4")
    # Public preparation accepts raw weight globals for both precisions.
    bundle = fused_moe.PackedWeights(
        w13=weights.w13_weight, w2=weights.w2_weight,
        w13_block_scales=weights.w13_blockscale_swizzled,
        w2_block_scales=weights.w2_blockscale_swizzled,
        w13_global_scales=raw_params.g1_alphas,
        w2_global_scales=raw_params.g2_alphas,
        input_scale=raw_params.a1_gscale, intermediate_scale=raw_params.a2_gscale,
    )
    report["native_weight_sha256"] = _weight_hashes(bundle)
    source = fused_moe.PackedSource(
        format=weights.source_format, w13_layout=weights.w13_layout,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=spec.num_experts, hidden_size=spec.hidden_size,
        intermediate_size=spec.I_tp,
    )
    arms = {}
    requests = []
    for name, mode in (("a4", fused_moe.ActivationMode.A4), ("a16", fused_moe.ActivationMode.AUTO)):
        weight_plan = fused_moe.plan_weights(
            source=source, geometry=geometry,
            activation=fused_moe.ActivationSpec(
                mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16,
            ),
            constraints=fused_moe.WeightPlanConstraints(
                required_packing=fused_moe.WeightPacking.SOURCE_NATIVE,
            ),
        )
        experts = fused_moe.prepare_weights(plan=weight_plan, weights=bundle)
        declaration = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=args.capacity, top_k=spec.top_k,
                warmup_token_counts=tuple(args.batch_sizes),
            ),
            override=(
                fused_moe.MoeDecodeConfig(
                    backend="w4a16", route_planner="internal",
                    max_active_clusters=None, w4a16_route_mode="direct",
                ) if name == "a16" else None
            ),
        )
        outputs = {m: torch.empty_like(inputs[m][0]) for m in sorted({
            args.capacity, *args.batch_sizes,
        })}
        calls = {
            m: prepared_call(
                output=outputs[m],
                bind=lambda state, scratch, m=m: state.bind(
                    scratch=scratch, a=inputs[m][0], experts=experts,
                    topk_weights=inputs[m][2], topk_ids=inputs[m][1],
                    output=outputs[m], input_scales_static=True,
                ),
            )
            for m in getattr(declaration, "token_counts", (args.capacity,))
        }
        requests.append(request_for_capacity(
            declaration, name=f"nvfp4-decode-{name}",
            calls=calls,
        ))
        arms[name] = {"experts": experts, "outputs": outputs, "plan": declaration}
        for attr, original in (("w1_fp4", bundle.w13), ("w2_fp4", bundle.w2)):
            assert getattr(experts._impl, attr).data_ptr() == original.data_ptr()
    session = PreparationSession(device=device)
    result = session.prepare(requests)
    native = arms["a16"]["experts"]._impl.representation_for("w4a16")
    for micro_attr, a4_attr, original in (
        ("micro_w13_scale", "w1_blockscale", bundle.w13_block_scales),
        ("micro_w2_scale", "w2_blockscale", bundle.w2_block_scales),
    ):
        assert getattr(native, micro_attr).data_ptr() == original.data_ptr()
        assert getattr(arms["a4"]["experts"]._impl, a4_attr).data_ptr() == original.data_ptr()
    assert native.w13_scale.data_ptr() == bundle.w13_block_scales.data_ptr()
    assert native.w2_scale.data_ptr() == bundle.w2_block_scales.data_ptr()
    assert _weight_hashes(bundle) == report["native_weight_sha256"]
    report["shared_payload_and_decode_block_scales"] = True
    cache_modes = ("cold", "warm") if args.cache == "both" else (args.cache,)
    l2_flush = make_l2_flush_fn("cold" in cache_modes, args.l2_flush_bytes)
    replay_cases = []
    failures = []
    for m in args.batch_sizes:
        x, ids, route_weights = inputs[m]
        case = {
            "tokens": m, "plans": {}, "oracle_metrics": {}, "cache": {},
            "active_experts": int(ids.unique().numel()),
            "input_digests": {
                name: tensor_digest(value) for name, value in
                (("x", x), ("ids", ids), ("route_weights", route_weights))
            },
        }
        report["cases"].append(case)
        replays = {}
        for name, arm in arms.items():
            out = arm["outputs"][m]
            variants = getattr(arm["plan"], "variants", None)
            plan = arm["plan"] if variants is None else variants.get(m, arm["plan"])
            binding = fused_moe.bind(
                plan, scratch=scratch_for(plan), a=x, experts=arm["experts"],
                topk_weights=route_weights, topk_ids=ids, output=out,
                input_scales_static=True,
            )
            config = plan.selection.config
            case["plans"][name] = {
                "variant": asdict(config),
                "bound": asdict(config),
                "bound_capacity": m,
                "scratch_bytes": sum(
                    spec.nbytes
                    for spec in require_prepared(plan, plan.component_id).scratch.scratch_specs()
                ),
            }
            if name == "a4":
                scale_math = {
                    "micro": "reciprocal_multiply", "dynamic": "direct_division",
                }[config.backend]
                case["a4_oracle_quant_scale_math"] = scale_math
                expected = moe_reference_nvfp4(
                    x, weights.w13_weight, weights.w13_blockscale_swizzled,
                    a4_params.g1_alphas,
                    weights.w2_weight, weights.w2_blockscale_swizzled,
                    a4_params.g2_alphas, a4_params.a1_gscale, a4_params.a2_gscale,
                    ids, route_weights, spec.num_experts, spec.hidden_size, spec.I_tp,
                    activation="silu", quant_scale_math=scale_math,
                )
            else:
                expected = make_oracle_reference(
                    "w4a16", "w4a16", x, weights, raw_params, ids, route_weights,
                    activation="silu",
                )
            for _ in range(args.warmup):
                fused_moe.run(binding=binding)
            torch.cuda.synchronize()
            case["oracle_metrics"][name] = asdict(compare_to_reference(out, expected))
            try:
                _check(
                    out, expected, name=name, m=m,
                    oracle_mode="nvfp4" if name == "a4" else "w4a16",
                )
            except AssertionError as error:
                failures.append(str(error))
                print(str(error), flush=True)
            replays[name] = {"binding": binding, "output": out, "expected": expected}
        replay_cases.append((case, replays))
        save()
        print(f"M={m}: oracle metrics={case['oracle_metrics']}, A4 backend={case['plans']['a4']['bound']['backend']}", flush=True)
    report["qualification_failures"] = failures
    save()
    if failures:
        raise AssertionError("\n".join(failures))

    with kernel_resolution_guard('NVFP4 A4/A16 decode graph qualification'):
        for case, replays in replay_cases:
            for arm in replays.values():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fused_moe.run(binding=arm["binding"])
                arm["graph"] = graph
            allocated = torch.cuda.memory_allocated()
            for name in ("a4", "a16", "a4", "a16"):
                arm = replays[name]
                arm["output"].fill_(float("nan"))
                arm["graph"].replay()
                _check(
                    arm["output"], arm["expected"], name=name + " precision switch",
                    m=case["tokens"], oracle_mode="nvfp4" if name == "a4" else "w4a16",
                )
            assert torch.cuda.memory_allocated() == allocated
            case["precision_switch_passed"] = True
            for cache_mode in cache_modes:
                timing = {
                    "samples_us": {name: [] for name in arms},
                    "replay_samples_us": {name: [] for name in arms}, "gpu": [],
                }
                for round_idx in range(args.rounds):
                    timing["gpu"].append(_gpu_snapshot())
                    order = ("a4", "a16") if round_idx % 2 == 0 else ("a16", "a4")
                    for name in order:
                        graph = replays[name]["graph"]
                        if cache_mode == "cold":
                            samples = [value * 1000 for value in bench_events(
                                graph.replay, warmup=args.warmup, iters=args.iterations,
                                l2_flush=l2_flush,
                            )]
                            timing["replay_samples_us"][name].append(samples)
                            timing["samples_us"][name].append(statistics.median(samples))
                        else:
                            for _ in range(args.warmup):
                                graph.replay()
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(args.iterations):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            timing["samples_us"][name].append(
                                start.elapsed_time(end) * 1000 / args.iterations,
                            )
                timing["gpu"].append(_gpu_snapshot())
                medians = {
                    name: statistics.median(samples)
                    for name, samples in timing["samples_us"].items()
                }
                timing["median_us"] = medians
                timing["a16_over_a4"] = medians["a16"] / medians["a4"]
                case["cache"][cache_mode] = timing
                assert torch.cuda.memory_allocated() == allocated
                for name, arm in replays.items():
                    _check(
                        arm["output"], arm["expected"], name=name + " timed replay",
                        m=case["tokens"], oracle_mode="nvfp4" if name == "a4" else "w4a16",
                    )
                save()
                print(
                    f"M={case['tokens']} {cache_mode}: A4={medians['a4']:.2f} us, "
                    f"A16={medians['a16']:.2f} us, A16/A4={timing['a16_over_a4']:.3f}",
                    flush=True,
                )
    assert _weight_hashes(bundle) == report["native_weight_sha256"]
    report["native_weights_unchanged"] = True
    report["replay_allocation_stable"] = True
    report["qualified"] = True
    save()
    for _, replays in replay_cases:
        for arm in replays.values():
            arm["graph"].reset()
    result.close()
    session.close()


if __name__ == "__main__":
    main()
