#!/usr/bin/env python3
"""Compile and export the source-W13 W4A4 / packed-W2 W4A16 Spark bundle.

Run this on the exact target GPU. The compiler rejects an architecture mismatch
so an SM120 workstation cannot accidentally produce the SM121 Spark release
bundle. Every capacity receives three C ABI objects plus a JSON manifest; a
bundle manifest records source and device provenance for native integrations.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Disk-cache hits are launchable external binaries but do not retain the IR
# required by ``export_to_c``. An exporter must compile exportable objects.
os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "0"
os.environ["SPARKINFER_COMPILE_MEMORY_CACHE"] = "1"

import torch

from sparkinfer.moe.fused_moe.aot import (
    W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS,
    W4A4FC1W4A16FC2SparkSpec,
    compile_w4a4_fc1_w4a16_fc2_spark_aot,
)


def _git_provenance(root: pathlib.Path) -> dict[str, object]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": revision,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def _parse_capacities(value: str) -> tuple[int, ...]:
    try:
        capacities = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not capacities or len(set(capacities)) != len(capacities):
        raise argparse.ArgumentTypeError("capacities must be unique and nonempty")
    unsupported = set(capacities) - set(W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported capacities {sorted(unsupported)}"
        )
    return capacities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument(
        "--capacities",
        type=_parse_capacities,
        default=W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS,
        help="comma-separated M capacities (default: all)",
    )
    parser.add_argument("--hidden-size", type=int, default=6_144)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--planning-sms", type=int, default=48)
    parser.add_argument("--grid-x", type=int, default=48)
    parser.add_argument(
        "--target-arch",
        choices=("auto", "sm_120", "sm_121"),
        default="auto",
    )
    parser.add_argument(
        "--file-prefix",
        default="sparkinfer_hybrid_w4a4_w4a16",
    )
    parser.add_argument(
        "--symbol-prefix",
        default="glmrt_sparkinfer_hybrid_w4a4_w4a16",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.hidden_size <= 0 or args.intermediate_size <= 0:
        raise SystemExit("hidden and intermediate sizes must be positive")
    if args.planning_sms <= 0 or args.grid_x <= 0:
        raise SystemExit("planning-sms and grid-x must be positive")

    torch.cuda.init()
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    device_arch = f"sm_{capability[0]}{capability[1]}"
    target_arch = device_arch if args.target_arch == "auto" else args.target_arch
    if target_arch != device_arch:
        raise SystemExit(
            f"target {target_arch} does not match compile device {device_arch}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[1]
    capacity_manifests: list[dict[str, Any]] = []
    for capacity in args.capacities:
        spec = W4A4FC1W4A16FC2SparkSpec(
            m=capacity,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            planning_sms=args.planning_sms,
            tuned_grid_x=args.grid_x,
            target_arch=target_arch,
        )
        artifact = compile_w4a4_fc1_w4a16_fc2_spark_aot(spec, device=device)
        file_base = f"{args.file_prefix}_m{capacity}"
        symbol_base = f"{args.symbol_prefix}_m{capacity}"
        manifest_path = artifact.export_to_c(
            output_dir,
            file_name=file_base,
            symbol_base=symbol_base,
        )
        capacity_manifests.append(
            {
                "capacity_m": capacity,
                "file_base": file_base,
                "symbol_base": symbol_base,
                "manifest": manifest_path.name,
                "objects": [
                    f"{file_base}_fc1.o",
                    f"{file_base}_activation.o",
                    f"{file_base}_fc2.o",
                ],
                "headers": [
                    f"{file_base}_fc1.h",
                    f"{file_base}_activation.h",
                    f"{file_base}_fc2.h",
                ],
            }
        )
        print(f"exported M={capacity}: {manifest_path}", flush=True)

    bundle = {
        "format": "sparkinfer_w4a4_fc1_w4a16_fc2_spark_bundle",
        "source": _git_provenance(root),
        "device": {
            "index": device_index,
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": list(capability),
            "target_arch": target_arch,
            "hardware_sm_count": int(properties.multi_processor_count),
        },
        "shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "top_k": 1,
        },
        "launch": {
            "planning_sms": args.planning_sms,
            "tuned_grid_x": args.grid_x,
        },
        "capacities": capacity_manifests,
    }
    bundle_path = output_dir / f"{args.file_prefix}_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(bundle_path)


if __name__ == "__main__":
    main()
