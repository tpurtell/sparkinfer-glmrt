#!/usr/bin/env python3
"""Native group32 MXFP8 decode versus prefill, exact-gated cold graph replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_cuteafd_minimax import NvmlWindow
from benchmarks.common import make_l2_flush_fn
from benchmarks.dense_autotune_common import bench_events, capture_graph_replay
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.integration.cuteafd.dense_mxfp8 import (
    compile_dense_mxfp8_aot, compile_dense_mxfp8_quant_aot, record_bytes,
)

# N x K, from the checkpoint's native projection headers and TP2 slices.
SHAPES = {
    "query": (8192, 6144), "key_value": (512, 6144), "index_key": (128, 6144),
    "output": (6144, 8192), "shared_gate_up": (3072, 6144),
    "shared_down": (6144, 3072), "dense_gate_up": (12288, 6144),
    "dense_down": (6144, 12288), "query_tp2": (4096, 6144),
    "key_value_tp2": (256, 6144), "output_tp2": (6144, 4096),
    "shared_gate_up_tp2": (1536, 6144), "shared_down_tp2": (6144, 1536),
    "dense_gate_up_tp2": (6144, 6144),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--shapes", default="query,key_value,output,shared_gate_up,shared_down")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--k-warps", type=int, choices=(1, 4), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows_list = [int(v) for v in args.rows.split(",")]
    names = args.shapes.split(",")
    if not rows_list or any(not 1 <= v <= 64 for v in rows_list) or args.samples < 1:
        parser.error("rows must be in 1..64 and samples positive")
    if not names or any(name not in SHAPES for name in names):
        parser.error("unknown projection shape")
    capacity = 64
    properties = torch.cuda.get_device_properties(0)
    limit = properties.multi_processor_count * 4
    flush = make_l2_flush_fn(True)
    report = dict(device=properties.name, uuid=str(properties.uuid),
        sm_count=properties.multi_processor_count, conditions=dict(
            cold_l2=True, graph_replay=True, quantization_excluded=True,
            native_group_size=32, accumulation="fixed K128", samples=args.samples,
            k_warps=args.k_warps),
        source_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] /
            "b12x/gemm/dense_mxfp8.py").read_bytes()).hexdigest(), measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        n, k = SHAPES[name]
        generator = torch.Generator(device="cuda").manual_seed(1856 + index)
        x = torch.randn((capacity, k), generator=generator, device="cuda").bfloat16()
        table = torch.tensor([x.data_ptr()], dtype=torch.int64, device="cuda")
        payload = torch.randn((n, k), generator=generator, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.randint(118, 125, (n, k // 32), generator=generator,
            device="cuda", dtype=torch.uint8)
        records = torch.empty(record_bytes(capacity, k), dtype=torch.uint8, device="cuda")
        outputs = {arm: torch.empty((capacity, n), dtype=torch.float32, device="cuda")
            for arm in ("prefill", "decode")}
        programs = {arm: compile_dense_mxfp8_aot(input_features=k, output_features=n,
            encoding="mxfp8", input_dtype="bf16", rotation_count=1, max_rows=capacity,
            decode=arm == "decode", decode_k_warps=args.k_warps) for arm in outputs}
        quant = compile_dense_mxfp8_quant_aot(input_features=k, input_dtype="bf16",
            rotation_count=1, max_rows=capacity)
        with kernel_resolution_guard("MiniMax decode curve: frozen M=1..64"):
            for rows in rows_list:
                quant.launch(table, records, scalars=(rows, 0, limit))
                runs = {arm: (lambda arm=arm: programs[arm].launch(table, payload,
                    scales, records, outputs[arm], scalars=(rows, 0, limit))) for arm in outputs}
                for output in outputs.values():
                    output.fill_(float("nan"))
                for run in runs.values():
                    run()
                torch.cuda.synchronize()
                if not torch.equal(outputs["prefill"][:rows].view(torch.int32), outputs["decode"][:rows].view(torch.int32)):
                    raise AssertionError(f"{name} M={rows}: decode differs from prefill")
                if any(not bool(torch.isnan(output[rows:]).all()) for output in outputs.values()):
                    raise AssertionError(f"{name} M={rows}: inactive output tail overwritten")
                graphs = {arm: capture_graph_replay(run) for arm, run in runs.items()}
                samples = {arm: [] for arm in graphs}
                with NvmlWindow() as thermal:
                    for iteration in range(args.samples):
                        order = ("prefill", "decode") if iteration % 2 == 0 else ("decode", "prefill")
                        for arm in order:
                            samples[arm].extend(v * 1000 for v in bench_events(graphs[arm],
                                warmup=1 if iteration == 0 else 0, iters=1, l2_flush=flush))
                us = {arm: statistics.median(values) for arm, values in samples.items()}
                weight_bytes = n * k + n * k // 32
                point = dict(shape=name, n=n, k=k, rows=rows, exact=True,
                    prefill_us=us["prefill"], decode_us=us["decode"],
                    speedup=us["prefill"] / us["decode"], samples_us=samples,
                    decode_weight_gbps=weight_bytes / us["decode"] / 1000,
                    weight_bytes=weight_bytes, nvml=thermal.result())
                report["measurements"].append(point)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in point.items() if k != "nvml"}), flush=True)
        del programs, graphs, runs, outputs, records, payload, scales, table, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
