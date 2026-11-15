#!/usr/bin/env python3
"""Muse BF16 versus same-scale E4M3 KV row curves through native AOT launches.

Run inside the coordinator dev image under the engine's golden-run --command.
This only constructs native inputs, checks exact output, and times warm graphs;
it neither starts serving nor acquires hardware independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cuteafd_minimax import NvmlWindow, measure


def records_qdq(records, heads):
    kv = records.reshape(-1, 2, heads, 64).float()
    amax = kv.abs().amax(-1)
    scales = torch.where(amax > 0, (amax.double() / 448).float(), torch.ones_like(amax))
    payload = (kv * (1.0 / scales.double()).float()[..., None]).to(torch.float8_e4m3fn)
    return (torch.cat((payload.view(torch.uint8).flatten(1), scales.contiguous().view(torch.uint8).flatten(1)), 1),
            (payload.float() * scales[..., None]).bfloat16().flatten(1))


def case(heads, rows, history, kind, route, iterations, samples):
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    gen = torch.Generator().manual_seed(1856)
    kv_heads, ring_rows, end = heads // 8, 2112, history + rows
    query = torch.randn(rows, heads, 64, generator=gen).bfloat16().cuda()
    source = torch.randn(end, kv_heads * 128, generator=gen).bfloat16().cuda()
    packed, qdq = records_qdq(source, kv_heads)
    positions = torch.arange(history, end, dtype=torch.int64, device="cuda")
    slots = positions % ring_rows
    first = torch.zeros(rows, dtype=torch.int32, device="cuda")
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    functions, outputs = {}, {}
    for fmt, records in (("bf16", qdq), ("fp8", packed)):
        out = torch.empty_like(query)
        program = compile_attention_aot(heads=heads, kind=kind, route=route, ring_rows=ring_rows,
                                        kv_cache=fmt, max_rows=4096)
        if kind == "full":
            pages = (end + 63) // 64
            cache = torch.zeros(pages * 64, records.shape[1], dtype=records.dtype, device="cuda")
            cache[:end] = records
            # Reversed physical pages exercise the real page-table resolve.
            cache = cache.reshape(pages, 64, -1).flip(0).contiguous().flatten(0, 1)
            table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device="cuda")
            args = (query, cache, positions, table, out, scratch)
            scalars = [rows, 0] + ([1] if route == "decode" else [])
        else:
            ring = torch.zeros(ring_rows, records.shape[1], dtype=records.dtype, device="cuda")
            prior = torch.arange(max(0, history - 2048), history, device="cuda")
            ring[prior % ring_rows] = records[prior]
            step = records[history:].contiguous()
            args = (query, step, ring, positions, slots, first, out, scratch)
            scalars = [rows]
        fn = lambda program=program, args=args, scalars=scalars: program.launch(*args, scalars=scalars)
        fn()
        functions[fmt], outputs[fmt] = fn, out
    torch.cuda.synchronize()
    assert torch.equal(outputs["bf16"], outputs["fp8"]), (heads, rows, history, kind, route,
        float((outputs["bf16"].float() - outputs["fp8"].float()).abs().max()))
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid

    def hardware():
        return subprocess.check_output(["nvidia-smi", "--id=" + gpu_uuid,
            "--query-gpu=uuid,pstate,power.limit,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active",
            "--format=csv,noheader"], text=True).strip()

    timings = {}
    with NvmlWindow() as thermal:
        with kernel_resolution_guard("Muse KV row curves use frozen compiled programs"):
            for fmt, fn in functions.items():
                before = hardware()
                median, raw = measure(fn, iterations, samples)
                after = hardware()
                timings[fmt] = dict(median_us=median, samples_us=raw, us_per_row=median / rows,
                                    hardware_before=before, hardware_after=after)
    window = thermal.result()
    values = window["samples"]
    result = dict(heads=heads, kv_heads=kv_heads, rows=rows, history=history, kind=kind, route=route,
                  exact=True, timings=timings, speedup=timings["bf16"]["median_us"] / timings["fp8"]["median_us"],
                  thermal=window)
    result["thermal_summary"] = dict(power_cap_s=window["power_cap_sampled_s"],
        thermal_s=window["thermal_sampled_s"],
        max_temperature_c=max((v["temperature_c"] for v in values if v["temperature_c"] is not None), default=None),
        min_sm_clock_mhz=min((v["sm_clock_mhz"] for v in values if v["sm_clock_mhz"] is not None), default=None),
        max_sm_clock_mhz=max((v["sm_clock_mhz"] for v in values if v["sm_clock_mhz"] is not None), default=None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default="64,128")
    parser.add_argument("--rows", default="1,2,3,5,8,16,32,64,80")
    parser.add_argument("--history", default="2050,8192")
    parser.add_argument("--kind", default="full,swa")
    parser.add_argument("--route", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.iterations > 0 and args.samples > 0
    results = []
    for heads in map(int, args.heads.split(",")):
        for history in map(int, args.history.split(",")):
            for kind in args.kind.split(","):
                for rows in map(int, args.rows.split(",")):
                    result = case(heads, rows, history, kind, args.route, args.iterations, args.samples)
                    results.append(result)
                    args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),
                        iterations=args.iterations, samples=args.samples, results=results), indent=2) + "\n")
                    print(json.dumps({k: v for k, v in result.items() if k != "thermal"}), flush=True)


if __name__ == "__main__":
    main()
