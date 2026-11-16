"""One warmed native convolution row curve, with a DRAM traffic floor.

Ceiling before measurement: four-tap convolution is bandwidth/launch-bound;
static H4608 ideal traffic is (4 input reads + output) * 2 bytes/channel;
dynamic KV1024 adds 4 coefficients/head and BF16 static taps. Reuse can make
this conservative traffic estimate exceed measured DRAM reads. Two launches
are unavoidable in this first implementation because all history reads must
complete before committing mutable state. Stop bar: oracle passes and no
>10% adjacent per-row cliff at 2048/2752/4096; no tuning campaign.
"""
from __future__ import annotations

import argparse
import json
import platform

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="2048,2752,4096")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot
    rows = [int(n) for n in args.rows.split(",")]
    records = []
    for width, dynamic in [(4608, False), (1024, False), (1024, True)]:
        program = compile_inkling2_convolution_aot(width=width, heads=8 if dynamic else 1,
                                                 dynamic=dynamic, max_rows=max(rows))
        for count in rows:
            x = torch.randn(count, width, device="cuda", dtype=torch.bfloat16)
            weights = torch.randn((4, width) if dynamic else (width, 4), device="cuda").bfloat16()
            coefficients = torch.randn(count, 8, 4, device="cuda", dtype=torch.bfloat16) if dynamic else None
            history = torch.zeros(1, 3, width, device="cuda", dtype=torch.bfloat16)
            first = torch.zeros(count, dtype=torch.int32, device="cuda")
            slots = torch.zeros(count, dtype=torch.int64, device="cuda")
            out = torch.empty_like(x)
            def run():
                program.launch(x, weights, coefficients, history, first, slots, out, scalars=(count,3*width))
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            times = []
            for _ in range(args.samples):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(begin.elapsed_time(end) * 1000)
            us = sorted(times)[len(times) // 2]
            row = dict(width=width, dynamic=dynamic, rows=count, us=us, us_per_row=us/count,
                       traffic_bytes=count * width * 10 + (count * 8 * 4 * 2 if dynamic else 0))
            print(json.dumps(row), flush=True)
            records.append(row)
    result = dict(host=platform.node(), gpu=torch.cuda.get_device_name(), samples=args.samples,
                  method="one warm launch, captured replay CUDA events, median", records=records)
    with open(args.json, "w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
