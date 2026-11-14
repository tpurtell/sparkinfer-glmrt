"""Warmed clip row curves. Ceiling: input/output minimum is M*heads*dim*4
bytes for BF16, with learned scale cached. Bandwidth/launch-bound; no tuning.
Stop bar: #1915 oracle passes and no >10% adjacent per-row slowdown.
"""
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
    from b12x.integration.cuteafd.inkling2_clip import compile_inkling2_clip_aot
    rows = [int(n) for n in args.rows.split(",")]
    records = []
    for heads, dim, scaled in [(48, 128, True), (48, 16, True), (8, 128, False)]:
        p = compile_inkling2_clip_aot(heads=heads, dim=dim, scaled=scaled, threshold=8.,
                                     log_floor=8192, log_alpha=.25, max_rows=max(rows))
        for count in rows:
            x = torch.randn(count, heads, dim, device="cuda", dtype=torch.bfloat16)
            scale = torch.randn(heads, dim, device="cuda", dtype=torch.bfloat16) if scaled else None
            positions = torch.arange(count, device="cuda", dtype=torch.int64)
            out = torch.empty_like(x)
            def run():
                p.launch(x, scale, positions, out, scalars=(count,))
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
            us = sorted(times)[len(times)//2]
            result = dict(heads=heads, dim=dim, scaled=scaled, rows=count, us=us,
                          us_per_row=us/count, minimum_bytes=count*heads*dim*4)
            print(json.dumps(result), flush=True)
            records.append(result)
    with open(args.json, "w") as stream:
        json.dump(dict(host=platform.node(), gpu=torch.cuda.get_device_name(), samples=args.samples,
                       method="one warm launch, captured replay events, median", records=records), stream, indent=2)


if __name__ == "__main__":
    main()
