"""First residual row curve. Ceiling: two F32 arena reads plus F32 output;
three independent queries repeat the reads (cache reuse is expected). Stop
when oracle passes and no >10% adjacent per-row slowdown; no tuning campaign.
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
    parser.add_argument("--cold-l2", action="store_true", help="read192MiB before each timed sample")
    args = parser.parse_args()
    from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot
    rows = [int(n) for n in args.rows.split(",")]
    records = []
    flush = torch.zeros(192*1024*1024//4, device="cuda") if args.cold_l2 else None
    for queries in (1, 3):
        p = compile_inkling2_residual_aot(queries=queries, max_rows=max(rows))
        for candidates in (2, 9):
            for count in rows:
                arena = torch.randn(9, count, 4608, device="cuda")
                query = torch.randn(queries, 4608, device="cuda")
                out = torch.empty(queries, count, 4608, device="cuda")
                def run():
                    p.launch(arena, None, query, out, scalars=(count, count, candidates))
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                times = []
                for _ in range(args.samples):
                    if flush is not None:
                        flush.max()
                        torch.cuda._sleep(3_000_000)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(begin.elapsed_time(end) * 1000)
                us = sorted(times)[len(times)//2]
                result = dict(queries=queries, candidates=candidates, rows=count, us=us, us_per_row=us/count,
                              traffic_bytes=queries*count*4608*4*(candidates*2+1))
                print(json.dumps(result), flush=True)
                records.append(result)
    with open(args.json, "w") as stream:
        json.dump(dict(host=platform.node(), gpu=torch.cuda.get_device_name(), samples=args.samples,
                       method="one warm launch, captured replay events, median", cold_l2=args.cold_l2, records=records), stream, indent=2)


if __name__ == "__main__":
    main()
