"""First router/gate curve. Ceiling: E384 logits + biases and k8 output,
then gated widths6144/2304. Router first version does eight fixed reductions;
stop bar is exact oracle ids/weights and no >10% adjacent per-row slowdown.
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
    from b12x.integration.cuteafd.inkling2_router import compile_inkling2_router_aot, compile_inkling2_gate_aot
    counts = [int(n) for n in args.rows.split(",")]
    router = compile_inkling2_router_aot(max_rows=max(counts))
    gate = compile_inkling2_gate_aot(width=6144, max_rows=max(counts))
    situ = compile_inkling2_gate_aot(width=2304, situ=True, max_rows=max(counts))
    records = []
    for kind, program, width in [("router", router, 384), ("gate", gate, 6144), ("situ", situ, 2304)]:
        for count in counts:
            if kind == "router":
                buffers = (torch.randn(count, 384, device="cuda"), torch.randn(384, device="cuda"),
                           torch.empty(count, 8, dtype=torch.int32, device="cuda"), torch.empty(count, 8, device="cuda"))
            else:
                buffers = (torch.randn(count, width*2 if kind=="situ" else width, device="cuda").bfloat16(),
                           torch.randn(count, width, device="cuda").bfloat16(), torch.rand(count, device="cuda"),
                           torch.empty(count, width, device="cuda", dtype=torch.bfloat16))
            def run():
                program.launch(*buffers, scalars=(count,))
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
                times.append(begin.elapsed_time(end)*1000)
            us = sorted(times)[len(times)//2]
            row = dict(kind=kind, rows=count, us=us, us_per_row=us/count)
            print(json.dumps(row), flush=True)
            records.append(row)
    with open(args.json, "w") as stream:
        json.dump(dict(host=platform.node(), gpu=torch.cuda.get_device_name(), samples=args.samples,
                       method="one warm launch, captured replay events, median", records=records), stream, indent=2)


if __name__ == "__main__":
    main()
