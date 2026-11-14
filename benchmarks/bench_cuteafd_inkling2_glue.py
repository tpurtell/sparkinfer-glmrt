"""Warmed RMSNorm/KV pack row curve. Ceiling: 6 bytes per norm value
(F32 read+BF16 write) and 12 bytes per KV pair (2 F32 read+2 BF16 write).
Stop bar oracle and no adjacent per-row slowdown>10%; no tuning campaign.
"""
import argparse
import json
import platform
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', default='2048,2752,4096')
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--json', required=True)
    args = parser.parse_args()
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_norm_aot, compile_inkling2_kv_pack_aot
    counts = [int(n) for n in args.rows.split(',')]
    records = []
    for kind in ['norm', 'kv_pack']:
        p = compile_inkling2_norm_aot(max_rows=max(counts)) if kind=='norm' else compile_inkling2_kv_pack_aot(max_rows=max(counts))
        for rows in counts:
            if kind=='norm':
                buffers = (torch.randn(rows, 4608, device='cuda'), torch.randn(4608, device='cuda').bfloat16(),
                           torch.empty(rows, 4608, device='cuda', dtype=torch.bfloat16))
            else:
                buffers = (torch.randn(rows, 1024, device='cuda'), torch.randn(rows, 1024, device='cuda'),
                           torch.arange(rows, device='cuda', dtype=torch.int64),
                           torch.empty(rows, 2048, device='cuda', dtype=torch.bfloat16))
            def run():
                p.launch(*buffers, scalars=(rows,))
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            times=[]
            for _ in range(args.samples):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record(); graph.replay(); end.record(); end.synchronize()
                times.append(begin.elapsed_time(end)*1000)
            us=sorted(times)[len(times)//2]
            row=dict(kind=kind, rows=rows, us=us, us_per_row=us/rows)
            print(json.dumps(row),flush=True); records.append(row)
    with open(args.json,'w') as stream:
        json.dump(dict(host=platform.node(), gpu=torch.cuda.get_device_name(), samples=args.samples,
                       method='one warm launch, captured replay events, median',records=records),stream,indent=2)


if __name__=='__main__':
    main()
