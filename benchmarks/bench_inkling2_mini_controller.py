#!/usr/bin/env python3
"""Controller projection row-invariance and timing on the native CuTe path."""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', default=','.join(map(str, (*range(1,17), 32,64,128,2048,2752,4096))))
    p.add_argument('--compare-routes-exact', action='store_true')
    p.add_argument('--samples', type=int, default=12)
    args = p.parse_args()
    import torch
    import cutlass
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection, SkinnyBf16Gemv
    from b12x.integration.cuteafd.inkling2_loop_controller import _ControllerDownSplitK
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from benchmarks.common import capture_cuda_graph, bench_cuda_graph, make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
    rows_list = sorted(set(map(int,args.rows.split(','))))
    cap = max(rows_list)
    torch.manual_seed(2804)
    x = torch.randn(cap,4608,device='cuda').bfloat16()
    w = (torch.randn(64,4608,device='cuda')*.03).bfloat16()
    out = torch.empty(cap,64,device='cuda',dtype=torch.bfloat16)
    partial = torch.empty(9,cap,64,device='cuda')
    operands = (Operand('x',torch.bfloat16,'x'),Operand('w',torch.bfloat16,'w'),Operand('out',torch.bfloat16,'out','out'))
    candidates = {
        'base': TmaBf16Projection(64,4608,tiles_by_rows=((16,1,4,16,64,6),)),
        'narrow': TmaBf16Projection(64,4608,tiles_by_rows=((16,1,1,16,8,6),)),
        'skinny': SkinnyBf16Gemv(64,4608,cols=1,rows_per_tile=1,threads=288),
        'split8': _ControllerDownSplitK(),
        'split9': _ControllerDownSplitK(splits=9,decode_tile=(1,1,16,8,4,64)),
        'split4': _ControllerDownSplitK(splits=4,decode_tile=(1,1,16,8,4,64)),
    }
    programs = {}
    for name, op in candidates.items():
        is_split = name.startswith('split')
        ops = operands + ((Operand('partial',torch.float32,'partial','out'),) if is_split else ())
        program = compile_program(op,name='mini_controller_probe_'+name,operands=ops,scalars=(Scalar('rows'),),key=(name,))
        programs[name] = lambda m, p=program, s=is_split: p.launch(x,w,out,*((partial,) if s else ()),scalars=(m,))
    flush = make_l2_flush_fn(True)
    print(nvidia_smi_gpu_mode_snapshot(),flush=True)
    for m in rows_list:
        ref = None
        if m > 128:
            active = [n for n in programs if n not in ('skinny',)]
        else:
            active = list(programs)
        gs = {}
        for name in active:
            f = programs[name]
            out.fill_(float('nan')); f(m); torch.cuda.synchronize()
            result = out[:m].clone()
            assert result.isfinite().all() and out[m:].isnan().all(), (name,m)
            if name == 'base': ref = result
            if args.compare_routes_exact and name == 'narrow': assert torch.equal(result.view(torch.int16),ref.view(torch.int16)),(name,m)
            f(1); torch.cuda.synchronize()
            assert torch.equal(result[0].view(torch.int16),out[0].view(torch.int16)), (name,m,'row invariance')
            torch.testing.assert_close(result.float(),(x[:m].float() @ w.float().T),rtol=.007,atol=.006)
            g = capture_cuda_graph(lambda: f(m),warmup=2)
            out.fill_(float('nan')); g.replay(); torch.cuda.synchronize()
            assert torch.equal(result.view(torch.int16),out[:m].view(torch.int16)),(name,m,'replay')
            gs[name] = g
        samples = {name: [] for name in active}
        for arm_order in (active,active[::-1]):
            for name in arm_order:
                samples[name].extend(bench_cuda_graph(gs[name],replays=args.samples,l2_flush=flush)['replay_us'])
        print(m,{name:round(statistics.median(ts),3) for name,ts in samples.items()},flush=True)
    print(nvidia_smi_gpu_mode_snapshot(),flush=True)

if __name__ == '__main__': main()
