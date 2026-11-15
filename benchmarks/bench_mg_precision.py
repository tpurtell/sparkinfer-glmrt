"""MG precision diagnostic: unchanged kernels, native packing, FP32 attention oracle.

DSV4 584B prefill changes QK only (PV is FP8 NoPE / BF16 RoPE). Native
DSV41 prefill changes canonical KV staging/PV; both arms retain BF16 QK.
Native decode compares shipped E4M3 QK/PV with BF16 QK/PV, and optionally
FP32 split partials. Synthetic full selected sets are not a model forward.

Prefill examples::

    python benchmarks/bench_mg_precision.py --format dsv4 --output dsv4.json
    python benchmarks/bench_mg_precision.py --format dsv41 --output dsv41.json

Decode/verify example::

    python benchmarks/bench_mg_precision.py --format dsv41 --mode decode \
        --rows 1,2,4,8,16,64 --timing-rows 1,2,4,8,16,64 --output decode.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits


def arm_traits(fmt, arm):
    if fmt == "dsv4":
        if arm not in ("fp8-qk", "bf16-qk"):
            raise ValueError(arm)
        return make_unified_traits(ModelType.DSV4,
            ComputeMode.FP8 if arm == "fp8-qk" else ComputeMode.BF16,
            ScaleFormat.UE8M0_BYTE, fp8_rope=False)
    if fmt != "dsv41" or arm not in ("bf16-pv", "fp8-pv"):
        raise ValueError((fmt, arm))
    return replace(make_unified_traits(ModelType.DSV41, ComputeMode.BF16,
        ScaleFormat.NVFP4_E4M3, fp8_rope=False), fp8_internal=arm == "fp8-pv")


def parse_args(argv=None):
    def ints(s):
        try:
            v = [int(x) for x in s.split(",")]
        except ValueError as e:
            raise argparse.ArgumentTypeError("expected comma-separated positive integers") from e
        if not v or min(v) < 1:
            raise argparse.ArgumentTypeError("counts must be positive")
        return v
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=("dsv4", "dsv41"), required=True)
    p.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--indexed-width", type=int, default=512)
    p.add_argument("--rows", type=ints, default=ints("1,64,2048,2752,4096"))
    p.add_argument("--timing-rows", type=ints, default=ints("64,1536,2048,2752,4096"))
    p.add_argument("--scores", default="spread,peaked")
    p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=41512)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native-row-curve", action="store_true", help="Fixed-capacity native pointer ABI for both arms")
    args = p.parse_args(argv)
    args.scores = args.scores.split(",")
    if args.accuracy_only:
        args.timing_rows = []
    if set(args.scores) - {"spread", "peaked"}:
        p.error("scores must be spread,peaked")
    if args.heads < 16 or args.heads % 16 or min(args.indexed_width, args.samples, args.replays, args.warmup) < 1:
        p.error("heads must be a positive multiple of 16; counts must be positive")
    if args.indexed_width > 4096:
        p.error("indexed width exceeds the 4096-key synthetic pool")
    if args.mode == "decode" and (args.format != "dsv41" or args.indexed_width != 512):
        p.error("native decode requires dsv41 with indexed width512")
    return args


def decode_arm_traits(arm):
    if arm not in ("fp8-decode", "bf16-decode", "fp8-f32-partials"):
        raise ValueError(arm)
    native = arm != "bf16-decode"
    traits = arm_traits("dsv41", "fp8-pv" if native else "bf16-pv")
    if native:
        traits = replace(traits, compute_mode=ComputeMode.FP8, q_nope_stride=528,
            kv_smem_stride=624, nt_per_warp_xv=traits.nt_per_warp_xv * 2)
    return traits


def build_decode(q, main_cache, extra_cache, mi, ei, sink, native_curve=False):
    """Compile existing native producer and generic BF16 producer, same10 splits.

    The optional F32 partial arm changes only the typed tensor supplied to the
    existing epilogue/merge, not their implementation. Native producer mapping
    uses the same committed-ring/private-window ABI as test_native_v41_producer.
    """
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel
    from b12x.attention._shared.mla.smem import make_smem_layout
    from b12x.attention._shared.mla.merge import SparseMLASplitDecodeSinkMergeKernel
    rows, heads, _ = q.shape
    sr, cr = main_cache.view(-1,528), extra_cache.view(-1,288)
    values = [sr[:128,:512].contiguous(), sr[128:128+rows,:512].contiguous(), cr[:,:256].contiguous(), cr[:1,:256].contiguous()]
    scales = [sr[:128,512:].contiguous(), sr[128:128+rows,512:].contiguous(), cr[:,256:].contiguous(), cr[:1,256:].contiguous()]
    end = torch.tensor([128], device=q.device, dtype=torch.uint64)
    source_end = torch.tensor([cr.shape[0]], device=q.device, dtype=torch.uint64)
    pages = torch.arange(cr.shape[0]//256, device=q.device).to(torch.uint32)
    descriptor = [x.data_ptr() for x in values+scales] + [end.data_ptr(),pages.data_ptr(),source_end.data_ptr(),rows,cr.shape[0],1,(2<<32)|(cr.shape[0]//256)]
    desc = torch.tensor([descriptor]*rows, device=q.device, dtype=torch.uint64)
    meta = torch.tensor([[128,0,rows,128+r,0,cr.shape[0],cr.shape[0],0,0,1] for r in range(rows)], device=q.device, dtype=torch.uint64)
    bounds = torch.zeros(rows,device=q.device,dtype=torch.uint64)
    # Native ring uses committed first128 plus private rows; same logical keys
    # as generic main-cache slots1..128, then2..129, etc.
    assert torch.equal(mi, torch.arange(128,device=q.device)[None] + torch.arange(rows,device=q.device)[:,None] + 1)
    retained = [values,scales,end,source_end,pages,desc,meta,bounds]
    runs, lses = {}, {}
    arms = ("fp8-decode", "bf16-decode") if native_curve else ("fp8-decode", "bf16-decode", "fp8-f32-partials")
    for arm in arms:
        native = arm != "bf16-decode"
        traits = decode_arm_traits(arm)
        partial = torch.empty(rows,heads,10,512,device=q.device,dtype=torch.float32 if arm == "fp8-f32-partials" else torch.bfloat16)
        lse = torch.empty(rows,heads,10,device=q.device)
        output = torch.empty_like(q)
        kernel = UnifiedDecodeKernel(traits,make_smem_layout(traits),64,1,h_blocks=heads//16,num_splits=10,num_heads=heads,q_head_dim=512,
            topk=128,extra_topk=512,q_stride=q.stride(),swa_indices_stride0=512 if native or native_curve else 128,extra_indices_stride0=512,
            mid_out_stride=partial.stride(),mid_lse_stride=lse.stride(),has_extra=True,pbs_extra=256,valid_hpb=16,native_dsv41_fp8=native)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        scale = cutlass.Float32(512**-.5 * math.log2(math.e))
        if native_curve:
            from b12x.attention._shared.mla.native_v41_aot import NativeV41Attention
            from b12x._lib.utils import make_ptr, current_cuda_stream
            launch = NativeV41Attention(heads)
            if native:
                launch.decode = kernel
            tensors = (q,desc,meta,ei,bounds,sink,partial,lse,output)
            types = (cutlass.BFloat16,cutlass.Uint64,cutlass.Uint64,cutlass.Int32,cutlass.Uint64,cutlass.Float32,cutlass.BFloat16,cutlass.Float32,cutlass.BFloat16)
            aligns = (16,8,8,4,8,4,16,4,16)
            pointers = tuple(make_ptr(t,x.data_ptr(),cute.AddressSpace.gmem,assumed_align=a) for x,t,a in zip(tensors,types,aligns))
            compiled = cute.compile(launch,*pointers,cutlass.Int32(1),current_cuda_stream())
            def run(live_rows=None,compiled=compiled,pointers=pointers,retained=(retained,tensors)):
                compiled(*pointers,cutlass.Int32(rows if live_rows is None else live_rows),current_cuda_stream())
            runs[arm] = (run,output)
            lses[arm] = lse
            continue
        if native or native_curve:
            params = tuple(from_dlpack(x) for x in (q,desc,meta,ei,bounds,partial,lse))
            compiled = cute.compile(kernel.call_native_v41,*params,scale,cutlass.Int32(1 if native_curve else rows),stream)
            decode_args = (*params,scale,cutlass.Int32(rows))
        else:
            params = tuple(from_dlpack(x) for x in (q,main_cache.flatten(),mi,partial,lse))
            extra = tuple(from_dlpack(x) for x in (extra_cache.flatten(),ei))
            decode_args = (*params,scale,cutlass.Float32(1),cutlass.Int32(128),cutlass.Int64(main_cache.stride(0)),*extra,
                cutlass.Int32(512),cutlass.Int32(2),cutlass.Int64(extra_cache.stride(0)),cutlass.Int32(rows))
            compiled = cute.compile(kernel.call_extra,*decode_args,stream)
        merge = SparseMLASplitDecodeSinkMergeKernel(static_num_chunks=10)
        merge_args = tuple(from_dlpack(x) for x in (partial,lse,bounds,sink,output))
        merged = cute.compile(merge,*merge_args,stream)
        def run(live_rows=None, compiled=compiled,decode_args=decode_args,merged=merged,merge_args=merge_args,retained=retained):
            current = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            launch_args = (*decode_args[:-1],cutlass.Int32(live_rows)) if live_rows is not None else decode_args
            compiled(*launch_args,current)
            merged(*merge_args,current)
        runs[arm] = (run,output)
        lses[arm] = lse
    return runs, lses


class Telemetry:
    """Point-bounded NVML-backed telemetry at a 200 ms cadence."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        fields = "uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,temperature.gpu.tlimit,power.draw,power.limit,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown"
        try:
            while True:
                result = subprocess.check_output(["nvidia-smi", "--query-gpu="+fields,"--format=csv,noheader,nounits"],text=True)
                self.samples.append({"time_s":time.time(),"fields":fields,"csv":result.strip()})
                if self.stop.wait(.2):
                    return
        except Exception as exc:
            self.error = str(exc)

    def __enter__(self):
        self.error = None
        self.before = subprocess.check_output(["nvidia-smi","-q"],text=True)
        self.thread.start()
        return self

    def __exit__(self,*unused):
        self.stop.set()
        self.thread.join()
        self.after = subprocess.check_output(["nvidia-smi","-q"],text=True)
        if self.error:
            raise RuntimeError(self.error)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from b12x.attention._shared.mla.compressed_reference import (
        pack_compressed_sparse_mla_kv_cache_reference, pack_deepseek_v41_cache_reference,
        unpack_compressed_sparse_mla_kv_cache_reference, unpack_deepseek_v41_cache_reference)
    from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
    from b12x.attention._shared.mla.prefill import _mg_head_partitions

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    max_rows = max(args.rows + args.timing_rows)
    pool, window = 4096, 128
    # Native pages: DSV4 main 256 / C4 64; V4.1 main 32 / C1 256.
    main_page, extra_page = (256, 64) if args.format == "dsv4" else (32, 256)
    scales = torch.linspace(.3, 1.2, 512, device=device)
    values = [(torch.randn(pool, 512, device=device) * scales).bfloat16() for _ in range(2)]
    def pack(v, page, kind):
        if args.format == "dsv41":
            c = pack_deepseek_v41_cache_reference(v, page_size=page, cache_kind=kind)
            k = unpack_deepseek_v41_cache_reference(c, page_size=page, cache_kind=kind)
            return c, k, k
        c = pack_compressed_sparse_mla_kv_cache_reference(v[:, :448], v[:, 448:], page_size=page)
        k, val = unpack_compressed_sparse_mla_kv_cache_reference(c, page_size=page)
        return c, k, val
    if args.mode == "decode":
        if max_rows > 64:
            raise ValueError("decode diagnostic is bounded to64 rows")
        main_page = 64
    main_cache, mk, mv = pack(values[0], main_page, "swa")
    extra_cache, ek, ev = pack(values[1], extra_page, "indexed")
    row = torch.arange(max_rows, device=device)[:, None]
    mi = ((row + (1 if args.mode == "decode" else 0) + torch.arange(window, device=device)[None]) % pool).int().contiguous()
    ei = ((row * 37 + torch.arange(args.indexed_width, device=device)[None] * 17) % pool).int().contiguous()
    ml = torch.full((max_rows,), window, device=device, dtype=torch.int32)
    el = torch.full((max_rows,), args.indexed_width, device=device, dtype=torch.int32)
    sink = torch.linspace(-1, 1, args.heads, device=device)
    arms = ("fp8-qk", "bf16-qk") if args.format == "dsv4" else ("bf16-pv", "fp8-pv")
    if args.mode == "decode":
        arms = ("fp8-decode", "bf16-decode") if args.native_row_curve else ("fp8-decode", "bf16-decode", "fp8-f32-partials")
    report = {"mode": args.mode, "lse_unit": "natural_log", "format": args.format, "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "fork_sha": subprocess.check_output(["git", "-C", os.environ.get("CUTEAFD_SPARKINFER_SOURCE_DIR", str(Path(__file__).resolve().parents[1])), "rev-parse", "HEAD"], text=True).strip(),
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "traits": {arm: asdict(arm_traits(args.format, arm) if args.mode == "prefill" else decode_arm_traits(arm)) for arm in arms}, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pattern in args.scores:
        q = (torch.randn(max_rows, args.heads, 512, device=device) * (.03 if pattern == "spread" else .2)).bfloat16()
        if pattern == "peaked":
            target = ek[ei[:, 0].long()]
            q = (q.float() + target[:, None] * (14 * 512**.5 / target.square().sum(-1))[:, None, None]).bfloat16()
        out = {arm: torch.empty_like(q) for arm in arms}
        lse = {arm: torch.empty((max_rows, args.heads), device=device) for arm in arms}
        def run(arm, rows):
            # Native mixed KV needs more SMEM: use the existing single-group
            # specialization, since paired groups exceed SM120's per-CTA cap.
            partitions = [(1, args.heads, 0)] if args.format == "dsv41" else _mg_head_partitions(args.heads, 16)
            for n_hg, active_heads, offset in partitions:
                run_unified_prefill_mg(q=q[:rows], kv_cache=main_cache, topk_indices=mi[:rows],
                    topk_length=ml[:rows], sm_scale=512**-.5, page_block_size=main_page,
                    attn_sink=sink, output=out[arm][:rows], lse_out=lse[arm][:rows],
                    extra_kv_cache=extra_cache, extra_indices=ei[:rows], extra_topk_length=el[:rows],
                    extra_page_block_size=extra_page, mg_n_hg=n_hg, active_heads=active_heads,
                    head_offset=offset, traits_override=arm_traits(args.format, arm))
        # One FP32 oracle over the exact decoded cache; no cast back to BF16.
        ref = torch.empty_like(q, dtype=torch.float32)
        ref_lse = torch.empty_like(lse[arms[0]])
        pmax, entropy, tiny = [], [], []
        for lo in range(0, max_rows, 16):
            hi = min(lo + 16, max_rows)
            k = torch.cat((mk[mi[lo:hi].long()], ek[ei[lo:hi].long()]), 1)
            v = torch.cat((mv[mi[lo:hi].long()], ev[ei[lo:hi].long()]), 1)
            scores = torch.bmm(q[lo:hi].float(), k.transpose(1,2)) * 512**-.5
            all_scores = torch.cat((scores, sink[None,:,None].expand(hi-lo,-1,-1)), -1)
            probs = torch.softmax(all_scores, -1)
            ref[lo:hi] = torch.bmm(probs[:,:,:-1], v)
            ref_lse[lo:hi] = torch.logsumexp(all_scores, -1)
            pmax.append(probs.max(-1).values.mean())
            entropy.append(-(probs * probs.clamp_min(1e-30).log()).sum(-1).mean())
            tiny.append((probs[:,:,:-1] < 2**-9).float().mean())
        distribution = {"mean_pmax": float(torch.stack(pmax).mean()),
            "mean_entropy_nats": float(torch.stack(entropy).mean()),
            "probability_fraction_below_e4m3_min": float(torch.stack(tiny).mean())}
        fixed_runs, fixed_lses = build_decode(q,main_cache,extra_cache,mi,ei,sink,True) if args.native_row_curve else ({},{})
        full_outputs = {}
        if fixed_runs:
            for arm in arms:
                fixed_runs[arm][0](max_rows)
                full_outputs[arm] = fixed_runs[arm][1].clone()
        for rows in sorted(set(args.rows + args.timing_rows)):
            record = {"scores": pattern, "rows": rows, "distribution": distribution, "arms": {}}
            graphs = {}
            try:
                decode_runs, decode_lses = (fixed_runs,fixed_lses) if fixed_runs else (build_decode(q[:rows], main_cache, extra_cache, mi[:rows], ei[:rows], sink) if args.mode == "decode" else ({}, {}))
                def execute(arm):
                    if decode_runs:
                        decode_runs[arm][0](rows if fixed_runs else None)
                    else:
                        run(arm, rows)
                for arm in arms:
                    execute(arm)
                    torch.cuda.synchronize()
                    actual = decode_runs[arm][1][:rows] if decode_runs else out[arm][:rows]
                    actual_lse = torch.logaddexp(torch.logsumexp(decode_lses[arm][:rows] * math.log(2), -1), sink[None]) if decode_runs else lse[arm][:rows] * math.log(2)
                    if fixed_runs:
                        assert torch.equal(actual,full_outputs[arm][:rows]), (arm,rows,"row invariance")
                    delta = actual.float() - ref[:rows]
                    ld = actual_lse - ref_lse[:rows]
                    if not bool(torch.isfinite(delta).all() & torch.isfinite(ld).all()):
                        raise RuntimeError(f"nonfinite result: {arm}/{rows}/{pattern}")
                    record["arms"][arm] = {"output_rel_rms": float(delta.square().mean().sqrt() / ref[:rows].square().mean().sqrt()),
                        "output_max_abs": float(delta.abs().max()), "lse_max_abs": float(ld.abs().max()),
                        "lse_rms": float(ld.square().mean().sqrt())}
                    if rows in args.timing_rows:
                        for _ in range(args.warmup):
                            execute(arm)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            execute(arm)
                        graphs[arm] = graph
                        graph.replay()
                torch.cuda.synchronize()
                samples = {arm: [] for arm in arms}
                with Telemetry() as telemetry:
                    for sample in range(args.samples if graphs else 0):
                        for arm in arms[::1 if sample % 2 == 0 else -1]:
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(args.replays):
                                graphs[arm].replay()
                            end.record(); end.synchronize()
                            samples[arm].append(start.elapsed_time(end) * 1000 / args.replays)
                record["thermal"] = {"before":telemetry.before,"after":telemetry.after,"samples":telemetry.samples}
                for arm in graphs:
                    record["arms"][arm].update(samples_us=samples[arm], median_us=statistics.median(samples[arm]))
                if graphs:
                    record["second_over_first"] = record["arms"][arms[1]]["median_us"] / record["arms"][arms[0]]["median_us"]
            finally:
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
