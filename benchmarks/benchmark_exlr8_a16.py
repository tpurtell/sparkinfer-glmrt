#!/usr/bin/env python3
"""Graph-first matched-byte EXLR8 up1 versus EXL3 K4 Spark-step gate.

Width arms come either from the synthetic all-K4 payloads or from a
checkpoint's own exlr8-v2 manifest (`--manifest NAME:LAYER`, one layer per
arm), which carries the real per-(expert, unit, projection) K2/K3/K4 split.
Throughput is reported against a measured read-only streaming roofline on the
payload bytes the kernel actually reads; the copy roofline counts read plus
write and stays for continuity.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import statistics
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import benchmark_mixed_trellis as exl3
from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot

HIDDEN, UNITS, TOPK, EXPERTS = 6144, 4, 8, 256
# Payload bytes per (expert, unit, projection) at width 1: 128 channels * hidden
# weights / 8 bits.
UNIT_BYTES = HIDDEN * 16


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def timing(graph, replays):
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end)*1000/replays)
    return statistics.median(values)


def copy_roofline():
    # Count both the read and write; expert payload throughput counts reads only.
    size = 256 * 1024 * 1024
    src = torch.empty(size, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    src.fill_(17)
    graph = capture(lambda: dst.copy_(src))
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(2 * size * 10 / (start.elapsed_time(end) * 1e6))
    return dict(buffer_bytes=size, gbps=max(values), repeats_gbps=values,
                convention="2N bytes per copy (read plus write); expert payload uses N read bytes")


def read_roofline():
    # Read-only stream: the reduction writes one scalar, so the whole cost is
    # streaming N bytes in. This is the denominator for payload throughput.
    size = 256 * 1024 * 1024
    src = torch.empty(size // 4, dtype=torch.float32, device="cuda").fill_(1.0)
    graph = capture(lambda: torch.sum(src))
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(size * 10 / (start.elapsed_time(end) * 1e6))
    return dict(buffer_bytes=size, gbps=max(values), repeats_gbps=values,
                convention="N read bytes per sum reduction; payload throughput uses N read bytes")


def finite_nonzero(tensor, label):
    if not bool(torch.isfinite(tensor).all()) or not bool(torch.count_nonzero(tensor)):
        raise RuntimeError(f"{label} output must be finite and nonzero")


def stage_profile(graph, directory):
    # Profiling is a separate replay: it never supplies the performance verdict.
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", dir=directory) as stream:
        prof.export_chrome_trace(stream.name)
        events = json.loads(Path(stream.name).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    stages = {}
    for stage in ("fc1", "activation", "fc2", "reduce"):
        matches = [e for e in kernels
                   if f"{stage}_kernel" in e.get("name", "")
                   or f"{stage}_balanced_kernel" in e.get("name", "")]
        if not matches:
            raise RuntimeError(f"profiler did not identify {stage} kernel")
        stages[stage+"_us"] = sum(e["dur"] for e in matches)
    span = max(e["ts"]+e["dur"] for e in kernels)-min(e["ts"] for e in kernels)
    stages["launch_gaps_us"] = max(0.0, span-sum(e["dur"] for e in kernels))
    stages["profiled_span_us"] = span
    return stages


def make_candidate(experts=EXPERTS, hidden=HIDDEN, units=UNITS, topk=TOPK):
    owners = []
    pointers = torch.empty((experts, units, 3, 3), dtype=torch.int64)
    for e in range(experts):
        for u in range(units):
            for p in range(3):
                tensors = (torch.randint(0, 256, (hidden*16*4,), dtype=torch.uint8, device="cuda"),
                           torch.ones(128, dtype=torch.float16, device="cuda"),
                           torch.full((hidden//128,), 1 / hidden**0.5, dtype=torch.float16, device="cuda"))
                owners.extend(tensors)
                pointers[e,u,p] = torch.tensor([t.data_ptr() for t in tensors])
    groups = 64*topk
    tensors = [torch.randn((64,hidden), dtype=torch.float16, device="cuda")*.02,
               pointers.to("cuda"), torch.full((experts,units,3),4,dtype=torch.int32,device="cuda"),
               torch.full((groups,),-1,dtype=torch.int32,device="cuda"),
               torch.full((groups,16),-1,dtype=torch.int32,device="cuda"),
               torch.full((64,topk),-1,dtype=torch.int32,device="cuda"),
               torch.full((64,topk),1/topk,device="cuda"),
               torch.empty((groups,units,2,(hidden+1023)//1024,16,128),device="cuda"),
               torch.empty((groups,units,16,128),dtype=torch.float16,device="cuda"),
               torch.empty((groups,16,hidden),device="cuda"),
               torch.empty((64,hidden),device="cuda")]
    return tensors, owners


def manifest_widths(path, layer, rank, ranks=4, total_units=16, experts=EXPERTS, units=UNITS):
    """Per-(expert, unit, projection) widths for one rank's owned units.

    Mirrors the loader's TP ownership rule (total_units * expert + unit +
    layer) % ranks == rank, with the rank's units in ascending order.
    """
    document = json.loads(Path(path).read_text())
    entry = document["layers"][str(layer)]
    gate, up, down = entry["gate"], entry["up"], entry["down"]
    if len(gate) != experts or len(down) != experts:
        raise SystemExit(f"{path}: layer {layer} does not cover {experts} experts")
    bits = np.zeros((experts, units, 3), dtype=np.int32)
    for e in range(experts):
        owned = [u for u in range(total_units) if (total_units*e + u + layer) % ranks == rank]
        if len(owned) != units:
            raise SystemExit(f"{path}: expert {e} owns {len(owned)} units, expected {units}")
        for j, u in enumerate(owned):
            bits[e, j, 0] = gate[e][u]
            bits[e, j, 1] = up[e][u]
            bits[e, j, 2] = down[e] if not isinstance(down[e], list) else down[e][u]
    return bits, f"{document.get('name', Path(path).stem)}:{layer}"


def read_accounting(bits, experts):
    """Payload bytes the kernel reads: gate/up pairs and down, per expert."""
    selection = bits[np.asarray(experts, dtype=np.int64)]
    pairs = int(selection[:, :, :2].astype(np.int64).sum())*UNIT_BYTES
    down = int(selection[:, :, 2].astype(np.int64).sum())*UNIT_BYTES
    return dict(payload_bytes=pairs+down, fc1_payload_bytes=pairs, fc2_payload_bytes=down)


def pack(tensors, ids):
    slots = ids.cpu().numpy()
    expert, routes = [], []
    inverse = np.full(slots.shape,-1,dtype=np.int32)
    for e in sorted(set(slots.flat)):
        positions = np.flatnonzero(slots.ravel()==e)
        for start in range(0,len(positions),16):
            subset = positions[start:start+16]
            g = len(expert)
            expert.append(int(e))
            routes.append(list(subset)+[-1]*(16-len(subset)))
            inverse.flat[subset] = g*16+np.arange(len(subset))
    tensors[3][:len(expert)] = torch.tensor(expert,dtype=torch.int32,device="cuda")
    tensors[4][:len(expert)] = torch.tensor(routes,dtype=torch.int32,device="cuda")
    tensors[5][:len(ids)] = torch.from_numpy(inverse).cuda()
    return len(expert)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows",default="1,2,3,4,5,6,7,8,16,64")
    parser.add_argument("--replays",type=int,default=100)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--cta-mult",default="",
                        help="EXLR8 cta_limit = SMs * mult (default: the program's ctas_per_sm); "
                             "a comma list times extra arms, the first is the verdict arm")
    parser.add_argument("--manifests",type=Path,
                        help="directory holding the checkpoint's exlr8-manifest-*.json files")
    parser.add_argument("--manifest",default="",
                        help="comma list of NAME:LAYER width arms read from --manifests "
                             "(e.g. home:40,bit325:40); default: the synthetic all-K4 arm")
    parser.add_argument("--rank",type=int,default=0,help="TP rank whose owned units the arm uses")
    parser.add_argument("--ranks",type=int,default=4)
    parser.add_argument("--total-units",type=int,default=16)
    parser.add_argument("--schedules",default="static,balanced",
                        help="EXLR8 width schedules to measure (static, balanced, identity)")
    parser.add_argument("--stage-rows",default="8,16,64",
                        help="rows whose arms also get a stage profile; empty disables")
    parser.add_argument("--profile",action="store_true",
                        help="one eager launch per arm inside cudaProfilerStart/Stop (for ncu); no timing")
    args = parser.parse_args()
    if not args.profile and args.output is None:
        parser.error("--output is required unless --profile")
    if torch.cuda.get_device_capability() != (12,1):
        parser.error("this performance gate requires SM121")
    props = torch.cuda.get_device_properties(0)
    if not args.profile:
        args.output.parent.mkdir(parents=True,exist_ok=True)
    roofline = None if args.profile else copy_roofline()
    streaming = None if args.profile else read_roofline()
    stage_rows = {int(r) for r in args.stage_rows.split(",") if r} if args.stage_rows else set()
    schedules = [s for s in args.schedules.split(",") if s]
    for schedule in schedules:
        if schedule not in ("static", "balanced", "identity"):
            parser.error(f"unknown schedule {schedule}")
    tensors, owners = make_candidate()
    arms = [("k4", None)]
    if args.manifest:
        if args.manifests is None:
            parser.error("--manifest requires --manifests")
        for spec in args.manifest.split(","):
            name, layer = spec.rsplit(":", 1)
            bits, label = manifest_widths(args.manifests/f"exlr8-manifest-{name}.json",
                                          int(layer), args.rank, args.ranks, args.total_units)
            arms.append((label, bits))
    candidates = {schedule: compile_exlr8_a16_aot(units=UNITS, schedule=schedule)
                  for schedule in schedules}
    # This is the existing EXL3 uniform-K4 implementation, not a proxy GEMM.
    tier = exl3._prepared(4,256,408,torch.device("cuda"))
    # The legacy helper predates the compiler's codebook default change.
    exl3.compile_w4a16_fused_moe = functools.partial(
        exl3.compile_w4a16_fused_moe, trellis_codebook=tier.trellis_codebook)
    expert_map = torch.arange(EXPERTS,dtype=torch.int32,device="cuda")
    records = []
    for label, bits in arms:
        host_bits = (torch.full((EXPERTS,UNITS,3),4,dtype=torch.int32) if bits is None
                     else torch.from_numpy(bits)).numpy()
        tensors[2].copy_(torch.from_numpy(host_bits).cuda())
        for rows in map(int,args.rows.split(",")):
            gen = torch.Generator().manual_seed(408+rows)
            ids = torch.stack([torch.randperm(EXPERTS,generator=gen)[:TOPK] for _ in range(rows)]).cuda().int()
            groups = pack(tensors,ids)
            unique = ids.unique().cpu().numpy()
            accounting = read_accounting(host_bits, unique)
            payload_bytes = accounting["payload_bytes"]
            mults = [int(m) for m in args.cta_mult.split(",")] if args.cta_mult else [candidates[schedules[0]].geometry["ctas_per_sm"]]
            cta_limit = props.multi_processor_count*mults[0]
            control_us = None
            if label == "k4" and not args.profile:
                state = (tier,*exl3._serial_state(rows,tier,expert_map,props,torch.device("cuda")))
                baseline_x = tensors[0][:rows].to(torch.bfloat16)
                baseline = lambda: exl3._run_serial(state,baseline_x,tensors[6][:rows],ids)
                baseline_graph = capture(baseline)
            for schedule in schedules:
                candidate = candidates[schedule]
                launch = lambda: candidate.launch(*(t.flatten() for t in tensors),scalars=(rows,groups,cta_limit))
                if args.profile:
                    for _ in range(3):
                        launch()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    launch()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    finite_nonzero(tensors[-1][:rows], "EXLR8")
                    print(f"profiled widths={label} schedule={schedule} rows={rows} "
                          f"groups={groups} unique={len(unique)}", flush=True)
                    continue
                extra = {}
                candidate_graph = capture(launch)
                for mult in mults[1:]:
                    limit = props.multi_processor_count*mult
                    g = capture(lambda: candidate.launch(*(t.flatten() for t in tensors),scalars=(rows,groups,limit)))
                    extra[f"exlr8_us_cta{mult}x"] = timing(g,args.replays)
                candidate_graph.replay()
                finite_nonzero(tensors[-1][:rows], "EXLR8")
                candidate_us = timing(candidate_graph,args.replays)
                stages = stage_profile(candidate_graph,args.output.parent) if rows in stage_rows else {}
                record = dict(rows=rows,groups=groups,widths=label,schedule=schedule,
                              unique_experts=int(len(unique)),payload_bytes=payload_bytes,
                              fc1_payload_bytes=accounting["fc1_payload_bytes"],
                              fc2_payload_bytes=accounting["fc2_payload_bytes"],
                              exlr8_up1_us=candidate_us,
                              exlr8_gbps=payload_bytes/candidate_us/1000,
                              read_roofline_percent=100*payload_bytes/candidate_us/1000/streaming["gbps"],
                              copy_roofline_percent=100*payload_bytes/candidate_us/1000/roofline["gbps"],
                              cta_mult=mults[0],**extra)
                if stages:
                    for stage in ("fc1", "fc2"):
                        record[stage+"_payload_gbps"] = record[stage+"_payload_bytes"]/stages[stage+"_us"]/1000
                        record[stage+"_read_roofline_percent"] = 100*record[stage+"_payload_gbps"]/streaming["gbps"]
                    record.update(stages)
                if label == "k4":
                    if control_us is None:
                        baseline_graph.replay()
                        finite_nonzero(state[1].output, "EXL3")
                        control_us = timing(baseline_graph,args.replays)
                    record.update(exl3_k4_us=control_us,speedup=control_us/candidate_us,
                                  exl3_gbps=payload_bytes/control_us/1000,
                                  optimization_target=0.97,
                                  target_met=control_us/candidate_us>=0.97)
                records.append(record)
    if args.profile:
        return
    verdict_schedule = "balanced" if "balanced" in schedules else schedules[0]
    by_schedule = {s: all(r["target_met"] for r in records
                           if r["widths"] == "k4" and r["schedule"] == s)
                   for s in schedules}
    result = dict(schema=1,kind="exlr8_widths_roofline",device=props.name,
                  sm_count=props.multi_processor_count,l2_bytes=getattr(props,"L2_cache_size",None),
                  hidden=HIDDEN,units_per_rank=UNITS,top_k=TOPK,experts=EXPERTS,
                  width_arms=[label for label, _ in arms],schedules=schedules,
                  measured_copy_roofline=roofline,measured_read_roofline=streaming,
                  conditions="TP4 synthetic payloads with manifest width patterns, graph replay, "
                             "one untimed warm batch per rows; profiler separate; read percentages "
                             "are payload bytes only (scratch traffic is excluded)",
                  rows=records,
                  target_met=by_schedule.get(verdict_schedule, False),
                  target_met_by_schedule=by_schedule)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
