"""Warmed row curve of the Inkling 2 dynamic-amax NVFP4 W4A4 experts next to the calibrated A4 stream.

Per routed row at 2048/2752/4096 rows (top-8, random routing over all 384 experts) for the
`inkling2_nvfp4` stream program at a TP slice, and the control `glmf_nvfp4`-style calibrated
A4 stream at the same H x I x E x k (`StreamNvfp4GateUpA4` + `StreamNvfp4LinearA4`: the same
GEMMs with a static input_scale and a 6-only quantizer). One untimed launch, then the median of
``--samples`` CUDA-event timings with an L2 flush before each; achieved weight GB/s printed.

    python benchmarks/bench_cuteafd_inkling2_nvfp4_moe.py --tp 1 --rows 2048,2752,4096 --samples 15 --json out.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import torch


def time_us(fn, iters: int, flush: torch.Tensor) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.max()
        torch.cuda._sleep(3_000_000)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def packed_weights(g, gen, dynamic: bool):
    """Random packed E2M1 codes and E4M3 block scales; tail = amax[E] + denominator (dynamic) or
    alpha[E] + input_scale[E] (calibrated). Timing only: values are not checked here."""
    from b12x.integration.cuteafd._nvfp4_moe_kernels import nvfp4_alpha_offset

    h, i, e = g.hidden, g.slice, g.experts

    def codes(*shape):
        return torch.randint(0, 256, shape, device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)

    def scales(*shape):
        return (torch.randint(4, 11, shape, device="cuda", generator=gen, dtype=torch.int32) * 8
                + torch.randint(0, 8, shape, device="cuda", generator=gen, dtype=torch.int32)).to(torch.uint8)

    def operand(rows, k):
        w, s = codes(e, rows, k // 2), scales(e, rows, k // 16)
        if dynamic:
            tail = torch.cat([(torch.rand(e, device="cuda", generator=gen) * 0.5 + 0.5).view(torch.uint8),
                              torch.tensor([1.0], device="cuda").view(torch.uint8)])
        else:
            tail = torch.cat([(torch.rand(e, device="cuda", generator=gen) * 4e-3 + 1e-3).view(torch.uint8),
                              torch.full((e,), 5.0 / (6 * 448), device="cuda").view(torch.uint8)])
        return w, torch.cat([s.flatten(), tail]).contiguous()

    w1, s1 = operand(i, h)
    w3, s3 = operand(i, h)
    w2, s2 = operand(h, i)
    return (w1, s1, w3, s3, w2, s2)


def routing(g, rows, seed):
    """One routing draw per row count, shared by both arms so they touch the same experts."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 7919 + rows)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen).bfloat16()
    ids = torch.rand(rows, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    gammas = (torch.rand(rows, g.top_k, device="cuda", generator=gen) * 2).contiguous()
    return x, ids, gammas


def arm(g, weights, rows, capacity, samples, flush, inputs):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    program = compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=False)
    x, ids, gammas = inputs
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", capacity, False), dtype=torch.uint8, device="cuda")
    us = time_us(lambda: program.launch(x, ids, gammas, *weights, out, scratch, scalars=(rows,)), samples, flush)
    touched = int(ids.unique().numel())
    weight_bytes = sum(int(w.numel()) for w in weights) * touched / g.experts
    return {"us": us, "us_per_row": us / rows, "touched_experts": touched,
            "weight_gbps": weight_bytes / (us * 1e-6) / 1e9}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--rows", default="2048,2752,4096")
    p.add_argument("--capacity", type=int, default=4096)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, Fp8MoeGeometry

    device = torch.cuda.get_device_properties(0)
    print(f"{device.name} sm_{device.major}{device.minor} {device.multi_processor_count} SMs, {platform.node()}")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(7)
    g = GEOMETRIES["inkling2_nvfp4"].with_tp(args.tp)
    control = None if args.no_control else Fp8MoeGeometry(
        "inkling2_nvfp4a4_control", hidden=g.hidden, experts=g.experts, top_k=g.top_k,
        intermediate=g.intermediate, tp=args.tp, weights="nvfp4", activations="a4")
    report = {"device": device.name, "sm": f"{device.major}{device.minor}", "host": platform.node(),
              "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tp": args.tp, "slice": g.slice, "rows": []}
    print(f"inkling2_nvfp4 (dynamic amax) tp{args.tp} slice {g.slice}; control: "
          f"{'calibrated A4 stream' if control else 'none'}")
    print(f"{'rows':>6} {'dyn us':>10} {'/row':>8} {'W GB/s':>7} {'a4 us':>10} {'/row':>8} {'W GB/s':>7} {'ratio':>6}")
    row_list = [int(r) for r in args.rows.split(",")]
    inputs = {rows: routing(g, rows, 7) for rows in row_list}
    # One arm resident at a time (two TP1 packed sets together exceed a 16g Spark container).
    weights = packed_weights(g, gen, dynamic=True)
    first = [arm(g, weights, rows, args.capacity, args.samples, flush, inputs[rows]) for rows in row_list]
    del weights
    torch.cuda.empty_cache()
    second = []
    if control:
        control_weights = packed_weights(control, gen, dynamic=False)
        second = [arm(control, control_weights, rows, args.capacity, args.samples, flush, inputs[rows])
                  for rows in row_list]
        del control_weights
        torch.cuda.empty_cache()
    for idx, rows in enumerate(row_list):
        a = first[idx]
        row = {"rows": rows, "dynamic": a}
        line = f"{rows:>6} {a['us']:>10.1f} {a['us_per_row']:>8.3f} {a['weight_gbps']:>7.0f}"
        if control:
            b = second[idx]
            row["a4_control"] = b
            row["ratio"] = a["us"] / b["us"]
            line += f" {b['us']:>10.1f} {b['us_per_row']:>8.3f} {b['weight_gbps']:>7.0f} {row['ratio']:>6.3f}"
        print(line)
        report["rows"].append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
