"""Warmed row curve of the Inkling 2 MXFP8 W8A8 routed experts next to the GLM FP8 W8A8 stream.

Per routed row at the policy's live-row band (default 2048, 2752, 4096 rows,
top-8) for the `inkling2_mxfp8` stream program at a TP slice, and the
reference control `glmf` FP8 (E4M3 + FP32 128x128 block scales, W8A8 stream,
same 8 bpw weight traffic) at the same H x I when its geometry fits. Random
routing over every expert (the realistic screen: at 2752 rows every one of
the 384 experts is touched), one untimed launch, then the median of
``--samples`` CUDA-event timings with an L2 flush before each.

Memory floor per layer = all touched expert bytes / DRAM bandwidth; the
report prints the achieved weight GB/s so the bar (within 1.15x of the floor
on SM121; within 1.15x of GLM FP8 W8A8 per row at matched H x I) reads off.

    python benchmarks/bench_cuteafd_inkling2_mxfp8_moe.py --tp 1 --rows 2048,2752,4096 --samples 15 --json out.json
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


def _mxfp8_codes(x):
    blocks = x.float().reshape(-1, 32)
    raw = blocks.abs().amax(-1, keepdim=True) * (1.0 / 448.0)
    exponent = ((raw.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0 ** -127, torch.zeros_like(exponent), exponent.clamp(max=254))
    q = (blocks / torch.ldexp(torch.ones_like(raw), exponent - 127)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), exponent.to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 32)


def mxfp8_weights(g, seed: int):
    """E4M3 + UE8M0 weights quantized one expert at a time from a seeded generator: the full
    F32 tensor never exists (peak = quantized tensors + one expert slab)."""
    from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes

    h, i, e = g.hidden, g.slice, g.experts

    def operand(tag, rows, k, scale_cols=None):
        scale_cols = k // 32 if scale_cols is None else scale_cols
        w = torch.empty((e, rows, k), dtype=torch.float8_e4m3fn, device="cuda")
        s = torch.zeros((e, rows, scale_cols), dtype=torch.uint8, device="cuda")
        for expert in range(e):
            gen = torch.Generator(device="cuda").manual_seed(seed * 1000003 + tag * 100003 + expert)
            q, sc = _mxfp8_codes(torch.randn((rows, k), device="cuda", generator=gen) * 0.02)
            w[expert] = q
            s[expert, :, :k // 32] = sc
        return w.contiguous(), s.contiguous()

    w1, s1 = operand(1, i, h)
    w3, s3 = operand(3, i, h)
    w2, s2 = operand(2, h, i, scale_cols=mxfp4_scale_row_bytes(i))
    return (w1, s1, w3, s3, w2, s2)


def fp8_weights(g, seed: int):
    """FP8 block-scaled control weights, one expert at a time."""
    h, i, e = g.hidden, g.slice, g.experts
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = lambda rows, k: torch.empty((e, rows, k), dtype=torch.float8_e4m3fn, device="cuda")  # noqa: E731
    w1, w3, w2 = w(i, h), w(i, h), w(h, i)
    for expert in range(e):
        for t, (rows, k) in zip((w1, w3, w2), ((i, h), (i, h), (h, i))):
            t[expert] = (torch.randn((rows, k), device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    sc = lambda *shape: (torch.rand(shape, device="cuda", generator=gen) * 0.02 + 0.01).float()  # noqa: E731
    return (w1, sc(e, i // 128, h // 128), w3, sc(e, i // 128, h // 128), w2, sc(e, h // 128, i // 128))


def wire_rows(x):
    q, s = _mxfp8_codes(x)
    return torch.cat([q.view(torch.uint8), s], 1).contiguous()


def routing(rows, g, gen):
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    gammas = (torch.rand(rows, g.top_k, device="cuda", generator=gen) * 2).contiguous()
    return ids, gammas


def arm(g, weights, rows, capacity, samples, flush, gen):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    program = compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=True)
    x = wire_rows(torch.randn(rows, g.hidden, device="cuda", generator=gen))
    ids, gammas = routing(rows, g, gen)
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", capacity, True), dtype=torch.uint8, device="cuda")
    us = time_us(lambda: program.launch(x, ids, gammas, *weights, out, scratch, scalars=(rows,)), samples, flush)
    touched = int(ids.unique().numel())
    weight_bytes = sum(int(w.numel() * w.element_size()) for w in weights) * touched / g.experts
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
    g = GEOMETRIES["inkling2_mxfp8"].with_tp(args.tp)
    # Control: FP8 W8A8 stream at the same H x I x E x k (block scales need 128-aligned I).
    control = None
    if not args.no_control and g.slice % 128 == 0:
        control = Fp8MoeGeometry("inkling2_fp8_control", hidden=g.hidden, experts=g.experts, top_k=g.top_k,
                                 intermediate=g.intermediate, tp=args.tp)
    report = {"device": device.name, "sm": f"{device.major}{device.minor}", "host": platform.node(),
              "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tp": args.tp, "slice": g.slice, "rows": []}
    print(f"inkling2_mxfp8 tp{args.tp} slice {g.slice}; control: {'fp8 W8A8 stream' if control else 'none'}")
    print(f"{'rows':>6} {'mxfp8 us':>10} {'/row':>8} {'W GB/s':>7} {'fp8 us':>10} {'/row':>8} {'W GB/s':>7} {'ratio':>6}")
    row_list = [int(r) for r in args.rows.split(",")]
    # One arm resident at a time (two TP1 sets are ~25 GB together: too much for a 16g Spark
    # container): the full mxfp8 curve first, then free it, then the control's curve.
    weights = mxfp8_weights(g, 7)
    first = [arm(g, weights, rows, args.capacity, args.samples, flush, gen) for rows in row_list]
    del weights
    torch.cuda.empty_cache()
    second = []
    if control:
        control_weights = fp8_weights(control, 11)
        second = [arm(control, control_weights, rows, args.capacity, args.samples, flush, gen) for rows in row_list]
        del control_weights
        torch.cuda.empty_cache()
    for idx, rows in enumerate(row_list):
        a = first[idx]
        row = {"rows": rows, "mxfp8": a}
        line = f"{rows:>6} {a['us']:>10.1f} {a['us_per_row']:>8.3f} {a['weight_gbps']:>7.0f}"
        if control:
            b = second[idx]
            row["fp8_control"] = b
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
