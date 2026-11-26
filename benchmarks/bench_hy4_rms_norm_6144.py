#!/usr/bin/env python3
"""Interleaved CUDA-graphed row-curve benchmark: Hy4 width=6144 BF16 RMSNorm (1-warp vs 768-thread)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64,256,2048,2752,4096")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd.pointwise import AffineNorm, compile_affine_norm_aot

    live_rows = [int(v) for v in args.rows.split(",") if v]
    cap = max(live_rows)
    width = 6144
    torch.manual_seed(273)
    x = torch.randn((cap, width), device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.5 + 1.0
    dummy_bf16 = torch.empty(16, device="cuda", dtype=torch.bfloat16)
    dummy_i64 = torch.zeros(cap, device="cuda", dtype=torch.int64)
    out_b = torch.empty_like(x)
    out_c = torch.empty_like(x)

    cand = compile_affine_norm_aot(
        width=width, heads=1, eps=1e-5,
        input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16,
    )
    base = compile_program(
        AffineNorm(width, 1, 1e-5, False, False, False, 0),
        name="affine_norm_legacy_6144",
        operands=(
            Operand("x", torch.bfloat16, f"[rows,1,{width}]"),
            Operand("weight", torch.bfloat16, f"[{width}]"),
            Operand("bias", torch.bfloat16, f"[{width}] (ignored)"),
            Operand("cos", torch.bfloat16, "[positions,0] (ignored)"),
            Operand("sin", torch.bfloat16, "[positions,0] (ignored)"),
            Operand("positions", torch.int64, "[rows] (ignored)"),
            Operand("out", torch.bfloat16, f"[rows,1,{width}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=("legacy", width),
        geometry=dict(width=width, heads=1, eps=1e-5),
    )

    l2 = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    records = []
    for r in live_rows:
        g_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_b):
            base.launch(x, w, dummy_bf16, dummy_bf16, dummy_bf16, dummy_i64, out_b, scalars=(r,))
        g_c = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_c):
            cand.launch(x, w, dummy_bf16, dummy_bf16, dummy_bf16, dummy_i64, out_c, scalars=(r,))
        for _ in range(3):
            g_b.replay()
            g_c.replay()
        torch.cuda.synchronize()
        max_abs = (out_b[:r].float() - out_c[:r].float()).abs().max().item()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tb, tc = [], []
        for s in range(args.samples):
            for arm in (("b", "c") if s % 2 == 0 else ("c", "b")):
                l2.zero_()
                e0.record()
                (g_b if arm == "b" else g_c).replay()
                e1.record()
                e1.synchronize()
                (tb if arm == "b" else tc).append(e0.elapsed_time(e1) * 1000.0)
        mb, mc = statistics.median(tb), statistics.median(tc)
        rec = dict(
            rows=r,
            legacy_us=round(mb, 3),
            wide_us=round(mc, 3),
            speedup=round(mb / mc, 3),
            us_per_row=round(mc / r, 4),
            max_abs_vs_legacy=round(max_abs, 6),
        )
        records.append(rec)
        print(
            f"R={r:4d} | legacy={mb:7.2f}us wide={mc:6.2f}us ({mb/mc:5.2f}x) | us/row={mc/r:6.4f} | max_abs={max_abs:.5f}",
            flush=True,
        )

    props = torch.cuda.get_device_properties(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "device": props.name,
        "sm_count": props.multi_processor_count,
        "width": width,
        "samples": args.samples,
        "records": records,
    }, indent=2) + "\n")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
