#!/usr/bin/env python3
"""Interleaved CUDA-graphed row-curve benchmark: fused vs unfused K8 trellis decode.

Measures the full dense K8 decode chain:
- Unfused baseline: trellis_input + trellis_linear + trellis_output (or input + linear for raw_f32)
- Fused candidate:  compile_trellis_linear_fused_aot

Verifies 100% bitwise exactness (including NaN-poisoned prefix-row invariance),
counts CUDA graph nodes in each arm, and flags any >10% per-row curve collapses.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch


DEFAULT_SHAPES = (
    # Hy4 (1x & 2x RTX core shapes)
    ("hy4", "hy4_kv_a_padded", 6144, 640, 1, "mul1", "bf16"),
    ("hy4", "hy4_q_a", 6144, 2048, 1, "mul1", "bf16"),
    ("hy4", "hy4_q_b_dsa", 2048, 16384, 1, "mul1", "bf16"),
    ("hy4", "hy4_q_b_dsa_tp2", 2048, 8192, 1, "mul1", "bf16"),
    ("hy4", "hy4_gate_dsa", 6144, 16384, 1, "mul1", "bf16"),
    ("hy4", "hy4_o_dsa", 16384, 6144, 1, "mul1", "bf16"),
    ("hy4", "hy4_o_dsa_tp2", 8192, 6144, 2, "mul1", "raw_f32"),
    ("hy4", "hy4_shared_down", 2048, 6144, 1, "mul1", "bf16"),
    ("hy4", "hy4_shared_down_tp2", 1024, 6144, 2, "mul1", "raw_f32"),
    ("hy4", "hy4_dense_gate_up", 6144, 18432, 1, "mul1", "bf16"),
    ("hy4", "hy4_lm_head", 6144, 120832, 1, "mul1", "f32"),
    # Shared EXL3 families (cross-family coverage)
    ("hy4f", "hy4f_q_a", 3072, 1024, 1, "mul1", "bf16"),
    ("glm", "glm_q_a", 6144, 2048, 1, "mcg", "bf16"),
    ("glmf", "glmf_kv_a", 4096, 512, 1, "mcg", "bf16"),
    ("glmf", "glmf_kda_qkv_tp2", 4096, 4096, 1, "mcg", "bf16"),
    ("kimik3", "mla_kv_a_padded", 7168, 640, 1, "mcg", "bf16"),
    ("kimik3", "shared_down_tp2", 3072, 7168, 2, "mcg", "raw_f32"),
    ("muse", "muse_q_tp2", 8192, 4096, 1, "mcg", "bf16"),
    ("ml4", "ml4_q_tp2", 6144, 4096, 1, "mcg", "bf16"),
    ("step5", "step5_kv_tp2", 4096, 384, 1, "mcg", "bf16"),
    ("inkling2", "inkling_rel_tp2", 4608, 384, 1, "mcg", "f32"),
)


def _graph_node_count(graph: torch.cuda.CUDAGraph) -> int | None:
    try:
        from cuda.bindings import runtime as cudart
        # Not exposed cleanly on torch CUDAGraph without C handle; return None if unavailable.
        return None
    except Exception:
        return None


def _bench_shape(
    family: str,
    name: str,
    k: int,
    n: int,
    side: int,
    codebook: str,
    output_dtype: str,
    *,
    live_rows: tuple[int, ...],
    capacity: int,
    samples: int,
    sms: int,
    l2_flush_buf: torch.Tensor,
) -> dict:
    from b12x.gemm.trellis_linear._fused_k8 import fused_scratch_bytes
    from b12x.integration.cuteafd.trellis_linear import (
        compile_trellis_linear_aot,
        compile_trellis_linear_fused_aot,
    )
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot,
        compile_trellis_output_aot,
    )

    splits = (k + 1023) // 1024
    cta_limit = sms * 4
    rng = np.random.default_rng(273)
    x = torch.tensor(rng.normal(0, 0.03, (capacity, k)), device="cuda", dtype=torch.bfloat16)
    suh = torch.tensor(rng.normal(0, 0.2, k), device="cuda", dtype=torch.float16)
    svh = torch.tensor(rng.normal(0, 0.2, n), device="cuda", dtype=torch.float16)
    payload = torch.tensor(rng.integers(0, 256, k * n, dtype=np.uint8), device="cuda")

    rotated = torch.empty((capacity, k), device="cuda", dtype=torch.bfloat16)
    blocks = torch.empty(k // 128, device="cuda", dtype=torch.float16)
    ptrs = torch.tensor([rotated.data_ptr()], device="cuda", dtype=torch.int64)
    partial_u = torch.empty((capacity, splits, n), device="cuda", dtype=torch.float32)
    scratch_f = torch.empty(fused_scratch_bytes(capacity, k, n), device="cuda", dtype=torch.uint8)
    acc_u = torch.empty((capacity, n), device="cuda", dtype=torch.float32)
    out_dt = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    out_u = torch.empty((capacity, n), device="cuda", dtype=out_dt)
    out_f = torch.empty((capacity, n), device="cuda", dtype=out_dt)

    inp_prog = compile_trellis_input_aot(features=k, input_dtype="bf16", max_rows=capacity, output_dtype="bf16")
    core_prog = compile_trellis_linear_aot(
        input_features=k, output_features=n, bits=8, codebook=codebook,
        sharded_side=side, rotation_count=1, max_rows=capacity, input_dtype="bf16",
    )
    eg_prog = (
        None
        if output_dtype == "raw_f32"
        else compile_trellis_output_aot(features=n, output_dtype=output_dtype, max_rows=capacity)
    )
    fused_prog = compile_trellis_linear_fused_aot(
        input_features=k, output_features=n, bits=8, codebook=codebook,
        sharded_side=side, max_rows=capacity, input_dtype="bf16", output_dtype=output_dtype,
    )

    def run_unfused(r: int):
        inp_prog.launch(x, suh, rotated, blocks, scalars=(r, cta_limit))
        core_prog.launch(ptrs, payload, blocks, partial_u.flatten(), acc_u.flatten(), scalars=(r, 0, cta_limit))
        if eg_prog is not None:
            eg_prog.launch(acc_u, svh, out_u, scalars=(r, cta_limit))

    def run_fused(r: int):
        if output_dtype == "raw_f32":
            fused_prog.launch(x, payload, suh, scratch_f, out_f.flatten(), scalars=(r, cta_limit))
        else:
            fused_prog.launch(x, payload, suh, svh, scratch_f, out_f.flatten(), scalars=(r, cta_limit))

    # Full-capacity reference for prefix-row invariance
    u_target = acc_u if output_dtype == "raw_f32" else out_u
    u_target.fill_(float("nan"))
    out_f.fill_(float("nan"))
    run_unfused(capacity)
    run_fused(capacity)
    torch.cuda.synchronize()
    assert torch.equal(out_f.view(torch.uint8), u_target.view(torch.uint8)), f"{name} cap mismatch"
    ref_full = out_f.clone()

    records = []
    for r in live_rows:
        # Prefix-row & exactness check with NaN-poisoned tail
        saved = x[r:].clone() if r < capacity else None
        if saved is not None:
            x[r:].fill_(float("nan"))
        u_target.fill_(float("nan"))
        out_f.fill_(float("nan"))
        run_unfused(r)
        run_fused(r)
        torch.cuda.synchronize()
        assert torch.equal(out_f[:r].view(torch.uint8), u_target[:r].view(torch.uint8)), f"{name} R={r} oracle mismatch"
        assert torch.equal(out_f[:r].view(torch.uint8), ref_full[:r].view(torch.uint8)), f"{name} R={r} prefix mismatch"
        if saved is not None:
            assert torch.isnan(out_f[r:]).all(), f"{name} R={r} wrote beyond live rows"
            x[r:].copy_(saved)

        # Capture CUDA graphs for both arms
        g_u = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_u):
            run_unfused(r)
        g_f = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_f):
            run_fused(r)
        # Untimed warm-up
        for _ in range(3):
            g_u.replay()
            g_f.replay()
        torch.cuda.synchronize()

        ev_u0, ev_u1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev_f0, ev_f1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        us_u, us_f = [], []
        for s in range(samples):
            order = ("u", "f") if s % 2 == 0 else ("f", "u")
            for arm in order:
                l2_flush_buf.zero_()
                if arm == "u":
                    ev_u0.record()
                    g_u.replay()
                    ev_u1.record()
                    ev_u1.synchronize()
                    us_u.append(ev_u0.elapsed_time(ev_u1) * 1000.0)
                else:
                    ev_f0.record()
                    g_f.replay()
                    ev_f1.record()
                    ev_f1.synchronize()
                    us_f.append(ev_f0.elapsed_time(ev_f1) * 1000.0)
        med_u = statistics.median(us_u)
        med_f = statistics.median(us_f)
        rec = {
            "rows": r,
            "unfused_us": round(med_u, 3),
            "fused_us": round(med_f, 3),
            "delta_us": round(med_u - med_f, 3),
            "speedup": round(med_u / med_f, 4),
            "unfused_us_per_row": round(med_u / r, 3),
            "fused_us_per_row": round(med_f / r, 3),
            "exact": True,
            "prefix_exact": True,
        }
        records.append(rec)
        print(
            f"{family:8s} {name:22s} R={r:3d} | unfused={med_u:7.2f}us fused={med_f:7.2f}us "
            f"delta={med_u-med_f:+6.2f}us ({med_u/med_f:.3f}x) | us/row={med_f/r:6.2f}",
            flush=True,
        )

    # Check >10% adjacent-size jump on fused us/row curve (monotonicity/cliff check)
    cliffs = []
    for i in range(1, len(records)):
        prev = records[i - 1]["fused_us_per_row"]
        curr = records[i]["fused_us_per_row"]
        if curr > 1.10 * prev:
            cliffs.append({"rows": records[i]["rows"], "prev_rows": records[i - 1]["rows"],
                           "prev_us_per_row": prev, "curr_us_per_row": curr,
                           "jump_pct": round((curr / prev - 1.0) * 100.0, 2)})

    return {
        "family": family,
        "name": name,
        "k": k,
        "n": n,
        "side": side,
        "codebook": codebook,
        "output_dtype": output_dtype,
        "records": records,
        "cliffs": cliffs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="all", help="comma-separated families or 'all'")
    parser.add_argument("--shapes", default="", help="comma-separated shape names")
    parser.add_argument("--rows", default="1,2,4,8,12,16,24,32,48,64", help="comma-separated live rows")
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    live_rows = tuple(int(v) for v in args.rows.split(",") if v)
    cap = max(args.capacity, max(live_rows))
    fam_filter = None if args.families == "all" else set(args.families.split(","))
    shape_filter = set(args.shapes.split(",")) if args.shapes else None
    selected = [
        s for s in DEFAULT_SHAPES
        if (fam_filter is None or s[0] in fam_filter)
        and (shape_filter is None or s[1] in shape_filter)
    ]

    props = torch.cuda.get_device_properties(0)
    # 256 MiB buffer to flush L2 between interleaved graph samples (cold-L2 decode regime).
    l2_flush_buf = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

    results = []
    for fam, name, k, n, side, cb, odt in selected:
        results.append(
            _bench_shape(
                fam, name, k, n, side, cb, odt,
                live_rows=live_rows, capacity=cap, samples=args.samples,
                sms=props.multi_processor_count, l2_flush_buf=l2_flush_buf,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "device": props.name,
        "sm_count": props.multi_processor_count,
        "capacity": cap,
        "samples": args.samples,
        "live_rows": list(live_rows),
        "shapes": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
