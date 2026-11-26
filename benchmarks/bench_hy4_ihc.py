#!/usr/bin/env python3
"""Benchmark, exact-route verifier, and prefix-row-invariance checker for Hy4 iHC (H=6144 / H=3072).

Measures warmed L2-cold CUDA-graph timings across the decode (1..64) and
prefill policy band (2048, 2688, 2752, 4096) for:
  - ihc_pre
  - ihc_post_pre
  - ihc_post
and whole-round chain totals (78 layers for H=6144, 42 layers for H=3072), both
stock 3-call/layer (78*pre + 78*post_pre + 78*post) and fused 2-call/layer
(1*pre + 155*post_pre + 1*post).

Supports AGENTS.md / hy4-c1-common.md flags:
  --capacity N             allocate buffers for capacity N (default 4096)
  --routes LIST            comma list of routes: tiled,legacy (default tiled,legacy)
  --compare-routes-exact   verify ihc_post_pre == ihc_post + ihc_pre bit-exact and
                           verify tiled against IEEE F64 reference
  --check-prefix-rows      verify every row in every batch size (1..64 and policy band)
                           is bit-exact with M=1 (and poisoned inactive tail untouched)
"""
from __future__ import annotations

import argparse
import json
import statistics

import torch

from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd.hy4_ihc import (
    HY4_FLASH_IHC,
    HY4_IHC,
    Hy4IhcGeometry,
    _Hy4IhcPostKernel6144,
    _Hy4IhcPreKernel6144,
    _Hy4IhcPreStandalone6144,
    compile_hy4_ihc_post_aot,
    compile_hy4_ihc_post_pre_aot,
    compile_hy4_ihc_pre_aot,
)


def _compile_legacy_programs(g: Hy4IhcGeometry):
    h = g.hidden
    pre = compile_program(
        _Hy4IhcPreStandalone6144(g),
        name="hy4_ihc_pre_legacy",
        operands=(
            Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
            Operand("fn", torch.float32, f"[8,{4 * h}]"),
            Operand("scale", torch.float32, "[2]", align=4),
            Operand("base", torch.float32, "[8]", align=4),
            Operand("norm", torch.bfloat16, f"[{h}]"),
            Operand("post_out", torch.float32, "[rows,4]", "out"),
            Operand("y_out", torch.bfloat16, f"[rows,{h}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(g, "pre_legacy"),
        geometry=vars(g),
    )
    post_pre = compile_program(
        _Hy4IhcPreKernel6144(g, fuse_post=True),
        name="hy4_ihc_post_pre_legacy",
        operands=(
            Operand("delta", torch.bfloat16, f"[rows,{h}]"),
            Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
            Operand("prev_post", torch.float32, "[rows,4]", align=4),
            Operand("fn", torch.float32, f"[8,{4 * h}]"),
            Operand("scale", torch.float32, "[2]", align=4),
            Operand("base", torch.float32, "[8]", align=4),
            Operand("norm", torch.bfloat16, f"[{h}]"),
            Operand("residual_out", torch.bfloat16, f"[rows,4,{h}]", "out"),
            Operand("post_out", torch.float32, "[rows,4]", "out"),
            Operand("y_out", torch.bfloat16, f"[rows,{h}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(g, "post_pre_legacy"),
        geometry=vars(g),
    )
    post = compile_program(
        _Hy4IhcPostKernel6144(g),
        name="hy4_ihc_post_legacy",
        operands=(
            Operand("delta", torch.bfloat16, f"[rows,{h}]"),
            Operand("residual", torch.bfloat16, f"[rows,4,{h}]"),
            Operand("prev_post", torch.float32, "[rows,4]", align=4),
            Operand("out", torch.bfloat16, f"[rows,4,{h}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(g, "post_legacy"),
        geometry=vars(g),
    )
    return {"pre": pre, "post_pre": post_pre, "post": post}


def _compile_tiled_programs(g: Hy4IhcGeometry):
    return {
        "pre": compile_hy4_ihc_pre_aot(g),
        "post_pre": compile_hy4_ihc_post_pre_aot(g),
        "post": compile_hy4_ihc_post_aot(g),
    }


def _time_graph_cold(launch_n, n_calls: int, flush_buf, stream, *, repeats: int) -> float:
    """Time `n_calls` distinct-weight launches inside one CUDA graph after a 128 MB L2 flush."""
    with torch.cuda.stream(stream):
        for _ in range(2):
            launch_n()
    stream.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        launch_n()

    samples = []
    with torch.cuda.stream(stream):
        for _ in range(2):
            flush_buf.zero_()
            g.replay()
        stream.synchronize()
        for _ in range(repeats):
            flush_buf.zero_()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record(stream)
            g.replay()
            e_ev.record(stream)
            e_ev.synchronize()
            samples.append(s_ev.elapsed_time(e_ev) * 1000.0 / n_calls)
    return statistics.median(samples)


def _check_exact_and_prefix(progs, g: Hy4IhcGeometry, cap: int, check_rows: list[int], dev):
    h = g.hidden
    torch.manual_seed(2026)
    r = torch.randn(cap, 4, h, dtype=torch.bfloat16, device=dev) * 0.5
    delta = torch.randn(cap, h, dtype=torch.bfloat16, device=dev) * 0.5
    prev_post = torch.rand(cap, 4, dtype=torch.float32, device=dev) + 0.5
    fn = torch.randn(8, 4 * h, dtype=torch.float32, device=dev) * 0.015
    scale = torch.tensor([0.3, 0.7], dtype=torch.float32, device=dev)
    base = torch.randn(8, dtype=torch.float32, device=dev) * 0.2
    norm = torch.randn(h, dtype=torch.bfloat16, device=dev) + 1.0

    m1_rows = min(64, cap)
    ref_pre_post = torch.empty(m1_rows, 4, dtype=torch.float32, device=dev)
    ref_pre_y = torch.empty(m1_rows, h, dtype=torch.bfloat16, device=dev)
    ref_post_r = torch.empty(m1_rows, 4, h, dtype=torch.bfloat16, device=dev)
    ref_pp_r = torch.empty(m1_rows, 4, h, dtype=torch.bfloat16, device=dev)
    ref_pp_post = torch.empty(m1_rows, 4, dtype=torch.float32, device=dev)
    ref_pp_y = torch.empty(m1_rows, h, dtype=torch.bfloat16, device=dev)

    p1 = torch.empty(1, 4, dtype=torch.float32, device=dev)
    y1 = torch.empty(1, h, dtype=torch.bfloat16, device=dev)
    r1 = torch.empty(1, 4, h, dtype=torch.bfloat16, device=dev)
    for i in range(m1_rows):
        progs["pre"].launch(r[i : i + 1].contiguous(), fn, scale, base, norm, p1, y1, scalars=(1,))
        ref_pre_post[i : i + 1].copy_(p1)
        ref_pre_y[i : i + 1].copy_(y1)

        progs["post"].launch(
            delta[i : i + 1].contiguous(), r[i : i + 1].contiguous(), prev_post[i : i + 1].contiguous(), r1, scalars=(1,)
        )
        ref_post_r[i : i + 1].copy_(r1)

        progs["post_pre"].launch(
            delta[i : i + 1].contiguous(),
            r[i : i + 1].contiguous(),
            prev_post[i : i + 1].contiguous(),
            fn, scale, base, norm, r1, p1, y1,
            scalars=(1,),
        )
        ref_pp_r[i : i + 1].copy_(r1)
        ref_pp_post[i : i + 1].copy_(p1)
        ref_pp_y[i : i + 1].copy_(y1)
    torch.cuda.synchronize()

    r_out = torch.empty(cap, 4, h, dtype=torch.bfloat16, device=dev)
    post_out = torch.empty(cap, 4, dtype=torch.float32, device=dev)
    y_out = torch.empty(cap, h, dtype=torch.bfloat16, device=dev)
    r_post = torch.empty(cap, 4, h, dtype=torch.bfloat16, device=dev)
    post_seq = torch.empty(cap, 4, dtype=torch.float32, device=dev)
    y_seq = torch.empty(cap, h, dtype=torch.bfloat16, device=dev)

    poison_bf16 = torch.tensor(0x7F80, dtype=torch.uint16, device=dev).view(torch.bfloat16)
    poison_f32 = float("nan")

    for rows in check_rows:
        if rows > cap:
            continue
        r_out.fill_(poison_bf16)
        post_out.fill_(poison_f32)
        y_out.fill_(poison_bf16)
        progs["pre"].launch(r, fn, scale, base, norm, post_out, y_out, scalars=(rows,))
        torch.cuda.synchronize()
        k = min(rows, m1_rows)
        assert torch.equal(post_out[:k], ref_pre_post[:k]), f"pre post_out mismatch vs M=1 at rows={rows}"
        assert torch.equal(y_out[:k], ref_pre_y[:k]), f"pre y_out mismatch vs M=1 at rows={rows}"
        if rows < cap:
            assert torch.isnan(post_out[rows:]).all(), f"pre clobbered tail post_out at rows={rows}"
            assert (y_out[rows:].view(torch.uint16) == 0x7F80).all(), f"pre clobbered tail y_out at rows={rows}"

        r_post.fill_(poison_bf16)
        progs["post"].launch(delta, r, prev_post, r_post, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(r_post[:k], ref_post_r[:k]), f"post mismatch vs M=1 at rows={rows}"
        if rows < cap:
            assert (r_post[rows:].view(torch.uint16) == 0x7F80).all(), f"post clobbered tail at rows={rows}"

        r_out.fill_(poison_bf16)
        post_out.fill_(poison_f32)
        y_out.fill_(poison_bf16)
        progs["post_pre"].launch(delta, r, prev_post, fn, scale, base, norm, r_out, post_out, y_out, scalars=(rows,))
        progs["pre"].launch(r_post, fn, scale, base, norm, post_seq, y_seq, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(r_out[:k], ref_pp_r[:k]), f"post_pre r_out mismatch vs M=1 at rows={rows}"
        assert torch.equal(post_out[:k], ref_pp_post[:k]), f"post_pre post_out mismatch vs M=1 at rows={rows}"
        assert torch.equal(y_out[:k], ref_pp_y[:k]), f"post_pre y_out mismatch vs M=1 at rows={rows}"
        assert torch.equal(r_out[:rows], r_post[:rows]), f"post_pre r_out != post at rows={rows}"
        assert torch.equal(post_out[:rows], post_seq[:rows]), f"post_pre post_out != post+pre at rows={rows}"
        assert torch.equal(y_out[:rows], y_seq[:rows]), f"post_pre y_out != post+pre at rows={rows}"
        if rows < cap:
            assert (r_out[rows:].view(torch.uint16) == 0x7F80).all(), f"post_pre clobbered tail r_out at rows={rows}"
            assert torch.isnan(post_out[rows:]).all(), f"post_pre clobbered tail post_out at rows={rows}"
            assert (y_out[rows:].view(torch.uint16) == 0x7F80).all(), f"post_pre clobbered tail y_out at rows={rows}"

    for rows in (1, min(64, cap)):
        flat64 = r[:rows].double().reshape(rows, 4 * h)
        inv = torch.rsqrt(flat64.square().mean(-1, keepdim=True) + 1e-5)
        mix = (flat64 @ fn.double().T) * inv
        pre_ref = (torch.sigmoid(mix[:, :4] * scale[0].double() + base[:4].double()) + 1e-6).float()
        post_ref = (2.0 * torch.sigmoid(mix[:, 4:] * scale[1].double() + base[4:].double()) + 1e-6).float()
        col_ref = (r[:rows].float() * pre_ref.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
        xf = col_ref.float()
        y_ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5) * norm.float()).to(torch.bfloat16)

        progs["pre"].launch(r, fn, scale, base, norm, post_out, y_out, scalars=(rows,))
        torch.cuda.synchronize()
        max_post = (post_out[:rows] - post_ref).abs().max().item()
        max_y = (y_out[:rows].float() - y_ref.float()).abs().max().item()
        assert max_post <= 1e-5, f"F64 oracle post err {max_post} at rows={rows}"
        assert max_y <= 0.03, f"F64 oracle y err {max_y} at rows={rows}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=6144, choices=(3072, 6144))
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--routes", type=str, default="tiled,legacy")
    parser.add_argument("--rows", type=str, default="1,2,3,6,7,8,9,16,32,64,2048,2688,2752,4096")
    parser.add_argument("--compare-routes-exact", action="store_true")
    parser.add_argument("--check-prefix-rows", action="store_true")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    g = HY4_IHC if args.hidden == 6144 else HY4_FLASH_IHC
    layers = 78 if args.hidden == 6144 else 42
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    row_list = [int(x) for x in args.rows.split(",") if x.strip()]

    compiled = {}
    for route in routes:
        if route == "tiled":
            compiled[route] = _compile_tiled_programs(g)
        elif route == "legacy":
            compiled[route] = _compile_legacy_programs(g)
        else:
            raise ValueError(f"unknown route {route!r}")

    if args.compare_routes_exact or args.check_prefix_rows:
        check_rows = list(range(1, min(65, args.capacity + 1))) + [
            r for r in row_list if r > 64 and r <= args.capacity
        ]
        for route in routes:
            if route == "tiled":
                _check_exact_and_prefix(compiled[route], g, args.capacity, check_rows, dev)
                print(
                    f"[PASS] {route} H={g.hidden}: exact oracle + post_pre==post+pre + "
                    f"prefix row-invariance across {len(check_rows)} row counts (1..64 + band)",
                    flush=True,
                )

    cap = max(args.capacity, max(row_list))
    h = g.hidden
    # Use 16 distinct weight sets so back-to-back calls inside a cold-L2 graph hit DRAM
    # for weights (matching real layer-by-layer execution) while giving 16x sub-tick resolution.
    n_cold = 16
    delta = torch.randn(cap, h, dtype=torch.bfloat16, device=dev)
    resid = torch.randn(cap, 4, h, dtype=torch.bfloat16, device=dev)
    prev_p = torch.ones(cap, 4, dtype=torch.float32, device=dev)
    fns = [torch.randn(8, 4 * h, dtype=torch.float32, device=dev) * 0.01 for _ in range(n_cold)]
    scs = [torch.ones(2, dtype=torch.float32, device=dev) for _ in range(n_cold)]
    bss = [torch.zeros(8, dtype=torch.float32, device=dev) for _ in range(n_cold)]
    nms = [torch.ones(h, dtype=torch.bfloat16, device=dev) for _ in range(n_cold)]
    resid_o = torch.empty_like(resid)
    post_o = torch.empty_like(prev_p)
    y_o = torch.empty_like(delta)

    # 128 MB L2 flush buffer (larger than SM120 96 MB L2)
    flush_buf = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=dev)
    stream = torch.cuda.Stream(device=dev)

    all_results = {}
    for route in routes:
        progs = compiled[route]
        rows_out = []
        for rows in row_list:
            nc = n_cold if rows <= 64 else 1

            def run_pre(r=rows, k=nc, p=progs["pre"]):
                for i in range(k):
                    p.launch(resid, fns[i], scs[i], bss[i], nms[i], post_o, y_o, scalars=(r,))

            def run_pp(r=rows, k=nc, p=progs["post_pre"]):
                for i in range(k):
                    p.launch(delta, resid, prev_p, fns[i], scs[i], bss[i], nms[i], resid_o, post_o, y_o, scalars=(r,))

            def run_post(r=rows, k=nc, p=progs["post"]):
                for i in range(k):
                    p.launch(delta, resid, prev_p, resid_o, scalars=(r,))

            t_pre = _time_graph_cold(run_pre, nc, flush_buf, stream, repeats=args.repeats)
            t_pp = _time_graph_cold(run_pp, nc, flush_buf, stream, repeats=args.repeats)
            t_post = _time_graph_cold(run_post, nc, flush_buf, stream, repeats=args.repeats)
            unfused_ms = layers * (t_pre + t_pp + t_post) / 1000.0
            fused_ms = (t_pre + (2 * layers - 1) * t_pp + t_post) / 1000.0
            entry = {
                "route": route,
                "rows": rows,
                "pre_us": round(t_pre, 2),
                "post_pre_us": round(t_pp, 2),
                "post_us": round(t_post, 2),
                "pre_us_per_row": round(t_pre / rows, 4),
                "post_pre_us_per_row": round(t_pp / rows, 4),
                "post_us_per_row": round(t_post / rows, 4),
                "round_unfused_ms": round(unfused_ms, 3),
                "round_fused_ms": round(fused_ms, 3),
            }
            rows_out.append(entry)
            print(json.dumps(entry), flush=True)
        all_results[route] = rows_out

    if "tiled" in all_results and "legacy" in all_results:
        print("\n=== Summary (tiled vs legacy, H=%d, %d layers) ===" % (h, layers), flush=True)
        print(
            f"{'rows':>5} | {'pre (us)':>13} | {'post_pre (us)':>13} | {'post (us)':>13} | "
            f"{'unfused (ms)':>15} | {'fused (ms)':>15} | {'speedup':>13}",
            flush=True,
        )
        for t, l in zip(all_results["tiled"], all_results["legacy"]):
            sp_unfused = l["round_unfused_ms"] / max(t["round_unfused_ms"], 1e-6)
            sp_fused = l["round_unfused_ms"] / max(t["round_fused_ms"], 1e-6)
            print(
                f"{t['rows']:>5} | {t['pre_us']:>5.2f}/{l['pre_us']:<6.2f} | "
                f"{t['post_pre_us']:>5.2f}/{l['post_pre_us']:<6.2f} | "
                f"{t['post_us']:>5.2f}/{l['post_us']:<6.2f} | "
                f"{t['round_unfused_ms']:>6.3f}/{l['round_unfused_ms']:<7.3f} | "
                f"{t['round_fused_ms']:>6.3f}/{l['round_unfused_ms']:<7.3f} | "
                f"{sp_unfused:5.2f}x/{sp_fused:5.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
