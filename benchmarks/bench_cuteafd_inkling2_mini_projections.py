"""Row-curve and bitwise oracle benchmark for Inkling 2 Mini small-row BF16 projections.

Tests the three shipping AOT programs:
- ``compile_inkling2_mini_attn_in_proj_local_aot``  (6 outputs, N=8600, 39.63 MB)
- ``compile_inkling2_mini_attn_in_proj_global_aot`` (8 outputs, N=8664, 39.92 MB)
- ``compile_inkling2_mini_router_proj_aot``         (128 F32 outputs, 0.59 MB)

Against stock ``TmaBf16Projection`` (keeping O 3072->2304 and Rel 64->1024 at stock)
over rows 1..8 plus route-switch rows 9, 16, 64.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutlass
from common import bench_cuda_graph, capture_cuda_graph, make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd.inkling2_mini_projections import (
    STOCK_TILES,
    compile_inkling2_mini_attn_in_proj_global_aot,
    compile_inkling2_mini_attn_in_proj_local_aot,
    compile_inkling2_mini_router_proj_aot,
)


def exact_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def poison_val(dtype: torch.dtype):
    return (torch.int16, 0x7FFF) if dtype == torch.bfloat16 else (torch.int32, 0x7FFFFFFF)


def fill_poison(t: torch.Tensor):
    vt, val = poison_val(t.dtype)
    t.view(vt).fill_(val)


def check_poison(t: torch.Tensor, label: str):
    vt, val = poison_val(t.dtype)
    assert torch.all(t.view(vt) == val), f"tail poison overwritten: {label}"


def bench_single(fn, *, replays: int, flush):
    g = capture_cuda_graph(fn, warmup=2)
    cold = statistics.median(bench_cuda_graph(g, replays=replays, l2_flush=flush)["replay_us"])

    def batch():
        for _ in range(32):
            fn()

    gw = capture_cuda_graph(batch, warmup=1)
    warm = statistics.median(bench_cuda_graph(gw, replays=8)["replay_us"]) / 32.0
    return cold, warm


def bench_pair_interleaved(fn_stock, fn_cand, *, replays: int, flush, divisor: float = 1.0):
    g_st = capture_cuda_graph(fn_stock, warmup=2)
    g_ca = capture_cuda_graph(fn_cand, warmup=2)
    st_ts, ca_ts = [], []
    half = max(replays // 2, 4)
    for order in ((0, 1), (1, 0)):
        for which in order:
            g = g_st if which == 0 else g_ca
            ts = bench_cuda_graph(g, replays=half, l2_flush=flush)["replay_us"]
            (st_ts if which == 0 else ca_ts).extend(ts)
    return statistics.median(st_ts) / divisor, statistics.median(ca_ts) / divisor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", type=int, default=24)
    p.add_argument("--rows", default="1,2,3,4,5,6,7,8,9,16,64")
    p.add_argument("--compare-routes-exact", action="store_true", default=True)
    p.add_argument("--check-prefix-rows", action="store_true", default=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    rows_list = [int(v) for v in args.rows.split(",")]
    flush = make_l2_flush_fn(True)
    torch.manual_seed(2804)
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu_before = nvidia_smi_gpu_mode_snapshot()

    max_m = max(rows_list)
    xq = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xk = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xv = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xg = (torch.randn(max_m, 3072, device="cuda") * 0.1).bfloat16()
    xr = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xrel = (torch.randn(max_m * 24, 64, device="cuda") * 0.1).bfloat16()
    for t in (xq, xk, xv, xg, xr):
        for r in (2, 7, 15, 63):
            if r < max_m:
                t[r].copy_(t[0])

    N_SETS = 4
    sets = []
    for _ in range(N_SETS):
        sets.append({
            "wq": (torch.randn(3072, 2304, device="cuda") * 0.02).bfloat16(),
            "wr": (torch.randn(384, 2304, device="cuda") * 0.02).bfloat16(),
            "wgate": (torch.randn(3072, 2304, device="cuda") * 0.02).bfloat16(),
            "wsink": (torch.randn(24, 2304, device="cuda") * 0.02).bfloat16(),
            "wk": (torch.randn(1024, 2304, device="cuda") * 0.02).bfloat16(),
            "wv": (torch.randn(1024, 2304, device="cuda") * 0.02).bfloat16(),
            "wk_dyn": (torch.randn(32, 2304, device="cuda") * 0.02).bfloat16(),
            "wv_dyn": (torch.randn(32, 2304, device="cuda") * 0.02).bfloat16(),
            "wo": (torch.randn(2304, 3072, device="cuda") * 0.02).bfloat16(),
            "wrouter": (torch.randn(128, 2304, device="cuda") * 0.02).bfloat16(),
            "wrel": (torch.randn(1024, 64, device="cuda") * 0.02).bfloat16(),
        })
    s0 = sets[0]

    oq = torch.empty(max_m + 1, 3072, device="cuda", dtype=torch.bfloat16)
    orr = torch.empty(max_m + 1, 384, device="cuda", dtype=torch.bfloat16)
    ogate = torch.empty(max_m + 1, 3072, device="cuda", dtype=torch.bfloat16)
    osink = torch.empty(max_m + 1, 24, device="cuda", dtype=torch.float32)
    ok = torch.empty(max_m + 1, 1024, device="cuda", dtype=torch.bfloat16)
    ov = torch.empty(max_m + 1, 1024, device="cuda", dtype=torch.bfloat16)
    okd = torch.empty(max_m + 1, 32, device="cuda", dtype=torch.float32)
    ovd = torch.empty(max_m + 1, 32, device="cuda", dtype=torch.float32)
    oo = torch.empty(max_m + 1, 2304, device="cuda", dtype=torch.bfloat16)
    orouter = torch.empty(max_m + 1, 128, device="cuda", dtype=torch.float32)
    orel = torch.empty(max_m * 24 + 1, 1024, device="cuda", dtype=torch.bfloat16)

    def compile_stock(n, k, f32=False):
        dt = torch.float32 if f32 else torch.bfloat16
        cd = cutlass.Float32 if f32 else cutlass.BFloat16
        tma = TmaBf16Projection(n, k, out_dtype=cd, tiles_by_rows=STOCK_TILES)
        return compile_program(tma, name=f"stock_{k}_{n}_{'f32' if f32 else 'bf16'}", operands=(
            Operand("x", torch.bfloat16, f"[rows,{k}]"),
            Operand("w", torch.bfloat16, f"[{n},{k}]"),
            Operand("out", dt, f"[rows,{n}]", "out"),
        ), scalars=(Scalar("rows"),), key=("stock", k, n, f32))

    st_q = compile_stock(3072, 2304)
    st_r = compile_stock(384, 2304)
    st_kv = compile_stock(1024, 2304)
    st_sink = compile_stock(24, 2304, True)
    st_dyn = compile_stock(32, 2304, True)
    st_o = compile_stock(2304, 3072)
    st_router = compile_stock(128, 2304, True)
    st_rel = compile_stock(1024, 64)

    prog_local = compile_inkling2_mini_attn_in_proj_local_aot(max_rows=4096)
    prog_global = compile_inkling2_mini_attn_in_proj_global_aot(max_rows=4096)
    prog_router = compile_inkling2_mini_router_proj_aot(max_rows=4096)

    def run_stock_local(m, ws=s0):
        st_q.launch(xq, ws["wq"], oq, scalars=(m,))
        st_r.launch(xq, ws["wr"], orr, scalars=(m,))
        st_q.launch(xq, ws["wgate"], ogate, scalars=(m,))
        st_sink.launch(xq, ws["wsink"], osink, scalars=(m,))
        st_kv.launch(xk, ws["wk"], ok, scalars=(m,))
        st_kv.launch(xv, ws["wv"], ov, scalars=(m,))

    def run_stock_global(m, ws=s0):
        run_stock_local(m, ws)
        st_dyn.launch(xk, ws["wk_dyn"], okd, scalars=(m,))
        st_dyn.launch(xv, ws["wv_dyn"], ovd, scalars=(m,))

    def run_cand_local(m, ws=s0):
        prog_local.launch(xq, xk, xv, ws["wq"], ws["wr"], ws["wgate"], ws["wsink"], ws["wk"], ws["wv"],
                          oq, orr, ogate, osink, ok, ov, scalars=(m,))

    def run_cand_global(m, ws=s0):
        prog_global.launch(xq, xk, xv, ws["wq"], ws["wr"], ws["wgate"], ws["wsink"], ws["wk"], ws["wv"],
                           ws["wk_dyn"], ws["wv_dyn"], oq, orr, ogate, osink, ok, ov, okd, ovd, scalars=(m,))

    def run_cand_router(m, ws=s0):
        prog_router.launch(xr, ws["wrouter"], orouter, scalars=(m,))

    # Bitwise oracle + prefix row invariance + exact route switch
    refs = {
        "q": xq.float() @ s0["wq"].float().T, "r": xq.float() @ s0["wr"].float().T,
        "gate": xq.float() @ s0["wgate"].float().T, "sink": xq.float() @ s0["wsink"].float().T,
        "k": xk.float() @ s0["wk"].float().T, "v": xv.float() @ s0["wv"].float().T,
        "k_dyn": xk.float() @ s0["wk_dyn"].float().T, "v_dyn": xv.float() @ s0["wv_dyn"].float().T,
        "router": xr.float() @ s0["wrouter"].float().T,
    }
    bufs_local = [("q", oq), ("r", orr), ("gate", ogate), ("sink", osink), ("k", ok), ("v", ov)]
    bufs_global = [*bufs_local, ("k_dyn", okd), ("v_dyn", ovd)]

    saved_small = {"local": {}, "global": {}, "router": {}}
    for tag, run_st, run_ca, out_list, all_exact_stock in [
        ("local", run_stock_local, run_cand_local, bufs_local, True),
        ("global", run_stock_global, run_cand_global, bufs_global, True),
        ("router", lambda m: st_router.launch(xr, s0["wrouter"], orouter, scalars=(m,)),
         run_cand_router, [("router", orouter)], False),
    ]:
        for m in rows_list:
            for _, b in out_list:
                fill_poison(b)
            run_ca(m)
            torch.cuda.synchronize()
            eager_snap = {}
            for name, b in out_list:
                check_poison(b[m:], f"{tag}/{name}/eager/m={m}")
                live = b[:m]
                assert torch.isfinite(live).all()
                rtol, atol = (1e-2, 2e-3) if b.dtype == torch.bfloat16 else (5e-5, 5e-5)
                torch.testing.assert_close(live.float(), refs[name][:m], rtol=rtol, atol=atol)
                eager_snap[name] = live.clone()

            g = capture_cuda_graph(lambda m=m, rc=run_ca: rc(m), warmup=1)
            for _, b in out_list:
                fill_poison(b)
            g.replay()
            torch.cuda.synchronize()
            for name, b in out_list:
                check_poison(b[m:], f"{tag}/{name}/graph/m={m}")
                assert exact_bits(b[:m], eager_snap[name])

            if m <= 8 and args.check_prefix_rows:
                saved_small[tag][m] = eager_snap
                for name in eager_snap:
                    cur = eager_snap[name]
                    assert exact_bits(cur[0:1], saved_small[tag][1][name][0:1])
                    for dup in (2, 7):
                        if dup < m:
                            assert exact_bits(cur[dup:dup + 1], saved_small[tag][1][name][0:1])
                    for m1 in saved_small[tag]:
                        if m1 < m:
                            assert exact_bits(cur[:m1], saved_small[tag][m1][name])
            if args.compare_routes_exact and (all_exact_stock or m > 8):
                for _, b in out_list:
                    fill_poison(b)
                run_st(m)
                torch.cuda.synchronize()
                for name, b in out_list:
                    assert exact_bits(eager_snap[name], b[:m]), f"stock mismatch {tag}/{name}/m={m}"
        print(f"PASS oracle: {tag}", flush=True)

    # Interleaved 48-visit streaming DRAM-cold + L2-warm row curve (O and Rel stay stock)
    curve = []
    for m in rows_list:
        def run_48v_stock(m=m):
            for v in range(48):
                ws = sets[v % N_SETS]
                if v % 6 == 5:
                    run_stock_global(m, ws)
                else:
                    run_stock_local(m, ws)
                st_o.launch(xg, ws["wo"], oo, scalars=(m,))
                st_router.launch(xr, ws["wrouter"], orouter, scalars=(m,))
                st_rel.launch(xrel, ws["wrel"], orel, scalars=(m * 24,))

        def run_48v_cand(m=m):
            for v in range(48):
                ws = sets[v % N_SETS]
                if v % 6 == 5:
                    run_cand_global(m, ws)
                else:
                    run_cand_local(m, ws)
                st_o.launch(xg, ws["wo"], oo, scalars=(m,))
                run_cand_router(m, ws)
                st_rel.launch(xrel, ws["wrel"], orel, scalars=(m * 24,))

        st_48_us, ca_48_us = bench_pair_interleaved(run_48v_stock, run_48v_cand, replays=args.replays, flush=None)
        loc_st, loc_ca = bench_pair_interleaved(
            lambda m=m: [run_stock_local(m, ws) for ws in sets],
            lambda m=m: [run_cand_local(m, ws) for ws in sets],
            replays=args.replays, flush=None, divisor=N_SETS,
        )
        glb_st, glb_ca = bench_pair_interleaved(
            lambda m=m: [run_stock_global(m, ws) for ws in sets],
            lambda m=m: [run_cand_global(m, ws) for ws in sets],
            replays=args.replays, flush=None, divisor=N_SETS,
        )
        r_st_c, r_st_w = bench_single(lambda m=m: st_router.launch(xr, s0["wrouter"], orouter, scalars=(m,)), replays=16, flush=flush)
        r_ca_c, r_ca_w = bench_single(lambda m=m: run_cand_router(m, s0), replays=16, flush=flush)
        cell = {
            "rows": m,
            "stream_48v_ms": {"stock": st_48_us / 1e3, "cand": ca_48_us / 1e3, "speedup": st_48_us / ca_48_us},
            "local_dram_us": {"stock": loc_st, "cand": loc_ca},
            "global_dram_us": {"stock": glb_st, "cand": glb_ca},
            "router_us": {"stock_cold": r_st_c, "cand_cold": r_ca_c, "stock_warm": r_st_w, "cand_warm": r_ca_w},
        }
        curve.append(cell)
        print(
            f"M={m:2d}: 48v DRAM {st_48_us/1e3:5.3f}->{ca_48_us/1e3:5.3f}ms ({st_48_us/ca_48_us:4.2f}x) | "
            f"local {loc_st:5.1f}->{loc_ca:5.1f}us | global {glb_st:5.1f}->{glb_ca:5.1f}us | "
            f"router warm {r_st_w:4.2f}->{r_ca_w:4.2f}us (cold {r_st_c:4.2f}->{r_ca_c:4.2f}us)",
            flush=True,
        )

    gpu_after = nvidia_smi_gpu_mode_snapshot()
    if args.output:
        Path(args.output).write_text(json.dumps({"curve": curve, "gpu_before": gpu_before, "gpu_after": gpu_after}, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
