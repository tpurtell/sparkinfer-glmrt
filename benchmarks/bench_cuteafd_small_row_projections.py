"""Row-curve and route-exactness benchmark for small-row BF16 TMA tiles and Muse F32 router.

Follows CuteAFD's kernel row-curve rule (AGENTS.md):
- Measures warmed L2-cold time per routed row across decode/verify (1..64) and
  the live prefill band (2048/2752/4096 plus real tails).
- Supports ``--capacity``, ``--routes``, ``--compare-routes-exact``, and
  ``--check-prefix-rows``. Every route is verified bitwise-identical across all
  row thresholds (1..capacity) against both stock prefill/scalar and a 64-row
  single-row oracle.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import cutlass
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bench_cuda_graph, capture_cuda_graph, make_l2_flush_fn

from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
from b12x.integration.cuteafd import muse, pointwise
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

DEFAULT_ROWS = (1, 2, 4, 6, 8, 16, 17, 32, 64, 65, 256, 2048, 2050, 2721, 2752, 4096)

CASES = {
    "muse_tp1_kv": ("p1", 8192, 1024, torch.bfloat16),
    "hy4f_latent_down": ("p1", 3072, 1536, torch.bfloat16),
    "hy4_q_a": ("p1", 6144, 2048, torch.bfloat16),
    "step5_router": ("p1", 4096, 352, torch.float32),
    "step5_attn_gate": ("p1", 4096, 64, torch.bfloat16),
    "ml4_index_kw": ("p1", 6144, 160, torch.bfloat16),
    "hy4f_indexer_wk": ("p1", 3072, 128, torch.bfloat16),
    "step5_index_weight": ("p1_scale", 4096, 16, torch.float32),
    "muse_router": ("p3", 8192, 256, torch.float32),
}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _nvidia_telemetry() -> str:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,pstate,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ], text=True).strip()
        return out.splitlines()[0] if out else "unavailable"
    except Exception as exc:
        return f"unavailable ({exc})"


def _bench_us(fn, flush, replays: int) -> float:
    graph = capture_cuda_graph(fn, warmup=2)
    return statistics.median(bench_cuda_graph(graph, replays=replays, l2_flush=flush)["replay_us"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--routes", default=",".join(CASES.keys()),
                        help="Comma-separated route names from: " + ", ".join(CASES.keys()))
    parser.add_argument("--rows", default=",".join(str(r) for r in DEFAULT_ROWS))
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--compare-routes-exact", action="store_true")
    parser.add_argument("--check-prefix-rows", action="store_true")
    args = parser.parse_args()

    row_list = [int(r) for r in args.rows.split(",") if r.strip() and int(r) <= args.capacity]
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    flush = make_l2_flush_fn(True)
    print(f"GPU telemetry (start): {_nvidia_telemetry()}", flush=True)

    for name in routes:
        kind, k, n, dtype = CASES[name]
        c_dt = cutlass.Float32 if dtype == torch.float32 else cutlass.BFloat16
        gen = torch.Generator(device="cuda").manual_seed(k + n)
        x = (torch.randn((args.capacity, k), generator=gen, device="cuda") * 0.1).bfloat16()
        w = ((torch.randn((n, k), generator=gen, device="cuda") * 0.02).float()
             if kind == "p3" else (torch.randn((n, k), generator=gen, device="cuda") * 0.02).bfloat16())
        out_new = torch.empty((args.capacity, n), dtype=dtype, device="cuda")
        out_old = torch.empty((args.capacity, n), dtype=dtype, device="cuda")
        scratch = torch.empty((args.capacity, n), dtype=torch.bfloat16, device="cuda")

        if kind == "p1":
            prog_new = pointwise.compile_bf16_projection_aot(width=k, outputs=n, output_dtype=dtype)
            old_launch = TmaBf16Projection(n, k, out_dtype=c_dt, tiles_by_rows=())
            prog_old = compile_program(
                old_launch, name=f"old_{name}",
                operands=(Operand("x", torch.bfloat16, f"[r,{k}]"),
                          Operand("w", torch.bfloat16, f"[{n},{k}]"),
                          Operand("o", dtype, f"[r,{n}]", "out")),
                scalars=(Scalar("r"),), key=("old", old_launch.key()),
            )
            run_new = lambda m: prog_new.launch(x[:m], w, out_new[:m], scalars=(m,))
            run_old = lambda m: prog_old.launch(x[:m], w, out_old[:m], scalars=(m,))
            run_one = lambda r, dst: prog_new.launch(x[r:r + 1], w, dst, scalars=(1,))
        elif kind == "p1_scale":
            prog_new = pointwise.compile_bf16_scaled_projection_aot(width=k, outputs=n, prescale=0.5)
            old_tma = TmaBf16Projection(n, k, out_dtype=cutlass.BFloat16, tiles_by_rows=())
            old_scale = pointwise.ScaleProjection(k, n, 0.5)
            old_scale.projection = old_tma
            prog_old = compile_program(
                old_scale, name=f"old_{name}",
                operands=(Operand("x", torch.bfloat16, f"[r,{k}]"),
                          Operand("w", torch.bfloat16, f"[{n},{k}]"),
                          Operand("o", torch.float32, f"[r,{n}]", "out"),
                          Operand("s", torch.bfloat16, f"[r,{n}]", "scratch")),
                scalars=(Scalar("r"),), key=("old_scale_tma", k, n),
                scratch={"s": lambda r: int(r) * int(n) * 2},
            )
            run_new = lambda m: prog_new.launch(x[:m], w, out_new[:m], scratch[:m], scalars=(m,))
            run_old = lambda m: prog_old.launch(x[:m], w, out_old[:m], scratch[:m], scalars=(m,))
            run_one = lambda r, dst: prog_new.launch(x[r:r + 1], w, dst, scratch[:1], scalars=(1,))
        else:
            prog_new = muse.compile_router_aot()
            prog_old = muse.compile_router_aot(scalar_max_rows=2**31 - 1)
            prog_old_tiled = muse.compile_router_aot(scalar_max_rows=0)
            run_new = lambda m: prog_new.launch(x[:m], w, out_new[:m], scalars=(m,))
            run_old = lambda m: (prog_old if m <= 16 else prog_old_tiled).launch(x[:m], w, out_old[:m], scalars=(m,))
            run_one = lambda r, dst: prog_new.launch(x[r:r + 1], w, dst, scalars=(1,))

        print(f"\n=== {name} ({kind}, K={k}, N={n}) ===", flush=True)
        print(f"{'rows':>6} | {'old_us':>8} | {'new_us':>8} | {'speedup':>7} | {'us/row':>8} | checks", flush=True)
        singles = None
        if args.check_prefix_rows:
            n_ref = min(64, args.capacity)
            singles = torch.empty((n_ref, n), dtype=dtype, device="cuda")
            for r in range(n_ref):
                run_one(r, singles[r:r + 1])
            torch.cuda.synchronize()
        for m in row_list:
            checks = []
            if args.compare_routes_exact:
                run_new(m)
                run_old(m)
                torch.cuda.synchronize()
                ok = bool(torch.equal(_bits(out_new[:m]), _bits(out_old[:m])))
                checks.append(f"exact={'OK' if ok else 'FAIL'}")
                if not ok:
                    raise AssertionError(f"{name} rows={m} failed --compare-routes-exact")
            if args.check_prefix_rows:
                run_new(m)
                torch.cuda.synchronize()
                check = min(m, singles.shape[0])
                ok = bool(torch.equal(_bits(out_new[:check]), _bits(singles[:check])))
                checks.append(f"prefix={'OK' if ok else 'FAIL'}")
                if not ok:
                    raise AssertionError(f"{name} rows={m} failed --check-prefix-rows")
            t_old = _bench_us(lambda m=m: run_old(m), flush, args.replays)
            t_new = _bench_us(lambda m=m: run_new(m), flush, args.replays)
            print(f"{m:6d} | {t_old:8.2f} | {t_new:8.2f} | {t_old/t_new:6.2f}x | {t_new/m:8.4f} | {' '.join(checks)}",
                  flush=True)

    print(f"\nGPU telemetry (end): {_nvidia_telemetry()}", flush=True)


if __name__ == "__main__":
    main()
