#!/usr/bin/env python3
"""Time the shipped and target-only controller projections; gate route invariance."""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=",".join(map(str, (*range(1, 17), 32, 64, 128, 2048, 2752, 4096))))
    parser.add_argument("--compare-routes-exact", action="store_true", help="require bitwise row/route invariance")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    import cutlass
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd.inkling2_loop_controller import (
        compile_inkling2_mini_controller_down_aot, compile_inkling2_mini_controller_up_aot,
    )
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from benchmarks.common import capture_cuda_graph, bench_cuda_graph, make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
    rows_list = sorted(set(map(int, args.rows.split(","))))
    if not rows_list or min(rows_list) != 1 or max(rows_list) > 4096 or args.samples < 1:
        parser.error("rows must include 1 and be in 1..4096; samples must be positive")
    cap = max(rows_list)
    torch.manual_seed(2804)
    torch.backends.cuda.matmul.allow_tf32 = False
    flush = make_l2_flush_fn(True)
    report = {"gpu_before": nvidia_smi_gpu_mode_snapshot(), "rows": rows_list, "results": []}
    for direction, k, n, dtype, compiler in (
        ("down", 4608, 64, torch.bfloat16, compile_inkling2_mini_controller_down_aot),
        ("up", 64, 4608, torch.float32, compile_inkling2_mini_controller_up_aot),
    ):
        x = torch.randn(cap, k, device="cuda").bfloat16()
        w = (torch.randn(n, k, device="cuda") * .03).bfloat16()
        out = torch.empty(cap, n, device="cuda", dtype=dtype)
        bit_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
        op = TmaBf16Projection(n, k, out_dtype=cutlass.BFloat16 if direction == "down" else cutlass.Float32,
                               tiles_by_rows=((16, 1, 4, 16, 64, 6),))
        base = compile_program(op, name="mini_controller_shipped", operands=(
            Operand("x", torch.bfloat16, "x"), Operand("w", torch.bfloat16, "w"),
            Operand("out", dtype, "out", "out")), scalars=(Scalar("rows"),), key=op.key() + (direction,))
        candidate = compiler(max_rows=cap)
        partial = torch.empty(candidate.scratch_bytes(cap).get("partial", 0) // 4, device="cuda")
        def run(name, m, source=x):
            program = base if name == "base" else candidate
            program.launch(source, w, out, *((partial,) if name == "candidate" and direction == "down" else ()), scalars=(m,))
        def exact(a, b):
            return torch.equal(a.view(bit_dtype), b.view(bit_dtype))
        # Compare each actual route against row=1, not merely the row-0 prefix.
        names = ["base", "candidate"]
        singles = {}
        for name in names:
            single_rows = []
            for i in range(min(16, cap)):
                run(name, 1, x[i:]); single_rows.append(out[0].clone())
            singles[name] = torch.stack(single_rows)
        for m in rows_list:
            full, graphs = {}, {}
            for name in names:
                out.fill_(float("nan")); run(name, m); torch.cuda.synchronize()
                result = out[:m].clone()
                assert result.isfinite().all() and out[m:].isnan().all(), (direction, name, m, "extent")
                if args.compare_routes_exact:
                    r = min(m, 16)
                    assert exact(result[:r], singles[name][:r]), (direction, name, m, "row invariance")
                    # Check all prefix rows against the other tiling's direct route.
                    if m > 16:
                        offset = m - 1
                        run(name, 1, x[offset:]); torch.cuda.synchronize()
                        assert exact(result[offset], out[0]), (direction, name, m, "tail invariance")
                torch.testing.assert_close(result.float(), x[:m].float() @ w.float().T, rtol=.007, atol=.006)
                full[name] = result
                graph = capture_cuda_graph(lambda: run(name, m), warmup=2)
                out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                assert exact(result, out[:m]) and out[m:].isnan().all(), (direction, name, m, "graph replay")
                graphs[name] = graph
            if direction == "up":
                assert exact(full["base"], full["candidate"]), (direction, m, "shipped route")
            samples = {name: [] for name in names}
            for order in (names, names[::-1]):
                for name in order:
                    samples[name].extend(bench_cuda_graph(graphs[name], replays=args.samples, l2_flush=flush)["replay_us"])
            # Batch 64 identical launches inside a graph to avoid CUDA-event timer quantization.
            warm = {}
            for name in names:
                def call():
                    for _ in range(64):
                        run(name, m)
                graph = capture_cuda_graph(call, warmup=1)
                warm[name] = statistics.median(bench_cuda_graph(graph, replays=8)["replay_us"]) / 64
            cell = {"direction": direction, "rows": m, "cold_us": {name: statistics.median(ts) for name, ts in samples.items()},
                    "warm_us": warm, "exact_to_base": exact(full["base"], full["candidate"]),
                    "max_abs": (full["base"].float() - full["candidate"].float()).abs().max().item()}
            report["results"].append(cell)
            print(json.dumps(cell), flush=True)
    report["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gpu_before": report["gpu_before"], "gpu_after": report["gpu_after"]}), flush=True)
    print("controller projection oracle/graph/row sweep passed", flush=True)


if __name__ == "__main__":
    main()
