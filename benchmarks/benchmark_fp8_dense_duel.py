#!/usr/bin/env python3
"""Duel A microbenchmark: shared checkpoint-FP8 dense kernels on SM120 RTX.

Compares the baseline production ``Fp8Projection`` (``MmaFp8Gemv`` for decode,
``TmaFp8Gemm`` for W8A16 prefill) against ``CandidateFp8Projection`` (separate
constructor and ``key()``) using interleaved CUDA-graph replay timing over
L2-cold weight copies (>= 2x queried L2 size) and a warm single-copy reference,
followed by bitwise output verification (``torch.equal`` on viewed ``int16``)
and a full 1..64 row-invariance sweep with ``0x7FFF`` sentinel-filled outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd._fp8_weights import CandidateFp8Projection, Fp8Projection
from benchmarks._exlr8_dense_arms import cold_copies
from benchmarks.benchmark_exlr8_a16 import capture, timing

_SENTINEL = 0x7FFF


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    mode: str  # "decode" or "prefill"
    n: int
    k: int
    row_scales: bool = False
    kmajor: bool = False
    warps: int | None = None
    groups: int = 4
    important_rows: tuple[int, ...] = ()


DECODE_SHAPES: tuple[ShapeSpec, ...] = (
    # Primary: Qwen 3.8 27B FP8 census groups + true production widths
    ShapeSpec("qwen_gu", "decode", 34816, 5120, important_rows=(1, 16)),
    ShapeSpec("qwen_down", "decode", 5120, 17408, important_rows=(1, 16)),
    ShapeSpec("qwen_qkv", "decode", 16384, 5120, important_rows=(1, 16)),
    ShapeSpec("qwen_gdn_in", "decode", 16480, 5120),
    ShapeSpec("qwen_attn_in", "decode", 14336, 5120),
    # GLM Flash TP2 half-head decode shapes
    ShapeSpec("glmf_kda_in_half", "decode", 12576, 4096, row_scales=True, kmajor=True, warps=8, groups=2),
    ShapeSpec("glmf_kda_o_half", "decode", 4096, 4096, row_scales=True, kmajor=True, warps=8, groups=2),
    ShapeSpec("glmf_mla_q_b_half", "decode", 8192, 1536, important_rows=(1, 16)),
    ShapeSpec("glmf_mla_o_half", "decode", 4096, 8192, warps=8, groups=2, important_rows=(1, 16)),
    ShapeSpec("glmf_output_shard", "decode", 2048, 8192),
    ShapeSpec("glmf_dense_gu_half", "decode", 12288, 4096),
    ShapeSpec("glmf_dense_down_half", "decode", 4096, 6144),
    ShapeSpec("glmf_shared_gu_half", "decode", 2048, 4096, warps=8, groups=4),
    ShapeSpec("glmf_shared_down_half", "decode", 4096, 1024),
    # MiMo V2.6 Pro TP2 decode shapes
    ShapeSpec("mimo_pro_qkv", "decode", 27648, 6144, row_scales=True, important_rows=(1, 16)),
    ShapeSpec("mimo_pro_o", "decode", 6144, 16384, row_scales=True, important_rows=(1, 16)),
    ShapeSpec("mimo_pro_dense_gu", "decode", 32768, 6144, row_scales=True),
    ShapeSpec("mimo_pro_dense_down", "decode", 6144, 16384, row_scales=True),
)

PREFILL_SHAPES: tuple[ShapeSpec, ...] = (
    # Primary W8A16 O/Q prefill shapes + Qwen down / gate_up
    ShapeSpec("mimo_pro_o", "prefill", 6144, 16384, row_scales=True, important_rows=(2048,)),
    ShapeSpec("mimo_pro_qkv", "prefill", 27648, 6144, row_scales=True, important_rows=(2048,)),
    ShapeSpec("glmf_mla_o", "prefill", 4096, 16384, important_rows=(2048,)),
    ShapeSpec("glmf_mla_q_b", "prefill", 16384, 1536, important_rows=(2048,)),
    ShapeSpec("glmf_mla_o_half", "prefill", 4096, 8192, important_rows=(2048,)),
    ShapeSpec("glmf_mla_q_b_half", "prefill", 8192, 1536, important_rows=(2048,)),
    ShapeSpec("qwen_down", "prefill", 5120, 17408),
    ShapeSpec("qwen_gu", "prefill", 34816, 5120),
)


def _gpu_snapshot() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,power.draw,power.limit,clocks.current.sm,temperature.gpu",
        "--format=csv,noheader",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=5).strip()
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            rows.append({
                "index": parts[0],
                "name": parts[1],
                "power_draw": parts[2],
                "power_limit": parts[3],
                "sm_clock": parts[4],
                "temp_gpu": parts[5],
            })
        else:
            rows.append({"raw": line})
    return rows


def _compile_proj(tag: str, spec: ShapeSpec, proj):
    class Launch:
        def __init__(self):
            self.p = proj

        def key(self):
            return ("duel_fp8_proj", tag, spec.mode, spec.n, spec.k, self.p.key())

        @__import__("cutlass").cute.jit
        def __call__(self, x, w, s, o, rows, fp8_rows, stream):
            from cutlass import Int64
            self.p(x, w, s, o, rows, fp8_rows, Int64(0), stream)

    body = Launch()
    ops = (
        Operand("x", torch.bfloat16, "x"),
        Operand("w", torch.float8_e4m3fn, "w"),
        Operand("s", torch.float32, "s", align=16),
        Operand("o", torch.bfloat16, "o", "out"),
    )
    scalars = (Scalar("rows"), Scalar("fp8_rows"))
    return compile_program(body, name=f"duel_{tag}_{spec.mode}_{spec.name}", operands=ops, scalars=scalars, key=body.key())


def _make_weights(spec: ShapeSpec, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((spec.n, spec.k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kb = spec.k // 128
    if not spec.row_scales:
        scale = torch.rand(((spec.n + 127) // 128, kb), generator=gen, device="cuda", dtype=torch.float32) * 0.01 + 1e-3
    else:
        row_s = torch.rand((spec.n, kb), generator=gen, device="cuda", dtype=torch.float32) * 0.01 + 1e-3
        use_kmajor = spec.kmajor or (spec.mode == "prefill")
        scale = row_s.t().contiguous() if use_kmajor else row_s.contiguous()
    return w8, scale


def _build_arms(spec: ShapeSpec, *, cand_groups: int | None = None, cand_cta_tiles: int | None = None):
    if spec.mode == "prefill":
        base = Fp8Projection(spec.n, spec.k, prefill_rows=4096, row_scales=spec.row_scales)
        cand = CandidateFp8Projection(spec.n, spec.k, prefill_rows=4096, row_scales=spec.row_scales)
    else:
        base = Fp8Projection(
            spec.n, spec.k, decode_rows=64, row_scales=spec.row_scales, kmajor=spec.kmajor,
            warps=spec.warps, groups=spec.groups,
        )
        cand = CandidateFp8Projection(
            spec.n, spec.k, decode_rows=64, row_scales=spec.row_scales, kmajor=spec.kmajor,
            warps=spec.warps, groups=spec.groups, cand_groups=cand_groups, cand_cta_tiles=cand_cta_tiles,
        )
    assert base.key() != cand.key(), f"Candidate key must differ from baseline for {spec}"
    base_prog = _compile_proj("base", spec, base)
    cand_prog = _compile_proj("cand", spec, cand)
    return base, cand, base_prog, cand_prog


def _check_row_invariance(spec: ShapeSpec, prog, w8: torch.Tensor, scale: torch.Tensor, seed: int) -> tuple[bool, str]:
    if spec.mode != "decode":
        return True, "n/a (prefill)"
    gen = torch.Generator(device="cuda").manual_seed(seed + 999)
    x64 = (torch.randn((64, spec.k), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    out64 = torch.empty((64, spec.n), dtype=torch.bfloat16, device="cuda")
    out64.view(torch.int16).fill_(_SENTINEL)
    prog.launch(x64, w8, scale, out64, scalars=(64, 64))
    torch.cuda.synchronize()
    bits64 = out64.view(torch.int16)
    if bool((bits64 == _SENTINEL).any()):
        return False, "M=64 left sentinel unwritten"
    for r in range(1, 65):
        xr = x64[:r].contiguous()
        outr = torch.empty((r, spec.n), dtype=torch.bfloat16, device="cuda")
        outr.view(torch.int16).fill_(_SENTINEL)
        prog.launch(xr, w8, scale, outr, scalars=(r, 64))
        torch.cuda.synchronize()
        bitsr = outr.view(torch.int16)
        if bool((bitsr == _SENTINEL).any()):
            return False, f"M={r} left sentinel unwritten"
        if not torch.equal(bitsr, bits64[:r]):
            diff = int((bitsr != bits64[:r]).sum().item())
            return False, f"M={r} vs M=64 mismatch ({diff} elements)"
    return True, "pass (rows 1..64 == M=64)"


def _select_shapes(arg: str) -> list[ShapeSpec]:
    key = arg.strip().lower()
    if key == "decode":
        return list(DECODE_SHAPES)
    if key == "prefill":
        return list(PREFILL_SHAPES)
    if key == "all":
        return list(DECODE_SHAPES) + list(PREFILL_SHAPES)
    if key == "important":
        return [s for s in (list(DECODE_SHAPES) + list(PREFILL_SHAPES)) if s.important_rows]
    selected: list[ShapeSpec] = []
    tokens = [t.strip() for t in arg.split(",") if t.strip()]
    pool = list(DECODE_SHAPES) + list(PREFILL_SHAPES)
    for tok in tokens:
        matched = [s for s in pool if s.name == tok or f"{s.mode}:{s.name}" == tok or f"{s.n}x{s.k}" == tok]
        if not matched:
            raise ValueError(f"Unknown shape selector {tok!r}")
        selected.extend(matched)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Duel A FP8 dense benchmark (SM120 RTX)")
    parser.add_argument("--shapes", default="decode", help="decode, prefill, important, all, or comma-separated names")
    parser.add_argument("--rows", default=None, help="Comma-separated row counts (default: 1,4,8,16,32,64 for decode; 128,512,2048,4096 for prefill)")
    parser.add_argument("--format", default="e4m3", choices=("e4m3",), help="Weight format")
    parser.add_argument("--cold-l2", action="store_true", default=True, help="Cycle weight copies >= 2x L2 capacity")
    parser.add_argument("--warmup", type=int, default=4, help="Warmup launches before graph capture (>= 3)")
    parser.add_argument("--replays", type=int, default=5, help="Graph replay timing repeats per interleave round")
    parser.add_argument("--interleave-rounds", type=int, default=4, help="Interleaved baseline/candidate rounds")
    parser.add_argument("--cand-groups", type=int, default=None, help="Override candidate GEMV groups")
    parser.add_argument("--cand-cta-tiles", type=int, default=None, help="Override candidate GEMV cta_tiles")
    parser.add_argument("--output", type=str, default=None, help="JSON artifact output path")
    args = parser.parse_args()

    if args.warmup < 3:
        raise ValueError("--warmup must be >= 3")

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    sm_count = int(props.multi_processor_count)
    l2_bytes = int(props.L2_cache_size)
    shapes = _select_shapes(args.shapes)

    snap_before = _gpu_snapshot()
    t_start = time.time()
    records = []

    for idx, spec in enumerate(shapes):
        if args.rows is not None:
            row_list = [int(r) for r in args.rows.split(",") if r.strip()]
        else:
            row_list = [1, 4, 8, 16, 32, 64] if spec.mode == "decode" else [128, 512, 2048, 4096]

        base_obj, cand_obj, base_prog, cand_prog = _build_arms(
            spec, cand_groups=args.cand_groups, cand_cta_tiles=args.cand_cta_tiles
        )
        seed = 10_000 + idx * 137 + (spec.n * 31 + spec.k * 7) % 10_000
        w0, s0 = _make_weights(spec, seed)
        weight_bytes = w0.numel() * w0.element_size() + s0.numel() * s0.element_size()
        n_copies = cold_copies(weight_bytes, l2_bytes) if args.cold_l2 else 1
        # Ensure each graph captures enough inner launches for accurate event timing while cycling all cold copies.
        graph_calls = max(n_copies, 8) if spec.mode == "decode" else max(n_copies, 4)

        w_copies = [w0] + [w0.clone() for _ in range(n_copies - 1)]
        s_copies = [s0] + [s0.clone() for _ in range(n_copies - 1)]
        fp8_rows_scalar = 64 if spec.mode == "decode" else 0

        # Row-invariance check (rows 1..64 vs M=64 with sentinel 0x7FFF) once per decode shape
        base_inv_ok, base_inv_msg = _check_row_invariance(spec, base_prog, w0, s0, seed)
        cand_inv_ok, cand_inv_msg = _check_row_invariance(spec, cand_prog, w0, s0, seed)
        if not cand_inv_ok:
            raise RuntimeError(f"Row-invariance failure on {spec.name} ({spec.n}x{spec.k}): {cand_inv_msg}")

        for rows in row_list:
            gen_x = torch.Generator(device="cuda").manual_seed(seed + rows * 19)
            x = (torch.randn((rows, spec.k), generator=gen_x, device="cuda") * 0.5).to(torch.bfloat16)
            out_base = torch.empty((rows, spec.n), dtype=torch.bfloat16, device="cuda")
            out_cand = torch.empty((rows, spec.n), dtype=torch.bfloat16, device="cuda")

            def step_base_cold(_x=x, _o=out_base, _r=rows, _f=fp8_rows_scalar):
                for c in range(n_copies):
                    base_prog.launch(_x, w_copies[c], s_copies[c], _o, scalars=(_r, _f))

            def step_cand_cold(_x=x, _o=out_cand, _r=rows, _f=fp8_rows_scalar):
                for c in range(n_copies):
                    cand_prog.launch(_x, w_copies[c], s_copies[c], _o, scalars=(_r, _f))

            def step_base_warm(_x=x, _o=out_base, _r=rows, _f=fp8_rows_scalar):
                base_prog.launch(_x, w0, s0, _o, scalars=(_r, _f))

            def step_cand_warm(_x=x, _o=out_cand, _r=rows, _f=fp8_rows_scalar):
                cand_prog.launch(_x, w0, s0, _o, scalars=(_r, _f))

            g_base_cold = capture(step_base_cold)
            g_cand_cold = capture(step_cand_cold)
            g_base_warm = capture(step_base_warm)
            g_cand_warm = capture(step_cand_warm)

            base_cold_samples: list[float] = []
            cand_cold_samples: list[float] = []
            base_warm_samples: list[float] = []
            cand_warm_samples: list[float] = []

            for r_idx in range(args.interleave_rounds):
                if r_idx % 2 == 0:
                    base_cold_samples.append(timing(g_base_cold, args.replays) / n_copies)
                    cand_cold_samples.append(timing(g_cand_cold, args.replays) / n_copies)
                    base_warm_samples.append(timing(g_base_warm, args.replays))
                    cand_warm_samples.append(timing(g_cand_warm, args.replays))
                else:
                    cand_cold_samples.append(timing(g_cand_cold, args.replays) / n_copies)
                    base_cold_samples.append(timing(g_base_cold, args.replays) / n_copies)
                    cand_warm_samples.append(timing(g_cand_warm, args.replays))
                    base_warm_samples.append(timing(g_base_warm, args.replays))

            base_cold_Samples_sorted = sorted(base_cold_samples)
            cand_cold_Samples_sorted = sorted(cand_cold_samples)
            base_warm_Samples_sorted = sorted(base_warm_samples)
            cand_warm_Samples_sorted = sorted(cand_warm_samples)

            base_us = base_cold_Samples_sorted[len(base_cold_Samples_sorted) // 2]
            cand_us = cand_cold_Samples_sorted[len(cand_cold_Samples_sorted) // 2]
            base_warm_us = base_warm_Samples_sorted[len(base_warm_Samples_sorted) // 2]
            cand_warm_us = cand_warm_Samples_sorted[len(cand_warm_Samples_sorted) // 2]

            # Exactness verification after timing on sentinel-filled output buffers
            out_base.view(torch.int16).fill_(_SENTINEL)
            out_cand.view(torch.int16).fill_(_SENTINEL)
            base_prog.launch(x, w0, s0, out_base, scalars=(rows, fp8_rows_scalar))
            cand_prog.launch(x, w0, s0, out_cand, scalars=(rows, fp8_rows_scalar))
            torch.cuda.synchronize()

            base_bits = out_base.view(torch.int16)
            cand_bits = out_cand.view(torch.int16)
            no_sentinel = bool((base_bits != _SENTINEL).all() and (cand_bits != _SENTINEL).all())
            exact = no_sentinel and bool(torch.equal(base_bits, cand_bits))
            mismatch_elems = int((base_bits != cand_bits).sum().item()) if not exact else 0
            out_sha256 = hashlib.sha256(cand_bits.cpu().numpy().tobytes()).hexdigest()

            if not exact:
                raise RuntimeError(
                    f"Bitwise mismatch on {spec.mode}:{spec.name} ({spec.n}x{spec.k}) rows={rows}: "
                    f"mismatch_elems={mismatch_elems}, no_sentinel={no_sentinel}"
                )

            speedup = base_us / cand_us if cand_us > 0 else 0.0
            base_gbps = (weight_bytes / (base_us * 1e-6)) / 1e9
            cand_gbps = (weight_bytes / (cand_us * 1e-6)) / 1e9
            is_important = rows in spec.important_rows

            records.append({
                "name": spec.name,
                "mode": spec.mode,
                "shape": [spec.n, spec.k],
                "rows": rows,
                "important": is_important,
                "row_scales": spec.row_scales,
                "kmajor": spec.kmajor,
                "weight_bytes": weight_bytes,
                "cold_copies": n_copies,
                "graph_calls": graph_calls,
                "baseline_us": round(base_us, 3),
                "candidate_us": round(cand_us, 3),
                "speedup": round(speedup, 4),
                "baseline_gbps_cold": round(base_gbps, 2),
                "candidate_gbps_cold": round(cand_gbps, 2),
                "baseline_warm_us": round(base_warm_us, 3),
                "candidate_warm_us": round(cand_warm_us, 3),
                "warm_speedup": round(base_warm_us / cand_warm_us, 4) if cand_warm_us > 0 else 0.0,
                "exact": exact,
                "mismatch_elems": mismatch_elems,
                "row_invariance_baseline": base_inv_ok,
                "row_invariance_candidate": cand_inv_ok,
                "row_invariance_detail": cand_inv_msg,
                "output_sha256": out_sha256,
                "baseline_key": str(base_obj.key()),
                "candidate_key": str(cand_obj.key()),
                "raw_baseline_cold_us": [round(v, 3) for v in base_cold_samples],
                "raw_candidate_cold_us": [round(v, 3) for v in cand_cold_samples],
            })

        del w_copies, s_copies, w0, s0
        torch.cuda.empty_cache()

    snap_after = _gpu_snapshot()
    wall_s = time.time() - t_start

    all_speedups = [r["speedup"] for r in records]
    imp_speedups = [r["speedup"] for r in records if r["important"]]
    geomean_all = math.exp(sum(math.log(s) for s in all_speedups) / len(all_speedups)) if all_speedups else 1.0
    geomean_imp = math.exp(sum(math.log(s) for s in imp_speedups) / len(imp_speedups)) if imp_speedups else 1.0
    min_imp = min(imp_speedups) if imp_speedups else 1.0

    print(f"# FP8 Dense Duel Benchmark ({gpu_name}, {sm_count} SMs, L2={l2_bytes // (1024 * 1024)} MiB)")
    print(f"# Warmup={args.warmup}, Replays={args.replays}, InterleaveRounds={args.interleave_rounds}, Wall={wall_s:.1f}s")
    print(f"# GPU Before: {json.dumps(snap_before)}")
    print(f"# GPU After:  {json.dumps(snap_after)}")
    print()
    print("| shape (N,K) | mode | rows | imp | baseline us | variant us | ratio | GB/s cold (B->V) | warm us (B->V) | exact | row_inv |")
    print("|---|---|---:|:---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for r in records:
        n, k = r["shape"]
        imp_mark = "*" if r["important"] else ""
        ex_str = "yes" if r["exact"] else "NO"
        inv_str = "yes" if r["row_invariance_candidate"] else "NO"
        print(
            f"| {r['name']} ({n},{k}) | {r['mode']} | {r['rows']} | {imp_mark} | "
            f"{r['baseline_us']:.2f} | {r['candidate_us']:.2f} | {r['speedup']:.3f}x | "
            f"{r['baseline_gbps_cold']:.0f} -> {r['candidate_gbps_cold']:.0f} | "
            f"{r['baseline_warm_us']:.2f} -> {r['candidate_warm_us']:.2f} | {ex_str} | {inv_str} |"
        )
    print()
    print(f"Geomean speedup (all {len(all_speedups)} cases): {geomean_all:.4f}x")
    if imp_speedups:
        print(f"Geomean speedup (important {len(imp_speedups)} cases): {geomean_imp:.4f}x (min={min_imp:.4f}x)")

    artifact = {
        "gpu_name": gpu_name,
        "sm_count": sm_count,
        "l2_bytes": l2_bytes,
        "args": vars(args),
        "wall_seconds": round(wall_s, 2),
        "gpu_snapshot_before": snap_before,
        "gpu_snapshot_after": snap_after,
        "geomean_speedup_all": round(geomean_all, 4),
        "geomean_speedup_important": round(geomean_imp, 4),
        "min_speedup_important": round(min_imp, 4),
        "all_exact": all(r["exact"] for r in records),
        "all_row_invariant": all(r["row_invariance_candidate"] for r in records),
        "records": records,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"Wrote JSON artifact to {out_path}")


if __name__ == "__main__":
    main()
