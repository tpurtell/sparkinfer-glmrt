#!/usr/bin/env python3
"""RTX kernel duel R1: row-invariant decode router GEMV on SM120 (RTX PRO 6000).

Measures the five router shapes across live rows 1, 2, 4, 8, 16, 32, 48, 64:
  - 384 x 5120 (V4.1 backbone router)
  - 128 x 5120 (V4.1 dSpark drafter router)
  - 256 x 6144 (GLM 5.3 router)
  - 256 x 4096 (V4 Flash router)
  - 384 x 7168 (V4 Pro router)

Enforces the hard numerics gate (bitwise int32 equality against today's skinny M=1
for every width 1..64 on both random and adversarial inputs, with NaN-prefilled
unwritten tail rows preserved) and interleaved cold-L2 (512 MiB flush) timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from b12x.gemm.bf16_gemv._skinny import RowTiledSkinnyBf16Gemv, SkinnyBf16Gemv
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program, exportable_compilation
from b12x.moe._shared.v41_router import _V41RouterScores

SHAPES = (
    (384, 5120, "v41_e384", "V4.1 E384"),
    (128, 5120, "v41_e128", "V4.1 E128"),
    (256, 6144, "glm53", "GLM 5.3"),
    (256, 4096, "v4_flash", "V4 Flash"),
    (384, 7168, "v4_pro", "V4 Pro"),
)

TIMED_ROWS = (1, 2, 4, 8, 16, 32, 48, 64)


def baseline_skinny_config(n: int, k: int) -> dict:
    """Today's baseline skinny_config(n, k, Float32) before duel tuning."""
    if (n, k) == (256, 4096):
        return dict(cols=4, rows_per_tile=2, threads=256)
    if (n, k) == (384, 7168):
        return dict(cols=4, rows_per_tile=2, threads=224)
    return dict(cols=1, rows_per_tile=8, threads=128)


# Variant 1: Single-kernel resident-weight CTA (1 compiled kernel per shape, no row-tile dispatch).
V1_SINGLE_RESIDENT = {
    (384, 5120): dict(cols=2, rows_per_tile=8, threads=128, cta_rows=64),
    (128, 5120): dict(cols=1, rows_per_tile=8, threads=128, cta_rows=64),
    (256, 6144): dict(cols=1, rows_per_tile=8, threads=128, cta_rows=64),
    (256, 4096): dict(cols=4, rows_per_tile=4, threads=256, cta_rows=16),
    (384, 7168): dict(cols=2, rows_per_tile=8, threads=224, cta_rows=64),
}

# Variant 2: Row-tiled high-cols single-batch tiles (with spill-free step loop for outputs >= 32).
V2_ROW_TILED_COL4 = {
    (384, 5120): ((4, 1, 8, 128), (16, 4, 8, 128), (64, 4, 16, 128)),
    (128, 5120): ((8, 1, 8, 128), (16, 1, 16, 128), (64, 2, 16, 128)),
    (256, 6144): ((4, 1, 8, 128), (16, 2, 8, 128), (64, 4, 16, 128)),
    (256, 4096): ((8, 4, 2, 256), (16, 4, 4, 256), (64, 4, 8, 256)),
    (384, 7168): ((4, 4, 2, 224), (16, 4, 4, 224), (64, 4, 16, 224)),
}

# Variant 3: Row-tiled hybrid (today's exact tile at low rows <= 8, resident-W 2-wave tile above).
V3_ROW_TILED_RESIDENT = {
    (384, 5120): ((8, 1, 8, 128), (64, 2, 8, 128, 64)),
    (128, 5120): ((8, 1, 8, 128), (32, 1, 8, 128, 16), (64, 1, 8, 128, 32)),
    (256, 6144): ((8, 1, 8, 128), (32, 2, 8, 128, 16), (64, 2, 8, 128, 32)),
    (256, 4096): ((8, 4, 2, 256), (16, 4, 4, 256), (64, 4, 8, 256, 16)),
    (384, 7168): ((8, 4, 2, 224), (32, 2, 8, 224, 32), (64, 2, 8, 224, 64)),
}


class _Launcher:
    def __init__(self, op, config_desc):
        self.op = op
        self.config_desc = config_desc

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, logits: cute.Pointer,
                 live_rows: Int32, stream: cuda.CUstream):
        self.op(x, w, logits, live_rows, stream)


def _cuda_images(data: bytes):
    offset = 0
    while (offset := data.find(b"\x7fELF", offset)) >= 0:
        start = offset
        offset += 4
        if len(data) - start < 64:
            continue
        blob = memoryview(data)[start:]
        if bytes(blob[:6]) != b"\x7fELF\x02\x01":
            continue
        if struct.unpack_from("<H", blob, 18)[0] != 190:
            continue
        phoff, shoff = struct.unpack_from("<QQ", blob, 32)
        phsize, phcount, shsize, shcount = struct.unpack_from("<HHHH", blob, 54)
        if phsize != 56 or shsize != 64 or not shcount:
            continue
        end = max(64, phoff + phsize * phcount, shoff + shsize * shcount)
        for i in range(shcount):
            entry = shoff + i * shsize
            kind = struct.unpack_from("<I", blob, entry + 4)[0]
            pos, size = struct.unpack_from("<QQ", blob, entry + 24)
            if kind != 8:
                end = max(end, pos + size)
        for i in range(phcount):
            entry = phoff + i * phsize
            pos = struct.unpack_from("<Q", blob, entry + 8)[0]
            size = struct.unpack_from("<Q", blob, entry + 32)[0]
            end = max(end, pos + size)
        yield bytes(blob[:end])
        offset = start + end


def inspect_resources(program, tmp_dir: Path, tag: str) -> list[dict]:
    out_dir = tmp_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    program.export_to_c(str(out_dir), "prog", "cuteafd_prog")
    obj_bytes = (out_dir / "prog.o").read_bytes()
    records = []
    for idx, cubin in enumerate(_cuda_images(obj_bytes)):
        cubin_path = out_dir / f"prog.{idx}.cubin"
        cubin_path.write_bytes(cubin)
        raw = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin_path)],
            text=True,
        )
        funcs = []
        cur_fn = None
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith("Function "):
                cur_fn = s.removeprefix("Function ").rstrip(":")
            elif s.startswith("REG:") and cur_fn:
                funcs.append({"function": cur_fn, "usage": s})
                cur_fn = None
        records.append({
            "cubin_index": idx,
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "kernels": funcs,
        })
    return records


def query_gpu_state() -> dict:
    props = torch.cuda.get_device_properties(0)
    smi_gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,pstate",
         "--format=csv,noheader"],
        text=True,
    ).strip()
    smi_apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "device_name": props.name,
        "capability": [props.major, props.minor],
        "multi_processor_count": props.multi_processor_count,
        "l2_cache_size": getattr(props, "L2_cache_size", None),
        "nvidia_smi_gpu": smi_gpu,
        "nvidia_smi_compute_apps": [line for line in smi_apps.splitlines() if line.strip()],
        "pid": os.getpid(),
    }


def make_inputs(e: int, h: int, device: str):
    gen_w = torch.Generator(device="cpu").manual_seed(10000 + e * 31 + h)
    w = (torch.randn((e, h), generator=gen_w) * 0.02).bfloat16().to(device)

    # Suite 1: random normal
    gen_x1 = torch.Generator(device="cpu").manual_seed(736 + e + h)
    x_rand = (torch.randn((64, h), generator=gen_x1) * 0.5).bfloat16().to(device)

    # Suite 2: adversarial (zero rows, repeated rows, scale sweep 1e-4..1e2, cancellation)
    gen_x2 = torch.Generator(device="cpu").manual_seed(9999 + e + h)
    x_adv = torch.randn((64, h), generator=gen_x2, dtype=torch.float32)
    scales = torch.logspace(-4, 2, 64, dtype=torch.float32).unsqueeze(1)
    x_adv = (x_adv * scales).bfloat16().to(device)
    x_adv[0].zero_()
    x_adv[17].zero_()
    x_adv[7].copy_(x_adv[3])
    x_adv[42].copy_(x_adv[3])
    # Cancellation / alternating sign large magnitudes
    half = h // 2
    x_adv[15, :half] = torch.full((half,), 64.0, dtype=torch.bfloat16, device=device)
    x_adv[15, half:] = torch.full((h - half,), -64.0, dtype=torch.bfloat16, device=device)
    x_adv[53, 0::2] = 32.0
    x_adv[53, 1::2] = -32.0

    return w, [("random", x_rand), ("adversarial", x_adv)]


def verify_numerics(arm_name: str, program, ref_program, e: int, h: int, w: torch.Tensor, suites) -> dict:
    if arm_name == "mma":
        # MMA is known row-variant above 32 rows; still check M=1..64 vs M=1 reference to record status
        pass
    pad_rows = 2
    failures = []
    nan_failures = []
    for suite_name, x in suites:
        # Compute M=1 reference per row using today_skinny
        ref_rows = []
        for r in range(64):
            buf = torch.full((1 + pad_rows, e), float("nan"), dtype=torch.float32, device=x.device)
            ref_program.launch(x[r:r + 1].contiguous(), w, buf[:1], scalars=(1,))
            torch.cuda.synchronize()
            ref_rows.append(buf[0].clone())
        ref = torch.stack(ref_rows, dim=0)

        for m in range(1, 65):
            buf = torch.full((m + pad_rows, e), float("nan"), dtype=torch.float32, device=x.device)
            program.launch(x[:m].contiguous(), w, buf[:m], scalars=(m,))
            torch.cuda.synchronize()
            if not bool(torch.isnan(buf[m:]).all()):
                nan_failures.append(f"{suite_name}:m{m}")
            if not torch.equal(buf[:m].view(torch.int32), ref[:m].view(torch.int32)):
                failures.append(f"{suite_name}:m{m}")
    return {
        "bitwise_exact_all_1_to_64": len(failures) == 0 and len(nan_failures) == 0,
        "first_mismatch": failures[0] if failures else None,
        "mismatch_count": len(failures),
        "nan_tail_preserved": len(nan_failures) == 0,
    }


def build_arms(e: int, h: int, tmp_dir: Path):
    base_cfg = baseline_skinny_config(e, h)
    v1_cfg = V1_SINGLE_RESIDENT[(e, h)]
    v2_tiles = V2_ROW_TILED_COL4[(e, h)]
    v3_tiles = V3_ROW_TILED_RESIDENT[(e, h)]

    specs = [
        ("today_skinny", SkinnyBf16Gemv(e, h, out_dtype=Float32, **base_cfg), {"type": "SkinnyBf16Gemv", **base_cfg}),
        ("today_skinny_dup", SkinnyBf16Gemv(e, h, out_dtype=Float32, **base_cfg), {"type": "SkinnyBf16Gemv_dup", **base_cfg}),
        ("mma", _V41RouterScores(e, hidden=h), {"type": "_V41RouterScores", "tile_m": 64}),
        ("v1_single_resident", SkinnyBf16Gemv(e, h, out_dtype=Float32, **v1_cfg), {"type": "SkinnyBf16Gemv", **v1_cfg}),
        ("v2_row_tiled_col4", RowTiledSkinnyBf16Gemv(e, h, v2_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": v2_tiles}),
        ("v3_row_tiled_resident", RowTiledSkinnyBf16Gemv(e, h, v3_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": v3_tiles}),
    ]

    operands = (
        Operand("x", torch.bfloat16, f"[rows,{h}]"),
        Operand("w", torch.bfloat16, f"[{e},{h}]"),
        Operand("logits", torch.float32, f"[rows,{e}]", "out"),
    )
    scalars = (Scalar("live_rows"),)

    programs = {}
    compile_seconds = {}
    resources = {}
    with exportable_compilation():
        for arm_name, op, desc in specs:
            t0 = time.monotonic()
            prog = compile_program(
                _Launcher(op, desc),
                name=f"duel_r1_{arm_name}_{e}x{h}",
                operands=operands,
                scalars=scalars,
                key=(arm_name, e, h, repr(desc)),
            )
            compile_seconds[arm_name] = round(time.monotonic() - t0, 3)
            programs[arm_name] = (prog, desc)
            resources[arm_name] = inspect_resources(prog, tmp_dir, f"{e}x{h}_{arm_name}")
    return programs, compile_seconds, resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=4, help="Interleaved blocks per (shape, rows) cell")
    parser.add_argument("--reps-per-block", type=int, default=15, help="Timed L2-flushed launches per block per arm")
    parser.add_argument("--results-dir", type=Path, default=Path("/opt/cuteafd-target/results"))
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.results_dir / "tmp_inspect"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    gpu_before = query_gpu_state()
    print(f"GPU before: {gpu_before['nvidia_smi_gpu']}", flush=True)
    print(f"Active compute apps before: {gpu_before['nvidia_smi_compute_apps']}", flush=True)

    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    flush.zero_()
    torch.cuda.synchronize()

    all_results = {
        "gpu_before": gpu_before,
        "blocks": args.blocks,
        "reps_per_block": args.reps_per_block,
        "total_reps_per_cell": args.blocks * args.reps_per_block,
        "shapes": {},
    }

    for e, h, slug, label in SHAPES:
        print(f"\n=== {label} ({e} x {h}) ===", flush=True)
        programs, compile_seconds, resources = build_arms(e, h, tmp_dir)
        for arm_name, res_list in resources.items():
            usages = [k["usage"] for r in res_list for k in r["kernels"]]
            print(f"  compiled {arm_name:22s} in {compile_seconds[arm_name]:5.2f}s | {'; '.join(usages)}", flush=True)

        w, suites = make_inputs(e, h, "cuda")
        ref_prog = programs["today_skinny"][0]
        numerics = {}
        for arm_name, (prog, _) in programs.items():
            num_res = verify_numerics(arm_name, prog, ref_prog, e, h, w, suites)
            numerics[arm_name] = num_res
            print(f"  numerics {arm_name:22s}: exact={num_res['bitwise_exact_all_1_to_64']} "
                  f"nan_tail={num_res['nan_tail_preserved']} first_mismatch={num_res['first_mismatch']}", flush=True)
            if arm_name != "mma" and not num_res["bitwise_exact_all_1_to_64"]:
                raise RuntimeError(f"Hard numerics gate failed for {label} {arm_name}: {num_res}")

        x_bench = suites[0][1]
        arm_names = list(programs.keys())
        cells = {}

        for rows in TIMED_ROWS:
            x_slice = x_bench[:rows].contiguous()
            outs = {name: torch.empty((rows, e), dtype=torch.float32, device="cuda") for name in arm_names}

            # One untimed warm launch per arm before timing
            for name in arm_names:
                programs[name][0].launch(x_slice, w, outs[name], scalars=(rows,))
            torch.cuda.synchronize()

            samples = {name: [] for name in arm_names}
            block_medians = {name: [] for name in arm_names}

            for blk in range(args.blocks):
                order = arm_names if (blk % 2 == 0) else list(reversed(arm_names))
                for name in order:
                    prog = programs[name][0]
                    out_t = outs[name]
                    blk_times = []
                    for _ in range(args.reps_per_block):
                        flush.zero_()
                        ev_start = torch.cuda.Event(enable_timing=True)
                        ev_end = torch.cuda.Event(enable_timing=True)
                        ev_start.record()
                        prog.launch(x_slice, w, out_t, scalars=(rows,))
                        ev_end.record()
                        torch.cuda.synchronize()
                        blk_times.append(ev_start.elapsed_time(ev_end) * 1000.0)
                    samples[name].extend(blk_times)
                    block_medians[name].append(round(statistics.median(blk_times), 3))

            cell_arms = {}
            for name in arm_names:
                s = sorted(samples[name])
                n_s = len(s)
                cell_arms[name] = {
                    "median_us": round(statistics.median(s), 3),
                    "min_us": round(s[0], 3),
                    "p25_us": round(s[n_s // 4], 3),
                    "p75_us": round(s[(3 * n_s) // 4], 3),
                    "max_us": round(s[-1], 3),
                    "iqr_us": round(s[(3 * n_s) // 4] - s[n_s // 4], 3),
                    "block_medians_us": block_medians[name],
                    "bitwise_exact": numerics[name]["bitwise_exact_all_1_to_64"],
                }

            base_med = cell_arms["today_skinny"]["median_us"]
            dup_med = cell_arms["today_skinny_dup"]["median_us"]
            spread_us = round(abs(base_med - dup_med), 3)
            cells[str(rows)] = {
                "rows": rows,
                "identical_config_spread_us": spread_us,
                "arms": cell_arms,
            }
            summary_str = "  ".join(
                f"{n}={cell_arms[n]['median_us']:.1f}us" for n in arm_names
            )
            print(f"  m={rows:2d} (spread={spread_us:.2f}us): {summary_str}", flush=True)

        shape_record = {
            "label": label,
            "slug": slug,
            "experts": e,
            "hidden": h,
            "compile_seconds": compile_seconds,
            "configs": {name: desc for name, (_, desc) in programs.items()},
            "resources": resources,
            "numerics": numerics,
            "cells": cells,
        }
        all_results["shapes"][slug] = shape_record
        (args.results_dir / f"{slug}.json").write_text(json.dumps(shape_record, indent=2) + "\n")

    gpu_after = query_gpu_state()
    all_results["gpu_after"] = gpu_after
    print(f"\nGPU after: {gpu_after['nvidia_smi_gpu']}", flush=True)
    print(f"Active compute apps after: {gpu_after['nvidia_smi_compute_apps']}", flush=True)
    (args.results_dir / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
