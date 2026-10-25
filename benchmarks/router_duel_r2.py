#!/usr/bin/env python3
"""RTX kernel duel R1, Round 2 (final round): Gemini entrant vs Astra's router tile table.

Measures the five router shapes across live rows 1, 2, 4, 8, 16, 32, 48, 64:
  - 384 x 5120 (V4.1 backbone router)
  - 128 x 5120 (V4.1 dSpark drafter router)
  - 256 x 6144 (GLM 5.3 router)
  - 256 x 4096 (V4 Flash router)
  - 384 x 7168 (V4 Pro router)

Enforces the hard numerics gate (bitwise int32 equality against today's skinny M=1
for every width 1..64 on random, cancellation, dynamic, and adversarial inputs,
with NaN-prefilled unwritten tail rows preserved) and interleaved cold-L2
(512 MiB flush) timing across 4 blocks x 15 reps = 60 reps per cell.
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

from b12x.gemm.bf16_gemv._skinny import RowTiledSkinnyBf16Gemv, SkinnyBf16Gemv, skinny_config
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

# Astra's Round 3 tile table (baseline to beat).
ASTRA_TILES = {
    (384, 5120): ((16, 2, 4, 128), (64, 8, 4, 128)),
    (128, 5120): ((64, 2, 4, 128),),
    (256, 6144): ((16, 2, 4, 128), (64, 8, 4, 128)),
    (256, 4096): ((16, 4, 2, 256), (64, 8, 4, 256)),
    (384, 7168): ((16, 4, 2, 224), (64, 4, 4, 224)),
}

# Gemini Variant 1 (g1):
# - 384x5120: c2r8 for m<=16 (halves W loads at m8/m16), Astra's c8r4 for m>16
# - 128x5120: single-tile c4r4 (halves X loads vs c2r4)
# - 256x6144: c2r4 for m<=16, c8r8 for m>16 (halves W loads vs c8r4, 2 CTAs/SM)
# - 256x4096: c4r2 for m<=16, c8r8 for m>16 (halves W loads vs c8r4)
# - 384x7168: c4r2 for m<=16, c6r8 for m>16 (REG=146 -> 2 CTAs/SM, cuts m64 traffic 168M -> 98M)
G1_TILES = {
    (384, 5120): ((16, 2, 8, 128), (64, 8, 4, 128)),
    (128, 5120): ((64, 4, 4, 128),),
    (256, 6144): ((16, 2, 4, 128), (64, 8, 8, 128)),
    (256, 4096): ((16, 4, 2, 256), (64, 8, 8, 256)),
    (384, 7168): ((16, 4, 2, 224), (64, 6, 8, 224)),
}

# Gemini Variant 2 (g2):
# - 384x5120: c2r8 for m<=16, c6r8 for m>16 (REG=168 -> 3 CTAs/SM = 564 CTAs/wave, fits all 512 m64 CTAs in 1 wave!)
# - 128x5120: single-tile c2r8 (halves W loads at m>=8)
# - 256x6144: c2r4 for m<=32, c8r8 for m>32 (targets m48/m64 with c8r8 while keeping c2r4 through m32)
# - 256x4096: c4r2 for m<=32, c8r8 for m>32 (targets m48/m64 with c8r8)
# - 384x7168: c4r2 for m<=16, c8r8 for m>16 (halves both W and X traffic vs c4r4: 168M -> 84M)
G2_TILES = {
    (384, 5120): ((16, 2, 8, 128), (64, 6, 8, 128)),
    (128, 5120): ((64, 2, 8, 128),),
    (256, 6144): ((32, 2, 4, 128), (64, 8, 8, 128)),
    (256, 4096): ((32, 4, 2, 256), (64, 8, 8, 256)),
    (384, 7168): ((16, 4, 2, 224), (64, 8, 8, 224)),
}

# Gemini Variant 3 (g3):
# - 384x5120: Astra's c2r4 for m<=16, c8r8 for m>16 (cuts m48/m64 W traffic in half vs c8r4)
# - 128x5120: c2r4 for m<=16, c4r4 for m>16
# - 256x6144: c2r4 for m<=16, c4r8 for m>16 (REG=96 -> 5 CTAs/SM, halves W traffic vs c8r4)
# - 256x4096: c4r2 for m<=16, c4r8 for m>16 (REG=93 -> 2 CTAs/SM, halves W traffic vs c8r4)
# - 384x7168: c4r2 for m<=16, c8r4 for m>16 (REG=133 -> 2 CTAs/SM, halves X traffic vs c4r4)
G3_TILES = {
    (384, 5120): ((16, 2, 4, 128), (64, 8, 8, 128)),
    (128, 5120): ((16, 2, 4, 128), (64, 4, 4, 128)),
    (256, 6144): ((16, 2, 4, 128), (64, 4, 8, 128)),
    (256, 4096): ((16, 4, 2, 256), (64, 4, 8, 256)),
    (384, 7168): ((16, 4, 2, 224), (64, 8, 4, 224)),
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


def make_suites(e: int, h: int, device: str):
    suites = []

    # Mode 1: random normal
    gen_w1 = torch.Generator(device="cpu").manual_seed(10000 + e * 31 + h)
    w_rand = (torch.randn((e, h), generator=gen_w1) * 0.02).bfloat16().to(device)
    gen_x1 = torch.Generator(device="cpu").manual_seed(736 + e + h)
    x_rand = (torch.randn((64, h), generator=gen_x1) * 0.5).bfloat16().to(device)
    suites.append(("random", w_rand, x_rand))

    # Mode 2: cancellation (structured alternating-sign cancellation)
    w_canc = w_rand.clone()
    x_canc = x_rand.clone()
    w_canc[:, 1::2] = -w_canc[:, ::2]
    x_canc[:, 1::2] = x_canc[:, ::2]
    x_canc[:, ::257] *= 1.0078125
    suites.append(("cancellation", w_canc, x_canc))

    # Mode 3: dynamic range + zero row + repeated rows + adversarial scale sweep
    scale = torch.pow(2.0, (torch.arange(h, device=device) % 41 - 20).float())
    x_dyn = (x_rand.float() * scale).bfloat16()
    w_dyn = (w_rand.float() / scale).bfloat16()
    x_dyn[0].zero_()
    x_dyn[17].zero_()
    x_dyn[7].copy_(x_dyn[3])
    x_dyn[42].copy_(x_dyn[3])
    suites.append(("dynamic", w_dyn, x_dyn))

    return w_rand, x_rand, suites


def verify_numerics(arm_name: str, program, ref_program, e: int, h: int, suites) -> dict:
    out_buf = torch.empty((64, e), dtype=torch.float32, device="cuda")
    first_mismatch = None
    nan_tail_ok = True

    for suite_name, w, x in suites:
        ref_rows = []
        for r in range(64):
            row_out = torch.full((1, e), float("nan"), dtype=torch.float32, device="cuda")
            ref_program.launch(x[r : r + 1], w, row_out, scalars=(1,))
            ref_rows.append(row_out)
        torch.cuda.synchronize()
        ref_all = torch.cat(ref_rows, dim=0)

        for m in range(1, 65):
            out_buf.fill_(float("nan"))
            program.launch(x[:m], w, out_buf, scalars=(m,))
            torch.cuda.synchronize()

            if m < 64:
                if not torch.isnan(out_buf[m:]).all().item():
                    nan_tail_ok = False
                    if first_mismatch is None:
                        first_mismatch = f"{suite_name}:m{m}:nan_tail_overwritten"

            got_bits = out_buf[:m].view(torch.int32)
            ref_bits = ref_all[:m].view(torch.int32)
            if not torch.equal(got_bits, ref_bits):
                if first_mismatch is None:
                    diff_mask = got_bits != ref_bits
                    bad_idx = torch.nonzero(diff_mask, as_tuple=False)[0]
                    r_idx, c_idx = int(bad_idx[0]), int(bad_idx[1])
                    max_abs = float((out_buf[:m] - ref_all[:m]).abs().max().item())
                    first_mismatch = (
                        f"{suite_name}:m{m}:row{r_idx}:col{c_idx}:"
                        f"got=0x{int(got_bits[r_idx, c_idx]):08x}!=ref=0x{int(ref_bits[r_idx, c_idx]):08x}:"
                        f"max_abs={max_abs:.3e}"
                    )

    return {
        "exact_m1_to_m64": first_mismatch is None,
        "nan_tail_preserved": nan_tail_ok,
        "first_mismatch": first_mismatch,
    }


def build_arms(e: int, h: int, tmp_dir: Path):
    base_cfg = skinny_config(e, h, Float32)
    astra_tiles = ASTRA_TILES[(e, h)]
    g1_tiles = G1_TILES[(e, h)]
    g2_tiles = G2_TILES[(e, h)]
    g3_tiles = G3_TILES[(e, h)]

    specs = [
        ("old_skinny", SkinnyBf16Gemv(e, h, out_dtype=Float32, **base_cfg), {"type": "SkinnyBf16Gemv", **base_cfg}),
        ("astra", RowTiledSkinnyBf16Gemv(e, h, astra_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": astra_tiles}),
        ("astra_dup", RowTiledSkinnyBf16Gemv(e, h, astra_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": astra_tiles, "dup": True}),
        ("mma", _V41RouterScores(e, hidden=h), {"type": "_V41RouterScores", "tile_m": 64}),
        ("g1", RowTiledSkinnyBf16Gemv(e, h, g1_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": g1_tiles}),
        ("g2", RowTiledSkinnyBf16Gemv(e, h, g2_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": g2_tiles}),
        ("g3", RowTiledSkinnyBf16Gemv(e, h, g3_tiles, out_dtype=Float32), {"type": "RowTiledSkinnyBf16Gemv", "tiles": g3_tiles}),
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
                name=f"duel_r2_{arm_name}_{e}x{h}",
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
    parser.add_argument("--results-dir", type=Path, default=Path("/opt/cuteafd-target/results_r2"))
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
            print(f"  compiled {arm_name:12s} in {compile_seconds[arm_name]:5.2f}s | {'; '.join(usages)}", flush=True)

        w, x_bench, suites = make_suites(e, h, "cuda")
        ref_prog = programs["old_skinny"][0]
        numerics = {}
        for arm_name, (prog, _) in programs.items():
            num_res = verify_numerics(arm_name, prog, ref_prog, e, h, suites)
            numerics[arm_name] = num_res
            print(
                f"  numerics {arm_name:12s}: exact={num_res['exact_m1_to_m64']} "
                f"nan_tail={num_res['nan_tail_preserved']} first_mismatch={num_res['first_mismatch']}",
                flush=True,
            )
            if arm_name != "mma" and (not num_res["exact_m1_to_m64"] or not num_res["nan_tail_preserved"]):
                raise RuntimeError(f"Hard numerics gate failed for {arm_name} on {e}x{h}: {num_res}")

        arm_names = list(programs.keys())
        out_bench = torch.empty((64, e), dtype=torch.float32, device="cuda")

        shape_rows_data = {}
        for m in TIMED_ROWS:
            for arm_name in arm_names:
                programs[arm_name][0].launch(x_bench[:m], w, out_bench[:m], scalars=(m,))
            torch.cuda.synchronize()

            samples_us = {arm_name: [] for arm_name in arm_names}
            block_medians_us = {arm_name: [] for arm_name in arm_names}

            for blk in range(args.blocks):
                shift = blk % len(arm_names)
                order = arm_names[shift:] + arm_names[:shift]
                if blk % 2 == 1:
                    order = list(reversed(order))

                for arm_name in order:
                    prog = programs[arm_name][0]
                    blk_samples = []
                    for _ in range(args.reps_per_block):
                        flush.zero_()
                        start_ev = torch.cuda.Event(enable_timing=True)
                        end_ev = torch.cuda.Event(enable_timing=True)
                        start_ev.record()
                        prog.launch(x_bench[:m], w, out_bench[:m], scalars=(m,))
                        end_ev.record()
                        end_ev.synchronize()
                        us = start_ev.elapsed_time(end_ev) * 1000.0
                        samples_us[arm_name].append(us)
                        blk_samples.append(us)
                    block_medians_us[arm_name].append(round(statistics.median(blk_samples), 3))

            cell_summary = {}
            for arm_name in arm_names:
                s = sorted(samples_us[arm_name])
                n_s = len(s)
                med = statistics.median(s)
                p10 = s[max(0, int(0.10 * n_s))]
                p90 = s[min(n_s - 1, int(0.90 * n_s))]
                cell_summary[arm_name] = {
                    "median_us": round(med, 3),
                    "p10_us": round(p10, 3),
                    "p90_us": round(p90, 3),
                    "min_us": round(s[0], 3),
                    "max_us": round(s[-1], 3),
                    "block_medians_us": block_medians_us[arm_name],
                    "exact": numerics[arm_name]["exact_m1_to_m64"],
                    "config": programs[arm_name][1],
                }

            spread_us = round(
                abs(cell_summary["astra"]["median_us"] - cell_summary["astra_dup"]["median_us"]),
                3,
            )
            shape_rows_data[str(m)] = {
                "rows": m,
                "identical_config_spread_us": spread_us,
                "arms": cell_summary,
            }
            line_parts = [f"{a}={cell_summary[a]['median_us']:5.1f}us" for a in arm_names]
            print(f"  m={m:2d} (spread={spread_us:4.2f}us): " + "  ".join(line_parts), flush=True)

        shape_record = {
            "e": e,
            "h": h,
            "slug": slug,
            "label": label,
            "compile_seconds": compile_seconds,
            "resources": resources,
            "numerics": numerics,
            "cells": shape_rows_data,
        }
        all_results["shapes"][f"{e}x{h}"] = shape_record
        shape_file = args.results_dir / f"r2_{slug}_{e}x{h}.json"
        shape_file.write_text(json.dumps(shape_record, indent=2) + "\n")

    gpu_after = query_gpu_state()
    all_results["gpu_after"] = gpu_after
    print(f"\nGPU after: {gpu_after['nvidia_smi_gpu']}", flush=True)
    print(f"Active compute apps after: {gpu_after['nvidia_smi_compute_apps']}", flush=True)

    summary_file = args.results_dir / "summary_r2.json"
    summary_file.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"Saved results to {summary_file}", flush=True)


if __name__ == "__main__":
    main()
