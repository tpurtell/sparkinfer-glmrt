#!/usr/bin/env python3
"""Race retained native SIMT and 32x64x64 MMA BF16 projections, not dispatch.

Example (run on an explicitly isolated GPU):
  CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_bf16_projection.py \
    --cases wo-a,compressor --rows 8,9,16,24 --output /tmp/projection.json

Use --rows all for 1/6/8/9/16/24/64/256/4096, --output-dtypes bf16,fp32,
--layouts contiguous,row-strided, and --cases vocab --vocab-width 32768 for
expanded diagnosis. Large-M SIMT and vocabulary sweeps are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as compile_native,
    run_compiled,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils import current_cuda_stream
from b12x.gemm.bf16_gemv import _kernel as native
from b12x.gemm.bf16_gemv import query_from_call
from b12x.gemm.bf16_gemv._tuning import TUNING
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
    resolve_l2_flush_bytes,
)

CASES = {
    "wo-a": (1024, 4096),
    "compressor": (512, 5120),
    "dspark-aux": (5120, 15360),
    "router": (256, 5120),
}
ALL_ROWS = (1, 6, 8, 9, 16, 24, 64, 256, 4096)
D1_SHAPES = (
    (24752, 7168),
    (6144, 7168),
    (7168, 6144),
    (7168, 3072),
    (3584, 7168),
    (7168, 3584),
    (2112, 7168),
)
D1_BASE_ROWS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40, 48, 64)
# Exact c324b67f baseline table for Kimi D1 shapes (only 7168x6144 had an entry).
C324B67F_DECODE_TMA_TILES: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {
    (7168, 6144): ((16, 1, 4, 16, 64, 6), (32, 2, 2, 32, 32, 10)),
}
# C1 decode call counts per emitted token from kimi-bf16-baseline/kernel-catalog.json (255 tokens).
D1_C1_CALLS_PER_TOKEN: dict[tuple[int, int], dict[str, float]] = {
    (24752, 7168): {"total": 5037 / 255.0, "m2": 69 / 255.0, "m4": 759 / 255.0, "m5": 3795 / 255.0, "m6": 414 / 255.0},
    (6144, 7168): {"total": 8468 / 255.0, "m2": 116 / 255.0, "m4": 1276 / 255.0, "m5": 6380 / 255.0, "m6": 696 / 255.0},
    (7168, 6144): {"total": 6789 / 255.0, "m2": 93 / 255.0, "m4": 1023 / 255.0, "m5": 5115 / 255.0, "m6": 558 / 255.0},
    (7168, 3072): {"total": 6716 / 255.0, "m2": 92 / 255.0, "m4": 1012 / 255.0, "m5": 5060 / 255.0, "m6": 552 / 255.0},
    (3584, 7168): {"total": 3358 / 255.0, "m2": 46 / 255.0, "m4": 506 / 255.0, "m5": 2530 / 255.0, "m6": 276 / 255.0},
    (7168, 3584): {"total": 3358 / 255.0, "m2": 46 / 255.0, "m4": 506 / 255.0, "m5": 2530 / 255.0, "m6": 276 / 255.0},
    (2112, 7168): {"total": 1752 / 255.0, "m2": 24 / 255.0, "m4": 264 / 255.0, "m5": 1320 / 255.0, "m6": 144 / 255.0},
}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
ARMS = ("simt", "mma")


def _parse_shapes_arg(raw: str, parser: argparse.ArgumentParser) -> list[tuple[int, int]]:
    if raw == "d1":
        return list(D1_SHAPES)
    shapes: list[tuple[int, int]] = []
    for item in raw.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 2:
            parser.error(f"invalid shape {item!r}: expected NxK")
        try:
            n, k = int(parts[0]), int(parts[1])
        except ValueError:
            parser.error(f"invalid shape {item!r}: N and K must be integers")
        if n <= 0 or k <= 0 or n % 8 != 0 or k % 64 != 0:
            parser.error(f"invalid shape {n}x{k}: N must be multiple of 8 and K multiple of 64")
        if (n, k) in shapes:
            parser.error(f"duplicate shape {n}x{k}")
        shapes.append((n, k))
    if not shapes:
        parser.error("--shapes must not be empty")
    return shapes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("native", "cuteafd-decode"),
        default="native",
        help="native (default SIMT vs 32x64x64 MMA) or cuteafd-decode (RoutedBf16Projection / DECODE_TMA_TILES)",
    )
    parser.add_argument(
        "--shapes",
        default="d1",
        help="For cuteafd-decode: comma-separated NxK shapes, or 'd1' for all 7 Kimi D1 shapes",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="For cuteafd-decode: measure only the c324b67f baseline route",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=None,
        help="For cuteafd-decode: JSON file mapping NxK to candidate tiles_by_rows or sweep list",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="For cuteafd-decode with --candidate-json: race multiple named candidates per shape",
    )
    parser.add_argument(
        "--auto-switch-neighbors",
        action="store_true",
        help="For cuteafd-decode: automatically add limit-1, limit, limit+1 for every row-band switch",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate CLI and Python syntax without touching CUDA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON file outside the repository",
    )
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help="Comma-separated wo-a,compressor,dspark-aux,router,vocab",
    )
    parser.add_argument(
        "--rows",
        default="1,8,9,16,24,256",
        help="Comma-separated positive M values, 'all', or 'd1'",
    )
    parser.add_argument(
        "--output-dtypes", default="bf16", help="bf16, fp32, or bf16,fp32"
    )
    parser.add_argument(
        "--layouts",
        default="contiguous",
        help="contiguous,row-strided; strided rows have 16 padding elements",
    )
    parser.add_argument("--vocab-width", type=int, default=32768)
    parser.add_argument("--vocab-hidden", type=int, default=5120)
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Even number of paired blocks; alternate AB/BA",
    )
    parser.add_argument(
        "--replays", type=int, default=8, help="Graph samples per arm per paired block"
    )
    parser.add_argument(
        "--launches",
        type=int,
        default=1,
        help="Native calls per graph replay; samples normalized per call",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Calls per arm before capture"
    )
    parser.add_argument(
        "--cache",
        choices=("warm", "cold"),
        default="warm",
        help="Cold flushes L2 before each replay, outside timing",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="0 selects common helper's automatic L2 flush size",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)
    for name, choices in (
        ("cases", (*CASES, "vocab")),
        ("output_dtypes", DTYPES),
        ("layouts", ("contiguous", "row-strided")),
    ):
        values = getattr(args, name).split(",")
        if (
            not values
            or len(set(values)) != len(values)
            or any(value not in choices for value in values)
        ):
            parser.error(
                f"invalid or duplicate --{name.replace('_', '-')}: choose from {','.join(choices)}"
            )
        setattr(args, name, values)
    try:
        if args.rows == "all":
            args.rows = list(ALL_ROWS)
        elif args.rows == "d1":
            args.rows = list(D1_BASE_ROWS)
        else:
            args.rows = [int(value) for value in args.rows.split(",")]
    except ValueError:
        parser.error("--rows must contain integers, 'all', or 'd1'")
    if (
        not args.rows
        or any(value < 1 or value >= 2**31 for value in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("--rows must be distinct positive int32 values")
    if args.mode == "cuteafd-decode":
        args.parsed_shapes = _parse_shapes_arg(args.shapes, parser)
        if args.candidate_json is not None:
            if not args.candidate_json.is_file():
                parser.error(f"--candidate-json file not found: {args.candidate_json}")
            args.candidate_spec = json.loads(args.candidate_json.read_text())
        else:
            args.candidate_spec = None
        if args.sweep and args.candidate_spec is None:
            parser.error("--sweep requires --candidate-json")
        if args.baseline_only and args.candidate_spec is not None:
            parser.error("--baseline-only cannot be combined with --candidate-json")
    if not args.baseline_only and (args.repeats < 2 or args.repeats % 2):
        parser.error("--repeats must be positive and even for balanced arm ordering")
    if args.baseline_only and args.repeats < 1:
        parser.error("--repeats must be positive")
    if min(args.replays, args.launches, args.warmup) < 1 or args.l2_flush_bytes < 0:
        parser.error(
            "--replays, --launches, --warmup must be positive; flush bytes nonnegative"
        )
    if min(args.vocab_width, args.vocab_hidden) < 16:
        parser.error("vocabulary N and K must be at least 16 for the native MMA arm")
    if args.cache == "cold" and args.launches != 1:
        parser.error(
            "cold-L2 requires --launches 1 so every measured projection follows eviction"
        )
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error("--output must be outside the repository")
    if args.output.exists() and not args.preflight:
        parser.error(f"refusing to overwrite {args.output}")
    return args


def toolchain():
    versions = {}
    for name in (
        "torch",
        "nvidia-cutlass-dsl",
        "cuda-python",
        "cuda-bindings",
        "triton",
        "b12x",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "torch_git": torch.version.git_version,
    }


def oracle(x, weight):
    # Full FP32 contraction of the actual BF16 operands, never a timed arm.
    # Column chunking bounds temporary FP32 weights even for vocabulary widths.
    expected = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32
    )
    x32 = x.float()
    for start in range(0, weight.shape[0], 512):
        stop = min(start + 512, weight.shape[0])
        expected[:, start:stop] = x32 @ weight[start:stop].float().T
    torch.cuda.synchronize()
    if (
        not torch.isfinite(expected).all().item()
        or not (expected.abs().amax(dim=1) > 0).all().item()
    ):
        raise AssertionError(
            "FP32 oracle must be finite with nonzero output in every row"
        )
    return expected


def check_output(out, expected, label):
    actual = out.float()
    finite = bool(torch.isfinite(actual).all().item())
    nonzero_rows = bool((actual.abs().amax(dim=1) > 0).all().item())
    if not finite or not nonzero_rows:
        raise AssertionError(
            f"{label}: output is nonfinite, unwritten, or has an all-zero row"
        )
    # BF16 rounding is part of the output contract, not an FP32 accumulation error.
    target = expected.to(out.dtype).float()
    rtol, atol = (0.0, 5e-5) if out.dtype == torch.bfloat16 else (2e-4, 3e-5)
    if out.dtype == torch.bfloat16:
        # Independent FP32 summation orders can straddle a BF16 midpoint.
        # Permit one representable BF16 step, plus a small cancellation floor.
        ulps = (
            out.view(torch.int16).int() - expected.to(out.dtype).view(torch.int16).int()
        ).abs()
        bad = (ulps > 1) & ((actual - target).abs() > atol)
        if bad.any().item():
            raise AssertionError(
                f"{label}: {bad.sum().item()} outputs exceed one BF16 ULP "
                f"and the {atol} cancellation floor"
            )
    else:
        torch.testing.assert_close(
            actual, target, rtol=rtol, atol=atol, msg=lambda msg: f"{label}: {msg}"
        )
    error = actual - expected
    return {
        "finite": finite,
        "every_row_nonzero": nonzero_rows,
        "all_elements_checked": out.numel(),
        "rtol": rtol,
        "atol": atol,
        "bf16_ulp_budget": 1 if out.dtype == torch.bfloat16 else None,
        "oracle": "FP32 x @ weight.T with TF32 disabled; BF16 allows one ULP plus cancellation floor",
        "max_abs_error_vs_fp32": float(error.abs().max().item()),
        "relative_rms_error_vs_fp32": float(
            (error.square().mean() / expected.square().mean()).sqrt().item()
        ),
    }


def run_case(args, name, rows, dtype_name, layout, case_index, flush):
    n, k = (args.vocab_width, args.vocab_hidden) if name == "vocab" else CASES[name]
    padding = 16 if layout == "row-strided" else 0
    # A geometry-specific seed keeps operands stable when selecting a CLI subset.
    seed = args.seed + rows * 1000003 + n * 1009 + k
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty((rows, k + padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    weight = torch.empty((n, k + padding), device="cuda", dtype=torch.bfloat16)[:, :k]
    x.normal_(generator=generator)
    weight.normal_(std=1 / math.sqrt(k), generator=generator)
    out = torch.empty((rows, n + padding), device="cuda", dtype=DTYPES[dtype_name])[
        :, :n
    ]
    native._validate(x, weight, out, None)
    key = (n, k, str(x.dtype), str(weight.dtype), str(out.dtype), 8)
    default = TUNING.configure(query_from_call(x, weight, out=out), device=None).default
    pointer_types = (x.dtype, weight.dtype, x.dtype, out.dtype)
    pointers = tuple(
        native.make_ptr(
            native._DTYPES[dtype],
            16,
            native.cute.AddressSpace.gmem,
            assumed_align=dtype.itemsize,
        )
        for dtype in pointer_types
    )
    # Compile the native entrypoints directly to compare both implementations.
    compiled = tuple(
        compile_native(
            kernel,
            *pointers,
            native.Int32(1),
            native.Int64(k),
            native.Int64(k),
            native.Int64(n),
            native.Int64(1),
            native.Int64(1),
            native.Int32(0),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "benchmark.bf16_projection",
                1,
                (*key, arm),
            ),
        )
        for arm, kernel in zip(
            ARMS,
            (
                native.SmallNGemvKernel(n, k, True, False),
                native.Bf16GemmKernel(n, k, False),
            ),
            strict=True,
        )
    )
    vector_loads = int(
        x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and x.stride(0) % 8 == 0
        and weight.stride(0) % 8 == 0
        and x.stride(1) == weight.stride(1) == 1
        and k % 8 == 0
    )
    launch_args = (
        native._pointer(x),
        native._pointer(weight),
        native._pointer(x),
        native._pointer(out),
        rows,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        x.stride(1),
        weight.stride(1),
        vector_loads,
    )

    def launch_kernel(kernel):
        # torch.cuda.graph uses its own capture stream; never bind that stream
        # before entering capture, even though pointers and kernels are fixed.
        run_compiled(kernel, (*launch_args, current_cuda_stream()))

    launches = {
        arm: partial(launch_kernel, kernel)
        for arm, kernel in zip(ARMS, compiled, strict=True)
    }
    expected = oracle(x, weight)
    correctness = {}
    initial_order = ARMS if case_index % 2 == 0 else ARMS[::-1]
    for arm in initial_order:
        out.fill_(float("nan"))
        for _ in range(args.warmup):
            launches[arm]()
        torch.cuda.synchronize()
        correctness[arm] = {"eager": check_output(out, expected, arm)}

    graphs = {}
    blocks = []
    samples = {arm: [] for arm in ARMS}
    with kernel_resolution_guard("BF16 projection native SIMT/MMA graph capture and paired replay"):
        for arm in initial_order:

            def repeated(launch=launches[arm]):
                for _ in range(args.launches):
                    launch()

            graphs[arm] = capture_cuda_graph(repeated, warmup=0)
            out.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            correctness[arm]["graph_before_timing"] = check_output(out, expected, arm)
        gpu_before = nvidia_smi_gpu_mode_snapshot()
        for repeat in range(args.repeats):
            order = initial_order if repeat % 2 == 0 else initial_order[::-1]
            block = {"repeat": repeat, "order": list(order), "replay_us": {}}
            for arm in order:
                raw = bench_cuda_graph(
                    graphs[arm], replays=args.replays, l2_flush=flush
                )["replay_us"]
                if any(not math.isfinite(value) or value <= 0 for value in raw):
                    raise RuntimeError(f"Invalid CUDA event samples for {arm}: {raw}")
                block["replay_us"][arm] = raw
                samples[arm].extend(value / args.launches for value in raw)
            block["simt_over_mma_ratio"] = statistics.median(
                block["replay_us"]["simt"]
            ) / statistics.median(block["replay_us"]["mma"])
            blocks.append(block)
        gpu_after = nvidia_smi_gpu_mode_snapshot()
        for arm in ARMS:
            out.fill_(float("nan"))
            graphs[arm].replay()
            torch.cuda.synchronize()
            correctness[arm]["graph_after_timing"] = check_output(out, expected, arm)
    for graph in graphs.values():
        graph.reset()
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    ratio = medians["simt"] / medians["mma"]
    return {
        "case": name,
        "M": rows,
        "N": n,
        "K": k,
        "seed": seed,
        "operand_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "output_dtype": dtype_name,
        "layout": layout,
        "strides": {
            "x": list(x.stride()),
            "weight": list(weight.stride()),
            "out": list(out.stride()),
        },
        "shared_operands_and_output_between_arms": True,
        "simt_vector_loads": bool(vector_loads),
        "cache_key": list(key),
        "implementations": {
            "simt": "SmallNGemvKernel",
            "mma": "Bf16GemmKernel(32,64,64)",
        },
        "production_policy_arm": default.backend,
        "production_policy_regresses_vs_simt": (
            ratio < 1 if default.backend == "mma" else
            False if default.backend == "simt" else None
        ),
        "correctness": correctness,
        "median_us_per_projection": medians,
        "simt_over_mma_ratio": ratio,
        "faster_arm": "mma" if ratio > 1 else "simt" if ratio < 1 else "tie",
        "mma_winning_blocks": sum(block["simt_over_mma_ratio"] > 1 for block in blocks),
        "samples_us_per_projection": samples,
        "blocks": blocks,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def _bits_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    raise TypeError(f"unsupported output dtype for bit view: {tensor.dtype}")


def _sample_stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)

    def pct(q: float) -> float:
        pos = q * (count - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        frac = pos - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    p25, p75 = pct(0.25), pct(0.75)
    return {
        "count": count,
        "median_us": float(statistics.median(ordered)),
        "iqr_us": float(p75 - p25),
        "p10_us": pct(0.10),
        "p25_us": p25,
        "p75_us": p75,
        "p90_us": pct(0.90),
        "min_us": float(ordered[0]),
        "max_us": float(ordered[-1]),
        "mean_us": float(statistics.fmean(ordered)),
        "stdev_us": float(statistics.stdev(ordered)) if count > 1 else 0.0,
    }


def _build_cuteafd_arm_program(
    arm_name: str,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    spec: dict[str, object] | None,
):
    import cutlass
    from b12x.gemm.bf16_gemv._skinny import RoutedBf16Projection, TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    cute_dtype = cutlass.Float32 if out_dtype == torch.float32 else cutlass.BFloat16
    if spec is None:
        # Production route from current checkout (uses live DECODE_TMA_TILES).
        launch_obj = RoutedBf16Projection(n, k, out_dtype=cute_dtype, max_skinny_rows=0)
        desc = {"kind": "RoutedBf16Projection(max_skinny_rows=0)", "tiles_by_rows": [list(t) for t in launch_obj.tiles_by_rows]}
    else:
        tiles = tuple(tuple(int(v) for v in entry) for entry in spec.get("tiles_by_rows", ()))
        base_kw = dict(spec.get("base", {}))
        if "warp_layout" in base_kw and base_kw["warp_layout"] is not None:
            base_kw["warp_layout"] = tuple(int(v) for v in base_kw["warp_layout"])
        launch_obj = TmaBf16Projection(n, k, out_dtype=cute_dtype, tiles_by_rows=tiles, **base_kw)
        desc = {"kind": "TmaBf16Projection", "tiles_by_rows": [list(t) for t in tiles], "base": base_kw}
    prog = compile_program(
        launch_obj,
        name="bench_d1_proj",
        operands=(
            Operand("x", torch.bfloat16, f"[rows,{k}]"),
            Operand("w", torch.bfloat16, f"[{n},{k}]"),
            Operand("out", out_dtype, f"[rows,{n}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=(n, k, str(out_dtype), launch_obj.key()),
    )
    return prog, desc, tuple(launch_obj.tiles_by_rows)


def _verify_cuteafd_row_invariance(
    prog,
    *,
    w: torch.Tensor,
    out_dtype: torch.dtype,
    seed: int,
    extra_rows: tuple[int, ...] = (),
) -> dict[str, object]:
    n, k = w.shape
    gate_rows = tuple(sorted({1, 2, 5, 8, 16, 17, 32, 33, 64, 65, *(r for r in extra_rows if 1 <= r <= 96)}))
    offsets = (0, 1, 5, 17, 63)
    cap_rows = max(offsets) + max(gate_rows) + 16
    gen = torch.Generator(device="cuda").manual_seed(seed + 7919)
    valid_pool = torch.randn((max(gate_rows), k), generator=gen, device="cuda", dtype=torch.bfloat16)
    row_zero_in = valid_pool[0].clone()
    x_buf = torch.empty((cap_rows, k), device="cuda", dtype=torch.bfloat16)
    out_buf = torch.empty((cap_rows, n), device="cuda", dtype=out_dtype)

    # Capture canonical single-row output with NaN-poisoned surroundings.
    x_buf.fill_(float("nan"))
    x_buf[0].copy_(row_zero_in)
    out_buf.fill_(float("nan"))
    prog.launch(x_buf, w, out_buf, scalars=(1,))
    torch.cuda.synchronize()
    row0_want = out_buf[0].clone()
    if not bool(torch.isfinite(row0_want.float()).all().item()) or not bool((row0_want.float().abs().max() > 0).item()):
        raise AssertionError(f"row0 canonical output invalid for {n}x{k}")
    if not bool(torch.isnan(out_buf[1:]).all().item()):
        raise AssertionError(f"row0 canonical launch wrote past row 1 for {n}x{k}")

    # Prefix invariance: full max-M launch vs every smaller M on distinct rows.
    max_m = max(gate_rows)
    x_buf.fill_(float("nan"))
    x_buf[:max_m].copy_(valid_pool[:max_m])
    out_buf.fill_(float("nan"))
    prog.launch(x_buf, w, out_buf, scalars=(max_m,))
    torch.cuda.synchronize()
    full_want = out_buf[:max_m].clone()
    if not bool(torch.isnan(out_buf[max_m:]).all().item()):
        raise AssertionError(f"full M={max_m} launch corrupted poisoned output tail for {n}x{k}")

    for m in gate_rows:
        # Prefix check at offset 0 with NaN input tail right at row m.
        x_buf.fill_(float("nan"))
        x_buf[:m].copy_(valid_pool[:m])
        out_buf.fill_(float("nan"))
        prog.launch(x_buf, w, out_buf, scalars=(m,))
        torch.cuda.synchronize()
        if not bool(torch.equal(_bits_view(out_buf[:m]), _bits_view(full_want[:m]))):
            raise AssertionError(f"prefix row-invariance failed at M={m} vs M={max_m} for {n}x{k}")
        if not bool(torch.isnan(out_buf[m:]).all().item()):
            raise AssertionError(f"poisoned output tail corrupted at M={m} for {n}x{k}")

    # Offset + neighbor NaN-poison check across offsets and representative M.
    offset_test_rows = tuple(sorted({1, 2, 5, 16, 32, 64} & set(gate_rows)))
    for offset in offsets:
        for m in offset_test_rows:
            x_buf.fill_(float("nan"))
            x_buf[offset : offset + m].copy_(valid_pool[:m])
            x_buf[offset].copy_(row_zero_in)
            x_buf[offset + m - 1].copy_(row_zero_in)
            out_buf.fill_(float("nan"))
            prog.launch(x_buf[offset:], w, out_buf, scalars=(m,))
            torch.cuda.synchronize()
            if not bool(torch.equal(_bits_view(out_buf[0]), _bits_view(row0_want))):
                raise AssertionError(f"offset={offset} M={m} row 0 mismatch for {n}x{k}")
            if not bool(torch.equal(_bits_view(out_buf[m - 1]), _bits_view(row0_want))):
                raise AssertionError(f"offset={offset} M={m} last row mismatch for {n}x{k}")
            if not bool(torch.isnan(out_buf[m:]).all().item()):
                raise AssertionError(f"offset={offset} M={m} poisoned tail corrupted for {n}x{k}")

    return {
        "row_invariant": True,
        "gate_rows": list(gate_rows),
        "offsets_checked": list(offsets),
        "poisoned_input_neighbors": True,
        "poisoned_output_tail": True,
    }


def run_cuteafd_decode_mode(args):
    require_sm120()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    flush_bytes = (
        resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cache == "cold" else 0
    )
    flush = make_l2_flush_fn(args.cache == "cold", flush_bytes)
    sources = [
        Path(__file__),
        ROOT / "b12x/gemm/bf16_gemv/_prefill.py",
        ROOT / "b12x/gemm/bf16_gemv/_skinny.py",
        ROOT / "b12x/gemm/bf16_gemv/_tuning.py",
        ROOT / "b12x/integration/cuteafd/_common.py",
        ROOT / "b12x/integration/cuteafd/pointwise.py",
        ROOT / "benchmarks/common.py",
    ]
    result: dict[str, object] = {
        "status": "running",
        "mode": "cuteafd-decode",
        "baseline_only": bool(args.baseline_only),
        "sweep": bool(args.sweep),
        "started_unix_ns": time.time_ns(),
        "command": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "candidate_spec"
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "B12X_", "CUTE_", "CUTLASS_"))
        },
        "toolchain": toolchain(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "device": {
            "logical_index": torch.cuda.current_device(),
            "uuid": str(getattr(properties, "uuid", "")),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        },
        "timing_scope": (
            "CUDA events around single-launch captured AotProgram CUDA graph replay only; "
            "L2 eviction buffer sweep executed before every timed replay outside timing when cache=cold; "
            "paired alternating arm blocks; bitwise torch.equal and NaN-poisoned row-invariance verified."
        ),
        "ratio_definition": (
            "baseline_over_variant_ratio = baseline median_us / variant median_us; "
            ">1.0 means variant is faster than c324b67f baseline."
        ),
        "l2_flush_bytes": flush_bytes,
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "shape_gates": [],
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for dtype_name in args.output_dtypes:
            out_dtype = DTYPES[dtype_name]
            for shape_idx, (n, k) in enumerate(args.parsed_shapes):
                shape_key = f"{n}x{k}"
                base_tiles = C324B67F_DECODE_TMA_TILES.get((n, k), ())
                base_spec = {"tiles_by_rows": base_tiles, "base": {}}

                arms_to_build: list[tuple[str, dict[str, object] | None]] = [("baseline", base_spec)]
                if not args.baseline_only:
                    if args.sweep:
                        if shape_key not in args.candidate_spec:
                            raise ValueError(f"--candidate-json missing shape {shape_key}")
                        for entry in args.candidate_spec[shape_key]:
                            arms_to_build.append((str(entry["name"]), {
                                "tiles_by_rows": entry.get("tiles_by_rows", ()),
                                "base": entry.get("base", {}),
                            }))
                    elif args.candidate_spec is not None:
                        if shape_key not in args.candidate_spec:
                            raise ValueError(f"--candidate-json missing shape {shape_key}")
                        raw_entry = args.candidate_spec[shape_key]
                        if isinstance(raw_entry, dict):
                            cand_spec = {
                                "tiles_by_rows": raw_entry.get("tiles_by_rows", ()),
                                "base": raw_entry.get("base", {}),
                            }
                        else:
                            cand_spec = {"tiles_by_rows": raw_entry, "base": {}}
                        arms_to_build.append(("variant", cand_spec))
                    else:
                        arms_to_build.append(("variant", None))

                programs: dict[str, object] = {}
                arm_descs: dict[str, dict[str, object]] = {}
                switch_limits: set[int] = set()
                for arm_name, spec in arms_to_build:
                    prog, desc, tiles_tuple = _build_cuteafd_arm_program(arm_name, n, k, out_dtype, spec)
                    programs[arm_name] = prog
                    arm_descs[arm_name] = desc
                    for tile_entry in tiles_tuple:
                        switch_limits.add(int(tile_entry[0]))

                rows_for_shape = list(args.rows)
                if args.auto_switch_neighbors and switch_limits:
                    expanded = set(rows_for_shape)
                    for lim in switch_limits:
                        for neighbor in (lim - 1, lim, lim + 1):
                            if neighbor >= 1:
                                expanded.add(neighbor)
                    rows_for_shape = sorted(expanded)

                w_seed = args.seed + n * 1009 + k
                w_gen = torch.Generator(device="cuda").manual_seed(w_seed)
                weight = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
                weight.normal_(std=1.0 / math.sqrt(k), generator=w_gen)

                invariance_gates: dict[str, object] = {}
                for arm_name, prog in programs.items():
                    invariance_gates[arm_name] = _verify_cuteafd_row_invariance(
                        prog,
                        w=weight,
                        out_dtype=out_dtype,
                        seed=w_seed,
                        extra_rows=tuple(rows_for_shape),
                    )
                result["shape_gates"].append({
                    "shape": shape_key,
                    "N": n,
                    "K": k,
                    "output_dtype": dtype_name,
                    "arms": arm_descs,
                    "invariance": invariance_gates,
                })
                save()

                max_m = max(rows_for_shape)
                x_seed = args.seed + 999983 + n * 1009 + k
                x_gen = torch.Generator(device="cuda").manual_seed(x_seed)
                x_all = torch.empty((max_m + 16, k), device="cuda", dtype=torch.bfloat16)
                x_all.fill_(float("nan"))
                x_all[:max_m].normal_(generator=x_gen)
                expected_all = oracle(x_all[:max_m], weight)

                arm_names = [a for a, _ in arms_to_build]
                out_buffers = {
                    arm_name: torch.empty((max_m + 16, n), device="cuda", dtype=out_dtype)
                    for arm_name in arm_names
                }

                for row_idx, rows in enumerate(rows_for_shape):
                    x_view = x_all[:rows]
                    expected = expected_all[:rows]
                    eager_outputs: dict[str, torch.Tensor] = {}
                    correctness: dict[str, object] = {}
                    for arm_name in arm_names:
                        buf = out_buffers[arm_name]
                        buf.fill_(float("nan"))
                        for _ in range(args.warmup):
                            programs[arm_name].launch(x_view, weight, buf, scalars=(rows,))
                        torch.cuda.synchronize()
                        if not bool(torch.isnan(buf[rows:]).all().item()):
                            raise AssertionError(f"{arm_name} eager launch corrupted tail at {shape_key} M={rows}")
                        eager_outputs[arm_name] = buf[:rows].clone()
                        correctness[arm_name] = {"eager": check_output(eager_outputs[arm_name], expected, arm_name)}

                    graphs: dict[str, torch.cuda.CUDAGraph] = {}
                    samples: dict[str, list[float]] = {arm_name: [] for arm_name in arm_names}
                    blocks: list[dict[str, object]] = []
                    case_order = arm_names if (shape_idx + row_idx) % 2 == 0 else arm_names[::-1]

                    with kernel_resolution_guard("Kimi D1 BF16 projection graph capture and paired cold replay"):
                        for arm_name in case_order:
                            buf = out_buffers[arm_name]

                            def launch_once(prog=programs[arm_name], out_b=buf, r=rows):
                                for _ in range(args.launches):
                                    prog.launch(x_view, weight, out_b, scalars=(r,))

                            graphs[arm_name] = capture_cuda_graph(launch_once, warmup=0)
                            buf.fill_(float("nan"))
                            graphs[arm_name].replay()
                            torch.cuda.synchronize()
                            if not bool(torch.isnan(buf[rows:]).all().item()):
                                raise AssertionError(f"{arm_name} graph replay corrupted tail at {shape_key} M={rows}")
                            if not bool(torch.equal(_bits_view(buf[:rows]), _bits_view(eager_outputs[arm_name]))):
                                raise AssertionError(f"{arm_name} graph output != eager output at {shape_key} M={rows}")
                            correctness[arm_name]["graph_before_timing"] = check_output(buf[:rows], expected, arm_name)

                        base_bits = _bits_view(eager_outputs["baseline"])
                        bitwise_vs_baseline: dict[str, dict[str, object]] = {}
                        for arm_name in arm_names:
                            cand_bits = _bits_view(eager_outputs[arm_name])
                            eq = bool(torch.equal(cand_bits, base_bits))
                            mismatches = 0 if eq else int((cand_bits != base_bits).sum().item())
                            max_ulp = 0 if eq else int((cand_bits.int() - base_bits.int()).abs().max().item())
                            bitwise_vs_baseline[arm_name] = {
                                "torch_equal": eq,
                                "mismatched_elements": mismatches,
                                "total_elements": int(base_bits.numel()),
                                "max_ulp_vs_baseline": max_ulp,
                            }
                            if not eq:
                                raise AssertionError(
                                    f"HARD GATE FAILED: {arm_name} is NOT bitwise identical to baseline "
                                    f"at {shape_key} M={rows} ({mismatches} mismatches, max_ulp={max_ulp})"
                                )

                        gpu_before = nvidia_smi_gpu_mode_snapshot()
                        for repeat in range(args.repeats):
                            order = case_order if repeat % 2 == 0 else case_order[::-1]
                            block: dict[str, object] = {"repeat": repeat, "order": list(order), "replay_us": {}}
                            for arm_name in order:
                                raw = bench_cuda_graph(
                                    graphs[arm_name], replays=args.replays, l2_flush=flush
                                )["replay_us"]
                                if any(not math.isfinite(v) or v <= 0 for v in raw):
                                    raise RuntimeError(f"Invalid CUDA event samples for {arm_name}: {raw}")
                                block["replay_us"][arm_name] = raw
                                samples[arm_name].extend(v / args.launches for v in raw)
                            if "variant" in block["replay_us"]:
                                block["baseline_over_variant_ratio"] = (
                                    statistics.median(block["replay_us"]["baseline"])
                                    / statistics.median(block["replay_us"]["variant"])
                                )
                            blocks.append(block)
                        gpu_after = nvidia_smi_gpu_mode_snapshot()

                        for arm_name in arm_names:
                            buf = out_buffers[arm_name]
                            buf.fill_(float("nan"))
                            graphs[arm_name].replay()
                            torch.cuda.synchronize()
                            if not bool(torch.isnan(buf[rows:]).all().item()):
                                raise AssertionError(f"{arm_name} post-timing replay corrupted tail at {shape_key} M={rows}")
                            if not bool(torch.equal(_bits_view(buf[:rows]), base_bits)):
                                raise AssertionError(f"{arm_name} post-timing replay lost bitwise identity at {shape_key} M={rows}")
                            correctness[arm_name]["graph_after_timing"] = check_output(buf[:rows], expected, arm_name)

                    for g in graphs.values():
                        g.reset()

                    stats_by_arm = {arm_name: _sample_stats(vals) for arm_name, vals in samples.items()}
                    compulsory_bytes = int(2 * n * k + rows * (2 * k + out_dtype.itemsize * n))
                    gbps_by_arm = {
                        arm_name: float(compulsory_bytes / (st["median_us"] * 1e3))
                        for arm_name, st in stats_by_arm.items()
                    }
                    ratios_vs_baseline = {
                        arm_name: float(stats_by_arm["baseline"]["median_us"] / st["median_us"])
                        for arm_name, st in stats_by_arm.items()
                    }
                    rec: dict[str, object] = {
                        "shape": shape_key,
                        "M": rows,
                        "N": n,
                        "K": k,
                        "output_dtype": dtype_name,
                        "compulsory_bytes": compulsory_bytes,
                        "floor_1792_us": float(compulsory_bytes / 1.792e6),
                        "arms": arm_descs,
                        "stats_us": stats_by_arm,
                        "median_us": {a: st["median_us"] for a, st in stats_by_arm.items()},
                        "iqr_us": {a: st["iqr_us"] for a, st in stats_by_arm.items()},
                        "effective_gbps": gbps_by_arm,
                        "baseline_over_arm_ratio": ratios_vs_baseline,
                        "bitwise_vs_baseline": bitwise_vs_baseline,
                        "correctness": correctness,
                        "samples_us": samples,
                        "blocks": blocks,
                        "gpu_before": gpu_before,
                        "gpu_after": gpu_after,
                    }
                    if "variant" in stats_by_arm:
                        rec["baseline_over_variant_ratio"] = ratios_vs_baseline["variant"]
                        rec["variant_winning_blocks"] = sum(
                            b.get("baseline_over_variant_ratio", 0.0) > 1.0 for b in blocks
                        )
                    result["records"].append(rec)
                    save()
                    summary_line = {
                        "shape": shape_key,
                        "M": rows,
                        "median_us": {a: round(v, 2) for a, v in rec["median_us"].items()},
                        "iqr_us": {a: round(v, 2) for a, v in rec["iqr_us"].items()},
                        "gbps": {a: round(v, 1) for a, v in gbps_by_arm.items()},
                        "ratio": {a: round(v, 4) for a, v in ratios_vs_baseline.items() if a != "baseline"},
                        "bitwise": all(v["torch_equal"] for v in bitwise_vs_baseline.values()),
                    }
                    print(json.dumps(summary_line), flush=True)

                del programs, out_buffers, x_all, expected_all, weight
                torch.cuda.empty_cache()

        # Compute estimated C1 decode serving ms/token across shapes that measured M=2,4,5,6.
        by_shape_row: dict[tuple[int, int, int], dict[str, object]] = {}
        for rec in result["records"]:
            if rec["output_dtype"] == "bf16":
                by_shape_row[(int(rec["N"]), int(rec["K"]), int(rec["M"]))] = rec
        c1_estimate: dict[str, object] = {"per_shape": {}, "totals_ms_per_token": {}}
        all_present = True
        for (n, k), calls in D1_C1_CALLS_PER_TOKEN.items():
            if all((n, k, m) in by_shape_row for m in (2, 4, 5, 6)):
                shape_ms: dict[str, float] = {}
                for arm_name in by_shape_row[(n, k, 5)]["median_us"]:
                    ms_tok = (
                        calls["m2"] * by_shape_row[(n, k, 2)]["median_us"][arm_name]
                        + calls["m4"] * by_shape_row[(n, k, 4)]["median_us"][arm_name]
                        + calls["m5"] * by_shape_row[(n, k, 5)]["median_us"][arm_name]
                        + calls["m6"] * by_shape_row[(n, k, 6)]["median_us"][arm_name]
                    ) / 1000.0
                    shape_ms[arm_name] = float(ms_tok)
                    c1_estimate["totals_ms_per_token"][arm_name] = (
                        c1_estimate["totals_ms_per_token"].get(arm_name, 0.0) + float(ms_tok)
                    )
                if "variant" in shape_ms and "baseline" in shape_ms:
                    shape_ms["saved_ms_per_token"] = float(shape_ms["baseline"] - shape_ms["variant"])
                c1_estimate["per_shape"][f"{n}x{k}"] = shape_ms
            else:
                all_present = False
        if "variant" in c1_estimate["totals_ms_per_token"] and "baseline" in c1_estimate["totals_ms_per_token"]:
            c1_estimate["saved_ms_per_token"] = float(
                c1_estimate["totals_ms_per_token"]["baseline"]
                - c1_estimate["totals_ms_per_token"]["variant"]
            )
        c1_estimate["complete_all_7_shapes"] = all_present
        result["c1_decode_serving_estimate"] = c1_estimate
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_unix_ns"] = time.time_ns()
        result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
        save()


def main():
    args = parse_args()
    if args.preflight:
        print(f"benchmark_bf16_projection preflight ok (mode={args.mode})")
        return
    if args.mode == "cuteafd-decode":
        run_cuteafd_decode_mode(args)
        return
    require_sm120()
    # These options apply only to the untimed FP32 numerical oracle.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    flush_bytes = (
        resolve_l2_flush_bytes(args.l2_flush_bytes) if args.cache == "cold" else 0
    )
    flush = make_l2_flush_fn(args.cache == "cold", flush_bytes)
    sources = [
        Path(__file__),
        Path(native.__file__),
        ROOT / "benchmarks/common.py",
        ROOT / "b12x/_lib/compiler.py",
        ROOT / "b12x/_lib/intrinsics.py",
        ROOT / "b12x/_lib/runtime_control.py",
        ROOT / "b12x/_lib/utils.py",
    ]
    result = {
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "command": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "B12X_", "CUTE_", "CUTLASS_"))
        },
        "toolchain": toolchain(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "device": {
            "logical_index": torch.cuda.current_device(),
            "uuid": str(getattr(properties, "uuid", "")),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        },
        "timing_scope": "CUDA events around captured native cached callable replay only; oracle, compilation, warmup, snapshots, correctness and optional L2 flush excluded. Paired alternating AB/BA blocks; one shared output; no Torch/CUBLAS measured arm.",
        "ratio_definition": "simt_over_mma_ratio = median SIMT us / median MMA us; >1 means MMA faster, <1 means MMA is slower than SIMT. No significance claim.",
        "l2_flush_bytes": flush_bytes,
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for name in args.cases:
            for rows in args.rows:
                for dtype in args.output_dtypes:
                    for layout in args.layouts:
                        record = run_case(
                            args,
                            name,
                            rows,
                            dtype,
                            layout,
                            len(result["records"]),
                            flush,
                        )
                        result["records"].append(record)
                        save()
                        print(
                            json.dumps(
                                {
                                    key: record[key]
                                    for key in (
                                        "case",
                                        "M",
                                        "N",
                                        "K",
                                        "output_dtype",
                                        "layout",
                                        "median_us_per_projection",
                                        "simt_over_mma_ratio",
                                        "mma_winning_blocks",
                                        "production_policy_regresses_vs_simt",
                                    )
                                }
                            ),
                            flush=True,
                        )
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_unix_ns"] = time.time_ns()
        result["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
        save()


if __name__ == "__main__":
    main()
