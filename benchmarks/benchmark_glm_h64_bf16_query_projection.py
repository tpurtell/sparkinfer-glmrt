#!/usr/bin/env python3
"""Benchmark one-launch GLM H64 MLA projection against both controls.

The controls are SparkInfer's existing four-H16 composition and GLMRT's
current native launch structure.  ``decode`` reproduces one 64-way cuBLAS
launch per query row plus two cudaMemcpy2D assembly nodes; ``prefill`` uses
the native transpose, one 64-way cuBLAS launch, and compose kernel.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Callable, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sparkinfer.gemm import mla_query_projection


HEADS = 64
GROUP_HEADS = 16
NOPE_DIM = 192
KV_B_ROWS = 448
LATENT_DIM = 512
ROPE_DIM = 64
QUERY_DIM = LATENT_DIM + ROPE_DIM
BF16_BYTES = 2


class DeviceBuffer(ctypes.Structure):
    _fields_ = (
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    )


@dataclass(frozen=True)
class NativeApi:
    library: ctypes.CDLL
    matmul: object
    copy_2d: object
    transpose: object
    compose: object


def _int_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(part) for part in value.split(",") if part))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m-values", type=_int_csv, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--routes", choices=("decode", "prefill", "both"), default="both"
    )
    parser.add_argument("--weight-count", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.native_library.is_file():
        parser.error(f"native library does not exist: {args.native_library}")
    if any(m < 1 or m > 32 for m in args.m_values):
        parser.error("--m-values must stay in [1,32]")
    if not 1 <= args.weight_count <= 79:
        parser.error("--weight-count must stay in [1,79]")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        parser.error("warmup/iterations must be positive and repeats >= 3")
    return args


def _native(path: Path) -> NativeApi:
    library = ctypes.CDLL(str(path.resolve()))
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    matmul = library.glmrt_cuda_matmul_bf16_strided_batched_cublas_async
    matmul.argtypes = (
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        size,
        size,
        size,
        size,
        pointer,
    )
    matmul.restype = ctypes.c_int
    copy_2d = library.glmrt_copy_d2d_2d_async
    copy_2d.argtypes = (
        DeviceBuffer,
        size,
        DeviceBuffer,
        size,
        size,
        size,
        pointer,
    )
    copy_2d.restype = ctypes.c_int
    transpose = library.glmrt_cuda_transpose_rows_heads_bf16_async
    transpose.argtypes = (pointer, pointer, size, size, size, pointer)
    transpose.restype = ctypes.c_int
    compose = library.glmrt_cuda_mla_compose_absorbed_query_bf16_async
    compose.argtypes = (
        pointer,
        pointer,
        pointer,
        size,
        size,
        size,
        size,
        pointer,
    )
    compose.restype = ctypes.c_int
    return NativeApi(library, matmul, copy_2d, transpose, compose)


def _check(api: NativeApi, status: int, action: str) -> None:
    if status == 0:
        return
    message = ctypes.create_string_buffer(1024)
    api.library.glmrt_last_error(message, len(message))
    raise RuntimeError(
        f"{action} failed status={status}: {message.value.decode(errors='replace')}"
    )


def _ptr(tensor: torch.Tensor, element_offset: int = 0) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr() + element_offset * tensor.element_size())


def _buffer(
    tensor: torch.Tensor, *, byte_offset: int = 0, byte_span: int | None = None
) -> DeviceBuffer:
    total = tensor.numel() * tensor.element_size()
    remaining = total - byte_offset
    span = remaining if byte_span is None else byte_span
    if byte_offset < 0 or span < 0 or byte_offset + span > total:
        raise ValueError(
            f"invalid buffer view offset={byte_offset} span={span} total={total}"
        )
    return DeviceBuffer(
        tensor.data_ptr() + byte_offset,
        span,
        tensor.device.index or 0,
        0,
    )


def _stream() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _native_decode(
    api: NativeApi,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    weight: torch.Tensor,
    projected: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    m = int(q_nope.shape[0])
    q_row = HEADS * NOPE_DIM
    projected_row = HEADS * LATENT_DIM
    weight_head_stride = KV_B_ROWS * LATENT_DIM
    output_pitch = QUERY_DIM * BF16_BYTES
    latent_pitch = LATENT_DIM * BF16_BYTES
    rope_pitch = ROPE_DIM * BF16_BYTES
    rows = m * HEADS
    latent_span = (rows - 1) * output_pitch + latent_pitch
    rope_span = (rows - 1) * output_pitch + rope_pitch
    latent_dst = _buffer(out, byte_span=latent_span)
    latent_src = _buffer(projected)
    rope_dst = _buffer(out, byte_offset=LATENT_DIM * BF16_BYTES, byte_span=rope_span)
    rope_src = _buffer(q_pe)

    def launch() -> None:
        stream = _stream()
        for row in range(m):
            _check(
                api,
                api.matmul(
                    _ptr(q_nope, row * q_row),
                    _ptr(weight),
                    _ptr(projected, row * projected_row),
                    HEADS,
                    1,
                    NOPE_DIM,
                    LATENT_DIM,
                    NOPE_DIM,
                    weight_head_stride,
                    LATENT_DIM,
                    stream,
                ),
                f"decode matmul row {row}",
            )
        _check(
            api,
            api.copy_2d(
                latent_dst,
                output_pitch,
                latent_src,
                latent_pitch,
                latent_pitch,
                rows,
                stream,
            ),
            "decode latent assembly",
        )
        _check(
            api,
            api.copy_2d(
                rope_dst,
                output_pitch,
                rope_src,
                rope_pitch,
                rope_pitch,
                rows,
                stream,
            ),
            "decode rope assembly",
        )

    return launch


def _native_prefill(
    api: NativeApi,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    weight: torch.Tensor,
    transposed: torch.Tensor,
    projected: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    m = int(q_nope.shape[0])
    weight_head_stride = KV_B_ROWS * LATENT_DIM

    def launch() -> None:
        stream = _stream()
        _check(
            api,
            api.transpose(
                _ptr(q_nope),
                _ptr(transposed),
                m,
                HEADS,
                NOPE_DIM,
                stream,
            ),
            "prefill transpose",
        )
        _check(
            api,
            api.matmul(
                _ptr(transposed),
                _ptr(weight),
                _ptr(projected),
                HEADS,
                m,
                NOPE_DIM,
                LATENT_DIM,
                m * NOPE_DIM,
                weight_head_stride,
                m * LATENT_DIM,
                stream,
            ),
            "prefill matmul",
        )
        _check(
            api,
            api.compose(
                _ptr(projected),
                _ptr(q_pe),
                _ptr(out),
                m,
                HEADS,
                LATENT_DIM,
                ROPE_DIM,
                stream,
            ),
            "prefill compose",
        )

    return launch


def _four_h16(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    q_head_major = q_nope.permute(1, 0, 2)
    k_weight = weight[:, :NOPE_DIM, :]
    views = tuple(
        (
            q_head_major[head : head + GROUP_HEADS],
            k_weight[head : head + GROUP_HEADS],
            q_pe[:, head : head + GROUP_HEADS, :],
            out[:, head : head + GROUP_HEADS, :],
        )
        for head in range(0, HEADS, GROUP_HEADS)
    )

    def launch() -> None:
        for q_group, w_group, pe_group, out_group in views:
            mla_query_projection.run(q_group, w_group, pe_group, out_group)

    return launch


def _one_h64(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> Callable[[], None]:
    q_head_major = q_nope.permute(1, 0, 2)
    k_weight = weight[:, :NOPE_DIM, :]

    def launch() -> None:
        mla_query_projection.run_glm_h64_bf16(q_head_major, k_weight, q_pe, out)

    return launch


def _capture(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    operations: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, dict[str, object]]:
    labels = tuple(graphs)
    for step in range(warmup):
        for offset in range(len(labels)):
            graphs[labels[(step + offset) % len(labels)]].replay()
    torch.cuda.synchronize()
    samples = {label: [] for label in labels}
    for repeat in range(repeats):
        for offset in range(len(labels)):
            label = labels[(repeat + offset) % len(labels)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[label].replay()
            end.record()
            end.synchronize()
            samples[label].append(
                start.elapsed_time(end) * 1000.0 / iterations / operations
            )
    return {
        label: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "samples_us": values,
        }
        for label, values in samples.items()
    }


def _case(
    api: NativeApi,
    *,
    route: Literal["decode", "prefill"],
    m: int,
    weights: torch.Tensor,
    generator: torch.Generator,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    device = weights.device
    q_nope = torch.randn(
        (m, HEADS, NOPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q_pe = torch.randn(
        (m, HEADS, ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    outputs = {
        name: torch.empty((m, HEADS, QUERY_DIM), dtype=torch.bfloat16, device=device)
        for name in ("native", "four_h16", "one_h64")
    }
    native_projected = torch.empty(
        (m, HEADS, LATENT_DIM), dtype=torch.bfloat16, device=device
    )
    transposed = torch.empty((HEADS, m, NOPE_DIM), dtype=torch.bfloat16, device=device)
    native_projected_head_major = torch.empty(
        (HEADS, m, LATENT_DIM), dtype=torch.bfloat16, device=device
    )
    launches: dict[str, list[Callable[[], None]]] = {
        "native": [],
        "four_h16": [],
        "one_h64": [],
    }
    correctness = []
    for index, weight in enumerate(weights):
        native = (
            _native_decode(
                api,
                q_nope,
                q_pe,
                weight,
                native_projected,
                outputs["native"],
            )
            if route == "decode"
            else _native_prefill(
                api,
                q_nope,
                q_pe,
                weight,
                transposed,
                native_projected_head_major,
                outputs["native"],
            )
        )
        four = _four_h16(q_nope, q_pe, weight, outputs["four_h16"])
        one = _one_h64(q_nope, q_pe, weight, outputs["one_h64"])
        native()
        four()
        one()
        torch.cuda.synchronize()
        reference = outputs["native"].view(torch.uint8)
        correctness.append(
            {
                "weight_index": index,
                "four_h16_bitwise_native": bool(
                    torch.equal(reference, outputs["four_h16"].view(torch.uint8))
                ),
                "one_h64_bitwise_native": bool(
                    torch.equal(reference, outputs["one_h64"].view(torch.uint8))
                ),
                "one_h64_rope_bitwise_source": bool(
                    torch.equal(
                        outputs["one_h64"][..., LATENT_DIM:].view(torch.uint8),
                        q_pe.view(torch.uint8),
                    )
                ),
                "one_h64_finite": bool(
                    torch.isfinite(outputs["one_h64"].float()).all().item()
                ),
                "one_h64_nonzero": bool(torch.count_nonzero(outputs["one_h64"]).item()),
            }
        )
        launches["native"].append(native)
        launches["four_h16"].append(four)
        launches["one_h64"].append(one)

    def cycle(items: list[Callable[[], None]]) -> Callable[[], None]:
        def launch() -> None:
            for item in items:
                item()

        return launch

    graphs = {name: _capture(cycle(items)) for name, items in launches.items()}
    allocation_before = torch.cuda.memory_allocated(device)
    for graph in graphs.values():
        graph.replay()
        graph.replay()
    torch.cuda.synchronize()
    allocation_after = torch.cuda.memory_allocated(device)
    timing = _measure(
        graphs,
        operations=len(weights),
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    native_us = float(timing["native"]["median_us"])
    four_us = float(timing["four_h16"]["median_us"])
    one_us = float(timing["one_h64"]["median_us"])
    return {
        "route": route,
        "m": m,
        "weight_count": len(weights),
        "correctness": {
            "passed": all(
                item["four_h16_bitwise_native"]
                and item["one_h64_bitwise_native"]
                and item["one_h64_rope_bitwise_source"]
                and item["one_h64_finite"]
                and item["one_h64_nonzero"]
                for item in correctness
            ),
            "per_weight": correctness,
        },
        "graph": {
            "captured": True,
            "torch_allocation_delta_bytes_six_replays": allocation_after
            - allocation_before,
        },
        "timing": timing,
        "four_h16_over_native": four_us / native_us,
        "one_h64_over_native": one_us / native_us,
        "one_h64_over_four_h16": one_us / four_us,
        "ratio_direction": "candidate/control; lower is better",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _gpu_snapshot() -> list[str]:
    fields = (
        "index,name,uuid,pstate,clocks.current.sm,clocks.current.memory,"
        "power.limit,clocks_throttle_reasons.active"
    )
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return output.strip().splitlines()


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("SM120/SM121 is required")
    api = _native(args.native_library)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    weights = (
        torch.randn(
            (args.weight_count, HEADS, KV_B_ROWS, LATENT_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.02
    )
    k_weight = weights[0, :, :NOPE_DIM, :]
    mla_query_projection.prewarm_glm_h64_bf16(k_weight, args.m_values)
    mla_query_projection.prewarm(
        k_weight[:GROUP_HEADS],
        args.m_values,
        output_dtype=torch.bfloat16,
    )
    routes: tuple[Literal["decode", "prefill"], ...] = (
        ("decode", "prefill") if args.routes == "both" else (args.routes,)
    )
    cases = []
    for route in routes:
        for m in args.m_values:
            if route == "decode" and m > 16:
                continue
            cases.append(
                _case(
                    api,
                    route=route,
                    m=m,
                    weights=weights,
                    generator=generator,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
            )
    repo = Path(__file__).resolve().parents[1]
    native = args.native_library.resolve()
    result = {
        "schema": "sparkinfer-glm-h64-bf16-query-projection-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "repo": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "native_library": str(native),
        "native_library_sha256": _sha256(native),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_snapshot": _gpu_snapshot(),
        "weight_count": args.weight_count,
        "resident_weight_bytes": weights.numel() * weights.element_size(),
        "single_resident_kv_b_allocation": True,
        "cases": cases,
    }
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
