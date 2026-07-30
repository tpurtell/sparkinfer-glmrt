#!/usr/bin/env python3
"""Benchmark the exact Spark W4A4-FC1/W4A16-FC2 composed artifact.

The candidate arm invokes ``W4A4FC1W4A16FC2SparkAOTArtifact.launch``—the same
three compiled objects and launch composition exported to C. The comparison
arm is the retained packed-W13/packed-W2 fused W4A16 kernel. Inputs are
prequantized NVFP4 in the desired source ``[up; gate]`` W13 order; the W4A16
arm consumes the exact BF16 decode of that payload.

Default output includes correctness gates, hot and rotating-weight CUDA-graph
samples, candidate phase timings, replay allocation stability, exact object
hashes, and GPU clock/throttle snapshots. ``--candidate-only`` is a compile and
graph smoke mode and is explicitly not acceptance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# A disk-cache hit is an external-binary launch object without the IR required
# by ``export_to_c``. Acceptance needs hashes for the exact objects being
# timed, so compile fresh exportable objects and retain them in memory.
os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "0"
os.environ["SPARKINFER_COMPILE_MEMORY_CACHE"] = "1"

import cutlass
import torch
import torch.nn.functional as F

from sparkinfer._lib.intrinsics import (
    as_grouped_scale_view,
    swizzle_block_scale,
)
from sparkinfer._lib.utils import make_ptr
from sparkinfer.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    select_route_block_size_m,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    _cutlass_element_dtype,
    compile_w4a16_fused_moe,
    cuda,
    cute,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    prepare_w4a16_modelopt_nvfp4_weights,
)
from sparkinfer.moe.fused_moe.aot import (
    W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS,
    W4A4FC1W4A16FC2SparkSpec,
    bind_w4a4_fc1_w4a16_fc2_expert,
    bind_w4a4_fc1_w4a16_fc2_spark_workspace,
    compile_w4a4_fc1_w4a16_fc2_spark_aot,
    initialize_w4a4_fc1_w4a16_fc2_spark_routes,
    prepare_w4a4_fc1_w4a16_fc2_weights,
)


HIDDEN = 6_144
INTERMEDIATE = 512
FC1_COLS = 2 * INTERMEDIATE
ACCEPTED_PERFORMANCE_PSTATES = ("P0", "P1")


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def _run(
    command: tuple[str, ...],
    *,
    cwd: pathlib.Path | None = None,
) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_provenance(root: pathlib.Path) -> dict[str, object]:
    source_paths = (
        pathlib.Path("pyproject.toml"),
        pathlib.Path("benchmarks/benchmark_w4a4_fc1_w4a16_fc2.py"),
        pathlib.Path("sparkinfer/_lib/dense_gemm.py"),
        pathlib.Path("sparkinfer/moe/_shared/kernels/w4a16/kernel.py"),
        pathlib.Path("sparkinfer/moe/_shared/kernels/w4a16/prepare.py"),
        pathlib.Path("sparkinfer/moe/fused_moe/aot.py"),
        pathlib.Path("sparkinfer/moe/_shared/kernels/w4a4_w4a16/composed.py"),
        pathlib.Path("sparkinfer/moe/_shared/kernels/w4a4_w4a16/prepare.py"),
    )
    digest = hashlib.sha256()
    for relative in source_paths:
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    dirty_paths = _run(("git", "status", "--porcelain"), cwd=root).splitlines()
    return {
        "commit": _run(("git", "rev-parse", "HEAD"), cwd=root),
        "worktree_dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "kernel_source_sha256": digest.hexdigest(),
        "hashed_paths": [str(path) for path in source_paths],
    }


def _nvidia_smi_snapshot(uuid: str) -> dict[str, object]:
    fields = (
        "uuid",
        "compute_mode",
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks_throttle_reasons.active",
        "power.draw",
        "power.limit",
    )
    command = (
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
        "-i",
        f"GPU-{uuid}",
    )
    try:
        values = [value.strip() for value in _run(command).split(",")]
        return dict(zip(fields, values, strict=True))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        return {
            "error": str(error),
            "command": list(command),
        }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tool_record(name: str) -> dict[str, object]:
    executable = shutil.which(name)
    if executable is None:
        return {"error": f"{name} was not found on PATH"}
    path = pathlib.Path(executable).resolve()
    try:
        version_output = _run((str(path), "--version"))
    except subprocess.CalledProcessError as error:
        version_output = f"exit={error.returncode}: {error.stderr.strip()}"
    return {
        "executable": str(path),
        "sha256": _sha256(path),
        "version_output": version_output,
    }


def _file_record(path: pathlib.Path) -> dict[str, object]:
    record: dict[str, object] = {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix != ".o":
        return record
    printable = _run(("strings", "-a", str(path)))
    versions = sorted(
        set(
            re.findall(
                r"Cuda compilation tools, release [^\r\n]+",
                printable,
            )
        )
    )
    flags = sorted(
        {
            line.strip()
            for line in printable.splitlines()
            if re.match(r"^-O\s+\d+\s+-arch\s+sm_\d+a(?:\s|$)", line)
        }
    )
    if not versions or not flags:
        raise RuntimeError(f"exported object {path.name} omitted PTXAS version/flags")
    record["source_ptxas_versions"] = versions
    record["source_ptxas_flags"] = flags
    return record


def _artifact_hashes(
    artifact,
    production,
    *,
    rows: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"sparkinfer-w4a4-w4a16-m{rows}.") as temp:
        output = pathlib.Path(temp)
        candidate_base = f"candidate_m{rows}"
        artifact.export_to_c(
            output,
            file_name=candidate_base,
            symbol_base=f"benchmark_candidate_m{rows}",
        )
        paths = sorted(output.glob(f"{candidate_base}*"))
        identity = {
            "fc1_launch_is_exported_object": (
                getattr(
                    artifact._fc1_launch,
                    "_sparkinfer_compiled_kernel",
                    None,
                )
                is artifact.fc1_compiled
            ),
            "activation_launch_is_exported_object": (
                artifact._activation_result.compiled is artifact.activation_compiled
            ),
            "fc2_launch_is_exported_object": (
                artifact._fc2_result.compiled is artifact.fc2_compiled
            ),
        }
        result: dict[str, object] = {
            "candidate": {path.name: _file_record(path) for path in paths},
            "candidate_identity": identity,
        }
        if not all(identity.values()):
            raise RuntimeError(f"candidate launch/export identity mismatch: {identity}")
        if production is not None:
            production.compiled.export_to_c(
                str(output),
                f"production_w4a16_m{rows}",
                f"benchmark_production_w4a16_m{rows}",
            )
            production_paths = sorted(output.glob(f"production_w4a16_m{rows}*"))
            result["production_w4a16"] = {
                path.name: _file_record(path) for path in production_paths
            }
        return result


def _comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int | bool]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    reference_norm = torch.linalg.vector_norm(reference_f32)
    candidate_norm = torch.linalg.vector_norm(candidate_f32)
    difference_norm = torch.linalg.vector_norm(difference)
    norm_product = reference_norm * candidate_norm
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "different_elements": int(torch.count_nonzero(reference != candidate).item()),
        "finite": bool(torch.isfinite(candidate_f32).all().item()),
        "nonzero": int(torch.count_nonzero(candidate).item()),
        "max_abs_error": float(difference.abs().amax().item()),
        "relative_l2_error": float(
            (difference_norm / reference_norm).item()
            if float(reference_norm) != 0.0
            else difference_norm.item()
        ),
        "cosine_similarity": float(
            (torch.sum(reference_f32 * candidate_f32) / norm_product).item()
            if float(norm_product) != 0.0
            else 1.0
        ),
    }


def _unpack_unit_scale_fp4(packed: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=torch.bfloat16,
        device=packed.device,
    )
    values = packed[..., 0]
    return torch.stack(
        (table[(values & 0x0F).long()], table[(values >> 4).long()]),
        dim=-1,
    ).reshape(values.shape[0], -1)


def _capture_graph(
    operation: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    eager_warmup: int = 2,
) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(stream):
        for _ in range(eager_warmup):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation()
    graph.replay()
    stream.synchronize()
    return graph


def _measure_batch(
    graphs: list[torch.cuda.CUDAGraph],
    *,
    stream: torch.cuda.Stream,
    iterations: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for index in range(iterations):
            graphs[index % len(graphs)].replay()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _measure_pair(
    candidate: list[torch.cuda.CUDAGraph],
    production: list[torch.cuda.CUDAGraph] | None,
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[list[float], list[float] | None]:
    with torch.cuda.stream(stream):
        for index in range(warmup):
            candidate[index % len(candidate)].replay()
            if production is not None:
                production[index % len(production)].replay()
    stream.synchronize()
    candidate_samples: list[float] = []
    production_samples: list[float] | None = [] if production is not None else None
    for repeat in range(repeats):
        if production is None or repeat % 2 == 0:
            candidate_samples.append(
                _measure_batch(
                    candidate,
                    stream=stream,
                    iterations=iterations,
                )
            )
            if production_samples is not None:
                production_samples.append(
                    _measure_batch(
                        production,
                        stream=stream,
                        iterations=iterations,
                    )
                )
        else:
            assert production_samples is not None
            production_samples.append(
                _measure_batch(
                    production,
                    stream=stream,
                    iterations=iterations,
                )
            )
            candidate_samples.append(
                _measure_batch(
                    candidate,
                    stream=stream,
                    iterations=iterations,
                )
            )
    return candidate_samples, production_samples


def _warm_graphs(
    candidate: list[torch.cuda.CUDAGraph],
    production: list[torch.cuda.CUDAGraph] | None,
    phases: dict[str, torch.cuda.CUDAGraph],
    *,
    stream: torch.cuda.Stream,
    iterations: int,
) -> None:
    with torch.cuda.stream(stream):
        for index in range(iterations):
            candidate[index % len(candidate)].replay()
            if production is not None:
                production[index % len(production)].replay()
            for graph in phases.values():
                graph.replay()
    stream.synchronize()


def _timing(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "samples_ms": samples,
    }


@dataclass(frozen=True)
class _WeightSet:
    candidate: object
    candidate_expert: object
    production: object | None


def _positive_scale(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return (
        0.0078125 + 0.0234375 * torch.rand(shape, device=device, generator=generator)
    ).to(torch.float8_e4m3fn)


def _prepare_weight_set(
    *,
    device: torch.device,
    seed: int,
    candidate_only: bool,
) -> _WeightSet:
    generator = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (1, FC1_COLS, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (1, HIDDEN, INTERMEDIATE // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scale = swizzle_block_scale(
        _positive_scale(
            (1, FC1_COLS, HIDDEN // 16),
            device=device,
            generator=generator,
        )
    )
    w2_scale = swizzle_block_scale(
        _positive_scale(
            (1, HIDDEN, INTERMEDIATE // 16),
            device=device,
            generator=generator,
        )
    )
    w13_alpha = torch.tensor((0.375,), dtype=torch.float32, device=device)
    w2_alpha = torch.tensor((0.625,), dtype=torch.float32, device=device)

    production = None
    if not candidate_only:
        # Oracle allocations are intentionally separate and benchmark-only.
        # They never become part of candidate residency accounting.
        production = prepare_w4a16_modelopt_nvfp4_weights(
            w13.clone(),
            w13_scale.clone(),
            w13_alpha.clone(),
            w2.clone(),
            w2_scale.clone(),
            w2_alpha.clone(),
            activation="silu",
            params_dtype=torch.bfloat16,
            w13_layout="w13",
            reuse_input_storage=True,
        )
    candidate = prepare_w4a4_fc1_w4a16_fc2_weights(
        w13,
        w13_scale,
        w13_alpha,
        w2,
        w2_scale,
        w2_alpha,
        activation="silu",
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    if candidate.has_packed_w13 or candidate.has_source_w2:
        raise RuntimeError("candidate violated single-residency weight policy")
    return _WeightSet(
        candidate=candidate,
        candidate_expert=bind_w4a4_fc1_w4a16_fc2_expert(candidate, 0),
        production=production,
    )


@dataclass
class _RowState:
    candidate_graphs: list[torch.cuda.CUDAGraph]
    production_graphs: list[torch.cuda.CUDAGraph] | None
    phase_graphs: dict[str, torch.cuda.CUDAGraph]
    correctness: dict[str, dict[str, float | int | bool]]
    artifact_hashes: dict[str, object]
    config: dict[str, object]


def _build_row_state(
    *,
    rows: int,
    weights: list[_WeightSet],
    device: torch.device,
    stream: torch.cuda.Stream,
    grid_x: int,
    planning_sms: int,
    seed: int,
) -> _RowState:
    capability = torch.cuda.get_device_capability(device)
    target_arch = f"sm_{capability[0]}{capability[1]}"
    spec = W4A4FC1W4A16FC2SparkSpec(
        m=rows,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        w13_layout="w13",
        planning_sms=planning_sms,
        tuned_grid_x=grid_x,
        target_arch=target_arch,
    )
    artifact = compile_w4a4_fc1_w4a16_fc2_spark_aot(spec, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + rows)
    input_packed = torch.randint(
        0,
        256,
        (rows, HIDDEN // 2, 1),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    input_scale_storage = swizzle_block_scale(
        torch.ones(
            (rows, HIDDEN // 16),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
    ).unsqueeze(0)
    input_scale = as_grouped_scale_view(
        input_scale_storage,
        rows,
        HIDDEN,
    )
    input_bf16 = _unpack_unit_scale_fp4(input_packed).contiguous()
    arena = torch.empty(
        spec.workspace_layout().size_bytes + spec.workspace_layout().alignment,
        dtype=torch.uint8,
        device=device,
    )
    aligned_offset = (-int(arena.data_ptr())) % spec.workspace_layout().alignment
    aligned_arena = arena.narrow(
        0,
        aligned_offset,
        spec.workspace_layout().size_bytes,
    )
    workspace = bind_w4a4_fc1_w4a16_fc2_spark_workspace(
        aligned_arena,
        spec,
    )
    initialize_w4a4_fc1_w4a16_fc2_spark_routes(
        workspace,
        spec,
        active_rows=rows,
    )
    candidate_output = torch.empty(
        (rows, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )

    def launch_candidate(weight: _WeightSet) -> None:
        artifact.launch(
            input_packed=input_packed,
            input_scale=input_scale,
            weights=weight.candidate_expert,
            workspace=workspace,
            output=candidate_output,
            active_rows=rows,
            stream=stream,
        )

    production_compiled = None
    production_output = None
    launch_production: Callable[[_WeightSet], None] | None = None
    if weights[0].production is not None:
        properties = torch.cuda.get_device_properties(device)
        block_size = select_route_block_size_m(rows, 1, 1)
        route_slots = max_packed_route_slots(rows, block_size, 1)
        route_blocks = (route_slots + block_size - 1) // block_size
        direct_topk = rows <= 8
        production_compiled = compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=1,
            top_k=1,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=rows if direct_topk else route_blocks,
            element_dtype="bf16",
            fast_math=True,
            sms=planning_sms,
            max_shared_mem=int(properties.shared_memory_per_block_optin),
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=direct_topk,
            tc_decode_fused_sum=direct_topk,
        )
        packed_routes = torch.full(
            (route_slots,),
            rows,
            dtype=torch.int32,
            device=device,
        )
        packed_routes[:rows].copy_(torch.arange(rows, dtype=torch.int32, device=device))
        block_experts = torch.zeros(
            route_blocks,
            dtype=torch.int32,
            device=device,
        )
        packed_route_count = torch.tensor(
            (_align_up(rows, block_size),),
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.ones(
            rows,
            dtype=torch.float32,
            device=device,
        )
        route_slots_for_scratch = rows * block_size if direct_topk else route_slots
        production_fc1 = torch.empty(
            rows * max(FC1_COLS, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        production_activated = torch.empty(
            (rows, INTERMEDIATE),
            dtype=torch.bfloat16,
            device=device,
        )
        production_output = torch.empty(
            (rows, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        production_fc1_scratch = torch.empty(
            packed_gemm_scratch_elements(
                size_n=FC1_COLS,
                route_slots=route_slots_for_scratch,
                moe_block_size=block_size,
                sms=planning_sms,
            ),
            dtype=torch.float32,
            device=device,
        )
        production_fc2_scratch = torch.empty(
            packed_gemm_scratch_elements(
                size_n=HIDDEN,
                route_slots=route_slots_for_scratch,
                moe_block_size=block_size,
                sms=planning_sms,
            ),
            dtype=torch.float32,
            device=device,
        )
        production_workspace = torch.zeros(
            planning_sms * 4 + 2,
            dtype=torch.int32,
            device=device,
        )
        production_routes = (
            torch.zeros(rows, dtype=torch.int32, device=device)
            if direct_topk
            else packed_routes
        )
        production_blocks = production_routes if direct_topk else block_experts
        production_count = production_routes if direct_topk else packed_route_count
        rotation_placeholder = torch.zeros(
            1,
            dtype=torch.float16,
            device=device,
        )
        stream_arg = cuda.CUstream(stream.cuda_stream)
        input_pointer = make_ptr(
            _cutlass_element_dtype("bf16"),
            input_bf16.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        topk_pointer = make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )

        def launch_production_impl(weight: _WeightSet) -> None:
            assert weight.production is not None
            prepared = weight.production
            production_workspace.zero_()
            production_compiled.compiled(
                input_pointer,
                input_pointer,
                input_pointer,
                prepared.w13.view(torch.int32).view(-1),
                prepared.w2.view(torch.int32).view(-1),
                production_fc1[: rows * FC1_COLS],
                production_activated,
                production_output,
                prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
                prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
                prepared.w13_global_scale,
                prepared.w2_global_scale,
                production_routes,
                production_blocks,
                production_count,
                prepared.w13_global_scale,
                0,
                topk_pointer,
                production_fc1_scratch,
                production_fc2_scratch,
                prepared.workspace,
                rotation_placeholder,
                rotation_placeholder,
                rotation_placeholder,
                rows,
                grid_x,
                stream_arg,
            )

        launch_production = launch_production_impl

    launch_candidate(weights[0])
    if launch_production is not None:
        launch_production(weights[0])
    stream.synchronize()

    source_fc1 = workspace.fc1_bf16[:rows, :, 0].clone()
    source_activation = workspace.activated_bf16[:rows].clone()
    source_output = candidate_output.clone()
    gate = source_fc1[:, INTERMEDIATE:]
    up = source_fc1[:, :INTERMEDIATE]
    torch_activation = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    correctness: dict[str, dict[str, float | int | bool]] = {
        "source_w13_activation_order": _comparison(
            torch_activation,
            source_activation,
        ),
        "candidate_self": _comparison(source_output, source_output),
    }
    if launch_production is not None:
        assert production_output is not None
        production_fc1_view = production_fc1[: rows * FC1_COLS].view(rows, FC1_COLS)
        source_fc1_w31 = torch.cat((gate, up), dim=1)
        correctness.update(
            {
                "fc1_source_w4a4_vs_packed_w4a16": _comparison(
                    production_fc1_view,
                    source_fc1_w31,
                ),
                "activation_source_w4a4_vs_packed_w4a16": _comparison(
                    production_activated,
                    source_activation,
                ),
                "candidate_vs_production_w4a16": _comparison(
                    production_output,
                    source_output,
                ),
            }
        )
    for name, result in correctness.items():
        if not bool(result["finite"]) or int(result["nonzero"]) == 0:
            raise RuntimeError(f"{name} failed finite/nonzero gate: {result}")
    if launch_production is not None:
        for name in (
            "fc1_source_w4a4_vs_packed_w4a16",
            "activation_source_w4a4_vs_packed_w4a16",
            "candidate_vs_production_w4a16",
        ):
            if float(correctness[name]["cosine_similarity"]) < 0.999:
                raise RuntimeError(f"{name} failed cosine gate: {correctness[name]}")

    candidate_graphs = [
        _capture_graph(
            lambda weight=weight: launch_candidate(weight),
            stream=stream,
        )
        for weight in weights
    ]
    production_graphs = (
        [
            _capture_graph(
                lambda weight=weight: launch_production(weight),
                stream=stream,
            )
            for weight in weights
        ]
        if launch_production is not None
        else None
    )
    for index, candidate_graph in enumerate(candidate_graphs):
        candidate_graph.replay()
        stream.synchronize()
        cycling_candidate = candidate_output.clone()
        if production_graphs is None:
            cycling_comparison = _comparison(
                cycling_candidate,
                cycling_candidate,
            )
        else:
            production_graphs[index].replay()
            stream.synchronize()
            assert production_output is not None
            cycling_comparison = _comparison(
                production_output,
                cycling_candidate,
            )
        if (
            not bool(cycling_comparison["finite"])
            or int(cycling_comparison["nonzero"]) == 0
            or (
                production_graphs is not None
                and float(cycling_comparison["cosine_similarity"]) < 0.999
            )
        ):
            raise RuntimeError(
                f"cycling weight set {index} failed correctness: {cycling_comparison}"
            )
        correctness[f"cycling_weight_set_{index}"] = cycling_comparison

    phase_graphs = {
        "source_w13_w4a4_fc1": _capture_graph(
            lambda: artifact.launch_fc1(
                input_packed=input_packed,
                input_scale=input_scale,
                weights=weights[0].candidate_expert,
                workspace=workspace,
                active_rows=rows,
                stream=stream,
            ),
            stream=stream,
        ),
        "source_order_bf16_silu_product": _capture_graph(
            lambda: artifact.launch_activation(
                workspace=workspace,
                active_rows=rows,
                stream=stream,
            ),
            stream=stream,
        ),
        "packed_w2_w4a16_fc2": _capture_graph(
            lambda: artifact.launch_fc2(
                weights=weights[0].candidate_expert,
                workspace=workspace,
                output=candidate_output,
                active_rows=rows,
                stream=stream,
            ),
            stream=stream,
        ),
    }

    before_replay = torch.cuda.memory_allocated(device)
    for _ in range(32):
        candidate_graphs[0].replay()
    stream.synchronize()
    after_replay = torch.cuda.memory_allocated(device)
    replay_correctness = _comparison(source_output, candidate_output)
    if (
        after_replay != before_replay
        or not bool(replay_correctness["finite"])
        or int(replay_correctness["nonzero"]) == 0
    ):
        raise RuntimeError(
            "CUDA graph replay stability gate failed: "
            f"allocation_delta={after_replay - before_replay}, "
            f"correctness={replay_correctness}"
        )
    correctness["candidate_graph_replay"] = replay_correctness

    artifact_hashes = _artifact_hashes(
        artifact,
        production_compiled,
        rows=rows,
    )
    return _RowState(
        candidate_graphs=candidate_graphs,
        production_graphs=production_graphs,
        phase_graphs=phase_graphs,
        correctness=correctness,
        artifact_hashes=artifact_hashes,
        config={
            "capacity_m": rows,
            "active_rows": rows,
            "w13_layout": "w13_up_then_gate",
            "planning_sms": planning_sms,
            "tuned_grid_x": grid_x,
            "hardware_sm_count": int(
                torch.cuda.get_device_properties(device).multi_processor_count
            ),
            "workspace_bytes": spec.workspace_layout().size_bytes,
            "fc1_tile_mn": artifact.fc1_plan,
            "fc2_tile_kn": artifact.fc2_plan,
            "materialized_intermediate_bytes": (rows * (FC1_COLS + INTERMEDIATE) * 2),
            "reorder_buffer_bytes": 0,
            "cooperative_grid_barrier": False,
            "graph_replay_allocation_delta_bytes": (after_replay - before_replay),
        },
    )


def _ratio(
    candidate: dict[str, object],
    production: dict[str, object] | None,
) -> dict[str, object] | None:
    if production is None:
        return None
    candidate_ms = float(candidate["median_ms"])
    production_ms = float(production["median_ms"])
    return {
        "candidate_over_production": candidate_ms / production_ms,
        "production_over_candidate_speedup": production_ms / candidate_ms,
        "direction": (
            "candidate_faster"
            if candidate_ms < production_ms
            else "candidate_slower_or_equal"
        ),
    }


def _clock_evidence(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    try:
        before_sm = int(str(before["clocks.current.sm"]))
        after_sm = int(str(after["clocks.current.sm"]))
        delta_pct = 100.0 * abs(before_sm - after_sm) / max(before_sm, after_sm)
    except (KeyError, TypeError, ValueError):
        delta_pct = None
    return {
        "before": before,
        "after": after,
        "sm_clock_delta_pct": delta_pct,
    }


def _acceptance_reasons(
    *,
    args: argparse.Namespace,
    source: dict[str, object],
    runtime: dict[str, object],
    device: dict[str, object],
    results: list[dict[str, object]],
) -> list[str]:
    reasons: list[str] = []
    if args.candidate_only:
        reasons.append("production W4A16 comparison was disabled")
    if bool(source["worktree_dirty"]):
        reasons.append("source worktree is dirty")
    if args.weight_sets < 8:
        reasons.append(f"weight_sets={args.weight_sets} is below 8")
    if args.repeats < 5:
        reasons.append(f"repeats={args.repeats} is below 5")
    if args.iterations < 100:
        reasons.append(f"iterations={args.iterations} is below 100")
    if args.warmup < 16:
        reasons.append(f"warmup={args.warmup} is below 16")
    if device["compute_capability"] != [12, 1]:
        reasons.append(
            "release acceptance requires physical SM121, got "
            f"{device['compute_capability']}"
        )

    required_rows = {1, 2, 4, 8, 16, 32, 64, 128, 256}
    observed_rows = {int(result["rows"]) for result in results}
    missing_rows = sorted(required_rows - observed_rows)
    if missing_rows:
        reasons.append(f"balanced capacity coverage is missing {missing_rows}")

    cutlass_packages = runtime.get("cutlass_packages")
    if not isinstance(cutlass_packages, dict) or set(cutlass_packages.values()) != {
        "4.6.1"
    }:
        reasons.append("CUTLASS DSL five-package map is not exactly 4.6.1")
    ptxas_tool = runtime.get("ptxas_tool")
    if not isinstance(ptxas_tool, dict) or "error" in ptxas_tool:
        reasons.append("PTXAS executable/version/hash evidence is incomplete")

    uuid = str(device["uuid"])
    expected_arch_flag = "-arch sm_121a"
    for result in results:
        rows = int(result["rows"])
        config = result["config"]
        if int(config["graph_replay_allocation_delta_bytes"]) != 0:
            reasons.append(f"M{rows} replay allocated device memory")

        correctness = result["correctness"]
        if any(
            not bool(comparison["finite"]) or int(comparison["nonzero"]) == 0
            for comparison in correctness.values()
        ):
            reasons.append(f"M{rows} has a failed finite/nonzero correctness gate")
        production_comparison = correctness.get("candidate_vs_production_w4a16")
        if not isinstance(production_comparison, dict):
            reasons.append(f"M{rows} lacks the production correctness oracle")
        elif (
            not bool(production_comparison["finite"])
            or int(production_comparison["nonzero"]) == 0
            or float(production_comparison["cosine_similarity"]) < 0.999
        ):
            reasons.append(f"M{rows} failed production correctness thresholds")

        for regime in ("hot", "cycling"):
            ratio = result[regime].get("ratio")
            if not isinstance(ratio, dict):
                reasons.append(f"M{rows} {regime} lacks a production ratio")
            elif float(ratio["candidate_over_production"]) > 1.0:
                reasons.append(f"M{rows} {regime} regressed versus W4A16")

        artifact_hashes = result.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            reasons.append(f"M{rows} artifact hashes are missing")
        else:
            candidate = artifact_hashes.get("candidate")
            production = artifact_hashes.get("production_w4a16")
            candidate_objects = (
                [record for name, record in candidate.items() if name.endswith(".o")]
                if isinstance(candidate, dict)
                else []
            )
            production_objects = (
                [record for name, record in production.items() if name.endswith(".o")]
                if isinstance(production, dict)
                else []
            )
            if len(candidate_objects) != 3 or len(production_objects) != 1:
                reasons.append(f"M{rows} exact object map is incomplete")
            for record in candidate_objects + production_objects:
                if (
                    not isinstance(record, dict)
                    or len(str(record.get("sha256", ""))) != 64
                    or not record.get("source_ptxas_versions")
                    or not any(
                        expected_arch_flag in flag
                        for flag in record.get("source_ptxas_flags", ())
                    )
                ):
                    reasons.append(
                        f"M{rows} object hash/PTXAS/SM121 evidence is incomplete"
                    )
                    break

        clock = result.get("clock_evidence")
        if not isinstance(clock, dict):
            reasons.append(f"M{rows} warmed clock evidence is missing")
            continue
        before = clock.get("before")
        after = clock.get("after")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or "error" in before
            or "error" in after
        ):
            reasons.append(f"M{rows} warmed clock snapshots are incomplete")
            continue
        for label, snapshot in (("before", before), ("after", after)):
            if not str(snapshot.get("uuid", "")).endswith(uuid):
                reasons.append(f"M{rows} {label} snapshot UUID differs")
            if snapshot.get("compute_mode") != "Default":
                reasons.append(f"M{rows} {label} compute mode is not Default")
            if snapshot.get("pstate") not in ACCEPTED_PERFORMANCE_PSTATES:
                reasons.append(
                    f"M{rows} {label} snapshot pstate "
                    f"{snapshot.get('pstate')!r} is outside "
                    f"{ACCEPTED_PERFORMANCE_PSTATES}"
                )
            if (
                snapshot.get("clocks_throttle_reasons.active")
                != args.accepted_throttle_mask
            ):
                reasons.append(
                    f"M{rows} {label} throttle mask is "
                    f"{snapshot.get('clocks_throttle_reasons.active')}, expected "
                    f"{args.accepted_throttle_mask}"
                )
        if before.get("clocks.current.memory") != after.get("clocks.current.memory"):
            reasons.append(f"M{rows} memory clock changed during timing")
        if before.get("pstate") != after.get("pstate"):
            reasons.append(f"M{rows} pstate changed during timing")
        delta_pct = clock.get("sm_clock_delta_pct")
        if (
            not isinstance(delta_pct, (int, float))
            or float(delta_pct) > args.max_sm_clock_delta_pct
        ):
            reasons.append(
                f"M{rows} SM-clock delta exceeds {args.max_sm_clock_delta_pct:.2f}%"
            )
    return list(dict.fromkeys(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument(
        "--weight-sets",
        type=int,
        choices=range(1, 17),
        default=8,
    )
    parser.add_argument("--planning-sms", type=int, default=48)
    parser.add_argument("--grid-x", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument(
        "--accepted-throttle-mask",
        default="0x0000000000000000",
    )
    parser.add_argument(
        "--max-sm-clock-delta-pct",
        type=float,
        default=5.0,
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(value not in W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS for value in rows)
    ):
        parser.error("--rows must be unique supported Spark capacity buckets")
    if (
        min(
            args.planning_sms,
            args.grid_x,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        < 1
    ):
        parser.error("planning/timing arguments must be positive")
    if args.max_sm_clock_delta_pct < 0:
        parser.error("max-sm-clock-delta-pct must be nonnegative")

    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    root = pathlib.Path(__file__).resolve().parents[1]
    stream = torch.cuda.Stream()
    weights = [
        _prepare_weight_set(
            device=device,
            seed=args.seed + index,
            candidate_only=args.candidate_only,
        )
        for index in range(args.weight_sets)
    ]
    report_rows: list[dict[str, object]] = []
    for row_count in rows:
        state = _build_row_state(
            rows=row_count,
            weights=weights,
            device=device,
            stream=stream,
            grid_x=args.grid_x,
            planning_sms=args.planning_sms,
            seed=args.seed,
        )
        _warm_graphs(
            state.candidate_graphs,
            state.production_graphs,
            state.phase_graphs,
            stream=stream,
            iterations=args.warmup,
        )
        mode_before_timing = _nvidia_smi_snapshot(str(properties.uuid))
        candidate_hot_samples, production_hot_samples = _measure_pair(
            state.candidate_graphs[:1],
            (
                state.production_graphs[:1]
                if state.production_graphs is not None
                else None
            ),
            stream=stream,
            warmup=0,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        candidate_cycling_samples, production_cycling_samples = _measure_pair(
            state.candidate_graphs,
            state.production_graphs,
            stream=stream,
            warmup=0,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        candidate_hot = _timing(candidate_hot_samples)
        production_hot = (
            _timing(production_hot_samples)
            if production_hot_samples is not None
            else None
        )
        candidate_cycling = _timing(candidate_cycling_samples)
        production_cycling = (
            _timing(production_cycling_samples)
            if production_cycling_samples is not None
            else None
        )
        phases = {
            name: _timing(
                [
                    _measure_batch(
                        [graph],
                        stream=stream,
                        iterations=args.iterations,
                    )
                    for _ in range(args.repeats)
                ]
            )
            for name, graph in state.phase_graphs.items()
        }
        mode_after_timing = _nvidia_smi_snapshot(str(properties.uuid))
        row_report = {
            "rows": row_count,
            "config": state.config,
            "correctness": state.correctness,
            "artifact_hashes": state.artifact_hashes,
            "clock_evidence": _clock_evidence(
                mode_before_timing,
                mode_after_timing,
            ),
            "hot": {
                "candidate": candidate_hot,
                "production_w4a16": production_hot,
                "ratio": _ratio(candidate_hot, production_hot),
            },
            "cycling": {
                "candidate": candidate_cycling,
                "production_w4a16": production_cycling,
                "ratio": _ratio(candidate_cycling, production_cycling),
            },
            "candidate_hot_phases": phases,
        }
        report_rows.append(row_report)
        print(json.dumps(row_report, sort_keys=True), flush=True)

    source = _source_provenance(root)
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_packages": {
            package: importlib.metadata.version(package)
            for package in (
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-core",
                "nvidia-cutlass-dsl-libs-cu12",
                "nvidia-cutlass-dsl-libs-cu13",
            )
        },
        "ptxas_tool": _tool_record("ptxas"),
    }
    device_record = {
        "name": properties.name,
        "uuid": str(properties.uuid),
        "compute_capability": list(capability),
        "hardware_sm_count": int(properties.multi_processor_count),
        "clock_policy": {
            "accepted_performance_pstates": list(ACCEPTED_PERFORMANCE_PSTATES),
            "require_stable_pstate": True,
            "accepted_throttle_mask": args.accepted_throttle_mask,
            "max_sm_clock_delta_pct": args.max_sm_clock_delta_pct,
        },
    }
    ineligibility_reasons = _acceptance_reasons(
        args=args,
        source=source,
        runtime=runtime,
        device=device_record,
        results=report_rows,
    )
    report = {
        "benchmark": "sparkinfer_exact_w4a4_fc1_w4a16_fc2_spark",
        "command": {
            "argv": [sys.executable, *sys.argv],
            "cwd": str(pathlib.Path.cwd()),
        },
        "acceptance": {
            "scope": "balanced M1..256 on physical SM121",
            "eligible": not ineligibility_reasons,
            "ineligibility_reasons": ineligibility_reasons,
        },
        "acceptance_eligible": not ineligibility_reasons,
        "source": source,
        "runtime": runtime,
        "device": device_record,
        "shape": {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "top_k": 1,
            "source_w13_order": "[up; gate]",
        },
        "candidate": {
            "operator": "graph-safe three-kernel composed fused-MoE",
            "weight_residency": (
                "one source-layout W13 plus one in-place packed-layout W2"
            ),
            "input": "prequantized E2M1 with complete E4M3 K16 scales",
        },
        "production_oracle": (
            None if args.candidate_only else "retained packed-W13/packed-W2 fused W4A16"
        ),
        "weight_sets": args.weight_sets,
        "timing": {
            "mode": "CUDA graph replay with CUDA events",
            "compile_cache": (
                "disk disabled; exact timed/exported objects retained in memory"
            ),
            "ordering": "candidate/production order alternates by repeat",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "ratio_direction": (
                "candidate_over_production < 1 and "
                "production_over_candidate_speedup > 1 mean candidate wins"
            ),
        },
        "results": report_rows,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
