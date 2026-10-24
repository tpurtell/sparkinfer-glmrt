#!/usr/bin/env python3
"""Paired AB/BA CUDA-graph benchmark for compensated BF16 prefill projections.

Benchmarks the baseline ``Bf16PrefillKernel(n, k, legacy=True)`` against the
candidate ``Bf16PrefillKernel(n, k)`` (and regime screening variants) in the
same session with:
- Bitwise equality verification (``torch.equal`` on int16/int32 bit patterns)
- Synthetic cancellation accuracy verification against FP64 reference
- Row-invariance verification across live ``num_tokens`` values on one compiled
  callable (no ``M`` in compile/cache keys)
- Paired interleaved AB/BA CUDA graph replay timing and GPU mode snapshots
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cutlass
import cutlass.cute as cute
from cutlass import Int32
import torch

from benchmarks.common import nvidia_smi_gpu_mode_snapshot
from benchmarks.duel_b.ds_a86147cc import _prefill as ds_prefill, _skinny as ds_skinny
from b12x._lib.compiler import DimKey, KernelCompileSpec, compile as b12x_compile, run_compiled, tensor_key
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm.bf16_gemv._prefill import Bf16PrefillKernel, _to_kernel_tensor

STEP5_M_VALUES = (
    1, 2, 4, 8, 16,
    59, 64,
    128, 192, 256,
    257, 300, 320, 383,
    512, 1024, 2048, 4096, 8192,
)

KIMI_M_VALUES = (1, 8, 16, 59, 64, 256, 1024, 2048, 4096, 8192)
SMALL_N_M_VALUES = (1, 8, 64, 256)
SMALL_N_SHAPES = ((384, 5120), (512, 5120), (1024, 5120))
HEAD2HEAD_M_VALUES = (1, 16, 59, 64, 256, 1024, 2048)
HEAD2HEAD_SHAPES = (
    ("q", 6144, 4096, HEAD2HEAD_M_VALUES),
    ("kimi", 24752, 7168, HEAD2HEAD_M_VALUES),
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _compile_kernel_arm(
    arm_name: str,
    kernel_obj: Any,
    max_m: int,
    n: int,
    k: int,
    device: torch.device,
    out_dtype: torch.dtype = torch.bfloat16,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    cute_out_dtype = cutlass.Float32 if out_dtype == torch.float32 else cutlass.BFloat16
    if isinstance(kernel_obj, ds_skinny.RowTiledBf16Projection):
        with torch.cuda.device(device):
            compile_args = (
                make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cute_out_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(max_m),
                current_cuda_stream(),
            )
            key = (arm_name, n, k, str(out_dtype), "pointer_abi")
            compiled = b12x_compile(
                kernel_obj,
                *compile_args,
                compile_spec=KernelCompileSpec.from_key(f"bench.bf16_prefill.{arm_name}", 1, key),
            )

        def run(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
            runtime_args = (
                make_ptr(cutlass.BFloat16, int(x.data_ptr()), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.BFloat16, int(w.data_ptr()), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cute_out_dtype, int(out.data_ptr()), cute.AddressSpace.gmem, assumed_align=16),
                Int32(x.shape[0]),
                current_cuda_stream(),
            )
            run_compiled(compiled, runtime_args)

        return run, compiled

    with torch.cuda.device(device), FakeTensorMode():
        fx = torch.empty((max_m, k), dtype=torch.bfloat16, device=device)
        fw = torch.empty((n, k), dtype=torch.bfloat16, device=device)
        fout = torch.empty((max_m, n), dtype=out_dtype, device=device)
        compile_args = (
            _to_kernel_tensor(fx, cutlass.BFloat16, dynamic_layout=True),
            _to_kernel_tensor(fw, cutlass.BFloat16),
            _to_kernel_tensor(fout, cute_out_dtype, dynamic_layout=True),
            Int32(max_m),
            current_cuda_stream(),
        )
        key = (
            arm_name,
            tuple(
                tensor_key(
                    name,
                    t,
                    dims=(
                        DimKey.dynamic() if name != "weight" else DimKey.exact(t.shape[0]),
                        DimKey.exact(t.shape[1]),
                    ),
                )
                for name, t in (("source", fx), ("weight", fw), ("output", fout))
            ),
        )
        compiled = b12x_compile(
            kernel_obj,
            *compile_args,
            compile_spec=KernelCompileSpec.from_key(f"bench.bf16_prefill.{arm_name}", 1, key),
        )

    def run(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
        runtime_args = (
            _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
            _to_kernel_tensor(w, cutlass.BFloat16),
            _to_kernel_tensor(out, cute_out_dtype, dynamic_layout=True),
            Int32(x.shape[0]),
            current_cuda_stream(),
        )
        run_compiled(compiled, runtime_args)

    return run, compiled


def _verify_cancellation_and_row_invariance(
    n: int,
    k: int,
    base_run,
    cand_run,
    device: torch.device,
) -> dict[str, Any]:
    # 1. Synthetic cancellation test (exact FP64 agreement on cancelling large terms + small carry)
    m_cancel = 64
    x_cancel = torch.ones((m_cancel, k), dtype=torch.bfloat16, device=device)
    w_cancel = torch.zeros((n, k), dtype=torch.bfloat16, device=device)
    w_cancel[:, ::32] = 1024.0
    w_cancel[:, 1::32] = 0.015625
    w_cancel[:, 31::32] = -1024.0
    out_base = torch.empty((m_cancel, n), dtype=torch.bfloat16, device=device)
    out_cand = torch.empty((m_cancel, n), dtype=torch.bfloat16, device=device)
    base_run(x_cancel, w_cancel, out_base)
    cand_run(x_cancel, w_cancel, out_cand)
    torch.cuda.synchronize(device)
    ref_cancel = (x_cancel.double() @ w_cancel.double().T).to(torch.bfloat16)
    cancel_exact_ref = _bitwise_equal(out_cand, ref_cancel)
    cancel_exact_base = _bitwise_equal(out_cand, out_base)
    if not (cancel_exact_ref and cancel_exact_base):
        raise RuntimeError(
            f"Cancellation gate failed on ({n}, {k}): "
            f"cancel_exact_ref={cancel_exact_ref}, cancel_exact_base={cancel_exact_base}"
        )

    # 2. Row invariance across regime boundaries using one compiled callable
    row_test_ms = (1, 8, 16, 59, 64, 128, 256, 257, 512)
    max_m = max(row_test_ms)
    gen = torch.Generator(device=device).manual_seed(9000 + n + k)
    x_full = torch.randn((max_m, k), generator=gen, device=device, dtype=torch.float32).bfloat16()
    w_full = (torch.randn((n, k), generator=gen, device=device, dtype=torch.float32) / math.sqrt(k)).bfloat16()
    outputs: dict[int, torch.Tensor] = {}
    for m in row_test_ms:
        out_m = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        cand_run(x_full[:m], w_full, out_m)
        torch.cuda.synchronize(device)
        outputs[m] = out_m

    for i, m1 in enumerate(row_test_ms):
        for m2 in row_test_ms[i + 1 :]:
            if not _bitwise_equal(outputs[m1], outputs[m2][:m1]):
                raise RuntimeError(
                    f"Row-invariance gate failed on ({n}, {k}) between M1={m1} and M2={m2}"
                )

    return {
        "cancellation_exact_fp64": cancel_exact_ref,
        "cancellation_exact_base": cancel_exact_base,
        "row_invariance_checked_ms": list(row_test_ms),
        "row_invariance_passed": True,
    }


def _time_graph_interleaved(
    arms: list[tuple[str, Any, torch.Tensor]],
    x: torch.Tensor,
    w: torch.Tensor,
    device: torch.device,
    *,
    warmup: int = 5,
    rounds: int = 10,
    replays_per_round: int = 5,
) -> dict[str, list[float]]:
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, run_fn, out_buf in arms:
        for _ in range(warmup):
            run_fn(x, w, out_buf)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(replays_per_round):
                run_fn(x, w, out_buf)
        torch.cuda.synchronize(device)
        mem_before_replay = torch.cuda.memory_allocated(device)
        g.replay()
        torch.cuda.synchronize(device)
        mem_after_replay = torch.cuda.memory_allocated(device)
        if mem_after_replay != mem_before_replay:
            raise RuntimeError(
                f"Arm {name} allocated memory during replay: {mem_before_replay} -> {mem_after_replay}"
            )
        graphs[name] = g

    samples_us: dict[str, list[float]] = {name: [] for name, _, _ in arms}
    start_events = [torch.cuda.Event(enable_timing=True) for _ in arms]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in arms]

    for r in range(rounds):
        order = list(range(len(arms))) if (r % 2 == 0) else list(reversed(range(len(arms))))
        for idx in order:
            name = arms[idx][0]
            start_events[idx].record()
            graphs[name].replay()
            end_events[idx].record()
        torch.cuda.synchronize(device)
        for idx in order:
            name = arms[idx][0]
            elapsed_ms = start_events[idx].elapsed_time(end_events[idx])
            samples_us[name].append((elapsed_ms * 1000.0) / replays_per_round)

    for g in graphs.values():
        g.reset()

    return samples_us


def _tflops(m: int, n: int, k: int, us: float) -> float:
    return (2.0 * m * n * k) / (us * 1e6)


def _build_screening_candidates(
    n: int, k: int, m_list: tuple[int, ...]
) -> list[tuple[str, Bf16PrefillKernel, tuple[int, ...]]]:
    candidates: list[tuple[str, Bf16PrefillKernel, tuple[int, ...]]] = [
        ("baseline", Bf16PrefillKernel(n, k, legacy=True), m_list),
        ("auto_default", Bf16PrefillKernel(n, k), m_list),
    ]
    if (n, k) not in ((6144, 4096), (24752, 7168)):
        candidates.extend([
            ("s_16x32_s8_u1", Bf16PrefillKernel(n, k, warp_layout=(1, 4), tile_m=16, tile_n=32, num_stages=8, k_unroll=1, buffer_align_bytes=128, group_m=1), (1, 8, 64)),
            ("sh_w2x2_32x32_s5_u1", Bf16PrefillKernel(n, k, warp_layout=(2, 2), tile_m=32, tile_n=32, num_stages=5, k_unroll=1, buffer_align_bytes=128, group_m=1), (8, 64, 256)),
            ("sh_w4x2_64x32_s4_u1", Bf16PrefillKernel(n, k, warp_layout=(4, 2), tile_m=64, tile_n=32, num_stages=4, k_unroll=1, buffer_align_bytes=128, group_m=1), (64, 256)),
        ])
    return candidates


def _extract_fatbin_bytes(ir_text: str) -> bytes:
    for line in ir_text.splitlines():
        if "@kernels_binary(" in line:
            start = line.find('@kernels_binary("') + len('@kernels_binary("')
            end = line.rfind('")')
            raw = line[start:end]
            out = bytearray()
            i = 0
            while i < len(raw):
                if raw[i] == "\\":
                    if raw[i + 1] == "\\":
                        out.append(ord("\\"))
                        i += 2
                    elif raw[i + 1] == '"':
                        out.append(ord('"'))
                        i += 2
                    else:
                        out.append(int(raw[i + 1 : i + 3], 16))
                        i += 3
                else:
                    out.append(ord(raw[i]))
                    i += 1
            return bytes(out)
    raise RuntimeError("No @kernels_binary found in compiled IR module")


def _verify_single_kernel_invariants(k_obj: Bf16PrefillKernel, m_values: tuple[int, ...], m_tile_offset: int = 0) -> None:
    assert not hasattr(k_obj, "setmaxnreg"), "setmaxnreg path must not exist on Bf16PrefillKernel"
    assert k_obj.buffer_align_bytes >= 128 and (k_obj.buffer_align_bytes & (k_obj.buffer_align_bytes - 1)) == 0
    assert k_obj.k_tiles == k_obj.k // 64 and k_obj.k_tiles >= 1
    assert k_obj.n_tiles == (k_obj.n + k_obj.tile_n - 1) // k_obj.tile_n and k_obj.n_tiles >= 1
    assert k_obj.producer_warp == k_obj.num_compute_warps
    assert k_obj.num_threads == 32 * (k_obj.num_compute_warps + 1)
    for m in m_values:
        active_rows = m - m_tile_offset * k_obj.tile_m
        if active_rows <= 0:
            continue
        grid_m = (active_rows + k_obj.tile_m - 1) // k_obj.tile_m
        assert grid_m >= 1
        if k_obj.group_m > 1:
            visited: set[tuple[int, int]] = set()
            for raw_m in range(grid_m):
                for raw_n in range(k_obj.n_tiles):
                    group_id = raw_m // k_obj.group_m
                    first_pid_m = group_id * k_obj.group_m
                    in_group = (raw_m % k_obj.group_m) * k_obj.n_tiles + raw_n
                    rem_m = grid_m - first_pid_m
                    assert rem_m >= 1
                    if rem_m >= k_obj.group_m:
                        local_m = first_pid_m + (in_group % k_obj.group_m)
                        n_tile = in_group // k_obj.group_m
                    else:
                        local_m = first_pid_m + (in_group % rem_m)
                        n_tile = in_group // rem_m
                    assert 0 <= local_m < grid_m and 0 <= n_tile < k_obj.n_tiles
                    visited.add((local_m, n_tile))
            assert len(visited) == grid_m * k_obj.n_tiles


def _verify_no_hang_cpu(compile_sass: bool = True) -> None:
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    all_shapes = (
        (6144, 4096, STEP5_M_VALUES),
        (24752, 7168, KIMI_M_VALUES),
        (384, 5120, SMALL_N_M_VALUES),
        (512, 5120, SMALL_N_M_VALUES),
        (1024, 5120, SMALL_N_M_VALUES),
    )
    for n, k, m_list in all_shapes:
        for arm_name, k_obj, target_ms in _build_screening_candidates(n, k, m_list):
            _verify_single_kernel_invariants(k_obj, target_ms, 0)
            if k_obj._row_kernels is not None:
                for sub_k, fl, lim in zip(k_obj._row_kernels, k_obj._row_floors, k_obj._row_limits):
                    sub_ms = tuple(m for m in target_ms if fl < m <= lim)
                    _verify_single_kernel_invariants(sub_k, sub_ms or (lim,), 0)
            if k_obj._chunk256_tail is not None:
                _verify_single_kernel_invariants(k_obj._chunk256_kernel, (256,), 0)
                tail_ms = tuple(m for m in target_ms if k_obj._tail_floor < m <= k_obj._tail_limit)
                _verify_single_kernel_invariants(
                    k_obj._tail_kernel,
                    tail_ms,
                    256 // k_obj._tail_kernel.tile_m,
                )

    if compile_sass:
        to_compile = []
        seen_sigs = set()
        for n, k, m_list in (
            (6144, 4096, STEP5_M_VALUES),
            (24752, 7168, KIMI_M_VALUES),
            (384, 5120, SMALL_N_M_VALUES),
        ):
            for arm_name, k_obj, _ in _build_screening_candidates(n, k, m_list):
                is_default = arm_name in ("baseline", "auto_default")
                sig = (
                    n, k, is_default,
                    k_obj.warp_layout, k_obj.tile_m, k_obj.tile_n, k_obj.num_stages,
                    k_obj.k_unroll, k_obj.group_m > 1,
                    k_obj._row_limits, k_obj._chunk256_tail is not None,
                )
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                to_compile.append((f"{n}x{k}_{arm_name}", n, k, k_obj, is_default))
        for label, n, k, k_obj, is_default in to_compile:
            m_sym, s_sym = cute.sym_int(32), cute.sym_int(64)
            fx = make_fake_tensor(cutlass.BFloat16, (m_sym, k), (s_sym, 1), assumed_align=16)
            fw = make_fake_tensor(cutlass.BFloat16, (n, k), (k, 1), assumed_align=16)
            fout = make_fake_tensor(cutlass.BFloat16, (m_sym, n), (s_sym, 1), assumed_align=16)
            compiled = cute.compile(
                k_obj,
                fx,
                fw,
                fout,
                Int32(256),
                make_fake_stream(),
                options="--gpu-arch=sm_120a",
                no_jit_engine=True,
            )
            fatbin = _extract_fatbin_bytes(str(compiled.ir_module))
            cubin = fatbin[fatbin.find(b"\x7fELF") :]
            with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
                f.write(cubin)
                f.flush()
                res_out = subprocess.check_output(["cuobjdump", "-res-usage", f.name], text=True)
                sass_out = subprocess.check_output(["cuobjdump", "-sass", f.name], text=True)
            assert "USETMAXREG" not in sass_out and "SETMAXNREG" not in sass_out, (
                f"{label} cubin contains SETMAXNREG instruction!"
            )
            if is_default:
                for line in res_out.splitlines():
                    if "REG:" in line:
                        assert "STACK:0" in line, f"{label} has non-zero stack: {line}"
            fn_blocks = sass_out.split("Function : ")[1:]
            assert len(fn_blocks) >= 1, f"{label} has no SASS functions"
            for fn_block in fn_blocks:
                assert "SYNCS.PHASECHK.TRANS64.TRYWAIT" in fn_block, f"{label} missing consumer/producer wait"
                assert "SYNCS.ARRIVE.TRANS64" in fn_block, f"{label} missing barrier commit/release"
                assert "UTMALDG.2D" in fn_block, f"{label} missing TMA 2D load"

        for shape_label, n, k, _ in HEAD2HEAD_SHAPES:
            ds_canon = ds_prefill.Bf16PrefillKernel(n, k)
            ds_rt = ds_skinny.RowTiledBf16Projection(
                n, k, ds_skinny.COMPENSATED_TILES_BY_ROWS[(n, k)], compensated=True
            )
            m_sym, s_sym = cute.sym_int(32), cute.sym_int(64)
            fx = make_fake_tensor(cutlass.BFloat16, (m_sym, k), (s_sym, 1), assumed_align=16)
            fw = make_fake_tensor(cutlass.BFloat16, (n, k), (k, 1), assumed_align=16)
            fout = make_fake_tensor(cutlass.BFloat16, (m_sym, n), (s_sym, 1), assumed_align=16)
            compiled_canon = cute.compile(
                ds_canon, fx, fw, fout, Int32(256), make_fake_stream(),
                options="--gpu-arch=sm_120a", no_jit_engine=True,
            )
            compiled_rt = cute.compile(
                ds_rt,
                make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(256),
                make_fake_stream(),
                options="--gpu-arch=sm_120a",
                no_jit_engine=True,
            )
            for tag, comp_mod in ((f"{shape_label}_ds_canon", compiled_canon), (f"{shape_label}_ds_rowtile", compiled_rt)):
                fatbin = _extract_fatbin_bytes(str(comp_mod.ir_module))
                cubin = fatbin[fatbin.find(b"\x7fELF") :]
                with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
                    f.write(cubin)
                    f.flush()
                    sass_out = subprocess.check_output(["cuobjdump", "-sass", f.name], text=True)
                assert "USETMAXREG" not in sass_out and "SETMAXNREG" not in sass_out, (
                    f"{tag} cubin contains SETMAXNREG instruction!"
                )
                fn_blocks = sass_out.split("Function : ")[1:]
                assert len(fn_blocks) >= 1, f"{tag} has no SASS functions"
                for fn_block in fn_blocks:
                    assert "SYNCS.PHASECHK.TRANS64.TRYWAIT" in fn_block, f"{tag} missing consumer/producer wait"
                    assert "SYNCS.ARRIVE.TRANS64" in fn_block, f"{tag} missing barrier commit/release"
                    assert "UTMALDG.2D" in fn_block, f"{tag} missing TMA 2D load"


def _run_screening(
    device: torch.device,
    output_path: pathlib.Path | None = None,
    shape_filter: set[str] | None = None,
    m_filter: set[int] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "gpu_mode_before": nvidia_smi_gpu_mode_snapshot(),
    }
    for n, k, m_list in (
        (6144, 4096, STEP5_M_VALUES),
        (24752, 7168, (1, 8, 16, 59, 64, 256, 512, 1024, 2048, 8192)),
        (384, 5120, SMALL_N_M_VALUES),
        (512, 5120, SMALL_N_M_VALUES),
        (1024, 5120, SMALL_N_M_VALUES),
    ):
        shape_key = f"{n}x{k}"
        if shape_filter is not None and shape_key not in shape_filter:
            continue
        active_m_list = tuple(m for m in m_list if m_filter is None or m in m_filter)
        if not active_m_list:
            continue
        print(f"\n=== Screening shape N={n}, K={k} (M={active_m_list}) ===", flush=True)
        all_candidates = _build_screening_candidates(n, k, active_m_list)
        candidates = [
            (arm_name, k_obj, tuple(m for m in target_ms if m in active_m_list))
            for arm_name, k_obj, target_ms in all_candidates
            if any(m in active_m_list for m in target_ms)
        ]

        compiled_arms: dict[str, tuple[Any, tuple[int, ...]]] = {}
        max_m = max(active_m_list)
        for arm_name, k_obj, target_ms in candidates:
            run_fn, _ = _compile_kernel_arm(f"{shape_key}_{arm_name}", k_obj, max_m, n, k, device)
            compiled_arms[arm_name] = (run_fn, target_ms)

        gen = torch.Generator(device=device).manual_seed(1234 + n + k)
        w = (torch.randn((n, k), generator=gen, device=device, dtype=torch.float32) / math.sqrt(k)).bfloat16()
        x_max = torch.randn((max_m, k), generator=gen, device=device, dtype=torch.float32).bfloat16()

        shape_rows: dict[int, Any] = {}
        aborted_arms: set[str] = set()
        for m in active_m_list:
            x = x_max[:m]
            active_arms = []
            arm_wall_ms: dict[str, float] = {}
            base_wall_s: float | None = None
            base_out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
            check_buf = torch.empty((m, n), dtype=torch.bfloat16, device=device)
            row_summary: dict[str, Any] = {}
            for arm_name, (run_fn, target_ms) in compiled_arms.items():
                if m in target_ms and arm_name not in aborted_arms:
                    buf = base_out if arm_name == "baseline" else check_buf
                    torch.cuda.synchronize(device)
                    t_arm0 = time.perf_counter()
                    run_fn(x, w, buf)
                    torch.cuda.synchronize(device)
                    wall_s = time.perf_counter() - t_arm0
                    arm_wall_ms[arm_name] = wall_s * 1000.0
                    if arm_name == "baseline":
                        base_wall_s = wall_s
                    elif base_wall_s is not None and wall_s > max(3.0 * base_wall_s, 0.050):
                        print(
                            f"  [FAST-ABORT] Arm {arm_name} wall {wall_s*1e3:.2f}ms > 3x base "
                            f"({base_wall_s*1e3:.2f}ms) on {shape_key} M={m}; skipping arm.",
                            flush=True,
                        )
                        aborted_arms.add(arm_name)
                        row_summary[arm_name] = {
                            "aborted_slow": True,
                            "first_call_wall_ms": arm_wall_ms[arm_name],
                        }
                        continue
                    if arm_name != "baseline" and not _bitwise_equal(buf, base_out):
                        raise RuntimeError(f"Bitwise mismatch on {shape_key} M={m} for arm {arm_name}")
                    active_arms.append((arm_name, run_fn))

            replays = 2 if m >= 4096 else (5 if m >= 256 else 10)
            base_run_fn = compiled_arms["baseline"][0]
            cand_only = [item for item in active_arms if item[0] != "baseline"]
            batch_size = 10
            all_samples: dict[str, list[float]] = {"baseline": []}
            for b_start in range(0, max(1, len(cand_only)), batch_size):
                batch_cands = cand_only[b_start : b_start + batch_size]
                batch_bufs = [
                    ("baseline", base_run_fn, base_out),
                    *[
                        (c_name, c_fn, torch.empty((m, n), dtype=torch.bfloat16, device=device))
                        for c_name, c_fn in batch_cands
                    ],
                ]
                batch_samples = _time_graph_interleaved(
                    batch_bufs, x, w, device, warmup=3, rounds=6, replays_per_round=replays
                )
                all_samples["baseline"].extend(batch_samples["baseline"])
                for c_name, _ in batch_cands:
                    all_samples[c_name] = batch_samples[c_name]
                del batch_bufs

            base_med = statistics.median(all_samples["baseline"])
            line_parts = [f"M={m:4d} | base={base_med:7.1f}us ({_tflops(m, n, k, base_med):5.1f} TF)"]
            for arm_name, _ in active_arms:
                med = statistics.median(all_samples[arm_name])
                tf = _tflops(m, n, k, med)
                sp = base_med / med
                row_summary[arm_name] = {
                    "median_us": med,
                    "tflops": tf,
                    "speedup": sp,
                    "first_call_wall_ms": arm_wall_ms[arm_name],
                }
                if arm_name != "baseline":
                    line_parts.append(f"{arm_name}={med:6.1f}us ({tf:5.1f} TF, {sp:4.2f}x)")
            print(" | ".join(line_parts), flush=True)
            shape_rows[m] = row_summary
            results[shape_key] = shape_rows
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(results, indent=2) + "\n")
    results["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def _run_duel(
    device: torch.device,
    include_small_n: bool = True,
    output_path: pathlib.Path | None = None,
    shape_filter: set[str] | None = None,
    m_filter: set[int] | None = None,
) -> dict[str, Any]:
    shapes_to_run = [
        ("step5_6144x4096", 6144, 4096, STEP5_M_VALUES),
        ("kimi_24752x7168", 24752, 7168, KIMI_M_VALUES),
    ]
    if include_small_n:
        for sn, sk in SMALL_N_SHAPES:
            shapes_to_run.append((f"small_{sn}x{sk}", sn, sk, SMALL_N_M_VALUES))

    mode_before = nvidia_smi_gpu_mode_snapshot()
    report: dict[str, Any] = {
        "gpu_mode_before": mode_before,
        "shapes": {},
    }

    for label, n, k, m_list in shapes_to_run:
        shape_key = f"{n}x{k}"
        if shape_filter is not None and shape_key not in shape_filter and label not in shape_filter:
            continue
        active_m_list = tuple(m for m in m_list if m_filter is None or m in m_filter)
        if not active_m_list:
            continue
        print(f"\n=== Duel AB/BA on {label} (N={n}, K={k}, M={active_m_list}) ===", flush=True)
        max_m = max(active_m_list)
        base_kernel = Bf16PrefillKernel(n, k, legacy=True)
        cand_kernel = Bf16PrefillKernel(n, k)
        base_run, _ = _compile_kernel_arm(f"{label}_baseline", base_kernel, max_m, n, k, device)
        cand_run, _ = _compile_kernel_arm(f"{label}_candidate", cand_kernel, max_m, n, k, device)

        invariants = _verify_cancellation_and_row_invariance(n, k, base_run, cand_run, device)
        print(f"Invariants {label}: {invariants}", flush=True)

        gen = torch.Generator(device=device).manual_seed(4242 + n + k)
        w = (torch.randn((n, k), generator=gen, device=device, dtype=torch.float32) / math.sqrt(k)).bfloat16()
        x_max = torch.randn((max_m, k), generator=gen, device=device, dtype=torch.float32).bfloat16()

        cells: list[dict[str, Any]] = []
        print(f"{'M':>5s} | {'Base (us)':>10s} | {'Cand (us)':>10s} | {'Base TF/s':>10s} | {'Cand TF/s':>10s} | {'Speedup':>8s} | {'Bitwise':>7s}", flush=True)
        print("-" * 78, flush=True)
        for m in active_m_list:
            x = x_max[:m]
            out_base = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
            out_cand = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
            torch.cuda.synchronize(device)
            t_b0 = time.perf_counter()
            base_run(x, w, out_base)
            torch.cuda.synchronize(device)
            base_wall_s = time.perf_counter() - t_b0

            t_c0 = time.perf_counter()
            cand_run(x, w, out_cand)
            torch.cuda.synchronize(device)
            cand_wall_s = time.perf_counter() - t_c0
            if cand_wall_s > max(3.0 * base_wall_s, 0.050):
                raise RuntimeError(
                    f"Candidate wall time {cand_wall_s*1e3:.2f}ms exceeded 3x baseline "
                    f"({base_wall_s*1e3:.2f}ms) on {label} M={m}; aborting fast."
                )

            bit_eq = _bitwise_equal(out_cand, out_base)
            if not bit_eq:
                raise RuntimeError(f"Bitwise equality check failed on {label} M={m}")

            replays = 3 if m >= 4096 else (5 if m >= 256 else 10)
            samples = _time_graph_interleaved(
                [("baseline", base_run, out_base), ("candidate", cand_run, out_cand)],
                x,
                w,
                device,
                warmup=5,
                rounds=10,
                replays_per_round=replays,
            )
            base_med = statistics.median(samples["baseline"])
            cand_med = statistics.median(samples["candidate"])
            base_tf = _tflops(m, n, k, base_med)
            cand_tf = _tflops(m, n, k, cand_med)
            speedup = base_med / cand_med
            print(
                f"{m:5d} | {base_med:10.2f} | {cand_med:10.2f} | {base_tf:10.2f} | {cand_tf:10.2f} | {speedup:7.2f}x | {str(bit_eq):>7s}",
                flush=True,
            )
            cells.append({
                "m": m,
                "n": n,
                "k": k,
                "baseline_median_us": base_med,
                "candidate_median_us": cand_med,
                "baseline_tflops": base_tf,
                "candidate_tflops": cand_tf,
                "speedup": speedup,
                "bitwise_equal": bit_eq,
                "baseline_first_call_wall_ms": base_wall_s * 1000.0,
                "candidate_first_call_wall_ms": cand_wall_s * 1000.0,
                "baseline_samples_us": samples["baseline"],
                "candidate_samples_us": samples["candidate"],
            })

        report["shapes"][label] = {
            "n": n,
            "k": k,
            "invariants": invariants,
            "cells": cells,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2) + "\n")

    report["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _run_head2head(
    device: torch.device,
    output_path: pathlib.Path | None = None,
    shape_filter: set[str] | None = None,
    m_filter: set[int] | None = None,
) -> dict[str, Any]:
    mode_before = nvidia_smi_gpu_mode_snapshot()
    report: dict[str, Any] = {
        "gpu_mode_before": mode_before,
        "shapes": {},
    }

    for label, n, k, m_list in HEAD2HEAD_SHAPES:
        shape_key = f"{n}x{k}"
        if shape_filter is not None and shape_key not in shape_filter and label not in shape_filter:
            continue
        active_m_list = tuple(m for m in m_list if m_filter is None or m in m_filter)
        if not active_m_list:
            continue
        print(f"\n=== Head-to-Head on {label} (N={n}, K={k}, rows={active_m_list}) ===", flush=True)
        max_m = max(max(active_m_list), 512)

        canon_obj = ds_prefill.Bf16PrefillKernel(n, k)
        canon_repeat_obj = ds_prefill.Bf16PrefillKernel(n, k)
        ds_rowtile_obj = ds_skinny.RowTiledBf16Projection(
            n, k, ds_skinny.COMPENSATED_TILES_BY_ROWS[(n, k)], compensated=True
        )
        gemini_obj = Bf16PrefillKernel(n, k)

        canon_run, _ = _compile_kernel_arm(f"h2h_{label}_canon", canon_obj, max_m, n, k, device)
        canon_repeat_run, _ = _compile_kernel_arm(
            f"h2h_{label}_canon_repeat", canon_repeat_obj, max_m, n, k, device
        )
        ds_rowtile_run, _ = _compile_kernel_arm(
            f"h2h_{label}_ds_rowtile", ds_rowtile_obj, max_m, n, k, device
        )
        gemini_run, _ = _compile_kernel_arm(f"h2h_{label}_gemini", gemini_obj, max_m, n, k, device)

        inv_ds = _verify_cancellation_and_row_invariance(n, k, canon_run, ds_rowtile_run, device)
        inv_gemini = _verify_cancellation_and_row_invariance(n, k, canon_run, gemini_run, device)
        invariants = {
            "ds_rowtile": inv_ds,
            "gemini": inv_gemini,
        }
        print(f"Invariants {label}: {invariants}", flush=True)

        gen = torch.Generator(device=device).manual_seed(4242 + n + k)
        w = (torch.randn((n, k), generator=gen, device=device, dtype=torch.float32) / math.sqrt(k)).bfloat16()
        x_max = torch.randn((max_m, k), generator=gen, device=device, dtype=torch.float32).bfloat16()

        cells: list[dict[str, Any]] = []
        dropped_arms: dict[str, str] = {}
        print(
            f"{'shape':>6s} | {'rows':>5s} | {'canon us':>9s} | {'canon2 us':>9s} | "
            f"{'ds_rowtile us (ratio)':>22s} | {'gemini us (ratio)':>20s} | "
            f"{'spread':>7s} | {'margin':>7s} | {'winner':>18s}",
            flush=True,
        )
        print("-" * 124, flush=True)

        all_arm_runners = [
            ("canon", canon_run),
            ("canon_repeat", canon_repeat_run),
            ("ds_rowtile", ds_rowtile_run),
            ("gemini", gemini_run),
        ]

        for m in active_m_list:
            x = x_max[:m]
            arm_bufs: dict[str, torch.Tensor] = {}
            bitwise_flags: dict[str, bool] = {}
            active_for_cell: list[tuple[str, Any, torch.Tensor]] = []

            for arm_name, run_fn in all_arm_runners:
                if arm_name in dropped_arms:
                    bitwise_flags[arm_name] = False
                    continue
                buf = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
                run_fn(x, w, buf)
                torch.cuda.synchronize(device)
                arm_bufs[arm_name] = buf
                if arm_name == "canon":
                    bitwise_flags[arm_name] = True
                    active_for_cell.append((arm_name, run_fn, buf))
                else:
                    eq = _bitwise_equal(buf, arm_bufs["canon"])
                    bitwise_flags[arm_name] = eq
                    if not eq:
                        if arm_name == "canon_repeat":
                            raise RuntimeError(f"canon_repeat bitwise mismatch on {label} rows={m}")
                        dropped_arms[arm_name] = f"bitwise_mismatch_at_rows_{m}"
                        print(f"  [DROP] {arm_name} failed bitwise equality on {label} rows={m}", flush=True)
                    else:
                        active_for_cell.append((arm_name, run_fn, buf))

            replays = 3 if m >= 4096 else (5 if m >= 256 else 10)
            samples = _time_graph_interleaved(
                active_for_cell,
                x,
                w,
                device,
                warmup=5,
                rounds=10,
                replays_per_round=replays,
            )

            canon_med = statistics.median(samples["canon"])
            canon_repeat_med = statistics.median(samples["canon_repeat"])
            canon_mean = 0.5 * (canon_med + canon_repeat_med)
            raw_spread = abs(canon_med - canon_repeat_med) / canon_mean
            threshold = max(0.015, raw_spread)

            ds_med = statistics.median(samples["ds_rowtile"]) if "ds_rowtile" in samples else float("inf")
            gem_med = statistics.median(samples["gemini"]) if "gemini" in samples else float("inf")
            ds_ratio = canon_med / ds_med if math.isfinite(ds_med) else 0.0
            gem_ratio = canon_med / gem_med if math.isfinite(gem_med) else 0.0
            ds_ratio_mean = canon_mean / ds_med if math.isfinite(ds_med) else 0.0
            gem_ratio_mean = canon_mean / gem_med if math.isfinite(gem_med) else 0.0

            if not math.isfinite(ds_med) and math.isfinite(gem_med):
                faster_arm = "gemini"
                margin = float("inf")
                verdict = "WIN"
                winner = "gemini"
            elif math.isfinite(ds_med) and not math.isfinite(gem_med):
                faster_arm = "ds_rowtile"
                margin = float("inf")
                verdict = "WIN"
                winner = "ds_rowtile"
            else:
                if gem_med < ds_med:
                    faster_arm = "gemini"
                    margin = (ds_med / gem_med) - 1.0
                else:
                    faster_arm = "ds_rowtile"
                    margin = (gem_med / ds_med) - 1.0
                if margin > threshold:
                    verdict = "WIN"
                    winner = faster_arm
                else:
                    verdict = "TIE"
                    winner = "TIE -> ds_rowtile"

            print(
                f"{label:>6s} | {m:5d} | {canon_med:9.2f} | {canon_repeat_med:9.2f} | "
                f"{ds_med:9.2f} ({ds_ratio:6.3f}x)    | {gem_med:9.2f} ({gem_ratio:6.3f}x)  | "
                f"{raw_spread*100:6.2f}% | {margin*100:6.2f}% | {winner:>18s}",
                flush=True,
            )

            cells.append({
                "shape": label,
                "rows": m,
                "n": n,
                "k": k,
                "baseline_us": canon_med,
                "canon_median_us": canon_med,
                "canon_repeat_median_us": canon_repeat_med,
                "canon_mean_us": canon_mean,
                "ds_rowtile_median_us": ds_med if math.isfinite(ds_med) else None,
                "gemini_median_us": gem_med if math.isfinite(gem_med) else None,
                "ds_rowtile_ratio_vs_canon": ds_ratio,
                "gemini_ratio_vs_canon": gem_ratio,
                "ds_rowtile_ratio_vs_canon_mean": ds_ratio_mean,
                "gemini_ratio_vs_canon_mean": gem_ratio_mean,
                "canon_tflops": _tflops(m, n, k, canon_med),
                "canon_repeat_tflops": _tflops(m, n, k, canon_repeat_med),
                "ds_rowtile_tflops": _tflops(m, n, k, ds_med) if math.isfinite(ds_med) else 0.0,
                "gemini_tflops": _tflops(m, n, k, gem_med) if math.isfinite(gem_med) else 0.0,
                "bitwise_flags": bitwise_flags,
                "raw_spread": raw_spread,
                "threshold": threshold,
                "faster_arm": faster_arm,
                "margin": margin,
                "verdict": verdict,
                "winner": winner,
                "samples_us": samples,
            })

        report["shapes"][label] = {
            "n": n,
            "k": k,
            "invariants": invariants,
            "dropped_arms": dropped_arms,
            "cells": cells,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2) + "\n")

    report["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("duel", "screen", "both", "head2head"), default="duel")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--include-small-n", action="store_true", default=True)
    parser.add_argument("--no-small-n", action="store_false", dest="include_small_n")
    parser.add_argument("--shapes", type=str, default=None, help="Comma-separated NxK filter, e.g. 6144x4096")
    parser.add_argument("--m-values", type=str, default=None, help="Comma-separated M filter, e.g. 256")
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--output-screen-json", type=pathlib.Path, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-cpu", action="store_true")
    args = parser.parse_args()

    if args.preflight or args.verify_cpu:
        base_k = Bf16PrefillKernel(6144, 4096, legacy=True)
        cand_k = Bf16PrefillKernel(6144, 4096)
        kimi_k = Bf16PrefillKernel(24752, 7168)
        assert base_k.legacy and not cand_k.legacy and not kimi_k.legacy
        _verify_no_hang_cpu(compile_sass=args.verify_cpu)
        print(
            f"Preflight OK (compile_sass={args.verify_cpu}): Bf16PrefillKernel baseline and candidate "
            "configurations validated (no setmaxnreg, PipelineTmaAsync bounds and coords verified).",
            flush=True,
        )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmark_bf16_prefill_compensated.py")

    shape_filter = set(s.strip() for s in args.shapes.split(",") if s.strip()) if args.shapes else None
    m_filter = set(int(x.strip()) for x in args.m_values.split(",") if x.strip()) if args.m_values else None

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    t0 = time.time()
    if args.mode == "head2head":
        data = _run_head2head(
            device, output_path=args.output_json, shape_filter=shape_filter, m_filter=m_filter
        )
        data["elapsed_wall_seconds"] = time.time() - t0
    elif args.mode == "screen":
        data = _run_screening(
            device, output_path=args.output_json, shape_filter=shape_filter, m_filter=m_filter
        )
        data["elapsed_wall_seconds"] = time.time() - t0
    elif args.mode == "both":
        duel_t0 = time.time()
        data = _run_duel(
            device,
            include_small_n=args.include_small_n,
            output_path=args.output_json,
            shape_filter=shape_filter,
            m_filter=m_filter,
        )
        data["elapsed_wall_seconds"] = time.time() - duel_t0
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(data, indent=2) + "\n")
            print(f"\nWrote duel receipt to {args.output_json}", flush=True)
        screen_t0 = time.time()
        screen_data = _run_screening(
            device,
            output_path=args.output_screen_json,
            shape_filter=shape_filter,
            m_filter=m_filter,
        )
        screen_data["elapsed_wall_seconds"] = time.time() - screen_t0
        if args.output_screen_json is not None:
            args.output_screen_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_screen_json.write_text(json.dumps(screen_data, indent=2) + "\n")
            print(f"\nWrote screening receipt to {args.output_screen_json}", flush=True)
        return 0
    else:
        data = _run_duel(
            device,
            include_small_n=args.include_small_n,
            output_path=args.output_json,
            shape_filter=shape_filter,
            m_filter=m_filter,
        )
        data["elapsed_wall_seconds"] = time.time() - t0

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(data, indent=2) + "\n")
        print(f"\nWrote receipt to {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
