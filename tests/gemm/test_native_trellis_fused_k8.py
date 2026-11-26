"""Bitwise oracles, row invariance, graph replay, and CPU export tests for fused K8 trellis decode."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest
import torch


# Representative K8 shapes covering all nine shared EXL3 families, both codebooks,
# both storage sides, all output dtypes (bf16, f32, raw_f32), and every route regime
# (narrow N16, wide SMEM64, wide SMEM128, splits=1, non-power-of-2 K, lm_head).
FAMILY_GATE_SHAPES = (
    # (family, name, k, n, side, codebook, output_dtype)
    ("hy4", "hy4_kv_a_padded", 6144, 640, 1, "mul1", "bf16"),
    ("hy4", "hy4_q_a", 6144, 2048, 1, "mul1", "bf16"),
    ("hy4", "hy4_q_b_dsa", 2048, 16384, 1, "mul1", "bf16"),
    ("hy4", "hy4_gate_dsa", 6144, 16384, 1, "mul1", "bf16"),
    ("hy4", "hy4_o_dsa", 16384, 6144, 1, "mul1", "bf16"),
    ("hy4", "hy4_o_dsa_tp2", 8192, 6144, 2, "mul1", "raw_f32"),
    ("hy4", "hy4_shared_down", 2048, 6144, 1, "mul1", "bf16"),
    ("hy4", "hy4_shared_down_tp2", 1024, 6144, 2, "mul1", "raw_f32"),
    ("hy4", "hy4_lm_head", 6144, 120832, 1, "mul1", "f32"),
    ("hy4f", "hy4f_q_a", 3072, 1024, 1, "mul1", "bf16"),
    ("hy4f", "hy4f_o_dsa_tp2", 4096, 3072, 2, "mul1", "raw_f32"),
    ("glm", "glm_q_a", 6144, 2048, 1, "mcg", "bf16"),
    ("glm", "glm_o_tp2", 8192, 6144, 2, "mcg", "raw_f32"),
    ("glmf", "glmf_kv_a", 4096, 512, 1, "mcg", "bf16"),
    ("glmf", "glmf_kda_qkv_tp2", 4096, 4096, 1, "mcg", "bf16"),
    ("glmf", "glmf_shared_down_tp2", 1024, 4096, 2, "mcg", "raw_f32"),
    ("kimik3", "mla_kv_a_padded", 7168, 640, 1, "mcg", "bf16"),
    ("kimik3", "shared_down_tp2", 3072, 7168, 2, "mcg", "raw_f32"),
    ("muse", "muse_q_tp2", 8192, 4096, 1, "mcg", "bf16"),
    ("muse", "muse_o_tp2", 4096, 8192, 2, "mcg", "f32"),
    ("ml4", "ml4_q_tp2", 6144, 4096, 1, "mcg", "bf16"),
    ("ml4", "ml4_shared_down_tp2", 3072, 6144, 2, "mcg", "raw_f32"),
    ("step5", "step5_kv_tp2", 4096, 384, 1, "mcg", "bf16"),
    ("step5", "step5_o_tp2", 6144, 4096, 2, "mcg", "raw_f32"),
    ("inkling2", "inkling_rel_tp2", 4608, 384, 1, "mcg", "f32"),
    ("inkling2", "inkling_o_tp2", 3072, 4608, 2, "mcg", "raw_f32"),
)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _run_unfused_oracle(
    x: torch.Tensor,
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    k: int,
    n: int,
    side: int,
    codebook: str,
    output_dtype: str,
    max_rows: int,
    rows: int,
    cta_limit: int,
) -> torch.Tensor:
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot,
        compile_trellis_output_aot,
    )

    splits = (k + 1023) // 1024
    rotated = torch.empty((max_rows, k), device="cuda", dtype=torch.bfloat16)
    blocks = torch.empty(k // 128, device="cuda", dtype=torch.float16)
    ptrs = torch.tensor([rotated.data_ptr()], device="cuda", dtype=torch.int64)
    partial = torch.empty((max_rows, splits, n), device="cuda", dtype=torch.float32)
    acc = torch.empty((max_rows, n), device="cuda", dtype=torch.float32)

    inp = compile_trellis_input_aot(features=k, input_dtype="bf16", max_rows=max_rows, output_dtype="bf16")
    core = compile_trellis_linear_aot(
        input_features=k, output_features=n, bits=8, codebook=codebook,
        sharded_side=side, rotation_count=1, max_rows=max_rows, input_dtype="bf16",
    )
    inp.launch(x, suh, rotated, blocks, scalars=(rows, cta_limit))
    core.launch(ptrs, payload, blocks, partial.flatten(), acc.flatten(), scalars=(rows, 0, cta_limit))
    if output_dtype == "raw_f32":
        return acc[:rows].clone()
    out_dt = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    out = torch.empty((max_rows, n), device="cuda", dtype=out_dt)
    eg = compile_trellis_output_aot(features=n, output_dtype=output_dtype, max_rows=max_rows)
    eg.launch(acc, svh, out, scalars=(rows, cta_limit))
    return out[:rows].clone()


@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("output_dtype", ["bf16", "f32", "raw_f32"])
@pytest.mark.parametrize("k,n", [(1152, 256), (2048, 512), (4096, 8192)])
def test_fused_k8_bitwise_exact_rows_cta_and_graph(codebook, side, output_dtype, k, n):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.gemm.trellis_linear._fused_k8 import fused_scratch_bytes
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_fused_aot

    max_rows = 129
    rng = np.random.default_rng(408)
    x = torch.tensor(rng.normal(0, 0.04, (max_rows, k)), device="cuda", dtype=torch.bfloat16)
    suh = torch.tensor(rng.normal(0, 0.25, k), device="cuda", dtype=torch.float16)
    svh = torch.tensor(rng.normal(0, 0.25, n), device="cuda", dtype=torch.float16)
    payload = torch.tensor(rng.integers(0, 256, k * n, dtype=np.uint8), device="cuda")

    prog = compile_trellis_linear_fused_aot(
        input_features=k, output_features=n, bits=8, codebook=codebook,
        sharded_side=side, max_rows=max_rows, input_dtype="bf16", output_dtype=output_dtype,
    )
    out_dt = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    scratch = torch.empty(fused_scratch_bytes(max_rows, k, n), device="cuda", dtype=torch.uint8)
    out = torch.empty((max_rows, n), device="cuda", dtype=out_dt)

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    ref_129 = _run_unfused_oracle(
        x, payload, suh, svh, k=k, n=n, side=side, codebook=codebook,
        output_dtype=output_dtype, max_rows=max_rows, rows=max_rows, cta_limit=sms * 4,
    )

    def run(rows, limit):
        scratch.fill_(0xFF)
        out.fill_(float("nan"))
        if output_dtype == "raw_f32":
            prog.launch(x, payload, suh, scratch, out.flatten(), scalars=(rows, limit))
        else:
            prog.launch(x, payload, suh, svh, scratch, out.flatten(), scalars=(rows, limit))

    with kernel_resolution_guard("fused K8 program precompiled"):
        for rows in (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 129):
            # Poison inactive input tail to verify prefix-row invariance.
            saved = x[rows:].clone() if rows < max_rows else None
            if saved is not None:
                x[rows:].fill_(float("nan"))
            for limit in (1, 48, 170 * 4, 188 * 4):
                run(rows, limit)
                assert _bitwise_equal(out[:rows], ref_129[:rows]), (
                    f"{codebook} side={side} {output_dtype} {k}x{n} rows={rows} limit={limit}"
                )
                if rows < max_rows:
                    assert torch.isnan(out[rows:]).all()
            if saved is not None:
                x[rows:].copy_(saved)

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if output_dtype == "raw_f32":
                prog.launch(x, payload, suh, scratch, out.flatten(), scalars=(16, sms * 4))
            else:
                prog.launch(x, payload, suh, svh, scratch, out.flatten(), scalars=(16, sms * 4))
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            assert _bitwise_equal(out[:16], ref_129[:16])
            assert torch.isnan(out[16:]).all()


@pytest.mark.parametrize("family,name,k,n,side,codebook,output_dtype", FAMILY_GATE_SHAPES)
def test_fused_k8_all_families_bitwise_oracle(family, name, k, n, side, codebook, output_dtype):
    from b12x.gemm.trellis_linear._fused_k8 import fused_scratch_bytes
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_fused_aot

    max_rows = 64
    rng = np.random.default_rng(273)
    x = torch.tensor(rng.normal(0, 0.03, (max_rows, k)), device="cuda", dtype=torch.bfloat16)
    suh = torch.tensor(rng.normal(0, 0.2, k), device="cuda", dtype=torch.float16)
    svh = torch.tensor(rng.normal(0, 0.2, n), device="cuda", dtype=torch.float16)
    payload = torch.tensor(rng.integers(0, 256, k * n, dtype=np.uint8), device="cuda")

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    prog = compile_trellis_linear_fused_aot(
        input_features=k, output_features=n, bits=8, codebook=codebook,
        sharded_side=side, max_rows=max_rows, input_dtype="bf16", output_dtype=output_dtype,
    )
    out_dt = torch.bfloat16 if output_dtype == "bf16" else torch.float32
    scratch = torch.empty(fused_scratch_bytes(max_rows, k, n), device="cuda", dtype=torch.uint8)
    out = torch.empty((max_rows, n), device="cuda", dtype=out_dt)

    ref_64 = _run_unfused_oracle(
        x, payload, suh, svh, k=k, n=n, side=side, codebook=codebook,
        output_dtype=output_dtype, max_rows=max_rows, rows=64, cta_limit=sms * 4,
    )
    for rows in (1, 4, 5, 8, 9, 16, 17, 32, 48, 64):
        out.fill_(float("nan"))
        if output_dtype == "raw_f32":
            prog.launch(x, payload, suh, scratch, out.flatten(), scalars=(rows, sms * 4))
        else:
            prog.launch(x, payload, suh, svh, scratch, out.flatten(), scalars=(rows, sms * 4))
        assert _bitwise_equal(out[:rows], ref_64[:rows]), (
            f"{family}/{name} rows={rows} diverged from unfused 3-stage oracle"
        )
        if rows < max_rows:
            assert torch.isnan(out[rows:]).all()


def test_fused_k8_cpu_only_export_sm120_and_sm121(tmp_path):
    """Verify CPU-only AOT export produces valid headers/objects on SM120 and SM121."""
    import subprocess
    import sys

    code = """
import multiprocessing, sys
from pathlib import Path
import torch
from b12x._lib.compile_pool import _initialize_worker
from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_fused_aot

out_dir = Path(sys.argv[1])
minor = int(sys.argv[2])
sms = 188 if minor == 0 else 48
_initialize_worker(0, (12, minor), "fused-export", "fused-export", sms,
                   101376, 102400, multiprocessing.Array("i", [0, 0]))
for name, k, n, side, cb, odt, expected_args in (
    ("hy4_q_a", 6144, 2048, 1, "mul1", "bf16", 9),
    ("hy4_o_tp2", 8192, 6144, 2, "mul1", "raw_f32", 8),
    ("hy4_lm_head", 6144, 120832, 1, "mul1", "f32", 9),
    ("glmf_kv_a", 4096, 512, 1, "mcg", "bf16", 9),
):
    with exportable_compilation():
        prog = compile_trellis_linear_fused_aot(
            input_features=k, output_features=n, bits=8, codebook=cb,
            sharded_side=side, max_rows=64, input_dtype="bf16", output_dtype=odt,
        )
    stem = f"{name}_sm12{minor}"
    prog.export_to_c(str(out_dir), stem, "cuteafd_" + stem)
    checked = validate_exported_header(prog, out_dir / f"{stem}.h", "cuteafd_" + stem)
    assert checked["argument_count"] == expected_args
    assert prog.geometry["fused_transform"] is True
    assert prog.geometry["decode_route_thresholds"] == []
assert not torch.cuda.is_initialized()
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for minor in (0, 1):
        sub = tmp_path / f"sm12{minor}"
        sub.mkdir()
        subprocess.check_call([sys.executable, "-c", code, str(sub), str(minor)], env=env)
