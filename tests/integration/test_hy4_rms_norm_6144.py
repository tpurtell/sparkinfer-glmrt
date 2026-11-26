"""Oracle, row-invariance, and CPU-only SM120/SM121 AOT export tests for Hy4 width=6144 BF16 RMSNorm."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch


def _rms_norm_ref_f64(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.double()
    inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * weight.double()).to(torch.bfloat16)


def test_hy4_rms_norm_6144_oracle_row_invariance_and_graph():
    from b12x.integration.cuteafd.pointwise import AffineNorm, compile_affine_norm_aot
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    torch.manual_seed(273)
    max_rows = 256
    width = 6144
    x = torch.randn((max_rows, width), device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.5 + 1.0
    dummy_bf16 = torch.empty(16, device="cuda", dtype=torch.bfloat16)
    dummy_i64 = torch.zeros(max_rows, device="cuda", dtype=torch.int64)
    out = torch.empty_like(x)
    out_legacy = torch.empty_like(x)

    prog = compile_affine_norm_aot(
        width=width, heads=1, eps=1e-5,
        input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16,
    )
    legacy = compile_program(
        AffineNorm(width, 1, 1e-5, False, False, False, 0),
        name="affine_norm_legacy_6144",
        operands=(
            Operand("x", torch.bfloat16, f"[rows,1,{width}]"),
            Operand("weight", torch.bfloat16, f"[{width}]"),
            Operand("bias", torch.bfloat16, f"[{width}] (ignored)"),
            Operand("cos", torch.bfloat16, "[positions,0] (ignored)"),
            Operand("sin", torch.bfloat16, "[positions,0] (ignored)"),
            Operand("positions", torch.int64, "[rows] (ignored)"),
            Operand("out", torch.bfloat16, f"[rows,1,{width}]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=("legacy", width),
        geometry=dict(width=width, heads=1, eps=1e-5),
    )

    prog.launch(x, w, dummy_bf16, dummy_bf16, dummy_bf16, dummy_i64, out, scalars=(max_rows,))
    legacy.launch(x, w, dummy_bf16, dummy_bf16, dummy_bf16, dummy_i64, out_legacy, scalars=(max_rows,))
    ref = _rms_norm_ref_f64(x, w)

    # Numerically within 1 BF16 ULP of F64 oracle (and <= legacy 1-warp error).
    err_new = (out.float() - ref.float()).abs().max().item()
    err_leg = (out_legacy.float() - ref.float()).abs().max().item()
    assert err_new <= max(err_leg, 1e-2) + 1e-5, (err_new, err_leg)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    # Bit-exact row invariance across decode (<=64) and prefill (>64) routes.
    full = out.clone()
    for rows in (1, 2, 4, 8, 16, 32, 64, 65, 128, 256):
        out.fill_(float("nan"))
        prog.launch(x, w, dummy_bf16, dummy_bf16, dummy_bf16, dummy_i64, out, scalars=(rows,))
        assert torch.equal(out[:rows].view(torch.int16), full[:rows].view(torch.int16)), f"rows={rows}"
        if rows < max_rows:
            assert torch.isnan(out[rows:]).all()

    # CUDA graph capture + replay
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        prog.launch(x, w, dummy_bf16, dummy_bf16, dummy_bf16, dummy_i64, out, scalars=(16,))
    for _ in range(3):
        out.fill_(float("nan"))
        g.replay()
        assert torch.equal(out[:16].view(torch.int16), full[:16].view(torch.int16))
        assert torch.isnan(out[16:]).all()


def test_hy4_rms_norm_6144_cpu_only_export_sm120_and_sm121(tmp_path):
    """Verify exact AffineNorm ABI is preserved on SM120 and SM121 with zero CUDA init."""
    code = """
import multiprocessing, sys
from pathlib import Path
import torch
from b12x._lib.compile_pool import _initialize_worker
from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot

out_dir = Path(sys.argv[1])
minor = int(sys.argv[2])
sms = 188 if minor == 0 else 48
_initialize_worker(0, (12, minor), "rms6144", "rms6144", sms,
                   101376, 102400, multiprocessing.Array("i", [0, 0]))
with exportable_compilation():
    prog = compile_affine_norm_aot(
        width=6144, heads=1, eps=1e-5,
        input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16,
    )
stem = f"rms_norm_6144_sm12{minor}"
prog.export_to_c(str(out_dir), stem, "cuteafd_" + stem)
checked = validate_exported_header(prog, out_dir / f"{stem}.h", "cuteafd_" + stem)
assert checked["argument_count"] == 9
assert not torch.cuda.is_initialized()
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for minor in (0, 1):
        sub = tmp_path / f"sm12{minor}"
        sub.mkdir()
        subprocess.check_call([sys.executable, "-c", code, str(sub), str(minor)], env=env)
