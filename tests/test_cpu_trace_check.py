"""CPU-only trace check (b12x.tools.cpu_trace_check): runs with no GPU.

The check must lower a real routed mixed-Trellis program with no device and
stop at a seeded CuTe DSL trace-time type error with the DSL's own diagnostic,
the failure that voided duel entries on hardware (#2571).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("cutlass")

ROOT = Path(__file__).resolve().parents[1]
# Smallest real routed shape: Qwen-like H2560/I256 would do too; H1536 I256 at
# m1 is a 3-block FC1 and traces in seconds on one core.
SHAPE = ("hidden=1536,intermediate=256,experts=512,topk=16,capacity=1,bits=4/5,"
         "codebook=mul1,tile=64/256/64/256,output_dtype=bf16")
SEED_ANCHOR = """        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(
            self.coupled_hadamard and self.broadcast_svh
        ):"""
SEED = """        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        seeded_policy = None
        if Int32(bidx) < active_m:
            seeded_policy = create_l2_evict_first_policy()
        if cutlass.const_expr(
            self.coupled_hadamard and self.broadcast_svh
        ):"""


def _run(tree: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
           "PYTHONPATH": str(tree), "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "b12x.tools.cpu_trace_check", *args],
                          cwd=tree, env=env, capture_output=True, text=True, timeout=900)


def _tree_copy(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    shutil.copytree(ROOT / "b12x", tree / "b12x",
                    ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    return tree


def test_stock_shape_lowers_on_cpu_for_both_arches(tmp_path):
    result = _run(ROOT, tmp_path, "--shape", SHAPE)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "cpu-trace-check sm120: PASSED (0 failing of 1 program(s)" in output
    assert "cpu-trace-check sm121: PASSED (0 failing of 1 program(s)" in output


def test_seeded_type_unstable_join_is_caught(tmp_path):
    tree = _tree_copy(tmp_path)
    kernel = tree / "b12x/moe/_shared/kernels/w4a16/kernel.py"
    source = kernel.read_text()
    assert source.count(SEED_ANCHOR) == 1, "seed anchor moved; update the test"
    kernel.write_text(source.replace(SEED_ANCHOR, SEED))
    seeded_line = source[: source.index(SEED_ANCHOR)].count("\n") + 4

    result = _run(tree, tmp_path, "--arch", "sm120", "--shape", SHAPE)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "TYPE_UNSTABLE_JOIN" in output, output
    assert "`seeded_policy` has type `None` on one path" in output, output
    assert f"kernel.py:{seeded_line}" in output, output
    assert "cpu-trace-check sm120: FAILED (1 failing of 1 program(s)" in output


def test_package_manifests_and_shape_lists_round_trip(tmp_path):
    variant = tmp_path / "pkg/exl3-x/rtx-tp1/m1"
    variant.mkdir(parents=True)
    (variant / "v41_exl3.json").write_text(json.dumps({
        "compute": [12, 0], "hidden": 1536, "intermediate": 256, "experts": 512, "top_k": 16,
        "capacity": 1, "bits": [4, 5], "codebook": "mul1", "output_dtype": "fp32",
        "tile": [64, 256, 64, 256], "blocks_per_sm": 1, "swiglu_limit": 10.0, "direct": False}))
    (variant / "routes").mkdir()
    (variant / "routes/v41_exl3.json").write_text("{}")
    (tmp_path / "pkg/exl3-x/manifest.json").write_text("{}")
    shapes = tmp_path / "shapes.json"
    # One package directory or a root of packages label variants alike.
    for root in (tmp_path / "pkg", tmp_path / "pkg/exl3-x"):
        listed = _run(ROOT, tmp_path, "--package", str(root), "--match", "exl3-x/", "--list")
        assert listed.returncode == 0, listed.stdout + listed.stderr
        assert "(exl3-x/rtx-tp1/m1)" in listed.stdout
    saved = _run(ROOT, tmp_path, "--package", str(tmp_path / "pkg"), "--save-shapes", str(shapes))
    assert saved.returncode == 0, saved.stdout + saved.stderr
    listed = _run(ROOT, tmp_path, "--shapes", str(shapes), "--list")
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert listed.stdout.strip().splitlines() == [
        "sm120 H1536 I256 E512 k16 m1 K45 mul1 tile64/256/64/256 bps1  (exl3-x/rtx-tp1/m1)"]
    narrowed = _run(ROOT, tmp_path, "--shapes", str(shapes), "--arch", "sm121", "--list")
    assert narrowed.returncode == 2 and "matched no programs" in narrowed.stderr
