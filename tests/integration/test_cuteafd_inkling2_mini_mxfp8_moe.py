"""cuteafd Inkling 2 Mini MXFP8 W8A8 routed experts oracle (layers 20-23, TP1).

Exercises ``GEOMETRIES["inkling2_mini_mxfp8"]`` (hidden=2304, intermediate=1152,
experts=128, top_k=8, SiTU 4/25) across decode tails and the prefill live-row band.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from . import test_cuteafd_inkling2_mxfp8_moe as small_mxfp8


def _geometry():
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["inkling2_mini_mxfp8"].with_tp(1)


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    (1, 1, 0),
    (16, 16, 0),
    (80, 80, 0),
    (1024, 1000, 0),
    (4096, 2048, 0),
    (4096, 2752, 0),
    (4096, 4096, 8),
]


@pytest.mark.parametrize("capacity,rows,hot", CASES)
def test_inkling2_mini_mxfp8_moe(capacity, rows, hot):
    g = _geometry()
    c, worst, delta, scale = small_mxfp8._run(g, g.intermediate, capacity, rows, hot=hot)
    print(f"inkling2_mini_mxfp8 tp1 capacity {capacity} rows {rows} hot {hot}: "
          f"cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e} (max|ref| {scale:.3e})")
    assert c >= small_mxfp8.COS and worst >= small_mxfp8.ROW_COS


def test_inkling2_mini_mxfp8_route_weight_before_fc2():
    g = _geometry()
    c, worst, delta, _ = small_mxfp8._run(g, g.intermediate, 256, 256, seed=3, gamma_spread=8.0)
    print(f"inkling2_mini_mxfp8 gamma spread: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= small_mxfp8.COS and worst >= small_mxfp8.ROW_COS


def test_inkling2_mini_mxfp8_situ_saturation():
    g = _geometry()
    c, worst, delta, _ = small_mxfp8._run(g, g.intermediate, 256, 256, seed=5, saturate=True)
    print(f"inkling2_mini_mxfp8 saturation: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= small_mxfp8.COS and worst >= small_mxfp8.ROW_COS


def test_inkling2_mini_mxfp8_geometry():
    g = _geometry()
    assert (g.hidden, g.intermediate, g.slice, g.experts, g.top_k) == (2304, 1152, 1152, 128, 8)
    assert g.kind == "mxfp8" and g.activation == "situ"
