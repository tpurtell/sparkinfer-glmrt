"""Bitwise MXFP4 decode contract against an explicitly supplied baseline core."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize("routing", ["spread", "hot", "masked"])
def test_decode_master_rows_packing_graph(routing):
    require_b12x()
    control = os.environ.get("B12X_MXFP4_CONTROL_CORE")
    if not control:
        pytest.skip("B12X_MXFP4_CONTROL_CORE must name the immutable master snapshot")
    from b12x.integration.cuteafd import fp8_moe
    from b12x.integration.cuteafd._mxfp4_moe_kernels import GroupedMxfp4Gemv
    from .test_cuteafd_fp8_moe_aot import wire_rows

    name = "b12x.integration.cuteafd._mxfp4_exact_control"
    spec = importlib.util.spec_from_file_location(name, Path(control))
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)

    class Control(baseline.GroupedMxfp4Gemv):
        def key(self):
            return ("exact_control", super().key())

    g = fp8_moe.Fp8MoeGeometry("exact_mxfp4", hidden=6144, experts=16, top_k=8,
                              intermediate=384, weights="mxfp4")
    gen = torch.Generator(device="cuda").manual_seed(500)
    x = torch.randn(128, g.hidden, device="cuda", generator=gen).bfloat16()
    source, _ = wire_rows(x)
    ids = torch.rand(128, g.experts, device="cuda", generator=gen).topk(g.top_k, -1).indices.int()
    if routing == "hot":
        ids[:] = ids[0].clone()
    elif routing == "masked":
        ids[:, 0] = -1
    mix = torch.rand(128, g.top_k, device="cuda", generator=gen)
    weights = []
    for shape in ((16, 384, 3072), (16, 384, 192), (16, 384, 3072),
                  (16, 384, 192), (16, 6144, 192), (16, 6144, 12)):
        scales = len(weights) % 2
        weights.append(torch.randint(118 if scales else 0, 124 if scales else 256, shape,
                                     device="cuda", dtype=torch.uint8, generator=gen))
    programs = {}
    for arm, core in (("master", Control), ("candidate", GroupedMxfp4Gemv)):
        try:
            fp8_moe.GroupedMxfp4Gemv = core
            for capacity in (1, 16, 64, 128):
                programs[arm, capacity] = fp8_moe.compile_fp8_moe_aot(g, route="decode", max_rows=capacity)
        finally:
            fp8_moe.GroupedMxfp4Gemv = GroupedMxfp4Gemv
    scratch = torch.empty(fp8_moe.fp8_moe_scratch_bytes(g, "decode", 128), device="cuda", dtype=torch.uint8)
    out = torch.empty_like(x)

    def execute(arm, capacity, indices, graph=False):
        rows = len(indices)
        a, selected, factors = source[indices].contiguous(), ids[indices].contiguous(), mix[indices].contiguous()
        scratch.fill_(0xA5)
        out.fill_(float("nan"))

        def launch():
            programs[arm, capacity].launch(a, selected, factors, *weights, out, scratch, scalars=(rows,))

        launch()
        expected = out[:rows].clone()
        if graph:
            replay = torch.cuda.CUDAGraph()
            with torch.cuda.graph(replay):
                launch()
            for _ in range(3):
                out.fill_(float("nan"))
                replay.replay()
                assert torch.equal(out[:rows].view(torch.uint8), expected.view(torch.uint8))
        assert torch.isfinite(out[:rows]).all() and out[:rows].count_nonzero()
        return out[:rows].clone().view(torch.uint8)

    singles = torch.cat([execute("master", 1, [row]) for row in range(128)])
    for rows in (*range(1, 65), 128):
        capacity = 1 if rows == 1 else 16 if rows <= 16 else 64 if rows <= 64 else 128
        indices = torch.arange(rows, device="cuda")
        for arm in ("master", "candidate"):
            actual = execute(arm, capacity, indices, graph=rows in (1, 16, 43, 64, 128))
            assert torch.equal(actual, singles[:rows]), (arm, rows, "row invariance")
            reverse = indices.flip(0)
            actual = execute(arm, 128, reverse)
            assert torch.equal(actual, singles[:rows].flip(0)), (arm, rows, "packing/capacity invariance")
