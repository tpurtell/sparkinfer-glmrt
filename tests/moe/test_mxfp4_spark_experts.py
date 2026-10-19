"""Bitwise correctness, 1-64 row invariance, tail, graph-replay, and cubin resource gates
for MiMo V2.6 Pro MXFP4 Spark routed experts against origin/master."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest
import torch

from benchmarks import benchmark_mxfp4_spark_experts as bench
from ..conftest import require_b12x


def _control_dir() -> Path:
    env = os.environ.get("CUTEAFD_MXFP4_CONTROL_DIR")
    if env:
        return Path(env)
    default = Path("/workspace/cuteafd/results/control")
    if default.is_dir():
        return default
    pytest.skip("CUTEAFD_MXFP4_CONTROL_DIR not set and /workspace/cuteafd/results/control not found")


def test_bench_capacity_mapping_and_wire_encoding():
    assert [bench.capacity_for_rows(r) for r in (1, 4, 8, 16, 43, 64, 128)] == [1, 16, 16, 16, 80, 80, 256]
    gen = torch.Generator().manual_seed(42)
    x = (torch.randn((7, 6144), generator=gen) * 2.5).bfloat16()
    wire, exact = bench.make_wire(x)
    assert wire.shape == (7, 6144 + 192)
    values = wire[:, :6144].view(torch.float8_e4m3fn).float().reshape(7, 192, 32)
    scales = torch.exp2(wire[:, 6144:].float() - 127.0).unsqueeze(-1)
    assert torch.equal((values * scales).reshape(7, 6144).bfloat16(), exact)


@pytest.mark.parametrize("capacity,rows_list", [
    (1, (1,)),
    (16, (1, 4, 7, 8, 9, 15, 16)),
    (80, (1, 4, 8, 16, 17, 32, 43, 64, 80)),
    (256, (1, 8, 16, 43, 64, 128, 256)),
])
@pytest.mark.parametrize("hot", [0, 4, 8])
@pytest.mark.parametrize("wire", [True, False])
def test_bitwise_identical_to_master_across_rows_packings_and_capacities(capacity, rows_list, hot, wire):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    ctrl_mod, _ = bench.load_control_modules(_control_dir())
    g = GEOMETRIES["mimop"].with_tp(6)
    ctrl_g = ctrl_mod.GEOMETRIES["mimop"].with_tp(6)

    gen = torch.Generator(device="cuda").manual_seed(300 + capacity + 10 * hot + int(wire))
    h, i, e = g.hidden, g.slice, g.experts
    real = 352
    w1 = torch.randint(0, 256, (e, i, h // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    s1 = torch.randint(118, 124, (e, i, h // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    w3 = torch.randint(0, 256, (e, i, h // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    s3 = torch.randint(118, 124, (e, i, h // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    s2 = torch.randint(118, 124, (e, h, i // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    for w in (w1, s1, w3, s3):
        w[:, real:] = 0
    w2[:, :, real // 2:] = 0
    s2[:, :, real // 32:] = 0
    weights_tuple = (w1, s1, w3, s3, w2, s2)

    cand_prog = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=wire)
    ctrl_prog = ctrl_mod.compile_fp8_moe_aot(ctrl_g, route="auto", max_rows=capacity, wire=wire)
    cand_scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", capacity, wire), dtype=torch.uint8, device="cuda")
    ctrl_scratch = torch.empty(ctrl_mod.fp8_moe_scratch_bytes(ctrl_g, "auto", capacity, wire),
                               dtype=torch.uint8, device="cuda")

    x_full = torch.randn((capacity, h), device="cuda", generator=gen).bfloat16()
    src_full = bench.make_wire(x_full)[0] if wire else x_full
    scores = torch.rand((capacity, e), device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 4.0
    ids_full = scores.topk(g.top_k, -1).indices.int().contiguous()
    if capacity > 4 and hot == 0:
        ids_full[-1, -1] = -1  # also exercise invalid route skipping
    rw_full = torch.rand((capacity, g.top_k), device="cuda", generator=gen).contiguous()
    rw_full = (rw_full / rw_full.sum(-1, keepdim=True)).contiguous()

    cand_out = torch.empty((capacity, h), dtype=torch.bfloat16, device="cuda")
    ctrl_out = torch.empty((capacity, h), dtype=torch.bfloat16, device="cuda")
    cand_prog.launch(src_full, ids_full, rw_full, *weights_tuple, cand_out, cand_scratch, scalars=(1,))
    ctrl_prog.launch(src_full, ids_full, rw_full, *weights_tuple, ctrl_out, ctrl_scratch, scalars=(1,))
    torch.cuda.synchronize()

    with kernel_resolution_guard("MXFP4 bitwise identity and graph replay"):
        for rows in rows_list:
            cand_scratch.fill_(0xFF)
            ctrl_scratch.fill_(0xFF)
            cand_out.fill_(float("nan"))
            ctrl_out.fill_(float("nan"))
            cand_prog.launch(src_full, ids_full, rw_full, *weights_tuple, cand_out, cand_scratch, scalars=(rows,))
            ctrl_prog.launch(src_full, ids_full, rw_full, *weights_tuple, ctrl_out, ctrl_scratch, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.isfinite(cand_out[:rows]).all() and torch.count_nonzero(cand_out[:rows]) > 0
            assert torch.equal(cand_out[:rows].view(torch.int16), ctrl_out[:rows].view(torch.int16)), (
                f"Mismatch vs master at cap={capacity}, rows={rows}, hot={hot}, wire={wire}"
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cand_prog.launch(src_full, ids_full, rw_full, *weights_tuple, cand_out, cand_scratch, scalars=(rows_list[-1],))
        cand_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cand_out[:rows_list[-1]].view(torch.int16), ctrl_out[:rows_list[-1]].view(torch.int16))


@pytest.mark.parametrize("wire", [True, False])
@pytest.mark.parametrize("same_experts", [False, True])
def test_mxfp4_row_invariance_1_to_64(wire, same_experts):
    """Bitwise row invariance across every row count 1..64."""
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = GEOMETRIES["mimop"].with_tp(6)
    gen = torch.Generator(device="cuda").manual_seed(707 + int(wire) * 10 + int(same_experts))
    h, i, e = g.hidden, g.slice, g.experts
    w1 = torch.randint(0, 256, (e, i, h // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    s1 = torch.randint(118, 124, (e, i, h // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    w3 = torch.randint(0, 256, (e, i, h // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    s3 = torch.randint(118, 124, (e, i, h // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    s2 = torch.randint(118, 124, (e, h, i // 32), device="cuda", generator=gen, dtype=torch.int32).to(torch.uint8)
    weights_tuple = (w1, s1, w3, s3, w2, s2)

    x = torch.randn((64, h), device="cuda", generator=gen).bfloat16()
    source = bench.make_wire(x)[0] if wire else x
    ids = torch.rand((64, e), device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    if same_experts:
        ids[:] = ids[0].clone()
    scores = torch.rand((64, g.top_k), device="cuda", generator=gen)
    scores = (scores / scores.sum(1, keepdim=True)).contiguous()

    progs = {cap: compile_fp8_moe_aot(g, route="auto", max_rows=cap, wire=wire) for cap in (1, 16, 80)}
    scratches = {cap: torch.empty(fp8_moe_scratch_bytes(g, "auto", cap, wire), dtype=torch.uint8, device="cuda")
                 for cap in progs}

    def run(first, count, cap):
        out = torch.empty((count, h), dtype=torch.bfloat16, device="cuda")
        progs[cap].launch(source[first:first + count], ids[first:first + count],
                          scores[first:first + count], *weights_tuple, out, scratches[cap], scalars=(count,))
        return out

    serial = torch.cat([run(r, 1, 1) for r in range(64)], dim=0)
    torch.cuda.synchronize()
    for count in range(1, 65):
        cap = 16 if count <= 16 else 80
        got = run(0, count, cap)
        torch.cuda.synchronize()
        diff = int((got.view(torch.int16) != serial[:count].view(torch.int16)).sum())
        assert diff == 0, f"Row invariance failed at M={count} (cap={cap}, wire={wire}, same={same_experts}): {diff} diffs"


@pytest.mark.parametrize("width", [32, 64, 96, 288, 320, 352])
def test_exact_tail_bitwise_identical_to_master(width):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    ctrl_mod, _ = bench.load_control_modules(_control_dir())

    @dataclass(frozen=True)
    class Exact(Fp8MoeGeometry):
        width: int = 352

        @property
        def slice(self):
            return self.width

    @dataclass(frozen=True)
    class CtrlExact(ctrl_mod.Fp8MoeGeometry):
        width: int = 352

        @property
        def slice(self):
            return self.width

    g = Exact("tail", hidden=6144, experts=16, top_k=8, intermediate=2048, tp=6, weights="mxfp4", width=width)
    ctrl_g = CtrlExact("tail", hidden=6144, experts=16, top_k=8, intermediate=2048, tp=6, weights="mxfp4", width=width)

    gen = torch.Generator(device="cuda").manual_seed(97 + width)
    capacity = 80
    source, _ = bench.make_wire(torch.randn((capacity, g.hidden), device="cuda", generator=gen).bfloat16())
    ids = torch.rand((capacity, g.experts), device="cuda", generator=gen).topk(g.top_k, -1).indices.int().contiguous()
    weights = torch.rand((capacity, g.top_k), device="cuda", generator=gen).contiguous()
    e, h = g.experts, g.hidden
    w1 = torch.randint(0, 256, (e, width, h // 2), device="cuda", generator=gen).byte()
    s1 = torch.randint(118, 124, (e, width, h // 32), device="cuda", generator=gen).byte()
    w3 = torch.randint(0, 256, (e, width, h // 2), device="cuda", generator=gen).byte()
    s3 = torch.randint(118, 124, (e, width, h // 32), device="cuda", generator=gen).byte()
    w2 = torch.randint(0, 256, (e, h, width // 2), device="cuda", generator=gen).byte()
    stride = (width // 32 + 3) // 4 * 4
    s2 = torch.full((e, h, stride), 255, device="cuda", dtype=torch.uint8)
    s2[..., :width // 32] = torch.randint(118, 124, (e, h, width // 32), device="cuda", generator=gen).byte()
    packed = (w1, s1, w3, s3, w2, s2)

    cand = compile_fp8_moe_aot(g, route="decode", max_rows=capacity)
    ctrl = ctrl_mod.compile_fp8_moe_aot(ctrl_g, route="decode", max_rows=capacity)
    cand_out = torch.empty((capacity, h), device="cuda", dtype=torch.bfloat16)
    ctrl_out = torch.empty((capacity, h), device="cuda", dtype=torch.bfloat16)
    cand_scratch = torch.empty(fp8_moe_scratch_bytes(g, "decode", capacity), device="cuda", dtype=torch.uint8)
    ctrl_scratch = torch.empty(ctrl_mod.fp8_moe_scratch_bytes(ctrl_g, "decode", capacity), device="cuda", dtype=torch.uint8)

    for rows in (1, 17, 64, 80):
        cand_out.fill_(float("nan"))
        ctrl_out.fill_(float("nan"))
        cand.launch(source, ids, weights, *packed, cand_out, cand_scratch, scalars=(rows,))
        ctrl.launch(source, ids, weights, *packed, ctrl_out, ctrl_scratch, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(cand_out[:rows].view(torch.int16), ctrl_out[:rows].view(torch.int16))
