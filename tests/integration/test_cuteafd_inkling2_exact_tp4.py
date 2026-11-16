"""Real Inkling TP4 K16/K32 width-576 exported objects versus ordinary routes."""
from dataclasses import dataclass, fields

import pytest
import torch

from ..conftest import require_b12x
from tests.moe.segment_aot import ExportedProgram


def _exact_geometry(name):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    base = GEOMETRIES[name].with_tp(4)

    @dataclass(frozen=True)
    class ExactSlice(type(base)):
        @property
        def slice(self):
            return 576

    return ExactSlice(**{field.name: getattr(base, field.name) for field in fields(base)})


def _weights(g):
    from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes

    operands = []
    for rows, columns in [(g.slice, g.hidden), (g.slice, g.hidden), (g.hidden, g.slice)]:
        if g.weights == "mxfp8":
            weights = torch.randint(-8, 9, (g.experts, rows, columns), device="cuda").to(torch.float8_e4m3fn)
            pitch = mxfp4_scale_row_bytes(columns)
            scales = torch.full((g.experts, rows, pitch), 120, dtype=torch.uint8, device="cuda")
        else:
            weights = torch.randint(0, 256, (g.experts, rows, columns // 2), dtype=torch.uint8, device="cuda")
            scales = torch.full((g.experts * rows * columns // 16,), 0.125,
                dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
            tail = torch.cat([torch.full((g.experts,), 16.0, device="cuda"),
                torch.tensor([2688.0], device="cuda")]).view(torch.uint8)
            scales = torch.cat([scales, tail])
        operands.extend([weights.contiguous(), scales.contiguous()])
    return operands


@pytest.mark.parametrize("name", ["inkling2_mxfp8", "inkling2_nvfp4"])
def test_tp4_w576_exported_object_matches_ordinary(name, tmp_path):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes
    from .test_cuteafd_inkling2_mxfp8_moe import _mxfp8

    torch.manual_seed(576384)
    g = _exact_geometry(name)
    assert (g.hidden, g.experts, g.top_k, g.intermediate, g.slice) == (4608, 384, 8, 2304, 576)
    capacity = 4096
    wire = g.weights == "mxfp8"
    weights = _weights(g)
    x = torch.randn(capacity, g.hidden, device="cuda", dtype=torch.bfloat16)
    if wire:
        codes, scales, _ = _mxfp8(x)
        x = torch.cat([codes.view(torch.uint8), scales], -1).contiguous()
    ids = torch.rand(capacity, g.experts, device="cuda").topk(g.top_k, -1).indices.int().contiguous()
    gamma = torch.rand(capacity, g.top_k, device="cuda").contiguous()
    ordinary = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=wire)
    with exportable_compilation():
        program = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=wire)
    exported = ExportedProgram(program, tmp_path, name + "_w576")
    scratch_bytes = fp8_moe_scratch_bytes(g, "auto", capacity, wire)
    scratch = [torch.empty(scratch_bytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
    outputs = [torch.full((capacity + 2, g.hidden), 11, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    for rows in [1, 7, 80, 2752, 4096]:
        for route, output, workspace in zip([ordinary, exported], outputs, scratch):
            route.launch(x, ids, gamma, *weights, output[1:-1], workspace, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(outputs[0][1:rows + 1], outputs[1][1:rows + 1]), (name, rows)
        assert torch.isfinite(outputs[1][1:rows + 1]).all()
        assert outputs[1][1:rows + 1].abs().max() > 0
        assert all(torch.all(output[[0, -1]] == 11) for output in outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exported.launch(x, ids, gamma, *weights, outputs[1][1:-1], scratch[1], scalars=(80,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outputs[0][1:81], outputs[1][1:81])
    print(f"{name} w576: exported object exact at rows 1,7,80,2752,4096 and graph replay")
