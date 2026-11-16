"""Real Inkling TP4 K16/K32 width-576 exported objects versus ordinary routes."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from ..conftest import require_b12x
from tests.moe.segment_aot import ExportedProgram

try:
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Int32
except ImportError:
    cute = None


def _exact_geometry(name):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    root = Path(os.environ['CUTEAFD_SOURCE_ROOT'])
    sys.path.insert(0, str(root / 'python/tools/lib'))
    spec = importlib.util.spec_from_file_location('inkling_tp4_packager',
        root / 'python/tools/aot/package_fp8_moe_aot.py')
    packager = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = packager
    spec.loader.exec_module(packager)
    tp, geometry = packager.layout_geometry(GEOMETRIES[name], 'tp4-w576')
    assert tp == 4
    return geometry


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


@pytest.mark.parametrize('k', [16, 48, 64, 112, 384, 448, 464, 560, 576, 592, 640, 768, 1040])
@pytest.mark.parametrize('n', [80, 128])
def test_dynamic_down_native_k64_matches_exact_grid(k, n, tmp_path):
    require_b12x()
    from b12x.integration.cuteafd._nvfp4_moe_dynamic import StreamNvfp4LinearDyn
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd import exportable_compilation

    experts, rows = 1, 128
    linear = StreamNvfp4LinearDyn(n=n, k=k, experts=experts)

    class Down:
        def key(self):
            return ('inkling_native_down_oracle', linear.key())

        @cute.jit
        def __call__(self, aq: cute.Pointer, scales: cute.Pointer, meta: cute.Pointer,
                     weight: cute.Pointer, ws: cute.Pointer, output: cute.Pointer,
                     amax: cute.Pointer, live: Int32, stream: cuda.CUstream):
            linear(aq, scales, meta, weight, ws, output, Int32(1), stream, row_amax=amax)

    ops = (Operand('aq', torch.uint8, '[128,K/2]'), Operand('scales', torch.uint8, '[128,K/16]'),
        Operand('meta', torch.int32, '[8]', align=4), Operand('weight', torch.uint8, '[N,K/2]'),
        Operand('ws', torch.uint8, '[S]', align=4), Operand('output', torch.bfloat16, '[128,N]', 'out'),
        Operand('amax', torch.float32, '[128]', align=4))
    with exportable_compilation():
        program = compile_program(Down(), name=f'inkling_down_k{k}', operands=ops,
            scalars=(Scalar('live'),), key=('inkling_native_down_oracle', linear.key()))
    exported = ExportedProgram(program, tmp_path, f'inkling_down_k{k}')
    torch.manual_seed(k)
    aq = torch.randint(0, 256, (rows, k // 2), device='cuda', dtype=torch.uint8)
    weight = torch.randint(0, 256, (n, k // 2), device='cuda', dtype=torch.uint8)
    scales = torch.ones(rows, k // 16, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
    ws = torch.cat([torch.ones(n * k // 16, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8),
        torch.tensor([1536.0, 1.0], device='cuda').view(torch.uint8)])
    amax = torch.ones(rows, device='cuda')
    meta = torch.tensor([0, 1, 0, 0, rows, 0, 0, 0], device='cuda', dtype=torch.int32)
    output = torch.empty(rows, n, device='cuda', dtype=torch.bfloat16)
    exported.launch(aq, scales, meta, weight, ws, output, amax, scalars=(rows,))
    torch.cuda.synchronize()
    table = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], device='cuda')
    def decode(packed):
        return table[torch.stack([packed.long() & 15, packed.long() >> 4], -1).flatten(-2)]
    expected = (decode(aq).double() @ decode(weight).double().T).float().bfloat16()
    assert torch.equal(output, expected), (k, (output.float() - expected.float()).abs().max().item())
    print(f'NVFP4 dynamic down K{k}: exact integer-grid F64 reference, stored K{k}')
