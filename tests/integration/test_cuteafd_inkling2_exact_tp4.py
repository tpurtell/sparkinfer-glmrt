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


@pytest.mark.parametrize('k', [16, 48, 64, 96, 112, 160, 208, 224, 320, 336, 352, 384, 400, 416, 432, 448, 464, 560, 576, 592, 608, 624, 640, 672, 688, 768, 1040])
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


# Native shard widths, including the rotated remainder ranks. Keep the family's
# real hidden/expert/top-k geometry; only the stored rank width changes.
FAMILY_WIDTHS = [
    ('inkling2_nvfp4', 3, 768), ('inkling2_nvfp4', 4, 576),
    ('inkling2_nvfp4', 5, 448), ('inkling2_nvfp4', 5, 464),
    ('inkling2_nvfp4', 6, 384),
    ('glmf_nvfp4', 3, 688), ('glmf_nvfp4', 3, 672),
    ('glmf_nvfp4', 5, 416), ('glmf_nvfp4', 5, 400),
    ('glmf_nvfp4', 6, 352), ('glmf_nvfp4', 6, 336),
    ('glm_nvfp4', 3, 688), ('glm_nvfp4', 3, 672),
    ('qwen4_nvfp4', 2, 320), ('qwen4_nvfp4', 3, 224),
    ('qwen4_nvfp4', 3, 208), ('qwen4_nvfp4', 4, 160),
    ('qwen4_nvfp4', 6, 112), ('qwen4_nvfp4', 6, 96),
    ('minimax_nvfp4a4', 2, 1536), ('minimax_nvfp4a4', 3, 1024),
    ('minimax_nvfp4a4', 4, 768), ('minimax_nvfp4a4', 6, 512),
    ('minimax_nvfp4a4', 5, 624), ('minimax_nvfp4a4', 5, 608),
    ('minimax_nvfp4a4', 7, 448), ('minimax_nvfp4a4', 7, 432),
    ('inkling2_nvfp4', 4, 560), ('inkling2_nvfp4', 4, 592),
    ('inkling2_nvfp4', 4, 1040),
]


def _native_geometry(name, tp, width):
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, GEOMETRIES

    class NativeWidth(Fp8MoeGeometry):
        @property
        def slice(self):
            return width

    return NativeWidth(**{**vars(GEOMETRIES[name]), 'tp': tp})


@pytest.mark.parametrize('name,tp,width', FAMILY_WIDTHS)
def test_native_family_exported_route(name, tp, width, tmp_path):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = _native_geometry(name, tp, width)
    torch.manual_seed(142)
    if g.dynamic_nvfp4:
        w = _weights(g)
    else:
        from .test_cuteafd_nvfp4_moe_aot import _weights as static_weights
        w = static_weights(g, width)
    capacity = 4096
    x = torch.randn(capacity, g.hidden, device='cuda', dtype=torch.bfloat16)
    ids = torch.rand(capacity, g.experts, device='cuda').topk(g.top_k, -1).indices.int().contiguous()
    gamma = torch.rand(capacity, g.top_k, device='cuda')
    ordinary = compile_fp8_moe_aot(g, route='auto', max_rows=capacity, wire=False)
    with exportable_compilation():
        program = compile_fp8_moe_aot(g, route='auto', max_rows=capacity, wire=False)
    exported = ExportedProgram(program, tmp_path, f'{name}_w{width}')
    scratch_bytes = fp8_moe_scratch_bytes(g, 'auto', capacity, False)
    scratch = [torch.empty(scratch_bytes, dtype=torch.uint8, device='cuda') for _ in range(2)]
    outputs = [torch.full((capacity + 2, g.hidden), 11, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    for rows in [1, 7, 80, 2752, 4096]:
        for route, output, workspace in zip([ordinary, exported], outputs, scratch):
            route.launch(x, ids, gamma, *w, output[1:-1], workspace, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(outputs[0][1:rows + 1], outputs[1][1:rows + 1]), (name, width, rows)
        assert torch.isfinite(outputs[1][1:rows + 1]).all()
        assert outputs[1][1:rows + 1].abs().max() > 0
        assert all(torch.all(output[[0, -1]] == 11) for output in outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exported.launch(x, ids, gamma, *w, outputs[1][1:-1], scratch[1], scalars=(80,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outputs[0][1:81], outputs[1][1:81])
    print(f'{name} native width {width}: exported exact rows 1/7/80/2752/4096 + replay')


@pytest.mark.parametrize('width', [560, 592, 1040])
def test_native_dynamic_stage_reference(width, tmp_path, monkeypatch):
    require_b12x()
    from dataclasses import replace
    from b12x.integration.cuteafd import exportable_compilation
    from . import test_cuteafd_inkling2_nvfp4_moe as oracle

    # Independent F64 stage checks are intentionally small; the full resident
    # family tests above separately exercise real H/E and serving row counts.
    g = replace(_native_geometry('inkling2_nvfp4', 4, width), hidden=128, experts=4, top_k=2)
    with exportable_compilation():
        program = oracle._stage_program(g, 80)
    exported = ExportedProgram(program, tmp_path, f'dynamic_stages_w{width}')
    monkeypatch.setattr(oracle, '_stage_program', lambda *_: exported)
    for rows in [1, 7, 80]:
        report = oracle._run(g, width, 80, rows, gamma_spread=2, bimodal=True)
        oracle._print(f'exported native width {width}, rows {rows}', report)


@pytest.mark.parametrize('hidden,inter', [(16, 16), (560, 80), (592, 112), (1040, 176), (512, 128)])
def test_gate_input_and_column_tails(hidden, inter, tmp_path):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._nvfp4_moe_a4 import (
        StreamNvfp4GateUpA4, StreamNvfp4LinearA4, ChunkSwiGLUA4)

    gate = StreamNvfp4GateUpA4(inter=inter, hidden=hidden, experts=1)
    linear = StreamNvfp4LinearA4(n=inter, k=hidden, experts=1)
    quant = ChunkSwiGLUA4(inter=inter, hidden=hidden, experts=1)

    class Pair:
        def key(self):
            return ('native_gate_tail_pair', gate.key(), linear.key(), quant.key())

        @cute.jit
        def __call__(self, aq: cute.Pointer, scales: cute.Pointer, meta: cute.Pointer,
                     w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                     s2: cute.Pointer, g: cute.Pointer, u: cute.Pointer, fq: cute.Pointer,
                     fs: cute.Pointer, rq: cute.Pointer, rs: cute.Pointer,
                     live: Int32, stream: cuda.CUstream):
            gate(aq, scales, meta, w1, s1, w3, s3, s2, fq, fs, Int32(1), stream)
            linear(aq, scales, meta, w1, s1, g, Int32(1), stream)
            linear(aq, scales, meta, w3, s3, u, Int32(1), stream)
            quant(g, u, meta, s2, rq, rs, Int32(1), stream)

    inputs = ['aq', 'scales', 'meta', 'w1', 's1', 'w3', 's3', 's2']
    outputs = ['g', 'u', 'fq', 'fs', 'rq', 'rs']
    ops = tuple(Operand(n, torch.int32 if n == 'meta' else torch.uint8, '[bytes]', align=4)
                for n in inputs) + tuple(Operand(n, torch.bfloat16 if n in ('g', 'u') else torch.uint8,
                                                '[bytes]', 'out', align=4) for n in outputs)
    pair = Pair()
    with exportable_compilation():
        program = compile_program(pair, name='native_gate_tail_pair', operands=ops,
                                  scalars=(Scalar('live'),), key=pair.key())
    exported = ExportedProgram(program, tmp_path, 'gate_tail_pair')
    torch.manual_seed(hidden)
    codes = lambda *shape: torch.randint(0, 256, shape, device='cuda', dtype=torch.uint8)
    def scale(rows, k):
        return torch.cat([torch.ones(rows * k // 16, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8),
                          torch.tensor([1.0, 1.0], device='cuda').view(torch.uint8)])
    aq = codes(128, hidden // 2)
    scales = torch.ones(128, hidden // 16, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
    w1, w3 = codes(inter, hidden // 2), codes(inter, hidden // 2)
    s1, s3, s2 = scale(inter, hidden), scale(inter, hidden), scale(hidden, inter)
    meta = torch.tensor([0, 1, 0, 0, 128, 0, 0, 0], device='cuda', dtype=torch.int32)
    g, u = [torch.empty(128, inter, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    fq, rq = [torch.full((128, inter // 2), 255, device='cuda', dtype=torch.uint8) for _ in range(2)]
    fs, rs = [torch.full((128, inter // 16), 255, device='cuda', dtype=torch.uint8) for _ in range(2)]
    args = (aq, scales, meta, w1, s1, w3, s3, s2, g, u, fq, fs, rq, rs)
    table = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], device='cuda')
    def decode(p):
        return table[torch.stack([p.long() & 15, p.long() >> 4], -1).flatten(-2)]
    expected_g = (decode(aq).double() @ decode(w1).double().T).bfloat16()
    expected_u = (decode(aq).double() @ decode(w3).double().T).bfloat16()
    for rows in [1, 7, 80, 128]:
        meta[4] = rows
        exported.launch(*args, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(g[:rows], expected_g[:rows])
        assert torch.equal(u[:rows], expected_u[:rows])
        assert torch.equal(fq[:rows], rq[:rows])
        assert torch.equal(fs[:rows], rs[:rows])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exported.launch(*args, scalars=(128,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(fq, rq) and torch.equal(fs, rs)


@pytest.mark.parametrize('k', [128, 256, 640, 1536])
def test_aligned_exported_baseline_bytes(k, tmp_path):
    require_b12x()
    baseline_path = os.environ.get('NVFP4_DUEL_BASELINE_A4')
    if not baseline_path:
        pytest.skip('set NVFP4_DUEL_BASELINE_A4 to the pristine master A4 module')
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._nvfp4_moe_a4 import StreamNvfp4LinearA4

    name = 'b12x.integration.cuteafd._nvfp4_duel_baseline'
    spec = importlib.util.spec_from_file_location(name, baseline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    ops = tuple(Operand(n, torch.int32 if n == 'meta' else torch.uint8, '[bytes]', align=4)
                for n in ['aq', 'a_s', 'meta', 'w', 's']) + (
                    Operand('y', torch.bfloat16, '[rows,N]', 'out'),)
    torch.manual_seed(k)
    n, rows = 128, 128
    aq = torch.randint(0, 256, (rows, k // 2), device='cuda', dtype=torch.uint8)
    w = torch.randint(0, 256, (n, k // 2), device='cuda', dtype=torch.uint8)
    a_s = torch.randint(16, 80, (rows, k // 16), device='cuda', dtype=torch.uint8)
    s = torch.cat([torch.randint(16, 80, (n * k // 16,), device='cuda', dtype=torch.uint8),
                   torch.tensor([0.003, 0.002], device='cuda').view(torch.uint8)])
    meta = torch.tensor([0, 1, 0, 0, rows, 0, 0, 0], device='cuda', dtype=torch.int32)
    outputs, programs = [], []
    for label, cls in [('candidate', StreamNvfp4LinearA4), ('baseline', module.StreamNvfp4LinearA4)]:
        kernel = cls(n=n, k=k, experts=1)
        with exportable_compilation():
            program = compile_program(kernel, name=label, operands=ops, scalars=(Scalar('max_tiles'),),
                                      key=(label, kernel.key()))
        exported = ExportedProgram(program, tmp_path, label)
        programs.append(exported)
        outputs.append(torch.empty(rows, n, device='cuda', dtype=torch.bfloat16))
    for live in [1, 7, 80, 128]:
        meta[4] = live
        for exported, output in zip(programs, outputs):
            exported.launch(aq, a_s, meta, w, s, output, scalars=(1,))
        torch.cuda.synchronize()
        assert torch.equal(outputs[0][:live], outputs[1][:live]), (k, live)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        programs[0].launch(aq, a_s, meta, w, s, outputs[0], scalars=(1,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(*outputs)


def test_glmf_identical_logical_operands(tmp_path, monkeypatch):
    """Test-only zero extension compares native688 with pristine master768.

    CPU generation makes both architectures consume byte-identical operands;
    hashes cover native inputs before the baseline is extended on device.
    """
    require_b12x()
    import hashlib
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd import fp8_moe

    baseline_path = os.environ.get('NVFP4_DUEL_BASELINE_A4')
    if not baseline_path:
        pytest.skip('set NVFP4_DUEL_BASELINE_A4 to pristine master A4 module')
    name = 'b12x.integration.cuteafd._nvfp4_identical_baseline'
    spec = importlib.util.spec_from_file_location(name, baseline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    native, padded = 688, 768
    g = _native_geometry('glmf_nvfp4', 3, native)
    bg = _native_geometry('glmf_nvfp4', 3, padded)
    capacity, h, e = 4096, g.hidden, g.experts
    gen = torch.Generator(device='cpu').manual_seed(688768)
    hashes = {}
    def device(label, tensor):
        cpu = tensor.contiguous()
        hashes[label] = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
        return cpu.cuda()
    weights, baseline_weights = [], []
    for label, rows, k, input_scale in [('gate', native, h, 5/(6*448)),
                                       ('up', native, h, 5/(6*448)),
                                       ('down', h, native, 100/(6*448))]:
        w = device(label + '_codes', torch.randint(0, 256, (e, rows, k // 2), generator=gen, dtype=torch.uint8))
        sc = device(label + '_scales', torch.randint(32, 88, (e, rows, k // 16), generator=gen, dtype=torch.uint8))
        tail = device(label + '_globals', torch.cat([
            torch.rand(e, generator=gen) * .004 + .001,
            input_scale * torch.pow(2., torch.arange(e).float().remainder(7)-3)]).view(torch.uint8))
        weights.extend([w, torch.cat([sc.flatten(), tail])])
        # Only the reference arm receives zeros; native allocation remains exact.
        if label == 'down':
            bw = torch.nn.functional.pad(w, (0, (padded-native)//2))
            bs = torch.nn.functional.pad(sc, (0, (padded-native)//16))
        else:
            bw = torch.nn.functional.pad(w, (0, 0, 0, padded-native))
            bs = torch.nn.functional.pad(sc, (0, 0, 0, padded-native))
        baseline_weights.extend([bw, torch.cat([bs.flatten(), tail])])
    x = device('x', (torch.randint(-256, 257, (capacity, h), generator=gen).float() / 128).bfloat16())
    ids = device('ids', torch.rand(capacity, e, generator=gen).topk(g.top_k, -1).indices.int())
    gamma = device('gamma', torch.rand(capacity, g.top_k, generator=gen))
    print('identical_logical_sha256', hashes, flush=True)
    with exportable_compilation():
        candidate = fp8_moe.compile_fp8_moe_aot(g, route='auto', max_rows=capacity, wire=False)
    candidate = ExportedProgram(candidate, tmp_path, 'glmf_native688')
    for symbol in ['StreamNvfp4LinearA4', 'StreamNvfp4GateUpA4', 'ChunkRowsA4', 'ChunkSwiGLUA4']:
        monkeypatch.setattr(fp8_moe, symbol, getattr(module, symbol))
    with exportable_compilation():
        baseline = fp8_moe.compile_fp8_moe_aot(bg, route='auto', max_rows=capacity, wire=False)
    baseline = ExportedProgram(baseline, tmp_path, 'glmf_master768')
    scratches = [torch.empty(fp8_moe.fp8_moe_scratch_bytes(geometry, 'auto', capacity, False),
                             device='cuda', dtype=torch.uint8) for geometry in [g, bg]]
    outputs = [torch.empty(capacity, h, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    for rows in [1, 7, 80, 2048, 2752, 4096]:
        for program, w, scratch, out in zip([candidate, baseline], [weights, baseline_weights], scratches, outputs):
            scratch.fill_(0xa5)
            program.launch(x, ids, gamma, *w, out, scratch, scalars=(rows,))
        torch.cuda.synchronize()
        different = (outputs[0][:rows] != outputs[1][:rows]).nonzero()
        print('identical_logical_result', rows, 'differing_elements', len(different), flush=True)
        assert not len(different), (rows, different[:20].tolist())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate.launch(x, ids, gamma, *weights, outputs[0], scratches[0], scalars=(4096,))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(*outputs)
