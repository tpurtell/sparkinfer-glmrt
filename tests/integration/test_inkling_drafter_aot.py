"""Exported-object oracles for pinned Inkling 2 MTP math, including replay."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
from b12x.integration.cuteafd import inkling_drafter as kernels


@pytest.fixture(scope="module", params=["dflare_mix", "mask_init", "recurrent_gate", "head_fold", "logit_finish"])
def program(request, tmp_path_factory):
    kind = request.param
    with exportable_compilation():
        compiled = getattr(kernels, f"compile_inkling_{kind}_aot")()
    directory = tmp_path_factory.mktemp(kind)
    stem = "inkling_" + kind
    compiled.export_to_c(str(directory), stem, stem)
    validate_exported_header(compiled, directory / (stem + ".h"), stem)
    # Link the exported object rather than exercising the JIT executor alone.
    source = directory / "bridge.cpp"
    pointers = ", ".join(f"void *{o.name}" for o in compiled.operands)
    scalars = ", ".join(("float" if s.ctype == "float32" else "int32_t") + " " + s.name
                        for s in compiled.scalars)
    arguments = ", ".join([o.name for o in compiled.operands] + [s.name for s in compiled.scalars])
    source.write_text(f'''#include "{stem}.h"
extern "C" void load(void *module) {{
    {stem}_Kernel_Module_Load(({stem}_Kernel_Module_t *)module);
}}
extern "C" void unload(void *module) {{
    {stem}_Kernel_Module_Unload(({stem}_Kernel_Module_t *)module);
}}
extern "C" int launch(void *module, {pointers}, {scalars}, void *stream) {{
    return cute_dsl_{stem}_wrapper(({stem}_Kernel_Module_t *)module,
        {arguments}, (cudaStream_t)stream);
}}
''')
    runtime_dir = subprocess.check_output(
        [sys.executable, "-m", "cutlass.cute.export.aot_config", "--libdir"], text=True).strip()
    library = directory / "primitive.so"
    subprocess.run(["g++", "-shared", "-fPIC", str(source), str(directory / (stem + ".o")),
                    "-I/usr/local/cuda/include", "-L/usr/local/cuda/lib64", "-lcudart",
                    f"-L{runtime_dir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{runtime_dir}",
                    "-Wl,-z,defs", "-o", str(library)], check=True)
    exported = ctypes.CDLL(str(library))
    exported.load.argtypes = exported.unload.argtypes = [ctypes.c_void_p]
    exported.load.restype = exported.unload.restype = None
    exported.launch.argtypes = ([ctypes.c_void_p] * (len(compiled.operands) + 1)
                               + [ctypes.c_float if s.ctype == "float32" else ctypes.c_int32
                                  for s in compiled.scalars] + [ctypes.c_void_p])
    exported.launch.restype = ctypes.c_int32
    module = ctypes.c_void_p()
    exported.load(ctypes.byref(module))
    yield kind, compiled, exported, module
    torch.cuda.synchronize()
    exported.unload(ctypes.byref(module))


def _reference_mix(features, alpha, gamma):
    mixed = (features.float() * alpha.float()[None, :, None]).sum(1)
    return (mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
            * gamma.float()).bfloat16()


def _reference_gate(raw, state):
    gate, candidate, out = raw.bfloat16().chunk(3, dim=-1)
    gate = gate.sigmoid()
    return gate * state + (1 - gate) * candidate.tanh(), out.tanh()


def test_oracle_and_graph(program):
    kind, compiled, exported, module = program
    torch.manual_seed(851)
    if kind in ("head_fold", "logit_finish"):
        _check_head_program(kind, exported, module)
        return
    maximum = 4096 if kind == "dflare_mix" else 64
    if kind == "dflare_mix":
        inputs = [torch.randn(maximum, 8, 4608, device="cuda", dtype=torch.bfloat16),
                  torch.randn(8, device="cuda").softmax(0).bfloat16(),
                  (0.5 + torch.rand(4608, device="cuda")).bfloat16()]
        outputs = [torch.empty(maximum, 4608, device="cuda", dtype=torch.bfloat16)]
        expected = [_reference_mix(*inputs)]
    elif kind == "mask_init":
        inputs = [torch.randn(maximum, 4608, device="cuda", dtype=torch.bfloat16),
                  torch.randn(4608, device="cuda", dtype=torch.bfloat16)]
        outputs = [torch.empty(maximum, 8, 4608, device="cuda", dtype=torch.bfloat16)]
        expected = [torch.cat((inputs[0][:, None], inputs[1][None, None].expand(maximum, 7, -1)), 1)]
    else:
        inputs = [torch.randn(maximum, 768, device="cuda") * 4,
                  torch.randn(maximum, 256, device="cuda", dtype=torch.bfloat16)]
        inputs[0][0] = torch.linspace(-90, 90, 768, device="cuda")
        outputs = [torch.empty_like(inputs[1]), torch.empty_like(inputs[1])]
        expected = list(_reference_gate(*inputs))
    tensors = inputs + outputs
    saved_inputs = [tensor.clone() for tensor in inputs]

    def launch(rows):
        scalars = (rows, 1e-6) if kind == "dflare_mix" else (rows,)
        assert exported.launch(ctypes.byref(module), *(t.data_ptr() for t in tensors),
                               *scalars, torch.cuda.current_stream().cuda_stream) == 0

    def check(rows):
        for actual, ref in zip(outputs, expected):
            if kind == "dflare_mix":
                torch.testing.assert_close(actual[:rows], ref[:rows], atol=0, rtol=1e-2)
            else:
                assert torch.equal(actual[:rows].view(torch.int16), ref[:rows].view(torch.int16))
            assert torch.isnan(actual[rows:]).all()

    with kernel_resolution_guard("Inkling primitives compiled before all live row counts"):
        for rows in (0, 1, 17, maximum, maximum + 1):
            for out in outputs:
                out.fill_(float("nan"))
            launch(rows)
            check(rows if rows <= maximum else 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(maximum)
        for _ in range(3):
            for out in outputs:
                out.fill_(float("nan"))
            graph.replay()
            check(maximum)

    for actual, saved in zip(inputs, saved_inputs):
        assert torch.equal(actual.view(torch.uint8), saved.view(torch.uint8))
    # A captured object must read the current input, not an accidental constant
    # fixture or stale temporary. Restore before taking timing receipts.
    inputs[0].zero_()
    if kind == "dflare_mix":
        expected = [_reference_mix(*inputs)]
    elif kind == "mask_init":
        expected[0][:, 0].zero_()
    else:
        expected = list(_reference_gate(*inputs))
    graph.replay()
    check(maximum)
    for actual, saved in zip(inputs, saved_inputs):
        actual.copy_(saved)
    launch(maximum)

    # Queue many nodes in one replay to exclude Python/ctypes submission gaps.
    # Bandwidth is logical device traffic (cache reuse allowed), not DRAM BW.
    for rows in (1, maximum):
        timed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(timed_graph):
            for _ in range(100):
                launch(rows)
        timed_graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        timed_graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) * 10
        logical_bytes = {"dflare_mix": rows * 4608 * 18 + 2 * (4608 + 8),
                         "mask_init": rows * 8 * 4608 * 4,
                         "recurrent_gate": rows * 256 * (3 * 4 + 3 * 2)}[kind]
        receipt = {"primitive": kind, "rows": rows, "warmed_us": elapsed,
                   "logical_bytes": logical_bytes, "logical_gbps": logical_bytes / (elapsed * 1000),
                   "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                   "oracle": "vendor tolerance" if kind == "dflare_mix" else "bit-exact"}
        if kind == "dflare_mix":
            reference = _reference_mix(*inputs)
            a, e = outputs[0][:rows].view(torch.int16).int(), reference[:rows].view(torch.int16).int()
            receipt.update(bit_equal_fraction=float((a == e).float().mean()), max_ulp=int((a - e).abs().max()))
        print("INKLING_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
        destination = os.environ.get("INKLING_RECEIPTS")
        if destination:
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{kind}-{rows}.json").write_text(json.dumps(receipt, indent=2) + "\n")


def _check_head_program(kind, exported, module):
    def invoke(tensors, *scalars):
        assert exported.launch(ctypes.byref(module), *(t.data_ptr() for t in tensors),
                               *scalars, torch.cuda.current_stream().cuda_stream) == 0

    if kind == "head_fold":
        # Real K/N geometry and padded vocab, including an absolute-offset tail.
        lm = torch.randn(201024, 4608, device="cuda", dtype=torch.bfloat16) * .03
        projection = torch.randn(4608, 256, device="cuda", dtype=torch.bfloat16) * .03
        bias = torch.randn(4608, device="cuda", dtype=torch.bfloat16) * .03
        weight = torch.full((201024, 256), float("nan"), device="cuda", dtype=torch.bfloat16)
        folded_bias = torch.full((201024,), float("nan"), device="cuda", dtype=torch.bfloat16)
        tensors = (lm, projection, bias, weight, folded_bias)
        launch = lambda first, rows: invoke(tensors, first, rows)
        for first, rows in ((0, 1), (3, 17), (201024 - 4096, 4096)):
            launch(first, rows)
            reference = (lm[first:first+rows].double() @ projection.double()).bfloat16()
            ref_bias = (lm[first:first+rows].double() @ bias.double()).bfloat16()
            assert torch.equal(weight[first:first+rows].view(torch.int16), reference.view(torch.int16))
            assert torch.equal(folded_bias[first:first+rows].view(torch.int16), ref_bias.view(torch.int16))
        assert torch.isnan(weight[20:201024-4096]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(0, 17)
        graph.replay()
        assert torch.equal(weight[:17], (lm[:17].double() @ projection.double()).bfloat16())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        launch(0, 4096)
        start.record()
        launch(0, 4096)
        end.record()
        end.synchronize()
        tile_ms = start.elapsed_time(end)
        print("INKLING_FOLD " + json.dumps({"tile_rows": 4096, "tile_ms": tile_ms,
              "projected_full_ms": tile_ms * 201024 / 4096, "device": torch.cuda.get_device_name(),
              "scratch_bytes": 0, "cta_smem_bytes": 512}), flush=True)
        if torch.cuda.get_device_capability() == (12, 0):
            # Startup runs once, not a 100-replay timing of a huge operation.
            start.record()
            for first in range(0, 201024, 4096):
                launch(first, min(4096, 201024-first))
            end.record()
            end.synchronize()
            full_ms = start.elapsed_time(end)
            print("INKLING_FULL_FOLD " + json.dumps({"ms": full_ms, "device": torch.cuda.get_device_name()}), flush=True)
            for first in range(0, 201024, 4096):
                count = min(4096, 201024-first)
                head = lm[first:first+count].double()
                assert torch.equal(weight[first:first+count], (head @ projection.double()).bfloat16())
                assert torch.equal(folded_bias[first:first+count], (head @ bias.double()).bfloat16())
    else:
        maximum = 448
        base = (torch.randn(maximum, 201024, device="cuda") * 20).bfloat16()
        correction = (torch.randn_like(base.float()) * 20).bfloat16()
        # Padded vocabulary is poison and must never appear in the output.
        base[:, 200058:] = float("nan")
        output = torch.full((maximum, 200058), float("nan"), device="cuda", dtype=torch.bfloat16)
        tensors = (base, correction, output)
        reference = ((base[:, :200058] + correction[:, :200058]).float() / 30).tanh().mul(30).bfloat16()
        for rows in (0, 1, 64, 448):
            output.fill_(float("nan"))
            invoke(tensors, rows)
            assert torch.equal(output[:rows].view(torch.int16), reference[:rows].view(torch.int16))
            assert torch.isnan(output[rows:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke(tensors, 64)
        output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(output[:64], reference[:64])
