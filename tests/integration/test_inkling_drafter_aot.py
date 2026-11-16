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


@pytest.fixture(scope="module", params=["dflare_mix", "mask_init", "recurrent_gate"])
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
        outputs = [torch.empty_like(inputs[1]), torch.empty_like(inputs[1])]
        expected = list(_reference_gate(*inputs))
    tensors = inputs + outputs

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
        for rows in (0, 1, 17, maximum):
            for out in outputs:
                out.fill_(float("nan"))
            launch(rows)
            check(rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(maximum)
        for _ in range(3):
            for out in outputs:
                out.fill_(float("nan"))
            graph.replay()
            check(maximum)

    # Queue many kernel nodes in one replay: Python/ctypes submission gaps are
    # larger than these tiny decode operations and are not device execution.
    timed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(timed_graph):
        for _ in range(100):
            launch(maximum)
    timed_graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    timed_graph.replay()
    end.record()
    end.synchronize()
    receipt = {"primitive": kind, "rows": maximum, "warmed_us": start.elapsed_time(end) * 10,
               "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
               "oracle": "vendor tolerance" if kind == "dflare_mix" else "bit-exact"}
    if kind == "dflare_mix":
        a, e = outputs[0].view(torch.int16).int(), expected[0].view(torch.int16).int()
        receipt.update(bit_equal_fraction=float((a == e).float().mean()), max_ulp=int((a - e).abs().max()))
    print("INKLING_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
    destination = os.environ.get("INKLING_RECEIPTS")
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / (kind + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
