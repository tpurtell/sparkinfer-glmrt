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


@pytest.fixture(scope="module", params=["dflare_mix", "mask_init", "recurrent_gate", "head_fold", "logit_finish", "residual_norm", ("qk_norm_rope", 48, 8), ("qk_norm_rope", 24, 4), ("qk_norm_rope", 0, 8), ("qk_norm_rope", 0, 4), ("silu_mul", 18432), ("silu_mul", 9216), ("bias_add", 768), ("bias_add", 201024), ("kv_store", 8, "fp8"), ("kv_store", 4, "fp8"), ("kv_store", 8, "bf16"), ("kv_store", 4, "bf16")])
def program(request, tmp_path_factory):
    kind = request.param
    geometry = kind[1:] if isinstance(kind, tuple) else None
    kind = kind[0] if geometry else kind
    with exportable_compilation():
        kwargs = ({"intermediate": geometry[0]} if kind == "silu_mul" else
                  {"width": geometry[0]} if kind == "bias_add" else
                  dict(kv_heads=geometry[0], kv_format=geometry[1]) if kind == "kv_store" else
                  dict(q_heads=geometry[0], kv_heads=geometry[1]) if geometry else {})
        compiled = getattr(kernels, f"compile_inkling_{kind}_aot")(**kwargs)
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
    if kind in ("residual_norm", "qk_norm_rope"):
        _check_norm_program(kind, exported, module, compiled.geometry)
        return
    if kind == "silu_mul":
        _check_silu_program(exported, module, compiled.geometry["intermediate"])
        return
    if kind == "bias_add":
        _check_bias_program(exported, module, compiled.geometry["width"])
        return
    if kind == "kv_store":
        _check_kv_program(exported, module, compiled.geometry, Path(exported._name).parent)
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
            actual = weight[first:first+rows]
            if not torch.equal(actual.view(torch.int16), reference.view(torch.int16)):
                mismatch = actual.view(torch.int16) != reference.view(torch.int16)
                precise = lm[first:first+rows].double() @ projection.double()
                print("FOLD_MISMATCH", int(mismatch.sum()),
                      precise[mismatch][:16].tolist(), actual[mismatch][:16].float().tolist(),
                      reference[mismatch][:16].float().tolist(), flush=True)
                print("FOLD_DOUBLE_ROUND_REFERENCE", torch.equal(reference, precise.float().bfloat16()), flush=True)
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
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


def test_vendor_double_rounding_cast():
    # These exact FP64 values straddle BF16 midpoint ties after narrowing F32.
    values = torch.tensor([-0.00214385989420407, 0.020568848382026772,
                           0.05627441587694193, 0.047729490797792096], dtype=torch.float64)
    expected = torch.tensor([-0.00213623046875, 0.0205078125,
                            0.05615234375, 0.0478515625], dtype=torch.bfloat16)
    for device in ("cpu", "cuda"):
        x = values.to(device)
        assert torch.equal(x.bfloat16().cpu(), expected)
        assert torch.equal(x.bfloat16(), x.float().bfloat16())


def _check_norm_program(kind, exported, module, geometry):
    def invoke(tensors, rows):
        assert exported.launch(ctypes.byref(module), *(t.data_ptr() for t in tensors), rows,
                               1e-6, torch.cuda.current_stream().cuda_stream) == 0

    if kind == "residual_norm":
        residual = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16)
        delta = torch.randn_like(residual)
        gamma = (.5 + torch.rand(4608, device="cuda")).bfloat16()
        residual_out, norm_out = torch.empty_like(residual), torch.empty_like(residual)
        tensors = (residual, delta, gamma, residual_out, norm_out)
        value = residual.float() + delta.float()
        expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                    * gamma.float()).bfloat16()
        for rows in (1, 64, 4096):
            invoke(tensors, rows)
            assert torch.equal(residual_out[:rows], value[:rows].bfloat16())
            torch.testing.assert_close(norm_out[:rows], expected[:rows], atol=0, rtol=1e-2)
        a, b = norm_out.view(torch.int16).int(), expected.view(torch.int16).int()
        print("INKLING_RESIDUAL_NORM " + json.dumps({
              "bit_equal_fraction": float((a == b).float().mean()),
              "max_ulp": int((a-b).abs().max())}), flush=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke(tensors, 64)
        graph.replay()
        torch.testing.assert_close(norm_out[:64], expected[:64], atol=0, rtol=1e-2)
    else:
        rows, q_heads, kv_heads = 4096, geometry["q_heads"], geometry["kv_heads"]
        qkv = torch.randn(rows, q_heads + 2*kv_heads, 128, device="cuda", dtype=torch.bfloat16)
        q_gamma = (.5 + torch.rand(128, device="cuda")).bfloat16()
        k_gamma = (.5 + torch.rand(128, device="cuda")).bfloat16()
        positions_cpu = torch.linspace(0, 1048575, rows).long()
        positions_cpu[:8] = torch.tensor([0, 1, 1023, 1024, 65535, 65536, 1048574, 1048575])
        positions = positions_cpu.cuda()
        # Pinned vLLM base.py builds the FP32 cache with a CPU outer product.
        inv_freq_cpu = 1. / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        inv_freq = inv_freq_cpu.cuda()
        angle = torch.einsum("i,j->ij", positions.float(), inv_freq)
        cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
        cpu_angle = torch.einsum("i,j->ij", positions_cpu.float(), inv_freq_cpu)
        for name, gpu, cpu in (("cos", cosine[:, 0], cpu_angle.cos().cuda()),
                               ("sin", sine[:, 0], cpu_angle.sin().cuda())):
            print("INKLING_TRIG_CPU " + json.dumps({"kind": name,
                  "bit_equal_fraction": float((gpu == cpu).float().mean()),
                  "max_abs": float((gpu-cpu).abs().max())}), flush=True)
        q_out = torch.empty(rows, q_heads, 128, device="cuda", dtype=torch.bfloat16)
        k_out = torch.empty(rows, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
        v_out = torch.empty_like(k_out)
        tensors = (qkv, q_gamma, k_gamma, positions, inv_freq, q_out, k_out, v_out)

        def norm_rope(source, gamma):
            value = source.float()
            inv = 1 / (value.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            value = (value * inv * gamma.float()).bfloat16().float()
            first, second = value.chunk(2, dim=-1)
            return torch.cat((first * cosine - second * sine,
                              second * cosine + first * sine), -1).bfloat16()

        q_expected = norm_rope(qkv[:, :q_heads], q_gamma)
        k_expected = norm_rope(qkv[:, q_heads:q_heads+kv_heads], k_gamma)
        for live in (1, 64, rows):
            invoke(tensors, live)
            torch.testing.assert_close(q_out[:live], q_expected[:live], atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(k_out[:live], k_expected[:live], atol=2e-2, rtol=1e-2)
            assert torch.equal(v_out[:live], qkv[:live, q_heads+kv_heads:])
        for name, actual, expected in (("q", q_out, q_expected), ("k", k_out, k_expected)):
            a, b = actual.view(torch.int16).int(), expected.view(torch.int16).int()
            if a.numel():
                ordered_a = torch.where(a < 0, -32768-a, a)
                ordered_b = torch.where(b < 0, -32768-b, b)
                print("INKLING_ROPE " + json.dumps({"kind": name, "rows": rows,
                      "heads": [q_heads, kv_heads],
                      "bit_equal_fraction": float((a == b).float().mean()),
                      "max_ulp": int((ordered_a-ordered_b).abs().max()),
                      "max_abs": float((actual.float()-expected.float()).abs().max())}), flush=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke(tensors, 64)
        graph.replay()
        torch.testing.assert_close(q_out[:64], q_expected[:64], atol=2e-2, rtol=1e-2)
        # Feed identical reference-normalized BF16 operands into the same
        # rotation body to distinguish norm reduction effects from trig/order.
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        normalized_qkv = qkv.clone()
        for start, count, gamma in ((0, q_heads, q_gamma), (q_heads, kv_heads, k_gamma)):
            value = qkv[:, start:start+count].float()
            inverse = 1 / (value.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            normalized_qkv[:, start:start+count] = (value * inverse * gamma.float()).bfloat16()
        with exportable_compilation():
            rotation = compile_program(kernels._QKNormRope(q_heads, kv_heads, normalize=False),
                name="inkling_rotation_oracle", operands=(
                    Operand("qkv", torch.bfloat16, "[rows,heads,128]"),
                    Operand("q_gamma", torch.bfloat16, "[128]"),
                    Operand("k_gamma", torch.bfloat16, "[128]"),
                    Operand("positions", torch.int64, "[rows]"),
                    Operand("inv_freq", torch.float32, "[64]"),
                    Operand("q_out", torch.bfloat16, "[rows,q,128]", "out"),
                    Operand("k_out", torch.bfloat16, "[rows,k,128]", "out"),
                    Operand("v_out", torch.bfloat16, "[rows,k,128]", "out"),
                ), scalars=(Scalar("rows"), Scalar("eps", "float32")),
                key=(q_heads, kv_heads, "isolated_rotation"))
        rotation.launch(normalized_qkv, *tensors[1:], scalars=(rows, 1e-6))
        print("INKLING_ROTATION_ONLY", [q_heads, kv_heads],
              int((q_out.view(torch.int16) != q_expected.view(torch.int16)).sum()),
              int((k_out.view(torch.int16) != k_expected.view(torch.int16)).sum()), flush=True)
        assert torch.equal(q_out.view(torch.int16), q_expected.view(torch.int16))
        assert torch.equal(k_out.view(torch.int16), k_expected.view(torch.int16))


def _check_silu_program(exported, module, intermediate):
    source = (torch.randn(4096, 2 * intermediate, device="cuda") * 4).bfloat16()
    output = torch.full((4096, intermediate), float("nan"), device="cuda", dtype=torch.bfloat16)
    gate, up = source.float().chunk(2, -1)
    expected = (gate / (1 + (-gate).exp()) * up).bfloat16()

    def launch(rows):
        assert exported.launch(ctypes.byref(module), source.data_ptr(), output.data_ptr(), rows,
                               torch.cuda.current_stream().cuda_stream) == 0

    for rows in (0, 1, 64, 512, 4096):
        output.fill_(float("nan"))
        launch(rows)
        assert torch.equal(output[:rows].view(torch.int16), expected[:rows].view(torch.int16))
        assert torch.isnan(output[rows:]).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(64)
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output[:64], expected[:64])


def _check_bias_program(exported, module, width):
    source = torch.randn(448, width, device="cuda")
    bias = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    dtype = torch.bfloat16 if width == 201024 else torch.float32
    output = torch.full((448, width), float("nan"), device="cuda", dtype=dtype)
    expected = (source + bias.float()).to(dtype)

    def launch(rows):
        assert exported.launch(ctypes.byref(module), source.data_ptr(), bias.data_ptr(),
                               output.data_ptr(), rows, torch.cuda.current_stream().cuda_stream) == 0

    for rows in (0, 1, 64, 448):
        output.fill_(float("nan"))
        launch(rows)
        assert torch.equal(output[:rows].view(torch.uint8), expected[:rows].view(torch.uint8))
        assert torch.isnan(output[rows:]).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(64)
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output[:64], expected[:64])


def _check_kv_program(exported, module, geometry, directory):
    heads, fp8, stride = geometry["kv_heads"], geometry["kv_format"] == "fp8", geometry["plane_bytes"]
    high = (2**31 // stride) + 2
    sources = [torch.randn(64, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    for source in sources:
        source[0].zero_()
    slots = torch.arange(64, device="cuda", dtype=torch.int64) * 2
    slots[1], slots[-1] = -1, high
    # Sparse writes into mostly-uninitialized backing exercise >2GiB offsets.
    dtype = torch.uint8 if fp8 else torch.bfloat16
    elements = stride if fp8 else stride // 2
    planes = [torch.empty(high+1, elements, device="cuda", dtype=dtype) for _ in range(2)]
    for plane in planes:
        plane[:128].zero_()
        plane[-1].zero_()

    def launch(rows):
        tensors = (*sources, slots, *planes)
        assert exported.launch(ctypes.byref(module), *(t.data_ptr() for t in tensors), rows,
                               torch.cuda.current_stream().cuda_stream) == 0

    codec = None
    references = None
    if fp8:
        library = directory / "codec.so"
        arch = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
        subprocess.run(["nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=" + arch,
                        str(Path(__file__).with_name("inkling_kv_codec.cu")),
                        "-o", str(library)], check=True)
        codec = ctypes.CDLL(str(library))
        codec.launch_codec.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        codec.launch_codec.restype = ctypes.c_int
        references = [torch.zeros(128, elements, device="cuda", dtype=torch.uint8) for _ in range(2)]
        small_slots = slots.clone()
        small_slots[-1] = 127
        assert codec.launch_codec(*(t.data_ptr() for t in (*sources, small_slots, *references)),
                                  64, heads, torch.cuda.current_stream().cuda_stream) == 0

    def check(rows):
        for plane_index, (source, plane) in enumerate(zip(sources, planes)):
            for row in range(rows):
                slot = int(slots[row])
                if slot < 0:
                    continue
                if not fp8:
                    assert torch.equal(plane[slot].reshape(heads, 128), source[row])
                else:
                    value = source[row].float()
                    maximum = value.abs().amax(-1)
                    scale = torch.where(maximum > 0, (maximum.double() / 448).float(), 1)
                    inverse = (1 / scale.double()).float()
                    codes = (value * inverse[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
                    expected = torch.zeros(stride, device="cuda", dtype=torch.uint8)
                    expected[:heads*128] = codes.flatten()
                    expected[heads*128:heads*132] = scale.view(torch.uint8)
                    native_expected = references[plane_index][127 if row == 63 else slot]
                    if not torch.equal(plane[slot], native_expected):
                        mismatch = plane[slot] != native_expected
                        print("KV_CODEC_MISMATCH", heads, row, int(mismatch.sum()),
                              torch.nonzero(mismatch).flatten()[:24].tolist(), flush=True)
                    assert torch.equal(plane[slot], native_expected)
                    assert torch.equal(expected, native_expected)
            assert torch.count_nonzero(plane[1]) == 0

    for rows in (0, 1, 64):
        launch(rows)
        check(rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(64)
    graph.replay()
    check(64)
