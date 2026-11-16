"""Exported FIR commit/sink CPU bit-model, packed boundaries and CUDA replay."""
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
from b12x.integration.cuteafd.fir_history import compile_fir_history_commit_aot
from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot
from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot


@pytest.fixture(scope="module", params=["commit", (4608, False), (1024, False), (1024, True)])
def program(request, tmp_path_factory):
    kind = request.param
    with exportable_compilation():
        p = (compile_fir_history_commit_aot() if kind == "commit" else
             compile_inkling2_convolution_replay_aot(width=kind[0], dynamic=kind[1],
                                                    heads=8 if kind[1] else 1))
    directory = tmp_path_factory.mktemp(p.name)
    p.export_to_c(str(directory), p.name, p.name)
    validate_exported_header(p, directory / (p.name + ".h"), p.name)
    pointers = ", ".join(f"void *{o.name}" for o in p.operands)
    types = {"int32": "int32_t", "int64": "int64_t", "float32": "float"}
    scalars = ", ".join(types[s.ctype] + " " + s.name for s in p.scalars)
    arguments = ", ".join([o.name for o in p.operands] + [s.name for s in p.scalars])
    source = directory / "bridge.cpp"
    source.write_text(f'''#include "{p.name}.h"
extern "C" void load(void *m) {{ {p.name}_Kernel_Module_Load(({p.name}_Kernel_Module_t *)m); }}
extern "C" void unload(void *m) {{ {p.name}_Kernel_Module_Unload(({p.name}_Kernel_Module_t *)m); }}
extern "C" int launch(void *m, {pointers}, {scalars}, void *stream) {{
  return cute_dsl_{p.name}_wrapper(({p.name}_Kernel_Module_t *)m, {arguments}, (cudaStream_t)stream);
}}
''')
    runtime = subprocess.check_output(
        [sys.executable, "-m", "cutlass.cute.export.aot_config", "--libdir"], text=True).strip()
    library = directory / "fir.so"
    subprocess.run(["g++", "-shared", "-fPIC", str(source), str(directory / (p.name + ".o")),
                    "-I/usr/local/cuda/include", "-L/usr/local/cuda/lib64", "-lcudart",
                    f"-L{runtime}", "-lcute_dsl_runtime", f"-Wl,-rpath,{runtime}",
                    "-Wl,-z,defs", "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.load.argtypes = lib.unload.argtypes = [ctypes.c_void_p]
    lib.load.restype = lib.unload.restype = None
    lib.launch.argtypes = ([ctypes.c_void_p] * (len(p.operands) + 1)
                          + [{"int32": ctypes.c_int32, "int64": ctypes.c_int64,
                              "float32": ctypes.c_float}[s.ctype] for s in p.scalars]
                          + [ctypes.c_void_p])
    lib.launch.restype = ctypes.c_int32
    module = ctypes.c_void_p()
    lib.load(ctypes.byref(module))

    def launch(tensors, scalars):
        assert lib.launch(ctypes.byref(module), *(t.data_ptr() for t in tensors),
                          *scalars, torch.cuda.current_stream().cuda_stream) == 0

    yield kind, p, launch
    torch.cuda.synchronize()
    lib.unload(ctypes.byref(module))


def bits(shape):
    value = torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    value.flatten()[:8] = torch.tensor([0, -(2**31), 0x7F800000, -0x800000,
                                      0x7FC00001, 0x7F800123, -0x3FFFFF, 1], dtype=torch.int32)
    return value


def timing(name, launch, sequences, logical_bytes):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            launch()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1000 / 20
    receipt = dict(primitive=name, sequences=sequences, warmed_us=us,
                   logical_bytes=logical_bytes, logical_gbps=logical_bytes / (1000 * us),
                   device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                   oracle="byte-exact CPU model and exported-object graph replay")
    print("FIR_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
    dest = os.environ.get("FIR_RECEIPTS")
    if dest:
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}-{sequences}.json").write_text(json.dumps(receipt, indent=2) + "\n")


def commit_model(old, replay, table, initial, width, strides):
    lag_stride, layer_stride, slot_stride, row_stride, replay_layer_stride = strides
    result = initial.clone()
    slots = len(old) // slot_stride
    for slot, first, keep in table.t().tolist():
        if not (0 <= slot < slots and 0 <= first <= 64 and 0 <= keep <= 8 and first + keep <= 64):
            continue
        # CPU append/truncate model, not the kernel's source-selection formula.
        for layer in range(48):
            base = slot * slot_stride + layer * layer_stride
            history = [old[base + lag * lag_stride:base + lag * lag_stride + width] for lag in range(3)]
            for row in range(first, first + keep):
                start = layer * replay_layer_stride + row * row_stride
                history = [replay[start:start + width]] + history[:2]
            for lag, value in enumerate(history):
                result[base + lag * lag_stride:base + lag * lag_stride + width] = value
    return result


def test_exported_oracle_and_graph(program):
    kind, p, launch = program
    if kind != "commit":
        check_sink(kind, p, launch)
        return
    torch.manual_seed(1930)
    # Both TP widths, independent row/lag pitches and explicit layer padding.
    for width, padding in ((11264, 0), (5632, 16)):
        lag_stride = width + padding
        layer_stride = 3 * lag_stride + padding
        slot_stride = 48 * layer_stride + padding
        row_stride = width + 2 * padding
        replay_layer_stride = 64 * row_stride + padding
        strides = (lag_stride, layer_stride, slot_stride, row_stride, replay_layer_stride)
        for count in (1, 4, 16, 64):
            slots = count + 2
            old = bits((slots * slot_stride,))
            replay = bits((48 * replay_layer_stride,))
            initial = torch.full_like(old, -1414812757)
            table = torch.tensor([[(i * 17) % count for i in range(count)],
                                  [(i * 7) % 57 for i in range(count)],
                                  [i % 9 for i in range(count)]], dtype=torch.int32)
            # Include a full accept at the upper row boundary.
            table[1, -1], table[2, -1] = 56, 8
            gpu = [old.cuda().view(torch.float32), replay.cuda().view(torch.float32),
                   table.cuda(), initial.cuda().view(torch.float32)]
            scalars = (count, slots, width, *strides)
            call = lambda: launch(gpu, scalars)
            with kernel_resolution_guard("FIR runtime counts/widths use one precompiled export"):
                for keep in range(9):
                    table[2].fill_(keep)
                    gpu[2].copy_(table)
                    gpu[3].view(torch.int32).copy_(initial)
                    call()
                    assert torch.equal(gpu[3].view(torch.int32).cpu(),
                                       commit_model(old, replay, table, initial, width, strides))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                # Table and both source backings mutate at fixed device addresses.
                old.bitwise_xor_(0x12345678)
                replay.bitwise_xor_(0x13579BDF)
                table[2] = torch.arange(count, dtype=torch.int32) % 9
                if count >= 4:
                    table[:, 0] = torch.tensor([-1, 0, 1])
                    table[:, 1] = torch.tensor([slots, 0, 1])
                    table[:, 2] = torch.tensor([table[0, 2], 63, 2])
                    table[:, 3] = torch.tensor([table[0, 3], 0, 9])
                gpu[0].view(torch.int32).copy_(old)
                gpu[1].view(torch.int32).copy_(replay)
                gpu[2].copy_(table)
                gpu[3].view(torch.int32).copy_(initial)
                graph.replay()
                assert torch.equal(gpu[3].view(torch.int32).cpu(),
                                   commit_model(old, replay, table, initial, width, strides))
                for bad_count in (0, 65, -1):
                    gpu[3].view(torch.int32).copy_(initial)
                    launch(gpu, (bad_count, slots, width, *strides))
                    assert torch.equal(gpu[3].view(torch.int32).cpu(), initial)
                assert torch.equal(gpu[0].view(torch.int32).cpu(), old)
                assert torch.equal(gpu[1].view(torch.int32).cpu(), replay)
                # Timing uses valid metadata after the invalid-row safety oracle.
                table = torch.tensor([list(range(count)), [0] * count, [8] * count], dtype=torch.int32)
                gpu[2].copy_(table)
                timing(f"commit-w{width}", call, count, count * 48 * 3 * width * 8)


def check_sink(kind, p, launch):
    width, dynamic = kind
    old = compile_inkling2_convolution_aot(width=width, dynamic=dynamic,
                                         heads=8 if dynamic else 1, dtype=torch.float32)
    count, slots, pitch, field = 64, 5, 11264 + 16, 4608
    # Prefix/suffix guards prove only selected layer/field/input rows are written.
    lag_stride, layer_stride = pitch, 3 * pitch + 16
    slot_stride = 48 * layer_stride
    history = torch.randn(slots * slot_stride, device="cuda")
    view = torch.as_strided(history, (slots, 3, width), (slot_stride, lag_stride, 1),
                            storage_offset=7 * layer_stride + field)
    checkpoint = history.clone()
    dense = view.contiguous()
    x = torch.randn(count, width, device="cuda")
    weights = torch.randn((4, width) if dynamic else (width, 4), device="cuda").bfloat16()
    coeff = torch.randn(count, 8, 4, device="cuda")
    first = torch.tensor([0] * 1 + [1] * 2 + [3] * 8 + [11] * 53, dtype=torch.int32, device="cuda")
    ids = torch.tensor([3] * 1 + [1] * 2 + [4] * 8 + [0] * 53, dtype=torch.int64, device="cuda")
    out, reference = torch.empty_like(x), torch.empty_like(x)
    replay = torch.full((48 * 64 * pitch,), -1414812757, dtype=torch.int32, device="cuda")
    sink = replay[9 * 64 * pitch + field:].view(torch.float32)
    hist = history[7 * layer_stride + field:]
    tensors = [x, weights, coeff, hist, first, ids, out, sink]
    scalars = (count, slot_stride, lag_stride, pitch)

    def call():
        launch(tensors, scalars)

    def baseline():
        old.launch(x, weights, coeff, dense, first, ids, reference, scalars=(count, 3 * width))

    def check():
        assert torch.equal(out.view(torch.int32), reference.view(torch.int32))
        assert torch.equal(view.view(torch.int32), dense.view(torch.int32))
        expected = torch.full_like(replay, -1414812757)
        target = expected.view(48, 64, pitch)[9, :, field:field + width]
        target.copy_(x.view(torch.int32))
        assert torch.equal(replay, expected)
        expected_history = checkpoint.clone()
        target_history = torch.as_strided(expected_history, (slots, 3, width),
                                         (slot_stride, lag_stride, 1),
                                         storage_offset=7 * layer_stride + field)
        target_history.copy_(dense)
        assert torch.equal(history.view(torch.int32), expected_history.view(torch.int32))

    baseline()
    with kernel_resolution_guard("FIR sink graph compiled before replay"):
        call()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        x.mul_(0.5)
        coeff.add_(0.25)
        history.copy_(checkpoint)
        dense.copy_(torch.as_strided(checkpoint, (slots, 3, width), (slot_stride, lag_stride, 1),
                                     storage_offset=7 * layer_stride + field))
        baseline()
        graph.replay()
        check()
        # Arbitrary F32 payloads include signaling NaNs and signed zero.
        x.view(torch.int32).copy_(bits(x.shape))
        history.copy_(checkpoint)
        dense.copy_(torch.as_strided(checkpoint, (slots, 3, width), (slot_stride, lag_stride, 1),
                                     storage_offset=7 * layer_stride + field))
        baseline()
        graph.replay()
        check()
        # Timing uses finite source data; history recurrence remains in the launch.
        x.normal_()
        timing(f"sink-w{width}-dynamic{int(dynamic)}", call, count, count * width * 4)
