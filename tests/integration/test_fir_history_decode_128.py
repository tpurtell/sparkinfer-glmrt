"""Wide decode (128-row) Inkling2 FIR convolution replay & history commit oracles.

Verifies:
1. Default max_rows=64 exports retain byte-identical CUDA ELF images and C headers
   compared to a baseline SparkInfer checkout (when FIR_BASELINE_SOURCE is set).
2. compile_fir_history_commit_aot(max_rows=128) matches an independent CPU bit-model
   at sequences up to 128 (including signaling NaNs, padded strides, boundary rejects
   at 0 / -1 / 129, and CUDA graph replay with mutated inputs).
3. compile_inkling2_convolution_replay_aot(..., max_rows=128) matches
   compile_inkling2_convolution_aot bit-for-bit across all 5 production shapes
   ((4608, False), (1024, False), (1024, True), (512, False), (512, True)) at
   rows up to 128, preserves canary guards, rejects rows > 128, and replays under
   CUDA graphs.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ..conftest import require_b12x
from .test_fir_history_aot import bits


def _build_exported_launcher(program, directory: Path):
    from b12x.integration.cuteafd import validate_exported_header

    directory.mkdir(parents=True, exist_ok=True)
    program.export_to_c(str(directory), program.name, program.name)
    validate_exported_header(program, directory / (program.name + ".h"), program.name)
    pointers = ", ".join(f"void *{o.name}" for o in program.operands)
    types = {"int32": "int32_t", "int64": "int64_t", "float32": "float"}
    scalars = ", ".join(types[s.ctype] + " " + s.name for s in program.scalars)
    arguments = ", ".join([o.name for o in program.operands] + [s.name for s in program.scalars])
    source = directory / "bridge.cpp"
    source.write_text(f'''#include "{program.name}.h"
extern "C" void load(void *m) {{ {program.name}_Kernel_Module_Load(({program.name}_Kernel_Module_t *)m); }}
extern "C" void unload(void *m) {{ {program.name}_Kernel_Module_Unload(({program.name}_Kernel_Module_t *)m); }}
extern "C" int launch(void *m, {pointers}, {scalars}, void *stream) {{
  return cute_dsl_{program.name}_wrapper(({program.name}_Kernel_Module_t *)m, {arguments}, (cudaStream_t)stream);
}}
''')
    runtime = subprocess.check_output(
        [sys.executable, "-m", "cutlass.cute.export.aot_config", "--libdir"], text=True).strip()
    library = directory / "fir128.so"
    subprocess.run(["g++", "-shared", "-fPIC", str(source), str(directory / (program.name + ".o")),
                    "-I/usr/local/cuda/include", "-L/usr/local/cuda/lib64", "-lcudart",
                    f"-L{runtime}", "-lcute_dsl_runtime", f"-Wl,-rpath,{runtime}",
                    "-Wl,-z,defs", "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.load.argtypes = lib.unload.argtypes = [ctypes.c_void_p]
    lib.load.restype = lib.unload.restype = None
    lib.launch.argtypes = ([ctypes.c_void_p] * (len(program.operands) + 1)
                           + [{"int32": ctypes.c_int32, "int64": ctypes.c_int64,
                               "float32": ctypes.c_float}[s.ctype] for s in program.scalars]
                           + [ctypes.c_void_p])
    lib.launch.restype = ctypes.c_int32
    module = ctypes.c_void_p()
    lib.load(ctypes.byref(module))

    def launch(tensors, scalar_values):
        assert lib.launch(ctypes.byref(module), *(t.data_ptr() for t in tensors),
                          *scalar_values, torch.cuda.current_stream().cuda_stream) == 0

    return lib, module, launch


def _commit_model(old, replay, table, initial, width, strides, max_rows=128):
    lag_stride, layer_stride, slot_stride, row_stride, replay_layer_stride = strides
    result = initial.clone()
    slots = len(old) // slot_stride
    for slot, first, keep in table.t().tolist():
        if not (0 <= slot < slots and 0 <= first <= max_rows and 0 <= keep <= 8 and first + keep <= max_rows):
            continue
        for layer in range(48):
            base = slot * slot_stride + layer * layer_stride
            history = [old[base + lag * lag_stride:base + lag * lag_stride + width] for lag in range(3)]
            for row in range(first, first + keep):
                start = layer * replay_layer_stride + row * row_stride
                history = [replay[start:start + width]] + history[:2]
            for lag, value in enumerate(history):
                result[base + lag * lag_stride:base + lag * lag_stride + width] = value
    return result


def test_fir_history_commit_128_oracle_and_graph(tmp_path):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.fir_history import compile_fir_history_commit_aot

    with exportable_compilation():
        prog = compile_fir_history_commit_aot(max_rows=128)
    assert prog.geometry["max_rows"] == 128
    assert prog.geometry["max_sequences"] == 128
    lib, module, launch = _build_exported_launcher(prog, tmp_path / "commit128")
    try:
        torch.manual_seed(2184)
        for width, padding in ((11264, 0), (5632, 16)):
            lag_stride = width + padding
            layer_stride = 3 * lag_stride + padding
            slot_stride = 48 * layer_stride + padding
            row_stride = width + 2 * padding
            replay_layer_stride = 128 * row_stride + padding
            strides = (lag_stride, layer_stride, slot_stride, row_stride, replay_layer_stride)
            for count in (1, 16, 64, 65, 96, 128):
                slots = count + 2
                old = bits((slots * slot_stride,))
                replay = bits((48 * replay_layer_stride,))
                initial = torch.full_like(old, -1414812757)
                table = torch.tensor([[(i * 17) % count for i in range(count)],
                                      [(i * 7) % 121 for i in range(count)],
                                      [i % 9 for i in range(count)]], dtype=torch.int32)
                table[1, -1], table[2, -1] = 120, 8
                gpu = [old.cuda().view(torch.float32), replay.cuda().view(torch.float32),
                       table.cuda(), initial.cuda().view(torch.float32)]
                scalars = (count, slots, width, *strides)
                call = lambda: launch(gpu, scalars)
                with kernel_resolution_guard("FIR commit 128 uses precompiled export"):
                    for keep in (0, 1, 3, 8):
                        table[2].fill_(keep)
                        gpu[2].copy_(table)
                        gpu[3].view(torch.int32).copy_(initial)
                        call()
                        torch.cuda.synchronize()
                        assert torch.equal(gpu[3].view(torch.int32).cpu(),
                                           _commit_model(old, replay, table, initial, width, strides, 128))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        call()
                    old.bitwise_xor_(0x12345678)
                    replay.bitwise_xor_(0x13579BDF)
                    table[2] = torch.arange(count, dtype=torch.int32) % 9
                    if count >= 4:
                        table[:, 0] = torch.tensor([-1, 0, 1])
                        table[:, 1] = torch.tensor([slots, 0, 1])
                        table[:, 2] = torch.tensor([table[0, 2], 127, 2])
                        table[:, 3] = torch.tensor([table[0, 3], 0, 9])
                    gpu[0].view(torch.int32).copy_(old)
                    gpu[1].view(torch.int32).copy_(replay)
                    gpu[2].copy_(table)
                    gpu[3].view(torch.int32).copy_(initial)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(gpu[3].view(torch.int32).cpu(),
                                       _commit_model(old, replay, table, initial, width, strides, 128))
                    for bad_count in (0, 129, -1):
                        gpu[3].view(torch.int32).copy_(initial)
                        launch(gpu, (bad_count, slots, width, *strides))
                        torch.cuda.synchronize()
                        assert torch.equal(gpu[3].view(torch.int32).cpu(), initial)
    finally:
        torch.cuda.synchronize()
        lib.unload(ctypes.byref(module))


@pytest.mark.parametrize("width,dynamic", [
    (4608, False), (1024, False), (1024, True), (512, False), (512, True),
])
def test_inkling2_convolution_replay_128_oracle_and_graph(tmp_path, width, dynamic):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot
    from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot

    heads = width // 128 if dynamic else 1
    with exportable_compilation():
        prog = compile_inkling2_convolution_replay_aot(width=width, dynamic=dynamic, heads=heads, max_rows=128)
    assert prog.geometry["max_rows"] == 128
    old = compile_inkling2_convolution_aot(width=width, dynamic=dynamic, heads=heads, dtype=torch.float32)
    lib, module, launch = _build_exported_launcher(prog, tmp_path / f"replay128_{width}_{int(dynamic)}")
    try:
        slots, pitch, field = 6, 11264 + 16, 4608
        lag_stride, layer_stride = pitch, 3 * pitch + 16
        slot_stride = 48 * layer_stride
        torch.manual_seed(width + 100 * int(dynamic))
        for count in (1, 16, 64, 65, 96, 128):
            history = torch.randn(slots * slot_stride, device="cuda")
            view = torch.as_strided(history, (slots, 3, width), (slot_stride, lag_stride, 1),
                                    storage_offset=7 * layer_stride + field)
            checkpoint = history.clone()
            dense = view.contiguous()
            x = torch.randn(count, width, device="cuda")
            weights = torch.randn((4, width) if dynamic else (width, 4), device="cuda").bfloat16()
            coeff = torch.randn(count, heads, 4, device="cuda")
            first_list = [0] * count if count <= 4 else [0] + [1] * 2 + [3] * 8 + [11] * (count - 11)
            ids_list = [3] * count if count <= 4 else [3] + [1] * 2 + [4] * 8 + [0] * (count - 11)
            first = torch.tensor(first_list, dtype=torch.int32, device="cuda")
            ids = torch.tensor(ids_list, dtype=torch.int64, device="cuda")
            out, reference = torch.empty_like(x), torch.empty_like(x)
            replay = torch.full((48 * 128 * pitch,), -1414812757, dtype=torch.int32, device="cuda")
            sink = replay[9 * 128 * pitch + field:].view(torch.float32)
            hist = history[7 * layer_stride + field:]
            tensors = [x, weights, coeff, hist, first, ids, out, sink]
            scalars = (count, slot_stride, lag_stride, pitch)

            old.launch(x, weights, coeff, dense, first, ids, reference, scalars=(count, 3 * width))
            with kernel_resolution_guard("FIR sink 128 compiled before replay"):
                launch(tensors, scalars)
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int32), reference.view(torch.int32))
                assert torch.equal(view.view(torch.int32), dense.view(torch.int32))
                expected = torch.full_like(replay, -1414812757)
                expected.view(48, 128, pitch)[9, :count, field:field + width].copy_(x.view(torch.int32))
                assert torch.equal(replay, expected)

                if count == 128:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch(tensors, scalars)
                    x.mul_(0.5)
                    coeff.add_(0.25)
                    history.copy_(checkpoint)
                    dense.copy_(torch.as_strided(checkpoint, (slots, 3, width), (slot_stride, lag_stride, 1),
                                                 storage_offset=7 * layer_stride + field))
                    replay.fill_(-1414812757)
                    old.launch(x, weights, coeff, dense, first, ids, reference, scalars=(count, 3 * width))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out.view(torch.int32), reference.view(torch.int32))
                    assert torch.equal(view.view(torch.int32), dense.view(torch.int32))
                    expected.fill_(-1414812757)
                    expected.view(48, 128, pitch)[9, :count, field:field + width].copy_(x.view(torch.int32))
                    assert torch.equal(replay, expected)

                    # Guard rejects invalid row counts > 128 without writing output or replay.
                    out.fill_(123.0)
                    replay.fill_(-1414812757)
                    launch(tensors, (129, slot_stride, lag_stride, pitch))
                    torch.cuda.synchronize()
                    assert (out == 123.0).all()
                    assert (replay == -1414812757).all()
    finally:
        torch.cuda.synchronize()
        lib.unload(ctypes.byref(module))


def test_fir_default_64_exports_byte_identical(tmp_path):
    """Verify default max_rows=64 exports retain exact CUDA ELF and header bytes."""
    baseline = os.environ.get("FIR_BASELINE_SOURCE")
    if not baseline:
        pytest.skip("FIR_BASELINE_SOURCE not set")
    inspector_path = Path(os.environ.get(
        "CUTEAFD_INSPECT_OBJECT",
        "/home/tj/Developer/cute-worktrees/cuteafd-verify-rows/scripts/qualify/deepseek_v41/inspect-cute-object.py",
    ))
    spec = importlib.util.spec_from_file_location("fir_cuda_images", inspector_path)
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)

    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.fir_history import compile_fir_history_commit_aot
    from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot

    cases = [
        ("commit", "from b12x.integration.cuteafd.fir_history import compile_fir_history_commit_aot\np = compile_fir_history_commit_aot()",
         lambda: compile_fir_history_commit_aot()),
    ] + [
        (f"replay_{w}_{int(d)}",
         f"from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot\n"
         f"p = compile_inkling2_convolution_replay_aot(width={w}, dynamic={d}, heads={w // 128 if d else 1})",
         lambda w=w, d=d: compile_inkling2_convolution_replay_aot(width=w, dynamic=d, heads=w // 128 if d else 1))
        for w, d in ((4608, False), (1024, False), (1024, True), (512, False), (512, True))
    ]
    for label, snippet, build in cases:
        bdir, cdir = tmp_path / f"base-{label}", tmp_path / f"cand-{label}"
        bdir.mkdir()
        cdir.mkdir()
        script = (
            "from b12x._lib.aot_target import describe_target\n"
            "describe_target('sm120')\n"
            "from b12x.integration.cuteafd import exportable_compilation\n"
            "with exportable_compilation():\n"
            + "\n".join("    " + line for line in snippet.splitlines()) + "\n"
            f"p.export_to_c({str(bdir)!r}, 'prog', 'prog')\n"
        )
        subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": baseline}, check=True)
        with exportable_compilation():
            p = build()
        p.export_to_c(str(cdir), "prog", "prog")
        b_imgs = sorted(blob for _, blob in inspector.cuda_images((bdir / "prog.o").read_bytes()))
        c_imgs = sorted(blob for _, blob in inspector.cuda_images((cdir / "prog.o").read_bytes()))
        assert b_imgs and b_imgs == c_imgs, f"{label}: CUDA ELF changed"
        assert (bdir / "prog.h").read_bytes() == (cdir / "prog.h").read_bytes(), f"{label}: header changed"
