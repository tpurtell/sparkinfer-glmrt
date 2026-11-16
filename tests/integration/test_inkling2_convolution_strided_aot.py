"""FIR strided prefill exported-object oracle at the production row band."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.integration.cuteafd import exportable_compilation
from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot
from b12x.integration.cuteafd.inkling2_convolution_strided import compile_inkling2_convolution_strided_aot
from .test_fir_history_aot import program as _export_fixture, timing


@pytest.fixture(scope="module", params=[(4608, False), (1024, False), (1024, True)])
def program(request, tmp_path_factory):
    # Reuse the exported-object linker/launcher, not its replay compiler.
    from . import test_fir_history_aot as shared
    original = shared.compile_inkling2_convolution_replay_aot
    shared.compile_inkling2_convolution_replay_aot = compile_inkling2_convolution_strided_aot
    try:
        yield from _export_fixture.__wrapped__(request, tmp_path_factory)
    finally:
        shared.compile_inkling2_convolution_replay_aot = original


def reference(x, weight, coefficients, old, first, ids, dynamic):
    x, weight, coefficients, old = (t.cpu() for t in (x, weight, coefficients, old))
    expected = torch.empty_like(x)
    history = old.clone()
    starts = first.cpu().tolist()
    slots = ids.cpu().tolist()
    for begin in sorted(set(starts)):
        end = starts.index(begin) + starts.count(begin)
        slot = slots[begin]
        source = torch.cat((old[slot].flip(0), x[begin:end]))
        result = torch.zeros_like(x[begin:end])
        for lag in range(4):
            value = source[3 - lag:3 - lag + end - begin]
            if dynamic:
                tap = weight[lag].float() + coefficients[begin:end, :, lag].repeat_interleave(x.shape[1] // 8, 1)
            else:
                tap = weight[:, 3 - lag].float()
            result = result + value * tap
        expected[begin:end] = (result + x[begin:end]) * ((9 if dynamic else 5) ** -0.5)
        history[slot] = source[-3:].flip(0)
    return expected, history


def test_strided_prefill(program):
    kind, p, launch = program
    width, dynamic = kind
    old_program = compile_inkling2_convolution_aot(width=width, heads=8 if dynamic else 1,
                                                 dynamic=dynamic, dtype=torch.float32)
    torch.set_num_threads(4)
    for rows in (1, 64, 2752, 4096):
        x = torch.randn(rows, width, device="cuda") * 0.25
        weight = torch.randn((4, width) if dynamic else (width, 4), device="cuda").bfloat16()
        coefficients = torch.randn(rows, 8, 4, device="cuda") * 0.25
        starts = [0] if rows == 1 else [0, 1, 3, rows]
        first = torch.empty(rows, dtype=torch.int32, device="cuda")
        ids = torch.empty(rows, dtype=torch.int64, device="cuda")
        for begin, end, slot in zip(starts[:-1] if rows > 1 else [0],
                                    starts[1:] if rows > 1 else [1], [3, 1, 4]):
            first[begin:end], ids[begin:end] = begin, slot
        initial = torch.randn(5, 3, width, device="cuda")
        dense, legacy = initial.clone(), initial.clone()
        out, old_out = torch.empty_like(x), torch.empty_like(x)
        tensors = [x, weight, coefficients, dense, first, ids, out]
        with kernel_resolution_guard("all prefill row counts share one exported callable"):
            launch(tensors, (rows, 3 * width, width))
            old_program.launch(x, weight, coefficients, legacy, first, ids, old_out,
                               scalars=(rows, 3 * width))
            assert torch.equal(out.view(torch.int32), old_out.view(torch.int32))
            assert torch.equal(dense.view(torch.int32), legacy.view(torch.int32))
            pitch, layer_pitch, field = 11280, 3 * 11280 + 16, 4608
            slot_pitch = 48 * layer_pitch
            bank = torch.full((5 * slot_pitch,), -1414812757, dtype=torch.int32, device="cuda")
            view = torch.as_strided(bank.view(torch.float32), (5, 3, width),
                                    (slot_pitch, pitch, 1), storage_offset=7 * layer_pitch + field)
            view.copy_(initial)
            untouched = bank.clone()
            tensors[3] = bank[7 * layer_pitch + field:].view(torch.float32)
            call = lambda: launch(tensors, (rows, slot_pitch, pitch))
            call()
            assert torch.equal(out.view(torch.int32), old_out.view(torch.int32))
            assert torch.equal(view.view(torch.int32), legacy.view(torch.int32))
            expected_out, expected_history = reference(x, weight, coefficients, initial, first, ids, dynamic)
            assert torch.equal(out.cpu().view(torch.int32), expected_out.view(torch.int32))
            assert torch.equal(view.cpu().view(torch.int32), expected_history.view(torch.int32))
            expected_bank = untouched.clone()
            expected_view = torch.as_strided(expected_bank.view(torch.float32), (5, 3, width),
                                            (slot_pitch, pitch, 1), storage_offset=7 * layer_pitch + field)
            expected_view.copy_(legacy)
            assert torch.equal(bank, expected_bank)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            x.mul_(0.5)
            coefficients.add_(0.125)
            bank.copy_(untouched)
            graph.replay()
            expected_out, expected_history = reference(x, weight, coefficients, initial, first, ids, dynamic)
            assert torch.equal(out.cpu().view(torch.int32), expected_out.view(torch.int32))
            assert torch.equal(view.cpu().view(torch.int32), expected_history.view(torch.int32))
            for invalid in (0, -1, 4097):
                bank.copy_(untouched)
                out.fill_(123)
                launch(tensors, (invalid, slot_pitch, pitch))
                assert torch.equal(bank, untouched)
                assert (out == 123).all()
            timing(f"strided-w{width}-dynamic{int(dynamic)}", call, rows, rows * width * 8)


def test_replay_export_identity(tmp_path):
    """Compare exact CUDA ELF bytes against a frozen pre-change package."""
    baseline = os.environ.get("FIR_BASELINE_SOURCE")
    if not baseline:
        pytest.fail("FIR_BASELINE_SOURCE is required for old-export identity")
    inspector_path = Path("/workspace/cuteafd/scripts/qualify/deepseek_v41/inspect-cute-object.py")
    spec = importlib.util.spec_from_file_location("fir_cuda_images", inspector_path)
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    for width, dynamic in ((4608, False), (1024, False), (1024, True)):
        baseline_dir = tmp_path / f"baseline-{width}-{dynamic}"
        candidate_dir = tmp_path / f"candidate-{width}-{dynamic}"
        baseline_dir.mkdir()
        candidate_dir.mkdir()
        script = f'''from b12x.integration.cuteafd import exportable_compilation
from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot
with exportable_compilation():
 p=compile_inkling2_convolution_replay_aot(width={width},dynamic={dynamic},heads={8 if dynamic else 1})
p.export_to_c({str(baseline_dir)!r}, "replay", "replay")
'''
        subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                       env={**os.environ, "PYTHONPATH": baseline}, check=True)
        from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot
        with exportable_compilation():
            p = compile_inkling2_convolution_replay_aot(width=width, dynamic=dynamic,
                                                       heads=8 if dynamic else 1)
        p.export_to_c(str(candidate_dir), "replay", "replay")
        blobs = [(d / "replay.o").read_bytes() for d in (baseline_dir, candidate_dir)]
        images = [sorted(blob for _, blob in inspector.cuda_images(b)) for b in blobs]
        assert images[0] and images[0] == images[1], "existing replay CUDA ELF bytes changed"
        assert (baseline_dir / "replay.h").read_bytes() == (candidate_dir / "replay.h").read_bytes()
        print(f"FIR_EXPORT_IDENTITY width={width} dynamic={dynamic} exact_cuda_elf=True", flush=True)
