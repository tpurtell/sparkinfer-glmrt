"""Report W8 bug-catcher metrics without changing the original strict gate.

Full-model top-1/KL decides the numerical tradeoff. This separate gate retains
all row errors and records what the original 1.01x per-row assertion says.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture
def harness(monkeypatch, request):
    if os.environ.get("CUTEAFD_KIMI_W8_GPU") != "1":
        pytest.skip("explicit CUDA qualification required")
    spec = importlib.util.spec_from_file_location("w8_original_gate", os.environ["CUTEAFD_W8_HARNESS"])
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    torch.set_num_threads(16)
    original_assert = h.assert_accuracy

    def assert_bug_catcher(actual, baseline, oracle):
        actual, baseline = actual.cpu(), baseline.cpu()
        oracle = oracle.cpu().double()
        a, b = actual.double(), baseline.double()
        assert a.shape == b.shape == oracle.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(oracle).all()
        candidate = (a - oracle).square().sum(-1)
        reference = (b - oracle).square().sum(-1)
        ratios = torch.where(reference > 0, (candidate / reference).sqrt(),
                             torch.where(candidate == 0, 0.0, float("inf")))

        def ordered(x):
            bits = x.bfloat16().contiguous().view(torch.int16).long() & 65535
            return torch.where(bits & 32768 != 0, 32768 - (bits & 32767), 32768 + bits)

        row_ulp = (ordered(actual) - ordered(baseline)).abs().amax(-1)
        old_pass = True
        try:
            original_assert(actual, baseline, oracle)
        except AssertionError:
            old_pass = False
        total, ref_total = candidate.sum().item(), reference.sum().item()
        global_ratio = (total / ref_total) ** 0.5 if ref_total else (0.0 if total == 0 else float("inf"))
        record = dict(test=request.node.name, candidate=os.environ["CUTEAFD_W8_CANDIDATE"],
                      candidate_sse=total, official_sse=ref_total, global_ratio=global_ratio,
                      worst_row_ratio=ratios.max().item(), max_bf16_ulp=row_ulp.max().item(),
                      old_gate_pass=old_pass,
                      old_failing_rows=torch.nonzero(candidate > reference * 1.01**2).flatten().tolist(),
                      new_gate_pass=bool(total <= ref_total * 1.01**2 and
                                         (candidate <= reference * 1.25**2).all() and (row_ulp <= 4).all()))
        output = Path(os.environ["CUTEAFD_W8_POLICY_OUT"])
        output.mkdir(parents=True, exist_ok=True)
        stem = request.node.name.replace("[", "-").replace("]", "")
        torch.save(dict(candidate_sse=candidate, official_sse=reference,
                        row_ratio=ratios, row_max_bf16_ulp=row_ulp), output / (stem + ".pt"))
        (output / (stem + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print("W8_POLICY " + json.dumps(record), flush=True)
        assert total <= ref_total * 1.01**2, "global error exceeds 1.01x official"
        assert bool((candidate <= reference * 1.25**2).all()), "row error exceeds 1.25x official"
        assert bool((row_ulp <= 4).all()), "output differs by more than 4 BF16 ULP"

    monkeypatch.setattr(h, "assert_accuracy", assert_bug_catcher)
    return h


@pytest.mark.parametrize("candidates", [2, 9])
def test_residual_bug_catcher(harness, candidates):
    harness.test_residual_prefill_official_fp64(candidates)


@pytest.mark.parametrize("intermediate", [3072, 6144, 16896, 33792])
@pytest.mark.parametrize("compensation", ["direct", "gate_up_only"])
def test_ffn_bug_catcher(harness, intermediate, compensation):
    harness.test_ffn_chain_official_fp64(intermediate, compensation)
