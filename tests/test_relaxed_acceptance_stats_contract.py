"""CPU-only formula and AOT ABI contracts; no CUDA or b12x imports."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "b12x/integration/cuteafd/sampling.py"
UINT32_MAX = 0xFFFFFFFF


def reference_row(logits, proposal):
    """Independent F64 oracle for the finite raw-logit device contract."""
    if proposal != UINT32_MAX and (proposal < 0 or proposal >= len(logits)):
        return [0.0, 0.0, 0.0], UINT32_MAX, 3
    if not all(math.isfinite(z) for z in logits):
        return [0.0, 0.0, 0.0], UINT32_MAX, 2
    maximum = max(logits)
    if proposal == UINT32_MAX:
        return [0.0, 0.0, 0.0], logits.index(maximum), 1
    weights = [math.exp(z - maximum) for z in logits]
    mass = math.fsum(weights)
    moment = math.fsum(w * (z - maximum) for z, w in zip(logits, weights) if w > 0)
    return [weights[proposal] / mass, 1 / mass, math.log(mass) - moment / mass], logits.index(maximum), 0


@pytest.mark.parametrize("logits,proposal", [
    ([0.0], 0), ([0.0] * 257, 256), ([2.0, 2.0, -7.0], 1),
    ([10000.0, 9999.0, -10000.0], 2), ([3e38, -3e38, 0.0], 1),
    ([-3e38, -3e38], 0), ([1e-40, -1e-40, 0.0], 2),
])
def test_stable_formula_matches_probability_entropy(logits, proposal):
    stats, winner, status = reference_row(logits, proposal)
    weights = [math.exp(z - max(logits)) for z in logits]
    probs = [w / math.fsum(weights) for w in weights]
    expected_entropy = -math.fsum(p * math.log(p) for p in probs if p > 0)
    assert status == 0
    assert winner == logits.index(max(logits))
    assert stats == pytest.approx([probs[proposal], max(probs), expected_entropy], abs=1e-12)
    assert 0 <= stats[0] <= stats[1] <= 1
    assert 0 <= stats[2] <= math.log(len(logits)) + 1e-12


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejects_even_nonwinning_columns(nonfinite):
    assert reference_row([7.0, nonfinite], 0) == ([0.0] * 3, UINT32_MAX, 2)


def test_bad_proposal_does_not_read_logits():
    class Poison:
        def __len__(self):
            return 3

        def __iter__(self):
            raise AssertionError("disabled row touched logits")

    assert reference_row(Poison(), 3) == ([0.0] * 3, UINT32_MAX, 3)


def test_skip_returns_checked_argmax_without_stats():
    assert reference_row([1.0, 3.0, 3.0], UINT32_MAX) == ([0.0] * 3, 1, 1)
    assert reference_row([float("nan"), 3.0], UINT32_MAX) == ([0.0] * 3, UINT32_MAX, 2)


def test_shift_invariance_and_smallest_id_tie():
    logits = [-7.5, 1.25, 1.25, -2.5]
    expected = reference_row(logits, 2)
    shifted = reference_row([z + 1000.0 for z in logits], 2)
    assert expected[0] == pytest.approx(shifted[0])
    assert expected[1:] == shifted[1:] == (1, 0)


def test_export_contract_without_cuda_imports():
    tree = ast.parse(SOURCE.read_text())
    compile_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "compile_relaxed_acceptance_stats_f32_aot")
    namespace = dict(
        RelaxedAcceptanceStatsF32=lambda: None,
        Operand=lambda *args, **kwargs: (args, kwargs),
        Scalar=lambda *args: args,
        torch=SimpleNamespace(float32="f32", uint32="u32"),
        compile_program=lambda launch, **kwargs: kwargs,
        UINT32_MAX=UINT32_MAX,
    )
    exec(compile(ast.Module(body=[compile_fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    result = namespace[compile_fn.name]()
    assert result["name"] == "relaxed_acceptance_stats_f32"
    assert [(args[0], args[1]) for args, _ in result["operands"]] == [
        ("logits", "f32"), ("proposals", "u32"), ("stats", "f32"),
        ("argmax_ids", "u32"), ("argmax_scores", "f32"), ("status", "u32")]
    assert all(kwargs["align"] == 4 for _, kwargs in result["operands"])
    assert result["scalars"] == (("rows",), ("vocab",), ("row_stride", "int64"))
    assert result["key"] == ()
    assert result.get("scratch", {}) == {}


def test_skipped_branch_precedes_softmax_and_offsets_are_int64():
    tree = ast.parse(SOURCE.read_text())
    kernel = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "kernel")
    branch = next(node for node in kernel.body if isinstance(node, ast.If))
    assert "UINT32_MAX" in ast.unparse(branch.test)
    assert "vocab" in ast.unparse(branch.test)
    assert "logits[" not in ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
    skipped = next(node for node in ast.walk(kernel) if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "proposal == Uint32(UINT32_MAX)")
    assert "SKIPPED" in ast.unparse(skipped.body[0])
    assert "cute.math.exp" not in ast.unparse(ast.Module(body=skipped.body, type_ignores=[]))
    assert "cute.math.exp" in ast.unparse(ast.Module(body=skipped.orelse, type_ignores=[]))
    assert "base = row * row_stride" in ast.unparse(kernel)
    assert "row = Int64(" in ast.unparse(kernel)
