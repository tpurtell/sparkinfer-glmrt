"""Compact resident-logit statistics through the production raw-pointer AOT ABI."""
import pytest
import torch

from ..conftest import require_b12x
from ..test_relaxed_acceptance_stats_contract import UINT32_MAX, reference_row
from ._cuteafd import check_export


@pytest.fixture(scope="module")
def program():
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.sampling import compile_relaxed_acceptance_stats_f32_aot

    with exportable_compilation():
        return compile_relaxed_acceptance_stats_f32_aot()


def launch(program, logits, proposals):
    rows, vocab = logits.shape
    ids = torch.tensor(proposals, dtype=torch.uint32, device=logits.device)
    stats = torch.full((rows, 3), float("nan"), device=logits.device)
    argmax = torch.full((rows,), 17, dtype=torch.uint32, device=logits.device)
    scores = torch.full((rows,), float("nan"), device=logits.device)
    status = torch.full_like(argmax, 17)
    from b12x._lib.runtime_control import kernel_resolution_guard

    with kernel_resolution_guard("runtime rows/vocab/stride must reuse one compiled stats program"):
        program.launch(logits, ids, stats, argmax, scores, status,
                       scalars=(rows, vocab, logits.stride(0)))
    torch.cuda.synchronize()
    return stats.cpu(), argmax.cpu().tolist(), scores.cpu(), status.cpu().tolist()


@pytest.mark.parametrize("rows,vocab,padding", [(1, 1, 0), (3, 257, 7), (17, 4093, 5),
                                                (4, 152576, 3), (2, 248320, 1)])
def test_runtime_shapes_stride_and_formula(program, rows, vocab, padding):
    device = require_b12x()
    gen = torch.Generator().manual_seed(314)
    backing = torch.randn((rows, vocab + padding), generator=gen) * 7
    # Poison every padding value: a wrong stride or an unmasked read must fail.
    backing[:, vocab:] = float("nan")
    backing[0, :vocab] = 1.25
    proposals = [(i * 131 + 7) % vocab for i in range(rows)]
    logits = backing.to(device)[:, :vocab]
    before = logits.clone()
    stats, ids, scores, status = launch(program, logits, proposals)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
    for row in range(rows):
        expected, winner, code = reference_row(backing[row, :vocab].tolist(), proposals[row])
        assert ids[row] == winner and status[row] == code
        assert scores[row].item() == backing[row, winner].item()
        torch.testing.assert_close(stats[row], torch.tensor(expected), rtol=3e-5, atol=3e-6)
    assert program.scratch_bytes(rows) == {}


def test_mixed_skipped_invalid_nonfinite_and_extreme_rows(program):
    device = require_b12x()
    logits = torch.tensor([
        [2.0, 2.0, -4.0], [float("nan")] * 3, [0.0, 1.0, 2.0],
        [3.0, float("nan"), 1.0], [float("inf"), 0.0, 1.0],
        [0.0, -float("inf"), 1.0], [3e38, -3e38, 0.0], [-3e38] * 3,
        [1e-40, -1e-40, 0.0], [3.0, 3.0, 1.0],
    ], dtype=torch.float32)
    proposals = [1, UINT32_MAX, 3, 0, 0, 2, 1, 2, 0, UINT32_MAX]
    stats, ids, scores, status = launch(program, logits.to(device), proposals)
    for row, proposal in enumerate(proposals):
        expected, winner, code = reference_row(logits[row].tolist(), proposal)
        assert ids[row] == winner and status[row] == code
        if code == 2:
            assert torch.isnan(scores[row])
        elif code == 3:
            assert scores[row].item() == 0.0
        else:
            assert scores[row].item() == logits[row, winner].item()
        torch.testing.assert_close(stats[row], torch.tensor(expected), rtol=3e-5, atol=3e-6)


def test_bad_proposals_never_read_logit_pointer(program):
    device = require_b12x()
    proposals = torch.full((3,), 257, dtype=torch.uint32, device=device)
    stats = torch.full((3, 3), float("nan"), device=device)
    argmax = torch.zeros(3, dtype=torch.uint32, device=device)
    scores = torch.full((3,), float("nan"), device=device)
    status = torch.zeros_like(argmax)
    # AotProgram passes an invalid dummy device address for None. Invalid
    # proposals must not dereference it (also useful under compute-sanitizer).
    program.launch(None, proposals, stats, argmax, scores, status, scalars=(3, 257, 261))
    torch.cuda.synchronize()
    assert stats.cpu().tolist() == [[0.0] * 3] * 3
    assert argmax.cpu().tolist() == [UINT32_MAX] * 3
    assert scores.cpu().tolist() == [0.0] * 3
    assert status.cpu().tolist() == [3] * 3


def test_batch_width_does_not_change_row_statistics(program):
    device = require_b12x()
    gen = torch.Generator().manual_seed(71)
    logits = torch.randn((9, 1031), generator=gen).to(device)
    proposals = [i * 7 for i in range(9)]
    full_stats, full_ids, full_scores, full_status = launch(program, logits, proposals)
    for start, end in ((0, 1), (1, 3), (3, 9)):
        stats, ids, scores, status = launch(program, logits[start:end], proposals[start:end])
        assert torch.equal(stats.view(torch.int32), full_stats[start:end].view(torch.int32))
        assert ids == full_ids[start:end] and status == full_status[start:end]
        assert torch.equal(scores.view(torch.int32), full_scores[start:end].view(torch.int32))


def test_exported_abi(program, tmp_path):
    result = check_export(program, tmp_path, "relaxed_acceptance_stats_f32")
    assert result["argument_count"] == 10
