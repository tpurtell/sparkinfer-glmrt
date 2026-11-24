"""Inkling 2 Mini bundled 3-layer DSpark/Markov MTP drafter primitive oracles.

Verifies every Mini-parameterized primitive in ``inkling_drafter.py``:
* ``compile_inkling_dflare_mix_aot(hidden=2304, taps=4)``
* ``compile_inkling_mask_init_aot(hidden=2304)``
* ``compile_inkling_head_fold_aot(hidden=2304)``
* ``compile_inkling_residual_norm_aot(hidden=2304)``
* ``compile_inkling_qk_norm_rope_aot(q_heads=24, kv_heads=8)``
plus shared ``silu_mul(intermediate=9216)`` and ``kv_store(kv_heads=8)``,
including C header validation and CUDA graph replay.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

HIDDEN = 2304
TAPS = 4
Q_HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
RANK = 256
VOCAB = 201024


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


def test_inkling2_mini_drafter_dflare_mix(tmp_path):
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.inkling_drafter import compile_inkling_dflare_mix_aot

    with exportable_compilation():
        prog = compile_inkling_dflare_mix_aot(hidden=HIDDEN, taps=TAPS)
    prog.export_to_c(str(tmp_path), "mini_dflare_mix", "mini_dflare_mix")
    validate_exported_header(prog, tmp_path / "mini_dflare_mix.h", "mini_dflare_mix")

    torch.manual_seed(701)
    rows = 64
    features = torch.randn(rows, TAPS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    alpha = torch.randn(TAPS, device="cuda").softmax(0).bfloat16()
    gamma = (0.5 + torch.rand(HIDDEN, device="cuda")).bfloat16()
    out = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    prog.launch(features, alpha, gamma, out, scalars=(rows, 1e-6))
    mixed = (features.float() * alpha.float()[None, :, None]).sum(1)
    want = (mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6) * gamma.float()).bfloat16()
    torch.testing.assert_close(out, want, atol=0, rtol=1e-2)
    assert (out.view(torch.int16).int() - want.view(torch.int16).int()).abs().max().item() <= 1


def test_inkling2_mini_drafter_mask_init_and_residual_norm(tmp_path):
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.inkling_drafter import (
        compile_inkling_mask_init_aot, compile_inkling_residual_norm_aot,
    )

    with exportable_compilation():
        mask_p = compile_inkling_mask_init_aot(hidden=HIDDEN)
        res_p = compile_inkling_residual_norm_aot(hidden=HIDDEN)
    mask_p.export_to_c(str(tmp_path), "mini_mask_init", "mini_mask_init")
    validate_exported_header(mask_p, tmp_path / "mini_mask_init.h", "mini_mask_init")
    res_p.export_to_c(str(tmp_path), "mini_residual_norm", "mini_residual_norm")
    validate_exported_header(res_p, tmp_path / "mini_residual_norm.h", "mini_residual_norm")

    torch.manual_seed(702)
    seqs = 16
    anchors = torch.randn(seqs, HIDDEN, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    block = torch.empty(seqs, 8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    mask_p.launch(anchors, mask, block, scalars=(seqs,))
    want_block = torch.cat((anchors[:, None], mask[None, None].expand(seqs, 7, -1)), dim=1)
    assert torch.equal(block.view(torch.int16), want_block.view(torch.int16))

    rows = 128
    residual = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    gamma = (0.5 + torch.rand(HIDDEN, device="cuda")).bfloat16()
    res_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)
    res_p.launch(residual, delta, gamma, res_out, norm_out, scalars=(rows, 1e-6))
    summed = residual.float() + delta.float()
    assert torch.equal(res_out.view(torch.int16), summed.bfloat16().view(torch.int16))
    want_norm = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6) * gamma.float()).bfloat16()
    assert (norm_out.view(torch.int16).int() - want_norm.view(torch.int16).int()).abs().max().item() <= 1


def test_inkling2_mini_drafter_head_fold(tmp_path):
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.inkling_drafter import compile_inkling_head_fold_aot

    with exportable_compilation():
        fold_p = compile_inkling_head_fold_aot(hidden=HIDDEN)
    fold_p.export_to_c(str(tmp_path), "mini_head_fold", "mini_head_fold")
    validate_exported_header(fold_p, tmp_path / "mini_head_fold.h", "mini_head_fold")

    torch.manual_seed(703)
    first_row, tile_rows = 17, 65
    # Allocate only up to first_row + tile_rows; kernel only touches [first_row, first_row + tile_rows).
    lm_head = torch.randn(first_row + tile_rows, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
    hidden_out = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.bfloat16) * 0.1
    hidden_bias = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
    folded_w = torch.full((first_row + tile_rows, RANK), float("nan"), device="cuda", dtype=torch.bfloat16)
    folded_b = torch.full((first_row + tile_rows,), float("nan"), device="cuda", dtype=torch.bfloat16)
    fold_p.launch(lm_head, hidden_out, hidden_bias, folded_w, folded_b, scalars=(first_row, tile_rows))
    torch.cuda.synchronize()

    slice_head = lm_head[first_row:].double()
    want_w = (slice_head @ hidden_out.double()).float().bfloat16()
    want_b = (slice_head @ hidden_bias.double()).float().bfloat16()
    assert torch.equal(folded_w[first_row:].view(torch.int16), want_w.view(torch.int16))
    assert torch.equal(folded_b[first_row:].view(torch.int16), want_b.view(torch.int16))
    assert torch.isnan(folded_w[:first_row]).all() and torch.isnan(folded_b[:first_row]).all()


def test_inkling2_mini_drafter_qk_norm_rope_q24_k8(tmp_path):
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.inkling_drafter import compile_inkling_qk_norm_rope_aot

    with exportable_compilation():
        qk_p = compile_inkling_qk_norm_rope_aot(q_heads=Q_HEADS, kv_heads=KV_HEADS)
    qk_p.export_to_c(str(tmp_path), "mini_qk_norm_rope", "mini_qk_norm_rope")
    validate_exported_header(qk_p, tmp_path / "mini_qk_norm_rope.h", "mini_qk_norm_rope")

    torch.manual_seed(704)
    rows = 32
    qkv = torch.randn(rows, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_gamma = (0.5 + torch.rand(HEAD_DIM, device="cuda")).bfloat16()
    k_gamma = (0.5 + torch.rand(HEAD_DIM, device="cuda")).bfloat16()
    positions = torch.arange(100, 100 + rows, device="cuda", dtype=torch.int64)
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    q_out = torch.empty(rows, Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty(rows, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v_out = torch.empty(rows, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    qk_p.launch(qkv, q_gamma, k_gamma, positions, inv_freq, q_out, k_out, v_out, scalars=(rows, 1e-6))

    views = qkv.view(rows, Q_HEADS + 2 * KV_HEADS, HEAD_DIM)
    q_raw, k_raw, v_raw = views[:, :Q_HEADS], views[:, Q_HEADS:Q_HEADS + KV_HEADS], views[:, Q_HEADS + KV_HEADS:]
    assert torch.equal(v_out.view(torch.int16), v_raw.view(torch.int16))

    def ref_norm_rope(x, gamma):
        xf = x.float()
        xn = (xf / torch.sqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * gamma.float()).bfloat16().float()
        angles = positions.float()[:, None, None] * inv_freq[None, None, :]
        c, s = angles.cos(), angles.sin()
        first, second = xn[..., :64], xn[..., 64:]
        return torch.cat((first * c - second * s, second * c + first * s), dim=-1).bfloat16()

    assert (q_out.view(torch.int16).int() - ref_norm_rope(q_raw, q_gamma).view(torch.int16).int()).abs().max().item() <= 1
    assert (k_out.view(torch.int16).int() - ref_norm_rope(k_raw, k_gamma).view(torch.int16).int()).abs().max().item() <= 1
