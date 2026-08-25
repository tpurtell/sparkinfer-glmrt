from __future__ import annotations

import hashlib

import pytest
import torch

from b12x._lib.quant.sqg_fp16_d3l import (
    SQG_FP16_D3L_DESCRIPTOR_BYTES,
    SQG_FP16_D3L_DESCRIPTOR_COUNT,
    SQG_FP16_D3L_DESCRIPTOR_SHA256,
    decode_sqg_fp16_d3l_ranks_torch,
    sqg_fp16_d3l_descriptors_cpu,
    sqg_fp16_d3l_direct_lut_cpu,
    sqg_xor_high_rate_rank_for_codewords,
)


def test_d3l_descriptor_payload_has_frozen_identity() -> None:
    payload = sqg_fp16_d3l_descriptors_cpu()
    assert payload.dtype == torch.uint8
    assert payload.numel() == SQG_FP16_D3L_DESCRIPTOR_BYTES
    assert payload.view(torch.float16).numel() == 2 * SQG_FP16_D3L_DESCRIPTOR_COUNT
    assert hashlib.sha256(payload.numpy().tobytes()).hexdigest() == (
        SQG_FP16_D3L_DESCRIPTOR_SHA256
    )


def test_d3l_rank_law_is_finite_sign_symmetric_and_close_to_gaussian() -> None:
    ranks = torch.arange(1 << 16, dtype=torch.int64)
    actual = decode_sqg_fp16_d3l_ranks_torch(ranks)
    target = (
        1.5 * torch.special.ndtri((ranks.double() + 0.5) / (1 << 16))
    ).to(torch.float16)
    assert actual.dtype == torch.float16
    assert bool(torch.isfinite(actual).all())
    assert torch.equal(actual, -actual.flip(0))
    error = actual.float() - target.float()
    assert float(torch.sqrt(torch.mean(error.square()))) < 7.0e-4
    assert float(error.abs().max()) <= 4.0e-3


@pytest.mark.parametrize("bits", [5, 6])
def test_d3l_direct_lut_uses_bijective_carry_mixed_graph(bits: int) -> None:
    codewords = torch.arange(1 << 16, dtype=torch.int64)
    ranks = sqg_xor_high_rate_rank_for_codewords(codewords, bits)
    assert torch.equal(torch.sort(ranks).values, codewords)
    expected = decode_sqg_fp16_d3l_ranks_torch(ranks)
    assert torch.equal(sqg_fp16_d3l_direct_lut_cpu(bits), expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 7])
def test_d3l_rejects_non_high_rates(bits: int) -> None:
    with pytest.raises(ValueError, match="K5/K6"):
        sqg_xor_high_rate_rank_for_codewords(torch.zeros(1), bits)
