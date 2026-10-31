"""Mistral Large 4 programs vs the official modeling arithmetic.

The references are the reference-module formulas (``modeling_mistral_large4``,
transformers 5.22.1; cuteafd ``python/reference/families/mistral_large4/ops.py``
asserts them bit-exact against the official classes): interleaved RoPE with
BF16 cos/sin and BF16 products, then the scalable-softmax factor
``bf16(bf16(s * bf16(log(pos + 1))) + b)`` per head; the indexer's
``LayerNorm(x)`` with weight and bias. Synthetic shapes, big positions.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


def _inv_freq():
    # YaRN over 512 tokens, factor 2048, beta 32/1, no mscale; LeRoPE alpha random.
    import math
    dim, theta, factor, original = 64, 1e4, 2048.0, 512
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation, interpolation = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)

    def correction(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    low, high = max(math.floor(correction(32.0)), 0), min(math.ceil(correction(1.0)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    inv = interpolation * ramp + extrapolation * (1 - ramp)
    alpha = (torch.randn(32, generator=torch.Generator().manual_seed(7)) * 0.1).bfloat16()
    return inv * alpha.float().exp()


def _cos_sin_table(positions: torch.Tensor) -> torch.Tensor:
    """The engine's FP32 [P, 64] cos | sin table (only the rows used)."""
    span = int(positions.max()) + 1
    angles = torch.arange(span, dtype=torch.float32)[:, None] * _inv_freq()[None, :]
    return torch.cat([angles.cos(), angles.sin()], -1)


def _reference_query(q, positions, s, b, nope, rope):
    """ops.mla_query: RoPE (BF16 cos/sin, BF16 products) then SSMax."""
    angles = positions.float()[:, None] * _inv_freq()[None, :]
    cos, sin = angles.cos().bfloat16()[:, None, :], angles.sin().bfloat16()[:, None, :]
    if rope:
        rot = q[..., nope:]
        even, odd = rot[..., 0::2], rot[..., 1::2]
        q = torch.cat([q[..., :nope], torch.cat([even * cos - odd * sin, odd * cos + even * sin], -1)], -1)
    log_pos = (positions + 1).float().log().bfloat16()
    scale = s.bfloat16()[None, :] * log_pos[:, None] + b.bfloat16()[None, :]
    return q * scale[..., None]


def _positions(rows, device):
    # Small, mid and >2^20 positions (log and table rows far from zero).
    base = torch.tensor([0, 1, 2, 511, 512, 4095, 65535, 1048575], dtype=torch.int64)
    return base.repeat(rows // len(base) + 1)[:rows].to(device)


@pytest.mark.parametrize("rope", [True, False])
def test_query_scale_matches_reference(rope):
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import ML4, compile_query_scale_aot

    rows, heads = 40, ML4.heads
    gen = torch.Generator().manual_seed(11)
    q = (torch.randn(rows, heads, 128, generator=gen) * 3).bfloat16()
    s = (torch.rand(heads, generator=gen) * 0.3).bfloat16()
    b = (0.8 + torch.rand(heads, generator=gen) * 0.4).bfloat16()
    positions = _positions(rows, "cpu")
    nope = ML4.nope_dim(rope)
    expected = _reference_query(q, positions, s, b, nope, rope)

    program = compile_query_scale_aot(ML4, rope=rope, max_rows=64)
    q_dev = q.to(device)
    query = torch.full((rows, heads, 576), float("nan"), dtype=torch.bfloat16, device=device)
    table = _cos_sin_table(positions).to(device)
    program.launch(q_dev, positions.to(device), table, s.to(device), b.to(device), query, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(q_dev.cpu(), expected)
    if rope:
        assert torch.equal(query[..., 512:].cpu(), expected[..., nope:])
        assert torch.isnan(query[..., :512].float()).all(), "the latent part belongs to the absorption"


def test_indexer_query_scale_has_no_rope():
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import ML4, compile_query_scale_aot

    rows, heads = 17, ML4.index_heads
    gen = torch.Generator().manual_seed(12)
    q = torch.randn(rows, heads, 128, generator=gen).bfloat16()
    s = (torch.rand(heads, generator=gen) * 0.3).bfloat16()
    b = (0.8 + torch.rand(heads, generator=gen) * 0.4).bfloat16()
    positions = _positions(rows, "cpu")
    program = compile_query_scale_aot(ML4, rope=False, max_rows=64, heads=heads, head_dim=128)
    q_dev = q.to(device)
    program.launch(q_dev, positions.to(device), _cos_sin_table(positions).to(device), s.to(device), b.to(device),
                   torch.empty(1, dtype=torch.bfloat16, device=device), scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(q_dev.cpu(), _reference_query(q, positions, s, b, 128, False))


def test_query_scale_live_rows_reuse_one_program():
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import ML4, compile_query_scale_aot

    program = compile_query_scale_aot(ML4, rope=True, max_rows=64)
    s = torch.full((64,), 0.25, dtype=torch.bfloat16)
    b = torch.ones(64, dtype=torch.bfloat16)
    for rows in (1, 7, 64):
        q = torch.randn(rows, 64, 128).bfloat16()
        positions = _positions(rows, "cpu")
        q_dev = q.to(device)
        query = torch.zeros(rows, 64, 576, dtype=torch.bfloat16, device=device)
        program.launch(q_dev, positions.to(device), _cos_sin_table(positions).to(device), s.to(device),
                       b.to(device), query, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(q_dev.cpu(), _reference_query(q, positions, s, b, 64, True)), rows


def test_layer_norm_bias_matches_torch():
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import compile_layer_norm_aot

    rows, width = 33, 6144
    gen = torch.Generator().manual_seed(13)
    x = (torch.randn(rows, width, generator=gen) * 2 + 0.5).bfloat16()
    w = torch.randn(width, generator=gen).bfloat16()
    bias = torch.randn(width, generator=gen).bfloat16()
    expected = torch.nn.functional.layer_norm(x.float(), (width,), w.float(), bias.float(), 1e-6).bfloat16()
    reference = torch.nn.functional.layer_norm(x.to(device), (width,), w.to(device), bias.to(device), 1e-6)
    program = compile_layer_norm_aot(width=width, max_rows=64)
    out = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
    program.launch(x.to(device), w.to(device), bias.to(device), out, scalars=(rows,))
    torch.cuda.synchronize()
    # F32 statistics: equal to the F32 reference up to reduction order (one BF16 ulp).
    diff = (out.float().cpu() - expected.float()).abs()
    ulp = expected.float().abs().clamp_min(2 ** -126) * 2 ** -7
    assert (diff <= ulp).all(), float((diff / ulp).max())
    assert (out != reference).float().mean() < 1e-3, "matches the CUDA BF16 layer_norm almost everywhere"


def _rotate_index(x, positions):
    """GlmIndexPost's RoPE on dims 0:64 (FP32 cos/sin, one BF16 rounding)."""
    angles = positions.float()[:, None] * _inv_freq()[None, :]
    cos, sin = angles.cos(), angles.sin()
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    xf = x.float()
    even, odd = xf[..., 0:64:2], xf[..., 1:64:2]
    rot = torch.cat([even * cos - odd * sin, odd * cos + even * sin], -1)
    return torch.cat([rot, xf[..., 64:]], -1).bfloat16()


def _e4m3(x):
    """Per-row amax/448 scale (div.rn) and E4M3 codes, as the epilogue stores them."""
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (xf / scale).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("rope", [True, False])
def test_index_post_matches_epilogue_reference(rope):
    """rope=True is the unchanged GLM epilogue; rope=False quantizes the
    unrotated query/key (Mistral Large 4 NoPE layers). Exact codes and scales."""
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import ML4, compile_index_post_aot

    rows, heads = 9, ML4.index_heads
    gen = torch.Generator().manual_seed(14)
    iq = torch.randn(rows, heads, 128, generator=gen).bfloat16()
    kw = torch.randn(rows, 128 + heads, generator=gen).bfloat16()
    k_weight, k_bias = torch.randn(128, generator=gen).bfloat16(), torch.randn(128, generator=gen).bfloat16()
    positions = _positions(rows, "cpu")
    # Pool pages parked past 2^31 / page bytes (the 64-bit offset rule).
    page_bytes = 64 * 132
    high_page = (2 ** 31) // page_bytes + 3
    slots = torch.arange(rows, dtype=torch.int64) + high_page * 64
    cache = torch.zeros((high_page + 1) * page_bytes, dtype=torch.uint8, device=device)
    q_fp8 = torch.empty(rows, heads, 128, dtype=torch.float8_e4m3fn, device=device)
    head_weights = torch.empty(rows, heads, dtype=torch.float32, device=device)
    program = compile_index_post_aot(ML4, rope=rope, max_rows=64)
    program.launch(iq.to(device), kw.to(device), positions.to(device), slots.to(device),
                   _cos_sin_table(positions).to(device), k_weight.to(device), k_bias.to(device),
                   q_fp8, head_weights, cache, scalars=(rows,))
    torch.cuda.synchronize()

    query = _rotate_index(iq, positions) if rope else iq
    codes, q_scale = _e4m3(query)
    assert torch.equal(q_fp8.cpu().view(torch.uint8), codes.view(torch.uint8))
    expected_weights = kw[:, 128:].float() * (float(heads) ** -0.5 * 128.0 ** -0.5) * q_scale[..., 0]
    assert torch.equal(head_weights.cpu(), expected_weights)

    raw = kw[:, :128].float()
    mean = raw.mean(-1, keepdim=True)
    rstd = torch.rsqrt((raw - mean).pow(2).mean(-1, keepdim=True) + 1e-6)
    key = ((raw - mean) * rstd * k_weight.float() + k_bias.float()).bfloat16()
    key = _rotate_index(key, positions) if rope else key
    key_codes, key_scale = _e4m3(key)
    page = cache[high_page * page_bytes:(high_page + 1) * page_bytes].cpu()
    stored = page[:rows * 128].view(rows, 128)
    scales = page[64 * 128:64 * 128 + rows * 4].view(torch.float32)
    # The key's F32 LayerNorm statistics are warp-reduced: allow one E4M3 code step.
    assert (stored.view(torch.float8_e4m3fn).float() * scales[:, None]
            - key_codes.float() * key_scale).abs().max() <= (key_scale * 2 ** -3).max()
