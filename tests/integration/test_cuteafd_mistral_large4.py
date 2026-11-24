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


# ---------------------------------------------------------------------------
# One attention layer from the shared GLM programs at Mistral Large 4 geometry
# ---------------------------------------------------------------------------

def _official_attention(x, positions, w, rope, kv_source=None):
    """``MistralLarge4TextAttention.forward`` (eager, no indexer: every causal key),
    F32 from the BF16 operands: q = ssmax(cat(q_nope, rope(q_rot))), latent =
    rms_norm(kv_a[:512]), k_rot = rope(kv_a[512:]), [k_nope | v] = kv_b(latent),
    softmax(q k^T * 128^-0.5) v, wo. Returns (out [T, H], latent, k_rot)."""
    T, H, nope = x.shape[0], 64, (64 if rope else 128)
    q = (x.float() @ w["wq"].float().t()).bfloat16().view(T, H, 128)
    q = _reference_query(q, positions.cpu(), w["s"], w["b"], nope, rope)
    if kv_source is None:
        kv = (x.float() @ w["wkv_a"].float().t()).bfloat16()
        lat = kv[:, :512].float()
        lat = (w["kv_a_norm"].float() * (lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16().float()).bfloat16()
        k_rot = None
        if rope:
            angles = positions.cpu().float()[:, None] * _inv_freq()[None, :]
            c, s = angles.cos().bfloat16(), angles.sin().bfloat16()
            even, odd = kv[:, 512::2], kv[:, 513::2]
            k_rot = torch.cat([even * c - odd * s, odd * c + even * s], -1)
    else:
        lat, k_rot = kv_source
    kvb = (lat.float() @ w["wkv_b"].float().t()).bfloat16().view(T, H, nope + 128)
    k = kvb[..., :nope]
    if rope:
        k = torch.cat([k, k_rot[:, None, :].expand(T, H, 64)], -1)
    v = kvb[..., nope:]
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * 128 ** -0.5
    scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    attn = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float()).bfloat16()
    return (attn.float().reshape(T, -1) @ w["wo"].float().t()), lat, k_rot


def _attention_weights(rope, gen):
    nope = 64 if rope else 128
    scale = lambda *shape, s: (torch.randn(*shape, generator=gen) * s).bfloat16()  # noqa: E731
    return dict(wq=scale(64 * 128, 6144, s=6144 ** -0.5), wkv_a=scale(512 + (64 if rope else 0), 6144, s=6144 ** -0.5),
                kv_a_norm=(1 + torch.randn(512, generator=gen) * 0.1).bfloat16(),
                wkv_b=scale(64 * (nope + 128), 512, s=512 ** -0.5), wo=scale(6144, 64 * 128, s=8192 ** -0.5),
                s=(torch.rand(64, generator=gen) * 0.2).bfloat16(), b=(0.9 + torch.rand(64, generator=gen) * 0.2).bfloat16())


@pytest.mark.parametrize("rope", [True, False])
def test_attention_layer_chain_matches_official(rope):
    """wq -> query_scale -> absorb; wkv_a -> rank_pack (no q-LoRA) -> FP8 records;
    sparse MLA over every causal key (scale 128^-0.5) -> values -> wo: vs the
    official layer's eager attention. RoPE layers run the GLM 5.3 656-byte
    contract, NoPE layers GLM 5.3 Flash's 528-byte one."""
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import (ML4, compile_kv_b_aot, compile_query_scale_aot,
                                                         compile_rank_pack_aot,
                                                         compile_sparse_mla_aot, nope_attention, rope_attention)

    T, H = 256, 64
    nope = 64 if rope else 128
    gen = torch.Generator().manual_seed(21)
    w = _attention_weights(rope, gen)
    x = torch.randn(T, 6144, generator=gen).bfloat16()
    positions = torch.arange(T, dtype=torch.int64)
    expected, _, _ = _official_attention(x, positions, w, rope)

    d = lambda t: t.to(device)  # noqa: E731
    g = rope_attention() if rope else nope_attention()
    latent = 576 if rope else 512
    q = d((x.float() @ w["wq"].float().t()).bfloat16().view(T, H, 128).contiguous())
    query = torch.zeros(T, H, latent, dtype=torch.bfloat16, device=device)
    table = d(_cos_sin_table(positions))
    compile_query_scale_aot(ML4, rope=rope, max_rows=T).launch(q, d(positions), table, d(w["s"]), d(w["b"]),
                                                               query if rope else torch.empty(1, dtype=torch.bfloat16, device=device),
                                                               scalars=(T,))
    kv_b = w["wkv_b"].view(H, nope + 128, 512)
    w_uk = d(kv_b[:, :nope, :].transpose(1, 2).contiguous())
    w_uv = d(kv_b[:, nope:, :].contiguous())
    # Absorb reads q_nope [rows, H, nope] with the head stride of the 128-wide q.
    compile_kv_b_aot(ML4, stage="absorb", rope=rope, max_rows=T).launch(q, w_uk, query, scalars=(T,))
    kv = (x.float() @ w["wkv_a"].float().t()).bfloat16()
    pages = T // 64 + 1
    cache = torch.zeros(pages * 64 * g.record_bytes, dtype=torch.uint8, device=device)
    slots = d(torch.arange(T, dtype=torch.int64))
    compile_rank_pack_aot(ML4, rope=rope, max_rows=T).launch(d(kv), d(w["kv_a_norm"]), d(positions), slots, table,
                                                             cache, scalars=(T,))
    mla = compile_sparse_mla_aot(ML4, rope=rope, route="prefill", max_rows=T)
    topk = g.index_topk if rope else g.sparse_topk
    indices = torch.full((T, topk), -1, dtype=torch.int32)
    for row in range(T):
        indices[row, :row + 1] = torch.arange(row + 1)
    lengths = torch.arange(1, T + 1, dtype=torch.int32)
    out = torch.empty(T, H, 512, dtype=torch.bfloat16, device=device)
    mla.launch(query, cache, d(indices), d(lengths), out,
               torch.empty(mla.scratch_bytes(T)["scratch"], dtype=torch.uint8, device=device), scalars=(T,))
    values = torch.empty(T, H, 128, dtype=torch.bfloat16, device=device)
    compile_kv_b_aot(ML4, stage="values", rope=rope, max_rows=T).launch(out, w_uv, values, scalars=(T,))
    torch.cuda.synchronize()
    actual = values.float().reshape(T, -1).cpu() @ w["wo"].float().t()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    print(f"ML4 attention chain rope={rope}: cosine min {float(cosine.min()):.6f} mean {float(cosine.mean()):.6f}")
    assert cosine.min() > 0.995, float(cosine.min())


def _fp8_block_weights(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    # Power-of-two block scales, like the official qscale_weight grid.
    s = torch.exp2(torch.randint(-12, -6, ((n + 127) // 128, k // 128), generator=gen, device="cuda").float())
    grown = s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return w8, s, (w8.float() * grown).bfloat16()


@pytest.mark.parametrize("mode,rows", [("decode", 1), ("decode", 16), ("decode", 40), ("decode", 64),
                                       ("prefill", 300)])
def test_index_q_fp8_equals_the_bf16_projection_of_the_dequantized_weight(mode, rows):
    """ml4_index_q over the shipped FP8 block-128 wq_b: decode (GEMV to 16 rows, W8A16
    above) and prefill W8A16 equal the BF16 projection over bf16(w * s) bitwise, the
    reference's Fp8Dequantize followed by its BF16 linear up to GEMM order; prefill W8A8
    matches its own activation-quantized float reference."""
    device = require_b12x()
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._fp8_weights import RoutedFp8Projection
    from b12x.integration.cuteafd.mistral_large4 import ML4, compile_index_q_fp8_aot

    n, k = ML4.index_heads * ML4.index_head_dim, ML4.hidden
    w8, s, wd = _fp8_block_weights(n, k, rows)
    x = torch.randn((rows, k), device=device).bfloat16()
    program = compile_index_q_fp8_aot(ML4, mode=mode, max_rows=64 if mode == "decode" else 4096)
    scratch = torch.empty(max(program.scratch_bytes(rows)["scratch"], 16), dtype=torch.uint8, device=device)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
    ref = torch.empty_like(out)
    if mode == "decode":
        program.launch(x, w8, s, out, scratch, scalars=(rows,))
        routed = compile_program(RoutedFp8Projection(n, k), name="ml4_iq_ref", operands=(
            Operand("x", torch.bfloat16, "x"), Operand("wb", torch.bfloat16, "wb"),
            Operand("w", torch.float8_e4m3fn, "w"), Operand("s", torch.float32, "s", align=4),
            Operand("o", torch.bfloat16, "o", "out")), scalars=(Scalar("rows"),), key=("ml4_iq_ref",))
        routed.launch(x, wd, w8, s, ref, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
        return
    program.launch(x, w8, s, out, scratch, scalars=(rows, 0))
    bf16 = compile_program(TmaBf16Projection(n, k), name="ml4_iq_bf16", operands=(
        Operand("x", torch.bfloat16, "x"), Operand("w", torch.bfloat16, "w"),
        Operand("o", torch.bfloat16, "o", "out")), scalars=(Scalar("rows"),), key=("ml4_iq_bf16",))
    bf16.launch(x, wd, ref, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(out, ref), "prefill W8A16"
    program.launch(x, w8, s, out, scratch, scalars=(rows, 1))
    torch.cuda.synchronize()
    xb = x.float().view(rows, k // 128, 128)
    amax = xb.abs().amax(-1, keepdim=True)
    sx = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    qx = ((xb / sx).to(torch.float8_e4m3fn).float() * sx).view(rows, k)
    expected = qx @ wd.float().t()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), expected.flatten(), dim=0)
    assert cos > 0.99999, float(cos)


def _ml4_programs():
    from b12x.integration.cuteafd import mistral_large4 as ml4
    g = ml4.ML4
    out = []
    for rope in (True, False):
        out += [lambda rope=rope: ml4.compile_query_scale_aot(g, rope=rope, max_rows=64),
                lambda rope=rope: ml4.compile_rank_pack_aot(g, rope=rope, max_rows=64),
                lambda rope=rope: ml4.compile_kv_b_aot(g, stage="absorb", rope=rope, max_rows=64),
                lambda rope=rope: ml4.compile_index_post_aot(g, rope=rope, max_rows=64)]
    out += [lambda: ml4.compile_kv_b_aot(g, stage="values", rope=True, max_rows=64),
            lambda: ml4.compile_query_scale_aot(g, rope=False, max_rows=64, heads=g.index_heads,
                                                head_dim=g.index_head_dim),
            lambda: ml4.compile_layer_norm_aot(width=g.hidden, max_rows=64, eps=g.index_norm_eps),
            lambda: ml4.compile_index_q_fp8_aot(g, mode="decode", max_rows=64),
            lambda: ml4.compile_index_q_fp8_aot(g, mode="prefill", max_rows=256),
            lambda: ml4.compile_draft_producer_aot(ml4.ML4_DRAFT, max_rows=64),
            lambda: ml4.compile_draft_attention_aot(ml4.ML4_DRAFT, max_rows=64, max_context=8192),
            lambda: ml4.compile_draft_output_aot(ml4.ML4_DRAFT, max_rows=64),
            lambda: ml4.compile_draft_producer_aot(ml4.ML4_DRAFT_TP2, max_rows=64),
            lambda: ml4.compile_draft_attention_aot(ml4.ML4_DRAFT_TP2, max_rows=64, max_context=8192),
            lambda: ml4.compile_draft_output_aot(ml4.ML4_DRAFT_TP2, max_rows=64)]
    return out


@pytest.mark.parametrize("index", range(19))
def test_every_ml4_program_exports_with_its_declared_abi(index, tmp_path):
    """The exporter's header check (pointer names = the declared operands) on
    every ml4 program: launches here pass pointers positionally, so only the
    export sees a call parameter named differently from its operand."""
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header

    thunks = _ml4_programs()
    assert len(thunks) == 19
    with exportable_compilation():
        program = thunks[index]()
    label = f"ml4_export_check_{index}"
    program.export_to_c(str(tmp_path), label, "cuteafd_" + label)
    checked = validate_exported_header(program, tmp_path / f"{label}.h", "cuteafd_" + label)
    assert checked["argument_count"] == len(program.operands) + len(program.scalars) + 1


@pytest.mark.parametrize("kv_cache_requests", [((2100, 3), (65, 3)), ((2048, 3), (1, 3))])
def test_draft_attention_windows_every_block_row_before_its_block(kv_cache_requests):
    """ml4_draft_attention (the shared MLA block-draft attention, block_window):
    each request's 3 block rows see the 2048 records before the block (the
    trained dSpark mask, the same for every row) and the whole block,
    non-causally; ragged requests, FP8 records, CUDA-graph replay bit-identical."""
    device = require_b12x()
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
    from b12x.integration.cuteafd.mistral_large4 import ML4_DRAFT, compile_draft_attention_aot

    g = ML4_DRAFT
    p = compile_draft_attention_aot(g, max_rows=16, max_context=4352)
    contexts = [c for c, _ in kv_cache_requests]
    counts = [n for _, n in kv_cache_requests]
    lengths = [c + n for c, n in kv_cache_requests]
    width = max(-(-length // 64) for length in lengths)
    gen = torch.Generator(device="cuda").manual_seed(91)
    latent = (torch.randn(2 * width * 64, 576, generator=gen, device="cuda") * 0.1).bfloat16()
    packed = pack_mla_kv_cache_reference(latent[:, :512], latent[:, 512:])
    oracle = unpack_mla_kv_cache_reference(packed).bfloat16().reshape(2 * width, 64, 576)
    cache = packed.reshape(2 * width, 64, 656)
    table = torch.arange(2 * width, dtype=torch.int32, device="cuda").flip(0).reshape(width, 2).T.contiguous()
    rows = sum(counts)
    q = (torch.randn(rows, 64, 576, generator=gen, device="cuda") * 0.5).bfloat16()
    cu = torch.tensor([0, counts[0], rows], dtype=torch.int32, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, 64, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(rows, 64, device="cuda")
    scratch = torch.empty(p.scratch_bytes(16)["scratch"], dtype=torch.uint8, device="cuda")
    scale = 128 ** -0.5

    def run():
        p.launch(q, cache, table, lens, cu, out, lse, scratch, scalars=(rows, 2, width, p.geometry["num_splits"], scale))

    run()
    torch.cuda.synchronize()
    begin = 0
    for index, (context, count, length) in enumerate(zip(contexts, counts, lengths)):
        records = oracle[table[index].long()].flatten(0, 1)[:length].float()
        scores = torch.einsum("thd,sd->hts", q[begin:begin + count].float(), records) * scale
        keys = torch.arange(length, device="cuda")
        visible = keys >= max(context - g.window_size, 0)  # every row: the window before the block, plus the block
        scores.masked_fill_(~visible[None, None], -torch.inf)
        expected = torch.einsum("hts,sd->thd", scores.softmax(-1), records[:, :512])
        torch.testing.assert_close(out[begin:begin + count].float(), expected, atol=0.003, rtol=0.03)
        torch.testing.assert_close(lse[begin:begin + count], scores.logsumexp(-1).T, atol=2e-4, rtol=2e-5)
        begin += count
    saved = out.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    for actual, expected in zip((out, lse), saved):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_draft_layer_attention_through_the_shared_mla_draft_path():
    """ml4_draft_producer (context rows, then the block) -> ml4_draft_attention ->
    ml4_draft_output vs the drafter's BF16 attention (dspark.py: plain YaRN
    RoPE, no SSMax, the trained window, non-causal block): one sequence,
    context 2100, block 3."""
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import (ML4_DRAFT, compile_draft_attention_aot,
                                                         compile_draft_output_aot, compile_draft_producer_aot)

    g, H, depth, context = ML4_DRAFT, 64, 3, 2100
    gen = torch.Generator().manual_seed(57)
    w = _attention_weights(True, gen)
    n = context + depth
    x = (torch.randn(n, 6144, generator=gen)).bfloat16()
    positions = torch.arange(n, dtype=torch.int64)
    kv_b = w["wkv_b"].view(H, 64 + 128, 512)
    d = lambda t: t.to(device)  # noqa: E731
    pages = -(-n // 64)
    cache = torch.zeros(pages, 64, 656, dtype=torch.uint8, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device)[None]
    unit_s, unit_b = d(torch.zeros(H, dtype=torch.bfloat16)), d(torch.ones(H, dtype=torch.bfloat16))
    weights = dict(w_q=d(w["wq"]), w_kv_a=d(w["wkv_a"]), kv_a_norm=d(w["kv_a_norm"]),
                   w_uk=d(kv_b[:, :64, :].transpose(1, 2).contiguous()), w_uv=d(kv_b[:, 64:, :].contiguous()),
                   w_o=d(w["wo"]))
    table_cs = d(_cos_sin_table(positions))
    producer = compile_draft_producer_aot(g, max_rows=4096)
    query = torch.zeros(n, H, 576, dtype=torch.bfloat16, device=device)
    scratch = torch.empty(producer.scratch_bytes(4096)["scratch"], dtype=torch.uint8, device=device)
    # Context rows first (their queries are unused), then the block.
    for first, count in ((0, context), (context, depth)):
        producer.launch(d(x[first:first + count]), d(positions[first:first + count]), d(positions[first:first + count]),
                        table_cs, unit_s, unit_b, weights["w_q"], weights["w_kv_a"], weights["kv_a_norm"],
                        weights["w_uk"], cache, query[first:first + count], scratch, scalars=(count,))
    attention = compile_draft_attention_aot(g, max_rows=16, max_context=4352)
    out = torch.empty(depth, H, 512, dtype=torch.bfloat16, device=device)
    lse = torch.empty(depth, H, device=device)
    attention.launch(query[context:].contiguous(), cache, table, torch.tensor([n], dtype=torch.int32, device=device),
                     torch.tensor([0, depth], dtype=torch.int32, device=device), out, lse,
                     torch.empty(attention.scratch_bytes(16)["scratch"], dtype=torch.uint8, device=device),
                     scalars=(depth, 1, pages, attention.geometry["num_splits"], 128 ** -0.5))
    output = compile_draft_output_aot(g, max_rows=16)
    actual = torch.empty(depth, 6144, dtype=torch.bfloat16, device=device)
    output.launch(out, weights["w_uv"], weights["w_o"], actual,
                  torch.empty(output.scratch_bytes(16)["scratch"], dtype=torch.uint8, device=device), scalars=(depth,))
    torch.cuda.synchronize()
    # Reference: dspark.py's draft attention in BF16 (keys of the window and the block).
    angles = positions.float()[:, None] * _inv_freq()[None, :]
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kv = (x.float() @ w["wkv_a"].float().t()).bfloat16()
    lat = kv[:, :512].float()
    lat = (lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16() * w["kv_a_norm"]
    rot = kv[:, 512:]
    k_rot = torch.cat([rot[:, 0::2] * cos - rot[:, 1::2] * sin, rot[:, 1::2] * cos + rot[:, 0::2] * sin], -1)
    keep = torch.arange(n) >= context - 2048
    kvb = (lat[keep].float() @ w["wkv_b"].float().t()).bfloat16().view(-1, H, 192)
    k = torch.cat([kvb[..., :64], k_rot[keep][:, None, :].expand(-1, H, -1)], -1)
    v = kvb[..., 64:]
    q = (x[context:].float() @ w["wq"].float().t()).bfloat16().view(depth, H, 128)
    c, s = cos[context:][:, None], sin[context:][:, None]
    e, o = q[..., 64::2], q[..., 65::2]
    q = torch.cat([q[..., :64], e * c - o * s, o * c + e * s], -1)
    scores = torch.einsum("rhd,khd->hrk", q.float(), k.float()) * 128 ** -0.5
    attn = torch.einsum("hrk,khd->rhd", scores.softmax(-1).bfloat16().float(), v.float()).bfloat16()
    expected = attn.reshape(depth, -1).float() @ w["wo"].float().t()
    cosine = torch.nn.functional.cosine_similarity(actual.float().cpu(), expected, dim=-1)
    print(f"ML4 MLA draft layer attention: cosine min {float(cosine.min()):.6f}")
    assert cosine.min() > 0.995, float(cosine.min())


@pytest.mark.parametrize("rope", [True, False])
def test_head_split_context_query_equals_the_whole_query_of_its_heads(rope):
    """Each 32-head share (``ml4_2``) writes its heads' absorbed query, RoPE tail included,
    into the combined two-GPU query (``_context``: token rows 2 x 32 x latent apart). Both
    shares together equal the 64-head programs' query exactly; the stride never moves bytes
    outside a share's heads."""
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import ML4, ML4_TP2, compile_kv_b_aot, compile_query_scale_aot

    rows, heads, latent = 40, ML4.heads, 512 + (64 if rope else 0)
    nope = ML4.nope_dim(rope)
    gen = torch.Generator().manual_seed(21)
    q = (torch.randn(rows, heads, 128, generator=gen) * 2).bfloat16()
    s = (torch.rand(heads, generator=gen) * 0.3).bfloat16()
    b = (0.8 + torch.rand(heads, generator=gen) * 0.4).bfloat16()
    w_uk = (torch.randn(heads, 512, nope, generator=gen) * 0.05).bfloat16()
    positions = _positions(rows, "cpu")
    d = lambda t: t.to(device)
    table = d(_cos_sin_table(positions))

    whole_q, whole = d(q), torch.full((rows, heads, latent), float("nan"), dtype=torch.bfloat16, device=device)
    compile_query_scale_aot(ML4, rope=rope, max_rows=64).launch(
        whole_q, d(positions), table, d(s), d(b), whole if rope else d(torch.empty(1, dtype=torch.bfloat16)),
        scalars=(rows,))
    compile_kv_b_aot(ML4, stage="absorb", rope=rope, max_rows=64).launch(whole_q, d(w_uk), whole, scalars=(rows,))

    row = heads * latent
    combined = torch.full((rows, heads, latent), float("nan"), dtype=torch.bfloat16, device=device)
    scale = (compile_query_scale_aot(ML4_TP2, rope=True, max_rows=64, query_row=row) if rope
             else compile_query_scale_aot(ML4_TP2, rope=False, max_rows=64))
    absorb = compile_kv_b_aot(ML4_TP2, stage="absorb", rope=rope, max_rows=64, query_row=row)
    half = heads // 2
    for share in (0, 1):
        sl = slice(share * half, (share + 1) * half)
        part_q = d(q[:, sl].contiguous())
        out = combined[:, sl]
        scale.launch(part_q, d(positions), table, d(s[sl].contiguous()), d(b[sl].contiguous()),
                     out if rope else d(torch.empty(1, dtype=torch.bfloat16)), scalars=(rows,))
        absorb.launch(part_q, d(w_uk[sl].contiguous()), out, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(combined.cpu(), whole.cpu())


def test_head_split_context_strides_refuse_misuse():
    require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import ML4_TP2, compile_kv_b_aot, compile_query_scale_aot
    with pytest.raises(ValueError):
        compile_query_scale_aot(ML4_TP2, rope=False, max_rows=64, query_row=2 * 32 * 512)
    with pytest.raises(ValueError):
        compile_query_scale_aot(ML4_TP2, rope=True, max_rows=64, query_row=32 * 576 - 576)
    with pytest.raises(ValueError):
        compile_kv_b_aot(ML4_TP2, stage="values", rope=True, max_rows=64, query_row=2 * 32 * 128)


@pytest.mark.parametrize("kv_cache_requests", [((2100, 3), (65, 3)), ((2048, 3), (1, 3))])
def test_draft_tp2_attention_matches_whole_and_replays_bit_identically(kv_cache_requests):
    """ml4_draft2_attention (32 heads, block_window): each 32-head share matches the
    corresponding head slice of 64-head ml4_draft_attention bit-for-bit, matches the
    PyTorch block_window oracle, and replays identically under CUDA graphs."""
    device = require_b12x()
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
    from b12x.integration.cuteafd.mistral_large4 import (ML4_DRAFT, ML4_DRAFT_TP2, compile_draft_attention_aot)

    p_whole = compile_draft_attention_aot(ML4_DRAFT, max_rows=16, max_context=4352)
    p_tp2 = compile_draft_attention_aot(ML4_DRAFT_TP2, max_rows=16, max_context=4352)
    assert p_tp2.name == "ml4_draft2_attention"
    contexts = [c for c, _ in kv_cache_requests]
    counts = [n for _, n in kv_cache_requests]
    lengths = [c + n for c, n in kv_cache_requests]
    width = max(-(-length // 64) for length in lengths)
    gen = torch.Generator(device="cuda").manual_seed(91)
    latent = (torch.randn(2 * width * 64, 576, generator=gen, device="cuda") * 0.1).bfloat16()
    packed = pack_mla_kv_cache_reference(latent[:, :512], latent[:, 512:])
    oracle = unpack_mla_kv_cache_reference(packed).bfloat16().reshape(2 * width, 64, 576)
    cache = packed.reshape(2 * width, 64, 656)
    table = torch.arange(2 * width, dtype=torch.int32, device="cuda").flip(0).reshape(width, 2).T.contiguous()
    rows = sum(counts)
    q = (torch.randn(rows, 64, 576, generator=gen, device="cuda") * 0.5).bfloat16()
    cu = torch.tensor([0, counts[0], rows], dtype=torch.int32, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    scale = 128 ** -0.5

    out_whole = torch.empty(rows, 64, 512, dtype=torch.bfloat16, device="cuda")
    lse_whole = torch.empty(rows, 64, device="cuda")
    p_whole.launch(q, cache, table, lens, cu, out_whole, lse_whole,
                   torch.empty(p_whole.scratch_bytes(16)["scratch"], dtype=torch.uint8, device="cuda"),
                   scalars=(rows, 2, width, p_whole.geometry["num_splits"], scale))

    out_parts, lse_parts = [], []
    scratch_tp2 = torch.empty(p_tp2.scratch_bytes(16)["scratch"], dtype=torch.uint8, device="cuda")
    for rank in (0, 1):
        sl = slice(rank * 32, (rank + 1) * 32)
        q_part = q[:, sl].contiguous()
        out_p = torch.empty(rows, 32, 512, dtype=torch.bfloat16, device="cuda")
        lse_p = torch.empty(rows, 32, device="cuda")

        def run(q_part=q_part, out_p=out_p, lse_p=lse_p):
            p_tp2.launch(q_part, cache, table, lens, cu, out_p, lse_p, scratch_tp2,
                         scalars=(rows, 2, width, p_tp2.geometry["num_splits"], scale))

        run()
        torch.cuda.synchronize()
        saved = out_p.clone(), lse_p.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        for actual, expected in zip((out_p, lse_p), saved):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        out_parts.append(out_p)
        lse_parts.append(lse_p)

    out_cat = torch.cat(out_parts, dim=1)
    lse_cat = torch.cat(lse_parts, dim=1)
    assert torch.equal(out_cat.view(torch.uint8), out_whole.view(torch.uint8))
    assert torch.equal(lse_cat.view(torch.uint8), lse_whole.view(torch.uint8))
    begin = 0
    for index, (context, count, length) in enumerate(zip(contexts, counts, lengths)):
        records = oracle[table[index].long()].flatten(0, 1)[:length].float()
        scores = torch.einsum("thd,sd->hts", q[begin:begin + count].float(), records) * scale
        keys = torch.arange(length, device="cuda")
        visible = keys >= max(context - ML4_DRAFT_TP2.window_size, 0)
        scores.masked_fill_(~visible[None, None], -torch.inf)
        expected = torch.einsum("hts,sd->thd", scores.softmax(-1), records[:, :512])
        torch.testing.assert_close(out_cat[begin:begin + count].float(), expected, atol=0.003, rtol=0.03)
        torch.testing.assert_close(lse_cat[begin:begin + count], scores.logsumexp(-1).T, atol=2e-4, rtol=2e-5)
        begin += count


def test_draft_tp2_layer_sums_to_whole_draft_layer_and_reference():
    """Two 32-head TP2 ranks running ml4_draft2_{producer,attention,output} produce
    bit-identical FP8 KV records on both ranks, match 64-head ml4_draft_* query and
    attention per head slice, and sum their o_proj partials to the whole 64-head
    output (cosine > 0.9999) and the BF16 dspark.py reference (cosine > 0.995)."""
    device = require_b12x()
    from b12x.integration.cuteafd.mistral_large4 import (ML4_DRAFT, ML4_DRAFT_TP2, compile_draft_attention_aot,
                                                         compile_draft_output_aot, compile_draft_producer_aot)

    H, half, depth, context = 64, 32, 3, 2100
    gen = torch.Generator().manual_seed(57)
    w = _attention_weights(True, gen)
    n = context + depth
    x = (torch.randn(n, 6144, generator=gen)).bfloat16()
    positions = torch.arange(n, dtype=torch.int64)
    kv_b = w["wkv_b"].view(H, 64 + 128, 512)
    d = lambda t: t.to(device)  # noqa: E731
    pages = -(-n // 64)
    table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    table_cs = d(_cos_sin_table(positions))
    w_q_full = w["wq"].view(H, 128, 6144)
    w_uk_full = kv_b[:, :64, :].transpose(1, 2).contiguous()
    w_uv_full = kv_b[:, 64:, :].contiguous()
    w_o_full = w["wo"].view(6144, H, 128)

    # Whole 64-head path
    p_prod_w = compile_draft_producer_aot(ML4_DRAFT, max_rows=4096)
    p_attn_w = compile_draft_attention_aot(ML4_DRAFT, max_rows=16, max_context=4352)
    p_out_w = compile_draft_output_aot(ML4_DRAFT, max_rows=16)
    cache_w = torch.zeros(pages, 64, 656, dtype=torch.uint8, device=device)
    query_w = torch.zeros(n, H, 576, dtype=torch.bfloat16, device=device)
    scratch_pw = torch.empty(p_prod_w.scratch_bytes(4096)["scratch"], dtype=torch.uint8, device=device)
    unit_s_w, unit_b_w = d(torch.zeros(H, dtype=torch.bfloat16)), d(torch.ones(H, dtype=torch.bfloat16))
    for first, count in ((0, context), (context, depth)):
        p_prod_w.launch(d(x[first:first + count]), d(positions[first:first + count]), d(positions[first:first + count]),
                        table_cs, unit_s_w, unit_b_w, d(w["wq"]), d(w["wkv_a"]), d(w["kv_a_norm"]),
                        d(w_uk_full), cache_w, query_w[first:first + count], scratch_pw, scalars=(count,))
    attn_w = torch.empty(depth, H, 512, dtype=torch.bfloat16, device=device)
    lse_w = torch.empty(depth, H, device=device)
    p_attn_w.launch(query_w[context:].contiguous(), cache_w, table,
                    torch.tensor([n], dtype=torch.int32, device=device),
                    torch.tensor([0, depth], dtype=torch.int32, device=device), attn_w, lse_w,
                    torch.empty(p_attn_w.scratch_bytes(16)["scratch"], dtype=torch.uint8, device=device),
                    scalars=(depth, 1, pages, p_attn_w.geometry["num_splits"], 128 ** -0.5))
    actual_w = torch.empty(depth, 6144, dtype=torch.bfloat16, device=device)
    p_out_w.launch(attn_w, d(w_uv_full), d(w["wo"]), actual_w,
                   torch.empty(p_out_w.scratch_bytes(16)["scratch"], dtype=torch.uint8, device=device),
                   scalars=(depth,))

    # Split 32-head TP2 path (rank 0 and rank 1)
    p_prod_2 = compile_draft_producer_aot(ML4_DRAFT_TP2, max_rows=4096)
    p_attn_2 = compile_draft_attention_aot(ML4_DRAFT_TP2, max_rows=16, max_context=4352)
    p_out_2 = compile_draft_output_aot(ML4_DRAFT_TP2, max_rows=16)
    assert (p_prod_2.name, p_attn_2.name, p_out_2.name) == (
        "ml4_draft2_producer", "ml4_draft2_attention", "ml4_draft2_output")
    scratch_p2 = torch.empty(p_prod_2.scratch_bytes(4096)["scratch"], dtype=torch.uint8, device=device)
    scratch_a2 = torch.empty(p_attn_2.scratch_bytes(16)["scratch"], dtype=torch.uint8, device=device)
    scratch_o2 = torch.empty(p_out_2.scratch_bytes(16)["scratch"], dtype=torch.uint8, device=device)
    unit_s_2, unit_b_2 = d(torch.zeros(half, dtype=torch.bfloat16)), d(torch.ones(half, dtype=torch.bfloat16))
    caches, queries, attns, lses, partials = [], [], [], [], []
    for rank in (0, 1):
        sl = slice(rank * half, (rank + 1) * half)
        cache_r = torch.zeros(pages, 64, 656, dtype=torch.uint8, device=device)
        query_r = torch.zeros(n, half, 576, dtype=torch.bfloat16, device=device)
        w_q_r = d(w_q_full[sl].reshape(half * 128, 6144).contiguous())
        w_uk_r = d(w_uk_full[sl].contiguous())
        w_uv_r = d(w_uv_full[sl].contiguous())
        w_o_r = d(w_o_full[:, sl, :].reshape(6144, half * 128).contiguous())
        for first, count in ((0, context), (context, depth)):
            p_prod_2.launch(d(x[first:first + count]), d(positions[first:first + count]),
                            d(positions[first:first + count]), table_cs, unit_s_2, unit_b_2, w_q_r,
                            d(w["wkv_a"]), d(w["kv_a_norm"]), w_uk_r, cache_r, query_r[first:first + count],
                            scratch_p2, scalars=(count,))
        attn_r = torch.empty(depth, half, 512, dtype=torch.bfloat16, device=device)
        lse_r = torch.empty(depth, half, device=device)
        p_attn_2.launch(query_r[context:].contiguous(), cache_r, table,
                        torch.tensor([n], dtype=torch.int32, device=device),
                        torch.tensor([0, depth], dtype=torch.int32, device=device), attn_r, lse_r,
                        scratch_a2, scalars=(depth, 1, pages, p_attn_2.geometry["num_splits"], 128 ** -0.5))
        part_r = torch.empty(depth, 6144, dtype=torch.bfloat16, device=device)
        p_out_2.launch(attn_r, w_uv_r, w_o_r, part_r, scratch_o2, scalars=(depth,))
        caches.append(cache_r)
        queries.append(query_r)
        attns.append(attn_r)
        lses.append(lse_r)
        partials.append(part_r)
    torch.cuda.synchronize()

    # Replicated FP8 KV records are bit-identical across both ranks and match the 64-head producer.
    assert torch.equal(caches[0], caches[1])
    assert torch.equal(caches[0], cache_w)
    # Absorbed queries and attention outputs match 64-head slices closely.
    q_cos = torch.nn.functional.cosine_similarity(
        torch.cat(queries, dim=1)[context:].float().flatten(), query_w[context:].float().flatten(), dim=0)
    a_cos = torch.nn.functional.cosine_similarity(
        torch.cat(attns, dim=1).float().flatten(), attn_w.float().flatten(), dim=0)
    # Reduced TP2 BF16 sum matches 64-head output.
    reduced = (partials[0] + partials[1]).float().cpu()
    whole_cos = torch.nn.functional.cosine_similarity(reduced, actual_w.float().cpu(), dim=-1)
    assert float(q_cos) > 0.99999, float(q_cos)
    assert float(a_cos) > 0.99999, float(a_cos)
    assert float(whole_cos.min()) > 0.9999, float(whole_cos.min())

    # Compare against PyTorch BF16 reference (dspark.py).
    angles = positions.float()[:, None] * _inv_freq()[None, :]
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kv = (x.float() @ w["wkv_a"].float().t()).bfloat16()
    lat = kv[:, :512].float()
    lat = (lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16() * w["kv_a_norm"]
    rot = kv[:, 512:]
    k_rot = torch.cat([rot[:, 0::2] * cos - rot[:, 1::2] * sin, rot[:, 1::2] * cos + rot[:, 0::2] * sin], -1)
    keep = torch.arange(n) >= context - 2048
    kvb = (lat[keep].float() @ w["wkv_b"].float().t()).bfloat16().view(-1, H, 192)
    k = torch.cat([kvb[..., :64], k_rot[keep][:, None, :].expand(-1, H, -1)], -1)
    v = kvb[..., 64:]
    q = (x[context:].float() @ w["wq"].float().t()).bfloat16().view(depth, H, 128)
    c, s = cos[context:][:, None], sin[context:][:, None]
    e, o = q[..., 64::2], q[..., 65::2]
    q = torch.cat([q[..., :64], e * c - o * s, o * c + e * s], -1)
    scores = torch.einsum("rhd,khd->hrk", q.float(), k.float()) * 128 ** -0.5
    attn_ref = torch.einsum("hrk,khd->rhd", scores.softmax(-1).bfloat16().float(), v.float()).bfloat16()
    expected = attn_ref.reshape(depth, -1).float() @ w["wo"].float().t()
    ref_cos = torch.nn.functional.cosine_similarity(reduced, expected, dim=-1)
    print(f"ML4 TP2 draft layer: q_cos={float(q_cos):.6f} a_cos={float(a_cos):.6f} "
          f"vs_whole_min={float(whole_cos.min()):.6f} vs_ref_min={float(ref_cos.min()):.6f}")
    assert float(ref_cos.min()) > 0.995, float(ref_cos.min())
