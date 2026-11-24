"""Exact GPU oracle tests for all Hy4-flash CuTe AOT kernels against IEEE-F32 reference math."""
from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from tests._reference.helpers import require_b12x

torch.set_float32_matmul_precision("highest")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

from b12x.integration.cuteafd.hy4_ihc import (
    HY4_FLASH_IHC,
    compile_hy4_hc_head_aot,
    compile_hy4_ihc_post_aot,
    compile_hy4_ihc_post_pre_aot,
    compile_hy4_ihc_pre_aot,
)
from b12x.integration.cuteafd.hy4_engram import (
    HY4_FLASH_ENGRAM,
    compile_hy4_engram_gate_add_aot,
    compile_hy4_engram_head_norm_aot,
)
from b12x.integration.cuteafd.context_split import (
    compile_lse_combine2_aot,
    compile_paged_staging_gather_aot,
)
from b12x.integration.cuteafd.hy4_mla import (
    Hy4MlaGeometry,
    compile_hy4_mla_attention_aot,
    compile_hy4_mla_dsa_partial_aot,
    compile_hy4_mla_gate_uv_aot,
    compile_hy4_mla_q_absorb_aot,
    compile_hy4_mla_rank_norm_pack_aot,
    compile_hy4_mla_sink_gate_uv_aot,
    hy4_attention_layout,
    hy4_dsa_partial_scratch_bytes,
)
from b12x.integration.cuteafd.hy4_indexer import (
    HY4_FLASH_INDEXER,
    compile_hy4_indexer_expand_slots_aot,
    compile_hy4_indexer_merge_expand_owned_aot,
    compile_hy4_indexer_pool_group_aot,
    compile_hy4_indexer_query_rope_aot,
    compile_hy4_indexer_score_groups_aot,
    compile_hy4_indexer_scored_topk_aot,
    compile_hy4_indexer_topk_expand_aot,
)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)) * w.float()


def _interleaved_rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    width = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, width, 2, dtype=torch.float32, device=x.device) / width))
    freq = pos.float().unsqueeze(-1) * inv
    cos = torch.cos(freq).repeat_interleave(2, dim=-1)
    sin = torch.sin(freq).repeat_interleave(2, dim=-1)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    xf = x.float()
    alt = torch.stack((-xf[..., 1::2], xf[..., ::2]), dim=-1).flatten(-2)
    return xf * cos + alt * sin


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_hy4_ihc_and_hc_head_exact_oracle(rows: int):
    dev = require_b12x()
    torch.manual_seed(42 + rows)
    h = 3072
    r = torch.randn(rows, 4, h, dtype=torch.bfloat16, device=dev) * 0.5
    delta = torch.randn(rows, h, dtype=torch.bfloat16, device=dev) * 0.5
    fn = torch.randn(8, 4 * h, dtype=torch.float32, device=dev) * 0.02
    scale = torch.tensor([0.3, 0.7], dtype=torch.float32, device=dev)
    base = torch.randn(8, dtype=torch.float32, device=dev) * 0.2
    norm = torch.randn(h, dtype=torch.bfloat16, device=dev) + 1.0

    # Reference iHC pre (true IEEE F64 oracle to eliminate any cuBLAS TF32/F32 drift)
    flat64 = r.double().reshape(rows, 4 * h)
    inv = torch.rsqrt(flat64.square().mean(-1, keepdim=True) + 1e-5)
    mix = (flat64 @ fn.double().T) * inv
    pre_ref = (torch.sigmoid(mix[:, :4] * scale[0].double() + base[:4].double()) + 1e-6).float()
    post_ref = (2.0 * torch.sigmoid(mix[:, 4:] * scale[1].double() + base[4:].double()) + 1e-6).float()
    flat = r.float().reshape(rows, 4 * h)
    inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    col_ref = (r.float() * pre_ref.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
    y_ref = _rms_norm(col_ref, norm, 1e-5).to(torch.bfloat16)

    prog_pre = compile_hy4_ihc_pre_aot(HY4_FLASH_IHC)
    post_out = torch.empty(rows, 4, dtype=torch.float32, device=dev)
    y_out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_pre.launch(r, fn, scale, base, norm, post_out, y_out, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.allclose(post_out, post_ref, atol=1e-5, rtol=1e-5)
    assert (y_out.float() - y_ref.float()).abs().max().item() <= 0.03

    # Reference iHC post (GPU uses FMA vs PyTorch mul+add -> at most 1 BF16 ULP)
    prog_post = compile_hy4_ihc_post_aot(HY4_FLASH_IHC)
    r_after = torch.empty_like(r)
    prog_post.launch(delta, r, post_ref, r_after, scalars=(rows,))
    r_after_ref = (post_ref.unsqueeze(-1) * delta.float().unsqueeze(1) + r.float()).to(torch.bfloat16)
    torch.cuda.synchronize()
    assert (r_after.float() - r_after_ref.float()).abs().max().item() <= 0.02
    assert (r_after.view(torch.int16).int() - r_after_ref.view(torch.int16).int()).abs().max().item() <= 1

    # Fused post_pre == post then pre
    prog_pp = compile_hy4_ihc_post_pre_aot(HY4_FLASH_IHC)
    r_pp = torch.empty_like(r)
    post_pp = torch.empty(rows, 4, dtype=torch.float32, device=dev)
    y_pp = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_pp.launch(delta, r, post_ref, fn, scale, base, norm, r_pp, post_pp, y_pp, scalars=(rows,))

    post_seq = torch.empty(rows, 4, dtype=torch.float32, device=dev)
    y_seq = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_pre.launch(r_after, fn, scale, base, norm, post_seq, y_seq, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(r_pp, r_after)
    assert torch.allclose(post_pp, post_seq, atol=1e-5, rtol=1e-5)
    assert torch.equal(y_pp, y_seq)

    # Terminal hc_head
    hfn = torch.randn(4, 4 * h, dtype=torch.float32, device=dev) * 0.02
    hscale = torch.tensor([0.5], dtype=torch.float32, device=dev)
    hbase = torch.randn(4, dtype=torch.float32, device=dev) * 0.2
    prog_head = compile_hy4_hc_head_aot(HY4_FLASH_IHC, store_collapsed=True)
    col_h = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    out_h = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_head.launch(r, hfn, hscale, hbase, norm, col_h, out_h, scalars=(rows,))
    mix_h = F.linear(flat, hfn) * inv
    pre_h = torch.sigmoid(mix_h * hscale + hbase) + 1e-6
    col_href = (r.float() * pre_h.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
    out_href = _rms_norm(col_href, norm, 1e-5).to(torch.bfloat16)
    torch.cuda.synchronize()
    assert (out_h.float() - out_href.float()).abs().max().item() <= 0.03


def test_hy4_engram_kernels_exact_oracle():
    dev = require_b12x()
    torch.manual_seed(7)
    rows, h = 11, 3072
    eps = float(torch.finfo(torch.float32).eps)

    raw = torch.randn(rows, 16, 96, dtype=torch.bfloat16, device=dev)
    w_hn = torch.randn(16, 96, dtype=torch.bfloat16, device=dev) + 1.0
    out_hn = torch.empty(rows, 16 * 96, dtype=torch.bfloat16, device=dev)
    compile_hy4_engram_head_norm_aot(HY4_FLASH_ENGRAM).launch(raw, w_hn, out_hn, scalars=(rows,))
    ref_hn = torch.cat([_rms_norm(raw[:, j], w_hn[j], eps) for j in range(16)], dim=-1).to(torch.bfloat16)
    torch.cuda.synchronize()
    assert (out_hn.float() - ref_hn.float()).abs().max().item() <= 0.02

    res = torch.randn(rows, 4, h, dtype=torch.bfloat16, device=dev) * 0.5
    keys = torch.randn(rows, 4, h, dtype=torch.bfloat16, device=dev) * 0.5
    key_bias = torch.randn(4, h, dtype=torch.bfloat16, device=dev) * 0.25
    val = torch.randn(rows, h, dtype=torch.bfloat16, device=dev) * 0.5
    val_bias = torch.randn(h, dtype=torch.bfloat16, device=dev) * 0.25
    nk = torch.randn(4, h, dtype=torch.bfloat16, device=dev) + 1.0
    nq = torch.randn(4, h, dtype=torch.bfloat16, device=dev) + 1.0
    pg = torch.randn(h, dtype=torch.bfloat16, device=dev) + 1.0
    hs = torch.randn(4, h, dtype=torch.bfloat16, device=dev) * 0.2
    masked = torch.zeros(rows, dtype=torch.uint8, device=dev)
    masked[3] = 1
    masked[8] = 1

    # Reference
    vf = val.float() + val_bias.float()
    vn = pg.float() * (vf * torch.rsqrt(vf.square().mean(-1, keepdim=True) + eps))
    ref = res.float().clone()
    for j in range(4):
        kj = _rms_norm(keys[:, j].float() + key_bias[j].float(), nk[j], eps)
        qj = _rms_norm(res[:, j], nq[j], eps)
        g = (kj * qj).sum(-1) * (h ** -0.5)
        gate = torch.sigmoid(g.abs().clamp_min(1e-6).sqrt() * g.sign())
        delta = gate.unsqueeze(-1) * vn + hs[j].float() * vn
        delta = delta.masked_fill(masked.bool().unsqueeze(-1), 0.0)
        ref[:, j] = (res[:, j].float() + delta).to(torch.bfloat16).float()

    res_gpu = res.clone()
    compile_hy4_engram_gate_add_aot(HY4_FLASH_ENGRAM).launch(
        res_gpu, keys, key_bias, val, val_bias, nk, nq, pg, hs, masked, scalars=(rows,)
    )
    torch.cuda.synchronize()
    assert torch.equal(res_gpu[3], res[3])
    assert torch.equal(res_gpu[8], res[8])
    assert (res_gpu.float() - ref).abs().max().item() <= 0.03


@pytest.mark.parametrize("active_splits", [1, 4])
def test_hy4_mla_long_dsa_dynamic_splits(active_splits: int):
    """Verify dynamic_split_chunks covers all 384 (> 2*64) tokens even at active_splits=1."""
    dev = require_b12x()
    torch.manual_seed(777)
    seq_len = 384
    g = Hy4MlaGeometry(heads=16, window_size=None, kv_cache="fp8")
    q = torch.randn(1, 16, 576, dtype=torch.bfloat16, device=dev) * 0.2
    kv_a = torch.randn(seq_len, 576, dtype=torch.bfloat16, device=dev) * 0.2
    q_a = torch.zeros(seq_len, 1024, dtype=torch.bfloat16, device=dev)
    q_norm = torch.ones(1024, dtype=torch.bfloat16, device=dev)
    kv_norm = torch.ones(512, dtype=torch.bfloat16, device=dev)
    inv_freq = (1.0 / (10_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()
    positions = torch.arange(seq_len, dtype=torch.int64, device=dev)
    slots = torch.arange(seq_len, dtype=torch.int64, device=dev)
    cache = torch.zeros(6, 64, 656, dtype=torch.uint8, device=dev)
    q_resid = torch.empty(seq_len, 1024, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_rank_norm_pack_aot(g).launch(
        q_a, kv_a, q_norm, kv_norm, slots, positions, inv_freq, q_resid, cache, scalars=(seq_len,)
    )
    prog = compile_hy4_mla_attention_aot(g, max_rows=16, max_context=2048, mode="decode", page_size=1)
    scratch = torch.empty(max(1024, hy4_attention_layout(g, 16, max_context=2048, mode="decode")["scratch_bytes"]),
                          dtype=torch.uint8, device=dev)
    pt = torch.arange(seq_len, dtype=torch.int32, device=dev).unsqueeze(0)
    seqlens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    out = torch.empty(1, 16, 512, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(1, 16, dtype=torch.float32, device=dev)
    prog.launch(q, cache, pt, seqlens, cu, out, lse, scratch, scalars=(1, 1, seq_len, active_splits))
    torch.cuda.synchronize()
    k0 = _rms_norm(kv_a[:, :512], kv_norm, 1e-5)
    kr = _interleaved_rope(kv_a[:, 512:], positions, 10_000_000.0)
    k_full = torch.cat([k0, kr], dim=-1)  # [T, 576]
    scores = torch.einsum("hd,td->ht", q[0].float(), k_full) * (256.0 ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, k0)
    assert (out[0].float() - ref).abs().max().item() <= 0.05


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("heads,window", [(16, None), (32, None), (32, 512), (64, 512)])
def test_hy4_mla_end_to_end_oracle(kv_cache: str, heads: int, window: int | None):
    dev = require_b12x()
    torch.manual_seed(123)
    seq_len = 96
    g = Hy4MlaGeometry(heads=heads, window_size=window, kv_cache=kv_cache)
    theta = 10_000.0 if window is not None else 10_000_000.0
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    q_a = torch.randn(seq_len, 1024, dtype=torch.bfloat16, device=dev) * 0.3
    kv_a = torch.randn(seq_len, 576, dtype=torch.bfloat16, device=dev) * 0.3
    q_norm = torch.randn(1024, dtype=torch.bfloat16, device=dev) + 1.0
    kv_norm = torch.randn(512, dtype=torch.bfloat16, device=dev) + 1.0
    q_raw = torch.randn(seq_len, heads, 256, dtype=torch.bfloat16, device=dev) * 0.3
    w_uk = torch.randn(heads, 512, 192, dtype=torch.bfloat16, device=dev) * 0.05
    w_uv = torch.randn(heads, 256, 512, dtype=torch.bfloat16, device=dev) * 0.05
    gate = torch.randn(seq_len, heads, 256, dtype=torch.bfloat16, device=dev) * 0.5
    sink = torch.randn(heads, dtype=torch.float32, device=dev) * 0.5

    positions = torch.arange(seq_len, dtype=torch.int64, device=dev)
    slots = torch.arange(seq_len, dtype=torch.int64, device=dev)
    pages = 4
    cache = torch.zeros(pages, 64, 656 if kv_cache == "fp8" else 576,
                        dtype=torch.uint8 if kv_cache == "fp8" else torch.bfloat16, device=dev)
    q_resid = torch.empty(seq_len, 1024, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_rank_norm_pack_aot(g).launch(
        q_a, kv_a, q_norm, kv_norm, slots, positions, inv_freq, q_resid, cache, scalars=(seq_len,)
    )

    query = torch.empty(seq_len, heads, 576, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_q_absorb_aot(g).launch(q_raw, w_uk, positions, inv_freq, query, scalars=(seq_len,))

    # Attention: page_size=64 for SWA, page_size=1 selected-token gather for DSA
    ps = 64 if window is not None else 1
    prog_attn = compile_hy4_mla_attention_aot(g, max_rows=seq_len, max_context=2048, mode="decode", page_size=ps)
    scratch = torch.empty(max(1024, hy4_attention_layout(g, seq_len, max_context=2048, mode="decode")["scratch_bytes"]),
                          dtype=torch.uint8, device=dev)
    attn_out = torch.empty(seq_len, heads, 512, dtype=torch.bfloat16, device=dev)
    lse_out = torch.empty(seq_len, heads, dtype=torch.float32, device=dev)
    if ps == 64:
        pt = torch.arange(pages, dtype=torch.int32, device=dev).unsqueeze(0)
        seqlens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
        prog_attn.launch(query, cache, pt, seqlens, cu, attn_out, lse_out, scratch, scalars=(seq_len, 1, pages, 1))
    else:
        pt = torch.full((seq_len, seq_len), -1, dtype=torch.int32, device=dev)
        for r in range(seq_len):
            pt[r, :r + 1] = torch.arange(r + 1, dtype=torch.int32, device=dev)
        seqlens = torch.arange(1, seq_len + 1, dtype=torch.int32, device=dev)
        cu = torch.arange(seq_len + 1, dtype=torch.int32, device=dev)
        prog_attn.launch(query, cache, pt, seqlens, cu, attn_out, lse_out, scratch, scalars=(seq_len, seq_len, seq_len, 1))

    uv_out = torch.empty(seq_len, heads, 256, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_sink_gate_uv_aot(g).launch(attn_out, w_uv, gate, lse_out, sink, uv_out, scalars=(seq_len,))
    torch.cuda.synchronize()

    # Exact F32 unabsorbed MLA reference with F32 sinks + sigmoid gate
    k0_ref = _rms_norm(kv_a[:, :512], kv_norm, 1e-5)
    kr_ref = _interleaved_rope(kv_a[:, 512:], positions, theta)
    qn_ref = q_raw[:, :, :192].float()
    qr_ref = _interleaved_rope(q_raw[:, :, 192:], positions, theta)
    k_nope = torch.einsum("tc,hcd->thd", k0_ref, w_uk.float())
    v_full = torch.einsum("tc,hvc->thv", k0_ref, w_uv.float())
    q_full = torch.cat([qn_ref, qr_ref], dim=-1).transpose(0, 1)  # [H, T, 256]
    k_full = torch.cat([k_nope, kr_ref.unsqueeze(1).expand(-1, heads, -1)], dim=-1).transpose(0, 1)
    v_full = v_full.transpose(0, 1)
    scores = torch.matmul(q_full, k_full.transpose(-1, -2)) * (256.0 ** -0.5)
    idx = torch.arange(seq_len, device=dev)
    dist = idx[:, None] - idx[None, :]
    allowed = dist >= 0
    if window is not None:
        allowed &= dist < window
    scores = scores.masked_fill(~allowed.unsqueeze(0), -1e30)
    scores = torch.cat([scores, sink[:, None, None].expand(heads, seq_len, 1)], dim=-1)
    probs = torch.softmax(scores, dim=-1)[..., :-1]
    ref_uv = torch.matmul(probs, v_full).transpose(0, 1) * torch.sigmoid(gate.float())
    tol = 0.08 if kv_cache == "fp8" else 0.04
    assert (uv_out.float() - ref_uv).abs().max().item() <= tol


def test_hy4_indexer_oracle():
    dev = require_b12x()
    torch.manual_seed(99)
    g = HY4_FLASH_INDEXER
    theta = 10_000_000.0
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    groups = 32  # 128 tokens = 2 pages
    tokens = torch.randn(groups, 4, 128, dtype=torch.bfloat16, device=dev) * 0.4
    w_kv = torch.randn(128, 128, dtype=torch.float32, device=dev) * 0.1
    w_kg = torch.randn(128, 128, dtype=torch.float32, device=dev) * 0.1
    ape = torch.randn(4, 128, dtype=torch.float32, device=dev) * 0.2
    key_norm = torch.randn(128, dtype=torch.float32, device=dev) + 1.0
    g_pos = torch.arange(groups, dtype=torch.int64, device=dev) * 4
    g_slots = torch.arange(groups, dtype=torch.int64, device=dev)
    cache = torch.zeros(4, g.page_bytes, dtype=torch.uint8, device=dev)

    compile_hy4_indexer_pool_group_aot(g).launch(
        tokens, w_kv, w_kg, ape, key_norm, g_pos, g_slots, inv_freq, cache, scalars=(groups,)
    )

    rows = 8
    q_raw = torch.randn(rows, 16, 128, dtype=torch.bfloat16, device=dev) * 0.4
    w_raw = torch.randn(rows, 16, dtype=torch.bfloat16, device=dev) * 0.4
    positions = torch.tensor([16, 20, 40, 63, 64, 90, 120, 127], dtype=torch.int64, device=dev)
    q_rot = torch.empty_like(q_raw)
    w_out = torch.empty(rows, 16, dtype=torch.float32, device=dev)
    compile_hy4_indexer_query_rope_aot(g).launch(q_raw, w_raw, positions, inv_freq, q_rot, w_out, scalars=(rows,))

    pt = torch.tensor([[0, 1, 2, 3]] * rows, dtype=torch.int32, device=dev)
    limits = (positions // 4).to(torch.int32)
    scores = torch.empty(rows, groups, dtype=torch.float32, device=dev)
    compile_hy4_indexer_score_groups_aot(g).launch(q_rot, w_out, cache, pt, limits, scores, scalars=(rows, groups, 4))
    torch.cuda.synchronize()

    # Exact F32 reference
    vals = F.linear(tokens.float(), w_kv)
    gates = F.linear(tokens.float(), w_kg) + ape
    pooled = _rms_norm((vals * gates.softmax(dim=1)).sum(dim=1), key_norm, 1e-5)
    k_ref = torch.cat([pooled[:, :64], _interleaved_rope(pooled[:, 64:], g_pos, theta)], dim=-1)
    q_ref = torch.cat([q_raw[:, :, :64].float(), _interleaved_rope(q_raw[:, :, 64:], positions, theta)], dim=-1)
    w_ref = w_raw.float() * g.weight_scale
    s_ref = (w_ref.unsqueeze(-2) @ torch.relu(torch.einsum("rhd,gd->rhg", q_ref, k_ref))).squeeze(-2)
    for r in range(rows):
        lim = int(limits[r])
        assert torch.allclose(scores[r, :lim], s_ref[r, :lim], atol=0.05, rtol=0.05)
        assert torch.all(torch.isneginf(scores[r, lim:]))

    # Slot expansion check
    slots_out = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
    lens_out = torch.empty(rows, dtype=torch.int32, device=dev)
    dummy_rec = torch.zeros(rows, 1, dtype=torch.int32, device=dev)
    compile_hy4_indexer_expand_slots_aot(g).launch(
        dummy_rec, positions.to(torch.int32), pt, slots_out, lens_out, scalars=(rows, 1, 4)
    )
    torch.cuda.synchronize()
    for r in range(rows):
        n = int(positions[r]) + 1
        assert int(lens_out[r]) == n
        assert torch.equal(slots_out[r, :n], torch.arange(n, dtype=torch.int32, device=dev))


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("heads", [32, 64])
def test_hy4_mla_swa_extend_clips_window(kv_cache: str, heads: int):
    """Verify mode='extend' prefill on SWA-512 with seq_len=640 (> 512) clips old tokens."""
    dev = require_b12x()
    torch.manual_seed(321)
    seq_len = 640
    window = 512
    g = Hy4MlaGeometry(heads=heads, window_size=window, kv_cache=kv_cache)
    theta = 10_000.0
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    q_a = torch.randn(seq_len, 1024, dtype=torch.bfloat16, device=dev) * 0.25
    kv_a = torch.randn(seq_len, 576, dtype=torch.bfloat16, device=dev) * 0.25
    q_norm = torch.randn(1024, dtype=torch.bfloat16, device=dev) + 1.0
    kv_norm = torch.randn(512, dtype=torch.bfloat16, device=dev) + 1.0
    q_raw = torch.randn(seq_len, heads, 256, dtype=torch.bfloat16, device=dev) * 0.25
    w_uk = torch.randn(heads, 512, 192, dtype=torch.bfloat16, device=dev) * 0.04
    w_uv = torch.randn(heads, 256, 512, dtype=torch.bfloat16, device=dev) * 0.04
    gate = torch.randn(seq_len, heads, 256, dtype=torch.bfloat16, device=dev) * 0.5
    sink = torch.randn(heads, dtype=torch.float32, device=dev) * 0.5

    # Permuted physical pages to also test non-identity page_table translation.
    pages = 12
    perm = torch.tensor([5, 2, 9, 0, 11, 3, 7, 1, 10, 4, 8, 6], dtype=torch.int32, device=dev)
    positions = torch.arange(seq_len, dtype=torch.int64, device=dev)
    slots = perm[(positions // 64)].to(torch.int64) * 64 + (positions % 64)
    cache = torch.zeros(
        pages, 64, 656 if kv_cache == "fp8" else 576,
        dtype=torch.uint8 if kv_cache == "fp8" else torch.bfloat16, device=dev,
    )
    q_resid = torch.empty(seq_len, 1024, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_rank_norm_pack_aot(g).launch(
        q_a, kv_a, q_norm, kv_norm, slots, positions, inv_freq, q_resid, cache, scalars=(seq_len,)
    )
    query = torch.empty(seq_len, heads, 576, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_q_absorb_aot(g).launch(q_raw, w_uk, positions, inv_freq, query, scalars=(seq_len,))

    prog_attn = compile_hy4_mla_attention_aot(g, max_rows=seq_len, max_context=2048, mode="extend", page_size=64)
    scratch = torch.empty(
        max(1024, hy4_attention_layout(g, seq_len, max_context=2048, mode="extend")["scratch_bytes"]),
        dtype=torch.uint8, device=dev,
    )
    attn_out = torch.empty(seq_len, heads, 512, dtype=torch.bfloat16, device=dev)
    lse_out = torch.empty(seq_len, heads, dtype=torch.float32, device=dev)
    pt = perm.unsqueeze(0)
    seqlens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
    prog_attn.launch(query, cache, pt, seqlens, cu, attn_out, lse_out, scratch, scalars=(seq_len, 1, pages, 1))

    uv_out = torch.empty(seq_len, heads, 256, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_sink_gate_uv_aot(g).launch(attn_out, w_uv, gate, lse_out, sink, uv_out, scalars=(seq_len,))
    torch.cuda.synchronize()

    k0_ref = _rms_norm(kv_a[:, :512], kv_norm, 1e-5)
    kr_ref = _interleaved_rope(kv_a[:, 512:], positions, theta)
    qn_ref = q_raw[:, :, :192].float()
    qr_ref = _interleaved_rope(q_raw[:, :, 192:], positions, theta)
    k_nope = torch.einsum("tc,hcd->thd", k0_ref, w_uk.float())
    v_full = torch.einsum("tc,hvc->thv", k0_ref, w_uv.float())
    q_full = torch.cat([qn_ref, qr_ref], dim=-1).transpose(0, 1)
    k_full = torch.cat([k_nope, kr_ref.unsqueeze(1).expand(-1, heads, -1)], dim=-1).transpose(0, 1)
    v_full = v_full.transpose(0, 1)
    scores = torch.matmul(q_full, k_full.transpose(-1, -2)) * (256.0 ** -0.5)
    idx = torch.arange(seq_len, device=dev)
    dist = idx[:, None] - idx[None, :]
    allowed = (dist >= 0) & (dist < window)
    scores = scores.masked_fill(~allowed.unsqueeze(0), -1e30)
    scores = torch.cat([scores, sink[:, None, None].expand(heads, seq_len, 1)], dim=-1)
    probs = torch.softmax(scores, dim=-1)[..., :-1]
    ref_uv = torch.matmul(probs, v_full).transpose(0, 1) * torch.sigmoid(gate.float())
    tol = 0.08 if kv_cache == "fp8" else 0.04
    assert (uv_out.float() - ref_uv).abs().max().item() <= tol


def test_hy4_indexer_long_context_topk_and_ties():
    """Verify exact GPU radix top-511 + straddling-rank-511 ties + non-identity page_table at seq_len > 2048."""
    dev = require_b12x()
    torch.manual_seed(777)
    g = HY4_FLASH_INDEXER
    rows = 7
    max_groups = 700  # 2800 tokens = 44 pages
    pages = 48
    perm = torch.randperm(pages, dtype=torch.int32, device=dev)
    pt = perm.unsqueeze(0).expand(rows, pages).contiguous()

    # Mix short (<=2048) and long (>2048) positions, including exact boundaries 2047, 2048, 2051.
    positions = torch.tensor([130, 2047, 2048, 2051, 2619, 2619, 2619], dtype=torch.int32, device=dev)
    scores = torch.randn(rows, max_groups, dtype=torch.float32, device=dev)
    # Row 4: arbitrary blocks of duplicates.
    scores[4, 100:250] = 0.5
    scores[4, 250:320] = -0.0
    scores[4, 320:380] = 0.0
    # Row 5: exact tie block straddling rank 511 (490 strictly > 1.0, 100 tied == 1.0 -> rem=21 > 1 in pass 5).
    scores[5, :] = -1.0
    scores[5, 0:490] = 2.0
    scores[5, 490:590] = 1.0
    # Row 6: +0.0 / -0.0 tie block straddling rank 511 across non-contiguous gids (500 strictly > 0, 100 signed zeros -> rem=11).
    scores[6, :] = -2.0
    scores[6, 0:500] = 3.0
    scores[6, 500:600:2] = -0.0
    scores[6, 501:600:2] = 0.0

    slots_out = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
    lens_out = torch.empty(rows, dtype=torch.int32, device=dev)
    prog = compile_hy4_indexer_topk_expand_aot(g)
    def launch():
        prog.launch(scores, positions, pt, slots_out, lens_out, scalars=(rows, max_groups, pages))
    launch()
    first_slots, first_lens = slots_out.clone(), lens_out.clone()
    for _ in range(8):
        slots_out.fill_(-2)
        lens_out.fill_(-1)
        launch()
        assert torch.equal(slots_out, first_slots)
        assert torch.equal(lens_out, first_lens)

    for r in range(rows):
        pos = int(positions[r])
        seq_len = pos + 1
        if seq_len <= 2048:
            assert int(lens_out[r]) == seq_len
            expected = [int(perm[t // 64]) * 64 + (t % 64) for t in range(seq_len)]
            assert slots_out[r, :seq_len].tolist() == expected
            assert torch.all(slots_out[r, seq_len:] == -1)
        else:
            cur_group = pos // 4
            cur_valid = (pos % 4) + 1
            expected_len = 511 * 4 + cur_valid
            assert int(lens_out[r]) == expected_len
            assert torch.all(slots_out[r, expected_len:] == -1)
            # Exact deterministic reference: sort descending by score, ascending by gid.
            row_scores = scores[r, :cur_group].cpu().tolist()
            ref_groups = sorted(range(cur_group), key=lambda gid: (-row_scores[gid], gid))[:511]
            cutoff = row_scores[ref_groups[-1]]
            ordered = sorted(gid for gid in ref_groups if row_scores[gid] > cutoff)
            ordered += sorted(gid for gid in ref_groups if row_scores[gid] == cutoff)
            logical_tokens = [4 * gid + sub for gid in ordered for sub in range(4)]
            logical_tokens += list(range(4 * cur_group, 4 * cur_group + cur_valid))
            expected_tokens = [int(perm[t // 64]) * 64 + (t % 64) for t in logical_tokens]
            assert slots_out[r, :expected_len].tolist() == expected_tokens, f"row {r} order mismatch"


def _local_complete_groups(pos: int, rank: int) -> int:
    """Number of complete 4-token groups in [0, pos // 4) owned by parity rank."""
    total_groups = pos // 4
    full_pairs = total_groups // 32
    rem = total_groups % 32
    if rank == 0:
        return full_pairs * 16 + min(rem, 16)
    return full_pairs * 16 + max(rem - 16, 0)


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("active_splits", [1, 2, 4, 8, 16])
def test_hy4_mla_dsa_context_split_oracle(kv_cache: str, active_splits: int):
    """Oracle 1: 2-rank h0+h16 partial + lse_combine2(sink) + gate_uv vs whole attention & F32 ref."""
    dev = require_b12x()
    torch.manual_seed(404 + active_splits)
    g32 = Hy4MlaGeometry(heads=32, window_size=None, kv_cache=kv_cache)
    g16 = Hy4MlaGeometry(heads=16, window_size=None, kv_cache=kv_cache)
    theta = 10_000_000.0
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    # Build a 2560-token pool (40 logical pages: 20 on rank 0, 20 on rank 1) with permuted physical pages.
    total_tokens = 2560
    logical_pages = total_tokens // 64
    half_pages = logical_pages // 2
    perm_full = torch.randperm(logical_pages, dtype=torch.int32, device=dev)
    perm_r0 = torch.randperm(half_pages, dtype=torch.int32, device=dev)
    perm_r1 = torch.randperm(half_pages, dtype=torch.int32, device=dev)

    kv_a = torch.randn(total_tokens, 576, dtype=torch.bfloat16, device=dev) * 0.25
    q_a_dummy = torch.zeros(total_tokens, 1024, dtype=torch.bfloat16, device=dev)
    q_norm = torch.ones(1024, dtype=torch.bfloat16, device=dev)
    kv_norm = torch.randn(512, dtype=torch.bfloat16, device=dev) + 1.0
    kv_pos = torch.arange(total_tokens, dtype=torch.int64, device=dev)

    cache_dtype = torch.uint8 if kv_cache == "fp8" else torch.bfloat16
    cache_width = 656 if kv_cache == "fp8" else 576
    cache_full = torch.zeros(logical_pages, 64, cache_width, dtype=cache_dtype, device=dev)
    cache_r0 = torch.zeros(half_pages, 64, cache_width, dtype=cache_dtype, device=dev)
    cache_r1 = torch.zeros(half_pages, 64, cache_width, dtype=cache_dtype, device=dev)

    slots_full = perm_full[kv_pos // 64].to(torch.int64) * 64 + (kv_pos % 64)
    lpage = kv_pos // 64
    slots_r0 = torch.where(
        (lpage % 2) == 0,
        perm_r0[lpage // 2].to(torch.int64) * 64 + (kv_pos % 64),
        torch.full_like(kv_pos, -1),
    )
    slots_r1 = torch.where(
        (lpage % 2) == 1,
        perm_r1[lpage // 2].to(torch.int64) * 64 + (kv_pos % 64),
        torch.full_like(kv_pos, -1),
    )
    q_resid = torch.empty(total_tokens, 1024, dtype=torch.bfloat16, device=dev)
    pack_prog = compile_hy4_mla_rank_norm_pack_aot(g32)
    pack_prog.launch(q_a_dummy, kv_a, q_norm, kv_norm, slots_full, kv_pos, inv_freq, q_resid, cache_full, scalars=(total_tokens,))
    pack_prog.launch(q_a_dummy, kv_a, q_norm, kv_norm, slots_r0, kv_pos, inv_freq, q_resid, cache_r0, scalars=(total_tokens,))
    pack_prog.launch(q_a_dummy, kv_a, q_norm, kv_norm, slots_r1, kv_pos, inv_freq, q_resid, cache_r1, scalars=(total_tokens,))

    # Diverse rows: empty rank 1, empty rank 0, unbalanced, balanced, full 2048-token union, both-empty.
    row_token_lists: list[list[int]] = [
        list(range(0, 17)),                                     # rank 0 only (pos < 64)
        list(range(64, 128)),                                   # rank 1 only (empty rank 0)
        list(range(0, 64)) + list(range(64, 67)),               # heavily unbalanced r0 >> r1
        list(range(0, 5)) + list(range(64, 128)) + list(range(192, 256)),  # unbalanced r1 >> r0
        list(range(0, 128)),                                    # exact 64 / 64 balanced
        list(range(0, 512)),                                    # 256 / 256
        list(range(0, 1536)),                                   # 768 / 768
        list(range(0, 2048)),                                   # 1024 / 1024 full budget
        list(range(0, 1920, 2)) + list(range(1920, 2560, 2)),   # non-contiguous mix across both ranks
        [],                                                     # both shards empty -> zero + -inf
    ]
    rows = len(row_token_lists)
    q_raw = torch.randn(rows, 32, 256, dtype=torch.bfloat16, device=dev) * 0.25
    w_uk = torch.randn(32, 512, 192, dtype=torch.bfloat16, device=dev) * 0.04
    w_uv = torch.randn(32, 256, 512, dtype=torch.bfloat16, device=dev) * 0.04
    gate = torch.randn(rows, 32, 256, dtype=torch.bfloat16, device=dev) * 0.5
    sink = torch.randn(32, dtype=torch.float32, device=dev) * 0.5
    q_pos = torch.arange(100, 100 + rows, dtype=torch.int64, device=dev)
    query = torch.empty(rows, 32, 576, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_q_absorb_aot(g32).launch(q_raw, w_uk, q_pos, inv_freq, query, scalars=(rows,))

    pt_full = torch.full((rows, 2048), -1, dtype=torch.int32, device=dev)
    pt_r0 = torch.full((rows, 2048), -1, dtype=torch.int32, device=dev)
    pt_r1 = torch.full((rows, 2048), -1, dtype=torch.int32, device=dev)
    len_full = torch.zeros(rows, dtype=torch.int32, device=dev)
    len_r0 = torch.zeros(rows, dtype=torch.int32, device=dev)
    len_r1 = torch.zeros(rows, dtype=torch.int32, device=dev)
    p_full_cpu = perm_full.cpu().tolist()
    p_r0_cpu = perm_r0.cpu().tolist()
    p_r1_cpu = perm_r1.cpu().tolist()
    for r, toks in enumerate(row_token_lists):
        full_s = [p_full_cpu[t // 64] * 64 + (t % 64) for t in toks]
        r0_s = [p_r0_cpu[t // 128] * 64 + (t % 64) for t in toks if (t // 64) % 2 == 0]
        r1_s = [p_r1_cpu[t // 128] * 64 + (t % 64) for t in toks if (t // 64) % 2 == 1]
        if full_s:
            pt_full[r, :len(full_s)] = torch.tensor(full_s, dtype=torch.int32, device=dev)
        if r0_s:
            pt_r0[r, :len(r0_s)] = torch.tensor(r0_s, dtype=torch.int32, device=dev)
        if r1_s:
            pt_r1[r, :len(r1_s)] = torch.tensor(r1_s, dtype=torch.int32, device=dev)
        len_full[r] = len(full_s)
        len_r0[r] = len(r0_s)
        len_r1[r] = len(r1_s)
    cu = torch.arange(rows + 1, dtype=torch.int32, device=dev)

    # Compile h0 and h16 partials; each rank launches BOTH h0 and h16 over its own pages.
    p_h0 = compile_hy4_mla_dsa_partial_aot(g32, max_rows=256, head_begin=0, head_count=16)
    p_h16 = compile_hy4_mla_dsa_partial_aot(g32, max_rows=256, head_begin=16, head_count=16)
    comb = compile_lse_combine2_aot(heads=16, has_sink=True)
    gate_uv = compile_hy4_mla_gate_uv_aot(g16)

    scratch_bytes = hy4_dsa_partial_scratch_bytes(rows, head_count=16, num_splits=16, row_chunk_size=64)
    assert scratch_bytes == p_h0.scratch_bytes(rows)["scratch"]
    scratch = torch.full((scratch_bytes // 4,), float("nan"), dtype=torch.float32, device=dev).view(torch.uint8)

    o_r0_h0 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
    l_r0_h0 = torch.empty(rows, 16, dtype=torch.float32, device=dev)
    o_r0_h16 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
    l_r0_h16 = torch.empty(rows, 16, dtype=torch.float32, device=dev)
    o_r1_h0 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
    l_r1_h0 = torch.empty(rows, 16, dtype=torch.float32, device=dev)
    o_r1_h16 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
    l_r1_h16 = torch.empty(rows, 16, dtype=torch.float32, device=dev)

    p_h0.launch(query, cache_r0, pt_r0, len_r0, cu, o_r0_h0, l_r0_h0, scratch, scalars=(rows, 2048, active_splits))
    p_h16.launch(query, cache_r0, pt_r0, len_r0, cu, o_r0_h16, l_r0_h16, scratch, scalars=(rows, 2048, active_splits))
    p_h0.launch(query, cache_r1, pt_r1, len_r1, cu, o_r1_h0, l_r1_h0, scratch, scalars=(rows, 2048, active_splits))
    p_h16.launch(query, cache_r1, pt_r1, len_r1, cu, o_r1_h16, l_r1_h16, scratch, scalars=(rows, 2048, active_splits))

    # Fixed GPU0-then-GPU1 combine order on each rank's own 16 heads.
    c_h0 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
    c_h16 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev)
    comb.launch(o_r0_h0, l_r0_h0, o_r1_h0, l_r1_h0, sink[:16].contiguous(), c_h0, scalars=(rows,))
    comb.launch(o_r0_h16, l_r0_h16, o_r1_h16, l_r1_h16, sink[16:].contiguous(), c_h16, scalars=(rows,))

    uv_h0 = torch.empty(rows, 16, 256, dtype=torch.bfloat16, device=dev)
    uv_h16 = torch.empty(rows, 16, 256, dtype=torch.bfloat16, device=dev)
    gate_uv.launch(c_h0, w_uv[:16].contiguous(), gate[:, :16].contiguous(), uv_h0, scalars=(rows,))
    gate_uv.launch(c_h16, w_uv[16:].contiguous(), gate[:, 16:].contiguous(), uv_h16, scalars=(rows,))
    uv_split = torch.cat([uv_h0, uv_h16], dim=1)

    # Whole single-GPU kernel baseline on non-empty rows (rows 0..rows-2).
    non_empty = rows - 1
    whole_prog = compile_hy4_mla_attention_aot(g32, max_rows=non_empty, max_context=2048, mode="decode", page_size=1)
    whole_scratch = torch.empty(
        max(1024, hy4_attention_layout(g32, non_empty, max_context=2048, mode="decode")["scratch_bytes"]),
        dtype=torch.uint8, device=dev,
    )
    whole_attn = torch.empty(non_empty, 32, 512, dtype=torch.bfloat16, device=dev)
    whole_lse = torch.empty(non_empty, 32, dtype=torch.float32, device=dev)
    whole_prog.launch(
        query[:non_empty], cache_full, pt_full[:non_empty], len_full[:non_empty], cu[:non_empty + 1],
        whole_attn, whole_lse, whole_scratch, scalars=(non_empty, non_empty, 2048, max(2, active_splits)),
    )
    whole_uv = torch.empty(non_empty, 32, 256, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_sink_gate_uv_aot(g32).launch(
        whole_attn, w_uv, gate[:non_empty], whole_lse, sink, whole_uv, scalars=(non_empty,)
    )
    torch.cuda.synchronize()

    # Empty-shard invariants.
    assert torch.all(o_r1_h0[0] == 0) and torch.all(torch.isneginf(l_r1_h0[0]))
    assert torch.all(o_r0_h0[1] == 0) and torch.all(torch.isneginf(l_r0_h0[1]))
    assert torch.all(uv_split[-1] == 0) and torch.all(torch.isneginf(l_r0_h0[-1])) and torch.all(torch.isneginf(l_r1_h16[-1]))

    # Compare split vs whole GPU kernel.
    assert (uv_split[:non_empty].float() - whole_uv.float()).abs().max().item() <= (0.03 if kv_cache == "fp8" else 0.02)

    # Compare split vs exact IEEE F32 reference.
    k0_ref = _rms_norm(kv_a[:, :512], kv_norm, 1e-5)
    kr_ref = _interleaved_rope(kv_a[:, 512:], kv_pos, theta)
    k_all = torch.cat([k0_ref, kr_ref], dim=-1)  # [T, 576]
    qn_ref = q_raw[:non_empty, :, :192].float()
    qr_ref = _interleaved_rope(q_raw[:non_empty, :, 192:], q_pos[:non_empty], theta)
    q_abs_nope = torch.einsum("rhd,hcd->rhc", qn_ref, w_uk.float())
    q_abs = torch.cat([q_abs_nope, qr_ref], dim=-1)  # [R, 32, 576]
    for r in range(non_empty):
        toks = torch.tensor(row_token_lists[r], dtype=torch.long, device=dev)
        ks = k_all[toks]  # [N, 576]
        scores = torch.matmul(q_abs[r], ks.T) * (256.0 ** -0.5)  # [32, N]
        scores = torch.cat([scores, sink[:, None]], dim=-1)
        probs = torch.softmax(scores, dim=-1)[:, :-1]
        attn_ref = torch.matmul(probs, k0_ref[toks])  # [32, 512]
        uv_ref = torch.einsum("hc,hvc->hv", attn_ref, w_uv.float()) * torch.sigmoid(gate[r].float())
        tol = 0.08 if kv_cache == "fp8" else 0.04
        assert (uv_split[r].float() - uv_ref).abs().max().item() <= tol, f"row {r} F32 ref mismatch"


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("rows", [1, 4, 64, 130, 256])
def test_hy4_mla_dsa_partial_chunking(kv_cache: str, rows: int):
    """Verify row_chunk_size=64 chunking matches single-GPU whole attention across tail & multi-chunk sizes."""
    dev = require_b12x()
    torch.manual_seed(909 + rows)
    g32 = Hy4MlaGeometry(heads=32, window_size=None, kv_cache=kv_cache)
    g16 = Hy4MlaGeometry(heads=16, window_size=None, kv_cache=kv_cache)
    total_tokens = 512  # 8 pages (4 per rank)
    cache_dtype = torch.uint8 if kv_cache == "fp8" else torch.bfloat16
    cache_width = 656 if kv_cache == "fp8" else 576
    kv_a = torch.randn(total_tokens, 576, dtype=torch.bfloat16, device=dev) * 0.2
    q_a_dummy = torch.zeros(total_tokens, 1024, dtype=torch.bfloat16, device=dev)
    q_norm = torch.ones(1024, dtype=torch.bfloat16, device=dev)
    kv_norm = torch.ones(512, dtype=torch.bfloat16, device=dev)
    inv_freq = (1.0 / (10_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()
    kv_pos = torch.arange(total_tokens, dtype=torch.int64, device=dev)
    cache_full = torch.zeros(8, 64, cache_width, dtype=cache_dtype, device=dev)
    cache_r0 = torch.zeros(4, 64, cache_width, dtype=cache_dtype, device=dev)
    cache_r1 = torch.zeros(4, 64, cache_width, dtype=cache_dtype, device=dev)
    lpage = kv_pos // 64
    s_r0 = torch.where((lpage % 2) == 0, (lpage // 2) * 64 + (kv_pos % 64), torch.full_like(kv_pos, -1))
    s_r1 = torch.where((lpage % 2) == 1, (lpage // 2) * 64 + (kv_pos % 64), torch.full_like(kv_pos, -1))
    q_resid = torch.empty(total_tokens, 1024, dtype=torch.bfloat16, device=dev)
    pack = compile_hy4_mla_rank_norm_pack_aot(g32)
    pack.launch(q_a_dummy, kv_a, q_norm, kv_norm, kv_pos, kv_pos, inv_freq, q_resid, cache_full, scalars=(total_tokens,))
    pack.launch(q_a_dummy, kv_a, q_norm, kv_norm, s_r0, kv_pos, inv_freq, q_resid, cache_r0, scalars=(total_tokens,))
    pack.launch(q_a_dummy, kv_a, q_norm, kv_norm, s_r1, kv_pos, inv_freq, q_resid, cache_r1, scalars=(total_tokens,))

    query = torch.randn(rows, 32, 576, dtype=torch.bfloat16, device=dev) * 0.2
    sink = torch.randn(32, dtype=torch.float32, device=dev) * 0.4
    w_uv = torch.randn(32, 256, 512, dtype=torch.bfloat16, device=dev) * 0.04
    gate = torch.randn(rows, 32, 256, dtype=torch.bfloat16, device=dev) * 0.5

    pt_full = torch.arange(total_tokens, dtype=torch.int32, device=dev).unsqueeze(0).expand(rows, total_tokens).contiguous()
    pt_r0 = torch.arange(256, dtype=torch.int32, device=dev).unsqueeze(0).expand(rows, 256).contiguous()
    pt_r1 = torch.arange(256, dtype=torch.int32, device=dev).unsqueeze(0).expand(rows, 256).contiguous()
    len_full = torch.full((rows,), total_tokens, dtype=torch.int32, device=dev)
    len_half = torch.full((rows,), 256, dtype=torch.int32, device=dev)
    cu = torch.arange(rows + 1, dtype=torch.int32, device=dev)

    p_h0 = compile_hy4_mla_dsa_partial_aot(g32, max_rows=4096, head_begin=0, head_count=16, row_chunk_size=64)
    p_h16 = compile_hy4_mla_dsa_partial_aot(g32, max_rows=4096, head_begin=16, head_count=16, row_chunk_size=64)
    scratch = torch.empty(p_h0.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device=dev)
    o00, l00 = torch.empty(rows, 16, 512, dtype=torch.bfloat16, device=dev), torch.empty(rows, 16, dtype=torch.float32, device=dev)
    o01, l01 = torch.empty_like(o00), torch.empty_like(l00)
    o10, l10 = torch.empty_like(o00), torch.empty_like(l00)
    o11, l11 = torch.empty_like(o00), torch.empty_like(l00)
    p_h0.launch(query, cache_r0, pt_r0, len_half, cu, o00, l00, scratch, scalars=(rows, 256, 4))
    p_h16.launch(query, cache_r0, pt_r0, len_half, cu, o01, l01, scratch, scalars=(rows, 256, 4))
    p_h0.launch(query, cache_r1, pt_r1, len_half, cu, o10, l10, scratch, scalars=(rows, 256, 4))
    p_h16.launch(query, cache_r1, pt_r1, len_half, cu, o11, l11, scratch, scalars=(rows, 256, 4))

    comb = compile_lse_combine2_aot(heads=16, has_sink=True)
    gate_uv = compile_hy4_mla_gate_uv_aot(g16)
    c0, c1 = torch.empty_like(o00), torch.empty_like(o00)
    comb.launch(o00, l00, o10, l10, sink[:16].contiguous(), c0, scalars=(rows,))
    comb.launch(o01, l01, o11, l11, sink[16:].contiguous(), c1, scalars=(rows,))
    u0, u1 = torch.empty(rows, 16, 256, dtype=torch.bfloat16, device=dev), torch.empty(rows, 16, 256, dtype=torch.bfloat16, device=dev)
    gate_uv.launch(c0, w_uv[:16].contiguous(), gate[:, :16].contiguous(), u0, scalars=(rows,))
    gate_uv.launch(c1, w_uv[16:].contiguous(), gate[:, 16:].contiguous(), u1, scalars=(rows,))
    uv_split = torch.cat([u0, u1], dim=1)

    whole = compile_hy4_mla_attention_aot(g32, max_rows=rows, max_context=2048, mode="decode", page_size=1)
    w_scratch = torch.empty(max(1024, hy4_attention_layout(g32, rows, max_context=2048, mode="decode")["scratch_bytes"]),
                            dtype=torch.uint8, device=dev)
    w_attn = torch.empty(rows, 32, 512, dtype=torch.bfloat16, device=dev)
    w_lse = torch.empty(rows, 32, dtype=torch.float32, device=dev)
    whole.launch(query, cache_full, pt_full, len_full, cu, w_attn, w_lse, w_scratch, scalars=(rows, rows, total_tokens, 4))
    w_uv_out = torch.empty(rows, 32, 256, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_sink_gate_uv_aot(g32).launch(w_attn, w_uv, gate, w_lse, sink, w_uv_out, scalars=(rows,))
    torch.cuda.synchronize()
    assert (uv_split.float() - w_uv_out.float()).abs().max().item() <= (0.03 if kv_cache == "fp8" else 0.02)


def test_hy4_indexer_context_split_oracle():
    """Oracle 2: 2-rank scored_topk + merge_expand_owned vs 1-GPU topk_expand (ties, -0.0, boundaries, 32K)."""
    dev = require_b12x()
    torch.manual_seed(2026)
    g = HY4_FLASH_INDEXER

    # 32K logical context = 512 pages = 8192 groups (256 pages / 4096 groups per rank).
    max_pages = 512
    half_pages = max_pages // 2
    max_groups = max_pages * 16
    half_groups = half_pages * 16

    perm_full = torch.randperm(max_pages, dtype=torch.int32, device=dev)
    perm_r0 = torch.randperm(half_pages, dtype=torch.int32, device=dev)
    perm_r1 = torch.randperm(half_pages, dtype=torch.int32, device=dev)

    # Inverse maps: physical token slot -> global logical token id.
    inv_full = torch.empty(max_pages * 64, dtype=torch.int32)
    inv_r0 = torch.empty(half_pages * 64, dtype=torch.int32)
    inv_r1 = torch.empty(half_pages * 64, dtype=torch.int32)
    for lp, pp in enumerate(perm_full.cpu().tolist()):
        inv_full[pp * 64:(pp + 1) * 64] = torch.arange(lp * 64, (lp + 1) * 64, dtype=torch.int32)
    for idx, pp in enumerate(perm_r0.cpu().tolist()):
        lp = idx * 2
        inv_r0[pp * 64:(pp + 1) * 64] = torch.arange(lp * 64, (lp + 1) * 64, dtype=torch.int32)
    for idx, pp in enumerate(perm_r1.cpu().tolist()):
        lp = idx * 2 + 1
        inv_r1[pp * 64:(pp + 1) * 64] = torch.arange(lp * 64, (lp + 1) * 64, dtype=torch.int32)

    positions = torch.tensor(
        [
            0,      # 0: single token (rank 1 empty)
            63,     # 1: full first page (rank 1 empty)
            64,     # 2: first token on rank 1
            130,    # 3: short multi-page
            2047,   # 4: exact 2048 boundary (short path)
            2048,   # 5: exact 2049 boundary (long path, local_limit=256 < 511)
            2051,   # 6: 2052 boundary (cur_valid=4, local_limit=256 < 511)
            4099,   # 7: local_limit=512 (just above 511!)
            8200,   # 8: all-equal ties across both ranks
            8201,   # 9: ties straddling rank 511 across both ranks
            8202,   # 10: -0.0 / +0.0 signed-zero ties straddling rank 511 across both ranks
            8203,   # 11: all 511 prior winners on rank 0 (rank 1 has 0 prior winners)
            32767,  # 12: full 32K context (4096 local groups per rank)
        ],
        dtype=torch.int32,
        device=dev,
    )
    rows = int(positions.numel())
    scores_full = torch.randn(rows, max_groups, dtype=torch.float32, device=dev)

    # Inject NaNs on short rows to prove scores are never read when pos+1 <= 2048.
    scores_full[:5, :] = float("nan")
    # Row 8: all-equal ties -> exact lowest 511 global gids (0..510).
    scores_full[8, :] = 1.25
    # Row 9: ties straddling rank 511 across both ranks (480 strict > 1.0, 120 tied == 1.0).
    scores_full[9, :] = -5.0
    scores_full[9, :480] = 3.0
    scores_full[9, 480:600] = 1.0
    # Row 10: signed zeros -0.0 / +0.0 straddling rank 511 across both ranks.
    scores_full[10, :] = -2.0
    scores_full[10, :500] = 4.0
    scores_full[10, 500:620:2] = -0.0
    scores_full[10, 501:620:2] = 0.0
    # Row 11: rank 0 pages win all 511 prior slots (rank 1 gets 0 prior groups).
    scores_full[11, :] = -10.0
    r0_mask = ((torch.arange(max_groups, device=dev) // 16) % 2) == 0
    scores_full[11, r0_mask] = 5.0

    # Split scores into compact parity shards (exact output layout of score_groups on each rank).
    scores_r0 = torch.empty(rows, half_groups, dtype=torch.float32, device=dev)
    scores_r1 = torch.empty(rows, half_groups, dtype=torch.float32, device=dev)
    for lp in range(max_pages):
        dst_s = scores_r0 if (lp % 2 == 0) else scores_r1
        lp_half = lp // 2
        dst_s[:, lp_half * 16:(lp_half + 1) * 16] = scores_full[:, lp * 16:(lp + 1) * 16]

    limits_r0 = torch.tensor([_local_complete_groups(int(p), 0) for p in positions.cpu().tolist()], dtype=torch.int32, device=dev)
    limits_r1 = torch.tensor([_local_complete_groups(int(p), 1) for p in positions.cpu().tolist()], dtype=torch.int32, device=dev)

    pt_full = perm_full.unsqueeze(0).expand(rows, max_pages).contiguous()
    pt_r0 = perm_r0.unsqueeze(0).expand(rows, half_pages).contiguous()
    pt_r1 = perm_r1.unsqueeze(0).expand(rows, half_pages).contiguous()

    # 1-GPU baseline.
    ref_slots = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
    ref_lens = torch.empty(rows, dtype=torch.int32, device=dev)
    compile_hy4_indexer_topk_expand_aot(g).launch(
        scores_full, positions, pt_full, ref_slots, ref_lens, scalars=(rows, max_groups, max_pages)
    )

    # 2-rank scored_topk + exchange concat + merge_expand_owned.
    scored_topk = compile_hy4_indexer_scored_topk_aot(g)
    merge_owned = compile_hy4_indexer_merge_expand_owned_aot(g)

    cand_s = torch.empty(rows, 1024, dtype=torch.float32, device=dev)
    cand_i = torch.empty(rows, 1024, dtype=torch.int32, device=dev)
    s0 = torch.empty(rows, 512, dtype=torch.float32, device=dev)
    i0 = torch.empty(rows, 512, dtype=torch.int32, device=dev)
    s1 = torch.empty(rows, 512, dtype=torch.float32, device=dev)
    i1 = torch.empty(rows, 512, dtype=torch.int32, device=dev)

    scored_topk.launch(scores_r0, positions, limits_r0, s0, i0, scalars=(rows, half_groups, 2, 0))
    scored_topk.launch(scores_r1, positions, limits_r1, s1, i1, scalars=(rows, half_groups, 2, 1))
    cand_s[:, :512].copy_(s0)
    cand_s[:, 512:].copy_(s1)
    cand_i[:, :512].copy_(i0)
    cand_i[:, 512:].copy_(i1)

    loc_slots_r0 = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
    loc_lens_r0 = torch.empty(rows, dtype=torch.int32, device=dev)
    draft_r0 = torch.full((rows, 2048), -7, dtype=torch.int32, device=dev)
    loc_slots_r1 = torch.empty(rows, 2048, dtype=torch.int32, device=dev)
    loc_lens_r1 = torch.empty(rows, dtype=torch.int32, device=dev)
    draft_r1 = torch.full((rows, 2048), -7, dtype=torch.int32, device=dev)

    # Verify write_draft=0 leaves draft_slots untouched.
    merge_owned.launch(cand_s, cand_i, positions, pt_r0, loc_slots_r0, loc_lens_r0, draft_r0, scalars=(rows, half_pages, 0, 0))
    merge_owned.launch(cand_s, cand_i, positions, pt_r1, loc_slots_r1, loc_lens_r1, draft_r1, scalars=(rows, half_pages, 1, 0))
    torch.cuda.synchronize()
    assert torch.all(draft_r0 == -7) and torch.all(draft_r1 == -7)

    # Repeat 8 times with write_draft=1 to verify bit-identical determinism.
    merge_owned.launch(cand_s, cand_i, positions, pt_r0, loc_slots_r0, loc_lens_r0, draft_r0, scalars=(rows, half_pages, 0, 1))
    merge_owned.launch(cand_s, cand_i, positions, pt_r1, loc_slots_r1, loc_lens_r1, draft_r1, scalars=(rows, half_pages, 1, 1))
    torch.cuda.synchronize()
    snap = (loc_slots_r0.clone(), loc_lens_r0.clone(), draft_r0.clone(),
            loc_slots_r1.clone(), loc_lens_r1.clone(), draft_r1.clone())
    for _ in range(8):
        loc_slots_r0.fill_(-2)
        loc_slots_r1.fill_(-2)
        merge_owned.launch(cand_s, cand_i, positions, pt_r0, loc_slots_r0, loc_lens_r0, draft_r0, scalars=(rows, half_pages, 0, 1))
        merge_owned.launch(cand_s, cand_i, positions, pt_r1, loc_slots_r1, loc_lens_r1, draft_r1, scalars=(rows, half_pages, 1, 1))
        torch.cuda.synchronize()
        assert torch.equal(loc_slots_r0, snap[0]) and torch.equal(loc_lens_r0, snap[1]) and torch.equal(draft_r0, snap[2])
        assert torch.equal(loc_slots_r1, snap[3]) and torch.equal(loc_lens_r1, snap[4]) and torch.equal(draft_r1, snap[5])

    # Verify exact token-set equivalence vs 1-GPU topk_expand, ascending local order, and parity ownership.
    for r in range(rows):
        n_ref = int(ref_lens[r])
        n0 = int(loc_lens_r0[r])
        n1 = int(loc_lens_r1[r])
        assert n0 + n1 == n_ref, f"row {r}: {n0}+{n1} != {n_ref}"
        assert torch.all(loc_slots_r0[r, n0:] == -1)
        assert torch.all(loc_slots_r1[r, n1:] == -1)

        toks_ref = sorted(int(inv_full[s]) for s in ref_slots[r, :n_ref].cpu().tolist())
        toks_r0 = [int(inv_r0[s]) for s in loc_slots_r0[r, :n0].cpu().tolist()]
        toks_r1 = [int(inv_r1[s]) for s in loc_slots_r1[r, :n1].cpu().tolist()]
        assert toks_r0 == sorted(toks_r0), f"row {r}: rank 0 not ascending logical order"
        assert toks_r1 == sorted(toks_r1), f"row {r}: rank 1 not ascending logical order"
        assert all((t // 64) % 2 == 0 for t in toks_r0)
        assert all((t // 64) % 2 == 1 for t in toks_r1)
        assert sorted(toks_r0 + toks_r1) == toks_ref, f"row {r}: 2-rank union != 1-GPU topk_expand"

    # Also check end-to-end with real score_groups on parity-sharded compressed group caches.
    n_groups = 1280  # 5120 tokens = 80 logical pages (40 per rank)
    tokens = torch.randn(n_groups, 4, 128, dtype=torch.bfloat16, device=dev) * 0.3
    w_kv = torch.randn(128, 128, dtype=torch.float32, device=dev) * 0.08
    w_kg = torch.randn(128, 128, dtype=torch.float32, device=dev) * 0.08
    ape = torch.randn(4, 128, dtype=torch.float32, device=dev) * 0.2
    key_norm = torch.randn(128, dtype=torch.float32, device=dev) + 1.0
    inv_freq = (1.0 / (10_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()
    g_pos = torch.arange(n_groups, dtype=torch.int64, device=dev) * 4
    g_lpage = torch.arange(n_groups, dtype=torch.int64, device=dev) // 16
    g_off = torch.arange(n_groups, dtype=torch.int64, device=dev) % 16
    gcache_full = torch.zeros(80, g.page_bytes, dtype=torch.uint8, device=dev)
    gcache_r0 = torch.zeros(40, g.page_bytes, dtype=torch.uint8, device=dev)
    gcache_r1 = torch.zeros(40, g.page_bytes, dtype=torch.uint8, device=dev)
    pool_prog = compile_hy4_indexer_pool_group_aot(g)
    gs_full = g_lpage * 16 + g_off
    gs_r0 = torch.where((g_lpage % 2) == 0, (g_lpage // 2) * 16 + g_off, torch.full_like(g_lpage, -1))
    gs_r1 = torch.where((g_lpage % 2) == 1, (g_lpage // 2) * 16 + g_off, torch.full_like(g_lpage, -1))
    pool_prog.launch(tokens, w_kv, w_kg, ape, key_norm, g_pos, gs_full, inv_freq, gcache_full, scalars=(n_groups,))
    pool_prog.launch(tokens, w_kv, w_kg, ape, key_norm, g_pos, gs_r0, inv_freq, gcache_r0, scalars=(n_groups,))
    pool_prog.launch(tokens, w_kv, w_kg, ape, key_norm, g_pos, gs_r1, inv_freq, gcache_r1, scalars=(n_groups,))

    q_raw = torch.randn(4, 16, 128, dtype=torch.bfloat16, device=dev) * 0.3
    w_raw = torch.randn(4, 16, dtype=torch.bfloat16, device=dev) * 0.3
    q_pos = torch.tensor([120, 2047, 2049, 5118], dtype=torch.int64, device=dev)
    q_rot = torch.empty_like(q_raw)
    w_out = torch.empty(4, 16, dtype=torch.float32, device=dev)
    compile_hy4_indexer_query_rope_aot(g).launch(q_raw, w_raw, q_pos, inv_freq, q_rot, w_out, scalars=(4,))

    sg = compile_hy4_indexer_score_groups_aot(g)
    pt80 = torch.arange(80, dtype=torch.int32, device=dev).unsqueeze(0).expand(4, 80).contiguous()
    pt40 = torch.arange(40, dtype=torch.int32, device=dev).unsqueeze(0).expand(4, 40).contiguous()
    sc_f = torch.empty(4, n_groups, dtype=torch.float32, device=dev)
    sc_0 = torch.empty(4, n_groups // 2, dtype=torch.float32, device=dev)
    sc_1 = torch.empty(4, n_groups // 2, dtype=torch.float32, device=dev)
    lim_f = (q_pos // 4).to(torch.int32)
    lim_0 = torch.tensor([_local_complete_groups(int(p), 0) for p in q_pos.cpu().tolist()], dtype=torch.int32, device=dev)
    lim_1 = torch.tensor([_local_complete_groups(int(p), 1) for p in q_pos.cpu().tolist()], dtype=torch.int32, device=dev)
    sg.launch(q_rot, w_out, gcache_full, pt80, lim_f, sc_f, scalars=(4, n_groups, 80))
    sg.launch(q_rot, w_out, gcache_r0, pt40, lim_0, sc_0, scalars=(4, n_groups // 2, 40))
    sg.launch(q_rot, w_out, gcache_r1, pt40, lim_1, sc_1, scalars=(4, n_groups // 2, 40))

    r_slots = torch.empty(4, 2048, dtype=torch.int32, device=dev)
    r_lens = torch.empty(4, dtype=torch.int32, device=dev)
    compile_hy4_indexer_topk_expand_aot(g).launch(sc_f, q_pos.to(torch.int32), pt80, r_slots, r_lens, scalars=(4, n_groups, 80))
    cs, ci = torch.empty(4, 1024, dtype=torch.float32, device=dev), torch.empty(4, 1024, dtype=torch.int32, device=dev)
    s0e, i0e = torch.empty(4, 512, dtype=torch.float32, device=dev), torch.empty(4, 512, dtype=torch.int32, device=dev)
    s1e, i1e = torch.empty(4, 512, dtype=torch.float32, device=dev), torch.empty(4, 512, dtype=torch.int32, device=dev)
    scored_topk.launch(sc_0, q_pos.to(torch.int32), lim_0, s0e, i0e, scalars=(4, n_groups // 2, 2, 0))
    scored_topk.launch(sc_1, q_pos.to(torch.int32), lim_1, s1e, i1e, scalars=(4, n_groups // 2, 2, 1))
    cs[:, :512].copy_(s0e)
    cs[:, 512:].copy_(s1e)
    ci[:, :512].copy_(i0e)
    ci[:, 512:].copy_(i1e)
    ls0, ll0, d0 = torch.empty(4, 2048, dtype=torch.int32, device=dev), torch.empty(4, dtype=torch.int32, device=dev), torch.empty(4, 2048, dtype=torch.int32, device=dev)
    ls1, ll1, d1 = torch.empty(4, 2048, dtype=torch.int32, device=dev), torch.empty(4, dtype=torch.int32, device=dev), torch.empty(4, 2048, dtype=torch.int32, device=dev)
    merge_owned.launch(cs, ci, q_pos.to(torch.int32), pt40, ls0, ll0, d0, scalars=(4, 40, 0, 1))
    merge_owned.launch(cs, ci, q_pos.to(torch.int32), pt40, ls1, ll1, d1, scalars=(4, 40, 1, 1))
    torch.cuda.synchronize()
    for r in range(4):
        n_ref = int(r_lens[r])
        n0, n1 = int(ll0[r]), int(ll1[r])
        assert n0 + n1 == n_ref
        t0 = [(s // 64) * 128 + (s % 64) for s in ls0[r, :n0].cpu().tolist()]
        t1 = [((s // 64) * 2 + 1) * 64 + (s % 64) for s in ls1[r, :n1].cpu().tolist()]
        assert sorted(t0 + t1) == sorted(r_slots[r, :n_ref].cpu().tolist())


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_hy4_context_split_draft_dual_gather_oracle(kv_cache: str):
    """Oracle 3: draft_slots dual-gather identity over parity-split KV caches vs single-GPU gather."""
    dev = require_b12x()
    torch.manual_seed(5150)
    g_mla = Hy4MlaGeometry(heads=32, window_size=None, kv_cache=kv_cache)
    g_idx = HY4_FLASH_INDEXER

    logical_pages = 160  # 10240 tokens
    half_pages = logical_pages // 2
    max_groups = logical_pages * 16
    half_groups = half_pages * 16
    row_bytes = g_mla.record_bytes

    perm_full = torch.randperm(logical_pages, dtype=torch.int32, device=dev)
    perm_r0 = torch.randperm(half_pages, dtype=torch.int32, device=dev)
    perm_r1 = torch.randperm(half_pages, dtype=torch.int32, device=dev)

    # Populate byte-exact KV records in full and parity-split caches.
    raw_records = torch.randint(0, 256, (logical_pages * 64, row_bytes), dtype=torch.uint8, device=dev)
    pool_full = torch.empty(logical_pages * 64, row_bytes, dtype=torch.uint8, device=dev)
    pool_r0 = torch.empty(half_pages * 64, row_bytes, dtype=torch.uint8, device=dev)
    pool_r1 = torch.empty(half_pages * 64, row_bytes, dtype=torch.uint8, device=dev)
    for lp in range(logical_pages):
        src = raw_records[lp * 64:(lp + 1) * 64]
        pp_f = int(perm_full[lp])
        pool_full[pp_f * 64:(pp_f + 1) * 64] = src
        if lp % 2 == 0:
            pp = int(perm_r0[lp // 2])
            pool_r0[pp * 64:(pp + 1) * 64] = src
        else:
            pp = int(perm_r1[lp // 2])
            pool_r1[pp * 64:(pp + 1) * 64] = src

    positions = torch.tensor([15, 130, 2047, 2048, 2051, 4099, 8200], dtype=torch.int32, device=dev)
    rows = int(positions.numel())
    scores_full = torch.randn(rows, max_groups, dtype=torch.float32, device=dev)
    scores_r0 = torch.empty(rows, half_groups, dtype=torch.float32, device=dev)
    scores_r1 = torch.empty(rows, half_groups, dtype=torch.float32, device=dev)
    for lp in range(logical_pages):
        dst = scores_r0 if lp % 2 == 0 else scores_r1
        dst[:, (lp // 2) * 16:(lp // 2 + 1) * 16] = scores_full[:, lp * 16:(lp + 1) * 16]

    lim_r0 = torch.tensor([_local_complete_groups(int(p), 0) for p in positions.cpu().tolist()], dtype=torch.int32, device=dev)
    lim_r1 = torch.tensor([_local_complete_groups(int(p), 1) for p in positions.cpu().tolist()], dtype=torch.int32, device=dev)
    pt_r0 = perm_r0.unsqueeze(0).expand(rows, half_pages).contiguous()
    pt_r1 = perm_r1.unsqueeze(0).expand(rows, half_pages).contiguous()

    scored_topk = compile_hy4_indexer_scored_topk_aot(g_idx)
    merge_owned = compile_hy4_indexer_merge_expand_owned_aot(g_idx)
    gather = compile_paged_staging_gather_aot(row_bytes=row_bytes, page_rows=1)

    s0, i0 = torch.empty(rows, 512, dtype=torch.float32, device=dev), torch.empty(rows, 512, dtype=torch.int32, device=dev)
    s1, i1 = torch.empty(rows, 512, dtype=torch.float32, device=dev), torch.empty(rows, 512, dtype=torch.int32, device=dev)
    scored_topk.launch(scores_r0, positions, lim_r0, s0, i0, scalars=(rows, half_groups, 2, 0))
    scored_topk.launch(scores_r1, positions, lim_r1, s1, i1, scalars=(rows, half_groups, 2, 1))
    cand_s = torch.cat([s0, s1], dim=1).contiguous()
    cand_i = torch.cat([i0, i1], dim=1).contiguous()

    ls0, ll0, draft_r0 = (torch.empty(rows, 2048, dtype=torch.int32, device=dev),
                          torch.empty(rows, dtype=torch.int32, device=dev),
                          torch.empty(rows, 2048, dtype=torch.int32, device=dev))
    ls1, ll1, draft_r1 = (torch.empty(rows, 2048, dtype=torch.int32, device=dev),
                          torch.empty(rows, dtype=torch.int32, device=dev),
                          torch.empty(rows, 2048, dtype=torch.int32, device=dev))
    merge_owned.launch(cand_s, cand_i, positions, pt_r0, ls0, ll0, draft_r0, scalars=(rows, half_pages, 0, 1))
    merge_owned.launch(cand_s, cand_i, positions, pt_r1, ls1, ll1, draft_r1, scalars=(rows, half_pages, 1, 1))
    torch.cuda.synchronize()

    inv_r0 = torch.empty(half_pages * 64, dtype=torch.int32)
    inv_r1 = torch.empty(half_pages * 64, dtype=torch.int32)
    for idx, pp in enumerate(perm_r0.cpu().tolist()):
        inv_r0[pp * 64:(pp + 1) * 64] = torch.arange(idx * 2 * 64, (idx * 2 + 1) * 64, dtype=torch.int32)
    for idx, pp in enumerate(perm_r1.cpu().tolist()):
        inv_r1[pp * 64:(pp + 1) * 64] = torch.arange((idx * 2 + 1) * 64, (idx * 2 + 2) * 64, dtype=torch.int32)

    for r in range(rows):
        pos = int(positions[r])
        recall_len = min(pos + 1, 2044 + (pos % 4) + 1)
        d0 = draft_r0[r]
        d1 = draft_r1[r]
        valid0 = d0[:recall_len] >= 0
        valid1 = d1[:recall_len] >= 0
        # Exact disjoint partition of [0, recall_len) and -1 padding beyond recall_len.
        assert torch.all(valid0 ^ valid1), f"row {r}: overlap or gap in draft_slots"
        assert torch.all(d0[recall_len:] == -1) and torch.all(d1[recall_len:] == -1)

        # Dual gather: rank 0 local pool then rank 1 peer pool into same staging buffer.
        staging_dual = torch.full((2048, row_bytes), 0xA5, dtype=torch.uint8, device=dev)
        gather.launch(pool_r0, d0, staging_dual, scalars=(recall_len, 1, 0))
        gather.launch(pool_r1, d1, staging_dual, scalars=(recall_len, 1, 0))

        # Single-GPU gather over the exact unified global physical slot sequence.
        d0_cpu, d1_cpu = d0[:recall_len].cpu().tolist(), d1[:recall_len].cpu().tolist()
        global_toks = [int(inv_r0[a]) if a >= 0 else int(inv_r1[b]) for a, b in zip(d0_cpu, d1_cpu)]
        full_slots = torch.tensor(
            [int(perm_full[t // 64]) * 64 + (t % 64) for t in global_toks],
            dtype=torch.int32, device=dev,
        )
        staging_single = torch.full((2048, row_bytes), 0xA5, dtype=torch.uint8, device=dev)
        gather.launch(pool_full, full_slots, staging_single, scalars=(recall_len, 1, 0))
        torch.cuda.synchronize()
        assert torch.equal(staging_dual[:recall_len], staging_single[:recall_len]), f"row {r}: dual gather mismatch"
        assert torch.equal(staging_dual[recall_len:], staging_single[recall_len:])
