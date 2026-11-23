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
from b12x.integration.cuteafd.hy4_mla import (
    Hy4MlaGeometry,
    compile_hy4_mla_attention_aot,
    compile_hy4_mla_q_absorb_aot,
    compile_hy4_mla_rank_norm_pack_aot,
    compile_hy4_mla_sink_gate_uv_aot,
    hy4_attention_layout,
)
from b12x.integration.cuteafd.hy4_indexer import (
    HY4_FLASH_INDEXER,
    compile_hy4_indexer_expand_slots_aot,
    compile_hy4_indexer_pool_group_aot,
    compile_hy4_indexer_query_rope_aot,
    compile_hy4_indexer_score_groups_aot,
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
