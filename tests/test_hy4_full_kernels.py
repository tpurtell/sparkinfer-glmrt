"""Exact GPU oracle and SM120 row-band tests for Hy4 (hy_v4_vl) AOT kernels.

Covers every Hy4-specific kernel geometry:
- 4-stream iHC + terminal hc_head at H=6144 (HY4_IHC)
- Absorbed gated MLA at q_lora_rank=2048, qa_kva_eps=1e-6, 64H (TP1) / 32H (TP2)
- 64-head ContextSplit DSA partial (32H/rank) + lse_combine2(heads=32, has_sink=True)
- Hy4IndexPost (32H x 128, LayerNorm eps=1e-5, trailing-64 interleaved M-RoPE, 8448 B/page)
  verified standalone and end-to-end through glm_index_topk and ContextSplit scored_index_topk
- Dynamic vision attention at (16, 96) for Hy4's 40-layer ViT
"""
from __future__ import annotations

import json
import math
import os
import statistics
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from tests._reference.helpers import require_b12x

torch.set_float32_matmul_precision("highest")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

from b12x.integration.cuteafd._common import GLM53
from b12x.integration.cuteafd.context_split import (
    compile_dsa_candidate_merge_aot,
    compile_lse_combine2_aot,
    compile_scored_index_topk_aot,
)
from b12x.integration.cuteafd.glm_indexer import compile_glm_index_topk_aot
from b12x.integration.cuteafd.hy4_ihc import (
    HY4_IHC,
    compile_hy4_hc_head_aot,
    compile_hy4_ihc_post_aot,
    compile_hy4_ihc_post_pre_aot,
    compile_hy4_ihc_pre_aot,
)
from b12x.integration.cuteafd.hy4_indexer import (
    HY4_INDEXER,
    compile_hy4_index_post_aot,
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
)
from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)) * w.float()


def _thw(pos: torch.Tensor) -> torch.Tensor:
    return pos.to(torch.int64).unsqueeze(-1).expand(*pos.shape, 3).contiguous()


def _interleaved_mrope_tail64(x: torch.Tensor, positions_thw: torch.Tensor, theta: float,
                              section: tuple[int, int, int] = (12, 10, 10)) -> torch.Tensor:
    """Reference interleaved M-RoPE on trailing 64 dims matching Hy4 F32 reference."""
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=x.device) / 64.0))
    axes = []
    for p in range(32):
        if p % 3 == 1 and p < 3 * section[1]:
            axes.append(1)
        elif p % 3 == 2 and p < 3 * section[2]:
            axes.append(2)
        else:
            axes.append(0)
    axis_idx = torch.tensor(axes, dtype=torch.int64, device=x.device)
    pos_per_pair = positions_thw[:, axis_idx].float()  # [rows, 32]
    freq = pos_per_pair * inv  # [rows, 32]
    cos = torch.cos(freq).repeat_interleave(2, dim=-1)
    sin = torch.sin(freq).repeat_interleave(2, dim=-1)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    xf = x.float()
    prefix, tail = xf[..., :-64], xf[..., -64:]
    alt = torch.stack((-tail[..., 1::2], tail[..., ::2]), dim=-1).flatten(-2)
    rot = tail * cos + alt * sin
    return torch.cat([prefix, rot], dim=-1)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_hy4_ihc_6144_oracle(rows: int):
    dev = require_b12x()
    torch.manual_seed(142 + rows)
    h = 6144
    r = torch.randn(rows, 4, h, dtype=torch.bfloat16, device=dev) * 0.5
    delta = torch.randn(rows, h, dtype=torch.bfloat16, device=dev) * 0.5
    fn = torch.randn(8, 4 * h, dtype=torch.float32, device=dev) * 0.015
    scale = torch.tensor([0.3, 0.7], dtype=torch.float32, device=dev)
    base = torch.randn(8, dtype=torch.float32, device=dev) * 0.2
    norm = torch.randn(h, dtype=torch.bfloat16, device=dev) + 1.0

    flat64 = r.double().reshape(rows, 4 * h)
    inv = torch.rsqrt(flat64.square().mean(-1, keepdim=True) + 1e-5)
    mix = (flat64 @ fn.double().T) * inv
    pre_ref = (torch.sigmoid(mix[:, :4] * scale[0].double() + base[:4].double()) + 1e-6).float()
    post_ref = (2.0 * torch.sigmoid(mix[:, 4:] * scale[1].double() + base[4:].double()) + 1e-6).float()
    col_ref = (r.float() * pre_ref.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
    y_ref = _rms_norm(col_ref, norm, 1e-5).to(torch.bfloat16)

    prog_pre = compile_hy4_ihc_pre_aot(HY4_IHC)
    post_out = torch.empty(rows, 4, dtype=torch.float32, device=dev)
    y_out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_pre.launch(r, fn, scale, base, norm, post_out, y_out, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.allclose(post_out, post_ref, atol=1e-5, rtol=1e-5)
    assert (y_out.float() - y_ref.float()).abs().max().item() <= 0.03

    prog_post = compile_hy4_ihc_post_aot(HY4_IHC)
    r_after = torch.empty_like(r)
    prog_post.launch(delta, r, post_ref, r_after, scalars=(rows,))
    r_after_ref = (post_ref.unsqueeze(-1) * delta.float().unsqueeze(1) + r.float()).to(torch.bfloat16)
    torch.cuda.synchronize()
    assert (r_after.float() - r_after_ref.float()).abs().max().item() <= 0.02
    large = r_after_ref.float().abs() > 1e-3
    assert (r_after.view(torch.int16).int() - r_after_ref.view(torch.int16).int()).abs()[large].max().item() <= 1

    prog_pp = compile_hy4_ihc_post_pre_aot(HY4_IHC)
    r_pp = torch.empty_like(r)
    post_pp = torch.empty(rows, 4, dtype=torch.float32, device=dev)
    y_pp = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_pp.launch(delta, r, post_ref, fn, scale, base, norm, r_pp, post_pp, y_pp, scalars=(rows,))
    post_seq = torch.empty(rows, 4, dtype=torch.float32, device=dev)
    y_seq = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_pre.launch(r_after, fn, scale, base, norm, post_seq, y_seq, scalars=(rows,))
    torch.cuda.synchronize()
    assert torch.equal(r_pp, r_after)
    assert torch.equal(post_pp, post_seq)
    assert torch.equal(y_pp, y_seq)

    # Row invariance: every row in a multi-row launch must be bit-identical to an M=1 launch.
    if rows > 1:
        post_1 = torch.empty(1, 4, dtype=torch.float32, device=dev)
        y_1 = torch.empty(1, h, dtype=torch.bfloat16, device=dev)
        r_1 = torch.empty(1, 4, h, dtype=torch.bfloat16, device=dev)
        for idx in (0, rows // 2, rows - 1):
            prog_pre.launch(
                r[idx : idx + 1].contiguous(), fn, scale, base, norm, post_1, y_1, scalars=(1,)
            )
            torch.cuda.synchronize()
            assert torch.equal(post_out[idx : idx + 1], post_1)
            assert torch.equal(y_out[idx : idx + 1], y_1)

            prog_post.launch(
                delta[idx : idx + 1].contiguous(),
                r[idx : idx + 1].contiguous(),
                post_ref[idx : idx + 1].contiguous(),
                r_1,
                scalars=(1,),
            )
            torch.cuda.synchronize()
            assert torch.equal(r_after[idx : idx + 1], r_1)

            prog_pp.launch(
                delta[idx : idx + 1].contiguous(),
                r[idx : idx + 1].contiguous(),
                post_ref[idx : idx + 1].contiguous(),
                fn,
                scale,
                base,
                norm,
                r_1,
                post_1,
                y_1,
                scalars=(1,),
            )
            torch.cuda.synchronize()
            assert torch.equal(r_pp[idx : idx + 1], r_1)
            assert torch.equal(post_pp[idx : idx + 1], post_1)
            assert torch.equal(y_pp[idx : idx + 1], y_1)

    hfn = torch.randn(4, 4 * h, dtype=torch.float32, device=dev) * 0.015
    hscale = torch.tensor([0.5], dtype=torch.float32, device=dev)
    hbase = torch.randn(4, dtype=torch.float32, device=dev) * 0.2
    prog_head = compile_hy4_hc_head_aot(HY4_IHC, store_collapsed=True)
    col_h = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    out_h = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
    prog_head.launch(r, hfn, hscale, hbase, norm, col_h, out_h, scalars=(rows,))
    mix_h = (flat64 @ hfn.double().T) * inv
    pre_h = (torch.sigmoid(mix_h * hscale.double() + hbase.double()) + 1e-6).float()
    col_href = (r.float() * pre_h.unsqueeze(-1)).sum(-2).to(torch.bfloat16)
    out_href = _rms_norm(col_href, norm, 1e-5).to(torch.bfloat16)
    torch.cuda.synchronize()
    # 1 BF16 ULP at magnitude ~4.0 is 0.03125
    assert (out_h.float() - out_href.float()).abs().max().item() <= 0.04


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
@pytest.mark.parametrize("heads", [32, 64])
def test_hy4_mla_2048_end_to_end_oracle(kv_cache: str, heads: int):
    dev = require_b12x()
    torch.manual_seed(321 + heads)
    seq_len = 96
    theta = 10_000_000.0
    g = Hy4MlaGeometry(heads=heads, q_lora_rank=2048, norm_eps=1e-6, window_size=None, kv_cache=kv_cache)
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    q_a = torch.randn(seq_len, 2048, dtype=torch.bfloat16, device=dev) * 0.3
    kv_a = torch.randn(seq_len, 576, dtype=torch.bfloat16, device=dev) * 0.3
    q_norm = torch.randn(2048, dtype=torch.bfloat16, device=dev) + 1.0
    kv_norm = torch.randn(512, dtype=torch.bfloat16, device=dev) + 1.0
    q_raw = torch.randn(seq_len, heads, 256, dtype=torch.bfloat16, device=dev) * 0.3
    w_uk = torch.randn(heads, 512, 192, dtype=torch.bfloat16, device=dev) * 0.05
    w_uv = torch.randn(heads, 256, 512, dtype=torch.bfloat16, device=dev) * 0.05
    gate = torch.randn(seq_len, heads, 256, dtype=torch.bfloat16, device=dev) * 0.5
    sink = torch.randn(heads, dtype=torch.float32, device=dev) * 0.5

    # Mix text positions and 3D image M-RoPE positions
    pos_thw = _thw(torch.arange(seq_len, dtype=torch.int64, device=dev))
    for i in range(16, 40):
        pos_thw[i, 0] = 16
        pos_thw[i, 1] = 16 + (i - 16) // 6
        pos_thw[i, 2] = 16 + (i - 16) % 6
    slots = torch.arange(seq_len, dtype=torch.int64, device=dev)
    pages = 4
    cache = torch.zeros(
        pages, 64, 656 if kv_cache == "fp8" else 576,
        dtype=torch.uint8 if kv_cache == "fp8" else torch.bfloat16,
        device=dev,
    )
    q_resid = torch.empty(seq_len, 2048, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_rank_norm_pack_aot(g).launch(
        q_a, kv_a, q_norm, kv_norm, slots, pos_thw, inv_freq, q_resid, cache, scalars=(seq_len,)
    )

    q_resid_ref = _rms_norm(q_a, q_norm, 1e-6).to(torch.bfloat16)
    torch.cuda.synchronize()
    assert (q_resid.float() - q_resid_ref.float()).abs().max().item() <= 0.02

    query = torch.empty(seq_len, heads, 576, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_q_absorb_aot(g).launch(q_raw, w_uk, pos_thw, inv_freq, query, scalars=(seq_len,))

    prog_attn = compile_hy4_mla_attention_aot(g, max_rows=seq_len, max_context=2048, mode="decode", page_size=1)
    scratch = torch.empty(
        max(1024, hy4_attention_layout(g, seq_len, max_context=2048, mode="decode")["scratch_bytes"]),
        dtype=torch.uint8,
        device=dev,
    )
    pt = torch.full((seq_len, seq_len), -1, dtype=torch.int32, device=dev)
    for r in range(seq_len):
        pt[r, : r + 1] = torch.arange(r + 1, dtype=torch.int32, device=dev)
    seqlens = torch.arange(1, seq_len + 1, dtype=torch.int32, device=dev)
    cu = torch.arange(seq_len + 1, dtype=torch.int32, device=dev)
    attn_out = torch.empty(seq_len, heads, 512, dtype=torch.bfloat16, device=dev)
    lse_out = torch.empty(seq_len, heads, dtype=torch.float32, device=dev)
    prog_attn.launch(query, cache, pt, seqlens, cu, attn_out, lse_out, scratch, scalars=(seq_len, seq_len, seq_len, 1))

    uv_out = torch.empty(seq_len, heads, 256, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_sink_gate_uv_aot(g).launch(attn_out, w_uv, gate, lse_out, sink, uv_out, scalars=(seq_len,))
    torch.cuda.synchronize()

    k0_ref = _rms_norm(kv_a[:, :512], kv_norm, 1e-6)
    kr_ref = _interleaved_mrope_tail64(kv_a[:, 512:], pos_thw, theta)
    qn_ref = q_raw[:, :, :192].float()
    qr_ref = _interleaved_mrope_tail64(q_raw[:, :, 192:], pos_thw, theta)
    q_abs = torch.cat([torch.einsum("thd,hld->thl", qn_ref, w_uk.float()), qr_ref], dim=-1)
    k_full = torch.cat([k0_ref, kr_ref], dim=-1)
    scores = torch.einsum("thd,sd->ths", q_abs, k_full) * (256.0 ** -0.5)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=dev))
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    sink_b = sink.view(1, heads, 1).expand(seq_len, heads, 1)
    probs = torch.softmax(torch.cat([scores, sink_b], dim=-1), dim=-1)[..., :-1]
    ctx = torch.einsum("ths,sl->thl", probs, k0_ref)
    uv_ref = torch.einsum("thl,hvl->thv", ctx, w_uv.float()) * torch.sigmoid(gate.float())
    tol = 0.08 if kv_cache == "fp8" else 0.03
    assert (uv_out.float() - uv_ref).abs().max().item() <= tol


@pytest.mark.parametrize("kv_cache", ["bf16", "fp8"])
def test_hy4_mla_64h_context_split_oracle(kv_cache: str):
    dev = require_b12x()
    torch.manual_seed(555)
    seq_len = 160
    theta = 10_000_000.0
    g64 = Hy4MlaGeometry(heads=64, q_lora_rank=2048, norm_eps=1e-6, window_size=None, kv_cache=kv_cache)
    g32 = Hy4MlaGeometry(heads=32, q_lora_rank=2048, norm_eps=1e-6, window_size=None, kv_cache=kv_cache)
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    q_a = torch.randn(seq_len, 2048, dtype=torch.bfloat16, device=dev) * 0.25
    kv_a = torch.randn(seq_len, 576, dtype=torch.bfloat16, device=dev) * 0.25
    q_norm = torch.ones(2048, dtype=torch.bfloat16, device=dev)
    kv_norm = torch.ones(512, dtype=torch.bfloat16, device=dev)
    q_raw = torch.randn(seq_len, 64, 256, dtype=torch.bfloat16, device=dev) * 0.25
    w_uk = torch.randn(64, 512, 192, dtype=torch.bfloat16, device=dev) * 0.04
    w_uv = torch.randn(64, 256, 512, dtype=torch.bfloat16, device=dev) * 0.04
    gate = torch.randn(seq_len, 64, 256, dtype=torch.bfloat16, device=dev) * 0.4
    sink = torch.randn(64, dtype=torch.float32, device=dev) * 0.5

    positions = torch.arange(seq_len, dtype=torch.int64, device=dev)
    pos_thw = _thw(positions)
    # Parity-split pages across rank 0 (even logical pages 0, 2) and rank 1 (odd logical page 1).
    slots_r0 = torch.tensor([(p // 128) * 64 + (p % 64) if (p // 64) % 2 == 0 else -1 for p in range(seq_len)],
                            dtype=torch.int64, device=dev)
    slots_r1 = torch.tensor([(p // 128) * 64 + (p % 64) if (p // 64) % 2 == 1 else -1 for p in range(seq_len)],
                            dtype=torch.int64, device=dev)
    dtype = torch.uint8 if kv_cache == "fp8" else torch.bfloat16
    width = 656 if kv_cache == "fp8" else 576
    cache0 = torch.zeros(4, 64, width, dtype=dtype, device=dev)
    cache1 = torch.zeros(4, 64, width, dtype=dtype, device=dev)
    q_resid = torch.empty(seq_len, 2048, dtype=torch.bfloat16, device=dev)
    pack = compile_hy4_mla_rank_norm_pack_aot(g64)
    pack.launch(q_a, kv_a, q_norm, kv_norm, slots_r0, pos_thw, inv_freq, q_resid, cache0, scalars=(seq_len,))
    pack.launch(q_a, kv_a, q_norm, kv_norm, slots_r1, pos_thw, inv_freq, q_resid, cache1, scalars=(seq_len,))

    query = torch.empty(seq_len, 64, 576, dtype=torch.bfloat16, device=dev)
    compile_hy4_mla_q_absorb_aot(g64).launch(q_raw, w_uk, pos_thw, inv_freq, query, scalars=(seq_len,))

    pt0 = torch.full((seq_len, seq_len), -1, dtype=torch.int32, device=dev)
    pt1 = torch.full((seq_len, seq_len), -1, dtype=torch.int32, device=dev)
    len0 = torch.zeros(seq_len, dtype=torch.int32, device=dev)
    len1 = torch.zeros(seq_len, dtype=torch.int32, device=dev)
    for r in range(seq_len):
        s0 = [int(slots_r0[j]) for j in range(r + 1) if slots_r0[j] >= 0]
        s1 = [int(slots_r1[j]) for j in range(r + 1) if slots_r1[j] >= 0]
        len0[r] = len(s0)
        len1[r] = len(s1)
        if s0:
            pt0[r, : len(s0)] = torch.tensor(s0, dtype=torch.int32, device=dev)
        if s1:
            pt1[r, : len(s1)] = torch.tensor(s1, dtype=torch.int32, device=dev)

    cu = torch.arange(seq_len + 1, dtype=torch.int32, device=dev)
    p0 = compile_hy4_mla_dsa_partial_aot(g64, max_rows=256, head_begin=0, head_count=32, num_splits=16, row_chunk_size=64)
    p32 = compile_hy4_mla_dsa_partial_aot(g64, max_rows=256, head_begin=32, head_count=32, num_splits=16, row_chunk_size=64)
    comb = compile_lse_combine2_aot(heads=32, has_sink=True)
    gate_uv = compile_hy4_mla_gate_uv_aot(g32)
    scratch = torch.empty(p0.scratch_bytes(256)["scratch"], dtype=torch.uint8, device=dev)

    o_parts = []
    for head_begin, partial in ((0, p0), (32, p32)):
        o0 = torch.empty(seq_len, 32, 512, dtype=torch.bfloat16, device=dev)
        l0 = torch.empty(seq_len, 32, dtype=torch.float32, device=dev)
        o1 = torch.empty(seq_len, 32, 512, dtype=torch.bfloat16, device=dev)
        l1 = torch.empty(seq_len, 32, dtype=torch.float32, device=dev)
        partial.launch(query, cache0, pt0, len0, cu, o0, l0, scratch, scalars=(seq_len, seq_len, 4))
        partial.launch(query, cache1, pt1, len1, cu, o1, l1, scratch, scalars=(seq_len, seq_len, 4))
        merged = torch.empty(seq_len, 32, 512, dtype=torch.bfloat16, device=dev)
        comb.launch(o0, l0, o1, l1, sink[head_begin : head_begin + 32].contiguous(), merged, scalars=(seq_len,))
        uv = torch.empty(seq_len, 32, 256, dtype=torch.bfloat16, device=dev)
        gate_uv.launch(
            merged,
            w_uv[head_begin : head_begin + 32].contiguous(),
            gate[:, head_begin : head_begin + 32].contiguous(),
            uv,
            scalars=(seq_len,),
        )
        o_parts.append(uv)
    torch.cuda.synchronize()
    uv_all = torch.cat(o_parts, dim=1)

    k0_ref = _rms_norm(kv_a[:, :512], kv_norm, 1e-6)
    kr_ref = _interleaved_mrope_tail64(kv_a[:, 512:], pos_thw, theta)
    qn_ref = q_raw[:, :, :192].float()
    qr_ref = _interleaved_mrope_tail64(q_raw[:, :, 192:], pos_thw, theta)
    q_abs = torch.cat([torch.einsum("thd,hld->thl", qn_ref, w_uk.float()), qr_ref], dim=-1)
    k_full = torch.cat([k0_ref, kr_ref], dim=-1)
    scores = torch.einsum("thd,sd->ths", q_abs, k_full) * (256.0 ** -0.5)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=dev))
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    sink_b = sink.view(1, 64, 1).expand(seq_len, 64, 1)
    probs = torch.softmax(torch.cat([scores, sink_b], dim=-1), dim=-1)[..., :-1]
    ctx = torch.einsum("ths,sl->thl", probs, k0_ref)
    uv_ref = torch.einsum("thl,hvl->thv", ctx, w_uv.float()) * torch.sigmoid(gate.float())
    tol = 0.08 if kv_cache == "fp8" else 0.03
    assert (uv_all.float() - uv_ref).abs().max().item() <= tol


def test_hy4_index_post_exact_and_topk_oracle():
    """Verify Hy4IndexPost against F32 reference (text + M-RoPE) and through glm_index_topk + ContextSplit."""
    dev = require_b12x()
    torch.manual_seed(20261125)
    seq_len = 2560
    heads, dim, topk = 32, 128, 2048
    theta = 10_000_000.0
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    iq = torch.randn(seq_len, heads, dim, dtype=torch.bfloat16, device=dev) * 0.5
    k_raw = torch.randn(seq_len, dim, dtype=torch.bfloat16, device=dev) * 0.5
    w_raw = torch.randn(seq_len, heads, dtype=torch.bfloat16, device=dev) * 0.5
    k_weight = torch.randn(dim, dtype=torch.bfloat16, device=dev) + 1.0
    k_bias = torch.randn(dim, dtype=torch.bfloat16, device=dev) * 0.2

    pos_thw = _thw(torch.arange(seq_len, dtype=torch.int64, device=dev))
    # Inject a 3D M-RoPE image span to verify multi-axis coordinates on both Q and K.
    for i in range(100, 148):
        pos_thw[i, 0] = 100
        pos_thw[i, 1] = 100 + (i - 100) // 8
        pos_thw[i, 2] = 100 + (i - 100) % 8
    slots = torch.arange(seq_len, dtype=torch.int64, device=dev)
    pages = (seq_len + 63) // 64
    cache = torch.zeros(pages, 8448, dtype=torch.uint8, device=dev)
    q_fp8 = torch.empty(seq_len, heads, dim, dtype=torch.float8_e4m3fn, device=dev)
    head_weights = torch.empty(seq_len, heads, dtype=torch.float32, device=dev)

    post = compile_hy4_index_post_aot(HY4_INDEXER)
    post.launch(iq, k_raw, w_raw, pos_thw, slots, inv_freq, k_weight, k_bias, q_fp8, head_weights, cache,
                scalars=(seq_len,))
    torch.cuda.synchronize()

    # Dequantize Q and K back to F32 and compare against exact F32 Hy4 reference math.
    q_ref = _interleaved_mrope_tail64(iq, pos_thw, theta)
    k_ln = F.layer_norm(k_raw.float(), (dim,), k_weight.float(), k_bias.float(), 1e-5)
    k_ref = _interleaved_mrope_tail64(k_ln, pos_thw, theta)
    w_ref = w_raw.float() * (heads ** -0.5) * (dim ** -0.5)

    # Check K cache page layout (64x128 E4M3 + 64 FP32 scales per 8448B page)
    k_deq = torch.empty(seq_len, dim, dtype=torch.float32, device=dev)
    for p in range(pages):
        n = min(64, seq_len - p * 64)
        codes = cache[p, : 64 * 128].view(torch.float8_e4m3fn).view(64, 128)[:n].float()
        scales = cache[p, 64 * 128 :].view(torch.float32)[:n].unsqueeze(-1)
        k_deq[p * 64 : p * 64 + n] = codes * scales
    k_rel = (k_deq - k_ref).norm() / k_ref.norm()
    assert k_rel.item() < 0.04, f"Hy4IndexPost K E4M3 rel L2 too high: {k_rel.item()}"

    # Check Q * w_folded against exact F32 scores on tail rows
    q_scale_expected = q_ref.to(torch.bfloat16).float().abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    w_expected = w_ref * q_scale_expected
    assert torch.allclose(head_weights, w_expected, atol=2e-3, rtol=2e-2)

    # Verify slot < 0 leaves cache untouched
    sentinel = cache.clone()
    neg_slots = torch.full((4,), -1, dtype=torch.int64, device=dev)
    post.launch(iq[:4], k_raw[:4], w_raw[:4], pos_thw[:4], neg_slots, inv_freq, k_weight, k_bias,
                q_fp8[:4], head_weights[:4], cache, scalars=(4,))
    torch.cuda.synchronize()
    assert torch.equal(cache, sentinel)

    # End-to-end through glm_index_topk (prefill + decode) on tail queries (> 2048 tokens)
    tail_rows = 32
    tail_start = seq_len - tail_rows
    topk_prefill = compile_glm_index_topk_aot(GLM53, max_rows=tail_rows, max_pages=pages, mode="prefill")
    scratch_p = torch.zeros(topk_prefill.scratch_bytes(tail_rows)["scratch"], dtype=torch.uint8, device=dev)
    pt_p = torch.arange(pages, dtype=torch.int32, device=dev)
    lens_p = torch.arange(tail_start + 1, seq_len + 1, dtype=torch.int32, device=dev)
    out_idx = torch.empty(tail_rows, topk, dtype=torch.int32, device=dev)
    topk_prefill.launch(
        q_fp8[tail_start:].contiguous(), head_weights[tail_start:].contiguous(),
        cache, pt_p, lens_p, out_idx, scratch_p, scalars=(tail_rows, pages, pages),
    )
    torch.cuda.synchronize()

    # Compare recalled top-2048 slots against F32 reference scores
    ref_scores = torch.einsum(
        "rh,rhk->rk",
        w_ref[tail_start:],
        F.relu(torch.einsum("rhd,kd->rhk", q_ref[tail_start:], k_ref)),
    )
    overlaps = []
    for r in range(tail_rows):
        vis = tail_start + r + 1
        got = out_idx[r].tolist()
        assert all(0 <= s < vis for s in got), f"non-causal or invalid slot in row {r}"
        assert len(set(got)) == topk, f"duplicate slots in row {r}"
        ref_top = set(ref_scores[r, :vis].topk(topk).indices.tolist())
        overlaps.append(len(ref_top.intersection(got)) / topk)
    assert min(overlaps) >= 0.96, f"low top-2048 overlap vs F32 ref: min={min(overlaps):.4f}"

    # Verify ContextSplit parity-sharded scored_index_topk + dsa_candidate_merge matches single-GPU topk
    half_pages = pages // 2
    cache_e = cache[0::2].contiguous()
    cache_o = cache[1::2].contiguous()
    scored = compile_scored_index_topk_aot(GLM53, max_rows=tail_rows, max_pages=half_pages, mode="decode")
    merge = compile_dsa_candidate_merge_aot(topk=topk, unit_rows=64)
    sc_buf = torch.zeros(scored.scratch_bytes(tail_rows)["scratch"], dtype=torch.uint8, device=dev)
    pt_half = torch.arange(half_pages, dtype=torch.int32, device=dev).unsqueeze(0).expand(tail_rows, half_pages).contiguous()
    # Count local visible tokens per parity shard for each tail row
    len_e = torch.tensor(
        [sum(min(64, (tail_start + r + 1) - p * 64) for p in range(0, pages, 2) if p * 64 < tail_start + r + 1)
         for r in range(tail_rows)],
        dtype=torch.int32, device=dev,
    )
    len_o = torch.tensor(
        [sum(min(64, (tail_start + r + 1) - p * 64) for p in range(1, pages, 2) if p * 64 < tail_start + r + 1)
         for r in range(tail_rows)],
        dtype=torch.int32, device=dev,
    )
    idx_e = torch.empty(tail_rows, topk, dtype=torch.int32, device=dev)
    scr_e = torch.empty(tail_rows, topk, dtype=torch.float32, device=dev)
    idx_o = torch.empty(tail_rows, topk, dtype=torch.int32, device=dev)
    scr_o = torch.empty(tail_rows, topk, dtype=torch.float32, device=dev)
    scored.launch(q_fp8[tail_start:].contiguous(), head_weights[tail_start:].contiguous(),
                  cache_e, pt_half, len_e, idx_e, scr_e, sc_buf,
                  scalars=(tail_rows, half_pages, half_pages, 2, 0, half_pages))
    scored.launch(q_fp8[tail_start:].contiguous(), head_weights[tail_start:].contiguous(),
                  cache_o, pt_half, len_o, idx_o, scr_o, sc_buf,
                  scalars=(tail_rows, half_pages, half_pages, 2, 1, half_pages))
    all_s = torch.cat([scr_e, scr_o], dim=1).contiguous()
    all_i = torch.cat([idx_e, idx_o], dim=1).contiguous()
    out_s = torch.empty(tail_rows, topk, dtype=torch.float32, device=dev)
    out_i = torch.empty(tail_rows, topk, dtype=torch.int32, device=dev)
    loc_slots = torch.empty(tail_rows, topk, dtype=torch.int32, device=dev)
    loc_lens = torch.empty(tail_rows, dtype=torch.int32, device=dev)
    merge.launch(all_s, all_i, out_s, out_i, pt_half, loc_slots, loc_lens, scalars=(tail_rows, half_pages, 0))
    torch.cuda.synchronize()
    for r in range(tail_rows):
        assert set(out_i[r].tolist()) == set(out_idx[r].tolist()), f"ContextSplit topk mismatch at row {r}"


def test_hy4_vision_dynamic_attention_16x96_oracle():
    dev = require_b12x()
    torch.manual_seed(96)
    heads, dim, segments = 16, 96, 8
    seg_lens = [192, 320, 64]
    rows = sum(seg_lens)
    q = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    k = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    v = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=dev) * 0.3
    out = torch.empty_like(q)
    lse = torch.empty(heads, rows, dtype=torch.float32, device=dev)
    cum = [0]
    for s in seg_lens:
        cum.append(cum[-1] + s)
    while len(cum) < segments + 1:
        cum.append(cum[-1])
    lengths = torch.tensor(cum, dtype=torch.int32, device=dev)
    scale = dim ** -0.5
    l2_bytes = torch.cuda.get_device_properties(dev).L2_cache_size

    prog = compile_dynamic_vision_attention_aot(heads=heads, dim=dim, segments=segments)
    prog.launch(q, k, v, out, lse, lengths, scalars=(rows, scale, l2_bytes))
    torch.cuda.synchronize()

    ref = torch.empty_like(q, dtype=torch.float32)
    off = 0
    for s in seg_lens:
        qs = q[off : off + s].float().transpose(0, 1)
        ks = k[off : off + s].float().transpose(0, 1)
        vs = v[off : off + s].float().transpose(0, 1)
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        ref[off : off + s] = torch.matmul(probs, vs).transpose(0, 1)
        off += s
    assert (out.float() - ref).abs().max().item() <= 0.02


@pytest.mark.skipif(not os.environ.get("HY4_ROW_BAND"), reason="opt-in Hy4 SM120 row-band timing")
def test_hy4_full_row_band():
    dev = require_b12x()
    props = torch.cuda.get_device_properties(dev)
    sm_count = props.multi_processor_count
    row_list = (1, 8, 16, 40, 64, 2048, 2688, 2752, 4096)
    theta = 10_000_000.0
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64.0))).contiguous()

    post_prog = compile_hy4_index_post_aot(HY4_INDEXER)
    ihc_pp = compile_hy4_ihc_post_pre_aot(HY4_IHC)
    g64_fp8 = Hy4MlaGeometry(heads=64, q_lora_rank=2048, norm_eps=1e-6, kv_cache="fp8")
    pack_prog = compile_hy4_mla_rank_norm_pack_aot(g64_fp8)
    partial_fp8 = compile_hy4_mla_dsa_partial_aot(g64_fp8, max_rows=4096, head_begin=0, head_count=32, row_chunk_size=64)

    max_r = max(row_list)
    pages = (max_r + 63) // 64
    iq = torch.randn(max_r, 32, 128, dtype=torch.bfloat16, device=dev)
    k_raw = torch.randn(max_r, 128, dtype=torch.bfloat16, device=dev)
    w_raw = torch.randn(max_r, 32, dtype=torch.bfloat16, device=dev)
    pos_thw = _thw(torch.arange(max_r, dtype=torch.int64, device=dev))
    slots = torch.arange(max_r, dtype=torch.int64, device=dev)
    k_w = torch.ones(128, dtype=torch.bfloat16, device=dev)
    k_b = torch.zeros(128, dtype=torch.bfloat16, device=dev)
    q_fp8 = torch.empty(max_r, 32, 128, dtype=torch.float8_e4m3fn, device=dev)
    hw = torch.empty(max_r, 32, dtype=torch.float32, device=dev)
    idx_cache = torch.zeros(pages, 8448, dtype=torch.uint8, device=dev)

    delta = torch.randn(max_r, 6144, dtype=torch.bfloat16, device=dev)
    resid = torch.randn(max_r, 4, 6144, dtype=torch.bfloat16, device=dev)
    prev_p = torch.ones(max_r, 4, dtype=torch.float32, device=dev)
    fn = torch.randn(8, 24576, dtype=torch.float32, device=dev) * 0.01
    sc = torch.ones(2, dtype=torch.float32, device=dev)
    bs = torch.zeros(8, dtype=torch.float32, device=dev)
    nm = torch.ones(6144, dtype=torch.bfloat16, device=dev)
    resid_o = torch.empty_like(resid)
    post_o = torch.empty_like(prev_p)
    y_o = torch.empty_like(delta)

    q_a = torch.randn(max_r, 2048, dtype=torch.bfloat16, device=dev)
    kv_a = torch.randn(max_r, 576, dtype=torch.bfloat16, device=dev)
    q_nm = torch.ones(2048, dtype=torch.bfloat16, device=dev)
    kv_nm = torch.ones(512, dtype=torch.bfloat16, device=dev)
    q_res = torch.empty_like(q_a)
    kv_cache = torch.zeros(pages, 64, 656, dtype=torch.uint8, device=dev)

    query = torch.randn(max_r, 64, 576, dtype=torch.bfloat16, device=dev)
    pt_tok = torch.arange(1024, dtype=torch.int32, device=dev).unsqueeze(0).expand(max_r, 1024).contiguous()
    seqlens = torch.full((max_r,), 1024, dtype=torch.int32, device=dev)
    cu = torch.arange(max_r + 1, dtype=torch.int32, device=dev)
    o0 = torch.empty(max_r, 32, 512, dtype=torch.bfloat16, device=dev)
    l0 = torch.empty(max_r, 32, dtype=torch.float32, device=dev)
    scratch = torch.empty(partial_fp8.scratch_bytes(max_r)["scratch"], dtype=torch.uint8, device=dev)

    stream = torch.cuda.Stream(device=dev)
    results = []
    for rows in row_list:
        replays = 16 if rows <= 64 else 6
        splits = max(2, min(16, (sm_count + rows * 4 - 1) // (rows * 4)))
        arms = {
            "hy4_index_post": lambda r=rows: post_prog.launch(
                iq, k_raw, w_raw, pos_thw, slots, inv_freq, k_w, k_b, q_fp8, hw, idx_cache, scalars=(r,)
            ),
            "hy4_ihc_post_pre": lambda r=rows: ihc_pp.launch(
                delta, resid, prev_p, fn, sc, bs, nm, resid_o, post_o, y_o, scalars=(r,)
            ),
            "hy4_mla_rank_norm_pack_fp8": lambda r=rows: pack_prog.launch(
                q_a, kv_a, q_nm, kv_nm, slots, pos_thw, inv_freq, q_res, kv_cache, scalars=(r,)
            ),
            "hy4_mla_dsa_partial_fp8_32h": lambda r=rows, s=splits: partial_fp8.launch(
                query, kv_cache, pt_tok, seqlens, cu, o0, l0, scratch, scalars=(r, 1024, s)
            ),
        }
        med = {}
        for name, work in arms.items():
            with torch.cuda.stream(stream):
                for _ in range(3):
                    work()
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                work()
            samples = []
            for _ in range(5):
                for _ in range(2):
                    g.replay()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(replays):
                    g.replay()
                e_ev.record()
                e_ev.synchronize()
                samples.append(s_ev.elapsed_time(e_ev) * 1000.0 / replays)
            med[name] = statistics.median(samples)
        entry = {
            "rows": rows,
            "us": {k: round(v, 2) for k, v in med.items()},
            "us_per_row": {k: round(v / rows, 4) for k, v in med.items()},
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)

    band = [r for r in results if r["rows"] in (2048, 2688, 2752, 4096)]
    for key in ("hy4_index_post", "hy4_ihc_post_pre", "hy4_mla_rank_norm_pack_fp8", "hy4_mla_dsa_partial_fp8_32h"):
        vals = [r["us_per_row"][key] for r in band]
        # No adjacent-size per-row slowdown > 10% across the 2048..4096 policy band.
        for idx in range(1, len(vals)):
            assert vals[idx] <= vals[idx - 1] * 1.10, (
                f"{key} adjacent-size per-row slowdown at index {idx}: {vals}"
            )
        if key == "hy4_index_post":
            assert max(vals) <= min(vals) * 1.10, f"{key} high-row band collapse: {vals}"
