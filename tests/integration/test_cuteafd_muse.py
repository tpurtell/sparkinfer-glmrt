"""Muse coordinator rounding boundaries and live-row invariance."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
import torch
from ..conftest import require_b12x

ROWS = (1, 16, 80, 2752, 4096)


def _canonical_norm(x, eps=1e-5):
    """Independent CPU transcription of reference.rs sum_squares and inv_rms."""
    width = x.shape[-1]
    work = x.float().reshape(-1, width)
    lanes = torch.zeros(work.shape[0], 8, 32)
    for slab in range((width + 127) // 128):
        for j in range(4):
            at = slab * 128 + torch.arange(32) * 4 + j
            valid = at < width
            lanes[:, slab % 8, valid] += work[:, at[valid]].square()
    for stage in range(5):
        lanes = lanes + lanes[:, :, torch.arange(32) ^ (1 << stage)]
    total = torch.zeros(work.shape[0])
    for warp in range(8):
        total = total + lanes[:, warp, 0]
    inv = 1.0 / ((total / width) + eps).sqrt()
    return (work * inv[:, None]).reshape_as(x)


def _bf16_ulp_close(actual, expected, count=1):
    a, e = actual.float(), expected.float()
    ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(2.0 ** -126))) - 7)
    assert ((a - e).abs() <= count * ulp).all(), ((a - e).abs().max(), ((a - e).abs() / ulp).max())


@pytest.mark.parametrize("rows", ROWS)
def test_residual_gate(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_residual_gate_aot
    gen = torch.Generator().manual_seed(31)
    residual = torch.randn(rows, 8192, generator=gen).bfloat16()
    x = torch.randn(rows, 8192, generator=gen).bfloat16()
    gate = (torch.randn(8192, generator=gen) * 0.2).bfloat16()
    w = gate.float() / 0.3
    beta = 1 / (1 + torch.exp(-w))
    alpha = ((1 / (1 + torch.exp(w))) * (1 + beta)).clamp_min(1e-3).sqrt()
    expected = (alpha * residual.float() + beta * x.float()).bfloat16()
    r, h, a, b = (t.cuda() for t in (residual, x, alpha, beta))
    out = torch.empty_like(r)
    program = compile_residual_gate_aot()
    program.launch(r, h, a, b, out, scalars=[rows])
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), expected), ((out.cpu().float() - expected.float()).abs().max(),
        torch.nonzero(out.cpu() != expected)[:10], out.cpu()[out.cpu() != expected][:10], expected[out.cpu() != expected][:10])
    one = torch.empty_like(out[:1])
    program.launch(r[:1], h[:1], a, b, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_route_select(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_route_select_aot
    gen = torch.Generator().manual_seed(32)
    logits = torch.randn(rows, 256, generator=gen)
    bias = torch.randn(256, generator=gen) * 0.01
    logits[0] = 0
    bias[:16] = 0.25
    # Rust f32::exp uses scalar libm expf, unlike Torch CPU's vector exp.
    expf = ctypes.CDLL("libm.so.6").expf
    expf.argtypes, expf.restype = (ctypes.c_float,), ctypes.c_float
    scaled = (logits * 0.1767766952966369).numpy()
    exponent = np.fromiter((expf(float(-v)) for v in scaled.flat), dtype=np.float32).reshape(scaled.shape)
    scores = torch.from_numpy(np.float32(1.0) / (np.float32(1.0) + exponent))
    ids = torch.argsort(scores + bias, descending=True, stable=True)[:, :8].int()
    selected = scores.gather(1, ids.long())
    total = torch.zeros(rows)
    for j in range(8):
        total = total + selected[:, j]
    expected = selected / (total[:, None] + 1e-15)
    l, b = logits.cuda(), bias.cuda()
    out_ids = torch.empty_like(ids, device="cuda")
    weights = torch.empty(rows, 8, device="cuda")
    program = compile_route_select_aot()
    program.launch(l, b, out_ids, weights, scalars=[rows])
    torch.cuda.synchronize()
    assert torch.equal(out_ids.cpu(), ids)
    actual = weights.cpu()
    ulps = (actual.view(torch.int32) - expected.view(torch.int32)).abs()
    assert ulps.max() <= 2, (ulps.max(), (actual - expected).abs().max())
    # Every row must retain its exact weights outside its original live batch.
    for start in range(0, rows, 16):
        count = min(16, rows - start)
        small_ids, small_weights = torch.empty_like(out_ids[:count]), torch.empty_like(weights[:count])
        program.launch(l[start:start + count], b, small_ids, small_weights, scalars=[count])
        assert torch.equal(small_ids, out_ids[start:start + count])
        assert torch.equal(small_weights, weights[start:start + count])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("kind", ("swa", "full"))
def test_qk_norm_rope(rows, kind):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_qk_norm_rope_aot
    gen = torch.Generator().manual_seed(33)
    q = torch.randn(rows, 128, 64, generator=gen).bfloat16()
    k = torch.randn(rows, 16, 64, generator=gen).bfloat16()
    positions = torch.arange(rows, dtype=torch.int64)
    freq = positions.float()[:, None] / (500000 ** (torch.arange(0, 64, 2).float() / 64))
    angle = torch.cat((freq, freq), -1)
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()

    def norm(t):
        return _canonical_norm(t).bfloat16()

    expected_q, expected_k = (norm(q) * 5.4730064863875).bfloat16(), norm(k)
    if kind == "swa":
        def rope(t):
            half = torch.cat((-t[..., 32:], t[..., :32]), -1)
            return t * cos[:, None] + half * sin[:, None]
        expected_q, expected_k = rope(expected_q), rope(expected_k)
    operands = [t.cuda() for t in (q, k, positions, cos, sin)]
    qo, ko = torch.empty_like(operands[0]), torch.empty_like(operands[1])
    program = compile_qk_norm_rope_aot(kind=kind)
    program.launch(*operands, qo, ko, scalars=[rows])
    # Reduction differences are examined separately; all explicit BF16 steps are checked here.
    for actual, expected in ((qo.cpu(), expected_q), (ko.cpu(), expected_k)):
        assert torch.equal(actual, expected), ((actual != expected).sum(), (actual.float() - expected.float()).abs().max())
    one_q, one_k = torch.empty_like(qo[:1]), torch.empty_like(ko[:1])
    program.launch(operands[0][:1], operands[1][:1], *operands[2:], one_q, one_k, scalars=[1])
    assert torch.equal(one_q, qo[:1]) and torch.equal(one_k, ko[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_attention_output(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_attention_output_aot
    gen = torch.Generator().manual_seed(34)
    x = torch.randn(rows, 128, 64, generator=gen).bfloat16()
    gate = torch.randn(rows, 128, 64, generator=gen).bfloat16()
    norm = _canonical_norm(x).bfloat16()
    expected = norm * gate.sigmoid()
    xd, gd = x.cuda(), gate.cuda()
    out = torch.empty_like(xd)
    program = compile_attention_output_aot()
    program.launch(xd, gd, out, scalars=[rows])
    _bf16_ulp_close(out.cpu(), expected)
    one = torch.empty_like(out[:1])
    program.launch(xd[:1], gd[:1], one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("op", ("embed_norm", "input_norm", "pre_ffn_norm", "pre_expert_norm", "post_attn_norm", "post_ffn_norm", "final_norm"))
def test_row_norms(rows, op):
    require_b12x()
    from b12x.integration.cuteafd.muse import NORMS, compile_norm_aot
    width, mode, eps = NORMS[op]
    gen = torch.Generator().manual_seed(36)
    x = torch.randn(rows, width, generator=gen).bfloat16()
    weight = (torch.randn(width, generator=gen) * 0.1).bfloat16()
    scale = (1 + weight).float()
    source = (x.float() * scale).bfloat16() if mode == "input_scaled" else x
    expected = _canonical_norm(source, eps)
    if mode == "centered":
        expected = expected * scale
    elif mode == "scaled":
        expected = expected * weight.float()
    expected = expected.bfloat16()
    xd, wd = x.cuda(), weight.cuda()
    out = torch.empty_like(xd)
    args = (xd,) if mode == "weightless" else (xd, wd)
    program = compile_norm_aot(op=op)
    program.launch(*args, out, scalars=[rows])
    assert torch.equal(out.cpu(), expected), ((out.cpu() != expected).sum(), (out.cpu().float() - expected.float()).abs().max())
    one = torch.empty_like(out[:1])
    args = (xd[:1],) if mode == "weightless" else (xd[:1], wd)
    program.launch(*args, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_softcap(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_softcap_aot
    # Include every finite BF16 code across rows; a compact head exercises tail masks.
    vocab = 257
    codes = torch.arange(rows * vocab, dtype=torch.int32)
    values = (codes.remainder(0x7f80) | ((codes % 2) << 15)).to(torch.int16)
    x = values.view(torch.bfloat16).reshape(rows, vocab)
    scaled = (x * 0.17677669529663687) / 20
    expected = (scaled.tanh() * 20).float()
    xd = x.float().cuda()
    out = torch.empty_like(xd, dtype=torch.float32)
    program = compile_softcap_aot(vocab=vocab)
    program.launch(xd, out, scalars=[rows])
    _bf16_ulp_close(out.cpu(), expected)
    one = torch.empty_like(out[:1])
    program.launch(xd[:1], one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("rows", ROWS)
def test_native_f32_router(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_router_aot
    gen = torch.Generator().manual_seed(37)
    x = torch.randn(rows, 8192, generator=gen).bfloat16()
    weight = torch.randn(256, 8192, generator=gen) * 0.01
    # The checkpoint's low F32 bits must influence the result.
    expected = x.float() @ weight.T
    xd, wd = x.cuda(), weight.cuda()
    out = torch.empty(rows, 256, device="cuda")
    program = compile_router_aot()
    program.launch(xd, wd, out, scalars=[rows])
    torch.testing.assert_close(out.cpu(), expected, atol=3e-6, rtol=3e-4)
    one = torch.empty_like(out[:1])
    program.launch(xd[:1], wd, one, scalars=[1])
    assert torch.equal(one, out[:1])


@pytest.mark.parametrize("heads", (64, 128))
@pytest.mark.parametrize("history", (2050, 6140))
@pytest.mark.parametrize("kind", ("full", "swa"))
def test_attention_oracle_and_unit_invariance(kind, history, heads):
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot, compile_swa_commit_aot
    gen = torch.Generator().manual_seed(35)
    rows, kv_heads, ring_rows = 17, heads // 8, 6144
    end = history + rows
    query = torch.randn(rows, heads, 64, generator=gen).bfloat16().cuda()
    keys = torch.randn(end, kv_heads, 64, generator=gen).bfloat16()
    values = torch.randn(end, kv_heads, 64, generator=gen).bfloat16()
    records = torch.cat((keys.flatten(1), values.flatten(1)), -1).cuda()
    positions = torch.arange(history, end, dtype=torch.int64, device="cuda")
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    prefill = compile_attention_aot(heads=heads, kind=kind, route="prefill")
    decode = compile_attention_aot(heads=heads, kind=kind, route="decode")
    out = torch.empty_like(query)
    singles = torch.empty_like(query)
    if kind == "full":
        cache = torch.zeros(((end + 63) // 64 * 64, kv_heads * 128), dtype=torch.bfloat16, device="cuda")
        cache[:end].copy_(records)
        table = torch.arange(cache.shape[0] // 64, dtype=torch.int32, device="cuda")
        prefill.launch(query, cache, positions, table, out, scratch, scalars=[rows, 0])
        for row in range(rows):
            decode.launch(query[row:row + 1], cache, positions[row:row + 1], table,
                          singles[row:row + 1], scratch, scalars=[1, 0, 1])
    else:
        ring = torch.zeros(ring_rows, kv_heads * 128, dtype=torch.bfloat16, device="cuda")
        prior = torch.arange(max(0, history - 2048), history, device="cuda")
        ring[prior % ring_rows] = records[prior]
        slots = positions % ring_rows
        seq_first = torch.zeros(rows, dtype=torch.int32, device="cuda")
        step = records[history:].contiguous()
        prefill.launch(query, step, ring, positions, slots, seq_first, out, scratch, scalars=[rows])
        commit = compile_swa_commit_aot(heads=heads)
        for row in range(rows):
            decode.launch(query[row:row + 1], step[row:row + 1], ring, positions[row:row + 1], slots[row:row + 1],
                          seq_first[:1], singles[row:row + 1], scratch, scalars=[1])
            commit.launch(step[row:row + 1], ring, slots[row:row + 1], seq_first[:1], scalars=[1])
    assert torch.equal(out, singles), (out.float() - singles.float()).abs().max()
    from b12x._lib.runtime_control import kernel_resolution_guard
    expected_graph = out.clone()
    args = ((query, cache, positions, table, out, scratch) if kind == "full" else
            (query, step, ring, positions, slots, seq_first, out, scratch))
    scalars = [rows, 0] if kind == "full" else [rows]
    with kernel_resolution_guard("Muse attention reuses precompiled program on graph replay"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prefill.launch(*args, scalars=scalars)
        for _ in range(2):
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected_graph)
    k, v = keys.cuda().repeat_interleave(8, dim=1), values.cuda().repeat_interleave(8, dim=1)
    for row in range(rows):
        position = history + row
        first = max(0, position - 2047) if kind == "swa" else 0
        q = query[row].float()
        kk, vv = k[first:position + 1], v[first:position + 1]
        # Official BF16 QK and scale boundaries; online PV normalization may reorder.
        score = torch.einsum("hd,khd->hk", q, kk.float()).bfloat16()
        score = (score * 0.125).bfloat16()
        prob = torch.softmax(score.float(), -1).bfloat16()
        expected = torch.einsum("hk,khd->hd", prob.float(), vv.float()).bfloat16()
        torch.testing.assert_close(out[row].float(), expected.float(), atol=0.002, rtol=0.02)


@pytest.mark.parametrize("layer,kind", ((0, "swa"), (1, "full")))
def test_engine_attention_exchange(layer, kind):
    import os
    from pathlib import Path
    source = os.environ.get("CUTEAFD_MUSE_ATTENTION_FIXTURE")
    if not source:
        pytest.skip("Muse engine attention exchange fixture not configured")
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot
    root = Path(source)
    def read(leaf, shape):
        bits = np.fromfile(root / f"layer{layer}.{leaf}.bin", dtype="<u2").copy()
        return torch.from_numpy(bits).view(torch.bfloat16).reshape(shape).cuda()
    q = read("q", (9, 8, 64))
    k, v = (read(leaf, (9, 2, 64)) for leaf in ("k", "v"))
    expected = read("out", (9, 8, 64))
    records = torch.cat((k.flatten(1), v.flatten(1)), -1)
    pos = torch.arange(9, dtype=torch.int64, device="cuda")
    out, scratch = torch.empty_like(q), torch.empty(1024, dtype=torch.uint8, device="cuda")
    program = compile_attention_aot(heads=8, kv_heads=2, window=5, kind=kind,
                                    route="prefill", ring_rows=64)
    if kind == "full":
        cache = torch.zeros(64, 256, dtype=torch.bfloat16, device="cuda")
        cache[:9] = records
        table = torch.zeros(1, dtype=torch.int32, device="cuda")
        program.launch(q, cache, pos, table, out, scratch, scalars=[9, 0])
    else:
        ring = torch.zeros(64, 256, dtype=torch.bfloat16, device="cuda")
        first = torch.zeros(9, dtype=torch.int32, device="cuda")
        program.launch(q, records, ring, pos, pos, first, out, scratch, scalars=[9])
    torch.testing.assert_close(out.float(), expected.float(), atol=0.002, rtol=0.02)
    destination = Path(os.environ["CUTEAFD_MUSE_ATTENTION_OUTPUT"])
    destination.mkdir(parents=True, exist_ok=True)
    out.cpu().view(torch.uint16).numpy().tofile(destination / f"layer{layer}.out.bin")

@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("kind", ("full", "swa"))
def test_attention_live_row_ladder(rows, kind):
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot, compile_swa_commit_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    heads, kv, ring_rows = 64, 8, 6144
    gen = torch.Generator(device="cuda").manual_seed(39)
    q = torch.randn(rows, heads, 64, device="cuda", generator=gen).bfloat16()
    records = torch.randn(rows, kv * 128, device="cuda", generator=gen).bfloat16()
    pos = torch.arange(rows, device="cuda", dtype=torch.int64)
    out = torch.empty_like(q)
    scratch = torch.empty(1024, device="cuda", dtype=torch.uint8)
    prefill = compile_attention_aot(heads=heads, kind=kind, route="prefill")
    decode = compile_attention_aot(heads=heads, kind=kind, route="decode")
    if kind == "full":
        cache = torch.zeros((rows + 63) // 64 * 64, kv * 128, device="cuda", dtype=torch.bfloat16)
        cache[:rows] = records
        table = torch.arange(cache.shape[0] // 64, device="cuda", dtype=torch.int32)
        prefill.launch(q, cache, pos, table, out, scratch, scalars=[rows, 0])
    else:
        ring = torch.zeros(ring_rows, kv * 128, device="cuda", dtype=torch.bfloat16)
        first = torch.zeros(rows, device="cuda", dtype=torch.int32)
        prefill.launch(q, records, ring, pos, pos, first, out, scratch, scalars=[rows])
        commit = compile_swa_commit_aot(heads=heads)
        commit.launch(records, ring, pos, first, scalars=[rows])
    one = torch.empty_like(out[:1])
    with kernel_resolution_guard("Muse attention live rows reuse fixed geometry"):
        for row in sorted({0, rows // 2, rows - 1}):
            if kind == "full":
                decode.launch(q[row:row + 1], cache, pos[row:row + 1], table,
                              one, scratch, scalars=[1, 0, 1])
            else:
                decode.launch(q[row:row + 1], records[row:row + 1], ring, pos[row:row + 1],
                              pos[row:row + 1], first[:1], one, scratch, scalars=[1])
            assert torch.equal(one[0], out[row])


def test_full_attention_large_page_id_and_mixed_sequences():
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot
    # The second sequence starts beyond a signed 32-bit byte offset.
    page = 16385
    cache = torch.empty((page + 1) * 64, 1024, device="cuda", dtype=torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(40)
    first = torch.randn(2, 1024, device="cuda", generator=gen).bfloat16()
    second = torch.randn(2, 1024, device="cuda", generator=gen).bfloat16()
    cache[:2], cache[page * 64:page * 64 + 2] = first, second
    table = torch.tensor([0, page], dtype=torch.int32, device="cuda")
    pos = torch.tensor([1, 1], dtype=torch.int64, device="cuda")
    q = torch.randn(2, 64, 64, device="cuda", generator=gen).bfloat16()
    out = torch.empty_like(q)
    scratch = torch.empty(1024, device="cuda", dtype=torch.uint8)
    program = compile_attention_aot(heads=64, kind="full", route="decode")
    program.launch(q, cache, pos, table, out, scratch, scalars=[2, 1, 1])
    one = torch.empty_like(out[:1])
    for row in range(2):
        program.launch(q[row:row + 1], cache, pos[row:row + 1], table[row:row + 1],
                       one, scratch, scalars=[1, 1, 1])
        assert torch.equal(one[0], out[row])
        compact = torch.zeros(64, 1024, device="cuda", dtype=torch.bfloat16)
        compact[:2] = (first, second)[row]
        zero_page = torch.zeros(1, device="cuda", dtype=torch.int32)
        program.launch(q[row:row + 1], compact, pos[row:row + 1], zero_page,
                       one, scratch, scalars=[1, 0, 1])
        assert torch.equal(one[0], out[row])
