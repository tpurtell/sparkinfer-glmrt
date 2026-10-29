"""Bitwise native projection seams against GLM Flash's existing fused stages."""
from dataclasses import replace

import pytest
import torch

from .test_cuteafd_kimi_split import rand, empty, scratch, same, projection
from ..conftest import require_b12x


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    require_b12x()
    torch.manual_seed(836)


@pytest.mark.parametrize("rows,decode", [(1, True), (16, True), (64, True), (129, False)])
@pytest.mark.parametrize("heads", [32, 64])
def test_flash_kda_core_bitwise(rows, decode, heads):
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_kda_aot
    from b12x.integration.cuteafd.glmf_split import compile_glmf_kda_core_aot
    from b12x.integration.cuteafd._glmf_kernels import kda_replay_layout
    g = replace(GLM53_FLASH, kda_heads=heads)
    cap, d, h, p = max(64, rows), g.kda_width, g.hidden, g.kda_in_width
    fused = compile_glmf_kda_aot(g, max_rows=cap, decode=decode, output_kind="norm")
    split = compile_glmf_kda_core_aot(g, max_rows=cap, decode=decode)
    x, w = rand(rows, h), rand(p, h)
    wf, cw = rand(2, d, 128), rand(3*d, 4, dtype=torch.float32)
    al, dt, norm = rand(heads, dtype=torch.float32), rand(d, dtype=torch.float32), rand(128, scale=1)
    proj = empty(rows, p)
    projection(p, h).launch(x, w, proj, scalars=(rows,))
    q, k, v = [proj[:, i*d:(i+1)*d].contiguous() for i in range(3)]
    fab = w[3*d:].contiguous()
    wo = empty(h, d)
    state0, conv0 = rand(2, heads, 128, 128, dtype=torch.float32), rand(2, 3, 3*d)
    slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    first = torch.zeros_like(slots)
    if decode and rows > 1:
        slots[rows//2:] = 1
        first[rows//2:] = rows//2
        slots[-1], first[-1] = -1, rows-1
    out, ref = empty(rows, d), empty(rows, d)
    state, conv, sr, cr = state0.clone(), conv0.clone(), state0.clone(), conv0.clone()
    replay = torch.zeros(kda_replay_layout(heads, 3*d, 64)[2]//4, device="cuda")
    rr = replay.clone()
    fs, ss = scratch(fused, cap), scratch(split, cap)
    def run(spec):
        state.copy_(state0); conv.copy_(conv0); sr.copy_(state0); cr.copy_(conv0)
        replay.zero_(); rr.zero_()
        args = (x, w, wf, cw, al, dt, norm, wo, cr, sr, slots, first, ref)
        if decode:
            fused.launch(*args, rr, fs, scalars=(rows, spec))
        else:
            fused.launch(*args, fs, scalars=(rows,))
        split.launch(x, q, k, v, q, fab, wf, cw, al, dt, norm, conv, state,
            slots, first, out, replay, ss, scalars=(rows, spec))
    for spec in ((0, 1) if decode else (0,)):
        run(spec)
        same(out, ref); same(state, sr); same(conv, cr)
        if spec:
            same(replay, rr)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(0)
    graph.replay(); graph.replay()
    same(out, ref); same(state, sr); same(conv, cr)
    assert torch.isfinite(out).all() and torch.count_nonzero(out) > 0


@pytest.mark.parametrize("rows", [1, 16, 64, 129])
@pytest.mark.parametrize("heads", [32, 64])
def test_flash_mla_stages_bitwise(rows, heads):
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_mla_producer_aot, compile_glmf_o_aot
    from b12x.integration.cuteafd.glmf_split import compile_glmf_stage_aot
    g = replace(GLM53_FLASH, heads=heads)
    cap, h, q = max(rows, 64), g.hidden, g.q_lora_rank
    fused = compile_glmf_mla_producer_aot(g, max_rows=cap)
    norm = compile_glmf_stage_aot(g, stage="rank_norm", max_rows=cap)
    absorb = compile_glmf_stage_aot(g, stage="absorb", max_rows=cap)
    values = compile_glmf_stage_aot(g, stage="values", max_rows=cap)
    output = compile_glmf_o_aot(g, max_rows=cap)
    x, wa, wb, uk = rand(rows, h), rand(q+512, h), rand(heads*256, q), rand(heads, 512, 256)
    qn, kn = rand(q, scale=1), rand(512, scale=1)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    slots[-1] = -1
    cache = torch.full((rows, 528), 173, device="cuda", dtype=torch.uint8)
    cref = cache.clone()
    packed = empty(rows, q+512)
    projection(q+512, h).launch(x, wa, packed, scalars=(rows,))
    qa, ka = packed[:, :q].contiguous(), packed[:, q:].contiguous()
    query, ref = empty(rows, heads, 512), empty(rows, heads, 512)
    qr, resid, qb = empty(rows, q), empty(rows, q), empty(rows, heads*256)
    bp = projection(heads*256, q)
    fs, ns = scratch(fused, cap), scratch(norm, cap)
    attn, uv, wo = rand(rows, heads, 512), rand(heads, 256, 512), rand(h, heads*256)
    vs, out, oref = empty(rows, heads*256), empty(rows, h), empty(rows, h)
    os, op = scratch(output, cap), projection(h, heads*256)
    def run():
        fused.launch(x, slots, wa, qn, kn, wb, uk, cref, ref, resid, fs, scalars=(rows,))
        norm.launch(qa, ka, qn, kn, slots, qr, cache, ns, scalars=(rows,))
        bp.launch(qr, wb, qb, scalars=(rows,))
        absorb.launch(qb, uk, query, scalars=(rows,))
        output.launch(attn, uv, wo, oref, os, scalars=(rows,))
        values.launch(attn, uv, vs, scalars=(rows,))
        op.launch(vs, wo, out, scalars=(rows,))
    run()
    same(qr, resid); same(cache, cref); same(query, ref); same(out, oref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay(); graph.replay()
    same(cache, cref); same(query, ref); same(out, oref)


@pytest.mark.parametrize("inter", [1024, 2048, 6144, 12288])
def test_flash_ffn_clamped_stage_bitwise(inter):
    from b12x.integration.cuteafd._common import GLM53_FLASH
    from b12x.integration.cuteafd.glmf import compile_glmf_ffn_aot
    from b12x.integration.cuteafd.glmf_split import compile_glmf_stage_aot
    rows, h = 16, GLM53_FLASH.hidden
    fused = compile_glmf_ffn_aot(inter=inter, max_rows=64)
    stage = compile_glmf_stage_aot(stage="swiglu", inter=inter, max_rows=64)
    # Large projections exercise the activation's +/-10 clamp.
    x, w, wd = rand(rows, h, scale=.4), rand(2*inter, h, scale=.4), rand(h, inter)
    gu, hidden, out, ref = empty(rows, 2*inter), empty(rows, inter), empty(rows, h), empty(rows, h)
    projection(2*inter, h).launch(x, w, gu, scalars=(rows,))
    gate, up = gu[:, :inter].contiguous(), gu[:, inter:].contiguous()
    fs, ss, down = scratch(fused, 64), scratch(stage, 64), projection(h, inter)
    fused.launch(x, w, wd, ref, fs, scalars=(rows,))
    stage.launch(gate, up, hidden, ss, scalars=(rows,))
    down.launch(hidden, wd, out, scalars=(rows,))
    same(out, ref)
