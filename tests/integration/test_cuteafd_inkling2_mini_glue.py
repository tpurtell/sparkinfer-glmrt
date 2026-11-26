"""Bit-exactness and graph-replay oracle for Inkling 2 Mini fused per-visit glue kernels.

Verifies every fused program in ``b12x.integration.cuteafd.inkling2_mini_glue``
against the exact stock multi-kernel sequence across rows 1..64 (plus 128 for
row-parallel ops), multi-segment packed batches, negative zero inputs, and
CUDA graph replay with mutated inputs.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

HIDDEN = 2304
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
KV_WIDTH = KV_HEADS * HEAD_DIM
REL_DIM = 16
REL_PADDED = 64
TAPS = 4
EPS = 1e-6


def _assert_bit_exact(got: torch.Tensor, want: torch.Tensor, label: str = ""):
    assert got.dtype == want.dtype and got.shape == want.shape, f"{label}: {got.dtype}/{got.shape} vs {want.dtype}/{want.shape}"
    view_dtype = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.int32: torch.int32,
        torch.uint8: torch.uint8,
    }[got.dtype]
    g = got.contiguous().view(view_dtype)
    w = want.contiguous().view(view_dtype)
    if not torch.equal(g, w):
        diff = (g != w).nonzero()
        first = tuple(diff[0].tolist())
        raise AssertionError(
            f"{label}: {diff.shape[0]} bit diffs; first at {first}: "
            f"got={got[first].item()} (0x{int(g[first].item()) & 0xffffffff:x}) vs "
            f"want={want[first].item()} (0x{int(w[first].item()) & 0xffffffff:x})"
        )


def _widen_zero(x_bf16: torch.Tensor) -> torch.Tensor:
    return x_bf16.float() + 0.0


def _round_bf16_f32(x_f32: torch.Tensor) -> torch.Tensor:
    return x_f32.bfloat16().float() + 0.0


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()


def test_fast_norm_bit_exact_all_rows():
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_norm_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_fast_norm_aot

    stock = compile_inkling2_norm_aot(width=HIDDEN, heads=1, weighted=True, eps=EPS, output_dtype=torch.bfloat16)
    fused = compile_inkling2_mini_fast_norm_aot(width=HIDDEN, eps=EPS)
    gen = torch.Generator(device="cuda").manual_seed(11)
    w = (0.5 + torch.randn((HIDDEN,), generator=gen, device="cuda")).bfloat16()
    for rows in list(range(1, 65)) + [128]:
        x = torch.randn((rows, HIDDEN), generator=gen, device="cuda") * 3.0
        out_s = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_f = torch.empty_like(out_s)
        stock.launch(x, w, out_s, scalars=(rows,))
        fused.launch(x, w, out_f, scalars=(rows,))
        _assert_bit_exact(out_f, out_s, f"fast_norm rows={rows}")


@pytest.mark.parametrize("queries", [1, 3])
def test_residual_norm_bit_exact_and_graph(queries):
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_norm_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_residual_norm_aot
    from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot

    stock_res = compile_inkling2_residual_aot(hidden=HIDDEN, queries=queries, max_candidates=5, addend=False, eps=EPS)
    stock_norm = compile_inkling2_norm_aot(width=HIDDEN, heads=1, weighted=True, eps=EPS, output_dtype=torch.bfloat16)
    fused = compile_inkling2_mini_residual_norm_aot(hidden=HIDDEN, queries=queries, max_candidates=5, eps=EPS)
    gen = torch.Generator(device="cuda").manual_seed(20 + queries)
    q = torch.randn((queries, HIDDEN), generator=gen, device="cuda") * 2.0
    w = (0.5 + torch.randn((queries, HIDDEN), generator=gen, device="cuda")).bfloat16()
    for rows in (1, 2, 3, 4, 7, 8, 16, 31, 64):
        cap = rows + 5
        arena = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
        arena[:, rows:] = float("nan")
        dummy = torch.zeros((rows, HIDDEN), device="cuda")
        mixed = torch.empty((queries, rows, HIDDEN), device="cuda")
        out_s = torch.empty((queries, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_f = torch.empty_like(out_s)
        for cand in range(1, 6):
            stock_res.launch(arena, dummy, q, mixed, scalars=(rows, cap, cand))
            for slot in range(queries):
                stock_norm.launch(mixed[slot], w[slot], out_s[slot], scalars=(rows,))
            fused.launch(arena, q, w, out_f, scalars=(rows, cap, cand))
            _assert_bit_exact(out_f, out_s, f"residual_norm q={queries} r={rows} c={cand}")
            assert torch.isnan(arena[:, rows:]).all()

    # Graph replay with mutated inputs.
    rows, cap, cand = 8, 12, 4
    arena = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
    dummy = torch.zeros((rows, HIDDEN), device="cuda")
    mixed = torch.empty((queries, rows, HIDDEN), device="cuda")
    out_s = torch.empty((queries, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    out_f = torch.empty_like(out_s)
    fused.launch(arena, q, w, out_f, scalars=(rows, cap, cand))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused.launch(arena, q, w, out_f, scalars=(rows, cap, cand))
    arena.mul_(-1.7).add_(0.25)
    g.replay()
    stock_res.launch(arena, dummy, q, mixed, scalars=(rows, cap, cand))
    for slot in range(queries):
        stock_norm.launch(mixed[slot], w[slot], out_s[slot], scalars=(rows,))
    _assert_bit_exact(out_f, out_s, f"residual_norm graph q={queries}")


@pytest.mark.parametrize("slots,bcast", [(1, False), (3, True), (3, False)])
def test_loop_blend_norm_fast_bit_exact(slots, bcast):
    from b12x.integration.cuteafd.inkling2_loop_controller import compile_inkling2_loop_blend_norm_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_loop_blend_norm_aot

    stock = compile_inkling2_loop_blend_norm_aot(
        hidden=HIDDEN, slots=slots, broadcast_dest=bcast, broadcast_weight=bcast, eps=EPS,
    )
    fused = compile_inkling2_mini_loop_blend_norm_aot(
        hidden=HIDDEN, slots=slots, broadcast_dest=bcast, broadcast_weight=bcast, eps=EPS,
    )
    gen = torch.Generator(device="cuda").manual_seed(30 + slots * 2 + int(bcast))
    d_slots = 1 if bcast else slots
    w_slots = 1 if bcast else slots
    w = (0.5 + torch.randn((w_slots, HIDDEN), generator=gen, device="cuda")).bfloat16()
    for rows in (1, 2, 3, 4, 8, 17, 64):
        dest = torch.randn((d_slots, rows, HIDDEN), generator=gen, device="cuda") * 2.0
        msg = torch.randn((slots, rows, HIDDEN), generator=gen, device="cuda") * 2.0
        fg = torch.randn((slots, rows, 2 * HIDDEN), generator=gen, device="cuda") * 3.0
        out_s = torch.empty((slots, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_f = torch.empty_like(out_s)
        stock.launch(dest, msg, fg, w, out_s, scalars=(rows,))
        fused.launch(dest, msg, fg, w, out_f, scalars=(rows,))
        _assert_bit_exact(out_f, out_s, f"loop_blend_norm s={slots} b={bcast} r={rows}")


def test_residual_pair_norm_bit_exact():
    from b12x.integration.cuteafd.inkling2_loop_controller import compile_inkling2_loop_pair_norm_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_residual_pair_norm_aot
    from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot

    stock_res = compile_inkling2_residual_aot(hidden=HIDDEN, queries=1, max_candidates=5, addend=False, eps=EPS)
    stock_pair = compile_inkling2_loop_pair_norm_aot(hidden=HIDDEN, slots=1, broadcast_dest=False, eps=EPS)
    fused = compile_inkling2_mini_residual_pair_norm_aot(hidden=HIDDEN, max_candidates=5, eps=EPS)
    gen = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randn((1, HIDDEN), generator=gen, device="cuda") * 2.0
    for rows in (1, 3, 8, 33, 64):
        cap = rows + 4
        arena = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
        msg = torch.randn((rows, HIDDEN), generator=gen, device="cuda") * 1.5
        dummy = torch.zeros((rows, HIDDEN), device="cuda")
        mixed_s = torch.empty((rows, HIDDEN), device="cuda")
        mixed_f = torch.empty_like(mixed_s)
        paired_s = torch.empty((rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        paired_f = torch.empty_like(paired_s)
        for cand in (2, 5):
            stock_res.launch(arena, dummy, q, mixed_s, scalars=(rows, cap, cand))
            stock_pair.launch(mixed_s, msg, paired_s, scalars=(rows,))
            fused.launch(arena, q, msg, mixed_f, paired_f, scalars=(rows, cap, cand))
            _assert_bit_exact(mixed_f, mixed_s, f"res_pair mixed r={rows} c={cand}")
            _assert_bit_exact(paired_f, paired_s, f"res_pair paired r={rows} c={cand}")


@pytest.mark.parametrize("bcast", [True, False])
def test_attn_cross_pair_norm_bit_exact(bcast):
    from b12x.integration.cuteafd.inkling2_loop_controller import compile_inkling2_loop_pair_norm_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_attn_cross_pair_norm_aot
    from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot

    dq = 1 if bcast else 3
    stock_dest = compile_inkling2_residual_aot(hidden=HIDDEN, queries=dq, max_candidates=5, addend=False, eps=EPS)
    stock_cross = compile_inkling2_residual_aot(hidden=HIDDEN, queries=4, max_candidates=2, addend=False, eps=EPS)
    stock_pair = compile_inkling2_loop_pair_norm_aot(hidden=HIDDEN, slots=3, broadcast_dest=bcast, eps=EPS)
    fused = compile_inkling2_mini_attn_cross_pair_norm_aot(hidden=HIDDEN, broadcast_dest=bcast, max_candidates=5, eps=EPS)
    gen = torch.Generator(device="cuda").manual_seed(50 + int(bcast))
    aq = torch.randn((dq, HIDDEN), generator=gen, device="cuda") * 2.0
    cq = torch.randn((4, HIDDEN), generator=gen, device="cuda") * 2.0
    for rows in (1, 3, 8, 25, 64):
        cap = rows + 3
        arena = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
        prev = torch.randn((2, cap, HIDDEN), generator=gen, device="cuda")
        dummy = torch.zeros((rows, HIDDEN), device="cuda")
        dest_s = torch.empty((dq, rows, HIDDEN), device="cuda")
        dest_f = torch.empty_like(dest_s)
        msg_s = torch.empty((4, rows, HIDDEN), device="cuda")
        msg_f = torch.empty_like(msg_s)
        paired_s = torch.empty((3, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        paired_f = torch.empty_like(paired_s)
        for cand in (2, 4):
            stock_dest.launch(arena, dummy, aq, dest_s, scalars=(rows, cap, cand))
            stock_cross.launch(prev, dummy, cq, msg_s, scalars=(rows, cap, 2))
            stock_pair.launch(dest_s, msg_s[:3].contiguous(), paired_s, scalars=(rows,))
            fused.launch(arena, aq, prev, cq, dest_f, msg_f, paired_f, scalars=(rows, cap, cand))
            _assert_bit_exact(dest_f, dest_s, f"attn_cross dest b={bcast} r={rows} c={cand}")
            _assert_bit_exact(msg_f, msg_s, f"attn_cross msg b={bcast} r={rows} c={cand}")
            _assert_bit_exact(paired_f, paired_s, f"attn_cross paired b={bcast} r={rows} c={cand}")


def _make_segments(rows: int, num_slots: int = 8):
    """Build multi-segment packed metadata (1..4 segments) with exclusive conv_slots."""
    if rows == 1:
        lens = [1]
    elif rows <= 4:
        lens = [rows]
    elif rows <= 10:
        lens = [3, rows - 3]
    else:
        a = rows // 3
        b = rows // 3
        lens = [a, b, rows - a - b]
    first, cslots = [], []
    cur = 0
    for s_idx, length in enumerate(lens):
        first.extend([cur] * length)
        cslots.extend([(s_idx * 2 + 1) % num_slots] * length)
        cur += length
    return (
        torch.tensor(first, dtype=torch.int32, device="cuda"),
        torch.tensor(cslots, dtype=torch.int64, device="cuda"),
    )


@pytest.mark.parametrize("local,fp8", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("replay", [True, False])
def test_attn_prep_bit_exact_all_rows_and_graph(local, fp8, replay):
    from b12x.integration.cuteafd.inkling2_clip import compile_inkling2_clip_aot
    from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot
    from b12x.integration.cuteafd.inkling2_convolution_strided import compile_inkling2_convolution_strided_aot
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot, compile_inkling2_norm_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_attn_prep_aot

    dynamic = not local
    conv_mod = compile_inkling2_convolution_replay_aot if replay else compile_inkling2_convolution_strided_aot
    s_conv = conv_mod(width=KV_WIDTH, taps=TAPS, heads=KV_HEADS if dynamic else 1, dynamic=dynamic)
    s_clip_q = compile_inkling2_clip_aot(
        heads=HEADS, dim=HEAD_DIM, threshold=HEAD_DIM**0.5 * 2, scaled=True,
        log_floor=0 if local else 8192, log_alpha=0.0 if local else 0.1, dtype=torch.bfloat16,
    )
    s_clip_r = compile_inkling2_clip_aot(
        heads=HEADS, dim=REL_DIM, threshold=8.0, scaled=True,
        log_floor=0 if local else 8192, log_alpha=0.0 if local else 0.1, dtype=torch.bfloat16,
    )
    s_clip_k = compile_inkling2_clip_aot(
        heads=KV_HEADS, dim=HEAD_DIM, threshold=HEAD_DIM**0.5 * 2, scaled=False,
        log_floor=0, log_alpha=0.0, dtype=torch.bfloat16,
    )
    s_norm_v = compile_inkling2_norm_aot(width=HEAD_DIM, heads=KV_HEADS, weighted=False, eps=EPS, output_dtype=torch.bfloat16)
    s_pack = compile_inkling2_kv_pack_aot(
        heads=KV_HEADS, dim=HEAD_DIM, input_dtype=torch.bfloat16, kv_format="fp8" if fp8 else "bf16",
    )
    fused = compile_inkling2_mini_attn_prep_aot(local=local, fp8=fp8, replay=replay, max_rows=64)

    gen = torch.Generator(device="cuda").manual_seed(100 + int(local) * 4 + int(fp8) * 2 + int(replay))
    q_scale = torch.randn((HEADS, HEAD_DIM), generator=gen, device="cuda").bfloat16()
    r_scale = torch.randn((HEADS, REL_DIM), generator=gen, device="cuda").bfloat16()
    w_shape = (KV_WIDTH, TAPS) if local else (TAPS, KV_WIDTH)
    kw = torch.randn(w_shape, generator=gen, device="cuda").bfloat16()
    vw = torch.randn(w_shape, generator=gen, device="cuda").bfloat16()

    # Include edge row counts plus full 1..16 sweep and 32, 63, 64.
    for rows in list(range(1, 17)) + [21, 32, 47, 63, 64]:
        q = torch.randn((rows, HEADS, HEAD_DIM), generator=gen, device="cuda").bfloat16()
        r = torch.randn((rows, HEADS, REL_DIM), generator=gen, device="cuda").bfloat16()
        k_bf16 = torch.randn((rows, KV_WIDTH), generator=gen, device="cuda").bfloat16()
        v_bf16 = torch.randn((rows, KV_WIDTH), generator=gen, device="cuda").bfloat16()
        # Inject exact -0.0 BF16 to verify sign-of-zero equivalence with widen_zero!
        k_bf16[0, 0] = -0.0
        v_bf16[0, 0] = -0.0
        k_dyn = torch.randn((rows, KV_HEADS, TAPS), generator=gen, device="cuda")
        v_dyn = torch.randn((rows, KV_HEADS, TAPS), generator=gen, device="cuda")
        positions = torch.randint(0, 20000, (rows,), generator=gen, dtype=torch.int64, device="cuda")
        first, cslots = _make_segments(rows)
        # Include a negative cache_slot to test skip behavior.
        cache_slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        if rows >= 3:
            cache_slots[1] = -1

        lag_stride = 6656
        slot_stride = 3 * lag_stride
        hist_k_s = torch.randn((8, 3, lag_stride), generator=gen, device="cuda")
        hist_v_s = torch.randn((8, 3, lag_stride), generator=gen, device="cuda")
        hist_k_f = hist_k_s.clone()
        hist_v_f = hist_v_s.clone()

        rep_stride = 6656
        rep_k_s = torch.full((64, rep_stride), -7.0, device="cuda")
        rep_v_s = torch.full((64, rep_stride), -7.0, device="cuda")
        rep_k_f = rep_k_s.clone()
        rep_v_f = rep_v_s.clone()

        if fp8:
            cache_s = torch.randint(0, 255, (rows + 4, 2112), generator=gen, dtype=torch.uint8, device="cuda")
        else:
            cache_s = torch.randn((rows + 4, 2 * KV_WIDTH), generator=gen, device="cuda").bfloat16()
        cache_f = cache_s.clone()

        # Stock chain (15 kernels):
        k_f32 = _widen_zero(k_bf16)
        v_f32 = _widen_zero(v_bf16)
        conv_k = torch.empty((rows, KV_WIDTH), device="cuda")
        conv_v = torch.empty((rows, KV_WIDTH), device="cuda")
        if replay:
            s_conv.launch(k_f32, kw, k_dyn, hist_k_s, first, cslots, conv_k, rep_k_s,
                          scalars=(rows, slot_stride, lag_stride, rep_stride))
            s_conv.launch(v_f32, vw, v_dyn, hist_v_s, first, cslots, conv_v, rep_v_s,
                          scalars=(rows, slot_stride, lag_stride, rep_stride))
        else:
            s_conv.launch(k_f32, kw, k_dyn, hist_k_s, first, cslots, conv_k,
                          scalars=(rows, slot_stride, lag_stride))
            s_conv.launch(v_f32, vw, v_dyn, hist_v_s, first, cslots, conv_v,
                          scalars=(rows, slot_stride, lag_stride))
        k_narrow = conv_k.bfloat16()
        v_narrow = conv_v.bfloat16()
        q_out_s = torch.empty_like(q)
        r_clip_s = torch.empty_like(r)
        k_clip_s = torch.empty_like(k_bf16)
        v_norm_s = torch.empty_like(v_bf16)
        s_clip_q.launch(q, q_scale, positions, q_out_s, scalars=(rows,))
        s_clip_r.launch(r, r_scale, positions, r_clip_s, scalars=(rows,))
        s_clip_k.launch(k_narrow, k_narrow, positions, k_clip_s, scalars=(rows,))
        v_widen = _widen_zero(v_narrow)
        s_norm_v.launch(v_widen, v_widen, v_norm_s, scalars=(rows,))
        rel_in_s = torch.zeros((rows, HEADS, REL_PADDED), dtype=torch.bfloat16, device="cuda")
        rel_in_s[:, :, :REL_DIM] = r_clip_s
        s_pack.launch(k_clip_s, v_norm_s, cache_slots, cache_s, scalars=(rows,))

        # Fused (1 kernel):
        q_out_f = torch.empty_like(q)
        rel_in_f = torch.full((rows, HEADS, REL_PADDED), 3.5, dtype=torch.bfloat16, device="cuda")
        fused.launch(
            q, q_scale, r, r_scale, positions, q_out_f, rel_in_f,
            k_bf16, kw, k_dyn, hist_k_f,
            v_bf16, vw, v_dyn, hist_v_f,
            first, cslots, cache_slots, cache_f,
            rep_k_f, rep_v_f,
            scalars=(rows, slot_stride, lag_stride, rep_stride),
        )

        tag = f"attn_prep local={local} fp8={fp8} replay={replay} r={rows}"
        _assert_bit_exact(q_out_f, q_out_s, f"{tag} q_out")
        _assert_bit_exact(rel_in_f, rel_in_s, f"{tag} rel_in")
        _assert_bit_exact(cache_f, cache_s, f"{tag} cache")
        _assert_bit_exact(hist_k_f, hist_k_s, f"{tag} hist_k")
        _assert_bit_exact(hist_v_f, hist_v_s, f"{tag} hist_v")
        if replay:
            _assert_bit_exact(rep_k_f, rep_k_s, f"{tag} rep_k")
            _assert_bit_exact(rep_v_f, rep_v_s, f"{tag} rep_v")


@pytest.mark.parametrize("starts_block", [True, False])
@pytest.mark.parametrize("replay", [True, False])
def test_post_attn_conv_bit_exact(starts_block, replay):
    from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot
    from b12x.integration.cuteafd.inkling2_convolution_strided import compile_inkling2_convolution_strided_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_post_attn_conv_aot

    conv_mod = compile_inkling2_convolution_replay_aot if replay else compile_inkling2_convolution_strided_aot
    s_conv = conv_mod(width=HIDDEN, taps=TAPS, heads=1, dynamic=False)
    fused = compile_inkling2_mini_post_attn_conv_aot(starts_block=starts_block, replay=replay, max_rows=64)
    gen = torch.Generator(device="cuda").manual_seed(200 + int(starts_block) * 2 + int(replay))
    w = torch.randn((HIDDEN, TAPS), generator=gen, device="cuda").bfloat16()

    for rows in (1, 2, 3, 4, 8, 17, 64):
        cap = rows + 4
        mlp_idx = 2 if starts_block else 1
        x_bf16 = torch.randn((rows, HIDDEN), generator=gen, device="cuda").bfloat16()
        x_bf16[0, 3] = -0.0
        partial = torch.randn((rows, HIDDEN), generator=gen, device="cuda")
        arena_s = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
        arena_f = arena_s.clone()
        first, cslots = _make_segments(rows)
        lag_stride = 6656
        slot_stride = 3 * lag_stride
        hist_s = torch.randn((8, 3, lag_stride), generator=gen, device="cuda")
        hist_f = hist_s.clone()
        rep_stride = 6656
        rep_s = torch.full((64, rep_stride), -3.0, device="cuda")
        rep_f = rep_s.clone()

        x_f32 = _widen_zero(x_bf16)
        conv_out = torch.empty((rows, HIDDEN), device="cuda")
        if replay:
            s_conv.launch(x_f32, w, x_f32, hist_s, first, cslots, conv_out, rep_s,
                          scalars=(rows, slot_stride, lag_stride, rep_stride))
        else:
            s_conv.launch(x_f32, w, x_f32, hist_s, first, cslots, conv_out,
                          scalars=(rows, slot_stride, lag_stride))
        attn_s = _round_bf16_f32(conv_out)
        if starts_block:
            arena_s[mlp_idx - 1, :rows] = partial
            arena_s[mlp_idx, :rows] = attn_s
            out_s = attn_s
        else:
            out_s = partial + attn_s
            arena_s[mlp_idx, :rows] = out_s

        out_f = torch.empty((rows, HIDDEN), device="cuda")
        fused.launch(
            x_bf16, w, hist_f, first, cslots, partial, arena_f, out_f, rep_f,
            scalars=(rows, cap, mlp_idx, slot_stride, lag_stride, rep_stride),
        )
        tag = f"post_attn start={starts_block} replay={replay} r={rows}"
        _assert_bit_exact(out_f, out_s, f"{tag} out")
        _assert_bit_exact(arena_f, arena_s, f"{tag} arena")
        _assert_bit_exact(hist_f, hist_s, f"{tag} hist")
        if replay:
            _assert_bit_exact(rep_f, rep_s, f"{tag} replay")


@pytest.mark.parametrize("replay", [True, False])
def test_post_mlp_conv_residual_bit_exact(replay):
    from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot
    from b12x.integration.cuteafd.inkling2_convolution_strided import compile_inkling2_convolution_strided_aot
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_post_mlp_conv_residual_aot

    conv_mod = compile_inkling2_convolution_replay_aot if replay else compile_inkling2_convolution_strided_aot
    s_conv = conv_mod(width=HIDDEN, taps=TAPS, heads=1, dynamic=False)
    fused = compile_inkling2_mini_post_mlp_conv_residual_aot(replay=replay, max_rows=64)
    gen = torch.Generator(device="cuda").manual_seed(300 + int(replay))
    w = torch.randn((HIDDEN, TAPS), generator=gen, device="cuda").bfloat16()

    for rows in (1, 2, 3, 4, 8, 17, 64):
        cap = rows + 4
        mlp_idx = 3
        x_bf16 = torch.randn((rows, HIDDEN), generator=gen, device="cuda").bfloat16()
        x_bf16[0, 5] = -0.0
        base = torch.randn((rows, HIDDEN), generator=gen, device="cuda")
        arena_s = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
        arena_f = arena_s.clone()
        first, cslots = _make_segments(rows)
        lag_stride = 6656
        slot_stride = 3 * lag_stride
        hist_s = torch.randn((8, 3, lag_stride), generator=gen, device="cuda")
        hist_f = hist_s.clone()
        rep_stride = 6656
        rep_s = torch.full((64, rep_stride), -3.0, device="cuda")
        rep_f = rep_s.clone()

        x_f32 = _widen_zero(x_bf16)
        conv_out = torch.empty((rows, HIDDEN), device="cuda")
        if replay:
            s_conv.launch(x_f32, w, x_f32, hist_s, first, cslots, conv_out, rep_s,
                          scalars=(rows, slot_stride, lag_stride, rep_stride))
        else:
            s_conv.launch(x_f32, w, x_f32, hist_s, first, cslots, conv_out,
                          scalars=(rows, slot_stride, lag_stride))
        mlp_s = _round_bf16_f32(conv_out)
        partial_s = _round_bf16_f32(base + mlp_s)
        arena_s[mlp_idx, :rows] = partial_s

        partial_f = torch.empty((rows, HIDDEN), device="cuda")
        fused.launch(
            x_bf16, w, hist_f, first, cslots, base, partial_f, arena_f, rep_f,
            scalars=(rows, cap, mlp_idx, slot_stride, lag_stride, rep_stride),
        )
        tag = f"post_mlp replay={replay} r={rows}"
        _assert_bit_exact(partial_f, partial_s, f"{tag} partial")
        _assert_bit_exact(arena_f, arena_s, f"{tag} arena")
        _assert_bit_exact(hist_f, hist_s, f"{tag} hist")
        if replay:
            _assert_bit_exact(rep_f, rep_s, f"{tag} replay")


def test_fast_router_bit_exact_including_ties():
    from b12x.integration.cuteafd.inkling2_mini_glue import compile_inkling2_mini_fast_router_aot
    from b12x.integration.cuteafd.inkling2_router import compile_inkling2_router_aot

    stock = compile_inkling2_router_aot(experts=128, topk=8, route_scale=8.0)
    fused = compile_inkling2_mini_fast_router_aot(experts=128, topk=8, route_scale=8.0)
    gen = torch.Generator(device="cuda").manual_seed(401)
    bias = torch.randn((128,), generator=gen, device="cuda") * 0.1
    for rows in (1, 2, 3, 4, 8, 33, 64):
        logits = torch.randn((rows, 128), generator=gen, device="cuda") * 2.0
        # Force ties across lanes on row 0.
        logits[0, 5] = logits[0, 37] = logits[0, 69] = 10.0
        bias_t = bias.clone()
        bias_t[5] = bias_t[37] = bias_t[69] = 0.0
        ids_s = torch.empty((rows, 8), dtype=torch.int32, device="cuda")
        w_s = torch.empty((rows, 8), dtype=torch.float32, device="cuda")
        ids_f = torch.empty_like(ids_s)
        w_f = torch.empty_like(w_s)
        stock.launch(logits, bias_t, ids_s, w_s, scalars=(rows,))
        fused.launch(logits, bias_t, ids_f, w_f, scalars=(rows,))
        _assert_bit_exact(ids_f, ids_s, f"router ids r={rows}")
        _assert_bit_exact(w_f, w_s, f"router weights r={rows}")
