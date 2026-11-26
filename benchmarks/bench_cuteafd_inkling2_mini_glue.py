"""Warmed CUDA-graph benchmark, row curve, and prefix/route invariance check for
Inkling 2 Mini fused per-visit glue kernels.

Supports ``--compare-routes-exact`` (bit-exact comparison of fused vs stock
chains across every row count) and ``--check-prefix-rows`` (row invariance:
every row in a multi-row batch is bit-exact against single-row M=1 execution
for row-parallel ops and independent single-row segments for conv ops).
"""
from __future__ import annotations

import argparse
import statistics
import torch

from b12x.integration.cuteafd.inkling2_clip import compile_inkling2_clip_aot
from b12x.integration.cuteafd.inkling2_convolution_replay import compile_inkling2_convolution_replay_aot
from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot, compile_inkling2_norm_aot
from b12x.integration.cuteafd.inkling2_loop_controller import (
    compile_inkling2_loop_blend_norm_aot,
    compile_inkling2_loop_pair_norm_aot,
)
from b12x.integration.cuteafd.inkling2_mini_glue import (
    compile_inkling2_mini_attn_cross_pair_norm_aot,
    compile_inkling2_mini_attn_prep_aot,
    compile_inkling2_mini_fast_norm_aot,
    compile_inkling2_mini_fast_router_aot,
    compile_inkling2_mini_loop_blend_norm_aot,
    compile_inkling2_mini_post_attn_conv_aot,
    compile_inkling2_mini_post_mlp_conv_residual_aot,
    compile_inkling2_mini_residual_norm_aot,
    compile_inkling2_mini_residual_pair_norm_aot,
)
from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot
from b12x.integration.cuteafd.inkling2_router import compile_inkling2_router_aot

HIDDEN = 2304
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
KV_WIDTH = 1024
REL_DIM = 16
REL_PADDED = 64
TAPS = 4
EPS = 1e-6


def _bit_eq(a: torch.Tensor, b: torch.Tensor, label: str):
    vd = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.int32: torch.int32, torch.uint8: torch.uint8}[a.dtype]
    if not torch.equal(a.contiguous().view(vd), b.contiguous().view(vd)):
        raise AssertionError(f"bit mismatch in {label}")


def _time_graph_us(fn, *, reps: int = 80, inner: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0 / inner)
    return statistics.median(times)


def _check_prefix_and_exact(progs, rows_sweep: list[int], compare_exact: bool, check_prefix: bool):
    gen = torch.Generator(device="cuda").manual_seed(777)
    max_r = max(rows_sweep)
    cap = max_r + 4
    arena = torch.randn((5, cap, HIDDEN), generator=gen, device="cuda")
    prev = torch.randn((2, cap, HIDDEN), generator=gen, device="cuda")
    q1 = torch.randn((1, HIDDEN), generator=gen, device="cuda")
    q3 = torch.randn((3, HIDDEN), generator=gen, device="cuda")
    q4 = torch.randn((4, HIDDEN), generator=gen, device="cuda")
    w1 = torch.randn((1, HIDDEN), generator=gen, device="cuda").bfloat16()
    w3 = torch.randn((3, HIDDEN), generator=gen, device="cuda").bfloat16()
    fg1 = torch.randn((1, max_r, 2 * HIDDEN), generator=gen, device="cuda")
    fg3 = torch.randn((3, max_r, 2 * HIDDEN), generator=gen, device="cuda")
    partial = torch.randn((max_r, HIDDEN), generator=gen, device="cuda")
    base_h = torch.randn((max_r, HIDDEN), generator=gen, device="cuda")
    msg_all = torch.randn((4, max_r, HIDDEN), generator=gen, device="cuda")
    r_logits = torch.randn((max_r, 128), generator=gen, device="cuda")
    r_bias = torch.randn((128,), generator=gen, device="cuda")

    # Single-row M=1 reference for row 0 (prefix invariance).
    def run_row_parallel(r: int):
        d1 = torch.zeros((r, HIDDEN), device="cuda")
        out_n = torch.empty((r, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_rn1 = torch.empty((1, r, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_rn3 = torch.empty((3, r, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_bl1 = torch.empty((1, r, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out_bl3 = torch.empty((3, r, HIDDEN), dtype=torch.bfloat16, device="cuda")
        mix_rp = torch.empty((r, HIDDEN), device="cuda")
        pair_rp = torch.empty((r, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        dest_cp = torch.empty((1, r, HIDDEN), device="cuda")
        msg_cp = torch.empty((4, r, HIDDEN), device="cuda")
        pair_cp = torch.empty((3, r, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        ids = torch.empty((r, 8), dtype=torch.int32, device="cuda")
        rw = torch.empty((r, 8), device="cuda")

        progs["f_norm"].launch(partial[:r], w1, out_n, scalars=(r,))
        progs["f_resnorm1"].launch(arena, q1, w1, out_rn1, scalars=(r, cap, 4))
        progs["f_resnorm3"].launch(arena, q3, w3, out_rn3, scalars=(r, cap, 4))
        progs["f_blend1"].launch(partial[:r], msg_all[3, :r], fg1[:, :r].contiguous(), w1, out_bl1, scalars=(r,))
        progs["f_blend3b"].launch(partial[:r], msg_all[:3, :r].contiguous(), fg3[:, :r].contiguous(), w1, out_bl3, scalars=(r,))
        progs["f_respair1"].launch(arena, q1, msg_all[3, :r], mix_rp, pair_rp, scalars=(r, cap, 4))
        progs["f_cross3b"].launch(arena, q1, prev, q4, dest_cp, msg_cp, pair_cp, scalars=(r, cap, 4))
        progs["f_router"].launch(r_logits[:r], r_bias, ids, rw, scalars=(r,))

        if compare_exact:
            s_out_n = torch.empty_like(out_n)
            progs["s_norm"].launch(partial[:r], w1, s_out_n, scalars=(r,))
            _bit_eq(out_n, s_out_n, f"norm r={r}")
            s_m1 = torch.empty((1, r, HIDDEN), device="cuda")
            s_rn1 = torch.empty_like(out_rn1)
            progs["s_res1"].launch(arena, d1, q1, s_m1, scalars=(r, cap, 4))
            progs["s_norm"].launch(s_m1[0], w1[0], s_rn1[0], scalars=(r,))
            _bit_eq(out_rn1, s_rn1, f"resnorm1 r={r}")
            s_m3 = torch.empty((3, r, HIDDEN), device="cuda")
            s_rn3 = torch.empty_like(out_rn3)
            progs["s_res3"].launch(arena, d1, q3, s_m3, scalars=(r, cap, 4))
            for sl in range(3):
                progs["s_norm"].launch(s_m3[sl], w3[sl], s_rn3[sl], scalars=(r,))
            _bit_eq(out_rn3, s_rn3, f"resnorm3 r={r}")
            s_bl1 = torch.empty_like(out_bl1)
            progs["s_blend1"].launch(partial[:r], msg_all[3, :r], fg1[:, :r].contiguous(), w1, s_bl1, scalars=(r,))
            _bit_eq(out_bl1, s_bl1, f"blend1 r={r}")
            s_bl3 = torch.empty_like(out_bl3)
            progs["s_blend3b"].launch(partial[:r], msg_all[:3, :r].contiguous(), fg3[:, :r].contiguous(), w1, s_bl3, scalars=(r,))
            _bit_eq(out_bl3, s_bl3, f"blend3b r={r}")
            s_ids = torch.empty_like(ids)
            s_rw = torch.empty_like(rw)
            progs["s_router"].launch(r_logits[:r], r_bias, s_ids, s_rw, scalars=(r,))
            _bit_eq(ids, s_ids, f"router_ids r={r}")
            _bit_eq(rw, s_rw, f"router_w r={r}")
        return out_n, out_rn1, out_rn3, out_bl1, out_bl3, mix_rp, pair_rp, dest_cp, msg_cp, pair_cp, ids, rw

    ref1 = run_row_parallel(1)
    for r in rows_sweep:
        got = run_row_parallel(r)
        if check_prefix:
            for idx, (g, ref) in enumerate(zip(got, ref1)):
                if g.ndim == 3:
                    _bit_eq(g[:, :1], ref[:, :1], f"prefix op={idx} r={r}")
                else:
                    _bit_eq(g[:1], ref[:1], f"prefix op={idx} r={r}")

    # Prefix invariance for decode conv ops (independent 1-row segments in rows 1..64).
    q = torch.randn((64, HEADS, HEAD_DIM), generator=gen, device="cuda").bfloat16()
    qs = torch.randn((HEADS, HEAD_DIM), generator=gen, device="cuda").bfloat16()
    r_in = torch.randn((64, HEADS, REL_DIM), generator=gen, device="cuda").bfloat16()
    rs = torch.randn((HEADS, REL_DIM), generator=gen, device="cuda").bfloat16()
    pos = torch.arange(64, dtype=torch.int64, device="cuda")
    k_bf16 = torch.randn((64, KV_WIDTH), generator=gen, device="cuda").bfloat16()
    v_bf16 = torch.randn((64, KV_WIDTH), generator=gen, device="cuda").bfloat16()
    kw = torch.randn((KV_WIDTH, TAPS), generator=gen, device="cuda").bfloat16()
    vw = torch.randn((KV_WIDTH, TAPS), generator=gen, device="cuda").bfloat16()
    x_h = torch.randn((64, HIDDEN), generator=gen, device="cuda").bfloat16()
    w_h = torch.randn((HIDDEN, TAPS), generator=gen, device="cuda").bfloat16()
    lag_stride = 6656
    slot_stride = 3 * lag_stride
    hist_init = torch.randn((64, 3, lag_stride), generator=gen, device="cuda")

    def run_conv_ops(rows: int):
        first = torch.arange(rows, dtype=torch.int32, device="cuda")
        cslots = torch.arange(rows, dtype=torch.int64, device="cuda")
        kslots = torch.arange(rows, dtype=torch.int64, device="cuda")
        hk, hv, ha, hm = hist_init.clone(), hist_init.clone(), hist_init.clone(), hist_init.clone()
        rk = torch.zeros((64, lag_stride), device="cuda")
        rv = torch.zeros((64, lag_stride), device="cuda")
        ra = torch.zeros((64, lag_stride), device="cuda")
        rm = torch.zeros((64, lag_stride), device="cuda")
        q_o = torch.empty((rows, HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        rel_o = torch.empty((rows, HEADS, REL_PADDED), dtype=torch.bfloat16, device="cuda")
        kv_c = torch.zeros((64, 2 * KV_WIDTH), dtype=torch.bfloat16, device="cuda")
        ar_a = arena.clone()
        out_a = torch.empty((rows, HIDDEN), device="cuda")
        ar_m = arena.clone()
        out_m = torch.empty((rows, HIDDEN), device="cuda")
        progs["f_attn_prep"].launch(
            q[:rows], qs, r_in[:rows], rs, pos[:rows], q_o, rel_o,
            k_bf16[:rows], kw, partial[:rows], hk, v_bf16[:rows], vw, partial[:rows], hv,
            first, cslots, kslots, kv_c, rk, rv,
            scalars=(rows, slot_stride, lag_stride, lag_stride),
        )
        progs["f_post_attn"].launch(
            x_h[:rows], w_h, ha, first, cslots, partial[:rows], ar_a, out_a, ra,
            scalars=(rows, cap, 1, slot_stride, lag_stride, lag_stride),
        )
        progs["f_post_mlp"].launch(
            x_h[:rows], w_h, hm, first, cslots, base_h[:rows], out_m, ar_m, rm,
            scalars=(rows, cap, 1, slot_stride, lag_stride, lag_stride),
        )
        return q_o[:1], rel_o[:1], kv_c[:1], hk[:1], hv[:1], out_a[:1], ar_a[1, :1], ha[:1], out_m[:1], ar_m[1, :1], hm[:1]

    if check_prefix:
        cref = run_conv_ops(1)
        for r in [2, 3, 4, 8, 16, 32, 64]:
            cgot = run_conv_ops(r)
            for idx, (g, ref) in enumerate(zip(cgot, cref)):
                _bit_eq(g, ref, f"conv_prefix op={idx} r={r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,4,8,16,32,64")
    ap.add_argument("--wide-rows", default="2048,2752,4096")
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--compare-routes-exact", action="store_true")
    ap.add_argument("--check-prefix-rows", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    row_list = [int(x) for x in args.rows.split(",") if x.strip()]
    wide_list = [int(x) for x in args.wide_rows.split(",") if x.strip()]

    progs = dict(
        s_norm=compile_inkling2_norm_aot(width=HIDDEN, heads=1, weighted=True, eps=EPS, output_dtype=torch.bfloat16),
        f_norm=compile_inkling2_mini_fast_norm_aot(width=HIDDEN, eps=EPS),
        s_res1=compile_inkling2_residual_aot(hidden=HIDDEN, queries=1, max_candidates=5, addend=False, eps=EPS),
        s_res3=compile_inkling2_residual_aot(hidden=HIDDEN, queries=3, max_candidates=5, addend=False, eps=EPS),
        s_res4=compile_inkling2_residual_aot(hidden=HIDDEN, queries=4, max_candidates=2, addend=False, eps=EPS),
        f_resnorm1=compile_inkling2_mini_residual_norm_aot(hidden=HIDDEN, queries=1, max_candidates=5, eps=EPS),
        f_resnorm3=compile_inkling2_mini_residual_norm_aot(hidden=HIDDEN, queries=3, max_candidates=5, eps=EPS),
        s_pair1=compile_inkling2_loop_pair_norm_aot(hidden=HIDDEN, slots=1, broadcast_dest=False, eps=EPS),
        s_pair3b=compile_inkling2_loop_pair_norm_aot(hidden=HIDDEN, slots=3, broadcast_dest=True, eps=EPS),
        f_respair1=compile_inkling2_mini_residual_pair_norm_aot(hidden=HIDDEN, max_candidates=5, eps=EPS),
        f_cross3b=compile_inkling2_mini_attn_cross_pair_norm_aot(hidden=HIDDEN, broadcast_dest=True, max_candidates=5, eps=EPS),
        s_blend1=compile_inkling2_loop_blend_norm_aot(hidden=HIDDEN, slots=1, broadcast_dest=False, broadcast_weight=False, eps=EPS),
        s_blend3b=compile_inkling2_loop_blend_norm_aot(hidden=HIDDEN, slots=3, broadcast_dest=True, broadcast_weight=True, eps=EPS),
        f_blend1=compile_inkling2_mini_loop_blend_norm_aot(hidden=HIDDEN, slots=1, broadcast_dest=False, broadcast_weight=False, eps=EPS),
        f_blend3b=compile_inkling2_mini_loop_blend_norm_aot(hidden=HIDDEN, slots=3, broadcast_dest=True, broadcast_weight=True, eps=EPS),
        s_conv_kv=compile_inkling2_convolution_replay_aot(width=KV_WIDTH, taps=TAPS, heads=1, dynamic=False),
        s_conv_h=compile_inkling2_convolution_replay_aot(width=HIDDEN, taps=TAPS, heads=1, dynamic=False),
        s_clip_q=compile_inkling2_clip_aot(heads=HEADS, dim=HEAD_DIM, threshold=HEAD_DIM**0.5 * 2, scaled=True, dtype=torch.bfloat16),
        s_clip_r=compile_inkling2_clip_aot(heads=HEADS, dim=REL_DIM, threshold=8.0, scaled=True, dtype=torch.bfloat16),
        s_clip_k=compile_inkling2_clip_aot(heads=KV_HEADS, dim=HEAD_DIM, threshold=HEAD_DIM**0.5 * 2, scaled=False, dtype=torch.bfloat16),
        s_norm_v=compile_inkling2_norm_aot(width=HEAD_DIM, heads=KV_HEADS, weighted=False, eps=EPS, output_dtype=torch.bfloat16),
        s_pack=compile_inkling2_kv_pack_aot(heads=KV_HEADS, dim=HEAD_DIM, input_dtype=torch.bfloat16, kv_format="bf16"),
        f_attn_prep=compile_inkling2_mini_attn_prep_aot(local=True, fp8=False, replay=True, max_rows=64),
        f_post_attn=compile_inkling2_mini_post_attn_conv_aot(starts_block=False, replay=True, max_rows=64),
        f_post_mlp=compile_inkling2_mini_post_mlp_conv_residual_aot(replay=True, max_rows=64),
        s_router=compile_inkling2_router_aot(experts=128, topk=8, route_scale=8.0),
        f_router=compile_inkling2_mini_fast_router_aot(experts=128, topk=8, route_scale=8.0),
    )

    print(f"Device: {torch.cuda.get_device_name(0)}")
    if args.compare_routes_exact or args.check_prefix_rows:
        sweep = sorted(set(row_list + [1, 2, 3, 4, 8, 16, 32, 64]))
        _check_prefix_and_exact(progs, sweep, args.compare_routes_exact, args.check_prefix_rows)
        print(f"PASS exact/prefix invariance across rows={sweep}")
    if args.check_only:
        return

    for rows in row_list:
        cap = 64
        arena = torch.randn((5, cap, HIDDEN), device="cuda")
        prev = torch.randn((2, cap, HIDDEN), device="cuda")
        q1 = torch.randn((1, HIDDEN), device="cuda")
        q3 = torch.randn((3, HIDDEN), device="cuda")
        q4 = torch.randn((4, HIDDEN), device="cuda")
        w1 = torch.randn((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
        w3 = torch.randn((3, HIDDEN), dtype=torch.bfloat16, device="cuda")
        dummy = torch.zeros((rows, HIDDEN), device="cuda")
        mixed1 = torch.empty((1, rows, HIDDEN), device="cuda")
        mixed3 = torch.empty((3, rows, HIDDEN), device="cuda")
        msg4 = torch.empty((4, rows, HIDDEN), device="cuda")
        out1 = torch.empty((1, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out3 = torch.empty((3, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        paired1 = torch.empty((1, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        paired3 = torch.empty((3, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        fg1 = torch.randn((1, rows, 2 * HIDDEN), device="cuda")
        fg3 = torch.randn((3, rows, 2 * HIDDEN), device="cuda")

        q = torch.randn((rows, HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        qs = torch.randn((HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
        r = torch.randn((rows, HEADS, REL_DIM), dtype=torch.bfloat16, device="cuda")
        rs = torch.randn((HEADS, REL_DIM), dtype=torch.bfloat16, device="cuda")
        pos = torch.arange(rows, dtype=torch.int64, device="cuda")
        q_out = torch.empty_like(q)
        r_clip = torch.empty_like(r)
        rel_in = torch.zeros((rows, HEADS, REL_PADDED), dtype=torch.bfloat16, device="cuda")
        k_bf16 = torch.randn((rows, KV_WIDTH), dtype=torch.bfloat16, device="cuda")
        v_bf16 = torch.randn((rows, KV_WIDTH), dtype=torch.bfloat16, device="cuda")
        k_f32 = torch.empty((rows, KV_WIDTH), device="cuda")
        v_f32 = torch.empty((rows, KV_WIDTH), device="cuda")
        conv_k = torch.empty((rows, KV_WIDTH), device="cuda")
        conv_v = torch.empty((rows, KV_WIDTH), device="cuda")
        k_clip = torch.empty_like(k_bf16)
        v_norm = torch.empty_like(v_bf16)
        kw = torch.randn((KV_WIDTH, TAPS), dtype=torch.bfloat16, device="cuda")
        vw = torch.randn((KV_WIDTH, TAPS), dtype=torch.bfloat16, device="cuda")
        lag_stride = 6656
        slot_stride = 3 * lag_stride
        hist_k = torch.zeros((8, 3, lag_stride), device="cuda")
        hist_v = torch.zeros((8, 3, lag_stride), device="cuda")
        hist_h = torch.zeros((8, 3, lag_stride), device="cuda")
        rep_k = torch.empty((64, lag_stride), device="cuda")
        rep_v = torch.empty((64, lag_stride), device="cuda")
        rep_h = torch.empty((64, lag_stride), device="cuda")
        first = torch.zeros((rows,), dtype=torch.int32, device="cuda")
        cslots = torch.zeros((rows,), dtype=torch.int64, device="cuda")
        kslots = torch.arange(rows, dtype=torch.int64, device="cuda")
        kv_cache = torch.empty((64, 2 * KV_WIDTH), dtype=torch.bfloat16, device="cuda")

        x_h_bf16 = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        w_h = torch.randn((HIDDEN, TAPS), dtype=torch.bfloat16, device="cuda")
        partial = torch.randn((rows, HIDDEN), device="cuda")
        base_h = torch.randn((rows, HIDDEN), device="cuda")
        out_h = torch.empty((rows, HIDDEN), device="cuda")

        r_logits = torch.randn((rows, 128), device="cuda")
        r_bias = torch.randn((128,), device="cuda")
        r_ids = torch.empty((rows, 8), dtype=torch.int32, device="cuda")
        r_w = torch.empty((rows, 8), device="cuda")

        t_norm_s = _time_graph_us(lambda: progs["s_norm"].launch(partial, w1, out1, scalars=(rows,)), reps=args.reps)
        t_norm_f = _time_graph_us(lambda: progs["f_norm"].launch(partial, w1, out1, scalars=(rows,)), reps=args.reps)
        t_rn1_s = _time_graph_us(lambda: (
            progs["s_res1"].launch(arena, dummy, q1, mixed1, scalars=(rows, cap, 3)),
            progs["s_norm"].launch(mixed1, w1, out1, scalars=(rows,)),
        ), reps=args.reps)
        t_rn1_f = _time_graph_us(lambda: progs["f_resnorm1"].launch(arena, q1, w1, out1, scalars=(rows, cap, 3)), reps=args.reps)
        t_rn3_s = _time_graph_us(lambda: (
            progs["s_res3"].launch(arena, dummy, q3, mixed3, scalars=(rows, cap, 3)),
            progs["s_norm"].launch(mixed3[0], w3[0], out3[0], scalars=(rows,)),
            progs["s_norm"].launch(mixed3[1], w3[1], out3[1], scalars=(rows,)),
            progs["s_norm"].launch(mixed3[2], w3[2], out3[2], scalars=(rows,)),
        ), reps=args.reps)
        t_rn3_f = _time_graph_us(lambda: progs["f_resnorm3"].launch(arena, q3, w3, out3, scalars=(rows, cap, 3)), reps=args.reps)
        t_rp1_s = _time_graph_us(lambda: (
            progs["s_res1"].launch(arena, dummy, q1, mixed1, scalars=(rows, cap, 3)),
            progs["s_pair1"].launch(mixed1, msg4[3], paired1, scalars=(rows,)),
        ), reps=args.reps)
        t_rp1_f = _time_graph_us(lambda: progs["f_respair1"].launch(arena, q1, msg4[3], mixed1, paired1, scalars=(rows, cap, 3)), reps=args.reps)
        t_cp3_s = _time_graph_us(lambda: (
            progs["s_res1"].launch(arena, dummy, q1, mixed1, scalars=(rows, cap, 3)),
            progs["s_res4"].launch(prev, dummy, q4, msg4, scalars=(rows, cap, 2)),
            progs["s_pair3b"].launch(mixed1, msg4[:3], paired3, scalars=(rows,)),
        ), reps=args.reps)
        t_cp3_f = _time_graph_us(lambda: progs["f_cross3b"].launch(arena, q1, prev, q4, mixed1, msg4, paired3, scalars=(rows, cap, 3)), reps=args.reps)
        t_bl1_s = _time_graph_us(lambda: progs["s_blend1"].launch(mixed1, msg4[3], fg1, w1, out1, scalars=(rows,)), reps=args.reps)
        t_bl1_f = _time_graph_us(lambda: progs["f_blend1"].launch(mixed1, msg4[3], fg1, w1, out1, scalars=(rows,)), reps=args.reps)
        t_bl3_s = _time_graph_us(lambda: progs["s_blend3b"].launch(mixed1, msg4[:3], fg3, w1, out3, scalars=(rows,)), reps=args.reps)
        t_bl3_f = _time_graph_us(lambda: progs["f_blend3b"].launch(mixed1, msg4[:3], fg3, w1, out3, scalars=(rows,)), reps=args.reps)
        t_ap_s = _time_graph_us(lambda: (
            k_f32.copy_(k_bf16),
            progs["s_conv_kv"].launch(k_f32, kw, k_f32, hist_k, first, cslots, conv_k, rep_k, scalars=(rows, slot_stride, lag_stride, lag_stride)),
            k_bf16.copy_(conv_k),
            v_f32.copy_(v_bf16),
            progs["s_conv_kv"].launch(v_f32, vw, v_f32, hist_v, first, cslots, conv_v, rep_v, scalars=(rows, slot_stride, lag_stride, lag_stride)),
            v_bf16.copy_(conv_v),
            progs["s_clip_q"].launch(q, qs, pos, q_out, scalars=(rows,)),
            progs["s_clip_r"].launch(r, rs, pos, r_clip, scalars=(rows,)),
            progs["s_clip_k"].launch(k_bf16, k_bf16, pos, k_clip, scalars=(rows,)),
            v_f32.copy_(v_bf16),
            progs["s_norm_v"].launch(v_f32, v_f32, v_norm, scalars=(rows,)),
            rel_in[:, :, :REL_DIM].copy_(r_clip),
            progs["s_pack"].launch(k_clip, v_norm, kslots, kv_cache, scalars=(rows,)),
        ), reps=args.reps)
        t_ap_f = _time_graph_us(lambda: progs["f_attn_prep"].launch(
            q, qs, r, rs, pos, q_out, rel_in,
            k_bf16, kw, k_f32, hist_k, v_bf16, vw, v_f32, hist_v,
            first, cslots, kslots, kv_cache, rep_k, rep_v,
            scalars=(rows, slot_stride, lag_stride, lag_stride),
        ), reps=args.reps)
        t_pa_s = _time_graph_us(lambda: (
            out_h.copy_(x_h_bf16),
            progs["s_conv_h"].launch(out_h, w_h, out_h, hist_h, first, cslots, base_h, rep_h, scalars=(rows, slot_stride, lag_stride, lag_stride)),
            x_h_bf16.copy_(base_h),
            base_h.copy_(x_h_bf16),
            torch.add(partial, base_h, out=out_h),
            arena[1, :rows].copy_(out_h),
        ), reps=args.reps)
        t_pa_f = _time_graph_us(lambda: progs["f_post_attn"].launch(
            x_h_bf16, w_h, hist_h, first, cslots, partial, arena, out_h, rep_h,
            scalars=(rows, cap, 1, slot_stride, lag_stride, lag_stride),
        ), reps=args.reps)
        t_pm_s = _time_graph_us(lambda: (
            out_h.copy_(x_h_bf16),
            progs["s_conv_h"].launch(out_h, w_h, out_h, hist_h, first, cslots, base_h, rep_h, scalars=(rows, slot_stride, lag_stride, lag_stride)),
            x_h_bf16.copy_(base_h),
            base_h.copy_(x_h_bf16),
            torch.add(partial, base_h, out=out_h),
            x_h_bf16.copy_(out_h),
            partial.copy_(x_h_bf16),
            arena[1, :rows].copy_(partial),
        ), reps=args.reps)
        t_pm_f = _time_graph_us(lambda: progs["f_post_mlp"].launch(
            x_h_bf16, w_h, hist_h, first, cslots, base_h, partial, arena, rep_h,
            scalars=(rows, cap, 1, slot_stride, lag_stride, lag_stride),
        ), reps=args.reps)
        t_rt_s = _time_graph_us(lambda: progs["s_router"].launch(r_logits, r_bias, r_ids, r_w, scalars=(rows,)), reps=args.reps)
        t_rt_f = _time_graph_us(lambda: progs["f_router"].launch(r_logits, r_bias, r_ids, r_w, scalars=(rows,)), reps=args.reps)

        tot_s = (t_norm_s + 43 * t_rn1_s + 4 * t_rn3_s + 24 * t_rp1_s + 24 * t_cp3_s
                 + 24 * t_bl1_s + 24 * t_bl3_s + 48 * t_ap_s + 48 * t_pa_s + 48 * t_pm_s + 48 * t_rt_s) / 1000.0
        tot_f = (t_norm_f + 43 * t_rn1_f + 4 * t_rn3_f + 24 * t_rp1_f + 24 * t_cp3_f
                 + 24 * t_bl1_f + 24 * t_bl3_f + 48 * t_ap_f + 48 * t_pa_f + 48 * t_pm_f + 48 * t_rt_f) / 1000.0

        print(
            f"M={rows:2d} | norm {t_norm_s:5.2f}->{t_norm_f:4.2f} | rn1 {t_rn1_s:5.2f}->{t_rn1_f:4.2f} "
            f"| rn3 {t_rn3_s:5.2f}->{t_rn3_f:4.2f} | rp1 {t_rp1_s:5.2f}->{t_rp1_f:4.2f} "
            f"| cp3 {t_cp3_s:5.2f}->{t_cp3_f:4.2f} | bl1 {t_bl1_s:5.2f}->{t_bl1_f:4.2f} "
            f"| bl3 {t_bl3_s:5.2f}->{t_bl3_f:4.2f} | ap {t_ap_s:5.2f}->{t_ap_f:4.2f} "
            f"| pa {t_pa_s:5.2f}->{t_pa_f:4.2f} | pm {t_pm_s:5.2f}->{t_pm_f:4.2f} "
            f"| rt {t_rt_s:5.2f}->{t_rt_f:4.2f} || 48v {tot_s:.3f}->{tot_f:.3f} ms (save {tot_s - tot_f:.3f} ms)"
        )

    for rows in wide_list:
        cap = rows
        arena = torch.randn((5, cap, HIDDEN), device="cuda")
        prev = torch.randn((2, cap, HIDDEN), device="cuda")
        q1 = torch.randn((1, HIDDEN), device="cuda")
        q3 = torch.randn((3, HIDDEN), device="cuda")
        q4 = torch.randn((4, HIDDEN), device="cuda")
        w1 = torch.randn((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
        w3 = torch.randn((3, HIDDEN), dtype=torch.bfloat16, device="cuda")
        dummy = torch.zeros((rows, HIDDEN), device="cuda")
        mixed1 = torch.empty((1, rows, HIDDEN), device="cuda")
        mixed3 = torch.empty((3, rows, HIDDEN), device="cuda")
        msg4 = torch.empty((4, rows, HIDDEN), device="cuda")
        out1 = torch.empty((1, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        out3 = torch.empty((3, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        paired1 = torch.empty((1, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        paired3 = torch.empty((3, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        fg1 = torch.randn((1, rows, 2 * HIDDEN), device="cuda")
        fg3 = torch.randn((3, rows, 2 * HIDDEN), device="cuda")
        partial = torch.randn((rows, HIDDEN), device="cuda")
        r_logits = torch.randn((rows, 128), device="cuda")
        r_bias = torch.randn((128,), device="cuda")
        r_ids = torch.empty((rows, 8), dtype=torch.int32, device="cuda")
        r_w = torch.empty((rows, 8), device="cuda")

        t_norm_s = _time_graph_us(lambda: progs["s_norm"].launch(partial, w1, out1, scalars=(rows,)), reps=30, inner=5)
        t_norm_f = _time_graph_us(lambda: progs["f_norm"].launch(partial, w1, out1, scalars=(rows,)), reps=30, inner=5)
        t_rn1_s = _time_graph_us(lambda: (
            progs["s_res1"].launch(arena, dummy, q1, mixed1, scalars=(rows, cap, 3)),
            progs["s_norm"].launch(mixed1, w1, out1, scalars=(rows,)),
        ), reps=30, inner=5)
        t_rn1_f = _time_graph_us(lambda: progs["f_resnorm1"].launch(arena, q1, w1, out1, scalars=(rows, cap, 3)), reps=30, inner=5)
        t_bl1_s = _time_graph_us(lambda: progs["s_blend1"].launch(mixed1, msg4[3], fg1, w1, out1, scalars=(rows,)), reps=30, inner=5)
        t_bl1_f = _time_graph_us(lambda: progs["f_blend1"].launch(mixed1, msg4[3], fg1, w1, out1, scalars=(rows,)), reps=30, inner=5)
        t_rt_s = _time_graph_us(lambda: progs["s_router"].launch(r_logits, r_bias, r_ids, r_w, scalars=(rows,)), reps=30, inner=5)
        t_rt_f = _time_graph_us(lambda: progs["f_router"].launch(r_logits, r_bias, r_ids, r_w, scalars=(rows,)), reps=30, inner=5)
        print(
            f"WIDE M={rows:4d} (us/row) | norm {t_norm_s/rows:.4f}->{t_norm_f/rows:.4f} "
            f"| rn1 {t_rn1_s/rows:.4f}->{t_rn1_f/rows:.4f} | bl1 {t_bl1_s/rows:.4f}->{t_bl1_f/rows:.4f} "
            f"| rt {t_rt_s/rows:.4f}->{t_rt_f/rows:.4f}"
        )


if __name__ == "__main__":
    main()
