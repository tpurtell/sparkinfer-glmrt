"""cuteafd Inkling 2 Mini biased-sink GQA attention AOT programs oracle.

Exercises ``INKLING2_MINI`` (hidden=2304, heads=24, kv_heads=8, head_dim=128,
group=3, prefill_tokens=21) across local SWA-1024 BF16 and global BF16/FP8 routes.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from . import test_cuteafd_inkling2_attention as small_attn

DECODE, PREFILL = 64, 4096
COS = small_attn.COS
FP8_COS = 0.995
_PROGRAMS: dict = {}


@pytest.fixture(scope="module")
def g():
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_MINI

    torch.backends.cuda.matmul.allow_tf32 = False
    return INKLING2_MINI


def _program(g, **kw):
    key = (g.name, *sorted(kw.items()))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

        _PROGRAMS[key] = compile_inkling2_attention_aot(g, **kw)
    return _PROGRAMS[key]


def _step(g, kind, route, sequences, max_splits=32, kv_format="bf16"):
    from b12x.integration.cuteafd.inkling2_attention import prefill_tokens
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot

    capacity = DECODE if route == "decode" else PREFILL
    rows = sum(n for _, _, n in sequences)
    program = _program(g, kind=kind, route=route, max_rows=capacity, kv_format=kv_format,
                       **({"max_splits": max_splits} if kind == "global" and route == "decode" else {}))
    positions = torch.cat([torch.arange(s.len, s.len + n, device="cuda") for s, _, n in sequences])
    q = torch.cat([x.q[s.len:s.len + n] for s, x, n in sequences]).contiguous()
    rel = torch.cat([x.rel[s.len:s.len + n] for s, x, n in sequences]).contiguous()
    sinks = torch.cat([x.sinks[s.len:s.len + n] for s, x, n in sequences]).contiguous()
    out = torch.empty((rows, g.heads, g.head_dim), dtype=torch.bfloat16, device="cuda")
    step_tokens = 1 if route == "decode" else prefill_tokens(g)
    if kind == "global":
        cache = sequences[0][0].cache
        slots = torch.cat([s.slots(torch.arange(s.len, s.len + n, device="cuda")) for s, _, n in sequences])
        if kv_format == "fp8":
            pack_key = ("pack_fp8", g.kv_heads)
            if pack_key not in _PROGRAMS:
                _PROGRAMS[pack_key] = compile_inkling2_kv_pack_aot(
                    heads=g.kv_heads, dim=g.head_dim, input_dtype=torch.bfloat16,
                    max_rows=PREFILL, kv_format="fp8",
                )
            k = torch.cat([x.k[s.len:s.len + n] for s, x, n in sequences]).contiguous()
            v = torch.cat([x.v[s.len:s.len + n] for s, x, n in sequences]).contiguous()
            _PROGRAMS[pack_key].launch(k, v, slots.long(), cache, scalars=(rows,))
        else:
            records = torch.cat([x.records(s.len, n) for s, x, n in sequences])
            cache[slots] = records
        stride = max(len(s.pages) for s, _, _ in sequences)
        page_table = torch.zeros((len(sequences), stride), dtype=torch.int32, device="cuda")
        indices, firsts, tiles, r0 = [], [], [], 0
        for member, (seq, _, n) in enumerate(sequences):
            page_table[member, :len(seq.pages)] = seq.pages
            indices.extend([member] * n)
            firsts.extend([r0] * n)
            tiles.extend(range(r0, r0 + n, step_tokens))
            r0 += n
        i32 = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
        scalars = (rows, stride, len(tiles))
        program.launch(q, cache, positions, page_table, i32(indices), i32(firsts), i32(tiles),
                       rel, sinks, out, small_attn._scratch(program, rows), scalars=scalars)
    else:
        kv_step = torch.cat([x.records(s.len, n) for s, x, n in sequences]).contiguous()
        ring_slots = torch.cat([s.slots(torch.arange(s.len, s.len + n, device="cuda")) for s, _, n in sequences])
        firsts, tiles, r0 = [], [], 0
        for _, _, n in sequences:
            firsts.append(torch.full((n,), r0, dtype=torch.int32, device="cuda"))
            tiles.extend(range(r0, r0 + n, step_tokens))
            r0 += n
        tile_rows = torch.tensor(tiles, dtype=torch.int32, device="cuda")
        program.launch(q, kv_step, sequences[0][0].cache, positions, ring_slots, torch.cat(firsts),
                       tile_rows, rel, sinks, out, small_attn._scratch(program, rows),
                       scalars=(rows, len(tiles)))
    for s, _, n in sequences:
        s.len += n
    return out


@pytest.mark.parametrize("kind", ["local", "global"])
@pytest.mark.parametrize("schedule", ["short", "chunked", "decode", "prefill"])
def test_inkling2_mini_attention_chain(g, kind, schedule):
    steps = small_attn.SCHEDULES[schedule]
    total = sum(n for _, n in steps)
    x = small_attn._Inputs(g, kind, total, seed=hash(("mini", kind, schedule)) & 0xFFFF)
    expected, ceiling = x.reference(), x.ceiling()
    seq = small_attn._Sequence(g, kind, total)
    worst, worst_ratio = 1.0, 0.0
    for route, n in steps:
        while n > 0:
            take = min(n, DECODE if route == "decode" else PREFILL)
            start = seq.len
            got = _step(g, kind, route, [(seq, x, take)])
            torch.cuda.synchronize()
            c, delta, bound = small_attn._check(
                f"inkling2_mini_{kind}_attention {schedule} {route} rows {start}..{start + take}",
                got, expected[start:start + take], ceiling[start:start + take],
            )
            worst, worst_ratio = min(worst, c), max(worst_ratio, delta / bound)
            n -= take
    assert worst >= COS and worst_ratio <= 1.0


@pytest.mark.parametrize("max_splits", [1, 8, 32])
def test_inkling2_mini_global_decode_splits_bf16_and_fp8(g, max_splits):
    from b12x.integration.cuteafd.inkling2_attention import record_bytes

    length, new = 2200, 16
    x = small_attn._Inputs(g, "global", length + new, seed=41)
    expected, ceiling = x.reference(), x.ceiling()
    for kv_format in ("bf16", "fp8"):
        seq = small_attn._Sequence(g, "global", length + new)
        if kv_format == "fp8":
            pages_total = len(seq.pages)
            seq.cache = torch.zeros(
                (pages_total * g.page_rows, record_bytes(g, "fp8")),
                dtype=torch.uint8, device="cuda",
            )
        _step(g, "global", "prefill", [(seq, x, length)], kv_format=kv_format)
        got = _step(g, "global", "decode", [(seq, x, new)], max_splits=max_splits, kv_format=kv_format)
        torch.cuda.synchronize()
        c = small_attn.cosine(got, expected[length:])
        print(f"inkling2_mini_global {kv_format} decode max_splits={max_splits}: cosine {c:.7f}")
        assert c >= (COS if kv_format == "bf16" else FP8_COS)


@pytest.mark.parametrize("kind", ["local", "global"])
def test_inkling2_mini_attention_two_sequences(g, kind):
    a_len, b_len, a_new, b_new = 1500, 300, 3, 5
    xa = small_attn._Inputs(g, kind, a_len + a_new, seed=51)
    xb = small_attn._Inputs(g, kind, b_len + b_new, seed=52)
    ea, eb = xa.reference(), xb.reference()
    ca_, cb_ = xa.ceiling(), xb.ceiling()
    if kind == "global":
        pages = torch.randperm(48, device="cuda").int()
        a = small_attn._Sequence(g, kind, a_len + a_new, pages=pages[:24], pages_total=48)
        b = small_attn._Sequence(g, kind, b_len + b_new, pages=pages[24:30], cache=a.cache)
    else:
        a = small_attn._Sequence(g, kind, 0, ring_id=1, rings=3)
        b = small_attn._Sequence(g, kind, 0, ring_id=2, ring=a.cache)
    _step(g, kind, "prefill", [(a, xa, a_len)])
    _step(g, kind, "prefill", [(b, xb, b_len)])
    got = _step(g, kind, "decode", [(a, xa, a_new), (b, xb, b_new)])
    torch.cuda.synchronize()
    ca, da, ba = small_attn._check(f"inkling2_mini_{kind} two_seq a", got[:a_new], ea[a_len:], ca_[a_len:])
    cb, db, bb = small_attn._check(f"inkling2_mini_{kind} two_seq b", got[a_new:], eb[b_len:], cb_[b_len:])
    assert min(ca, cb) >= COS and da <= ba and db <= bb


def test_inkling2_mini_prefill_tokens_and_bias_selects_distance(g):
    from b12x.integration.cuteafd.inkling2_attention import prefill_tokens

    assert prefill_tokens(g) == 21
    length = 300
    x = small_attn._Inputs(g, "local", length, seed=61)
    x.rel.zero_()
    x.sinks.fill_(-30.0)
    distances = torch.arange(g.heads, device="cuda") * 7 % 200 + 1
    for head in range(g.heads):
        x.rel[:, head, int(distances[head])] = 40.0
    seq = small_attn._Sequence(g, "local", length)
    got = _step(g, "local", "prefill", [(seq, x, length)])
    torch.cuda.synchronize()
    pos = torch.arange(length, device="cuda")
    group = g.heads // g.kv_heads
    for head in range(g.heads):
        d = int(distances[head])
        rows = pos >= d
        want = x.v[(pos - d).clamp_min(0), head // group].float()[rows]
        assert small_attn.cosine(got[rows, head], want) >= 0.999
