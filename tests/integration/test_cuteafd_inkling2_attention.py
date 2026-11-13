"""cuteafd Inkling 2 attention AOT programs vs the family's F32 reference arithmetic.

Oracle: ``fixtures/inkling2_reference.py::relative_attention`` (the cuteafd
#1915 reference, see that file's provenance) in F32 over the whole sequence;
the programs run as the engine chains them (prefill chunks, decode steps,
mixed sequences) over BF16 records. Inputs are random BF16 at the model's
scale (unit q/k/v, bias std 2, sinks around +4 so the sink share is a visible
fraction of Z) with iid bias per distance, so an off-by-one in the distance
gather or a wrong (row, head) sink shows as a cosine drop.

Bound: cosine >= 0.9999 per step (the fork's BF16 attention bar) and max |delta|
within ``ULPS`` BF16 ulps of the output's magnitude. The format ceiling is
printed next to it: the same reference with BF16-rounded probabilities into
the PV product and a BF16 output, which is the arithmetic the kernel runs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x

DECODE, PREFILL = 64, 4096
COS = 0.9999
ULPS = 4  # of BF16 at max |expected|: 4 * 2^-8 relative


def _oracle():
    path = Path(__file__).parent / "fixtures" / "inkling2_reference.py"
    spec = importlib.util.spec_from_file_location("inkling2_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.relative_attention


relative_attention = _oracle()


@pytest.fixture(scope="module")
def g():
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_SMALL

    torch.backends.cuda.matmul.allow_tf32 = False
    return INKLING2_SMALL


_PROGRAMS: dict = {}


def _program(g, **kw):
    key = tuple(sorted(kw.items()))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

        _PROGRAMS[key] = compile_inkling2_attention_aot(g, **kw)
    return _PROGRAMS[key]


def _scratch(program, rows):
    size = program.scratch_bytes(rows).get("scratch", 0)
    return torch.empty(max(size, 1024), dtype=torch.uint8, device="cuda")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


class _Inputs:
    """One sequence's BF16 activations: q, k, v, bias table and sinks per row."""

    def __init__(self, g, kind, length, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        extent = g.extent(kind)

        def randn(*shape, scale=1.0, shift=0.0):
            return (torch.randn(shape, generator=gen, device="cuda") * scale + shift).to(torch.bfloat16)

        self.q = randn(length, g.heads, g.head_dim)
        self.k = randn(length, g.kv_heads, g.head_dim)
        self.v = randn(length, g.kv_heads, g.head_dim)
        self.rel = randn(length, g.heads, extent, scale=2.0)
        self.sinks = (torch.randn((length, g.heads), generator=gen, device="cuda") * 1.5 + 4.0).float()
        self.kind, self.g = kind, g

    def records(self, start, count):
        """BF16 KV records of rows ``start..start+count``: K heads then V heads."""
        return torch.cat([self.k[start:start + count].reshape(count, -1),
                          self.v[start:start + count].reshape(count, -1)], 1).contiguous()

    def reference(self):
        window = self.g.window if self.kind == "local" else None
        gate = torch.zeros_like(self.q, dtype=torch.float32)
        out = relative_attention(self.q.float(), self.k.float(), self.v.float(), self.rel.float(), gate,
                                 self.sinks, window=window)
        return 2.0 * out  # sigmoid(0) = 1/2

    def ceiling(self):
        """The reference with BF16 probabilities and output: the kernel's arithmetic."""
        window = self.g.window if self.kind == "local" else None
        q, k, v = self.q.float(), self.k.float(), self.v.float()
        rel, sinks = self.rel.float(), self.sinks
        length, heads, dim = q.shape
        group = heads // k.shape[1]
        out = torch.empty_like(q)
        extent = rel.shape[-1]
        pos = torch.arange(length, device="cuda")
        distance = pos[:, None] - pos[None, :]
        valid = distance >= 0
        if window is not None:
            valid &= distance < window
        for head in range(heads):
            kv = k[:, head // group], v[:, head // group]
            bias = rel[:, head].gather(-1, distance.clamp(0, extent - 1))
            bias = bias.masked_fill((distance < 0) | (distance >= extent), 0.0)
            scores = (q[:, head] @ kv[0].T + bias / dim ** -0.5) * dim ** -0.5
            scores.masked_fill_(~valid, -float("inf"))
            top = torch.maximum(scores.max(-1).values, sinks[:, head])
            p = torch.exp(scores - top[:, None])
            z = p.sum(-1) + torch.exp(sinks[:, head] - top)
            out[:, head] = (p.to(torch.bfloat16).float() @ kv[1]) / z[:, None]
        return out.to(torch.bfloat16).float()


class _Sequence:
    """One sequence's engine state: paged records (global) or its ring (local)."""

    def __init__(self, g, kind, capacity_tokens, ring_id=0, rings=1, pages_total=None, pages=None, cache=None,
                 ring=None):
        self.g, self.kind, self.len = g, kind, 0
        if kind == "global":
            count = -(-capacity_tokens // g.page_rows)
            self.pages = pages if pages is not None else torch.randperm(count, device="cuda").int()
            rows = (pages_total or count) * g.page_rows
            self.cache = cache if cache is not None else torch.zeros((rows, g.record_elems), dtype=torch.bfloat16,
                                                                     device="cuda")
        else:
            self.ring_id = ring_id
            self.cache = ring if ring is not None else torch.zeros((rings * g.ring_rows, g.record_elems),
                                                                   dtype=torch.bfloat16, device="cuda")

    def slots(self, positions):
        if self.kind == "global":
            return self.pages.long()[positions // self.g.page_rows] * self.g.page_rows + positions % self.g.page_rows
        return self.ring_id * self.g.ring_rows + positions % self.g.ring_rows


def _step(g, kind, route, sequences, splits=8):
    """One engine step over ``(sequence, inputs, n)`` rows in order; returns the BF16 output."""
    capacity = DECODE if route == "decode" else PREFILL
    rows = sum(n for _, _, n in sequences)
    program = _program(g, kind=kind, route=route, max_rows=capacity)
    positions = torch.cat([torch.arange(s.len, s.len + n, device="cuda") for s, _, n in sequences])
    q = torch.cat([x.q[s.len:s.len + n] for s, x, n in sequences]).contiguous()
    rel = torch.cat([x.rel[s.len:s.len + n] for s, x, n in sequences]).contiguous()
    sinks = torch.cat([x.sinks[s.len:s.len + n] for s, x, n in sequences]).contiguous()
    records = torch.cat([x.records(s.len, n) for s, x, n in sequences])
    out = torch.empty((rows, g.heads, g.head_dim), dtype=torch.bfloat16, device="cuda")
    if kind == "global":
        cache = sequences[0][0].cache
        slots = torch.cat([s.slots(torch.arange(s.len, s.len + n, device="cuda")) for s, _, n in sequences])
        cache[slots] = records
        stride = max(len(s.pages) for s, _, _ in sequences)
        if route == "prefill":
            page_table, table_stride = sequences[0][0].pages.contiguous(), 0
        else:
            page_table = torch.zeros((rows, stride), dtype=torch.int32, device="cuda")
            r0 = 0
            for s, _, n in sequences:
                page_table[r0:r0 + n, :len(s.pages)] = s.pages
                r0 += n
            table_stride = stride
        scalars = (rows, table_stride) + ((splits,) if route == "decode" else ())
        program.launch(q, cache, positions, page_table, rel, sinks, out, _scratch(program, rows), scalars=scalars)
    else:
        kv_step = records.contiguous()
        ring_slots = torch.cat([s.slots(torch.arange(s.len, s.len + n, device="cuda")) for s, _, n in sequences])
        firsts, r0 = [], 0
        for _, _, n in sequences:
            firsts.append(torch.full((n,), r0, dtype=torch.int32, device="cuda"))
            r0 += n
        program.launch(q, kv_step, sequences[0][0].cache, positions, ring_slots, torch.cat(firsts), rel, sinks,
                       out, _scratch(program, rows), scalars=(rows,))
    for s, _, n in sequences:
        s.len += n
    return out


def _check(label, got, expected, ceiling):
    c = cosine(got, expected)
    delta = float((got.float() - expected).abs().max())
    scale = float(expected.abs().max())
    bound = ULPS * 2.0 ** -8 * scale
    c_ceiling = cosine(ceiling, expected)
    d_ceiling = float((ceiling - expected).abs().max())
    print(f"{label}: cosine {c:.7f} max|d| {delta:.3e} (bound {bound:.3e}); "
          f"format ceiling cosine {c_ceiling:.7f} max|d| {d_ceiling:.3e}")
    return c, delta, bound


# (schedule of (route, rows) steps over one sequence)
SCHEDULES = {
    "prefill": [("prefill", 1571)],
    "prefill_4096": [("prefill", 4096)],
    "chunked": [("prefill", 700), ("prefill", 300), ("prefill", 571)],
    "decode": [("prefill", 1400)] + [("decode", 1)] * 4 + [("decode", 5)] * 2 + [("decode", 64), ("decode", 79)],
    "short": [("prefill", 37), ("decode", 1), ("decode", 3)],
}


@pytest.mark.parametrize("kind", ["local", "global"])
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_inkling2_attention_chain(g, kind, schedule):
    steps = SCHEDULES[schedule]
    total = sum(n for _, n in steps)
    x = _Inputs(g, kind, total, seed=hash((kind, schedule)) & 0xFFFF)
    expected, ceiling = x.reference(), x.ceiling()
    seq = _Sequence(g, kind, total)
    worst, worst_ratio = 1.0, 0.0
    for route, n in steps:
        while n > 0:
            take = min(n, DECODE if route == "decode" else PREFILL)
            start = seq.len
            got = _step(g, kind, route, [(seq, x, take)])
            torch.cuda.synchronize()
            c, delta, bound = _check(f"inkling2_{kind}_attention {schedule} {route} rows {start}..{start + take}",
                                     got, expected[start:start + take], ceiling[start:start + take])
            worst, worst_ratio = min(worst, c), max(worst_ratio, delta / bound)
            n -= take
    print(f"inkling2_{kind}_attention chain {schedule} ({total} rows): worst cosine {worst:.7f}, "
          f"worst max|d| / bound {worst_ratio:.3f}")
    assert worst >= COS and worst_ratio <= 1.0


@pytest.mark.parametrize("splits", [1, 8, 32])
def test_inkling2_global_decode_splits(g, splits):
    """Split-K decode over a long context: partials + the per-(row, head) sink merge."""
    kind, length, new = "global", 3000, 16
    x = _Inputs(g, kind, length + new, seed=11)
    expected, ceiling = x.reference(), x.ceiling()
    seq = _Sequence(g, kind, length + new)
    _step(g, kind, "prefill", [(seq, x, length)])
    got = _step(g, kind, "decode", [(seq, x, new)], splits=splits)
    torch.cuda.synchronize()
    c, delta, bound = _check(f"inkling2_global_attention decode splits={splits} at {length}", got,
                             expected[length:], ceiling[length:])
    assert c >= COS and delta <= bound


@pytest.mark.parametrize("kind", ["local", "global"])
def test_inkling2_attention_two_sequences(g, kind):
    """One decode step mixing two sequences of different lengths (and ring ids / page sets)."""
    a_len, b_len, a_new, b_new = 1500, 300, 3, 5
    xa, xb = _Inputs(g, kind, a_len + a_new, seed=21), _Inputs(g, kind, b_len + b_new, seed=22)
    ea, eb = xa.reference(), xb.reference()
    ca_, cb_ = xa.ceiling(), xb.ceiling()
    if kind == "global":
        pages = torch.randperm(48, device="cuda").int()
        a = _Sequence(g, kind, a_len + a_new, pages=pages[:24], pages_total=48)
        b = _Sequence(g, kind, b_len + b_new, pages=pages[24:30], cache=a.cache)
    else:
        a = _Sequence(g, kind, 0, ring_id=1, rings=3)
        b = _Sequence(g, kind, 0, ring_id=2, ring=a.cache)
    _step(g, kind, "prefill", [(a, xa, a_len)])
    _step(g, kind, "prefill", [(b, xb, b_len)])
    got = _step(g, kind, "decode", [(a, xa, a_new), (b, xb, b_new)])
    torch.cuda.synchronize()
    ca, da, ba = _check(f"inkling2_{kind}_attention two sequences a", got[:a_new], ea[a_len:], ca_[a_len:])
    cb, db, bb = _check(f"inkling2_{kind}_attention two sequences b", got[a_new:], eb[b_len:], cb_[b_len:])
    assert min(ca, cb) >= COS and da <= ba and db <= bb


def test_inkling2_bias_selects_the_distance(g):
    """A single large bias at one distance per head makes the output that key's value.

    Direct check of the gather index (a shifted index would pick the wrong key)
    with sinks far below the scores.
    """
    kind, length = "local", 300
    x = _Inputs(g, kind, length, seed=31)
    x.rel.zero_()
    x.sinks.fill_(-30.0)
    distances = torch.arange(g.heads, device="cuda") * 5 % 200 + 1  # 1..196, per head
    for head in range(g.heads):
        x.rel[:, head, int(distances[head])] = 40.0  # bias/scale * scale = 40 nats over any q.k
    seq = _Sequence(g, kind, length)
    got = _step(g, kind, "prefill", [(seq, x, length)])
    torch.cuda.synchronize()
    pos = torch.arange(length, device="cuda")
    group = g.heads // g.kv_heads
    for head in range(g.heads):
        d = int(distances[head])
        rows = pos >= d
        want = x.v[(pos - d).clamp_min(0), head // group].float()[rows]
        c = cosine(got[rows, head], want)
        assert c >= 0.999, f"head {head} distance {d}: cosine {c:.6f}"
    print(f"inkling2_local_attention bias selection: {g.heads} heads at distances 1..196 ok")
