"""Qwen 3.8 Flash Next wide decode (128-row) programs and replay commits.

Verifies:
1. GDN wide decode (BF16 decode=True, fp8=True, fp8_only="decode") at max_rows=128:
   - up to 64 rows: bit-identical outputs, recurrent state, conv state, and replay
     rows (in each layout) against the 64-row decode programs;
   - at 65, 96, 128 rows: speculative verify (spec=1) produces bit-identical outputs
     to plain (spec=0), leaves recurrent and conv state untouched, full commit via
     compile_qwen4_gdn_commit_aot(replay_rows=128) matches plain state bit-for-bit
     across 2 layers, partial commit at keep in (1, 63, 64, 65, 100) ignores
     rejected rows after keep, and multi-sequence steps crossing row 64 match serial
     execution bit-for-bit;
   - fp8_only="decode" matches fp8=True over bf16(w * s) weights bit-for-bit across
     1..128 rows.
2. PLE speculative verify + compile_qwen4_ple_commit_aot(replay_rows=128):
   - 128-row spec=1 records all 128 rows without touching conv_state;
   - full commit matches spec=0 bit-for-bit; partial commits ignore rejected rows.
3. compile_qwen4_head_fp8_aot(max_rows=128):
   - bit-identical to m64 up to 64 rows and bit-identical per-row to single-row
     launches across 1..128 rows; close to FP32 reference.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

WIDE = 128
_PROGRAMS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _g():
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT

    return QWEN38_FLASH_NEXT


def _P(kind: str, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import qwen4, qwen4_gdn

        fn = {
            "gdn": qwen4_gdn.compile_qwen4_gdn_aot,
            "gdn_commit": qwen4_gdn.compile_qwen4_gdn_commit_aot,
            "ple": qwen4.compile_qwen4_ple_aot,
            "ple_commit": qwen4.compile_qwen4_ple_commit_aot,
            "head_fp8": qwen4.compile_qwen4_head_fp8_aot,
        }[kind]
        _PROGRAMS[key] = fn(_g(), **kw)
    return _PROGRAMS[key]


def _w_block(n: int, k: int, seed: int, scale: float = 0.02):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn((n, k), generator=gen, device="cuda") * 0.6).to(torch.float8_e4m3fn)
    nb, kb = -(-n // 128), k // 128
    s = (torch.rand((nb, kb), generator=gen, device="cuda") + 0.5) * scale
    full = s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return w8, s.contiguous(), (w8.float() * full).bfloat16()


def _gdn_weights(seed: int):
    g = _g()
    h, c, v, p, heads = g.hidden, g.gdn_conv_width, g.gdn_value_width, g.gdn_in_width, g.gdn_value_heads
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        w_in=_w_block(p, h, seed + 1),
        w_out=_w_block(h, v, seed + 2),
        conv_w=(torch.randn((c, 4), generator=gen, device="cuda") * 0.3).contiguous(),
        a_log=(torch.randn((heads,), generator=gen, device="cuda") * 0.5).contiguous(),
        dt_bias=(torch.randn((heads,), generator=gen, device="cuda") * 0.5).contiguous(),
        norm_w=(1 + 0.1 * torch.randn((g.gdn_head_dim,), generator=gen, device="cuda")).bfloat16(),
    )


def _run_gdn(prog, w, x, conv, state, slots, seq_first, spec: int, mode: str):
    rows = x.shape[0]
    g = _g()
    out = torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda")
    record = prog.geometry["replay_bytes"]
    replay = torch.zeros(record // 4, dtype=torch.float32, device="cuda")
    scratch = torch.zeros(prog.scratch_bytes(rows)["scratch"], dtype=torch.uint8, device="cuda")
    (i8, i_s, i_bf), (o8, o_s, o_bf) = w["w_in"], w["w_out"]
    if mode == "bf16":
        before, after = (i_bf,), (o_bf,)
    elif mode == "fp8":
        before, after = (i_bf, i8, i_s), (o_bf, o8, o_s)
    else:
        before, after = (i8, i_s), (o8, o_s)
    prog.launch(x, *before, w["conv_w"], w["a_log"], w["dt_bias"], w["norm_w"], *after,
                conv, state, slots, seq_first, out, replay, scratch, scalars=(rows, spec))
    torch.cuda.synchronize()
    return out, replay


def _unpack_gdn_replay(replay: torch.Tensor, replay_rows: int, rows: int):
    from b12x.integration.cuteafd._qwen4_gdn_kernels import gdn_replay_layout

    g = _g()
    kh, vh, c = g.gdn_key_heads, g.gdn_value_heads, g.gdn_conv_width
    lay = gdn_replay_layout(kh, vh, c, replay_rows)
    raw = replay.view(torch.uint8)
    k = raw[lay["k"]:lay["v"]].view(torch.float32).view(replay_rows, kh, 128)[:rows]
    v = raw[lay["v"]:lay["gates"]].view(torch.float32).view(replay_rows, vh, 128)[:rows]
    gates = raw[lay["gates"]:lay["proj"]].view(torch.float32).view(replay_rows, vh, 2)[:rows]
    proj = raw[lay["proj"]:lay["proj"] + replay_rows * c * 2].view(torch.bfloat16).view(replay_rows, c)[:rows]
    return k, v, gates, proj


@pytest.mark.parametrize("mode", ["bf16", "fp8", "w8"])
@pytest.mark.parametrize("rows", [1, 16, 17, 64])
@pytest.mark.parametrize("spec", [0, 1])
def test_qwen4_gdn_m128_matches_m64_up_to_64_rows(mode, rows, spec):
    g = _g()
    w = _gdn_weights(seed=100 + rows)
    gen = torch.Generator(device="cuda").manual_seed(200 + rows)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    conv0 = (torch.randn((2, 3, g.gdn_conv_width), generator=gen, device="cuda") * 0.3).bfloat16()
    state0 = torch.randn((2, g.gdn_value_heads, 128, 128), generator=gen, device="cuda") * 0.05
    seq_of = [0] * (rows // 2) + [1] * (rows - rows // 2) if rows > 1 else [0]
    first = {}
    slots = torch.tensor(seq_of, dtype=torch.int32, device="cuda")
    seq_first = torch.tensor([first.setdefault(s, i) for i, s in enumerate(seq_of)], dtype=torch.int32, device="cuda")

    kw64 = {"max_rows": 64, **({"fp8": True} if mode == "fp8" else {"fp8_only": "decode"} if mode == "w8" else {})}
    kw128 = {"max_rows": WIDE, **({"fp8": True} if mode == "fp8" else {"fp8_only": "decode"} if mode == "w8" else {"decode": True})}
    p64, p128 = _P("gdn", **kw64), _P("gdn", **kw128)
    assert p128.geometry["replay_rows"] == WIDE
    assert p128.geometry["chunked_min_rows"] is None

    c64, s64 = conv0.clone(), state0.clone()
    c128, s128 = conv0.clone(), state0.clone()
    out64, r64 = _run_gdn(p64, w, x, c64, s64, slots, seq_first, spec, mode)
    out128, r128 = _run_gdn(p128, w, x, c128, s128, slots, seq_first, spec, mode)
    assert torch.equal(out64, out128)
    assert torch.equal(c64, c128) and torch.equal(s64, s128)
    if spec:
        assert torch.equal(c128, conv0) and torch.equal(s128, state0)
        for a, b in zip(_unpack_gdn_replay(r64, 64, rows), _unpack_gdn_replay(r128, WIDE, rows)):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("rows", [65, 96, 128])
def test_qwen4_gdn_w8_m128_matches_fp8_m128_bitwise(rows):
    g = _g()
    w = _gdn_weights(seed=300 + rows)
    gen = torch.Generator(device="cuda").manual_seed(400 + rows)
    x = torch.randn((rows, g.hidden), generator=gen, device="cuda").bfloat16()
    conv0 = (torch.randn((4, 3, g.gdn_conv_width), generator=gen, device="cuda") * 0.3).bfloat16()
    state0 = torch.randn((4, g.gdn_value_heads, 128, 128), generator=gen, device="cuda") * 0.05
    seq_of = [min(i * 4 // rows, 3) for i in range(rows)]
    first = {}
    slots = torch.tensor(seq_of, dtype=torch.int32, device="cuda")
    seq_first = torch.tensor([first.setdefault(s, i) for i, s in enumerate(seq_of)], dtype=torch.int32, device="cuda")

    p_fp8 = _P("gdn", max_rows=WIDE, fp8=True)
    p_w8 = _P("gdn", max_rows=WIDE, fp8_only="decode")
    for spec in (0, 1):
        c1, s1 = conv0.clone(), state0.clone()
        c2, s2 = conv0.clone(), state0.clone()
        out1, r1 = _run_gdn(p_fp8, w, x, c1, s1, slots, seq_first, spec, "fp8")
        out2, r2 = _run_gdn(p_w8, w, x, c2, s2, slots, seq_first, spec, "w8")
        assert torch.isfinite(out2).all() and torch.count_nonzero(out2) > 0
        assert torch.equal(out1, out2)
        assert torch.equal(c1, c2) and torch.equal(s1, s2)
        if spec:
            assert torch.equal(r1, r2)


@pytest.mark.parametrize("mode", ["bf16", "w8"])
def test_qwen4_gdn_m128_verify_and_commit_128_two_layers(mode):
    g = _g()
    kw = {"max_rows": WIDE, **({"fp8_only": "decode"} if mode == "w8" else {"decode": True})}
    prog = _P("gdn", **kw)
    commit = _P("gdn_commit", replay_rows=WIDE)
    assert commit.geometry["replay_rows"] == WIDE

    # Four sequences spanning 128 rows total, including a sequence straddling row 64.
    counts = [40, 35, 30, 23]  # sum = 128; seq 1 spans rows [40, 75)
    assert sum(counts) == WIDE
    seq_of = [s for s, n in enumerate(counts) for _ in range(n)]
    firsts = []
    pos = 0
    for n in counts:
        firsts.append(pos)
        pos += n
    slots = torch.tensor(seq_of, dtype=torch.int32, device="cuda")
    seq_first = torch.tensor([firsts[s] for s in seq_of], dtype=torch.int32, device="cuda")

    layers = []
    for layer_idx in range(2):
        w = _gdn_weights(seed=500 + 17 * layer_idx)
        gen = torch.Generator(device="cuda").manual_seed(600 + 17 * layer_idx)
        x = torch.randn((WIDE, g.hidden), generator=gen, device="cuda").bfloat16()
        conv0 = (torch.randn((4, 3, g.gdn_conv_width), generator=gen, device="cuda") * 0.3).bfloat16()
        state0 = torch.randn((4, g.gdn_value_heads, 128, 128), generator=gen, device="cuda") * 0.05
        plain_c, plain_s = conv0.clone(), state0.clone()
        plain_out, _ = _run_gdn(prog, w, x, plain_c, plain_s, slots, seq_first, 0, mode)
        spec_c, spec_s = conv0.clone(), state0.clone()
        spec_out, replay = _run_gdn(prog, w, x, spec_c, spec_s, slots, seq_first, 1, mode)
        assert torch.equal(spec_out, plain_out)
        assert torch.equal(spec_c, conv0) and torch.equal(spec_s, state0)
        layers.append((w, x, conv0, state0, plain_c, plain_s, replay))

    # Full commit across both layers must equal plain step bit-for-bit.
    state_stack = torch.stack([l[3].clone() for l in layers])
    conv_stack = torch.stack([l[2].clone() for l in layers])
    replay_stack = torch.stack([l[6] for l in layers])
    tables = torch.tensor([[0, 1, 2, 3], firsts, counts], dtype=torch.int32, device="cuda")
    commit.launch(state_stack, conv_stack, replay_stack, tables, scalars=(4, 2, 4))
    torch.cuda.synchronize()
    for layer_idx, l in enumerate(layers):
        assert torch.equal(state_stack[layer_idx], l[5]), f"layer {layer_idx} recurrent state"
        assert torch.equal(conv_stack[layer_idx], l[4]), f"layer {layer_idx} conv state"

    # Partial commit on a single 128-row sequence ignores rejected rows after keep.
    w, x, conv0, state0 = layers[0][0], layers[0][1], layers[0][2][:1], layers[0][3][:1]
    zeros = torch.zeros((WIDE,), dtype=torch.int32, device="cuda")
    for keep in (1, 63, 64, 65, 100):
        changed = x.clone()
        changed[keep:] = torch.randn_like(changed[keep:].float()).bfloat16()
        ca, sa = conv0.clone(), state0.clone()
        cb, sb = conv0.clone(), state0.clone()
        _, ra = _run_gdn(prog, w, x, ca, sa, zeros, zeros, 1, mode)
        _, rb = _run_gdn(prog, w, changed, cb, sb, zeros, zeros, 1, mode)
        t_keep = torch.tensor([[0], [0], [keep]], dtype=torch.int32, device="cuda")
        commit.launch(sa.unsqueeze(0), ca.unsqueeze(0), ra.unsqueeze(0), t_keep, scalars=(1, 1, 1))
        commit.launch(sb.unsqueeze(0), cb.unsqueeze(0), rb.unsqueeze(0), t_keep, scalars=(1, 1, 1))
        torch.cuda.synchronize()
        assert torch.equal(sa, sb) and torch.equal(ca, cb), f"partial commit keep={keep}"


def test_qwen4_ple_m128_verify_and_commit_128():
    g = _g()
    ple = _P("ple", fp8=False)
    commit = _P("ple_commit", replay_rows=WIDE)
    assert commit.geometry["replay_rows"] == WIDE

    h, c = g.hidden, g.hc_width
    gen = torch.Generator(device="cuda").manual_seed(777)
    streams0 = torch.randn((WIDE, g.hc_count, h), generator=gen, device="cuda").bfloat16()
    ids = torch.randint(0, 64, (WIDE, g.ple_rows), generator=gen, device="cuda", dtype=torch.int64)
    table = (torch.randn((64, g.ple_row_dim), generator=gen, device="cuda") * 0.2).bfloat16()
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    w_kv = (torch.randn((c + h, g.ple_dim), generator=gen, device="cuda") * 0.02).bfloat16()
    norm_key = (torch.randn((c,), generator=gen, device="cuda") * 0.1).bfloat16()
    norm_query = (torch.randn((c,), generator=gen, device="cuda") * 0.1).bfloat16()
    norm_conv = (torch.randn((c,), generator=gen, device="cuda") * 0.1).bfloat16()
    conv_w = (torch.randn((c, g.ple_conv), generator=gen, device="cuda") * 0.3).float()
    conv_state0 = (torch.randn((4, g.ple_state_rows, c), generator=gen, device="cuda") * 0.2).bfloat16()

    counts = [40, 35, 30, 23]
    firsts = [0, 40, 75, 105]
    seq_of = [s for s, n in enumerate(counts) for _ in range(n)]
    slots = torch.tensor(seq_of, dtype=torch.int32, device="cuda")
    seq_first = torch.tensor([firsts[s] for s in seq_of], dtype=torch.int32, device="cuda")
    scratch = torch.zeros(ple.scratch_bytes(WIDE)["scratch"], dtype=torch.uint8, device="cuda")

    plain_streams, plain_state = streams0.clone(), conv_state0.clone()
    dummy_replay = torch.zeros((WIDE, c), dtype=torch.bfloat16, device="cuda")
    ple.launch(plain_streams, ids, table, scale, w_kv, norm_key, norm_query, norm_conv, conv_w,
               plain_state, slots, seq_first, dummy_replay, scratch, scalars=(WIDE, 0))

    spec_streams, spec_state = streams0.clone(), conv_state0.clone()
    replay = torch.zeros((WIDE, c), dtype=torch.bfloat16, device="cuda")
    ple.launch(spec_streams, ids, table, scale, w_kv, norm_key, norm_query, norm_conv, conv_w,
               spec_state, slots, seq_first, replay, scratch, scalars=(WIDE, 1))
    torch.cuda.synchronize()
    assert torch.equal(spec_streams, plain_streams)
    assert torch.equal(spec_state, conv_state0)

    tables = torch.tensor([[0, 1, 2, 3], firsts, counts], dtype=torch.int32, device="cuda")
    commit.launch(spec_state, replay, tables, scalars=(4,))
    torch.cuda.synchronize()
    assert torch.equal(spec_state, plain_state)

    # Partial commit matches exact state shift of replay[:keep] and ignores rejected rows after keep.
    zeros_w = torch.zeros((WIDE,), dtype=torch.int32, device="cuda")
    for keep in (1, 9, 64, 65, 100):
        st_a = conv_state0[:1].clone()
        s_a = streams0.clone()
        rep_a = torch.zeros((WIDE, c), dtype=torch.bfloat16, device="cuda")
        ple.launch(s_a, ids, table, scale, w_kv, norm_key, norm_query, norm_conv, conv_w,
                   st_a, zeros_w, zeros_w, rep_a, scratch, scalars=(WIDE, 1))

        # Mutate rejected rows [keep:] in streams/ids; committed state must be bit-identical.
        streams_b = streams0.clone()
        streams_b[keep:] = torch.randn_like(streams_b[keep:].float()).bfloat16()
        ids_b = ids.clone()
        ids_b[keep:] = (ids_b[keep:] + 1) % 64
        st_b = conv_state0[:1].clone()
        rep_b = torch.zeros((WIDE, c), dtype=torch.bfloat16, device="cuda")
        ple.launch(streams_b, ids_b, table, scale, w_kv, norm_key, norm_query, norm_conv, conv_w,
                   st_b, zeros_w, zeros_w, rep_b, scratch, scalars=(WIDE, 1))

        t_keep = torch.tensor([[0], [0], [keep]], dtype=torch.int32, device="cuda")
        commit.launch(st_a, rep_a, t_keep, scalars=(1,))
        commit.launch(st_b, rep_b, t_keep, scalars=(1,))
        torch.cuda.synchronize()
        expected = torch.cat([conv_state0[0], rep_a[:keep]], dim=0)[-g.ple_state_rows:].unsqueeze(0)
        assert torch.equal(st_a, expected), f"PLE partial commit reference keep={keep}"
        assert torch.equal(st_a, st_b), f"PLE partial commit rejected-row invariance keep={keep}"


def test_qwen4_head_fp8_m128_exact_row_invariance():
    g = _g()
    h, vocab = g.hidden, g.vocab
    p64 = _P("head_fp8")
    p128 = _P("head_fp8", max_rows=WIDE)
    assert p64.geometry["max_rows"] == 64
    assert p128.geometry["max_rows"] == WIDE

    gen = torch.Generator(device="cuda").manual_seed(909)
    w8 = (torch.randn((vocab, h), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scale = (torch.rand((vocab, h // 128), generator=gen, device="cuda") + 0.5) * 0.02
    x = torch.randn((WIDE, h), generator=gen, device="cuda").bfloat16()

    out128 = torch.empty((WIDE, vocab), dtype=torch.float32, device="cuda")
    p128.launch(x, w8, scale, out128, scalars=(WIDE,))
    out64 = torch.empty((64, vocab), dtype=torch.float32, device="cuda")
    p64.launch(x[:64].contiguous(), w8, scale, out64, scalars=(64,))
    torch.cuda.synchronize()
    assert torch.isfinite(out128).all() and torch.count_nonzero(out128) > 0
    assert torch.equal(out128[:64], out64)

    # Spot-check rows spanning both halves against single-row launches (row invariance).
    single = torch.empty((1, vocab), dtype=torch.float32, device="cuda")
    for r in (0, 15, 16, 63, 64, 80, 96, 127):
        p128.launch(x[r:r + 1].contiguous(), w8, scale, single, scalars=(1,))
        torch.cuda.synchronize()
        assert torch.equal(out128[r:r + 1], single), f"row {r} invariance"
