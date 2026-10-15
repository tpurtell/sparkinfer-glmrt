"""Qwen 3.8 Flash Next decode projections with the BF16 TMA tiles picked by the live rows
(``decode_tiles=True``, ``qwen4_tiles.QWEN4_DECODE_TILES``, ``RowTiledBf16Projection``): the bytes of the
default route.

- The three decode projection shapes alone (``glm_projection`` with the shape's table against
  ``glm_projection`` without), at 1, 2, 8, 9, 12, 16, 17, 32, 33, 64, 65, 127 and 128 rows (each table
  entry at its first and last rows, at one and at several row tiles, and the rows past the last entry):
  byte-identical outputs, and the default route's outputs within the routed projection's bound of the
  FP64 product.
- The decode programs built with ``decode_tiles`` against the default build: the same ABI and scratch and
  byte-identical outputs at 1, 8, 9, 16, 17, 32 and 64 rows -- the attention o_proj's output, the GDN
  program's output, recurrent state, conv state and replay record (the in- and out-projection), and the
  attention producer's query, gate, index query, KV cache, token keys and index-pool keys.
- A CUDA graph of the tiled attn_o launch replays new activations written into the captured buffers with
  the bytes of the eager default launch.
- The FP8 and FP8-only routes (their own decode schedules) and a shape without a table are refused, and
  the tile table stays out of the arithmetic of a program that does not ask for it.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

ROWS = (1, 2, 8, 9, 12, 16, 17, 32, 33, 64, 65, 127, 128)
#: Rows the decode programs are launched at (their capacity is 64).
PROGRAM_ROWS = (1, 8, 9, 16, 17, 32, 64)
_PROGRAMS: dict = {}
_TENSORS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _g():
    from b12x.integration.cuteafd import QWEN38_FLASH_NEXT

    return QWEN38_FLASH_NEXT


def _shape(name: str) -> tuple[int, int]:
    """The weight shape ``[N, K]`` of the GDN in- or o-projection or the attention in-projection."""
    g = _g()
    return {"gdn_in": (g.gdn_in_width, g.hidden), "gdn_out": (g.hidden, g.gdn_value_width),
            "attn_in": (g.attn_in_width, g.hidden), "attn_out": (g.hidden, g.heads * g.head_dim)}[name]


def _tiles(name: str) -> tuple:
    from b12x.integration.cuteafd.qwen4_tiles import qwen4_decode_tiles

    return qwen4_decode_tiles(_shape(name))


def _projection(name: str, tiled: bool):
    """``glm_projection`` of a decode projection's weight shape alone, with or without the shape's tiles."""
    key = ("projection", name, tiled)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
        from b12x.integration.cuteafd._glm_kernels import glm_projection

        n, k = _shape(name)
        launch = glm_projection(n, k, tiles_by_rows=_tiles(name) if tiled else ())
        _PROGRAMS[key] = compile_program(
            launch, name="test_qwen4_decode_tiles_projection",
            operands=(Operand("x", torch.bfloat16, f"[rows,{k}]"), Operand("w", torch.bfloat16, f"[{n},{k}]"),
                      Operand("out", torch.bfloat16, f"[rows,{n}]", "out")),
            scalars=(Scalar("rows"),), key=launch.key())
    return _PROGRAMS[key]


def _program(kind: str, tiled: bool):
    """A qwen4 decode program (capacity 64): ``attn_o``, ``gdn`` or ``attn_producer``, tiled or default."""
    key = (kind, tiled)
    if key not in _PROGRAMS:
        from b12x.integration.cuteafd import qwen4_attention, qwen4_gdn

        kw = {"decode_tiles": True} if tiled else {}
        _PROGRAMS[key] = {
            "attn_o": lambda: qwen4_attention.compile_qwen4_attn_o_aot(_g(), max_rows=64, **kw),
            "gdn": lambda: qwen4_gdn.compile_qwen4_gdn_aot(_g(), max_rows=64, **kw),
            "attn_producer": lambda: qwen4_attention.compile_qwen4_attn_producer_aot(_g(), max_rows=64, **kw),
        }[kind]()
    return _PROGRAMS[key]


def _weight(name: str) -> torch.Tensor:
    if ("w", name) not in _TENSORS:
        n, k = _shape(name)
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        _TENSORS[("w", name)] = (torch.randn((n, k), generator=gen, device="cuda") / 16).bfloat16()
    return _TENSORS[("w", name)]


def _activations(name: str, rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, _shape(name)[1]), generator=gen, device="cuda").bfloat16()


def _rand(shape, seed: int, *, scale: float = 1.0 / 16, dtype=torch.bfloat16) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tensor = torch.randn(shape, generator=gen, device="cuda") * scale
    return tensor.bfloat16() if dtype == torch.bfloat16 else tensor.to(dtype)


def _scratch(prog, rows: int) -> torch.Tensor:
    nbytes = max(prog.scratch_bytes(rows).values(), default=256)
    return torch.zeros(max(nbytes, 256), dtype=torch.uint8, device="cuda")


def _project(prog, x, w) -> torch.Tensor:
    out = torch.full((x.shape[0], w.shape[0]), float("nan"), dtype=torch.bfloat16, device="cuda")
    prog.launch(x, w, out, scalars=(int(x.shape[0]),))
    torch.cuda.synchronize()
    return out


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-identical: the bit patterns, so signed zeros and NaN payloads count."""
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _within_bound(out, x, w, chunk: int = 2048) -> bool:
    """``|out - ref| <= K * 2^-24 * sum_k |x_k w_k| + half a BF16 ulp`` of the FP64 product (the routed
    projection's contract, test_cuteafd_skinny_projection.py), a block of outputs at a time."""
    xd, xa = x.double(), x.double().abs()
    for first in range(0, w.shape[0], chunk):
        wd = w[first:first + chunk].double()
        ref = xd @ wd.t()
        bound = x.shape[1] * 2.0 ** -24 * (xa @ wd.abs().t()) + ref.abs() * 2.0 ** -8 + 1e-30
        if not bool(((out[:, first:first + chunk].double() - ref).abs() <= bound).all()):
            return False
    return True


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", ["gdn_in", "gdn_out", "attn_in", "attn_out"])
def test_tiles_by_rows_give_the_bytes_of_the_default_route(name, rows):
    w = _weight(name)
    x = _activations(name, rows, seed=1000 * rows + w.shape[0])
    default = _project(_projection(name, False), x, w)
    assert not bool(torch.isnan(default).any()) and _within_bound(default, x, w)
    tiled = _project(_projection(name, True), x, w)
    assert _same(tiled, default), f"{name}-projection at {rows} rows: row tiles {_tiles(name)} against the default"


def _launch_attn_o(prog, rows: int, seed: int) -> tuple:
    """One attn_o launch: (out, inputs); the program is deterministic, so the inputs rebuild the launch."""
    g = _g()
    w = g.heads * g.head_dim
    attn, gate = _rand((rows, w), 3000 + rows), _rand((rows, w), 4000 + rows)
    x = {"attn": attn, "gate": gate, "w_o": _weight("attn_out")}
    out = torch.full((rows, g.hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    prog.launch(attn, gate, x["w_o"], out, _scratch(prog, rows), scalars=(rows,))
    torch.cuda.synchronize()
    return out, x


def _launch_gdn(prog, rows: int, seed: int) -> tuple:
    """One GDN decode launch (spec 0): (out, state, conv state, inputs) on fresh pools."""
    g = _g()
    x = _rand((rows, g.hidden), 5000 + rows + seed)
    pools = {"conv_state": torch.zeros(4, 3, g.gdn_conv_width, dtype=torch.bfloat16, device="cuda"),
             "state": torch.zeros(4, g.gdn_value_heads, 128, 128, dtype=torch.float32, device="cuda")}
    ins = {"x": x, "w_in": _weight("gdn_in"), "w_out": _weight("gdn_out"),
           "conv_w": _rand((g.gdn_conv_width, 4), 6000, scale=0.1, dtype=torch.float32),
           "a_log": -_rand((g.gdn_value_heads,), 7000, scale=1.0, dtype=torch.float32).abs(),
           "dt_bias": _rand((g.gdn_value_heads,), 8000, scale=0.1, dtype=torch.float32),
           "norm_w": _rand((g.gdn_head_dim,), 9000),
           "slots": torch.zeros(rows, dtype=torch.int32, device="cuda"),
           "seq_first": torch.zeros(rows, dtype=torch.int32, device="cuda")}
    out = torch.full((rows, g.hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    record_bytes = int(prog.geometry["replay_bytes"])
    replay = torch.zeros(record_bytes // 4, dtype=torch.float32, device="cuda")
    prog.launch(ins["x"], ins["w_in"], ins["conv_w"], ins["a_log"], ins["dt_bias"], ins["norm_w"], ins["w_out"],
                pools["conv_state"], pools["state"], ins["slots"], ins["seq_first"], out, replay,
                _scratch(prog, rows), scalars=(rows, 0))
    torch.cuda.synchronize()
    return out, pools["state"], pools["conv_state"], replay, ins


def _launch_producer(prog, rows: int, seed: int) -> tuple:
    """One attention-producer decode launch on a one-page cache: (query, gate, index_q, caches, inputs)."""
    g = _g()
    pages = 1
    x = _rand((rows, g.hidden), 11000 + rows + seed)
    kv_slots = torch.arange(rows, dtype=torch.int64, device="cuda")          # page 0, position i
    pool_slots = torch.tensor([(p // 4) if p % 4 == 3 else -1 for p in range(rows)],
                              dtype=torch.int64, device="cuda")
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    caches = {"kv": torch.zeros(pages, g.page_rows * g.kv_record_bytes("bf16"), dtype=torch.uint8, device="cuda"),
              "token_keys": torch.zeros(pages * g.page_rows, g.index_head_dim, dtype=torch.bfloat16, device="cuda"),
              "index": torch.zeros(64, g.index_head_dim, dtype=torch.bfloat16, device="cuda")}
    ins = {"x": x, "w_in": _weight("attn_in"), "q_norm": _rand((g.head_dim,), 12000),
           "k_norm": _rand((g.head_dim,), 13000), "iq_norm": _rand((g.index_head_dim,), 14000),
           "ik_norm": _rand((g.index_head_dim,), 15000), "positions": positions,
           "rope_positions": positions.to(torch.int32)[:, None].expand(-1, 3).contiguous(),
           "block_rope_positions": (positions - positions % 4).to(torch.int32)[:, None].expand(-1, 3).contiguous(),
           "kv_slots": kv_slots, "pool_slots": pool_slots}
    query = torch.full((rows, g.heads, g.head_dim), float("nan"), dtype=torch.bfloat16, device="cuda")
    gate = torch.full((rows, g.heads * g.head_dim), float("nan"), dtype=torch.bfloat16, device="cuda")
    index_q = torch.full((rows, g.index_heads, g.index_head_dim), float("nan"), dtype=torch.bfloat16, device="cuda")
    prog.launch(ins["x"], ins["w_in"], ins["q_norm"], ins["k_norm"], ins["iq_norm"], ins["ik_norm"],
                ins["positions"], ins["rope_positions"], ins["block_rope_positions"], ins["kv_slots"],
                ins["pool_slots"], caches["kv"], caches["token_keys"], caches["index"], query, gate, index_q,
                _scratch(prog, rows), scalars=(rows,))
    torch.cuda.synchronize()
    return query, gate, index_q, caches, ins


_LAUNCHERS = {"attn_o": _launch_attn_o, "gdn": _launch_gdn, "attn_producer": _launch_producer}
#: What each program's launch gives back besides its inputs.
_OUTPUTS = {"attn_o": ("out",), "gdn": ("out", "state", "conv_state", "replay"),
            "attn_producer": ("query", "gate", "index_q", "kv_cache", "token_keys", "index_cache")}


def _named(kind: str, result: tuple) -> dict:
    """The launch's outputs by name (the caches after the attention producer's writes)."""
    if kind == "attn_producer":
        query, gate, index_q, caches, _ = result
        return {"query": query, "gate": gate, "index_q": index_q, "kv_cache": caches["kv"],
                "token_keys": caches["token_keys"], "index_cache": caches["index"]}
    return dict(zip(_OUTPUTS[kind], result[:len(_OUTPUTS[kind])]))


@pytest.mark.parametrize("kind", ["attn_o", "gdn", "attn_producer"])
def test_the_program_records_its_decode_tiles(kind):
    """``decode_tiles`` reaches the program's geometry (and nothing else does)."""
    g, base, tiled = _g(), _program(kind, False), _program(kind, True)
    want = ({"in": _tiles("attn_in")} if kind == "attn_producer" else
            {"in": _tiles("gdn_in"), "out": _tiles("gdn_out")} if kind == "gdn" else {"out": _tiles("attn_out")})
    assert "decode_tiles" not in base.geometry
    assert tiled.geometry["decode_tiles"] == want, f"{kind}: the geometry's decode tiles"
    assert {k: v for k, v in tiled.geometry.items() if k != "decode_tiles"} == dict(base.geometry)
    assert tiled.name == base.name and tiled.abi == base.abi, f"{kind}: the tiled program's ABI"
    assert tiled.scratch_bytes(64) == base.scratch_bytes(64)
    # The table covers the shapes the qwen4 layers actually build (the 48 o_proj launches share one shape).
    assert g.gdn_value_width == g.heads * g.head_dim == _shape("attn_out")[1]


@pytest.mark.parametrize("rows", PROGRAM_ROWS)
@pytest.mark.parametrize("kind", ["attn_o", "gdn", "attn_producer"])
def test_programs_with_decode_tiles_are_the_default_program(kind, rows):
    """The tiled and default programs on the same inputs: every output's bytes."""
    base, tiled = _program(kind, False), _program(kind, True)
    launch = _LAUNCHERS[kind]
    want, got = _named(kind, launch(base, rows, 0)), _named(kind, launch(tiled, rows, 1))
    for name, tensor in want.items():
        assert bool(tensor.float().abs().sum() > 0), f"{kind} {name}: the case produced nothing"
    for name in want:
        assert _same(want[name], got[name]), f"{kind} {name}: decode_tiles against the default at {rows} rows"


@pytest.mark.parametrize("rows", [16, 64])
def test_a_captured_graph_replays_the_tiled_route(rows):
    """cuteafd captures one graph per row bucket: the captured tiled launch (pointers and ``rows`` fixed)
    then runs whatever activations are in its buffers."""
    prog = _program("attn_o", True)
    out, x = _launch_attn_o(_program("attn_o", False), rows, 0)
    buffer = {name: torch.zeros_like(tensor) for name, tensor in x.items()}
    captured = torch.zeros_like(out)
    prog.launch(buffer["attn"], buffer["gate"], buffer["w_o"], captured, _scratch(prog, rows), scalars=(rows,))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prog.launch(buffer["attn"], buffer["gate"], buffer["w_o"], captured, _scratch(prog, rows), scalars=(rows,))
    torch.cuda.synchronize()
    for _ in range(2):
        for name, tensor in x.items():
            buffer[name].copy_(tensor)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _same(captured, out), f"attn_o at {rows} rows: replayed row tiles against the eager default"


def test_decode_tiles_are_checked_and_the_table_stays_out_of_the_default_programs():
    from b12x.integration.cuteafd import qwen4_attention, qwen4_gdn
    from b12x.integration.cuteafd.qwen4_tiles import qwen4_decode_tiles, qwen4_projection

    g = _g()
    assert qwen4_decode_tiles(_shape("gdn_in")) and qwen4_decode_tiles(_shape("attn_in"))
    assert qwen4_decode_tiles(_shape("attn_out")) == qwen4_decode_tiles(_shape("gdn_out"))
    assert qwen4_decode_tiles((2560, 3072)) == ()  # no table: no retile, no key, no geometry entry
    with pytest.raises(ValueError, match="tiles_by_rows"):
        qwen4_projection(2560, 6144, True, tiles_by_rows=qwen4_decode_tiles(_shape("gdn_out")))
    # The FP8 routes schedule their own decode GEMVs and refuse the BF16 tiles.
    for kw in ({"fp8": True}, {"fp8_only": "decode"}, {"fp8_only": "prefill"}):
        with pytest.raises(ValueError, match="decode_tiles"):
            qwen4_gdn.compile_qwen4_gdn_aot(g, max_rows=64, decode_tiles=True, **kw)
        with pytest.raises(ValueError, match="decode_tiles"):
            qwen4_attention.compile_qwen4_attn_producer_aot(g, max_rows=64, decode_tiles=True, **kw)
        with pytest.raises(ValueError, match="decode_tiles"):
            qwen4_attention.compile_qwen4_attn_o_aot(g, max_rows=64, decode_tiles=True, **kw)
