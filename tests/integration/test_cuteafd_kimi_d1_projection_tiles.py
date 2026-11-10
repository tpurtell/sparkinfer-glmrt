"""Kimi K3 BF16 decode/verify projections (rows 1..64) with ``DECODE_TMA_TILES``.

Covers all 7 Kimi K3 BF16 projection shapes used in KDA, NoPE MLA, MLA block
drafter, shared FFN, and latent bridge:

- ``24752x7168``: KDA packed ``q/k/v/g/f_a/beta``
- ``6144x7168``: shared ``gate/up`` + MLA h48 ``gate_proj``
- ``7168x6144``: KDA / MLA h48 ``o_proj``
- ``7168x3072``: shared ``down``
- ``3584x7168``: ``latent_down``
- ``7168x3584``: ``latent_up``
- ``2112x7168``: MLA packed ``q_a/kv_a``

Checks:
1. Canonical ``RoutedBf16Projection(max_skinny_rows=0)`` and ``glm_projection``
   select the shape's ``DECODE_TMA_TILES`` entry up to ``<= 64`` rows and hand
   off ``rows > 64`` unchanged to ``prefill_config`` / ``WIDE_PREFILL_CONFIG``.
2. Outputs are byte-identical (``int16`` bit patterns) to ``prefill_config`` at
   decode, band-boundary, and past-table handoff rows ``1, 5, 16, 17, 32, 33,
   64, 65``, with NaN-poisoned input neighbors and output tail preserved across
   arena row offsets.
3. Captured CUDA graphs replay fresh activations with the exact bytes of the
   eager ``prefill_config`` launch.
"""

from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

#: (name, K = input width, N = output features, output dtype)
SHAPES = {
    "kda_in": (7168, 24752, torch.bfloat16),
    "shared_gate_up": (7168, 6144, torch.bfloat16),
    "kda_mla_out": (6144, 7168, torch.bfloat16),
    "shared_down": (3072, 7168, torch.bfloat16),
    "latent_down": (7168, 3584, torch.bfloat16),
    "latent_up": (3584, 7168, torch.bfloat16),
    "mla_qkv_a": (7168, 2112, torch.bfloat16),
}
ROWS = (1, 5, 16, 17, 32, 33, 64, 65)
_PROGRAMS: dict = {}
_TENSORS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _b12x():
    require_b12x()


def _cute_dtype(dtype):
    import cutlass

    return cutlass.Float32 if dtype == torch.float32 else cutlass.BFloat16


def _canonical(name: str):
    from b12x.integration.cuteafd import pointwise

    if ("canonical", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        _PROGRAMS[("canonical", name)] = pointwise.compile_bf16_projection_aot(
            width=width, outputs=outputs, output_dtype=dtype
        )
    return _PROGRAMS[("canonical", name)]


def _default(name: str):
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    if ("default", name) not in _PROGRAMS:
        width, outputs, dtype = SHAPES[name]
        launch = TmaBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), tiles_by_rows=())
        _PROGRAMS[("default", name)] = compile_program(
            launch,
            name="test_kimi_d1_projection_default",
            operands=(
                Operand("x", torch.bfloat16, f"[rows,{width}]"),
                Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
                Operand("out", dtype, f"[rows,{outputs}]", "out"),
            ),
            scalars=(Scalar("rows"),),
            key=launch.key(),
        )
    return _PROGRAMS[("default", name)]


def _weight(name: str) -> torch.Tensor:
    if ("w", name) not in _TENSORS:
        width, outputs, _ = SHAPES[name]
        gen = torch.Generator(device="cuda").manual_seed(outputs * 13 + width)
        _TENSORS[("w", name)] = (
            torch.randn((outputs, width), generator=gen, device="cuda") / 16
        ).bfloat16()
    return _TENSORS[("w", name)]


def _activations(name: str, rows: int, seed: int) -> torch.Tensor:
    width = SHAPES[name][0]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, width), generator=gen, device="cuda").bfloat16()


def _project(prog, x: torch.Tensor, w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = torch.full((x.shape[0], w.shape[0]), float("nan"), dtype=dtype, device="cuda")
    prog.launch(x, w, out, scalars=(int(x.shape[0]),))
    torch.cuda.synchronize()
    return out


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def test_every_kimi_d1_shape_selects_its_decode_table_and_preserves_prefill_handoff():
    from b12x.gemm.bf16_gemv._skinny import DECODE_TMA_TILES, RoutedBf16Projection
    from b12x.integration.cuteafd._glm_kernels import glm_projection

    for name, (width, outputs, dtype) in SHAPES.items():
        assert (outputs, width) in DECODE_TMA_TILES, f"{name}: missing DECODE_TMA_TILES entry"
        table = DECODE_TMA_TILES[(outputs, width)]
        route = RoutedBf16Projection(outputs, width, out_dtype=_cute_dtype(dtype), max_skinny_rows=0)
        glm_route = glm_projection(outputs, width, wide=True)
        assert route.tiles_by_rows == table
        assert glm_route.tiles_by_rows == table
        assert glm_route.wide is not None
        assert table and 1 <= table[0][0] <= table[-1][0] <= 64, (
            f"{name}: decode/verify table must stay within rows 1..64"
        )
        assert not route.large.config["compensated"], f"{name}: must stay uncompensated K16"


def test_kimi_d1_7tuple_tile_k_and_tile_n_validation():
    from b12x.gemm.bf16_gemv._skinny import RowTiledBf16Projection

    rt = RowTiledBf16Projection(24752, 7168, ((16, 1, 4, 16, 32, 6, 128),))
    assert rt.tiled[0].kernel.tile_k == 128
    assert rt.default.kernel.tile_k == 64
    for bad, match in (
        (((16, 1, 4, 16, 32, 6, 96),), "tile_k"),
        (((16, 1, 1, 16, 8, 6),), "tile_n"),
        (((16, 1, 4, 16, 32, 9, 128),), "shared memory"),
    ):
        with pytest.raises(ValueError, match=match):
            RowTiledBf16Projection(24752, 7168, bad)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_kimi_d1_row_tiles_match_default_route_and_preserve_poisoned_tails(name: str, rows: int):
    width, outputs, dtype = SHAPES[name]
    w = _weight(name)
    x = _activations(name, rows, seed=3000 * rows + outputs)
    want = _project(_default(name), x, w, dtype)
    assert bool(torch.isfinite(want.float()).all().item()) and bool((want.float().abs().max() > 0).item())

    # Launch at a nonzero row offset surrounded by NaN-poisoned input rows and NaN output tail.
    offset = 5
    x_arena = torch.full((offset + rows + 8, width), float("nan"), dtype=torch.bfloat16, device="cuda")
    x_arena[offset : offset + rows].copy_(x)
    out_arena = torch.full((rows + 8, outputs), float("nan"), dtype=dtype, device="cuda")
    _canonical(name).launch(x_arena[offset:], w, out_arena, scalars=(rows,))
    torch.cuda.synchronize()
    assert _same(out_arena[:rows], want), f"{name} at {rows} rows: canonical route != prefill_config"
    assert bool(torch.isnan(out_arena[rows:]).all().item()), f"{name} at {rows} rows: poisoned output tail corrupted"


@pytest.mark.parametrize("rows", [16, 64, 65])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_kimi_d1_captured_graph_replays_canonical_bytes(name: str, rows: int):
    _, _, dtype = SHAPES[name]
    w = _weight(name)
    x = _activations(name, rows, seed=9000 + rows)
    want = _project(_default(name), x, w, dtype)
    prog = _canonical(name)
    buffer = torch.zeros_like(x)
    out = torch.zeros_like(want)
    prog.launch(buffer, w, out, scalars=(rows,))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prog.launch(buffer, w, out, scalars=(rows,))
    torch.cuda.synchronize()
    for _ in range(2):
        buffer.copy_(x)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, want), f"{name} at {rows} rows: graph replay != eager prefill_config"
