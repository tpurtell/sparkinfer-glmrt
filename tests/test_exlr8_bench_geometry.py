"""CPU-only serving geometry contracts; no CUDA imports required."""
import ast
import importlib.util
import os
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_exlr8_a16.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "rank_geometry")
namespace = {}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"), namespace)
rank_geometry = namespace["rank_geometry"]


@pytest.mark.parametrize("world", range(1, 7))
@pytest.mark.parametrize("layer", [0, 3, 40, 77])
def test_rotated_ownership_partitions_each_expert(world, layer):
    all_owned = [rank_geometry(world, rank, layer) for rank in range(world)]
    assert sum(width for _, width in all_owned) == 2048
    for expert in range(256):
        assert sorted(u for owned, _ in all_owned for u in owned[expert]) == list(range(16))
        for rank, (owned, width) in enumerate(all_owned):
            assert width == 128 * len(range(rank, 16, world))
            assert all((16 * expert + u + layer) % world == rank for u in owned[expert])


def test_tp6_keeps_global_slots_and_rotating_heavy_experts():
    r0, w0 = rank_geometry(6, 0, 3)
    r4, w4 = rank_geometry(6, 4, 3)
    assert (w0, w4) == (384, 256)
    assert r0[:3] == [(3, 9, 15), (5, 11), (1, 7, 13)]
    assert r4[:3] == [(1, 7, 13), (3, 9, 15), (5, 11)]
    assert {len(u) for u in r0 + r4} == {2, 3}
    text = SOURCE.read_text()
    assert "units = 16 if owned is not None else 4" in text
    assert "bits[e, list(range(units) if owned is None else owned[e]), :] = 4" in text
    assert "grouped = compile_exlr8_a16_aot(units=units" in text
    assert "state = buffers(4096, 6144, units" in text


@pytest.mark.parametrize("args", [(0, 0, 3), (17, 0, 3), (6, -1, 3), (6, 6, 3), (6, 0, -1)])
def test_invalid_geometry(args):
    with pytest.raises(ValueError):
        rank_geometry(*args)


def test_against_engine_tp_placement():
    path = os.environ.get("CUTEAFD_TP_PLACEMENT")
    if not path:
        pytest.skip("set CUTEAFD_TP_PLACEMENT to the engine's scripts/bench/tp-placement.py")
    spec = importlib.util.spec_from_file_location("engine_tp_placement", path)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    geometry = engine.Geometry(experts=256)
    for world in range(1, 7):
        for layer in (0, 3, 40, 77):
            for rank in range(world):
                owned, width = rank_geometry(world, rank, layer)
                expected = engine.rank_units("rotated", geometry, layer, world, rank)
                assert owned == [expected.units[e] for e in range(256)]
                contiguous = engine.rank_units("contiguous", geometry, layer, world, rank)
                assert width == 128 * len(contiguous.units[0])


@pytest.mark.parametrize("width,tile", [(384, (64, 192, 64, 256)),
                                       (256, (64, 256, 64, 256))])
def test_serving_k4_ws_tile_contract(width, tile):
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "serving_k4_options")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    options = scope["serving_k4_options"](width)
    assert options["force_tile_config"] == tile
    assert width % tile[1] == 0
    assert 6144 % tile[3] == 0
    assert options == dict(force_tile_config=tile, moe_block_size=64,
                          warp_specialized=True, input_format="e4m3_k32",
                          ws_input_stages=3, ws_dynamic_tiles=True, force_blocks_per_sm=1)
    # Unequal N is a WS feature, not a cooperative launch with unequal CTA threads.
    text = SOURCE.read_text()
    assert 'grouped_mixed_state(total, ts, props, width, serving=True)' in text
    assert 'if owned is None else' in text


def test_serving_k4_against_sealed_package():
    import json
    root = os.environ.get("CUTEAFD_EXL3_PACKAGE")
    if not root:
        pytest.skip("set CUTEAFD_EXL3_PACKAGE to sealed GLM K4/K5 package")
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "serving_k4_options")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    for rank in (0, 4):
        meta = json.loads((Path(root) / f"tp6-rank{rank}/m4096/v41_exl3.json").read_text())
        options = scope["serving_k4_options"](meta["intermediate"])
        assert list(options.pop("force_tile_config")) == meta["tile"]
        assert options.pop("moe_block_size") == meta["route_block"]
        assert options.pop("force_blocks_per_sm") == meta["blocks_per_sm"]
        assert all(value == meta[key] for key, value in options.items())
        assert (meta["capacity"], meta["experts"], meta["bits"]) == (4096, 256, [4, 5])
