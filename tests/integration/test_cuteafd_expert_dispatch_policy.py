"""CPU contracts for shared coordinator route planning and live-row dispatch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd/fp8_moe.py"


def policy(sm):
    tree = ast.parse(SOURCE.read_text())
    names = {"_w8a8", "_mxfp4_down_a8", "_streams", "auto_large_rows", "_large_route", "prefill_forms", "_resolve"}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and all(isinstance(t, ast.Name) for t in n.targets)
             and isinstance(n.value, (ast.Constant, ast.Tuple))]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"_sm121": lambda: sm == 121}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
    return tree, env


@pytest.mark.parametrize("sm", [120, 121])
@pytest.mark.parametrize("weights", ["fp8", "mxfp4", "nvfp4a4"])
def test_coordinator_auto_covers_decode_and_prefill(sm, weights):
    tree, env = policy(sm)
    threshold = env["auto_large_rows"](False, weights)
    resolved = env["_resolve"]("auto", 4096, False, weights)
    if weights == "nvfp4a4":
        assert resolved == "stream"  # W4A4 for every row, never an A16 GEMV.
        assert threshold == 0
    else:
        assert resolved == "auto"
        assert env["_resolve"]("auto", threshold, False, weights) == "decode"
        assert env["_resolve"]("auto", threshold + 1, False, weights) == "auto"
    expected_large = "prefill" if weights == "fp8" else "stream"
    assert env["_large_route"](False, weights) == expected_large

    # Exercise the actual DSL entry's scalar branch without CUDA imports.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Fp8Moe")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    run.decorator_list = []
    run.returns = None
    for arg in run.args.args:
        arg.annotation = None
    calls = []
    env.update(Int32=int, cutlass=SimpleNamespace(const_expr=bool))
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(SOURCE), "exec"), env)
    plan = SimpleNamespace(route=resolved, threshold=threshold,
                           decode=lambda *args: calls.append("decode"),
                           prefill=lambda *args: calls.append(expected_large))
    for rows in [1, 16, 80, threshold or 128, max(threshold + 1, 256), 4096]:
        env["__call__"](plan, *([None] * 11), rows, None)
        assert calls[-1] == ("decode" if resolved == "auto" and rows <= threshold else expected_large)


def grouping_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Fp8Moe")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "weight_read_grouping")
    env = {"_Route": type("Route", (), {}), "_StreamRoute": type("Stream", (), {})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), env)
    return env


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
@pytest.mark.parametrize("capacity,tiles", [(1, 1), (16, 1), (17, 2), (40, 3), (80, 4), (4096, 4)])
def test_weight_read_groups_use_actual_gemv_launch_tiles(weights, capacity, tiles):
    env = grouping_method()
    route = env["_Route"]()
    route.route = "decode"
    route.gate_up = SimpleNamespace(m_tiles=tiles)
    route.down = SimpleNamespace(m_tiles=tiles)
    launch = SimpleNamespace(route="decode", decode=route)
    assert env["weight_read_grouping"](launch, weights, capacity) == {
        "contract": "exact_grouping", "segments": [{"end_rows": capacity, "group_rows": 16 * tiles}]}


@pytest.mark.parametrize("stream,large", [(False, 64), (True, 128)])
def test_weight_read_groups_follow_actual_auto_cutoff_and_large_route(stream, large):
    env = grouping_method()
    decode = env["_Route"]()
    decode.route = "decode"
    decode.gate_up = decode.down = SimpleNamespace(m_tiles=4)
    prefill = env["_StreamRoute" if stream else "_Route"]()
    if stream:
        prefill.prep = SimpleNamespace(tile_rows=large)
    else:
        prefill.route = "prefill"
        prefill.gate_up = prefill.down = SimpleNamespace(tile_m=large)
    launch = SimpleNamespace(route="auto", threshold=80, decode=decode, prefill=prefill)
    expected = ([{"end_rows": 80, "group_rows": 64}, {"end_rows": 4096, "group_rows": large}]
                if stream else [{"end_rows": 4096, "group_rows": 64}])
    assert env["weight_read_grouping"](launch, "mxfp4" if stream else "fp8", 4096) == {
        "contract": "exact_grouping", "segments": expected}


def test_nvfp4_reports_a_bound_not_a_fabricated_installed_tile():
    env = grouping_method()
    # It must not inspect the occupancy-dependent sparse/wide launch objects.
    assert env["weight_read_grouping"](None, "nvfp4", 4096) == {
        "contract": "conservative_route_upper_bound",
        "segments": [{"end_rows": 4096, "group_rows": 1}]}


def test_weight_read_groups_reject_different_gate_and_down_tiles():
    env = grouping_method()
    route = env["_Route"]()
    route.route = "decode"
    route.gate_up = SimpleNamespace(m_tiles=1)
    route.down = SimpleNamespace(m_tiles=2)
    with pytest.raises(ValueError, match="groups differ"):
        env["weight_read_grouping"](SimpleNamespace(route="decode", decode=route), "fp8", 16)


def test_mxfp4_stream_preserves_native_bf16():
    _, env = policy(120)
    assert env["_streams"](False, "mxfp4")
    assert env["auto_large_rows"](False, "mxfp4") == 1000
    assert env["_resolve"]("auto", 1000, False, "mxfp4") == "decode"
    assert env["_resolve"]("auto", 1001, False, "mxfp4") == "auto"
    assert env["_resolve"]("auto", 1024, False, "mxfp4") == "auto"
    assert env["_resolve"]("stream", 4096, False, "mxfp4") == "stream"
    g = SimpleNamespace(kind="mxfp4")
    assert env["prefill_forms"](g, 1000, True) == []
    assert env["prefill_forms"](g, 1024, True) == ["w8a16"]
    assert env["prefill_forms"](g, 1024, False) == []
    with pytest.raises(ValueError, match="use stream"):
        env["_resolve"]("prefill", 4096, False, "mxfp4")
    _, env121 = policy(121)
    assert env121["auto_large_rows"](True, "mxfp4") == 640
    assert env121["prefill_forms"](g, 640, True) == []
    assert env121["prefill_forms"](g, 1024, True) == ["w8a16"]


def arithmetic_method():
    env = grouping_method()
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Fp8Moe")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "arithmetic_routes")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), env)
    return env


@pytest.mark.parametrize("sm,cutoff", [(120, 1000), (121, 640)])
@pytest.mark.parametrize("weights,wire,prefill", [("mxfp4", True, "auto"),
    ("mxfp4", False, "auto"), ("fp8", True, "auto"), ("fp8", False, "auto"),
    ("fp8", False, "w8a16")])
def test_arithmetic_boundary_survives_identical_traffic_groups(sm, cutoff, weights, wire, prefill):
    _, authority = policy(sm)
    threshold = authority["auto_large_rows"](wire, weights, prefill)
    if weights == "mxfp4":
        assert threshold == cutoff
    env = arithmetic_method()
    decode, large = env["_Route"](), env["_Route"]()
    decode.route, large.route = "decode", "prefill"
    decode.gate_up = decode.down = SimpleNamespace(m_tiles=4)
    large.gate_up = large.down = SimpleNamespace(tile_m=64)
    launch = SimpleNamespace(route="auto", threshold=threshold, decode=decode, prefill=large)
    assert env["weight_read_grouping"](launch, weights, 4096)["segments"] == [
        {"end_rows": 4096, "group_rows": 64}]
    assert env["arithmetic_routes"](launch, weights, 4096) == [
        {"end_rows": threshold, "route_id": 1}, {"end_rows": 4096, "route_id": 2}]
    assert env["arithmetic_routes"](launch, weights, threshold) == [
        {"end_rows": threshold, "route_id": 1}]


def test_unknown_occupancy_route_and_fixed_stream():
    env = arithmetic_method()
    launch = SimpleNamespace(route="stream", prefill=env["_StreamRoute"]())
    assert env["arithmetic_routes"](launch, "nvfp4", 4096) == [{"end_rows": 4096, "route_id": 0}]
    assert env["arithmetic_routes"](launch, "mxfp4", 4096) == [{"end_rows": 4096, "route_id": 3}]


@pytest.mark.parametrize("weights,form,act", [("fp8", "auto", 768), ("fp8", "w8a16", 768),
                                              ("mxfp4", "auto", 400), ("mxfp4", "w8a16", 768)])
def test_segment_scratch_uses_live_rows_but_compiled_metadata_offset(weights, form, act):
    tree, env = policy(121)
    names = {"_regions", "fp8_moe_segment_scratch_bytes"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env.update(Fp8MoeGeometry=object, STREAM_TILE_M=128,
               stream_max_tiles=lambda experts, pairs: (pairs + 127) // 128 + min(experts, pairs),
               meta_words=lambda experts, tiles: 4 + 3 * experts + tiles,
               _align=lambda value: (value + 1023) // 1024 * 1024,
               mxfp8_down_row_bytes=lambda width: (width + width // 32 + 15) // 16 * 16)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
    geometry = SimpleNamespace(weights=weights, kind=weights, hidden=4096, experts=256,
                               top_k=8, slice=384, activations="a16")
    scratch = env["fp8_moe_segment_scratch_bytes"]
    align = env["_align"]
    pairs = 8256 * 8
    tiles = (pairs + 127) // 128 + 256
    expected = align(4 * (4 + 3 * 256 + 16384 * 8 // 128 + 256)) + 2 * align(pairs * 4)
    expected += align(tiles * 128 * act) + align(pairs * 4096 * 2)
    assert scratch(geometry, 8256, 16384, form) == expected
    assert expected < scratch(geometry, 16384, 16384, form)
    # Negative control: sizing the metadata at live rows misses the compiled offset.
    assert scratch(geometry, 8256, 8256, form) < expected
    for rows in (0, 16385):
        with pytest.raises(ValueError):
            scratch(geometry, rows, 16384, form)


def test_indirect_row_address_resolver_and_singleton_key_contract():
    source = SOURCE.with_name("_fp8_moe_stream.py")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StreamFp8GateUp")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("row_address", "key")]
    for method in methods:
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
    addresses = [0x1000, 0x8000, 0x2000]
    env = {"cutlass": SimpleNamespace(const_expr=bool), "Int64": int,
           "cute": SimpleNamespace(make_ptr=lambda *args, **kwargs: addresses,
                                   AddressSpace=SimpleNamespace(gmem=0))}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), env)
    obj = SimpleNamespace(row_addresses=False, row_bytes=4224, inter=384, hidden=4096, experts=256,
                          limit=0.0, a_ring=3, w_ring=3, qmma=True, post_silu=False)
    pointer = SimpleNamespace(toint=lambda: 0x100000)
    old_key = env["key"](obj)
    assert env["row_address"](obj, pointer, 2) == 0x100000 + 2 * 4224
    obj.row_addresses = True
    assert env["row_address"](obj, pointer, 2) == addresses[2]
    assert env["key"](obj) == old_key + ("row_addresses",)
    assert env["row_address"](obj, pointer, 2) != 0x100000 + 2 * 4224
