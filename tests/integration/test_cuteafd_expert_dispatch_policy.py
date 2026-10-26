"""CPU contracts for shared coordinator route planning and live-row dispatch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd/fp8_moe.py"


def policy(sm):
    tree = ast.parse(SOURCE.read_text())
    names = {"_w8a8", "_mxfp4_down_a8", "_streams", "_bf16_fp8_large_rows", "auto_large_rows",
             "_large_route", "prefill_forms", "_resolve"}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and all(isinstance(t, ast.Name) for t in n.targets)
             and (isinstance(n.value, (ast.Constant, ast.Tuple))
                  or n.targets[0].id == "_BF16_FP8_GEMM_ROWS")]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"_sm121": lambda: sm == 121}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
    return tree, env


@pytest.mark.parametrize("sm", [120, 121])
@pytest.mark.parametrize("weights", ["fp8", "mxfp4", "nvfp4a4"])
def test_coordinator_auto_covers_decode_and_prefill(sm, weights):
    tree, env = policy(sm)
    threshold = env["auto_large_rows"](False, weights)
    capacity = max(4096, threshold + 1)
    resolved = env["_resolve"]("auto", capacity, False, weights)
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


def fp8_geometry(hidden, intermediate, width, activation="silu_bf16_product", limit=0.0):
    return SimpleNamespace(hidden=hidden, intermediate=intermediate, slice=width,
                           weights="fp8", kind="fp8", activations="a16", projection="bf16",
                           activation=activation, swiglu_limit=limit)


@pytest.mark.parametrize("sm,hidden,intermediate,width,activation,limit,threshold", [
    (120, 4096, 1536, 768, "silu_post_clamp_fp32_product", 7.0, 2304),
    (120, 6144, 2048, 1024, "silu_bf16_product", 0.0, 2752),
    (120, 4096, 2048, 1024, "silu_bf16_product", 10.0, 4096),
    (120, 4096, 2048, 1024, "silu_bf16_product", 0.0, 4096),
    (121, 4096, 1536, 256, "silu_post_clamp_fp32_product", 7.0, 4096),
    (121, 4096, 1536, 384, "silu_post_clamp_fp32_product", 7.0, 4096),
    (121, 6144, 2048, 384, "silu_bf16_product", 0.0, 4096),
    (121, 4096, 2048, 512, "silu_bf16_product", 10.0, 4096),
    (121, 4096, 2048, 512, "silu_bf16_product", 0.0, 4096),
])
def test_measured_bf16_fp8_cutoffs(sm, hidden, intermediate, width, activation, limit, threshold):
    _, env = policy(sm)
    g = fp8_geometry(hidden, intermediate, width, activation, limit)
    for form in ("auto", "w8a16"):
        assert env["auto_large_rows"](False, "fp8", form, g) == threshold
        assert env["_resolve"]("auto", threshold, False, "fp8", form, g) == "decode"
        assert env["_resolve"]("auto", threshold + 1, False, "fp8", form, g) == "auto"
        assert env["_resolve"]("prefill", 4096, False, "fp8", form, g) == "prefill"
    assert env["auto_large_rows"](False, "fp8", "w8a8", g) == (768 if sm == 121 else 512)
    assert env["auto_large_rows"](True, "fp8", "auto", g) == (768 if sm == 121 else 512)
    assert env["auto_large_rows"](True, "fp8", "w8a16", g) == (1024 if sm == 121 else 2048)
    assert env["auto_large_rows"](False, "bf16", "auto", g) == 2048


@pytest.mark.parametrize("sm,expected", [(120, 2048), (121, 4096)])
def test_unmeasured_bf16_fp8_defaults(sm, expected):
    _, env = policy(sm)
    # Qwen TP1, unknown TP widths and unknown full widths do not inherit a neighbor.
    for g in (fp8_geometry(2560, 640, 640), fp8_geometry(4096, 2048, 768),
              fp8_geometry(4096, 4096, 1024)):
        assert env["auto_large_rows"](False, "fp8", "auto", g) == expected
    g = fp8_geometry(4096, 1536, 768)  # Wrong activation contract for Step's entry.
    assert env["auto_large_rows"](False, "fp8", "auto", g) == expected
    g.activations = "a4"
    assert env["auto_large_rows"](False, "fp8", "auto", g) == 2048
