"""CPU-only CLI contracts; importing the benchmark must not import CUDA libraries."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_cuteafd_projection_routes.py"
_SPEC = importlib.util.spec_from_file_location("projection_route_benchmark", _PATH)
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)


def test_compensation_pair_accepts_production_shapes_and_wide_rows():
    args = _BENCH.parse_args(["--recipes", "compensation", "--shapes", "7168x6144,6144x6144",
                              "--rows", "1,16,4096"])
    assert args.recipes == "compensation"
    assert args.shapes == [(7168, 6144), (6144, 6144)]
    assert args.rows == [1, 16, 4096]


def test_existing_route_defaults_are_preserved():
    args = _BENCH.parse_args([])
    assert args.recipes == "routes"
    assert args.shapes == [(4096, 4096)]
    assert args.rows == [1, 8, 9, 10, 16, 32, 64]
    assert args.samples == 20


def test_compensation_arms_use_identical_tiles_and_runtime_rows(monkeypatch):
    calls = []

    def tma(n, k, **kwargs):
        calls.append((n, k, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "cutlass", SimpleNamespace(BFloat16="bf16"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        SkinnyBf16Gemv=None, TmaBf16Projection=tma, skinny_config=None))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._common", SimpleNamespace(
        Operand=lambda *a: a, Scalar=lambda *a: a,
        compile_program=lambda op, **kwargs: kwargs))
    direct = _BENCH._program(7168, 6144, "uncompensated")
    compensated = _BENCH._program(7168, 6144, "compensated")
    assert calls == [(7168, 6144, {"tiles_by_rows": (), "compensated": False}),
                     (7168, 6144, {"tiles_by_rows": (), "compensated": True})]
    assert direct["scalars"] == compensated["scalars"] == (("rows",),)
    assert direct["key"] != compensated["key"]


def test_projection_compensation_is_opt_in_and_default_key_is_unchanged(monkeypatch):
    source = _PATH.parents[1] / "b12x/integration/cuteafd/pointwise.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "compile_bf16_projection_aot")
    calls = []

    class Projection:
        exact_row_cap = 2**31 - 1

        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))
            self.kwargs = kwargs

        def key(self):
            return tuple(sorted(self.kwargs.items()))

    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        COMPENSATED_TILES_BY_ROWS={}, RoutedBf16Projection=Projection,
        TmaBf16Projection=Projection, wide_prefill_min_rows=lambda *args: 4097))
    namespace = dict(torch=SimpleNamespace(bfloat16="bf16", float32="f32"),
                     BFloat16="bf16", Float32="f32", Operand=lambda *a: a,
                     Scalar=lambda *a: a, compile_program=lambda op, **kwargs: kwargs)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    build = namespace["compile_bf16_projection_aot"]
    default = build(width=6144, outputs=7168)
    explicit = build(width=6144, outputs=7168, compensated=False)
    compensated = build(width=6144, outputs=7168, compensated=True)
    assert default == explicit
    assert calls[:2] == [(7168, 6144, {"out_dtype": "bf16", "max_skinny_rows": 0,
                                     "wide": False, "wide_min_rows": None})] * 2
    assert calls[2] == (7168, 6144, {"out_dtype": "bf16", "tiles_by_rows": (), "compensated": True})
    assert default["geometry"] == dict(width=6144, outputs=7168, output_dtype="bf16", skinny_rows=0,
                                       exact_row_cap=2**31 - 1)
    assert compensated["geometry"]["compensated"] is True
    assert default["key"] != compensated["key"]
    with pytest.raises(ValueError, match="boolean"):
        build(width=6144, outputs=7168, compensated="yes")


def _compensated_build(monkeypatch, table):
    """``compile_bf16_projection_aot`` with stub projections; returns (build, calls)."""
    source = _PATH.parents[1] / "b12x/integration/cuteafd/pointwise.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "compile_bf16_projection_aot")
    calls = []

    class Projection:
        exact_row_cap = 2**31 - 1

        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))
            self.kwargs = kwargs

        def key(self):
            return tuple(sorted(self.kwargs.items()))

    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        COMPENSATED_TILES_BY_ROWS=table, RoutedBf16Projection=Projection,
        TmaBf16Projection=Projection, wide_prefill_min_rows=lambda *args: 4097))
    namespace = dict(torch=SimpleNamespace(bfloat16="bf16", float32="f32"),
                     BFloat16="bf16", Float32="f32", Operand=lambda *a: a,
                     Scalar=lambda *a: a, compile_program=lambda op, **kwargs: kwargs)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["compile_bf16_projection_aot"], calls


def test_compensated_route_selects_its_measured_row_tiles(monkeypatch):
    """A shape with a ``COMPENSATED_TILES_BY_ROWS`` entry carries it; others stay fixed."""
    table = ((16, 1, 4, 16, 32, 12), (4096, 2, 4, 64, 128, 3))
    build, calls = _compensated_build(monkeypatch, {(6144, 4096): table})
    program = build(width=4096, outputs=6144, compensated=True)
    assert calls == [(6144, 4096, {"out_dtype": "bf16", "tiles_by_rows": table, "compensated": True})]
    assert program["geometry"]["compensated"] is True
    assert program["geometry"]["tiles_by_rows"] == table
    plain = build(width=6144, outputs=7168, compensated=True)
    assert calls[-1] == (7168, 6144, {"out_dtype": "bf16", "tiles_by_rows": (), "compensated": True})
    assert "tiles_by_rows" not in plain["geometry"]


def test_trellis_accepts_config_and_continuous_pairing_without_cuda():
    args = _BENCH.parse_args(["--recipes", "trellis", "--shapes", "6144x16384",
                              "--rows", "2752,4096", "--trellis-config", "shapes.json",
                              "--trellis-paired", "--trellis-steady-ms", "200"])
    assert args.shapes == [(6144, 16384)]
    assert args.rows == [2752, 4096]
    assert args.trellis_config == Path("shapes.json")
    assert args.trellis_paired
    assert args.trellis_steady_ms == 200


@pytest.mark.parametrize("samples,flagged", [
    ([100, 100, 100, 100], False),
    ([100, 101, 100, 101], False),
    ([100, 120, 100, 120, 100], True),
    ([100], True),
])
def test_trellis_order_audit_flags_bias_or_missing_reverse_order(samples, flagged):
    audit = _BENCH._alternating_order_audit(samples)
    assert audit["flagged"] is flagged
    assert audit["mad_us"] >= 0
    assert (audit["delta_us"] is None) == (len(samples) == 1)


@pytest.mark.parametrize("argv", [
    ["--recipes", "trellis", "--rows", "4097"],
    ["--recipes", "trellis", "--trellis-steady-ms", "199"],
    ["--recipes", "trellis", "--trellis-steady-ms", "nan"],
    ["--recipes", "trellis", "--trellis-steady-ms", "inf"],
    ["--recipes", "trellis", "--trellis-arms", "ring3"],
    ["--rows", "4096"],
    ["--recipes", "compensation", "--rows", "4097"],
    ["--recipes", "compensation", "--rows", "0"],
    ["--rows", ""],
    ["--samples", "0"],
    ["--shapes", "7168"],
    ["--shapes", "7168x6144x8"],
    ["--shapes", "7168x6145"],
    ["--shapes", "0x6144"],
    ["--shapes", "nonsense"],
])
def test_invalid_geometry_is_rejected_before_cuda_import(argv):
    with pytest.raises(SystemExit) as error:
        _BENCH.parse_args(argv)
    assert error.value.code == 2


def _native_threshold_bench():
    path = _PATH.with_name("kimi_k6_dense.py")
    spec = importlib.util.spec_from_file_location("native_threshold_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text", ["narrow=9", "decode=17", "wide2=33", "wide3=49",
                                  "folded=17", "prefill=131", "prefill=0",
                                  "narrow=8,decode=4", "splits=4", "stages=3"])
def test_native_thresholds_refuse_unsafe_knobs(text):
    with pytest.raises(ValueError):
        _native_threshold_bench().threshold_config(text)


def test_reference_encoder_build_requires_explicit_official_source(monkeypatch, tmp_path):
    bench = _native_threshold_bench()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--encoder-build-only"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


@pytest.mark.parametrize("text", ["decode=8", "wide2=24"])
def test_grouped_threshold_builder_refuses_partial_plane_overlap(text, tmp_path):
    bench = _native_threshold_bench()
    with pytest.raises(ValueError, match="next projection's partial plane"):
        bench.threshold_config(text, groups=4)
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    with pytest.raises(ValueError, match="next projection's partial plane"):
        bench.threshold_variant(source, tmp_path / "bad.py", bench.threshold_config(text), groups=4)


@pytest.mark.parametrize("decode,wide2", [(8, 32), (16, 24)])
def test_native_grouped_constructor_refuses_partial_plane_overlap(decode, wide2):
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    tree = ast.parse(source.read_text())
    check = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "validate_grouped_route_thresholds")
    core = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "NativeTrellisLinear")
    init = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    namespace = {"_GROUPED_DECODE_ROWS": decode, "_GROUPED_WIDE2_ROWS": wide2}
    exec(compile(ast.Module(body=[check, init], type_ignores=[]), str(source), "exec"), namespace)
    with pytest.raises(AssertionError, match="next projection's partial plane"):
        namespace["__init__"](SimpleNamespace(), {"groups": 4})


def test_standalone_threshold_variant_keeps_grouped_routes_frozen(tmp_path):
    bench = _native_threshold_bench()
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    variant = bench.threshold_variant(source, tmp_path / "standalone.py", bench.threshold_config("decode=8"))
    def grouped(text):
        core = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
                    and n.name == "NativeTrellisLinear")
        return ast.dump(next(n for n in core.body if isinstance(n, ast.FunctionDef)
                             and n.name == "_generic_grouped"))
    assert grouped(variant.read_text()) == grouped(source.read_text())
    assert "_GROUPED_DECODE_ROWS = 16" in variant.read_text()


def test_reference_fixtures_require_prebuilt_official_encoder(monkeypatch, tmp_path):
    bench = _native_threshold_bench()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--encode-fixtures"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_kimi_decision_requires_encoded_fixtures_and_profile_before_cuda(monkeypatch, tmp_path):
    bench = _native_threshold_bench()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--kimi-decision"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_kimi_decision_keeps_independent_projection_and_native_remainder_shapes():
    source = _PATH.with_name("kimi_k6_dense.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "benchmark_kimi_decision")
    assert '("kda", 6144, 4, range(4), 24752)' in ast.get_source_segment(source, node)
    assert '("gate_up", 3072, 1, range(4, 6), 6144)' in ast.get_source_segment(source, node)
    assert 'outputs=176' in ast.get_source_segment(source, node)
    assert '"KDA and shared gate/up only, not a full-core speed estimate"' in source


def _kimi_profile_fixture():
    def entry(n, k, count, symbols):
        return dict(device=0, lane="target", geometry=dict(n=n, k=k), phases=dict(decode=dict(
            count=count, unit_count=255, unknown_rows=0, rows={"5": count}, symbols=symbols)))
    catalog = [entry(24752, 7168, 5037, ["kda"]),
               entry(6144, 7168, 8468, ["shared", "mla"]),
               entry(7168, 3072, 6716, ["down"]), entry(9216, 1536, 1752, ["qb"])]
    inventory = dict(phase="decode", unit="emitted-token", unit_count=255, kernels=[
        dict(device=0, name="shared", count=3000, rows={"5": 3000}),
        dict(device=0, name="shared", count=3716, rows={"5": 3716}),
        dict(device=0, name="mla", count=1752, rows={"5": 1752}),
        dict(device=1, name="shared", count=999, rows={"5": 999})])
    return catalog, inventory


def test_kimi_profile_accounting_separates_shared_gate_up_from_single_mla_gate():
    catalog, inventory = _kimi_profile_fixture()
    accounting = _native_threshold_bench().kimi_profile_accounting(catalog, inventory)
    assert accounting["calls_per_emitted_token"] == dict(kda=5037 / 255, gate_up=6716 / 255)
    assert accounting["symbols"] == {"packed shared gate/up": "shared", "single MLA h48 gate": "mla"}
    assert accounting["shared_gate_up_count"] + accounting["mla_gate_count"] == accounting["mixed_count"]


@pytest.mark.parametrize("bad", ["units", "missing_symbol", "count", "histogram", "unknown", "duplicate"])
def test_kimi_profile_accounting_refuses_ambiguous_or_incomplete_data(bad):
    catalog, inventory = _kimi_profile_fixture()
    if bad == "units":
        inventory["unit_count"] = 256
    elif bad == "missing_symbol":
        inventory["kernels"] = inventory["kernels"][:2]
    elif bad == "count":
        inventory["kernels"][0]["count"] -= 1
    elif bad == "histogram":
        inventory["kernels"][0]["rows"] = {"4": 3000}
    elif bad == "unknown":
        inventory["kernels"][0]["row_unknown"] = 1
    elif bad == "duplicate":
        catalog.append(catalog[0])
    with pytest.raises(ValueError):
        _native_threshold_bench().kimi_profile_accounting(catalog, inventory)


def test_kimi_decision_requires_symbol_inventory_before_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path), "--kimi-decision",
                                     "--fixture-dir", str(tmp_path), "--profile-catalog", "catalog.json"])
    with pytest.raises(SystemExit) as error:
        _native_threshold_bench().main()
    assert error.value.code == 2


def _full_core_config():
    return dict(hidden_size=7168,num_hidden_layers=93,num_attention_heads=96,
        linear_attn_config=dict(kda_layers=list(range(69)),full_attn_layers=list(range(24)),
                                num_heads=96,head_dim=128),
        num_shared_experts=2,moe_intermediate_size=3072,routed_expert_hidden_size=3584,
        q_lora_rank=1536,kv_lora_rank=512,qk_rope_head_dim=64,qk_nope_head_dim=128,
        num_experts=896,intermediate_size=33792,vocab_size=163840)


def test_full_core_config_keeps_physical_counts_and_independent_padded_rank_shapes():
    bench = _native_threshold_bench()
    operations = bench.kimi_full_core_operations(_full_core_config())
    roles = {o["role"]:o for o in operations}
    assert roles["kda"]["count"] == [69,69]
    assert roles["attn_out"]["count"] == [93,93]
    assert roles["latent_down"]["count"] == [46,46]
    assert roles["mla_gate"]["count"] == [24,24]
    assert roles["head"]["count"] == [0,1]
    assert roles["layer0_down"]["native"] and roles["layer0_down"]["output"] == "bf16"
    shapes = bench.full_core_fixture_shapes(operations)
    rank = [s for s in shapes if s["role"] == "mla_rank"]
    assert [s["n"] for s in rank] == [1536,640]
    assert len([s for s in shapes if s["role"] == "kda"]) == 4
    assert next(s for s in shapes if s["role"] == "shared_down")["side"] == 2


def test_full_core_aggregation_never_sums_concurrent_gpus_and_uses_actual_head_rows():
    bench = _native_threshold_bench()
    operations = bench.kimi_full_core_operations(_full_core_config())
    records = [dict(role=o["role"],rows=r,total_call_us={"native":1} if o["native"] else
                    dict(k6=1,k8=2,bf16=3)) for o in operations
               for r in ([1,5] if o["head"] else [5,2752,2688,2240])]
    result = bench.aggregate_kimi_full_core(operations,records,
                dict(verify_steps_per_emitted_token=73/255),[5,2752,2688,2240])
    decode = result["decode"][0]["local_dense_us_per_verify_step"]["k6"]
    assert decode[1]-decode[0] == 1
    assert [p["head_calls"] for p in result["prefill"]] == [1,8]
    assert [p["local_dense_us"]["k6"][1]-p["local_dense_us"]["k6"][0]
            for p in result["prefill"]] == [1,8]


def test_qwen_full_core_keeps_whole_owner_counts_and_independent_members():
    bench = _native_threshold_bench()
    config = dict(hidden_size=2560, num_hidden_layers=2,
                  layer_types=["linear_attention", "full_attention"], ple_layer_ids=[1])
    tensors = {}
    def tensor(name, n, k):
        tensors[name] = dict(dtype="BF16", shape=[n, k])
    for layer in range(2):
        p = f"model.language_model.layers.{layer}."
        for site in ("attn", "mlp"):
            hc = p+site+"_hyper_connection."
            tensor(hc+"input_mix_weight_down.weight", 320, 10240)
            tensor(hc+"block_inject_weight.weight", 4, 10240)
            tensor(hc+"input_mix_weight_up.weight", 10240, 320)
        names = (("in_proj_qkv", 10240), ("in_proj_z", 6144), ("in_proj_b", 48),
                 ("in_proj_a", 48)) if layer == 0 else (("q_proj", 12288),
                 ("k_proj", 512), ("v_proj", 512), ("indexer.index_qk_proj", 640))
        a = p+("linear_attn." if layer == 0 else "self_attn.")
        for name, n in names:
            tensor(a+name+".weight", n, 2560)
        tensor(a+("out_proj" if layer == 0 else "o_proj")+".weight", 2560, 6144)
        tensor(p+"mlp.gate.weight", 512, 2560)
        for name in ("gate", "up"):
            tensor(p+"mlp.shared_expert."+name+"_proj.weight", 640, 2560)
        tensor(p+"mlp.shared_expert_gate.weight", 1, 2560)
        tensor(p+"mlp.shared_expert.down_proj.weight", 2560, 640)
    for name, n in (("key", 10240), ("value", 2560)):
        tensor("model.language_model.layers.0.ple."+name+"_proj.weight", n, 2560)
    tensor("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", 320, 10240)
    tensor("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", 10240, 320)
    tensor("lm_head.weight", 248320, 2560)
    operations = bench.qwen_full_core_operations(config, tensors, [0, 1])
    roles = {op["role"]: op for op in operations}
    assert roles["layer0.gdn_in"]["packed_n"] == 16480
    assert [m["n"] for m in roles["layer0.gdn_in"]["members"]] == [10240, 6144, 48, 48]
    assert roles["layer1.attn_in"]["packed_n"] == 13952
    assert roles["layer0.shared_in"]["native_padding_rows"] == 15
    assert roles["layer0.attn.hc_down_inject"]["packed_n"] == 324
    assert roles["layer0.gdn_out"]["count"] == [1, 0]
    assert roles["layer1.attn_out"]["count"] == [0, 1]
    assert roles["head"]["count"] == [1, 0]
    assert roles["head"]["fp8_scale_layout"] == "row_k128_f32"
    assert roles["layer0.gdn_in"]["fp8_scale_layout"] == "block128x128_f32"
    assert roles["layer0.ple_kv"]["fp8_scale_layout"] is None
    assert len(operations) == 22
    shapes = bench.qwen_full_core_fixture_shapes(operations)
    assert {s["role"] for s in shapes} == {
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight", "self_attn.q_proj.weight",
        "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight", "lm_head.weight",
    }
    assert not any("index_qk" in s["role"] for s in shapes)
    assert [m["hypothetical_trellis"] for m in roles["layer0.gdn_in"]["members"]] == [True, True, False, False]
    assert [m["hypothetical_trellis"] for m in roles["layer1.attn_in"]["members"]] == [True, True, True, False]
    assert [m["hypothetical_trellis"] for m in roles["layer0.shared_in"]["members"]] == [True, True, False]
    assert not any("in_proj_a" in s["role"] or "shared_expert_gate" in s["role"] for s in shapes)
    assert all(s["side"] == 1 for s in shapes)
    with pytest.raises(ValueError, match="owner map"):
        bench.qwen_full_core_operations(config, tensors, [0])


def test_qwen_projection_specs_keep_actual_prefill_and_output_contracts():
    bench = _native_threshold_bench()
    def op(role, n, k=2560, scale=None):
        return dict(role=role, k=k, packed_n=n, fp8_scale_layout=scale)
    gdn = op("layer0.gdn_in", 16480, scale="block128x128_f32")
    decode = bench.qwen_full_core_projection_spec(gdn)
    prefill = bench.qwen_full_core_projection_spec(gdn, prefill=True)
    assert decode["bf16_factory"] == "glm_projection"
    assert decode["fp8_factory"] == "fp8_projection"
    assert prefill["fp8_factory"] == "prefill_w8a16"
    assert prefill["fp8_prefill_w8a8"] is False
    attn = op("layer3.attn_in", 13952, scale="block128x128_f32")
    assert bench.qwen_full_core_projection_spec(attn, prefill=True)["fp8_factory"] == "fp8_projection"
    hc = op("layer0.attn.hc_up", 10240, k=320)
    assert bench.qwen_full_core_projection_spec(hc)["bf16_factory"] == "qwen_projection"
    assert bench.qwen_full_core_projection_spec(hc)["fp8_factory"] is None
    router = op("layer0.router", 512)
    assert bench.qwen_full_core_projection_spec(router)["output"] == "f32"
    assert bench.qwen_full_core_projection_spec(router, prefill=True)["bf16_factory"] == "router_prefill"
    head = op("head", 248320, scale="row_k128_f32")
    assert bench.qwen_full_core_projection_spec(head)["head_capacity"] == 64
    assert bench.qwen_full_core_projection_spec(head)["bf16_factory"] == "native_head"


def test_qwen_unique_operations_sum_only_identical_role_geometries():
    bench = _native_threshold_bench()
    base = dict(k=2560, packed_n=6144, fp8_scale_layout="block128x128_f32",
                members=[dict(name="first", k=2560, n=6144)])
    ops = [dict(base, role="layer0.gdn_out", count=[1, 0]),
           dict(base, role="layer1.gdn_out", count=[0, 1]),
           dict(base, role="layer2.attn_out", count=[0, 1])]
    merged = bench.qwen_full_core_unique_operations(ops)
    assert len(merged) == 2
    assert merged[0]["role"] == "gdn_out" and merged[0]["count"] == [1, 1]
    assert ops[0]["count"] == [1, 0]


def test_qwen_projection_program_uses_native_factories_and_i64_scratch(monkeypatch):
    bench = _native_threshold_bench()
    calls = []
    class Launch:
        def __init__(self, kind, *args, **kwargs):
            calls.append((kind, args, kwargs))
        def key(self):
            return ("projection",)
    def factory(kind):
        return lambda *args, **kwargs: Launch(kind, *args, **kwargs)
    for module, names in {
        "b12x.integration.cuteafd._fp8_weights": {"Fp8Projection": factory("fp8")},
        "b12x.integration.cuteafd._glm_kernels": {"glm_projection": factory("glm")},
        "b12x.integration.cuteafd.qwen4": {"_projection": factory("qwen")},
        "b12x.integration.cuteafd.qwen4_gdn": {"_PrefillW8A16": factory("w8a16")},
        "b12x.integration.cuteafd.glmf": {"_HeadFp8": factory("head")},
        "b12x.moe._shared.v41_router": {"_V41RouterScores": factory("router")},
    }.items():
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**names))
    monkeypatch.setitem(sys.modules, "cutlass", SimpleNamespace(BFloat16="bf16", Float32="f32"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="f32", float8_e4m3fn="fp8"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._common", SimpleNamespace(
        Operand=lambda *a, **kw: (a, kw), Scalar=lambda *a: a,
        compile_program=lambda launch, **kwargs: kwargs))
    op = dict(role="gdn_in", k=2560, packed_n=16480, fp8_scale_layout="block128x128_f32")
    result = bench.qwen_full_core_projection_program(op, prefill=True, fp8=True)
    assert calls[-1] == ("w8a16", (16480, 2560), {})
    assert result["scalars"] == (("rows",), ("fp8_rows",), ("qscratch", "int64"))
    assert result["operands"][2][0][2] == "[129,20]"
    assert [operand[0][0] for operand in result["operands"]] == ["x", "w_fp8", "scale", "out"]
    head = dict(op, role="head", packed_n=248320, fp8_scale_layout="row_k128_f32")
    result = bench.qwen_full_core_projection_program(head, fp8=True)
    assert calls[-1] == ("head", (248320, 2560), {"max_rows": 64})
    assert [operand[0][0] for operand in result["operands"]] == ["x", "w_fp8", "scale", "logits"]
    assert result["scalars"] == (("rows",),)
    bench.qwen_full_core_projection_program(op)
    assert calls[-1] == ("glm", (16480, 2560), {})
    router = dict(op, role="router", packed_n=512, fp8_scale_layout=None)
    result = bench.qwen_full_core_projection_program(router, prefill=True)
    assert calls[-1] == ("router", (512,), {"hidden": 2560})
    assert [operand[0][0] for operand in result["operands"]] == ["x", "w", "logits"]
    assert result["scalars"] == (("live_rows",),)


def test_qwen_core_aggregate_separates_owners_and_labels_token_projection():
    bench = _native_threshold_bench()
    ops = [dict(role="gdn", count=[18, 18]), dict(role="head", count=[1, 0])]
    records = [dict(role="gdn", rows=2, total_call_us=dict(k6=10, k8=12, bf16=20, fp8=9)),
               dict(role="head", rows=2, total_call_us=dict(k6=2, k8=3, bf16=4, fp8=1))]
    result = bench.aggregate_qwen_full_core(ops, records, [2, 16])
    assert len(result["decode"]) == 1
    row = result["decode"][0]
    assert row["local_dense_us_per_verify_step"]["k6"] == [182, 180]
    assert row["kimi_rate_normalized_local_dense_us_per_emitted_token"]["k6"] == pytest.approx([182*73/255, 180*73/255])
    assert "not measured Qwen speculation" in result["normalization"]
    assert result["prefill"] == []


def test_qwen_core_aggregate_native_cost_is_charged_once_per_owner():
    bench = _native_threshold_bench()
    ops = [dict(role="native", count=[24, 24]), dict(role="head", count=[1, 0])]
    records = [dict(role="native", rows=4096, total_call_us=dict(native=3))]
    records.append(dict(role="head", rows=1, total_call_us=dict(k6=2, k8=3, bf16=4, fp8=1)))
    result = bench.aggregate_qwen_full_core(ops, records, [1])
    assert result["decode"] == []
    prefill = result["prefill"][0]
    assert prefill["tokens"] == 8192
    assert prefill["body_row_histogram"] == {4096: 2}
    assert prefill["head_calls"] == 2
    assert prefill["local_dense_us"]["bf16"] == [152, 144]
    assert prefill["local_dense_us"]["fp8"] == [146, 144]
    assert result["prefill"][1]["body_row_histogram"] == {4096: 16}
    assert result["prefill"][1]["head_calls"] == 16


def test_qwen_memory_replaces_one_arena_and_tail_without_double_charging():
    bench = _native_threshold_bench()
    ops = [dict(role="head", k=128, packed_n=128, count=[1, 0],
                fp8_scale_layout="row_k128_f32", hypothetical_trellis=True,
                native_padding_rows=0, members=[dict(n=128, hypothetical_trellis=True)]),
           dict(role="sensitive", k=128, packed_n=1, count=[1, 1],
                fp8_scale_layout=None, hypothetical_trellis=False, native_padding_rows=0,
                members=[dict(n=1, hypothetical_trellis=False)])]
    plan = dict(family="qwen4", fits=True, memory_layout=dict(devices=[
        dict(kind="rtx", index=gpu, items=[
            dict(category="weights", group="weight arena", bytes=100000),
            dict(category="weights", group="allocation granularity",
                 format=f"qwen4/target/rank{gpu}.tail (test)", bytes=(2 << 20)-100000)])
        for gpu in (0, 1)]))
    result = bench.qwen_full_core_memory_forecast(ops, plan)
    assert result["arms"]["bf16"][0]["arena_delta_bytes"] == 32768 - (16384 + 512)
    assert result["arms"]["k6"][0]["arena_delta_bytes"] == 12288 + 256 + 256 - (16384 + 512)
    assert result["arms"]["k8"][0]["arena_delta_bytes"] == 0
    for owners in result["arms"].values():
        assert owners[1]["arena_delta_bytes"] == 0
        for owner in owners:
            assert owner["charged_bytes"] == 2 << 20
            assert owner["charged_delta_bytes"] == 0
    plan["memory_layout"]["devices"][0]["items"][1]["bytes"] += 1
    with pytest.raises(ValueError, match="allocation-tail policy"):
        bench.qwen_full_core_memory_forecast(ops, plan)


def _glm_core_headers():
    config = dict(hidden_size=6144, num_hidden_layers=4, num_attention_heads=64,
        q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
        qk_rope_head_dim=64, v_head_dim=256, first_k_dense_replace=1,
        indexer_types=["full", "shared", "full", "shared"])
    tensors = {"lm_head.weight": dict(dtype="BF16", shape=[154880, 6144])}
    for layer in range(5):
        p = f"model.layers.{layer}."
        for name, shape in {
            "q_a_proj": [2048, 6144], "kv_a_proj_with_mqa": [576, 6144],
            "q_b_proj": [16384, 2048], "kv_b_proj": [28672, 512],
            "o_proj": [6144, 16384], "indexer.wq_b": [4096, 2048],
            "indexer.wk": [128, 6144], "indexer.weights_proj": [32, 6144],
        }.items():
            tensors[p+"self_attn."+name+".weight"] = dict(dtype="BF16", shape=shape)
        m = p+("mlp.shared_experts." if layer else "mlp.")
        inter = 2048 if layer else 12288
        for name in ("gate", "up"):
            tensors[m+name+"_proj.weight"] = dict(dtype="BF16", shape=[inter, 6144])
        tensors[m+"down_proj.weight"] = dict(dtype="BF16", shape=[6144, inter])
        tensors[p+"mlp.gate.weight"] = dict(dtype="BF16", shape=[256, 6144])
    return config, tensors


def test_glm_context_core_keeps_head_shares_and_independent_rank_members():
    bench = _native_threshold_bench()
    config, tensors = _glm_core_headers()
    operations = bench.glm_full_core_operations(config, tensors, onboard_layers=1)
    roles = {op["role"]: op for op in operations}
    assert roles["layer0.rank_in"]["packed_n"] == 2624
    assert [(m["n"], m["stored_n"]) for m in roles["layer0.rank_in"]["members"]] == [(2048, 2048), (576, 640)]
    assert roles["layer0.query_up"]["packed_n"] == 8192
    assert roles["layer0.attn_out"]["k"] == 8192
    assert roles["layer0.attn_out"]["reduction_count"] == [1, 1]
    assert roles["layer0.dense_in"]["packed_n"] == 12288
    assert [m["n"] for m in roles["layer0.dense_in"]["members"]] == [6144, 6144]
    assert roles["layer1.shared_in"]["packed_n"] == 2048
    assert roles["layer1.shared_down"]["k"] == 1024
    assert roles["layer1.shared_down"]["reduction_count"] == [1, 0]
    assert roles["layer0.absorb"]["batch"] == 32
    assert (roles["layer0.absorb"]["packed_n"], roles["layer0.absorb"]["k"]) == (512, 192)
    assert (roles["layer0.values"]["packed_n"], roles["layer0.values"]["k"]) == (256, 512)
    assert roles["layer0.index_kw"]["packed_n"] == 160
    assert not roles["layer0.index_query"]["hypothetical_trellis"]
    assert roles["layer0.index_query"]["fp8_scale_layout"] == "block128x128_f32"
    assert roles["layer1.router"]["count"] == [1, 1]
    assert roles["layer2.router"]["count"] == [1, 0]
    assert roles["head"]["count"] == [1, 0]
    assert roles["head"]["output"] == "f32"
    assert not any(role.startswith("layer4.") for role in roles)
    assert "layer1.index_query" not in roles
    assert all(op["prefill_w8a8"] for op in operations)
    assert all(op["count"] == [1, 1] for op in operations if not op["role"].endswith("router") and op["role"] != "head")


def test_glm_unique_calls_preserve_peer_router_and_shared_reduction_counts():
    bench = _native_threshold_bench()
    config, tensors = _glm_core_headers()
    operations = bench.glm_full_core_operations(config, tensors, onboard_layers=1)
    unique = {op["role"]: op for op in bench.glm_full_core_unique_operations(operations)}
    assert unique["router"]["count"] == [3, 1]
    assert unique["attn_out"]["count"] == [4, 4]
    assert unique["attn_out"]["reduction_count"] == [4, 4]
    assert unique["shared_down"]["count"] == [3, 3]
    assert unique["shared_down"]["reduction_count"] == [3, 0]
    assert unique["index_query"]["count"] == [2, 2]
    fixtures = bench.glm_full_core_fixture_shapes(operations)
    assert len(fixtures) == 11
    rank = [f for f in fixtures if f["role"] == "rank_in"]
    assert [(f["n"], f["logical_n"]) for f in rank] == [(2048, 2048), (640, 576)]
    assert all(f["side"] == 2 for f in fixtures if f["role"].endswith("down") or f["role"] == "attn_out")
    assert not any("index" in f["role"] for f in fixtures)


def test_glm_prefill_uses_actual_span_and_final_tail():
    bench = _native_threshold_bench()
    assert bench.glm_full_core_prefill_schedule(8192) == dict(
        body_row_histogram={2752: 2, 2688: 1}, head_calls=1, head_rows=1)
    assert bench.glm_full_core_prefill_schedule(65536) == dict(
        body_row_histogram={2752: 23, 2240: 1}, head_calls=8, head_rows=1)
    with pytest.raises(ValueError, match="rebalance"):
        bench.glm_full_core_prefill_schedule(1024)


def test_glm_factories_keep_prefill_kmajor_scales_and_context_query_pitch(monkeypatch):
    from dataclasses import make_dataclass
    bench = _native_threshold_bench()
    geometry = make_dataclass("Geometry", ["heads", "kv_lora_rank", "qk_rope_dim"])(64, 512, 64)
    calls = []
    class Launch:
        def key(self):
            return ("native_glm",)
    def projection(*args, **kwargs):
        calls.append((args, kwargs))
        return Launch()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="f32", float8_e4m3fn="fp8"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._common", SimpleNamespace(
        GLM53=geometry, Operand=lambda *a, **kw: (a, kw), Scalar=lambda *a: a,
        compile_program=lambda launch, **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._glm_kernels", SimpleNamespace(glm_projection=projection))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd.glm_attention", SimpleNamespace(
        _w8_projection=projection, compile_glm_attention_stage_aot=lambda *a, **kw: (a, kw)))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd.glm_ffn", SimpleNamespace(
        compile_glm_router_scores_aot=lambda: "router_decode",
        compile_glm_prefill_router_scores_aot=lambda: "router_prefill"))
    op = dict(role="rank_in", k=6144, packed_n=2624, batch=1,
              fp8_scale_layout="block128x128_f32_decode_kmajor_row_k128_f32_prefill")
    result = bench.glm_full_core_projection_program(op, prefill=True, fp8=True)
    assert calls[-1] == ((2624, 6144, 4096), dict(row_scales=True))
    assert result["operands"][2][0][2] == "[48,2624]"
    assert result["scalars"] == (("rows",), ("fp8_rows",), ("qscratch", "int64"))
    result = bench.glm_full_core_projection_program(op, fp8=True)
    assert result["operands"][2][0][2] == "[21,48]"
    result = bench.glm_full_core_projection_program(dict(op, role="absorb", batch=32), prefill=True, fp8=True)
    assert result[0][0].heads == 32
    assert result[1] == dict(stage="absorb", max_rows=4096, fp8_only="prefill", query_row=64*576)
    result = bench.glm_full_core_projection_program(dict(op, role="values", batch=32))
    assert result[1]["query_row"] is None and result[1]["fp8_only"] is None
    assert bench.glm_full_core_projection_program(dict(op, role="router"), prefill=True) == "router_prefill"
    with pytest.raises(ValueError, match="native DSO"):
        bench.glm_full_core_projection_program(dict(op, role="head"))


def test_glm_aggregate_charges_shared_meet_only_on_rank_zero():
    bench = _native_threshold_bench()
    ops = [dict(role="shared_down", split="input", count=[3, 3], reduction_count=[3, 0]),
           dict(role="head", split="whole", count=[1, 0], reduction_count=[0, 0]),
           dict(role="router", split="whole", count=[3, 1], reduction_count=[0, 0])]
    records = []
    for row in (1, 2, 2240, 2688, 2752):
        records.extend([
            dict(role="shared_down", rows=row, phase="producer", total_call_us=dict(bf16=1, fp8=2, k6=3, k8=4)),
            dict(role="shared_down", rows=row, phase="reduce0", total_call_us=dict(k6=5, k8=6)),
            dict(role="shared_down", rows=row, phase="reduce1", total_call_us=dict(k6=7, k8=8)),
            dict(role="router", rows=row, phase="producer", total_call_us=dict(native=10)),
            dict(role="head", rows=row, phase="producer", total_call_us=dict(bf16=11, fp8=11, k6=12, k8=13))])
    result = bench.aggregate_glm_full_core(ops, records, [2, 64])
    assert len(result["decode"]) == 1
    decode = result["decode"][0]
    assert decode["local_dense_us_per_verify_step"] == dict(bf16=[44, 13], fp8=[47, 16], k6=[66, 19], k8=[73, 22])
    assert decode["kimi_rate_normalized_local_dense_us_per_emitted_token"]["k6"] == [66*73/255, 19*73/255]
    assert [r["local_dense_us"]["k6"] for r in result["prefill"]] == [[174, 57], [1392, 456]]
    assert "fused" in result["scope"] and "not measured GLM" in result["normalization"]


def test_glm_header_fixture_selection_uses_explicit_onboard_prefix(tmp_path):
    import json
    bench = _native_threshold_bench()
    config, tensors = _glm_core_headers()
    path = tmp_path / "headers.json"
    path.write_text(json.dumps(dict(glm=dict(config=config, tensors=tensors))))
    ops = bench.glm_full_core_from_headers(path, 1)
    assert len(bench.glm_full_core_fixture_shapes(ops)) == 11
    assert sum(o["count"][1] for o in ops if o["role"].endswith("router")) == 1


@pytest.mark.parametrize("extra", [[], ["--onboard-layers", "9"],
                                   ["--onboard-layers", "9", "--qwen-core", "headers.json"]])
def test_glm_cli_rejects_incomplete_measurements_before_cuda(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path),
                                     "--glm-core", "headers.json", *extra])
    with pytest.raises(SystemExit) as error:
        _native_threshold_bench().main()
    assert error.value.code == 2


def test_glm_output_checks_skip_per_head_rope_and_cover_all_members():
    import numpy as np

    bench = _native_threshold_bench()
    output = np.full((3, 64*576), np.nan, dtype=np.float32)
    targets = bench.glm_full_core_output_targets("absorb", output, [], rows=2, batch=32, n=512)
    assert targets[0].shape == (2, 32, 512)
    targets[0].fill(1)
    pitched = output.reshape(3, 64, 576)
    assert np.isfinite(pitched[:2, :32, :512]).all()
    assert np.isnan(pitched[:2, :32, 512:]).all()
    assert np.isnan(pitched[:2, 32:]).all()
    assert np.isnan(pitched[2]).all()
    finals = [np.zeros((3, 1024)), np.zeros((3, 1024))]
    targets = bench.glm_full_core_output_targets("shared_in", output, finals, rows=2, batch=1, n=2048)
    assert len(targets) == 2
    for target in targets:
        target.fill(7)
    assert all(np.equal(value[:2], 7).all() and np.equal(value[2], 0).all() for value in finals)
    packed = bench.glm_full_core_output_targets("rank_in", output, [], rows=2, batch=1, n=2624)
    assert packed[0].shape == (2, 64*576)


def test_glm_core_catalog_rejects_bad_original_headers_and_ownership():
    bench = _native_threshold_bench()
    config, tensors = _glm_core_headers()
    with pytest.raises(ValueError, match="onboarded prefix"):
        bench.glm_full_core_operations(config, tensors, onboard_layers=4)
    tensors["model.layers.0.self_attn.q_a_proj.weight"]["dtype"] = "F8_E4M3"
    with pytest.raises(ValueError, match="original matrix is not BF16"):
        bench.glm_full_core_operations(config, tensors, onboard_layers=0)
    tensors["model.layers.0.self_attn.q_a_proj.weight"]["dtype"] = "BF16"
    tensors["model.layers.0.self_attn.kv_b_proj.weight"]["shape"] = [28671, 512]
    with pytest.raises(ValueError, match="kv_b source geometry differs"):
        bench.glm_full_core_operations(config, tensors, onboard_layers=0)


def test_qwen_measurement_requires_fixture_and_native_library_before_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path),
                                     "--qwen-core", str(tmp_path/"headers.json")])
    with pytest.raises(SystemExit) as error:
        _native_threshold_bench().main()
    assert error.value.code == 2


def test_full_core_requires_config_before_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,"argv",["benchmark","--output-dir",str(tmp_path),"--full-core","--kimi-decision"])
    with pytest.raises(SystemExit) as error:
        _native_threshold_bench().main()
    assert error.value.code == 2


def _muse_core_headers():
    config = dict(hidden_size=8192, num_hidden_layers=2, num_attention_heads=128,
        num_key_value_heads=16, head_dim=64, moe_hidden_size=4096, vocab_size=202048,
        num_local_experts=256, num_shared_experts=0,
        layer_types=["sliding_attention", "full_attention"], mlp_layer_types=["sparse"]*2)
    tensors = {"lm_head.weight": dict(dtype="BF16", shape=[202048, 8192])}
    for layer in range(2):
        p = f"model.language_model.layers.{layer}."
        for leaf, shape in (("self_attn.q_proj", [8192, 8192]), ("self_attn.k_proj", [1024, 8192]),
                ("self_attn.v_proj", [1024, 8192]), ("self_attn.o_proj", [8192, 8192]),
                ("self_attn.gate_proj", [8192, 8192]), ("mlp.pre_expert_proj", [4096, 8192]),
                ("mlp.post_expert_proj", [8192, 4096]), ("mlp.gate", [256, 8192])):
            tensors[p+leaf+".weight"] = dict(dtype="F32" if leaf == "mlp.gate" else "BF16", shape=shape)
    return config, tensors


def test_muse_catalog_covers_all_original_target_matrices_and_exact_boundaries():
    bench = _native_threshold_bench()
    config, tensors = _muse_core_headers()
    operations = bench.muse_full_core_operations(config, tensors)
    assert len(operations) == 17
    unique = {op["role"]: op for op in bench.glm_full_core_unique_operations(operations)}
    assert len(unique) == 9
    assert unique["head"]["count"] == [0, 1]
    assert unique["head"]["members"][0]["stored_n"] == 202112
    assert unique["attn_k"]["packed_n"] == 512
    assert unique["pre_expert"]["packed_n"] == 2048
    assert unique["post_expert"]["k"] == 2048
    assert unique["post_expert"]["reduction_count"] == [2, 2]
    assert "F32 retained producer copy" in unique["attn_o"]["boundary"]
    assert all(not unique[role]["hypothetical_trellis"] for role in ("router", "attn_gate"))
    assert unique["router"]["weight_dtype"] == "F32" and unique["router"]["count"] == [2, 2]
    assert len(bench.glm_full_core_fixture_shapes(operations)) == 7
    for name in ("model.language_model.layers.0.self_attn.k_proj.weight", "lm_head.weight"):
        bad = {k: dict(v) for k, v in tensors.items()}
        bad[name]["shape"] = [128, 128]
        with pytest.raises(ValueError, match="original matrix differs"):
            bench.muse_full_core_operations(config, bad)
    tensors["model.language_model.layers.0.mlp.extra.weight"] = dict(dtype="BF16", shape=[128, 128])
    with pytest.raises(ValueError, match="uncovered"):
        bench.muse_full_core_operations(config, tensors)


def test_muse_fp8_specs_keep_selectable_attention_separate_from_mixed_full_core():
    bench = _native_threshold_bench()
    config, tensors = _muse_core_headers()
    operations = bench.glm_full_core_unique_operations(bench.muse_full_core_operations(config, tensors))
    for op in operations:
        if op["role"] in ("router", "attn_gate"):
            with pytest.raises(ValueError, match="kept-native"):
                bench.muse_full_core_projection_spec(op, fp8=True)
            continue
        spec = bench.muse_full_core_projection_spec(op, fp8=True, prefill=True)
        assert spec["fp8_prefill_w8a8"] == (op["output"] == "bf16")
        assert spec["fp8_scale_layout"] == "block128x128_f32"
        if op["shipping_fp8_attention"]:
            ship = bench.muse_full_core_projection_spec(op, fp8=True, prefill=True, shipping=True)
            assert not ship["fp8_prefill_w8a8"] and ship["fp8_decode_rows"] == 64
        else:
            with pytest.raises(ValueError, match="shipping-selectable"):
                bench.muse_full_core_projection_spec(op, fp8=True, shipping=True)


def test_muse_factories_match_underlying_abi_and_preserve_f32_prefill(monkeypatch):
    bench = _native_threshold_bench()
    calls = []

    class Projection:
        def __init__(self, n, k, **kwargs):
            calls.append((n, k, kwargs))

        def key(self):
            return ("projection",)

    monkeypatch.setitem(sys.modules, "cutlass", SimpleNamespace(Float32="f32", BFloat16="bf16"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="f32", bfloat16="bf16", uint8="u8"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._common", SimpleNamespace(
        Operand=lambda *a, **kw: (a, kw), Scalar=lambda *a: a,
        compile_program=lambda launch, **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd._fp8_weights", SimpleNamespace(Fp8Projection=Projection))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd.muse", SimpleNamespace(compile_router_aot=lambda: "router"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd.pointwise", SimpleNamespace(
        compile_bf16_projection_aot=lambda **kw: ("gate", kw)))
    monkeypatch.setitem(sys.modules, "b12x.gemm.bf16_gemv._skinny", SimpleNamespace(
        DECODE_TMA_TILES={(8192, 4096): ("measured",)}, TmaBf16Projection=Projection))
    config, tensors = _muse_core_headers()
    ops = {o["role"]: o for o in bench.glm_full_core_unique_operations(bench.muse_full_core_operations(config, tensors))}
    for role, w8a8 in (("attn_q", True), ("attn_o", False), ("head", False)):
        program = bench.muse_full_core_projection_program(ops[role], prefill=True, fp8=True)
        assert calls[-1][2]["prefill_mask"] == (255 if w8a8 else 0)
        assert calls[-1][2]["out_dtype"] == ("bf16" if w8a8 else "f32")
        assert calls[-1][2]["prefill_rows"] == 4096
        assert [operand[0][0] for operand in program["operands"]] == ["x", "w_fp8", "scale", "out"]
        assert program["scalars"] == (("rows",), ("fp8_rows",), ("qscratch", "int64"))
        hash(program["key"])
    bench.muse_full_core_projection_program(ops["attn_o"], prefill=True, fp8=True, shipping=True)
    assert calls[-1][2]["prefill_rows"] is None and calls[-1][2]["decode_rows"] == 64
    bench.muse_full_core_projection_program(ops["attn_o"])
    assert calls[-1][2] == dict(out_dtype="f32", tiles_by_rows=("measured",))
    assert bench.muse_full_core_projection_program(ops["attn_gate"])[0] == "gate"
    assert bench.muse_full_core_projection_program(ops["router"]) == "router"


def test_muse_aggregate_includes_all_format_meets_and_f32_fp8_ceiling():
    bench = _native_threshold_bench()
    ops = [dict(role="attn_o", split="input", hypothetical_trellis=True, shipping_fp8_attention=True,
                output="f32", count=[2, 2], reduction_count=[2, 2]),
           dict(role="head", split="whole", hypothetical_trellis=True, shipping_fp8_attention=False,
                output="f32", count=[0, 1], reduction_count=[0, 0]),
           dict(role="router", split="whole", hypothetical_trellis=False, shipping_fp8_attention=False,
                output="f32", count=[2, 2], reduction_count=[0, 0])]
    records = []
    for row in (1, 2, 2240, 2688, 2752):
        records.extend([
            dict(role="attn_o", rows=row, phase="producer", total_call_us=dict(bf16=1, k6=2, k8=3, fp8_attention=4, fp8_full=5)),
            dict(role="attn_o", rows=row, phase="reduce0", total_call_us=dict(bf16=6, k6=7, k8=8, fp8_attention=9, fp8_full=10)),
            dict(role="attn_o", rows=row, phase="reduce1", total_call_us=dict(bf16=6, k6=7, k8=8, fp8_attention=9, fp8_full=10)),
            dict(role="head", rows=row, phase="producer", total_call_us=dict(bf16=11, k6=12, k8=13, fp8_full=14)),
            dict(role="router", rows=row, phase="producer", total_call_us=dict(native=1))])
    result = bench.aggregate_muse_full_core(ops, records, [2, 64])
    decode = result["decode"][0]
    assert decode["local_dense_us"] == dict(bf16=[16, 27], k6=[20, 32], k8=[24, 37],
                                          fp8_selectable=[28, 40], fp8_full=[32, 46])
    f32 = decode["mixed_fp8_f32_roles"]
    assert f32["complete_role_us"] == [30, 44]
    assert f32["w8a16_producer_us"] == [10, 24]
    assert f32["zero_producer_cost_lower_bound_us"] == [22, 22]
    assert f32["complete_role_share"] == [30/32, 44/46]
    assert decode["kimi_rate_normalized_local_dense_us_per_emitted_token"]["fp8_full"] == [32*73/255, 46*73/255]
    assert result["prefill"][0]["local_dense_us"]["fp8_selectable"] == [84, 96]
    assert "F32-output W8A16" in result["fp8_full"]


@pytest.mark.parametrize("extra", [[], ["--glm-core", "headers.json"], ["--qwen-core", "headers.json"]])
def test_muse_measurement_refuses_incomplete_cli_before_cuda(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output-dir", str(tmp_path),
                                     "--muse-core", "headers.json", *extra])
    with pytest.raises(SystemExit) as error:
        _native_threshold_bench().main()
    assert error.value.code == 2


def _native_route_policy():
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                (t.id.startswith("_K6_") or t.id.startswith("_GROUPED_") or
                 t.id in ("DECODE_ROWS_BY_SHAPE", "_FOLDED_DECODE_ROWS", "_GEMINI_NARROW_ROWS"))
                for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("decode_rows", "decode_route_thresholds"):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_measured_decode_rows_are_scoped_to_qualified_format_and_storage_sides():
    policy = _native_route_policy()
    build = policy["decode_rows"]
    for (bits, n, k), (edge, sides) in policy["DECODE_ROWS_BY_SHAPE"].items():
        for side in (1, 2):
            g = dict(bits=bits, output_features=n, input_features=k, sharded_side=side,
                     codebook="mcg", input_dtype="bf16", groups=1)
            assert build(g) == (edge if side in sides else 16)
            for key, value in (("codebook", "mul1"), ("input_dtype", "f16"), ("groups", 4)):
                assert build(dict(g, **{key: value})) == 16
    assert build(dict(bits=6, output_features=3072, input_features=7168,
                      sharded_side=1, codebook="mcg", input_dtype="bf16")) == 16


@pytest.mark.parametrize("bits,n,k,groups,expected", [
    (6, 6144, 7168, 1, (8, 32, 48)),
    (6, 6144, 7168, 4, (16, 32, 48)),
    (6, 3072, 7168, 1, (16, 32, 48)),
    (6, 163840, 7168, 1, (16, 32, 48)),
    (8, 6144, 7168, 1, (8, 32, 48)),
    (8, 6144, 7168, 4, (16, 32, 48)),
    (8, 7168, 3072, 1, (16, 32, 48)),
    (4, 6144, 7168, 1, ()),
])
def test_native_route_metadata_describes_each_actual_program(bits, n, k, groups, expected):
    policy = _native_route_policy()
    g = dict(bits=bits, output_features=n, input_features=k, splits=(k + 1023)//1024,
             sharded_side=1, codebook="mcg", input_dtype="bf16", groups=groups, max_rows=4096)
    assert policy["decode_route_thresholds"](g) == expected
    g["max_rows"] = 8
    assert policy["decode_route_thresholds"](g) == ()


def test_native_threshold_variants_only_change_legal_dispatch(tmp_path):
    bench = _native_threshold_bench()
    source = _PATH.parents[1] / "b12x/gemm/trellis_linear/_native.py"
    original = source.read_text()
    unchanged = bench.threshold_variant(source, tmp_path / "defaults.py", bench.threshold_config(None))
    assert unchanged.read_text() == original
    values = bench.threshold_config("narrow=4,decode=8,wide2=24,wide3=32,prefill=64")
    variant = bench.threshold_variant(source, tmp_path / "variant.py", values).read_text()
    assert "_K6_NARROW_ROWS = 4" in variant
    assert "_PREFILL_MIN_ROWS = 64" in variant
    before, after = ast.parse(original), ast.parse(variant)
    base = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name == "NativeTrellisLinear")
    changed = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name == "NativeTrellisLinear")
    methods = {n.name: n for n in changed.body if isinstance(n, ast.FunctionDef)}
    for method in (n for n in base.body if isinstance(n, ast.FunctionDef)):
        if method.name not in ("__init__", "_legacy", "_generic_grouped"):
            assert ast.dump(method) == ast.dump(methods[method.name]), method.name
    assert source.read_text() == original
