from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass")


def _load_benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "benchmark_mixed_trellis.py"
    spec = importlib.util.spec_from_file_location("benchmark_mixed_trellis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Tier:
    num_experts: int
    trellis_bits: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor


def test_benchmark_materializes_full_abi5_tier_storage(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark, "HIDDEN", 32)
    monkeypatch.setattr(benchmark, "INTERMEDIATE", 16)

    materialized = 2
    compiled = 3
    bits = 3
    w13 = torch.arange(2 * materialized * 2 * 1 * 8 * bits, dtype=torch.int32)
    w2 = torch.arange(materialized * 1 * 2 * 8 * bits, dtype=torch.int32)
    tier = _Tier(
        num_experts=materialized,
        trellis_bits=bits,
        w13=w13,
        w2=w2,
        w13_global_scale=torch.tensor([2.0, 3.0]),
        w2_global_scale=torch.tensor([4.0, 5.0]),
    )

    padded = benchmark._pad_exact_tier_storage(tier, compiled)

    assert padded.num_experts == compiled
    padded_w13 = padded.w13.view(2, compiled, 2, 1, 8 * bits)
    assert torch.equal(
        padded_w13[:, :materialized], w13.view(2, materialized, 2, 1, 8 * bits)
    )
    assert torch.count_nonzero(padded_w13[:, materialized:]) == 0
    padded_w2 = padded.w2.view(compiled, 1, 2, 8 * bits)
    assert torch.equal(padded_w2[:materialized], w2.view(materialized, 1, 2, 8 * bits))
    assert torch.count_nonzero(padded_w2[materialized:]) == 0
    assert torch.equal(padded.w13_global_scale, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.equal(padded.w2_global_scale, torch.tensor([4.0, 5.0, 1.0]))


def test_benchmark_kimi_variant_parsing_and_geometry() -> None:
    benchmark = _load_benchmark_module()
    assert (
        benchmark.KIMI_HIDDEN,
        benchmark.KIMI_INTERMEDIATE,
        benchmark.KIMI_EXPERTS,
        benchmark.KIMI_TOPK,
        benchmark.KIMI_BITS,
    ) == (3584, 512, 896, 16, (2, 3))
    parsed = [
        benchmark._parse_kimi_variant(v) for v in benchmark.KIMI_DEFAULT_VARIANTS
    ]
    assert parsed == [
        ("old", None, (64, 256, 64, 256), 1),
        ("n128-2cta", None, (64, 128, 64, 128), 2),
    ]


@pytest.mark.parametrize("value", ["ws:32:ws", "coop8:8:cooperative", "coop16:16:cooperative"])
def test_prefill_arm(value):
    bench = _load_benchmark_module()
    name, block, mode = value.split(":")
    assert bench.parse_prefill_arm(value) == (name, int(block), mode)


@pytest.mark.parametrize("value", ["bad", "ws:8:ws", "a:17:ws", "a_control:32:ws", ":32:ws", "a:32:other"])
def test_prefill_arm_rejects_invalid(value):
    bench = _load_benchmark_module()
    with pytest.raises(bench.argparse.ArgumentTypeError):
        bench.parse_prefill_arm(value)


@pytest.mark.parametrize("correctness_only,layers,accepted", [
    (False, 1, False), (False, 2, True), (True, 1, True), (True, 0, False),
])
def test_prefill_single_layer_is_correctness_only(correctness_only, layers, accepted):
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    args = bench.benchmark_parser().parse_args([
        "--prefill-family", "dsv4p", "--capacities", "2752", "--layers", str(layers),
        "--rounds", "3", "--replays", "1",
        *(["--correctness-only"] if correctness_only else []),
    ])
    # Stop immediately after the layer gate; no CUDA work or fixture allocation.
    args.prefill_arm = [("duplicate", 8, "cooperative")] * 2
    props = SimpleNamespace(major=12, minor=1)
    with pytest.raises(ValueError, match="uniquely named arms" if accepted else "cold layers"):
        bench._run_prefill_benchmark(args, None, props)


def test_prefill_policy_separates_route_block_from_ws():
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    qwen_coop = bench.prefill_compile_options("qwen4", 256, 32, "cooperative", props)
    qwen_ws = bench.prefill_compile_options("qwen4", 256, 32, "ws", props)
    assert qwen_coop["moe_block_size"] == qwen_ws["moe_block_size"] == 32
    assert qwen_coop["token_major_rotation"] and not qwen_ws["token_major_rotation"]
    assert qwen_ws["warp_specialized"] and not qwen_coop["warp_specialized"]
    assert qwen_ws["tier0_bits"] == 4 and qwen_ws["tier1_bits"] == 5
    assert qwen_ws["route_num_experts"] == 512 and qwen_ws["top_k"] == 10
    coop = bench.prefill_compile_options("kimik3", 512, 16, "cooperative", props)
    ws = bench.prefill_compile_options("kimik3", 512, 16, "ws", props)
    assert {k for k in coop if coop[k] != ws[k]} == {"warp_specialized"}
    assert ws["activation"] == "situ" and ws["tier1_num_experts"] == 1
    assert bench.prefill_compile_options("kimik3", 512, 8, "cooperative", props)["moe_block_size"] == 8
    for width in (128, 256, 384, 640):
        opts = bench.prefill_compile_options("qwen4", width, 32, "ws", props)
        assert width % opts["force_tile_config"][1] == 0
    with pytest.raises(ValueError):
        bench.prefill_compile_options("kimik3", 384, 16, "ws", props)


@pytest.mark.parametrize("bits", [(2, 3), (3, 4), (4, 5)])
def test_prefill_bits_override_reaches_compile_and_compact_fixture(tmp_path, bits):
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    args = bench.benchmark_parser().parse_args(
        ["--prefill-family", "v41", "--bits", *map(str, bits)])
    assert args.bits == list(bits)
    ordinary = bench.prefill_compile_options("v41", 640, 64, "ws", props, bits=bits)
    compact = bench.prefill_compile_options("v41", 640, 64, "ws", props, compact=True, bits=bits)
    coordinator = bench.coordinator_compile_options("rtx-tp1", 256, {}, props, "v41", bits=bits)
    for options in (ordinary, compact, coordinator):
        assert (options["tier0_bits"], options["tier1_bits"]) == bits
    entries = [dict(combined_local=e, global_expert=e, width=640,
                    projections=[dict(projection=p, tier=e % 2, bits=bits[e % 2], tier_local=e // 2)
                                 for p in ("gate", "up", "down")]) for e in range(384)]
    fixture = tmp_path / "compact.json"
    fixture.write_text(bench.json.dumps(dict(ranks=[dict(
        rank=0, compiled_max_intermediate=640, experts=entries,
        projection_counts=[[192] * 3, [192] * 3], unit_map_fingerprint="override")])))
    assert bench.compact_prefill_fixture(fixture, 0, "v41", 640, bits=bits)["counts"] == [[192] * 3] * 2
    if bits != (3, 4):
        with pytest.raises(ValueError, match="tier/bit mismatch"):
            bench.compact_prefill_fixture(fixture, 0, "v41", 640)


def test_prefill_route_fixture_is_family_specific(tmp_path):
    bench = _load_benchmark_module()
    path = tmp_path / "routes.json"
    wave = dict(layer=23, rows=4096, expert_rows=[[e, 4096] for e in range(10)])
    data = dict(schema=1, family="qwen4", experts=512, top_k=10, waves=[wave])
    path.write_text(bench.json.dumps(data))
    assert bench.prefill_route_waves(path, "qwen4", 23) == [wave]
    with pytest.raises(ValueError, match="metadata"):
        bench.prefill_route_waves(path, "kimik3", 23)
    with pytest.raises(ValueError, match="no waves"):
        bench.prefill_route_waves(path, "qwen4", 0)
    data["waves"][0]["expert_rows"][0][0] = 512
    path.write_text(bench.json.dumps(data))
    with pytest.raises(ValueError, match="histogram"):
        bench.prefill_route_waves(path, "qwen4", 23)


def test_prefill_kimi_routes_preserve_high_expert_ids(tmp_path):
    bench = _load_benchmark_module()
    wave = dict(layer=23, rows=4096, expert_rows=[[e, 4096] for e in range(880, 896)])
    path = tmp_path / "routes.json"
    path.write_text(bench.json.dumps(dict(schema=1, family="kimik3", experts=896, top_k=16, waves=[wave])))
    assert bench.prefill_route_waves(path, "kimik3", 23) == [wave]


def test_compact_prefill_fixture_requires_complete_ownership(tmp_path):
    bench = _load_benchmark_module()
    entries = [dict(global_expert=e, combined_local=e, width=128,
                    projections=[dict(projection=p, tier=0, bits=4,
                                      tier_local=e, payload_word_offset=0)
                                 for p in ("gate", "up", "down")]) for e in range(512)]
    rank = dict(rank=0, compiled_max_intermediate=256, experts=entries,
                projection_counts=[[512, 512, 512], [0, 0, 0]], unit_map_fingerprint="test")
    path = tmp_path / "ownership.json"
    path.write_text(bench.json.dumps(dict(ranks=[rank])))
    assert len(bench.compact_prefill_fixture(path, 0, "qwen4", 256, 512)["entries"]) == 512
    with pytest.raises(ValueError, match="requested rank"):
        bench.compact_prefill_fixture(path, 1, "qwen4", 256, 512)
    with pytest.raises(ValueError, match="requires Qwen"):
        bench.compact_prefill_fixture(path, 0, "kimik3", 256, 512)
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 0, "qwen4", 128, 512)
    entries[0]["width"] = 129
    path.write_text(bench.json.dumps(dict(ranks=[rank])))
    with pytest.raises(ValueError, match="live width"):
        bench.compact_prefill_fixture(path, 0, "qwen4", 256, 512)
    entries[0]["width"] = 128
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 0, "qwen4", 256, 256)


def test_compact_dsv4p_fixture_validates_tp5_geometry(tmp_path):
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    entries = [dict(global_expert=e, combined_local=e, width=640 if e % 5 else 512,
                    projections=[dict(projection=p, tier=0, bits=2, tier_local=e,
                                      payload_word_offset=0)
                                 for p in ("gate", "up", "down")]) for e in range(384)]
    rank = dict(rank=0, compiled_max_intermediate=640, experts=entries,
                projection_counts=[[384] * 3, [0] * 3], unit_map_fingerprint="v4pro-tp5")
    path = tmp_path / "dsv4p-ownership.json"
    path.write_text(bench.json.dumps(dict(tiers=[2, 3], ranks=[rank])))
    fixture = bench.compact_prefill_fixture(path, 0, "dsv4p", 640)
    assert {e["width"] for e in fixture["entries"]} == {512, 640}
    assert fixture["fingerprint"] == "v4pro-tp5"
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 0, "dsv4p", 512)
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 0, "dsv4p", 640, experts=256)
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    options = bench.prefill_compile_options("dsv4p", 640, 64, "ws", props, compact=True)
    assert options["compact_weights"] and options["warp_specialized"]
    assert options["route_num_experts"] == 384 and options["top_k"] == 6
    assert (options["tier0_bits"], options["tier1_bits"]) == (2, 3)


def test_compact_glm_fixture_uses_live_width_ownership(tmp_path):
    bench = _load_benchmark_module()
    entries = [dict(global_expert=e, combined_local=e, width=384 if e < 85 else 256,
                    projections=[dict(projection=p, tier=e & 1, bits=4 + (e & 1),
                                      tier_local=e, payload_word_offset=0)
                                 for p in ("gate", "up", "down")]) for e in range(256)]
    rank = dict(rank=2, compiled_max_intermediate=384, experts=entries,
                projection_counts=[[82, 80, 64], [86, 88, 96]], unit_map_fingerprint="test")
    path = tmp_path / "glm-ownership.json"
    path.write_text(bench.json.dumps(dict(ranks=[rank])))
    fixture = bench.compact_prefill_fixture(path, 2, "glm", 384, 256)
    assert sorted({e["width"] for e in fixture["entries"]}) == [256, 384]
    with pytest.raises(ValueError, match="geometry"):
        bench.compact_prefill_fixture(path, 2, "glm", 384, 512)
    # Only the TP6 Spark profile ships compact live widths.
    from types import SimpleNamespace
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)
    with pytest.raises(ValueError, match="spark-tp6"):
        bench.coordinator_compile_options("spark-tp4-width512", 4096, {}, props, "glm", compact=True)
    options = bench.coordinator_compile_options("spark-tp6-width384", 4096, {"mode": "ws"}, props, "glm", compact=True)
    assert options["compact_weights"] and options["warp_specialized"]
    assert options["force_tile_config"] == (64, 128, 64, 128)
    assert options["moe_block_size"] == 32 and not options["token_major_rotation"]
    baseline = bench.coordinator_compile_options("spark-tp6-width384", 4096, {}, props, "glm", compact=True)
    assert baseline["moe_block_size"] == 64 and baseline["token_major_rotation"]
    assert not baseline["warp_specialized"] and baseline["compact_weights"]
    widened = bench.coordinator_compile_options("spark-tp6-width384", 4096,
                                                {"mode": "ws", "tile": (64, 128, 64, 256)},
                                                props, "glm", compact=True)
    assert widened["force_tile_config"] == (64, 128, 64, 256)
    t384 = bench.coordinator_compile_options("spark-tp6-width384", 4096,
                                             {"mode": "ws", "tile": (64, 192, 64, 256)},
                                             props, "glm", compact=True)
    assert t384["force_tile_config"] == (64, 192, 64, 256)


def test_compact_prefill_compile_options_match_served_exports(tmp_path):
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=101376)

    # Qwen4 compact (TP4/TP3 w256, TP2 w384): cooperative must force (64,128,64,128) and no token rotation.
    for width in (256, 384):
        qwen_coop = bench.prefill_compile_options("qwen4", width, 32, "cooperative", props, compact=True)
        assert qwen_coop["compact_weights"] is True
        assert qwen_coop["warp_specialized"] is False
        assert qwen_coop["force_tile_config"] == (64, 128, 64, 128)
        assert qwen_coop["token_major_rotation"] is False
        assert qwen_coop["ws_dynamic_tiles"] is False
        assert qwen_coop["swiglu_limit"] is None
        assert (qwen_coop["tier0_num_experts"], qwen_coop["tier1_num_experts"]) == (512, 512)

        qwen_ws = bench.prefill_compile_options("qwen4", width, 64, "ws", props, compact=True)
        assert qwen_ws["compact_weights"] is True
        assert qwen_ws["warp_specialized"] is True
        assert qwen_ws["force_tile_config"] == (64, 128, 64, 256)
        assert qwen_ws["token_major_rotation"] is False
        assert qwen_ws["ws_dynamic_tiles"] is True
        assert qwen_ws["swiglu_limit"] is None

    qwen_t384 = bench.prefill_compile_options("qwen4", 384, 64, "ws+t384", props, compact=True)
    assert qwen_t384["force_tile_config"] == (64, 192, 64, 256) and qwen_t384["ws_dynamic_tiles"] is True
    with pytest.raises(ValueError, match="t384"):
        bench.prefill_compile_options("qwen4", 256, 64, "ws+t384", props, compact=True)

    # GLM 5.3 (glm, bits=(4,5), swiglu_limit=None), GLM Flash (glmf, bits=(3,4), swiglu_limit=10.0),
    # V4 Flash (dsv4f, bits=(4,5), swiglu_limit=10.0), and V4.1 (v41, bits=(3,4), swiglu_limit=10.0)
    for family, experts, topk, bits, block, clamp, widths in (
        ("glm", 256, 8, (4, 5), 64, None, (384, 768)),
        ("glmf", 288, 8, (3, 4), 64, 10.0, (384, 768)),
        ("dsv4f", 256, 6, (4, 5), 8, 10.0, (384, 768)),
        ("v41", 384, 6, (3, 4), 8, 10.0, (640,)),
    ):
        for width in widths:
            coop = bench.prefill_compile_options(family, width, block, "cooperative", props, compact=True)
            assert coop["compact_weights"] is True and not coop["warp_specialized"]
            assert coop["force_tile_config"] == (64, 128, 64, 128)
            assert not coop["token_major_rotation"] and not coop["ws_dynamic_tiles"]
            assert coop["swiglu_limit"] == clamp
            assert (coop["tier0_bits"], coop["tier1_bits"]) == bits
            assert (coop["tier0_num_experts"], coop["tier1_num_experts"], coop["route_num_experts"]) == (experts, experts, experts)
            assert coop["top_k"] == topk and coop["moe_block_size"] == block

            ws = bench.prefill_compile_options(family, width, 64, "ws", props, compact=True)
            assert ws["compact_weights"] is True and ws["warp_specialized"]
            assert ws["force_tile_config"] == (64, 128, 64, 256)
            assert ws["ws_dynamic_tiles"] is True and ws["swiglu_limit"] == clamp

            ws_t384 = bench.prefill_compile_options(family, width, 64, "ws+t384", props, compact=True)
            assert ws_t384["force_tile_config"] == (64, 192, 64, 256) and ws_t384["ws_dynamic_tiles"] is True

    # Validate compact_prefill_fixture for glmf (bits 3/4), dsv4f (bits 4/5), and v41 (bits 3/4, width 640)
    glmf_entries = [dict(global_expert=e, combined_local=e, width=384 if e < 96 else 256,
                         projections=[dict(projection=p, tier=e & 1, bits=3 + (e & 1),
                                           tier_local=e // 2, payload_word_offset=0)
                                      for p in ("gate", "up", "down")]) for e in range(288)]
    glmf_rank = dict(rank=0, compiled_max_intermediate=384, experts=glmf_entries,
                     projection_counts=[[144, 144, 144], [144, 144, 144]], unit_map_fingerprint="glmf-test")
    glmf_path = tmp_path / "glmf-ownership.json"
    glmf_path.write_text(bench.json.dumps(dict(synthetic=True, source="synthetic", ranks=[glmf_rank])))
    glmf_fix = bench.compact_prefill_fixture(glmf_path, 0, "glmf", 384)
    assert glmf_fix["synthetic"] is True and len(glmf_fix["entries"]) == 288

    # V4.1 compact TP4 w640: (3,4) bits, swiglu_limit 10.0, and a non-divisible
    # 192-wide FC1 tile arm (640 = 2 x 320-row halves -> 2 x 192-wide tiles).
    v41 = bench.prefill_compile_options("v41", 640, 64, "ws", props, compact=True)
    assert v41["force_tile_config"] == (64, 128, 64, 256)
    assert (v41["tier0_bits"], v41["tier1_bits"]) == (3, 4)
    assert (v41["tier0_num_experts"], v41["route_num_experts"], v41["top_k"]) == (384, 384, 6)
    assert v41["swiglu_limit"] == 10.0
    v41_t384 = bench.prefill_compile_options("v41", 640, 64, "ws+t384", props, compact=True)
    assert v41_t384["force_tile_config"] == (64, 192, 64, 256) and v41_t384["ws_dynamic_tiles"] is True
    with pytest.raises(ValueError, match="t384"):
        bench.prefill_compile_options("v41", 256, 64, "ws+t384", props, compact=True)

    v41_entries = [dict(global_expert=e, combined_local=e, width=640 if e < 192 else 512,
                        projections=[dict(projection=p, tier=e & 1, bits=3 + (e & 1),
                                          tier_local=e // 2, payload_word_offset=0)
                                     for p in ("gate", "up", "down")]) for e in range(384)]
    v41_rank = dict(rank=0, compiled_max_intermediate=640, experts=v41_entries,
                    projection_counts=[[192, 192, 192], [192, 192, 192]], unit_map_fingerprint="v41-test")
    v41_path = tmp_path / "v41-ownership.json"
    v41_path.write_text(bench.json.dumps(dict(synthetic=True, source="synthetic", ranks=[v41_rank])))
    v41_fix = bench.compact_prefill_fixture(v41_path, 0, "v41", 640)
    assert v41_fix["synthetic"] is True and len(v41_fix["entries"]) == 384


@pytest.mark.parametrize("bits", [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
def test_compact_pair_remap_preserves_planner_ownership(tmp_path, bits):
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import _compact_trellis_layout
    bench = _load_benchmark_module()
    entries = [dict(global_expert=e, combined_local=e, width=128 if e % 3 else 256,
                    h128_units=[e % 5], projections=[
                        dict(projection=p, tier=e % 2, bits=(4, 5)[e % 2],
                             tier_local=e // 2) for p in ("gate", "up", "down")])
               for e in range(512)]
    rows = [[((e % 2) << 9) | (e // 2) for e in range(512)] for _ in range(3)]
    rows.append([e["width"] for e in entries])
    canonical, _, counts = _compact_trellis_layout(
        rows, hidden_size=2560, intermediate_size=256, bits=(4, 5))
    for e in entries:
        for p, projection in enumerate(e["projections"]):
            projection["payload_word_offset"] = canonical[4 + p][e["combined_local"]]
    rank = dict(rank=0, compiled_max_intermediate=256, experts=entries,
                projection_counts=[list(c) for c in counts], unit_map_fingerprint="planner-real")
    data = dict(tiers=[4, 5], ranks=[rank], source="planner")
    path = tmp_path / "ownership.json"
    path.write_text(bench.json.dumps(data))
    original = bench.compact_prefill_fixture(path, 0, "qwen4", 256)
    assert bench.compact_prefill_fixture(path, 0, "qwen4", 256, bits=(4, 5)) == original
    remapped = bench.compact_prefill_fixture(path, 0, "qwen4", 256, bits=bits)
    expected, _, _ = _compact_trellis_layout(rows, hidden_size=2560,
                                            intermediate_size=256, bits=bits)
    assert remapped["counts"] == original["counts"]
    assert remapped["fingerprint"] == original["fingerprint"]
    assert bench.json.loads(path.read_text()) == data
    for before, after in zip(original["entries"], remapped["entries"]):
        assert {k: v for k, v in after.items() if k != "projections"} == {
            k: v for k, v in before.items() if k != "projections"}
        for p, (old, new) in enumerate(zip(before["projections"], after["projections"])):
            assert new["bits"] == bits[new["tier"]]
            assert new["payload_word_offset"] == expected[4 + p][after["combined_local"]]
            assert (new["tier"], new["tier_local"]) == (old["tier"], old["tier_local"])


def test_mul1_coordinator_arm_and_legacy_defaults() -> None:
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    default_args = bench.benchmark_parser().parse_args([
        "--prefill-family", "glm", "--role", "coordinator", "--output", "out.json"
    ])
    assert default_args.codebook == "mcg" and default_args.compare_aot_root is None
    props = SimpleNamespace(multi_processor_count=188, shared_memory_per_block_optin=101376)
    mcg_opts = bench.coordinator_compile_options("rtx-tp2", 256, {}, props, "glm")
    assert mcg_opts["trellis_codebook"] == "mcg"
    name, parsed = bench.parse_coordinator_arm("mul1_ws:mode=ws,route_block=64,dynamic=1,codebook=mul1")
    assert name == "mul1_ws" and parsed["codebook"] == "mul1"
    mul1_opts = bench.coordinator_compile_options("rtx-tp2", 256, parsed, props, "glm")
    assert mul1_opts["trellis_codebook"] == "mul1"
    with pytest.raises(bench.argparse.ArgumentTypeError):
        bench.parse_coordinator_arm("bad_cb:mode=ws,codebook=sqg")
    with pytest.raises(ValueError, match="codebook"):
        bench.coordinator_compile_options("rtx-tp2", 256, {}, props, "glm", codebook="sqg")


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_mul1_oracle_decoder_matches_independent_reference_bitwise(bits: int) -> None:
    bench = _load_benchmark_module()
    rows, columns = 32, 48
    payload = torch.randint(
        -(1 << 31), (1 << 31) - 1, (rows // 16, columns // 16, 8 * bits),
        dtype=torch.int32, generator=torch.Generator().manual_seed(91 + bits),
    )
    actual = bench.decode_mcg_matrix(payload.flatten(), rows, columns, bits, "mul1")
    mcg_actual = bench.decode_mcg_matrix(payload.flatten(), rows, columns, bits)
    assert not torch.equal(actual, mcg_actual)

    # Independent bit-plane GPTQModel MUL1 reconstruction (0x83DCD12D, 0x6400, 0x1EEE, 0xC931).
    raw_i16 = payload.view(torch.int16)
    num_tiles = (rows // 16) * (columns // 16)
    words = raw_i16.contiguous().view(torch.uint16).reshape(num_tiles, 16 * bits).to(torch.int32) & 0xFFFF
    swapped = words.reshape(num_tiles, 8 * bits, 2).flip(-1).reshape(num_tiles, 16 * bits)
    bit_shifts = torch.arange(15, -1, -1, dtype=torch.int32)
    bit_plane = ((swapped.unsqueeze(-1) >> bit_shifts) & 1).reshape(num_tiles, 256 * bits)
    ends = torch.arange(1, 257, dtype=torch.long) * bits - 1
    shifts = torch.arange(16, dtype=torch.long)
    idx = (ends[:, None] - shifts[None, :]) % (256 * bits)
    states = ((bit_plane[:, idx].to(torch.int64) << shifts[None, None, :]).sum(dim=-1) & 0xFFFF).to(torch.int64)
    prod = (states * 0x83DCD12D) & 0xFFFFFFFF
    bsum = (prod & 0xFF) + ((prod >> 8) & 0xFF) + ((prod >> 16) & 0xFF) + ((prod >> 24) & 0xFF)
    h = (0x6400 + bsum).to(torch.uint16).view(torch.float16).to(torch.float64)
    kinv = float(torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16).item())
    kbias = float(torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16).item())
    vals = (h * kinv + kbias).to(torch.float16).to(torch.float32)
    row_idx, col_idx = [], []
    for t in range(32):
        r, c = 2 * (t % 4), t // 4
        for rr, cc in ((r, c), (r + 1, c), (r + 8, c), (r + 9, c),
                       (r, c + 8), (r + 1, c + 8), (r + 8, c + 8), (r + 9, c + 8)):
            row_idx.append(rr)
            col_idx.append(cc)
    tiles = torch.empty((num_tiles, 16, 16), dtype=torch.float32)
    tiles[:, torch.tensor(row_idx), torch.tensor(col_idx)] = vals
    expected = tiles.reshape(rows // 16, columns // 16, 16, 16).permute(0, 2, 1, 3).reshape(rows, columns).contiguous()
    assert torch.equal(actual, expected)


def test_mixed_codebook_oracle_case_qualifies_own_codebooks_and_reports_deltas(monkeypatch) -> None:
    import contextlib
    import sys
    import types
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    noise_module = types.ModuleType("benchmarks.benchmark_exlr8_a16")
    noise_module.k4_noise_summary = lambda values, control: {"seen": sorted(values)}
    runtime_module = types.ModuleType("b12x._lib.runtime_control")
    runtime_module.kernel_resolution_guard = lambda label: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, noise_module.__name__, noise_module)
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    monkeypatch.setitem(bench.COORDINATOR_FAMILIES, "ml4", dict(hidden=128, experts=2, swiglu_limit=None))

    generator = torch.Generator().manual_seed(7)
    words = 8 * 8 * 8 * 4
    tiers = [
        SimpleNamespace(
            num_experts=2, intermediate_size=128, trellis_bits=4,
            w13=torch.randint(-(1 << 31), (1 << 31) - 1, (4 * words,), generator=generator, dtype=torch.int32),
            w2=torch.randint(-(1 << 31), (1 << 31) - 1, (2 * words,), generator=generator, dtype=torch.int32),
            w13_global_scale=torch.ones(2), w2_global_scale=torch.ones(2),
        )
        for _ in range(2)
    ]
    descriptors = torch.tensor([[0, 512, -1, -1], [0, 512, -1, -1], [0, 512, -1, -1]], dtype=torch.int32).flatten()
    rotations = SimpleNamespace(
        intermediate=torch.ones((4, 384)),
        gate_suh=torch.ones((4, 128)),
        up_suh=torch.ones((4, 128)),
        down_svh=torch.ones((4, 128)),
    )
    layer = (*tiers, torch.tensor([0, 1], dtype=torch.int32), descriptors, rotations)
    decoded = torch.randn((4, 128), generator=generator) * 0.25
    ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.full((4, 2), 0.5)

    mcg_ref = bench.coordinator_f32_oracle(layer, decoded, ids, weights, "ml4", "mcg")
    mul1_ref = bench.coordinator_f32_oracle(layer, decoded, ids, weights, "ml4", "mul1")
    assert not torch.equal(mcg_ref, mul1_ref)

    def make_state():
        ws = SimpleNamespace(output=torch.zeros((4, 128)), fc1=torch.zeros(4), activated=torch.zeros(4),
                             packed_route_indices=torch.zeros(64, dtype=torch.int32),
                             block_expert_ids=torch.zeros(8, dtype=torch.int32),
                             packed_route_count=torch.zeros(1, dtype=torch.int32))
        bind = [SimpleNamespace(launch=SimpleNamespace(moe_block_size=8))]
        return ws, bind

    states = {"coop": make_state(), "mul1_arm": make_state()}
    modules = {"coop": SimpleNamespace(pack_topk_routes_by_expert=lambda *a, **k: None),
               "mul1_arm": SimpleNamespace(pack_topk_routes_by_expert=lambda *a, **k: None)}

    class FakeGraph:
        def __init__(self, fn): self.fn = fn
        def replay(self): self.fn()

    monkeypatch.setattr(bench, "_capture", lambda fn, _w: FakeGraph(fn))

    def run(name, rows=None, start=0):
        count = 4 if rows is None else rows
        ref = mcg_ref if name == "coop" else mul1_ref
        out = ref[start:start + count].clone()
        states[name][0].output[:count].copy_(out)
        return out

    args = SimpleNamespace(prefill_family="ml4", warmup=1, prefill_exhaustive=False,
                           live_rows=[], correctness_only=True, rounds=1, replays=1)
    specs = [("coop", {}), ("mul1_arm", {"codebook": "mul1"})]
    result = bench.coordinator_oracle_case(args, specs, states, modules, run, decoded, ids, weights,
                                           layer, 0, 4, torch.int32)
    assert result["arm_checks"]["coop"]["qualified"]
    assert result["arm_checks"]["mul1_arm"]["qualified"]
    assert result["arm_checks"]["mul1_arm"]["control_bitwise"]
    assert "mul1_arm" in result["codebook_deltas"]
    assert result["codebook_deltas"]["mul1_arm"]["max_abs"] > 0.0
    assert -1.0 <= result["codebook_deltas"]["mul1_arm"]["cosine"] <= 1.0

    # Legacy MCG-only invocation omits cross-codebook keys and keeps exact oracle string.
    mcg_only = bench.coordinator_oracle_case(args, [("coop", {})], {"coop": states["coop"]},
                                             {"coop": modules["coop"]}, run, decoded, ids, weights,
                                             layer, 0, 4, torch.int32)
    assert mcg_only["oracle"] == "decoded-MCG/full-F32/no-TF32"
    assert "codebook_deltas" not in mcg_only


def test_aot_replay_variant_derives_compute_and_dtype_from_manifest(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    monkeypatch.setattr(bench.c, "CDLL", lambda _p: SimpleNamespace(
        create=lambda *a: 0, destroy=lambda *a: 0, core=lambda *a: 0, sum=lambda *a: 0,
        routes_create=lambda *a: 0, routes_destroy=lambda *a: 0, routes_launch=lambda *a: 0,
    ))
    monkeypatch.setattr(bench, "bind", lambda lib, name, *a: getattr(lib, name))
    monkeypatch.setattr(bench, "native_call", lambda fn, *a: 0)

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype
            elem = 4 if dtype in ("fp32", "i32") else 2 if dtype in ("bf16", "f16") else 1
            n = 1
            for s in self.shape:
                n *= s
            self.nbytes = n * elem
        def zero_(self): return self
        def to(self, _d): return self

    fake_torch = SimpleNamespace(
        float32="fp32", bfloat16="bf16", float16="f16", int32="i32", uint8="u8",
        empty=lambda shape, dtype, device=None: FakeTensor(shape, dtype),
        frombuffer=lambda buf, dtype: FakeTensor((len(buf) // 2,), dtype),
        cuda=SimpleNamespace(
            get_device_capability=lambda _d=None: (12, 0),
            get_device_properties=lambda _d=None: SimpleNamespace(multi_processor_count=188),
        ),
    )

    lut = b"\x00" * 32
    routes_dir = tmp_path / "routes"
    routes_dir.mkdir()
    routes_manifest = routes_dir / "v41_exl3_routes.json"
    routes_manifest.write_text("{}\n")
    (routes_dir / "libv41_exl3_routes.so").write_bytes(b"")
    (tmp_path / "trellis_lut.bin").write_bytes(lut)
    (tmp_path / "libcuteafd_exl3.so").write_bytes(b"")
    meta = {
        "compute": [12, 0], "hidden": 4096, "intermediate": 2048, "experts": 288, "top_k": 8,
        "bits": [3, 4], "capacity": 80, "output_dtype": "fp32", "codebook": "mul1",
        "requires_route_preparation": True, "blocks_per_sm": 1,
        "objects": [{"label": "v41_exl3_core", "pointer_slots": [], "scalar_slots": []},
                    {"label": "v41_exl3_sum", "pointer_slots": [], "scalar_slots": []}],
        "buffers": {
            name: {"shape": [8], "dtype": dtype, "bytes": 8 * size, "allocation": name, "zero_on_create": False}
            for name, dtype, size in (
                ("fc2", "torch.float16", 2), ("output", "torch.float32", 4),
                ("packed_route_indices", "torch.int32", 4), ("block_expert_ids", "torch.int32", 4),
                ("packed_route_count", "torch.int32", 4), ("expert_offsets", "torch.int32", 4),
                ("expert_counts", "torch.int32", 4),
            )
        },
        "trellis_lut": {"file": "trellis_lut.bin", "sha256": bench.hashlib.sha256(lut).hexdigest()},
        "route_preparation": {"manifest": "routes/v41_exl3_routes.json",
                              "sha256": bench.hashlib.sha256(routes_manifest.read_bytes()).hexdigest()},
    }
    (tmp_path / "v41_exl3.json").write_text(bench.json.dumps(meta))
    weights = ({"expert_map_ptr": FakeTensor((288,), "i32")}, {})
    variant = bench._AotReplayVariant(fake_torch, tmp_path, 80, weights, (4096, 2048, 288, 8, [3, 4]))
    assert variant.meta["compute"] == [12, 0] and variant.buffers["output"].dtype == "fp32"

    # Device capability mismatch fails closed from manifest without CLI flags.
    fake_torch.cuda.get_device_capability = lambda _d=None: (12, 1)
    with pytest.raises(RuntimeError, match="AOT compute capability mismatch"):
        bench._AotReplayVariant(fake_torch, tmp_path, 80, weights, (4096, 2048, 288, 8, [3, 4]))

    default_args = bench.benchmark_parser().parse_args(["--spark-replay", "f.json", "--aot-root", str(tmp_path)])
    assert default_args.verify_own_codebook is False
    verify_args = bench.benchmark_parser().parse_args(
        ["--spark-replay", "f.json", "--aot-root", str(tmp_path), "--verify-own-codebook"]
    )
    assert verify_args.verify_own_codebook is True


def test_legacy_spark_prefill_rejects_codebook_mul1_across_cli_dispatch_and_entry(tmp_path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace
    bench = _load_benchmark_module()
    out_file = tmp_path / "prefill.json"

    # Guard against any CUDA access: CLI dispatch must reject --codebook mul1 before touching CUDA.
    def _forbid_cuda(*_a, **_k):
        raise AssertionError("legacy Spark prefill --codebook mul1 must fail before CUDA initialization")

    monkeypatch.setattr(bench.torch.cuda, "current_device", _forbid_cuda)

    # 1. Non-coordinator Spark prefill family (dsv4f) rejects --codebook mul1 on the actual CLI dispatch path.
    with pytest.raises(SystemExit) as exc_dsv4f:
        bench.main(["--prefill-family", "dsv4f", "--codebook", "mul1", "--output", str(out_file)])
    assert exc_dsv4f.value.code == 2
    assert "legacy Spark prefill does not support --codebook mul1" in capsys.readouterr().err

    # 2. Dual-role families routed into Spark prefill via --rows / --prefill-arm also reject --codebook mul1.
    for fam, spark_flag in (("qwen4", ["--rows", "256"]), ("glmf", ["--rows", "256"]), ("glmf", ["--prefill-arm", "coop8:8:cooperative"])):
        with pytest.raises(SystemExit) as exc_spark:
            bench.main(["--prefill-family", fam, *spark_flag, "--codebook", "mul1", "--output", str(out_file)])
        assert exc_spark.value.code == 2
        assert "legacy Spark prefill does not support --codebook mul1" in capsys.readouterr().err

    # 3. Direct entry into _run_prefill_benchmark also refuses --codebook mul1 explicitly.
    direct_args = bench.benchmark_parser().parse_args(
        ["--prefill-family", "qwen4", "--codebook", "mul1", "--output", str(out_file)]
    )
    with pytest.raises(ValueError, match="legacy Spark prefill does not support --codebook mul1"):
        bench._run_prefill_benchmark(direct_args, None, SimpleNamespace(major=12, minor=1))

    # 4. Coordinator preflight dispatch still accepts --codebook mul1 cleanly without CUDA.
    bench.main([
        "--prefill-family", "glm", "--role", "coordinator",
        "--codebook", "mul1", "--preflight-only", "--output", str(out_file),
    ])
    preflight = bench.json.loads(capsys.readouterr().out)
    assert preflight["preflight"] is True

    # 5. Hy4-flash geometry is registered with latent H=1536, I=1536, E=512, K=16, swiglu_limit=10.
    assert bench.PREFILL_FAMILIES["hy4f"] == (1536, 512, 16, (4, 5), "silu")
    assert bench.PREFILL_SWIGLU_LIMITS["hy4f"] == 10.0
    assert bench.PREFILL_TIER_COUNTS["hy4f"] == (512, 0)
    spec = bench.COORDINATOR_FAMILIES["hy4f"]
    assert (spec["hidden"], spec["experts"], spec["top_k"], spec["bits"], spec["activation"], spec["swiglu_limit"]) == (
        1536, 512, 16, (4, 5), "silu", 10.0,
    )
    assert spec["profiles"]["rtx-tp1"] == 1536
    assert spec["profiles"]["rtx-tp2"] == 768
    assert spec["profiles"]["spark-tp1-width1536"] == 1536
    assert spec["profiles"]["spark-tp2-width768"] == 768
    assert spec["profiles"]["spark-tp4-width384"] == 384
    assert spec["profiles"]["spark-tp6-width256"] == 256
    # Hy4-flash records no realistic route fixture (routes=None), so its
    # coordinator preflight uses uniform random routes only; the default
    # random,realistic pair must keep failing closed rather than invent one.
    with pytest.raises(SystemExit) as exc_hy4_realistic:
        bench.main([
            "--prefill-family", "hy4f", "--role", "coordinator",
            "--codebook", "mul1", "--preflight-only", "--output", str(out_file),
        ])
    assert exc_hy4_realistic.value.code == 2
    assert "no realistic route fixture for coordinator family 'hy4f'" in capsys.readouterr().err
    bench.main([
        "--prefill-family", "hy4f", "--role", "coordinator", "--route-distributions", "random",
        "--codebook", "mul1", "--preflight-only", "--output", str(out_file),
    ])
    hy4_preflight = bench.json.loads(capsys.readouterr().out)
    assert hy4_preflight["preflight"] is True
    assert hy4_preflight["distributions"] == ["random"]
    assert hy4_preflight["histogram_waves"] == 0



