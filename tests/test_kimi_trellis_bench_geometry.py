"""CPU-only geometry and protocol tests for the Kimi K3 Trellis harness.

These tests never need a GPU. They freeze the numbers the harness is built on:
the trace-calibrated distinct-expert curve, the engine's AOT compile options and
runtime scalars, the canonical route-packing geometry (checked against b12x's
own host helper when the fork source is importable), the row policy for extra
decode capacities, the fixture generator, and the share-weighted ship-bar
arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "benchmarks" / "bench_kimi_trellis_decode.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_harness():
    spec = importlib.util.spec_from_file_location("kimi_trellis_bench", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = _load_harness()

# Content hash of the generated fixtures; the duel stages this exact manifest
# on both seeds, and cross-seed digest equality depends on it.
FIXTURES_SHA256 = "8a73a66bfc8e00aa747ed09c6d5a2d2cc2501c01bd63069809045acc92e6553f"


def test_frozen_blobs_are_the_pin_literals():
    assert h.PIN_COMMIT == "26a1e5072fb707864845b88db65ddf1e80c5c872"
    assert h.FROZEN_BLOBS["b12x/moe/_shared/kernels/w4a16/mixed_trellis.py"] == (
        "27281fe21837a68caf46c41b2cd3df8f4c008f33781d4c68ae20489cff161fbb")
    assert h.FROZEN_BLOBS["b12x/moe/_shared/kernels/w4a16/route_pack.py"] == (
        "ef50f695bfc4a5960a5300bbae9301cb86da9ddcb64a904d540d22f8907e4bb6")
    assert len(h.FROZEN_BLOBS) == 7
    for digest in h.FROZEN_BLOBS.values():
        assert len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def test_frozen_geometry_literals():
    assert (h.HIDDEN, h.INTERMEDIATE, h.FULL_INTERMEDIATE, h.EXPERTS, h.TOPK) == (
        3584, 512, 3072, 896, 16)
    assert h.TIER_BITS == (2, 3)
    assert h.ROUTE_BLOCK == 8 and h.SMS == 48 and h.WORKSPACE_SMS == 48
    assert h.ACTIVATION == "situ" and h.SWIGLU_LIMIT is None
    assert h.EXPERT_BYTES == 1_376_256
    assert h.EXACTNESS_ROWS == tuple(range(1, 81))
    assert h.TIMING_ROWS == (1, 5, 6, 31, 32, 33, 41, 72)
    assert h.checkpoint_source_shapes(2)["w1"] == (224, 192, 32)
    assert h.checkpoint_source_shapes(2)["w2"] == (192, 224, 32)
    assert h.checkpoint_rank_shapes(2)["w1"] == (224, 32, 32)
    assert h.checkpoint_rank_shapes(2)["w2"] == (32, 224, 32)
    assert h.checkpoint_rank_shapes(3)["w1"] == (224, 32, 48)
    assert h.empty_tier_plane_elements(3) == 32 * 224 * 24 == 172_032
    assert h.empty_tier_plane_elements(2) == 32 * 224 * 16 == 114_688


def test_trace_table_matches_the_saved_profile():
    assert len(h.TRACE_ROWS) == 30
    assert h.TRACE_ROWS[0] == (1, 160, 0.156)
    assert h.TRACE_ROWS[-1] == (72, 1360, 2.457)
    assert h.TRACE_META["m80_share_pct"] == 96.792
    assert h.TRACE_META["m1_share_pct"] == 0.146
    assert h.TRACE_META["combined_share_pct"] == 96.938
    total = sum(h.TIMING_SHARES_PCT.values())
    assert total == pytest.approx(73.728)
    assert h.TIMING_SHARES_PCT[72] == 19.544


def test_per_expert_and_roofline_are_nominal_estimates():
    assert h.per_expert_ms() == pytest.approx(1_376_256 / 273e6, rel=1e-12)
    assert h.per_expert_ms() == pytest.approx(0.00504123077, rel=1e-9)
    assert h.roofline_ms(16) == pytest.approx(16 * 0.00504123077, rel=1e-9)
    assert h.roofline_ms(487) == pytest.approx(2.45508, rel=1e-4)
    assert h.roofline_ms(0) == 0.0


def test_derived_curve_matches_the_trace_fit():
    curve = h.derive_distinct_curve()
    expected = {1: 16, 5: 77, 6: 87, 31: 269, 32: 269, 33: 269,
                41: 269, 72: 487, 80: 547}
    for row, value in expected.items():
        assert curve[row] == value, row
    values = [curve[row] for row in range(1, 81)]
    assert values == sorted(values)
    for row in range(1, 81):
        assert 16 <= curve[row] <= min(16 * row, h.EXPERTS)
    # Extrapolation past the last observed row keeps the fitted slope.
    assert curve[80] == 547
    assert curve[80] > curve[72] > curve[47]


def test_route_slots_match_b12x_host():
    host = pytest.importorskip("b12x.moe._shared.kernels.w4a16.host")
    for capacity in h.EXACTNESS_ROWS:
        expected = host.route_pack_capacity(
            capacity * h.TOPK, h.ROUTE_BLOCK, h.EXPERTS, topk=h.TOPK)
        assert h.route_slots_for_capacity(capacity) == expected
    assert h.route_slots_for_capacity(1) == (16, 128, 16)
    assert h.route_slots_for_capacity(80) == (2048, 8320, 1040)
    with pytest.raises(ValueError):
        h.route_slots_for_capacity(0)


def test_planner_tile_matches_b12x():
    impl = pytest.importorskip("b12x.moe.fused_moe._impl")
    for capacity in h.EXACTNESS_ROWS:
        expected = impl._projection_mixed_tile_config(
            None, hidden_size=h.HIDDEN, intermediate_size=h.INTERMEDIATE,
            token_count=capacity, direct_topk_routes=False)
        assert h.planner_tile(capacity) == expected, capacity
    assert h.planner_tile(16) == (128, 128, 128, 128)
    assert h.planner_tile(80) == (64, 256, 64, 256)
    assert h.planner_tile(80, configured=(64, 128, 64, 128)) == (64, 128, 64, 128)


def test_compile_options_are_the_engine_export():
    caps = h.normalize_capacities()
    entry80 = h.capacity_entry(caps, 80)
    options = h.compile_options(80, entry80)
    assert options["size_m"] == 80
    assert options["hidden_size"] == 3584
    assert options["intermediate_size"] == 512
    assert options["tier0_num_experts"] == options["tier1_num_experts"] == 896
    assert options["route_num_experts"] == 896
    assert options["top_k"] == 16
    assert options["max_m_blocks"] == 1040
    assert options["sms"] == options["workspace_sms"] == 48
    assert options["force_tile_config"] == (64, 256, 64, 256)
    assert options["tier0_bits"] == 2 and options["tier1_bits"] == 3
    assert options["trellis_codebook"] == "mcg"
    assert options["swiglu_limit"] is None
    assert options["activation"] == "situ"
    assert options["moe_block_size"] == 8
    assert options["rotation_input_dtype"] == "bf16"
    assert options["full_rotation_output_dtype"] == "bf16"
    assert options["route_ids_dtype"] == "torch.int32"
    assert options["direct_topk_routes"] is False
    assert options["force_blocks_per_sm"] is None
    assert options["decode_schedule"] == "l2=2,pf1=0,pf2=0,pdl=1"
    assert options["route_slots"] == 8320
    entry1 = h.capacity_entry(caps, 1)
    one = h.compile_options(1, entry1)
    assert one["max_m_blocks"] == 16 and one["route_slots"] == 128
    assert one["force_tile_config"] == (64, 256, 64, 256)


def test_engine_scalars_are_the_rust_executor_values():
    caps = h.normalize_capacities()
    for capacity, expected in (
        (1, {"tier0_num_experts": 896, "tier1_num_experts": 896,
             "tier0_fc2": 896, "tier1_fc2": 0,
             "tier0_gate": 896, "tier1_gate": 0,
             "tier0_up": 896, "tier1_up": 0,
             "route_num_experts": 896, "grid_x": 48}),
        (80, {"tier0_num_experts": 896, "tier1_num_experts": 896,
              "tier0_fc2": 896, "tier1_fc2": 0,
              "tier0_gate": 896, "tier1_gate": 0,
              "tier0_up": 896, "tier1_up": 0,
              "route_num_experts": 896, "grid_x": 48}),
    ):
        assert h.engine_scalars(capacity, h.capacity_entry(caps, capacity)) == expected


def test_default_capacities_and_row_policy():
    caps = h.normalize_capacities()
    assert [entry["capacity"] for entry in caps] == [1, 80]
    assert h.capacity_for_rows(1, caps) == 1
    assert h.capacity_for_rows(2, caps) == 80
    assert h.capacity_for_rows(80, caps) == 80
    for entry in caps:
        assert entry["tile"] == (64, 256, 64, 256)
        assert entry["tile_source"] == "planner"
        assert entry["blocks_per_sm"] == 1
        assert entry["decode_schedule"] == "l2=2,pf1=0,pf2=0,pdl=1"
    with pytest.raises(ValueError):
        h.capacity_for_rows(0, caps)
    with pytest.raises(ValueError):
        h.capacity_for_rows(81, caps)
    with pytest.raises(ValueError):
        h.normalize_capacities([{"capacity": 80}])
    with pytest.raises(ValueError):
        h.normalize_capacities([{"capacity": 1}, {"capacity": 1}, {"capacity": 80}])
    with pytest.raises(ValueError):
        h.normalize_capacities([{"capacity": 0}, {"capacity": 1}, {"capacity": 80}])


def test_capacity_variants_are_allowed_with_disclosure():
    caps = h.normalize_capacities(
        [{"capacity": 1}, {"capacity": 32}, {"capacity": 80}])
    assert h.capacity_for_rows(2, caps) == 32
    assert h.capacity_for_rows(31, caps) == 32
    assert h.capacity_for_rows(32, caps) == 32
    assert h.capacity_for_rows(33, caps) == 80
    assert h.capacity_for_rows(1, caps) == 1
    entry32 = h.capacity_entry(caps, 32)
    assert entry32["tile"] == (128, 128, 128, 128)
    assert entry32["tile_source"] == "planner"
    disclosure = h.capacity_disclosure(caps)
    assert disclosure["installed"] == [1, 32, 80]
    assert disclosure["extra_exports"] == 1
    assert disclosure["extra_capacities"] == [32]
    m16 = h.normalize_capacities([{"capacity": 1}, {"capacity": 16}, {"capacity": 80}])
    assert h.capacity_entry(m16, 16)["package_policy"].startswith("excluded")
    with pytest.raises(ValueError):
        h.normalize_capacities(
            [{"capacity": 1, "decode_schedule": "l2=1,pf1=0,pf2=0,pdl=1"},
             {"capacity": 80}])
    with pytest.raises(ValueError):
        h.normalize_capacities(
            [{"capacity": 1, "tile": (64, 128)}, {"capacity": 80}])
    # Explicit tiles and residency must be internally consistent.
    explicit = h.normalize_capacities(
        [{"capacity": 1, "tile": (64, 128, 64, 128), "blocks_per_sm": 2,
          "force_blocks_per_sm": 2}, {"capacity": 80}])
    assert h.capacity_entry(explicit, 1)["tile"] == (64, 128, 64, 128)
    with pytest.raises(ValueError):
        h.normalize_capacities(
            [{"capacity": 1, "force_blocks_per_sm": 2}, {"capacity": 80}])


def test_output_view_contract():
    assert h.output_check_shapes(1, 1) == [
        ("capacity", (1, h.HIDDEN)), ("live-parent", (9, h.HIDDEN))]
    assert h.output_check_shapes(72, 80) == [
        ("capacity", (80, h.HIDDEN)), ("live-parent", (80, h.HIDDEN))]
    with pytest.raises(ValueError):
        h.output_check_shapes(81, 80)
    with pytest.raises(ValueError):
        h.output_check_shapes(0, 80)


def test_unique_storage_merges_aliased_fields():
    first = torch.empty(4, dtype=torch.float32)
    second = torch.empty(8, dtype=torch.int16)
    assert h.unique_storage_bytes([first, second]) == 16 + 16
    assert h.unique_storage_bytes([first, first]) == 16
    assert h.unique_storage_bytes([first, first[:3]]) == 16
    assert h.unique_storage_bytes([]) == 0


def test_fixture_generation_is_deterministic_and_trace_faithful():
    first = h.build_fixture_manifest()
    first_data = first.pop("_layer_data")
    second = h.build_fixture_manifest()
    second.pop("_layer_data")
    assert first["fixtures_sha256"] == second["fixtures_sha256"]
    curve = h.derive_distinct_curve()
    assert first["layers"] == [13, 92]
    assert first["distinct"]["values"] == {str(k): v for k, v in curve.items()}
    for layer in (13, 92):
        data = first_data[layer]
        assert data["x"].shape == (80, h.HIDDEN)
        assert data["x"].dtype == torch.bfloat16
        for row in h.EXACTNESS_ROWS:
            fixture = data["rows"][row]
            ids, weights = fixture["ids"], fixture["weights"]
            assert ids.shape == (row, h.TOPK) and ids.dtype == torch.int32
            assert weights.shape == (row, h.TOPK)
            assert weights.dtype == torch.float32
            assert fixture["distinct"] == curve[row]
            assert torch.unique(ids).numel() == curve[row]
            assert torch.unique(ids[0]).numel() == h.TOPK
            assert torch.allclose(weights.sum(dim=1), torch.ones(row))
            order = torch.argsort(weights, dim=1, descending=True, stable=True)
            assert torch.equal(order, torch.arange(h.TOPK).expand(row, h.TOPK))
            assert int(ids.min()) >= 0 and int(ids.max()) < h.EXPERTS
        # One shared input tensor whose prefixes are the per-row inputs.
        assert torch.equal(data["x"][:5],
                           h.build_layer_fixtures(layer, curve)["x"][:5])
    assert first["exactness_rows"] == list(h.EXACTNESS_ROWS)
    assert first["timing_rows"] == list(h.TIMING_ROWS)
    if FIXTURES_SHA256 is not None:
        assert first["fixtures_sha256"] == FIXTURES_SHA256


def test_fixture_verification_rejects_tampering(tmp_path):
    manifest = h.build_fixture_manifest()
    manifest.pop("_layer_data")
    path = tmp_path / "fixtures.json"
    h.write_json(path, manifest)
    loaded = h.load_json(path)
    rebuilt, curve = h.verify_fixture_manifest(loaded)
    assert curve == h.derive_distinct_curve()
    assert sorted(rebuilt) == [13, 92]
    edited = json.loads(json.dumps(loaded))
    edited["distinct"]["values"]["72"] += 1
    with pytest.raises(ValueError):
        h.verify_fixture_manifest(edited)
    stale = json.loads(json.dumps(loaded))
    stale["fixtures_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        h.verify_fixture_manifest(stale)
    wrong_pin = json.loads(json.dumps(loaded))
    wrong_pin["pin"]["fork_commit"] = "deadbeef"
    with pytest.raises(ValueError):
        h.verify_fixture_manifest(wrong_pin)
    wrong_schema = json.loads(json.dumps(loaded))
    wrong_schema["schema"] = "something-else"
    with pytest.raises(ValueError):
        h.verify_fixture_manifest(wrong_schema)


def test_timing_weights_use_trace_shares():
    weights = h.timing_weights()
    assert set(weights) == set(h.TIMING_ROWS)
    assert sum(weights.values()) == pytest.approx(1.0)
    total = sum(h.TIMING_SHARES_PCT.values())
    for row in h.TIMING_ROWS:
        assert weights[row] == pytest.approx(h.TIMING_SHARES_PCT[row] / total)
    assert weights[72] == pytest.approx(0.26509, rel=1e-4)
    assert weights[41] == pytest.approx(0.25088, rel=1e-4)


def test_fixture_sample_stats_use_medians_and_duplicate_spread():
    samples = {"l13/r1": {"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0, 2.0],
                          "capacity": 1, "layer": 13, "row": 1}}
    stats = h.fixture_sample_stats(samples)
    item = stats["l13/r1"]
    assert item["a_ms"] == 2.0 and item["b_ms"] == 2.0
    assert item["median_ms"] == 2.0 and item["dup_spread"] == 0.0
    assert item["samples"] == 6 and item["capacity"] == 1


def _uniform_stats(rows=h.TIMING_ROWS, layers=(13, 92), ms=1.0, spread=0.0):
    stats = {}
    for layer in layers:
        for row in rows:
            stats[f"l{layer}/r{row}"] = {"median_ms": ms, "dup_spread": spread,
                                         "capacity": 1 if row == 1 else 80}
    return stats


def test_compare_arm_share_weighted_bar_and_regressions():
    oracle = _uniform_stats()
    candidate = _uniform_stats(ms=1.0 / 1.2, spread=0.005)
    result = h.compare_arm(oracle, candidate)
    assert result["ship_bar_met"] is True
    assert result["layers"]["13"]["weighted_speedup"] == pytest.approx(1.2)
    assert result["layers"]["92"]["bar_met"] is True
    assert result["combined_weighted_speedup"] == pytest.approx(1.2)
    assert result["layers"]["13"]["regressions"] == []
    spot = "l13/r41"
    regressed = _uniform_stats(ms=1.0 / 1.2, spread=0.005)
    regressed[spot]["median_ms"] = 1.05
    verdict = h.compare_arm(oracle, regressed)
    assert verdict["layers"]["13"]["bar_met"] is False
    assert spot in verdict["layers"]["13"]["regressions"]
    assert verdict["layers"]["92"]["bar_met"] is True
    assert verdict["ship_bar_met"] is False
    # A dip inside duplicate-control noise is tolerated.
    noisy = _uniform_stats(ms=1.0 / 1.2, spread=0.05)
    noisy[spot]["median_ms"] = (1.0 / 1.2) * 1.02
    assert h.compare_arm(oracle, noisy)["layers"]["13"]["regressions"] == []
    # Below the bar even without any regression.
    slow = _uniform_stats(ms=1.0 / 1.05)
    assert h.compare_arm(oracle, slow)["ship_bar_met"] is False
    # Ratios are per fixture, so a layer-position difference is reported, not
    # averaged away.
    uneven = _uniform_stats(ms=1.0 / 1.2)
    uneven["l13/r72"]["median_ms"] = 1.0 / 1.6
    uneven["l13/r41"]["median_ms"] = 1.0 / 1.6
    verdict = h.compare_arm(oracle, uneven)
    ratios = {item["fixture"]: item["ratio"]
              for item in verdict["layers"]["13"]["fixtures"]}
    assert ratios["l13/r72"] == pytest.approx(1.6)
    assert ratios["l13/r1"] == pytest.approx(1.2)


def test_compare_arm_requires_every_timed_fixture():
    oracle = _uniform_stats()
    incomplete = _uniform_stats()
    incomplete.pop("l92/r33")
    with pytest.raises(KeyError):
        h.compare_arm(oracle, incomplete)


def test_file_hashes_reports_missing_files(tmp_path):
    hashes = h.file_hashes(tmp_path)
    assert set(hashes) == set(h.FROZEN_BLOBS)
    assert all(value is None for value in hashes.values())


def test_stage_requires_a_hf_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        h.stage_checkpoint(tmp_path, tmp_path / "out")


def test_cli_exposes_the_harness_surface():
    parser = h.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"fixtures", "stage", "freeze-oracle", "run",
                            "serve", "session"}


def test_fixtures_command_writes_a_portable_manifest(tmp_path, capsys):
    out = tmp_path / "fixtures.json"
    h.main(["fixtures", "--out", str(out)])
    capsys.readouterr()
    manifest = h.load_json(out)
    assert "_layer_data" not in manifest
    assert manifest["fixtures_sha256"] == h.build_fixture_manifest()["fixtures_sha256"]
    rebuilt, curve = h.verify_fixture_manifest(manifest)
    assert sorted(rebuilt) == [13, 92]
    assert curve[72] == 487
