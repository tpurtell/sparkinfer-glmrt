"""CPU-only controls for fixed-export row sweeps and recorded route replay."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PATH = Path(__file__).resolve().parents[2] / "benchmarks/bench_cuteafd_fp8_moe.py"
_SPEC = importlib.util.spec_from_file_location("fp8_moe_benchmark", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


@pytest.mark.parametrize("precision", ["a8", "a16"])
@pytest.mark.parametrize("width", [32, 64, 128, 192])
@pytest.mark.parametrize("cta", [1, 2, 4, 8])
def test_cross_cta_scratch_includes_every_raw_projection(precision, width, cta):
    path = _PATH.parents[1] / "b12x/moe/_shared/kernels/v41_slice_pipeline.py"
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "fc1_partial_words")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), str(path), "exec"), namespace)
    compact = width // 2 if precision == "a16" else width // 4 + width // 32
    planes = (288 + width - 1) // width
    expected = planes * 80 * 8 * (compact + (2 * width * cta if cta > 1 else 0))
    assert namespace["fc1_partial_words"](80, width, 8, 288, precision, cta) == expected


def test_mxfp4_pair_widths_cover_all_worlds_without_padding(monkeypatch):
    monkeypatch.setattr(bench, "engine_layout_policy", lambda root: SimpleNamespace(
        ownership_unit=lambda intermediate, world: 64 if (intermediate, world) == (2304, 5) else 32))
    assert bench.paired_widths(2304) == [288, 320, 352, 384, 448, 512, 576, 768, 1152]
    assert bench.paired_widths(2048) == [256, 288, 320, 352, 384, 416, 512, 672, 704, 1024]
    assert bench.paired_widths(3072) == [384, 416, 448, 512, 608, 640, 768, 1024, 1536]
    args = bench.parse_args(["--mxfp4-pair", "--native-slices", "--geometry", "v41", "--widths", "448,480"])
    assert args.native_slices and args.widths == "448,480"
    with pytest.raises(ValueError):
        bench.paired_widths(2048, "2080")


@pytest.mark.parametrize("family", ["mimof", "mimop"])
def test_unify_screen_is_mimo_pair_only(family):
    args = bench.parse_args(["--mxfp4-pair", "--mxfp4-unify", "--geometry", family,
                             "--widths", "256,288", "--graph"])
    assert args.mxfp4_unify and args.input == "wire"
    for extra in (["--native-slices"], ["--mxfp4-tilings"]):
        with pytest.raises(SystemExit):
            bench.parse_args(["--mxfp4-pair", "--mxfp4-unify", "--geometry", family, *extra])
    with pytest.raises(SystemExit):
        bench.parse_args(["--mxfp4-unify", "--geometry", family])


def test_cross_cta_screen_and_profile_share_explicit_geometry(tmp_path):
    args = bench.parse_args(["--mxfp4-pair", "--mxfp4-unify", "--geometry", "mimof",
                             "--fc1-k-splits", "1,2", "--fc1-cta-splits", "2,4,8",
                             "--profile-dir", str(tmp_path), "--profile-arm", "native-a8-t64-k2-z4"])
    assert args.fc1_cta_splits == "2,4,8"
    for argv in (["--fc1-cta-splits", "2"],
                 ["--mxfp4-pair", "--mxfp4-unify", "--geometry", "mimof", "--fc1-cta-splits", "3"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


def test_fc1_k_split_screen_geometry_and_cli():
    args = bench.parse_args(["--mxfp4-pair", "--mxfp4-unify", "--geometry", "mimof",
                             "--fc1-k-splits", "2,4,8"])
    assert args.fc1_k_splits == "2,4,8"
    arms = bench.fc1_split_tiles(5120, [2, 4, 8])
    assert (128, 2) in arms and (128, 4) in arms and (64, 8) in arms
    assert (128, 8) not in arms and (192, 4) not in arms
    assert bench.fc1_split_tiles(4096, iter([2, 4])) == bench.fc1_split_tiles(4096, [2, 4])
    for values in ("", "0", "3", "2,2", "x"):
        with pytest.raises(SystemExit):
            bench.parse_args(["--mxfp4-pair", "--native-slices", "--fc1-k-splits", values])
    with pytest.raises(SystemExit):
        bench.parse_args(["--mxfp4-pair", "--fc1-k-splits", "2"])


@pytest.mark.parametrize("family,rank_width,expected", [
    ("v41", 288, 192), ("v41", 576, 192), ("v41", 768, 192),
    ("v41", 448, 128), ("dsv4f", 384, 128), ("dsv4p", 768, 128),
])
def test_native_pair_retains_shipped_slice_recipe(family, rank_width, expected):
    assert bench.native_slice_width(family, rank_width) == expected


@pytest.mark.parametrize("argv", [["--native-slices"], ["--widths", "448"],
    ["--mxfp4-pair", "--widths", "33"], ["--mxfp4-pair", "--widths", "oops"],
    ["--mxfp4-pair", "--input", "bf16"], ["--mxfp4-pair", "--profile-dir", "out"]])
def test_invalid_pair_modes_fail_before_cuda(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


@pytest.mark.parametrize("family,arm,mode", [("mimof", "current", "--mxfp4-unify"),
    ("mimof", "native-a8", "--mxfp4-unify"),
    ("mimof", "native-a8-t64-k2", "--mxfp4-unify"), ("v41", "native-exact", "--native-slices")])
def test_paired_profile_selects_one_warmed_arm(tmp_path, monkeypatch, family, arm, mode):
    args = bench.parse_args(["--mxfp4-pair", mode, "--geometry", family, "--rows", "1",
                             "--profile-dir", str(tmp_path), "--profile-arm", arm,
                             *(["--fc1-k-splits", "2"] if "-t64-k2" in arm else [])])
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(name))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))
    bench.profile_pair_arm({arm: lambda: events.append(arm)}, args, family, 256, 1)
    assert events == [arm, "sync", "mxfp4-pair", arm, "sync", "pop"]
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["arm"] == arm and record["warmed_launches"] == 1 and record["attribution_only"]
    with pytest.raises(SystemExit):
        bench.parse_args(["--profile-arm", arm])


@pytest.mark.parametrize("width,expected", [(288, [1, 2, 3, 4, 8]), (576, [1, 2, 4, 5, 8]),
                                          (672, [1, 2, 4, 6, 8]), (608, [1, 2, 4, 5, 8])])
def test_tiling_screen_includes_old_four_warps_and_matching_partition(width, expected):
    assert bench.mxfp4_decode_partitions(width) == expected


def test_tiling_controls_are_development_pair_only():
    assert bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings"]).mxfp4_tilings
    for argv in (["--mxfp4-tilings"], ["--mxfp4-pair", "--mxfp4-tilings", "--capacity", "4096"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


@pytest.mark.parametrize("width", [288, 576])
def test_aligned_w2_changes_only_payload_tile_pitch(monkeypatch, width):
    from tests.moe.test_v41_exact_slices import pack
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    experts, hidden = 2, 256
    weights, scales = {}, {}
    for name, shape in (("w1", (experts, width, hidden // 2)),
                        ("w3", (experts, width, hidden // 2)),
                        ("w2", (experts, hidden, width // 2))):
        weights[name] = (torch.arange(shape[0] * shape[1] * shape[2]) % 251 + 1).byte().reshape(shape)
        scales[name] = torch.full((*shape[:-1], shape[-1] // 16), 122, dtype=torch.uint8)
    pitch = (width + 127) // 128 * 128
    exact = pack(weights, scales, True)
    aligned = pack(weights, scales, True, w2_pitch=pitch)
    for plane in (0, 1, 3):
        assert torch.equal(exact[plane], aligned[plane])
    expected = exact[2].reshape(experts, hidden // 128, width // 32, 512)
    actual = aligned[2].reshape(experts, hidden // 128, pitch // 32, 512)
    assert torch.equal(actual[:, :, :width // 32], expected)
    assert torch.count_nonzero(actual[:, :, width // 32:].view(torch.int32)) == 0


@pytest.mark.parametrize("width", [288, 576])
def test_w2_coded_pitch_is_already_aligned_and_extension_preserves_bytes(width):
    from b12x.moe._shared.scale_codes import pack_scale_codes
    columns = width // 32
    source = (torch.arange(256 * columns).reshape(1, 256, columns) % 3 + 121).byte()
    aligned_columns = (columns + 3) // 4 * 4
    extension = source.amin(-1, keepdim=True).expand(1, 256, aligned_columns - columns)
    exact = pack_scale_codes(source, tensor_name="w2.scale")
    aligned = pack_scale_codes(torch.cat((source, extension), -1), tensor_name="w2.scale")
    assert exact.shape == (1, 256 * (2 * ((columns + 3) // 4) + 1))
    assert torch.equal(exact, aligned)


def test_w2_pitch_arm_is_native_development_only():
    assert bench.parse_args(["--mxfp4-pair", "--mxfp4-tilings", "--native-slices",
                             "--mxfp4-w2-pitch"]).mxfp4_w2_pitch
    for argv in (["--mxfp4-w2-pitch"], ["--mxfp4-pair", "--mxfp4-tilings", "--mxfp4-w2-pitch"],
                 ["--mxfp4-pair", "--native-slices", "--mxfp4-w2-pitch"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


def test_f32_errors_measure_unrounded_reference():
    error = bench.f32_errors(torch.tensor([1., 3.]), torch.tensor([2., 5.]))
    assert error == pytest.approx(dict(max_abs=2., rms=(2.5 ** .5)))


@pytest.mark.parametrize("native", [False, True])
def test_mxfp4_f32_oracle_has_explicit_quantization_boundaries(native):
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows
    x = torch.full((1, 32), .25, dtype=torch.bfloat16)
    ids, routing = torch.zeros((1, 1), dtype=torch.int32), torch.full((1, 1), .5)
    weights = {name: torch.full((1, 32, 16), 0x22, dtype=torch.uint8) for name in ("w1", "w3", "w2")}
    scales = {name: torch.full((1, 32, 1), 126, dtype=torch.uint8) for name in weights}
    output = bench.mxfp4_f32_reference(x, ids, routing, weights, scales, 10. if native else 0., native=native)
    gate = torch.full((1, 32), 4.)
    if native:
        act = (torch.nn.functional.silu(gate) * gate * .5).bfloat16()
        _, act = wire_rows(act)
        expected = act.float().sum() * .5
    else:
        act = (torch.nn.functional.silu(gate).bfloat16().float() * gate).bfloat16()
        expected = act.float().sum() * .5 * .5
    assert torch.equal(output, torch.full((1, 32), expected.item()))


def test_mimo_a8_oracle_quantizes_after_bf16_silu_product():
    from tests.integration.test_cuteafd_fp8_moe_aot import wire_rows
    x = torch.linspace(-.5, .5, 32).reshape(1, 32).bfloat16()
    ids, routing = torch.zeros((1, 1), dtype=torch.int32), torch.full((1, 1), .5)
    generator = torch.Generator().manual_seed(1608)
    weights = {name: torch.randint(0, 256, (1, 32, 16), dtype=torch.uint8, generator=generator)
               for name in ("w1", "w3", "w2")}
    scales = {name: torch.full((1, 32, 1), 124, dtype=torch.uint8) for name in weights}
    output = bench.mxfp4_f32_reference(x, ids, routing, weights, scales, activation_precision="a8")
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    def matrix(name):
        packed = weights[name][0]
        return torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1).flatten(-2) / 8
    qx = wire_rows(x)[1].float()
    gate, up = (qx @ matrix("w1").T).bfloat16(), (qx @ matrix("w3").T).bfloat16()
    mid = (torch.nn.functional.silu(gate).float() * up.float()).bfloat16()
    expected = wire_rows(mid)[1].float() @ matrix("w2").T * .5
    assert torch.equal(output, expected)


def test_capacity_is_independent_of_live_rows():
    args = bench.parse_args(["--capacity", "4096", "--rows", "512,2048,2752,4096"])
    assert args.capacity == 4096
    assert args.rows == "512,2048,2752,4096"
    assert bench.parse_args([]).capacity is None


@pytest.mark.parametrize("argv", [
    ["--capacity", "0"], ["--capacity", "2048", "--rows", "2752"],
    ["--rows", "0"], ["--rows", "-1"], ["--rows", ""], ["--rows", "1,broken"],
    ["--routes", "routes.json", "--routing", "hot"],
    ["--capacity", "4096", "--exact-sweep", "out"],
    ["--routes", "routes.json", "--exact-sweep", "out"],
    ["--route", "auto,auto"], ["--route", "auto,"],
    ["--compare-routes-exact"],
    ["--continue-on-fail"],
    ["--check-prefix-rows", "2048"],
    ["--check", "--check-prefix-rows", "0", "--capacity", "4096"],
    ["--check", "--check-prefix-rows", "4096", "--capacity", "4096"],
    ["--continue-on-fail", "--check", "--exact-sweep", "out"],
    ["--compare-routes-exact", "--route", "decode,auto", "--exact-sweep", "out"],
])
def test_invalid_cases_fail_before_cuda(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


def test_fixed_capacity_reuses_compiled_callable_and_workspace(monkeypatch):
    from b12x.integration.cuteafd import fp8_moe

    calls = []
    def compile_program(g, **kwargs):
        calls.append(("compile", kwargs))
        return object()
    def scratch_bytes(g, route, capacity, wire, prefill):
        calls.append(("scratch", capacity))
        return capacity
    monkeypatch.setattr(fp8_moe, "compile_fp8_moe_aot", compile_program)
    monkeypatch.setattr(fp8_moe, "fp8_moe_scratch_bytes", scratch_bytes)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"}))
    args = bench.parse_args(["--capacity", "4096", "--rows", "2048,2752"])
    cache = {}
    first = bench.compiled_case(None, args, 2048, "auto", cache)
    second = bench.compiled_case(None, args, 2752, "auto", cache)
    assert first is second
    assert first[1].numel() == 4096
    assert calls == [("compile", dict(route="auto", max_rows=4096, wire=True, prefill="auto")),
                     ("scratch", 4096)]
    args.capacity = None
    bench.compiled_case(None, args, 2048, "auto", cache)
    assert calls[-2][1]["max_rows"] == 2048


def test_exact_comparison_reports_raw_bytes_and_first_element():
    reference = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    changed = reference.clone()
    changed[1, 0] = 3.5
    result = bench.compare_route_outputs(dict(decode=reference, auto=reference.clone(), prefill=changed))
    assert result["auto"] == dict(reference_route="decode", bitwise_equal=True,
                                  first_mismatch=None, max_abs_diff=0.)
    assert not result["prefill"]["bitwise_equal"]
    assert result["prefill"]["first_mismatch"]["element_index"] == [1, 0]
    assert result["prefill"]["max_abs_diff"] == .5
    assert bench.parse_args(["--compare-routes-exact", "--route", "decode,auto"]).compare_routes_exact


def test_exact_comparison_distinguishes_signed_zero():
    result = bench.compare_route_outputs(dict(a=torch.tensor([[0.]], dtype=torch.bfloat16),
                                             b=torch.tensor([[-0.]], dtype=torch.bfloat16)))
    assert not result["b"]["bitwise_equal"]
    assert result["b"]["max_abs_diff"] == 0.


@pytest.mark.parametrize("actual", [torch.ones(1, 3), torch.ones(2, 2, dtype=torch.bfloat16)])
def test_exact_comparison_rejects_output_contract_mismatch(actual):
    with pytest.raises(ValueError, match="shapes and dtypes"):
        bench.compare_route_outputs(dict(a=torch.ones(2, 2), b=actual))


@pytest.mark.parametrize("changed", [False, True])
def test_shared_prefix_compares_snapshot_before_scratch_reuse(changed):
    full = torch.arange(12).reshape(4, 3).to(torch.bfloat16)
    prefix = full[:3].clone()
    if changed:
        prefix[1, 2] += 1
    def execute(rows):
        assert rows == 3
        full.fill_(99)
        return prefix
    result = bench.check_shared_prefix(full, 3, execute)
    assert result["bitwise_equal"] == (not changed)
    assert result["differing_rows"] == int(changed)
    assert (result["prefix_sha256"] == result["full_prefix_sha256"]) == (not changed)
    if changed:
        assert result["first_mismatch"]["element_index"] == [1, 2]
    assert bench.parse_args(["--check", "--capacity", "4096", "--check-prefix-rows", "2048"]).check_prefix_rows == 2048


def test_oracle_reports_global_rms_and_outlier_location():
    reference = torch.tensor([[0., 1.], [1., 1.]])
    output = reference.clone()
    output[0, 0] = .2
    result = bench.oracle_metrics(output, reference, nvfp4_a4=True)
    assert result["reference_rms"] == pytest.approx(3 ** .5 / 2)
    assert result["max_error"] == pytest.approx(dict(row=0, col=0, output=.2, reference=0., abs_error=.2))
    assert result["normalized_max"] == pytest.approx(.2 / (3 ** .5 / 2))
    assert result["elements_over_threshold"] == 1
    assert not result["passed"]
    with pytest.raises(AssertionError):
        bench.enforce_oracle(result, False)
    bench.enforce_oracle(result, True)
    assert bench.parse_args(["--check", "--continue-on-fail"]).continue_on_fail


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_oracle_continuation_never_turns_invalid_outputs_green(value):
    result = bench.oracle_metrics(torch.full((2, 2), value), torch.ones(2, 2), nvfp4_a4=True)
    assert not result["passed"]
    bench.enforce_oracle(result, True)


def test_oracle_equal_nonzero_output_passes():
    result = bench.oracle_metrics(torch.ones(2, 2), torch.ones(2, 2), nvfp4_a4=True)
    assert result["passed"] and result["elements_over_threshold"] == 0
    bench.enforce_oracle(result, False)


def fixture(tmp_path, **updates):
    data = dict(schema=1, family="glm", model="zai-org/GLM-5.3", experts=4, top_k=2,
                waves=[dict(layer=7, lane=0, rows=4, expert_rows=[[0, 4], [1, 2], [2, 2]])])
    data.update(updates)
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(data))
    return path


def test_reuses_exlr8_loader_and_histogram_reconstruction(tmp_path):
    from benchmarks.benchmark_exlr8_a16 import real_route_ids

    geometry = SimpleNamespace(name="glm_nvfp4", experts=4, top_k=2)
    waves = bench.recorded_waves(fixture(tmp_path), geometry)
    ids = real_route_ids(4, waves[0], torch.Generator().manual_seed(5), 4, 2)
    assert torch.bincount(ids.flatten(), minlength=4).tolist() == [4, 2, 2, 0]
    assert bool((ids[:, 0] != ids[:, 1]).all())
    replay = real_route_ids(4, waves[0], torch.Generator().manual_seed(5), 4, 2)
    assert torch.equal(ids, replay)


@pytest.mark.parametrize("updates", [
    dict(family="step5"), dict(experts=5), dict(top_k=3), dict(schema=2), dict(waves=[]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 4], [0, 4]])]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 5], [1, 3]])]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 4], [4, 4]])]),
    dict(waves=[dict(rows=4, expert_rows=[[0, 4]])]),
])
def test_wrong_family_or_malformed_histograms_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        bench.recorded_waves(fixture(tmp_path, **updates), SimpleNamespace(name="glm", experts=4, top_k=2))


def test_legacy_glm_fixture_is_not_relabelled_for_other_families():
    path = _PATH.parent / "fixtures/glm53_prefill_routes_realistic.json"
    assert bench.recorded_waves(path, SimpleNamespace(name="glm", experts=256, top_k=8))
    with pytest.raises(ValueError, match="family"):
        bench.recorded_waves(path, SimpleNamespace(name="mimo", experts=256, top_k=8))


@pytest.mark.parametrize("tile", [32, 64, 128, 192])
@pytest.mark.parametrize("logical", [288, 416, 576, 608])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("activation_precision", ["a8", "a16"])
def test_native_partial_allocation_has_exact_fc1_activation_extent(tile, logical, atomic, activation_precision):
    function = next(node for node in ast.parse(_PATH.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "native_pair_arm")
    assignment, = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "partial" for target in node.targets)]
    expression = compile(ast.Expression(assignment.value), str(_PATH), "eval")
    # Run the real allocation expression on CPU; CUDA dataflow is GPU-qualified separately.
    allocator = SimpleNamespace(empty=lambda shape, **kwargs: torch.empty(shape))
    words_per_route = tile // 2 if activation_precision == "a16" else tile // 4 + tile // 32
    partial_words = 1 if atomic else ((logical + tile - 1) // tile) * 96 * words_per_route
    partial = eval(expression, {"__builtins__": {}, "torch": allocator},
                   dict(partial_words=partial_words))
    route_bytes = 2 * tile if activation_precision == "a16" else tile + tile // 8
    assert partial.numel() * 4 == (4 if atomic else
        ((logical + tile - 1) // tile) * 96 * route_bytes)


def test_prefill_unify_controls_are_fixed_stream_and_record_prefix_failures():
    function = next(n for n in ast.parse(_PATH.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "mxfp4_unify_screen")
    text = ast.unparse(function)
    assert "control_route = 'stream' if rows > 80 else 'auto'" in text
    assert "strict_prefix=rows <= 80" in text
    assert "if not v4 and control_route != 'stream':" in text
    assert "fp8_moe_scratch_bytes(geometry, control_route, capacity, prefill=args.prefill)" in text
    assert "program = template" in text
    override = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "not v4 and control_route != 'stream'")
    # Stream must retain the public compiler's A8-down selection and scratch ABI.
    assert "launch = _Fp8Moe(" in ast.unparse(override)
    assert "program = compile_program(" in ast.unparse(override)
    assert "control_qualification=control_qualification" in text
    helper = next(n for n in ast.parse(_PATH.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "qualify_pair_replay")
    assert helper.args.kw_defaults[0].value is True
    assert "prefix_failures" in ast.unparse(helper)


def test_row_major_a8_is_benchmark_only_and_uses_raw_scratch():
    function = next(n for n in ast.parse(_PATH.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "native_pair_arm")
    text = ast.unparse(function)
    assert "stream_decode or row_major_a8" in text
    assert "pipeline.fc1 = RowMajorA8Fc1(" in text
    assert "pipeline.fc2 = RowMajorA8Fc2(" in text
    assert "rows > 80" in text
    args = bench.parse_args(["--geometry", "mimof", "--mxfp4-pair", "--mxfp4-unify",
                             "--row-major-a8", "--rows", "1,16,80"])
    assert args.row_major_a8
    with pytest.raises(SystemExit):
        bench.parse_args(["--geometry", "mimof", "--mxfp4-pair", "--mxfp4-unify",
                          "--row-major-a8", "--rows", "256"])


@pytest.mark.parametrize("flag,rows", [("--row-major-a8", "1,16,80"), ("--v4-stream", "256,1024,4096")])
def test_v4_contract_benchmarks_are_explicit(flag, rows):
    args = bench.parse_args(["--geometry", "v41", "--mxfp4-pair", "--mxfp4-unify", flag, "--rows", rows])
    assert args.geometry == "v41"
    with pytest.raises(SystemExit):
        bench.parse_args(["--geometry", "v41", "--mxfp4-pair", "--mxfp4-unify", "--rows", rows])
    other_rows = "256" if flag == "--row-major-a8" else "80"
    with pytest.raises(SystemExit):
        bench.parse_args(["--geometry", "v41", "--mxfp4-pair", "--mxfp4-unify", flag, "--rows", other_rows])


def test_v4_stream_adapter_has_ordered_f32_output_and_weight_before_fc2():
    path = _PATH.parents[1] / "b12x/moe/_shared/kernels/mxfp4_stream_contract.py"
    text = path.read_text()
    assert "v4=True" in text
    assert "self.gate_up.route_weight_before_fc2 = v4" in text
    assert "self.gate_up.silu_bf16_product = not v4" in text
    assert "self.down.round_output_bf16 = not v4" in text
    assert "fadd_rn_f32(value, route)" in text
    assert "route *= routing[row * self.g.top_k + slot]" in text
    assert "atomic" not in text
    helper = next(n for n in ast.parse(_PATH.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "v4_stream_pair_arm")
    assert "return self.intermediate" in ast.unparse(helper)
    assert "geometry = ExactStreamGeometry(" in ast.unparse(helper)


@pytest.mark.parametrize("family,capacity,rows", [
    ("v41", 1, "1"), ("v41", 80, "8,16,40,80"),
    ("mimof", 1, "1"), ("mimof", 16, "8,16"), ("mimop", 80, "32,48,64")])
def test_unify_fixed_capacity_live_matrix(family, capacity, rows):
    args = bench.parse_args(["--geometry", family, "--mxfp4-pair", "--mxfp4-unify",
                             "--capacity", str(capacity), "--rows", rows])
    assert args.capacity == capacity
    assert max(map(int, args.rows.split(","))) <= capacity


def test_unify_fixed_capacity_calls_preserve_live_rows():
    text = _PATH.read_text()
    cls = ast.parse(text)
    native = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "native_pair_arm")
    screen = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "mxfp4_unify_screen")
    assert "execute(live_rows)" in ast.unparse(native)
    calls = [n for n in ast.walk(screen) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "native_pair_arm"]
    assert any(any(k.arg == "live_rows" and ast.unparse(k.value) == "rows" for k in n.keywords)
               for n in calls)
    assert "max_rows=capacity" in ast.unparse(screen)
    assert "x[:rows], ids[:rows], routing[:rows]" in ast.unparse(screen)


def test_native_stream_weights_share_activation_completion_ring():
    path = _PATH.parents[1] / "b12x/integration/cuteafd/_mxfp4_moe_stream.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "StreamMxfp4GateUp")
    text = ast.unparse(cls)
    ring_defaults = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id in ("a_ring", "w_ring")}
    assert ring_defaults == {"a_ring": 3, "w_ring": 3}
    assert "self.load_native_weights(native_gate, native_up, sw, e, n0, p, tidx)" in text
    assert "step + Int32(self.w_ring - 1), tidx)" in text
    assert "native_base = sw + step % Int32(self.w_ring) * Int32(4096)" in text
    assert "shared + Int32(step) % Int32(self.w_ring) * Int32(4096)" in text


def test_stream_decode_cli_contract():
    args = bench.parse_args(["--geometry", "mimof", "--mxfp4-pair", "--mxfp4-unify",
                             "--rows", "1,16,80", "--stream-decode"])
    assert args.stream_decode
    for argv in (["--stream-decode"], ["--mxfp4-pair", "--mxfp4-unify", "--rows", "256", "--stream-decode"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


@pytest.mark.parametrize("width", [64, 128, 192])
def test_stream_decode_scratch_contract(width):
    path = _PATH.parents[1] / "b12x/moe/_shared/kernels/v41_slice_pipeline.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "fc1_partial_words")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    words = namespace["fc1_partial_words"](80, width, 8, 288, "a16", stream_decode=True)
    assert words == ((288 + width - 1) // width) * 80 * 8 * (width // 2 + 2 * width)


def test_explicit_rank_extents_and_identical_controls():
    args = bench.parse_args(["--mxfp4-pair", "--native-slices", "--mxfp4-tilings",
                             "--mxfp4-arm", "compact:288:288:128",
                             "--mxfp4-arm", "pad64:288:320:64", "--control-repeats", "3"])
    specs, controls = bench.tiling_specs(args, 288, 192, (64, 128, 192))
    assert controls == ["padded", "padded-control-1", "padded-control-2"]
    assert specs == [(name, False, 192, None) for name in controls] + [
        ("compact", True, 128, 288), ("pad64", True, 64, 320)]
    samples = {name: [1., 2., 3.] for name in controls}
    samples.update(compact=[2., 4., 6.], pad64=[4., 8., 12.])
    result = bench.tiling_timing_summary(samples, {name: True for name in samples}, controls)
    assert result["winner"] == "compact" and result["speedup"] == .5
    assert result["identical_control_speedups"] == {name: [1., 1., 1.] for name in controls[1:]}
    assert bench.tiling_timing_summary(samples, {name: False for name in samples}, controls)["winner"] is None


@pytest.mark.parametrize("value", ["bad", "a:0:128:64", "a:33:128:64", "a:128:96:64",
                                   "a:96:97:64", "a:96:96:16", "padded-x:96:96:64",
                                   "exact-64:96:96:64", "aligned-w2-x:96:96:64", "a/b:96:96:64"])
def test_invalid_explicit_extents_fail_before_cuda(value):
    with pytest.raises(SystemExit):
        bench.parse_args(["--mxfp4-pair", "--native-slices", "--mxfp4-tilings", "--mxfp4-arm", value])


def test_explicit_legacy_split_and_profile_selection():
    values = ["legacy128:384:0:128:2", "legacy192:384:0:192:2", "exact128:384:384:128"]
    assert bench.native_screen_arms(values) == [
        ("legacy128", 384, 0, 128, 2), ("legacy192", 384, 0, 192, 2),
        ("exact128", 384, 384, 128, 1)]
    argv = ["--mxfp4-pair", "--native-slices", "--mxfp4-tilings", "--control-repeats", "3"]
    for value in values:
        argv += ["--mxfp4-arm", value]
    args = bench.parse_args(argv + ["--profile-dir", "out", "--profile-arm", "legacy128"])
    specs, controls = bench.tiling_specs(args, 384, 192, (64, 128, 192))
    assert specs == [(name, False, 192, None) for name in controls] + [
        ("legacy128", False, 128, None), ("legacy192", False, 192, None),
        ("exact128", True, 128, 384)]
    with pytest.raises(SystemExit):
        bench.parse_args(argv + ["--profile-dir", "out", "--profile-arm", "missing"])
    for value in ["a:384:0:128:3", "a:384:0:192:4", "a:384:-32:128:2"]:
        with pytest.raises(ValueError):
            bench.native_screen_arms([value])


def test_explicit_aligned_extent_uses_exact_staging():
    function = next(node for node in ast.parse(_PATH.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "native_pair_arm")
    branch = next(node for node in function.body if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Compare) and ast.unparse(node.test) == "stored_width is not None"
                  and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                      and target.id == "tail" for target in child.targets) for child in node.body))
    assignment, = [child for child in branch.body if isinstance(child, ast.Assign)]
    assert isinstance(assignment.value, ast.Constant) and assignment.value.value is True


def test_explicit_tile32_is_benchmark_only():
    args = bench.parse_args(["--mxfp4-pair", "--native-slices", "--mxfp4-tilings",
                             "--mxfp4-arm", "tile32:288:288:32"])
    specs, _ = bench.tiling_specs(args, 288, 192, (64, 128, 192))
    assert specs[-1] == ("tile32", True, 32, 288)


def test_explicit_names_and_mode_are_unambiguous():
    with pytest.raises(ValueError, match="duplicate"):
        bench.native_screen_arms(["a:96:96:64", "a:128:128:64"])
    for argv in (["--mxfp4-arm", "a:96:96:64"], ["--control-repeats", "1"],
                 ["--mxfp4-pair", "--native-slices", "--mxfp4-tilings", "--widths", "96",
                  "--mxfp4-arm", "a:96:96:64"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


@pytest.mark.parametrize("stored", [96, 128, 160])
def test_explicit_padding_preserves_real_payload_and_scale_blocks(stored):
    logical, hidden, experts = 96, 128, 2
    shapes = {"w1": (experts, logical, hidden // 2), "w3": (experts, logical, hidden // 2),
              "w2": (experts, hidden, logical // 2)}
    weights = {name: torch.full(shape, 0x32, dtype=torch.uint8) for name, shape in shapes.items()}
    scales = {name: torch.full((*shape[:-1], shape[-1] // 16), 122, dtype=torch.uint8)
              for name, shape in shapes.items()}
    resident, factors = bench.extend_native_weights(weights, scales, stored)
    assert resident["w1"].shape == (experts, stored, hidden // 2)
    assert resident["w2"].shape == (experts, hidden, stored // 2)
    for name in ("w1", "w3"):
        assert torch.equal(resident[name][:, :logical], weights[name])
        assert torch.equal(factors[name][:, :logical], scales[name])
        assert not bool(resident[name][:, logical:].count_nonzero())
    assert torch.equal(resident["w2"][..., :logical // 2], weights["w2"])
    assert not bool(resident["w2"][..., logical // 2:].count_nonzero())
    assert torch.equal(factors["w2"][..., :logical // 32], scales["w2"])
    assert bool((factors["w2"] == 122).all())


def jsonl_routes(tmp_path, records):
    path = tmp_path / "routes.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_existing_jsonl_replays_exact_narrow_rows_and_wide_histograms(tmp_path):
    narrow = dict(event="expert_routes", layer=7, lane=0, rows=3, topk=[[3, 0], [0, 1], [3, 0]])
    wide = dict(event="expert_routes", layer=8, lane=1, rows=80, expert_rows=[[0, 80], [1, 40], [3, 40]])
    waves = bench.recorded_waves(jsonl_routes(tmp_path, [narrow, wide]),
                                 SimpleNamespace(name="mimo", experts=4, top_k=2))
    assert waves[0]["expert_rows"] == [(0, 3), (1, 1), (3, 2)]
    assert bench.screen_route_wave(waves, 3, 0) is waves[0]
    assert bench.screen_route_wave(waves + [waves[0]], 3, 1) is waves[0]
    assert bench.screen_route_ids(waves[0], 4, 2, 1).tolist() == narrow["topk"]
    ids = bench.screen_route_ids(waves[1], 4, 2, 5)
    assert torch.bincount(ids.flatten().long(), minlength=4).tolist() == [80, 40, 0, 40]
    assert bool((ids[:, 0] != ids[:, 1]).all())
    assert torch.equal(ids, bench.screen_route_ids(waves[1], 4, 2, 5))
    with pytest.raises(ValueError, match="exactly 16"):
        bench.screen_route_wave(waves, 16, 0)


@pytest.mark.parametrize("record", [dict(event="other", rows=1, topk=[[0, 1]]),
    dict(event="expert_routes", rows=1, topk=[[0, 0]]),
    dict(event="expert_routes", rows=1, topk=[[0, 4]]),
    dict(event="expert_routes", rows=1, topk=[[0]]),
    dict(event="expert_routes", rows=2, topk=[[0, 1]]),
    dict(event="expert_routes", rows=1, expert_rows=[[0, 1]])])
def test_malformed_jsonl_routes_fail_closed(tmp_path, record):
    with pytest.raises(ValueError):
        bench.recorded_waves(jsonl_routes(tmp_path, [record]), SimpleNamespace(name="v41", experts=4, top_k=2))


@pytest.mark.parametrize("family,width,tile,atomic,coded", [
    ("dsv4f", 256, 128, True, False), ("dsv4f", 512, 128, True, False),
    ("dsv4f", 1024, 128, True, False), ("dsv4f", 320, 128, False, False),
    ("v41", 288, 64, False, True), ("v41", 512, 128, False, True),
    ("v41", 576, 192, False, True)])
def test_shipped_prefill_recipe_uses_policy(monkeypatch, family, width, tile, atomic, coded):
    calls = []
    def select(hidden, width, sm, capacity, fallback):
        calls.append((hidden, width, sm, capacity, fallback))
        return 64 if hidden == 5120 and width == 288 else fallback
    monkeypatch.setattr(bench, "engine_layout_policy", lambda root: SimpleNamespace(
        native_slice_width=select, stored_width=lambda hidden, width: width))
    hidden = 5120 if family == "v41" else 4096
    assert bench.shipped_prefill_recipe(family, hidden, width, 256, "engine") == dict(
        tile=tile, exact=True, atomic=atomic, coded=coded)
    assert calls[0][2:4] == (121, 256)


@pytest.mark.parametrize("family", ["v41", "dsv4f"])
def test_shipped_prefill_cli_is_bounded(family):
    args = ["--geometry", family, "--mxfp4-pair", "--mxfp4-unify", "--native-stream",
            "--shipped-prefill", "--engine-policy-root", "engine", "--rows", "256,1024,4096"]
    assert bench.parse_args(args).shipped_prefill
    for extra in (["--rows", "80"], ["--capacity", "4096"], ["--row-major-mma"]):
        with pytest.raises(SystemExit):
            bench.parse_args(args + extra)
    with pytest.raises(SystemExit):
        bench.parse_args(["--shipped-prefill"])


def test_implicit_widths_require_shared_policy():
    with pytest.raises(ValueError, match="engine-policy-root"):
        bench.paired_widths(2304)
    assert bench.paired_widths(2304, "448,512") == [448, 512]


def test_atomic_controls_do_not_claim_determinism():
    tree = ast.parse(_PATH.read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    atomic = ast.unparse(funcs["qualify_atomic_pair_replay"])
    assert "assert_close" in atomic and "deterministic=False" in atomic
    ordered = ast.unparse(funcs["qualify_pair_replay"])
    assert "repeat determinism" in ordered and "torch.equal(output, expected)" in ordered
    native = ast.unparse(funcs["native_pair_arm"])
    assert "1 if atomic else topk" in native
    assert "qualify_atomic_pair_replay if atomic else qualify_pair_replay" in native
