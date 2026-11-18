"""CPU-only contracts for the MG precision diagnostic."""
import pytest

from benchmarks.bench_mg_precision import aot_cell_seed, arm_traits, critical_path_samples, decode_arm_traits, parse_args
from b12x.attention._shared.mla.traits import ComputeMode, ModelType


def test_default_geometry_and_controls(tmp_path):
    args = parse_args(["--format", "dsv4", "--output", str(tmp_path / "out.json")])
    assert (args.heads, args.indexed_width) == (64, 512)
    assert args.rows == [1, 64, 2048, 2752, 4096]
    assert args.scores == ["spread", "peaked"]


@pytest.mark.parametrize("option,value", [("--rows", "0"), ("--scores", "other"), ("--heads", "17"), ("--samples", "0")])
def test_invalid_controls(option, value, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--format", "dsv41", "--output", str(tmp_path / "o.json"), option, value])


def test_584b_qk_modes_preserve_cache_format():
    fp8 = arm_traits("dsv4", "fp8-qk")
    bf16 = arm_traits("dsv4", "bf16-qk")
    assert fp8.model_type == bf16.model_type == ModelType.DSV4
    assert fp8.kv_gmem_stride == bf16.kv_gmem_stride == 584
    assert fp8.compute_mode == ComputeMode.FP8
    assert bf16.compute_mode == ComputeMode.BF16
    assert not fp8.fp8_internal and not bf16.fp8_internal


def test_native_internal_fp8_does_not_select_fp8_qk():
    bf16 = arm_traits("dsv41", "bf16-pv")
    fp8 = arm_traits("dsv41", "fp8-pv")
    assert bf16.model_type == fp8.model_type == ModelType.DSV41
    assert bf16.compute_mode == fp8.compute_mode == ComputeMode.BF16
    assert not bf16.fp8_internal and fp8.fp8_internal


def test_decode_traits_match_native_aot():
    fp8 = decode_arm_traits("fp8-decode")
    assert fp8.compute_mode == ComputeMode.FP8 and fp8.fp8_internal
    assert (fp8.q_nope_stride, fp8.kv_smem_stride, fp8.nt_per_warp_xv) == (528, 624, 2)
    assert decode_arm_traits("fp8-f32-partials") == fp8
    bf16 = decode_arm_traits("bf16-decode")
    assert bf16.compute_mode == ComputeMode.BF16 and not bf16.fp8_internal


def test_decode_contract_and_accuracy_only(tmp_path):
    args = parse_args(["--format", "dsv41", "--mode", "decode", "--rows", "1,2,4,8,16,64",
        "--accuracy-only", "--output", str(tmp_path / "out.json")])
    assert args.timing_rows == []
    assert args.rows == [1, 2, 4, 8, 16, 64]
    with pytest.raises(SystemExit):
        parse_args(["--format", "dsv4", "--mode", "decode", "--output", str(tmp_path / "o.json")])


def test_wrong_format_arm_rejected():
    with pytest.raises(ValueError):
        arm_traits("dsv41", "fp8-qk")


def test_decode_aot_has_fixed_production_cells_and_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("B12X_MLA_SM120_NUM_SPLITS", raising=False)
    argv = ["--mode", "decode-aot", "--output", str(tmp_path / "decode.json")]
    assert parse_args(argv).cells is None
    assert parse_args(argv + ["--cells", "glm-h64,v4p-h128"]).cells == ["glm-h64", "v4p-h128"]
    for flags in (["--rows", "1"], ["--cells", "v4p-h16"], ["--samples", "1"]):
        with pytest.raises(SystemExit):
            parse_args(argv + flags)
    monkeypatch.setenv("B12X_MLA_SM120_NUM_SPLITS", "2")
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_aot_cell_seeds_do_not_depend_on_selection_or_iteration_order():
    cells = (("glmf-h64", "sparse"), ("v4p-h64", "c128"), ("v4p-h64", "c4"))
    forward = {cell: aot_cell_seed(2146, *cell) for cell in cells}
    assert forward == {cell: aot_cell_seed(2146, *cell) for cell in reversed(cells)}
    assert len(set(forward.values())) == len(cells)
    assert aot_cell_seed(2147, *cells[0]) != forward[cells[0]]


def test_c128_1m_confirmation_only_accepts_v4_cells(tmp_path):
    argv = ["--mode", "decode-aot", "--confirm-c128-1m", "--output", str(tmp_path / "decode.json")]
    assert parse_args(argv + ["--cells", "v4p-h128"]).confirm_c128_1m
    with pytest.raises(SystemExit):
        parse_args(argv + ["--cells", "glm-h64"])


def test_context_critical_path_pairs_alternating_blocks(tmp_path):
    args = parse_args(["--mode", "context-aot", "--cells", "v4p,ml4nope", "--output", str(tmp_path / "out.json")])
    assert args.cells == ["v4p", "ml4nope"]
    assert critical_path_samples([[1.0, 4.0, 1.0], [3.0, 2.0, 3.0]]) == [3.0, 4.0, 3.0]
