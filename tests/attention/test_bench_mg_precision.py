"""CPU-only contracts for the MG precision diagnostic."""
import pytest

from benchmarks.bench_mg_precision import arm_traits, decode_arm_traits, parse_args
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
