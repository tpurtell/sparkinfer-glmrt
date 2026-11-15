"""CPU-only contracts for the MG precision diagnostic."""
from dataclasses import replace

import pytest

from benchmarks.bench_mg_precision import arm_traits, parse_args
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


def test_medium_scores_are_inline_only(tmp_path):
    args = parse_args(["--format", "glm", "--scores", "medium", "--output", str(tmp_path / "o.json")])
    assert args.scores == ["medium"]
    with pytest.raises(SystemExit):
        parse_args(["--format", "dsv4", "--scores", "medium", "--output", str(tmp_path / "o.json")])


def test_wrong_format_arm_rejected():
    with pytest.raises(ValueError):
        arm_traits("dsv41", "fp8-qk")


@pytest.mark.parametrize("fmt,model,stride", [("glm", ModelType.GLM_NSA, 656), ("kimi", ModelType.GLM_NSA, 656),
                                              ("glmf", ModelType.GLM_NEXT, 528)])
def test_inline_fp32_modes_preserve_cache_format(fmt, model, stride):
    from b12x.attention._shared.mla.smem import SM120_SMEM_CARVEOUT_BYTES
    from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg, max_mg_head_groups

    fp8, bf16 = arm_traits(fmt, "fp8"), arm_traits(fmt, "bf16")
    assert fp8.model_type == bf16.model_type == model
    assert fp8.kv_gmem_stride == bf16.kv_gmem_stride == stride
    assert (fp8.compute_mode, fp8.pv_bf16) == (ComputeMode.FP8, False)
    assert (bf16.compute_mode, bf16.pv_bf16) == (ComputeMode.BF16, True)
    assert (max_mg_head_groups(fp8), max_mg_head_groups(bf16)) == (2, 1)
    assert make_smem_layout_mg(bf16, 1).total_bytes < SM120_SMEM_CARVEOUT_BYTES


def test_inline_defaults_and_mixed_mode_rejected(tmp_path):
    from dataclasses import replace

    from b12x.attention._shared.mla.prefill_mg import UnifiedPrefillMGKernel
    from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg

    args = parse_args(["--format", "kimi", "--output", str(tmp_path / "o.json")])
    assert args.heads == 96
    with pytest.raises(SystemExit):
        parse_args(["--format", "glm", "--selected", "2049", "--output", str(tmp_path / "o.json")])
    with pytest.raises(ValueError):
        arm_traits("glm", "bf16-qk")
    shape = dict(page_block_size=64, num_tiles=32, replicate_h=4, num_heads=64, q_stride=(64 * 576, 576, 1),
                 indices_stride0=2048, output_stride=(64 * 512, 512, 1), out_lse_stride=(64, 1), has_sink=False,
                 topk=2048)
    for traits in (replace(arm_traits("glm", "bf16"), pv_bf16=False),          # BF16 QK / FP8 PV
                   replace(arm_traits("glm", "fp8"), pv_bf16=True)):           # FP8 QK / BF16 PV
        with pytest.raises(ValueError, match="FP8 QK/PV or BF16 QK/PV"):
            UnifiedPrefillMGKernel(traits, make_smem_layout_mg(traits, 1), **shape)
    bf16 = arm_traits("glm", "bf16")
    UnifiedPrefillMGKernel(bf16, make_smem_layout_mg(bf16, 1), **shape)
    with pytest.raises(ValueError, match="mg_n_hg=1"):
        UnifiedPrefillMGKernel(bf16, make_smem_layout_mg(bf16, 2), **{**shape, "replicate_h": 2})
