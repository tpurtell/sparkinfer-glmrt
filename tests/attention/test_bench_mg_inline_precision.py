"""CPU-only contracts for the inline-FP32-scale MG precision bench."""
from dataclasses import replace

import pytest

from benchmarks.bench_mg_inline_precision import arm_traits, parse_args
from b12x.attention._shared.mla.traits import ComputeMode, ModelType


def test_defaults_and_controls(tmp_path):
    args = parse_args(["--format", "kimi", "--output", str(tmp_path / "o.json")])
    assert (args.heads, args.scores) == (96, ["spread", "medium", "peaked"])
    assert args.rows == [1, 64, 2048, 2752, 4096]
    for extra in (["--selected", "2049"], ["--scores", "other"], ["--heads", "17"], ["--rows", "0"]):
        with pytest.raises(SystemExit):
            parse_args(["--format", "glm", "--output", str(tmp_path / "o.json"), *extra])
    with pytest.raises(SystemExit):
        parse_args(["--format", "dsv4", "--output", str(tmp_path / "o.json")])


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


def test_only_fp8_and_bf16_modes_exist():
    from b12x.attention._shared.mla.prefill_mg import UnifiedPrefillMGKernel
    from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg

    for arm in ("bf16-qk", "bf16-rounded-k", "fp8-qk"):
        with pytest.raises(ValueError):
            arm_traits("glm", arm)
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
