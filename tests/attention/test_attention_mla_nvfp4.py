from __future__ import annotations

import pytest
import torch

from b12x.attention._shared.mla import api, traits
from b12x.attention._shared.mla.kernel import run_unified_decode
from b12x.attention._shared.mla.prefill import run_unified_prefill
from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
from b12x.attention._shared.mla.smem import make_smem_layout
from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat

from ..conftest import require_b12x
from .._reference.helpers import (
    dequantize_nvfp4_mla_nope,
    pack_dsv4_nvfp4_record_reference,
)


def test_nvfp4_decode_allocates_bf16_q_staging() -> None:
    nvfp4_traits = traits.make_unified_traits(
        ModelType.GLM_NSA,
        ComputeMode.BF16,
        ScaleFormat.NVFP4_E4M3,
        fp8_rope=False,
    )
    layout = make_smem_layout(nvfp4_traits)

    expected_bytes = nvfp4_traits.hpb * nvfp4_traits.q_nope_stride * 2
    assert layout.q_fp8_bytes == expected_bytes
    assert layout.q_sc_off >= layout.q_fp8_off + expected_bytes


def test_nvfp4_decode_rejects_unknown_record_width() -> None:
    with pytest.raises(ValueError, match="must be 368 or 432 bytes"):
        _run_invalid_nvfp4_decode(record_bytes=400, fp8_rope=None)


def test_nvfp4_decode_rejects_fp8_rope_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees with fp8_rope_override"):
        _run_invalid_nvfp4_decode(record_bytes=368, fp8_rope=False)


def test_nvfp4_mg_prefill_rejects_unknown_record_width() -> None:
    with pytest.raises(ValueError, match="must be 368 or 432 bytes"):
        _run_invalid_nvfp4_mg_prefill(record_bytes=400, fp8_rope=None)


def test_nvfp4_mg_prefill_rejects_explicit_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees with fp8_rope"):
        _run_invalid_nvfp4_mg_prefill(record_bytes=368, fp8_rope=False)


def test_api_uses_traits_fp8_rope_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(traits, "KV_FP8_ROPE_ENABLED", True)
    assert api._resolve_kv_fp8_rope(None) is True

    monkeypatch.setattr(traits, "KV_FP8_ROPE_ENABLED", False)
    assert api._resolve_kv_fp8_rope(None) is False


@torch.inference_mode()
def test_dsv4_nvfp4_mg_dual_cache_reads_rope_from_each_pool() -> None:
    device = require_b12x()
    torch.manual_seed(121_434)
    rows, heads = 2, 32
    main_page_size, extra_page_size = 256, 64

    q = torch.zeros((rows, heads, 512), dtype=torch.bfloat16, device=device)
    q[..., 448:] = torch.randn(
        (rows, heads, 64), dtype=torch.bfloat16, device=device
    )
    main_values = torch.randn(
        (main_page_size, 512), dtype=torch.bfloat16, device=device
    )
    main_values[:, 448:] = 0
    extra_values = torch.randn(
        (extra_page_size, 512), dtype=torch.bfloat16, device=device
    )
    extra_values[:, 448:] *= 2

    main_records = pack_dsv4_nvfp4_record_reference(main_values)
    extra_records = pack_dsv4_nvfp4_record_reference(extra_values)
    main_cache = main_records.view(1, main_page_size, 1, 432)
    extra_cache = extra_records.view(1, extra_page_size, 1, 432)
    main_dequant, _ = dequantize_nvfp4_mla_nope(main_records)
    extra_dequant, _ = dequantize_nvfp4_mla_nope(extra_records)

    main_indices = torch.arange(
        128, dtype=torch.int32, device=device
    ).repeat(rows, 1)
    extra_indices = torch.arange(
        extra_page_size, dtype=torch.int32, device=device
    ).repeat(rows, 1)
    main_lengths = torch.tensor([128, 97], dtype=torch.int32, device=device)
    extra_lengths = torch.tensor([64, 43], dtype=torch.int32, device=device)
    output = torch.empty_like(q)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=device)

    run_unified_prefill(
        q=q,
        kv_cache=main_cache,
        topk_indices=main_indices,
        topk_length=main_lengths,
        sm_scale=512**-0.5,
        page_block_size=main_page_size,
        output=output,
        lse_out=lse,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=False,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        extra_page_block_size=extra_page_size,
    )
    torch.cuda.synchronize()

    expected = torch.empty_like(output)
    main_rope = main_records[:, 304:].contiguous().view(torch.bfloat16)
    extra_rope = extra_records[:, 304:].contiguous().view(torch.bfloat16)
    for row in range(rows):
        main_selected = main_indices[row, : int(main_lengths[row])].long()
        extra_selected = extra_indices[row, : int(extra_lengths[row])].long()
        keys = torch.cat(
            (
                torch.cat(
                    (main_dequant[main_selected, :448], main_rope[main_selected]),
                    dim=-1,
                ),
                torch.cat(
                    (
                        extra_dequant[extra_selected, :448],
                        extra_rope[extra_selected],
                    ),
                    dim=-1,
                ),
            )
        )
        values = torch.cat(
            (main_dequant[main_selected], extra_dequant[extra_selected])
        )
        scores = q[row].float() @ keys.float().T * (512**-0.5)
        expected[row] = (scores.softmax(dim=-1) @ values.float()).to(torch.bfloat16)

    torch.testing.assert_close(output, expected, atol=3.0e-3, rtol=1.0e-2)


def _run_invalid_nvfp4_decode(*, record_bytes: int, fp8_rope: bool | None) -> None:
    rows = 1
    topk = 4
    run_unified_decode(
        q_all=torch.empty((rows, 8, 576), dtype=torch.bfloat16),
        swa_k_cache=torch.empty((1, 1, record_bytes), dtype=torch.uint8),
        swa_indices=torch.zeros((rows, topk), dtype=torch.int32),
        swa_topk_lengths=torch.full((rows,), topk, dtype=torch.int32),
        workspace=object(),
        sm_scale=0.1,
        swa_page_size=64,
        scale_format_override=ScaleFormat.NVFP4_E4M3,
        fp8_rope_override=fp8_rope,
    )


def _run_invalid_nvfp4_mg_prefill(*, record_bytes: int, fp8_rope: bool | None) -> None:
    rows = 1
    topk = 4
    run_unified_prefill_mg(
        q=torch.empty((rows, 16, 576), dtype=torch.bfloat16),
        kv_cache=torch.empty((1, 1, record_bytes), dtype=torch.uint8),
        topk_indices=torch.zeros((rows, topk), dtype=torch.int32),
        sm_scale=0.1,
        page_block_size=1,
        model_type=ModelType.GLM_NSA,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=fp8_rope,
    )
