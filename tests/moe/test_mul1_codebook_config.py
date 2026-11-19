from __future__ import annotations

import pytest

from b12x.moe._shared.trellis_codebooks import (
    CODEBOOKS,
    MCG,
    MUL1,
    MUL1_MULTIPLIER,
    normalize_codebook,
    validate_codebook_bits,
)
from b12x.moe.fused_moe.config import TrellisCodebook, TrellisConfig


def test_mul1_codebook_registry_and_bits_validation() -> None:
    assert MUL1 == "mul1"
    assert MUL1 in CODEBOOKS
    assert MUL1_MULTIPLIER == 0x83DCD12D
    assert normalize_codebook("mul1") == MUL1
    assert normalize_codebook("MUL1") == MUL1
    assert normalize_codebook(0x83DCD12D) == MUL1

    for bits in (2, 3, 4, 5, 6):
        validate_codebook_bits(MUL1, bits)
    for invalid_bits in (0, 1, 7):
        with pytest.raises(ValueError, match="mul1 is defined only for K2/K3/K4/K5/K6"):
            validate_codebook_bits(MUL1, invalid_bits)

    with pytest.raises(ValueError, match="mcg is defined only for K2/K3/K4/K5/K6"):
        validate_codebook_bits(MCG, 1)


def test_mul1_trellis_config_validates_on_cpu() -> None:
    raw = {
        "quant_method": "b12x_trellis",
        "b12x_trellis": {
            "version": 2,
            "codebook": "mul1",
            "rate": {"granularity": "per_expert_projection"},
            "scale": {
                "input_scales": {"vectors": "per_layer", "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": "per_layer", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        },
    }
    parsed = TrellisConfig.from_quantization_config(raw)
    assert parsed.codebook is TrellisCodebook.MUL1
    assert parsed.to_dict() == raw["b12x_trellis"]


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_w4a16_fused_moe_kernel_accepts_mul1_config_on_cpu(bits: int) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    common = dict(
        size_m=8,
        hidden_size=512,
        intermediate_size=256,
        num_experts=8,
        top_k=2,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        fc1_tile_n=128,
        fc1_tile_k=64,
        fc2_tile_n=128,
        fc2_tile_k=64,
        moe_block_size=8,
        max_m_blocks=8,
        element_dtype="fp16",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        intermediate_rotation=True,
        full_rotation=True,
        workspace_sms=48,
        schedule_whole_tiles=True,
    )
    mul1_kernel = W4A16FusedMoeKernel(**common, trellis_codebook="mul1")
    mcg_kernel = W4A16FusedMoeKernel(**common, trellis_codebook="mcg")
    assert mul1_kernel.trellis_codebook == "mul1"
    assert not mul1_kernel.sqg_xor_cheb_t12_smem
    assert mul1_kernel.__cache_key__ != mcg_kernel.__cache_key__


def test_mul1_rejects_k1_in_registry_and_w4a16_kernel() -> None:
    """MUL1 rejects K1 in both the codebook registry and W4A16 t256 kernel tiles (K2..K6 only)."""
    with pytest.raises(ValueError, match="mul1 is defined only for K2/K3/K4/K5/K6"):
        validate_codebook_bits(MUL1, 1)
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16GemmKernel

    with pytest.raises(ValueError, match="trellis_t256 bits must be one of"):
        W4A16GemmKernel(
            size_m=8,
            size_n=128,
            size_k=128,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=128,
            tile_k=64,
            moe_block_size=8,
            max_m_blocks=1,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="packed",
            trellis_bits=1,
            trellis_codebook="mul1",
        )
