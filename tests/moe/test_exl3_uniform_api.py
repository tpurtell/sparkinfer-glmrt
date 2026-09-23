"""Uniform EXL3 K4 checkpoints through the canonical fused-MoE weight API."""

import pytest
import torch

from b12x.moe import fused_moe


def _plan():
    return fused_moe.plan_weights(
        source=fused_moe.Exl3TrellisSource(bits=4, tile_config=(64, 128, 64, 128)),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=2, hidden_size=128, intermediate_size=128
        ),
    )


def test_uniform_k4_plans_native_trellis_without_changing_public_dtype():
    plan = _plan()
    assert plan._impl.source_format == "exl3_trellis_mcg"
    assert plan._impl.trellis_bits == 4
    assert plan.prepared_format.packing is fused_moe.WeightPacking.TRELLIS_NATIVE
    assert plan.activation.io_dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_uniform_k4_prepares_fp16_rotations_under_bf16_io():
    device = torch.device("cuda:0")
    weights = fused_moe.Exl3TrellisWeights(
        w13=torch.zeros((2, 2, 8, 8, 64), dtype=torch.int16, device=device),
        w2=torch.zeros((2, 8, 8, 64), dtype=torch.int16, device=device),
        gate_suh=torch.ones((2, 128), dtype=torch.float16, device=device),
        up_suh=torch.ones((2, 128), dtype=torch.float16, device=device),
        intermediate_rotations=torch.ones((2, 384), dtype=torch.float16, device=device),
        down_svh=torch.ones((2, 128), dtype=torch.float16, device=device),
        mcg=0xCBAC1FED,
    )
    prepared = fused_moe.prepare_weights(plan=_plan(), weights=weights)
    assert prepared._impl.representation.value.params_dtype is torch.float16
    assert prepared.plan.activation.io_dtype is torch.bfloat16
