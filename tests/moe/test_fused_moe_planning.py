from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import b12x.moe.fused_moe._impl as fused_moe_impl
import b12x.moe.fused_moe.trellis as trellis_impl
from b12x.moe.fused_moe._impl import TPMoEScratchCaps
from b12x.moe.fused_moe._tuning import MoeDecodeConfig

# Preserve fork-native full-rotation contracts after removal of legacy vLLM
# and profile APIs. This backend does not support the W4A16 direct route.
_TRELLIS_CONFIG = MoeDecodeConfig(backend="w4a16", route_planner="internal",
    max_active_clusters=None, w4a16_route_mode="packed")


def _trellis_caps() -> TPMoEScratchCaps:
    weight_plan = fused_moe_impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w31",
        w4a16_layout="trellis_native",
        trellis_bits=3,
        trellis_tile_config=(128, 128, 128, 128),
        trellis_codebook="mcg",
        trellis_rate_granularity="per_expert_projection",
    )
    return TPMoEScratchCaps(
        decode_config=_TRELLIS_CONFIG,
        max_tokens=3072,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        w4a16_block_size_m=64,
    )



def _exl3_projection_caps(
    *,
    num_experts: int = 256,
    max_tokens: int = 16,
    num_topk: int = 6,
) -> TPMoEScratchCaps:
    weight_plan = fused_moe_impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=4096,
        intermediate_size=512,
        w13_layout="trellis_t256_proj",
        w4a16_layout="trellis_native",
        trellis_bits=2,
        trellis_codebook="mcg",
        trellis_rate_granularity="per_expert_projection",
    )
    return TPMoEScratchCaps(
        decode_config=_TRELLIS_CONFIG,
        max_tokens=max_tokens,
        num_topk=num_topk,
        route_num_experts=num_experts,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )



@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_preparation_preserves_finalized_mcg_trellis_tensors() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts, hidden_size, intermediate_size, bits = 2, 128, 128, 4
    plan = fused_moe_impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        trellis_bits=bits,
        trellis_codebook="mcg",
        trellis_tile_config=(64, 128, 64, 128),
    )
    w13 = torch.zeros(
        (
            2,
            num_experts,
            hidden_size // 16,
            intermediate_size // 16,
            16 * bits,
        ),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.zeros(
        (
            num_experts,
            intermediate_size // 16,
            hidden_size // 16,
            16 * bits,
        ),
        dtype=torch.int16,
        device=device,
    )

    def ones(*shape: int) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.float16, device=device)

    experts = fused_moe_impl.prepare_b12x_fp4_moe_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=ones(num_experts, hidden_size),
        up_suh=ones(num_experts, hidden_size),
        intermediate_rotations=ones(num_experts, 3 * intermediate_size),
        down_svh=ones(num_experts, hidden_size),
        trellis_mcg=0xCBAC1FED,
    )

    assert experts.source_format == "b12x_trellis"
    assert experts.representation is not None
    assert experts.representation.layout.value == "trellis_native"
    assert experts.representation.value.params_dtype == torch.float16
    assert experts.w1_fp4.untyped_storage().data_ptr() == w13.untyped_storage().data_ptr()
    assert experts.w2_fp4.untyped_storage().data_ptr() == w2.untyped_storage().data_ptr()



def test_projection_mixed_config_preserves_bf16_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    caps = replace(
        _trellis_caps(), full_rotation_output_dtype=torch.bfloat16
    )
    plan = fused_moe_impl.plan_tp_moe_scratch(caps)
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.full_rotation_output_dtype == torch.bfloat16
    assert specs["full_rotation_output"].dtype == torch.bfloat16



def test_direct_exl3_projection_plan_preserves_k2_tier_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    plan = fused_moe_impl.plan_tp_moe_scratch(_exl3_projection_caps())

    assert plan.caps.source_format == "exl3_trellis_mcg"
    assert plan._core_workspace_plan.implementation == "trellis_mixed"
    assert plan._core_workspace_plan.trellis_bits == 2
    assert plan._core_workspace_plan.projection_mixed_trellis
    assert plan._core_workspace_plan.trellis_tile_config is None



def test_projection_mixed_plan_allows_larger_ep_route_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    base = _exl3_projection_caps()
    caps = TPMoEScratchCaps(
        decode_config=_TRELLIS_CONFIG,
        max_tokens=base.max_tokens,
        num_topk=base.num_topk,
        route_num_experts=288,
        device=base.device,
        weight_plan=base.weight_plan,
        quant_mode=base.quant_mode,
        w4a16_block_size_m=base.w4a16_block_size_m,
    )

    plan = fused_moe_impl.plan_tp_moe_scratch(caps)

    assert plan._core_workspace_plan.weight_E == 256
    assert plan._core_workspace_plan.route_E == 288
    assert plan._core_workspace_plan.projection_mixed_trellis



def test_projection_mixed_plan_accepts_glm_288_expert_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe_impl.plan_tp_moe_scratch(_exl3_projection_caps(num_experts=288))

    assert plan._core_workspace_plan.weight_E == 288
    assert plan._core_workspace_plan.route_E == 288
    assert plan._core_workspace_plan.projection_mixed_trellis



def test_projection_mixed_packed_workspace_covers_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe_impl.plan_tp_moe_scratch(
        _exl3_projection_caps(num_experts=4, max_tokens=8, num_topk=8)
    )
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    # The measured production policy keeps even tiny mixed-Trellis batches on
    # the expert-packed route. Its bounded capacity requires six blocks for
    # this four-expert, 64-route arena rather than one block per live route.
    assert specs["block_expert_ids"].shape == (6,)



@pytest.mark.parametrize(
    ("capacity", "expected"),
    (
        (1, (1,)),
        (4, (1, 2, 3, 4)),
        (32, tuple(range(1, 33))),
        (33, (33,)),
        (4096, (4096,)),
    ),
)
def test_trellis_exact_launch_widths_cover_decode_only(
    capacity: int, expected: tuple[int, ...]
) -> None:
    assert fused_moe_impl._trellis_exact_launch_token_counts(capacity) == expected



@pytest.mark.parametrize(
    ("tokens", "top_k", "direct_exl3", "expected"),
    (
        (1, 6, True, False),
        (12, 6, True, False),
        (13, 6, True, False),
        (9, 8, True, False),
        (10, 8, True, False),
        (12, 6, False, False),
    ),
)
def test_projection_mixed_direct_route_limit_is_bounded(
    tokens: int,
    top_k: int,
    direct_exl3: bool,
    expected: bool,
) -> None:
    assert (
        fused_moe_impl._projection_mixed_direct_topk_routes(
            tokens,
            top_k,
            direct_exl3=direct_exl3,
        )
        is expected
    )



def test_projection_mixed_tile_config_preserves_whole_tile_geometry() -> None:
    configured = (64, 256, 64, 128)

    assert fused_moe_impl._projection_mixed_tile_config(
        configured,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=16,
        direct_topk_routes=True,
    ) == configured
    assert fused_moe_impl._projection_mixed_tile_config(
        configured,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=16,
        direct_topk_routes=False,
    ) == configured
    assert fused_moe_impl._projection_mixed_tile_config(
        None,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=9,
        direct_topk_routes=True,
    ) == (64, 256, 64, 256)
    assert fused_moe_impl._projection_mixed_tile_config(
        None,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=9,
        direct_topk_routes=False,
    ) == (64, 256, 64, 256)
    assert fused_moe_impl._projection_mixed_tile_config(
        None,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=10,
        direct_topk_routes=False,
    ) == (128, 128, 128, 128)
    assert fused_moe_impl._projection_mixed_tile_config(
        None,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=32,
        direct_topk_routes=False,
    ) == (128, 128, 128, 128)
    assert fused_moe_impl._projection_mixed_tile_config(
        None,
        hidden_size=4096,
        intermediate_size=2048,
        token_count=33,
        direct_topk_routes=False,
    ) == (64, 256, 64, 256)



def test_projection_mixed_launch_selection_reuses_prefill_capacity() -> None:
    exact = object()
    capacity = object()
    launches = (
        (1, torch.int32, False, False, exact),
        (8192, torch.int32, False, False, capacity),
        (8192, torch.int64, False, False, object()),
    )

    assert (
        fused_moe_impl._select_projection_mixed_trellis_launch(
            launches,
            live_tokens=1,
            route_ids_dtype=torch.int32,
            broadcast_suh=False,
            broadcast_svh=False,
        )
        is exact
    )
    assert (
        fused_moe_impl._select_projection_mixed_trellis_launch(
            launches,
            live_tokens=99,
            route_ids_dtype=torch.int32,
            broadcast_suh=False,
            broadcast_svh=False,
        )
        is capacity
    )
    assert (
        fused_moe_impl._select_projection_mixed_trellis_launch(
            launches,
            live_tokens=8193,
            route_ids_dtype=torch.int32,
            broadcast_suh=False,
            broadcast_svh=False,
        )
        is None
    )



def test_projection_mixed_accepts_precomposed_ep_route_map() -> None:
    prepared = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
    global_to_tier = torch.tensor(
        [3, -1, 1, -1, 0, -1, 2, -1],
        dtype=torch.int32,
    )

    selected = fused_moe_impl._projection_mixed_route_map(
        prepared,
        global_to_tier,
        route_num_experts=8,
        device=torch.device("cpu"),
    )

    assert selected.data_ptr() == global_to_tier.data_ptr()
    assert selected.tolist() == [3, -1, 1, -1, 0, -1, 2, -1]



def test_projection_mixed_ep_route_map_fails_closed() -> None:
    prepared = torch.arange(4, dtype=torch.int32)

    with pytest.raises(TypeError, match="torch.int32"):
        fused_moe_impl._projection_mixed_route_map(
            prepared,
            torch.arange(8, dtype=torch.int64),
            route_num_experts=8,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match=r"int32\[8\]"):
        fused_moe_impl._projection_mixed_route_map(
            prepared,
            torch.arange(7, dtype=torch.int32),
            route_num_experts=8,
            device=torch.device("cpu"),
        )



def test_projection_mixed_bind_zeroes_cooperative_workspace_before_launch() -> None:
    source = inspect.getsource(
        fused_moe_impl._bind_projection_mixed_trellis_from_views
    )

    zero = source.index('tensors["kernel_workspace"].zero_()')
    buffers = source.index("buffers = MixedTrellisBuffers(")
    assert zero < buffers



def test_empty_projection_tier_uses_one_dummy_plane() -> None:
    value = trellis_impl._projection_native(
        torch.empty((4, 1), dtype=torch.uint8),
        experts=[],
        projection=0,
        bits=2,
        offsets=[],
        hidden_size=128,
        fc1=True,
    )

    assert value.shape == (1, 8, 4, 32)



def test_projection_mixed_qwen_geometry_has_whole_projection_tiles():
    for rows in (1, 3, 9, 10, 16, 32, 33, 2048):
        for direct in (False, True):
            tiles = fused_moe_impl._projection_mixed_tile_config(
                None, hidden_size=2560, intermediate_size=640,
                token_count=rows, direct_topk_routes=direct,
            )
            assert 640 % tiles[1] == 0
            assert 2560 % tiles[3] == 0
