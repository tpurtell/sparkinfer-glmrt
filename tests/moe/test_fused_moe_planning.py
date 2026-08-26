from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import b12x.moe.fused_moe._impl as fused_moe_impl
import b12x.moe.fused_moe.trellis as trellis_impl
from b12x.moe import fused_moe


def _weight_plan() -> fused_moe.WeightsPlan:
    return fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
    )


def _caps(*, block_size_m: int | None) -> fused_moe.Caps:
    weight_plan = _weight_plan()
    return fused_moe.Caps(
        max_tokens=64,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        w4a16_block_size_m=block_size_m,
    )


def _trellis_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
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
    return fused_moe.Caps(
        max_tokens=3072,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        w4a16_block_size_m=64,
    )


def _exl3_projection_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=256,
        hidden_size=4096,
        intermediate_size=512,
        w13_layout="trellis_t256_proj",
        w4a16_layout="trellis_native",
        trellis_bits=2,
        trellis_tile_config=(128, 128, 128, 128),
        trellis_codebook="mcg",
        trellis_rate_granularity="per_expert_projection",
    )
    return fused_moe.Caps(
        max_tokens=16,
        num_topk=6,
        route_num_experts=256,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def _small_packed_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=16,
        hidden_size=128,
        intermediate_size=128,
    )
    return fused_moe.Caps(
        max_tokens=4,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def _subset_router_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=128,
        intermediate_size=128,
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def _mapped_packed_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=128,
        intermediate_size=128,
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=2,
        route_num_experts=12,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def test_required_nbytes_avoids_launch_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    def fail_launch_prewarm(**_kwargs) -> None:
        raise AssertionError("launch prewarm called")

    monkeypatch.setattr(
        fused_moe_impl,
        "_plan_full_rotation_w4a16_launches",
        fail_launch_prewarm,
    )
    caps = _trellis_caps()

    required = fused_moe.required_nbytes(caps)

    assert required > 800 * 1024 * 1024
    assert "required_nbytes" in fused_moe.META.entry_points
    with pytest.raises(TypeError, match="TPMoEScratchCaps"):
        fused_moe.required_nbytes(object())


def test_required_nbytes_matches_scratch_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    caps = _caps(block_size_m=8)

    plan = fused_moe.plan(caps)

    assert fused_moe.required_nbytes(caps) == plan.scratch_specs()[0].shape[0]


def test_small_packed_plan_covers_direct_topk_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_small_packed_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert specs["fc1_c_tmp"].shape == (131072,)
    assert specs["fc2_c_tmp"].shape == (65536,)


def test_non_trellis_core_sizes_routes_for_weight_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_subset_router_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.route_E == 160
    assert specs["packed_route_indices"].shape == (512,)
    assert specs["block_expert_ids"].shape == (64,)
    assert specs["expert_offsets"].shape == (161,)


def test_mapped_packed_plan_covers_global_route_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_mapped_packed_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.weight_E == 8
    assert plan._core_workspace_plan.route_E == 12
    assert specs["expert_offsets"].shape == (13,)
    assert specs["expert_counts"].shape == (12,)


def test_unpinned_small_capacity_matches_reachable_block_8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    automatic = fused_moe.required_nbytes(_caps(block_size_m=None))
    exact = fused_moe.required_nbytes(_caps(block_size_m=8))
    oversized = fused_moe.required_nbytes(_caps(block_size_m=64))

    assert automatic == exact
    assert oversized - automatic > 64 * 1024 * 1024


def _k3_config() -> fused_moe.TrellisConfig:
    return fused_moe.TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "sqg_e4m3",
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {
                    "vectors": "per_layer",
                    "gains": "per_layer",
                },
                "intermediate_scales": {
                    "vectors": "per_layer",
                    "gains": "per_expert",
                },
                "output_scales": {
                    "vectors": "per_layer",
                    "gains": "per_layer",
                },
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {
                    "kind": "coupled_hadamard",
                    "pre_block_size": 512,
                    "post_block_size": 128,
                    "draw_granularity": "per_expert",
                },
            },
        }
    )


def test_trellis_plan_reports_prepared_format() -> None:
    source = _k3_config()
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="situ",
            io_dtype=torch.float16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=8,
            hidden_size=256,
            intermediate_size=256,
        ),
    )

    assert plan.source is source
    assert plan.activation.mode is fused_moe.ActivationMode.A16
    assert plan.prepared_format == fused_moe.PreparedWeightFormat(
        weights=fused_moe.WeightEncoding.TRELLIS,
        scales=fused_moe.ScaleEncoding.TRELLIS_SCALES,
        packing=fused_moe.WeightPacking.TRELLIS_NATIVE,
        available_packings=frozenset({fused_moe.WeightPacking.TRELLIS_NATIVE}),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_canonical_trellis_preparation_uses_the_typed_source() -> None:
    source = _k3_config()
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="situ",
            io_dtype=torch.float16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=1,
            hidden_size=256,
            intermediate_size=256,
        ),
    )
    device = torch.device("cuda", torch.cuda.current_device())

    def ones(*shape: int) -> torch.Tensor:
        return torch.ones(shape, dtype=torch.float16, device=device)

    row_stride = 3 * (256 // 16) * 64 * 3
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.TrellisWeights(
            atoms=torch.zeros((8, row_stride), dtype=torch.uint8, device=device),
            rate=torch.tensor([0x33], dtype=torch.uint8, device=device),
            input_scales=fused_moe.ScaleFactors(ones(256), ones(1)),
            intermediate_scales=fused_moe.ScaleFactors(ones(3, 256), ones(1)),
            output_scales=fused_moe.ScaleFactors(ones(256), ones(1)),
            expert_transform_draws=torch.zeros(1, dtype=torch.uint8, device=device),
        ),
    )

    assert experts.plan is weight_plan
    assert experts.num_experts == 1
    assert experts.hidden_size == 256
    assert experts.intermediate_size == 256


def test_runtime_rejects_unimplemented_orthogonal_trellis_combination() -> None:
    value = _k3_config().to_dict()
    value["codebook"] = "mcg"
    config = fused_moe.TrellisConfig.from_dict(value)

    with pytest.raises(NotImplementedError, match="sqg_e4m3.*uniform"):
        fused_moe.plan_weights(
            source=config,
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="situ",
                io_dtype=torch.float16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=8,
                hidden_size=256,
                intermediate_size=256,
            ),
        )


def test_runtime_rejects_schema_only_sqg_fp16_codebook() -> None:
    value = _k3_config().to_dict()
    value["codebook"] = "sqg_fp16"
    value["transform"]["expert"] = {"kind": "none"}
    config = fused_moe.TrellisConfig.from_dict(value)

    with pytest.raises(NotImplementedError, match="sqg_fp16"):
        fused_moe.plan_weights(
            source=config,
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=torch.float16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=8,
                hidden_size=256,
                intermediate_size=256,
            ),
        )


def test_trellis_output_finalization_casts_into_caller_buffer() -> None:
    accumulated = torch.tensor([[1.25, -2.5]], dtype=torch.float32)
    target = torch.empty_like(accumulated, dtype=torch.bfloat16)

    result = fused_moe_impl._finalize_trellis_output(
        SimpleNamespace(output_cast_target=target),
        accumulated,
    )

    assert result.data_ptr() == target.data_ptr()
    torch.testing.assert_close(result, accumulated.to(torch.bfloat16))


def test_projection_mixed_config_selects_fixed_mixed_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    caps = _trellis_caps()
    plan = fused_moe.plan(caps)
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.implementation == "trellis_mixed3"
    assert plan._core_workspace_plan.projection_mixed_trellis
    assert specs["intermediate_cache13"].shape == (3072 * 8 * 1024,)
    assert specs["intermediate_cache2"].shape == (3072 * 8 * 512,)
    assert specs["rotation_a_gate"].shape == (3072 * 8, 6144)
    assert specs["rotation_a_up"].shape == (3072 * 8, 6144)
    assert specs["full_rotation_output"].shape == (3072, 6144)


def test_direct_exl3_projection_plan_preserves_k2_tier_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    plan = fused_moe.plan(_exl3_projection_caps())

    assert plan.caps.source_format == "exl3_trellis_mcg"
    assert plan._core_workspace_plan.implementation == "trellis_mixed3"
    assert plan._core_workspace_plan.trellis_bits == 2
    assert plan._core_workspace_plan.projection_mixed_trellis


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


@pytest.mark.parametrize(
    ("source_format", "activation_mode", "packing", "scale_encoding"),
    (
        (
            "fp4_e8m0_k32",
            fused_moe.ActivationMode.A16,
            fused_moe.WeightPacking.MMA_PACKED,
            fused_moe.ScaleEncoding.E8M0_K32,
        ),
        (
            "fp4_e8m0_k32",
            fused_moe.ActivationMode.A8,
            fused_moe.WeightPacking.QMMA_REPACKED,
            fused_moe.ScaleEncoding.E8M0_K32,
        ),
        (
            "modelopt_nvfp4",
            fused_moe.ActivationMode.A4,
            fused_moe.WeightPacking.SOURCE_NATIVE,
            fused_moe.ScaleEncoding.E4M3_K16,
        ),
        (
            "modelopt_nvfp4",
            fused_moe.ActivationMode.A8,
            fused_moe.WeightPacking.SOURCE_NATIVE,
            fused_moe.ScaleEncoding.E8M0_K32_E4M3_RESIDUAL,
        ),
    ),
)
def test_typed_packed_planning_keeps_representation_axes_independent(
    source_format: str,
    activation_mode: fused_moe.ActivationMode,
    packing: fused_moe.WeightPacking,
    scale_encoding: fused_moe.ScaleEncoding,
) -> None:
    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat(source_format),
        w13_layout=fused_moe.W13Layout.W31,
    )
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode=activation_mode,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=8,
            hidden_size=256,
            intermediate_size=256,
        ),
        constraints=fused_moe.WeightPlanConstraints(
            required_packing=packing,
        ),
    )

    assert plan.source is source
    assert plan.activation.mode is activation_mode
    assert plan.prepared_format.weights is fused_moe.WeightEncoding.FP4_E2M1
    assert plan.prepared_format.scales is scale_encoding
    assert plan.prepared_format.packing is packing


@pytest.mark.parametrize(
    ("quant_mode", "source_format", "w13_layout", "weight_layout"),
    (
        ("w4a8_mx", "fp4_e8m0_k32", "w31", "qmma_repacked"),
        ("w4a16", "fp4_e8m0_k32", "w31", "mma_packed"),
        ("nvfp4", "modelopt_nvfp4", "w31", "source_native"),
        ("w4a8_nvfp4", "modelopt_nvfp4", "w31", "source_native"),
        ("w4a16", "modelopt_nvfp4", "w13", "source_native"),
    ),
)
def test_vllm_1_2_6_planning_contract_remains_supported(
    quant_mode: str,
    source_format: str,
    w13_layout: str,
    weight_layout: str,
) -> None:
    plan = fused_moe.plan_weights(
        quant_modes=quant_mode,
        source_format=source_format,
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=256,
        intermediate_size=256,
        w13_layout=w13_layout,
    )

    assert plan.quant_modes == frozenset({quant_mode})
    assert plan.source_format == source_format
    assert plan.w13_layout == w13_layout
    caps = fused_moe.Caps(
        weight_plan=plan,
        quant_mode=quant_mode,
        max_tokens=1,
        num_topk=1,
        device="cpu",
    )
    assert caps.quant_mode == quant_mode
    execution = fused_moe.plan_execution(
        num_tokens=1,
        num_topk=1,
        device="cpu",
        weight_plan=plan,
        quant_mode=quant_mode,
    )
    assert execution.execution.weight_layout.value == weight_layout
    scratch_plan = fused_moe.plan(caps)
    assert fused_moe.required_nbytes(caps) == scratch_plan.scratch_specs()[0].nbytes


def test_vllm_bind_contract_accepts_unit_scale_contract() -> None:
    parameters = inspect.signature(fused_moe.Plan.bind).parameters

    assert "unit_scale_contract" in parameters


def test_vllm_flat_weight_preparation_contract() -> None:
    plan = fused_moe.plan_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=2,
        hidden_size=128,
        intermediate_size=64,
        w13_layout="w31",
    )
    prepared = fused_moe.prepare_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        w1_fp4=torch.zeros((2, 128, 64), dtype=torch.uint8),
        w2_fp4=torch.zeros((2, 128, 32), dtype=torch.uint8),
        w1_blockscale=torch.ones(
            (2, 128, 8),
            dtype=torch.float8_e4m3fn,
        ),
        w2_blockscale=torch.ones(
            (2, 128, 4),
            dtype=torch.float8_e4m3fn,
        ),
        w1_global_scale=torch.ones(2),
        w2_global_scale=torch.ones(2),
        a1_gscale=torch.ones(2),
        a2_gscale=torch.ones(2),
    )

    assert isinstance(prepared, fused_moe.ExpertWeights)
    assert prepared.plan is plan


def test_canonical_lifecycle_keeps_planning_axes_visible() -> None:
    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
        w13_layout=fused_moe.W13Layout.W31,
    )
    activation = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A4,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=2,
        hidden_size=128,
        intermediate_size=64,
    )
    weight_plan = fused_moe.plan_weights(
        source=source,
        activation=activation,
        geometry=geometry,
    )
    weights = fused_moe.PackedWeights(
        w13=torch.zeros((2, 128, 64), dtype=torch.uint8),
        w2=torch.zeros((2, 128, 32), dtype=torch.uint8),
        w13_block_scales=torch.ones(
            (2, 128, 8),
            dtype=torch.float8_e4m3fn,
        ),
        w2_block_scales=torch.ones(
            (2, 128, 4),
            dtype=torch.float8_e4m3fn,
        ),
        w13_global_scales=torch.ones(2),
        w2_global_scales=torch.ones(2),
        input_scale=torch.ones(2),
        intermediate_scale=torch.ones(2),
    )

    experts = fused_moe.prepare_weights(plan=weight_plan, weights=weights)
    execution = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=4,
            top_k=1,
            warmup_token_counts=(1, 4),
        ),
    )

    assert weight_plan.source is source
    assert weight_plan.activation is activation
    assert weight_plan.geometry is geometry
    assert weight_plan.prepared_format.packing is fused_moe.WeightPacking.SOURCE_NATIVE
    assert execution.scratch.nbytes > 0
    assert [variant.tokens for variant in execution.variants] == [1, 4]
    assert not execution.is_prewarmed
    fused_moe.prewarm(execution)
    assert execution.is_prewarmed


def test_canonical_w4a8_lifecycle_preserves_activation_and_packing() -> None:
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
            w13_layout=fused_moe.W13Layout.W31,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A8,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=2,
            hidden_size=256,
            intermediate_size=128,
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=torch.zeros((2, 256, 128), dtype=torch.uint8),
            w2=torch.zeros((2, 256, 64), dtype=torch.uint8),
            w13_block_scales=torch.ones((2, 256, 8), dtype=torch.uint8),
            w2_block_scales=torch.ones((2, 256, 4), dtype=torch.uint8),
            w13_global_scales=torch.ones(2),
            w2_global_scales=torch.ones(2),
        ),
    )
    execution = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=4, top_k=1),
    )

    assert weight_plan.activation.mode is fused_moe.ActivationMode.A8
    assert weight_plan.prepared_format.packing is fused_moe.WeightPacking.QMMA_REPACKED
    assert experts.plan is weight_plan
    assert execution.variants[0].execution.gemm_engine.value == "mxfp8_qmma"


def test_typed_planning_rejects_legacy_recipe_arguments() -> None:
    with pytest.raises(TypeError, match="quant_modes"):
        fused_moe.plan_weights(
            source=fused_moe.PackedSource(
                format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
            ),
            quant_modes="w4a16",
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=8,
                hidden_size=256,
                intermediate_size=256,
            ),
        )


def test_config_keyword_is_not_a_public_planning_contract() -> None:
    assert not hasattr(fused_moe, "PackedConfig")
    with pytest.raises(TypeError, match="config="):
        fused_moe.plan_weights(
            config=object(),
            activation="silu",
            dtype=torch.bfloat16,
        )
