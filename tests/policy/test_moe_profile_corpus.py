from __future__ import annotations

from benchmarks.benchmark_moe import MODEL_PROFILES
from b12x.policy.generation.moe_corpus import (
    COMMON_DECODE_TOKENS,
    COMMON_MOE_MODELS,
    COMMON_PLAN_TOKEN_COUNTS,
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_ROUTE_PATTERNS,
    COMMON_TP_SIZES,
    MOE_BENCHMARK_PRESETS,
    MOE_RECIPES,
    MoeModelGeometry,
    MoeRecipe,
    corpus_manifest,
    expand_physical_geometries,
    expand_sweep_cases,
)


def test_common_models_expand_across_tp1_through_tp16() -> None:
    geometries = expand_physical_geometries()
    covered = {
        (alias.model_id, geometry.recipe.recipe_id, alias.tp_size)
        for geometry in geometries
        for alias in geometry.aliases
    }

    for model in COMMON_MOE_MODELS:
        for recipe_id in model.recipe_ids:
            for tp_size in COMMON_TP_SIZES:
                assert (model.model_id, recipe_id, tp_size) in covered


def test_unaligned_three_wide_shard_is_padded_instead_of_rejected() -> None:
    recipe = MoeRecipe(
        recipe_id="nvfp4-test",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    )
    model = MoeModelGeometry(
        model_id="small-test",
        hidden_size=256,
        intermediate_size=47,
        num_experts=16,
        native_top_k=2,
        activation="silu",
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=(16,),
    )

    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    (alias,) = geometry.aliases

    assert alias.logical_intermediate_sizes == (2, 3)
    assert alias.physical_intermediate_size == 16
    assert alias.padding_per_tp_group == 209


def test_three_wide_trellis_shard_uses_the_kernel_minimum() -> None:
    recipe = MoeRecipe(
        recipe_id="trellis-test",
        quant_mode="w4a16",
        source_format="btx",
        intermediate_alignment=256,
        minimum_intermediate_size=256,
        trellis_variant="k3-sqg-uniform-coupled",
    )
    model = MoeModelGeometry(
        model_id="small-trellis-test",
        hidden_size=512,
        intermediate_size=47,
        num_experts=16,
        native_top_k=2,
        activation="silu",
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=(16,),
    )

    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    (alias,) = geometry.aliases

    assert alias.logical_intermediate_sizes == (2, 3)
    assert alias.physical_intermediate_size == 256


def test_three_wide_modelopt_w4a16_shard_uses_the_main_kernel_minimum() -> None:
    recipe = MoeRecipe(
        recipe_id="modelopt-w4a16-test",
        quant_mode="w4a16",
        source_format="modelopt_nvfp4",
        intermediate_alignment=64,
        minimum_intermediate_size=64,
    )
    model = MoeModelGeometry(
        model_id="small-w4a16-test",
        hidden_size=256,
        intermediate_size=47,
        num_experts=16,
        native_top_k=2,
        activation="silu",
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=(16,),
    )

    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    (alias,) = geometry.aliases

    assert alias.logical_intermediate_sizes == (2, 3)
    assert alias.physical_intermediate_size == 64


def test_nondivisible_tp_shards_share_one_padded_physical_geometry() -> None:
    geometries = expand_physical_geometries()
    aliases = [
        (geometry, alias)
        for geometry in geometries
        for alias in geometry.aliases
        if alias.model_id == "qwen3.8-flash-next-180b"
        and alias.tp_size == 3
        and geometry.recipe.recipe_id == "modelopt-nvfp4"
    ]

    assert len(aliases) == 1
    geometry, alias = aliases[0]
    assert alias.logical_intermediate_sizes == (213, 214)
    assert geometry.intermediate_size == 224


def test_sweep_deduplicates_model_aliases_before_crossing_runtime_axes() -> None:
    recipe = MoeRecipe(
        recipe_id="same",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    )
    shared = {
        "hidden_size": 256,
        "intermediate_size": 64,
        "num_experts": 16,
        "native_top_k": 2,
        "activation": "silu",
        "recipe_ids": (recipe.recipe_id,),
        "source": "test",
        "tp_sizes": (1,),
    }
    models = (
        MoeModelGeometry(model_id="left", **shared),
        MoeModelGeometry(model_id="right", **shared),
    )

    (geometry,) = expand_physical_geometries(
        models=models,
        recipes=(recipe,),
    )
    cases = expand_sweep_cases(
        geometries=(geometry,),
        top_ks=(2,),
        token_counts=(1,),
        route_patterns=("balanced",),
    )

    assert [alias.model_id for alias in geometry.aliases] == ["left", "right"]
    assert len(cases) == 1


def test_default_sweep_has_stable_complete_cross_product() -> None:
    geometries = expand_physical_geometries()
    cases = expand_sweep_cases(geometries=geometries)

    assert len(geometries) == 253
    assert len(cases) == 138_652
    assert {case.num_tokens for case in cases} == set(COMMON_PLAN_TOKEN_COUNTS)
    assert {case.route_pattern for case in cases} == set(COMMON_ROUTE_PATTERNS)
    assert len({case.case_id for case in cases}) == len(cases)
    assert len(corpus_manifest()["corpus_sha256"]) == 64


def test_moe_token_axis_separates_decode_and_prefill_capacities() -> None:
    assert COMMON_DECODE_TOKENS == (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
    assert COMMON_PREFILL_TOKEN_CAPACITIES == (512, 1_024, 2_048, 4_096, 8_192)
    assert (
        *COMMON_DECODE_TOKENS,
        *COMMON_PREFILL_TOKEN_CAPACITIES,
    ) == COMMON_PLAN_TOKEN_COUNTS
    assert corpus_manifest()["prefill_token_capacities"] == list(
        COMMON_PREFILL_TOKEN_CAPACITIES
    )


def test_glm_benchmark_recipes_expand_across_all_profiled_tp_sizes() -> None:
    geometries = expand_physical_geometries()
    covered = {
        (alias.model_id, geometry.recipe.quant_mode, alias.tp_size)
        for geometry in geometries
        for alias in geometry.aliases
        if alias.model_id.startswith("glm-")
    }

    for tp_size in COMMON_TP_SIZES:
        assert ("glm-5.2", "w4a8_nvfp4", tp_size) in covered
        assert ("glm-5.3-flash", "nvfp4", tp_size) in covered
        assert ("glm-5.3-flash", "w4a16", tp_size) in covered


def test_moe_benchmark_preset_catalog_is_fully_mapped_to_the_corpus() -> None:
    expected = {
        "deepseek-v4-flash",
        "dsv4f",
        "dsv4f-nvfp4",
        "glm51",
        "glm52",
        "glm53-flash-shape",
        "laguna-s21-shape",
        "minimax-m27",
        "minimax-m3",
        "minimax-m3-shape",
        "nano35-w4a16",
        "nano35-w4a16-shape",
        "nemotron-backbone",
        "qwen38-flash-next",
        "qwen38-flash-next-shape",
        "qwen397b",
    }
    assert set(MODEL_PROFILES) == expected
    assert {preset.preset_id for preset in MOE_BENCHMARK_PRESETS} == expected
    models = {model.model_id: model for model in COMMON_MOE_MODELS}
    recipes = {recipe.recipe_id: recipe for recipe in MOE_RECIPES}
    for preset in MOE_BENCHMARK_PRESETS:
        assert preset.model_id in models
        assert preset.recipe_id in models[preset.model_id].recipe_ids
        benchmark = MODEL_PROFILES[preset.preset_id]
        assert preset.tp_size == benchmark.tp_size
        assert recipes[preset.recipe_id].quant_mode == (
            benchmark.default_quant_mode or "nvfp4"
        )
