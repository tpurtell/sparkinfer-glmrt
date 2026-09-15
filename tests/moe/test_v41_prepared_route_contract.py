"""Keep native V4.1 route eligibility distinct from upstream compact SiLU."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import (
    MoeDecodeConfig,
    MoeDecodeQuery,
    validate_moe_decode_config,
)
from b12x.preparation import FrozenMapping


def _query():
    return MoeDecodeQuery(
        quant_mode="w4a8_mx", quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32", activation="silu_v41",
        io_dtype="bfloat16", num_experts=384, hidden_size=5120,
        intermediate_size=576, top_k=6, num_tokens=1, routed_rows=6,
        route_num_experts=None, route_logits_dtype=None,
        apply_router_weight_on_input=False, collect_activation_amax=False,
        deterministic_output=True, swiglu_limit=None, swiglu_alpha=None,
        swiglu_beta=None, w13_layout="split", weight_layouts=("fp4_e8m0_k32",),
        w4a16_weight_layout=None, w4a16_scale_format=None,
        w4a16_block_size_m=None, fast_math=True, numerical_recipe="default",
        controls=FrozenMapping(),
    )


def _direct():
    return MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=16, dynamic_route_mode="direct",
    )


def test_native_spark_direct_route_remains_valid():
    validate_moe_decode_config(_query(), _direct(), None)


@pytest.mark.parametrize("changes", [
    {"num_tokens": 2, "routed_rows": 12}, {"intermediate_size": 1152},
    {"hidden_size": 4096}, {"num_experts": 256}, {"top_k": 8},
    {"source_format": "modelopt_nvfp4"},
])
def test_native_direct_rejects_other_geometry(changes):
    with pytest.raises(ValueError, match="direct V4.1"):
        validate_moe_decode_config(replace(_query(), **changes), _direct(), None)


@pytest.mark.parametrize("changes", [
    {"dynamic_tile_m": 32}, {"route_planner": "triton"},
])
def test_native_direct_rejects_other_launch_contract(changes):
    with pytest.raises(ValueError, match="direct V4.1"):
        validate_moe_decode_config(_query(), replace(_direct(), **changes), None)


def test_upstream_compact_silu_still_requires_grouped_route():
    query = replace(_query(), activation="silu")
    validate_moe_decode_config(
        query, replace(_direct(), dynamic_route_mode="grouped"), None,
    )
    with pytest.raises(ValueError, match="compact N64"):
        validate_moe_decode_config(query, _direct(), None)
