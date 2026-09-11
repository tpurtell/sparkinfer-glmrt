"""The V4.1 direct/split plan is restricted to the qualified Spark geometry."""
from dataclasses import replace
import pytest
from b12x.moe.fused_moe._policy import MoeDecodeQuery, MoeDecodeConfig, validate_moe_decode_config

QUERY = MoeDecodeQuery(quant_mode="w4a8_mx", source_format="fp4_e8m0_k32",
    activation="silu_v41", num_experts=384, hidden_size=5120,
    intermediate_size=576, top_k=6, num_tokens=1, routed_rows=6)
CONFIG = MoeDecodeConfig(backend="dynamic", route_planner="internal",
    max_active_clusters=None, dynamic_tile_m=16, dynamic_route_mode="direct")

def test_native_direct_geometry():
    validate_moe_decode_config(QUERY, CONFIG, None)

@pytest.mark.parametrize("change", [dict(num_tokens=2, routed_rows=12),
    dict(hidden_size=4096), dict(intermediate_size=2304), dict(num_experts=128),
    dict(top_k=3, routed_rows=3), dict(source_format="modelopt_nvfp4"),
    dict(quant_mode="nvfp4")])
def test_reject_unqualified_direct_geometry(change):
    with pytest.raises(ValueError, match="direct V4.1"):
        validate_moe_decode_config(replace(QUERY, **change), CONFIG, None)
    validate_moe_decode_config(replace(QUERY, **change),
        replace(CONFIG, dynamic_route_mode="grouped"), None)
