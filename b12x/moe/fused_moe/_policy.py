"""Typed component-policy contract for fused-MoE decode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from b12x.policy import (
    MOE_DECODE,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True)
class MoeDecodeQuery:
    quant_mode: str
    source_format: str
    activation: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_tokens: int
    routed_rows: int

    def profile_fields(self) -> dict[str, object]:
        return {
            "activation": self.activation,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_tokens": self.num_tokens,
            "quant_mode": self.quant_mode,
            "routed_rows": self.routed_rows,
            "source_format": self.source_format,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class MoeDecodeConfig:
    backend: str
    route_planner: str
    max_active_clusters: int | None

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "MoeDecodeConfig":
        expected = {
            "backend",
            "max_active_clusters",
            "route_planner",
        }
        if set(payload) != expected:
            raise ValueError(
                "MoE decode profile fields must be exactly "
                f"{sorted(expected)}; got {sorted(payload)}"
            )
        backend = payload["backend"]
        route_planner = payload["route_planner"]
        max_active_clusters = payload["max_active_clusters"]
        if not isinstance(backend, str) or not isinstance(route_planner, str):
            raise TypeError("MoE backend and route_planner must be strings")
        if max_active_clusters is not None and (
            not isinstance(max_active_clusters, int)
            or isinstance(max_active_clusters, bool)
        ):
            raise TypeError("max_active_clusters must be an integer or null")
        return cls(
            backend=backend,
            route_planner=route_planner,
            max_active_clusters=max_active_clusters,
        )


def validate_moe_decode_config(
    query: MoeDecodeQuery,
    config: MoeDecodeConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend not in {"micro", "dynamic"}:
        raise ValueError(f"unsupported MoE backend {config.backend!r}")
    if config.route_planner not in {"internal", "triton"}:
        raise ValueError(f"unsupported MoE route planner {config.route_planner!r}")
    if config.route_planner == "triton" and config.backend != "dynamic":
        raise ValueError("the Triton route planner requires dynamic MoE")
    if config.route_planner == "triton" and not (
        query.quant_mode == "nvfp4"
        and query.activation == "silu"
        and 0 < query.routed_rows <= 256
    ):
        raise ValueError(
            "the Triton route planner only supports small NVFP4 SiLU workloads"
        )
    if config.max_active_clusters is not None and config.max_active_clusters <= 0:
        raise ValueError("max_active_clusters must be positive when set")


def make_moe_decode_policy(
    heuristic: Callable[
        [MoeDecodeQuery, DeviceIdentity | None],
        MoeDecodeConfig,
    ],
) -> ComponentPolicy[MoeDecodeQuery, MoeDecodeConfig]:
    return ComponentPolicy(
        component_id=MOE_DECODE,
        query_schema_version=2,
        config_schema_version=1,
        query_fields=frozenset(
            {
                "activation",
                "hidden_size",
                "intermediate_size",
                "num_experts",
                "num_tokens",
                "quant_mode",
                "routed_rows",
                "source_format",
                "top_k",
            }
        ),
        config_fields=frozenset(
            {
                "backend",
                "max_active_clusters",
                "route_planner",
            }
        ),
        encode_query=MoeDecodeQuery.profile_fields,
        decode_profile=MoeDecodeConfig.from_profile,
        heuristic=heuristic,
        validate_config=validate_moe_decode_config,
    )


def _heuristic(
    query: MoeDecodeQuery,
    device: DeviceIdentity | None,
) -> MoeDecodeConfig:
    from ._impl import _heuristic_moe_decode_config

    return _heuristic_moe_decode_config(query, device)


MOE_DECODE_POLICY = make_moe_decode_policy(_heuristic)


__all__ = [
    "MOE_DECODE_POLICY",
    "MoeDecodeConfig",
    "MoeDecodeQuery",
    "make_moe_decode_policy",
    "validate_moe_decode_config",
]
