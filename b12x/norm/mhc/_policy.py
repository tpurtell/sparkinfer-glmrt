"""Typed component policy for mHC residual planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import MHC, BackendConfig, make_fixed_backend_policy


@dataclass(frozen=True, kw_only=True)
class MhcQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    split_k: int


MHC_POLICY = make_fixed_backend_policy(
    component_id=MHC,
    query_type=MhcQuery,
    backend="native",
)
MhcConfig = BackendConfig


__all__ = ["MHC_POLICY", "MhcConfig", "MhcQuery"]
