"""Shared activation metadata for fused MoE paths."""

from __future__ import annotations

import math


SILU_V41 = "silu_v41"
SITU = "situ"
SITU_DEFAULT_BETA = 4.0
SITU_DEFAULT_LINEAR_BETA = 25.0
SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
# Step-5 SwiGLU: min(silu(gate), limit) * clamp(up, +-limit), multiplied in
# FP32 and rounded once. The gate is clamped after SiLU, never before it.
SILU_POST_CLAMP = "silu_post_clamp"
SWIGLUOAI_DEFAULT_LIMIT = 7.0
SWIGLUOAI_DEFAULT_ALPHA = 1.702
SWIGLUOAI_DEFAULT_BETA = 1.0

SUPPORTED_MOE_ACTIVATIONS = frozenset(
    {"silu", SILU_V41, SITU, "relu2", SWIGLUOAI_UNINTERLEAVE, SILU_POST_CLAMP}
)
GATED_MOE_ACTIVATIONS = frozenset(
    {"silu", SILU_V41, SITU, SWIGLUOAI_UNINTERLEAVE, SILU_POST_CLAMP}
)
# Only the W4A16 trellis kernels and their reference implement these.
TRELLIS_ONLY_MOE_ACTIVATIONS = frozenset({SILU_POST_CLAMP})


def reject_trellis_only_activation(activation: str, kernel: str) -> None:
    if activation in TRELLIS_ONLY_MOE_ACTIVATIONS:
        raise ValueError(f"{kernel} does not implement activation {activation!r}")


def normalize_moe_activation(activation: str) -> str:
    activation = str(activation)
    if activation not in SUPPORTED_MOE_ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    return activation


def is_gated_moe_activation(activation: str) -> bool:
    return normalize_moe_activation(activation) in GATED_MOE_ACTIVATIONS


def moe_activation_w1_rows(activation: str, n: int) -> int:
    return (2 if is_gated_moe_activation(activation) else 1) * int(n)


def normalize_swiglu_limit_for_activation(
    activation: str,
    swiglu_limit: float | None,
) -> float | None:
    activation = normalize_moe_activation(activation)
    if activation == SILU_V41:
        if swiglu_limit is not None and float(swiglu_limit) != 10.0:
            raise ValueError("silu_v41 requires the trained SwiGLU limit 10")
        return 10.0
    if swiglu_limit is None:
        if activation == SILU_POST_CLAMP:
            raise ValueError("silu_post_clamp requires a positive swiglu_limit")
        return SWIGLUOAI_DEFAULT_LIMIT if activation == SWIGLUOAI_UNINTERLEAVE else None
    if activation == SITU:
        raise ValueError("swiglu_limit is not part of the situ activation contract")
    if activation not in GATED_MOE_ACTIVATIONS:
        raise ValueError("swiglu_limit requires a gated MoE activation")
    limit = float(swiglu_limit)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"swiglu_limit must be positive and finite, got {limit}")
    return limit


def normalize_swiglu_alpha_for_activation(
    activation: str,
    swiglu_alpha: float | None,
) -> float:
    activation = normalize_moe_activation(activation)
    if activation != SWIGLUOAI_UNINTERLEAVE:
        if swiglu_alpha is not None and float(swiglu_alpha) != 1.0:
            raise ValueError(
                "swiglu_alpha is only configurable for swigluoai_uninterleave"
            )
        return 1.0
    alpha = SWIGLUOAI_DEFAULT_ALPHA if swiglu_alpha is None else float(swiglu_alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"swiglu_alpha must be finite, got {alpha}")
    return alpha


def normalize_swiglu_beta_for_activation(
    activation: str,
    swiglu_beta: float | None,
) -> float:
    activation = normalize_moe_activation(activation)
    if activation != SWIGLUOAI_UNINTERLEAVE:
        if swiglu_beta is not None and float(swiglu_beta) != 0.0:
            raise ValueError(
                "swiglu_beta is only configurable for swigluoai_uninterleave"
            )
        return 0.0
    beta = SWIGLUOAI_DEFAULT_BETA if swiglu_beta is None else float(swiglu_beta)
    if not math.isfinite(beta):
        raise ValueError(f"swiglu_beta must be finite, got {beta}")
    return beta
