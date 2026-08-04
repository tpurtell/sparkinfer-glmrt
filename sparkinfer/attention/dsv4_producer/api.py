"""Public surface for :mod:`sparkinfer.attention.dsv4_producer`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._impl import (
    DSV4ProducerBinding as Binding,
    DSV4ProducerCaps as Caps,
    DSV4ProducerPlan as Plan,
    DSV4ProducerWeights as Weights,
    pack_dsv4_producer_weights as pack_weights,
    plan_dsv4_producer as plan,
    run_dsv4_producer as run,
)


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch; creates views only."""

    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with the block-FP8 GEMM dependencies installed."""

    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "Weights",
    "plan",
    "bind",
    "pack_weights",
    "run",
    "is_supported",
]
