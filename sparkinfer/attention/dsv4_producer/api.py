"""Public surface for :mod:`sparkinfer.attention.dsv4_producer`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._impl import (
    DSV4IndexerProducerBinding as IndexerBinding,
    DSV4IndexerProducerCaps as IndexerCaps,
    DSV4IndexerProducerPlan as IndexerPlan,
    DSV4IndexerProducerWeights as IndexerWeights,
    DSV4ProducerBinding as Binding,
    DSV4ProducerCaps as Caps,
    DSV4ProducerPlan as Plan,
    DSV4ProducerWeights as Weights,
    pack_dsv4_indexer_producer_weights as pack_indexer_weights,
    pack_dsv4_producer_weights as pack_weights,
    plan_dsv4_indexer_producer as plan_indexer,
    plan_dsv4_producer as plan,
    run_dsv4_indexer_producer as run_indexer,
    run_dsv4_producer as run,
)


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch; creates views only."""

    return plan.bind(**kwargs)


def bind_indexer(plan: IndexerPlan, **kwargs) -> IndexerBinding:
    """Bind learned-index producer tensors and caller-owned scratch; views only."""

    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with the block-FP8 GEMM dependencies installed."""

    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "IndexerCaps",
    "Plan",
    "IndexerPlan",
    "Binding",
    "IndexerBinding",
    "Weights",
    "IndexerWeights",
    "plan",
    "plan_indexer",
    "bind",
    "bind_indexer",
    "pack_weights",
    "pack_indexer_weights",
    "run",
    "run_indexer",
    "is_supported",
]
