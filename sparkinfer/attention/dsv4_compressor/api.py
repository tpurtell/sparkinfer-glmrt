"""Public surface for :mod:`sparkinfer.attention.dsv4_compressor`."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._impl import (
    DSV4CompressorBinding as Binding,
    DSV4CompressorCaps as Caps,
    DSV4CompressorPlan as Plan,
    DSV4CompressorPrefillBinding as PrefillBinding,
    DSV4CompressorWeights as Weights,
    pack_dsv4_compressor_weights as pack_weights,
    plan_dsv4_compressor as plan,
    run_dsv4_compressor_decode as run_decode,
    run_dsv4_compressor_prefill as run_prefill,
)


def bind_decode(plan: Plan, **kwargs) -> Binding:
    """Bind decode tensors and caller-owned scratch; creates views only."""

    return plan.bind_decode(**kwargs)


def bind_prefill(plan: Plan, **kwargs) -> PrefillBinding:
    """Bind initial-prefill tensors and caller-owned scratch; creates views only."""

    return plan.bind_prefill(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with Triton installed."""

    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "PrefillBinding",
    "Weights",
    "plan",
    "bind_decode",
    "bind_prefill",
    "pack_weights",
    "run_decode",
    "run_prefill",
    "is_supported",
]
