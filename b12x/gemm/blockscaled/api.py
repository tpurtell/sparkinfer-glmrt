"""Public surface for gemm.blockscaled (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._linear import (
    Weight,
    blockscaled_mm as mm,
    pack_weight,
    prewarm,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Weight",
    "is_supported",
    "mm",
    "pack_weight",
    "prewarm",
]
