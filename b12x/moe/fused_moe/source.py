"""Checkpoint-side fused-MoE weight representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .config import TrellisCodebook, TrellisConfig


class PackedSourceFormat(str, Enum):
    """Packed checkpoint encoding, including its scale-grid contract."""

    MXFP4_E8M0_K32 = "fp4_e8m0_k32"
    MODELOPT_NVFP4 = "modelopt_nvfp4"
    COMPRESSED_TENSORS_FP4 = "compressed_tensors"
    MXFP6_E8M0_K32 = "mxfp6_e2m3"


class W13Layout(str, Enum):
    """Logical order of the two gated FC1 projections."""

    W13 = "w13"
    W31 = "w31"


@dataclass(frozen=True, kw_only=True)
class PackedSource:
    """Source encoding and packing of one ordinary MoE checkpoint."""

    format: PackedSourceFormat
    w13_layout: W13Layout = W13Layout.W13

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", PackedSourceFormat(self.format))
        object.__setattr__(self, "w13_layout", W13Layout(self.w13_layout))


@dataclass(frozen=True, kw_only=True)
class Exl3TrellisSource:
    """Uniform EXL3 MCG/MUL1 projection tiles with independent Hadamard vectors."""

    bits: int
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256)
    codebook: TrellisCodebook | str = TrellisCodebook.MCG

    def __post_init__(self) -> None:
        if isinstance(self.codebook, str):
            try:
                resolved = TrellisCodebook(self.codebook)
            except ValueError as exc:
                raise ValueError(
                    "uniform EXL3 codebook must be 'mcg' or 'mul1'"
                ) from exc
        elif isinstance(self.codebook, TrellisCodebook):
            resolved = self.codebook
        else:
            raise TypeError("EXL3 codebook must be TrellisCodebook or str")
        if resolved not in (TrellisCodebook.MCG, TrellisCodebook.MUL1):
            raise ValueError("uniform EXL3 codebook must be 'mcg' or 'mul1'")
        object.__setattr__(self, "codebook", resolved)
        if resolved is TrellisCodebook.MCG:
            if type(self.bits) is not int or self.bits not in (3, 4, 5, 6):
                raise ValueError("uniform EXL3 MCG rate must be K3, K4, K5, or K6")
        elif type(self.bits) is not int or self.bits not in (2, 3, 4, 5, 6):
            raise ValueError("uniform EXL3 MUL1 rate must be K2, K3, K4, K5, or K6")
        if (
            len(self.tile_config) != 4
            or any(type(value) is not int or value <= 0 for value in self.tile_config)
        ):
            raise ValueError("EXL3 tile_config must contain four positive integers")


WeightSource: TypeAlias = PackedSource | TrellisConfig | Exl3TrellisSource


__all__ = [
    "PackedSource",
    "PackedSourceFormat",
    "Exl3TrellisSource",
    "W13Layout",
    "WeightSource",
]
