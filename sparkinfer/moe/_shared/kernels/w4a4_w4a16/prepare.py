"""Single-residency weight preparation for W4A4 FC1 / W4A16 FC2.

The serving representation is deliberately asymmetric:

* W13 remains in checkpoint-native ModelOpt NVFP4 layout for W4A4 FC1.
* W2 is repacked in place into the tensor-core W4A16 layout for BF16 FC2.

There is no packed W13 member and no source-layout W2 member in the returned
object.  The caller must replace its source-W2 reference with the returned
packed view after preparation; both views alias the same allocation while the
in-place transform is running.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sparkinfer._lib.intrinsics import as_grouped_scale_view
from sparkinfer.moe._shared.kernels.w4a16.host import (
    unswizzle_expert_scales,
    validate_w4a16_packed_inputs,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    _normalize_w13_layout,
    _permute_nvfp4_scales,
    _repack_weight,
    _source_global_scale,
)


@dataclass(frozen=True)
class W4A4FC1W4A16FC2Weights:
    """Persistent model-level tensors for the mixed Spark recipe."""

    w13_weight_source: torch.Tensor
    w13_scale_source: torch.Tensor
    w13_runtime_alpha: torch.Tensor
    w2_weight_packed: torch.Tensor
    w2_scale_packed: torch.Tensor
    w2_w4a16_alpha: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    w13_layout: str
    source_format: str = "modelopt_nvfp4"
    storage_policy: str = "source_w13_packed_w2_in_place"

    @property
    def has_packed_w13(self) -> bool:
        return False

    @property
    def has_source_w2(self) -> bool:
        return False


@dataclass(frozen=True)
class BoundW4A4FC1W4A16FC2Expert:
    """Allocation-free one-expert views used by a Spark CUDA graph."""

    w13_weight_source: torch.Tensor
    w13_scale_source: torch.Tensor
    w13_runtime_alpha: torch.Tensor
    w2_weight_packed: torch.Tensor
    w2_scale_packed: torch.Tensor
    w2_w4a16_alpha: torch.Tensor


def prepare_w4a4_fc1_w4a16_fc2_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
) -> W4A4FC1W4A16FC2Weights:
    """Keep W13 source-native and repack W2 in place for W4A16.

    ``w13_runtime_alpha`` is exactly the raw checkpoint W13 global weight
    scale.  A prequantized NVFP4 payload already carries complete per-K16
    activation scales, so this contract intentionally has no input-global
    multiplier.  ``w2_w4a16_alpha`` is the compensated BF16 W4A16 scalar
    produced by the normal packed-weight preparation math.
    """

    if params_dtype != torch.bfloat16:
        raise TypeError("the mixed Spark recipe currently requires BF16 FC2")
    w13_layout = _normalize_w13_layout(w13_layout)
    if w13_layout != "w13":
        raise ValueError("the Spark mixed recipe requires source [up; gate] W13 order")
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    for name, tensor in (
        ("w13_fp4", w13_fp4),
        ("w13_blockscale", w13_blockscale),
        ("w2_fp4", w2_fp4),
        ("w2_blockscale", w2_blockscale),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for single-residency prep")
    if w13_fp4.device != w2_fp4.device:
        raise ValueError("W13 and W2 must be on the same device")

    # Materialize the small logical scale grid before its source storage can be
    # released.  W13 scale storage remains untouched for the dense W4A4 FC1.
    w2_scale = unswizzle_expert_scales(
        w2_blockscale,
        rows=shape.hidden_size,
        cols=shape.intermediate_size,
    )
    source_w13_alpha = _source_global_scale(
        w13_global_scale,
        source_format="modelopt_nvfp4",
    )
    source_w2_alpha = _source_global_scale(
        w2_global_scale,
        source_format="modelopt_nvfp4",
    )

    source_w2_storage = w2_fp4.untyped_storage().data_ptr()
    packed_w2 = _repack_weight(
        w2_fp4,
        size_k=shape.intermediate_size,
        size_n=shape.hidden_size,
        reuse_input_storage=True,
    )
    if packed_w2.untyped_storage().data_ptr() != source_w2_storage:
        raise RuntimeError(
            "in-place W2 preparation unexpectedly allocated a weight copy"
        )
    packed_w2_scale, packed_w2_alpha = _permute_nvfp4_scales(
        w2_scale,
        source_w2_alpha,
        size_k=shape.intermediate_size,
        size_n=shape.hidden_size,
        a_dtype=params_dtype,
    )

    return W4A4FC1W4A16FC2Weights(
        w13_weight_source=w13_fp4,
        w13_scale_source=w13_blockscale,
        w13_runtime_alpha=source_w13_alpha,
        w2_weight_packed=packed_w2,
        w2_scale_packed=packed_w2_scale,
        w2_w4a16_alpha=packed_w2_alpha,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        is_gated=shape.is_gated,
        params_dtype=params_dtype,
        w13_layout=w13_layout,
    )


def bind_w4a4_fc1_w4a16_fc2_expert(
    prepared: W4A4FC1W4A16FC2Weights,
    expert: int,
) -> BoundW4A4FC1W4A16FC2Expert:
    """Bind one expert without allocation or a weight/scale copy."""

    if not isinstance(prepared, W4A4FC1W4A16FC2Weights):
        raise TypeError("prepared must be W4A4FC1W4A16FC2Weights")
    expert = int(expert)
    if not 0 <= expert < prepared.num_experts:
        raise IndexError(f"expert {expert} is outside [0, {prepared.num_experts})")
    w13 = prepared.w13_weight_source[expert].unsqueeze(-1)
    w13_scale_storage = prepared.w13_scale_source[expert : expert + 1]
    w13_scale = as_grouped_scale_view(
        w13_scale_storage,
        2 * prepared.intermediate_size
        if prepared.is_gated
        else prepared.intermediate_size,
        prepared.hidden_size,
    )
    return BoundW4A4FC1W4A16FC2Expert(
        w13_weight_source=w13,
        w13_scale_source=w13_scale,
        w13_runtime_alpha=prepared.w13_runtime_alpha[expert : expert + 1],
        w2_weight_packed=prepared.w2_weight_packed[expert : expert + 1],
        w2_scale_packed=prepared.w2_scale_packed[expert : expert + 1],
        w2_w4a16_alpha=prepared.w2_w4a16_alpha[expert : expert + 1],
    )


__all__ = [
    "BoundW4A4FC1W4A16FC2Expert",
    "W4A4FC1W4A16FC2Weights",
    "bind_w4a4_fc1_w4a16_fc2_expert",
    "prepare_w4a4_fc1_w4a16_fc2_weights",
]
