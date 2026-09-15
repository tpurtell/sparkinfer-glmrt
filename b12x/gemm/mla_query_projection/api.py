"""Prepared public API for fused MLA query projection."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias

import torch

from b12x.preparation import Plan
from ..._lib.gating import default_is_supported
from . import META
from . import _bf16
from ._preparation import plan

Mxfp8Weight: TypeAlias = tuple[torch.Tensor, torch.Tensor]
MlaQueryWeight: TypeAlias = torch.Tensor | Mxfp8Weight
GlmH64Bf16Workload: TypeAlias = Literal["packed_decode", "prefill"]
GlmH64Bf16Policy: TypeAlias = Literal["auto", "force", "disable"]
GlmH64Bf16Backend: TypeAlias = Literal["native", "sparkinfer_glm_h64_bf16"]


@dataclass(frozen=True, kw_only=True)
class GlmH64Bf16QueryProjectionPlan:
    """Capture-static backend decision for the explicit GLM H=64 contract."""

    backend: GlmH64Bf16Backend
    workload: GlmH64Bf16Workload
    policy: GlmH64Bf16Policy
    query_rows: int
    h64_supported: bool
    reason: str

    @property
    def use_sparkinfer(self) -> bool:
        return self.backend == "sparkinfer_glm_h64_bf16"


def plan_glm_h64_bf16(
    *,
    workload: GlmH64Bf16Workload,
    policy: GlmH64Bf16Policy,
    query_rows: int,
    num_heads: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device=None,
) -> GlmH64Bf16QueryProjectionPlan:
    """Choose SparkInfer H64 or the caller's native fallback.

    Automatic policy is deliberately narrower than kernel support: the
    all-layer GLMRT gate initially promotes packed decode with ``M=2..16``.
    H64 remains available through ``policy="force"`` for the measured M=1 and
    prefill diagnostics, without changing automatic serving behavior.  Force
    means "use H64 wherever supported"; unsupported buckets retain the native
    fallback so a full mixed-size qualification run remains executable.
    """
    if workload not in ("packed_decode", "prefill"):
        raise ValueError(
            "GLM H64 BF16 workload must be 'packed_decode' or 'prefill', "
            f"got {workload!r}"
        )
    if policy not in ("auto", "force", "disable"):
        raise ValueError(
            f"GLM H64 BF16 policy must be 'auto', 'force', or 'disable', got {policy!r}"
        )
    rows = int(query_rows)
    supported = can_implement_glm_h64_bf16(
        num_heads=num_heads,
        max_m=rows,
        nope_dim=nope_dim,
        latent_dim=latent_dim,
        output_dtype=output_dtype,
        device=device,
    )
    if policy == "force":
        if supported:
            return GlmH64Bf16QueryProjectionPlan(
                backend="sparkinfer_glm_h64_bf16",
                workload=workload,
                policy=policy,
                query_rows=rows,
                h64_supported=True,
                reason="explicit_force",
            )
        return GlmH64Bf16QueryProjectionPlan(
            backend="native",
            workload=workload,
            policy=policy,
            query_rows=rows,
            h64_supported=False,
            reason="explicit_force_unsupported_native_fallback",
        )
    if policy == "disable":
        return GlmH64Bf16QueryProjectionPlan(
            backend="native",
            workload=workload,
            policy=policy,
            query_rows=rows,
            h64_supported=supported,
            reason="explicit_disable",
        )
    if supported and workload == "packed_decode" and 2 <= rows <= 16:
        return GlmH64Bf16QueryProjectionPlan(
            backend="sparkinfer_glm_h64_bf16",
            workload=workload,
            policy=policy,
            query_rows=rows,
            h64_supported=True,
            reason="automatic_packed_decode_m2_m16",
        )
    reason = (
        "automatic_native_pending_m1_gate"
        if supported and workload == "packed_decode" and rows == 1
        else (
            "automatic_native_pending_prefill_gate"
            if supported and workload == "prefill"
            else "h64_contract_unsupported"
        )
    )
    return GlmH64Bf16QueryProjectionPlan(
        backend="native",
        workload=workload,
        policy=policy,
        query_rows=rows,
        h64_supported=supported,
        reason=reason,
    )


def run_glm_h64_bf16(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    out: torch.Tensor,
    *,
    stream: Optional[object] = None,
) -> torch.Tensor:
    """Run the explicit one-launch GLM H=64 BF16 query projection.

    This intentionally does not widen :func:`run`'s generic BF16 geometry.
    It accepts a head-major ``q_nope`` view ``[64,M,192]`` plus a 64-wide
    ``q_pe``, or GLM-5.3's NoPE ``q_nope`` view ``[64,M,256]`` plus an empty
    ``q_pe`` view.  The corresponding K slice of the resident KV-B tensor is
    ``[64,K,512]`` and caller-owned BF16 ``out`` is ``[M,64,576]``.  The
    NoPE form writes an exact-zero suffix directly in the fused epilogue.
    Arbitrary outer strides are supported; innermost dimensions must remain
    contiguous.  ``1 <= M <= 32``.
    """
    return _bf16.run_glm_h64_bf16(
        q_nope,
        weight,
        q_pe,
        out,
        stream=stream,
    )


def prewarm_glm_h64_bf16(
    weight: torch.Tensor,
    m_values: Iterable[int],
    *,
    stream: Optional[object] = None,
    synchronize: bool = True,
    nope: bool = False,
) -> int:
    """Compile and first-launch the declared H=64 BF16 graph regimes."""
    return _bf16.prewarm_glm_h64_bf16(
        weight,
        m_values,
        stream=stream,
        synchronize=synchronize,
        nope=nope,
    )


def can_implement_glm_h64_bf16(
    *,
    num_heads: int,
    max_m: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device=None,
) -> bool:
    """Return whether metadata exactly matches the narrow GLM H=64 spec."""
    return bool(
        is_supported(device)
        and _bf16.can_implement_glm_h64_bf16(
            num_heads=num_heads,
            max_m=max_m,
            nope_dim=nope_dim,
            latent_dim=latent_dim,
            output_dtype=output_dtype,
            device=device,
        )
    )


def run(q_nope: torch.Tensor, weight: MlaQueryWeight, q_pe: torch.Tensor,
        out: torch.Tensor, *, plan: Plan,
        q_scale: Optional[torch.Tensor] = None,
        stream: Optional[object] = None) -> torch.Tensor:
    """Assemble the admitted exact-M MLA query into caller-owned ``out``."""
    if isinstance(weight, torch.Tensor):
        from . import _bf16
        return _bf16.run(q_nope, weight, q_pe, out, plan=plan, q_scale=q_scale, stream=stream)
    from .._shared import mxfp8_bmm
    values, scales = mxfp8_bmm._rhs_tensors(weight)
    stream_int = None if stream is None else int(mxfp8_bmm._torch_stream(stream, q_nope.device).cuda_stream)
    torch.ops.b12x.mla_query_projection_mxfp8(
        q_nope, values, scales, q_pe, q_scale, out, 1, plan.handle, stream_int,
    )
    return out


def can_implement(*, num_heads: int, max_m: int, nope_dim: int, latent_dim: int,
                  output_dtype: torch.dtype, weight_format: Literal["bf16", "mxfp8"] = "mxfp8",
                  device=None) -> bool:
    if not is_supported(device):
        return False
    if weight_format == "bf16":
        from . import _bf16
        return _bf16.can_implement(num_heads=num_heads, max_m=max_m, nope_dim=nope_dim,
                                   latent_dim=latent_dim, output_dtype=output_dtype, device=device)
    if weight_format == "mxfp8":
        from .._shared import mxfp8_bmm
        return mxfp8_bmm.can_implement_mla_query_projection(
            batch=num_heads, max_m=max_m, n=latent_dim, k=nope_dim,
            output_dtype=output_dtype, b_major="n", sf_axis="n")
    return False


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    from .._shared import mxfp8_bmm
    mxfp8_bmm.clear_mla_query_projection_caches()


__all__ = ["Mxfp8Weight", "MlaQueryWeight", "plan", "run", "can_implement", "is_supported"]
