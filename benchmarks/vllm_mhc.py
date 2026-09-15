"""Production vLLM adapters for the residual mHC benchmark.

This module deliberately owns only the short-lived vLLM runtime needed to
exercise the production adapters.  It does not load a model or substitute an
implementation for any vLLM operation.
"""

from __future__ import annotations

import importlib
import sys
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator


_V4_PROFILE = "deepseek-v4-flash"
_V41_PROFILE = "deepseek-v4.1-flash"
_SUPPORTED_PROFILES = frozenset({_V4_PROFILE, _V41_PROFILE})
_REQUIRED_MODEL_FIELDS = (
    "hidden_size",
    "hc_mult",
    "rms_norm_eps",
    "hc_eps",
    "hc_sinkhorn_iters",
)


def _checked_checkout(vllm_path: str | Path) -> Path:
    checkout = Path(vllm_path).expanduser().resolve()
    package = checkout / "vllm"
    if not package.is_dir():
        raise ValueError(f"vLLM checkout has no vllm package: {checkout}")
    return checkout


def _import_vllm(checkout: Path) -> Any:
    """Import vLLM only from ``checkout``; never fall through to site packages."""
    expected_package = checkout / "vllm"
    existing = sys.modules.get("vllm")
    if existing is None:
        sys.path.insert(0, str(checkout))
        existing = importlib.import_module("vllm")

    origin = getattr(existing, "__file__", None)
    if origin is None:
        raise RuntimeError("Imported vLLM has no __file__; cannot verify its checkout.")
    try:
        Path(origin).resolve().relative_to(expected_package)
    except ValueError as exc:
        raise RuntimeError(
            "The imported vLLM package does not come from the requested checkout: "
            f"requested {checkout}, imported {origin}."
        ) from exc
    return existing


def _validated_model_config(model_config: dict[str, Any]) -> SimpleNamespace:
    if not isinstance(model_config, dict):
        raise TypeError("model_config must be a configuration dictionary.")
    # V4.1 checkpoints retain the text model settings under text_config. The
    # V4 profile supplies those fields at the root, so accept either source.
    values = model_config.get("text_config", model_config)
    if not isinstance(values, dict):
        raise TypeError("model_config.text_config must be a configuration dictionary.")

    missing = [field for field in _REQUIRED_MODEL_FIELDS if field not in values]
    if missing:
        raise ValueError(f"model_config is missing required mHC fields: {', '.join(missing)}")

    try:
        normalized = {
            "hidden_size": int(values["hidden_size"]),
            "hc_mult": int(values["hc_mult"]),
            "rms_norm_eps": float(values["rms_norm_eps"]),
            "hc_eps": float(values["hc_eps"]),
            "hc_sinkhorn_iters": int(values["hc_sinkhorn_iters"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("model_config contains non-numeric mHC geometry.") from exc

    if (
        normalized["hidden_size"] <= 0
        or normalized["hc_mult"] <= 0
        or normalized["rms_norm_eps"] <= 0
        or normalized["hc_eps"] <= 0
        or normalized["hc_sinkhorn_iters"] <= 0
    ):
        raise ValueError("model_config mHC geometry and epsilons must be positive.")
    return SimpleNamespace(**normalized)


@dataclass(slots=True)
class VllmMHCRunner:
    """Live production mHC adapter plus its deliberately bounded runtime."""

    profile_name: str
    adapter: Any
    workspace_manager: Any
    capture_resources: list[Any]
    vllm_path: Path
    vllm_file: Path
    adapter_file: Path
    workspace_file: Path
    norm_eps: float
    capture_sizes: tuple[int, ...]
    _lock_workspace: Any

    @property
    def integration_files(self) -> tuple[Path, ...]:
        """Source provenance for the call site, adapter, and workspace in use."""
        call_site = self.adapter_file.parent / (
            "model.py" if self.profile_name == _V4_PROFILE else "nvidia/model.py"
        )
        return self.vllm_file, call_site, self.adapter_file, self.workspace_file

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "checkout": str(self.vllm_path),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.vllm_path, text=True,
            ).strip(),
            "adapter": f"{type(self.adapter).__module__}.{type(self.adapter).__name__}",
            "capture_sizes": self.capture_sizes,
            "scope": "Production attention-post/FFN-pre adapter, one rank; no model forward or collective",
        }

    def lock_workspace(self) -> None:
        """Forbid workspace growth after the caller's warmup sequence."""
        self._lock_workspace()

    def run(
        self,
        x: Any,
        residual: Any,
        prev_post: Any,
        prev_comb: Any,
        fn: Any,
        scale: Any,
        bias: Any,
        norm_weight: Any,
        pre_mix: Any = None,
        fn_bf16: Any = None,
    ) -> tuple[Any, Any, Any, Any, Any | None]:
        """Run the selected production boundary without altering its arguments."""
        if self.profile_name == _V4_PROFILE:
            out, post, comb, y = self.adapter.run_post_pre(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                norm_weight=norm_weight,
                norm_eps=self.norm_eps,
                hc_fn_bf16=fn_bf16,
            )
            return out, post, comb, y, None

        out, post, comb, y, pre_out = self.adapter.post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            norm_weight,
            pre_mix,
        )
        return out, post, comb, y, pre_out


@contextmanager
def vllm_mhc_runner(
    vllm_path: str | Path,
    *,
    profile_name: str,
    model_config: dict[str, Any],
    capture_sizes: tuple[int, ...],
) -> Iterator[VllmMHCRunner]:
    """Provide one real vLLM mHC adapter with isolated workspace ownership.

    The caller must warm every intended row-count bucket before calling
    :meth:`VllmMHCRunner.lock_workspace`.  CUDA graph capture, if used, must
    remain inside this context so the adapter's retained scratch owners remain
    alive with the graph.
    """
    if profile_name not in _SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported mHC profile {profile_name!r}; expected one of "
            f"{sorted(_SUPPORTED_PROFILES)}."
        )
    if not capture_sizes or any(not isinstance(size, int) or size <= 0 for size in capture_sizes):
        raise ValueError("capture_sizes must be a non-empty tuple of positive integers.")
    if tuple(sorted(set(capture_sizes))) != capture_sizes:
        raise ValueError("capture_sizes must be sorted and unique.")

    checkout = _checked_checkout(vllm_path)
    hf_config = _validated_model_config(model_config)
    vllm = _import_vllm(checkout)

    import torch
    from vllm.config import CompilationConfig, SchedulerConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.worker import workspace as workspace_module
    from vllm.v1.worker.workspace import (
        collect_cuda_graph_capture_resources,
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
        reset_workspace_manager,
    )

    # Keep the scheduler capacity independent of the tiny benchmark batches.
    # The adapter caps its selected decode bucket at 64 rows; exact small
    # graph buckets remain small. Prefill has a separate 4096-token capacity.
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max(4096, max(capture_sizes), 64),
        max_num_seqs=4,
        max_model_len=4096,
        is_encoder_decoder=False,
    )
    compilation_config = CompilationConfig(
        cudagraph_capture_sizes=list(capture_sizes),
        max_cudagraph_capture_size=max(capture_sizes),
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
    )
    # mHC is replicated across TP ranks and invokes no collective. This
    # rank-local operator benchmark needs only one visible device.
    # A model_config would cause unrelated model-loading validation. The mHC
    # adapter receives the validated HF metadata directly, while the real
    # VllmConfig still declares the exact graph buckets it consults.
    vllm_config.compilation_config.cudagraph_capture_sizes = list(capture_sizes)
    vllm_config.compilation_config.max_cudagraph_capture_size = max(capture_sizes)

    old_workspace = workspace_module._manager
    device = torch.device("cuda", torch.cuda.current_device())
    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        with set_current_vllm_config(vllm_config), collect_cuda_graph_capture_resources() as resources:
            if profile_name == _V4_PROFILE:
                adapter_module = importlib.import_module("vllm.models.deepseek_v4.nvidia.b12x")
                adapter = adapter_module.B12xMHCResidual(
                    hidden_size=hf_config.hidden_size,
                    hc_mult=hf_config.hc_mult,
                    rms_eps=hf_config.rms_norm_eps,
                    hc_eps=hf_config.hc_eps,
                    sinkhorn_iters=hf_config.hc_sinkhorn_iters,
                )
            else:
                adapter_module = importlib.import_module("vllm.models.deepseek_v4_1.b12x_layers")
                adapter = adapter_module.B12xMHC(hf_config)

            runner = VllmMHCRunner(
                profile_name=profile_name,
                adapter=adapter,
                workspace_manager=current_workspace_manager(),
                capture_resources=resources,
                vllm_path=checkout,
                vllm_file=Path(vllm.__file__).resolve(),
                adapter_file=Path(adapter_module.__file__).resolve(),
                workspace_file=Path(workspace_module.__file__).resolve(),
                norm_eps=hf_config.rms_norm_eps,
                capture_sizes=capture_sizes,
                _lock_workspace=lock_workspace,
            )
            yield runner
    finally:
        reset_workspace_manager()
        workspace_module._manager = old_workspace
