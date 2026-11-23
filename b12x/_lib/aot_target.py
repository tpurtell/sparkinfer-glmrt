"""Single canonical offline target description for CPU-only AOT compilation.

Measured per-arch device attributes come from ``cuDeviceGetAttribute`` on
SM120 (RTX PRO 6000) and SM121 (GB10). Shared by ``compile_pool``,
``cpu_trace_check``, and every CuteAFD AOT exporter so no compilation path
needs a GPU or maintains a second copy of target attributes.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class AotTarget:
    """Measured target description for offline CuTe DSL and Triton compilation."""

    arch: str
    dsl_arch: str
    major: int
    minor: int
    name: str
    multi_processor_count: int
    shared_memory_per_block: int
    shared_memory_per_block_optin: int
    shared_memory_per_multiprocessor: int
    reserved_shared_memory_per_block: int
    regs_per_multiprocessor: int
    regs_per_block: int
    max_threads_per_multi_processor: int
    max_blocks_per_multiprocessor: int
    warp_size: int
    L2_cache_size: int
    total_memory: int
    is_integrated: bool
    device_ordinal: int = 0
    uuid: str = ""

    @property
    def compute_capability(self) -> tuple[int, int]:
        return (self.major, self.minor)


# Measured from cuDeviceGetAttribute on raptor (RTX PRO 6000, SM120) and
# kakapo (GB10, SM121). SM counts equal EXL3_AOT_MAX_SMS.
TARGETS: dict[str, AotTarget] = {
    "sm120": AotTarget(
        arch="sm120",
        dsl_arch="sm_120a",
        major=12,
        minor=0,
        name="NVIDIA RTX PRO 6000 Blackwell (offline AOT)",
        multi_processor_count=188,
        shared_memory_per_block=49152,
        shared_memory_per_block_optin=101376,
        shared_memory_per_multiprocessor=102400,
        reserved_shared_memory_per_block=1024,
        regs_per_multiprocessor=65536,
        regs_per_block=65536,
        max_threads_per_multi_processor=1536,
        max_blocks_per_multiprocessor=24,
        warp_size=32,
        L2_cache_size=128 << 20,
        total_memory=96 << 30,
        is_integrated=False,
        uuid="offline-sm120",
    ),
    "sm121": AotTarget(
        arch="sm121",
        dsl_arch="sm_121a",
        major=12,
        minor=1,
        name="NVIDIA GB10 (offline AOT)",
        multi_processor_count=48,
        shared_memory_per_block=49152,
        shared_memory_per_block_optin=101376,
        shared_memory_per_multiprocessor=102400,
        reserved_shared_memory_per_block=1024,
        regs_per_multiprocessor=65536,
        regs_per_block=65536,
        max_threads_per_multi_processor=1536,
        max_blocks_per_multiprocessor=24,
        warp_size=32,
        L2_cache_size=24 << 20,
        total_memory=121 << 30,
        is_integrated=True,
        uuid="offline-sm121",
    ),
}

ARCH_OF_COMPUTE: dict[tuple[int, int], str] = {
    (spec.major, spec.minor): arch for arch, spec in TARGETS.items()
}

_ACTIVE_TARGET: AotTarget | None = None
_RUNTIME_LIBRARIES: tuple[ctypes.CDLL, ...] = ()


def active_target() -> AotTarget | None:
    """Return the active offline AOT target, or None on a live CUDA host."""
    return _ACTIVE_TARGET


def is_offline() -> bool:
    """Return True when this process has configured an offline AOT target."""
    return _ACTIVE_TARGET is not None


def resolve_arch(target: str | tuple[int, int] | int) -> str:
    """Normalize an architecture specifier into a key in ``TARGETS``."""
    if isinstance(target, tuple):
        if len(target) == 2:
            cap = (int(target[0]), int(target[1]))
            if cap in ARCH_OF_COMPUTE:
                return ARCH_OF_COMPUTE[cap]
        raise ValueError(f"unsupported compute capability: {target!r}")
    if isinstance(target, int):
        cap = (target // 10, target % 10)
        if cap in ARCH_OF_COMPUTE:
            return ARCH_OF_COMPUTE[cap]
        raise ValueError(f"unsupported capability integer: {target!r}")
    text = str(target).strip().lower()
    if text in TARGETS:
        return text
    cleaned = text.removeprefix("sm_").removeprefix("sm").rstrip("a")
    if cleaned.isdigit():
        return resolve_arch(int(cleaned))
    if "." in cleaned:
        major_s, _, minor_s = cleaned.partition(".")
        if major_s.isdigit() and minor_s.isdigit():
            return resolve_arch((int(major_s), int(minor_s)))
    raise ValueError(f"unsupported offline target arch: {target!r}")


def target_for(
    target: str | tuple[int, int] | int,
    *,
    device_ordinal: int = 0,
    device_uuid: str | None = None,
    product_name: str | None = None,
    sm_count: int | None = None,
    max_shared_memory_per_block_optin: int | None = None,
    max_shared_memory_per_multiprocessor: int | None = None,
    l2_cache_size: int | None = None,
) -> AotTarget:
    """Build an ``AotTarget`` from the canonical table with optional overrides."""
    if isinstance(target, tuple) and len(target) == 2:
        cap = (int(target[0]), int(target[1]))
        base = TARGETS.get(ARCH_OF_COMPUTE.get(cap, ""))
        if base is None:
            major, minor = cap
            arch_suffix = "a" if major >= 9 else ""
            base = replace(
                TARGETS["sm120"],
                arch=f"sm{major}{minor}",
                dsl_arch=f"sm_{major}{minor}{arch_suffix}",
                major=major,
                minor=minor,
                name=product_name or f"SM{major}{minor} (offline AOT)",
                is_integrated=False,
                uuid=device_uuid or f"offline-sm{major}{minor}",
            )
    else:
        base = TARGETS[resolve_arch(target)]

    updates: dict[str, Any] = {"device_ordinal": int(device_ordinal)}
    if device_uuid is not None:
        updates["uuid"] = str(device_uuid).strip()
    if product_name is not None:
        updates["name"] = str(product_name).strip()
    if sm_count is not None:
        updates["multi_processor_count"] = int(sm_count)
    if max_shared_memory_per_block_optin is not None:
        updates["shared_memory_per_block_optin"] = int(max_shared_memory_per_block_optin)
    if max_shared_memory_per_multiprocessor is not None:
        updates["shared_memory_per_multiprocessor"] = int(max_shared_memory_per_multiprocessor)
    if l2_cache_size is not None:
        updates["L2_cache_size"] = int(l2_cache_size)
    return replace(base, **updates)


def configure_offline_cutlass_device_attributes(
    cuda_helpers: Any,
    target: AotTarget | int,
) -> None:
    """Patch CuTe's driver attribute helper to answer from ``target`` and fail closed."""
    attr_enum = cuda_helpers.cuda.CUdevice_attribute
    if isinstance(target, int):
        attr_values = {
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": int(target),
        }
        label = "compiler target metadata"
    else:
        attr_values = {
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": target.shared_memory_per_multiprocessor,
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK": target.shared_memory_per_block,
            "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": target.shared_memory_per_block_optin,
            "CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK": target.reserved_shared_memory_per_block,
            "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT": target.multi_processor_count,
            "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE": target.L2_cache_size,
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR": target.regs_per_multiprocessor,
            "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK": target.regs_per_block,
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR": target.max_threads_per_multi_processor,
            "CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR": target.max_blocks_per_multiprocessor,
            "CU_DEVICE_ATTRIBUTE_WARP_SIZE": target.warp_size,
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR": target.major,
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR": target.minor,
        }
        label = f"offline target {target.arch}"

    known: dict[Any, int] = {}
    for attr_name, value in attr_values.items():
        enum_value = getattr(attr_enum, attr_name, None)
        if enum_value is not None:
            known[enum_value] = value

    def get_device_attribute(attribute, device_id: int = 0):
        del device_id
        if attribute in known:
            return known[attribute]
        raise RuntimeError(
            f"{label} does not provide CUDA device attribute {attribute!r}"
        )

    cuda_helpers.get_device_attribute = get_device_attribute


def describe_target(
    target: str | tuple[int, int] | int,
    *,
    device_ordinal: int = 0,
    device_uuid: str | None = None,
    product_name: str | None = None,
    sm_count: int | None = None,
    max_shared_memory_per_block_optin: int | None = None,
    max_shared_memory_per_multiprocessor: int | None = None,
    l2_cache_size: int | None = None,
    preserve_existing: bool = True,
) -> AotTarget:
    """Configure Torch, CuTe DSL, b12x compiler, and Triton for CPU-only AOT export.

    Nothing here allocates device memory or initializes CUDA. Execution-side
    Torch CUDA entrypoints fail closed so any accidental device use raises.
    When an offline target for the same compute capability is already active
    and ``preserve_existing`` is True with no explicit overrides, the existing
    target (for example a synthetic SM/L2 comparison profile) is retained.
    """
    global _ACTIVE_TARGET, _RUNTIME_LIBRARIES

    has_overrides = any(
        value is not None
        for value in (
            device_uuid,
            product_name,
            sm_count,
            max_shared_memory_per_block_optin,
            max_shared_memory_per_multiprocessor,
            l2_cache_size,
        )
    )
    resolved = target_for(
        target,
        device_ordinal=device_ordinal,
        device_uuid=device_uuid,
        product_name=product_name,
        sm_count=sm_count,
        max_shared_memory_per_block_optin=max_shared_memory_per_block_optin,
        max_shared_memory_per_multiprocessor=max_shared_memory_per_multiprocessor,
        l2_cache_size=l2_cache_size,
    )

    if _ACTIVE_TARGET is not None:
        if (
            _ACTIVE_TARGET.compute_capability != resolved.compute_capability
            or _ACTIVE_TARGET.device_ordinal != resolved.device_ordinal
        ):
            raise RuntimeError(
                f"offline target already configured for {_ACTIVE_TARGET.arch} "
                f"(ordinal {_ACTIVE_TARGET.device_ordinal}), cannot switch to "
                f"{resolved.arch} (ordinal {resolved.device_ordinal})"
            )
        if preserve_existing and not has_overrides:
            return _ACTIVE_TARGET
        if _ACTIVE_TARGET == resolved:
            return _ACTIVE_TARGET
        raise RuntimeError(
            f"offline target {_ACTIVE_TARGET.arch} already configured with different attributes"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from . import compiler

    if torch.cuda.is_initialized():
        raise RuntimeError("offline AOT target requires an uninitialized CUDA runtime")

    compiler._configure_offline_compile_target(resolved.device_ordinal, resolved.uuid)

    def target_ordinal(device=None):
        if device is None:
            return resolved.device_ordinal
        if type(device) is int:
            ordinal = device
        else:
            dev = torch.device(device)
            if dev.type != "cuda":
                raise ValueError(f"compiler target must be CUDA, got {dev}")
            ordinal = resolved.device_ordinal if dev.index is None else dev.index
        if ordinal != resolved.device_ordinal:
            raise ValueError(
                f"compiler target ordinal changed from {resolved.device_ordinal} to {ordinal}"
            )
        return ordinal

    @contextmanager
    def device_scope(device):
        target_ordinal(device)
        yield

    class OfflineStream:
        cuda_stream = 0
        device = torch.device("cuda", resolved.device_ordinal)

        def synchronize(self):
            raise RuntimeError("offline AOT workers cannot synchronize CUDA")

    properties = type(
        "OfflineCudaDeviceProperties",
        (),
        {
            "major": resolved.major,
            "minor": resolved.minor,
            "multi_processor_count": resolved.multi_processor_count,
            "shared_memory_per_block": resolved.shared_memory_per_block,
            "shared_memory_per_block_optin": resolved.shared_memory_per_block_optin,
            "shared_memory_per_multiprocessor": resolved.shared_memory_per_multiprocessor,
            "name": resolved.name,
            "uuid": resolved.uuid,
            "total_memory": resolved.total_memory,
            "L2_cache_size": resolved.L2_cache_size,
            "is_integrated": resolved.is_integrated,
            "max_threads_per_multi_processor": resolved.max_threads_per_multi_processor,
            "regs_per_multiprocessor": resolved.regs_per_multiprocessor,
            "warp_size": resolved.warp_size,
            "gcnArchName": "",
            "pci_bus_id": 0,
            "pci_device_id": 0,
            "pci_domain_id": 0,
        },
    )()
    offline_stream = OfflineStream()

    def get_device_properties(device=None):
        target_ordinal(device)
        return properties

    def get_device_capability(device=None):
        target_ordinal(device)
        return resolved.compute_capability

    def current_stream(device=None):
        target_ordinal(device)
        return offline_stream

    def reject_cuda_initialization(*_args, **_kwargs):
        raise RuntimeError("offline AOT compilation cannot initialize CUDA")

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: max(1, resolved.device_ordinal + 1)
    torch.cuda.current_device = lambda: resolved.device_ordinal
    torch.cuda.set_device = target_ordinal
    torch.cuda.device = device_scope
    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.get_device_name = lambda device=None: get_device_properties(device).name
    torch.cuda.current_stream = current_stream
    torch.cuda.default_stream = current_stream
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.synchronize = reject_cuda_initialization
    torch.cuda.init = reject_cuda_initialization
    torch.cuda._lazy_init = reject_cuda_initialization

    # CuTe's PIC object writer resolves the host launch shim while exporting.
    # Loading the shim supplies symbols only; CUDA remains invisible and any
    # execution-side Torch entrypoint above still fails closed.
    from cutlass.runtime import find_runtime_libraries

    _RUNTIME_LIBRARIES = tuple(
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        for path in find_runtime_libraries(enable_tvm_ffi=False)
    )

    from cutlass.cutlass_dsl import CuTeDSL

    CuTeDSL._get_dsl().envar.arch = resolved.dsl_arch

    from cutlass.base_dsl.runtime import cuda as cuda_helpers

    configure_offline_cutlass_device_attributes(cuda_helpers, resolved)

    try:
        from triton.backends.compiler import GPUTarget
        from triton.runtime import driver
    except ImportError:
        pass
    else:
        triton_target = GPUTarget("cuda", resolved.major * 10 + resolved.minor, resolved.warp_size)

        class OfflineTritonDriver:
            def get_current_device(self):
                return resolved.device_ordinal

            def get_device_capability(self, device):
                target_ordinal(device)
                return resolved.compute_capability

            def get_current_target(self):
                return triton_target

            def get_current_stream(self, device):
                target_ordinal(device)
                return 0

        driver.set_active(OfflineTritonDriver())

    if torch.cuda.is_initialized():
        raise RuntimeError("offline AOT target setup initialized CUDA")

    _ACTIVE_TARGET = resolved
    return resolved


__all__ = [
    "ARCH_OF_COMPUTE",
    "AotTarget",
    "TARGETS",
    "active_target",
    "configure_offline_cutlass_device_attributes",
    "describe_target",
    "is_offline",
    "resolve_arch",
    "target_for",
]
