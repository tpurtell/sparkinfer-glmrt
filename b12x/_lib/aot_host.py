"""Host CPU defaults for CuTe DSL AOT exports, not runtime JIT codegen."""

import platform


_HOST_TARGETS = {
    "x86_64": "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3",
    # GB10's fast core; this matches the engine's existing Spark exports.
    "aarch64": "llvm -mtriple=aarch64-unknown-linux-gnu -mcpu=cortex-x925",
}


def pin_aot_host_target() -> None:
    """Give unset CuTe AOT HostTargets a portable process-wide default."""
    from cutlass.base_dsl.compiler import HostTarget

    machine = platform.machine()
    if machine not in _HOST_TARGETS:
        raise RuntimeError(f"no CuTe AOT host target for {machine}")
    target = _HOST_TARGETS[machine]
    if getattr(HostTarget, "_b12x_aot_host_target", None) == target:
        return

    # Some HostTargets predate the import of b12x. Fill the default on read
    # at object emission, leaving the option value and explicit targets alone.
    triple, cpu, features = HostTarget._parse_target(target)
    HostTarget.triple = property(lambda self: self._triple or triple)
    HostTarget.cpu = property(lambda self: self._cpu if self._triple else cpu)
    HostTarget.features = property(lambda self: self._features if self._triple else features)
    HostTarget._b12x_aot_host_target = target
