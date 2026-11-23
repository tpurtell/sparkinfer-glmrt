"""Tests for the single canonical offline AOT target description (b12x._lib.aot_target)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("cutlass")

ROOT = Path(__file__).resolve().parents[1]


def _run_py(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "PYTHONPATH": str(ROOT),
        "TMPDIR": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize(
    ("arch", "cap", "dsl_arch", "sms", "l2"),
    [
        ("sm120", (12, 0), "sm_120a", 188, 128 << 20),
        ("sm121", (12, 1), "sm_121a", 48, 24 << 20),
    ],
)
def test_describe_target_configures_torch_cute_triton_and_helpers_without_cuda(
    tmp_path, arch, cap, dsl_arch, sms, l2
):
    code = f"""
import pytest
import torch
from b12x._lib.aot_target import active_target, describe_target, is_offline
from b12x._lib.utils import get_l2_cache_bytes
from b12x.attention._shared.mla.native_v41_aot import (
    _can_query_cutlass_max_smem_per_mp,
    query_sm_occupancy_limits,
)
from b12x.integration.cuteafd._fp8_gemv_nc import _query_device_caps
from b12x.moe._shared.kernels.w4a16.exlr8_grouped import query_optin_shared_memory
from b12x.moe._shared.kernels.w4a16.kernel import _trellis256_execution_lut
from cutlass.base_dsl.runtime import cuda as dsl_cuda
from cutlass.cutlass_dsl import CuTeDSL
from triton.runtime import driver

target = describe_target({arch!r})
assert is_offline()
assert active_target() == target
assert not torch.cuda.is_initialized()
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == {cap!r}
props = torch.cuda.get_device_properties(0)
assert props.multi_processor_count == {sms}
assert props.shared_memory_per_block_optin == 101376
assert props.shared_memory_per_multiprocessor == 102400
assert props.L2_cache_size == {l2}
assert CuTeDSL._get_dsl().envar.arch == {dsl_arch!r}
assert _can_query_cutlass_max_smem_per_mp()

# Fail closed on execution-side CUDA entrypoints.
for fn in (torch.cuda.init, torch.cuda._lazy_init, torch.cuda.synchronize):
    try:
        fn()
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{{fn}} did not fail closed")

# Unknown CuTe device attribute fails closed.
try:
    dsl_cuda.get_device_attribute(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unknown device attribute did not fail closed")

# Triton offline driver reports matching GPUTarget.
triton_target = driver.active.get_current_target()
assert triton_target.backend == "cuda"
assert triton_target.arch == {cap[0] * 10 + cap[1]}
assert triton_target.warp_size == 32

# Raw CUDA driver query helpers answer from described target without CUDA init.
limits = query_sm_occupancy_limits(0)
assert limits is not None
assert limits.num_sms == {sms}
assert limits.max_smem_per_mp == 102400
assert limits.max_smem_per_block == 49152
assert limits.max_smem_per_block_optin == 101376
assert limits.reserved_smem_per_block == 1024
assert limits.max_blocks_per_mp == 24
assert query_optin_shared_memory() == 101376
assert _query_device_caps() == ({sms}, {l2})
assert get_l2_cache_bytes() == {l2}

# Trellis LUT stays on CPU when offline.
lut = _trellis256_execution_lut(torch.device("cuda", 0), "mcg")
assert lut.device.type == "cpu"

assert not torch.cuda.is_initialized()
print("OK")
"""
    res = _run_py(code, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK" in res.stdout


def test_describe_target_preserves_synthetic_overrides_and_rejects_arch_switch(tmp_path):
    code = """
import torch
from b12x._lib.aot_target import describe_target

first = describe_target("sm120", sm_count=170, l2_cache_size=96 << 20)
assert first.multi_processor_count == 170
assert first.L2_cache_size == 96 << 20

# Re-describing the same capability without overrides retains synthetic values.
second = describe_target((12, 0))
assert second == first
assert torch.cuda.get_device_properties(0).multi_processor_count == 170

# Switching capability in the same process fails closed.
try:
    describe_target("sm121")
except RuntimeError as exc:
    assert "already configured" in str(exc)
else:
    raise AssertionError("capability switch did not fail closed")
print("OK")
"""
    res = _run_py(code, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK" in res.stdout
