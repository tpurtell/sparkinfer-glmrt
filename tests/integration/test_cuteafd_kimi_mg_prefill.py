"""Unchanged Kimi oracle and fail-closed unsupported MG arithmetic."""
from types import SimpleNamespace

import pytest

from ..conftest import require_b12x


@pytest.mark.parametrize("heads", [48, 96])
def test_kimi_mg_records_oracle_rejection(heads):
    require_b12x()
    from benchmarks.bench_kimi_mg_prefill import diagnose

    report = {}
    diagnose(SimpleNamespace(heads=[heads]), report)
    assert report["correctness_passed"] is False
    single = next(c for c in report["diagnostic"] if c["length"] == 1)
    assert single["baseline_lse_mismatches"] == 0
    assert single["mg_lse_mismatches"] > 0


def test_kimi_mg_rejects_unsupported_bf16_compute_and_cache():
    require_b12x()
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry
    from b12x.integration.cuteafd.kimi_mla_prefill_mg import compile_kimi_mg_prefill_aot

    with pytest.raises(ValueError, match="inline FP32 scales"):
        compile_kimi_mg_prefill_aot(KimiMlaGeometry(kv_cache="fp8"), bf16_compute=True)
    with pytest.raises(ValueError, match="not raw BF16"):
        compile_kimi_mg_prefill_aot(KimiMlaGeometry(kv_cache="bf16"))
