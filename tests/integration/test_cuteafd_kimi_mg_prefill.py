"""Kimi oracle over the exact decoded record: FP8 QK fails it, BF16 attention passes it."""
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
    assert single["mg_lse_mismatches"] > 0


@pytest.mark.parametrize("heads", [48, 96])
def test_kimi_mg_bf16_passes_unchanged_oracle(heads):
    require_b12x()
    from benchmarks.bench_kimi_mg_prefill import oracle

    report = {}
    oracle(SimpleNamespace(heads=[heads], bf16_compute=True), report)
    assert [r["route"] for r in report["oracle"]] == ["sparse", "dense"]


def test_kimi_mg_rejects_raw_bf16_cache():
    require_b12x()
    from b12x.integration.cuteafd.kimi_mla import KimiMlaGeometry
    from b12x.integration.cuteafd.kimi_mla_prefill_mg import compile_kimi_mg_prefill_aot

    with pytest.raises(ValueError, match="not raw BF16"):
        compile_kimi_mg_prefill_aot(KimiMlaGeometry(kv_cache="bf16"))
