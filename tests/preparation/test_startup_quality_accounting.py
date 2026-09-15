"""Qualification accounting must distinguish fixed work from complete races."""
from __future__ import annotations

import math

import pytest

from benchmarks.startup_quality import aggregate_quality


def _row(*, effective=1, measured=0, checked=1, ratio=None):
    return {
        "effective_count": effective, "measured_count": measured,
        "cosine_checked_count": checked, "cosine_passed_count": checked,
        "latency_ratio": ratio, "requires_race": effective > 1,
    }


def _report(**rows):
    return {"requests": rows, "failures": []}


def test_fixed_paths_qualify_without_fictitious_measurements():
    result = aggregate_quality([_report(fixed=_row())], cached=False)
    assert result["passed"]
    assert result["measured_count"] == 0
    assert result["cosine_expected_count"] == 1


def test_complete_races_are_weighted_per_request_not_per_rank():
    ranks = [
        _report(a=_row(effective=2, measured=2, checked=2, ratio=1.001)),
        _report(b=_row(effective=3, measured=3, checked=3, ratio=1.004),
                c=_row(effective=2, measured=2, checked=2, ratio=1.006),
                fixed=_row()),
    ]
    result = aggregate_quality(ranks, cached=False)
    expected = math.exp(sum(math.log(value) for value in (1.001, 1.004, 1.006)) / 3) - 1
    assert result["passed"]
    assert result["geometric_mean_regret"] == pytest.approx(expected)
    assert result["measured_count"] == 7
    assert result["cosine_expected_count"] == 8


def test_partial_race_cannot_pass_with_a_reported_latency_ratio():
    result = aggregate_quality([
        _report(incomplete=_row(effective=3, measured=2, checked=3, ratio=1.0))
    ], cached=False)
    assert not result["passed"]


def test_cached_checks_are_numerical_only():
    report = _report(selected=_row(effective=3, measured=0, checked=1))
    assert aggregate_quality([report], cached=True)["passed"]
    report["requests"]["selected"]["measured_count"] = 1
    assert not aggregate_quality([report], cached=True)["passed"]


def test_empty_qualification_never_passes():
    assert not aggregate_quality([_report()], cached=False)["passed"]
    assert not aggregate_quality([_report()], cached=True)["passed"]
