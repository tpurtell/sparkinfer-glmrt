from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmarks._dense_trellis_curve import capture_stages, curve_flags, select_shapes
from benchmarks.kimi_k6_dense import parse_args


CLI = ["--row-curve", "--capacity", "4096", "--family", "test", "--routes", "routes.json",
       "--output-dir", "result"]


def test_curve_keeps_capacity_separate_from_live_rows():
    args, _ = parse_args(CLI + ["--rows", "2752,2048,2752", "--check-prefix-rows"])
    assert args.capacity == 4096 and args.live_rows == [2048, 2752]
    assert parse_args(CLI)[0].live_rows == [2048, 2752, 4096]


@pytest.mark.parametrize("extra", [
    ["--rows", "4097"], ["--rows", "0"], ["--rows", "bad"], ["--bits", "4"],
    ["--check-prefix-rows", "--rows", "4096"], ["--compare-routes-exact"],
    ["--baseline-native", "base.py"],
])
def test_curve_rejects_invalid_contract(extra):
    with pytest.raises(SystemExit):
        parse_args(CLI + extra)


def test_curve_selects_geometry_and_width_from_exporter():
    policy = dict(shape="projection", input_features=512, output_features=128, sharded_side=2,
                  output_program="trellis_output_n128_f32", codebook="mcg")
    policies = {"test": [policy, dict(policy, shape="other", bits=8)]}
    refs = dict(bf16={"program": "exact_bf16"}, fp8={"reason": "fused"})
    routes = {"test": {"projection": {"reduction": "f32_add", "references": refs}}}
    shape, = select_shapes(policies, routes, ["test"], ["projection"], 6, "mcg")
    assert (shape.k, shape.n, shape.egress, shape.reduction) == (512, 128, "f32", "f32_add")
    with pytest.raises(ValueError, match="not exported"):
        select_shapes(policies, routes, ["test"], ["projection"], 8, "mcg")
    with pytest.raises(ValueError, match="unconfigured shapes"):
        select_shapes(policies, routes, ["test"], ["other"], 6, "mcg")
    routes["test"]["projection"]["groups"] = 4
    with pytest.raises(ValueError, match="does not export 4 groups"):
        select_shapes(policies, routes, ["test"], [], 6, "mcg")


def test_curve_flags_collapse_and_adjacent_jump():
    flags = curve_flags([dict(rows=rows, us_per_row={"trellis": value})
                         for rows, value in [(2048, 1.), (2752, 1.09), (4096, 1.3)]])
    point, = flags["trellis"]
    assert point["rows"] == 4096 and point["above_neighbor"] == pytest.approx(1.3 / 1.09 - 1)


def test_capture_uses_active_stream_for_every_native_stage():
    class Cuda:
        stream = 11

        def current_stream(self):
            return SimpleNamespace(cuda_stream=self.stream)

        @contextmanager
        def graph(self, graph):
            self.stream = 29
            try:
                yield
            finally:
                self.stream = 11

        CUDAGraph = object

        @staticmethod
        def Event(**kwargs):
            return SimpleNamespace(record=lambda: None)

    log = []
    calls = {name: lambda rows, stream, stage=name: log.append((stage, rows, stream))
             for name in ("ingress", "trellis", "egress", "reference")}
    capture_stages(Cuda(), calls, 2752)
    assert log == [(name, 2752, 29) for name in calls]


def test_final_narrow_requires_fp32_h128_egress():
    policy = dict(shape="projection", input_features=512, output_features=128, sharded_side=2,
                  output_program="trellis_output_n128_f32", codebook="mcg")
    refs = dict(bf16={"reason": "fused"}, fp8={"reason": "fused"})
    route = dict(reduction="f32_add_then_narrow", references=refs)
    args = ({"test": [policy]}, {"test": {"projection": route}}, ["test"], [], 6, "mcg")
    shape, = select_shapes(*args)
    assert shape.reduction == "f32_add_then_narrow" and shape.egress == "f32"
    route["egress"] = "bf16"
    with pytest.raises(ValueError, match="invalid egress dtype"):
        select_shapes(*args)
