from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from b12x.policy import DeviceIdentity
from b12x.policy.generation import (
    CheckpointStore,
    GenerationContext,
    GenerationSettings,
)
from b12x.policy.generation.moe_corpus import (
    MoeModelGeometry,
    MoeRecipe,
    expand_physical_geometries,
    expand_sweep_cases,
)
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers import moe_gpu_worker
from b12x.policy.generation.providers.moe import (
    _config_covers_query,
    _synthesize_token_capacity_coverage,
    MoeCandidate,
    MoeDecodeGenerator,
    MoeMeasurement,
)
from b12x.policy.generation.reducer import (
    DecisionRecord,
    build_axis_tree,
    decision_node_to_dict,
    synthesize_integer_axis_coverage,
)
from b12x.policy.generation.providers.moe_gpu_worker import (
    MoeGpuBenchmarkFactory,
    _activation_mode,
    _benchmark_input_scale,
    _candidate_reference_output,
    _cosine_similarity,
    _finite_float_or_none,
    _is_fatal_accelerator_error,
    _measurement_seed,
    _MoeGeometrySession,
    _MoeProcessSession,
    _MoeRemoteWorkerError,
    _packed_weights,
    _reset_cuda_graphs,
    _trellis_weights,
)
from b12x.policy.serialization import profile_from_dict

_DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic GPU",
)


def test_modelopt_w4a8_profile_worker_uses_a8_activation() -> None:
    from b12x.moe import fused_moe

    assert _activation_mode("w4a8_nvfp4") is fused_moe.ActivationMode.A8


def test_relu2_w4a16_profile_fixture_conditions_synthetic_inputs() -> None:
    geometry = next(
        geometry
        for geometry in expand_physical_geometries()
        if geometry.recipe.recipe_id == "modelopt-w4a16"
        and geometry.activation == "relu2"
    )

    assert _benchmark_input_scale(geometry) == 2.0**-20


class _Session(AbstractContextManager["_Session"]):
    def __init__(
        self,
        calls: list[tuple[str, tuple[str, ...]]],
        *,
        tunable: bool,
    ) -> None:
        self._calls = calls
        self._candidates = tuple(
            MoeCandidate.create({"backend": backend})
            for backend in (("micro", "dynamic") if tunable else ("dynamic",))
        )

    @property
    def candidates(self) -> tuple[MoeCandidate, ...]:
        return self._candidates

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def eligible_candidates(self, case, candidates):
        return tuple(
            candidate
            for candidate in candidates
            if not (candidate.config["backend"] == "micro" and case.num_tokens > 8)
        )

    def measure(self, case, candidates, *, correctness=False):
        del correctness
        self._calls.append(
            (case.case_id, tuple(candidate.candidate_id for candidate in candidates))
        )
        measurements = []
        for candidate in candidates:
            backend = candidate.config["backend"]
            if case.num_tokens == 1:
                latency = 10.0 if backend == "micro" else 20.0
            else:
                latency = 30.0 if backend == "micro" else 15.0
            if case.route_pattern == "hot":
                latency *= 1.1
            measurements.append(
                MoeMeasurement(
                    candidate=candidate,
                    latency_us=latency,
                    cosine=0.9995,
                    metrics={"graph_cosine": 1.0},
                )
            )
        return tuple(measurements)


@dataclass
class _Factory:
    calls: list[tuple[str, tuple[str, ...]]]

    def __call__(self, geometry, context):
        del context
        return _Session(
            self.calls,
            tunable=geometry.recipe.quant_mode == "nvfp4",
        )


def _generator(
    calls,
    *,
    quant_mode: str = "nvfp4",
    tp_sizes: tuple[int, ...] = (1,),
    token_counts: tuple[int, ...] = (1, 4),
    top_ks: tuple[int, ...] = (2,),
):
    recipe = MoeRecipe(
        recipe_id=f"test-{quant_mode}",
        quant_mode=quant_mode,
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    )
    model = MoeModelGeometry(
        model_id="test-model",
        hidden_size=256,
        intermediate_size=64,
        num_experts=16,
        native_top_k=2,
        activation="silu",
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=tp_sizes,
    )
    geometries = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    cases = expand_sweep_cases(
        geometries=geometries,
        top_ks=top_ks,
        token_counts=token_counts,
        route_patterns=("balanced", "hot"),
    )
    return MoeDecodeGenerator(
        benchmark_factory=_Factory(calls),
        geometries=geometries,
        cases=cases,
    )


def _context(tmp_path):
    return GenerationContext(
        device=_DEVICE,
        device_ordinal=0,
        work_dir=tmp_path,
        source_revision="abc123",
        settings=GenerationSettings(),
    )


def test_moe_measurement_partition_keeps_one_physical_geometry(tmp_path) -> None:
    generator = _generator([], tp_sizes=(1, 2))
    context = _context(tmp_path)

    partitions = generator.measurement_partitions(context)
    restricted = generator.select_measurement_partitions((partitions[0].partition_id,))

    assert len(partitions) == 2
    assert partitions[0].component_id == "moe.decode"
    assert restricted.estimate(context).case_count == partitions[0].case_count
    assert (
        restricted.estimate(context).case_count < generator.estimate(context).case_count
    )


def test_staged_moe_generator_keeps_regional_winners_and_resumes(tmp_path) -> None:
    calls = []
    generator = _generator(calls)
    context = _context(tmp_path)
    checkpoints = CheckpointStore(tmp_path / "checkpoints")

    result = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )
    first_call_count = len(calls)
    resumed = generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )

    assert first_call_count == 9
    assert len(calls) == first_call_count
    assert result.component == resumed.component
    assert result.component["coverage"]["measured_query_points"] == 2
    assert result.component["coverage"]["runtime_query_points"] == 4

    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.48sm",
            "targets": [
                {
                    "vendor": "nvidia",
                    "compute_capability": [12, 1],
                    "sm_count": 48,
                    "product_name": "Synthetic GPU",
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("moe.decode")
    assert component is not None
    base_query = {
        "quant_mode": "nvfp4",
        "source_format": "modelopt_nvfp4",
        "activation": "silu",
        "num_experts": 16,
        "hidden_size": 256,
        "intermediate_size": 64,
        "top_k": 2,
    }
    assert (
        component.lookup({**base_query, "num_tokens": 1}).config["backend"] == "micro"
    )
    assert (
        component.lookup({**base_query, "num_tokens": 4}).config["backend"] == "dynamic"
    )
    assert (
        component.lookup({**base_query, "num_tokens": 2}).config["backend"] == "micro"
    )
    assert (
        component.lookup({**base_query, "num_tokens": 3}).config["backend"] == "dynamic"
    )
    assert component.lookup({**base_query, "num_tokens": 2_048}) is None


def test_prefill_capacities_are_qualified_once_per_geometry(
    tmp_path,
) -> None:
    calls = []
    generator = _generator(
        calls,
        token_counts=(1, 512, 1_024),
        top_ks=(1, 2),
    )

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    cases_by_id = {case.case_id: case for case in generator._cases}
    capacity_calls = [
        cases_by_id[case_id]
        for case_id, _candidate_ids in calls
        if cases_by_id[case_id].num_tokens >= 512
    ]
    assert {(case.num_tokens, case.top_k) for case in capacity_calls} == {
        (512, 2),
        (1_024, 2),
    }
    assert result.component["coverage"]["qualified_capacity_query_points"] == 2

    profile = profile_from_dict(
        {
            "profile_id": "nvidia.synthetic.48sm",
            "targets": [
                {
                    "vendor": "nvidia",
                    "compute_capability": [12, 1],
                    "sm_count": 48,
                    "product_name": "Synthetic GPU",
                }
            ],
            "components": [result.component],
        }
    )
    component = profile.component("moe.decode")
    assert component is not None
    for num_tokens in (512, 1_024):
        for top_k in (1, 2):
            hit = component.lookup(
                {
                    "quant_mode": "nvfp4",
                    "source_format": "modelopt_nvfp4",
                    "activation": "silu",
                    "num_experts": 16,
                    "hidden_size": 256,
                    "intermediate_size": 64,
                    "top_k": top_k,
                    "num_tokens": num_tokens,
                }
            )
            assert hit is not None
            assert hit.config["backend"] == "dynamic"


@pytest.mark.parametrize("top_k", (1, 10, 16))
def test_sparse_token_capacity_coverage_matches_dense_reference(top_k: int) -> None:
    base_query = {
        "quant_mode": "nvfp4",
        "source_format": "modelopt_nvfp4",
        "activation": "silu",
        "num_experts": 16,
        "hidden_size": 256,
        "intermediate_size": 64,
        "top_k": top_k,
    }
    configs = {
        "micro": {
            "backend": "micro",
            "route_planner": "internal",
            "max_active_clusters": None,
        },
        "triton": {
            "backend": "dynamic",
            "route_planner": "triton",
            "max_active_clusters": 24,
        },
        "dynamic": {
            "backend": "dynamic",
            "route_planner": "internal",
            "max_active_clusters": None,
        },
    }
    measured = tuple(
        DecisionRecord.create(
            query={**base_query, "num_tokens": num_tokens},
            config=configs[config],
        )
        for num_tokens, config in (
            (1, "micro"),
            (min(128, 256 // top_k), "triton"),
            (512, "dynamic"),
            (1_024, "dynamic"),
        )
    )
    dense = synthesize_integer_axis_coverage(
        measured,
        field="num_tokens",
        minimum=1,
        maximum=1_024,
        config_is_valid=_config_covers_query,
    )
    sparse = _synthesize_token_capacity_coverage(
        measured,
        minimum=1,
        maximum=1_024,
    )
    assert len(sparse) < len(dense) // 10

    field_order = (*base_query, "num_tokens")
    dense_tree = build_axis_tree(
        dense,
        field_order=field_order,
        range_fields=frozenset({"num_tokens"}),
    )
    sparse_tree = build_axis_tree(
        sparse,
        field_order=field_order,
        range_fields=frozenset({"num_tokens"}),
        nearest_range_bounds={"num_tokens": (1, 1_024)},
    )

    def component(planner):
        profile = profile_from_dict(
            {
                "profile_id": "nvidia.synthetic.48sm",
                "targets": [
                    {
                        "vendor": "nvidia",
                        "compute_capability": [12, 1],
                        "sm_count": 48,
                        "product_name": "Synthetic GPU",
                    }
                ],
                "components": [
                    {
                        "component_id": "moe.decode",
                        "query_schema_version": 2,
                        "config_schema_version": 1,
                        "planner": decision_node_to_dict(planner),
                    }
                ],
            }
        )
        return profile.component("moe.decode")

    dense_component = component(dense_tree)
    sparse_component = component(sparse_tree)
    assert dense_component is not None
    assert sparse_component is not None
    for num_tokens in range(1, 1_025):
        query = {**base_query, "num_tokens": num_tokens}
        assert (
            sparse_component.lookup(query).config
            == dense_component.lookup(query).config
        )


def test_moe_resume_retries_checkpoint_when_every_candidate_errored(
    tmp_path,
) -> None:
    calls = []
    generator = _generator(calls)
    context = _context(tmp_path)
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    case = next(
        case
        for case in generator._cases
        if case.num_tokens == 4 and case.route_pattern == "balanced"
    )
    candidates = _Session([], tunable=True).candidates
    checkpoints.save(
        generator.component_id,
        f"screen-{case.case_id}",
        {
            "schema_version": 1,
            "generation": context.checkpoint_metadata(),
            "case_id": case.case_id,
            "query": case.query(),
            "route_pattern": case.route_pattern,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "measurements": [
                MoeMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    cosine=None,
                    error="transient worker failure",
                ).to_dict()
                for candidate in candidates
            ],
        },
    )

    generator.generate(
        context,
        progress=NullProgressReporter(),
        checkpoints=checkpoints,
    )

    assert calls[0] == (
        case.case_id,
        tuple(candidate.candidate_id for candidate in candidates),
    )
    refreshed = checkpoints.load(
        generator.component_id,
        f"screen-{case.case_id}",
    )
    assert refreshed is not None
    assert all(
        measurement["error"] is None for measurement in refreshed["measurements"]
    )


def test_moe_measurement_inputs_are_route_pattern_controlled() -> None:
    cases = _generator([])._cases
    matched = [case for case in cases if case.top_k == 2 and case.num_tokens == 4]

    assert len({_measurement_seed(case, base_seed=17) for case in matched}) == 1


def test_moe_zero_outputs_have_defined_correctness_cosine() -> None:
    zeros = torch.zeros(8)

    assert _cosine_similarity(zeros, zeros.clone()) == 1.0
    assert _cosine_similarity(zeros, torch.ones(8)) == 0.0


def test_moe_cosine_does_not_overflow_on_large_finite_outputs() -> None:
    large = torch.full((8_192,), 1.0e30)

    assert _cosine_similarity(large, large.clone()) == pytest.approx(1.0)


def test_moe_nonfinite_diagnostics_are_strict_json_values() -> None:
    assert _finite_float_or_none(1.25) == 1.25
    assert _finite_float_or_none(float("inf")) is None
    assert _finite_float_or_none(float("nan")) is None


def test_moe_accelerator_errors_are_fatal() -> None:
    assert _is_fatal_accelerator_error(torch.AcceleratorError("CUDA fault"))
    assert not _is_fatal_accelerator_error(RuntimeError("candidate unsupported"))


def test_moe_query_teardown_explicitly_resets_cuda_graphs() -> None:
    resets = []
    prepared = tuple(
        SimpleNamespace(graph=SimpleNamespace(reset=lambda i=index: resets.append(i)))
        for index in range(3)
    )

    _reset_cuda_graphs(prepared)

    assert resets == [0, 1, 2]


def test_failed_first_candidate_cannot_become_cross_candidate_reference() -> None:
    output = object()

    assert (
        _candidate_reference_output(
            output,
            correct=False,
            has_reference=False,
        )
        is None
    )
    assert (
        _candidate_reference_output(
            output,
            correct=True,
            has_reference=False,
        )
        is output
    )


def test_independent_nvfp4_oracle_matches_kernel_scale_math(monkeypatch) -> None:
    from b12x.moe._shared.kernels import reference

    geometry = _generator([])._geometries[0]
    session = object.__new__(_MoeGeometrySession)
    session._geometry = geometry
    session._experts = SimpleNamespace(
        _impl=SimpleNamespace(
            w1_fp4=object(),
            w1_blockscale=object(),
            w1_alphas=object(),
            w2_fp4=object(),
            w2_blockscale=object(),
            w2_alphas=object(),
            a1_gscale=object(),
            a2_gscale=object(),
        )
    )
    captured = {}

    def fake_reference(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "reference"

    monkeypatch.setattr(reference, "moe_reference_nvfp4", fake_reference)

    result = session._independent_reference(
        x=object(),
        topk_ids=object(),
        topk_weights=object(),
    )

    assert result == "reference"
    assert captured["kwargs"]["quant_scale_math"] == "reciprocal_multiply"


def test_modelopt_profile_weights_pad_and_swizzle_scale_atoms() -> None:
    recipe = MoeRecipe(
        recipe_id="modelopt-nvfp4-test",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        intermediate_alignment=16,
        minimum_intermediate_size=16,
    )
    model = MoeModelGeometry(
        model_id="low-width-test",
        hidden_size=256,
        intermediate_size=32,
        num_experts=1,
        native_top_k=1,
        activation="silu",
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=(1,),
    )
    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )

    prepared = _packed_weights(geometry, device="cpu")._impl

    assert tuple(prepared.w1_blockscale.shape) == (1, 128, 16)
    assert tuple(prepared.w2_blockscale.shape) == (1, 256, 4)


@pytest.mark.parametrize(
    ("variant", "activation", "expected_dtype"),
    (
        ("glm-mcg-projection-tiered", "silu", torch.bfloat16),
        ("k3-sqg-uniform-coupled", "situ", torch.float16),
    ),
)
def test_trellis_profile_worker_uses_serving_dtype(
    monkeypatch,
    variant,
    activation,
    expected_dtype,
) -> None:
    from b12x.moe import fused_moe

    recipe = MoeRecipe(
        recipe_id="trellis-test",
        quant_mode="w4a16",
        source_format="btx",
        intermediate_alignment=128,
        minimum_intermediate_size=128,
        trellis_variant=variant,
    )
    model = MoeModelGeometry(
        model_id="trellis-test",
        hidden_size=128,
        intermediate_size=128,
        num_experts=3,
        native_top_k=1,
        activation=activation,
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=(1,),
    )
    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    captured = {}

    def plan_weights(*, source, activation, geometry):
        captured.update(
            source=source,
            activation=activation,
            geometry=geometry,
        )
        return object()

    monkeypatch.setattr(fused_moe, "plan_weights", plan_weights)
    monkeypatch.setattr(
        fused_moe,
        "prepare_weights",
        lambda *, plan, weights: SimpleNamespace(plan=plan, weights=weights),
    )

    _trellis_weights(geometry, device="cpu")

    assert captured["activation"].io_dtype is expected_dtype


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(torch.cuda.current_device())[0] != 12,
    reason="requires an SM120/SM121 GPU",
)
def test_projection_mixed_bind_initializes_grid_barrier_workspace() -> None:
    from b12x.moe import fused_moe
    from b12x.policy import MOE_DECODE, get_auto_policy

    recipe = MoeRecipe(
        recipe_id="trellis-bind-test",
        quant_mode="w4a16",
        source_format="btx",
        intermediate_alignment=128,
        minimum_intermediate_size=128,
        trellis_variant="glm-mcg-projection-tiered",
    )
    model = MoeModelGeometry(
        model_id="trellis-bind-test",
        hidden_size=128,
        intermediate_size=256,
        num_experts=3,
        native_top_k=1,
        activation="silu",
        recipe_ids=(recipe.recipe_id,),
        source="test",
        tp_sizes=(1,),
    )
    (geometry,) = expand_physical_geometries(
        models=(model,),
        recipes=(recipe,),
    )
    device = torch.device("cuda", torch.cuda.current_device())
    experts = _trellis_weights(geometry, device=device)
    policy = get_auto_policy(device).with_override(
        MOE_DECODE,
        fused_moe.MoeDecodeConfig.from_profile(
            {
                "backend": "dynamic",
                "route_planner": "internal",
                "max_active_clusters": None,
            }
        ),
    )
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=1, top_k=1),
        policy=policy,
    )
    fused_moe.prewarm(plan)
    scratch = {
        spec.name: torch.empty(
            spec.shape,
            dtype=spec.dtype,
            device=spec.device,
        ).fill_(1)
        for spec in plan.scratch_specs()
    }
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=torch.zeros((1, 128), dtype=torch.bfloat16, device=device),
        experts=experts,
        topk_weights=torch.ones((1, 1), dtype=torch.float32, device=device),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32, device=device),
        input_scales_static=True,
    )

    workspace = binding.mixed_trellis_buffers.workspace
    assert int(torch.count_nonzero(workspace)) == 0


def test_single_backend_geometries_are_gpu_qualified_before_profile_coverage(
    tmp_path,
) -> None:
    calls = []
    generator = _generator(calls, quant_mode="w4a16")

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    assert len(calls) == 5
    assert result.evidence["gpu_measurement_cases"] == 5
    assert "precomputed_cases" not in result.evidence
    assert result.component["coverage"]["measured_query_points"] == 2
    assert result.component["coverage"]["runtime_query_points"] == 4


def test_single_backend_does_not_repeat_qualification_at_capacity(tmp_path) -> None:
    calls = []
    generator = _generator(
        calls,
        quant_mode="w4a16",
        token_counts=(1, 512, 1_024),
    )

    result = generator.generate(
        _context(tmp_path),
        progress=NullProgressReporter(),
        checkpoints=CheckpointStore(tmp_path / "checkpoints"),
    )

    cases_by_id = {case.case_id: case for case in generator._cases}
    measured_tokens = {
        cases_by_id[case_id].num_tokens for case_id, _candidate_ids in calls
    }
    assert measured_tokens == {1}
    assert result.component["coverage"]["qualified_capacity_query_points"] == 0


def test_production_moe_factory_isolates_each_geometry_process(tmp_path) -> None:
    generator = _generator([])

    session = MoeGpuBenchmarkFactory()(generator._geometries[0], _context(tmp_path))

    assert isinstance(session, _MoeProcessSession)


def test_moe_process_session_retries_accelerator_failure_once(
    monkeypatch,
    tmp_path,
) -> None:
    generator = _generator([])
    geometry = generator._geometries[0]
    case = next(case for case in generator._cases if case.geometry == geometry)
    session = _MoeProcessSession(geometry, _context(tmp_path))
    candidate = session.candidates[0]
    measurement = MoeMeasurement(
        candidate=candidate,
        latency_us=10.0,
        cosine=1.0,
    )
    requests = []
    starts = []
    discards = []

    monkeypatch.setattr(session, "_start", lambda: starts.append("start"))

    def request(operation, **payload):
        requests.append((operation, payload))
        if len(requests) == 1:
            raise _MoeRemoteWorkerError(
                operation="measure",
                exception_type="AcceleratorError",
                error="CUDA illegal address",
            )
        return {"measurements": [measurement.to_dict()]}

    monkeypatch.setattr(session, "_request", request)
    monkeypatch.setattr(
        session,
        "_discard_worker",
        lambda: discards.append("discard"),
    )

    (result,) = session.measure(case, (candidate,), correctness=True)

    assert len(starts) == 2
    assert len(requests) == 2
    assert discards == ["discard"]
    assert result.metrics["worker_retries"] == 1


def test_moe_geometry_worker_binds_device_before_starting_session(
    monkeypatch,
) -> None:
    events = []

    class Connection:
        def __init__(self) -> None:
            self.sent = []

        def send(self, message) -> None:
            self.sent.append(message)

        def recv(self):
            return {"operation": "close"}

        def close(self) -> None:
            events.append("connection.close")

    class Session(AbstractContextManager["Session"]):
        candidates = ()

        def __init__(self, geometry, context) -> None:
            del geometry, context
            events.append("session.init")

        def __enter__(self) -> "Session":
            events.append("session.enter")
            return self

        def __exit__(self, *_exc: object) -> None:
            events.append("session.exit")

    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda ordinal: events.append(("set_device", ordinal)),
    )
    monkeypatch.setattr(moe_gpu_worker, "_MoeGeometrySession", Session)
    connection = Connection()

    moe_gpu_worker._moe_geometry_worker(
        connection,
        object(),
        SimpleNamespace(device_ordinal=7),
    )

    assert events[:3] == [("set_device", 7), "session.init", "session.enter"]
    assert connection.sent == [
        {"ok": True, "candidates": []},
        {"ok": True, "closed": True},
    ]


def test_production_moe_resume_does_not_start_a_gpu_worker(tmp_path) -> None:
    generator = _generator([])
    geometry = generator._geometries[0]
    case = next(case for case in generator._cases if case.geometry == geometry)
    session = MoeGpuBenchmarkFactory()(geometry, _context(tmp_path))

    with session:
        candidates = session.candidates
        assert candidates
        assert session.eligible_candidates(case, candidates)
        assert session._process is None
