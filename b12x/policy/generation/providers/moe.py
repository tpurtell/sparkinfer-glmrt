"""Staged, route-robust provider for generated MoE decode planners."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import ContextManager, Protocol

from b12x.policy.components import MOE_DECODE
from b12x.policy.generation.contracts import (
    ComponentGenerationResult,
    GenerationContext,
    MeasurementPartition,
    ProgressReporter,
    WorkEstimate,
)
from b12x.policy.generation.moe_corpus import (
    COMMON_PREFILL_TOKEN_CAPACITIES,
    MoePhysicalGeometry,
    MoeSweepCase,
    expand_physical_geometries,
    expand_sweep_cases,
)
from b12x.policy.generation.reducer import (
    DecisionRecord,
    build_axis_tree,
    decision_node_to_dict,
)
from b12x.policy.generation.store import CheckpointStore
from b12x.policy.types import FrozenMapping

_QUERY_FIELDS = (
    "quant_mode",
    "source_format",
    "activation",
    "num_experts",
    "hidden_size",
    "intermediate_size",
    "top_k",
    "num_tokens",
)
_COARSE_TOKENS = frozenset({1, 4, 8, 32, 128})
_COARSE_PATTERNS = frozenset({"balanced", "hot"})
_COARSE_TARGET_RATIO = 1.01
_MICRO_MAX_TOKENS = 8
_TRITON_ROUTE_MAX_ROWS = 256
_NVFP4_CAPACITY_TOKENS = frozenset(COMMON_PREFILL_TOKEN_CAPACITIES)
_QUALIFICATION_TOKENS = frozenset({1, 4, 7, 32, 128})
_QUALIFICATION_PATTERNS = frozenset({"balanced", "hot"})


def _config_id(config: FrozenMapping) -> str:
    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, kw_only=True)
class MoeCandidate:
    config: FrozenMapping
    candidate_id: str = ""

    @classmethod
    def create(cls, config: Mapping[str, object]) -> "MoeCandidate":
        frozen = FrozenMapping(config)
        return cls(config=frozen, candidate_id=_config_id(frozen))

    def __post_init__(self) -> None:
        if not self.config:
            raise ValueError("MoE candidates require a non-empty config")
        if self.candidate_id != _config_id(self.config):
            raise ValueError("MoE candidate ID does not match its config")


@dataclass(frozen=True, kw_only=True)
class MoeMeasurement:
    candidate: MoeCandidate
    latency_us: float | None
    cosine: float | None
    error: str | None = None
    metrics: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, FrozenMapping):
            if not isinstance(self.metrics, Mapping):
                raise TypeError("metrics must be an object")
            object.__setattr__(self, "metrics", FrozenMapping(self.metrics))
        if self.latency_us is not None and (
            not math.isfinite(self.latency_us) or self.latency_us <= 0
        ):
            raise ValueError("latency_us must be finite and positive")
        if self.cosine is not None and not math.isfinite(self.cosine):
            raise ValueError("cosine must be finite")
        if self.error is not None and not self.error:
            raise ValueError("measurement errors must be non-empty")

    def passes(self, minimum_cosine: float) -> bool:
        return (
            self.error is None
            and self.latency_us is not None
            and self.cosine is not None
            and self.cosine >= minimum_cosine
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "config": self.candidate.config.to_dict(),
            "latency_us": self.latency_us,
            "cosine": self.cosine,
            "error": self.error,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MoeMeasurement":
        candidate = MoeCandidate.create(value["config"])
        if value.get("candidate_id") != candidate.candidate_id:
            raise ValueError("checkpoint candidate ID does not match its config")
        latency = value.get("latency_us")
        cosine = value.get("cosine")
        error = value.get("error")
        metrics = value.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise TypeError("checkpoint measurement metrics must be an object")
        return cls(
            candidate=candidate,
            latency_us=None if latency is None else float(latency),
            cosine=None if cosine is None else float(cosine),
            error=None if error is None else str(error),
            metrics=FrozenMapping(metrics),
        )


class MoeGeometrySession(Protocol):
    @property
    def candidates(self) -> tuple[MoeCandidate, ...]: ...

    def eligible_candidates(
        self,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
    ) -> tuple[MoeCandidate, ...]: ...

    def measure(
        self,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
        *,
        correctness: bool = False,
    ) -> tuple[MoeMeasurement, ...]: ...


class MoeBenchmarkFactory(Protocol):
    def __call__(
        self,
        geometry: MoePhysicalGeometry,
        context: GenerationContext,
    ) -> ContextManager[MoeGeometrySession]: ...


def _query_key(case: MoeSweepCase) -> tuple[object, ...]:
    query = case.query()
    return tuple(query[field] for field in _QUERY_FIELDS)


def _query_dict(case: MoeSweepCase) -> dict[str, object]:
    query = case.query()
    return {field: query[field] for field in _QUERY_FIELDS}


def _geometry_partition_id(geometry: MoePhysicalGeometry) -> str:
    encoded = json.dumps(geometry.key, separators=(",", ":"))
    return f"geometry-{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"


def _config_covers_query(
    query: Mapping[str, object],
    config: Mapping[str, object],
) -> bool:
    num_tokens = int(query["num_tokens"])
    top_k = int(query["top_k"])
    if config["backend"] == "micro" and num_tokens > _MICRO_MAX_TOKENS:
        return False
    if config.get("route_planner") == "triton":
        return (
            query["quant_mode"] == "nvfp4"
            and query["activation"] == "silu"
            and num_tokens * top_k <= _TRITON_ROUTE_MAX_ROWS
        )
    return True


def _synthesize_token_capacity_coverage(
    records: Sequence[DecisionRecord],
    *,
    minimum: int,
    maximum: int,
) -> tuple[DecisionRecord, ...]:
    """Represent nearest valid token anchors with compact transition points."""

    if minimum > maximum:
        raise ValueError("token coverage minimum cannot exceed maximum")
    grouped: dict[FrozenMapping, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        value = record.query.get("num_tokens")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("num_tokens decision fields must be integers")
        if not minimum <= value <= maximum:
            raise ValueError(f"num_tokens={value} is outside [{minimum}, {maximum}]")
        group = FrozenMapping(
            {key: value for key, value in record.query.items() if key != "num_tokens"}
        )
        grouped[group].append(record)

    synthesized: list[DecisionRecord] = []
    for group in sorted(grouped, key=repr):
        anchors = sorted(
            grouped[group],
            key=lambda item: int(item.query["num_tokens"]),
        )
        anchor_values = tuple(int(item.query["num_tokens"]) for item in anchors)
        if len(anchor_values) != len(set(anchor_values)):
            raise ValueError("token capacity anchors must be unique per query")
        for anchor in anchors:
            if not _config_covers_query(anchor.query, anchor.config):
                raise ValueError(
                    "measured MoE config is invalid at "
                    f"num_tokens={anchor.query['num_tokens']}"
                )

        transition_points = {minimum, maximum, *anchor_values}
        for index, left in enumerate(anchor_values):
            for right in anchor_values[index + 1 :]:
                midpoint = (left + right) // 2
                transition_points.update((midpoint, midpoint + 1))
        top_k = int(group["top_k"])
        triton_limit = _TRITON_ROUTE_MAX_ROWS // top_k
        transition_points.update(
            (
                _MICRO_MAX_TOKENS,
                _MICRO_MAX_TOKENS + 1,
                triton_limit,
                triton_limit + 1,
            )
        )

        for num_tokens in sorted(
            value for value in transition_points if minimum <= value <= maximum
        ):
            query = {**group.to_dict(), "num_tokens": num_tokens}
            winner = next(
                (
                    anchor
                    for anchor in sorted(
                        anchors,
                        key=lambda item: (
                            abs(int(item.query["num_tokens"]) - num_tokens),
                            int(item.query["num_tokens"]),
                        ),
                    )
                    if _config_covers_query(query, anchor.config)
                ),
                None,
            )
            if winner is None:
                raise ValueError(
                    f"no measured MoE config covers num_tokens={num_tokens} for {group}"
                )
            synthesized.append(DecisionRecord.create(query=query, config=winner.config))
    return tuple(synthesized)


def _coarse_cases(cases: Sequence[MoeSweepCase]) -> tuple[MoeSweepCase, ...]:
    return tuple(
        case
        for case in cases
        if case.is_model_native_top_k
        and case.num_tokens in _COARSE_TOKENS
        and case.route_pattern in _COARSE_PATTERNS
    )


def _has_tunable_backend(geometry: MoePhysicalGeometry) -> bool:
    return geometry.recipe.quant_mode == "nvfp4"


def _measurement_cases(
    cases: Sequence[MoeSweepCase],
) -> tuple[MoeSweepCase, ...]:
    cases = tuple(cases)
    if not cases:
        return ()
    if _has_tunable_backend(cases[0].geometry):
        return tuple(
            case
            for case in cases
            if case.num_tokens not in _NVFP4_CAPACITY_TOKENS
            or (
                case.is_model_native_top_k
                and case.route_pattern in _QUALIFICATION_PATTERNS
            )
        )
    return tuple(
        case
        for case in cases
        if case.is_model_native_top_k
        and case.num_tokens in _QUALIFICATION_TOKENS
        and case.route_pattern in _QUALIFICATION_PATTERNS
    )


def _select_coarse_candidates(
    *,
    candidates: tuple[MoeCandidate, ...],
    measurements_by_case: Sequence[tuple[MoeMeasurement, ...]],
    minimum_cosine: float,
) -> tuple[MoeCandidate, ...]:
    valid_by_case = [
        tuple(
            measurement
            for measurement in measurements
            if measurement.passes(minimum_cosine)
        )
        for measurements in measurements_by_case
    ]
    if any(not measurements for measurements in valid_by_case):
        raise ValueError("coarse candidate selection requires a valid candidate")
    kept = {
        candidate.candidate_id
        for candidate in candidates
        if candidate.config.get("route_planner") != "triton"
    }
    explicit_triton = [
        candidate
        for candidate in candidates
        if candidate.config.get("route_planner") == "triton"
        and isinstance(candidate.config.get("max_active_clusters"), int)
    ]
    if explicit_triton:
        kept.add(
            max(
                explicit_triton,
                key=lambda candidate: int(candidate.config["max_active_clusters"]),
            ).candidate_id
        )

    def covered(
        measurements: tuple[MoeMeasurement, ...],
        candidate_ids: set[str],
    ) -> bool:
        best = min(float(item.latency_us) for item in measurements)
        selected = [
            float(item.latency_us)
            for item in measurements
            if item.candidate.candidate_id in candidate_ids
        ]
        return bool(selected) and min(selected) <= best * _COARSE_TARGET_RATIO

    while True:
        uncovered = [
            measurements
            for measurements in valid_by_case
            if not covered(measurements, kept)
        ]
        if not uncovered:
            break
        available = [
            candidate for candidate in candidates if candidate.candidate_id not in kept
        ]
        if not available:
            raise RuntimeError("coarse candidate coverage is incomplete")

        def rank(candidate: MoeCandidate) -> tuple[int, float, str]:
            candidate_id = candidate.candidate_id
            gain = 0
            normalized_latency = 0.0
            for measurements in uncovered:
                best = min(float(item.latency_us) for item in measurements)
                selected = next(
                    (
                        item
                        for item in measurements
                        if item.candidate.candidate_id == candidate_id
                    ),
                    None,
                )
                if selected is None:
                    continue
                ratio = float(selected.latency_us) / best
                normalized_latency += ratio
                gain += ratio <= _COARSE_TARGET_RATIO
            return gain, -normalized_latency, candidate_id

        winner = max(available, key=rank)
        if rank(winner)[0] == 0:
            raise RuntimeError("no candidate can cover a coarse measurement")
        kept.add(winner.candidate_id)
    return tuple(
        candidate for candidate in candidates if candidate.candidate_id in kept
    )


class MoeDecodeGenerator:
    """Generate a broad MoE planner from staged per-geometry GPU races."""

    component_id = MOE_DECODE
    query_schema_version = 2
    config_schema_version = 1

    def __init__(
        self,
        *,
        benchmark_factory: MoeBenchmarkFactory | None = None,
        geometries: tuple[MoePhysicalGeometry, ...] | None = None,
        cases: tuple[MoeSweepCase, ...] | None = None,
    ) -> None:
        self._geometries = geometries or expand_physical_geometries()
        self._cases = cases or expand_sweep_cases(geometries=self._geometries)
        if benchmark_factory is None:
            from .moe_gpu_worker import MoeGpuBenchmarkFactory

            benchmark_factory = MoeGpuBenchmarkFactory()
        self._benchmark_factory = benchmark_factory
        known_keys = {geometry.key for geometry in self._geometries}
        if any(case.geometry.key not in known_keys for case in self._cases):
            raise ValueError("MoE sweep cases reference unknown geometries")

    def estimate(self, context: GenerationContext) -> WorkEstimate:
        del context
        cases_by_geometry: dict[tuple[object, ...], list[MoeSweepCase]] = defaultdict(
            list
        )
        for case in self._cases:
            cases_by_geometry[case.geometry.key].append(case)
        coarse = tuple(
            case
            for geometry in self._geometries
            if _has_tunable_backend(geometry)
            for case in _coarse_cases(cases_by_geometry[geometry.key])
        )
        measured = tuple(
            case
            for geometry in self._geometries
            for case in _measurement_cases(cases_by_geometry[geometry.key])
        )
        query_count = len({_query_key(case) for case in self._cases})
        work_units = len(self._geometries) + len(coarse) + len(measured) + query_count
        return WorkEstimate(
            component_id=self.component_id,
            work_units=work_units,
            case_count=len(measured),
            description=(
                f"{len(self._geometries)} physical geometries; full NVFP4 "
                "candidate race, fixed-backend qualification, tree reduction"
            ),
            dimensions={
                "physical_geometries": len(self._geometries),
                "measurement_cases": len(measured),
                "runtime_route_cases": len(self._cases),
                "runtime_queries": query_count,
                "coarse_cases": len(coarse),
            },
        )

    def measurement_partitions(
        self,
        context: GenerationContext,
    ) -> tuple[MeasurementPartition, ...]:
        del context
        cases_by_geometry: dict[tuple[object, ...], list[MoeSweepCase]] = defaultdict(
            list
        )
        for case in self._cases:
            cases_by_geometry[case.geometry.key].append(case)
        partitions = []
        for geometry in self._geometries:
            cases = tuple(cases_by_geometry[geometry.key])
            measured = _measurement_cases(cases)
            coarse = _coarse_cases(cases) if _has_tunable_backend(geometry) else ()
            query_count = len({_query_key(case) for case in cases})
            partitions.append(
                MeasurementPartition(
                    component_id=self.component_id,
                    partition_id=_geometry_partition_id(geometry),
                    work_units=1 + len(coarse) + len(measured) + query_count,
                    case_count=len(measured),
                    description=(
                        f"{geometry.recipe.recipe_id} E={geometry.num_experts} "
                        f"K={geometry.hidden_size} N={geometry.intermediate_size}"
                    ),
                )
            )
        return tuple(partitions)

    def select_measurement_partitions(
        self,
        partition_ids: tuple[str, ...],
    ) -> "MoeDecodeGenerator":
        selected = frozenset(partition_ids)
        by_id = {
            _geometry_partition_id(geometry): geometry for geometry in self._geometries
        }
        unknown = selected - frozenset(by_id)
        if not selected or unknown:
            raise ValueError(
                "invalid MoE measurement partitions: "
                f"{sorted(unknown) if unknown else 'empty selection'}"
            )
        geometries = tuple(
            geometry
            for geometry in self._geometries
            if _geometry_partition_id(geometry) in selected
        )
        geometry_keys = frozenset(geometry.key for geometry in geometries)
        restricted = copy(self)
        restricted._geometries = geometries
        restricted._cases = tuple(
            case for case in self._cases if case.geometry.key in geometry_keys
        )
        return restricted

    def _race(
        self,
        *,
        stage: str,
        case: MoeSweepCase,
        candidates: tuple[MoeCandidate, ...],
        session: MoeGeometrySession,
        context: GenerationContext,
        checkpoints: CheckpointStore,
    ) -> tuple[MoeMeasurement, ...]:
        key = f"{stage}-{case.case_id}"
        cached = checkpoints.load(self.component_id, key)
        expected_ids = [candidate.candidate_id for candidate in candidates]
        if (
            cached is not None
            and cached.get("schema_version") == 1
            and context.checkpoint_metadata_matches(cached.get("generation"))
            and cached.get("case_id") == case.case_id
            and cached.get("candidate_ids") == expected_ids
        ):
            raw_measurements = cached.get("measurements")
            if not isinstance(raw_measurements, list):
                raise TypeError("MoE race checkpoint measurements must be an array")
            cached_measurements = tuple(
                MoeMeasurement.from_dict(item) for item in raw_measurements
            )
            if any(
                item.error is None and item.latency_us is not None
                for item in cached_measurements
            ):
                return cached_measurements

        measurements = session.measure(
            case,
            candidates,
            correctness=stage == "screen",
        )
        measured_ids = [item.candidate.candidate_id for item in measurements]
        if measured_ids != expected_ids:
            raise ValueError(
                "MoE measurement session must preserve the requested candidate order"
            )
        checkpoints.save(
            self.component_id,
            key,
            {
                "schema_version": 1,
                "generation": context.checkpoint_metadata(),
                "case_id": case.case_id,
                "query": case.query(),
                "route_pattern": case.route_pattern,
                "candidate_ids": expected_ids,
                "measurements": [item.to_dict() for item in measurements],
            },
        )
        return measurements

    def _screen(
        self,
        *,
        cases: tuple[MoeSweepCase, ...],
        session: MoeGeometrySession,
        context: GenerationContext,
        checkpoints: CheckpointStore,
    ) -> tuple[MoeCandidate, ...]:
        anchor = min(
            cases,
            key=lambda case: (
                not case.is_model_native_top_k,
                abs(case.num_tokens - 4),
                case.route_pattern != "balanced",
                case.num_tokens,
            ),
        )
        measurements = self._race(
            stage="screen",
            case=anchor,
            candidates=session.candidates,
            session=session,
            context=context,
            checkpoints=checkpoints,
        )
        survivors = tuple(
            item.candidate
            for item in measurements
            if item.passes(context.settings.minimum_cosine)
        )
        if not survivors:
            raise RuntimeError(
                f"all MoE candidates failed correctness for {anchor.case_id}"
            )
        return survivors

    def _coarse_race(
        self,
        *,
        cases: tuple[MoeSweepCase, ...],
        candidates: tuple[MoeCandidate, ...],
        session: MoeGeometrySession,
        context: GenerationContext,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> tuple[MoeCandidate, ...]:
        if len(candidates) == 1:
            return candidates
        measurements_by_case: list[tuple[MoeMeasurement, ...]] = []
        for case in _coarse_cases(cases):
            progress.advance(
                self.component_id,
                units=0,
                detail=f"coarse {case.case_id}",
            )
            measurements = self._race(
                stage="coarse",
                case=case,
                candidates=candidates,
                session=session,
                context=context,
                checkpoints=checkpoints,
            )
            valid = [
                item
                for item in measurements
                if item.passes(context.settings.minimum_cosine)
            ]
            if not valid:
                raise RuntimeError(
                    f"all MoE candidates failed coarse race {case.case_id}"
                )
            measurements_by_case.append(tuple(valid))
            progress.advance(
                self.component_id,
                detail=f"coarse {case.case_id}",
            )
        if not measurements_by_case:
            return candidates
        return _select_coarse_candidates(
            candidates=candidates,
            measurements_by_case=measurements_by_case,
            minimum_cosine=context.settings.minimum_cosine,
        )

    def generate(
        self,
        context: GenerationContext,
        *,
        progress: ProgressReporter,
        checkpoints: CheckpointStore,
    ) -> ComponentGenerationResult:
        cases_by_geometry: dict[
            tuple[object, ...],
            list[MoeSweepCase],
        ] = defaultdict(list)
        for case in self._cases:
            cases_by_geometry[case.geometry.key].append(case)
        full_results: list[tuple[MoeSweepCase, tuple[MoeMeasurement, ...]]] = []
        fixed_configs: dict[tuple[object, ...], FrozenMapping] = {}
        capacity_configs: dict[tuple[object, ...], FrozenMapping] = {}
        single_candidate_route_cases = 0
        coarse_measurement_cases = 0
        for geometry_index, geometry in enumerate(self._geometries, start=1):
            geometry_cases = tuple(cases_by_geometry[geometry.key])
            measurement_cases = _measurement_cases(geometry_cases)
            progress.start_stage(
                self.component_id,
                stage=(
                    f"geometry {geometry_index}/{len(self._geometries)}: "
                    "prepare and correctness screen"
                ),
                total=1,
            )
            progress.advance(
                self.component_id,
                units=0,
                detail=f"prepare {geometry.key}",
            )
            with self._benchmark_factory(geometry, context) as session:
                progress.advance(
                    self.component_id,
                    units=0,
                    detail=f"screen {geometry.key}",
                )
                candidates = self._screen(
                    cases=geometry_cases,
                    session=session,
                    context=context,
                    checkpoints=checkpoints,
                )
                if not _has_tunable_backend(geometry):
                    if len(candidates) != 1:
                        raise RuntimeError(
                            "fixed-backend MoE geometry exposed multiple candidates"
                        )
                    fixed_configs[geometry.key] = candidates[0].config
                progress.advance(
                    self.component_id,
                    detail=f"screen {geometry.key}",
                )
                progress.start_stage(
                    self.component_id,
                    stage=(
                        f"geometry {geometry_index}/{len(self._geometries)}: "
                        "coarse candidate race"
                    ),
                    total=(
                        len(_coarse_cases(geometry_cases)) if len(candidates) > 1 else 0
                    ),
                )
                if len(candidates) > 1:
                    coarse_measurement_cases += len(_coarse_cases(geometry_cases))
                candidates = self._coarse_race(
                    cases=geometry_cases,
                    candidates=candidates,
                    session=session,
                    context=context,
                    progress=progress,
                    checkpoints=checkpoints,
                )
                if _has_tunable_backend(geometry):
                    for case in geometry_cases:
                        if case.num_tokens not in _NVFP4_CAPACITY_TOKENS:
                            continue
                        eligible = session.eligible_candidates(case, candidates)
                        if len(eligible) != 1:
                            raise RuntimeError(
                                "prefill capacity planning requires exactly one "
                                f"eligible MoE candidate for {case.case_id}"
                            )
                        key = _query_key(case)
                        config = eligible[0].config
                        previous = capacity_configs.setdefault(key, config)
                        if previous != config:
                            raise RuntimeError(
                                "prefill capacity routes disagree on the forced "
                                f"MoE config for {_query_dict(case)}"
                            )
                progress.start_stage(
                    self.component_id,
                    stage=(
                        f"geometry {geometry_index}/{len(self._geometries)}: "
                        "measured route race or qualification"
                    ),
                    total=len(measurement_cases),
                )
                for case in measurement_cases:
                    progress.advance(
                        self.component_id,
                        units=0,
                        detail=f"full {case.case_id}",
                    )
                    eligible = session.eligible_candidates(case, candidates)
                    if not eligible:
                        raise RuntimeError(
                            f"no applicable MoE candidate for {case.case_id}"
                        )
                    if len(eligible) == 1:
                        single_candidate_route_cases += 1
                    measurements = self._race(
                        stage="full",
                        case=case,
                        candidates=eligible,
                        session=session,
                        context=context,
                        checkpoints=checkpoints,
                    )
                    full_results.append((case, measurements))
                    progress.advance(
                        self.component_id,
                        detail=f"full {case.case_id}",
                    )

        results_by_query: dict[
            tuple[object, ...],
            list[tuple[MoeSweepCase, tuple[MoeMeasurement, ...]]],
        ] = defaultdict(list)
        for case, measurements in full_results:
            results_by_query[_query_key(case)].append((case, measurements))
        records: list[DecisionRecord] = []
        winner_counts: dict[str, int] = defaultdict(int)
        progress.start_stage(
            self.component_id,
            stage="route-robust reduction",
            total=len(results_by_query),
        )
        for grouped in results_by_query.values():
            by_candidate: dict[str, list[MoeMeasurement]] = defaultdict(list)
            for _case, measurements in grouped:
                for measurement in measurements:
                    if measurement.passes(context.settings.minimum_cosine):
                        by_candidate[measurement.candidate.candidate_id].append(
                            measurement
                        )
            required_patterns = {case.route_pattern for case, _ in grouped}
            robust: list[tuple[float, MoeCandidate]] = []
            for measurements in by_candidate.values():
                if len(measurements) != len(required_patterns):
                    continue
                score = math.exp(
                    sum(math.log(float(item.latency_us)) for item in measurements)
                    / len(measurements)
                )
                robust.append((score, measurements[0].candidate))
            if not robust:
                case = grouped[0][0]
                raise RuntimeError(
                    f"no route-robust MoE candidate for {_query_dict(case)}"
                )
            _, winner = min(robust, key=lambda item: (item[0], item[1].candidate_id))
            representative = grouped[0][0]
            records.append(
                DecisionRecord.create(
                    query=_query_dict(representative),
                    config=winner.config,
                )
            )
            winner_counts[winner.candidate_id] += 1
            progress.advance(
                self.component_id,
                detail=f"reduce {representative.case_id}",
            )

        measured_query_keys = {record.query for record in records}
        qualified_capacity_pairs = {
            (case.geometry.key, case.num_tokens)
            for case, _measurements in full_results
            if case.num_tokens in _NVFP4_CAPACITY_TOKENS
        }
        qualified_fixed_query_points = 0
        qualified_capacity_query_points = 0
        for case in self._cases:
            config = fixed_configs.get(case.geometry.key)
            is_capacity_config = False
            if config is None and case.num_tokens in _NVFP4_CAPACITY_TOKENS:
                qualification_key = (case.geometry.key, case.num_tokens)
                if qualification_key not in qualified_capacity_pairs:
                    raise RuntimeError(
                        "missing GPU qualification for MoE prefill capacity "
                        f"{case.num_tokens} and geometry {case.geometry.key}"
                    )
                config = capacity_configs.get(_query_key(case))
                is_capacity_config = True
            if config is None:
                continue
            query = FrozenMapping(_query_dict(case))
            if query in measured_query_keys:
                continue
            records.append(DecisionRecord(query=query, config=config))
            measured_query_keys.add(query)
            if is_capacity_config:
                qualified_capacity_query_points += 1
            else:
                qualified_fixed_query_points += 1

        measured_records = tuple(records)
        token_values = {case.num_tokens for case in self._cases}
        records = list(
            _synthesize_token_capacity_coverage(
                measured_records,
                minimum=min(token_values),
                maximum=max(token_values),
            )
        )
        planner = build_axis_tree(
            records,
            field_order=_QUERY_FIELDS,
            range_fields=frozenset({"num_tokens"}),
            nearest_range_bounds={
                "num_tokens": (min(token_values), max(token_values)),
            },
        )
        component = {
            "component_id": self.component_id,
            "query_schema_version": self.query_schema_version,
            "config_schema_version": self.config_schema_version,
            "coverage": {
                "physical_geometries": len(self._geometries),
                "measured_query_points": len(results_by_query),
                "qualified_capacity_query_points": qualified_capacity_query_points,
                "qualified_fixed_query_points": qualified_fixed_query_points,
                "runtime_query_points": len(records),
            },
            "planner": decision_node_to_dict(planner),
        }
        estimate = self.estimate(context)
        return ComponentGenerationResult(
            component=component,
            evidence={
                "coarse_target_ratio": _COARSE_TARGET_RATIO,
                "measurement_process_scope": "one_cuda_process_per_physical_geometry",
                "winner_query_counts": dict(sorted(winner_counts.items())),
                "route_measurements": len(full_results),
                "single_candidate_route_cases": single_candidate_route_cases,
                "gpu_measurement_cases": (
                    len(self._geometries) + coarse_measurement_cases + len(full_results)
                ),
                "fixed_backend_qualification_tokens": sorted(_QUALIFICATION_TOKENS),
                "fixed_backend_qualification_patterns": sorted(_QUALIFICATION_PATTERNS),
                "prefill_token_capacities": sorted(
                    token_values & _NVFP4_CAPACITY_TOKENS
                ),
            },
            completed_work_units=estimate.work_units,
        )


__all__ = [
    "MoeBenchmarkFactory",
    "MoeCandidate",
    "MoeDecodeGenerator",
    "MoeGeometrySession",
    "MoeMeasurement",
]
