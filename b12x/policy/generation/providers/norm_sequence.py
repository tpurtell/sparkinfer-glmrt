"""Measured launch-policy providers for norm and sequence fusions."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from contextlib import AbstractContextManager

from b12x.policy.components import HYPERCONNECTION, MTP_FEEDBACK
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import _l2_flush_fn, _median_of_group_medians


def _hyperconnection_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"qwen-flash-next-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": 2_560,
                "streams": 4,
                "lowrank": 320,
            },
        )
        for tokens in (1, 4, 16, 64, 128)
    )


def _mtp_feedback_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"qwen-flash-next-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": 2_560,
                "streams": 4,
            },
        )
        for tokens in (1, 4, 16, 64, 128)
    )


class _GpuSession(AbstractContextManager):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None


class _HyperConnectionSession(_GpuSession):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "reduction_block_h": 4_096,
                "pointwise_block": pointwise_block,
                "reduction_num_warps": num_warps,
            }
        )
        for pointwise_block in (128, 256, 512)
        for num_warps in (4, 8)
    )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.norm.hyperconnection._policy import HyperConnectionConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_hyperconnection import (
            Profile,
            _graph_samples_us,
            _make_case,
        )

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        profile = Profile(tokens=int(case.query["max_tokens"]))
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for candidate in candidates:
            try:
                config = HyperConnectionConfig.from_profile(candidate.config)
                policy = base_policy.with_override(HYPERCONNECTION, config)
                active = _make_case(
                    profile,
                    seed=settings.seed + int(candidate.candidate_id[-8:], 16),
                    device=device,
                    policy=policy,
                )
                samples, graph_contract, correctness = _graph_samples_us(
                    active,
                    "full_chain",
                    warmup=settings.warmup,
                    samples=settings.groups * settings.repetitions,
                    l2_flush=flush,
                )
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=_median_of_group_medians(
                            tuple(samples),
                            groups=settings.groups,
                            repetitions=settings.repetitions,
                        ),
                        correct=(
                            correctness.get("status") == "passed"
                            and graph_contract.get(
                                "replay_allocation_delta_bytes"
                            )
                            == 0
                        ),
                        metrics={
                            "operator": "full_chain",
                            "replay_allocation_bytes": graph_contract[
                                "replay_allocation_delta_bytes"
                            ],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(measurements)


class _MtpFeedbackSession(_GpuSession):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "norm_block_h": 4_096,
                "norm_block_s": 4,
                "norm_num_warps": num_warps,
            }
        )
        for num_warps in (4, 8)
    )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.policy import PolicyContext, PolicyMode
        from b12x.sequence.mtp_feedback._policy import MtpFeedbackConfig
        from benchmarks.benchmark_mtp_feedback import Profile, _benchmark_profile

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        tokens = int(case.query["max_tokens"])
        profile = Profile(name=f"profile-m{tokens}", phase="mixed", tokens=tokens)
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for candidate in candidates:
            try:
                config = MtpFeedbackConfig.from_profile(candidate.config)
                policy = base_policy.with_override(MTP_FEEDBACK, config)
                result = _benchmark_profile(
                    profile,
                    seed=settings.seed + int(candidate.candidate_id[-8:], 16),
                    device=device,
                    eps=1.0e-6,
                    warmup=settings.warmup,
                    samples=settings.groups * settings.repetitions,
                    l2_flush=flush,
                    capacity_tokens=tokens,
                    policy=policy,
                )
                timings = result["timings"]
                correctness = result["correctness"]
                storage = result["storage"]
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=float(
                            timings["cuda_graph_replay"]["median_us"]
                        ),
                        correct=bool(
                            correctness["passed"]
                            and storage["graph_replay_allocation_delta_bytes"] == 0
                        ),
                        metrics={
                            "cosine": correctness[
                                "graph_replay_after_output_poison"
                            ]["cosine"],
                            "replay_allocation_bytes": storage[
                                "graph_replay_allocation_delta_bytes"
                            ],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(measurements)


class _OneCaseFactory:
    def __init__(self, session_type) -> None:
        self._session_type = session_type

    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("norm/sequence allocation groups contain one case")
        return self._session_type(context)


class HyperConnectionGenerator(DiscreteSweepGenerator):
    """Race production HyperConnection launch geometry."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=HYPERCONNECTION,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "dtype",
                "max_tokens",
                "hidden_size",
                "streams",
                "lowrank",
            ),
            range_fields=frozenset({"max_tokens"}),
            cases=_hyperconnection_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_HyperConnectionSession),
            coverage={},
            nearest_range_bounds={"max_tokens": (1, 128)},
        )


class MtpFeedbackGenerator(DiscreteSweepGenerator):
    """Race production MTP feedback normalization launch geometry."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=MTP_FEEDBACK,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=("dtype", "max_tokens", "hidden_size", "streams"),
            range_fields=frozenset({"max_tokens"}),
            cases=_mtp_feedback_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_MtpFeedbackSession),
            coverage={},
            nearest_range_bounds={"max_tokens": (1, 128)},
        )


__all__ = ["HyperConnectionGenerator", "MtpFeedbackGenerator"]
