"""Correctness-gated graph races for chunked KDA prefill.

Only the exact measured geometries are covered; other capacities and launch
knobs retain the component heuristic. The recurrent oracle runs outside timing.
"""

from __future__ import annotations

import gc
from contextlib import AbstractContextManager

from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)


def kda_prefill_cases() -> tuple[SweepCase, ...]:
    from b12x.sequence.kda_prefill._policy import KdaPrefillQuery

    return tuple(
        SweepCase.create(
            group_id=f"kda-h{heads}",
            query=KdaPrefillQuery(
                heads=heads,
                head_dim=128,
                model_dtype="bfloat16",
                state_dtype="float32",
                qk_l2norm=True,
                checkpoint_export=checkpoint,
                max_tokens=tokens,
                max_seqs=1,
            ).profile_fields(),
            scenario="random-bounded-gate",
        )
        for heads in (16, 48)
        for tokens in (128, 1024)
        for checkpoint in (False, True)
    )


def _benchmark(case, candidate, context):
    import torch

    from benchmarks.benchmark_ple import _time_case
    from b12x.policy import KDA_PREFILL, PolicyContext, PolicyMode
    from b12x.sequence.kda_prefill import api
    from b12x.sequence.kda_prefill._policy import KdaPrefillConfig
    from b12x.sequence.kda_prefill.reference import prefill_kda

    from .gpu_workers import _l2_flush_fn

    query = case.query
    tokens, heads = int(query["max_tokens"]), int(query["heads"])
    device = torch.device("cuda", context.device_ordinal)
    generator = torch.Generator(device=device).manual_seed(context.settings.seed)

    def random(shape, scale=0.25, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device, generator=generator) * scale).to(dtype)

    def index(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    checkpoint = bool(query["checkpoint_export"])
    tensors = {
        name: random((tokens, heads, 128)) for name in ("q", "k", "v")
    }
    tensors.update(
        raw_g=random((tokens, heads, 128), 1.0),
        raw_beta=random((tokens, heads), 1.0),
        A_log=random((heads,), 0.1, torch.float32),
        dt_bias=random((heads, 128), 0.1, torch.float32),
        recurrent_state=random((3, heads, 128, 128), 0.1, torch.float32),
        cu_seqlens=index([0, tokens]),
        initial_state_indices=index([0]),
        final_state_indices=index([1]),
        checkpoint_state_indices=index([2]),
        checkpoint_offsets=index([tokens // 2 if checkpoint else 0]),
        num_seqs=index([1]),
        num_tokens=index([tokens]),
    )
    initial_pool = tensors["recurrent_state"].clone()
    expected_pool = initial_pool.clone()
    oracle_inputs = dict(tensors, recurrent_state=expected_pool)
    expected_output = prefill_kda(**oracle_inputs, lower_bound=-5.0)
    policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
    policy = policy.with_override(
        KDA_PREFILL, KdaPrefillConfig.from_profile(candidate.config)
    )
    plan = api.plan(
        api.Caps(
            device=device, max_tokens=tokens, max_seqs=1, max_state_slots=3,
            heads=heads, checkpoint_export=checkpoint,
        ),
        policy=policy,
    )
    scratch = torch.empty(plan.scratch_specs()[0].shape, device=device, dtype=torch.uint8)
    output = torch.empty_like(tensors["q"])
    binding = api.bind(plan, scratch=scratch, output=output, **tensors)

    def restore():
        tensors["recurrent_state"].copy_(initial_pool)

    def error_ratio(name, expected, actual, limit):
        if not bool(torch.isfinite(actual).all()) or not bool(actual.count_nonzero()):
            raise AssertionError(f"{name}: non-finite or zero result")
        expected, actual = expected.float(), actual.float()
        delta = (expected - actual).abs()
        rms = expected.square().mean().sqrt()
        ratio = float(delta.square().mean().sqrt() / (rms + 1e-8))
        if ratio >= limit or float(delta.max()) > float(
            0.04 * rms + 2**-6 * expected.abs().max()
        ):
            raise AssertionError(f"{name}: oracle error ratio {ratio}")
        cosine = float(torch.nn.functional.cosine_similarity(
            expected.flatten(), actual.flatten(), dim=0
        ))
        if cosine < context.settings.minimum_cosine:
            raise AssertionError(f"{name}: oracle cosine {cosine}")
        return ratio

    def validate():
        if binding.error_code.item() != 0:
            raise AssertionError(f"KDA metadata error {binding.error_code.item()}")
        out_error = error_ratio("output", expected_output, output, 1e-2)
        written = [1, 2] if checkpoint else [1]
        state_error = max(
            error_ratio("state", expected_pool[slot], tensors["recurrent_state"][slot], 5e-3)
            for slot in written
        )
        for slot in set(range(3)) - set(written):
            torch.testing.assert_close(
                tensors["recurrent_state"][slot], initial_pool[slot], rtol=0, atol=0
            )
        return {"status": "passed", "output_rmse_ratio": out_error,
                "state_rmse_ratio": state_error}

    timing, correctness, graph_contract = _time_case(
        launch=lambda: api.run(binding, lower_bound=-5.0),
        validate=validate,
        address_tensors=dict(tensors, out=output, scratch=scratch),
        prepare=restore,
        mode="graph",
        warmup=context.settings.warmup,
        samples=context.settings.groups * context.settings.repetitions,
        l2_flush=_l2_flush_fn(device, enabled=context.settings.cold_l2),
        device=device,
    )
    return SweepMeasurement(
        candidate=candidate,
        latency_us=float(timing["kernel"]["median"]),
        correct=correctness["status"] == "passed" and graph_contract is not None,
        metrics={"timing": timing, "correctness": correctness,
                 "graph_contract": graph_contract},
    )


class _KdaSession(AbstractContextManager):
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        import torch

        gc.collect()
        torch.cuda.synchronize(self.context.device_ordinal)
        torch.cuda.empty_cache()

    def candidates(self, case):
        from b12x.sequence.kda_prefill._policy import (
            KdaPrefillConfig, default_window_tiles,
        )

        return tuple(
            SweepCandidate.create(KdaPrefillConfig(
                v_split=split,
                window_tiles=default_window_tiles(
                    int(case.query["heads"]), int(case.query["max_tokens"]), 1
                ),
            ).to_dict())
            for split in (64, 32)
        )

    def measure(self, case, candidates):
        results = []
        for candidate in candidates:
            try:
                results.append(_benchmark(case, candidate, self.context))
            except Exception as error:
                results.append(SweepMeasurement(
                    candidate=candidate, latency_us=None, correct=False,
                    error=f"{type(error).__name__}: {error}",
                ))
        return tuple(results)


class _KdaBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _KdaSession(context)


class KdaPrefillGenerator(DiscreteSweepGenerator):
    def __init__(self, *, benchmark_factory=None, cases=None):
        from b12x.sequence.kda_prefill._policy import KDA_PREFILL_POLICY, KdaPrefillQuery

        super().__init__(
            component_id=KDA_PREFILL_POLICY.component_id,
            query_schema_version=KDA_PREFILL_POLICY.query_schema_version,
            config_schema_version=KDA_PREFILL_POLICY.config_schema_version,
            query_fields=tuple(KdaPrefillQuery.__dataclass_fields__),
            range_fields=frozenset(),
            cases=kda_prefill_cases() if cases is None else cases,
            benchmark_factory=benchmark_factory or _KdaBenchmarkFactory(),
            coverage={"candidate_v_splits": [64, 32], "unmeasured_queries": "heuristic"},
            candidate_contract_version=1,
        )
