"""Prepared NVFP4 shared-input and split-materialization numerical contracts."""
import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from .test_nvfp4_phase_kernels import _build_domain, _bf16_output_bound
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4
from ..conftest import require_b12x


def _experts(domain, *, swiglu_limit=None):
    e, h, n = domain["E"], domain["K"], domain["n"]
    declaration = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=swiglu_limit),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    one = torch.ones(e, device=domain["x"].device)
    weights = moe.PackedWeights(
        w13=domain["w13_packed"], w2=domain["w2_packed"],
        w13_block_scales=domain["w13_sfb"].view(e, -1),
        w2_block_scales=domain["w2_sfb"].view(e, -1),
        w13_global_scales=one, w2_global_scales=one,
        input_scale=one.clone(), intermediate_scale=one.clone(), immutable_input_scales=True,
    )
    return moe.prepare_weights(plan=declaration, weights=weights), weights


def _check(output, reference):
    assert torch.isfinite(output).all() and output.abs().sum() > 0
    metrics = compare_to_reference(output.float(), reference)
    assert metrics.cos > 0.9999, metrics
    assert metrics.max_abs <= _bf16_output_bound(reference), metrics


@pytest.mark.parametrize("capacity", (64, 256))
def test_prepared_shared_and_materialized_nvfp4_replay(capacity):
    device = require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=capacity, top_k=1, seed=701 + capacity)
    experts, weights = _experts(domain)
    source, ids, probabilities = domain["x"], domain["topk_ids"], domain["topk_weights"]
    capacity_spec = moe.ExecutionCapacity(max_tokens=capacity, top_k=1)
    configs = [moe.MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=128, dynamic_route_mode="grouped", nvfp4_share_input=shared,
        nvfp4_materialize_intermediate=materialized,
    ) for shared, materialized in ((False, False), (True, False), (True, True))]

    def call(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                        for spec in state.scratch.scratch_specs())
        output = torch.empty_like(source)
        binding = state.bind(a=source, topk_ids=ids, topk_weights=probabilities,
                             output=output, scratch=scratch, input_scales_static=True)
        return PreparedCall(run=lambda: state.run(binding), output=output, owners=(scratch, binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        plans = [moe.plan_execution(experts=experts, capacity=capacity_spec,
                                   invocation={"fast_math": False},
                                   routing=moe.RoutingSpec(deterministic_output=False), override=config)
                 for config in configs]
        session.prepare(tuple(plan.request(name=f"nvfp4-{i}", prepare_call=call) for i, plan in enumerate(plans)))
        scratch = [tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                         for spec in plan.scratch_specs()) for plan in plans]
        outputs = [torch.empty_like(source) for _ in plans]
        session.freeze()
        for rows in (1, capacity - 1, capacity):
            bindings = [moe.bind(plan, a=source[:rows], topk_ids=ids[:rows], topk_weights=probabilities[:rows],
                                 output=output[:rows], scratch=storage, input_scales_static=True)
                        for plan, output, storage in zip(plans, outputs, scratch, strict=True)]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for binding in bindings:
                    moe.run(binding=binding)
            source.neg_()
            ids.add_(1).remainder_(domain["E"])
            reference = moe_reference_nvfp4(
                source[:rows].float(), weights.w13, weights.w13_block_scales, weights.w13_global_scales,
                weights.w2, weights.w2_block_scales, weights.w2_global_scales,
                weights.input_scale, weights.intermediate_scale, ids[:rows], probabilities[:rows],
                domain["E"], domain["K"], domain["n"], activation="silu", quant_scale_math="direct_division",
            )
            for output in outputs:
                output.fill_(float("nan"))
            addresses = tuple(t.data_ptr() for t in outputs) + tuple(t.data_ptr() for group in scratch for t in group)
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            assert addresses == tuple(t.data_ptr() for t in outputs) + tuple(t.data_ptr() for group in scratch for t in group)
            for output in outputs:
                _check(output[:rows], reference)
                assert torch.isnan(output[rows:]).all()
            for output in outputs[1:]:
                _check(output[:rows], outputs[0][:rows].float())
            graph.reset()
        with pytest.raises(ValueError, match="prepared numerical contract"):
            moe.bind(plans[0], a=source, topk_ids=ids, topk_weights=probabilities,
                     output=outputs[0], scratch=scratch[0], fast_math=True)
        weights.input_scale[-1] = 2
        with pytest.raises(ValueError, match="shared-input contract"):
            moe.bind(plans[-1], a=source, topk_ids=ids, topk_weights=probabilities,
                     output=outputs[-1], scratch=scratch[-1], input_scales_static=True)


def test_materialized_nvfp4_swiglu_limit_changes_output():
    device = require_b12x()
    domain = _build_domain(E=8, K=256, n=128, m=512, top_k=2, seed=42)
    source, ids, probabilities = domain["x"], domain["topk_ids"], domain["topk_weights"]
    outputs = []
    config = moe.MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None, dynamic_tile_m=128,
        dynamic_route_mode="grouped", nvfp4_share_input=True, nvfp4_materialize_intermediate=True,
    )
    for limit in (None, 2.0):
        experts, weights = _experts(domain, swiglu_limit=limit)
        plan = moe.plan_execution(
            experts=experts, capacity=moe.ExecutionCapacity(max_tokens=512, top_k=2),
            invocation={"fast_math": False}, override=config,
        )
        def prepare(state):
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                            for spec in state.scratch.scratch_specs())
            output = torch.empty_like(source)
            binding = state.bind(a=source, topk_ids=ids, topk_weights=probabilities,
                                 output=output, scratch=scratch, input_scales_static=True)
            return PreparedCall(run=lambda: state.run(binding), output=output, owners=(scratch, binding))
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            session.prepare((plan.request(name="materialized-nvfp4", prepare_call=prepare),))
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in plan.scratch_specs())
            output = torch.empty_like(source)
            binding = moe.bind(plan, a=source, topk_ids=ids, topk_weights=probabilities,
                               output=output, scratch=scratch, input_scales_static=True)
            session.freeze()
            moe.run(binding=binding)
            reference = moe_reference_nvfp4(
                source.float(), weights.w13, weights.w13_block_scales, weights.w13_global_scales,
                weights.w2, weights.w2_block_scales, weights.w2_global_scales,
                weights.input_scale, weights.intermediate_scale, ids, probabilities,
                8, 256, 128, activation="silu", swiglu_limit=limit,
            )
            _check(output, reference)
            outputs.append(output.clone())
    effect = compare_to_reference(outputs[0], outputs[1])
    assert effect.rmse > _bf16_output_bound(domain["oracle"]), effect
