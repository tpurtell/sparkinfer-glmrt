from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from b12x.gemm import mla_query_projection
from b12x.gemm.mla_query_projection._tuning import ProjectionQuery
from b12x.preparation import PreparationSession, PreparedCall
from tests._reference.helpers import require_b12x
from tests.gemm.test_bmm import _make_pack, _rhs_views


def _inputs(*, heads, m, weight_format, seed=31):
    torch.manual_seed(seed)
    q_nope = torch.randn(heads, m, 192, device="cuda", dtype=torch.bfloat16)
    q_full = torch.randn(m, heads, 576, device="cuda", dtype=torch.bfloat16)
    q_pe = q_full[..., 512:]
    q_scale = torch.tensor([.037], device="cuda", dtype=torch.float32)
    if weight_format == "mxfp8":
        values, scales = _make_pack(seed=seed, batch=heads)
        weight = _rhs_views(values, scales, batch=heads)["n"]
    else:
        weight = torch.randn(heads, 192, 512, device="cuda", dtype=torch.bfloat16) * .05
    return q_nope, weight, q_pe, q_scale


@contextmanager
def _prepared(weight, *, heads, m, output_dtype):
    weight_format = "bf16" if isinstance(weight, torch.Tensor) else "mxfp8"
    query = ProjectionQuery(heads=heads, max_rows=m, weight_format=weight_format,
                            output_dtype=str(output_dtype).removeprefix("torch."), b_major="n", sf_axis="n")
    q_nope = torch.zeros(heads, m, 192, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.zeros(m, heads, 64, device="cuda", dtype=torch.bfloat16)
    q_scale = torch.ones(1, device="cuda", dtype=torch.float32) if output_dtype == torch.float8_e4m3fn else None
    out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
    plan = mla_query_projection.plan(query)
    request = plan.request(name="mla",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(q_nope, weight, q_pe, out, q_scale=q_scale)))
    session = PreparationSession(autotune=False)
    result = session.prepare((request,))
    try: yield plan
    finally: result.close(); session.close()


def _reference(q_nope, weight, q_pe):
    if isinstance(weight, torch.Tensor): projected = torch.bmm(q_nope, weight)
    else:
        values, scales = weight
        physical = values.to(torch.bfloat16) * scales.view(torch.float8_e8m0fnu).to(torch.bfloat16).repeat_interleave(32, dim=-1)
        projected = torch.bmm(q_nope, physical)
    return torch.cat((projected.transpose(0, 1), q_pe), dim=-1)


@pytest.mark.parametrize("weight_format,heads", [("mxfp8", 8), ("mxfp8", 16), ("bf16", 11)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_projection_preserves_weight_forms_and_output_modes(weight_format, heads, output_dtype):
    require_b12x(); m = 4
    q_nope, weight, q_pe, q_scale = _inputs(heads=heads, m=m, weight_format=weight_format)
    out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
    with _prepared(weight, heads=heads, m=m, output_dtype=output_dtype) as plan:
        assert mla_query_projection.run(q_nope, weight, q_pe, out, plan=plan,
                                        q_scale=q_scale if output_dtype == torch.float8_e4m3fn else None) is out
    expected = _reference(q_nope, weight, q_pe)
    if output_dtype == torch.bfloat16: torch.testing.assert_close(out, expected, rtol=.03, atol=.03)
    else: assert torch.equal(out.view(torch.uint8), (expected.float() / q_scale).clamp(-448, 448).to(output_dtype).view(torch.uint8))


@pytest.mark.parametrize("weight_format,heads", [("mxfp8", 8), ("bf16", 11)])
def test_prepared_projection_graph_replays_changed_inputs(weight_format, heads):
    require_b12x(); m = 4
    q_nope, weight, q_pe, q_scale = _inputs(heads=heads, m=m, weight_format=weight_format)
    out = torch.empty(m, heads, 576, device="cuda", dtype=torch.bfloat16)
    with _prepared(weight, heads=heads, m=m, output_dtype=torch.bfloat16) as plan:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): mla_query_projection.run(q_nope, weight, q_pe, out, plan=plan)
        fresh_nope, fresh_pe = torch.randn_like(q_nope), torch.randn_like(q_pe)
        q_nope.copy_(fresh_nope); q_pe.copy_(fresh_pe); graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(out, _reference(fresh_nope, weight, fresh_pe), rtol=.03, atol=.03)


def test_projection_rejects_unprepared_execution():
    require_b12x()
    q_nope, weight, q_pe, _ = _inputs(heads=8, m=1, weight_format="mxfp8")
    out = torch.empty(1, 8, 576, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        mla_query_projection.run(q_nope, weight, q_pe, out)


def _glm_h64_bf16_inputs(
    *, m: int, seed: int = 73, nope_dim: int = 192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Exercise the actual GLM views: token-major Q exposed head-major without
    # a copy, a K prefix of the [H,448,512] resident KV-B tensor, and a RoPE
    # suffix retaining the full 576-element token/head stride.
    q_token_major = (
        torch.randn(
            m, 64, nope_dim + 64, device="cuda", dtype=torch.bfloat16
        )
        * 0.5
    )
    q_nope = q_token_major[..., :nope_dim].permute(1, 0, 2)
    kv_b = torch.randn(64, 448, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    if nope_dim > int(kv_b.shape[1]):
        raise ValueError(f"test nope_dim exceeds resident KV-B width: {nope_dim}")
    weight = kv_b[:, :nope_dim, :]
    q_full = torch.randn(m, 64, 576, device="cuda", dtype=torch.bfloat16)
    q_pe = q_full[..., 512:]
    return q_nope, weight, q_pe


@pytest.mark.parametrize("m", [1, 2, 16, 32])
def test_glm_h64_bf16_matches_staged_projection_with_production_views(m: int) -> None:
    require_b12x()
    q_nope, weight, q_pe = _glm_h64_bf16_inputs(m=m)
    expected = torch.cat(
        (torch.bmm(q_nope, weight).transpose(0, 1), q_pe),
        dim=-1,
    )
    backing = torch.empty(m, 64, 584, device="cuda", dtype=torch.bfloat16)
    out = backing[..., :576]

    returned = mla_query_projection.run_glm_h64_bf16(q_nope, weight, q_pe, out)

    assert returned is out
    assert torch.equal(out, expected)
    assert torch.equal(out[..., 512:], q_pe)


@pytest.mark.parametrize("m", [2, 16])
def test_glm_h64_bf16_nope_writes_exact_zero_suffix(m: int) -> None:
    require_b12x()
    q_nope, weight, q_pe = _glm_h64_bf16_inputs(m=m, nope_dim=256)
    q_pe = q_pe[..., :0]
    expected_nope = torch.bmm(q_nope, weight).transpose(0, 1)
    out = torch.full(
        (m, 64, 576), 17.0, device="cuda", dtype=torch.bfloat16
    )

    returned = mla_query_projection.run_glm_h64_bf16(q_nope, weight, q_pe, out)

    assert returned is out
    assert torch.equal(out[..., :512], expected_nope)
    assert torch.count_nonzero(out[..., 512:]) == 0


def test_glm_h64_bf16_nope_cuda_graph_replay() -> None:
    require_b12x()
    q_nope, weight, q_pe = _glm_h64_bf16_inputs(m=4, nope_dim=256)
    q_pe = q_pe[..., :0]
    assert mla_query_projection.prewarm_glm_h64_bf16(
        weight, [4], nope=True
    ) == 1
    out = torch.empty(4, 64, 576, device="cuda", dtype=torch.bfloat16)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_query_projection.run_glm_h64_bf16(q_nope, weight, q_pe, out)
    fresh_nope = torch.randn_like(q_nope)
    q_nope.copy_(fresh_nope)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(out[..., :512], torch.bmm(fresh_nope, weight).transpose(0, 1))
    assert torch.count_nonzero(out[..., 512:]) == 0


def test_glm_h64_bf16_cuda_graph_replays_fresh_strided_inputs() -> None:
    require_b12x()
    q_nope, weight, q_pe = _glm_h64_bf16_inputs(m=4)
    assert mla_query_projection.prewarm_glm_h64_bf16(weight, [4]) == 1
    out = torch.empty(4, 64, 576, device="cuda", dtype=torch.bfloat16)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_query_projection.run_glm_h64_bf16(q_nope, weight, q_pe, out)

    first_nope = torch.randn_like(q_nope)
    first_pe = torch.randn_like(q_pe)
    q_nope.copy_(first_nope)
    q_pe.copy_(first_pe)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.cat(
        (torch.bmm(first_nope, weight).transpose(0, 1), first_pe),
        dim=-1,
    )
    assert torch.equal(out, expected)

    second_nope = torch.randn_like(q_nope)
    q_nope.copy_(second_nope)
    allocation_before = torch.cuda.memory_allocated()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    allocation_after = torch.cuda.memory_allocated()
    expected = torch.cat(
        (torch.bmm(second_nope, weight).transpose(0, 1), first_pe),
        dim=-1,
    )
    assert torch.equal(out, expected)
    assert allocation_after == allocation_before


def test_glm_h64_bf16_support_gate_is_explicit_and_narrow() -> None:
    device = require_b12x()
    kwargs = dict(
        num_heads=64,
        max_m=32,
        nope_dim=192,
        latent_dim=512,
        output_dtype=torch.bfloat16,
        device=device,
    )
    assert mla_query_projection.can_implement_glm_h64_bf16(**kwargs)
    assert mla_query_projection.can_implement_glm_h64_bf16(
        **{**kwargs, "nope_dim": 256}
    )
    assert not mla_query_projection.can_implement(**{**kwargs, "weight_format": "bf16"})
    assert not mla_query_projection.can_implement_glm_h64_bf16(
        **{**kwargs, "num_heads": 16}
    )
    assert not mla_query_projection.can_implement_glm_h64_bf16(
        **{**kwargs, "max_m": 33}
    )
    assert not mla_query_projection.can_implement_glm_h64_bf16(
        **{**kwargs, "output_dtype": torch.float8_e4m3fn}
    )


def test_glm_h64_bf16_planner_keeps_fallbacks_diagnostic_reachable() -> None:
    device = require_b12x()
    geometry = dict(
        num_heads=64,
        nope_dim=192,
        latent_dim=512,
        output_dtype=torch.bfloat16,
        device=device,
    )

    automatic_decode = mla_query_projection.plan_glm_h64_bf16(
        workload="packed_decode",
        policy="auto",
        query_rows=8,
        **geometry,
    )
    assert automatic_decode.use_sparkinfer
    assert automatic_decode.reason == "automatic_packed_decode_m2_m16"

    m1_fallback = mla_query_projection.plan_glm_h64_bf16(
        workload="packed_decode",
        policy="auto",
        query_rows=1,
        **geometry,
    )
    assert not m1_fallback.use_sparkinfer
    assert m1_fallback.h64_supported
    assert m1_fallback.reason == "automatic_native_pending_m1_gate"

    prefill_fallback = mla_query_projection.plan_glm_h64_bf16(
        workload="prefill",
        policy="auto",
        query_rows=32,
        **geometry,
    )
    assert not prefill_fallback.use_sparkinfer
    assert prefill_fallback.h64_supported
    assert prefill_fallback.reason == "automatic_native_pending_prefill_gate"

    for workload, query_rows in (("packed_decode", 1), ("prefill", 32)):
        forced = mla_query_projection.plan_glm_h64_bf16(
            workload=workload,
            policy="force",
            query_rows=query_rows,
            **geometry,
        )
        assert forced.use_sparkinfer
        assert forced.reason == "explicit_force"

    disabled = mla_query_projection.plan_glm_h64_bf16(
        workload="packed_decode",
        policy="disable",
        query_rows=8,
        **geometry,
    )
    assert not disabled.use_sparkinfer
    assert disabled.h64_supported
    assert disabled.reason == "explicit_disable"

    forced_unsupported = mla_query_projection.plan_glm_h64_bf16(
        workload="prefill",
        policy="force",
        query_rows=33,
        **geometry,
    )
    assert not forced_unsupported.use_sparkinfer
    assert not forced_unsupported.h64_supported
    assert forced_unsupported.reason == "explicit_force_unsupported_native_fallback"


def test_glm_h64_bf16_rejects_fp8_output() -> None:
    require_b12x()
    q_nope, weight, q_pe = _glm_h64_bf16_inputs(m=2)
    out = torch.empty(2, 64, 576, device="cuda", dtype=torch.float8_e4m3fn)

    with pytest.raises(TypeError, match="output must be bfloat16"):
        mla_query_projection.run_glm_h64_bf16(q_nope, weight, q_pe, out)
