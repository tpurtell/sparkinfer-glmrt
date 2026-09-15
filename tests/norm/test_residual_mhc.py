from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall, require_prepared
from b12x.testing.mhc import make_inputs, post_reference, pre_reference

from ..conftest import require_b12x


def _invocation(*, norm: bool, rms_eps: float = 1e-6) -> FrozenMapping:
    return FrozenMapping({
        "operation": "post_pre", "output_mode": "provided", "has_norm_weight": norm,
        "rms_eps": rms_eps, "hc_eps": 1e-6, "sinkhorn_iters": 20,
        "norm_eps": rms_eps if norm else 0.0,
    })


def _reference(residual, x, fn, scale, bias, norm, *, rms_eps):
    previous_y, previous_post, previous_comb = pre_reference(residual, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20)
    carry = post_reference(x, residual, previous_post, previous_comb)
    raw, post, comb = pre_reference(carry, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20, y_dtype=torch.float32)
    if norm is None:
        y = raw.bfloat16()
    else:
        y = (raw.bfloat16().float() * torch.rsqrt(raw.square().mean(dim=-1, keepdim=True) + rms_eps) * norm.float()).bfloat16()
    return previous_post, previous_comb, carry, post, comb, y


def _bound_call(state, *, x, residual, previous_post, previous_comb, fn, scale, bias, norm, rms_eps):
    tokens, hidden = x.shape
    scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=x.device)
    binding = state.bind(
        scratch=scratch, tokens=tokens,
        out=torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=x.device),
        post=torch.empty((tokens, 4), dtype=torch.float32, device=x.device),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=x.device),
        y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device),
    )
    return PreparedCall(run=lambda: _impl._b12x_mhc_post_pre_impl(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=rms_eps if norm is not None else 0.0, _state=state, binding=binding), owners=(scratch, binding))


@pytest.mark.parametrize(("tokens", "rms_eps", "hidden"), [(1, 1e-20, 4096), (3, 1e-6, 4096), (1, 1e-6, 5120), (3, 1e-5, 4096)], ids=["vision-epsilon", "default-epsilon", "ds41", "glm-epsilon"])
def test_mhc_prepared_norm_matches_native_oracle(tokens: int, rms_eps: float, hidden: int) -> None:
    device = require_b12x()
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=53_001 + tokens, device=device)
    norm = torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
    previous_post, previous_comb, expected_carry, expected_post, expected_comb, expected_y = _reference(residual, x, fn, scale, bias, norm, rms_eps=rms_eps)
    declaration = mhc.plan(mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=hidden // 64), invocation=_invocation(norm=True, rms_eps=rms_eps))

    def prepare(state):
        return _bound_call(state, x=x, residual=residual, previous_post=previous_post, previous_comb=previous_comb, fn=fn, scale=scale, bias=bias, norm=norm, rms_eps=rms_eps)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="norm", prepare_call=prepare),))
        plan = declaration
        state = require_prepared(plan, "norm.mhc")
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(plan, scratch=scratch, tokens=tokens, out=torch.empty_like(residual), post=torch.empty_like(previous_post), comb=torch.empty_like(previous_comb), y=torch.empty_like(x))
        actual = mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=rms_eps, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=rms_eps, binding=binding)
        torch.testing.assert_close(actual[0], expected_carry, rtol=0.0, atol=2e-2)
        torch.testing.assert_close(actual[3], expected_y, rtol=0.0, atol=6e-3)
        torch.testing.assert_close(actual[1], expected_post, rtol=2e-6, atol=1e-5)
        torch.testing.assert_close(actual[2], expected_comb, rtol=2e-6, atol=4e-5)


def test_mhc_prepared_projection_splits_share_projection_and_keep_finalizers_distinct() -> None:
    device, hidden = require_b12x(), 4096
    requests = []
    for rows in (1, 128):
        residual, x, fn, scale, bias = make_inputs(tokens=rows, hidden_size=hidden, seed=17, device=device)
        _, previous_post, previous_comb = pre_reference(residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
        norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
        for splits in (1, 16):
            config = mhc.MhcConfig(backend="tf32_tma", projection_tile_m=16, projection_tile_n=8, projection_tile_k=256, projection_num_stages=1, projection_num_m_warps=1, projection_num_n_warps=1, projection_k_splits=splits)
            declaration = mhc.plan(mhc.Caps(device=device, max_tokens=rows, hidden_size=hidden, split_k=64), invocation=_invocation(norm=True), override=config)
            name = f"m{rows}-s{splits}"
            requests.append(declaration.request(name=name, prepare_call=lambda state, values=(x, residual, previous_post, previous_comb, fn, scale, bias, norm): _bound_call(state, x=values[0], residual=values[1], previous_post=values[2], previous_comb=values[3], fn=values[4], scale=values[5], bias=values[6], norm=values[7], rms_eps=1e-6)))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        result = session.prepare(requests)
        projection, finalizers = set(), set()
        for plan in result.plans.values():
            for program in require_prepared(plan, "norm.mhc").launchers["partial"].__b12x_programs__:
                if "mhc_prefill_tf32_project_tma_" in program.name:
                    projection.add(program)
                if "mhc_finalize_gram_" in program.name:
                    finalizers.add(program)
        assert len(projection) == 1
        assert len(finalizers) == 2


def test_mhc_prepared_high_slot_capture_replays_changed_inputs() -> None:
    device, tokens, hidden = require_b12x(), 1, 7168
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=91_480, device=device)
    previous_post, previous_comb, _, _, _, _ = _reference(residual, x, fn, scale, bias, None, rms_eps=1e-6)
    declaration = mhc.plan(mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=112), invocation=_invocation(norm=False))

    def prepare(state):
        return _bound_call(state, x=x, residual=residual, previous_post=previous_post, previous_comb=previous_comb, fn=fn, scale=scale, bias=bias, norm=None, rms_eps=1e-6)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="high-slot", prepare_call=prepare),))
        plan = declaration
        state = require_prepared(plan, "norm.mhc")
        scratch = torch.empty(state.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(plan, scratch=scratch, tokens=tokens, out=torch.empty_like(residual), post=torch.empty_like(previous_post), comb=torch.empty_like(previous_comb), y=torch.empty_like(x))
        def run():
            return mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, binding=binding)
        run()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture():
                with torch.cuda.graph(graph):
                    outputs = run()
            pointers = tuple(output.data_ptr() for output in outputs)
            x.mul_(-0.5).add_(0.01171875)
            residual.mul_(0.625).sub_(0.01953125)
            expected_carry = post_reference(x, residual, previous_post, previous_comb)
            expected_y, expected_post, expected_comb = pre_reference(expected_carry, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert tuple(output.data_ptr() for output in outputs) == pointers
            torch.testing.assert_close(outputs[0], expected_carry, rtol=0.0, atol=2e-2)
            torch.testing.assert_close(outputs[3], expected_y, rtol=0.0, atol=8e-3)
            torch.testing.assert_close(outputs[1], expected_post, rtol=2e-6, atol=1e-5)
            torch.testing.assert_close(outputs[2], expected_comb, rtol=2e-6, atol=1e-5)
        finally:
            graph.reset()


@pytest.mark.parametrize("tokens", [3, 129])
def test_mhc_prefill_capacity_reuses_launchers_and_scratch(tokens):
    from b12x._lib.runtime_control import kernel_resolution_guard

    device, hidden, capacity = require_b12x(), 4096, 4096
    residual, x, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=hidden, seed=193 + tokens, device=device)
    norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    previous_post, previous_comb, _, _, _, _ = _reference(residual, x, fn, scale, bias, norm, rms_eps=1e-6)
    declaration = mhc.plan(
        mhc.Caps(device=device, max_tokens=capacity, hidden_size=hidden, split_k=64),
        invocation=_invocation(norm=True),
    )
    def prepare(state):
        return _bound_call(state, x=x, residual=residual, previous_post=previous_post, previous_comb=previous_comb, fn=fn, scale=scale, bias=bias, norm=norm, rms_eps=1e-6)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(name="prefill-capacity", prepare_call=prepare),))
        exact = mhc.plan(
            mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=64),
            invocation=_invocation(norm=True), override=declaration.selection.config,
        )
        session.prepare((exact.request(name="exact-configured-execution", prepare_call=prepare),))
        reference = prepare(require_prepared(exact, "norm.mhc"))
        expected = tuple(value.clone() for value in reference.invoke())
        scratch = torch.empty(declaration.scratch_specs()[0].shape, dtype=torch.uint8, device=device)
        binding = mhc.bind(declaration, scratch=scratch, tokens=tokens, out=torch.empty_like(residual), post=torch.empty_like(previous_post), comb=torch.empty_like(previous_comb), y=torch.empty_like(x))
        pointers = (scratch.data_ptr(), binding.partials.data_ptr(), binding.out.data_ptr(), binding.y.data_ptr())
        def run():
            return mhc.run_post_pre(x, residual, previous_post, previous_comb, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=norm, norm_eps=1e-6, binding=binding)
        session.freeze()
        with kernel_resolution_guard("mHC prefill capacity"):
            actual = run()
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    captured = run()
                x.mul_(-0.5)
                graph.replay()
                torch.cuda.synchronize()
                replayed = tuple(value.clone() for value in captured)
                eager = reference.invoke()
                for got, want in zip(replayed, eager, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                assert (scratch.data_ptr(), binding.partials.data_ptr(), binding.out.data_ptr(), binding.y.data_ptr()) == pointers
            finally:
                graph.reset()


from b12x.norm.mhc._impl import (
    _is_default_mhc_epsilon, _is_supported_mhc_rms_epsilon, b12x_mhc_head,
)


def test_default_mhc_epsilon_accepts_f32_abi_round_trip() -> None:
    f32_epsilon = float(torch.tensor(1.0e-6, dtype=torch.float32).item())

    assert _is_default_mhc_epsilon(1.0e-6)
    assert _is_default_mhc_epsilon(f32_epsilon)
    assert not _is_default_mhc_epsilon(1.1e-6)


def test_glm_mhc_rms_epsilon_is_supported() -> None:
    glm_f32_epsilon = float(torch.tensor(1.0e-5, dtype=torch.float32).item())

    assert _is_supported_mhc_rms_epsilon(1.0e-6)
    assert _is_supported_mhc_rms_epsilon(1.0e-5)
    assert _is_supported_mhc_rms_epsilon(glm_f32_epsilon)
    assert not _is_supported_mhc_rms_epsilon(1.1e-5)


def _mhc_head_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes * scale + bias) + hc_eps
    collapsed = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    collapsed_f32 = collapsed.float()
    return (
        collapsed_f32
        * torch.rsqrt(collapsed_f32.square().mean(dim=-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).to(residual.dtype)


def _mhc_head_collapsed_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes * scale + bias) + hc_eps
    return (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)


@pytest.mark.parametrize(
    ("tokens", "hidden_size", "seed"),
    [(1, 4096, 92_001), (16, 4096, 92_016), (1, 7168, 92_112), (1, 5120, 92_120)],
)
def test_b12x_mhc_head_matches_reference_and_capture(
    tokens: int,
    hidden_size: int,
    seed: int,
) -> None:
    device = require_b12x()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn(
            (tokens, 4, hidden_size), generator=gen, dtype=torch.float32
        ).to(device)
        / 3
    ).to(torch.bfloat16).contiguous()
    fn = (
        torch.randn(
            (4, 4 * hidden_size), generator=gen, dtype=torch.float32
        ).to(device)
        / 64
    ).contiguous()
    scale = (
        torch.randn((1,), generator=gen, dtype=torch.float32).to(device) / 3
    ).contiguous()
    bias = (
        torch.randn((4,), generator=gen, dtype=torch.float32).to(device) / 5
    ).contiguous()
    norm_weight = (
        1
        + torch.randn((hidden_size,), generator=gen, dtype=torch.float32).to(device)
        / 8
    ).to(torch.bfloat16).contiguous()
    rms_eps = 1e-6
    hc_eps = 1e-6
    norm_eps = 1e-6
    output = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device=device)
    collapsed = torch.empty(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    )

    actual = b12x_mhc_head(
        residual,
        fn,
        scale,
        bias,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        collapsed_out=collapsed,
        out=output,
    )
    expected = _mhc_head_reference(
        residual,
        fn,
        scale,
        bias,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )
    assert actual.data_ptr() == output.data_ptr()
    expected_collapsed = _mhc_head_collapsed_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
    )
    torch.testing.assert_close(collapsed, expected_collapsed, atol=1.6e-2, rtol=0.0)
    torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=0.0)

    functional = b12x_mhc_head(
        residual,
        fn,
        scale,
        bias,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )
    torch.testing.assert_close(functional, expected, atol=1.6e-2, rtol=0.0)

    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = b12x_mhc_head(
            residual,
            fn,
            scale,
            bias,
            norm_weight,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            collapsed_out=collapsed,
            out=output,
        )
    captured_ptr = captured.data_ptr()
    baseline = captured.clone()
    collapsed_baseline = collapsed.clone()
    graph.replay()
    assert captured.data_ptr() == captured_ptr
    torch.testing.assert_close(captured, baseline, atol=0.0, rtol=0.0)
    torch.testing.assert_close(collapsed, collapsed_baseline, atol=0.0, rtol=0.0)

    residual.add_(0.125)
    graph.replay()
    expected_live = _mhc_head_reference(
        residual,
        fn,
        scale,
        bias,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
    )
    expected_collapsed_live = _mhc_head_collapsed_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
    )
    assert torch.isfinite(captured).all()
    torch.testing.assert_close(
        collapsed, expected_collapsed_live, atol=1.6e-2, rtol=0.0
    )
    torch.testing.assert_close(captured, expected_live, atol=1.6e-2, rtol=0.0)

    graph.reset()


def test_b12x_mhc_head_torch_compile_functional() -> None:
    device = require_b12x()
    hidden_size = 4096
    gen = torch.Generator(device="cpu")
    gen.manual_seed(92_101)
    residual = torch.randn(
        (1, 4, hidden_size), generator=gen, dtype=torch.float32
    ).to(device).to(torch.bfloat16).contiguous()
    fn = (
        torch.randn(
            (4, 4 * hidden_size), generator=gen, dtype=torch.float32
        ).to(device)
        / 64
    ).contiguous()
    scale = torch.ones((1,), dtype=torch.float32, device=device)
    bias = torch.zeros((4,), dtype=torch.float32, device=device)
    norm_weight = torch.ones((hidden_size,), dtype=torch.bfloat16, device=device)

    def head(
        residual_arg: torch.Tensor,
        fn_arg: torch.Tensor,
        scale_arg: torch.Tensor,
        bias_arg: torch.Tensor,
        norm_weight_arg: torch.Tensor,
    ) -> torch.Tensor:
        return b12x_mhc_head(
            residual_arg,
            fn_arg,
            scale_arg,
            bias_arg,
            norm_weight_arg,
            rms_eps=1e-6,
            hc_eps=1e-6,
            norm_eps=1e-6,
        )

    eager = head(residual, fn, scale, bias, norm_weight)
    compiled = torch.compile(head, fullgraph=True)(
        residual,
        fn,
        scale,
        bias,
        norm_weight,
    )
    torch.testing.assert_close(compiled, eager, atol=0.0, rtol=0.0)
