"""Smoke tests for MX-FP6 fused MoE kernel setup (no GPU compile)."""

from types import SimpleNamespace


# TODO(port): the FP6 extensions of the fused-MoE backends (the static backend
# base `_MoEStaticKernelBase`, the micro backend's `MXFP6_BLOCK_SIZE` /
# `is_supported_mxfp6`) were not ported upstream; the b12x FP6 MoE route
# is b12x.moe.fused_moe with quant_mode="w6a8_mx" instead. The affected
# tests are kept as reference behind pytest.skip gates.

import cutlass
import pytest
import torch

from b12x._lib.utils import mxfp6_tile_k, mxfp6_num_k_blocks
from b12x.moe._shared.kernels import mxfp6_moe
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend


def test_mxfp6_moe_helpers_import():
    assert mxfp6_moe.moe_emit_mma_k_block is not None


def test_mxfp6_tile_k_matches_moe_expectation():
    assert mxfp6_tile_k() == 128
    assert mxfp6_num_k_blocks(128) == 4


def test_moe_static_dynamic_init_tile_k_default():
    pytest.skip("pending: FP6 static-backend port (_MoEStaticKernelBase not upstream)")
    from b12x.moe._shared.kernels.static import _MoEStaticKernelBase

    backend = _MoEStaticKernelBase(
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=1,
    )
    assert backend.tile_shape_mnk[2] == 256
    dyn = MoEDynamicKernelBackend(sf_vec_size=32, mma_tiler_mn=(128, 128))
    assert dyn.tile_shape_mnk[2] == 256


def test_micro_mxfp6_not_supported():
    pytest.skip("pending: FP6 micro kernel port (is_supported_mxfp6 not upstream)")
    from b12x.moe._shared.kernels.micro import (
        MXFP6_BLOCK_SIZE,
        MoEMicroKernelBackend,
    )

    assert MXFP6_BLOCK_SIZE == 32
    assert MoEMicroKernelBackend.is_supported_mxfp6(1, 128, 128, 8, 8) is False


def test_fp6_element_types_exist():
    assert cutlass.Float6E3M2FN is not None
    assert cutlass.Float6E2M3FN is not None

def test_fp6_preparation_declares_producing_calls_for_every_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod
    from b12x.moe import fused_moe

    captured: dict[str, object] = {}

    class _Declaration:
        token_counts = (1, 4)

        def request(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

    monkeypatch.setattr(fused_moe, "plan_execution", lambda **kwargs: _Declaration())
    prepared = SimpleNamespace(
        device=torch.device("cpu"),
        hidden_size=8,
        num_experts=4,
    )
    weight_plan = SimpleNamespace(
        geometry=SimpleNamespace(top_k=2),
        activation=SimpleNamespace(io_dtype=torch.bfloat16),
    )
    method = B12XFP6MoEMethod(prepared, weight_plan)
    (unit,) = method.get_b12x_preparation_units(
        object(),
        SimpleNamespace(
            stage="weights", token_counts=(1, 4), output_dtype=torch.bfloat16,
            eager_only=False,
        ),
    )
    (request,) = unit.requests

    assert set(captured["prepare_calls"]) == {1, 4}
    assert set(captured["benchmark_calls"]) == {1, 4}

    class _State:
        scratch = SimpleNamespace(scratch_specs=lambda: ())

        def __init__(self):
            self.bound = None

        def bind(self, **kwargs):
            self.bound = kwargs
            return SimpleNamespace(run=lambda: None)

    for count in (1, 4):
        state = _State()
        call = request.benchmark_calls[count](state)
        call.reset()
        call.produce()
        assert torch.count_nonzero(state.bound["a"])
        assert state.bound["topk_ids"].dtype is torch.int32
        assert torch.all(state.bound["topk_ids"] < prepared.num_experts)
        torch.testing.assert_close(
            state.bound["topk_weights"].sum(dim=-1),
            torch.ones(count, dtype=torch.float32),
        )
        call.restore()
