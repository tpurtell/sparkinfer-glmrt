"""Whole-expert top-k sum with Muse Spark's per-route post-expert norm.

The kernel is bit-exact against its host model (`route_norm_sum.reference`),
and the host model follows the official module order: expert output (BF16),
input-scaled RMSNorm (scale, round, normalize, round), routing weight (round),
sum. Unowned routes (expert_map -1) contribute nothing.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from b12x.moe._shared.kernels.w4a16.route_norm_sum import (  # noqa: E402
    RouteNormTopKSumKernel,
    compile_route_norm_topk_sum,
    reference,
    route_norm,
)

HIDDEN = 4096
EPS = 1e-8


def _had128(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Sylvester H128 on the last axis (the kernel's transform)."""
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return x.reshape(*x.shape[:-1], -1, 128) @ (h / 128 ** 0.5)


def _case(m: int, topk: int, experts: int, owned: list[int], seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:topk] for _ in range(m)]).to(torch.int32)
    weights = torch.rand(m, topk, generator=g)
    weights = weights / weights.sum(-1, keepdim=True)
    expert_map = torch.full((experts,), -1, dtype=torch.int32)
    for local, expert in enumerate(owned):
        expert_map[expert] = local
    # Expert outputs of mixed magnitude, stored rotated by H128 with per-expert svh.
    svh = (torch.rand(len(owned), HIDDEN, generator=g) * 0.5 + 0.75).to(torch.float16)
    raw = torch.randn(m * topk, HIDDEN, generator=g) * torch.logspace(-2, 1, m * topk)[:, None]
    fc2 = _had128(raw).reshape(m * topk, HIDDEN).to(torch.float16)
    norm_weight = (torch.randn(HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
    return fc2, weights, ids, expert_map, svh, norm_weight


def test_host_model_follows_the_official_module_order():
    """Given the same BF16 expert output, the norm stage equals the official
    MuseSparkTextInputScaledRMSNorm then the BF16 routing-weight product, up
    to rsqrt vs 1/sqrt (at most one BF16 ulp, on a few columns)."""
    import numpy as np
    g = torch.Generator().manual_seed(3)
    norm_weight = (torch.randn(HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
    scale = (1.0 + norm_weight).float().numpy().astype(np.float32)
    for magnitude in (1e-3, 1.0, 40.0):
        y = (torch.randn(HIDDEN, generator=g) * magnitude).to(torch.bfloat16)
        weight = float(torch.rand(1, generator=g))
        got = torch.from_numpy(route_norm(y.float().numpy(), scale, EPS, np.float32(weight)))
        out = (y.float() * (1.0 + norm_weight).float()).to(torch.bfloat16).float()
        out = (out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + EPS)).to(torch.bfloat16)
        expected = (out.float() * weight).to(torch.bfloat16).float()
        ulp = torch.exp2(torch.floor(torch.log2(expected.abs().clamp_min(1e-30))) - 7)
        assert ((got - expected).abs() <= ulp).all(), magnitude
        assert (got != expected).float().mean() < 0.05, magnitude


def test_unrotation_is_the_orthonormal_sylvester_h128():
    """The host model's butterfly (the kernel's) is Sylvester H128 / sqrt(128)."""
    import numpy as np
    g = torch.Generator().manual_seed(4)
    expert = (torch.randn(HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
    rotated = _had128(expert.float()).reshape(1, HIDDEN).to(torch.float16)
    ids = torch.zeros(1, 1, dtype=torch.int32)
    weights = torch.ones(1, 1)
    norm_weight = torch.zeros(HIDDEN, dtype=torch.bfloat16)
    svh = torch.ones(1, HIDDEN, dtype=torch.float16)
    got = reference(rotated, weights, ids, torch.zeros(1, dtype=torch.int32), svh, norm_weight, EPS)
    direct = torch.from_numpy(route_norm(_had128(rotated.float()).reshape(HIDDEN).to(torch.bfloat16).float().numpy(),
        np.ones(HIDDEN, dtype=np.float32), EPS, np.float32(1.0)))
    ulp = torch.exp2(torch.floor(torch.log2(direct.abs().clamp_min(1e-30))) - 7)
    assert ((got[0] - direct).abs() <= ulp).all()


def test_unowned_routes_contribute_nothing():
    fc2, weights, ids, expert_map, svh, norm_weight = _case(2, 8, 16, [], 2)
    expert_map[:] = -1
    svh = svh[:0]
    assert torch.count_nonzero(reference(fc2, weights, ids, expert_map, torch.zeros(1, HIDDEN, dtype=torch.float16),
                                         norm_weight, EPS)) == 0


def test_kernel_rejects_untiled_shapes():
    with pytest.raises(ValueError):
        RouteNormTopKSumKernel(topk=8, hidden_size=2048, eps=EPS)
    with pytest.raises(ValueError):
        RouteNormTopKSumKernel(topk=17, hidden_size=HIDDEN, eps=EPS)
    with pytest.raises(ValueError):
        RouteNormTopKSumKernel(topk=8, hidden_size=HIDDEN, eps=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("output_dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("m,owned", [(1, list(range(0, 256, 6))), (5, list(range(1, 256, 6))), (37, list(range(256)))])
def test_kernel_is_bit_exact_against_the_host_model(output_dtype, m, owned):
    from cutlass.cute.runtime import from_dlpack  # noqa: F401
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32
    from b12x._lib.utils import current_cuda_stream, make_ptr

    fc2, weights, ids, expert_map, svh, norm_weight = _case(m, 8, 256, owned, 10 + m)
    expected = reference(fc2, weights, ids, expert_map, svh, norm_weight, EPS, output_dtype)
    launch = compile_route_norm_topk_sum(topk=8, hidden_size=HIDDEN, eps=EPS, output_dtype=output_dtype)
    dev = torch.device("cuda")
    tensors = [t.to(dev).contiguous() for t in (fc2, weights, ids, expert_map, svh, norm_weight)]
    out = torch.full((m, HIDDEN), float("nan"), device=dev,
                     dtype=torch.bfloat16 if output_dtype == "bf16" else torch.float32)

    def ptr(t, dtype, align):
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)
    fc2_d, w_d, ids_d, map_d, svh_d, norm_d = tensors
    launch.compiled(ptr(fc2_d, cutlass.Float16, 16),
                    ptr(out, cutlass.BFloat16 if output_dtype == "bf16" else cutlass.Float32, 16),
                    ptr(w_d, cutlass.Float32, 4), ptr(ids_d, cutlass.Int32, 4), ptr(map_d, cutlass.Int32, 4),
                    ptr(svh_d, cutlass.Float16, 16), ptr(norm_d, cutlass.BFloat16, 16),
                    Int32(len(owned)), Int32(256), Int32(m), current_cuda_stream())
    torch.cuda.synchronize()
    got = out.cpu()
    assert torch.equal(got.view(torch.int16 if output_dtype == "bf16" else torch.int32),
                       expected.view(torch.int16 if output_dtype == "bf16" else torch.int32)), \
        (got.float() - expected.float()).abs().max()
