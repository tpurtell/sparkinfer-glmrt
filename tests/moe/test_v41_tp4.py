"""Four native intermediate shards reduced at the global expert boundary."""

import pytest
import torch
from tests.conftest import require_b12x
from tests._reference.helpers import prepare_tp_moe_fp4_experts, make_tp_moe_fp4_binding
from tests.moe.test_v41_expert_numerics import reference


@pytest.mark.parametrize("m", [1, 16, 80])
def test_v41_four_shards_match_full_experts(m):
    require_b12x()
    from b12x.moe import fused_moe

    torch.manual_seed(4180 + m)
    e, h, n, topk = 8, 5120, 2304, 6
    x = (torch.randn(m, h, device="cuda") * 0.5).bfloat16()
    ids = torch.rand(m, e, device="cuda").topk(topk, -1).indices.int()
    routing = torch.rand(m, topk, device="cuda")
    routing = routing / routing.sum(-1, keepdim=True) * 1.5
    shared = torch.randn_like(x)
    weights, scales = {}, {}
    for name, shape in [
        ("w1", (e, n, h // 2)),
        ("w3", (e, n, h // 2)),
        ("w2", (e, h, n // 2)),
    ]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
        scales[name] = torch.randint(
            121, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8, device="cuda"
        )
    expected = (reference(x, ids, routing, weights, scales) + shared.float()).bfloat16()
    changed_ids = (ids + 1) % e
    changed = (
        reference(x * -0.5, changed_ids, routing, weights, scales) - shared.float()
    ).bfloat16()
    bindings = []
    ones = torch.ones(e, device="cuda", dtype=torch.float32)
    for rank in range(4):
        start, stop = rank * 576, (rank + 1) * 576
        prepared = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=ones,
            w1_fp4=torch.cat(
                (weights["w3"][:, start:stop], weights["w1"][:, start:stop]), 1
            ),
            w1_blockscale=torch.cat(
                (scales["w3"][:, start:stop], scales["w1"][:, start:stop]), 1
            ),
            w1_alphas=ones,
            a2_gscale=ones,
            w2_fp4=weights["w2"][:, :, start // 2 : stop // 2].contiguous(),
            w2_blockscale=scales["w2"][:, :, start // 32 : stop // 32].contiguous(),
            w2_alphas=ones,
            quant_mode="w4a8_mx",
            source_format="fp4_e8m0_k32",
            activation="silu_v41",
        )
        bindings.append(
            make_tp_moe_fp4_binding(
                a=x,
                experts=prepared,
                topk_weights=routing,
                topk_ids=ids,
                quant_mode="w4a8_mx",
                swiglu_limit=10,
            )
        )
    output = torch.empty_like(x)

    def run():
        planes = [binding.run_route_partials() for binding in bindings]
        assert all(p.shape == (m, topk, h) and p.dtype == torch.float32 for p in planes)
        return fused_moe.reduce_v41_tp4_routes(planes, output=output, shared=shared)

    def check(wanted):
        actual = output.float()
        assert torch.isfinite(actual).all() and actual.norm() > 0
        relative = (actual - wanted.float()).norm() / wanted.float().norm()
        assert relative < 0.01, relative.item()
        assert (
            torch.nn.functional.cosine_similarity(
                actual.flatten(), wanted.float().flatten(), dim=0
            )
            > 0.9999
        )

    for _ in range(3):
        run()
    check(expected)
    before_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    x.mul_(-0.5)
    ids.copy_(changed_ids)
    shared.neg_()
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    assert torch.cuda.memory_allocated() == allocated
    check(changed)
