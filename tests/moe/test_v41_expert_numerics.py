"""Native V4.1 expert boundaries, routing placement and graph replay."""

import pytest
import torch
from tests.conftest import require_b12x
from tests._reference.helpers import prepare_tp_moe_fp4_experts, make_tp_moe_fp4_binding


def quantized_rows(x):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scale = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    return (
        (blocks / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
    ).reshape(x.shape)


def reference(x, ids, routing, weights, scales):
    """FP32 oracle for the official BF16 -> FP8 -> BF16 projection sequence."""
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=x.device
    )

    def weight(name, expert):
        codes = weights[name][expert]
        values = torch.stack(
            (lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1
        ).flatten(-2)
        return values * scales[name][expert].view(
            torch.float8_e8m0fnu
        ).float().repeat_interleave(32, -1)

    out = torch.zeros_like(x, dtype=torch.float32)
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        qx = quantized_rows(x)
        for expert in ids.unique().tolist():
            rows, slots = torch.where(ids == expert)
            gate = (qx[rows] @ weight("w1", expert).T).bfloat16().float().clamp_max(10)
            up = (qx[rows] @ weight("w3", expert).T).bfloat16().float().clamp(-10, 10)
            mid = (
                torch.nn.functional.silu(gate) * up * routing[rows, slots, None]
            ).bfloat16()
            partial = (quantized_rows(mid) @ weight("w2", expert).T).bfloat16().float()
            out[rows] += partial
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior
    return out


@pytest.mark.parametrize(
    "m,n,experts",
    [
        (1, 576, 8),
        (16, 576, 384),
        (80, 576, 8),
        (1, 2304, 8),
        (16, 2304, 128),
        (80, 2304, 8),
    ],
)
def test_v41_native_expert_routing_and_graph(m, n, experts):
    require_b12x()
    torch.manual_seed(4100 + m + n)
    h = 5120
    topk = 6 if n == 576 else 3
    x = (torch.randn(m, h, device="cuda") * 0.5).bfloat16()
    ids = torch.rand(m, experts, device="cuda").topk(topk, -1).indices.int()
    routing = torch.rand(m, topk, device="cuda")
    routing = (routing / routing.sum(-1, keepdim=True) * 1.5).float()
    weights, scales = {}, {}
    for name, shape in [
        ("w1", (experts, n, h // 2)),
        ("w3", (experts, n, h // 2)),
        ("w2", (experts, h, n // 2)),
    ]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
        scales[name] = torch.randint(
            121, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8, device="cuda"
        )
    expected = reference(x, ids, routing, weights, scales)
    changed_ids = (ids + 1) % experts
    expected_changed = reference(x * -0.5, changed_ids, routing, weights, scales)
    ones = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=ones,
        w1_fp4=torch.cat((weights["w3"], weights["w1"]), 1),
        w1_blockscale=torch.cat((scales["w3"], scales["w1"]), 1),
        w1_alphas=ones,
        a2_gscale=ones,
        w2_fp4=weights["w2"],
        w2_blockscale=scales["w2"],
        w2_alphas=ones,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu_v41",
    )
    binding = make_tp_moe_fp4_binding(
        a=x,
        experts=prepared,
        topk_weights=routing,
        topk_ids=ids,
        quant_mode="w4a8_mx",
        swiglu_limit=10,
        output=torch.empty_like(x),
    )

    def check(actual, wanted):
        actual = actual.float()
        assert torch.isfinite(actual).all() and actual.norm() > 0
        rel_l2 = (actual - wanted).norm() / wanted.norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), wanted.flatten(), dim=0
        )
        assert rel_l2 < 0.01, rel_l2.item()
        assert cosine > 0.9999, cosine.item()

    for _ in range(3):
        result = binding.run()
    check(result, expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = binding.run()
    x.mul_(-0.5)
    ids.copy_(changed_ids)
    before = torch.cuda.memory_allocated()
    graph.replay()
    assert torch.cuda.memory_allocated() == before
    check(result, expected_changed)


@pytest.mark.parametrize("m,n", [(1,576),(16,576),(80,576),(1,2304),(16,2304),(80,2304)])
def test_v41_expert_tiny_activation_floor_and_graph(m,n):
    """Isolate FC1 and FC2 floors with constant exact FP4 weights and changed inputs."""
    require_b12x()
    h,experts = 5120,8
    topk = 6 if n == 576 else 3
    x = torch.full((m,h),1e-10,device="cuda",dtype=torch.bfloat16)
    ids = torch.arange(topk,device="cuda",dtype=torch.int32).expand(m,topk).contiguous()
    routing = torch.ones((m,topk),device="cuda")
    weights,scales = {},{}
    for name,shape in [("w1",(experts,n,h//2)),("w3",(experts,n,h//2)),("w2",(experts,h,n//2))]:
        weights[name] = torch.full(shape,0x22,device="cuda",dtype=torch.uint8)
        scales[name] = torch.full((*shape[:-1],shape[-1]//16),121 if name == "w2" else 137,
                                  device="cuda",dtype=torch.uint8)
    ones = torch.ones(experts,device="cuda")
    prepared = prepare_tp_moe_fp4_experts(
        a=x,a1_gscale=ones,w1_fp4=torch.cat((weights["w3"],weights["w1"]),1),
        w1_blockscale=torch.cat((scales["w3"],scales["w1"]),1),w1_alphas=ones,
        a2_gscale=ones,w2_fp4=weights["w2"],w2_blockscale=scales["w2"],
        w2_alphas=ones,quant_mode="w4a8_mx",source_format="fp4_e8m0_k32",activation="silu_v41")
    binding = make_tp_moe_fp4_binding(a=x,experts=prepared,topk_weights=routing,topk_ids=ids,
        quant_mode="w4a8_mx",swiglu_limit=10,output=torch.empty_like(x))
    result = binding.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = binding.run()
    # Tiny input below E4M3's minimum after the floor; tiny routed intermediate;
    # nonzero tiny input amplified by weights; nonzero tiny routed intermediate.
    for value,route,zero in [(1e-10,1.0,True),(.0002,1e-12,True),(1e-8,1.0,False),(.0002,1e-7,False)]:
        x.fill_(value)
        routing.fill_(route)
        ids.add_(1).remainder_(experts)
        expected = reference(x,ids,routing,weights,scales).bfloat16()
        assert bool((expected == 0).all()) == zero
        graph.replay()
        assert torch.equal(result,expected), (m,n,value,route,(result.float()-expected.float()).abs().max().item())
    torch.cuda.synchronize()
    graph.reset()
