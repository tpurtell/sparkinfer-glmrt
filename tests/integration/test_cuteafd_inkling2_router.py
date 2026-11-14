"""#1915 routing ids exact, route weights and SiTU/gate F32 2e-6.

Tie rows, negative saturation, biased selection/unbiased weights, nonzero
route scale, graph replay mutations and uncapped-gate sigmoid are explicit.
"""
import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x


def test_router_oracle_and_replay():
    require_b12x()
    from b12x.integration.cuteafd.inkling2_router import compile_inkling2_router_aot
    p = compile_inkling2_router_aot(route_scale=2.5)
    for count in (1, 17, 2048, 2752, 4096):
        logits = torch.randn(count, 384, device="cuda") * 4
        logits[0] = -1000
        if count > 1:
            logits[1] = 1000
        bias = torch.zeros(384, device="cuda")
        ids = torch.empty(count, 8, device="cuda", dtype=torch.int32)
        weights = torch.empty(count, 8, device="cuda")
        def run():
            p.launch(logits, bias, ids, weights, scalars=(count,))
        run()
        expected_ids = (logits.sigmoid() + bias).argsort(dim=-1, descending=True, stable=True)[:, :8]
        assert torch.equal(ids, expected_ids)
        expected = F.logsigmoid(logits.gather(-1, expected_ids)).softmax(-1) * 2.5
        torch.testing.assert_close(weights, expected, atol=2e-6, rtol=2e-6)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        logits.mul_(.5)
        bias.copy_(torch.arange(384, device="cuda") * .001)
        graph.replay()
        expected_ids = (logits.sigmoid() + bias).argsort(dim=-1, descending=True, stable=True)[:, :8]
        assert torch.equal(ids, expected_ids)
        expected = F.logsigmoid(logits.gather(-1, expected_ids)).softmax(-1) * 2.5
        torch.testing.assert_close(weights, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("situ,width", [(False, 6144), (True, 2304)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_oracle(situ, width, dtype):
    require_b12x()
    from b12x.integration.cuteafd.inkling2_router import compile_inkling2_gate_aot
    p = compile_inkling2_gate_aot(width=width, situ=situ, dtype=dtype)
    for count in (1, 17, 2752):
        x = (torch.randn(count, width*2 if situ else width, device="cuda")*8).to(dtype)
        gate = (torch.randn(count, width, device="cuda")*4).to(dtype)
        gamma = torch.rand(count, device="cuda")*2
        out = torch.empty(count, width, device="cuda", dtype=dtype)
        p.launch(x, gate, gamma, out, scalars=(count,))
        if situ:
            g, u = x.float()[:, 0::2], x.float()[:, 1::2]
            want = ((4*torch.tanh(g/4))*g.sigmoid())*(25*torch.tanh(u/25))*gamma[:, None]
        else:
            want = x.float()*gate.float().sigmoid()
        torch.testing.assert_close(out.float(), want, atol=2e-6 if dtype==torch.float32 else .02,
                                   rtol=2e-6 if dtype==torch.float32 else .008)
