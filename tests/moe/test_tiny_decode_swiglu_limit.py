"""Prepared tiny decode must preserve the declared SwiGLU clamp."""
import pytest
import torch

from tests._reference.helpers import prepare_tp_moe_fp4_experts, make_tp_moe_fp4_binding


@pytest.mark.parametrize('n', [1152, 1184])
@pytest.mark.parametrize('limit', [None, 10.0, 20.0])
def test_tiny_decode_clamp_and_graph(n, limit):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('Blackwell GPU required')
    e, h, topk = 8, 5120, 6
    x = torch.full((1, h), .5, device='cuda', dtype=torch.bfloat16)
    ids = torch.arange(topk, device='cuda', dtype=torch.int32).reshape(1, topk)
    routing = torch.ones((1, topk), device='cuda')
    ones = torch.ones(e, device='cuda')
    # Exact FP4 value 1, gate/up weights 1/64, down weights 1/256.
    prepared = prepare_tp_moe_fp4_experts(
        a=x, a1_gscale=ones, a2_gscale=ones, w1_alphas=ones, w2_alphas=ones,
        w1_fp4=torch.full((e, 2*n, h//2), 0x22, device='cuda', dtype=torch.uint8),
        w1_blockscale=torch.full((e, 2*n, h//32), 121, device='cuda', dtype=torch.uint8),
        w2_fp4=torch.full((e, h, n//2), 0x22, device='cuda', dtype=torch.uint8),
        w2_blockscale=torch.full((e, h, n//32), 119, device='cuda', dtype=torch.uint8),
        quant_mode='w4a8_mx', activation='silu', swiglu_limit=limit,
    )
    output = torch.empty_like(x)
    with make_tp_moe_fp4_binding(
        a=x, experts=prepared, topk_ids=ids, topk_weights=routing,
        output=output, quant_mode='w4a8_mx', swiglu_limit=limit, fast_math=False,
    ) as binding:
        assert binding.implementation == 'micro'
        binding.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): actual = binding.run()
        for value in [.5, .25, .5]:
            x.fill_(value); output.fill_(float('nan'))
            allocated = torch.cuda.memory_allocated()
            graph.replay(); torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            gate = torch.tensor(h*value/64)
            up = gate.clone()
            if limit is not None:
                gate = gate.clamp_max(limit); up = up.clamp(-limit, limit)
            mid = float((torch.nn.functional.silu(gate)*up).half())
            # Model the actual BF16 atomic contract exactly, including every
            # possible interleaving of equal full-tile and tail contributions.
            # A single FP32 sum is not its oracle: 54 additions of 50 yield
            # 2624 in BF16, rather than the mathematical sum 2700.
            kt = (n+127)//128
            task_width = 256 if kt%2==0 else 128
            full, tail = divmod(n, task_width)
            full *= topk
            tails = topk if tail else 0
            add_full = float(torch.tensor(mid*task_width/256).bfloat16())
            add_tail = float(torch.tensor(mid*tail/256).bfloat16())
            reachable = {(0, 0): {0.0}}
            for i in range(full+1):
                for j in range(tails+1):
                    if not (i or j): continue
                    values = set()
                    for previous, addend in (((i-1,j),add_full), ((i,j-1),add_tail)):
                        for value_before in reachable.get(previous, ()):
                            values.add(float(torch.tensor(value_before+addend).bfloat16()))
                    reachable[i,j] = values
            assert torch.isfinite(actual).all()
            assert set(actual.float().unique().cpu().tolist()) <= reachable[full,tails]

        graph.reset()
