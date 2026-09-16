"""Compact V4.1 routing, FP32 output, tails and frozen-capacity replay."""

import pytest
import torch

from tests._reference.helpers import prepare_tp_moe_fp4_experts
from tests.moe.test_v41_expert_numerics import quantized_rows


@pytest.mark.parametrize('n', [576, 1152])
def test_native_compact_boundaries_and_live_rows(n):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe._shared.kernels.w4a8_compact_micro import (
        _layout, launch_w4a8_compact_micro, micro_scratch_nbytes,
    )

    h, e, cap, topk = 5120, 8, 4, 6
    x = torch.full((cap, h), .5, device='cuda', dtype=torch.bfloat16)
    ids = torch.arange(topk, device='cuda', dtype=torch.int32).repeat(cap, 1)
    routing = torch.ones(cap, topk, device='cuda')
    ones = torch.ones(e, device='cuda')
    w13 = torch.full((e, 2*n, h//2), 0x22, device='cuda', dtype=torch.uint8)
    s13 = torch.full((e, 2*n, h//32), 121, device='cuda', dtype=torch.uint8)
    w2 = torch.full((e, h, n//2), 0x22, device='cuda', dtype=torch.uint8)
    s2 = torch.full((e, h, n//32), 119, device='cuda', dtype=torch.uint8)
    experts = prepare_tp_moe_fp4_experts(
        a=x, a1_gscale=ones, w1_fp4=w13, w1_blockscale=s13,
        w1_alphas=ones, a2_gscale=ones, w2_fp4=w2,
        w2_blockscale=s2, w2_alphas=ones, activation='silu',
        quant_mode='w4a8_mx', source_format='fp4_e8m0_k32', swiglu_limit=10)
    runtime = experts._impl.representation_for('w4a8_mx')
    scratch = torch.empty(micro_scratch_nbytes(cap, h, n, topk, True),
                          device='cuda', dtype=torch.uint8)
    offset, size = _layout(cap, topk, h, n, True)['route_output']
    all_routes = scratch[offset:offset+size].view(torch.float32).view(cap*topk, h)

    def run(rows):
        return launch_w4a8_compact_micro(
            scratch=scratch, a=x[:rows], topk_ids=ids[:rows],
            topk_weights=routing[:rows], w13=runtime.w13_rp,
            w13_scales=runtime.w13_sfb, w2=runtime.w2_rp,
            w2_scales=runtime.w2_sfb, alpha1=ones, alpha2=ones,
            input_scale=ones, down_scale=ones, max_tokens=cap,
            num_topk=topk, swiglu_limit=10, fast_math=False, native_v41=True)

    run(cap)
    torch.cuda.synchronize()
    with kernel_resolution_guard('compact native capacity must cover live rows'):
        for rows in (1, 3, 4):
            run(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run(rows)
            for value, weight, invalid in [(.5, .25, False), (.01, .125, False),
                                            (1e-10, 1., False), (.5, 1e-12, False),
                                            (.5, 1., True)]:
                x.fill_(value); routing.fill_(weight)
                ids.fill_(-1 if invalid else 2)
                # Independent constant-weight oracle, including native FP8 floor.
                gate = (quantized_rows(x[:rows]).sum(-1) / 64).bfloat16().float().clamp_max(10)
                up = (quantized_rows(x[:rows]).sum(-1) / 64).bfloat16().float().clamp(-10, 10)
                mid = (torch.nn.functional.silu(gate) * up * weight).bfloat16()
                expected = quantized_rows(mid[:, None].expand(rows, n).contiguous()).sum(-1) / 256
                if invalid: expected.zero_()
                all_routes.fill_(float('nan'))
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay(); torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
                assert actual.dtype == torch.float32
                torch.testing.assert_close(actual.view(rows, topk, h),
                    expected[:, None, None].expand(rows, topk, h), rtol=1e-6, atol=0)
                assert torch.isnan(all_routes[rows*topk:]).all()
            graph.reset()
