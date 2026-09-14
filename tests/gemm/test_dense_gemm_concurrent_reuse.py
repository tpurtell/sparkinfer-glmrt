"""Shared stages must remain readable until the last operand load completes."""
import pytest
import torch

from b12x.gemm._shared.block_fp8 import (
    BlockFP8LinearScratchCaps,
    pack_block_fp8_linear_weight_mxfp8,
    plan_block_fp8_linear_scratch,
)
from tests.conftest import require_b12x


@pytest.mark.parametrize("rows,capacity", [(1, 1), (3, 16)])
def test_dense_gemm_stage_reuse_under_concurrent_memory_traffic(rows, capacity):
    require_b12x()
    torch.manual_seed(41093)
    n, k, steps = 32768, 1280, 64
    source = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn((n, k), device="cuda") / 8).to(torch.float8_e4m3fn)
    exponents = (torch.arange(n // 32, device="cuda")[:, None]
                 + 3 * torch.arange(k // 32, device="cuda")[None, :]) % 5 + 124
    scales = exponents.to(torch.uint8).view(torch.float8_e8m0fnu)
    packed = pack_block_fp8_linear_weight_mxfp8(weight, scales, block_size=(32, 32))
    plan = plan_block_fp8_linear_scratch(BlockFP8LinearScratchCaps(
        device=source.device, max_tokens=capacity, in_features=k,
        out_features=n, block_size=(32, 32)))
    scratch = tuple(torch.empty(shape, dtype=dtype, device="cuda")
                    for shape, dtype in plan.shapes_and_dtypes())
    outputs = torch.empty((steps, rows, n, 1), device="cuda", dtype=torch.bfloat16)
    bindings = [plan.bind(scratch=scratch, source=source, packed_weight=packed,
                         output=outputs[i], expected_m=capacity,
                         activation_block_size=32) for i in range(steps)]
    traffic = torch.zeros(256 * 1024 * 1024 // 4, device="cuda")
    compute, competing = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(compute):
        bindings[0].run()
    compute.synchronize()
    expected = outputs[0].clone()
    assert torch.isfinite(expected).all() and expected.abs().sum() > 0
    torch.cuda.synchronize()

    target_graph, traffic_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(target_graph, stream=compute):
        for binding in bindings:
            binding.run()
    with torch.cuda.graph(traffic_graph, stream=competing):
        for _ in range(steps):
            traffic.add_(0.001)

    # Explicit replay streams are essential: default-stream replay serializes
    # the graphs and concealed the original last-K-block stage reuse defect.
    for _ in range(4):
        with torch.cuda.stream(compute):
            target_graph.replay()
        with torch.cuda.stream(competing):
            traffic_graph.replay()
        compute.synchronize()
        competing.synchronize()
        torch.testing.assert_close(outputs, expected.expand_as(outputs), rtol=0, atol=0)
