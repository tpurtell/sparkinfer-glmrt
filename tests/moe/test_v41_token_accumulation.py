"""FP32 direct token accumulation versus ordered fused route/slice output."""
import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
from tests.moe.test_v41_grouped_slices import _check_grouped_slices


@pytest.mark.parametrize("width", [64, 128, 192])
def test_token_accumulation(width):
    state = {}

    def compare(**case):
        capacity, h, routes = 80, 5120, 480
        packed, wire, ids, weights = case["native_inputs"]
        if not state:
            tensors = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in [
                (routes, torch.int32), (routes, torch.float32), (1, torch.int32),
                (384 * routes, torch.int32), (384, torch.int32), ((384, 2), torch.int32),
                ((routes, 19), torch.int32), (routes, torch.float32),
                (routes, torch.int32), (1, torch.float32), (capacity * h, torch.float32),
            ]]
            args = [from_dlpack(t, assumed_align=16) for t in [
                wire[:, :h].view(torch.uint32), wire[:, h:], *packed, *tensors,
            ]]
            fn = cute.compile(V41SlicePipeline(capacity, width, atomic_tokens=True),
                              *args, cutlass.Int32(capacity), current_cuda_stream())
            state.update(tensors=tensors, args=args, fn=fn, addresses=[t.data_ptr() for t in tensors])
        tensors, args, fn = state["tensors"], state["args"], state["fn"]
        m = case["rows"]
        tensors[0].fill_(-1)
        tensors[0][:m * 6].copy_(ids.flatten())
        tensors[1].fill_(float("nan"))
        tensors[1][:m * 6].copy_(weights.flatten())
        partial = torch.zeros_like(case["out"][0, :m * 6])
        for plane in case["out"]:
            partial.add_(plane[:m * 6])
        ordered = torch.empty(m, 6, h, device="cuda")
        pair = case["pair_gpu"]
        ordered[pair[:, 0], pair[:, 1]] = partial
        expected = torch.zeros(m, h, device="cuda")
        for route in range(6):
            expected.add_(ordered[:, route])
        fn(*args, m, current_cuda_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*args, m, current_cuda_stream())
        for _ in range(5):
            tensors[-1].fill_(12345)
            before = torch.cuda.memory_allocated()
            graph.replay()
            assert torch.cuda.memory_allocated() == before
            actual = tensors[-1][:m * h].reshape(m, h)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
            assert (tensors[-1][m * h:] == 12345).all()
            assert [t.data_ptr() for t in tensors] == state["addresses"]
        graph.reset()
        tensors[-1].fill_(12345)
        fn(*args, 0, current_cuda_stream())
        assert (tensors[-1] == 12345).all()

    _check_grouped_slices(width, after_case=compare)
