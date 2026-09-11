"""Device-side live count, stable grouping and stale-capacity replay checks."""

import pytest
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.v41_route_plan import V41RoutePlan


@pytest.mark.parametrize("capacity", [1, 16, 80, 4096])
def test_route_plan(capacity):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    r = capacity * 6
    specs = [
        (r, torch.int32),
        (r, torch.float32),
        (1, torch.int32),
        (384 * r, torch.int32),
        (384, torch.int32),
        ((384, 2), torch.int32),
        ((r, 19), torch.int32),
        (r, torch.float32),
        (r, torch.int32),
    ]
    tensors = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in specs]
    ids, weights, live, packed, counts, prefixes, meta, grouped, inverse = tensors
    ids.zero_()
    weights.fill_(1)
    live.fill_(capacity)
    args = [from_dlpack(t) for t in tensors]
    compiled = cute.compile(V41RoutePlan(capacity), *args, current_cuda_stream())
    compiled(*args, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, current_cuda_stream())
    pointers = [t.data_ptr() for t in tensors]
    generator = torch.Generator().manual_seed(41)
    for case, rows in enumerate(
        [
            capacity,
            1,
            0,
            min(2, capacity),
            min(6, capacity),
            min(16, capacity),
            capacity,
            1,
        ]
    ):
        source = torch.randint(-2, 386, (r,), generator=generator, dtype=torch.int32)
        if case == 0:
            source.fill_(383)  # duplicates and long runs, even within one row
        values = torch.randn(r, generator=generator)
        ids.copy_(source)
        weights.copy_(values)
        live.fill_(rows)
        meta.fill_(9876)
        inverse.fill_(9876)
        grouped.fill_(float("nan"))
        graph.replay()
        tasks, permutation = [], []
        for expert in range(384):
            locations = (source[: rows * 6] == expert).nonzero().flatten().tolist()
            for start in range(0, len(locations), 16):
                chunk = locations[start : start + 16]
                tasks.append(
                    [expert, len(chunk), len(permutation)]
                    + [p // 6 for p in chunk]
                    + [-1] * (16 - len(chunk))
                )
                permutation.extend(chunk)
        n = len(permutation)
        expected_inverse = torch.full((r,), -1, dtype=torch.int32)
        expected_inverse[permutation] = torch.arange(n, dtype=torch.int32)
        assert torch.equal(inverse.cpu(), expected_inverse)
        assert torch.equal(grouped[:n].cpu(), values[permutation])
        if tasks:
            assert torch.equal(
                meta[: len(tasks)].cpu(), torch.tensor(tasks, dtype=torch.int32)
            )
        assert (meta[len(tasks) :, 1] == 0).all()
        assert [t.data_ptr() for t in tensors] == pointers
