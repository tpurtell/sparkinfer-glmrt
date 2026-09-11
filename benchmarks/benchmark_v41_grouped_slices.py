"""Compare grouped V4.1 slice widths on identical encoded inputs and weights.

Includes ordered CuTe FP32 slice reduction. Uses the GPU oracle fixture before
sampling. This is a synchronous-kernel diagnostic, not a serving benchmark.
Run with the repository root on PYTHONPATH; the tests package supplies fixtures.
"""

import json, statistics, torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
from tests.moe.test_v41_grouped_slices import _check_grouped_slices
import cuda.bindings.driver as cuda


class OrderedReduce:
    def __init__(self, slices):
        self.slices = slices

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        routes: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(source, dest, routes).launch(
            grid=((routes * 5120 + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, dest: cute.Tensor, routes: cutlass.Int32):
        index = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cutlass.Int64(
            cute.arch.thread_idx()[0]
        )
        row = index // 5120
        col = index % 5120
        if row < routes:
            value = cutlass.Float32(0)
            for plane in cutlass.range_constexpr(self.slices):
                value += source[plane, row, col]
            dest[row, col] = value


state = {}


def measure(**case):
    args = case["args"]
    meta = case["meta_view"]
    m = case["rows"]
    groups = case["groups"]
    routes = case["routes"]
    if not state:
        for width in [64, 192]:
            out = torch.empty(((576 + width - 1) // width, 480, 5120), device="cuda")
            alt_args = list(args)
            alt_args[-1] = from_dlpack(out, assumed_align=16)
            compiled = cute.compile(
                V41FusedSliceKernel(width, grouped=True),
                *alt_args,
                cutlass.Int32(80),
                current_cuda_stream(),
                meta,
                cutlass.Int32(64),
            )
            state[width] = (out, alt_args, compiled)
        state[128] = (case["out"], args, case["baseline_compiled"])
        for width, (out, alt_args, compiled) in list(state.items()):
            reduced = torch.empty((480, 5120), device="cuda")
            rv = from_dlpack(reduced, assumed_align=16)
            reducer = cute.compile(
                OrderedReduce((576 + width - 1) // width),
                alt_args[-1],
                rv,
                cutlass.Int32(480),
                current_cuda_stream(),
            )
            state[width] = (out, alt_args, compiled, reduced, rv, reducer)
    graphs = {}
    for width, (out, alt_args, compiled, reduced, rv, reducer) in state.items():
        out.fill_(12345)
        compiled(*alt_args, m, current_cuda_stream(), meta, 64)
        reducer(alt_args[-1], rv, routes, current_cuda_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(*alt_args, m, current_cuda_stream(), meta, 64)
            reducer(alt_args[-1], rv, routes, current_cuda_stream())
        before = torch.cuda.memory_allocated()
        graph.replay()
        assert before == torch.cuda.memory_allocated()
        expected_reduced = torch.zeros_like(reduced[:routes])
        for plane in range(out.shape[0]):
            expected_reduced.add_(out[plane, :routes])
        assert torch.equal(reduced[:routes], expected_reduced)
        partial = reduced[:routes].bfloat16().float()
        actual = torch.zeros_like(case["expected"])
        actual.index_add_(0, case["pair_gpu"][:, 0], partial)
        rel = ((actual - case["expected"]).norm() / case["expected"].norm()).item()
        assert rel < 0.01 and bool((out[:, routes:] == 12345).all())
        graphs[width] = graph
    for _ in range(5):
        for graph in graphs.values():
            graph.replay()
    samples = {w: [] for w in [64, 128, 192]}
    for iteration in range(12):
        order = [64, 128, 192]
        order = order[iteration % 3 :] + order[: iteration % 3]
        if iteration % 2:
            order = order[::-1]
        for width in order:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                graphs[width].replay()
            end.record()
            end.synchronize()
            samples[width].append(start.elapsed_time(end) * 1000 / 20)
    print(
        "TIMING "
        + json.dumps(
            dict(
                case=case["case"],
                rows=m,
                groups=groups,
                launch_group_capacity=64,
                routes=routes,
                graph_us=samples,
                median_us={w: statistics.median(v) for w, v in samples.items()},
                scope="fused compute and ordered CuTe FP32 slice reduction; excludes route planning, encoding and transport; synchronous staging",
            )
        ),
        flush=True,
    )
    for graph in graphs.values():
        graph.reset()


if __name__ == "__main__":
    import hashlib, inspect
    from pathlib import Path

    print(
        "SOURCE "
        + json.dumps(
            {
                name: hashlib.sha256(
                    Path(inspect.getsourcefile(obj)).read_bytes()
                ).hexdigest()
                for name, obj in [
                    ("kernel", V41FusedSliceKernel),
                    ("fixture", _check_grouped_slices),
                    ("benchmark", measure),
                ]
            }
        ),
        flush=True,
    )
    _check_grouped_slices(128, measure)
