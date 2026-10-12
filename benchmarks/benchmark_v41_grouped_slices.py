"""Compare grouped V4.1 slice widths on identical encoded inputs and weights.

Includes ordered CuTe FP32 slice reduction. Uses the GPU oracle fixture before
sampling. This is a synchronous-kernel diagnostic, not a serving benchmark.
Run with the repository root on PYTHONPATH; the tests package supplies fixtures.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess

import torch
import cutlass
import cutlass.cute as cute
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


def reference_routes_fp64(x, ids, routing, weights, scales):
    """Independent FP64 projections with the checkpoint's BF16/A8 boundaries."""
    from tests.moe.test_v41_expert_numerics import quantized_rows

    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       device=x.device, dtype=torch.float64)

    def unpack(name, expert):
        codes = weights[name][expert]
        values = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1).flatten(-2)
        exponent = scales[name][expert].double() - 127
        return values * torch.exp2(exponent).repeat_interleave(32, -1)

    qx = quantized_rows(x).double()
    result = torch.zeros((*ids.shape, x.shape[1]), device=x.device, dtype=torch.float64)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = (qx[rows] @ unpack("w1", expert).T).bfloat16().double().clamp_max(10)
        up = (qx[rows] @ unpack("w3", expert).T).bfloat16().double().clamp(-10, 10)
        mid = (torch.nn.functional.silu(gate) * up * routing[rows, slots, None].double()).bfloat16()
        result[rows, slots] = (quantized_rows(mid).double() @ unpack("w2", expert).T).bfloat16().double()
    return result.reshape(-1, x.shape[1])


def benchmark_pipeline(options):
    """The exporter ABI's real RTX wire-input route/compute/reduce pipeline."""
    from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline
    from b12x.moe.fused_moe._impl import (
        _logical_weight_to_w4a8_rp_inplace, _e8m0_scale_to_w4a8_sfb_inplace,
    )

    h, experts, topk = 5120, 384, 6
    n = 2304 if options.role == "rtx_backbone" else 1152
    rows_list = [int(value) for value in options.rows.split(",")]
    widths = [int(value) for value in options.widths.split(",")]
    modes = options.routing.split(",")
    if (not rows_list or min(rows_list) < 1 or max(rows_list) > options.capacity
            or any(width not in (64, 128, 192) for width in widths)
            or any(mode not in ("random", "hot") for mode in modes) or options.iters < 1):
        raise ValueError("invalid rows, widths, routing or iterations")
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) == (12, 0), "RTX SM120 required"
    torch.manual_seed(4164)
    torch.backends.cuda.matmul.allow_tf32 = False
    weights, scales = {}, {}
    for name, shape in [("w1", (experts, n, h // 2)), ("w3", (experts, n, h // 2)),
                        ("w2", (experts, h, n // 2))]:
        weights[name] = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1] // 16),
                                    device="cuda", dtype=torch.uint8)
    w13 = _logical_weight_to_w4a8_rp_inplace(torch.cat([weights["w3"], weights["w1"]], 1),
                                           size_k=h, size_n=n * 2, gated_half_rows=n)
    s13 = _e8m0_scale_to_w4a8_sfb_inplace(torch.cat([scales["w3"], scales["w1"]], 1),
                                        weight_E=experts, rows=n * 2, k_dim=h, gated_half_rows=n)
    w2 = _logical_weight_to_w4a8_rp_inplace(weights["w2"].clone(), size_k=n, size_n=h)
    s2 = _e8m0_scale_to_w4a8_sfb_inplace(scales["w2"].clone(), weight_E=experts, rows=h, k_dim=n)
    packed_weights = [tensor.view(torch.uint32).flatten() for tensor in (w13, s13, w2, s2)]
    capacity, routes = options.capacity, options.capacity * topk
    x = torch.randn(capacity, h, device="cuda").mul_(.5).bfloat16()
    blocks = x.float().reshape(capacity, h // 32, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    wire = torch.empty((capacity, h + h // 32), device="cuda", dtype=torch.uint8)
    wire[:, :h].copy_((blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
                     .view(torch.uint8).reshape(capacity, h))
    wire[:, h:].copy_((exponent + 127).to(torch.uint8))
    ids = torch.full((capacity, topk), -1, device="cuda", dtype=torch.int32)
    routing = torch.rand(capacity, topk, device="cuda")
    routing.mul_(1.5 / routing.sum(-1, keepdim=True))
    live = torch.zeros(1, device="cuda", dtype=torch.int32)
    packed_routes = torch.empty(experts * routes, device="cuda", dtype=torch.int32)
    counts = torch.empty(experts, device="cuda", dtype=torch.int32)
    prefixes = torch.empty(experts, 2, device="cuda", dtype=torch.int32)
    metadata = torch.empty(routes, 19, device="cuda", dtype=torch.int32)
    grouped = torch.empty(routes, device="cuda")
    inverse = torch.empty(routes, device="cuda", dtype=torch.int32)
    output = torch.empty(routes, h, device="cuda")
    prefix = [wire[:, :h].view(torch.uint32), wire[:, h:], *packed_weights, ids.flatten(),
              routing.flatten(), live, packed_routes, counts, prefixes, metadata, grouped, inverse]
    programs = {}
    for width in widths:
        partial = torch.empty((n // width, routes, h), device="cuda")
        tensors = prefix + [partial, output]
        args = [from_dlpack(tensor, assumed_align=16) for tensor in tensors]
        compiled = cute.compile(V41SlicePipeline(capacity, width, experts=experts,
                                topk=topk, intermediate=n), *args, cutlass.Int32(capacity),
                                current_cuda_stream())
        programs[width] = (compiled, args, partial)
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    records = []
    for rows in rows_list:
        for mode in modes:
            ids.fill_(-1)
            if mode == "hot":
                ids[:rows].copy_(torch.randperm(experts, device="cuda")[:topk].int().expand(rows, topk))
            else:
                ids[:rows].copy_(torch.rand(rows, experts, device="cuda").topk(topk, -1).indices.int())
            active_ids = ids[:rows]
            expert_counts = torch.bincount(active_ids.flatten().long(), minlength=experts)
            unique = int((expert_counts > 0).sum())
            groups = int(((expert_counts + 15) // 16).sum())
            launched_groups = max(1, min(rows * topk, experts + max(rows * topk - experts, 0) // 16))
            expected = reference_routes_fp64(x[:rows], active_ids, routing[:rows], weights, scales) if options.check else None
            graphs, correctness = {}, {}
            for width, (compiled, args, partial) in programs.items():
                def run(compiled=compiled, args=args):
                    compiled(*args, rows, current_cuda_stream())
                output.fill_(12345)
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                actual = output[:rows * topk].bfloat16().double()
                assert bool(torch.isfinite(actual).all()) and bool(actual.count_nonzero())
                assert bool((output[rows * topk:] == 12345).all()), "wrote past live routes"
                if expected is not None:
                    rel = float((actual - expected).norm() / expected.norm())
                    cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
                    correctness[width] = dict(relative_l2=rel, cosine=cosine, oracle="FP64 projections/BF16 boundaries")
                    assert rel < .01 and cosine > .9999, (options.role, rows, mode, width, correctness[width])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                before = torch.cuda.memory_allocated()
                graph.replay()
                assert before == torch.cuda.memory_allocated(), "replay allocated"
                graphs[width] = graph
            samples = {width: [] for width in widths}
            for iteration in range(options.iters):
                order = widths if iteration % 2 == 0 else widths[::-1]
                for width in order:
                    if not options.l2_warm:
                        flush.max()
                    torch.cuda._sleep(5_000_000)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[width].replay()
                    end.record()
                    end.synchronize()
                    samples[width].append(start.elapsed_time(end) * 1000)
            for width, graph in graphs.items():
                weight_bytes = unique * 3 * h * n * 17 // 32
                us = statistics.median(samples[width])
                record = dict(role=options.role, rows=rows, routing=mode, width=width,
                              live_rows_per_expert=expert_counts[expert_counts > 0].tolist(),
                              unique_experts=unique, groups=groups, active_ctas=groups * (n // width),
                              launched_ctas=launched_groups * (n // width), sm_count=props.multi_processor_count,
                              graph=True, l2_cold=not options.l2_warm, us=us, samples_us=samples[width],
                              weight_bytes=weight_bytes, gbps=weight_bytes / us / 1e3,
                              theoretical_bandwidth_gbps=options.bandwidth_gbps,
                              weight_roofline_us=weight_bytes / options.bandwidth_gbps / 1e3,
                              correctness=correctness.get(width, {"checked": False}),
                              scratch_bytes=sum(t.numel() * t.element_size() for t in prefix[8:] + [programs[width][2], output]),
                              scope="wire-input route planning, fused MXFP4 compute and ordered FP32 route reduction; excludes wire quantization, shared expert and transport")
                if options.profile_dir:
                    options.profile_dir.mkdir(parents=True, exist_ok=True)
                    flush.max()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        graph.replay()
                        torch.cuda.synchronize()
                    trace = options.profile_dir / f"{options.role}-{mode}-r{rows}-w{width}.json"
                    prof.export_chrome_trace(str(trace))
                    kernels = [event for event in json.loads(trace.read_text())["traceEvents"]
                               if event.get("cat") == "kernel" and "dur" in event]
                    record["kernels"] = [dict(name=event["name"], us=event["dur"],
                                              grid=event.get("args", {}).get("grid"),
                                              block=event.get("args", {}).get("block")) for event in kernels]
                records.append(record)
                print("TIMING " + json.dumps(record), flush=True)
                graph.reset()
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        options.output.write_text(json.dumps(dict(source=revision, device=str(props), torch=torch.__version__,
                                                results=records), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("legacy", "rtx_backbone", "rtx_tp2"), default="legacy")
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--widths", default="192,128,64")
    parser.add_argument("--routing", default="random,hot")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--l2-warm", action="store_true")
    parser.add_argument("--bandwidth-gbps", type=float, default=1792, help="RTX PRO 6000 nominal weight roofline")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    print("SOURCE " + json.dumps({name: hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest()
                                 for name, obj in [("kernel", V41FusedSliceKernel),
                                                   ("fixture", _check_grouped_slices), ("benchmark", main)]}), flush=True)
    if options.role == "legacy":
        _check_grouped_slices(128, measure)
    else:
        benchmark_pipeline(options)


if __name__ == "__main__":
    main()
