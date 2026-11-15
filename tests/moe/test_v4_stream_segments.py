"""Byte-exact native MXFP4 input gather / BF16 reply scatter and row curve."""
import json
import statistics

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.v4_stream_pipeline import (
    NativeTokenCombine, V4StreamPipeline, compile_v4_stream_segments_aot,
    stream_scratch_specs,
)
from tests.moe.test_v41_exact_slices import pack


@pytest.mark.parametrize("hidden,intermediate,coded", [(5120, 576, True), (4096, 512, False), (7168, 768, False)])
def test_native_segments_match_singletons_and_row_curve(hidden, intermediate, coded):
    torch.manual_seed(844)
    experts, topk, capacity = 384 if coded else 256, 6, 8192
    weights, scales = {}, {}
    for name, shape in (("w1", (experts, intermediate, hidden // 2)),
                        ("w3", (experts, intermediate, hidden // 2)),
                        ("w2", (experts, hidden, intermediate // 2))):
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
        scales[name] = torch.randint(119, 124, (*shape[:-1], shape[-1] // 16), dtype=torch.uint8)
    packed = pack(weights, scales, intermediate % 128 != 0)
    if coded:
        from b12x.moe._shared.scale_codes import pack_scale_codes
        packed[1] = pack_scale_codes(torch.cat((scales["w3"], scales["w1"]), 1),
            tensor_name="w13.scale", gated_half_rows=intermediate).view(torch.uint32).flatten().cuda()
        packed[3] = pack_scale_codes(scales["w2"], tensor_name="w2.scale").view(torch.uint32).flatten().cuda()
    del weights, scales
    pitch = hidden * 33 // 32
    wire = torch.randint(8, 100, (capacity, pitch), device="cuda", dtype=torch.uint8)
    wire[:, hidden:] = torch.randint(119, 124, (capacity, hidden // 32), device="cuda", dtype=torch.uint8)
    ids = torch.randint(experts, (capacity, topk), device="cuda", dtype=torch.int32)
    # Two shared hot experts exercise crossing a member boundary within a group.
    ids[:, :2] = torch.arange(2, device="cuda", dtype=torch.int32)
    routing = torch.rand((capacity, topk), device="cuda") / topk
    output = torch.full((capacity + 2, hidden), 11, dtype=torch.bfloat16, device="cuda")
    # Reverse the physical row layout to catch accidental contiguous accesses.
    physical = torch.arange(capacity - 1, -1, -1, device="cuda")
    resident = wire[physical].contiguous()
    input_rows = (resident.data_ptr() + physical * pitch).to(torch.int64)
    output_rows = (output.data_ptr() + (torch.arange(capacity, device="cuda") + 1) * hidden * 2).to(torch.int64)
    program = compile_v4_stream_segments_aot(hidden=hidden, intermediate=intermediate,
        experts=experts, topk=topk, scale_codes=coded)
    sizes = {"i32": torch.int32, "u8": torch.uint8, "f32": torch.float32}
    scratch = [torch.empty(shape, dtype=sizes[dtype], device="cuda") for _, dtype, shape in
        stream_scratch_specs(capacity, hidden, intermediate, experts, topk, row_addresses=True)]
    args = [input_rows, *packed, ids, routing, *scratch, output_rows]

    specs = stream_scratch_specs(4096, hidden, intermediate, experts, topk)
    ordinary_scratch = [torch.empty(shape, dtype=sizes[dtype], device="cuda") for _, dtype, shape in specs]
    ordinary = V4StreamPipeline(4096, hidden=hidden, intermediate=intermediate,
        experts=experts, topk=topk, scale_codes=coded)
    ptr = lambda t: from_dlpack(t, assumed_align=16).iterator
    ordinary_args = list(map(ptr, [wire, *packed, ids, routing, *ordinary_scratch]))
    compiled = cute.compile(ordinary, *ordinary_args, cutlass.Int32(4096), current_cuda_stream())
    expected = torch.empty((capacity, hidden), dtype=torch.bfloat16, device="cuda")
    combine = cute.compile(NativeTokenCombine(hidden, topk), ptr(ordinary_scratch[-1]), ptr(expected),
                           cutlass.Int32(4096), current_cuda_stream())

    def baseline(rows):
        for first in range(0, rows, 4096):
            count = min(4096, rows - first)
            pointers = list(map(ptr, [wire[first:], *packed, ids[first:], routing[first:], *ordinary_scratch]))
            compiled(*pointers, count, current_cuda_stream())
            combine(ptr(ordinary_scratch[-1]), ptr(expected[first:]), count, current_cuda_stream())

    def staged(rows):
        program.launch(*args, scalars=(rows,))

    for rows in (1, 333, 2048, 2752, 4096, 5504, 8192):
        baseline(rows)
        staged(rows)
        torch.cuda.synchronize()
        assert torch.equal(output[1:rows + 1].view(torch.int16), expected[:rows].view(torch.int16)), rows
        assert torch.isfinite(output[1:rows + 1]).all() and output[1:rows + 1].count_nonzero() > 0
        assert (output[0] == 11).all() and (output[-1] == 11).all()
        if rows >= 2048:
            measurements = {}
            for name, run in (("ordinary", baseline), ("segments", staged)):
                run(rows)
                samples = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    run(rows)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                measurements[name] = statistics.median(samples)
            print(json.dumps(dict(hidden=hidden, intermediate=intermediate, coded=coded,
                rows=rows, byte_identical=True, us=measurements,
                us_per_row={name: value / rows for name, value in measurements.items()})), flush=True)
    # A merged pair with a non-4096 boundary must equal separate ordinary units.
    baseline(2752)
    first = expected[:2752].clone()
    pointers = list(map(ptr, [wire[2752:], *packed, ids[2752:], routing[2752:], *ordinary_scratch]))
    compiled(*pointers, 2752, current_cuda_stream())
    combine(ptr(ordinary_scratch[-1]), ptr(expected[2752:]), 2752, current_cuda_stream())
    staged(5504)
    torch.cuda.synchronize()
    assert torch.equal(output[1:2753].view(torch.int16), first.view(torch.int16))
    assert torch.equal(output[2753:5505].view(torch.int16), expected[2752:5504].view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        staged(5504)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output[1:2753].view(torch.int16), first.view(torch.int16))
