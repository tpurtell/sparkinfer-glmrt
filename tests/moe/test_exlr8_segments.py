"""Real EXLR8 checkpoint row-addressed merged waves versus ordinary units."""
import json
import os
import statistics

import pytest
import torch

from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
from benchmarks.benchmark_exlr8_a16 import compact_unit_descriptors
from tests.moe.exlr8_oracle import Case, make_wire
from tests.moe.test_exlr8_grouped import buffers, run


@pytest.mark.parametrize("activation", ["a16", "a8"])
def test_real_checkpoint_segments_are_byte_identical_and_row_curve(activation):
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=40, hidden=6144, units=4)
    rows, hidden, topk = 8192, case.h, case.topk
    torch.manual_seed(844)
    wire, _ = make_wire(torch.randn((rows, hidden), device="cuda") * 0.02)
    ids = torch.randint(0, 4, (rows, topk), device="cuda", dtype=torch.int32)
    ids[:, 0] = 0
    weights = torch.rand((rows, topk), device="cuda") / topk
    groups = (rows * topk + 63) // 64 + 4
    state = buffers(rows, hidden, case.u, topk, groups, 64)
    state["compact"] = compact_unit_descriptors(case.bits)
    base = buffers(4096, hidden, case.u, topk, groups, 64)
    base["compact"] = state["compact"]
    signs = case.signs_in, case.signs_out
    ordinary = (compile_exlr8_group_aot(experts=4, top_k=topk, max_groups=groups,
        hidden=hidden, max_rows=4096, group_rows=64),
        compile_exlr8_a16_aot(hidden=hidden, units=case.u, top_k=topk, max_groups=groups,
            device_count=True, grouped=True, activations=activation))
    group = compile_exlr8_group_aot(experts=4, top_k=topk, max_groups=groups,
        hidden=hidden, max_rows=8192, group_rows=64, row_addresses=True)
    core = compile_exlr8_a16_aot(hidden=hidden, units=case.u, top_k=topk, max_groups=groups,
        device_count=True, grouped=True, grouped_rows=8192, activations=activation, row_addresses=True)
    physical = torch.arange(rows - 1, -1, -1, device="cuda")
    resident = wire[physical].contiguous()
    input_rows = (resident.data_ptr() + physical * (hidden * 33 // 32)).to(torch.int64)
    reply = torch.full((rows + 2, hidden), 11, device="cuda", dtype=torch.bfloat16)
    output_rows = (reply.data_ptr() + (torch.arange(rows, device="cuda") + 1) * hidden * 2).to(torch.int64)
    expected = torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16)
    from benchmarks.bench_exlr8_grouped_prefill import runtime_l2_bytes
    l2 = runtime_l2_bytes()
    limit = torch.cuda.get_device_properties(0).multi_processor_count * 2

    def candidate(count):
        group.launch(ids, input_rows, signs[0], state["group_expert"], state["route_slots"],
            state["inverse"], state["count"], state["activation"], scalars=(count, limit))
        core.launch(state["activation"], case.pointers, case.bits, state["group_expert"], state["route_slots"],
            state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"], signs[1],
            output_rows, state["count"], *state["compact"][:2],
            scalars=(count, limit, state["compact"][2], l2))

    def baseline(count, split=4096):
        for first in range(0, count, split):
            live = min(split, count - first)
            run(ordinary, base, wire[first:], ids[first:], weights[first:], case.pointers,
                case.bits, signs, live, limit)
            expected[first:first + live].copy_(base["output"][:live])

    for count in (1, 333, 2048, 2752, 4096, 5504, 8192):
        baseline(count)
        candidate(count)
        torch.cuda.synchronize()
        assert torch.equal(reply[1:count + 1].view(torch.int16), expected[:count].view(torch.int16)), count
        assert torch.isfinite(reply[1:count + 1]).all() and reply[1:count + 1].count_nonzero() > 0
        assert (reply[0] == 11).all() and (reply[-1] == 11).all()
        if count >= 2048:
            measurements = {}
            for name, launch in (("ordinary", baseline), ("segments", candidate)):
                launch(count)
                samples = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch(count)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                measurements[name] = statistics.median(samples)
            print(json.dumps(dict(activation=activation, rows=count, byte_identical=True,
                us=measurements, us_per_row={name: value / count for name, value in measurements.items()})), flush=True)
    baseline(5504, split=2752)
    candidate(5504)
    torch.cuda.synchronize()
    assert torch.equal(reply[1:5505].view(torch.int16), expected[:5504].view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate(5504)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(reply[1:5505].view(torch.int16), expected[:5504].view(torch.int16))
