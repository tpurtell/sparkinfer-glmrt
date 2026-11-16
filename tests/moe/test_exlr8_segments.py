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


def _run_exlr8_segments(activation, aot_dir=None):
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
    from b12x.integration.cuteafd import exportable_compilation
    with exportable_compilation():
        group = compile_exlr8_group_aot(experts=4, top_k=topk, max_groups=groups,
            hidden=hidden, max_rows=8192, group_rows=64, row_addresses=True)
        core = compile_exlr8_a16_aot(hidden=hidden, units=case.u, top_k=topk, max_groups=groups,
            device_count=True, grouped=True, grouped_rows=8192, activations=activation, row_addresses=True)
    if aot_dir is not None:
        from tests.moe.segment_aot import ExportedProgram
        group = ExportedProgram(group, aot_dir, "exlr8_group_segments")
        core = ExportedProgram(core, aot_dir, "exlr8_core_segments")
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


def test_glmf_288_expert_segments_exported_grouping_matches_ordinary(tmp_path):
    """The real GLMF expert domain crosses the 256-thread prefix-scan boundary."""
    from b12x.integration.cuteafd import exportable_compilation
    from tests.moe.segment_aot import ExportedProgram

    rows, hidden, experts, topk = 8192, 4096, 288, 8
    group_rows = 64
    groups = (rows * topk + group_rows - 1) // group_rows + experts
    torch.manual_seed(2282)
    wire, _ = make_wire(torch.randn((rows, hidden), device="cuda") * 0.02)
    ids = torch.randint(0, experts, (rows, topk), device="cuda", dtype=torch.int32)
    # Exercise every expert above 255 even in the first live-row tail.
    ids[0] = torch.tensor([0, 255, 256, 257, 284, 285, 286, 287], device="cuda")
    signs = torch.ones(hidden, dtype=torch.int8, device="cuda")
    physical = torch.arange(rows - 1, -1, -1, device="cuda")
    resident = wire[physical].contiguous()
    input_rows = (resident.data_ptr() + physical * (hidden * 33 // 32)).to(torch.int64)

    def buffers_for_group():
        return [torch.full((groups + 8,), -99, dtype=torch.int32, device="cuda"),
            torch.full((groups + 8, group_rows), -99, dtype=torch.int32, device="cuda"),
            torch.full((rows + 8, topk), -99, dtype=torch.int32, device="cuda"),
            torch.full((9,), -99, dtype=torch.int32, device="cuda"),
            torch.full((rows + 8, hidden), 11, dtype=torch.float16, device="cuda")]

    expected, actual = buffers_for_group(), buffers_for_group()
    guard_rows = 4
    ordinary = compile_exlr8_group_aot(experts=experts, top_k=topk, max_groups=groups,
        hidden=hidden, max_rows=4096, group_rows=group_rows)
    with exportable_compilation():
        program = compile_exlr8_group_aot(experts=experts, top_k=topk, max_groups=groups,
            hidden=hidden, max_rows=8192, group_rows=group_rows, row_addresses=True)
    candidate = ExportedProgram(program, tmp_path, "exlr8_glmf_group_segments")
    limit = torch.cuda.get_device_properties(0).multi_processor_count * 2

    def launch(program, inputs, state, count):
        program.launch(ids, inputs, signs, *(tensor[guard_rows:-guard_rows] for tensor in state),
            scalars=(count, limit))

    def check(count):
        torch.cuda.synchronize()
        checked = [tensor[guard_rows:-guard_rows] for tensor in actual]
        reference = [tensor[guard_rows:-guard_rows] for tensor in expected]
        total = int(checked[3][0].item())
        assert 0 < total <= groups, (count, total)
        live_ids = ids[:count].cpu().flatten().tolist()
        expected_experts, expected_slots = [], []
        for expert in range(experts):
            routes = [route for route, value in enumerate(live_ids) if value == expert]
            for first in range(0, len(routes), group_rows):
                chunk = routes[first:first + group_rows]
                expected_experts.append(expert)
                expected_slots.append(chunk + [-1] * (group_rows - len(chunk)))
        assert total == len(expected_experts), (count, total, len(expected_experts))
        assert checked[0][:total].cpu().tolist() == expected_experts
        assert checked[1][:total].cpu().tolist() == expected_slots
        for state in actual:
            guard = 11 if state.dtype == torch.float16 else -99
            assert (state[:guard_rows] == guard).all() and (state[-guard_rows:] == guard).all()
        if count <= 4096:
            assert int(reference[3][0].item()) == total
            assert torch.equal(checked[0][:total], reference[0][:total])
            assert torch.equal(checked[1][:total], reference[1][:total])
            assert torch.equal(checked[2][:count], reference[2][:count])
            assert torch.equal(checked[4][:count], reference[4][:count])
        assert torch.isfinite(checked[4][:count]).all()
        print(json.dumps(dict(experts=experts, rows=count, groups=total, exported_object=True)), flush=True)

    for count in (1, 333, 2048, 2752, 4096, 5504, 8192):
        if count <= 4096:
            launch(ordinary, wire, expected, count)
        launch(candidate, input_rows, actual, count)
        check(count)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(candidate, input_rows, actual, 5504)
    graph.replay()
    check(5504)


@pytest.mark.parametrize("activation", ["a16", "a8"])
def test_real_checkpoint_segments_are_byte_identical_and_row_curve(activation):
    _run_exlr8_segments(activation)


@pytest.mark.parametrize("activation", ["a16", "a8"])
def test_exlr8_segments_exported_header_and_byte_oracle(activation, tmp_path):
    _run_exlr8_segments(activation, aot_dir=tmp_path)
