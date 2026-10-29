"""Two-member WS BF16 row-address byte gate and warmed merge ceiling."""
from dataclasses import fields, replace
import json

import pytest
import torch

pytest.importorskip("cutlass")

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    bind_mixed_trellis, build_tiered_maps, combine_trellis_rotations,
    compile_mixed_trellis, make_mixed_trellis_buffers,
    run_bound_mixed_trellis, warmup_mixed_trellis_route_pack,
)
from tests.moe.test_w4a16_mixed_trellis import _prepared


def _fc1_resources(launch):
    from cuda.bindings import driver, runtime

    def checked(result):
        assert int(result[0]) == 0, result
        return result[1]

    executable = getattr(launch.compiled, "_executable", launch.compiled)
    module = executable.jit_module
    kernels = [entry.kernel for entry in getattr(module, "cuda_modules", ())]
    for library in getattr(module, "cuda_library", ()):
        count = checked(runtime.cudaLibraryGetKernelCount(library))
        kernels.extend(checked(runtime.cudaLibraryEnumerateKernels(count, library)))
    records = []
    for value in kernels:
        kernel = driver.CUkernel(int(value))
        name = checked(driver.cuKernelGetName(kernel))
        name = name.decode() if isinstance(name, bytes) else str(name)
        if "ws_fc1_kernel" not in name:
            continue
        def attribute(attr):
            return int(checked(driver.cuKernelGetAttribute(attr, kernel, driver.CUdevice(0))))
        attrs = driver.CUfunction_attribute
        threads = (256 if launch.fc1_tile_n == 192 else launch.fc1_tile_n) + 128
        occupancy = checked(runtime.cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            value, threads, launch.shared_memory_bytes))
        records.append(dict(name=name, registers=attribute(attrs.CU_FUNC_ATTRIBUTE_NUM_REGS),
            static_smem=attribute(attrs.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            dynamic_smem=launch.shared_memory_bytes, threads=threads, ctas_per_sm=int(occupancy)))
    assert len(records) == 1, records
    return records[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM121")
def test_glmf_two_member_byte_gate_and_ceiling():
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 1):
        pytest.skip("requires SM121")
    torch.cuda.set_per_process_memory_fraction(min(0.20, (16 << 30) / props.total_memory))
    torch.manual_seed(844)
    hidden, intermediate, topk, experts, block = 4096, 1024, 8, 288, 64
    # Every expert is materialized: random routing exercises the real weight set.
    tiers = tuple(_prepared(experts=experts // 2, hidden=hidden,
        intermediate=intermediate, bits=bits, seed=bits, device=device,
        tile_config=(64, 256, 64, 256)) for bits in (3, 4))
    expert_map, descriptor = build_tiered_maps(range(144), range(144, 288), device=device)
    rotations = combine_trellis_rotations(*tiers)
    launches, buffers, bindings = [], [], []
    for capacity, indirect in ((2752, False), (5504, True)):
        slots = route_pack_capacity(capacity * topk, block, experts, topk=topk)[1]
        launch = compile_mixed_trellis(size_m=capacity, hidden_size=hidden,
            intermediate_size=intermediate, tier0_num_experts=144, tier1_num_experts=144,
            top_k=topk, route_num_experts=experts, max_m_blocks=(slots + block - 1) // block,
            sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
            force_tile_config=(64, 256, 64, 256), tier0_bits=3, tier1_bits=4,
            moe_block_size=block, full_rotation_output_dtype="bf16",
            warp_specialized=True, ws_dynamic_tiles=True, row_addresses=indirect)
        launches.append(launch)
        buffers.append(make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count))
        bindings.append(bind_mixed_trellis(*tiers, expert_map, descriptor, rotations, launch))
        warmup_mixed_trellis_route_pack(launch, buffers[-1], expert_map=expert_map)
    assert launches[0].compiled is not launches[1].compiled
    source = [(torch.randn(2754, hidden, device=device) * 0.01).bfloat16() for _ in range(2)]
    outputs = [torch.full((2754, hidden), 123, dtype=torch.bfloat16, device=device) for _ in range(2)]
    weights = torch.softmax(torch.randn(5504, topk, device=device), -1)
    cases = 0
    ceiling = None
    for hot in (False, True):
        ids = torch.randint(experts, (5504, topk), device=device, dtype=torch.int32)
        if hot:
            ids[:] = torch.tensor([0, 1, 2, 3, 144, 145, 146, 147], device=device)
        for lengths, order in (((2752, 2752), (0, 1)), ((2752, 2558), (0, 1)),
                               ((2752, 2558), (1, 0)), ((2752, 17), (1, 0))):
            row_inputs, row_outputs, member_ids, member_weights = [], [], [], []
            refs = {}
            offset = 0
            for member in order:
                rows = lengths[member]
                x = source[member][1:rows + 1]
                member_ids.append(ids[offset:offset + rows])
                member_weights.append(weights[offset:offset + rows])
                with kernel_resolution_guard("EXL3 singleton row-address reference"):
                    refs[member] = run_bound_mixed_trellis(x, member_weights[-1], member_ids[-1],
                        bindings[0], buffers[0]).clone()
                row_inputs.append(torch.arange(rows, device=device, dtype=torch.int64) * hidden * 2 + x.data_ptr())
                row_outputs.append(torch.arange(rows, device=device, dtype=torch.int64) * hidden * 2 + outputs[member][1:].data_ptr())
                offset += rows
            addresses = torch.cat(row_inputs)
            out_addresses = torch.cat(row_outputs)
            merged_ids, merged_weights = torch.cat(member_ids), torch.cat(member_weights)
            merged = replace(buffers[1], output=out_addresses)
            # Poison data/tails; counters retain their required zero initial state.
            for field in fields(merged):
                tensor = getattr(merged, field.name)
                if field.name == "workspace":
                    tensor.zero_()
                elif field.name != "output":
                    tensor.fill_(float("nan") if tensor.is_floating_point() else -777)
            for out in outputs:
                out.fill_(123)
            with kernel_resolution_guard("EXL3 addressed live-row gate"):
                run_bound_mixed_trellis(addresses, merged_weights, merged_ids, bindings[1], merged)
            torch.cuda.synchronize()
            for member in order:
                rows = lengths[member]
                actual = outputs[member][1:rows + 1]
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
                assert torch.equal(actual.view(torch.uint8), refs[member].view(torch.uint8))
                assert torch.all(outputs[member][0] == 123)
                assert torch.all(outputs[member][rows + 1:] == 123)
            cases += 1
            if not hot and lengths == (2752, 2752):
                def merged_call():
                    run_bound_mixed_trellis(addresses, merged_weights, merged_ids, bindings[1], merged)
                # Retain this warmed random-route case; time it after every byte gate.
                timing_args = (addresses, out_addresses, merged_ids, merged_weights,
                               tuple(member_ids), tuple(member_weights), merged)
                graph = torch.cuda.CUDAGraph()
                with kernel_resolution_guard("EXL3 addressed graph capture"), torch.cuda.graph(graph):
                    merged_call()
                graph.replay()
                torch.cuda.synchronize()
                for member in order:
                    assert torch.equal(outputs[member][1:2753].view(torch.uint8), refs[member].view(torch.uint8))
                graph.reset()
    addresses, out_addresses, merged_ids, merged_weights, member_ids, member_weights, merged = timing_args
    def singleton():
        for member in (0, 1):
            run_bound_mixed_trellis(source[member][1:2753], member_weights[member], member_ids[member],
                                   bindings[0], buffers[0])
    def merged_call():
        run_bound_mixed_trellis(addresses, merged_weights, merged_ids, bindings[1], merged)
    resources = [_fc1_resources(launch) for launch in launches]
    assert resources[1]["ctas_per_sm"] >= resources[0]["ctas_per_sm"]
    timings = []
    for call in (singleton, merged_call):
        call()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); call(); end.record(); end.synchronize()
        timings.append(start.elapsed_time(end))
    ceiling = dict(singleton_x2_us=timings[0] * 1000, merged_us=timings[1] * 1000, speedup=timings[0] / timings[1])
    print(json.dumps(dict(byte_cases=cases, ceiling=ceiling, fc1_resources=resources, gpu=props.name,
                          sms=props.multi_processor_count, shared_bytes=launches[1].shared_memory_bytes)), flush=True)
