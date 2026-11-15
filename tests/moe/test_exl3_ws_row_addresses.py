"""Mapped transport-frame singleton/merged byte gates for WS EXL3 A16."""
from dataclasses import replace
from itertools import permutations
import json

import pytest
import torch

pytest.importorskip("cutlass")

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe._shared.kernels.w4a16.host import (
    exl3_ws_segment_scratch_bytes, exl3_ws_segment_scratch_regions, route_pack_capacity,
)
from b12x.moe._shared.kernels.w4a16.kernel import W4A16TopKSumKernel
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    bind_mixed_trellis, build_tiered_maps, combine_trellis_rotations,
    compile_mixed_trellis, make_mixed_trellis_buffers, make_mixed_trellis_segment_buffers,
    run_bound_mixed_trellis, warmup_mixed_trellis_route_pack,
)
from b12x.sequence._shared.disk_table import MappedHostAllocation
from benchmarks.benchmark_exlr8_a16 import make_wire
from tests.moe.test_w4a16_mixed_trellis import _prepared


def test_compressed_reply_addresses_refused():
    for reply in (1, 2):
        with pytest.raises(ValueError, match="do not support fused FP8/NVFP4 replies"):
            W4A16TopKSumKernel(topk=8, hidden_size=6144, element_dtype="fp16",
                full_rotation=True, full_rotation_output_dtype="bf16", num_experts=256,
                reply_format=reply, row_addresses=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM121")
@pytest.mark.parametrize("family,input_format,reply_dtype", [("glm", "e4m3_k32", "bf16"),
    ("glmf", "e4m3_k32", "bf16"), ("glm", "bf16", "bf16"),
    ("glm", "e4m3_k32", "fp32")])
def test_mapped_frame_member_orders(family, input_format, reply_dtype):
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 1):
        pytest.skip("requires SM121")
    torch.cuda.set_per_process_memory_fraction(min(0.9, (12 << 30) / props.total_memory))
    torch.manual_seed(844)
    hidden, intermediate, experts, bits, clamp = ((6144, 512, 256, (4, 5), None)
        if family == "glm" else (4096, 512, 288, (3, 4), 10.0))
    topk, block, maximum = 8, 64, 8192
    reply_bytes = 2 if reply_dtype == "bf16" else 4
    high = experts // 4
    counts = (experts - high, high)
    tiers = tuple(_prepared(experts=count, hidden=hidden, intermediate=intermediate,
        bits=bit, seed=bit, device=device, tile_config=(64, 256, 64, 256))
        for bit, count in zip(bits, counts))
    mapping, descriptor = build_tiered_maps(range(counts[0]), range(counts[0], experts), device=device)
    rotations = combine_trellis_rotations(*tiers)
    launches, buffers, bindings = [], [], []
    for indirect in (False, True):
        capacity = maximum if indirect else 4096
        _, _, blocks = route_pack_capacity(capacity * topk, block, experts, topk=topk)
        launch = compile_mixed_trellis(size_m=capacity, hidden_size=hidden,
            intermediate_size=intermediate, tier0_num_experts=counts[0], tier1_num_experts=counts[1],
            top_k=topk, route_num_experts=experts, max_m_blocks=blocks,
            sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
            force_tile_config=(64, 256, 64, 256), tier0_bits=bits[0], tier1_bits=bits[1],
            moe_block_size=block, full_rotation_output_dtype=reply_dtype, swiglu_limit=clamp,
            warp_specialized=True, ws_dynamic_tiles=True, ws_prerotate=False,
            input_format=input_format, row_addresses=indirect)
        launches.append(launch)
        workspace = (make_mixed_trellis_segment_buffers(launch, rows=maximum, device=device,
            sms=props.multi_processor_count) if indirect else
            make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count))
        buffers.append(workspace)
        bindings.append(bind_mixed_trellis(*tiers, mapping, descriptor, rotations, launch))
        warmup_mixed_trellis_route_pack(launch, workspace, expert_map=mapping)
    # A fixed compiled metadata prefix, with exactly live-sized data regions.
    offsets = set()
    for rows in (2048, 2752, 4096, 5504):
        geometry = dict(rows=rows, maximum=maximum, hidden=hidden, intermediate=intermediate,
            topk=topk, experts=experts, block=block, max_m_blocks=launches[1].max_m_blocks,
            sms=props.multi_processor_count)
        regions = exl3_ws_segment_scratch_regions(**geometry)
        size = exl3_ws_segment_scratch_bytes(**geometry)
        arena = torch.empty(size, dtype=torch.uint8, device=device)
        live = make_mixed_trellis_segment_buffers(launches[1], rows=rows, device=device,
            sms=props.multi_processor_count, scratch=arena)
        offset = 0
        for name, extent in regions.items():
            view = getattr(live, name)
            assert view.numel() * view.element_size() == extent
            assert view.data_ptr() == arena.data_ptr() + offset
            offset += (extent + 255) & ~255
        assert offset == arena.numel()
        offsets.add(live.packed_route_indices.data_ptr() - arena.data_ptr())
        assert live.rotation_gate.data_ptr() == live.rotation_up.data_ptr() == live.fc2.data_ptr()
    assert len(offsets) == 1
    del live, arena

    lengths = (2752, 2752, 2487, 1900, 333)
    stride = hidden + hidden // 32 if input_format == "e4m3_k32" else 2 * hidden
    request_stride = (96 + 4096 * (40 + 12 * topk + stride) + 4095) & ~4095
    response_stride = (96 + 4096 * (reply_bytes * hidden + 4) + 4095) & ~4095
    allocations = []
    segments = []
    cases = identical_members = identical_bytes = 0
    try:
        for member, rows in enumerate(lengths):
            x = (torch.randn(rows, hidden, device=device) * 0.01).bfloat16()
            source = make_wire(x) if input_format == "e4m3_k32" else x
            prefix = (96 + rows * (40 + 12 * topk) + 15) & ~15
            start, out_start = member * request_stride + prefix, member * response_stride + 96
            inp = MappedHostAllocation(((member + 1) * request_stride,), torch.uint8, device)
            out = MappedHostAllocation(((member + 1) * response_stride,), torch.uint8, device)
            allocations.extend((inp, out))
            inp.host_view.fill_(173)
            inp.host_view[start:start + rows * stride].copy_(source.view(torch.uint8).flatten().cpu())
            assert (inp.device_view.data_ptr() + start) % 16 == 0
            segments.append(dict(inp=inp, out=out, snapshot=inp.host_view.clone(), start=start,
                                 out_start=out_start, rows=rows, source=source))
        for hot in (False, True):
            for segment in segments:
                rows = segment["rows"]
                ids = torch.rand(rows, experts, device=device).topk(topk, -1).indices.int().contiguous()
                if hot:
                    ids[:] = torch.tensor([0, 1, 2, 3, counts[0], counts[0] + 1,
                                           counts[0] + 2, counts[0] + 3], device=device)
                weights = torch.softmax(torch.randn(rows, topk, device=device), -1)
                run = lambda: run_bound_mixed_trellis(segment["source"], weights, ids, bindings[0], buffers[0])
                run()  # Warm each live-route bucket before freezing resolution.
                with kernel_resolution_guard("EXL3 ordinary singleton oracle"):
                    reference = run().clone()
                assert torch.isfinite(reference).all() and reference.count_nonzero()
                segment.update(ids=ids, weights=weights, reference=reference.cpu())
            for count in (1, 2):
                for order in permutations(range(len(segments)), count):
                    for segment in segments:
                        segment["out"].host_view.fill_(173)
                    x_addresses, y_addresses, route_ids, route_weights = [], [], [], []
                    for position, member in enumerate(order):
                        s = segments[member]
                        rows = s["rows"]
                        permutation = torch.arange(rows - 1, -1, -1, device=device) if position else torch.randperm(rows, device=device)
                        x_addresses.append(s["inp"].device_view.data_ptr() + s["start"] + permutation.long() * stride)
                        y_addresses.append(s["out"].device_view.data_ptr() + s["out_start"] + permutation.long() * hidden * reply_bytes)
                        route_ids.append(s["ids"][permutation])
                        route_weights.append(s["weights"][permutation])
                    addresses, output = torch.cat(x_addresses), torch.cat(y_addresses)
                    ids, weights = torch.cat(route_ids), torch.cat(route_weights)
                    live = make_mixed_trellis_segment_buffers(launches[1], rows=len(addresses),
                        device=device, sms=props.multi_processor_count)
                    live = replace(live, output=output)
                    # Route-pack buckets depend on live rows; warm outside the resolution guard.
                    from b12x.moe._shared.kernels.w4a16.kernel import pack_topk_routes_by_expert
                    pack_topk_routes_by_expert(ids, block, experts, expert_map=mapping,
                        packed_route_indices=live.packed_route_indices, block_expert_ids=live.block_expert_ids,
                        packed_route_count=live.packed_route_count, expert_offsets=live.expert_offsets,
                        expert_counts=live.expert_counts)
                    live.fc1.fill_(float("nan")); live.activated.fill_(float("nan")); live.fc2.fill_(float("nan"))
                    with kernel_resolution_guard("EXL3 merged mapped transport rows"):
                        run_bound_mixed_trellis(addresses, weights, ids, bindings[1], live)
                    if not hot and order == (0, 1):
                        graph = torch.cuda.CUDAGraph()
                        with kernel_resolution_guard("EXL3 mapped graph capture"), torch.cuda.graph(graph):
                            run_bound_mixed_trellis(addresses, weights, ids, bindings[1], live)
                        graph.replay()
                        graph.reset()
                    torch.cuda.synchronize()
                    for member, s in enumerate(segments):
                        assert torch.equal(s["inp"].host_view, s["snapshot"]), "input/guards modified"
                        raw = s["out"].host_view
                        start, end = s["out_start"], s["out_start"] + s["rows"] * hidden * reply_bytes
                        assert (raw[:start] == 173).all() and (raw[end:] == 173).all(), "reply guards modified"
                        if member in order:
                            assert torch.equal(raw[start:end], s["reference"].view(torch.uint8).flatten()), (family, input_format, hot, order, member)
                            identical_members += 1
                            identical_bytes += end - start
                        else:
                            assert (raw == 173).all(), "nonmember reply modified"
                    cases += 1
                    print(json.dumps(dict(family=family, input_format=input_format, reply_dtype=reply_dtype, hot=hot,
                        order=order, rows=len(addresses), byte_identical=True)), flush=True)
        print(json.dumps(dict(family=family, input_format=input_format, reply_dtype=reply_dtype, cases=cases,
            identical_members=identical_members, identical_bytes=identical_bytes,
            capacity=maximum, gpu=props.name)), flush=True)
    finally:
        torch.cuda.synchronize()
        for allocation in reversed(allocations):
            allocation.close()
