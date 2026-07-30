from __future__ import annotations

import pytest
import torch

from sparkinfer._lib.intrinsics import (
    as_grouped_scale_view,
    swizzle_block_scale,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_activation,
    cuda,
)
from sparkinfer.moe.fused_moe.aot import (
    W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS,
    W4A4FC1W4A16FC2SparkSpec,
    bind_w4a4_fc1_w4a16_fc2_expert,
    bind_w4a4_fc1_w4a16_fc2_spark_workspace,
    compile_w4a4_fc1_w4a16_fc2_spark_aot,
    initialize_w4a4_fc1_w4a16_fc2_spark_routes,
    prepare_w4a4_fc1_w4a16_fc2_weights,
)


def _source_scale(
    *,
    experts: int,
    rows: int,
    cols: int,
    value: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return swizzle_block_scale(
        torch.full(
            (experts, rows, cols // 16),
            value,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
    )


def test_spark_spec_covers_balanced_and_long_capacity_buckets() -> None:
    assert W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS == (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    )
    for capacity in W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS:
        spec = W4A4FC1W4A16FC2SparkSpec(
            m=capacity,
            hidden_size=6144,
            intermediate_size=512,
        )
        assert spec.m == capacity
        assert spec.route_slots >= capacity


def test_spark_workspace_has_no_reorder_or_cooperative_barrier() -> None:
    spec = W4A4FC1W4A16FC2SparkSpec(
        m=256,
        hidden_size=6144,
        intermediate_size=512,
    )
    layout = spec.workspace_layout()
    names = {region.name for region in layout.regions}

    assert "fc1_reordered" not in names
    assert names == {
        "fc1_bf16",
        "activated_bf16",
        "packed_route_indices",
        "block_expert_ids",
        "packed_route_count",
        "topk_weights",
        "fc2_scratch",
        "fc2_locks",
    }
    assert layout.region("fc1_bf16").size_bytes == 256 * 1024 * 2
    assert layout.region("activated_bf16").size_bytes == 256 * 512 * 2


def test_spark_recipe_rejects_preordered_w31_source() -> None:
    with pytest.raises(ValueError, match=r"source \[up; gate\] W13 order"):
        W4A4FC1W4A16FC2SparkSpec(
            m=1,
            hidden_size=6144,
            intermediate_size=512,
            w13_layout="w31",
        )


def test_m1_workspace_manifest_abi_is_fixed_and_aligned() -> None:
    spec = W4A4FC1W4A16FC2SparkSpec(
        m=1,
        hidden_size=6144,
        intermediate_size=512,
    )
    layout = spec.workspace_layout()

    assert layout.alignment == 256
    assert layout.size_bytes == 401_408
    assert {
        region.name: (region.offset_bytes, region.size_bytes)
        for region in layout.regions
    } == {
        "fc1_bf16": (0, 2_048),
        "activated_bf16": (2_048, 1_024),
        "packed_route_indices": (3_072, 32),
        "block_expert_ids": (3_328, 4),
        "packed_route_count": (3_584, 4),
        "topk_weights": (3_840, 4),
        "fc2_scratch": (4_096, 393_216),
        "fc2_locks": (397_312, 4_096),
    }


def test_weight_prep_keeps_only_source_w13_and_in_place_packed_w2() -> None:
    experts, hidden, intermediate = 2, 128, 64
    w13 = torch.full(
        (experts, 2 * intermediate, hidden // 2),
        0x11,
        dtype=torch.uint8,
    )
    w2 = torch.full(
        (experts, hidden, intermediate // 2),
        0x11,
        dtype=torch.uint8,
    )
    w13_scale = _source_scale(
        experts=experts,
        rows=2 * intermediate,
        cols=hidden,
    )
    w2_scale = _source_scale(
        experts=experts,
        rows=hidden,
        cols=intermediate,
    )
    w13_alpha = torch.tensor((0.375, 1.25), dtype=torch.float32)
    w2_alpha = torch.tensor((0.625, 1.5), dtype=torch.float32)
    source_w2_storage = w2.untyped_storage().data_ptr()

    prepared = prepare_w4a4_fc1_w4a16_fc2_weights(
        w13,
        w13_scale,
        w13_alpha,
        w2,
        w2_scale,
        w2_alpha,
        activation="silu",
    )
    bound = bind_w4a4_fc1_w4a16_fc2_expert(prepared, 0)

    assert prepared.w13_weight_source.data_ptr() == w13.data_ptr()
    assert prepared.w13_scale_source.data_ptr() == w13_scale.data_ptr()
    assert prepared.w2_weight_packed.untyped_storage().data_ptr() == source_w2_storage
    assert not prepared.has_packed_w13
    assert not prepared.has_source_w2
    assert "w2_fp4" not in prepared.__dataclass_fields__
    assert "packed_w13" not in prepared.__dataclass_fields__

    # A prequantized payload's per-K16 scales are complete. With E2M1 values
    # 0.5, unit K16 scales, and 128 terms, the unscaled FC1 dot is 32. The
    # only FC1 scalar must be the raw non-unity W13 checkpoint alpha.
    unscaled_fc1 = 128 * 0.5 * 0.5
    assert float(bound.w13_runtime_alpha.item()) == pytest.approx(0.375)
    assert unscaled_fc1 * float(bound.w13_runtime_alpha.item()) == pytest.approx(12.0)
    assert float(bound.w2_w4a16_alpha.item()) != pytest.approx(
        float(w2_alpha[0].item())
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("capacity", "active_rows"), ((1, 1), (8, 3)))
def test_composed_artifact_launch_and_graph_replay(
    capacity: int,
    active_rows: int,
) -> None:
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    hidden, intermediate = 128, 64
    spec = W4A4FC1W4A16FC2SparkSpec(
        m=capacity,
        hidden_size=hidden,
        intermediate_size=intermediate,
        target_arch=f"sm_{capability[0]}{capability[1]}",
    )
    w13 = torch.full(
        (1, 2 * intermediate, hidden // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.full(
        (1, hidden, intermediate // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = _source_scale(
        experts=1,
        rows=2 * intermediate,
        cols=hidden,
        value=0.015625,
        device=device,
    )
    w2_scale = _source_scale(
        experts=1,
        rows=hidden,
        cols=intermediate,
        value=0.015625,
        device=device,
    )
    prepared = prepare_w4a4_fc1_w4a16_fc2_weights(
        w13,
        w13_scale,
        torch.tensor((0.375,), dtype=torch.float32, device=device),
        w2,
        w2_scale,
        torch.tensor((0.625,), dtype=torch.float32, device=device),
        activation="silu",
        w13_layout="w13",
    )
    bound = bind_w4a4_fc1_w4a16_fc2_expert(prepared, 0)
    artifact = compile_w4a4_fc1_w4a16_fc2_spark_aot(spec, device=device)
    layout = spec.workspace_layout()
    arena_storage = torch.empty(
        layout.size_bytes + layout.alignment,
        dtype=torch.uint8,
        device=device,
    )
    arena_offset = (-arena_storage.data_ptr()) % layout.alignment
    arena = arena_storage.narrow(0, arena_offset, layout.size_bytes)
    workspace = bind_w4a4_fc1_w4a16_fc2_spark_workspace(arena, spec)
    initialize_w4a4_fc1_w4a16_fc2_spark_routes(
        workspace,
        spec,
        active_rows=active_rows,
    )
    input_packed = torch.full(
        (active_rows, hidden // 2, 1),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    input_scale_storage = _source_scale(
        experts=1,
        rows=active_rows,
        cols=hidden,
        value=0.5,
        device=device,
    )
    input_scale = as_grouped_scale_view(
        input_scale_storage,
        active_rows,
        hidden,
    )
    output = torch.empty(
        (active_rows, hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    stream = torch.cuda.Stream()

    def launch() -> None:
        artifact.launch(
            input_packed=input_packed,
            input_scale=input_scale,
            weights=bound,
            workspace=workspace,
            output=output,
            active_rows=active_rows,
            stream=stream,
        )

    launch()
    stream.synchronize()
    expected_fc1 = hidden * 0.5 * 0.5 * 0.5 * 0.015625 * 0.375
    assert torch.all(
        workspace.fc1_bf16[:active_rows]
        == torch.tensor(expected_fc1, dtype=torch.bfloat16, device=device)
    )
    assert torch.isfinite(workspace.activated_bf16[:active_rows]).all()
    assert torch.count_nonzero(workspace.activated_bf16[:active_rows]) > 0
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == active_rows * hidden
    reference = output.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    before_replay = torch.cuda.memory_allocated(device)
    for _ in range(4):
        graph.replay()
    stream.synchronize()
    after_replay = torch.cuda.memory_allocated(device)

    assert after_replay == before_replay
    assert torch.equal(output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_source_order_activation_eliminates_reorder_bitwise() -> None:
    device = torch.device("cuda")
    rows, intermediate = 3, 64
    generator = torch.Generator(device=device).manual_seed(20260730)
    up = torch.randn(
        (rows, intermediate),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate = torch.randn(
        (rows, intermediate),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    source_w13 = torch.cat((up, gate), dim=1).contiguous()
    reordered_w31 = torch.cat((gate, up), dim=1).contiguous()
    source_output = torch.empty_like(up)
    reordered_output = torch.empty_like(up)
    source = compile_w4a16_activation(
        rows=rows,
        intermediate_size=intermediate,
        activation="silu",
        w13_layout="w13",
    )
    reordered = compile_w4a16_activation(
        rows=rows,
        intermediate_size=intermediate,
        activation="silu",
        w13_layout="w31",
    )
    source_again = compile_w4a16_activation(
        rows=rows,
        intermediate_size=intermediate,
        activation="silu",
        w13_layout="w13",
    )
    stream = torch.cuda.Stream()
    stream_arg = cuda.CUstream(stream.cuda_stream)
    with torch.cuda.stream(stream):
        source.compiled(
            source_w13.view(-1),
            source_output.view(-1),
            rows,
            stream_arg,
        )
        reordered.compiled(
            reordered_w31.view(-1),
            reordered_output.view(-1),
            rows,
            stream_arg,
        )
    stream.synchronize()

    assert source.w13_layout == "w13"
    assert reordered.w13_layout == "w31"
    assert source.compiled is source_again.compiled
    assert source.compiled is not reordered.compiled
    assert torch.equal(source_output, reordered_output)
