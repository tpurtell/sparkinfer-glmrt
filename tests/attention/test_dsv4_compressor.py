from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sparkinfer.attention import dsv4_compressor
from sparkinfer.attention._shared.mla.compressed_reference import (
    pack_compressed_mla_kv_cache_reference,
    unpack_compressed_mla_kv_cache_reference,
)
from sparkinfer.attention.nsa_indexer.reference import (
    pack_index_k_cache_reference,
    unpack_index_k_cache_reference,
)

from ..conftest import require_sparkinfer


def _rope_forward(
    values: torch.Tensor,
    position: int,
    cos_sin: torch.Tensor,
    *,
    nope_dim: int,
) -> torch.Tensor:
    output = values.clone()
    pairs = values[nope_dim:].float().unflatten(-1, (-1, 2))
    cos_v, sin_v = cos_sin[position].chunk(2)
    even = pairs[:, 0] * cos_v - pairs[:, 1] * sin_v
    odd = pairs[:, 1] * cos_v + pairs[:, 0] * sin_v
    output[nope_dim:] = torch.stack((even, odd), dim=-1).flatten().to(torch.bfloat16)
    return output


def _rmsnorm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = values.to(torch.bfloat16).float()
    return (values * torch.rsqrt(values.square().mean() + eps) * weight.float()).to(
        torch.bfloat16
    )


def _overlap_pool(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    kv = torch.cat((kv_state[:4, :head_dim], kv_state[4:, head_dim:]), dim=0)
    score = torch.cat((score_state[:4, :head_dim], score_state[4:, head_dim:]), dim=0)
    return (kv * score.softmax(dim=0)).sum(dim=0)


def _initial_prefill_pool(
    projection: torch.Tensor,
    ape: torch.Tensor,
    *,
    source_start: int,
    rope_position: int,
    head_dim: int,
) -> torch.Tensor:
    ratio, projected_width = ape.shape
    overlap = projected_width == 2 * head_dim
    if overlap:
        current_kv = projection[
            source_start : source_start + ratio, head_dim:projected_width
        ]
        current_score = (
            projection[
                source_start : source_start + ratio,
                projected_width + head_dim : 2 * projected_width,
            ]
            + ape[:, head_dim:]
        )
        kv = current_kv
        score = current_score
        if rope_position > 0:
            previous_kv = projection[source_start - ratio : source_start, :head_dim]
            previous_score = (
                projection[
                    source_start - ratio : source_start,
                    projected_width : projected_width + head_dim,
                ]
                + ape[:, :head_dim]
            )
            kv = torch.cat((previous_kv, current_kv), dim=0)
            score = torch.cat((previous_score, current_score), dim=0)
    else:
        kv = projection[source_start : source_start + ratio, :head_dim]
        score = (
            projection[
                source_start : source_start + ratio,
                projected_width : 2 * projected_width,
            ]
            + ape
        )
    return (kv.float() * score.float().softmax(dim=0)).sum(dim=0)


def _state_from_history(
    projection: torch.Tensor,
    ape: torch.Tensor,
    *,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_width = int(ape.shape[1])
    overlap = projected_width in (1_024, 256) and ratio == 4
    state_rows = ratio * (2 if overlap else 1)
    kv = torch.zeros((state_rows, projected_width), device="cuda", dtype=torch.float32)
    score = torch.full_like(kv, float("-inf"))
    source_tokens = int(projection.shape[0])
    cutoff = source_tokens // ratio * ratio
    remainder = source_tokens - cutoff
    if overlap and cutoff >= ratio:
        previous = projection[cutoff - ratio : cutoff]
        kv[:ratio] = previous[:, :projected_width].float()
        score[:ratio] = previous[:, projected_width:].float() + ape
    if remainder:
        current = projection[cutoff:source_tokens]
        state_start = ratio if overlap else 0
        kv[state_start : state_start + remainder] = current[:, :projected_width].float()
        score[state_start : state_start + remainder] = (
            current[:, projected_width:].float() + ape[:remainder]
        )
    return kv, score


def _fwht_128(values: torch.Tensor) -> torch.Tensor:
    output = values.float()
    width = 1
    while width < 128:
        groups = output.view(-1, 2, width)
        first = groups[:, 0].clone()
        second = groups[:, 1].clone()
        output = torch.stack((first + second, first - second), dim=1).flatten()
        width *= 2
    return output * (128**-0.5)


def _fp4_qat_reference(values: torch.Tensor) -> torch.Tensor:
    blocks = values.float().view(-1, 32)
    scales = blocks.abs().amax(dim=1).clamp_min(6 * torch.finfo(torch.float32).tiny) / 6
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    magnitude = (blocks.abs() / scales[:, None]).clamp_max(6)
    fp4 = torch.where(
        magnitude < 0.25,
        0.0,
        torch.where(
            magnitude < 0.75,
            0.5,
            torch.where(
                magnitude < 1.25,
                1.0,
                torch.where(
                    magnitude < 1.75,
                    1.5,
                    torch.where(
                        magnitude < 2.5,
                        2.0,
                        torch.where(
                            magnitude < 3.5,
                            3.0,
                            torch.where(magnitude < 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return (torch.copysign(fp4, blocks) * scales[:, None]).flatten().to(torch.bfloat16)


def _random(shape: tuple[int, ...], dtype: torch.dtype, scale: float = 1.0):
    return (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()


def _cos_sin(rows: int) -> torch.Tensor:
    angles = torch.randn((rows, 32), device="cuda", dtype=torch.float32)
    return torch.cat((angles.cos(), angles.sin()), dim=1).contiguous()


def test_plan_pins_c4_and_c128_arena_and_state_geometry() -> None:
    c4 = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cpu",
            max_tokens=2_048,
            hidden=4_096,
            compress_ratio=4,
            with_indexer=True,
        )
    )
    c128 = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cpu",
            max_tokens=2_048,
            hidden=4_096,
            compress_ratio=128,
            with_indexer=False,
        )
    )

    assert c4.caps.joint_projection_width == 2_560
    assert c4.caps.state_rows == 8
    assert c4.caps.main_page_rows == 64
    assert c4.caps.main_page_bytes == 37_440
    assert c4.scratch_specs()[0].nbytes == 10_485_760
    assert c128.caps.joint_projection_width == 1_024
    assert c128.caps.state_rows == 128
    assert c128.caps.main_page_rows == 2
    assert c128.caps.main_page_bytes == 1_728
    assert c128.scratch_specs()[0].nbytes == 4_194_304


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compress_ratio": 4, "with_indexer": False},
        {"compress_ratio": 128, "with_indexer": True},
        {"compress_ratio": 8, "with_indexer": False},
    ],
)
def test_caps_fail_closed_on_non_model_compressor_combinations(kwargs) -> None:
    with pytest.raises(ValueError):
        dsv4_compressor.Caps(device="cpu", max_tokens=1, hidden=4_096, **kwargs)


def test_c128_decode_updates_fp32_state_and_direct_packed_page() -> None:
    require_sparkinfer()
    torch.manual_seed(20260804)
    ratio, hidden, eps = 128, 4_096, 1.0e-6
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=1,
            hidden=hidden,
            compress_ratio=ratio,
            with_indexer=False,
        )
    )
    main_wkv = _random((512, hidden), torch.bfloat16, 1 / 64)
    main_wgate = _random((512, hidden), torch.bfloat16, 1 / 64)
    main_ape = _random((ratio, 512), torch.float32, 1 / 16)
    main_norm = (_random((512,), torch.bfloat16, 1 / 16) + 1).contiguous()
    weights = dsv4_compressor.pack_weights(main_wkv, main_wgate, main_ape, main_norm)
    hidden_states = _random((1, hidden), torch.bfloat16)
    positions = torch.tensor([127], device="cuda", dtype=torch.int32)
    sequence_ids = torch.tensor([0], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    cos_sin = _cos_sin(256)
    main_cache = torch.zeros(
        (1, plan.caps.main_page_bytes), device="cuda", dtype=torch.uint8
    )
    kv_state = _random((1, 128, 512), torch.float32, 1 / 4)
    score_state = _random((1, 128, 512), torch.float32, 1 / 4)
    expected_kv = kv_state.clone()
    expected_score = score_state.clone()
    projection = F.linear(hidden_states, weights.joint_projection)
    expected_kv[0, 127] = projection[0, :512].float()
    expected_score[0, 127] = projection[0, 512:].float() + main_ape[127]
    pooled = (expected_kv[0] * expected_score[0].softmax(dim=0)).sum(dim=0)
    expected = _rope_forward(_rmsnorm(pooled, main_norm, eps), 0, cos_sin, nope_dim=448)
    expected_cache = pack_compressed_mla_kv_cache_reference(
        expected[None, :448], expected[None, 448:], page_size=2, num_pages=1
    )

    scratch = torch.empty(
        plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
    )
    binding = dsv4_compressor.bind_decode(
        plan,
        scratch=scratch,
        hidden_states=hidden_states,
        positions=positions,
        sequence_ids=sequence_ids,
        compressed_slots=slots,
        compressed_cos_sin_cache=cos_sin,
        compressed_main_cache=main_cache,
        main_kv_state=kv_state,
        main_score_state=score_state,
        weights=weights,
        eps=eps,
        rows_are_sequence_unique=True,
    )
    dsv4_compressor.run_decode(binding=binding)
    torch.cuda.synchronize()

    assert torch.equal(kv_state, expected_kv)
    assert torch.equal(score_state, expected_score)
    actual_nope, actual_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=2, n_tokens=1
    )
    expected_nope, expected_rope = unpack_compressed_mla_kv_cache_reference(
        expected_cache, page_size=2, n_tokens=1
    )
    torch.testing.assert_close(actual_nope, expected_nope, atol=0.04, rtol=0.04)
    torch.testing.assert_close(actual_rope, expected_rope, atol=0.02, rtol=0.02)

    eager_cache = main_cache.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_compressor.run_decode(binding=binding)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(main_cache, eager_cache)


def test_c4_decode_rolls_overlap_and_writes_main_and_index_pages() -> None:
    require_sparkinfer()
    torch.manual_seed(20260805)
    hidden, eps = 4_096, 1.0e-6
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=1,
            hidden=hidden,
            compress_ratio=4,
            with_indexer=True,
        )
    )
    main_wkv = _random((1_024, hidden), torch.bfloat16, 1 / 64)
    main_wgate = _random((1_024, hidden), torch.bfloat16, 1 / 64)
    main_ape = _random((4, 1_024), torch.float32, 1 / 16)
    main_norm = (_random((512,), torch.bfloat16, 1 / 16) + 1).contiguous()
    index_wkv = _random((256, hidden), torch.bfloat16, 1 / 64)
    index_wgate = _random((256, hidden), torch.bfloat16, 1 / 64)
    index_ape = _random((4, 256), torch.float32, 1 / 16)
    index_norm = (_random((128,), torch.bfloat16, 1 / 16) + 1).contiguous()
    weights = dsv4_compressor.pack_weights(
        main_wkv,
        main_wgate,
        main_ape,
        main_norm,
        index_wkv=index_wkv,
        index_wgate=index_wgate,
        index_ape=index_ape,
        index_norm=index_norm,
    )
    hidden_states = _random((1, hidden), torch.bfloat16)
    positions = torch.tensor([3], device="cuda", dtype=torch.int32)
    sequence_ids = torch.tensor([0], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    cos_sin = _cos_sin(32)
    main_cache = torch.zeros((1, 37_440), device="cuda", dtype=torch.uint8)
    index_cache = torch.zeros((1, 8_448), device="cuda", dtype=torch.uint8)
    main_kv_state = _random((1, 8, 1_024), torch.float32, 1 / 4)
    main_score_state = _random((1, 8, 1_024), torch.float32, 1 / 4)
    index_kv_state = _random((1, 8, 256), torch.float32, 1 / 4)
    index_score_state = _random((1, 8, 256), torch.float32, 1 / 4)
    expected_main_kv = main_kv_state.clone()
    expected_main_score = main_score_state.clone()
    expected_index_kv = index_kv_state.clone()
    expected_index_score = index_score_state.clone()
    projection = F.linear(hidden_states, weights.joint_projection)[0]
    expected_main_kv[0, 7] = projection[:1_024].float()
    expected_main_score[0, 7] = projection[1_024:2_048].float() + main_ape[3]
    expected_index_kv[0, 7] = projection[2_048:2_304].float()
    expected_index_score[0, 7] = projection[2_304:].float() + index_ape[3]
    main_pooled = _overlap_pool(
        expected_main_kv[0], expected_main_score[0], head_dim=512
    )
    expected_main = _rope_forward(
        _rmsnorm(main_pooled, main_norm, eps), 0, cos_sin, nope_dim=448
    )
    index_pooled = _overlap_pool(
        expected_index_kv[0], expected_index_score[0], head_dim=128
    )
    expected_index = _rope_forward(
        _rmsnorm(index_pooled, index_norm, eps), 0, cos_sin, nope_dim=64
    )
    expected_index = _fp4_qat_reference(_fwht_128(expected_index))
    expected_main_cache = pack_compressed_mla_kv_cache_reference(
        expected_main[None, :448],
        expected_main[None, 448:],
        page_size=64,
        num_pages=1,
    )
    expected_index_cache = pack_index_k_cache_reference(expected_index[None])
    rolled_main_kv = expected_main_kv[:, 4:].clone()
    rolled_main_score = expected_main_score[:, 4:].clone()
    rolled_index_kv = expected_index_kv[:, 4:].clone()
    rolled_index_score = expected_index_score[:, 4:].clone()

    scratch = torch.empty(
        plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
    )
    binding = dsv4_compressor.bind_decode(
        plan,
        scratch=scratch,
        hidden_states=hidden_states,
        positions=positions,
        sequence_ids=sequence_ids,
        compressed_slots=slots,
        compressed_cos_sin_cache=cos_sin,
        compressed_main_cache=main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        weights=weights,
        index_cache=index_cache,
        index_kv_state=index_kv_state,
        index_score_state=index_score_state,
        eps=eps,
        rows_are_sequence_unique=True,
    )
    dsv4_compressor.run_decode(binding=binding)
    torch.cuda.synchronize()

    assert torch.equal(main_kv_state[:, :4], rolled_main_kv)
    assert torch.equal(main_score_state[:, :4], rolled_main_score)
    assert torch.equal(index_kv_state[:, :4], rolled_index_kv)
    assert torch.equal(index_score_state[:, :4], rolled_index_score)
    main_nope, main_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=64, n_tokens=1
    )
    expected_nope, expected_rope = unpack_compressed_mla_kv_cache_reference(
        expected_main_cache, page_size=64, n_tokens=1
    )
    torch.testing.assert_close(main_nope, expected_nope, atol=0.04, rtol=0.04)
    torch.testing.assert_close(main_rope, expected_rope, atol=0.02, rtol=0.02)
    index_actual = unpack_index_k_cache_reference(
        index_cache, num_tokens=1, page_size=64
    )
    index_expected = unpack_index_k_cache_reference(
        expected_index_cache, num_tokens=1, page_size=64
    )
    index_error = (index_actual - index_expected).abs()
    # FP32 reduction order can move a value across one discontinuous E2M1
    # threshold. The remaining 127 lanes match the checkpoint QAT path.
    assert int((index_error > 0.04).sum()) <= 1
    assert float(index_error.max()) <= 0.25


def test_c4_initial_prefill_emits_parallel_groups_and_exact_terminal_state() -> None:
    require_sparkinfer()
    torch.manual_seed(20260806)
    hidden, eps = 4_096, 1.0e-6
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=32,
            hidden=hidden,
            compress_ratio=4,
            with_indexer=True,
        )
    )
    main_wkv = _random((1_024, hidden), torch.bfloat16, 1 / 64)
    main_wgate = _random((1_024, hidden), torch.bfloat16, 1 / 64)
    main_ape = _random((4, 1_024), torch.float32, 1 / 16)
    main_norm = (_random((512,), torch.bfloat16, 1 / 16) + 1).contiguous()
    index_wkv = _random((256, hidden), torch.bfloat16, 1 / 64)
    index_wgate = _random((256, hidden), torch.bfloat16, 1 / 64)
    index_ape = _random((4, 256), torch.float32, 1 / 16)
    index_norm = (_random((128,), torch.bfloat16, 1 / 16) + 1).contiguous()
    weights = dsv4_compressor.pack_weights(
        main_wkv,
        main_wgate,
        main_ape,
        main_norm,
        index_wkv=index_wkv,
        index_wgate=index_wgate,
        index_ape=index_ape,
        index_norm=index_norm,
    )
    hidden_states = _random((32, hidden), torch.bfloat16)
    projection = F.linear(hidden_states, weights.joint_projection)
    active_groups = torch.tensor([3], device="cuda", dtype=torch.int32)
    group_starts = torch.tensor(
        [0, 4, 10, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    group_positions = torch.tensor(
        [0, 4, 0, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    slots = torch.tensor(
        [0, 1, 2, 63, 63, 63, 63, 63], device="cuda", dtype=torch.int32
    )
    active_sequences = torch.tensor([3], device="cuda", dtype=torch.int32)
    sequence_offsets = torch.tensor(
        [0, 10, 17, 20, 20], device="cuda", dtype=torch.int32
    )
    state_sequence_ids = torch.tensor([1, 0, 2, 3], device="cuda", dtype=torch.int32)
    cos_sin = _cos_sin(32)
    main_cache = torch.zeros((1, 37_440), device="cuda", dtype=torch.uint8)
    index_cache = torch.zeros((1, 8_448), device="cuda", dtype=torch.uint8)
    main_kv_state = _random((4, 8, 1_024), torch.float32)
    main_score_state = _random((4, 8, 1_024), torch.float32)
    index_kv_state = _random((4, 8, 256), torch.float32)
    index_score_state = _random((4, 8, 256), torch.float32)
    expected_main_kv = main_kv_state.clone()
    expected_main_score = main_score_state.clone()
    expected_index_kv = index_kv_state.clone()
    expected_index_score = index_score_state.clone()

    for sequence_slot, (source_start, source_end) in enumerate(
        ((0, 10), (10, 17), (17, 20))
    ):
        state_sequence = (1, 0, 2)[sequence_slot]
        source_tokens = source_end - source_start
        cutoff = source_tokens // 4 * 4
        remainder = source_tokens - cutoff
        for expected_kv, expected_score, offset, width, ape in (
            (expected_main_kv, expected_main_score, 0, 1_024, main_ape),
            (expected_index_kv, expected_index_score, 2_048, 256, index_ape),
        ):
            expected_kv[state_sequence].zero_()
            expected_score[state_sequence].fill_(float("-inf"))
            if cutoff >= 4:
                previous = projection[
                    source_start + cutoff - 4 : source_start + cutoff,
                    offset : offset + width,
                ]
                previous_score = projection[
                    source_start + cutoff - 4 : source_start + cutoff,
                    offset + width : offset + 2 * width,
                ]
                expected_kv[state_sequence, :4] = previous.float()
                expected_score[state_sequence, :4] = previous_score.float() + ape
            if remainder:
                current = projection[
                    source_start + cutoff : source_end, offset : offset + width
                ]
                current_score = projection[
                    source_start + cutoff : source_end,
                    offset + width : offset + 2 * width,
                ]
                expected_kv[state_sequence, 4 : 4 + remainder] = current.float()
                expected_score[state_sequence, 4 : 4 + remainder] = (
                    current_score.float() + ape[:remainder]
                )

    expected_main_rows = []
    expected_index_rows = []
    for source_start, rope_position in ((0, 0), (4, 4), (10, 0)):
        main_pooled = _initial_prefill_pool(
            projection[:, :2_048],
            main_ape,
            source_start=source_start,
            rope_position=rope_position,
            head_dim=512,
        )
        expected_main_rows.append(
            _rope_forward(
                _rmsnorm(main_pooled, main_norm, eps),
                rope_position,
                cos_sin,
                nope_dim=448,
            )
        )
        index_pooled = _initial_prefill_pool(
            projection[:, 2_048:],
            index_ape,
            source_start=source_start,
            rope_position=rope_position,
            head_dim=128,
        )
        expected_index = _rope_forward(
            _rmsnorm(index_pooled, index_norm, eps),
            rope_position,
            cos_sin,
            nope_dim=64,
        )
        expected_index_rows.append(_fp4_qat_reference(_fwht_128(expected_index)))
    expected_main_rows = torch.stack(expected_main_rows)
    expected_index_rows = torch.stack(expected_index_rows)
    expected_main_cache = pack_compressed_mla_kv_cache_reference(
        expected_main_rows[:, :448],
        expected_main_rows[:, 448:],
        page_size=64,
        num_pages=1,
    )
    expected_index_cache = pack_index_k_cache_reference(expected_index_rows)

    scratch = torch.empty(
        plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
    )
    binding = dsv4_compressor.bind_prefill(
        plan,
        scratch=scratch,
        hidden_states=hidden_states,
        active_groups=active_groups,
        group_source_starts=group_starts,
        group_rope_positions=group_positions,
        compressed_slots=slots,
        active_sequences=active_sequences,
        sequence_offsets=sequence_offsets,
        state_sequence_ids=state_sequence_ids,
        compressed_cos_sin_cache=cos_sin,
        compressed_main_cache=main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        weights=weights,
        index_cache=index_cache,
        index_kv_state=index_kv_state,
        index_score_state=index_score_state,
        eps=eps,
        initial_prefill=True,
    )
    dsv4_compressor.run_prefill(binding=binding)
    torch.cuda.synchronize()

    assert torch.equal(main_kv_state, expected_main_kv)
    assert torch.equal(main_score_state, expected_main_score)
    assert torch.equal(index_kv_state, expected_index_kv)
    assert torch.equal(index_score_state, expected_index_score)
    actual_nope, actual_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=64, n_tokens=3
    )
    expected_nope, expected_rope = unpack_compressed_mla_kv_cache_reference(
        expected_main_cache, page_size=64, n_tokens=3
    )
    torch.testing.assert_close(actual_nope, expected_nope, atol=0.04, rtol=0.04)
    torch.testing.assert_close(actual_rope, expected_rope, atol=0.02, rtol=0.02)
    index_actual = unpack_index_k_cache_reference(
        index_cache, num_tokens=3, page_size=64
    )
    index_expected = unpack_index_k_cache_reference(
        expected_index_cache, num_tokens=3, page_size=64
    )
    index_error = (index_actual - index_expected).abs()
    assert int((index_error > 0.04).sum()) <= 3
    assert float(index_error.max()) <= 0.25


def test_c128_initial_prefill_graph_replays_and_retains_only_remainder() -> None:
    require_sparkinfer()
    torch.manual_seed(20260807)
    ratio, hidden, eps = 128, 4_096, 1.0e-6
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=384,
            hidden=hidden,
            compress_ratio=ratio,
            with_indexer=False,
        )
    )
    main_wkv = _random((512, hidden), torch.bfloat16, 1 / 64)
    main_wgate = _random((512, hidden), torch.bfloat16, 1 / 64)
    main_ape = _random((ratio, 512), torch.float32, 1 / 16)
    main_norm = (_random((512,), torch.bfloat16, 1 / 16) + 1).contiguous()
    weights = dsv4_compressor.pack_weights(main_wkv, main_wgate, main_ape, main_norm)
    hidden_states = _random((384, hidden), torch.bfloat16)
    projection = F.linear(hidden_states, weights.joint_projection)
    active_groups = torch.tensor([2], device="cuda", dtype=torch.int32)
    group_starts = torch.tensor([0, 128, 0], device="cuda", dtype=torch.int32)
    group_positions = torch.tensor([0, 128, 0], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)
    active_sequences = torch.tensor([1], device="cuda", dtype=torch.int32)
    sequence_offsets = torch.tensor([0, 260, 260], device="cuda", dtype=torch.int32)
    state_sequence_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cos_sin = _cos_sin(384)
    main_cache = torch.zeros((1, 1_728), device="cuda", dtype=torch.uint8)
    kv_state = _random((2, 128, 512), torch.float32)
    score_state = _random((2, 128, 512), torch.float32)
    expected_kv = kv_state.clone()
    expected_score = score_state.clone()
    expected_kv[0].zero_()
    expected_score[0].fill_(float("-inf"))
    expected_kv[0, :4] = projection[256:260, :512].float()
    expected_score[0, :4] = projection[256:260, 512:].float() + main_ape[:4]
    expected_rows = []
    for source_start, rope_position in ((0, 0), (128, 128)):
        pooled = _initial_prefill_pool(
            projection,
            main_ape,
            source_start=source_start,
            rope_position=rope_position,
            head_dim=512,
        )
        expected_rows.append(
            _rope_forward(
                _rmsnorm(pooled, main_norm, eps),
                rope_position,
                cos_sin,
                nope_dim=448,
            )
        )
    expected_rows = torch.stack(expected_rows)
    expected_cache = pack_compressed_mla_kv_cache_reference(
        expected_rows[:, :448],
        expected_rows[:, 448:],
        page_size=2,
        num_pages=1,
    )

    scratch = torch.empty(
        plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
    )
    binding = dsv4_compressor.bind_prefill(
        plan,
        scratch=scratch,
        hidden_states=hidden_states,
        active_groups=active_groups,
        group_source_starts=group_starts,
        group_rope_positions=group_positions,
        compressed_slots=slots,
        active_sequences=active_sequences,
        sequence_offsets=sequence_offsets,
        state_sequence_ids=state_sequence_ids,
        compressed_cos_sin_cache=cos_sin,
        compressed_main_cache=main_cache,
        main_kv_state=kv_state,
        main_score_state=score_state,
        weights=weights,
        eps=eps,
        initial_prefill=True,
    )
    dsv4_compressor.run_prefill(binding=binding)
    torch.cuda.synchronize()

    assert torch.equal(kv_state, expected_kv)
    assert torch.equal(score_state, expected_score)
    actual_nope, actual_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=2, n_tokens=2
    )
    expected_nope, expected_rope = unpack_compressed_mla_kv_cache_reference(
        expected_cache, page_size=2, n_tokens=2
    )
    torch.testing.assert_close(actual_nope, expected_nope, atol=0.04, rtol=0.04)
    torch.testing.assert_close(actual_rope, expected_rope, atol=0.02, rtol=0.02)

    eager_cache = main_cache.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_compressor.run_prefill(binding=binding)
    main_cache.zero_()
    active_groups.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.count_nonzero(main_cache[0, 576:1_152])
    assert not torch.count_nonzero(main_cache[0, 1_160:1_168])
    one_nope, one_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=2, n_tokens=1
    )
    torch.testing.assert_close(one_nope, expected_nope[:1], atol=0.04, rtol=0.04)
    torch.testing.assert_close(one_rope, expected_rope[:1], atol=0.02, rtol=0.02)

    active_groups.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(main_cache, eager_cache)


def test_c4_ordered_continuation_combines_carried_state_and_chunk_projection() -> None:
    require_sparkinfer()
    torch.manual_seed(20260808)
    hidden, eps = 4_096, 1.0e-6
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=12,
            hidden=hidden,
            compress_ratio=4,
            with_indexer=True,
        )
    )
    main_wkv = _random((1_024, hidden), torch.bfloat16, 1 / 64)
    main_wgate = _random((1_024, hidden), torch.bfloat16, 1 / 64)
    main_ape = _random((4, 1_024), torch.float32, 1 / 16)
    main_norm = (_random((512,), torch.bfloat16, 1 / 16) + 1).contiguous()
    index_wkv = _random((256, hidden), torch.bfloat16, 1 / 64)
    index_wgate = _random((256, hidden), torch.bfloat16, 1 / 64)
    index_ape = _random((4, 256), torch.float32, 1 / 16)
    index_norm = (_random((128,), torch.bfloat16, 1 / 16) + 1).contiguous()
    weights = dsv4_compressor.pack_weights(
        main_wkv,
        main_wgate,
        main_ape,
        main_norm,
        index_wkv=index_wkv,
        index_wgate=index_wgate,
        index_ape=index_ape,
        index_norm=index_norm,
    )

    prefix_hidden = (
        _random((6, hidden), torch.bfloat16),
        _random((3, hidden), torch.bfloat16),
        _random((6, hidden), torch.bfloat16),
    )
    chunk_hidden = (
        _random((7, hidden), torch.bfloat16),
        _random((1, hidden), torch.bfloat16),
        _random((1, hidden), torch.bfloat16),
    )
    hidden_states = torch.cat(
        (*chunk_hidden, torch.zeros((3, hidden), device="cuda", dtype=torch.bfloat16))
    ).contiguous()
    prefix_projection = tuple(
        F.linear(rows, weights.joint_projection) for rows in prefix_hidden
    )
    bucket_projection = F.linear(hidden_states, weights.joint_projection)
    chunk_projection = (
        bucket_projection[:7],
        bucket_projection[7:8],
        bucket_projection[8:9],
    )
    full_projection = tuple(
        torch.cat((prefix, chunk), dim=0)
        for prefix, chunk in zip(prefix_projection, chunk_projection, strict=True)
    )

    main_kv_state = _random((4, 8, 1_024), torch.float32)
    main_score_state = _random((4, 8, 1_024), torch.float32)
    index_kv_state = _random((4, 8, 256), torch.float32)
    index_score_state = _random((4, 8, 256), torch.float32)
    state_ids = (1, 0, 2)
    for sequence_slot, state_sequence in enumerate(state_ids):
        prefix = prefix_projection[sequence_slot]
        main_state = _state_from_history(prefix[:, :2_048], main_ape, ratio=4)
        index_state = _state_from_history(prefix[:, 2_048:], index_ape, ratio=4)
        main_kv_state[state_sequence], main_score_state[state_sequence] = main_state
        index_kv_state[state_sequence], index_score_state[state_sequence] = index_state
    initial_main_kv = main_kv_state.clone()
    initial_main_score = main_score_state.clone()
    initial_index_kv = index_kv_state.clone()
    initial_index_score = index_score_state.clone()
    expected_main_kv = main_kv_state.clone()
    expected_main_score = main_score_state.clone()
    expected_index_kv = index_kv_state.clone()
    expected_index_score = index_score_state.clone()
    for sequence_slot, state_sequence in enumerate(state_ids):
        full = full_projection[sequence_slot]
        main_state = _state_from_history(full[:, :2_048], main_ape, ratio=4)
        index_state = _state_from_history(full[:, 2_048:], index_ape, ratio=4)
        expected_main_kv[state_sequence], expected_main_score[state_sequence] = (
            main_state
        )
        expected_index_kv[state_sequence], expected_index_score[state_sequence] = (
            index_state
        )

    cos_sin = _cos_sin(32)
    main_cache = torch.zeros((1, 37_440), device="cuda", dtype=torch.uint8)
    index_cache = torch.zeros((1, 8_448), device="cuda", dtype=torch.uint8)
    output_specs = ((1, 0, 0), (0, 4, 1), (0, 8, 2))
    expected_main_rows = []
    expected_index_rows = []
    for sequence_slot, group_start, _slot in output_specs:
        full = full_projection[sequence_slot]
        main_pooled = _initial_prefill_pool(
            full[:, :2_048],
            main_ape,
            source_start=group_start,
            rope_position=group_start,
            head_dim=512,
        )
        expected_main_rows.append(
            _rope_forward(
                _rmsnorm(main_pooled, main_norm, eps),
                group_start,
                cos_sin,
                nope_dim=448,
            )
        )
        index_pooled = _initial_prefill_pool(
            full[:, 2_048:],
            index_ape,
            source_start=group_start,
            rope_position=group_start,
            head_dim=128,
        )
        index_output = _rope_forward(
            _rmsnorm(index_pooled, index_norm, eps),
            group_start,
            cos_sin,
            nope_dim=64,
        )
        expected_index_rows.append(_fp4_qat_reference(_fwht_128(index_output)))
    expected_main_rows = torch.stack(expected_main_rows)
    expected_index_rows = torch.stack(expected_index_rows)
    expected_main_cache = pack_compressed_mla_kv_cache_reference(
        expected_main_rows[:, :448],
        expected_main_rows[:, 448:],
        page_size=64,
        num_pages=1,
    )
    expected_index_cache = pack_index_k_cache_reference(expected_index_rows)

    binding = dsv4_compressor.bind_continuation(
        plan,
        scratch=torch.empty(
            plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
        ),
        hidden_states=hidden_states,
        active_groups=torch.tensor([3], device="cuda", dtype=torch.int32),
        group_sequence_slots=torch.tensor(
            [1, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32
        ),
        group_rope_positions=torch.tensor(
            [0, 4, 8, 0, 0, 0], device="cuda", dtype=torch.int32
        ),
        compressed_slots=torch.tensor(
            [0, 1, 2, 63, 63, 63], device="cuda", dtype=torch.int32
        ),
        active_sequences=torch.tensor([3], device="cuda", dtype=torch.int32),
        sequence_offsets=torch.tensor(
            [0, 7, 8, 9, 9], device="cuda", dtype=torch.int32
        ),
        sequence_start_positions=torch.tensor(
            [6, 3, 6, 0], device="cuda", dtype=torch.int32
        ),
        state_sequence_ids=torch.tensor([1, 0, 2, 3], device="cuda", dtype=torch.int32),
        compressed_cos_sin_cache=cos_sin,
        compressed_main_cache=main_cache,
        main_kv_state=main_kv_state,
        main_score_state=main_score_state,
        weights=weights,
        index_cache=index_cache,
        index_kv_state=index_kv_state,
        index_score_state=index_score_state,
        eps=eps,
        ordered_continuation=True,
    )
    dsv4_compressor.run_continuation(binding=binding)
    torch.cuda.synchronize()

    torch.testing.assert_close(main_kv_state, expected_main_kv, atol=0, rtol=0)
    torch.testing.assert_close(main_score_state, expected_main_score, atol=0, rtol=0)
    torch.testing.assert_close(index_kv_state, expected_index_kv, atol=0, rtol=0)
    torch.testing.assert_close(index_score_state, expected_index_score, atol=0, rtol=0)
    actual_nope, actual_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=64, n_tokens=3
    )
    expected_nope, expected_rope = unpack_compressed_mla_kv_cache_reference(
        expected_main_cache, page_size=64, n_tokens=3
    )
    torch.testing.assert_close(actual_nope, expected_nope, atol=0.04, rtol=0.04)
    torch.testing.assert_close(actual_rope, expected_rope, atol=0.02, rtol=0.02)
    index_actual = unpack_index_k_cache_reference(
        index_cache, num_tokens=3, page_size=64
    )
    index_expected = unpack_index_k_cache_reference(
        expected_index_cache, num_tokens=3, page_size=64
    )
    index_error = (index_actual - index_expected).abs()
    # A continuation can combine state-resident and projected rows in a
    # different FP32 reduction order. One symmetric FWHT pair may cross a
    # discontinuous E2M1 threshold; every other lane remains identical.
    assert int((index_error > 0.04).sum()) <= 3
    assert float(index_error.max()) <= 0.30

    eager_main_cache = main_cache.clone()
    eager_index_cache = index_cache.clone()
    main_kv_state.copy_(initial_main_kv)
    main_score_state.copy_(initial_main_score)
    index_kv_state.copy_(initial_index_kv)
    index_score_state.copy_(initial_index_score)
    main_cache.zero_()
    index_cache.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_compressor.run_continuation(binding=binding)
    main_kv_state.copy_(initial_main_kv)
    main_score_state.copy_(initial_main_score)
    index_kv_state.copy_(initial_index_kv)
    index_score_state.copy_(initial_index_score)
    main_cache.zero_()
    index_cache.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(main_kv_state, expected_main_kv)
    assert torch.equal(main_score_state, expected_main_score)
    assert torch.equal(index_kv_state, expected_index_kv)
    assert torch.equal(index_score_state, expected_index_score)
    assert torch.equal(main_cache, eager_main_cache)
    assert torch.equal(index_cache, eager_index_cache)


def test_c128_ordered_continuation_crosses_boundary_and_graph_replays() -> None:
    require_sparkinfer()
    torch.manual_seed(20260809)
    ratio, hidden, eps = 128, 4_096, 1.0e-6
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=32,
            hidden=hidden,
            compress_ratio=ratio,
            with_indexer=False,
        )
    )
    main_wkv = _random((512, hidden), torch.bfloat16, 1 / 64)
    main_wgate = _random((512, hidden), torch.bfloat16, 1 / 64)
    main_ape = _random((ratio, 512), torch.float32, 1 / 16)
    main_norm = (_random((512,), torch.bfloat16, 1 / 16) + 1).contiguous()
    weights = dsv4_compressor.pack_weights(main_wkv, main_wgate, main_ape, main_norm)
    prefix_hidden = (
        _random((250, hidden), torch.bfloat16),
        _random((260, hidden), torch.bfloat16),
    )
    chunk_hidden = (
        _random((20, hidden), torch.bfloat16),
        _random((4, hidden), torch.bfloat16),
    )
    hidden_states = torch.cat(
        (*chunk_hidden, torch.zeros((8, hidden), device="cuda", dtype=torch.bfloat16))
    ).contiguous()
    prefix_projection = tuple(
        F.linear(rows, weights.joint_projection) for rows in prefix_hidden
    )
    bucket_projection = F.linear(hidden_states, weights.joint_projection)
    chunk_projection = (bucket_projection[:20], bucket_projection[20:24])
    full_projection = tuple(
        torch.cat((prefix, chunk), dim=0)
        for prefix, chunk in zip(prefix_projection, chunk_projection, strict=True)
    )

    kv_state = _random((3, 128, 512), torch.float32)
    score_state = _random((3, 128, 512), torch.float32)
    for sequence_slot in range(2):
        kv, score = _state_from_history(
            prefix_projection[sequence_slot], main_ape, ratio=ratio
        )
        kv_state[sequence_slot], score_state[sequence_slot] = kv, score
    initial_kv = kv_state.clone()
    initial_score = score_state.clone()
    expected_kv = kv_state.clone()
    expected_score = score_state.clone()
    for sequence_slot in range(2):
        kv, score = _state_from_history(
            full_projection[sequence_slot], main_ape, ratio=ratio
        )
        expected_kv[sequence_slot], expected_score[sequence_slot] = kv, score

    cos_sin = _cos_sin(384)
    main_cache = torch.zeros((1, 1_728), device="cuda", dtype=torch.uint8)
    pooled = _initial_prefill_pool(
        full_projection[0],
        main_ape,
        source_start=128,
        rope_position=128,
        head_dim=512,
    )
    expected_output = _rope_forward(
        _rmsnorm(pooled, main_norm, eps), 128, cos_sin, nope_dim=448
    )
    expected_cache = pack_compressed_mla_kv_cache_reference(
        expected_output[None, :448],
        expected_output[None, 448:],
        page_size=2,
        num_pages=1,
    )

    binding = dsv4_compressor.bind_continuation(
        plan,
        scratch=torch.empty(
            plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
        ),
        hidden_states=hidden_states,
        active_groups=torch.tensor([1], device="cuda", dtype=torch.int32),
        group_sequence_slots=torch.tensor(
            [0, 0, 0, 0], device="cuda", dtype=torch.int32
        ),
        group_rope_positions=torch.tensor(
            [128, 0, 0, 0], device="cuda", dtype=torch.int32
        ),
        compressed_slots=torch.tensor([0, 1, 1, 1], device="cuda", dtype=torch.int32),
        active_sequences=torch.tensor([2], device="cuda", dtype=torch.int32),
        sequence_offsets=torch.tensor(
            [0, 20, 24, 24], device="cuda", dtype=torch.int32
        ),
        sequence_start_positions=torch.tensor(
            [250, 260, 0], device="cuda", dtype=torch.int32
        ),
        state_sequence_ids=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        compressed_cos_sin_cache=cos_sin,
        compressed_main_cache=main_cache,
        main_kv_state=kv_state,
        main_score_state=score_state,
        weights=weights,
        eps=eps,
        ordered_continuation=True,
    )
    dsv4_compressor.run_continuation(binding=binding)
    torch.cuda.synchronize()

    torch.testing.assert_close(kv_state, expected_kv, atol=0, rtol=0)
    torch.testing.assert_close(score_state, expected_score, atol=0, rtol=0)
    actual_nope, actual_rope = unpack_compressed_mla_kv_cache_reference(
        main_cache, page_size=2, n_tokens=1
    )
    expected_nope, expected_rope = unpack_compressed_mla_kv_cache_reference(
        expected_cache, page_size=2, n_tokens=1
    )
    torch.testing.assert_close(actual_nope, expected_nope, atol=0.04, rtol=0.04)
    torch.testing.assert_close(actual_rope, expected_rope, atol=0.02, rtol=0.02)

    eager_cache = main_cache.clone()
    kv_state.copy_(initial_kv)
    score_state.copy_(initial_score)
    main_cache.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_compressor.run_continuation(binding=binding)
    kv_state.copy_(initial_kv)
    score_state.copy_(initial_score)
    main_cache.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(main_cache, eager_cache)
    assert torch.equal(kv_state, expected_kv)
    assert torch.equal(score_state, expected_score)


def test_decode_binding_requires_explicit_sequence_unique_contract() -> None:
    require_sparkinfer()
    plan = dsv4_compressor.plan(
        dsv4_compressor.Caps(
            device="cuda",
            max_tokens=1,
            hidden=4_096,
            compress_ratio=128,
            with_indexer=False,
        )
    )
    weights = dsv4_compressor.pack_weights(
        torch.zeros((512, 4_096), device="cuda", dtype=torch.bfloat16),
        torch.zeros((512, 4_096), device="cuda", dtype=torch.bfloat16),
        torch.zeros((128, 512), device="cuda", dtype=torch.float32),
        torch.ones((512,), device="cuda", dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="sequence-unique"):
        dsv4_compressor.bind_decode(
            plan,
            scratch=torch.empty(
                plan.scratch_specs()[0].shape, device="cuda", dtype=torch.uint8
            ),
            hidden_states=torch.zeros((1, 4_096), device="cuda", dtype=torch.bfloat16),
            positions=torch.zeros((1,), device="cuda", dtype=torch.int32),
            sequence_ids=torch.zeros((1,), device="cuda", dtype=torch.int32),
            compressed_slots=torch.zeros((1,), device="cuda", dtype=torch.int32),
            compressed_cos_sin_cache=torch.zeros(
                (128, 64), device="cuda", dtype=torch.float32
            ),
            compressed_main_cache=torch.zeros(
                (1, 1_728), device="cuda", dtype=torch.uint8
            ),
            main_kv_state=torch.zeros(
                (1, 128, 512), device="cuda", dtype=torch.float32
            ),
            main_score_state=torch.full(
                (1, 128, 512),
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            ),
            weights=weights,
        )
