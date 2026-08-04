from __future__ import annotations

import torch

from sparkinfer.attention import dsv4_producer
from sparkinfer.attention._shared.mla.compressed_reference import (
    pack_compressed_mla_kv_cache_reference,
)
from sparkinfer.attention.dsv4_producer._impl import (
    DSV4_KV_PAGE_BYTES,
    _run_normalize_query_rope,
    _run_normalize_rank_pack_kv,
)

from ..conftest import require_sparkinfer


def _rope_forward(
    values: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    nope_dim: int,
) -> torch.Tensor:
    output = values.clone()
    rope = values[..., nope_dim:].float()
    pairs = rope.unflatten(-1, (-1, 2))
    selected = cos_sin[positions.long()]
    cos_v, sin_v = selected.chunk(2, dim=-1)
    even = pairs[..., 0] * cos_v - pairs[..., 1] * sin_v
    odd = pairs[..., 1] * cos_v + pairs[..., 0] * sin_v
    output[..., nope_dim:] = torch.stack((even, odd), dim=-1).flatten(-2).to(
        torch.bfloat16
    )
    return output


def _rmsnorm(
    values: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + eps
    )
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(torch.bfloat16)


def _block_fp8_weight(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) / 16
    ).to(torch.float8_e4m3fn)
    scale = torch.full(
        (rows // 128, cols // 128),
        127,
        device="cuda",
        dtype=torch.uint8,
    ).view(torch.float8_e8m0fnu)
    return weight, scale


def test_plan_pins_flash_and_pro_caller_owned_workspace() -> None:
    flash = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cpu",
            max_tokens=2048,
            hidden=4096,
            q_lora_rank=1024,
            heads=64,
        )
    )
    pro = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cpu",
            max_tokens=2048,
            hidden=7168,
            q_lora_rank=1536,
            heads=128,
        )
    )

    assert flash.scratch_specs()[0].name == "dsv4_producer.scratch"
    assert flash.scratch_specs()[0].nbytes == 21_626_880
    assert pro.scratch_specs()[0].nbytes == 33_619_968
    assert flash.layout.qkv_linear_offset == 0
    assert flash.layout.q_linear_offset % 1024 == 0
    assert flash.layout.qkv_output_offset % 1024 == 0
    assert flash.layout.q_rank_offset % 1024 == 0


def test_caps_reject_absorbed_mla_and_non_native_page_geometry() -> None:
    for kwargs in (
        {"hidden": 4096, "q_lora_rank": 1024, "heads": 8},
        {"hidden": 4096, "q_lora_rank": 512, "heads": 64},
        {"hidden": 4096, "q_lora_rank": 1024, "heads": 64, "head_dim": 576},
        {"hidden": 4096, "q_lora_rank": 1024, "heads": 64, "page_size": 64},
    ):
        try:
            dsv4_producer.Caps(device="cpu", max_tokens=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid DSV4 producer geometry was accepted: {kwargs}")


def test_rank_norm_and_main_kv_pack_match_checkpoint_reference() -> None:
    require_sparkinfer()
    torch.manual_seed(20260804)
    tokens, q_rank, eps = 2, 1024, 1.0e-6
    qkv = (
        torch.randn((tokens, q_rank + 512), device="cuda", dtype=torch.bfloat16)
        / 3
    ).contiguous()
    q_norm = (
        torch.randn((q_rank,), device="cuda", dtype=torch.bfloat16) / 8 + 1
    ).contiguous()
    kv_norm = (
        torch.randn((512,), device="cuda", dtype=torch.bfloat16) / 8 + 1
    ).contiguous()
    positions = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    angles = torch.randn((4, 32), device="cuda", dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
    slots = torch.tensor([0, 256], device="cuda", dtype=torch.int64)
    q_rank_out = torch.empty((tokens, q_rank), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((2, DSV4_KV_PAGE_BYTES), device="cuda", dtype=torch.uint8)

    _run_normalize_rank_pack_kv(
        qkv,
        q_norm,
        kv_norm,
        positions,
        slots,
        cos_sin,
        q_rank_out,
        cache,
        eps=eps,
    )
    torch.cuda.synchronize()

    expected_q_rank = _rmsnorm(qkv[:, :q_rank], q_norm, eps)
    expected_kv = _rmsnorm(qkv[:, q_rank:], kv_norm, eps)
    expected_kv = _rope_forward(expected_kv, positions, cos_sin, nope_dim=448)
    torch.testing.assert_close(q_rank_out, expected_q_rank, rtol=0, atol=0)
    for token in range(tokens):
        expected_page = pack_compressed_mla_kv_cache_reference(
            expected_kv[token : token + 1, :448],
            expected_kv[token : token + 1, 448:],
            page_size=256,
            num_pages=1,
        )
        torch.testing.assert_close(cache[token], expected_page[0], rtol=0, atol=0)


def test_query_post_matches_per_head_rmsnorm_and_partial_rope() -> None:
    require_sparkinfer()
    torch.manual_seed(20260805)
    tokens, heads, eps = 2, 64, 1.0e-6
    query = (
        torch.randn((tokens, heads, 512), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    original = query.clone()
    positions = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    angles = torch.randn((4, 32), device="cuda", dtype=torch.float32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()

    _run_normalize_query_rope(query, positions, cos_sin, eps=eps)
    torch.cuda.synchronize()

    expected = _rmsnorm(original, None, eps)
    expanded_positions = positions[:, None].expand(tokens, heads).reshape(-1)
    expected = _rope_forward(
        expected.reshape(tokens * heads, 512),
        expanded_positions,
        cos_sin,
        nope_dim=448,
    ).reshape(tokens, heads, 512)
    torch.testing.assert_close(query, expected, rtol=0, atol=0)


def test_full_flash_plan_replays_without_serving_allocations() -> None:
    require_sparkinfer()
    torch.manual_seed(20260806)
    hidden, q_rank, heads, tokens = 4096, 1024, 64, 1
    wq_a, wq_a_scale = _block_fp8_weight(q_rank, hidden)
    wq_b, wq_b_scale = _block_fp8_weight(heads * 512, q_rank)
    wkv, wkv_scale = _block_fp8_weight(512, hidden)
    q_norm = torch.ones((q_rank,), device="cuda", dtype=torch.bfloat16)
    kv_norm = torch.ones((512,), device="cuda", dtype=torch.bfloat16)
    weights = dsv4_producer.pack_weights(
        wq_a,
        wq_a_scale,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        q_norm,
        kv_norm,
    )
    plan = dsv4_producer.plan(
        dsv4_producer.Caps(
            device="cuda",
            max_tokens=tokens,
            hidden=hidden,
            q_lora_rank=q_rank,
            heads=heads,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    hidden_states = (
        torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.int64)
    slots = torch.zeros((tokens,), device="cuda", dtype=torch.int64)
    cos_sin = torch.zeros((1, 64), device="cuda", dtype=torch.float32)
    cos_sin[:, :32] = 1
    cache = torch.zeros((1, DSV4_KV_PAGE_BYTES), device="cuda", dtype=torch.uint8)
    query = torch.empty((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)
    binding = dsv4_producer.bind(
        plan,
        scratch=scratch,
        hidden_states=hidden_states,
        positions=positions,
        main_slots=slots,
        cos_sin_cache=cos_sin,
        main_kv_cache=cache,
        query=query,
        weights=weights,
        expected_m=tokens,
    )

    dsv4_producer.run(binding=binding)
    torch.cuda.synchronize()
    eager_query = query.clone()
    eager_cache = cache.clone()
    assert bool(torch.isfinite(eager_query).all())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_producer.run(binding=binding)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(query, eager_query, rtol=0, atol=0)
    torch.testing.assert_close(cache, eager_cache, rtol=0, atol=0)
