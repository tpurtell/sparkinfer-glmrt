"""attention.sparse_mla: decode over top-k-selected tokens vs the in-tree
pure-torch reference (packed NSA MLA cache layout), through the public
plan -> bind -> run_decode lifecycle, including -1-padded selections.
"""

from __future__ import annotations

import torch

from b12x.attention import compressed_sparse_mla as compressed_mla
from b12x.attention import sparse_mla
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference as pack_compressed_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)

from ..conftest import require_b12x

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
V_HEAD_DIM = NOPE_DIM


def _make_case(*, rows: int, heads: int, cache_tokens: int, width: int):
    torch.manual_seed(20260716)
    k_nope = (
        torch.randn(cache_tokens, NOPE_DIM, device="cuda", dtype=torch.bfloat16) / 4
    )
    k_rope = (
        torch.randn(cache_tokens, ROPE_DIM, device="cuda", dtype=torch.bfloat16) / 4
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope, k_rope)
    q_all = torch.randn(rows, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16) / 4

    selected = torch.stack(
        [
            torch.randperm(cache_tokens, device="cuda")[:width].sort().values
            for _ in range(rows)
        ]
    ).to(torch.int32)
    cache_seqlens = torch.full((rows,), cache_tokens, dtype=torch.int32, device="cuda")
    active = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    return q_all, kv_cache, selected, cache_seqlens, active


def _run_public_decode(q_all, kv_cache, selected, cache_seqlens, active, *, width):
    rows, heads, _ = q_all.shape
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=q_all.device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            kv_dtype=torch.uint8,  # packed NSA byte cache (fp8+scale+rope)
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q_all.device)
    binding = sparse_mla.bind(
        plan,
        scratch=scratch,
        q=q_all,
        selected_indices=selected,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active,
    )
    sm_scale = HEAD_DIM**-0.5
    out = sparse_mla.run_decode(
        binding=binding,
        kv_cache=kv_cache,
        sm_scale=sm_scale,
        v_head_dim=V_HEAD_DIM,
    )
    ref = sparse_mla_reference(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=sm_scale,
        v_head_dim=V_HEAD_DIM,
    )
    return out, ref


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> None:
    torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all().item())
    assert int(torch.count_nonzero(out).item()) > 0
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    assert float(cosine.item()) > 0.99, f"cosine {float(cosine.item()):.5f}"
    torch.testing.assert_close(
        out.float(), ref.to(out.dtype).float(), rtol=5e-2, atol=5e-2
    )


def test_run_decode_matches_reference() -> None:
    require_b12x()
    q, kv, sel, lens, active = _make_case(rows=4, heads=16, cache_tokens=512, width=128)
    out, ref = _run_public_decode(q, kv, sel, lens, active, width=128)
    _assert_matches(out, ref)


def test_run_decode_masks_padded_selection() -> None:
    require_b12x()
    width = 128
    q, kv, sel, lens, active = _make_case(
        rows=4, heads=16, cache_tokens=512, width=width
    )
    # Invalidate the back half of every row's selection: pad with -1 and
    # shrink the active counts to match; the kernel and reference must both
    # attend only to the front half.
    sel[:, width // 2 :] = -1
    active.fill_(width // 2)
    out, ref = _run_public_decode(q, kv, sel, lens, active, width=width)
    _assert_matches(out, ref)


def test_dsv4_dspark_decode_attends_target_ring_and_all_proposal_kv() -> None:
    require_b12x()
    torch.manual_seed(20260805)
    rows = 5
    heads = 64
    target_ring = 128
    proposal_tokens = 5
    width = target_ring + proposal_tokens
    page_size = 256
    k_nope = torch.randn(
        page_size, 448, device="cuda", dtype=torch.bfloat16
    ) / 4
    k_rope = torch.randn(
        page_size, 64, device="cuda", dtype=torch.bfloat16
    ) / 4
    kv_cache = pack_compressed_mla_kv_cache_reference(
        k_nope, k_rope, page_size=page_size
    )
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) / 4
    selected = torch.cat(
        (
            torch.arange(target_ring, device="cuda"),
            torch.arange(target_ring, width, device="cuda"),
        )
    ).to(torch.int32)
    selected = selected.view(1, width).expand(rows, width).contiguous()
    active = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    plan = compressed_mla.plan(
        compressed_mla.Caps(
            device="cuda",
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            kv_dtype=torch.uint8,
            head_dim=512,
            v_head_dim=512,
            max_chunks_per_row=12,
            page_size=page_size,
        )
    )
    (scratch_spec,) = plan.scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    binding = compressed_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        swa_indices=selected,
        swa_lengths=active,
    )
    out = compressed_mla.run(
        binding=binding,
        swa_k_cache=kv_cache,
        sm_scale=512**-0.5,
        swa_page_size=page_size,
    )
    ref = compressed_sparse_mla_reference(
        q=q,
        swa_k_cache=kv_cache,
        swa_indices=selected,
        swa_topk_lengths=active,
        sm_scale=512**-0.5,
        swa_page_size=page_size,
    )
    _assert_matches(out, ref)
