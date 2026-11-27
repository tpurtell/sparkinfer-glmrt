"""Bitwise oracle, per-row split-K batch-invariance, and verify-vs-decode checks for Inkling global attention."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x


def _pack_and_dequant_fp8(k: torch.Tensor, v: torch.Tensor, g):
    """Pack BF16 K/V [context, kv_heads, 128] into native FP8 records and return (fp8_cache, bf16_dequant_cache)."""
    from b12x.integration.cuteafd.inkling2_attention import record_bytes
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot

    context = k.shape[0]
    pages = (context + g.page_rows - 1) // g.page_rows
    slots = torch.arange(context, device="cuda", dtype=torch.int64)
    fp8_cache = torch.zeros((pages * g.page_rows, record_bytes(g, "fp8")), dtype=torch.uint8, device="cuda")
    pack = compile_inkling2_kv_pack_aot(
        heads=g.kv_heads, dim=g.head_dim, input_dtype=torch.bfloat16,
        max_rows=max(context, 4096), kv_format="fp8",
    )
    pack.launch(k.contiguous(), v.contiguous(), slots, fp8_cache, scalars=(context,))
    torch.cuda.synchronize()

    # Exact host-equivalent dequantization matching _e4m3x8_scaled_bf16: bf16(f32(e4m3) * f32(scale)).
    plane_bytes = g.kv_heads * g.head_dim
    scales = fp8_cache[:, 2 * plane_bytes:].contiguous().view(torch.float32).reshape(-1, 2, g.kv_heads)
    bf16_cache = torch.zeros((pages * g.page_rows, g.record_elems), dtype=torch.bfloat16, device="cuda")
    for plane in range(2):
        raw = fp8_cache[:, plane * plane_bytes:(plane + 1) * plane_bytes].contiguous()
        e4m3 = raw.view(torch.float8_e4m3fn).float().reshape(-1, g.kv_heads, g.head_dim)
        deq = (e4m3 * scales[:, plane, :, None]).to(torch.bfloat16).reshape(-1, plane_bytes)
        bf16_cache[:, plane * plane_bytes:(plane + 1) * plane_bytes] = deq
    return fp8_cache, bf16_cache


def _launch_global(program, q, cache, positions, tables, indices, first, tile_rows, rel, sinks,
                   scratch=None):
    out = torch.empty_like(q)
    rows = q.shape[0]
    if scratch is None:
        scratch = torch.empty(max(program.scratch_bytes(rows)["scratch"], 1024), dtype=torch.uint8, device="cuda")
    scalars = (rows, tables.shape[-1], tile_rows.shape[0])
    program.launch(q, cache, positions, tables, indices, first, tile_rows, rel, sinks, out, scratch, scalars=scalars)
    return out


@pytest.mark.parametrize("geom_name", ["small_tp1", "small_tp2", "mini"])
@pytest.mark.parametrize("context", [400, 512, 513, 2131, 4096, 8192, 16384])
def test_fp8_global_decode_bit_identical_to_dequant_bf16(geom_name, context):
    """Async cp.async FP8 decode matches BF16 decode over dequantized FP8 KV bit-for-bit across
    per-row split counts n=1..32, and rows with n==1 leave split scratch untouched."""
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_MINI, INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot, row_splits

    geoms = {
        "small_tp1": INKLING2_SMALL,
        "small_tp2": replace(INKLING2_SMALL, heads=24, kv_heads=4),
        "mini": INKLING2_MINI,
    }
    g = geoms[geom_name]
    torch.manual_seed(1701 + context)
    rows = 5
    pages = (context + g.page_rows - 1) // g.page_rows
    k = torch.randn((context, g.kv_heads, g.head_dim), device="cuda").bfloat16()
    v = torch.randn_like(k)
    fp8_cache, bf16_cache = _pack_and_dequant_fp8(k, v, g)

    # Permute pages to exercise non-contiguous page table lookup.
    perm = torch.randperm(pages, device="cuda").int()
    fp8_perm = torch.zeros_like(fp8_cache)
    bf16_perm = torch.zeros_like(bf16_cache)
    for logical in range(pages):
        phys = int(perm[logical])
        fp8_perm[phys * g.page_rows:(phys + 1) * g.page_rows] = fp8_cache[logical * g.page_rows:(logical + 1) * g.page_rows]
        bf16_perm[phys * g.page_rows:(phys + 1) * g.page_rows] = bf16_cache[logical * g.page_rows:(logical + 1) * g.page_rows]

    tables = perm[None, :].contiguous()
    positions = torch.arange(context - rows, context, device="cuda", dtype=torch.int64)
    indices = torch.zeros(rows, device="cuda", dtype=torch.int32)
    first = torch.zeros(rows, device="cuda", dtype=torch.int32)
    tile_rows = torch.arange(rows, device="cuda", dtype=torch.int32)
    q = torch.randn((rows, g.heads, g.head_dim), device="cuda").bfloat16()
    rel = (torch.randn((rows, g.heads, g.rel_extent), device="cuda") * 2.0).bfloat16()
    sinks = (torch.randn((rows, g.heads), device="cuda") + 4.0).float()

    prog_fp8 = compile_inkling2_attention_aot(g, kind="global", route="decode", max_rows=64, kv_format="fp8")
    prog_bf16 = compile_inkling2_attention_aot(g, kind="global", route="decode", max_rows=64, kv_format="bf16")
    scratch_fp8 = torch.full((max(prog_fp8.scratch_bytes(rows)["scratch"], 1024),), 0xA5, dtype=torch.uint8, device="cuda")
    out_fp8 = _launch_global(prog_fp8, q, fp8_perm, positions, tables, indices, first, tile_rows, rel, sinks, scratch=scratch_fp8)
    out_bf16 = _launch_global(prog_bf16, q, bf16_perm, positions, tables, indices, first, tile_rows, rel, sinks)
    torch.cuda.synchronize()
    assert torch.equal(out_fp8, out_bf16), (
        f"{geom_name} ctx={context}: max diff {(out_fp8.float() - out_bf16.float()).abs().max().item()}"
    )
    if all(row_splits(int(p) + 1) == 1 for p in positions.tolist()):
        assert torch.all(scratch_fp8 == 0xA5), f"{geom_name} ctx={context}: n==1 rows unexpectedly wrote scratch"


@pytest.mark.parametrize("geom_name", ["small_tp1", "small_tp2"])
@pytest.mark.parametrize("rows", [1, 17, 137])
def test_fp8_global_prefill_bit_identical_to_dequant_bf16(geom_name, rows):
    """2-stage pipelined async FP8 prefill matches BF16 prefill over dequantized FP8 KV bit-for-bit."""
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot, prefill_tokens

    g = INKLING2_SMALL if geom_name == "small_tp1" else replace(INKLING2_SMALL, heads=24, kv_heads=4)
    torch.manual_seed(1801 + rows)
    context = 1403  # crosses rel_extent=1024 and non-page-aligned
    pages = (context + g.page_rows - 1) // g.page_rows
    k = torch.randn((context, g.kv_heads, g.head_dim), device="cuda").bfloat16()
    v = torch.randn_like(k)
    fp8_cache, bf16_cache = _pack_and_dequant_fp8(k, v, g)

    tables = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :].contiguous()
    positions = torch.arange(context - rows, context, device="cuda", dtype=torch.int64)
    indices = torch.zeros(rows, device="cuda", dtype=torch.int32)
    first = torch.zeros(rows, device="cuda", dtype=torch.int32)
    tile_rows = torch.arange(0, rows, prefill_tokens(g), device="cuda", dtype=torch.int32)
    q = torch.randn((rows, g.heads, g.head_dim), device="cuda").bfloat16()
    rel = (torch.randn((rows, g.heads, g.rel_extent), device="cuda") * 2.0).bfloat16()
    sinks = (torch.randn((rows, g.heads), device="cuda") + 4.0).float()

    prog_fp8 = compile_inkling2_attention_aot(g, kind="global", route="prefill", max_rows=4096, kv_format="fp8")
    prog_bf16 = compile_inkling2_attention_aot(g, kind="global", route="prefill", max_rows=4096, kv_format="bf16")
    out_fp8 = _launch_global(prog_fp8, q, fp8_cache, positions, tables, indices, first, tile_rows, rel, sinks)
    out_bf16 = _launch_global(prog_bf16, q, bf16_cache, positions, tables, indices, first, tile_rows, rel, sinks)
    torch.cuda.synchronize()
    assert torch.equal(out_fp8, out_bf16), (
        f"{geom_name} prefill rows={rows}: max diff {(out_fp8.float() - out_bf16.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("geom_name", ["small_tp1", "mini"])
def test_fp8_global_decode_batch_invariance(geom_name):
    """Same row run alone vs packed with BOTH shorter and longer rows produces 0 bit differences."""
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_MINI, INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    g = INKLING2_SMALL if geom_name == "small_tp1" else INKLING2_MINI
    torch.manual_seed(1901 if geom_name == "small_tp1" else 1902)
    max_ctx = 17000
    pages = (max_ctx + g.page_rows - 1) // g.page_rows
    k = torch.randn((max_ctx, g.kv_heads, g.head_dim), device="cuda").bfloat16()
    v = torch.randn_like(k)
    fp8_cache, _ = _pack_and_dequant_fp8(k, v, g)
    tables = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :].contiguous()
    prog = compile_inkling2_attention_aot(g, kind="global", route="decode", max_rows=64, kv_format="fp8")

    # Target positions covering n=1, n=1->2 edge, n=2->3 edge, n=8->9 edge, n=16, n=32.
    target_positions = [399, 511, 512, 1023, 1024, 4095, 4096, 8191, 16383]
    # Pack with a very short row (n=1) AND a very long row (n=32) plus intermediate rows.
    batch_positions = [64, *target_positions, 16999, 128, 16500]
    n_batch = len(batch_positions)
    q_batch = torch.randn((n_batch, g.heads, g.head_dim), device="cuda").bfloat16()
    rel_batch = (torch.randn((n_batch, g.heads, g.rel_extent), device="cuda") * 2.0).bfloat16()
    sinks_batch = (torch.randn((n_batch, g.heads), device="cuda") + 4.0).float()
    pos_batch = torch.tensor(batch_positions, device="cuda", dtype=torch.int64)
    idx_batch = torch.zeros(n_batch, device="cuda", dtype=torch.int32)
    first_batch = torch.arange(n_batch, device="cuda", dtype=torch.int32)
    tiles_batch = torch.arange(n_batch, device="cuda", dtype=torch.int32)

    out_batch = _launch_global(prog, q_batch, fp8_cache, pos_batch, tables, idx_batch, first_batch, tiles_batch,
                               rel_batch, sinks_batch)
    torch.cuda.synchronize()

    z1 = torch.zeros(1, device="cuda", dtype=torch.int32)
    for row_idx in range(1, 1 + len(target_positions)):
        out_alone = _launch_global(
            prog,
            q_batch[row_idx:row_idx + 1].contiguous(),
            fp8_cache,
            pos_batch[row_idx:row_idx + 1].contiguous(),
            tables, z1, z1, z1,
            rel_batch[row_idx:row_idx + 1].contiguous(),
            sinks_batch[row_idx:row_idx + 1].contiguous(),
        )
        torch.cuda.synchronize()
        assert torch.equal(out_alone[0], out_batch[row_idx]), (
            f"{geom_name} pos={batch_positions[row_idx]}: alone vs mixed-batch drifted "
            f"(max diff {(out_alone[0].float() - out_batch[row_idx].float()).abs().max().item()})"
        )


@pytest.mark.parametrize("geom_name", ["small_tp1", "mini"])
def test_fp8_global_decode_verify_matches_decode_across_bucket_edges(geom_name):
    """Multi-row speculative verify straddling split-count bucket edges matches 1-by-1 decode bit-for-bit."""
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_MINI, INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    g = INKLING2_SMALL if geom_name == "small_tp1" else INKLING2_MINI
    torch.manual_seed(2001 if geom_name == "small_tp1" else 2002)
    max_ctx = 16500
    pages = (max_ctx + g.page_rows - 1) // g.page_rows
    k = torch.randn((max_ctx, g.kv_heads, g.head_dim), device="cuda").bfloat16()
    v = torch.randn_like(k)
    fp8_cache, _ = _pack_and_dequant_fp8(k, v, g)
    tables = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :].contiguous()
    prog = compile_inkling2_attention_aot(g, kind="global", route="decode", max_rows=64, kv_format="fp8")

    # 5-row verify windows straddling bucket boundaries (511/512, 1023/1024, 4095/4096, 8191/8192, 15871/15872, 16383/16384).
    z1 = torch.zeros(1, device="cuda", dtype=torch.int32)
    for start_pos in (510, 1022, 4094, 8190, 15870, 16382):
        verify_rows = 5
        q = torch.randn((verify_rows, g.heads, g.head_dim), device="cuda").bfloat16()
        rel = (torch.randn((verify_rows, g.heads, g.rel_extent), device="cuda") * 2.0).bfloat16()
        sinks = (torch.randn((verify_rows, g.heads), device="cuda") + 4.0).float()
        positions = torch.arange(start_pos, start_pos + verify_rows, device="cuda", dtype=torch.int64)
        idx = torch.zeros(verify_rows, device="cuda", dtype=torch.int32)
        first = torch.zeros(verify_rows, device="cuda", dtype=torch.int32)
        tiles = torch.arange(verify_rows, device="cuda", dtype=torch.int32)

        out_verify = _launch_global(prog, q, fp8_cache, positions, tables, idx, first, tiles, rel, sinks)
        torch.cuda.synchronize()
        for r in range(verify_rows):
            out_decode = _launch_global(
                prog,
                q[r:r + 1].contiguous(),
                fp8_cache,
                positions[r:r + 1].contiguous(),
                tables, z1, z1, z1,
                rel[r:r + 1].contiguous(),
                sinks[r:r + 1].contiguous(),
            )
            torch.cuda.synchronize()
            assert torch.equal(out_verify[r], out_decode[0]), (
                f"{geom_name} verify start={start_pos} pos={start_pos + r}: verify vs decode drifted "
                f"(max diff {(out_verify[r].float() - out_decode[0].float()).abs().max().item()})"
            )
