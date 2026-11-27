"""Bitwise oracle, split-merge skip, and row-invariance checks for async FP8 Inkling global attention."""
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
                   route, splits):
    out = torch.empty_like(q)
    rows = q.shape[0]
    scratch = torch.empty(max(program.scratch_bytes(rows)["scratch"], 1024), dtype=torch.uint8, device="cuda")
    scalars = (rows, tables.shape[-1], tile_rows.shape[0]) + ((splits,) if route == "decode" else ())
    program.launch(q, cache, positions, tables, indices, first, tile_rows, rel, sinks, out, scratch, scalars=scalars)
    return out


@pytest.mark.parametrize("geom_name", ["small_tp1", "small_tp2", "mini"])
@pytest.mark.parametrize("splits", [1, 4, 8, 16, 32])
def test_fp8_global_decode_bit_identical_to_dequant_bf16(geom_name, splits):
    """Async cp.async FP8 decode matches BF16 decode over the exact dequantized FP8 KV bit-for-bit."""
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_MINI, INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    geoms = {
        "small_tp1": INKLING2_SMALL,
        "small_tp2": replace(INKLING2_SMALL, heads=24, kv_heads=4),
        "mini": INKLING2_MINI,
    }
    g = geoms[geom_name]
    torch.manual_seed(1701 + splits)
    context = 2131  # non-page-aligned context crossing rel_extent=1024
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
    out_fp8 = _launch_global(prog_fp8, q, fp8_perm, positions, tables, indices, first, tile_rows, rel, sinks, "decode", splits)
    out_bf16 = _launch_global(prog_bf16, q, bf16_perm, positions, tables, indices, first, tile_rows, rel, sinks, "decode", splits)
    torch.cuda.synchronize()
    assert torch.equal(out_fp8, out_bf16), (
        f"{geom_name} splits={splits}: max diff {(out_fp8.float() - out_bf16.float()).abs().max().item()}"
    )


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
    out_fp8 = _launch_global(prog_fp8, q, fp8_cache, positions, tables, indices, first, tile_rows, rel, sinks, "prefill", 1)
    out_bf16 = _launch_global(prog_bf16, q, bf16_cache, positions, tables, indices, first, tile_rows, rel, sinks, "prefill", 1)
    torch.cuda.synchronize()
    assert torch.equal(out_fp8, out_bf16), (
        f"{geom_name} prefill rows={rows}: max diff {(out_fp8.float() - out_bf16.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("geom_name", ["small_tp1", "mini"])
@pytest.mark.parametrize("splits", [1, 8, 16])
def test_fp8_global_decode_row_invariance_and_direct_skip(geom_name, splits):
    """At a fixed split count, a row's output bits are identical across batch sizes 1, 5, 16, 64,
    and splits==1 does not touch scratch (InklingSplitMerge is skipped)."""
    require_b12x()
    from b12x.integration.cuteafd import INKLING2_MINI, INKLING2_SMALL
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    g = INKLING2_SMALL if geom_name == "small_tp1" else INKLING2_MINI
    torch.manual_seed(1901 + splits)
    context = 2048
    pages = context // g.page_rows
    k = torch.randn((context, g.kv_heads, g.head_dim), device="cuda").bfloat16()
    v = torch.randn_like(k)
    fp8_cache, _ = _pack_and_dequant_fp8(k, v, g)
    tables = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :].contiguous()
    q_all = torch.randn((64, g.heads, g.head_dim), device="cuda").bfloat16()
    rel_all = (torch.randn((64, g.heads, g.rel_extent), device="cuda") * 2.0).bfloat16()
    sinks_all = (torch.randn((64, g.heads), device="cuda") + 4.0).float()
    pos_all = torch.full((64,), context - 1, device="cuda", dtype=torch.int64)
    prog = compile_inkling2_attention_aot(g, kind="global", route="decode", max_rows=64, kv_format="fp8")

    ref_row = None
    for rows in (1, 5, 16, 64):
        indices = torch.zeros(rows, device="cuda", dtype=torch.int32)
        first = torch.zeros(rows, device="cuda", dtype=torch.int32)
        tile_rows = torch.arange(rows, device="cuda", dtype=torch.int32)
        scratch = torch.full((max(prog.scratch_bytes(rows)["scratch"], 1024),), 0xA5, dtype=torch.uint8, device="cuda")
        out = torch.empty((rows, g.heads, g.head_dim), dtype=torch.bfloat16, device="cuda")
        prog.launch(
            q_all[:rows].contiguous(), fp8_cache, pos_all[:rows].contiguous(), tables,
            indices, first, tile_rows, rel_all[:rows].contiguous(), sinks_all[:rows].contiguous(),
            out, scratch, scalars=(rows, pages, rows, splits),
        )
        torch.cuda.synchronize()
        if splits == 1:
            # Direct path never writes split partials or LSE into scratch.
            assert torch.all(scratch == 0xA5), f"{geom_name}: splits=1 unexpectedly wrote to split scratch"
        if ref_row is None:
            ref_row = out[0].clone()
        else:
            assert torch.equal(out[0], ref_row), f"{geom_name} splits={splits} rows={rows}: row 0 drifted"
