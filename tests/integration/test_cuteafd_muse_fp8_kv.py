"""Muse E4M3 per-token/head KV: exact record, QDQ attention and graph replay."""
from __future__ import annotations

import json
import os

import pytest
import torch

from ..conftest import require_b12x


ROWS = (1, 7, 16, 64, 80, 2048, 2752, 4096)


def _quantize(records, heads):
    """Independent QDQ oracle after Muse's checkpoint BF16 boundaries."""
    kv = records.reshape(-1, 2, heads, 64).float()
    amax = kv.abs().amax(-1)
    scales = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    payload = (kv * scales.reciprocal()[..., None]).to(torch.float8_e4m3fn)
    packed = torch.cat((payload.view(torch.uint8).flatten(1),
                        scales.contiguous().view(torch.uint8).flatten(1)), dim=1)
    qdq = (payload.float() * scales[..., None]).bfloat16().flatten(1)
    assert packed.shape[1] % 16 == 0
    return packed, qdq


@pytest.mark.parametrize("heads", (64, 128))
@pytest.mark.parametrize("kind", ("swa", "full"))
@pytest.mark.parametrize("rows", ROWS)
def test_muse_fp8_kv_record_exact(heads, kind, rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_qk_norm_rope_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    gen = torch.Generator().manual_seed(948)
    kv_heads = heads // 8
    q = torch.randn(rows, heads, 64, generator=gen).bfloat16().cuda()
    k = torch.randn(rows, kv_heads, 64, generator=gen).bfloat16().cuda()
    v = torch.randn(rows, kv_heads, 64, generator=gen).bfloat16().cuda()
    # All-zero values, extreme finite values, sign asymmetry and per-head ranges.
    v[0] = 0
    if rows > 1:
        v[1] *= 512
        k[1] *= 0.001
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    freq = positions.float()[:, None] / (500000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    cos, sin = freq.cos().bfloat16(), freq.sin().bfloat16()
    slots = (torch.randperm(rows, generator=gen).long() * 2 + 1).cuda()
    old_q, new_q = torch.empty_like(q), torch.empty_like(q)
    bf16 = torch.empty(rows * 2 + 2, kv_heads * 128, dtype=torch.bfloat16, device="cuda")
    fp8 = torch.full((rows * 2 + 2, kv_heads * 136), 0xA5, dtype=torch.uint8, device="cuda")
    old = compile_qk_norm_rope_aot(heads=heads, kind=kind)
    new = compile_qk_norm_rope_aot(heads=heads, kind=kind, kv_cache="fp8")
    args = (q, k, v, positions, cos, sin, slots)
    old.launch(*args, old_q, bf16, scalars=[rows])
    expected = fp8.clone()
    expected[slots] = _quantize(bf16[slots], kv_heads)[0]
    new.launch(*args, new_q, fp8, scalars=[rows])
    assert torch.equal(new_q, old_q)
    assert torch.equal(fp8, expected), int((fp8 != expected).sum())
    saved = fp8.clone()
    new.launch(q, k, v, positions, cos, sin, torch.full_like(slots, -1), new_q, fp8, scalars=[rows])
    assert torch.equal(fp8, saved)
    singleton = torch.empty_like(fp8[:1])
    single_q = torch.empty_like(q[:1])
    zero = torch.zeros(1, dtype=torch.int64, device="cuda")
    with kernel_resolution_guard("Muse FP8 KV is precompiled independent of live rows"):
        for row in sorted({0, rows // 2, rows - 1}):
            new.launch(q[row:row + 1], k[row:row + 1], v[row:row + 1], positions[row:row + 1],
                       cos, sin, zero, single_q, singleton, scalars=[1])
            assert torch.equal(singleton, expected[slots[row:row + 1]])
            assert torch.equal(single_q, old_q[row:row + 1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            new.launch(*args, new_q, fp8, scalars=[rows])
        fp8.fill_(0xA5)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(fp8, expected)
        new.launch(*args, new_q, fp8, scalars=[0])
        assert torch.equal(fp8, expected)


@pytest.mark.parametrize("heads", (64, 128))
@pytest.mark.parametrize("kind", ("full", "swa"))
@pytest.mark.parametrize("rows", (1, 16, 65))
def test_muse_fp8_attention_same_scale_qdq(heads, kind, rows):
    require_b12x()
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot, compile_swa_commit_aot
    from b12x._lib.runtime_control import kernel_resolution_guard

    gen = torch.Generator().manual_seed(949)
    history, window, ring_rows = 2050, 2048, 2112
    end, kv_heads = history + rows, heads // 8
    query = torch.randn(rows, heads, 64, generator=gen).bfloat16().cuda()
    records = torch.randn(end, kv_heads * 128, generator=gen).bfloat16().cuda()
    packed, qdq = _quantize(records, kv_heads)
    positions = torch.arange(history, end, dtype=torch.int64, device="cuda")
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    outputs, singles, graph_args = {}, {}, {}
    for fmt, source in (("bf16", qdq), ("fp8", packed)):
        prefill = compile_attention_aot(heads=heads, kind=kind, route="prefill", ring_rows=ring_rows,
                                        window=window, kv_cache=fmt)
        decode = compile_attention_aot(heads=heads, kind=kind, route="decode", ring_rows=ring_rows,
                                       window=window, kv_cache=fmt)
        out, single = torch.empty_like(query), torch.empty_like(query)
        if kind == "full":
            pages = (end + 63) // 64
            cache = torch.zeros(pages * 2 * 64, source.shape[1], dtype=source.dtype, device="cuda")
            table = (torch.arange(pages, dtype=torch.int32, device="cuda") * 2 + 1).flip(0).contiguous()
            address = table.long()[torch.arange(end, device="cuda") // 64] * 64 + torch.arange(end, device="cuda") % 64
            cache[address] = source
            args = (query, cache, positions, table, out, scratch)
            scalars = [rows, 0]
            prefill.launch(*args, scalars=scalars)
            for row in range(rows):
                decode.launch(query[row:row + 1], cache, positions[row:row + 1], table,
                              single[row:row + 1], scratch, scalars=[1, 0, 1])
        else:
            ring = torch.zeros(ring_rows, source.shape[1], dtype=source.dtype, device="cuda")
            prior = torch.arange(history - window, history, device="cuda")
            ring[prior % ring_rows] = source[prior]
            slots = positions % ring_rows
            first = torch.zeros(rows, dtype=torch.int32, device="cuda")
            step = source[history:].contiguous()
            args = (query, step, ring, positions, slots, first, out, scratch)
            scalars = [rows]
            prefill.launch(*args, scalars=scalars)
            commit = compile_swa_commit_aot(heads=heads, ring_rows=ring_rows, kv_cache=fmt)
            for row in range(rows):
                decode.launch(query[row:row + 1], step[row:row + 1], ring, positions[row:row + 1],
                              slots[row:row + 1], first[:1], single[row:row + 1], scratch, scalars=[1])
                commit.launch(step[row:row + 1], ring, slots[row:row + 1], first[:1], scalars=[1])
                assert torch.equal(ring[slots[row]], step[row])
        assert torch.equal(out, single), (fmt, (out.float() - single.float()).abs().max())
        with kernel_resolution_guard("Muse attention uses the same frozen program on graph replay"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                prefill.launch(*args, scalars=scalars)
            saved = out.clone()
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, saved)
        outputs[fmt] = out
        singles[fmt] = single
        graph_args[fmt] = (prefill, args, scalars)
    assert torch.equal(outputs["bf16"], outputs["fp8"]), (outputs["bf16"].float() - outputs["fp8"].float()).abs().max()
    assert torch.equal(singles["bf16"], singles["fp8"])
    if os.environ.get("CUTEAFD_MUSE_KV_BENCH") == "1":
        timings = {}
        for fmt, (program, args, scalars) in graph_args.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    program.launch(*args, scalars=scalars)
            for _ in range(3):
                graph.replay()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            stop.record()
            stop.synchronize()
            timings[fmt] = start.elapsed_time(stop) * 1000 / 100
        print(json.dumps(dict(kind=kind, heads=heads, rows=rows, history=history,
                              microseconds=timings, speedup=timings["bf16"] / timings["fp8"])))


@pytest.mark.parametrize("heads", (64, 128))
def test_muse_fp8_kv_high_slot(heads):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_qk_norm_rope_aot
    from b12x.integration.cuteafd._muse_attention import compile_attention_aot

    kv_heads, page_rows = heads // 8, 64
    record = kv_heads * 136
    page = (1 << 31) // (record * page_rows) + 1
    # Only one tail page is touched; allocation stays under 2.2 GiB.
    pool = torch.empty((page + 1) * page_rows, record, dtype=torch.uint8, device="cuda")
    q = torch.randn(1, heads, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, kv_heads, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    position = torch.zeros(1, dtype=torch.int64, device="cuda")
    slots = torch.tensor([page * page_rows], dtype=torch.int64, device="cuda")
    cos, sin = torch.ones(1, 32, dtype=torch.bfloat16, device="cuda"), torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
    qo = torch.empty_like(q)
    producer = compile_qk_norm_rope_aot(heads=heads, kind="full", kv_cache="fp8")
    producer.launch(q, k, v, position, cos, sin, slots, qo, pool, scalars=[1])
    local = torch.empty(page_rows, record, dtype=torch.uint8, device="cuda")
    producer.launch(q, k, v, position, cos, sin, position, qo, local, scalars=[1])
    assert torch.equal(pool[slots], local[:1])
    attention = compile_attention_aot(heads=heads, kind="full", route="decode", kv_cache="fp8")
    out, expected = torch.empty_like(q), torch.empty_like(q)
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    attention.launch(qo, pool, position, torch.tensor([page], dtype=torch.int32, device="cuda"),
                     out, scratch, scalars=[1, 0, 1])
    attention.launch(qo, local, position, torch.zeros(1, dtype=torch.int32, device="cuda"),
                     expected, scratch, scalars=[1, 0, 1])
    assert torch.equal(out, expected)
