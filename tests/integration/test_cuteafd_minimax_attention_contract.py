"""CPU execution of MSA metadata adapters; QK/attention kernels need GPU gates."""
from __future__ import annotations

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2] / "b12x/integration/cuteafd"


def module(name):
    tree = ast.parse((ROOT / f"{name}.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    arch = SimpleNamespace(thread_idx=lambda: (0, 0, 0), block_idx=lambda: (0, 0, 0))
    env = dict(Int32=np.int32, Int64=np.int64, Float32=SimpleNamespace(inf=np.inf),
        const_expr=bool, cutlass=SimpleNamespace(range_constexpr=range, range=range, min=min),
        cute=SimpleNamespace(arch=arch), attention_ops=SimpleNamespace(fmax=np.maximum),
        Qwen4AttnPost=object, _SPLIT=4)  # dsa_indexer.kernel schedule constant
    class F32:
        inf = np.inf

        def __new__(cls, value):
            return np.float32(value)
    env["Float32"] = F32
    exec(compile(tree, str(ROOT / f"{name}.py"), "exec", flags=__future__.annotations.compiler_flag), env)
    return env, arch


def execute(arch, fn, grid, threads, *args):
    for x in range(grid[0]):
        for y in range(grid[1]):
            arch.block_idx = lambda: (x, y, 0)
            for tid in range(threads):
                arch.thread_idx = lambda: (tid, 0, 0)
                fn(*args)


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("rows,pages", [(1, 1), (3, 31), (128, 16384)])
def test_index_scratch_regions_are_aligned_and_disjoint(mode, rows, pages, heads):
    env, _ = module("minimax_indexer")
    blocks, offsets, size = env["_layout"](rows, pages, mode, heads)
    assert blocks == (pages + 1) // 2
    assert all(x % 256 == 0 for x in offsets.values())
    assert list(offsets.values()) == sorted(offsets.values())
    assert size >= offsets["controls"] + 256
    assert env["minimax_index_scratch_bytes"](max_rows=rows, max_pages=pages, mode=mode, heads=heads) == size


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("lengths", [[1], [65], [16000], [700] * 16, [5, 70000, 300, 64, 129], [4096] * 64])
def test_decode_schedule_spreads_every_live_page_once_over_balanced_intervals(lengths, heads):
    env, arch = module("minimax_indexer")
    obj = object.__new__(env["_Index"])
    obj.mode, obj.rows, obj.reference, obj.heads = "decode", 64, False, heads
    rows, width = len(lengths), 1024 * 64
    lengths = np.array(lengths, dtype=np.int32)
    scale, weights = np.ones(rows * heads, np.float32), np.zeros(rows * heads, np.float32)
    schedule, controls = np.full((obj.rows + 1) * 2, -7, np.int32), np.zeros(4, np.int32)
    grid = (rows * heads + 127) // 128 + (obj.rows + 128) // 128
    execute(arch, obj.init, (grid, 1), 128, scale, weights, schedule, controls, lengths, rows, width)
    schedule = schedule.reshape(-1, 2)
    live = np.minimum(lengths, width)
    pages = (live + 63) // 64
    splits = (pages + 3) // 4
    visited, work = {}, []
    # The scorer's walk (DSAScheduledMultiRowLogitsKernel): interval i covers
    # schedule[i] up to schedule[i + 1], 4 pages per split.
    for (q, s), (end_q, end_s) in zip(schedule[:-1], schedule[1:]):
        done = 0
        while q < rows and (q < end_q or (q == end_q and end_s > 0)):
            first, last = s, splits[q] if q != end_q else min(end_s, splits[q])
            for page in range(first * 4, min(last * 4, pages[q])):
                visited[(q, page)] = visited.get((q, page), 0) + 1
            done += max(last - first, 0)
            q, s = q + 1, 0
        work.append(done)
    assert visited == {(r, p): 1 for r in range(rows) for p in range(pages[r])}
    total = int(splits.sum())
    assert max(work) == -(-total // obj.rows) and sum(work) == total
    assert tuple(schedule[-1]) == (rows, 0)
    assert controls[0] == width and controls[1] == 1  # shared scalar packed-page loads


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_attention_metadata_preserves_request_boundaries(mode):
    env, arch = module("minimax_attention")
    obj = object.__new__(env["_Attention"])
    obj.tokens = 1 if mode == "decode" else 8
    cu = np.array([0, 1, 10, 23], dtype=np.int32)
    rows, batch = 23, 3
    work = rows if mode == "decode" else (rows + 7) // 8 + batch
    requests, tiles, zeros, valid = [np.full(work, -99, dtype=np.int32) for _ in range(4)]
    execute(arch, obj.metadata, (1, 1), 128, cu, requests, tiles, zeros, valid, work, batch)
    expected = [(r, t) for r in range(batch) for t in range((cu[r + 1] - cu[r] + obj.tokens - 1) // obj.tokens)]
    assert list(zip(requests[valid == 1], tiles[valid == 1])) == expected
    assert (zeros == 0).all()
    assert (valid[len(expected):] == 0).all()


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_pool_force_and_publication_reuse_fixed_head_stride(mode, heads):
    env, arch = module("minimax_indexer")
    obj = object.__new__(env["_Index"])
    obj.mode, obj.rows, obj.pages, obj.blocks, obj.heads = mode, 7, 37, 19, heads
    rows = 3
    rng = np.random.default_rng(17)
    pages = rng.normal(size=(heads, rows, obj.pages)).astype(np.float32)
    scores = rng.normal(size=(heads, rows, obj.blocks)).astype(np.float32)
    original = scores.copy()
    k_start = np.array([0, 128, 256], dtype=np.int32) if mode == "prefill" else np.zeros(rows, np.int32)
    k_end = np.array([1, 17 * 128 + 5, 18 * 128 + 3], dtype=np.int32)
    starts, lengths = np.full(rows * heads, -99, np.int32), np.full(rows * heads, -99, np.int32)
    execute(arch, obj.finish_scores, (rows * heads, 1), 128, pages.ravel(), scores.ravel(),
            starts, lengths, k_start, k_end, rows, obj.pages)
    selected = np.full((heads, rows, 16), -1, dtype=np.int32)
    for h in range(heads):
        for r in range(rows):
            begin, end = k_start[r] // 128, (k_end[r] + 127) // 128
            assert starts[h * rows + r] == begin
            assert lengths[h * rows + r] == end - begin
            assert np.isposinf(scores[h, r, end - 1])
            for b in range(begin, end - 1):
                expected = max(pages[h, r, 2*b:2*b+2]) if mode == "decode" else original[h, r, b]
                assert scores[h, r, b] == expected
            assert np.isneginf(scores[h, r, :begin]).all()
            assert np.isneginf(scores[h, r, end:]).all()
            chosen = sorted(range(begin, end), key=lambda b: (-scores[h, r, b], b))[:16]
            chosen.sort()
            selected[h, r, :len(chosen)] = chosen
    output = np.full((heads, obj.rows, 16), -88, dtype=np.int32)
    shuffled = selected[:, :, ::-1].copy()
    execute(arch, obj.publish, (2, 1), 128, shuffled.ravel(), starts, output.ravel(), rows)
    for h in range(heads):
        for r in range(rows):
            expected = [int(v - k_start[r] // 128) if v >= 0 else -1 for v in selected[h, r]]
            assert output[h, r].tolist() == expected
        assert (output[h, rows:] == -88).all()


@pytest.mark.parametrize("heads", [2, 4])
def test_union_masks_and_partial_tiles_keep_each_rows_selection(heads):
    env, arch = module("minimax_attention")
    obj = object.__new__(env["_Attention"])
    obj.rows, obj.kv_heads = 20, heads
    cu = np.array([0, 3, 12], dtype=np.int32)
    lengths = np.array([300, 2200], dtype=np.int32)
    requests, tiles, valid = np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0]), np.array([1, 1, 1, 0])
    selected = np.full((heads, obj.rows, 16), -1, dtype=np.int32)
    for h in range(heads):
        for row in range(12):
            selected[h, row, :3] = [0, 1 + (row + h) % 2, 2 if row < 3 else 17]
    # A later row introduces a lower block after the first row's local block.
    selected[:, 3, :3] = [0, 7, 17]
    selected[:, 4, :3] = [0, 3, 17]
    blocks, masks = np.full((4, heads, 128), -99, np.int32), np.full((4, heads, 128), -99, np.int32)
    counts = np.full((4, heads), -99, np.int32)
    execute(arch, obj.union, (4, heads), 1, selected.ravel(), lengths, cu, requests, tiles, valid,
            blocks.ravel(), masks.ravel(), counts.ravel(), 4)
    for work in range(4):
        for h in range(heads):
            if not valid[work]:
                assert counts[work, h] == 0
                continue
            req, first = requests[work], tiles[work] * 8
            qlen = cu[req + 1] - cu[req]
            union = blocks[work, h, :counts[work, h]].tolist()
            assert union == sorted(set(union))
            mapping = dict(zip(union, masks[work, h, :counts[work, h]]))
            for token in range(min(8, qlen - first)):
                row = cu[req] + first + token
                visible = lengths[req] - qlen + first + token + 1
                expected = {int(b) for b in selected[h, row] if b >= 0 and b * 128 < visible}
                assert {int(b) for b, mask in mapping.items() if mask & (1 << token)} == expected


def test_invalid_capacities_fail_before_compilation():
    for name, args in [("minimax_indexer", (0, 1, "decode")),
                       ("minimax_indexer", (1, 1, "extend")),
                       ("minimax_indexer", (1 << 20, 1 << 20, "prefill")),
                       ("minimax_attention", (0, 1, "decode")),
                       ("minimax_attention", (1, 1, "prefill"))]:
        env, _ = module(name)
        with pytest.raises(ValueError):
            env["_layout"](*args)


@pytest.mark.parametrize("heads", [0, 1, 3, 8, True, 2.0])
def test_unsupported_head_geometry_fails_before_compilation(heads):
    for name, args in [("minimax_indexer", (16, 64, "decode", heads)),
                       ("minimax_attention", (16, 4, "extend", 32, heads))]:
        env, _ = module(name)
        with pytest.raises(ValueError, match="heads"):
            env["_layout"](*args)
    for name, function in [("minimax_producer", "compile_minimax_index_producer_aot"),
                           ("minimax", "compile_minimax_kv_pack_aot")]:
        env, _ = module(name)
        with pytest.raises(ValueError, match="heads"):
            env[function](heads=heads)


def test_half_head_attention_scratch_has_only_local_union_storage():
    env, _ = module("minimax_attention")
    full_work, full_offsets, full_size = env["_layout"](4096, 64, "extend")
    work, offsets, size = env["_layout"](4096, 64, "extend", 32, 2)
    assert work == full_work
    assert offsets["union_blocks"] == full_offsets["union_blocks"]
    assert full_size - size == work * 2 * (128 * 4 * 2 + 4)
    assert env["_layout"](64, 64, "decode", 32, 2) == env["_layout"](64, 64, "decode")
    with pytest.raises(ValueError, match="heads"):
        env["_layout"](16, 4, "extend", 64, 2)


def test_half_head_compile_keys_and_pointer_contracts_preserve_four_head_default():
    for name, function, kwargs, head_kwargs, head_geometry in [
        ("minimax_attention", "compile_minimax_attention_aot", dict(max_rows=16, max_batch=4),
         dict(query_heads=32, kv_heads=2), "kv_heads"),
        ("minimax_indexer", "compile_minimax_index_aot", dict(max_rows=16, max_pages=64),
         dict(heads=2), "heads"),
        ("minimax_producer", "compile_minimax_index_producer_aot", {}, dict(heads=2), "heads"),
        ("minimax", "compile_minimax_kv_pack_aot", {}, dict(heads=2), "heads")]:
        env, _ = module(name)
        env.update(torch=SimpleNamespace(bfloat16="bf16", float32="f32", int32="i32", int64="i64",
                                        uint8="u8", float8_e4m3fn="fp8"),
                   Operand=lambda *args, **kwargs: (args, kwargs),
                   Scalar=lambda *args, **kwargs: (args, kwargs),
                   compile_program=lambda launch, **contract: contract)
        for cls in ("_Attention", "_Decode", "_Prefill", "_Index", "_KvPack"):
            env[cls] = lambda *args: SimpleNamespace(nbytes=1024)
        full = env[function](**kwargs)
        explicit = env[function](**kwargs, **{k: 64 if k == "query_heads" else 4 for k in head_kwargs})
        half = env[function](**kwargs, **head_kwargs)
        for contract in (full, explicit, half):
            contract["scratch"] = {k: query(7) for k, query in contract.get("scratch", {}).items()}
        assert full == explicit
        assert full["geometry"][head_geometry] == 4 and half["geometry"][head_geometry] == 2
        assert full["key"] != half["key"]
        assert full["scalars"] == half["scalars"]
        assert [args[0] for args, _ in full["operands"]] == [args[0] for args, _ in half["operands"]]
        if name == "minimax_producer":
            assert half["scratch"]["scratch"] == 7 * 5 * 128 * 2


def test_scalar_index_page_load_widens_recycled_physical_ids():
    path = ROOT.parents[1] / "attention/dsa_indexer/kernel.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_index_k_page_scalar"]
    tree.body[0].decorator_list = []
    env = dict(Int32=np.int32, Int64=np.int64, _PAGE_SIZE=64, _INDEX_HEAD_DIM=128,
               _PAGED_THREADS_PER_CTA=128)
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), env)
    addresses, shared = [], {}

    class Cache:
        def __getitem__(self, coords):
            page, row, col = coords
            address = page * np.int32(8448) + row * np.int32(128) + col
            addresses.append(int(address))
            return address & 255

    page = np.int32(2**31 // 8448 + 13)
    for thread in range(128):
        env["_load_index_k_page_scalar"](Cache(), page, shared, np.int32(thread))
    expected = int(page) * 8448
    assert sorted(addresses) == list(range(expected, expected + 8192))
    assert min(addresses) > 2**31
    assert len(shared) == 8192
