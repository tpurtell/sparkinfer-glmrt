"""MiniMax decode index scoring on the GPU: balanced and per-row page schedules
select exactly the same blocks using the current shared scalar packed-page loads.

Both programs score every live page whole in one CTA, so the per-page head maxima, the
block scores and the top-16 selections must be the same bits for single rows, long rows
and mixed batches, with page tables that scatter rows over a pool wider than the table.
(No postponed annotations: tensors are built at run time only.)
"""
import pytest

from ..conftest import require_b12x

torch = pytest.importorskip("torch")

MAX_ROWS, MAX_PAGES, POOL = 64, 1024, 1100
CASES = {
    "one_token": [1],
    "one_page_and_a_bit": [65],
    "short": [700],
    "long": [15403],
    "full_table": [MAX_PAGES * 64 - 3],
    "mixed": [5, 300, 64, 129, 4096, 20000, 1],
    "c16": [700 + 37 * i for i in range(16)],
    "c64": [1 + (i * 977) % 30000 for i in range(64)],
}


@pytest.fixture(scope="module", params=[2, 4])
def programs(request):
    require_b12x()
    from b12x.integration.cuteafd.minimax_indexer import compile_minimax_index_aot

    new = compile_minimax_index_aot(max_rows=MAX_ROWS, max_pages=MAX_PAGES, mode="decode", heads=request.param)
    reference = compile_minimax_index_aot(max_rows=MAX_ROWS, max_pages=MAX_PAGES, mode="decode", reference=True, heads=request.param)
    return new, reference


def _cache(generator):
    keys = (torch.rand((POOL, 64, 128), generator=generator, device="cuda") * 4 - 2).to(torch.float8_e4m3fn)
    scales = torch.rand((POOL, 64), generator=generator, device="cuda") + 0.5
    cache = torch.empty((POOL, 8448), dtype=torch.uint8, device="cuda")
    cache[:, :8192] = keys.view(torch.uint8).reshape(POOL, 8192)
    cache[:, 8192:] = scales.contiguous().view(torch.uint8).reshape(POOL, 256)
    return cache


@pytest.mark.parametrize("case", list(CASES))
def test_balanced_scoring_selects_exactly_what_the_row_schedule_did(programs, case):
    from b12x.integration.cuteafd.minimax_indexer import _layout

    heads = programs[0].geometry["heads"]
    lengths = CASES[case]
    rows = len(lengths)
    generator = torch.Generator(device="cuda").manual_seed(rows * 7919 + lengths[0])
    cache = _cache(generator)
    q = (torch.rand((rows, heads, 128), generator=generator, device="cuda") * 4 - 2).to(torch.float8_e4m3fn)
    q_scale = torch.rand((rows, heads), generator=generator, device="cuda") + 0.5
    widths = [(n + 63) // 64 for n in lengths]
    stride = max(widths) + 3  # a table pitch wider than its live width
    table = torch.full((rows, stride), -1, dtype=torch.int32)
    picker = torch.Generator().manual_seed(len(lengths))
    for row, width in enumerate(widths):
        table[row, :width] = torch.randperm(POOL, generator=picker)[:width].to(torch.int32)
    table = table.cuda().contiguous()
    cache_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _, offsets, nbytes = _layout(MAX_ROWS, MAX_PAGES, "decode", heads)
    results = []
    for program in programs:
        q2k = torch.full((heads, MAX_ROWS, 16), -7, dtype=torch.int32, device="cuda")
        scratch = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        program.launch(q.view(torch.uint8), q_scale, cache, table, cache_lengths, q2k, scratch,
                       scalars=(rows, max(widths), stride, POOL))
        torch.cuda.synchronize()
        pages = scratch[offsets["pages"]:offsets["pages"] + heads * rows * MAX_PAGES * 4]
        scores = scratch[offsets["scores"]:offsets["scores"] + heads * rows * ((MAX_PAGES + 1) // 2) * 4]
        results.append((q2k[:, :rows].clone(), pages.clone(), scores.clone()))
    (new_q2k, new_pages, new_scores), (ref_q2k, ref_pages, ref_scores) = results
    assert torch.equal(new_pages, ref_pages), "per-page head maxima differ"
    assert torch.equal(new_scores, ref_scores), "block scores differ"
    assert torch.equal(new_q2k, ref_q2k), "selections differ"
    for row, length in enumerate(lengths):
        own = (length - 1) // 128
        assert (new_q2k[:, row] == own).any(dim=-1).all(), f"row {row} lost its own block"


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("high_pages", [False, True], ids=["small-pid", "int64-pid"])
def test_partial_page_tail_poison_cannot_change_live_scores_or_selections(heads, high_pages):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.minimax_indexer import _layout, compile_minimax_index_aot

    capacity, max_pages, stride = 8, 4, 7
    frontiers = [65, 129, 201]
    rows = len(frontiers)
    widths = [(length + 63) // 64 for length in frontiers]
    # Only nine pages are initialized, even for a pool beyond signed byte addressing.
    first_pid = 2**31 // 8448 + 17 if high_pages else 3
    ids = torch.arange(sum(widths), device="cuda", dtype=torch.int64).flip(0) * 3 + first_pid
    pool_pages = first_pid + sum(widths) * 3 + 1
    cache = torch.empty((pool_pages, 8448), dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20261114)
    keys = torch.randn((sum(widths), 64, 128), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((sum(widths), 64), generator=generator, device="cuda") + .5
    clean = torch.empty((sum(widths), 8448), dtype=torch.uint8, device="cuda")
    clean[:, :8192] = keys.view(torch.uint8).reshape(-1, 8192)
    clean[:, 8192:] = scales.view(torch.uint8).reshape(-1, 256)
    table = torch.full((rows, stride), -1, dtype=torch.int32, device="cuda")
    tails, cursor = [], 0
    for row, (length, width) in enumerate(zip(frontiers, widths)):
        table[row, :width] = ids[cursor:cursor + width].int()
        last, valid = cursor + width - 1, length % 64
        clean[last, valid * 128:8192] = 0
        clean[last, 8192 + valid * 4:] = 0
        tails.append((last, valid))
        cursor += width
    lengths = torch.tensor(frontiers, dtype=torch.int32, device="cuda")
    query = torch.randn((rows, heads, 128), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    query_scales = torch.rand((rows, heads), generator=generator, device="cuda") + .5
    immutable = [tensor.clone() for tensor in (query, query_scales, table, lengths)]
    blocks, offsets, nbytes = _layout(capacity, max_pages, "decode", heads)
    program = compile_minimax_index_aot(max_rows=capacity, max_pages=max_pages, mode="decode", heads=heads)
    output = torch.full((heads, capacity, 16), -7, dtype=torch.int32, device="cuda")
    scratch = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def launch():
        program.launch(query, query_scales, cache, table, lengths, output, scratch,
                       scalars=(rows, max_pages, stride, pool_pages))

    def snapshot():
        torch.cuda.synchronize()
        page_scores = scratch[offsets["pages"]:offsets["pages"] + heads * rows * max_pages * 4].view(
            torch.float32).reshape(heads, rows, max_pages)
        for row, width in enumerate(widths):
            assert torch.isfinite(page_scores[:, row, :width]).all(), (row, "nonfinite live page score")
            assert torch.isneginf(page_scores[:, row, width:]).all(), (row, "non-live page scored")
        assert (output[:, rows:] == -7).all(), "inactive capacity rows were overwritten"
        for row, length in enumerate(frontiers):
            count = (length + 127) // 128
            expected = list(range(count)) + [-1] * (16 - count)
            assert output[:, row].cpu().tolist() == [expected] * heads
        result = [output.view(torch.uint8).clone(), (output[:, :rows] >= 0).sum(-1)]
        for name, size in (("pages", heads * rows * max_pages * 4),
                           ("scores", heads * rows * blocks * 4),
                           ("starts", heads * rows * 4), ("lengths", heads * rows * 4)):
            result.append(scratch[offsets[name]:offsets[name] + size].clone())
        return result

    cache.index_copy_(0, ids, clean)
    launch()
    eager = snapshot()
    with kernel_resolution_guard("MiniMax scalar packed-page poison noninterference"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        baseline = snapshot()
        for actual, expected in zip(baseline, eager):
            assert torch.equal(actual, expected), "clean eager/replay mismatch"
        # E4M3FN +/-NaN and +/-448, paired with nonfinite or enormous scales.
        for key_byte, scale in ((0x7f, 1e30), (0xff, float("inf")), (0x7e, float("nan")),
                                (0xfe, 1e30), (0x7e, float("inf")), (0xff, float("nan"))):
            poisoned = clean.clone()
            for last, valid in tails:
                poisoned[last, valid * 128:8192] = key_byte
                poisoned[last, 8192:].view(torch.float32)[valid:] = scale
            assert not torch.equal(poisoned, clean)
            cache.index_copy_(0, ids, poisoned)
            graph.replay()
            actual = snapshot()
            for got, expected in zip(actual, baseline):
                assert torch.equal(got, expected), (key_byte, scale, "tail changed a published result")
            assert torch.equal(cache.index_select(0, ids), poisoned), "index scoring mutated packed pages"
        # Replay must recompute scores rather than retaining a clean result.
        cache.index_copy_(0, ids, clean)
        query.copy_((-query.float()).to(torch.float8_e4m3fn))
        graph.replay()
        changed = snapshot()
        assert not torch.equal(changed[2], baseline[2]), "query mutation did not change live page scores"
        query.copy_(immutable[0])
        graph.replay()
        for actual, expected in zip(snapshot(), baseline):
            assert torch.equal(actual, expected), "restored query did not restore exact scores"
    for actual, expected in zip((query, query_scales, table, lengths), immutable):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("mode,rows", [
    *(('decode', rows) for rows in range(1, 65)),
    *(('prefill', rows) for rows in (1, 63, 64, 65, 2047, 2048, 2049,
                                    2751, 2752, 2753, 4095, 4096)),
])
def test_half_head_scores_and_selections_are_byte_exact_whole_head_slices(mode, rows):
    require_b12x()
    from b12x.integration.cuteafd.minimax_indexer import _layout, compile_minimax_index_aot

    capacity, pages = (64 if mode == "decode" else 4096), 96
    gen = torch.Generator(device="cuda").manual_seed(20261113)
    query = torch.randn((rows, 4, 128), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((rows, 4), generator=gen, device="cuda") + .5
    key = torch.randn((pages * 64, 128), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    key_scales = torch.rand(pages * 64, generator=gen, device="cuda") + .5
    starts = torch.tensor([0 if r % 2 else 2816 for r in range(rows)], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2600 + r % 100 for r in range(rows)], dtype=torch.int32, device="cuda")
    ends = starts + lengths
    table = torch.stack([torch.randperm(pages, generator=gen, device="cuda").int() for _ in range(rows)])
    cache = torch.empty((pages, 8448), dtype=torch.uint8, device="cuda")
    cache[:, :8192] = key.view(torch.uint8).reshape(pages, 8192)
    cache[:, 8192:] = key_scales.view(torch.uint8).reshape(pages, 256)

    def run(first, heads):
        program = compile_minimax_index_aot(max_rows=capacity, max_pages=pages, mode=mode, heads=heads)
        blocks, offsets, nbytes = _layout(capacity, pages, mode, heads)
        output = torch.full((heads, capacity, 16), -7, dtype=torch.int32, device="cuda")
        scratch = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        q = query[:, first:first + heads].contiguous()
        qs = scales[:, first:first + heads].clone(memory_format=torch.contiguous_format)
        operands = ((q, qs, cache, table, lengths) if mode == "decode" else
                    (q, qs, key, key_scales, starts, ends))
        frozen = [tensor.clone() for tensor in operands]
        scalars = (rows, pages, pages, pages) if mode == "decode" else (rows, int(ends.max().item()))
        program.launch(*operands, output, scratch, scalars=scalars)
        torch.cuda.synchronize()
        assert (output[:, rows:] == -7).all(), "inactive index rows were overwritten"
        assert (output[:, :rows] >= 0).all(), "live rows must select 16 nonempty blocks"
        for actual, expected in zip(operands, frozen):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        scores = scratch[offsets["scores"]:offsets["scores"] + heads * rows * blocks * 4]
        result = [output.view(torch.uint8).reshape(heads, -1), scores.reshape(heads, -1)]
        if mode == "decode":
            page_scores = scratch[offsets["pages"]:offsets["pages"] + heads * rows * pages * 4]
            result.append(page_scores.reshape(heads, -1))
        return result

    whole = run(0, 4)
    for first in (0, 2):
        half = run(first, 2)
        for actual, reference in zip(half, whole):
            assert torch.equal(actual, reference[first:first + 2]), (mode, first)


@pytest.mark.parametrize("tied", [False, True])
def test_signed_heads_and_low_id_ties_match_independent_oracle(programs, tied):
    program = programs[0]
    heads = program.geometry["heads"]
    blocks, length = 21, 20 * 128 + 17
    pages = (length + 63) // 64
    cache = torch.zeros((pages, 8448), dtype=torch.uint8, device="cuda")
    block_values = torch.ones(blocks, device="cuda") if tied else torch.arange(
        -10, 11, dtype=torch.float32, device="cuda")
    keys = block_values.repeat_interleave(128)[:pages * 64, None].expand(-1, 128)
    # The final physical page is padded, but only its first 17 keys are causal.
    keys = torch.nn.functional.pad(keys, (0, 0, 0, pages * 64 - keys.shape[0]))
    cache[:, :8192] = keys.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).reshape(pages, 8192)
    cache[:, 8192:] = torch.ones((pages, 64), device="cuda").view(torch.uint8).reshape(pages, 256)
    signs = torch.tensor([1., -1., 1., -1.][:heads], device="cuda")
    q = signs[:, None].expand(heads, 128).contiguous().to(torch.float8_e4m3fn)[None]
    scales = torch.ones((1, heads), device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    output = torch.full((heads, MAX_ROWS, 16), -7, dtype=torch.int32, device="cuda")
    scratch = torch.empty(program.scratch_bytes(MAX_ROWS)["scratch"], dtype=torch.uint8, device="cuda")

    def launch():
        program.launch(q.view(torch.uint8), scales, cache, table, lengths, output, scratch,
                       scalars=(1, pages, pages, pages))

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Mutate input under stable pointers: replay must compute, not reuse old scores.
    for multiplier in [1., -1.]:
        q.copy_((signs[:, None] * multiplier).expand(heads, 128).to(torch.float8_e4m3fn)[None])
        graph.replay()
        actual = output[:, 0].cpu().tolist()
        for head, sign in enumerate([1., -1., 1., -1.][:heads]):
            scores = [(float(block_values[b]) * sign * multiplier, b) for b in range(blocks - 1)]
            picked = [b for _, b in sorted(scores, key=lambda item: (-item[0], item[1]))[:15]]
            expected = sorted(picked + [blocks - 1])
            assert actual[head] == expected
        assert (output[:, 1:] == -7).all(), "inactive capacity rows were overwritten"


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("tied", [False, True])
def test_contiguous_prefill_matches_independent_signed_block_oracle(tied, heads):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.minimax_indexer import compile_minimax_index_aot

    capacity, max_pages = 16, 96
    program = compile_minimax_index_aot(max_rows=capacity, max_pages=max_pages, mode="prefill", heads=heads)
    extent = max_pages * 64
    generator = torch.Generator(device="cuda").manual_seed(20261112)
    keys = torch.randn((extent, 128), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    if tied:
        keys.fill_(1)
    key_scales = torch.rand(extent, generator=generator, device="cuda") + .5
    if tied:
        key_scales.fill_(1)
    query = torch.empty((capacity, heads, 128), dtype=torch.float8_e4m3fn, device="cuda")
    query_scales = torch.rand((capacity, heads), generator=generator, device="cuda") + .5
    starts = torch.empty(capacity, dtype=torch.int32, device="cuda")
    ends = torch.empty_like(starts)
    output = torch.full((heads, capacity, 16), -7, dtype=torch.int32, device="cuda")
    scratch = torch.empty(program.scratch_bytes(capacity)["scratch"], dtype=torch.uint8, device="cuda")
    cases = [([0], [65]), ([0, 2816, 2816, 0], [2700, 2817, 2816 + 129, 257])]
    signs = torch.tensor([1., -1., 1., -1.][:heads], device="cuda")
    for first, last in cases:
        rows, k_rows = len(first), max(last)
        query.copy_(signs[None, :, None].expand(capacity, heads, 128).to(torch.float8_e4m3fn))
        starts[:rows] = torch.tensor(first, dtype=torch.int32, device="cuda")
        ends[:rows] = torch.tensor(last, dtype=torch.int32, device="cuda")
        output.fill_(-7)

        def launch():
            program.launch(query, query_scales, keys, key_scales, starts, ends, output, scratch,
                           scalars=(rows, k_rows))

        with kernel_resolution_guard("MiniMax prefill index live rows and graph replay"):
            launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            for _ in range(2):
                query.copy_((-query.float()).to(torch.float8_e4m3fn))
                graph.replay()
                torch.cuda.synchronize()
                actual = output[:, :rows].cpu().tolist()
                for row, (begin, end) in enumerate(zip(first, last)):
                    k = keys[begin:end].float() * key_scales[begin:end, None]
                    scores = (query[row].float() @ k.T) * query_scales[row, :, None] * 128 ** -.5
                    block_scores = torch.stack([scores[:, p:p + 128].amax(-1)
                                                for p in range(0, end - begin, 128)], dim=-1).cpu().tolist()
                    for head in range(heads):
                        own = len(block_scores[head]) - 1
                        history = sorted(range(own), key=lambda b: (-block_scores[head][b], b))[:15]
                        picked = sorted(history + [own])
                        expected = picked + [-1] * (16 - len(picked))
                        assert actual[head][row] == expected, (row, head, actual[head][row], expected)
                assert (output[:, rows:] == -7).all(), "inactive index rows were overwritten"
