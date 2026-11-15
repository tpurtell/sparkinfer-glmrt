"""F32 residual norm #1915 oracle and exact BF16 KV packing including big pid."""
import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize("width,heads,weighted", [(4608, 1, True), (128, 8, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_norm_reference_replay(width, heads, weighted, dtype):
    require_b12x()
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_norm_aot
    p = compile_inkling2_norm_aot(width=width, heads=heads, weighted=weighted, output_dtype=dtype)
    for rows in (1, 17, 2048, 4096):
        x = torch.randn(rows, heads, width, device="cuda")
        x[0] = 0
        w = torch.randn(heads, width, device="cuda").bfloat16() if weighted else None
        out = torch.empty_like(x, dtype=dtype)
        def run():
            p.launch(x, w, out, scalars=(rows,))
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.mul_(.3)
        if weighted:
            w.mul_(2)
        graph.replay()
        want = x*torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6)
        if weighted:
            want *= w.float()
        torch.testing.assert_close(out.float(), want, atol=2e-6 if dtype==torch.float32 else .02,
                                   rtol=2e-6 if dtype==torch.float32 else .008)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int8, torch.uint8,
                                   torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(16, 1024, 64), (3, 7, 5), (9, 17, 9)])
def test_padded_transpose_is_byte_exact_and_graph_replayable(dtype, shape):
    require_b12x()
    from b12x.integration.cuteafd.inkling2_glue import compile_padded_transpose_aot
    rows, cols, padded = shape
    element_bytes = torch.empty((), dtype=dtype).element_size()
    # Arbitrary bit patterns include nonfinite floats and signed-zero payloads;
    # compare bytes, never floating-point equality or a dequantized reference.
    raw = ((torch.arange(rows*cols*element_bytes, dtype=torch.int64)*37+17)%256).to(torch.uint8)
    source = raw.view(dtype).reshape(rows, cols).cuda()
    guard = torch.full((cols*padded*element_bytes+32,), 0x5a, dtype=torch.uint8, device="cuda")
    output = guard[16:-16].view(dtype).reshape(cols, padded)
    program = compile_padded_transpose_aot(dtype=dtype, rows=rows, cols=cols, padded_cols=padded)
    program.launch(source, output, scalars=())
    want = torch.zeros(cols, padded, element_bytes, dtype=torch.uint8)
    want[:, :rows] = raw.reshape(rows, cols, element_bytes).permute(1, 0, 2)
    assert torch.equal(output.view(torch.uint8).cpu().reshape(cols, padded, element_bytes), want)
    assert torch.all(guard[:16] == 0x5a) and torch.all(guard[-16:] == 0x5a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        program.launch(source, output, scalars=())
    source.view(torch.uint8).bitwise_xor_(0xff)
    graph.replay()
    want[:, :rows] = raw.bitwise_xor(0xff).reshape(rows, cols, element_bytes).permute(1, 0, 2)
    assert torch.equal(output.view(torch.uint8).cpu().reshape(cols, padded, element_bytes), want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("big", [False, True])
def test_kv_records_exact_big_slot(dtype, big):
    require_b12x()
    from b12x.integration.cuteafd.inkling2_glue import compile_inkling2_kv_pack_aot
    p = compile_inkling2_kv_pack_aot(input_dtype=dtype)
    rows = 7
    # Element offset > 2^31 (8GiB with BF16), not just >2^31 bytes.
    base = (2**31)//2048 + 3 if big else 3
    cache = torch.empty(base+rows+3, 2048, device="cuda", dtype=torch.bfloat16)
    cache[base-1:base+rows+2].fill_(123)
    k = torch.randn(rows, 8, 128, device="cuda").to(dtype)
    v = torch.randn(rows, 8, 128, device="cuda").to(dtype)
    slots = torch.arange(base, base+rows, device="cuda", dtype=torch.int64)
    slots[2] = -1
    p.launch(k, v, slots, cache, scalars=(rows,))
    expected = torch.cat([k.flatten(1), v.flatten(1)], 1).bfloat16()
    for row in range(rows):
        if row != 2:
            assert torch.equal(cache[base+row], expected[row])
    assert torch.all(cache[base+2] == 123)
    assert torch.all(cache[base-1] == 123)
    assert torch.all(cache[base+rows] == 123)
