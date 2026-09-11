"""Wire-row quantization preserves K32 semantics and caller-owned storage."""
import pytest
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_ptr
from b12x._lib.quant.mxfp8_rows import (
    compile_mxfp8_rows_quant_aot, mxfp8_rows_quant_aot_grid,
)


@pytest.mark.parametrize("expected_m", [1, 80])
def test_wire_rows_runtime_counts_and_graph(expected_m):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    k, capacity = 5120, 4096
    stride = k + k // 32
    x = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1].mul_(1e-7)
    x[2].mul_(1e-4)
    x[3].mul_(300)
    wire = torch.full((capacity + 1, stride), 255, device="cuda", dtype=torch.uint8)
    unused = torch.full((16,), 93, device="cuda", dtype=torch.uint8)
    kernel = compile_mxfp8_rows_quant_aot(size_k=k, expected_m=expected_m,
                                        amax_floor=1e-4, wire_rows=True)
    def pointer(dtype, address):
        return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)
    pointers = (pointer(cutlass.BFloat16, x.data_ptr()),
                pointer(cutlass.Uint32, wire.data_ptr()),
                pointer(cutlass.Uint8, wire.data_ptr()+k),
                pointer(cutlass.Uint8, unused.data_ptr()))
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    def run(rows):
        grid = mxfp8_rows_quant_aot_grid(size_k=k, rows=rows, expected_m=expected_m, sm_count=sms)
        kernel(*pointers, rows, grid, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    for rows in [1, 2, 6, 16, 80, 256, 4096, 1]:
        wire.fill_(255)
        run(rows)
        blocks = x[:rows].float().reshape(rows, -1, 32)
        exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448))
        payload = (blocks/torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(wire[:rows, :k], payload.reshape(rows, k), atol=0, rtol=0)
        torch.testing.assert_close(wire[:rows, k:], (exponent+127).to(torch.uint8), atol=0, rtol=0)
        assert (wire[rows:] == 255).all()
        assert (unused == 93).all()
    x[:6].fill_(1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(6)
    graph.replay()
    assert (wire[:6, :k] != 0).all()
    x[:6].zero_()
    before = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert (wire[:6, :k] == 0).all()
    assert (wire[:6, k:] == 105).all()
    assert (unused == 93).all()
