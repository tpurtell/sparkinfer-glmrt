"""Address-only oracle: no cache payload allocation or numerical claims."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch
import pytest

from b12x.attention._shared.mla.native_v41_records import resolve_native_v41_record
from tests.conftest import require_b12x


@cute.kernel
def _resolve(descriptors: cute.Tensor, metadata: cute.Tensor, selected: cute.Tensor,
             bounds: cute.Tensor, output: cute.Tensor):
    row, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    for part in cutlass.range_constexpr(5):
        key = thread + part * 128
        value, scale = resolve_native_v41_record(descriptors[row, None], metadata[row, None],
                                                 selected[row, None], key, bounds[row])
        output[row, key, 0] = value
        output[row, key, 1] = scale


@cute.jit
def _launch(descriptors: cute.Tensor, metadata: cute.Tensor, selected: cute.Tensor,
            bounds: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
    _resolve(descriptors, metadata, selected, bounds, output).launch(
        grid=(metadata.shape[0], 1, 1), block=(128, 1, 1), stream=stream)


@pytest.mark.parametrize("source_format", (1, 2))
def test_native_record_addresses_graph_and_recycled_pages(source_format):
    require_b12x()
    rows = 8
    end = torch.tensor([2048], dtype=torch.uint64, device="cuda")
    source_end = torch.tensor([512], dtype=torch.uint64, device="cuda")
    pages = torch.tensor([50000, 1], dtype=torch.uint32, device="cuda")
    # Payload bases are intentionally synthetic: verify >2GiB address arithmetic
    # without allocating or dereferencing a payload pool.
    base = [0x10000000000 + i * 0x1000000000 for i in range(8)]
    descriptor = base + [end.data_ptr(), pages.data_ptr(), source_end.data_ptr(),
                          16, 67108864, 16, (source_format << 32) | 2]
    descriptors = torch.tensor([descriptor] * rows, dtype=torch.uint64, device="cuda")
    meta = [[2048, 3, 7, 2048+r % 7, 0, 515, 512, 3, 2, 2] for r in range(rows)]
    meta[1][7] = 0
    meta[1][5] = 512
    meta[2][9] = 0  # malformed stride must not divide by zero
    meta[3][0] = 2047  # stale window
    meta[4][5] = 516  # noncausal source count
    meta[5][8] = 15  # private capacity overflow
    meta[6] = [2048, 0, 7, 2050, 0, 256, 256, 0, 0, 1]  # earlier committed prefix
    metadata = torch.tensor(meta, dtype=torch.uint64, device="cuda")
    selection = list(range(510)) + [512, 514]
    selection[4], selection[5] = -1, 515
    selected = torch.tensor([selection]*rows, dtype=torch.int32, device="cuda")
    bounds_host = [0, 2040, 0, 0, 0, 0, 2048, 2049]
    bounds = torch.tensor(bounds_host, dtype=torch.uint64, device="cuda")
    output = torch.empty(rows, 640, 2, dtype=torch.uint64, device="cuda")
    tensors = tuple(from_dlpack(x) for x in (descriptors, metadata, selected, bounds, output))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_launch, *tensors, stream)

    def check(page_ids):
        expected = torch.zeros(rows, 640, 2, dtype=torch.uint64)
        for r in (0, 1, 6):
            m = meta[r]
            for key in range(640):
                if key < 128:
                    pos = m[3] + 1 - 128 + key
                    if pos < bounds_host[r]:
                        continue
                    source, physical = (0, pos % 128) if pos < 2048 else (1, m[1]+pos-2048)
                else:
                    pos = selection[key-128]
                    if not 0 <= pos < m[5]:
                        continue
                    if pos >= m[6]:
                        source, physical = 3, m[8]+(pos-m[6])*m[9]
                    else:
                        source, physical = 2, page_ids[pos//256]*256+pos%256
                        if physical >= descriptor[12]:
                            continue
                vb, sb = (256, 32) if source >= 2 and source_format == 2 else (512, 16)
                expected[r, key, 0] = base[source]+physical*vb
                expected[r, key, 1] = base[source+4]+physical*sb
        assert torch.equal(output.cpu(), expected)

    compiled(*tensors, stream)
    check([50000, 1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    pages.copy_(torch.tensor([2, 400000], dtype=torch.uint32, device="cuda"))
    graph.replay()
    check([2, 400000])
