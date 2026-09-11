"""Byte-exact slice staging, crossing N256 and 32-bit global offsets."""
import pytest
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from b12x._lib.intrinsics import shared_ptr_to_u32
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_staging import (
    stage_repacked_b_slice, stage_repacked_sfb_slice,
    stage_repacked_b_k_slice, stage_repacked_sfb_k_slice,
)


class SliceProbe:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, weights: cute.Tensor, scales: cute.Tensor,
                 metadata: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(weights, scales, metadata, output).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, weights: cute.Tensor, scales: cute.Tensor,
               metadata: cute.Tensor, output: cute.Tensor):
        thread = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        buffer = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(self.width * 17 + 32), byte_alignment=16)
        for index in range(thread, self.width * 17 + 32, 32):
            buffer[index] = cutlass.Int32(-12345)
        cute.arch.sync_threads()
        base = shared_ptr_to_u32(buffer.iterator)
        stage_repacked_b_slice(weights, base + 64, metadata[0],
            cutlass.Int32(metadata[2]), cutlass.Int32(metadata[3]),
            cutlass.Int32(metadata[4]), thread, 32, self.width)
        stage_repacked_sfb_slice(scales, base + 64 + self.width * 64, metadata[1],
            cutlass.Int32(metadata[2]), cutlass.Int32(metadata[3]),
            cutlass.Int32(metadata[4]), thread, 32, self.width)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        for index in range(thread, self.width * 17 + 32, 32):
            output[index] = buffer[index]


def indices(width, k_tiles, k_tile, n_start, bbase=0, sbase=0):
    # Independent scalar interpretation of the documented packed layout.
    b = []
    for kb in range(4):
        for chunk in range(width // 32):
            n = n_start + chunk * 32
            for lane in range(32):
                for n8 in range(4):
                    b.append(bbase + ((n // 256) * k_tiles + k_tile) * 4096
                             + kb * 1024 + (n % 256 // 32) * 128 + lane * 4 + n8)
    s = [sbase + (((n_start + row) // 256) * k_tiles + k_tile) * 256
         + (n_start + row) % 256 for row in range(width)]
    return b, s


@pytest.mark.parametrize('width', [64, 128, 192])
@pytest.mark.parametrize('large_stride', [False, True])
def test_slice_staging(width, large_stride):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('Blackwell GPU required')
    # The large case forces N-tile * K-stride * tile-words beyond INT32_MAX.
    # Allocate sparsely initialized storage, not a materialized huge fixture.
    k_tiles = 524289 if large_stride else 40
    starts = [192, 256, 192] if large_stride else [0, 32, 64, 96, 128, 192, 224, 384, 640, 832, 0]
    specs = [(0, 0, k_tiles - (i % 2 if large_stride else 0), i % 40, n)
             for i, n in enumerate(starts)]
    addresses = [indices(width, *spec[2:], *spec[:2]) for spec in specs]
    weights = torch.empty(max(max(b) for b, _ in addresses) + 1,
                          device='cuda', dtype=torch.int32)
    scales = torch.empty(max(max(s) for _, s in addresses) + 1,
                         device='cuda', dtype=torch.int32)
    metadata = torch.tensor(specs[0], device='cuda', dtype=torch.int64)
    output = torch.empty(width * 17 + 32, device='cuda', dtype=torch.int32)
    views = [from_dlpack(t, assumed_align=16) for t in (weights, scales, metadata, output)]
    compiled = cute.compile(SliceProbe(width), *views, current_cuda_stream())
    compiled(*views, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*views, current_cuda_stream())
    for iteration, (spec, (bi, si)) in enumerate(zip(specs, addresses)):
        ib = torch.tensor(bi, device='cuda', dtype=torch.int64)
        iss = torch.tensor(si, device='cuda', dtype=torch.int64)
        bv = ((ib * 31 + iteration * 7919) % 2147483647).int()
        sv = ((iss * 17 + iteration * 8291) % 2147483647).int()
        weights[ib] = bv
        scales[iss] = sv
        metadata.copy_(torch.tensor(spec, device='cuda'))
        output.fill_(123)
        before = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == before
        assert torch.equal(output[16:16 + width * 16], bv)
        assert torch.equal(output[16 + width * 16:-16], sv)
        assert bool((output[:16] == -12345).all())
        assert bool((output[-16:] == -12345).all())
    graph.reset()


class KSliceProbe:
    def __init__(self, depth):
        self.depth = depth
        self.bwords = 128 * depth // 8
        self.swords = 128 * ((depth + 127) // 128)
        self.count = self.bwords + self.swords + 32

    @cute.jit
    def __call__(self, weights: cute.Tensor, scales: cute.Tensor,
                 metadata: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(weights, scales, metadata, output).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, weights: cute.Tensor, scales: cute.Tensor,
               metadata: cute.Tensor, output: cute.Tensor):
        thread = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        buffer = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(self.count), byte_alignment=16)
        for index in range(thread, self.count, 32):
            buffer[index] = cutlass.Int32(-12345)
        cute.arch.sync_threads()
        base = shared_ptr_to_u32(buffer.iterator)
        stage_repacked_b_k_slice(weights, base + 64, metadata[0],
            cutlass.Int32(metadata[2]), cutlass.Int32(metadata[3]),
            cutlass.Int32(metadata[4]), thread, 32, self.depth)
        sf = cute.make_tensor(buffer.iterator + 16 + self.bwords,
                              cute.make_layout(self.swords))
        stage_repacked_sfb_k_slice(scales, sf, metadata[1],
            cutlass.Int32(metadata[2]), cutlass.Int32(metadata[3]),
            cutlass.Int32(metadata[4]), thread, 32, self.depth)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        for index in range(thread, self.count, 32):
            output[index] = buffer[index]


def k_indices(depth, k_tiles, k_start, n_start, bbase, sbase):
    b, s = [], []
    for kb in range(depth // 32):
        k = k_start // 32 + kb
        for chunk in range(4):
            n = n_start + chunk * 32
            for lane in range(32):
                for n8 in range(4):
                    b.append(bbase + ((n // 256) * k_tiles + k // 4) * 4096
                             + (k % 4) * 1024 + (n % 256 // 32) * 128 + lane * 4 + n8)
    for group in range((depth + 127) // 128):
        for row in range(128):
            n = n_start + row
            for byte in range(4):
                k = k_start // 32 + group * 4 + byte
                address = sbase + ((n // 256) * k_tiles + k // 4) * 256 + n % 256
                s.append((address, k % 4, group * 4 + byte < depth // 32))
    return b, s


@pytest.mark.parametrize('depth', [64, 128, 192])
@pytest.mark.parametrize('large_stride', [False, True])
def test_k_slice_staging(depth, large_stride):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('Blackwell GPU required')
    k_tiles = 524289 if large_stride else 40
    pairs = [(192, 64), (256, 96), (192, 64)] if large_stride else [
        (0, 0), (32, 32), (128, 64), (192, 96), (224, 128), (640, 192), (0, 0)]
    specs = [(16, 16, k_tiles - (i % 2 if large_stride else 0), k, n)
             for i, (n, k) in enumerate(pairs)]
    addresses = [k_indices(depth, *spec[2:], *spec[:2]) for spec in specs]
    weights = torch.empty(max(max(b) for b, _ in addresses) + 1,
                          device='cuda', dtype=torch.int32)
    scales = torch.empty(max(max(a for a, _, _ in s) for _, s in addresses) + 1,
                         device='cuda', dtype=torch.int32)
    metadata = torch.tensor(specs[0], device='cuda', dtype=torch.int64)
    probe = KSliceProbe(depth)
    output = torch.empty(probe.count, device='cuda', dtype=torch.int32)
    views = [from_dlpack(t, assumed_align=16) for t in (weights, scales, metadata, output)]
    compiled = cute.compile(probe, *views, current_cuda_stream())
    compiled(*views, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*views, current_cuda_stream())
    for iteration, (spec, (bi, si)) in enumerate(zip(specs, addresses)):
        ib = torch.tensor(bi, device='cuda', dtype=torch.int64)
        iss = torch.tensor([a for a, _, _ in si], device='cuda', dtype=torch.int64)
        shift = torch.tensor([b * 8 for _, b, _ in si], device='cuda', dtype=torch.int64)
        mask = torch.tensor([valid for _, _, valid in si], device='cuda')
        bv = (ib * 2654435761 + iteration * 7919).int()
        sv = (iss * 2246822519 + iteration * 8291).int()
        weights[ib], scales[iss] = bv, sv
        values = ((sv.long() >> shift) & 255) * mask
        expected_scales = (values.reshape(-1, 4) << torch.tensor(
            [0, 8, 16, 24], device='cuda')).sum(-1).int()
        metadata.copy_(torch.tensor(spec, device='cuda'))
        output.fill_(123)
        before = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == before
        assert torch.equal(output[16:16 + probe.bwords], bv)
        assert torch.equal(output[16 + probe.bwords:-16], expected_scales)
        assert bool((output[:16] == -12345).all())
        assert bool((output[-16:] == -12345).all())
    graph.reset()
