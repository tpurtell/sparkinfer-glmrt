"""Actual four-source payload reads versus the upstream interleaved producer."""
from dataclasses import replace
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch

from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference as pack
from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel
from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits
from b12x.attention._shared.mla.smem import make_smem_layout
from tests.conftest import require_b12x


def test_native_producer_all_sources_and_graph_replay():
    require_b12x()
    torch.manual_seed(4151)
    rows = 3
    q = (.2*torch.randn(rows, 64, 512, device="cuda")).bfloat16()
    swa = pack(torch.randn(128+rows, 512, device="cuda").bfloat16(), page_size=64, cache_kind="swa")
    source = pack(torch.randn(768, 512, device="cuda").bfloat16(), page_size=256, cache_kind="indexed")
    sr, cr = swa.view(-1, 528), source.view(-1, 288)
    values = [sr[:128, :512].contiguous(), sr[128:128+rows, :512].contiguous(),
              cr[:512, :256].contiguous(), cr[512:528, :256].contiguous()]
    scales = [sr[:128, 512:].contiguous(), sr[128:128+rows, 512:].contiguous(),
              cr[:512, 256:].contiguous(), cr[512:528, 256:].contiguous()]
    end = torch.tensor([128], dtype=torch.uint64, device="cuda")
    source_end = torch.tensor([512], dtype=torch.uint64, device="cuda")
    pages = torch.tensor([0, 1], dtype=torch.uint32, device="cuda")
    descriptor = [x.data_ptr() for x in values+scales] + [end.data_ptr(), pages.data_ptr(),
                    source_end.data_ptr(), rows, 512, 16, (2 << 32) | 2]
    descriptors = torch.tensor([descriptor]*rows, dtype=torch.uint64, device="cuda")
    metadata = torch.tensor([[128, 0, rows, 128+r, 0, 515, 512, 3, 1, 2]
                             for r in range(rows)], dtype=torch.uint64, device="cuda")
    bounds = torch.zeros(rows, dtype=torch.uint64, device="cuda")
    selected = torch.arange(512, dtype=torch.int32, device="cuda").repeat(rows, 1)
    selected[:, -2:] = torch.tensor([512, 514], device="cuda")
    selected[:, 5] = -1
    upstream_selected = selected.clone()
    upstream_selected[:, -2:] = torch.tensor([513, 517], device="cuda")
    swa_indices = torch.arange(128, dtype=torch.int32, device="cuda")[None] + torch.arange(rows, device="cuda", dtype=torch.int32)[:, None] + 1
    partials = [torch.empty(rows, 64, 10, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    lses = [torch.empty(rows, 64, 10, device="cuda") for _ in range(2)]
    traits = make_unified_traits(ModelType.DSV41, ComputeMode.FP8, ScaleFormat.NVFP4_E4M3, fp8_rope=False)
    traits = replace(traits, compute_mode=ComputeMode.FP8, fp8_internal=True,
                     q_nope_stride=528, kv_smem_stride=624, nt_per_warp_xv=traits.nt_per_warp_xv*2)

    def kernel(swa_stride):
        return UnifiedDecodeKernel(traits, make_smem_layout(traits), 64, 1,
            h_blocks=4, num_splits=10, num_heads=64, q_head_dim=512,
            topk=128, extra_topk=512, q_stride=q.stride(), swa_indices_stride0=swa_stride,
            extra_indices_stride0=512, mid_out_stride=partials[0].stride(),
            mid_lse_stride=lses[0].stride(), has_extra=True, pbs_extra=256,
            valid_hpb=16, native_dsv41_fp8=True)

    nt = tuple(from_dlpack(x) for x in (q, descriptors, metadata, selected, bounds, partials[0], lses[0]))
    ut = tuple(from_dlpack(x) for x in (q, swa.flatten(), swa_indices, partials[1], lses[1]))
    extra = (from_dlpack(source.flatten()), from_dlpack(upstream_selected))
    scale = cutlass.Float32(512**-.5 * math.log2(math.e))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    native = cute.compile(kernel(512).call_native_v41, *nt, scale, cutlass.Int32(rows), stream)
    ua = (*ut, scale, cutlass.Float32(1), cutlass.Int32(128), cutlass.Int64(swa.stride(0)),
          *extra, cutlass.Int32(512), cutlass.Int32(2), cutlass.Int64(source.stride(0)), cutlass.Int32(rows))
    upstream = cute.compile(kernel(128).call_extra, *ua, stream)

    def launch():
        current = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        native(*nt, scale, cutlass.Int32(rows), current)
        upstream(*ua, current)

    def check(invalid_row=False):
        active = [0, 2] if invalid_row else [0, 1, 2]
        torch.testing.assert_close(lses[0][active], lses[1][active], atol=0, rtol=0)
        valid = torch.isfinite(lses[0])
        torch.testing.assert_close(partials[0][valid], partials[1][valid], atol=0, rtol=0)
        assert torch.isfinite(partials[0][valid]).all()
        assert partials[0][valid].abs().max() > 0

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    # Changed payloads and query values must be consumed by the same graph.
    q.mul_(.7)
    values[1].zero_()
    sr[128:128+rows, :512].zero_()
    graph.replay()
    check()
    # Recycled pages, bounded replay, and one wholly invalid request change
    # only live data; reuse both captured kernels without recompilation.
    pages.copy_(torch.tensor([1, 0], dtype=torch.uint32, device="cuda"))
    mapped = selected.clone()
    ordinary = (mapped >= 0) & (mapped < 512)
    mapped[ordinary] = mapped[ordinary] ^ 256
    mapped[:, -2:] = torch.tensor([513, 517], device="cuda")
    upstream_selected.copy_(mapped)
    bounds[0] = 127
    swa_indices[0, swa_indices[0] < 127] = -1
    metadata[1, 0] = 127
    swa_indices[1].fill_(-1)
    upstream_selected[1].fill_(-1)
    graph.replay()
    check(invalid_row=True)
    assert torch.isneginf(lses[0][1]).all()
    assert torch.equal(partials[0][1], torch.zeros_like(partials[0][1]))

    # Static 32-head TP2 AOT geometry must preserve the 64-head producer and
    # sink merge, including invalid requests and live row-count transitions.
    from cutlass import BFloat16, Float32, Int32, Uint64
    from b12x._lib.utils import make_ptr, current_cuda_stream
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention._shared.mla.native_v41_aot import compile_native_v41_attention_aot

    full = compile_native_v41_attention_aot()
    half = compile_native_v41_attention_aot(32)
    sink = torch.linspace(-2, 2, 64, device="cuda")
    full_out = torch.empty_like(q)
    half_q = [q[:, rank*32:(rank+1)*32].contiguous() for rank in range(2)]
    half_p = [torch.empty(rows, 32, 10, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    half_lse = [torch.empty(rows, 32, 10, device="cuda") for _ in range(2)]
    half_out = [torch.empty_like(x) for x in half_q]
    types = (BFloat16, Uint64, Uint64, Int32, Uint64, Float32, BFloat16, Float32, BFloat16)
    alignments = (16, 8, 8, 4, 8, 4, 16, 4, 16)

    def run_aot(kernel, query, sinks, partial, lse, output, count):
        tensors = (query, descriptors, metadata, selected, bounds, sinks, partial, lse, output)
        pointers = [make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)
                    for tensor, dtype, align in zip(tensors, types, alignments)]
        kernel(*pointers, Int32(count), current_cuda_stream())

    def run_split(count):
        run_aot(full, q, sink, partials[0], lses[0], full_out, count)
        for rank in range(2):
            run_aot(half, half_q[rank], sink[rank*32:(rank+1)*32],
                    half_p[rank], half_lse[rank], half_out[rank], count)

    def check_split(count):
        joined = torch.cat([x[:count] for x in half_out], dim=1)
        torch.testing.assert_close(joined, full_out[:count], atol=0, rtol=0)
        assert torch.isfinite(joined).all()
        assert joined.abs().max() > 0
        torch.testing.assert_close(torch.cat([x[:count] for x in half_lse], dim=1),
                                   lses[0][:count], atol=0, rtol=0)

    # Warm each compiled pointer ABI before freezing resolution and capture.
    run_split(rows)
    check_split(rows)
    with kernel_resolution_guard("native TP2 local heads prepared"):
        for count in (1, rows, 2, rows):
            run_split(count)
            check_split(count)
        compact_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(compact_graph):
            run_split(rows)
        q.mul_(.6)
        for rank in range(2):
            half_q[rank].copy_(q[:, rank*32:(rank+1)*32])
        sink.add_(.25)
        compact_graph.replay()
        check_split(rows)
