"""Native 32x32 checkpoint scales must never use the K128 replication shortcut."""
import pytest
import torch

from b12x.gemm._shared.block_fp8 import (
    BlockFP8LinearScratchCaps, block_fp8_linear_mxfp8,
    pack_block_fp8_linear_weight_mxfp8, plan_block_fp8_linear_scratch,
)
from tests.conftest import require_b12x


@pytest.mark.parametrize('tokens,n,k', [(1,256,256), (4,256,256), (16,256,256),
                                      (80,256,256), (256,256,256),
                                      (1,25600,6144), (16,25600,6144)])
def test_native_v41_scales_gemm_and_graph(tokens, n, k):
    require_b12x()
    torch.manual_seed(4100 + tokens)
    x = torch.randn((tokens,k), device='cuda', dtype=torch.bfloat16)
    w = (torch.randn((n,k), device='cuda') * 0.125).to(torch.float8_e4m3fn)
    # Adjacent K32 and N32 blocks deliberately differ inside every K128/N128 block.
    exponents = (torch.arange(n//32, device='cuda')[:,None] +
                 3*torch.arange(k//32, device='cuda')[None,:]) % 5 + 124
    scale = exponents.to(torch.uint8).view(torch.float8_e8m0fnu)
    original = w.view(torch.uint8).clone()
    packed = pack_block_fp8_linear_weight_mxfp8(w, scale, block_size=(32,32))
    assert packed.block_size == (32,32)
    assert torch.equal(w.view(torch.uint8), original)
    assert torch.equal(packed.weight.values.reshape(n,k).view(torch.uint8), original)
    expected_scales = scale.view(torch.uint8).repeat_interleave(32,dim=0)
    assert torch.equal(packed.weight.scale_rows.reshape(n,k//32).view(torch.uint8), expected_scales)
    plan = plan_block_fp8_linear_scratch(BlockFP8LinearScratchCaps(
        device=x.device, max_tokens=tokens, in_features=k, out_features=n, block_size=(32,32)))
    scratch = tuple(torch.empty(shape, dtype=dtype, device='cuda') for shape,dtype in plan.shapes_and_dtypes())
    out = torch.empty((tokens,n,1), device='cuda', dtype=torch.bfloat16)
    binding = plan.bind(scratch=scratch, source=x, packed_weight=packed, output=out,
                        expected_m=tokens, activation_block_size=32)
    weight = w.float() * scale.float().repeat_interleave(32,dim=0).repeat_interleave(32,dim=1)

    def reference():
        blocks = x.float().reshape(tokens,k//32,32)
        scales = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448)))
        recovered = ((blocks/scales[...,None]).to(torch.float8_e4m3fn).float()*scales[...,None]).reshape(tokens,k)
        prior = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            return (recovered @ weight.T).bfloat16()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prior

    def check(actual):
        expected = reference()
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
        similarity = torch.nn.functional.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0)
        assert similarity > 0.99999

    for _ in range(3): binding.run()
    check(out[:,:,0])
    check(block_fp8_linear_mxfp8(source=x,packed_weight=packed,expected_m=tokens))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): binding.run()
    x.mul_(-0.5)
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    assert torch.cuda.memory_allocated() == allocated
    check(out[:,:,0])
    assert torch.equal(w.view(torch.uint8), original)


@pytest.mark.parametrize("slices", [2, 4])
def test_precise_split_k_reduces_every_plane(slices):
    from b12x._lib.dense_gemm import _reduce_split_k_bf16

    require_b12x()
    # Different planes expose an accidentally hard-coded two-plane reducer.
    m, n = 3, 256
    planes = torch.arange(slices * m * n, device="cuda", dtype=torch.float32).reshape(slices, m, n) / 1024
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    _reduce_split_k_bf16(planes.permute(1, 2, 0), out, m=m, n=n)
    torch.testing.assert_close(out[:, :, 0], planes.sum(0).bfloat16(), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _reduce_split_k_bf16(planes.permute(1, 2, 0), out, m=m, n=n)
    planes[-1].add_(2)
    graph.replay()
    torch.testing.assert_close(out[:, :, 0], planes.sum(0).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("n,k", [(5120,15360),(512,5120)])
def test_mxfp8_aot_split_metadata_and_export(n,k,tmp_path):
    from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot
    require_b12x()
    compiled,slices=compile_dense_gemm_mxfp8_aot(
        size_m=1,size_n=n,size_k=k,return_split_k_metadata=True)
    assert slices in (1,2,4)
    compiled.export_to_c(str(tmp_path),"split","ds41_split")
    assert (tmp_path/"split.h").stat().st_size>0
    assert (tmp_path/"split.o").stat().st_size>0
    if slices > 1:
        with pytest.raises(ValueError,match="standalone export does not support split-K"):
            compile_dense_gemm_mxfp8_aot(size_m=1,size_n=n,size_k=k)
    else:
        assert hasattr(compile_dense_gemm_mxfp8_aot(size_m=1,size_n=n,size_k=k),"export_to_c")
