"""Native BF16 AOT attention against FP32, decode/verify and compact heads."""
import math

import pytest
import torch

from tests.conftest import require_b12x


@pytest.mark.parametrize("heads", [32, 64])
def test_native_bf16_geometry_and_smem(heads):
    from b12x.attention._shared.mla.native_v41_aot import NativeV41Attention
    from b12x.attention._shared.mla.traits import ComputeMode
    kernel = NativeV41Attention(heads).decode
    assert kernel.traits.compute_mode == ComputeMode.BF16
    assert not kernel.traits.fp8_internal and not kernel.native_dsv41_fp8
    assert kernel.num_splits == 10 and kernel.chunks_per_split == 1
    assert kernel.layout.total_bytes <= 101376


@pytest.mark.parametrize("heads", [32, 64])
def test_native_bf16_f32_oracle(heads):
    require_b12x()
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Uint64
    from b12x._lib.utils import make_ptr, current_cuda_stream
    from b12x.attention._shared.mla.native_v41_aot import compile_native_v41_attention_aot
    from b12x.attention._shared.mla.compressed_reference import (
        pack_deepseek_v41_cache_reference as pack,
        unpack_deepseek_v41_cache_reference as unpack)
    torch.manual_seed(41512)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    rows = 64
    swa = pack(torch.randn(256,512,device=device).bfloat16(),page_size=64,cache_kind="swa")
    src = pack(torch.randn(512,512,device=device).bfloat16(),page_size=256,cache_kind="indexed")
    sk = unpack(swa,page_size=64,cache_kind="swa")
    ck = unpack(src,page_size=256,cache_kind="indexed")
    sr, cr = swa.view(-1,528), src.view(-1,288)
    values = [sr[:128,:512].contiguous(),sr[128:192,:512].contiguous(),cr[:,:256].contiguous(),cr[:1,:256].contiguous()]
    scales = [sr[:128,512:].contiguous(),sr[128:192,512:].contiguous(),cr[:,256:].contiguous(),cr[:1,256:].contiguous()]
    end = torch.tensor([128],device=device,dtype=torch.uint64)
    src_end = torch.tensor([512],device=device,dtype=torch.uint64)
    pages = torch.tensor([0,1],device=device,dtype=torch.uint32)
    descriptor = [x.data_ptr() for x in values+scales]+[end.data_ptr(),pages.data_ptr(),src_end.data_ptr(),64,512,1,(2<<32)|2]
    desc = torch.tensor([descriptor]*rows,device=device,dtype=torch.uint64)
    meta = torch.tensor([[128,0,64,128+r,0,512,512,0,0,1] for r in range(rows)],device=device,dtype=torch.uint64)
    selected = torch.arange(512,device=device,dtype=torch.int32).repeat(rows,1)
    bounds = torch.zeros(rows,device=device,dtype=torch.uint64)
    sink = torch.linspace(-1,1,heads,device=device)
    q = torch.empty(rows,heads,512,device=device,dtype=torch.bfloat16)
    partial = torch.empty(rows,heads,10,512,device=device,dtype=torch.bfloat16)
    lse = torch.empty(rows,heads,10,device=device)
    output = torch.empty_like(q)
    tensors = (q,desc,meta,selected,bounds,sink,partial,lse,output)
    types = (BFloat16,Uint64,Uint64,Int32,Uint64,Float32,BFloat16,Float32,BFloat16)
    aligns = (16,8,8,4,8,4,16,4,16)
    pointers = [make_ptr(t,x.data_ptr(),cute.AddressSpace.gmem,assumed_align=a) for x,t,a in zip(tensors,types,aligns)]
    compiled = compile_native_v41_attention_aot(heads)
    def run(n):
        compiled(*pointers,Int32(n),current_cuda_stream())
    for pattern in ("spread","peaked"):
        q.copy_((torch.randn_like(q.float())*(.03 if pattern=="spread" else .2)).bfloat16())
        if pattern=="peaked":
            target = ck[0]
            q.copy_((q.float()+target*(14*512**.5/target.square().sum())).bfloat16())
        ref, ref_lse = [], []
        for r in range(rows):
            keys = torch.cat((sk[r+1:r+129],ck))
            scores = q[r].float() @ keys.T * 512**-.5
            joined = torch.cat((scores,sink[:,None]),-1)
            p = torch.softmax(joined,-1)
            ref.append(p[:,:-1] @ keys)
            ref_lse.append(torch.logsumexp(joined,-1))
        ref,ref_lse = torch.stack(ref),torch.stack(ref_lse)
        for n in (1,2,4,8,16,64):
            run(n)
            error = (output[:n].float()-ref[:n]).square().mean().sqrt()/ref[:n].square().mean().sqrt()
            assert float(error)<.004
            torch.testing.assert_close(output[:n].float(),ref[:n],rtol=.012,atol=.012)
            actual_lse = torch.logaddexp(torch.logsumexp(lse[:n]*math.log(2),-1),sink[None])
            torch.testing.assert_close(actual_lse,ref_lse[:n],rtol=0,atol=3e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(rows)
        graph.replay()
        torch.testing.assert_close(output.float(),ref,rtol=.012,atol=.012)
        graph.reset()
