"""Native-oracle and high-address checks for the private V4.1 score kernel."""
import ctypes as C
import os
from pathlib import Path

import pytest
import torch
import cutlass.cute as cute
from cutlass import Uint8, Uint32, Int32, Int64, BFloat16, Float32

from b12x._lib.compiler import compile as compile_kernel, KernelCompileSpec
from b12x._lib.utils import make_ptr, current_cuda_stream
from b12x.attention.dsa_indexer._v41_overlay import V41OverlayScore

DTYPES = [Uint8, Uint8, BFloat16, Uint8, Uint8, Uint32, Int64, Int64,
          Int64, Float32, Uint8, Uint8]


@pytest.fixture(scope='module')
def kernels():
    path = os.environ.get('DS41RT_INDEX_BASELINE_LIB')
    if not path or not Path(path).is_file():
        pytest.skip('requires DS41RT_INDEX_BASELINE_LIB with native overlay scorer')
    lib = C.CDLL(path)
    native = lib.ds41rt_v41_index_scores_overlay
    native.argtypes = [C.c_void_p]*12 + [C.c_int32]*4 + [C.c_uint64]*2 + [C.c_void_p]
    native.restype = C.c_int32
    raw = compile_kernel(V41OverlayScore(),
        *[make_ptr(t,16,cute.AddressSpace.gmem,assumed_align=1) for t in DTYPES],
        Int32(1),Int32(1),Int32(1),Int32(1),Int64(256),Int64(1),current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key('attention.indexer.v41_overlay',1,()))
    return raw, native


def launch(raw, tensors, rows, width, capacity, proposal_capacity=256):
    raw(*[make_ptr(t,x.data_ptr(),cute.AddressSpace.gmem,assumed_align=1)
          for x,t in zip(tensors,DTYPES)], rows,width,1,capacity//256,
        capacity,proposal_capacity,current_cuda_stream())


def inputs(rows, width):
    torch.manual_seed(4191+rows+width)
    q=torch.randint(256,(rows,32,64),device='cuda',dtype=torch.uint8)
    qs=torch.randint(120,128,(rows,32,4),device='cuda',dtype=torch.uint8)
    w=torch.randn((rows,32),device='cuda',dtype=torch.bfloat16)
    k=torch.randint(256,(256,64),device='cuda',dtype=torch.uint8)
    ks=torch.randint(120,128,(256,4),device='cuda',dtype=torch.uint8)
    pages=torch.zeros(1,device='cuda',dtype=torch.int32)
    lengths=torch.tensor([256],device='cuda',dtype=torch.int64)
    meta=torch.tensor([0,384,256,128,0,2],device='cuda',dtype=torch.int64).expand(rows,6).contiguous()
    positions=torch.arange(width,device='cuda',dtype=torch.int64).expand(rows,width).contiguous()
    positions[:,1::67]=-1
    out=torch.empty((rows,width),device='cuda')
    p=torch.randint(256,(256,64),device='cuda',dtype=torch.uint8)
    ps=torch.randint(120,128,(256,4),device='cuda',dtype=torch.uint8)
    return [q,qs,w,k,ks,pages,lengths,meta,positions,out,p,ps]


@pytest.mark.parametrize('rows,width',[(1,1),(6,65),(16,255),(80,513),(256,1024),(1024,4096)])
def test_overlay_replay(kernels, rows, width):
    raw,native=kernels
    t=inputs(rows,width)
    ref=torch.empty_like(t[9])
    graph=torch.cuda.CUDAGraph()
    launch(raw,t,rows,width,256)
    with torch.cuda.graph(graph):
        launch(raw,t,rows,width,256)
    pointers=[x.data_ptr() for x in t]
    for cycle in range(6):
        t[0].bitwise_xor_(255)
        t[2].neg_()
        if cycle==1: t[5].fill_(-1)  # invalid committed page, proposals still reachable
        if cycle==2: t[7][:,5]=0     # malformed proposal descriptor
        if cycle==3:
            t[5].zero_();t[7][:,5]=2;t[7][:,2]=255  # stale committed start
        if cycle==4:
            t[7][:,2]=256;t[7][:,1]=0              # fully masked causal window
        if cycle==5:
            t[7][:,1]=384;t[8][:,0]=383            # final strided proposal row
        n=t[:];n[9]=ref
        assert native(*[x.data_ptr() for x in n],rows,width,1,1,256,256,
                      torch.cuda.current_stream().cuda_stream)==0
        t[9].fill_(float('nan'))
        before=torch.cuda.memory_allocated()
        graph.replay();torch.cuda.synchronize()
        assert torch.cuda.memory_allocated()==before
        assert pointers==[x.data_ptr() for x in t]
        torch.testing.assert_close(t[9],ref,rtol=0,atol=0)
        assert not torch.isnan(t[9]).any()
        if cycle in [0,5]:
            assert torch.isfinite(t[9]).any() and (t[9][torch.isfinite(t[9])]!=0).any()


def test_high_physical_page(kernels):
    raw,native=kernels
    t=inputs(6,65)
    launch(raw,t,6,65,256)
    expected=t[9].clone()
    # Park live keys just beyond the signed-32-bit byte-offset boundary.
    base=2**31//64+256
    capacity=base+256
    pool=torch.empty((capacity,64),device='cuda',dtype=torch.uint8)
    scales=torch.empty((capacity,4),device='cuda',dtype=torch.uint8)
    pool[base:].copy_(t[3]);scales[base:].copy_(t[4])
    t[3]=pool;t[4]=scales
    # Only logical page 0 is read; reserve a valid table for the declared stride.
    t[5]=torch.zeros(capacity//256,device='cuda',dtype=torch.int32)
    t[5][0]=base//256
    t[9].fill_(float('nan'))
    launch(raw,t,6,65,capacity)
    torch.testing.assert_close(t[9],expected,rtol=0,atol=0)
