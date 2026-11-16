"""New head geometries reuse the existing varlen attention arithmetic."""
import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("heads,dim",[(12,128),(16,80)])
def test_new_vision_attention_geometries(heads,dim):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_vision_attention_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    from cuda.bindings import runtime
    generator=torch.Generator().manual_seed(591)
    q,k,v=[torch.randn(16384,heads,dim,generator=generator).bfloat16().cuda() for _ in range(3)]
    out=torch.full_like(q,42)
    lse=torch.empty(heads,16384,device="cuda")
    cu=torch.zeros(2,device="cuda",dtype=torch.int32)
    status,l2=runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,torch.cuda.current_device())
    assert int(status)==0
    program=compile_vision_attention_aot(dim=dim,heads=heads)
    for rows in (1,17,80,257):
        cu[1]=rows
        with kernel_resolution_guard():
            program.launch(q,k,v,out,lse,cu,scalars=[dim**-.5,l2])
        qq,kk,vv=[x[:rows].double().transpose(0,1) for x in (q,k,v)]
        probability=(qq@kk.transpose(-1,-2)*(dim**-.5)).softmax(-1)
        expected=(probability@vv).transpose(0,1)
        terms=(probability@vv.abs()).transpose(0,1)
        ulp=torch.exp2(torch.floor(torch.log2(expected.abs().clamp_min(2**-126)))-7)
        assert ((out[:rows].double()-expected).abs()<=terms*(2**-7)+ulp).all()
        assert (out[max(rows,257):]==42).all()


@pytest.mark.parametrize("heads,dim,windowed",[(0,128,False),(12,80,False),(12,128,True),(16,80,True)])
def test_invalid_attention_geometry(heads,dim,windowed):
    from b12x.integration.cuteafd.media import compile_vision_attention_aot
    with pytest.raises(ValueError):
        compile_vision_attention_aot(heads=heads,dim=dim,windowed=windowed)
