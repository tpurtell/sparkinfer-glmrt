"""Dynamic row extents retain legacy math and cover the official image range."""
import pytest
import torch
from ..conftest import require_b12x


def _l2():
    from cuda.bindings import runtime
    status,size=runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,torch.cuda.current_device())
    assert int(status)==0
    return size


@pytest.mark.parametrize("heads,dim",[(12,128),(16,80),(16,128)])
def test_runtime_rows_tile_boundaries_and_legacy_exactness(heads,dim):
    require_b12x()
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    from b12x.integration.cuteafd.media import compile_vision_attention_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    generator=torch.Generator().manual_seed(811)
    q,k,v=[torch.randn(16384,heads,dim,generator=generator).bfloat16().cuda() for _ in range(3)]
    dynamic=compile_dynamic_vision_attention_aot(heads=heads,dim=dim)
    legacy=compile_vision_attention_aot(heads=heads,dim=dim)
    cu=torch.empty(2,device="cuda",dtype=torch.int32)
    full=torch.empty_like(q)
    legacy_lse=torch.empty(heads,16384,device="cuda")
    for rows in (1,63,64,65,127,128,129,255,256,257):
        cu.copy_(torch.tensor([0,rows],device="cuda",dtype=torch.int32))
        out=torch.empty(rows,heads,dim,device="cuda",dtype=torch.bfloat16)
        lse=torch.empty(heads,rows,device="cuda")
        with kernel_resolution_guard():
            dynamic.launch(q,k,v,out,lse,cu,scalars=[rows,dim**-.5,_l2()])
            legacy.launch(q,k,v,full,legacy_lse,cu,scalars=[dim**-.5,_l2()])
        assert torch.equal(out,full[:rows]), (heads,dim,rows)
        assert torch.equal(lse,legacy_lse[:,:rows]), (heads,dim,rows,"lse")


@pytest.mark.parametrize("heads,dim",[(12,128),(16,80)])
def test_official_maximum_65536_rows_without_new_compile_key(heads,dim):
    require_b12x()
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    rows=65536
    q=torch.zeros(rows,heads,dim,device="cuda",dtype=torch.bfloat16)
    k=torch.zeros_like(q)
    values=torch.arange(heads,device="cuda",dtype=torch.float32).div(16).bfloat16()
    v=values[None,:,None].expand(rows,heads,dim).contiguous()
    out=torch.empty_like(q)
    lse=torch.empty(heads,rows,device="cuda")
    cu=torch.tensor([0,rows],device="cuda",dtype=torch.int32)
    program=compile_dynamic_vision_attention_aot(heads=heads,dim=dim)
    with kernel_resolution_guard():
        program.launch(q,k,v,out,lse,cu,scalars=[rows,dim**-.5,_l2()])
    assert torch.equal(out,v)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("heads,dim",[(0,128),(12,80),(16,96)])
def test_invalid_dynamic_geometry(heads,dim):
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    with pytest.raises(ValueError):
        compile_dynamic_vision_attention_aot(heads=heads,dim=dim)
