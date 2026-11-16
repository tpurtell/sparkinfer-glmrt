"""Remaining PE forward boundaries against pinned vendor activations."""
from __future__ import annotations
import pytest
import torch
from ..conftest import require_b12x
from . import perception_vendor as vendor


@pytest.mark.parametrize("hidden,vendor_rsqrt",[(1024,True),(1536,False)])
@pytest.mark.parametrize("rows",[1,17,257])
def test_centered_layer_norm_matches_vendor(hidden,vendor_rsqrt,rows):
    require_b12x()
    from b12x.integration.cuteafd.perception_forward import compile_layer_norm_aot
    gen=torch.Generator(device="cuda").manual_seed(2325)
    x=torch.randn(rows,hidden,device="cuda",generator=gen).bfloat16()
    gamma=torch.randn(hidden,device="cuda",generator=gen).bfloat16()
    beta=torch.randn_like(gamma)
    norm=torch.nn.LayerNorm(hidden,eps=1e-5,device="cuda",dtype=torch.bfloat16)
    norm.weight.data.copy_(gamma);norm.bias.data.copy_(beta)
    if vendor_rsqrt:
        # The vendor hook returns normalized output separately from residual.
        ref=vendor.inkling().scaled_residual_norm_(x.clone(),torch.zeros_like(x),torch.ones_like(gamma),norm)
    else:
        ref=norm(x)
    out=torch.empty_like(x)
    program=compile_layer_norm_aot(hidden=hidden,vendor_rsqrt=vendor_rsqrt)
    program.launch(x,gamma,beta,out,scalars=[rows,1e-5])
    ulp=vendor.metrics(f"centered_norm_h{hidden}_rows{rows}",out,ref,source="vendor-Triton" if vendor_rsqrt else "torch.nn.LayerNorm")
    assert ulp<=2
    torch.testing.assert_close(out,ref,atol=.015625,rtol=.008)
    from b12x._lib.runtime_control import kernel_resolution_guard
    with kernel_resolution_guard("PE centered norm live rows"):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):program.launch(x,gamma,beta,out,scalars=[rows,1e-5])
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(out,ref,atol=.015625,rtol=.008)


@pytest.mark.parametrize("patch,hwc",[(20,True),(14,False)])
def test_normalized_patch_order(patch,hwc):
    require_b12x()
    from b12x.integration.cuteafd.perception_forward import compile_normalized_patches_aot
    h,w=patch*4,patch*6
    x=(torch.arange(h*w*3,device="cuda")%255).float().reshape(h,w,3)/100
    out=torch.empty(24,3*patch*patch,device="cuda",dtype=torch.bfloat16)
    program=compile_normalized_patches_aot(patch=patch,hwc=hwc)
    program.launch(x,out,scalars=[h,w])
    order=(0,2,1,3,4) if hwc else (0,2,4,1,3)
    ref=x.reshape(4,patch,6,patch,3).permute(order).reshape_as(out).bfloat16()
    assert vendor.metrics(f"patch_p{patch}_hwc{hwc}",out,ref,source="vendor-native-patch-layout")==0


@pytest.mark.parametrize("height,width",[(2,4),(32,32),(36,36),(52,52)])
def test_inkling_learned_positions_class_and_pre_norm(height,width):
    require_b12x()
    from b12x.integration.cuteafd.perception_forward import compile_learned_position_aot,compile_layer_norm_aot
    hidden=1024;rows=height*width+1
    gen=torch.Generator(device="cuda").manual_seed(2326)
    patch=torch.randn(rows-1,hidden,device="cuda",generator=gen).bfloat16()
    table=torch.randn(32,32,hidden,device="cuda",generator=gen).bfloat16()
    cls=torch.randn(hidden,device="cuda",generator=gen).bfloat16();cls_pos=torch.randn_like(cls)
    y,x=torch.meshgrid(torch.arange(height,device="cuda"),torch.arange(width,device="cuda"),indexing="ij")
    meta=torch.empty(rows,4,device="cuda",dtype=torch.int32)
    meta[0,:2]=-1;meta[1:,0]=y.flatten();meta[1:,1]=x.flatten();meta[:,2]=height;meta[:,3]=width
    patch_rows=torch.arange(-1,rows-1,device="cuda",dtype=torch.int32);patch_rows[0]=0
    norm=torch.nn.LayerNorm(hidden,eps=1e-5,device="cuda",dtype=torch.bfloat16)
    first=torch.nn.LayerNorm(hidden,eps=1e-5,device="cuda",dtype=torch.bfloat16)
    ref,_=vendor.inkling().embed_tokens(patch,patch_rows,table,meta,cls,cls_pos,norm,first)
    raw=torch.empty(rows,hidden,device="cuda",dtype=torch.bfloat16)
    out=torch.empty_like(raw)
    program=compile_learned_position_aot(hidden=hidden,grid=32,class_token=True)
    program.launch(patch,table,cls,cls_pos,raw,scalars=[height,width])
    normalized=compile_layer_norm_aot(hidden=hidden,vendor_rsqrt=True)
    normalized.launch(raw,norm.weight,norm.bias,out,scalars=[rows,1e-5])
    assert vendor.metrics(f"position_norm_inkling_{height}x{width}",out,ref)<=2
    torch.testing.assert_close(out,ref,atol=.015625,rtol=.008)
