"""PE boundary oracles: layer-scale rounding, rotary, merge and conv lowering."""
from __future__ import annotations

import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("width", [1024,1536])
@pytest.mark.parametrize("rows", [1,17,257])
def test_scaled_residual_rounds_product_before_add(width, rows):
    require_b12x()
    from b12x.integration.cuteafd.perception import compile_scaled_residual_aot
    gen = torch.Generator(device="cuda").manual_seed(2318)
    x, delta = [torch.randn(rows+1,width,device="cuda",generator=gen).bfloat16() for _ in range(2)]
    gamma = torch.randn(width,device="cuda",generator=gen).bfloat16()
    # Chosen values distinguish separately rounded multiply from fused addition.
    x[0,0], delta[0,0], gamma[0] = 1.0, 1.0078125, .5
    expected = x[:rows] + delta[:rows]*gamma
    out = torch.full_like(x,42)
    program = compile_scaled_residual_aot(width=width)
    program.launch(x,delta,gamma,out,scalars=[rows])
    assert torch.equal(out[:rows],expected)
    assert (out[rows:] == 42).all()
    one = torch.empty_like(out[:1])
    program.launch(x,delta,gamma,one,scalars=[1])
    assert torch.equal(one,out[:1])
    program.launch(x,delta,gamma,x,scalars=[rows])
    assert torch.equal(x[:rows],expected)


@pytest.mark.parametrize("width", [4096,8960])
@pytest.mark.parametrize("bf16_intermediates", [False,True])
def test_quick_gelu_preserves_bf16_boundaries(width,bf16_intermediates):
    require_b12x()
    from b12x.integration.cuteafd.perception import compile_quick_gelu_aot
    x = torch.linspace(-12,12,width*17,device="cuda").bfloat16().reshape(17,width)
    ref = x*torch.sigmoid(1.702*x) if bf16_intermediates else (x.float()*torch.sigmoid(1.702*x.float())).bfloat16()
    out = torch.empty_like(x)
    program = compile_quick_gelu_aot(width=width,bf16_intermediates=bf16_intermediates)
    program.launch(x,out,scalars=[17])
    torch.testing.assert_close(out,ref,atol=0,rtol=0.008)
    program.launch(x,x,scalars=[17])
    assert torch.equal(x,out)


@pytest.mark.parametrize("dim,class_token", [(64,True),(96,False)])
@pytest.mark.parametrize("height,width", [(2,4),(10,6),(52,52)])
def test_rotary_matches_adjacent_pair_xy_reference(dim,class_token,height,width):
    require_b12x()
    from b12x.integration.cuteafd.perception import compile_rotary_2d_aot
    rows = height*width+int(class_token)
    gen = torch.Generator(device="cuda").manual_seed(2319)
    qkv = torch.randn(rows,3,16,dim,device="cuda",generator=gen).bfloat16()
    q,k,v = [torch.empty((rows,16,dim),device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    program = compile_rotary_2d_aot(heads=16,dim=dim,class_token=class_token)
    program.launch(qkv,q,k,v,scalars=[rows,width])
    y,x = torch.meshgrid(torch.arange(height,device="cuda"),torch.arange(width,device="cuda"),indexing="ij")
    coords = torch.stack((x.flatten(),y.flatten()),-1).float()+int(class_token)
    if class_token:
        coords = torch.cat((torch.zeros(1,2,device="cuda"),coords))
    freq = 10000.0**(-torch.arange(0,dim,4,device="cuda").float()/dim)
    angle = torch.cat((coords[:,:1]*freq,coords[:,1:]*freq),-1)[:,None]
    for index,out in [(0,q),(1,k)]:
        source = qkv[:,index].float()
        even,odd = source[...,0::2],source[...,1::2]
        ref = torch.stack((even*angle.cos()-odd*angle.sin(),even*angle.sin()+odd*angle.cos()),-1).flatten(-2).bfloat16()
        torch.testing.assert_close(out,ref,atol=0.03125,rtol=0.008)
    assert torch.equal(v,qkv[:,2])
    if class_token:
        assert torch.equal(q[0],qkv[0,0]) and torch.equal(k[0],qkv[0,1])


@pytest.mark.parametrize("channels", [16,1536,3072])
@pytest.mark.parametrize("height,width", [(1,1),(3,5),(52,52)])
def test_conv_lowering_matches_torch_unfold(channels,height,width):
    require_b12x()
    from b12x.integration.cuteafd.perception import compile_conv3_stride2_lower_aot
    x = (torch.arange(height*width*channels,device="cuda")%127).bfloat16().reshape(height,width,channels)
    rows = ((height+1)//2)*((width+1)//2)
    out = torch.full((rows+1,channels*9),42,device="cuda",dtype=torch.bfloat16)
    program = compile_conv3_stride2_lower_aot(channels=channels)
    program.launch(x,out,scalars=[height,width])
    ref = torch.nn.functional.unfold(x.permute(2,0,1)[None].float(),3,padding=1,stride=2)[0].T.bfloat16()
    assert torch.equal(out[:rows],ref)
    assert (out[rows:] == 42).all()


@pytest.mark.parametrize("class_token", [False,True])
def test_merge_is_spatial_major_and_excludes_class_token(class_token):
    require_b12x()
    from b12x.integration.cuteafd.perception import compile_merge2_aot
    h,w,c = 6,10,1024
    x = (torch.arange((h*w+int(class_token))*c,device="cuda")%251).bfloat16().reshape(-1,c)
    out = torch.empty((h*w//4,c*4),device="cuda",dtype=torch.bfloat16)
    program = compile_merge2_aot(width=c,class_token=class_token)
    program.launch(x,out,scalars=[h,w])
    ref = x[int(class_token):].reshape(h//2,2,w//2,2,c).permute(0,2,1,3,4).reshape_as(out)
    assert torch.equal(out,ref)


@pytest.mark.parametrize("width", [1024,1536])
def test_scaled_residual_frozen_resolution_graph_and_live_rows(width):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.perception import compile_scaled_residual_aot
    x = torch.randn(257,width,device="cuda").bfloat16()
    delta = torch.randn_like(x)
    gamma = torch.randn(width,device="cuda").bfloat16()
    expected = x+delta*gamma
    out = torch.empty_like(x)
    program = compile_scaled_residual_aot(width=width)
    program.launch(x,delta,gamma,out,scalars=[257])
    with kernel_resolution_guard("PE scaled residual live rows"):
        for live in [1,17,257]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(x,delta,gamma,out,scalars=[live])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:live],expected[:live])
