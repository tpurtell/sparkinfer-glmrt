"""PE boundary oracles: layer-scale rounding, rotary, merge and conv lowering."""
from __future__ import annotations

import pytest
import torch
from ..conftest import require_b12x
from . import perception_vendor as vendor


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
    if width == 1024:
        expected = x[:rows].clone()
        vendor.inkling().scaled_residual_norm_(expected,delta[:rows],gamma,None)
    else:
        scale = vendor.step5().PerceptionEncoderLayerScale(width).cuda().bfloat16()
        scale.gamma.data.copy_(gamma)
        expected = x[:rows]+scale(delta[:rows])
    out = torch.full_like(x,42)
    program = compile_scaled_residual_aot(width=width)
    program.launch(x,delta,gamma,out,scalars=[rows])
    assert vendor.metrics(f"scale_h{width}_r{rows}",out[:rows],expected)==0
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
    ref = vendor.step5().forward_native(None,x) if bf16_intermediates else vendor.fused_quick_gelu()(x)
    out = torch.empty_like(x)
    program = compile_quick_gelu_aot(width=width,bf16_intermediates=bf16_intermediates)
    program.launch(x,out,scalars=[17])
    assert vendor.metrics(f"quick_gelu_h{width}_bf16_{bf16_intermediates}",out,ref)<=1
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
    if class_token:
        official = vendor.inkling()
        freq = official._rope_inv_freq(dim,qkv.device)
        y,x = torch.meshgrid(torch.arange(height,device="cuda"),torch.arange(width,device="cuda"),indexing="ij")
        meta = torch.empty(rows,4,device="cuda",dtype=torch.int32)
        meta[0,:2] = -1;meta[1:,0] = y.flatten();meta[1:,1] = x.flatten()
        meta[:,2],meta[:,3] = height,width
        ref = qkv.clone().flatten(1)
        official.apply_rotary_qk_(ref,meta,16)
        ref = ref.reshape(rows,3,16,dim)
        reference_q,reference_k = ref[:,0],ref[:,1]
    else:
        rope = vendor.step5().PerceptionEncoderRope2D(dim,height,width)
        freq = rope._compute_inv_freq(rope.theta,dim//2).cuda()
        rope = rope.cuda()
        reference_q,reference_k = rope(qkv[:,0].transpose(0,1)[None],qkv[:,1].transpose(0,1)[None],(height,width))
        reference_q,reference_k = reference_q[0].transpose(0,1),reference_k[0].transpose(0,1)
    program.launch(qkv,freq,q,k,v,scalars=[rows,width])
    for name,out,ref in [("q",q,reference_q),("k",k,reference_k)]:
        assert vendor.metrics(f"rotary_d{dim}_cls{class_token}_{height}x{width}_{name}",out,ref)<=1
    assert torch.equal(v,qkv[:,2])
    if class_token:
        assert torch.equal(q[0],qkv[0,0]) and torch.equal(k[0],qkv[0,1])
    # A second immutable frequency table proves theta is caller data, not an
    # accidentally hidden static constant in the compiled program.
    alternate = freq*2
    program.launch(qkv,alternate,q,k,v,scalars=[rows,width])
    assert not torch.equal(q,reference_q)


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
    assert vendor.metrics(f"conv_lower_c{channels}_{height}x{width}",out[:rows],ref,source="torch.unfold-layout-only")==0
    assert (out[rows:] == 42).all()
    # A small exact dyadic contraction verifies the official Conv2d boundary,
    # not merely a self-consistent lowering layout. Large tower GEMMs qualify
    # with the full encoder and its own checkpoint later.
    if channels == 16:
        from b12x.integration.cuteafd.media import compile_biased_projection_aot
        weight = ((torch.arange(32*channels*9,device="cuda")%5).float()/16).bfloat16().reshape(32,channels,3,3)
        bias = torch.linspace(-.05,.05,32,device="cuda").bfloat16()
        projected = torch.empty(rows,32,device="cuda",dtype=torch.bfloat16)
        projection = compile_biased_projection_aot(width=channels*9,outputs=32)
        projection.launch(out,weight,bias,projected,scalars=[rows])
        reference = torch.nn.functional.conv2d(x.permute(2,0,1)[None].float(),weight.float(),bias.float(),stride=2,padding=1)
        reference = reference[0].permute(1,2,0).reshape_as(projected).bfloat16()
        assert vendor.metrics(f"conv3s2_c{channels}_{height}x{width}",projected,reference,source="torch.conv2d-vendor-geometry")==0


@pytest.mark.parametrize("class_token", [False,True])
def test_merge_is_spatial_major_and_excludes_class_token(class_token):
    require_b12x()
    from b12x.integration.cuteafd.perception import compile_merge2_aot
    h,w,c = 6,10,1024
    x = (torch.arange((h*w+int(class_token))*c,device="cuda")%251).bfloat16().reshape(-1,c)
    out = torch.empty((h*w//4,c*4),device="cuda",dtype=torch.bfloat16)
    program = compile_merge2_aot(width=c,class_token=class_token)
    program.launch(x,out,scalars=[h,w])
    ref = vendor.merge(x,h,w,c,class_token)
    assert vendor.metrics(f"merge_cls{class_token}",out,ref)==0
    for invalid in [(h+1,w),(h,w+1),(0,w)]:
        with pytest.raises(ValueError,match="positive even"):
            program.launch(x,out,scalars=invalid)


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
