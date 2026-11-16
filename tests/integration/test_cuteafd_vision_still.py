"""Exact official still patchification/grouping; normalization table is an operand."""
import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("height,width",[(28,28),(28,56),(56,84),(672,672)])
@pytest.mark.parametrize("temporal,group_major",[(1,False),(2,True)])
def test_still_rgb_patch_order_and_duplicate_frame_exact(height,width,temporal,group_major):
    require_b12x()
    from b12x.integration.cuteafd.vision_still import compile_still_patches_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    rgb=(torch.arange(height*width*3,device="cuda",dtype=torch.int32)%256).to(torch.uint8).reshape(height,width,3)
    lut=((torch.arange(256,device="cuda",dtype=torch.float32)/255)-.5)*2
    lut=lut.repeat(3,1).contiguous()
    normalized=lut[torch.arange(3,device="cuda")[None,None,:],rgb.long()]
    raster=normalized.permute(2,0,1)[None].expand(temporal,-1,-1,-1)
    gh,gw=height//14,width//14
    if group_major:
        expected=raster.reshape(1,temporal,3,gh//2,2,14,gw//2,2,14).permute(0,3,6,4,7,2,1,5,8).reshape(gh*gw,3*temporal*14*14).bfloat16()
    else:
        expected=normalized.reshape(gh,14,gw,14,3).permute(0,2,4,1,3).reshape(gh*gw,3*14*14).bfloat16()
    out=torch.empty_like(expected)
    program=compile_still_patches_aot(temporal=temporal,group_major=group_major)
    with kernel_resolution_guard():program.launch(rgb,lut,out,scalars=[height,width])
    assert torch.equal(out,expected)


@pytest.mark.parametrize("group_major",[False,True])
def test_spatial_group_matches_official_concat_order(group_major):
    require_b12x()
    from b12x.integration.cuteafd.vision_still import compile_spatial_group_aot
    height,width,channels=6,10,1024
    raster=(torch.arange(height*width*channels,device="cuda")%251).reshape(height,width,channels).bfloat16()
    expected=raster.reshape(height//2,2,width//2,2,channels).permute(0,2,1,3,4).reshape(-1,4*channels)
    x=expected.reshape(height*width,channels).contiguous() if group_major else raster.reshape(-1,channels)
    out=torch.empty(expected.shape,device="cuda",dtype=torch.bfloat16)
    compile_spatial_group_aot(channels=channels,group_major=group_major).launch(x,out,scalars=[height,width])
    assert torch.equal(out,expected)


@pytest.mark.parametrize("kwargs",[dict(patch=16),dict(temporal=3),dict(temporal=1,group_major=True)])
def test_invalid_still_layout(kwargs):
    from b12x.integration.cuteafd.vision_still import compile_still_patches_aot
    with pytest.raises(ValueError):compile_still_patches_aot(**kwargs)
