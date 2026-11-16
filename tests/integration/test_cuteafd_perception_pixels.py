"""Inkling native RGB8 content+sentinel patches against pinned vendor Triton."""
from __future__ import annotations
import pytest
import torch
from ..conftest import require_b12x
from . import perception_vendor as vendor


@pytest.mark.parametrize("h,w,content",[(40,40,(0,0,40,40)),(80,120,(0,0,113,77)),(80,80,(3,5,71,69))])
def test_rgb_content_patches_match_vendor(h,w,content):
    require_b12x()
    from b12x.integration.cuteafd.perception_pixels import compile_pixel_patches_aot
    rgb=(torch.arange(h*w*3,device="cuda")%256).to(torch.uint8).reshape(h,w,3)
    x,y,cw,ch=content
    mean_values=(.48145466,.4578275,.40821073)
    processor_values=(.26862954,.2613026,.2757771)
    std_values=(.26862954,.26130258,.27577711)
    mean=torch.tensor(mean_values,device="cuda");processor_std=torch.tensor(processor_values,device="cuda");std=torch.tensor(std_values,device="cuda")
    raw=torch.full((h,w,3),-1,device="cuda",dtype=torch.bfloat16)
    raw[y:y+ch,x:x+cw]=rgb[y:y+ch,x:x+cw].bfloat16()
    # Vendor takes one packed40x40 canvas cell per LM row, then gathers native20.
    packed=raw.reshape(h//40,40,w//40,40,3).permute(0,2,1,3,4).reshape(-1,40,40,3).contiguous()
    offsets=[]
    for iy in range(h//20):
        for ix in range(w//20):
            cell=(iy//2)*(w//40)+ix//2
            offsets.append(cell*40*40*3+(iy%2)*20*40*3+(ix%2)*20*3)
    offsets=torch.tensor(offsets,device="cuda",dtype=torch.int32)
    expected=vendor.inkling().gather_patches(packed,offsets,20,mean_values,processor_values,std_values)
    out=torch.empty_like(expected)
    program=compile_pixel_patches_aot(patch=20)
    program.launch(rgb,mean,processor_std,std,out,scalars=[h,w,x,y,cw,ch])
    assert vendor.metrics(f"rgb_sentinel_{h}x{w}_{content}",out,expected)==0
    from b12x._lib.runtime_control import kernel_resolution_guard
    with kernel_resolution_guard("PE RGB content patches"):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):program.launch(rgb,mean,processor_std,std,out,scalars=[h,w,x,y,cw,ch])
        graph.replay();torch.cuda.synchronize();assert torch.equal(out,expected)
    for invalid in [[h,w,-1,y,cw,ch],[h,w,x,y,w+1,ch],[h-1,w,x,y,cw,ch]]:
        with pytest.raises(ValueError):program.launch(rgb,mean,processor_std,std,out,scalars=invalid)
