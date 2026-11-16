"""Check existing FP32 resize arithmetic against supported M3.1 Pillow bytes."""
import numpy as np
from PIL import Image
import torch
from ..conftest import require_b12x


def test_existing_resize_reuse_against_minimax_default_pillow_bytes():
    require_b12x()
    from b12x.integration.cuteafd.image_resize import (
        compile_resize_coefficients_aot,compile_resize_pass_aot,resize_max_taps)
    coeff=compile_resize_coefficients_aot(filter="bicubic",antialias=True)
    horizontal=compile_resize_pass_aot(vertical=False)
    vertical=compile_resize_pass_aot(vertical=True)
    exact_final=True;exact_each_pass=True
    for h,w,oh,ow in [(13,17,28,28),(432,640,448,644),(640,432,644,448),
                       (672,672,672,672),(1009,731,672,504)]:
        raw=np.arange(h*w*3,dtype=np.uint32).reshape(h,w,3)
        raw=((raw*37+11)%256).astype(np.uint8)
        expected=torch.from_numpy(np.array(Image.fromarray(raw).resize((ow,oh),Image.Resampling.BICUBIC))).cuda()
        x=torch.from_numpy(raw).cuda().float().contiguous()
        def axis(size,out):
            taps=resize_max_taps(size,out,filter="bicubic",antialias=True)
            ix=torch.empty(out,device="cuda",dtype=torch.int32);n=torch.empty_like(ix)
            weights=torch.empty(out,taps,device="cuda")
            coeff.launch(ix,n,weights,scalars=[size,out,taps])
            return ix,n,weights,taps
        ix,nx,wx,tx=axis(w,ow);iy,ny,wy,ty=axis(h,oh)
        mid=torch.empty(h,ow,3,device="cuda");out=torch.empty(oh,ow,3,device="cuda")
        horizontal.launch(x,ix,nx,wx,mid,scalars=[h,w,ow,tx])
        vertical.launch(mid,iy,ny,wy,out,scalars=[h,ow,oh,ty])
        final=out.round().clamp(0,255).to(torch.uint8)
        rounded_mid=mid.round().clamp(0,255).contiguous()
        vertical.launch(rounded_mid,iy,ny,wy,out,scalars=[h,ow,oh,ty])
        each=out.round().clamp(0,255).to(torch.uint8)
        ef=torch.equal(final,expected);ee=torch.equal(each,expected)
        print(f"MINIMAX_RESIZE_REUSE source={h}x{w} output={oh}x{ow} final_exact={ef} each_pass_exact={ee} "
              f"final_mismatch={int((final!=expected).sum())} each_mismatch={int((each!=expected).sum())} "
              f"final_max={int((final.int()-expected.int()).abs().max())} each_max={int((each.int()-expected.int()).abs().max())}",flush=True)
        exact_final &= ef;exact_each_pass &= ee
    assert exact_final or exact_each_pass,"Existing FP32 coefficients/passes do not reproduce default Pillow uint8 bytes"
