"""Unclamped FP32 owner resize against the installed Torch tensor operator."""
from __future__ import annotations
import math
import os
from pathlib import Path
import pytest
import torch
from ..conftest import require_b12x


def _images():
    from PIL import Image
    import numpy as np
    root=Path(os.environ["CUTEAFD_PE_REAL_IMAGES"])
    files=sorted(root.glob("*.png"))+sorted(root.glob("*.jpg"))
    if not files:
        raise ValueError("real image fixture directory must contain PNG/JPEG images")
    return [torch.from_numpy(np.array(Image.open(path).convert("RGB"))).cuda() for path in files]


def _coeff(program,size,out):
    taps=2*math.ceil(2*max(1,size/out))+1
    indices=torch.empty(out,device="cuda",dtype=torch.int32)
    sizes=torch.empty_like(indices)
    weights=torch.empty(out,taps,device="cuda")
    program.launch(indices,sizes,weights,scalars=[size,out,taps])
    return indices,sizes,weights,taps


def test_rgb_normalize_all_unsigned_byte_values():
    require_b12x()
    from b12x.integration.cuteafd.image_resize import compile_rgb_normalize_aot
    rgb=torch.arange(256,device="cuda",dtype=torch.int32).to(torch.uint8)[:,None,None].expand(256,1,3).contiguous()
    mean=torch.tensor([.48145466,.4578275,.40821073],device="cuda")
    std=torch.tensor([.26862954,.26130258,.27577711],device="cuda")
    out=torch.empty(256,1,3,device="cuda")
    compile_rgb_normalize_aot().launch(rgb,mean,std,out,scalars=[256,1])
    expected=(rgb.float()/255-mean)/std
    print(f"RGB_LUT max_abs={float((out-expected).abs().max()):.9g}",flush=True)
    assert torch.equal(out,expected)


@pytest.mark.parametrize("out_size",[728,504])
def test_real_image_resize_matches_tensor_antialias(out_size):
    require_b12x()
    import torchvision
    from b12x.integration.cuteafd.image_resize import (compile_rgb_normalize_aot,
        compile_resize_coefficients_aot,compile_resize_pass_aot)
    mean=torch.tensor([.48145466,.4578275,.40821073],device="cuda")
    std=torch.tensor([.26862954,.26130258,.27577711],device="cuda")
    normalize=compile_rgb_normalize_aot()
    coeff=compile_resize_coefficients_aot(filter="bicubic",antialias=True)
    horizontal=compile_resize_pass_aot(vertical=False)
    vertical=compile_resize_pass_aot(vertical=True)
    print(f"RESIZE_RUNTIME torch={torch.__version__} torchvision={torchvision.__version__}",flush=True)
    for rgb in _images():
        h,w=rgb.shape[:2]
        normalized=torch.empty(h,w,3,device="cuda")
        normalize.launch(rgb,mean,std,normalized,scalars=[h,w])
        reference_input=(rgb.permute(2,0,1).float()/255-mean[:,None,None])/std[:,None,None]
        assert torch.equal(normalized.permute(2,0,1),reference_input)
        ix,nx,wx,tx=_coeff(coeff,w,out_size)
        iy,ny,wy,ty=_coeff(coeff,h,out_size)
        intermediate=torch.empty(h,out_size,3,device="cuda")
        actual=torch.empty(out_size,out_size,3,device="cuda")
        horizontal.launch(normalized,ix,nx,wx,intermediate,scalars=[h,w,out_size,tx])
        vertical.launch(intermediate,iy,ny,wy,actual,scalars=[h,out_size,out_size,ty])
        reference=torchvision.transforms.functional.resize(reference_input,[out_size,out_size],
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,antialias=True).permute(1,2,0)
        error=(actual-reference).abs()
        equal=int((actual.view(torch.int32)==reference.contiguous().view(torch.int32)).sum())
        print(f"RESIZE_ORACLE input={h}x{w} output={out_size} max_abs={float(error.max()):.9g} bit_equal={equal/actual.numel():.12f} min={float(actual.min()):.9g} max={float(actual.max()):.9g}",flush=True)
        torch.testing.assert_close(actual,reference,atol=3e-6,rtol=3e-6)
        # All live dimensions share compiled code even during graph replay.
        from b12x._lib.runtime_control import kernel_resolution_guard
        with kernel_resolution_guard("owner image tensor resize"):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                normalize.launch(rgb,mean,std,normalized,scalars=[h,w])
                horizontal.launch(normalized,ix,nx,wx,intermediate,scalars=[h,w,out_size,tx])
                vertical.launch(intermediate,iy,ny,wy,actual,scalars=[h,out_size,out_size,ty])
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(actual,reference,atol=3e-6,rtol=3e-6)
