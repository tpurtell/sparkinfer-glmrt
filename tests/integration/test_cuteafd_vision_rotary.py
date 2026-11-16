"""Official MiniMax split-half and MoonViT complex-pair arithmetic boundaries."""
from __future__ import annotations

import pytest
import torch
from ..conftest import require_b12x


def _reference(x, cos, sin, rotated, adjacent):
    if adjacent:
        pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        frequencies = torch.complex(cos, sin)[:, None]
        return torch.view_as_real(pairs * frequencies).flatten(-2).to(x.dtype)
    leading = x[..., :rotated].float()
    half = rotated // 2
    rotate = torch.cat((-leading[..., half:], leading[..., :half]), -1)
    c, s = cos.repeat(1,2)[:,None], sin.repeat(1,2)[:,None]
    return torch.cat(((leading * c + rotate * s).to(x.dtype), x[...,rotated:]), -1)


@pytest.mark.parametrize("rows", (1,2,17,80,257))
@pytest.mark.parametrize("heads,dim,rotated,adjacent", [(16,80,78,False),(12,128,128,True)])
def test_vendor_qkv_rotary_and_passthrough_exact(rows,heads,dim,rotated,adjacent):
    require_b12x()
    from b12x.integration.cuteafd.vision_rotary import compile_vision_qkv_rotary_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    generator = torch.Generator().manual_seed(998)
    source = torch.randn(rows,3,heads,dim,generator=generator).bfloat16().cuda()
    phases = torch.randn(rows,rotated//2,generator=generator).cuda()
    cos, sin = phases.cos(), phases.sin()
    outputs = [torch.empty(rows,heads,dim,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    program = compile_vision_qkv_rotary_aot(heads=heads,dim=dim,rotated=rotated,adjacent=adjacent)
    program.launch(source,cos,sin,*outputs,scalars=[rows])
    assert torch.equal(outputs[0],_reference(source[:,0],cos,sin,rotated,adjacent))
    assert torch.equal(outputs[1],_reference(source[:,1],cos,sin,rotated,adjacent))
    assert torch.equal(outputs[2],source[:,2])
    if rotated<dim:
        assert torch.equal(outputs[0][...,rotated:],source[:,0,...,rotated:])
        assert torch.equal(outputs[1][...,rotated:],source[:,1,...,rotated:])
    singleton = [torch.empty(1,heads,dim,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    with kernel_resolution_guard():
        program.launch(source,cos,sin,*singleton,scalars=[1])
    assert all(torch.equal(out,expected[:1]) for out,expected in zip(singleton,outputs))


@pytest.mark.parametrize("kwargs", [dict(heads=0,dim=80,rotated=78),
    dict(heads=16,dim=80,rotated=81),dict(heads=16,dim=80,rotated=77),
    dict(heads=12,dim=128,rotated=126,adjacent=True)])
def test_invalid_static_rotary_geometry(kwargs):
    from b12x.integration.cuteafd.vision_rotary import compile_vision_qkv_rotary_aot
    with pytest.raises(ValueError):
        compile_vision_qkv_rotary_aot(**kwargs)
