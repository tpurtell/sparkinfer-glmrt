"""FP32 sin/cos match Torch cos/sin or complex-polar at actual image grids."""
import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("height,width",[(2,2),(2,48),(30,46),(48,48),(132,506)])
@pytest.mark.parametrize("adjacent",[False,True])
def test_source_matched_vision_rotary_tables(height,width,adjacent):
    require_b12x()
    from b12x.integration.cuteafd.vision_rotary_tables import compile_vision_rotary_tables_aot
    if not adjacent and max(height,width)>48:pytest.skip("MiniMax supported default axis cap48")
    axes=(0,64,64) if adjacent else (26,26,26)
    count=32 if adjacent else 13
    inv=1/(10000**(torch.arange(count,device="cuda",dtype=torch.float32)*2/(64 if adjacent else 26)))
    t=inv if not adjacent else torch.empty(1,device="cuda")
    h=torch.arange(height,device="cuda")[:,None].expand(height,width)
    w=torch.arange(width,device="cuda")[None,:].expand(height,width)
    if not adjacent:
        h=h.reshape(height//2,2,width//2,2).permute(0,2,1,3).reshape(-1)
        w=w.reshape(height//2,2,width//2,2).permute(0,2,1,3).reshape(-1)
        phase=torch.cat((torch.zeros(height*width,13,device="cuda"),h[:,None].float()*inv,w[:,None].float()*inv),-1)
        expected_cos,expected_sin=phase.cos(),phase.sin()
    else:
        phase=torch.stack((w.reshape(-1)[:,None].float()*inv,h.reshape(-1)[:,None].float()*inv),-1).reshape(height*width,-1)
        expected=torch.polar(torch.ones_like(phase),phase)
        expected_cos,expected_sin=expected.real,expected.imag
    cos=torch.empty(height*width,sum(axes)//2,device="cuda");sin=torch.empty_like(cos)
    compile_vision_rotary_tables_aot(axes=axes,adjacent=adjacent,group_major=not adjacent).launch(t,inv,inv,cos,sin,scalars=[height,width])
    assert torch.equal(cos,expected_cos),(adjacent,height,width,"cos",(cos-expected_cos).abs().max().item())
    assert torch.equal(sin,expected_sin),(adjacent,height,width,"sin",(sin-expected_sin).abs().max().item())


@pytest.mark.parametrize("axes,adjacent,group",[((0,64,64),False,False),((26,26,26),False,False),((0,32,32),True,False)])
def test_invalid_table_recipe(axes,adjacent,group):
    from b12x.integration.cuteafd.vision_rotary_tables import compile_vision_rotary_tables_aot
    with pytest.raises(ValueError):compile_vision_rotary_tables_aot(axes=axes,adjacent=adjacent,group_major=group)
