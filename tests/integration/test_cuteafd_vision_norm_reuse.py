"""Screen existing norm arithmetic against official BF16 tower boundaries."""
import pytest
import torch
import torch.nn.functional as F
from ..conftest import require_b12x


@pytest.mark.parametrize("width,layernorm,epsilon",[(1024,False,2**-23),(7168,False,1e-5),(1280,True,1e-5)])
def test_existing_affine_norm_matches_tower_reference_exact(width,layernorm,epsilon):
    require_b12x()
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    generator=torch.Generator().manual_seed(670)
    gamma=(torch.randn(width,generator=generator)*.2+1).bfloat16().cuda()
    bias=(torch.randn(width,generator=generator)*.1).bfloat16().cuda() if layernorm else None
    if width==7168:
        from b12x.integration.cuteafd.vision_rms import compile_vectorized_vision_rms_aot
        program=compile_vectorized_vision_rms_aot(width=width)
    elif layernorm:
        from b12x.integration.cuteafd.vision_layer_norm import compile_vectorized_vision_layer_norm_aot
        program=compile_vectorized_vision_layer_norm_aot(width=width)
    else:
        program=compile_affine_norm_aot(width=width,eps=None,subtract_mean=layernorm,bias=layernorm,
            input_dtype=torch.bfloat16,parameter_dtype=torch.bfloat16)
    for rows in (1,17,80,257):
        for magnitude in (1.,1e-4):
            x=(torch.randn(rows,width,generator=generator)*magnitude).bfloat16().cuda()
            expected=F.layer_norm(x,[width],gamma,bias,epsilon) if layernorm else F.rms_norm(x,[width],gamma,epsilon)
            out=torch.empty_like(x)
            with kernel_resolution_guard():
                if width==7168:program.launch(x,gamma,out,scalars=[rows,epsilon])
                elif layernorm:program.launch(x,gamma,bias,out,scalars=[rows,epsilon])
                else:program.launch(x,gamma,bias,None,None,None,out,scalars=[rows,epsilon])
            diff=(out.view(torch.int16)!=expected.view(torch.int16)).sum().item()
            print(f"VISION_NORM width={width} layernorm={layernorm} eps={epsilon} rows={rows} magnitude={magnitude} mismatched={diff} max_abs={(out.float()-expected.float()).abs().max().item()}",flush=True)
            assert torch.equal(out.view(torch.int16),expected.view(torch.int16))


def test_invalid_norm_width():
    from b12x.integration.cuteafd.pointwise import compile_affine_norm_aot
    with pytest.raises(ValueError):compile_affine_norm_aot(width=0)
