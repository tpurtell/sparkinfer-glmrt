"""All finite BF16 inputs pin the official tanh-GELU output boundary."""
import pytest
import torch
import torch.nn.functional as F
from ..conftest import require_b12x


def test_tanh_gelu_all_finite_bfloat16_patterns_and_frozen_reuse():
    require_b12x()
    from b12x.integration.cuteafd.vision_gelu import compile_tanh_gelu_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    bits=torch.arange(65536,dtype=torch.int32,device="cuda").to(torch.int16)
    x=bits.view(torch.bfloat16).reshape(16,4096)
    out=torch.empty_like(x)
    program=compile_tanh_gelu_aot(width=4096)
    with kernel_resolution_guard():program.launch(x,out,scalars=[16])
    expected=F.gelu(x,approximate="tanh")
    finite=torch.isfinite(x)
    assert torch.equal(out[finite].view(torch.int16),expected[finite].view(torch.int16)),int((out[finite].view(torch.int16)!=expected[finite].view(torch.int16)).sum())
    singleton=torch.empty(1,4096,device="cuda",dtype=torch.bfloat16)
    with kernel_resolution_guard():program.launch(x,singleton,scalars=[1])
    assert torch.equal(singleton.view(torch.int16),out[:1].view(torch.int16))


def test_invalid_tanh_gelu_width():
    from b12x.integration.cuteafd.vision_gelu import compile_tanh_gelu_aot
    with pytest.raises(ValueError):compile_tanh_gelu_aot(width=0)
