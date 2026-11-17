"""Exhaustive finite-BF16 screen against the installed ATen erf-GELU kernel."""
import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("width", [4096, 5120, 6144])
def test_exact_vision_gelu_all_finite_bf16(width):
    require_b12x()
    from b12x.integration.cuteafd.vision_exact_gelu import compile_exact_vision_gelu_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = patterns[torch.isfinite(patterns)]
    rows = (finite.numel() + width - 1) // width
    x = torch.zeros(rows * width, dtype=torch.bfloat16)
    x[:finite.numel()] = finite
    x = x.reshape(rows, width).cuda()
    expected = torch.nn.functional.gelu(x, approximate="none")
    output = torch.empty_like(x)
    program = compile_exact_vision_gelu_aot(width=width)
    with kernel_resolution_guard():
        program.launch(x, output, scalars=[rows])
        program.launch(x[:1], output[:1], scalars=[1])
    mismatch = output.view(torch.int16) != expected.view(torch.int16)
    where = mismatch.nonzero()
    details = [(int(x[r,c].view(torch.int16)), float(x[r,c]), int(output[r,c].view(torch.int16)),
                int(expected[r,c].view(torch.int16))) for r,c in where[:20].tolist()]
    print(f"EXACT_GELU torch={torch.__version__} width={width} mismatches={int(mismatch.sum())} details={details}", flush=True)
    assert not bool(mismatch.any())


def test_invalid_exact_vision_gelu_width():
    from b12x.integration.cuteafd.vision_exact_gelu import compile_exact_vision_gelu_aot
    with pytest.raises(ValueError):
        compile_exact_vision_gelu_aot(width=0)
