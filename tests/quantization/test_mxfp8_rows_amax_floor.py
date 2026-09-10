"""Explicit K32 activation floor preserves V4.1 tiny/zero group semantics."""
import pytest
import torch
from b12x._lib.quant.mxfp8_rows import _get_compiled_mxfp8_rows_quant
from tests.conftest import require_b12x


@pytest.mark.parametrize("subgroup", [0, 4, 8])
@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("floor", [0.0, 1e-4])
def test_amax_floor_values_scales_and_replay(subgroup, block, floor):
    require_b12x()
    rows, k = 16, 256
    factors = torch.tensor([0, 1e-7, 1e-5, 1e-4, 1, 4, 128, .002], device="cuda")
    x = (torch.linspace(-1, 1, 32, device="cuda")[None, None, :] *
         factors[None, :, None]).expand(rows, -1, -1).reshape(rows, k).bfloat16().contiguous()
    values = torch.empty_like(x, dtype=torch.uint8)
    scales = torch.empty((rows,k//32), device="cuda", dtype=torch.uint8)
    mma = torch.empty(512*(k//128), device="cuda", dtype=torch.uint8)
    run = _get_compiled_mxfp8_rows_quant(k, torch.bfloat16, subgroup, 256, block, "linear", floor)

    def check():
        groups = x.float().reshape(rows,k//block,block)
        maximum = groups.abs().amax(-1).clamp_min(max(floor,1e-4) if block == 128 else floor)
        expected_scales = torch.exp2(torch.ceil(torch.log2(maximum/448)))
        expected_scales = torch.where(maximum == 0, 1.0, expected_scales)
        expected = (groups/expected_scales[:,:,None]).to(torch.float8_e4m3fn).reshape(rows,k)
        assert torch.equal(values, expected.view(torch.uint8))
        actual_scales = scales.view(torch.float8_e8m0fnu).float()
        assert torch.equal(actual_scales,expected_scales.repeat_interleave(block//32,-1))
        # Also inspect all live hardware MMA scale slots.
        r = torch.arange(rows,device="cuda")[:,None]
        g = torch.arange(k//32,device="cuda")[None,:]
        offsets = (r%32)*16 + (r//32%4)*4 + (r//128)*(k//128)*512 + g%4 + (g//4)*512
        assert torch.equal(mma[offsets],scales)

    run(x,values,scales,mma)
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run(x,values,scales,mma)
    x.mul_(-.25)
    graph.replay()
    check()
    torch.cuda.synchronize()
    graph.reset()


@pytest.mark.parametrize("floor", [-1.0, float("nan"), float("inf"), 2.0])
def test_invalid_floor(floor):
    with pytest.raises(ValueError,match="amax_floor"):
        _get_compiled_mxfp8_rows_quant(256, torch.bfloat16, 4, 256, 32, "linear", floor)
