"""#1915 convolution oracle, packed boundaries, short-history shifts and replay.

The frozen oracle model.py SHA256 is
1e0b4f50d6a6c807f60c205fcc62209d853323a2e9f50a36045a26b8bd488cf0.
FP32 output must match to 2e-6 absolute / 2e-6 relative; BF16 must equal the
rounded oracle exactly. History is input state, so always byte-exact.
"""
from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x


def short_conv(x, weight, dynamic=None):
    """Verbatim #1915 short_conv, with the same oldest/lag-first conventions."""
    if dynamic is None:
        taps = weight.reshape(x.shape[-1], -1)
        width = taps.shape[-1]
    else:
        width = weight.shape[0]
        if (dynamic.ndim != 3 or dynamic.shape[0] != x.shape[0]
                or dynamic.shape[2] != width or x.shape[1] % dynamic.shape[1]):
            raise ValueError("invalid dynamic convolution geometry")
    padded = F.pad(x.float(), (0, 0, width - 1, 0))
    result = torch.zeros_like(x, dtype=torch.float32)
    for lag in range(width):
        value = padded[width - 1 - lag:width - 1 - lag + len(x)]
        if dynamic is None:
            tap = taps[:, width - 1 - lag].float()
        else:
            tap = weight[lag].float() + dynamic[:, :, lag].float().repeat_interleave(
                x.shape[1] // dynamic.shape[1], dim=1)
        result += value * tap
    return (result + x.float()) * (width * (2 if dynamic is not None else 1) + 1) ** -.5


@functools.lru_cache(None)
def program(width, dynamic, dtype):
    from b12x.integration.cuteafd.inkling2_convolution import compile_inkling2_convolution_aot
    return compile_inkling2_convolution_aot(width=width, heads=8 if dynamic else 1,
                                           dynamic=dynamic, dtype=dtype, max_rows=4096)


@pytest.mark.parametrize("dynamic,width", [(False, 4608), (False, 1024), (True, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunks", [(37,), (1, 2, 1, 7, 3), (2048, 1, 3)])
def test_reference_and_chunk_state(dynamic, width, dtype, chunks):
    require_b12x()
    p = program(width, dynamic, dtype)
    gen = torch.Generator(device="cuda").manual_seed(1930)
    n = sum(chunks)
    x = torch.randn(n, width, generator=gen, device="cuda").to(dtype)
    weight = torch.randn((4, width) if dynamic else (width, 4), generator=gen,
                         device="cuda").bfloat16()
    coefficients = torch.randn(n, 8, 4, generator=gen, device="cuda").to(dtype) if dynamic else None
    reference = short_conv(x, weight, coefficients).to(dtype)
    history = torch.zeros(1, 3, width, dtype=dtype, device="cuda")
    start = 0
    for count in chunks:
        out = torch.full((count + 1, width), float("nan"), device="cuda", dtype=dtype)
        first = torch.zeros(count, dtype=torch.int32, device="cuda")
        slots = torch.zeros(count, dtype=torch.int64, device="cuda")
        p.launch(x[start:start + count], weight, coefficients[start:start + count] if dynamic else None,
                 history, first, slots, out, scalars=(count,))
        expected = reference[start:start + count]
        torch.testing.assert_close(out[:count], expected, rtol=0 if dtype == torch.bfloat16 else 2e-6,
                                   atol=0 if dtype == torch.bfloat16 else 2e-6)
        assert torch.isnan(out[count]).all()
        start += count
        expected_history = F.pad(x[:start].flip(0)[:3], (0, 0, 0, max(0, 3 - start)))
        assert torch.equal(history[0], expected_history)


@pytest.mark.parametrize("dynamic", [False, True])
def test_packed_replay_and_parameter_mutation(dynamic):
    require_b12x()
    width, count = 1024, 11
    p = program(width, dynamic, torch.float32)
    x = torch.randn(count, width, device="cuda")
    weights = torch.randn((4, width) if dynamic else (width, 4), device="cuda").bfloat16()
    coefficients = torch.randn(count, 8, 4, device="cuda") if dynamic else None
    # Nonsequential slots detect indexing by row rather than sequence.
    history = torch.zeros(9, 3, width, device="cuda")
    slots = torch.tensor([7] * 2 + [3] * 9, dtype=torch.int64, device="cuda")
    first = torch.tensor([0] * 2 + [2] * 9, dtype=torch.int32, device="cuda")
    out = torch.empty_like(x)
    def launch():
        history.zero_()
        p.launch(x, weights, coefficients, history, first, slots, out, scalars=(count,))
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    weights.add_(0.25)
    if dynamic:
        coefficients.mul_(1.5)
    x.mul_(0.5)
    graph.replay()
    expected = torch.cat([short_conv(x[a:b], weights, coefficients[a:b] if dynamic else None)
                          for a, b in [(0, 2), (2, 11)]])
    torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-6)
    assert torch.equal(history[7], F.pad(x[:2].flip(0), (0, 0, 0, 1)))
    assert torch.equal(history[3], x[-3:].flip(0))
    assert torch.count_nonzero(history[[0, 1, 2, 4, 5, 6, 8]]) == 0
