"""Shared expert partial reply codecs and Muse canonical TP combine."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from ..conftest import require_b12x
from .test_cuteafd_muse import _canonical_norm

ROWS = (1, 8, 16, 64, 65, 80, 2752, 4096)
FORMATS = ("bf16", "nvfp4", "fp8")


def _encode(x, format):
    x = x.cpu().float().contiguous()
    width = x.shape[-1]
    if x.numel() > 2**21 and format != "bf16":
        records = x.reshape(-1, width)
        chunks = [_encode(chunk, format) for chunk in records.split(256)]
        return torch.cat(chunks).reshape(*x.shape[:-1], chunks[0].shape[-1])
    if format == "bf16":
        return x.bfloat16()
    block = 16 if format == "nvfp4" else 32
    groups = x.reshape(*x.shape[:-1], width // block, block)
    amax = groups.abs().amax(-1)
    if format == "nvfp4":
        scale = (amax / 6).clamp(max=448).to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9)
        values = groups / scale[..., None]
        grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        dist = (values.abs().clamp_max(6)[..., None] - grid).abs()
        # An exact tie picks the even code, independently of grid index order.
        best = torch.zeros_like(values, dtype=torch.int64)
        minimum = dist[..., 0]
        for i in range(1, 8):
            choose = (dist[..., i] < minimum) | ((dist[..., i] == minimum) & (i % 2 == 0) & (best % 2 == 1))
            best = torch.where(choose, i, best)
            minimum = torch.minimum(minimum, dist[..., i])
        code = best.byte() | (torch.signbit(values).byte() << 3)
        code = code.reshape(*x.shape[:-1], width)
        payload = code[..., ::2] | (code[..., 1::2] << 4)
        scales = torch.where(amax == 0, 0., scale).to(torch.float8_e4m3fn).view(torch.uint8)
        payload = torch.where((amax == 0).repeat_interleave(block // 2, -1), 0, payload)
    else:
        scale = torch.exp2(torch.ceil(torch.log2(amax.double() / 448)).clamp(-127, 127)).float()
        payload = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).reshape_as(x)
        payload = torch.where((amax == 0).repeat_interleave(block, -1), 0, payload)
        scales = (torch.log2(scale.double()) + 127).byte()
    return torch.cat((payload, scales), -1)


def _decode(data, format, width):
    if format == "bf16":
        return data.float()
    if format == "nvfp4":
        packed, scales = data[..., :width // 2], data[..., width // 2:]
        code = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
        grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        values = grid[(code & 7).long()]
        values = torch.where((code & 8) != 0, -values, values)
        return (values.reshape(*values.shape[:-1], width // 16, 16) *
                scales.view(torch.float8_e4m3fn).float()[..., None]).reshape_as(values)
    values = data[..., :width].view(torch.float8_e4m3fn).float()
    scales = torch.exp2(data[..., width:].float() - 127)
    return (values.reshape(*values.shape[:-1], width // 32, 32) * scales[..., None]).reshape_as(values)


def _combine(partials, weights, norm):
    ranks, rows, topk, width = partials.shape
    out = torch.zeros(rows, width)
    for route in range(topk):
        total = torch.zeros(rows, width)
        for rank in range(ranks):
            total = total + partials[rank, :, route]
        scaled = (total.bfloat16().float() * (1 + norm.float()).bfloat16().float()).bfloat16()
        value = _canonical_norm(scaled, eps=1e-8).bfloat16().float()
        out = (out + (value * weights[:, route, None]).bfloat16().float()).bfloat16().float()
    return out.bfloat16()


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("rows", ROWS)
def test_partial_codec(rows, format):
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.expert_partial import compile_expert_partial_aot
    # Codec qualification takes the exact F32 partial as input; the worker
    # unrotation is qualified separately against the existing epilogue.
    gen = torch.Generator().manual_seed(51)
    x = torch.randn(rows, 8, 128, generator=gen)
    x[0, 0] = -0.
    x[0, 1, :16] = torch.linspace(-1e-5, 1e-5, 16)
    x[0, 2, :16] = torch.tensor([0., -.0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6., -6., -.25, -.75, -1.25, -3.5, -5.])
    if format == "fp8":
        for route, value in enumerate((448., 449., 1.75, 2.0**-140, 2.0**-126, 2.0**120), 2):
            x[0, route] = value
    expect = _encode(x, format)
    program = compile_expert_partial_aot(width=128, topk=8, format=format, rotated=False)
    data = x.cuda()
    rowmap = torch.arange(rows, dtype=torch.int32, device="cuda").view(torch.uint32)
    empty = torch.empty(1, device="cuda")
    out = torch.empty_like(expect, device="cuda")
    program.launch(data, empty, empty, empty, rowmap, out, scalars=[rows, rows, 0, 0])
    assert torch.equal(out.cpu().view(torch.uint8), expect.view(torch.uint8))
    small = torch.empty_like(out[:1])
    for row in sorted({0, rows // 2, rows - 1}):
        program.launch(data, empty, empty, empty, rowmap[row:row + 1], small, scalars=[1, rows, 0, 0])
        assert torch.equal(small, out[row:row + 1])


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("rows", ROWS)
def test_tp_combine(rows, format):
    require_b12x()
    from b12x.integration.cuteafd.muse_tp import compile_tp_combine_aot
    gen = torch.Generator().manual_seed(52)
    ranks, topk, width = 2, 8, 4096
    partials = torch.randn(ranks, rows, topk, width, generator=gen)
    encoded = _encode(partials, format)
    dequantized = _decode(encoded, format, width)
    weights = torch.rand(rows, topk, generator=gen)
    weights /= weights.sum(-1, keepdim=True)
    norm = (torch.randn(width, generator=gen) * .1).bfloat16()
    expected = _combine(dequantized, weights, norm)
    program = compile_tp_combine_aot(format=format)
    data, w, n = encoded.cuda(), weights.cuda(), norm.cuda()
    rowmap = torch.arange(rows, dtype=torch.int32, device="cuda").view(torch.uint32)
    out = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
    program.launch(data, rowmap, w, n, out, scalars=[ranks, rows, rows])
    assert torch.equal(out.cpu(), expected), (out.cpu().float() - expected.float()).abs().max()
    small = torch.empty_like(out[:1])
    for row in sorted({0, rows // 2, rows - 1}):
        compact = data[:, row:row + 1].contiguous()
        zero = torch.zeros(1, dtype=torch.int32, device="cuda").view(torch.uint32)
        program.launch(compact, zero, w[row:row + 1], n, small, scalars=[ranks, 1, 1])
        assert torch.equal(small[0], out[row])


def _unrotate(x):
    work = x.float().reshape(*x.shape[:-1], -1, 128)
    for stage in range(7):
        pairs = work.reshape(*work.shape[:-1], -1, 2, 1 << stage)
        lo, hi = pairs[..., 0, :].clone(), pairs[..., 1, :].clone()
        work = torch.stack((lo + hi, lo - hi), -2).reshape_as(work)
    return (work * np.float32(0.088388347648)).reshape_as(x)


@pytest.mark.parametrize("format", FORMATS)
def test_rotated_partial_and_mixed_maps(format):
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.expert_partial import compile_expert_partial_aot
    gen = torch.Generator().manual_seed(53)
    rows, topk, width, experts = 145, 8, 4096, 16
    x = torch.randn(rows, topk, width, generator=gen).half()
    svh = (torch.randn(experts, width, generator=gen) * .2 + 1).half()
    ids = torch.randint(experts, (rows, topk), generator=gen, dtype=torch.int32)
    expert_map = torch.randperm(experts, generator=gen).int()
    ids[0, 0], ids[0, 1] = -1, experts
    expert_map[2] = -1
    local = expert_map[ids.clamp(0, experts - 1).long()].long()
    valid = (ids >= 0) & (ids < experts) & (local >= 0)
    partial = _unrotate(x) * svh[local.clamp_min(0)].float()
    partial = torch.where(valid[..., None], partial, 0.)
    expect = _encode(partial, format)
    program = compile_expert_partial_aot(width=width, topk=topk, format=format)
    device_args = [v.cuda() for v in (x, svh, ids, expert_map)]
    all_rows = torch.arange(rows, dtype=torch.int32, device="cuda").view(torch.uint32)
    full = torch.empty_like(expect, device="cuda")
    program.launch(*device_args, all_rows, full, scalars=[rows, rows, experts, experts])
    assert torch.equal(full.cpu().view(torch.uint8), expect.view(torch.uint8))
    from b12x._lib.runtime_control import kernel_resolution_guard
    with kernel_resolution_guard("Expert partial epilogue is precompiled"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program.launch(*device_args, all_rows, full, scalars=[rows, rows, experts, experts])
        full.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(full.cpu().view(torch.uint8), expect.view(torch.uint8))
    # Request segments 1/64/80 remain exact in discontiguous mixed regions.
    for indices in (torch.cat((torch.arange(65), torch.arange(130, 145))), torch.arange(65, 130)):
        rowmap = indices.int().cuda().view(torch.uint32)
        region = torch.empty_like(full[:len(indices)])
        program.launch(*device_args, rowmap, region, scalars=[len(indices), rows, experts, experts])
        assert torch.equal(region, full[indices.cuda()])


@pytest.mark.parametrize("ranks", (1, 3, 6, 8))
def test_combine_format_neutrality_and_wave_mix(ranks):
    require_b12x()
    from b12x.integration.cuteafd.muse_tp import compile_tp_combine_aot
    gen = torch.Generator().manual_seed(54)
    rows, topk, width = 81, 8, 4096
    # These dyadic values are exactly representable in every reply format.
    grid = torch.tensor([-6., -4., -3., -2., -1., 0., 1., 2., 3., 4., 6.])
    partials = grid[torch.randint(len(grid), (ranks, rows, topk, width), generator=gen)]
    partials[..., ::16] = 6.
    weights = torch.rand(rows, topk, generator=gen)
    norm = (torch.randn(width, generator=gen) * .1).bfloat16().cuda()
    reference = None
    for format in FORMATS:
        data = _encode(partials, format)
        assert torch.equal(_decode(data, format, width), partials)
        data = data.cuda()
        out = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        rowmap = torch.arange(rows, dtype=torch.int32, device="cuda").view(torch.uint32)
        program = compile_tp_combine_aot(format=format)
        program.launch(data, rowmap, weights.cuda(), norm, out, scalars=[ranks, rows, rows])
        if reference is None:
            reference = out.clone()
        assert torch.equal(out, reference)
        # Every format replays the same precompiled program without resolution.
        from b12x._lib.runtime_control import kernel_resolution_guard
        w = weights.cuda()
        with kernel_resolution_guard("TP replies use precompiled geometry only"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program.launch(data, rowmap, w, norm, out, scalars=[ranks, rows, rows])
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, reference)
            compact = data[:, :1].contiguous()
            one = torch.empty_like(out[:1])
            program.launch(compact, rowmap[:1], w, norm, one, scalars=[ranks, 1, rows])
            assert torch.equal(one[0], reference[0])
        # Disjoint mixed regions write their original wave rows, no repacking output.
        out.fill_(float("nan"))
        for indices in (torch.cat((torch.arange(1), torch.arange(17, rows))), torch.arange(1, 17)):
            compact = data[:, indices.cuda()].contiguous()
            mapped = indices.int().cuda().view(torch.uint32)
            program.launch(compact, mapped, weights.cuda(), norm, out, scalars=[ranks, len(indices), rows])
        assert torch.equal(out, reference)
