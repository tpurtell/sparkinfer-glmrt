"""Inkling 2 Mini looped cross-pass residual (4-query) and controller boundary oracles.

Covers:
* ``compile_inkling2_residual_aot(hidden=2304, queries=4, max_candidates=2)``
* ``compile_inkling2_loop_pair_norm_aot`` (MLP, local broadcast-dest attn, global attn)
* ``compile_inkling2_loop_blend_norm_aot`` (MLP, local broadcast-dest/weight attn, global attn)
* CUDA graph replay with mutated inputs (proving zero stale captures)
* End-to-end controller delta between engine BF16-rank GEMM and F32-rank oracle.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from .test_cuteafd_inkling2_residual import reference as residual_reference

HIDDEN = 2304
EPS = 1e-6


def _rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = EPS) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("queries", [1, 3, 4])
@pytest.mark.parametrize("has_addend", [False, True])
def test_inkling2_mini_residual_queries(queries, has_addend):
    from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot

    max_cand = 2 if queries == 4 else 5
    p = compile_inkling2_residual_aot(
        hidden=HIDDEN, queries=queries, max_candidates=max_cand, addend=has_addend, eps=EPS,
    )
    gen = torch.Generator(device="cuda").manual_seed(100 + queries * 10 + int(has_addend))
    for rows in (1, 17, 64):
        cap = rows + 3
        arena = torch.randn((max_cand, cap, HIDDEN), generator=gen, device="cuda")
        arena *= torch.linspace(0.05, 15.0, max_cand, device="cuda")[:, None, None]
        arena[:, rows:] = float("nan")
        query = torch.randn((queries, HIDDEN), generator=gen, device="cuda") * 2.5
        addend = torch.randn((rows, HIDDEN), generator=gen, device="cuda") if has_addend else None
        orig = arena[:, :rows].clone()
        out = torch.empty((queries, rows, HIDDEN), device="cuda")
        for cand in range(1, max_cand + 1):
            p.launch(arena, addend, query, out, scalars=(rows, cap, cand))
            want = residual_reference(arena[:cand, :rows], query, addend)
            torch.testing.assert_close(out, want, atol=2e-5, rtol=2e-5)
            assert torch.equal(arena[:, :rows], orig)
            assert torch.isnan(arena[:, rows:]).all()


@pytest.mark.parametrize("slots,broadcast_dest", [(1, False), (3, True), (3, False)])
def test_inkling2_mini_loop_pair_norm_oracle_and_graph(slots, broadcast_dest):
    from b12x.integration.cuteafd.inkling2_loop_controller import compile_inkling2_loop_pair_norm_aot

    p = compile_inkling2_loop_pair_norm_aot(
        hidden=HIDDEN, slots=slots, broadcast_dest=broadcast_dest, eps=EPS,
    )
    gen = torch.Generator(device="cuda").manual_seed(200 + slots * 2 + int(broadcast_dest))
    dest_slots = 1 if broadcast_dest else slots
    for rows in (1, 17, 128):
        dest = torch.randn((dest_slots, rows, HIDDEN), generator=gen, device="cuda") * 3.0
        msg = torch.randn((slots, rows, HIDDEN), generator=gen, device="cuda") * 0.2
        out = torch.empty((slots, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
        p.launch(dest, msg, out, scalars=(rows,))
        d_exp = dest.expand(slots, rows, HIDDEN)
        want = torch.cat([_rms_norm(d_exp), _rms_norm(msg)], dim=-1).bfloat16()
        torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=1e-2)
        ulps = (out.view(torch.int16).int() - want.view(torch.int16).int()).abs().max().item()
        assert ulps <= 1, f"pair_norm max BF16 ulp {ulps}"

    # CUDA graph replay check with mutated inputs.
    rows = 16
    dest = torch.randn((dest_slots, rows, HIDDEN), generator=gen, device="cuda")
    msg = torch.randn((slots, rows, HIDDEN), generator=gen, device="cuda")
    out = torch.empty((slots, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
    p.launch(dest, msg, out, scalars=(rows,))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.launch(dest, msg, out, scalars=(rows,))
    dest.mul_(-2.5)
    msg.add_(1.25)
    graph.replay()
    want = torch.cat([_rms_norm(dest.expand(slots, rows, HIDDEN)), _rms_norm(msg)], dim=-1).bfloat16()
    assert (out.view(torch.int16).int() - want.view(torch.int16).int()).abs().max().item() <= 1


@pytest.mark.parametrize(
    "slots,broadcast_dest,broadcast_weight",
    [(1, False, False), (3, True, True), (3, False, False)],
)
def test_inkling2_mini_loop_blend_norm_oracle_and_graph(slots, broadcast_dest, broadcast_weight):
    from b12x.integration.cuteafd.inkling2_loop_controller import compile_inkling2_loop_blend_norm_aot

    p = compile_inkling2_loop_blend_norm_aot(
        hidden=HIDDEN, slots=slots, broadcast_dest=broadcast_dest,
        broadcast_weight=broadcast_weight, eps=EPS,
    )
    gen = torch.Generator(device="cuda").manual_seed(300 + slots * 4 + int(broadcast_dest) * 2 + int(broadcast_weight))
    dest_slots = 1 if broadcast_dest else slots
    w_slots = 1 if broadcast_weight else slots
    for rows in (1, 17, 128):
        dest = torch.randn((dest_slots, rows, HIDDEN), generator=gen, device="cuda") * 2.0
        msg = torch.randn((slots, rows, HIDDEN), generator=gen, device="cuda") * 2.0
        fg = torch.randn((slots, rows, 2 * HIDDEN), generator=gen, device="cuda") * 3.0
        weight = (0.5 + torch.rand((w_slots, HIDDEN), generator=gen, device="cuda")).bfloat16()
        out = torch.empty((slots, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        p.launch(dest, msg, fg, weight, out, scalars=(rows,))
        f_gate = fg[..., :HIDDEN].tanh()
        g_gate = fg[..., HIDDEN:].tanh()
        blended = (1.0 - f_gate) * dest.expand(slots, rows, HIDDEN) + g_gate * msg
        want = _rms_norm(blended, weight.expand(slots, HIDDEN)[:, None, :]).bfloat16()
        torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=1e-2)
        large = want.float().abs() >= 1e-2
        if large.any():
            ulps = (out[large].view(torch.int16).int() - want[large].view(torch.int16).int()).abs().max().item()
            assert ulps <= 1, f"blend_norm max BF16 ulp on |v|>=1e-2: {ulps}"

    # CUDA graph replay check with mutated inputs.
    rows = 16
    dest = torch.randn((dest_slots, rows, HIDDEN), generator=gen, device="cuda")
    msg = torch.randn((slots, rows, HIDDEN), generator=gen, device="cuda")
    fg = torch.randn((slots, rows, 2 * HIDDEN), generator=gen, device="cuda") * 2.0
    weight = (0.5 + torch.rand((w_slots, HIDDEN), generator=gen, device="cuda")).bfloat16()
    out = torch.empty((slots, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    p.launch(dest, msg, fg, weight, out, scalars=(rows,))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.launch(dest, msg, fg, weight, out, scalars=(rows,))
    dest.mul_(1.75)
    msg.mul_(-0.5)
    fg.add_(0.7)
    graph.replay()
    blended = (1.0 - fg[..., :HIDDEN].tanh()) * dest.expand(slots, rows, HIDDEN) + fg[..., HIDDEN:].tanh() * msg
    want = _rms_norm(blended, weight.expand(slots, HIDDEN)[:, None, :]).bfloat16()
    torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=1e-2)
    large = want.float().abs() >= 1e-2
    if large.any():
        assert (out[large].view(torch.int16).int() - want[large].view(torch.int16).int()).abs().max().item() <= 1


def test_inkling2_mini_controller_matches_f1940f40eb_oracle():
    """Verify exact controller agreement with authoritative precision commit f1940f40eb.

    ``f1940f40eb`` computes ``rank = (paired_bf16 @ down_bf16.T).bfloat16().float()``
    and ``forget/gain = (rank @ w_bf16.float().T).tanh()`` in F32 before the F32 blend
    and weighted RMSNorm -> BF16.
    """
    from b12x.integration.cuteafd.inkling2_loop_controller import (
        compile_inkling2_loop_blend_norm_aot, compile_inkling2_loop_pair_norm_aot,
    )

    pair_norm = compile_inkling2_loop_pair_norm_aot(hidden=HIDDEN, slots=1, broadcast_dest=False)
    blend_norm = compile_inkling2_loop_blend_norm_aot(hidden=HIDDEN, slots=1, broadcast_dest=False, broadcast_weight=False)
    gen = torch.Generator(device="cuda").manual_seed(404)
    rows, rank = 64, 64
    dest = torch.randn((1, rows, HIDDEN), generator=gen, device="cuda")
    msg = torch.randn((1, rows, HIDDEN), generator=gen, device="cuda")
    down_w = (torch.randn((rank, 2 * HIDDEN), generator=gen, device="cuda") * 0.02).bfloat16()
    forget_w = (torch.randn((HIDDEN, rank), generator=gen, device="cuda") * 0.02).bfloat16()
    gain_w = (torch.randn((HIDDEN, rank), generator=gen, device="cuda") * 0.02).bfloat16()
    fg_w = torch.cat([forget_w, gain_w], dim=0).contiguous()
    norm_w = (0.5 + torch.rand((1, HIDDEN), generator=gen, device="cuda")).bfloat16()

    paired = torch.empty((1, rows, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
    pair_norm.launch(dest, msg, paired, scalars=(rows,))
    rank_bf16 = (paired[0].float() @ down_w.float().T).bfloat16()
    fg_engine = (rank_bf16.float() @ fg_w.float().T).unsqueeze(0).contiguous()
    out_engine = torch.empty((1, rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    blend_norm.launch(dest, msg, fg_engine, norm_w, out_engine, scalars=(rows,))
    torch.cuda.synchronize()

    # Authoritative f1940f40eb oracle (reference/model.py::controller).
    paired_ref = torch.cat([_rms_norm(dest[0]), _rms_norm(msg[0])], dim=-1).bfloat16()
    rank_ref = (paired_ref @ down_w.T).float()
    forget_ref = (rank_ref @ forget_w.float().T).tanh()
    gain_ref = (rank_ref @ gain_w.float().T).tanh()
    blended_ref = (1.0 - forget_ref) * dest[0] + gain_ref * msg[0]
    out_oracle = _rms_norm(blended_ref, norm_w[0]).bfloat16()

    a, b = out_engine[0].float().flatten(), out_oracle.float().flatten()
    cos = float((a @ b) / (a.norm() * b.norm()))
    max_diff = float((a - b).abs().max())
    ulps = (out_engine[0].view(torch.int16).int() - out_oracle.view(torch.int16).int()).abs().max().item()
    print(f"inkling2_mini controller engine vs f1940f40eb oracle: cosine {cos:.7f} max|d| {max_diff:.3e} max_ulp {ulps}")
    assert cos >= 0.99999 and max_diff <= 2e-2
