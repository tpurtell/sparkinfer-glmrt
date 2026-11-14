"""cuteafd Inkling 2 MXFP8 W8A8 routed experts vs the family's reference arithmetic.

Oracle: the cuteafd #1915 reference's ``moe_forward`` rule (``python/reference/
families/inkling2/model.py``, branch ``work/inkling2-fidelity`` ``02cf57ff8``),
reproduced here in F32 with the kernel's documented rounding points:

    inputs = mxfp8(x);  gate_up = inputs . dequant(w13_e)            (F32 accumulate)
    act    = bf16(situ(bf16(gate), bf16(up)) * gamma)                 (BF16 GEMM output, BF16 act)
    y      = mxfp8(act) . dequant(w2_e);  out = sum_slots y            (unweighted; gamma is in act)

with ``mxfp8`` the UE8M0 ceil(amax/448) + E4M3 RN-satfinite rule (``quant_dequant_
mxfp8_torch`` matches the kernel's ``quantize_block_fp8_mx`` bit for bit) and SiTU
``4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)``. The reference computes SiTU on the
BF16-rounded GEMM output in F32 and feeds F32 to ``quantize_mxfp8``; the kernel
rounds the SiTU product to BF16 first (the engine's act rows are BF16), so the
oracle here does too and the remaining difference is F32 accumulation order.

Bound: cosine >= 0.9999 overall and >= 0.999 per row, like the fork's MXFP4
stream bar; plus a tanh-saturation case (large gate/up) and a route-weight case
(gamma spread 0.01..8) that would expose a gamma applied after the FC2 quant.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

COS, ROW_COS = 0.9999, 0.999
_PROGRAMS: dict = {}
_WEIGHTS: dict = {}


def _geometry(tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["inkling2_mxfp8"].with_tp(tp)


def _mxfp8(x: torch.Tensor):
    """UE8M0 ceil(amax/448) per 32, E4M3 RN satfinite: (e4m3 codes, ue8m0 bytes, exact F32)."""
    shape = x.shape
    blocks = x.float().reshape(-1, 32)
    raw = blocks.abs().amax(-1, keepdim=True) * (1.0 / 448.0)
    exponent = ((raw.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0 ** -127, torch.zeros_like(exponent), exponent.clamp(max=254))
    scale = torch.ldexp(torch.ones_like(raw), exponent - 127)
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    exact = (q.float() * scale).reshape(shape)
    return q.reshape(shape), exponent.to(torch.uint8).reshape(*shape[:-1], shape[-1] // 32), exact


def _weights(g, real: int):
    """MXFP8 weights synthesized and quantized one expert at a time from a seeded generator (the
    full F32 tensor never exists: peak = the quantized tensors + one [rows, k] F32 slab). The
    cache holds one geometry's set; a TP switch clears it before building the next."""
    key = (g, real)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        h, i, e = g.hidden, g.slice, g.experts
        from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes

        def operand(seed, rows, k, real_rows=None, real_k=None, scale_cols=None):
            scale_cols = k // 32 if scale_cols is None else scale_cols
            w = torch.empty((e, rows, k), dtype=torch.float8_e4m3fn, device="cuda")
            s = torch.zeros((e, rows, scale_cols), dtype=torch.uint8, device="cuda")
            for expert in range(e):
                gen = torch.Generator(device="cuda").manual_seed(seed * 100003 + expert)
                slab = torch.randn((rows, k), device="cuda", generator=gen) * 0.02
                if real_rows is not None:
                    slab[real_rows:] = 0
                if real_k is not None:
                    slab[:, real_k:] = 0
                q, sc, _ = _mxfp8(slab)
                w[expert] = q
                s[expert, :, :k // 32] = sc
            return w.contiguous(), s.contiguous()

        w1, s1 = operand(1, i, h, real_rows=real)
        w3, s3 = operand(3, i, h, real_rows=real)
        w2, s2 = operand(2, h, i, real_k=real, scale_cols=mxfp4_scale_row_bytes(i))
        _WEIGHTS[key] = (w1, s1, w3, s3, w2, s2)
    return _WEIGHTS[key]


def _dequant(w, s):
    return torch.ldexp(w.float().reshape(*s.shape, 32), s.int()[..., None] - 127).reshape(w.shape)


def _situ(gate, up, gate_cap=4.0, up_cap=25.0):
    return ((gate_cap * torch.tanh(gate / gate_cap)) * gate.sigmoid()) * (up_cap * torch.tanh(up / up_cap))


def reference(x_exact, ids, gammas, w1, s1, w3, s3, w2, s2_rows, g):
    i = g.slice
    out = torch.zeros((x_exact.shape[0], g.hidden), device="cuda", dtype=torch.float32)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = (x_exact[rows] @ _dequant(w1[expert], s1[expert]).T).bfloat16().float()
        up = (x_exact[rows] @ _dequant(w3[expert], s3[expert]).T).bfloat16().float()
        act = (_situ(gate, up, g.situ_gate_cap, g.situ_up_cap) * gammas[rows, slots][:, None]).bfloat16().float()
        _, _, act_q = _mxfp8(act)
        s2 = s2_rows[expert, :, :i // 32]
        y = act_q @ _dequant(w2[expert], s2).T
        out.index_add_(0, rows, y)
    return out.bfloat16()


def _run(g, real, capacity, rows, seed=1, hot=0, gamma_spread=1.0, saturate=False):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = _weights(g, real)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen)
    if saturate:
        x = x * 40.0  # gate/up far past the softcaps: tanh saturation and sigmoid tails
    q, s, x_exact = _mxfp8(x)
    wire = torch.cat([q.view(torch.uint8), s], 1).contiguous()
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 2.0
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    gammas = torch.rand(rows, g.top_k, device="cuda", generator=gen)
    if gamma_spread != 1.0:
        gammas = torch.exp((gammas * 2 - 1) * torch.log(torch.tensor(gamma_spread))).float()
    gammas = gammas.contiguous()
    key = (g, capacity)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=True)
    program = _PROGRAMS[key]
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", capacity, True), dtype=torch.uint8, device="cuda")
    program.launch(wire, ids, gammas, *w, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    expected = reference(x_exact, ids, gammas, *w, g)
    a, b = out.float(), expected.float()
    assert torch.isfinite(a).all()
    c = float((a * b).sum() / (a.norm() * b.norm()))
    worst = float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())
    delta = float((a - b).abs().max())
    return c, worst, delta, float(b.abs().max())


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    # (tp, real rows of the slice, capacity, rows, hot)
    (1, 2304, 1, 1, 0), (1, 2304, 16, 16, 0), (1, 2304, 80, 80, 0), (1, 2304, 1024, 1000, 0),
    (1, 2304, 4096, 2752, 0), (1, 2304, 4096, 4096, 8),
    (2, 1152, 16, 7, 0), (2, 1152, 256, 256, 0), (2, 1152, 4096, 4096, 0),
    (6, 384, 16, 16, 0), (6, 384, 4096, 2752, 0), (6, 384, 4096, 4096, 8),
]


@pytest.mark.parametrize("tp,real,capacity,rows,hot", CASES)
def test_inkling2_mxfp8_moe(tp, real, capacity, rows, hot):
    g = _geometry(tp)
    c, worst, delta, scale = _run(g, real, capacity, rows, hot=hot)
    print(f"inkling2_mxfp8 tp{tp} real {real} capacity {capacity} rows {rows} hot {hot}: "
          f"cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e} (max|ref| {scale:.3e})")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_mxfp8_route_weight_before_fc2():
    """gamma spread over 0.01..8: a weight applied after the FC2 quant would misplace the K32 scales."""
    g = _geometry(1)
    c, worst, delta, scale = _run(g, 2304, 256, 256, seed=3, gamma_spread=8.0)
    print(f"inkling2_mxfp8 gamma spread: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_mxfp8_situ_saturation():
    """Inputs far past the softcaps: tanh saturates at +-4 / +-25, sigmoid tails to 0 / 1."""
    g = _geometry(1)
    c, worst, delta, scale = _run(g, 2304, 256, 256, seed=5, saturate=True)
    print(f"inkling2_mxfp8 saturation: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_mxfp8_geometry():
    assert [_geometry(tp).slice for tp in (1, 2, 3, 6)] == [2304, 1152, 768, 384]
    g = _geometry(1)
    assert g.kind == "mxfp8" and g.activation == "situ" and (g.situ_gate_cap, g.situ_up_cap) == (4.0, 25.0)
    from b12x.integration.cuteafd.fp8_moe import _resolve, auto_large_rows

    assert _resolve("auto", 1, True, "mxfp8", "auto") == "stream" and auto_large_rows(True, "mxfp8") == 0
    with pytest.raises(ValueError):
        _resolve("auto", 1, False, "mxfp8", "auto")
