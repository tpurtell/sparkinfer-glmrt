"""cuteafd Inkling 2 native NVFP4 W4A4 routed experts vs the checkpoint's own quantization rule.

The #1915 reference covers the MXFP8 checkpoint only, so the oracle here is the
native checkpoint's contract (``hf_quant_config``: ``amax_type per_token``,
``scale_selection_rule fouroversix_mae``, ``weight_scale_format amax_denominator``)
as both pinned vendor kernels implement it — vLLM ``d664cd1e``
``ops/{nvfp4,moe,activation}.py`` and SGLang ``ec5e7ff5`` ``kernels/ops/inkling2/
nvfp4.py`` — reproduced in F64 on the CPU with the kernel's documented rounding points:

    per row: amax = max|x|; enc = 1536/amax; outer = amax/1536
    per 16-block: s6 = e4m3(min(bmax*enc/6, 448)), s4 = e4m3(min(bmax*enc/4, 448));
                  q_k = e2m1_rn(x / (s_k/enc)); keep the candidate with the smaller
                  sum |x - q_k*s_k*outer| (ties -> 6)
    FC1 out = bf16(sum (q_a s_a)(q_w s_w) * (amax * w_amax) / (1536 * denominator))
    act     = bf16(situ(gate, up) * gamma);  quantize act by the same rule (its own amax)
    FC2 out = bf16(... * (amax_act * w2_amax) / (1536 * denominator));  out = sum over slots

Weights follow the checkpoint's convention (probed on ``aac97412``): each expert matrix is
quantized as one row with encoder ``denominator / amax`` (2688 here, as the release), so
``dequant = q * s * amax / denominator`` and the vendor's output factor
``(x_amax * w_amax) / (1536 * denominator)`` splits into the activation outer ``x_amax / 1536``
and the weight outer ``w_amax / denominator``. Dequantization is exact in F64.

Bound: cosine >= 0.9999 overall and >= 0.999 per row (the fork's NVFP4 bar), plus a
tanh-saturation case, a gamma-spread case (route weight placement) and a bimodal-row
case (rows with one large block, where the 4.0 candidate wins for the small blocks).
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

COS, ROW_COS = 0.9999, 0.999
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_PROGRAMS: dict = {}
_WEIGHTS: dict = {}


def _geometry(tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["inkling2_nvfp4"].with_tp(tp)


def _e2m1_rn(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest E2M1 (ties to even at the representable midpoints), saturating at 6."""
    mag = x.abs().clamp(max=6.0)
    table = E2M1.to(x.device, x.dtype)
    # Midpoints between consecutive codes; ties go to the even code (the one with mantissa 0).
    mids = (table[:-1] + table[1:]) / 2  # 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    idx = torch.bucketize(mag, mids, right=False)  # mag < mid -> lower code
    tie = torch.isin(mag, mids)
    # Even codes (mantissa 0): 0.0, 1.0, 2.0, 4.0 at indices 0, 2, 4, 6. On a tie, choose the even neighbour.
    lower = torch.bucketize(mag, mids, right=True) - 1
    lower = lower.clamp(min=0)
    even_lower = (lower % 2) == 0
    idx = torch.where(tie, torch.where(even_lower, lower, lower + 1), idx)
    return torch.sign(x) * table[idx.clamp(max=7)]


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def quantize_rows(x: torch.Tensor, encoder: float = 1536.0):
    """Per-row fouroversix NVFP4 of F64 ``[rows, K]`` with encoder constant ``encoder``
    (activations: 1536; weights: the checkpoint's ``outer_scale_denominator``, e.g. 2688, the
    whole expert matrix as one row): (packed codes u8 [rows, K/2], e4m3 bytes [rows, K/16],
    amax [rows] F32, q*s F64 [rows, K] without the outer scale amax/encoder)."""
    rows, k = x.shape
    amax = x.abs().amax(-1)
    enc = torch.clamp(encoder / amax, max=3.4028234663852886e38)
    enc = torch.where(amax == 0, torch.full_like(enc, 3.4028234663852886e38), enc)
    dec = 1.0 / enc
    outer = amax / encoder
    blocks = x.reshape(rows, k // 16, 16)
    bmax = blocks.abs().amax(-1, keepdim=True)
    s6 = _e4m3((bmax * (enc[:, None, None] / 6.0)).clamp(max=448.0))
    s4 = _e4m3((bmax * (enc[:, None, None] * 0.25)).clamp(max=448.0))

    def candidate(s):
        inv = torch.clamp(1.0 / (s * dec[:, None, None]), max=3.4028234663852886e38)
        inv = torch.where(s == 0, torch.zeros_like(inv), inv)
        q = _e2m1_rn(blocks * inv)
        err = (q * s * outer[:, None, None] - blocks).abs().sum(-1, keepdim=True)
        return q, err

    q6, e6 = candidate(s6)
    q4, e4 = candidate(s4)
    choose4 = e4 < e6
    q = torch.where(choose4, q4, q6)
    s = torch.where(choose4, s4, s6)
    zero = bmax == 0
    q = torch.where(zero, torch.zeros_like(q), q)
    s = torch.where(zero, torch.zeros_like(s), s)
    codes = torch.bucketize(q.abs(), E2M1.to(q.device, q.dtype), right=False)
    codes = torch.where(q.abs() == 0, torch.zeros_like(codes), codes)
    codes = codes.clamp(max=7) | torch.where(q < 0, 8, 0)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8).reshape(rows, k // 2)
    exact = (q * s).reshape(rows, k)
    return packed, s.squeeze(-1).to(torch.float8_e4m3fn).view(torch.uint8), amax.float(), exact


def _weights(g, real: int):
    """Packed NVFP4 weights synthesized one expert at a time (the full F64 tensor never exists:
    peak = the quantized tensors + one expert slab) from a seeded CPU generator; one amax per
    expert (the checkpoint's amax/denominator rule applied to the whole expert matrix as one
    "row"), denominator 1. Returns the device operands and a per-expert dequantizer."""
    key = (g, real)
    if key not in _WEIGHTS:
        _WEIGHTS.clear()
        h, i, e = g.hidden, g.slice, g.experts
        # The checkpoint's weight convention (probed on aac97412 layer 0: block scales 52..448 with
        # 32k blocks at exactly 1536/6 = 256, q*s max == denominator 2688): the expert matrix is
        # one "row" encoded with denominator/amax; dequant = q * s * amax / denominator.
        DENOMINATOR = 2688.0
        denominator = torch.tensor([DENOMINATOR], dtype=torch.float32)

        def operand(seed, rows, k, real_rows=None, real_k=None):
            packed = torch.empty((e, rows, k // 2), dtype=torch.uint8)
            scales = torch.empty((e, rows, k // 16), dtype=torch.uint8)
            amax = torch.empty(e, dtype=torch.float32)
            for expert in range(e):
                gen = torch.Generator(device="cpu").manual_seed(seed * 100003 + expert)
                w = torch.randn((rows, k), generator=gen, dtype=torch.float64) * 0.02
                if real_rows is not None:
                    w[real_rows:] = 0
                if real_k is not None:
                    w[:, real_k:] = 0
                p_, s_, a_, _ = quantize_rows(w.reshape(1, -1), encoder=DENOMINATOR)
                packed[expert] = p_.reshape(rows, k // 2)
                scales[expert] = s_.reshape(rows, k // 16)
                amax[expert] = a_[0]
            tail = torch.cat([amax.view(torch.uint8), denominator.view(torch.uint8)])
            return (packed.contiguous().cuda(), torch.cat([scales.flatten(), tail]).contiguous().cuda(),
                    (packed, scales, amax, DENOMINATOR))

        w1, s1, m1 = operand(1, i, h, real_rows=real)
        w3, s3, m3 = operand(3, i, h, real_rows=real)
        w2, s2, m2 = operand(2, h, i, real_k=real)
        _WEIGHTS[key] = ((w1, s1, w3, s3, w2, s2), (m1, m3, m2))
    return _WEIGHTS[key]


def _dequant_expert(meta, expert) -> torch.Tensor:
    """F64 ``[rows, k]`` of one expert from its packed codes, E4M3 block scales, amax and the
    denominator: q * s * amax / denominator (the vendor's (x_amax * w_amax) / (1536 * denominator)
    split into its activation and weight halves)."""
    packed, scales, amax, denominator = meta
    p = packed[expert].long()
    codes = torch.stack([p & 0xF, p >> 4], -1).flatten(-2)
    table = torch.cat([E2M1, -E2M1]).double()
    q = table[codes]
    s = scales[expert].view(torch.float8_e4m3fn).double().repeat_interleave(16, -1)
    return q * s * (amax[expert].double() / denominator)


def _situ(gate, up, gate_cap, up_cap):
    return ((gate_cap * torch.tanh(gate / gate_cap)) * torch.sigmoid(gate)) * (up_cap * torch.tanh(up / up_cap))


def reference(g, x, ids, gammas, dequant):
    """F64 CPU: the contract above with the kernel's BF16 rounding points."""
    m1, m3, m2 = dequant
    x, ids, gammas = x.cpu().double(), ids.cpu(), gammas.cpu().double()
    out = torch.zeros((x.shape[0], g.hidden), dtype=torch.float64)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        xr = x[rows]
        _, _, amax, xq = quantize_rows(xr)
        xq = xq * (amax.double() / 1536.0)[:, None]  # q*s*outer: the dequantized activation
        # Dequantize this expert only (one [rows, k] F64 slab at a time).
        gate = (xq @ _dequant_expert(m1, expert).T).float().bfloat16().double()
        up = (xq @ _dequant_expert(m3, expert).T).float().bfloat16().double()
        act = (_situ(gate, up, g.situ_gate_cap, g.situ_up_cap) * gammas[rows, slots][:, None]).float().bfloat16().double()
        _, _, a_amax, aq = quantize_rows(act)
        aq = aq * (a_amax.double() / 1536.0)[:, None]
        y = (aq @ _dequant_expert(m2, expert).T).float().bfloat16().double()
        out.index_add_(0, rows, y)
    return out.float().bfloat16()


def _run(g, real, capacity, rows, seed=1, hot=0, scale=1.0, gamma_spread=1.0, bimodal=False):
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    gen = torch.Generator(device="cuda").manual_seed(seed)
    (w, dequant) = _weights(g, real)
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen) * scale
    if bimodal:
        # One 16-block per row 30x larger than the rest: the 4.0 candidate wins for the small blocks.
        x[:, :16] *= 30.0
    x = x.bfloat16()
    scores = torch.rand(rows, g.experts, device="cuda", generator=gen)
    if hot:
        scores[:, :hot] += 2.0
    ids = scores.topk(g.top_k, -1).indices.int().contiguous()
    gammas = torch.rand(rows, g.top_k, device="cuda", generator=gen)
    if gamma_spread != 1.0:
        gammas = torch.exp((gammas * 2 - 1) * torch.log(torch.tensor(gamma_spread, device="cuda")))
    gammas = gammas.float().contiguous()
    key = (g, capacity)
    if key not in _PROGRAMS:
        _PROGRAMS[key] = compile_fp8_moe_aot(g, route="stream", max_rows=capacity, wire=False)
    program = _PROGRAMS[key]
    out = torch.empty(rows, g.hidden, dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", capacity, False), dtype=torch.uint8, device="cuda")
    program.launch(x, ids, gammas, *w, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()
    expected = reference(g, x, ids, gammas, dequant).cuda()
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
    (1, 2304, 128, 17, 0), (1, 2304, 128, 128, 0), (1, 2304, 1024, 1000, 0),
    (1, 2304, 4096, 2752, 0), (1, 2304, 4096, 4096, 8),
    (2, 1152, 256, 256, 0), (2, 1152, 4096, 4096, 0),
    (6, 384, 256, 200, 8), (6, 384, 4096, 2752, 0),
]


@pytest.mark.parametrize("tp,real,capacity,rows,hot", CASES)
def test_inkling2_nvfp4_moe(tp, real, capacity, rows, hot):
    g = _geometry(tp)
    c, worst, delta, scale = _run(g, real, capacity, rows, hot=hot)
    print(f"inkling2_nvfp4 tp{tp} real {real} capacity {capacity} rows {rows} hot {hot}: "
          f"cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e} (max|ref| {scale:.3e})")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_nvfp4_route_weight_before_fc2():
    g = _geometry(1)
    c, worst, delta, _ = _run(g, 2304, 256, 256, seed=3, gamma_spread=8.0)
    print(f"inkling2_nvfp4 gamma spread: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_nvfp4_situ_saturation():
    g = _geometry(1)
    c, worst, delta, _ = _run(g, 2304, 256, 256, seed=5, scale=40.0)
    print(f"inkling2_nvfp4 saturation: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_nvfp4_fouroversix_selection():
    """Bimodal rows: the 4.0-based block scale is chosen for the small blocks; a 6-only quantizer
    would lose ~0.3 bits there and miss the per-row bound."""
    g = _geometry(1)
    c, worst, delta, _ = _run(g, 2304, 256, 256, seed=7, bimodal=True)
    print(f"inkling2_nvfp4 fouroversix: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= COS and worst >= ROW_COS


def test_inkling2_nvfp4_geometry():
    assert [_geometry(tp).slice for tp in (1, 2, 3, 6)] == [2304, 1152, 768, 384]
    g = _geometry(1)
    assert g.kind == "nvfp4dyn" and g.dynamic_nvfp4 and g.activation == "situ"
    from b12x.integration.cuteafd.fp8_moe import _resolve

    assert _resolve("auto", 1, False, "nvfp4dyn", "auto") == "stream"
    with pytest.raises(ValueError):
        _resolve("auto", 1, True, "nvfp4dyn", "auto")
