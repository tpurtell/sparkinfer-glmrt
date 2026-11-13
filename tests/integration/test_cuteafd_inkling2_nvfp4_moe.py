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
    act     = bf16(situ(gate, up) * gamma)  with the vendor's own PTX (ex2/rcp/tanh .approx);  same row rule
    FC2 out = bf16(... * (amax_act * w2_amax) / (1536 * denominator));  out = sum over slots

Weights follow the checkpoint's convention (probed on ``aac97412``): each expert matrix is
quantized as one row with encoder ``denominator / amax`` (2688 here, as the release), so
``dequant = q * s * amax / denominator`` and the vendor's output factor
``(x_amax * w_amax) / (1536 * denominator)`` splits into the activation outer ``x_amax / 1536``
and the weight outer ``w_amax / denominator``. Dequantization is exact in F64.

Oracle form (stage-exact, main's decision on #295): the one input taken from the kernel, its own FC1
gate/up exchange, is independently pinned, so the end-to-end check is legitimate. Four gates, each strict:

  (1) FC1 exchange bit-exact against the standalone dynamic linears, and at most 1 BF16 ulp from the F64
      FC1, asserted per row (max reported);
  (2) input and act quantizers exact on every non-tie block (tie counts reported);
  (3) from the kernel's own FC1 exchange: oracle act-quant -> F64 down -> unweighted sum matches the
      route's output at cosine >= 0.9999 / per row >= 0.999 (measured 0.999996);
  (4) down GEMM at most 1 BF16 ulp from the F64 down over the kernel's own packed act; combine exact.

The end-to-end F64 cosine (full oracle from BF16 x) is reported, not gated: the per-token re-quantization
amplifies FC1's F32-accumulation ulps into whole-row re-quantizations (0.99988 / 0.9982 at 128 rows), which
no oracle short of the MMA's accumulation order reproduces. The model-level golden and fidelity are the
backstop once serving exists.

The 6-vs-4 candidate choice is the format's rule as both pinned vendor kernels fix it:
reconstruction errors in F32 (vLLM ``d664cd1e`` ``ops/nvfp4.py:190`` loads x ``.to(tl.float32)``;
SGLang ``ec5e7ff5`` ``nvfp4.py:93``), summed pairwise over 8, 4, 2, 1 (``_block_error``, vLLM
``:39-45`` / SGLang ``:31-38``), strict ``error4 < error6`` (vLLM ``:77``). The checkpoint names
only the rule (``scale_selection_rule: fouroversix_mae``); the vendor kernels define its precision,
and the oracle mirrors them, not the kernel. A block whose two candidate errors agree within
``TIE_REL`` of their magnitude is a tie: either choice is the rule's answer up to the summation's
last ulp, so such blocks are counted, not judged. Every non-tie block must match the kernel's
scale byte exactly (``test_inkling2_nvfp4_quantizer_exact``).

Bound: cosine >= 0.9999 overall and >= 0.999 per row (the fork's NVFP4 bar), plus a
tanh-saturation case, a gamma-spread case (route weight placement) and a bimodal-row
case (rows with one large block, where the 4.0 candidate wins for the small blocks).
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x

try:  # DSL imports at module scope: the decorated bodies below are re-evaluated by source there.
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32, Int32, Int64
    from b12x._lib.activation import situ_vendor
except ImportError:  # collection on a host without the DSL; the fixture skips the tests
    cutlass = cute = cuda = None

COS, ROW_COS = 0.9999, 0.999
TIE_REL = 2 ** -16  # two F32 pairwise sums of 16 terms agree to this relative gap only at a genuine tie
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_PROGRAMS: dict = {}
_WEIGHTS: dict = {}


def _geometry(tp):
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["inkling2_nvfp4"].with_tp(tp)


def _e2m1_rn(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest E2M1 with ties to even (the ``cvt.rn.satfinite.e2m1x2.f32`` rule), saturating at 6.

    Nearest of the eight magnitudes; an exact midpoint goes to the code with an even mantissa
    (codes 0, 2, 4, 6: values 0, 1, 2, 4). The rounding is applied to the F32 value the
    hardware sees, so callers pass F32-rounded inputs.
    """
    table = E2M1.to(x.device, x.dtype)
    mag = x.abs().clamp(max=6.0)
    dist = (mag[..., None] - table).abs()                      # [..., 8]
    best = dist.min(-1, keepdim=True).values
    near = dist <= best                                        # candidates at the minimum (1 or 2 on a tie)
    even = torch.tensor([True, False, True, False, True, False, True, False], device=x.device)
    pick = near & (near.sum(-1, keepdim=True) == 1) | (near & even)
    idx = pick.float().argmax(-1)
    return torch.sign(x) * table[idx]


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
    # The vendor forms these in F32 (Triton fp32): mirror it exactly, then widen for the error sum.
    enc32, dec32, bmax32 = enc.float(), dec.float(), bmax.float()
    s6 = _e4m3((bmax32 * (enc32[:, None, None] * (1.0 / 6.0))).clamp(max=448.0))
    s4 = _e4m3((bmax32 * (enc32[:, None, None] * 0.25)).clamp(max=448.0))

    def candidate(s32):
        inv32 = torch.clamp(1.0 / (s32 * dec32[:, None, None]), max=3.4028234663852886e38)
        inv32 = torch.where(s32 == 0, torch.zeros_like(inv32), inv32)
        scaled32 = blocks.float() * inv32                      # the F32 value cvt.rn sees
        q = _e2m1_rn(scaled32).double()
        s = s32.double()
        # The 6-vs-4 choice compares F32 reconstruction errors summed pairwise (vendor _block_error,
        # the kernel's order): |q*s*outer - x| in F32, then 8 pairs, 4, 2, 1. Near-ties flip with
        # any other precision or order, so the oracle mirrors it exactly.
        recon32 = (q.float() * s32) * outer.float()[:, None, None]
        err32 = (recon32 - blocks.float()).abs()
        for width in (8, 4, 2, 1):
            err32 = err32[..., :width] + err32[..., width:2 * width]
        return q, err32.double(), s

    q6, e6, s6d = candidate(s6)
    q4, e4, s4d = candidate(s4)
    choose4 = e4 < e6
    tie = (e4 - e6).abs() <= TIE_REL * torch.maximum(e4, e6).clamp_min(1e-30)
    q = torch.where(choose4, q4, q6)
    s = torch.where(choose4, s4d, s6d)
    zero = bmax == 0
    q = torch.where(zero, torch.zeros_like(q), q)
    s = torch.where(zero, torch.zeros_like(s), s)
    codes = torch.bucketize(q.abs(), E2M1.to(q.device, q.dtype), right=False)
    codes = torch.where(q.abs() == 0, torch.zeros_like(codes), codes)
    # Sign nibble as cvt.rn.satfinite.e2m1x2 emits it: a negative value that rounds to zero keeps
    # its sign (0x8, decoding to -0), so byte-level probes against the kernel are exact.
    negative = (q < 0) | ((q == 0) & (blocks < 0))
    codes = codes.clamp(max=7) | torch.where(negative, 8, 0)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8).reshape(rows, k // 2)
    exact = (q * s).reshape(rows, k)
    quantize_rows.last_tie = tie.squeeze(-1)  # [rows, K/16] bool: blocks at a candidate tie
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
            # On the GPU: this test runs under the hardware lock, so its CPU-side work must stay
            # negligible (the per-expert F64 loop took minutes on the CPU).
            packed = torch.empty((e, rows, k // 2), dtype=torch.uint8, device="cuda")
            scales = torch.empty((e, rows, k // 16), dtype=torch.uint8, device="cuda")
            amax = torch.empty(e, dtype=torch.float32, device="cuda")
            for expert in range(e):
                gen = torch.Generator(device="cuda").manual_seed(seed * 100003 + expert)
                w = torch.randn((rows, k), generator=gen, dtype=torch.float64, device="cuda") * 0.02
                if real_rows is not None:
                    w[real_rows:] = 0
                if real_k is not None:
                    w[:, real_k:] = 0
                p_, s_, a_, _ = quantize_rows(w.reshape(1, -1), encoder=DENOMINATOR)
                packed[expert] = p_.reshape(rows, k // 2)
                scales[expert] = s_.reshape(rows, k // 16)
                amax[expert] = a_[0]
            tail = torch.cat([amax.view(torch.uint8), denominator.to("cuda").view(torch.uint8)])
            return (packed.contiguous(), torch.cat([scales.flatten(), tail]).contiguous(),
                    (packed, scales, amax, DENOMINATOR))

        w1, s1, m1 = operand(1, i, h, real_rows=real)
        w3, s3, m3 = operand(3, i, h, real_rows=real)
        w2, s2, m2 = operand(2, h, i, real_k=real)
        _WEIGHTS[key] = ((w1, s1, w3, s3, w2, s2), (m1, m3, m2))
    return _WEIGHTS[key]


def _dequant_expert(meta, expert, device="cuda") -> torch.Tensor:
    """F64 ``[rows, k]`` of one expert from its packed codes, E4M3 block scales, amax and the
    denominator: q * s * amax / denominator (the vendor's (x_amax * w_amax) / (1536 * denominator)
    split into its activation and weight halves). Decoded on ``device`` (exact; the CPU path took
    seconds per expert)."""
    packed, scales, amax, denominator = meta
    p = packed[expert].to(device).long()  # no-op copy when already resident on the device
    codes = torch.stack([p & 0xF, p >> 4], -1).flatten(-2)
    table = torch.cat([E2M1, -E2M1]).double().to(device)
    q = table[codes]
    s = scales[expert].to(device).view(torch.float8_e4m3fn).double().repeat_interleave(16, -1)
    return q * s * (amax[expert].double() / denominator)


_SITU_PROGRAM: dict = {}


if cute is not None:
    class _Situ:
        """Elementwise situ_vendor over BF16 gate/up rows with one F32 gamma per row -> BF16 act."""

        def __init__(self, gate_cap: float, up_cap: float):
            self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)

        def key(self):
            return ("situ_vendor_rows", self.gate_cap, self.up_cap)

        @cute.jit
        def __call__(self, gate: cute.Pointer, up: cute.Pointer, gamma: cute.Pointer, act: cute.Pointer,
                     rows: Int32, cols: Int32, stream: cuda.CUstream):
            self.kernel(gate, up, gamma, act, cols).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

        @cute.kernel
        def kernel(self, gate: cute.Pointer, up: cute.Pointer, gamma: cute.Pointer, act: cute.Pointer, cols: Int32):
            row = Int64(cute.arch.block_idx()[0])
            g_t = cute.make_tensor(gate, cute.make_layout((Int64(1) << Int64(40),)))
            u_t = cute.make_tensor(up, cute.make_layout((Int64(1) << Int64(40),)))
            a_t = cute.make_tensor(act, cute.make_layout((Int64(1) << Int64(40),)))
            gm = Float32(cute.make_tensor(gamma, cute.make_layout((Int64(1) << Int64(40),)))[row])
            c = Int32(cute.arch.thread_idx()[0])
            while c < cols:
                at = row * Int64(cols) + Int64(c)
                a_t[at] = cutlass.BFloat16(situ_vendor(Float32(g_t[at]), Float32(u_t[at]), gm,
                                                       gate_cap=self.gate_cap, up_cap=self.up_cap))
                c = c + Int32(256)


def _situ_program(gate_cap: float, up_cap: float):
    """Elementwise ``situ_vendor`` over BF16 gate/up rows and an F32 gamma per row -> BF16 act.
    The oracle's SiTU step runs the format's own PTX (ex2/rcp/tanh .approx) rather than torch's
    exact transcendentals: the reference kernels define SiTU that way (vLLM d664cd1e
    ops/activation.py:20-28, :48-58), so an exact-math oracle is not the format's rule."""
    key = (gate_cap, up_cap)
    if key not in _SITU_PROGRAM:
        from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

        _SITU_PROGRAM[key] = compile_program(_Situ(gate_cap, up_cap), name="situ_vendor_rows", operands=(
            Operand("gate", torch.bfloat16, "[rows,cols]", align=2), Operand("up", torch.bfloat16, "[rows,cols]", align=2),
            Operand("gamma", torch.float32, "[rows]", align=4), Operand("act", torch.bfloat16, "[rows,cols]", "out", align=2)),
            scalars=(Scalar("rows"), Scalar("cols")), key=("situ_vendor_rows", gate_cap, up_cap))
    return _SITU_PROGRAM[key]


def _situ(gate, up, gamma, gate_cap=4.0, up_cap=25.0):
    """BF16 act = situ_vendor(bf16 gate, bf16 up) * gamma on the GPU: the format's own arithmetic."""
    gate_b, up_b = gate.to(torch.bfloat16).contiguous(), up.to(torch.bfloat16).contiguous()
    act = torch.empty_like(gate_b)
    _situ_program(gate_cap, up_cap).launch(gate_b, up_b, gamma.float().contiguous(), act,
                                           scalars=(gate_b.shape[0], gate_b.shape[1]))
    torch.cuda.synchronize()
    return act


def reference(g, x, ids, gammas, dequant):
    """F64 CPU: the contract above with the kernel's BF16 rounding points."""
    m1, m3, m2 = dequant
    dev = x.device
    x, ids, gammas = x.double(), ids, gammas.double()
    out = torch.zeros((x.shape[0], g.hidden), dtype=torch.float64, device=dev)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        xr = x[rows]
        _, _, amax, xq = quantize_rows(xr)
        xq = xq * (amax.double() / 1536.0)[:, None]  # q*s*outer: the dequantized activation
        # Dequantize this expert only (one [rows, k] F64 slab at a time).
        gate = (xq @ _dequant_expert(m1, expert, dev).T).float().bfloat16().double()
        up = (xq @ _dequant_expert(m3, expert, dev).T).float().bfloat16().double()
        act = _situ(gate, up, gammas[rows, slots], g.situ_gate_cap, g.situ_up_cap).double()
        _, _, a_amax, aq = quantize_rows(act)
        aq = aq * (a_amax.double() / 1536.0)[:, None]
        y = (aq @ _dequant_expert(m2, expert, dev).T).float().bfloat16().double()
        out.index_add_(0, rows, y)
    return out.float().bfloat16()


_STAGE_PROGRAMS: dict = {}


LARGE = 1e-2  # elements this large must be within 1 ulp outright: a real error there can't hide behind the metric


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> dict:
    """|a - b| in BF16 ulps at the magnitude of the larger operand (floor: BF16's smallest normal ulp).
    Returns the max ulps overall, the max ulps over elements with |a| >= LARGE, the absolute max |d| and
    the magnitude of the element carrying it."""
    mag = torch.maximum(a.abs(), b.abs()).clamp_min(2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
    d = (a - b).abs()
    ulps = d / ulp
    large = a.abs() >= LARGE
    at = int(d.argmax())
    return {"ulp": float(ulps.max()), "ulp_large": float(ulps[large].max()) if large.any() else 0.0,
            "abs": float(d.flatten()[at]), "abs_at": float(a.flatten()[at].abs())}


def _worst(acc: dict, m: dict) -> dict:
    if not acc:
        return dict(m)
    out = {"ulp": max(acc["ulp"], m["ulp"]), "ulp_large": max(acc["ulp_large"], m["ulp_large"])}
    out.update({"abs": m["abs"], "abs_at": m["abs_at"]} if m["abs"] > acc["abs"] else {"abs": acc["abs"], "abs_at": acc["abs_at"]})
    return out


def _stage_program(g, capacity):
    """The route's stages as _StreamNvfp4DynRoute runs them, with every intermediate returned by name and the
    fused gate/up's exchange dumped (diagnostics hook); plus the standalone dynamic linears' FC1."""
    key = (g, capacity)
    if key in _STAGE_PROGRAMS:
        return _STAGE_PROGRAMS[key]
    from b12x.integration.cuteafd.fp8_moe import _make_route, SMALL_GATE_CTAS_PER_SM
    from b12x.integration.cuteafd._fp8_moe_stream import STREAM_TILE_M
    from b12x.integration.cuteafd._nvfp4_moe_dynamic import StreamNvfp4LinearDyn
    from b12x._lib.utils import runtime_num_sm
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    route = _make_route(g, "stream", capacity, False)
    e, k, h, i = g.experts, g.top_k, g.hidden, g.slice
    lin = StreamNvfp4LinearDyn(n=i, k=h, experts=e)

    class Stages:
        def key(self):
            return ("inkling2_nvfp4_stages", route.key(), lin.key())

        @cute.jit
        def __call__(self, x: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                     w3: cute.Pointer, s3: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                     meta: cute.Pointer, pair_row: cute.Pointer, pair_pos: cute.Pointer, xq: cute.Pointer, xs: cute.Pointer,
                     xamax: cute.Pointer, gamma: cute.Pointer, act16: cute.Pointer, part: cute.Pointer, aq: cute.Pointer,
                     a_s: cute.Pointer, aamax: cute.Pointer, y: cute.Pointer, out: cute.Pointer, exch: cute.Pointer,
                     exch_rows: cute.Pointer, lin_g: cute.Pointer, lin_u: cute.Pointer, rows: Int32, stream: cuda.CUstream):
            pairs = rows * Int32(k)
            groups = cutlass.min(pairs, Int32(e))
            mt = (pairs + Int32(STREAM_TILE_M - 1)) // Int32(STREAM_TILE_M) + groups
            ids_i = cute.make_ptr(cutlass.Int32, Int64(ids.toint()), cute.AddressSpace.gmem, assumed_align=4)
            route.prep(ids_i, meta, pair_row, pair_pos, rows, pairs, stream)
            route.gamma(pair_pos, weights, gamma, rows, stream)
            route.rows_in(x, pair_row, meta, xq, xs, xamax, pairs, stream)
            dyn = dict(row_amax=xamax, gamma=gamma, act_bf16=act16, act_amax_part=part,
                       exchange=exch, exchange_rows=exch_rows, pair_row=pair_row)
            sms = runtime_num_sm()
            target = sms * Int32(SMALL_GATE_CTAS_PER_SM)
            if Int32(k * (i // 16)) >= target:
                route.small_gate[0](xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, mt, stream, **dyn)
            elif Int32(k * (i // 32)) >= target:
                route.small_gate[1](xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, mt, stream, **dyn)
            else:
                route.small_gate[2](xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, mt, stream, **dyn)
            if pairs > Int32(16):
                route.gate_up(xq, xs, meta, w1, s1, w3, s3, s2, aq, a_s, mt, stream, **dyn)
            route.act(act16, part, meta, aq, a_s, aamax, mt, stream)
            if Int32(k * (h // 128)) >= sms * Int32(2):
                route.small_down[2](aq, a_s, meta, w2, s2, y, mt, stream, row_amax=aamax)
            elif Int32(k * (h // 64)) >= sms * Int32(2):
                route.small_down[1](aq, a_s, meta, w2, s2, y, mt, stream, row_amax=aamax)
            else:
                route.small_down[0](aq, a_s, meta, w2, s2, y, mt, stream, row_amax=aamax)
            if pairs > Int32(16):
                route.down(aq, a_s, meta, w2, s2, y, mt, stream, row_amax=aamax)
            route.combine(y, pair_pos, weights, out, rows, stream)
            lin(xq, xs, meta, w1, s1, lin_g, mt, stream, row_amax=xamax)
            lin(xq, xs, meta, w3, s3, lin_u, mt, stream, row_amax=xamax)

    O = Operand
    ops = (O("x", torch.bfloat16, "[rows,H]"), O("ids", torch.int32, "[rows,k]", align=4), O("weights", torch.float32, "[rows,k]", align=4),
           O("w1", torch.uint8, "[w]"), O("s1", torch.uint8, "[s]", align=4), O("w3", torch.uint8, "[w]"), O("s3", torch.uint8, "[s]", align=4),
           O("w2", torch.uint8, "[w]"), O("s2", torch.uint8, "[s]", align=4),
           O("meta", torch.int32, "[m]", "out", align=4), O("pair_row", torch.int32, "[p]", "out", align=4), O("pair_pos", torch.int32, "[p]", "out", align=4),
           O("xq", torch.uint8, "[P,H/2]", "out"), O("xs", torch.uint8, "[P,H/16]", "out"), O("xamax", torch.float32, "[P]", "out", align=4),
           O("gamma", torch.float32, "[p]", "out", align=4), O("act16", torch.bfloat16, "[P,I]", "out"), O("part", torch.float32, "[P,I/16]", "out", align=4),
           O("aq", torch.uint8, "[P,I/2]", "out"), O("a_s", torch.uint8, "[P,I/16]", "out"), O("aamax", torch.float32, "[P]", "out", align=4),
           O("y", torch.bfloat16, "[p,H]", "out"), O("out", torch.bfloat16, "[rows,H]", "out"),
           O("exch", torch.bfloat16, "[P,2I]", "out"), O("exch_rows", torch.int32, "[P,4]", "out", align=4),
           O("lin_g", torch.bfloat16, "[p,I]", "out"), O("lin_u", torch.bfloat16, "[p,I]", "out"))
    _STAGE_PROGRAMS[key] = compile_program(Stages(), name="inkling2_nvfp4_stages", operands=ops, scalars=(Scalar("rows"),),
                                           key=("inkling2_nvfp4_stages", capacity, route.key(), lin.key()))
    return _STAGE_PROGRAMS[key]


def _run(g, real, capacity, rows, seed=1, hot=0, scale=1.0, gamma_spread=1.0, bimodal=False):
    """Run the route's stages on one case and judge the four strict gates; return the report dict."""
    from b12x.integration.cuteafd.fp8_moe import meta_words, stream_max_tiles
    from b12x.integration.cuteafd._fp8_moe_stream import STREAM_TILE_M

    gen = torch.Generator(device="cuda").manual_seed(seed)
    (w, dequant) = _weights(g, real)
    m1, m3, m2 = dequant
    x = torch.randn(rows, g.hidden, device="cuda", generator=gen) * scale
    if bimodal:
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
    e, k, h, i = g.experts, g.top_k, g.hidden, g.slice
    pairs = rows * k
    max_tiles = stream_max_tiles(e, pairs)
    padded = max_tiles * STREAM_TILE_M
    dev = "cuda"
    # Sentinels: 0xFF for packed/scale bytes (an E4M3 scale of 0xFF is NaN, never written), NaN elsewhere.
    ff = lambda *shape: torch.full(shape, 0xFF, dtype=torch.uint8, device=dev)  # noqa: E731
    nan = lambda *shape, dt=torch.float32: torch.full(shape, float("nan"), dtype=dt, device=dev)  # noqa: E731
    t = dict(meta=torch.zeros(meta_words(e, max_tiles), dtype=torch.int32, device=dev),
             pair_row=torch.full((pairs,), -7, dtype=torch.int32, device=dev), pair_pos=torch.full((pairs,), -7, dtype=torch.int32, device=dev),
             xq=ff(padded, h // 2), xs=ff(padded, h // 16), xamax=nan(padded), gamma=nan(pairs),
             act16=nan(padded, i, dt=torch.bfloat16), part=nan(padded, i // 16), aq=ff(padded, i // 2), a_s=ff(padded, i // 16),
             aamax=nan(padded), y=nan(pairs, h, dt=torch.bfloat16), out=nan(rows, h, dt=torch.bfloat16),
             exch=nan(padded, 2 * i, dt=torch.bfloat16), exch_rows=torch.full((padded, 4), -1, dtype=torch.int32, device=dev),
             lin_g=nan(pairs, i, dt=torch.bfloat16), lin_u=nan(pairs, i, dt=torch.bfloat16))
    program = _stage_program(g, capacity)
    program.launch(x, ids, gammas, *w, *t.values(), scalars=(rows,))
    torch.cuda.synchronize()
    # Grouped position -> padded index.
    meta = t["meta"].cpu().tolist()
    offsets = meta[4 + e:4 + 2 * e]
    tiles = meta[4 + 3 * e:4 + 3 * e + meta[1]]
    tile_of = {((c & 0xFFFF), c >> 16): tt for tt, c in enumerate(tiles)}
    ids_c, pos = ids.cpu(), t["pair_pos"].cpu().long()
    padded_of, expert_of = [], []
    for p in range(pairs):
        rr, slot = divmod(p, k)
        ex = int(ids_c[rr, slot])
        local = int(pos[p]) - offsets[ex]
        padded_of.append(tile_of[(ex, local // 128)] * 128 + local % 128)
        expert_of.append(ex)
    pi = torch.tensor(padded_of, device=dev)
    gp = pos.to(dev)
    # Written-row assertions (the sentinel discipline): every claimed row of every stage.
    for name, tens, idx, kind in (("xq", t["xq"], pi, "ff"), ("xs", t["xs"], pi, "ff"), ("xamax", t["xamax"], pi, "nan"),
                                  ("gamma", t["gamma"], gp, "nan"), ("act16", t["act16"], pi, "nan"), ("aq", t["aq"], pi, "ff"),
                                  ("a_s", t["a_s"], pi, "ff"), ("aamax", t["aamax"], pi, "nan"), ("y", t["y"], gp, "nan"),
                                  ("exch", t["exch"], pi, "nan"), ("lin_g", t["lin_g"], gp, "nan"), ("lin_u", t["lin_u"], gp, "nan")):
        sel = tens[idx]
        unwritten = (sel == 0xFF).all(-1) if kind == "ff" else torch.isnan(sel.float()).any(-1) if sel.dim() > 1 else torch.isnan(sel.float())
        assert not unwritten.any(), f"{name}: {int(unwritten.sum())} claimed rows still hold their sentinel"
    assert not torch.isnan(t["out"].float()).any(), "out holds its NaN sentinel"
    assert torch.equal(t["exch_rows"][pi, 2].cpu(), pos) and torch.equal(t["exch_rows"][pi, 3].cpu(), t["pair_row"].cpu()[pos]), \
        "exchange row records disagree with the grouping (misaddressed rows)"
    report = {}
    # (1) FC1 exchange: bit-exact vs the standalone linears; <= 1 BF16 ulp from the F64 FC1, per row.
    exch_g, exch_u = t["exch"][pi, :i], t["exch"][pi, i:]
    assert torch.equal(exch_g, t["lin_g"][gp]) and torch.equal(exch_u, t["lin_u"][gp]), "fused exchange differs from the standalone linears"
    xd = x.double()
    _, _, am_x, xqd = quantize_rows(xd)
    report["input_ties"] = int(quantize_rows.last_tie.sum())
    xqd = xqd * (am_x.double() / 1536.0)[:, None]
    fc1 = {}
    oracle_act = torch.empty(pairs, i, dtype=torch.bfloat16, device=dev)
    for ex in sorted(set(expert_of)):
        sel = [p for p in range(pairs) if expert_of[p] == ex]
        rr = torch.tensor([p // k for p in sel], device=dev)
        og = (xqd[rr] @ _dequant_expert(m1, ex, dev).T).float()
        ou = (xqd[rr] @ _dequant_expert(m3, ex, dev).T).float()
        kg, ku = exch_g[sel].float(), exch_u[sel].float()
        fc1 = _worst(_worst(fc1, _bf16_ulps(kg, og)), _bf16_ulps(ku, ou))
        # (3) from the kernel's own exchange: vendor SiTU (GPU) -> oracle act-quant -> F64 down.
        slots = torch.tensor([p % k for p in sel], device=dev)
        act = _situ(kg, ku, gammas[rr, slots], g.situ_gate_cap, g.situ_up_cap)
        oracle_act[torch.tensor(sel, device=dev)] = act
    report["fc1"] = fc1
    assert fc1["ulp"] <= 1.0 + 1e-6 and fc1["ulp_large"] <= 1.0 + 1e-6, \
        f"FC1 exchange vs F64 FC1: {fc1['ulp']:.3f} ulp (large elements {fc1['ulp_large']:.3f}), max|d| {fc1['abs']:.3e} at |v| {fc1['abs_at']:.3e}"
    # (2) quantizers exact on non-tie blocks.
    _, sc_x, _, _ = quantize_rows(xd)
    tie_x = quantize_rows.last_tie
    mism = (t["xs"][pi].cpu() != sc_x.cpu()[torch.arange(pairs) // k]) & ~tie_x.cpu()[torch.arange(pairs) // k]
    report["input_nontie_mismatch"] = int(mism.sum())
    assert report["input_nontie_mismatch"] == 0 and torch.equal(t["xamax"][pi].cpu(), am_x.float().cpu()[torch.arange(pairs) // k]), \
        "input quantizer differs from the oracle on a non-tie block or an amax"
    kernel_act = t["act16"][pi]
    assert torch.equal(kernel_act, oracle_act), "kernel act rows differ from vendor SiTU of the kernel's own exchange"
    _, sc_a, am_a, aqd = quantize_rows(kernel_act.double())
    tie_a = quantize_rows.last_tie
    report["act_ties"] = int(tie_a.sum())
    mism_a = (t["a_s"][pi] != sc_a) & ~tie_a
    report["act_nontie_mismatch"] = int(mism_a.sum())
    assert report["act_nontie_mismatch"] == 0 and torch.equal(t["aamax"][pi], am_a.float()), "act quantizer differs on a non-tie block or an amax"
    # (4) down <= 1 ulp over the kernel's own packed act; combine exact.
    table = torch.cat([E2M1, -E2M1]).double().to(dev)
    aq_k = t["aq"][pi].long()
    codes = torch.stack([aq_k & 0xF, aq_k >> 4], -1).flatten(-2)
    qk = (table[codes] * t["a_s"][pi].view(torch.float8_e4m3fn).double().repeat_interleave(16, -1)) * (t["aamax"][pi].double() / 1536.0)[:, None]
    y_k = t["y"][gp].float()
    down = {}
    o3 = torch.zeros(rows, h, dtype=torch.float64, device=dev)
    for ex in sorted(set(expert_of)):
        sel = torch.tensor([p for p in range(pairs) if expert_of[p] == ex], device=dev)
        yo = (qk[sel] @ _dequant_expert(m2, ex, dev).T).float()
        down = _worst(down, _bf16_ulps(y_k[sel], yo))
        yo3 = ((aqd[sel] * (am_a[sel].double() / 1536.0)[:, None]) @ _dequant_expert(m2, ex, dev).T).float().bfloat16().double()
        o3.index_add_(0, sel // k, yo3)
    report["down"] = down
    assert down["ulp"] <= 1.0 + 1e-6 and down["ulp_large"] <= 1.0 + 1e-6, \
        f"down vs F64 down: {down['ulp']:.3f} ulp (large elements {down['ulp_large']:.3f}), max|d| {down['abs']:.3e} at |v| {down['abs_at']:.3e}"
    ysum = torch.zeros(rows, h, dtype=torch.float32, device=dev).index_add_(0, torch.arange(pairs, device=dev) // k, y_k)
    assert torch.equal(t["out"], ysum.bfloat16()), "combine differs from the F32 slot sum of the kernel's own y rows"
    # (3) gate: kernel exchange -> oracle act-quant -> F64 down -> sum vs the route's output.
    out_f = t["out"].float()
    ref3 = o3.float().bfloat16().float()
    c3 = float((out_f.double() * ref3.double()).sum() / (out_f.double().norm() * ref3.double().norm()))
    w3_ = float(torch.nn.functional.cosine_similarity(out_f, ref3, dim=1).min())
    report["stage_exact_cos"], report["stage_exact_worst_row"] = c3, w3_
    assert c3 >= COS and w3_ >= ROW_COS, f"stage-exact reference: cosine {c3:.7f} worst row {w3_:.6f}"
    # Diagnostic (not gated): the full F64 oracle end to end.
    expected = reference(g, x, ids, gammas, dequant).float()
    report["e2e_cos"] = float((out_f.double() * expected.double()).sum() / (out_f.double().norm() * expected.double().norm()))
    report["e2e_worst_row"] = float(torch.nn.functional.cosine_similarity(out_f, expected, dim=1).min())
    return report


def _print(label, r):
    f, d = r["fc1"], r["down"]
    print(f"{label}: stage-exact cos {r['stage_exact_cos']:.7f} worst row {r['stage_exact_worst_row']:.6f}; "
          f"FC1 {f['ulp']:.2f} ulp (|v|>=1e-2: {f['ulp_large']:.2f}; max|d| {f['abs']:.3e} at |v| {f['abs_at']:.3e}); "
          f"down {d['ulp']:.2f} ulp (|v|>=1e-2: {d['ulp_large']:.2f}; max|d| {d['abs']:.3e} at |v| {d['abs_at']:.3e}); "
          f"non-tie mismatches input {r['input_nontie_mismatch']} act {r['act_nontie_mismatch']}; ties input {r['input_ties']} act {r['act_ties']}; "
          f"e2e F64 diagnostic cos {r['e2e_cos']:.7f} worst row {r['e2e_worst_row']:.6f}")


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    # (tp, real rows of the slice, capacity, rows, hot); 17 rows: ~one row per expert
    (1, 2304, 128, 17, 0), (1, 2304, 128, 128, 0), (1, 2304, 1024, 1000, 0),
    (1, 2304, 4096, 2752, 0), (1, 2304, 4096, 4096, 8),
    (2, 1152, 256, 256, 0), (2, 1152, 4096, 4096, 0),
    (6, 384, 256, 200, 8), (6, 384, 4096, 2752, 0),
]


@pytest.mark.parametrize("tp,real,capacity,rows,hot", CASES)
def test_inkling2_nvfp4_moe(tp, real, capacity, rows, hot):
    g = _geometry(tp)
    _print(f"inkling2_nvfp4 tp{tp} real {real} capacity {capacity} rows {rows} hot {hot}", _run(g, real, capacity, rows, hot=hot))


def test_inkling2_nvfp4_route_weight_before_fc2():
    _print("inkling2_nvfp4 gamma spread", _run(_geometry(1), 2304, 256, 256, seed=3, gamma_spread=8.0))


def test_inkling2_nvfp4_situ_saturation():
    _print("inkling2_nvfp4 saturation", _run(_geometry(1), 2304, 256, 256, seed=5, scale=40.0))


def test_inkling2_nvfp4_fouroversix_selection():
    _print("inkling2_nvfp4 fouroversix", _run(_geometry(1), 2304, 256, 256, seed=7, bimodal=True))


def test_inkling2_nvfp4_geometry():
    assert [_geometry(tp).slice for tp in (1, 2, 3, 6)] == [2304, 1152, 768, 384]
    g = _geometry(1)
    assert g.kind == "nvfp4dyn" and g.dynamic_nvfp4 and g.activation == "situ"
    from b12x.integration.cuteafd.fp8_moe import _resolve

    assert _resolve("auto", 1, False, "nvfp4dyn", "auto") == "stream"
    with pytest.raises(ValueError):
        _resolve("auto", 1, True, "nvfp4dyn", "auto")
