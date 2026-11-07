"""Native FP8 expert weights: the decode route against an exact FP64 oracle.

The BF16-input decode route (``fp8_moe``, ``route="auto"``, ``wire=False``,
<= 64 rows) widens every E4M3 weight to the exact BF16 value and multiplies the
FP32 accumulator by the checkpoint's FP32 128x128 block scale once per K-128
block, so the weight operand carries no rounding beyond the checkpoint's own.
``bf16(w * s)``, the pre-native form, rounds the dequantized weight before the
MMA and adds error the checkpoint does not ship.

The oracle dequantizes with the unrounded ``w * s`` in FP64 (``exact=True``
semantics) and keeps the program's own BF16 product/rounding points. The
fixture is one real checkpoint layer: expert-stacked E4M3
``moe.{gate,up,down}_proj.weight`` and ``weight_scale_inv`` sliced to the first
experts, stored by ``CUTEAFD_FP8_LAYER_FIXTURE`` as ``torch.save`` of ``name``,
``hidden``, ``intermediate``, ``experts``, ``top_k``, ``swiglu_limit``,
``activation`` and ``w1/s1`` (gate), ``w3/s3`` (up), ``w2/s2`` (down) with the
checkpoint's bytes kept as ``w.view(torch.uint8)``.
"""

from __future__ import annotations

import os

import pytest
import torch

from ..conftest import require_b12x


def _fixture():
    path = os.environ.get("CUTEAFD_FP8_LAYER_FIXTURE")
    if not path:
        pytest.skip("set CUTEAFD_FP8_LAYER_FIXTURE to a real-layer expert fixture")
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry

    data = torch.load(path, map_location="cpu", weights_only=True)
    g = Fp8MoeGeometry(data["name"], hidden=int(data["hidden"]), experts=int(data["experts"]),
                       top_k=int(data["top_k"]), intermediate=int(data["intermediate"]),
                       swiglu_limit=float(data["swiglu_limit"]), activation=data["activation"])
    tensors = tuple(data[name] for name in ("w1", "s1", "w3", "s3", "w2", "s2"))
    return g, tensors


def _dequant(w, s):
    """The checkpoint's exact weight, FP64: E4M3 value times its 128x128 FP32 scale."""
    return w.view(torch.float8_e4m3fn).double() * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)


def _exact_reference(x, ids, weights, w1, s1, w3, s3, w2, s2, *, limit, activation):
    """FP64 exact-weight expert contract, the program's BF16 rounding points kept."""
    x, ids, weights = x.cpu(), ids.cpu(), weights.cpu()
    routes = torch.zeros((*ids.shape, x.shape[1]), dtype=torch.float32)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = (x[rows].double() @ _dequant(w1[expert], s1[expert]).T).float().bfloat16().float()
        up = (x[rows].double() @ _dequant(w3[expert], s3[expert]).T).float().bfloat16().float()
        if activation == "silu_post_clamp_fp32_product":
            silu = torch.nn.functional.silu(gate).clamp(max=limit)
            act = (silu * up.clamp(-limit, limit)).bfloat16()
        else:
            act = (torch.nn.functional.silu(gate).clamp(max=limit) * up).bfloat16()
        y = (act.double() @ _dequant(w2[expert], s2[expert]).T).float().bfloat16()
        routes[rows, slots] = y.float()
    out = torch.zeros(ids.shape[0], x.shape[1], dtype=torch.float32)
    for slot in range(ids.shape[1]):
        out += routes[:, slot] * weights[:, slot, None]
    return out.bfloat16()


def _metrics(out, expected):
    """Per-row relative L2 error (max and mean over rows) and cosine."""
    a = out.double()
    b = expected.double()
    relative = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)
    cosine = float((a * b).sum() / (a.norm() * b.norm()))
    return float(relative.max()), float(relative.mean()), cosine


def _weight_error(tensors):
    """max and mean |bf16(w * s) - w * s| / |w * s| over the real weights: the
    error the pre-native operand adds and the native route removes."""
    worst, total, count = 0.0, 0.0, 0
    for w, s in ((tensors[0], tensors[1]), (tensors[2], tensors[3]), (tensors[4], tensors[5])):
        for expert in range(w.shape[0]):
            exact = _dequant(w[expert], s[expert]).float()
            rounded = exact.bfloat16().float()
            relative = ((rounded - exact).abs() / exact.abs().clamp_min(1e-30)).flatten()
            worst = max(worst, float(relative.max()))
            total += float(relative.double().mean()) * relative.numel()
            count += relative.numel()
    return worst, total / count


def test_native_scale_real_layer_exact_oracle():
    """Real layer weights, rows 1/16/64: the native route matches the exact-weight
    oracle, and a row's bytes do not depend on the live row count."""
    require_b12x()
    g, tensors = _fixture()
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    device = "cuda"
    w = tuple(t.to(device) for t in tensors)
    capacity = 64
    gen = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(capacity, g.hidden, device=device, generator=gen).bfloat16()
    ids = torch.stack([torch.randperm(g.experts, device=device) for _ in range(capacity)]).int().contiguous()
    weights = torch.rand(capacity, g.top_k, device=device, generator=gen).contiguous()

    program = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", capacity, False), dtype=torch.uint8, device=device)
    out = torch.empty(capacity, g.hidden, dtype=torch.bfloat16, device=device)

    worst_weight, mean_weight = _weight_error(tensors)
    print(f"fp8_native_scale {g.name}: bf16(w*s) weight operand error max {worst_weight:.3e} "
          f"mean {mean_weight:.3e} (the pre-native route's added error; native is exact)")

    captured, measured = {}, {}
    for rows in (1, 16, 64):
        out.fill_(float("nan"))
        program.launch(x, ids, weights, *w, out, scratch, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.isfinite(out[:rows]).all(), f"rows={rows}: non-finite output"
        captured[rows] = out[:rows].clone()
        expected = _exact_reference(x[:rows], ids[:rows], weights[:rows], *tensors,
                                    limit=g.swiglu_limit, activation=g.activation)
        maximum, mean, cosine = _metrics(captured[rows].float().cpu(), expected.float())
        measured[rows] = (maximum, mean, cosine)
        print(f"fp8_native_scale {g.name} rows={rows}: max row rel-L2 {maximum:.3e} "
              f"mean row rel-L2 {mean:.3e} cosine {cosine:.9f}")

    # Row invariance: the same program and inputs, more live rows, the same bytes.
    for rows in (16, 64):
        assert torch.equal(captured[1], captured[rows][:1]), "row 0 changed with the live row count"
        assert torch.equal(captured[16], captured[rows][:16]), "rows 0-15 changed with the live row count"

    # Measured on this layer: the pre-native bf16(w * s) operand gives a 5.0e-3
    # mean relative L2 error and 0.999987 cosine at every row count; the native
    # route gives at most 8.8e-5 mean / 6.2e-4 worst-row and 0.99999999 cosine.
    for rows in (1, 16, 64):
        assert measured[rows][0] <= 2.0e-3, (rows, measured)
        assert measured[rows][1] <= 3.0e-4, (rows, measured)
        assert measured[rows][2] >= 0.9999999, (rows, measured)
