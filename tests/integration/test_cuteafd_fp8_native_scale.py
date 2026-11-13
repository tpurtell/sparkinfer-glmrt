"""Native FP8 expert weights: the decode route against an exact FP64 oracle.

The BF16-input decode route (``fp8_moe``, ``route="auto"``, ``wire=False``,
<= 64 rows) widens every E4M3 weight to the exact BF16 value and multiplies the
FP32 accumulator by the checkpoint's FP32 128x128 block scale once per K-128
block, so the weight operand carries no rounding beyond the checkpoint's own.
``bf16(w * s)``, the pre-native form, rounds the dequantized weight before the
MMA and adds error the checkpoint does not ship.

The oracle dequantizes with the unrounded ``w * s`` in FP64 and keeps the
program's BF16 rounding points. A synthetic nonuniform-scale case always runs
on SM12x. The real case reads only the first expert of Step-5-Preview layer 10
from the local HF cache (or ``CUTEAFD_FP8_SNAPSHOT``), without downloads or a
private fixture format. It skips only when the checkpoint is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ..conftest import require_b12x


def _fixture(source):
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry

    if source == "synthetic":
        g = Fp8MoeGeometry("native_scale_synthetic", hidden=512, intermediate=512,
                           experts=2, top_k=2, swiglu_limit=7.,
                           activation="silu_post_clamp_fp32_product")
        gen = torch.Generator().manual_seed(475)
        tensors = []
        for n, k in ((512, 512),) * 3:
            weight = (torch.randn(g.experts, n, k, generator=gen) * 100).clamp(-448, 448)
            scale = torch.rand(g.experts, n // 128, k // 128, generator=gen) * 9e-4 + 1e-4
            tensors.extend((weight.to(torch.float8_e4m3fn).view(torch.uint8), scale))
        return g, tuple(tensors)

    from safetensors import safe_open

    override = os.environ.get("CUTEAFD_FP8_SNAPSHOT")
    if override:
        snapshot = Path(override)
        assert (snapshot / "config.json").is_file(), f"missing checkpoint: {snapshot}"
    else:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
        snapshots = sorted((cache / "models--stepfun-ai--Step-5-Preview" / "snapshots").glob("*"))
        if not snapshots:
            pytest.skip("Step-5-Preview is not cached; set CUTEAFD_FP8_SNAPSHOT")
        snapshot = snapshots[-1]
    config = json.loads((snapshot / "config.json").read_text())["text_config"]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    g = Fp8MoeGeometry("step5_layer10_expert0", hidden=int(config["hidden_size"]),
                       intermediate=int(config["moe_intermediate_size"]), experts=1, top_k=1,
                       swiglu_limit=float(config["swiglu_limits"][10]),
                       activation="silu_post_clamp_fp32_product")
    tensors = []
    for projection, shape in (("gate", (g.intermediate, g.hidden)),
                              ("up", (g.intermediate, g.hidden)),
                              ("down", (g.hidden, g.intermediate))):
        for suffix, dtype, dims in (("weight", "F8_E4M3", shape),
                                    ("weight_scale_inv", "F32", tuple(d // 128 for d in shape))):
            name = f"model.layers.10.moe.{projection}_proj.{suffix}"
            with safe_open(snapshot / index[name], framework="pt", device="cpu") as shard:
                view = shard.get_slice(name)
                assert view.get_dtype() == dtype, (name, view.get_dtype())
                assert tuple(view.get_shape()[1:]) == dims, (name, view.get_shape())
                tensor = view[:1].contiguous()
            tensors.append(tensor.view(torch.uint8) if suffix == "weight" else tensor)
    print(f"fp8_native_scale checkpoint={snapshot} layer=10 expert=0")
    return g, tuple(tensors)


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


@pytest.mark.parametrize("source", ["synthetic", "checkpoint"])
def test_native_scale_real_layer_exact_oracle(source):
    """Match exact weights and preserve bytes under changing live row counts."""
    require_b12x()
    g, tensors = _fixture(source)
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
