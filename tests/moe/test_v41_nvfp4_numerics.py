"""ModelOpt NVFP4 W4A8 execution of the fused V4.1 activation boundary.

The trained V4.1 expert boundary is arithmetic, not a storage format: FC1
outputs are clamped by the trained SwiGLU limit, rounded to BF16, weighted by
the router probability, rounded to BF16 again, and quantized to block FP8
before FC2. First and last projections quantize their BF16 operand with a
1e-4 amax floor.

These cases hold the activation fixed and move only the storage contract to
ModelOpt NVFP4 (K/16 E4M3 weight blocks plus per-tensor weight and activation
global scales), which is what the W4A8-on-NVFP4 recipe consumes. The oracle is
an explicit FP32 dequantization of that same storage, so a pass means the
kernel reproduces the declared arithmetic instead of merely a similar one.
"""

from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from tests.conftest import require_b12x
from tests._reference.helpers import make_tp_moe_fp4_binding, prepare_tp_moe_fp4_experts

# E2M1 codebook shared by MXFP4 and NVFP4 payloads.
E2M1_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _e4m3_scale_bytes(shape, generator) -> torch.Tensor:
    """E4M3 block scales that the W4A8-on-NVFP4 decomposition reproduces exactly.

    The hardware block-scale MMA shares one UE8M0 K/32 exponent per adjacent
    K/16 scale pair and applies the quotient as an in-register E4M3 residual.
    Power-of-two scales make that quotient exactly 1.0, so this fixture
    isolates kernel arithmetic instead of measuring the decomposition's own
    rounding, which ``test_v41_nvfp4_scale_decomposition_error_is_quantified``
    covers separately.
    """
    exponents = torch.randint(
        -3, 4, shape, device="cuda", generator=generator, dtype=torch.int32
    )
    # Powers of two are direct E4M3 encodings: biased exponent, zero mantissa.
    return ((exponents + 7) << 3).to(torch.uint8).contiguous()


def _fp8_rows(x: torch.Tensor, elements_per_scale: int) -> torch.Tensor:
    """Dynamic block-FP8 quantization with the V4.1 FC1/FC2 amax floor."""
    blocks = x.float().reshape(x.shape[0], -1, elements_per_scale)
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    return (
        (blocks / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
    ).reshape(x.shape)


def _reference(x, ids, routing, weights, scales, weight_global, activation_global):
    """FP32 oracle for the V4.1 boundary over NVFP4-stored routed experts."""
    lut = torch.tensor(E2M1_LUT, device=x.device)

    def dequantized(name, expert):
        codes = weights[name][expert].long()
        values = torch.stack((lut[codes & 15], lut[(codes >> 4) & 15]), -1).flatten(-2)
        block = scales[name][expert].view(torch.float8_e4m3fn).float()
        return values * block.repeat_interleave(16, -1) * weight_global

    out = torch.zeros_like(x, dtype=torch.float32)
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        qx = _fp8_rows(x, 32) * activation_global
        for expert in ids.unique().tolist():
            rows, slots = torch.where(ids == expert)
            gate = (qx[rows] @ dequantized("w1", expert).T).bfloat16().float().clamp_max(10)
            up = (qx[rows] @ dequantized("w3", expert).T).bfloat16().float().clamp(-10, 10)
            mid = (torch.nn.functional.silu(gate) * up * routing[rows, slots, None]).bfloat16()
            partial = (
                (_fp8_rows(mid, 32) * activation_global)
                @ dequantized("w2", expert).T
            ).bfloat16().float()
            out[rows] += partial
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior
    return out


def _build_case(m: int, n: int, experts: int, seed: int):
    torch.manual_seed(seed)
    h = 5120
    topk = 6 if n == 576 else 3
    x = (torch.randn(m, h, device="cuda") * 0.5).bfloat16()
    ids = torch.rand(m, experts, device="cuda").topk(topk, -1).indices.int()
    routing = torch.rand(m, topk, device="cuda")
    routing = (routing / routing.sum(-1, keepdim=True) * 1.5).float()

    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    weights, scales = {}, {}
    for name, shape in [
        ("w1", (experts, n, h // 2)),
        ("w3", (experts, n, h // 2)),
        ("w2", (experts, h, n // 2)),
    ]:
        weights[name] = torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=generator
        )
        scales[name] = _e4m3_scale_bytes(
            (*shape[:-1], shape[-1] // 8), generator
        )
    return x, ids, routing, weights, scales


def _run_case(m: int, n: int, experts: int, seed: int, after_graph_check=None):
    require_b12x()
    h = 5120
    weight_global = 1.0
    activation_global = 1.0
    x, ids, routing, weights, scales = _build_case(m, n, experts, seed)
    expected = _reference(
        x, ids, routing, weights, scales, weight_global, activation_global
    )

    ones = torch.ones(experts, dtype=torch.float32, device="cuda")
    prepared = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=ones,
        w1_fp4=torch.cat((weights["w3"], weights["w1"]), 1).contiguous(),
        w1_blockscale=swizzle_block_scale(
            torch.cat((scales["w3"], scales["w1"]), 1).view(torch.float8_e4m3fn)
        )
        .view(torch.uint8)
        .contiguous(),
        w1_alphas=ones,
        a2_gscale=ones,
        w2_fp4=weights["w2"].contiguous(),
        w2_blockscale=swizzle_block_scale(scales["w2"].view(torch.float8_e4m3fn))
        .view(torch.uint8)
        .contiguous(),
        w2_alphas=ones,
        quant_mode="w4a8_nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu_v41",
        swiglu_limit=10,
    )
    with make_tp_moe_fp4_binding(
        a=x,
        experts=prepared,
        topk_weights=routing,
        topk_ids=ids,
        quant_mode="w4a8_nvfp4",
        swiglu_limit=10,
        output=torch.empty_like(x),
    ) as binding:

        def check(actual, wanted, label):
            actual = actual.float()
            assert torch.isfinite(actual).all(), f"{label}: non-finite output"
            assert actual.norm() > 0, f"{label}: all-zero output"
            rel_l2 = ((actual - wanted).norm() / wanted.norm()).item()
            cosine = torch.nn.functional.cosine_similarity(
                actual.flatten(), wanted.flatten(), dim=0
            ).item()
            assert rel_l2 < 0.01, f"{label}: rel_l2 {rel_l2}"
            assert cosine > 0.9999, f"{label}: cosine {cosine}"

        for _ in range(3):
            result = binding.run()
        check(result, expected, "eager")

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = binding.run()
        x.mul_(-0.5)
        expected_changed = _reference(
            x, ids, routing, weights, scales, weight_global, activation_global
        )
        before = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == before, "graph replay allocated"
        check(result, expected_changed, "replay")
        if after_graph_check is not None:
            after_graph_check(binding, result, expected_changed)


def test_v41_nvfp4_scale_decomposition_error_is_quantified():
    """The W4A8-on-NVFP4 grid decomposition must stay high fidelity.

    ``w4a8_nvfp4`` cannot use the K/16 E4M3 weight scales directly: the
    hardware block-scale MMA shares one UE8M0 K/32 exponent per adjacent pair
    and re-applies the quotient as an in-register E4M3 residual. This test
    quantifies that step on exactly representable scales (identity) and on a
    realistic log-uniform sweep, so the fixture used by the routing cases above
    is justified rather than assumed.
    """
    require_b12x()
    from b12x.moe._shared.kernels.reference import (
        decompose_nvfp4_scales_to_mx_residual,
        nvfp4_mx_residual_quality_report,
    )

    powers_of_two = torch.exp2(
        torch.randint(-3, 4, (4096,), device="cuda").float()
    ).to(torch.float8_e4m3fn)
    ue8m0, residual = decompose_nvfp4_scales_to_mx_residual(powers_of_two)
    assert ue8m0.dtype is torch.uint8 and residual.dtype is torch.float8_e4m3fn
    recovered = (
        torch.exp2((ue8m0.to(torch.int32) - 127).float()).repeat_interleave(2, -1)
        * residual.float()
    )
    assert torch.equal(recovered, powers_of_two.float()), "powers of two must be exact"

    log_uniform = torch.exp2(
        torch.empty(8192, device="cuda").uniform_(-4.0, 4.0)
    ).to(torch.float8_e4m3fn)
    report = nvfp4_mx_residual_quality_report(log_uniform)
    # Measured envelope for a deliberately wide 2^-4..2^4 scale sweep: nothing
    # is flushed to zero, the mean residual rounding error stays under 5%, and
    # the worst single K/16 block stays under 25%. Real calibrated checkpoints
    # concentrate their scales far more tightly than this sweep.
    assert report["flushed_fraction"] == 0.0, report
    assert report["mean_rel_residual_error"] < 0.05, report
    assert report["max_rel_residual_error"] < 0.25, report


@pytest.mark.parametrize(
    "m,n,experts",
    [
        (1, 576, 8),
        (1, 576, 384),
        (16, 576, 384),
        (80, 576, 8),
        (1, 2304, 8),
        (16, 2304, 128),
        (80, 2304, 8),
    ],
)
def test_v41_nvfp4_expert_routing_and_graph(m, n, experts):
    _run_case(m, n, experts, seed=5200 + m + n)


def test_v41_nvfp4_matches_native_fp4_k32_plan_geometry():
    """The NVFP4 recipe must accept the same qualified V4.1 capacity shapes."""
    require_b12x()
    from b12x.moe import fused_moe

    for n in (576, 2304):
        plan = fused_moe.plan_weights(
            source=fused_moe.PackedSource(
                format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
                w13_layout=fused_moe.W13Layout.W13,
            ),
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A8,
                nonlinearity="silu_v41",
                io_dtype=torch.bfloat16,
                swiglu_limit=10,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=384, hidden_size=5120, intermediate_size=n
            ),
        )
        assert plan.source.format is fused_moe.PackedSourceFormat.MODELOPT_NVFP4
        assert plan.activation.mode is fused_moe.ActivationMode.A8
        assert plan.activation.nonlinearity == "silu_v41"


def test_v41_nvfp4_rejects_direct_routing_outside_qualified_shape():
    """Wider or larger A4 requests must fall back to grouped compaction."""
    require_b12x()
    from b12x.moe.fused_moe._impl import _w4a8_dynamic_direct_candidate

    assert _w4a8_dynamic_direct_candidate(
        quant_mode="w4a8_nvfp4",
        activation="silu_v41",
        routed_rows=6,
        num_experts=384,
        n=640,
        deterministic_output=True,
        planned_tile_m=16,
    )
    assert not _w4a8_dynamic_direct_candidate(
        quant_mode="w4a8_nvfp4",
        activation="silu_v41",
        routed_rows=24,
        num_experts=384,
        n=640,
        deterministic_output=True,
        planned_tile_m=16,
    )
    assert not _w4a8_dynamic_direct_candidate(
        quant_mode="w4a8_nvfp4",
        activation="silu_v41",
        routed_rows=6,
        num_experts=384,
        n=2560,
        deterministic_output=True,
        planned_tile_m=16,
    )
