"""cuteafd Inkling 2 Mini MXFP8 W8A8 routed experts oracle (layers 20-23, TP1).

Exercises ``GEOMETRIES["inkling2_mini_mxfp8"]`` (hidden=2304, intermediate=1152,
experts=128, top_k=8, SiTU 4/25) across decode tails and the prefill live-row band.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from . import test_cuteafd_inkling2_mxfp8_moe as small_mxfp8


def _geometry():
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["inkling2_mini_mxfp8"].with_tp(1)


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


CASES = [
    (1, 1, 0),
    (16, 16, 0),
    (80, 80, 0),
    (1024, 1000, 0),
    (4096, 2048, 0),
    (4096, 2752, 0),
    (4096, 4096, 8),
]


@pytest.mark.parametrize("capacity,rows,hot", CASES)
def test_inkling2_mini_mxfp8_moe(capacity, rows, hot):
    g = _geometry()
    c, worst, delta, scale = small_mxfp8._run(g, g.intermediate, capacity, rows, hot=hot)
    print(f"inkling2_mini_mxfp8 tp1 capacity {capacity} rows {rows} hot {hot}: "
          f"cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e} (max|ref| {scale:.3e})")
    assert c >= small_mxfp8.COS and worst >= small_mxfp8.ROW_COS


def test_inkling2_mini_mxfp8_route_weight_before_fc2():
    g = _geometry()
    c, worst, delta, _ = small_mxfp8._run(g, g.intermediate, 256, 256, seed=3, gamma_spread=8.0)
    print(f"inkling2_mini_mxfp8 gamma spread: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= small_mxfp8.COS and worst >= small_mxfp8.ROW_COS


def test_inkling2_mini_mxfp8_situ_saturation():
    g = _geometry()
    c, worst, delta, _ = small_mxfp8._run(g, g.intermediate, 256, 256, seed=5, saturate=True)
    print(f"inkling2_mini_mxfp8 saturation: cosine {c:.7f} worst row {worst:.6f} max|d| {delta:.3e}")
    assert c >= small_mxfp8.COS and worst >= small_mxfp8.ROW_COS


try:
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32, Int32, Int64, Uint32
except ImportError:
    cutlass = cute = cuda = None


if cute is not None:
    class _DirectMxfp8BlockQuantProbe:
        """Exercise the production ``max_abs_32`` + ``quantize_block_fp8_mx`` directly on F32 32-element blocks."""

        @cute.jit
        def __call__(self, x: cute.Pointer, payload: cute.Pointer, scales: cute.Pointer,
                     blocks: Int32, stream: cuda.CUstream):
            self.kernel(x, payload, scales, blocks).launch(grid=(blocks, 1, 1), block=(1, 1, 1), stream=stream)

        @cute.kernel
        def kernel(self, x: cute.Pointer, payload: cute.Pointer, scales: cute.Pointer, blocks: Int32):
            from b12x._lib.intrinsics import max_abs_32, quantize_block_fp8_mx, st_global_u8, st_global_v4_u32

            b = Int64(cute.arch.block_idx()[0])
            x_t = cute.make_tensor(x, cute.make_layout((Int64(1) << Int64(40),)))
            vals = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            base = b * Int64(32)
            for j in cutlass.range_constexpr(32):
                vals[j] = Float32(x_t[base + Int64(j)])
            words, scale_byte = quantize_block_fp8_mx(vals, max_abs_32(vals))
            p_at = Int64(payload.toint()) + b * Int64(32)
            for half in cutlass.range_constexpr(2):
                st_global_v4_u32(
                    p_at + Int64(16 * half),
                    words[4 * half + 0],
                    words[4 * half + 1],
                    words[4 * half + 2],
                    words[4 * half + 3],
                )
            st_global_u8(Int64(scales.toint()) + b, cutlass.Uint8(scale_byte & Uint32(0xFF)))


def test_inkling2_mini_mxfp8_exact_ue8m0_and_block_amax():
    """Exact UE8M0 & 32-element block reduction negative control (f1940f40eb):

    1. Direct block quantizer: ``amax = 448 * (1 + 2^-12)`` requires UE8M0 exponent 128
       (scale 2.0), whereas BF16-rounding block amax truncates to 448.0 and picks UE8M0
       exponent 127 (scale 1.0, 2x wrong). Assert production ``max_abs_32`` +
       ``quantize_block_fp8_mx`` is bit-exact with exact F32 ``_mxfp8`` while the
       BF16-rounded negative control diverges.
    2. End-to-end MXFP8 MoE kernel: extreme UE8M0 weight scale ``2^-126`` with E4M3
       subnormal ``2^-9`` produces ``2^-135`` which flushes to exact 0.0 under BF16
       weight expansion, while exact F32 / hardware ``mxf8f6f4`` MMA with ``x`` scale
       ``2^+127`` yields exact finite output.
    """
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp4_scale_row_bytes
    from b12x.integration.cuteafd.fp8_moe import compile_fp8_moe_aot, fp8_moe_scratch_bytes

    probe = compile_program(
        _DirectMxfp8BlockQuantProbe(),
        name="mxfp8_exact_block_quant_probe",
        operands=(
            Operand("x", torch.float32, "[blocks,32]", align=16),
            Operand("payload", torch.uint8, "[blocks,32]", "out", align=16),
            Operand("scales", torch.uint8, "[blocks]", "out", align=1),
        ),
        scalars=(Scalar("blocks"),),
        key=("mxfp8_exact_block_quant_probe",),
    )

    gen = torch.Generator(device="cuda").manual_seed(2702)
    # Blocks 0..3 sit strictly above 448 * 2^e by 2^-12 relative (BF16 rounds amax down to 448 * 2^e).
    boundary_scales = [1.0, 0.5, 2.0, 2.0 ** -100]
    special = []
    for sc in boundary_scales:
        blk = torch.linspace(-200.0, 200.0, 32, dtype=torch.float32, device="cuda") * sc
        blk[0] = float(448.0 * sc * (1.0 + 2.0 ** -12))
        special.append(blk)
    rand_blks = torch.randn((28, 32), generator=gen, device="cuda", dtype=torch.float32) * 100.0
    x = torch.cat([torch.stack(special, dim=0), rand_blks], dim=0).contiguous()

    blocks = x.shape[0]
    got_q = torch.empty((blocks, 32), dtype=torch.uint8, device="cuda")
    got_s = torch.empty((blocks,), dtype=torch.uint8, device="cuda")
    probe.launch(x, got_q, got_s, scalars=(blocks,))
    torch.cuda.synchronize()

    ref_q, ref_s, _ = small_mxfp8._mxfp8(x)
    ref_q_u8 = ref_q.view(torch.uint8)
    ref_s = ref_s.squeeze(-1)

    # Negative control: BF16-rounded block amax inside MXFP8 quantization.
    raw_bug = x.abs().amax(-1, keepdim=True).bfloat16().float() * (1.0 / 448.0)
    exp_bug = ((raw_bug.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exp_bug = torch.where(raw_bug <= 2.0 ** -127, torch.zeros_like(exp_bug), exp_bug.clamp(max=254))
    bug_s = exp_bug.to(torch.uint8).squeeze(-1)

    bug_scale_flips = int((bug_s != ref_s).sum().item())
    assert bug_scale_flips >= len(boundary_scales), (
        f"MXFP8 negative control failed to fire: only {bug_scale_flips} UE8M0 flips"
    )
    assert torch.equal(got_s, ref_s), "MXFP8 kernel UE8M0 scale bytes must match exact F32 reduction bit-for-bit"
    assert torch.equal(got_q, ref_q_u8), "MXFP8 kernel E4M3 payload bytes must match exact F32 quantization bit-for-bit"

    # End-to-end MoE kernel negative control: extreme UE8M0 exponent 1 (2^-126) x E4M3 subnormal 2^-9
    # = 2^-135 (flushes to 0.0 in BF16 expansion, since BF16 min subnormal is 2^-133), paired with
    # x scale UE8M0 254 (2^+127) and x E4M3 = 1.0 -> exact product per K element is 2^-8.
    g = _geometry()
    e, k, h, i = g.experts, g.top_k, g.hidden, g.slice
    rows = 2
    # Wire input: first 32 columns have E4M3=1.0 (0x38) and UE8M0=254 (2^+127); remaining columns 0.
    x_q = torch.zeros((rows, h), dtype=torch.uint8, device="cuda")
    x_s = torch.zeros((rows, h // 32), dtype=torch.uint8, device="cuda")
    x_q[:, :32] = 0x38  # E4M3 1.0
    x_s[:, 0] = 254     # UE8M0 2^+127
    wire = torch.cat([x_q, x_s], dim=1).contiguous()
    x_exact = small_mxfp8._dequant(x_q.view(torch.float8_e4m3fn), x_s)

    # Expert 0 weights: w1/w3 have E4M3 subnormal 2^-9 (0x01) and UE8M0=1 (2^-126) on first 32 cols;
    # w2 has normal E4M3=1.0 (0x38) and UE8M0=127 (1.0) on first 32 cols.
    w1 = torch.zeros((e, i, h), dtype=torch.uint8, device="cuda")
    s1 = torch.zeros((e, i, h // 32), dtype=torch.uint8, device="cuda")
    w1[0, :32, :32] = 0x01  # E4M3 2^-9
    s1[0, :32, 0] = 1       # UE8M0 2^-126 -> exact dequant = 2^-135 (BF16 flushes to 0.0!)
    w3 = w1.clone()
    s3 = s1.clone()
    w2 = torch.zeros((e, h, i), dtype=torch.uint8, device="cuda")
    s2 = torch.zeros((e, h, mxfp4_scale_row_bytes(i)), dtype=torch.uint8, device="cuda")
    w2[0, :32, :32] = 0x38  # E4M3 1.0
    s2[0, :32, 0] = 127     # UE8M0 1.0

    w1_fp8, w3_fp8, w2_fp8 = w1.view(torch.float8_e4m3fn), w3.view(torch.float8_e4m3fn), w2.view(torch.float8_e4m3fn)
    # Negative control: BF16-expanded w1 flushes to exact zero!
    w1_exact = small_mxfp8._dequant(w1_fp8[0], s1[0])
    assert float(w1_exact.bfloat16().abs().max().item()) == 0.0, "expected BF16 flush-to-zero negative control"
    assert float(w1_exact.abs().max().item()) > 0.0, "exact F32 decode must remain nonzero"

    ids = torch.zeros((rows, k), dtype=torch.int32, device="cuda")
    ids[:, 1:] = torch.arange(1, k, dtype=torch.int32, device="cuda")
    gammas = torch.ones((rows, k), dtype=torch.float32, device="cuda")

    prog = compile_fp8_moe_aot(g, route="stream", max_rows=16, wire=True)
    out = torch.empty((rows, h), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "stream", 16, True), dtype=torch.uint8, device="cuda")
    prog.launch(wire, ids, gammas, w1_fp8, s1, w3_fp8, s3, w2_fp8, s2, out, scratch, scalars=(rows,))
    torch.cuda.synchronize()

    expected = small_mxfp8.reference(x_exact, ids, gammas, w1_fp8, s1, w3_fp8, s3, w2_fp8, s2, g)
    assert float(expected.float().abs().max().item()) > 1e-2, "expected nonzero exact MXFP8 MoE output"
    assert torch.allclose(out.float(), expected.float(), atol=1e-3, rtol=1e-2), (
        f"MXFP8 kernel failed extreme UE8M0 exact decode: max|d|={(out.float() - expected.float()).abs().max().item():.3e}"
    )
    print(f"inkling2_mini mxfp8 exact UE8M0/block-amax negative control PASSED: "
          f"quantizer 0/0 diffs ({bug_scale_flips} BF16-amax scale flips); "
          f"extreme UE8M0=1*E4M3(2^-9) exact max|out|={expected.float().abs().max().item():.4e} vs BF16-expanded 0.0")


def test_inkling2_mini_mxfp8_geometry():
    g = _geometry()
    assert (g.hidden, g.intermediate, g.slice, g.experts, g.top_k) == (2304, 1152, 1152, 128, 8)
    assert g.kind == "mxfp8" and g.activation == "situ"
