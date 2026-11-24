"""cuteafd Inkling 2 Mini native NVFP4 W4A4 routed experts + bit-exact div.rn oracle.

Exercises ``GEOMETRIES["inkling2_mini_nvfp4"]`` (TP1: hidden=2304, intermediate=1152,
experts=128, top_k=8, SiTU 4/25, fouroversix_mae dynamic amax) through the same
4-gate stage-exact oracle as Inkling 2 Small, plus an explicit bit-exact validation
of the device quantizer's ``div.rn`` scale and E2M1 code selection against the
pinned Inkling 2 Mini reference oracle (cuteafd PR #2702 commits ``50cca1a1a8``,
``6bcb2a8da0``, head ``ece18287ff``), including the pinned ``6bcb2a8da0`` F32 hex
regression vector and synthetic MAE/E2M1 tie cases.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x
from . import test_cuteafd_inkling2_nvfp4_moe as small_nvfp4

try:
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32, Int32, Int64, Uint32, Uint64
except ImportError:
    cutlass = cute = cuda = None

# Pinned NVFP4 div.rn regression vector (cuteafd commit 6bcb2a8da0).
PINNED_DIV_RN_HEX = (
    "0x1.d74f54p+0", "0x1.3696c0p+2", "-0x1.6b615ap+1", "0x1.3fb5d8p+1",
    "0x1.1796acp+0", "0x1.3e88bap+0", "0x1.9a6f64p-1", "-0x1.fd129ap+0",
    "0x1.5b7c24p+2", "-0x1.a0fb5ep+3", "-0x1.131224p+2", "0x1.2f7a24p+2",
    "0x1.13015ep+1", "0x1.150cfcp+0", "-0x1.77c082p+2", "0x1.99cc62p+1",
)
PINNED_DIV_RN_LANE8_EXPECTED = 4.34356164932251


def _geometry():
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES

    return GEOMETRIES["inkling2_mini_nvfp4"].with_tp(1)


def _reference_quantize_nvfp4_exact(x: torch.Tensor):
    """Exact mirror of cuteafd reference ``quantize_nvfp4`` (commit ece18287ff).

    Returns ``(packed u8 [rows, K/2], scale u8 [rows, K/16], amax f32 [rows],
    reconstructed f32 [rows, K])`` so tests can check bit-exact scale bytes,
    packed E2M1 codes, and reconstructed values simultaneously.
    """
    x = x.float()
    rows, k = x.shape
    blocks = x.reshape(rows, k // 16, 16)
    amax = x.abs().amax(-1, keepdim=True).unsqueeze(-1)
    largest = torch.finfo(torch.float32).max
    encoder = torch.full_like(amax, 1536.0).div(amax).clamp_max(largest)
    decoder = torch.ones_like(encoder).div(encoder)
    outer = amax * (1.0 / 1536.0)
    bmax = blocks.abs().amax(-1, keepdim=True)
    levels = small_nvfp4.E2M1.to(x.device, torch.float32)
    cuts = (levels[:-1] + levels[1:]) * 0.5

    def candidate(inverse: float):
        scale_u8 = (bmax * (encoder * inverse)).clamp_max(448.0).to(torch.float8_e4m3fn)
        scale = scale_u8.float()
        inverse_scale = torch.ones_like(scale).div(scale * decoder).clamp_max(largest)
        normalized = blocks * inverse_scale
        magnitude = normalized.abs()
        code = torch.bucketize(magnitude.contiguous(), cuts)
        upper = (code < 7) & ((code & 1) != 0) & (magnitude == cuts[code.clamp_max(6)])
        mag_code = code + upper
        value = torch.copysign(levels[mag_code], normalized)
        reconstructed = (value * scale) * outer
        error = (reconstructed - blocks).abs()
        for _ in range(4):
            error = error[..., ::2] + error[..., 1::2]
        sign_bit = torch.signbit(normalized).to(mag_code.dtype) << 3
        return reconstructed, error, scale_u8.view(torch.uint8), (mag_code | sign_bit)

    six, err6, s6, c6 = candidate(1.0 / 6.0)
    four, err4, s4, c4 = candidate(0.25)
    choose4 = err4 < err6
    zero = bmax == 0
    recon = torch.where(zero, torch.zeros_like(six), torch.where(choose4, four, six)).reshape(rows, k)
    scales = torch.where(zero, torch.zeros_like(s6), torch.where(choose4, s4, s6)).squeeze(-1)
    codes = torch.where(zero, torch.zeros_like(c6), torch.where(choose4, c4, c6))
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8).reshape(rows, k // 2)
    return packed, scales, amax.reshape(rows), recon


if cute is not None:
    class _DirectBlockQuantProbe:
        """Exercise the production ``_row_factors`` + ``_quantize_row16_dyn`` / ``_candidate_fast`` directly on F32 rows."""

        def __init__(self, fast: bool = False):
            self.fast = bool(fast)

        @cute.jit
        def __call__(self, x: cute.Pointer, packed: cute.Pointer, scales: cute.Pointer,
                     amax_out: cute.Pointer, rows: Int32, blocks_per_row: Int32, stream: cuda.CUstream):
            self.kernel(x, packed, scales, amax_out, rows, blocks_per_row).launch(
                grid=(rows, 1, 1), block=(1, 1, 1), stream=stream)

        @cute.kernel
        def kernel(self, x: cute.Pointer, packed: cute.Pointer, scales: cute.Pointer,
                   amax_out: cute.Pointer, rows: Int32, blocks_per_row: Int32):
            from b12x._lib.intrinsics import cvt_f32_to_e4m3, fabs_f32, fmax_f32, fmin_f32, fp8_e4m3_to_f32
            from b12x._lib.quant.nvfp4_e2m1 import max_abs_16
            from b12x.integration.cuteafd._nvfp4_moe_dynamic import (
                _candidate_fast, _quantize_row16_dyn, _row_factors, st_global_u64, st_global_u8,
            )

            row = Int64(cute.arch.block_idx()[0])
            k = Int64(blocks_per_row) * Int64(16)
            x_t = cute.make_tensor(x, cute.make_layout((Int64(1) << Int64(40),)))
            amax_t = cute.make_tensor(amax_out, cute.make_layout((Int64(1) << Int64(40),)))
            amax = Float32(0.0)
            for idx in range(k):
                amax = fmax_f32(amax, fabs_f32(Float32(x_t[row * k + idx])))
            amax_t[row] = amax
            encoder, decoder, outer = _row_factors(amax)
            vals = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
            for b in range(blocks_per_row):
                base = row * k + Int64(b) * Int64(16)
                for j in cutlass.range_constexpr(16):
                    vals[j] = Float32(x_t[base + Int64(j)])
                p_at = Int64(packed.toint()) + (row * Int64(blocks_per_row) + Int64(b)) * Int64(8)
                s_at = Int64(scales.toint()) + row * Int64(blocks_per_row) + Int64(b)
                if cutlass.const_expr(not self.fast):
                    _quantize_row16_dyn(vals, encoder, decoder, outer, p_at, s_at)
                else:
                    bmax = max_abs_16(vals)
                    s6_u32 = cvt_f32_to_e4m3(fmin_f32(bmax * (encoder * Float32(1.0 / 6.0)), Float32(448.0)))
                    s4_u32 = cvt_f32_to_e4m3(fmin_f32(bmax * (encoder * Float32(0.25)), Float32(448.0)))
                    p6, e6 = _candidate_fast(vals, fp8_e4m3_to_f32(s6_u32), decoder, outer)
                    p4, e4 = _candidate_fast(vals, fp8_e4m3_to_f32(s4_u32), decoder, outer)
                    choose4 = e4 < e6
                    zero = bmax == Float32(0.0)
                    pk = Uint64(cutlass.select_(zero, Uint64(0), cutlass.select_(choose4, p4, p6)))
                    sc = Uint32(cutlass.select_(zero, Uint32(0), cutlass.select_(choose4, s4_u32, s6_u32)))
                    st_global_u64(p_at, pk)
                    st_global_u8(s_at, cutlass.Uint8(sc & Uint32(0xFF)))


def _decode_packed(packed: torch.Tensor, scales: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    p = packed.long()
    codes = torch.stack([p & 0xF, p >> 4], -1).flatten(-2)
    table = torch.cat([small_nvfp4.E2M1, -small_nvfp4.E2M1]).to(packed.device, torch.float32)
    q = table[codes]
    s = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    outer = (amax.float() * (1.0 / 1536.0))[:, None]
    return (q * s) * outer


@pytest.fixture(scope="module", autouse=True)
def _setup():
    require_b12x()
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("fast", [False, True])
def test_inkling2_mini_nvfp4_div_rn_bit_exact(fast):
    """Bit-exact scale and E2M1 code agreement with reference commit ece18287ff on pinned div.rn + tie cases."""
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    probe = compile_program(
        _DirectBlockQuantProbe(fast=fast),
        name=f"nvfp4_div_rn_probe_{'fast' if fast else 'exact'}",
        operands=(
            Operand("x", torch.float32, "[rows,k]", align=16),
            Operand("packed", torch.uint8, "[rows,k/2]", "out", align=8),
            Operand("scales", torch.uint8, "[rows,k/16]", "out", align=1),
            Operand("amax_out", torch.float32, "[rows]", "out", align=4),
        ),
        scalars=(Scalar("rows"), Scalar("blocks_per_row")),
        key=("nvfp4_div_rn_probe", fast),
    )

    pinned = torch.tensor([float.fromhex(v) for v in PINNED_DIV_RN_HEX], dtype=torch.float32, device="cuda")
    # Row 0: exact 16-element pinned div.rn vector (padded with zeros to 32 elements -> identical amax).
    # Row 1: exact E2M1 midpoint tie cases (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 -> even codes 0, 1, 1, 2, 2, 4, 4).
    # Row 2: exact scale-tie / exact-representable 6-vs-4 cases + zero block.
    # Row 3: extreme bimodal row triggering E4M3 448 clamp and tiny sub-block.
    # Rows 4..67: seeded random F32 + BF16-rounded rows spanning normal and bimodal scales.
    gen = torch.Generator(device="cuda").manual_seed(2702)
    row0 = torch.cat([pinned, torch.zeros(16, device="cuda")])
    midpoints = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
         -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0],
        dtype=torch.float32, device="cuda",
    )
    row1 = torch.cat([midpoints, midpoints * 0.5])
    row2 = torch.cat([torch.full((16,), 4.0, device="cuda"), torch.zeros(16, device="cuda")])
    row3 = torch.cat([torch.tensor([1e4] + [0.5] * 15, device="cuda"), torch.linspace(-1.0, 1.0, 16, device="cuda")])
    rand_f32 = torch.randn((32, 32), generator=gen, device="cuda", dtype=torch.float32) * 3.0
    rand_bf16 = (torch.randn((32, 32), generator=gen, device="cuda", dtype=torch.float32) * 3.0).bfloat16().float()
    x = torch.cat([row0[None], row1[None], row2[None], row3[None], rand_f32, rand_bf16], dim=0).contiguous()

    rows, k = x.shape
    packed = torch.empty((rows, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((rows, k // 16), dtype=torch.uint8, device="cuda")
    amax_out = torch.empty((rows,), dtype=torch.float32, device="cuda")
    probe.launch(x, packed, scales, amax_out, scalars=(rows, k // 16))
    torch.cuda.synchronize()

    ref_packed, ref_scales, ref_amax, ref_recon = _reference_quantize_nvfp4_exact(x)
    got_recon = _decode_packed(packed, scales, amax_out)

    assert torch.equal(amax_out, ref_amax), "row amax mismatch vs reference oracle"
    assert torch.equal(scales, ref_scales), f"E4M3 scale bytes mismatch (fast={fast})"
    assert torch.equal(packed, ref_packed), f"packed E2M1 codes mismatch (fast={fast})"
    assert torch.equal(got_recon, ref_recon), f"reconstructed values mismatch (fast={fast})"
    assert got_recon[0, 8].item() == PINNED_DIV_RN_LANE8_EXPECTED, (
        f"pinned 6bcb2a8da0 lane 8: got {got_recon[0, 8].item()}, want {PINNED_DIV_RN_LANE8_EXPECTED}"
    )
    print(f"inkling2_mini nvfp4 div.rn bit-exact (fast={fast}): {rows} rows x {k // 16} blocks exact; "
          f"6bcb2a8da0 lane8={got_recon[0, 8].item()!r}")


CASES = [
    # (capacity, rows, hot) covering decode tails, sparse->wide transition (16->17), and prefill band
    (16, 1, 0),
    (16, 8, 0),
    (16, 16, 0),
    (128, 17, 0),
    (128, 128, 0),
    (1024, 1000, 0),
    (4096, 2048, 0),
    (4096, 2752, 0),
    (4096, 4096, 8),
]


@pytest.mark.parametrize("capacity,rows,hot", CASES)
def test_inkling2_mini_nvfp4_moe(capacity, rows, hot):
    g = _geometry()
    small_nvfp4._print(
        f"inkling2_mini_nvfp4 tp1 capacity {capacity} rows {rows} hot {hot}",
        small_nvfp4._run(g, g.intermediate, capacity, rows, hot=hot),
    )


def test_inkling2_mini_nvfp4_route_weight_before_fc2():
    g = _geometry()
    small_nvfp4._print(
        "inkling2_mini_nvfp4 gamma spread",
        small_nvfp4._run(g, g.intermediate, 256, 256, seed=3, gamma_spread=8.0),
    )


def test_inkling2_mini_nvfp4_situ_saturation():
    g = _geometry()
    small_nvfp4._print(
        "inkling2_mini_nvfp4 saturation",
        small_nvfp4._run(g, g.intermediate, 256, 256, seed=5, scale=40.0),
    )


def test_inkling2_mini_nvfp4_fouroversix_selection():
    g = _geometry()
    small_nvfp4._print(
        "inkling2_mini_nvfp4 fouroversix",
        small_nvfp4._run(g, g.intermediate, 256, 256, seed=7, bimodal=True),
    )


def test_inkling2_mini_nvfp4_geometry():
    g = _geometry()
    assert (g.hidden, g.intermediate, g.slice, g.experts, g.top_k) == (2304, 1152, 1152, 128, 8)
    assert g.kind == "nvfp4dyn" and g.dynamic_nvfp4 and g.activation == "situ"
