"""Vendor fused-FMA ordering is intentionally not ATen complex64 ordering."""
import pytest
import torch
from ..conftest import require_b12x


@pytest.mark.parametrize("rows", [1, 17, 60, 257])
def test_vendor_fma_rotation_matches_pinned_sglang(rows):
    require_b12x()
    from sglang.kernels.ops.attention.vision_rope import apply_fused_qk_complex_rope
    from b12x.integration.cuteafd.vision_vendor_rotary import compile_vendor_vision_qkv_rotary_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    generator = torch.Generator().manual_seed(670)
    qkv = torch.randn(rows, 3, 12, 128, generator=generator).bfloat16().cuda()
    phase = torch.randn(rows, 64, generator=generator).float().cuda()
    frequencies = torch.polar(torch.ones_like(phase), phase)
    q, k, v = qkv.unbind(1)
    expected_q, expected_k = apply_fused_qk_complex_rope(q, k, frequencies)
    outputs = [torch.empty_like(q) for _ in range(3)]
    program = compile_vendor_vision_qkv_rotary_aot(heads=12, dim=128)
    with kernel_resolution_guard():
        program.launch(qkv, frequencies.real.contiguous(), frequencies.imag.contiguous(),
            *outputs, scalars=[rows])
    for actual, expected in zip(outputs, (expected_q, expected_k, v)):
        assert torch.equal(actual.view(torch.int16), expected.contiguous().view(torch.int16))


def test_vendor_fma_preserves_runtime_row_reuse_and_previous_mismatch():
    require_b12x()
    from sglang.srt.models.kimi_k3_vl import Rope2DPosEmbRepeated
    from sglang.kernels.ops.attention.vision_rope import apply_fused_qk_complex_rope
    from b12x.integration.cuteafd.vision_vendor_rotary import compile_vendor_vision_qkv_rotary_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    generator = torch.Generator().manual_seed(670)
    # Reproduce RNG consumption of the authoritative RMS screens before K rows.
    for width in (1024, 7168):
        torch.randn(width, generator=generator)
        for rows in (1, 17, 80, 257):
            for magnitude in (1., 1e-4):
                torch.randn(rows, width, generator=generator)
    qkv = torch.randn(60, 3, 12, 128, generator=generator).bfloat16().cuda()
    grid = torch.tensor([[1, 6, 10]], dtype=torch.int64)
    frequencies = Rope2DPosEmbRepeated(128, 512, 512).get_freqs_cis(grid, torch.device("cuda"))
    program = compile_vendor_vision_qkv_rotary_aot(heads=12, dim=128)
    with kernel_resolution_guard():
        for rows in (60, 17, 1):
            source = qkv[:rows].contiguous()
            q, k, v = source.unbind(1)
            expected_q, expected_k = apply_fused_qk_complex_rope(q, k, frequencies[:rows])
            outputs = [torch.empty_like(q) for _ in range(3)]
            program.launch(source, frequencies[:rows].real.contiguous(), frequencies[:rows].imag.contiguous(),
                *outputs, scalars=[rows])
            for actual, expected in zip(outputs, (expected_q, expected_k, v)):
                assert torch.equal(actual.view(torch.int16), expected.contiguous().view(torch.int16))


@pytest.mark.parametrize("heads,dim", [(0, 128), (12, 0), (12, 127)])
def test_invalid_vendor_rotary_geometry(heads, dim):
    from b12x.integration.cuteafd.vision_vendor_rotary import compile_vendor_vision_qkv_rotary_aot
    with pytest.raises(ValueError):
        compile_vendor_vision_qkv_rotary_aot(heads=heads, dim=dim)
