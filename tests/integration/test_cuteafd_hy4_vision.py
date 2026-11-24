"""Hy4-flash SigLIP2 tower primitives against the official PyTorch arithmetic."""
import pytest
import torch
import torch.nn.functional as F
from ..conftest import require_b12x


def _l2():
    from cuda.bindings import runtime
    status, size = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,
                                                  torch.cuda.current_device())
    assert int(status) == 0
    return size


def _official_position(table, h, w):
    """HYV4FlashVisionPatchEmbed.interpolate_pos_encoding / SGLang HYV4VisionPatchEmbed."""
    grid = table.float().permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(grid, scale_factor=((h + 0.1) / 128, (w + 0.1) / 128),
                        mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(-1, table.shape[-1]).bfloat16()


def _rscale(size):
    # ATen area_pixel_compute_scale with an explicit scale_factor: float(1/scale).
    return float(torch.tensor(1.0 / ((size + 0.1) / 128), dtype=torch.float32))


GRIDS = [(h, w) for h in (2, 4, 8, 16, 18, 32, 46, 64, 66, 96, 126, 128) for w in (2, 6, 32, 64, 90, 128)]


def test_scaled_bilinear_position_matches_official_interpolate():
    require_b12x()
    from b12x.integration.cuteafd.vision_tower import compile_scaled_bilinear_position_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    gen = torch.Generator().manual_seed(4)
    table = (torch.randn(128, 128, 1152, generator=gen) * 0.05).bfloat16().cuda()
    program = compile_scaled_bilinear_position_aot()
    mismatches = {}
    for h, w in GRIDS:
        expected = _official_position(table, h, w)
        out = torch.empty_like(expected)
        with kernel_resolution_guard():
            program.launch(table, out, scalars=[h, w, _rscale(h), _rscale(w)])
        bad = int((out.view(torch.int16) != expected.view(torch.int16)).sum())
        if bad:
            ulps = (out.view(torch.int16).int() - expected.view(torch.int16).int()).abs().max().item()
            mismatches[(h, w)] = (bad, ulps)
    assert not mismatches, mismatches


@pytest.mark.parametrize("height,width", [(32, 32), (32, 96), (512, 288), (2048, 1024)])
def test_still_patches_p16_raster(height, width):
    require_b12x()
    from b12x.integration.cuteafd.vision_still import compile_still_patches_aot
    rgb = (torch.arange(height * width * 3, device="cuda", dtype=torch.int32) % 253).to(torch.uint8)
    rgb = rgb.reshape(height, width, 3)
    lut = ((torch.arange(256, device="cuda", dtype=torch.float32) / 255) - .5) / .5
    lut = lut.repeat(3, 1).contiguous()
    normalized = lut[torch.arange(3, device="cuda")[None, None, :], rgb.long()]
    gh, gw = height // 16, width // 16
    # SGLang/Tencent patchify: (t,gh,gw,c,py,px) with raster patch order.
    expected = normalized.permute(2, 0, 1).reshape(3, gh, 16, gw, 16).permute(1, 3, 0, 2, 4)
    expected = expected.reshape(gh * gw, 3 * 256).bfloat16()
    out = torch.empty_like(expected)
    compile_still_patches_aot(patch=16).launch(rgb, lut, out, scalars=[height, width])
    assert torch.equal(out, expected)


def test_channel_major_group_flattens_conv2x2_exactly():
    require_b12x()
    from b12x.integration.cuteafd.vision_still import compile_spatial_group_aot
    h, w, c = 6, 10, 1152
    x = (torch.randn(h * w, c, device="cuda") * 0.3).bfloat16()
    out = torch.empty(h * w // 4, 4 * c, device="cuda", dtype=torch.bfloat16)
    compile_spatial_group_aot(channels=c, channel_major=True).launch(x, out, scalars=[h, w])
    expected = x.reshape(h // 2, 2, w // 2, 2, c).permute(0, 2, 4, 1, 3).reshape(-1, 4 * c)
    assert torch.equal(out, expected)
    # The flattened grouping is the conv's im2col for its [O,C,2,2] weight.
    weight = (torch.randn(16, c, 2, 2, device="cuda") * 0.02).bfloat16()
    conv = F.conv2d(x.float().reshape(h, w, c).permute(2, 0, 1)[None], weight.float(), stride=2)
    flat = out.float() @ weight.float().reshape(16, -1).T
    torch.testing.assert_close(flat, conv[0].permute(1, 2, 0).reshape(-1, 16), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("height,width", [(1, 1), (3, 5), (64, 32)])
def test_row_markers_close_each_merged_row(height, width):
    require_b12x()
    from b12x.integration.cuteafd.vision_tower import compile_row_markers_aot
    x = torch.randn(height * width, 3072, device="cuda").bfloat16()
    marker = torch.randn(3072, device="cuda").bfloat16()
    out = torch.empty(height * (width + 1), 3072, device="cuda", dtype=torch.bfloat16)
    compile_row_markers_aot(width=3072).launch(x, marker, out, scalars=[height, width])
    expected = torch.cat([x.reshape(height, width, 3072),
                          marker.reshape(1, 1, 3072).expand(height, 1, 3072)], dim=1).reshape(-1, 3072)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("width", [1152, 1280])
def test_bf16_residual_elementwise_matches_vector_math(width):
    require_b12x()
    from b12x.integration.cuteafd.media import compile_bf16_residual_add_aot
    rows = 333
    a, b = (torch.randn(rows, width, device="cuda").bfloat16() for _ in range(2))
    out = torch.empty_like(a)
    compile_bf16_residual_add_aot(width=width).launch(a, b, out, scalars=[rows])
    assert torch.equal(out, (a.float() + b.float()).bfloat16())


def test_dynamic_attention_h16_d72_matches_reference():
    require_b12x()
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    gen = torch.Generator().manual_seed(72)
    q, k, v = [torch.randn(16384, 16, 72, generator=gen).bfloat16().cuda() for _ in range(3)]
    dynamic = compile_dynamic_vision_attention_aot(heads=16, dim=72)
    cu = torch.empty(2, device="cuda", dtype=torch.int32)
    for rows in (1, 63, 64, 65, 129, 1024, 4160):
        cu.copy_(torch.tensor([0, rows], device="cuda", dtype=torch.int32))
        out = torch.empty(rows, 16, 72, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(16, rows, device="cuda")
        with kernel_resolution_guard():
            dynamic.launch(q, k, v, out, lse, cu, scalars=[rows, 72 ** -.5, _l2()])
        ref = F.scaled_dot_product_attention(*(t[:rows].float().transpose(0, 1) for t in (q, k, v)))
        err = (out.float().transpose(0, 1) - ref).abs().max().item()
        assert err < 2e-2, (rows, err)


@pytest.mark.parametrize("lengths", [[4160], [64, 1040, 4], [4, 64, 1040, 256, 16, 4160, 576, 1024]])
def test_dynamic_attention_segments_isolate_images_and_skip_padding(lengths):
    """The engine's only 16x72 program: segments=8, padded tails, k=1/3/8 images."""
    require_b12x()
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    batch = compile_dynamic_vision_attention_aot(heads=16, dim=72, segments=8)
    rows = sum(lengths)
    gen = torch.Generator().manual_seed(9 + len(lengths))
    q, k, v = [torch.randn(rows, 16, 72, generator=gen).bfloat16().cuda() for _ in range(3)]
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    cu = torch.tensor(offsets + [offsets[-1]] * (8 - len(lengths)), device="cuda", dtype=torch.int32)
    sentinel = torch.full((rows + 128, 16, 72), 7.0, device="cuda", dtype=torch.bfloat16)
    out = sentinel.clone()
    lse = torch.empty(16, rows, device="cuda")
    with kernel_resolution_guard():
        batch.launch(q, k, v, out, lse, cu, scalars=[rows, 72 ** -.5, _l2()])
    torch.cuda.synchronize()
    assert torch.equal(out[rows:], sentinel[rows:])
    single_cu = torch.empty(9, device="cuda", dtype=torch.int32)
    for start, n in zip(offsets, lengths):
        # The same program on that image alone (k=1) must agree bitwise: images
        # never attend across their segment, and the batch changes nothing.
        one = torch.empty(n, 16, 72, device="cuda", dtype=torch.bfloat16)
        one_lse = torch.empty(16, n, device="cuda")
        single_cu.copy_(torch.tensor([0] + [n] * 8, dtype=torch.int32))
        with kernel_resolution_guard():
            batch.launch(q[start:start + n].contiguous(), k[start:start + n].contiguous(),
                         v[start:start + n].contiguous(), one, one_lse, single_cu, scalars=[n, 72 ** -.5, _l2()])
        torch.cuda.synchronize()
        assert torch.equal(out[start:start + n], one), (start, n)
        ref = F.scaled_dot_product_attention(*(t[start:start + n].float().transpose(0, 1) for t in (q, k, v)))
        assert (one.float().transpose(0, 1) - ref).abs().max().item() < 2e-2, (start, n)


def test_segmented_attention_keeps_single_segment_key():
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    with pytest.raises(ValueError):
        compile_dynamic_vision_attention_aot(heads=16, dim=72, segments=0)
    with pytest.raises(ValueError):
        compile_dynamic_vision_attention_aot(heads=16, dim=96)
