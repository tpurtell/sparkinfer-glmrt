"""MoonViT position oracle matches the official compiled interpolation path."""
from __future__ import annotations

import hashlib
import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x

# Superset of processor-reachable even axes: pad to28px, patch14, side cap512.
# Includes near-prime neighbours and both sides of nominal65536-patch boundary.
GRIDS = ((30,46),(64,64),(14,28),(80,96),(2,2),(2,512),(512,2),(512,512),
         (4,4),(8,8),(16,16),(28,28),(32,32),(62,62),(66,66),(126,126),
         (128,128),(130,130),(254,254),(256,256),(258,258),(510,510),
         (2,510),(510,2),(6,502),(502,6),(30,258),(258,30),(64,512),(512,64),
         (126,512),(128,512),(130,504),(254,258),(258,254),(256,258),(258,256),
         (194,338),(338,194),(22,298))


@torch.compile(dynamic=True)
def _official_shape(table, shape):
    return F.interpolate(table.permute(2, 0, 1).unsqueeze(0), size=shape,
                         mode="bilinear").squeeze(0).permute(1, 2, 0).flatten(0, 1)


def test_compiled_bilinear_position_reuses_one_program_across_grids():
    require_b12x()
    from b12x.integration.cuteafd.vision_tower import compile_bilinear_position_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    table = torch.arange(64 * 64 * 1024, device="cuda", dtype=torch.float32).reshape(64, 64, 1024)
    table = (((table % 251) - 125) / 97).bfloat16()
    # The vendor wrapper first warms the same callable on the learned64x64 grid.
    _official_shape(table, (64, 64))
    program = compile_bilinear_position_aot()
    for height, width in GRIDS:
        expected = _official_shape(table, (height, width))
        out = torch.empty(height * width, 1024, device="cuda", dtype=torch.bfloat16)
        program.launch(table, out, scalars=[height, width])
        assert torch.equal(out, expected), (height, width, (out.float() - expected.float()).abs().max().item())
        if (height, width) == (30, 46):
            digest = hashlib.sha256(expected.cpu().view(torch.int16).numpy().tobytes()).hexdigest()
            assert digest == "271f0789a7cc129ade052b640f64de0c0f9eaa398d9904b3926a8a03070c6d53"
    # Live geometry changes use the resolved callable, including frozen replay.
    expected = _official_shape(table, (14, 28))
    # Compiled permute output may be a noncontiguous view; the raw ABI requires raster storage.
    out = torch.empty(expected.shape, device=expected.device, dtype=expected.dtype)
    with kernel_resolution_guard():
        program.launch(table, out, scalars=[14, 28])
    assert torch.equal(out, expected)


@pytest.mark.parametrize("kwargs", [dict(channels=0), dict(table_height=0), dict(table_width=-1)])
def test_bilinear_position_rejects_invalid_static_geometry(kwargs):
    from b12x.integration.cuteafd.vision_tower import compile_bilinear_position_aot
    with pytest.raises(ValueError):
        compile_bilinear_position_aot(**kwargs)


def test_every_axis_scale_coordinate_index_and_weight_matches_compiled_reference():
    require_b12x()
    import triton
    import triton.language as tl
    from b12x.integration.cuteafd.vision_tower import compile_position_axis_probe_aot

    @triton.jit
    def reference_axis(floats, indices, length: tl.int64, XBLOCK: tl.constexpr):
        # Instrumented copy of captured Inductor axis: same runtime integer
        # true-division, FP32 cast, halfpixel/FMA, clamp and truncation.
        at = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = at < length
        scale = (64 / length).to(tl.float32)
        position = tl.maximum((at.to(tl.float32) + 0.5) * scale - 0.5, 0.0)
        low = position.to(tl.int64)
        high = tl.minimum(low + 1, 63)
        weight = tl.minimum(tl.maximum(position - low.to(tl.float32), 0.0), 1.0)
        tl.store(floats + at * 3, scale, mask)
        tl.store(floats + at * 3 + 1, position, mask)
        tl.store(floats + at * 3 + 2, weight, mask)
        tl.store(indices + at * 2, low, mask)
        tl.store(indices + at * 2 + 1, high, mask)

    program = compile_position_axis_probe_aot()
    float_ref = torch.empty(512,3,device="cuda",dtype=torch.float32)
    index_ref = torch.empty(512,2,device="cuda",dtype=torch.int64)
    actual = torch.empty_like(float_ref)
    indices = torch.empty_like(index_ref)
    coordinates = 0
    for length in range(2,513):
        reference_axis[(triton.cdiv(length,256),)](float_ref,index_ref,length,XBLOCK=256,enable_fp_fusion=True)
        program.launch(actual,indices,scalars=[length])
        assert torch.equal(actual[:length].view(torch.int32),float_ref[:length].view(torch.int32)), length
        assert torch.equal(indices[:length],index_ref[:length]), length
        coordinates += length
    print(f"axis-proof lengths=511 coordinates={coordinates} scale_position_weight_bits=exact indices=exact")
    print(f"full-tensor-proof grids={len(GRIDS)} channels=1024")
