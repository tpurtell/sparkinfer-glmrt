"""MoonViT position oracle matches the official compiled interpolation path."""
from __future__ import annotations

import hashlib
import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x


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
    for height, width in ((30, 46), (64, 64), (14, 28), (80, 96)):
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
