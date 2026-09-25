"""CPU contracts for compact and legacy strided sparse-MLA records."""
from types import SimpleNamespace

import pytest
import torch

from b12x.attention.sparse_mla import strided


def caps(**kwargs):
    return strided.Caps(device='cpu', num_q_heads=64, tp_size=2,
        max_q_rows=4, num_cache_blocks=12, **kwargs)


def test_legacy_record_default_and_invalid_widths():
    assert caps().physical_record_width == 1088
    for width in (0, 512, 577, 1024, 1089):
        with pytest.raises(ValueError, match='physical_record_width'):
            caps(physical_record_width=width)


@pytest.mark.parametrize('width', [576, 1088])
def test_record_width_reaches_native_caps_query_and_compile_job(width, monkeypatch):
    metadata = caps(physical_record_width=width)
    layout = strided._materialize_layout(metadata)
    assert layout.native.caps.physical_record_width == width
    monkeypatch.setattr(strided, 'PreparationPlan', lambda **kwargs: SimpleNamespace(**kwargs))
    declaration = strided.plan(metadata)
    assert declaration.query.cache_record_bytes == width
    assert declaration.query.physical_record_width == width
    assert declaration.query.qk_head_dim == 576
    assert declaration.query.v_head_dim == 512
    jobs = declaration._compile_jobs(None, SimpleNamespace(ordinal=0))
    assert len(jobs) == 1
    assert jobs[0].args[0].physical_record_width == width
    assert jobs[0].args[1] == 0


@pytest.mark.parametrize('width', [576, 1088])
def test_interleaved_record_view_and_capacity(width):
    metadata = caps(physical_record_width=width, max_physical_records=12*3*64)
    layout = SimpleNamespace(caps=metadata)
    owner = torch.empty((12,3,64,width), dtype=strided.FP8)
    cache = owner[:,1]
    flat, block_stride, token_stride = strided._physical_record_view(layout, cache)
    assert (block_stride, token_stride) == (3*64, 1)
    assert flat.shape == ((12-1)*3*64+64, 1, width)
    assert flat.data_ptr() == cache.data_ptr()
    assert flat.stride() == (width,width,1)
    # Selecting compact geometry cannot accidentally bind the legacy format.
    other = torch.empty((12,64,1088 if width==576 else 576), dtype=strided.FP8)
    with pytest.raises(ValueError, match='shape'):
        strided._physical_record_view(layout, other)
    with pytest.raises(ValueError, match='exceeds planned capacity'):
        strided._physical_record_view(SimpleNamespace(caps=caps(physical_record_width=width)), cache)
    # Allocator byte strides must remain an integer number of physical records.
    misaligned = torch.empty_strided((12,64,width), (64*width+16,width,1), dtype=strided.FP8)
    with pytest.raises(ValueError, match='whole'):
        strided._physical_record_view(layout, misaligned)


@pytest.mark.parametrize('tp_size,heads', [(1,128), (2,64), (8,16)])
def test_owner_and_sharded_head_geometry(tp_size, heads):
    metadata = strided.Caps(device='cpu', num_q_heads=heads, tp_size=tp_size,
        max_q_rows=4, num_cache_blocks=12, physical_record_width=576)
    layout = strided._materialize_layout(metadata)
    assert layout.native.caps.num_q_heads == heads
    with pytest.raises(ValueError, match='num_q_heads'):
        strided.Caps(device='cpu', num_q_heads=heads+1, tp_size=tp_size,
            max_q_rows=4, num_cache_blocks=12)
