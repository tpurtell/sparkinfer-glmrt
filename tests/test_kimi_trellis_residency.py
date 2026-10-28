"""CPU gate for the opt-in Kimi dual-resident cooperative experiment."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py"
module = ast.parse(SOURCE.read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == "_apply_mixed_residency")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
apply = namespace["_apply_mixed_residency"]


def kernel():
    def tier(bits):
        return SimpleNamespace(num_experts=896, trellis_bits=bits, moe_block_size=8,
                               fc1=SimpleNamespace(tile_k=64, tile_n=256),
                               fc2=SimpleNamespace(tile_k=64, tile_n=256))
    return SimpleNamespace(size_m=80, hidden_size=3584, intermediate_size=512,
                           top_k=16, cta_threads=256, shared_words=8736,
                           blocks_per_sm=1, tier0=tier(2), tier1=tier(3))


def test_explicit_kimi_m80_dual_residency():
    plan = kernel()
    apply(plan, None, 101376)
    assert plan.blocks_per_sm == 1
    apply(plan, 2, 101376)
    assert plan.blocks_per_sm == 2
    assert 48 * plan.blocks_per_sm == 96


@pytest.mark.parametrize("field,value", [
    ("size_m", 1), ("size_m", 48), ("hidden_size", 4096),
    ("intermediate_size", 1024), ("top_k", 8), ("cta_threads", 512),
])
def test_other_geometries_keep_conservative_bound(field, value):
    plan = kernel()
    setattr(plan, field, value)
    with pytest.raises(ValueError, match="register"):
        apply(plan, 2, 101376)


@pytest.mark.parametrize("tier_name", ["tier0", "tier1"])
@pytest.mark.parametrize("field,value", [("num_experts", 256), ("trellis_bits", 4),
                                         ("moe_block_size", 16)])
def test_other_tiers_keep_conservative_bound(tier_name, field, value):
    plan = kernel()
    setattr(getattr(plan, tier_name), field, value)
    with pytest.raises(ValueError, match="register"):
        apply(plan, 2, 101376)


@pytest.mark.parametrize("phase", ["fc1", "fc2"])
def test_other_tiles_keep_conservative_bound(phase):
    plan = kernel()
    getattr(plan.tier0, phase).tile_n = 128
    with pytest.raises(ValueError, match="register"):
        apply(plan, 2, 101376)


def test_shared_memory_bound_still_applies():
    plan = kernel()
    plan.shared_words = 13056
    with pytest.raises(ValueError, match="shared-memory"):
        apply(plan, 2, 101376)
