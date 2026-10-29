"""CPU-only geometry tests for 512-wide warp-specialized N tiles.

The 512-wide tile keeps eight consumer warps (one thread per column would
need 512 consumers and overrun the register file) by giving each warp four
N16 records instead of two, and it sizes its own weight ring because a
512-wide unit is twice as large as a 256-wide one.
"""
import pytest

from b12x.moe._shared.kernels.w4a16 import mixed_trellis_ws as W

BLOCK = 64
MAX_SHARED = 101376  # RTX PRO 6000 Blackwell opt-in per block
REGS = 65536
CONSUMER_REGS = W._CONSUMER_REGS
PRODUCER_REGS = W._PRODUCER_REGS


def stub(tiles, bits=(2, 3)):
    """Kernel object with only the attributes the shared-layout math reads."""
    obj = object.__new__(W.W4A16MixedTrellisWSKernel)
    obj.ws_bits = bits
    obj.ws_max_bits = max(bits)
    obj.ws_a_slot = BLOCK * 256
    obj.ws_s_slot = 0
    obj.ws_tile_slot = 4 * (W._TILE_HEADER_WORDS + BLOCK)
    fixed = W._TILE_STAGES * (obj.ws_tile_slot + 2 * W._BARRIER_BYTES) + 4 * W._CLAIM_SLOTS
    obj.ws_budget = MAX_SHARED - 1024 - fixed
    obj.ws_tiles = tiles
    return obj


def plans(obj):
    return tuple(
        W.W4A16MixedTrellisWSKernel._ws_ring_plan(obj, tile, obj.ws_budget, 0, BLOCK)
        for tile in obj.ws_tiles
    )


@pytest.mark.parametrize("tile_n", (128, 192, 256, 512))
def test_ring_plan_fits_device_and_keeps_two_weight_stages(tile_n):
    obj = stub((tile_n, 256))
    plan = plans(obj)[0]
    assert plan["bc_stages"] >= 2
    assert plan["bc_slot"] == 4 * (tile_n // 16) * 32 * max(obj.ws_bits)
    assert plan["smem_bytes"] <= MAX_SHARED
    assert plan["claim_off"] > plan["tile_off"] >= plan["s_off"] >= plan["bc_off"] >= plan["a_off"] == 0


def test_wide_tile_doubles_the_unit_and_halves_the_ring():
    obj = stub((512, 256))
    wide, narrow = plans(obj)
    assert wide["bc_slot"] == 2 * narrow["bc_slot"]
    assert wide["bc_stages"] < narrow["bc_stages"]
    assert wide["bc_stages"] * wide["bc_slot"] <= MAX_SHARED


@pytest.mark.parametrize("tiles,expect", (
    ((128, 256), ((2, 4, 128, 128), (2, 8, 256, 256))),
    ((192, 256), ((2, 6, 192, 256), (2, 8, 256, 256))),
    ((256, 256), ((2, 8, 256, 256), (2, 8, 256, 256))),
    ((512, 256), ((4, 8, 256, 256), (2, 8, 256, 256))),
    ((512, 512), ((4, 8, 256, 256), (4, 8, 256, 256))),
))
def test_phase_geometry(tiles, expect):
    obj = stub(tiles)
    obj.ws_rings = plans(obj)
    kernel = W.W4A16MixedTrellisWSKernel
    for phase, (cols, warps, threads, slots) in enumerate(expect):
        kernel._ws_set_phase(obj, phase == 0)
        tile_n = tiles[phase]
        assert obj.ws_cols == cols
        assert obj.ws_consumer_warps == warps
        assert obj.ws_consumer_threads == threads
        assert obj.ws_consumer_slots == slots
        assert obj.ws_threads == slots + W._PRODUCER_THREADS
        # Every column of the tile is owned by exactly one warp.
        assert obj.ws_consumer_warps * obj.ws_cols * 16 == tile_n
        assert obj.shared_words * 4 == obj.ws_smem_bytes
        assert obj.ws_smem_bytes <= MAX_SHARED


@pytest.mark.parametrize("tile_n", (128, 192, 256, 512))
def test_cta_fits_the_register_file(tile_n):
    obj = stub((tile_n, 256))
    obj.ws_rings = plans(obj)
    W.W4A16MixedTrellisWSKernel._ws_set_phase(obj, True)
    assert obj.ws_threads <= 1024
    assert obj.ws_consumer_threads * CONSUMER_REGS + W._PRODUCER_THREADS * PRODUCER_REGS <= REGS


def test_phase_plans_are_independent_of_launch_order():
    """Each launch embeds its own ring geometry: the FC2 phase must not inherit
    the wider FC1 ring (its units are half the size, so it keeps more of them)."""
    obj = stub((512, 256))
    obj.ws_rings = plans(obj)
    kernel = W.W4A16MixedTrellisWSKernel
    kernel._ws_set_phase(obj, True)
    wide = (obj.ws_a_stages, obj.ws_bc_stages, obj.ws_bc_slot, obj.shared_words)
    kernel._ws_set_phase(obj, False)
    narrow = (obj.ws_a_stages, obj.ws_bc_stages, obj.ws_bc_slot, obj.shared_words)
    kernel._ws_set_phase(obj, True)
    assert (obj.ws_a_stages, obj.ws_bc_stages, obj.ws_bc_slot, obj.shared_words) == wide
    assert narrow[2] * 2 == wide[2]
    assert narrow[1] > wide[1]
