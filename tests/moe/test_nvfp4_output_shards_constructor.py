"""CPU-only factory coverage for experimental NVFP4 output sharding."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("cutlass")

from b12x.moe.fused_moe._impl import _ACTIVATION_KERNEL_SPECS


@pytest.mark.parametrize("activation", tuple(_ACTIVATION_KERNEL_SPECS))
def test_dynamic_factory_default_output_shards(activation):
    kwargs = dict(sf_vec_size=16, mma_tiler_mn=(16, 128))
    if activation == "silu_v41":
        kwargs.update(
            quant_recipe="w4a8_mx",
            w4a8_repacked=True,
            deterministic_output=True,
        )
    kernel = _ACTIVATION_KERNEL_SPECS[activation].make_dynamic_kernel(**kwargs)
    assert kernel.nvfp4_output_shards == 1


@pytest.mark.parametrize("shards", (0, 1, 2, 4, 5, 8, 10, 20, 40))
@pytest.mark.parametrize("swap_ab", (False, True))
@pytest.mark.parametrize("direct_routing", (False, True))
def test_silu_factory_forwards_nvfp4_output_shards(shards, swap_ab, direct_routing):
    kernel = _ACTIVATION_KERNEL_SPECS["silu"].make_dynamic_kernel(
        sf_vec_size=16,
        mma_tiler_mn=(16, 128),
        direct_routing=direct_routing,
        deterministic_output=True,
        swap_ab=swap_ab,
        nvfp4_output_shards=shards,
    )
    assert kernel.nvfp4_output_shards == shards
    assert kernel.v41_output_shards == shards
    assert kernel.swap_ab == swap_ab


@pytest.mark.parametrize("shards", (-1, 3, 6, True, 5.0, "5", None))
def test_silu_factory_rejects_invalid_nvfp4_output_shards(shards):
    with pytest.raises(ValueError, match="positive integer divisor of 40"):
        _ACTIVATION_KERNEL_SPECS["silu"].make_dynamic_kernel(
            sf_vec_size=16,
            mma_tiler_mn=(16, 128),
            direct_routing=True,
            deterministic_output=True,
            nvfp4_output_shards=shards,
        )


@pytest.mark.parametrize(
    "override",
    (
        {"deterministic_output": False},
        {"materialize_intermediate": True},
        {"external_route_plan": True},
        {"quant_recipe": "w4a8_mx"},
        {"work_source": "ready_queue"},
        {"mma_tiler_mn": (16, 64)},
        {"sf_vec_size": 32},
    ),
)
def test_silu_factory_rejects_unsupported_sharding(override):
    kwargs = dict(
        sf_vec_size=16,
        mma_tiler_mn=(16, 128),
        direct_routing=True,
        deterministic_output=True,
        nvfp4_output_shards=5,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match="NVFP4 output sharding requires"):
        _ACTIVATION_KERNEL_SPECS["silu"].make_dynamic_kernel(**kwargs)
