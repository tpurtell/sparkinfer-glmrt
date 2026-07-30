from __future__ import annotations

import pytest

from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    _w4a16_fused_dsl_opt_level,
)


@pytest.mark.parametrize("size_m", (1, 2, 4, 6))
def test_packed_bf16_direct_fused_w4a16_uses_o3(size_m: int) -> None:
    assert (
        _w4a16_fused_dsl_opt_level(
            size_m=size_m,
            top_k=8,
            element_dtype="bf16",
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=True,
            collect_activation_amax=False,
            intermediate_rotation=False,
        )
        == 3
    )


def test_unqualified_direct_m8_retains_o2() -> None:
    assert (
        _w4a16_fused_dsl_opt_level(
            size_m=8,
            top_k=8,
            element_dtype="bf16",
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=True,
            collect_activation_amax=False,
            intermediate_rotation=False,
        )
        == 2
    )


@pytest.mark.parametrize("size_m", (8, 16))
def test_unproven_small_route_packed_topk8_retains_o2(size_m: int) -> None:
    assert (
        _w4a16_fused_dsl_opt_level(
            size_m=size_m,
            top_k=8,
            element_dtype="bf16",
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=False,
            collect_activation_amax=False,
            intermediate_rotation=False,
        )
        == 2
    )


@pytest.mark.parametrize("size_m", (32, 64, 128, 256))
def test_proven_route_packed_fused_w4a16_uses_o3(size_m: int) -> None:
    assert (
        _w4a16_fused_dsl_opt_level(
            size_m=size_m,
            top_k=8,
            element_dtype="bf16",
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=False,
            collect_activation_amax=False,
            intermediate_rotation=False,
        )
        == 3
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"size_m": 257}, 2),
        ({"element_dtype": "fp16"}, 2),
        ({"weight_layout": "modelopt"}, 2),
        ({"weight_layout": "nf3_2p1"}, 2),
        ({"weight_layout": "trellis3_t256"}, 2),
        ({"scale_format": "e4m3_k32"}, 2),
        ({"collect_activation_amax": True}, 2),
        ({"intermediate_rotation": True}, 2),
    ),
)
def test_other_fused_w4a16_specializations_retain_o2(
    overrides: dict[str, object],
    expected: int,
) -> None:
    arguments: dict[str, object] = {
        "size_m": 256,
        "top_k": 8,
        "element_dtype": "bf16",
        "weight_layout": "packed",
        "scale_format": "e4m3_k16",
        "direct_topk_routes": False,
        "collect_activation_amax": False,
        "intermediate_rotation": False,
    }
    arguments.update(overrides)
    assert _w4a16_fused_dsl_opt_level(**arguments) == expected


@pytest.mark.parametrize("size_m", (1, 4, 32, 256))
@pytest.mark.parametrize("direct_topk_routes", (False, True))
def test_unqualified_top1_retains_o2(
    size_m: int,
    direct_topk_routes: bool,
) -> None:
    assert (
        _w4a16_fused_dsl_opt_level(
            size_m=size_m,
            top_k=1,
            element_dtype="bf16",
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=direct_topk_routes,
            collect_activation_amax=False,
            intermediate_rotation=False,
        )
        == 2
    )
