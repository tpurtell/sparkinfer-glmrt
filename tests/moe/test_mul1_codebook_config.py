from __future__ import annotations

import pytest

from b12x.moe._shared.trellis_codebooks import (
    CODEBOOKS,
    MCG,
    MUL1,
    MUL1_MULTIPLIER,
    normalize_codebook,
    validate_codebook_bits,
)
from b12x.moe.fused_moe.config import TrellisCodebook, TrellisConfig


def test_mul1_codebook_registry_and_bits_validation() -> None:
    assert MUL1 == "mul1"
    assert MUL1 in CODEBOOKS
    assert MUL1_MULTIPLIER == 0x83DCD12D
    assert normalize_codebook("mul1") == MUL1
    assert normalize_codebook("MUL1") == MUL1
    assert normalize_codebook(0x83DCD12D) == MUL1

    for bits in (2, 3, 4, 5, 6):
        validate_codebook_bits(MUL1, bits)
    for invalid_bits in (0, 1, 7):
        with pytest.raises(ValueError, match="mul1 is defined only for K2/K3/K4/K5/K6"):
            validate_codebook_bits(MUL1, invalid_bits)

    with pytest.raises(ValueError, match="mcg is defined only for K2/K3/K4/K5/K6"):
        validate_codebook_bits(MCG, 1)


def test_mul1_trellis_config_validates_on_cpu() -> None:
    raw = {
        "quant_method": "b12x_trellis",
        "b12x_trellis": {
            "version": 2,
            "codebook": "mul1",
            "rate": {"granularity": "per_expert_projection"},
            "scale": {
                "input_scales": {"vectors": "per_layer", "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": "per_layer", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        },
    }
    parsed = TrellisConfig.from_quantization_config(raw)
    assert parsed.codebook is TrellisCodebook.MUL1
    assert parsed.to_dict() == raw["b12x_trellis"]


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_w4a16_fused_moe_kernel_accepts_mul1_config_on_cpu(bits: int) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    common = dict(
        size_m=8,
        hidden_size=512,
        intermediate_size=256,
        num_experts=8,
        top_k=2,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        fc1_tile_n=128,
        fc1_tile_k=64,
        fc2_tile_n=128,
        fc2_tile_k=64,
        moe_block_size=8,
        max_m_blocks=8,
        element_dtype="fp16",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        intermediate_rotation=True,
        full_rotation=True,
        workspace_sms=48,
        schedule_whole_tiles=True,
    )
    mul1_kernel = W4A16FusedMoeKernel(**common, trellis_codebook="mul1")
    mcg_kernel = W4A16FusedMoeKernel(**common, trellis_codebook="mcg")
    assert mul1_kernel.trellis_codebook == "mul1"
    assert not mul1_kernel.sqg_xor_cheb_t12_smem
    assert mul1_kernel.__cache_key__ != mcg_kernel.__cache_key__


def test_mul1_rejects_k1_in_registry_and_w4a16_kernel() -> None:
    """MUL1 rejects K1 in both the codebook registry and W4A16 t256 kernel tiles (K2..K6 only)."""
    with pytest.raises(ValueError, match="mul1 is defined only for K2/K3/K4/K5/K6"):
        validate_codebook_bits(MUL1, 1)
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16GemmKernel

    with pytest.raises(ValueError, match="trellis_t256 bits must be one of"):
        W4A16GemmKernel(
            size_m=8,
            size_n=128,
            size_k=128,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=128,
            tile_k=64,
            moe_block_size=8,
            max_m_blocks=1,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="packed",
            trellis_bits=1,
            trellis_codebook="mul1",
        )


@pytest.mark.parametrize(
    ("codebook_arg", "bits"),
    [
        ("mul1", (2, 3)),
        ("mul1", (3, 4)),
        ("mul1", (4, 5)),
        ("mul1", (5, 6)),
        ("mul1", (2, 4, 6)),
        (MUL1_MULTIPLIER, (3, 4, 5)),
    ],
)
def test_mixed_trellis_format_accepts_mul1_and_separates_cache_keys_on_cpu(
    codebook_arg: str | int,
    bits: tuple[int, ...],
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        W4A16MixedTrellis3Kernel,
        W4A16MixedTrellisKernel,
        _normalize_mixed_trellis_format,
    )
    from b12x.moe._shared.kernels.w4a16.mixed_trellis_ws import W4A16MixedTrellisWSKernel

    normalized, normalized_bits = _normalize_mixed_trellis_format(codebook_arg, bits)
    assert normalized == "mul1"
    assert normalized_bits == bits

    for bad_bits, match in (
        ((1, 2), "mul1 is defined only for K2/K3/K4/K5/K6"),
        ((3, 3), "tiers must be distinct"),
        ((2, 3, 7), "mul1 is defined only for K2/K3/K4/K5/K6"),
    ):
        with pytest.raises(ValueError, match=match):
            _normalize_mixed_trellis_format("mul1", bad_bits)

    def make_moe(num_experts: int, tier_bits: int, codebook: str, block_m: int = 16) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=16,
            hidden_size=512,
            intermediate_size=256,
            num_experts=num_experts,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=128,
            fc1_tile_k=64,
            fc2_tile_n=128,
            fc2_tile_k=64,
            moe_block_size=block_m,
            max_m_blocks=8,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=tier_bits,
            trellis_codebook=codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            workspace_sms=48,
            schedule_whole_tiles=True,
        )

    if len(bits) == 2:
        mul1_k = W4A16MixedTrellisKernel(
            driver=make_moe(4, bits[0], "mul1"),
            tier0=make_moe(2, bits[0], "mul1"),
            tier1=make_moe(2, bits[1], "mul1"),
        )
        mcg_k = W4A16MixedTrellisKernel(
            driver=make_moe(4, bits[0], "mcg"),
            tier0=make_moe(2, bits[0], "mcg"),
            tier1=make_moe(2, bits[1], "mcg"),
        )
        assert mul1_k.__cache_key__ != mcg_k.__cache_key__

        mul1_ws = W4A16MixedTrellisWSKernel(
            driver=make_moe(4, bits[0], "mul1"),
            tier0=make_moe(2, bits[0], "mul1"),
            tier1=make_moe(2, bits[1], "mul1"),
            max_shared_mem=101376,
            prerotate=False,
        )
        mcg_ws = W4A16MixedTrellisWSKernel(
            driver=make_moe(4, bits[0], "mcg"),
            tier0=make_moe(2, bits[0], "mcg"),
            tier1=make_moe(2, bits[1], "mcg"),
            max_shared_mem=101376,
            prerotate=False,
        )
        assert mul1_ws.__cache_key__ != mcg_ws.__cache_key__
    else:
        mul1_k3 = W4A16MixedTrellis3Kernel(
            driver=make_moe(6, bits[0], "mul1"),
            tier0=make_moe(2, bits[0], "mul1"),
            tier1=make_moe(2, bits[1], "mul1"),
            tier2=make_moe(2, bits[2], "mul1"),
        )
        mcg_k3 = W4A16MixedTrellis3Kernel(
            driver=make_moe(6, bits[0], "mcg"),
            tier0=make_moe(2, bits[0], "mcg"),
            tier1=make_moe(2, bits[1], "mcg"),
            tier2=make_moe(2, bits[2], "mcg"),
        )
        assert mul1_k3.__cache_key__ != mcg_k3.__cache_key__


@pytest.mark.parametrize("tier1_bits", [2, 3, 5, 6])
@pytest.mark.parametrize("size_m", [1, 80])
@pytest.mark.parametrize("compact", [False, True])
def test_mixed_trellis_k4_decode_accepts_mul1_and_separates_cache_keys_on_cpu(
    tier1_bits: int,
    size_m: int,
    compact: bool,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import W4A16MixedTrellisKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis_k4 import W4A16MixedTrellisK4Decode

    def make_moe(num_experts: int, bits: int, codebook: str) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=512,
            intermediate_size=256,
            num_experts=num_experts,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=128,
            fc1_tile_k=64,
            fc2_tile_n=128,
            fc2_tile_k=64,
            moe_block_size=8,
            max_m_blocks=32,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            trellis_codebook=codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            workspace_sms=48,
            schedule_whole_tiles=True,
        )

    mul1_k4 = W4A16MixedTrellisK4Decode(
        driver=make_moe(4, 4, "mul1"),
        tier0=make_moe(2, 4, "mul1"),
        tier1=make_moe(2, tier1_bits, "mul1"),
        compact_weights=compact,
    )
    mcg_k4 = W4A16MixedTrellisK4Decode(
        driver=make_moe(4, 4, "mcg"),
        tier0=make_moe(2, 4, "mcg"),
        tier1=make_moe(2, tier1_bits, "mcg"),
        compact_weights=compact,
    )
    coop_mul1 = W4A16MixedTrellisKernel(
        driver=make_moe(4, 4, "mul1"),
        tier0=make_moe(2, 4, "mul1"),
        tier1=make_moe(2, tier1_bits, "mul1"),
        compact_weights=compact,
    )
    assert mul1_k4.__cache_key__ != mcg_k4.__cache_key__
    assert mul1_k4.__cache_key__ != coop_mul1.__cache_key__

    if tier1_bits == 3 and size_m == 1 and not compact:
        with pytest.raises(ValueError, match="W4A16MixedTrellisK4Decode"):
            W4A16MixedTrellisK4Decode(
                driver=make_moe(4, 4, "sqg_e4m3"),
                tier0=make_moe(2, 4, "sqg_e4m3"),
                tier1=make_moe(2, 3, "sqg_e4m3"),
            )


@pytest.mark.parametrize(
    ("hidden", "inter", "route_experts", "topk", "compact", "tier1_bits"),
    [
        (4608, 384, 384, 8, True, 5),
        (4096, 384, 352, 8, True, 6),
        (6144, 256, 256, 8, False, 5),
        (6144, 384, 256, 8, True, 5),
        (6144, 512, 256, 8, False, 5),
        (6144, 1024, 128, 4, False, 5),
        (4096, 768, 256, 8, True, 5),
    ],
)
def test_sm121_k4_decode_selector_dispatches_mul1_and_mcg_before_cuda(
    monkeypatch,
    hidden: int,
    inter: int,
    route_experts: int,
    topk: int,
    compact: bool,
    tier1_bits: int,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from types import SimpleNamespace
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mt_mod
    from b12x.moe._shared.kernels.w4a16.mixed_trellis_k4 import W4A16MixedTrellisK4Decode

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (12, 1))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_a, **_k: SimpleNamespace(
            multi_processor_count=48, shared_memory_per_block_optin=101376
        ),
    )

    class _StopBeforeCompile(Exception):
        pass

    selected: list[object] = []

    def capture(build, **_kwargs):
        selected.append(build(False))
        raise _StopBeforeCompile

    monkeypatch.setattr(mt_mod, "_select_mixed_fc2_kernel", capture)
    for cb in ("mul1", "mcg"):
        selected.clear()
        with pytest.raises(_StopBeforeCompile):
            mt_mod.compile_mixed_trellis(
                size_m=8,
                hidden_size=hidden,
                intermediate_size=inter,
                tier0_num_experts=2,
                tier1_num_experts=2,
                route_num_experts=route_experts,
                top_k=topk,
                max_m_blocks=64,
                sms=48,
                max_shared_mem=101376,
                force_tile_config=(64, 128, 64, 128),
                tier0_bits=4,
                tier1_bits=tier1_bits,
                trellis_codebook=cb,
                moe_block_size=8,
                rotation_input_dtype="bf16",
                full_rotation_output_dtype="bf16",
                compact_weights=compact,
                workspace_sms=48,
            )
        assert len(selected) == 1
        assert isinstance(selected[0], W4A16MixedTrellisK4Decode)
        assert selected[0].tier0.trellis_codebook == cb


@pytest.mark.parametrize(
    ("codebook_arg", "bits"),
    [
        ("mul1", (2, 3, 4, 5)),
        ("mul1", (3, 4, 5, 6)),
        ("mul1", (2, 4, 5, 6)),
        (MUL1_MULTIPLIER, (2, 3, 5, 6)),
    ],
)
def test_mixed_trellis4_format_accepts_mul1_and_separates_cache_keys_on_cpu(
    monkeypatch,
    codebook_arg: str | int,
    bits: tuple[int, int, int, int],
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis4
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis4 import (
        W4A16MixedTrellis4Kernel,
        _normalize_mixed_trellis_format,
    )

    normalized, normalized_bits = _normalize_mixed_trellis_format(codebook_arg, bits)
    assert normalized == "mul1"
    assert normalized_bits == bits

    for bad_bits, match in (
        ((1, 2, 3, 4), "mul1 is defined only for K2/K3/K4/K5/K6"),
        ((2, 3, 4, 7), "mul1 is defined only for K2/K3/K4/K5/K6"),
        ((2, 3, 4), "four distinct bitrates"),
        ((2, 3, 3, 4), "four distinct bitrates"),
    ):
        with pytest.raises(ValueError, match=match):
            _normalize_mixed_trellis_format("mul1", bad_bits)

    def make_moe(num_experts: int, tier_bits: int, codebook: str) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=16,
            hidden_size=512,
            intermediate_size=256,
            num_experts=num_experts,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=128,
            fc1_tile_k=64,
            fc2_tile_n=128,
            fc2_tile_k=64,
            moe_block_size=16,
            max_m_blocks=8,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=tier_bits,
            trellis_codebook=codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            workspace_sms=48,
            schedule_whole_tiles=True,
        )

    mul1_k4t = W4A16MixedTrellis4Kernel(
        driver=make_moe(8, bits[0], "mul1"),
        tier0=make_moe(2, bits[0], "mul1"),
        tier1=make_moe(2, bits[1], "mul1"),
        tier2=make_moe(2, bits[2], "mul1"),
        tier3=make_moe(2, bits[3], "mul1"),
    )
    mcg_k4t = W4A16MixedTrellis4Kernel(
        driver=make_moe(8, bits[0], "mcg"),
        tier0=make_moe(2, bits[0], "mcg"),
        tier1=make_moe(2, bits[1], "mcg"),
        tier2=make_moe(2, bits[2], "mcg"),
        tier3=make_moe(2, bits[3], "mcg"),
    )
    assert mul1_k4t.__cache_key__ != mcg_k4t.__cache_key__

    with pytest.raises(ValueError, match="disagree on trellis_codebook"):
        W4A16MixedTrellis4Kernel(
            driver=make_moe(8, bits[0], "mul1"),
            tier0=make_moe(2, bits[0], "mul1"),
            tier1=make_moe(2, bits[1], "mul1"),
            tier2=make_moe(2, bits[2], "mul1"),
            tier3=make_moe(2, bits[3], "mcg"),
        )

    class _StopBeforeCuda(Exception):
        pass

    captured: list[W4A16MixedTrellis4Kernel] = []

    def capture(build, **_kwargs):
        captured.append(build(False))
        raise _StopBeforeCuda

    monkeypatch.setattr(mixed_trellis4, "_select_mixed_fc2_kernel", capture)
    with pytest.raises(_StopBeforeCuda):
        mixed_trellis4.compile_mixed_trellis4(
            size_m=16,
            hidden_size=512,
            intermediate_size=256,
            tier0_num_experts=2,
            tier1_num_experts=2,
            tier2_num_experts=2,
            tier3_num_experts=2,
            route_num_experts=600,
            top_k=2,
            max_m_blocks=16,
            sms=48,
            max_shared_mem=101376,
            force_tile_config=(64, 128, 64, 128),
            tier0_bits=bits[0],
            tier1_bits=bits[1],
            tier2_bits=bits[2],
            tier3_bits=bits[3],
            trellis_codebook=codebook_arg,
            decode_schedule="gb10",
        )
    assert len(captured) == 1
    assert captured[0].descriptor_bits == 10
    for name in ("driver", "tier0", "tier1", "tier2", "tier3"):
        fused = getattr(captured[0], name)
        assert fused.trellis_codebook == "mul1"
        assert fused.fc1.l2_evict_first_b and fused.fc2.l2_evict_first_b


@pytest.mark.parametrize("tier1_bits", [3, 4, 5, 6])
@pytest.mark.parametrize("size_m", [1, 80])
def test_mixed_trellis_kimi_tp6_accepts_mul1_and_separates_cache_keys_on_cpu(
    tier1_bits: int,
    size_m: int,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import W4A16MixedTrellisKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis_kimi import W4A16MixedTrellisKimiTP6

    def make_moe(num_experts: int, bits: int, codebook: str) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=3584,
            intermediate_size=512,
            num_experts=num_experts,
            top_k=16,
            activation="situ",
            swiglu_limit=None,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=256,
            fc1_tile_k=64,
            fc2_tile_n=256,
            fc2_tile_k=64,
            moe_block_size=8,
            max_m_blocks=256,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            trellis_codebook=codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            workspace_sms=48,
            schedule_whole_tiles=True,
            l2_evict_first_b=True,
        )

    mul1_kimi = W4A16MixedTrellisKimiTP6(
        driver=make_moe(8, 2, "mul1"),
        tier0=make_moe(6, 2, "mul1"),
        tier1=make_moe(2, tier1_bits, "mul1"),
        descriptor_bits=10,
    )
    mcg_kimi = W4A16MixedTrellisKimiTP6(
        driver=make_moe(8, 2, "mcg"),
        tier0=make_moe(6, 2, "mcg"),
        tier1=make_moe(2, tier1_bits, "mcg"),
        descriptor_bits=10,
    )
    coop_mul1 = W4A16MixedTrellisKernel(
        driver=make_moe(8, 2, "mul1"),
        tier0=make_moe(6, 2, "mul1"),
        tier1=make_moe(2, tier1_bits, "mul1"),
        descriptor_bits=10,
    )
    assert mul1_kimi.max_span == (1 if size_m == 1 else 4)
    assert mul1_kimi.__cache_key__ != mcg_kimi.__cache_key__
    assert mul1_kimi.__cache_key__ != coop_mul1.__cache_key__

    if tier1_bits == 3 and size_m == 1:
        with pytest.raises(ValueError, match="W4A16MixedTrellisKimiTP6"):
            W4A16MixedTrellisKimiTP6(
                driver=make_moe(8, 2, "sqg_e4m3"),
                tier0=make_moe(6, 2, "sqg_e4m3"),
                tier1=make_moe(2, 3, "sqg_e4m3"),
                descriptor_bits=10,
            )


@pytest.mark.parametrize("size_m", [1, 80])
@pytest.mark.parametrize("codebook_arg", ["mul1", MUL1_MULTIPLIER, "mcg"])
def test_sm121_kimi_tp6_selector_dispatches_mul1_and_mcg_before_cuda(
    monkeypatch,
    size_m: int,
    codebook_arg: str | int,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from types import SimpleNamespace
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mt_mod
    from b12x.moe._shared.kernels.w4a16.mixed_trellis_kimi import W4A16MixedTrellisKimiTP6

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (12, 1))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_a, **_k: SimpleNamespace(
            multi_processor_count=48, shared_memory_per_block_optin=101376
        ),
    )

    class _StopBeforeCompile(Exception):
        pass

    selected: list[object] = []

    def capture(build, **_kwargs):
        selected.append(build(False))
        raise _StopBeforeCompile

    monkeypatch.setattr(mt_mod, "_select_mixed_fc2_kernel", capture)
    with pytest.raises(_StopBeforeCompile):
        mt_mod.compile_mixed_trellis(
            size_m=size_m,
            hidden_size=3584,
            intermediate_size=512,
            tier0_num_experts=6,
            tier1_num_experts=2,
            route_num_experts=896,
            top_k=16,
            max_m_blocks=256,
            sms=48,
            max_shared_mem=101376,
            force_tile_config=(64, 256, 64, 256),
            tier0_bits=2,
            tier1_bits=3,
            trellis_codebook=codebook_arg,
            activation="situ",
            swiglu_limit=None,
            moe_block_size=8,
            rotation_input_dtype="bf16",
            full_rotation_output_dtype="bf16",
            decode_schedule="gb10",
            workspace_sms=48,
        )
    assert len(selected) == 1
    assert isinstance(selected[0], W4A16MixedTrellisKimiTP6)
    expected_cb = "mcg" if codebook_arg == "mcg" else "mul1"
    assert selected[0].tier0.trellis_codebook == expected_cb


@pytest.mark.parametrize("bits", [(2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])
@pytest.mark.parametrize("input_format", ["bf16", "e4m3_k32"])
def test_ws_a8_accepts_mul1_and_separates_cache_keys_on_cpu(
    monkeypatch,
    bits: tuple[int, int],
    input_format: str,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mt_mod
    from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel
    from b12x.moe._shared.kernels.w4a16.mixed_trellis_ws import W4A16MixedTrellisWSKernel

    def make_moe(num_experts: int, tier_bits: int, codebook: str, a8: bool) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=64,
            hidden_size=512,
            intermediate_size=256,
            num_experts=num_experts,
            top_k=2,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=128,
            fc1_tile_k=64,
            fc2_tile_n=128,
            fc2_tile_k=64,
            moe_block_size=16,
            max_m_blocks=16,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=tier_bits,
            trellis_codebook=codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            workspace_sms=48,
            schedule_whole_tiles=True,
            activation_output_a8=a8,
        )

    mul1_a8 = W4A16MixedTrellisWSKernel(
        driver=make_moe(4, bits[0], "mul1", True),
        tier0=make_moe(2, bits[0], "mul1", True),
        tier1=make_moe(2, bits[1], "mul1", True),
        max_shared_mem=101376,
        input_format=input_format,
        activations="a8",
        prerotate=False,
    )
    mcg_a8 = W4A16MixedTrellisWSKernel(
        driver=make_moe(4, bits[0], "mcg", True),
        tier0=make_moe(2, bits[0], "mcg", True),
        tier1=make_moe(2, bits[1], "mcg", True),
        max_shared_mem=101376,
        input_format=input_format,
        activations="a8",
        prerotate=False,
    )
    mul1_a16 = W4A16MixedTrellisWSKernel(
        driver=make_moe(4, bits[0], "mul1", False),
        tier0=make_moe(2, bits[0], "mul1", False),
        tier1=make_moe(2, bits[1], "mul1", False),
        max_shared_mem=101376,
        input_format=input_format,
        activations="a16",
        prerotate=False,
    )
    assert mul1_a8.ws_a8
    assert mul1_a8.__cache_key__ != mcg_a8.__cache_key__
    assert mul1_a8.__cache_key__ != mul1_a16.__cache_key__

    if bits == (3, 4) and input_format == "bf16":
        with pytest.raises(ValueError, match="A8 activation output requires MCG or MUL1"):
            make_moe(2, 3, "sqg_e4m3", True)
        with pytest.raises(ValueError, match="requires the MCG codebook"):
            mt_mod.compile_mixed_trellis(
                size_m=64,
                hidden_size=512,
                intermediate_size=256,
                tier0_num_experts=2,
                tier1_num_experts=2,
                top_k=2,
                max_m_blocks=16,
                sms=48,
                max_shared_mem=101376,
                force_tile_config=(64, 128, 64, 128),
                moe_block_size=16,
                trellis_codebook="sqg_e4m3",
                warp_specialized=True,
                activations="a8",
            )

    class _StopBeforeCuda(Exception):
        pass

    captured: list[W4A16MixedTrellisWSKernel] = []
    orig_ws = W4A16MixedTrellisWSKernel

    def capture_ws(**kwargs):
        captured.append(orig_ws(**kwargs))
        raise _StopBeforeCuda

    import b12x.moe._shared.kernels.w4a16.mixed_trellis_ws as mt_ws_mod
    monkeypatch.setattr(mt_ws_mod, "W4A16MixedTrellisWSKernel", capture_ws)
    for cb_arg in ("mul1", MUL1_MULTIPLIER, "mcg"):
        captured.clear()
        with pytest.raises(_StopBeforeCuda):
            mt_mod.compile_mixed_trellis(
                size_m=64,
                hidden_size=512,
                intermediate_size=384 if input_format == "e4m3_k32" else 256,
                tier0_num_experts=2,
                tier1_num_experts=2,
                top_k=2,
                max_m_blocks=16,
                sms=48,
                max_shared_mem=101376,
                force_tile_config=(64, 192, 64, 256) if input_format == "e4m3_k32" else (64, 128, 64, 128),
                tier0_bits=bits[0],
                tier1_bits=bits[1],
                trellis_codebook=cb_arg,
                moe_block_size=16,
                warp_specialized=True,
                input_format=input_format,
                activations="a8",
                workspace_sms=48,
            )
        assert len(captured) == 1
        assert isinstance(captured[0], W4A16MixedTrellisWSKernel)
        assert captured[0].ws_a8
        expected_cb = "mcg" if cb_arg == "mcg" else "mul1"
        assert captured[0].driver.trellis_codebook == expected_cb
        assert captured[0].driver.activation_output_a8


def test_resolve_exl3_checkpoint_codebook_markers_on_cpu() -> None:
    """Verify checkpoint-defined `.mcg` (0xCBAC1FED) and `.mul1` (0x83DCD12D) marker resolution."""
    import torch
    from b12x.moe._shared.kernels.w4a16.prepare import (
        resolve_exl3_checkpoint_codebook,
    )

    mcg_i32 = torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32)
    mcg_1d = mcg_i32.reshape(1)
    mul1_i32 = torch.tensor(MUL1_MULTIPLIER, dtype=torch.uint32).view(torch.int32)
    mul1_1d = mul1_i32.reshape(1)
    mul1_u32 = torch.tensor(MUL1_MULTIPLIER, dtype=torch.uint32)

    assert resolve_exl3_checkpoint_codebook(mcg=0xCBAC1FED) == "mcg"
    assert resolve_exl3_checkpoint_codebook(mcg=mcg_i32) == "mcg"
    assert resolve_exl3_checkpoint_codebook(mcg=mcg_1d, codebook="mcg") == "mcg"
    assert resolve_exl3_checkpoint_codebook(mul1=MUL1_MULTIPLIER) == "mul1"
    assert resolve_exl3_checkpoint_codebook(mul1=mul1_i32) == "mul1"
    assert resolve_exl3_checkpoint_codebook(mul1=mul1_1d) == "mul1"
    assert resolve_exl3_checkpoint_codebook(mul1=mul1_u32, codebook="mul1") == "mul1"
    assert resolve_exl3_checkpoint_codebook(codebook="mul1") == "mul1"
    assert resolve_exl3_checkpoint_codebook() == "mcg"

    with pytest.raises(ValueError, match="multiple EXL3 codebook markers"):
        resolve_exl3_checkpoint_codebook(mcg=mcg_i32, mul1=mul1_i32)
    with pytest.raises(ValueError, match="unexpected EXL3 MUL1 marker"):
        resolve_exl3_checkpoint_codebook(mul1=0xCBAC1FED)
    with pytest.raises(ValueError, match="unexpected EXL3 MCG marker"):
        resolve_exl3_checkpoint_codebook(mcg=MUL1_MULTIPLIER)
    with pytest.raises(ValueError, match="disagrees"):
        resolve_exl3_checkpoint_codebook(mul1=mul1_i32, codebook="mcg")
    with pytest.raises(ValueError, match="disagrees"):
        resolve_exl3_checkpoint_codebook(mcg=mcg_i32, codebook="mul1")
    with pytest.raises(ValueError, match="requires the checkpoint MCG or MUL1 marker"):
        resolve_exl3_checkpoint_codebook(codebook="mul1", require_marker=True)
    with pytest.raises(TypeError, match="scalar tensor or int"):
        resolve_exl3_checkpoint_codebook(mul1=True)
    with pytest.raises(ValueError, match="scalar or 1-element"):
        resolve_exl3_checkpoint_codebook(
            mul1=torch.tensor([MUL1_MULTIPLIER, MUL1_MULTIPLIER], dtype=torch.uint32)
        )
    with pytest.raises(ValueError, match="int32/uint32"):
        resolve_exl3_checkpoint_codebook(
            mul1=torch.tensor(1.0, dtype=torch.float32)
        )


def test_exl3_uniform_and_projection_mixed_planning_accepts_mul1_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Exl3TrellisSource, Exl3TrellisWeights, TrellisConfig, and TPMoEScratchPlan for MUL1."""
    import torch
    from b12x.moe import fused_moe
    import b12x.moe.fused_moe._impl as fused_moe_impl
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 48)

    for bits in (2, 3, 4, 5, 6):
        src = fused_moe.Exl3TrellisSource(
            bits=bits, tile_config=(64, 128, 64, 128), codebook="mul1"
        )
        assert src.codebook is TrellisCodebook.MUL1
        assert src.bits == bits

    with pytest.raises(ValueError, match="K2, K3, K4, K5, or K6"):
        fused_moe.Exl3TrellisSource(bits=1, codebook="mul1")
    with pytest.raises(ValueError, match="K3, K4, K5, or K6"):
        fused_moe.Exl3TrellisSource(bits=2)
    with pytest.raises(ValueError, match="'mcg' or 'mul1'"):
        fused_moe.Exl3TrellisSource(bits=3, codebook="sqg_e4m3")

    dummy_t = torch.zeros((1,), dtype=torch.int16)
    dummy_f = torch.ones((1,), dtype=torch.float16)
    w_mul1 = fused_moe.Exl3TrellisWeights(
        w13=dummy_t,
        w2=dummy_t,
        gate_suh=dummy_f,
        up_suh=dummy_f,
        intermediate_rotations=dummy_f,
        down_svh=dummy_f,
        mul1=MUL1_MULTIPLIER,
    )
    assert w_mul1.mul1 == MUL1_MULTIPLIER
    assert w_mul1.mcg is None

    with pytest.raises(ValueError, match="either mcg or mul1"):
        fused_moe.Exl3TrellisWeights(
            w13=dummy_t,
            w2=dummy_t,
            gate_suh=dummy_f,
            up_suh=dummy_f,
            intermediate_rotations=dummy_f,
            down_svh=dummy_f,
        )
    with pytest.raises(ValueError, match="cannot declare both"):
        fused_moe.Exl3TrellisWeights(
            w13=dummy_t,
            w2=dummy_t,
            gate_suh=dummy_f,
            up_suh=dummy_f,
            intermediate_rotations=dummy_f,
            down_svh=dummy_f,
            mcg=0xCBAC1FED,
            mul1=MUL1_MULTIPLIER,
        )

    act = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A16,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geom = fused_moe.MoEGeometry(
        num_experts=8, hidden_size=256, intermediate_size=256
    )

    uniform_plan = fused_moe.plan_weights(
        source=fused_moe.Exl3TrellisSource(
            bits=4, tile_config=(64, 128, 64, 128), codebook="mul1"
        ),
        activation=act,
        geometry=geom,
    )
    assert uniform_plan._impl.trellis_codebook == "mul1"
    assert uniform_plan._impl.trellis_bits == 4

    v2_proj_cfg = TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "mul1",
            "rate": {"granularity": "per_expert_projection"},
            "scale": {
                "input_scales": {"vectors": "per_layer", "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": "per_layer", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        }
    )
    v2_proj_plan = fused_moe.plan_weights(
        source=v2_proj_cfg, activation=act, geometry=geom
    )
    assert v2_proj_plan._impl.trellis_codebook == "mul1"
    assert v2_proj_plan._impl.trellis_tile_config == (128, 128, 128, 128)
    assert fused_moe_impl._uses_projection_mixed_trellis(v2_proj_plan._impl)

    trellis_decode_cfg = MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode="packed",
    )
    scratch_v2 = fused_moe_impl.plan_tp_moe_scratch(
        fused_moe_impl.TPMoEScratchCaps(
            decode_config=trellis_decode_cfg,
            max_tokens=16,
            num_topk=4,
            route_num_experts=8,
            device="cpu",
            weight_plan=v2_proj_plan._impl,
            quant_mode="w4a16",
        )
    )
    assert scratch_v2._core_workspace_plan.implementation == "trellis_mixed3"
    assert scratch_v2._core_workspace_plan.projection_mixed_trellis
    assert scratch_v2._core_workspace_plan.trellis_codebook == "mul1"

    direct_exl3_plan = fused_moe_impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=16,
        hidden_size=512,
        intermediate_size=256,
        w13_layout="trellis_t256_proj",
        w4a16_layout="trellis_native",
        trellis_bits=3,
        trellis_codebook="mul1",
        trellis_rate_granularity="per_expert_projection",
    )
    scratch_exl3 = fused_moe_impl.plan_tp_moe_scratch(
        fused_moe_impl.TPMoEScratchCaps(
            decode_config=trellis_decode_cfg,
            max_tokens=16,
            num_topk=4,
            route_num_experts=16,
            device="cpu",
            weight_plan=direct_exl3_plan,
            quant_mode="w4a16",
        )
    )
    assert scratch_exl3._core_workspace_plan.implementation == "trellis_mixed"
    assert scratch_exl3._core_workspace_plan.projection_mixed_trellis
    assert scratch_exl3._core_workspace_plan.trellis_codebook == "mul1"


def test_btx_compat_lifts_accept_mul1_and_preserve_sqg_default_on_cpu() -> None:
    """Verify btx_compat lifts preserve sqg_e4m3 default and populate codebook_seed for mul1/mcg."""
    import torch
    from b12x.moe._shared.btx_schema import matrix_atom_bytes
    from b12x.moe._shared.kernels.w4a16.btx_compat import (
        lift_qsrt_atoms_v1_extent,
        lift_qsrt_atoms_v2_extent,
    )

    hidden, global_i, experts = 256, 3072, 3
    mbytes = matrix_atom_bytes(hidden, 3, 3)
    v1_bundle = 3 * mbytes + 3 * 64
    v1_payload = torch.zeros((8, experts, v1_bundle), dtype=torch.uint8)
    side = torch.ones((hidden,), dtype=torch.float16)
    expert_ids = torch.arange(experts, dtype=torch.int32)
    format_codes = torch.tensor([0x10, 0x01, 0x00], dtype=torch.uint8)

    default_v1 = lift_qsrt_atoms_v1_extent(
        v1_payload,
        first_atom_slot=0,
        layer_index=0,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        gate_suh=side,
        up_suh=side,
        down_svh=side,
    )
    assert default_v1.manifest.codebook == "sqg_e4m3"
    assert default_v1.manifest.codebook_seed is None

    mul1_v1 = lift_qsrt_atoms_v1_extent(
        v1_payload,
        first_atom_slot=0,
        layer_index=0,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        gate_suh=side,
        up_suh=side,
        down_svh=side,
        codebook="mul1",
    )
    assert mul1_v1.manifest.codebook == "mul1"
    assert mul1_v1.manifest.codebook_seed == MUL1_MULTIPLIER

    mcg_v1 = lift_qsrt_atoms_v1_extent(
        v1_payload,
        first_atom_slot=0,
        layer_index=0,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        gate_suh=side,
        up_suh=side,
        down_svh=side,
        codebook="mcg",
    )
    assert mcg_v1.manifest.codebook == "mcg"
    assert mcg_v1.manifest.codebook_seed == 0xCBAC1FED

    p33_bytes = 3 * matrix_atom_bytes(hidden, 3, 3) + 3 * 64
    p43_bytes = 3 * matrix_atom_bytes(hidden, 4, 3) + 3 * 64
    # At layer 0, first_slot 0: expert 0 has base_pair=0 (P43), experts 1,2 have P33.
    v2_row_bytes = 2 * p33_bytes + 1 * p43_bytes
    v2_payload = torch.zeros((8, v2_row_bytes), dtype=torch.uint8)
    mul1_v2 = lift_qsrt_atoms_v2_extent(
        v2_payload,
        profile="k3x22_k4x2",
        first_atom_slot=0,
        layer_index=0,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        num_experts=experts,
        gate_suh=side,
        up_suh=side,
        down_svh=side,
        codebook="mul1",
    )
    assert mul1_v2.manifest.codebook == "mul1"
    assert mul1_v2.manifest.codebook_seed == MUL1_MULTIPLIER
    assert mul1_v2.manifest.rates.pair_kinds == frozenset({"P33", "P43"})


def test_dense_trellis256_prepare_and_dense_lut_contract_reject_mul1() -> None:
    """Dense trellis256 preparation and dense kernel runner keep {MCG, SQG_E4M3, SQG_FP16} only."""
    import inspect
    from types import SimpleNamespace

    pytest.importorskip("torch")
    import torch

    from b12x.moe._shared.kernels.w4a16.prepare import (
        _TRELLIS256_DENSE_CODEBOOKS,
        _trellis256_marker_codebook,
        prepare_trellis256_dense_weight,
        prepare_trellis256_pair_dense_weight,
    )

    assert set(_TRELLIS256_DENSE_CODEBOOKS) == {"mcg", "sqg_e4m3", "sqg_fp16"}
    assert _trellis256_marker_codebook(mcg=0xCBAC1FED, mul1_e4m3=None, codebook=None) == "mcg"
    assert _trellis256_marker_codebook(mcg=None, mul1_e4m3=None, codebook="sqg_e4m3") == "sqg_e4m3"
    assert _trellis256_marker_codebook(mcg=None, mul1_e4m3=None, codebook="sqg_fp16") == "sqg_fp16"

    for bad_mcg in (
        MUL1_MULTIPLIER,
        torch.tensor(MUL1_MULTIPLIER, dtype=torch.uint32),
        torch.tensor(MUL1_MULTIPLIER - (1 << 32), dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="expected MCG 0xcbac1fed"):
            _trellis256_marker_codebook(mcg=bad_mcg, mul1_e4m3=None, codebook=None)

    for bad_codebook in ("mul1", "MUL1", MUL1_MULTIPLIER):
        with pytest.raises(ValueError, match="unsupported trellis codebook"):
            _trellis256_marker_codebook(mcg=None, mul1_e4m3=None, codebook=bad_codebook)  # type: ignore[arg-type]

    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

    assert w4a16_kernel._TRELLIS256_DENSE_CODEBOOKS == {"mcg", "sqg_e4m3", "sqg_fp16"}
    dense_src = inspect.getsource(w4a16_kernel._run_trellis256_dense_current_device)
    assert 'if trellis_codebook != "mcg" and execution_lut is None:' in dense_src
    assert 'dummy_i32 if trellis_codebook == "mcg" else execution_lut,' in dense_src
    assert "MUL1" not in dense_src

    with pytest.raises(ValueError, match="dense_route_fast_path trellis_t256 codebook"):
        w4a16_kernel.W4A16GemmKernel(
            size_m=8,
            size_n=128,
            size_k=128,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=128,
            tile_k=64,
            moe_block_size=8,
            max_m_blocks=1,
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="packed",
            trellis_bits=4,
            trellis_codebook="mul1",
            dense_route_fast_path=True,
        )

    fake_dense = SimpleNamespace(
        weight_layout="trellis_t256",
        num_experts=1,
        trellis_codebook="mul1",
    )
    with pytest.raises(NotImplementedError, match="run_trellis256_dense has no decoder for codebook 'mul1'"):
        w4a16_kernel._run_trellis256_dense_current_device(
            torch.empty((1, 128)),
            fake_dense,
            launch=None,  # type: ignore[arg-type]
            execution_lut=None,
            grid_cap=1,
            hadamard_128=lambda *_: None,
        )

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        trellis = torch.zeros((16, 16, 64), dtype=torch.int16, device=device)
        suh = torch.ones((256,), dtype=torch.float16, device=device)
        svh = torch.ones((256,), dtype=torch.float16, device=device)
        for kwargs in (
            {"codebook": "mul1"},
            {"mcg": torch.tensor(MUL1_MULTIPLIER, dtype=torch.uint32, device=device)},
            {"mcg": MUL1_MULTIPLIER},
        ):
            with pytest.raises(ValueError, match="unsupported trellis codebook|expected MCG 0xcbac1fed"):
                prepare_trellis256_dense_weight(trellis, suh, svh, **kwargs)  # type: ignore[arg-type]
        pair_payload = torch.zeros((16, 16, 48), dtype=torch.int16, device=device).reshape(-1)
        for kwargs in (
            {"codebook": "mul1"},
            {"mcg": MUL1_MULTIPLIER, "codebook": None},
            {"mcg": torch.tensor(MUL1_MULTIPLIER, dtype=torch.uint32, device=device), "codebook": None},
        ):
            with pytest.raises(ValueError, match="unsupported trellis codebook|expected MCG 0xcbac1fed"):
                prepare_trellis256_pair_dense_weight(
                    pair_payload, suh, svh, pair_kind="P33", rate_axis="n", **kwargs
                )


def test_typed_v2_trellis_config_uniform_k4_propagates_bits_into_weight_and_execution_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify typed v2 TrellisConfig(uniform, mul1) rate 0x44 propagates K4 into WeightPlan & compile payload."""
    from types import SimpleNamespace
    import torch
    from b12x.moe import fused_moe
    import b12x.moe.fused_moe._impl as fused_moe_impl
    import b12x.moe.fused_moe._preparation as fused_moe_prep
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 48)
    v2_cfg = TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "mul1",
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {"vectors": "per_expert", "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": "per_expert", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        }
    )
    w_plan = fused_moe.plan_weights(
        source=v2_cfg,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=4, hidden_size=256, intermediate_size=256
        ),
    )

    dummy = torch.zeros((1,), dtype=torch.int32)
    fake_prepared_k4 = SimpleNamespace(
        w13=dummy,
        w13_scale=dummy,
        w13_global_scale=torch.ones((4,), dtype=torch.float32),
        w2=dummy,
        w2_scale=dummy,
        w2_global_scale=torch.ones((4,), dtype=torch.float32),
        trellis=SimpleNamespace(bits=4, codebook="mul1"),
    )
    monkeypatch.setattr(
        "b12x.moe.fused_moe.trellis.prepare_trellis_weights",
        lambda *_a, **_k: fake_prepared_k4,
    )

    weights = fused_moe.TrellisWeights(
        atoms=torch.zeros((8, 128), dtype=torch.uint8),
        rate=torch.tensor([0x44], dtype=torch.uint8),
        input_scales=fused_moe.ScaleFactors(vectors=torch.ones((4, 2, 256), dtype=torch.float16)),
        intermediate_scales=fused_moe.ScaleFactors(vectors=torch.ones((4, 3, 256), dtype=torch.float16)),
        output_scales=fused_moe.ScaleFactors(vectors=torch.ones((4, 256), dtype=torch.float16)),
    )
    experts = fused_moe.prepare_weights(plan=w_plan, weights=weights)
    assert experts.plan._impl.trellis_bits == 4
    assert experts._impl.plan.trellis_bits == 4

    payload = fused_moe_prep._weight_payload(experts)
    assert payload["trellis_bits"] == 4
    assert payload["trellis_codebook"] == "mul1"

    scratch = fused_moe_impl.plan_tp_moe_scratch(
        fused_moe_impl.TPMoEScratchCaps(
            decode_config=MoeDecodeConfig(
                backend="w4a16",
                route_planner="internal",
                max_active_clusters=None,
                w4a16_route_mode="packed",
            ),
            max_tokens=16,
            num_topk=2,
            route_num_experts=4,
            device="cpu",
            weight_plan=experts.plan._impl,
            quant_mode="w4a16",
        ),
        prewarm_launches=False,
    )
    assert scratch._core_workspace_plan.trellis_bits == 4
    assert scratch._core_workspace_plan.trellis_codebook == "mul1"






