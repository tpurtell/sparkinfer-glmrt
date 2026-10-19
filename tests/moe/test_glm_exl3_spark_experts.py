"""Bitwise correctness, 1-64 row invariance, CTA-limit, and graph-replay gates
for GLM 5.3 EXL3 K4/K5 Spark routed experts against origin/master."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from benchmarks import benchmark_glm_exl3_spark_experts as bench
from ..conftest import require_b12x


def _control_dir() -> Path:
    env = os.environ.get("CUTEAFD_MXFP4_CONTROL_DIR")
    if env:
        return Path(env)
    default = Path("/workspace/cuteafd/results/control")
    if default.is_dir():
        return default
    pytest.skip("CUTEAFD_MXFP4_CONTROL_DIR not set and /workspace/cuteafd/results/control not found")


def _make_test_fixture(seed: int, device: torch.device, num_experts: int = 32):
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import build_tiered_maps, combine_trellis_rotations
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

    tier0_ids = tuple(range(0, num_experts - 8))
    tier1_ids = tuple(range(num_experts - 8, num_experts))
    gen = torch.Generator(device=device).manual_seed(seed)

    def scales(shape):
        return (0.875 + 0.25 * torch.rand(shape, generator=gen, device=device)).to(torch.float16)

    tiers = []
    for offset, (ids, bits) in enumerate(((tier0_ids, 4), (tier1_ids, 5))):
        e = len(ids)
        tiers.append(prepare_trellis256_moe_weights(
            hidden_size=bench.HIDDEN, intermediate_size=bench.INTERMEDIATE, num_experts=e,
            activation="silu", fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=seed + offset, params_dtype=torch.float16,
            w13_layout="trellis_t256_proj", trellis_bits=bits, codebook="mcg",
            gate_suh=scales((e, bench.HIDDEN)), up_suh=scales((e, bench.HIDDEN)),
            intermediate_rotations=scales((e, 3 * bench.INTERMEDIATE)),
            down_svh=scales((e, bench.HIDDEN)), tile_config=bench.DEFAULT_TILE,
        ))
    g2c, desc = build_tiered_maps(tier0_ids, tier1_ids, device=device)
    return tiers[0], tiers[1], g2c, desc, combine_trellis_rotations(*tiers), num_experts


def test_bench_capacity_mapping():
    assert [bench.capacity_for_rows(r) for r in (1, 4, 8, 16, 43, 64, 128)] == [1, 80, 80, 80, 80, 80, 256]


@pytest.mark.parametrize("capacity,rows_list", [
    (1, (1,)),
    (16, (1, 4, 7, 8, 16)),
    (80, (1, 4, 8, 16, 43, 64, 80)),
    (256, (1, 8, 43, 64, 128, 256)),
])
@pytest.mark.parametrize("hot", [0, 4, 8])
def test_glm_exl3_bitwise_identical_to_master(capacity, rows_list, hot):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    import b12x.moe._shared.kernels.w4a16.mixed_trellis as cand_mod

    ctrl_mod, _ = bench.load_control_modules(_control_dir())
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_smem = int(props.shared_memory_per_block_optin)

    t0, t1, g2c, desc, rot, num_experts = _make_test_fixture(400 + capacity + 10 * hot, device)
    slots = route_pack_capacity(capacity * bench.TOPK, 8, num_experts, topk=bench.TOPK)[1]
    common_kw = dict(
        size_m=capacity, hidden_size=bench.HIDDEN, intermediate_size=bench.INTERMEDIATE,
        tier0_num_experts=t0.num_experts, tier1_num_experts=t1.num_experts, top_k=bench.TOPK,
        route_num_experts=num_experts, max_m_blocks=(slots + 7) // 8, sms=sms,
        max_shared_mem=max_smem, force_tile_config=bench.DEFAULT_TILE,
        tier0_bits=4, tier1_bits=5, trellis_codebook="mcg", swiglu_limit=None,
        moe_block_size=8, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16",
        route_ids_dtype=torch.int32,
    )
    cand_launch = cand_mod.compile_mixed_trellis(**common_kw)
    ctrl_launch = ctrl_mod.compile_mixed_trellis(**common_kw)
    cand_buf = cand_mod.make_mixed_trellis_buffers(cand_launch, device=device, sms=sms)
    ctrl_buf = ctrl_mod.make_mixed_trellis_buffers(ctrl_launch, device=device, sms=sms)
    cand_bind = cand_mod.bind_mixed_trellis(t0, t1, g2c, desc, rot, cand_launch)
    ctrl_bind = ctrl_mod.bind_mixed_trellis(t0, t1, g2c, desc, rot, ctrl_launch)

    gen = torch.Generator(device=device).manual_seed(900 + capacity + 17 * hot)
    x_full = (torch.randn((capacity, bench.HIDDEN), device=device, generator=gen) * 0.0625).bfloat16().contiguous()
    scores = torch.rand((capacity, num_experts), device=device, generator=gen)
    if hot:
        scores[:, :hot] += 4.0
    ids_full = scores.topk(bench.TOPK, -1).indices.to(torch.int32).contiguous()
    if capacity > 4 and hot == 0:
        ids_full[-1, -1] = -1  # exercise invalid route sentinel skipping
    rw_full = torch.rand((capacity, bench.TOPK), device=device, generator=gen).contiguous()
    rw_full = (2.5 * rw_full / rw_full.sum(-1, keepdim=True)).to(torch.float32).contiguous()

    # Warmup all token count buckets used by Triton route pack before freezing resolution.
    for rows in rows_list:
        cand_mod.run_bound_mixed_trellis(x_full[:rows], rw_full[:rows], ids_full[:rows], cand_bind, cand_buf)
        ctrl_mod.run_bound_mixed_trellis(x_full[:rows], rw_full[:rows], ids_full[:rows], ctrl_bind, ctrl_buf)
    torch.cuda.synchronize()

    with kernel_resolution_guard("GLM EXL3 bitwise identity and graph replay"):
        for rows in rows_list:
            cand_buf.output.fill_(float("nan"))
            ctrl_buf.output.fill_(float("nan"))
            c_out = cand_mod.run_bound_mixed_trellis(x_full[:rows], rw_full[:rows], ids_full[:rows], cand_bind, cand_buf)
            m_out = ctrl_mod.run_bound_mixed_trellis(x_full[:rows], rw_full[:rows], ids_full[:rows], ctrl_bind, ctrl_buf)
            torch.cuda.synchronize()
            assert torch.isfinite(c_out).all() and torch.count_nonzero(c_out) > 0
            assert torch.equal(c_out.view(torch.int16), m_out.view(torch.int16)), (
                f"GLM EXL3 mismatch vs master at cap={capacity}, rows={rows}, hot={hot}"
            )
        last_rows = rows_list[-1]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cand_mod.run_bound_mixed_trellis(
                x_full[:last_rows], rw_full[:last_rows], ids_full[:last_rows], cand_bind, cand_buf
            )
        cand_buf.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cand_buf.output[:last_rows].view(torch.int16), ctrl_buf.output[:last_rows].view(torch.int16))


@pytest.mark.parametrize("same_experts", [False, True])
def test_glm_exl3_row_invariance_1_to_64(same_experts):
    """Bitwise row invariance across every row count 1..64."""
    require_b12x()
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    import b12x.moe._shared.kernels.w4a16.mixed_trellis as cand_mod

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_smem = int(props.shared_memory_per_block_optin)

    t0, t1, g2c, desc, rot, num_experts = _make_test_fixture(808 + int(same_experts), device)
    arms = {}
    for cap in (1, 80):
        slots = route_pack_capacity(cap * bench.TOPK, 8, num_experts, topk=bench.TOPK)[1]
        launch = cand_mod.compile_mixed_trellis(
            size_m=cap, hidden_size=bench.HIDDEN, intermediate_size=bench.INTERMEDIATE,
            tier0_num_experts=t0.num_experts, tier1_num_experts=t1.num_experts, top_k=bench.TOPK,
            route_num_experts=num_experts, max_m_blocks=(slots + 7) // 8, sms=sms,
            max_shared_mem=max_smem, force_tile_config=bench.DEFAULT_TILE,
            tier0_bits=4, tier1_bits=5, trellis_codebook="mcg", swiglu_limit=None,
            moe_block_size=8, rotation_input_dtype="bf16", full_rotation_output_dtype="bf16",
            route_ids_dtype=torch.int32,
        )
        buf = cand_mod.make_mixed_trellis_buffers(launch, device=device, sms=sms)
        bind = cand_mod.bind_mixed_trellis(t0, t1, g2c, desc, rot, launch)
        arms[cap] = (bind, buf)

    gen = torch.Generator(device=device).manual_seed(1707 + int(same_experts))
    x = (torch.randn((64, bench.HIDDEN), device=device, generator=gen) * 0.25).bfloat16().contiguous()
    ids = torch.rand((64, num_experts), device=device, generator=gen).topk(bench.TOPK, -1).indices.int().contiguous()
    if same_experts:
        ids[:] = ids[0].clone()
    scores = torch.rand((64, bench.TOPK), device=device, generator=gen)
    scores = (2.5 * scores / scores.sum(1, keepdim=True)).to(torch.float32).contiguous()

    def run(first, count, cap):
        bind, buf = arms[cap]
        xs = x[first:first + count].contiguous()
        ws = scores[first:first + count].contiguous()
        ks = ids[first:first + count].contiguous()
        return cand_mod.run_bound_mixed_trellis(xs, ws, ks, bind, buf).clone()

    serial = torch.cat([run(r, 1, 1) for r in range(64)], dim=0)
    torch.cuda.synchronize()
    for count in range(1, 65):
        got = run(0, count, 80)
        torch.cuda.synchronize()
        diff = int((got.view(torch.int16) != serial[:count].view(torch.int16)).sum())
        assert diff == 0, f"GLM EXL3 row invariance failed at M={count} (same={same_experts}): {diff} diffs"
