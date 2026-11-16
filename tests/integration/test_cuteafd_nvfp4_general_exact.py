"""General exact-width NVFP4 W4A4 and dynamic-amax oracles (zero stored padding).

Covers:
  1. Peeled K_block=256 step-count invariants (e.g. 576=2x256+64 -> 9 exact m16n8k64 steps,
     624 -> 10, 688 -> 11, 160 -> 3) and standalone exact-grid F64 checks across K % 16 == 0
     tails (16..1040), partial N % 8 == 0 / I % 16 == 0 shards, wide & sparse tiles,
     multi-expert N-boundary isolation, and 0xFF dead-row scratch sentinels.
  2. Bit-for-bit identity between fused StreamNvfp4GateUpA4 + sparse tiles and separate
     StreamNvfp4LinearA4 + ChunkSwiGLUA4 across odd 16*k and 32*k slices.
  3. Cross-host byte-identical integer-PRNG dyadic operands across all NVFP4 families
     (glmf_nvfp4, glm_nvfp4, qwen4_nvfp4, minimax_nvfp4a4, inkling2_nvfp4) at exact unpadded
     TP shard widths plus aligned widths and synthetic 560, 592, 1040 against independent
     F64 references, plus ExportedProgram C-ABI parity and CUDA graph replay at rows
     1, 7, 80, 2752, 4096 under capacity=4096.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import os
import pytest
import torch

from ..conftest import require_b12x
from tests.moe.segment_aot import ExportedProgram

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
except ImportError:
    cuda = cutlass = cute = Int32 = None

E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def exact_slice_geometry(base, width: int):
    width = int(width)

    @dataclass(frozen=True)
    class ExactGeometry(type(base)):
        @property
        def slice(self) -> int:
            return width

    return ExactGeometry(**{f.name: getattr(base, f.name) for f in fields(base)})


def _splitmix_u32(shape: tuple[int, ...], seed: int, device: str = "cuda") -> torch.Tensor:
    """Architecture-independent 32-bit integer stream (identical bytes on x86_64 and aarch64)."""
    n = 1
    for d in shape:
        n *= int(d)
    s64 = (int(seed) * 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF
    if s64 >= (1 << 63):
        s64 -= 1 << 64
    idx = torch.arange(n, dtype=torch.int64, device=device)
    # Pure wrapping int64 MurmurHash3 / SplitMix finalizer: exact two's-complement bit ops.
    x = idx + s64
    x = (x ^ ((x >> 30) & 0x3FFFFFFFF)) * (-4658895280553007687)  # 0xBF58476D1CE4E5B9
    x = (x ^ ((x >> 27) & 0x1FFFFFFFFF)) * (-7723592293110705685)  # 0x94D049BB133111EB
    x = x ^ ((x >> 31) & 0x1FFFFFFFF)
    return (x & 0xFFFFFFFF).to(torch.int64).reshape(shape)


def _dyadic_bf16(shape: tuple[int, ...], seed: int, scale_pow2: float = 0.0625, device: str = "cuda") -> torch.Tensor:
    """Exact dyadic BF16 values k * 2^-p for k in [-63, 64]: zero rounding difference across hosts."""
    u = _splitmix_u32(shape, seed, device=device)
    k = (u & 0x7F) - 63  # [-63, 64], <= 7 significant bits -> exact in BF16/FP32/FP64
    return (k.to(torch.float32) * float(scale_pow2)).to(torch.bfloat16).contiguous()


def _deterministic_routes(rows: int, experts: int, top_k: int, seed: int, hot: bool = True, device: str = "cuda"):
    """Architecture-independent distinct expert ids [rows, top_k] and dyadic route weights."""
    u_row = _splitmix_u32((rows,), seed, device=device)
    # Pick a stride coprime to all expert counts (128, 256, 288, 352, 384, 512 -> odd primes > 11).
    stride = 13
    slots = torch.arange(top_k, dtype=torch.int64, device=device)[None, :]
    if hot and experts >= 16:
        # First half of rows concentrate on experts 0..11 (fires wide >=17 chunks), rest spread (fires sparse <=16).
        pool = torch.where(
            (torch.arange(rows, device=device)[:, None] < (rows * 3) // 4),
            (u_row[:, None] % 6 + slots * 2) % min(experts, 16),
            (u_row[:, None] + slots * stride) % experts,
        )
    else:
        pool = (u_row[:, None] + slots * stride) % experts
    ids = pool.to(torch.int32).contiguous()
    wu = (_splitmix_u32((rows, top_k), seed + 1, device=device) & 0x7) + 1  # 1..8
    weights = (wu.to(torch.float32) * (1.0 / 32.0)).contiguous()
    return ids, weights


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(memoryview(t.detach().contiguous().view(torch.uint8).cpu().numpy()))
    return h.hexdigest()


def _decode_nvfp4_f64(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(E2M1_POS + tuple(-v for v in E2M1_POS), dtype=torch.float64, device=packed.device)
    p = packed.long()
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).flatten(-2)
    sc = scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=-1)
    return table[codes] * sc


def _make_calibrated_weights(g, seed: int = 17):
    h, i, e = g.hidden, g.slice, g.experts
    operands = []
    for idx, (rows, cols) in enumerate(((i, h), (i, h), (h, i))):
        base_seed = seed * 1009 + idx * 97
        w = (_splitmix_u32((e, rows, cols // 2), base_seed) & 0xFF).to(torch.uint8).contiguous()
        su = _splitmix_u32((e, rows, cols // 16), base_seed + 1)
        exp = 4 + (su % 6)  # exponent 4..9 (finite E4M3 powers)
        man = (su >> 4) & 0x7
        s = (exp * 8 + man).to(torch.uint8)
        au = (_splitmix_u32((e,), base_seed + 2) & 0x7) + 1  # 1..8
        alpha = (au.to(torch.float32) * (1.0 / 2048.0)).contiguous()
        # Exact power-of-two input scales so 1/inscale is also exact in IEEE-754 FP32!
        in_pow = torch.pow(2.0, ((torch.arange(e, device="cuda", dtype=torch.int64) % 5) - 9).to(torch.float32))
        if idx == 2:
            in_pow = in_pow * 16.0
        inscale = in_pow.contiguous()
        if g.scale_convention == "divisor":
            alpha, inscale = alpha.reciprocal(), inscale.reciprocal()
        s_full = torch.cat([s.flatten(), alpha.view(torch.uint8), inscale.view(torch.uint8)]).contiguous()
        operands.extend([w, s_full])
    return tuple(operands)


def _make_dynamic_weights_fast(g, seed: int = 29):
    """Deterministic cross-host byte-identical dynamic-amax NVFP4 weights."""
    h, i, e = g.hidden, g.slice, g.experts
    denom_val = 2688.0
    denom_t = torch.tensor([denom_val], dtype=torch.float32, device="cuda")
    operands = []
    metas = []
    for idx, (rows, cols) in enumerate(((i, h), (i, h), (h, i))):
        base_seed = seed * 1009 + idx * 97
        w = (_splitmix_u32((e, rows, cols // 2), base_seed) & 0xFF).to(torch.uint8).contiguous()
        su = _splitmix_u32((e, rows, cols // 16), base_seed + 1)
        exp = 5 + (su % 5)  # 5..9
        man = (su >> 4) & 0x7
        s = (exp * 8 + man).to(torch.uint8)
        au = (_splitmix_u32((e,), base_seed + 2) & 0x7) + 4  # 4..11 exact integer F32
        amax = au.to(torch.float32).contiguous()
        s_full = torch.cat([s.flatten(), amax.view(torch.uint8), denom_t.view(torch.uint8)]).contiguous()
        operands.extend([w, s_full])
        metas.append((w, s, amax, denom_val))
    return tuple(operands), tuple(metas)


def _calibrated_f64_reference_gpu(g, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                                  w1: torch.Tensor, s1: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor,
                                  w2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    from .test_cuteafd_nvfp4_moe_aot import input_scales, quantize_nvfp4, split_scales

    h, i, e = g.hidden, g.slice, g.experts
    (q1, a1), (q3, a3), (q2, a2) = (
        split_scales(s1, e, i, h),
        split_scales(s3, e, i, h),
        split_scales(s2, e, h, i),
    )
    in1, in2 = input_scales(s1, e, i, h), input_scales(s2, e, h, i)
    limit = g.swiglu_limit
    routes = torch.zeros((*ids.shape, h), device=x.device, dtype=torch.float32)

    def project(a_d: torch.Tensor, w_d: torch.Tensor, alpha: torch.Tensor, inscale: torch.Tensor) -> torch.Tensor:
        acc = (a_d @ w_d.T).float()
        if g.scale_convention == "divisor":
            res = (acc / alpha) / inscale
        else:
            res = acc * (alpha * inscale)
        return res.bfloat16().double()

    for expert in ids.unique().tolist():
        if expert < 0 or expert >= e:
            continue
        r_idx, s_idx = torch.where(ids == expert)
        xr = quantize_nvfp4(
            x[r_idx].float(), float(in1[expert]), include_input_scale=False, scale_convention=g.scale_convention
        ).double()
        gate = project(xr, _decode_nvfp4_f64(w1[expert], q1[expert]), a1[expert], in1[expert])
        up = project(xr, _decode_nvfp4_f64(w3[expert], q3[expert]), a3[expert], in1[expert])
        if g.activation == "swigluoai":
            from b12x.moe._shared.kernels.reference import _apply_gated_activation

            act = _apply_gated_activation(
                gate.float(), up.float(), activation="swigluoai_uninterleave",
                swiglu_limit=limit, swiglu_alpha=g.swiglu_alpha, swiglu_beta=g.swiglu_beta,
            ).bfloat16().double()
        elif g.activation == "silu_post_clamp_fp32_product":
            silu = torch.nn.functional.silu(gate.float())
            up32 = up.float()
            if limit > 0:
                silu, up32 = silu.clamp(max=limit), up32.clamp(-limit, limit)
            act = (silu * up32).bfloat16().double()
        else:
            if limit > 0:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            act = (torch.nn.functional.silu(gate.float()).bfloat16().double() * up).bfloat16().double()
        act_q = quantize_nvfp4(
            act.float(), float(in2[expert]), include_input_scale=False, scale_convention=g.scale_convention
        ).double()
        y = project(act_q, _decode_nvfp4_f64(w2[expert], q2[expert]), a2[expert], in2[expert]).bfloat16()
        routes[r_idx, s_idx] = y.float()

    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for slot in range(g.top_k):
        out = out + routes[:, slot] * weights[:, slot, None]
    return out.bfloat16()


def _dynamic_f64_reference(g, x: torch.Tensor, ids: torch.Tensor, gammas: torch.Tensor, metas) -> torch.Tensor:
    from .test_cuteafd_inkling2_nvfp4_moe import _situ, quantize_rows

    m1, m3, m2 = metas
    rows = x.shape[0]
    xd = x.double()
    _, _, am_x, x_qs = quantize_rows(xd)
    x_deq = x_qs * (am_x.double() / 1536.0)[:, None]
    routes = torch.zeros((rows, g.top_k, g.hidden), dtype=torch.float32, device=x.device)
    for ex in ids.unique().tolist():
        if ex < 0 or ex >= g.experts:
            continue
        r_idx, s_idx = torch.where(ids == ex)
        xr = x_deq[r_idx]
        w1_d = _decode_nvfp4_f64(m1[0][ex], m1[1][ex]) * (float(m1[2][ex]) / m1[3])
        w3_d = _decode_nvfp4_f64(m3[0][ex], m3[1][ex]) * (float(m3[2][ex]) / m3[3])
        w2_d = _decode_nvfp4_f64(m2[0][ex], m2[1][ex]) * (float(m2[2][ex]) / m2[3])
        gate = (xr @ w1_d.T).float().bfloat16().float()
        up = (xr @ w3_d.T).float().bfloat16().float()
        act_bf16 = _situ(gate, up, gammas[r_idx, s_idx], g.situ_gate_cap, g.situ_up_cap)
        _, _, am_a, a_qs = quantize_rows(act_bf16.double())
        act_deq = a_qs * (am_a.double() / 1536.0)[:, None]
        routes[r_idx, s_idx] = (act_deq @ w2_d.T).float().bfloat16().float()
    out = torch.zeros((rows, g.hidden), dtype=torch.float32, device=x.device)
    for slot in range(g.top_k):
        out = out + routes[:, slot]
    return out.bfloat16()


def test_peeled_k256_step_counts():
    """Verify exact m16n8k64 step counts across full and peeled K_block=256 tails (zero wasted MMAs)."""
    from b12x.integration.cuteafd._nvfp4_moe_a4 import StreamNvfp4LinearA4

    expected_steps = {
        96: (0, 96, 2, 2),
        112: (0, 112, 2, 2),
        128: (0, 128, 2, 2),
        160: (0, 160, 3, 3),
        208: (0, 208, 4, 4),
        224: (0, 224, 4, 4),
        256: (1, 0, 0, 4),
        288: (1, 32, 1, 5),
        320: (1, 64, 1, 5),
        336: (1, 80, 2, 6),
        352: (1, 96, 2, 6),
        384: (1, 128, 2, 6),
        400: (1, 144, 3, 7),
        416: (1, 160, 3, 7),
        432: (1, 176, 3, 7),
        448: (1, 192, 3, 7),
        464: (1, 208, 4, 8),
        480: (1, 224, 4, 8),
        512: (2, 0, 0, 8),
        560: (2, 48, 1, 9),
        576: (2, 64, 1, 9),
        592: (2, 80, 2, 10),
        608: (2, 96, 2, 10),
        624: (2, 112, 2, 10),
        640: (2, 128, 2, 10),
        672: (2, 160, 3, 11),
        688: (2, 176, 3, 11),
        768: (3, 0, 0, 12),
        1024: (4, 0, 0, 16),
        1040: (4, 16, 1, 17),
        1536: (6, 0, 0, 24),
        2048: (8, 0, 0, 32),
    }
    for k, (full_b, tail_k, tail_s, total_s) in expected_steps.items():
        lin = StreamNvfp4LinearA4(n=128, k=k, experts=1)
        assert lin.k_block == 256
        assert (lin.full_blocks, lin.tail_k, lin.tail_steps) == (full_b, tail_k, tail_s), k
        issued = lin.full_blocks * 4 + lin.tail_steps
        assert issued == total_s == (k + 63) // 64, (k, issued, total_s)
        assert lin.use_tma_w == (k % 32 == 0), k


@pytest.mark.parametrize(
    "n,k,tile_m,tile_n,live",
    [
        # Wide TMA (K % 32 == 0) + partial K + partial N
        (128, 32, 128, 128, 73),
        (128, 96, 128, 128, 128),
        (128, 160, 128, 128, 45),
        (128, 288, 128, 128, 19),
        (128, 576, 128, 128, 128),
        (104, 352, 128, 128, 37),
        (136, 672, 128, 128, 65),
        # Wide K % 32 == 16 (8B cp.async zfill + non-4B scales) + partial N
        (128, 16, 128, 128, 23),
        (128, 48, 128, 128, 128),
        (128, 112, 128, 128, 31),
        (128, 336, 128, 128, 91),
        (128, 464, 128, 128, 128),
        (128, 560, 128, 128, 52),
        (128, 592, 128, 128, 128),
        (128, 624, 128, 128, 64),
        (128, 688, 128, 128, 128),
        (128, 1040, 128, 128, 41),
        (72, 592, 128, 128, 29),
        (8, 464, 128, 128, 17),
        # Sparse multi-stage tiles (M=16, N in 32/64/128) across K % 32 == 0 and 16
        (40, 160, 16, 32, 1),
        (24, 464, 16, 32, 7),
        (72, 592, 16, 64, 15),
        (136, 688, 16, 128, 16),
        (128, 1040, 16, 64, 9),
    ],
)
def test_standalone_linear_exact_f64_grid(n, k, tile_m, tile_n, live):
    require_b12x()
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._nvfp4_moe_a4 import StreamNvfp4LinearA4
    from b12x.integration.cuteafd._nvfp4_moe_dynamic import StreamNvfp4LinearDyn

    experts = 3
    target_expert = 1
    padded_rows = 128
    seed = n * 1009 + k * 37 + tile_m + tile_n

    aq = torch.full((padded_rows, k // 2), 0xFF, dtype=torch.uint8, device="cuda")
    aq[:live] = (_splitmix_u32((live, k // 2), seed) & 0xFF).to(torch.uint8)
    a_s = torch.full((padded_rows, k // 16), 0xFF, dtype=torch.uint8, device="cuda")
    a_s[:live] = torch.ones((live, k // 16), device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)

    w = torch.full((experts, n, k // 2), 0x77, dtype=torch.uint8, device="cuda")
    w[target_expert] = (_splitmix_u32((n, k // 2), seed + 1) & 0xFF).to(torch.uint8)

    ws_grid = torch.full((experts, n, k // 16), 0x7E, dtype=torch.uint8, device="cuda")
    ws_grid[target_expert] = torch.ones((n, k // 16), device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)

    meta = torch.tensor(
        [1, 1, live, 0, 0, live, 0, 0, 0, live, target_expert, 0, 0, target_expert],
        dtype=torch.int32, device="cuda",
    )
    a_f64 = _decode_nvfp4_f64(aq[:live], a_s[:live])
    w_f64 = _decode_nvfp4_f64(w[target_expert], ws_grid[target_expert])
    expected = (a_f64 @ w_f64.T).float().bfloat16()

    lin_a4 = StreamNvfp4LinearA4(n=n, k=k, experts=experts, tile_m=tile_m, tile_n=tile_n)
    ws_a4 = torch.cat([
        ws_grid.flatten(),
        torch.ones(2 * experts, dtype=torch.float32, device="cuda").view(torch.uint8),
    ]).contiguous()

    class LaunchA4:
        def key(self):
            return ("test_lin_a4", lin_a4.key())

        @cute.jit
        def __call__(self, p_aq: cute.Pointer, p_as: cute.Pointer, p_meta: cute.Pointer,
                     p_w: cute.Pointer, p_ws: cute.Pointer, p_out: cute.Pointer, rows_s: Int32, stream: cuda.CUstream):
            lin_a4(p_aq, p_as, p_meta, p_w, p_ws, p_out, Int32(1), stream)

    ops_a4 = (
        Operand("aq", torch.uint8, "[128,K/2]"),
        Operand("as", torch.uint8, "[128,K/16]"),
        Operand("meta", torch.int32, "[14]", align=4),
        Operand("w", torch.uint8, "[E,N,K/2]"),
        Operand("ws", torch.uint8, "[S]", align=4),
        Operand("out", torch.bfloat16, "[live,N]", "out"),
    )
    prog_a4 = compile_program(LaunchA4(), name="test_lin_a4", operands=ops_a4,
                              scalars=(Scalar("rows"),), key=("test_lin_a4", lin_a4.key()))
    out_a4 = torch.full((live + 2, n), -99.0, dtype=torch.bfloat16, device="cuda")
    prog_a4.launch(aq, a_s, meta, w, ws_a4, out_a4[1:-1], scalars=(live,))
    torch.cuda.synchronize()
    assert torch.equal(out_a4[1:-1], expected), f"A4 mismatch at N={n} K={k} M={tile_m} TN={tile_n}"
    assert torch.all(out_a4[0] == -99.0) and torch.all(out_a4[-1] == -99.0)

    lin_dyn = StreamNvfp4LinearDyn(n=n, k=k, experts=experts, tile_m=tile_m, tile_n=tile_n)
    ws_dyn = torch.cat([
        ws_grid.flatten(),
        torch.full((experts,), 1536.0, dtype=torch.float32, device="cuda").view(torch.uint8),
        torch.tensor([1.0], dtype=torch.float32, device="cuda").view(torch.uint8),
    ]).contiguous()
    row_amax = torch.full((padded_rows,), float("nan"), dtype=torch.float32, device="cuda")
    row_amax[:live] = 1.0

    class LaunchDyn:
        def key(self):
            return ("test_lin_dyn", lin_dyn.key())

        @cute.jit
        def __call__(self, p_aq: cute.Pointer, p_as: cute.Pointer, p_meta: cute.Pointer,
                     p_w: cute.Pointer, p_ws: cute.Pointer, p_out: cute.Pointer,
                     p_amax: cute.Pointer, rows_s: Int32, stream: cuda.CUstream):
            lin_dyn(p_aq, p_as, p_meta, p_w, p_ws, p_out, Int32(1), stream, row_amax=p_amax)

    ops_dyn = (*ops_a4, Operand("amax", torch.float32, "[128]", align=4))
    prog_dyn = compile_program(LaunchDyn(), name="test_lin_dyn", operands=ops_dyn,
                               scalars=(Scalar("rows"),), key=("test_lin_dyn", lin_dyn.key()))
    out_dyn = torch.full((live + 2, n), -99.0, dtype=torch.bfloat16, device="cuda")
    prog_dyn.launch(aq, a_s, meta, w, ws_dyn, out_dyn[1:-1], row_amax, scalars=(live,))
    torch.cuda.synchronize()
    assert torch.equal(out_dyn[1:-1], expected), f"Dyn mismatch at N={n} K={k} M={tile_m} TN={tile_n}"


@pytest.mark.parametrize(
    "geom_name,width",
    [
        ("glmf_nvfp4", 688),
        ("glmf_nvfp4", 416),
        ("glmf_nvfp4", 336),
        ("qwen4_nvfp4", 160),
        ("qwen4_nvfp4", 112),
        ("qwen4_nvfp4", 96),
        ("minimax_nvfp4a4", 624),
        ("minimax_nvfp4a4", 432),
    ],
)
def test_calibrated_fused_matches_unfused_bit_exact(geom_name, width, monkeypatch):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = exact_slice_geometry(GEOMETRIES[geom_name], width)
    w = _make_calibrated_weights(g, seed=101 + width)
    capacity = 256
    x = _dyadic_bf16((capacity, g.hidden), seed=303 + width)
    ids, weights = _deterministic_routes(capacity, g.experts, g.top_k, seed=505 + width)

    monkeypatch.setenv("B12X_NVFP4_A4_FUSED", "1")
    prog_fused = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    scr_fused = torch.empty(fp8_moe_scratch_bytes(g, "auto", capacity, False), dtype=torch.uint8, device="cuda")

    monkeypatch.setenv("B12X_NVFP4_A4_FUSED", "0")
    prog_split = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    scr_split = torch.empty(fp8_moe_scratch_bytes(g, "auto", capacity, False), dtype=torch.uint8, device="cuda")

    for rows in (1, 7, 80, 200):
        scr_fused.fill_(0xA5)
        scr_split.fill_(0x5A)
        out_f = torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda")
        out_s = torch.empty((rows, g.hidden), dtype=torch.bfloat16, device="cuda")
        prog_fused.launch(x[:rows], ids[:rows], weights[:rows], *w, out_f, scr_fused, scalars=(rows,))
        prog_split.launch(x[:rows], ids[:rows], weights[:rows], *w, out_s, scr_split, scalars=(rows,))
        torch.cuda.synchronize()
        assert torch.equal(out_f, out_s), f"{geom_name} w{width} rows={rows}: fused != split"


# Complete 30 exact/TP family widths across all 5 NVFP4 families + synthetic 560/592/1040 + 480:
#   glmf_nvfp4 (6): 688, 672, 416, 400, 352, 336
#   glm_nvfp4  (4): 688, 672, 352, 336
#   qwen4_nvfp4 (6): 320, 224, 208, 160, 112, 96
#   minimax_nvfp4a4 (8): 1536, 1024, 768, 624, 608, 512, 448, 432
#   inkling2_nvfp4 (6 exact + 2 aligned + 4 boundary/synthetic):
#       576, 464, 448, 336, 320, 288 (+ 768, 384, 480, 560, 592, 1040)
EXACT_FAMILY_WIDTHS = [
    ("glmf_nvfp4", 688), ("glmf_nvfp4", 672), ("glmf_nvfp4", 416),
    ("glmf_nvfp4", 400), ("glmf_nvfp4", 352), ("glmf_nvfp4", 336),
    ("glm_nvfp4", 688), ("glm_nvfp4", 672), ("glm_nvfp4", 352), ("glm_nvfp4", 336),
    ("qwen4_nvfp4", 320), ("qwen4_nvfp4", 224), ("qwen4_nvfp4", 208),
    ("qwen4_nvfp4", 160), ("qwen4_nvfp4", 112), ("qwen4_nvfp4", 96),
    ("minimax_nvfp4a4", 1536), ("minimax_nvfp4a4", 1024), ("minimax_nvfp4a4", 768), ("minimax_nvfp4a4", 624),
    ("minimax_nvfp4a4", 608), ("minimax_nvfp4a4", 512), ("minimax_nvfp4a4", 448), ("minimax_nvfp4a4", 432),
    ("inkling2_nvfp4", 768), ("inkling2_nvfp4", 576), ("inkling2_nvfp4", 480), ("inkling2_nvfp4", 464),
    ("inkling2_nvfp4", 448), ("inkling2_nvfp4", 384), ("inkling2_nvfp4", 336), ("inkling2_nvfp4", 320),
    ("inkling2_nvfp4", 288), ("inkling2_nvfp4", 560), ("inkling2_nvfp4", 592), ("inkling2_nvfp4", 1040),
]


@pytest.mark.parametrize("geom_name,width", EXACT_FAMILY_WIDTHS)
def test_all_families_exact_unpadded_f64_oracle(geom_name, width):
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    torch.backends.cuda.matmul.allow_tf32 = False
    g = exact_slice_geometry(GEOMETRIES[geom_name], width)
    assert g.slice == width
    capacity = 128
    family_tag = sum(ord(ch) * (i + 1) for i, ch in enumerate(geom_name))
    seed = 7001 + family_tag * 13 + width
    x = _dyadic_bf16((capacity, g.hidden), seed=seed)
    ids, weights = _deterministic_routes(capacity, g.experts, g.top_k, seed=seed + 17)

    if g.dynamic_nvfp4:
        w, metas = _make_dynamic_weights_fast(g, seed=seed + 31)
    else:
        w = _make_calibrated_weights(g, seed=seed + 31)
        metas = None

    op_sha = _tensor_sha256(x, ids, weights, *w)
    program = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", capacity, False), dtype=torch.uint8, device="cuda")
    out = torch.full((capacity + 2, g.hidden), 13.0, dtype=torch.bfloat16, device="cuda")

    actuals = {}
    with kernel_resolution_guard(f"{geom_name} w{width} live-row reuse"):
        for rows in (1, 7, 80):
            scratch.fill_(0xA5)
            out.fill_(13.0)
            program.launch(x, ids, weights, *w, out[1:-1], scratch, scalars=(rows,))
            torch.cuda.synchronize()
            actual = out[1:rows + 1].clone()
            assert torch.isfinite(actual).all() and actual.count_nonzero() > 0
            assert torch.all(out[0] == 13.0) and torch.all(out[rows + 1:] == 13.0)
            actuals[rows] = actual

    out_shas = {}
    for rows, actual in actuals.items():
        if g.dynamic_nvfp4:
            expected = _dynamic_f64_reference(g, x[:rows], ids[:rows], weights[:rows], metas)
        else:
            expected = _calibrated_f64_reference_gpu(g, x[:rows], ids[:rows], weights[:rows], *w)
        af, ef = actual.float(), expected.float()
        cos = float((af.double() * ef.double()).sum() / (af.double().norm() * ef.double().norm()))
        row_cos = float(torch.nn.functional.cosine_similarity(af, ef, dim=1).min())
        assert cos >= 0.9999 and row_cos >= 0.999, (geom_name, width, rows, cos, row_cos)
        out_shas[rows] = _tensor_sha256(actual)
    print(f"DYADIC_ORACLE {geom_name} w{width}: op_sha={op_sha[:16]} out80_sha={out_shas[80][:16]} cos80={cos:.7f}")


@pytest.mark.parametrize(
    "geom_name,width",
    [
        ("glmf_nvfp4", 688),
        ("qwen4_nvfp4", 160),
        ("minimax_nvfp4a4", 624),
        ("inkling2_nvfp4", 464),
        ("inkling2_nvfp4", 592),
    ],
)
def test_representative_exported_object_4096_and_graph(geom_name, width, tmp_path):
    """Full capacity=4096 ExportedProgram C-ABI + rows 1/7/80/2752/4096 + graph replay."""
    require_b12x()
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.fp8_moe import GEOMETRIES, compile_fp8_moe_aot, fp8_moe_scratch_bytes

    g = exact_slice_geometry(GEOMETRIES[geom_name], width)
    capacity = 4096
    seed = 9001 + width
    x = _dyadic_bf16((capacity, g.hidden), seed=seed)
    ids, weights = _deterministic_routes(capacity, g.experts, g.top_k, seed=seed + 11)
    w = _make_dynamic_weights_fast(g, seed=seed + 23)[0] if g.dynamic_nvfp4 else _make_calibrated_weights(g, seed=seed + 23)

    ordinary = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    with exportable_compilation():
        exp_prog = compile_fp8_moe_aot(g, route="auto", max_rows=capacity, wire=False)
    exported = ExportedProgram(exp_prog, tmp_path, f"{geom_name}_w{width}")

    sbytes = fp8_moe_scratch_bytes(g, "auto", capacity, False)
    scratch = [torch.empty(sbytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
    outputs = [torch.full((capacity + 2, g.hidden), 11.0, dtype=torch.bfloat16, device="cuda") for _ in range(2)]

    with kernel_resolution_guard(f"{geom_name} w{width} exported object"):
        for rows in (1, 7, 80, 2752, 4096):
            for prog, out, ws in zip((ordinary, exported), outputs, scratch):
                ws.fill_(0xA5)
                out.fill_(11.0)
                prog.launch(x, ids, weights, *w, out[1:-1], ws, scalars=(rows,))
            torch.cuda.synchronize()
            assert torch.equal(outputs[0][1:rows + 1], outputs[1][1:rows + 1]), (geom_name, width, rows)
            assert torch.isfinite(outputs[1][1:rows + 1]).all()
            assert outputs[1][1:rows + 1].abs().max() > 0
            assert torch.all(outputs[1][0] == 11.0) and torch.all(outputs[1][rows + 1:] == 11.0)
        graph = torch.cuda.CUDAGraph()
        outputs[1].fill_(11.0)
        with torch.cuda.graph(graph):
            exported.launch(x, ids, weights, *w, outputs[1][1:-1], scratch[1], scalars=(80,))
        outputs[1].fill_(11.0)
        graph.replay()
        torch.cuda.synchronize()
        ordinary.launch(x, ids, weights, *w, outputs[0][1:-1], scratch[0], scalars=(80,))
        torch.cuda.synchronize()
        assert torch.equal(outputs[0][1:81], outputs[1][1:81])
