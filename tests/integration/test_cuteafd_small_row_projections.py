"""Small-row BF16 projection tiles, narrow skinny routes, scaled skinny GEMV,
and bit-exact multi-band Muse F32 router tests."""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


# ---------------------------------------------------------------------------
# Pure policy tests (CPU-only, no CUDA required)
# ---------------------------------------------------------------------------

def test_default_decode_tma_tiles_policy():
    import cutlass.utils as cutlass_utils
    from b12x.gemm.bf16_gemv._skinny import (
        DECODE_TMA_TILES,
        decode_tma_tiles,
        default_decode_tma_tiles,
    )

    cap = cutlass_utils.get_smem_capacity_in_bytes("sm_120")
    for (n, k), table in DECODE_TMA_TILES.items():
        assert decode_tma_tiles(n, k) == table
    assert default_decode_tma_tiles(16, 3072) == ((16, 1, 2, 16, 16, 8), (64, 2, 2, 32, 16, 8))
    assert default_decode_tma_tiles(1024, 8192) == ((16, 1, 2, 16, 16, 8), (64, 2, 2, 32, 32, 8))
    assert default_decode_tma_tiles(1536, 3072) == (
        (16, 1, 4, 16, 32, 10), (32, 2, 2, 32, 32, 10), (64, 2, 2, 64, 64, 6)
    )
    assert default_decode_tma_tiles(6144, 4608) == (
        (16, 1, 4, 16, 64, 9), (32, 2, 2, 32, 64, 8), (64, 2, 2, 64, 64, 6)
    )
    assert default_decode_tma_tiles(16384, 6144) == ()
    assert default_decode_tma_tiles(512, 32) == ()
    for n in (8, 16, 24, 48, 352, 512, 1024, 1536, 2048, 3072, 4608, 6144, 8192):
        tiles = default_decode_tma_tiles(n, 4096)
        assert tiles
        prev = 0
        for entry in tiles:
            assert len(entry) == 6
            rows, wm, wn, tm, tn, stages = entry
            assert rows > prev
            prev = rows
            assert tm % (16 * wm) == 0 and tn % (8 * wn) == 0
            smem = 1024 + (tm + tn) * 64 * 2 * stages
            assert smem <= cap, (n, entry, smem, cap)


@pytest.mark.parametrize("n,k", [
    (8, 4096), (16, 4096), (16, 3072), (32, 4096), (32, 6144),
    (64, 4096), (128, 3072), (128, 6144), (160, 6144), (256, 4096),
    (352, 4096), (384, 4608),
])
def test_narrow_skinny_tiles_single_thread_count(n, k):
    from b12x.gemm.bf16_gemv._skinny import narrow_skinny_tiles

    tiles = narrow_skinny_tiles(n, k, max_rows=64)
    assert tiles and tiles[-1][0] == 64
    assert len({t for _, _, _, t in tiles}) == 1
    prev = 0
    for rows, cols, rpt, threads in tiles:
        assert rows > prev
        prev = rows
        assert n % cols == 0
        assert (k // 8) % threads == 0
        assert threads % 32 == 0
        assert cols * rpt <= threads


# ---------------------------------------------------------------------------
# Pillar 1: Default TMA decode/verify tiles are bit-exact to prefill_config
# ---------------------------------------------------------------------------

PILLAR1_SHAPES = [
    (8192, 1024, torch.bfloat16),   # Muse TP1 k/v_proj
    (8192, 512, torch.bfloat16),    # Muse TP2 k/v_proj
    (3072, 1536, torch.bfloat16),   # Hy4-flash latent_down
    (6144, 2048, torch.bfloat16),   # Hy4 q_a
    (4608, 6144, torch.float32),    # Inkling2 Q (F32 out)
    (4608, 48, torch.float32),      # Inkling2 sink (F32 out)
    (3072, 16, torch.bfloat16),     # Hy4-flash indexer_weights (N<=16 tile)
]


@pytest.mark.parametrize("k,n,dtype", PILLAR1_SHAPES)
def test_pillar1_default_decode_tma_bit_exact_against_prefill(k, n, dtype):
    require_b12x()
    import cutlass
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd import pointwise
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    c_dt = cutlass.Float32 if dtype == torch.float32 else cutlass.BFloat16
    canon = pointwise.compile_bf16_projection_aot(width=k, outputs=n, output_dtype=dtype)
    default_launch = TmaBf16Projection(n, k, out_dtype=c_dt, tiles_by_rows=())
    default = compile_program(
        default_launch, name="test_p1_default",
        operands=(Operand("x", torch.bfloat16, f"[rows,{k}]"),
                  Operand("w", torch.bfloat16, f"[{n},{k}]"),
                  Operand("out", dtype, f"[rows,{n}]", "out")),
        scalars=(Scalar("rows"),), key=("p1_def", default_launch.key()),
    )
    gen = torch.Generator(device="cuda").manual_seed(k * 7 + n)
    w = (torch.randn((n, k), generator=gen, device="cuda") / 16).bfloat16()
    max_rows = 128
    x = torch.randn((max_rows, k), generator=gen, device="cuda").bfloat16()

    for rows in (1, 7, 16, 17, 32, 33, 64, 65, 128):
        out_c = torch.full((max_rows, n), float("nan"), dtype=dtype, device="cuda")
        out_d = torch.full((max_rows, n), float("nan"), dtype=dtype, device="cuda")
        canon.launch(x[:rows], w, out_c, scalars=(rows,))
        default.launch(x[:rows], w, out_d, scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out_c[:rows], out_d[:rows]), f"K={k} N={n} rows={rows}: default TMA tiles differ"
        if rows < max_rows:
            assert torch.isnan(out_c[rows:]).all(), f"K={k} N={n} rows={rows}: wrote past live rows"

    # CUDA graph capture and replay at M=16 and M=64
    for rows in (16, 64):
        buf = torch.zeros((rows, k), dtype=torch.bfloat16, device="cuda")
        out_g = torch.zeros((rows, n), dtype=dtype, device="cuda")
        want = torch.empty((rows, n), dtype=dtype, device="cuda")
        default.launch(x[:rows], w, want, scalars=(rows,))
        canon.launch(buf, w, out_g, scalars=(rows,))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            canon.launch(buf, w, out_g, scalars=(rows,))
        torch.cuda.synchronize()
        buf.copy_(x[:rows])
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(out_g, want)


# ---------------------------------------------------------------------------
# Pillar 2: Narrow skinny projection (1..64 exact row invariance, >64 TMA exact)
# ---------------------------------------------------------------------------

PILLAR2_SHAPES = [
    (4096, 32, torch.bfloat16),   # step52_attn_gate
    (4096, 64, torch.bfloat16),   # step5_attn_gate
    (4096, 256, torch.bfloat16),  # step5_index_kz
    (4096, 352, torch.float32),   # step5_router
    (3072, 16, torch.bfloat16),   # hy4f_indexer_weights
    (3072, 128, torch.bfloat16),  # hy4f_indexer_wk
    (6144, 160, torch.bfloat16),  # ml4_index_kw
]


@pytest.mark.parametrize("k,n,dtype", PILLAR2_SHAPES)
def test_pillar2_skinny_projection_row_invariance_and_cutoff(k, n, dtype):
    require_b12x()
    from b12x.integration.cuteafd import pointwise

    prog = pointwise.compile_bf16_projection_aot(width=k, outputs=n, output_dtype=dtype, skinny_max_rows=64)
    assert prog.geometry["skinny_rows"] == 64
    assert prog.geometry["exact_row_cap"] == 64
    tma_ref = pointwise.compile_bf16_projection_aot(width=k, outputs=n, output_dtype=dtype)

    gen = torch.Generator(device="cuda").manual_seed(k * 13 + n)
    w = (torch.randn((n, k), generator=gen, device="cuda") / 16).bfloat16()
    x = torch.randn((128, k), generator=gen, device="cuda").bfloat16()

    # Single-row oracle across all 64 rows
    singles = torch.empty((64, n), dtype=dtype, device="cuda")
    for r in range(64):
        prog.launch(x[r:r + 1], w, singles[r:r + 1], scalars=(1,))
    torch.cuda.synchronize()

    # Numerical accuracy check against FP64
    ref64 = (x[:64].double() @ w.double().T).to(dtype)
    atol = 1e-2 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(singles.float(), ref64.float(), atol=atol, rtol=1e-2)

    # Exact bitwise row invariance across all skinny sub-bands (1..64) + tail poison
    for rows in (1, 4, 8, 16, 17, 32, 64):
        out = torch.full((80, n), float("nan"), dtype=dtype, device="cuda")
        prog.launch(x[:rows], w, out, scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out[:rows], singles[:rows]), f"K={k} N={n} rows={rows}: skinny row invariance broken"
        assert torch.isnan(out[rows:]).all(), f"K={k} N={n} rows={rows}: wrote past live rows"

    # Past skinny cutoff (>64): bit-identical to TMA prefill_config
    for rows in (65, 128):
        out_s = torch.empty((rows, n), dtype=dtype, device="cuda")
        out_t = torch.empty((rows, n), dtype=dtype, device="cuda")
        prog.launch(x[:rows], w, out_s, scalars=(rows,))
        tma_ref.launch(x[:rows], w, out_t, scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out_s, out_t), f"K={k} N={n} rows={rows}: >64 route must match TMA bit-for-bit"

    # CUDA graph replay
    for rows in (8, 64):
        buf = torch.zeros((rows, k), dtype=torch.bfloat16, device="cuda")
        out_g = torch.zeros((rows, n), dtype=dtype, device="cuda")
        prog.launch(buf, w, out_g, scalars=(rows,))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prog.launch(buf, w, out_g, scalars=(rows,))
        torch.cuda.synchronize()
        buf.copy_(x[:rows])
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(out_g, singles[:rows])


@pytest.mark.parametrize("outputs", (8, 16))
def test_pillar2_scaled_projection_fused_skinny_and_cutoff(outputs):
    require_b12x()
    from b12x.integration.cuteafd import pointwise
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    width, prescale = 4096, 0.5
    prog = pointwise.compile_bf16_scaled_projection_aot(width=width, outputs=outputs, prescale=prescale)
    assert prog.geometry["exact_row_cap"] == 64
    stock_launch = pointwise.ScaleProjection(width, outputs, prescale, skinny_max_rows=0)
    stock = compile_program(
        stock_launch, name="test_scale_stock",
        operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                  Operand("weight", torch.bfloat16, f"[{outputs},{width}]"),
                  Operand("out", torch.float32, f"[rows,{outputs}]", "out"),
                  Operand("scratch", torch.bfloat16, f"[rows,{outputs}]", "scratch")),
        scalars=(Scalar("rows"),), key=("stock_scale", width, outputs, prescale),
        scratch={"scratch": lambda rows: int(rows) * int(outputs) * 2},
    )
    gen = torch.Generator(device="cuda").manual_seed(99 + outputs)
    w = (torch.randn((outputs, width), generator=gen, device="cuda") / 16).bfloat16()
    x = torch.randn((128, width), generator=gen, device="cuda").bfloat16()
    scratch = torch.empty((128, outputs), dtype=torch.bfloat16, device="cuda")

    singles = torch.empty((64, outputs), dtype=torch.float32, device="cuda")
    for r in range(64):
        prog.launch(x[r:r + 1], w, singles[r:r + 1], scratch[:1], scalars=(1,))
    torch.cuda.synchronize()
    ref64 = (x[:64].double() @ w.double().T).bfloat16().float() * prescale
    torch.testing.assert_close(singles, ref64, atol=1e-2, rtol=1e-2)

    for rows in (1, 8, 16, 17, 64):
        out = torch.full((80, outputs), float("nan"), dtype=torch.float32, device="cuda")
        prog.launch(x[:rows], w, out, scratch[:rows], scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out[:rows], singles[:rows])
        assert torch.isnan(out[rows:]).all()

    for rows in (65, 128):
        out_p = torch.empty((rows, outputs), dtype=torch.float32, device="cuda")
        out_s = torch.empty((rows, outputs), dtype=torch.float32, device="cuda")
        prog.launch(x[:rows], w, out_p, scratch[:rows], scalars=(rows,))
        stock.launch(x[:rows], w, out_s, scratch[:rows], scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out_p, out_s)


# ---------------------------------------------------------------------------
# Pillar 3: Muse F32 router multi-band bit-exactness across all widths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows", (1, 2, 4, 6, 8, 9, 16, 17, 32, 64, 65, 256, 2048, 2752, 4096))
def test_pillar3_muse_router_multiband_bit_exact(rows):
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_router_aot

    gen = torch.Generator(device="cuda").manual_seed(77 + rows)
    x = torch.randn((rows, 8192), generator=gen, device="cuda").bfloat16()
    w = (torch.randn((256, 8192), generator=gen, device="cuda") * 0.01).float()
    shipped = compile_router_aot()
    scalar = compile_router_aot(scalar_max_rows=2**31 - 1)
    out_fast = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
    out_ref = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
    shipped.launch(x, w, out_fast, scalars=(rows,))
    scalar.launch(x, w, out_ref, scalars=(rows,))
    torch.cuda.synchronize()
    assert _same_bits(out_fast, out_ref), f"muse_router multiband differs from _RouterScalar at rows={rows}"
    # Single-row invariance check on boundary rows
    for at in sorted({0, rows // 2, rows - 1}):
        one = torch.empty((1, 256), dtype=torch.float32, device="cuda")
        shipped.launch(x[at:at + 1], w, one, scalars=(1,))
        torch.cuda.synchronize()
        assert _same_bits(one, out_fast[at:at + 1])
