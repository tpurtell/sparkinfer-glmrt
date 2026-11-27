"""Small-row BF16 projection TMA decode tiles and bit-exact multi-band Muse F32 router tests.

Every route split verified here is strictly bitwise-identical (``torch.equal`` on raw bit
representations, never by tolerance) across every threshold from 1 through 4096 rows and
matches a single-row oracle row-for-row across every band boundary.
"""
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
    for n in (8, 16, 24, 48, 64, 128, 160, 256, 352, 512, 1024, 1536, 2048, 3072, 4608, 6144, 8192):
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


# ---------------------------------------------------------------------------
# Pillar 1: Default TMA decode/verify tiles are bit-exact to prefill_config
# and to a 64-row single-row oracle across EVERY threshold (1..4096)
# ---------------------------------------------------------------------------

PILLAR1_SHAPES = [
    (8192, 1024, torch.bfloat16),   # Muse TP1 k/v_proj
    (8192, 512, torch.bfloat16),    # Muse TP2 k/v_proj
    (3072, 1536, torch.bfloat16),   # Hy4-flash latent_down
    (6144, 2048, torch.bfloat16),   # Hy4 q_a
    (4608, 6144, torch.float32),    # Inkling2 Q (F32 out)
    (4608, 48, torch.float32),      # Inkling2 sink (F32 out)
    (4096, 64, torch.bfloat16),     # Step-5 attn_gate
    (4096, 256, torch.bfloat16),    # Step-5 index_kz
    (4096, 352, torch.float32),     # Step-5 router (F32 out)
    (3072, 128, torch.bfloat16),    # Hy4-flash indexer_wk
    (3072, 16, torch.bfloat16),     # Hy4-flash indexer_weights (N<=16 tile)
    (6144, 160, torch.bfloat16),    # ML4 index_kw
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
    max_rows = 4096
    x = torch.randn((max_rows, k), generator=gen, device="cuda").bfloat16()

    # Single-row oracle across 64 distinct rows: every launch (1..4096, across every threshold)
    # must match singles[:min(rows, 64)] bit-for-bit.
    singles = torch.empty((64, n), dtype=dtype, device="cuda")
    for r in range(64):
        canon.launch(x[r:r + 1], w, singles[r:r + 1], scalars=(1,))
    torch.cuda.synchronize()

    for rows in (1, 7, 16, 17, 32, 33, 64, 65, 128, 256, 2048, 2752, 4096):
        out_c = torch.full((max_rows, n), float("nan"), dtype=dtype, device="cuda")
        out_d = torch.full((max_rows, n), float("nan"), dtype=dtype, device="cuda")
        canon.launch(x[:rows], w, out_c, scalars=(rows,))
        default.launch(x[:rows], w, out_d, scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out_c[:rows], out_d[:rows]), f"K={k} N={n} rows={rows}: default TMA tiles differ"
        check = min(rows, 64)
        assert _same_bits(out_c[:check], singles[:check]), (
            f"K={k} N={n} rows={rows}: bitwise single-row invariance broken across threshold"
        )
        if rows < max_rows:
            assert torch.isnan(out_c[rows:]).all(), f"K={k} N={n} rows={rows}: wrote past live rows"

    # CUDA graph capture and replay at M=16 and M=64
    for rows in (16, 64):
        buf = torch.zeros((rows, k), dtype=torch.bfloat16, device="cuda")
        out_g = torch.zeros((rows, n), dtype=dtype, device="cuda")
        canon.launch(buf, w, out_g, scalars=(rows,))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            canon.launch(buf, w, out_g, scalars=(rows,))
        torch.cuda.synchronize()
        buf.copy_(x[:rows])
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(out_g, singles[:rows])


@pytest.mark.parametrize("outputs", (8, 16))
def test_pillar1_scaled_projection_tma_tiles_bit_exact(outputs):
    require_b12x()
    import cutlass
    from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
    from b12x.integration.cuteafd import pointwise
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    width, prescale = 4096, 0.5
    prog = pointwise.compile_bf16_scaled_projection_aot(width=width, outputs=outputs, prescale=prescale)
    raw_ref = TmaBf16Projection(outputs, width, out_dtype=cutlass.BFloat16, tiles_by_rows=())
    ref_proj = compile_program(
        raw_ref, name="test_scale_ref_proj",
        operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                  Operand("w", torch.bfloat16, f"[{outputs},{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{outputs}]", "out")),
        scalars=(Scalar("rows"),), key=("scale_ref", raw_ref.key()),
    )
    gen = torch.Generator(device="cuda").manual_seed(99 + outputs)
    w = (torch.randn((outputs, width), generator=gen, device="cuda") / 16).bfloat16()
    x = torch.randn((4096, width), generator=gen, device="cuda").bfloat16()
    scratch = torch.empty((4096, outputs), dtype=torch.bfloat16, device="cuda")

    singles = torch.empty((64, outputs), dtype=torch.float32, device="cuda")
    for r in range(64):
        prog.launch(x[r:r + 1], w, singles[r:r + 1], scratch[:1], scalars=(1,))
    torch.cuda.synchronize()

    for rows in (1, 7, 16, 17, 32, 33, 64, 65, 128, 256, 2048, 2752, 4096):
        out_p = torch.empty((rows, outputs), dtype=torch.float32, device="cuda")
        proj_ref = torch.empty((rows, outputs), dtype=torch.bfloat16, device="cuda")
        prog.launch(x[:rows], w, out_p, scratch[:rows], scalars=(rows,))
        ref_proj.launch(x[:rows], w, proj_ref, scalars=(rows,))
        torch.cuda.synchronize()
        want = proj_ref.float() * prescale
        assert _same_bits(out_p, want), f"scaled outputs={outputs} rows={rows}: TMA tiles differ"
        check = min(rows, 64)
        assert _same_bits(out_p[:check], singles[:check]), (
            f"scaled outputs={outputs} rows={rows}: bitwise invariance broken across threshold"
        )


# ---------------------------------------------------------------------------
# Pillar 3: Muse F32 router multi-band bit-exactness across all widths
# ---------------------------------------------------------------------------

def test_pillar3_muse_router_multiband_bit_exact():
    require_b12x()
    from b12x.integration.cuteafd.muse import compile_router_aot

    max_rows = 4096
    gen = torch.Generator(device="cuda").manual_seed(777)
    x = torch.randn((max_rows, 8192), generator=gen, device="cuda").bfloat16()
    w = (torch.randn((256, 8192), generator=gen, device="cuda") * 0.01).float()
    shipped = compile_router_aot()
    scalar = compile_router_aot(scalar_max_rows=2**31 - 1)

    singles = torch.empty((64, 256), dtype=torch.float32, device="cuda")
    for r in range(64):
        shipped.launch(x[r:r + 1], w, singles[r:r + 1], scalars=(1,))
    torch.cuda.synchronize()

    for rows in (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 256, 2048, 2752, 4096):
        out_fast = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
        out_ref = torch.empty((rows, 256), dtype=torch.float32, device="cuda")
        shipped.launch(x[:rows], w, out_fast, scalars=(rows,))
        scalar.launch(x[:rows], w, out_ref, scalars=(rows,))
        torch.cuda.synchronize()
        assert _same_bits(out_fast, out_ref), f"muse_router differs from _RouterScalar at rows={rows}"
        check = min(rows, 64)
        assert _same_bits(out_fast[:check], singles[:check]), (
            f"muse_router single-row invariance broken across threshold at rows={rows}"
        )
