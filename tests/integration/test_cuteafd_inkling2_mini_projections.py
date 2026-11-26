"""Tests for Inkling 2 Mini dedicated small-row attention-input and router BF16 projections."""
from __future__ import annotations

import pytest
import torch

from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection
from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
from b12x.integration.cuteafd.inkling2_mini_projections import (
    STOCK_TILES,
    compile_inkling2_mini_attn_in_proj_global_aot,
    compile_inkling2_mini_attn_in_proj_local_aot,
    compile_inkling2_mini_router_proj_aot,
)


def _exact_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inkling2_mini_small_row_projections_oracle_and_row_invariance():
    torch.manual_seed(2804)
    torch.backends.cuda.matmul.allow_tf32 = False
    max_m = 64
    rows_to_check = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64)

    xq = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xk = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xv = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    xr = (torch.randn(max_m, 2304, device="cuda") * 0.1).bfloat16()
    for t in (xq, xk, xv, xr):
        for r in (2, 7, 15, 63):
            t[r].copy_(t[0])

    wq = (torch.randn(3072, 2304, device="cuda") * 0.02).bfloat16()
    wr = (torch.randn(384, 2304, device="cuda") * 0.02).bfloat16()
    wgate = (torch.randn(3072, 2304, device="cuda") * 0.02).bfloat16()
    wsink = (torch.randn(24, 2304, device="cuda") * 0.02).bfloat16()
    wk = (torch.randn(1024, 2304, device="cuda") * 0.02).bfloat16()
    wv = (torch.randn(1024, 2304, device="cuda") * 0.02).bfloat16()
    wk_dyn = (torch.randn(32, 2304, device="cuda") * 0.02).bfloat16()
    wv_dyn = (torch.randn(32, 2304, device="cuda") * 0.02).bfloat16()
    wrouter = (torch.randn(128, 2304, device="cuda") * 0.02).bfloat16()

    oq = torch.empty(max_m + 1, 3072, device="cuda", dtype=torch.bfloat16)
    orr = torch.empty(max_m + 1, 384, device="cuda", dtype=torch.bfloat16)
    ogate = torch.empty(max_m + 1, 3072, device="cuda", dtype=torch.bfloat16)
    osink = torch.empty(max_m + 1, 24, device="cuda", dtype=torch.float32)
    ok = torch.empty(max_m + 1, 1024, device="cuda", dtype=torch.bfloat16)
    ov = torch.empty(max_m + 1, 1024, device="cuda", dtype=torch.bfloat16)
    okd = torch.empty(max_m + 1, 32, device="cuda", dtype=torch.float32)
    ovd = torch.empty(max_m + 1, 32, device="cuda", dtype=torch.float32)
    orouter = torch.empty(max_m + 1, 128, device="cuda", dtype=torch.float32)

    prog_local = compile_inkling2_mini_attn_in_proj_local_aot(max_rows=4096)
    prog_global = compile_inkling2_mini_attn_in_proj_global_aot(max_rows=4096)
    prog_router = compile_inkling2_mini_router_proj_aot(max_rows=4096)

    import cutlass
    def stock(n, k, f32=False):
        dt = torch.float32 if f32 else torch.bfloat16
        cd = cutlass.Float32 if f32 else cutlass.BFloat16
        return compile_program(
            TmaBf16Projection(n, k, out_dtype=cd, tiles_by_rows=STOCK_TILES),
            name=f"test_stock_{k}_{n}_{'f32' if f32 else 'bf16'}",
            operands=(
                Operand("x", torch.bfloat16, f"[rows,{k}]"),
                Operand("w", torch.bfloat16, f"[{n},{k}]"),
                Operand("out", dt, f"[rows,{n}]", "out"),
            ),
            scalars=(Scalar("rows"),),
            key=("test_stock", k, n, f32),
        )

    st_q = stock(3072, 2304)
    st_r = stock(384, 2304)
    st_kv = stock(1024, 2304)
    st_sink = stock(24, 2304, True)
    st_dyn = stock(32, 2304, True)
    st_router = stock(128, 2304, True)

    router_small = {}
    for m in rows_to_check:
        prog_global.launch(
            xq, xk, xv, wq, wr, wgate, wsink, wk, wv, wk_dyn, wv_dyn,
            oq, orr, ogate, osink, ok, ov, okd, ovd, scalars=(m,),
        )
        torch.cuda.synchronize()
        cand_g = [t[:m].clone() for t in (oq, orr, ogate, osink, ok, ov, okd, ovd)]

        prog_local.launch(
            xq, xk, xv, wq, wr, wgate, wsink, wk, wv,
            oq, orr, ogate, osink, ok, ov, scalars=(m,),
        )
        torch.cuda.synchronize()
        for a, b in zip((oq, orr, ogate, osink, ok, ov), cand_g[:6]):
            assert _exact_bits(a[:m], b)

        st_q.launch(xq, wq, oq, scalars=(m,))
        st_r.launch(xq, wr, orr, scalars=(m,))
        st_q.launch(xq, wgate, ogate, scalars=(m,))
        st_sink.launch(xq, wsink, osink, scalars=(m,))
        st_kv.launch(xk, wk, ok, scalars=(m,))
        st_kv.launch(xv, wv, ov, scalars=(m,))
        st_dyn.launch(xk, wk_dyn, okd, scalars=(m,))
        st_dyn.launch(xv, wv_dyn, ovd, scalars=(m,))
        torch.cuda.synchronize()
        for a, b in zip((oq, orr, ogate, osink, ok, ov, okd, ovd), cand_g):
            assert _exact_bits(a[:m], b), f"global attn-in mismatch at m={m}"

        prog_router.launch(xr, wrouter, orouter, scalars=(m,))
        torch.cuda.synchronize()
        r_cand = orouter[:m].clone()
        torch.testing.assert_close(r_cand, xr[:m].float() @ wrouter.float().T, rtol=5e-5, atol=5e-5)
        if m <= 8:
            router_small[m] = r_cand
            assert _exact_bits(r_cand[0:1], router_small[1][0:1])
            for m1 in router_small:
                if m1 < m:
                    assert _exact_bits(r_cand[:m1], router_small[m1])
        else:
            st_router.launch(xr, wrouter, orouter, scalars=(m,))
            torch.cuda.synchronize()
            assert _exact_bits(r_cand, orouter[:m]), f"router >8 route switch mismatch at m={m}"
