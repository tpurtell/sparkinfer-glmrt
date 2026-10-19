"""K128/F32 native expert arithmetic: independent block oracle and replay."""
from dataclasses import replace
import importlib.util
import os
import sys
from functools import lru_cache
import pytest
import torch
from ..conftest import require_b12x


@lru_cache(None)
def official_module():
    path = os.environ.get("CUTEAFD_A8_OFFICIAL_KERNEL")
    if not path:
        pytest.skip("set CUTEAFD_A8_OFFICIAL_KERNEL to the checkpoint inference/kernel.py")
    spec = importlib.util.spec_from_file_location("official_a8_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quantize(x):
    rows, k = x.shape
    padded = torch.nn.functional.pad(x.float(), (0, (-k) % 128)).reshape(rows, -1, 128)
    scale = padded.abs().amax(-1).clamp_min(1e-10) * (1.0 / 448.0)
    q = (padded * scale.reciprocal()[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(rows, -1)[:, :k].contiguous(), scale


def project(x, w, s, exact=False):
    q, scale = official_module().act_quant(x)
    if exact == "official":
        return official_module().fp8_gemm(q, scale, w, s)
    dtype = torch.float64 if exact is True else torch.float32
    out = torch.zeros((x.shape[0], w.shape[0]), dtype=dtype, device=x.device)
    for kb in range((x.shape[1] + 127) // 128):
        lo, hi = kb * 128, min((kb + 1) * 128, x.shape[1])
        partial = q[:, lo:hi].to(dtype) @ w[:, lo:hi].to(dtype).T
        out += (partial * scale[:, kb, None].to(dtype)) * s[:, kb].repeat_interleave(128)[None, :].to(dtype)
    return out.float().bfloat16()


def reference(x, ids, routes, ws, exact=False):
    outputs = torch.zeros((*ids.shape, x.shape[1]), device=x.device)
    for e in range(ws[0].shape[0]):
        rows, slots = torch.where(ids == e)
        if not rows.numel():
            continue
        gate = project(x[rows], ws[0][e], ws[1][e], exact).float()
        up = project(x[rows], ws[2][e], ws[3][e], exact).float()
        act = official_module().clamped_swiglu(gate.bfloat16().contiguous(), up.bfloat16().contiguous(), 7.0)
        outputs[rows, slots] = project(act, ws[4][e], ws[5][e], exact).float()
    result = torch.zeros(x.shape, device=x.device)
    for slot in range(ids.shape[1]):
        result = result + outputs[:, slot] * routes[:, slot, None]
    return result.bfloat16()


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 16, 64, 640, 767, 768, 769, 2048, 2049])
def test_native_a8(rows):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    g = Fp8MoeGeometry("k128_test", 256, 4, 2, 256, activations="a8_k128_f32",
                       swiglu_limit=7, activation="silu_post_clamp_fp32_product")
    def weight():
        return (torch.randn(4, 256, 256, device="cuda") * 30).clamp(-448, 448).to(torch.float8_e4m3fn)
    def scales():
        return torch.rand(4, 2, 2, device="cuda") * 0.001 + 0.00031
    ws = (weight(), scales(), weight(), scales(), weight(), scales())
    x = torch.randn(rows, 256, device="cuda").bfloat16()
    x[0].zero_()
    if rows > 1:
        x[1] *= 1e-12
    ids = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int32)
    routes = torch.rand(rows, 2, device="cuda")
    program = compile_fp8_moe_aot(g, route="auto", max_rows=rows, wire=False)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", rows, False), device="cuda", dtype=torch.uint8)
    out = torch.empty_like(x)
    def launch():
        program.launch(x, ids, routes, *ws, out, scratch, scalars=(rows,))
    launch()
    torch.cuda.synchronize()
    actual = out.clone()
    expected, fp64 = reference(x, ids, routes, ws, "official"), reference(x, ids, routes, ws, True)
    error = (actual.float() - fp64.float()).square().sum(1)
    official = (expected.float() - fp64.float()).square().sum(1)
    over = error > official * 1.01
    ratio = torch.where(official > 0, error / official, torch.where(error == 0, 0., float("inf")))
    ulp = (actual.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
    print(f"rows={rows} native_error={error.sum().item():.9g} official_error={official.sum().item():.9g} "
          f"rows_over={over.sum().item()} worst_ratio={ratio.max().item()} max_ulp={ulp.max().item()}")
    # Report the strict bar after collecting replay/width evidence, including
    # zero-denominator BF16 ties; no exception is silently blessed here.
    launch()
    assert torch.equal(out, actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        graph.replay()
        assert torch.equal(out, actual)
    if rows > 1:
        one = compile_fp8_moe_aot(g, route="auto", max_rows=1, wire=False)
        one_scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", 1, False), device="cuda", dtype=torch.uint8)
        one_out = torch.empty_like(x[:1])
        at = min(rows - 1, 5)
        one.launch(x[at:at+1], ids[at:at+1], routes[at:at+1], *ws, one_out, one_scratch, scalars=(1,))
        assert torch.equal(one_out, actual[at:at+1])
    assert torch.all(error <= official * 1.01), "strict per-row native/official error bar"
