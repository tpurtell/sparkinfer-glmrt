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
    return official_module().weighted_topk_gather(outputs.bfloat16().contiguous(), routes)


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 16, 64, 640, 767, 768, 769, 2048, 2049])
def test_native_a8(rows):
    require_b12x()
    from b12x.integration.cuteafd.fp8_moe import Fp8MoeGeometry, compile_fp8_moe_aot, fp8_moe_scratch_bytes
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    tp = int(os.environ.get("CUTEAFD_A8_TP", "0"))
    g = (Fp8MoeGeometry("k128_test", 4096, 352, 8, 1536, tp=tp, activations="a8_k128_f32",
                        swiglu_limit=7, activation="silu_post_clamp_fp32_product") if tp else
         Fp8MoeGeometry("k128_test", 256, 4, 2, 256, activations="a8_k128_f32",
                        swiglu_limit=7, activation="silu_post_clamp_fp32_product"))
    h, i, e, topk = g.hidden, g.slice, g.experts, g.top_k
    def weight(n, k):
        return (torch.randn(e, n, k, device="cuda") * 30).clamp(-448, 448).to(torch.float8_e4m3fn)
    def scales(n, k):
        return torch.rand(e, n // 128, k // 128, device="cuda") * 0.001 + 0.00031
    ws = (weight(i, h), scales(i, h), weight(i, h), scales(i, h), weight(h, i), scales(h, i))
    x = torch.randn(rows, h, device="cuda").bfloat16()
    x[0].zero_()
    if rows > 1:
        x[1] *= 1e-12
    ids = torch.randint(0, e, (rows, topk), device="cuda", dtype=torch.int32)
    routes = torch.rand(rows, topk, device="cuda")
    program = compile_fp8_moe_aot(g, route="auto", max_rows=rows, wire=False)
    scratch = torch.empty(fp8_moe_scratch_bytes(g, "auto", rows, False), device="cuda", dtype=torch.uint8)
    out = torch.empty_like(x)
    def launch():
        program.launch(x, ids, routes, *ws, out, scratch, scalars=(rows,))
    launch()
    torch.cuda.synchronize()
    actual = out.clone()
    if rows == 640:
        from b12x.integration.cuteafd.fp8_moe import _regions, _align
        regions = _regions(g, "prefill", rows, False)
        offsets = [0]
        for region in regions:
            offsets.append(offsets[-1] + _align(region))
        grouped = regions[1] // 4
        pair_rows = scratch[offsets[1]:offsets[1]+regions[1]].view(torch.int32)
        live = pair_rows >= 0
        candidate_q = scratch[offsets[3]:offsets[3]+regions[3]].reshape(grouped, h)[live]
        candidate_s = scratch[offsets[4]:offsets[4]+regions[4]].view(torch.float32).reshape(grouped, h//128)[live]
        reference_q, reference_s = official_module().act_quant(x[pair_rows[live].long()])
        print(f"quant_input mismatched_values={(candidate_q != reference_q.view(torch.uint8)).sum().item()} "
              f"mismatched_scales={(candidate_s != reference_s).sum().item()}")
    expected, fp64 = reference(x, ids, routes, ws, "official"), reference(x, ids, routes, ws, True)
    error = (actual.float() - fp64.float()).square().sum(1)
    official = (expected.float() - fp64.float()).square().sum(1)
    over = error > official * 1.01
    ratio = torch.where(official > 0, error / official, torch.where(error == 0, 0., float("inf")))
    def ordered_bf16(value):
        bits = value.view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)
    ulp = (ordered_bf16(actual) - ordered_bf16(expected)).abs()
    print(f"tp={tp} rows={rows} native_error={error.sum().item():.9g} official_error={official.sum().item():.9g} "
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
    if rows in (1, 16, 64, 640):
        legacy = replace(g, activations="a16")
        old = compile_fp8_moe_aot(legacy, route="auto", max_rows=rows, wire=False)
        old_scratch = torch.empty(fp8_moe_scratch_bytes(legacy, "auto", rows, False), device="cuda", dtype=torch.uint8)
        def old_launch():
            old.launch(x, ids, routes, *ws, out, old_scratch, scalars=(rows,))
        def elapsed(call):
            call()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                call()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 100.0
        base_us, native_us = elapsed(old_launch), elapsed(launch)
        print(f"timing tp={tp} rows={rows} w8a16_us={base_us:.3f} native_a8_us={native_us:.3f} speedup={base_us/native_us:.4f}")
    print(f"replay_width tp={tp} rows={rows} passed")
    assert torch.all(error <= official * 1.01), "strict per-row native/official error bar"
