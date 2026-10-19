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
    return project_quantized(q, scale, w, s, exact)


def project_quantized(q, scale, w, s, exact=True):
    dtype = torch.float64 if exact is True else torch.float32
    out = torch.zeros((q.shape[0], w.shape[0]), dtype=dtype, device=q.device)
    for kb in range((q.shape[1] + 127) // 128):
        lo, hi = kb * 128, min((kb + 1) * 128, q.shape[1])
        partial = q[:, lo:hi].to(dtype) @ w[:, lo:hi].to(dtype).T
        out += (partial * scale[:, kb, None].to(dtype)) * s[:, kb].repeat_interleave(128)[None, :].to(dtype)
    return out.float().bfloat16()


def ordered_bf16(value):
    bits = value.view(torch.int16).int()
    return torch.where(bits < 0, -32768 - bits, bits)


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
    sm121 = torch.cuda.get_device_capability() == (12, 1)
    if sm121 or rows == 640:
        from b12x.integration.cuteafd.fp8_moe import _regions, _align
        regions = _regions(g, "prefill", rows, False)
        offsets = [0]
        for region in regions:
            offsets.append(offsets[-1] + _align(region))
        grouped = regions[1] // 4
        pair_rows = scratch[offsets[1]:offsets[1]+regions[1]].view(torch.int32)
        live = pair_rows >= 0
        def region(index, dtype, width):
            return scratch[offsets[index]:offsets[index]+regions[index]].view(dtype).reshape(grouped, width)
        candidate_q = region(3, torch.float8_e4m3fn, h)
        candidate_s = region(4, torch.float32, h//128)
        reference_q, reference_s = official_module().act_quant(x[pair_rows[live].long()])
        assert torch.equal(candidate_q[live].view(torch.uint8), reference_q.view(torch.uint8))
        assert torch.equal(candidate_s[live], reference_s)
        act = region(6, torch.bfloat16, i)
        down_q, down_s = region(7, torch.float8_e4m3fn, i), region(8, torch.float32, i//128)
        rq, rs = official_module().act_quant(act[live].contiguous())
        assert torch.equal(down_q[live].view(torch.uint8), rq.view(torch.uint8))
        assert torch.equal(down_s[live], rs)
        from b12x.integration.cuteafd._fp8_moe_kernels import META_HEAD
        meta = scratch[:regions[0]].view(torch.int32)
        tile_count = meta[1].item()
        experts = meta[META_HEAD+3*e:META_HEAD+3*e+tile_count].repeat_interleave(16)
        gu, down = region(5, torch.bfloat16, 2*i), region(9, torch.bfloat16, h)
        differences = [0, 0, 0]
        stage_error = torch.zeros(3, dtype=torch.float64, device=x.device)
        stage_official = torch.zeros_like(stage_error)
        stage_ulp = torch.zeros(3, dtype=torch.int32, device=x.device)
        def check_stage(index, got, want, q, scale, weight, weight_scale):
            oracle = project_quantized(q, scale, weight, weight_scale)
            stage_error[index] += (got.double() - oracle.double()).square().sum()
            stage_official[index] += (want.double() - oracle.double()).square().sum()
            stage_ulp[index] = torch.maximum(stage_ulp[index],
                (ordered_bf16(got) - ordered_bf16(want)).abs().max())
        official_gu = gu.clone()
        for expert in range(e):
            positions = torch.where((experts == expert) & live[:tile_count*16])[0]
            if not positions.numel():
                continue
            # Quantizer equality above makes these identical official operands.
            for half, wi in ((0, 0), (1, 2)):
                want = official_module().fp8_gemm(candidate_q[positions].contiguous(),
                    candidate_s[positions].contiguous(), ws[wi][expert], ws[wi+1][expert])
                differences[half] += (gu[positions, half*i:(half+1)*i] != want).sum().item()
                check_stage(half, gu[positions, half*i:(half+1)*i], want,
                    candidate_q[positions], candidate_s[positions], ws[wi][expert], ws[wi+1][expert])
                official_gu[positions, half*i:(half+1)*i] = want
            want = official_module().fp8_gemm(down_q[positions].contiguous(),
                down_s[positions].contiguous(), ws[4][expert], ws[5][expert])
            differences[2] += (down[positions] != want).sum().item()
            check_stage(2, down[positions], want, down_q[positions], down_s[positions],
                ws[4][expert], ws[5][expert])
        print(f"stage quant_input=exact quant_down=exact projection_differences={differences}")
        print(f"stage_gate tp={tp} rows={rows} native_sse={stage_error.tolist()} "
              f"official_sse={stage_official.tolist()} max_ulp={stage_ulp.tolist()}")
        # SM121's official kernel widens E4M3 exactly and uses K16 MMA; our
        # native K32 MMA associates the same operands differently. A gate/up
        # BF16 tie can change the next quantizer's block scale and amplify
        # end-to-end error. Gate each projection on identical operands instead.
        if sm121:
            assert torch.all(stage_error <= stage_official * 1.01)
            assert torch.all(stage_ulp <= 8)
        else:
            assert differences == [0, 0, 0]
    if rows == 640:
        import cutlass
        from cutlass import cute
        from cutlass.cute.runtime import from_dlpack
        from cuda.bindings import driver as cuda
        from b12x.integration.cuteafd._fp8_moe_a8 import QuantizeK128, GroupedK128Gemm
        from b12x.integration.cuteafd._fp8_moe_kernels import MoeSwiGLU, MoeCombine
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        def run_kernel(kernel, tensors, scalars):
            args = [from_dlpack(t).iterator for t in tensors]
            compiled = cute.compile(kernel, *args, *[cutlass.Int32(v) for v in scalars], stream)
            compiled(*args, *scalars, stream)
        run_kernel(MoeSwiGLU(inter=i, limit=7, activation=g.activation), [official_gu, meta, act], [grouped])
        # QuantizeK128 has its row scalar before the unused gather pointer.
        args = [from_dlpack(t).iterator for t in (act, down_q, down_s)]
        pp = from_dlpack(pair_rows).iterator
        quant = cute.compile(QuantizeK128(i), *args, cutlass.Int32(grouped), pp, stream)
        quant(*args, grouped, pp, stream)
        run_kernel(GroupedK128Gemm(h, i, e), [down_q, down_s, meta, ws[4], ws[5], ws[4], ws[5], down], [tile_count])
        injected = torch.empty_like(x)
        pair_pos = scratch[offsets[2]:offsets[2]+regions[2]].view(torch.int32)
        run_kernel(MoeCombine(hidden=h, top_k=topk), [down, pair_pos, routes, injected], [rows])
    expected, fp64 = reference(x, ids, routes, ws, "official"), reference(x, ids, routes, ws, True)
    if rows == 640:
        injected_error = (injected.float() - fp64.float()).square().sum()
        positions = (actual.float() - expected.float()).square().sum(1).topk(3).indices
        print(f"injected_official_gu sse={injected_error.item():.9g} "
              f"mismatches={(injected != expected).sum().item()} worst_rows={positions.tolist()} "
              f"native_row_sse={(actual[positions].float()-fp64[positions].float()).square().sum(1).tolist()} "
              f"official_row_sse={(expected[positions].float()-fp64[positions].float()).square().sum(1).tolist()}")
    error = (actual.float() - fp64.float()).square().sum(1)
    official = (expected.float() - fp64.float()).square().sum(1)
    over = error > official * 1.01
    ratio = torch.where(official > 0, error / official, torch.where(error == 0, 0., float("inf")))
    ulp = (ordered_bf16(actual) - ordered_bf16(expected)).abs()
    print(f"tp={tp} rows={rows} native_error={error.sum().item():.9g} official_error={official.sum().item():.9g} "
          f"rows_over={over.sum().item()} worst_ratio={ratio.max().item()} max_ulp={ulp.max().item()}")
    # End-to-end error is informational on SM121; replay and width stay exact.
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
    if not sm121:
        assert torch.equal(actual, expected), "SM120 official arithmetic must be bitwise"
        assert torch.all(error <= official * 1.01)


@pytest.mark.parametrize("k", [1, 127, 128, 129, 255, 384])
def test_quantizer_live_tail(k):
    require_b12x()
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from b12x.integration.cuteafd._fp8_moe_a8 import QuantizeK128
    torch.manual_seed(100 + k)
    x = torch.randn(7, k, device="cuda").bfloat16()
    x[0].zero_()
    x[1] *= 1e-12
    x[2] *= 1e3
    x[3].fill_(-0.)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((7, (k+127)//128), device="cuda", dtype=torch.float32)
    dummy = torch.zeros(7, device="cuda", dtype=torch.int32)
    xp, qp, sp, dp = [from_dlpack(t).iterator for t in (x, q, scales, dummy)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = cute.compile(QuantizeK128(k), xp, qp, sp, cutlass.Int32(7), dp, stream)
    kernel(xp, qp, sp, 7, dp, stream)
    expected, expected_scales = official_module().act_quant(x)
    assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))
