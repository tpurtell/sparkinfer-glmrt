"""Bounded stage isolation for the seeded C2 wide-prefill accuracy gate."""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch


def test_seeded_prefill_rounding_stages():
    if os.environ.get("CUTEAFD_KIMI_W8_GPU") != "1":
        pytest.skip("explicit CUDA diagnostic required")
    from b12x.integration.cuteafd.kimi_residual import (
        KimiAttnResRmsNorm, compile_kimi_attn_res_rmsnorm_aot, compile_kimi_rmsnorm_aot)
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    source = Path(os.environ["CUTEAFD_W8_HARNESS"])
    spec = importlib.util.spec_from_file_location("w8_harness", source)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    torch.set_num_threads(16)
    rows, hidden, capacity = 4096, 7168, 4099
    arena, delta, norm, proj, following = h.inputs(rows, 2, "normal")
    original = arena.clone()
    prefix = arena[8, :rows] + delta
    v = torch.stack((arena[0, :rows], prefix), 1).float()
    weight = norm.float() * proj.float()
    rms = torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-5)
    scores = (v * rms * weight).sum(-1)
    probs = scores.softmax(-1)
    mixed_raw = torch.matmul(probs[:, None], v).squeeze(1)
    mixed = mixed_raw.bfloat16()
    inv = torch.rsqrt(mixed.float().square().mean(-1, keepdim=True) + 1e-5)
    normalized_raw = mixed.float() * inv
    normalized = normalized_raw.bfloat16()
    baseline = normalized * following
    oracle = h.residual_oracle(v, weight, following, 1e-5)
    actual = torch.empty_like(prefix)
    program = compile_kimi_attn_res_rmsnorm_aot(7168, 2, "add", False, True)
    program.launch(arena, delta, weight, following, actual, scalars=(rows, capacity, 1e-5))
    assert torch.equal(arena[8, :rows], prefix), "prefix addition boundary differs"
    trace = torch.zeros(5, capacity, hidden, device="cuda", dtype=torch.float32)
    diagnostic = compile_program(KimiAttnResRmsNorm(7168, 2, "add", False, True, diagnostic=True),
        name="kimi_residual_stage_diagnostic", operands=(
            Operand("arena", torch.bfloat16, "[9,arena_rows,7168]", "inout"),
            Operand("delta", torch.bfloat16, "[rows,7168]"),
            Operand("score_weight", torch.float32, "[7168]"),
            Operand("norm_weight", torch.bfloat16, "[7168]"),
            Operand("trace", torch.float32, "[5,arena_rows,7168]", "out")),
        scalars=(Scalar("rows"), Scalar("arena_rows"), Scalar("eps", "float32")), key=(2,))
    arena.copy_(original)
    diagnostic.launch(arena, delta, weight, following, trace, scalars=(rows, capacity, 1e-5))
    norm_only = torch.empty_like(prefix)
    compile_kimi_rmsnorm_aot(7168).launch(mixed, following, norm_only, scalars=(rows, 1e-5))
    a, b, o = actual.cpu().double(), baseline.cpu().double(), oracle.double()
    ae, be = (a-o).square().sum(-1), (b-o).square().sum(-1)
    failing = torch.nonzero(ae > be * 1.01**2).flatten().tolist()
    print("FAILING", failing, "max_l2_ratio", (ae/be).sqrt().max().item(), flush=True)
    print("diagnostic_output_equal", torch.equal(trace[0,:rows].bfloat16(), actual), flush=True)
    stages = dict(score_rsqrt=(trace[4,:rows,2:4], rms.squeeze(-1)),
                  scores=(trace[4,:rows,4:6],scores), probs=(trace[4,:rows,6:8],probs),
                  mixed_raw=(trace[1,:rows],mixed_raw), mixed_bf16=(trace[1,:rows].bfloat16().float(),mixed.float()),
                  norm_rsqrt=(trace[4,:rows,:1],inv), normalized_raw=(trace[2,:rows],normalized_raw),
                  normalized_bf16=(trace[3,:rows],normalized.float()), final=(actual.float(),baseline.float()))
    records=[]
    for row in failing[:10]:
        rec=dict(row=row,candidate_sse=ae[row].item(),official_sse=be[row].item(),stages={})
        for name,(got,ref) in stages.items():
            g,r=got[row].float().cpu(),ref[row].float().cpu()
            ulp=(g.view(torch.int32).to(torch.int64)-r.view(torch.int32).to(torch.int64)).abs()
            rec["stages"][name]=dict(max_abs=(g-r).abs().max().item(),max_ulp=ulp.max().item(),different=(g!=r).sum().item())
        rec["norm_only_sse"]=(norm_only[row].cpu().double()-o[row]).square().sum().item()
        print(json.dumps(rec),flush=True)
        records.append(rec)
    output=Path(os.environ["CUTEAFD_W8_DIAGNOSTIC_OUT"])
    output.mkdir(parents=True,exist_ok=True)
    (output/"stages.json").write_text(json.dumps(records,indent=2))
    ids=torch.tensor(failing[:10],device="cuda",dtype=torch.long)
    torch.save(dict(rows=failing[:10],prefix=prefix[ids].cpu(),values=v[ids].cpu(),weight=weight.cpu(),following=following.cpu(),
        oracle=oracle[failing[:10]],stages={k:(g[ids].cpu(),r[ids].cpu()) for k,(g,r) in stages.items()}),output/"stages.pt")
    assert torch.equal(trace[0,:rows].bfloat16(),actual), "instrumentation changed arithmetic"
