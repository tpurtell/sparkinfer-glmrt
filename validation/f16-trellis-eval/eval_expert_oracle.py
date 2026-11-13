#!/usr/bin/env python3
"""EXL3 routed-expert (mixed_trellis) F64 oracle: shipped F16 arm vs emulated BF16 arm.

The shipped expert kernel has no BF16 MMA arm (full_rotation requires
element_dtype="fp16"; every scratch plane is F16). So:
  kernel_f16  : the shipped compile_mixed_trellis kernel, real family geometry,
                real expert count, random top-k routes, official-encoder weights.
  emu(dtype)  : torch replay of the kernel's rounding chain with each storage /
                MMA-operand point rounded to dtype (f16 must reproduce the kernel;
                bf16 is the hypothetical BF16 arm).
  f64_q       : same chain in F64 with the quantized weights, no rounding.
  f64_src     : same chain in F64 with the encoder's unquantized source weights.
Ranges: max |x|, |x*suh|, |rotated in|, |gate/up FC1 out|, |act|, |down in| and
|FC2 out| in the F16 chain, so the F16 headroom is visible.
Usage: eval_expert_oracle.py OUT.json FAMILY[:bits0,bits1[:scale]] ...
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/cuteafd/third_party/sparkinfer")
sys.path.insert(0, "/workspace/cuteafd/third_party/sparkinfer/tests/moe")
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (  # noqa: E402
    bind_mixed_trellis, build_tiered_maps, combine_trellis_rotations,
    compile_mixed_trellis, make_mixed_trellis_buffers, run_bound_mixed_trellis)
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights  # noqa: E402
from b12x.moe._shared.kernels.activations import SITU_DEFAULT_BETA, SITU_DEFAULT_LINEAR_BETA  # noqa: E402
from b12x.moe.fused_moe._impl import _projection_mixed_direct_topk_routes, _projection_mixed_tile_config  # noqa: E402
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity  # noqa: E402
from test_fused_moe_trellis import _reconstruct_native  # noqa: E402

ENCODER = "/opt/cuteafd-target/enc/trellis_bench_encoder.so"
# (hidden, intermediate, experts, topk, activation, swiglu_limit) from package_exl3_aot.GEOMETRIES.
FAMILIES = {
    "glm": (6144, 2048, 256, 8, "silu", None),
    "kimik3": (3584, 3072, 896, 16, "situ", None),
    "step5": (4096, 1536, 352, 8, "silu_post_clamp", 7.0),
    "glmf": (4096, 2048, 288, 8, "silu", 10.0),
    "ml4": (6144, 6144, 128, 4, "silu", None),
}
ROWS = [1, 8, 64, 2048]
OUTLIER = 0.0
SUH = 0.02  # real checkpoints: suh rms 0.014-0.029, svh rms ~1.0 (scan_rotations.py)
POOL = 4  # unique encoded weight sets per tier; experts reuse them cyclically
PERM = []
for lane in range(32):
    r, c = 2 * (lane % 4), lane // 4
    PERM.extend((r*16+c, (r+1)*16+c, (r+8)*16+c, (r+9)*16+c,
                 r*16+c+8, (r+1)*16+c+8, (r+8)*16+c+8, (r+9)*16+c+8))


def hadamard(device):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / 128 ** 0.5).to(device)


class Encoder:
    def __init__(self, bits, scale):
        self.bits, self.scale = bits, scale
        self.costs = torch.empty((256, 2, 65536 >> bits), device="cuda", dtype=torch.float16)
        self.edges = torch.empty((256, 256, 65536 >> bits), device="cuda", dtype=torch.int16)
        self.inverse = torch.argsort(torch.tensor(PERM, device="cuda"))

    def projection(self, k, n):
        """Native payload [K/16,N/16,16*bits] i16 plus W_src / W_q as [K,N] f16-exact f32 tiles."""
        tiles = torch.randn((k * n // 256, 256), device="cuda") * self.scale
        quant = torch.empty_like(tiles)
        idx = torch.empty_like(tiles, dtype=torch.int16)
        torch.ops.trellis_bench_encoder.quantize_tiles(tiles, quant, idx, self.costs, self.edges, self.bits, True, False)
        packed = torch.empty((k // 16, n // 16, 16 * self.bits), device="cuda", dtype=torch.int16)
        torch.ops.trellis_bench_encoder.pack_trellis(packed, idx.view(k // 16, n // 16, 256), self.bits)
        to_kn = lambda t: t[:, self.inverse].view(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)
        return packed, to_kn(tiles).half(), to_kn(quant).half()


def emulate(x, ids, pids, wts, W, rot, dt, act, limit, H):
    """Kernel rounding chain (mixed_trellis.py cooperative path) with storage dtype dt.

    dt=None runs everything in F64 (oracle). W: dict of [POOL*tiers,K,N]; pids maps routes to them.
    """
    f64 = dt is None
    rd = (lambda t: t) if f64 else (lambda t: t.to(dt).to(torch.float64 if f64 else torch.float32))
    work = torch.float64 if f64 else torch.float32
    rows, topk = ids.shape
    out = torch.zeros(rows, x.shape[1], dtype=work, device=x.device)
    flat = ids.reshape(-1).long()
    pflat = pids.reshape(-1).long()
    tok = torch.arange(rows, device=x.device).repeat_interleave(topk)
    xin = x.to(work)[tok]
    if not f64:
        xin = rd(xin)  # F16 arm: BF16 row -> F16 -> F32 (mixed_trellis.py:1105-1106)
    I = W["down"].shape[1]  # down is [POOL, K=I, N=H]
    Hm = H.to(work)

    def had(v):
        r, w = v.shape
        return (v.view(r, w // 128, 128) @ Hm).view(r, w)

    rng = {}
    outs = {}
    for p in ("gate", "up"):
        s = rot[f"{p}_suh"][flat].to(work)
        v = rd(xin * s)                                            # x*suh -> F16 (1115-1119)
        a = rd(had(v))                                             # F32 butterfly, F16 store
        y = grouped(a, W[p], pflat, work)                          # MMA, F32 accumulate
        y = rd(y)                                                  # fc1 plane F16
        rng[f"{p}_in"], rng[f"{p}_out"] = float(a.abs().max()), float(y.abs().max())
        outs[p] = y
    inter = rot["intermediate"][flat].to(work)
    g = had(outs["gate"]) * inter[:, :I]
    u = had(outs["up"]) * inter[:, I:2 * I]
    rng["gate_logical"], rng["up_logical"] = float(g.abs().max()), float(u.abs().max())
    if act == "situ":
        h = SITU_DEFAULT_BETA * torch.tanh(g / SITU_DEFAULT_BETA) * torch.sigmoid(g) * \
            SITU_DEFAULT_LINEAR_BETA * torch.tanh(u / SITU_DEFAULT_LINEAR_BETA)
    elif act == "silu_post_clamp":
        h = (g * torch.sigmoid(g)).clamp(max=limit) * u.clamp(-limit, limit)
    else:
        if limit is not None:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        h = g * torch.sigmoid(g) * u
    rng["act"] = float(h.abs().max())
    a2 = rd(had(h * inter[:, 2 * I:]))                             # activated plane F16
    rng["down_in"] = float(a2.abs().max())
    y2 = rd(grouped(a2, W["down"], pflat, work))                  # fc2 plane F16
    rng["down_out"] = float(y2.abs().max())
    route = had(y2) * rot["down_svh"][flat].to(work)
    out.index_add_(0, tok, route * wts.reshape(-1, 1).to(work))
    return out, rng


def grouped(a, W, pflat, work):
    out = torch.empty(a.shape[0], W.shape[2], dtype=work, device=a.device)
    for m in torch.unique(pflat).tolist():
        sel = pflat == m
        out[sel] = a[sel] @ W[m].to(work)
    return out


def rel(a, b):
    return float(torch.linalg.norm(a.double() - b.double()) / torch.linalg.norm(b.double()))


def run_family(name, bits, scale, xscale):
    hidden, inter, experts, topk, act, limit = FAMILIES[name]
    torch.manual_seed(1009)
    dev = torch.device("cuda", 0)
    enc = {b: Encoder(b, scale) for b in set(bits)}
    split = experts // 2
    tiers, W = [], {"gate": [], "up": [], "down": []}
    pool = {}  # combined expert index -> (tier, pool member)
    for t, b in enumerate(bits):
        ne = split if t == 0 else experts - split
        w13 = torch.empty((2, ne, hidden // 16, inter // 16, 16 * b), device=dev, dtype=torch.int16)
        w2 = torch.empty((ne, inter // 16, hidden // 16, 16 * b), device=dev, dtype=torch.int16)
        wq = {"gate": [], "up": [], "down": []}
        ws = {"gate": [], "up": [], "down": []}
        for m in range(POOL):
            for j, p in enumerate(("gate", "up")):
                pk, src, q = enc[b].projection(hidden, inter)
                w13[j, m::POOL] = pk
                wq[p].append(q), ws[p].append(src)
            pk, src, q = enc[b].projection(inter, hidden)
            w2[m::POOL] = pk
            wq["down"].append(q), ws["down"].append(src)
        for e in range(ne):
            pool[(0 if t == 0 else split) + e] = t * POOL + e % POOL
        if t == 0:  # layout cross-check: independent decoder of one stored tile vs W_q
            chk = torch.as_tensor(_reconstruct_native(w13[0, 0, :1, :1].contiguous())).float()
            if not torch.equal(chk.cuda(), wq["gate"][0][:16, :16].float()):
                raise RuntimeError("encoder W_q layout disagrees with the kernel's tile decoder")
        gen = lambda shape: (0.875 + 0.25 * torch.rand(shape, device=dev))
        sign = lambda shape: torch.where(torch.rand(shape, device=dev) < .5, -1.0, 1.0)
        su = lambda shape, rms: (gen(shape) * sign(shape) * rms).half()
        inter_rot = torch.cat([su((ne, inter), 1.0), su((ne, inter), 1.0), su((ne, inter), SUH)], 1)
        rots = dict(gate_suh=su((ne, hidden), SUH), up_suh=su((ne, hidden), SUH),
                    intermediate=inter_rot, down_svh=su((ne, hidden), 1.0))
        prep = prepare_trellis256_moe_weights(
            w13, w2, hidden_size=hidden, intermediate_size=inter, num_experts=ne, activation=act,
            fc1_tile_n=128, fc2_tile_n=128, params_dtype=torch.float16, w13_layout="trellis_t256_proj",
            trellis_bits=b, codebook="mcg", gate_suh=rots["gate_suh"], up_suh=rots["up_suh"],
            intermediate_rotations=rots["intermediate"], down_svh=rots["down_svh"],
            tile_config=(128, 128, 128, 128))
        tiers.append((prep, rots, wq, ws))
    rot = {k: torch.cat([t[1][k] for t in tiers]) for k in tiers[0][1]}
    Wq = {p: torch.stack(sum([t[2][p] for t in tiers], [])) for p in W}
    Ws = {p: torch.stack(sum([t[3][p] for t in tiers], [])) for p in W}
    # Global expert ids interleave tiers (as shipped); combined namespace is tier-ordered.
    perm = torch.randperm(experts).tolist()
    t0_ids, t1_ids = sorted(perm[:split]), sorted(perm[split:])
    g2c, desc = build_tiered_maps(t0_ids, t1_ids, device=dev)
    rotations = combine_trellis_rotations(tiers[0][0], tiers[1][0])
    combined_of = {g: i for i, g in enumerate(t0_ids + t1_ids)}
    props = torch.cuda.get_device_properties(dev)
    H = hadamard(dev)
    pool_of = torch.tensor([pool[i] for i in range(experts)], device=dev)
    report = dict(family=name, hidden=hidden, intermediate=inter, experts=experts, topk=topk,
                  activation=act, swiglu_limit=limit, bits=bits, weight_scale=scale, x_scale=xscale, outlier=OUTLIER, suh_rms=SUH,
                  max_abs_weight=max(float(Wq[p].abs().max()) for p in Wq), records=[])
    for rows in ROWS:
        cap = rows
        direct = _projection_mixed_direct_topk_routes(cap, topk, direct_exl3=True)
        tile = _projection_mixed_tile_config(None, hidden_size=hidden, intermediate_size=inter,
                                             token_count=cap, direct_topk_routes=direct)
        blocks = cap * topk if direct else (route_pack_capacity(cap * topk, 8, experts, topk=topk)[1] + 7) // 8
        launch = compile_mixed_trellis(
            size_m=cap, hidden_size=hidden, intermediate_size=inter, tier0_num_experts=split,
            tier1_num_experts=experts - split, top_k=topk, route_num_experts=experts, max_m_blocks=blocks,
            sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
            force_tile_config=tile, tier0_bits=bits[0], tier1_bits=bits[1], trellis_codebook="mcg",
            swiglu_limit=limit, activation=act, moe_block_size=8, rotation_input_dtype="bf16",
            full_rotation_output_dtype="fp32", route_ids_dtype=torch.int32,
            direct_topk_routes=direct)
        binding = bind_mixed_trellis(tiers[0][0], tiers[1][0], g2c, desc, rotations, launch)
        buffers = make_mixed_trellis_buffers(launch, device=dev, sms=props.multi_processor_count)
        x = torch.randn(rows, hidden, device=dev) * xscale
        if OUTLIER:
            ch = torch.randperm(hidden, device=dev)[:max(1, hidden // 512)]
            x[:, ch] *= OUTLIER / xscale
        x = x.bfloat16()
        ids = torch.stack([torch.randperm(experts, device=dev)[:topk] for _ in range(rows)]).int()
        wts = torch.rand(rows, topk, device=dev)
        wts /= wts.sum(1, keepdim=True)
        y = run_bound_mixed_trellis(x, wts, ids, binding, buffers).clone()
        torch.cuda.synchronize()
        cids = torch.tensor([[combined_of[int(g)] for g in r] for r in ids.tolist()], device=dev)
        pids = pool_of[cids]
        emu16, rng = emulate(x, cids, pids, wts, Wq, rot, torch.float16, act, limit, H)
        emub, rngb = emulate(x, cids, pids, wts, Wq, rot, torch.bfloat16, act, limit, H)
        ref_q, _ = emulate(x, cids, pids, wts, Wq, rot, None, act, limit, H)
        ref_s, _ = emulate(x, cids, pids, wts, Ws, rot, None, act, limit, H)
        rec = dict(rows=rows, direct_routes=direct, finite=bool(torch.isfinite(y).all()), quant=rel(ref_q, ref_s),
                   kernel_f16=dict(vs_q=rel(y, ref_q), total=rel(y, ref_s), vs_emu_f16=rel(y, emu16)),
                   emu_f16=dict(vs_q=rel(emu16, ref_q), total=rel(emu16, ref_s)),
                   emu_bf16=dict(vs_q=rel(emub, ref_q), total=rel(emub, ref_s)),
                   max_abs=dict(x=float(x.float().abs().max()), **rng))
        rec["f16_reduction_share_of_total"] = (rec["emu_bf16"]["vs_q"] - rec["kernel_f16"]["vs_q"]) / rec["emu_bf16"]["total"]
        report["records"].append(rec)
        print(json.dumps(dict(family=name, **rec)), flush=True)
        del binding, buffers, launch
        torch.cuda.empty_cache()
    return report


def main():
    out = sys.argv[1]
    torch.ops.load_library(ENCODER)
    results = []
    for spec in sys.argv[2:]:
        parts = spec.split(":")
        bits = tuple(map(int, parts[1].split(","))) if len(parts) > 1 else (3, 4)
        scale = float(parts[2]) if len(parts) > 2 else 1.0
        xscale = float(parts[3]) if len(parts) > 3 else 1.0
        global OUTLIER
        OUTLIER = float(parts[4]) if len(parts) > 4 else 0.0
        results.append(run_family(parts[0], bits, scale, xscale))
        with open(out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
