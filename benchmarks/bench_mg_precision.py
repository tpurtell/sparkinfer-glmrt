"""MG precision diagnostic: unchanged kernels, native packing, FP32 attention oracle.

DSV4 584B changes QK only (PV is FP8 NoPE / BF16 RoPE). Native DSV41
changes canonical KV staging and PV only; both arms retain BF16 QK.
Synthetic full selected sets model long-context attention, not a model forward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits


def arm_traits(fmt, arm):
    if fmt == "dsv4":
        if arm not in ("fp8-qk", "bf16-qk"):
            raise ValueError(arm)
        return make_unified_traits(ModelType.DSV4,
            ComputeMode.FP8 if arm == "fp8-qk" else ComputeMode.BF16,
            ScaleFormat.UE8M0_BYTE, fp8_rope=False)
    if fmt != "dsv41" or arm not in ("bf16-pv", "fp8-pv"):
        raise ValueError((fmt, arm))
    return replace(make_unified_traits(ModelType.DSV41, ComputeMode.BF16,
        ScaleFormat.NVFP4_E4M3, fp8_rope=False), fp8_internal=arm == "fp8-pv")


def parse_args(argv=None):
    def ints(s):
        try:
            v = [int(x) for x in s.split(",")]
        except ValueError as e:
            raise argparse.ArgumentTypeError("expected comma-separated positive integers") from e
        if not v or min(v) < 1:
            raise argparse.ArgumentTypeError("counts must be positive")
        return v
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=("dsv4", "dsv41"), required=True)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--indexed-width", type=int, default=512)
    p.add_argument("--rows", type=ints, default=ints("1,64,2048,2752,4096"))
    p.add_argument("--timing-rows", type=ints, default=ints("64,1536,2048,2752,4096"))
    p.add_argument("--scores", default="spread,peaked")
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=41512)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.scores = args.scores.split(",")
    if set(args.scores) - {"spread", "peaked"}:
        p.error("scores must be spread,peaked")
    if args.heads < 16 or args.heads % 16 or min(args.indexed_width, args.samples, args.replays, args.warmup) < 1:
        p.error("heads must be a positive multiple of 16; counts must be positive")
    if args.indexed_width > 4096:
        p.error("indexed width exceeds the 4096-key synthetic pool")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from b12x.attention._shared.mla.compressed_reference import (
        pack_compressed_sparse_mla_kv_cache_reference, pack_deepseek_v41_cache_reference,
        unpack_compressed_sparse_mla_kv_cache_reference, unpack_deepseek_v41_cache_reference)
    from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
    from b12x.attention._shared.mla.prefill import _mg_head_partitions

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    max_rows = max(args.rows + args.timing_rows)
    pool, window = 4096, 128
    # Native pages: DSV4 main 256 / C4 64; V4.1 main 32 / C1 256.
    main_page, extra_page = (256, 64) if args.format == "dsv4" else (32, 256)
    scales = torch.linspace(.3, 1.2, 512, device=device)
    values = [(torch.randn(pool, 512, device=device) * scales).bfloat16() for _ in range(2)]
    def pack(v, page, kind):
        if args.format == "dsv41":
            c = pack_deepseek_v41_cache_reference(v, page_size=page, cache_kind=kind)
            k = unpack_deepseek_v41_cache_reference(c, page_size=page, cache_kind=kind)
            return c, k, k
        c = pack_compressed_sparse_mla_kv_cache_reference(v[:, :448], v[:, 448:], page_size=page)
        k, val = unpack_compressed_sparse_mla_kv_cache_reference(c, page_size=page)
        return c, k, val
    main_cache, mk, mv = pack(values[0], main_page, "swa")
    extra_cache, ek, ev = pack(values[1], extra_page, "indexed")
    row = torch.arange(max_rows, device=device)[:, None]
    mi = ((row + torch.arange(window, device=device)[None]) % pool).int().contiguous()
    ei = ((row * 37 + torch.arange(args.indexed_width, device=device)[None] * 17) % pool).int().contiguous()
    ml = torch.full((max_rows,), window, device=device, dtype=torch.int32)
    el = torch.full((max_rows,), args.indexed_width, device=device, dtype=torch.int32)
    sink = torch.linspace(-1, 1, args.heads, device=device)
    arms = ("fp8-qk", "bf16-qk") if args.format == "dsv4" else ("bf16-pv", "fp8-pv")
    report = {"format": args.format, "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "fork_sha": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip(),
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "traits": {arm: asdict(arm_traits(args.format, arm)) for arm in arms}, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pattern in args.scores:
        q = (torch.randn(max_rows, args.heads, 512, device=device) * (.03 if pattern == "spread" else .2)).bfloat16()
        if pattern == "peaked":
            target = ek[ei[:, 0].long()]
            q = (q.float() + target[:, None] * (14 * 512**.5 / target.square().sum(-1))[:, None, None]).bfloat16()
        out = {arm: torch.empty_like(q) for arm in arms}
        lse = {arm: torch.empty((max_rows, args.heads), device=device) for arm in arms}
        def run(arm, rows):
            for n_hg, active_heads, offset in _mg_head_partitions(args.heads, 16):
                run_unified_prefill_mg(q=q[:rows], kv_cache=main_cache, topk_indices=mi[:rows],
                    topk_length=ml[:rows], sm_scale=512**-.5, page_block_size=main_page,
                    attn_sink=sink, output=out[arm][:rows], lse_out=lse[arm][:rows],
                    extra_kv_cache=extra_cache, extra_indices=ei[:rows], extra_topk_length=el[:rows],
                    extra_page_block_size=extra_page, mg_n_hg=n_hg, active_heads=active_heads,
                    head_offset=offset, traits_override=arm_traits(args.format, arm))
        # One FP32 oracle over the exact decoded cache; no cast back to BF16.
        ref = torch.empty_like(q, dtype=torch.float32)
        ref_lse = torch.empty_like(lse[arms[0]])
        pmax, entropy, tiny = [], [], []
        for lo in range(0, max_rows, 16):
            hi = min(lo + 16, max_rows)
            k = torch.cat((mk[mi[lo:hi].long()], ek[ei[lo:hi].long()]), 1)
            v = torch.cat((mv[mi[lo:hi].long()], ev[ei[lo:hi].long()]), 1)
            scores = torch.bmm(q[lo:hi].float(), k.transpose(1,2)) * 512**-.5
            all_scores = torch.cat((scores, sink[None,:,None].expand(hi-lo,-1,-1)), -1)
            probs = torch.softmax(all_scores, -1)
            ref[lo:hi] = torch.bmm(probs[:,:,:-1], v)
            ref_lse[lo:hi] = torch.logsumexp(all_scores, -1)
            pmax.append(probs.max(-1).values.mean())
            entropy.append(-(probs * probs.clamp_min(1e-30).log()).sum(-1).mean())
            tiny.append((probs[:,:,:-1] < 2**-9).float().mean())
        distribution = {"mean_pmax": float(torch.stack(pmax).mean()),
            "mean_entropy_nats": float(torch.stack(entropy).mean()),
            "probability_fraction_below_e4m3_min": float(torch.stack(tiny).mean())}
        for rows in sorted(set(args.rows + args.timing_rows)):
            record = {"scores": pattern, "rows": rows, "distribution": distribution, "arms": {}}
            graphs = {}
            try:
                for arm in arms:
                    run(arm, rows)
                    torch.cuda.synchronize()
                    delta = out[arm][:rows].float() - ref[:rows]
                    ld = lse[arm][:rows] - ref_lse[:rows]
                    if not bool(torch.isfinite(delta).all() & torch.isfinite(ld).all()):
                        raise RuntimeError(f"nonfinite result: {arm}/{rows}/{pattern}")
                    record["arms"][arm] = {"output_rel_rms": float(delta.square().mean().sqrt() / ref[:rows].square().mean().sqrt()),
                        "output_max_abs": float(delta.abs().max()), "lse_max_abs": float(ld.abs().max()),
                        "lse_rms": float(ld.square().mean().sqrt())}
                    if rows in args.timing_rows:
                        for _ in range(args.warmup):
                            run(arm, rows)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run(arm, rows)
                        graphs[arm] = graph
                        graph.replay()
                torch.cuda.synchronize()
                samples = {arm: [] for arm in arms}
                for sample in range(args.samples if graphs else 0):
                    for arm in arms[::1 if sample % 2 == 0 else -1]:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.replays):
                            graphs[arm].replay()
                        end.record(); end.synchronize()
                        samples[arm].append(start.elapsed_time(end) * 1000 / args.replays)
                for arm in graphs:
                    record["arms"][arm].update(samples_us=samples[arm], median_us=statistics.median(samples[arm]))
                if graphs:
                    record["second_over_first"] = record["arms"][arms[1]]["median_us"] / record["arms"][arms[0]]["median_us"]
            finally:
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
