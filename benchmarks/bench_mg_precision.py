"""MG precision diagnostic: unchanged kernels, native packing, FP32 attention oracle.

DSV4 584B changes QK only (PV is FP8 NoPE / BF16 RoPE). Native DSV41
changes canonical KV staging and PV only; both arms retain BF16 QK.
Inline-FP32-scale records (glm 656B, glmf 528B, kimi 656B with 96 or 48 heads
and 192^-0.5) compare FP8 QK/PV with BF16 QK/PV (exact keys); they also check
ragged, empty and masked rows, slot-0 poison and row invariance.
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


INLINE = {  # heads, absorbed query width, selected slots, softmax scale
    "glm": (64, 576, 2048, 256 ** -.5),
    "glmf": (64, 512, 2112, 256 ** -.5),
    "kimi": (96, 576, 2048, 192 ** -.5),
}


def fork_sha():
    """Source revision; a submodule checkout inside a container has no reachable gitdir."""
    import os
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("B12X_SOURCE_REVISION", "unknown")


def arm_traits(fmt, arm):
    if fmt in INLINE:
        from b12x.attention._shared.mla.traits import with_bf16_attention
        if arm not in ("fp8", "bf16"):
            raise ValueError((fmt, arm))
        traits = make_unified_traits(ModelType.GLM_NEXT if fmt == "glmf" else ModelType.GLM_NSA,
            ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)
        return with_bf16_attention(traits) if arm == "bf16" else traits
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
    p.add_argument("--format", choices=("dsv4", "dsv41", *INLINE), required=True)
    p.add_argument("--heads", type=int)
    p.add_argument("--selected", type=int, help="inline formats: valid slots per row (default: all)")
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
    if args.heads is None:
        args.heads = INLINE[args.format][0] if args.format in INLINE else 64
    if args.format in INLINE and not 1 <= (args.selected or 1) <= INLINE[args.format][2]:
        p.error("selected exceeds the format's slots per row")
    if set(args.scores) - {"spread", "peaked", "medium"}:
        p.error("scores must be spread, peaked or medium")
    if "medium" in args.scores and args.format not in INLINE:
        p.error("medium scores are measured on the inline-FP32 formats")
    if args.heads < 16 or args.heads % 16 or min(args.indexed_width, args.samples, args.replays, args.warmup) < 1:
        p.error("heads must be a positive multiple of 16; counts must be positive")
    if args.indexed_width > 4096:
        p.error("indexed width exceeds the 4096-key synthetic pool")
    return args


def measure(args, report, arms, patterns):
    """Accuracy against each pattern's FP32 oracle, then interleaved warm graph timings."""
    import torch
    for pattern, prepare in patterns:
        run, out, lse, ref, ref_lse, distribution = prepare()
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
                    record["over_first"] = {arm: record["arms"][arm]["median_us"] / record["arms"][arms[0]]["median_us"]
                                            for arm in arms[1:]}
            finally:
                torch.cuda.synchronize()
                for graph in graphs.values():
                    graph.reset()
            report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)


def inline_main(args):
    """glm / glmf / kimi: one cache of inline-FP32-scale records, 64 per page."""
    import torch
    from b12x.attention._shared.mla.prefill import _mg_head_partitions
    from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
    from b12x.attention._shared.mla.smem_mg import max_mg_head_groups

    _, qdim, topk, sm_scale = INLINE[args.format]
    heads, device, pool = args.heads, torch.device("cuda", torch.cuda.current_device()), 8192
    selected = args.selected or topk
    max_rows = max(args.rows + args.timing_rows)
    raw = (torch.randn(pool, 576, device=device) * torch.linspace(.3, 1.2, 576, device=device)).bfloat16()
    cache = pack_mla_kv_cache_reference(raw[:, :512], raw[:, 512:] if qdim == 576 else None)
    keys = unpack_mla_kv_cache_reference(cache).squeeze(1)
    cache = cache.reshape(pool // 64, -1).contiguous()
    row = torch.arange(max_rows, device=device)[:, None]
    indices = (1 + (row * 37 + torch.arange(topk, device=device)[None] * 17) % (pool - 1)).int().contiguous()
    arms = ("fp8", "bf16")
    traits = {arm: arm_traits(args.format, arm) for arm in arms}
    report = {"format": args.format, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "fork_sha": fork_sha(),
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
        "traits": {arm: asdict(t) for arm, t in traits.items()}, "lse_unit": "log2", "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def launch(arm, q, idx, lengths, out, lse):
        for n_hg, active, offset in _mg_head_partitions(heads, 16, max_mg_head_groups(traits[arm])):
            run_unified_prefill_mg(q=q, kv_cache=cache, topk_indices=idx, topk_length=lengths,
                sm_scale=sm_scale, page_block_size=64, output=out, lse_out=lse, mg_n_hg=n_hg,
                active_heads=active, head_offset=offset, traits_override=traits[arm])

    def oracle(q, idx, lengths):
        """FP32 attention over the decoded records; base-2 LSE; empty rows 0 / -inf."""
        out = torch.zeros((q.shape[0], heads, 512), device=device)
        lse = torch.full((q.shape[0], heads), float("-inf"), device=device)
        stats = [[], [], []]
        for lo in range(0, q.shape[0], 8):
            hi = min(lo + 8, q.shape[0])
            k = keys[idx[lo:hi].clamp_min(0).long()]
            scores = torch.bmm(q[lo:hi].float(), k[:, :, :qdim].transpose(1, 2)) * sm_scale
            valid = (torch.arange(topk, device=device)[None] < lengths[lo:hi, None]) & (idx[lo:hi] >= 0)
            scores = scores.masked_fill(~valid[:, None], float("-inf"))
            live = valid.any(-1)
            probs = torch.softmax(scores[live], -1)
            out[lo:hi][live] = torch.bmm(probs, k[live][:, :, :512])
            lse[lo:hi][live] = torch.logsumexp(scores[live], -1) / 0.6931471805599453
            stats[0].append(probs.max(-1).values.mean())
            stats[1].append(-(probs * probs.clamp_min(1e-30).log()).sum(-1).mean())
            stats[2].append(((probs < 2**-9) & valid[live][:, None]).float().sum() / valid[live].sum() / heads)
        return out, lse, stats

    def pattern(name):
        def prepare():
            # medium: tens of keys carry the mass, where score error moves the output most.
            q = (torch.randn(max_rows, heads, qdim, device=device)
                 * {"spread": .03, "medium": 1.6, "peaked": .2}[name]).bfloat16()
            if name == "peaked":
                target = keys[indices[:, 0].long()][:, :qdim]
                q = (q.float() + target[:, None] * (14 / sm_scale / target.square().sum(-1))[:, None, None]).bfloat16()
            lengths = torch.full((max_rows,), selected, device=device, dtype=torch.int32)
            out = {arm: torch.empty((max_rows, heads, 512), device=device, dtype=torch.bfloat16) for arm in arms}
            lse = {arm: torch.empty((max_rows, heads), device=device) for arm in arms}
            ref, ref_lse, stats = oracle(q, indices, lengths)
            distribution = {"selected": selected, "mean_pmax": float(torch.stack(stats[0]).mean()),
                "mean_entropy_nats": float(torch.stack(stats[1]).mean()),
                "probability_fraction_below_e4m3_min": float(torch.stack(stats[2]).mean())}
            run = lambda arm, rows: launch(arm, q[:rows], indices[:rows], lengths[:rows], out[arm][:rows], lse[arm][:rows])
            return run, out, lse, ref, ref_lse, distribution
        return name, prepare

    # Ragged final tiles, empty rows, masked interior slots and slot-0 poison.
    edges = sorted({0, 1, 2, 31, 63, 64, 65, 127, 128, 129, 777, topk - 65, topk - 1, topk})
    n = len(edges)
    q = (torch.randn(n, heads, qdim, device=device) * .2).bfloat16()
    lengths = torch.tensor(edges, device=device, dtype=torch.int32)
    idx = indices[:n].clone()
    idx[:, 5::11] = -1
    ref, ref_lse, _ = oracle(q, idx, lengths)
    checks = {}
    for arm in arms:
        out = torch.full((n, heads, 512), float("nan"), device=device, dtype=torch.bfloat16)
        lse = torch.full((n, heads), float("nan"), device=device)
        launch(arm, q, idx, lengths, out, lse)
        torch.cuda.synchronize()
        live = (ref_lse > float("-inf")).all(-1)
        checks[arm] = {"lengths": edges, "output_max_abs": float((out.float() - ref)[live].abs().max()),
            "lse_max_abs": float((lse - ref_lse)[live].abs().max()),
            "empty_rows_zero": bool((out[~live] == 0).all() and torch.isneginf(lse[~live]).all())}
        saved, first = (out.clone(), lse.clone()), cache[0, :cache.shape[1] // 64].clone()
        cache[0, :cache.shape[1] // 64] = 255
        launch(arm, q, idx, lengths, out, lse)
        torch.cuda.synchronize()
        cache[0, :cache.shape[1] // 64] = first
        checks[arm]["slot0_poison_bytes_equal"] = bool(
            torch.equal(saved[0].view(torch.uint8), out.view(torch.uint8))
            and torch.equal(saved[1].view(torch.uint8), lse.view(torch.uint8)))
        # One row's result does not depend on its position or the live width.
        same_q, same_idx = q[1:2].expand(65, -1, -1).contiguous(), indices[:1].expand(65, -1).contiguous()
        full = torch.full((65,), topk, device=device, dtype=torch.int32)
        canonical, invariant = None, True
        for width in (1, 6, 64, 65):
            out = torch.empty((width, heads, 512), device=device, dtype=torch.bfloat16)
            lse = torch.empty((width, heads), device=device)
            launch(arm, same_q[:width], same_idx[:width], full[:width], out, lse)
            torch.cuda.synchronize()
            canonical = canonical or (out[0].clone(), lse[0].clone())
            invariant &= bool((out.view(torch.int16) == canonical[0].view(torch.int16)[None]).all()
                              and (lse.view(torch.int32) == canonical[1].view(torch.int32)[None]).all())
        checks[arm]["row_invariant"] = invariant
    report["checks"] = checks
    print(json.dumps({"checks": checks}), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    # The FP8 arm launched here does not zero masked V (its exports opt in),
    # so its poison result is recorded only; BF16 PV always zeroes it.
    bad = [f"{arm}.{key}" for arm in arms for key in ("empty_rows_zero", "slot0_poison_bytes_equal", "row_invariant")
           if not checks[arm][key] and (traits[arm].pv_bf16 or key != "slot0_poison_bytes_equal")]
    if bad:
        raise RuntimeError(f"structural checks failed: {bad}")
    measure(args, report, arms, [pattern(name) for name in args.scores])
    return 0


def main(argv=None):
    args = parse_args(argv)
    import torch
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.format in INLINE:
        return inline_main(args)
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
        "fork_sha": fork_sha(),
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
