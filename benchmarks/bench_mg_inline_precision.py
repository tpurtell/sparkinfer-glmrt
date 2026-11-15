"""Inline-FP32-scale MG prefill precision: FP8 QK/PV against BF16 QK/PV with exact keys.

GLM 5.3 (656 B), GLM Flash (528 B) and Kimi (656 B, 96 or 48 heads, 192^-0.5)
records over the unchanged MG kernel, against FP32 attention over the exact
decoded record (TF32 off). Spread, medium and peaked score patterns; ragged,
empty and masked rows, slot-0 poison and row invariance; interleaved warm
graph timings. LSE is reported in log2. Synthetic selected sets, not a model
forward. `bench_mg_precision.py` covers the DSV4 and native V4.1 formats.

    python benchmarks/bench_mg_inline_precision.py --format glm --output glm.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
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
    """Source revision, through the real worktree mount when the container provides it."""
    source = os.environ.get("CUTEAFD_SPARKINFER_SOURCE_DIR", str(Path(__file__).resolve().parents[1]))
    return subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()


def arm_traits(fmt, arm):
    from b12x.attention._shared.mla.traits import with_bf16_attention
    if fmt not in INLINE or arm not in ("fp8", "bf16"):
        raise ValueError((fmt, arm))
    traits = make_unified_traits(ModelType.GLM_NEXT if fmt == "glmf" else ModelType.GLM_NSA,
        ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)
    return with_bf16_attention(traits) if arm == "bf16" else traits


def parse_args(argv=None):
    def ints(s):
        try:
            v = [int(x) for x in s.split(",")]
        except ValueError as e:
            raise argparse.ArgumentTypeError("expected comma-separated positive integers") from e
        if not v or min(v) < 1:
            raise argparse.ArgumentTypeError("counts must be positive")
        return v
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=tuple(INLINE), required=True)
    p.add_argument("--heads", type=int)
    p.add_argument("--selected", type=int, help="valid slots per row (default: all)")
    p.add_argument("--rows", type=ints, default=ints("1,64,2048,2752,4096"))
    p.add_argument("--timing-rows", type=ints, default=ints("64,1536,2048,2752,4096"))
    p.add_argument("--scores", default="spread,medium,peaked")
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=41512)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.scores = args.scores.split(",")
    if set(args.scores) - {"spread", "medium", "peaked"}:
        p.error("scores must be spread, medium or peaked")
    if args.heads is None:
        args.heads = INLINE[args.format][0]
    if args.heads < 16 or args.heads % 16 or min(args.samples, args.replays, args.warmup) < 1:
        p.error("heads must be a positive multiple of 16; counts must be positive")
    if not 1 <= (args.selected or 1) <= INLINE[args.format][2]:
        p.error("selected exceeds the format's slots per row")
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
    return inline_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
