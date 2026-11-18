"""MG precision diagnostic: unchanged kernels, native packing, FP32 attention oracle.

DSV4 584B prefill changes QK only (PV is FP8 NoPE / BF16 RoPE). Native
DSV41 prefill changes canonical KV staging/PV; both arms retain BF16 QK.
Native decode compares shipped E4M3 QK/PV with BF16 QK/PV, and optionally
FP32 split partials. Synthetic full selected sets are not a model forward.

Prefill examples::

    python benchmarks/bench_mg_precision.py --format dsv4 --output dsv4.json
    python benchmarks/bench_mg_precision.py --format dsv41 --output dsv41.json

Decode/verify example::

    python benchmarks/bench_mg_precision.py --format dsv41 --mode decode \
        --rows 1,2,4,8,16,64 --timing-rows 1,2,4,8,16,64 --output decode.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import sys

SOURCE = Path(os.environ.get("CUTEAFD_SPARKINFER_SOURCE_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SOURCE))

AOT_CELLS = (("glm", 64), ("glm", 32), ("glmf", 64), ("glmf", 32),
             ("v4f", 64), ("v4f", 32), ("v4p", 128), ("v4p", 64))
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
    p.add_argument("--format", choices=("dsv4", "dsv41"))
    p.add_argument("--mode", choices=("prefill", "decode", "decode-aot"), default="prefill")
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--indexed-width", type=int, default=512)
    p.add_argument("--rows", type=ints)
    p.add_argument("--timing-rows", type=ints)
    p.add_argument("--scores", default="spread,peaked")
    p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=41512)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells", type=lambda s: s.split(","),
                   help="decode-aot subset, e.g. glm-h64,v4p-h128")
    args = p.parse_args(argv)
    if args.mode == "decode-aot":
        if args.format or args.rows or args.timing_rows:
            p.error("decode-aot uses fixed AOT geometries and all live rows 1–64")
        known = {f"{family}-h{heads}" for family, heads in AOT_CELLS}
        if args.cells and set(args.cells) - known:
            p.error(f"unknown decode-aot cells: {sorted(set(args.cells) - known)}")
        if min(args.samples, args.replays, args.warmup) < 1 or args.samples < 2:
            p.error("decode-aot needs at least two sample blocks and positive replay/warmup counts")
        if os.environ.get("B12X_MLA_SM120_NUM_SPLITS"):
            p.error("decode-aot must use the runtime planner, not a split override")
        return args
    if not args.format or args.cells:
        p.error("prefill/decode require --format and do not take --cells")
    args.rows = args.rows or ints("1,64,2048,2752,4096")
    args.timing_rows = args.timing_rows or ints("64,1536,2048,2752,4096")
    args.scores = args.scores.split(",")
    if args.accuracy_only:
        args.timing_rows = []
    if set(args.scores) - {"spread", "peaked"}:
        p.error("scores must be spread,peaked")
    if args.heads < 16 or args.heads % 16 or min(args.indexed_width, args.samples, args.replays, args.warmup) < 1:
        p.error("heads must be a positive multiple of 16; counts must be positive")
    if args.indexed_width > 4096:
        p.error("indexed width exceeds the 4096-key synthetic pool")
    if args.mode == "decode" and (args.format != "dsv41" or args.indexed_width != 512):
        p.error("native decode requires dsv41 with indexed width512")
    return args


def decode_arm_traits(arm):
    if arm not in ("fp8-decode", "bf16-decode", "fp8-f32-partials"):
        raise ValueError(arm)
    native = arm != "bf16-decode"
    traits = arm_traits("dsv41", "fp8-pv" if native else "bf16-pv")
    if native:
        traits = replace(traits, compute_mode=ComputeMode.FP8, q_nope_stride=528,
            kv_smem_stride=624, nt_per_warp_xv=traits.nt_per_warp_xv * 2)
    return traits


def build_decode(q, main_cache, extra_cache, mi, ei, sink):
    """Compile existing native producer and generic BF16 producer, same10 splits.

    The optional F32 partial arm changes only the typed tensor supplied to the
    existing epilogue/merge, not their implementation. Native producer mapping
    uses the same committed-ring/private-window ABI as test_native_v41_producer.
    """
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel
    from b12x.attention._shared.mla.smem import make_smem_layout
    from b12x.attention._shared.mla.merge import SparseMLASplitDecodeSinkMergeKernel
    rows, heads, _ = q.shape
    sr, cr = main_cache.view(-1,528), extra_cache.view(-1,288)
    values = [sr[:128,:512].contiguous(), sr[128:128+rows,:512].contiguous(), cr[:,:256].contiguous(), cr[:1,:256].contiguous()]
    scales = [sr[:128,512:].contiguous(), sr[128:128+rows,512:].contiguous(), cr[:,256:].contiguous(), cr[:1,256:].contiguous()]
    end = torch.tensor([128], device=q.device, dtype=torch.uint64)
    source_end = torch.tensor([cr.shape[0]], device=q.device, dtype=torch.uint64)
    pages = torch.arange(cr.shape[0]//256, device=q.device).to(torch.uint32)
    descriptor = [x.data_ptr() for x in values+scales] + [end.data_ptr(),pages.data_ptr(),source_end.data_ptr(),rows,cr.shape[0],1,(2<<32)|(cr.shape[0]//256)]
    desc = torch.tensor([descriptor]*rows, device=q.device, dtype=torch.uint64)
    meta = torch.tensor([[128,0,rows,128+r,0,cr.shape[0],cr.shape[0],0,0,1] for r in range(rows)], device=q.device, dtype=torch.uint64)
    bounds = torch.zeros(rows,device=q.device,dtype=torch.uint64)
    # Native ring uses committed first128 plus private rows; same logical keys
    # as generic main-cache slots1..128, then2..129, etc.
    assert torch.equal(mi, torch.arange(128,device=q.device)[None] + torch.arange(rows,device=q.device)[:,None] + 1)
    retained = [values,scales,end,source_end,pages,desc,meta,bounds]
    runs, lses = {}, {}
    for arm in ("fp8-decode", "bf16-decode", "fp8-f32-partials"):
        native = arm != "bf16-decode"
        traits = decode_arm_traits(arm)
        partial = torch.empty(rows,heads,10,512,device=q.device,dtype=torch.float32 if arm == "fp8-f32-partials" else torch.bfloat16)
        lse = torch.empty(rows,heads,10,device=q.device)
        output = torch.empty_like(q)
        kernel = UnifiedDecodeKernel(traits,make_smem_layout(traits),64,1,h_blocks=heads//16,num_splits=10,num_heads=heads,q_head_dim=512,
            topk=128,extra_topk=512,q_stride=q.stride(),swa_indices_stride0=512 if native else 128,extra_indices_stride0=512,
            mid_out_stride=partial.stride(),mid_lse_stride=lse.stride(),has_extra=True,pbs_extra=256,valid_hpb=16,native_dsv41_fp8=native)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        scale = cutlass.Float32(512**-.5 * math.log2(math.e))
        if native:
            params = tuple(from_dlpack(x) for x in (q,desc,meta,ei,bounds,partial,lse))
            compiled = cute.compile(kernel.call_native_v41,*params,scale,cutlass.Int32(rows),stream)
            decode_args = (*params,scale,cutlass.Int32(rows))
        else:
            params = tuple(from_dlpack(x) for x in (q,main_cache.flatten(),mi,partial,lse))
            extra = tuple(from_dlpack(x) for x in (extra_cache.flatten(),ei))
            decode_args = (*params,scale,cutlass.Float32(1),cutlass.Int32(128),cutlass.Int64(main_cache.stride(0)),*extra,
                cutlass.Int32(512),cutlass.Int32(2),cutlass.Int64(extra_cache.stride(0)),cutlass.Int32(rows))
            compiled = cute.compile(kernel.call_extra,*decode_args,stream)
        merge = SparseMLASplitDecodeSinkMergeKernel(static_num_chunks=10)
        merge_args = tuple(from_dlpack(x) for x in (partial,lse,bounds,sink,output))
        merged = cute.compile(merge,*merge_args,stream)
        def run(compiled=compiled,decode_args=decode_args,merged=merged,merge_args=merge_args,retained=retained):
            current = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled(*decode_args,current)
            merged(*merge_args,current)
        runs[arm] = (run,output)
        lses[arm] = lse
    return runs, lses


def run_decode_aot(args):
    """Compare the production launch ABI over full-width packed records."""
    import torch
    from b12x.attention._shared.mla.compressed_reference import (
        pack_compressed_sparse_mla_kv_cache_reference as pack_v4,
        unpack_compressed_sparse_mla_kv_cache_reference as unpack_v4,
    )
    from b12x.attention._shared.mla.reference import (
        pack_mla_kv_cache_reference as pack_glm, unpack_mla_kv_cache_reference as unpack_glm,
    )
    from b12x.integration.cuteafd import FLASH, PRO, GLM53, GLM53_FLASH, exportable_compilation
    from b12x.integration.cuteafd.dsv4_sparse_mla import compile_dsv4_sparse_mla_aot
    from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    capacity = 64
    arms = ("fp8", "bf16")
    # This is the cold unique-KV bandwidth ceiling's numerator, not an assertion
    # that repeated graph replays miss L2 or that CTAs share cache reads perfectly.
    report = {"mode": args.mode, "seed": args.seed, "capacity": capacity,
              "device": props.name, "sms": props.multi_processor_count,
              "capability": torch.cuda.get_device_capability(device),
              "fork_sha": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
              "bench_sha": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip(),
              "lse_unit": "natural_log", "warmup": args.warmup, "replays": args.replays,
              "sample_blocks": args.samples, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    geometries = {"glm": GLM53, "glmf": GLM53_FLASH, "v4f": FLASH, "v4p": PRO}

    def make_records(width, page, qdim, v4):
        slots = max(8192, width)
        raw = torch.randn((slots, qdim), device=device).bfloat16()
        if v4:
            cache = pack_v4(raw[:, :448], raw[:, 448:], page_size=page)
            k, v = unpack_v4(cache, page_size=page)
        else:
            records = pack_glm(raw[:, :512], raw[:, 512:] if qdim == 576 else None)
            cache = records.reshape(slots // page, -1).contiguous()
            k = unpack_glm(records).squeeze(1)
            v = k[:, :512]
        ids = (torch.arange(capacity, device=device)[:, None] * 149
               + torch.arange(width, device=device)[None, :] * 59) % slots
        return cache, ids.int().contiguous(), torch.full((capacity,), width, device=device, dtype=torch.int32), k, v

    for family, heads in AOT_CELLS:
        cell = f"{family}-h{heads}"
        if args.cells and cell not in args.cells:
            continue
        g = replace(geometries[family], heads=heads)
        v4 = family.startswith("v4")
        qdim = 512 if v4 else g.latent_dim
        scale = 512 ** -.5 if v4 else g.softmax_scale
        record_bytes = 584 if v4 else g.record_bytes
        main_width = 128 if v4 else getattr(g, "sparse_topk", g.index_topk)
        main_cache, main_ids, main_lengths, mk, mv = make_records(main_width, 256 if v4 else g.page_rows, qdim, v4)
        q = (torch.randn((capacity, heads, qdim), device=device) * 2.5).bfloat16()
        sink = torch.linspace(-1, 1, heads, device=device)
        kinds = (("win", 0, 64), ("c4", g.index_topk, 64), ("c128", 1024, 2)) if v4 else (("sparse", 0, 0),)
        for kind, width, page in kinds:
            inputs = (q, main_cache, main_ids, main_lengths)
            if width:
                extra, extra_ids, extra_lengths, ek, ev = make_records(width, page, qdim, True)
            if v4:
                inputs += ((extra, extra_ids, extra_lengths) if width else (None, None, None)) + (sink,)
            ref_out, ref_lse = [], []
            for lo in range(0, capacity, 8):
                hi = lo + 8
                k, v = mk[main_ids[lo:hi].long()], mv[main_ids[lo:hi].long()]
                if width:
                    k = torch.cat((k, ek[extra_ids[lo:hi].long()]), 1)
                    v = torch.cat((v, ev[extra_ids[lo:hi].long()]), 1)
                scores = torch.bmm(q[lo:hi].float(), k.transpose(1, 2)) * scale
                all_scores = torch.cat((scores, sink[None, :, None].expand(hi-lo, -1, -1)), -1) if v4 else scores
                ref_lse.append(torch.logsumexp(all_scores, -1))
                ref_out.append(torch.bmm(torch.softmax(all_scores, -1)[:, :, :main_width + width], v))
            ref, lse = torch.cat(ref_out), torch.cat(ref_lse)
            with exportable_compilation():
                programs = {arm: (compile_dsv4_sparse_mla_aot(g, route="decode", max_rows=capacity,
                                 indexed_width=width, indexed_page_rows=page, precision=arm) if v4 else
                                 compile_glm_sparse_mla_aot(g, route="decode", max_rows=capacity, precision=arm,
                                     name="glmf_sparse_mla" if family == "glmf" else "glm_sparse_mla",
                                     fp32_partials=family == "glmf")) for arm in arms}
            outs = {arm: torch.empty((capacity, heads, 512), device=device, dtype=torch.bfloat16) for arm in arms}
            scratch = {arm: torch.empty(programs[arm].scratch_bytes(capacity)["scratch"],
                                       dtype=torch.uint8, device=device) for arm in arms}
            # GLM batches keep a recipe fixed, independently of live width. Cover
            # every exported plan bucket across the row curve; V4 has one m64 plan.
            plans = (None,) if v4 else tuple(cap for cap, _, _ in programs["fp8"].geometry["decode_buckets"])
            for plan_rows in plans:
                for rows in range(1, capacity + 1):
                    record = {"cell": cell, "kind": kind, "rows": rows, "plan_rows": plan_rows,
                              "unique_kv_bytes_per_row": (main_width + width) * record_bytes, "arms": {}}
                    scalars = (rows,) if v4 else (rows, plan_rows)
                    def execute(arm):
                        programs[arm].launch(*inputs, outs[arm], scratch[arm], scalars=scalars)
                    graphs = {}
                    for arm in arms:
                        program = programs[arm]
                        splits = (program.geometry["num_splits"] if v4 else
                                  next(n for cap, n, _ in program.geometry["decode_buckets"] if cap >= plan_rows))
                        for _ in range(args.warmup):
                            execute(arm)
                        torch.cuda.synchronize()
                        offset = (rows * heads * splits * 512 * (4 if family == "glmf" else 2) + 1023) // 1024 * 1024
                        split_lse = scratch[arm][offset:offset+rows*heads*splits*4].view(torch.float32).reshape(rows, heads, splits)
                        actual_lse = torch.logsumexp(split_lse * math.log(2), -1)
                        if v4:
                            actual_lse = torch.logaddexp(actual_lse, sink[None, :])
                        delta = outs[arm][:rows].float() - ref[:rows]
                        lse_delta = actual_lse - lse[:rows]
                        if not bool(torch.isfinite(delta).all() & torch.isfinite(lse_delta).all()):
                            raise RuntimeError(f"nonfinite at {cell}/{kind}/{plan_rows}/{rows}/{arm}")
                        record["arms"][arm] = {"splits": splits, "output_rel_rms": float(delta.square().mean().sqrt() / ref[:rows].square().mean().sqrt()),
                                               "lse_rms": float(lse_delta.square().mean().sqrt()), "lse_max_abs": float(lse_delta.abs().max())}
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            execute(arm)
                        graph.replay()
                        graphs[arm] = graph
                    torch.cuda.synchronize()
                    samples = {arm: [] for arm in arms}
                    for block in range(args.samples):
                        for arm in arms[::1 if (block + rows) % 2 else -1]:
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(args.replays):
                                graphs[arm].replay()
                            end.record()
                            end.synchronize()
                            samples[arm].append(start.elapsed_time(end) * 1000 / args.replays)
                    for arm in arms:
                        times = samples[arm]
                        record["arms"][arm].update(samples_us=times, median_us=statistics.median(times),
                                                   min_us=min(times), max_us=max(times))
                        graphs[arm].reset()
                    report["records"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{cell}/{kind}: {len(report['records'])} rows at {time.strftime('%H:%M:%S')}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "decode-aot":
        return run_decode_aot(args)
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
    if args.mode == "decode":
        if max_rows > 64:
            raise ValueError("decode diagnostic is bounded to64 rows")
        main_page = 64
    main_cache, mk, mv = pack(values[0], main_page, "swa")
    extra_cache, ek, ev = pack(values[1], extra_page, "indexed")
    row = torch.arange(max_rows, device=device)[:, None]
    mi = ((row + (1 if args.mode == "decode" else 0) + torch.arange(window, device=device)[None]) % pool).int().contiguous()
    ei = ((row * 37 + torch.arange(args.indexed_width, device=device)[None] * 17) % pool).int().contiguous()
    ml = torch.full((max_rows,), window, device=device, dtype=torch.int32)
    el = torch.full((max_rows,), args.indexed_width, device=device, dtype=torch.int32)
    sink = torch.linspace(-1, 1, args.heads, device=device)
    arms = ("fp8-qk", "bf16-qk") if args.format == "dsv4" else ("bf16-pv", "fp8-pv")
    if args.mode == "decode":
        arms = ("fp8-decode", "bf16-decode", "fp8-f32-partials")
    report = {"mode": args.mode, "lse_unit": "natural_log", "format": args.format, "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "fork_sha": subprocess.check_output(["git", "-C", os.environ.get("CUTEAFD_SPARKINFER_SOURCE_DIR", str(Path(__file__).resolve().parents[1])), "rev-parse", "HEAD"], text=True).strip(),
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "traits": {arm: asdict(arm_traits(args.format, arm) if args.mode == "prefill" else decode_arm_traits(arm)) for arm in arms}, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pattern in args.scores:
        q = (torch.randn(max_rows, args.heads, 512, device=device) * (.03 if pattern == "spread" else .2)).bfloat16()
        if pattern == "peaked":
            target = ek[ei[:, 0].long()]
            q = (q.float() + target[:, None] * (14 * 512**.5 / target.square().sum(-1))[:, None, None]).bfloat16()
        out = {arm: torch.empty_like(q) for arm in arms}
        lse = {arm: torch.empty((max_rows, args.heads), device=device) for arm in arms}
        def run(arm, rows):
            # Native mixed KV needs more SMEM: use the existing single-group
            # specialization, since paired groups exceed SM120's per-CTA cap.
            partitions = [(1, args.heads, 0)] if args.format == "dsv41" else _mg_head_partitions(args.heads, 16)
            for n_hg, active_heads, offset in partitions:
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
                decode_runs, decode_lses = build_decode(q[:rows], main_cache, extra_cache, mi[:rows], ei[:rows], sink) if args.mode == "decode" else ({}, {})
                def execute(arm):
                    if decode_runs:
                        decode_runs[arm][0]()
                    else:
                        run(arm, rows)
                for arm in arms:
                    execute(arm)
                    torch.cuda.synchronize()
                    actual = decode_runs[arm][1] if decode_runs else out[arm][:rows]
                    actual_lse = torch.logaddexp(torch.logsumexp(decode_lses[arm] * math.log(2), -1), sink[None]) if decode_runs else lse[arm][:rows] * math.log(2)
                    delta = actual.float() - ref[:rows]
                    ld = actual_lse - ref_lse[:rows]
                    if not bool(torch.isfinite(delta).all() & torch.isfinite(ld).all()):
                        raise RuntimeError(f"nonfinite result: {arm}/{rows}/{pattern}")
                    record["arms"][arm] = {"output_rel_rms": float(delta.square().mean().sqrt() / ref[:rows].square().mean().sqrt()),
                        "output_max_abs": float(delta.abs().max()), "lse_max_abs": float(ld.abs().max()),
                        "lse_rms": float(ld.square().mean().sqrt())}
                    if rows in args.timing_rows:
                        for _ in range(args.warmup):
                            execute(arm)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            execute(arm)
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
