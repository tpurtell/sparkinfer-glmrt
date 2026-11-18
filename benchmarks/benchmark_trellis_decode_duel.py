#!/usr/bin/env python3
"""Score the SM120 trellis_linear ABI, without transforms, using frozen AOT arms.

`catalog` selects production ABI geometry; `compile` exports it without a GPU;
`check` verifies the sealed catalog/bundles before queuing; `run` compares them
in one CUDA process. Bind a common directory for the harness, AOTs and results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import ctypes as C
import csv
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import sys

from kimi_k6_dense import exported, thermal_snapshot

ROWS = (1, 2, 4, 8, 16, 32, 48, 64)
PREFIX = "dense_trellis_k6_trellis_"
GEOMETRY = ("input_features", "output_features", "bits", "codebook", "sharded_side",
            "rotation_count", "max_rows", "groups", "input_dtype")
# Physical launches per rank in the owned-MLA TP2 Kimi 93-layer target pass.
# Zero-count entries qualify alternate shipped routes without weighting them.
KIMI = {
    "kda_qkvg_tp2": 69, "kda_o_tp2": 69, "shared_gate_up_tp2": 184,
    "shared_down_tp2": 92, "latent_down": 46, "latent_up": 46,
    "mla_q_a": 12, "mla_kv_a_padded": 12, "mla_q_b_owned": 12,
    "mla_q_b_tp2": 0, "mla_g_owned": 12, "mla_o_owned": 12,
    "lm_head": 1, "shared_down_owned": 0,
}
CONTROLS = {
    "GLM": ((6, 8), ("glm_q_b_tp2", "glm_o_tp2", "glm_kv_a_padded")),
    "GLM Flash": ((6, 8), ("glmf_kda_qkv_tp2", "glmf_o_tp2", "glmf_shared_down_tp2")),
    "Muse": ((6, 8), ("muse_kv_tp2", "muse_o_tp2", "muse_post_expert_tp2")),
    "ML4": ((6,), ("ml4_q_tp2", "ml4_o_tp2", "ml4_dense_down_tp2")),
    "Step": ((6, 8), ("step5_kv_tp2", "step5_dense_gate_up", "step5_dense_down")),
    "Step MTP": ((4,), ("step5_mtp_kv", "step5_mtp_o", "step5_mtp_gate_up")),
}
# Full target/drafter geometry, not a proposed TP placement. K6 and K8 are
# refused by the current engine loader; keep their results out of scoring.
INKLING = {"q_gate": (4608, 6144), "kv": (4608, 1024), "wr": (4608, 768),
           "output": (6144, 4608), "head": (4608, 201088),
           "mtp_gate_up": (4608, 18432), "mtp_down": (18432, 4608)}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_catalog(manifest):
    source = read_json(manifest)
    programs = {p["name"]: p["geometry"] for p in source["programs"] if p["op"] == "trellis_linear"}
    cells = []

    def add(family, role, name, count=0, geometry=None):
        g = programs[name] if geometry is None else geometry
        params = {k: g[k] for k in GEOMETRY if k in g}
        cells.append(dict(family=family, role=role, program=name, params=params,
                          rank1_count=count, shipping=geometry is None))

    def dense(family, role, bits, book, group=1, count=0):
        name = f"{PREFIX}{role}_{book}" + ("_k8" if bits == 8 else "")
        add(family, role, name + ("_g4" if group == 4 else ""), count)
        assert cells[-1]["params"]["bits"] == bits

    for bits in (6, 8):
        for book in ("mcg", "mul1"):
            for role, count in KIMI.items():
                dense("Kimi", role, bits, book, 4 if role == "kda_qkvg_tp2" else 1, count)
            dense("Kimi", "kda_qkvg_tp2", bits, book)  # standalone fallback
    for family, (widths, roles) in CONTROLS.items():
        for bits in widths:
            for book in ("mcg", "mul1"):
                for role in roles:
                    dense(family, role, bits, book)
    for dtype in ("f16", "bf16"):
        for role in ("q_b_tp2", "o_tp2", "k4096_n4096_s2"):
            add("EXLR8", role, f"exlr8_dense_trellis_{role}" + ("_bf16" if dtype == "bf16" else ""))
    for bits in (6, 8):
        for role, (k, n) in INKLING.items():
            name = f"planned_inkling_{role}_mcg_k{bits}"
            add("Inkling (planned)", role, name, geometry=dict(
                input_features=k, output_features=n, bits=bits, codebook="mcg",
                sharded_side=1, rotation_count=1, max_rows=4096, input_dtype="bf16"))
    assert len({c["program"] for c in cells}) == len(cells)
    return dict(manifest_sha256=sha(manifest), manifest_fork_revision=source["sparkinfer_revision"], cells=cells)


def compile_init():
    # Offline compile metadata matches the SM120 deployment export, not launch
    # sizing. The timed launch's CTA budget is derived from the actual device.
    from b12x._lib.compile_pool import _initialize_worker
    _initialize_worker(0, (12, 0), "trellis-duel-sm120", "trellis-duel-sm120", 188,
                       101376, 102400, multiprocessing.Array("i", [0, 0]))


def compile_cell(task):
    directory, cell = task
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    name = cell["program"]
    out = directory / name
    out.mkdir(exist_ok=True)
    with exportable_compilation():
        program = compile_trellis_linear_aot(**cell["params"])
        exported(program, out, "trellis_duel", load_module=False)
    files = [out / f"trellis_duel.{ext}" for ext in ("h", "o", "so")]
    return name, dict(geometry=program.geometry, sha256={f.name: sha(f) for f in files})


def compile_bundle(args, catalog):
    args.bundle.mkdir(exist_ok=True, parents=True)
    receipt = args.bundle / "bundle.json"
    receipt.unlink(missing_ok=True)
    tasks = [(args.bundle, cell) for cell in catalog["cells"]]
    with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=compile_init) as workers:
        artifacts = {}
        for future in concurrent.futures.as_completed([workers.submit(compile_cell, t) for t in tasks]):
            name, entry = future.result()
            artifacts[name] = entry
            print(f"exported {name}", flush=True)
    write_json(receipt, dict(source=args.source_sha, catalog_sha256=sha(args.catalog), artifacts=artifacts))


def check_bundles(arms, catalog_sha, cells):
    bundles = {}
    for arm, path in arms.items():
        bundle = read_json(path / "bundle.json")
        if bundle["catalog_sha256"] != catalog_sha:
            raise ValueError(f"{arm}: catalog digest differs")
        for cell in cells:
            stem = cell["program"]
            entry = bundle["artifacts"][stem]
            g = entry["geometry"]
            if any(g.get(k, 1 if k == "groups" else None) != v for k, v in cell["params"].items()):
                raise ValueError(f"{arm}: production geometry differs: {stem}")
            for file, expected in entry["sha256"].items():
                if sha(path / stem / file) != expected:
                    raise ValueError(f"{arm}: changed artifact {stem}/{file}")
        bundles[arm] = bundle
    return bundles


def load_lib(path, groups):
    lib = C.CDLL(str(path / "trellis_duel.so"), mode=os.RTLD_LOCAL)
    lib.load.argtypes, lib.load.restype = [], None
    lib.run.argtypes = [C.c_void_p] * (2 + 3*groups) + [C.c_int32]*3 + [C.c_void_p]
    lib.run.restype = C.c_int32
    lib.load()
    return lib


def independent_weight_decoder():
    import triton
    import triton.language as tl

    @triton.jit
    def unpack(raw, lut, dst, n0, K: tl.constexpr, N: tl.constexpr,
               WIDTH: tl.constexpr, SIZE: tl.constexpr, SIDE: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        k, n = i % K, n0 + i // K
        if SIDE == 1:
            tile = (n // 128 * (K // 16) + k // 16)*8 + n % 128 // 16
        else:
            tile = ((k // 128 * (N // 128) + n // 128)*8 + k % 128 // 16)*8 + n % 128 // 16
        rr, cc = k % 16, n % 16
        lane = (cc % 8)*4 + rr % 8 // 2
        j = (cc // 8)*4 + (rr // 8)*2 + rr % 2
        end = (lane*8 + j + 1)*WIDTH
        word = (end - 1)//32
        shift = (-end) & 31
        last = tl.load(raw + tile*(8*WIDTH) + word, i < SIZE).to(tl.uint32)
        first = tl.load(raw + tile*(8*WIDTH) + ((word + 8*WIDTH - 1) % (8*WIDTH)),
                        (i < SIZE) & (shift > 16), other=0).to(tl.uint32)
        state = ((last >> shift) | (first << ((32-shift) & 31))) & 65535
        tl.store(dst + i, tl.load(lut + state), i < SIZE)

    return unpack


def codebook_lut(book):
    import numpy as np
    import torch

    states = np.arange(65536, dtype=np.uint32)
    if book == "mul1":
        raw = states * np.uint32(0x83DCD12D)
        summed = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24))
        acc = (summed + 0x6400).astype(np.uint16).view(np.float16).astype(np.float64)
        inv, bias = np.array([0x1EEE, 0xC931], dtype=np.uint16).view(np.float16).astype(np.float64)
        values = (acc*inv + bias).astype(np.float16)
    else:
        raw = (states * np.uint32(0xCBAC1FED) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
        lo = (raw & 65535).astype(np.uint16).view(np.float16)
        hi = (raw >> 16).astype(np.uint16).view(np.float16)
        values = lo + hi
    lut = torch.from_numpy(values)
    if book == "e4m3":
        lut = lut.to(torch.float8_e4m3fn).to(torch.float16)
    return lut.cuda()


def reference_outputs(decoder, lut, x, payload, scales, k, n, bits, side, independent):
    import numpy as np
    import torch

    ref = torch.empty((x.shape[0], n), device="cuda", dtype=torch.float32)
    xf = x.float() * scales.float().repeat_interleave(128)
    words = payload.view(torch.int32)
    for n0 in range(0, n, 2048):
        span = min(n-n0, 2048)
        w = torch.empty((span, k), dtype=x.dtype, device="cuda")
        decoder[((w.numel() + 255)//256,)](words, lut, w, n0, k, n, bits, w.numel(), side, 256)
        if n0 == 0:
            # Audit the independent GPU unpack against the existing scalar
            # decoder on tightly bounded raw checkpoint storage units.
            if side == 1:
                raw = payload[:k*128*bits//8].cpu().numpy().tobytes()
                sample = independent.decode_unit(raw, bits, "gate", k).T
                got = w[:128]
            else:
                raw = payload[:n*128*bits//8].cpu().numpy().tobytes()
                # Side 2 stores full N in each K128 unit. Decode only N128.
                sample = independent.decode_unit(raw[:128*128*bits//8], bits, "down", 128).T
                got = w[:128, :128]
            expected = torch.from_numpy(np.ascontiguousarray(sample)).to(dtype=x.dtype, device="cuda")
            if not torch.equal(got, expected):
                raise RuntimeError("independent packed-weight decoder mismatch")
        ref[:, n0:n0+span] = xf @ w.float().T
    return ref


def seed_for(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def same_bits(a, b):
    import torch
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def row_sets(entries):
    # Verify every decode arithmetic switch, first/last row of both production
    # single and multi-tile prefill routes, and the headline row samples.
    edges = {edge for entry in entries for edge in entry["geometry"]["decode_route_thresholds"]}
    transitions = set(edges)
    for entry in entries:
        transitions.add(64 if entry["geometry"]["codebook"] == "e4m3" else 129)
    timing = sorted(set(ROWS) | {edge + offset for edge in edges for offset in (0, 1)})
    oracle = sorted(set(timing) | {edge + offset for edge in transitions for offset in (0, 1)} | {257})
    return timing, oracle


def check_oracles(invoke, refs, buffers, oracle_rows):
    import torch
    canonical = {}
    reports = {}
    for arm in buffers:
        partial, outs = buffers[arm]
        canonical[arm], reports[arm] = {}, {}
        for rows in reversed(oracle_rows):
            partial.fill_(float("nan"))
            for out in outs:
                out.fill_(float("nan"))
            invoke(arm, rows)
            sync = torch.cuda.current_stream()
            sync.synchronize()
            live = tuple(out[:rows] for out in outs)
            finite = all(torch.isfinite(out).all().item() and torch.count_nonzero(out).item()
                         for out in live)
            tail_ok = all(torch.isnan(out[rows:]).all().item() for out in outs)
            if not finite or not tail_ok:
                raise RuntimeError(f"{arm} row {rows}: output incomplete/nonfinite or tail overwritten")
            if not canonical[arm]:
                canonical[arm] = tuple(out.clone() for out in live)
            invariant = all(same_bits(a, b[:rows]) for a, b in zip(live, canonical[arm], strict=True))
            errors = torch.stack([a.float()-b[:rows] for a, b in zip(live, refs, strict=True)])
            abs_by_output = errors.abs().amax(dim=1)
            rms_by_output = errors.double().square().mean(dim=1).sqrt()
            metrics = dict(row_invariant=invariant, max_abs=abs_by_output.amax().item(),
                           rms=errors.double().square().mean().sqrt().item())
            if arm == "baseline":
                reports[arm][rows] = (abs_by_output, rms_by_output, metrics)
            else:
                a, b, _ = reports["baseline"][rows]
                metrics.update(worse_max_outputs=int((abs_by_output > a).count_nonzero().item()),
                               worse_rms_outputs=int((rms_by_output > b).count_nonzero().item()),
                               max_abs_delta=(abs_by_output-a).amax().item(),
                               max_rms_delta=(rms_by_output-b).amax().item(),
                               baseline_exact=all(same_bits(x, y[:rows]) for x, y in
                                                  zip(live, canonical["baseline"], strict=True)))
                reports[arm][rows] = metrics
    reports["baseline"] = {r: reps[2] for r, reps in reports["baseline"].items()}
    return canonical, reports


def timing_graph(torch, arm, rows, invoke, out, canonical, flush, intervals):
    invoke(arm, rows)
    if not all(same_bits(a[:rows], b[:rows]) for a, b in zip(out, canonical, strict=True)):
        # A candidate can fail invariance and still get an explicitly rejected
        # timing datum. Replay must reproduce *its own* eager cell output.
        canonical = tuple(o[:rows].clone() for o in out)
    graph = torch.cuda.CUDAGraph()
    pairs = [(torch.cuda.Event(enable_timing=True, external=True),
              torch.cuda.Event(enable_timing=True, external=True)) for _ in range(intervals)]
    with torch.cuda.graph(graph):
        for start, end in pairs:
            flush.add_(1)
            start.record()
            invoke(arm, rows)
            end.record()
    for o in out:
        o.fill_(float("nan"))
    graph.replay()
    pairs[-1][1].synchronize()
    if not all(same_bits(a[:rows], b[:rows]) for a, b in zip(out, canonical, strict=True)):
        raise RuntimeError(f"{arm} rows={rows}: poisoned graph replay mismatch")
    return graph, pairs


def run_cell(cell, arms, bundles, torch, reference_type, decoder, lut, flush, args, cta_limit):
    name, params = cell["program"], cell["params"]
    k, n, bits, book, side, rotations, _, groups, dtype = (params.get(key, 1) for key in GEOMETRY)
    weight_bytes = groups*k*n*bits//8
    entries = [b["artifacts"][name] for b in bundles.values()]
    timing_rows, oracle_rows = row_sets(entries)
    capacity = max(oracle_rows) + 1  # Include a poisoned tail even at the largest launch.
    rng = torch.Generator(device="cuda").manual_seed(seed_for(name))
    dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    rotated = [torch.randn((capacity, k), dtype=dtype, device="cuda", generator=rng)*0.1 for _ in range(rotations)]
    inputs = torch.tensor([x.data_ptr() for x in rotated], device="cuda", dtype=torch.int64)
    payloads = [torch.randint(0, 256, (weight_bytes//groups,), dtype=torch.uint8,
                              device="cuda", generator=rng) for _ in range(groups)]
    scales = [torch.empty(k//128, dtype=torch.float16, device="cuda").uniform_(0.5, 2, generator=rng)
              for _ in range(groups)]
    refs = [reference_outputs(decoder, lut, x[:max(oracle_rows)], p, s, k, n, bits, side,
                              reference_type(book)) for x, p, s in zip(rotated[:groups], payloads, scales, strict=True)]
    libs = {arm: load_lib(path / name, groups) for arm, path in arms.items()}
    buffers = {arm: (torch.empty(capacity*((k+1023)//1024)*n*groups, device="cuda"),
                     [torch.empty((capacity, n), device="cuda") for _ in range(groups)]) for arm in arms}
    common = [inputs.data_ptr(), *(p.data_ptr() for p in payloads), *(s.data_ptr() for s in scales)]
    arm_args = {arm: [*common, partial.data_ptr(), *(o.data_ptr() for o in out)]
                for arm, (partial, out) in buffers.items()}

    def invoke(arm, rows):
        status = libs[arm].run(*arm_args[arm], rows, 0, cta_limit, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"{arm} {name} rows={rows}: CUDA wrapper returned {status}")

    canonical, oracle = check_oracles(invoke, refs, buffers, oracle_rows)
    del refs
    rows_data = {}
    for rows in timing_rows:
        graphs = {arm: timing_graph(torch, arm, rows, invoke, buffers[arm][1], canonical[arm], flush,
                                    args.intervals) for arm in arms}
        values = {arm: [] for arm in arms}
        keys = tuple(graphs)
        for block in range(args.blocks):
            # Reverse every other block; rotate starting arm across pairs.
            phase = block//2 % len(keys)
            order = keys[phase:] + keys[:phase]
            for arm in (order if block % 2 == 0 else reversed(order)):
                graph, pairs = graphs[arm]
                graph.replay()
                pairs[-1][1].synchronize()
                values[arm].append(statistics.mean(a.elapsed_time(b)*1000 for a, b in pairs))
        base = values["baseline"]
        rows_data[rows] = {}
        for arm, blocks in values.items():
            us = statistics.median(blocks)
            ratios = [a/b for a, b in zip(base, blocks, strict=True)]
            rows_data[rows][arm] = dict(us=us, gbs=weight_bytes/(us*1000), blocks_us=blocks,
                                        baseline_over_arm=statistics.median(ratios),
                                        paired_min=min(ratios), paired_max=max(ratios))
    return dict(**cell, weight_bytes=weight_bytes, one_read_us=weight_bytes/(args.dram_gbs*1000),
                routes={arm: entry["geometry"]["decode_route_thresholds"]
                        for arm, entry in zip(arms, entries, strict=True)},
                oracle=oracle, thermal_after=thermal_snapshot(), rows=rows_data)


def aggregate(cells):
    output = []
    for bits in (6, 8):
        for book in ("mcg", "mul1"):
            mix = [c for c in cells if c["rank1_count"] and c["params"]["bits"] == bits
                   and c["params"]["codebook"] == book]
            if sum(c["rank1_count"] for c in mix) != sum(KIMI.values()):
                continue
            for rows in ROWS:
                arms = mix[0]["rows"][rows]
                sums = {a: sum(c["rows"][rows][a]["us"]*c["rank1_count"] for c in mix) for a in arms}
                blocks = {
                    a: [sum(v*c["rank1_count"] for v, c in zip(values, mix, strict=True))
                        for values in zip(*(c["rows"][rows][a]["blocks_us"] for c in mix), strict=True)]
                    for a in arms
                }
                output.append(dict(bits=bits, codebook=book, rows=rows,
                    rank1_ms={a: v/1000 for a, v in sums.items()},
                    per_layer_us={a: v/93 for a, v in sums.items()},
                    baseline_over_arm_blocks={a: [x/y for x, y in zip(blocks["baseline"], v, strict=True)]
                                              for a, v in blocks.items()},
                    one_read_ms=sum(c["one_read_us"]*c["rank1_count"] for c in mix)/1000))
    return output


def score(input_path, output_path):
    report = read_json(input_path)
    if not report["completed"]:
        raise ValueError("incomplete measurement report")
    scores = {}
    for arm in report["metadata"]["sources"]:
        if arm == "baseline":
            continue
        cohorts = defaultdict(lambda: {"timing_cells": 0, "oracle_cells": 0, "errors": [],
                                       "regressions": [], "ratios": []})
        for cell in report["cells"]:
            cohort = cohorts[(cell["family"], cell["params"]["bits"], cell["shipping"])]
            for rows, result in cell["oracle"][arm].items():
                cohort["oracle_cells"] += 1
                if not result["row_invariant"] or result["worse_max_outputs"] or result["worse_rms_outputs"]:
                    cohort["errors"].append(dict(program=cell["program"], rows=int(rows), **result))
            for rows, arms in cell["rows"].items():
                cohort["timing_cells"] += 1
                result = arms[arm]
                cohort["ratios"].append(result["baseline_over_arm"])
                # A noisy/one-sided measurement is a diagnostic, not a hard
                # gate; preserve the paired range for borderline control runs.
                if result["paired_max"] < 1.0:
                    cohort["regressions"].append(dict(program=cell["program"], rows=int(rows), **result))
        scores[arm] = [dict(family=f, bits=b, shipping=s, speedup_median=statistics.median(c["ratios"]),
                            speedup_min=min(c["ratios"]), speedup_max=max(c["ratios"]),
                            **{k: v for k, v in c.items() if k != "ratios"})
                       for (f, b, s), c in cohorts.items()]
    write_json(output_path, dict(scores=scores, aggregate=report["aggregate"]))


def table(input_path, output_path):
    report = read_json(input_path)
    if not report["completed"]:
        raise ValueError("incomplete measurement report")
    with output_path.open("w", newline="") as f:
        rows = csv.writer(f, delimiter="\t")
        rows.writerow(("family", "shipping", "role", "program", "K", "N", "bits", "book", "side",
                       "groups", "rows", "arm", "us", "GB/s", "one_read_us", "paired_speedup",
                       "paired_min", "paired_max", "invariant", "max_abs", "rms", "worse_max", "worse_rms"))
        for c in report["cells"]:
            p = c["params"]
            for m, arms in c["rows"].items():
                for arm, s in arms.items():
                    o = c["oracle"][arm][m]
                    rows.writerow((c["family"], c["shipping"], c["role"], c["program"],
                                   p["input_features"], p["output_features"], p["bits"], p["codebook"],
                                   p["sharded_side"], p.get("groups", 1), m, arm,
                                   round(s["us"], 4), round(s["gbs"], 3), round(c["one_read_us"], 4),
                                   s["baseline_over_arm"], s["paired_min"], s["paired_max"],
                                   o["row_invariant"], o["max_abs"], o["rms"],
                                   o.get("worse_max_outputs", 0), o.get("worse_rms_outputs", 0)))


def run(args, catalog, arms, bundles):
    import torch
    from cuda.bindings import driver

    if torch.cuda.get_device_capability(0) != (12, 0) or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one SM120 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(0)
    status, l2 = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0)
    if int(status) or l2 <= 0:
        raise RuntimeError(f"cannot resolve runtime L2 size: {status}, {l2}")
    cta_limit = properties.multi_processor_count * 4
    oracle_spec = importlib.util.spec_from_file_location("native_reference", Path(__file__).resolve().parents[1]/"tests/_reference/native_trellis.py")
    reference = importlib.util.module_from_spec(oracle_spec)
    oracle_spec.loader.exec_module(reference)
    lut = {book: codebook_lut(book) for book in {c["params"]["codebook"] for c in catalog["cells"]}}
    decoder = independent_weight_decoder()
    flush = torch.zeros(2*l2, device="cuda", dtype=torch.int32)
    warm = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda")
    for _ in range(128):
        torch.mm(warm, warm)
    torch.cuda.synchronize()
    report = dict(metadata=dict(catalog_sha256=sha(args.catalog), argv=sys.argv, gpu=properties.name,
                  sm_count=properties.multi_processor_count, gpu_uuid=str(properties.uuid), l2=l2,
                  cta_limit=cta_limit, dram_gbs=args.dram_gbs, torch=torch.__version__,
                  thermal_before=thermal_snapshot(), sources={a: b["source"] for a,b in bundles.items()}),
                  cells=[], completed=False)
    for cell in catalog["cells"]:
        try:
            record = run_cell(cell, arms, bundles, torch, reference.Reference, decoder,
                              lut[cell["params"]["codebook"]], flush, args, cta_limit)
        except Exception as exc:
            raise RuntimeError(f"{cell['program']}: {exc}") from exc
        report["cells"].append(record)
        print(f"{cell['program']}: " + ", ".join(f"{a} M1 {v['us']:.2f} us" for a,v in record["rows"][1].items()), flush=True)
        report["aggregate"] = aggregate(report["cells"])
        write_json(args.output, report)
    report["metadata"]["thermal_after"] = thermal_snapshot()
    report["completed"] = True
    write_json(args.output, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("catalog", "compile", "check", "run", "table", "score"))
    p.add_argument("--catalog", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--source-sha")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--output", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--intervals", type=int, default=12)
    p.add_argument("--dram-gbs", type=float, default=1792.)
    p.add_argument("--only", help="restrict to program names containing this string (pilot only)")
    args = p.parse_args()
    if args.command in ("score", "table"):
        if not args.input or not args.output:
            p.error(f"{args.command} requires --input and --output")
        (table if args.command == "table" else score)(args.input, args.output)
        return
    if not args.catalog:
        p.error("--catalog is required")
    if args.command == "catalog":
        write_json(args.catalog, build_catalog(args.manifest))
        return
    catalog = read_json(args.catalog)
    if args.only:
        catalog["cells"] = [c for c in catalog["cells"] if args.only in c["program"]]
        if not catalog["cells"]:
            p.error("--only selected no programs")
    if args.command == "compile":
        if not args.bundle or not args.source_sha or args.jobs < 1:
            p.error("compile requires --bundle, --source-sha and positive --jobs")
        compile_bundle(args, catalog)
        return
    configured = dict(text.split("=", 1) for text in args.arm)
    if len(configured) != len(args.arm) or "baseline" not in configured:
        p.error("provide exactly one --arm baseline=PATH and unique candidate names")
    arms = {name: Path(configured[name]) for name in ("baseline", *[a for a in configured if a != "baseline"])}
    bundles = check_bundles(arms, sha(args.catalog), catalog["cells"])
    if args.command == "check":
        print(f"verified {len(catalog['cells'])} cells and {len(arms)} AOT arms")
        return
    if not args.output or min(args.blocks, args.intervals, args.dram_gbs) <= 0:
        p.error("run requires --output and positive timing settings")
    run(args, catalog, arms, bundles)


if __name__ == "__main__":
    main()
