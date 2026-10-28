#!/usr/bin/env python3
"""DeepSeek entrant instrumentation for the Kimi K3 trellis duel (not frozen).

This is entrant tooling, never part of the frozen harness. It reuses the
harness's own loading, compilation and launch functions so that a profile sees
the exact engine sequence, and adds only what the frozen harness must not
carry: DRAM calibration probes, an eager single-fixture runner with profiler
markers, and a per-row timing sweep in one process.

Modes:
  bandwidth   torch fill/copy/reduce probes; reports GB/s.
  single      load one layer at the row policy's capacity, verify the fixture
              digest against the frozen digests, then time each requested row
              eagerly (L2 flushed before every timed launch). With --profile,
              the last launch of the first row is bracketed by
              cudaProfilerStart/Stop so `ncu --profile-from-start off` sees
              only that launch sequence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_harness(path: Path):
    spec = importlib.util.spec_from_file_location("bench_kimi_trellis_decode", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# DRAM calibration


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _time_op(op, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(int(repeats)):
        start.record()
        op()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def cmd_bandwidth(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    nbytes = int(float(args.size_gib) * (1 << 30))
    nbytes -= nbytes % 16
    result = {
        "mode": "bandwidth",
        "device": props.name,
        "sms": int(props.multi_processor_count),
        "l2_cache_size": int(getattr(props, "L2_cache_size", 0)),
        "size_bytes": nbytes,
        "repeats": int(args.repeats),
    }
    src = torch.randint(-(1 << 30), 1 << 30, (nbytes // 4,),
                        dtype=torch.int32, device=device)
    dst = torch.empty_like(src)
    ops = {
        "fill": lambda: dst.fill_(7),
        "copy": lambda: dst.copy_(src),
        "read_sum_f32": lambda: src.view(torch.float32).sum(),
        "read_sum_i64": lambda: src.sum(dtype=torch.int64),
    }
    # warm every op once so first-touch / workspace allocation is untimed
    for op in ops.values():
        op()
    torch.cuda.synchronize()
    for name, op in ops.items():
        samples = _time_op(op, args.repeats)
        best = min(samples)
        moved = nbytes * (2 if name == "copy" else 1)
        result[name] = {
            "ms": samples,
            "median_ms": _median(samples),
            "best_ms": best,
            "best_gb_s": moved / best / 1e6,
            "median_gb_s": moved / _median(samples) / 1e6,
        }
    result["triton_read"] = _triton_read_probe(nbytes, device, args.repeats)
    del src, dst
    torch.cuda.empty_cache()
    payload = json.dumps(result, sort_keys=True)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    return 0


def _triton_read_probe(nbytes, device, repeats):
    """Read-only stream with a launch shape that saturates a 48-SM part."""
    try:
        import triton
        import triton.language as tl
    except Exception as error:  # noqa: BLE001 - probe only, report and move on
        return {"error": f"{type(error).__name__}: {error}"}

    programs, block = 512, 4096
    stride = programs * block
    n = nbytes // 4
    iters = n // stride
    if iters <= 0:
        return {"error": "buffer too small"}

    @triton.jit
    def _stream_read(x_ptr, out_ptr, iters, stride, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        for _ in range(iters):
            acc += tl.load(x_ptr + offs)
            offs += stride
        tl.store(out_ptr + pid, tl.sum(acc))

    x = torch.randn(iters * stride, dtype=torch.float32, device=device)
    out = torch.empty(programs, dtype=torch.float32, device=device)
    grid = (programs,)
    launch = lambda: _stream_read[grid](x, out, iters, stride, block, num_warps=8)
    launch()
    torch.cuda.synchronize()
    samples = _time_op(launch, repeats)
    moved = iters * stride * 4
    best = min(samples)
    result = {"ms": samples, "best_ms": best,
              "best_gb_s": moved / best / 1e6,
              "median_gb_s": moved / _median(samples) / 1e6,
              "bytes": int(moved), "programs": programs, "block": block}
    del x, out
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Single-fixture eager runner


def _read_phase_stamps(mt, launch, buffers):
    """Decode the probe build's per-CTA phase stamps (B12X_MIXED_TRELLIS_PHASE_TIMING).

    Slot meaning for the gated schedule: 0 entry, 1 rotation done, 2 after the
    rotation barrier, 3 FC1 done, 4 after the FC1 barrier, 5 activation done,
    6 after the activation barrier, 7 FC2 done.
    """
    workspace = getattr(buffers, "workspace", None)
    if workspace is None:
        return {"error": "buffers carry no workspace tensor"}
    sms = int(getattr(launch, "workspace_sms", None) or getattr(launch, "sms", 0) or 0)
    blocks = int(getattr(launch, "max_m_blocks", 0) or 0)
    stride = int(getattr(mt, "_PHASE_STAMP_STRIDE", 8))
    planes = int(getattr(mt, "_PHASE_READY_PLANES", 3))
    offset = sms * 4 + 2 + planes * blocks
    if sms <= 0 or workspace.numel() < offset + sms * stride:
        return {"error": "workspace too small for stamps"}
    raw = (workspace[offset:offset + sms * stride].to(torch.int64)
           & 0xFFFFFFFF).cpu().numpy()
    raw = raw.reshape(sms, stride)
    if not raw.any():
        return {"error": "stamps are zero; was B12X_MIXED_TRELLIS_PHASE_TIMING=1"}
    base = int(raw[0, 0])
    delta = ((raw - base) & 0xFFFFFFFF).astype("float64")
    delta[delta > 2 ** 31] -= 2 ** 32
    us = delta / 1000.0
    slots = {
        "entry": 0, "rotation_done": 1, "rot_barrier": 2, "fc1_done": 3,
        "fc1_barrier": 4, "act_done": 5, "act_barrier": 6, "fc2_done": 7,
    }
    summary = {}
    for name, slot in slots.items():
        column = us[:, slot]
        summary[name] = {
            "min_us": float(column.min()),
            "median_us": float(_median(column.tolist())),
            "max_us": float(column.max()),
        }
    span = us[:, int(max(slots.values()))].max() - us[:, 0].min()
    per_cta = {name: [round(float(v), 1) for v in us[:, slot]]
               for name, slot in slots.items()}
    return {"span_us": float(span), "slots": summary, "per_cta_us": per_cta}


def cmd_single(args):
    started = time.time()
    harness = load_harness(HERE / "bench_kimi_trellis_decode.py")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    stage = Path(args.stage)
    manifest = harness.load_json(Path(args.fixtures))
    layer_data, curve = harness.verify_fixture_manifest(manifest)
    stage_manifest = harness.verify_stage_manifest(
        stage, harness.load_json(stage / "manifest.json"))
    layers = [int(layer) for layer in layer_data]
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in fixtures {layers}")

    capacities = harness.normalize_capacities(
        json.loads(args.capacities) if args.capacities else None)
    mt = harness._import_kernel(Path(args.source))
    route_launches = {int(e["capacity"]): harness._compile_route_plan(int(e["capacity"]))
                      for e in capacities}
    launches = {}
    meta = {}
    for entry in capacities:
        capacity = int(entry["capacity"])
        launch, launch_meta = harness.compile_capacity(mt, capacity, entry, device)
        launches[capacity] = launch
        meta[str(capacity)] = launch_meta

    prepared = harness.prepare_layer(stage, stage_manifest, args.layer, device)
    bindings = {}
    for entry in capacities:
        capacity = int(entry["capacity"])
        bindings[capacity] = mt.bind_mixed_trellis(
            *prepared.tiers,
            prepared.global_to_combined,
            prepared.descriptor_map,
            prepared.rotations,
            launches[capacity],
            gate_experts=prepared.gate_counts,
            up_experts=prepared.up_counts,
        )
    buffer_sets = {}
    for entry in capacities:
        capacity = int(entry["capacity"])
        buffer_sets[capacity] = harness.make_capacity_buffers(
            mt, launches[capacity], device, int(props.multi_processor_count))

    expected = {}
    if args.digests:
        digests = harness.load_json(Path(args.digests))
        expected = digests.get("digests", {})

    flush = torch.empty(
        max(int(getattr(props, "L2_cache_size", 0)) * 2, 1 << 26),
        dtype=torch.int8, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rows = [int(row) for row in args.rows.split(",") if row.strip()]
    records = []
    for index, row in enumerate(rows):
        capacity = harness.capacity_for_rows(row, capacities)
        entry = harness.capacity_entry(capacities, capacity)
        scalars = harness.engine_scalars(capacity, entry)
        binding = bindings[capacity]
        buffers = buffer_sets[capacity]
        buffers.output = torch.empty((capacity, harness.HIDDEN),
                                     dtype=torch.bfloat16, device=device)
        data = layer_data[args.layer]
        x = data["x"][:row].to(device, non_blocking=False)
        ids = data["rows"][row]["ids"].to(device, non_blocking=False)
        weights = data["rows"][row]["weights"].to(device, non_blocking=False)

        def run():
            return harness.launch_engine_sequence(
                mt, x, weights, ids, binding, buffers, scalars,
                route_launches[capacity])

        torch.cuda.synchronize(device)
        for _ in range(int(args.warmup)):
            flush.zero_()
            run()
        torch.cuda.synchronize(device)
        samples = []
        profiled = args.profile and index == 0
        if profiled:
            torch.cuda.cudart().cudaProfilerStart()
            flush.zero_()
            start.record()
            run()
            end.record()
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()
            samples.append(start.elapsed_time(end))
        for _ in range(int(args.iters)):
            flush.zero_()
            start.record()
            run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        live = buffers.output[:row]
        digest = harness._digest_live(live)
        key = f"l{args.layer}/r{row}"
        record = {
            "layer": int(args.layer),
            "row": row,
            "capacity": int(capacity),
            "distinct": int(curve[row]),
            "roofline_ms": harness.roofline_ms(int(curve[row])),
            "ms": samples,
            "best_ms": min(samples),
            "median_ms": _median(samples),
            "digest": digest,
            "digest_expected": expected.get(key),
            "digest_ok": expected.get(key) == digest if expected else None,
        }
        records.append(record)
        line = {"row": row, "median_ms": record["median_ms"],
                "best_ms": record["best_ms"],
                "digest_ok": record["digest_ok"],
                "distinct": record["distinct"]}
        if os.environ.get("B12X_MIXED_TRELLIS_PHASE_TIMING") == "1":
            stamps = _read_phase_stamps(mt, launches[capacity], buffers)
            record["phase_stamps"] = stamps
            line["phase_span_us"] = stamps.get("span_us")
            line["phase_slots_us"] = {
                name: value.get("median_us")
                for name, value in stamps.get("slots", {}).items()
            }
        print(json.dumps(line, sort_keys=True), flush=True)

    payload = {
        "mode": "single",
        "device": props.name,
        "sms": int(props.multi_processor_count),
        "launch_meta": meta,
        "elapsed_s": round(time.time() - started, 1),
        "rows": records,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    if expected:
        bad = [f"l{r['layer']}/r{r['row']}" for r in records
               if not r["digest_ok"]]
        if bad:
            raise SystemExit(f"digest mismatch on {bad}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bandwidth = sub.add_parser("bandwidth")
    bandwidth.add_argument("--device", default="cuda:0")
    bandwidth.add_argument("--size-gib", default="2")
    bandwidth.add_argument("--repeats", type=int, default=3)
    bandwidth.add_argument("--out", default=None)
    bandwidth.set_defaults(func=cmd_bandwidth)

    single = sub.add_parser("single")
    single.add_argument("--source", required=True)
    single.add_argument("--stage", required=True)
    single.add_argument("--fixtures", required=True)
    single.add_argument("--digests", default=None)
    single.add_argument("--capacities", default=None)
    single.add_argument("--device", default="cuda:0")
    single.add_argument("--layer", type=int, required=True)
    single.add_argument("--rows", default="72")
    single.add_argument("--warmup", type=int, default=2)
    single.add_argument("--iters", type=int, default=5)
    single.add_argument("--profile", action="store_true",
                        help="bracket the first timed launch with profiler markers")
    single.add_argument("--out", default=None)
    single.set_defaults(func=cmd_single)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
