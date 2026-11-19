"""Family-neutral dense prefill curve over fixed-capacity exports.

The routes JSON names the engine exporter, native_library, manifest, and a
families mapping of exporter shape names to routes. Each route chooses groups
(1 or 4), egress (bf16/f32), reduction (none/bf16_partial/f32_add), and
references {bf16, fp8}. References are either {program, scalars} (the program
and launch flags selected by the engine, excluding the live rows scalar), or
{reason} if an exact-shape standalone shipping reference does not exist.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys


@dataclass(frozen=True)
class Shape:
    family: str
    name: str
    k: int
    n: int
    side: int
    groups: int
    egress: str
    reduction: str
    references: dict


def select_shapes(policies, routes, families, requested, bits, codebook):
    selected = []
    for family in families:
        if family not in policies or family not in routes:
            raise ValueError(f"unknown or unconfigured exporter family: {family}")
        available = {p["shape"]: p for p in policies[family]
                     if p.get("bits", 6) == bits and p["codebook"] == codebook}
        for name, route in routes[family].items():
            if requested and name not in requested:
                continue
            if name not in available:
                raise ValueError(f"{family}/{name} is not exported for K{bits} {codebook}")
            p = available[name]
            groups = route.get("groups", 1)
            reduction = route.get("reduction", "none")
            egress = route.get("egress", p["output_program"].rsplit("_", 1)[1])
            if groups != 1 and groups != p.get("grouped", {}).get("groups"):
                raise ValueError(f"{family}/{name} does not export {groups} groups")
            if reduction not in ("none", "bf16_partial", "f32_add") or (
                    reduction != "none" and (groups != 1 or p["sharded_side"] != 2)):
                raise ValueError(f"invalid TP reduction for {family}/{name}: {reduction}")
            if egress not in ("bf16", "f32") or (groups != 1 and egress != "bf16"):
                raise ValueError(f"invalid egress dtype for {family}/{name}: {egress}")
            refs = route.get("references", {})
            if set(refs) != {"bf16", "fp8"} or any(
                    not (r.get("program") and set(r) <= {"program", "scalars"})
                    and not (set(r) == {"reason"} and r["reason"]) for r in refs.values()):
                raise ValueError(f"{family}/{name} requires explicit bf16 and fp8 routes or reasons")
            selected.append(Shape(family, name, p["input_features"], p["output_features"],
                                  p["sharded_side"], groups, egress, reduction, refs))
    if missing := set(requested) - {s.name for s in selected}:
        raise ValueError(f"unconfigured shapes: {', '.join(sorted(missing))}")
    if not selected:
        raise ValueError("no shapes selected")
    return selected


def read_policies(exporter, families):
    # The engine exporter imports its locked fork. Do not let that replace the
    # candidate fork under test in the benchmark process.
    code = ("import contextlib, importlib.util, json, sys\n"
            "spec = importlib.util.spec_from_file_location('engine_exporter', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "with contextlib.redirect_stdout(sys.stderr):\n"
            "    spec.loader.exec_module(module)\n"
            "def policies(family):\n"
            "    entries = module.row_policies((family,))\n"
            "    for name, k, n, side, *width in module.FAMILY_SHAPES[family]:\n"
            "        for p in entries[:]:\n"
            "            if (p['input_features'], p['output_features'], p['sharded_side']) == (k, n, side) and p['shape'] != name and (not width or p.get('bits', 6) == width[0]):\n"
            "                yield dict(p, shape=name)\n"
            "    yield from entries\n"
            "print(json.dumps({f: list(policies(f)) for f in sys.argv[2:]}))\n")
    return json.loads(subprocess.check_output([sys.executable, "-c", code, str(exporter), *families]))


class ProgramInfo(C.Structure):
    _fields_ = [("name", C.c_char * 64), ("pointers", C.c_uint32),
                ("scalars", C.c_uint32), ("kinds", C.c_char * 16)]


def require_success(status, name):
    if status:
        raise RuntimeError(f"{name}: CUDA status {status}")


class Registry:
    def __init__(self, library, manifest):
        self.lib = C.CDLL(str(library))
        self.manifest = {p["name"]: p for p in manifest["programs"]}
        self.lib.cuteafd_dsv4_program_count.restype = C.c_uint32
        self.lib.cuteafd_dsv4_program_info.argtypes = [C.c_uint32, C.POINTER(ProgramInfo)]
        self.lib.cuteafd_dsv4_program_load.argtypes = [C.c_uint32]
        self.launch = self.lib.cuteafd_dsv4_program_launch
        self.launch.argtypes = [C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64), C.c_void_p]
        self.programs = {}
        for index in range(self.lib.cuteafd_dsv4_program_count()):
            info = ProgramInfo()
            require_success(self.lib.cuteafd_dsv4_program_info(index, C.byref(info)), "program_info")
            self.programs[info.name.decode()] = (index, info.pointers, info.scalars, info.kinds)

    def bind(self, name, operands, flags):
        index, pointers, scalars, kinds = self.programs[name]
        if pointers != len(operands) or scalars != len(flags) + 1 or kinds != b"i" * scalars:
            raise ValueError(f"{name}: incompatible native registry ABI")
        require_success(self.lib.cuteafd_dsv4_program_load(index), name)
        ptrs = (C.c_void_p * pointers)(*(t.data_ptr() for t in operands))
        args = (C.c_uint64 * scalars)(0, *flags)

        def run(rows, stream):
            args[0] = rows
            require_success(self.launch(index, ptrs, args, stream), name)

        return run

    def glue(self, name, operands, count_multiplier, sms=None):
        fn = getattr(self.lib, name)
        fn.argtypes = [C.c_void_p] * len(operands) + [C.c_int64] + (
            [C.c_int32] if sms is not None else []) + [C.c_void_p]
        args = [t.data_ptr() for t in operands]
        tail = () if sms is None else (sms,)
        return lambda rows, stream: require_success(fn(*args, rows * count_multiplier, *tail, stream), name)


def bind_reference(torch, registry, route, shape, x, capacity):
    if "reason" in route:
        return None
    if shape.groups != 1:
        raise ValueError("grouped shapes require a matched grouped reference or an omission reason")
    name = route["program"]
    p = registry.manifest[name]
    abi = [operand["name"] for operand in p["pointers"]]
    g = p["geometry"]
    if (g.get("input_features", g.get("width")), g.get("output_features", g.get("outputs"))) != (shape.k, shape.n):
        raise ValueError(f"{name}: mismatched reference K/N for {shape.name}")
    reference_capacity = p["capacity_rows"] or 0
    if reference_capacity < 1:
        raise ValueError(f"{name}: missing reference capacity")
    dtype = getattr(torch, p["pointers"][abi.index("out")]["dtype"].removeprefix("torch."))
    out = torch.empty((capacity, shape.n), device=x.device, dtype=dtype)
    flags = route.get("scalars", {})
    names = [s["name"] for s in p["scalars"]]
    if not names or names[0] != "rows" or names[1:] != list(flags) or any(
            type(v) is not int or not 0 <= v <= 2**31 - 1 for v in flags.values()):
        raise ValueError(f"{name}: must provide the selected runtime scalars in ABI order")
    weight = torch.randn((shape.n, shape.k), device=x.device, dtype=torch.bfloat16) * .1
    if abi == ["x", "w", "out"]:
        bound = registry.bind(name, (x, weight, out), ())

        def run(rows, stream, _storage=weight):
            bound(rows, stream)

        return run, out, reference_capacity
    if abi != ["x", "weight", "scales", "out", "scratch"] or g.get("scale_block", 128) != 128:
        raise ValueError(f"{name}: unsupported reference ABI {abi}")
    # QDQ is not timed; these are ordinary row-major E4M3 block-128 weights.
    packed = weight.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.ones((shape.n // 128, shape.k // 128), device=x.device, dtype=torch.float32)
    scratch = torch.empty(p["scratch_bytes_at_capacity"]["scratch"], device=x.device, dtype=torch.uint8)
    # The closure owns the backing allocations through graph replay.
    bound = registry.bind(name, (x, packed, scales, out, scratch), tuple(flags.values()))

    def run(rows, stream, _storage=(packed, scales, scratch)):
        bound(rows, stream)

    return run, out, reference_capacity


def exact(torch, x, y):
    return torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def curve_flags(records):
    curves = {}
    for record in records:
        for stage, value in record["us_per_row"].items():
            curves.setdefault(stage, []).append((record["rows"], value))
    out = {}
    for stage, points in curves.items():
        best = min(value for _, value in points)
        out[stage] = [{"rows": rows, "above_best": value / best - 1,
                       "above_neighbor": max((value / p[1] - 1 for p in points[max(0, i-1):i+2]), default=0)}
                      for i, (rows, value) in enumerate(points)
                      if value > 1.1 * best]
    return out


def _compile_variant(path, options):
    from b12x.gemm.trellis_linear import _native
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot

    if path is None:
        return compile_trellis_linear_aot(**options)
    spec = importlib.util.spec_from_file_location("_dense_trellis_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = _native.NativeTrellisLinear
    try:
        _native.NativeTrellisLinear = module.NativeTrellisLinear
        return compile_trellis_linear_aot(**options)
    finally:
        _native.NativeTrellisLinear = original


def capture_stages(cuda, calls, rows):
    events = [cuda.Event(enable_timing=True, external=True) for _ in range(len(calls) + 1)]
    graph = cuda.CUDAGraph()
    with cuda.graph(graph):
        stream = cuda.current_stream().cuda_stream
        events[0].record()
        for run, event in zip(calls.values(), events[1:]):
            run(rows, stream)
            event.record()
    return graph, events


def _shape_curve(args, shape, registry, exported, sms):
    import torch
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd.trellis_transform import compile_trellis_input_aot, compile_trellis_output_aot

    k, n, groups, cap = shape.k, shape.n, shape.groups, args.capacity
    out_dir = args.output_dir / shape.family / shape.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = torch.cuda.current_stream().cuda_stream
    cuda = {"device": "cuda"}
    x = torch.randn((cap, k), dtype=torch.bfloat16, **cuda) * .1
    rotation = [torch.empty_like(x) for _ in range(groups)]
    ptrs = torch.tensor([t.data_ptr() for t in rotation], dtype=torch.int64, **cuda)
    weights = [torch.randint(256, (k * n * args.bits // 8,), dtype=torch.uint8, **cuda) for _ in range(groups)]
    scales = [torch.empty(k // 128, dtype=torch.float16, **cuda) for _ in range(groups)]
    suh = [torch.rand(k, dtype=torch.float16, **cuda) + .125 for _ in range(groups)]
    svh = [torch.rand(n, dtype=torch.float16, **cuda) + .125 for _ in range(groups)]
    options = dict(input_features=k, output_features=n, bits=args.bits, codebook=args.codebook,
                   sharded_side=shape.side, rotation_count=groups, max_rows=cap,
                   input_dtype="bf16", groups=groups)
    with exportable_compilation():
        candidate = _compile_variant(None, options)
        oracle = _compile_variant(args.baseline_native, options) if args.baseline_native else None
        ingress = compile_trellis_input_aot(features=k, max_rows=cap, groups=groups, output_dtype="bf16")
        egress = compile_trellis_output_aot(features=n, max_rows=cap, groups=groups, output_dtype=shape.egress)
    body = exported(candidate, out_dir, "candidate")
    baseline = exported(oracle, out_dir, "baseline") if oracle else None
    ingress = exported(ingress, out_dir, "ingress")
    egress = exported(egress, out_dir, "egress")
    partial = torch.empty((cap, (k + 1023) // 1024, groups * n), dtype=torch.float32, **cuda)
    raw = [torch.empty((cap, n), dtype=torch.float32, **cuda) for _ in range(groups)]
    final_dtype = torch.float32 if shape.egress == "f32" else torch.bfloat16
    final = [torch.empty((cap, n), dtype=final_dtype, **cuda) for _ in range(groups)]

    def bind(lib, tensors, flags):
        pointers = [t.data_ptr() for t in tensors]

        def run(rows, active_stream):
            require_success(lib.run(*pointers, rows, *flags, active_stream), lib._name)

        return run

    stages = {"ingress": bind(ingress, [x, *suh, *rotation, *scales], (sms * 4,)),
              "trellis": bind(body, [ptrs, *weights, *scales, partial, *raw], (0, sms * 4))}
    egress_source = raw
    if shape.reduction != "none":
        summed = torch.empty_like(raw[0])
        egress_source = [summed]
        if shape.reduction == "bf16_partial":
            local = torch.empty_like(raw[0], dtype=torch.bfloat16)
            peer = torch.randn_like(local) * .1
            stages["narrow"] = registry.glue("cuteafd_load_narrow_f32_bf16", (raw[0], local), n, sms)
            stages["partial_add"] = registry.glue("cuteafd_bf16_f32_add_async", (local, peer, summed), n, sms)
        else:
            peer = torch.randn_like(raw[0]) * .1
            stages["partial_add"] = registry.glue("cuteafd_cuda_residual_add_f32_async", (raw[0], peer, summed), n)
    stages["egress"] = bind(egress, [*egress_source, *svh, *final], (sms * 4,))
    base_body = bind(baseline, [ptrs, *weights, *scales, partial, *raw], (0, sms * 4)) if baseline else None
    references = {name: bound for name, route in shape.references.items()
                  if (bound := bind_reference(torch, registry, route, shape, x, cap)) is not None}

    def chain(rows, core):
        for target in (*raw, *final):
            target.fill_(float("nan"))
        for name, run in stages.items():
            (core if name == "trellis" else run)(rows, stream)
        for target in (*raw, *final):
            if not torch.isfinite(target[:rows]).all().item() or not torch.isnan(target[rows:]).all().item():
                raise RuntimeError(f"{shape.name} M={rows}: missing writes or write into inactive rows")
        return [t[:rows].clone() for t in (*raw, *final)]

    full = chain(cap, stages["trellis"]) if args.check_prefix_rows else None
    records = []
    for rows in args.live_rows:
        if full is not None and rows < cap:
            saved_input = x[rows:].clone()
            for target in (x, *rotation):
                target[rows:].fill_(float("nan"))
        observed = chain(rows, stages["trellis"])
        if full is not None:
            if not all(exact(torch, t, f[:rows]) for t, f in zip(observed, full)):
                raise RuntimeError(f"{shape.name} M={rows}: poisoned prefix differs from M={cap}")
        if base_body:
            expected = chain(rows, base_body)
            if not all(exact(torch, t, f) for t, f in zip(observed, expected)):
                raise RuntimeError(f"{shape.name} M={rows}: candidate differs from baseline")
        if full is not None and rows < cap:
            x[rows:].copy_(saved_input)
        live_refs = {name: (run, out) for name, (run, out, limit) in references.items() if rows <= limit}
        for name, (run, out) in live_refs.items():
            out.fill_(float("nan"))
            run(rows, stream)
            if not torch.isfinite(out[:rows]).all().item():
                raise RuntimeError(f"{shape.name} M={rows}: {name} did not overwrite poison")

        graphs = {}
        arms = {"candidate": stages, **{name: {name: run} for name, (run, _) in live_refs.items()}}
        if base_body:
            arms["baseline"] = {"baseline_trellis" if s == "trellis" else f"baseline_{s}":
                                base_body if s == "trellis" else run for s, run in stages.items()}
        for arm, calls in arms.items():
            graph, events = capture_stages(torch.cuda, calls, rows)
            targets = (*raw, *final) if arm in ("candidate", "baseline") else (live_refs[arm][1],)
            for target in targets:
                target.fill_(float("nan"))
            graph.replay()
            for target in targets:
                if not torch.isfinite(target[:rows]).all().item():
                    raise RuntimeError(f"{shape.name} M={rows}: {arm} graph did not overwrite poisoned output")
            graphs[arm] = (graph, events, calls, targets)
        torch.cuda.synchronize()
        samples = {name: [] for calls in arms.values() for name in calls}
        samples.update({f"{arm}_total": [] for arm in arms if len(arms[arm]) > 1})
        for sample in range(args.samples):
            for graph, events, calls, targets in (graphs.values() if sample % 2 == 0 else reversed(graphs.values())):
                # Outside the timed graph: a last-live-row sentinel catches a
                # missing replay without writing the full output every sample.
                for target in targets:
                    target[rows-1:rows].fill_(float("nan"))
                graph.replay()
                events[-1].synchronize()
                for target in targets:
                    if not torch.isfinite(target[rows-1:rows]).all().item():
                        raise RuntimeError(f"{shape.name} M={rows}: timed graph did not overwrite poison")
                for name, start, end in zip(calls, events, events[1:]):
                    samples[name].append(start.elapsed_time(end) * 1000)
                if len(calls) > 1:
                    total = "baseline_total" if "baseline_trellis" in calls else "candidate_total"
                    samples[total].append(events[0].elapsed_time(events[-1]) * 1000)
        us = {name: statistics.median(values) for name, values in samples.items()}
        record = dict(rows=rows, us=us, us_per_row={name: val / rows for name, val in us.items()},
                      samples_us=samples, prefix_exact=True if full is not None else None,
                      route_exact=True if base_body else None)
        records.append(record)
        print(json.dumps(dict(family=shape.family, shape=shape.name, rows=rows, us_per_row=record["us_per_row"])), flush=True)
    return dict(shape=shape.__dict__, records=records, above_10_percent=curve_flags(records))


def benchmark(args, exported):
    routes = json.loads(args.routes.read_text())
    resolve = lambda key: (args.routes.parent / routes[key]).resolve()
    families = args.family.split(",")
    policies = read_policies(resolve("exporter"), families)
    shapes = select_shapes(policies, routes["families"], families, args.shapes.split(",") if args.shapes else (),
                           args.bits, args.codebook)
    import torch
    from cuda.bindings import driver as cuda
    from b12x.gemm.trellis_linear import _native

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    status, l2 = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
    require_success(status, "L2 cache size")
    manifest_path = resolve("manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("capability") != [props.major, props.minor]:
        raise ValueError("native reference manifest does not target this GPU")
    registry = Registry(resolve("native_library"), manifest)
    torch.manual_seed(273)
    report = dict(schema=1, device=props.name, sm_count=props.multi_processor_count, l2_bytes=l2,
                  capability=manifest["capability"], capacity=args.capacity, bits=args.bits, codebook=args.codebook,
                  samples=args.samples, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
                  routes_sha256=hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                  oracle_sha256=hashlib.sha256(args.baseline_native.read_bytes()).hexdigest() if args.baseline_native else None,
                  shapes=[])
    for shape in shapes:
        report["shapes"].append(_shape_curve(args, shape, registry, exported, props.multi_processor_count))
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
