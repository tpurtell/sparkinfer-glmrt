"""CPU-only CuTe DSL trace check for routed mixed-Trellis EXL3 exports.

Kernel-duel entrants work in a sandbox with no GPU. A CuTe DSL trace-time
error (``TYPE_UNSTABLE_JOIN``, an undefined name in a traced branch, a bad
``const_expr``) otherwise first surfaces when the organizer's packager compiles
the entry on hardware, which voids the round. CuTe DSL needs no device to
trace: given an explicit target arch it lowers the kernel to MLIR (and, without
``--trace-only``, through NVVM/ptxas to a cubin) on the CPU alone.

This command describes the target device (SM120 RTX PRO 6000 or SM121 GB10) to
Torch and the DSL, then calls ``compile_mixed_trellis`` with the same options
the engine's exporter (``export_b12x_exl3_aot.export``) derives for each
package variant, and stops at the first DSL error with its diagnostic. Shapes
come either from built EXL3 packages (``--package``: every variant's sealed
``v41_exl3.json``, i.e. the duel's exact shapes) or from ``--shape`` specs.

Examples::

  # Every variant of the duel's stock packages, both arches:
  python -m b12x.tools.cpu_trace_check --package /srv/duel/pkg-coordinator \\
      --package /srv/duel/pkg-spark
  # One explicit shape (hidden, intermediate, experts, top-k, capacity):
  python -m b12x.tools.cpu_trace_check --arch sm120 \\
      --shape hidden=1536,intermediate=1536,experts=512,topk=16,capacity=1,bits=4/5,codebook=mul1

What it proves: every selected program traces and lowers for that arch with
this tree's kernels. What it does not: runtime correctness, speed, register
or shared-memory limits the driver enforces at launch, and native link steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Per-arch target description, read from cuDeviceGetAttribute on raptor (RTX
# PRO 6000) and kakapo (GB10). SM counts equal the AOT workspace ceilings the
# exporter records (``EXL3_AOT_MAX_SMS``).
TARGETS = {
    "sm120": dict(major=12, minor=0, arch="sm_120a", name="NVIDIA RTX PRO 6000 Blackwell (CPU trace)",
                  multi_processor_count=188, shared_memory_per_block_optin=101376,
                  shared_memory_per_multiprocessor=102400, L2_cache_size=128 << 20,
                  total_memory=96 << 30),
    "sm121": dict(major=12, minor=1, arch="sm_121a", name="NVIDIA GB10 (CPU trace)",
                  multi_processor_count=48, shared_memory_per_block_optin=101376,
                  shared_memory_per_multiprocessor=102400, L2_cache_size=24 << 20,
                  total_memory=121 << 30),
}
ARCH_OF_COMPUTE = {(12, 0): "sm120", (12, 1): "sm121"}


@dataclass
class Shape:
    """One exporter variant: the ``export_b12x_exl3_aot.export`` arguments."""

    arch: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    capacity: int
    bits: tuple[int, int]
    codebook: str = "mcg"
    output_dtype: str = "bf16"
    tile: tuple[int, int, int, int] | None = None
    route_block: int = 8
    swiglu_limit: float | None = 10.0
    activation: str = "silu"
    blocks_per_sm: int | None = None
    decode_schedule: str | None = None
    token_major_rotation: bool = False
    warp_specialized: bool = False
    wire_input: bool = False
    ws_input_stages: int | None = None
    ws_dynamic_tiles: bool = False
    compact_weights: bool = False
    row_addresses: bool = False
    direct: bool | None = None
    label: str = ""
    sources: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        values = dict(vars(self))
        values.pop("label")
        values.pop("sources")
        return tuple(sorted((k, json.dumps(v)) for k, v in values.items()))

    def describe(self) -> str:
        parts = [f"{self.arch}", f"H{self.hidden}", f"I{self.intermediate}", f"E{self.experts}",
                 f"k{self.topk}", f"m{self.capacity}", f"K{self.bits[0]}{self.bits[1]}", self.codebook]
        if self.tile:
            parts.append("tile" + "/".join(map(str, self.tile)))
        if self.route_block != 8:
            parts.append(f"rb{self.route_block}")
        for flag in ("warp_specialized", "wire_input", "ws_dynamic_tiles", "token_major_rotation",
                     "compact_weights", "row_addresses"):
            if getattr(self, flag):
                parts.append(flag)
        if self.decode_schedule:
            parts.append(f"schedule={self.decode_schedule}")
        if self.blocks_per_sm:
            parts.append(f"bps{self.blocks_per_sm}")
        return " ".join(parts)


def shape_from_manifest(meta: dict, label: str) -> Shape:
    compute = tuple(meta["compute"])
    if compute not in ARCH_OF_COMPUTE:
        raise ValueError(f"{label}: unsupported compute {compute}")
    tile = tuple(meta["tile"]) if meta.get("tile") else None
    return Shape(
        arch=ARCH_OF_COMPUTE[compute], hidden=int(meta["hidden"]),
        intermediate=int(meta["intermediate"]), experts=int(meta["experts"]),
        topk=int(meta["top_k"]), capacity=int(meta["capacity"]), bits=tuple(meta["bits"]),
        codebook=meta.get("codebook", "mcg"), output_dtype=meta.get("output_dtype", "bf16"),
        tile=tile, route_block=int(meta.get("route_block", 8)),
        swiglu_limit=meta.get("swiglu_limit"), activation=meta.get("activation", "silu"),
        blocks_per_sm=int(meta["blocks_per_sm"]) if meta.get("blocks_per_sm") else None,
        decode_schedule=meta.get("decode_schedule"),
        token_major_rotation=bool(meta.get("token_major_rotation")),
        warp_specialized=bool(meta.get("warp_specialized")),
        wire_input=meta.get("input_format") == "e4m3_k32",
        ws_input_stages=meta.get("ws_input_stages"),
        ws_dynamic_tiles=bool(meta.get("ws_dynamic_tiles")),
        compact_weights=bool(meta.get("descriptor_rows")),
        row_addresses=bool(meta.get("row_addresses")),
        direct=bool(meta.get("direct")), label=label, sources=[label],
    )


def shapes_from_package(root: Path) -> list[Shape]:
    """Every sealed variant manifest under a package (or a root of packages)."""
    manifests = sorted(root.rglob("v41_exl3.json"))
    if not manifests:
        raise ValueError(f"no v41_exl3.json variant manifests under {root}")
    # Labels keep the package directory name (exl3-hy4f-k45-mul1/rtx-tp1/m1),
    # whether ``root`` is one package or a directory of packages.
    base = root.parent if (root / "manifest.json").is_file() else root
    return [shape_from_manifest(json.loads(p.read_text()), str(p.parent.relative_to(base)))
            for p in manifests if p.parent.name != "routes"]


_SHAPE_KEYS = {
    "hidden": int, "intermediate": int, "experts": int, "topk": int, "capacity": int,
    "route_block": int, "blocks_per_sm": int, "ws_input_stages": int,
}


def parse_shape(text: str, arch: str) -> Shape:
    """``key=value,...``; ``bits=4/5``, ``tile=64/256/64/256``, booleans as 1/0."""
    values: dict = {"arch": arch}
    for item in text.split(","):
        key, sep, value = item.strip().partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"shape item needs key=value: {item!r}")
        key = key.replace("-", "_")
        if key in _SHAPE_KEYS:
            values[key] = _SHAPE_KEYS[key](value)
        elif key in ("bits", "tile"):
            values[key] = tuple(int(v) for v in value.split("/"))
        elif key == "swiglu_limit":
            values[key] = None if value.lower() == "none" else float(value)
        elif key in ("codebook", "output_dtype", "activation", "decode_schedule", "arch"):
            values[key] = value
        elif key in ("token_major_rotation", "warp_specialized", "wire_input",
                     "ws_dynamic_tiles", "compact_weights", "row_addresses"):
            values[key] = value not in ("0", "false", "False", "")
        else:
            raise argparse.ArgumentTypeError(f"unknown shape key {key!r}")
    missing = {"hidden", "intermediate", "experts", "topk", "capacity", "bits"} - set(values)
    if missing:
        raise argparse.ArgumentTypeError(f"shape is missing {sorted(missing)}")
    values["label"] = text
    values["sources"] = ["--shape"]
    return Shape(**values)


def describe_target(arch: str) -> None:
    """Make Torch and CuTe DSL report the target device without touching CUDA.

    Kernel code selects SM120/SM121 paths with
    ``torch.cuda.is_available() and torch.cuda.get_device_capability(...)``
    and sizes with ``get_device_properties``; the DSL lowers for its envar
    arch. Nothing here allocates device memory or opens a context.
    """
    import torch

    spec = TARGETS[arch]
    props = type("CpuTraceTargetProperties", (), {
        **spec, "uuid": f"cpu-trace-{arch}", "is_integrated": arch == "sm121",
        "max_threads_per_multi_processor": 1536, "regs_per_multiprocessor": 65536,
        "warp_size": 32, "shared_memory_per_block": 49152, "gcnArchName": "", "pci_bus_id": 0,
        "pci_device_id": 0, "pci_domain_id": 0,
    })()

    class _Stream:
        cuda_stream = 0

        def synchronize(self):
            return None

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (spec["major"], spec["minor"])
    torch.cuda.get_device_properties = lambda device=None: props
    torch.cuda.get_device_name = lambda device=None: spec["name"]
    torch.cuda.current_stream = lambda device=None: _Stream()
    torch.cuda.is_current_stream_capturing = lambda: False

    from cutlass.cutlass_dsl import CuTeDSL

    CuTeDSL._get_dsl().envar.arch = spec["arch"]

    # The DSL itself reads a few driver attributes while generating launch
    # attributes (e.g. max shared memory per SM for ``min_blocks_per_mp > 1``,
    # which lands in the IR). Answer them from the measured target table and
    # refuse any other attribute rather than guess.
    from cutlass.base_dsl.runtime import cuda as dsl_cuda

    attribute = dsl_cuda.cuda.CUdevice_attribute
    known = {
        attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
            spec["shared_memory_per_multiprocessor"],
        attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
            spec["shared_memory_per_block_optin"],
        attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: spec["multi_processor_count"],
        attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: spec["L2_cache_size"],
        attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: spec["major"],
        attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: spec["minor"],
    }

    def get_device_attribute(which, device_id: int = 0):
        if which not in known:
            raise RuntimeError(f"cpu-trace-check has no {arch} value for device attribute {which}; "
                               "add the measured value to TARGETS")
        return known[which]

    dsl_cuda.get_device_attribute = get_device_attribute

    # The compile result carries the Trellis decode LUT as a device tensor
    # (host data, created after the kernel is compiled); keep it on the host.
    import importlib

    for name in ("mixed_trellis", "mixed_trellis4"):
        module = importlib.import_module(f"b12x.moe._shared.kernels.w4a16.{name}")
        build_lut = getattr(module, "_trellis256_execution_lut", None)
        if build_lut is not None and not getattr(build_lut, "_cpu_trace", False):
            def host_lut(device, codebook, _build=build_lut):
                return _build(torch.device("cpu"), codebook)

            host_lut._cpu_trace = True
            module._trellis256_execution_lut = host_lut


def _checked_environment() -> list[str]:
    """Say when a GPU is visible: the check never asks for it (the target is
    described, not queried), but a sandboxed entrant should see none."""
    return [f"{node} is visible; the check does not use it"
            for node in ("/dev/nvidiactl", "/dev/nvidia0") if os.path.exists(node)]


def _mixed_options(shape: Shape):
    """The exporter's ``compile_mixed_trellis`` options for one variant
    (mirrors ``export_b12x_exl3_aot.export``, cuteafd python/tools/aot)."""
    import torch
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import EXL3_AOT_MAX_SMS
    from b12x.moe.fused_moe._impl import (
        _projection_mixed_direct_topk_routes, _projection_mixed_tile_config,
    )

    spec = TARGETS[shape.arch]
    workspace_sms = EXL3_AOT_MAX_SMS[(spec["major"], spec["minor"])]
    direct = _projection_mixed_direct_topk_routes(shape.capacity, shape.topk, direct_exl3=True)
    if shape.direct is not None:
        direct = shape.direct
    if shape.row_addresses:
        direct = False
    block_m = 8 if direct else shape.route_block
    route_slots = (shape.capacity * shape.topk if direct else
                   route_pack_capacity(shape.capacity * shape.topk, block_m, shape.experts,
                                       topk=shape.topk)[1])
    route_blocks = route_slots if direct else (route_slots + block_m - 1) // block_m
    options = dict(
        size_m=shape.capacity, hidden_size=shape.hidden, intermediate_size=shape.intermediate,
        tier0_num_experts=shape.experts, tier1_num_experts=shape.experts, top_k=shape.topk,
        route_num_experts=shape.experts, max_m_blocks=route_blocks,
        sms=workspace_sms, workspace_sms=workspace_sms,
        max_shared_mem=spec["shared_memory_per_block_optin"],
        force_tile_config=_projection_mixed_tile_config(
            shape.tile, hidden_size=shape.hidden, intermediate_size=shape.intermediate,
            token_count=shape.capacity, direct_topk_routes=direct),
        tier0_bits=shape.bits[0], tier1_bits=shape.bits[1], trellis_codebook=shape.codebook,
        swiglu_limit=shape.swiglu_limit, moe_block_size=block_m, rotation_input_dtype="bf16",
        full_rotation_output_dtype=shape.output_dtype, route_ids_dtype=torch.int32,
        direct_topk_routes=direct, force_blocks_per_sm=None,
    )
    if shape.activation != "silu":
        options["activation"] = shape.activation
    if shape.token_major_rotation and not direct:
        options["token_major_rotation"] = True
    if shape.warp_specialized and not direct:
        options["warp_specialized"] = True
        if shape.wire_input:
            options["input_format"] = "e4m3_k32"
        if shape.ws_input_stages is not None:
            options["ws_input_stages"] = shape.ws_input_stages
        if shape.ws_dynamic_tiles:
            options["ws_dynamic_tiles"] = True
    if shape.row_addresses:
        options["row_addresses"] = True
    if shape.compact_weights:
        options["compact_weights"] = True
    if shape.decode_schedule is not None:
        options["decode_schedule"] = shape.decode_schedule
    return options


def _manifest_blocks_per_sm(shape: Shape, options: dict) -> dict:
    """A manifest records the resolved residency, not the request. Only an
    explicit override (2 blocks/SM on cooperative decode) needs passing on."""
    if shape.blocks_per_sm == 2 and not shape.warp_specialized:
        return {**options, "force_blocks_per_sm": 2}
    return options


def trace_one(shape: Shape) -> dict:
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import compile_mixed_trellis

    options = _manifest_blocks_per_sm(shape, _mixed_options(shape))
    started = time.monotonic()
    launch = compile_mixed_trellis(**options)
    result = {"seconds": round(time.monotonic() - started, 2)}
    if shape.blocks_per_sm is not None and int(launch.blocks_per_sm) != shape.blocks_per_sm:
        result["warning"] = (f"resolved {launch.blocks_per_sm} blocks/SM; the package recorded "
                             f"{shape.blocks_per_sm}")
    return result


def _dsl_error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    cause = exc.__cause__ or exc.__context__
    while not text and cause is not None:
        text = str(cause).strip()
        cause = cause.__cause__ or cause.__context__
    return text or repr(exc)


def run(shapes: list[Shape], *, keep_going: bool, trace_only: bool, verbose: bool) -> int:
    if trace_only:
        os.environ["CUTE_DSL_DRYRUN"] = "1"
    os.environ.setdefault("B12X_COMPILE_DISK_CACHE", "0")
    os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    for note in _checked_environment():
        print(f"note: {note}", flush=True)
    by_arch: dict[str, list[Shape]] = {}
    for shape in shapes:
        by_arch.setdefault(shape.arch, []).append(shape)
    if len(by_arch) > 1:
        # One process describes one target; re-exec per arch so the DSL and
        # Torch never see a mixed description.
        import subprocess
        import tempfile

        failures = 0
        for subset in by_arch.values():
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                json.dump(dump_shapes(subset), handle)
                path = handle.name
            try:
                command = [sys.executable, "-m", "b12x.tools.cpu_trace_check", "--shapes-json", path]
                command += ["--keep-going"] * keep_going + ["--trace-only"] * trace_only
                command += ["--verbose"] * verbose
                failures += subprocess.run(command, check=False).returncode != 0
            finally:
                os.unlink(path)
        return 1 if failures else 0

    arch = next(iter(by_arch))
    describe_target(arch)
    mode = "trace (MLIR only)" if trace_only else "trace + lower to cubin"
    print(f"cpu-trace-check: {len(shapes)} program(s) for {arch} [{mode}], no GPU", flush=True)
    failed = 0
    started = time.monotonic()
    for index, shape in enumerate(shapes, 1):
        where = ", ".join(shape.sources[:3]) + (" ..." if len(shape.sources) > 3 else "")
        print(f"[{index}/{len(shapes)}] {shape.describe()}  ({where})", flush=True)
        try:
            result = trace_one(shape)
        except Exception as exc:  # noqa: BLE001 - report the DSL diagnostic and stop
            failed += 1
            print(f"FAIL {shape.describe()}", flush=True)
            print(_dsl_error_text(exc), flush=True)
            if verbose:
                traceback.print_exc()
            if not keep_going:
                break
            continue
        line = f"  ok in {result['seconds']} s"
        if "warning" in result:
            line += f"  (warning: {result['warning']})"
        print(line, flush=True)
    elapsed = round(time.monotonic() - started, 1)
    status = "FAILED" if failed else "PASSED"
    print(f"cpu-trace-check {arch}: {status} ({failed} failing of {len(shapes)} program(s), "
          f"{elapsed} s)", flush=True)
    return 1 if failed else 0


def dump_shapes(shapes: list[Shape]) -> list[dict]:
    return [{**vars(s), "bits": list(s.bits), "tile": list(s.tile) if s.tile else None}
            for s in shapes]


def load_shapes(path: Path) -> list[Shape]:
    out = []
    for raw in json.loads(path.read_text()):
        raw = dict(raw)
        raw["bits"] = tuple(raw["bits"])
        raw["tile"] = tuple(raw["tile"]) if raw.get("tile") else None
        out.append(Shape(**raw))
    return out


def dedupe(shapes: list[Shape]) -> list[Shape]:
    unique: dict[tuple, Shape] = {}
    for shape in shapes:
        seen = unique.get(shape.key())
        if seen is None:
            unique[shape.key()] = shape
        else:
            seen.sources.extend(shape.sources)
    return list(unique.values())


def select(shapes: list[Shape], arches: set[str] | None, capacities: set[int] | None,
           match: list[str]) -> list[Shape]:
    out = []
    for shape in shapes:
        if arches and shape.arch not in arches:
            continue
        if capacities and shape.capacity not in capacities:
            continue
        if match and not any(m in s for m in match for s in shape.sources):
            continue
        out.append(shape)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trace routed mixed-Trellis EXL3 kernels on the CPU (no GPU) and stop at the "
                    "first CuTe DSL error.", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[1].split("What it proves", 1)[0])
    parser.add_argument("--package", action="append", type=Path, default=[],
                        help="EXL3 package directory (or a root of packages): trace every sealed "
                             "variant's v41_exl3.json recipe; arch comes from the manifest")
    parser.add_argument("--shape", action="append", default=[],
                        help="explicit variant: hidden=,intermediate=,experts=,topk=,capacity=,"
                             "bits=A/B[,codebook=,tile=a/b/c/d,route_block=,output_dtype=,"
                             "swiglu_limit=,activation=,decode_schedule=,warp_specialized=1,...]")
    parser.add_argument("--arch", action="append", choices=sorted(TARGETS),
                        help="restrict to these arches; --shape uses each (default: both)")
    parser.add_argument("--capacities", help="restrict to these capacities, e.g. 1,80")
    parser.add_argument("--match", action="append", default=[],
                        help="restrict package variants to directories containing this text "
                             "(repeatable), e.g. rtx-tp2/ or exl3-hy4f")
    parser.add_argument("--keep-going", action="store_true",
                        help="trace every program and report all failures (default: stop at first)")
    parser.add_argument("--trace-only", action="store_true",
                        help="stop after MLIR generation (skips NVVM/ptxas; faster, catches all "
                             "trace-time Python/DSL errors but not lowering errors)")
    parser.add_argument("--list", action="store_true", help="print the selected programs and exit")
    parser.add_argument("--verbose", action="store_true", help="print Python tracebacks")
    parser.add_argument("--shapes", action="append", type=Path, default=[],
                        help="shape list JSON written by --save-shapes (the organizer hands "
                             "entrants the duel's scored shapes this way)")
    parser.add_argument("--save-shapes", type=Path,
                        help="write the selected programs as a shape list JSON and exit")
    parser.add_argument("--shapes-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    arches = set(args.arch) if args.arch else None
    shapes: list[Shape] = []
    for path in ([args.shapes_json] if args.shapes_json else []) + args.shapes:
        shapes += load_shapes(path)
    for root in args.package:
        shapes += shapes_from_package(root)
    for text in args.shape:
        for arch in sorted(arches or TARGETS):
            shapes.append(parse_shape(text, arch))
    if not shapes:
        parser.error("give --package or --shape")
    capacities = {int(v) for v in args.capacities.split(",")} if args.capacities else None
    shapes = dedupe(select(shapes, arches, capacities, args.match))
    if not shapes:
        parser.error("the selection matched no programs")
    if args.save_shapes:
        args.save_shapes.write_text(json.dumps(dump_shapes(shapes), indent=1) + "\n")
        print(f"wrote {len(shapes)} program(s) to {args.save_shapes}")
        return 0
    if args.list:
        for shape in shapes:
            print(f"{shape.describe()}  ({', '.join(shape.sources)})")
        return 0
    return run(shapes, keep_going=args.keep_going, trace_only=args.trace_only,
               verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
