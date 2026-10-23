#!/usr/bin/env python3
"""Track T1: decode-once shared E4M3 trellis tiles for many-row dense prefill.

Compares, at the GLM 5.3 EXLR8 dense projection shapes and many-row prefill
counts, the frozen native trellis dense kernel (the numeric reference), the
``TrellisPrefillDense`` variants (F16 HMMA consumer over the exact E4M3 tile,
and FP8 MMA consumer with per-row per-K128 A8 scales) and the serving
block-FP8 prefill projection in both its W8A16 and W8A8 modes.

Weights are the same synthetic trellis payload the frozen kernel consumes;
activations are F16 randn (the trellis ABI input) for the trellis arms.

``--variants`` takes ``f16-64``, ``f16-128``, ``fp8-64``, ``fp8-128``, a
``m<mpw>`` warp-tile suffix (``f16-64m2``: two m16 row blocks per warp) and an
optional ``_<stages>`` suffix (``fp8-128_3``).

``mx8-64`` is the MXFP8 arm: one shared quantize pass writes per-32 E8M0
records, then the block-scaled ``mxf8f6f4`` m16n8k32 MMA consumes them against
the decoded E4M3 tile at E8M0 127. ``mx8q-64`` runs only the quantize pass and
``mx8g-64`` only the GEMM (timing split; a ``g`` arm must follow a ``q`` arm on
the same buffers). Only ``mx8-*`` and the exact arms take part in the error
comparison; ``--fp64-rows R`` adds the FP64 reference, ``x[:R] @ W`` with W read
back from the frozen native kernel through a one-hot activation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_exlr8_a16 import capture, timing, finite_nonzero
from b12x.integration.cuteafd._common import (
    Operand, Scalar, compile_program, exportable_compilation)
from b12x.integration.cuteafd import trellis_linear
from b12x.integration.cuteafd._fp8_weights import Fp8Projection
from b12x.gemm.trellis_linear._prefill import TrellisPrefillDense, TrellisPrefillMxfp8

SHAPES = {
    "q_b": dict(k=2048, n=16384, side=1),
    "o": dict(k=16384, n=6144, side=2),
}
MAX_ROWS = 4096


def trellis_operands(k, n, splits):
    return (Operand("rotated_inputs", torch.int64, "[4]"),
            Operand("payload", torch.uint8, f"[{k*n//2}]"),
            Operand("input_block_scales", torch.float16, f"[{k//128}]"),
            Operand("partial", torch.float32, f"[rows,{splits},{n}]", "scratch"),
            Operand("output", torch.float32, f"[rows,{n}]", "out"))


TRELLIS_SCALARS = (Scalar("rows"), Scalar("rotation_id"), Scalar("cta_limit"))


class BlockFp8:
    """Serving block-FP8 prefill projection: fp8_rows 0 = W8A16, 255 = W8A8."""

    def __init__(self, k, n, fp8_rows):
        self.projection = Fp8Projection(n, k, prefill_rows=MAX_ROWS)
        self.fp8_rows = fp8_rows

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer,
                 out: cute.Pointer, rows: cutlass.Int32, qscratch: cutlass.Int64,
                 stream: cuda.CUstream):
        self.projection(x, w, scale, out, rows, cutlass.Int32(self.fp8_rows),
                        qscratch, stream)


def trellis_weights(k, n, seed):
    """One synthetic trellis payload plus its K128 scales per shape.

    Shared by every row count of a shape, so a shorter run's rows are a prefix
    of a longer run's rows (the row-invariance comparison needs that).
    """
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    payload = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, device="cuda",
                            generator=g)
    scales = (torch.rand((k // 128,), device="cuda", generator=g) * .5 + .25).half()
    return payload, scales


def trellis_activation(k, rows, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return torch.randn((rows, k), dtype=torch.float16, device="cuda", generator=g) * .1


def trellis_tensors(k, n, rows, splits, payload, scales, seed):
    x = trellis_activation(k, rows, seed)
    inputs = torch.tensor([x.data_ptr()] * 4, dtype=torch.int64, device="cuda")
    partial = torch.empty((rows, splits, n), dtype=torch.float32, device="cuda")
    output = torch.empty((rows, n), dtype=torch.float32, device="cuda")
    return (inputs, payload, scales, partial, output)


def blockfp8_tensors(k, n, rows, core):
    x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda") * .1
    w = torch.randint(-16, 17, (n, k), device="cuda").to(torch.float8_e4m3fn)
    sc = torch.full(((n + 127) // 128, k // 128), .01, device="cuda")
    scratch = torch.empty(max(1024, core.projection.prefill_scratch_bytes(rows)),
                          dtype=torch.uint8, device="cuda")
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    return (x, w, sc, out), scratch


def parse_label(label):
    """``mode-width[.m<mpw>][_<stages>]`` -> (mode, n_cta, stages, mpw)."""
    mode, width = label.split("-")
    stages = 2
    if "_" in width:
        width, stages = width.split("_")
    mpw = 1
    if "m" in width:
        width, mpw = width.split("m")
    return mode, int(width), int(stages), int(mpw)


MX8_PARTS = {"mx8": "both", "mx8q": "quant", "mx8g": "gemm"}


def compile_variant(label, k, n, side, mode, n_cta, stages, mpw=1):
    g = trellis_linear.geometry(k, n, sharded_side=side, max_rows=MAX_ROWS)
    if mode in MX8_PARTS:
        core = TrellisPrefillMxfp8(g, n_cta=n_cta, stages=stages,
                                   part=MX8_PARTS[mode])
    else:
        core = TrellisPrefillDense(g, mode=mode, n_cta=n_cta, stages=stages, mpw=mpw)
    return compile_program(core, name=f"trellis_prefill_{label}",
                           operands=trellis_operands(k, n, g["splits"]),
                           scalars=TRELLIS_SCALARS,
                           key=(k, n, side, mode, n_cta, stages, mpw), version=1,
                           geometry=g)


def compile_blockfp8(k, n, fp8_rows):
    core = BlockFp8(k, n, fp8_rows)
    ops = (Operand("x", torch.bfloat16, "[rows,K]"),
           Operand("w", torch.float8_e4m3fn, "[N,K]"),
           Operand("scale", torch.float32, "[ceil(N/128),K/128]"),
           Operand("output", torch.bfloat16, "[rows,N]", "out"))
    return compile_program(core, name="bench_dense_block_fp8", operands=ops,
                           scalars=(Scalar("rows"), Scalar("qscratch", "int64")),
                           key=(k, n, fp8_rows))


def resource_census(program, directory):
    """Register/spill instruction census of the exported cubin (no profiling)."""
    directory.mkdir(parents=True, exist_ok=True)
    program.export_to_c(str(directory), "native", "census")
    obj = directory / "native.o"
    try:
        usage = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(obj)],
            text=True, stderr=subprocess.STDOUT)
        sass = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(obj)],
            text=True, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        return dict(error=str(error)[:400])
    ops = re.findall(r"^\s+/\*[0-9a-f]+\*/\s+@?!?\S*\s*([A-Z][A-Z0-9.]*)", sass, re.M)
    hmma = sum(1 for o in ops if o.startswith("HMMA"))
    return dict(registers=max((int(v) for v in re.findall(r"REG:(\d+)", usage)), default=0),
                local_bytes=max((int(v) for v in re.findall(r"LOCAL:(\d+)", usage)), default=0),
                stack_bytes=max((int(v) for v in re.findall(r"STACK:(\d+)", usage)), default=0),
                instructions=len(ops), hmma=hmma,
                per_hmma=round(len(ops) / hmma, 2) if hmma else None,
                ldl_stl=sum(1 for o in ops if o.startswith(("LDL", "STL"))),
                lds=sum(1 for o in ops if o.startswith("LDS")),
                ldg=sum(1 for o in ops if o.startswith("LDG")))


def error_stats(got, ref):
    diff = (got - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-30)
    return dict(bitwise=bool(torch.equal(got, ref)),
                exact_fraction=round(float((got == ref).double().mean()), 8),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                max_rel=float(rel.max()), mean_rel=float(rel.mean()),
                rms_rel=float((rel ** 2).mean().sqrt()))


def fp64_reference(k, n, side, x, payload, scales, limit, chunk=1024):
    """``x @ W`` in FP64, with W read back out of the frozen native kernel.

    A one-hot activation over one K-chunk at a time makes every native output
    ``out[c, n] = W[n][c] * scale[c//128]`` (one F16 product, exact in F32), so
    an FP64 matmul against the chunk's output is the exact value of the same
    weight arithmetic the kernels approximate.
    """
    R = x.shape[0]
    acc = torch.zeros((R, n), dtype=torch.float64, device="cuda")
    with exportable_compilation():
        program = trellis_linear.compile_trellis_linear_aot(
            input_features=k, output_features=n, sharded_side=side, max_rows=chunk)
    splits = k // 1024
    a = torch.zeros((chunk, k), dtype=torch.float16, device="cuda")
    out = torch.empty((chunk, n), dtype=torch.float32, device="cuda")
    partial = torch.empty((chunk, splits, n), dtype=torch.float32, device="cuda")
    inputs = torch.empty(4, dtype=torch.int64, device="cuda")
    for c0 in range(0, k, chunk):
        width = min(chunk, k - c0)
        a.zero_()
        idx = torch.arange(width, device="cuda")
        a[idx, idx + c0] = 1.0
        inputs.fill_(a.data_ptr())
        program.launch(inputs, payload, scales, partial, out, scalars=(chunk, 0, limit))
        acc += x[:, c0:c0 + width].double() @ out[:width].double()
    del a, out, partial, inputs
    torch.cuda.empty_cache()
    return acc


def fp64_stats(got, ref64):
    d = got.double() - ref64
    rel = d.abs() / ref64.abs().clamp_min(1e-12)
    return dict(fp64_rows=int(ref64.shape[0]),
                fp64_max_rel=float(rel.max()), fp64_mean_rel=float(rel.mean()),
                fp64_rms_rel=float((rel ** 2).mean().sqrt()),
                fp64_fro_rel=float(d.norm() / ref64.norm()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default="f16-64,fp8-64")
    ap.add_argument("--shapes", default="q_b,o")
    ap.add_argument("--rows", default="2048,4096")
    ap.add_argument("--check-rows", type=int, default=1024)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--fp64-rows", type=int, default=64,
                    help="rows of the FP64 reference (0 disables it)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    spec = []
    for label in args.variants.split(","):
        spec.append((label, *parse_label(label)))
    modes = {label: mode for label, mode, *_ in spec}
    labels = [label for label, *_ in spec]
    for i, label in enumerate(labels):
        if modes[label] == "mx8g":
            assert any(modes[l] == "mx8q" for l in labels[:i]), \
                "a mx8g arm reads the records of an earlier mx8q arm"
    shapes = [name for name in args.shapes.split(",") if name]
    rows_list = [int(v) for v in args.rows.split(",")]

    torch.manual_seed(408)
    props = torch.cuda.get_device_properties(0)
    limit = props.multi_processor_count * 4
    report = dict(device=props.name, sms=props.multi_processor_count,
                  variants=args.variants, measurements=[], errors=[], census=[])

    def save():
        """Keep every finished measurement on disk: a later arm can fault."""
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=1) + "\n")

    for name in shapes:
        s = SHAPES[name]
        k, n, side = s["k"], s["n"], s["side"]
        splits = (k + 1023) // 1024
        flops = lambda rows: 2.0 * rows * k * n
        payload, scales = trellis_weights(k, n, 408 + k)
        tensors = trellis_tensors(k, n, args.check_rows, splits, payload, scales, 1408 + k)
        with exportable_compilation():
            ref_program = None if args.no_baseline else \
                trellis_linear.compile_trellis_linear_aot(
                    input_features=k, output_features=n, sharded_side=side,
                    max_rows=MAX_ROWS)
            variants = {label: compile_variant(label, k, n, side, mode, n_cta, stages, mpw)
                        for label, mode, n_cta, stages, mpw in spec}
        ref = torch.empty_like(tensors[4])
        ref64 = None
        if args.fp64_rows and ref_program:
            xa = trellis_activation(k, args.check_rows, 1408 + k)
            ref64 = fp64_reference(k, n, side, xa[:args.fp64_rows], payload, scales, limit)
            del xa
            torch.cuda.empty_cache()
            report["errors"].append(dict(shape=name, variant="fp64", rows=int(ref64.shape[0])))
            save()
        programs = {"native": ref_program, **variants} if ref_program else dict(variants)
        for label, program in programs.items():
            if modes.get(label) in ("mx8q", "mx8g"):
                # Half-route timing arms: no independent numeric result.
                program.launch(*tensors, scalars=(args.check_rows, 0, limit))
                torch.cuda.synchronize()
                continue
            program.launch(*tensors, scalars=(args.check_rows, 0, limit))
            torch.cuda.synchronize()
            if label == "native":
                ref.copy_(tensors[4])
                finite_nonzero(ref, "native")
                if ref64 is not None:
                    report["errors"].append(dict(shape=name, variant="native",
                                                 rows=args.check_rows,
                                                 **fp64_stats(ref[:ref64.shape[0]], ref64)))
                    save()
                continue
            got = tensors[4].clone()
            finite_nonzero(got, label)
            stats = error_stats(got, ref) if ref_program else {}
            if ref64 is not None:
                stats.update(fp64_stats(got[:ref64.shape[0]], ref64))
            # Row invariance: a row's result cannot depend on which M block it
            # lands in or on the live row count (identical K128 order, no
            # row-dependent tile choice), and every row of the CTA is computed
            # by the same instruction stream.
            program.launch(*tensors, scalars=(args.check_rows, 0, limit))
            torch.cuda.synchronize()
            stats.update(shape=name, variant=label, rows=args.check_rows,
                         rerun_bitwise=bool(torch.equal(got, tensors[4])))
            if rows_list and max(rows_list) != args.check_rows:
                big = trellis_tensors(k, n, max(rows_list), splits, payload, scales, 1408 + k)
                program.launch(*big, scalars=(max(rows_list), 0, limit))
                torch.cuda.synchronize()
                stats["sub_rows_bitwise"] = bool(torch.equal(got, big[4][:args.check_rows]))
                del big
                torch.cuda.empty_cache()
            report["errors"].append(stats)
            save()
        for label, program in variants.items():
            report["census"].append(dict(
                shape=name, variant=label,
                **resource_census(program, Path("/tmp") / "t1census" / f"{name}-{label}")))
        save()
        del tensors
        torch.cuda.empty_cache()
        if args.no_timing:
            continue
        for rows in rows_list:
            timed = trellis_tensors(k, n, rows, splits, payload, scales, 1408 + k)
            for label, program in programs.items():
                graph = capture(lambda p=program: p.launch(*timed, scalars=(rows, 0, limit)))
                finite_nonzero(timed[4], label)
                us = timing(graph, args.replays)
                report["measurements"].append(
                    dict(shape=name, variant=label, rows=rows, us=round(us, 1),
                         tflops=round(flops(rows) / us / 1e6, 1)))
                save()
                del graph
                torch.cuda.empty_cache()
            del timed
            torch.cuda.empty_cache()
        if args.no_baseline:
            continue
        for fp8_rows, tag in ((0, "w8a16"), (255, "w8a8")):
            core = BlockFp8(k, n, fp8_rows)
            with exportable_compilation():
                program = compile_blockfp8(k, n, fp8_rows)
            for rows in rows_list:
                t, scratch = blockfp8_tensors(k, n, rows, core)
                graph = capture(lambda: program.launch(
                    *t, scalars=(rows, scratch.data_ptr())))
                finite_nonzero(t[3], tag)
                us = timing(graph, args.replays)
                report["measurements"].append(
                    dict(shape=name, variant=tag, rows=rows, us=round(us, 1),
                         tflops=round(flops(rows) / us / 1e6, 1)))
                save()
                del graph, t, scratch
                torch.cuda.empty_cache()

    text = json.dumps(report, indent=1)
    print(text)
    save()


if __name__ == "__main__":
    main()
