"""Warmed row curve of the Inkling 2 biased-sink attention programs vs an unbiased D128 control.

Prefill rows (default 2048, 2752, 4096: the policy's live-row band) on the
local (window 1024, ring) and global (paged, full context) programs, and
decode at C1 / C16 / C64 over a 3000-token context (global: 8 splits). The
control is MiMo's own GQA tile configured at D128/D128, window 1024, per-head
sink and no bias (the same staging, MMA and softmax code), so the ratio is
the cost of the bias gather and the per-row sink alone. Times are median
CUDA-event microseconds over ``--samples`` launches after one untimed launch,
with an L2 flush before each.

    python benchmarks/bench_cuteafd_inkling2_attention.py --rows 2048,2752,4096 --samples 15 --json out.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int32, Int64


def time_us(fn, iters: int, flush: torch.Tensor) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.max()  # read-only L2 flush: no dirty lines to write back inside fn
        torch.cuda._sleep(3_000_000)  # covers the host-side launch of fn
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


class _Control:
    """MiMo's SWA tile at Inkling's head geometry: D128/D128, window 1024, per-head sink, no bias."""

    def __init__(self, g, tokens: int):
        from b12x.integration.cuteafd._mimo_kernels import MimoGqaAttention

        self.attn = MimoGqaAttention(heads=g.heads, kv_heads=g.kv_heads, tokens=tokens, window=g.window,
                                     paged=False, sink=True, direct=True, softmax_scale=g.softmax_scale,
                                     ring_rows=2048, head=g.head_dim, v_head=g.head_dim)

    def key(self):
        return (self.attn.key(), "control")

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_step: cute.Pointer, ring: cute.Pointer, positions: cute.Pointer,
                 ring_slots: cute.Pointer, seq_first: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        f32 = cute.make_ptr(cutlass.Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        i32 = cute.make_ptr(cutlass.Int32, Int64(seq_first.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.attn(q, ring, kv_step, positions, i32, ring_slots, seq_first, sinks, out, f32, f32, rows, Int32(0),
                  Int32(1), stream)


def compile_control(g, route: str, max_rows: int):
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd.inkling2_attention import prefill_tokens

    launch = _Control(g, 1 if route == "decode" else prefill_tokens(g))
    n, d, r = g.heads, g.head_dim, g.record_elems
    operands = (Operand("q", torch.bfloat16, f"[rows,{n},{d}]"), Operand("kv_step", torch.bfloat16, f"[rows,{r}]"),
                Operand("ring", torch.bfloat16, f"[rings*2048,{r}]", "inout"),
                Operand("positions", torch.int64, "[rows]", align=8),
                Operand("ring_slots", torch.int64, "[rows]", align=8),
                Operand("seq_first", torch.int32, "[rows]", align=4),
                Operand("sinks", torch.bfloat16, f"[{n}]", align=2),
                Operand("out", torch.bfloat16, f"[rows,{n},{d}]", "out"),
                Operand("scratch", torch.uint8, "[1024]", "scratch"))
    return compile_program(launch, name="inkling2_control_attention", operands=operands,
                           scalars=(Scalar("rows"),), key=(max_rows, route, launch.key()))


def _inputs(g, rows: int, kind: str, gen):
    extent = g.extent(kind)
    q = torch.randn((rows, g.heads, g.head_dim), generator=gen, device="cuda").bfloat16()
    rel = (torch.randn((rows, g.heads, extent), generator=gen, device="cuda") * 2).bfloat16()
    sinks = (torch.randn((rows, g.heads), generator=gen, device="cuda") + 4).float()
    records = torch.randn((rows, g.record_elems), generator=gen, device="cuda").bfloat16()
    out = torch.empty_like(q)
    return q, rel, sinks, records, out


def bench_prefill(g, rows: int, samples: int, flush, gen, max_rows: int) -> dict:
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    q, rel, sinks, records, out = _inputs(g, rows, "local", gen)
    positions = torch.arange(rows, device="cuda")
    result = {"rows": rows}

    # Local: this step's records plus an (initially empty) ring.
    local = compile_inkling2_attention_aot(g, kind="local", route="prefill", max_rows=max_rows)
    ring = torch.zeros((g.ring_rows, g.record_elems), dtype=torch.bfloat16, device="cuda")
    slots = positions % g.ring_rows
    first = torch.zeros(rows, dtype=torch.int32, device="cuda")
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    result["local_us"] = time_us(lambda: local.launch(q, records, ring, positions, slots, first, rel, sinks, out,
                                                      scratch, scalars=(rows,)), samples, flush)

    control = compile_control(g, "prefill", max_rows)
    ring2 = torch.zeros((2048, g.record_elems), dtype=torch.bfloat16, device="cuda")
    sinks_head = (torch.randn(g.heads, generator=gen, device="cuda") + 4).bfloat16()
    result["control_us"] = time_us(lambda: control.launch(q, records, ring2, positions, positions % 2048, first,
                                                          sinks_head, out, scratch, scalars=(rows,)), samples, flush)

    # Global: the whole prefix in a paged cache (one sequence per prefill step).
    glob = compile_inkling2_attention_aot(g, kind="global", route="prefill", max_rows=max_rows)
    pages = -(-rows // g.page_rows)
    table = torch.randperm(pages, device="cuda").int()
    cache = torch.zeros((pages * g.page_rows, g.record_elems), dtype=torch.bfloat16, device="cuda")
    cache[table.long()[positions // g.page_rows] * g.page_rows + positions % g.page_rows] = records
    rel_g = rel if g.extent("global") == g.extent("local") else _inputs(g, rows, "global", gen)[1]
    gscratch = torch.empty(max(glob.scratch_bytes(rows)["scratch"], 1024), dtype=torch.uint8, device="cuda")
    result["global_us"] = time_us(lambda: glob.launch(q, cache, positions, table, rel_g, sinks, out, gscratch,
                                                      scalars=(rows, 0)), samples, flush)
    for key in ("local", "control", "global"):
        result[f"{key}_us_per_row"] = result[f"{key}_us"] / rows
    result["local_over_control"] = result["local_us"] / result["control_us"]
    return result


def bench_decode(g, concurrency: int, context: int, samples: int, flush, gen, splits: int) -> dict:
    from b12x.integration.cuteafd.inkling2_attention import compile_inkling2_attention_aot

    rows = concurrency
    q, rel, sinks, records, out = _inputs(g, rows, "local", gen)
    positions = torch.full((rows,), context, dtype=torch.int64, device="cuda")
    first = torch.arange(rows, dtype=torch.int32, device="cuda")
    scratch = torch.empty(1024, dtype=torch.uint8, device="cuda")
    result = {"concurrency": rows, "context": context}

    local = compile_inkling2_attention_aot(g, kind="local", route="decode", max_rows=64)
    ring = torch.randn((rows * g.ring_rows, g.record_elems), generator=gen, device="cuda").bfloat16()
    slots = torch.arange(rows, device="cuda") * g.ring_rows + context % g.ring_rows
    result["local_us"] = time_us(lambda: local.launch(q, records, ring, positions, slots, first, rel, sinks, out,
                                                      scratch, scalars=(rows,)), samples, flush)

    control = compile_control(g, "decode", 64)
    ring2 = torch.randn((rows * 2048, g.record_elems), generator=gen, device="cuda").bfloat16()
    slots2 = torch.arange(rows, device="cuda") * 2048 + context % 2048
    sinks_head = (torch.randn(g.heads, generator=gen, device="cuda") + 4).bfloat16()
    result["control_us"] = time_us(lambda: control.launch(q, records, ring2, positions, slots2, first, sinks_head,
                                                          out, scratch, scalars=(rows,)), samples, flush)

    glob = compile_inkling2_attention_aot(g, kind="global", route="decode", max_rows=64)
    pages = -(-(context + 1) // g.page_rows)
    cache = torch.randn((rows * pages * g.page_rows, g.record_elems), generator=gen, device="cuda").bfloat16()
    table = (torch.arange(rows, device="cuda")[:, None] * pages + torch.arange(pages, device="cuda")[None]).int()
    rel_g = rel if g.extent("global") == g.extent("local") else _inputs(g, rows, "global", gen)[1]
    gscratch = torch.empty(max(glob.scratch_bytes(rows)["scratch"], 1024), dtype=torch.uint8, device="cuda")
    result["global_us"] = time_us(lambda: glob.launch(q, cache, positions, table, rel_g, sinks, out, gscratch,
                                                      scalars=(rows, pages, splits)), samples, flush)
    result["local_over_control"] = result["local_us"] / result["control_us"]
    # Bytes the local decode must stream: K/V of the window for every row (bias is 96 KiB/row on top).
    kv_bytes = rows * g.window * g.record_elems * 2
    result["local_kv_gbps"] = kv_bytes / (result["local_us"] * 1e-6) / 1e9
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="2048,2752,4096")
    p.add_argument("--decode", default="1,16,64")
    p.add_argument("--context", type=int, default=3000)
    p.add_argument("--splits", type=int, default=8)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--max-rows", type=int, default=4096)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from b12x.integration.cuteafd import INKLING2_SMALL as g

    device = torch.cuda.get_device_properties(0)
    print(f"{device.name} sm_{device.major}{device.minor} {device.multi_processor_count} SMs, {platform.node()}")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(7)
    report = {"device": device.name, "sm": f"{device.major}{device.minor}", "host": platform.node(),
              "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "prefill": [], "decode": []}
    print(f"{'rows':>6} {'local us':>10} {'/row':>8} {'control us':>11} {'/row':>8} {'ratio':>6} "
          f"{'global us':>10} {'/row':>8}")
    for rows in (int(r) for r in args.rows.split(",")):
        r = bench_prefill(g, rows, args.samples, flush, gen, args.max_rows)
        report["prefill"].append(r)
        print(f"{rows:>6} {r['local_us']:>10.1f} {r['local_us_per_row']:>8.3f} {r['control_us']:>11.1f} "
              f"{r['control_us_per_row']:>8.3f} {r['local_over_control']:>6.3f} {r['global_us']:>10.1f} "
              f"{r['global_us_per_row']:>8.3f}")
    print(f"\ndecode at context {args.context} (global splits {args.splits})")
    print(f"{'C':>4} {'local us':>10} {'control us':>11} {'ratio':>6} {'KV GB/s':>8} {'global us':>10}")
    for c in (int(c) for c in args.decode.split(",")):
        r = bench_decode(g, c, args.context, args.samples, flush, gen, args.splits)
        report["decode"].append(r)
        print(f"{c:>4} {r['local_us']:>10.1f} {r['control_us']:>11.1f} {r['local_over_control']:>6.3f} "
              f"{r['local_kv_gbps']:>8.0f} {r['global_us']:>10.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
