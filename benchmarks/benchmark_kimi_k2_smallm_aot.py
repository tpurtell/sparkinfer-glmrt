#!/usr/bin/env python3
"""Measure shipping Kimi K2 TP6 decode AOT objects on GB10, without a model.

Pass an engine package's tp6-rank0 directory as --baseline and, optionally,
--candidate. Defaults cover every live row 1..64. The harness follows the
shipping selector (m1 for one row, m80 otherwise; Kimi does not ship m16).
The same full-width weights and route banks feed both arms. Timing includes
native route pack, cooperative core and weighted sum, but no input copy/RDMA.

No route trace is needed: each cell uses a seeded, randomly relabeled Zipf
prior tuned to #2462's mean touched counts (~70 at 8 rows, ~244 at 64).
These are synthetic inputs, not a replay of measured per-expert histograms.
"""
from __future__ import annotations

import argparse
import ctypes as c
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import socket
import statistics
import subprocess


HIDDEN, WIDTH, EXPERTS, TOPK = 3584, 512, 896, 16
# K2 w1/w3/w2 trellises, plus three FP16 rotations on each projection axis.
WEIGHT_BYTES = 3 * HIDDEN * WIDTH * 2 // 8 + 3 * (HIDDEN + WIDTH) * 2
PEAK_GBPS = 273.0


def seed_for(name):
    return int.from_bytes(sha256(name.encode()).digest()[:8], 'little')


def routes_for(cell, rows):
    rng = random.Random(seed_for(cell))
    order = list(range(EXPERTS))
    rng.shuffle(order)
    log_inverse_rates = [1.5 * math.log(rank + 5) for rank in range(EXPERTS)]
    # Independent exponential races draw top-k without replacement per row.
    return [[order[i] for i in sorted(range(EXPERTS), key=lambda i:
            math.log(-math.log(1.0 - rng.random())) + log_inverse_rates[i])[:TOPK]]
            for _ in range(rows)]


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def native_call(fn, *args):
    code = fn(*args)
    check(code == 0, f'{fn.__name__} failed: {code}')


def bind(lib, name, types, result=c.c_int):
    fn = getattr(lib, 'cuteafd_exl3_' + name)
    fn.argtypes, fn.restype = types, result
    return fn


def make_weights(torch):
    device = 'cuda'
    rng = torch.Generator(device=device).manual_seed(seed_for('Kimi-TP6-K2-weights'))
    descriptors = torch.full((3, EXPERTS * 2), -1, device=device, dtype=torch.int32)
    expert_map = torch.arange(EXPERTS, device=device, dtype=torch.int32)
    descriptors[:, :EXPERTS] = expert_map  # tier bit zero for all three projections

    def rotation(shape, normalizer=1.0):
        # Nonuniform finite signed K2 rotation/scaling operands; bound the
        # down projection so random words do not create pathological activations.
        values = torch.rand(shape, device=device, generator=rng) * .50 + .75
        signs = torch.randint(0, 2, shape, device=device, generator=rng) * 2 - 1
        return (values * signs * normalizer).half()

    ptr = dict(descriptor_map_ptr=descriptors, global_to_combined_ptr=expert_map,
               expert_map_ptr=expert_map, intermediate_rotations_ptr=rotation((2 * EXPERTS, 3 * WIDTH)),
               gate_suh_ptr=rotation((2 * EXPERTS, HIDDEN)), up_suh_ptr=rotation((2 * EXPERTS, HIDDEN)),
               svh_ptr=rotation((2 * EXPERTS, HIDDEN), .01))
    scalar = dict(route_num_experts=EXPERTS, weight_num_experts=2 * EXPERTS)
    plane_words = HIDDEN * WIDTH * 2 // 32
    zeros = torch.zeros(16, device=device, dtype=torch.uint8)
    ones = torch.ones(EXPERTS, device=device, dtype=torch.float32)
    for tier, count in enumerate((EXPERTS, 0)):
        for kind, planes in [('w13', 2), ('w2', 1)]:
            ptr[f't{tier}_{kind}_ptr'] = (torch.randint(-2**31, 2**31 - 1,
                (count * planes, plane_words), generator=rng, device=device,
                dtype=torch.int32) if count else zeros)
            ptr[f't{tier}_{kind}_scales_ptr'] = zeros
            ptr[f't{tier}_{kind}_global_ptr'] = ones
        scalar[f'tier{tier}_num_experts'] = EXPERTS
        for p in ['fc2', 'gate', 'up']:
            scalar[f'tier{tier}_{p}_experts'] = count
    return ptr, scalar


class Variant:
    def __init__(self, torch, directory, capacity, weights):
        self.torch = torch
        self.capacity = capacity
        self.graphs = {}
        meta = json.loads((directory / 'v41_exl3.json').read_text())
        self.meta = meta
        check((meta['compute'], meta['hidden'], meta['intermediate'], meta['experts'], meta['top_k'],
               meta['capacity'], meta['bits'], meta['activation'], meta['output_dtype'],
               meta['requires_route_preparation'], meta.get('route_block', 8)) ==
              ([12, 1], HIDDEN, WIDTH, EXPERTS, TOPK, capacity, [2, 3], 'situ', 'bf16', True, 8)
              and not meta.get('warp_specialized'), f'unexpected decode recipe: {directory}')
        check([o['label'] for o in meta['objects']] == ['v41_exl3_core', 'v41_exl3_sum'],
              f'unexpected launch sequence: {directory}')
        self.buffers = buffers = {}
        for name, spec in meta['buffers'].items():
            if name == spec['allocation']:
                value = torch.empty(spec['shape'], dtype=getattr(torch, spec['dtype'].removeprefix('torch.')),
                                    device='cuda')
                check(value.nbytes == spec['bytes'], f'size mismatch: {name}')
                if spec['zero_on_create']:
                    value.zero_()
                buffers[name] = value
        for name, spec in meta['buffers'].items():
            buffers[name] = buffers[spec['allocation']]
            check(spec['bytes'] <= buffers[name].nbytes, f'bad alias: {name}')
        raw = (directory / meta['trellis_lut']['file']).read_bytes()
        check(sha256(raw).hexdigest() == meta['trellis_lut']['sha256'], 'trellis LUT checksum mismatch')
        lut = torch.frombuffer(bytearray(raw), dtype=torch.float16).to('cuda')
        self.x = torch.empty((capacity, HIDDEN), dtype=torch.bfloat16, device='cuda')
        self.ids = torch.empty((capacity, TOPK), dtype=torch.int32, device='cuda')
        self.gates = torch.empty((capacity, TOPK), dtype=torch.float32, device='cuda')
        weight_ptr, scalar = weights
        self.ptr = dict(buffers, **weight_ptr, rotation_input_ptr=self.x, raw_topk_ids=self.ids,
                        route_expert_ids_ptr=self.ids, topk_weights_ptr=self.gates, trellis_lut_ptr=lut,
                        fc2_ptr=buffers['fc2'], output_ptr=buffers['output'])
        # Match the daemon's requested grid; the bridge queries device attributes
        # at create time and clamps to the runtime device before any capture.
        self.scalar = dict(scalar, grid_x=meta['sms'] * meta['blocks_per_sm'])
        self.lib = c.CDLL(str(directory / 'libcuteafd_exl3.so'))
        route = directory / meta['route_preparation']['manifest']
        check(sha256(route.read_bytes()).hexdigest() == meta['route_preparation']['sha256'],
              'route manifest checksum mismatch')
        self.route_lib = c.CDLL(str(route.parent / 'libv41_exl3_routes.so'))
        self.ctx, self.route_ctx = c.c_void_p(), c.c_void_p()
        self.destroy = bind(self.lib, 'destroy', [c.c_void_p], None)
        self.destroy_routes = bind(self.route_lib, 'routes_destroy', [c.c_void_p])
        native_call(bind(self.lib, 'create', [c.POINTER(c.c_void_p)]), c.byref(self.ctx))
        native_call(bind(self.route_lib, 'routes_create', [c.POINTER(c.c_void_p)]), c.byref(self.route_ctx))
        self.pack = bind(self.route_lib, 'routes_launch', [c.c_void_p, c.POINTER(c.c_void_p),
                         c.POINTER(c.c_uint64), c.c_int32, c.c_void_p])
        self.objects = [bind(self.lib, name, [c.c_void_p, c.POINTER(c.c_void_p),
                        c.POINTER(c.c_int32), c.c_void_p]) for name in ('core', 'sum')]
        self.route_buffers = [self.ids, weight_ptr['expert_map_ptr']] + [buffers[name] for name in (
            'packed_route_indices', 'block_expert_ids', 'packed_route_count', 'expert_offsets', 'expert_counts')]

    def load(self, tensors, rows):
        self.x[:rows].copy_(tensors[0][:rows])
        self.ids[:rows].copy_(tensors[1][:rows])
        self.gates[:rows].copy_(tensors[2][:rows])

    def run(self, rows):
        stream = self.torch.cuda.current_stream().cuda_stream
        native_call(self.pack, self.route_ctx,
                    (c.c_void_p * 7)(*(x.data_ptr() for x in self.route_buffers)),
                    (c.c_uint64 * 7)(rows * TOPK * 4, *(x.nbytes for x in self.route_buffers[1:])),
                    rows, stream)
        for fn, obj in zip(self.objects, self.meta['objects']):
            p = obj['pointer_slots']
            s = obj['scalar_slots']
            scalars = dict(self.scalar, active_m=rows)
            native_call(fn, self.ctx, (c.c_void_p * len(p))(*(self.ptr[k].data_ptr() for k in p)),
                        (c.c_int32 * len(s))(*(scalars[k] for k in s)), stream)

    def capture(self, rows):
        torch = self.torch
        self.run(rows)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.run(rows)
        self.graphs[rows] = graph
        return graph

    def output(self, rows):
        return self.buffers['output'][:rows]

    def poison_output(self):
        self.buffers['output'].fill_(float('nan'))

    def close(self):
        self.torch.cuda.synchronize()
        self.graphs.clear()
        native_call(self.destroy_routes, self.route_ctx)
        self.destroy(self.ctx)


def inputs(torch, rows, role):
    name = f'kimi-k2-tp6-m{rows}.{role}'
    ids = routes_for(name, 80)
    rng = torch.Generator(device='cuda').manual_seed(seed_for(name))
    x = (torch.randn((80, HIDDEN), device='cuda', generator=rng) * .03).to(torch.bfloat16)
    gates = torch.randn((80, TOPK), device='cuda', generator=rng).softmax(-1)
    touched = len({i for row in ids[:rows] for i in row})
    return (x, torch.tensor(ids, device='cuda', dtype=torch.int32), gates), touched


def exact(torch, actual, expected, context):
    lhs, rhs = actual.view(torch.int16), expected.view(torch.int16)
    mismatches = int((lhs != rhs).sum())
    check(mismatches == 0, f'{context}: {mismatches} different BF16 words; '
          f'max abs {(actual.float() - expected.float()).abs().max().item():.8g}')


def verify(torch, name, variants, rows, data):
    v = variants[1 if rows == 1 else 80]
    v.load(data, rows)
    v.poison_output()
    v.capture(rows)
    v.poison_output()
    v.graphs[rows].replay()
    out = v.output(rows).clone()
    check(bool(out.isfinite().all()) and bool(out.abs().sum(dtype=torch.float32) > 0),
          f'{name} m{rows}: nonfinite or all-zero graph output')
    wide = variants[80]
    wider_rows = min(80, max(rows + 1, rows * 2))
    wide.load(data, wider_rows)
    wide.poison_output()
    wide.run(wider_rows)
    exact(torch, wide.output(rows), out, f'{name} m{rows} vs m{wider_rows} prefix')
    return out


def elapsed(torch, graph, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def smi(*args):
    p = subprocess.run(['nvidia-smi', *args], text=True, capture_output=True, timeout=10)
    return dict(stdout=p.stdout.strip(), stderr=p.stderr.strip(), returncode=p.returncode)


def telemetry():
    return dict(performance=smi('-q', '-d', 'PERFORMANCE'),
                state=smi('--query-gpu=name,uuid,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit',
                          '--format=csv'))


def run(args, torch):
    check(torch.cuda.get_device_capability() == (12, 1), 'the production package requires SM121')
    arms = {'baseline': args.baseline}
    if args.candidate:
        arms['candidate'] = args.candidate
    weights = make_weights(torch)
    variants = {arm: {cap: Variant(torch, root / f'm{cap}', cap, weights) for cap in (1, 80)}
                for arm, root in arms.items()}
    rows_list = sorted(set(args.rows))
    try:
        cells = {}
        for rows in rows_list:
            data, _ = inputs(torch, rows, 'oracle')
            out = {arm: verify(torch, arm, by_cap, rows, data) for arm, by_cap in variants.items()}
            if 'candidate' in out:
                exact(torch, out['candidate'], out['baseline'], f'candidate vs baseline at m{rows}')
            # Reuse the captured graphs with changed pointers' contents; timing
            # uses the second, separately seeded bank after verifying it too.
            data, touched = inputs(torch, rows, 'timing')
            reference = None
            for arm, by_cap in variants.items():
                v = by_cap[1 if rows == 1 else 80]
                v.load(data, rows)
                v.poison_output()
                v.run(rows)
                eager = v.output(rows).clone()
                v.poison_output()
                v.graphs[rows].replay()
                exact(torch, v.output(rows), eager, f'{arm} m{rows} changed-input graph vs eager')
                check(bool(eager.isfinite().all() and eager.abs().sum(dtype=torch.float32) > 0),
                      f'{arm} m{rows}: second bank is nonfinite or all-zero')
                if reference is not None:
                    exact(torch, v.output(rows), reference, f'candidate vs baseline at m{rows} second bank')
                reference = eager
            cells[rows] = dict(rows=rows, capacity=1 if rows == 1 else 80, distinct_experts=touched,
                               routes=rows * TOPK, touched_weight_bytes=touched * WEIGHT_BYTES,
                               one_read_us=touched * WEIGHT_BYTES / (PEAK_GBPS * 1000),
                               samples={arm: [] for arm in arms})
        print('Oracle passed: exact BF16 words, wider-prefix invariance, changed-input graph replay', flush=True)
        pre = telemetry()
        for block in range(args.blocks):
            for rows in rows_list[::1 if block % 2 == 0 else -1]:
                data, _ = inputs(torch, rows, 'timing')
                order = list(arms)[::1 if (block + rows) % 2 == 0 else -1]
                for arm in order:
                    v = variants[arm][cells[rows]['capacity']]
                    v.load(data, rows)
                    graph = v.graphs[rows]
                    for _ in range(3):
                        graph.replay()
                    cells[rows]['samples'][arm].append(elapsed(torch, graph, args.iterations))
        post = telemetry()
        for cell in cells.values():
            cell['results'] = {}
            for arm, samples in cell['samples'].items():
                us = statistics.median(samples)
                cell['results'][arm] = dict(us=us, min_us=min(samples), max_us=max(samples),
                                            effective_gbps=cell['touched_weight_bytes'] / (us * 1000))
            if len(arms) == 2:
                cell['paired_speedup'] = statistics.median(b / v for b, v in zip(
                    cell['samples']['baseline'], cell['samples']['candidate']))
        return dict(host=args.host or socket.gethostname(), torch=torch.__version__,
                    sms=torch.cuda.get_device_properties(0).multi_processor_count,
                    peak_gbps=PEAK_GBPS, bytes_per_expert=WEIGHT_BYTES,
                    routing='synthetic skew fitted to the #2462 mean touched counts',
                    oracle='bitwise BF16 (two banks); wider batch prefix; changed-input replays',
                    iterations=args.iterations, blocks=args.blocks, thermal_before=pre, thermal_after=post,
                    packages={arm: {cap: {k: v.meta[k] for k in ('sparkinfer_revision', 'tile', 'blocks_per_sm',
                        'objects', 'decode_schedule') if k in v.meta} for cap, v in by_cap.items()}
                        for arm, by_cap in variants.items()}, cells=list(cells.values()))
    finally:
        for by_cap in variants.values():
            for v in by_cap.values():
                v.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--rows', type=int, nargs='+', default=list(range(1, 65)))
    p.add_argument('--iterations', type=int, default=24)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--host', help='physical host name when running inside a container')
    p.add_argument('--json', type=Path, required=True)
    args = p.parse_args()
    if not all(1 <= rows <= 64 for rows in args.rows) or args.iterations < 1 or args.blocks < 2:
        p.error('rows must be in 1..64, iterations positive and blocks >=2')
    import torch
    result = run(args, torch)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    for cell in result['cells']:
        parts = [f'{arm}={r["us"]:.2f} us/{r["effective_gbps"]:.1f} GB/s' for arm, r in cell['results'].items()]
        print(f'm{cell["rows"]:02d}/{cell["capacity"]}: {cell["distinct_experts"]:3d} experts, '
              f'floor={cell["one_read_us"]:.2f} us; ' + '; '.join(parts), flush=True)


if __name__ == '__main__':
    main()
