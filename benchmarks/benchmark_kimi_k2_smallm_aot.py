#!/usr/bin/env python3
"""Measure shipping Kimi K2 TP6 decode AOT objects on GB10, without a model.

Pass an engine package's tp6-rank0 directory as --baseline and, optionally,
--candidate. Defaults cover every live row 1..64. The harness follows the
shipping selector (m1 for one row, m80 otherwise; Kimi does not ship m16).
Use --candidate-capacities to test additional decode buckets. Both arms share
the same full-width weights and route banks. Timing includes
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


def decode_mcg(torch, payload, rows, columns):
    """Vectorized checkpoint-layout K2 decoder, independent of AOT kernels.

    Same window, mapping and lane permutation as decode_mcg_matrix in
    benchmark_mixed_trellis.py, batched over tiles.
    """
    lane = torch.arange(32, device=payload.device)[:, None]
    word = torch.arange(8, device=payload.device)[None, :]
    end = (lane * 8 + word + 257) * 2
    first, last = (end - 16) // 32, (end - 1) // 32
    words = payload.reshape(rows // 16, columns // 16, 16).long() & 0xffffffff
    merged = (words[..., first % 16] << 32) | words[..., last % 16]
    window = (merged >> ((last + 1) * 32 - end)) & 0xffff
    value = ((window * 0xcbac1fed) & 0x8fff8fff) ^ 0x3b603b60
    low = (value & 0xffff).short().view(torch.float16)
    high = (value >> 16).short().view(torch.float16)
    row = (lane % 4) * 2 + (word % 4 // 2) * 8 + word % 2
    col = 2 * (lane // 8 + 4 * (word >= 4)) + ((lane >> 2) & 1)
    output = torch.empty((*words.shape[:2], 256), dtype=torch.float32, device=payload.device)
    output[..., (row * 16 + col).flatten()] = (low + high).flatten(-2).float()
    return output.reshape(rows // 16, columns // 16, 16, 16).transpose(1, 2).reshape(rows, columns)


def f32_oracle(torch, weights, banks):
    """Full-F32 ordinary K2/H128/Situ, decoding one expert at a time."""
    ptr, _ = weights
    x, ids, gates = (torch.cat(parts) for parts in zip(*banks))
    h = torch.ones((1, 1), dtype=torch.float32, device=x.device)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= math.sqrt(128)

    def had(v):
        return (v.reshape(-1, v.shape[-1] // 128, 128) @ h).reshape_as(v)

    route_outputs = torch.empty((*ids.shape, HIDDEN), device=x.device, dtype=torch.float32)
    by_expert = {}
    for token, row in enumerate(ids.tolist()):
        for slot, expert in enumerate(row):
            by_expert.setdefault(expert, []).append((token, slot))
    for expert, routes in sorted(by_expert.items()):
        token, slot = torch.tensor(routes, device=x.device).unbind(1)
        gate_svh, up_svh, down_suh = ptr['intermediate_rotations_ptr'][expert].float().split(WIDTH)
        gate = decode_mcg(torch, ptr['t0_w13_ptr'][expert], HIDDEN, WIDTH)
        up = decode_mcg(torch, ptr['t0_w13_ptr'][EXPERTS + expert], HIDDEN, WIDTH)
        down = decode_mcg(torch, ptr['t0_w2_ptr'][expert], WIDTH, HIDDEN)
        source = x[token].float()
        g = had(had(source * ptr['gate_suh_ptr'][expert].float()) @ gate) * gate_svh
        u = had(had(source * ptr['up_suh_ptr'][expert].float()) @ up) * up_svh
        activated = (4 * torch.tanh(g / 4) * torch.sigmoid(g)) * (25 * torch.tanh(u / 25))
        y = had(had(activated * down_suh) @ down) * ptr['svh_ptr'][expert].float()
        route_outputs[token, slot] = y * gates[token, slot, None]
    result = torch.zeros_like(route_outputs[:, 0])
    for slot in range(TOPK):
        result += route_outputs[:, slot]
    return result.split([len(bank[0]) for bank in banks])


def error_by_row(value, reference):
    difference = value.double() - reference.double()
    return list(map(list, zip(difference.abs().amax(1).tolist(),
                              difference.square().mean(1).sqrt().tolist())))


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


def choose_capacity(variants, rows):
    return min(cap for cap in variants if rows <= cap)


def verify(torch, name, variants, rows, data, *, capture):
    v = variants[choose_capacity(variants, rows)]
    v.load(data, rows)
    v.poison_output()
    v.run(rows)
    eager = v.output(rows).clone()
    if capture:
        v.capture(rows)
    v.poison_output()
    v.graphs[rows].replay()
    out = v.output(rows).clone()
    exact(torch, out, eager, f'{name} m{rows} graph vs eager')
    check(bool(out.isfinite().all()) and bool(out.abs().sum(dtype=torch.float32) > 0),
          f'{name} m{rows}: nonfinite or all-zero graph output')

    # Check each row through m1 and every overlapping decode export. Full
    # capacities test both ends; an additional check uses this exact live size.
    mismatches = {}
    for cap, other in variants.items():
        for start in range(0, rows, cap):
            count = min(rows - start, cap)
            for live, offset in sorted({(count, 0), (cap, 0), (cap, cap - count)}):
                first = (start - offset) % len(data[0])
                arranged = tuple(t.roll(-first, 0) for t in data)
                other.load(arranged, live)
                other.poison_output()
                other.run(live)
                actual = other.output(live)[offset:offset + count]
                bad = int((actual.view(torch.int16) != out[start:start + count].view(torch.int16)).sum())
                if bad:
                    mismatches[f'm{cap}/live{live}:{start}+{count}@{offset}'] = bad
    return out, mismatches


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
    torch.set_float32_matmul_precision('highest')
    arms = {'baseline': (args.baseline, [1, 80])}
    if args.candidate:
        arms['candidate'] = (args.candidate, sorted(set(args.candidate_capacities)))
    weights = make_weights(torch)
    variants = {arm: {cap: Variant(torch, root / f'm{cap}', cap, weights) for cap in capacities}
                for arm, (root, capacities) in arms.items()}
    rows_list = sorted(set(args.rows))
    probes = {(rows, bank): inputs(torch, rows, bank) for rows in rows_list for bank in ('oracle', 'timing')}
    expected = dict(zip(probes, f32_oracle(torch, weights, [tuple(t[:rows] for t in data)
                         for (rows, _), (data, _) in probes.items()])))
    try:
        cells = {}
        for rows in rows_list:
            touched = probes[rows, 'timing'][1]
            cell = cells[rows] = dict(rows=rows, capacity=choose_capacity(variants['baseline'], rows),
                    capacities={arm: choose_capacity(by_cap, rows) for arm, by_cap in variants.items()},
                    distinct_experts=touched, routes=rows * TOPK, touched_weight_bytes=touched * WEIGHT_BYTES,
                    one_read_us=touched * WEIGHT_BYTES / (PEAK_GBPS * 1000), samples={arm: [] for arm in arms},
                    validation={})
            for bank in ('oracle', 'timing'):
                data, _ = probes[rows, bank]
                reference = expected[rows, bank]
                control_errors = None
                control_out = None
                checks = cell['validation'][bank] = {}
                for arm, by_cap in variants.items():
                    out, bad = verify(torch, arm, by_cap, rows, data, capture=bank == 'oracle')
                    errors = error_by_row(out, reference)
                    if arm == 'baseline':
                        control_errors, control_out = errors, out
                    worse = [i for i, (err, ctrl) in enumerate(zip(errors, control_errors))
                             if not all(e <= c for e, c in zip(err, ctrl))]
                    checks[arm] = dict(row_invariance_mismatches=bad, max_abs_rms_by_row=errors,
                                       worse_rows=worse, bitwise_to_baseline=torch.equal(
                                           out.view(torch.int16), control_out.view(torch.int16)))
        qualified = {arm: all(not bank[arm]['row_invariance_mismatches'] and not bank[arm]['worse_rows']
                             for cell in cells.values() for bank in cell['validation'].values()) for arm in arms}
        print(f'Oracle qualification: {qualified}; per-row F32 error and cross-export invariance in JSON', flush=True)
        pre = telemetry()
        for block in range(args.blocks):
            for rows in rows_list[::1 if block % 2 == 0 else -1]:
                data, _ = probes[rows, 'timing']
                order = list(arms)[::1 if (block + rows) % 2 == 0 else -1]
                for arm in order:
                    v = variants[arm][cells[rows]['capacities'][arm]]
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
                    oracle='decoded MCG/full F32 Situ; per-row max abs and RMS <= baseline; '
                           'bitwise across supplied capacities; exact graph/eager (two banks)',
                    qualified=qualified,
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
    p.add_argument('--candidate-capacities', type=int, nargs='+', default=[1, 80])
    p.add_argument('--rows', type=int, nargs='+', default=list(range(1, 65)))
    p.add_argument('--iterations', type=int, default=24)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--host', help='physical host name when running inside a container')
    p.add_argument('--json', type=Path, required=True)
    args = p.parse_args()
    if not all(1 <= rows <= 64 for rows in args.rows) or args.iterations < 1 or args.blocks < 2:
        p.error('rows must be in 1..64, iterations positive and blocks >=2')
    if not ({1, 80} <= set(args.candidate_capacities) <= set(range(1, 81))):
        p.error('candidate capacities must include 1 and 80 and fit the decode band')
    import torch
    result = run(args, torch)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    for cell in result['cells']:
        parts = [f'{arm}={r["us"]:.2f} us/{r["effective_gbps"]:.1f} GB/s' for arm, r in cell['results'].items()]
        print(f'm{cell["rows"]:02d}/{cell["capacity"]}: {cell["distinct_experts"]:3d} experts, '
              f'floor={cell["one_read_us"]:.2f} us; ' + '; '.join(parts), flush=True)
    if not all(result['qualified'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
