"""Replay planned Spark expert waves in a device-only AOT benchmark."""

import ctypes as c
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch


class ReplayLibraries:
    def __init__(self, root, override=None):
        self.root = root
        self.override = override
        self.loaded = {}

    def load(self, path, *, expected=None):
        path = path.resolve(strict=True)
        if path in self.loaded:
            lib, digest = self.loaded[path]
        else:
            with path.open('rb') as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
                digest = digest.hexdigest()
            lib = None
        if expected is not None and expected != digest:
            raise ValueError(f'replay library digest mismatch: {path}')
        if lib is None:
            lib = c.CDLL(str(path))
            self.loaded[path] = (lib, digest)
        return lib

    def native(self, *, exl3_support=False, expected=None):
        path = self.root / 'lib/libcuteafd_native.so'
        if exl3_support and not path.exists():
            # Standalone EXL3 package roots carry only the expert libraries.
            path = Path('/opt/cuteafd/lib/libcuteafd_native.so')
        return self.load(self.override if self.override is not None else path, expected=expected)

    def receipt(self):
        return [{'path': str(path), 'sha256': digest}
                for path, (_, digest) in sorted(self.loaded.items())]


def routes(shape, rows, seed):
    """Recorded prefixes when supplied, otherwise a stable skewed top-k prior."""
    experts, top_k = shape['experts'], shape['top_k']
    recorded = shape.get('routes', [])
    rng = random.Random(seed)
    popularity = list(range(experts))
    rng.shuffle(popularity)
    weights = [(i + 16) ** -0.85 for i in range(experts)]
    result = list(recorded[:rows])
    for row in result:
        if len(row) != top_k or len(set(row)) != top_k or any(
            type(e) is not int or not 0 <= e < experts for e in row
        ):
            raise ValueError('invalid recorded top-k route')
    while len(result) < rows:
        picked = {}
        while len(picked) < top_k:
            for expert in rng.choices(popularity, weights=weights, k=top_k - len(picked)):
                picked[expert] = None
        result.append(list(picked))
    return result


def _check(status, name):
    if status:
        raise RuntimeError(f'{name} failed: {status}')


def _bind(lib, name, args, result=c.c_int32):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, result
    return fn


class _ProgramInfo(c.Structure):
    _fields_ = [('name', c.c_char * 64), ('pointers', c.c_uint32), ('scalars', c.c_uint32),
                ('scalar_kinds', c.c_char * 16)]


class _SparkIO:
    def __init__(self, v, hidden, experts, lib, epilogue):
        self.v, self.experts = v, experts
        self.input = torch.empty((v.capacity, hidden + hidden // 32), dtype=torch.uint8, device='cuda')
        self.context = c.c_void_p()
        if v.meta.get('input_format') == 'e4m3_k32':
            v.ptr['rotation_input_ptr'] = self.input
        else:
            init = _bind(lib, 'cuteafd_exl3_wire_initialize', [c.POINTER(c.c_void_p)])
            _check(init(c.byref(self.context)), 'wire initialize')
            self.wire = _bind(lib, 'cuteafd_exl3_wire_decode',
                [c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64, c.c_uint32, c.c_void_p])
            self.destroy = _bind(lib, 'cuteafd_exl3_wire_destroy', [c.c_void_p], None)
        self.partial = None
        if epilogue == 'partial_bf16':
            count = _bind(lib, 'cuteafd_dsv4_program_count', [], c.c_uint32)
            info = _bind(lib, 'cuteafd_dsv4_program_info', [c.c_uint32, c.POINTER(_ProgramInfo)])
            load = _bind(lib, 'cuteafd_dsv4_program_load', [c.c_uint32])
            record = _ProgramInfo()
            for index in range(count()):
                _check(info(index, c.byref(record)), 'program info')
                if record.name == b'expert_partial_bf16':
                    if (record.pointers, record.scalars, record.scalar_kinds) != (6, 4, b'iiii'):
                        raise ValueError('unexpected BF16 partial reply ABI')
                    _check(load(index), 'partial load')
                    self.index = index
                    break
            else:
                raise ValueError('missing BF16 partial reply program')
            self.partial = torch.empty((v.capacity, v.top_k, hidden), device='cuda', dtype=torch.bfloat16)
            self.row_map = torch.arange(v.capacity, device='cuda', dtype=torch.int32)
            self.partial_launch = _bind(lib, 'cuteafd_dsv4_program_launch',
                [c.c_uint32, c.POINTER(c.c_void_p), c.POINTER(c.c_uint64), c.c_void_p])

    def decode(self, rows, stream):
        if self.context:
            _check(self.wire(self.context, self.input.data_ptr(), self.input.nbytes,
                             self.v.x.data_ptr(), self.v.x.nbytes, rows, stream), 'wire decode')

    def run_partial(self, rows, stream):
        ptrs = [self.v.ptr[k] for k in ('fc2_ptr', 'svh_ptr', 'raw_topk_ids', 'expert_map_ptr')]
        _check(self.partial_launch(self.index, (c.c_void_p * 6)(*(t.data_ptr() for t in
               (*ptrs, self.row_map, self.partial))), (c.c_uint64 * 4)(rows, rows,
               self.experts, self.experts), stream), 'partial launch')

    def close(self):
        if self.context:
            self.destroy(self.context)


def _weights(shape, seed):
    h, n, e, bits = (shape[k] for k in ('hidden', 'intermediate', 'experts', 'bits'))
    widths = shape['expert_widths']
    projection_bits = shape['projection_bits']
    compact = shape['compact']
    rng = torch.Generator(device='cuda').manual_seed(seed)
    expert_map = torch.arange(e, device='cuda', dtype=torch.int32)
    # Same descriptor stride as the loader, even for a homogeneous layer.
    stride = e * len(bits)
    desc = [[-1] * stride for _ in range(7 if compact else 3)]
    if compact:
        desc[3][:e] = widths
    gs = [[0 for _ in bits] for _ in range(3)]
    offsets = [[0 for _ in bits] for _ in range(3)]
    for p in range(3):
        for expert, width in enumerate(widths):
            tier = bits.index(projection_bits[p][expert])
            desc[p][expert] = (tier << (9 if e <= 512 else 10)) | gs[p][tier]
            if compact:
                desc[p + 4][expert] = offsets[p][tier] + (offsets[0][tier] if p == 1 else 0)
            gs[p][tier] += 1
            offsets[p][tier] += h * width * bits[tier] // 32

    def rotation(dim, norm=1.):
        scale = torch.rand((stride, dim), device='cuda', generator=rng) * .5 + .75
        sign = torch.randint(0, 2, scale.shape, device='cuda', generator=rng) * 2 - 1
        return (scale * sign * norm).half()

    ptr = dict(descriptor_map_ptr=torch.tensor(desc, device='cuda', dtype=torch.int32),
               global_to_combined_ptr=expert_map, expert_map_ptr=expert_map,
               intermediate_rotations_ptr=rotation(3 * n), gate_suh_ptr=rotation(h),
               up_suh_ptr=rotation(h), svh_ptr=rotation(h, .01))
    scalar = dict(route_num_experts=e, weight_num_experts=stride)
    stub = torch.zeros(16, dtype=torch.uint8, device='cuda')
    ones = torch.ones(e, dtype=torch.float32, device='cuda')
    for tier in range(len(bits)):
        for kind, words in (('w13', offsets[0][tier] + offsets[1][tier]), ('w2', offsets[2][tier])):
            ptr[f't{tier}_{kind}_ptr'] = torch.randint(-2**31, 2**31 - 1, (words,),
                device='cuda', dtype=torch.int32, generator=rng) if words else stub
            ptr[f't{tier}_{kind}_scales_ptr'] = stub
            ptr[f't{tier}_{kind}_global_ptr'] = ones
        scalar[f'tier{tier}_num_experts'] = e
        for p, projection in enumerate(('gate', 'up', 'fc2')):
            scalar[f'tier{tier}_{projection}_experts'] = gs[p][tier]
    return ptr, scalar


def _replay_buckets(shape, root, rows):
    paths = {int(k): root / v for k, v in shape['buckets'].items()}
    metas = {k: json.loads((p / 'v41_exl3.json').read_bytes()) for k, p in paths.items()}
    dec = sorted(k for k, m in metas.items() if m['decode_capacity'])
    if not dec or dec[0] != 1 or dec[-1] < 80:
        raise ValueError('missing canonical m1 program')
    def recipe(m):
        return (m['tile'], m.get('route_block', 8), m.get('input_format', 'bf16'),
                *(bool(m.get(k)) for k in ('direct', 'warp_specialized', 'fused_input_rotation',
                                          'token_major_rotation')),
                *(m.get(k) for k in ('bits', 'swiglu_limit', 'output_dtype', 'hidden',
                                     'intermediate', 'experts', 'top_k', 'sparkinfer_revision')))
    canonical = [k for k in dec if recipe(metas[k]) == recipe(metas[1])]
    selected = {m: next(k for k in (canonical if m <= dec[-1] else sorted(metas)) if k >= m)
                for m in rows}
    return paths, metas, selected


def _run_shape(args, shape, rows, cls, source, libraries):
    if shape['format'] != 'exl3':
        from spark_native_replay import native_plan
        plan = native_plan(shape, args.aot_root, libraries)
        return _run_replay(args, shape, rows, source, plan.select(rows),
                           plan.select(tuple(shape.get('prefix_rows', (1, 8, 80)))),
                           plan.make, plan.metadata)
    paths, metas, selected = _replay_buckets(shape, args.aot_root, rows)
    resolved = [shape[k] for k in ('hidden', 'intermediate', 'experts', 'top_k', 'bits')]
    h, _, e, _, _ = resolved
    lib = libraries.native(exl3_support=True)
    _check(_bind(lib, 'cuteafd_set_expert_hidden', [c.c_uint32])(h), 'set expert hidden')
    io = lambda v, hidden, experts: _SparkIO(v, hidden, experts, lib, shape['epilogue'])
    weights = _weights(shape, args.seed)

    def make(cap, _seed):
        return cls(torch, paths[cap], cap, weights, resolved, io, load_library=libraries.load)

    def metadata(cap, _rows, distinct):
        return dict(weight_bytes=sum(h * shape['expert_widths'][expert] * sum(
                        p[expert] for p in shape['projection_bits']) // 8 for expert in distinct),
                    manifest_sha256=hashlib.sha256((paths[cap] / 'v41_exl3.json').read_bytes()).hexdigest(),
                    core_sha256=metas[cap]['objects'][0]['object_sha256'], tile=metas[cap]['tile'],
                    route_block=metas[cap].get('route_block', 8), ws=bool(metas[cap].get('warp_specialized')))

    return _run_replay(args, shape, rows, source, selected, {1: 1, 8: 80, 80: 80}, make, metadata)


def _run_replay(args, shape, rows, source, selected, prefixes, make, metadata):
    ids = routes(shape, max(max(rows), max(prefixes), 80), args.seed)
    h, top_k = shape['hidden'], shape['top_k']
    rng = torch.Generator(device='cuda').manual_seed(args.seed + 1)
    x = (torch.randn((len(ids), h), device='cuda', generator=rng) * 2.).to(torch.float8_e4m3fn).view(torch.uint8)
    x = torch.cat((x, torch.full((len(ids), h // 32), 121, device='cuda', dtype=torch.uint8)), dim=1)
    data = (x, torch.tensor(ids, dtype=torch.int32, device='cuda'),
            torch.randn((len(ids), top_k), device='cuda', generator=rng).softmax(-1))
    # GB10 L2 is 24 MiB; normal cached kernel copies touch eight times that.
    cold = torch.ones(192 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
    sink = torch.empty_like(cold)
    variants = {}
    graphs = {}
    cells = {}
    try:
        for cap in sorted(set(selected.values()) | set(prefixes.values())):
            v = variants[cap] = make(cap, args.seed)
            v.load(data, min(v.capacity, len(ids)))
        prefix = None
        prefix_exact = {}
        for m, cap in prefixes.items():
            v = variants[cap]
            v.poison_output()
            v.run(m)
            first = v.output(1).view(torch.uint8)
            if prefix is None:
                prefix = first.clone()
            prefix_exact[m] = torch.equal(prefix, first)
        if shape['format'] == 'exl3' and not all(prefix_exact.values()):
            raise ValueError(f'canonical m1/m8/m80 prefix mismatch: {prefix_exact}')
        for m, cap in selected.items():
            v = variants[cap]
            v.poison_output()
            v.run(m)
            out = v.output(m).clone()
            if not bool(out.count_nonzero()) or out.dtype != torch.uint8 and not bool(out.isfinite().all()):
                raise ValueError(f'invalid eager output at m{m}')
            graph = graphs[m] = v.capture(m)
            v.poison_output()
            graph.replay()
            if not torch.equal(out.view(torch.uint8), v.output(m).view(torch.uint8)):
                raise ValueError(f'eager/graph bit mismatch at m{m}')
            distinct = sorted({expert for row in ids[:m] for expert in row})
            cells[m] = dict(rows=m, capacity=v.capacity, distinct=len(distinct), samples_us=[],
                            digest=hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                            **metadata(cap, m, distinct))
            for _ in range(args.warmup if not args.profile_replay else 1):
                graph.replay()
        order = list(rows)
        order_rng = random.Random(args.seed + 2)
        if args.profile_replay:
            for m in order:
                sink.copy_(cold)
                torch.cuda.synchronize()
                label = f'spark_floor:{shape["id"]}:{m}'
                torch.cuda.nvtx.range_push(label)
                graphs[m].replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        elif not args.correctness_only:
            for _ in range(args.rounds):
                order_rng.shuffle(order)
                events = []
                for rep in range(args.replays):
                    for m in order:
                        if m >= 256 and rep >= max(1, args.replays // 4):
                            continue
                        start = torch.cuda.Event(enable_timing=True)
                        stop = torch.cuda.Event(enable_timing=True)
                        sink.copy_(cold)
                        start.record()
                        graphs[m].replay()
                        stop.record()
                        events.append((m, start, stop))
                torch.cuda.synchronize()
                for m, start, stop in events:
                    cells[m]['samples_us'].append(start.elapsed_time(stop) * 1000)
        for cell in cells.values():
            samples = cell.pop('samples_us')
            if samples:
                cell.update(median_us=statistics.median(samples), samples=len(samples),
                            min_us=min(samples), max_us=max(samples))
        return dict(id=shape['id'], source_sha256=source, device=torch.cuda.get_device_name(),
                    prefix_exact=prefix_exact, routes_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), cells=list(cells.values()))
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        for variant in variants.values():
            variant.close()


def replay_spark_shapes(args, cls, source):
    fixture = json.loads(args.spark_replay.read_bytes())
    if fixture.get('schema') != 'spark-expert-replay.v1' or not args.aot_root:
        raise ValueError('Spark replay requires a schema v1 fixture and --aot-root')
    rows = sorted(set(map(int, args.rows.split(',')))) if args.rows else fixture['rows']
    shapes = [s for s in fixture['shapes'] if not args.replay_shape or s['id'] in args.replay_shape]
    if len(shapes) != 1:
        raise ValueError('select exactly one --replay-shape per process (native hidden size is process-global)')
    shape = shapes[0]
    if not rows or rows[0] < 1:
        raise ValueError('unsupported replay format or invalid live rows')
    libraries = ReplayLibraries(args.aot_root, args.replay_native_library)
    if args.preflight_only:
        if shape['format'] == 'exl3':
            plan = _replay_buckets(shape, args.aot_root, rows)[2]
        else:
            from spark_native_replay import native_plan
            native = native_plan(shape, args.aot_root, libraries)
            native.select(tuple(shape.get('prefix_rows', (1, 8, 80))))
            plan = native.select(rows)
        print(json.dumps(plan))
        return
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError('Spark AOT replay requires SM121')
    report = _run_shape(args, shape, rows, cls, source, libraries)
    report['libraries'] = libraries.receipt()
    report['fixture_sha256'] = hashlib.sha256(args.spark_replay.read_bytes()).hexdigest()
    if args.replay_reference:
        reference = json.loads(args.replay_reference.read_bytes())
        if any(reference[k] != report[k] for k in ('id', 'routes_sha256')):
            raise ValueError('replay reference uses different routes or shape')
        expected = {c['rows']: c['digest'] for c in reference['cells']}
        for cell in report['cells']:
            if expected.get(cell['rows']) != cell['digest']:
                raise ValueError(f"reference output differs at m{cell['rows']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
