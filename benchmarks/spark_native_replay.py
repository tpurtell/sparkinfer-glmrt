"""Adapters for sealed Spark expert exports; no compilation or host-side routing."""

import ctypes as c
import hashlib
import json
import math
import os

import torch

from spark_expert_replay import _bind, _check, _ProgramInfo

P, U, I, Q = c.c_void_p, c.c_uint32, c.c_int32, c.c_uint64


def _read(path):
    body = path.read_bytes()
    return json.loads(body), hashlib.sha256(body).hexdigest()


def _lib(root):
    return c.CDLL(os.environ.get('CUTEAFD_NATIVE_LIB', str(root / 'lib/libcuteafd_native.so')))


def _byte(count, value=None):
    if value is not None:
        return torch.full((count,), value, device='cuda', dtype=torch.uint8)
    return torch.empty(count, device='cuda', dtype=torch.uint8)


def _pointers(tensors):
    return (P * len(tensors))(*(t.data_ptr() if t is not None else 0 for t in tensors))


class _Wave:
    def __init__(self, capacity, hidden):
        self.capacity = capacity
        self.out = torch.empty((capacity, hidden), device='cuda', dtype=torch.bfloat16)

    def poison_output(self):
        self.out.fill_(0x7F if self.out.dtype == torch.uint8 else float('nan'))

    def output(self, rows):
        return self.out[:rows]

    def capture(self, rows):
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            self.run(rows)
        return graph

    def close(self):
        pass


class _Plan:
    def select(self, rows):
        return {m: next(cap for cap in self.capacities if cap >= m) for m in rows}


def _reply_kind(shape):
    return {'bf16': 0, 'fp8': 1, 'nvfp4': 2}[shape.get('reply', 'bf16')]


def _reply_output(cap, h, reply):
    stride = h + h // 32 if reply == 1 else h // 2 + h // 16
    return _byte(cap * stride).view(cap, stride)


class _Fp8Wave(_Wave):
    def __init__(self, plan, cap):
        super().__init__(cap, plan.h)
        self.plan = plan
        self.scratch = _byte(plan.scratch[cap])
        self.epilogues = {}
        if plan.reply:
            self.out = _reply_output(cap, plan.h, plan.reply)

    def load(self, data, rows):
        x, ids, routing = (t[:rows] for t in data)
        if self.plan.input == 'bf16':
            scale = torch.exp2(x[:, self.plan.h:].float() - 127).unsqueeze(-1)
            x = (x[:, :self.plan.h].contiguous().view(torch.float8_e4m3fn).float()
                 .reshape(rows, -1, 32) * scale).reshape(rows, self.plan.h).bfloat16()
        elif self.plan.input == 'nvfp4':
            # Calibrated K16 ingress is packed E2M1, E4M3 scales and a zero tail.
            # Materialize it before capture: the Spark does not own this quantizer.
            h = self.plan.h
            packed = torch.zeros((rows, h // 2 + h // 16 + 16), device=x.device, dtype=torch.uint8)
            codes = x[:, :h:2].bitwise_and(15) | x[:, 1:h:2].bitwise_left_shift(4)
            packed[:, :h // 2] = codes
            packed[:, h // 2:h // 2 + h // 16] = 0x20
            x = packed
        self.inputs = x, ids, routing
        self.ptr = _pointers((*self.inputs, *self.plan.weights, self.out, self.scratch))

    def run(self, rows):
        stream = torch.cuda.current_stream().cuda_stream
        plan = self.plan
        if plan.reply and rows not in self.epilogues:
            words = (Q * 4)()
            _check(plan.reply_layout(self.capacity, 0, rows, 0, words, 4), 'FP8 MoE reply layout')
            ver, pair, y, size = words
            pair_bytes = rows * plan.topk * 4
            if ver != 1 or pair + pair_bytes > self.scratch.nbytes or y + size > self.scratch.nbytes:
                raise ValueError('invalid FP8 MoE reply scratch layout')
            operands = dict(y=self.scratch[y:y + size], pair_pos=self.scratch[pair:pair + pair_bytes],
                            weights=self.inputs[2], out=self.out)
            self.epilogues[rows] = plan.programs.bind(plan.reply_spec, operands)
        _check(plan.launch(plan.context, self.capacity, self.ptr, rows, stream), 'FP8 MoE launch')
        if plan.reply:
            self.epilogues[rows]((rows,), stream)

    def close(self):
        self.plan.close()


class _Fp8Plan(_Plan):
    def __init__(self, shape, root):
        folder = root / shape['package']
        self.layout_name = shape['layout']
        manifest, self.digest = _read(folder / 'manifest.json')
        meta = self.meta = manifest['layouts'][self.layout_name]
        self.h, self.n, self.e = (shape[k] for k in ('hidden', 'intermediate', 'experts'))
        self.topk = shape['top_k']
        self.input = meta['input']
        self.reply, self.root = _reply_kind(shape), root
        if self.reply and (shape['format'] == 'nvfp4' or not meta['reply_producers']):
            raise ValueError('this FP8 MoE package has no compressed reply core')
        self.storage = {'mxfp4': (2,), 'mxfp8': (6,), 'nvfp4': (4, 7, 8)}[shape['format']]
        self.nv = shape['format'] == 'nvfp4'
        expected = self.h, self.n, self.e, self.topk
        if tuple(meta[k] for k in ('hidden', 'slice', 'experts', 'top_k')) != expected:
            raise ValueError(f'FP8 fixture/manifest geometry mismatch: {expected}')
        if meta['weights'] != shape['format'] or meta['abi_version'] != 6:
            raise ValueError('FP8 fixture/manifest format mismatch')
        self.scratch = {p['capacity']: p['scratch_bytes'] for p in meta['capacities']}
        self.capacities = tuple(sorted(self.scratch))
        self.path = folder / self.layout_name / 'libcuteafd_fp8moe.so'
        self.core = manifest['files'][str(self.path.relative_to(folder))]['sha256']
        self.context = P()
        self.weights = []
        self.plane_sizes = []
        for nr, nc in ((self.n, self.h), (self.n, self.h), (self.h, self.n)):
            stride = nc // 16 if self.nv else math.ceil((nc // 32) / 4) * 4
            self.plane_sizes.append((nr * nc // (1 if self.storage == (6,) else 2), nr * stride))

    def _open(self, seed):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.core:
            raise ValueError('FP8 MoE library digest mismatch')
        self.lib = c.CDLL(str(self.path))
        sym = lambda suffix, types, result=I: _bind(self.lib, 'cuteafd_fp8moe_' + suffix, types, result)
        words = (U * 18)()
        _check(sym('info', [c.POINTER(U), U])(words, len(words)), 'FP8 MoE info')
        version, storage, _, projection = words[0].to_bytes(4, 'little')
        if (version != 6 or storage not in self.storage or projection != 1
                or list(words)[1:5] != [self.h, self.n, self.e, self.topk]
                or words[7] != {'bf16': 1, 'wire': 7, 'nvfp4': 8}[self.input]
                or tuple(words[10:10 + words[9]]) != self.capacities):
            raise ValueError('FP8 MoE library/manifest contract mismatch')
        scratch_query = sym('scratch_bytes', [U, c.POINTER(Q)])
        for cap, size in self.scratch.items():
            found = Q()
            _check(scratch_query(cap, c.byref(found)), 'FP8 MoE scratch query')
            if found.value != size:
                raise ValueError('FP8 MoE scratch extent mismatch')
        # A fixed random signed code pattern and legal scale grids; no dequantized weights.
        rng = torch.Generator(device='cuda').manual_seed(seed)
        for code_size, scale_size in self.plane_sizes:
            codes = torch.randint(0, 256, (code_size * self.e,), device='cuda',
                                  dtype=torch.uint8, generator=rng)
            if storage == 6:
                codes.bitwise_and_(0xbf)  # finite E4M3 with modest magnitude
            scales = _byte((scale_size + (8 if self.nv else 0)) * self.e,
                           0x20 if self.nv else 121)
            if self.nv:
                scales[scale_size * self.e:].view(torch.float32).fill_(1.)
            self.weights.extend((codes, scales))
        self.launch = sym('launch_core' if self.reply else 'launch', [P, U, c.POINTER(P), I, P])
        if self.reply:
            self.reply_layout = sym('reply_layout', [U, U, U, U, c.POINTER(Q), U])
            self.programs = _ProgramTable(self.root)
            self.reply_spec = self.programs.specs[f'expert_reply_fp8_h{self.h}_k{self.topk}_{self.reply}']
        self.destroy = sym('destroy', [P], None)
        _check(sym('create', [c.POINTER(P)])(c.byref(self.context)), 'FP8 MoE create')

    def make(self, cap, seed):
        if not self.context:
            self._open(seed)
        return _Fp8Wave(self, cap)

    def close(self):
        if self.context:
            self.destroy(self.context)
            self.context = P()

    def metadata(self, cap, rows, distinct):
        route = next((r for r in self.meta['arithmetic_routes']
                      if r['capacity'] == cap and r['form'] == 'auto'), None)
        route_id = next((s['route_id'] for s in route['segments'] if s['end_rows'] >= rows), 0) if route else 0
        return dict(weight_bytes=len(distinct) * (sum(w + s for w, s in self.plane_sizes)
                    + (24 if self.nv else 0)), manifest_sha256=self.digest, core_sha256=self.core,
                    program=f'{self.layout_name}/m{cap}/{("default", "decode", "prefill", "stream")[route_id]}',
                    reply=self.reply,
                    tile=None, route_block=None, ws=False)


class _Info(c.Structure):
    _fields_ = [(k, U) for k in ('abi', 'role', 'experts', 'hidden', 'logical', 'width', 'topk', 'capacity')]
    _fields_ += [('scratch', Q)] + [(k, I) for k in ('max_rows', 'padded', 'tasks', 'tiles', 'clusters')]
    _fields_ += [('input_dtype', U)]


class _Launch(c.Structure):
    _fields_ = [('tensors', P * 44)] + [(k, I) for k in (
        'rows', 'max_rows', 'scatter', 'padded', 'tasks', 'tiles', 'clusters')]
    _fields_ += [('stream', P)]


class _Mxfp4Wave(_Wave):
    def __init__(self, plan, cap):
        super().__init__(cap, plan.h)
        self.plan = plan
        info = self.info = _Info()
        self.handle = P()
        _check(plan.info(cap, c.byref(info)), 'MXFP4 info')
        variant = plan.variants[cap]
        if (info.capacity != cap or info.scratch != variant['core_scratch_nbytes'] or info.input_dtype != 7
                or (info.hidden, info.width, info.experts, info.topk) != (plan.h, plan.n, plan.e, plan.topk)):
            raise ValueError('native MXFP4 library/manifest contract mismatch')
        _check(plan.init(cap, c.byref(self.handle)), 'MXFP4 initialize')
        self.scratch = _byte(info.scratch)
        self.slots = (P * 44)()
        _check(plan.bind(self.handle, self.scratch.data_ptr(), self.scratch.nbytes, self.slots), 'bind scratch')
        actual = [p - self.scratch.data_ptr() if p else None for p in self.slots]
        if actual != variant['scratch_pointer_offsets']:
            raise ValueError('native MXFP4 pointer bindings differ from manifest')
        _check(plan.reset(self.handle, self.scratch.data_ptr(), self.scratch.nbytes,
                          torch.cuda.current_stream().cuda_stream), 'initialize scratch')
        w13, s13, w2, s2 = (t.data_ptr() for t in plan.weights)
        for slot, p in zip(range(22, 34), (w13, s13, w2, s2, s13, s2, self.slots[34],
                                           self.slots[34], w13, s13, w2, s2)):
            self.slots[slot] = p
        self.slots[38] = self.slots[39] = self.slots[37]
        kind = variant['output_kind']
        if plan.reply:
            self.out = _reply_output(cap, plan.h, plan.reply)
            offset = variant['scratch_pointer_offsets'][41]
            spec = plan.programs.specs[f'expert_reply_native_h{plan.h}_{plan.reply}']
            self.reduce = plan.programs.bind(spec, dict(source=self.scratch[offset:], out=self.out))
            self.source_kind = ('fp32_tokens', 'fp32_routes', 'bf16_routes').index(kind)
        else:
            self.reduce = plan.reducers[kind]

    def load(self, data, rows):
        self.inputs = tuple(t[:rows] for t in data)
        for slot, tensor in enumerate(self.inputs):
            self.slots[slot] = tensor.data_ptr()
        if not all(self.slots):
            raise ValueError('missing native MXFP4 launch pointer')

    def run(self, rows):
        info = self.info
        stream = torch.cuda.current_stream().cuda_stream
        args = _Launch(self.slots, rows, info.max_rows, rows * info.topk, info.padded, info.tasks,
                       info.tiles, info.clusters, stream)
        _check(self.plan.launch(self.handle, c.byref(args)), 'MXFP4 launch')
        if self.plan.reply:
            self.reduce((rows, info.topk, self.source_kind), stream)
        else:
            _check(self.reduce(self.slots[41], self.out.data_ptr(), rows, stream), 'MXFP4 BF16 compact')


class _Mxfp4Plan(_Plan):
    def __init__(self, shape, root):
        self.root = root
        self.role, self.reply = shape['role'], _reply_kind(shape)
        manifest, self.digest = _read(root / 'share/V41_EXPERT_TP_AOT.json')
        self.meta = manifest['native_packages'][self.role]
        self.h, self.n, self.e, self.topk = (shape[k] for k in ('hidden', 'intermediate', 'experts', 'top_k'))
        geom = self.meta['geometry']
        if (geom['hidden'], geom['kernel_intermediate'], geom['experts'], geom['topk']) != (
                self.h, self.n, self.e, self.topk):
            raise ValueError('native MXFP4 fixture/manifest geometry mismatch')
        self.variants = {v['capacity_rows']: v for v in self.meta['variants']}
        self.capacities = tuple(sorted(self.variants))
        self.decode = tuple(self.meta['decode_capacities'])
        if not self.decode or self.decode[0] != 1 or any(
                self.variants[cap]['numerical_recipe'] != self.variants[1]['numerical_recipe'] for cap in self.decode):
            raise ValueError('native MXFP4 lacks canonical decode recipes')
        self.weights = None

    def _open(self, seed):
        self.lib = _lib(self.root)
        if self.reply:
            self.programs = _ProgramTable(self.root)
        _check(_bind(self.lib, 'cuteafd_set_expert_hidden', [U])(self.h), 'set expert hidden')
        family, role = self.role.split(':')
        prefix = f'cuteafd_{family}' + (f'_{role}' if role != 'spark' else '') + '_expert_'
        sym = lambda name, types: _bind(self.lib, prefix + name, types)
        self.info = sym('info', [I, c.POINTER(_Info)])
        self.init = sym('initialize', [I, c.POINTER(P)])
        self.bind = sym('bind_scratch', [P, P, Q, c.POINTER(P)])
        self.reset = sym('initialize_scratch_async', [P, P, Q, P])
        self.launch = sym('launch', [P, c.POINTER(_Launch)])
        self.reducers = {kind: _bind(self.lib, 'cuteafd_compact_' + name + '_async', [P, P, U, P])
                         for kind, name in (('fp32_routes', 'routes_bf16'), ('fp32_tokens', 'tokens_bf16'),
                                            ('bf16_routes', 'bf16_routes'))}
        rng = torch.Generator(device='cuda').manual_seed(seed)
        self.weights = tuple(_byte(size * self.e, 121) if i % 2 else
                             torch.randint(0, 256, (size * self.e,), device='cuda',
                                           dtype=torch.uint8, generator=rng)
                             for i, size in enumerate(self.meta['expert_plane_bytes']))

    def make(self, cap, seed):
        if self.weights is None:
            self._open(seed)
        return _Mxfp4Wave(self, cap)

    def select(self, rows):
        return {m: next(cap for cap in (self.decode if m <= self.decode[-1] else self.capacities)
                        if cap >= m) for m in rows}

    def metadata(self, cap, rows, distinct):
        v = self.variants[cap]
        stem = next(k for k in self.meta['artifact_sha256'] if k.endswith(f'_m{cap}.o'))
        return dict(weight_bytes=len(distinct) * sum(self.meta['expert_plane_bytes']), manifest_sha256=self.digest,
                    core_sha256=self.meta['artifact_sha256'][stem], program=v['name'], reply=self.reply,
                    tile=v.get('width'), route_block=v.get('group_rows'), ws=v['implementation'] == 'native_stream')


def native_plan(shape, root):
    fmt = shape['format']
    if fmt == 'mxfp4' and 'role' in shape:
        return _Mxfp4Plan(shape, root)
    if fmt in ('mxfp4', 'mxfp8', 'nvfp4'):
        return _Fp8Plan(shape, root)
    if fmt == 'exlr8':
        return _Exlr8Plan(shape, root)
    raise ValueError(f'unsupported native replay format: {fmt}')


class _ProgramTable:
    def __init__(self, root):
        self.lib = lib = _lib(root)
        self.specs = {p['name']: p for p in _read(root / 'share/PROGRAMS.json')[0]['programs']}
        count = _bind(lib, 'cuteafd_dsv4_program_count', [], U)
        info = _bind(lib, 'cuteafd_dsv4_program_info', [U, c.POINTER(_ProgramInfo)])
        self.load = _bind(lib, 'cuteafd_dsv4_program_load', [U])
        self.launch = _bind(lib, 'cuteafd_dsv4_program_launch', [U, c.POINTER(P), c.POINTER(Q), P])
        self.records = {}
        for i in range(count()):
            rec = _ProgramInfo()
            _check(info(i, c.byref(rec)), 'native program info')
            self.records[rec.name.decode()] = i, rec

    def bind(self, spec, tensors):
        name = spec['name']
        index, rec = self.records[name]
        ptr = _pointers([tensors[p['name']] for p in spec['pointers']])
        if (len(ptr), len(spec['scalars']), rec.scalar_kinds) != (rec.pointers, rec.scalars, b'i' * rec.scalars):
            raise ValueError(f'native table manifest/ABI mismatch for {name}')
        _check(self.load(index), f'load {name}')

        def run(values, stream):
            if len(values) != rec.scalars:
                raise ValueError(f'wrong scalar count for {name}')
            _check(self.launch(index, ptr, (Q * len(values))(*values), stream), name)
        return run


class _Exlr8Wave(_Wave):
    def __init__(self, plan, kind):
        core, group = plan.specs[kind]
        super().__init__(core['capacity_rows'], plan.h)
        self.plan, self.core, self.group = plan, core, group
        geom = core['geometry']
        g, r = geom['max_groups'], geom.get('group_rows', 16)
        def tensor(shape, dtype):
            return torch.empty(shape, device='cuda', dtype=dtype)
        ws = {name: _byte(size // plan.units * plan.unit_limit if kind > 0 and name != 'fc2'
                          or kind == 0 and name == 'fc1' else size)
              for name, size in core['scratch_bytes_at_capacity'].items()}
        self.values = dict(plan.tables, **ws, output=self.out,
                           group_expert=tensor(g, torch.int32), route_slots=tensor((g, r), torch.int32),
                           route_group_lane=tensor((self.capacity, plan.topk), torch.int32),
                           group_count=tensor(1, torch.int32),
                           activation=tensor((self.capacity, plan.h), torch.float16))
        self.group_ctas = plan.sms * group['geometry']['ctas_per_sm']
        self.core_ctas = plan.sms * geom['ctas_per_sm']
        self.owned = kind > 0
        self.epilogue = None
        self.epilogue_spec = None
        if plan.reply:
            self.out = _reply_output(self.capacity, plan.h, plan.reply)
            self.values['output'] = self.out
            if kind == 0:
                self.core = plan.programs.specs[core['name'] + '_core']
                self.epilogue_spec = plan.programs.specs[f'exlr8_reply_{plan.reply}']
            else:
                self.core = plan.programs.specs[f'{core["name"]}_reply{plan.reply}']

    def load(self, data, rows):
        x, ids, weights = self.inputs = tuple(t[:rows] for t in data)
        self.group_run = self.plan.programs.bind(self.group, dict(self.values, ids=ids, x=x))
        operands = dict(self.values, x=self.values['activation'], route_weights=weights)
        self.core_run = self.plan.programs.bind(self.core, operands)
        if self.epilogue_spec:
            self.epilogue = self.plan.programs.bind(self.epilogue_spec, operands)

    def run(self, rows):
        stream = torch.cuda.current_stream().cuda_stream
        self.group_run((rows, self.group_ctas), stream)
        scalars = (rows, self.core_ctas)
        if self.owned:
            scalars += (self.plan.unit_limit, self.plan.l2)
        self.core_run(scalars, stream)
        if self.epilogue:
            self.epilogue(scalars[:2], stream)


class _Exlr8Plan:
    def __init__(self, shape, root):
        self.root = root
        self.h, self.e, self.topk = (shape[k] for k in ('hidden', 'experts', 'top_k'))
        self.units, self.reply = shape['units'], _reply_kind(shape)
        self.owned, self.tiers = shape['owned_units'], shape['unit_bits']
        self.unit_limit = max(map(len, self.owned))
        if len(self.owned) != self.e or len(self.tiers) != self.e:
            raise ValueError('incomplete EXLR8 ownership fixture')
        manifest, self.digest = _read(root / 'share/PROGRAMS.json')
        programs = {p['name']: p for p in manifest['programs']}
        self.specs = [(programs[p['core']], programs[p['group']]) for p in shape['programs']]
        if len(self.specs) != 3 or any(
                (a['geometry']['hidden'], a['geometry']['units'], a['geometry']['top_k'],
                 b['geometry']['experts'], b['geometry']['hidden']) != (self.h, self.units, self.topk, self.e, self.h)
                for a, b in self.specs):
            raise ValueError('EXLR8 fixture/manifest geometry mismatch')
        self.threshold = 16 if self.unit_limit <= 3 else 8
        self.programs = None

    def select(self, rows):
        if max(rows) > self.specs[2][0]['capacity_rows']:
            raise ValueError('rows beyond EXLR8 installed capacity')
        return {m: 2 if m > self.specs[0][0]['capacity_rows'] else int(m >= self.threshold) for m in rows}

    def _open(self, seed):
        self.programs = _ProgramTable(self.root)
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.l2 = _bind(self.programs.lib, 'cuteafd_l2_cache_bytes', [], c.c_int64)()
        if self.l2 <= 0:
            raise ValueError('EXLR8 needs the runtime L2 working-set limit')
        block_size = self.h // 128 * 2
        offsets = []
        per_expert_bytes = []
        offset = 0
        for expert, owned in enumerate(self.owned):
            start = offset
            for unit in owned:
                for projection, bits in enumerate(self.tiers[expert][unit]):
                    count = self.h * 16 * bits
                    offsets.append((expert, unit, projection, bits, offset, count))
                    offset += count
            projections = {(p, b) for u in owned for p, b in enumerate(self.tiers[expert][u])}
            per_expert_bytes.append(offset - start + len(owned) * 3 * 256 + len(projections) * block_size)
        self.weight_bytes = per_expert_bytes
        rng = torch.Generator(device='cuda').manual_seed(seed)
        payload = torch.randint(0, 256, (offset,), device='cuda', dtype=torch.uint8, generator=rng)
        channel = torch.full((len(offsets), 128), 1 / self.h ** .5, device='cuda', dtype=torch.float16)
        block = torch.ones((self.e, 3, 3, self.h // 128), device='cuda', dtype=torch.float16)
        ptr = torch.zeros((self.e, self.units, 3, 3), dtype=torch.int64)
        tier = torch.zeros((self.e, self.units, 3), dtype=torch.int32)
        for i, (e, u, p, bits, start, _) in enumerate(offsets):
            ptr[e, u, p] = torch.tensor([payload.data_ptr() + start, channel[i].data_ptr(),
                                         block[e, p, bits - 2].data_ptr()])
            tier[e, u, p] = bits
        unit_map = [units + [0] * (self.units - len(units)) for units in self.owned]
        signs = torch.ones(self.h, device='cuda', dtype=torch.int8)
        self.tables = dict(pointers=ptr.cuda(), bits=tier.cuda(), signs_in=signs, signs_out=signs,
                           unit_count=torch.tensor(list(map(len, self.owned)), device='cuda', dtype=torch.int32),
                           unit_map=torch.tensor(unit_map, device='cuda', dtype=torch.int32))
        self.planes = payload, channel, block

    def make(self, kind, seed):
        if self.programs is None:
            self._open(seed)
        return _Exlr8Wave(self, kind)

    def metadata(self, kind, rows, distinct):
        core, group = self.specs[kind]
        return dict(weight_bytes=sum(self.weight_bytes[e] for e in distinct), manifest_sha256=self.digest,
                    core_sha256=core['object_sha256'], program=f'{group["name"]}+{core["name"]}', reply=self.reply,
                    tile=None, route_block=core['geometry'].get('group_rows', 16), ws=kind > 0)
