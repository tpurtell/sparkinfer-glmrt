"""Replay planned Spark and RTX EXL3 expert waves in a device-only AOT benchmark."""

import ctypes as c
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import torch


_REF_GEMM_CACHE = {}
_REF_HAD_CACHE = {}
_REF_ACT_CACHE = {}
_REF_FC2_ROT_CACHE = {}


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


def _validate_positive_scale(value, name):
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'{name} must be finite and positive, got {value!r}')
    return scale


def _weights(shape, seed):
    h, n, e, bits = (shape[k] for k in ('hidden', 'intermediate', 'experts', 'bits'))
    widths = shape['expert_widths']
    projection_bits = shape['projection_bits']
    compact = shape['compact']
    svh_scale = _validate_positive_scale(shape.get('svh_scale', .01), 'svh_scale')
    global_scale = _validate_positive_scale(shape.get('global_scale', 1.0), 'global_scale')
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
               up_suh_ptr=rotation(h), svh_ptr=rotation(h, svh_scale))
    scalar = dict(route_num_experts=e, weight_num_experts=stride)
    stub = torch.zeros(16, dtype=torch.uint8, device='cuda')
    gscale = (torch.ones(e, dtype=torch.float32, device='cuda') if global_scale == 1.0
              else torch.full((e,), global_scale, dtype=torch.float32, device='cuda'))
    for tier in range(len(bits)):
        for kind, words in (('w13', offsets[0][tier] + offsets[1][tier]), ('w2', offsets[2][tier])):
            ptr[f't{tier}_{kind}_ptr'] = torch.randint(-2**31, 2**31 - 1, (words,),
                device='cuda', dtype=torch.int32, generator=rng) if words else stub
            ptr[f't{tier}_{kind}_scales_ptr'] = stub
            ptr[f't{tier}_{kind}_global_ptr'] = gscale
        scalar[f'tier{tier}_num_experts'] = e
        for p, projection in enumerate(('gate', 'up', 'fc2')):
            scalar[f'tier{tier}_{projection}_experts'] = gs[p][tier]
    return ptr, scalar


def _decode_trellis_f16(payload_i32, k_dim, n_dim, bits, codebook):
    """Independent native Trellis [k16, n16, 8*bits] int32 tile decoder to [K, N] FP16."""
    if codebook not in ('mcg', 'mul1'):
        raise ValueError(f'unsupported codebook: {codebook}')
    device = payload_i32.device
    k16, n16 = k_dim // 16, n_dim // 16
    num_tiles = k16 * n16
    raw_i16 = payload_i32.contiguous().view(torch.int16).reshape(k16, n16, 16 * bits)
    words = raw_i16.view(torch.uint16).reshape(num_tiles, 16 * bits).to(torch.int32) & 0xFFFF
    swapped = words.reshape(num_tiles, 8 * bits, 2).flip(-1).reshape(num_tiles, 16 * bits)
    bit_shifts = torch.arange(15, -1, -1, device=device, dtype=torch.int32)
    bit_plane = ((swapped.unsqueeze(-1) >> bit_shifts) & 1).reshape(num_tiles, 256 * bits)
    ends = torch.arange(1, 257, device=device, dtype=torch.long) * bits - 1
    shifts = torch.arange(16, device=device, dtype=torch.long)
    idx = (ends[:, None] - shifts[None, :]) % (256 * bits)
    states = ((bit_plane[:, idx].to(torch.int64) << shifts[None, None, :]).sum(dim=-1) & 0xFFFF).to(torch.int64)
    if codebook == 'mcg':
        val = ((states * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60
        low = (val & 0xFFFF).to(torch.uint16).view(torch.float16)
        high = ((val >> 16) & 0xFFFF).to(torch.uint16).view(torch.float16)
        decoded = low + high
    else:
        prod = (states * 0x83DCD12D) & 0xFFFFFFFF
        bsum = (prod & 0xFF) + ((prod >> 8) & 0xFF) + ((prod >> 16) & 0xFF) + ((prod >> 24) & 0xFF)
        h = (0x6400 + bsum).to(torch.uint16).view(torch.float16).to(torch.float64)
        kinv = torch.tensor([0x1EEE], dtype=torch.uint16, device=device).view(torch.float16).to(torch.float64)
        kbias = torch.tensor([0xC931], dtype=torch.uint16, device=device).view(torch.float16).to(torch.float64)
        decoded = torch.addcmul(kbias, h, kinv).to(torch.float16)
    row_idx, col_idx = [], []
    for t in range(32):
        r, c_idx = 2 * (t % 4), t // 4
        for rr, cc in ((r, c_idx), (r + 1, c_idx), (r + 8, c_idx), (r + 9, c_idx),
                       (r, c_idx + 8), (r + 1, c_idx + 8), (r + 8, c_idx + 8), (r + 9, c_idx + 8)):
            row_idx.append(rr)
            col_idx.append(cc)
    tiles = torch.empty((num_tiles, 16, 16), dtype=torch.float16, device=device)
    tiles[:, torch.tensor(row_idx, device=device), torch.tensor(col_idx, device=device)] = decoded
    return tiles.reshape(k16, n16, 16, 16).permute(0, 2, 1, 3).reshape(k_dim, n_dim).contiguous()


def _own_codebook_reference(shape, weights, wire_x, topk_ids, topk_weights, meta, rows):
    """Independent own-codebook m16n8k16 + H128 bitwise reference (zero b12x kernel imports)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import Float16, Float32, Int32, Uint32, T
    from cutlass._mlir.dialects import llvm

    @cute.jit
    def _pack_half2(src: cute.Tensor, i0: Int32, i1: Int32) -> Uint32:
        a = src[i0].to(Float32)
        b = src[i1].to(Float32)
        return Uint32(llvm.inline_asm(
            T.i32(), [Float32(a).ir_value(), Float32(b).ir_value()],
            'cvt.rn.f16x2.f32 $0, $2, $1;', '=r,f,f',
            has_side_effects=False, is_align_stack=False))

    @cute.jit
    def _mma_m16n8k16(d0: Float32, d1: Float32, d2: Float32, d3: Float32,
                      a0: Uint32, a1: Uint32, a2: Uint32, a3: Uint32, b0: Uint32, b1: Uint32):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
            [Uint32(a0).ir_value(), Uint32(a1).ir_value(), Uint32(a2).ir_value(), Uint32(a3).ir_value(),
             Uint32(b0).ir_value(), Uint32(b1).ir_value(),
             Float32(d0).ir_value(), Float32(d1).ir_value(), Float32(d2).ir_value(), Float32(d3).ir_value()],
            'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
            '{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};',
            '=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f',
            has_side_effects=False, is_align_stack=False)
        return (Float32(llvm.extractvalue(T.f32(), res, [0])),
                Float32(llvm.extractvalue(T.f32(), res, [1])),
                Float32(llvm.extractvalue(T.f32(), res, [2])),
                Float32(llvm.extractvalue(T.f32(), res, [3])))

    @cute.jit
    def _had128_quad(v0: Float32, v1: Float32, v2: Float32, v3: Float32, lane: Int32):
        s0, d0 = v0 + v1, v0 - v1
        s1, d1 = v2 + v3, v2 - v3
        h0, h1, h2, h3 = s0 + s1, d0 + d1, s0 - s1, d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0, h1, h2, h3 = p0 - h0, p1 - h1, p2 - h2, p3 - h3
            else:
                h0, h1, h2, h3 = p0 + h0, p1 + h1, p2 + h2, p3 + h3
        rs = Float32(0.088388347648)
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    def run_had128(x_f32):
        blocks = x_f32.contiguous().reshape(-1, 128)
        nblk = int(blocks.shape[0])
        out = torch.empty_like(blocks)
        if nblk not in _REF_HAD_CACHE:
            @cute.kernel
            def _had_k(src: cute.Tensor, dst: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                blk, _, _ = cute.arch.block_idx()
                base = blk * Int32(128) + lane * Int32(4)
                h0, h1, h2, h3 = _had128_quad(
                    src[base + Int32(0)].to(Float32), src[base + Int32(1)].to(Float32),
                    src[base + Int32(2)].to(Float32), src[base + Int32(3)].to(Float32), lane)
                dst[base + Int32(0)] = h0
                dst[base + Int32(1)] = h1
                dst[base + Int32(2)] = h2
                dst[base + Int32(3)] = h3

            @cute.jit
            def _launch_h(src: cute.Tensor, dst: cute.Tensor, stream):
                _had_k(src, dst).launch(grid=(nblk, 1, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_HAD_CACHE[nblk] = cute.compile(_launch_h, from_dlpack(blocks.reshape(-1)), from_dlpack(out.reshape(-1)), st)
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_HAD_CACHE[nblk](from_dlpack(blocks.reshape(-1)), from_dlpack(out.reshape(-1)), st)
        return out.reshape(x_f32.shape)

    def run_gemm(a_f16, b_f16, tile_k, gscale_f32):
        m_rows, k_dim = a_f16.shape
        _, n_dim = b_f16.shape
        chains = tile_k // 32
        pad_m = ((max(m_rows, 1) + 7) // 8) * 8
        a_work = torch.zeros((pad_m, k_dim), dtype=torch.float16, device=a_f16.device)
        a_work[:m_rows].copy_(a_f16)
        b_work = b_f16.contiguous()
        c_work = torch.empty((pad_m, n_dim), dtype=torch.float16, device=a_f16.device)
        gs_t = torch.tensor([float(gscale_f32)], dtype=torch.float32, device=a_f16.device)
        key = (k_dim, n_dim, chains, pad_m)
        if key not in _REF_GEMM_CACHE:
            k_iters = k_dim // tile_k

            @cute.kernel
            def _gk(a_t: cute.Tensor, b_t: cute.Tensor, gs_v: cute.Tensor, c_t: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                m_blk, n_tile, _ = cute.arch.block_idx()
                r = Int32(2) * (lane % Int32(4))
                col = lane // Int32(4)
                m_row = m_blk * Int32(8) + col
                n_col0 = n_tile * Int32(16) + col
                n_col1 = n_col0 + Int32(8)
                acc = cute.make_rmem_tensor((chains, 4), Float32)
                for ch in cutlass.range_constexpr(chains):
                    for j in cutlass.range_constexpr(4):
                        acc[ch, j] = Float32(0.0)
                for kt in range(k_iters):
                    for ch in cutlass.range_constexpr(chains):
                        for sub in cutlass.range_constexpr(2):
                            kb = Int32((kt * tile_k) + (ch * 32) + (sub * 16))
                            k0, k1, k2, k3 = kb + r, kb + r + Int32(1), kb + r + Int32(8), kb + r + Int32(9)
                            a0 = _pack_half2(a_t, m_row * Int32(k_dim) + k0, m_row * Int32(k_dim) + k1)
                            a1 = _pack_half2(a_t, m_row * Int32(k_dim) + k2, m_row * Int32(k_dim) + k3)
                            o0 = _pack_half2(b_t, k0 * Int32(n_dim) + n_col0, k1 * Int32(n_dim) + n_col0)
                            o1 = _pack_half2(b_t, k2 * Int32(n_dim) + n_col0, k3 * Int32(n_dim) + n_col0)
                            o2 = _pack_half2(b_t, k0 * Int32(n_dim) + n_col1, k1 * Int32(n_dim) + n_col1)
                            o3 = _pack_half2(b_t, k2 * Int32(n_dim) + n_col1, k3 * Int32(n_dim) + n_col1)
                            d0, d1, d2, d3 = _mma_m16n8k16(
                                acc[ch, 0], acc[ch, 1], acc[ch, 2], acc[ch, 3], o0, o2, o1, o3, a0, a1)
                            acc[ch, 0], acc[ch, 1], acc[ch, 2], acc[ch, 3] = d0, d1, d2, d3
                if cutlass.const_expr(chains == 4):
                    s0 = acc[0, 0] + ((acc[1, 0] + acc[3, 0]) + acc[2, 0])
                    s1 = acc[0, 1] + ((acc[1, 1] + acc[3, 1]) + acc[2, 1])
                    s2 = acc[0, 2] + ((acc[1, 2] + acc[3, 2]) + acc[2, 2])
                    s3 = acc[0, 3] + ((acc[1, 3] + acc[3, 3]) + acc[2, 3])
                else:
                    s0, s1, s2, s3 = acc[0, 0] + acc[1, 0], acc[0, 1] + acc[1, 1], acc[0, 2] + acc[1, 2], acc[0, 3] + acc[1, 3]
                scale = gs_v[Int32(0)].to(Float32)
                out_r0, out_r1 = m_blk * Int32(8) + r, m_blk * Int32(8) + r + Int32(1)
                c0, c1 = n_tile * Int32(16) + col, n_tile * Int32(16) + col + Int32(8)
                c_t[out_r0 * Int32(n_dim) + c0] = (s0 * scale).to(Float16)
                c_t[out_r1 * Int32(n_dim) + c0] = (s1 * scale).to(Float16)
                c_t[out_r0 * Int32(n_dim) + c1] = (s2 * scale).to(Float16)
                c_t[out_r1 * Int32(n_dim) + c1] = (s3 * scale).to(Float16)

            @cute.jit
            def _launch_g(a_t: cute.Tensor, b_t: cute.Tensor, gs_v: cute.Tensor, c_t: cute.Tensor, stream):
                _gk(a_t, b_t, gs_v, c_t).launch(grid=(pad_m // 8, n_dim // 16, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_GEMM_CACHE[key] = cute.compile(
                _launch_g, from_dlpack(a_work.reshape(-1)), from_dlpack(b_work.reshape(-1)),
                from_dlpack(gs_t), from_dlpack(c_work.reshape(-1)), st)
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_GEMM_CACHE[key](
            from_dlpack(a_work.reshape(-1)), from_dlpack(b_work.reshape(-1)),
            from_dlpack(gs_t), from_dlpack(c_work.reshape(-1)), st)
        return c_work[:m_rows]

    activation = meta.get('activation', 'silu')
    swiglu_limit = meta.get('swiglu_limit')

    def run_inter_act(g_f16, u_f16, rot_row, full_inter):
        m_rows, live_inter = g_f16.shape
        nblk = live_inter // 128
        out_f16 = torch.empty((m_rows, live_inter), dtype=torch.float16, device=g_f16.device)
        has_lim = swiglu_limit is not None
        lim_v = float(swiglu_limit or 0.0)
        is_situ = activation == 'situ'
        is_post = activation == 'silu_post_clamp'
        key = (activation, has_lim, lim_v, m_rows, live_inter, full_inter)
        if key not in _REF_ACT_CACHE:
            @cute.kernel
            def _ak(g_t: cute.Tensor, u_t: cute.Tensor, r_t: cute.Tensor, o_t: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                unit, _, _ = cute.arch.block_idx()
                row = unit // Int32(nblk)
                blk = unit % Int32(nblk)
                col0 = blk * Int32(128) + lane * Int32(4)
                base = row * Int32(live_inter) + col0
                gh0, gh1, gh2, gh3 = _had128_quad(
                    g_t[base + Int32(0)].to(Float32), g_t[base + Int32(1)].to(Float32),
                    g_t[base + Int32(2)].to(Float32), g_t[base + Int32(3)].to(Float32), lane)
                uh0, uh1, uh2, uh3 = _had128_quad(
                    u_t[base + Int32(0)].to(Float32), u_t[base + Int32(1)].to(Float32),
                    u_t[base + Int32(2)].to(Float32), u_t[base + Int32(3)].to(Float32), lane)
                gh = cute.make_rmem_tensor((4,), Float32)
                uh = cute.make_rmem_tensor((4,), Float32)
                gh[0], gh[1], gh[2], gh[3] = gh0, gh1, gh2, gh3
                uh[0], uh[1], uh[2], uh[3] = uh0, uh1, uh2, uh3
                out_v = cute.make_rmem_tensor((4,), Float32)
                for j in cutlass.range_constexpr(4):
                    c_idx = col0 + Int32(j)
                    ig = gh[j] * r_t[c_idx].to(Float32)
                    iu = uh[j] * r_t[Int32(full_inter) + c_idx].to(Float32)
                    sd = r_t[Int32(2 * full_inter) + c_idx].to(Float32)
                    if cutlass.const_expr(has_lim):
                        lim = Float32(lim_v)
                        nlim = Float32(-lim_v)
                        if cutlass.const_expr(not is_post):
                            if ig > lim:
                                ig = lim
                        if iu > lim:
                            iu = lim
                        if iu < nlim:
                            iu = nlim
                    sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-ig, fastmath=True))
                    if cutlass.const_expr(is_situ):
                        beta = Float32(4.0)
                        lbeta = Float32(25.0)
                        out_v[j] = (beta * cute.math.tanh(ig / beta, fastmath=True) * sig
                                    * lbeta * cute.math.tanh(iu / lbeta, fastmath=True) * sd)
                    else:
                        silu = ig * sig
                        if cutlass.const_expr(is_post and has_lim):
                            lim = Float32(lim_v)
                            if silu > lim:
                                silu = lim
                        out_v[j] = silu * iu * sd
                rh0, rh1, rh2, rh3 = _had128_quad(out_v[0], out_v[1], out_v[2], out_v[3], lane)
                o_t[base + Int32(0)] = rh0.to(Float16)
                o_t[base + Int32(1)] = rh1.to(Float16)
                o_t[base + Int32(2)] = rh2.to(Float16)
                o_t[base + Int32(3)] = rh3.to(Float16)

            @cute.jit
            def _launch_a(g_t: cute.Tensor, u_t: cute.Tensor, r_t: cute.Tensor, o_t: cute.Tensor, stream):
                _ak(g_t, u_t, r_t, o_t).launch(grid=(m_rows * nblk, 1, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_ACT_CACHE[key] = cute.compile(
                _launch_a, from_dlpack(g_f16.reshape(-1)), from_dlpack(u_f16.reshape(-1)),
                from_dlpack(rot_row.reshape(-1)), from_dlpack(out_f16.reshape(-1)), st)
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_ACT_CACHE[key](
            from_dlpack(g_f16.reshape(-1)), from_dlpack(u_f16.reshape(-1)),
            from_dlpack(rot_row.reshape(-1)), from_dlpack(out_f16.reshape(-1)), st)
        return out_f16

    def run_fc2_rot(fc2_f16, svh_row):
        m_rows, hid = fc2_f16.shape
        nblk = hid // 128
        out_f32 = torch.empty((m_rows, hid), dtype=torch.float32, device=fc2_f16.device)
        key = (m_rows, hid)
        if key not in _REF_FC2_ROT_CACHE:
            @cute.kernel
            def _fk(src: cute.Tensor, svh: cute.Tensor, dst: cute.Tensor):
                lane, _, _ = cute.arch.thread_idx()
                unit, _, _ = cute.arch.block_idx()
                row = unit // Int32(nblk)
                blk = unit % Int32(nblk)
                col0 = blk * Int32(128) + lane * Int32(4)
                base = row * Int32(hid) + col0
                h0, h1, h2, h3 = _had128_quad(
                    src[base + Int32(0)].to(Float32), src[base + Int32(1)].to(Float32),
                    src[base + Int32(2)].to(Float32), src[base + Int32(3)].to(Float32), lane)
                dst[base + Int32(0)] = h0 * svh[col0 + Int32(0)].to(Float32)
                dst[base + Int32(1)] = h1 * svh[col0 + Int32(1)].to(Float32)
                dst[base + Int32(2)] = h2 * svh[col0 + Int32(2)].to(Float32)
                dst[base + Int32(3)] = h3 * svh[col0 + Int32(3)].to(Float32)

            @cute.jit
            def _launch_f(src: cute.Tensor, svh: cute.Tensor, dst: cute.Tensor, stream):
                _fk(src, svh, dst).launch(grid=(m_rows * nblk, 1, 1), block=(32, 1, 1), stream=stream)

            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            _REF_FC2_ROT_CACHE[key] = cute.compile(
                _launch_f, from_dlpack(fc2_f16.reshape(-1)), from_dlpack(svh_row.reshape(-1)),
                from_dlpack(out_f32.reshape(-1)), st)
        st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_FC2_ROT_CACHE[key](
            from_dlpack(fc2_f16.reshape(-1)), from_dlpack(svh_row.reshape(-1)),
            from_dlpack(out_f32.reshape(-1)), st)
        return out_f32

    ptr, scalar = weights
    h, full_inter, e, top_k, bits = (shape[k] for k in ('hidden', 'intermediate', 'experts', 'top_k', 'bits'))
    compact = bool(shape['compact'])
    codebook = meta.get('codebook', 'mcg')
    fc1_tile_k, _, fc2_tile_k, _ = meta['tile']
    descriptor_bits = 9 if e <= 512 else 10
    stride = scalar['weight_num_experts']
    desc = ptr['descriptor_map_ptr'].detach().cpu().reshape(7 if compact else 3, stride)
    expert_map = ptr['expert_map_ptr'].detach().cpu().tolist()

    wire = wire_x[:rows]
    exponent = wire[:, h:].float() - 127.0
    x_bf16 = (wire[:, :h].contiguous().view(torch.float8_e4m3fn).float().reshape(rows, h // 32, 32)
              * torch.exp2(exponent).unsqueeze(-1)).reshape(rows, h).bfloat16()
    ids_slice = topk_ids[:rows]
    ids_cpu = ids_slice.detach().cpu()
    route_outputs = torch.zeros((rows, top_k, h), dtype=torch.float32, device=wire.device)

    for global_expert in sorted({int(v) for v in ids_cpu.reshape(-1).tolist() if int(v) >= 0}):
        combined = expert_map[global_expert]
        live_inter = int(desc[3, combined]) if compact else full_inter
        pos = (ids_cpu == global_expert).nonzero(as_tuple=False)
        token_idx = pos[:, 0].to(device=wire.device, dtype=torch.long)
        slot_idx = pos[:, 1].to(device=wire.device, dtype=torch.long)
        if live_inter <= 0:
            continue
        mats = []
        gscales = []
        for p in range(3):
            code = int(desc[p, combined])
            tier = code >> descriptor_bits
            local = code & ((1 << descriptor_bits) - 1)
            t_bits = bits[tier]
            k_dim, n_dim = (h, live_inter) if p < 2 else (live_inter, h)
            words = k_dim * n_dim * t_bits // 32
            if compact:
                word_offset = int(desc[4 + p, combined])
            elif p == 0:
                word_offset = local * words
            elif p == 1:
                word_offset = (scalar[f'tier{tier}_gate_experts'] + local) * words
            else:
                word_offset = local * words
            backing = ptr[f't{tier}_{"w13" if p < 2 else "w2"}_ptr']
            gs_backing = ptr[f't{tier}_{"w13" if p < 2 else "w2"}_global_ptr']
            mats.append(_decode_trellis_f16(backing[word_offset:word_offset + words], k_dim, n_dim, t_bits, codebook))
            gscales.append(float(gs_backing[local].item()))

        x_e = x_bf16.index_select(0, token_idx).to(torch.float16).to(torch.float32)
        rot_g = run_had128((x_e * ptr['gate_suh_ptr'][combined].to(torch.float32)).to(torch.float16).to(torch.float32)).to(torch.float16)
        rot_u = run_had128((x_e * ptr['up_suh_ptr'][combined].to(torch.float32)).to(torch.float16).to(torch.float32)).to(torch.float16)
        fc1_g = run_gemm(rot_g, mats[0], fc1_tile_k, gscales[0])
        fc1_u = run_gemm(rot_u, mats[1], fc1_tile_k, gscales[1])
        act = run_inter_act(fc1_g, fc1_u, ptr['intermediate_rotations_ptr'][combined], full_inter)
        fc2 = run_gemm(act, mats[2], fc2_tile_k, gscales[2])
        route_outputs[token_idx, slot_idx] = run_fc2_rot(fc2, ptr['svh_ptr'][combined])

    if shape['epilogue'] == 'partial_bf16':
        return route_outputs.to(torch.bfloat16)
    acc = torch.zeros((rows, h), dtype=torch.float32, device=wire.device)
    w_f32 = topk_weights[:rows].to(torch.float32)
    for k_idx in range(top_k):
        acc = torch.addcmul(acc, route_outputs[:, k_idx], w_f32[:, k_idx:k_idx + 1])
    return acc.to(torch.bfloat16) if meta['output_dtype'] == 'bf16' else acc


def _numeric_delta(candidate, baseline):
    cand_f = candidate.float().flatten()
    base_f = baseline.float().flatten()
    diff = candidate.double() - baseline.double()
    denom = float(cand_f.norm() * base_f.norm())
    return dict(
        max_abs=float(diff.abs().max().item()) if diff.numel() else 0.0,
        rms=float(diff.square().mean().sqrt().item()) if diff.numel() else 0.0,
        cosine=float(torch.dot(cand_f, base_f).item() / denom) if denom > 0.0 else 1.0,
    )


def _nvml_device_handle(pynvml, device=None):
    if device is None:
        device = torch.cuda.current_device()
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    return pynvml.nvmlDeviceGetHandleByUUID(uuid)


def _sample_nvml_state():
    info = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = _nvml_device_handle(pynvml)
        for key, fn in (
            ('sm_clock_mhz', lambda: int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))),
            ('mem_clock_mhz', lambda: int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))),
            ('power_w', lambda: round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 2)),
            ('power_limit_w', lambda: round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, 2)),
            ('temp_c', lambda: int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))),
            ('throttle_reasons', lambda: hex(int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)))),
        ):
            try:
                info[key] = fn()
            except Exception:
                info[key] = None
    except Exception:
        pass
    return info


def _measure_streaming_read_bandwidth(l2_bytes, samples=10):
    try:
        from kimi_k6_dense import streaming_read_bandwidth
    except ImportError:
        from benchmarks.kimi_k6_dense import streaming_read_bandwidth
    return streaming_read_bandwidth(int(l2_bytes), int(samples))


_PAIRED_REPLAY_IGNORED_KEYS = frozenset({
    'codebook',
    'sparkinfer_revision',
    'gpu',
    'physical_sms',
    'native_execution_verified',
})


def _executable_recipe_discriminators(meta: dict) -> dict:
    """Normalize executable manifest discriminators, stripping only codebook and provenance."""
    experts = int(meta.get('experts', 0))
    direct = bool(meta.get('direct', False))
    sig = {
        k: v for k, v in meta.items()
        if k not in _PAIRED_REPLAY_IGNORED_KEYS
        and k not in ('objects', 'trellis_lut', 'route_preparation')
    }
    sig.update(
        activation=meta.get('activation', 'silu'),
        swiglu_limit=meta.get('swiglu_limit', None),
        route_block=meta.get('route_block', 8),
        input_format=meta.get('input_format', 'bf16'),
        direct=direct,
        warp_specialized=bool(meta.get('warp_specialized', False)),
        ws_input_stages=meta.get('ws_input_stages', None),
        ws_dynamic_tiles=bool(meta.get('ws_dynamic_tiles', False)),
        fused_input_rotation=bool(meta.get('fused_input_rotation', False)),
        token_major_rotation=bool(meta.get('token_major_rotation', False)),
        row_addresses=bool(meta.get('row_addresses', False)),
        decode_schedule=meta.get('decode_schedule', None),
        activation_output_a8=bool(meta.get('activation_output_a8', False)),
        descriptor_rows=meta.get('descriptor_rows', 3),
        native_info_version=meta.get('native_info_version', None),
        descriptor_bits=meta.get('descriptor_bits', 10 if experts > 512 else 9),
        requires_route_preparation=bool(meta.get('requires_route_preparation', not direct)),
        objects=[
            {
                'label': obj.get('label'),
                'parameters': obj.get('parameters'),
                'pointer_slots': obj.get('pointer_slots'),
                'scalar_slots': obj.get('scalar_slots'),
                'pointer_bindings': obj.get('pointer_bindings'),
            }
            for obj in meta.get('objects', ())
        ],
        trellis_lut=(
            {'bytes': meta['trellis_lut'].get('bytes')}
            if isinstance(meta.get('trellis_lut'), dict)
            else None
        ),
        route_preparation=(
            {'manifest': meta['route_preparation'].get('manifest')}
            if isinstance(meta.get('route_preparation'), dict)
            else None
        ),
    )
    return sig


def _validate_paired_replay_buckets(metas, selected, c_metas, c_selected, *, prefix_buckets=(1, 80)):
    if c_selected != selected:
        raise ValueError('paired EXL3 package selected different capacity buckets')
    checked_caps = sorted(
        set(selected.values()) | (set(prefix_buckets) & (set(metas) | set(c_metas)))
    )
    for cap in checked_caps:
        if cap not in metas or cap not in c_metas:
            raise ValueError(f'paired EXL3 package missing capacity bucket m{cap}')
        sig_a = _executable_recipe_discriminators(metas[cap])
        sig_b = _executable_recipe_discriminators(c_metas[cap])
        if sig_a != sig_b:
            diff_fields = sorted(
                k for k in set(sig_a) | set(sig_b) if sig_a.get(k) != sig_b.get(k)
            )
            raise ValueError(
                f"paired EXL3 recipe mismatch on {', '.join(diff_fields)} at cap {cap}"
            )


def _replay_buckets(shape, root, rows, *, buckets_key='buckets'):
    paths = {int(k): root / v for k, v in shape[buckets_key].items()}
    metas = {k: json.loads((p / 'v41_exl3.json').read_bytes()) for k, p in paths.items()}
    dec = sorted(k for k, m in metas.items() if m['decode_capacity'])
    if not dec or dec[0] != 1 or dec[-1] < 80:
        raise ValueError('missing canonical m1 program')
    targets = {(tuple(m.get('compute', ())), m.get('output_dtype'), m.get('codebook', 'mcg'))
               for m in metas.values()}
    if len(targets) != 1:
        raise ValueError('inconsistent compute/output_dtype/codebook across EXL3 buckets')
    compute, output_dtype, codebook = next(iter(targets))
    if compute not in ((12, 0), (12, 1)) or output_dtype not in ('bf16', 'fp32') or codebook not in ('mcg', 'mul1'):
        raise ValueError(f'unsupported EXL3 manifest target/dtype/codebook: {compute}/{output_dtype}/{codebook}')

    def recipe(m):
        return (m.get('compute'), m.get('codebook', 'mcg'), m['tile'], m.get('route_block', 8),
                m.get('input_format', 'bf16'), m.get('activation', 'silu'),
                m.get('ws_input_stages'), m.get('decode_schedule'), m.get('descriptor_rows', 3),
                *(bool(m.get(k)) for k in ('direct', 'warp_specialized', 'ws_dynamic_tiles',
                                          'fused_input_rotation', 'token_major_rotation',
                                          'row_addresses', 'activation_output_a8')),
                *(m.get(k) for k in ('bits', 'swiglu_limit', 'output_dtype', 'hidden',
                                     'intermediate', 'experts', 'top_k', 'sparkinfer_revision')))
    canonical = [k for k in dec if recipe(metas[k]) == recipe(metas[1])]
    selected = {m: next(k for k in (canonical if m <= dec[-1] else sorted(metas)) if k >= m)
                for m in rows}
    return paths, metas, selected


def _reference_probe_rows(rows):
    probes = {rows[0]}
    for target in (1, 8, 16):
        if target in rows:
            probes.add(target)
    prefill = [m for m in rows if m >= 256]
    if prefill:
        probes.add(prefill[0])
    return sorted(probes)


def _run_shape(args, shape, rows, cls, source, libraries, compare_libraries=None):
    if shape['format'] != 'exl3':
        if getattr(args, 'compare_aot_root', None) is not None:
            raise ValueError('--compare-aot-root requires an EXL3 replay shape')
        from spark_native_replay import native_plan
        plan = native_plan(shape, args.aot_root, libraries)
        return _run_replay(args, shape, rows, source, plan.select(rows),
                           plan.select(tuple(shape.get('prefix_rows', (1, 8, 80)))),
                           plan.make, plan.metadata)
    paths, metas, selected = _replay_buckets(shape, args.aot_root, rows)
    expected_compute = tuple(next(iter(metas.values()))['compute'])
    if torch.cuda.get_device_capability() != expected_compute:
        raise ValueError(f'EXL3 AOT replay requires SM{expected_compute[0]}{expected_compute[1]}')
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

    compare_root = getattr(args, 'compare_aot_root', None)
    compare_spec = None
    if compare_root is not None:
        bkey = 'compare_buckets' if 'compare_buckets' in shape else 'buckets'
        c_paths, c_metas, c_selected = _replay_buckets(shape, compare_root, rows, buckets_key=bkey)
        _validate_paired_replay_buckets(metas, selected, c_metas, c_selected)
        c_loads = compare_libraries.load if compare_libraries is not None else libraries.load

        def make_compare(cap, _seed):
            return cls(torch, c_paths[cap], cap, weights, resolved, io, load_library=c_loads)

        def compare_metadata(cap):
            return dict(capacity=cap, codebook=c_metas[cap].get('codebook', 'mcg'),
                        manifest_sha256=hashlib.sha256((c_paths[cap] / 'v41_exl3.json').read_bytes()).hexdigest(),
                        core_sha256=c_metas[cap]['objects'][0]['object_sha256'], tile=c_metas[cap]['tile'],
                        route_block=c_metas[cap].get('route_block', 8),
                        ws=bool(c_metas[cap].get('warp_specialized')))

        compare_spec = (c_metas, c_selected, make_compare, compare_metadata)

    return _run_replay(args, shape, rows, source, selected, {1: 1, 8: 80, 80: 80}, make, metadata,
                       exl3_state=(weights, metas), compare_spec=compare_spec)


def _run_replay(args, shape, rows, source, selected, prefixes, make, metadata, *,
                exl3_state=None, compare_spec=None):
    ids = routes(shape, max(max(rows), max(prefixes), 80), args.seed)
    h, top_k = shape['hidden'], shape['top_k']
    rng = torch.Generator(device='cuda').manual_seed(args.seed + 1)
    x = (torch.randn((len(ids), h), device='cuda', generator=rng) * 2.).to(torch.float8_e4m3fn).view(torch.uint8)
    x = torch.cat((x, torch.full((len(ids), h // 32), 121, device='cuda', dtype=torch.uint8)), dim=1)
    data = (x, torch.tensor(ids, dtype=torch.int32, device='cuda'),
            torch.randn((len(ids), top_k), device='cuda', generator=rng).softmax(-1))
    # GB10 L2 is 24 MiB (192 MiB = 8x); scale to 8x runtime L2 on larger-L2 GPUs (e.g. SM120).
    l2_bytes = getattr(torch.cuda.get_device_properties('cuda'), 'l2_cache_size', 24 * 1024 * 1024)
    flush_bytes = max(192 * 1024 * 1024, 8 * int(l2_bytes))
    cold = torch.ones(flush_bytes // 4, device='cuda', dtype=torch.float32)
    sink = torch.empty_like(cold)
    variants = {}
    compare_variants = {}
    graphs = {}
    compare_graphs = {}
    cells = {}
    primary_cb = (next(iter(exl3_state[1].values())).get('codebook', 'mcg')
                  if exl3_state is not None else 'mcg')
    verify_own_ref = bool(
        exl3_state is not None and (
            compare_spec is not None
            or primary_cb != 'mcg'
            or getattr(args, 'verify_own_codebook', False)
        )
    )
    probe_set = set(_reference_probe_rows(rows)) if verify_own_ref else set()
    try:
        needed_caps = sorted(set(selected.values()) | set(prefixes.values()))
        for cap in needed_caps:
            v = variants[cap] = make(cap, args.seed)
            v.load(data, min(v.capacity, len(ids)))
            if compare_spec is not None:
                cv = compare_variants[cap] = compare_spec[2](cap, args.seed)
                cv.load(data, min(cv.capacity, len(ids)))
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

        compare_prefix_exact = None
        if compare_spec is not None:
            c_prefix = None
            compare_prefix_exact = {}
            for m, cap in prefixes.items():
                cv = compare_variants[cap]
                cv.poison_output()
                cv.run(m)
                first = cv.output(1).view(torch.uint8)
                if c_prefix is None:
                    c_prefix = first.clone()
                compare_prefix_exact[m] = torch.equal(c_prefix, first)
            if not all(compare_prefix_exact.values()):
                raise ValueError(f'compare canonical m1/m8/m80 prefix mismatch: {compare_prefix_exact}')

        for m, cap in selected.items():
            v = variants[cap]
            v.poison_output()
            v.run(m)
            out = v.output(m).clone()
            if not bool(out.count_nonzero()) or out.dtype != torch.uint8 and not bool(out.isfinite().all()):
                raise ValueError(f'invalid eager output at m{m}')
            own_ref_checked = False
            if m in probe_set:
                weights, metas = exl3_state
                ref_out = _own_codebook_reference(shape, weights, data[0], data[1], data[2], metas[cap], m)
                if not torch.equal(out.view(torch.uint8), ref_out.view(torch.uint8)):
                    raise ValueError(f'own-codebook bitwise reference mismatch at m{m} ({metas[cap].get("codebook", "mcg")})')
                own_ref_checked = True
            graph = graphs[m] = v.capture(m)
            v.poison_output()
            graph.replay()
            if not torch.equal(out.view(torch.uint8), v.output(m).view(torch.uint8)):
                raise ValueError(f'eager/graph bit mismatch at m{m}')
            distinct = sorted({expert for row in ids[:m] for expert in row})
            cell = dict(rows=m, capacity=v.capacity, distinct=len(distinct), samples_us=[],
                        digest=hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                        **metadata(cap, m, distinct))
            for _ in range(args.warmup if not args.profile_replay else 1):
                graph.replay()

            if compare_spec is not None:
                c_metas, _, _, compare_metadata = compare_spec
                cv = compare_variants[cap]
                cv.poison_output()
                cv.run(m)
                c_out = cv.output(m).clone()
                if not bool(c_out.count_nonzero()) or c_out.dtype != torch.uint8 and not bool(c_out.isfinite().all()):
                    raise ValueError(f'invalid compare eager output at m{m}')
                c_own_ref = False
                if m in probe_set:
                    weights, _ = exl3_state
                    c_ref = _own_codebook_reference(shape, weights, data[0], data[1], data[2], c_metas[cap], m)
                    if not torch.equal(c_out.view(torch.uint8), c_ref.view(torch.uint8)):
                        raise ValueError(f'compare own-codebook bitwise reference mismatch at m{m} ({c_metas[cap].get("codebook", "mcg")})')
                    c_own_ref = True
                c_graph = compare_graphs[m] = cv.capture(m)
                cv.poison_output()
                c_graph.replay()
                if not torch.equal(c_out.view(torch.uint8), cv.output(m).view(torch.uint8)):
                    raise ValueError(f'compare eager/graph bit mismatch at m{m}')
                for _ in range(args.warmup if not args.profile_replay else 1):
                    c_graph.replay()
                cell['codebook'] = primary_cb
                cell['own_ref_bitwise'] = own_ref_checked
                cell['compare_samples_us'] = []
                cell['compare'] = dict(
                    **compare_metadata(cap),
                    own_ref_bitwise=c_own_ref,
                    digest=hashlib.sha256(c_out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                )
                cell['numeric_delta'] = _numeric_delta(c_out, out)
            elif primary_cb != 'mcg':
                cell['own_ref_bitwise'] = own_ref_checked
            cells[m] = cell

        order = list(rows)
        order_rng = random.Random(args.seed + 2)
        nvml_samples = []
        if args.profile_replay:
            for m in order:
                sink.copy_(cold)
                torch.cuda.synchronize()
                label = f'spark_floor:{shape["id"]}:{m}'
                torch.cuda.nvtx.range_push(label)
                graphs[m].replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                if compare_spec is not None:
                    sink.copy_(cold)
                    torch.cuda.synchronize()
                    c_label = f'spark_floor_compare:{shape["id"]}:{m}'
                    torch.cuda.nvtx.range_push(c_label)
                    compare_graphs[m].replay()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
        elif not args.correctness_only:
            for r_idx in range(args.rounds):
                order_rng.shuffle(order)
                events = []
                for rep in range(args.replays):
                    for m in order:
                        if m >= 256 and rep >= max(1, args.replays // 4):
                            continue
                        if compare_spec is None:
                            start = torch.cuda.Event(enable_timing=True)
                            stop = torch.cuda.Event(enable_timing=True)
                            sink.copy_(cold)
                            start.record()
                            graphs[m].replay()
                            stop.record()
                            events.append(('primary', m, start, stop))
                        else:
                            arms = (('primary', graphs[m]), ('compare', compare_graphs[m]))
                            if (r_idx + rep) % 2:
                                arms = (arms[1], arms[0])
                            for tag, g in arms:
                                start = torch.cuda.Event(enable_timing=True)
                                stop = torch.cuda.Event(enable_timing=True)
                                sink.copy_(cold)
                                start.record()
                                g.replay()
                                stop.record()
                                events.append((tag, m, start, stop))
                if compare_spec is not None:
                    nvml_samples.append(_sample_nvml_state())
                torch.cuda.synchronize()
                for tag, m, start, stop in events:
                    key = 'samples_us' if tag == 'primary' else 'compare_samples_us'
                    cells[m][key].append(start.elapsed_time(stop) * 1000)

        for cell in cells.values():
            samples = cell.pop('samples_us')
            if samples:
                cell.update(median_us=statistics.median(samples), samples=len(samples),
                            min_us=min(samples), max_us=max(samples))
            if compare_spec is not None:
                c_samples = cell.pop('compare_samples_us')
                if c_samples:
                    cell['compare'].update(
                        median_us=statistics.median(c_samples), samples=len(c_samples),
                        min_us=min(c_samples), max_us=max(c_samples))
                    if samples:
                        cell['compare_over_primary'] = cell['compare']['median_us'] / cell['median_us']

        report = dict(id=shape['id'], source_sha256=source, device=torch.cuda.get_device_name(),
                      prefix_exact=prefix_exact,
                      routes_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      cells=list(cells.values()))
        if exl3_state is not None and (primary_cb != 'mcg' or compare_spec is not None):
            report['codebook'] = primary_cb
        if compare_spec is not None:
            report['compare_codebook'] = next(iter(compare_spec[0].values())).get('codebook', 'mcg')
            report['compare_prefix_exact'] = compare_prefix_exact
            report['l2_flush_bytes'] = flush_bytes
            report['nvml_samples'] = nvml_samples
            if not args.correctness_only and not args.profile_replay:
                report['streaming_read_bandwidth'] = _measure_streaming_read_bandwidth(
                    l2_bytes, max(5, args.rounds * 2))
        return report
    finally:
        torch.cuda.synchronize()
        for graph in (*graphs.values(), *compare_graphs.values()):
            graph.reset()
        for variant in (*variants.values(), *compare_variants.values()):
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
    compare_root = getattr(args, 'compare_aot_root', None)
    compare_libraries = ReplayLibraries(compare_root, args.replay_native_library) if compare_root else None
    if args.preflight_only:
        if shape['format'] == 'exl3':
            plan = _replay_buckets(shape, args.aot_root, rows)[2]
            if compare_root is not None:
                bkey = 'compare_buckets' if 'compare_buckets' in shape else 'buckets'
                c_plan = _replay_buckets(shape, compare_root, rows, buckets_key=bkey)[2]
                if c_plan != plan:
                    raise ValueError('paired EXL3 package selected different capacity buckets')
        else:
            from spark_native_replay import native_plan
            native = native_plan(shape, args.aot_root, libraries)
            native.select(tuple(shape.get('prefix_rows', (1, 8, 80))))
            plan = native.select(rows)
        print(json.dumps(plan))
        return
    if shape['format'] != 'exl3' and torch.cuda.get_device_capability() != (12, 1):
        raise ValueError('Spark AOT replay requires SM121')
    report = _run_shape(args, shape, rows, cls, source, libraries, compare_libraries)
    report['libraries'] = libraries.receipt()
    if compare_libraries is not None:
        report['compare_libraries'] = compare_libraries.receipt()
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
