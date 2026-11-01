"""Benchmark-only fixed-M16 QMMA decode over checkpoint row-major K32 weights."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Int32, Int64, Uint32, Float32
from b12x._lib.intrinsics import (
    e2m1x8_to_qmma_e2m1x8, mxfp8_mma_m16n8k32_f32_e2m1,
    pack_f32x2_to_bfloat2, bfloat2_to_float2_scaled, get_ptr_as_int64,
)
from b12x.gemm.bf16_gemv._skinny import _ld_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41SliceFc1Kernel, fadd_rn_f32
from b12x.moe._shared.kernels.mxfp4_stream_decode import V41StreamFc2Kernel


class RowMajorA8Linear:
    """Generic register streaming, native stable16 routes and ordered epilogues."""

    def __init__(self, fc1, *, gated):
        self.gated = gated
        self.hidden, self.intermediate = fc1.hidden, fc1.intermediate
        self.width, self.slices = fc1.width, fc1.slices
        self.routes, self.topk = fc1.routes, fc1.topk
        self.before, self.round_fc2 = fc1.route_weight_before_fc2, False
        self.k = self.hidden if gated else self.intermediate
        self.n = 2 * self.intermediate if gated else self.hidden
        blocks = (self.k + 127) // 128
        self.warps = (8 if self.hidden >= 6144 else 4) if gated else (
            4 if blocks % 4 == 0 else max(d for d in range(1, 9) if blocks % d == 0))
        self.blocks = (blocks + self.warps - 1) // self.warps
        self.groups = 1 if gated else 4
        self.cols = 8 * self.groups
        self.scale_columns = (self.k // 32 + 3) // 4 * 4

    @cute.jit
    def load(self, words, factors, weights, scales, expert, n0, k0, g, j, block):
        for gi in cutlass.range_constexpr(self.groups):
            n = n0 + Int32(8 * gi) + g
            k = k0 + block * 128 + j * 32
            factors[gi] = Uint32(127)
            for t in cutlass.range_constexpr(4):
                words[gi * 4 + t] = Uint32(0)
            if k < self.k:
                base = (Int64(expert) * self.n + Int64(n)) * Int64(self.k // 8) + Int64(k // 8)
                values = _ld_stream(get_ptr_as_int64(weights, base))
                for t in cutlass.range_constexpr(4):
                    words[gi * 4 + t] = values[t]
                factors[gi] = Uint32(scales[(Int64(expert) * self.n + Int64(n))
                                           * self.scale_columns + Int64(k // 32)])

    @cute.jit
    def __call__(self, x, xs, weights, scales, routing, partial, out, metadata, groups, stream):
        self.kernel(x, xs, weights, scales, routing, partial, out, metadata).launch(
            grid=(self.n // self.cols, groups, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, xs: cute.Tensor, weights: cute.Tensor, scales: cute.Tensor,
               routing: cute.Tensor, partial: cute.Tensor, out: cute.Tensor, metadata: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        g, j = lane // 4, lane % 4
        group = cute.arch.block_idx()[1]
        expert, rows = Int64(metadata[group, 0]), Int32(metadata[group, 1])
        route_base = Int64(metadata[group, 2])
        n0 = cute.arch.block_idx()[0] * self.cols
        smem = cutlass.utils.SmemAllocator()
        fold = smem.allocate_tensor(Float32, cute.make_layout((self.warps, self.groups * 4, 32)),
                                    byte_alignment=16)
        data = cute.make_tensor(cute.recast_ptr(partial.iterator, dtype=Uint32),
                                cute.make_layout(self.slices * self.routes
                                                 * (self.width // 4 + self.width // 32)))
        if rows > 0:
            acc = cute.make_rmem_tensor((self.groups * 4,), Float32)
            acc.fill(0)
            cur = cute.make_rmem_tensor((self.groups * 4,), Uint32)
            nxt = cute.make_rmem_tensor((self.groups * 4,), Uint32)
            fc = cute.make_rmem_tensor((self.groups,), Uint32)
            fn = cute.make_rmem_tensor((self.groups,), Uint32)
            k0 = warp * Int32(self.blocks * 128)
            self.load(cur, fc, weights, scales, expert, n0, k0, g, j, Int32(0))
            for block in range(self.blocks):
                if block + 1 < self.blocks:
                    self.load(nxt, fn, weights, scales, expert, n0, k0, g, j, block + 1)
                for ks in cutlass.range_constexpr(4):
                    k = k0 + block * 128 + Int32(ks * 32)
                    a = cute.make_rmem_tensor((4,), Uint32)
                    a.fill(0)
                    asc = Uint32(0)
                    if k < self.k:
                        for upper in cutlass.range_constexpr(2):
                            row = g + Int32(8 * upper)
                            if row < rows:
                                if cutlass.const_expr(self.gated):
                                    source = Int64(metadata[group, 3 + row]) // self.topk
                                    a[upper] = x[source, k // 4 + j * 2]
                                    a[upper + 2] = x[source, k // 4 + j * 2 + 1]
                                else:
                                    base = ((Int64(k // self.width) * self.routes + route_base + row)
                                            * Int64(self.width // 4) + Int64(k % self.width // 4 + j * 2))
                                    a[upper] = data[base]
                                    a[upper + 2] = data[base + 1]
                        srow = g + j * 8
                        if j < 2 and srow < rows:
                            if cutlass.const_expr(self.gated):
                                source = Int64(metadata[group, 3 + srow]) // self.topk
                                asc = Uint32(xs[source, k // 32])
                            else:
                                offset = (self.slices * self.routes * (self.width // 4)
                                          + (Int64(k // self.width) * self.routes + route_base + srow)
                                          * (self.width // 32) + k % self.width // 32)
                                asc = Uint32(data[offset])
                    for gi in cutlass.range_constexpr(self.groups):
                        words = cute.make_rmem_tensor((4,), Uint32)
                        for t in cutlass.range_constexpr(4):
                            words[t] = cute.arch.shuffle_sync(cur[gi * 4 + t], g * 4 + Int32(ks))
                        # Direct A fragments use consecutive K8 per lane, unlike ldmatrix's permutation.
                        packed = Uint32(0)
                        for t in cutlass.range_constexpr(4):
                            if j == Int32(t):
                                packed = Uint32(words[t])
                        b0, b1 = e2m1x8_to_qmma_e2m1x8(packed)
                        bsc = Uint32(cute.arch.shuffle_sync(fc[gi], g * 4 + Int32(ks)))
                        f = gi * 4
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                            a[0], a[1], a[2], a[3], b0, b1, asc, bsc)
                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = d0, d1, d2, d3
                for t in cutlass.range_constexpr(self.groups * 4):
                    cur[t] = nxt[t]
                for t in cutlass.range_constexpr(self.groups):
                    fc[t] = fn[t]
            for f in cutlass.range_constexpr(self.groups * 4):
                fold[warp, f, lane] = acc[f]
            cute.arch.sync_threads()
            for f in cutlass.range_constexpr(self.groups * 4):
                if warp == Int32(f % self.warps):
                    value = Float32(0)
                    for src in cutlass.range_constexpr(self.warps):
                        value = fadd_rn_f32(value, fold[src, f, lane])
                    row = g + Int32(8 * ((f % 4) // 2))
                    col = n0 + Int32(8 * (f // 4)) + 2 * j + Int32(f % 2)
                    if row < rows:
                        if cutlass.const_expr(self.gated):
                            projection, logical = col // self.intermediate, col % self.intermediate
                            compact = self.slices * self.routes * (self.width // 4 + self.width // 32)
                            raw = cute.make_tensor(cute.recast_ptr(partial.iterator, dtype=Float32) + compact,
                                                    cute.make_layout(self.slices * self.routes * 2 * self.width))
                            offset = (((Int64(logical // self.width) * self.routes + route_base + row) * 2
                                       + 1 - projection) * self.width + logical % self.width)
                            raw[offset] = value
                        else:
                            if cutlass.const_expr(self.round_fc2):
                                value, _ = bfloat2_to_float2_scaled(pack_f32x2_to_bfloat2(value, value), Float32(1))
                            if cutlass.const_expr(not self.before):
                                value *= routing[route_base + row]
                            out[Int64(metadata[group, 3 + row]), col] = value


class RowMajorA8Fc1(V41SliceFc1Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.a16 or self.scale_codes or self.k_splits != 1 or self.cta_splits != 1:
            raise ValueError("row-major A8 benchmark requires raw scales and split1")
        self.stream = RowMajorA8Linear(self, gated=True)

    @cute.jit
    def __call__(self, x, xs, w13, s13, routing, partial, metadata, groups, stream):
        self.stream(x, xs, w13, s13, routing, partial, partial, metadata, groups, stream)
        self.fold_kernel(routing, partial, metadata).launch(
            grid=(self.slices, groups, 1), block=(128, 1, 1), stream=stream)


class RowMajorA8Fc2(V41StreamFc2Kernel):
    def __init__(self, fc1, *, round_fc2_bf16):
        self.stream = RowMajorA8Linear(fc1, gated=False)
        self.stream.round_fc2 = round_fc2_bf16

    @cute.jit
    def __call__(self, w2, s2, partial, out, metadata, inverse, live_rows, groups, stream, routing):
        self.stream(partial, partial, w2, s2, routing, partial, out, metadata, groups, stream)
        self.clear_invalid(out, inverse, live_rows).launch(
            grid=(self.stream.routes, 1, 1), block=(128, 1, 1), stream=stream)
