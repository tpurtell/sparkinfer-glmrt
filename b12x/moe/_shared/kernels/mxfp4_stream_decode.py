"""Fixed M16 register-streaming decode over the native resident MXFP4 pack."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Int32, Int64, Uint32, Float32
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, e2m1x2_scaled_bf16,
    pack_f32x2_to_bfloat2, bfloat2_to_float2_scaled,
)
from b12x.moe._shared.kernels.w4a8_v41_slice import (
    V41SliceFc1Kernel, fadd_rn_f32,
)


class NativeStreamLinear:
    """Narrow N, register-double-buffered K128, capacity-independent M16 math."""

    def __init__(self, fc1, *, gated):
        self.gated = gated
        self.hidden = fc1.hidden
        self.intermediate = fc1.intermediate
        self.stored = fc1.kernel_intermediate
        self.exact = fc1.exact_storage
        self.topk = fc1.topk
        self.routes = fc1.routes
        self.width = fc1.width
        self.slices = fc1.slices
        self.before = fc1.route_weight_before_fc2
        self.round_fc2 = False
        self.k = self.hidden if gated else self.intermediate
        self.n = 2 * self.intermediate if gated else self.hidden
        blocks = (self.k + 127) // 128
        self.warps = (8 if self.hidden >= 6144 else 4) if gated else (
            4 if blocks % 4 == 0 else max(d for d in range(1, 9) if blocks % d == 0))
        self.blocks = (blocks + self.warps - 1) // self.warps
        self.groups = 1 if gated else 4
        self.cols = 8 * self.groups

    @cute.jit
    def packed_word(self, weights, expert, n, k):
        """Invert the resident lane permutation; no unpacked weight allocation."""
        nk = self.hidden if cutlass.const_expr(self.gated) else self.stored
        rows = 2 * self.stored if cutlass.const_expr(self.gated) else self.hidden
        tile_n = Int64(256)
        plane = Int64(0)
        local_n = Int64(n)
        if cutlass.const_expr(self.gated and self.exact):
            projection = local_n // self.intermediate
            local_n = local_n % self.intermediate
            plane = projection * Int64(self.intermediate * self.hidden // 8)
        elif cutlass.const_expr(self.gated):
            local_n = (local_n // self.intermediate) * self.stored + local_n % self.intermediate
        if cutlass.const_expr(self.exact):
            tile_n = Int64(128)
        live_n = tile_n
        if cutlass.const_expr(self.gated and self.exact and self.intermediate % 128):
            if local_n // 128 == self.intermediate // 128:
                live_n = Int64(self.intermediate % 128)
        offset = (Int64(expert) * Int64(rows * nk // 8) + plane
                  + (local_n // tile_n) * tile_n * Int64(nk // 8)
                  + (Int64(k) // 128) * live_n * 16
                  + (Int64(k) % 128 // 32) * (live_n // 32) * 128
                  + (local_n % tile_n // 32) * 128
                  + (local_n % 8) * 16 + (Int64(k) % 32 // 8) * 4
                  + local_n % 32 // 8)
        return Uint32(weights[offset])

    @cute.jit
    def scale(self, scales, expert, n, k):
        nk = self.hidden if cutlass.const_expr(self.gated) else self.stored
        rows = 2 * self.stored if cutlass.const_expr(self.gated) else self.hidden
        columns = nk // 32
        tile_n = Int64(128 if self.exact else 256)
        local_n, plane = Int64(n), Int64(0)
        if cutlass.const_expr(self.gated and self.exact):
            plane = (local_n // self.intermediate) * Int64(self.intermediate * columns)
            local_n %= self.intermediate
        elif cutlass.const_expr(self.gated):
            local_n = (local_n // self.intermediate) * self.stored + local_n % self.intermediate
        live_n = tile_n
        if cutlass.const_expr(self.gated and self.exact and self.intermediate % 128):
            if local_n // 128 == self.intermediate // 128:
                live_n = Int64(self.intermediate % 128)
        count = Int64(4)
        if cutlass.const_expr(not self.gated and self.exact and self.intermediate % 128):
            if k // 128 == self.intermediate // 128:
                count = Int64(self.intermediate % 128 // 32)
        offset = (Int64(expert) * Int64(rows * columns) + plane
                  + (local_n // tile_n) * tile_n * columns
                  + (Int64(k) // 128) * live_n * 4
                  + local_n % tile_n * count + Int64(k) % 128 // 32)
        data = cute.make_tensor(cute.recast_ptr(scales.iterator, dtype=cutlass.Uint8),
                                cute.make_layout(rows * columns * (expert + 1)))
        return Uint32(data[offset])

    @cute.jit
    def load(self, words, factors, weights, scales, expert, n0, k0, g, j, block):
        for gi in cutlass.range_constexpr(self.groups):
            n = n0 + Int32(8 * gi) + g
            k = k0 + block * 128 + j * 32
            factors[gi] = Uint32(127)
            for t in cutlass.range_constexpr(4):
                words[gi * 4 + t] = Uint32(0)
            if k < self.k:
                factors[gi] = self.scale(scales, expert, n, k)
                for t in cutlass.range_constexpr(4):
                    words[gi * 4 + t] = self.packed_word(weights, expert, n, k + t * 8)

    @cute.jit
    def __call__(self, x, weights, scales, routing, partial, out, metadata, groups, stream):
        self.kernel(x, weights, scales, routing, partial, out, metadata).launch(
            grid=(self.n // self.cols, groups, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, weights: cute.Tensor, scales: cute.Tensor,
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
                wide = cute.make_rmem_tensor((16 * self.groups,), Uint32)
                for gi in cutlass.range_constexpr(self.groups):
                    for t in cutlass.range_constexpr(4):
                        for byte in cutlass.range_constexpr(4):
                            wide[gi * 16 + t * 4 + byte] = e2m1x2_scaled_bf16(
                                cur[gi * 4 + t] >> Uint32(byte * 8), fc[gi])
                for half in cutlass.range_constexpr(2):
                    xa = cute.make_rmem_tensor((16,), Uint32)
                    xa.fill(0)
                    for upper in cutlass.range_constexpr(2):
                        row = g + Int32(8 * upper)
                        k = k0 + block * 128 + j * 32 + Int32(half * 16)
                        if row < rows and k < self.k:
                            if cutlass.const_expr(self.gated):
                                source = Int64(metadata[group, 3 + row]) // self.topk
                                for t in cutlass.range_constexpr(8):
                                    xa[upper * 8 + t] = Uint32(x[source, (k // 2) + t])
                            else:
                                for t in cutlass.range_constexpr(8):
                                    offset = ((Int64(k // self.width) * self.routes + route_base + row)
                                              * self.width // 2 + (k % self.width) // 2 + t)
                                    data = cute.make_tensor(cute.recast_ptr(partial.iterator, dtype=Uint32),
                                                            cute.make_layout(self.slices * self.routes * self.width // 2))
                                    xa[upper * 8 + t] = data[offset]
                    for gi in cutlass.range_constexpr(self.groups):
                        f = gi * 4
                        for t in cutlass.range_constexpr(4):
                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                acc[f], acc[f+1], acc[f+2], acc[f+3],
                                xa[2*t], xa[8+2*t], xa[2*t+1], xa[8+2*t+1],
                                wide[gi*16+half*8+2*t], wide[gi*16+half*8+2*t+1])
                            acc[f], acc[f+1], acc[f+2], acc[f+3] = d0, d1, d2, d3
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
                            projection = col // self.intermediate
                            logical = col % self.intermediate
                            compact = self.slices * self.routes * self.width // 2
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


class V41StreamFc1Kernel(V41SliceFc1Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.a16 or self.scale_codes or self.k_splits != 1 or self.cta_splits != 1:
            raise ValueError("streaming qualification initially requires A16, raw scales and split1")
        self.stream = NativeStreamLinear(self, gated=True)

    @cute.jit
    def __call__(self, x, xs, w13, s13, routing, partial, metadata, groups, stream):
        self.stream(x, w13, s13, routing, partial, partial, metadata, groups, stream)
        self.fold_kernel(routing, partial, metadata).launch(
            grid=(self.slices, groups, 1), block=(128, 1, 1), stream=stream)


class V41StreamFc2Kernel:
    def __init__(self, fc1, *, round_fc2_bf16):
        self.stream = NativeStreamLinear(fc1, gated=False)
        self.stream.round_fc2 = round_fc2_bf16

    @cute.jit
    def __call__(self, w2, s2, partial, out, metadata, inverse, live_rows, groups, stream, routing):
        self.stream(partial, w2, s2, routing, partial, out, metadata, groups, stream)
        self.clear_invalid(out, inverse, live_rows).launch(
            grid=(self.stream.routes, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def clear_invalid(self, out: cute.Tensor, inverse: cute.Tensor, live: cute.Tensor):
        route = cute.arch.block_idx()[0]
        if route < live[0] * self.stream.topk and inverse[route] < 0:
            for col in range(cute.arch.thread_idx()[0], self.stream.hidden, 128):
                out[Int64(route), col] = Float32(0)
