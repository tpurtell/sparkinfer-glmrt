"""Native E4M3 x E4M3 expert projections, arbitrary F32 K128 scales.

A single 16-row warp-MMA reduction tree serves every capacity. FP8 operands
are never widened or rescaled before MMA; each K128 partial starts at zero.
"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from b12x._lib.intrinsics import div_rn_f32, mma_m16n8k32_f32_e4m3
from ._fp8_moe_kernels import _i32_at, META_HEAD


class QuantizeK128:
    def __init__(self, k, gather=False):
        self.k = int(k)
        self.gather = bool(gather)
        self.blocks = (self.k + 127) // 128

    @cute.jit
    def __call__(self, x: cute.Pointer, q: cute.Pointer, scales: cute.Pointer,
                 rows: Int32, pair_row: cute.Pointer, stream: cuda.CUstream):
        self.kernel(x, q, scales, pair_row).launch(grid=(self.blocks, rows, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, q: cute.Pointer, scales: cute.Pointer, pair_row: cute.Pointer):
        block = Int32(cute.arch.block_idx()[0])
        row = Int32(cute.arch.block_idx()[1])
        lane = Int32(cute.arch.thread_idx()[0])
        source = row
        if cutlass.const_expr(self.gather):
            source = _i32_at(Int64(pair_row.toint()), row)
        values = cute.make_rmem_tensor((4,), Float32)
        amax = Float32(1e-10)
        for j in cutlass.range_constexpr(4):
            col = block * Int32(128) + lane + Int32(32 * j)
            value = Float32(0)
            if (col < Int32(self.k)) & (source >= Int32(0)):
                value = Float32(x[Int64(source) * Int64(self.k) + Int64(col)])
            values[j] = value
            absolute = cute.math.abs(value)
            amax = cutlass.select_(absolute > amax, absolute, amax)
        for delta in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(amax, offset=1 << delta)
            amax = cutlass.select_(other > amax, other, amax)
        scale = amax * Float32(1.0 / 448.0)
        # Official wiring forms an F32 reciprocal then multiplies.
        inv = div_rn_f32(Float32(1.0), scale)
        if lane == Int32(0):
            scales[Int64(row) * Int64(self.blocks) + Int64(block)] = scale
        for j in cutlass.range_constexpr(4):
            col = block * Int32(128) + lane + Int32(32 * j)
            value = values[j] * inv
            value = cutlass.select_(value > Float32(448), Float32(448), value)
            value = cutlass.select_(value < Float32(-448), Float32(-448), value)
            if col < Int32(self.k):
                q[Int64(row) * Int64(self.k) + Int64(col)] = value.to(cutlass.Float8E4M3FN)


class GroupedK128Gemm:
    """16 rows x 32 columns per CTA; four independent N warps."""
    def __init__(self, n, k, experts, halves=1):
        self.n, self.k, self.experts, self.halves = int(n), int(k), int(experts), int(halves)
        if self.n % 128 or self.k % 128:
            raise ValueError("grouped native A8 GEMM requires whole K128/N128 blocks")

    @cute.jit
    def __call__(self, q: cute.Pointer, scales: cute.Pointer, meta: cute.Pointer,
                 w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                 out: cute.Pointer, tiles: Int32, stream: cuda.CUstream):
        self.kernel(q, scales, meta, w1, s1, w3, s3, out).launch(
            grid=(self.n // 32, tiles, self.halves), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Pointer, scales: cute.Pointer, meta: cute.Pointer,
               w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, out: cute.Pointer):
        tile = Int32(cute.arch.block_idx()[1])
        if tile < _i32_at(Int64(meta.toint()), 1):
            half = Int32(cute.arch.block_idx()[2])
            tid = Int32(cute.arch.thread_idx()[0])
            lane = tid % Int32(32)
            nbase = Int32(cute.arch.block_idx()[0]) * Int32(32) + (tid // Int32(32)) * Int32(8)
            e = _i32_at(Int64(meta.toint()), Int32(META_HEAD + 3 * self.experts) + tile)
            row = tile * Int32(16) + lane // Int32(4)
            waddr, saddr = Int64(w1.toint()), Int64(s1.toint())
            if half != Int32(0):
                waddr, saddr = Int64(w3.toint()), Int64(s3.toint())
            weights = cute.make_ptr(Uint32, waddr, cute.AddressSpace.gmem, assumed_align=4)
            ws = cute.make_ptr(Float32, saddr, cute.AddressSpace.gmem, assumed_align=4)
            aq = cute.make_ptr(Uint32, Int64(q.toint()), cute.AddressSpace.gmem, assumed_align=4)
            acc = cute.make_rmem_tensor((4,), Float32)
            for v in cutlass.range_constexpr(4):
                acc[v] = Float32(0)
            for kb in cutlass.range(self.k // 128, unroll=1):
                part = cute.make_rmem_tensor((4,), Float32)
                for v in cutlass.range_constexpr(4):
                    part[v] = Float32(0)
                for ks in cutlass.range_constexpr(4):
                    col = Int32(kb * 128 + ks * 32) + (lane % Int32(4)) * Int32(4)
                    ai = (Int64(row) * Int64(self.k) + Int64(col)) // Int64(4)
                    bi = ((Int64(e) * Int64(self.n) + Int64(nbase + lane // Int32(4))) * Int64(self.k)
                          + Int64(col)) // Int64(4)
                    a0, a1 = aq[ai], aq[ai + Int64(8 * self.k // 4)]
                    a2, a3 = aq[ai + Int64(4)], aq[ai + Int64(8 * self.k // 4 + 4)]
                    b0, b1 = weights[bi], weights[bi + Int64(4)]
                    part[0], part[1], part[2], part[3] = mma_m16n8k32_f32_e4m3(
                        part[0], part[1], part[2], part[3], a0, a1, a2, a3, b0, b1)
                weight_scale = ws[(Int64(e) * Int64(self.n // 128) + Int64(nbase // Int32(128)))
                                  * Int64(self.k // 128) + Int64(kb)]
                for v in cutlass.range_constexpr(4):
                    r = row + Int32(8 * (v // 2))
                    a_scale = scales[Int64(r) * Int64(self.k // 128) + Int64(kb)]
                    acc[v] = acc[v] + (part[v] * a_scale) * weight_scale
            for v in cutlass.range_constexpr(4):
                r = row + Int32(8 * (v // 2))
                n = nbase + (lane % Int32(4)) * Int32(2) + Int32(v % 2)
                out[Int64(r) * Int64(self.n * self.halves) + Int64(half * Int32(self.n) + n)] = acc[v].to(BFloat16)
