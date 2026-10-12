"""Experimental native swap-AB FP4 GEMV over the existing A4 quantizer output."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from b12x._lib.intrinsics import nvfp4_mma_m16n8k64_f32_e2m1
from ._fp8_moe_kernels import _i32_at, _round_bf16
from ._fp8_moe_stream import _chunk, STREAM_TILE_M
from ._nvfp4_moe_kernels import _ld_f32, nvfp4_alpha_offset
from ._nvfp4_moe_a4 import nvfp4_input_scale_offset, _quantize_row16
from b12x._lib.intrinsics import div_rn_f32
from b12x.gemm.bf16_gemv._skinny import _ld_stream
from ._fp8_moe_kernels import _ld_v2_u32


@cute.jit
def _word(at: Int64):
    return Uint32(cute.make_ptr(Uint32, at, cute.AddressSpace.gmem, assumed_align=4)[0])


class ChunkNvfp4GemvA4:
    """Weights fill MMA M16; up to eight live token rows fill N8.

    The CTA's warps split K and reduce their FP32 sums before applying global
    scales. Operands remain packed FP4 with block-16 E4M3 scales throughout.
    """

    def __init__(self, *, n: int, k: int, experts: int, warps: int = 4, groups: int = 2):
        self.fused = False
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.warps, self.groups = warps, groups
        self.cols = 16 * groups
        if n % self.cols or k % (64 * warps):
            raise ValueError("GEMV requires complete output tiles and K splits")
        self.blocks = k // (128 * warps) if k % (128 * warps) == 0 else k // (64 * warps)
        self.wide = k % (128 * warps) == 0
        self.steps = 2 if self.wide else 1
        self.frags = groups * 4
        self.alpha_offset = nvfp4_alpha_offset(experts, n, k)
        self.in_offset = nvfp4_input_scale_offset(experts, n, k)

    def key(self):
        return ("native_swap_gemv_a4", 4, self.n, self.k, self.experts, self.warps, self.groups)

    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, a: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
                 s: cute.Pointer, out: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(a, a_s, meta, w, s, w, s, s, out, out).launch(
            grid=(self.n // self.cols, max_tiles, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _load(self, dest, scales, wbase, sbase, wup, sup, col, q, c, kb):
        for gi in cutlass.range_constexpr(self.groups):
            wb, sb = wbase, sbase
            offset = Int32(16 * gi)
            if cutlass.const_expr(self.fused):
                offset = Int32(0)
                if cutlass.const_expr(gi == 1):
                    wb, sb = wup, sup
            wr = wb + Int64(col + offset + q) * Int64(self.k // 2) + Int64(kb) * Int64(32 * self.steps)
            if cutlass.const_expr(self.wide):
                # A lane quad reads one contiguous K128 row per M8 half.
                # Keep this natural layout until _fragment shuffles into MMA order.
                for half in cutlass.range_constexpr(2):
                    words = _ld_stream(wr + Int64(half * 8 * (self.k // 2)) + Int64(c) * Int64(16))
                    for word in cutlass.range_constexpr(4):
                        dest[gi * 8 + half * 4 + word] = words[word]
                sr = col + offset + q + Int32(8) * (c & Int32(1))
                lo, hi = _ld_v2_u32(sb + Int64(sr) * Int64(self.k // 16) + Int64(kb) * Int64(8))
                scales[gi * 2], scales[gi * 2 + 1] = lo, hi
            else:
                for half in cutlass.range_constexpr(2):
                    at = wr + Int64(half * 8 * (self.k // 2)) + Int64(c * Int32(4))
                    dest[gi * 4 + half] = _word(at)
                    dest[gi * 4 + half + 2] = _word(at + Int64(16))
                sr = col + offset + q + Int32(8) * (c & Int32(1))
                scales[gi] = _word(sb + Int64(sr) * Int64(self.k // 16) + Int64(kb) * Int64(4))

    @cute.jit
    def _fragment(self, data, gi, half, source, c):
        v = Uint32(0)
        for i in cutlass.range_constexpr(4):
            candidate = Uint32(cute.arch.shuffle_sync(data[gi * 8 + half * 4 + i], source))
            if c == Int32(i):
                v = candidate
        return v

    @cute.kernel
    def kernel(self, a: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
               s: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer,
               out: cute.Pointer, out_s: cute.Pointer):
        tile = Int32(cute.arch.block_idx()[1])
        storage = cutlass_utils.SmemAllocator().allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.frags * 32,)))
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tid = Int32(cute.arch.thread_idx()[0])
            warp = tid // Int32(32)
            lane = tid % Int32(32)
            q, c = lane // Int32(4), lane % Int32(4)
            col = Int32(cute.arch.block_idx()[0]) * Int32(self.cols)
            wb = Int64(w.toint()) + Int64(e) * Int64(self.n * self.k // 2)
            sb = Int64(s.toint()) + Int64(e) * Int64(self.n * self.k // 16)
            wb3 = Int64(w3.toint()) + Int64(e) * Int64(self.n * self.k // 2)
            sb3 = Int64(s3.toint()) + Int64(e) * Int64(self.n * self.k // 16)
            factor3 = _ld_f32(Int64(s3.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                * _ld_f32(Int64(s3.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
            factor = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                * _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
            for row0 in cutlass.range(0, live, 8, unroll=1):
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,)), Float32)
                cur = cute.make_rmem_tensor(cute.make_layout((self.frags * self.steps,)), Uint32)
                nxt = cute.make_rmem_tensor(cute.make_layout((self.frags * self.steps,)), Uint32)
                sc = cute.make_rmem_tensor(cute.make_layout((self.groups * self.steps,)), Uint32)
                sn = cute.make_rmem_tensor(cute.make_layout((self.groups * self.steps,)), Uint32)
                for i in cutlass.range_constexpr(self.frags):
                    acc[i] = Float32(0.0)
                self._load(cur, sc, wb, sb, wb3, sb3, col, q, c, warp * Int32(self.blocks))
                for block in cutlass.range(self.blocks, unroll=1):
                    kb = warp * Int32(self.blocks) + block
                    if block + Int32(1) < Int32(self.blocks):
                        self._load(nxt, sn, wb, sb, wb3, sb3, col, q, c, kb + Int32(1))
                    for step in cutlass.range_constexpr(self.steps):
                        b0, b1, bs = Uint32(0), Uint32(0), Uint32(0)
                        if row0 + q < live:
                            ar = tile * Int32(STREAM_TILE_M) + row0 + q
                            kk = kb * Int32(self.steps) + Int32(step)
                            at = Int64(a.toint()) + Int64(ar) * Int64(self.k // 2) + Int64(kk) * Int64(32) + Int64(c) * Int64(4)
                            b0 = _word(at)
                            b1 = _word(at + Int64(16))
                            bs = _word(Int64(a_s.toint()) + Int64(ar) * Int64(self.k // 16) + Int64(kk) * Int64(4))
                        for gi in cutlass.range_constexpr(self.groups):
                            f = 4 * gi
                            if cutlass.const_expr(self.wide):
                                source = q * Int32(4) + Int32(step * 2)
                                a0 = self._fragment(cur, gi, 0, source, c)
                                a1 = self._fragment(cur, gi, 1, source, c)
                                a2 = self._fragment(cur, gi, 0, source + Int32(1), c)
                                a3 = self._fragment(cur, gi, 1, source + Int32(1), c)
                            else:
                                a0, a1, a2, a3 = cur[f], cur[f + 1], cur[f + 2], cur[f + 3]
                            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                                acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                a0, a1, a2, a3, b0, b1, sc[gi * self.steps + step], bs)
                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3] = d0, d1, d2, d3
                    for i in cutlass.range_constexpr(self.frags * self.steps):
                        cur[i] = nxt[i]
                    for i in cutlass.range_constexpr(self.groups * self.steps):
                        sc[i] = sn[i]
                for i in cutlass.range_constexpr(self.frags):
                    partial[(warp * Int32(self.frags) + Int32(i)) * Int32(32) + lane] = acc[i]
                cute.arch.sync_threads()
                totals = cute.make_rmem_tensor(cute.make_layout((self.frags,)), Float32)
                for i in cutlass.range_constexpr(self.frags):
                    if warp == Int32(i % self.warps):
                        total = Float32(0.0)
                        for src in cutlass.range_constexpr(self.warps):
                            total = total + partial[Int32(src * self.frags * 32 + i * 32) + lane]
                        scale = factor
                        if cutlass.const_expr(self.fused and i >= 4):
                            scale = factor3
                        totals[i] = _round_bf16(total * scale)
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(self.frags):
                    if warp == Int32(i % self.warps):
                        r = row0 + Int32(2) * c + Int32(i % 2)
                        n = col + Int32(16 * (i // 4) + 8 * ((i % 4) // 2)) + q
                        if r < live:
                            if cutlass.const_expr(self.fused):
                                partial[(r - row0) * Int32(32) + (n - col)] = totals[i]
                            else:
                                at = Int64(out.toint()) + (Int64(first + r) * Int64(self.n) + Int64(n)) * Int64(2)
                                cute.make_ptr(BFloat16, at, cute.AddressSpace.gmem, assumed_align=2)[0] = totals[i].to(BFloat16)
                cute.arch.sync_threads()
                if cutlass.const_expr(self.fused):
                    if (tid < Int32(8)) & (row0 + tid < live):
                        gs = Float32(1.0) / _ld_f32(Int64(s2.toint()) + Int64(self.down_in_offset) + Int64(e) * Int64(4))
                        values = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
                        for i in cutlass.range_constexpr(16):
                            gate = partial[tid * Int32(32) + Int32(i)]
                            up = partial[tid * Int32(32) + Int32(16 + i)]
                            if cutlass.const_expr(self.limit > 0.0):
                                gate = cutlass.select_(gate > Float32(self.limit), Float32(self.limit), gate)
                                up = cutlass.select_(up > Float32(self.limit), Float32(self.limit), up)
                                up = cutlass.select_(up < Float32(-self.limit), Float32(-self.limit), up)
                            silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
                            values[i] = _round_bf16(silu * up)
                        ar = tile * Int32(STREAM_TILE_M) + row0 + tid
                        _quantize_row16(values, gs,
                            Int64(out.toint()) + Int64(ar) * Int64(self.n // 2) + Int64(col // Int32(2)),
                            Int64(out_s.toint()) + Int64(ar) * Int64(self.n // 16) + Int64(col // Int32(16)))
                    cute.arch.sync_threads()


class ChunkNvfp4GateUpGemvA4(ChunkNvfp4GemvA4):
    """Native swap-AB gate/up with exact SwiGLU and A4 quantization epilogue."""

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0):
        super().__init__(n=inter, k=hidden, experts=experts, warps=8, groups=2)
        self.fused = True
        # Preserve the measured iteration-3 fused variant for comparison.
        self.wide = False
        self.steps = 1
        self.blocks = hidden // (64 * self.warps)
        self.cols = 16
        self.limit = limit
        self.down_in_offset = nvfp4_input_scale_offset(experts, hidden, inter)

    def key(self):
        return ("fused", self.limit) + super().key()

    @cute.jit
    def __call__(self, a: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
                 s: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer,
                 out: cute.Pointer, out_s: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(a, a_s, meta, w, s, w3, s3, s2, out, out_s).launch(
            grid=(self.n // self.cols, max_tiles, 1), block=(32 * self.warps, 1, 1), stream=stream)
