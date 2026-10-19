"""Streamed FP8 W8A8 MoE kernels with K128/F32 dynamic E4M3 activation quantization.

Every active K128 block on a live row is quantized to E4M3FN with:
  amax = max(|x_i|)
  scale = div.rn.f32(max(amax, 1e-10), 448.0)
  inv_scale = div.rn.f32(1.0, scale)
  q_i = cvt.rn.satfinite.e4m3x2.f32(clamp(x_i * inv_scale, -448.0, 448.0))

GEMMs use mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32. Each K128 block
accumulates 4 K32 MMA steps into a block-local F32 partial, then scales by
(a_scale * w_scale) before accumulating into the cross-K F32 accumulator.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    cvt_f32x4_to_e4m3x4,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    ld_shared_f32,
    ld_shared_u32,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_u32,
    st_shared_u32,
    warp_reduce,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import (
    _ceil,
    _i32_at,
    _ld_v2_u32,
    _post_silu_product,
    _round_bf16,
)
from ._fp8_moe_stream import STREAM_TILE_M, _chunk


@cute.jit
def _quantize_k128_warp(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    lane: Int32,
    q_at: Int64,
    s_at: Int64,
):
    a0 = fmax_f32(fabs_f32(v0), fabs_f32(v1))
    a1 = fmax_f32(fabs_f32(v2), fabs_f32(v3))
    amax = warp_reduce(fmax_f32(a0, a1), fmax_f32)
    amax = fmax_f32(amax, Float32(1e-10))
    scale = div_rn_f32(amax, Float32(448.0))
    inv = div_rn_f32(Float32(1.0), scale)
    s0 = fmin_f32(fmax_f32(v0 * inv, Float32(-448.0)), Float32(448.0))
    s1 = fmin_f32(fmax_f32(v1 * inv, Float32(-448.0)), Float32(448.0))
    s2 = fmin_f32(fmax_f32(v2 * inv, Float32(-448.0)), Float32(448.0))
    s3 = fmin_f32(fmax_f32(v3 * inv, Float32(-448.0)), Float32(448.0))
    packed = cvt_f32x4_to_e4m3x4(s0, s1, s2, s3)
    st_global_u32(q_at + Int64(lane) * Int64(4), packed)
    if lane == Int32(0):
        st_global_f32(s_at, scale)


class QuantRowsA8:
    threads = 128

    def __init__(self, k: int):
        if k % 128 != 0:
            raise ValueError(f"QuantRowsA8 requires K % 128 == 0, got {k}")
        self.k = int(k)
        self.blocks = self.k // 128

    def key(self) -> tuple:
        return ("quant_rows_a8", self.k)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        xq: cute.Pointer,
        xs: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, xq, xs, rows).launch(
            grid=(rows, _ceil(self.blocks, 4), 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        xq: cute.Pointer,
        xs: cute.Pointer,
        rows: Int32,
    ):
        row = Int32(cute.arch.block_idx()[0])
        grp = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid & Int32(31)
        blk = grp * Int32(4) + warp
        if (row < rows) & (blk < Int32(self.blocks)):
            col = blk * Int32(128) + lane * Int32(4)
            p0, p1 = _ld_v2_u32(
                Int64(x.toint()) + (Int64(row) * Int64(self.k) + Int64(col)) * Int64(2)
            )
            v0 = _bf16_lo(p0)
            v1 = _bf16_hi(p0)
            v2 = _bf16_lo(p1)
            v3 = _bf16_hi(p1)
            q_at = Int64(xq.toint()) + Int64(row) * Int64(self.k) + Int64(blk) * Int64(128)
            s_at = (
                Int64(xs.toint())
                + (Int64(row) * Int64(self.blocks) + Int64(blk)) * Int64(4)
            )
            _quantize_k128_warp(v0, v1, v2, v3, lane, q_at, s_at)


class ChunkQuantA8:
    threads = 128

    def __init__(self, inter: int, experts: int):
        if inter % 128 != 0:
            raise ValueError(f"ChunkQuantA8 requires inter % 128 == 0, got {inter}")
        self.inter = int(inter)
        self.blocks = self.inter // 128
        self.experts = int(experts)

    def key(self) -> tuple:
        return ("chunk_quant_a8", self.inter, self.experts)

    @cute.jit
    def __call__(
        self,
        act: cute.Pointer,
        meta: cute.Pointer,
        aq: cute.Pointer,
        a_s: cute.Pointer,
        pairs: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(act, meta, aq, a_s).launch(
            grid=(pairs, _ceil(self.blocks, 4), 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        act: cute.Pointer,
        meta: cute.Pointer,
        aq: cute.Pointer,
        a_s: cute.Pointer,
    ):
        p = Int32(cute.arch.block_idx()[0])
        grp = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid & Int32(31)
        blk = grp * Int32(4) + warp
        tiles = _i32_at(Int64(meta.toint()), 1)
        used = _i32_at(Int64(meta.toint()), 2)
        if (p < used) & (blk < Int32(self.blocks)):
            lo = Int32(0)
            hi = tiles
            while lo + Int32(1) < hi:
                mid = (lo + hi) // Int32(2)
                _, first_m, _ = _chunk(meta, self.experts, mid)
                if first_m <= p:
                    lo = mid
                else:
                    hi = mid
            if lo < tiles:
                _, first, live = _chunk(meta, self.experts, lo)
                r = p - first
                if r < live:
                    i = Int64(lo) * Int64(STREAM_TILE_M) + Int64(r)
                    col = blk * Int32(128) + lane * Int32(4)
                    p0, p1 = _ld_v2_u32(
                        Int64(act.toint())
                        + (Int64(p) * Int64(self.inter) + Int64(col)) * Int64(2)
                    )
                    v0 = _bf16_lo(p0)
                    v1 = _bf16_hi(p0)
                    v2 = _bf16_lo(p1)
                    v3 = _bf16_hi(p1)
                    q_at = (
                        Int64(aq.toint())
                        + i * Int64(self.inter)
                        + Int64(blk) * Int64(128)
                    )
                    s_at = (
                        Int64(a_s.toint())
                        + (i * Int64(self.blocks) + Int64(blk)) * Int64(4)
                    )
                    _quantize_k128_warp(v0, v1, v2, v3, lane, q_at, s_at)


class ChunkSwiGLUA8:
    threads = 128

    def __init__(
        self,
        inter: int,
        experts: int,
        limit: float = 0.0,
        activation: str = "silu_bf16_product",
    ):
        if inter % 128 != 0:
            raise ValueError(f"ChunkSwiGLUA8 requires inter % 128 == 0, got {inter}")
        if activation not in ("silu_bf16_product", "silu_post_clamp_fp32_product"):
            raise ValueError(f"unknown activation: {activation!r}")
        self.inter = int(inter)
        self.blocks = self.inter // 128
        self.experts = int(experts)
        self.limit = float(limit)
        self.post_clamp = activation == "silu_post_clamp_fp32_product"

    def key(self) -> tuple:
        return ("chunk_swiglu_a8", self.inter, self.experts, self.limit, self.post_clamp)

    @cute.jit
    def _act(self, gate: Float32, up: Float32) -> Float32:
        if cutlass.const_expr(self.post_clamp):
            return _round_bf16(_post_silu_product(gate, up, self.limit))
        g = gate
        u = up
        if cutlass.const_expr(self.limit > 0.0):
            lim = Float32(self.limit)
            g = cutlass.select_(g > lim, lim, g)
            u = cutlass.select_(u > lim, lim, u)
            u = cutlass.select_(u < -lim, -lim, u)
        silu = _round_bf16(
            div_rn_f32(g, Float32(1.0) + cute.math.exp(-g, fastmath=False))
        )
        return _round_bf16(silu * u)

    @cute.jit
    def __call__(
        self,
        g: cute.Pointer,
        u: cute.Pointer,
        meta: cute.Pointer,
        aq: cute.Pointer,
        a_s: cute.Pointer,
        pairs: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(g, u, meta, aq, a_s).launch(
            grid=(pairs, _ceil(self.blocks, 4), 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        g: cute.Pointer,
        u: cute.Pointer,
        meta: cute.Pointer,
        aq: cute.Pointer,
        a_s: cute.Pointer,
    ):
        p = Int32(cute.arch.block_idx()[0])
        grp = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid & Int32(31)
        blk = grp * Int32(4) + warp
        tiles = _i32_at(Int64(meta.toint()), 1)
        used = _i32_at(Int64(meta.toint()), 2)
        if (p < used) & (blk < Int32(self.blocks)):
            lo = Int32(0)
            hi = tiles
            while lo + Int32(1) < hi:
                mid = (lo + hi) // Int32(2)
                _, first_m, _ = _chunk(meta, self.experts, mid)
                if first_m <= p:
                    lo = mid
                else:
                    hi = mid
            if lo < tiles:
                _, first, live = _chunk(meta, self.experts, lo)
                r = p - first
                if r < live:
                    i = Int64(lo) * Int64(STREAM_TILE_M) + Int64(r)
                    col = blk * Int32(128) + lane * Int32(4)
                    off = (Int64(p) * Int64(self.inter) + Int64(col)) * Int64(2)
                    gp0, gp1 = _ld_v2_u32(Int64(g.toint()) + off)
                    up0, up1 = _ld_v2_u32(Int64(u.toint()) + off)
                    v0 = self._act(_bf16_lo(gp0), _bf16_lo(up0))
                    v1 = self._act(_bf16_hi(gp0), _bf16_hi(up0))
                    v2 = self._act(_bf16_lo(gp1), _bf16_lo(up1))
                    v3 = self._act(_bf16_hi(gp1), _bf16_hi(up1))
                    q_at = (
                        Int64(aq.toint())
                        + i * Int64(self.inter)
                        + Int64(blk) * Int64(128)
                    )
                    s_at = (
                        Int64(a_s.toint())
                        + (i * Int64(self.blocks) + Int64(blk)) * Int64(4)
                    )
                    _quantize_k128_warp(v0, v1, v2, v3, lane, q_at, s_at)


class StreamFp8LinearA8:
    k_block = 128
    row_bytes = 128
    stride = 128

    def __init__(
        self,
        n: int,
        k: int,
        experts: int,
        tile_m: int = 128,
        tile_n: int = 128,
        live_min: int = 1,
        live_max: int = 128,
        stages: int = 2,
        gather: bool = False,
    ):
        if n % tile_n != 0 or k % 128 != 0:
            raise ValueError(
                f"StreamFp8LinearA8 requires N % {tile_n} == 0 and K % 128 == 0, got {(n, k)}"
            )
        if tile_m not in (16, 32, 64, 128) or tile_n not in (64, 128):
            raise ValueError(f"unsupported A8 tile {(tile_m, tile_n)}")
        if stages not in (2, 3):
            raise ValueError(f"unsupported stage count {stages}")
        self.n = int(n)
        self.k = int(k)
        self.blocks = self.k // 128
        self.experts = int(experts)
        self.tile_m = int(tile_m)
        self.tile_n = int(tile_n)
        self.live_min = int(live_min)
        self.live_max = int(live_max)
        self.stages = int(stages)
        self.gather = bool(gather)

        self.warp_rows = 16 if self.tile_m == 16 else 32
        self.m_warps = self.tile_m // self.warp_rows
        self.n_warps = 2
        self.warps = self.m_warps * self.n_warps
        self.threads = self.warps * 32
        self.warp_cols = self.tile_n // self.n_warps
        self.m_tiles = self.warp_rows // 16
        self.n_tiles = self.warp_cols // 8
        self.frags = self.m_tiles * self.n_tiles * 4

        self.tile_bytes = self.tile_m * self.stride
        self.weight_bytes = self.tile_n * self.stride
        self.sc_bytes = self.tile_m * 4
        self.weight_sc_bytes = 64 if self.tile_m == 16 else 128
        self.stage_bytes = (
            self.tile_bytes + self.weight_bytes + self.sc_bytes + self.weight_sc_bytes
        )
        self.smem_bytes = self.stages * self.stage_bytes

    def key(self) -> tuple:
        return (
            "stream_fp8_linear_a8",
            self.n,
            self.k,
            self.experts,
            self.tile_m,
            self.tile_n,
            self.live_min,
            self.live_max,
            self.stages,
            self.gather,
        )

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "ring": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.smem_bytes], 128
            ],
        }
        return cute.struct(Storage)

    @cute.jit
    def _stage_ptrs(self, smem: Int32, stage: Int32):
        base = smem + stage * Int32(self.stage_bytes)
        sa = base
        sb = sa + Int32(self.tile_bytes)
        ssa = sb + Int32(self.weight_bytes)
        ssb = ssa + Int32(self.sc_bytes)
        return sa, sb, ssa, ssb

    @cute.jit
    def _swizzle(self, row: Int32) -> Int32:
        return ((row & Int32(3)) << Int32(1)) | ((row >> Int32(2)) & Int32(1))

    @cute.jit
    def _load_a(
        self,
        aq: cute.Pointer,
        a_s: cute.Pointer,
        pair_row: cute.Pointer,
        sa: Int32,
        ssa: Int32,
        tid: Int32,
        first: Int32,
        live: Int32,
        tile: Int32,
        k_blk: Int32,
    ):
        for q in cutlass.range_constexpr(_ceil(self.tile_m * 8, self.threads)):
            idx = tid + Int32(q * self.threads)
            if idx < Int32(self.tile_m * 8):
                row = idx >> Int32(3)
                col_vec = idx & Int32(7)
                dst_vec = col_vec ^ self._swizzle(row)
                dst = sa + row * Int32(self.stride) + dst_vec * Int32(16)
                if row < live:
                    src_row = tile * Int32(128) + row
                    if cutlass.const_expr(self.gather):
                        src_row = _i32_at(Int64(pair_row.toint()), first + row)
                    src = (
                        Int64(aq.toint())
                        + Int64(src_row) * Int64(self.k)
                        + Int64(k_blk * 128)
                        + Int64(col_vec * 16)
                    )
                    cp_async4_shared_global(dst, src)

        for q in cutlass.range_constexpr(_ceil(self.tile_m, self.threads)):
            sidx = tid + Int32(q * self.threads)
            if sidx < Int32(self.tile_m):
                if sidx < live:
                    src_row = tile * Int32(128) + sidx
                    if cutlass.const_expr(self.gather):
                        src_row = _i32_at(Int64(pair_row.toint()), first + sidx)
                    s_addr = (
                        Int64(a_s.toint())
                        + (Int64(src_row) * Int64(self.blocks) + Int64(k_blk)) * Int64(4)
                    )
                    cp_async_u32_shared_global(ssa + sidx * Int32(4), s_addr)
                else:
                    st_shared_u32(ssa + sidx * Int32(4), Uint32(0))

    @cute.jit
    def _load_b(
        self,
        wq: cute.Pointer,
        w_s: cute.Pointer,
        sb: Int32,
        ssb: Int32,
        tid: Int32,
        expert: Int32,
        n0: Int32,
        k_blk: Int32,
    ):
        w_base = (
            Int64(wq.toint())
            + (Int64(expert) * Int64(self.n) + Int64(n0)) * Int64(self.k)
            + Int64(k_blk * 128)
        )
        for q in cutlass.range_constexpr(_ceil(self.tile_n * 8, self.threads)):
            idx = tid + Int32(q * self.threads)
            if idx < Int32(self.tile_n * 8):
                row = idx >> Int32(3)
                col_vec = idx & Int32(7)
                dst_vec = col_vec ^ self._swizzle(row)
                dst = sb + row * Int32(self.stride) + dst_vec * Int32(16)
                src = w_base + Int64(row) * Int64(self.k) + Int64(col_vec * 16)
                cp_async4_shared_global(dst, src)
        if tid == Int32(0):
            s_idx = (
                Int64(expert) * Int64(self.n // 128) + Int64(n0 // Int32(128))
            ) * Int64(self.blocks) + Int64(k_blk)
            cp_async_u32_shared_global(ssb, Int64(w_s.toint()) + s_idx * Int64(4))

    @cute.jit
    def _mma_stage(
        self,
        smem: Int32,
        stage: Int32,
        acc: cute.Tensor,
        warp_m: Int32,
        warp_n: Int32,
        q: Int32,
        c: Int32,
        mat: Int32,
        mrow: Int32,
        dual_scale: cutlass.Constexpr = False,
    ):
        sa, sb, ssa, ssb = self._stage_ptrs(smem, stage)
        w_scale = ld_shared_f32(ssb)
        if cutlass.const_expr(dual_scale):
            w_scale = ld_shared_f32(ssb + warp_n * Int32(4))

        part = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
        for fi in cutlass.range_constexpr(self.frags):
            part[fi] = Float32(0.0)

        a = cute.make_rmem_tensor(cute.make_layout((self.m_tiles * 4,), stride=(1,)), Uint32)
        bw = cute.make_rmem_tensor(cute.make_layout((self.n_tiles * 2,), stride=(1,)), Uint32)
        swz = self._swizzle(mrow)

        for ks in cutlass.range_constexpr(4):
            apiece = ((Int32(2 * ks) + (mat >> Int32(1))) ^ swz) * Int32(16)
            for mt in cutlass.range_constexpr(self.m_tiles):
                arow = (
                    warp_m * Int32(self.warp_rows)
                    + Int32(16 * mt)
                    + (mat & Int32(1)) * Int32(8)
                    + mrow
                )
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    sa + arow * Int32(self.stride) + apiece
                )
                a[4 * mt + 0] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3

            bpiece = ((Int32(2 * ks) + (mat & Int32(1))) ^ swz) * Int32(16)
            for pair in cutlass.range_constexpr(self.n_tiles // 2):
                brow = (
                    warp_n * Int32(self.warp_cols)
                    + Int32(16 * pair)
                    + (mat >> Int32(1)) * Int32(8)
                    + mrow
                )
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                    sb + brow * Int32(self.stride) + bpiece
                )
                bw[4 * pair + 0] = b0
                bw[4 * pair + 1] = b1
                bw[4 * pair + 2] = b2
                bw[4 * pair + 3] = b3

            for mt in cutlass.range_constexpr(self.m_tiles):
                for nt in cutlass.range_constexpr(self.n_tiles):
                    f = 4 * (self.n_tiles * mt + nt)
                    d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                        part[f + 0],
                        part[f + 1],
                        part[f + 2],
                        part[f + 3],
                        a[4 * mt + 0],
                        a[4 * mt + 1],
                        a[4 * mt + 2],
                        a[4 * mt + 3],
                        bw[2 * nt + 0],
                        bw[2 * nt + 1],
                    )
                    part[f + 0] = d0
                    part[f + 1] = d1
                    part[f + 2] = d2
                    part[f + 3] = d3

        for mt in cutlass.range_constexpr(self.m_tiles):
            for half in cutlass.range_constexpr(2):
                r = (
                    warp_m * Int32(self.warp_rows)
                    + Int32(16 * mt + 8 * half)
                    + q
                )
                a_scale = ld_shared_f32(ssa + r * Int32(4))
                scale = a_scale * w_scale
                for nt in cutlass.range_constexpr(self.n_tiles):
                    f = 4 * (self.n_tiles * mt + nt) + 2 * half
                    acc[f + 0] = acc[f + 0] + part[f + 0] * scale
                    acc[f + 1] = acc[f + 1] + part[f + 1] * scale

    @cute.jit
    def __call__(
        self,
        aq: cute.Pointer,
        a_s: cute.Pointer,
        pair_row: cute.Pointer,
        meta: cute.Pointer,
        wq: cute.Pointer,
        w_s: cute.Pointer,
        out: cute.Pointer,
        max_tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(aq, a_s, pair_row, meta, wq, w_s, out).launch(
            grid=(self.n // self.tile_n, max_tiles, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        aq: cute.Pointer,
        a_s: cute.Pointer,
        pair_row: cute.Pointer,
        meta: cute.Pointer,
        wq: cute.Pointer,
        w_s: cute.Pointer,
        out: cute.Pointer,
    ):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        smem_alloc = cutlass_utils.SmemAllocator()
        storage = smem_alloc.allocate(self._storage())
        tiles = _i32_at(Int64(meta.toint()), 1)
        if tile < tiles:
            expert, first, count = _chunk(meta, self.experts, tile)
            if (count >= Int32(self.live_min)) & (count <= Int32(self.live_max)):
                live = count
                n0 = n_blk * Int32(self.tile_n)
                warp = tid // Int32(32)
                lane = tid & Int32(31)
                warp_m = warp // Int32(self.n_warps)
                warp_n = warp % Int32(self.n_warps)
                active = warp_m * Int32(self.warp_rows) < live
                q = lane >> Int32(2)
                c = lane & Int32(3)
                mat = lane >> Int32(3)
                mrow = lane & Int32(7)

                smem = shared_ptr_to_u32(storage.ring.data_ptr())
                acc = cute.make_rmem_tensor(
                    cute.make_layout((self.frags,), stride=(1,)), Float32
                )
                for fi in cutlass.range_constexpr(self.frags):
                    acc[fi] = Float32(0.0)

                for pre in cutlass.range_constexpr(self.stages - 1):
                    if cutlass.const_expr(pre < self.blocks):
                        sap, sbp, ssap, ssbp = self._stage_ptrs(smem, Int32(pre))
                        self._load_a(
                            aq, a_s, pair_row, sap, ssap, tid, first, live, tile, Int32(pre)
                        )
                        self._load_b(wq, w_s, sbp, ssbp, tid, expert, n0, Int32(pre))
                    cute.arch.cp_async_commit_group()

                for k_blk in cutlass.range(self.blocks, unroll=1):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.sync_threads()
                    prefetch = k_blk + Int32(self.stages - 1)
                    if prefetch < Int32(self.blocks):
                        next_stage = prefetch % Int32(self.stages)
                        san, sbn, ssan, ssbn = self._stage_ptrs(smem, next_stage)
                        self._load_a(
                            aq, a_s, pair_row, san, ssan, tid, first, live, tile, prefetch
                        )
                        self._load_b(wq, w_s, sbn, ssbn, tid, expert, n0, prefetch)
                    cute.arch.cp_async_commit_group()
                    if active:
                        stage = k_blk % Int32(self.stages)
                        self._mma_stage(
                            smem, stage, acc, warp_m, warp_n, q, c, mat, mrow
                        )
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = (
                                warp_m * Int32(self.warp_rows)
                                + Int32(16 * mt + 8 * half)
                                + q
                            )
                            if row < live:
                                for nt in cutlass.range_constexpr(self.n_tiles):
                                    f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                    col = (
                                        n0
                                        + warp_n * Int32(self.warp_cols)
                                        + Int32(8 * nt)
                                        + c * Int32(2)
                                    )
                                    dst = (
                                        Int64(out.toint())
                                        + (Int64(first + row) * Int64(self.n) + Int64(col))
                                        * Int64(2)
                                    )
                                    st_global_u32(
                                        dst,
                                        pack_f32x2_to_bfloat2(acc[f], acc[f + 1]),
                                    )


class StreamFp8GateUpA8(StreamFp8LinearA8):
    def __init__(
        self,
        inter: int,
        hidden: int,
        experts: int,
        tile_m: int = 128,
        tile_n: int = 128,
        live_min: int = 1,
        live_max: int = 128,
        stages: int = 2,
        limit: float = 0.0,
        activation: str = "silu_bf16_product",
    ):
        if inter % (tile_n // 2) != 0 or inter % 64 != 0:
            raise ValueError(
                f"StreamFp8GateUpA8 requires inter divisible by {tile_n // 2} and 64, got {inter}"
            )
        if activation not in ("silu_bf16_product", "silu_post_clamp_fp32_product"):
            raise ValueError(f"unknown activation: {activation!r}")
        super().__init__(
            n=2 * int(inter),
            k=hidden,
            experts=experts,
            tile_m=tile_m,
            tile_n=tile_n,
            live_min=live_min,
            live_max=live_max,
            stages=stages,
            gather=True,
        )
        self.inter = int(inter)
        self.half_n = self.tile_n // 2
        self.limit = float(limit)
        self.post_clamp = activation == "silu_post_clamp_fp32_product"
        self.x_stride = 2 * self.tile_n + 16
        self.x_bytes = self.tile_m * self.x_stride
        self.smem_bytes = max(self.stages * self.stage_bytes, self.x_bytes)

    def key(self) -> tuple:
        return (
            "stream_fp8_gate_up_a8",
            self.inter,
            self.k,
            self.experts,
            self.tile_m,
            self.tile_n,
            self.live_min,
            self.live_max,
            self.stages,
            self.limit,
            self.post_clamp,
        )

    @cute.jit
    def _act(self, gate: Float32, up: Float32) -> Float32:
        if cutlass.const_expr(self.post_clamp):
            return _round_bf16(_post_silu_product(gate, up, self.limit))
        g = gate
        u = up
        if cutlass.const_expr(self.limit > 0.0):
            lim = Float32(self.limit)
            g = cutlass.select_(g > lim, lim, g)
            u = cutlass.select_(u > lim, lim, u)
            u = cutlass.select_(u < -lim, -lim, u)
        silu = _round_bf16(
            div_rn_f32(g, Float32(1.0) + cute.math.exp(-g, fastmath=False))
        )
        return _round_bf16(silu * u)

    @cute.jit
    def _load_gate_up_b(
        self,
        w1: cute.Pointer,
        s1: cute.Pointer,
        w3: cute.Pointer,
        s3: cute.Pointer,
        sb: Int32,
        ssb: Int32,
        tid: Int32,
        expert: Int32,
        n0: Int32,
        k_blk: Int32,
    ):
        w_off = (Int64(expert) * Int64(self.inter) + Int64(n0)) * Int64(self.k) + Int64(
            k_blk * 128
        )
        w1_base = Int64(w1.toint()) + w_off
        w3_base = Int64(w3.toint()) + w_off
        for q in cutlass.range_constexpr(_ceil(self.tile_n * 8, self.threads)):
            idx = tid + Int32(q * self.threads)
            if idx < Int32(self.tile_n * 8):
                row = idx >> Int32(3)
                col_vec = idx & Int32(7)
                dst_vec = col_vec ^ self._swizzle(row)
                dst = sb + row * Int32(self.stride) + dst_vec * Int32(16)
                src = w1_base + Int64(row) * Int64(self.k) + Int64(col_vec * 16)
                if row >= Int32(self.half_n):
                    src = (
                        w3_base
                        + Int64(row - Int32(self.half_n)) * Int64(self.k)
                        + Int64(col_vec * 16)
                    )
                cp_async4_shared_global(dst, src)
        if tid < Int32(2):
            s_idx = (
                Int64(expert) * Int64(self.inter // 128) + Int64(n0 // Int32(128))
            ) * Int64(self.blocks) + Int64(k_blk)
            s_ptr = cutlass.select_(
                tid == Int32(0), Int64(s1.toint()), Int64(s3.toint())
            )
            cp_async_u32_shared_global(
                ssb + tid * Int32(4), s_ptr + s_idx * Int64(4)
            )

    @cute.jit
    def __call__(
        self,
        xq: cute.Pointer,
        xs: cute.Pointer,
        pair_row: cute.Pointer,
        meta: cute.Pointer,
        w1: cute.Pointer,
        s1: cute.Pointer,
        w3: cute.Pointer,
        s3: cute.Pointer,
        act: cute.Pointer,
        max_tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(xq, xs, pair_row, meta, w1, s1, w3, s3, act).launch(
            grid=(self.inter // self.half_n, max_tiles, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        xq: cute.Pointer,
        xs: cute.Pointer,
        pair_row: cute.Pointer,
        meta: cute.Pointer,
        w1: cute.Pointer,
        s1: cute.Pointer,
        w3: cute.Pointer,
        s3: cute.Pointer,
        act: cute.Pointer,
    ):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        smem_alloc = cutlass_utils.SmemAllocator()
        storage = smem_alloc.allocate(self._storage())
        tiles = _i32_at(Int64(meta.toint()), 1)
        if tile < tiles:
            expert, first, count = _chunk(meta, self.experts, tile)
            if (count >= Int32(self.live_min)) & (count <= Int32(self.live_max)):
                live = count
                n0 = n_blk * Int32(self.half_n)
                warp = tid // Int32(32)
                lane = tid & Int32(31)
                warp_m = warp // Int32(self.n_warps)
                warp_n = warp % Int32(self.n_warps)
                active = warp_m * Int32(self.warp_rows) < live
                q = lane >> Int32(2)
                c = lane & Int32(3)
                mat = lane >> Int32(3)
                mrow = lane & Int32(7)

                smem = shared_ptr_to_u32(storage.ring.data_ptr())
                acc = cute.make_rmem_tensor(
                    cute.make_layout((self.frags,), stride=(1,)), Float32
                )
                for fi in cutlass.range_constexpr(self.frags):
                    acc[fi] = Float32(0.0)

                for pre in cutlass.range_constexpr(self.stages - 1):
                    if cutlass.const_expr(pre < self.blocks):
                        sap, sbp, ssap, ssbp = self._stage_ptrs(smem, Int32(pre))
                        self._load_a(
                            xq, xs, pair_row, sap, ssap, tid, first, live, tile, Int32(pre)
                        )
                        self._load_gate_up_b(
                            w1, s1, w3, s3, sbp, ssbp, tid, expert, n0, Int32(pre)
                        )
                    cute.arch.cp_async_commit_group()

                for k_blk in cutlass.range(self.blocks, unroll=1):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.sync_threads()
                    prefetch = k_blk + Int32(self.stages - 1)
                    if prefetch < Int32(self.blocks):
                        next_stage = prefetch % Int32(self.stages)
                        san, sbn, ssan, ssbn = self._stage_ptrs(smem, next_stage)
                        self._load_a(
                            xq, xs, pair_row, san, ssan, tid, first, live, tile, prefetch
                        )
                        self._load_gate_up_b(
                            w1, s1, w3, s3, sbn, ssbn, tid, expert, n0, prefetch
                        )
                    cute.arch.cp_async_commit_group()
                    if active:
                        stage = k_blk % Int32(self.stages)
                        self._mma_stage(
                            smem,
                            stage,
                            acc,
                            warp_m,
                            warp_n,
                            q,
                            c,
                            mat,
                            mrow,
                            dual_scale=True,
                        )
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                ring = smem
                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = (
                                warp_m * Int32(self.warp_rows)
                                + Int32(16 * mt + 8 * half)
                                + q
                            )
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                col = (
                                    warp_n * Int32(self.warp_cols)
                                    + Int32(8 * nt)
                                    + c * Int32(2)
                                )
                                st_shared_u32(
                                    ring + row * Int32(self.x_stride) + col * Int32(2),
                                    pack_f32x2_to_bfloat2(acc[f], acc[f + 1]),
                                )
                cute.arch.sync_threads()

                pairs_per_row = self.half_n // 2
                for it in cutlass.range_constexpr(
                    _ceil(self.tile_m * pairs_per_row, self.threads)
                ):
                    idx = tid + Int32(it * self.threads)
                    row = idx // Int32(pairs_per_row)
                    col_pair = idx % Int32(pairs_per_row)
                    if row < live:
                        row_base = ring + row * Int32(self.x_stride)
                        gp = ld_shared_u32(row_base + col_pair * Int32(4))
                        up_p = ld_shared_u32(
                            row_base + Int32(self.half_n * 2) + col_pair * Int32(4)
                        )
                        v0 = self._act(_bf16_lo(gp), _bf16_lo(up_p))
                        v1 = self._act(_bf16_hi(gp), _bf16_hi(up_p))
                        dst = (
                            Int64(act.toint())
                            + (
                                Int64(first + row) * Int64(self.inter)
                                + Int64(n0 + col_pair * Int32(2))
                            )
                            * Int64(2)
                        )
                        st_global_u32(dst, pack_f32x2_to_bfloat2(v0, v1))


__all__ = [
    "ChunkQuantA8",
    "ChunkSwiGLUA8",
    "QuantRowsA8",
    "StreamFp8GateUpA8",
    "StreamFp8LinearA8",
]
