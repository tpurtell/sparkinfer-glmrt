"""MXFP4 weights for the exact routed-expert program (``fp8_moe``, ``weights="mxfp4"``).

MiMo V2.6 Pro stores routed experts as OCP MXFP4: ``weight`` U8 ``[N, K/2]``
(two E2M1 codes per byte, the even element in the low nibble) and
``weight_scale`` U8 ``[N, K/32]`` (UE8M0, ``2^(s - 127)``). Every E2M1 value
times a power of two is a BF16 number, so the kernels widen the weights to
``bf16(e2m1 * 2^(s-127))`` exactly (``cvt.rn.bf16x2.e2m1x2`` then one
``mul.bf16x2`` by the BF16 power of two) and run BF16 tensor-core MMAs with
FP32 accumulation, as the FP8 kernels do over ``bf16(w * s)``: results match
``F.linear(x, dequant(w))`` up to FP32 summation order. Nothing is
re-quantized.

Resident scale rows pad ``K/32`` to four bytes; packed weights retain their
exact K32 width. Invalid K32 lanes skip both weight and activation loads.

``GroupedMxfp4Gemv``  decode: ``GroupedFp8Gemv``'s grouped weight-streaming GEMV with each lane
    reading 32 consecutive K values (16 bytes, one scale byte) per 128-wide
    K block.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import bf16_mma_m16n8k16_f32, div_rn_f32, st_global_u32
from b12x.gemm.bf16_gemv._skinny import _ld_cached, _ld_stream

from ._fp8_moe_kernels import META_HEAD, _ceil, _i32_at, _ld_u8, _round_bf16
from ._mxfp4_down_a8_plan import mxfp4_scale_row_bytes

__all__ = ["GroupedMxfp4Gemv", "e2m1x8_scaled_bf16", "ue8m0_bf16x2"]


@dsl_user_op
def ue8m0_bf16x2(exponent, *, loc=None, ip=None):
    """``2^(s - 127)`` as a BF16 pair (``s = 0``: the subnormal 2^-127)."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Uint32(exponent).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 f;
            .reg .pred zero;
            shl.b32 f, $1, 7;
            setp.eq.u32 zero, $1, 0;
            selp.b32 f, 64, f, zero;
            prmt.b32 $0, f, f, 0x1010;
        }
        """,
        "=r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def e2m1x8_scaled_bf16(word, factor, *, loc=None, ip=None):
    """Eight E2M1 codes (byte 0 low nibble first) times a BF16 pair ``factor``,
    as four bf16x2 words (exact)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip), Uint32(factor).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 q0, q1, q2, q3;
            .reg .b32 v0, v1, v2, v3;
            mov.b32 {q0, q1, q2, q3}, $4;
            cvt.rn.bf16x2.e2m1x2 v0, q0;
            cvt.rn.bf16x2.e2m1x2 v1, q1;
            cvt.rn.bf16x2.e2m1x2 v2, q2;
            cvt.rn.bf16x2.e2m1x2 v3, q3;
            mul.rn.bf16x2 $0, v0, $5;
            mul.rn.bf16x2 $1, v1, $5;
            mul.rn.bf16x2 $2, v2, $5;
            mul.rn.bf16x2 $3, v3, $5;
        }
        """,
        "=r,=r,=r,=r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _round_bf16_mem(value: Float32, *, loc=None, ip=None) -> Float32:
    """Round FP32 to BF16 as ``Float32(x.to(BFloat16))`` through global memory does."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 b; cvt.rn.bf16x2.f32 b, $1, $1; shl.b32 $0, b, 16; }",
        "=f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _pack_bf16x2_rn(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two FP32 values into ``bf16x2`` with ``cvt.rn.bf16x2.f32``."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@cute.jit
def _swiglu_epilogue(g_sum: Float32, u_sum: Float32, limit: cutlass.Constexpr) -> Float32:
    gate = _round_bf16_mem(g_sum)
    up = _round_bf16_mem(u_sum)
    if const_expr(limit > 0.0):
        gate = cutlass.select_(gate > Float32(limit), Float32(limit), gate)
        up = cutlass.select_(up > Float32(limit), Float32(limit), up)
        up = cutlass.select_(up < Float32(-limit), Float32(-limit), up)
    silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
    return silu * up


class GroupedMxfp4Gemv:
    """Per active expert ``e`` (grid y), ``out[off_e + r] = a_r @ bf16(w_e)^T``
    over MXFP4 weights for its grouped rows (``max_rows`` per chunk; larger
    groups stream the weights again per chunk). Columns ``[0, split)`` use
    ``w_a``/``s_a`` and ``[split, N)`` ``w_b``/``s_b`` (gate then up), each U8
    ``[E, part, K/2]`` with U8 ``[E, part, K/32]`` scales; ``out`` BF16
    ``[grouped, N]`` (or ``[grouped, split]`` with ``fuse_swiglu=True``)."""

    def __init__(self, *, n: int, k: int, experts: int, split: int | None = None, max_rows: int = 64,
                 warps: int = 4, groups: int = 4, gather: bool = False,
                 fuse_swiglu: bool = False, swiglu_limit: float = 0.0):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.split = self.n if split is None else int(split)
        self.warps, self.groups = int(warps), int(groups)
        self.cols = 8 * self.groups
        self.m_tiles = _ceil(max_rows, 16)
        self.gather = bool(gather)
        self.fuse_swiglu = bool(fuse_swiglu)
        self.swiglu_limit = float(swiglu_limit)
        if self.fuse_swiglu and self.n != 2 * self.split:
            raise ValueError("fused SwiGLU requires N == 2 * split")
        if self.n % self.cols or self.split % self.cols or self.k % 32 or self.k <= 0 or self.warps <= 0:
            raise ValueError("grouped MXFP4 GEMV needs N and the gate/up split % (8*groups) == 0 and "
                             "positive K % 32 == 0 and positive warps")
        self.out_cols = self.split if self.fuse_swiglu else self.n
        self.total_groups = (2 * self.groups) if self.fuse_swiglu else self.groups
        self.k_per_warp = _ceil(_ceil(self.k, 128), self.warps) * 128
        self.blocks = self.k_per_warp // 128
        self.row_bytes = self.k // 2
        self.scale_bytes = mxfp4_scale_row_bytes(self.k)
        self.tile_frags = self.total_groups * 4
        self.frags = self.m_tiles * self.tile_frags

    def key(self) -> tuple:
        return ("mxfp4", 4, self.n, self.k, self.experts, self.split, self.warps, self.groups, self.m_tiles,
                self.gather, self.fuse_swiglu, self.swiglu_limit)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.tile_frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, a: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer,
                 s_a: cute.Pointer, w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer, max_groups: Int32,
                 stream: cuda.CUstream):
        self.kernel(a, pair_row, meta, w_a, s_a, w_b, s_b, out).launch(
            grid=(self.out_cols // self.cols, max_groups, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _load_block(self, dest: cute.Tensor, factors: cute.Tensor, w_row_a: Int64, s_row_a: Int64,
                    w_row_b: Int64, s_row_b: Int64, block):
        for gi in cutlass.range_constexpr(self.total_groups):
            for t in cutlass.range_constexpr(4):
                dest[gi * 4 + t] = Uint32(0)
            factors[gi] = Uint32(127)
            live = Int32(1)
            if const_expr(self.k % (128 * self.warps) != 0):
                tidx = Int32(cute.arch.thread_idx()[0])
                live = (tidx // Int32(32) * Int32(self.k_per_warp) + Int32(block) * Int32(128)
                        + tidx % Int32(4) * Int32(32) < Int32(self.k)).to(Int32)
            if live != Int32(0):
                if const_expr(self.fuse_swiglu and gi >= self.groups):
                    lg = gi - self.groups
                    words = _ld_stream(w_row_b + Int64(lg * 8 * self.row_bytes) + Int64(block) * Int64(64))
                    for t in cutlass.range_constexpr(4):
                        dest[gi * 4 + t] = words[t]
                    factors[gi] = _ld_u8(s_row_b + Int64(lg * 8 * self.scale_bytes) + Int64(block) * Int64(4))
                else:
                    words = _ld_stream(w_row_a + Int64(gi * 8 * self.row_bytes) + Int64(block) * Int64(64))
                    for t in cutlass.range_constexpr(4):
                        dest[gi * 4 + t] = words[t]
                    factors[gi] = _ld_u8(s_row_a + Int64(gi * 8 * self.scale_bytes) + Int64(block) * Int64(4))

    @cute.jit
    def _row_address(self, a: cute.Pointer, pair_row: cute.Pointer, grouped: Int32) -> Int64:
        source = grouped
        if const_expr(self.gather):
            source = _i32_at(Int64(pair_row.toint()), grouped)
        return Int64(a.toint()) + Int64(source) * Int64(self.k * 2)

    @cute.kernel
    def kernel(self, a: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w_a: cute.Pointer,
               s_a: cute.Pointer, w_b: cute.Pointer, s_b: cute.Pointer, out: cute.Pointer):
        group = Int32(cute.arch.block_idx()[1])
        m_at = Int64(meta.toint())
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.tile_frags * 32,)))
        if group < _i32_at(m_at, 0):
            e = _i32_at(m_at + Int64(4 * (META_HEAD + 2 * self.experts)), group)
            n_e = _i32_at(m_at + Int64(4 * META_HEAD), e)
            first = _i32_at(m_at + Int64(4 * (META_HEAD + self.experts)), e)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            g = lane // Int32(4)
            j = lane % Int32(4)
            n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
            k_begin = Int64(warp_id) * Int64(self.k_per_warp)
            if const_expr(self.fuse_swiglu):
                part = Int64(self.split)
                w_base_a = Int64(w_a.toint()) + Int64(e) * part * Int64(self.row_bytes)
                s_base_a = Int64(s_a.toint()) + Int64(e) * part * Int64(self.scale_bytes)
                w_base_b = Int64(w_b.toint()) + Int64(e) * part * Int64(self.row_bytes)
                s_base_b = Int64(s_b.toint()) + Int64(e) * part * Int64(self.scale_bytes)
                w_row_a = w_base_a + (n0 + Int64(g)) * Int64(self.row_bytes) + k_begin // Int64(2) + Int64(16) * Int64(j)
                s_row_a = s_base_a + (n0 + Int64(g)) * Int64(self.scale_bytes) + k_begin // Int64(32) + Int64(j)
                w_row_b = w_base_b + (n0 + Int64(g)) * Int64(self.row_bytes) + k_begin // Int64(2) + Int64(16) * Int64(j)
                s_row_b = s_base_b + (n0 + Int64(g)) * Int64(self.scale_bytes) + k_begin // Int64(32) + Int64(j)
            else:
                w_base = Int64(w_a.toint())
                s_base = Int64(s_a.toint())
                part = Int64(self.split)
                local = n0
                if n0 >= Int64(self.split):
                    w_base = Int64(w_b.toint())
                    s_base = Int64(s_b.toint())
                    part = Int64(self.n - self.split)
                    local = n0 - Int64(self.split)
                w_base = w_base + Int64(e) * part * Int64(self.row_bytes)
                s_base = s_base + Int64(e) * part * Int64(self.scale_bytes)
                w_row_a = w_base + (local + Int64(g)) * Int64(self.row_bytes) + k_begin // Int64(2) + Int64(16) * Int64(j)
                s_row_a = s_base + (local + Int64(g)) * Int64(self.scale_bytes) + k_begin // Int64(32) + Int64(j)
                w_row_b = w_row_a
                s_row_b = s_row_a
            x_off = (k_begin + Int64(32) * Int64(j)) * Int64(2)
            chunks = (n_e + Int32(16 * self.m_tiles - 1)) // Int32(16 * self.m_tiles)
            for c in cutlass.range(chunks, unroll=1):
                row0 = first + c * Int32(16 * self.m_tiles)
                n_c = n_e - c * Int32(16 * self.m_tiles)
                if n_c > Int32(16 * self.m_tiles):
                    n_c = Int32(16 * self.m_tiles)
                live_tiles = (n_c + Int32(15)) // Int32(16)
                per_block = self.total_groups * 4
                cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
                f_cur = cute.make_rmem_tensor(cute.make_layout((self.total_groups,), stride=(1,)), Uint32)
                self._load_block(cur, f_cur, w_row_a, s_row_a, w_row_b, s_row_b, Int32(0))
                if const_expr(self.blocks == 1 and not self.fuse_swiglu and self.m_tiles > 1):
                    src_lo = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int32)
                    src_hi = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int32)
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        r_lo = Int32(16 * mt) + g
                        r_hi = r_lo + Int32(8)
                        s_lo = Int32(-1)
                        s_hi = Int32(-1)
                        if r_lo < n_c:
                            if const_expr(self.gather):
                                s_lo = _i32_at(Int64(pair_row.toint()), row0 + r_lo)
                            else:
                                s_lo = row0 + r_lo
                        if r_hi < n_c:
                            if const_expr(self.gather):
                                s_hi = _i32_at(Int64(pair_row.toint()), row0 + r_hi)
                            else:
                                s_hi = row0 + r_hi
                        src_lo[mt] = s_lo
                        src_hi[mt] = s_hi
                    live_k = Int32(1)
                    if const_expr(self.k % (128 * self.warps) != 0):
                        live_k = (k_begin + Int64(32) * Int64(j) < Int64(self.k)).to(Int32)
                    factors = cute.make_rmem_tensor(cute.make_layout((self.total_groups,), stride=(1,)), Uint32)
                    for gi in cutlass.range_constexpr(self.total_groups):
                        factors[gi] = ue8m0_bf16x2(f_cur[gi])
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        if Int32(mt) < live_tiles:
                            acc_tile = cute.make_rmem_tensor(cute.make_layout((self.tile_frags,), stride=(1,)), Float32)
                            for i in cutlass.range_constexpr(self.tile_frags):
                                acc_tile[i] = Float32(0.0)
                            s_lo = src_lo[mt]
                            s_hi = src_hi[mt]
                            for chunk in cutlass.range_constexpr(2):
                                k_off = Int64(chunk * 16) * Int64(2)
                                xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                                for i in cutlass.range_constexpr(16):
                                    xa[i] = Uint32(0)
                                if (s_lo >= Int32(0)) & (live_k != Int32(0)):
                                    at = Int64(a.toint()) + Int64(s_lo) * Int64(self.k * 2) + x_off + k_off
                                    lo = _ld_cached(at)
                                    hi = _ld_cached(at + Int64(16))
                                    for i in cutlass.range_constexpr(4):
                                        xa[i] = lo[i]
                                        xa[4 + i] = hi[i]
                                if (s_hi >= Int32(0)) & (live_k != Int32(0)):
                                    at = Int64(a.toint()) + Int64(s_hi) * Int64(self.k * 2) + x_off + k_off
                                    lo = _ld_cached(at)
                                    hi = _ld_cached(at + Int64(16))
                                    for i in cutlass.range_constexpr(4):
                                        xa[8 + i] = lo[i]
                                        xa[12 + i] = hi[i]
                                for gi in cutlass.range_constexpr(self.total_groups):
                                    w0, w1, w2, w3 = e2m1x8_scaled_bf16(cur[gi * 4 + 2 * chunk], factors[gi])
                                    w4, w5, w6, w7 = e2m1x8_scaled_bf16(cur[gi * 4 + 2 * chunk + 1], factors[gi])
                                    f = 4 * gi
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        acc_tile[f], acc_tile[f + 1], acc_tile[f + 2], acc_tile[f + 3],
                                        xa[0], xa[8], xa[1], xa[9], w0, w1)
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        d0, d1, d2, d3, xa[2], xa[10], xa[3], xa[11], w2, w3)
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        d0, d1, d2, d3, xa[4], xa[12], xa[5], xa[13], w4, w5)
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        d0, d1, d2, d3, xa[6], xa[14], xa[7], xa[15], w6, w7)
                                    acc_tile[f] = d0
                                    acc_tile[f + 1] = d1
                                    acc_tile[f + 2] = d2
                                    acc_tile[f + 3] = d3
                            for i in cutlass.range_constexpr(self.tile_frags):
                                partial[(warp_id * Int32(self.tile_frags) + Int32(i)) * Int32(32) + lane] = acc_tile[i]
                            cute.arch.sync_threads()
                            for p in cutlass.range_constexpr(2 * self.total_groups):
                                if Int32(p % self.warps) == warp_id:
                                    gi = p // 2
                                    half = p % 2
                                    row = Int32(16 * mt + 8 * half) + g
                                    if row < n_c:
                                        f0 = 4 * gi + 2 * half
                                        t0 = Float32(0.0)
                                        t1 = Float32(0.0)
                                        for src in cutlass.range_constexpr(self.warps):
                                            s_base_idx = Int32(src * self.tile_frags) * Int32(32) + lane
                                            t0 = t0 + partial[s_base_idx + Int32(f0 * 32)]
                                            t1 = t1 + partial[s_base_idx + Int32((f0 + 1) * 32)]
                                        col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                                        address = Int64(out.toint()) + (Int64(row0 + row) * Int64(self.out_cols) + col) * Int64(2)
                                        st_global_u32(address, _pack_bf16x2_rn(t0, t1))
                            cute.arch.sync_threads()
                else:
                    acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                    for i in cutlass.range_constexpr(self.frags):
                        acc[i] = Float32(0.0)
                    nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
                    f_nxt = cute.make_rmem_tensor(cute.make_layout((self.total_groups,), stride=(1,)), Uint32)
                    if const_expr(self.m_tiles == 1):
                        for block in cutlass.range(self.blocks, unroll=1):
                            if block + 1 < self.blocks:
                                self._load_block(nxt, f_nxt, w_row_a, s_row_a, w_row_b, s_row_b, block + 1)
                            wide = cute.make_rmem_tensor(cute.make_layout((16 * self.total_groups,), stride=(1,)), Uint32)
                            for gi in cutlass.range_constexpr(self.total_groups):
                                factor = ue8m0_bf16x2(f_cur[gi])
                                for t in cutlass.range_constexpr(4):
                                    v0, v1, v2, v3 = e2m1x8_scaled_bf16(cur[gi * 4 + t], factor)
                                    wide[16 * gi + 4 * t] = v0
                                    wide[16 * gi + 4 * t + 1] = v1
                                    wide[16 * gi + 4 * t + 2] = v2
                                    wide[16 * gi + 4 * t + 3] = v3
                            for chunk in cutlass.range_constexpr(2):
                                k_off = (Int64(block) * Int64(128) + Int64(chunk * 16)) * Int64(2)
                                live_k = Int32(1)
                                if const_expr(self.k % (128 * self.warps) != 0):
                                    live_k = (k_begin + Int64(block) * Int64(128) + Int64(32) * Int64(j) < Int64(self.k)).to(Int32)
                                for mt in cutlass.range_constexpr(self.m_tiles):
                                    if Int32(mt) < live_tiles:
                                        r_lo = Int32(16 * mt) + g
                                        r_hi = r_lo + Int32(8)
                                        xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                                        for i in cutlass.range_constexpr(16):
                                            xa[i] = Uint32(0)
                                        if (r_lo < n_c) & (live_k != Int32(0)):
                                            at = self._row_address(a, pair_row, row0 + r_lo) + x_off + k_off
                                            lo = _ld_cached(at)
                                            hi = _ld_cached(at + Int64(16))
                                            for i in cutlass.range_constexpr(4):
                                                xa[i] = lo[i]
                                                xa[4 + i] = hi[i]
                                        if (r_hi < n_c) & (live_k != Int32(0)):
                                            at = self._row_address(a, pair_row, row0 + r_hi) + x_off + k_off
                                            lo = _ld_cached(at)
                                            hi = _ld_cached(at + Int64(16))
                                            for i in cutlass.range_constexpr(4):
                                                xa[8 + i] = lo[i]
                                                xa[12 + i] = hi[i]
                                        for gi in cutlass.range_constexpr(self.total_groups):
                                            f = 4 * (mt * self.total_groups + gi)
                                            for t in cutlass.range_constexpr(4):
                                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                                    xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                                    wide[16 * gi + 8 * chunk + 2 * t], wide[16 * gi + 8 * chunk + 2 * t + 1])
                                                acc[f] = d0
                                                acc[f + 1] = d1
                                                acc[f + 2] = d2
                                                acc[f + 3] = d3
                            for i in cutlass.range_constexpr(per_block):
                                cur[i] = nxt[i]
                            for i in cutlass.range_constexpr(self.total_groups):
                                f_cur[i] = f_nxt[i]
                        for i in cutlass.range_constexpr(self.frags):
                            partial[(warp_id * Int32(self.frags) + Int32(i)) * Int32(32) + lane] = acc[i]
                        cute.arch.sync_threads()
                        for i in cutlass.range_constexpr(self.frags):
                            if Int32(i % self.warps) == warp_id:
                                total = Float32(0.0)
                                for src in cutlass.range_constexpr(self.warps):
                                    total = total + partial[(Int32(src * self.frags + i)) * Int32(32) + lane]
                                mt = i // (4 * self.total_groups)
                                gi = (i // 4) % self.total_groups
                                row = Int32(16 * mt) + g + Int32(8 * ((i % 4) // 2))
                                col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                                if row < n_c:
                                    address = Int64(out.toint()) + (Int64(row0 + row) * Int64(self.out_cols) + col) * Int64(2)
                                    target = cute.make_ptr(BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)
                                    target[0] = total.to(BFloat16)
                        cute.arch.sync_threads()
                    else:
                        src_lo = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int32)
                        src_hi = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int32)
                        for mt in cutlass.range_constexpr(self.m_tiles):
                            r_lo = Int32(16 * mt) + g
                            r_hi = r_lo + Int32(8)
                            s_lo = Int32(-1)
                            s_hi = Int32(-1)
                            if r_lo < n_c:
                                if const_expr(self.gather):
                                    s_lo = _i32_at(Int64(pair_row.toint()), row0 + r_lo)
                                else:
                                    s_lo = row0 + r_lo
                            if r_hi < n_c:
                                if const_expr(self.gather):
                                    s_hi = _i32_at(Int64(pair_row.toint()), row0 + r_hi)
                                else:
                                    s_hi = row0 + r_hi
                            src_lo[mt] = s_lo
                            src_hi[mt] = s_hi
                        for block in cutlass.range(self.blocks, unroll=1):
                            if block + 1 < self.blocks:
                                self._load_block(nxt, f_nxt, w_row_a, s_row_a, w_row_b, s_row_b, block + 1)
                            live_k = Int32(1)
                            if const_expr(self.k % (128 * self.warps) != 0):
                                live_k = (k_begin + Int64(block) * Int64(128) + Int64(32) * Int64(j) < Int64(self.k)).to(Int32)
                            factors = cute.make_rmem_tensor(cute.make_layout((self.total_groups,), stride=(1,)), Uint32)
                            for gi in cutlass.range_constexpr(self.total_groups):
                                factors[gi] = ue8m0_bf16x2(f_cur[gi])
                            for chunk in cutlass.range_constexpr(2):
                                k_off = (Int64(block) * Int64(128) + Int64(chunk * 16)) * Int64(2)
                                for mt in cutlass.range_constexpr(self.m_tiles):
                                    if Int32(mt) < live_tiles:
                                        s_lo = src_lo[mt]
                                        s_hi = src_hi[mt]
                                        xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                                        for i in cutlass.range_constexpr(16):
                                            xa[i] = Uint32(0)
                                        if (s_lo >= Int32(0)) & (live_k != Int32(0)):
                                            at = Int64(a.toint()) + Int64(s_lo) * Int64(self.k * 2) + x_off + k_off
                                            lo = _ld_cached(at)
                                            hi = _ld_cached(at + Int64(16))
                                            for i in cutlass.range_constexpr(4):
                                                xa[i] = lo[i]
                                                xa[4 + i] = hi[i]
                                        if (s_hi >= Int32(0)) & (live_k != Int32(0)):
                                            at = Int64(a.toint()) + Int64(s_hi) * Int64(self.k * 2) + x_off + k_off
                                            lo = _ld_cached(at)
                                            hi = _ld_cached(at + Int64(16))
                                            for i in cutlass.range_constexpr(4):
                                                xa[8 + i] = lo[i]
                                                xa[12 + i] = hi[i]
                                        for gi in cutlass.range_constexpr(self.total_groups):
                                            w0, w1, w2, w3 = e2m1x8_scaled_bf16(cur[gi * 4 + 2 * chunk], factors[gi])
                                            w4, w5, w6, w7 = e2m1x8_scaled_bf16(cur[gi * 4 + 2 * chunk + 1], factors[gi])
                                            f = 4 * (mt * self.total_groups + gi)
                                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                                acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                                xa[0], xa[8], xa[1], xa[9], w0, w1)
                                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                                d0, d1, d2, d3, xa[2], xa[10], xa[3], xa[11], w2, w3)
                                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                                d0, d1, d2, d3, xa[4], xa[12], xa[5], xa[13], w4, w5)
                                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                                d0, d1, d2, d3, xa[6], xa[14], xa[7], xa[15], w6, w7)
                                            acc[f] = d0
                                            acc[f + 1] = d1
                                            acc[f + 2] = d2
                                            acc[f + 3] = d3
                            for i in cutlass.range_constexpr(per_block):
                                cur[i] = nxt[i]
                            for i in cutlass.range_constexpr(self.total_groups):
                                f_cur[i] = f_nxt[i]
                        for mt in cutlass.range_constexpr(self.m_tiles):
                            if Int32(mt) < live_tiles:
                                base_f = 4 * mt * self.total_groups
                                for i in cutlass.range_constexpr(self.tile_frags):
                                    partial[(warp_id * Int32(self.tile_frags) + Int32(i)) * Int32(32) + lane] = acc[base_f + i]
                                cute.arch.sync_threads()
                                if const_expr(self.fuse_swiglu):
                                    for p in cutlass.range_constexpr(2 * self.groups):
                                        if Int32(p % self.warps) == warp_id:
                                            gi = p // 2
                                            half = p % 2
                                            row = Int32(16 * mt + 8 * half) + g
                                            if row < n_c:
                                                fg = 4 * gi + 2 * half
                                                fu = 4 * (gi + self.groups) + 2 * half
                                                g0 = Float32(0.0)
                                                g1 = Float32(0.0)
                                                u0 = Float32(0.0)
                                                u1 = Float32(0.0)
                                                for src in cutlass.range_constexpr(self.warps):
                                                    s_base_idx = Int32(src * self.tile_frags) * Int32(32) + lane
                                                    g0 = g0 + partial[s_base_idx + Int32(fg * 32)]
                                                    g1 = g1 + partial[s_base_idx + Int32((fg + 1) * 32)]
                                                    u0 = u0 + partial[s_base_idx + Int32(fu * 32)]
                                                    u1 = u1 + partial[s_base_idx + Int32((fu + 1) * 32)]
                                                v0 = _swiglu_epilogue(g0, u0, self.swiglu_limit)
                                                v1 = _swiglu_epilogue(g1, u1, self.swiglu_limit)
                                                col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                                                address = Int64(out.toint()) + (Int64(row0 + row) * Int64(self.out_cols) + col) * Int64(2)
                                                st_global_u32(address, _pack_bf16x2_rn(v0, v1))
                                else:
                                    for p in cutlass.range_constexpr(2 * self.total_groups):
                                        if Int32(p % self.warps) == warp_id:
                                            gi = p // 2
                                            half = p % 2
                                            row = Int32(16 * mt + 8 * half) + g
                                            if row < n_c:
                                                f0 = 4 * gi + 2 * half
                                                t0 = Float32(0.0)
                                                t1 = Float32(0.0)
                                                for src in cutlass.range_constexpr(self.warps):
                                                    s_base_idx = Int32(src * self.tile_frags) * Int32(32) + lane
                                                    t0 = t0 + partial[s_base_idx + Int32(f0 * 32)]
                                                    t1 = t1 + partial[s_base_idx + Int32((f0 + 1) * 32)]
                                                col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                                                address = Int64(out.toint()) + (Int64(row0 + row) * Int64(self.out_cols) + col) * Int64(2)
                                                st_global_u32(address, _pack_bf16x2_rn(t0, t1))
                                cute.arch.sync_threads()
