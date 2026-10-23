"""Decode-once shared E4M3 trellis tiles for many-row dense prefill (track T1).

The K4 payload is decoded once per CTA into **exact E4M3** bytes in shared
memory, laid out in the consumer's MMA B-fragment order, and every warp reads
the whole tile for its own m16 rows. Two consumers are selectable:

``mode="f16"``
    widen E4M3 -> F16 in registers (``cvt.rn.f16x2.e4m3x2`` is exact for every
    E4M3 value) and use ``mma.m16n8k16.f32.f16.f16.f32``; activations stay
    exactly what the W8A16 / native path feeds.
``mode="fp8"``
    feed ``mma.m16n8k32.f32.e4m3.e4m3.f32`` with the same E4M3 weights and
    E4M3 activations quantized per row and per K128 block, applying the
    checkpoint's per-K128 weight scale and the row's activation scale to the
    FP32 accumulator at each K128 boundary, exactly where W8A16 applies it.

Both modes keep one FP32 accumulator per K128 block plus the running FP32
total, reuse the decoded tile across the CTA's whole M range, and write the
F32 output directly (no split-K partials, no reduce pass). Only K4 is handled.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from b12x._lib.intrinsics import (
    broadcast_f32_to_half2,
    byte_perm,
    f16_mma_m16n8k16_f32,
    fp8x4_e4m3_to_half2x2,
    habs2,
    half2_mul,
    hmax2,
    hmax_to_f32,
    ld_global_nc_u32,
    ld_global_nc_u8,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ld_shared_u32,
    ld_shared_v2_u32,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k32_f32_e4m3,
    mxfp8_mma_m16n8k32_f32_e4m3,
    packed_dequant_trellis_to_half2x4,
    pow2_ceil_ue8m0,
    shared_ptr_to_u32,
    st_global_u8,
    st_global_v2_f32,
    st_global_v4_u32,
    st_shared_u16,
    st_shared_u32,
    ue8m0_to_output_scale,
)
from b12x.moe._shared.kernels.w4a16.exlr8 import _side_value

M_CTA = 128
NWARP = 8
K16_PER_BLOCK = 8
E4M3_MAX = 448.0


def _e4m3x4_from_half2x2(lo: Uint32, hi: Uint32) -> Uint32:
    """Round two half2 fragment registers to E4M3 and pack four bytes.

    ``lo`` supplies bytes 0-1 and ``hi`` bytes 2-3, exactly the order
    ``fp8x4_e4m3_to_half2x2`` reads back. Rounding is what the native kernel
    already does (``_e4m3_round_half2``); a decoded trellis value is an exact
    E4M3 number, so this is a no-op on it and bitwise for the F16 consumer.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(lo).ir_value(), Uint32(hi).ir_value()],
            """
            {
                .reg .b16 e0, e1;
                cvt.rn.satfinite.e4m3x2.f16x2 e0, $1;
                cvt.rn.satfinite.e4m3x2.f16x2 e1, $2;
                mov.b32 $0, {e0, e1};
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _load_words4(address: Int64, tile: Int32, lane: Int32):
    """One K4 payload tile (128 B): the lane word and the rotated neighbour."""
    base = address + tile.to(Int64) * Int64(128)
    b = ld_global_nc_u32(base + lane.to(Int64) * Int64(4))
    a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
    return b, (a << Uint32(16)) | (b >> Uint32(16))


class TrellisPrefillDense:
    """Many-row native trellis dense linear, decode-once shared E4M3 tile."""

    ABI_VERSION = 1

    def __init__(self, g, *, mode: str = "f16", n_cta: int = 64, stages: int = 2,
                 mpw: int = 1):
        self.k, self.n = g["input_features"], g["output_features"]
        self.bits = g["bits"]
        if self.bits != 4:
            raise ValueError("the T1 prefill kernel covers K4 trellis only")
        # The E4M3 shared tile is the native kernel's own weight arithmetic:
        # ``decode_tile`` rounds the e4m3 codebook through
        # ``cvt.rn.satfinite.e4m3x2.f16x2``, which is exactly what packing the
        # bytes does here. Any other codebook keeps wider decoded levels and
        # would need an FP16 tile instead.
        if g["codebook"] != "e4m3":
            raise ValueError("the T1 E4M3 tile covers the e4m3 codebook only")
        if mode not in ("f16", "fp8"):
            raise ValueError("mode must be f16 or fp8")
        if n_cta not in (64, 128) or n_cta > self.n:
            raise ValueError("n_cta must be 64 or 128")
        if mpw not in (1, 2) or n_cta % (8 * mpw):
            raise ValueError("mpw must be 1 or 2 and divide the CTA width")
        self.e4m3 = g["codebook"] == "e4m3"
        self.mul1 = g["codebook"] == "mul1"
        self.input_units = g["sharded_side"] == 2
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]
        self.mode, self.n_cta, self.stages = mode, n_cta, stages
        # mpw = m16 fragments per warp. One warp covers mpw row groups over
        # n_cta/mpw columns, so the B-fragment widen and the shared load are
        # reused across mpw MMAs and the running accumulators halve.
        self.mpw = mpw
        self.n16 = n_cta // 16
        self.n8w = n_cta // (8 * mpw)

    def _storage(self):
        words = self.stages * 32 * self.n_cta

        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[Uint32, words], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def _tile_index(self, nb: Int32, kb: Int32, k16: Int32, n16_local: Int32) -> Int32:
        n16g = nb * Int32(self.n_cta // 16) + n16_local
        group = n16g // Int32(8)
        slot = n16g % Int32(8)
        if cutlass.const_expr(self.input_units):
            return (kb * Int32(self.n // 16 * 8) + group * Int32(64)
                    + k16 * Int32(8) + slot)
        return (group * Int32(self.k // 16 * 8)
                + (kb * Int32(8) + k16) * Int32(8) + slot)

    # ------------------------------------------------------------------ A side
    @cute.jit
    def _pair(self, xrow: Int64, kcol: Int32) -> Uint32:
        """Two F16 activations: one F16-MMA A fragment register.

        ``xrow`` is the row's own base address (``x + row*K*2``), hoisted out of
        the K loop, so each load is one address add plus the load itself.
        """
        return ld_global_nc_u32(xrow + kcol.to(Int64) * Int64(2))

    @cute.jit
    def _quad(self, xrow: Int64, kcol: Int32):
        """Four F16 activations: one E4M3-MMA A fragment register before quantizing."""
        return ld_global_nc_v2_u32(xrow + kcol.to(Int64) * Int64(2))

    # -------------------------------------------------------------- weight side
    @cute.jit
    def _stage(self, sbase: Int32, stage: Int32, k16: Int32, n16: Int32,
               lane: Int32, wa: Uint32, wb: Uint32, b_sel: Int32,
               src1: Int32, src2: Int32):
        """Decode one K16 x N16 tile and store exact E4M3 in fragment order.

        Stage layout, per k16 (F16 consumer) or k32 pair (FP8 consumer) and per
        n8: 32 lane words. The F16 consumer reads its (b0, b1) pair as one u32;
        the FP8 consumer reads k=4q..4q+3 (and +16) as a 64-bit pair.
        """
        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, self.bits)
        for s in cutlass.range_constexpr(2):
            if cutlass.const_expr(s == 0):
                e = _e4m3x4_from_half2x2(b0, b1)
            else:
                e = _e4m3x4_from_half2x2(b2, b3)
            n8 = n16 * Int32(2) + Int32(s)
            if cutlass.const_expr(self.mode == "f16"):
                # [k16][n8][lane]: the consumer's (b0, b1) pair is one u32.
                st_shared_u32(
                    sbase + (stage * Int32(32 * self.n_cta) + k16 * Int32(4 * self.n_cta)
                             + n8 * Int32(32) + lane) * Int32(4), e)
            else:
                # [k32][n8][lane][k16-of-pair]: the consumer's 64-bit k32
                # fragment; the producer packs k=4q..4q+3 from two lanes.
                word = byte_perm(cute.arch.shuffle_sync(e, src1),
                                 cute.arch.shuffle_sync(e, src2), b_sel)
                st_shared_u32(
                    sbase + (stage * Int32(32 * self.n_cta)
                             + (k16 // Int32(2)) * Int32(8 * self.n_cta)
                             + n8 * Int32(64) + lane * Int32(2)
                             + k16 % Int32(2)) * Int32(4), word)

    # ------------------------------------------------------------------- kernel
    @cute.kernel
    def prefill(self, inputs: cute.Tensor, payload: cute.Tensor,
                scales: cute.Tensor, output: cute.Tensor,
                rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        x = inputs[rotation_id]
        wbase = payload.iterator.toint()
        obase = output.iterator.toint()
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sbase = shared_ptr_to_u32(storage.w.data_ptr())
        n_tiles = Int32(self.n // self.n_cta)
        nkb = self.k // 128
        total_tasks = ((rows + Int32(M_CTA - 1)) // Int32(M_CTA)) * n_tiles
        # Rows past the tail re-read the last valid row; the store is predicated.
        # (Exact in f32 for every row count below 2**24.)
        row_last = Float32(rows - Int32(1))
        # FP8 consumer B-fragment gather: lane q reads the khalf word pair of
        # producer lanes 2*(q%2) and 2*(q%2)+1 and selects the low (q<2) or the
        # high (q>=2) bytes of each.
        b_sel = Int32(0x5410) + ((lane >> Int32(1)) & Int32(1)) * Int32(0x2222)
        src1 = (lane & Int32(28)) + (lane & Int32(1)) * Int32(2)
        src2 = src1 + Int32(1)
        while task < total_tasks:
            nb = task % n_tiles
            m_base = (task // n_tiles) * Int32(M_CTA)
            # This warp's row blocks: mpw m16 fragments. mpw = 1 is one row
            # group across the whole CTA width; mpw = 2 is two row groups over
            # half the width, so one shared load and one widen feed two MMAs.
            row_grp = warp // Int32(self.mpw)
            col_grp = warp % Int32(self.mpw)
            row = []
            arow = []
            for b in cutlass.range_constexpr(self.mpw):
                r = (m_base + (row_grp * Int32(self.mpw) + Int32(b)) * Int32(16)
                     + (lane >> Int32(2)))
                # Rows past the tail re-read the last valid row; the store is
                # predicated. (Exact in f32 for every row count below 2**24.)
                row.append(r)
                c = Int32(cute.arch.fmin(Float32(r), row_last))
                # Per-row A bases: one row-stride multiply per task, not per load.
                arow.append(x + c.to(Int64) * Int64(2 * self.k))
            slane = sbase + lane * Int32(4)
            slane2 = sbase + lane * Int32(8)
            total = cute.make_rmem_tensor((self.n8w * self.mpw, 4), Float32)
            total.fill(Float32(0.0))
            words = cute.make_rmem_tensor((2 * self.n16,), Uint32)
            for j in cutlass.range_constexpr(self.n16):
                tile = self._tile_index(nb, Int32(0), warp, Int32(j))
                wa, wb = _load_words4(wbase, tile, lane)
                self._stage(sbase, Int32(0), warp, Int32(j), lane, wa, wb, b_sel, src1, src2)
            cute.arch.sync_threads()
            for kb in cutlass.range(nkb, unroll=1):
                if kb + Int32(1) < Int32(nkb):
                    for j in cutlass.range_constexpr(self.n16):
                        tile = self._tile_index(nb, kb + Int32(1), warp, Int32(j))
                        wa, wb = _load_words4(wbase, tile, lane)
                        words[2 * j], words[2 * j + 1] = wa, wb
                acc = cute.make_rmem_tensor((self.n8w * self.mpw, 4), Float32)
                acc.fill(Float32(0.0))
                stage = (kb % Int32(self.stages)) * Int32(32 * self.n_cta)
                if cutlass.const_expr(self.mode == "f16"):
                    for k16 in cutlass.range_constexpr(K16_PER_BLOCK):
                        kcol = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        a = []
                        for b in cutlass.range_constexpr(self.mpw):
                            q = arow[b] + Int64(16 * self.k)
                            a.append((self._pair(arow[b], kcol), self._pair(q, kcol),
                                      self._pair(arow[b], kcol + Int32(8)),
                                      self._pair(q, kcol + Int32(8))))
                        for i in cutlass.range_constexpr(self.n8w):
                            gi = col_grp * Int32(self.n8w) + Int32(i)
                            addr = slane + (stage + Int32(k16) * Int32(4 * self.n_cta)
                                            + gi * Int32(32)) * Int32(4)
                            b0, b1 = fp8x4_e4m3_to_half2x2(ld_shared_u32(addr))
                            for b in cutlass.range_constexpr(self.mpw):
                                r = i * self.mpw + b
                                d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                    acc[r, 0], acc[r, 1], acc[r, 2], acc[r, 3],
                                    a[b][0], a[b][1], a[b][2], a[b][3], b0, b1)
                                acc[r, 0], acc[r, 1], acc[r, 2], acc[r, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    for r in cutlass.range_constexpr(self.n8w * self.mpw):
                        for v in cutlass.range_constexpr(4):
                            total[r, v] = total[r, v] + acc[r, v] * scale
                else:
                    q4 = (lane & Int32(3)) * Int32(4)
                    nrow = 2 * self.mpw
                    mx = [Uint32(0) for _ in range(nrow)]
                    for k32 in cutlass.range_constexpr(4):
                        kcol = kb * Int32(128) + Int32(k32 * 32) + q4
                        for b in cutlass.range_constexpr(self.mpw):
                            q = arow[b] + Int64(16 * self.k)
                            f0, f1 = self._quad(arow[b], kcol)
                            f2, f3 = self._quad(q, kcol)
                            f4, f5 = self._quad(arow[b], kcol + Int32(16))
                            f6, f7 = self._quad(q, kcol + Int32(16))
                            mx[2 * b] = hmax2(mx[2 * b],
                                              hmax2(hmax2(habs2(f0), habs2(f1)),
                                                    hmax2(habs2(f4), habs2(f5))))
                            mx[2 * b + 1] = hmax2(mx[2 * b + 1],
                                                  hmax2(hmax2(habs2(f2), habs2(f3)),
                                                        hmax2(habs2(f6), habs2(f7))))
                    s = []
                    hsc = []
                    for t in cutlass.range_constexpr(nrow):
                        m = hmax_to_f32(mx[t])
                        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1))
                        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=2))
                        v = cute.arch.fmin(Float32(E4M3_MAX) / cute.arch.fmax(m, Float32(1e-30)),
                                           Float32(60000.0))
                        s.append(v)
                        hsc.append(broadcast_f32_to_half2(v))
                    for k32 in cutlass.range_constexpr(4):
                        kcol = kb * Int32(128) + Int32(k32 * 32) + q4
                        qa = []
                        for b in cutlass.range_constexpr(self.mpw):
                            q = arow[b] + Int64(8 * self.k)
                            hl = hsc[2 * b]
                            hh = hsc[2 * b + 1]
                            f0, f1 = self._quad(arow[b], kcol)
                            f2, f3 = self._quad(q, kcol)
                            f4, f5 = self._quad(arow[b], kcol + Int32(16))
                            f6, f7 = self._quad(q, kcol + Int32(16))
                            qa.append((_e4m3x4_from_half2x2(half2_mul(f0, hl),
                                                            half2_mul(f1, hl)),
                                       _e4m3x4_from_half2x2(half2_mul(f2, hh),
                                                            half2_mul(f3, hh)),
                                       _e4m3x4_from_half2x2(half2_mul(f4, hl),
                                                            half2_mul(f5, hl)),
                                       _e4m3x4_from_half2x2(half2_mul(f6, hh),
                                                            half2_mul(f7, hh))))
                        for i in cutlass.range_constexpr(self.n8w):
                            gi = col_grp * Int32(self.n8w) + Int32(i)
                            addr = slane2 + (stage + Int32(k32) * Int32(8 * self.n_cta)
                                             + gi * Int32(64)) * Int32(4)
                            b0, b1 = ld_shared_v2_u32(addr)
                            for b in cutlass.range_constexpr(self.mpw):
                                r = i * self.mpw + b
                                d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                                    acc[r, 0], acc[r, 1], acc[r, 2], acc[r, 3],
                                    qa[b][0], qa[b][1], qa[b][2], qa[b][3], b0, b1)
                                acc[r, 0], acc[r, 1], acc[r, 2], acc[r, 3] = d0, d1, d2, d3
                    scale = _side_value(scales.iterator.toint(), kb)
                    sa = [scale * (Float32(1.0) / s[t]) for t in range(nrow)]
                    for r in cutlass.range_constexpr(self.n8w * self.mpw):
                        b = r % self.mpw
                        total[r, 0] = total[r, 0] + acc[r, 0] * sa[2 * b]
                        total[r, 1] = total[r, 1] + acc[r, 1] * sa[2 * b]
                        total[r, 2] = total[r, 2] + acc[r, 2] * sa[2 * b + 1]
                        total[r, 3] = total[r, 3] + acc[r, 3] * sa[2 * b + 1]
                if kb + Int32(1) < Int32(nkb):
                    for j in cutlass.range_constexpr(self.n16):
                        self._stage(sbase, (kb + Int32(1)) % Int32(self.stages), warp,
                                    Int32(j), lane, words[2 * j], words[2 * j + 1],
                                    b_sel, src1, src2)
                cute.arch.sync_threads()
            for i in cutlass.range_constexpr(self.n8w):
                gi = col_grp * Int32(self.n8w) + Int32(i)
                col = nb * Int32(self.n_cta) + gi * Int32(8) + (lane & Int32(3)) * Int32(2)
                for b in cutlass.range_constexpr(self.mpw):
                    r = i * self.mpw + b
                    addr = obase + (row[b].to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
                    if row[b] < rows:
                        st_global_v2_f32(addr, total[r, 0], total[r, 1])
                    if row[b] + Int32(8) < rows:
                        st_global_v2_f32(addr + Int64(8 * self.n * 4), total[r, 2], total[r, 3])
            task += step

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, payload: cute.Pointer,
                 input_block_scales: cute.Pointer, partial: cute.Pointer,
                 output: cute.Pointer, rows: Int32, rotation_id: Int32,
                 cta_limit: Int32, stream: cuda.CUstream):
        # ``partial`` is part of ABI v1 and unused: this kernel writes the F32
        # output directly, with no split-K partials and no reduce pass.
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations)
                and cta_limit > Int32(0)):
            layout = cute.make_layout(1, stride=1)
            tasks = ((rows + Int32(M_CTA - 1)) // Int32(M_CTA)) * Int32(self.n // self.n_cta)
            self.prefill(cute.make_tensor(rotated_inputs, layout),
                         cute.make_tensor(payload, layout),
                         cute.make_tensor(input_block_scales, layout),
                         cute.make_tensor(output, layout),
                         rows, rotation_id).launch(
                grid=(min(tasks, cta_limit), 1, 1), block=(32 * NWARP, 1, 1),
                stream=stream)


# ---------------------------------------------------------------- MXFP8 records
# One record per (m16 row block, k32): 32 lanes x 16 fragment bytes plus the 16
# per-row E8M0 scale bytes. The 16 fragment bytes at lane ``l`` are the A
# operand of ``mxfp8_mma_m16n8k32_f32_e4m3`` in register order: rows l/4 and
# l/4+8, k = 4*(l&3)..+3 and 16+4*(l&3)..+3. A record is read back with one
# 128-bit lane load plus one scale byte, so no activation ever touches smem and
# every column CTA shares the same quantize pass.
MX8_REC = 528


class TrellisMxfp8Quant:
    """Quantize F16 activations into MXFP8 records (one pass, all column CTAs).

    Per-32-k block scales are E8M0 (``pow2_ceil_ue8m0`` of the block amax over
    448, the exact FlashInfer/OCP rule), so the MMA's ``scale_vec::1X`` operand
    carries them with no in-kernel per-block fold.
    """

    def __init__(self, g):
        self.k = g["input_features"]
        self.bits = g["bits"]
        if self.k % 128:
            raise ValueError("MXFP8 records need K a multiple of 128")
        self.nk32 = self.k // 32
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]

    @cute.kernel
    def quantize(self, inputs: cute.Tensor, records: cute.Tensor,
                 rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        x = inputs[rotation_id]
        rbase = records.iterator.toint()
        n_rb = (rows + Int32(15)) // Int32(16)
        total = n_rb * Int32(self.nk32)
        task = task * Int32(NWARP) + warp
        stride = step * Int32(NWARP)
        row_last = Float32(rows - Int32(1))
        while task < total:
            rb = task // Int32(self.nk32)
            s = task % Int32(self.nk32)
            r0 = rb * Int32(16) + (lane >> Int32(2))
            c = (lane & Int32(3)) * Int32(4)
            klo = s * Int32(32) + c
            khi = klo + Int32(16)
            r0c = Int32(cute.arch.fmin(Float32(r0), row_last))
            r1c = Int32(cute.arch.fmin(Float32(r0 + Int32(8)), row_last))
            x0 = x + r0c.to(Int64) * Int64(2 * self.k)
            x1 = x + r1c.to(Int64) * Int64(2 * self.k)
            # Two rows x two 4-value groups each: a lane holds 16 values.
            f00, f01 = ld_global_nc_v2_u32(x0 + klo.to(Int64) * Int64(2))
            f10, f11 = ld_global_nc_v2_u32(x1 + klo.to(Int64) * Int64(2))
            f20, f21 = ld_global_nc_v2_u32(x0 + khi.to(Int64) * Int64(2))
            f30, f31 = ld_global_nc_v2_u32(x1 + khi.to(Int64) * Int64(2))
            m0 = hmax_to_f32(hmax2(hmax2(habs2(f00), habs2(f01)),
                                   hmax2(habs2(f20), habs2(f21))))
            m1 = hmax_to_f32(hmax2(hmax2(habs2(f10), habs2(f11)),
                                   hmax2(habs2(f30), habs2(f31))))
            # The 32-k block lives in the four lanes of this quad.
            m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=1))
            m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=1))
            m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=2))
            m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=2))
            _, code0 = pow2_ceil_ue8m0(m0 * Float32(1.0 / E4M3_MAX))
            _, code1 = pow2_ceil_ue8m0(m1 * Float32(1.0 / E4M3_MAX))
            h0 = broadcast_f32_to_half2(ue8m0_to_output_scale(code0))
            h1 = broadcast_f32_to_half2(ue8m0_to_output_scale(code1))
            w0 = _e4m3x4_from_half2x2(half2_mul(f00, h0), half2_mul(f01, h0))
            w1 = _e4m3x4_from_half2x2(half2_mul(f10, h1), half2_mul(f11, h1))
            w2 = _e4m3x4_from_half2x2(half2_mul(f20, h0), half2_mul(f21, h0))
            w3 = _e4m3x4_from_half2x2(half2_mul(f30, h1), half2_mul(f31, h1))
            rec = rbase + (rb * Int32(self.nk32) + s).to(Int64) * Int64(MX8_REC)
            st_global_v4_u32(rec + lane.to(Int64) * Int64(16), w0, w1, w2, w3)
            row_lo = (lane >> Int32(2)).to(Int64)
            st_global_u8(rec + Int64(512) + row_lo, Uint8(code0))
            st_global_u8(rec + Int64(520) + row_lo, Uint8(code1))
            task += stride

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, records: cute.Pointer,
                 rows: Int32, rotation_id: Int32, cta_limit: Int32,
                 stream: cuda.CUstream):
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations)
                and cta_limit > Int32(0)):
            layout = cute.make_layout(1, stride=1)
            tasks = ((rows + Int32(15)) // Int32(16)) * Int32(self.nk32)
            warps = (tasks + Int32(NWARP - 1)) // Int32(NWARP)
            self.quantize(cute.make_tensor(rotated_inputs, layout),
                          cute.make_tensor(records, layout),
                          rows, rotation_id).launch(
                grid=(min(warps, cta_limit), 1, 1), block=(32 * NWARP, 1, 1),
                stream=stream)


class TrellisPrefillMxfp8:
    """Many-row trellis dense linear on MXFP8 records.

    ``TrellisMxfp8Quant`` writes the records into the ABI's ``partial`` scratch
    (``rows*K`` bytes of the 4-byte-per-element split-K scratch, so no new
    allocation); this kernel then runs the decode-once E4M3 tile in smem in
    ``ldmatrix`` order -- ``[n][k128]`` bytes with the k16 group XOR-swizzled by
    ``n & 7`` -- and feeds ``mma.kind::mxf8f6f4.block_scale.scale_vec::1X``
    m16n8k32 with the activation's E8M0 scale byte and B at E8M0 127.

    ``part`` selects the pass to launch: ``both`` (the whole route), ``quant``
    or ``gemm`` (the two halves, for separate timing).
    """

    ABI_VERSION = 1

    def __init__(self, g, *, n_cta: int = 64, stages: int = 2,
                 part: str = "both"):
        self.k, self.n = g["input_features"], g["output_features"]
        self.bits = g["bits"]
        if self.bits != 4:
            raise ValueError("the T1 prefill kernel covers K4 trellis only")
        if g["codebook"] != "e4m3":
            raise ValueError("the T1 E4M3 tile covers the e4m3 codebook only")
        if n_cta not in (64, 128) or n_cta > self.n:
            raise ValueError("n_cta must be 64 or 128")
        if n_cta % 16 or (n_cta // 8) % 2:
            raise ValueError("n_cta must cover whole pairs of n8 tiles")
        if part not in ("both", "quant", "gemm"):
            raise ValueError("part must be both, quant or gemm")
        self.input_units = g["sharded_side"] == 2
        self.rotations, self.max_rows = g["rotation_count"], g["max_rows"]
        self.n_cta, self.stages, self.part = n_cta, stages, part
        self.nk32 = self.k // 32
        self.n16 = n_cta // 16
        self.n8w = n_cta // 8
        self.quant = TrellisMxfp8Quant(g)

    def _storage(self):
        # E4M3 B tile: stages x n_cta rows of 128 k-bytes.
        words = self.stages * 32 * self.n_cta

        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[Uint32, words], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def _tile_index(self, nb: Int32, kb: Int32, k16: Int32, n16_local: Int32) -> Int32:
        n16g = nb * Int32(self.n_cta // 16) + n16_local
        group = n16g // Int32(8)
        slot = n16g % Int32(8)
        if cutlass.const_expr(self.input_units):
            return (kb * Int32(self.n // 16 * 8) + group * Int32(64)
                    + k16 * Int32(8) + slot)
        return (group * Int32(self.k // 16 * 8)
                + (kb * Int32(8) + k16) * Int32(8) + slot)

    @cute.jit
    def _stage(self, sbase: Int32, stage: Int32, k16: Int32, n16: Int32,
               lane: Int32, wa: Uint32, wb: Uint32):
        """Decode one K16 x N16 tile into the ldmatrix-order byte tile.

        Byte address of (n, k) inside a stage: ``n*128 + ((k>>4) ^ (n&7))*16 +
        (k&15)``. Every lane of an ``ldmatrix.x4`` reads one 16-byte group, and
        the 8 rows it reads (n = lane&7) land on 8 different bank quads.
        """
        b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, self.bits)
        n0 = n16 * Int32(16) + (lane >> Int32(2))
        kb_off = stage * Int32(128 * self.n_cta) + (lane & Int32(3)) * Int32(2)
        g = k16 ^ ((lane >> Int32(2)) & Int32(7))
        for s in cutlass.range_constexpr(2):
            if cutlass.const_expr(s == 0):
                e = _e4m3x4_from_half2x2(b0, b1)
                n = n0
            else:
                e = _e4m3x4_from_half2x2(b2, b3)
                n = n0 + Int32(8)
            base = sbase + n * Int32(128) + (g << 4) + kb_off
            st_shared_u16(base, e)
            st_shared_u16(base + Int32(8), e >> 16)

    @cute.kernel
    def prefill(self, inputs: cute.Tensor, payload: cute.Tensor,
                scales: cute.Tensor, records: cute.Tensor, output: cute.Tensor,
                rows: Int32, rotation_id: Int32):
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        lane, warp = tid & Int32(31), tid >> Int32(5)
        x = inputs[rotation_id]
        wbase = payload.iterator.toint()
        abase = records.iterator.toint()
        obase = output.iterator.toint()
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        sbase = shared_ptr_to_u32(storage.w.data_ptr())
        n_tiles = Int32(self.n // self.n_cta)
        nkb = self.k // 128
        total_tasks = ((rows + Int32(M_CTA - 1)) // Int32(M_CTA)) * n_tiles
        row_last = Float32(rows - Int32(1))
        # B-fragment ldmatrix rows: matrix m = lane>>3 selects (n8 of the pair,
        # k16 half); every lane addresses column n = (m>>1)*8 + (lane&7).
        blk = Int32((lane >> Int32(3)) & Int32(1))
        bcol = (lane >> Int32(4)) * Int32(8) + (lane & Int32(7))
        b_row_base = sbase + bcol * Int32(128)
        # sfa byte: row l/4 for even lanes, row l/4+8 for odd lanes.
        srow = (lane >> Int32(2)) + (lane & Int32(1)) * Int32(8)
        while task < total_tasks:
            nb = task % n_tiles
            m_base = (task // n_tiles) * Int32(M_CTA)
            row = m_base + warp * Int32(16) + (lane >> Int32(2))
            rc = Int32(cute.arch.fmin(Float32(row), row_last))
            total = cute.make_rmem_tensor((self.n8w, 4), Float32)
            total.fill(Float32(0.0))
            mb = (task // n_tiles) * Int32(NWARP) + warp
            arec = abase + (mb * Int32(self.nk32)).to(Int64) * Int64(MX8_REC)
            krec = Int64(4 * MX8_REC)
            words = cute.make_rmem_tensor((2 * self.n16,), Uint32)
            for j in cutlass.range_constexpr(self.n16):
                tile = self._tile_index(nb, Int32(0), warp, Int32(j))
                wa, wb = _load_words4(wbase, tile, lane)
                self._stage(sbase, Int32(0), warp, Int32(j), lane, wa, wb)
            cute.arch.sync_threads()
            for kb in cutlass.range(nkb, unroll=1):
                if kb + Int32(1) < Int32(nkb):
                    for j in cutlass.range_constexpr(self.n16):
                        tile = self._tile_index(nb, kb + Int32(1), warp, Int32(j))
                        wa, wb = _load_words4(wbase, tile, lane)
                        words[2 * j], words[2 * j + 1] = wa, wb
                acc = cute.make_rmem_tensor((self.n8w, 4), Float32)
                acc.fill(Float32(0.0))
                stage = (kb % Int32(self.stages)) * Int32(128 * self.n_cta)
                rec_kb = arec + kb.to(Int64) * krec
                for k32 in cutlass.range_constexpr(4):
                    rec = rec_kb + Int64(k32 * MX8_REC)
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(rec + lane.to(Int64) * Int64(16))
                    sfa = ld_global_nc_u8(rec + Int64(512) + srow.to(Int64))
                    for p in cutlass.range_constexpr(self.n8w // 2):
                        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                            b_row_base + stage + Int32(2048 * p)
                            + (((Int32(2 * k32) + blk) ^ (lane & Int32(7))) << 4))
                        i = 2 * p
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                            acc[i, 0], acc[i, 1], acc[i, 2], acc[i, 3],
                            a0, a1, a2, a3, b0, b1, sfa, Uint32(0x7F))
                        acc[i, 0], acc[i, 1], acc[i, 2], acc[i, 3] = d0, d1, d2, d3
                        i = 2 * p + 1
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                            acc[i, 0], acc[i, 1], acc[i, 2], acc[i, 3],
                            a0, a1, a2, a3, b2, b3, sfa, Uint32(0x7F))
                        acc[i, 0], acc[i, 1], acc[i, 2], acc[i, 3] = d0, d1, d2, d3
                scale = _side_value(scales.iterator.toint(), kb)
                for r in cutlass.range_constexpr(self.n8w):
                    for v in cutlass.range_constexpr(4):
                        total[r, v] = total[r, v] + acc[r, v] * scale
                if kb + Int32(1) < Int32(nkb):
                    for j in cutlass.range_constexpr(self.n16):
                        self._stage(sbase, (kb + Int32(1)) % Int32(self.stages), warp,
                                    Int32(j), lane, words[2 * j], words[2 * j + 1])
                cute.arch.sync_threads()
            for i in cutlass.range_constexpr(self.n8w):
                col = nb * Int32(self.n_cta) + i * Int32(8) + (lane & Int32(3)) * Int32(2)
                addr = obase + (row.to(Int64) * Int64(self.n) + col.to(Int64)) * Int64(4)
                if row < rows:
                    st_global_v2_f32(addr, total[i, 0], total[i, 1])
                if row + Int32(8) < rows:
                    st_global_v2_f32(addr + Int64(8 * self.n * 4), total[i, 2], total[i, 3])
            task += step

    @cute.jit
    def __call__(self, rotated_inputs: cute.Pointer, payload: cute.Pointer,
                 input_block_scales: cute.Pointer, partial: cute.Pointer,
                 output: cute.Pointer, rows: Int32, rotation_id: Int32,
                 cta_limit: Int32, stream: cuda.CUstream):
        # ``partial`` (the ABI's split-K scratch, 4 B/element) carries the MXFP8
        # records (K*rows bytes) for every column CTA: it is at least
        # ceil(rows/16)*(K/32)*528 B whenever N >= 264.
        if (rows > Int32(0) and rows <= Int32(self.max_rows)
                and rotation_id >= Int32(0) and rotation_id < Int32(self.rotations)
                and cta_limit > Int32(0)):
            if self.part in ("both", "quant"):
                self.quant(rotated_inputs, partial, rows, rotation_id,
                           cta_limit, stream)
            if self.part in ("both", "gemm"):
                layout = cute.make_layout(1, stride=1)
                tasks = ((rows + Int32(M_CTA - 1)) // Int32(M_CTA)) * Int32(self.n // self.n_cta)
                self.prefill(cute.make_tensor(rotated_inputs, layout),
                             cute.make_tensor(payload, layout),
                             cute.make_tensor(input_block_scales, layout),
                             cute.make_tensor(partial, layout),
                             cute.make_tensor(output, layout),
                             rows, rotation_id).launch(
                    grid=(min(tasks, cta_limit), 1, 1), block=(32 * NWARP, 1, 1),
                    stream=stream)
