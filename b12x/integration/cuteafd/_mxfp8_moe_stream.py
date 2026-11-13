"""Streaming routed MXFP8 W8A8 experts (Inkling 2 Small: E4M3 weights with
UE8M0 scales per row and 32-wide K group, MXFP8 K32 activations).

Arithmetic follows the cuteafd Inkling 2 reference (``python/reference/
families/inkling2/model.py::moe_forward``, #1915; the same rule as vLLM
``d664cd1e`` ``ops/{mxfp8,moe,activation}.py``):

    inputs    = mxfp8(x)                      (K32: UE8M0 ceil(amax/448), E4M3 RN satfinite)
    gate, up  = inputs . dequant(w1_e), inputs . dequant(w3_e)
    act       = (4 tanh(gate/4) * sigmoid(gate)) * (25 tanh(up/25)) * gamma_r      (SiTU)
    y         = mxfp8(act) . dequant(w2_e)
    out[r]    = sum_k y[r, k]                 (route weight already in act)

``gamma_r`` is the row's routing weight (``route_scale * softmax(logsigmoid)``),
applied before the FC2 quantization: the combine is an unweighted sum.

Kernels, both on the ``stream`` route's 128-row expert chunks (``MoePrep``):

``StreamMxfp8GateUp`` (``StreamFp8GateUp`` with real B scales)
    gate/up straight from the FP8 K32 wire rows and the E4M3 weight tiles
    (TMA, 128B-swizzled 32-row boxes) through ``kind::mxf8f6f4`` m16n8k32
    MMAs: the wire rows' UE8M0 bytes as the A scales and the weight rows'
    UE8M0 bytes (``s1``/``s3`` ``[E, I, H/32]``, staged per CTA like the
    MXFP4 stream's) as the B scales, FP32 accumulation across the whole K.
    The epilogue rounds gate/up to BF16 (the reference's BF16 GEMM output),
    applies SiTU x gamma in FP32, rounds to BF16, then quantizes each row's
    32-column slice to one MXFP8 K32 block (``act`` rows of
    ``mxfp8_down_row_bytes(I)``: E4M3 payload then UE8M0 bytes).

``StreamMxfp8Down`` (``StreamMxfp4DownA8``'s structure over E4M3 bytes)
    ``y[p] = act_p . dequant(w2_e)^T`` with E4M3 ``w2 [E, H, I]`` and UE8M0
    ``s2 [E, H, I/32]`` (``mxfp4_scale_row_bytes(I)`` per row), 128-K blocks
    double-buffered by cp.async, the same block-scaled MMA with the act
    rows' scales as A and the weight rows' as B. BF16 output ``[pairs, H]``.

Both kernels read every expert weight byte once per layer; the weight
scales add 1/32 of the weight bytes.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.activation import gated_activation_value
from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
    cp_async_u32_shared_global,
    ld_shared_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    max_abs_32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_global_u8,
    st_global_v4_u32,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import _i32_at, _round_bf16
from ._fp8_moe_stream import STREAM_TILE_M, StreamFp8GateUp, _chunk, _u8_weight
from ._mxfp4_down_a8_plan import mxfp4_scale_row_bytes, mxfp8_down_row_bytes

__all__ = ["GroupedGamma", "StreamMxfp8Down", "StreamMxfp8GateUp", "SITU_GATE_CAP", "SITU_UP_CAP"]

# Inkling 2 Small: mlp_gate_softcap 4.0, mlp_up_softcap 25.0.
SITU_GATE_CAP = 4.0
SITU_UP_CAP = 25.0


class StreamMxfp8GateUp(StreamFp8GateUp):
    """``act[p] = mxfp8(situ(x_p . gate_e, x_p . up_e) * gamma_p)`` over E4M3
    ``w1``/``w3`` ``[E, I, H]`` with UE8M0 ``s1``/``s3`` ``[E, I, H/32]``;
    grid ``(I / 32, max_tiles)``, 256 threads, 128-K steps (see module docstring)."""

    inter_alignment = 32
    # Shared memory on SM120 (101,376 B max): A ring 3 x 16 KiB, weights 3 x 8 KiB,
    # A scales 3 x 512 B, weight scales 64 x 148 B. A 4-deep A ring with the scale
    # tile added overran the limit by 2 KiB; the W8A8 stream measured a 3-deep ring
    # at ~4% (its a 3 / w 5 vs a 4 / w 3 note).
    a_ring = 3

    def __init__(self, *, inter: int, hidden: int, experts: int, gate_cap: float = SITU_GATE_CAP,
                 up_cap: float = SITU_UP_CAP, row_addresses: bool = False):
        super().__init__(inter=inter, hidden=hidden, experts=experts, limit=0.0, qmma=True,
                         row_addresses=row_addresses)
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        if self.gate_cap <= 0 or self.up_cap <= 0:
            raise ValueError("SiTU caps must be positive")
        if self.hidden % 128:
            raise ValueError("MXFP8 stream gate/up needs H % 128 == 0 (whole scale words per K step)")
        self.act_row_bytes = mxfp8_down_row_bytes(self.inter)
        # Drained A ring reused for the BF16 activation tile before the K32 quantization.
        self.act_stride = self.cols * 2 + 16
        self.scale_cols = self.hidden // 32
        # Scale rows padded by one word: the stride in words (37 for H 4608) is odd,
        # so a fragment's 8 column rows hit 8 distinct banks.
        self.sc_stride = self.scale_cols + 4
        if (self.sc_stride // 4) % 2 == 0:
            raise ValueError("the weight scale row stride must be an odd number of words")

    def key(self) -> tuple:
        return ("stream_gate_up_mxfp8", 1, self.inter, self.hidden, self.experts, self.a_ring, self.w_ring,
                self.gate_cap, self.up_cap, self.act_row_bytes) + (("row_addresses",) if self.row_addresses else ())

    def _storage(self, w_layout):
        class Storage:
            pass

        Storage.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.a_bytes], 1024],
            "s": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.s_bytes], 16],
            "sc": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.cols * self.sc_stride], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer,
                 s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer, act: cute.Pointer, max_tiles: Int32,
                 stream: cuda.CUstream, grouped_gamma: cute.Pointer = None):
        if cutlass.const_expr(grouped_gamma is None):
            raise ValueError("the MXFP8 stream gate/up applies the route weight before FC2: pass grouped_gamma")
        w_layout = self._w_layout()
        rows_w = self.experts * self.inter
        box = cute.slice_(w_layout, (None, None, 0))
        tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w1, rows_w, self.hidden), box, (self.cols, self.k_step),
            num_multicast=1)
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w3, rows_w, self.hidden), box, (self.cols, self.k_step),
            num_multicast=1)
        self.kernel(x, pair_row, meta, tma_tensor_g, tma_tensor_u, s1, s3, act, tma_g, tma_u, w_layout,
                    self._storage(w_layout), grouped_gamma).launch(
            grid=(self.inter // self.cols, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.jit
    def _step_mx(self, step, slot, sa, sw, ss, sc, warp_m, warp_n, lane, g, acc):
        """One 128-K step: E4M3 A (wire rows, UE8M0 per 32 as the A scales) x E4M3 B
        (weight rows, UE8M0 per 32 as the B scales), accumulating into ``acc``."""
        stage = step % Int32(self.a_ring)
        a_base = sa + stage * Int32(self.a_bytes)
        w_base = sw + slot * Int32(2 * self.half_bytes)
        s_base = ss + stage * Int32(self.s_bytes)
        # A scales: row g (even lanes) or g + 8 (odd lanes) of each m16 tile; byte ks
        # is this step's 32-K group ks.
        asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
        for mt in cutlass.range_constexpr(2):
            srow = warp_m * Int32(32) + Int32(16 * mt) + g + (lane & Int32(1)) * Int32(8)
            asc[mt] = ld_shared_u32(s_base + srow * Int32(4))
        # B scales: this step's four UE8M0 bytes of column g of each n8 tile (gate rows
        # 0..31 of the staged scale tile, up rows 32..63).
        wsc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
        for nt in cutlass.range_constexpr(4):
            col = Int32((nt // 2) * self.cols + (nt % 2) * 8) + warp_n * Int32(16) + g
            wsc[nt] = ld_shared_u32(sc + col * Int32(self.sc_stride) + step * Int32(4))
        for ks in cutlass.range_constexpr(4):
            a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            for mt in cutlass.range_constexpr(2):
                row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                chunk = Int32(2 * ks) + lane // Int32(16)
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    a_base + row * Int32(self.k_step) + ((chunk ^ (row % Int32(8))) * Int32(16)))
                a[4 * mt] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3
            b = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            for p in cutlass.range_constexpr(2):
                # (b[4p], b[4p+1]): column g, K 4j.. and 16+4j..; (b[4p+2], b[4p+3]): column 8 + g.
                nrow = warp_n * Int32(16) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                chunk = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                    w_base + Int32(p * self.half_bytes) + nrow * Int32(self.k_step)
                    + ((chunk ^ (nrow % Int32(8))) * Int32(16)))
                b[4 * p] = b0
                b[4 * p + 1] = b1
                b[4 * p + 2] = b2
                b[4 * p + 3] = b3
            for mt in cutlass.range_constexpr(2):
                for nt in cutlass.range_constexpr(4):
                    f = 4 * (4 * mt + nt)
                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b[2 * nt], b[2 * nt + 1],
                        asc[mt], wsc[nt], bid_a=ks, bid_b=ks)
                    acc[f] = d0
                    acc[f + 1] = d1
                    acc[f + 2] = d2
                    acc[f + 3] = d3

    @cute.jit
    def activation(self, gate: Float32, up: Float32, gamma: Float32) -> Float32:
        """BF16 gate/up (the reference's BF16 GEMM output) -> SiTU x gamma in FP32 -> BF16."""
        value = gated_activation_value(_round_bf16(gate), _round_bf16(up), is_situ=True,
                                       situ_beta=self.gate_cap, situ_linear_beta=self.up_cap)
        return _round_bf16(value * gamma)

    @cute.kernel
    def kernel(self, x: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, weight_g: cute.Tensor,
               weight_u: cute.Tensor, s1: cute.Pointer, s3: cute.Pointer, act: cute.Pointer,
               tma_g: cute.CopyAtom, tma_u: cute.CopyAtom, w_layout: cute.ComposedLayout,
               Storage: cutlass.Constexpr, grouped_gamma: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            mbar = storage.mbar.data_ptr()
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            sa = shared_ptr_to_u32(storage.a.data_ptr())
            ss = shared_ptr_to_u32(storage.s.data_ptr())
            sc = shared_ptr_to_u32(storage.sc.data_ptr())
            h, i = self.hidden, self.inter
            n0 = n_blk * Int32(self.cols)
            w_tile = e * Int32(i // self.cols) + n_blk
            t_ws, t_wg = cpasync.tma_partition(tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                               cute.group_modes(cute.local_tile(weight_g, (self.cols, self.k_step),
                                                                                (None, None)), 0, 2))
            _, t_wu = cpasync.tma_partition(tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                                            cute.group_modes(cute.local_tile(weight_u, (self.cols, self.k_step),
                                                                             (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_g)
                cpasync.prefetch_descriptor(tma_u)
                for slot in cutlass.range_constexpr(self.w_ring):
                    cute.arch.mbarrier_init(mbar + slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            # Loader roles: 16-byte chunk ch of rows tidx/8 + 32q; one row's UE8M0 word per step.
            ch = tidx % Int32(8)
            r_base = tidx // Int32(8)
            src = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
            for q in cutlass.range_constexpr(4):
                row = r_base + Int32(32 * q)
                src[q] = Int64(-1)
                if row < live:
                    source = _i32_at(Int64(pair_row.toint()), first + row)
                    src[q] = self.row_address(x, source) + Int64(ch) * Int64(16)
            s_src = Int64(-1)
            if tidx < live:
                source = _i32_at(Int64(pair_row.toint()), first + tidx)
                s_src = self.row_address(x, source) + Int64(h)
            # The CTA's weight scale rows (32 gate then 32 up columns, H/32 bytes each) as
            # 4-byte words (the padded rows are word-aligned, not 16-byte-aligned), once;
            # they join the first cp.async group.
            scale_words = self.scale_cols // 4
            for q in cutlass.range_constexpr((2 * self.cols * scale_words + self.threads - 1) // self.threads):
                u = tidx + Int32(q * self.threads)
                if u < Int32(2 * self.cols * scale_words):
                    r = u // Int32(scale_words)
                    c = u % Int32(scale_words)
                    base = Int64(s1.toint())
                    col = r
                    if r >= Int32(self.cols):
                        base = Int64(s3.toint())
                        col = r - Int32(self.cols)
                    cp_async_u32_shared_global(
                        sc + r * Int32(self.sc_stride) + c * Int32(4),
                        base + (Int64(e) * Int64(i) + Int64(n0 + col)) * Int64(self.scale_cols) + Int64(c) * Int64(4))

            # Prologue: rows of steps 0 .. a_ring-2 (cp.async groups), weights of
            # steps 0 .. w_ring-2 (TMA).
            for p in cutlass.range_constexpr(self.a_ring - 1):
                self._load_a(p, sa, ss, src, s_src, ch, r_base, tidx)
                cute.arch.cp_async_commit_group()
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(self.w_ring - 1):
                    self._load_w(p, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)

            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g = lane // Int32(4)
            j = lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(32):
                acc[v] = Float32(0.0)
            for step in cutlass.range(self.k_steps, unroll=1):
                slot = step % Int32(self.w_ring)
                cute.arch.cp_async_wait_group(self.a_ring - 2)
                cute.arch.mbarrier_wait(mbar + slot, (step // Int32(self.w_ring)) & Int32(1))
                cute.arch.sync_threads()
                if step + Int32(self.a_ring - 1) < Int32(self.k_steps):
                    self._load_a(step + Int32(self.a_ring - 1), sa, ss, src, s_src, ch, r_base, tidx)
                cute.arch.cp_async_commit_group()
                if warp_id == Int32(0):
                    if step + Int32(self.w_ring - 1) < Int32(self.k_steps):
                        self._load_w(step + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                if active:
                    self._step_mx(step, slot, sa, sw, ss, sc, warp_m, warp_n, lane, g, acc)
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            # Epilogue: SiTU x gamma per (row, column) into the drained A ring as BF16 ...
            gamma_t = cute.make_ptr(Float32, Int64(grouped_gamma.toint()), cute.AddressSpace.gmem, assumed_align=4)
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live:
                            gamma = Float32(gamma_t[first + row])
                            for nt in cutlass.range_constexpr(2):
                                fg = 4 * (4 * mt + nt) + 2 * half
                                fu = 4 * (4 * mt + nt + 2) + 2 * half
                                col = warp_n * Int32(16) + Int32(8 * nt) + Int32(2) * j
                                v0 = self.activation(acc[fg], acc[fu], gamma)
                                v1 = self.activation(acc[fg + 1], acc[fu + 1], gamma)
                                st_shared_u32(sa + row * Int32(self.act_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(v0, v1))
            cute.arch.sync_threads()
            # ... then one thread per row quantizes the CTA's 32 columns as one K32 block.
            if tidx < live:
                values = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
                for q in cutlass.range_constexpr(4):
                    words = ld_shared_v4_u32(sa + tidx * Int32(self.act_stride) + Int32(16 * q))
                    for word in cutlass.range_constexpr(4):
                        values[8 * q + 2 * word] = _bf16_lo(words[word])
                        values[8 * q + 2 * word + 1] = _bf16_hi(words[word])
                payload, scale = quantize_block_fp8_mx(values, max_abs_32(values))
                at = Int64(act.toint()) + (Int64(tile) * Int64(STREAM_TILE_M) + Int64(tidx)) * Int64(self.act_row_bytes)
                for half in cutlass.range_constexpr(2):
                    st_global_v4_u32(at + Int64(n0) + Int64(16 * half),
                                     payload[4 * half], payload[4 * half + 1],
                                     payload[4 * half + 2], payload[4 * half + 3])
                st_global_u8(at + Int64(i) + Int64(n0 // Int32(32)), cutlass.Uint8(scale))


class StreamMxfp8Down:
    """``y[p] = act_p . dequant(w2_e)^T`` over MXFP8 K32 ``act`` rows and E4M3
    ``w2 [E, H, I]`` with UE8M0 ``s2 [E, H, I/32]``; grid ``(H / 128, max_tiles)``,
    256 threads: warps 4 (32 rows each) x 2 (64 columns each). For 128-aligned
    ``I``, ``w2`` streams by TMA in 128B-swizzled 128x128 tiles across a 3-deep
    ring issued ahead of ``_mma_stage_tma``, ``s2`` is preloaded coalesced once
    per CTA, ``act`` UE8M0 scales batch 4 K-blocks per coalesced load, and only
    live ``act`` rows are staged in a 2-deep XOR-swizzled cp.async ring."""

    threads = 256
    tile_n = 128
    k_block = 128
    w_ring = 3
    a_ring = 2
    sa_group = 4

    def __init__(self, *, hidden: int, inter: int, experts: int):
        self.hidden, self.inter, self.experts = int(hidden), int(inter), int(experts)
        if self.hidden % self.tile_n or self.inter % 32 or self.inter <= 0:
            raise ValueError("MXFP8 stream down needs a 128-aligned H and a positive 32-aligned I")
        self.blocks = (self.inter + self.k_block - 1) // self.k_block
        self.row_bytes = mxfp8_down_row_bytes(self.inter)
        self.scale_row_bytes = mxfp4_scale_row_bytes(self.inter)
        self.out_stride = self.tile_n * 2 + 16
        # TMA weight ring + coalesced scale staging within the 101,376 B SMEM budget:
        # mbar (24 B) + pad->1024 + w (49,152 B) + a (32,768 B) + sa (5,120 B) + sb (<= 11,264 B) <= 99,328 B.
        self.sa_stride = (self.sa_group | 1) * 4
        self.sb_words = self.blocks
        self.sb_stride = (self.sb_words | 1) * 4
        self.tma_w_bytes = self.tile_n * self.k_block
        self.tma_a_bytes = STREAM_TILE_M * self.k_block
        self.tma_sa_bytes = STREAM_TILE_M * self.sa_stride
        self.tma_sb_bytes = self.tile_n * self.sb_stride
        self.tma_w = (self.inter % self.k_block == 0) and (self.tma_sb_bytes <= 11264)
        # Fallback cp.async ring parameters for non-128-aligned I or oversized scale rows.
        self.a_stride = self.k_block + 16
        self.w_stride = self.k_block + 16
        self.s_stride = self.k_block // 32 + 4
        self.a_bytes = STREAM_TILE_M * self.a_stride
        self.w_bytes = self.tile_n * self.w_stride
        self.s_bytes = STREAM_TILE_M * self.s_stride  # == tile_n * s_stride
        self.stage_bytes = self.a_bytes + self.w_bytes + 2 * self.s_bytes
        if self.tma_w:
            if STREAM_TILE_M * self.out_stride > self.w_ring * self.tma_w_bytes:
                raise ValueError("the output staging tile must fit the drained TMA weight ring")
        elif STREAM_TILE_M * self.out_stride > 2 * self.stage_bytes:
            raise ValueError("the output staging tile must fit the drained ring")

    def key(self) -> tuple:
        return ("stream_down_mxfp8", 2, self.hidden, self.inter, self.experts,
                self.tma_w, self.w_ring, self.a_ring, self.sa_group)

    def _w_layout(self):
        w_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_block),
            cutlass.Uint8)
        return cute.tile_to_shape(w_atom, (self.tile_n, self.k_block, self.w_ring), order=(0, 1, 2))

    def _storage_tma(self, w_layout):
        class StorageTma:
            pass

        StorageTma.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.tma_a_bytes], 1024],
            "sa": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.tma_sa_bytes], 16],
            "sb": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.tma_sb_bytes], 16],
        }
        return cute.struct(StorageTma)

    def _storage_legacy(self):
        class StorageLegacy:
            pass

        StorageLegacy.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.stage_bytes], 128],
        }
        return cute.struct(StorageLegacy)

    @cute.jit
    def __call__(self, act: cute.Pointer, meta: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.tma_w):
            w_layout = self._w_layout()
            tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                _u8_weight(w2, self.experts * self.hidden, self.inter),
                cute.slice_(w_layout, (None, None, 0)),
                (self.tile_n, self.k_block),
                num_multicast=1)
            self.kernel_tma(act, meta, tma_tensor_w, s2, y, tma_w, w_layout,
                            self._storage_tma(w_layout)).launch(
                grid=(self.hidden // self.tile_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
                min_blocks_per_mp=1)
        else:
            self.kernel_legacy(act, meta, w2, s2, y, self._storage_legacy()).launch(
                grid=(self.hidden // self.tile_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
                min_blocks_per_mp=1)

    @cute.jit
    def _load_w_tma(self, block, tma_w, t_wg, t_ws, mbar, w_tile):
        slot = Int32(block) % Int32(self.w_ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.tma_w_bytes)
        cute.copy(tma_w, t_wg[(None, w_tile, Int32(block))], t_ws[(None, slot)], tma_bar_ptr=barrier)

    @cute.jit
    def _load_sa_group(self, base_block, ssa_base, a_rows: Int64, live: Int32, tidx):
        """Coalesced 4-word (4 K-blocks = 512 K cols) UE8M0 scale load for live act rows;
        4-byte cp.async guarded per word so 12 B (TP6) and 72 B (TP1) tails stay in-bounds."""
        grp_slot = (Int32(base_block) // Int32(self.sa_group)) % Int32(2)
        dst_base = ssa_base + grp_slot * Int32(self.tma_sa_bytes)
        t_row = tidx // Int32(self.sa_group)
        t_col = tidx % Int32(self.sa_group)
        blk = Int32(base_block) + t_col
        rows_per_iter = self.threads // self.sa_group
        for q in cutlass.range_constexpr(STREAM_TILE_M // (self.threads // self.sa_group)):
            row = t_row + Int32(q * rows_per_iter)
            if (row < live) & (blk < Int32(self.blocks)):
                cp_async_u32_shared_global(
                    dst_base + row * Int32(self.sa_stride) + t_col * Int32(4),
                    a_rows + Int64(row) * Int64(self.row_bytes) + Int64(self.inter) + Int64(blk) * Int64(4))

    @cute.jit
    def _load_a_tma(self, block, sa_base, ssa_base, a_rows: Int64, a_src, live: Int32, ch, r_base, tidx):
        stage = Int32(block) % Int32(self.a_ring)
        k_off = Int64(block) * Int64(self.k_block)
        a_stage = sa_base + stage * Int32(self.tma_a_bytes)
        for q in cutlass.range_constexpr(4):
            if a_src[q] >= Int64(0):
                row = r_base + Int32(32 * q)
                cp_async4_shared_global(
                    a_stage + row * Int32(self.k_block) + ((ch ^ (row % Int32(8))) * Int32(16)),
                    a_src[q] + k_off)
        if Int32(block) % Int32(self.sa_group) == Int32(0):
            self._load_sa_group(block, ssa_base, a_rows, live, tidx)

    @cute.jit
    def _mma_stage_tma(self, block, slot, sa_base, sw_base, ssa_base, ssb_base, acc: cute.Tensor,
                       warp_m, warp_n, lane):
        stage = Int32(block) % Int32(self.a_ring)
        a_stage = sa_base + stage * Int32(self.tma_a_bytes)
        w_stage = sw_base + slot * Int32(self.tma_w_bytes)
        sa_grp_slot = (Int32(block) // Int32(self.sa_group)) % Int32(2)
        sa_sub = Int32(block) % Int32(self.sa_group)
        s_stage = ssa_base + sa_grp_slot * Int32(self.tma_sa_bytes) + sa_sub * Int32(4)
        g = lane // Int32(4)
        asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
        for mt in cutlass.range_constexpr(2):
            srow = warp_m * Int32(32) + Int32(16 * mt) + g + (lane & Int32(1)) * Int32(8)
            asc[mt] = ld_shared_u32(s_stage + srow * Int32(self.sa_stride))
        bsc = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
        for nt in cutlass.range_constexpr(8):
            col = warp_n * Int32(64) + Int32(8 * nt) + g
            bsc[nt] = ld_shared_u32(ssb_base + col * Int32(self.sb_stride) + Int32(block) * Int32(4))
        for ks in cutlass.range_constexpr(self.k_block // 32):
            a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
            for mt in cutlass.range_constexpr(2):
                row = warp_m * Int32(32) + Int32(16 * mt) + (lane % Int32(16))
                chunk = Int32(2 * ks) + lane // Int32(16)
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    a_stage + row * Int32(self.k_block) + ((chunk ^ (row % Int32(8))) * Int32(16)))
                a[4 * mt] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3
            for np in cutlass.range_constexpr(4):
                nrow = warp_n * Int32(64) + Int32(16 * np) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                chunk = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                    w_stage + nrow * Int32(self.k_block) + ((chunk ^ (nrow % Int32(8))) * Int32(16)))
                for half in cutlass.range_constexpr(2):
                    nt = 2 * np + half
                    bl = b0 if half == 0 else b2
                    bh = b1 if half == 0 else b3
                    for mt in cutlass.range_constexpr(2):
                        f = 4 * (8 * mt + nt)
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                            acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                            a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], bl, bh,
                            asc[mt], bsc[nt], bid_a=ks, bid_b=ks)
                        acc[f] = d0
                        acc[f + 1] = d1
                        acc[f + 2] = d2
                        acc[f + 3] = d3

    @cute.kernel
    def kernel_tma(self, act: cute.Pointer, meta: cute.Pointer, weight: cute.Tensor, s: cute.Pointer,
                   y: cute.Pointer, tma_w: cute.CopyAtom, w_layout: cute.ComposedLayout,
                   StorageTma: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(StorageTma)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id, lane = tidx // Int32(32), tidx % Int32(32)
            mbar = storage.mbar.data_ptr()
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            sa = shared_ptr_to_u32(storage.a.data_ptr())
            ssa = shared_ptr_to_u32(storage.sa.data_ptr())
            ssb = shared_ptr_to_u32(storage.sb.data_ptr())
            n0 = n_blk * Int32(self.tile_n)
            w_tile = e * Int32(self.hidden // self.tile_n) + n_blk
            t_ws, t_wg = cpasync.tma_partition(
                tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight, (self.tile_n, self.k_block), (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_w)
                for init_slot in cutlass.range_constexpr(self.w_ring):
                    cute.arch.mbarrier_init(mbar + init_slot, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            a_rows = Int64(act.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.row_bytes)
            s_rows = Int64(s.toint()) + (Int64(e) * Int64(self.hidden) + Int64(n0)) * Int64(self.scale_row_bytes)
            ch = tidx % Int32(8)
            r_base = tidx // Int32(8)
            a_src = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int64)
            for qa in cutlass.range_constexpr(4):
                a_row = r_base + Int32(32 * qa)
                a_src[qa] = Int64(-1)
                if a_row < live:
                    a_src[qa] = a_rows + Int64(a_row) * Int64(self.row_bytes) + Int64(ch) * Int64(16)

            # Coalesced once-per-CTA 4B preload of all K-block UE8M0 scales for the 128 weight rows.
            total_sb = self.tile_n * self.sb_words
            for qsb in cutlass.range_constexpr((total_sb + self.threads - 1) // self.threads):
                u = tidx + Int32(qsb * self.threads)
                if u < Int32(total_sb):
                    sb_r = u // Int32(self.sb_words)
                    sb_c = u % Int32(self.sb_words)
                    cp_async_u32_shared_global(
                        ssb + sb_r * Int32(self.sb_stride) + sb_c * Int32(4),
                        s_rows + Int64(sb_r) * Int64(self.scale_row_bytes) + Int64(sb_c) * Int64(4))

            self._load_a_tma(0, sa, ssa, a_rows, a_src, live, ch, r_base, tidx)
            cute.arch.cp_async_commit_group()
            if warp_id == Int32(0):
                for p in cutlass.range_constexpr(min(self.w_ring - 1, self.blocks)):
                    self._load_w_tma(p, tma_w, t_wg, t_ws, mbar, w_tile)

            warp_m, warp_n = warp_id // Int32(2), warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g, j = lane // Int32(4), lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc[v] = Float32(0.0)
            for block in cutlass.range(self.blocks, unroll=1):
                slot = block % Int32(self.w_ring)
                cute.arch.cp_async_wait_group(0)
                cute.arch.mbarrier_wait(mbar + slot, (block // Int32(self.w_ring)) & Int32(1))
                cute.arch.sync_threads()
                if block + Int32(1) < Int32(self.blocks):
                    self._load_a_tma(block + Int32(1), sa, ssa, a_rows, a_src, live, ch, r_base, tidx)
                    cute.arch.cp_async_commit_group()
                if warp_id == Int32(0):
                    if block + Int32(self.w_ring - 1) < Int32(self.blocks):
                        self._load_w_tma(block + Int32(self.w_ring - 1), tma_w, t_wg, t_ws, mbar, w_tile)
                if active:
                    self._mma_stage_tma(block, slot, sa, sw, ssa, ssb, acc, warp_m, warp_n, lane)
            cute.arch.sync_threads()
            # Coalesced BF16 epilogue through the drained TMA weight ring.
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live:
                            for nt in cutlass.range_constexpr(8):
                                f = 4 * (8 * mt + nt) + 2 * half
                                col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                                st_shared_u32(sw + row * Int32(self.out_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
            cute.arch.sync_threads()
            segments = self.tile_n * 2 // 16
            for q in cutlass.range_constexpr(STREAM_TILE_M * segments // self.threads):
                c = tidx + Int32(self.threads * q)
                row, seg = c // Int32(segments), c % Int32(segments)
                if row < live:
                    v0, v1, v2, v3 = ld_shared_v4_u32(sw + row * Int32(self.out_stride) + seg * Int32(16))
                    st_global_v4_u32(Int64(y.toint()) + ((Int64(first) + Int64(row)) * Int64(self.hidden)
                                                        + Int64(n0)) * Int64(2) + Int64(seg) * Int64(16),
                                     v0, v1, v2, v3)

    @cute.jit
    def _load(self, block, ring, a_rows: Int64, w_rows: Int64, s_rows: Int64, live: Int32, tidx):
        """Fallback block loader for non-128-aligned ``I``; bounds ``act`` reads to ``live`` rows."""
        stage = ring + (Int32(block) % Int32(2)) * Int32(self.stage_bytes)
        for q in cutlass.range_constexpr(STREAM_TILE_M * (self.k_block // 16) // self.threads):
            c = tidx + Int32(q * self.threads)
            row = c // Int32(self.k_block // 16)
            piece = c % Int32(self.k_block // 16)
            dest = stage + row * Int32(self.a_stride) + piece * Int32(16)
            source = a_rows + Int64(row) * Int64(self.row_bytes) + Int64(block) * Int64(self.k_block) \
                + Int64(piece) * Int64(16)
            if const_expr(self.inter % self.k_block):
                in_k = (row < live) & (Int32(block) * Int32(self.k_block) + piece * Int32(16) < Int32(self.inter))
                cp_async4_shared_global_pred(dest, source, in_k.to(Int32))
            else:
                cp_async4_shared_global_pred(dest, source, (row < live).to(Int32))
        for q in cutlass.range_constexpr(self.tile_n * (self.k_block // 16) // self.threads):
            c = tidx + Int32(q * self.threads)
            row = c // Int32(self.k_block // 16)
            piece = c % Int32(self.k_block // 16)
            dest = stage + Int32(self.a_bytes) + row * Int32(self.w_stride) + piece * Int32(16)
            source = w_rows + Int64(row) * Int64(self.inter) + Int64(block) * Int64(self.k_block) \
                + Int64(piece) * Int64(16)
            if const_expr(self.inter % self.k_block):
                live_w = Int32(block) * Int32(self.k_block) + piece * Int32(16) < Int32(self.inter)
                cp_async4_shared_global_pred(dest, source, live_w.to(Int32))
            else:
                cp_async4_shared_global(dest, source)
        # Four K32 scales per 128-K block of each act row (A) and weight row (B).
        if tidx < Int32(STREAM_TILE_M):
            at = stage + Int32(self.a_bytes + self.w_bytes) + tidx * Int32(self.s_stride)
            if tidx < live:
                cp_async_u32_shared_global(at, a_rows + Int64(tidx) * Int64(self.row_bytes) + Int64(self.inter)
                                           + Int64(block) * Int64(4))
            cp_async_u32_shared_global(at + Int32(self.s_bytes),
                                       s_rows + Int64(tidx) * Int64(self.scale_row_bytes) + Int64(block) * Int64(4))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _mma_stage(self, block, stage, acc: cute.Tensor, warp_m, warp_n, lane):
        sa = stage
        sw = stage + Int32(self.a_bytes)
        ssa = sw + Int32(self.w_bytes)
        ssb = ssa + Int32(self.s_bytes)
        g = lane // Int32(4)
        asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
        for mt in cutlass.range_constexpr(2):
            srow = warp_m * Int32(32) + Int32(16 * mt) + g + (lane & Int32(1)) * Int32(8)
            asc[mt] = ld_shared_u32(ssa + srow * Int32(self.s_stride))
        bsc = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
        for nt in cutlass.range_constexpr(8):
            col = warp_n * Int32(64) + Int32(8 * nt) + g
            bsc[nt] = ld_shared_u32(ssb + col * Int32(self.s_stride))
        for ks in cutlass.range_constexpr(self.k_block // 32):
            live_k = Int32(1)
            if const_expr(self.inter % self.k_block):
                live_k = (Int32(block) * Int32(self.k_block // 32) + Int32(ks) < Int32(self.inter // 32)).to(Int32)
            if live_k != Int32(0):
                a = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                for mt in cutlass.range_constexpr(2):
                    row = warp_m * Int32(32) + Int32(16 * mt) + lane % Int32(16)
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        sa + row * Int32(self.a_stride) + Int32(32 * ks) + lane // Int32(16) * Int32(16))
                    a[4 * mt] = a0
                    a[4 * mt + 1] = a1
                    a[4 * mt + 2] = a2
                    a[4 * mt + 3] = a3
                for np in cutlass.range_constexpr(4):
                    # Two n8 tiles per ldmatrix: columns 16 np + (0..7 | 8..15) of this warp's 64.
                    nrow = warp_n * Int32(64) + Int32(16 * np) + (lane & Int32(7)) + (lane >> Int32(4)) * Int32(8)
                    chunk = Int32(2 * ks) + ((lane >> Int32(3)) & Int32(1))
                    b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(sw + nrow * Int32(self.w_stride) + chunk * Int32(16))
                    for half in cutlass.range_constexpr(2):
                        nt = 2 * np + half
                        bl = b0 if half == 0 else b2
                        bh = b1 if half == 0 else b3
                        for mt in cutlass.range_constexpr(2):
                            f = 4 * (8 * mt + nt)
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], bl, bh,
                                asc[mt], bsc[nt], bid_a=ks, bid_b=ks)
                            acc[f] = d0
                            acc[f + 1] = d1
                            acc[f + 2] = d2
                            acc[f + 3] = d3

    @cute.kernel
    def kernel_legacy(self, act: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                      y: cute.Pointer, StorageLegacy: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(StorageLegacy)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id, lane = tidx // Int32(32), tidx % Int32(32)
            ring = shared_ptr_to_u32(storage.ring.data_ptr())
            n0 = n_blk * Int32(self.tile_n)
            a_rows = Int64(act.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.row_bytes)
            w_rows = Int64(w.toint()) + (Int64(e) * Int64(self.hidden) + Int64(n0)) * Int64(self.inter)
            s_rows = Int64(s.toint()) + (Int64(e) * Int64(self.hidden) + Int64(n0)) * Int64(self.scale_row_bytes)
            self._load(0, ring, a_rows, w_rows, s_rows, live, tidx)
            if const_expr(self.blocks > 1):
                self._load(1, ring, a_rows, w_rows, s_rows, live, tidx)
            warp_m, warp_n = warp_id // Int32(2), warp_id % Int32(2)
            active = warp_m * Int32(32) < live
            g, j = lane // Int32(4), lane % Int32(4)
            acc = cute.make_rmem_tensor(cute.make_layout((64,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(64):
                acc[v] = Float32(0.0)
            for block in cutlass.range(self.blocks, unroll=1):
                if block + Int32(1) < Int32(self.blocks):
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if active:
                    self._mma_stage(block, ring + (block % Int32(2)) * Int32(self.stage_bytes), acc, warp_m, warp_n,
                                    lane)
                cute.arch.sync_threads()
                if block + Int32(2) < Int32(self.blocks):
                    self._load(block + Int32(2), ring, a_rows, w_rows, s_rows, live, tidx)
            # Coalesced BF16 epilogue through the drained ring.
            if active:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live:
                            for nt in cutlass.range_constexpr(8):
                                f = 4 * (8 * mt + nt) + 2 * half
                                col = warp_n * Int32(64) + Int32(8 * nt) + Int32(2) * j
                                st_shared_u32(ring + row * Int32(self.out_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f], acc[f + 1]))
            cute.arch.sync_threads()
            segments = self.tile_n * 2 // 16
            for q in cutlass.range_constexpr(STREAM_TILE_M * segments // self.threads):
                c = tidx + Int32(self.threads * q)
                row, seg = c // Int32(segments), c % Int32(segments)
                if row < live:
                    v0, v1, v2, v3 = ld_shared_v4_u32(ring + row * Int32(self.out_stride) + seg * Int32(16))
                    st_global_v4_u32(Int64(y.toint()) + ((Int64(first) + Int64(row)) * Int64(self.hidden)
                                                        + Int64(n0)) * Int64(2) + Int64(seg) * Int64(16),
                                     v0, v1, v2, v3)


class GroupedGamma:
    """``gamma[pair_pos[p]] = weights[p]`` for every (row, slot) pair ``p``: the
    route weight of each grouped row, read by the gate/up epilogue before the
    FC2 quantization. Invalid pairs (position -1) are skipped."""

    threads = 256

    def __init__(self, *, top_k: int):
        self.top_k = int(top_k)

    def key(self) -> tuple:
        return ("grouped_gamma", self.top_k)

    @cute.jit
    def __call__(self, pair_pos: cute.Pointer, weights: cute.Pointer, gamma: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        pairs = rows * Int32(self.top_k)
        self.kernel(pair_pos, weights, gamma, pairs).launch(
            grid=((pairs + Int32(self.threads - 1)) // Int32(self.threads), 1, 1), block=(self.threads, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, pair_pos: cute.Pointer, weights: cute.Pointer, gamma: cute.Pointer, pairs: Int32):
        p = Int32(cute.arch.block_idx()[0]) * Int32(self.threads) + Int32(cute.arch.thread_idx()[0])
        if p < pairs:
            position = _i32_at(Int64(pair_pos.toint()), p)
            if position >= Int32(0):
                src = cute.make_ptr(Float32, Int64(weights.toint()), cute.AddressSpace.gmem, assumed_align=4)
                dst = cute.make_ptr(Float32, Int64(gamma.toint()), cute.AddressSpace.gmem, assumed_align=4)
                dst[position] = Float32(src[p])
