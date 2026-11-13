"""W4A4 kernels of the dynamic-amax NVFP4 stream route (Inkling 2 Small native
checkpoint: ``fp8_moe`` ``weights="nvfp4"``, ``activation="situ"``).

Arithmetic follows the checkpoint's own ``hf_quant_config`` contract
(``amax_type: per_token``, ``scale_selection_rule: fouroversix_mae``,
``weight_scale_format: amax_denominator``) as the pinned vendor kernels
implement it (vLLM ``d664cd1e`` ``ops/{nvfp4,moe,activation}.py``, SGLang
``ec5e7ff5`` ``kernels/ops/inkling2/nvfp4.py``):

    per row r:  amax_r = max |x_r|;  enc = 1536 / amax_r;  outer = amax_r / 1536
    per 16-block: s6 = e4m3(min(bmax * enc / 6, 448)),  s4 = e4m3(min(bmax * enc / 4, 448))
                  q_k = e2m1_rn_satfinite(x / (s_k / enc));  keep the candidate whose
                  sum |x - q_k * s_k * outer| is smaller (ties -> 6)
    GEMM:       acc = sum (q_a * s_a) . (q_w * s_w)            (kind::mxf4nvf4, FP32)
    FC1 out:    bf16(acc * (amax_r * w_amax_e) / (1536 * denominator))
    act:        bf16(situ(gate, up) * gamma_r);   then the same row quantization on act
    FC2 out:    bf16(acc * (amax_act * w2_amax_e) / (1536 * denominator));  out = sum over slots

So the block scales the MMA consumes are the raw E4M3 bytes on both sides;
the per-row activation amax and the per-expert weight amax enter once, in
the epilogue. This differs from the calibrated ModelOpt route
(``_nvfp4_moe_a4``: a static ``input_scale`` per expert, ``alpha_e * in_e``
epilogue, one 6.0-based candidate) only in the quantizer and the epilogue
factor; the GEMM kernels are inherited.

``ChunkRowsDyn``        chunk-padded packed activations + E4M3 scales from BF16 rows,
    plus the row amax ``[max_tiles * 128]`` (F32).
``StreamNvfp4GateUpDyn`` ``StreamNvfp4GateUpA4`` with the per-row FC1 factor, the
    SiTU x gamma epilogue and the dynamic quantization of the down input
    (``act`` packed + scales + ``act_amax``).
``StreamNvfp4LinearDyn``  ``StreamNvfp4LinearA4`` with the per-row FC2 factor.

Scale operands: E4M3 ``[E, rows, K/16]`` then F32 ``amax [E]`` then F32
``denominator [1]`` (``nvfp4_amax_offset`` / ``nvfp4_denominator_offset``).
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64, const_expr

from b12x._lib.activation import gated_activation_value
from b12x._lib.intrinsics import (
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    fp8_e4m3_to_f32,
    ld_shared_v4_u32,
    max_abs_16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_u32,
    st_global_u64,
    st_global_u8,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import _ceil, _i32_at, _ld_v2_u32, _round_bf16
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._fp8_weights import _ld_f32
from ._nvfp4_moe_a4 import StreamNvfp4GateUpA4, StreamNvfp4LinearA4
from ._nvfp4_moe_kernels import nvfp4_alpha_offset

__all__ = ["ChunkRowsDyn", "StreamNvfp4GateUpDyn", "StreamNvfp4LinearDyn", "nvfp4_amax_offset",
           "nvfp4_denominator_offset", "FP32_MAX"]

FP32_MAX = 3.4028234663852886e38
E2M1_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nvfp4_amax_offset(experts: int, rows: int, k: int) -> int:
    """Byte offset of the F32 weight amax ``[E]`` in a dynamic NVFP4 scale operand (after the block scales)."""
    return nvfp4_alpha_offset(experts, rows, k)


def nvfp4_denominator_offset(experts: int, rows: int, k: int) -> int:
    """Byte offset of the F32 outer-scale denominator (one value) after the amax."""
    return nvfp4_amax_offset(experts, rows, k) + 4 * int(experts)


@cute.jit
def _e2m1_decode(code: Uint32) -> Float32:
    """The value of one E2M1 nibble (sign, 2 exponent, 1 mantissa bits)."""
    mag = code & Uint32(7)
    value = Float32(0.0)
    if mag == Uint32(1):
        value = Float32(0.5)
    elif mag == Uint32(2):
        value = Float32(1.0)
    elif mag == Uint32(3):
        value = Float32(1.5)
    elif mag == Uint32(4):
        value = Float32(2.0)
    elif mag == Uint32(5):
        value = Float32(3.0)
    elif mag == Uint32(6):
        value = Float32(4.0)
    elif mag == Uint32(7):
        value = Float32(6.0)
    if (code & Uint32(8)) != Uint32(0):
        value = -value
    return value


@cute.jit
def _candidate(values: cute.Tensor, scale: Float32, decoder: Float32, outer: Float32) -> tuple:
    """Packed E2M1 of 16 ``values`` at block scale ``scale`` (``x / (scale * decoder)``,
    ``cvt.rn.satfinite``) and the summed |reconstruction error| with ``scale * outer``."""
    inverse = fmin_f32(div_rn_f32(Float32(1.0), scale * decoder), Float32(FP32_MAX))
    q = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = values[i] * inverse
    lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    recon = scale * outer
    # Pairwise-summed error, as the vendor's _block_error (tree over 2, 4, 8, 16).
    err = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
    for i in cutlass.range_constexpr(16):
        word = lo if i < 8 else hi
        nibble = (word >> Uint32(4 * (i % 8))) & Uint32(0xF)
        err[i] = fabs_f32(_e2m1_decode(nibble) * recon - values[i])
    for width in cutlass.range_constexpr(4):
        step = 1 << width
        for i in cutlass.range_constexpr(0, 16, 2 * step):
            err[i] = err[i] + err[i + step]
    return (Uint64(hi) << Uint64(32)) | Uint64(lo), err[0]


@cute.jit
def _quantize_row16_dyn(values: cute.Tensor, encoder: Float32, decoder: Float32, outer: Float32,
                        packed_at: Int64, scale_at: Int64):
    """One 16-value block under the row's ``encoder``/``decoder``/``outer`` (fouroversix_mae)."""
    bmax = max_abs_16(values)
    s6_u32 = cvt_f32_to_e4m3(fmin_f32(bmax * (encoder * Float32(1.0 / 6.0)), Float32(448.0)))
    s4_u32 = cvt_f32_to_e4m3(fmin_f32(bmax * (encoder * Float32(0.25)), Float32(448.0)))
    s6 = fp8_e4m3_to_f32(s6_u32)
    s4 = fp8_e4m3_to_f32(s4_u32)
    packed6, err6 = _candidate(values, s6, decoder, outer)
    packed4, err4 = _candidate(values, s4, decoder, outer)
    choose4 = err4 < err6
    # A zero block: scale 0 and payload 0 (every candidate reconstructs to zero). Selects
    # only: the DSL requires one type per value across branches.
    zero = bmax == Float32(0.0)
    packed = Uint64(cutlass.select_(choose4, packed4, packed6))
    packed = Uint64(cutlass.select_(zero, Uint64(0), packed))
    scale = Uint32(cutlass.select_(choose4, s4_u32, s6_u32))
    scale = Uint32(cutlass.select_(zero, Uint32(0), scale))
    st_global_u64(packed_at, packed)
    st_global_u8(scale_at, cutlass.Uint8(scale & Uint32(0xFF)))


@cute.jit
def _row_factors(amax: Float32):
    """``(encoder, decoder, outer)`` of a row with amax ``amax``: 1536/amax, its reciprocal, amax/1536."""
    encoder = fmin_f32(div_rn_f32(Float32(1536.0), amax), Float32(FP32_MAX))
    decoder = div_rn_f32(Float32(1.0), encoder)
    outer = amax * Float32(1.0 / 1536.0)
    return encoder, decoder, outer


class ChunkRowsDyn:
    """Chunk-padded dynamic NVFP4 activations of the chunk rows from BF16 ``[rows, K]``:
    packed ``[max_tiles * 128, K/2]``, E4M3 scales ``[max_tiles * 128, K/16]`` and the
    row amax ``[max_tiles * 128]``. One CTA per live grouped pair; the whole CTA
    reduces the row amax first, then one thread per 16-value group quantizes."""

    threads = 128

    def __init__(self, *, k: int, experts: int):
        self.k, self.experts = int(k), int(experts)
        if self.k % 16 or self.k % 8:
            raise ValueError("K must be a multiple of 16")

    def key(self) -> tuple:
        return ("chunk_rows_dyn", 1, self.k, self.experts)

    @cute.jit
    def __call__(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, aq: cute.Pointer,
                 a_s: cute.Pointer, amax: cute.Pointer, pairs: Int32, stream: cuda.CUstream):
        self.kernel(src, pair_row, meta, aq, a_s, amax).launch(
            grid=(pairs, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, aq: cute.Pointer,
               a_s: cute.Pointer, amax: cute.Pointer):
        p = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        red = smem.allocate_tensor(element_type=Float32, layout=cute.make_layout((self.threads,)), byte_alignment=16)
        tile = Int32(0)
        end = _i32_at(Int64(meta.toint()), 1)
        while tile + Int32(1) < end:
            mid = (tile + end) // Int32(2)
            _, start, _ = _chunk(meta, self.experts, mid)
            if start <= p:
                tile = mid
            else:
                end = mid
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            r = p - first
            i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(r)
            if r < live:
                source = _i32_at(Int64(pair_row.toint()), first + r)
                row_at = Int64(src.toint()) + Int64(source) * Int64(self.k) * Int64(2)
                # Row amax: each thread over its 8-element pieces, then a shared reduction.
                local = Float32(0.0)
                pieces = self.k // 8
                for it in cutlass.range_constexpr(_ceil(pieces, self.threads)):
                    v = Int32(it * self.threads) + tidx
                    if v < Int32(pieces):
                        lo, hi = _ld_v2_u32(row_at + Int64(v) * Int64(16))
                        lo2, hi2 = _ld_v2_u32(row_at + Int64(v) * Int64(16) + Int64(8))
                        for w in (lo, hi, lo2, hi2):
                            local = fmax_f32(local, fmax_f32(fabs_f32(_bf16_lo(w)), fabs_f32(_bf16_hi(w))))
                red[tidx] = local
                cute.arch.sync_threads()
                stride = self.threads // 2
                while stride > 0:
                    if tidx < Int32(stride):
                        red[tidx] = fmax_f32(Float32(red[tidx]), Float32(red[tidx + Int32(stride)]))
                    cute.arch.sync_threads()
                    stride = stride // 2
                row_amax = Float32(red[0])
                if tidx == Int32(0):
                    st_global_f32(Int64(amax.toint()) + i * Int64(4), row_amax)
                encoder, decoder, outer = _row_factors(row_amax)
                groups = self.k // 16
                for it in cutlass.range_constexpr(_ceil(groups, self.threads)):
                    grp = Int32(it * self.threads) + tidx
                    if grp < Int32(groups):
                        values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                        at = row_at + Int64(grp) * Int64(32)
                        lo, hi = _ld_v2_u32(at)
                        lo2, hi2 = _ld_v2_u32(at + Int64(8))
                        lo3, hi3 = _ld_v2_u32(at + Int64(16))
                        lo4, hi4 = _ld_v2_u32(at + Int64(24))
                        words = (lo, hi, lo2, hi2, lo3, hi3, lo4, hi4)
                        for q in cutlass.range_constexpr(8):
                            values[2 * q] = _bf16_lo(words[q])
                            values[2 * q + 1] = _bf16_hi(words[q])
                        _quantize_row16_dyn(values, encoder, decoder, outer,
                                            Int64(aq.toint()) + i * Int64(self.k // 2) + Int64(grp) * Int64(8),
                                            Int64(a_s.toint()) + i * Int64(groups) + Int64(grp))


class StreamNvfp4LinearDyn(StreamNvfp4LinearA4):
    """``y[p] = bf16(acc * (amax_p * w_amax_e) / (1536 * denominator))``: the A4 linear
    with the per-row factor read from ``row_amax`` and the scale operand's amax tail."""

    def __init__(self, *, n: int, k: int, experts: int, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128):
        super().__init__(n=n, k=k, experts=experts, tile_m=tile_m, tile_n=tile_n, live_min=live_min,
                         live_max=live_max)
        self.amax_offset = nvfp4_amax_offset(experts, self.n, self.k)
        self.denominator_offset = nvfp4_denominator_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("stream_linear_nvfp4_dyn", 1, *super().key()[2:])

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream, row_amax: cute.Pointer = None):
        if cutlass.const_expr(row_amax is None):
            raise ValueError("the dynamic NVFP4 linear needs the activation rows' amax")
        self.kernel_dyn(aq, a_s, meta, w, s, y, row_amax).launch(
            grid=(self.n // self.tile_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel_dyn(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                   y: cute.Pointer, row_amax: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                ring = shared_ptr_to_u32(storage.ring.data_ptr())
                n0 = n_blk * Int32(self.tile_n)
                a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
                as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
                w_off = Int64(e) * Int64(self.n) + Int64(n0)
                w_rows = Int64(w.toint()) + w_off * Int64(self.k // 2)
                s_rows = Int64(s.toint()) + w_off * Int64(self.k // 16)
                # Expert factor: w_amax_e / (1536 * denominator); the row's amax multiplies in the epilogue.
                expert_factor = _ld_f32(Int64(s.toint()) + Int64(self.amax_offset) + Int64(e) * Int64(4)) \
                    * cute.arch.rcp_approx(Float32(1536.0) * _ld_f32(Int64(s.toint()) + Int64(self.denominator_offset)))
                self._load(0, ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                if const_expr(self.blocks > 1):
                    self._load(1, ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                warp_m = warp_id // Int32(2)
                warp_n = warp_id % Int32(2)
                active = warp_m * Int32(self.warp_rows) < live
                q = lane // Int32(4)
                c = lane % Int32(4)
                mat = lane // Int32(8)
                mrow = lane % Int32(8)
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(self.frags):
                    acc[v] = Float32(0.0)
                for block in cutlass.range(self.blocks, unroll=1):
                    if block + Int32(1) < Int32(self.blocks):
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    if active:
                        self._mma_stage(ring + (block % Int32(2)) * Int32(self.stage_bytes), acc, warp_m, warp_n,
                                        q, c, mat, mrow)
                    cute.arch.sync_threads()
                    if block + Int32(2) < Int32(self.blocks):
                        self._load(block + Int32(2), ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                if active:
                    amax_t = Int64(row_amax.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(4)
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                            if row < live:
                                factor = _ld_f32(amax_t + Int64(row) * Int64(4)) * expert_factor
                                for nt in cutlass.range_constexpr(self.n_tiles):
                                    f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                    col = n0 + warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                    st_global_u32(Int64(y.toint()) + (Int64(first + row) * Int64(self.n) + Int64(col))
                                                  * Int64(2), pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))


class StreamNvfp4GateUpDyn(StreamNvfp4GateUpA4):
    """Fused W4A4 gate/up + SiTU x gamma + dynamic NVFP4 quantization of the down input.

    ``StreamNvfp4GateUpA4``'s main loop; the epilogue scales each row by
    ``amax_row * w_amax_e / (1536 * denominator)`` (gate rows with ``s1``'s amax, up rows
    with ``s3``'s), rounds to BF16 into the exchange, computes
    ``bf16(situ(g, u) * gamma_row)`` per 16 columns, then — needing the whole act row's
    amax before any block can be quantized — writes each 16-column group's amax to
    ``act_amax_part [rows, I / 16]`` and leaves the quantization to :class:`ChunkActDyn`,
    which runs once per row after every column CTA has written its BF16 act slice to
    ``act_bf16 [max_tiles * 128, I]``."""

    def __init__(self, *, inter: int, hidden: int, experts: int, gate_cap: float = 4.0, up_cap: float = 25.0,
                 k_block: int | None = None, stages: int = 2, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128):
        super().__init__(inter=inter, hidden=hidden, experts=experts, limit=0.0, k_block=k_block, stages=stages,
                         tile_m=tile_m, tile_n=tile_n, live_min=live_min, live_max=live_max)
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        self.amax_offset = nvfp4_amax_offset(experts, self.inter, self.k)
        self.denominator_offset = nvfp4_denominator_offset(experts, self.inter, self.k)
        self.parts = self.inter // self.half_n

    def key(self) -> tuple:
        return ("stream_gate_up_nvfp4_dyn", 2, self.inter, self.k, self.experts, self.k_block, self.stages,
                self.gate_cap, self.up_cap, self.tile_m, self.tile_n, self.live_min, self.live_max)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                 w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer,
                 max_tiles: Int32, stream: cuda.CUstream, row_amax: cute.Pointer = None,
                 gamma: cute.Pointer = None, act_bf16: cute.Pointer = None, act_amax_part: cute.Pointer = None):
        if cutlass.const_expr(row_amax is None or gamma is None or act_bf16 is None or act_amax_part is None):
            raise ValueError("the dynamic NVFP4 gate/up needs row_amax, gamma, act_bf16 and act_amax_part")
        self.fused_dyn(aq, a_s, meta, w1, s1, w3, s3, row_amax, gamma, act_bf16, act_amax_part).launch(
            grid=(self.inter // self.half_n, max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.jit
    def _situ(self, gate: Float32, up: Float32, gamma: Float32) -> Float32:
        value = gated_activation_value(gate, up, is_situ=True, situ_beta=self.gate_cap, situ_linear_beta=self.up_cap)
        return _round_bf16(value * gamma)

    @cute.kernel
    def fused_dyn(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                  w3: cute.Pointer, s3: cute.Pointer, row_amax: cute.Pointer, gamma: cute.Pointer,
                  act_bf16: cute.Pointer, act_amax_part: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                ring = shared_ptr_to_u32(storage.ring.data_ptr())
                n0 = n_blk * Int32(self.half_n)
                a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
                as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
                w_off = Int64(e) * Int64(self.inter) + Int64(n0)
                w_rows = Int64(w1.toint()) + w_off * Int64(self.k // 2)
                w_up = Int64(w3.toint()) + w_off * Int64(self.k // 2)
                s_rows = Int64(s1.toint()) + w_off * Int64(self.k // 16)
                s_up = Int64(s3.toint()) + w_off * Int64(self.k // 16)
                for pre in cutlass.range_constexpr(self.stages - 1):
                    if const_expr(pre < self.blocks):
                        self._load2(pre, ring, a_rows, as_rows, w_rows, w_up, s_rows, s_up, live, tidx)
                    else:
                        cute.arch.cp_async_commit_group()
                warp_m = warp_id // Int32(2)
                warp_n = warp_id % Int32(2)
                sp = cutlass.select_(warp_n == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
                expert_factor = _ld_f32(sp + Int64(self.amax_offset) + Int64(e) * Int64(4)) \
                    * cute.arch.rcp_approx(Float32(1536.0) * _ld_f32(sp + Int64(self.denominator_offset)))
                active = warp_m * Int32(self.warp_rows) < live
                q = lane // Int32(4)
                c = lane % Int32(4)
                mat = lane // Int32(8)
                mrow = lane % Int32(8)
                acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(self.frags):
                    acc[v] = Float32(0.0)
                for block in cutlass.range(self.blocks, unroll=1):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.sync_threads()
                    if block + Int32(self.stages - 1) < Int32(self.blocks):
                        self._load2(block + Int32(self.stages - 1), ring, a_rows, as_rows, w_rows, w_up, s_rows, s_up,
                                    live, tidx)
                    else:
                        cute.arch.cp_async_commit_group()
                    if active:
                        stage = ring + (block % Int32(self.stages)) * Int32(self.stage_bytes)
                        self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow)
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                # Exchange: BF16 gate / up tiles scaled by the row's FC1 factor.
                amax_t = Int64(row_amax.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(4)
                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                            factor = Float32(0.0)
                            if row < live:
                                factor = _ld_f32(amax_t + Int64(row) * Int64(4)) * expert_factor
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                col = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                st_shared_u32(ring + row * Int32(self.x_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
                cute.arch.sync_threads()
                # SiTU x gamma per 16 columns -> BF16 act slice + the slice's amax per row.
                groups = self.half_n // 16
                gamma_t = Int64(gamma.toint())
                for it in cutlass.range_constexpr(_ceil(self.tile_m * groups, self.threads)):
                    gi = tidx + Int32(it * self.threads)
                    row = gi // Int32(groups)
                    grp = gi % Int32(groups)
                    if row < live:
                        g_row = _ld_f32(gamma_t + Int64(first + row) * Int64(4))
                        at = ring + row * Int32(self.x_stride) + grp * Int32(32)
                        i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(row)
                        out_at = Int64(act_bf16.toint()) + (i * Int64(self.inter) + Int64(n0) + Int64(grp) * Int64(16)) * Int64(2)
                        local = Float32(0.0)
                        for half in cutlass.range_constexpr(2):
                            gw = ld_shared_v4_u32(at + Int32(16 * half))
                            uw = ld_shared_v4_u32(at + Int32(self.half_n * 2 + 16 * half))
                            packed = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                            for qq in cutlass.range_constexpr(4):
                                v0 = self._situ(_bf16_lo(gw[qq]), _bf16_lo(uw[qq]), g_row)
                                v1 = self._situ(_bf16_hi(gw[qq]), _bf16_hi(uw[qq]), g_row)
                                local = fmax_f32(local, fmax_f32(fabs_f32(v0), fabs_f32(v1)))
                                packed[qq] = pack_f32x2_to_bfloat2(v0, v1)
                            st_global_u32(out_at + Int64(16 * half), packed[0])
                            st_global_u32(out_at + Int64(16 * half + 4), packed[1])
                            st_global_u32(out_at + Int64(16 * half + 8), packed[2])
                            st_global_u32(out_at + Int64(16 * half + 12), packed[3])
                        # Per-(row, 16-column group) amax straight to global: act_amax_part is
                        # [padded rows, parts * groups]; ChunkActDyn takes the row max.
                        # Partial index = this 16-column group's position in the row: n0/16 + grp (tile-independent).
                        st_global_f32(Int64(act_amax_part.toint())
                                      + (i * Int64(self.inter // 16) + Int64(n0 // Int32(16)) + Int64(grp)) * Int64(4),
                                      local)


class ChunkActDyn:
    """Dynamic NVFP4 quantization of the BF16 act rows written by :class:`StreamNvfp4GateUpDyn`:
    the row amax is the max over its ``parts`` partial amaxes; one CTA per chunk row,
    one thread per 16-value group. Writes packed act, E4M3 scales and the row amax."""

    threads = 128

    def __init__(self, *, inter: int, experts: int, parts: int):
        self.inter, self.experts, self.parts = int(inter), int(experts), int(parts)
        if self.inter % 16 or self.inter % self.parts or (self.inter // self.parts) % 16:
            raise ValueError("I must be a multiple of 16 and of 16 * parts")
        # One partial per 16-column group of the row (the gate/up CTAs write parts * (half_n/16) of them).
        self.partials = self.inter // 16

    def key(self) -> tuple:
        return ("chunk_act_dyn", 2, self.inter, self.experts, self.parts)

    @cute.jit
    def __call__(self, act_bf16: cute.Pointer, act_amax_part: cute.Pointer, meta: cute.Pointer, aq: cute.Pointer,
                 a_s: cute.Pointer, amax: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(act_bf16, act_amax_part, meta, aq, a_s, amax).launch(
            grid=(max_tiles * Int32(STREAM_TILE_M), 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, act_bf16: cute.Pointer, act_amax_part: cute.Pointer, meta: cute.Pointer, aq: cute.Pointer,
               a_s: cute.Pointer, amax: cute.Pointer):
        i32 = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        tile = i32 // Int32(STREAM_TILE_M)
        r = i32 % Int32(STREAM_TILE_M)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if r < live:
                i = Int64(i32)
                row_amax = Float32(0.0)
                for part in cutlass.range_constexpr(self.partials):
                    row_amax = fmax_f32(row_amax, _ld_f32(Int64(act_amax_part.toint())
                                                          + (i * Int64(self.partials) + Int64(part)) * Int64(4)))
                if tidx == Int32(0):
                    st_global_f32(Int64(amax.toint()) + i * Int64(4), row_amax)
                encoder, decoder, outer = _row_factors(row_amax)
                groups = self.inter // 16
                row_at = Int64(act_bf16.toint()) + i * Int64(self.inter) * Int64(2)
                for it in cutlass.range_constexpr(_ceil(groups, self.threads)):
                    grp = Int32(it * self.threads) + tidx
                    if grp < Int32(groups):
                        values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                        at = row_at + Int64(grp) * Int64(32)
                        lo, hi = _ld_v2_u32(at)
                        lo2, hi2 = _ld_v2_u32(at + Int64(8))
                        lo3, hi3 = _ld_v2_u32(at + Int64(16))
                        lo4, hi4 = _ld_v2_u32(at + Int64(24))
                        words = (lo, hi, lo2, hi2, lo3, hi3, lo4, hi4)
                        for q in cutlass.range_constexpr(8):
                            values[2 * q] = _bf16_lo(words[q])
                            values[2 * q + 1] = _bf16_hi(words[q])
                        _quantize_row16_dyn(values, encoder, decoder, outer,
                                            Int64(aq.toint()) + i * Int64(self.inter // 2) + Int64(grp) * Int64(8),
                                            Int64(a_s.toint()) + i * Int64(groups) + Int64(grp))




class GroupedGamma:
    """``gamma[pair_pos[p]] = weights[p]`` for every (row, slot) pair ``p``: the route
    weight of each grouped row, read by the gate/up epilogue before the FC2 quantization."""

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


__all__ += ["ChunkActDyn", "GroupedGamma"]
