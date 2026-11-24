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
    act:        bf16(situ(gate, up) * gamma_r)  (vendor PTX: ex2/rcp/tanh .approx);  same row quantization on act
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
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, Uint64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.activation import _f32_asm, situ_vendor
from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    create_l2_evict_first_policy,
    ld_global_nc_u32_l2hint,
    ld_global_nc_v2_u32_l2hint,
    ld_global_nc_v4_u32,
    ld_global_nc_v4_u32_l2hint,
    ld_global_v4_u32,
    ld_shared_u32 as ld_shared_u32_at,
    ldmatrix_m8n8x4_b16,
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    fp8_e4m3_to_f32,
    ld_shared_v4_u32,
    max_abs_16,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_u32,
    st_global_u64,
    st_global_u8,
    st_global_v4_u32,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import META_HEAD, _atomic_add_shared_i32, _ceil, _i32_at, _ld_v2_u32, _round_bf16, _set_i32
from ._fp8_moe_stream import STREAM_TILE_M, _chunk, _u8_weight
from ._fp8_weights import _ld_f32
from ._nvfp4_moe_a4 import StreamNvfp4GateUpA4, StreamNvfp4LinearA4, _copy_nvfp4_group
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
    # Reconstruction error |(q * scale) * outer - x|: the vendor's operation order with every step
    # rounded separately. vLLM d664cd1e ops/nvfp4.py:108-110 (`_pair`: `(low * scale) * outer_scale - x`),
    # launched with enable_fp_fusion=False (:231; SGLang ec5e7ff5 nvfp4.py:145). Keep it as explicit
    # mul.rn / mul.rn / sub.rn PTX: the DSL's `q * (scale * outer) - x` reorders the products and NVPTX
    # contracts the multiply-subtract into an FMA, and either changes the error's last ulp, which is
    # what decides the 6-vs-4 choice on blocks whose two errors tie (measured: 3 of 36,864 blocks at
    # 128 rows, 100-290 BF16 ulps on those rows' FC1). The sums are add.rn over adjacent pairs, then
    # 4s, 8s, 16, as the vendor's `_block_error` (:39-45).
    err = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
    for i in cutlass.range_constexpr(16):
        word = lo if i < 8 else hi
        nibble = (word >> Uint32(4 * (i % 8))) & Uint32(0xF)
        err[i] = _f32_asm("{ .reg .f32 t; mul.rn.f32 t, $1, $2; mul.rn.f32 t, t, $3; sub.rn.f32 t, t, $4; abs.f32 $0, t; }",
                          _e2m1_decode(nibble), scale, outer, values[i])
    for width in cutlass.range_constexpr(4):
        step = 1 << width
        for i in cutlass.range_constexpr(0, 16, 2 * step):
            err[i] = _f32_asm("add.rn.f32 $0, $1, $2;", err[i], err[i + step])
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


@cute.jit
def _bf16_row_amax(row_at: Int64, k: cutlass.Constexpr, threads: cutlass.Constexpr,
                   tidx: Int32, red: cute.Tensor):
    local = Float32(0.0)
    pieces = k // 8
    for it in cutlass.range_constexpr(_ceil(pieces, threads)):
        v = Int32(it * threads) + tidx
        if v < Int32(pieces):
            lo, hi = _ld_v2_u32(row_at + Int64(v) * Int64(16))
            lo2, hi2 = _ld_v2_u32(row_at + Int64(v) * Int64(16) + Int64(8))
            for w in (lo, hi, lo2, hi2):
                local = fmax_f32(local, fmax_f32(fabs_f32(_bf16_lo(w)), fabs_f32(_bf16_hi(w))))
    red[tidx] = local
    cute.arch.sync_threads()
    stride = threads // 2
    while stride > 0:
        if tidx < Int32(stride):
            red[tidx] = fmax_f32(Float32(red[tidx]), Float32(red[tidx + Int32(stride)]))
        cute.arch.sync_threads()
        stride = stride // 2
    return Float32(red[0])


class ChunkRowsDyn:
    """Chunk-padded dynamic NVFP4 activations of the chunk rows from BF16 ``[rows, K]``:
    packed ``[max_tiles * 128, K/2]``, E4M3 scales ``[max_tiles * 128, K/16]`` and the
    row amax ``[max_tiles * 128]``. One CTA per live grouped pair; the whole CTA
    reduces the row amax first, then one thread per 16-value group quantizes."""

    threads = 128

    def __init__(self, *, k: int, experts: int, input_kind: str = "bf16"):
        self.k, self.experts = int(k), int(experts)
        if input_kind not in ("bf16", "nvfp4"):
            raise ValueError("dynamic NVFP4 input_kind must be bf16 or nvfp4")
        self.input_kind = input_kind
        self.row_bytes = self.k // 2 + self.k // 16 + 16 if input_kind == "nvfp4" else self.k * 2
        if self.k % 16 or self.k % 8:
            raise ValueError("K must be a multiple of 16")

    def key(self) -> tuple:
        return ("chunk_rows_dyn", 2, self.k, self.experts) + (("nvfp4_input",) if self.input_kind == "nvfp4" else ())

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
                row_at = Int64(src.toint()) + Int64(source) * Int64(self.row_bytes)
                if const_expr(self.input_kind == "nvfp4"):
                    row_amax = _ld_f32(row_at + Int64(self.k // 2 + self.k // 16))
                else:
                    row_amax = _bf16_row_amax(row_at, self.k, self.threads, tidx, red)
                if tidx == Int32(0):
                    st_global_f32(Int64(amax.toint()) + i * Int64(4), row_amax)
                encoder, decoder, outer = _row_factors(row_amax)
                groups = self.k // 16
                for it in cutlass.range_constexpr(_ceil(groups, self.threads)):
                    grp = Int32(it * self.threads) + tidx
                    if grp < Int32(groups):
                        packed_at = Int64(aq.toint()) + i * Int64(self.k // 2) + Int64(grp) * Int64(8)
                        scale_at = Int64(a_s.toint()) + i * Int64(groups) + Int64(grp)
                        if const_expr(self.input_kind == "nvfp4"):
                            _copy_nvfp4_group(row_at, self.k, grp, packed_at, scale_at)
                        else:
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
                            _quantize_row16_dyn(values, encoder, decoder, outer, packed_at, scale_at)


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
        return ("stream_linear_nvfp4_dyn", 4, *super().key()[2:])

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream, row_amax: cute.Pointer = None):
        if cutlass.const_expr(row_amax is None):
            raise ValueError("the dynamic NVFP4 linear needs the activation rows' amax")
        self.kernel_dyn(aq, a_s, meta, w, s, y, row_amax).launch(
            grid=(_ceil(self.n, self.tile_n), max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
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
                # Vendor epilogue scale association: (row_amax * w_amax) * rcp_rn(1536 * denominator)
                # (vLLM d664cd1e/a0c1f162 ops/moe.py:347-353, activation.py:155-171; SGLang ec5e7ff5 nvfp4.py:256-257).
                w_amax = _ld_f32(Int64(s.toint()) + Int64(self.amax_offset) + Int64(e) * Int64(4))
                rcp_denom = div_rn_f32(
                    Float32(1.0),
                    Float32(1536.0) * _ld_f32(Int64(s.toint()) + Int64(self.denominator_offset)),
                )
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
                                factor = (_ld_f32(amax_t + Int64(row) * Int64(4)) * w_amax) * rcp_denom
                                for nt in cutlass.range_constexpr(self.n_tiles):
                                    f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                    col = n0 + warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                    if col < Int32(self.n):
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
        self.parts = _ceil(self.inter, self.half_n)

    def key(self) -> tuple:
        return ("stream_gate_up_nvfp4_dyn", 5, self.inter, self.k, self.experts, self.k_block, self.stages,
                self.gate_cap, self.up_cap, self.tile_m, self.tile_n, self.live_min, self.live_max)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                 w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer,
                 max_tiles: Int32, stream: cuda.CUstream, row_amax: cute.Pointer = None,
                 gamma: cute.Pointer = None, act_bf16: cute.Pointer = None, act_amax_part: cute.Pointer = None,
                 exchange: cute.Pointer = None, exchange_rows: cute.Pointer = None, pair_row: cute.Pointer = None):
        if cutlass.const_expr(row_amax is None or gamma is None or act_bf16 is None or act_amax_part is None):
            raise ValueError("the dynamic NVFP4 gate/up needs row_amax, gamma, act_bf16 and act_amax_part")
        # Diagnostics: exchange [padded, 2I] BF16 (gate then up columns, post factor, pre SiTU) and
        # exchange_rows [padded, 4] int32 = (tile, row in chunk, grouped position, source row).
        dump = exchange is not None
        if cutlass.const_expr(not dump):
            exchange = act_bf16
            exchange_rows = meta
            pair_row = meta
        self.fused_dyn(aq, a_s, meta, w1, s1, w3, s3, row_amax, gamma, act_bf16, act_amax_part,
                       exchange, exchange_rows, pair_row, dump).launch(
            grid=(_ceil(self.inter, self.half_n), max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.jit
    def _situ(self, gate: Float32, up: Float32, gamma: Float32) -> Float32:
        # The checkpoint's SiTU as its reference kernels evaluate it (approximate ex2/rcp/tanh): the
        # format's rule, so the oracle can be the same arithmetic rather than a tolerance.
        return situ_vendor(gate, up, gamma, gate_cap=self.gate_cap, up_cap=self.up_cap)

    @cute.kernel
    def fused_dyn(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                  w3: cute.Pointer, s3: cute.Pointer, row_amax: cute.Pointer, gamma: cute.Pointer,
                  act_bf16: cute.Pointer, act_amax_part: cute.Pointer, exchange: cute.Pointer,
                  exchange_rows: cute.Pointer, pair_row: cute.Pointer, dump: cutlass.Constexpr):
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
                w_amax = _ld_f32(sp + Int64(self.amax_offset) + Int64(e) * Int64(4))
                rcp_denom = div_rn_f32(
                    Float32(1.0),
                    Float32(1536.0) * _ld_f32(sp + Int64(self.denominator_offset)),
                )
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
                                factor = (_ld_f32(amax_t + Int64(row) * Int64(4)) * w_amax) * rcp_denom
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                col = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                st_shared_u32(ring + row * Int32(self.x_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
                cute.arch.sync_threads()
                if const_expr(dump):
                    # Each live row's gate/up columns of this CTA, plus (tile, row, grouped pos, source row).
                    words = self.half_n // 2
                    for it in cutlass.range_constexpr(_ceil(self.tile_m * words, self.threads)):
                        gi = tidx + Int32(it * self.threads)
                        row = gi // Int32(words)
                        wd = gi % Int32(words)
                        if (row < live) & (n0 + wd * Int32(2) < Int32(self.inter)):
                            i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(row)
                            gw = ld_shared_u32_at(ring + row * Int32(self.x_stride) + wd * Int32(4))
                            uw = ld_shared_u32_at(ring + row * Int32(self.x_stride) + Int32(self.half_n * 2) + wd * Int32(4))
                            base = Int64(exchange.toint()) + (i * Int64(2 * self.inter) + Int64(n0)) * Int64(2)
                            st_global_u32(base + Int64(wd) * Int64(4), gw)
                            st_global_u32(base + Int64(self.inter) * Int64(2) + Int64(wd) * Int64(4), uw)
                    if n_blk == Int32(0):
                        if tidx < live:
                            i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(tidx)
                            rec = cute.make_ptr(Int32, Int64(exchange_rows.toint()) + i * Int64(16), cute.AddressSpace.gmem,
                                                assumed_align=4)
                            rec[0] = Int32(tile)
                            rec[1] = tidx
                            rec[2] = first + tidx
                            rec[3] = _i32_at(Int64(pair_row.toint()), first + tidx)
                # SiTU x gamma per 16 columns -> BF16 act slice + the slice's amax per row.
                groups = self.half_n // 16
                gamma_t = Int64(gamma.toint())
                for it in cutlass.range_constexpr(_ceil(self.tile_m * groups, self.threads)):
                    gi = tidx + Int32(it * self.threads)
                    row = gi // Int32(groups)
                    grp = gi % Int32(groups)
                    if (row < live) & (n0 + grp * Int32(16) < Int32(self.inter)):
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
        if self.inter <= 0 or self.inter % 16 or self.parts <= 0:
            raise ValueError("I must be a positive multiple of 16; parts must be positive")
        # One partial per 16-column group of the row (the gate/up CTAs write parts * (half_n/16) of them).
        self.partials = self.inter // 16

    def key(self) -> tuple:
        return ("chunk_act_dyn", 3, self.inter, self.experts, self.parts)

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


# ---------------------------------------------------------------------------
# Small-M register-streaming kernels for Inkling 2 Small NVFP4 TP4 (K=4608, N=576)
# ---------------------------------------------------------------------------


@dsl_user_op
def _e2m1x8_to_bf16x2x4(word, *, loc=None, ip=None):
    """Exact BF16 pairs of eight packed E2M1 nibbles (low nibble first -> _bf16_lo)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b8 q0, q1, q2, q3; "
        "mov.b32 {q0, q1, q2, q3}, $4; "
        "cvt.rn.bf16x2.e2m1x2 $0, q0; "
        "cvt.rn.bf16x2.e2m1x2 $1, q1; "
        "cvt.rn.bf16x2.e2m1x2 $2, q2; "
        "cvt.rn.bf16x2.e2m1x2 $3, q3; }",
        "=r,=r,=r,=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _quad_transpose_x4(a0, a1, a2, a3, c, *, loc=None, ip=None):
    """4x4 u32 transpose across the 4 lanes c = 0..3 of each quad."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3)] + [Int32(c).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p1, p2; .reg .b32 s0, s1, t0, t1, b0, b1, b2, b3, s2, s3, t2, t3; "
        "and.b32 s0, $8, 1; setp.ne.u32 p1, s0, 0; "
        "and.b32 s1, $8, 2; setp.ne.u32 p2, s1, 0; "
        "selp.b32 s0, $4, $5, p1; selp.b32 s1, $6, $7, p1; "
        "shfl.sync.bfly.b32 t0, s0, 1, 0x1f, 0xffffffff; "
        "shfl.sync.bfly.b32 t1, s1, 1, 0x1f, 0xffffffff; "
        "selp.b32 b0, t0, $4, p1; selp.b32 b1, $5, t0, p1; "
        "selp.b32 b2, t1, $6, p1; selp.b32 b3, $7, t1, p1; "
        "selp.b32 s2, b0, b2, p2; selp.b32 s3, b1, b3, p2; "
        "shfl.sync.bfly.b32 t2, s2, 2, 0x1f, 0xffffffff; "
        "shfl.sync.bfly.b32 t3, s3, 2, 0x1f, 0xffffffff; "
        "selp.b32 $0, t2, b0, p2; selp.b32 $1, t3, b1, p2; "
        "selp.b32 $2, b2, t2, p2; selp.b32 $3, b3, t3, p2; }",
        "=r,=r,=r,=r,r,r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _quad_unpack_v2(r0, r1, lane, *, loc=None, ip=None):
    """Unpack 8 contiguous u32 words loaded as v2.u32 across c=0..3 into MMA (w[c], w[4+c])."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(r0).ir_value(loc=loc, ip=ip), Uint32(r1).ir_value(loc=loc, ip=ip),
         Int32(lane).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; .reg .b32 base, i0, i1, x0, x1, y0, y1; "
        "and.b32 base, $4, 28; "
        "shr.u32 i0, $4, 1; and.b32 i0, i0, 1; "
        "add.u32 i0, base, i0; add.u32 i1, i0, 2; "
        "shfl.sync.idx.b32 x0, $2, i0, 0x1f, 0xffffffff; "
        "shfl.sync.idx.b32 x1, $3, i0, 0x1f, 0xffffffff; "
        "shfl.sync.idx.b32 y0, $2, i1, 0x1f, 0xffffffff; "
        "shfl.sync.idx.b32 y1, $3, i1, 0x1f, 0xffffffff; "
        "and.b32 base, $4, 1; setp.ne.u32 p, base, 0; "
        "selp.b32 $0, x1, x0, p; selp.b32 $1, y1, y0, p; }",
        "=r,=r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _shfl_idx_u32(val, src_lane, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(val).ir_value(loc=loc, ip=ip), Int32(src_lane).ir_value(loc=loc, ip=ip)],
        "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
        "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_nc_v4_u32_pred(address: Int64, cond: Int32, *, loc=None, ip=None):
    """Predicated 16-byte global read-only load (zeroes outputs when cond == 0)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(cond).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.s32 p, $5, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; "
        "@p ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4]; }",
        "=r,=r,=r,=r,l,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)))


@dsl_user_op
def _ld_nc_v2_u32_pred(address: Int64, cond: Int32, *, loc=None, ip=None):
    """Predicated 8-byte global read-only load (zeroes outputs when cond == 0)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(cond).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.s32 p, $3, 0; "
        "mov.b32 $0, 0; mov.b32 $1, 0; "
        "@p ld.global.nc.v2.u32 {$0, $1}, [$2]; }",
        "=r,=r,l,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _ld_nc_u32_pred(address: Int64, cond: Int32, *, loc=None, ip=None) -> Uint32:
    """Predicated 4-byte global read-only load (zeroes output when cond == 0)."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(cond).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; "
        "mov.b32 $0, 0; "
        "@p ld.global.nc.u32 $0, [$1]; }",
        "=r,l,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def _candidate_fast(values: cute.Tensor, scale: Float32, decoder: Float32, outer: Float32) -> tuple:
    """Bit-exact ``_candidate`` using hardware ``cvt.rn.bf16x2.e2m1x2`` nibble decode."""
    inverse = fmin_f32(div_rn_f32(Float32(1.0), scale * decoder), Float32(FP32_MAX))
    q = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = values[i] * inverse
    lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    d0, d1, d2, d3 = _e2m1x8_to_bf16x2x4(lo)
    d4, d5, d6, d7 = _e2m1x8_to_bf16x2x4(hi)
    decoded = (d0, d1, d2, d3, d4, d5, d6, d7)
    err = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
    for p in cutlass.range_constexpr(8):
        err[2 * p] = _f32_asm(
            "{ .reg .f32 t; mul.rn.f32 t, $1, $2; mul.rn.f32 t, t, $3; sub.rn.f32 t, t, $4; abs.f32 $0, t; }",
            _bf16_lo(decoded[p]), scale, outer, values[2 * p])
        err[2 * p + 1] = _f32_asm(
            "{ .reg .f32 t; mul.rn.f32 t, $1, $2; mul.rn.f32 t, t, $3; sub.rn.f32 t, t, $4; abs.f32 $0, t; }",
            _bf16_hi(decoded[p]), scale, outer, values[2 * p + 1])
    for width in cutlass.range_constexpr(4):
        step = 1 << width
        for i in cutlass.range_constexpr(0, 16, 2 * step):
            err[i] = _f32_asm("add.rn.f32 $0, $1, $2;", err[i], err[i + step])
    return (Uint64(hi) << Uint64(32)) | Uint64(lo), err[0]


@cute.jit
def _quantize_row16_dyn_fast(values: cute.Tensor, bmax: Float32, encoder: Float32, decoder: Float32,
                             outer: Float32, packed_at: Int64, scale_at: Int64):
    """Bit-exact ``_quantize_row16_dyn`` reusing precomputed ``bmax`` and PTX E2M1 decode."""
    s6_u32 = cvt_f32_to_e4m3(fmin_f32(bmax * (encoder * Float32(1.0 / 6.0)), Float32(448.0)))
    s4_u32 = cvt_f32_to_e4m3(fmin_f32(bmax * (encoder * Float32(0.25)), Float32(448.0)))
    s6 = fp8_e4m3_to_f32(s6_u32)
    s4 = fp8_e4m3_to_f32(s4_u32)
    packed6, err6 = _candidate_fast(values, s6, decoder, outer)
    packed4, err4 = _candidate_fast(values, s4, decoder, outer)
    choose4 = err4 < err6
    zero = bmax == Float32(0.0)
    packed = Uint64(cutlass.select_(choose4, packed4, packed6))
    packed = Uint64(cutlass.select_(zero, Uint64(0), packed))
    scale = Uint32(cutlass.select_(choose4, s4_u32, s6_u32))
    scale = Uint32(cutlass.select_(zero, Uint32(0), scale))
    st_global_u64(packed_at, packed)
    st_global_u8(scale_at, cutlass.Uint8(scale & Uint32(0xFF)))


class PrepQuantSmallDyn:
    """Fused route prep + grouped gamma scatter + single-pass per-source-row input quantization
    for small-M Inkling 2 NVFP4 TP4 (``pairs <= 128``, ``K = 4608``, ``E = 384``).

    Grid ``(rows + 1, 1, 1)``, block ``(384, 1, 1)``: CTA 0 runs the 12-warp prefix-scan route
    prep and scatters ``gamma``; CTAs ``1..rows`` concurrently quantize unique input rows
    ``0..rows-1`` (288 active threads = ``K / 16``) into compact ``[rows, ...]`` buffers."""

    def __init__(self, *, k: int, experts: int, top_k: int, max_tiles: int, input_kind: str = "bf16"):
        self.k, self.experts, self.top_k, self.max_tiles = int(k), int(experts), int(top_k), int(max_tiles)
        if input_kind not in ("bf16", "nvfp4"):
            raise ValueError("dynamic NVFP4 input_kind must be bf16 or nvfp4")
        self.input_kind = input_kind
        self.row_bytes = self.k // 2 + self.k // 16 + 16 if input_kind == "nvfp4" else self.k * 2
        self.groups = self.k // 16
        if self.experts <= 0 or self.experts % 32 or self.experts > 1024:
            raise ValueError("PrepQuantSmallDyn requires positive 32-aligned experts <= 1024")
        if self.k <= 0 or self.k % 512 or self.groups > self.experts:
            raise ValueError("PrepQuantSmallDyn requires positive K % 512 == 0 and K / 16 <= experts")
        self.threads = self.experts
        self.warps = self.threads // 32
        self.quant_warps = self.groups // 32

    def key(self) -> tuple:
        return ("prep_quant_small_dyn", 2, self.k, self.experts, self.top_k, self.max_tiles) + (
            ("nvfp4_input",) if self.input_kind == "nvfp4" else ())

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "counts": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "cursor": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "w_rows": cute.struct.Align[cute.struct.MemRange[Int32, self.warps], 16],
            "w_flags": cute.struct.Align[cute.struct.MemRange[Int32, self.warps], 16],
            "wmax": cute.struct.Align[cute.struct.MemRange[Float32, self.quant_warps], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, src: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, meta: cute.Pointer,
                 pair_row: cute.Pointer, pair_pos: cute.Pointer, gamma: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, amax: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(src, ids, weights, meta, pair_row, pair_pos, gamma, aq, a_s, amax, rows).launch(
            grid=(rows + Int32(1), 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer, meta: cute.Pointer,
               pair_row: cute.Pointer, pair_pos: cute.Pointer, gamma: cute.Pointer,
               aq: cute.Pointer, a_s: cute.Pointer, amax: cute.Pointer, rows: Int32):
        cta = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if cta == Int32(0):
            counts = storage.counts.get_tensor(cute.make_layout((self.experts,)))
            cursor = storage.cursor.get_tensor(cute.make_layout((self.experts,)))
            w_rows = storage.w_rows.get_tensor(cute.make_layout((self.warps,)))
            w_flags = storage.w_flags.get_tensor(cute.make_layout((self.warps,)))
            counts_smem = shared_ptr_to_u32(storage.counts.data_ptr())
            cursor_smem = shared_ptr_to_u32(storage.cursor.data_ptr())
            e_count = Int32(self.experts)
            m = Int64(meta.toint())
            counts_at = m + Int64(4 * META_HEAD)
            offsets_at = counts_at + Int64(4 * self.experts)
            active_at = offsets_at + Int64(4 * self.experts)
            tiles_at = active_at + Int64(4 * self.experts)
            ids_at = Int64(ids.toint())
            rows_at = Int64(pair_row.toint())
            pos_at = Int64(pair_pos.toint())
            pairs = rows * Int32(self.top_k)
            counts[tidx] = Int32(0)
            if tidx < pairs:
                _set_i32(rows_at, tidx, Int32(-1))
            cute.arch.sync_threads()
            if tidx < pairs:
                e = _i32_at(ids_at, tidx)
                if (e >= Int32(0)) & (e < e_count):
                    _atomic_add_shared_i32(counts_smem + e * Int32(4), Int32(1))
            cute.arch.sync_threads()
            # Each expert has at most pairs <= 128 == STREAM_TILE_M rows, so tiles_e == flag.
            n = Int32(counts[tidx])
            flag = Int32(cutlass.select_(n > Int32(0), Int32(1), Int32(0)))
            scan_r = n
            scan_f = flag
            for stage in cutlass.range_constexpr(5):
                off = 1 << stage
                up_r = Int32(cute.arch.shuffle_sync_up(scan_r, offset=off, mask_and_clamp=0))
                up_f = Int32(cute.arch.shuffle_sync_up(scan_f, offset=off, mask_and_clamp=0))
                if lane >= Int32(off):
                    scan_r = scan_r + up_r
                    scan_f = scan_f + up_f
            if lane == Int32(31):
                w_rows[warp_id] = scan_r
                w_flags[warp_id] = scan_f
            cute.arch.sync_threads()
            if warp_id == Int32(0):
                wr = Int32(0)
                wf = Int32(0)
                if lane < Int32(self.warps):
                    wr = Int32(w_rows[lane])
                    wf = Int32(w_flags[lane])
                for stage in cutlass.range_constexpr(5):
                    off = 1 << stage
                    if const_expr(off < self.warps):
                        up_r = Int32(cute.arch.shuffle_sync_up(wr, offset=off, mask_and_clamp=0))
                        up_f = Int32(cute.arch.shuffle_sync_up(wf, offset=off, mask_and_clamp=0))
                        if lane >= Int32(off):
                            wr = wr + up_r
                            wf = wf + up_f
                if lane < Int32(self.warps):
                    w_rows[lane] = wr
                    w_flags[lane] = wf
            cute.arch.sync_threads()
            base_r = Int32(0)
            base_f = Int32(0)
            if warp_id > Int32(0):
                base_r = Int32(w_rows[warp_id - Int32(1)])
                base_f = Int32(w_flags[warp_id - Int32(1)])
            end_r = base_r + scan_r
            end_f = base_f + scan_f
            offset = end_r - n
            active = end_f - flag
            _set_i32(counts_at, tidx, n)
            _set_i32(offsets_at, tidx, offset)
            cursor[tidx] = offset
            if n > Int32(0):
                _set_i32(active_at, active, tidx)
                if active < Int32(self.max_tiles):
                    _set_i32(tiles_at, active, tidx)
            if tidx == e_count - Int32(1):
                _set_i32(m, 0, end_f)
                _set_i32(m, 1, end_f)
                _set_i32(m, 2, end_r)
            cute.arch.sync_threads()
            if tidx < pairs:
                e = _i32_at(ids_at, tidx)
                position = Int32(-1)
                if (e >= Int32(0)) & (e < e_count):
                    position = _atomic_add_shared_i32(cursor_smem + e * Int32(4), Int32(1))
                    _set_i32(rows_at, position, tidx // Int32(self.top_k))
                    w_val = _ld_f32(Int64(weights.toint()) + Int64(tidx) * Int64(4))
                    st_global_f32(Int64(gamma.toint()) + Int64(position) * Int64(4), w_val)
                _set_i32(pos_at, tidx, position)
        else:
            row = cta - Int32(1)
            row_at = Int64(src.toint()) + Int64(row) * Int64(self.row_bytes)
            if const_expr(self.input_kind == "nvfp4"):
                if tidx == Int32(0):
                    st_global_f32(Int64(amax.toint()) + Int64(row) * Int64(4),
                                  _ld_f32(row_at + Int64(self.k // 2 + self.k // 16)))
                if tidx < Int32(self.groups):
                    packed_at = Int64(aq.toint()) + Int64(row) * Int64(self.k // 2) + Int64(tidx) * Int64(8)
                    scale_at = Int64(a_s.toint()) + Int64(row) * Int64(self.groups) + Int64(tidx)
                    _copy_nvfp4_group(row_at, self.k, tidx, packed_at, scale_at)
            else:
                wmax = storage.wmax.get_tensor(cute.make_layout((self.quant_warps,)))
                values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                bmax = Float32(0.0)
                if tidx < Int32(self.groups):
                    at = row_at + Int64(tidx) * Int64(32)
                    w0, w1, w2, w3 = ld_global_nc_v4_u32(at)
                    w4, w5, w6, w7 = ld_global_nc_v4_u32(at + Int64(16))
                    words = (w0, w1, w2, w3, w4, w5, w6, w7)
                    for q in cutlass.range_constexpr(8):
                        values[2 * q] = _bf16_lo(words[q])
                        values[2 * q + 1] = _bf16_hi(words[q])
                    bmax = max_abs_16(values)
                local = bmax
                for shift in cutlass.range_constexpr(5):
                    local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=1 << shift)))
                if (warp_id < Int32(self.quant_warps)) & (lane == Int32(0)):
                    wmax[warp_id] = local
                cute.arch.sync_threads()
                if warp_id == Int32(0):
                    v = Float32(0.0)
                    if lane < Int32(self.quant_warps):
                        v = Float32(wmax[lane])
                    for shift in cutlass.range_constexpr(5):
                        if const_expr((1 << shift) < self.quant_warps):
                            v = fmax_f32(v, Float32(cute.arch.shuffle_sync_bfly(v, offset=1 << shift)))
                    if lane == Int32(0):
                        wmax[0] = v
                        st_global_f32(Int64(amax.toint()) + Int64(row) * Int64(4), v)
                cute.arch.sync_threads()
                if tidx < Int32(self.groups):
                    row_amax = Float32(wmax[0])
                    encoder, decoder, outer = _row_factors(row_amax)
                    packed_at = Int64(aq.toint()) + Int64(row) * Int64(self.k // 2) + Int64(tidx) * Int64(8)
                    scale_at = Int64(a_s.toint()) + Int64(row) * Int64(self.groups) + Int64(tidx)
                    _quantize_row16_dyn_fast(values, bmax, encoder, decoder, outer, packed_at, scale_at)


class GateUpSmallDyn:
    """2D-TMA weight-streaming W4A4 gate/up + SiTU x gamma for small-M Inkling 2 NVFP4 TP4
    (``hidden=4608, inter=576``).

    Grid ``(inter // 16, grid_tiles)``, 4 warps (128 threads): warps 0..1 own 16 gate rows
    (``w1``), warps 2..3 own the matching 16 up rows (``w3``). Streams 128-byte-swizzled
    ``(16, 128)`` packed weight boxes (K=256) through a 3-stage TMA ring (13,312 B SMEM ->
    7 CTAs/SM = 28 warps/SM), loads weight fragments via conflict-free ``ldmatrix_m8n8x4_b16``,
    loads ``s1``/``s3`` scales via naturally 4-byte aligned ``ld_global_nc_u32_l2hint``
    (preserving the 4-byte scale ABI), and reads unique L1-cached input rows via ``pair_row``.
    Reuses the drained weight ring for the 1,152 B BF16 gate/up epilogue exchange."""

    threads = 128
    half_n = 16
    k_bytes = 128      # K = 256 packed E2M1 bytes per TMA step
    w_ring = 3
    x_stride = 72      # 32 BF16 (16 gate + 16 up) = 64 B + 8 B padding (conflict-free)

    def __init__(self, *, inter: int, hidden: int, experts: int, gate_cap: float = 4.0,
                 up_cap: float = 25.0, max_live: int = 16):
        self.inter, self.hidden, self.experts = int(inter), int(hidden), int(experts)
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        self.max_live = int(max_live)
        if self.hidden <= 0 or self.hidden % 256 or self.inter <= 0 or self.inter % self.half_n:
            raise ValueError("GateUpSmallDyn requires H % 256 == 0 and I % 16 == 0")
        self.row_bytes = self.hidden // 2
        self.scale_bytes = self.hidden // 16
        self.k_steps = self.row_bytes // self.k_bytes
        self.half_bytes = self.half_n * self.k_bytes
        if self.k_steps < self.w_ring:
            raise ValueError("GateUpSmallDyn needs at least w_ring K steps")
        self.amax_offset = nvfp4_amax_offset(experts, self.inter, self.hidden)
        self.denominator_offset = nvfp4_denominator_offset(experts, self.inter, self.hidden)

    def key(self) -> tuple:
        return ("gate_up_small_dyn", 5, self.inter, self.hidden, self.experts, self.w_ring,
                self.gate_cap, self.up_cap, self.max_live)

    def _w_layout(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.k_bytes),
            cutlass.Uint8)
        return cute.tile_to_shape(atom, (self.half_n, self.k_bytes, 2 * self.w_ring), order=(0, 1, 2))

    def _storage(self, w_layout):
        class Storage:
            pass

        # Place 1024B-aligned w (12,288 B) before mbar (24 B) to avoid 1,000 B of alignment padding
        # (12,312 B vs 13,312 B -> +1 resident CTA/SM under both 64 KiB and 100 KiB SMEM carveouts).
        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
                 w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                 row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
                 act_amax_part: cute.Pointer, grid_tiles: Int32, stream: cuda.CUstream):
        w_layout = self._w_layout()
        rows_w = self.experts * self.inter
        box = cute.slice_(w_layout, (None, None, 0))
        tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w1, rows_w, self.row_bytes), box,
            (self.half_n, self.k_bytes), num_multicast=1)
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), _u8_weight(w3, rows_w, self.row_bytes), box,
            (self.half_n, self.k_bytes), num_multicast=1)
        self.kernel(aq, a_s, pair_row, meta, tma_tensor_g, s1, tma_tensor_u, s3, row_amax, gamma,
                    act_bf16, act_amax_part, tma_g, tma_u, w_layout, self._storage(w_layout)).launch(
            grid=(self.inter // self.half_n, grid_tiles, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _load_w(self, step, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile):
        slot = Int32(step) % Int32(self.w_ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * self.half_bytes)
        cute.copy(tma_g, t_wg[(None, w_tile, Int32(step))], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
        cute.copy(tma_u, t_wu[(None, w_tile, Int32(step))], t_ws[(None, slot * Int32(2) + Int32(1))],
                  tma_bar_ptr=barrier)

    @cute.jit
    def _mma_tma_k128(self, acc: cute.Tensor, w_base: Int32, nrow: Int32, mat: Int32, mrow: Int32,
                      chunk_base: cutlass.Constexpr, a_lo_ptr: Int64, a_hi_ptr: Int64, k_byte: Int64,
                      live_lo: Int32, live_hi: Int32, has_hi: Int32, sfa_lo_w: Uint32, sfa_hi_w: Uint32,
                      sfb_w: Uint32, lane0: Int32, lane1: Int32, c: Int32):
        chunk = Int32(chunk_base) + mat
        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_base + nrow * Int32(self.k_bytes) + ((chunk ^ mrow) * Int32(16)))
        al0, al1, al2, al3 = _ld_nc_v4_u32_pred(a_lo_ptr + k_byte + Int64(c) * Int64(16), live_lo)
        al0, al1, al2, al3 = _quad_transpose_x4(al0, al1, al2, al3, c)
        ah0, ah1, ah2, ah3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                th0, th1, th2, th3 = _ld_nc_v4_u32_pred(a_hi_ptr + k_byte + Int64(c) * Int64(16), live_hi)
                th0, th1, th2, th3 = _quad_transpose_x4(th0, th1, th2, th3, c)
                ah0 = th0
                ah1 = th1
                ah2 = th2
                ah3 = th3
        sfa0 = _shfl_idx_u32(sfa_lo_w, lane0)
        sfa1 = _shfl_idx_u32(sfa_lo_w, lane1)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                sfa0_h = _shfl_idx_u32(sfa_hi_w, lane0)
                sfa1_h = _shfl_idx_u32(sfa_hi_w, lane1)
                sfa0 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa0_h, sfa0))
                sfa1 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa1_h, sfa1))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            acc[0], acc[1], acc[2], acc[3], al0, ah0, al1, ah1, b0, b1, sfa0, _shfl_idx_u32(sfb_w, lane0))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            d0, d1, d2, d3, al2, ah2, al3, ah3, b2, b3, sfa1, _shfl_idx_u32(sfb_w, lane1))
        acc[0], acc[1], acc[2], acc[3] = d0, d1, d2, d3

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
               weight_g: cute.Tensor, s1: cute.Pointer, weight_u: cute.Tensor, s3: cute.Pointer,
               row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
               act_amax_part: cute.Pointer, tma_g: cute.CopyAtom, tma_u: cute.CopyAtom,
               w_layout: cute.ComposedLayout, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            q = lane // Int32(4)
            c = lane % Int32(4)
            mat = lane // Int32(8)
            mrow = lane % Int32(8)
            quad_base = q * Int32(4)
            side = warp_id // Int32(2)
            sub_n = warp_id % Int32(2)
            nrow = sub_n * Int32(8) + mrow
            n0 = n_blk * Int32(self.half_n)
            mbar = storage.mbar.data_ptr()
            s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            w_tile = e * Int32(self.inter // self.half_n) + n_blk
            t_ws, t_wg = cpasync.tma_partition(
                tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_g, (self.half_n, self.k_bytes), (None, None)), 0, 2))
            _, t_wu = cpasync.tma_partition(
                tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
                cute.group_modes(cute.local_tile(weight_u, (self.half_n, self.k_bytes), (None, None)), 0, 2))
            if tidx == Int32(0):
                cpasync.prefetch_descriptor(tma_g)
                cpasync.prefetch_descriptor(tma_u)
            sp = cutlass.select_(side == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
            w_row_idx = Int64(e) * Int64(self.inter) + Int64(n0 + sub_n * Int32(8) + q)
            s_row = sp + w_row_idx * Int64(self.scale_bytes)
            w_amax = _ld_f32(sp + Int64(self.amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(
                Float32(1.0),
                Float32(1536.0) * _ld_f32(sp + Int64(self.denominator_offset)),
            )
            policy = create_l2_evict_first_policy()
            m_chunks = (live + Int32(15)) // Int32(16)
            for m_chunk in cutlass.range(m_chunks, unroll=1):
                if tidx == Int32(0):
                    for init_slot in cutlass.range_constexpr(self.w_ring):
                        cute.arch.mbarrier_init(mbar + init_slot, 1)
                    cute.arch.mbarrier_init_fence()
                cute.arch.sync_threads()
                if warp_id == Int32(0):
                    for pre in cutlass.range_constexpr(self.w_ring - 1):
                        self._load_w(pre, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                m0 = m_chunk * Int32(16)
                chunk_live = live - m0
                if chunk_live > Int32(16):
                    chunk_live = Int32(16)
                has_hi = Int32(cutlass.select_(chunk_live > Int32(8), Int32(1), Int32(0)))
                live_lo = Int32(cutlass.select_(q < chunk_live, Int32(1), Int32(0)))
                live_hi = Int32(cutlass.select_(q + Int32(8) < chunk_live, Int32(1), Int32(0)))
                src_lo = Int32(0)
                if q < chunk_live:
                    src_lo = _i32_at(Int64(pair_row.toint()), first + m0 + q)
                src_hi = Int32(0)
                if const_expr(self.max_live > 8):
                    if q + Int32(8) < chunk_live:
                        src_hi = _i32_at(Int64(pair_row.toint()), first + m0 + q + Int32(8))
                a_lo_ptr = Int64(aq.toint()) + Int64(src_lo) * Int64(self.row_bytes)
                as_lo_ptr = Int64(a_s.toint()) + Int64(src_lo) * Int64(self.scale_bytes)
                a_hi_ptr = Int64(aq.toint()) + Int64(src_hi) * Int64(self.row_bytes)
                as_hi_ptr = Int64(a_s.toint()) + Int64(src_hi) * Int64(self.scale_bytes)

                acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(4):
                    acc[v] = Float32(0.0)

                # Prefetch step 0 scales while TMA prologue is in flight (4-byte aligned, s1/s3 ABI align=4).
                sfb_cur = ld_global_nc_u32_l2hint(s_row + Int64(c) * Int64(4), policy)

                for step in cutlass.range(self.k_steps, unroll=1):
                    slot = step % Int32(self.w_ring)
                    cute.arch.mbarrier_wait(mbar + slot, (step // Int32(self.w_ring)) & Int32(1))
                    if (step > Int32(0)) & (step + Int32(self.w_ring - 1) < Int32(self.k_steps)):
                        cute.arch.sync_threads()
                    if warp_id == Int32(0):
                        if step + Int32(self.w_ring - 1) < Int32(self.k_steps):
                            self._load_w(step + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, w_tile)
                    # Prefetch next step's 4-byte aligned weight scale word across the current step's MMAs.
                    sfb_next = sfb_cur
                    if step + Int32(1) < Int32(self.k_steps):
                        sfb_next = ld_global_nc_u32_l2hint(
                            s_row + Int64(step + Int32(1)) * Int64(16) + Int64(c) * Int64(4), policy)
                    s_off = Int64(step) * Int64(16) + Int64(c) * Int64(4)
                    sfa_lo_w = _ld_nc_u32_pred(as_lo_ptr + s_off, live_lo)
                    sfa_hi_w = Uint32(0)
                    if const_expr(self.max_live > 8):
                        if has_hi != Int32(0):
                            sfa_hi_w = _ld_nc_u32_pred(as_hi_ptr + s_off, live_hi)
                    w_base = sw + (slot * Int32(2) + side) * Int32(self.half_bytes)
                    k_byte = Int64(step) * Int64(self.k_bytes)
                    self._mma_tma_k128(acc, w_base, nrow, mat, mrow, 0, a_lo_ptr, a_hi_ptr, k_byte,
                                       live_lo, live_hi, has_hi, sfa_lo_w, sfa_hi_w, sfb_cur,
                                       quad_base, quad_base + Int32(1), c)
                    self._mma_tma_k128(acc, w_base, nrow, mat, mrow, 4, a_lo_ptr, a_hi_ptr, k_byte + Int64(64),
                                       live_lo, live_hi, has_hi, sfa_lo_w, sfa_hi_w, sfb_cur,
                                       quad_base + Int32(2), quad_base + Int32(3), c)
                    sfb_cur = sfb_next

                # Reuse drained TMA weight ring for the 1,152 B BF16 gate/up exchange.
                cute.arch.sync_threads()
                col_xch = side * Int32(16) + sub_n * Int32(8) + Int32(2) * c
                if q < chunk_live:
                    factor_lo = (_ld_f32(Int64(row_amax.toint()) + Int64(src_lo) * Int64(4)) * w_amax) * rcp_denom
                    st_shared_u32(sw + q * Int32(self.x_stride) + col_xch * Int32(2),
                                  pack_f32x2_to_bfloat2(acc[0] * factor_lo, acc[1] * factor_lo))
                if const_expr(self.max_live > 8):
                    if q + Int32(8) < chunk_live:
                        factor_hi = (_ld_f32(Int64(row_amax.toint()) + Int64(src_hi) * Int64(4)) * w_amax) * rcp_denom
                        st_shared_u32(sw + (q + Int32(8)) * Int32(self.x_stride) + col_xch * Int32(2),
                                      pack_f32x2_to_bfloat2(acc[2] * factor_hi, acc[3] * factor_hi))
                cute.arch.sync_threads()
                erow = tidx // Int32(8)
                epair = tidx % Int32(8)
                local = Float32(0.0)
                if erow < chunk_live:
                    p = first + m0 + erow
                    g_row = _ld_f32(Int64(gamma.toint()) + Int64(p) * Int64(4))
                    gw = ld_shared_u32_at(sw + erow * Int32(self.x_stride) + epair * Int32(4))
                    uw = ld_shared_u32_at(sw + erow * Int32(self.x_stride) + Int32(self.half_n * 2) + epair * Int32(4))
                    v0 = situ_vendor(_bf16_lo(gw), _bf16_lo(uw), g_row, gate_cap=self.gate_cap, up_cap=self.up_cap)
                    v1 = situ_vendor(_bf16_hi(gw), _bf16_hi(uw), g_row, gate_cap=self.gate_cap, up_cap=self.up_cap)
                    local = fmax_f32(fabs_f32(v0), fabs_f32(v1))
                    out_at = Int64(act_bf16.toint()) + (Int64(p) * Int64(self.inter) + Int64(n0) + Int64(epair) * Int64(2)) * Int64(2)
                    st_global_u32(out_at, pack_f32x2_to_bfloat2(v0, v1))
                # Warp-convergent 8-lane reduction over this CTA's 16 columns.
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=1)))
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=2)))
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=4)))
                if (erow < chunk_live) & (epair == Int32(0)):
                    p = first + m0 + erow
                    st_global_f32(Int64(act_amax_part.toint())
                                  + (Int64(p) * Int64(self.inter // 16) + Int64(n_blk)) * Int64(4), local)
                if m_chunks > Int32(1):
                    cute.arch.sync_threads()


class ChunkActSmallDyn:
    """Compact per-pair dynamic NVFP4 quantizer of ``act_bf16 [pairs, inter]`` (for ``inter=576``):
    one CTA per live grouped pair (zero empty chunk-padding CTAs), 64 threads (2 warps),
    parallel 36-partial amax reduction and PTX E2M1 quantization into compact ``[pairs, ...]``."""

    threads = 64

    def __init__(self, *, inter: int):
        self.inter = int(inter)
        self.groups = self.inter // 16
        if self.inter <= 0 or self.inter % 16 or self.groups > self.threads:
            raise ValueError("ChunkActSmallDyn requires positive I % 16 == 0 and I / 16 <= 64")

    def key(self) -> tuple:
        return ("chunk_act_small_dyn", 1, self.inter)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "wmax": cute.struct.Align[cute.struct.MemRange[Float32, 2], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, act_bf16: cute.Pointer, act_amax_part: cute.Pointer, meta: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, amax: cute.Pointer, pairs: Int32,
                 stream: cuda.CUstream):
        self.kernel(act_bf16, act_amax_part, meta, aq, a_s, amax).launch(
            grid=(pairs, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, act_bf16: cute.Pointer, act_amax_part: cute.Pointer, meta: cute.Pointer,
               aq: cute.Pointer, a_s: cute.Pointer, amax: cute.Pointer):
        p = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        wmax = storage.wmax.get_tensor(cute.make_layout((2,)))
        if p < _i32_at(Int64(meta.toint()), 2):
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            bmax = Float32(0.0)
            if tidx < Int32(self.groups):
                bmax = _ld_f32(Int64(act_amax_part.toint())
                               + (Int64(p) * Int64(self.groups) + Int64(tidx)) * Int64(4))
            local = bmax
            for shift in cutlass.range_constexpr(5):
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=1 << shift)))
            if lane == Int32(0):
                wmax[warp_id] = local
            cute.arch.sync_threads()
            row_amax = fmax_f32(Float32(wmax[0]), Float32(wmax[1]))
            if tidx == Int32(0):
                st_global_f32(Int64(amax.toint()) + Int64(p) * Int64(4), row_amax)
            if tidx < Int32(self.groups):
                encoder, decoder, outer = _row_factors(row_amax)
                at = Int64(act_bf16.toint()) + (Int64(p) * Int64(self.inter) + Int64(tidx) * Int64(16)) * Int64(2)
                w0, w1, w2, w3 = ld_global_nc_v4_u32(at)
                w4, w5, w6, w7 = ld_global_nc_v4_u32(at + Int64(16))
                words = (w0, w1, w2, w3, w4, w5, w6, w7)
                values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                for q in cutlass.range_constexpr(8):
                    values[2 * q] = _bf16_lo(words[q])
                    values[2 * q + 1] = _bf16_hi(words[q])
                _quantize_row16_dyn_fast(
                    values, bmax, encoder, decoder, outer,
                    Int64(aq.toint()) + Int64(p) * Int64(self.inter // 2) + Int64(tidx) * Int64(8),
                    Int64(a_s.toint()) + Int64(p) * Int64(self.groups) + Int64(tidx))


class DownSmallDyn:
    """1D-bulk-TMA W4A4 down projection for small-M Inkling 2 NVFP4 TP4 (``n=4608, k=576``).

    Grid ``(n // 32, grid_tiles)`` = ``(144, grid_tiles)`` (exact multiple of 48 SMs for every
    expert count), 4 warps (128 threads, 8 output columns per warp). Each CTA issues one
    contiguous 9,216 B ``cp_async_bulk_g2s_mbar`` transfer of its 32x288 B packed ``w2`` tile
    into SMEM (9,344 B -> 10 CTAs/SM = 40 warps/SM), overlaps all 9 naturally 4-byte aligned
    ``s2`` scale loads with TMA flight, reads weight fragments via ``ldmatrix_m8n8x4_b16``
    (plus two SMEM loads for the K64 tail), and writes coalesced BF16 ``y [pairs, n]``."""

    threads = 128
    tile_n = 32

    def __init__(self, *, n: int, k: int, experts: int, max_live: int = 16):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.max_live = int(max_live)
        if self.n <= 0 or self.n % self.tile_n or self.k != 576:
            raise ValueError("DownSmallDyn requires N % 32 == 0 and K == 576")
        self.row_bytes = self.k // 2       # 288 B
        self.scale_bytes = self.k // 16    # 36 B (9 u32 words)
        self.tile_bytes = self.tile_n * self.row_bytes  # 9,216 B (multiple of 128 B)
        self.amax_offset = nvfp4_amax_offset(experts, self.n, self.k)
        self.denominator_offset = nvfp4_denominator_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("down_small_dyn", 5, self.n, self.k, self.experts, self.tile_n, self.max_live)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.tile_bytes], 128],
            "mbar": cute.struct.MemRange[cutlass.Int64, 1],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
                 s: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer, grid_tiles: Int32,
                 stream: cuda.CUstream):
        self.kernel(aq, a_s, meta, w, s, y, row_amax).launch(
            grid=(self.n // self.tile_n, grid_tiles, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _mma_smem_k128(self, acc: cute.Tensor, w_ldm_row: Int32, a_lo_ptr: Int64, a_hi_ptr: Int64,
                       k_byte: cutlass.Constexpr, live_lo: Int32, live_hi: Int32, has_hi: Int32,
                       sfa_lo_w: Uint32, sfa_hi_w: Uint32, sfb_w: Uint32, lane0: Int32, lane1: Int32,
                       c: Int32):
        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(w_ldm_row + Int32(k_byte))
        al0, al1, al2, al3 = _ld_nc_v4_u32_pred(a_lo_ptr + Int64(k_byte) + Int64(c) * Int64(16), live_lo)
        al0, al1, al2, al3 = _quad_transpose_x4(al0, al1, al2, al3, c)
        ah0, ah1, ah2, ah3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                th0, th1, th2, th3 = _ld_nc_v4_u32_pred(a_hi_ptr + Int64(k_byte) + Int64(c) * Int64(16), live_hi)
                th0, th1, th2, th3 = _quad_transpose_x4(th0, th1, th2, th3, c)
                ah0 = th0
                ah1 = th1
                ah2 = th2
                ah3 = th3
        sfa_0 = _shfl_idx_u32(sfa_lo_w, lane0)
        sfa_1 = _shfl_idx_u32(sfa_lo_w, lane1)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                sfa_0h = _shfl_idx_u32(sfa_hi_w, lane0)
                sfa_1h = _shfl_idx_u32(sfa_hi_w, lane1)
                sfa_0 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa_0h, sfa_0))
                sfa_1 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa_1h, sfa_1))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            acc[0], acc[1], acc[2], acc[3], al0, ah0, al1, ah1, b0, b1, sfa_0, _shfl_idx_u32(sfb_w, lane0))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            d0, d1, d2, d3, al2, ah2, al3, ah3, b2, b3, sfa_1, _shfl_idx_u32(sfb_w, lane1))
        acc[0], acc[1], acc[2], acc[3] = d0, d1, d2, d3

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
               s: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            q = lane // Int32(4)
            c = lane % Int32(4)
            mat = lane // Int32(8)
            mrow = lane % Int32(8)
            quad_base = q * Int32(4)
            mbar = storage.mbar.data_ptr()
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            if tidx == Int32(0):
                cute.arch.mbarrier_init(mbar, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            w_tile_addr = Int64(w.toint()) + (Int64(e) * Int64(self.n) + Int64(n_blk) * Int64(self.tile_n)) * Int64(self.row_bytes)
            if tidx == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, self.tile_bytes)
                cp_async_bulk_g2s_mbar(sw, w_tile_addr, Int32(self.tile_bytes), shared_ptr_to_u32(mbar))

            # Overlap naturally 4-byte aligned s2 scale preloads (ABI align=4) with bulk TMA flight.
            n0 = n_blk * Int32(self.tile_n) + warp_id * Int32(8)
            w_idx = Int64(e) * Int64(self.n) + Int64(n0 + q)
            s_row = Int64(s.toint()) + w_idx * Int64(self.scale_bytes)
            policy = create_l2_evict_first_policy()
            sfb0 = ld_global_nc_u32_l2hint(s_row + Int64(c) * Int64(4), policy)
            sfb1 = ld_global_nc_u32_l2hint(s_row + Int64(16) + Int64(c) * Int64(4), policy)
            sfb2 = ld_global_nc_u32_l2hint(s_row + Int64(32), policy)
            w_amax = _ld_f32(Int64(s.toint()) + Int64(self.amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(
                Float32(1.0),
                Float32(1536.0) * _ld_f32(Int64(s.toint()) + Int64(self.denominator_offset)),
            )

            w_ldm_row = sw + (warp_id * Int32(8) + mrow) * Int32(self.row_bytes) + mat * Int32(16)
            w_smem_row = sw + (warp_id * Int32(8) + q) * Int32(self.row_bytes) + c * Int32(4)

            m_chunks = (live + Int32(15)) // Int32(16)
            for m_chunk in cutlass.range(m_chunks, unroll=1):
                m0 = m_chunk * Int32(16)
                chunk_live = live - m0
                if chunk_live > Int32(16):
                    chunk_live = Int32(16)
                has_hi = Int32(cutlass.select_(chunk_live > Int32(8), Int32(1), Int32(0)))
                live_lo = Int32(cutlass.select_(q < chunk_live, Int32(1), Int32(0)))
                live_hi = Int32(cutlass.select_(q + Int32(8) < chunk_live, Int32(1), Int32(0)))
                p_lo = Int32(cutlass.select_(q < chunk_live, first + m0 + q, first))
                p_hi = Int32(cutlass.select_(q + Int32(8) < chunk_live, first + m0 + q + Int32(8), first))
                a_lo_ptr = Int64(aq.toint()) + Int64(p_lo) * Int64(self.row_bytes)
                as_lo_ptr = Int64(a_s.toint()) + Int64(p_lo) * Int64(self.scale_bytes)
                a_hi_ptr = Int64(aq.toint()) + Int64(p_hi) * Int64(self.row_bytes)
                as_hi_ptr = Int64(a_s.toint()) + Int64(p_hi) * Int64(self.scale_bytes)

                sfa0_lo = _ld_nc_u32_pred(as_lo_ptr + Int64(c) * Int64(4), live_lo)
                sfa1_lo = _ld_nc_u32_pred(as_lo_ptr + Int64(16) + Int64(c) * Int64(4), live_lo)
                sfa2_lo = _ld_nc_u32_pred(as_lo_ptr + Int64(32), live_lo)
                sfa0_hi, sfa1_hi, sfa2_hi = Uint32(0), Uint32(0), Uint32(0)
                if const_expr(self.max_live > 8):
                    if has_hi != Int32(0):
                        sfa0_hi = _ld_nc_u32_pred(as_hi_ptr + Int64(c) * Int64(4), live_hi)
                        sfa1_hi = _ld_nc_u32_pred(as_hi_ptr + Int64(16) + Int64(c) * Int64(4), live_hi)
                        sfa2_hi = _ld_nc_u32_pred(as_hi_ptr + Int64(32), live_hi)

                acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(4):
                    acc[v] = Float32(0.0)

                if m_chunk == Int32(0):
                    cute.arch.mbarrier_wait(mbar, 0)

                # Unrolled K = 576 = 4 * 128 + 64 directly from TMA-loaded SMEM tile.
                self._mma_smem_k128(acc, w_ldm_row, a_lo_ptr, a_hi_ptr, 0, live_lo, live_hi, has_hi,
                                    sfa0_lo, sfa0_hi, sfb0, quad_base, quad_base + Int32(1), c)
                self._mma_smem_k128(acc, w_ldm_row, a_lo_ptr, a_hi_ptr, 64, live_lo, live_hi, has_hi,
                                    sfa0_lo, sfa0_hi, sfb0, quad_base + Int32(2), quad_base + Int32(3), c)
                self._mma_smem_k128(acc, w_ldm_row, a_lo_ptr, a_hi_ptr, 128, live_lo, live_hi, has_hi,
                                    sfa1_lo, sfa1_hi, sfb1, quad_base, quad_base + Int32(1), c)
                self._mma_smem_k128(acc, w_ldm_row, a_lo_ptr, a_hi_ptr, 192, live_lo, live_hi, has_hi,
                                    sfa1_lo, sfa1_hi, sfb1, quad_base + Int32(2), quad_base + Int32(3), c)

                # Step 8 (K = 512..575).
                al0, al1 = _ld_nc_v2_u32_pred(a_lo_ptr + Int64(256) + Int64(c) * Int64(8), live_lo)
                al0, al1 = _quad_unpack_v2(al0, al1, lane)
                ah0, ah1 = Uint32(0), Uint32(0)
                if const_expr(self.max_live > 8):
                    if has_hi != Int32(0):
                        th0, th1 = _ld_nc_v2_u32_pred(a_hi_ptr + Int64(256) + Int64(c) * Int64(8), live_hi)
                        th0, th1 = _quad_unpack_v2(th0, th1, lane)
                        ah0 = th0
                        ah1 = th1
                sfa_t = sfa2_lo
                if const_expr(self.max_live > 8):
                    if has_hi != Int32(0):
                        sfa_t = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa2_hi, sfa2_lo))
                b0 = ld_shared_u32_at(w_smem_row + Int32(256))
                b1 = ld_shared_u32_at(w_smem_row + Int32(272))
                d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                    acc[0], acc[1], acc[2], acc[3], al0, ah0, al1, ah1, b0, b1, sfa_t, sfb2)

                # Epilogue: coalesced BF16 stores directly to y [pairs, n].
                if q < chunk_live:
                    factor_lo = (_ld_f32(Int64(row_amax.toint()) + Int64(p_lo) * Int64(4)) * w_amax) * rcp_denom
                    y_lo = Int64(y.toint()) + (Int64(p_lo) * Int64(self.n) + Int64(n0) + Int64(2) * Int64(c)) * Int64(2)
                    st_global_u32(y_lo, pack_f32x2_to_bfloat2(d0 * factor_lo, d1 * factor_lo))
                if const_expr(self.max_live > 8):
                    if q + Int32(8) < chunk_live:
                        factor_hi = (_ld_f32(Int64(row_amax.toint()) + Int64(p_hi) * Int64(4)) * w_amax) * rcp_denom
                        y_hi = Int64(y.toint()) + (Int64(p_hi) * Int64(self.n) + Int64(n0) + Int64(2) * Int64(c)) * Int64(2)
                        st_global_u32(y_hi, pack_f32x2_to_bfloat2(d2 * factor_hi, d3 * factor_hi))


class CombineSmallDyn:
    """Multi-CTA unweighted route combine for small-M Inkling 2 (``hidden=4608, top_k=8``):
    grid ``(hidden // 256, rows)``, 32 threads (1 warp) per CTA, 8 BF16 per thread,
    preserving exact slot-order FP32 summation."""

    threads = 32

    def __init__(self, *, hidden: int, top_k: int):
        self.hidden, self.top_k = int(hidden), int(top_k)
        if self.hidden <= 0 or self.hidden % (self.threads * 8):
            raise ValueError("CombineSmallDyn requires hidden % 256 == 0")
        self.col_blocks = self.hidden // (self.threads * 8)

    def key(self) -> tuple:
        return ("combine_small_dyn", 1, self.hidden, self.top_k)

    @cute.jit
    def __call__(self, y: cute.Pointer, pair_pos: cute.Pointer, out: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(y, pair_pos, out).launch(
            grid=(self.col_blocks, rows, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, y: cute.Pointer, pair_pos: cute.Pointer, out: cute.Pointer):
        col_blk = Int32(cute.arch.block_idx()[0])
        row = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        v = col_blk * Int32(self.threads) + tidx
        acc = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
        for j in cutlass.range_constexpr(8):
            acc[j] = Float32(0.0)
        pos_at = Int64(pair_pos.toint())
        y_at = Int64(y.toint())
        for k in cutlass.range_constexpr(self.top_k):
            p = _i32_at(pos_at, row * Int32(self.top_k) + Int32(k))
            if p >= Int32(0):
                words = ld_global_v4_u32(y_at + (Int64(p) * Int64(self.hidden) + Int64(v) * Int64(8)) * Int64(2))
                for j in cutlass.range_constexpr(4):
                    acc[2 * j] = acc[2 * j] + _bf16_lo(words[j])
                    acc[2 * j + 1] = acc[2 * j + 1] + _bf16_hi(words[j])
        st_global_v4_u32(
            Int64(out.toint()) + (Int64(row) * Int64(self.hidden) + Int64(v) * Int64(8)) * Int64(2),
            pack_f32x2_to_bfloat2(acc[0], acc[1]), pack_f32x2_to_bfloat2(acc[2], acc[3]),
            pack_f32x2_to_bfloat2(acc[4], acc[5]), pack_f32x2_to_bfloat2(acc[6], acc[7]))


class GateUpSmallDynReg:
    """Thin-wave zero-barrier register-streaming W4A4 gate/up + SiTU x gamma for small-M
    Inkling 2 NVFP4 TP4 (``hidden=4608, inter=576``, ``rows <= 3``).

    Grid ``(inter // 16, grid_tiles)``, 4 warps (128 threads): warps 0..1 own 16 gate rows
    (``w1``), warps 2..3 own the matching 16 up rows (``w3``). Uses 2-deep K=256 (128 B/row)
    register ping-pong with ``L1::no_allocate`` + ``L2::evict_first`` weight loads, naturally
    4-byte aligned ``ld_global_nc_u32_l2hint`` loads on ``s1``/``s3`` (preserving the 4-byte
    scale ABI), on-demand L1-cached ``a_s`` reads, zero K-loop barriers, and 1,152 B SMEM for
    the BF16 gate/up epilogue exchange."""

    threads = 128
    half_n = 16
    x_stride = 72      # 32 BF16 (16 gate + 16 up) = 64 B + 8 B padding (conflict-free)

    def __init__(self, *, inter: int, hidden: int, experts: int, gate_cap: float = 4.0,
                 up_cap: float = 25.0, max_live: int = 16):
        self.inter, self.hidden, self.experts = int(inter), int(hidden), int(experts)
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)
        self.max_live = int(max_live)
        if self.hidden <= 0 or self.hidden % 512 or self.inter <= 0 or self.inter % self.half_n:
            raise ValueError("GateUpSmallDynReg requires H % 512 == 0 and I % 16 == 0")
        self.blocks512 = self.hidden // 512
        self.row_bytes = self.hidden // 2
        self.scale_bytes = self.hidden // 16
        self.amax_offset = nvfp4_amax_offset(experts, self.inter, self.hidden)
        self.denominator_offset = nvfp4_denominator_offset(experts, self.inter, self.hidden)

    def key(self) -> tuple:
        return ("gate_up_small_dyn_reg", 3, self.inter, self.hidden, self.experts,
                self.gate_cap, self.up_cap, self.max_live)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "xch": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 16 * self.x_stride], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
                 w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                 row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
                 act_amax_part: cute.Pointer, grid_tiles: Int32, stream: cuda.CUstream):
        self.kernel(aq, a_s, pair_row, meta, w1, s1, w3, s3, row_amax, gamma, act_bf16,
                    act_amax_part).launch(
            grid=(self.inter // self.half_n, grid_tiles, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _mma_k128(self, acc: cute.Tensor, w_buf: cute.Tensor, w_off: cutlass.Constexpr,
                  a_lo_ptr: Int64, a_hi_ptr: Int64, k_byte: Int64, live_lo: Int32, live_hi: Int32,
                  has_hi: Int32, sfa_lo_w: Uint32, sfa_hi_w: Uint32, sfb_w: Uint32,
                  lane0: Int32, lane1: Int32, c: Int32):
        al0, al1, al2, al3 = _ld_nc_v4_u32_pred(a_lo_ptr + k_byte + Int64(c) * Int64(16), live_lo)
        al0, al1, al2, al3 = _quad_transpose_x4(al0, al1, al2, al3, c)
        ah0, ah1, ah2, ah3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                th0, th1, th2, th3 = _ld_nc_v4_u32_pred(a_hi_ptr + k_byte + Int64(c) * Int64(16), live_hi)
                th0, th1, th2, th3 = _quad_transpose_x4(th0, th1, th2, th3, c)
                ah0 = th0
                ah1 = th1
                ah2 = th2
                ah3 = th3
        b0, b1, b2, b3 = _quad_transpose_x4(
            w_buf[w_off], w_buf[w_off + 1], w_buf[w_off + 2], w_buf[w_off + 3], c)
        sfa0 = _shfl_idx_u32(sfa_lo_w, lane0)
        sfa1 = _shfl_idx_u32(sfa_lo_w, lane1)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                sfa0_h = _shfl_idx_u32(sfa_hi_w, lane0)
                sfa1_h = _shfl_idx_u32(sfa_hi_w, lane1)
                sfa0 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa0_h, sfa0))
                sfa1 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa1_h, sfa1))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            acc[0], acc[1], acc[2], acc[3], al0, ah0, al1, ah1, b0, b1, sfa0, _shfl_idx_u32(sfb_w, lane0))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            d0, d1, d2, d3, al2, ah2, al3, ah3, b2, b3, sfa1, _shfl_idx_u32(sfb_w, lane1))
        acc[0], acc[1], acc[2], acc[3] = d0, d1, d2, d3

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer,
               w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
               row_amax: cute.Pointer, gamma: cute.Pointer, act_bf16: cute.Pointer,
               act_amax_part: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            q = lane // Int32(4)
            c = lane % Int32(4)
            quad_base = q * Int32(4)
            side = warp_id // Int32(2)
            sub_n = warp_id % Int32(2)
            n0 = n_blk * Int32(self.half_n)
            wp = cutlass.select_(side == Int32(0), Int64(w1.toint()), Int64(w3.toint()))
            sp = cutlass.select_(side == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
            w_row_idx = Int64(e) * Int64(self.inter) + Int64(n0 + sub_n * Int32(8) + q)
            w_row = wp + w_row_idx * Int64(self.row_bytes)
            s_row = sp + w_row_idx * Int64(self.scale_bytes)
            w_amax = _ld_f32(sp + Int64(self.amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(
                Float32(1.0),
                Float32(1536.0) * _ld_f32(sp + Int64(self.denominator_offset)),
            )
            xch = shared_ptr_to_u32(storage.xch.data_ptr())
            policy = create_l2_evict_first_policy()

            m_chunks = (live + Int32(15)) // Int32(16)
            for m_chunk in cutlass.range(m_chunks, unroll=1):
                m0 = m_chunk * Int32(16)
                chunk_live = live - m0
                if chunk_live > Int32(16):
                    chunk_live = Int32(16)
                has_hi = Int32(cutlass.select_(chunk_live > Int32(8), Int32(1), Int32(0)))
                live_lo = Int32(cutlass.select_(q < chunk_live, Int32(1), Int32(0)))
                live_hi = Int32(cutlass.select_(q + Int32(8) < chunk_live, Int32(1), Int32(0)))
                src_lo = Int32(0)
                if q < chunk_live:
                    src_lo = _i32_at(Int64(pair_row.toint()), first + m0 + q)
                src_hi = Int32(0)
                if const_expr(self.max_live > 8):
                    if q + Int32(8) < chunk_live:
                        src_hi = _i32_at(Int64(pair_row.toint()), first + m0 + q + Int32(8))
                a_lo_ptr = Int64(aq.toint()) + Int64(src_lo) * Int64(self.row_bytes)
                as_lo_ptr = Int64(a_s.toint()) + Int64(src_lo) * Int64(self.scale_bytes)
                a_hi_ptr = Int64(aq.toint()) + Int64(src_hi) * Int64(self.row_bytes)
                as_hi_ptr = Int64(a_s.toint()) + Int64(src_hi) * Int64(self.scale_bytes)

                acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(4):
                    acc[v] = Float32(0.0)
                w0 = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)
                w1_buf = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Uint32)

                # Prologue: prefetch blk512=0 Half 0 (K=0..255, 128 B) and naturally 4-byte aligned s1/s3 scales.
                p0, p1, p2, p3 = ld_global_nc_v4_u32_l2hint(w_row + Int64(c) * Int64(16), policy, no_allocate=True)
                p4, p5, p6, p7 = ld_global_nc_v4_u32_l2hint(w_row + Int64(64) + Int64(c) * Int64(16), policy, no_allocate=True)
                w0[0], w0[1], w0[2], w0[3] = p0, p1, p2, p3
                w0[4], w0[5], w0[6], w0[7] = p4, p5, p6, p7
                sfb0 = ld_global_nc_u32_l2hint(s_row + Int64(c) * Int64(4), policy)
                sfb1 = ld_global_nc_u32_l2hint(s_row + Int64(16) + Int64(c) * Int64(4), policy)

                for blk512 in cutlass.range(self.blocks512, unroll=1):
                    k_base = Int64(blk512) * Int64(256)
                    s_base = Int64(blk512) * Int64(32)
                    # Prefetch Half 1 (K +256..+511, 128 B) while computing Half 0.
                    r0, r1, r2, r3 = ld_global_nc_v4_u32_l2hint(
                        w_row + k_base + Int64(128) + Int64(c) * Int64(16), policy, no_allocate=True)
                    r4, r5, r6, r7 = ld_global_nc_v4_u32_l2hint(
                        w_row + k_base + Int64(192) + Int64(c) * Int64(16), policy, no_allocate=True)
                    w1_buf[0], w1_buf[1], w1_buf[2], w1_buf[3] = r0, r1, r2, r3
                    w1_buf[4], w1_buf[5], w1_buf[6], w1_buf[7] = r4, r5, r6, r7

                    sfa_lo_0 = _ld_nc_u32_pred(as_lo_ptr + s_base + Int64(c) * Int64(4), live_lo)
                    sfa_hi_0 = Uint32(0)
                    if const_expr(self.max_live > 8):
                        if has_hi != Int32(0):
                            sfa_hi_0 = _ld_nc_u32_pred(as_hi_ptr + s_base + Int64(c) * Int64(4), live_hi)
                    self._mma_k128(acc, w0, 0, a_lo_ptr, a_hi_ptr, k_base, live_lo, live_hi,
                                   has_hi, sfa_lo_0, sfa_hi_0, sfb0, quad_base, quad_base + Int32(1), c)
                    self._mma_k128(acc, w0, 4, a_lo_ptr, a_hi_ptr, k_base + Int64(64), live_lo, live_hi,
                                   has_hi, sfa_lo_0, sfa_hi_0, sfb0, quad_base + Int32(2), quad_base + Int32(3), c)

                    # Prefetch next blk512 Half 0 weights and 4-byte aligned s1/s3 scales.
                    sfb0_nxt = sfb0
                    sfb1_nxt = sfb1
                    if blk512 + Int32(1) < Int32(self.blocks512):
                        nk = Int64(blk512 + Int32(1)) * Int64(256)
                        q0, q1, q2, q3 = ld_global_nc_v4_u32_l2hint(
                            w_row + nk + Int64(c) * Int64(16), policy, no_allocate=True)
                        q4, q5, q6, q7 = ld_global_nc_v4_u32_l2hint(
                            w_row + nk + Int64(64) + Int64(c) * Int64(16), policy, no_allocate=True)
                        w0[0], w0[1], w0[2], w0[3] = q0, q1, q2, q3
                        w0[4], w0[5], w0[6], w0[7] = q4, q5, q6, q7
                        ns = Int64(blk512 + Int32(1)) * Int64(32)
                        sfb0_nxt = ld_global_nc_u32_l2hint(s_row + ns + Int64(c) * Int64(4), policy)
                        sfb1_nxt = ld_global_nc_u32_l2hint(s_row + ns + Int64(16) + Int64(c) * Int64(4), policy)

                    sfa_lo_1 = _ld_nc_u32_pred(as_lo_ptr + s_base + Int64(16) + Int64(c) * Int64(4), live_lo)
                    sfa_hi_1 = Uint32(0)
                    if const_expr(self.max_live > 8):
                        if has_hi != Int32(0):
                            sfa_hi_1 = _ld_nc_u32_pred(as_hi_ptr + s_base + Int64(16) + Int64(c) * Int64(4), live_hi)
                    self._mma_k128(acc, w1_buf, 0, a_lo_ptr, a_hi_ptr, k_base + Int64(128), live_lo, live_hi,
                                   has_hi, sfa_lo_1, sfa_hi_1, sfb1, quad_base, quad_base + Int32(1), c)
                    self._mma_k128(acc, w1_buf, 4, a_lo_ptr, a_hi_ptr, k_base + Int64(192), live_lo, live_hi,
                                   has_hi, sfa_lo_1, sfa_hi_1, sfb1, quad_base + Int32(2), quad_base + Int32(3), c)
                    sfb0 = sfb0_nxt
                    sfb1 = sfb1_nxt

                # Epilogue: exchange BF16 gate/up via 1,152 B SMEM, compute SiTU x gamma and group amax.
                col_xch = side * Int32(16) + sub_n * Int32(8) + Int32(2) * c
                if q < chunk_live:
                    factor_lo = (_ld_f32(Int64(row_amax.toint()) + Int64(src_lo) * Int64(4)) * w_amax) * rcp_denom
                    st_shared_u32(xch + q * Int32(self.x_stride) + col_xch * Int32(2),
                                  pack_f32x2_to_bfloat2(acc[0] * factor_lo, acc[1] * factor_lo))
                if const_expr(self.max_live > 8):
                    if q + Int32(8) < chunk_live:
                        factor_hi = (_ld_f32(Int64(row_amax.toint()) + Int64(src_hi) * Int64(4)) * w_amax) * rcp_denom
                        st_shared_u32(xch + (q + Int32(8)) * Int32(self.x_stride) + col_xch * Int32(2),
                                      pack_f32x2_to_bfloat2(acc[2] * factor_hi, acc[3] * factor_hi))
                cute.arch.sync_threads()
                erow = tidx // Int32(8)
                epair = tidx % Int32(8)
                local = Float32(0.0)
                if erow < chunk_live:
                    p = first + m0 + erow
                    g_row = _ld_f32(Int64(gamma.toint()) + Int64(p) * Int64(4))
                    gw = ld_shared_u32_at(xch + erow * Int32(self.x_stride) + epair * Int32(4))
                    uw = ld_shared_u32_at(xch + erow * Int32(self.x_stride) + Int32(self.half_n * 2) + epair * Int32(4))
                    v0 = situ_vendor(_bf16_lo(gw), _bf16_lo(uw), g_row, gate_cap=self.gate_cap, up_cap=self.up_cap)
                    v1 = situ_vendor(_bf16_hi(gw), _bf16_hi(uw), g_row, gate_cap=self.gate_cap, up_cap=self.up_cap)
                    local = fmax_f32(fabs_f32(v0), fabs_f32(v1))
                    out_at = Int64(act_bf16.toint()) + (Int64(p) * Int64(self.inter) + Int64(n0) + Int64(epair) * Int64(2)) * Int64(2)
                    st_global_u32(out_at, pack_f32x2_to_bfloat2(v0, v1))
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=1)))
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=2)))
                local = fmax_f32(local, Float32(cute.arch.shuffle_sync_bfly(local, offset=4)))
                if (erow < chunk_live) & (epair == Int32(0)):
                    p = first + m0 + erow
                    st_global_f32(Int64(act_amax_part.toint())
                                  + (Int64(p) * Int64(self.inter // 16) + Int64(n_blk)) * Int64(4), local)
                if m_chunks > Int32(1):
                    cute.arch.sync_threads()


class DownSmallDynReg:
    """Thin-wave zero-SMEM, zero-barrier register-streaming W4A4 down projection for small-M
    Inkling 2 NVFP4 TP4 (``n=4608, k=576``, ``rows <= 3``).

    Grid ``(n // 32, grid_tiles)`` = ``(144, grid_tiles)`` (exact multiple of 48 SMs), 4 warps
    (128 threads, 1 group = 8 output columns per warp for half the register pressure of G=2).
    Preloads all 9 naturally 4-byte aligned ``s2`` scale words with ``L2::evict_first``, streams
    ``w2`` via unrolled 2-deep K=128 ping-pong (``L1::no_allocate`` + ``L2::evict_first``) plus
    the 8B-aligned K=64 tail on 16B-aligned ``w2``, and writes coalesced BF16 ``y [pairs, n]``."""

    threads = 128
    tile_n = 32

    def __init__(self, *, n: int, k: int, experts: int, max_live: int = 16):
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.max_live = int(max_live)
        if self.n <= 0 or self.n % self.tile_n or self.k != 576:
            raise ValueError("DownSmallDynReg requires N % 32 == 0 and K == 576")
        self.row_bytes = self.k // 2       # 288 B
        self.scale_bytes = self.k // 16    # 36 B (9 u32 words)
        self.amax_offset = nvfp4_amax_offset(experts, self.n, self.k)
        self.denominator_offset = nvfp4_denominator_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("down_small_dyn_reg", 3, self.n, self.k, self.experts, self.tile_n, self.max_live)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
                 s: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer, grid_tiles: Int32,
                 stream: cuda.CUstream):
        self.kernel(aq, a_s, meta, w, s, y, row_amax).launch(
            grid=(self.n // self.tile_n, grid_tiles, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.jit
    def _load_w128(self, buf: cute.Tensor, w_row: Int64, k_byte: cutlass.Constexpr, c: Int32, policy: Uint64):
        r0, r1, r2, r3 = ld_global_nc_v4_u32_l2hint(
            w_row + Int64(k_byte) + Int64(c) * Int64(16), policy, no_allocate=True)
        buf[0], buf[1], buf[2], buf[3] = r0, r1, r2, r3

    @cute.jit
    def _mma_k128(self, acc: cute.Tensor, w_buf: cute.Tensor, a_lo_ptr: Int64, a_hi_ptr: Int64,
                  k_byte: cutlass.Constexpr, live_lo: Int32, live_hi: Int32, has_hi: Int32,
                  sfa_lo_w: Uint32, sfa_hi_w: Uint32, sfb_w: Uint32, lane0: Int32, lane1: Int32,
                  c: Int32):
        al0, al1, al2, al3 = _ld_nc_v4_u32_pred(a_lo_ptr + Int64(k_byte) + Int64(c) * Int64(16), live_lo)
        al0, al1, al2, al3 = _quad_transpose_x4(al0, al1, al2, al3, c)
        ah0, ah1, ah2, ah3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                th0, th1, th2, th3 = _ld_nc_v4_u32_pred(a_hi_ptr + Int64(k_byte) + Int64(c) * Int64(16), live_hi)
                th0, th1, th2, th3 = _quad_transpose_x4(th0, th1, th2, th3, c)
                ah0 = th0
                ah1 = th1
                ah2 = th2
                ah3 = th3
        sfa_0 = _shfl_idx_u32(sfa_lo_w, lane0)
        sfa_1 = _shfl_idx_u32(sfa_lo_w, lane1)
        if const_expr(self.max_live > 8):
            if has_hi != Int32(0):
                sfa_0h = _shfl_idx_u32(sfa_hi_w, lane0)
                sfa_1h = _shfl_idx_u32(sfa_hi_w, lane1)
                sfa_0 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa_0h, sfa_0))
                sfa_1 = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa_1h, sfa_1))
        b0, b1, b2, b3 = _quad_transpose_x4(w_buf[0], w_buf[1], w_buf[2], w_buf[3], c)
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            acc[0], acc[1], acc[2], acc[3], al0, ah0, al1, ah1, b0, b1, sfa_0, _shfl_idx_u32(sfb_w, lane0))
        d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
            d0, d1, d2, d3, al2, ah2, al3, ah3, b2, b3, sfa_1, _shfl_idx_u32(sfb_w, lane1))
        acc[0], acc[1], acc[2], acc[3] = d0, d1, d2, d3

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer,
               s: cute.Pointer, y: cute.Pointer, row_amax: cute.Pointer):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            q = lane // Int32(4)
            c = lane % Int32(4)
            quad_base = q * Int32(4)
            n0 = n_blk * Int32(self.tile_n) + warp_id * Int32(8)
            w_idx = Int64(e) * Int64(self.n) + Int64(n0 + q)
            w_row = Int64(w.toint()) + w_idx * Int64(self.row_bytes)
            s_row = Int64(s.toint()) + w_idx * Int64(self.scale_bytes)
            w_amax = _ld_f32(Int64(s.toint()) + Int64(self.amax_offset) + Int64(e) * Int64(4))
            rcp_denom = div_rn_f32(
                Float32(1.0),
                Float32(1536.0) * _ld_f32(Int64(s.toint()) + Int64(self.denominator_offset)),
            )
            policy = create_l2_evict_first_policy()

            # Preload all 9 naturally 4-byte aligned s2 scale words (ABI align=4).
            sfb0 = ld_global_nc_u32_l2hint(s_row + Int64(c) * Int64(4), policy)
            sfb1 = ld_global_nc_u32_l2hint(s_row + Int64(16) + Int64(c) * Int64(4), policy)
            sfb2 = ld_global_nc_u32_l2hint(s_row + Int64(32), policy)

            m_chunks = (live + Int32(15)) // Int32(16)
            for m_chunk in cutlass.range(m_chunks, unroll=1):
                m0 = m_chunk * Int32(16)
                chunk_live = live - m0
                if chunk_live > Int32(16):
                    chunk_live = Int32(16)
                has_hi = Int32(cutlass.select_(chunk_live > Int32(8), Int32(1), Int32(0)))
                live_lo = Int32(cutlass.select_(q < chunk_live, Int32(1), Int32(0)))
                live_hi = Int32(cutlass.select_(q + Int32(8) < chunk_live, Int32(1), Int32(0)))
                p_lo = Int32(cutlass.select_(q < chunk_live, first + m0 + q, first))
                p_hi = Int32(cutlass.select_(q + Int32(8) < chunk_live, first + m0 + q + Int32(8), first))
                a_lo_ptr = Int64(aq.toint()) + Int64(p_lo) * Int64(self.row_bytes)
                as_lo_ptr = Int64(a_s.toint()) + Int64(p_lo) * Int64(self.scale_bytes)
                a_hi_ptr = Int64(aq.toint()) + Int64(p_hi) * Int64(self.row_bytes)
                as_hi_ptr = Int64(a_s.toint()) + Int64(p_hi) * Int64(self.scale_bytes)

                sfa0_lo = _ld_nc_u32_pred(as_lo_ptr + Int64(c) * Int64(4), live_lo)
                sfa1_lo = _ld_nc_u32_pred(as_lo_ptr + Int64(16) + Int64(c) * Int64(4), live_lo)
                sfa2_lo = _ld_nc_u32_pred(as_lo_ptr + Int64(32), live_lo)
                sfa0_hi, sfa1_hi, sfa2_hi = Uint32(0), Uint32(0), Uint32(0)
                if const_expr(self.max_live > 8):
                    if has_hi != Int32(0):
                        sfa0_hi = _ld_nc_u32_pred(as_hi_ptr + Int64(c) * Int64(4), live_hi)
                        sfa1_hi = _ld_nc_u32_pred(as_hi_ptr + Int64(16) + Int64(c) * Int64(4), live_hi)
                        sfa2_hi = _ld_nc_u32_pred(as_hi_ptr + Int64(32), live_hi)

                acc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Float32)
                for v in cutlass.range_constexpr(4):
                    acc[v] = Float32(0.0)
                w0 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                w1_buf = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)

                # Unrolled 2-deep ping-pong over K = 576 = 4 * 128 + 64.
                self._load_w128(w0, w_row, 0, c, policy)

                self._load_w128(w1_buf, w_row, 64, c, policy)
                self._mma_k128(acc, w0, a_lo_ptr, a_hi_ptr, 0, live_lo, live_hi, has_hi,
                               sfa0_lo, sfa0_hi, sfb0, quad_base, quad_base + Int32(1), c)

                self._load_w128(w0, w_row, 128, c, policy)
                self._mma_k128(acc, w1_buf, a_lo_ptr, a_hi_ptr, 64, live_lo, live_hi, has_hi,
                               sfa0_lo, sfa0_hi, sfb0, quad_base + Int32(2), quad_base + Int32(3), c)

                self._load_w128(w1_buf, w_row, 192, c, policy)
                self._mma_k128(acc, w0, a_lo_ptr, a_hi_ptr, 128, live_lo, live_hi, has_hi,
                               sfa1_lo, sfa1_hi, sfb1, quad_base, quad_base + Int32(1), c)

                # Prefetch final K64 tail (32 B at 16B-aligned offset 256 on align=16 w2) before block 3.
                t0, t1 = ld_global_nc_v2_u32_l2hint(w_row + Int64(256) + Int64(c) * Int64(8), policy)
                self._mma_k128(acc, w1_buf, a_lo_ptr, a_hi_ptr, 192, live_lo, live_hi, has_hi,
                               sfa1_lo, sfa1_hi, sfb1, quad_base + Int32(2), quad_base + Int32(3), c)

                # Step 8 (K = 512..575).
                al0, al1 = _ld_nc_v2_u32_pred(a_lo_ptr + Int64(256) + Int64(c) * Int64(8), live_lo)
                al0, al1 = _quad_unpack_v2(al0, al1, lane)
                ah0, ah1 = Uint32(0), Uint32(0)
                if const_expr(self.max_live > 8):
                    if has_hi != Int32(0):
                        th0, th1 = _ld_nc_v2_u32_pred(a_hi_ptr + Int64(256) + Int64(c) * Int64(8), live_hi)
                        th0, th1 = _quad_unpack_v2(th0, th1, lane)
                        ah0 = th0
                        ah1 = th1
                sfa_t = sfa2_lo
                if const_expr(self.max_live > 8):
                    if has_hi != Int32(0):
                        sfa_t = Uint32(cutlass.select_((c & Int32(1)) != Int32(0), sfa2_hi, sfa2_lo))
                b0, b1 = _quad_unpack_v2(t0, t1, lane)
                d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                    acc[0], acc[1], acc[2], acc[3], al0, ah0, al1, ah1, b0, b1, sfa_t, sfb2)

                # Epilogue: coalesced BF16 stores directly to y [pairs, n].
                if q < chunk_live:
                    factor_lo = (_ld_f32(Int64(row_amax.toint()) + Int64(p_lo) * Int64(4)) * w_amax) * rcp_denom
                    y_lo = Int64(y.toint()) + (Int64(p_lo) * Int64(self.n) + Int64(n0) + Int64(2) * Int64(c)) * Int64(2)
                    st_global_u32(y_lo, pack_f32x2_to_bfloat2(d0 * factor_lo, d1 * factor_lo))
                if const_expr(self.max_live > 8):
                    if q + Int32(8) < chunk_live:
                        factor_hi = (_ld_f32(Int64(row_amax.toint()) + Int64(p_hi) * Int64(4)) * w_amax) * rcp_denom
                        y_hi = Int64(y.toint()) + (Int64(p_hi) * Int64(self.n) + Int64(n0) + Int64(2) * Int64(c)) * Int64(2)
                        st_global_u32(y_hi, pack_f32x2_to_bfloat2(d2 * factor_hi, d3 * factor_hi))


__all__ += ["ChunkActDyn", "PrepQuantSmallDyn", "GateUpSmallDyn", "GateUpSmallDynReg",
            "ChunkActSmallDyn", "DownSmallDyn", "DownSmallDynReg", "CombineSmallDyn"]
