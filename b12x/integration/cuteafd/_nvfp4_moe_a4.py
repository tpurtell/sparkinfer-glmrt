"""W4A4 kernels of the NVFP4 stream route (``fp8_moe`` ``weights="nvfp4"``,
``activations="a4"``): the numerics NVIDIA calibrated its ModelOpt NVFP4
checkpoints for. Compressed-tensors globals instead remain checkpoint-native
divisors: the input global directly encodes activations, and projection
accumulators divide by the weight and input globals without a reciprocal round trip.

Activations are quantized to NVFP4 with the checkpoint's static
``input_scale`` (one per projection and expert, stored after the experts'
alphas in each scale operand): per 16 values an E4M3 block scale
``e4m3(amax * gs / 6)`` with ``gs = 1 / input_scale``, values rounded to E2M1
(nearest, ties to even) after division by ``scale / gs``
(``quantize_block_fp4``, TensorRT-LLM / ModelOpt semantics). The GEMMs run the
block-scaled FP4 MMA ``kind::mxf4nvf4.scale_vec::4X.m16n8k64`` over packed
activations and packed weights with both operands' E4M3 scales, and multiply
the FP32 accumulator by ``alpha_e * input_scale_e`` before the one BF16
rounding.

``ChunkRowsA4``       chunk-padded packed activations ``[max_tiles * 128, K/2]``
    and E4M3 scales ``[max_tiles * 128, K/16]`` from BF16 or FP8 K32 wire
    rows (each chunk with its expert's gate ``input_scale``).
``ChunkSwiGLUA4``     ``act = bf16(bf16(silu(g)) * u)`` (clamped as the BF16
    route) quantized the same way with the expert's down ``input_scale``.
``StreamNvfp4LinearA4``  ``y[p] = alpha_e * in_e * (a_p . w_e^T)`` over the
    packed operands: a CTA owns either a sparse 16-row tile with 32/64/128
    output columns, or a wide 128-row tile with 128 columns. Both stream
    256/128/64-K operand blocks and scales through a two-stage
    ``cp.async`` ring with conflict-free ``ldmatrix`` loads. Native K16 tails
    are zero-filled only in shared memory; partial output tiles are masked. Complementary
    live-count masks select exactly one tile per compact expert chunk.
``StreamNvfp4GateUpA4``  gate/up, SwiGLU and the down-input quantization in
    one launch: half the CTA's weight rows are gate and half are matching
    up rows, both BF16-rounded exactly as the linear's output, exchanged
    through shared memory and written as ``ChunkSwiGLUA4`` writes them
    (bit-identical ``act`` bytes, no BF16 ``[pairs, 2I]`` round trip).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64, const_expr

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    div_rn_f32,
    ld_shared_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    max_abs_16,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp4,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_u64,
    st_global_u8,
    st_shared_u32,
    div_rn_f32,
)

from b12x._lib.activation import gated_activation_value

from ._fp8_moe_kernels import _post_silu_product, _ceil, _i32_at, _ld_u8, _ld_v2_u32, _round_bf16, _u32_as_f32
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from ._nvfp4_moe_kernels import nvfp4_alpha_offset

__all__ = ["ChunkRowsA4", "ChunkSwiGLUA4", "StreamNvfp4GateUpA4", "StreamNvfp4LinearA4", "nvfp4_input_scale_offset"]


def nvfp4_input_scale_offset(experts: int, rows: int, k: int) -> int:
    """Byte offset of the FP32 input scales in an NVFP4 scale operand (after the alphas)."""
    return nvfp4_alpha_offset(experts, rows, k) + 4 * int(experts)


@cute.jit
def _quantize_row16(values: cute.Tensor, gs: Float32, packed_at: Int64, scale_at: Int64):
    packed, scale = quantize_block_fp4(values, max_abs_16(values), gs)
    st_global_u64(packed_at, packed)
    st_global_u8(scale_at, scale)


@cute.jit
def _input_global_scale(raw: Float32, divisor: cutlass.Constexpr):
    if const_expr(divisor):
        gs = raw
    else:
        gs = Float32(1.0) / raw
    return gs


@cute.jit
def _copy_nvfp4_group(row: Int64, k: cutlass.Constexpr, grp: Int32, packed_at: Int64, scale_at: Int64):
    lo, hi = _ld_v2_u32(row + Int64(grp) * Int64(8))
    st_global_u64(packed_at, (Uint64(hi) << Uint64(32)) | Uint64(lo))
    st_global_u8(scale_at, _ld_u8(row + Int64(k // 2) + Int64(grp)))


class ChunkRowsA4:
    """Chunk-padded NVFP4 activations of the chunk rows (see module docstring);
    one CTA per live grouped pair, one thread per 16-value group."""

    threads = 128

    def __init__(self, *, k: int, wire: bool, experts: int, scale_rows: int, scale_convention: str = "multiplier",
                 input_kind: str | None = None, row_addresses: bool = False):
        self.divisor = scale_convention == "divisor"
        self.row_addresses = bool(row_addresses)
        self.input_kind = input_kind or ("fp8" if wire else "bf16")
        if self.input_kind not in ("bf16", "fp8", "nvfp4"):
            raise ValueError("input_kind must be bf16, fp8 or nvfp4")
        self.k, self.wire, self.experts = int(k), self.input_kind == "fp8", int(experts)
        self.row_bytes = (self.k // 2 + self.k // 16 + 16 if self.input_kind == "nvfp4"
                          else self.k + self.k // 32 if self.wire else self.k * 2)
        # The scale operand whose input scales quantize these rows (gate: rows I, K = H).
        self.in_offset = nvfp4_input_scale_offset(experts, scale_rows, k)
        if self.k <= 0 or self.k % 16 or (self.wire and self.k % 32):
            raise ValueError("K must be positive and K16-aligned (K32 for FP8 wire)")

    def key(self) -> tuple:
        return (("chunk_rows_a4", 2, self.k, self.wire, self.experts, self.in_offset)
                + (("divisor",) if self.divisor else ())
                + (("row_addresses",) if self.row_addresses else ())
                + (("nvfp4_input",) if self.input_kind == "nvfp4" else ()))

    @cute.jit
    def row_base(self, src: cute.Pointer, row: Int32) -> Int64:
        """Byte address of an input row, orthogonal to its wire representation."""
        if const_expr(self.row_addresses):
            return cute.make_ptr(Int64, Int64(src.toint()), cute.AddressSpace.gmem, assumed_align=8)[row]
        else:
            return Int64(src.toint()) + Int64(row) * Int64(self.row_bytes)

    @cute.jit
    def __call__(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, pairs: Int32, stream: cuda.CUstream):
        self.kernel(src, pair_row, meta, scales, aq, a_s).launch(
            grid=(pairs, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Pointer, pair_row: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
               aq: cute.Pointer, a_s: cute.Pointer):
        p = Int32(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        # Locate the padded destination in the sorted compact chunk table;
        # one CTA per actual pair, not 128 mostly-empty CTAs per expert.
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
                row = self.row_base(src, source)
                if const_expr(self.input_kind != "nvfp4"):
                    gs = _input_global_scale(
                        _ld_f32(Int64(scales.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4)), self.divisor)
                groups = self.k // 16
                for it in cutlass.range_constexpr(_ceil(groups, self.threads)):
                    grp = Int32(it * self.threads) + tidx
                    if grp < Int32(groups):
                        packed_at = Int64(aq.toint()) + i * Int64(self.k // 2) + Int64(grp) * Int64(8)
                        scale_at = Int64(a_s.toint()) + i * Int64(groups) + Int64(grp)
                        if const_expr(self.input_kind == "nvfp4"):
                            _copy_nvfp4_group(row, self.k, grp, packed_at, scale_at)
                        else:
                            values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                            for half in cutlass.range_constexpr(2):
                                v = grp * Int32(2) + Int32(half)
                                if const_expr(self.wire):
                                    lo, hi = _ld_v2_u32(row + Int64(v) * Int64(8))
                                    exponent = _ld_u8(row + Int64(self.k) + Int64(v // Int32(4)))
                                    w = _e4m3x8_scaled_bf16(lo, hi, _u32_as_f32(exponent << Uint32(23)))
                                else:
                                    at = row + Int64(v) * Int64(16)
                                    lo, hi = _ld_v2_u32(at)
                                    lo2, hi2 = _ld_v2_u32(at + Int64(8))
                                    w = (lo, hi, lo2, hi2)
                                for q in cutlass.range_constexpr(4):
                                    values[8 * half + 2 * q] = _bf16_lo(w[q])
                                    values[8 * half + 2 * q + 1] = _bf16_hi(w[q])
                            _quantize_row16(values, gs, packed_at, scale_at)


class ChunkSwiGLUA4:
    """``act[128 t + r]`` of grouped row ``first_t + r``: ``bf16(bf16(silu(g)) *
    u)`` from ``g``/``u`` ``[pairs, I]``, quantized to NVFP4 with the expert's
    down ``input_scale``; one thread per 16-value group."""

    threads = 128

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0,
                 activation: str = "silu_bf16_product", alpha: float | None = None, beta: float | None = None,
                 scale_convention: str = "multiplier"):
        self.divisor = scale_convention == "divisor"
        self.inter, self.experts, self.limit = int(inter), int(experts), float(limit)
        self.post_silu = activation == "silu_post_clamp_fp32_product"
        self.oai = activation == "swigluoai"
        self.alpha, self.beta = alpha, beta
        self.in_offset = nvfp4_input_scale_offset(experts, hidden, inter)

    def key(self) -> tuple:
        return ("chunk_swiglu_a4", self.inter, self.experts, self.limit, self.in_offset) + (
            ("silu_post_clamp_fp32_product",) if self.post_silu else ()) + (
                ("swigluoai", self.alpha, self.beta) if self.oai else ()) + (("divisor",) if self.divisor else ())

    @cute.jit
    def __call__(self, g: cute.Pointer, u: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer,
                 aq: cute.Pointer, a_s: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(g, u, meta, scales, aq, a_s).launch(
            grid=(max_tiles * Int32(STREAM_TILE_M), _ceil(self.inter // 16, self.threads), 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, g: cute.Pointer, u: cute.Pointer, meta: cute.Pointer, scales: cute.Pointer, aq: cute.Pointer,
               a_s: cute.Pointer):
        i = Int32(cute.arch.block_idx()[0])
        grp = Int32(cute.arch.block_idx()[1]) * Int32(self.threads) + Int32(cute.arch.thread_idx()[0])
        tile = i // Int32(STREAM_TILE_M)
        r = i % Int32(STREAM_TILE_M)
        groups = self.inter // 16
        if (tile < _i32_at(Int64(meta.toint()), 1)) & (grp < Int32(groups)):
            e, first, live = _chunk(meta, self.experts, tile)
            if r < live:
                if const_expr(self.divisor):
                    gs = _ld_f32(Int64(scales.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                else:
                    gs = Float32(1.0) / _ld_f32(Int64(scales.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                at = (Int64(first + r) * Int64(self.inter) + Int64(grp) * Int64(16)) * Int64(2)
                values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                for half in cutlass.range_constexpr(2):
                    g0, g1 = _ld_v2_u32(Int64(g.toint()) + at + Int64(16 * half))
                    g2, g3 = _ld_v2_u32(Int64(g.toint()) + at + Int64(16 * half + 8))
                    u0, u1 = _ld_v2_u32(Int64(u.toint()) + at + Int64(16 * half))
                    u2, u3 = _ld_v2_u32(Int64(u.toint()) + at + Int64(16 * half + 8))
                    gw = (g0, g1, g2, g3)
                    uw = (u0, u1, u2, u3)
                    for q in cutlass.range_constexpr(4):
                        for side in cutlass.range_constexpr(2):
                            gate = _bf16_lo(gw[q]) if side == 0 else _bf16_hi(gw[q])
                            up = _bf16_lo(uw[q]) if side == 0 else _bf16_hi(uw[q])
                            if const_expr(self.oai):
                                values[8 * half + 2 * q + side] = _round_bf16(gated_activation_value(
                                    gate, up, has_limit=True, limit_value=self.limit,
                                    is_swigluoai=True, alpha=self.alpha, offset=self.beta))
                            elif const_expr(self.post_silu):
                                values[8 * half + 2 * q + side] = _round_bf16(_post_silu_product(gate, up, self.limit))
                            else:
                                if const_expr(self.limit > 0.0):
                                    gate = cutlass.select_(gate > Float32(self.limit), Float32(self.limit), gate)
                                    up = cutlass.select_(up > Float32(self.limit), Float32(self.limit), up)
                                    up = cutlass.select_(up < Float32(-self.limit), Float32(-self.limit), up)
                                silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
                                values[8 * half + 2 * q + side] = _round_bf16(silu * up)
                _quantize_row16(values, gs, Int64(aq.toint()) + Int64(i) * Int64(self.inter // 2) + Int64(grp) * Int64(8),
                                Int64(a_s.toint()) + Int64(i) * Int64(groups) + Int64(grp))


class StreamNvfp4LinearA4:
    """``y[p] = alpha_e * in_e * (a_p . w_e^T)`` with packed NVFP4 ``a``
    ``[max_tiles * 128, K/2]`` + E4M3 ``[max_tiles * 128, K/16]`` (chunk-padded)
    and NVFP4 ``w [E, N, K/2]`` with E4M3 ``s [E, N, K/16]``, alphas ``[E]``,
    input scales ``[E]``; ``y`` BF16 ``[pairs, N]`` in grouped rows. Grid ``(N /
    tile_n, max_tiles)``. Sparse tiles use two 16-row warps, wide tiles use
    eight 32-row warps. Operand staging has independent A and W row counts."""

    threads = 256
    tile_n = 128

    def __init__(self, *, n: int, k: int, experts: int, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128, scale_convention: str = "multiplier"):
        self.divisor = scale_convention == "divisor"
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.tile_m, self.tile_n = int(tile_m), int(tile_n)
        self.live_min, self.live_max = int(live_min), int(live_max)
        if self.tile_m not in (16, 128) or self.tile_n not in (32, 64, 128):
            raise ValueError("NVFP4 A4 tile needs M in (16,128), N in (32,64,128)")
        if self.tile_m == 128 and self.tile_n != 128:
            raise ValueError("the wide NVFP4 A4 tile needs N=128")
        self.threads = 64 if self.tile_m == 16 else 256
        self.warp_rows = 16 if self.tile_m == 16 else 32
        self.m_tiles = self.warp_rows // 16
        self.warp_cols = self.tile_n // 2
        self.n_tiles = self.warp_cols // 8
        self.frags = 4 * self.m_tiles * self.n_tiles
        if self.n <= 0 or self.n % 16 or self.k <= 0 or self.k % 16:
            raise ValueError("NVFP4 A4 linear needs positive N16-aligned N and K % 16 == 0")
        self.k_block = 256 if self.k % 256 == 0 else 128 if self.k % 128 == 0 else 64 if self.k % 64 == 0 else 128
        self.blocks = _ceil(self.k, self.k_block)
        self.row_bytes = self.k_block // 2
        # Shared rows retain a power-of-two pitch; global rows never gain padding.
        self.stride = self.row_bytes
        self.s_bytes = self.k_block // 16
        self.s_stride = self.s_bytes + 4
        self.tile_bytes = self.tile_m * self.stride
        self.weight_bytes = self.tile_n * self.stride
        self.sc_bytes = self.tile_m * self.s_stride
        self.weight_sc_bytes = self.tile_n * self.s_stride
        self.raw_offset = self.tile_bytes + self.weight_bytes + self.sc_bytes + self.weight_sc_bytes
        self.raw_stride = self.row_bytes + 16
        self.stage_bytes = self.raw_offset + ((self.tile_m + self.tile_n) * self.raw_stride if self.k % 32 else 0)
        self.alpha_offset = nvfp4_alpha_offset(experts, self.n, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("stream_linear_nvfp4_a4", 8, self.n, self.k, self.experts, self.k_block,
                self.tile_m, self.tile_n, self.live_min, self.live_max) + (("divisor",) if self.divisor else ())

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.stage_bytes], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        self.kernel(aq, a_s, meta, w, s, y).launch(grid=(_ceil(self.n, self.tile_n), max_tiles, 1),
                                                   block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1)

    @cute.jit
    def _swizzle(self, row):
        # Rotate the three row bits within the actual row, never into its
        # neighbour. For narrow rows the row stride supplies the high banks.
        mask = min(self.row_bytes // 16 - 1, 7)
        shift = min(self.row_bytes // 64, 2)
        return (((row & Int32(3)) << Int32(1)) & Int32(mask if self.row_bytes >= 128 else 0)) \
            | ((row >> Int32(max(shift, 1))) & Int32(1 if self.row_bytes >= 128 else mask))

    @cute.jit
    def _load(self, block, ring, a_rows: Int64, as_rows: Int64, w_rows: Int64, s_rows: Int64, live: Int32, tidx):
        stage = ring + (Int32(block) % Int32(2)) * Int32(self.stage_bytes)
        self._load_operand(stage, a_rows, as_rows, block, tidx, self.tile_m, 0,
                           self.tile_bytes + self.weight_bytes, live)
        weight_rows = Int32(self.tile_n)
        if const_expr(self.n % self.tile_n):
            weight_rows = Int32(self.n) - Int32(cute.arch.block_idx()[0]) * Int32(self.tile_n)
        self._load_operand(stage, w_rows, s_rows, block, tidx, self.tile_n, self.tile_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes, weight_rows)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_operand(self, stage, rows_at: Int64, scales_at: Int64, block, tidx,
                      count: cutlass.Constexpr, packed_offset: cutlass.Constexpr, scale_offset: cutlass.Constexpr,
                      row_limit: Int32):
        """Stage native rows; zero-fill K tails and skip output-masked rows.

        K16 rows can be only eight-byte aligned, and their scale rows can be
        byte aligned. Keep the aligned vector path compile-time specialized;
        never read a neighbouring row to manufacture an aligned transaction.
        """
        if const_expr(self.k % 32):
            # Copy aligned sectors into a temporary shared row. Odd native rows
            # start eight bytes into the first sector; only the previous row's
            # final eight bytes are fetched, never bytes before the allocation.
            raw_base = self.raw_offset + (packed_offset // self.stride) * self.raw_stride
            chunks = self.raw_stride // 16
            for q in cutlass.range_constexpr(_ceil(count * chunks, self.threads)):
                idx = tidx + Int32(q * self.threads)
                row = idx // Int32(chunks)
                piece = idx % Int32(chunks)
                shift = (row & Int32(1)) * Int32(8)
                logical = Int32(block) * Int32(self.row_bytes) + piece * Int32(16) - shift
                dst = stage + Int32(raw_base) + row * Int32(self.raw_stride) + piece * Int32(16)
                if (row < Int32(count)) & (row < row_limit):
                    src = rows_at + Int64(row) * Int64(self.k // 2) + Int64(logical)
                    if logical + Int32(16) <= Int32(self.k // 2):
                        cp_async4_shared_global(dst, src)
                    elif logical < Int32(self.k // 2):
                        cp_async_u64_shared_global(dst, src)
                        st_shared_u32(dst + Int32(8), Uint32(0))
                        st_shared_u32(dst + Int32(12), Uint32(0))
                    else:
                        for z in cutlass.range_constexpr(4):
                            st_shared_u32(dst + Int32(4 * z), Uint32(0))
        else:
            copy_bytes = 16
            chunks = self.row_bytes // copy_bytes
            for q in cutlass.range_constexpr(_ceil(count * chunks, self.threads)):
                idx = tidx + Int32(q * self.threads)
                row = idx // Int32(chunks)
                piece = idx % Int32(chunks)
                byte = piece * Int32(copy_bytes)
                dst = stage + Int32(packed_offset) + row * Int32(self.stride) \
                    + ((byte // Int32(16)) ^ self._swizzle(row)) * Int32(16) + byte % Int32(16)
                if (row < Int32(count)) & (row < row_limit):
                    valid = Int32(1)
                    if const_expr(self.k % self.k_block):
                        valid = Int32(Int32(block) * Int32(self.row_bytes) + byte < Int32(self.k // 2))
                    if valid != Int32(0):
                        src = rows_at + Int64(row) * Int64(self.k // 2) \
                            + Int64(block) * Int64(self.row_bytes) + Int64(byte)
                        if const_expr(copy_bytes == 16):
                            cp_async4_shared_global(dst, src)
                        else:
                            cp_async_u64_shared_global(dst, src)
                    else:
                        for z in cutlass.range_constexpr(copy_bytes // 4):
                            st_shared_u32(dst + Int32(4 * z), Uint32(0))
        words = self.s_bytes // 4
        for q in cutlass.range_constexpr(_ceil(count * words, self.threads)):
            idx = tidx + Int32(q * self.threads)
            row = idx // Int32(words)
            word = idx % Int32(words)
            col = Int32(block) * Int32(self.s_bytes) + word * Int32(4)
            dst = stage + Int32(scale_offset) + row * Int32(self.s_stride) + word * Int32(4)
            if (row < Int32(count)) & (row < row_limit):
                if const_expr(self.k % 64 == 0):
                    valid = Int32(1)
                    if const_expr(self.k % self.k_block):
                        valid = Int32(col < Int32(self.k // 16))
                    if valid != Int32(0):
                        cp_async_u32_shared_global(dst, scales_at + Int64(row) * Int64(self.k // 16) + Int64(col))
                    else:
                        st_shared_u32(dst, Uint32(0))
                else:
                    value = Uint32(0)
                    for byte in cutlass.range_constexpr(4):
                        if col + Int32(byte) < Int32(self.k // 16):
                            value = value | (Uint32(_ld_u8(scales_at + Int64(row) * Int64(self.k // 16)
                                                          + Int64(col) + Int64(byte))) << Uint32(8 * byte))
                    st_shared_u32(dst, value)

    @cute.jit
    def _realign_stage(self, stage, tidx):
        if const_expr(self.k % 32):
            chunks = self.row_bytes // 16
            for q in cutlass.range_constexpr(_ceil((self.tile_m + self.tile_n) * chunks, self.threads)):
                idx = tidx + Int32(q * self.threads)
                row = idx // Int32(chunks)
                piece = idx % Int32(chunks)
                if row < Int32(self.tile_m + self.tile_n):
                    src = stage + Int32(self.raw_offset) + row * Int32(self.raw_stride) \
                        + (row & Int32(1)) * Int32(8) + piece * Int32(16)
                    dst = stage + row * Int32(self.stride) + (piece ^ self._swizzle(row)) * Int32(16)
                    # src may be eight-byte aligned, so use four natural words.
                    for word in cutlass.range_constexpr(4):
                        st_shared_u32(dst + Int32(4 * word), ld_shared_u32(src + Int32(4 * word)))
            cute.arch.sync_threads()

    @cute.jit
    def _mma_stage(self, stage, acc: cute.Tensor, warp_m, warp_n, q, c, mat, mrow):
        """One K block of the 32 x 64 warp tile from ring stage ``stage``: A and W fragments
        by ``ldmatrix`` (lane ``t`` addresses row ``t % 8`` of matrix ``t / 8``), their E4M3
        scale words, 16 ``m16n8k64`` block-scaled FP4 MMAs per 64 K."""
        sa = stage
        sw = stage + Int32(self.tile_bytes)
        ssa = stage + Int32(self.tile_bytes + self.weight_bytes)
        ssb = ssa + Int32(self.sc_bytes)
        swz = self._swizzle(mrow)
        for step in cutlass.range_constexpr(self.k_block // 64):
            apiece = ((Int32(2 * step) + (mat >> Int32(1))) ^ swz) * Int32(16)
            bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ swz) * Int32(16)
            sfa = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Uint32)
            a = cute.make_rmem_tensor(cute.make_layout((4 * self.m_tiles,), stride=(1,)), Uint32)
            for mt in cutlass.range_constexpr(self.m_tiles):
                r0 = warp_m * Int32(self.warp_rows) + Int32(16 * mt) + q
                sfa[mt] = ld_shared_u32(ssa + (r0 + Int32(8) * (c & Int32(1))) * Int32(self.s_stride)
                                        + Int32(4 * step))
                # ldmatrix x4: lane t addresses row t % 8 of matrix t / 8 (rows +8, then K +16 B).
                arow = warp_m * Int32(self.warp_rows) + Int32(16 * mt) + Int32(8) * (mat & Int32(1)) + mrow
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sa + arow * Int32(self.stride) + apiece)
                a[4 * mt] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3
            bw = cute.make_rmem_tensor(cute.make_layout((2 * self.n_tiles,), stride=(1,)), Uint32)
            for pair in cutlass.range_constexpr(self.n_tiles // 2):
                # Matrices: b0, b1 of column block 2 * pair, then of 2 * pair + 1.
                bcol = warp_n * Int32(self.warp_cols) + Int32(16 * pair) + Int32(8) * (mat >> Int32(1)) + mrow
                b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(sw + bcol * Int32(self.stride) + bpiece)
                bw[4 * pair] = b0
                bw[4 * pair + 1] = b1
                bw[4 * pair + 2] = b2
                bw[4 * pair + 3] = b3
            for nt in cutlass.range_constexpr(self.n_tiles):
                col = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + q
                sfb = ld_shared_u32(ssb + col * Int32(self.s_stride) + Int32(4 * step))
                b0 = bw[2 * nt]
                b1 = bw[2 * nt + 1]
                for mt in cutlass.range_constexpr(self.m_tiles):
                    f = 4 * (self.n_tiles * mt + nt)
                    d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                        a[4 * mt], a[4 * mt + 1], a[4 * mt + 2], a[4 * mt + 3], b0, b1, sfa[mt], sfb)
                    acc[f] = d0
                    acc[f + 1] = d1
                    acc[f + 2] = d2
                    acc[f + 3] = d3

    @cute.kernel
    def kernel(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
               y: cute.Pointer):
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
                w_rows = Int64(w.toint()) + (Int64(e) * Int64(self.n) + Int64(n0)) * Int64(self.k // 2)
                s_rows = Int64(s.toint()) + (Int64(e) * Int64(self.n) + Int64(n0)) * Int64(self.k // 16)
                if const_expr(self.divisor):
                    factor = Float32(1.0)
                    weight_global = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4))
                    input_global = _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                else:
                    factor = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                        * _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
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
                    self._realign_stage(ring + (block % Int32(2)) * Int32(self.stage_bytes), tidx)
                    if active:
                        stage = ring + (block % Int32(2)) * Int32(self.stage_bytes)
                        self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow)
                    cute.arch.sync_threads()
                    if block + Int32(2) < Int32(self.blocks):
                        self._load(block + Int32(2), ring, a_rows, as_rows, w_rows, s_rows, live, tidx)
                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                            if row < live:
                                for nt in cutlass.range_constexpr(self.n_tiles):
                                    f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                    col = n0 + warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                    if const_expr(self.divisor):
                                        acc[f] = div_rn_f32(div_rn_f32(acc[f], weight_global), input_global)
                                        acc[f + 1] = div_rn_f32(div_rn_f32(acc[f + 1], weight_global), input_global)
                                    if col < Int32(self.n):
                                        st_global_u32(Int64(y.toint()) + (Int64(first + row) * Int64(self.n) + Int64(col))
                                                      * Int64(2), pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))


@cute.jit
def _swiglu16(gw, uw, values: cute.Tensor, base: cutlass.Constexpr, limit: cutlass.Constexpr,
              post_silu: cutlass.Constexpr = False, oai: cutlass.Constexpr = False,
              alpha: cutlass.Constexpr = None, beta: cutlass.Constexpr = None):
    """``values[base + j] = bf16(bf16(silu(g)) * u)`` of eight BF16 gate/up pairs (``ChunkSwiGLUA4``'s math)."""
    for q in cutlass.range_constexpr(4):
        for side in cutlass.range_constexpr(2):
            gate = _bf16_lo(gw[q]) if side == 0 else _bf16_hi(gw[q])
            up = _bf16_lo(uw[q]) if side == 0 else _bf16_hi(uw[q])
            if const_expr(oai):
                values[base + 2 * q + side] = _round_bf16(gated_activation_value(
                    gate, up, has_limit=True, limit_value=limit,
                    is_swigluoai=True, alpha=alpha, offset=beta))
            elif const_expr(post_silu):
                values[base + 2 * q + side] = _round_bf16(_post_silu_product(gate, up, limit))
            else:
                if const_expr(limit > 0.0):
                    gate = cutlass.select_(gate > Float32(limit), Float32(limit), gate)
                    up = cutlass.select_(up > Float32(limit), Float32(limit), up)
                    up = cutlass.select_(up < Float32(-limit), Float32(-limit), up)
                silu = _round_bf16(div_rn_f32(gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)))
                values[base + 2 * q + side] = _round_bf16(silu * up)


class StreamNvfp4GateUpA4(StreamNvfp4LinearA4):
    """Fused W4A4 gate/up + SwiGLU + NVFP4 quantization of the down input.

    Grid ``(I / half_n, max_tiles)``. A CTA streams its chunk's packed
    activations against ``tile_n`` weight rows, ``half_n`` gate rows of ``w1``
    then matching up rows of ``w3`` (warps ``warp_n`` 0 and 1), so the
    main loop is ``StreamNvfp4LinearA4``'s. Each warp scales its accumulators by
    its projection's ``alpha_e * in_e`` and rounds them to BF16 into shared
    memory (the ring, after the last K block); then each thread takes 16
    columns of one row, computes ``bf16(bf16(silu(g)) * u)`` and writes them as
    NVFP4 with the expert's down ``input_scale`` into ``aq [max_tiles * 128,
    I/2]`` / ``a_s [max_tiles * 128, I/16]``."""

    half_n = 64
    # Exchange rows: 64 gate then 64 up BF16 columns, padded (conflict-free stores).
    x_stride = 2 * 64 * 2 + 16

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0, k_block: int | None = None,
                 stages: int = 2, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128, activation: str = "silu_bf16_product", alpha: float | None = None, beta: float | None = None,
                 scale_convention: str = "multiplier"):
        super().__init__(n=2 * int(inter), k=hidden, experts=experts, tile_m=tile_m, tile_n=tile_n,
                         live_min=live_min, live_max=live_max, scale_convention=scale_convention)
        self.half_n = self.tile_n // 2
        self.x_stride = 2 * self.tile_n + 16
        self.inter, self.limit = int(inter), float(limit)
        self.post_silu = activation == "silu_post_clamp_fp32_product"
        self.oai = activation == "swigluoai"
        self.alpha, self.beta = alpha, beta
        if k_block is None:
            k_block = self.k_block
        if k_block not in (64, 128, 256):
            raise ValueError("NVFP4 K block must be 64, 128 or 256")
        self._set_k_block(k_block)
        self.stages = int(stages)
        if self.inter <= 0 or self.inter % 16:
            raise ValueError("the fused NVFP4 A4 gate/up needs positive I % 16 == 0")
        # Gate and up scale operands share the [E, I, H/16] shape.
        self.alpha_offset = nvfp4_alpha_offset(experts, self.inter, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.inter, self.k)
        self.down_in_offset = nvfp4_input_scale_offset(experts, hidden, self.inter)
        if self.stages < 2 or self.stages > 8:
            raise ValueError("NVFP4 gate/up needs 2..8 pipeline stages")

    def _set_k_block(self, k_block: int):
        self.k_block = int(k_block)
        self.blocks = _ceil(self.k, self.k_block)
        self.row_bytes = self.k_block // 2
        self.stride = self.row_bytes
        self.s_bytes = self.k_block // 16
        self.s_stride = self.s_bytes + 4
        self.tile_bytes = self.tile_m * self.stride
        self.weight_bytes = self.tile_n * self.stride
        self.sc_bytes = self.tile_m * self.s_stride
        self.weight_sc_bytes = self.tile_n * self.s_stride
        self.raw_offset = self.tile_bytes + self.weight_bytes + self.sc_bytes + self.weight_sc_bytes
        self.raw_stride = self.row_bytes + 16
        self.stage_bytes = self.raw_offset + ((self.tile_m + self.tile_n) * self.raw_stride if self.k % 32 else 0)

    def key(self) -> tuple:
        return ("stream_gate_up_nvfp4_a4", 8, self.inter, self.k, self.experts, self.k_block, self.stages,
                self.limit, self.down_in_offset, self.tile_m, self.tile_n, self.live_min, self.live_max) + (
                    ("silu_post_clamp_fp32_product",) if self.post_silu else ()) + (
                ("swigluoai", self.alpha, self.beta) if self.oai else ()) + (("divisor",) if self.divisor else ())

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, max(self.stages * self.stage_bytes, self.tile_m * self.x_stride)], 128],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                 w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer,
                 max_tiles: Int32, stream: cuda.CUstream):
        self.fused(aq, a_s, meta, w1, s1, w3, s3, s2, act, act_s).launch(
            grid=(_ceil(self.inter, self.half_n), max_tiles, 1), block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=1)

    @cute.jit
    def _load2(self, block, ring, a_rows: Int64, as_rows: Int64, w_rows: Int64, w_up: Int64, s_rows: Int64,
               s_up: Int64, live: Int32, tidx):
        stage = ring + (Int32(block) % Int32(self.stages)) * Int32(self.stage_bytes)
        self._load_operand(stage, a_rows, as_rows, block, tidx, self.tile_m, 0,
                           self.tile_bytes + self.weight_bytes, live)
        weight_rows = Int32(self.half_n)
        if const_expr(self.inter % self.half_n):
            weight_rows = Int32(self.inter) - Int32(cute.arch.block_idx()[0]) * Int32(self.half_n)
        self._load_operand(stage, w_rows, s_rows, block, tidx, self.half_n, self.tile_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes,
                           weight_rows)
        self._load_operand(stage, w_up, s_up, block, tidx, self.half_n,
                           self.tile_bytes + self.half_n * self.stride,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes + self.half_n * self.s_stride,
                           weight_rows)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def fused(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
              w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer):
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
                # ``stages``-deep cp.async ring, one group per K block (empty past the end): block b
                # waits for its group, the barrier frees stage (b - 1) % stages, which takes block
                # b + stages - 1.
                for pre in cutlass.range_constexpr(self.stages - 1):
                    if const_expr(pre < self.blocks):
                        self._load2(pre, ring, a_rows, as_rows, w_rows, w_up, s_rows, s_up, live, tidx)
                    else:
                        cute.arch.cp_async_commit_group()
                warp_m = warp_id // Int32(2)
                warp_n = warp_id % Int32(2)
                # warp_n 0: gate columns (w1 / s1); 1: up columns (w3 / s3).
                sp = cutlass.select_(warp_n == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
                if const_expr(self.divisor):
                    factor = Float32(1.0)
                    weight_global = _ld_f32(sp + Int64(self.alpha_offset) + Int64(e) * Int64(4))
                    input_global = _ld_f32(sp + Int64(self.in_offset) + Int64(e) * Int64(4))
                else:
                    factor = _ld_f32(sp + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                        * _ld_f32(sp + Int64(self.in_offset) + Int64(e) * Int64(4))
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
                    self._realign_stage(ring + (block % Int32(self.stages)) * Int32(self.stage_bytes), tidx)
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
                # The ring is free (every K block consumed, last barrier passed): exchange the
                # BF16-rounded gate / up tiles, row r at ring + r * x_stride (gate 0..127 B, up 128..255 B).
                if active:
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                col = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                if const_expr(self.divisor):
                                    acc[f] = div_rn_f32(div_rn_f32(acc[f], weight_global), input_global)
                                    acc[f + 1] = div_rn_f32(div_rn_f32(acc[f + 1], weight_global), input_global)
                                st_shared_u32(ring + row * Int32(self.x_stride) + col * Int32(2),
                                              pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
                cute.arch.sync_threads()
                if const_expr(self.divisor):
                    gs = _ld_f32(Int64(s2.toint()) + Int64(self.down_in_offset) + Int64(e) * Int64(4))
                else:
                    gs = Float32(1.0) / _ld_f32(Int64(s2.toint()) + Int64(self.down_in_offset) + Int64(e) * Int64(4))
                groups = self.half_n // 16
                for it in cutlass.range_constexpr(_ceil(self.tile_m * groups, self.threads)):
                    gi = tidx + Int32(it * self.threads)
                    row = gi // Int32(groups)
                    grp = gi % Int32(groups)
                    if (row < live) & (n0 + grp * Int32(16) < Int32(self.inter)):
                        values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                        at = ring + row * Int32(self.x_stride) + grp * Int32(32)
                        for half in cutlass.range_constexpr(2):
                            gw = ld_shared_v4_u32(at + Int32(16 * half))
                            uw = ld_shared_v4_u32(at + Int32(self.half_n * 2 + 16 * half))
                            _swiglu16(gw, uw, values, 8 * half, self.limit, self.post_silu,
                                      self.oai, self.alpha, self.beta)
                        i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(row)
                        col16 = Int64(n0 // Int32(16) + grp)
                        _quantize_row16(values, gs, Int64(act.toint()) + i * Int64(self.inter // 2) + col16 * Int64(8),
                                        Int64(act_s.toint()) + i * Int64(self.inter // 16) + col16)
