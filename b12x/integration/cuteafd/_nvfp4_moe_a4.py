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
    packed operands for any ``K % 16 == 0`` and ``N % 8 == 0`` with zero stored
    padding. Every specialization uses one canonical 256-K (128-byte) SMEM stage
    and one conflict-free ``SWIZZLE_128B`` layout (``piece ^ (row & 7)``),
    peeling any ``K % 256`` tail to ``ceil(tail_k / 64)`` MMA steps and
    hardware-zero-filling partial K/N tiles. Wide ``K % 32 == 0`` tiles stream
    weights through a 3-deep 3D TMA ring ``(N, K/2, E)`` paired with a 2-deep
    ``cp.async`` live-activation ring; sparse and ``K % 32 == 16`` tiles use a
    single-barrier multi-stage ``cp.async`` ring (8-byte ``cp.async.ca`` with
    ``src-size`` zero-fill when ``K % 32 == 16``).
``StreamNvfp4GateUpA4``  gate/up, SwiGLU and the down-input quantization in
    one launch for any ``I % 16 == 0`` and ``H % 16 == 0``: half the CTA's
    weight rows are gate and half are matching up rows, BF16-rounded exactly as
    the linear's output, exchanged through shared memory and written as
    ``ChunkSwiGLUA4`` writes them (bit-identical ``act`` bytes).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.activation import gated_activation_value
from b12x._lib.intrinsics import (
    div_rn_f32,
    ld_global_nc_u8,
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
    st_global_v4_u32,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from ._fp8_moe_kernels import _ceil, _i32_at, _ld_u8, _ld_v2_u32, _post_silu_product, _round_bf16, _u32_as_f32
from ._fp8_moe_stream import STREAM_TILE_M, _chunk
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from ._nvfp4_moe_kernels import nvfp4_alpha_offset

__all__ = ["ChunkRowsA4", "ChunkSwiGLUA4", "StreamNvfp4GateUpA4", "StreamNvfp4LinearA4", "nvfp4_input_scale_offset"]


def nvfp4_input_scale_offset(experts: int, rows: int, k: int) -> int:
    """Byte offset of the FP32 input scales in an NVFP4 scale operand (after the alphas)."""
    return nvfp4_alpha_offset(experts, rows, k) + 4 * int(experts)


@dsl_user_op
def _cp_async16_zfill(smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None):
    """16-byte ``cp.async.cg`` when ``pred != 0``, else hardware 16-byte SMEM zero-fill (no global read)."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 z; .reg .pred p; setp.ne.b32 p, $0, 0; selp.u32 z, 16, 0, p; "
        "cp.async.cg.shared.global [$1], [$2], 16, z; }",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async8_zfill(smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None):
    """8-byte ``cp.async.ca`` when ``pred != 0``, else hardware 8-byte SMEM zero-fill (no global read)."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 z; .reg .pred p; setp.ne.b32 p, $0, 0; selp.u32 z, 8, 0, p; "
        "cp.async.ca.shared.global [$1], [$2], 8, z; }",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async4_zfill(smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None):
    """4-byte ``cp.async.ca`` when ``pred != 0``, else hardware 4-byte SMEM zero-fill (no global read)."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 z; .reg .pred p; setp.ne.b32 p, $0, 0; selp.u32 z, 4, 0, p; "
        "cp.async.ca.shared.global [$1], [$2], 4, z; }",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _u8_weight_3d(pointer: cute.Pointer, rows: int, cols: int, experts: int) -> cute.Tensor:
    """Unpadded 3D weight view ``(rows, cols, experts)`` for TMA: per-expert N and K clamping."""
    u8 = cute.make_ptr(cutlass.Uint8, Int64(pointer.toint()), cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(u8, cute.make_layout((rows, cols, experts), stride=(cols, 1, rows * cols)))


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
        if self.wire:
            if self.k <= 0 or self.k % 32:
                raise ValueError("K must be a positive multiple of 32 for FP8 wire rows")
        elif self.k <= 0 or self.k % 16:
            raise ValueError("K must be a positive multiple of 16")

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
        if self.inter <= 0 or self.inter % 16:
            raise ValueError("I must be a positive multiple of 16")
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
    input scales ``[E]``; ``y`` BF16 ``[pairs, N]`` in grouped rows. Grid
    ``(ceil(N / tile_n), max_tiles)``. Accepts any ``K >= 16, K % 16 == 0`` and
    ``N >= 8, N % 8 == 0`` with zero stored padding."""

    threads = 256
    tile_n = 128
    w_ring = 3
    a_ring = 2

    def __init__(self, *, n: int, k: int, experts: int, tile_m: int = 128, tile_n: int = 128,
                 stages: int | None = None, live_min: int = 1, live_max: int = 128,
                 scale_convention: str = "multiplier"):
        self.divisor = scale_convention == "divisor"
        self.n, self.k, self.experts = int(n), int(k), int(experts)
        self.tile_m, self.tile_n = int(tile_m), int(tile_n)
        self.live_min, self.live_max = int(live_min), int(live_max)
        if self.tile_m not in (16, 128) or self.tile_n not in (32, 64, 128):
            raise ValueError("NVFP4 A4 tile needs M in (16,128), N in (32,64,128)")
        if self.tile_m == 128 and self.tile_n != 128:
            raise ValueError("the wide NVFP4 A4 tile needs N=128")
        if self.n <= 0 or self.n % 8 or self.k <= 0 or self.k % 16:
            raise ValueError("NVFP4 A4 linear needs N % 8 == 0 and K % 16 == 0")
        self.threads = 64 if self.tile_m == 16 else 256
        self.warp_rows = 16 if self.tile_m == 16 else 32
        self.m_tiles = self.warp_rows // 16
        self.warp_cols = self.tile_n // 2
        self.n_tiles = self.warp_cols // 8
        self.frags = 4 * self.m_tiles * self.n_tiles
        if stages is None:
            self.stages = 2 if (self.tile_m == 128 or self.tile_n == 128) else 3
        else:
            self.stages = int(stages)
        # Unified canonical 256-K (128-byte) SMEM stage across all K % 16 == 0 shapes.
        self.k_block = 256
        self.blocks = _ceil(self.k, self.k_block)
        self.full_blocks = self.k // self.k_block
        self.tail_k = self.k % self.k_block
        self.tail_steps = _ceil(self.tail_k, 64) if self.tail_k else 0
        self.row_bytes = self.k_block // 2
        self.stride = self.row_bytes
        self.s_bytes = self.k_block // 16
        self.s_stride = self.s_bytes + 4
        self.tile_bytes = self.tile_m * self.stride
        self.weight_bytes = self.tile_n * self.stride
        self.sc_bytes = self.tile_m * self.s_stride
        self.weight_sc_bytes = self.tile_n * self.s_stride
        self.stage_bytes = self.tile_bytes + self.weight_bytes + self.sc_bytes + self.weight_sc_bytes
        self.out_stride = self.tile_n * 2 + 16
        self.use_tma_w = (self.tile_m == 128) and (self.k % 32 == 0)
        self.coalesced_epi = (self.tile_m == 128) and (self.n % 128 == 0)
        self.min_blocks = 1 if self.tile_m == 128 else (4 if self.tile_n == 32 else 2)
        self.ring_min_bytes = self.tile_m * self.out_stride if self.coalesced_epi else 0
        self.alpha_offset = nvfp4_alpha_offset(experts, self.n, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.n, self.k)

    def key(self) -> tuple:
        return ("stream_linear_nvfp4_a4", 6, self.n, self.k, self.experts, self.k_block, self.stages,
                self.use_tma_w, self.tile_m, self.tile_n, self.live_min, self.live_max) + (
                    ("divisor",) if self.divisor else ())

    def _w_layout(self, box_n: int, slots: int):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Uint8, self.stride), cutlass.Uint8)
        return cute.tile_to_shape(atom, (box_n, self.stride, slots), order=(0, 1, 2))

    def _storage_tma(self, w_layout):
        class StorageTma:
            pass

        StorageTma.__annotations__ = {
            "mbar": cute.struct.MemRange[cutlass.Int64, self.w_ring],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(w_layout)], 1024],
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.tile_bytes], 1024],
            "sa": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.sc_bytes], 16],
            "sb": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.a_ring * self.weight_sc_bytes], 16],
        }
        return cute.struct(StorageTma)

    def _storage_ring(self):
        class StorageRing:
            pass

        alloc_bytes = max(self.stages * self.stage_bytes, self.ring_min_bytes)
        StorageRing.__annotations__ = {
            "ring": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, alloc_bytes], 128],
        }
        return cute.struct(StorageRing)

    @cute.jit
    def _swizzle(self, row):
        return row & Int32(7)

    @cute.jit
    def _w_limit_for(self, n0: Int32, total_n: cutlass.Constexpr, box_n: cutlass.Constexpr) -> Int32:
        if const_expr(total_n % box_n == 0):
            limit = Int32(box_n)
        else:
            rem = Int32(total_n) - n0
            limit = cutlass.select_(rem < Int32(box_n), rem, Int32(box_n))
        return limit

    @cute.jit
    def _load_packed_rows(self, smem_base: Int32, rows_at: Int64, block, tidx: Int32,
                          count: cutlass.Constexpr, row_limit: Int32, full_k: cutlass.Constexpr):
        """Stage ``count`` rows of 128 packed bytes into ``SWIZZLE_128B`` SMEM with hardware zero-fill."""
        chunks = self.row_bytes // 16
        r0 = tidx // Int32(chunks)
        piece = tidx % Int32(chunks)
        smem_piece = piece ^ self._swizzle(r0)
        row_step = self.threads // chunks
        row_bytes_total = self.k // 2
        for q in cutlass.range_constexpr(_ceil(count * chunks, self.threads)):
            row = r0 + Int32(q * row_step)
            dst = smem_base + row * Int32(self.stride) + smem_piece * Int32(16)
            g_row = rows_at + Int64(row) * Int64(row_bytes_total)
            k_col = Int32(block) * Int32(self.row_bytes) + piece * Int32(16)
            if const_expr(self.k % 32 == 0):
                if const_expr(full_k):
                    pred = cutlass.select_(row < row_limit, Int32(1), Int32(0))
                else:
                    pred = cutlass.select_((row < row_limit) & (k_col < Int32(row_bytes_total)), Int32(1), Int32(0))
                _cp_async16_zfill(dst, g_row + Int64(k_col), pred)
            else:
                for half in cutlass.range_constexpr(2):
                    k8 = k_col + Int32(8 * half)
                    if const_expr(full_k):
                        pred8 = cutlass.select_(row < row_limit, Int32(1), Int32(0))
                    else:
                        pred8 = cutlass.select_((row < row_limit) & (k8 < Int32(row_bytes_total)), Int32(1), Int32(0))
                    _cp_async8_zfill(dst + Int32(8 * half), g_row + Int64(k8), pred8)

    @cute.jit
    def _load_scale_rows(self, smem_scale_base: Int32, scales_at: Int64, block, tidx: Int32,
                         count: cutlass.Constexpr, row_limit: Int32, full_k: cutlass.Constexpr):
        """Stage ``count`` rows of 16 E4M3 scale bytes into 20B-strided SMEM with zero-fill."""
        words = self.s_bytes // 4
        sc_row_bytes = self.k // 16
        for q in cutlass.range_constexpr(_ceil(count * words, self.threads)):
            c = tidx + Int32(q * self.threads)
            if c < Int32(count * words):
                row = c // Int32(words)
                word = c % Int32(words)
                dst = smem_scale_base + row * Int32(self.s_stride) + word * Int32(4)
                sc_col = Int32(block) * Int32(self.s_bytes) + word * Int32(4)
                if const_expr(self.k % 64 == 0):
                    if const_expr(full_k):
                        pred = cutlass.select_(row < row_limit, Int32(1), Int32(0))
                    else:
                        pred = cutlass.select_((row < row_limit) & (sc_col < Int32(sc_row_bytes)), Int32(1), Int32(0))
                    _cp_async4_zfill(dst, scales_at + Int64(row) * Int64(sc_row_bytes) + Int64(sc_col), pred)
                else:
                    val = Uint32(0)
                    if const_expr(full_k):
                        valid = row < row_limit
                    else:
                        valid = (row < row_limit) & (sc_col < Int32(sc_row_bytes))
                    if valid:
                        base_b = scales_at + Int64(row) * Int64(sc_row_bytes) + Int64(sc_col)
                        for b in cutlass.range_constexpr(4):
                            if const_expr(full_k):
                                byte_v = Uint32(ld_global_nc_u8(base_b + Int64(b)))
                                val = val | (byte_v << Uint32(8 * b))
                            else:
                                if sc_col + Int32(b) < Int32(sc_row_bytes):
                                    byte_v = Uint32(ld_global_nc_u8(base_b + Int64(b)))
                                    val = val | (byte_v << Uint32(8 * b))
                    st_shared_u32(dst, val)

    @cute.jit
    def _load_operand(self, stage: Int32, rows_at: Int64, scales_at: Int64, block, tidx: Int32,
                      count: cutlass.Constexpr, packed_offset: cutlass.Constexpr, scale_offset: cutlass.Constexpr,
                      row_limit: Int32, full_k: cutlass.Constexpr = False):
        self._load_packed_rows(stage + Int32(packed_offset), rows_at, block, tidx, count, row_limit, full_k)
        self._load_scale_rows(stage + Int32(scale_offset), scales_at, block, tidx, count, row_limit, full_k)

    @cute.jit
    def _load_ring1(self, block, ring: Int32, a_rows: Int64, as_rows: Int64, w_rows: Int64, s_rows: Int64,
                    live: Int32, w_limit: Int32, tidx: Int32, full_k: cutlass.Constexpr = False):
        stage = ring + (Int32(block) % Int32(self.stages)) * Int32(self.stage_bytes)
        self._load_operand(stage, a_rows, as_rows, block, tidx, self.tile_m, 0,
                           self.tile_bytes + self.weight_bytes, live, full_k)
        self._load_operand(stage, w_rows, s_rows, block, tidx, self.tile_n, self.tile_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes, w_limit, full_k)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_tma_w1(self, block, tma_w, t_wg, t_ws, mbar, n_blk: Int32):
        slot = Int32(block) % Int32(self.w_ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.weight_bytes)
        cute.copy(tma_w, t_wg[(None, n_blk, Int32(block))], t_ws[(None, slot)], tma_bar_ptr=barrier)

    @cute.jit
    def _load_a_and_scales_tma1(self, block, sa_base: Int32, ssa_base: Int32, ssb_base: Int32,
                                a_rows: Int64, as_rows: Int64, s_rows: Int64,
                                live: Int32, w_limit: Int32, tidx: Int32, full_k: cutlass.Constexpr = False):
        slot = Int32(block) % Int32(self.a_ring)
        self._load_packed_rows(sa_base + slot * Int32(self.tile_bytes), a_rows, block, tidx, self.tile_m, live, full_k)
        self._load_scale_rows(ssa_base + slot * Int32(self.sc_bytes), as_rows, block, tidx, self.tile_m, live, full_k)
        self._load_scale_rows(ssb_base + slot * Int32(self.weight_sc_bytes), s_rows, block, tidx, self.tile_n,
                              w_limit, full_k)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _mma_pointers(self, sa: Int32, sw: Int32, ssa: Int32, ssb: Int32, acc: cute.Tensor,
                      warp_m: Int32, warp_n: Int32, q: Int32, c: Int32, mat: Int32, mrow: Int32,
                      steps: cutlass.Constexpr):
        swz = self._swizzle(mrow)
        for step in cutlass.range_constexpr(steps):
            apiece = ((Int32(2 * step) + (mat >> Int32(1))) ^ swz) * Int32(16)
            bpiece = ((Int32(2 * step) + (mat & Int32(1))) ^ swz) * Int32(16)
            sfa = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Uint32)
            a = cute.make_rmem_tensor(cute.make_layout((4 * self.m_tiles,), stride=(1,)), Uint32)
            for mt in cutlass.range_constexpr(self.m_tiles):
                r0 = warp_m * Int32(self.warp_rows) + Int32(16 * mt) + q
                sfa[mt] = ld_shared_u32(ssa + (r0 + Int32(8) * (c & Int32(1))) * Int32(self.s_stride)
                                        + Int32(4 * step))
                arow = warp_m * Int32(self.warp_rows) + Int32(16 * mt) + Int32(8) * (mat & Int32(1)) + mrow
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(sa + arow * Int32(self.stride) + apiece)
                a[4 * mt] = a0
                a[4 * mt + 1] = a1
                a[4 * mt + 2] = a2
                a[4 * mt + 3] = a3
            bw = cute.make_rmem_tensor(cute.make_layout((2 * self.n_tiles,), stride=(1,)), Uint32)
            for pair in cutlass.range_constexpr(self.n_tiles // 2):
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

    @cute.jit
    def _mma_stage(self, stage: Int32, acc: cute.Tensor, warp_m: Int32, warp_n: Int32,
                   q: Int32, c: Int32, mat: Int32, mrow: Int32, steps: cutlass.Constexpr = 4):
        sa = stage
        sw = stage + Int32(self.tile_bytes)
        ssa = stage + Int32(self.tile_bytes + self.weight_bytes)
        ssb = ssa + Int32(self.sc_bytes)
        self._mma_pointers(sa, sw, ssa, ssb, acc, warp_m, warp_n, q, c, mat, mrow, steps)

    @cute.jit
    def _run_linear_tma_loop(self, storage, w_layout, tma_w, weight_w, aq: cute.Pointer, a_s: cute.Pointer,
                             s: cute.Pointer, e: Int32, tile: Int32, n_blk: Int32, n0: Int32, live: Int32,
                             tidx: Int32, warp_id: Int32, warp_m: Int32, warp_n: Int32, active,
                             q: Int32, c: Int32, mat: Int32, mrow: Int32, acc: cute.Tensor):
        mbar = storage.mbar.data_ptr()
        s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
        sw_base = shared_ptr_to_u32(storage.w.data_ptr())
        sa_base = shared_ptr_to_u32(storage.a.data_ptr())
        ssa_base = shared_ptr_to_u32(storage.sa.data_ptr())
        ssb_base = shared_ptr_to_u32(storage.sb.data_ptr())
        t_ws, t_wg = cpasync.tma_partition(
            tma_w, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
            cute.group_modes(cute.local_tile(weight_w[None, None, e], (self.tile_n, self.stride), (None, None)), 0, 2))
        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_w)
            for slot in cutlass.range_constexpr(self.w_ring):
                cute.arch.mbarrier_init(mbar + slot, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
        as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
        s_rows = Int64(s.toint()) + (Int64(e) * Int64(self.n) + Int64(n0)) * Int64(self.k // 16)
        w_limit = self._w_limit_for(n0, self.n, self.tile_n)
        self._load_a_and_scales_tma1(0, sa_base, ssa_base, ssb_base, a_rows, as_rows, s_rows,
                                     live, w_limit, tidx, self.full_blocks > 0)
        if warp_id == Int32(0):
            for pre in cutlass.range_constexpr(min(self.w_ring - 1, self.blocks)):
                self._load_tma_w1(pre, tma_w, t_wg, t_ws, mbar, n_blk)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if const_expr(self.blocks > 1):
            self._load_a_and_scales_tma1(1, sa_base, ssa_base, ssb_base, a_rows, as_rows, s_rows,
                                         live, w_limit, tidx, self.full_blocks > 1)
        if const_expr(self.full_blocks > 0):
            for block in cutlass.range(self.full_blocks, unroll=1):
                w_slot = block % Int32(self.w_ring)
                cute.arch.mbarrier_wait(mbar + w_slot, (block // Int32(self.w_ring)) & Int32(1))
                if (block + Int32(self.w_ring - 1) < Int32(self.blocks)) & (warp_id == Int32(0)):
                    self._load_tma_w1(block + Int32(self.w_ring - 1), tma_w, t_wg, t_ws, mbar, n_blk)
                if active:
                    a_slot = block % Int32(self.a_ring)
                    self._mma_pointers(
                        sa_base + a_slot * Int32(self.tile_bytes),
                        sw_base + w_slot * Int32(self.weight_bytes),
                        ssa_base + a_slot * Int32(self.sc_bytes),
                        ssb_base + a_slot * Int32(self.weight_sc_bytes),
                        acc, warp_m, warp_n, q, c, mat, mrow, 4)
                if block + Int32(1) < Int32(self.blocks):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    if block + Int32(2) < Int32(self.blocks):
                        self._load_a_and_scales_tma1(block + Int32(2), sa_base, ssa_base, ssb_base,
                                                     a_rows, as_rows, s_rows, live, w_limit, tidx, False)
        if const_expr(self.tail_k > 0):
            b_tail = Int32(self.full_blocks)
            w_slot = b_tail % Int32(self.w_ring)
            cute.arch.mbarrier_wait(mbar + w_slot, (b_tail // Int32(self.w_ring)) & Int32(1))
            if active:
                a_slot = b_tail % Int32(self.a_ring)
                self._mma_pointers(
                    sa_base + a_slot * Int32(self.tile_bytes),
                    sw_base + w_slot * Int32(self.weight_bytes),
                    ssa_base + a_slot * Int32(self.sc_bytes),
                    ssb_base + a_slot * Int32(self.weight_sc_bytes),
                    acc, warp_m, warp_n, q, c, mat, mrow, self.tail_steps)
        cute.arch.sync_threads()
        return sw_base

    @cute.jit
    def _run_linear_ring_loop(self, ring: Int32, aq: cute.Pointer, a_s: cute.Pointer, w: cute.Pointer,
                              s: cute.Pointer, e: Int32, tile: Int32, n0: Int32, live: Int32, tidx: Int32,
                              warp_m: Int32, warp_n: Int32, active, q: Int32, c: Int32, mat: Int32, mrow: Int32,
                              acc: cute.Tensor):
        a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
        as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
        w_off = Int64(e) * Int64(self.n) + Int64(n0)
        w_rows = Int64(w.toint()) + w_off * Int64(self.k // 2)
        s_rows = Int64(s.toint()) + w_off * Int64(self.k // 16)
        w_limit = self._w_limit_for(n0, self.n, self.tile_n)
        for pre in cutlass.range_constexpr(self.stages - 1):
            if const_expr(pre < self.blocks):
                self._load_ring1(pre, ring, a_rows, as_rows, w_rows, s_rows, live, w_limit, tidx,
                                 pre < self.full_blocks)
            else:
                cute.arch.cp_async_commit_group()
        if const_expr(self.full_blocks > 0):
            for block in cutlass.range(self.full_blocks, unroll=1):
                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.sync_threads()
                if block + Int32(self.stages - 1) < Int32(self.blocks):
                    self._load_ring1(block + Int32(self.stages - 1), ring, a_rows, as_rows, w_rows, s_rows,
                                     live, w_limit, tidx, False)
                else:
                    cute.arch.cp_async_commit_group()
                if active:
                    stage = ring + (block % Int32(self.stages)) * Int32(self.stage_bytes)
                    self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow, 4)
        if const_expr(self.tail_k > 0):
            b_tail = Int32(self.full_blocks)
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            cute.arch.cp_async_commit_group()
            if active:
                stage = ring + (b_tail % Int32(self.stages)) * Int32(self.stage_bytes)
                self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow, self.tail_steps)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        return ring

    @cute.jit
    def _flush_coalesced_bf16(self, out_smem: Int32, y: cute.Pointer, first: Int32, n0: Int32,
                              live: Int32, tidx: Int32):
        cute.arch.sync_threads()
        r0 = tidx // Int32(16)
        vec = tidx % Int32(16)
        y_base = Int64(y.toint())
        for it in cutlass.range_constexpr(self.tile_m // 16):
            row = r0 + Int32(16 * it)
            if row < live:
                v0, v1, v2, v3 = ld_shared_v4_u32(out_smem + row * Int32(self.out_stride) + vec * Int32(16))
                st_global_v4_u32(
                    y_base + (Int64(first + row) * Int64(self.n) + Int64(n0) + Int64(vec * 8)) * Int64(2),
                    v0, v1, v2, v3)

    @cute.jit
    def _write_linear_a4_epilogue(self, out_smem: Int32, y: cute.Pointer, acc: cute.Tensor,
                                  first: Int32, n0: Int32, live: Int32, tidx: Int32,
                                  warp_m: Int32, warp_n: Int32, active, q: Int32, c: Int32,
                                  factor: Float32, weight_global: Float32, input_global: Float32):
        if const_expr(self.coalesced_epi):
            if active:
                for mt in cutlass.range_constexpr(self.m_tiles):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            f = 4 * (self.n_tiles * mt + nt) + 2 * half
                            col_local = warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                            if const_expr(self.divisor):
                                acc[f] = div_rn_f32(div_rn_f32(acc[f], weight_global), input_global)
                                acc[f + 1] = div_rn_f32(div_rn_f32(acc[f + 1], weight_global), input_global)
                            st_shared_u32(out_smem + row * Int32(self.out_stride) + col_local * Int32(2),
                                          pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
            self._flush_coalesced_bf16(out_smem, y, first, n0, live, tidx)
        else:
            if active:
                for mt in cutlass.range_constexpr(self.m_tiles):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(self.warp_rows) + Int32(16 * mt + 8 * half) + q
                        if row < live:
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                f = 4 * (self.n_tiles * mt + nt) + 2 * half
                                col = n0 + warp_n * Int32(self.warp_cols) + Int32(8 * nt) + Int32(2) * c
                                if const_expr(self.n % self.tile_n == 0):
                                    col_ok = cutlass.Boolean(True)
                                else:
                                    col_ok = col < Int32(self.n)
                                if col_ok:
                                    if const_expr(self.divisor):
                                        acc[f] = div_rn_f32(div_rn_f32(acc[f], weight_global), input_global)
                                        acc[f + 1] = div_rn_f32(div_rn_f32(acc[f + 1], weight_global), input_global)
                                    st_global_u32(
                                        Int64(y.toint()) + (Int64(first + row) * Int64(self.n) + Int64(col)) * Int64(2),
                                        pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                 y: cute.Pointer, max_tiles: Int32, stream: cuda.CUstream):
        grid = (_ceil(self.n, self.tile_n), max_tiles, 1)
        if const_expr(self.use_tma_w):
            w_layout = self._w_layout(self.tile_n, self.w_ring)
            box = cute.slice_(w_layout, (None, None, 0))
            tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                _u8_weight_3d(w, self.n, self.k // 2, self.experts),
                box, (self.tile_n, self.stride), num_multicast=1)
            self.kernel_tma(aq, a_s, meta, tma_tensor_w, s, y, tma_w, w_layout,
                            self._storage_tma(w_layout)).launch(
                grid=grid, block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1)
        else:
            self.kernel_ring(aq, a_s, meta, w, s, y, self._storage_ring()).launch(
                grid=grid, block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=self.min_blocks)

    @cute.kernel
    def kernel_tma(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, weight_w: cute.Tensor,
                   s: cute.Pointer, y: cute.Pointer, tma_w: cute.CopyAtom, w_layout: cute.ComposedLayout,
                   Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                n0 = n_blk * Int32(self.tile_n)
                if const_expr(self.divisor):
                    factor = Float32(1.0)
                    weight_global = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4))
                    input_global = _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                else:
                    weight_global = Float32(1.0)
                    input_global = Float32(1.0)
                    factor = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                        * _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
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
                out_smem = self._run_linear_tma_loop(
                    storage, w_layout, tma_w, weight_w, aq, a_s, s, e, tile, n_blk, n0, live,
                    tidx, warp_id, warp_m, warp_n, active, q, c, mat, mrow, acc)
                self._write_linear_a4_epilogue(
                    out_smem, y, acc, first, n0, live, tidx, warp_m, warp_n, active, q, c,
                    factor, weight_global, input_global)

    @cute.kernel
    def kernel_ring(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w: cute.Pointer, s: cute.Pointer,
                    y: cute.Pointer, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                ring = shared_ptr_to_u32(storage.ring.data_ptr())
                n0 = n_blk * Int32(self.tile_n)
                if const_expr(self.divisor):
                    factor = Float32(1.0)
                    weight_global = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4))
                    input_global = _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
                else:
                    weight_global = Float32(1.0)
                    input_global = Float32(1.0)
                    factor = _ld_f32(Int64(s.toint()) + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                        * _ld_f32(Int64(s.toint()) + Int64(self.in_offset) + Int64(e) * Int64(4))
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
                out_smem = self._run_linear_ring_loop(
                    ring, aq, a_s, w, s, e, tile, n0, live, tidx, warp_m, warp_n, active, q, c, mat, mrow, acc)
                self._write_linear_a4_epilogue(
                    out_smem, y, acc, first, n0, live, tidx, warp_m, warp_n, active, q, c,
                    factor, weight_global, input_global)


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

    Grid ``(ceil(I / half_n), max_tiles)`` for any ``I >= 16, I % 16 == 0`` and
    ``H >= 16, H % 16 == 0`` with zero stored padding."""

    half_n = 64
    x_stride = 2 * 64 * 2 + 16

    def __init__(self, *, inter: int, hidden: int, experts: int, limit: float = 0.0, k_block: int | None = None,
                 stages: int = 2, tile_m: int = 128, tile_n: int = 128,
                 live_min: int = 1, live_max: int = 128, activation: str = "silu_bf16_product",
                 alpha: float | None = None, beta: float | None = None,
                 scale_convention: str = "multiplier"):
        inter = int(inter)
        if inter <= 0 or inter % 16:
            raise ValueError("the fused NVFP4 A4 gate/up needs I % 16 == 0")
        super().__init__(n=2 * inter, k=hidden, experts=experts, tile_m=tile_m, tile_n=tile_n,
                         stages=stages, live_min=live_min, live_max=live_max,
                         scale_convention=scale_convention)
        self.half_n = self.tile_n // 2
        self.half_bytes = self.half_n * self.stride
        self.half_sc_bytes = self.half_n * self.s_stride
        self.x_stride = 2 * self.tile_n + 16
        self.ring_min_bytes = self.tile_m * self.x_stride
        self.inter, self.limit = inter, float(limit)
        self.post_silu = activation == "silu_post_clamp_fp32_product"
        self.oai = activation == "swigluoai"
        self.alpha, self.beta = alpha, beta
        if k_block is not None and int(k_block) != self.k_block:
            raise ValueError(f"canonical K block is {self.k_block}, got {k_block}")
        self.alpha_offset = nvfp4_alpha_offset(experts, self.inter, self.k)
        self.in_offset = nvfp4_input_scale_offset(experts, self.inter, self.k)
        self.down_in_offset = nvfp4_input_scale_offset(experts, hidden, self.inter)

    def key(self) -> tuple:
        return ("stream_gate_up_nvfp4_a4", 6, self.inter, self.k, self.experts, self.k_block, self.stages,
                self.use_tma_w, self.limit, self.down_in_offset, self.tile_m, self.tile_n,
                self.live_min, self.live_max) + (
                    ("silu_post_clamp_fp32_product",) if self.post_silu else ()) + (
                ("swigluoai", self.alpha, self.beta) if self.oai else ()) + (("divisor",) if self.divisor else ())

    @cute.jit
    def _load_tma_w2(self, block, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, n_blk: Int32):
        slot = Int32(block) % Int32(self.w_ring)
        barrier = mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * self.half_bytes)
        cute.copy(tma_g, t_wg[(None, n_blk, Int32(block))], t_ws[(None, slot * Int32(2))], tma_bar_ptr=barrier)
        cute.copy(tma_u, t_wu[(None, n_blk, Int32(block))], t_ws[(None, slot * Int32(2) + Int32(1))],
                  tma_bar_ptr=barrier)

    @cute.jit
    def _load_a_and_scales_tma2(self, block, sa_base: Int32, ssa_base: Int32, ssb_base: Int32,
                                a_rows: Int64, as_rows: Int64, s_rows: Int64, s_up: Int64,
                                live: Int32, w_limit: Int32, tidx: Int32, full_k: cutlass.Constexpr = False):
        slot = Int32(block) % Int32(self.a_ring)
        self._load_packed_rows(sa_base + slot * Int32(self.tile_bytes), a_rows, block, tidx, self.tile_m, live, full_k)
        self._load_scale_rows(ssa_base + slot * Int32(self.sc_bytes), as_rows, block, tidx, self.tile_m, live, full_k)
        sb_slot = ssb_base + slot * Int32(self.weight_sc_bytes)
        self._load_scale_rows(sb_slot, s_rows, block, tidx, self.half_n, w_limit, full_k)
        self._load_scale_rows(sb_slot + Int32(self.half_sc_bytes), s_up, block, tidx, self.half_n, w_limit, full_k)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_ring2(self, block, ring: Int32, a_rows: Int64, as_rows: Int64, w_rows: Int64, w_up: Int64,
                    s_rows: Int64, s_up: Int64, live: Int32, w_limit: Int32, tidx: Int32,
                    full_k: cutlass.Constexpr = False):
        stage = ring + (Int32(block) % Int32(self.stages)) * Int32(self.stage_bytes)
        self._load_operand(stage, a_rows, as_rows, block, tidx, self.tile_m, 0,
                           self.tile_bytes + self.weight_bytes, live, full_k)
        self._load_operand(stage, w_rows, s_rows, block, tidx, self.half_n, self.tile_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes, w_limit, full_k)
        self._load_operand(stage, w_up, s_up, block, tidx, self.half_n,
                           self.tile_bytes + self.half_bytes,
                           self.tile_bytes + self.weight_bytes + self.sc_bytes + self.half_sc_bytes,
                           w_limit, full_k)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _run_gate_up_tma_loop(self, storage, w_layout, tma_g, tma_u, weight_g, weight_u,
                              aq: cute.Pointer, a_s: cute.Pointer, s1: cute.Pointer, s3: cute.Pointer,
                              e: Int32, tile: Int32, n_blk: Int32, n0: Int32, live: Int32,
                              tidx: Int32, warp_id: Int32, warp_m: Int32, warp_n: Int32, active,
                              q: Int32, c: Int32, mat: Int32, mrow: Int32, acc: cute.Tensor):
        mbar = storage.mbar.data_ptr()
        s_w = storage.w.get_tensor(w_layout.outer, swizzle=w_layout.inner)
        sw_base = shared_ptr_to_u32(storage.w.data_ptr())
        sa_base = shared_ptr_to_u32(storage.a.data_ptr())
        ssa_base = shared_ptr_to_u32(storage.sa.data_ptr())
        ssb_base = shared_ptr_to_u32(storage.sb.data_ptr())
        t_ws, t_wg = cpasync.tma_partition(
            tma_g, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
            cute.group_modes(cute.local_tile(weight_g[None, None, e], (self.half_n, self.stride), (None, None)), 0, 2))
        _, t_wu = cpasync.tma_partition(
            tma_u, 0, cute.make_layout(1), cute.group_modes(s_w, 0, 2),
            cute.group_modes(cute.local_tile(weight_u[None, None, e], (self.half_n, self.stride), (None, None)), 0, 2))
        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_g)
            cpasync.prefetch_descriptor(tma_u)
            for slot in cutlass.range_constexpr(self.w_ring):
                cute.arch.mbarrier_init(mbar + slot, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
        as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
        w_off = Int64(e) * Int64(self.inter) + Int64(n0)
        s_rows = Int64(s1.toint()) + w_off * Int64(self.k // 16)
        s_up = Int64(s3.toint()) + w_off * Int64(self.k // 16)
        w_limit = self._w_limit_for(n0, self.inter, self.half_n)
        self._load_a_and_scales_tma2(0, sa_base, ssa_base, ssb_base, a_rows, as_rows, s_rows, s_up,
                                     live, w_limit, tidx, self.full_blocks > 0)
        if warp_id == Int32(0):
            for pre in cutlass.range_constexpr(min(self.w_ring - 1, self.blocks)):
                self._load_tma_w2(pre, tma_g, tma_u, t_wg, t_wu, t_ws, mbar, n_blk)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if const_expr(self.blocks > 1):
            self._load_a_and_scales_tma2(1, sa_base, ssa_base, ssb_base, a_rows, as_rows, s_rows, s_up,
                                         live, w_limit, tidx, self.full_blocks > 1)
        if const_expr(self.full_blocks > 0):
            for block in cutlass.range(self.full_blocks, unroll=1):
                w_slot = block % Int32(self.w_ring)
                cute.arch.mbarrier_wait(mbar + w_slot, (block // Int32(self.w_ring)) & Int32(1))
                if (block + Int32(self.w_ring - 1) < Int32(self.blocks)) & (warp_id == Int32(0)):
                    self._load_tma_w2(block + Int32(self.w_ring - 1), tma_g, tma_u, t_wg, t_wu, t_ws, mbar, n_blk)
                if active:
                    a_slot = block % Int32(self.a_ring)
                    self._mma_pointers(
                        sa_base + a_slot * Int32(self.tile_bytes),
                        sw_base + w_slot * Int32(self.weight_bytes),
                        ssa_base + a_slot * Int32(self.sc_bytes),
                        ssb_base + a_slot * Int32(self.weight_sc_bytes),
                        acc, warp_m, warp_n, q, c, mat, mrow, 4)
                if block + Int32(1) < Int32(self.blocks):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    if block + Int32(2) < Int32(self.blocks):
                        self._load_a_and_scales_tma2(block + Int32(2), sa_base, ssa_base, ssb_base,
                                                     a_rows, as_rows, s_rows, s_up, live, w_limit, tidx, False)
        if const_expr(self.tail_k > 0):
            b_tail = Int32(self.full_blocks)
            w_slot = b_tail % Int32(self.w_ring)
            cute.arch.mbarrier_wait(mbar + w_slot, (b_tail // Int32(self.w_ring)) & Int32(1))
            if active:
                a_slot = b_tail % Int32(self.a_ring)
                self._mma_pointers(
                    sa_base + a_slot * Int32(self.tile_bytes),
                    sw_base + w_slot * Int32(self.weight_bytes),
                    ssa_base + a_slot * Int32(self.sc_bytes),
                    ssb_base + a_slot * Int32(self.weight_sc_bytes),
                    acc, warp_m, warp_n, q, c, mat, mrow, self.tail_steps)
        cute.arch.sync_threads()
        return sw_base

    @cute.jit
    def _run_gate_up_ring_loop(self, ring: Int32, aq: cute.Pointer, a_s: cute.Pointer,
                               w1: cute.Pointer, s1: cute.Pointer, w3: cute.Pointer, s3: cute.Pointer,
                               e: Int32, tile: Int32, n0: Int32, live: Int32, tidx: Int32,
                               warp_m: Int32, warp_n: Int32, active, q: Int32, c: Int32, mat: Int32, mrow: Int32,
                               acc: cute.Tensor):
        a_rows = Int64(aq.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 2)
        as_rows = Int64(a_s.toint()) + Int64(tile) * Int64(STREAM_TILE_M) * Int64(self.k // 16)
        w_off = Int64(e) * Int64(self.inter) + Int64(n0)
        w_rows = Int64(w1.toint()) + w_off * Int64(self.k // 2)
        w_up = Int64(w3.toint()) + w_off * Int64(self.k // 2)
        s_rows = Int64(s1.toint()) + w_off * Int64(self.k // 16)
        s_up = Int64(s3.toint()) + w_off * Int64(self.k // 16)
        w_limit = self._w_limit_for(n0, self.inter, self.half_n)
        for pre in cutlass.range_constexpr(self.stages - 1):
            if const_expr(pre < self.blocks):
                self._load_ring2(pre, ring, a_rows, as_rows, w_rows, w_up, s_rows, s_up,
                                 live, w_limit, tidx, pre < self.full_blocks)
            else:
                cute.arch.cp_async_commit_group()
        if const_expr(self.full_blocks > 0):
            for block in cutlass.range(self.full_blocks, unroll=1):
                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.sync_threads()
                if block + Int32(self.stages - 1) < Int32(self.blocks):
                    self._load_ring2(block + Int32(self.stages - 1), ring, a_rows, as_rows, w_rows, w_up,
                                     s_rows, s_up, live, w_limit, tidx, False)
                else:
                    cute.arch.cp_async_commit_group()
                if active:
                    stage = ring + (block % Int32(self.stages)) * Int32(self.stage_bytes)
                    self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow, 4)
        if const_expr(self.tail_k > 0):
            b_tail = Int32(self.full_blocks)
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            cute.arch.cp_async_commit_group()
            if active:
                stage = ring + (b_tail % Int32(self.stages)) * Int32(self.stage_bytes)
                self._mma_stage(stage, acc, warp_m, warp_n, q, c, mat, mrow, self.tail_steps)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        return ring

    @cute.jit
    def _write_gate_up_a4_epilogue(self, exch_smem: Int32, s1: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer,
                                   act: cute.Pointer, act_s: cute.Pointer, acc: cute.Tensor,
                                   e: Int32, tile: Int32, n0: Int32, live: Int32, tidx: Int32,
                                   warp_m: Int32, warp_n: Int32, active, q: Int32, c: Int32):
        sp = cutlass.select_(warp_n == Int32(0), Int64(s1.toint()), Int64(s3.toint()))
        if const_expr(self.divisor):
            factor = Float32(1.0)
            weight_global = _ld_f32(sp + Int64(self.alpha_offset) + Int64(e) * Int64(4))
            input_global = _ld_f32(sp + Int64(self.in_offset) + Int64(e) * Int64(4))
        else:
            weight_global = Float32(1.0)
            input_global = Float32(1.0)
            factor = _ld_f32(sp + Int64(self.alpha_offset) + Int64(e) * Int64(4)) \
                * _ld_f32(sp + Int64(self.in_offset) + Int64(e) * Int64(4))
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
                        st_shared_u32(exch_smem + row * Int32(self.x_stride) + col * Int32(2),
                                      pack_f32x2_to_bfloat2(acc[f] * factor, acc[f + 1] * factor))
        cute.arch.sync_threads()
        if const_expr(self.divisor):
            gs = _ld_f32(Int64(s2.toint()) + Int64(self.down_in_offset) + Int64(e) * Int64(4))
        else:
            gs = Float32(1.0) / _ld_f32(Int64(s2.toint()) + Int64(self.down_in_offset) + Int64(e) * Int64(4))
        groups = self.half_n // 16
        total_groups = self.inter // 16
        for it in cutlass.range_constexpr(_ceil(self.tile_m * groups, self.threads)):
            gi = tidx + Int32(it * self.threads)
            row = gi // Int32(groups)
            grp = gi % Int32(groups)
            col16_i32 = n0 // Int32(16) + grp
            if const_expr(self.inter % self.half_n == 0):
                grp_ok = row < live
            else:
                grp_ok = (row < live) & (col16_i32 < Int32(total_groups))
            if grp_ok:
                values = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Float32)
                at = exch_smem + row * Int32(self.x_stride) + grp * Int32(32)
                for half in cutlass.range_constexpr(2):
                    gw = ld_shared_v4_u32(at + Int32(16 * half))
                    uw = ld_shared_v4_u32(at + Int32(self.half_n * 2 + 16 * half))
                    _swiglu16(gw, uw, values, 8 * half, self.limit, self.post_silu,
                              self.oai, self.alpha, self.beta)
                i = Int64(tile) * Int64(STREAM_TILE_M) + Int64(row)
                col16 = Int64(col16_i32)
                _quantize_row16(values, gs,
                                Int64(act.toint()) + i * Int64(self.inter // 2) + col16 * Int64(8),
                                Int64(act_s.toint()) + i * Int64(total_groups) + col16)

    @cute.jit
    def __call__(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                 w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer,
                 max_tiles: Int32, stream: cuda.CUstream):
        grid = (_ceil(self.inter, self.half_n), max_tiles, 1)
        if const_expr(self.use_tma_w):
            w_layout = self._w_layout(self.half_n, 2 * self.w_ring)
            box = cute.slice_(w_layout, (None, None, 0))
            tma_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                _u8_weight_3d(w1, self.inter, self.k // 2, self.experts),
                box, (self.half_n, self.stride), num_multicast=1)
            tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                _u8_weight_3d(w3, self.inter, self.k // 2, self.experts),
                box, (self.half_n, self.stride), num_multicast=1)
            self.fused_tma(aq, a_s, meta, tma_tensor_g, s1, tma_tensor_u, s3, s2, act, act_s,
                           tma_g, tma_u, w_layout, self._storage_tma(w_layout)).launch(
                grid=grid, block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=1)
        else:
            self.fused_ring(aq, a_s, meta, w1, s1, w3, s3, s2, act, act_s, self._storage_ring()).launch(
                grid=grid, block=(self.threads, 1, 1), stream=stream, min_blocks_per_mp=self.min_blocks)

    @cute.kernel
    def fused_tma(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, weight_g: cute.Tensor,
                  s1: cute.Pointer, weight_u: cute.Tensor, s3: cute.Pointer, s2: cute.Pointer,
                  act: cute.Pointer, act_s: cute.Pointer, tma_g: cute.CopyAtom, tma_u: cute.CopyAtom,
                  w_layout: cute.ComposedLayout, Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                n0 = n_blk * Int32(self.half_n)
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
                exch_smem = self._run_gate_up_tma_loop(
                    storage, w_layout, tma_g, tma_u, weight_g, weight_u, aq, a_s, s1, s3,
                    e, tile, n_blk, n0, live, tidx, warp_id, warp_m, warp_n, active, q, c, mat, mrow, acc)
                self._write_gate_up_a4_epilogue(
                    exch_smem, s1, s3, s2, act, act_s, acc, e, tile, n0, live, tidx, warp_m, warp_n, active, q, c)

    @cute.kernel
    def fused_ring(self, aq: cute.Pointer, a_s: cute.Pointer, meta: cute.Pointer, w1: cute.Pointer, s1: cute.Pointer,
                   w3: cute.Pointer, s3: cute.Pointer, s2: cute.Pointer, act: cute.Pointer, act_s: cute.Pointer,
                   Storage: cutlass.Constexpr):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        if tile < _i32_at(Int64(meta.toint()), 1):
            e, first, live = _chunk(meta, self.experts, tile)
            if (live >= Int32(self.live_min)) & (live <= Int32(self.live_max)):
                tidx = Int32(cute.arch.thread_idx()[0])
                warp_id = tidx // Int32(32)
                lane = tidx % Int32(32)
                ring = shared_ptr_to_u32(storage.ring.data_ptr())
                n0 = n_blk * Int32(self.half_n)
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
                exch_smem = self._run_gate_up_ring_loop(
                    ring, aq, a_s, w1, s1, w3, s3, e, tile, n0, live, tidx, warp_m, warp_n, active, q, c, mat, mrow, acc)
                self._write_gate_up_a4_epilogue(
                    exch_smem, s1, s3, s2, act, act_s, acc, e, tile, n0, live, tidx, warp_m, warp_n, active, q, c)
