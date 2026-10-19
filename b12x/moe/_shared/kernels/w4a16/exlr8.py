"""EXLR8 A16 unit experts, implemented from the published FORMAT.md.

Pointers are I64 [expert, unit, projection, payload/channel/block]. Widths
are I32 [expert, unit, projection]; zero marks a nonresident unit. Payloads
retain native gate/up grouped and down item128 order. The F16 input is already
in the rotated hidden domain (the grouping step applies H128/signs.in once per
token). The final stage sums the local experts' F32 partials in fixed slot
order, applies the inverse H128 and signs.out once per row and rounds to BF16.
All scratch is caller-owned, and live row/group counts and the device-derived
CTA limit are launch scalars, never compile keys.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    f16_mma_m16n8k16_f32,
    half2_to_float2_scaled,
    ld_global_b16,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    packed_dequant_trellis_to_half2x4,
    u32_as_f32,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


@dsl_user_op
def _e4m3_round_half2(value: Uint32, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(), [value.ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 e; cvt.rn.satfinite.e4m3x2.f16x2 e, $1; "
        "cvt.rn.f16x2.e4m3x2 $0, e; }",
        "=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def store_bf16x4(address: Int64, v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                 *, loc=None, ip=None):
    """Round four F32 values once to BF16 (nearest even, no saturation, as a
    torch cast) and store them as one 8-byte vector."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip), Float32(v2).ir_value(loc=loc, ip=ip),
         Float32(v3).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 lo, hi; cvt.rn.bf16x2.f32 lo, $2, $1; cvt.rn.bf16x2.f32 hi, $4, $3; "
        "st.global.v2.b32 [$0], {lo, hi}; }",
        "l,f,f,f,f", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.jit
def signed_byte(word: Uint32, index: cutlass.Constexpr):
    """Byte ``index`` of a little-endian word as a signed I8 value in F32."""
    shifted = (word << Uint32(24 - 8 * index)).to(Int32)
    return (shifted >> Int32(24)).to(Float32)


@cute.jit
def exlr8_decode_tile(payload_address: Int64, tile: Int32, lane: Int32,
                      bits: cutlass.Constexpr):
    """Return native t256 B fragments, with mandatory fp16 -> E4M3 -> fp16."""
    base = payload_address + tile.to(Int64) * Int64(32 * bits)
    if cutlass.const_expr(bits == 4):
        b = ld_global_nc_u32(base + lane.to(Int64) * Int64(4))
        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
        wa = b
        wb = (a << Uint32(16)) | (b >> Uint32(16))
    else:
        ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)
        a = ld_global_nc_u32(base + ia.to(Int64) * Int64(4))
        b = ld_global_nc_u32(base + ib.to(Int64) * Int64(4))
        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
        wa = (window >> shift.to(Int64)).to(Uint32)
        wb = (window >> (shift + Int32(4 * bits)).to(Int64)).to(Uint32)
    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, bits)
    return (_e4m3_round_half2(b0), _e4m3_round_half2(b1),
            _e4m3_round_half2(b2), _e4m3_round_half2(b3))


@cute.jit
def exlr8_decode_tile_width(payload_address: Int64, tile: Int32, lane: Int32, width: Int32):
    """Runtime bitwidth dispatch returning native t256 B fragments by value."""
    b0, b1, b2, b3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
    if width == Int32(4):
        b0, b1, b2, b3 = exlr8_decode_tile(payload_address, tile, lane, 4)
    elif width == Int32(3):
        b0, b1, b2, b3 = exlr8_decode_tile(payload_address, tile, lane, 3)
    elif width == Int32(2):
        b0, b1, b2, b3 = exlr8_decode_tile(payload_address, tile, lane, 2)
    return b0, b1, b2, b3


@cute.jit
def _side_value(address: Int64, index: Int32):
    packed = ld_global_b16(address + index.to(Int64) * Int64(2))
    lo, _ = half2_to_float2_scaled(packed, Float32(1.0))
    return lo


class Exlr8A16Kernel:
    ABI_VERSION = 3
    CODE_VERSION = 4
    # Resident CTAs per SM the stream kernels are built for (<=128 registers at
    # 128 threads). GB10 decode is latency-bound on dependent weight loads, so
    # the caller's cta_limit should be sm_count * CTAS_PER_SM (cudaDevAttr at
    # runtime); it never changes results, only how many tasks run at once.
    CTAS_PER_SM = 4
    PREFILL_CTAS_PER_SM = 2
    PREFILL_M_GROUPS = 4

    def __init__(self, hidden_size: int, units_per_expert: int, top_k: int,
                 max_groups: int, fast_math: bool = True, device_count: bool = False,
                 max_rows: int = 64, route: str = "auto"):
        if hidden_size <= 0 or hidden_size % 128:
            raise ValueError("EXLR8 hidden_size must be a positive multiple of 128")
        if min(units_per_expert, top_k, max_groups, max_rows) < 1:
            raise ValueError("EXLR8 units, top_k, max_rows and group capacity must be positive")
        if route not in ("auto", "decode", "prefill"):
            raise ValueError(f"unknown EXLR8 route {route!r}")
        self.hidden_size = int(hidden_size)
        self.units = int(units_per_expert)
        self.top_k = int(top_k)
        self.max_groups = int(max_groups)
        self.fast_math = bool(fast_math)
        self.device_count = bool(device_count)
        self.max_rows = int(max_rows)
        self.route = str(route)
        self.blocks = hidden_size // 128
        self.splits = (self.blocks + 7) // 8

    @property
    def __cache_key__(self):
        if self.max_rows == 64 and self.route == "auto":
            return ("exlr8_a16", self.ABI_VERSION, self.CODE_VERSION, self.hidden_size, self.units,
                    self.top_k, self.max_groups, self.fast_math, self.device_count)
        return ("exlr8_a16", self.ABI_VERSION, self.CODE_VERSION, self.hidden_size, self.units,
                self.top_k, self.max_groups, self.fast_math, self.device_count,
                self.max_rows, self.route)

    @cute.jit
    def _launch_decode(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                       group_expert: cute.Tensor, route_slots: cute.Tensor,
                       route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                       fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                       signs_out: cute.Tensor, output: cute.Tensor, rows: Int32, groups: Int32,
                       cta_limit: Int32, stream: cuda.CUstream):
        if groups > Int32(0):
            self.fc1_kernel(x, pointers, bits, group_expert, route_slots,
                            fc1, rows, groups).launch(
                grid=(min(groups * Int32(self.units * 2 * self.splits), cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
            self.activation_kernel(pointers, bits, group_expert, route_slots,
                                   fc1, intermediate, rows, groups).launch(
                grid=(min(groups * Int32(self.units), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)
            self.fc2_kernel(pointers, bits, group_expert, route_slots, intermediate,
                            fc2, rows, groups).launch(
                grid=(min(groups * Int32(self.blocks), cta_limit), 1, 1),
                block=(128, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
        self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                           rows, groups).launch(
            grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)

    @cute.jit
    def _launch_prefill(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                        group_expert: cute.Tensor, route_slots: cute.Tensor,
                        route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                        fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                        signs_out: cute.Tensor, output: cute.Tensor, rows: Int32, groups: Int32,
                        cta_limit: Int32, stream: cuda.CUstream):
        if groups > Int32(0):
            self.prefill_fc1_kernel(x, pointers, bits, group_expert, route_slots,
                                    fc1, rows, groups).launch(
                grid=(min(groups * Int32(self.units * 2), cta_limit), 1, 1),
                block=(256, 1, 1), min_blocks_per_mp=self.PREFILL_CTAS_PER_SM, stream=stream)
            self.prefill_activation_kernel(pointers, bits, group_expert, route_slots,
                                           fc1, intermediate, rows, groups).launch(
                grid=(min(groups * Int32(self.units), cta_limit), 1, 1),
                block=(128, 1, 1), stream=stream)
            self.prefill_fc2_kernel(pointers, bits, group_expert, route_slots, intermediate,
                                    fc2, rows, groups).launch(
                grid=(min(groups * Int32(self.blocks), cta_limit), 1, 1),
                block=(256, 1, 1), min_blocks_per_mp=self.PREFILL_CTAS_PER_SM, stream=stream)
        self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                           rows, groups).launch(
            grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)

    @cute.jit
    def _launch_serving_decode(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                               group_expert: cute.Tensor, route_slots: cute.Tensor,
                               route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                               fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                               signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                               group_count: cute.Tensor, cta_limit: Int32, stream: cuda.CUstream):
        self.fc1_kernel(x, pointers, bits, group_expert, route_slots,
                        fc1, rows, group_count).launch(
            grid=(min(Int32(self.max_groups * self.units * 2 * self.splits), cta_limit), 1, 1),
            block=(128, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
        self.activation_kernel(pointers, bits, group_expert, route_slots,
                               fc1, intermediate, rows, group_count).launch(
            grid=(min(Int32(self.max_groups * self.units), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)
        self.fc2_kernel(pointers, bits, group_expert, route_slots, intermediate,
                        fc2, rows, group_count).launch(
            grid=(min(Int32(self.max_groups * self.blocks), cta_limit), 1, 1),
            block=(128, 1, 1), min_blocks_per_mp=self.CTAS_PER_SM, stream=stream)
        self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                           rows, group_count).launch(
            grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)

    @cute.jit
    def _launch_serving_prefill(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                                group_expert: cute.Tensor, route_slots: cute.Tensor,
                                route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                                fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                                signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                                group_count: cute.Tensor, cta_limit: Int32, stream: cuda.CUstream):
        self.prefill_fc1_kernel(x, pointers, bits, group_expert, route_slots,
                                fc1, rows, group_count).launch(
            grid=(min(Int32(self.max_groups * self.units * 2), cta_limit), 1, 1),
            block=(256, 1, 1), min_blocks_per_mp=self.PREFILL_CTAS_PER_SM, stream=stream)
        self.prefill_activation_kernel(pointers, bits, group_expert, route_slots,
                                       fc1, intermediate, rows, group_count).launch(
            grid=(min(Int32(self.max_groups * self.units), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)
        self.prefill_fc2_kernel(pointers, bits, group_expert, route_slots, intermediate,
                                fc2, rows, group_count).launch(
            grid=(min(Int32(self.max_groups * self.blocks), cta_limit), 1, 1),
            block=(256, 1, 1), min_blocks_per_mp=self.PREFILL_CTAS_PER_SM, stream=stream)
        self.reduce_kernel(route_group_lane, route_weights, fc2, signs_out, output,
                           rows, group_count).launch(
            grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
            block=(128, 1, 1), stream=stream)

    @cute.jit
    def __call__(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                 group_expert: cute.Tensor, route_slots: cute.Tensor,
                 route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                 fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                 signs_out: cute.Tensor, output: cute.Tensor, rows: Int32, groups: Int32,
                 cta_limit: Int32, stream: cuda.CUstream):
        if cutlass.const_expr(self.route == "decode" or (self.route == "auto" and self.max_rows <= 64)):
            if rows > Int32(0) and rows <= Int32(self.max_rows) and groups >= Int32(0) and groups <= Int32(self.max_groups) and cta_limit > Int32(0):
                self._launch_decode(x, pointers, bits, group_expert, route_slots,
                                    route_group_lane, route_weights, fc1, intermediate,
                                    fc2, signs_out, output, rows, groups, cta_limit, stream)
        elif cutlass.const_expr(self.route == "prefill"):
            if rows > Int32(0) and rows <= Int32(self.max_rows) and groups >= Int32(0) and groups <= Int32(self.max_groups) and cta_limit > Int32(0):
                self._launch_prefill(x, pointers, bits, group_expert, route_slots,
                                     route_group_lane, route_weights, fc1, intermediate,
                                     fc2, signs_out, output, rows, groups, cta_limit, stream)
        else:
            if rows > Int32(0) and rows <= Int32(64) and groups >= Int32(0) and groups <= Int32(self.max_groups) and cta_limit > Int32(0):
                self._launch_decode(x, pointers, bits, group_expert, route_slots,
                                    route_group_lane, route_weights, fc1, intermediate,
                                    fc2, signs_out, output, rows, groups, cta_limit, stream)
            elif rows > Int32(64) and rows <= Int32(self.max_rows) and groups >= Int32(0) and groups <= Int32(self.max_groups) and cta_limit > Int32(0):
                self._launch_prefill(x, pointers, bits, group_expert, route_slots,
                                     route_group_lane, route_weights, fc1, intermediate,
                                     fc2, signs_out, output, rows, groups, cta_limit, stream)

    @cute.jit
    def serving(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                group_expert: cute.Tensor, route_slots: cute.Tensor,
                route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                group_count: cute.Tensor, cta_limit: Int32, stream: cuda.CUstream):
        # Grids and launch count are independent of route data, including zero
        # groups. Each kernel bounds its tasks with the device-resident count.
        if cutlass.const_expr(self.route == "decode" or (self.route == "auto" and self.max_rows <= 64)):
            if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
                self._launch_serving_decode(x, pointers, bits, group_expert, route_slots,
                                            route_group_lane, route_weights, fc1, intermediate,
                                            fc2, signs_out, output, rows, group_count, cta_limit, stream)
        elif cutlass.const_expr(self.route == "prefill"):
            if rows > Int32(0) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
                self._launch_serving_prefill(x, pointers, bits, group_expert, route_slots,
                                             route_group_lane, route_weights, fc1, intermediate,
                                             fc2, signs_out, output, rows, group_count, cta_limit, stream)
        else:
            if rows > Int32(0) and rows <= Int32(64) and cta_limit > Int32(0):
                self._launch_serving_decode(x, pointers, bits, group_expert, route_slots,
                                            route_group_lane, route_weights, fc1, intermediate,
                                            fc2, signs_out, output, rows, group_count, cta_limit, stream)
            elif rows > Int32(64) and rows <= Int32(self.max_rows) and cta_limit > Int32(0):
                self._launch_serving_prefill(x, pointers, bits, group_expert, route_slots,
                                             route_group_lane, route_weights, fc1, intermediate,
                                             fc2, signs_out, output, rows, group_count, cta_limit, stream)

    @cute.jit
    def _mma_block(self, payload: Int64, a_address: Int64,
                   row0: Int32, row1: Int32, a_stride: cutlass.Constexpr,
                   input_block: Int32, output_block: Int32, lane: Int32,
                   warp: Int32, width: cutlass.Constexpr, down: cutlass.Constexpr):
        """One K128 block, four N8 fragments per lane, FP32 accumulation."""
        acc = cute.make_rmem_tensor((4, 4), Float32)
        acc.fill(Float32(0.0))
        for k16 in cutlass.range_constexpr(8):
            k = input_block * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
            a0 = Uint32(0)
            a1 = Uint32(0)
            a2 = Uint32(0)
            a3 = Uint32(0)
            if row0 >= Int32(0):
                addr0 = a_address + (row0.to(Int64) * Int64(a_stride) + k.to(Int64)) * Int64(2)
                a0 = ld_global_nc_u32(addr0)
                a2 = ld_global_nc_u32(addr0 + Int64(16))
            if row1 >= Int32(0):
                addr1 = a_address + (row1.to(Int64) * Int64(a_stride) + k.to(Int64)) * Int64(2)
                a1 = ld_global_nc_u32(addr1)
                a3 = ld_global_nc_u32(addr1 + Int64(16))
            for n16 in cutlass.range_constexpr(2):
                tile = (input_block * Int32(8) + Int32(k16)) * Int32(8) + warp * Int32(2) + Int32(n16)
                if cutlass.const_expr(down):
                    tile = output_block * Int32(64) + Int32(k16 * 8) + warp * Int32(2) + Int32(n16)
                b0, b1, b2, b3 = exlr8_decode_tile(payload, tile, lane, width)
                d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                    acc[2*n16, 0], acc[2*n16, 1], acc[2*n16, 2], acc[2*n16, 3],
                    a0, a1, a2, a3, b0, b1)
                acc[2*n16, 0], acc[2*n16, 1] = d0, d1
                acc[2*n16, 2], acc[2*n16, 3] = d2, d3
                d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                    acc[2*n16+1, 0], acc[2*n16+1, 1], acc[2*n16+1, 2], acc[2*n16+1, 3],
                    a0, a1, a2, a3, b2, b3)
                acc[2*n16+1, 0], acc[2*n16+1, 1] = d0, d1
                acc[2*n16+1, 2], acc[2*n16+1, 3] = d2, d3
        return acc

    @cute.jit
    def _accumulate_block(self, total: cute.Tensor, payload: Int64, a_address: Int64,
                          row0: Int32, row1: Int32, a_stride: cutlass.Constexpr,
                          input_block: Int32, output_block: Int32, lane: Int32,
                          warp: Int32, width: cutlass.Constexpr,
                          down: cutlass.Constexpr, scale: Float32):
        acc = self._mma_block(payload, a_address, row0, row1, a_stride,
                              input_block, output_block, lane, warp, width, down)
        for n in cutlass.range_constexpr(4):
            for v in cutlass.range_constexpr(4):
                total[n, v] = total[n, v] + acc[n, v] * scale

    @cute.kernel
    def fc1_kernel(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                   group_expert: cute.Tensor, route_slots: cute.Tensor,
                   fc1: cute.Tensor, rows: Int32, groups):
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            # Invalid counts never index descriptor or scratch storage.
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        while task < groups * Int32(self.units * 2 * self.splits):
            split = task % Int32(self.splits)
            proj = (task // Int32(self.splits)) % Int32(2)
            unit = (task // Int32(2 * self.splits)) % Int32(self.units)
            group = task // Int32(self.units * 2 * self.splits)
            expert = group_expert[group]
            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
            width = bits[desc]
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            r = lane // Int32(4)
            slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
            slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
            row0, row1 = Int32(-1), Int32(-1)
            if slot0 >= Int32(0) and slot0 < rows * Int32(self.top_k):
                row0 = slot0 // Int32(self.top_k)
            if slot1 >= Int32(0) and slot1 < rows * Int32(self.top_k):
                row1 = slot1 // Int32(self.top_k)
            if width != Int32(0):
                payload = pointers[desc * Int64(3)]
                scales = pointers[desc * Int64(3) + Int64(2)]
                for block in cutlass.range(8):
                    kb = split * Int32(8) + block
                    if kb < Int32(self.blocks):
                        scale = _side_value(scales, kb)
                        if width == Int32(2):
                            self._accumulate_block(total, payload, x.iterator.toint(), row0, row1,
                                self.hidden_size, kb, Int32(0), lane, warp, 2, False, scale)
                        elif width == Int32(3):
                            self._accumulate_block(total, payload, x.iterator.toint(), row0, row1,
                                self.hidden_size, kb, Int32(0), lane, warp, 3, False, scale)
                        elif width == Int32(4):
                            self._accumulate_block(total, payload, x.iterator.toint(), row0, row1,
                                self.hidden_size, kb, Int32(0), lane, warp, 4, False, scale)
            for n in cutlass.range_constexpr(4):
                col = warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                base = task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)
                if width != Int32(0) and row0 >= Int32(0):
                    fc1[base], fc1[base + Int64(1)] = total[n, 0], total[n, 1]
                if width != Int32(0) and row1 >= Int32(0):
                    fc1[base + Int64(8 * 128)], fc1[base + Int64(8 * 128 + 1)] = total[n, 2], total[n, 3]
            task += step

    @cute.kernel
    def activation_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                          group_expert: cute.Tensor, route_slots: cute.Tensor,
                          fc1: cute.Tensor, intermediate: cute.Tensor,
                          rows: Int32, groups):
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            # Invalid counts never index descriptor or scratch storage.
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        while task < groups * Int32(self.units):
            group, unit = task // Int32(self.units), task % Int32(self.units)
            expert = group_expert[group]
            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)
            for iteration in cutlass.range_constexpr(4):
                row = warp + Int32(iteration * 4)
                slot = route_slots[group.to(Int64) * Int64(16) + row.to(Int64)]
                g = cute.make_rmem_tensor(4, Float32)
                u = cute.make_rmem_tensor(4, Float32)
                g.fill(Float32(0.0))
                u.fill(Float32(0.0))
                if bits[desc] != Int32(0) and slot >= Int32(0) and slot < rows * Int32(self.top_k):
                    for split in cutlass.range_constexpr(self.splits):
                        base = (task.to(Int64) * Int64(2 * self.splits) + Int64(split)) * Int64(16 * 128) + row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
                        for v in cutlass.range_constexpr(4):
                            g[v] = g[v] + fc1[base + Int64(v)]
                            u[v] = u[v] + fc1[base + Int64(self.splits * 16 * 128 + v)]
                    gh = _w4a8_had128_quad(g[0], g[1], g[2], g[3], lane)
                    uh = _w4a8_had128_quad(u[0], u[1], u[2], u[3], lane)
                    gs = pointers[desc * Int64(3) + Int64(1)]
                    us = pointers[(desc + Int64(1)) * Int64(3) + Int64(1)]
                    ds = pointers[(desc + Int64(2)) * Int64(3) + Int64(1)]
                    for v in cutlass.range_constexpr(4):
                        col = lane * Int32(4) + Int32(v)
                        gate = gh[v] * _side_value(gs, col)
                        up = uh[v] * _side_value(us, col)
                        sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                        g[v] = gate * sigmoid * up * _side_value(ds, col)
                    g[0], g[1], g[2], g[3] = _w4a8_had128_quad(g[0], g[1], g[2], g[3], lane)
                    base = (task.to(Int64) * Int64(16) + row.to(Int64)) * Int64(128) + lane.to(Int64) * Int64(4)
                    for v in cutlass.range_constexpr(4):
                        intermediate[base + Int64(v)] = g[v].to(cutlass.Float16)
            task += step

    @cute.kernel
    def fc2_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                   group_expert: cute.Tensor, route_slots: cute.Tensor,
                   intermediate: cute.Tensor, fc2: cute.Tensor,
                   rows: Int32, groups):
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            # Invalid counts never index descriptor or scratch storage.
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        while task < groups * Int32(self.blocks):
            group, block = task // Int32(self.blocks), task % Int32(self.blocks)
            expert = group_expert[group]
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            r = lane // Int32(4)
            slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
            slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
            row0, row1 = Int32(-1), Int32(-1)
            if slot0 >= Int32(0) and slot0 < rows * Int32(self.top_k):
                row0 = r
            if slot1 >= Int32(0) and slot1 < rows * Int32(self.top_k):
                row1 = r + Int32(8)
            for unit in cutlass.range(self.units):
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                width = bits[desc]
                if width != Int32(0):
                    payload = pointers[desc * Int64(3)]
                    scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                    addr = intermediate.iterator.toint() + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                    if width == Int32(2):
                        self._accumulate_block(total, payload, addr, row0, row1, 128,
                            Int32(0), block, lane, warp, 2, True, scale)
                    elif width == Int32(3):
                        self._accumulate_block(total, payload, addr, row0, row1, 128,
                            Int32(0), block, lane, warp, 3, True, scale)
                    elif width == Int32(4):
                        self._accumulate_block(total, payload, addr, row0, row1, 128,
                            Int32(0), block, lane, warp, 4, True, scale)
            for n in cutlass.range_constexpr(4):
                col = block * Int32(128) + warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                base = (group.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col.to(Int64)
                if row0 >= Int32(0):
                    fc2[base], fc2[base + Int64(1)] = total[n, 0], total[n, 1]
                if row1 >= Int32(0):
                    fc2[base + Int64(8 * self.hidden_size)], fc2[base + Int64(8 * self.hidden_size + 1)] = total[n, 2], total[n, 3]
            task += step

    @cute.jit
    def _build_super_groups(self, group_expert: cute.Tensor, smem_sg: cute.Tensor,
                            smem_nsg: cute.Tensor, groups: Int32, tid: Int32):
        g, nsg, exp, gl = Int32(0), Int32(0), Int32(0), Int32(0)
        if tid == Int32(0):
            while g < groups:
                exp = group_expert[g]
                gl = Int32(1)
                if g + Int32(1) < groups:
                    if group_expert[g + Int32(1)] == exp:
                        gl = Int32(2)
                        if g + Int32(2) < groups:
                            if group_expert[g + Int32(2)] == exp:
                                gl = Int32(3)
                                if g + Int32(3) < groups:
                                    if group_expert[g + Int32(3)] == exp:
                                        gl = Int32(4)
                smem_sg[nsg] = (g << Int32(8)) | gl
                nsg += Int32(1)
                g += gl
            smem_nsg[0] = nsg
        cute.arch.sync_threads()
        return smem_nsg[0]

    @cute.kernel
    def prefill_fc1_kernel(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           fc1: cute.Tensor, rows: Int32, groups):
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        smem = utils.SmemAllocator()
        smem_sg = smem.allocate_tensor(Int32, cute.make_layout(self.max_groups), byte_alignment=16)
        smem_nsg = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        smem_sum = smem.allocate_tensor(
            Float32, cute.make_layout((16, 256), stride=(256, 1)), byte_alignment=16
        )
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        nsg = self._build_super_groups(group_expert, smem_sg, smem_nsg, groups, Int32(tid))
        x_address = x.iterator.toint()
        while task < nsg * Int32(self.units * 2):
            proj = task % Int32(2)
            unit = (task // Int32(2)) % Int32(self.units)
            s = task // Int32(self.units * 2)
            packed_sg = smem_sg[s]
            g0 = packed_sg >> Int32(8)
            gl = packed_sg & Int32(0xFF)
            expert = group_expert[g0]
            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
            width = bits[desc]
            r = lane // Int32(4)
            row0 = cute.make_rmem_tensor(4, Int32)
            row1 = cute.make_rmem_tensor(4, Int32)
            for m in cutlass.range_constexpr(4):
                row0[m] = Int32(-1)
                row1[m] = Int32(-1)
                if Int32(m) < gl:
                    gm = g0 + Int32(m)
                    slot0 = route_slots[gm.to(Int64) * Int64(16) + r.to(Int64)]
                    slot1 = route_slots[gm.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                    if slot0 >= Int32(0) and slot0 < rows * Int32(self.top_k):
                        row0[m] = slot0 // Int32(self.top_k)
                    if slot1 >= Int32(0) and slot1 < rows * Int32(self.top_k):
                        row1[m] = slot1 // Int32(self.top_k)
            for idx in cutlass.range_constexpr(16):
                smem_sum[idx, tid] = Float32(0.0)
            cute.arch.sync_threads()
            seg_sum = cute.make_rmem_tensor((2, 2, 4), Float32)
            seg_sum.fill(Float32(0.0))
            total = cute.make_rmem_tensor((4, 2, 4), Float32)
            total.fill(Float32(0.0))
            for kb in cutlass.range(self.splits * 8):
                if (kb & Int32(7)) == Int32(0):
                    for m in cutlass.range_constexpr(4):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                total[m, n, v] = Float32(0.0)
                if width != Int32(0) and kb < Int32(self.blocks):
                    payload = pointers[desc * Int64(3)]
                    scale = _side_value(pointers[desc * Int64(3) + Int64(2)], kb)
                    acc = cute.make_rmem_tensor((4, 2, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = kb * Int32(128) + Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        tile = (kb * Int32(8) + Int32(k16)) * Int32(8) + warp
                        b0, b1, b2, b3 = exlr8_decode_tile_width(payload, tile, lane, width)
                        for m in cutlass.range_constexpr(4):
                            a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                            r0 = row0[m]
                            r1 = row1[m]
                            if r0 >= Int32(0):
                                addr0 = x_address + (r0.to(Int64) * Int64(self.hidden_size) + k.to(Int64)) * Int64(2)
                                a0 = ld_global_nc_u32(addr0)
                                a2 = ld_global_nc_u32(addr0 + Int64(16))
                            if r1 >= Int32(0):
                                addr1 = x_address + (r1.to(Int64) * Int64(self.hidden_size) + k.to(Int64)) * Int64(2)
                                a1 = ld_global_nc_u32(addr1)
                                a3 = ld_global_nc_u32(addr1 + Int64(16))
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[m, 0, 0], acc[m, 0, 1], acc[m, 0, 2], acc[m, 0, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[m, 0, 0], acc[m, 0, 1] = d0, d1
                            acc[m, 0, 2], acc[m, 0, 3] = d2, d3
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[m, 1, 0], acc[m, 1, 1], acc[m, 1, 2], acc[m, 1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[m, 1, 0], acc[m, 1, 1] = d0, d1
                            acc[m, 1, 2], acc[m, 1, 3] = d2, d3
                    for m in cutlass.range_constexpr(4):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                total[m, n, v] = total[m, n, v] + acc[m, n, v] * scale
                if (kb & Int32(7)) == Int32(7):
                    cute.arch.sync_threads()
                    for m in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                idx = m * 8 + n * 4 + v
                                smem_sum[idx, tid] = smem_sum[idx, tid] + total[m, n, v]
                    for m in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                seg_sum[m, n, v] = seg_sum[m, n, v] + total[m + 2, n, v]
            for m in cutlass.range_constexpr(4):
                if Int32(m) < gl:
                    gm = g0 + Int32(m)
                    r0 = row0[m]
                    r1 = row1[m]
                    tile_idx = (gm.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(2) + proj.to(Int64)
                    for n in cutlass.range_constexpr(2):
                        col = warp * Int32(16) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                        base = tile_idx * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)
                        if cutlass.const_expr(m < 2):
                            idx0 = m * 8 + n * 4
                            s0, s1 = smem_sum[idx0, tid], smem_sum[idx0 + 1, tid]
                            s2, s3 = smem_sum[idx0 + 2, tid], smem_sum[idx0 + 3, tid]
                        else:
                            s0, s1 = seg_sum[m - 2, n, 0], seg_sum[m - 2, n, 1]
                            s2, s3 = seg_sum[m - 2, n, 2], seg_sum[m - 2, n, 3]
                        if width != Int32(0) and r0 >= Int32(0):
                            fc1[base], fc1[base + Int64(1)] = s0, s1
                        if width != Int32(0) and r1 >= Int32(0):
                            fc1[base + Int64(8 * 128)], fc1[base + Int64(8 * 128 + 1)] = s2, s3
            task += step

    @cute.kernel
    def prefill_activation_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                                  group_expert: cute.Tensor, route_slots: cute.Tensor,
                                  fc1: cute.Tensor, intermediate: cute.Tensor,
                                  rows: Int32, groups):
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        while task < groups * Int32(self.units):
            group, unit = task // Int32(self.units), task % Int32(self.units)
            expert = group_expert[group]
            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)
            for iteration in cutlass.range_constexpr(4):
                row = warp + Int32(iteration * 4)
                slot = route_slots[group.to(Int64) * Int64(16) + row.to(Int64)]
                g = cute.make_rmem_tensor(4, Float32)
                u = cute.make_rmem_tensor(4, Float32)
                g.fill(Float32(0.0))
                u.fill(Float32(0.0))
                if bits[desc] != Int32(0) and slot >= Int32(0) and slot < rows * Int32(self.top_k):
                    base = (task.to(Int64) * Int64(2)) * Int64(16 * 128) + row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
                    for v in cutlass.range_constexpr(4):
                        g[v] = fc1[base + Int64(v)]
                        u[v] = fc1[base + Int64(16 * 128 + v)]
                    gh = _w4a8_had128_quad(g[0], g[1], g[2], g[3], lane)
                    uh = _w4a8_had128_quad(u[0], u[1], u[2], u[3], lane)
                    gs = pointers[desc * Int64(3) + Int64(1)]
                    us = pointers[(desc + Int64(1)) * Int64(3) + Int64(1)]
                    ds = pointers[(desc + Int64(2)) * Int64(3) + Int64(1)]
                    for v in cutlass.range_constexpr(4):
                        col = lane * Int32(4) + Int32(v)
                        gate = gh[v] * _side_value(gs, col)
                        up = uh[v] * _side_value(us, col)
                        sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                        g[v] = gate * sigmoid * up * _side_value(ds, col)
                    g[0], g[1], g[2], g[3] = _w4a8_had128_quad(g[0], g[1], g[2], g[3], lane)
                    base = (task.to(Int64) * Int64(16) + row.to(Int64)) * Int64(128) + lane.to(Int64) * Int64(4)
                    for v in cutlass.range_constexpr(4):
                        intermediate[base + Int64(v)] = g[v].to(cutlass.Float16)
            task += step

    @cute.kernel
    def prefill_fc2_kernel(self, pointers: cute.Tensor, bits: cute.Tensor,
                           group_expert: cute.Tensor, route_slots: cute.Tensor,
                           intermediate: cute.Tensor, fc2: cute.Tensor,
                           rows: Int32, groups):
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        smem = utils.SmemAllocator()
        smem_sg = smem.allocate_tensor(Int32, cute.make_layout(self.max_groups), byte_alignment=16)
        smem_nsg = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        tid, _, _ = cute.arch.thread_idx()
        task, _, _ = cute.arch.block_idx()
        step, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        nsg = self._build_super_groups(group_expert, smem_sg, smem_nsg, groups, Int32(tid))
        inter_address = intermediate.iterator.toint()
        while task < nsg * Int32(self.blocks):
            s = task // Int32(self.blocks)
            block = task % Int32(self.blocks)
            packed_sg = smem_sg[s]
            g0 = packed_sg >> Int32(8)
            gl = packed_sg & Int32(0xFF)
            expert = group_expert[g0]
            total = cute.make_rmem_tensor((4, 2, 4), Float32)
            total.fill(Float32(0.0))
            r = lane // Int32(4)
            row0 = cute.make_rmem_tensor(4, Int32)
            row1 = cute.make_rmem_tensor(4, Int32)
            for m in cutlass.range_constexpr(4):
                row0[m] = Int32(-1)
                row1[m] = Int32(-1)
                if Int32(m) < gl:
                    gm = g0 + Int32(m)
                    slot0 = route_slots[gm.to(Int64) * Int64(16) + r.to(Int64)]
                    slot1 = route_slots[gm.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                    if slot0 >= Int32(0) and slot0 < rows * Int32(self.top_k):
                        row0[m] = r
                    if slot1 >= Int32(0) and slot1 < rows * Int32(self.top_k):
                        row1[m] = r + Int32(8)
            for unit in cutlass.range(self.units):
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                width = bits[desc]
                if width != Int32(0):
                    payload = pointers[desc * Int64(3)]
                    scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                    acc = cute.make_rmem_tensor((4, 2, 4), Float32)
                    acc.fill(Float32(0.0))
                    for k16 in cutlass.range_constexpr(8):
                        k = Int32(k16 * 16) + (lane & Int32(3)) * Int32(2)
                        tile = block * Int32(64) + Int32(k16 * 8) + warp
                        b0, b1, b2, b3 = exlr8_decode_tile_width(payload, tile, lane, width)
                        for m in cutlass.range_constexpr(4):
                            a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                            gm = g0 + Int32(m)
                            addr_base = inter_address + (gm.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                            if row0[m] >= Int32(0):
                                addr0 = addr_base + (row0[m].to(Int64) * Int64(128) + k.to(Int64)) * Int64(2)
                                a0 = ld_global_nc_u32(addr0)
                                a2 = ld_global_nc_u32(addr0 + Int64(16))
                            if row1[m] >= Int32(0):
                                addr1 = addr_base + (row1[m].to(Int64) * Int64(128) + k.to(Int64)) * Int64(2)
                                a1 = ld_global_nc_u32(addr1)
                                a3 = ld_global_nc_u32(addr1 + Int64(16))
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[m, 0, 0], acc[m, 0, 1], acc[m, 0, 2], acc[m, 0, 3],
                                a0, a1, a2, a3, b0, b1)
                            acc[m, 0, 0], acc[m, 0, 1] = d0, d1
                            acc[m, 0, 2], acc[m, 0, 3] = d2, d3
                            d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                                acc[m, 1, 0], acc[m, 1, 1], acc[m, 1, 2], acc[m, 1, 3],
                                a0, a1, a2, a3, b2, b3)
                            acc[m, 1, 0], acc[m, 1, 1] = d0, d1
                            acc[m, 1, 2], acc[m, 1, 3] = d2, d3
                    for m in cutlass.range_constexpr(4):
                        for n in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                total[m, n, v] = total[m, n, v] + acc[m, n, v] * scale
            for m in cutlass.range_constexpr(4):
                if Int32(m) < gl:
                    gm = g0 + Int32(m)
                    for n in cutlass.range_constexpr(2):
                        col = block * Int32(128) + warp * Int32(16) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                        base = (gm.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col.to(Int64)
                        if row0[m] >= Int32(0):
                            fc2[base], fc2[base + Int64(1)] = total[m, n, 0], total[m, n, 1]
                        if row1[m] >= Int32(0):
                            fc2[base + Int64(8 * self.hidden_size)], fc2[base + Int64(8 * self.hidden_size + 1)] = total[m, n, 2], total[m, n, 3]
            task += step

    @cute.kernel
    def reduce_kernel(self, route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                      fc2: cute.Tensor, signs_out: cute.Tensor, output: cute.Tensor,
                      rows: Int32, groups):
        """Local expert sum and egress: one warp per (row, 128-column block).

        Each lane owns four adjacent columns. The F32 partials are summed in
        fixed slot order (route order within the row), then the inverse H128
        runs once on the summed partial, signs.out applies and the row rounds
        once to BF16. Every element depends only on its own row's routes, so
        the output is bitwise independent of row count, packing and CTA limit.
        """
        if cutlass.const_expr(self.device_count):
            groups = groups[0]
            # Invalid counts never index descriptor or scratch storage.
            if groups < Int32(0) or groups > Int32(self.max_groups):
                groups = Int32(0)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        task = Int32(cta) * Int32(4) + warp
        step = Int32(grid) * Int32(4)
        fc2_address = fc2.iterator.toint()
        while task < rows * Int32(self.blocks):
            row, block = task // Int32(self.blocks), task % Int32(self.blocks)
            col = block * Int32(128) + lane * Int32(4)
            values = cute.make_rmem_tensor(4, Float32)
            values.fill(Float32(0.0))
            for slot in cutlass.range_constexpr(self.top_k):
                route = row.to(Int64) * Int64(self.top_k) + Int64(slot)
                packed = route_group_lane[route]
                if packed >= Int32(0) and packed < groups * Int32(16):
                    weight = route_weights[route]
                    w0, w1, w2, w3 = ld_global_nc_v4_u32(
                        fc2_address + (packed.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(4))
                    values[0] = values[0] + weight * u32_as_f32(w0)
                    values[1] = values[1] + weight * u32_as_f32(w1)
                    values[2] = values[2] + weight * u32_as_f32(w2)
                    values[3] = values[3] + weight * u32_as_f32(w3)
            h0, h1, h2, h3 = _w4a8_had128_quad(values[0], values[1], values[2], values[3], lane)
            signs = ld_global_nc_u32(signs_out.iterator.toint() + col.to(Int64))
            store_bf16x4(output.iterator.toint() + (row.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(2),
                         h0 * signed_byte(signs, 0), h1 * signed_byte(signs, 1),
                         h2 * signed_byte(signs, 2), h3 * signed_byte(signs, 3))
            task += step
