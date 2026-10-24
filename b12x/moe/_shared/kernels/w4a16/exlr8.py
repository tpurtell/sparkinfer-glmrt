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
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    create_l2_evict_first_policy,
    f16_mma_m16n8k16_f32,
    half2_to_float2_scaled,
    ld_global_b16,
    ld_global_nc_u32,
    ld_global_nc_u32_l2hint,
    ld_global_nc_v4_u32,
    packed_dequant_trellis_to_half2x4,
    st_global_v2_f32,
    u32_as_f32,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


@dsl_user_op
def _ld_global_nc_u32_pred_offset(
    base_ptr: Int64, pred: Int32, offset: int, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
            f"{{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; @p ld.global.nc.b32 $0, [$1 + {int(offset)}]; }}",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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
                      bits: cutlass.Constexpr, policy: Uint64):
    """Return native t256 B fragments, with mandatory fp16 -> E4M3 -> fp16.

    Decode reads each trellis word exactly once, so the words carry an
    `evict_first` L2 policy: the weight stream then leaves activations, fc1
    partials and the pointer tables resident instead of displacing them with
    bytes no CTA re-reads.
    """
    base = payload_address + tile.to(Int64) * Int64(32 * bits)
    if cutlass.const_expr(bits == 4):
        b = ld_global_nc_u32_l2hint(base + lane.to(Int64) * Int64(4), policy)
        a = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
        wa = b
        wb = (a << Uint32(16)) | (b >> Uint32(16))
    else:
        ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)
        a = ld_global_nc_u32_l2hint(base + ia.to(Int64) * Int64(4), policy)
        b = ld_global_nc_u32_l2hint(base + ib.to(Int64) * Int64(4), policy)
        window = (a.to(Int64) << Int64(32)) | b.to(Int64)
        wa = (window >> shift.to(Int64)).to(Uint32)
        wb = (window >> (shift + Int32(4 * bits)).to(Int64)).to(Uint32)
    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, bits)
    return (_e4m3_round_half2(b0), _e4m3_round_half2(b1),
            _e4m3_round_half2(b2), _e4m3_round_half2(b3))


@cute.jit
def _side_value(address: Int64, index: Int32):
    packed = ld_global_b16(address + index.to(Int64) * Int64(2))
    lo, _ = half2_to_float2_scaled(packed, Float32(1.0))
    return lo


class Exlr8A16Kernel:
    ABI_VERSION = 3
    CODE_VERSION = 9
    # Resident CTAs per SM the stream kernels are built for (<=128 registers at
    # 128 threads). GB10 decode is latency-bound on dependent weight loads, so
    # the caller's cta_limit should be sm_count * CTAS_PER_SM (cudaDevAttr at
    # runtime); it never changes results, only how many tasks run at once.
    CTAS_PER_SM = 4

    def __init__(self, hidden_size: int, units_per_expert: int, top_k: int,
                 max_groups: int, fast_math: bool = True, device_count: bool = False):
        if hidden_size <= 0 or hidden_size % 128:
            raise ValueError("EXLR8 hidden_size must be a positive multiple of 128")
        if min(units_per_expert, top_k, max_groups) < 1:
            raise ValueError("EXLR8 units, top_k and group capacity must be positive")
        self.hidden_size = int(hidden_size)
        self.units = int(units_per_expert)
        self.top_k = int(top_k)
        self.max_groups = int(max_groups)
        self.fast_math = bool(fast_math)
        self.device_count = bool(device_count)
        self.group_rows = 16
        self.blocks = hidden_size // 128
        self.splits = (self.blocks + 7) // 8

    @property
    def __cache_key__(self):
        return ("exlr8_a16", self.ABI_VERSION, self.CODE_VERSION, self.hidden_size, self.units,
                self.top_k, self.max_groups, self.fast_math, self.device_count)

    @cute.jit
    def __call__(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                 group_expert: cute.Tensor, route_slots: cute.Tensor,
                 route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                 fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                 signs_out: cute.Tensor, output: cute.Tensor, rows: Int32, groups: Int32,
                 cta_limit: Int32, stream: cuda.CUstream):
        if rows > Int32(0) and rows <= Int32(64) and groups >= Int32(0) and groups <= Int32(self.max_groups) and cta_limit > Int32(0):
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
    def serving(self, x: cute.Tensor, pointers: cute.Tensor, bits: cute.Tensor,
                group_expert: cute.Tensor, route_slots: cute.Tensor,
                route_group_lane: cute.Tensor, route_weights: cute.Tensor,
                fc1: cute.Tensor, intermediate: cute.Tensor, fc2: cute.Tensor,
                signs_out: cute.Tensor, output: cute.Tensor, rows: Int32,
                group_count: cute.Tensor, cta_limit: Int32, stream: cuda.CUstream):
        # Grids and launch count are independent of route data, including zero
        # groups. Each kernel bounds its tasks with the device-resident count.
        if rows > Int32(0) and rows <= Int32(64) and cta_limit > Int32(0):
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
    def _mma_block(self, payload: Int64, a_address: Int64,
                   row0: Int32, row1: Int32, a_stride: cutlass.Constexpr,
                   input_block: Int32, output_block: Int32, lane: Int32,
                   warp: Int32, width: cutlass.Constexpr, down: cutlass.Constexpr,
                   policy: Uint64):
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
                b0, b1, b2, b3 = exlr8_decode_tile(payload, tile, lane, width, policy)
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
                          down: cutlass.Constexpr, scale: Float32, policy: Uint64):
        acc = self._mma_block(payload, a_address, row0, row1, a_stride,
                              input_block, output_block, lane, warp, width, down,
                              policy)
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
        policy = create_l2_evict_first_policy()
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
                                self.hidden_size, kb, Int32(0), lane, warp, 2, False, scale, policy)
                        elif width == Int32(3):
                            self._accumulate_block(total, payload, x.iterator.toint(), row0, row1,
                                self.hidden_size, kb, Int32(0), lane, warp, 3, False, scale, policy)
                        elif width == Int32(4):
                            self._accumulate_block(total, payload, x.iterator.toint(), row0, row1,
                                self.hidden_size, kb, Int32(0), lane, warp, 4, False, scale, policy)
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
            for iteration in cutlass.range_constexpr(self.group_rows // 4):
                row = warp + Int32(iteration * 4)
                slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
                g = cute.make_rmem_tensor(4, Float32)
                u = cute.make_rmem_tensor(4, Float32)
                g.fill(Float32(0.0))
                u.fill(Float32(0.0))
                if bits[desc] != Int32(0) and slot >= Int32(0) and slot < rows * Int32(self.top_k):
                    for split in cutlass.range_constexpr(self.splits):
                        base = (task.to(Int64) * Int64(2 * self.splits) + Int64(split)) * Int64(self.group_rows * 128) + row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
                        for v in cutlass.range_constexpr(4):
                            g[v] = g[v] + fc1[base + Int64(v)]
                            u[v] = u[v] + fc1[base + Int64(self.splits * self.group_rows * 128 + v)]
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
                    base = (task.to(Int64) * Int64(self.group_rows) + row.to(Int64)) * Int64(128) + lane.to(Int64) * Int64(4)
                    for v in cutlass.range_constexpr(4):
                        intermediate[base + Int64(v)] = g[v].to(cutlass.Float16)
            task += step

    @cute.jit
    def _fc2_accumulate_block(
        self,
        total: cute.Tensor,
        payload: Int64,
        a_base: Int64,
        valid0: Int32,
        valid1: Int32,
        output_block: Int32,
        lane: Int32,
        warp: Int32,
        width: cutlass.Constexpr,
        packed_scale: Uint32,
        l2_policy: Uint64,
    ):
        acc = cute.make_rmem_tensor((4, 4), Float32)
        acc.fill(Float32(0.0))
        base_tile = (output_block * Int32(64) + warp * Int32(2)).to(Int64) * Int64(32 * width)
        a0_buf = cute.make_rmem_tensor(8, Uint32)
        a1_buf = cute.make_rmem_tensor(8, Uint32)
        a2_buf = cute.make_rmem_tensor(8, Uint32)
        a3_buf = cute.make_rmem_tensor(8, Uint32)
        for k16 in cutlass.range_constexpr(8):
            a0_buf[k16] = _ld_global_nc_u32_pred_offset(a_base, valid0, k16 * 32)
            a2_buf[k16] = _ld_global_nc_u32_pred_offset(a_base, valid0, k16 * 32 + 16)
            a1_buf[k16] = _ld_global_nc_u32_pred_offset(a_base, valid1, 2048 + k16 * 32)
            a3_buf[k16] = _ld_global_nc_u32_pred_offset(a_base, valid1, 2048 + k16 * 32 + 16)
        if cutlass.const_expr(width == 4):
            b_ptr = payload + base_tile + lane.to(Int64) * Int64(4)
            prev_lane = (lane + Int32(31)) & Int32(31)
            wb_buf = cute.make_rmem_tensor((8, 2), Uint32)
            for k16 in cutlass.range_constexpr(8):
                for n16 in cutlass.range_constexpr(2):
                    wb_buf[k16, n16] = ld_global_nc_u32_l2hint(
                        b_ptr, l2_policy, (k16 * 8 + n16) * 128
                    )
            for k16 in cutlass.range_constexpr(8):
                a0, a1, a2, a3 = a0_buf[k16], a1_buf[k16], a2_buf[k16], a3_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    b = wb_buf[k16, n16]
                    a = cute.arch.shuffle_sync(b, prev_lane)
                    wa = b
                    wb = (a << Uint32(16)) | (b >> Uint32(16))
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 4)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16, 0], acc[2 * n16, 1], acc[2 * n16, 2], acc[2 * n16, 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[2 * n16, 0], acc[2 * n16, 1] = d0, d1
                    acc[2 * n16, 2], acc[2 * n16, 3] = d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1], acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3],
                        a0, a1, a2, a3, b2, b3,
                    )
                    acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1] = d0, d1
                    acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3] = d2, d3
        else:
            ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, width)
            b_ptr_a = payload + base_tile + ia.to(Int64) * Int64(4)
            b_ptr_b = payload + base_tile + ib.to(Int64) * Int64(4)
            shift_lo = shift.to(Int64)
            shift_hi = (shift + Int32(4 * width)).to(Int64)
            wa_buf = cute.make_rmem_tensor((8, 2), Uint32)
            wb_buf = cute.make_rmem_tensor((8, 2), Uint32)
            for k16 in cutlass.range_constexpr(8):
                for n16 in cutlass.range_constexpr(2):
                    off = (k16 * 8 + n16) * 32 * width
                    wa_buf[k16, n16] = ld_global_nc_u32_l2hint(b_ptr_a, l2_policy, off)
                    wb_buf[k16, n16] = ld_global_nc_u32_l2hint(b_ptr_b, l2_policy, off)
            for k16 in cutlass.range_constexpr(8):
                a0, a1, a2, a3 = a0_buf[k16], a1_buf[k16], a2_buf[k16], a3_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    a = wa_buf[k16, n16]
                    b = wb_buf[k16, n16]
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift_lo).to(Uint32)
                    wb = (window >> shift_hi).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16, 0], acc[2 * n16, 1], acc[2 * n16, 2], acc[2 * n16, 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[2 * n16, 0], acc[2 * n16, 1] = d0, d1
                    acc[2 * n16, 2], acc[2 * n16, 3] = d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1], acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3],
                        a0, a1, a2, a3, b2, b3,
                    )
                    acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1] = d0, d1
                    acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3] = d2, d3
        scale, _ = half2_to_float2_scaled(packed_scale, Float32(1.0))
        for n in cutlass.range_constexpr(4):
            for v in cutlass.range_constexpr(4):
                total[n, v] = total[n, v] + acc[n, v] * scale

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
        r = lane >> Int32(2)
        a_lane_off = (r * Int32(128) + (lane & Int32(3)) * Int32(2)).to(Int64) * Int64(2)
        inter_base = intermediate.iterator.toint() + a_lane_off
        l2_policy = create_l2_evict_first_policy()
        max_slot = rows * Int32(self.top_k)
        while task < groups * Int32(self.blocks):
            group, block = task // Int32(self.blocks), task % Int32(self.blocks)
            expert = group_expert[group]
            total = cute.make_rmem_tensor((4, 4), Float32)
            total.fill(Float32(0.0))
            slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
            slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
            valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
            valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
            for unit in cutlass.range(self.units):
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                width = bits[desc]
                if width != Int32(0):
                    payload = pointers[desc * Int64(3)]
                    packed_scale = ld_global_b16(pointers[desc * Int64(3) + Int64(2)] + block.to(Int64) * Int64(2))
                    a_base = inter_base + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                    if width == Int32(2):
                        self._fc2_accumulate_block(
                            total, payload, a_base, valid0, valid1, block, lane, warp, 2, packed_scale, l2_policy
                        )
                    elif width == Int32(3):
                        self._fc2_accumulate_block(
                            total, payload, a_base, valid0, valid1, block, lane, warp, 3, packed_scale, l2_policy
                        )
                    elif width == Int32(4):
                        self._fc2_accumulate_block(
                            total, payload, a_base, valid0, valid1, block, lane, warp, 4, packed_scale, l2_policy
                        )
            out_group, out_block = task // Int32(self.blocks), task % Int32(self.blocks)
            for n in cutlass.range_constexpr(4):
                col = out_block * Int32(128) + warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                base = (out_group.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col.to(Int64)
                if valid0 != Int32(0):
                    fc2[base], fc2[base + Int64(1)] = total[n, 0], total[n, 1]
                if valid1 != Int32(0):
                    fc2[base + Int64(8 * self.hidden_size)], fc2[base + Int64(8 * self.hidden_size + 1)] = total[n, 2], total[n, 3]
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
                if packed >= Int32(0) and packed < groups * Int32(self.group_rows):
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


@dsl_user_op
def store_f16x4(address: Int64, v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                *, loc=None, ip=None):
    """Round four F32 values to FP16 (nearest even) and store them as one 8-byte vector."""
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip), Float32(v2).ir_value(loc=loc, ip=ip),
         Float32(v3).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 lo, hi; cvt.rn.f16x2.f32 lo, $2, $1; cvt.rn.f16x2.f32 hi, $4, $3; "
        "st.global.v2.b32 [$0], {lo, hi}; }",
        "l,f,f,f,f", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _vote_ballot_sync(pred: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(pred).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc_u32(val: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(val).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ctz_u32(val: Uint32, *, loc=None, ip=None) -> Int32:
    """Index of lowest set bit in a non-zero u32 (0..31)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(val).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 r; brev.b32 r, $1; bfind.shiftamt.u32 $0, r; }",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_u32_pred_l2hint(
    base_ptr: Int64, pred: Int32, cache_policy: Uint64, offset: int = 0, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(base_ptr).ir_value(loc=loc, ip=ip),
                Int32(pred).ir_value(loc=loc, ip=ip),
                Uint64(cache_policy).ir_value(loc=loc, ip=ip),
            ],
            f"{{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0; @p ld.global.nc.L2::cache_hint.u32 $0, [$1 + {int(offset)}], $3; }}",
            "=r,l,r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_v2_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


SERVING_VARIANTS = (
    "k1_compact_prefetch",
    "k1_paired_fc1",
    "k1_reg_pair_b2",
    "k1_k23_coalesced",
    "k1_in_cta_fold",
    "k1_supergroup_fold",
)


class Exlr8RoutedServingKernel:
    """Routed <=64-row EXLR8 A16 serving kernel with compact owned-unit tasking
    and constant-offset B-first register prefetch in both FC1 and FC2.

    Variants:
    - ``k1_compact_prefetch``: shared-memory live-unit compaction in FC1/activation
      (eliminating the ``units * 2 * splits == 192`` CTA-pinning bug where
      12/16 CTAs on TP4 and >=8/16 CTAs on TP6 did zero work), bitmask owned-unit
      iteration in FC2, constant-offset B-first register prefetch in FC1/FC2,
      vectorized activation/reduce loads and stores, and row-bounded grid clamps.
    - ``k1_paired_fc1``: ``k1_compact_prefetch`` plus adaptive paired ``gate`` +
      ``up`` per ``(u_idx, split)`` task in FC1 so ``up`` reuses warm L1 ``x``
      activations from ``gate`` on the same SM.
    - ``k1_reg_pair_b2``: ``k1_paired_fc1`` plus register-shared ``a0..a3_buf[8]``
      across ``gate`` and ``up`` per K128 block in FC1 and paired 2-block
      ``(b0, b1)`` tasks with register-shared ``a0..a3_buf[8]`` in FC2.
    - ``k1_k23_coalesced``: coalesced 128B warp weight loads + register shuffle
      on K2 (8 loads/block vs 32) and K3 (16 loads/block vs 32) in FC1/FC2,
      plus compact ``slot_out``-indexed FC2 partials and constant-offset reduce.
    - ``k1_in_cta_fold``: ``k1_k23_coalesced`` plus adaptive in-CTA register
      split folding in FC1/activation when ``total_units >= 96`` (cutting
      FC1 -> activation global partial traffic 6x at M8..M64 while keeping
      6-split parallelism at M1..M4).
    - ``k1_supergroup_fold``: ``k1_in_cta_fold`` plus consecutive-group
      hot-expert coalescing (pairing ``(g, g+1)`` when ``group_expert[g] ==
      group_expert[g+1]`` so weights are streamed and dequantized once across
      both 16-row groups in FC1 and FC2).
    """

    ABI_VERSION = 3
    CODE_VERSION = 11
    CTAS_PER_SM = 4
    SMEM_GU_CAP = 2048
    IN_CTA_FOLD_MIN_UNITS = 96

    def __init__(
        self,
        hidden_size: int,
        units_per_expert: int,
        top_k: int,
        max_groups: int,
        fast_math: bool = True,
        variant: str = "k1_reg_pair_b2",
    ):
        if hidden_size <= 0 or hidden_size % 128:
            raise ValueError("EXLR8 hidden_size must be a positive multiple of 128")
        if min(units_per_expert, top_k, max_groups) < 1 or units_per_expert > 16:
            raise ValueError("EXLR8 units (1..16), top_k and group capacity must be positive")
        if variant not in SERVING_VARIANTS:
            raise ValueError(f"Unknown EXLR8 serving variant: {variant!r}")
        self.hidden_size = int(hidden_size)
        self.units = int(units_per_expert)
        self.top_k = int(top_k)
        self.max_groups = int(max_groups)
        self.fast_math = bool(fast_math)
        self.variant = str(variant)
        self.paired_fc1 = variant in (
            "k1_paired_fc1",
            "k1_reg_pair_b2",
            "k1_k23_coalesced",
            "k1_in_cta_fold",
            "k1_supergroup_fold",
        )
        self.reg_pair_fc1 = variant in (
            "k1_reg_pair_b2",
            "k1_k23_coalesced",
            "k1_in_cta_fold",
            "k1_supergroup_fold",
        )
        self.pair_fc2 = variant == "k1_reg_pair_b2" and (hidden_size // 128) % 2 == 0
        self.k23_coalesced = variant in (
            "k1_k23_coalesced",
            "k1_in_cta_fold",
            "k1_supergroup_fold",
        )
        self.compact_fc2 = variant in (
            "k1_k23_coalesced",
            "k1_in_cta_fold",
            "k1_supergroup_fold",
        )
        self.in_cta_fold = variant in ("k1_in_cta_fold", "k1_supergroup_fold")
        self.supergroup = variant == "k1_supergroup_fold"
        self.group_rows = 16
        self.blocks = hidden_size // 128
        self.splits = (self.blocks + 7) // 8

    @property
    def __cache_key__(self):
        return (
            "exlr8_routed_serving",
            self.ABI_VERSION,
            self.CODE_VERSION,
            self.hidden_size,
            self.units,
            self.top_k,
            self.max_groups,
            self.fast_math,
            self.variant,
        )

    @cute.jit
    def serving(
        self,
        x: cute.Tensor,
        pointers: cute.Tensor,
        bits: cute.Tensor,
        group_expert: cute.Tensor,
        route_slots: cute.Tensor,
        route_group_lane: cute.Tensor,
        route_weights: cute.Tensor,
        fc1: cute.Tensor,
        intermediate: cute.Tensor,
        fc2: cute.Tensor,
        signs_out: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        group_count: cute.Tensor,
        cta_limit: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0) and rows <= Int32(64) and cta_limit > Int32(0):
            max_live_groups = min(rows * Int32(self.top_k), Int32(self.max_groups))
            self.fc1_kernel(
                x, pointers, bits, group_expert, route_slots, fc1, rows, group_count
            ).launch(
                grid=(min(max_live_groups * Int32(self.units * 2 * self.splits), cta_limit), 1, 1),
                block=(128, 1, 1),
                min_blocks_per_mp=self.CTAS_PER_SM,
                stream=stream,
            )
            self.activation_kernel(
                pointers, bits, group_expert, route_slots, fc1, intermediate, rows, group_count
            ).launch(
                grid=(min(max_live_groups * Int32(self.units), cta_limit), 1, 1),
                block=(128, 1, 1),
                stream=stream,
            )
            self.fc2_kernel(
                pointers, bits, group_expert, route_slots, intermediate, fc2, rows, group_count
            ).launch(
                grid=(min(max_live_groups * Int32(self.blocks), cta_limit), 1, 1),
                block=(128, 1, 1),
                min_blocks_per_mp=self.CTAS_PER_SM,
                stream=stream,
            )
            self.reduce_kernel(
                route_group_lane, route_weights, fc2, signs_out, output, rows, group_count
            ).launch(
                grid=(min((rows * Int32(self.blocks) + Int32(3)) // Int32(4), cta_limit), 1, 1),
                block=(128, 1, 1),
                stream=stream,
            )

    @cute.jit
    def _build_live_units(
        self,
        bits: cute.Tensor,
        group_expert: cute.Tensor,
        groups: Int32,
        tid: Int32,
        lane: Int32,
        warp: Int32,
        s_meta: cute.Tensor,
        s_offset: cute.Tensor,
        s_gu: cute.Tensor,
        s_total: cute.Tensor,
    ) -> Int32:
        half_warp = tid >> Int32(4)
        u = tid & Int32(15)
        half_shift = Uint32(tid & Int32(16))
        g_base = Int32(0)
        while g_base < groups:
            g = g_base + half_warp
            owned = Int32(0)
            expert = Int32(0)
            if g < groups:
                expert = group_expert[g]
                if u < Int32(self.units) and expert >= Int32(0):
                    desc = (expert.to(Int64) * Int64(self.units) + u.to(Int64)) * Int64(3)
                    if bits[desc] != Int32(0):
                        owned = Int32(1)
            ballot = _vote_ballot_sync(owned)
            if g < groups and u == Int32(0):
                mask16 = (ballot >> half_shift) & Uint32(0xFFFF)
                s_meta[g] = (expert.to(Uint32) << Uint32(16)) | mask16
            g_base += Int32(8)
        cute.arch.sync_threads()
        if warp == Int32(0):
            running = Int32(0)
            g_scan = Int32(0)
            while g_scan < groups:
                g = g_scan + lane
                cnt = Int32(0)
                if g < groups:
                    cnt = _popc_u32(s_meta[g] & Uint32(0xFFFF))
                scan = cnt
                for d in cutlass.range_constexpr(5):
                    step_d = Int32(1 << d)
                    prev = cute.arch.shuffle_sync(scan, (lane - step_d) & Int32(31))
                    if lane >= step_d:
                        scan = scan + prev
                if g < groups:
                    s_offset[g] = running + scan - cnt
                running = running + cute.arch.shuffle_sync(scan, Int32(31))
                g_scan += Int32(32)
            if lane == Int32(0):
                s_total[0] = running
        cute.arch.sync_threads()
        total_units = s_total[0]
        if total_units <= Int32(self.SMEM_GU_CAP):
            g_scatter = Int32(0)
            while g_scatter < groups:
                g = g_scatter + half_warp
                if g < groups and u < Int32(self.units):
                    meta = s_meta[g]
                    mask16 = meta & Uint32(0xFFFF)
                    bit = Uint32(1) << u.to(Uint32)
                    if (mask16 & bit) != Uint32(0):
                        idx = s_offset[g] + _popc_u32(mask16 & (bit - Uint32(1)))
                        expert = ((meta >> Uint32(16)) & Uint32(0xFF)).to(Int32)
                        packed_gu = (expert << Int32(13)) | (g << Int32(4)) | u
                        if cutlass.const_expr(self.supergroup):
                            run_idx = Int32(0)
                            g_prev = g - Int32(1)
                            keep_scanning = (g_prev >= Int32(0)).to(Int32)
                            while keep_scanning != Int32(0):
                                prev_exp = ((s_meta[max(g_prev, Int32(0))] >> Uint32(16)) & Uint32(0xFF)).to(Int32)
                                if prev_exp == expert:
                                    run_idx += Int32(1)
                                    g_prev -= Int32(1)
                                    keep_scanning = (g_prev >= Int32(0)).to(Int32)
                                else:
                                    keep_scanning = Int32(0)
                            skip_g = run_idx & Int32(1)
                            has_g1 = Int32(0)
                            if skip_g == Int32(0) and (g + Int32(1)) < groups:
                                next_exp = ((s_meta[min(g + Int32(1), groups - Int32(1))] >> Uint32(16)) & Uint32(0xFF)).to(Int32)
                                if next_exp == expert:
                                    has_g1 = Int32(1)
                            owned_cnt = _popc_u32(mask16)
                            packed_gu = (
                                packed_gu
                                | (has_g1 << Int32(21))
                                | (skip_g << Int32(22))
                                | (owned_cnt << Int32(23))
                            )
                        s_gu[idx] = packed_gu
                g_scatter += Int32(8)
            cute.arch.sync_threads()
        return total_units

    @cute.jit
    def _lookup_live_unit(
        self,
        u_idx: Int32,
        groups: Int32,
        total_units: Int32,
        s_gu: cute.Tensor,
        s_offset: cute.Tensor,
        s_meta: cute.Tensor,
    ):
        u = Int32(0)
        g = Int32(0)
        expert = Int32(0)
        if total_units <= Int32(self.SMEM_GU_CAP):
            packed = s_gu[u_idx]
            u = packed & Int32(15)
            g = (packed >> Int32(4)) & Int32(511)
            expert = (packed >> Int32(13)) & Int32(255)
        else:
            lo = Int32(0)
            hi = groups - Int32(1)
            for _ in cutlass.range_constexpr(10):
                mid = (lo + hi + Int32(1)) >> Int32(1)
                if s_offset[mid] <= u_idx:
                    lo = mid
                else:
                    hi = mid - Int32(1)
            g = lo
            rem = u_idx - s_offset[g]
            meta = s_meta[g]
            mask16 = meta & Uint32(0xFFFF)
            expert = ((meta >> Uint32(16)) & Uint32(0xFF)).to(Int32)
            while rem > Int32(0):
                mask16 = mask16 & (mask16 - Uint32(1))
                rem -= Int32(1)
            u = _ctz_u32(mask16)
        return u, g, expert

    @cute.jit
    def _lookup_live_unit_ext(
        self,
        u_idx: Int32,
        groups: Int32,
        total_units: Int32,
        s_gu: cute.Tensor,
        s_offset: cute.Tensor,
        s_meta: cute.Tensor,
    ):
        u, g, expert = self._lookup_live_unit(u_idx, groups, total_units, s_gu, s_offset, s_meta)
        has_g1 = Int32(0)
        skip_g = Int32(0)
        owned_cnt = Int32(0)
        if total_units <= Int32(self.SMEM_GU_CAP):
            packed = s_gu[u_idx]
            has_g1 = (packed >> Int32(21)) & Int32(1)
            skip_g = (packed >> Int32(22)) & Int32(1)
            owned_cnt = (packed >> Int32(23)) & Int32(31)
        return u, g, expert, has_g1, skip_g, owned_cnt

    @cute.jit
    def _load_a_block(
        self,
        a_base0: Int64,
        a_base1: Int64,
        valid0: Int32,
        valid1: Int32,
    ):
        a0_buf = cute.make_rmem_tensor(8, Uint32)
        a1_buf = cute.make_rmem_tensor(8, Uint32)
        a2_buf = cute.make_rmem_tensor(8, Uint32)
        a3_buf = cute.make_rmem_tensor(8, Uint32)
        for k16 in cutlass.range_constexpr(8):
            a0_buf[k16] = _ld_global_nc_u32_pred_offset(a_base0, valid0, k16 * 32)
            a2_buf[k16] = _ld_global_nc_u32_pred_offset(a_base0, valid0, k16 * 32 + 16)
            a1_buf[k16] = _ld_global_nc_u32_pred_offset(a_base1, valid1, k16 * 32)
            a3_buf[k16] = _ld_global_nc_u32_pred_offset(a_base1, valid1, k16 * 32 + 16)
        return a0_buf, a1_buf, a2_buf, a3_buf

    @cute.jit
    def _mma_with_loaded_a(
        self,
        total: cute.Tensor,
        payload: Int64,
        a0_buf: cute.Tensor,
        a1_buf: cute.Tensor,
        a2_buf: cute.Tensor,
        a3_buf: cute.Tensor,
        tile_block: Int32,
        lane: Int32,
        warp: Int32,
        prev_lane: Int32,
        ia: Int32,
        ib: Int32,
        shift_lo: Int64,
        shift_hi: Int64,
        width: cutlass.Constexpr,
        scale: Float32,
        l2_policy: Uint64,
    ):
        acc = cute.make_rmem_tensor((4, 4), Float32)
        acc.fill(Float32(0.0))
        base_tile = (tile_block * Int32(64) + warp * Int32(2)).to(Int64) * Int64(32 * width)
        if cutlass.const_expr(width == 4):
            b_ptr = payload + base_tile + lane.to(Int64) * Int64(4)
            wb_buf = cute.make_rmem_tensor((8, 2), Uint32)
            for k16 in cutlass.range_constexpr(8):
                for n16 in cutlass.range_constexpr(2):
                    wb_buf[k16, n16] = ld_global_nc_u32_l2hint(
                        b_ptr, l2_policy, (k16 * 8 + n16) * 128
                    )
            for k16 in cutlass.range_constexpr(8):
                a0, a1, a2, a3 = a0_buf[k16], a1_buf[k16], a2_buf[k16], a3_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    b = wb_buf[k16, n16]
                    a = cute.arch.shuffle_sync(b, prev_lane)
                    wa = b
                    wb = (a << Uint32(16)) | (b >> Uint32(16))
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 4)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16, 0], acc[2 * n16, 1], acc[2 * n16, 2], acc[2 * n16, 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[2 * n16, 0], acc[2 * n16, 1] = d0, d1
                    acc[2 * n16, 2], acc[2 * n16, 3] = d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1], acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3],
                        a0, a1, a2, a3, b2, b3,
                    )
                    acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1] = d0, d1
                    acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3] = d2, d3
        elif cutlass.const_expr(self.k23_coalesced and width == 2):
            b_ptr = payload + base_tile + lane.to(Int64) * Int64(4)
            ia_hi = ia + Int32(16)
            ib_hi = ib + Int32(16)
            w_buf = cute.make_rmem_tensor(8, Uint32)
            for k16 in cutlass.range_constexpr(8):
                w_buf[k16] = ld_global_nc_u32_l2hint(b_ptr, l2_policy, k16 * 512)
            for k16 in cutlass.range_constexpr(8):
                a0, a1, a2, a3 = a0_buf[k16], a1_buf[k16], a2_buf[k16], a3_buf[k16]
                w = w_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    src_a = ia if cutlass.const_expr(n16 == 0) else ia_hi
                    src_b = ib if cutlass.const_expr(n16 == 0) else ib_hi
                    a = cute.arch.shuffle_sync(w, src_a)
                    b = cute.arch.shuffle_sync(w, src_b)
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift_lo).to(Uint32)
                    wb = (window >> shift_hi).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 2)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16, 0], acc[2 * n16, 1], acc[2 * n16, 2], acc[2 * n16, 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[2 * n16, 0], acc[2 * n16, 1] = d0, d1
                    acc[2 * n16, 2], acc[2 * n16, 3] = d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1], acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3],
                        a0, a1, a2, a3, b2, b3,
                    )
                    acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1] = d0, d1
                    acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3] = d2, d3
        elif cutlass.const_expr(self.k23_coalesced and width == 3):
            b_ptr = payload + base_tile + lane.to(Int64) * Int64(4)
            ia_hi = ia + Int32(8)
            ib_hi = ib + Int32(8)
            pred_lo = (lane < Int32(24)).to(Int32)
            pred_hi = (lane >= Int32(8)).to(Int32)
            w0_buf = cute.make_rmem_tensor(8, Uint32)
            w1_buf = cute.make_rmem_tensor(8, Uint32)
            for k16 in cutlass.range_constexpr(8):
                w0_buf[k16] = _ld_global_nc_u32_pred_l2hint(b_ptr, pred_lo, l2_policy, k16 * 768)
                w1_buf[k16] = _ld_global_nc_u32_pred_l2hint(b_ptr, pred_hi, l2_policy, k16 * 768 + 64)
            for k16 in cutlass.range_constexpr(8):
                a0, a1, a2, a3 = a0_buf[k16], a1_buf[k16], a2_buf[k16], a3_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    w = w0_buf[k16] if cutlass.const_expr(n16 == 0) else w1_buf[k16]
                    src_a = ia if cutlass.const_expr(n16 == 0) else ia_hi
                    src_b = ib if cutlass.const_expr(n16 == 0) else ib_hi
                    a = cute.arch.shuffle_sync(w, src_a)
                    b = cute.arch.shuffle_sync(w, src_b)
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift_lo).to(Uint32)
                    wb = (window >> shift_hi).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 3)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16, 0], acc[2 * n16, 1], acc[2 * n16, 2], acc[2 * n16, 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[2 * n16, 0], acc[2 * n16, 1] = d0, d1
                    acc[2 * n16, 2], acc[2 * n16, 3] = d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1], acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3],
                        a0, a1, a2, a3, b2, b3,
                    )
                    acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1] = d0, d1
                    acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3] = d2, d3
        else:
            b_ptr_a = payload + base_tile + ia.to(Int64) * Int64(4)
            b_ptr_b = payload + base_tile + ib.to(Int64) * Int64(4)
            wa_buf = cute.make_rmem_tensor((8, 2), Uint32)
            wb_buf = cute.make_rmem_tensor((8, 2), Uint32)
            for k16 in cutlass.range_constexpr(8):
                for n16 in cutlass.range_constexpr(2):
                    off = (k16 * 8 + n16) * 32 * width
                    wa_buf[k16, n16] = ld_global_nc_u32_l2hint(b_ptr_a, l2_policy, off)
                    wb_buf[k16, n16] = ld_global_nc_u32_l2hint(b_ptr_b, l2_policy, off)
            for k16 in cutlass.range_constexpr(8):
                a0, a1, a2, a3 = a0_buf[k16], a1_buf[k16], a2_buf[k16], a3_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    a = wa_buf[k16, n16]
                    b = wb_buf[k16, n16]
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift_lo).to(Uint32)
                    wb = (window >> shift_hi).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, width)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16, 0], acc[2 * n16, 1], acc[2 * n16, 2], acc[2 * n16, 3],
                        a0, a1, a2, a3, b0, b1,
                    )
                    acc[2 * n16, 0], acc[2 * n16, 1] = d0, d1
                    acc[2 * n16, 2], acc[2 * n16, 3] = d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1], acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3],
                        a0, a1, a2, a3, b2, b3,
                    )
                    acc[2 * n16 + 1, 0], acc[2 * n16 + 1, 1] = d0, d1
                    acc[2 * n16 + 1, 2], acc[2 * n16 + 1, 3] = d2, d3
        for n in cutlass.range_constexpr(4):
            for v in cutlass.range_constexpr(4):
                total[n, v] = total[n, v] + acc[n, v] * scale

    @cute.jit
    def _fc1_accumulate_block(
        self,
        total: cute.Tensor,
        payload: Int64,
        a_base0: Int64,
        a_base1: Int64,
        valid0: Int32,
        valid1: Int32,
        input_block: Int32,
        lane: Int32,
        warp: Int32,
        prev_lane: Int32,
        ia: Int32,
        ib: Int32,
        shift_lo: Int64,
        shift_hi: Int64,
        width: cutlass.Constexpr,
        scale: Float32,
        l2_policy: Uint64,
    ):
        a0_b, a1_b, a2_b, a3_b = self._load_a_block(a_base0, a_base1, valid0, valid1)
        self._mma_with_loaded_a(
            total, payload, a0_b, a1_b, a2_b, a3_b, input_block,
            lane, warp, prev_lane, ia, ib, shift_lo, shift_hi,
            width, scale, l2_policy,
        )

    @cute.jit
    def _accumulate_pair_groups_block(
        self,
        total0: cute.Tensor,
        total1: cute.Tensor,
        payload: Int64,
        a_base0_g0: Int64,
        a_base1_g0: Int64,
        valid0_g0: Int32,
        valid1_g0: Int32,
        a_base0_g1: Int64,
        a_base1_g1: Int64,
        valid0_g1: Int32,
        valid1_g1: Int32,
        tile_block: Int32,
        lane: Int32,
        warp: Int32,
        prev_lane: Int32,
        ia: Int32,
        ib: Int32,
        shift_lo: Int64,
        shift_hi: Int64,
        width: cutlass.Constexpr,
        scale: Float32,
        l2_policy: Uint64,
    ):
        acc0 = cute.make_rmem_tensor((4, 4), Float32)
        acc1 = cute.make_rmem_tensor((4, 4), Float32)
        acc0.fill(Float32(0.0))
        acc1.fill(Float32(0.0))
        base_tile = (tile_block * Int32(64) + warp * Int32(2)).to(Int64) * Int64(32 * width)
        b_ptr = payload + base_tile + lane.to(Int64) * Int64(4)
        if cutlass.const_expr(width == 4):
            wb_buf = cute.make_rmem_tensor((8, 2), Uint32)
            for k16 in cutlass.range_constexpr(8):
                for n16 in cutlass.range_constexpr(2):
                    wb_buf[k16, n16] = ld_global_nc_u32_l2hint(
                        b_ptr, l2_policy, (k16 * 8 + n16) * 128
                    )
            for k16 in cutlass.range_constexpr(8):
                a0_0 = _ld_global_nc_u32_pred_offset(a_base0_g0, valid0_g0, k16 * 32)
                a2_0 = _ld_global_nc_u32_pred_offset(a_base0_g0, valid0_g0, k16 * 32 + 16)
                a1_0 = _ld_global_nc_u32_pred_offset(a_base1_g0, valid1_g0, k16 * 32)
                a3_0 = _ld_global_nc_u32_pred_offset(a_base1_g0, valid1_g0, k16 * 32 + 16)
                a0_1 = _ld_global_nc_u32_pred_offset(a_base0_g1, valid0_g1, k16 * 32)
                a2_1 = _ld_global_nc_u32_pred_offset(a_base0_g1, valid0_g1, k16 * 32 + 16)
                a1_1 = _ld_global_nc_u32_pred_offset(a_base1_g1, valid1_g1, k16 * 32)
                a3_1 = _ld_global_nc_u32_pred_offset(a_base1_g1, valid1_g1, k16 * 32 + 16)
                for n16 in cutlass.range_constexpr(2):
                    b = wb_buf[k16, n16]
                    a = cute.arch.shuffle_sync(b, prev_lane)
                    wa = b
                    wb = (a << Uint32(16)) | (b >> Uint32(16))
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 4)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc0[2 * n16, 0], acc0[2 * n16, 1], acc0[2 * n16, 2], acc0[2 * n16, 3],
                        a0_0, a1_0, a2_0, a3_0, b0, b1,
                    )
                    acc0[2 * n16, 0], acc0[2 * n16, 1], acc0[2 * n16, 2], acc0[2 * n16, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc0[2 * n16 + 1, 0], acc0[2 * n16 + 1, 1], acc0[2 * n16 + 1, 2], acc0[2 * n16 + 1, 3],
                        a0_0, a1_0, a2_0, a3_0, b2, b3,
                    )
                    acc0[2 * n16 + 1, 0], acc0[2 * n16 + 1, 1], acc0[2 * n16 + 1, 2], acc0[2 * n16 + 1, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc1[2 * n16, 0], acc1[2 * n16, 1], acc1[2 * n16, 2], acc1[2 * n16, 3],
                        a0_1, a1_1, a2_1, a3_1, b0, b1,
                    )
                    acc1[2 * n16, 0], acc1[2 * n16, 1], acc1[2 * n16, 2], acc1[2 * n16, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc1[2 * n16 + 1, 0], acc1[2 * n16 + 1, 1], acc1[2 * n16 + 1, 2], acc1[2 * n16 + 1, 3],
                        a0_1, a1_1, a2_1, a3_1, b2, b3,
                    )
                    acc1[2 * n16 + 1, 0], acc1[2 * n16 + 1, 1], acc1[2 * n16 + 1, 2], acc1[2 * n16 + 1, 3] = d0, d1, d2, d3
        elif cutlass.const_expr(width == 2):
            ia_hi = ia + Int32(16)
            ib_hi = ib + Int32(16)
            w_buf = cute.make_rmem_tensor(8, Uint32)
            for k16 in cutlass.range_constexpr(8):
                w_buf[k16] = ld_global_nc_u32_l2hint(b_ptr, l2_policy, k16 * 512)
            for k16 in cutlass.range_constexpr(8):
                a0_0 = _ld_global_nc_u32_pred_offset(a_base0_g0, valid0_g0, k16 * 32)
                a2_0 = _ld_global_nc_u32_pred_offset(a_base0_g0, valid0_g0, k16 * 32 + 16)
                a1_0 = _ld_global_nc_u32_pred_offset(a_base1_g0, valid1_g0, k16 * 32)
                a3_0 = _ld_global_nc_u32_pred_offset(a_base1_g0, valid1_g0, k16 * 32 + 16)
                a0_1 = _ld_global_nc_u32_pred_offset(a_base0_g1, valid0_g1, k16 * 32)
                a2_1 = _ld_global_nc_u32_pred_offset(a_base0_g1, valid0_g1, k16 * 32 + 16)
                a1_1 = _ld_global_nc_u32_pred_offset(a_base1_g1, valid1_g1, k16 * 32)
                a3_1 = _ld_global_nc_u32_pred_offset(a_base1_g1, valid1_g1, k16 * 32 + 16)
                w = w_buf[k16]
                for n16 in cutlass.range_constexpr(2):
                    src_a = ia if cutlass.const_expr(n16 == 0) else ia_hi
                    src_b = ib if cutlass.const_expr(n16 == 0) else ib_hi
                    a = cute.arch.shuffle_sync(w, src_a)
                    b = cute.arch.shuffle_sync(w, src_b)
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift_lo).to(Uint32)
                    wb = (window >> shift_hi).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 2)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc0[2 * n16, 0], acc0[2 * n16, 1], acc0[2 * n16, 2], acc0[2 * n16, 3],
                        a0_0, a1_0, a2_0, a3_0, b0, b1,
                    )
                    acc0[2 * n16, 0], acc0[2 * n16, 1], acc0[2 * n16, 2], acc0[2 * n16, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc0[2 * n16 + 1, 0], acc0[2 * n16 + 1, 1], acc0[2 * n16 + 1, 2], acc0[2 * n16 + 1, 3],
                        a0_0, a1_0, a2_0, a3_0, b2, b3,
                    )
                    acc0[2 * n16 + 1, 0], acc0[2 * n16 + 1, 1], acc0[2 * n16 + 1, 2], acc0[2 * n16 + 1, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc1[2 * n16, 0], acc1[2 * n16, 1], acc1[2 * n16, 2], acc1[2 * n16, 3],
                        a0_1, a1_1, a2_1, a3_1, b0, b1,
                    )
                    acc1[2 * n16, 0], acc1[2 * n16, 1], acc1[2 * n16, 2], acc1[2 * n16, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc1[2 * n16 + 1, 0], acc1[2 * n16 + 1, 1], acc1[2 * n16 + 1, 2], acc1[2 * n16 + 1, 3],
                        a0_1, a1_1, a2_1, a3_1, b2, b3,
                    )
                    acc1[2 * n16 + 1, 0], acc1[2 * n16 + 1, 1], acc1[2 * n16 + 1, 2], acc1[2 * n16 + 1, 3] = d0, d1, d2, d3
        else:
            ia_hi = ia + Int32(8)
            ib_hi = ib + Int32(8)
            pred_lo = (lane < Int32(24)).to(Int32)
            pred_hi = (lane >= Int32(8)).to(Int32)
            w0_buf = cute.make_rmem_tensor(8, Uint32)
            w1_buf = cute.make_rmem_tensor(8, Uint32)
            for k16 in cutlass.range_constexpr(8):
                w0_buf[k16] = _ld_global_nc_u32_pred_l2hint(b_ptr, pred_lo, l2_policy, k16 * 768)
                w1_buf[k16] = _ld_global_nc_u32_pred_l2hint(b_ptr, pred_hi, l2_policy, k16 * 768 + 64)
            for k16 in cutlass.range_constexpr(8):
                a0_0 = _ld_global_nc_u32_pred_offset(a_base0_g0, valid0_g0, k16 * 32)
                a2_0 = _ld_global_nc_u32_pred_offset(a_base0_g0, valid0_g0, k16 * 32 + 16)
                a1_0 = _ld_global_nc_u32_pred_offset(a_base1_g0, valid1_g0, k16 * 32)
                a3_0 = _ld_global_nc_u32_pred_offset(a_base1_g0, valid1_g0, k16 * 32 + 16)
                a0_1 = _ld_global_nc_u32_pred_offset(a_base0_g1, valid0_g1, k16 * 32)
                a2_1 = _ld_global_nc_u32_pred_offset(a_base0_g1, valid0_g1, k16 * 32 + 16)
                a1_1 = _ld_global_nc_u32_pred_offset(a_base1_g1, valid1_g1, k16 * 32)
                a3_1 = _ld_global_nc_u32_pred_offset(a_base1_g1, valid1_g1, k16 * 32 + 16)
                for n16 in cutlass.range_constexpr(2):
                    w = w0_buf[k16] if cutlass.const_expr(n16 == 0) else w1_buf[k16]
                    src_a = ia if cutlass.const_expr(n16 == 0) else ia_hi
                    src_b = ib if cutlass.const_expr(n16 == 0) else ib_hi
                    a = cute.arch.shuffle_sync(w, src_a)
                    b = cute.arch.shuffle_sync(w, src_b)
                    window = (a.to(Int64) << Int64(32)) | b.to(Int64)
                    wa = (window >> shift_lo).to(Uint32)
                    wb = (window >> shift_hi).to(Uint32)
                    b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, 3)
                    b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                    b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc0[2 * n16, 0], acc0[2 * n16, 1], acc0[2 * n16, 2], acc0[2 * n16, 3],
                        a0_0, a1_0, a2_0, a3_0, b0, b1,
                    )
                    acc0[2 * n16, 0], acc0[2 * n16, 1], acc0[2 * n16, 2], acc0[2 * n16, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc0[2 * n16 + 1, 0], acc0[2 * n16 + 1, 1], acc0[2 * n16 + 1, 2], acc0[2 * n16 + 1, 3],
                        a0_0, a1_0, a2_0, a3_0, b2, b3,
                    )
                    acc0[2 * n16 + 1, 0], acc0[2 * n16 + 1, 1], acc0[2 * n16 + 1, 2], acc0[2 * n16 + 1, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc1[2 * n16, 0], acc1[2 * n16, 1], acc1[2 * n16, 2], acc1[2 * n16, 3],
                        a0_1, a1_1, a2_1, a3_1, b0, b1,
                    )
                    acc1[2 * n16, 0], acc1[2 * n16, 1], acc1[2 * n16, 2], acc1[2 * n16, 3] = d0, d1, d2, d3
                    d0, d1, d2, d3 = f16_mma_m16n8k16_f32(
                        acc1[2 * n16 + 1, 0], acc1[2 * n16 + 1, 1], acc1[2 * n16 + 1, 2], acc1[2 * n16 + 1, 3],
                        a0_1, a1_1, a2_1, a3_1, b2, b3,
                    )
                    acc1[2 * n16 + 1, 0], acc1[2 * n16 + 1, 1], acc1[2 * n16 + 1, 2], acc1[2 * n16 + 1, 3] = d0, d1, d2, d3
        for n in cutlass.range_constexpr(4):
            for v in cutlass.range_constexpr(4):
                total0[n, v] = total0[n, v] + acc0[n, v] * scale
                total1[n, v] = total1[n, v] + acc1[n, v] * scale

    @cute.jit
    def _run_fc1_proj_pair_groups(
        self,
        total0: cute.Tensor,
        total1: cute.Tensor,
        width: Int32,
        payload: Int64,
        scales: Int64,
        x_row0_g0: Int64,
        x_row1_g0: Int64,
        valid0_g0: Int32,
        valid1_g0: Int32,
        x_row0_g1: Int64,
        x_row1_g1: Int64,
        valid0_g1: Int32,
        valid1_g1: Int32,
        split: Int32,
        lane: Int32,
        warp: Int32,
        prev_lane: Int32,
        ia2: Int32,
        ib2: Int32,
        shift2_lo: Int64,
        shift2_hi: Int64,
        ia3: Int32,
        ib3: Int32,
        shift3_lo: Int64,
        shift3_hi: Int64,
        policy: Uint64,
    ):
        if width == Int32(2):
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    scale = _side_value(scales, kb)
                    kb_off = kb.to(Int64) * Int64(256)
                    self._accumulate_pair_groups_block(
                        total0, total1, payload,
                        x_row0_g0 + kb_off, x_row1_g0 + kb_off, valid0_g0, valid1_g0,
                        x_row0_g1 + kb_off, x_row1_g1 + kb_off, valid0_g1, valid1_g1,
                        kb, lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi, 2, scale, policy,
                    )
        elif width == Int32(3):
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    scale = _side_value(scales, kb)
                    kb_off = kb.to(Int64) * Int64(256)
                    self._accumulate_pair_groups_block(
                        total0, total1, payload,
                        x_row0_g0 + kb_off, x_row1_g0 + kb_off, valid0_g0, valid1_g0,
                        x_row0_g1 + kb_off, x_row1_g1 + kb_off, valid0_g1, valid1_g1,
                        kb, lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi, 3, scale, policy,
                    )
        elif width == Int32(4):
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    scale = _side_value(scales, kb)
                    kb_off = kb.to(Int64) * Int64(256)
                    self._accumulate_pair_groups_block(
                        total0, total1, payload,
                        x_row0_g0 + kb_off, x_row1_g0 + kb_off, valid0_g0, valid1_g0,
                        x_row0_g1 + kb_off, x_row1_g1 + kb_off, valid0_g1, valid1_g1,
                        kb, lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0), 4, scale, policy,
                    )

    @cute.jit
    def _run_fc1_proj(
        self,
        total: cute.Tensor,
        width: Int32,
        payload: Int64,
        scales: Int64,
        x_row0_base: Int64,
        x_row1_base: Int64,
        valid0: Int32,
        valid1: Int32,
        split: Int32,
        lane: Int32,
        warp: Int32,
        prev_lane: Int32,
        ia2: Int32,
        ib2: Int32,
        shift2_lo: Int64,
        shift2_hi: Int64,
        ia3: Int32,
        ib3: Int32,
        shift3_lo: Int64,
        shift3_hi: Int64,
        policy: Uint64,
    ):
        if width == Int32(2):
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    scale = _side_value(scales, kb)
                    kb_off = kb.to(Int64) * Int64(256)
                    self._fc1_accumulate_block(
                        total, payload, x_row0_base + kb_off, x_row1_base + kb_off,
                        valid0, valid1, kb, lane, warp, prev_lane,
                        ia2, ib2, shift2_lo, shift2_hi, 2, scale, policy,
                    )
        elif width == Int32(3):
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    scale = _side_value(scales, kb)
                    kb_off = kb.to(Int64) * Int64(256)
                    self._fc1_accumulate_block(
                        total, payload, x_row0_base + kb_off, x_row1_base + kb_off,
                        valid0, valid1, kb, lane, warp, prev_lane,
                        ia3, ib3, shift3_lo, shift3_hi, 3, scale, policy,
                    )
        elif width == Int32(4):
            for block in cutlass.range(8):
                kb = split * Int32(8) + block
                if kb < Int32(self.blocks):
                    scale = _side_value(scales, kb)
                    kb_off = kb.to(Int64) * Int64(256)
                    self._fc1_accumulate_block(
                        total, payload, x_row0_base + kb_off, x_row1_base + kb_off,
                        valid0, valid1, kb, lane, warp, prev_lane,
                        Int32(0), Int32(0), Int64(0), Int64(0), 4, scale, policy,
                    )

    @cute.kernel
    def fc1_kernel(
        self,
        x: cute.Tensor,
        pointers: cute.Tensor,
        bits: cute.Tensor,
        group_expert: cute.Tensor,
        route_slots: cute.Tensor,
        fc1: cute.Tensor,
        rows: Int32,
        groups: cute.Tensor,
    ):
        live_groups = groups[0]
        if live_groups < Int32(0) or live_groups > Int32(self.max_groups):
            live_groups = Int32(0)
        if live_groups > Int32(0):
            smem = utils.SmemAllocator()
            s_meta = smem.allocate_tensor(Uint32, cute.make_layout(self.max_groups), byte_alignment=16)
            s_offset = smem.allocate_tensor(Int32, cute.make_layout(self.max_groups), byte_alignment=16)
            s_gu = smem.allocate_tensor(Int32, cute.make_layout(self.SMEM_GU_CAP), byte_alignment=16)
            s_total = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
            tid, _, _ = cute.arch.thread_idx()
            task, _, _ = cute.arch.block_idx()
            step, _, _ = cute.arch.grid_dim()
            lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
            total_units = self._build_live_units(
                bits, group_expert, live_groups, Int32(tid), lane, warp, s_meta, s_offset, s_gu, s_total
            )
            policy = create_l2_evict_first_policy()
            r = lane >> Int32(2)
            col_in_warp = warp * Int32(32) + (lane & Int32(3)) * Int32(2)
            x_lane_off = ((lane & Int32(3)) * Int32(2)).to(Int64) * Int64(2)
            x_addr = x.iterator.toint() + x_lane_off
            fc1_addr = fc1.iterator.toint()
            max_slot = rows * Int32(self.top_k)
            prev_lane = (lane + Int32(31)) & Int32(31)
            ia2, ib2, shift2, _ = trellis256_lane_geom_bits(lane, 0, 8, 2)
            shift2_lo, shift2_hi = shift2.to(Int64), (shift2 + Int32(8)).to(Int64)
            ia3, ib3, shift3, _ = trellis256_lane_geom_bits(lane, 0, 8, 3)
            shift3_lo, shift3_hi = shift3.to(Int64), (shift3 + Int32(12)).to(Int64)

            if cutlass.const_expr(self.in_cta_fold):
                if total_units >= Int32(self.IN_CTA_FOLD_MIN_UNITS):
                    while task < total_units * Int32(2):
                        proj = task & Int32(1)
                        u_idx = task >> Int32(1)
                        if cutlass.const_expr(self.supergroup):
                            unit, group, expert, has_g1, skip_g, owned_cnt = self._lookup_live_unit_ext(
                                u_idx, live_groups, total_units, s_gu, s_offset, s_meta
                            )
                            if skip_g == Int32(0):
                                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
                                width = bits[desc]
                                slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                                slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                                valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                                valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                                row0 = max(slot0 // Int32(self.top_k), Int32(0))
                                row1 = max(slot1 // Int32(self.top_k), Int32(0))
                                x_row0_base = x_addr + row0.to(Int64) * Int64(self.hidden_size * 2)
                                x_row1_base = x_addr + row1.to(Int64) * Int64(self.hidden_size * 2)
                                if has_g1 != Int32(0):
                                    g1 = group + Int32(1)
                                    slot0_g1 = route_slots[g1.to(Int64) * Int64(16) + r.to(Int64)]
                                    slot1_g1 = route_slots[g1.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                                    valid0_g1 = ((slot0_g1 >= Int32(0)) & (slot0_g1 < max_slot)).to(Int32)
                                    valid1_g1 = ((slot1_g1 >= Int32(0)) & (slot1_g1 < max_slot)).to(Int32)
                                    row0_g1 = max(slot0_g1 // Int32(self.top_k), Int32(0))
                                    row1_g1 = max(slot1_g1 // Int32(self.top_k), Int32(0))
                                    x_row0_g1 = x_addr + row0_g1.to(Int64) * Int64(self.hidden_size * 2)
                                    x_row1_g1 = x_addr + row1_g1.to(Int64) * Int64(self.hidden_size * 2)
                                    fc1_task0 = u_idx * Int32(2) + proj
                                    fc1_task1 = (u_idx + owned_cnt) * Int32(2) + proj
                                    grand0 = cute.make_rmem_tensor((4, 4), Float32)
                                    grand1 = cute.make_rmem_tensor((4, 4), Float32)
                                    grand0.fill(Float32(0.0))
                                    grand1.fill(Float32(0.0))
                                    if width != Int32(0):
                                        payload = pointers[desc * Int64(3)]
                                        scales = pointers[desc * Int64(3) + Int64(2)]
                                        for split in cutlass.range(self.splits):
                                            total0 = cute.make_rmem_tensor((4, 4), Float32)
                                            total1 = cute.make_rmem_tensor((4, 4), Float32)
                                            total0.fill(Float32(0.0))
                                            total1.fill(Float32(0.0))
                                            self._run_fc1_proj_pair_groups(
                                                total0, total1, width, payload, scales,
                                                x_row0_base, x_row1_base, valid0, valid1,
                                                x_row0_g1, x_row1_g1, valid0_g1, valid1_g1,
                                                split, lane, warp, prev_lane,
                                                ia2, ib2, shift2_lo, shift2_hi, ia3, ib3, shift3_lo, shift3_hi, policy,
                                            )
                                            for n in cutlass.range_constexpr(4):
                                                for v in cutlass.range_constexpr(4):
                                                    grand0[n, v] = grand0[n, v] + total0[n, v]
                                                    grand1[n, v] = grand1[n, v] + total1[n, v]
                                    for n in cutlass.range_constexpr(4):
                                        col = col_in_warp + Int32(n * 8)
                                        base0 = fc1_addr + (fc1_task0.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                        base1 = fc1_addr + (fc1_task1.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                        if width != Int32(0) and valid0 != Int32(0):
                                            st_global_v2_f32(base0, grand0[n, 0], grand0[n, 1])
                                        if width != Int32(0) and valid1 != Int32(0):
                                            st_global_v2_f32(base0 + Int64(8 * 128 * 4), grand0[n, 2], grand0[n, 3])
                                        if width != Int32(0) and valid0_g1 != Int32(0):
                                            st_global_v2_f32(base1, grand1[n, 0], grand1[n, 1])
                                        if width != Int32(0) and valid1_g1 != Int32(0):
                                            st_global_v2_f32(base1 + Int64(8 * 128 * 4), grand1[n, 2], grand1[n, 3])
                                else:
                                    grand = cute.make_rmem_tensor((4, 4), Float32)
                                    grand.fill(Float32(0.0))
                                    if width != Int32(0):
                                        payload = pointers[desc * Int64(3)]
                                        scales = pointers[desc * Int64(3) + Int64(2)]
                                        for split in cutlass.range(self.splits):
                                            total = cute.make_rmem_tensor((4, 4), Float32)
                                            total.fill(Float32(0.0))
                                            self._run_fc1_proj(
                                                total, width, payload, scales, x_row0_base, x_row1_base,
                                                valid0, valid1, split, lane, warp, prev_lane,
                                                ia2, ib2, shift2_lo, shift2_hi, ia3, ib3, shift3_lo, shift3_hi, policy,
                                            )
                                            for n in cutlass.range_constexpr(4):
                                                for v in cutlass.range_constexpr(4):
                                                    grand[n, v] = grand[n, v] + total[n, v]
                                    for n in cutlass.range_constexpr(4):
                                        col = col_in_warp + Int32(n * 8)
                                        base = fc1_addr + (task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                        if width != Int32(0) and valid0 != Int32(0):
                                            st_global_v2_f32(base, grand[n, 0], grand[n, 1])
                                        if width != Int32(0) and valid1 != Int32(0):
                                            st_global_v2_f32(base + Int64(8 * 128 * 4), grand[n, 2], grand[n, 3])
                        else:
                            unit, group, expert = self._lookup_live_unit(
                                u_idx, live_groups, total_units, s_gu, s_offset, s_meta
                            )
                            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
                            width = bits[desc]
                            grand = cute.make_rmem_tensor((4, 4), Float32)
                            grand.fill(Float32(0.0))
                            slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                            slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                            valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                            valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                            row0 = max(slot0 // Int32(self.top_k), Int32(0))
                            row1 = max(slot1 // Int32(self.top_k), Int32(0))
                            x_row0_base = x_addr + row0.to(Int64) * Int64(self.hidden_size * 2)
                            x_row1_base = x_addr + row1.to(Int64) * Int64(self.hidden_size * 2)
                            if width != Int32(0):
                                payload = pointers[desc * Int64(3)]
                                scales = pointers[desc * Int64(3) + Int64(2)]
                                for split in cutlass.range(self.splits):
                                    total = cute.make_rmem_tensor((4, 4), Float32)
                                    total.fill(Float32(0.0))
                                    self._run_fc1_proj(
                                        total, width, payload, scales, x_row0_base, x_row1_base,
                                        valid0, valid1, split, lane, warp, prev_lane,
                                        ia2, ib2, shift2_lo, shift2_hi, ia3, ib3, shift3_lo, shift3_hi, policy,
                                    )
                                    for n in cutlass.range_constexpr(4):
                                        for v in cutlass.range_constexpr(4):
                                            grand[n, v] = grand[n, v] + total[n, v]
                            for n in cutlass.range_constexpr(4):
                                col = col_in_warp + Int32(n * 8)
                                base = fc1_addr + (task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                if width != Int32(0) and valid0 != Int32(0):
                                    st_global_v2_f32(base, grand[n, 0], grand[n, 1])
                                if width != Int32(0) and valid1 != Int32(0):
                                    st_global_v2_f32(base + Int64(8 * 128 * 4), grand[n, 2], grand[n, 3])
                        task += step
                    task = total_units * Int32(2 * self.splits)

            if cutlass.const_expr(self.paired_fc1):
                if total_units * Int32(self.splits) >= Int32(step):
                    while task < total_units * Int32(self.splits):
                        split = task % Int32(self.splits)
                        u_idx = task // Int32(self.splits)
                        unit, group, expert = self._lookup_live_unit(
                            u_idx, live_groups, total_units, s_gu, s_offset, s_meta
                        )
                        slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                        slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                        valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                        valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                        row0 = max(slot0 // Int32(self.top_k), Int32(0))
                        row1 = max(slot1 // Int32(self.top_k), Int32(0))
                        x_row0_base = x_addr + row0.to(Int64) * Int64(self.hidden_size * 2)
                        x_row1_base = x_addr + row1.to(Int64) * Int64(self.hidden_size * 2)
                        desc_base = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)

                        if cutlass.const_expr(self.reg_pair_fc1):
                            w_g = bits[desc_base]
                            w_u = bits[desc_base + Int64(1)]
                            pay_g = pointers[desc_base * Int64(3)]
                            sc_g = pointers[desc_base * Int64(3) + Int64(2)]
                            pay_u = pointers[(desc_base + Int64(1)) * Int64(3)]
                            sc_u = pointers[(desc_base + Int64(1)) * Int64(3) + Int64(2)]
                            total_g = cute.make_rmem_tensor((4, 4), Float32)
                            total_u = cute.make_rmem_tensor((4, 4), Float32)
                            total_g.fill(Float32(0.0))
                            total_u.fill(Float32(0.0))
                            for block in cutlass.range(8):
                                kb = split * Int32(8) + block
                                if kb < Int32(self.blocks):
                                    kb_off = kb.to(Int64) * Int64(256)
                                    a0_b, a1_b, a2_b, a3_b = self._load_a_block(
                                        x_row0_base + kb_off, x_row1_base + kb_off, valid0, valid1
                                    )
                                    if w_g != Int32(0):
                                        scale_g = _side_value(sc_g, kb)
                                        if w_g == Int32(2):
                                            self._mma_with_loaded_a(
                                                total_g, pay_g, a0_b, a1_b, a2_b, a3_b, kb,
                                                lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                                2, scale_g, policy,
                                            )
                                        elif w_g == Int32(3):
                                            self._mma_with_loaded_a(
                                                total_g, pay_g, a0_b, a1_b, a2_b, a3_b, kb,
                                                lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                                3, scale_g, policy,
                                            )
                                        elif w_g == Int32(4):
                                            self._mma_with_loaded_a(
                                                total_g, pay_g, a0_b, a1_b, a2_b, a3_b, kb,
                                                lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                                4, scale_g, policy,
                                            )
                                    if w_u != Int32(0):
                                        scale_u = _side_value(sc_u, kb)
                                        if w_u == Int32(2):
                                            self._mma_with_loaded_a(
                                                total_u, pay_u, a0_b, a1_b, a2_b, a3_b, kb,
                                                lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                                2, scale_u, policy,
                                            )
                                        elif w_u == Int32(3):
                                            self._mma_with_loaded_a(
                                                total_u, pay_u, a0_b, a1_b, a2_b, a3_b, kb,
                                                lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                                3, scale_u, policy,
                                            )
                                        elif w_u == Int32(4):
                                            self._mma_with_loaded_a(
                                                total_u, pay_u, a0_b, a1_b, a2_b, a3_b, kb,
                                                lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                                4, scale_u, policy,
                                            )
                            fc1_g_task = (u_idx * Int32(2)) * Int32(self.splits) + split
                            fc1_u_task = (u_idx * Int32(2) + Int32(1)) * Int32(self.splits) + split
                            for n in cutlass.range_constexpr(4):
                                col = col_in_warp + Int32(n * 8)
                                base_g = fc1_addr + (fc1_g_task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                base_u = fc1_addr + (fc1_u_task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                if w_g != Int32(0) and valid0 != Int32(0):
                                    st_global_v2_f32(base_g, total_g[n, 0], total_g[n, 1])
                                if w_g != Int32(0) and valid1 != Int32(0):
                                    st_global_v2_f32(base_g + Int64(8 * 128 * 4), total_g[n, 2], total_g[n, 3])
                                if w_u != Int32(0) and valid0 != Int32(0):
                                    st_global_v2_f32(base_u, total_u[n, 0], total_u[n, 1])
                                if w_u != Int32(0) and valid1 != Int32(0):
                                    st_global_v2_f32(base_u + Int64(8 * 128 * 4), total_u[n, 2], total_u[n, 3])
                        else:
                            for proj in cutlass.range(2):
                                desc = desc_base + proj.to(Int64)
                                width = bits[desc]
                                total = cute.make_rmem_tensor((4, 4), Float32)
                                total.fill(Float32(0.0))
                                if width != Int32(0):
                                    payload = pointers[desc * Int64(3)]
                                    scales = pointers[desc * Int64(3) + Int64(2)]
                                    self._run_fc1_proj(
                                        total, width, payload, scales, x_row0_base, x_row1_base,
                                        valid0, valid1, split, lane, warp, prev_lane,
                                        ia2, ib2, shift2_lo, shift2_hi, ia3, ib3, shift3_lo, shift3_hi, policy,
                                    )
                                fc1_task = (u_idx * Int32(2) + proj) * Int32(self.splits) + split
                                for n in cutlass.range_constexpr(4):
                                    col = col_in_warp + Int32(n * 8)
                                    base = fc1_addr + (fc1_task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                                    if width != Int32(0) and valid0 != Int32(0):
                                        st_global_v2_f32(base, total[n, 0], total[n, 1])
                                    if width != Int32(0) and valid1 != Int32(0):
                                        st_global_v2_f32(base + Int64(8 * 128 * 4), total[n, 2], total[n, 3])
                        task += step
                else:
                    while task < total_units * Int32(2 * self.splits):
                        split = task % Int32(self.splits)
                        proj = (task // Int32(self.splits)) % Int32(2)
                        u_idx = task // Int32(2 * self.splits)
                        unit, group, expert = self._lookup_live_unit(
                            u_idx, live_groups, total_units, s_gu, s_offset, s_meta
                        )
                        desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
                        width = bits[desc]
                        total = cute.make_rmem_tensor((4, 4), Float32)
                        total.fill(Float32(0.0))
                        slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                        slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                        valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                        valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                        row0 = max(slot0 // Int32(self.top_k), Int32(0))
                        row1 = max(slot1 // Int32(self.top_k), Int32(0))
                        x_row0_base = x_addr + row0.to(Int64) * Int64(self.hidden_size * 2)
                        x_row1_base = x_addr + row1.to(Int64) * Int64(self.hidden_size * 2)
                        if width != Int32(0):
                            payload = pointers[desc * Int64(3)]
                            scales = pointers[desc * Int64(3) + Int64(2)]
                            self._run_fc1_proj(
                                total, width, payload, scales, x_row0_base, x_row1_base,
                                valid0, valid1, split, lane, warp, prev_lane,
                                ia2, ib2, shift2_lo, shift2_hi, ia3, ib3, shift3_lo, shift3_hi, policy,
                            )
                        for n in cutlass.range_constexpr(4):
                            col = col_in_warp + Int32(n * 8)
                            base = fc1_addr + (task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                            if width != Int32(0) and valid0 != Int32(0):
                                st_global_v2_f32(base, total[n, 0], total[n, 1])
                            if width != Int32(0) and valid1 != Int32(0):
                                st_global_v2_f32(base + Int64(8 * 128 * 4), total[n, 2], total[n, 3])
                        task += step
            else:
                while task < total_units * Int32(2 * self.splits):
                    split = task % Int32(self.splits)
                    proj = (task // Int32(self.splits)) % Int32(2)
                    u_idx = task // Int32(2 * self.splits)
                    unit, group, expert = self._lookup_live_unit(
                        u_idx, live_groups, total_units, s_gu, s_offset, s_meta
                    )
                    desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + proj.to(Int64)
                    width = bits[desc]
                    total = cute.make_rmem_tensor((4, 4), Float32)
                    total.fill(Float32(0.0))
                    slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                    slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                    valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                    valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                    row0 = max(slot0 // Int32(self.top_k), Int32(0))
                    row1 = max(slot1 // Int32(self.top_k), Int32(0))
                    x_row0_base = x_addr + row0.to(Int64) * Int64(self.hidden_size * 2)
                    x_row1_base = x_addr + row1.to(Int64) * Int64(self.hidden_size * 2)
                    if width != Int32(0):
                        payload = pointers[desc * Int64(3)]
                        scales = pointers[desc * Int64(3) + Int64(2)]
                        self._run_fc1_proj(
                            total, width, payload, scales, x_row0_base, x_row1_base,
                            valid0, valid1, split, lane, warp, prev_lane,
                            ia2, ib2, shift2_lo, shift2_hi, ia3, ib3, shift3_lo, shift3_hi, policy,
                        )
                    for n in cutlass.range_constexpr(4):
                        col = col_in_warp + Int32(n * 8)
                        base = fc1_addr + (task.to(Int64) * Int64(16 * 128) + r.to(Int64) * Int64(128) + col.to(Int64)) * Int64(4)
                        if width != Int32(0) and valid0 != Int32(0):
                            st_global_v2_f32(base, total[n, 0], total[n, 1])
                        if width != Int32(0) and valid1 != Int32(0):
                            st_global_v2_f32(base + Int64(8 * 128 * 4), total[n, 2], total[n, 3])
                    task += step

    @cute.kernel
    def activation_kernel(
        self,
        pointers: cute.Tensor,
        bits: cute.Tensor,
        group_expert: cute.Tensor,
        route_slots: cute.Tensor,
        fc1: cute.Tensor,
        intermediate: cute.Tensor,
        rows: Int32,
        groups: cute.Tensor,
    ):
        live_groups = groups[0]
        if live_groups < Int32(0) or live_groups > Int32(self.max_groups):
            live_groups = Int32(0)
        if live_groups > Int32(0):
            smem = utils.SmemAllocator()
            s_meta = smem.allocate_tensor(Uint32, cute.make_layout(self.max_groups), byte_alignment=16)
            s_offset = smem.allocate_tensor(Int32, cute.make_layout(self.max_groups), byte_alignment=16)
            s_gu = smem.allocate_tensor(Int32, cute.make_layout(self.SMEM_GU_CAP), byte_alignment=16)
            s_total = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
            tid, _, _ = cute.arch.thread_idx()
            task, _, _ = cute.arch.block_idx()
            step, _, _ = cute.arch.grid_dim()
            lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
            total_units = self._build_live_units(
                bits, group_expert, live_groups, Int32(tid), lane, warp, s_meta, s_offset, s_gu, s_total
            )
            fc1_addr = fc1.iterator.toint()
            inter_addr = intermediate.iterator.toint()
            max_slot = rows * Int32(self.top_k)
            while task < total_units:
                unit, group, expert = self._lookup_live_unit(
                    task, live_groups, total_units, s_gu, s_offset, s_meta
                )
                desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3)
                if bits[desc] != Int32(0):
                    gs = pointers[desc * Int64(3) + Int64(1)]
                    us = pointers[(desc + Int64(1)) * Int64(3) + Int64(1)]
                    ds = pointers[(desc + Int64(2)) * Int64(3) + Int64(1)]
                    gs_val = cute.make_rmem_tensor(4, Float32)
                    us_val = cute.make_rmem_tensor(4, Float32)
                    ds_val = cute.make_rmem_tensor(4, Float32)
                    for v in cutlass.range_constexpr(4):
                        col = lane * Int32(4) + Int32(v)
                        gs_val[v] = _side_value(gs, col)
                        us_val[v] = _side_value(us, col)
                        ds_val[v] = _side_value(ds, col)
                    inter_unit_base = inter_addr + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(self.group_rows * 128 * 2)
                    if cutlass.const_expr(self.in_cta_fold):
                        if total_units >= Int32(self.IN_CTA_FOLD_MIN_UNITS):
                            fc1_unit_base = fc1_addr + task.to(Int64) * Int64(2 * self.group_rows * 128 * 4)
                            for iteration in cutlass.range_constexpr(self.group_rows // 4):
                                row = warp + Int32(iteration * 4)
                                slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
                                if slot >= Int32(0) and slot < max_slot:
                                    g_acc = cute.make_rmem_tensor(4, Float32)
                                    u_acc = cute.make_rmem_tensor(4, Float32)
                                    row_col_off = (row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)) * Int64(4)
                                    g_addr = fc1_unit_base + row_col_off
                                    u_addr = g_addr + Int64(self.group_rows * 128 * 4)
                                    g0, g1, g2, g3 = ld_global_nc_v4_u32(g_addr)
                                    u0, u1, u2, u3 = ld_global_nc_v4_u32(u_addr)
                                    g_acc[0] = u32_as_f32(g0)
                                    g_acc[1] = u32_as_f32(g1)
                                    g_acc[2] = u32_as_f32(g2)
                                    g_acc[3] = u32_as_f32(g3)
                                    u_acc[0] = u32_as_f32(u0)
                                    u_acc[1] = u32_as_f32(u1)
                                    u_acc[2] = u32_as_f32(u2)
                                    u_acc[3] = u32_as_f32(u3)
                                    gh = _w4a8_had128_quad(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane)
                                    uh = _w4a8_had128_quad(u_acc[0], u_acc[1], u_acc[2], u_acc[3], lane)
                                    for v in cutlass.range_constexpr(4):
                                        gate = gh[v] * gs_val[v]
                                        up = uh[v] * us_val[v]
                                        sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                                        g_acc[v] = gate * sigmoid * up * ds_val[v]
                                    h0, h1, h2, h3 = _w4a8_had128_quad(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane)
                                    out_addr = inter_unit_base + (row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)) * Int64(2)
                                    store_f16x4(out_addr, h0, h1, h2, h3)
                        else:
                            fc1_unit_base = fc1_addr + task.to(Int64) * Int64(2 * self.splits * self.group_rows * 128 * 4)
                            for iteration in cutlass.range_constexpr(self.group_rows // 4):
                                row = warp + Int32(iteration * 4)
                                slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
                                if slot >= Int32(0) and slot < max_slot:
                                    g_acc = cute.make_rmem_tensor(4, Float32)
                                    u_acc = cute.make_rmem_tensor(4, Float32)
                                    g_acc.fill(Float32(0.0))
                                    u_acc.fill(Float32(0.0))
                                    row_col_off = (row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)) * Int64(4)
                                    for split in cutlass.range_constexpr(self.splits):
                                        g_addr = fc1_unit_base + Int64(split * self.group_rows * 128 * 4) + row_col_off
                                        u_addr = g_addr + Int64(self.splits * self.group_rows * 128 * 4)
                                        g0, g1, g2, g3 = ld_global_nc_v4_u32(g_addr)
                                        u0, u1, u2, u3 = ld_global_nc_v4_u32(u_addr)
                                        g_acc[0] = g_acc[0] + u32_as_f32(g0)
                                        g_acc[1] = g_acc[1] + u32_as_f32(g1)
                                        g_acc[2] = g_acc[2] + u32_as_f32(g2)
                                        g_acc[3] = g_acc[3] + u32_as_f32(g3)
                                        u_acc[0] = u_acc[0] + u32_as_f32(u0)
                                        u_acc[1] = u_acc[1] + u32_as_f32(u1)
                                        u_acc[2] = u_acc[2] + u32_as_f32(u2)
                                        u_acc[3] = u_acc[3] + u32_as_f32(u3)
                                    gh = _w4a8_had128_quad(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane)
                                    uh = _w4a8_had128_quad(u_acc[0], u_acc[1], u_acc[2], u_acc[3], lane)
                                    for v in cutlass.range_constexpr(4):
                                        gate = gh[v] * gs_val[v]
                                        up = uh[v] * us_val[v]
                                        sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                                        g_acc[v] = gate * sigmoid * up * ds_val[v]
                                    h0, h1, h2, h3 = _w4a8_had128_quad(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane)
                                    out_addr = inter_unit_base + (row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)) * Int64(2)
                                    store_f16x4(out_addr, h0, h1, h2, h3)
                    else:
                        fc1_unit_base = fc1_addr + task.to(Int64) * Int64(2 * self.splits * self.group_rows * 128 * 4)
                        for iteration in cutlass.range_constexpr(self.group_rows // 4):
                            row = warp + Int32(iteration * 4)
                            slot = route_slots[group.to(Int64) * Int64(self.group_rows) + row.to(Int64)]
                            if slot >= Int32(0) and slot < max_slot:
                                g_acc = cute.make_rmem_tensor(4, Float32)
                                u_acc = cute.make_rmem_tensor(4, Float32)
                                g_acc.fill(Float32(0.0))
                                u_acc.fill(Float32(0.0))
                                row_col_off = (row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)) * Int64(4)
                                for split in cutlass.range_constexpr(self.splits):
                                    g_addr = fc1_unit_base + Int64(split * self.group_rows * 128 * 4) + row_col_off
                                    u_addr = g_addr + Int64(self.splits * self.group_rows * 128 * 4)
                                    g0, g1, g2, g3 = ld_global_nc_v4_u32(g_addr)
                                    u0, u1, u2, u3 = ld_global_nc_v4_u32(u_addr)
                                    g_acc[0] = g_acc[0] + u32_as_f32(g0)
                                    g_acc[1] = g_acc[1] + u32_as_f32(g1)
                                    g_acc[2] = g_acc[2] + u32_as_f32(g2)
                                    g_acc[3] = g_acc[3] + u32_as_f32(g3)
                                    u_acc[0] = u_acc[0] + u32_as_f32(u0)
                                    u_acc[1] = u_acc[1] + u32_as_f32(u1)
                                    u_acc[2] = u_acc[2] + u32_as_f32(u2)
                                    u_acc[3] = u_acc[3] + u32_as_f32(u3)
                                gh = _w4a8_had128_quad(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane)
                                uh = _w4a8_had128_quad(u_acc[0], u_acc[1], u_acc[2], u_acc[3], lane)
                                for v in cutlass.range_constexpr(4):
                                    gate = gh[v] * gs_val[v]
                                    up = uh[v] * us_val[v]
                                    sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math))
                                    g_acc[v] = gate * sigmoid * up * ds_val[v]
                                h0, h1, h2, h3 = _w4a8_had128_quad(g_acc[0], g_acc[1], g_acc[2], g_acc[3], lane)
                                out_addr = inter_unit_base + (row.to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)) * Int64(2)
                                store_f16x4(out_addr, h0, h1, h2, h3)
                task += step

    @cute.kernel
    def fc2_kernel(
        self,
        pointers: cute.Tensor,
        bits: cute.Tensor,
        group_expert: cute.Tensor,
        route_slots: cute.Tensor,
        intermediate: cute.Tensor,
        fc2: cute.Tensor,
        rows: Int32,
        groups: cute.Tensor,
    ):
        live_groups = groups[0]
        if live_groups < Int32(0) or live_groups > Int32(self.max_groups):
            live_groups = Int32(0)
        if live_groups > Int32(0):
            smem = utils.SmemAllocator()
            s_meta = smem.allocate_tensor(Uint32, cute.make_layout(self.max_groups), byte_alignment=16)
            tid, _, _ = cute.arch.thread_idx()
            task, _, _ = cute.arch.block_idx()
            step, _, _ = cute.arch.grid_dim()
            lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
            half_warp = Int32(tid) >> Int32(4)
            u_tid = Int32(tid) & Int32(15)
            half_shift = Uint32(Int32(tid) & Int32(16))
            g_base = Int32(0)
            while g_base < live_groups:
                g = g_base + half_warp
                owned = Int32(0)
                expert = Int32(0)
                if g < live_groups:
                    expert = group_expert[g]
                    if u_tid < Int32(self.units) and expert >= Int32(0):
                        desc = (expert.to(Int64) * Int64(self.units) + u_tid.to(Int64)) * Int64(3) + Int64(2)
                        if bits[desc] != Int32(0):
                            owned = Int32(1)
                ballot = _vote_ballot_sync(owned)
                if g < live_groups and u_tid == Int32(0):
                    mask16 = (ballot >> half_shift) & Uint32(0xFFFF)
                    s_meta[g] = (expert.to(Uint32) << Uint32(16)) | mask16
                g_base += Int32(8)
            cute.arch.sync_threads()

            r = lane >> Int32(2)
            a_lane_off = (r * Int32(128) + (lane & Int32(3)) * Int32(2)).to(Int64) * Int64(2)
            inter_base = intermediate.iterator.toint() + a_lane_off
            fc2_addr = fc2.iterator.toint()
            l2_policy = create_l2_evict_first_policy()
            max_slot = rows * Int32(self.top_k)
            prev_lane = (lane + Int32(31)) & Int32(31)
            ia2, ib2, shift2, _ = trellis256_lane_geom_bits(lane, 0, 8, 2)
            shift2_lo, shift2_hi = shift2.to(Int64), (shift2 + Int32(8)).to(Int64)
            ia3, ib3, shift3, _ = trellis256_lane_geom_bits(lane, 0, 8, 3)
            shift3_lo, shift3_hi = shift3.to(Int64), (shift3 + Int32(12)).to(Int64)

            if cutlass.const_expr(self.pair_fc2):
                half_blocks = Int32(self.blocks // 2)
                if live_groups * half_blocks >= Int32(step):
                    while task < live_groups * half_blocks:
                        group, bp = task // half_blocks, task % half_blocks
                        b0 = bp * Int32(2)
                        b1 = b0 + Int32(1)
                        meta = s_meta[group]
                        expert = (meta >> Uint32(16)).to(Int32)
                        mask = meta & Uint32(0xFFFF)
                        total0 = cute.make_rmem_tensor((4, 4), Float32)
                        total1 = cute.make_rmem_tensor((4, 4), Float32)
                        total0.fill(Float32(0.0))
                        total1.fill(Float32(0.0))
                        slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                        slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                        valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                        valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                        while mask != Uint32(0):
                            unit = _ctz_u32(mask)
                            mask = mask & (mask - Uint32(1))
                            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                            width = bits[desc]
                            if width != Int32(0):
                                payload = pointers[desc * Int64(3)]
                                sc_ptr = pointers[desc * Int64(3) + Int64(2)]
                                scale0 = _side_value(sc_ptr, b0)
                                scale1 = _side_value(sc_ptr, b1)
                                a_base = inter_base + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                                a0_b, a1_b, a2_b, a3_b = self._load_a_block(
                                    a_base, a_base + Int64(2048), valid0, valid1
                                )
                                if width == Int32(2):
                                    self._mma_with_loaded_a(
                                        total0, payload, a0_b, a1_b, a2_b, a3_b, b0,
                                        lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                        2, scale0, l2_policy,
                                    )
                                    self._mma_with_loaded_a(
                                        total1, payload, a0_b, a1_b, a2_b, a3_b, b1,
                                        lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                        2, scale1, l2_policy,
                                    )
                                elif width == Int32(3):
                                    self._mma_with_loaded_a(
                                        total0, payload, a0_b, a1_b, a2_b, a3_b, b0,
                                        lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                        3, scale0, l2_policy,
                                    )
                                    self._mma_with_loaded_a(
                                        total1, payload, a0_b, a1_b, a2_b, a3_b, b1,
                                        lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                        3, scale1, l2_policy,
                                    )
                                elif width == Int32(4):
                                    self._mma_with_loaded_a(
                                        total0, payload, a0_b, a1_b, a2_b, a3_b, b0,
                                        lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                        4, scale0, l2_policy,
                                    )
                                    self._mma_with_loaded_a(
                                        total1, payload, a0_b, a1_b, a2_b, a3_b, b1,
                                        lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                        4, scale1, l2_policy,
                                    )
                        for n in cutlass.range_constexpr(4):
                            col0 = b0 * Int32(128) + warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                            col1 = col0 + Int32(128)
                            base0 = fc2_addr + ((group.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col0.to(Int64)) * Int64(4)
                            base1 = fc2_addr + ((group.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col1.to(Int64)) * Int64(4)
                            if valid0 != Int32(0):
                                st_global_v2_f32(base0, total0[n, 0], total0[n, 1])
                                st_global_v2_f32(base1, total1[n, 0], total1[n, 1])
                            if valid1 != Int32(0):
                                st_global_v2_f32(base0 + Int64(8 * self.hidden_size * 4), total0[n, 2], total0[n, 3])
                                st_global_v2_f32(base1 + Int64(8 * self.hidden_size * 4), total1[n, 2], total1[n, 3])
                        task += step
                else:
                    while task < live_groups * Int32(self.blocks):
                        group, block = task // Int32(self.blocks), task % Int32(self.blocks)
                        meta = s_meta[group]
                        expert = (meta >> Uint32(16)).to(Int32)
                        mask = meta & Uint32(0xFFFF)
                        total = cute.make_rmem_tensor((4, 4), Float32)
                        total.fill(Float32(0.0))
                        slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                        slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                        valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                        valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                        while mask != Uint32(0):
                            unit = _ctz_u32(mask)
                            mask = mask & (mask - Uint32(1))
                            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                            width = bits[desc]
                            if width != Int32(0):
                                payload = pointers[desc * Int64(3)]
                                scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                                a_base = inter_base + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                                if width == Int32(2):
                                    self._fc1_accumulate_block(
                                        total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                        block, lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                        2, scale, l2_policy,
                                    )
                                elif width == Int32(3):
                                    self._fc1_accumulate_block(
                                        total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                        block, lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                        3, scale, l2_policy,
                                    )
                                elif width == Int32(4):
                                    self._fc1_accumulate_block(
                                        total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                        block, lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                        4, scale, l2_policy,
                                    )
                        for n in cutlass.range_constexpr(4):
                            col = block * Int32(128) + warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                            base = fc2_addr + ((group.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col.to(Int64)) * Int64(4)
                            if valid0 != Int32(0):
                                st_global_v2_f32(base, total[n, 0], total[n, 1])
                            if valid1 != Int32(0):
                                st_global_v2_f32(base + Int64(8 * self.hidden_size * 4), total[n, 2], total[n, 3])
                        task += step
            else:
                while task < live_groups * Int32(self.blocks):
                    group, block = task // Int32(self.blocks), task % Int32(self.blocks)
                    meta = s_meta[group]
                    expert = (meta >> Uint32(16)).to(Int32)
                    mask = meta & Uint32(0xFFFF)
                    if cutlass.const_expr(self.supergroup):
                        run_idx = Int32(0)
                        k = group - Int32(1)
                        while k >= Int32(0):
                            if (s_meta[k] >> Uint32(16)).to(Int32) == expert:
                                run_idx += Int32(1)
                                k -= Int32(1)
                            else:
                                k = Int32(-1)
                        skip_g = run_idx & Int32(1)
                        has_g1 = Int32(0)
                        if skip_g == Int32(0) and (group + Int32(1)) < live_groups:
                            if (s_meta[group + Int32(1)] >> Uint32(16)).to(Int32) == expert:
                                has_g1 = Int32(1)
                        if skip_g == Int32(0):
                            slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                            slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                            valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                            valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                            if has_g1 != Int32(0):
                                g1 = group + Int32(1)
                                slot0_g1 = route_slots[g1.to(Int64) * Int64(16) + r.to(Int64)]
                                slot1_g1 = route_slots[g1.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                                valid0_g1 = ((slot0_g1 >= Int32(0)) & (slot0_g1 < max_slot)).to(Int32)
                                valid1_g1 = ((slot1_g1 >= Int32(0)) & (slot1_g1 < max_slot)).to(Int32)
                                total0 = cute.make_rmem_tensor((4, 4), Float32)
                                total1 = cute.make_rmem_tensor((4, 4), Float32)
                                total0.fill(Float32(0.0))
                                total1.fill(Float32(0.0))
                                while mask != Uint32(0):
                                    unit = _ctz_u32(mask)
                                    mask = mask & (mask - Uint32(1))
                                    desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                                    width = bits[desc]
                                    if width != Int32(0):
                                        payload = pointers[desc * Int64(3)]
                                        scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                                        a_base0 = inter_base + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                                        a_base1 = inter_base + (g1.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                                        if width == Int32(2):
                                            self._accumulate_pair_groups_block(
                                                total0, total1, payload,
                                                a_base0, a_base0 + Int64(2048), valid0, valid1,
                                                a_base1, a_base1 + Int64(2048), valid0_g1, valid1_g1,
                                                block, lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                                2, scale, l2_policy,
                                            )
                                        elif width == Int32(3):
                                            self._accumulate_pair_groups_block(
                                                total0, total1, payload,
                                                a_base0, a_base0 + Int64(2048), valid0, valid1,
                                                a_base1, a_base1 + Int64(2048), valid0_g1, valid1_g1,
                                                block, lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                                3, scale, l2_policy,
                                            )
                                        elif width == Int32(4):
                                            self._accumulate_pair_groups_block(
                                                total0, total1, payload,
                                                a_base0, a_base0 + Int64(2048), valid0, valid1,
                                                a_base1, a_base1 + Int64(2048), valid0_g1, valid1_g1,
                                                block, lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                                4, scale, l2_policy,
                                            )
                                for n in cutlass.range_constexpr(4):
                                    col_in_block = warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                                    if valid0 != Int32(0):
                                        base0 = fc2_addr + ((block * max_slot + slot0).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                        st_global_v2_f32(base0, total0[n, 0], total0[n, 1])
                                    if valid1 != Int32(0):
                                        base1 = fc2_addr + ((block * max_slot + slot1).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                        st_global_v2_f32(base1, total0[n, 2], total0[n, 3])
                                    if valid0_g1 != Int32(0):
                                        base0_g1 = fc2_addr + ((block * max_slot + slot0_g1).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                        st_global_v2_f32(base0_g1, total1[n, 0], total1[n, 1])
                                    if valid1_g1 != Int32(0):
                                        base1_g1 = fc2_addr + ((block * max_slot + slot1_g1).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                        st_global_v2_f32(base1_g1, total1[n, 2], total1[n, 3])
                            else:
                                total = cute.make_rmem_tensor((4, 4), Float32)
                                total.fill(Float32(0.0))
                                while mask != Uint32(0):
                                    unit = _ctz_u32(mask)
                                    mask = mask & (mask - Uint32(1))
                                    desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                                    width = bits[desc]
                                    if width != Int32(0):
                                        payload = pointers[desc * Int64(3)]
                                        scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                                        a_base = inter_base + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                                        if width == Int32(2):
                                            self._fc1_accumulate_block(
                                                total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                                block, lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                                2, scale, l2_policy,
                                            )
                                        elif width == Int32(3):
                                            self._fc1_accumulate_block(
                                                total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                                block, lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                                3, scale, l2_policy,
                                            )
                                        elif width == Int32(4):
                                            self._fc1_accumulate_block(
                                                total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                                block, lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                                4, scale, l2_policy,
                                            )
                                for n in cutlass.range_constexpr(4):
                                    col_in_block = warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                                    if valid0 != Int32(0):
                                        base0 = fc2_addr + ((block * max_slot + slot0).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                        st_global_v2_f32(base0, total[n, 0], total[n, 1])
                                    if valid1 != Int32(0):
                                        base1 = fc2_addr + ((block * max_slot + slot1).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                        st_global_v2_f32(base1, total[n, 2], total[n, 3])
                    else:
                        total = cute.make_rmem_tensor((4, 4), Float32)
                        total.fill(Float32(0.0))
                        slot0 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64)]
                        slot1 = route_slots[group.to(Int64) * Int64(16) + r.to(Int64) + Int64(8)]
                        valid0 = ((slot0 >= Int32(0)) & (slot0 < max_slot)).to(Int32)
                        valid1 = ((slot1 >= Int32(0)) & (slot1 < max_slot)).to(Int32)
                        while mask != Uint32(0):
                            unit = _ctz_u32(mask)
                            mask = mask & (mask - Uint32(1))
                            desc = (expert.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(3) + Int64(2)
                            width = bits[desc]
                            if width != Int32(0):
                                payload = pointers[desc * Int64(3)]
                                scale = _side_value(pointers[desc * Int64(3) + Int64(2)], block)
                                a_base = inter_base + (group.to(Int64) * Int64(self.units) + unit.to(Int64)) * Int64(16 * 128 * 2)
                                if width == Int32(2):
                                    self._fc1_accumulate_block(
                                        total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                        block, lane, warp, prev_lane, ia2, ib2, shift2_lo, shift2_hi,
                                        2, scale, l2_policy,
                                    )
                                elif width == Int32(3):
                                    self._fc1_accumulate_block(
                                        total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                        block, lane, warp, prev_lane, ia3, ib3, shift3_lo, shift3_hi,
                                        3, scale, l2_policy,
                                    )
                                elif width == Int32(4):
                                    self._fc1_accumulate_block(
                                        total, payload, a_base, a_base + Int64(2048), valid0, valid1,
                                        block, lane, warp, prev_lane, Int32(0), Int32(0), Int64(0), Int64(0),
                                        4, scale, l2_policy,
                                    )
                        if cutlass.const_expr(self.compact_fc2):
                            for n in cutlass.range_constexpr(4):
                                col_in_block = warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                                if valid0 != Int32(0):
                                    base0 = fc2_addr + ((block * max_slot + slot0).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                    st_global_v2_f32(base0, total[n, 0], total[n, 1])
                                if valid1 != Int32(0):
                                    base1 = fc2_addr + ((block * max_slot + slot1).to(Int64) * Int64(128) + col_in_block.to(Int64)) * Int64(4)
                                    st_global_v2_f32(base1, total[n, 2], total[n, 3])
                        else:
                            for n in cutlass.range_constexpr(4):
                                col = block * Int32(128) + warp * Int32(32) + Int32(n * 8) + (lane & Int32(3)) * Int32(2)
                                base = fc2_addr + ((group.to(Int64) * Int64(16) + r.to(Int64)) * Int64(self.hidden_size) + col.to(Int64)) * Int64(4)
                                if valid0 != Int32(0):
                                    st_global_v2_f32(base, total[n, 0], total[n, 1])
                                if valid1 != Int32(0):
                                    st_global_v2_f32(base + Int64(8 * self.hidden_size * 4), total[n, 2], total[n, 3])
                    task += step

    @cute.kernel
    def reduce_kernel(
        self,
        route_group_lane: cute.Tensor,
        route_weights: cute.Tensor,
        fc2: cute.Tensor,
        signs_out: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        groups: cute.Tensor,
    ):
        live_groups = groups[0]
        if live_groups < Int32(0) or live_groups > Int32(self.max_groups):
            live_groups = Int32(0)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane, warp = Int32(tid) & Int32(31), Int32(tid) >> Int32(5)
        task = Int32(cta) * Int32(4) + warp
        step = Int32(grid) * Int32(4)
        fc2_address = fc2.iterator.toint()
        max_packed = live_groups * Int32(self.group_rows)
        max_slot = rows * Int32(self.top_k)
        while task < rows * Int32(self.blocks):
            row, block = task // Int32(self.blocks), task % Int32(self.blocks)
            col = block * Int32(128) + lane * Int32(4)
            values = cute.make_rmem_tensor(4, Float32)
            values.fill(Float32(0.0))
            if cutlass.const_expr(self.compact_fc2):
                row_fc2_base = fc2_address + (
                    (block * max_slot + row * Int32(self.top_k)).to(Int64) * Int64(128) + lane.to(Int64) * Int64(4)
                ) * Int64(4)
                for slot in cutlass.range_constexpr(self.top_k):
                    route = row.to(Int64) * Int64(self.top_k) + Int64(slot)
                    p = route_group_lane[route]
                    if p >= Int32(0) and p < max_packed:
                        weight = route_weights[route]
                        w0, w1, w2, w3 = ld_global_nc_v4_u32(row_fc2_base + Int64(slot * 512))
                        values[0] = values[0] + weight * u32_as_f32(w0)
                        values[1] = values[1] + weight * u32_as_f32(w1)
                        values[2] = values[2] + weight * u32_as_f32(w2)
                        values[3] = values[3] + weight * u32_as_f32(w3)
            else:
                packed_slots = cute.make_rmem_tensor(self.top_k, Int32)
                weights_buf = cute.make_rmem_tensor(self.top_k, Float32)
                for slot in cutlass.range_constexpr(self.top_k):
                    route = row.to(Int64) * Int64(self.top_k) + Int64(slot)
                    p = route_group_lane[route]
                    packed_slots[slot] = p
                    if p >= Int32(0) and p < max_packed:
                        weights_buf[slot] = route_weights[route]
                    else:
                        weights_buf[slot] = Float32(0.0)
                for slot in cutlass.range_constexpr(self.top_k):
                    p = packed_slots[slot]
                    if p >= Int32(0) and p < max_packed:
                        weight = weights_buf[slot]
                        w0, w1, w2, w3 = ld_global_nc_v4_u32(
                            fc2_address + (p.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(4)
                        )
                        values[0] = values[0] + weight * u32_as_f32(w0)
                        values[1] = values[1] + weight * u32_as_f32(w1)
                        values[2] = values[2] + weight * u32_as_f32(w2)
                        values[3] = values[3] + weight * u32_as_f32(w3)
            h0, h1, h2, h3 = _w4a8_had128_quad(values[0], values[1], values[2], values[3], lane)
            signs = ld_global_nc_u32(signs_out.iterator.toint() + col.to(Int64))
            store_bf16x4(
                output.iterator.toint() + (row.to(Int64) * Int64(self.hidden_size) + col.to(Int64)) * Int64(2),
                h0 * signed_byte(signs, 0),
                h1 * signed_byte(signs, 1),
                h2 * signed_byte(signs, 2),
                h3 * signed_byte(signs, 3),
            )
            task += step

