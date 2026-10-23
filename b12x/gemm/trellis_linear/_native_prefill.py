"""Native many-row trellis dense prefill kernel (M128xN64 decode-once to FP16 SMEM, 2 CTAs/SM).

Reads packed trellis words and per-K128 scales directly, decodes each 16x16
weight tile once per CTA into FP16 shared memory, and reuses the decoded
fragments across all 128 rows (four 32-row consumer warp groups) with
ldmatrix-fed HMMA, software-pipelined weight/scale loads, and full-K
in-register split folding.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    f16_mma_m16n8k16_f32,
    packed_dequant_trellis_mul1_to_half2x4,
    packed_dequant_trellis_stream_to_half2x4,
    packed_dequant_trellis_to_half2x4,
    shared_ptr_to_u32,
    trellis_align_stream_u32x2,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a16.exlr8 import _e4m3_round_half2, _side_value


@dsl_user_op
def _cp_async4_cg_zfill(smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; .reg .b32 s; setp.ne.b32 p, $2, 0; selp.b32 s, 16, 0, p; "
        "cp.async.cg.shared.global [$0], [$1], 16, s; }",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_nc_u32_offset(base_addr: Int64, offset: int, *, loc=None, ip=None) -> Uint32:
    disp = f" + {int(offset)}" if offset else ""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_addr).ir_value(loc=loc, ip=ip)],
            f"ld.global.nc.u32 $0, [$1{disp}];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_b32_offset(base_addr: Int32, offset: int, val: Uint32, *, loc=None, ip=None):
    disp = f" + {int(offset)}" if offset else ""
    llvm.inline_asm(
        None,
        [Int32(base_addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        f"st.shared.b32 [$0{disp}], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32_offset(base_addr: Int32, offset: int, *, loc=None, ip=None) -> Uint32:
    disp = f" + {int(offset)}" if offset else ""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(base_addr).ir_value(loc=loc, ip=ip)],
            f"ld.shared.b32 $0, [$1{disp}];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldmatrix_m8n8x4_b16_offset(base_addr: Int32, offset: int, *, loc=None, ip=None):
    disp = f" + {int(offset)}" if offset else ""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(base_addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0, $1, $2, $3}}, [$4{disp}];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
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


@dsl_user_op
def _st_global_v2_f32(addr: Int64, v0: Float32, v1: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _add_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def prefill_storage():
    class Storage:
        pass

    # 32 KiB for 2x (128x64 FP16) A stages + 8 KiB for (64x64 FP16) decoded B = 40 KiB
    Storage.__annotations__ = {
        "words": cute.struct.Align[cute.struct.MemRange[Uint32, 10240], 16],
    }
    return cute.struct(Storage)


@cute.jit
def stage_a_half(
    sa_store_ptr: Int32,
    x_lane_addr: Int64,
    group_r_a: Int32,
    rows: Int32,
    k_col_base: Int32,
    k_features: cutlass.Constexpr,
):
    base_gmem = x_lane_addr + (group_r_a.to(Int64) * Int64(k_features) + k_col_base.to(Int64)) * Int64(2)
    row_stride = 64 * k_features
    for s in cutlass.range_constexpr(4):
        row = group_r_a + Int32(s * 32)
        valid = (row < rows).to(Int32)
        _cp_async4_cg_zfill(
            sa_store_ptr + Int32(s * 4096),
            base_gmem + Int64(s * row_stride),
            valid,
        )


@cute.jit
def prefetch_b_half(
    payload_i0: Int64,
    nb64: Int32,
    kb: Int32,
    half: cutlass.Constexpr,
    warp: Int32,
    d1_bytes: Int32,
    d2_bytes: Int32,
    k_features: cutlass.Constexpr,
    n_features: cutlass.Constexpr,
    input_units: cutlass.Constexpr,
    bits: cutlass.Constexpr,
):
    nb128 = nb64 >> Int32(1)
    n64_half = nb64 & Int32(1)
    c4 = warp & Int32(3)
    ki_hi = (warp >> Int32(2)) * Int32(2)
    k16_base = kb * Int32(8) + Int32(half * 4) + ki_hi
    c8 = n64_half * Int32(4) + c4
    tile_base = nb128 * Int32(k_features // 16 * 8) + k16_base * Int32(8) + c8
    if cutlass.const_expr(input_units):
        tile_base = (
            kb * Int32(n_features // 16 * 8)
            + nb128 * Int32(64)
            + (Int32(half * 4) + ki_hi) * Int32(8)
            + c8
        )
    p0 = payload_i0 + tile_base.to(Int64) * Int64(32 * bits)
    p1 = p0 + d1_bytes.to(Int64)
    p2 = p0 + d2_bytes.to(Int64)
    step_bytes = 256 * bits
    z0_0 = _ld_global_nc_u32_offset(p0, 0)
    z1_0 = _ld_global_nc_u32_offset(p1, 0)
    z2_0 = _ld_global_nc_u32_offset(p2, 0)
    z0_1 = _ld_global_nc_u32_offset(p0, step_bytes)
    z1_1 = _ld_global_nc_u32_offset(p1, step_bytes)
    z2_1 = _ld_global_nc_u32_offset(p2, step_bytes)
    return z0_0, z1_0, z2_0, z0_1, z1_1, z2_1


@cute.jit
def store_decoded_b_half(
    sb_write_base: Int32,
    payload_addr: Int64,
    nb64: Int32,
    kb: Int32,
    half: cutlass.Constexpr,
    warp: Int32,
    lane: Int32,
    z0_0: Uint32,
    z1_0: Uint32,
    z2_0: Uint32,
    z0_1: Uint32,
    z1_1: Uint32,
    z2_1: Uint32,
    shift: Int32,
    delta: Int32,
    k_features: cutlass.Constexpr,
    n_features: cutlass.Constexpr,
    input_units: cutlass.Constexpr,
    bits: cutlass.Constexpr,
    e4m3: cutlass.Constexpr,
    mul1: cutlass.Constexpr,
):
    if cutlass.const_expr(bits >= 6):
        for pair_idx in cutlass.range_constexpr(2):
            z0 = z0_0 if pair_idx == 0 else z0_1
            z1 = z1_0 if pair_idx == 0 else z1_1
            z2 = z2_0 if pair_idx == 0 else z2_1
            wa, wb = trellis_align_stream_u32x2(z0, z1, z2, shift, delta)
            if cutlass.const_expr(mul1):
                b0, b1, b2, b3 = packed_dequant_trellis_mul1_to_half2x4(wa, wb, bits)
            elif cutlass.const_expr(bits == 6):
                b0, b1, b2, b3 = packed_dequant_trellis_stream_to_half2x4(wa, wb, bits)
            else:
                b0, b1, b2, b3 = packed_dequant_trellis_to_half2x4(wa, wb, bits)
            if cutlass.const_expr(e4m3):
                b0, b1 = _e4m3_round_half2(b0), _e4m3_round_half2(b1)
                b2, b3 = _e4m3_round_half2(b2), _e4m3_round_half2(b3)
            off = pair_idx * 2048
            _st_shared_b32_offset(sb_write_base, off + 0, b0)
            _st_shared_b32_offset(sb_write_base, off + 128, b1)
            _st_shared_b32_offset(sb_write_base, off + 256, b2)
            _st_shared_b32_offset(sb_write_base, off + 384, b3)
    else:
        from b12x.gemm.trellis_linear._native import decode_tile
        nb128 = nb64 >> Int32(1)
        n64_half = nb64 & Int32(1)
        c4 = warp & Int32(3)
        ki_hi = (warp >> Int32(2)) * Int32(2)
        k16_base = kb * Int32(8) + Int32(half * 4) + ki_hi
        c8 = n64_half * Int32(4) + c4
        tile_base = nb128 * Int32(k_features // 16 * 8) + k16_base * Int32(8) + c8
        if cutlass.const_expr(input_units):
            tile_base = (
                kb * Int32(n_features // 16 * 8)
                + nb128 * Int32(64)
                + (Int32(half * 4) + ki_hi) * Int32(8)
                + c8
            )
        for pair_idx in cutlass.range_constexpr(2):
            b0, b1, b2, b3 = decode_tile(
                payload_addr, tile_base + Int32(pair_idx * 8), lane, bits, e4m3, mul1
            )
            off = pair_idx * 2048
            _st_shared_b32_offset(sb_write_base, off + 0, b0)
            _st_shared_b32_offset(sb_write_base, off + 128, b1)
            _st_shared_b32_offset(sb_write_base, off + 256, b2)
            _st_shared_b32_offset(sb_write_base, off + 384, b3)
