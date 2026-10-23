"""Native many-row trellis dense prefill kernel (M128xN128 decode-once to FP16 SMEM).

Reads packed trellis words and per-K128 scales directly, decodes each 16x16
weight tile once per CTA into FP16 shared memory, and reuses the decoded
fragments across all 128 rows (four 32-row consumer warp groups) with
ldmatrix-fed HMMA and full-K in-register split folding.
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

    # 32 KiB for 2x (128x64 FP16) A stages + 16 KiB for (64x128 FP16) decoded B = 48 KiB
    Storage.__annotations__ = {
        "words": cute.struct.Align[cute.struct.MemRange[Uint32, 12288], 16],
    }
    return cute.struct(Storage)


@cute.jit
def stage_a_half(
    sa_stage: Int32,
    x_addr: Int64,
    group: Int32,
    rows: Int32,
    k_col_base: Int32,
    r_a: Int32,
    c_a: Int32,
    a_store_base: Int32,
    k_features: cutlass.Constexpr,
):
    k_col = k_col_base + c_a * Int32(8)
    row_base = group * Int32(128) + r_a
    for s in cutlass.range_constexpr(4):
        row = row_base + Int32(s * 32)
        valid = (row < rows).to(Int32)
        safe_row = min(row, rows - Int32(1))
        gmem = x_addr + (safe_row.to(Int64) * Int64(k_features) + k_col.to(Int64)) * Int64(2)
        _cp_async4_cg_zfill(sa_stage + a_store_base + Int32(s * 4096), gmem, valid)


@cute.jit
def decode_and_store_b_half(
    sb_write_base: Int32,
    payload_addr: Int64,
    nb: Int32,
    kb: Int32,
    half: Int32,
    warp: Int32,
    lane: Int32,
    i0_bytes: Int64,
    i1_bytes: Int64,
    i2_bytes: Int64,
    shift: Int32,
    delta: Int32,
    k_features: cutlass.Constexpr,
    n_features: cutlass.Constexpr,
    input_units: cutlass.Constexpr,
    bits: cutlass.Constexpr,
    e4m3: cutlass.Constexpr,
    mul1: cutlass.Constexpr,
):
    k16_base = kb * Int32(8) + half * Int32(4)
    tile_base = nb * Int32(k_features // 16 * 8) + k16_base * Int32(8) + warp
    if cutlass.const_expr(input_units):
        tile_base = kb * Int32(n_features // 16 * 8) + nb * Int32(64) + (half * Int32(32)) + warp
    base_addr = payload_addr + tile_base.to(Int64) * Int64(32 * bits)
    step_bytes = 256 * bits

    if cutlass.const_expr(bits >= 6):
        p0 = base_addr + i0_bytes
        p1 = base_addr + i1_bytes
        p2 = base_addr + i2_bytes
        z0_0 = _ld_global_nc_u32_offset(p0, 0 * step_bytes)
        z1_0 = _ld_global_nc_u32_offset(p1, 0 * step_bytes)
        z2_0 = _ld_global_nc_u32_offset(p2, 0 * step_bytes)
        z0_1 = _ld_global_nc_u32_offset(p0, 1 * step_bytes)
        z1_1 = _ld_global_nc_u32_offset(p1, 1 * step_bytes)
        z2_1 = _ld_global_nc_u32_offset(p2, 1 * step_bytes)
        z0_2 = _ld_global_nc_u32_offset(p0, 2 * step_bytes)
        z1_2 = _ld_global_nc_u32_offset(p1, 2 * step_bytes)
        z2_2 = _ld_global_nc_u32_offset(p2, 2 * step_bytes)
        z0_3 = _ld_global_nc_u32_offset(p0, 3 * step_bytes)
        z1_3 = _ld_global_nc_u32_offset(p1, 3 * step_bytes)
        z2_3 = _ld_global_nc_u32_offset(p2, 3 * step_bytes)

        for ki in cutlass.range_constexpr(4):
            z0 = z0_0 if ki == 0 else (z0_1 if ki == 1 else (z0_2 if ki == 2 else z0_3))
            z1 = z1_0 if ki == 0 else (z1_1 if ki == 1 else (z1_2 if ki == 2 else z1_3))
            z2 = z2_0 if ki == 0 else (z2_1 if ki == 1 else (z2_2 if ki == 2 else z2_3))
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
            off = ki * 4096
            _st_shared_b32_offset(sb_write_base, off + 0, b0)
            _st_shared_b32_offset(sb_write_base, off + 128, b1)
            _st_shared_b32_offset(sb_write_base, off + 256, b2)
            _st_shared_b32_offset(sb_write_base, off + 384, b3)
    else:
        from b12x.gemm.trellis_linear._native import decode_tile
        for ki in cutlass.range_constexpr(4):
            b0, b1, b2, b3 = decode_tile(
                payload_addr, tile_base + Int32(ki * 8), lane, bits, e4m3, mul1
            )
            off = ki * 4096
            _st_shared_b32_offset(sb_write_base, off + 0, b0)
            _st_shared_b32_offset(sb_write_base, off + 128, b1)
            _st_shared_b32_offset(sb_write_base, off + 256, b2)
            _st_shared_b32_offset(sb_write_base, off + 384, b3)
