"""Direct DS41RT four-source address resolution for a future native producer.

The descriptor is the native 120-byte ``ds41rt_v41_sparse_kv_t`` viewed as
15 uint64 words. Callers validate pointer spans/capacities before launch, retain
all storage, and order metadata updates on the consuming lane's stream. This
helper does not load or repack KV payloads and is not yet wired into attention.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ld_shared_u8_offset,
    ld_shared_u32,
    ld_shared_v2_u32,
    mma_m16n8k16_f32_bf16,
    shared_ptr_to_u32,
    st_shared_u32,
    st_shared_v4_u32,
)
from .io import stage_dsv41_fp8_scales

# Disjoint per-candidate tile layout inside the existing 69,632-byte kv_fp8
# allocation (64 rows * 1,088 bytes = 2 * 64 * 544 bytes). Each row stages its
# raw mixed FP8/FP4 record in [544, 1088) and decodes once to swizzled BF16 in
# [0, 1024), reused across both QK (ldmatrix.x2) and PV (ldmatrix.trans.x2).
DSV41_BF16_TILE_ROW_STRIDE = 1088
DSV41_BF16_RAW_ROW_OFFSET = 544


@cute.jit
def _load_u64(address: Int64):
    lo, hi = ld_global_nc_v2_u32(address)
    return Uint64(lo) | (Uint64(hi) << Uint64(32))


@cute.jit
def native_v41_metadata_valid(descriptor: cute.Tensor, metadata: cute.Tensor, window_begin: Uint64):
    m0, offset, count, position = metadata[0], metadata[1], metadata[2], metadata[3]
    committed, private_count = metadata[6], metadata[7]
    private_offset, private_stride = metadata[8], metadata[9]
    window_capacity = descriptor[11]
    source_capacity, private_capacity = descriptor[12], descriptor[13]
    page_stride = descriptor[14] & Uint64(0xFFFFFFFF)
    compressed = descriptor[14] >> Uint64(32)
    valid = (window_begin <= m0) & (m0 <= Uint64(1048576)) & (count > Uint64(0))
    valid = valid & (m0 == _load_u64(Int64(descriptor[8])))
    valid = valid & (count <= Uint64(1048576) - m0) & (offset <= window_capacity)
    valid = valid & (count <= window_capacity - offset)
    valid = valid & (position >= m0) & (position - m0 < count)
    if valid and compressed != Uint64(0):
        source_end = _load_u64(Int64(descriptor[10]))
        boundary_ok = committed == source_end
        if private_count == Uint64(0):
            boundary_ok = committed <= source_end
        valid = (metadata[4] == Uint64(0)) & boundary_ok & (committed <= Uint64(1048576))
        valid = valid & (private_count <= Uint64(1048576) - committed)
        valid = valid & (metadata[5] <= committed + private_count)
        valid = valid & ((private_stride == Uint64(1)) | (private_stride == Uint64(2)))
        valid = valid & (private_offset <= private_capacity)
        if private_count != Uint64(0):
            # Division is reached only for a valid nonzero stride.
            if valid:
                valid = private_offset < private_capacity
                if valid:
                    valid = private_count - Uint64(1) <= (private_capacity - Uint64(1) - private_offset) // private_stride
    return valid


@cute.jit
def native_v41_record_context(
    descriptor: cute.Tensor, metadata: cute.Tensor, window_begin: Uint64,
):
    """Read the key-independent descriptor/metadata words once per token row.

    ``resolve_native_v41_record`` re-read all twelve metadata words, five
    descriptor words and the whole validation load chain for every one of the
    sixty-four keys a gather stages. Every one of those words is a property of
    the token's descriptor row, not of the key, so the producer evaluates them
    once per chunk and threads them through the per-key address resolution.
    """
    m0, offset, position = metadata[0], metadata[1], metadata[3]
    logical_count = metadata[5]
    committed, private_count = metadata[6], metadata[7]
    private_offset, private_stride = metadata[8], metadata[9]
    source_capacity = descriptor[12]
    page_stride = descriptor[14] & Uint64(0xFFFFFFFF)
    compressed = descriptor[14] >> Uint64(32)
    valid = native_v41_metadata_valid(descriptor, metadata, window_begin)
    return (valid, m0, offset, position, logical_count, committed, private_count,
            private_offset, private_stride, source_capacity, page_stride,
            compressed)


@cute.jit
def resolve_native_v41_record(
    descriptor: cute.Tensor, metadata: cute.Tensor, selected: cute.Tensor,
    key: Int32, window_begin: Uint64, context=None,
):
    """Return (value address, scale address); (0, 0) means a masked key.

    Keys 0..127 are the causal window, 128..639 are selected source IDs.
    This fixed boundary pads short windows, independently of the native
    implementation's optional compact physical tile schedule. ``context`` is
    the tuple from :func:`native_v41_record_context` for this token row; when
    omitted the helper resolves it here (per-key callers only). ``context is
    None`` is a trace-time predicate: a plain ``if`` would be staged control
    flow and could not join ``None`` with the twelve-scalar tuple.
    """
    if cutlass.const_expr(context is None):
        context = native_v41_record_context(descriptor, metadata, window_begin)
    (valid, m0, offset, position, logical_count, committed, private_count,
     private_offset, private_stride, source_capacity, page_stride,
     compressed) = context
    value_address, scale_address = Uint64(0), Uint64(0)
    source, row = Int32(0), Uint64(0)
    if valid:
        if key >= Int32(0) and key < Int32(128):
            first = Uint64(0)
            if position >= Uint64(128):
                first = position + Uint64(1) - Uint64(128)
            logical = first + Uint64(key)
            valid = (logical >= window_begin) & (logical <= position)
            if logical < m0:
                row = logical % Uint64(128)
            else:
                source = Int32(1)
                row = offset + logical - m0
        elif key >= Int32(128) and key < Int32(640) and compressed != Uint64(0):
            logical_id = Int32(selected[key - Int32(128)])
            valid = (logical_id >= Int32(0)) & (Uint64(logical_id) < logical_count)
            if valid:
                logical = Uint64(logical_id)
                if logical >= committed:
                    source = Int32(3)
                    valid = logical - committed < private_count
                    row = private_offset + (logical - committed) * private_stride
                else:
                    source = Int32(2)
                    valid = logical < page_stride * Uint64(256)
                    if valid:
                        page = ld_global_nc_u32(Int64(descriptor[9] + (logical // Uint64(256)) * Uint64(4)))
                        row = Uint64(page) * Uint64(256) + logical % Uint64(256)
                        valid = row < source_capacity
        else:
            valid = False
        if valid:
            value_bytes, scale_bytes = Uint64(512), Uint64(16)
            if source >= Int32(2) and compressed == Uint64(2):
                value_bytes, scale_bytes = Uint64(256), Uint64(32)
            value_address = descriptor[source] + row * value_bytes
            scale_address = descriptor[source + Int32(4)] + row * scale_bytes
    return value_address, scale_address


@cute.jit
def issue_native_v41_gather(
    descriptor: cute.Tensor, metadata: cute.Tensor, selected: cute.Tensor,
    window_begin: Uint64, kv_addr: Int32, scale_addr: Int32,
    token_indices: cute.Tensor, full_mbar: cute.Pointer,
    start: Int32, io_lane: Int32, *, swa: cutlass.Constexpr,
    kv_stride: cutlass.Constexpr, io_threads: cutlass.Constexpr,
    fp8_internal: cutlass.Constexpr = True,
):
    """Stage native FP8-window/FP4-source records for BF16 or canonical FP8 math.

    Payload planes must be 16-byte aligned. Uses the existing 64-key producer
    stage and one arrival per IO warp. Native descriptor format must be FP4.
    Invalid keys read only the allocated ring's fallback row; their tag and
    canonical metadata mask that payload. No full-pool or selected-pool repack.
    """
    payload_bytes = 512 if swa else 256
    # One token row per gather: resolve its key-independent metadata once and
    # share it across all sixty-four key resolutions instead of reloading the
    # descriptor words and re-running the validation chain per key.
    context = native_v41_record_context(descriptor, metadata, window_begin)
    for part in cutlass.range_constexpr((64 + io_threads - 1) // io_threads):
        entry = io_lane + Int32(part * io_threads)
        if entry < Int32(64):
            key = start + entry + Int32(0 if swa else 128)
            value, scale = resolve_native_v41_record(
                descriptor, metadata, selected, key, window_begin, context
            )
            valid = value != Uint64(0)
            index, tag = key, Uint32(1 if swa else 0)
            if not valid:
                index, tag = Int32(-1), Uint32(2)
                value = Uint64(descriptor[0])
            token_indices[entry] = index
            row_addr = kv_addr + entry * Int32(kv_stride)
            st_shared_u32(row_addr + Int32(528), tag)
            if cutlass.const_expr(fp8_internal):
                stage_dsv41_fp8_scales(
                    descriptor, Int64(0), row_addr + Int32(544),
                    scale_addr + entry * Int32(8), valid,
                    swa=swa, scale_address=Int64(scale),
                )
                cp_async_bulk_g2s_mbar(row_addr, Int64(value), Int32(payload_bytes), shared_ptr_to_u32(full_mbar))
            else:
                # Overlap 128-bit scale vector loads with the bulk payload copy.
                s0, s1, s2, s3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid:
                    s0, s1, s2, s3 = ld_global_nc_v4_u32(Int64(scale))
                if cutlass.const_expr(not swa):
                    s4, s5, s6, s7 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if valid:
                        s4, s5, s6, s7 = ld_global_nc_v4_u32(Int64(scale) + Int64(16))
                cp_async_bulk_g2s_mbar(row_addr, Int64(value), Int32(payload_bytes), shared_ptr_to_u32(full_mbar))
                st_shared_v4_u32(row_addr + Int32(payload_bytes), s0, s1, s2, s3)
                if cutlass.const_expr(not swa):
                    st_shared_v4_u32(row_addr + Int32(payload_bytes + 16), s4, s5, s6, s7)
    cute.arch.fence_acq_rel_cta()
    if (io_lane & Int32(31)) == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar, Int32(64 * payload_bytes // (io_threads // 32)))


@dsl_user_op
def _ldmatrix_m8n8x4_b16_offset(smem_addr: Int32, offset: int, *, loc=None, ip=None):
    disp = f" + {int(offset)}" if int(offset) else ""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
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
def _ldmatrix_m8n8x2_b16_offset(smem_addr: Int32, offset: int, *, loc=None, ip=None):
    disp = f" + {int(offset)}" if int(offset) else ""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{$0, $1}}, [$2{disp}];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ldmatrix_m8n8x2_trans_b16_offset(smem_addr: Int32, offset: int, *, loc=None, ip=None):
    disp = f" + {int(offset)}" if int(offset) else ""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.trans.m8n8.x2.shared.b16 {{$0, $1}}, [$2{disp}];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _mxfp8x8_to_bf16x2x4_sm120(
    w0: Uint32, w1: Uint32, scale: Uint32, *, loc=None, ip=None
):
    """Decode 8 E4M3 bytes sharing one UE8M0 scale byte to four bf16x2 words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 q0, q1, q2, q3, sf;
            .reg .b32 v0, v1, v2, v3, factors, sf_pair, o0, o1, o2, o3;
            mov.b32 {q0, q1}, $4;
            mov.b32 {q2, q3}, $5;
            shl.b32 sf_pair, $6, 8;
            or.b32 sf_pair, $6, sf_pair;
            cvt.u16.u32 sf, sf_pair;
            cvt.rn.bf16x2.ue8m0x2 factors, sf;
            cvt.rn.bf16x2.e4m3x2 v0, q0;
            cvt.rn.bf16x2.e4m3x2 v1, q1;
            cvt.rn.bf16x2.e4m3x2 v2, q2;
            cvt.rn.bf16x2.e4m3x2 v3, q3;
            mul.bf16x2 o0, v0, factors;
            mul.bf16x2 o1, v1, factors;
            mul.bf16x2 o2, v2, factors;
            mul.bf16x2 o3, v3, factors;
            mov.b32 $0, o0;
            mov.b32 $1, o1;
            mov.b32 $2, o2;
            mov.b32 $3, o3;
        }
        """,
        "=r,=r,=r,=r,r,r,r",
        has_side_effects=False,
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
def _nvfp4x8_to_bf16x2x4_sm120(
    packed: Uint32, scale: Uint32, *, loc=None, ip=None
):
    """Decode 8 E2M1 nibbles sharing one E4M3 scale byte to four bf16x2 words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Uint32(scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 sf;
            .reg .b32 v0, v1, v2, v3, factors, sf_pair, o0, o1, o2, o3;
            mov.b32 {b0, b1, b2, b3}, $4;
            shl.b32 sf_pair, $5, 8;
            or.b32 sf_pair, $5, sf_pair;
            cvt.u16.u32 sf, sf_pair;
            cvt.rn.bf16x2.e4m3x2 factors, sf;
            cvt.rn.bf16x2.e2m1x2 v0, b0;
            cvt.rn.bf16x2.e2m1x2 v1, b1;
            cvt.rn.bf16x2.e2m1x2 v2, b2;
            cvt.rn.bf16x2.e2m1x2 v3, b3;
            mul.bf16x2 o0, v0, factors;
            mul.bf16x2 o1, v1, factors;
            mul.bf16x2 o2, v2, factors;
            mul.bf16x2 o3, v3, factors;
            mov.b32 $0, o0;
            mov.b32 $1, o1;
            mov.b32 $2, o2;
            mov.b32 $3, o3;
        }
        """,
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
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
def _st_shared_bf16x2_from_f32(
    addr: Int32, x0: Float32, x1: Float32, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(x0).ir_value(loc=loc, ip=ip),
            Float32(x1).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 tmp; cvt.rn.bf16x2.f32 tmp, $2, $1; st.shared.b32 [$0], tmp; }",
        "r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def s0_load_q_bf16_vec(
    q_token: cute.Tensor,
    q_nope_bf16_base_addr: Int32,
    head_base: Int32,
    tid: Int32,
    *,
    d_nope: cutlass.Constexpr,
    hpb: cutlass.Constexpr,
    valid_hpb_static: cutlass.Constexpr,
    q_nope_bf16_stride: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
):
    """Stage 16-byte-aligned BF16 Q rows into shared memory with 128-bit copies."""
    threads_per_head = num_threads // hpb
    elems_per_vec = 8
    iters = d_nope // (threads_per_head * elems_per_vec)
    h = tid // Int32(threads_per_head)
    lane_h = tid - h * Int32(threads_per_head)

    q_words = []
    for it in cutlass.range_constexpr(iters):
        d = Int32(it * threads_per_head * elems_per_vec) + lane_h * Int32(elems_per_vec)
        if cutlass.const_expr(valid_hpb_static >= hpb):
            q_off = cute.crd2idx((head_base + h, d), q_token.layout)
            q_words.append(ld_global_nc_v4_u32(get_ptr_as_int64(q_token, q_off)))
        else:
            w0, w1, w2, w3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if h < Int32(valid_hpb_static):
                q_off = cute.crd2idx((head_base + h, d), q_token.layout)
                w0, w1, w2, w3 = ld_global_nc_v4_u32(get_ptr_as_int64(q_token, q_off))
            q_words.append((w0, w1, w2, w3))

    row_addr = q_nope_bf16_base_addr + h * Int32(q_nope_bf16_stride * 2) + lane_h * Int32(16)
    for it in cutlass.range_constexpr(iters):
        w0, w1, w2, w3 = q_words[it]
        st_shared_v4_u32(
            row_addr + Int32(it * threads_per_head * 16), w0, w1, w2, w3
        )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)


@cute.jit
def decode_dsv41_kv_to_bf16_smem(
    kv_base_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    *,
    swa: cutlass.Constexpr,
    bi: cutlass.Constexpr = 64,
    row_stride: cutlass.Constexpr = DSV41_BF16_TILE_ROW_STRIDE,
    raw_offset: cutlass.Constexpr = DSV41_BF16_RAW_ROW_OFFSET,
    math_warps: cutlass.Constexpr = 8,
):
    """Decode staged SWA MXFP8 or indexed NVFP4 records once per tile to swizzled BF16.

    Each 1,088-byte row holds raw staging in [544, 1088) and writes 512 BF16
    values to [0, 1024) with 16-byte chunk XOR swizzle ``c16 ^ (entry & 7)``.
    With stride 1,088B (68 x 16B == 4 mod 8 chunks), ``((entry & 1) << 2) ^ (entry & 7)``
    permutes all eight 16-byte bank groups without conflict across decode STS.128,
    QK ldmatrix.m8n8.x2, and PV ldmatrix.trans.m8n8.x2.
    """
    expected_tag = Uint32(1 if swa else 0)
    for row_iter in cutlass.range(bi // math_warps, unroll=1):
        entry = warp_id + Int32(row_iter) * Int32(math_warps)
        row_base = kv_base_addr + entry * Int32(row_stride)
        raw_row = row_base + Int32(raw_offset)
        swz = entry & Int32(7)
        dst0 = row_base + (lane ^ swz) * Int32(16)
        dst1 = row_base + ((Int32(32) + lane) ^ swz) * Int32(16)
        tag = ld_shared_u32(raw_row + Int32(528))
        if tag == expected_tag:
            if cutlass.const_expr(swa):
                w00, w01 = ld_shared_v2_u32(raw_row + lane * Int32(8))
                w10, w11 = ld_shared_v2_u32(raw_row + Int32(256) + lane * Int32(8))
                s0 = ld_shared_u8_offset(raw_row + (lane >> Int32(2)), 512)
                s1 = ld_shared_u8_offset(raw_row + (lane >> Int32(2)), 520)
                cute.arch.sync_warp()
                d00, d01, d02, d03 = _mxfp8x8_to_bf16x2x4_sm120(w00, w01, s0)
                st_shared_v4_u32(dst0, d00, d01, d02, d03)
                d10, d11, d12, d13 = _mxfp8x8_to_bf16x2x4_sm120(w10, w11, s1)
                st_shared_v4_u32(dst1, d10, d11, d12, d13)
            else:
                w0 = ld_shared_u32(raw_row + lane * Int32(4))
                w1 = ld_shared_u32(raw_row + Int32(128) + lane * Int32(4))
                s0 = ld_shared_u8_offset(raw_row + (lane >> Int32(1)), 256)
                s1 = ld_shared_u8_offset(raw_row + (lane >> Int32(1)), 272)
                cute.arch.sync_warp()
                d00, d01, d02, d03 = _nvfp4x8_to_bf16x2x4_sm120(w0, s0)
                st_shared_v4_u32(dst0, d00, d01, d02, d03)
                d10, d11, d12, d13 = _nvfp4x8_to_bf16x2x4_sm120(w1, s1)
                st_shared_v4_u32(dst1, d10, d11, d12, d13)
        else:
            z = Uint32(0)
            st_shared_v4_u32(dst0, z, z, z, z)
            st_shared_v4_u32(dst1, z, z, z, z)


@cute.jit
def s1_qk_dsv41_bf16_smem(
    qk,
    q_nope_bf16_base_addr: Int32,
    kv_bf16_base_addr: Int32,
    warp_first_cand: Int32,
    lane: Int32,
    *,
    d_nope: cutlass.Constexpr = 512,
    q_nope_bf16_stride: cutlass.Constexpr = 520,
    kv_bf16_stride: cutlass.Constexpr = DSV41_BF16_TILE_ROW_STRIDE,
):
    """BF16 QK-NoPE reading decoded swizzled K via ldmatrix.m8n8.x2."""
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    a_base = q_nope_bf16_base_addr + a_row * Int32(q_nope_bf16_stride * 2) + a_col * Int32(2)

    lane_r = lane & Int32(7)
    lane_sub = (lane >> Int32(3)) & Int32(1)
    b_row_base = kv_bf16_base_addr + (warp_first_cand + lane_r) * Int32(kv_bf16_stride)
    swz_base = lane_sub ^ lane_r
    b_ptr0 = b_row_base + swz_base * Int32(16)
    b_ptr1 = b_row_base + (swz_base ^ Int32(2)) * Int32(16)
    b_ptr2 = b_row_base + (swz_base ^ Int32(4)) * Int32(16)
    b_ptr3 = b_row_base + (swz_base ^ Int32(6)) * Int32(16)
    b_ptrs = (b_ptr0, b_ptr1, b_ptr2, b_ptr3)

    for seg in cutlass.range_constexpr(d_nope // 64):
        for sub in cutlass.range_constexpr(4):
            ks = seg * 4 + sub
            a0, a1, a2, a3 = _ldmatrix_m8n8x4_b16_offset(a_base, ks * 32)
            b0, b1 = _ldmatrix_m8n8x2_b16_offset(b_ptrs[sub], seg * 128)
            qk[0], qk[1], qk[2], qk[3] = mma_m16n8k16_f32_bf16(
                qk[0], qk[1], qk[2], qk[3], a0, a1, a2, a3, b0, b1
            )
    return qk


@cute.jit
def s5_fill_sm_p_bf16(
    w_pre,
    sm_p_full_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    *,
    sm_p_stride: cutlass.Constexpr,
):
    """Store packed bf16x2 softmax weights to sm_p_full without FP8 scratch init."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    c0 = warp_id * Int32(8) + tid * Int32(2)
    _st_shared_bf16x2_from_f32(
        sm_p_full_addr + (gid * Int32(sm_p_stride) + c0) * Int32(2),
        w_pre[0],
        w_pre[1],
    )
    _st_shared_bf16x2_from_f32(
        sm_p_full_addr + ((gid + Int32(8)) * Int32(sm_p_stride) + c0) * Int32(2),
        w_pre[2],
        w_pre[3],
    )


@cute.jit
def s6_pv_dsv41_bf16_smem(
    acc_nope,
    sm_p_full_addr: Int32,
    kv_bf16_base_addr: Int32,
    warp_id: Int32,
    lane: Int32,
    *,
    n_v_chunks: cutlass.Constexpr = 8,
    bi: cutlass.Constexpr = 64,
    sm_p_stride: cutlass.Constexpr = 72,
    kv_bf16_stride: cutlass.Constexpr = DSV41_BF16_TILE_ROW_STRIDE,
):
    """BF16 P.V reusing decoded swizzled V via ldmatrix.trans.m8n8.x2 and hoisted P loads."""
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    p_base = sm_p_full_addr + (a_row * Int32(sm_p_stride) + a_col) * Int32(2)

    lane_r = lane & Int32(7)
    lane_sub = (lane >> Int32(3)) & Int32(1)
    v_base = (
        kv_bf16_base_addr
        + (lane_sub * Int32(8) + lane_r) * Int32(kv_bf16_stride)
        + (warp_id ^ lane_r) * Int32(16)
    )

    p_frags = [
        _ldmatrix_m8n8x4_b16_offset(p_base, ks * 32)
        for ks in range(bi // 16)
    ]
    for vc in cutlass.range_constexpr(n_v_chunks):
        x0 = Float32(0.0)
        x1 = Float32(0.0)
        x2 = Float32(0.0)
        x3 = Float32(0.0)
        for ks in cutlass.range_constexpr(bi // 16):
            a0, a1, a2, a3 = p_frags[ks]
            b0, b1 = _ldmatrix_m8n8x2_trans_b16_offset(
                v_base, ks * (16 * kv_bf16_stride) + vc * 128
            )
            x0, x1, x2, x3 = mma_m16n8k16_f32_bf16(
                x0, x1, x2, x3, a0, a1, a2, a3, b0, b1
            )
        acc_nope[vc][0] = acc_nope[vc][0] + x0
        acc_nope[vc][1] = acc_nope[vc][1] + x1
        acc_nope[vc][2] = acc_nope[vc][2] + x2
        acc_nope[vc][3] = acc_nope[vc][3] + x3
    return acc_nope
