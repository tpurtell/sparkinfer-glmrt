"""V4.1 M16 fused expert-slice kernel for intermediate tiling qualification.

Consumes one expert's packed weights and prequantized MXFP8 rows. Legacy
storage uses N256/K128 tiles; exact TP4 uses N128/K128 tiles with N64/K64 tails.
Intermediate width is immutable model geometry: 576 for TP4 backbone experts
and 2304 for local dSpark experts. Packed expert strides derive from its
storage width (exact N128/K128 tail layout when requested); live row/group counts remain runtime arguments.
The caller supplies 0..16 live rows in capacity-16 input/output storage. Width
is static model tiling; row count remains a runtime launch argument. Partial
FC2 output is FP32 and must be reduced across slices before BF16 conversion.
Grouped mode accepts runtime [expert, row_count, route_base, input_row_ids[16]]
metadata. The caller provides valid expert/input indices, disjoint contiguous
route spans and <=16 rows per group; long expert runs use multiple groups.
Nonpositive group row counts mark inactive launch slots. Output route indices
are grouped order. This low-level kernel is not a serving
binding or a replacement planner policy.

Hidden width is model geometry too (5120 for V4.1, 4096 for V4 Flash); it must
be a multiple of the 128-wide K/N tile.

With atomic_tokens=True, grouped mode accumulates directly into a zero-initialized
flat FP32 [token_capacity * hidden] output. Metadata input row IDs also select output
tokens. This removes route/slice planes but changes FP32 addition order and flushes
subnormal atomic operands/results; callers must qualify that numerical contract.
The caller owns output clearing and stream ordering.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Uint32, Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    e2m1x2_scaled_bf16,
    div_rn_f32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    pack_f32x2_to_bfloat2,
    bfloat2_to_float2_scaled,
    shared_ptr_to_u32,
    get_ptr_as_int64,
    red_add_global_f32,
    e2m1x8_to_qmma_e2m1x8,
    fmax_f32,
    fmin_f32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    quantize_block_fp8_mx,
    max_abs_32,
)


@cute.jit
def _swz_mid_idx(row: Int32, col: Int32, width: cutlass.Constexpr):
    """Bank-conflict-free 16xW shared index across MMA fragments, row-major A16, and 32-wide A8 blocks."""
    scol = col ^ (
        ((row & Int32(3)) << Int32(3))
        | ((row >> Int32(2)) & Int32(3))
        | (((col >> Int32(5)) & Int32(1)) << Int32(2))
    )
    return row * Int32(width) + scol


@cute.jit
def _swz_qa_word(row: Int32, word: Int32, qa_words: cutlass.Constexpr, a16: cutlass.Constexpr):
    """Zero-padding XOR word swizzle making FC2 shared activation loads and stores conflict-free."""
    swz = Int32(0)
    if cutlass.const_expr(a16):
        if cutlass.const_expr(qa_words % 32 == 16):
            swz = ((row >> Int32(1)) & Int32(3)) << Int32(2)
        else:
            swz = (row & Int32(7)) << Int32(2)
    else:
        if cutlass.const_expr(qa_words == 8):
            swz = Int32(0)
        elif cutlass.const_expr(qa_words % 32 == 16):
            swz = ((row >> Int32(1)) & Int32(1)) << Int32(3)
        else:
            swz = (row & Int32(3)) << Int32(3)
    return word ^ swz


@dsl_user_op
def fadd_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE-754 round-to-nearest-even FP32 add without reassociation or contraction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
from b12x.moe._shared.kernels.w4a8_staging import (
    stage_row_major_mma,
    stage_repacked_b_slice,
    stage_repacked_sfb_slice,
    stage_repacked_b_k_slice,
    stage_repacked_sfb_k_slice,
    stage_v41_exact_b,
    stage_v41_exact_sfb,
    stage_scale_codes_slice,
    stage_scale_codes_k_slice,
)


class V41FusedSliceKernel:
    def __init__(self, width, *, grouped=False, atomic_tokens=False, intermediate=576, hidden=5120,
                 exact_storage=False, fc2_n_splits=1, scale_codes=False):
        assert width in (64, 128, 192)
        assert hidden > 0 and hidden % 128 == 0
        self.hidden = hidden
        self.hidden_tiles = hidden // 128
        assert fc2_n_splits > 0 and self.hidden_tiles % fc2_n_splits == 0
        self.fc2_n_splits = fc2_n_splits
        assert not atomic_tokens or grouped
        self.atomic_tokens = atomic_tokens
        self.grouped = grouped
        self.width = width
        assert intermediate > 0 and intermediate % 32 == 0
        self.intermediate = intermediate
        self.exact_storage = bool(exact_storage)
        self.scale_codes = bool(scale_codes)
        self.kernel_intermediate = intermediate if self.exact_storage else (intermediate + 127) // 128 * 128
        self.slices = (intermediate + width - 1) // width
        if not self.exact_storage:
            assert self.slices * width <= self.kernel_intermediate, "slice exceeds packed storage"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
        metadata: cute.Tensor | None = None,
        groups: Int32 = 1,
    ):
        self.kernel(x, xs, w13, s13, w2, s2, routing, out, rows, metadata).launch(
            grid=(self.slices, groups if self.grouped else 1, self.fc2_n_splits),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        rows: Int32,
        metadata: cute.Tensor | None,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = tid // 32
        lane = tid % 32
        c = lane % 4
        g = lane // 4
        slice_id = cute.arch.block_idx()[0]
        expert = Int64(0)
        route_base = Int64(0)
        group = cute.arch.block_idx()[1]
        if cutlass.const_expr(self.grouped):
            expert = Int64(metadata[group, 0])
            rows = Int32(metadata[group, 1])
            route_base = Int64(metadata[group, 2])
        input_lo = g
        input_hi = g + 8
        input_scale = g + c * 8
        if cutlass.const_expr(self.grouped):
            if g < rows:
                input_lo = Int32(metadata[group, 3 + g])
            if g + 8 < rows:
                input_hi = Int32(metadata[group, 3 + g + 8])
            if c < 2 and g + c * 8 < rows:
                input_scale = Int32(metadata[group, 3 + g + c * 8])
        active = Int32(1)
        if cutlass.const_expr(self.grouped):
            active = Int32(rows > 0)
        if active > 0:
            start = slice_id * Int32(self.width)
            smem = cutlass.utils.SmemAllocator()
            # Shared weight slots also serve FC2 after the fused activation boundary.
            b = smem.allocate_tensor(
                Uint32, cute.make_layout(self.width * 32), byte_alignment=16
            )
            sf = smem.allocate_tensor(
                Uint32, cute.make_layout(max(self.width * 2, 256)), byte_alignment=16
            )
            qa_words = self.width // 4
            mid = smem.allocate_tensor(
                Float32, cute.make_layout(16 * self.width), byte_alignment=16
            )
            qa = smem.allocate_tensor(
                Uint32, cute.make_layout(16 * qa_words), byte_alignment=16
            )
            qs = smem.allocate_tensor(
                Uint32, cute.make_layout((self.width // 32) * 16), byte_alignment=16
            )
            bb = shared_ptr_to_u32(b.iterator)
            sb = shared_ptr_to_u32(sf.iterator)
            qa_ptr = shared_ptr_to_u32(qa.iterator)
            passes = 2 if self.width == 128 else (3 if self.width == 192 else 1)
            fc1_acc = cute.make_rmem_tensor((passes, 4, 4), Float32)
            fc1_acc.fill(0)
            for kt in range(self.hidden_tiles):
                if cutlass.const_expr(self.exact_storage):
                    stage_v41_exact_b(
                        w13,
                        bb,
                        expert * Int64(self.kernel_intermediate * self.hidden // 4),
                        start,
                        kt,
                        tid,
                        self.width,
                        self.intermediate,
                        self.hidden,
                        True,
                    )
                else:
                    stage_repacked_b_slice(
                        w13,
                        bb,
                        expert * Int64(self.kernel_intermediate * self.hidden // 4),
                        Int32(self.hidden_tiles),
                        kt,
                        start,
                        tid,
                        128,
                        self.width,
                    )
                if cutlass.const_expr(self.exact_storage):
                    stage_v41_exact_b(
                        w13,
                        bb + self.width * 64,
                        expert * Int64(self.kernel_intermediate * self.hidden // 4),
                        start,
                        kt,
                        tid,
                        self.width,
                        self.intermediate,
                        self.hidden,
                        True,
                        1,
                    )
                else:
                    stage_repacked_b_slice(
                        w13,
                        bb + self.width * 64,
                        expert * Int64(self.kernel_intermediate * self.hidden // 4),
                        Int32(self.hidden_tiles),
                        kt,
                        start + self.kernel_intermediate,
                        tid,
                        128,
                        self.width,
                    )
                if cutlass.const_expr(self.scale_codes):
                    stage_scale_codes_slice(s13, sb, expert, kt, start, tid,
                                            self.width, 2 * self.kernel_intermediate,
                                            self.hidden // 32,
                                            self.intermediate if self.exact_storage else 0)
                elif cutlass.const_expr(self.exact_storage):
                    stage_v41_exact_sfb(
                        s13,
                        sb,
                        expert * Int64(self.kernel_intermediate * self.hidden // 64),
                        start,
                        kt,
                        tid,
                        self.width,
                        self.intermediate,
                        self.hidden,
                        True,
                    )
                else:
                    stage_repacked_sfb_slice(
                        s13,
                        sb,
                        expert * Int64(self.kernel_intermediate * self.hidden // 64),
                        Int32(self.hidden_tiles),
                        kt,
                        start,
                        tid,
                        128,
                        self.width,
                    )
                if cutlass.const_expr(self.scale_codes):
                    stage_scale_codes_slice(s13, sb + self.width * 4, expert, kt,
                                            start + self.kernel_intermediate, tid,
                                            self.width, 2 * self.kernel_intermediate,
                                            self.hidden // 32,
                                            self.intermediate if self.exact_storage else 0)
                elif cutlass.const_expr(self.exact_storage):
                    stage_v41_exact_sfb(
                        s13,
                        sb + self.width * 4,
                        expert * Int64(self.kernel_intermediate * self.hidden // 64),
                        start,
                        kt,
                        tid,
                        self.width,
                        self.intermediate,
                        self.hidden,
                        True,
                        1,
                    )
                else:
                    stage_repacked_sfb_slice(
                        s13,
                        sb + self.width * 4,
                        expert * Int64(self.kernel_intermediate * self.hidden // 64),
                        Int32(self.hidden_tiles),
                        kt,
                        start + self.kernel_intermediate,
                        tid,
                        128,
                        self.width,
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                sf_w = cute.make_rmem_tensor((passes, 4), Uint32)
                for p in cutlass.range_constexpr(passes):
                    proj = Int32(0)
                    nf = Int32(0)
                    if cutlass.const_expr(self.width == 64):
                        proj = warp >> Int32(1)
                        nf = warp & Int32(1)
                    elif cutlass.const_expr(self.width == 128):
                        proj = Int32(p)
                        nf = warp
                    else:
                        if cutlass.const_expr(p < 2):
                            proj = Int32(p)
                            nf = warp
                        else:
                            proj = warp >> Int32(1)
                            nf = Int32(4) + (warp & Int32(1))
                    sf_off = proj * Int32(self.width) + nf * Int32(32) + g
                    for nt in cutlass.range_constexpr(4):
                        sf_w[p, nt] = sf[sf_off + Int32(nt * 8)]
                for kb in cutlass.range_constexpr(4):
                    a0 = Uint32(0)
                    a1 = Uint32(0)
                    a2 = Uint32(0)
                    a3 = Uint32(0)
                    if g < rows:
                        a0 = x[input_lo, kt * 32 + kb * 8 + c * 2]
                        a2 = x[input_lo, kt * 32 + kb * 8 + c * 2 + 1]
                    if g + 8 < rows:
                        a1 = x[input_hi, kt * 32 + kb * 8 + c * 2]
                        a3 = x[input_hi, kt * 32 + kb * 8 + c * 2 + 1]
                    scale = Uint32(0)
                    if c < 2 and g + c * 8 < rows:
                        scale = Uint32(xs[input_scale, kt * 4 + kb])
                    for p in cutlass.range_constexpr(passes):
                        proj = Int32(0)
                        nf = Int32(0)
                        if cutlass.const_expr(self.width == 64):
                            proj = warp >> Int32(1)
                            nf = warp & Int32(1)
                        elif cutlass.const_expr(self.width == 128):
                            proj = Int32(p)
                            nf = warp
                        else:
                            if cutlass.const_expr(p < 2):
                                proj = Int32(p)
                                nf = warp
                            else:
                                proj = warp >> Int32(1)
                                nf = Int32(4) + (warp & Int32(1))
                        bw = cute.make_rmem_tensor((4,), Uint32)
                        bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                            bb + proj * Int32(self.width * 64)
                            + ((Int32(kb * (self.width // 32)) + nf) * Int32(32) + lane) * Int32(16)
                        )
                        bw[0], bw[1], bw[2], bw[3] = bw0, bw1, bw2, bw3
                        for nt in cutlass.range_constexpr(4):
                            b0, b1 = e2m1x8_to_qmma_e2m1x8(bw[nt])
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                fc1_acc[p, nt, 0],
                                fc1_acc[p, nt, 1],
                                fc1_acc[p, nt, 2],
                                fc1_acc[p, nt, 3],
                                a0,
                                a1,
                                a2,
                                a3,
                                b0,
                                b1,
                                scale,
                                sf_w[p, nt],
                                bid_b=kb,
                            )
                            fc1_acc[p, nt, 0] = d0
                            fc1_acc[p, nt, 1] = d1
                            fc1_acc[p, nt, 2] = d2
                            fc1_acc[p, nt, 3] = d3
                cute.arch.sync_threads()
            if cutlass.const_expr(self.width == 128):
                for nt in cutlass.range_constexpr(4):
                    for elem in cutlass.range_constexpr(4):
                        row = g + (elem // 2) * 8
                        col = warp * Int32(32) + Int32(nt * 8) + c * Int32(2) + Int32(elem % 2)
                        uv = fc1_acc[0, nt, elem].to(cutlass.BFloat16).to(Float32)
                        gv = fc1_acc[1, nt, elem].to(cutlass.BFloat16).to(Float32)
                        gv = fmin_f32(gv, Float32(10))
                        uv = fmax_f32(Float32(-10), fmin_f32(uv, Float32(10)))
                        value = Float32(0)
                        if row < rows and start + col < self.intermediate:
                            sigmoid = cute.arch.rcp_approx(
                                Float32(1) + cute.math.exp(-gv, fastmath=True)
                            )
                            value = (
                                (gv * sigmoid * uv * routing[route_base + Int64(row)])
                                .to(cutlass.BFloat16)
                                .to(Float32)
                            )
                        mid[_swz_mid_idx(row, col, self.width)] = value
            else:
                fold = cute.make_tensor(
                    cute.recast_ptr(b.iterator, dtype=Float32),
                    cute.make_layout(2 * 16 * self.width),
                )
                for p in cutlass.range_constexpr(passes):
                    proj = Int32(0)
                    nf = Int32(0)
                    if cutlass.const_expr(self.width == 64):
                        proj = warp >> Int32(1)
                        nf = warp & Int32(1)
                    else:
                        if cutlass.const_expr(p < 2):
                            proj = Int32(p)
                            nf = warp
                        else:
                            proj = warp >> Int32(1)
                            nf = Int32(4) + (warp & Int32(1))
                    for nt in cutlass.range_constexpr(4):
                        for elem in cutlass.range_constexpr(4):
                            row = g + (elem // 2) * 8
                            col = nf * Int32(32) + Int32(nt * 8) + c * Int32(2) + Int32(elem % 2)
                            fold[proj * Int32(16 * self.width) + _swz_mid_idx(row, col, self.width)] = fc1_acc[p, nt, elem]
                cute.arch.sync_threads()
                for nf in cutlass.range_constexpr(self.width // 32):
                    for elem in cutlass.range_constexpr(4):
                        row = g + (elem // 2) * 8
                        col = Int32(nf * 32) + warp * Int32(8) + c * Int32(2) + Int32(elem % 2)
                        swz = _swz_mid_idx(row, col, self.width)
                        uv = fold[swz].to(cutlass.BFloat16).to(Float32)
                        gv = fold[Int32(16 * self.width) + swz].to(cutlass.BFloat16).to(Float32)
                        gv = fmin_f32(gv, Float32(10))
                        uv = fmax_f32(Float32(-10), fmin_f32(uv, Float32(10)))
                        value = Float32(0)
                        if row < rows and start + col < self.intermediate:
                            sigmoid = cute.arch.rcp_approx(
                                Float32(1) + cute.math.exp(-gv, fastmath=True)
                            )
                            value = (
                                (gv * sigmoid * uv * routing[route_base + Int64(row)])
                                .to(cutlass.BFloat16)
                                .to(Float32)
                            )
                        mid[swz] = value
            cute.arch.sync_threads()
            for task in range(tid, 16 * (self.width // 32), 128):
                row = task % 16
                block = task // 16
                values = cute.make_rmem_tensor((32,), Float32)
                for j in cutlass.range_constexpr(32):
                    values[j] = mid[_swz_mid_idx(row, block * Int32(32) + Int32(j), self.width)]
                payload, scale = quantize_block_fp8_mx(
                    values, fmax_f32(max_abs_32(values), Float32(1e-4))
                )
                for j in cutlass.range_constexpr(8):
                    qa[row * Int32(qa_words) + _swz_qa_word(row, block * Int32(8) + Int32(j), qa_words, False)] = payload[j]
                qs[block * Int32(16) + row] = scale
            cute.arch.sync_threads()
            # Replicate FC1, but partition disjoint FC2 outputs without regrouping K.
            output_begin = Int32(0)
            if cutlass.const_expr(self.fc2_n_splits > 1):
                output_begin = cute.arch.block_idx()[2] * (self.hidden_tiles // self.fc2_n_splits)
            for local_ot in range(self.hidden_tiles // self.fc2_n_splits):
                ot = output_begin + local_ot
                if cutlass.const_expr(self.exact_storage):
                    stage_v41_exact_b(
                        w2,
                        bb,
                        expert * Int64(self.kernel_intermediate * self.hidden // 8),
                        ot * 128,
                        start,
                        tid,
                        self.width,
                        self.intermediate,
                        self.hidden,
                        False,
                    )
                else:
                    stage_repacked_b_k_slice(
                        w2,
                        bb,
                        expert * Int64(self.kernel_intermediate * self.hidden // 8),
                        Int32(self.kernel_intermediate // 128),
                        start,
                        ot * 128,
                        tid,
                        128,
                        self.width,
                    )
                if cutlass.const_expr(self.scale_codes):
                    stage_scale_codes_k_slice(s2, sf, expert, start, ot * 128, tid,
                                              self.width, self.hidden,
                                              self.intermediate // 32)
                elif cutlass.const_expr(self.exact_storage):
                    stage_v41_exact_sfb(
                        s2,
                        sf,
                        expert * Int64(self.kernel_intermediate * self.hidden // 128),
                        ot * 128,
                        start,
                        tid,
                        self.width,
                        self.intermediate,
                        self.hidden,
                        False,
                    )
                else:
                    stage_repacked_sfb_k_slice(
                        s2,
                        sf,
                        expert * Int64(self.kernel_intermediate * self.hidden // 128),
                        Int32(self.kernel_intermediate // 128),
                        start,
                        ot * 128,
                        tid,
                        128,
                        self.width,
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                acc = cute.make_rmem_tensor((4, 4), Float32)
                acc.fill(0)
                sf_fc2 = cute.make_rmem_tensor((4,), Uint32)
                for kb in cutlass.range_constexpr(self.width // 32):
                    if cutlass.const_expr(kb % 4 == 0):
                        for nt in cutlass.range_constexpr(4):
                            sf_fc2[nt] = sf[Int32((kb // 4) * 128) + warp * Int32(32) + Int32(nt * 8) + g]
                    qa_w = _swz_qa_word(g, Int32(kb * 8) + c * Int32(2), qa_words, False)
                    qa_addr = qa_ptr + (g * Int32(qa_words) + qa_w) * Int32(4)
                    a0, a2 = ld_shared_v2_u32(qa_addr)
                    a1, a3 = ld_shared_v2_u32(qa_addr + Int32(8 * qa_words * 4))
                    scale = Uint32(0)
                    if c < 2:
                        scale = Uint32(qs[Int32(kb * 16) + g + c * Int32(8)])
                    bw = cute.make_rmem_tensor((4,), Uint32)
                    bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                        bb + ((Int32(kb * 4) + warp) * Int32(32) + lane) * Int32(16)
                    )
                    bw[0], bw[1], bw[2], bw[3] = bw0, bw1, bw2, bw3
                    for nt in cutlass.range_constexpr(4):
                        b0, b1 = e2m1x8_to_qmma_e2m1x8(bw[nt])
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            acc[nt, 0],
                            acc[nt, 1],
                            acc[nt, 2],
                            acc[nt, 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            scale,
                            sf_fc2[nt],
                            bid_b=kb % 4,
                        )
                        acc[nt, 0] = d0
                        acc[nt, 1] = d1
                        acc[nt, 2] = d2
                        acc[nt, 3] = d3
                for nt in cutlass.range_constexpr(4):
                    for elem in cutlass.range_constexpr(4):
                        row = g + (elem // 2) * 8
                        col = ot * 128 + warp * 32 + nt * 8 + c * 2 + elem % 2
                        if cutlass.const_expr(self.grouped):
                            if row < rows:
                                if cutlass.const_expr(self.atomic_tokens):
                                    token = Int64(metadata[group, 3 + row])
                                    red_add_global_f32(
                                        get_ptr_as_int64(out, token * Int64(self.hidden) + Int64(col)),
                                        acc[nt, elem],
                                    )
                                else:
                                    out[slice_id, route_base + Int64(row), col] = acc[nt, elem]
                        else:
                            out[slice_id, row, col] = acc[nt, elem]
                cute.arch.sync_threads()


class V41SliceFc1Kernel:
    """FC1 slice projection, SwiGLU and MXFP8 quantization into compact partial scratch."""

    def __init__(self, width, capacity, *, topk=6, intermediate=576, hidden=5120,
                 exact_storage=False, scale_codes=False, activation_precision="a8",
                 swiglu_limit=10.0, silu_bf16_product=False, route_weight_before_fc2=True,
                 fc1_k_splits=1, fc1_cta_splits=1, row_major_resident=False):
        assert width in (32, 64, 128, 192)
        assert capacity > 0 and topk > 0
        assert hidden > 0 and hidden % 128 == 0
        assert intermediate > 0 and intermediate % 32 == 0
        assert activation_precision in ("a8", "a16")
        assert fc1_k_splits in (1, 2, 4, 8)
        assert fc1_cta_splits in (1, 2, 4, 8)
        assert (hidden // 128) % (fc1_k_splits * fc1_cta_splits) == 0
        self.cta_splits = fc1_cta_splits
        self.k_splits = fc1_k_splits
        self.threads = 128 * fc1_k_splits
        self.shared_bytes = (fc1_k_splits * (width * 128 + max(width * 2, 256) * 4)
                             + 16 * width * 4)
        assert self.shared_bytes <= 96 * 1024, "FC1 split exceeds shared-memory budget"
        self.a16 = activation_precision == "a16"
        self.swiglu_limit = float(swiglu_limit)
        self.silu_bf16_product = bool(silu_bf16_product)
        self.route_weight_before_fc2 = bool(route_weight_before_fc2)
        self.width = width
        self.capacity = capacity
        self.topk = topk
        self.routes = capacity * topk
        self.hidden = hidden
        self.hidden_tiles = hidden // 128
        self.intermediate = intermediate
        self.exact_storage = bool(exact_storage)
        self.row_major_resident = bool(row_major_resident)
        self.scale_codes = bool(scale_codes)
        self.kernel_intermediate = intermediate if self.exact_storage else (intermediate + 127) // 128 * 128
        self.slices = (intermediate + width - 1) // width
        if not self.exact_storage:
            assert self.slices * width <= self.kernel_intermediate, "slice exceeds packed storage"

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        routing: cute.Tensor,
        partial: cute.Tensor,
        metadata: cute.Tensor,
        groups: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, xs, w13, s13, routing, partial, metadata).launch(
            grid=(self.slices, groups, self.cta_splits),
            block=(self.threads, 1, 1),
            stream=stream,
        )
        if cutlass.const_expr(self.cta_splits > 1):
            self.fold_kernel(routing, partial, metadata).launch(
                grid=(self.slices, groups, 1), block=(128, 1, 1), stream=stream)


    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        routing: cute.Tensor,
        partial: cute.Tensor,
        metadata: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        team = tid // 128
        team_tid = tid % 128
        warp = team_tid // 32
        lane = tid % 32
        c = lane % 4
        g = lane // 4
        slice_id = cute.arch.block_idx()[0]
        group = cute.arch.block_idx()[1]
        expert = Int64(metadata[group, 0])
        rows = Int32(metadata[group, 1])
        route_base = Int64(metadata[group, 2])
        input_lo = g
        input_hi = g + 8
        input_scale = g + c * 8
        if g < rows:
            input_lo = Int32(metadata[group, 3 + g]) // Int32(self.topk)
        if g + 8 < rows:
            input_hi = Int32(metadata[group, 3 + g + 8]) // Int32(self.topk)
        if c < 2 and g + c * 8 < rows:
            input_scale = Int32(metadata[group, 3 + g + c * 8]) // Int32(self.topk)
        if rows > 0:
            start = slice_id * Int32(self.width)
            smem = cutlass.utils.SmemAllocator()
            b = smem.allocate_tensor(
                Uint32, cute.make_layout(self.k_splits * self.width * 32), byte_alignment=16
            )
            if cutlass.const_expr(self.row_major_resident):
                raw = smem.allocate_tensor(Uint32, cute.make_layout(self.k_splits * self.width * 32), byte_alignment=16)
            sf = smem.allocate_tensor(
                Uint32, cute.make_layout(self.k_splits * max(self.width * 2, 256)), byte_alignment=16
            )
            mid = smem.allocate_tensor(
                Float32, cute.make_layout(16 * self.width), byte_alignment=16
            )
            bbase = team * Int32(self.width * 32)
            sbase = team * Int32(max(self.width * 2, 256))
            bb = shared_ptr_to_u32(b.iterator) + bbase * Int32(4)
            sb = shared_ptr_to_u32(sf.iterator) + sbase * Int32(4)
            passes = 2 if self.width == 128 else (3 if self.width == 192 else 1)
            nts = 2 if self.width == 32 else 4
            fc1_acc = cute.make_rmem_tensor((passes, nts, 4), Float32)
            fc1_acc.fill(0)
            # Each four-warp team owns disjoint K128 iterations; all teams
            # synchronize uniformly and fold their FP32 sums in team order.
            for step in range(self.hidden_tiles // (self.k_splits * self.cta_splits)):
                kt = ((step * Int32(self.cta_splits) + cute.arch.block_idx()[2])
                      * Int32(self.k_splits) + team)
                if cutlass.const_expr(self.row_major_resident):
                    stage_row_major_mma(w13, s13, raw, b, sf, expert, start, kt * 128,
                                        team_tid, self.width, self.intermediate, self.hidden,
                                        True, bbase, sbase)
                else:
                    if cutlass.const_expr(self.exact_storage):
                        stage_v41_exact_b(
                            w13,
                            bb,
                            expert * Int64(self.kernel_intermediate * self.hidden // 4),
                            start,
                            kt,
                            team_tid,
                            self.width,
                            self.intermediate,
                            self.hidden,
                            True,
                        )
                    else:
                        stage_repacked_b_slice(
                            w13,
                            bb,
                            expert * Int64(self.kernel_intermediate * self.hidden // 4),
                            Int32(self.hidden_tiles),
                            kt,
                            start,
                            team_tid,
                            128,
                            self.width,
                        )
                    if cutlass.const_expr(self.exact_storage):
                        stage_v41_exact_b(
                            w13,
                            bb + self.width * 64,
                            expert * Int64(self.kernel_intermediate * self.hidden // 4),
                            start,
                            kt,
                            team_tid,
                            self.width,
                            self.intermediate,
                            self.hidden,
                            True,
                            1,
                        )
                    else:
                        stage_repacked_b_slice(
                            w13,
                            bb + self.width * 64,
                            expert * Int64(self.kernel_intermediate * self.hidden // 4),
                            Int32(self.hidden_tiles),
                            kt,
                            start + self.kernel_intermediate,
                            team_tid,
                            128,
                            self.width,
                        )
                    if cutlass.const_expr(self.scale_codes):
                        stage_scale_codes_slice(s13, sb, expert, kt, start, team_tid,
                                                self.width, 2 * self.kernel_intermediate,
                                                self.hidden // 32,
                                                self.intermediate if self.exact_storage else 0)
                    elif cutlass.const_expr(self.exact_storage):
                        stage_v41_exact_sfb(
                            s13,
                            sb,
                            expert * Int64(self.kernel_intermediate * self.hidden // 64),
                            start,
                            kt,
                            team_tid,
                            self.width,
                            self.intermediate,
                            self.hidden,
                            True,
                        )
                    else:
                        stage_repacked_sfb_slice(
                            s13,
                            sb,
                            expert * Int64(self.kernel_intermediate * self.hidden // 64),
                            Int32(self.hidden_tiles),
                            kt,
                            start,
                            team_tid,
                            128,
                            self.width,
                        )
                    if cutlass.const_expr(self.scale_codes):
                        stage_scale_codes_slice(s13, sb + self.width * 4, expert, kt,
                                                start + self.kernel_intermediate, team_tid,
                                                self.width, 2 * self.kernel_intermediate,
                                                self.hidden // 32,
                                                self.intermediate if self.exact_storage else 0)
                    elif cutlass.const_expr(self.exact_storage):
                        stage_v41_exact_sfb(
                            s13,
                            sb + self.width * 4,
                            expert * Int64(self.kernel_intermediate * self.hidden // 64),
                            start,
                            kt,
                            team_tid,
                            self.width,
                            self.intermediate,
                            self.hidden,
                            True,
                            1,
                        )
                    else:
                        stage_repacked_sfb_slice(
                            s13,
                            sb + self.width * 4,
                            expert * Int64(self.kernel_intermediate * self.hidden // 64),
                            Int32(self.hidden_tiles),
                            kt,
                            start + self.kernel_intermediate,
                            team_tid,
                            128,
                            self.width,
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                sf_w = cute.make_rmem_tensor((passes, nts), Uint32)
                for p in cutlass.range_constexpr(passes):
                    proj = Int32(0)
                    nf = Int32(0)
                    nt_base = Int32(0)
                    if cutlass.const_expr(self.width == 32):
                        proj = warp >> Int32(1)
                        nt_base = (warp & Int32(1)) * Int32(2)
                    elif cutlass.const_expr(self.width == 64):
                        proj = warp >> Int32(1)
                        nf = warp & Int32(1)
                    elif cutlass.const_expr(self.width == 128):
                        proj = Int32(p)
                        nf = warp
                    else:
                        if cutlass.const_expr(p < 2):
                            proj = Int32(p)
                            nf = warp
                        else:
                            proj = warp >> Int32(1)
                            nf = Int32(4) + (warp & Int32(1))
                    sf_off = sbase + proj * Int32(self.width) + nf * Int32(32) + nt_base * Int32(8) + g
                    for nt in cutlass.range_constexpr(nts):
                        sf_w[p, nt] = sf[sf_off + Int32(nt * 8)]
                if cutlass.const_expr(self.a16):
                    shift_c = Uint32(c * 8)
                    for kb in cutlass.range_constexpr(4):
                        bw = cute.make_rmem_tensor((passes, nts), Uint32)
                        for p in cutlass.range_constexpr(passes):
                            proj = Int32(0)
                            nf = Int32(0)
                            nt_base = Int32(0)
                            if cutlass.const_expr(self.width == 32):
                                proj = warp >> Int32(1)
                                nt_base = (warp & Int32(1)) * Int32(2)
                            elif cutlass.const_expr(self.width == 64):
                                proj = warp >> Int32(1)
                                nf = warp & Int32(1)
                            elif cutlass.const_expr(self.width == 128):
                                proj = Int32(p)
                                nf = warp
                            else:
                                if cutlass.const_expr(p < 2):
                                    proj = Int32(p)
                                    nf = warp
                                else:
                                    proj = warp >> Int32(1)
                                    nf = Int32(4) + (warp & Int32(1))
                            b_addr = (
                                bb
                                + proj * Int32(self.width * 64)
                                + ((Int32(kb * (self.width // 32)) + nf) * Int32(32) + lane) * Int32(16)
                                + nt_base * Int32(4)
                            )
                            if cutlass.const_expr(self.width == 32):
                                bw0, bw1 = ld_shared_v2_u32(b_addr)
                                bw[p, 0], bw[p, 1] = bw0, bw1
                            else:
                                bw0, bw1, bw2, bw3 = ld_shared_v4_u32(b_addr)
                                bw[p, 0], bw[p, 1], bw[p, 2], bw[p, 3] = bw0, bw1, bw2, bw3
                        for half in cutlass.range_constexpr(2):
                            a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                            at = kt * 64 + kb * 16 + half * 8 + c
                            if g < rows:
                                a0, a2 = x[input_lo, at], x[input_lo, at + 4]
                            if g + 8 < rows:
                                a1, a3 = x[input_hi, at], x[input_hi, at + 4]
                            src0 = g * Int32(4) + Int32(half * 2)
                            src1 = src0 + Int32(1)
                            for p in cutlass.range_constexpr(passes):
                                for nt in cutlass.range_constexpr(nts):
                                    w0 = Uint32(cute.arch.shuffle_sync(bw[p, nt], src0))
                                    w1 = Uint32(cute.arch.shuffle_sync(bw[p, nt], src1))
                                    sw = (sf_w[p, nt] >> Uint32(kb * 8)) & Uint32(255)
                                    b0 = e2m1x2_scaled_bf16(w0 >> shift_c, sw)
                                    b1 = e2m1x2_scaled_bf16(w1 >> shift_c, sw)
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        fc1_acc[p, nt, 0], fc1_acc[p, nt, 1],
                                        fc1_acc[p, nt, 2], fc1_acc[p, nt, 3],
                                        a0, a1, a2, a3, b0, b1)
                                    fc1_acc[p, nt, 0], fc1_acc[p, nt, 1], fc1_acc[p, nt, 2], fc1_acc[p, nt, 3] = d0, d1, d2, d3
                else:
                    for kb in cutlass.range_constexpr(4):
                        a0 = Uint32(0)
                        a1 = Uint32(0)
                        a2 = Uint32(0)
                        a3 = Uint32(0)
                        if g < rows:
                            a0 = x[input_lo, kt * 32 + kb * 8 + c * 2]
                            a2 = x[input_lo, kt * 32 + kb * 8 + c * 2 + 1]
                        if g + 8 < rows:
                            a1 = x[input_hi, kt * 32 + kb * 8 + c * 2]
                            a3 = x[input_hi, kt * 32 + kb * 8 + c * 2 + 1]
                        scale = Uint32(0)
                        if c < 2 and g + c * 8 < rows:
                            scale = Uint32(xs[input_scale, kt * 4 + kb])
                        for p in cutlass.range_constexpr(passes):
                            proj = Int32(0)
                            nf = Int32(0)
                            nt_base = Int32(0)
                            if cutlass.const_expr(self.width == 32):
                                proj = warp >> Int32(1)
                                nt_base = (warp & Int32(1)) * Int32(2)
                            elif cutlass.const_expr(self.width == 64):
                                proj = warp >> Int32(1)
                                nf = warp & Int32(1)
                            elif cutlass.const_expr(self.width == 128):
                                proj = Int32(p)
                                nf = warp
                            else:
                                if cutlass.const_expr(p < 2):
                                    proj = Int32(p)
                                    nf = warp
                                else:
                                    proj = warp >> Int32(1)
                                    nf = Int32(4) + (warp & Int32(1))
                            b_addr = (
                                bb
                                + proj * Int32(self.width * 64)
                                + ((Int32(kb * (self.width // 32)) + nf) * Int32(32) + lane) * Int32(16)
                                + nt_base * Int32(4)
                            )
                            bw = cute.make_rmem_tensor((nts,), Uint32)
                            if cutlass.const_expr(self.width == 32):
                                bw0, bw1 = ld_shared_v2_u32(b_addr)
                                bw[0], bw[1] = bw0, bw1
                            else:
                                bw0, bw1, bw2, bw3 = ld_shared_v4_u32(b_addr)
                                bw[0], bw[1], bw[2], bw[3] = bw0, bw1, bw2, bw3
                            for nt in cutlass.range_constexpr(nts):
                                b0, b1 = e2m1x8_to_qmma_e2m1x8(bw[nt])
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                    fc1_acc[p, nt, 0],
                                    fc1_acc[p, nt, 1],
                                    fc1_acc[p, nt, 2],
                                    fc1_acc[p, nt, 3],
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    b0,
                                    b1,
                                    scale,
                                    sf_w[p, nt],
                                    bid_b=kb,
                                )
                                fc1_acc[p, nt, 0] = d0
                                fc1_acc[p, nt, 1] = d1
                                fc1_acc[p, nt, 2] = d2
                                fc1_acc[p, nt, 3] = d3
                cute.arch.sync_threads()
            gate = cute.make_rmem_tensor((self.width // 32, 4), Float32)
            up = cute.make_rmem_tensor((self.width // 32, 4), Float32)
            gate.fill(0)
            up.fill(0)
            if cutlass.const_expr(self.width == 128):
                for nt in cutlass.range_constexpr(4):
                    for elem in cutlass.range_constexpr(4):
                        up[nt, elem] = fc1_acc[0, nt, elem]
                        gate[nt, elem] = fc1_acc[1, nt, elem]
                if cutlass.const_expr(self.k_splits > 1):
                    # Weight staging is dead before the fold, so reuse its arena.
                    fold = cute.make_tensor(
                        cute.recast_ptr(b.iterator, dtype=Float32),
                        cute.make_layout((self.k_splits - 1) * 2 * 16 * 128),
                    )
                    if team > 0:
                        for nt in cutlass.range_constexpr(4):
                            for elem in cutlass.range_constexpr(4):
                                row = g + (elem // 2) * 8
                                col = warp * Int32(32) + Int32(nt * 8) + c * Int32(2) + Int32(elem % 2)
                                swz = _swz_mid_idx(row, col, 128)
                                fold[((team - Int32(1)) * Int32(2)) * Int32(16 * 128) + swz] = up[nt, elem]
                                fold[((team - Int32(1)) * Int32(2) + Int32(1)) * Int32(16 * 128) + swz] = gate[nt, elem]
                    cute.arch.sync_threads()
                    if team == 0:
                        for nt in cutlass.range_constexpr(4):
                            for elem in cutlass.range_constexpr(4):
                                row = g + (elem // 2) * 8
                                col = warp * Int32(32) + Int32(nt * 8) + c * Int32(2) + Int32(elem % 2)
                                swz = _swz_mid_idx(row, col, 128)
                                for part in cutlass.range_constexpr(1, self.k_splits):
                                    up[nt, elem] = fadd_rn_f32(up[nt, elem], fold[((part - 1) * 2) * (16 * 128) + swz])
                                    gate[nt, elem] = fadd_rn_f32(gate[nt, elem], fold[((part - 1) * 2 + 1) * (16 * 128) + swz])
            else:
                fold = cute.make_tensor(
                    cute.recast_ptr(b.iterator, dtype=Float32),
                    cute.make_layout(self.k_splits * 2 * 16 * self.width),
                )
                for p in cutlass.range_constexpr(passes):
                    proj = Int32(0)
                    nf = Int32(0)
                    nt_base = Int32(0)
                    if cutlass.const_expr(self.width == 32):
                        proj = warp >> Int32(1)
                        nt_base = (warp & Int32(1)) * Int32(2)
                    elif cutlass.const_expr(self.width == 64):
                        proj = warp >> Int32(1)
                        nf = warp & Int32(1)
                    else:
                        if cutlass.const_expr(p < 2):
                            proj = Int32(p)
                            nf = warp
                        else:
                            proj = warp >> Int32(1)
                            nf = Int32(4) + (warp & Int32(1))
                    for nt in cutlass.range_constexpr(nts):
                        for elem in cutlass.range_constexpr(4):
                            row = g + (elem // 2) * 8
                            col = nf * Int32(32) + (nt_base + Int32(nt)) * Int32(8) + c * Int32(2) + Int32(elem % 2)
                            swz = _swz_mid_idx(row, col, self.width)
                            fold[(team * Int32(2) + proj) * Int32(16 * self.width) + swz] = fc1_acc[p, nt, elem]
                cute.arch.sync_threads()
                if team == 0:
                    for nf in cutlass.range_constexpr(self.width // 32):
                        for elem in cutlass.range_constexpr(4):
                            row = g + (elem // 2) * 8
                            col = Int32(nf * 32) + warp * Int32(8) + c * Int32(2) + Int32(elem % 2)
                            swz = _swz_mid_idx(row, col, self.width)
                            u_val = fold[swz]
                            g_val = fold[Int32(16 * self.width) + swz]
                            for part in cutlass.range_constexpr(1, self.k_splits):
                                u_val = fadd_rn_f32(u_val, fold[(part * 2) * (16 * self.width) + swz])
                                g_val = fadd_rn_f32(g_val, fold[(part * 2 + 1) * (16 * self.width) + swz])
                            up[nf, elem] = u_val
                            gate[nf, elem] = g_val
            row, col, base = Int32(0), Int32(0), Int64(0)
            if cutlass.const_expr(self.cta_splits > 1):
                raw = self.raw_projection(partial)
                if team == 0:
                    for nf in cutlass.range_constexpr(self.width // 32):
                        for elem in cutlass.range_constexpr(4):
                            row = g + (elem // 2) * 8
                            if cutlass.const_expr(self.width == 128):
                                col = warp * Int32(32) + Int32(nf * 8) + c * Int32(2) + Int32(elem % 2)
                            else:
                                col = nf * 32 + warp * 8 + c * 2 + elem % 2
                            if row < rows:
                                base = (((Int64(slice_id) * self.routes + route_base + Int64(row))
                                         * self.cta_splits + cute.arch.block_idx()[2]) * 2) * self.width
                                raw[base + Int64(col)] = gate[nf, elem]
                                raw[base + Int64(self.width + col)] = up[nf, elem]
            else:
                self.finish(gate, up, mid, routing, partial, tid, team, warp, c, g,
                            slice_id, route_base, rows, start, self.width == 128)

    @cute.jit
    def raw_projection(self, partial: cute.Tensor):
        compact_words = self.slices * self.routes * (
            self.width // 2 if self.a16 else self.width // 4 + self.width // 32)
        return cute.make_tensor(
            cute.recast_ptr(partial.iterator, dtype=Float32) + compact_words,
            cute.make_layout(self.slices * self.routes * self.cta_splits * 2 * self.width))

    @cute.kernel
    def fold_kernel(self, routing: cute.Tensor, partial: cute.Tensor, metadata: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        c, g = lane % 4, lane // 4
        slice_id, group = cute.arch.block_idx()[0], cute.arch.block_idx()[1]
        rows = Int32(metadata[group, 1])
        route_base = Int64(metadata[group, 2])
        if rows > 0:
            raw = self.raw_projection(partial)
            gate = cute.make_rmem_tensor((self.width // 32, 4), Float32)
            up = cute.make_rmem_tensor((self.width // 32, 4), Float32)
            gate.fill(0)
            up.fill(0)
            for nf in cutlass.range_constexpr(self.width // 32):
                for elem in cutlass.range_constexpr(4):
                    row = g + (elem // 2) * 8
                    col = nf * 32 + warp * 8 + c * 2 + elem % 2
                    if row < rows:
                        base = ((Int64(slice_id) * self.routes + route_base + Int64(row))
                                * self.cta_splits * 2) * self.width
                        gate[nf, elem] = raw[base + Int64(col)]
                        up[nf, elem] = raw[base + Int64(self.width + col)]
                        for part in cutlass.range_constexpr(1, self.cta_splits):
                            offset = base + Int64(part * 2 * self.width + col)
                            gate[nf, elem] = fadd_rn_f32(gate[nf, elem], raw[offset])
                            up[nf, elem] = fadd_rn_f32(up[nf, elem], raw[offset + Int64(self.width)])
            smem = cutlass.utils.SmemAllocator()
            mid = smem.allocate_tensor(Float32, cute.make_layout(16 * self.width), byte_alignment=16)
            self.finish(gate, up, mid, routing, partial, tid, Int32(0), warp, c, g,
                        slice_id, route_base, rows, slice_id * Int32(self.width), False)

    @cute.jit
    def finish(self, gate, up, mid, routing, partial, tid, team, warp, c, g,
               slice_id, route_base, rows, start, chunk_per_warp: cutlass.Constexpr = False):
        row, col = Int32(0), Int32(0)
        if team == 0:
            for nf in cutlass.range_constexpr(self.width // 32):
                for elem in cutlass.range_constexpr(4):
                    row = g + (elem // 2) * 8
                    if cutlass.const_expr(chunk_per_warp):
                        col = warp * Int32(32) + Int32(nf * 8) + c * Int32(2) + Int32(elem % 2)
                    else:
                        col = nf * 32 + warp * 8 + c * 2 + elem % 2
                    gv = gate[nf, elem].to(cutlass.BFloat16).to(Float32)
                    uv = up[nf, elem].to(cutlass.BFloat16).to(Float32)
                    if cutlass.const_expr(self.swiglu_limit > 0):
                        gv = fmin_f32(gv, Float32(self.swiglu_limit))
                        uv = fmax_f32(Float32(-self.swiglu_limit), fmin_f32(uv, Float32(self.swiglu_limit)))
                    value = Float32(0)
                    if row < rows and start + col < self.intermediate:
                        if cutlass.const_expr(self.silu_bf16_product):
                            # Explicit packing preserves the BF16 store boundary through optimization.
                            gv, uv = bfloat2_to_float2_scaled(
                                pack_f32x2_to_bfloat2(gv, uv), Float32(1))
                            silu = div_rn_f32(gv, Float32(1) + cute.math.exp(-gv, fastmath=False))
                            silu, _ = bfloat2_to_float2_scaled(
                                pack_f32x2_to_bfloat2(silu, silu), Float32(1))
                            product = silu * uv
                            product, _ = bfloat2_to_float2_scaled(
                                pack_f32x2_to_bfloat2(product, product), Float32(1))
                        else:
                            sigmoid = cute.arch.rcp_approx(Float32(1) + cute.math.exp(-gv, fastmath=True))
                            product = gv * sigmoid * uv
                        if cutlass.const_expr(self.route_weight_before_fc2):
                            product *= routing[route_base + Int64(row)]
                        value = product.to(cutlass.BFloat16).to(Float32)
                    mid[_swz_mid_idx(row, col, self.width)] = value
        cute.arch.sync_threads()
        if cutlass.const_expr(self.a16):
            partial_bf16 = cute.make_tensor(
                cute.recast_ptr(partial.iterator, dtype=cutlass.BFloat16),
                cute.make_layout(self.slices * self.routes * self.width),
            )
            base = (Int64(slice_id) * Int64(self.routes) + route_base) * Int64(self.width)
            for task in range(tid, 16 * self.width, (128 if self.cta_splits > 1 else self.threads)):
                row, col = task // self.width, task % self.width
                if row < rows:
                    partial_bf16[base + Int64(task)] = mid[_swz_mid_idx(row, col, self.width)].to(cutlass.BFloat16)
        else:
            partial_u32 = cute.make_tensor(
                cute.recast_ptr(partial.iterator, dtype=Uint32),
                cute.make_layout(self.slices * self.routes * (self.width // 4 + self.width // 32)),
            )
            qa_base = (Int64(slice_id) * Int64(self.routes) + route_base) * Int64(self.width // 4)
            qs_offset = Int64(self.slices * self.routes * (self.width // 4))
            qs_base = qs_offset + (Int64(slice_id) * Int64(self.routes) + route_base) * Int64(self.width // 32)
            for task in range(tid, 16 * (self.width // 32), (128 if self.cta_splits > 1 else self.threads)):
                row = task % 16
                block = task // 16
                out_idx = row * (self.width // 32) + block
                values = cute.make_rmem_tensor((32,), Float32)
                for j in cutlass.range_constexpr(32):
                    values[j] = mid[_swz_mid_idx(row, block * Int32(32) + Int32(j), self.width)]
                payload, scale = quantize_block_fp8_mx(
                    values, fmax_f32(max_abs_32(values), Float32(1e-4))
                )
                if row < rows:
                    for j in cutlass.range_constexpr(8):
                        partial_u32[qa_base + Int64(out_idx * 8 + j)] = payload[j]
                    partial_u32[qs_base + Int64(out_idx)] = scale


class V41SliceFc2Kernel:
    """Ordered in-register FC2 slice accumulation into route-major FP32 output."""

    def __init__(self, width, capacity, *, topk=6, intermediate=576, hidden=5120,
                 exact_storage=False, fc2_n_splits=1, scale_codes=False,
                 activation_precision="a8", route_weight_before_fc2=True, round_fc2_bf16=False, row_major_resident=False):
        assert width in (32, 64, 128, 192)
        assert capacity > 0 and topk > 0
        assert hidden > 0 and hidden % 128 == 0
        assert intermediate > 0 and intermediate % 32 == 0
        assert activation_precision in ("a8", "a16")
        self.a16 = activation_precision == "a16"
        self.route_weight_before_fc2 = bool(route_weight_before_fc2)
        self.round_fc2_bf16 = bool(round_fc2_bf16)
        self.width = width
        self.capacity = capacity
        self.topk = topk
        self.routes = capacity * topk
        self.hidden = hidden
        self.hidden_tiles = hidden // 128
        assert fc2_n_splits > 0 and self.hidden_tiles % fc2_n_splits == 0
        self.fc2_n_splits = 8 if (self.hidden_tiles % 8 == 0 and 8 % fc2_n_splits == 0) else fc2_n_splits
        self.intermediate = intermediate
        self.exact_storage = bool(exact_storage)
        self.row_major_resident = bool(row_major_resident)
        self.scale_codes = bool(scale_codes)
        self.kernel_intermediate = intermediate if self.exact_storage else (intermediate + 127) // 128 * 128
        self.w2_intermediate = self.kernel_intermediate
        self.w2_scale_columns = intermediate // 32
        self.slices = (intermediate + width - 1) // width
        if not self.exact_storage:
            assert self.slices * width <= self.kernel_intermediate, "slice exceeds packed storage"

    @cute.jit
    def __call__(
        self,
        w2: cute.Tensor,
        s2: cute.Tensor,
        partial: cute.Tensor,
        out: cute.Tensor,
        metadata: cute.Tensor,
        inverse: cute.Tensor,
        live_rows: cute.Tensor,
        groups: Int32,
        stream: cuda.CUstream,
        routing: cute.Tensor = None,
    ):
        self.kernel(w2, s2, partial, out, metadata, inverse, live_rows, routing).launch(
            grid=(self.fc2_n_splits, groups, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        w2: cute.Tensor,
        s2: cute.Tensor,
        partial: cute.Tensor,
        out: cute.Tensor,
        metadata: cute.Tensor,
        inverse: cute.Tensor,
        live_rows: cute.Tensor,
        routing: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = tid // 32
        lane = tid % 32
        c = lane % 4
        g = lane // 4
        split_id = cute.arch.block_idx()[0]
        group = cute.arch.block_idx()[1]
        tiles_per_split = self.hidden_tiles // self.fc2_n_splits
        output_begin = split_id * Int32(tiles_per_split)
        expert = Int64(metadata[group, 0])
        rows = Int32(metadata[group, 1])
        route_base = Int64(metadata[group, 2])
        if rows > 0:
            route_lo = Int64(0)
            route_hi = Int64(0)
            if g < rows:
                route_lo = Int64(metadata[group, 3 + g])
            if g + 8 < rows:
                route_hi = Int64(metadata[group, 3 + g + 8])
            smem = cutlass.utils.SmemAllocator()
            b = smem.allocate_tensor(
                Uint32, cute.make_layout(self.width * 16), byte_alignment=16
            )
            if cutlass.const_expr(self.row_major_resident):
                raw = smem.allocate_tensor(Uint32, cute.make_layout(self.width * 16), byte_alignment=16)
            sf = smem.allocate_tensor(
                Uint32, cute.make_layout(max(self.width * 2, 256)), byte_alignment=16
            )
            qa_slots = self.slices if self.slices <= 8 else 1
            qa_words = self.width // 2 if self.a16 else self.width // 4
            qa = smem.allocate_tensor(
                Uint32, cute.make_layout(qa_slots * 16 * qa_words), byte_alignment=16
            )
            qs = smem.allocate_tensor(
                Uint32, cute.make_layout(1 if self.a16 else qa_slots * (self.width // 32) * 16), byte_alignment=16
            )
            bb = shared_ptr_to_u32(b.iterator)
            qa_ptr = shared_ptr_to_u32(qa.iterator)
            partial_u32 = cute.make_tensor(
                cute.recast_ptr(partial.iterator, dtype=Uint32),
                cute.make_layout(self.slices * self.routes * (self.width // 2 if self.a16 else self.width // 4 + self.width // 32)),
            )
            qs_offset = Int64(self.slices * self.routes * (self.width // 4))
            if cutlass.const_expr(self.slices <= 8):
                for s in range(self.slices):
                    qa_base = (Int64(s) * Int64(self.routes) + route_base) * Int64(qa_words)
                    for idx in range(tid, 16 * qa_words, 128):
                        r_idx = idx // qa_words
                        w_idx = idx % qa_words
                        dst = (s * Int32(16) + r_idx) * Int32(qa_words) + _swz_qa_word(r_idx, w_idx, qa_words, self.a16)
                        if r_idx < rows:
                            qa[dst] = Uint32(partial_u32[qa_base + Int64(idx)])
                        else:
                            qa[dst] = Uint32(0)
                    if cutlass.const_expr(not self.a16):
                        qs_base = qs_offset + (Int64(s) * Int64(self.routes) + route_base) * Int64(self.width // 32)
                        for idx in range(tid, 16 * (self.width // 32), 128):
                            r_idx = idx // (self.width // 32)
                            w_idx = idx % (self.width // 32)
                            dst_s = (s * Int32(self.width // 32) + w_idx) * Int32(16) + r_idx
                            if r_idx < rows:
                                qs[dst_s] = Uint32(partial_u32[qs_base + Int64(idx)])
                            else:
                                qs[dst_s] = Uint32(0)
                cute.arch.sync_threads()
            for local_ot in range(tiles_per_split):
                ot = output_begin + local_ot
                total = cute.make_rmem_tensor((4, 4), Float32)
                total.fill(0)
                for slice_id in range(self.slices):
                    start = slice_id * Int32(self.width)
                    if cutlass.const_expr(self.row_major_resident):
                        stage_row_major_mma(w2, s2, raw, b, sf, expert, ot * 128, start,
                                            tid, self.width, self.intermediate, self.hidden, False)
                    else:
                        if cutlass.const_expr(self.exact_storage):
                            stage_v41_exact_b(
                                w2,
                                bb,
                                expert * Int64(self.w2_intermediate * self.hidden // 8),
                                ot * 128,
                                start,
                                tid,
                                self.width,
                                self.intermediate,
                                self.hidden,
                                False,
                                stored_intermediate=self.w2_intermediate,
                            )
                        else:
                            stage_repacked_b_k_slice(
                                w2,
                                bb,
                                expert * Int64(self.kernel_intermediate * self.hidden // 8),
                                Int32(self.kernel_intermediate // 128),
                                start,
                                ot * 128,
                                tid,
                                128,
                                self.width,
                            )
                        if cutlass.const_expr(self.scale_codes):
                            stage_scale_codes_k_slice(s2, sf, expert, start, ot * 128, tid,
                                                      self.width, self.hidden,
                                                      self.w2_scale_columns)
                        elif cutlass.const_expr(self.exact_storage):
                            stage_v41_exact_sfb(
                                s2,
                                sf,
                                expert * Int64(self.kernel_intermediate * self.hidden // 128),
                                ot * 128,
                                start,
                                tid,
                                self.width,
                                self.intermediate,
                                self.hidden,
                                False,
                            )
                        else:
                            stage_repacked_sfb_k_slice(
                                s2,
                                sf,
                                expert * Int64(self.kernel_intermediate * self.hidden // 128),
                                Int32(self.kernel_intermediate // 128),
                                start,
                                ot * 128,
                                tid,
                                128,
                                self.width,
                            )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(self.slices > 8):
                        qa_base = (Int64(slice_id) * Int64(self.routes) + route_base) * Int64(qa_words)
                        for idx in range(tid, 16 * qa_words, 128):
                            r_idx = idx // qa_words
                            w_idx = idx % qa_words
                            dst = r_idx * Int32(qa_words) + _swz_qa_word(r_idx, w_idx, qa_words, self.a16)
                            if r_idx < rows:
                                qa[dst] = Uint32(partial_u32[qa_base + Int64(idx)])
                            else:
                                qa[dst] = Uint32(0)
                        if cutlass.const_expr(not self.a16):
                            qs_base = qs_offset + (Int64(slice_id) * Int64(self.routes) + route_base) * Int64(self.width // 32)
                            for idx in range(tid, 16 * (self.width // 32), 128):
                                r_idx = idx // (self.width // 32)
                                w_idx = idx % (self.width // 32)
                                dst_s = w_idx * Int32(16) + r_idx
                                if r_idx < rows:
                                    qs[dst_s] = Uint32(partial_u32[qs_base + Int64(idx)])
                                else:
                                    qs[dst_s] = Uint32(0)
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    acc = cute.make_rmem_tensor((4, 4), Float32)
                    acc.fill(0)
                    sf_fc2 = cute.make_rmem_tensor((4,), Uint32)
                    qa_slot = slice_id if cutlass.const_expr(self.slices <= 8) else Int32(0)
                    if cutlass.const_expr(self.a16):
                        shift_c = Uint32(c * 8)
                        qa_row_lo = (qa_slot * Int32(16) + g) * Int32(qa_words)
                        qa_row_hi = qa_row_lo + Int32(8 * qa_words)
                        for kb in cutlass.range_constexpr(self.width // 32):
                            if cutlass.const_expr(kb % 4 == 0):
                                for nt in cutlass.range_constexpr(4):
                                    sf_fc2[nt] = sf[Int32((kb // 4) * 128) + warp * Int32(32) + Int32(nt * 8) + g]
                            bw = cute.make_rmem_tensor((4,), Uint32)
                            bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                                bb + ((Int32(kb * 4) + warp) * Int32(32) + lane) * Int32(16)
                            )
                            bw[0], bw[1], bw[2], bw[3] = bw0, bw1, bw2, bw3
                            for half in cutlass.range_constexpr(2):
                                at0 = _swz_qa_word(g, Int32(kb * 16 + half * 8) + c, qa_words, True)
                                at1 = at0 ^ Int32(4)
                                a0, a2 = qa[qa_row_lo + at0], qa[qa_row_lo + at1]
                                a1, a3 = qa[qa_row_hi + at0], qa[qa_row_hi + at1]
                                src0 = g * Int32(4) + Int32(half * 2)
                                src1 = src0 + Int32(1)
                                for nt in cutlass.range_constexpr(4):
                                    w0 = Uint32(cute.arch.shuffle_sync(bw[nt], src0))
                                    w1 = Uint32(cute.arch.shuffle_sync(bw[nt], src1))
                                    sw = (sf_fc2[nt] >> Uint32((kb % 4) * 8)) & Uint32(255)
                                    b0 = e2m1x2_scaled_bf16(w0 >> shift_c, sw)
                                    b1 = e2m1x2_scaled_bf16(w1 >> shift_c, sw)
                                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                        acc[nt, 0], acc[nt, 1], acc[nt, 2], acc[nt, 3],
                                        a0, a1, a2, a3, b0, b1)
                                    acc[nt, 0], acc[nt, 1], acc[nt, 2], acc[nt, 3] = d0, d1, d2, d3
                    else:
                        for kb in cutlass.range_constexpr(self.width // 32):
                            if cutlass.const_expr(kb % 4 == 0):
                                for nt in cutlass.range_constexpr(4):
                                    sf_fc2[nt] = sf[Int32((kb // 4) * 128) + warp * Int32(32) + Int32(nt * 8) + g]
                            qa_w = _swz_qa_word(g, Int32(kb * 8) + c * Int32(2), qa_words, False)
                            qa_addr = qa_ptr + ((qa_slot * Int32(16) + g) * Int32(qa_words) + qa_w) * Int32(4)
                            a0, a2 = ld_shared_v2_u32(qa_addr)
                            a1, a3 = ld_shared_v2_u32(qa_addr + Int32(8 * qa_words * 4))
                            scale = Uint32(0)
                            if c < 2:
                                scale = Uint32(qs[(qa_slot * Int32(self.width // 32) + Int32(kb)) * Int32(16) + g + c * Int32(8)])
                            bw = cute.make_rmem_tensor((4,), Uint32)
                            bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                                bb + ((Int32(kb * 4) + warp) * Int32(32) + lane) * Int32(16)
                            )
                            bw[0], bw[1], bw[2], bw[3] = bw0, bw1, bw2, bw3
                            for nt in cutlass.range_constexpr(4):
                                b0, b1 = e2m1x8_to_qmma_e2m1x8(bw[nt])
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                    acc[nt, 0],
                                    acc[nt, 1],
                                    acc[nt, 2],
                                    acc[nt, 3],
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    b0,
                                    b1,
                                    scale,
                                    sf_fc2[nt],
                                    bid_b=kb % 4,
                                )
                                acc[nt, 0] = d0
                                acc[nt, 1] = d1
                                acc[nt, 2] = d2
                                acc[nt, 3] = d3
                    for nt in cutlass.range_constexpr(4):
                        for elem in cutlass.range_constexpr(4):
                            total[nt, elem] = fadd_rn_f32(total[nt, elem], acc[nt, elem])
                    cute.arch.sync_threads()
                for nt in cutlass.range_constexpr(4):
                    for elem in cutlass.range_constexpr(4):
                        row = g + (elem // 2) * 8
                        col = ot * 128 + warp * 32 + nt * 8 + c * 2 + elem % 2
                        if row < rows:
                            route = route_lo if cutlass.const_expr(elem < 2) else route_hi
                            value = total[nt, elem]
                            if cutlass.const_expr(self.round_fc2_bf16):
                                value, _ = bfloat2_to_float2_scaled(
                                    pack_f32x2_to_bfloat2(value, value), Float32(1))
                            if cutlass.const_expr(not self.route_weight_before_fc2):
                                value *= routing[route_base + Int64(row)]
                            out[route, col] = value
        live_routes = max(Int32(0), min(live_rows[0] * Int32(self.topk), Int32(self.routes)))
        col_begin = output_begin * Int32(128)
        col_end = col_begin + Int32(tiles_per_split * 128)
        for r in range(group, live_routes, cute.arch.grid_dim()[1]):
            if inverse[r] < 0:
                for col in range(col_begin + tid, col_end, 128):
                    out[Int64(r), col] = Float32(0)
