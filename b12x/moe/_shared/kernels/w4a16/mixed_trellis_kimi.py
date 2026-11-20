"""SM121 Kimi K3.1 TP6 small-M cooperative mixed-Trellis specialization.

Targets the DGX Spark (SM121, GB10) ``tp6-width512`` K2/K3 decode/verify
exports (``m1`` and ``m80``, ``hidden=3584, intermediate=512, experts=896,
top_k=16, situ, route_block=8, l2=2``).

Three structural optimizations work together across the full ``m=1..64`` curve
while preserving exact ``m1 == m80`` row invariance and baseline-identical FP16/
FP32 rounding:

1. Live-route vectorized input rotation and SiTU activation: both non-GEMM
   phases iterate strictly over the ``active_m * top_k`` live routes rather
   than ``packed_route_count`` (which is 58-88% block-alignment padding at
   ``m=1..64``), using 64-bit vector loads/stores and CTA-shared token-block
   ordering so all 8 warps on an SM share ``x_input`` in L1.
2. Double-buffered K256 slabs for K2 tiles with multicast+shuffle+``prmt``
   tail-biting ring extraction (halving B shared-memory loads and replacing
   64-bit funnel shifts), warp-local zero-barrier route resolution, and a
   3-barrier batched disjoint-SMEM epilogue.
3. Runtime-adaptive M8 expert-span merging (up to 4 blocks/expert at ``m80``):
   a parallel SMEM duplicate check bypasses compaction in 2 barriers when no
   expert spans multiple blocks (``m=2..10``); when spans form (``m>=11``),
   spans are emitted as ``[multi-block, single-block]`` with dedicated
   ``span_kind in {1, 2, 4}`` K2 loops and paired with ``reverse_tail_wave``
   in ``_run_persistent_gemm`` so ragged last-wave single-block tiles never
   stack onto the front CTAs that executed multi-block spans.
"""

from __future__ import annotations

from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, Uint64, dsl_user_op

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    create_l2_evict_first_policy,
    cvt_bf16x2_to_f16x2_via_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    f16x2_to_f32x2,
    get_ptr_as_int64,
    half2_mul,
    ld_global_nc_v2_u32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x2_b16,
    pack_f32x2_to_f16x2,
    packed_dequant_trellis_mul1_to_half2x4,
    packed_dequant_trellis_to_half2x4,
    shared_ptr_to_u32,
    st_global_v2_u32,
    st_global_v4_u32,
    st_shared_f16_from_f32,
    st_shared_i32,
    st_shared_v4_f32,
    st_shared_v4_u32,
)
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.trellis_codebooks import MCG, MUL1

from .kernel import (
    _SQG_XOR_CHEB_T12_SMEM_REGION_BYTES,
    W4A16FusedMoeKernel,
)
from .mixed_trellis import W4A16MixedTrellisKernel


@dsl_user_op
def _cp_async4_shared_global_l2hint(
    smem_addr: Int32,
    gmem_addr: Int64,
    cache_policy: Uint64,
    *,
    loc=None,
    ip=None,
):
    """Unconditional 16-byte ``cp.async.cg`` with L2 cache hint."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Uint64(cache_policy).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prmt_b32_reg(
    lo: Uint32,
    hi: Uint32,
    sel: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Byte-permute ``{lo, hi}`` with a lane-invariant register selector."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(lo).ir_value(loc=loc, ip=ip),
                Uint32(hi).ir_value(loc=loc, ip=ip),
                Uint32(sel).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class W4A16MixedTrellisKimiTP6(W4A16MixedTrellisKernel):
    """Kimi K3.1 TP6 (SM121) cooperative K2-slab + adaptive M8-span kernel."""

    B_SLAB_BYTES = 16384
    A_BLOCK_SLAB_BYTES = 4096
    MAX_SPAN = 4

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        descriptor_bits: int = 10,
    ):
        super().__init__(
            driver=driver,
            tier0=tier0,
            tier1=tier1,
            descriptor_bits=descriptor_bits,
            paired_boundary=None,
            compact_weights=False,
            token_major_rotation=False,
        )
        if (
            self.hidden_size != 3584
            or self.intermediate_size != 512
            or self.top_k != 16
            or self.cta_threads != 256
            or self.driver.moe_block_size != 8
            or self.driver.fc2.moe_block_size != 8
            or self.driver.fc2.schedule_route_block_factor != 1
            or self.tier0.trellis_bits != 2
            or self.tier0.trellis_codebook not in (MCG, MUL1)
            or self.tier0.fc1.tile_n != 256
            or self.tier0.fc2.tile_n != 256
            or self.tier0.fc1.tile_k != 64
            or self.tier0.fc2.tile_k != 64
            or self.driver.direct_topk_routes
            or self.driver.fused_input_rotation
            or self.driver.broadcast_suh
            or self.driver.coupled_hadamard
            or not self.driver.activation_is_situ
            or self.driver.rotation_input_dtype != "bf16"
        ):
            raise ValueError("W4A16MixedTrellisKimiTP6 requires exact Kimi TP6 K2/K3 geometry")

        self.slab_offset = int(self.tier0.fc1.sh_b_off * 16)
        if self.slab_offset != 128:
            raise ValueError(f"unexpected Trellis slab offset {self.slab_offset}")

        self.max_span = 1 if int(self.size_m) == 1 else self.MAX_SPAN
        slab_stage_bytes = self.B_SLAB_BYTES + self.max_span * self.A_BLOCK_SLAB_BYTES
        slab_end = self.slab_offset + 2 * slab_stage_bytes
        base_bytes = max(int(self.shared_words) * 4, slab_end)

        if self.max_span > 1:
            self.span_smem_off = ((base_bytes + 15) // 16) * 16
            span_region_bytes = ((16 + 4 * int(self.driver.max_m_blocks) + 15) // 16) * 16
            total_bytes = self.span_smem_off + span_region_bytes
        else:
            self.span_smem_off = 0
            total_bytes = ((base_bytes + 15) // 16) * 16

        self.shared_words = total_bytes // 4

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return super().__cache_key__ + (
            ("kimi_tp6_k2_slab_span_v2", self.max_span, self.span_smem_off),
        )

    @cute.jit
    def _rotate_live_routes(
        self,
        x_input: cute.Tensor,
        a_gate: cute.Tensor,
        a_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        global_to_combined: cute.Tensor,
        route_num_experts: Int32,
        total_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        # Order units as ((token * 28 + blk) * 16 + k) so all 8 warps of a CTA
        # share the same (token, H128 block) in L1 while keeping 448*active_m
        # warp units across the grid even at m=1.
        lane = tid & Int32(31)
        warps_per_cta = Int32(8)
        hidden = Int32(3584)
        nblk = Int32(28)
        unit = cta * warps_per_cta + (tid >> Int32(5))
        unit_stride = grid_x * warps_per_cta
        total_units = active_m * Int32(448)
        elem = lane * Int32(4)

        while unit < total_units:
            tb = unit >> Int32(4)
            k = unit & Int32(15)
            token = tb // nblk
            blk = tb - token * nblk
            route = (token << Int32(4)) + k

            ex_lane0 = Int32(-1)
            if lane == Int32(0):
                ge = raw_topk_ids[route].to(Int32)
                if ge >= Int32(0) and ge < route_num_experts:
                    ce = global_to_combined[ge].to(Int32)
                    if ce >= Int32(0) and ce < total_experts:
                        ex_lane0 = ce
            expert = cute.arch.shuffle_sync(ex_lane0, Int32(0))

            if expert >= Int32(0):
                col0 = blk * Int32(128) + elem
                x_off = Int64(token) * Int64(3584) + Int64(col0)
                s_off = Int64(expert) * Int64(3584) + Int64(col0)
                out_off = Int64(route) * Int64(3584) + Int64(col0)

                xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, x_off))
                gw0, gw1 = ld_global_nc_v2_u32(get_ptr_as_int64(gate_suh, s_off))
                uw0, uw1 = ld_global_nc_v2_u32(get_ptr_as_int64(up_suh, s_off))

                xh0 = cvt_bf16x2_to_f16x2_via_f32(xw0)
                xh1 = cvt_bf16x2_to_f16x2_via_f32(xw1)

                gv0, gv1 = f16x2_to_f32x2(half2_mul(xh0, gw0))
                gv2, gv3 = f16x2_to_f32x2(half2_mul(xh1, gw1))
                gh0, gh1, gh2, gh3 = self.driver._had128_quad(gv0, gv1, gv2, gv3, lane)
                st_global_v2_u32(
                    get_ptr_as_int64(a_gate, out_off),
                    pack_f32x2_to_f16x2(gh0, gh1),
                    pack_f32x2_to_f16x2(gh2, gh3),
                )

                uv0, uv1 = f16x2_to_f32x2(half2_mul(xh0, uw0))
                uv2, uv3 = f16x2_to_f32x2(half2_mul(xh1, uw1))
                uh0, uh1, uh2, uh3 = self.driver._had128_quad(uv0, uv1, uv2, uv3, lane)
                st_global_v2_u32(
                    get_ptr_as_int64(a_up, out_off),
                    pack_f32x2_to_f16x2(uh0, uh1),
                    pack_f32x2_to_f16x2(uh2, uh3),
                )
            unit += unit_stride

    @cute.jit
    def _activate_live_routes(
        self,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        global_to_combined: cute.Tensor,
        route_num_experts: Int32,
        total_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        lane = tid & Int32(31)
        warps_per_cta = Int32(8)
        unit = cta * warps_per_cta + (tid >> Int32(5))
        unit_stride = grid_x * warps_per_cta
        total_units = active_m * Int32(64)
        elem = lane * Int32(4)

        while unit < total_units:
            route = unit >> Int32(2)
            blk = unit & Int32(3)

            ex_lane0 = Int32(-1)
            if lane == Int32(0):
                ge = raw_topk_ids[route].to(Int32)
                if ge >= Int32(0) and ge < route_num_experts:
                    ce = global_to_combined[ge].to(Int32)
                    if ce >= Int32(0) and ce < total_experts:
                        ex_lane0 = ce
            expert = cute.arch.shuffle_sync(ex_lane0, Int32(0))

            if expert >= Int32(0):
                col0 = blk * Int32(128) + elem
                g_off = Int64(route) * Int64(1024) + Int64(col0)
                u_off = g_off + Int64(512)
                s_off = Int64(expert) * Int64(1536) + Int64(col0)

                gp01, gp23 = ld_global_nc_v2_u32(get_ptr_as_int64(fc1, g_off))
                up01, up23 = ld_global_nc_v2_u32(get_ptr_as_int64(fc1, u_off))
                svg01, svg23 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(intermediate_rotations, s_off)
                )
                svu01, svu23 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(intermediate_rotations, s_off + Int64(512))
                )
                sd01, sd23 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(intermediate_rotations, s_off + Int64(1024))
                )

                g0, g1 = f16x2_to_f32x2(gp01)
                g2, g3 = f16x2_to_f32x2(gp23)
                u0, u1 = f16x2_to_f32x2(up01)
                u2, u3 = f16x2_to_f32x2(up23)

                gh0, gh1, gh2, gh3 = self.driver._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self.driver._had128_quad(u0, u1, u2, u3, lane)

                svg0, svg1 = f16x2_to_f32x2(svg01)
                svg2, svg3 = f16x2_to_f32x2(svg23)
                svu0, svu1 = f16x2_to_f32x2(svu01)
                svu2, svu3 = f16x2_to_f32x2(svu23)
                sd0, sd1 = f16x2_to_f32x2(sd01)
                sd2, sd3 = f16x2_to_f32x2(sd23)

                ig0 = gh0 * svg0
                ig1 = gh1 * svg1
                ig2 = gh2 * svg2
                ig3 = gh3 * svg3
                iu0 = uh0 * svu0
                iu1 = uh1 * svu1
                iu2 = uh2 * svu2
                iu3 = uh3 * svu3

                beta = cutlass.Float32(SITU_DEFAULT_BETA)
                linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                a0 = (
                    beta
                    * cute.math.tanh(ig0 / beta, fastmath=self.driver.fast_math)
                    * self.driver._sigmoid_f32(ig0)
                    * linear_beta
                    * cute.math.tanh(iu0 / linear_beta, fastmath=self.driver.fast_math)
                    * sd0
                )
                a1 = (
                    beta
                    * cute.math.tanh(ig1 / beta, fastmath=self.driver.fast_math)
                    * self.driver._sigmoid_f32(ig1)
                    * linear_beta
                    * cute.math.tanh(iu1 / linear_beta, fastmath=self.driver.fast_math)
                    * sd1
                )
                a2 = (
                    beta
                    * cute.math.tanh(ig2 / beta, fastmath=self.driver.fast_math)
                    * self.driver._sigmoid_f32(ig2)
                    * linear_beta
                    * cute.math.tanh(iu2 / linear_beta, fastmath=self.driver.fast_math)
                    * sd2
                )
                a3 = (
                    beta
                    * cute.math.tanh(ig3 / beta, fastmath=self.driver.fast_math)
                    * self.driver._sigmoid_f32(ig3)
                    * linear_beta
                    * cute.math.tanh(iu3 / linear_beta, fastmath=self.driver.fast_math)
                    * sd3
                )
                o0, o1, o2, o3 = self.driver._had128_quad(a0, a1, a2, a3, lane)
                out_off = Int64(route) * Int64(512) + Int64(col0)
                st_global_v2_u32(
                    get_ptr_as_int64(activated, out_off),
                    pack_f32x2_to_f16x2(o0, o1),
                    pack_f32x2_to_f16x2(o2, o3),
                )
            unit += unit_stride

    @cute.jit
    def _build_m8_expert_spans(
        self,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
    ):
        span_cap = int(self.driver.max_m_blocks)
        iters = (span_cap + 255) // 256
        hdr_smem = smem_base + Int32(self.span_smem_off)
        out_smem = hdr_smem + Int32(16)
        # Temporary prologue scratch inside the not-yet-used slab region.
        raw_smem = smem_base + Int32(self.slab_offset)
        multi_smem = raw_smem + Int32(8192)
        single_smem = raw_smem + Int32(16384)

        raw_blocks = packed_route_count[Int32(0)].to(Int32) >> Int32(3)
        if raw_blocks < Int32(0):
            raw_blocks = Int32(0)
        if raw_blocks > Int32(span_cap):
            raw_blocks = Int32(span_cap)

        if tid == Int32(0):
            st_shared_i32(hdr_smem + Int32(0), raw_blocks)
            st_shared_i32(hdr_smem + Int32(4), Int32(0))
            st_shared_i32(raw_smem + raw_blocks * Int32(4), Int32(-2))

        for i in cutlass.range_constexpr(iters):
            idx = Int32(i * 256) + tid
            if idx < raw_blocks:
                e = block_expert_ids[idx].to(Int32)
                st_shared_i32(raw_smem + idx * Int32(4), e)
                st_shared_i32(out_smem + idx * Int32(4), idx << Int32(2))
        cute.arch.sync_threads()

        has_dup = Int32(0)
        for i in cutlass.range_constexpr(iters):
            idx = Int32(i * 256) + tid
            if idx + Int32(1) < raw_blocks:
                e0 = ld_shared_i32_relaxed(raw_smem + idx * Int32(4))
                e1 = ld_shared_i32_relaxed(raw_smem + (idx + Int32(1)) * Int32(4))
                if e0 >= Int32(0) and e0 == e1:
                    has_dup = Int32(1)
        if has_dup != Int32(0):
            st_shared_i32(hdr_smem + Int32(4), Int32(1))
        cute.arch.sync_threads()

        need_compact = ld_shared_i32_relaxed(hdr_smem + Int32(4))
        if need_compact != Int32(0):
            if tid == Int32(0):
                b = Int32(0)
                n_multi = Int32(0)
                n_single = Int32(0)
                while b < raw_blocks:
                    e0 = ld_shared_i32_relaxed(raw_smem + b * Int32(4))
                    slen = Int32(1)
                    if e0 >= Int32(0):
                        if b + Int32(1) < raw_blocks:
                            e1 = ld_shared_i32_relaxed(raw_smem + (b + Int32(1)) * Int32(4))
                            if e1 == e0:
                                slen = Int32(2)
                                if b + Int32(2) < raw_blocks:
                                    e2 = ld_shared_i32_relaxed(
                                        raw_smem + (b + Int32(2)) * Int32(4)
                                    )
                                    if e2 == e0:
                                        slen = Int32(3)
                                        if b + Int32(3) < raw_blocks:
                                            e3 = ld_shared_i32_relaxed(
                                                raw_smem + (b + Int32(3)) * Int32(4)
                                            )
                                            if e3 == e0:
                                                slen = Int32(4)
                    entry = (b << Int32(2)) | (slen - Int32(1))
                    if slen >= Int32(2):
                        st_shared_i32(multi_smem + n_multi * Int32(4), entry)
                        n_multi += Int32(1)
                    else:
                        st_shared_i32(single_smem + n_single * Int32(4), entry)
                        n_single += Int32(1)
                    b += slen
                st_shared_i32(hdr_smem + Int32(0), n_multi + n_single)
                st_shared_i32(hdr_smem + Int32(8), n_multi)
            cute.arch.sync_threads()

            n_total = ld_shared_i32_relaxed(hdr_smem + Int32(0))
            n_multi = ld_shared_i32_relaxed(hdr_smem + Int32(8))
            for i in cutlass.range_constexpr(iters):
                idx = Int32(i * 256) + tid
                if idx < n_total:
                    val = Int32(0)
                    if idx < n_multi:
                        val = ld_shared_i32_relaxed(multi_smem + idx * Int32(4))
                    else:
                        val = ld_shared_i32_relaxed(
                            single_smem + (idx - n_multi) * Int32(4)
                        )
                    st_shared_i32(out_smem + idx * Int32(4), val)
            cute.arch.sync_threads()

    @cute.jit
    def _fetch_k2_slab(
        self,
        is_fc1: cutlass.Constexpr,
        span_kind: cutlass.Constexpr[int],
        stage_base: Int32,
        b_src_base: Int64,
        a_src_bases,
        row_valid,
        a_swz_off: Int32,
        slab: Int32,
        tid: Int32,
        lane: Int32,
        b_l2_policy: Uint64,
    ):
        b_row_bytes = 2048 if is_fc1 else 14336
        # Materialize the per-thread shared B base through shuffle_sync so ptxas
        # keeps [R + imm] on LDGSTS with an L2 cache-hint descriptor in UR.
        b_dst_tid = cute.arch.shuffle_sync(stage_base + tid * Int32(16), lane)
        b_slab_src = b_src_base + Int64(slab) * Int64(16 * b_row_bytes)
        for i in cutlass.range_constexpr(4):
            b_src = b_slab_src + Int64(i * 4 * b_row_bytes)
            b_dst = b_dst_tid + Int32(i * 4096)
            if cutlass.const_expr(self.tier0.fc1.l2_evict_first_b):
                _cp_async4_shared_global_l2hint(b_dst, b_src, b_l2_policy)
            else:
                cp_async4_shared_global(b_dst, b_src)

        a_slab_byte_off = Int64(slab) * Int64(512)
        for b_idx in cutlass.range_constexpr(span_kind):
            a_dst = stage_base + Int32(16384 + b_idx * 4096) + a_swz_off
            if row_valid[b_idx] != Int32(0):
                cp_async4_shared_global(a_dst, a_src_bases[b_idx] + a_slab_byte_off)
            else:
                st_shared_v4_u32(
                    a_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                )

        cute.arch.cp_async_commit_group()

    @cute.jit
    def _consume_k2_slab(
        self,
        span_kind: cutlass.Constexpr[int],
        cur_stage_base: Int32,
        b_rd_warp_off: Int32,
        a_rd_off0: Int32,
        a_rd_off1: Int32,
        prev_lane: Int32,
        sel_a: Uint32,
        sel_b: Uint32,
        acc0,
        acc1,
        acc2,
        acc3,
    ):
        b_stage = cur_stage_base + b_rd_warp_off
        a_stage = cur_stage_base + Int32(16384)
        for sub in cutlass.range_constexpr(4):
            for kk in cutlass.range_constexpr(2):
                a_rd_off = a_rd_off0 if cutlass.const_expr(kk == 0) else a_rd_off1
                a0_0, a0_1 = ldmatrix_m8n8x2_b16(
                    a_stage + a_rd_off + Int32(sub * 128)
                )
                a1_0 = Uint32(0)
                a1_1 = Uint32(0)
                a2_0 = Uint32(0)
                a2_1 = Uint32(0)
                a3_0 = Uint32(0)
                a3_1 = Uint32(0)
                if cutlass.const_expr(span_kind >= 2):
                    a1_0, a1_1 = ldmatrix_m8n8x2_b16(
                        a_stage + a_rd_off + Int32(4096 + sub * 128)
                    )
                if cutlass.const_expr(span_kind >= 4):
                    a2_0, a2_1 = ldmatrix_m8n8x2_b16(
                        a_stage + a_rd_off + Int32(8192 + sub * 128)
                    )
                    a3_0, a3_1 = ldmatrix_m8n8x2_b16(
                        a_stage + a_rd_off + Int32(12288 + sub * 128)
                    )

                b_sub_off = b_stage + Int32(sub * 4096 + kk * 1024)
                bw0 = ld_shared_u32(b_sub_off + Int32(0))
                bw1 = ld_shared_u32(b_sub_off + Int32(64))
                bw2 = ld_shared_u32(b_sub_off + Int32(128))
                bw3 = ld_shared_u32(b_sub_off + Int32(192))

                for jj in cutlass.range_constexpr(4):
                    bw = bw0
                    if cutlass.const_expr(jj == 1):
                        bw = bw1
                    elif cutlass.const_expr(jj == 2):
                        bw = bw2
                    elif cutlass.const_expr(jj == 3):
                        bw = bw3
                    aw = cute.arch.shuffle_sync(bw, prev_lane)
                    win_a = _prmt_b32_reg(bw, aw, sel_a)
                    win_b = _prmt_b32_reg(bw, aw, sel_b)
                    if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                        o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                            win_a, win_b, 2
                        )
                    else:
                        o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                            win_a, win_b, 2
                        )

                    d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                        acc0[jj * 4 + 0][0],
                        acc0[jj * 4 + 1][0],
                        acc0[jj * 4 + 2][0],
                        acc0[jj * 4 + 3][0],
                        o0,
                        o2,
                        o1,
                        o3,
                        a0_0,
                        a0_1,
                    )
                    acc0[jj * 4 + 0][0] = d0
                    acc0[jj * 4 + 1][0] = d1
                    acc0[jj * 4 + 2][0] = d2
                    acc0[jj * 4 + 3][0] = d3

                    if cutlass.const_expr(span_kind >= 2):
                        d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                            acc1[jj * 4 + 0][0],
                            acc1[jj * 4 + 1][0],
                            acc1[jj * 4 + 2][0],
                            acc1[jj * 4 + 3][0],
                            o0,
                            o2,
                            o1,
                            o3,
                            a1_0,
                            a1_1,
                        )
                        acc1[jj * 4 + 0][0] = d0
                        acc1[jj * 4 + 1][0] = d1
                        acc1[jj * 4 + 2][0] = d2
                        acc1[jj * 4 + 3][0] = d3

                    if cutlass.const_expr(span_kind >= 4):
                        d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                            acc2[jj * 4 + 0][0],
                            acc2[jj * 4 + 1][0],
                            acc2[jj * 4 + 2][0],
                            acc2[jj * 4 + 3][0],
                            o0,
                            o2,
                            o1,
                            o3,
                            a2_0,
                            a2_1,
                        )
                        acc2[jj * 4 + 0][0] = d0
                        acc2[jj * 4 + 1][0] = d1
                        acc2[jj * 4 + 2][0] = d2
                        acc2[jj * 4 + 3][0] = d3

                        d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                            acc3[jj * 4 + 0][0],
                            acc3[jj * 4 + 1][0],
                            acc3[jj * 4 + 2][0],
                            acc3[jj * 4 + 3][0],
                            o0,
                            o2,
                            o1,
                            o3,
                            a3_0,
                            a3_1,
                        )
                        acc3[jj * 4 + 0][0] = d0
                        acc3[jj * 4 + 1][0] = d1
                        acc3[jj * 4 + 2][0] = d2
                        acc3[jj * 4 + 3][0] = d3

    @cute.jit
    def _tile_body_k2(
        self,
        is_fc1: cutlass.Constexpr,
        span_kind: cutlass.Constexpr[int],
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        start_blk: Int32,
        span_len: Int32,
        local_expert: Int32,
        output_n_tile: Int32,
        active_size_m: Int32,
    ):
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id >> Int32(2)
        w_n = warp_id & Int32(3)

        live_routes = (
            active_size_m * Int32(16)
            if cutlass.const_expr(is_fc1)
            else active_size_m
        )
        global_scale_f32 = t0_global_scale[local_expert].to(cutlass.Float32)

        a_stride_bytes = 7168 if is_fc1 else 1024
        a_tensor_base = get_ptr_as_int64(a_flat, Int64(0))
        if cutlass.const_expr(is_fc1):
            if output_n_tile >= Int32(2):
                a_tensor_base = get_ptr_as_int64(a_alt_flat, Int64(0))

        safe_r = []
        row_valid = []
        a_src_bases = []
        for b_idx in cutlass.range_constexpr(span_kind):
            blk_valid = Int32(1)
            if cutlass.const_expr(span_kind == 4):
                blk_valid = (Int32(b_idx) < span_len).to(Int32)
            r_lane = live_routes
            if blk_valid != Int32(0) and lane < Int32(8):
                r_lane = packed_route_indices[
                    (start_blk + Int32(b_idx)) * Int32(8) + lane
                ].to(Int32)
            r_w = cute.arch.shuffle_sync(r_lane, warp_id)
            v_w = Int32(0)
            safe_r_w = Int32(0)
            if blk_valid != Int32(0) and r_w >= Int32(0) and r_w < live_routes:
                v_w = Int32(1)
                safe_r_w = r_w
            safe_r.append(safe_r_w)
            row_valid.append(v_w)
            a_src_bases.append(
                a_tensor_base
                + Int64(safe_r_w) * Int64(a_stride_bytes)
                + Int64(lane * Int32(16))
            )

        if cutlass.const_expr(is_fc1):
            proj = (output_n_tile >= Int32(2)).to(Int32)
            local_n_tile = output_n_tile - proj * Int32(2)
            plane_u32 = Int64(cute.size(t0_b_flat)) // Int64(2)
            b_tile_u32 = (
                Int64(proj) * plane_u32
                + Int64(local_expert) * Int64(114688)
                + Int64(local_n_tile) * Int64(256)
            )
            b_row_u32 = 512
            slabs = 14
        else:
            b_tile_u32 = (
                Int64(local_expert) * Int64(114688)
                + Int64(output_n_tile) * Int64(256)
            )
            b_row_u32 = 3584
            slabs = 2

        b_src_base = get_ptr_as_int64(
            t0_b_flat,
            b_tile_u32
            + Int64(tid >> Int32(6)) * Int64(b_row_u32)
            + Int64((tid & Int32(63)) << Int32(2)),
        )

        a_swz_off = ((warp_id << Int32(5)) + (lane ^ warp_id)) << Int32(4)
        l16 = lane & Int32(15)
        r_ldm = l16 & Int32(7)
        c0_ldm = l16 >> Int32(3)
        k_base = (warp_row << Int32(2)) + c0_ldm
        a_rd_off0 = ((r_ldm << Int32(5)) + (k_base ^ r_ldm)) << Int32(4)
        a_rd_off1 = ((r_ldm << Int32(5)) + ((k_base + Int32(2)) ^ r_ldm)) << Int32(4)
        b_rd_warp_off = (
            (warp_row * Int32(2048))
            + (w_n * Int32(256))
            + ((lane >> Int32(1)) * Int32(4))
        )
        prev_lane = (lane + Int32(31)) & Int32(31)
        sel_a = Uint32(0x5432)
        sel_b = Uint32(0x6543)
        if (lane & Int32(1)) != Int32(0):
            sel_a = Uint32(0x3210)
            sel_b = Uint32(0x4321)

        b_l2_policy = Uint64(0)
        if cutlass.const_expr(self.tier0.fc1.l2_evict_first_b):
            b_l2_policy = create_l2_evict_first_policy()

        acc0 = [cute.make_rmem_tensor((1,), cutlass.Float32) for _ in range(16)]
        for idx in cutlass.range_constexpr(16):
            acc0[idx].fill(0.0)
        acc1 = acc0
        acc2 = acc0
        acc3 = acc0
        if cutlass.const_expr(span_kind >= 2):
            acc1 = [cute.make_rmem_tensor((1,), cutlass.Float32) for _ in range(16)]
            for idx in cutlass.range_constexpr(16):
                acc1[idx].fill(0.0)
        if cutlass.const_expr(span_kind >= 4):
            acc2 = [cute.make_rmem_tensor((1,), cutlass.Float32) for _ in range(16)]
            acc3 = [cute.make_rmem_tensor((1,), cutlass.Float32) for _ in range(16)]
            for idx in cutlass.range_constexpr(16):
                acc2[idx].fill(0.0)
                acc3[idx].fill(0.0)

        stage_bytes = self.B_SLAB_BYTES + span_kind * self.A_BLOCK_SLAB_BYTES
        slab0_base = smem_base + Int32(self.slab_offset)
        slab1_base = slab0_base + Int32(stage_bytes)

        self._fetch_k2_slab(
            is_fc1,
            span_kind,
            slab0_base,
            b_src_base,
            a_src_bases,
            row_valid,
            a_swz_off,
            Int32(0),
            tid,
            lane,
            b_l2_policy,
        )

        for slab in cutlass.range(slabs, unroll=1):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            cur_stage_base = slab0_base
            other_stage_base = slab1_base
            if (slab & Int32(1)) != Int32(0):
                cur_stage_base = slab1_base
                other_stage_base = slab0_base

            if slab + Int32(1) < Int32(slabs):
                self._fetch_k2_slab(
                    is_fc1,
                    span_kind,
                    other_stage_base,
                    b_src_base,
                    a_src_bases,
                    row_valid,
                    a_swz_off,
                    slab + Int32(1),
                    tid,
                    lane,
                    b_l2_policy,
                )

            self._consume_k2_slab(
                span_kind,
                cur_stage_base,
                b_rd_warp_off,
                a_rd_off0,
                a_rd_off1,
                prev_lane,
                sel_a,
                sel_b,
                acc0,
                acc1,
                acc2,
                acc3,
            )

        # Both FC1 (14 slabs) and FC2 (2 slabs) finish on odd slab index ->
        # stage 1, so stage 0 [slab0_base, slab0_base + span_kind*8192) is
        # immediately free for the warps 4..7 partial-sum fold without an
        # extra barrier.
        fold_base = slab0_base
        out_base = fold_base + Int32(span_kind * 8192)
        accs = (acc0, acc1, acc2, acc3)

        if tid >= Int32(128):
            t128_off = (tid - Int32(128)) * Int32(16)
            for b_idx in cutlass.range_constexpr(span_kind):
                acc_b = accs[b_idx]
                for jj in cutlass.range_constexpr(4):
                    st_shared_v4_f32(
                        fold_base + Int32(b_idx * 8192 + jj * 2048) + t128_off,
                        acc_b[jj * 4 + 0][0],
                        acc_b[jj * 4 + 1][0],
                        acc_b[jj * 4 + 2][0],
                        acc_b[jj * 4 + 3][0],
                    )
        cute.arch.sync_threads()

        if tid < Int32(128):
            r0_ep = (lane & Int32(3)) << Int32(1)
            col0_ep = (warp_id << Int32(6)) + (lane >> Int32(2))
            wr_byte_base = (r0_ep * Int32(264) + col0_ep) << Int32(1)
            t128_off = tid * Int32(16)
            for b_idx in cutlass.range_constexpr(span_kind):
                acc_b = accs[b_idx]
                for jj in cutlass.range_constexpr(4):
                    f0, f1, f2, f3 = ld_shared_v4_f32(
                        fold_base + Int32(b_idx * 8192 + jj * 2048) + t128_off
                    )
                    dst_ep = out_base + Int32(b_idx * 4224 + jj * 32) + wr_byte_base
                    st_shared_f16_from_f32(
                        dst_ep,
                        (acc_b[jj * 4 + 0][0] + f0) * global_scale_f32,
                    )
                    st_shared_f16_from_f32(
                        dst_ep + Int32(528),
                        (acc_b[jj * 4 + 1][0] + f1) * global_scale_f32,
                    )
                    st_shared_f16_from_f32(
                        dst_ep + Int32(16),
                        (acc_b[jj * 4 + 2][0] + f2) * global_scale_f32,
                    )
                    st_shared_f16_from_f32(
                        dst_ep + Int32(544),
                        (acc_b[jj * 4 + 3][0] + f3) * global_scale_f32,
                    )
        cute.arch.sync_threads()

        c_stride_elem = 1024 if is_fc1 else 3584
        col_elem = output_n_tile * Int32(256) + lane * Int32(8)
        rd_ep_off = warp_id * Int32(528) + lane * Int32(16)
        for b_idx in cutlass.range_constexpr(span_kind):
            if row_valid[b_idx] != Int32(0):
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    out_base + Int32(b_idx * 4224) + rd_ep_off
                )
                true_elem = (
                    Int64(safe_r[b_idx]) * Int64(c_stride_elem) + Int64(col_elem)
                )
                st_global_v4_u32(
                    get_ptr_as_int64(c_flat, true_elem), q0, q1, q2, q3
                )
        cute.arch.sync_threads()

    @cute.jit
    def _emit_tier_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_scales_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_scales_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        if cutlass.const_expr(self.max_span == 1):
            start_blk = route_block_idx
            span_len = Int32(1)
        else:
            entry = ld_shared_i32_relaxed(
                smem_base
                + Int32(self.span_smem_off + 16)
                + route_block_idx * Int32(4)
            )
            start_blk = entry >> Int32(2)
            span_len = (entry & Int32(3)) + Int32(1)

        combined_expert = block_expert_ids[start_blk].to(Int32)
        total_experts = tier0_num_experts + tier1_num_experts
        descriptor_row = Int32(2)
        if cutlass.const_expr(is_fc1):
            fc1_half_tiles = Int32(self.driver.fc1.n_tiles // 2)
            descriptor_row = Int32(0)
            if output_n_tile >= fc1_half_tiles:
                descriptor_row = Int32(1)

        if combined_expert >= Int32(0) and combined_expert < total_experts:
            descriptor = descriptor_map[
                descriptor_row * total_experts + combined_expert
            ].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(self.descriptor_bits)
                local_expert = descriptor & Int32((1 << self.descriptor_bits) - 1)

                if cutlass.const_expr(is_fc1):
                    tier0_in_bounds = local_expert < tier0_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        tier0_in_bounds = local_expert < tier0_up_experts
                else:
                    tier0_in_bounds = local_expert < tier0_fc2_experts

                if tier == Int32(0) and tier0_in_bounds:
                    if cutlass.const_expr(self.max_span == 1):
                        self._tile_body_k2(
                            is_fc1,
                            1,
                            a_flat,
                            a_alt_flat,
                            t0_b_flat,
                            t0_global_scale,
                            c_flat,
                            packed_route_indices,
                            smem_base,
                            tid,
                            start_blk,
                            span_len,
                            local_expert,
                            output_n_tile,
                            active_size_m,
                        )
                    else:
                        if span_len == Int32(1):
                            self._tile_body_k2(
                                is_fc1,
                                1,
                                a_flat,
                                a_alt_flat,
                                t0_b_flat,
                                t0_global_scale,
                                c_flat,
                                packed_route_indices,
                                smem_base,
                                tid,
                                start_blk,
                                span_len,
                                local_expert,
                                output_n_tile,
                                active_size_m,
                            )
                        elif span_len == Int32(2):
                            self._tile_body_k2(
                                is_fc1,
                                2,
                                a_flat,
                                a_alt_flat,
                                t0_b_flat,
                                t0_global_scale,
                                c_flat,
                                packed_route_indices,
                                smem_base,
                                tid,
                                start_blk,
                                span_len,
                                local_expert,
                                output_n_tile,
                                active_size_m,
                            )
                        else:
                            self._tile_body_k2(
                                is_fc1,
                                4,
                                a_flat,
                                a_alt_flat,
                                t0_b_flat,
                                t0_global_scale,
                                c_flat,
                                packed_route_indices,
                                smem_base,
                                tid,
                                start_blk,
                                span_len,
                                local_expert,
                                output_n_tile,
                                active_size_m,
                            )

                if cutlass.const_expr(is_fc1):
                    tier1_in_bounds = local_expert < tier1_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        tier1_in_bounds = local_expert < tier1_up_experts
                else:
                    tier1_in_bounds = local_expert < tier1_fc2_experts

                if tier == Int32(1) and tier1_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier1.fc1
                    else:
                        gemm = self.tier1.fc2
                    smem_base_k3 = cute.arch.shuffle_sync(
                        smem_base, tid & Int32(31)
                    )
                    b_off = Int32(0)
                    while b_off < span_len:
                        self._dispatch_bound(
                            is_fc1,
                            combined_expert,
                            gemm,
                            None,
                            a_flat,
                            a_alt_flat,
                            t1_b_flat,
                            c_flat,
                            t1_scales_flat,
                            t1_global_scale,
                            packed_route_indices,
                            topk_weights,
                            c_tmp,
                            locks,
                            trellis_lut_addr,
                            smem_base_k3,
                            tid,
                            start_blk + b_off,
                            local_expert,
                            output_n_tile,
                            reduce_k_tile,
                            reduce_tile_count,
                            reduce_slice_count,
                            reduce_slice_idx,
                            lock_slot,
                            active_size_m,
                        )
                        b_off += Int32(1)

    @cute.kernel
    def kernel(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_scales: cute.Tensor,
        t0_w2_scales: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_scales: cute.Tensor,
        t1_w2_scales: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        global_to_combined: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut: cute.Tensor,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        trellis_lut_addr = get_ptr_as_int64(trellis_lut, Int32(0))
        phase_lut_addr = trellis_lut_addr
        if cutlass.const_expr(self.driver.sqg_xor_cheb_t12_smem):
            self.driver._sqg_smem_copy(
                trellis_lut_addr,
                smem_base + Int32(self.driver.sqg_xor_cheb_t12_smem_off),
                _SQG_XOR_CHEB_T12_SMEM_REGION_BYTES,
                tid,
            )
            cute.arch.sync_threads()
            phase_lut_addr = Int64(
                smem_base + Int32(self.driver.sqg_xor_cheb_t12_smem_off)
            )

        total_experts = tier0_num_experts + tier1_num_experts
        self._rotate_live_routes(
            rotation_input,
            rotation_gate,
            rotation_up,
            gate_suh,
            up_suh,
            raw_topk_ids,
            global_to_combined,
            route_num_experts,
            total_experts,
            tid,
            cta,
            grid_x,
            active_m,
        )
        if cutlass.const_expr(self.max_span > 1):
            self._build_m8_expert_spans(
                block_expert_ids,
                packed_route_count,
                smem_base,
                tid,
            )
        self.driver._grid_barrier(workspace, tid, grid_x)

        fc1_emit = partial(
            self._emit_tier_tile,
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_scales,
            t0_w13_global,
            t1_w13,
            t1_w13_scales,
            t1_w13_global,
            fc1,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )
        fc2_emit = partial(
            self._emit_tier_tile,
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_scales,
            t0_w2_global,
            t1_w2,
            t1_w2_scales,
            t1_w2_global,
            fc2,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
            fc2_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )
        act_hook = partial(
            self._activate_live_routes,
            fc1,
            activated,
            intermediate_rotations,
            raw_topk_ids,
            global_to_combined,
            route_num_experts,
            total_experts,
            tid,
            cta,
            grid_x,
            active_m,
        )
        self.driver._moe_body(
            rotation_gate,
            rotation_up,
            rotation_input,
            t0_w13,
            t0_w2,
            fc1,
            activated,
            fc2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            t0_w13_global,
            Int32(0),
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            descriptor_map,
            trellis_lut_addr,
            trellis_lut_addr,
            total_experts,
            total_experts,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit,
            fc2_emit,
            input_rotated=True,
            fc1_prefetch_tile=None,
            fc2_prefetch_tile=None,
            span_count_smem_off=(self.span_smem_off if self.max_span > 1 else None),
            reverse_tail_wave=(self.max_span > 1),
            activation_hook=act_hook,
        )
