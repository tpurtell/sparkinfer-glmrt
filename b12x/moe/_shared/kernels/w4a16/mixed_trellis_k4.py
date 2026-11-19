"""SM121 (GB10) EXL3 K4 small-M non-cooperative mixed-Trellis decode/verify kernel.

Targets the DGX Spark (SM121, GB10) cooperative decode/verify path (``size_m <= 80``,
``moe_block_size == 8``, ``tier0_bits == 4``, MCG codebook) for both compact and
contiguous weight layouts across all gated activations (``silu``,
``silu_post_clamp``, ``situ``).

Structural design (bit-identical to ``W4A16MixedTrellisKernel`` and invariant
across ``m1``, ``m8``, and ``m80``):

1. Non-cooperative phase launches with Blackwell PDL (``use_pdl=True``):
   replaces the single cooperative mega-kernel and its three global-memory
   spinning grid barriers with lightweight non-cooperative launches on the exact
   ``v41_exl3_core`` ABI. Primary kernels signal ``griddepcontrol_launch_dependents``
   so dependent GEMMs immediately occupy idle SMs during tail waves, resolve
   read-only metadata, and prefetch their first two weight slabs from LPDDR5X
   into SMEM before blocking on ``griddepcontrol_wait``.
2. Fused last-CTA-activates inside FC1 (eliminating ``act_kernel`` whenever
   ``max_live_blocks <= 4 * sms`` from runtime ``grid_x``, covering ``m1..m16``):
   each active FC1 CTA for route block ``rb`` writes its output tile and
   atomically increments ``workspace[rb]`` with ``atom.acq_rel.gpu``; the final
   CTA for ``rb`` resets ``workspace[rb] = 0`` and immediately executes the
   vectorized H128 + SwiGLU/SiLU/SiTU + H128 activation for ``rb`` straight out
   of L2 using L2-coherent ``ld.global.cg.v2.u32``, cutting the small-M core to
   just two overlapped launches (``fc1+rot+act`` -> PDL -> ``fc2``).
3. Wave-matched FC1 tiling and unified 2-stage FC2 tiling:
   - Narrow waves (``active_m <= 3``): ``fc1_tile_n = 32`` when
     ``intermediate_size <= 384`` (``64`` when ``>= 512``) with 3-stage K128
     slabs and fused H128 BF16->FP16 input rotation (6 CTAs/SM, 128-154 live
     CTAs at ``m = 1``).
   - Small-M waves (``4 <= active_m <= 16``): ``fc1_tile_n = 64`` with 3-stage
     K128 slabs and fused input rotation, halving CTA count, fused rotation
     work, ``ldmatrix`` count, and atomic increments while doubling contiguous
     DRAM burst width (512 B/row).
   - Medium-M waves (``active_m > 16``, e.g. ``m64``/``m80``): vectorized
     live-route input rotation -> PDL -> ``fc1_tile_n = 128`` (1024 B/row,
     3-stage) with adaptive L2 cache policy for multi-block expert reuse.
   - Unified FC2 (``m1..m80``): ``fc2_tile_n = 128`` with 2-stage double-buffered
     K128 slabs (3 CTAs/SM = 12 warps/SM even at ``max_bits = 6``, 1024 B/row,
     exact multiples of 48 at ``m = 1`` for H=4608 and H=6144), prefetching both
     initial weight slabs (100% of K=256 weights) under PDL before waiting on
     ``activated``.
4. Exact K-reduction and hoisted 64-bit addressing:
   - Every GEMM preserves the exact 2-chain K-reduction order
     (``K16 % 4 in {0, 1}`` and ``{2, 3}``) of the production K64 kernel, uses
     hoisted 64-bit B and A pointers outside the K-loop, 1-cycle ``prmt.b32``
     K4 ring extraction, a zero-barrier warp-register prologue, a 2-barrier
     disjoint-SMEM epilogue, and adaptive per-block L2 cache policy.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, Uint64, dsl_user_op

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
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
    packed_dequant_trellis_stream_to_half2x4,
    packed_dequant_trellis_to_half2x4,
    shared_ptr_to_u32,
    st_global_v2_u32,
    st_global_v4_u32,
    st_shared_f16_from_f32,
    st_shared_i32,
    st_shared_v4_f32,
    st_shared_v4_u32,
    trellis_align_stream_u32x2,
)
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits

from .kernel import W4A16FusedMoeKernel
from .mixed_trellis import W4A16MixedTrellisKernel


_K4_NARROW_MAX_ROWS = 3
_K4_SMALLM_MAX_ROWS = 16
_FC1_SLAB_STAGES = 3
_FC2_SLAB_STAGES = 2
_A_SLAB_BYTES = 2048  # 8 rows x 128 FP16 = 2048 B


@dsl_user_op
def _cp_async4_l2hint(
    smem_addr: Int32,
    gmem_addr: Int64,
    cache_policy: Uint64,
    *,
    loc=None,
    ip=None,
):
    """Unconditional 16-byte ``cp.async.cg`` with L2 cache policy."""
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
def _cp_async4_pred_l2hint(
    smem_addr: Int32,
    gmem_addr: Int64,
    pred: Int32,
    cache_policy: Uint64,
    *,
    loc=None,
    ip=None,
):
    """Predicated 16-byte ``cp.async.cg`` with L2 cache policy."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Uint64(cache_policy).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; "
        "@p cp.async.cg.shared.global.L2::cache_hint [$1], [$2], 16, $3; }",
        "r,r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_v2_u32(
    smem_addr: Int32,
    v0: Uint32,
    v1: Uint32,
    *,
    loc=None,
    ip=None,
):
    """8-byte shared-memory store."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prmt_b32_5432(
    b: Uint32,
    a: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Compute ``(a << 16) | (b >> 16)`` in one ``prmt.b32`` instruction."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(a).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, 0x5432;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_cg_v2_u32(
    base_ptr: Int64,
    *,
    loc=None,
    ip=None,
):
    """L2-coherent 64-bit global load (bypasses L1 for cross-CTA visibility)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
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
def _atomic_add_acq_rel_gpu_i32(
    addr: Int64,
    val: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """GPU-scoped acquire-release global int32 atomic add."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_relaxed_gpu_i32(
    addr: Int64,
    val: Int32,
    *,
    loc=None,
    ip=None,
):
    """GPU-scoped relaxed int32 global store (resets block completion counter)."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.gpu.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _k4_smem_layout(
    tile_n: int,
    max_bits: int,
    stages: int,
) -> tuple[int, int, int, int]:
    """Return ``(b_stage_bytes, a_off, fold_off, total_words)`` for one GEMM kernel.

    Output transpose aliases offset 0 after epilogue barrier 1 (the K-loop has
    completely finished), while ``fold_off`` sits strictly disjoint past the A
    ring so warp_row 1 never races with warp_row 0's final K-loop SMEM reads.
    """
    cta_n_blocks = tile_n // 16
    b_stage_bytes = 8 * cta_n_blocks * 32 * max_bits
    a_off = stages * b_stage_bytes
    fold_off = a_off + stages * _A_SLAB_BYTES
    tb_n_warps = 2 if tile_n == 128 else 1
    n_frags = 2 if tile_n == 32 else 4
    fold_bytes = tb_n_warps * 32 * n_frags * 16
    out_bytes = 8 * (tile_n + 8) * 2
    total_bytes = max(fold_off + fold_bytes, out_bytes)
    total_bytes = ((total_bytes + 15) // 16) * 16
    return b_stage_bytes, a_off, fold_off, total_bytes // 4


class W4A16MixedTrellisK4Decode(W4A16MixedTrellisKernel):
    """SM121 (GB10) non-cooperative K4 mixed-Trellis decode/verify kernel."""

    K4_DECODE_VERSION = 2

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        descriptor_bits: int = 9,
        compact_weights: bool = False,
    ):
        super().__init__(
            driver=driver,
            tier0=tier0,
            tier1=tier1,
            descriptor_bits=descriptor_bits,
            paired_boundary=None,
            compact_weights=compact_weights,
            token_major_rotation=False,
        )
        if (
            self.tier0.trellis_bits != 4
            or self.tier0.trellis_codebook != "mcg"
            or self.driver.moe_block_size != 8
            or self.driver.fc2.moe_block_size != 8
            or self.driver.fc2.schedule_route_block_factor != 1
            or self.driver.direct_topk_routes
            or self.driver.fused_input_rotation
            or self.driver.broadcast_suh
            or self.driver.coupled_hadamard
            or self.driver.rotation_input_dtype != "bf16"
            or self.driver.fc1.tile_k != 64
            or self.driver.fc2.tile_k != 64
            or self.hidden_size % 128 != 0
            or self.intermediate_size % 128 != 0
        ):
            raise ValueError(
                "W4A16MixedTrellisK4Decode requires SM121 MCG K4 cooperative m8 geometry"
            )
        self.max_bits = max(int(self.tier0.trellis_bits), int(self.tier1.trellis_bits))
        self.narrow_fc1_tile_n = 32 if self.intermediate_size <= 384 else 64
        self.smallm_fc1_tile_n = 64
        self.medm_fc1_tile_n = 128
        self.fc2_tile_n = 128
        self.has_smallm_n64 = (
            int(self.size_m) > _K4_NARROW_MAX_ROWS
            and self.narrow_fc1_tile_n != self.smallm_fc1_tile_n
        )
        self.has_medm_arm = int(self.size_m) > _K4_SMALLM_MAX_ROWS

        max_words = 0
        specs = [
            (self.narrow_fc1_tile_n, _FC1_SLAB_STAGES),
            (self.fc2_tile_n, _FC2_SLAB_STAGES),
        ]
        if self.has_smallm_n64:
            specs.append((self.smallm_fc1_tile_n, _FC1_SLAB_STAGES))
        if self.has_medm_arm:
            specs.append((self.medm_fc1_tile_n, _FC1_SLAB_STAGES))
        for tile_n, stages in specs:
            _, _, _, words = _k4_smem_layout(tile_n, self.max_bits, stages)
            if words > max_words:
                max_words = words
        self.shared_words = max_words

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return super().__cache_key__ + (
            (
                "sm121_k4_decode_noncoop_pdl",
                self.K4_DECODE_VERSION,
                self.narrow_fc1_tile_n,
                self.smallm_fc1_tile_n,
                self.medm_fc1_tile_n,
                self.fc2_tile_n,
                self.has_smallm_n64,
                self.has_medm_arm,
            ),
        )

    # ------------------------------------------------------------------
    # Host launch entry point (identical ABI to W4A16MixedTrellisKernel)
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        rotation_input_ptr: cute.Pointer,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13_ptr: cute.Pointer,
        t0_w2_ptr: cute.Pointer,
        t0_w13_scales_ptr: cute.Pointer,
        t0_w2_scales_ptr: cute.Pointer,
        t0_w13_global_ptr: cute.Pointer,
        t0_w2_global_ptr: cute.Pointer,
        t1_w13_ptr: cute.Pointer,
        t1_w2_ptr: cute.Pointer,
        t1_w13_scales_ptr: cute.Pointer,
        t1_w2_scales_ptr: cute.Pointer,
        t1_w13_global_ptr: cute.Pointer,
        t1_w2_global_ptr: cute.Pointer,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map_ptr: cute.Pointer,
        global_to_combined_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations_ptr: cute.Pointer,
        gate_suh_ptr: cute.Pointer,
        up_suh_ptr: cute.Pointer,
        trellis_lut_ptr: cute.Pointer,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        total_experts = tier0_experts + tier1_experts

        h16 = cutlass.Int64(self.hidden_size // 16)
        fc1_n16 = cutlass.Int64(self.driver.fc1_cols // 16)
        i16 = cutlass.Int64(self.intermediate_size // 16)
        b0 = cutlass.Int64(8 * self.tier0.trellis_bits)
        b1 = cutlass.Int64(8 * self.tier1.trellis_bits)

        def flat(ptr, n):
            return cute.make_tensor(ptr, layout=cute.make_layout((n,), stride=(1,)))

        if cutlass.const_expr(self.compact_weights):
            t0_w13 = flat(t0_w13_ptr, cutlass.Int64(1))
            t1_w13 = flat(t1_w13_ptr, cutlass.Int64(1))
            t0_w2 = flat(t0_w2_ptr, cutlass.Int64(1))
            t1_w2 = flat(t1_w2_ptr, cutlass.Int64(1))
        else:
            t0_w13 = flat(t0_w13_ptr, tier0_gate * h16 * fc1_n16 * b0)
            t1_w13 = flat(t1_w13_ptr, tier1_gate * h16 * fc1_n16 * b1)
            t0_w2 = flat(t0_w2_ptr, tier0_fc2 * i16 * h16 * b0)
            t1_w2 = flat(t1_w2_ptr, tier1_fc2 * i16 * h16 * b1)

        t0_w13_global = flat(t0_w13_global_ptr, tier0_experts)
        t1_w13_global = flat(t1_w13_global_ptr, tier1_experts)
        t0_w2_global = flat(t0_w2_global_ptr, tier0_fc2)
        t1_w2_global = flat(t1_w2_global_ptr, tier1_fc2)
        descriptor_map = flat(
            descriptor_map_ptr,
            cutlass.Int64(self.COMPACT_DESCRIPTOR_ROWS if self.compact_weights else 3)
            * total_experts,
        )
        global_to_combined = flat(
            global_to_combined_ptr, route_num_experts.to(cutlass.Int64)
        )
        intermediate_rotations = flat(
            intermediate_rotations_ptr,
            total_experts * cutlass.Int64(3 * self.intermediate_size),
        )
        gate_suh = flat(gate_suh_ptr, total_experts * cutlass.Int64(self.hidden_size))
        up_suh = flat(up_suh_ptr, total_experts * cutlass.Int64(self.hidden_size))
        rotation_input = flat(
            rotation_input_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),
        )

        # Tight upper bound on non-empty M8 route blocks for ``active_m`` tokens:
        # sum_e ceil(c_e / 8) <= (R + 7 * min(R, E)) // 8.
        live_routes = active_m * Int32(self.top_k)
        active_experts_bound = live_routes
        if active_experts_bound > route_num_experts:
            active_experts_bound = route_num_experts
        max_live_blocks = (live_routes + Int32(7) * active_experts_bound) // Int32(8)
        if max_live_blocks > Int32(self.driver.max_m_blocks):
            max_live_blocks = Int32(self.driver.max_m_blocks)
        if max_live_blocks < Int32(1):
            max_live_blocks = Int32(1)

        # Runtime workspace capacity: ``workspace`` is allocated with
        # ``max(4 * sms, blocks_per_sm * sms) + 2`` zero-initialized int32s,
        # and callers pass ``grid_x = sms * blocks_per_sm``.
        runtime_sms = grid_x // Int32(self.blocks_per_sm)
        ws_cap = runtime_sms * Int32(4)
        do_fused_act = Int32(0)
        if max_live_blocks <= ws_cap:
            do_fused_act = Int32(1)

        act_units = live_routes * Int32(self.intermediate_size // 128)
        act_ctas = (act_units + Int32(3)) // Int32(4)
        if act_ctas < Int32(1):
            act_ctas = Int32(1)

        narrow_fc1_n_tiles = (2 * self.intermediate_size) // self.narrow_fc1_tile_n
        narrow_fc1_threads = 128 if self.narrow_fc1_tile_n == 128 else 64
        fc2_n_tiles = self.hidden_size // self.fc2_tile_n
        fc2_threads = 128 if self.fc2_tile_n == 128 else 64

        if cutlass.const_expr(not self.has_smallm_n64 and not self.has_medm_arm):
            self.fc1_fused_narrow_kernel(
                rotation_input,
                gate_suh,
                up_suh,
                t0_w13,
                t0_w13_global,
                t1_w13,
                t1_w13_global,
                fc1,
                activated,
                intermediate_rotations,
                workspace,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                descriptor_map,
                tier0_num_experts,
                tier1_num_experts,
                tier0_gate_experts,
                tier1_gate_experts,
                tier0_up_experts,
                tier1_up_experts,
                active_m,
                do_fused_act,
            ).launch(
                grid=(narrow_fc1_n_tiles, max_live_blocks, 1),
                block=[narrow_fc1_threads, 1, 1],
                stream=stream,
            )
            if do_fused_act == Int32(0):
                self.act_kernel(
                    fc1,
                    activated,
                    intermediate_rotations,
                    raw_topk_ids,
                    global_to_combined,
                    descriptor_map,
                    route_num_experts,
                    tier0_num_experts + tier1_num_experts,
                    active_m,
                ).launch(
                    grid=(act_ctas, 1, 1),
                    block=[128, 1, 1],
                    use_pdl=True,
                    stream=stream,
                )
        else:
            if active_m <= Int32(_K4_SMALLM_MAX_ROWS):
                if cutlass.const_expr(self.has_smallm_n64):
                    if active_m <= Int32(_K4_NARROW_MAX_ROWS):
                        self.fc1_fused_narrow_kernel(
                            rotation_input,
                            gate_suh,
                            up_suh,
                            t0_w13,
                            t0_w13_global,
                            t1_w13,
                            t1_w13_global,
                            fc1,
                            activated,
                            intermediate_rotations,
                            workspace,
                            packed_route_indices,
                            block_expert_ids,
                            packed_route_count,
                            descriptor_map,
                            tier0_num_experts,
                            tier1_num_experts,
                            tier0_gate_experts,
                            tier1_gate_experts,
                            tier0_up_experts,
                            tier1_up_experts,
                            active_m,
                            do_fused_act,
                        ).launch(
                            grid=(narrow_fc1_n_tiles, max_live_blocks, 1),
                            block=[narrow_fc1_threads, 1, 1],
                            stream=stream,
                        )
                    else:
                        smallm_fc1_n_tiles = (
                            2 * self.intermediate_size
                        ) // self.smallm_fc1_tile_n
                        smallm_fc1_threads = (
                            128 if self.smallm_fc1_tile_n == 128 else 64
                        )
                        self.fc1_fused_smallm_kernel(
                            rotation_input,
                            gate_suh,
                            up_suh,
                            t0_w13,
                            t0_w13_global,
                            t1_w13,
                            t1_w13_global,
                            fc1,
                            activated,
                            intermediate_rotations,
                            workspace,
                            packed_route_indices,
                            block_expert_ids,
                            packed_route_count,
                            descriptor_map,
                            tier0_num_experts,
                            tier1_num_experts,
                            tier0_gate_experts,
                            tier1_gate_experts,
                            tier0_up_experts,
                            tier1_up_experts,
                            active_m,
                            do_fused_act,
                        ).launch(
                            grid=(smallm_fc1_n_tiles, max_live_blocks, 1),
                            block=[smallm_fc1_threads, 1, 1],
                            stream=stream,
                        )
                else:
                    self.fc1_fused_narrow_kernel(
                        rotation_input,
                        gate_suh,
                        up_suh,
                        t0_w13,
                        t0_w13_global,
                        t1_w13,
                        t1_w13_global,
                        fc1,
                        activated,
                        intermediate_rotations,
                        workspace,
                        packed_route_indices,
                        block_expert_ids,
                        packed_route_count,
                        descriptor_map,
                        tier0_num_experts,
                        tier1_num_experts,
                        tier0_gate_experts,
                        tier1_gate_experts,
                        tier0_up_experts,
                        tier1_up_experts,
                        active_m,
                        do_fused_act,
                    ).launch(
                        grid=(narrow_fc1_n_tiles, max_live_blocks, 1),
                        block=[narrow_fc1_threads, 1, 1],
                        stream=stream,
                    )
            else:
                if cutlass.const_expr(self.has_medm_arm):
                    rot_units = live_routes * Int32(self.hidden_size // 128)
                    rot_ctas = (rot_units + Int32(7)) // Int32(8)
                    if rot_ctas < Int32(1):
                        rot_ctas = Int32(1)
                    medm_fc1_n_tiles = (
                        2 * self.intermediate_size
                    ) // self.medm_fc1_tile_n
                    medm_fc1_threads = 128 if self.medm_fc1_tile_n == 128 else 64

                    self.rot_kernel(
                        rotation_input,
                        rotation_gate,
                        rotation_up,
                        gate_suh,
                        up_suh,
                        raw_topk_ids,
                        global_to_combined,
                        route_num_experts,
                        tier0_num_experts + tier1_num_experts,
                        active_m,
                    ).launch(
                        grid=(rot_ctas, 1, 1),
                        block=[256, 1, 1],
                        stream=stream,
                    )
                    self.fc1_prerot_medm_kernel(
                        rotation_gate,
                        rotation_up,
                        t0_w13,
                        t0_w13_global,
                        t1_w13,
                        t1_w13_global,
                        fc1,
                        activated,
                        intermediate_rotations,
                        workspace,
                        packed_route_indices,
                        block_expert_ids,
                        packed_route_count,
                        descriptor_map,
                        tier0_num_experts,
                        tier1_num_experts,
                        tier0_gate_experts,
                        tier1_gate_experts,
                        tier0_up_experts,
                        tier1_up_experts,
                        active_m,
                        do_fused_act,
                    ).launch(
                        grid=(medm_fc1_n_tiles, max_live_blocks, 1),
                        block=[medm_fc1_threads, 1, 1],
                        use_pdl=True,
                        stream=stream,
                    )

            if do_fused_act == Int32(0):
                self.act_kernel(
                    fc1,
                    activated,
                    intermediate_rotations,
                    raw_topk_ids,
                    global_to_combined,
                    descriptor_map,
                    route_num_experts,
                    tier0_num_experts + tier1_num_experts,
                    active_m,
                ).launch(
                    grid=(act_ctas, 1, 1),
                    block=[128, 1, 1],
                    use_pdl=True,
                    stream=stream,
                )

        self.fc2_kernel(
            activated,
            t0_w2,
            t0_w2_global,
            t1_w2,
            t1_w2_global,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            active_m,
        ).launch(
            grid=(fc2_n_tiles, max_live_blocks, 1),
            block=[fc2_threads, 1, 1],
            use_pdl=True,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Helper methods shared by FC1/FC2 kernels
    # ------------------------------------------------------------------

    @cute.jit
    def _trellis_funnel256(self, a: Uint32, b: Uint32, s2: Int32) -> Uint32:
        merged = (Int64(a) << Int64(32)) | Int64(b)
        return Uint32(merged >> Int64(s2))

    @cute.jit
    def _fetch_b_slab_k4(
        self,
        tile_n: cutlass.Constexpr[int],
        cta_threads: cutlass.Constexpr[int],
        b_stage_base: Int32,
        b_base_addr: Int64,
        b_iter_step_bytes: Int64,
        b_slab_step_bytes: Int64,
        slab: Int32,
        tid: Int32,
        lane: Int32,
        use_evict_first: Int32,
    ):
        cta_n_blocks = tile_n // 16
        chunks_per_k16 = cta_n_blocks * 8
        iters = (8 * chunks_per_k16) // cta_threads
        slab_addr = b_base_addr + Int64(slab) * b_slab_step_bytes
        # Materialize the per-thread shared B base through shuffle_sync so ptxas
        # keeps [R + imm] on LDGSTS with an L2 cache-hint descriptor in UR.
        b_dst_tid = cute.arch.shuffle_sync(b_stage_base + tid * Int32(16), lane)

        if use_evict_first != Int32(0):
            b_l2_policy = create_l2_evict_first_policy()
            for i in cutlass.range_constexpr(iters):
                _cp_async4_l2hint(
                    b_dst_tid + Int32(i * cta_threads * 16),
                    slab_addr + Int64(i) * b_iter_step_bytes,
                    b_l2_policy,
                )
        else:
            for i in cutlass.range_constexpr(iters):
                cp_async4_shared_global(
                    b_dst_tid + Int32(i * cta_threads * 16),
                    slab_addr + Int64(i) * b_iter_step_bytes,
                )

    @cute.jit
    def _fetch_b_slab_generic(
        self,
        bits: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        cta_threads: cutlass.Constexpr[int],
        b_flat: cute.Tensor,
        b_stage_base: Int32,
        b_tile_base_u32: Int64,
        b_k16_stride_u32: Int64,
        slab: Int32,
        tid: Int32,
        lane: Int32,
        use_evict_first: Int32,
    ):
        cta_n_blocks = tile_n // 16
        chunks_per_k16 = cta_n_blocks * 2 * bits
        total_chunks = 8 * chunks_per_k16
        iters = (total_chunks + cta_threads - 1) // cta_threads
        exact = (total_chunks % cta_threads) == 0
        slab_base_u32 = b_tile_base_u32 + Int64(slab) * Int64(8) * b_k16_stride_u32
        b_dst_tid = cute.arch.shuffle_sync(b_stage_base + tid * Int32(16), lane)

        if use_evict_first != Int32(0):
            b_l2_policy = create_l2_evict_first_policy()
            for i in cutlass.range_constexpr(iters):
                chunk = Int32(i * cta_threads) + tid
                k16 = chunk // Int32(chunks_per_k16)
                in_k16 = chunk - k16 * Int32(chunks_per_k16)
                dst = b_dst_tid + Int32(i * cta_threads * 16)
                src_addr = get_ptr_as_int64(
                    b_flat,
                    slab_base_u32 + Int64(k16) * b_k16_stride_u32 + Int64(in_k16 * 4),
                )
                if cutlass.const_expr(exact):
                    _cp_async4_l2hint(dst, src_addr, b_l2_policy)
                else:
                    _cp_async4_pred_l2hint(
                        dst,
                        src_addr,
                        (chunk < Int32(total_chunks)).to(Int32),
                        b_l2_policy,
                    )
        else:
            for i in cutlass.range_constexpr(iters):
                chunk = Int32(i * cta_threads) + tid
                k16 = chunk // Int32(chunks_per_k16)
                in_k16 = chunk - k16 * Int32(chunks_per_k16)
                dst = b_dst_tid + Int32(i * cta_threads * 16)
                src_addr = get_ptr_as_int64(
                    b_flat,
                    slab_base_u32 + Int64(k16) * b_k16_stride_u32 + Int64(in_k16 * 4),
                )
                if cutlass.const_expr(exact):
                    cp_async4_shared_global(dst, src_addr)
                else:
                    cp_async4_shared_global_pred(
                        dst,
                        src_addr,
                        (chunk < Int32(total_chunks)).to(Int32),
                    )

    @cute.jit
    def _fetch_b_slab(
        self,
        bits: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        cta_threads: cutlass.Constexpr[int],
        b_flat: cute.Tensor,
        b_stage_base: Int32,
        b_tile_base_u32: Int64,
        b_k16_stride_u32: Int64,
        b_base_addr: Int64,
        b_iter_step_bytes: Int64,
        b_slab_step_bytes: Int64,
        slab: Int32,
        tid: Int32,
        lane: Int32,
        use_evict_first: Int32,
    ):
        if cutlass.const_expr(bits == 4):
            self._fetch_b_slab_k4(
                tile_n,
                cta_threads,
                b_stage_base,
                b_base_addr,
                b_iter_step_bytes,
                b_slab_step_bytes,
                slab,
                tid,
                lane,
                use_evict_first,
            )
        else:
            self._fetch_b_slab_generic(
                bits,
                tile_n,
                cta_threads,
                b_flat,
                b_stage_base,
                b_tile_base_u32,
                b_k16_stride_u32,
                slab,
                tid,
                lane,
                use_evict_first,
            )

    @cute.jit
    def _fetch_a_prerot_slab(
        self,
        cta_threads: cutlass.Constexpr[int],
        a_stage_base: Int32,
        a_row_base0: Int64,
        a_swz_off0: Int32,
        a_valid0: Int32,
        a_row_base1: Int64,
        a_swz_off1: Int32,
        a_valid1: Int32,
        slab: Int32,
    ):
        slab_byte_off = Int64(slab) * Int64(256)
        cp_async4_shared_global_pred(
            a_stage_base + a_swz_off0, a_row_base0 + slab_byte_off, a_valid0
        )
        if cutlass.const_expr(cta_threads == 64):
            cp_async4_shared_global_pred(
                a_stage_base + a_swz_off1, a_row_base1 + slab_byte_off, a_valid1
            )

    @cute.jit
    def _rotate_a_fused_slab(
        self,
        cta_threads: cutlass.Constexpr[int],
        x_input: cute.Tensor,
        suh_ptr_base: Int64,
        a_stage_base: Int32,
        tok_lane: Int32,
        valid_rows: Int32,
        slab: Int32,
        warp_id: Int32,
        lane: Int32,
    ):
        # Each warp rotates rows warp_id, warp_id + n_warps, ... (< valid_rows)
        # directly from BF16 x_input + FP16 suh into swizzled FP16 SMEM A.
        n_warps = cta_threads // 32
        if warp_id < valid_rows:
            col = slab * Int32(128) + lane * Int32(4)
            sw0, sw1 = ld_global_nc_v2_u32(suh_ptr_base + Int64(col) * Int64(2))
            chunk = lane >> Int32(1)
            half_byte = (lane & Int32(1)) << Int32(3)
            r = warp_id
            while r < valid_rows:
                tok = cute.arch.shuffle_sync(tok_lane, r)
                x_off = Int64(tok) * Int64(self.hidden_size) + Int64(col)
                xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, x_off))
                xh0 = cvt_bf16x2_to_f16x2_via_f32(xw0)
                xh1 = cvt_bf16x2_to_f16x2_via_f32(xw1)
                v0, v1 = f16x2_to_f32x2(half2_mul(xh0, sw0))
                v2, v3 = f16x2_to_f32x2(half2_mul(xh1, sw1))
                h0, h1, h2, h3 = self.driver._had128_quad(v0, v1, v2, v3, lane)
                dst = (
                    a_stage_base
                    + (((r << Int32(4)) + (chunk ^ r)) << Int32(4))
                    + half_byte
                )
                _st_shared_v2_u32(
                    dst,
                    pack_f32x2_to_f16x2(h0, h1),
                    pack_f32x2_to_f16x2(h2, h3),
                )
                r += Int32(n_warps)

    @cute.jit
    def _zero_invalid_a_stages(
        self,
        cta_threads: cutlass.Constexpr[int],
        stages: cutlass.Constexpr[int],
        a_base: Int32,
        valid_rows: Int32,
        tid: Int32,
    ):
        iters = (128 * stages) // cta_threads
        for i in cutlass.range_constexpr(iters):
            idx = Int32(i * cta_threads) + tid
            stage_idx = idx >> Int32(7)
            in_stage = idx & Int32(127)
            row = in_stage >> Int32(4)
            if row >= valid_rows:
                c = in_stage & Int32(15)
                dst = (
                    a_base
                    + stage_idx * Int32(_A_SLAB_BYTES)
                    + (((row << Int32(4)) + (c ^ row)) << Int32(4))
                )
                st_shared_v4_u32(dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def _consume_slab(
        self,
        bits: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        b_stage_base: Int32,
        a_stage_base: Int32,
        a_rd_off0: Int32,
        a_rd_off1: Int32,
        warp_row: Int32,
        w_n: Int32,
        lane: Int32,
        acc,
    ):
        cta_n_blocks = tile_n // 16
        n_frags = 2 if tile_n == 32 else 4
        tile_u32 = 8 * bits
        tile_bytes = 32 * bits
        k16_bytes = cta_n_blocks * tile_bytes
        prev_lane = (lane + Int32(31)) & Int32(31)

        if cutlass.const_expr(bits == 5):
            ib0, ib1, sb, _ = trellis256_lane_geom_bits(lane, 0, 4, 5)
            ia0, ia1, sa, _ = trellis256_lane_geom_bits(lane, 4, 4, 5)
        elif cutlass.const_expr(bits == 6):
            i0, i2, s2_6, delta_6 = trellis256_lane_geom_bits(lane, 0, 8, 6)
            i1 = i0 + Int32(1)
            i1 = i1 - Int32(tile_u32) * (i1 >= Int32(tile_u32)).to(Int32)
        elif cutlass.const_expr(bits != 4):
            ia_g, ib_g, s2_g, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)

        b_warp_base = (
            b_stage_base
            + warp_row * Int32(2 * k16_bytes)
            + w_n * Int32(n_frags * tile_bytes)
        )
        for sub in cutlass.range_constexpr(2):
            for kk in cutlass.range_constexpr(2):
                a_off = a_rd_off0 if cutlass.const_expr(kk == 0) else a_rd_off1
                a0, a1 = ldmatrix_m8n8x2_b16(a_stage_base + a_off + Int32(sub * 128))
                b_sub_base = b_warp_base + Int32((sub * 4 + kk) * k16_bytes)

                if cutlass.const_expr(bits == 4):
                    b_lane_off = b_sub_base + lane * Int32(4)
                    bw0 = ld_shared_u32(b_lane_off)
                    bw1 = ld_shared_u32(b_lane_off + Int32(tile_bytes))
                    if cutlass.const_expr(n_frags == 4):
                        bw2 = ld_shared_u32(b_lane_off + Int32(2 * tile_bytes))
                        bw3 = ld_shared_u32(b_lane_off + Int32(3 * tile_bytes))
                    for jj in cutlass.range_constexpr(n_frags):
                        bw = bw0
                        if cutlass.const_expr(jj == 1):
                            bw = bw1
                        elif cutlass.const_expr(jj == 2):
                            bw = bw2
                        elif cutlass.const_expr(jj == 3):
                            bw = bw3
                        aw = cute.arch.shuffle_sync(bw, prev_lane)
                        win_a = bw
                        win_b = _prmt_b32_5432(bw, aw)
                        o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                            win_a, win_b, 4
                        )
                        d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                            acc[jj * 4 + 0][0],
                            acc[jj * 4 + 1][0],
                            acc[jj * 4 + 2][0],
                            acc[jj * 4 + 3][0],
                            o0,
                            o2,
                            o1,
                            o3,
                            a0,
                            a1,
                        )
                        acc[jj * 4 + 0][0] = d0
                        acc[jj * 4 + 1][0] = d1
                        acc[jj * 4 + 2][0] = d2
                        acc[jj * 4 + 3][0] = d3
                else:
                    for jj in cutlass.range_constexpr(n_frags):
                        tbase = b_sub_base + Int32(jj * tile_bytes)
                        if cutlass.const_expr(bits == 5):
                            b0 = ld_shared_u32(tbase + ib0 * Int32(4))
                            b1 = ld_shared_u32(tbase + ib1 * Int32(4))
                            x0 = ld_shared_u32(tbase + ia0 * Int32(4))
                            x1 = ld_shared_u32(tbase + ia1 * Int32(4))
                            win_b = self._trellis_funnel256(b0, b1, sb)
                            win_a = self._trellis_funnel256(x0, x1, sa)
                            o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                                win_a, win_b, 5
                            )
                        elif cutlass.const_expr(bits == 6):
                            z0 = ld_shared_u32(tbase + i0 * Int32(4))
                            z1 = ld_shared_u32(tbase + i1 * Int32(4))
                            z2 = ld_shared_u32(tbase + i2 * Int32(4))
                            win_a, win_b = trellis_align_stream_u32x2(
                                z0, z1, z2, s2_6, delta_6
                            )
                            o0, o1, o2, o3 = packed_dequant_trellis_stream_to_half2x4(
                                win_a, win_b, 6
                            )
                        else:
                            aw = ld_shared_u32(tbase + ia_g * Int32(4))
                            bw = ld_shared_u32(tbase + ib_g * Int32(4))
                            win_a = self._trellis_funnel256(aw, bw, s2_g)
                            win_b = self._trellis_funnel256(
                                aw, bw, s2_g + Int32(4 * bits)
                            )
                            o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                                win_a, win_b, bits
                            )
                        d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                            acc[jj * 4 + 0][0],
                            acc[jj * 4 + 1][0],
                            acc[jj * 4 + 2][0],
                            acc[jj * 4 + 3][0],
                            o0,
                            o2,
                            o1,
                            o3,
                            a0,
                            a1,
                        )
                        acc[jj * 4 + 0][0] = d0
                        acc[jj * 4 + 1][0] = d1
                        acc[jj * 4 + 2][0] = d2
                        acc[jj * 4 + 3][0] = d3

    @cute.jit
    def _run_gemm_kloop(
        self,
        bits: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        cta_threads: cutlass.Constexpr[int],
        stages: cutlass.Constexpr[int],
        fused_rot: cutlass.Constexpr[bool],
        pdl_wait: cutlass.Constexpr[bool],
        pdl_signal_after_wait: cutlass.Constexpr[bool],
        b_stage_bytes: cutlass.Constexpr[int],
        a_off: cutlass.Constexpr[int],
        a_stride_bytes: cutlass.Constexpr[int],
        x_input: cute.Tensor,
        suh_ptr_base: Int64,
        a_tensor_base: Int64,
        b_flat: cute.Tensor,
        b_tile_base_u32: Int64,
        b_k16_stride_u32: Int64,
        k_slabs: Int32,
        route_lane: Int32,
        tok_lane: Int32,
        valid_rows: Int32,
        smem_base: Int32,
        tid: Int32,
        warp_id: Int32,
        warp_row: Int32,
        w_n: Int32,
        lane: Int32,
        use_evict_first: Int32,
        acc,
    ):
        a_base = smem_base + Int32(a_off)
        l16 = lane & Int32(15)
        r_ldm = l16 & Int32(7)
        c0_ldm = l16 >> Int32(3)
        k_base = (warp_row << Int32(2)) + c0_ldm
        a_rd_off0 = (((r_ldm << Int32(4)) + (k_base ^ r_ldm)) << Int32(4))
        a_rd_off1 = (((r_ldm << Int32(4)) + ((k_base + Int32(2)) ^ r_ldm)) << Int32(4))

        # Hoist K4 B addressing out of the K-loop (64-bit safe).
        b_base_addr = Int64(0)
        b_iter_step_bytes = Int64(0)
        b_slab_step_bytes = Int64(0)
        if cutlass.const_expr(bits == 4):
            cta_n_blocks = tile_n // 16
            chunks_per_k16 = cta_n_blocks * 8
            k16_step = cta_threads // chunks_per_k16
            if cutlass.const_expr(chunks_per_k16 == 16):
                k16_0 = tid >> Int32(4)
                in_k16_0 = tid & Int32(15)
            elif cutlass.const_expr(chunks_per_k16 == 32):
                k16_0 = tid >> Int32(5)
                in_k16_0 = tid & Int32(31)
            else:
                k16_0 = tid >> Int32(6)
                in_k16_0 = tid & Int32(63)
            b_base_addr = get_ptr_as_int64(
                b_flat,
                b_tile_base_u32
                + Int64(k16_0) * b_k16_stride_u32
                + Int64(in_k16_0 * 4),
            )
            b_iter_step_bytes = Int64(k16_step) * b_k16_stride_u32 * Int64(4)
            b_slab_step_bytes = Int64(8) * b_k16_stride_u32 * Int64(4)

        # Hoist pre-rotated A addressing out of the K-loop.
        a_row_base0 = Int64(0)
        a_swz_off0 = Int32(0)
        a_valid0 = Int32(0)
        a_row_base1 = Int64(0)
        a_swz_off1 = Int32(0)
        a_valid1 = Int32(0)
        if cutlass.const_expr(not fused_rot):
            row0 = tid >> Int32(4)
            c0 = tid & Int32(15)
            route0 = cute.arch.shuffle_sync(route_lane, row0)
            a_swz_off0 = (((row0 << Int32(4)) + (c0 ^ row0)) << Int32(4))
            a_row_base0 = (
                a_tensor_base
                + Int64(route0) * Int64(a_stride_bytes)
                + Int64(c0 << Int32(4))
            )
            a_valid0 = (row0 < valid_rows).to(Int32)
            if cutlass.const_expr(cta_threads == 64):
                idx1 = tid + Int32(64)
                row1 = idx1 >> Int32(4)
                c1 = idx1 & Int32(15)
                route1 = cute.arch.shuffle_sync(route_lane, row1)
                a_swz_off1 = (((row1 << Int32(4)) + (c1 ^ row1)) << Int32(4))
                a_row_base1 = (
                    a_tensor_base
                    + Int64(route1) * Int64(a_stride_bytes)
                    + Int64(c1 << Int32(4))
                )
                a_valid1 = (row1 < valid_rows).to(Int32)

        # Prologue: when ``pdl_wait`` is True (dependent GEMM), issue B slabs
        # 0 and 1 from read-only weights and zero invalid A rows BEFORE
        # ``griddepcontrol_wait`` so weight DRAM latency overlaps with the
        # predecessor's tail wave, then fetch L2-hot A slabs after wait.
        if cutlass.const_expr(pdl_wait):
            self._fetch_b_slab(
                bits,
                tile_n,
                cta_threads,
                b_flat,
                smem_base,
                b_tile_base_u32,
                b_k16_stride_u32,
                b_base_addr,
                b_iter_step_bytes,
                b_slab_step_bytes,
                Int32(0),
                tid,
                lane,
                use_evict_first,
            )
            if k_slabs > Int32(1):
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_flat,
                    smem_base + Int32(b_stage_bytes),
                    b_tile_base_u32,
                    b_k16_stride_u32,
                    b_base_addr,
                    b_iter_step_bytes,
                    b_slab_step_bytes,
                    Int32(1),
                    tid,
                    lane,
                    use_evict_first,
                )
            cute.arch.cp_async_commit_group()
            self._zero_invalid_a_stages(cta_threads, stages, a_base, valid_rows, tid)

            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(pdl_signal_after_wait):
                cute.arch.griddepcontrol_launch_dependents()

            self._fetch_a_prerot_slab(
                cta_threads,
                a_base,
                a_row_base0,
                a_swz_off0,
                a_valid0,
                a_row_base1,
                a_swz_off1,
                a_valid1,
                Int32(0),
            )
            cute.arch.cp_async_commit_group()
            if k_slabs > Int32(1):
                self._fetch_a_prerot_slab(
                    cta_threads,
                    a_base + Int32(_A_SLAB_BYTES),
                    a_row_base0,
                    a_swz_off0,
                    a_valid0,
                    a_row_base1,
                    a_swz_off1,
                    a_valid1,
                    Int32(1),
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            self._fetch_b_slab(
                bits,
                tile_n,
                cta_threads,
                b_flat,
                smem_base,
                b_tile_base_u32,
                b_k16_stride_u32,
                b_base_addr,
                b_iter_step_bytes,
                b_slab_step_bytes,
                Int32(0),
                tid,
                lane,
                use_evict_first,
            )
            if cutlass.const_expr(not fused_rot):
                self._fetch_a_prerot_slab(
                    cta_threads,
                    a_base,
                    a_row_base0,
                    a_swz_off0,
                    a_valid0,
                    a_row_base1,
                    a_swz_off1,
                    a_valid1,
                    Int32(0),
                )
            cute.arch.cp_async_commit_group()

            if k_slabs > Int32(1):
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_flat,
                    smem_base + Int32(b_stage_bytes),
                    b_tile_base_u32,
                    b_k16_stride_u32,
                    b_base_addr,
                    b_iter_step_bytes,
                    b_slab_step_bytes,
                    Int32(1),
                    tid,
                    lane,
                    use_evict_first,
                )
                if cutlass.const_expr(not fused_rot):
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + Int32(_A_SLAB_BYTES),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                cute.arch.cp_async_commit_group()

            self._zero_invalid_a_stages(cta_threads, stages, a_base, valid_rows, tid)

            if cutlass.const_expr(fused_rot):
                self._rotate_a_fused_slab(
                    cta_threads,
                    x_input,
                    suh_ptr_base,
                    a_base,
                    tok_lane,
                    valid_rows,
                    Int32(0),
                    warp_id,
                    lane,
                )
                if k_slabs > Int32(1):
                    self._rotate_a_fused_slab(
                        cta_threads,
                        x_input,
                        suh_ptr_base,
                        a_base + Int32(_A_SLAB_BYTES),
                        tok_lane,
                        valid_rows,
                        Int32(1),
                        warp_id,
                        lane,
                    )

            if k_slabs > Int32(1):
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        if cutlass.const_expr(stages == 2):
            stage = Int32(0)
            s = Int32(0)
            while s < k_slabs:
                self._consume_slab(
                    bits,
                    tile_n,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_base + stage * Int32(_A_SLAB_BYTES),
                    a_rd_off0,
                    a_rd_off1,
                    warp_row,
                    w_n,
                    lane,
                    acc,
                )
                next2 = s + Int32(2)
                if next2 < k_slabs:
                    cute.arch.sync_threads()
                    self._fetch_b_slab(
                        bits,
                        tile_n,
                        cta_threads,
                        b_flat,
                        smem_base + stage * Int32(b_stage_bytes),
                        b_tile_base_u32,
                        b_k16_stride_u32,
                        b_base_addr,
                        b_iter_step_bytes,
                        b_slab_step_bytes,
                        next2,
                        tid,
                        lane,
                        use_evict_first,
                    )
                    if cutlass.const_expr(not fused_rot):
                        self._fetch_a_prerot_slab(
                            cta_threads,
                            a_base + stage * Int32(_A_SLAB_BYTES),
                            a_row_base0,
                            a_swz_off0,
                            a_valid0,
                            a_row_base1,
                            a_swz_off1,
                            a_valid1,
                            next2,
                        )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(fused_rot):
                        self._rotate_a_fused_slab(
                            cta_threads,
                            x_input,
                            suh_ptr_base,
                            a_base + stage * Int32(_A_SLAB_BYTES),
                            tok_lane,
                            valid_rows,
                            next2,
                            warp_id,
                            lane,
                        )
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                elif s + Int32(1) < k_slabs:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                stage = stage ^ Int32(1)
                s += Int32(1)
        else:
            stage = Int32(0)
            s = Int32(0)
            while s < k_slabs:
                wr_stage = stage + Int32(2)
                if wr_stage >= Int32(3):
                    wr_stage -= Int32(3)
                next2 = s + Int32(2)
                if next2 < k_slabs:
                    self._fetch_b_slab(
                        bits,
                        tile_n,
                        cta_threads,
                        b_flat,
                        smem_base + wr_stage * Int32(b_stage_bytes),
                        b_tile_base_u32,
                        b_k16_stride_u32,
                        b_base_addr,
                        b_iter_step_bytes,
                        b_slab_step_bytes,
                        next2,
                        tid,
                        lane,
                        use_evict_first,
                    )
                    if cutlass.const_expr(not fused_rot):
                        self._fetch_a_prerot_slab(
                            cta_threads,
                            a_base + wr_stage * Int32(_A_SLAB_BYTES),
                            a_row_base0,
                            a_swz_off0,
                            a_valid0,
                            a_row_base1,
                            a_swz_off1,
                            a_valid1,
                            next2,
                        )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(fused_rot):
                        self._rotate_a_fused_slab(
                            cta_threads,
                            x_input,
                            suh_ptr_base,
                            a_base + wr_stage * Int32(_A_SLAB_BYTES),
                            tok_lane,
                            valid_rows,
                            next2,
                            warp_id,
                            lane,
                        )

                self._consume_slab(
                    bits,
                    tile_n,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_base + stage * Int32(_A_SLAB_BYTES),
                    a_rd_off0,
                    a_rd_off1,
                    warp_row,
                    w_n,
                    lane,
                    acc,
                )

                if next2 < k_slabs:
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                elif s + Int32(1) < k_slabs:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                stage += Int32(1)
                if stage == Int32(3):
                    stage = Int32(0)
                s += Int32(1)

    @cute.jit
    def _epilogue_store(
        self,
        tile_n: cutlass.Constexpr[int],
        c_stride_elem: cutlass.Constexpr[int],
        fold_off: cutlass.Constexpr[int],
        c_flat: cute.Tensor,
        acc,
        global_scale_f32: cutlass.Float32,
        output_n_tile: Int32,
        route_lane: Int32,
        valid_rows: Int32,
        smem_base: Int32,
        tid: Int32,
        warp_row: Int32,
        w_n: Int32,
        lane: Int32,
    ):
        n_frags = 2 if tile_n == 32 else 4
        fold_base = smem_base + Int32(fold_off)
        out_base = smem_base
        out_stride_elem = tile_n + 8

        if warp_row == Int32(1):
            for jj in cutlass.range_constexpr(n_frags):
                f_addr = (
                    fold_base
                    + ((w_n * Int32(n_frags) + Int32(jj)) * Int32(32) + lane)
                    * Int32(16)
                )
                st_shared_v4_f32(
                    f_addr,
                    acc[jj * 4 + 0][0],
                    acc[jj * 4 + 1][0],
                    acc[jj * 4 + 2][0],
                    acc[jj * 4 + 3][0],
                )
        cute.arch.sync_threads()

        if warp_row == Int32(0):
            row0 = (lane & Int32(3)) << Int32(1)
            row1 = row0 + Int32(1)
            col_base = w_n * Int32(n_frags * 16) + (lane >> Int32(2))
            for jj in cutlass.range_constexpr(n_frags):
                f_addr = (
                    fold_base
                    + ((w_n * Int32(n_frags) + Int32(jj)) * Int32(32) + lane)
                    * Int32(16)
                )
                f0, f1, f2, f3 = ld_shared_v4_f32(f_addr)
                col_lo = col_base + Int32(jj * 16)
                col_hi = col_lo + Int32(8)
                st_shared_f16_from_f32(
                    out_base + (row0 * Int32(out_stride_elem) + col_lo) * Int32(2),
                    (acc[jj * 4 + 0][0] + f0) * global_scale_f32,
                )
                st_shared_f16_from_f32(
                    out_base + (row1 * Int32(out_stride_elem) + col_lo) * Int32(2),
                    (acc[jj * 4 + 1][0] + f1) * global_scale_f32,
                )
                st_shared_f16_from_f32(
                    out_base + (row0 * Int32(out_stride_elem) + col_hi) * Int32(2),
                    (acc[jj * 4 + 2][0] + f2) * global_scale_f32,
                )
                st_shared_f16_from_f32(
                    out_base + (row1 * Int32(out_stride_elem) + col_hi) * Int32(2),
                    (acc[jj * 4 + 3][0] + f3) * global_scale_f32,
                )
        cute.arch.sync_threads()

        # Coalesced 16-byte stores: 8 rows x (tile_n / 8) chunks = tile_n chunks.
        chunks_per_row = tile_n // 8
        if tid < Int32(tile_n):
            if cutlass.const_expr(chunks_per_row == 4):
                row = tid >> Int32(2)
                c8 = tid & Int32(3)
            elif cutlass.const_expr(chunks_per_row == 8):
                row = tid >> Int32(3)
                c8 = tid & Int32(7)
            else:
                row = tid >> Int32(4)
                c8 = tid & Int32(15)
            route_idx = cute.arch.shuffle_sync(route_lane, row)
            if row < valid_rows:
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    out_base + (row * Int32(out_stride_elem) + c8 * Int32(8)) * Int32(2)
                )
                out_elem = Int64(route_idx) * Int64(c_stride_elem) + Int64(
                    output_n_tile * Int32(tile_n) + c8 * Int32(8)
                )
                st_global_v4_u32(get_ptr_as_int64(c_flat, out_elem), q0, q1, q2, q3)

    @cute.jit
    def _activate_h128_quad(
        self,
        gp01: Uint32,
        gp23: Uint32,
        up01: Uint32,
        up23: Uint32,
        svg01: Uint32,
        svg23: Uint32,
        svu01: Uint32,
        svu23: Uint32,
        sd01: Uint32,
        sd23: Uint32,
        lane: Int32,
    ):
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

        ig0, iu0 = self.driver._clamp_swiglu_inputs(ig0, iu0)
        ig1, iu1 = self.driver._clamp_swiglu_inputs(ig1, iu1)
        ig2, iu2 = self.driver._clamp_swiglu_inputs(ig2, iu2)
        ig3, iu3 = self.driver._clamp_swiglu_inputs(ig3, iu3)

        if cutlass.const_expr(self.driver.activation_is_situ):
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
        else:
            a0 = self.driver._gated_silu(ig0) * iu0 * sd0
            a1 = self.driver._gated_silu(ig1) * iu1 * sd1
            a2 = self.driver._gated_silu(ig2) * iu2 * sd2
            a3 = self.driver._gated_silu(ig3) * iu3 * sd3

        o0, o1, o2, o3 = self.driver._had128_quad(a0, a1, a2, a3, lane)
        return pack_f32x2_to_f16x2(o0, o1), pack_f32x2_to_f16x2(o2, o3)

    @cute.jit
    def _run_fused_act_block(
        self,
        cta_threads: cutlass.Constexpr[int],
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        combined_expert: Int32,
        live_width: Int32,
        route_lane: Int32,
        valid_rows: Int32,
        warp_id: Int32,
        lane: Int32,
    ):
        n_warps = cta_threads // 32
        nblk = live_width >> Int32(7)
        total_units = valid_rows * nblk
        isz = Int64(self.intermediate_size)
        exp_rot_base = Int64(combined_expert) * Int64(3 * self.intermediate_size)
        u = warp_id
        while u < total_units:
            r = u // nblk
            blk = u - r * nblk
            route = cute.arch.shuffle_sync(route_lane, r)
            col0 = blk * Int32(128) + lane * Int32(4)
            g_off = Int64(route) * Int64(2 * self.intermediate_size) + Int64(col0)
            u_off = g_off + isz
            s_off = exp_rot_base + Int64(col0)

            # L2-coherent loads on ``fc1`` guarantee visibility of sibling FC1
            # CTA stores synchronized via ``atom.acq_rel.gpu``.
            gp01, gp23 = _ld_global_cg_v2_u32(get_ptr_as_int64(fc1, g_off))
            up01, up23 = _ld_global_cg_v2_u32(get_ptr_as_int64(fc1, u_off))
            svg01, svg23 = ld_global_nc_v2_u32(
                get_ptr_as_int64(intermediate_rotations, s_off)
            )
            svu01, svu23 = ld_global_nc_v2_u32(
                get_ptr_as_int64(intermediate_rotations, s_off + isz)
            )
            sd01, sd23 = ld_global_nc_v2_u32(
                get_ptr_as_int64(intermediate_rotations, s_off + isz + isz)
            )

            p01, p23 = self._activate_h128_quad(
                gp01,
                gp23,
                up01,
                up23,
                svg01,
                svg23,
                svu01,
                svu23,
                sd01,
                sd23,
                lane,
            )
            out_off = Int64(route) * isz + Int64(col0)
            st_global_v2_u32(get_ptr_as_int64(activated, out_off), p01, p23)
            u += Int32(n_warps)

    @cute.jit
    def _gemm_tile_body(
        self,
        is_fc1: cutlass.Constexpr[bool],
        tile_n: cutlass.Constexpr[int],
        stages: cutlass.Constexpr[int],
        fused_rot: cutlass.Constexpr[bool],
        pdl_wait: cutlass.Constexpr[bool],
        pdl_signal_entry: cutlass.Constexpr[bool],
        pdl_signal_after_wait: cutlass.Constexpr[bool],
        x_or_a_gate: cute.Tensor,
        a_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        t0_b: cute.Tensor,
        t0_global: cute.Tensor,
        t1_b: cute.Tensor,
        t1_global: cute.Tensor,
        c_flat: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        workspace: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: Int32,
        tier1_num_experts: Int32,
        tier0_bound0: Int32,
        tier1_bound0: Int32,
        tier0_bound1: Int32,
        tier1_bound1: Int32,
        active_m: Int32,
        do_fused_act: Int32,
    ):
        if cutlass.const_expr(pdl_signal_entry):
            cute.arch.griddepcontrol_launch_dependents()

        cta_threads = 128 if tile_n == 128 else 64
        tb_n_warps = 2 if tile_n == 128 else 1
        n_frags = 2 if tile_n == 32 else 4
        b_stage_bytes, a_off, fold_off, smem_words = _k4_smem_layout(
            tile_n, self.max_bits, stages
        )

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        output_n_tile = Int32(bidx)
        rb = Int32(bidy)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, smem_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        actual_blocks = packed_route_count[Int32(0)].to(Int32) >> Int32(3)
        if rb < actual_blocks:
            combined_expert = block_expert_ids[rb].to(Int32)
            total_experts = tier0_num_experts + tier1_num_experts
            if combined_expert >= Int32(0) and combined_expert < total_experts:
                descriptor_row = Int32(2)
                proj = Int32(0)
                local_n_tile = output_n_tile
                if cutlass.const_expr(is_fc1):
                    fc1_half_tiles = self.intermediate_size // tile_n
                    if output_n_tile >= Int32(fc1_half_tiles):
                        descriptor_row = Int32(1)
                        proj = Int32(1)
                        local_n_tile = output_n_tile - Int32(fc1_half_tiles)
                    else:
                        descriptor_row = Int32(0)

                live_width = Int32(self.intermediate_size)
                weight_offset = Int64(0)
                if cutlass.const_expr(self.compact_weights):
                    live_width = descriptor_map[
                        Int64(3) * Int64(total_experts) + Int64(combined_expert)
                    ].to(Int32)
                    weight_offset = descriptor_map[
                        (Int64(descriptor_row) + Int64(4)) * Int64(total_experts)
                        + Int64(combined_expert)
                    ].to(Int64)

                in_live_n = (live_width > Int32(0)).to(Int32)
                if cutlass.const_expr(is_fc1 and self.compact_weights):
                    in_live_n = (
                        live_width > Int32(0)
                        and local_n_tile * Int32(tile_n) < live_width
                    ).to(Int32)

                if in_live_n != Int32(0):
                    lane = tid & Int32(31)
                    warp_id = tid >> Int32(5)
                    if cutlass.const_expr(tb_n_warps == 1):
                        w_n = Int32(0)
                        warp_row = warp_id
                    else:
                        w_n = warp_id & Int32(1)
                        warp_row = warp_id >> Int32(1)

                    live_routes = active_m * Int32(self.top_k)
                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")

                    if valid_rows > Int32(0):
                        route_lane = Int32(0)
                        if is_valid != Int32(0):
                            route_lane = r_lane

                        descriptor = descriptor_map[
                            Int64(descriptor_row) * Int64(total_experts)
                            + Int64(combined_expert)
                        ].to(Int32)

                        if descriptor >= Int32(0):
                            tier = descriptor >> Int32(self.descriptor_bits)
                            local_expert = descriptor & Int32(
                                (1 << self.descriptor_bits) - 1
                            )

                            if cutlass.const_expr(is_fc1):
                                t0_lim = tier0_bound0
                                t1_lim = tier1_bound0
                                if proj != Int32(0):
                                    t0_lim = tier0_bound1
                                    t1_lim = tier1_bound1
                            else:
                                t0_lim = tier0_bound0
                                t1_lim = tier1_bound0

                            is_t0 = tier == Int32(0) and local_expert < t0_lim
                            is_t1 = tier == Int32(1) and local_expert < t1_lim

                            if is_t0 or is_t1:
                                tok_lane = Int32(0)
                                if cutlass.const_expr(fused_rot):
                                    tok_lane = route_lane // Int32(self.top_k)

                                # Adaptive L2 policy: keep weights in L2 when the
                                # immediately following block belongs to the same
                                # expert (multi-block reuse); stream evict-first
                                # on single/last expert blocks.
                                use_evict_first = Int32(1)
                                if rb + Int32(1) < actual_blocks:
                                    if (
                                        block_expert_ids[rb + Int32(1)].to(Int32)
                                        == combined_expert
                                    ):
                                        use_evict_first = Int32(0)

                                suh_ptr_base = Int64(0)
                                a_tensor_base = Int64(0)
                                if cutlass.const_expr(fused_rot):
                                    if proj != Int32(0):
                                        suh_ptr_base = get_ptr_as_int64(
                                            up_suh,
                                            Int64(combined_expert)
                                            * Int64(self.hidden_size),
                                        )
                                    else:
                                        suh_ptr_base = get_ptr_as_int64(
                                            gate_suh,
                                            Int64(combined_expert)
                                            * Int64(self.hidden_size),
                                        )
                                else:
                                    a_tensor_base = get_ptr_as_int64(
                                        x_or_a_gate, Int64(0)
                                    )
                                    if cutlass.const_expr(is_fc1):
                                        if proj != Int32(0):
                                            a_tensor_base = get_ptr_as_int64(
                                                a_up, Int64(0)
                                            )

                                a_stride_bytes = (
                                    self.hidden_size * 2
                                    if is_fc1
                                    else self.intermediate_size * 2
                                )
                                c_stride_elem = (
                                    2 * self.intermediate_size
                                    if is_fc1
                                    else self.hidden_size
                                )
                                k_slabs = (
                                    Int32(self.hidden_size // 128)
                                    if cutlass.const_expr(is_fc1)
                                    else (live_width >> Int32(7))
                                )

                                acc = [
                                    cute.make_rmem_tensor((1,), cutlass.Float32)
                                    for _ in range(4 * n_frags)
                                ]
                                for idx in cutlass.range_constexpr(4 * n_frags):
                                    acc[idx].fill(0.0)

                                global_scale_f32 = cutlass.Float32(1.0)
                                cta_n_blocks = tile_n // 16

                                if is_t0:
                                    bits0 = self.tier0.trellis_bits
                                    tile_u32_0 = 8 * bits0
                                    global_scale_f32 = t0_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        if cutlass.const_expr(is_fc1):
                                            b_k16_stride_0 = (
                                                live_width.to(Int64) >> Int64(4)
                                            ) * Int64(tile_u32_0)
                                            b_tile_base_0 = weight_offset + Int64(
                                                local_n_tile
                                            ) * Int64(cta_n_blocks * tile_u32_0)
                                        else:
                                            b_k16_stride_0 = Int64(
                                                (self.hidden_size // 16) * tile_u32_0
                                            )
                                            b_tile_base_0 = weight_offset + Int64(
                                                output_n_tile
                                            ) * Int64(cta_n_blocks * tile_u32_0)
                                    elif cutlass.const_expr(is_fc1):
                                        half_n16 = self.intermediate_size // 16
                                        proj_expert_u32 = (
                                            (self.hidden_size // 16)
                                            * half_n16
                                            * tile_u32_0
                                        )
                                        plane_u32 = Int64(cute.size(t0_b)) // Int64(2)
                                        b_k16_stride_0 = Int64(half_n16 * tile_u32_0)
                                        b_tile_base_0 = (
                                            Int64(proj) * plane_u32
                                            + Int64(local_expert)
                                            * Int64(proj_expert_u32)
                                            + Int64(local_n_tile)
                                            * Int64(cta_n_blocks * tile_u32_0)
                                        )
                                    else:
                                        n16 = self.hidden_size // 16
                                        expert_u32 = (
                                            (self.intermediate_size // 16)
                                            * n16
                                            * tile_u32_0
                                        )
                                        b_k16_stride_0 = Int64(n16 * tile_u32_0)
                                        b_tile_base_0 = Int64(local_expert) * Int64(
                                            expert_u32
                                        ) + Int64(output_n_tile) * Int64(
                                            cta_n_blocks * tile_u32_0
                                        )

                                    self._run_gemm_kloop(
                                        bits0,
                                        tile_n,
                                        cta_threads,
                                        stages,
                                        fused_rot,
                                        pdl_wait,
                                        pdl_signal_after_wait,
                                        b_stage_bytes,
                                        a_off,
                                        a_stride_bytes,
                                        x_or_a_gate,
                                        suh_ptr_base,
                                        a_tensor_base,
                                        t0_b,
                                        b_tile_base_0,
                                        b_k16_stride_0,
                                        k_slabs,
                                        route_lane,
                                        tok_lane,
                                        valid_rows,
                                        smem_base,
                                        tid,
                                        warp_id,
                                        warp_row,
                                        w_n,
                                        lane,
                                        use_evict_first,
                                        acc,
                                    )
                                else:
                                    bits1 = self.tier1.trellis_bits
                                    tile_u32_1 = 8 * bits1
                                    global_scale_f32 = t1_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        if cutlass.const_expr(is_fc1):
                                            b_k16_stride_1 = (
                                                live_width.to(Int64) >> Int64(4)
                                            ) * Int64(tile_u32_1)
                                            b_tile_base_1 = weight_offset + Int64(
                                                local_n_tile
                                            ) * Int64(cta_n_blocks * tile_u32_1)
                                        else:
                                            b_k16_stride_1 = Int64(
                                                (self.hidden_size // 16) * tile_u32_1
                                            )
                                            b_tile_base_1 = weight_offset + Int64(
                                                output_n_tile
                                            ) * Int64(cta_n_blocks * tile_u32_1)
                                    elif cutlass.const_expr(is_fc1):
                                        half_n16 = self.intermediate_size // 16
                                        proj_expert_u32 = (
                                            (self.hidden_size // 16)
                                            * half_n16
                                            * tile_u32_1
                                        )
                                        plane_u32 = Int64(cute.size(t1_b)) // Int64(2)
                                        b_k16_stride_1 = Int64(half_n16 * tile_u32_1)
                                        b_tile_base_1 = (
                                            Int64(proj) * plane_u32
                                            + Int64(local_expert)
                                            * Int64(proj_expert_u32)
                                            + Int64(local_n_tile)
                                            * Int64(cta_n_blocks * tile_u32_1)
                                        )
                                    else:
                                        n16 = self.hidden_size // 16
                                        expert_u32 = (
                                            (self.intermediate_size // 16)
                                            * n16
                                            * tile_u32_1
                                        )
                                        b_k16_stride_1 = Int64(n16 * tile_u32_1)
                                        b_tile_base_1 = Int64(local_expert) * Int64(
                                            expert_u32
                                        ) + Int64(output_n_tile) * Int64(
                                            cta_n_blocks * tile_u32_1
                                        )

                                    self._run_gemm_kloop(
                                        bits1,
                                        tile_n,
                                        cta_threads,
                                        stages,
                                        fused_rot,
                                        pdl_wait,
                                        pdl_signal_after_wait,
                                        b_stage_bytes,
                                        a_off,
                                        a_stride_bytes,
                                        x_or_a_gate,
                                        suh_ptr_base,
                                        a_tensor_base,
                                        t1_b,
                                        b_tile_base_1,
                                        b_k16_stride_1,
                                        k_slabs,
                                        route_lane,
                                        tok_lane,
                                        valid_rows,
                                        smem_base,
                                        tid,
                                        warp_id,
                                        warp_row,
                                        w_n,
                                        lane,
                                        use_evict_first,
                                        acc,
                                    )

                                self._epilogue_store(
                                    tile_n,
                                    c_stride_elem,
                                    fold_off,
                                    c_flat,
                                    acc,
                                    global_scale_f32,
                                    output_n_tile,
                                    route_lane,
                                    valid_rows,
                                    smem_base,
                                    tid,
                                    warp_row,
                                    w_n,
                                    lane,
                                )

                        if cutlass.const_expr(is_fc1):
                            if do_fused_act != Int32(0):
                                cute.arch.sync_threads()
                                expected_tiles = Int32(2) * (
                                    live_width // Int32(tile_n)
                                )
                                if tid == Int32(0):
                                    ws_addr = get_ptr_as_int64(workspace, Int64(rb))
                                    old = _atomic_add_acq_rel_gpu_i32(
                                        ws_addr, Int32(1)
                                    )
                                    done = (
                                        old + Int32(1) == expected_tiles
                                    ).to(Int32)
                                    if done != Int32(0):
                                        _st_global_relaxed_gpu_i32(ws_addr, Int32(0))
                                    st_shared_i32(smem_base, done)
                                cute.arch.sync_threads()
                                if ld_shared_i32_relaxed(smem_base) != Int32(0):
                                    self._run_fused_act_block(
                                        cta_threads,
                                        c_flat,
                                        activated,
                                        intermediate_rotations,
                                        combined_expert,
                                        live_width,
                                        route_lane,
                                        valid_rows,
                                        warp_id,
                                        lane,
                                    )

        if cutlass.const_expr(not is_fc1):
            cute.arch.griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------
    # Compiled CUDA kernels
    # ------------------------------------------------------------------

    @cute.kernel
    def fc1_fused_narrow_kernel(
        self,
        rotation_input: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w13_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        workspace: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        do_fused_act: cutlass.Int32,
    ):
        self._gemm_tile_body(
            True,
            self.narrow_fc1_tile_n,
            _FC1_SLAB_STAGES,
            True,
            False,
            True,
            False,
            rotation_input,
            rotation_input,
            gate_suh,
            up_suh,
            t0_w13,
            t0_w13_global,
            t1_w13,
            t1_w13_global,
            fc1,
            activated,
            intermediate_rotations,
            workspace,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            active_m,
            do_fused_act,
        )

    @cute.kernel
    def fc1_fused_smallm_kernel(
        self,
        rotation_input: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w13_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        workspace: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        do_fused_act: cutlass.Int32,
    ):
        self._gemm_tile_body(
            True,
            self.smallm_fc1_tile_n,
            _FC1_SLAB_STAGES,
            True,
            False,
            True,
            False,
            rotation_input,
            rotation_input,
            gate_suh,
            up_suh,
            t0_w13,
            t0_w13_global,
            t1_w13,
            t1_w13_global,
            fc1,
            activated,
            intermediate_rotations,
            workspace,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            active_m,
            do_fused_act,
        )

    @cute.kernel
    def fc1_prerot_medm_kernel(
        self,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w13_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        workspace: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        do_fused_act: cutlass.Int32,
    ):
        self._gemm_tile_body(
            True,
            self.medm_fc1_tile_n,
            _FC1_SLAB_STAGES,
            False,
            True,
            False,
            True,
            rotation_gate,
            rotation_up,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_global,
            t1_w13,
            t1_w13_global,
            fc1,
            activated,
            intermediate_rotations,
            workspace,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            active_m,
            do_fused_act,
        )

    @cute.kernel
    def fc2_kernel(
        self,
        activated: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        self._gemm_tile_body(
            False,
            self.fc2_tile_n,
            _FC2_SLAB_STAGES,
            False,
            True,
            False,
            False,
            activated,
            activated,
            activated,
            activated,
            t0_w2,
            t0_w2_global,
            t1_w2,
            t1_w2_global,
            fc2,
            activated,
            activated,
            packed_route_count,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            active_m,
            Int32(0),
        )

    @cute.kernel
    def rot_kernel(
        self,
        x_input: cute.Tensor,
        a_gate: cute.Tensor,
        a_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        global_to_combined: cute.Tensor,
        route_num_experts: cutlass.Int32,
        total_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)
        lane = tid & Int32(31)
        unit = cta * Int32(8) + (tid >> Int32(5))

        nblk = Int32(self.hidden_size // 128)
        total_units = active_m * Int32(self.top_k) * nblk
        if unit < total_units:
            # Group by (token, H128 block) across warps of a CTA so all warps
            # share the token input block in L1.
            tb = unit // Int32(self.top_k)
            k = unit - tb * Int32(self.top_k)
            token = tb // nblk
            blk = tb - token * nblk
            route = token * Int32(self.top_k) + k

            ex_lane0 = Int32(-1)
            if lane == Int32(0):
                ge = raw_topk_ids[route].to(Int32)
                if ge >= Int32(0) and ge < route_num_experts:
                    ce = global_to_combined[ge].to(Int32)
                    if ce >= Int32(0) and ce < total_experts:
                        ex_lane0 = ce
            expert = cute.arch.shuffle_sync(ex_lane0, Int32(0))
            if expert >= Int32(0):
                col0 = blk * Int32(128) + lane * Int32(4)
                x_off = Int64(token) * Int64(self.hidden_size) + Int64(col0)
                s_off = Int64(expert) * Int64(self.hidden_size) + Int64(col0)
                out_off = Int64(route) * Int64(self.hidden_size) + Int64(col0)

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

    @cute.kernel
    def act_kernel(
        self,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        global_to_combined: cute.Tensor,
        descriptor_map: cute.Tensor,
        route_num_experts: cutlass.Int32,
        total_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)
        lane = tid & Int32(31)
        unit = cta * Int32(4) + (tid >> Int32(5))

        nblk = Int32(self.intermediate_size // 128)
        total_units = active_m * Int32(self.top_k) * nblk
        if unit < total_units:
            route = unit // nblk
            blk = unit - route * nblk

            ex_lane0 = Int32(-1)
            if lane == Int32(0):
                ge = raw_topk_ids[route].to(Int32)
                if ge >= Int32(0) and ge < route_num_experts:
                    ce = global_to_combined[ge].to(Int32)
                    if ce >= Int32(0) and ce < total_experts:
                        if cutlass.const_expr(self.compact_weights):
                            lw = descriptor_map[
                                Int64(3) * Int64(total_experts) + Int64(ce)
                            ].to(Int32)
                            if blk * Int32(128) < lw:
                                ex_lane0 = ce
                        else:
                            ex_lane0 = ce
            expert = cute.arch.shuffle_sync(ex_lane0, Int32(0))
            if expert >= Int32(0):
                isz = Int64(self.intermediate_size)
                col0 = blk * Int32(128) + lane * Int32(4)
                s_off = Int64(expert) * Int64(3 * self.intermediate_size) + Int64(col0)

                # Pre-load read-only rotation scales before waiting on FC1.
                svg01, svg23 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(intermediate_rotations, s_off)
                )
                svu01, svu23 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(intermediate_rotations, s_off + isz)
                )
                sd01, sd23 = ld_global_nc_v2_u32(
                    get_ptr_as_int64(intermediate_rotations, s_off + isz + isz)
                )

                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()

                g_off = Int64(route) * Int64(2 * self.intermediate_size) + Int64(col0)
                u_off = g_off + isz
                gp01, gp23 = _ld_global_cg_v2_u32(get_ptr_as_int64(fc1, g_off))
                up01, up23 = _ld_global_cg_v2_u32(get_ptr_as_int64(fc1, u_off))

                p01, p23 = self._activate_h128_quad(
                    gp01,
                    gp23,
                    up01,
                    up23,
                    svg01,
                    svg23,
                    svu01,
                    svu23,
                    sd01,
                    sd23,
                    lane,
                )
                out_off = Int64(route) * isz + Int64(col0)
                st_global_v2_u32(get_ptr_as_int64(activated, out_off), p01, p23)
