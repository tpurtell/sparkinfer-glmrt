"""SM121 (GB10) EXL3 K4 small-M non-cooperative mixed-Trellis decode/verify kernel.

Targets the DGX Spark (SM121, GB10) cooperative decode/verify path (``size_m <= 80``,
``moe_block_size == 8``, ``tier0_bits == 4``, MCG codebook) for both compact and
contiguous weight layouts across all gated activations (``silu``,
``silu_post_clamp``, ``situ``).

Structural design (bit-identical to ``W4A16MixedTrellisKernel`` and invariant
across ``m1``, ``m8``, and ``m80``):

1. Single-kernel persistent work-stealing decode for small-M (``m1..m11`` on
   GB10, whenever ``5 + 2 * max_live_blocks <= 4 * runtime_sms``):
   - Replaces the two-kernel PDL launch + global ``griddepcontrol_wait`` barrier
     and redundant fused input rotation with ONE non-cooperative persistent
     kernel (``grid = (3 * runtime_sms, 1, 1)``, ``block = 128`` = 4 warps/CTA
     = 12 warps/SM) on the exact ``v41_exl3_core`` ABI.
   - Phase 0 (``rot``): splits each live route block's ``H // 128`` H128 blocks
     into 4 contiguous chunks across 4 warps/CTA, loading ``x_input`` and
     ``suh`` once per chunk and writing L2-hot ``rotation_gate`` /
     ``rotation_up``. Early FC1 CTAs prefetch cold LPDDR5X weight slabs while
     waiting on per-block ``rot`` completion (``WS_STATE[rb]`` bits 28..30).
   - Phase 1 (``FC1``) + fused last-CTA activation + Phase 2 (``FC2``): shares
     ONE unified ``tile_n = 64, cta_threads = 128`` (``n_frags = 2``, 8 FP32
     accumulator regs/thread, 100%-coalesced 512 B/warp K16 bursts) GEMM body
     across both FC1 and FC2. Dead compact tiles are retired in ``tid == 0``
     with zero barriers and zero SM skew. The final FC1 tile of each route
     block ``rb`` immediately runs 4-warp vectorized activation out of L2 and
     publishes ``rb`` into the dynamic ``WS_READY_LIST`` queue, unblocking FC2
     for ``rb`` immediately in parallel with remaining FC1 blocks (no global
     FC1->FC2 barrier, no second kernel launch, and safe under ``fc2`` aliasing
     ``rotation_gate`` because each route belongs to a unique ``rb``).
   - Monotonic atomic cursors guarantee deadlock-freedom for any resident CTA
     count ``R >= 1``, and the unique last-exiting CTA resets ``workspace`` to
     zeros for CUDA graph replay.
   - Zero-line A SMEM addressing (``zero16_addr``) eliminates
     ``_zero_invalid_a_stages`` across all paths and shrinks A SMEM to 256 B/
     stage on compiled ``size_m == 1``, enabling a 4-stage branchless ring
     (``stages = 4``, 27 KB at K6 -> 3 CTAs/SM = 12 warps/SM, 3 slabs in flight)
     for ``m1`` and a 3-stage ring (26 KB at K6 -> 3 CTAs/SM) for ``m2..m11``.
2. Multi-kernel PDL paths for larger small-M and medium-M waves:
   - ``m12..m16``: fused-rotation ``fc1_tile_n = 64`` with fused last-CTA
     activation -> PDL -> 2-stage ``fc2_tile_n = 128``.
   - ``m17..m80``: vectorized live-route ``rot_kernel`` -> PDL ->
     ``fc1_tile_n = 128`` (3-stage, 1024 B rows, adaptive L2 policy) -> PDL ->
     vectorized ``act_kernel`` -> PDL -> 2-stage ``fc2_tile_n = 128``.
3. Exact K-reduction and hoisted 64-bit addressing:
   - Every GEMM preserves the exact 2-chain K-reduction order
     (``K16 % 4 in {0, 1}`` and ``{2, 3}``) of the production K64 kernel, uses
     hoisted 64-bit B and A pointers outside the K-loop, 1-cycle ``prmt.b32``
     K4 ring extraction, and adaptive per-block L2 evict-first policy.
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
    packed_dequant_trellis_mul1_to_half2x4,
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
from b12x.moe._shared.trellis_codebooks import MCG, MUL1

from .kernel import W4A16FusedMoeKernel
from .mixed_trellis import W4A16MixedTrellisKernel


_K4_SMALLM_MAX_ROWS = 16
_FC1_SLAB_STAGES = 3
_FC2_SLAB_STAGES = 2
_PERSISTENT_M1_STAGES = 4
_PERSISTENT_SMALLM_STAGES = 3
_PERSISTENT_CTAS_PER_SM = 3
_A_SLAB_BYTES_FULL = 2048  # 8 rows x 128 FP16 = 2048 B
_A_SLAB_BYTES_M1 = 256  # 1 row x 128 FP16 = 256 B

_ROT_SPLITS = 4
_ROT_DONE_STEP = 0x10000000
_ROT_READY_VAL = 0x40000000
_ROT_MASK = 0x70000000
_FC1_CNT_MASK = 0x0FFFFFFF

_WS_ROT_CURSOR = 0
_WS_FC1_CURSOR = 1
_WS_FC2_CURSOR = 2
_WS_READY_COUNT = 3
_WS_EXIT_COUNT = 4
_WS_HEADER = 5

_ACT_EXIT = 0
_ACT_ROT = 1
_ACT_FC1 = 2
_ACT_ACT_ONLY = 3
_ACT_FC2 = 4


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
def _ld_global_acquire_gpu_i32(
    addr: Int64,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """GPU-scoped acquire global int32 load."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_release_gpu_i32(
    addr: Int64,
    val: Int32,
    *,
    loc=None,
    ip=None,
):
    """GPU-scoped release global int32 store."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.release.gpu.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_relaxed_gpu_i32(
    addr: Int64,
    val: Int32,
    *,
    loc=None,
    ip=None,
):
    """GPU-scoped relaxed int32 global store (resets workspace slots)."""
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


@dsl_user_op
def _nanosleep(
    nanoseconds: Int32,
    *,
    loc=None,
    ip=None,
):
    """Yield SM issue slots while polling a per-block completion flag."""
    llvm.inline_asm(
        None,
        [Int32(nanoseconds).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
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
    """Return ``(b_stage_bytes, a_off, fold_off, total_words)`` for non-persistent GEMM.

    Output transpose aliases offset 0 after epilogue barrier 1 (the K-loop has
    completely finished), while ``fold_off`` sits strictly disjoint past the A
    ring so warp_row 1 never races with warp_row 0's final K-loop SMEM reads.
    The 16-byte zero line aliases ``fold_off`` (read only during the K-loop,
    overwritten only in the epilogue) so ``fc2_kernel`` stays at <= 32 KB even
    at ``max_bits = 6``.
    """
    cta_n_blocks = tile_n // 16
    b_stage_bytes = 8 * cta_n_blocks * 32 * max_bits
    a_off = stages * b_stage_bytes
    fold_off = a_off + stages * _A_SLAB_BYTES_FULL
    tb_n_warps = 2 if tile_n == 128 else 1
    n_frags = 2 if tile_n == 32 else 4
    fold_bytes = tb_n_warps * 32 * n_frags * 16
    out_bytes = 8 * (tile_n + 8) * 2
    total_bytes = max(fold_off + fold_bytes, out_bytes)
    total_bytes = ((total_bytes + 15) // 16) * 16
    return b_stage_bytes, a_off, fold_off, total_bytes // 4


def _k4_persistent_smem_layout(
    max_bits: int,
    stages: int,
    a_slab_bytes: int,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(b_stage_bytes, a_off, fold_off, zero16_off, ctrl_off, total_words)``
    for the unified ``tile_n = 64, cta_threads = 128`` persistent kernel.

    ``zero16_off`` (16 B) and ``ctrl_off`` (20 B: 4 dispatch words + 1
    ``is_last`` word) are strictly disjoint from B, A, fold, and output
    transpose so persistent tasks never race on control or zero-row reads.
    """
    cta_n_blocks = 4  # tile_n = 64
    b_stage_bytes = 8 * cta_n_blocks * 32 * max_bits
    a_off = stages * b_stage_bytes
    fold_off = a_off + stages * a_slab_bytes
    # tb_n_warps = 2, n_frags = 2 -> 2 * 32 * 2 * 16 = 2048 B
    fold_bytes = 2 * 32 * 2 * 16
    out_bytes = 8 * (64 + 8) * 2
    zero16_off = max(fold_off + fold_bytes, out_bytes)
    zero16_off = ((zero16_off + 15) // 16) * 16
    ctrl_off = zero16_off + 16
    total_bytes = ((ctrl_off + 20 + 15) // 16) * 16
    return b_stage_bytes, a_off, fold_off, zero16_off, ctrl_off, total_bytes // 4


class W4A16MixedTrellisK4Decode(W4A16MixedTrellisKernel):
    """SM121 (GB10) non-cooperative K4 mixed-Trellis decode/verify kernel."""

    K4_DECODE_VERSION = 3

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
            or self.tier0.trellis_codebook not in (MCG, MUL1)
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
                "W4A16MixedTrellisK4Decode requires SM121 MCG or MUL1 K4 cooperative m8 geometry"
            )
        self.max_bits = max(int(self.tier0.trellis_bits), int(self.tier1.trellis_bits))
        self.is_m1_bucket = int(self.size_m) == 1
        self.persistent_stages = (
            _PERSISTENT_M1_STAGES if self.is_m1_bucket else _PERSISTENT_SMALLM_STAGES
        )
        self.persistent_a_slab_bytes = (
            _A_SLAB_BYTES_M1 if self.is_m1_bucket else _A_SLAB_BYTES_FULL
        )
        self.smallm_fc1_tile_n = 64
        self.medm_fc1_tile_n = 128
        self.fc2_tile_n = 128
        self.has_medm_arm = int(self.size_m) > _K4_SMALLM_MAX_ROWS

        _, _, _, _, _, persistent_words = _k4_persistent_smem_layout(
            self.max_bits,
            self.persistent_stages,
            self.persistent_a_slab_bytes,
        )
        max_words = persistent_words
        specs = [
            (self.smallm_fc1_tile_n, _FC1_SLAB_STAGES),
            (self.fc2_tile_n, _FC2_SLAB_STAGES),
        ]
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
                "sm121_k4_decode_persistent_smallm",
                self.K4_DECODE_VERSION,
                self.is_m1_bucket,
                self.persistent_stages,
                self.persistent_a_slab_bytes,
                self.smallm_fc1_tile_n,
                self.medm_fc1_tile_n,
                self.fc2_tile_n,
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
        if runtime_sms < Int32(1):
            runtime_sms = Int32(1)
        ws_cap = runtime_sms * Int32(4)

        can_persistent = Int32(0)
        if (
            active_m <= Int32(_K4_SMALLM_MAX_ROWS)
            and Int32(_WS_HEADER) + Int32(2) * max_live_blocks <= ws_cap
        ):
            can_persistent = Int32(1)

        if can_persistent != Int32(0):
            persistent_ctas = runtime_sms * Int32(_PERSISTENT_CTAS_PER_SM)
            self.persistent_smallm_kernel(
                rotation_input,
                rotation_gate,
                rotation_up,
                gate_suh,
                up_suh,
                t0_w13,
                t0_w2,
                t0_w13_global,
                t0_w2_global,
                t1_w13,
                t1_w2,
                t1_w13_global,
                t1_w2_global,
                fc1,
                activated,
                fc2,
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
                tier0_fc2_experts,
                tier1_fc2_experts,
                active_m,
                max_live_blocks,
                persistent_ctas,
            ).launch(
                grid=(persistent_ctas, 1, 1),
                block=[128, 1, 1],
                stream=stream,
            )
        else:
            do_fused_act = Int32(0)
            if max_live_blocks <= ws_cap:
                do_fused_act = Int32(1)

            act_units = live_routes * Int32(self.intermediate_size // 128)
            act_ctas = (act_units + Int32(3)) // Int32(4)
            if act_ctas < Int32(1):
                act_ctas = Int32(1)

            smallm_fc1_n_tiles = (2 * self.intermediate_size) // self.smallm_fc1_tile_n
            smallm_fc1_threads = 128 if self.smallm_fc1_tile_n == 128 else 64
            fc2_n_tiles = self.hidden_size // self.fc2_tile_n
            fc2_threads = 128 if self.fc2_tile_n == 128 else 64

            if cutlass.const_expr(not self.has_medm_arm):
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
                if active_m <= Int32(_K4_SMALLM_MAX_ROWS):
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
    # Helper methods shared by persistent and non-persistent GEMM kernels
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
        b_l2_policy: Uint64,
    ):
        cta_n_blocks = tile_n // 16
        chunks_per_k16 = cta_n_blocks * 8
        iters = (8 * chunks_per_k16) // cta_threads
        slab_addr = b_base_addr + Int64(slab) * b_slab_step_bytes
        # Materialize the per-thread shared B base through shuffle_sync so ptxas
        # keeps [R + imm] on LDGSTS with an L2 cache-hint descriptor in UR.
        b_dst_tid = cute.arch.shuffle_sync(b_stage_base + tid * Int32(16), lane)

        if use_evict_first != Int32(0):
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
        b_tensor_base: Int64,
        b_stage_base: Int32,
        b_tile_base_u32: Int64,
        b_k16_stride_u32: Int64,
        slab: Int32,
        tid: Int32,
        lane: Int32,
        use_evict_first: Int32,
        b_l2_policy: Uint64,
    ):
        cta_n_blocks = tile_n // 16
        chunks_per_k16 = cta_n_blocks * 2 * bits
        total_chunks = 8 * chunks_per_k16
        iters = (total_chunks + cta_threads - 1) // cta_threads
        exact = (total_chunks % cta_threads) == 0
        slab_base_u32 = b_tile_base_u32 + Int64(slab) * Int64(8) * b_k16_stride_u32
        b_dst_tid = cute.arch.shuffle_sync(b_stage_base + tid * Int32(16), lane)

        if use_evict_first != Int32(0):
            for i in cutlass.range_constexpr(iters):
                chunk = Int32(i * cta_threads) + tid
                k16 = chunk // Int32(chunks_per_k16)
                in_k16 = chunk - k16 * Int32(chunks_per_k16)
                dst = b_dst_tid + Int32(i * cta_threads * 16)
                src_addr = b_tensor_base + (
                    slab_base_u32
                    + Int64(k16) * b_k16_stride_u32
                    + Int64(in_k16 * 4)
                ) * Int64(4)
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
                src_addr = b_tensor_base + (
                    slab_base_u32
                    + Int64(k16) * b_k16_stride_u32
                    + Int64(in_k16 * 4)
                ) * Int64(4)
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
        b_tensor_base: Int64,
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
        b_l2_policy: Uint64,
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
                b_l2_policy,
            )
        else:
            self._fetch_b_slab_generic(
                bits,
                tile_n,
                cta_threads,
                b_tensor_base,
                b_stage_base,
                b_tile_base_u32,
                b_k16_stride_u32,
                slab,
                tid,
                lane,
                use_evict_first,
                b_l2_policy,
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
    def _consume_slab(
        self,
        bits: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        n_frags: cutlass.Constexpr[int],
        b_stage_base: Int32,
        a_rd_stage0: Int32,
        a_rd_stage1: Int32,
        a_sub_stride: Int32,
        warp_row: Int32,
        w_n: Int32,
        lane: Int32,
        acc,
    ):
        cta_n_blocks = tile_n // 16
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
            sub_off = Int32(0) if cutlass.const_expr(sub == 0) else a_sub_stride
            for kk in cutlass.range_constexpr(2):
                a_addr = (
                    a_rd_stage0 if cutlass.const_expr(kk == 0) else a_rd_stage1
                ) + sub_off
                a0, a1 = ldmatrix_m8n8x2_b16(a_addr)
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
                        if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                            o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                win_a, win_b, 4
                            )
                        else:
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
                            if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                                o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                    win_a, win_b, 5
                                )
                            else:
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
                            if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                                o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                    win_a, win_b, 6
                                )
                            else:
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
                            if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                                o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                    win_a, win_b, bits
                                )
                            else:
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
        n_frags: cutlass.Constexpr[int],
        stages: cutlass.Constexpr[int],
        a_slab_bytes: cutlass.Constexpr[int],
        fused_rot: cutlass.Constexpr[bool],
        pdl_wait: cutlass.Constexpr[bool],
        pdl_signal_after_wait: cutlass.Constexpr[bool],
        enable_rot_wait: cutlass.Constexpr[bool],
        b_stage_bytes: cutlass.Constexpr[int],
        a_off: cutlass.Constexpr[int],
        zero16_off: cutlass.Constexpr[int],
        a_stride_bytes: Int32,
        x_input: cute.Tensor,
        suh_ptr_base: Int64,
        a_tensor_base: Int64,
        b_tensor_base: Int64,
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
        b_l2_policy: Uint64,
        wait_rot: Int32,
        ws_state_addr: Int64,
        acc,
    ):
        a_base = smem_base + Int32(a_off)
        zero16_addr = smem_base + Int32(zero16_off)
        l16 = lane & Int32(15)
        r_ldm = l16 & Int32(7)
        c0_ldm = l16 >> Int32(3)
        k_base = (warp_row << Int32(2)) + c0_ldm

        # Point invalid A rows (r_ldm >= valid_rows) at the shared 16-byte zero
        # line with zero stage/sub strides, eliminating per-CTA A-SMEM zeroing.
        a_rd_base0 = zero16_addr
        a_rd_base1 = zero16_addr
        a_stage_stride = Int32(0)
        a_sub_stride = Int32(0)
        if cutlass.const_expr(a_slab_bytes == _A_SLAB_BYTES_M1):
            if r_ldm == Int32(0) and valid_rows > Int32(0):
                a_rd_base0 = a_base + (k_base << Int32(4))
                a_rd_base1 = a_base + ((k_base + Int32(2)) << Int32(4))
                a_stage_stride = Int32(a_slab_bytes)
                a_sub_stride = Int32(128)
        else:
            if r_ldm < valid_rows:
                a_rd_base0 = a_base + (
                    ((r_ldm << Int32(4)) + (k_base ^ r_ldm)) << Int32(4)
                )
                a_rd_base1 = a_base + (
                    ((r_ldm << Int32(4)) + ((k_base + Int32(2)) ^ r_ldm)) << Int32(4)
                )
                a_stage_stride = Int32(a_slab_bytes)
                a_sub_stride = Int32(128)

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
            b_base_addr = b_tensor_base + (
                b_tile_base_u32
                + Int64(k16_0) * b_k16_stride_u32
                + Int64(in_k16_0 * 4)
            ) * Int64(4)
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
            if cutlass.const_expr(a_slab_bytes == _A_SLAB_BYTES_M1):
                a_swz_off0 = c0 << Int32(4)
                a_valid0 = (row0 == Int32(0) and valid_rows > Int32(0)).to(Int32)
            else:
                a_swz_off0 = ((row0 << Int32(4)) + (c0 ^ row0)) << Int32(4)
                a_valid0 = (row0 < valid_rows).to(Int32)
            a_row_base0 = (
                a_tensor_base
                + Int64(route0) * Int64(a_stride_bytes)
                + Int64(c0 << Int32(4))
            )
            if cutlass.const_expr(cta_threads == 64):
                idx1 = tid + Int32(64)
                row1 = idx1 >> Int32(4)
                c1 = idx1 & Int32(15)
                route1 = cute.arch.shuffle_sync(route_lane, row1)
                a_swz_off1 = ((row1 << Int32(4)) + (c1 ^ row1)) << Int32(4)
                a_row_base1 = (
                    a_tensor_base
                    + Int64(route1) * Int64(a_stride_bytes)
                    + Int64(c1 << Int32(4))
                )
                a_valid1 = (row1 < valid_rows).to(Int32)

        # Prologue
        if cutlass.const_expr(pdl_wait):
            self._fetch_b_slab(
                bits,
                tile_n,
                cta_threads,
                b_tensor_base,
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
                b_l2_policy,
            )
            if k_slabs > Int32(1):
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_tensor_base,
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
                    b_l2_policy,
                )
            cute.arch.cp_async_commit_group()

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
                    a_base + Int32(a_slab_bytes),
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
        elif cutlass.const_expr(enable_rot_wait):
            if wait_rot != Int32(0):
                # Prefetch initial cold B slabs while Phase 0 rotates this block.
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_tensor_base,
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
                    b_l2_policy,
                )
                if k_slabs > Int32(1):
                    self._fetch_b_slab(
                        bits,
                        tile_n,
                        cta_threads,
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                if cutlass.const_expr(stages == 4):
                    if k_slabs > Int32(2):
                        self._fetch_b_slab(
                            bits,
                            tile_n,
                            cta_threads,
                            b_tensor_base,
                            smem_base + Int32(2 * b_stage_bytes),
                            b_tile_base_u32,
                            b_k16_stride_u32,
                            b_base_addr,
                            b_iter_step_bytes,
                            b_slab_step_bytes,
                            Int32(2),
                            tid,
                            lane,
                            use_evict_first,
                            b_l2_policy,
                        )
                cute.arch.cp_async_commit_group()

                if tid == Int32(0):
                    st_v = _ld_global_acquire_gpu_i32(ws_state_addr)
                    while (st_v & Int32(_ROT_MASK)) != Int32(_ROT_READY_VAL):
                        _nanosleep(Int32(64))
                        st_v = _ld_global_acquire_gpu_i32(ws_state_addr)
                cute.arch.sync_threads()

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
                        a_base + Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stages == 4):
                        if k_slabs > Int32(2):
                            self._fetch_a_prerot_slab(
                                cta_threads,
                                a_base + Int32(2 * a_slab_bytes),
                                a_row_base0,
                                a_swz_off0,
                                a_valid0,
                                a_row_base1,
                                a_swz_off1,
                                a_valid1,
                                Int32(2),
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(2)
                        else:
                            cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            else:
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_tensor_base,
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
                    b_l2_policy,
                )
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
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stages == 4):
                        if k_slabs > Int32(2):
                            self._fetch_b_slab(
                                bits,
                                tile_n,
                                cta_threads,
                                b_tensor_base,
                                smem_base + Int32(2 * b_stage_bytes),
                                b_tile_base_u32,
                                b_k16_stride_u32,
                                b_base_addr,
                                b_iter_step_bytes,
                                b_slab_step_bytes,
                                Int32(2),
                                tid,
                                lane,
                                use_evict_first,
                                b_l2_policy,
                            )
                            self._fetch_a_prerot_slab(
                                cta_threads,
                                a_base + Int32(2 * a_slab_bytes),
                                a_row_base0,
                                a_swz_off0,
                                a_valid0,
                                a_row_base1,
                                a_swz_off1,
                                a_valid1,
                                Int32(2),
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(2)
                        else:
                            cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
        else:
            self._fetch_b_slab(
                bits,
                tile_n,
                cta_threads,
                b_tensor_base,
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
                b_l2_policy,
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
                    b_tensor_base,
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
                    b_l2_policy,
                )
                if cutlass.const_expr(not fused_rot):
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                cute.arch.cp_async_commit_group()

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
                        a_base + Int32(a_slab_bytes),
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

        # Main K-loop
        if cutlass.const_expr(stages == 2):
            stage = Int32(0)
            s = Int32(0)
            while s < k_slabs:
                a_stage_off = stage * a_stage_stride
                self._consume_slab(
                    bits,
                    tile_n,
                    n_frags,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_rd_base0 + a_stage_off,
                    a_rd_base1 + a_stage_off,
                    a_sub_stride,
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
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                    if cutlass.const_expr(not fused_rot):
                        self._fetch_a_prerot_slab(
                            cta_threads,
                            a_base + stage * Int32(a_slab_bytes),
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
                            a_base + stage * Int32(a_slab_bytes),
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
        elif cutlass.const_expr(stages == 4):
            stage = Int32(0)
            s = Int32(0)
            while s < k_slabs:
                wr_stage = (stage + Int32(3)) & Int32(3)
                next3 = s + Int32(3)
                if next3 < k_slabs:
                    self._fetch_b_slab(
                        bits,
                        tile_n,
                        cta_threads,
                        b_tensor_base,
                        smem_base + wr_stage * Int32(b_stage_bytes),
                        b_tile_base_u32,
                        b_k16_stride_u32,
                        b_base_addr,
                        b_iter_step_bytes,
                        b_slab_step_bytes,
                        next3,
                        tid,
                        lane,
                        use_evict_first,
                        b_l2_policy,
                    )
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + wr_stage * Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        next3,
                    )
                    cute.arch.cp_async_commit_group()

                a_stage_off = stage * a_stage_stride
                self._consume_slab(
                    bits,
                    tile_n,
                    n_frags,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_rd_base0 + a_stage_off,
                    a_rd_base1 + a_stage_off,
                    a_sub_stride,
                    warp_row,
                    w_n,
                    lane,
                    acc,
                )

                if next3 < k_slabs:
                    cute.arch.cp_async_wait_group(2)
                    cute.arch.sync_threads()
                elif s + Int32(2) < k_slabs:
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                elif s + Int32(1) < k_slabs:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                stage = (stage + Int32(1)) & Int32(3)
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
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                    if cutlass.const_expr(not fused_rot):
                        self._fetch_a_prerot_slab(
                            cta_threads,
                            a_base + wr_stage * Int32(a_slab_bytes),
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
                            a_base + wr_stage * Int32(a_slab_bytes),
                            tok_lane,
                            valid_rows,
                            next2,
                            warp_id,
                            lane,
                        )

                a_stage_off = stage * a_stage_stride
                self._consume_slab(
                    bits,
                    tile_n,
                    n_frags,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_rd_base0 + a_stage_off,
                    a_rd_base1 + a_stage_off,
                    a_sub_stride,
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
        n_frags: cutlass.Constexpr[int],
        fold_off: cutlass.Constexpr[int],
        c_tensor_base: Int64,
        c_stride_elem: Int32,
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
                st_global_v4_u32(c_tensor_base + out_elem * Int64(2), q0, q1, q2, q3)

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
        fc1_base: Int64,
        activated_base: Int64,
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
            gp01, gp23 = _ld_global_cg_v2_u32(fc1_base + g_off * Int64(2))
            up01, up23 = _ld_global_cg_v2_u32(fc1_base + u_off * Int64(2))
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
            st_global_v2_u32(activated_base + out_off * Int64(2), p01, p23)
            u += Int32(n_warps)

    # ------------------------------------------------------------------
    # Single-kernel persistent small-M (rot + FC1 + act + FC2)
    # ------------------------------------------------------------------

    @cute.kernel
    def persistent_smallm_kernel(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
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
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        max_live_blocks: cutlass.Int32,
        persistent_ctas: cutlass.Int32,
    ):
        tile_n = 64
        cta_threads = 128
        n_frags = 2
        stages = self.persistent_stages
        a_slab_bytes = self.persistent_a_slab_bytes
        (
            b_stage_bytes,
            a_off,
            fold_off,
            zero16_off,
            ctrl_off,
            smem_words,
        ) = _k4_persistent_smem_layout(self.max_bits, stages, a_slab_bytes)

        tidx, _, _ = cute.arch.thread_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        w_n = warp_id & Int32(1)
        warp_row = warp_id >> Int32(1)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, smem_words], 128
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        if tid == Int32(0):
            st_shared_v4_u32(
                smem_base + Int32(zero16_off),
                Uint32(0),
                Uint32(0),
                Uint32(0),
                Uint32(0),
            )

        actual_blocks = packed_route_count[Int32(0)].to(Int32) >> Int32(3)
        if actual_blocks > max_live_blocks:
            actual_blocks = max_live_blocks
        if actual_blocks < Int32(0):
            actual_blocks = Int32(0)

        ws_base = get_ptr_as_int64(workspace, Int64(0))
        ws_state_base = ws_base + Int64(_WS_HEADER * 4)
        ws_ready_base = ws_state_base + Int64(max_live_blocks) * Int64(4)

        if actual_blocks > Int32(0):
            total_experts = tier0_num_experts + tier1_num_experts
            live_routes = active_m * Int32(self.top_k)

            fc1_half_tiles = self.intermediate_size // tile_n
            fc1_n_tiles = 2 * fc1_half_tiles
            fc2_n_tiles = self.hidden_size // tile_n
            rot_nblk = self.hidden_size // 128
            blks_per_chunk = (rot_nblk + _ROT_SPLITS - 1) // _ROT_SPLITS

            total_rot_tasks = actual_blocks * Int32(_ROT_SPLITS)
            total_fc1_tasks = actual_blocks * Int32(fc1_n_tiles)
            total_fc2_tasks = actual_blocks * Int32(fc2_n_tiles)

            rot_gate_base = get_ptr_as_int64(rotation_gate, Int64(0))
            rot_up_base = get_ptr_as_int64(rotation_up, Int64(0))
            fc1_base = get_ptr_as_int64(fc1, Int64(0))
            activated_base = get_ptr_as_int64(activated, Int64(0))
            fc2_base = get_ptr_as_int64(fc2, Int64(0))
            t0_w13_base = get_ptr_as_int64(t0_w13, Int64(0))
            t0_w2_base = get_ptr_as_int64(t0_w2, Int64(0))
            t1_w13_base = get_ptr_as_int64(t1_w13, Int64(0))
            t1_w2_base = get_ptr_as_int64(t1_w2, Int64(0))
            t0_w13_plane_u32 = Int64(cute.size(t0_w13)) // Int64(2)
            t1_w13_plane_u32 = Int64(cute.size(t1_w13)) // Int64(2)
            b_l2_policy = create_l2_evict_first_policy()

            phase = Int32(0)
            running = Int32(1)
            while running != Int32(0):
                if tid == Int32(0):
                    action_d = Int32(_ACT_EXIT)
                    arg0_d = Int32(0)
                    arg1_d = Int32(0)
                    arg2_d = Int32(0)
                    got_work = Int32(0)
                    while got_work == Int32(0):
                        if phase == Int32(0):
                            t = _atomic_add_acq_rel_gpu_i32(
                                ws_base + Int64(_WS_ROT_CURSOR * 4), Int32(1)
                            )
                            if t < total_rot_tasks:
                                rb_t = t >> Int32(2)
                                chk_t = t & Int32(3)
                                ce_t = block_expert_ids[rb_t].to(Int32)
                                valid_blk = (
                                    ce_t >= Int32(0) and ce_t < total_experts
                                ).to(Int32)
                                if valid_blk != Int32(0):
                                    r0_t = packed_route_indices[
                                        rb_t * Int32(8)
                                    ].to(Int32)
                                    if r0_t < Int32(0) or r0_t >= live_routes:
                                        valid_blk = Int32(0)
                                    elif cutlass.const_expr(self.compact_weights):
                                        lw_t = descriptor_map[
                                            Int64(3) * Int64(total_experts)
                                            + Int64(ce_t)
                                        ].to(Int32)
                                        if lw_t <= Int32(0):
                                            valid_blk = Int32(0)
                                if valid_blk != Int32(0):
                                    action_d = Int32(_ACT_ROT)
                                    arg0_d = rb_t
                                    arg1_d = chk_t
                                    got_work = Int32(1)
                                else:
                                    _atomic_add_acq_rel_gpu_i32(
                                        ws_state_base + Int64(rb_t) * Int64(4),
                                        Int32(_ROT_DONE_STEP),
                                    )
                            else:
                                phase = Int32(1)
                        elif phase == Int32(1):
                            t = _atomic_add_acq_rel_gpu_i32(
                                ws_base + Int64(_WS_FC1_CURSOR * 4), Int32(1)
                            )
                            if t < total_fc1_tasks:
                                rb_t = t // Int32(fc1_n_tiles)
                                nt_t = t - rb_t * Int32(fc1_n_tiles)
                                ce_t = block_expert_ids[rb_t].to(Int32)
                                valid_blk = (
                                    ce_t >= Int32(0) and ce_t < total_experts
                                ).to(Int32)
                                lw_t = Int32(self.intermediate_size)
                                if valid_blk != Int32(0):
                                    r0_t = packed_route_indices[
                                        rb_t * Int32(8)
                                    ].to(Int32)
                                    if r0_t < Int32(0) or r0_t >= live_routes:
                                        valid_blk = Int32(0)
                                    elif cutlass.const_expr(self.compact_weights):
                                        lw_t = descriptor_map[
                                            Int64(3) * Int64(total_experts)
                                            + Int64(ce_t)
                                        ].to(Int32)
                                        if lw_t <= Int32(0):
                                            valid_blk = Int32(0)
                                active_tile = valid_blk
                                if cutlass.const_expr(self.compact_weights):
                                    if active_tile != Int32(0):
                                        loc_nt = nt_t
                                        if loc_nt >= Int32(fc1_half_tiles):
                                            loc_nt -= Int32(fc1_half_tiles)
                                        if loc_nt * Int32(tile_n) >= lw_t:
                                            active_tile = Int32(0)
                                if active_tile != Int32(0):
                                    st_val = _ld_global_acquire_gpu_i32(
                                        ws_state_base + Int64(rb_t) * Int64(4)
                                    )
                                    rot_done = (
                                        (st_val & Int32(_ROT_MASK))
                                        == Int32(_ROT_READY_VAL)
                                    ).to(Int32)
                                    action_d = Int32(_ACT_FC1)
                                    arg0_d = rb_t
                                    arg1_d = nt_t
                                    arg2_d = rot_done
                                    got_work = Int32(1)
                                else:
                                    old_st = _atomic_add_acq_rel_gpu_i32(
                                        ws_state_base + Int64(rb_t) * Int64(4),
                                        Int32(1),
                                    )
                                    if (old_st & Int32(_FC1_CNT_MASK)) + Int32(
                                        1
                                    ) == Int32(fc1_n_tiles):
                                        if valid_blk != Int32(0):
                                            action_d = Int32(_ACT_ACT_ONLY)
                                            arg0_d = rb_t
                                            got_work = Int32(1)
                                        else:
                                            slot_t = _atomic_add_acq_rel_gpu_i32(
                                                ws_base + Int64(_WS_READY_COUNT * 4),
                                                Int32(1),
                                            )
                                            _st_global_release_gpu_i32(
                                                ws_ready_base
                                                + Int64(slot_t) * Int64(4),
                                                rb_t + Int32(1),
                                            )
                            else:
                                phase = Int32(2)
                        else:
                            t = _atomic_add_acq_rel_gpu_i32(
                                ws_base + Int64(_WS_FC2_CURSOR * 4), Int32(1)
                            )
                            if t < total_fc2_tasks:
                                slot_t = t // Int32(fc2_n_tiles)
                                nt_t = t - slot_t * Int32(fc2_n_tiles)
                                ready_addr = ws_ready_base + Int64(slot_t) * Int64(4)
                                rval = _ld_global_acquire_gpu_i32(ready_addr)
                                while rval == Int32(0):
                                    _nanosleep(Int32(128))
                                    rval = _ld_global_acquire_gpu_i32(ready_addr)
                                rb_t = rval - Int32(1)
                                ce_t = block_expert_ids[rb_t].to(Int32)
                                valid_blk = (
                                    ce_t >= Int32(0) and ce_t < total_experts
                                ).to(Int32)
                                if valid_blk != Int32(0):
                                    r0_t = packed_route_indices[
                                        rb_t * Int32(8)
                                    ].to(Int32)
                                    if r0_t < Int32(0) or r0_t >= live_routes:
                                        valid_blk = Int32(0)
                                    elif cutlass.const_expr(self.compact_weights):
                                        lw_t = descriptor_map[
                                            Int64(3) * Int64(total_experts)
                                            + Int64(ce_t)
                                        ].to(Int32)
                                        if lw_t <= Int32(0):
                                            valid_blk = Int32(0)
                                if valid_blk != Int32(0):
                                    action_d = Int32(_ACT_FC2)
                                    arg0_d = rb_t
                                    arg1_d = nt_t
                                    got_work = Int32(1)
                            else:
                                action_d = Int32(_ACT_EXIT)
                                got_work = Int32(1)
                    st_shared_i32(smem_base + Int32(ctrl_off + 0), action_d)
                    st_shared_i32(smem_base + Int32(ctrl_off + 4), arg0_d)
                    st_shared_i32(smem_base + Int32(ctrl_off + 8), arg1_d)
                    st_shared_i32(smem_base + Int32(ctrl_off + 12), arg2_d)
                cute.arch.sync_threads()

                action = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 0))
                rb = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 4))
                arg1 = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 8))
                arg2 = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 12))

                if action == Int32(_ACT_EXIT):
                    running = Int32(0)
                elif action == Int32(_ACT_ROT):
                    combined_expert = block_expert_ids[rb].to(Int32)
                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")
                    route_lane = Int32(0)
                    if is_valid != Int32(0):
                        route_lane = r_lane
                    tok_lane = route_lane // Int32(self.top_k)

                    blk_start = arg1 * Int32(blks_per_chunk)
                    blk_end = blk_start + Int32(blks_per_chunk)
                    if blk_end > Int32(rot_nblk):
                        blk_end = Int32(rot_nblk)

                    blk = blk_start + warp_id
                    while blk < blk_end:
                        col0 = blk * Int32(128) + lane * Int32(4)
                        s_off = Int64(combined_expert) * Int64(
                            self.hidden_size
                        ) + Int64(col0)
                        gw0, gw1 = ld_global_nc_v2_u32(
                            get_ptr_as_int64(gate_suh, s_off)
                        )
                        uw0, uw1 = ld_global_nc_v2_u32(
                            get_ptr_as_int64(up_suh, s_off)
                        )
                        r_idx = Int32(0)
                        while r_idx < valid_rows:
                            route_r = cute.arch.shuffle_sync(route_lane, r_idx)
                            tok_r = cute.arch.shuffle_sync(tok_lane, r_idx)
                            x_off = Int64(tok_r) * Int64(self.hidden_size) + Int64(
                                col0
                            )
                            out_off = Int64(route_r) * Int64(self.hidden_size) + Int64(
                                col0
                            )
                            xw0, xw1 = ld_global_nc_v2_u32(
                                get_ptr_as_int64(rotation_input, x_off)
                            )
                            xh0 = cvt_bf16x2_to_f16x2_via_f32(xw0)
                            xh1 = cvt_bf16x2_to_f16x2_via_f32(xw1)

                            gv0, gv1 = f16x2_to_f32x2(half2_mul(xh0, gw0))
                            gv2, gv3 = f16x2_to_f32x2(half2_mul(xh1, gw1))
                            gh0, gh1, gh2, gh3 = self.driver._had128_quad(
                                gv0, gv1, gv2, gv3, lane
                            )
                            st_global_v2_u32(
                                rot_gate_base + out_off * Int64(2),
                                pack_f32x2_to_f16x2(gh0, gh1),
                                pack_f32x2_to_f16x2(gh2, gh3),
                            )

                            uv0, uv1 = f16x2_to_f32x2(half2_mul(xh0, uw0))
                            uv2, uv3 = f16x2_to_f32x2(half2_mul(xh1, uw1))
                            uh0, uh1, uh2, uh3 = self.driver._had128_quad(
                                uv0, uv1, uv2, uv3, lane
                            )
                            st_global_v2_u32(
                                rot_up_base + out_off * Int64(2),
                                pack_f32x2_to_f16x2(uh0, uh1),
                                pack_f32x2_to_f16x2(uh2, uh3),
                            )
                            r_idx += Int32(1)
                        blk += Int32(4)

                    cute.arch.sync_threads()
                    if tid == Int32(0):
                        _atomic_add_acq_rel_gpu_i32(
                            ws_state_base + Int64(rb) * Int64(4),
                            Int32(_ROT_DONE_STEP),
                        )
                elif action == Int32(_ACT_ACT_ONLY):
                    combined_expert = block_expert_ids[rb].to(Int32)
                    live_width = Int32(self.intermediate_size)
                    if cutlass.const_expr(self.compact_weights):
                        live_width = descriptor_map[
                            Int64(3) * Int64(total_experts) + Int64(combined_expert)
                        ].to(Int32)
                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")
                    route_lane = Int32(0)
                    if is_valid != Int32(0):
                        route_lane = r_lane
                    if valid_rows > Int32(0) and live_width > Int32(0):
                        self._run_fused_act_block(
                            cta_threads,
                            fc1_base,
                            activated_base,
                            intermediate_rotations,
                            combined_expert,
                            live_width,
                            route_lane,
                            valid_rows,
                            warp_id,
                            lane,
                        )
                    cute.arch.sync_threads()
                    if tid == Int32(0):
                        slot = _atomic_add_acq_rel_gpu_i32(
                            ws_base + Int64(_WS_READY_COUNT * 4), Int32(1)
                        )
                        _st_global_release_gpu_i32(
                            ws_ready_base + Int64(slot) * Int64(4),
                            rb + Int32(1),
                        )
                else:
                    # Unified FC1 / FC2 GEMM body (tile_n = 64, cta_threads = 128).
                    is_fc1_rt = (action == Int32(_ACT_FC1)).to(Int32)
                    output_n_tile = arg1
                    combined_expert = block_expert_ids[rb].to(Int32)

                    descriptor_row = Int32(2)
                    proj = Int32(0)
                    local_n_tile = output_n_tile
                    if is_fc1_rt != Int32(0):
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

                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")
                    route_lane = Int32(0)
                    if is_valid != Int32(0):
                        route_lane = r_lane

                    descriptor = descriptor_map[
                        Int64(descriptor_row) * Int64(total_experts)
                        + Int64(combined_expert)
                    ].to(Int32)

                    if valid_rows > Int32(0) and descriptor >= Int32(0):
                        tier = descriptor >> Int32(self.descriptor_bits)
                        local_expert = descriptor & Int32(
                            (1 << self.descriptor_bits) - 1
                        )

                        t0_lim = tier0_fc2_experts
                        t1_lim = tier1_fc2_experts
                        if is_fc1_rt != Int32(0):
                            if proj != Int32(0):
                                t0_lim = tier0_up_experts
                                t1_lim = tier1_up_experts
                            else:
                                t0_lim = tier0_gate_experts
                                t1_lim = tier1_gate_experts

                        is_t0 = tier == Int32(0) and local_expert < t0_lim
                        is_t1 = tier == Int32(1) and local_expert < t1_lim

                        if is_t0 or is_t1:
                            use_evict_first = Int32(1)
                            if rb + Int32(1) < actual_blocks:
                                if (
                                    block_expert_ids[rb + Int32(1)].to(Int32)
                                    == combined_expert
                                ):
                                    use_evict_first = Int32(0)

                            a_tensor_base = activated_base
                            a_stride_bytes = Int32(self.intermediate_size * 2)
                            c_tensor_base = fc2_base
                            c_stride_elem = Int32(self.hidden_size)
                            k_slabs = live_width >> Int32(7)
                            wait_rot = Int32(0)
                            if is_fc1_rt != Int32(0):
                                a_tensor_base = rot_gate_base
                                if proj != Int32(0):
                                    a_tensor_base = rot_up_base
                                a_stride_bytes = Int32(self.hidden_size * 2)
                                c_tensor_base = fc1_base
                                c_stride_elem = Int32(2 * self.intermediate_size)
                                k_slabs = Int32(self.hidden_size // 128)
                                if arg2 == Int32(0):
                                    wait_rot = Int32(1)

                            acc = [
                                cute.make_rmem_tensor((1,), cutlass.Float32)
                                for _ in range(4 * n_frags)
                            ]
                            for idx in cutlass.range_constexpr(4 * n_frags):
                                acc[idx].fill(0.0)

                            global_scale_f32 = cutlass.Float32(1.0)
                            cta_n_blocks = tile_n // 16
                            ws_state_addr = ws_state_base + Int64(rb) * Int64(4)

                            if is_t0:
                                bits0 = self.tier0.trellis_bits
                                tile_u32_0 = 8 * bits0
                                b_base_0 = t0_w2_base
                                b_k16_stride_0 = Int64(
                                    (self.hidden_size // 16) * tile_u32_0
                                )
                                b_tile_base_0 = Int64(0)
                                if is_fc1_rt != Int32(0):
                                    b_base_0 = t0_w13_base
                                    global_scale_f32 = t0_w13_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_k16_stride_0 = (
                                            live_width.to(Int64) >> Int64(4)
                                        ) * Int64(tile_u32_0)
                                        b_tile_base_0 = weight_offset + Int64(
                                            local_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_0)
                                    else:
                                        half_n16 = self.intermediate_size // 16
                                        proj_expert_u32 = (
                                            (self.hidden_size // 16)
                                            * half_n16
                                            * tile_u32_0
                                        )
                                        b_k16_stride_0 = Int64(half_n16 * tile_u32_0)
                                        b_tile_base_0 = (
                                            Int64(proj) * t0_w13_plane_u32
                                            + Int64(local_expert)
                                            * Int64(proj_expert_u32)
                                            + Int64(local_n_tile)
                                            * Int64(cta_n_blocks * tile_u32_0)
                                        )
                                else:
                                    global_scale_f32 = t0_w2_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_tile_base_0 = weight_offset + Int64(
                                            output_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_0)
                                    else:
                                        n16 = self.hidden_size // 16
                                        expert_u32 = (
                                            (self.intermediate_size // 16)
                                            * n16
                                            * tile_u32_0
                                        )
                                        b_tile_base_0 = Int64(local_expert) * Int64(
                                            expert_u32
                                        ) + Int64(output_n_tile) * Int64(
                                            cta_n_blocks * tile_u32_0
                                        )

                                self._run_gemm_kloop(
                                    bits0,
                                    tile_n,
                                    cta_threads,
                                    n_frags,
                                    stages,
                                    a_slab_bytes,
                                    False,
                                    False,
                                    False,
                                    True,
                                    b_stage_bytes,
                                    a_off,
                                    zero16_off,
                                    a_stride_bytes,
                                    rotation_input,
                                    Int64(0),
                                    a_tensor_base,
                                    b_base_0,
                                    b_tile_base_0,
                                    b_k16_stride_0,
                                    k_slabs,
                                    route_lane,
                                    Int32(0),
                                    valid_rows,
                                    smem_base,
                                    tid,
                                    warp_id,
                                    warp_row,
                                    w_n,
                                    lane,
                                    use_evict_first,
                                    b_l2_policy,
                                    wait_rot,
                                    ws_state_addr,
                                    acc,
                                )
                            else:
                                bits1 = self.tier1.trellis_bits
                                tile_u32_1 = 8 * bits1
                                b_base_1 = t1_w2_base
                                b_k16_stride_1 = Int64(
                                    (self.hidden_size // 16) * tile_u32_1
                                )
                                b_tile_base_1 = Int64(0)
                                if is_fc1_rt != Int32(0):
                                    b_base_1 = t1_w13_base
                                    global_scale_f32 = t1_w13_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_k16_stride_1 = (
                                            live_width.to(Int64) >> Int64(4)
                                        ) * Int64(tile_u32_1)
                                        b_tile_base_1 = weight_offset + Int64(
                                            local_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_1)
                                    else:
                                        half_n16 = self.intermediate_size // 16
                                        proj_expert_u32 = (
                                            (self.hidden_size // 16)
                                            * half_n16
                                            * tile_u32_1
                                        )
                                        b_k16_stride_1 = Int64(half_n16 * tile_u32_1)
                                        b_tile_base_1 = (
                                            Int64(proj) * t1_w13_plane_u32
                                            + Int64(local_expert)
                                            * Int64(proj_expert_u32)
                                            + Int64(local_n_tile)
                                            * Int64(cta_n_blocks * tile_u32_1)
                                        )
                                else:
                                    global_scale_f32 = t1_w2_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_tile_base_1 = weight_offset + Int64(
                                            output_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_1)
                                    else:
                                        n16 = self.hidden_size // 16
                                        expert_u32 = (
                                            (self.intermediate_size // 16)
                                            * n16
                                            * tile_u32_1
                                        )
                                        b_tile_base_1 = Int64(local_expert) * Int64(
                                            expert_u32
                                        ) + Int64(output_n_tile) * Int64(
                                            cta_n_blocks * tile_u32_1
                                        )

                                self._run_gemm_kloop(
                                    bits1,
                                    tile_n,
                                    cta_threads,
                                    n_frags,
                                    stages,
                                    a_slab_bytes,
                                    False,
                                    False,
                                    False,
                                    True,
                                    b_stage_bytes,
                                    a_off,
                                    zero16_off,
                                    a_stride_bytes,
                                    rotation_input,
                                    Int64(0),
                                    a_tensor_base,
                                    b_base_1,
                                    b_tile_base_1,
                                    b_k16_stride_1,
                                    k_slabs,
                                    route_lane,
                                    Int32(0),
                                    valid_rows,
                                    smem_base,
                                    tid,
                                    warp_id,
                                    warp_row,
                                    w_n,
                                    lane,
                                    use_evict_first,
                                    b_l2_policy,
                                    wait_rot,
                                    ws_state_addr,
                                    acc,
                                )

                            self._epilogue_store(
                                tile_n,
                                n_frags,
                                fold_off,
                                c_tensor_base,
                                c_stride_elem,
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

                    cute.arch.sync_threads()
                    if is_fc1_rt != Int32(0):
                        if tid == Int32(0):
                            old = _atomic_add_acq_rel_gpu_i32(
                                ws_state_base + Int64(rb) * Int64(4), Int32(1)
                            )
                            is_last = (
                                (old & Int32(_FC1_CNT_MASK)) + Int32(1)
                                == Int32(fc1_n_tiles)
                            ).to(Int32)
                            st_shared_i32(smem_base + Int32(ctrl_off + 16), is_last)
                        cute.arch.sync_threads()
                        if (
                            ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 16))
                            != Int32(0)
                        ):
                            if valid_rows > Int32(0) and live_width > Int32(0):
                                self._run_fused_act_block(
                                    cta_threads,
                                    fc1_base,
                                    activated_base,
                                    intermediate_rotations,
                                    combined_expert,
                                    live_width,
                                    route_lane,
                                    valid_rows,
                                    warp_id,
                                    lane,
                                )
                            cute.arch.sync_threads()
                            if tid == Int32(0):
                                slot = _atomic_add_acq_rel_gpu_i32(
                                    ws_base + Int64(_WS_READY_COUNT * 4), Int32(1)
                                )
                                _st_global_release_gpu_i32(
                                    ws_ready_base + Int64(slot) * Int64(4),
                                    rb + Int32(1),
                                )

        # Final CTA resets workspace back to zeros for CUDA graph replay.
        cute.arch.sync_threads()
        if tid == Int32(0):
            exited = _atomic_add_acq_rel_gpu_i32(
                ws_base + Int64(_WS_EXIT_COUNT * 4), Int32(1)
            )
            is_last_cta = (exited + Int32(1) == persistent_ctas).to(Int32)
            st_shared_i32(smem_base + Int32(ctrl_off + 16), is_last_cta)
        cute.arch.sync_threads()
        if ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 16)) != Int32(0):
            ws_used = Int32(_WS_HEADER) + Int32(2) * max_live_blocks
            idx = tid
            while idx < ws_used:
                _st_global_relaxed_gpu_i32(ws_base + Int64(idx) * Int64(4), Int32(0))
                idx += Int32(cta_threads)
        cute.arch.griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------
    # Non-persistent GEMM body (m12..m80 PDL paths)
    # ------------------------------------------------------------------

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
                cute.struct.MemRange[cutlass.Uint32, smem_words], 128
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
                                # Zero 16-byte sentinel at fold_off before prologue sync.
                                if tid == Int32(0):
                                    st_shared_v4_u32(
                                        smem_base + Int32(fold_off),
                                        Uint32(0),
                                        Uint32(0),
                                        Uint32(0),
                                        Uint32(0),
                                    )

                                tok_lane = Int32(0)
                                if cutlass.const_expr(fused_rot):
                                    tok_lane = route_lane // Int32(self.top_k)

                                use_evict_first = Int32(1)
                                if rb + Int32(1) < actual_blocks:
                                    if (
                                        block_expert_ids[rb + Int32(1)].to(Int32)
                                        == combined_expert
                                    ):
                                        use_evict_first = Int32(0)

                                b_l2_policy = create_l2_evict_first_policy()

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

                                a_stride_bytes = Int32(
                                    self.hidden_size * 2
                                    if is_fc1
                                    else self.intermediate_size * 2
                                )
                                c_stride_elem = Int32(
                                    2 * self.intermediate_size
                                    if is_fc1
                                    else self.hidden_size
                                )
                                c_tensor_base = get_ptr_as_int64(c_flat, Int64(0))
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
                                        n_frags,
                                        stages,
                                        _A_SLAB_BYTES_FULL,
                                        fused_rot,
                                        pdl_wait,
                                        pdl_signal_after_wait,
                                        False,
                                        b_stage_bytes,
                                        a_off,
                                        fold_off,
                                        a_stride_bytes,
                                        x_or_a_gate,
                                        suh_ptr_base,
                                        a_tensor_base,
                                        get_ptr_as_int64(t0_b, Int64(0)),
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
                                        b_l2_policy,
                                        Int32(0),
                                        Int64(0),
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
                                        n_frags,
                                        stages,
                                        _A_SLAB_BYTES_FULL,
                                        fused_rot,
                                        pdl_wait,
                                        pdl_signal_after_wait,
                                        False,
                                        b_stage_bytes,
                                        a_off,
                                        fold_off,
                                        a_stride_bytes,
                                        x_or_a_gate,
                                        suh_ptr_base,
                                        a_tensor_base,
                                        get_ptr_as_int64(t1_b, Int64(0)),
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
                                        b_l2_policy,
                                        Int32(0),
                                        Int64(0),
                                        acc,
                                    )

                                self._epilogue_store(
                                    tile_n,
                                    n_frags,
                                    fold_off,
                                    c_tensor_base,
                                    c_stride_elem,
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
                                        get_ptr_as_int64(c_flat, Int64(0)),
                                        get_ptr_as_int64(activated, Int64(0)),
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
    # Compiled CUDA kernels (m12..m80 PDL fallback paths)
    # ------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# SM120 (RTX PRO 6000 / Blackwell) EXL3 K4 mixed-Trellis decode kernel
# ----------------------------------------------------------------------

_SM120_FC2_SLAB_STAGES = 3
_SM120_ROT_DONE_STEP = 0x00010000
_SM120_ROT_MASK = 0x7FFF0000
_SM120_FC1_CNT_MASK = 0x0000FFFF


@dsl_user_op
def _shf_r_wrap_b32(
    lo: Uint32,
    hi: Uint32,
    shift: Int32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Single-instruction funnel shift ``((hi << 32) | lo) >> (shift & 31)``."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(lo).ir_value(loc=loc, ip=ip),
                Uint32(hi).ir_value(loc=loc, ip=ip),
                Int32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shf.r.wrap.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packed_dequant_trellis_mcg_fast(
    win_a: Uint32,
    win_b: Uint32,
    bits: int,
    *,
    loc=None,
    ip=None,
):
    """Bit-exact MCG decoder using ``bfe.u32`` / ``prmt.b32`` extraction.

    Matches ``packed_dequant_trellis_to_half2x4`` bit-for-bit for K2..K5 while
    replacing 14 ``shr``/``and`` instructions with 8 single-instruction
    extractions (using ``prmt.b32 ..., 0, 0x4421`` for byte-aligned shift 8).
    """
    bits = int(bits)
    if bits not in (2, 3, 4, 5):
        raise ValueError(f"unsupported fast MCG trellis bitrate {bits}")

    def _extract(dst: str, src: str, shift: int) -> str:
        if shift == 0:
            return f"and.b32 {dst}, {src}, 0xffff;"
        if shift == 8:
            return f"prmt.b32 {dst}, {src}, 0, 0x4421;"
        return f"bfe.u32 {dst}, {src}, {shift}, 16;"

    lines = [
        _extract("w7", "$4", 0),
        _extract("w6", "$4", bits),
        _extract("w5", "$4", 2 * bits),
        _extract("w4", "$4", 3 * bits),
        _extract("w3", "$5", 0),
        _extract("w2", "$5", bits),
        _extract("w1", "$5", 2 * bits),
        _extract("w0", "$5", 3 * bits),
    ]
    for i in range(8):
        lines.append(
            f"mul.lo.u32 w{i}, w{i}, 0xCBAC1FED; "
            f"lop3.b32 w{i}, w{i}, 0x8FFF8FFF, 0x3B603B60, 0x6a;"
        )
    for pair in range(4):
        lines.append(
            f"prmt.b32 lo, w{2 * pair}, w{2 * pair + 1}, 0x5410; "
            f"prmt.b32 hi, w{2 * pair}, w{2 * pair + 1}, 0x7632; "
            f"add.rn.f16x2 ${pair}, lo, hi;"
        )
    asm = (
        "{\n"
        "    .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, lo, hi;\n    "
        + "\n    ".join(lines)
        + "\n}"
    )
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(4)
    )


class W4A16MixedTrellisSM120K4Decode(W4A16MixedTrellisKernel):
    """SM120 (compute 12.0) non-cooperative K4 mixed-Trellis decode kernel.

    Preserves the exact ``v41_exl3_core`` AOT ABI, ``blocks_per_sm``, and
    reported ``shared_memory_bytes`` (``shared_words``) of stock
    ``W4A16MixedTrellisKernel`` while eliminating cooperative grid barriers,
    coarse N-tile wave cliffs, and cold-start atomic contention on wide-SM
    SM120 GPUs.
    """

    SM120_K4_DECODE_VERSION = 1

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
            or self.tier1.trellis_bits not in (5, 6)
            or self.tier0.trellis_codebook not in (MCG, MUL1)
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
                "W4A16MixedTrellisSM120K4Decode requires SM120 MCG or MUL1 K4/K5-K6 m8 geometry"
            )
        self.max_bits = max(int(self.tier0.trellis_bits), int(self.tier1.trellis_bits))
        self.is_m1_bucket = int(self.size_m) == 1
        # K4/K5 fits 4 stages at 3 CTAs/SM on both m1 (23.6 KB) and m2..m16
        # (30.8 KB; 3 * (30.8 + 1) KB <= 101.4 KB). Keep 3 stages for K6 m2+.
        self.persistent_stages = (
            _PERSISTENT_M1_STAGES
            if (self.is_m1_bucket or self.max_bits <= 5)
            else _PERSISTENT_SMALLM_STAGES
        )
        self.persistent_a_slab_bytes = (
            _A_SLAB_BYTES_M1 if self.is_m1_bucket else _A_SLAB_BYTES_FULL
        )
        # 4 H128 blocks per rot chunk -> all 4 warps in a 128-thread CTA run
        # 1 H128 block each with zero idle warps and high SM parallelism.
        rot_nblk = self.hidden_size // 128
        self.rot_splits = max(1, min((rot_nblk + 3) // 4, 16))
        self.smallm_fc1_tile_n = 64
        self.medm_fc1_tile_n = 128
        self.fc2_tile_n = 128
        self.has_medm_arm = int(self.size_m) > _K4_SMALLM_MAX_ROWS
        # Deliberately keep self.shared_words and self.blocks_per_sm unchanged
        # from super().__init__ so AOT manifest discriminators match stock.

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return super().__cache_key__ + (
            (
                "sm120_k4_decode_persistent_smallm",
                self.SM120_K4_DECODE_VERSION,
                self.is_m1_bucket,
                self.persistent_stages,
                self.persistent_a_slab_bytes,
                self.rot_splits,
                self.smallm_fc1_tile_n,
                self.medm_fc1_tile_n,
                self.fc2_tile_n,
                self.has_medm_arm,
            ),
        )

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

        live_routes = active_m * Int32(self.top_k)
        active_experts_bound = live_routes
        if active_experts_bound > route_num_experts:
            active_experts_bound = route_num_experts
        max_live_blocks = (live_routes + Int32(7) * active_experts_bound) // Int32(8)
        if max_live_blocks > Int32(self.driver.max_m_blocks):
            max_live_blocks = Int32(self.driver.max_m_blocks)
        if max_live_blocks < Int32(1):
            max_live_blocks = Int32(1)

        runtime_sms = grid_x // Int32(self.blocks_per_sm)
        if runtime_sms < Int32(1):
            runtime_sms = Int32(1)
        ws_cap = runtime_sms * Int32(4)

        can_persistent = Int32(0)
        if (
            active_m <= Int32(_K4_SMALLM_MAX_ROWS)
            and Int32(_WS_HEADER) + Int32(2) * max_live_blocks <= ws_cap
        ):
            can_persistent = Int32(1)

        if can_persistent != Int32(0):
            persistent_ctas = runtime_sms * Int32(_PERSISTENT_CTAS_PER_SM)
            self.persistent_smallm_kernel(
                rotation_input,
                rotation_gate,
                rotation_up,
                gate_suh,
                up_suh,
                t0_w13,
                t0_w2,
                t0_w13_global,
                t0_w2_global,
                t1_w13,
                t1_w2,
                t1_w13_global,
                t1_w2_global,
                fc1,
                activated,
                fc2,
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
                tier0_fc2_experts,
                tier1_fc2_experts,
                active_m,
                max_live_blocks,
                runtime_sms,
                persistent_ctas,
            ).launch(
                grid=(persistent_ctas, 1, 1),
                block=[128, 1, 1],
                stream=stream,
            )
        else:
            do_fused_act = Int32(0)
            if max_live_blocks <= ws_cap:
                do_fused_act = Int32(1)

            act_units = live_routes * Int32(self.intermediate_size // 128)
            act_ctas = (act_units + Int32(3)) // Int32(4)
            if act_ctas < Int32(1):
                act_ctas = Int32(1)

            smallm_fc1_n_tiles = (2 * self.intermediate_size) // self.smallm_fc1_tile_n
            smallm_fc1_threads = 128 if self.smallm_fc1_tile_n == 128 else 64
            fc2_n_tiles = self.hidden_size // self.fc2_tile_n
            fc2_threads = 128 if self.fc2_tile_n == 128 else 64

            if cutlass.const_expr(not self.has_medm_arm):
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
                if active_m <= Int32(_K4_SMALLM_MAX_ROWS):
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
        b_l2_policy: Uint64,
    ):
        cta_n_blocks = tile_n // 16
        chunks_per_k16 = cta_n_blocks * 8
        iters = (8 * chunks_per_k16) // cta_threads
        slab_addr = b_base_addr + Int64(slab) * b_slab_step_bytes
        b_dst_tid = cute.arch.shuffle_sync(b_stage_base + tid * Int32(16), lane)

        if use_evict_first != Int32(0):
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
        b_tensor_base: Int64,
        b_stage_base: Int32,
        b_tile_base_u32: Int64,
        b_k16_stride_u32: Int64,
        slab: Int32,
        tid: Int32,
        lane: Int32,
        use_evict_first: Int32,
        b_l2_policy: Uint64,
    ):
        cta_n_blocks = tile_n // 16
        chunks_per_k16 = cta_n_blocks * 2 * bits
        total_chunks = 8 * chunks_per_k16
        iters = (total_chunks + cta_threads - 1) // cta_threads
        exact = (total_chunks % cta_threads) == 0
        slab_base_u32 = b_tile_base_u32 + Int64(slab) * Int64(8) * b_k16_stride_u32
        b_dst_tid = cute.arch.shuffle_sync(b_stage_base + tid * Int32(16), lane)

        if use_evict_first != Int32(0):
            for i in cutlass.range_constexpr(iters):
                chunk = Int32(i * cta_threads) + tid
                k16 = chunk // Int32(chunks_per_k16)
                in_k16 = chunk - k16 * Int32(chunks_per_k16)
                dst = b_dst_tid + Int32(i * cta_threads * 16)
                src_addr = b_tensor_base + (
                    slab_base_u32
                    + Int64(k16) * b_k16_stride_u32
                    + Int64(in_k16 * 4)
                ) * Int64(4)
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
                src_addr = b_tensor_base + (
                    slab_base_u32
                    + Int64(k16) * b_k16_stride_u32
                    + Int64(in_k16 * 4)
                ) * Int64(4)
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
        b_tensor_base: Int64,
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
        b_l2_policy: Uint64,
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
                b_l2_policy,
            )
        else:
            self._fetch_b_slab_generic(
                bits,
                tile_n,
                cta_threads,
                b_tensor_base,
                b_stage_base,
                b_tile_base_u32,
                b_k16_stride_u32,
                slab,
                tid,
                lane,
                use_evict_first,
                b_l2_policy,
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
    def _consume_slab(
        self,
        bits: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        n_frags: cutlass.Constexpr[int],
        b_stage_base: Int32,
        a_rd_stage0: Int32,
        a_rd_stage1: Int32,
        a_sub_stride: Int32,
        warp_row: Int32,
        w_n: Int32,
        lane: Int32,
        prev_lane: Int32,
        g0_off: Int32,
        g1_off: Int32,
        g2_off: Int32,
        g3_off: Int32,
        g_s0: Int32,
        g_s1: Int32,
        acc,
    ):
        cta_n_blocks = tile_n // 16
        tile_bytes = 32 * bits
        k16_bytes = cta_n_blocks * tile_bytes

        b_warp_base = (
            b_stage_base
            + warp_row * Int32(2 * k16_bytes)
            + w_n * Int32(n_frags * tile_bytes)
        )
        for sub in cutlass.range_constexpr(2):
            sub_off = Int32(0) if cutlass.const_expr(sub == 0) else a_sub_stride
            for kk in cutlass.range_constexpr(2):
                a_addr = (
                    a_rd_stage0 if cutlass.const_expr(kk == 0) else a_rd_stage1
                ) + sub_off
                a0, a1 = ldmatrix_m8n8x2_b16(a_addr)
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
                        if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                            o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                win_a, win_b, 4
                            )
                        else:
                            o0, o1, o2, o3 = _packed_dequant_trellis_mcg_fast(
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
                            b0 = ld_shared_u32(tbase + g0_off)
                            b1 = ld_shared_u32(tbase + g1_off)
                            x0 = ld_shared_u32(tbase + g2_off)
                            x1 = ld_shared_u32(tbase + g3_off)
                            win_b = _shf_r_wrap_b32(b1, b0, g_s0)
                            win_a = _shf_r_wrap_b32(x1, x0, g_s1)
                            if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                                o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                    win_a, win_b, 5
                                )
                            else:
                                o0, o1, o2, o3 = _packed_dequant_trellis_mcg_fast(
                                    win_a, win_b, 5
                                )
                        else:
                            z0 = ld_shared_u32(tbase + g0_off)
                            z1 = ld_shared_u32(tbase + g1_off)
                            z2 = ld_shared_u32(tbase + g2_off)
                            win_a, win_b = trellis_align_stream_u32x2(
                                z0, z1, z2, g_s0, g_s1
                            )
                            if cutlass.const_expr(self.tier0.trellis_codebook == MUL1):
                                o0, o1, o2, o3 = packed_dequant_trellis_mul1_to_half2x4(
                                    win_a, win_b, 6
                                )
                            else:
                                o0, o1, o2, o3 = packed_dequant_trellis_stream_to_half2x4(
                                    win_a, win_b, 6
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
        n_frags: cutlass.Constexpr[int],
        stages: cutlass.Constexpr[int],
        a_slab_bytes: cutlass.Constexpr[int],
        fused_rot: cutlass.Constexpr[bool],
        pdl_wait: cutlass.Constexpr[bool],
        pdl_signal_after_wait: cutlass.Constexpr[bool],
        enable_rot_wait: cutlass.Constexpr[bool],
        b_stage_bytes: cutlass.Constexpr[int],
        a_off: cutlass.Constexpr[int],
        zero16_off: cutlass.Constexpr[int],
        a_stride_bytes: Int32,
        x_input: cute.Tensor,
        suh_ptr_base: Int64,
        a_tensor_base: Int64,
        b_tensor_base: Int64,
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
        b_l2_policy: Uint64,
        wait_rot: Int32,
        ws_state_addr: Int64,
        rot_ready_val: Int32,
        acc,
    ):
        a_base = smem_base + Int32(a_off)
        zero16_addr = smem_base + Int32(zero16_off)
        l16 = lane & Int32(15)
        r_ldm = l16 & Int32(7)
        c0_ldm = l16 >> Int32(3)
        k_base = (warp_row << Int32(2)) + c0_ldm
        prev_lane = (lane + Int32(31)) & Int32(31)

        # Hoist K5/K6 ring geometry and byte offsets outside the K-loop.
        g0_off = Int32(0)
        g1_off = Int32(0)
        g2_off = Int32(0)
        g3_off = Int32(0)
        g_s0 = Int32(0)
        g_s1 = Int32(0)
        if cutlass.const_expr(bits == 5):
            ib0, ib1, sb, _ = trellis256_lane_geom_bits(lane, 0, 4, 5)
            ia0, ia1, sa, _ = trellis256_lane_geom_bits(lane, 4, 4, 5)
            g0_off = ib0 * Int32(4)
            g1_off = ib1 * Int32(4)
            g2_off = ia0 * Int32(4)
            g3_off = ia1 * Int32(4)
            g_s0 = sb
            g_s1 = sa
        elif cutlass.const_expr(bits == 6):
            tile_u32 = 8 * 6
            i0, i2, s2_6, delta_6 = trellis256_lane_geom_bits(lane, 0, 8, 6)
            i1 = i0 + Int32(1)
            i1 = i1 - Int32(tile_u32) * (i1 >= Int32(tile_u32)).to(Int32)
            g0_off = i0 * Int32(4)
            g1_off = i1 * Int32(4)
            g2_off = i2 * Int32(4)
            g_s0 = s2_6
            g_s1 = delta_6

        a_rd_base0 = zero16_addr
        a_rd_base1 = zero16_addr
        a_stage_stride = Int32(0)
        a_sub_stride = Int32(0)
        if cutlass.const_expr(a_slab_bytes == _A_SLAB_BYTES_M1):
            if r_ldm == Int32(0) and valid_rows > Int32(0):
                a_rd_base0 = a_base + (k_base << Int32(4))
                a_rd_base1 = a_base + ((k_base + Int32(2)) << Int32(4))
                a_stage_stride = Int32(a_slab_bytes)
                a_sub_stride = Int32(128)
        else:
            if r_ldm < valid_rows:
                a_rd_base0 = a_base + (
                    ((r_ldm << Int32(4)) + (k_base ^ r_ldm)) << Int32(4)
                )
                a_rd_base1 = a_base + (
                    ((r_ldm << Int32(4)) + ((k_base + Int32(2)) ^ r_ldm)) << Int32(4)
                )
                a_stage_stride = Int32(a_slab_bytes)
                a_sub_stride = Int32(128)

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
            b_base_addr = b_tensor_base + (
                b_tile_base_u32
                + Int64(k16_0) * b_k16_stride_u32
                + Int64(in_k16_0 * 4)
            ) * Int64(4)
            b_iter_step_bytes = Int64(k16_step) * b_k16_stride_u32 * Int64(4)
            b_slab_step_bytes = Int64(8) * b_k16_stride_u32 * Int64(4)

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
            if cutlass.const_expr(a_slab_bytes == _A_SLAB_BYTES_M1):
                a_swz_off0 = c0 << Int32(4)
                a_valid0 = (row0 == Int32(0) and valid_rows > Int32(0)).to(Int32)
            else:
                a_swz_off0 = ((row0 << Int32(4)) + (c0 ^ row0)) << Int32(4)
                a_valid0 = (row0 < valid_rows).to(Int32)
            a_row_base0 = (
                a_tensor_base
                + Int64(route0) * Int64(a_stride_bytes)
                + Int64(c0 << Int32(4))
            )
            if cutlass.const_expr(cta_threads == 64):
                idx1 = tid + Int32(64)
                row1 = idx1 >> Int32(4)
                c1 = idx1 & Int32(15)
                route1 = cute.arch.shuffle_sync(route_lane, row1)
                a_swz_off1 = ((row1 << Int32(4)) + (c1 ^ row1)) << Int32(4)
                a_row_base1 = (
                    a_tensor_base
                    + Int64(route1) * Int64(a_stride_bytes)
                    + Int64(c1 << Int32(4))
                )
                a_valid1 = (row1 < valid_rows).to(Int32)

        # Prologue
        if cutlass.const_expr(pdl_wait):
            self._fetch_b_slab(
                bits,
                tile_n,
                cta_threads,
                b_tensor_base,
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
                b_l2_policy,
            )
            if k_slabs > Int32(1):
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_tensor_base,
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
                    b_l2_policy,
                )
            cute.arch.cp_async_commit_group()

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
                    a_base + Int32(a_slab_bytes),
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
        elif cutlass.const_expr(enable_rot_wait):
            if wait_rot != Int32(0):
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_tensor_base,
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
                    b_l2_policy,
                )
                if k_slabs > Int32(1):
                    self._fetch_b_slab(
                        bits,
                        tile_n,
                        cta_threads,
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                if cutlass.const_expr(stages == 4):
                    if k_slabs > Int32(2):
                        self._fetch_b_slab(
                            bits,
                            tile_n,
                            cta_threads,
                            b_tensor_base,
                            smem_base + Int32(2 * b_stage_bytes),
                            b_tile_base_u32,
                            b_k16_stride_u32,
                            b_base_addr,
                            b_iter_step_bytes,
                            b_slab_step_bytes,
                            Int32(2),
                            tid,
                            lane,
                            use_evict_first,
                            b_l2_policy,
                        )
                cute.arch.cp_async_commit_group()

                if tid == Int32(0):
                    st_v = _ld_global_acquire_gpu_i32(ws_state_addr)
                    while (st_v & Int32(_SM120_ROT_MASK)) != rot_ready_val:
                        _nanosleep(Int32(64))
                        st_v = _ld_global_acquire_gpu_i32(ws_state_addr)
                cute.arch.sync_threads()

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
                        a_base + Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stages == 4):
                        if k_slabs > Int32(2):
                            self._fetch_a_prerot_slab(
                                cta_threads,
                                a_base + Int32(2 * a_slab_bytes),
                                a_row_base0,
                                a_swz_off0,
                                a_valid0,
                                a_row_base1,
                                a_swz_off1,
                                a_valid1,
                                Int32(2),
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(2)
                        else:
                            cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            else:
                self._fetch_b_slab(
                    bits,
                    tile_n,
                    cta_threads,
                    b_tensor_base,
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
                    b_l2_policy,
                )
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
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stages == 4):
                        if k_slabs > Int32(2):
                            self._fetch_b_slab(
                                bits,
                                tile_n,
                                cta_threads,
                                b_tensor_base,
                                smem_base + Int32(2 * b_stage_bytes),
                                b_tile_base_u32,
                                b_k16_stride_u32,
                                b_base_addr,
                                b_iter_step_bytes,
                                b_slab_step_bytes,
                                Int32(2),
                                tid,
                                lane,
                                use_evict_first,
                                b_l2_policy,
                            )
                            self._fetch_a_prerot_slab(
                                cta_threads,
                                a_base + Int32(2 * a_slab_bytes),
                                a_row_base0,
                                a_swz_off0,
                                a_valid0,
                                a_row_base1,
                                a_swz_off1,
                                a_valid1,
                                Int32(2),
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(2)
                        else:
                            cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
        else:
            self._fetch_b_slab(
                bits,
                tile_n,
                cta_threads,
                b_tensor_base,
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
                b_l2_policy,
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
                    b_tensor_base,
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
                    b_l2_policy,
                )
                if cutlass.const_expr(not fused_rot):
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        Int32(1),
                    )
                cute.arch.cp_async_commit_group()

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
                        a_base + Int32(a_slab_bytes),
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

        # Main K-loop
        if cutlass.const_expr(stages == 4):
            stage = Int32(0)
            s = Int32(0)
            while s < k_slabs:
                wr_stage = (stage + Int32(3)) & Int32(3)
                next3 = s + Int32(3)
                if next3 < k_slabs:
                    self._fetch_b_slab(
                        bits,
                        tile_n,
                        cta_threads,
                        b_tensor_base,
                        smem_base + wr_stage * Int32(b_stage_bytes),
                        b_tile_base_u32,
                        b_k16_stride_u32,
                        b_base_addr,
                        b_iter_step_bytes,
                        b_slab_step_bytes,
                        next3,
                        tid,
                        lane,
                        use_evict_first,
                        b_l2_policy,
                    )
                    self._fetch_a_prerot_slab(
                        cta_threads,
                        a_base + wr_stage * Int32(a_slab_bytes),
                        a_row_base0,
                        a_swz_off0,
                        a_valid0,
                        a_row_base1,
                        a_swz_off1,
                        a_valid1,
                        next3,
                    )
                    cute.arch.cp_async_commit_group()

                a_stage_off = stage * a_stage_stride
                self._consume_slab(
                    bits,
                    tile_n,
                    n_frags,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_rd_base0 + a_stage_off,
                    a_rd_base1 + a_stage_off,
                    a_sub_stride,
                    warp_row,
                    w_n,
                    lane,
                    prev_lane,
                    g0_off,
                    g1_off,
                    g2_off,
                    g3_off,
                    g_s0,
                    g_s1,
                    acc,
                )

                if next3 < k_slabs:
                    cute.arch.cp_async_wait_group(2)
                    cute.arch.sync_threads()
                elif s + Int32(2) < k_slabs:
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                elif s + Int32(1) < k_slabs:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()

                stage = (stage + Int32(1)) & Int32(3)
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
                        b_tensor_base,
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
                        b_l2_policy,
                    )
                    if cutlass.const_expr(not fused_rot):
                        self._fetch_a_prerot_slab(
                            cta_threads,
                            a_base + wr_stage * Int32(a_slab_bytes),
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
                            a_base + wr_stage * Int32(a_slab_bytes),
                            tok_lane,
                            valid_rows,
                            next2,
                            warp_id,
                            lane,
                        )

                a_stage_off = stage * a_stage_stride
                self._consume_slab(
                    bits,
                    tile_n,
                    n_frags,
                    smem_base + stage * Int32(b_stage_bytes),
                    a_rd_base0 + a_stage_off,
                    a_rd_base1 + a_stage_off,
                    a_sub_stride,
                    warp_row,
                    w_n,
                    lane,
                    prev_lane,
                    g0_off,
                    g1_off,
                    g2_off,
                    g3_off,
                    g_s0,
                    g_s1,
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
        n_frags: cutlass.Constexpr[int],
        fold_off: cutlass.Constexpr[int],
        c_tensor_base: Int64,
        c_stride_elem: Int32,
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
                st_global_v4_u32(c_tensor_base + out_elem * Int64(2), q0, q1, q2, q3)

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
        fc1_base: Int64,
        activated_base: Int64,
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

            gp01, gp23 = _ld_global_cg_v2_u32(fc1_base + g_off * Int64(2))
            up01, up23 = _ld_global_cg_v2_u32(fc1_base + u_off * Int64(2))
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
            st_global_v2_u32(activated_base + out_off * Int64(2), p01, p23)
            u += Int32(n_warps)

    @cute.kernel
    def persistent_smallm_kernel(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
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
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        max_live_blocks: cutlass.Int32,
        runtime_sms: cutlass.Int32,
        persistent_ctas: cutlass.Int32,
    ):
        tile_n = 64
        cta_threads = 128
        n_frags = 2
        stages = self.persistent_stages
        a_slab_bytes = self.persistent_a_slab_bytes
        rot_splits = self.rot_splits
        rot_ready_val = Int32(rot_splits * _SM120_ROT_DONE_STEP)
        (
            b_stage_bytes,
            a_off,
            fold_off,
            zero16_off,
            ctrl_off,
            smem_words,
        ) = _k4_persistent_smem_layout(self.max_bits, stages, a_slab_bytes)

        tidx, _, _ = cute.arch.thread_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        w_n = warp_id & Int32(1)
        warp_row = warp_id >> Int32(1)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, smem_words], 128
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        if tid == Int32(0):
            st_shared_v4_u32(
                smem_base + Int32(zero16_off),
                Uint32(0),
                Uint32(0),
                Uint32(0),
                Uint32(0),
            )

        actual_blocks = packed_route_count[Int32(0)].to(Int32) >> Int32(3)
        if actual_blocks > max_live_blocks:
            actual_blocks = max_live_blocks
        if actual_blocks < Int32(0):
            actual_blocks = Int32(0)

        ws_base = get_ptr_as_int64(workspace, Int64(0))
        ws_state_base = ws_base + Int64(_WS_HEADER * 4)
        ws_ready_base = ws_state_base + Int64(max_live_blocks) * Int64(4)
        pdl_signaled = Int32(0)

        if actual_blocks > Int32(0):
            total_experts = tier0_num_experts + tier1_num_experts
            live_routes = active_m * Int32(self.top_k)

            fc1_half_tiles = self.intermediate_size // tile_n
            fc1_n_tiles = 2 * fc1_half_tiles
            fc2_n_tiles = self.hidden_size // tile_n
            rot_nblk = self.hidden_size // 128
            blks_per_chunk = (rot_nblk + rot_splits - 1) // rot_splits

            total_rot_tasks = actual_blocks * Int32(rot_splits)
            total_fc1_tasks = actual_blocks * Int32(fc1_n_tiles)
            total_fc2_tasks = actual_blocks * Int32(fc2_n_tiles)

            rot_gate_base = get_ptr_as_int64(rotation_gate, Int64(0))
            rot_up_base = get_ptr_as_int64(rotation_up, Int64(0))
            fc1_base = get_ptr_as_int64(fc1, Int64(0))
            activated_base = get_ptr_as_int64(activated, Int64(0))
            fc2_base = get_ptr_as_int64(fc2, Int64(0))
            t0_w13_base = get_ptr_as_int64(t0_w13, Int64(0))
            t0_w2_base = get_ptr_as_int64(t0_w2, Int64(0))
            t1_w13_base = get_ptr_as_int64(t1_w13, Int64(0))
            t1_w2_base = get_ptr_as_int64(t1_w2, Int64(0))
            t0_w13_plane_u32 = Int64(cute.size(t0_w13)) // Int64(2)
            t1_w13_plane_u32 = Int64(cute.size(t1_w13)) // Int64(2)
            b_l2_policy = create_l2_evict_first_policy()

            # Test-and-atomic-add monotonic cursors: guaranteed deadlock-free
            # for any resident CTA count R >= 1 and any hardware CTA scheduling
            # order, while avoiding wasted atomics on exhausted cursors.
            phase = Int32(0)
            running = Int32(1)
            while running != Int32(0):
                if tid == Int32(0):
                    action_d = Int32(_ACT_EXIT)
                    arg0_d = Int32(0)
                    arg1_d = Int32(0)
                    arg2_d = Int32(0)
                    got_work = Int32(0)
                    while got_work == Int32(0):
                        if phase == Int32(0):
                            if (
                                _ld_global_acquire_gpu_i32(
                                    ws_base + Int64(_WS_ROT_CURSOR * 4)
                                )
                                >= total_rot_tasks
                            ):
                                phase = Int32(1)
                            else:
                                t_rot = _atomic_add_acq_rel_gpu_i32(
                                    ws_base + Int64(_WS_ROT_CURSOR * 4), Int32(1)
                                )
                                if t_rot < total_rot_tasks:
                                    rb_r = t_rot // Int32(rot_splits)
                                    chk_r = t_rot - rb_r * Int32(rot_splits)
                                    ce_r = block_expert_ids[rb_r].to(Int32)
                                    valid_r = (
                                        ce_r >= Int32(0) and ce_r < total_experts
                                    ).to(Int32)
                                    if valid_r != Int32(0):
                                        r0_r = packed_route_indices[
                                            rb_r * Int32(8)
                                        ].to(Int32)
                                        if r0_r < Int32(0) or r0_r >= live_routes:
                                            valid_r = Int32(0)
                                        elif cutlass.const_expr(self.compact_weights):
                                            lw_r = descriptor_map[
                                                Int64(3) * Int64(total_experts)
                                                + Int64(ce_r)
                                            ].to(Int32)
                                            if lw_r <= Int32(0):
                                                valid_r = Int32(0)
                                    if valid_r != Int32(0):
                                        action_d = Int32(_ACT_ROT)
                                        arg0_d = rb_r
                                        arg1_d = chk_r
                                        got_work = Int32(1)
                                    else:
                                        _atomic_add_acq_rel_gpu_i32(
                                            ws_state_base + Int64(rb_r) * Int64(4),
                                            Int32(_SM120_ROT_DONE_STEP),
                                        )
                                else:
                                    phase = Int32(1)
                        elif phase == Int32(1):
                            if (
                                _ld_global_acquire_gpu_i32(
                                    ws_base + Int64(_WS_FC1_CURSOR * 4)
                                )
                                >= total_fc1_tasks
                            ):
                                phase = Int32(2)
                            else:
                                t_fc1 = _atomic_add_acq_rel_gpu_i32(
                                    ws_base + Int64(_WS_FC1_CURSOR * 4), Int32(1)
                                )
                                if t_fc1 < total_fc1_tasks:
                                    rb_f = t_fc1 // Int32(fc1_n_tiles)
                                    nt_f = t_fc1 - rb_f * Int32(fc1_n_tiles)
                                    ce_f = block_expert_ids[rb_f].to(Int32)
                                    valid_f = (
                                        ce_f >= Int32(0) and ce_f < total_experts
                                    ).to(Int32)
                                    lw_f = Int32(self.intermediate_size)
                                    if valid_f != Int32(0):
                                        r0_f = packed_route_indices[
                                            rb_f * Int32(8)
                                        ].to(Int32)
                                        if r0_f < Int32(0) or r0_f >= live_routes:
                                            valid_f = Int32(0)
                                        elif cutlass.const_expr(self.compact_weights):
                                            lw_f = descriptor_map[
                                                Int64(3) * Int64(total_experts)
                                                + Int64(ce_f)
                                            ].to(Int32)
                                            if lw_f <= Int32(0):
                                                valid_f = Int32(0)
                                    active_tile = valid_f
                                    if cutlass.const_expr(self.compact_weights):
                                        if active_tile != Int32(0):
                                            loc_nt = nt_f
                                            if loc_nt >= Int32(fc1_half_tiles):
                                                loc_nt -= Int32(fc1_half_tiles)
                                            if loc_nt * Int32(tile_n) >= lw_f:
                                                active_tile = Int32(0)
                                    if active_tile != Int32(0):
                                        st_val = _ld_global_acquire_gpu_i32(
                                            ws_state_base + Int64(rb_f) * Int64(4)
                                        )
                                        rot_done = (
                                            (st_val & Int32(_SM120_ROT_MASK))
                                            == rot_ready_val
                                        ).to(Int32)
                                        action_d = Int32(_ACT_FC1)
                                        arg0_d = rb_f
                                        arg1_d = nt_f
                                        arg2_d = rot_done
                                        got_work = Int32(1)
                                    else:
                                        old_st = _atomic_add_acq_rel_gpu_i32(
                                            ws_state_base + Int64(rb_f) * Int64(4),
                                            Int32(1),
                                        )
                                        if (
                                            old_st & Int32(_SM120_FC1_CNT_MASK)
                                        ) + Int32(1) == Int32(fc1_n_tiles):
                                            if valid_f != Int32(0):
                                                action_d = Int32(_ACT_ACT_ONLY)
                                                arg0_d = rb_f
                                                got_work = Int32(1)
                                            else:
                                                slot_f = _atomic_add_acq_rel_gpu_i32(
                                                    ws_base
                                                    + Int64(_WS_READY_COUNT * 4),
                                                    Int32(1),
                                                )
                                                _st_global_release_gpu_i32(
                                                    ws_ready_base
                                                    + Int64(slot_f) * Int64(4),
                                                    rb_f + Int32(1),
                                                )
                                else:
                                    phase = Int32(2)
                        else:
                            if (
                                _ld_global_acquire_gpu_i32(
                                    ws_base + Int64(_WS_FC2_CURSOR * 4)
                                )
                                >= total_fc2_tasks
                            ):
                                action_d = Int32(_ACT_EXIT)
                                got_work = Int32(1)
                            else:
                                t_fc2 = _atomic_add_acq_rel_gpu_i32(
                                    ws_base + Int64(_WS_FC2_CURSOR * 4), Int32(1)
                                )
                                if t_fc2 < total_fc2_tasks:
                                    slot_c = t_fc2 // Int32(fc2_n_tiles)
                                    nt_c = t_fc2 - slot_c * Int32(fc2_n_tiles)
                                    ready_addr = (
                                        ws_ready_base + Int64(slot_c) * Int64(4)
                                    )
                                    rval = _ld_global_acquire_gpu_i32(ready_addr)
                                    while rval == Int32(0):
                                        _nanosleep(Int32(256))
                                        rval = _ld_global_acquire_gpu_i32(ready_addr)
                                    rb_c = rval - Int32(1)
                                    ce_c = block_expert_ids[rb_c].to(Int32)
                                    valid_c = (
                                        ce_c >= Int32(0) and ce_c < total_experts
                                    ).to(Int32)
                                    if valid_c != Int32(0):
                                        r0_c = packed_route_indices[
                                            rb_c * Int32(8)
                                        ].to(Int32)
                                        if r0_c < Int32(0) or r0_c >= live_routes:
                                            valid_c = Int32(0)
                                        elif cutlass.const_expr(self.compact_weights):
                                            lw_c = descriptor_map[
                                                Int64(3) * Int64(total_experts)
                                                + Int64(ce_c)
                                            ].to(Int32)
                                            if lw_c <= Int32(0):
                                                valid_c = Int32(0)
                                    if valid_c != Int32(0):
                                        action_d = Int32(_ACT_FC2)
                                        arg0_d = rb_c
                                        arg1_d = nt_c
                                        got_work = Int32(1)
                                else:
                                    action_d = Int32(_ACT_EXIT)
                                    got_work = Int32(1)
                    st_shared_i32(smem_base + Int32(ctrl_off + 0), action_d)
                    st_shared_i32(smem_base + Int32(ctrl_off + 4), arg0_d)
                    st_shared_i32(smem_base + Int32(ctrl_off + 8), arg1_d)
                    st_shared_i32(smem_base + Int32(ctrl_off + 12), arg2_d)
                cute.arch.sync_threads()

                action = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 0))
                rb = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 4))
                arg1 = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 8))
                arg2 = ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 12))

                # Signal PDL once every CTA has moved past FC1 so v41_exl3_sum
                # can launch into freed SM slots during FC2's tail and preload svh.
                if pdl_signaled == Int32(0):
                    if action == Int32(_ACT_FC2) or action == Int32(_ACT_EXIT):
                        cute.arch.griddepcontrol_launch_dependents()
                        pdl_signaled = Int32(1)

                if action == Int32(_ACT_EXIT):
                    running = Int32(0)
                elif action == Int32(_ACT_ROT):
                    combined_expert = block_expert_ids[rb].to(Int32)
                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")
                    route_lane = Int32(0)
                    if is_valid != Int32(0):
                        route_lane = r_lane
                    tok_lane = route_lane // Int32(self.top_k)

                    blk_start = arg1 * Int32(blks_per_chunk)
                    blk_end = blk_start + Int32(blks_per_chunk)
                    if blk_end > Int32(rot_nblk):
                        blk_end = Int32(rot_nblk)

                    blk = blk_start + warp_id
                    while blk < blk_end:
                        col0 = blk * Int32(128) + lane * Int32(4)
                        s_off = Int64(combined_expert) * Int64(
                            self.hidden_size
                        ) + Int64(col0)
                        gw0, gw1 = ld_global_nc_v2_u32(
                            get_ptr_as_int64(gate_suh, s_off)
                        )
                        uw0, uw1 = ld_global_nc_v2_u32(
                            get_ptr_as_int64(up_suh, s_off)
                        )
                        r_idx = Int32(0)
                        while r_idx < valid_rows:
                            route_r = cute.arch.shuffle_sync(route_lane, r_idx)
                            tok_r = cute.arch.shuffle_sync(tok_lane, r_idx)
                            x_off = Int64(tok_r) * Int64(self.hidden_size) + Int64(
                                col0
                            )
                            out_off = Int64(route_r) * Int64(self.hidden_size) + Int64(
                                col0
                            )
                            xw0, xw1 = ld_global_nc_v2_u32(
                                get_ptr_as_int64(rotation_input, x_off)
                            )
                            xh0 = cvt_bf16x2_to_f16x2_via_f32(xw0)
                            xh1 = cvt_bf16x2_to_f16x2_via_f32(xw1)

                            gv0, gv1 = f16x2_to_f32x2(half2_mul(xh0, gw0))
                            gv2, gv3 = f16x2_to_f32x2(half2_mul(xh1, gw1))
                            gh0, gh1, gh2, gh3 = self.driver._had128_quad(
                                gv0, gv1, gv2, gv3, lane
                            )
                            st_global_v2_u32(
                                rot_gate_base + out_off * Int64(2),
                                pack_f32x2_to_f16x2(gh0, gh1),
                                pack_f32x2_to_f16x2(gh2, gh3),
                            )

                            uv0, uv1 = f16x2_to_f32x2(half2_mul(xh0, uw0))
                            uv2, uv3 = f16x2_to_f32x2(half2_mul(xh1, uw1))
                            uh0, uh1, uh2, uh3 = self.driver._had128_quad(
                                uv0, uv1, uv2, uv3, lane
                            )
                            st_global_v2_u32(
                                rot_up_base + out_off * Int64(2),
                                pack_f32x2_to_f16x2(uh0, uh1),
                                pack_f32x2_to_f16x2(uh2, uh3),
                            )
                            r_idx += Int32(1)
                        blk += Int32(4)

                    cute.arch.sync_threads()
                    if tid == Int32(0):
                        _atomic_add_acq_rel_gpu_i32(
                            ws_state_base + Int64(rb) * Int64(4),
                            Int32(_SM120_ROT_DONE_STEP),
                        )
                elif action == Int32(_ACT_ACT_ONLY):
                    combined_expert = block_expert_ids[rb].to(Int32)
                    live_width = Int32(self.intermediate_size)
                    if cutlass.const_expr(self.compact_weights):
                        live_width = descriptor_map[
                            Int64(3) * Int64(total_experts) + Int64(combined_expert)
                        ].to(Int32)
                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")
                    route_lane = Int32(0)
                    if is_valid != Int32(0):
                        route_lane = r_lane
                    if valid_rows > Int32(0) and live_width > Int32(0):
                        self._run_fused_act_block(
                            cta_threads,
                            fc1_base,
                            activated_base,
                            intermediate_rotations,
                            combined_expert,
                            live_width,
                            route_lane,
                            valid_rows,
                            warp_id,
                            lane,
                        )
                    cute.arch.sync_threads()
                    if tid == Int32(0):
                        slot = _atomic_add_acq_rel_gpu_i32(
                            ws_base + Int64(_WS_READY_COUNT * 4), Int32(1)
                        )
                        _st_global_release_gpu_i32(
                            ws_ready_base + Int64(slot) * Int64(4),
                            rb + Int32(1),
                        )
                else:
                    is_fc1_rt = (action == Int32(_ACT_FC1)).to(Int32)
                    output_n_tile = arg1
                    combined_expert = block_expert_ids[rb].to(Int32)

                    descriptor_row = Int32(2)
                    proj = Int32(0)
                    local_n_tile = output_n_tile
                    if is_fc1_rt != Int32(0):
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

                    r_lane = live_routes
                    if lane < Int32(8):
                        r_lane = packed_route_indices[rb * Int32(8) + lane].to(Int32)
                    is_valid = (
                        lane < Int32(8)
                        and r_lane >= Int32(0)
                        and r_lane < live_routes
                    ).to(Int32)
                    valid_rows = cute.arch.warp_redux_sync(is_valid, "add")
                    route_lane = Int32(0)
                    if is_valid != Int32(0):
                        route_lane = r_lane

                    descriptor = descriptor_map[
                        Int64(descriptor_row) * Int64(total_experts)
                        + Int64(combined_expert)
                    ].to(Int32)

                    if valid_rows > Int32(0) and descriptor >= Int32(0):
                        tier = descriptor >> Int32(self.descriptor_bits)
                        local_expert = descriptor & Int32(
                            (1 << self.descriptor_bits) - 1
                        )

                        t0_lim = tier0_fc2_experts
                        t1_lim = tier1_fc2_experts
                        if is_fc1_rt != Int32(0):
                            if proj != Int32(0):
                                t0_lim = tier0_up_experts
                                t1_lim = tier1_up_experts
                            else:
                                t0_lim = tier0_gate_experts
                                t1_lim = tier1_gate_experts

                        is_t0 = tier == Int32(0) and local_expert < t0_lim
                        is_t1 = tier == Int32(1) and local_expert < t1_lim

                        if is_t0 or is_t1:
                            use_evict_first = Int32(1)
                            if rb + Int32(1) < actual_blocks:
                                if (
                                    block_expert_ids[rb + Int32(1)].to(Int32)
                                    == combined_expert
                                ):
                                    use_evict_first = Int32(0)

                            a_tensor_base = activated_base
                            a_stride_bytes = Int32(self.intermediate_size * 2)
                            c_tensor_base = fc2_base
                            c_stride_elem = Int32(self.hidden_size)
                            k_slabs = live_width >> Int32(7)
                            wait_rot = Int32(0)
                            if is_fc1_rt != Int32(0):
                                a_tensor_base = rot_gate_base
                                if proj != Int32(0):
                                    a_tensor_base = rot_up_base
                                a_stride_bytes = Int32(self.hidden_size * 2)
                                c_tensor_base = fc1_base
                                c_stride_elem = Int32(2 * self.intermediate_size)
                                k_slabs = Int32(self.hidden_size // 128)
                                if arg2 == Int32(0):
                                    wait_rot = Int32(1)

                            acc = [
                                cute.make_rmem_tensor((1,), cutlass.Float32)
                                for _ in range(4 * n_frags)
                            ]
                            for idx in cutlass.range_constexpr(4 * n_frags):
                                acc[idx].fill(0.0)

                            global_scale_f32 = cutlass.Float32(1.0)
                            cta_n_blocks = tile_n // 16
                            ws_state_addr = ws_state_base + Int64(rb) * Int64(4)

                            if is_t0:
                                bits0 = self.tier0.trellis_bits
                                tile_u32_0 = 8 * bits0
                                b_base_0 = t0_w2_base
                                b_k16_stride_0 = Int64(
                                    (self.hidden_size // 16) * tile_u32_0
                                )
                                b_tile_base_0 = Int64(0)
                                if is_fc1_rt != Int32(0):
                                    b_base_0 = t0_w13_base
                                    global_scale_f32 = t0_w13_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_k16_stride_0 = (
                                            live_width.to(Int64) >> Int64(4)
                                        ) * Int64(tile_u32_0)
                                        b_tile_base_0 = weight_offset + Int64(
                                            local_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_0)
                                    else:
                                        half_n16 = self.intermediate_size // 16
                                        proj_expert_u32 = (
                                            (self.hidden_size // 16)
                                            * half_n16
                                            * tile_u32_0
                                        )
                                        b_k16_stride_0 = Int64(half_n16 * tile_u32_0)
                                        b_tile_base_0 = (
                                            Int64(proj) * t0_w13_plane_u32
                                            + Int64(local_expert)
                                            * Int64(proj_expert_u32)
                                            + Int64(local_n_tile)
                                            * Int64(cta_n_blocks * tile_u32_0)
                                        )
                                else:
                                    global_scale_f32 = t0_w2_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_tile_base_0 = weight_offset + Int64(
                                            output_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_0)
                                    else:
                                        n16 = self.hidden_size // 16
                                        expert_u32 = (
                                            (self.intermediate_size // 16)
                                            * n16
                                            * tile_u32_0
                                        )
                                        b_tile_base_0 = Int64(local_expert) * Int64(
                                            expert_u32
                                        ) + Int64(output_n_tile) * Int64(
                                            cta_n_blocks * tile_u32_0
                                        )

                                self._run_gemm_kloop(
                                    bits0,
                                    tile_n,
                                    cta_threads,
                                    n_frags,
                                    stages,
                                    a_slab_bytes,
                                    False,
                                    False,
                                    False,
                                    True,
                                    b_stage_bytes,
                                    a_off,
                                    zero16_off,
                                    a_stride_bytes,
                                    rotation_input,
                                    Int64(0),
                                    a_tensor_base,
                                    b_base_0,
                                    b_tile_base_0,
                                    b_k16_stride_0,
                                    k_slabs,
                                    route_lane,
                                    Int32(0),
                                    valid_rows,
                                    smem_base,
                                    tid,
                                    warp_id,
                                    warp_row,
                                    w_n,
                                    lane,
                                    use_evict_first,
                                    b_l2_policy,
                                    wait_rot,
                                    ws_state_addr,
                                    rot_ready_val,
                                    acc,
                                )
                            else:
                                bits1 = self.tier1.trellis_bits
                                tile_u32_1 = 8 * bits1
                                b_base_1 = t1_w2_base
                                b_k16_stride_1 = Int64(
                                    (self.hidden_size // 16) * tile_u32_1
                                )
                                b_tile_base_1 = Int64(0)
                                if is_fc1_rt != Int32(0):
                                    b_base_1 = t1_w13_base
                                    global_scale_f32 = t1_w13_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_k16_stride_1 = (
                                            live_width.to(Int64) >> Int64(4)
                                        ) * Int64(tile_u32_1)
                                        b_tile_base_1 = weight_offset + Int64(
                                            local_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_1)
                                    else:
                                        half_n16 = self.intermediate_size // 16
                                        proj_expert_u32 = (
                                            (self.hidden_size // 16)
                                            * half_n16
                                            * tile_u32_1
                                        )
                                        b_k16_stride_1 = Int64(half_n16 * tile_u32_1)
                                        b_tile_base_1 = (
                                            Int64(proj) * t1_w13_plane_u32
                                            + Int64(local_expert)
                                            * Int64(proj_expert_u32)
                                            + Int64(local_n_tile)
                                            * Int64(cta_n_blocks * tile_u32_1)
                                        )
                                else:
                                    global_scale_f32 = t1_w2_global[local_expert].to(
                                        cutlass.Float32
                                    )
                                    if cutlass.const_expr(self.compact_weights):
                                        b_tile_base_1 = weight_offset + Int64(
                                            output_n_tile
                                        ) * Int64(cta_n_blocks * tile_u32_1)
                                    else:
                                        n16 = self.hidden_size // 16
                                        expert_u32 = (
                                            (self.intermediate_size // 16)
                                            * n16
                                            * tile_u32_1
                                        )
                                        b_tile_base_1 = Int64(local_expert) * Int64(
                                            expert_u32
                                        ) + Int64(output_n_tile) * Int64(
                                            cta_n_blocks * tile_u32_1
                                        )

                                self._run_gemm_kloop(
                                    bits1,
                                    tile_n,
                                    cta_threads,
                                    n_frags,
                                    stages,
                                    a_slab_bytes,
                                    False,
                                    False,
                                    False,
                                    True,
                                    b_stage_bytes,
                                    a_off,
                                    zero16_off,
                                    a_stride_bytes,
                                    rotation_input,
                                    Int64(0),
                                    a_tensor_base,
                                    b_base_1,
                                    b_tile_base_1,
                                    b_k16_stride_1,
                                    k_slabs,
                                    route_lane,
                                    Int32(0),
                                    valid_rows,
                                    smem_base,
                                    tid,
                                    warp_id,
                                    warp_row,
                                    w_n,
                                    lane,
                                    use_evict_first,
                                    b_l2_policy,
                                    wait_rot,
                                    ws_state_addr,
                                    rot_ready_val,
                                    acc,
                                )

                            self._epilogue_store(
                                tile_n,
                                n_frags,
                                fold_off,
                                c_tensor_base,
                                c_stride_elem,
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

                    cute.arch.sync_threads()
                    if is_fc1_rt != Int32(0):
                        if tid == Int32(0):
                            old = _atomic_add_acq_rel_gpu_i32(
                                ws_state_base + Int64(rb) * Int64(4), Int32(1)
                            )
                            is_last = (
                                (old & Int32(_SM120_FC1_CNT_MASK)) + Int32(1)
                                == Int32(fc1_n_tiles)
                            ).to(Int32)
                            st_shared_i32(smem_base + Int32(ctrl_off + 16), is_last)
                        cute.arch.sync_threads()
                        if (
                            ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 16))
                            != Int32(0)
                        ):
                            if valid_rows > Int32(0) and live_width > Int32(0):
                                self._run_fused_act_block(
                                    cta_threads,
                                    fc1_base,
                                    activated_base,
                                    intermediate_rotations,
                                    combined_expert,
                                    live_width,
                                    route_lane,
                                    valid_rows,
                                    warp_id,
                                    lane,
                                )
                            cute.arch.sync_threads()
                            if tid == Int32(0):
                                slot = _atomic_add_acq_rel_gpu_i32(
                                    ws_base + Int64(_WS_READY_COUNT * 4), Int32(1)
                                )
                                _st_global_release_gpu_i32(
                                    ws_ready_base + Int64(slot) * Int64(4),
                                    rb + Int32(1),
                                )

        cute.arch.sync_threads()
        if tid == Int32(0):
            exited = _atomic_add_acq_rel_gpu_i32(
                ws_base + Int64(_WS_EXIT_COUNT * 4), Int32(1)
            )
            is_last_cta = (exited + Int32(1) == persistent_ctas).to(Int32)
            st_shared_i32(smem_base + Int32(ctrl_off + 16), is_last_cta)
        cute.arch.sync_threads()
        if ld_shared_i32_relaxed(smem_base + Int32(ctrl_off + 16)) != Int32(0):
            ws_used = Int32(_WS_HEADER) + Int32(2) * max_live_blocks
            idx = tid
            while idx < ws_used:
                _st_global_relaxed_gpu_i32(ws_base + Int64(idx) * Int64(4), Int32(0))
                idx += Int32(cta_threads)
        if pdl_signaled == Int32(0):
            cute.arch.griddepcontrol_launch_dependents()

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
        pdl_signaled = Int32(0)
        if cutlass.const_expr(pdl_signal_entry):
            cute.arch.griddepcontrol_launch_dependents()
            pdl_signaled = Int32(1)

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
                cute.struct.MemRange[cutlass.Uint32, smem_words], 128
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
                                if tid == Int32(0):
                                    st_shared_v4_u32(
                                        smem_base + Int32(fold_off),
                                        Uint32(0),
                                        Uint32(0),
                                        Uint32(0),
                                        Uint32(0),
                                    )

                                tok_lane = Int32(0)
                                if cutlass.const_expr(fused_rot):
                                    tok_lane = route_lane // Int32(self.top_k)

                                use_evict_first = Int32(1)
                                if rb + Int32(1) < actual_blocks:
                                    if (
                                        block_expert_ids[rb + Int32(1)].to(Int32)
                                        == combined_expert
                                    ):
                                        use_evict_first = Int32(0)

                                b_l2_policy = create_l2_evict_first_policy()

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

                                a_stride_bytes = Int32(
                                    self.hidden_size * 2
                                    if is_fc1
                                    else self.intermediate_size * 2
                                )
                                c_stride_elem = Int32(
                                    2 * self.intermediate_size
                                    if is_fc1
                                    else self.hidden_size
                                )
                                c_tensor_base = get_ptr_as_int64(c_flat, Int64(0))
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
                                        n_frags,
                                        stages,
                                        _A_SLAB_BYTES_FULL,
                                        fused_rot,
                                        pdl_wait,
                                        pdl_signal_after_wait,
                                        False,
                                        b_stage_bytes,
                                        a_off,
                                        fold_off,
                                        a_stride_bytes,
                                        x_or_a_gate,
                                        suh_ptr_base,
                                        a_tensor_base,
                                        get_ptr_as_int64(t0_b, Int64(0)),
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
                                        b_l2_policy,
                                        Int32(0),
                                        Int64(0),
                                        Int32(0),
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
                                        n_frags,
                                        stages,
                                        _A_SLAB_BYTES_FULL,
                                        fused_rot,
                                        pdl_wait,
                                        pdl_signal_after_wait,
                                        False,
                                        b_stage_bytes,
                                        a_off,
                                        fold_off,
                                        a_stride_bytes,
                                        x_or_a_gate,
                                        suh_ptr_base,
                                        a_tensor_base,
                                        get_ptr_as_int64(t1_b, Int64(0)),
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
                                        b_l2_policy,
                                        Int32(0),
                                        Int64(0),
                                        Int32(0),
                                        acc,
                                    )

                                if cutlass.const_expr(pdl_wait and pdl_signal_after_wait):
                                    pdl_signaled = Int32(1)

                                self._epilogue_store(
                                    tile_n,
                                    n_frags,
                                    fold_off,
                                    c_tensor_base,
                                    c_stride_elem,
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
                                        get_ptr_as_int64(c_flat, Int64(0)),
                                        get_ptr_as_int64(activated, Int64(0)),
                                        intermediate_rotations,
                                        combined_expert,
                                        live_width,
                                        route_lane,
                                        valid_rows,
                                        warp_id,
                                        lane,
                                    )

        if cutlass.const_expr(pdl_signal_after_wait or not is_fc1):
            if pdl_signaled == Int32(0):
                cute.arch.griddepcontrol_launch_dependents()

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
            _SM120_FC2_SLAB_STAGES,
            False,
            True,
            False,
            True,
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
