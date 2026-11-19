"""One-launch mixed-bitrate ``trellis_t256`` MoE execution.

The route packer assigns every global expert to one combined expert namespace.
Input/intermediate rotations therefore run once. Per-tile dispatch resolves the
combined expert to a bitrate-specialized K2 through K6 decoder while preserving
the single cooperative FC1/activation/FC2 grid used by homogeneous trellis
execution. The decoder codebook is a compile-time parameter shared with the
fused W4A16 kernel ABI.

The module stays internal because checkpoint interpretation and runtime planning
belong to the serving framework; B12X owns only the prepared kernel path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Protocol, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, Uint64, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    bfloat2_to_float2_scaled,
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
    cp_async4_shared_global_pred_l2hint,
    create_l2_evict_first_policy,
    cvt_bf16x2_to_f16x2_via_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    f16x2_to_f32x2,
    get_ptr_as_int64,
    half2_to_float2_scaled,
    ld_global_acquire_i32,
    ld_global_nc_v2_u32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x2_b16,
    pack_f32x2_to_f16x2,
    packed_dequant_trellis_stream_to_half2x4,
    packed_dequant_trellis_to_half2x4,
    red_add_global_release_i32,
    shared_ptr_to_u32,
    st_global_i32,
    st_global_v2_u32,
    st_global_v4_u32,
    st_shared_f16_from_f32,
    st_shared_i32,
    st_shared_v4_f32,
    trellis_align_stream_u32x2,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.trellis_codebooks import (
    normalize_codebook,
    validate_codebook_bits,
)

from .host import (
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    route_pack_warmup_token_counts,
)
from .kernel import (
    _SQG_XOR_CHEB_T12_LUT_ENTRIES,
    _SQG_XOR_CHEB_T12_SMEM_REGION_BYTES,
    W4A16FusedMoeKernel,
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _trellis256_execution_lut,
    _w4a16_small_m_splitk_enabled,
    compile_w4a16_topk_sum,
    pack_topk_routes_by_expert,
)
# Projection descriptors pack the tier in the high bits and the tier-local
# expert in the low bits. Preserve nine bits for existing E<=512 geometries;
# E896 uses ten. This is an in-process int32 artifact, not a checkpoint ABI.
_MAX_TIER_EXPERTS = 1024


def _descriptor_bits(route_num_experts: int) -> int:
    """Static routed geometry, not the padded projection-bank slot count."""
    if not 1 <= int(route_num_experts) <= _MAX_TIER_EXPERTS:
        raise ValueError("mixed Trellis routed expert count must be in [1, 1024]")
    return 9 if int(route_num_experts) <= 512 else 10


# One SM120 AOT package serves the 188-SM PRO 6000 and the 170-SM 5090.
# Only workspace sizing uses these bounds; cooperative grids use the live card.
EXL3_AOT_MAX_SMS = {(12, 0): 188, (12, 1): 48}


# Below this many rows a (route, block) unit per warp keeps the grid busy;
# above it, each token's input block is read once for all of its routes.
_TOKEN_MAJOR_ROTATION_MIN_ROWS = 64
# H128 blocks one warp rotates per token-major unit (expert ids load once).
_ROTATION_BLOCKS_PER_UNIT = 8


@dataclass(frozen=True)
class MixedTrellisCompileResult:
    compiled: object
    topk_sum: object
    trellis_lut: torch.Tensor
    size_m: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    tier0_num_experts: int
    tier1_num_experts: int
    tier0_bits: int
    tier1_bits: int
    trellis_codebook: str
    fc1_tile_k: int
    fc1_tile_n: int
    fc2_tile_k: int
    fc2_tile_n: int
    moe_block_size: int
    fc2_moe_block_size: int
    fc2_schedule_route_block_factor: int
    max_m_blocks: int
    blocks_per_sm: int
    sms: int
    shared_memory_bytes: int
    rotation_input_dtype: str
    route_ids_dtype: torch.dtype
    direct_topk_routes: bool
    broadcast_suh: bool
    broadcast_svh: bool
    paired_boundary: str | None = field(default=None, kw_only=True)
    compact_weights: bool = field(default=False, kw_only=True)
    # FC1 rotates staged token rows itself; nothing writes rotation_up.
    fused_input_rotation: bool = field(default=False, kw_only=True)
    # Warp-specialized prefill (mixed_trellis_ws): FC1, SwiGLU and FC2 as three
    # launches of producer/consumer CTAs; FC1 rotates its input rows itself.
    warp_specialized: bool = field(default=False, kw_only=True)
    ws_prerotate: bool = field(default=False, kw_only=True)
    # FC1 input rows: "bf16" [M, H], or "e4m3_k32" wire rows [M, H + H/32]
    # bytes (E4M3 values, then one UE8M0 scale per 32) that the
    # warp-specialized producers widen in shared memory.
    input_format: str = field(default="bf16", kw_only=True)
    row_addresses: bool = field(default=False, kw_only=True)
    # Warp-specialized input-row ring depth (None: the kernel's default).
    ws_input_stages: int | None = field(default=None, kw_only=True)
    # Opt-in INT8 activations/decoded weights for warp-specialized prefill.
    activations: str = field(default="a16", kw_only=True)
    # The L2 decode schedule's canonical spec (None: the default schedule).
    decode_schedule: str | None = field(default=None, kw_only=True)
    # AOT workspace capacity; None preserves device-sized JIT allocations.
    workspace_sms: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class MixedTrellis3CompileResult(MixedTrellisCompileResult):
    """Launch metadata for a descriptor-selected three-tier specialization."""

    tier2_num_experts: int
    tier2_bits: int


def _normalize_mixed_trellis_format(
    codebook: str | int,
    bits: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    """Validate one compile-time codebook/tier-set selection.

    Expert membership and projection descriptors remain runtime artifact data;
    only the distinct decoder tiers and their common codebook specialize a
    kernel. This is deliberately independent of model identity.
    """

    normalized = normalize_codebook(codebook)
    tiers = tuple(int(value) for value in bits)
    if len(tiers) not in (2, 3):
        raise ValueError("mixed Trellis requires exactly two or three tiers")
    if len(set(tiers)) != len(tiers):
        raise ValueError(
            "mixed Trellis tiers must be distinct; "
            f"got {tuple(f'K{value}' for value in tiers)}"
        )
    for value in tiers:
        validate_codebook_bits(normalized, value)
    return normalized, tiers


@dataclass(frozen=True)
class MixedTrellisRotations:
    intermediate: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor


@dataclass(frozen=True)
class MixedTrellisBinding:
    """Validated fixed artifacts for a two-tier launch."""

    launch: MixedTrellisCompileResult
    tiers: tuple[MixedTrellisTier, MixedTrellisTier]
    global_to_combined: torch.Tensor
    descriptor_map: torch.Tensor
    rotations: MixedTrellisRotations
    gate_counts: tuple[int, int]
    up_counts: tuple[int, int]
    fc2_counts: tuple[int, int]
    total_experts: int
    route_num_experts: int
    device: torch.device
    tier_pointers: tuple[tuple[object, ...], ...]
    descriptor_map_ptr: object
    intermediate_rotations_ptr: object
    gate_suh_ptr: object
    up_suh_ptr: object
    global_to_combined_ptr: object
    down_svh_ptr: object


@dataclass(frozen=True)
class MixedTrellis3Binding:
    """Validated fixed artifacts for a three-tier launch."""

    launch: MixedTrellis3CompileResult
    tiers: tuple[MixedTrellisTier, MixedTrellisTier, MixedTrellisTier]
    global_to_combined: torch.Tensor
    descriptor_map: torch.Tensor
    rotations: MixedTrellisRotations
    gate_counts: tuple[int, int, int]
    up_counts: tuple[int, int, int]
    fc2_counts: tuple[int, int, int]
    total_experts: int
    route_num_experts: int
    device: torch.device
    tier_pointers: tuple[tuple[object, ...], ...]
    descriptor_map_ptr: object
    intermediate_rotations_ptr: object
    gate_suh_ptr: object
    up_suh_ptr: object
    global_to_combined_ptr: object
    down_svh_ptr: object


@dataclass
class MixedTrellisBuffers:
    rotation_gate: torch.Tensor
    rotation_up: torch.Tensor
    fc1: torch.Tensor
    activated: torch.Tensor
    fc2: torch.Tensor
    output: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    fc1_scratch: torch.Tensor
    fc2_scratch: torch.Tensor
    workspace: torch.Tensor


class MixedTrellisTier(Protocol):
    """Prepared Trellis tier fields consumed by the mixed launch."""

    num_experts: int
    trellis_codebook: str
    intermediate_rotations: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor


class W4A16MixedTrellisKernel:
    """One cooperative grid over two native Trellis bitrates."""

    # Persistent compile keys do not include launch source text. Change this
    # version whenever the compiled argument or tensor-layout contract changes.
    ABI_VERSION = 21
    _token_major_rotation = False
    compact_weights = False
    COMPACT_DESCRIPTOR_ROWS = 7

    descriptor_bits = 9

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        descriptor_bits: int = 9,
        paired_boundary: str | None = None,
        compact_weights: bool = False,
        token_major_rotation: bool = False,
    ):
        for name, moe in (("driver", driver), ("tier0", tier0), ("tier1", tier1)):
            if not moe.full_rotation or not moe.intermediate_rotation:
                raise ValueError(f"mixed Trellis {name} requires full rotation")
            if moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} cannot use TC decode")
            if moe.weight_layout != "trellis_t256":
                raise ValueError(f"mixed Trellis {name} requires native t256 weights")
            if moe.element_dtype != "fp16":
                raise ValueError(f"mixed Trellis {name} requires fp16 GEMM operands")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "rotation_input_dtype",
            "broadcast_suh",
            "trellis_codebook",
            "direct_topk_routes",
            "cta_threads",
            "sms",
        ):
            values = tuple(getattr(moe, attr) for moe in (driver, tier0, tier1))
            if values[1:] != values[:-1]:
                raise ValueError(f"mixed Trellis kernels disagree on {attr}: {values}")
        for phase in ("fc1", "fc2"):
            gemms = tuple(getattr(moe, phase) for moe in (driver, tier0, tier1))
            geometry = tuple(
                (
                    g.n_tiles,
                    g.k_tiles,
                    g.tile_n,
                    g.tile_k,
                    g.cta_threads,
                    g.moe_block_size,
                    g.schedule_route_block_factor,
                )
                for g in gemms
            )
            if geometry[1:] != geometry[:-1]:
                raise ValueError(
                    f"mixed Trellis kernels disagree on {phase} geometry: {geometry}"
                )
        fc2_factor = int(driver.fc2.schedule_route_block_factor)
        expected_factor = int(driver.moe_block_size // driver.fc2.moe_block_size)
        if fc2_factor < 1 or expected_factor % fc2_factor != 0:
            raise ValueError(
                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        self.descriptor_bits = descriptor_bits
        if descriptor_bits not in (9, 10):
            raise ValueError("descriptor width must be 9 or 10 bits")
        if max(tier0.num_experts, tier1.num_experts) > (1 << descriptor_bits):
            raise ValueError("tier-local expert ids exceed descriptor width")
        if driver.num_experts != tier0.num_experts + tier1.num_experts:
            raise ValueError("driver expert count must equal the sum of both tiers")
        if paired_boundary not in (None, "first", "last"):
            raise ValueError("paired boundary must be first, last, or None")
        if paired_boundary is not None:
            if driver.intermediate_size < 256 or driver.intermediate_size % 128:
                raise ValueError("paired boundary requires at least two whole resident H128 blocks")
            for moe in (driver, tier0, tier1):
                if not moe.fc1.schedule_whole_tiles or not moe.fc2.schedule_whole_tiles:
                    raise ValueError("paired boundary requires whole-tile scheduling")
                if 128 % moe.fc1.tile_n or 128 % moe.fc2.tile_k:
                    raise ValueError("paired boundary must align to FC1 N and FC2 K tiles")
        if compact_weights:
            if paired_boundary is not None:
                raise ValueError("compact weights use live widths, not paired boundary flags")
            if driver.intermediate_size < 128 or driver.intermediate_size % 128:
                raise ValueError("compact weights require whole resident H128 blocks")
            for moe in (driver, tier0, tier1):
                if not moe.fc1.schedule_whole_tiles or not moe.fc2.schedule_whole_tiles:
                    raise ValueError("compact weights require whole-tile scheduling")
                if driver.intermediate_size % moe.fc1.tile_n or 128 % moe.fc2.tile_k:
                    raise ValueError("compact widths must align to FC1 N and FC2 K tiles")
        self.compact_weights = bool(compact_weights)
        self.paired_boundary = paired_boundary
        self._token_major_rotation = bool(token_major_rotation)
        self.driver = driver
        self.tier0 = tier0
        self.tier1 = tier1
        self.size_m = driver.size_m
        self.hidden_size = driver.hidden_size
        self.intermediate_size = driver.intermediate_size
        self.top_k = driver.top_k
        self.cta_threads = driver.cta_threads
        self.blocks_per_sm = min(
            driver.blocks_per_sm, tier0.blocks_per_sm, tier1.blocks_per_sm
        )
        self.shared_words = max(
            driver.shared_words, tier0.shared_words, tier1.shared_words
        )
        # Each bitrate has a different GEMM scratch footprint. The shared T12
        # table must follow the largest pre-LUT region, otherwise a wider tier
        # can overwrite a table placed at the driver's (K3) offset.
        self.sqg_xor_cheb_t12_smem_off = max(
            driver.sqg_xor_cheb_t12_smem_off,
            tier0.sqg_xor_cheb_t12_smem_off,
            tier1.sqg_xor_cheb_t12_smem_off,
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis",
            self.ABI_VERSION,
            self.driver.__cache_key__,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            # Expert counts are runtime artifact data. All E-sized views and
            # dispatch bounds are reconstructed from the launch scalars.
            self.blocks_per_sm,
            self.shared_words,
            self.paired_boundary,
            self.token_major_rotation,
        ) + (() if self.descriptor_bits == 9 else ("descriptor_bits", self.descriptor_bits)) + (("compact_weights_v2",) if self.compact_weights else ())

    @cute.jit
    def _dispatch_tier_gemm(
        self,
        gemm,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        b_flat: cute.Tensor,
        c_flat: cute.Tensor,
        scales_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        local_expert: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        factor = gemm.schedule_route_block_factor
        first_route_block = route_block_idx * Int32(factor)
        first_lock_slot = lock_slot * Int32(factor)
        for subtile in cutlass.range_constexpr(factor):
            gemm._run_tile(
                a_flat,
                a_alt_flat,
                b_flat,
                c_flat,
                scales_flat,
                global_scale,
                packed_route_indices,
                topk_weights,
                c_tmp,
                locks,
                trellis_lut_addr,
                smem_base,
                tid,
                first_route_block + Int32(subtile),
                local_expert,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                first_lock_slot + Int32(subtile),
                active_size_m,
            )

    def _dispatch_bound(self, is_fc1, combined_expert, gemm, compact_weight, *args):
        """Trace-time Python: bind the fused FC1 input rotation (SUH tables and
        the tile's combined expert) around one tier GEMM dispatch."""
        if is_fc1 and self.driver.fused_input_rotation:
            gemm._rotation = (*self._fused_suh, combined_expert)
        gemm._compact_weight = compact_weight
        try:
            self._dispatch_tier_gemm(gemm, *args)
        finally:
            gemm._rotation = None
            gemm._compact_weight = None

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
        if cutlass.const_expr(self.driver.direct_topk_routes):
            combined_expert = packed_route_indices[route_block_idx].to(Int32)
        else:
            metadata_block_idx = route_block_idx
            if cutlass.const_expr(not is_fc1):
                metadata_block_idx = route_block_idx // Int32(
                    self.driver.moe_block_size
                    // (
                        self.driver.fc2.moe_block_size
                        * self.driver.fc2.schedule_route_block_factor
                    )
                )
            combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
        total_experts = tier0_num_experts + tier1_num_experts
        # Gate and up projections may assign one expert to different tiers, so
        # FC1 selects a descriptor row from its output-N half. FC2 always uses
        # the down-projection row. The projection-aware tile policy keeps the
        # FC1 half boundary aligned to whole CTA N tiles.
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
            compact_weight = None
            if cutlass.const_expr(self.compact_weights):
                # Rows 0..2 retain tier/local IDs. Row 3 is the true H128
                # width; rows 4..6 are projection offsets in uint32 words.
                # Binding requires positive tile-aligned widths. A single
                # FC1 tile per half therefore proves every live width is full.
                full_width = self.driver.intermediate_size == self.driver.fc1.tile_n
                if cutlass.const_expr(full_width):
                    live_width = Int32(self.driver.intermediate_size)
                else:
                    live_width = descriptor_map[
                        Int64(3) * total_experts.to(Int64) + combined_expert.to(Int64)
                    ].to(Int32)
                weight_offset = descriptor_map[
                    (descriptor_row.to(Int64) + Int64(4)) * total_experts.to(Int64)
                    + combined_expert.to(Int64)
                ].to(Int64)
                compact_weight = (weight_offset, live_width, is_fc1, full_width)
                if cutlass.const_expr(not full_width):
                    if cutlass.const_expr(is_fc1):
                        if output_n_tile % fc1_half_tiles >= live_width // Int32(self.driver.fc1.tile_n):
                            descriptor = Int32(-1)
                    else:
                        reduce_tile_count = live_width // Int32(self.driver.fc2.tile_k)
            if cutlass.const_expr(self.paired_boundary is not None):
                # The fourth descriptor row is mutable batch metadata: one
                # means compute the boundary block, zero means omit it. The
                # first three rows and all physical tensor strides stay fixed.
                owns_boundary = descriptor_map[
                    Int64(3) * total_experts.to(Int64) + combined_expert.to(Int64)
                ].to(Int32)
                if owns_boundary == Int32(0):
                    if cutlass.const_expr(is_fc1):
                        local_n = output_n_tile % fc1_half_tiles
                        if cutlass.const_expr(self.paired_boundary == "first"):
                            if local_n < Int32(128 // self.driver.fc1.tile_n):
                                descriptor = Int32(-1)
                        else:
                            if local_n >= Int32((self.driver.intermediate_size - 128) // self.driver.fc1.tile_n):
                                descriptor = Int32(-1)
                    else:
                        # Whole-tile scheduling assigns the full reduction to
                        # this CTA. Trim actual K reads, retaining the resident
                        # physical A/B strides and the ordinary output store.
                        reduce_tile_count = Int32((self.driver.intermediate_size - 128) // self.driver.fc2.tile_k)
                        if cutlass.const_expr(self.paired_boundary == "first"):
                            reduce_k_tile = Int32(128 // self.driver.fc2.tile_k)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(self.descriptor_bits)
                local_expert = descriptor & Int32((1 << self.descriptor_bits) - 1)
                # FC1 is bounded by the tier's FC1 slot count; FC2 by its own
                # independent count, since per-projection membership lets the
                # two differ. Both remain real bounds.
                if cutlass.const_expr(is_fc1):
                    tier0_in_bounds = local_expert < tier0_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        tier0_in_bounds = local_expert < tier0_up_experts
                else:
                    tier0_in_bounds = local_expert < tier0_fc2_experts
                if tier == Int32(0) and tier0_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier0.fc1
                    else:
                        gemm = self.tier0.fc2
                    self._dispatch_bound(
                        is_fc1,
                        combined_expert,
                        gemm,
                        compact_weight,
                        a_flat,
                        a_alt_flat,
                        t0_b_flat,
                        c_flat,
                        t0_scales_flat,
                        t0_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
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
                    self._dispatch_bound(
                        is_fc1,
                        combined_expert,
                        gemm,
                        compact_weight,
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
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

    @cute.jit
    def _prefetch_tier_tile(
        self,
        is_fc1: cutlass.Constexpr,
        t0_b_flat: cute.Tensor,
        t1_b_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        tid: Int32,
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
    ):
        """L2 weight schedule: resolve a whole tile's expert exactly as
        _emit_tier_tile does (packed routes, disjoint ownership) and prefetch
        its first weight K tiles with the owning tier's GEMM geometry."""
        metadata_block_idx = route_block_idx
        if cutlass.const_expr(not is_fc1):
            metadata_block_idx = route_block_idx // Int32(
                self.driver.moe_block_size
                // (
                    self.driver.fc2.moe_block_size
                    * self.driver.fc2.schedule_route_block_factor
                )
            )
        combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
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
                    tier1_in_bounds = local_expert < tier1_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        tier0_in_bounds = local_expert < tier0_up_experts
                        tier1_in_bounds = local_expert < tier1_up_experts
                else:
                    tier0_in_bounds = local_expert < tier0_fc2_experts
                    tier1_in_bounds = local_expert < tier1_fc2_experts
                # GEMMs are named inline: CuTe DSL cannot carry a Python
                # object through a runtime branch.
                if tier == Int32(0) and tier0_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        self.tier0.fc1._prefetch_tile_b_l2(
                            t0_b_flat, packed_route_indices, tid,
                            route_block_idx, local_expert, output_n_tile,
                        )
                    else:
                        self.tier0.fc2._prefetch_tile_b_l2(
                            t0_b_flat, packed_route_indices, tid,
                            route_block_idx, local_expert, output_n_tile,
                        )
                if tier == Int32(1) and tier1_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        self.tier1.fc1._prefetch_tile_b_l2(
                            t1_b_flat, packed_route_indices, tid,
                            route_block_idx, local_expert, output_n_tile,
                        )
                    else:
                        self.tier1.fc2._prefetch_tile_b_l2(
                            t1_b_flat, packed_route_indices, tid,
                            route_block_idx, local_expert, output_n_tile,
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
        # Projection-count scalars form an ABI tail and callers pass them by
        # keyword, so the preceding positional slots remain stable. CuTe DSL
        # traces every parameter and cannot type a None default.
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        # FC2 extents are independent of the FC1 slot counts.
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        # The w13 descriptor is sized by the gate count so the GEMM's
        # cute.size(w13)//2 up-block base lands at gate_count*proj_stride over
        # a tight [gate|up] buffer. When projection counts are omitted, the
        # binding contract requires one gate and one up plane per tier slot.
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        total_experts = tier0_experts + tier1_experts

        t0_w13 = cute.make_tensor(
            t0_w13_ptr,
            layout=cute.make_layout(
                (
                    tier0_gate
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(self.driver.fc1_cols // 16)
                    * cutlass.Int64(8 * self.tier0.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t0_w2 = cute.make_tensor(
            t0_w2_ptr,
            layout=cute.make_layout(
                (
                    tier0_fc2
                    * cutlass.Int64(self.intermediate_size // 16)
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(8 * self.tier0.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t1_w13 = cute.make_tensor(
            t1_w13_ptr,
            layout=cute.make_layout(
                (
                    tier1_gate
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(self.driver.fc1_cols // 16)
                    * cutlass.Int64(8 * self.tier1.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t1_w2 = cute.make_tensor(
            t1_w2_ptr,
            layout=cute.make_layout(
                (
                    tier1_fc2
                    * cutlass.Int64(self.intermediate_size // 16)
                    * cutlass.Int64(self.hidden_size // 16)
                    * cutlass.Int64(8 * self.tier1.trellis_bits),
                ),
                stride=(1,),
            ),
        )
        t0_w13_scales = cute.make_tensor(
            t0_w13_scales_ptr,
            layout=cute.make_layout(
                (
                    tier0_experts
                    * cutlass.Int64(self.tier0.fc1.scale_k_groups)
                    * cutlass.Int64(self.tier0.fc1.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t0_w2_scales = cute.make_tensor(
            t0_w2_scales_ptr,
            layout=cute.make_layout(
                (
                    tier0_experts
                    * cutlass.Int64(self.tier0.fc2.scale_k_groups)
                    * cutlass.Int64(self.tier0.fc2.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t1_w13_scales = cute.make_tensor(
            t1_w13_scales_ptr,
            layout=cute.make_layout(
                (
                    tier1_experts
                    * cutlass.Int64(self.tier1.fc1.scale_k_groups)
                    * cutlass.Int64(self.tier1.fc1.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t1_w2_scales = cute.make_tensor(
            t1_w2_scales_ptr,
            layout=cute.make_layout(
                (
                    tier1_experts
                    * cutlass.Int64(self.tier1.fc2.scale_k_groups)
                    * cutlass.Int64(self.tier1.fc2.scale_size_n // 4),
                ),
                stride=(1,),
            ),
        )
        t0_w13_global = cute.make_tensor(
            t0_w13_global_ptr,
            layout=cute.make_layout((tier0_experts,), stride=(1,)),
        )
        t0_w2_global = cute.make_tensor(
            t0_w2_global_ptr,
            layout=cute.make_layout((tier0_fc2,), stride=(1,)),
        )
        t1_w13_global = cute.make_tensor(
            t1_w13_global_ptr,
            layout=cute.make_layout((tier1_experts,), stride=(1,)),
        )
        t1_w2_global = cute.make_tensor(
            t1_w2_global_ptr,
            layout=cute.make_layout((tier1_fc2,), stride=(1,)),
        )
        # Descriptor rows are gate, up, and down. A projection-independent
        # partition is represented by three identical rows.
        descriptor_map = cute.make_tensor(
            descriptor_map_ptr,
            layout=cute.make_layout((cutlass.Int64(self.COMPACT_DESCRIPTOR_ROWS if self.compact_weights else (4 if self.paired_boundary is not None else 3)) * total_experts,), stride=(1,)),
        )
        global_to_combined = cute.make_tensor(
            global_to_combined_ptr,
            layout=cute.make_layout(
                (route_num_experts.to(cutlass.Int64),), stride=(1,)
            ),
        )
        intermediate_rotations = cute.make_tensor(
            intermediate_rotations_ptr,
            layout=cute.make_layout(
                (total_experts * cutlass.Int64(3 * self.intermediate_size),),
                stride=(1,),
            ),
        )
        suh_rows = total_experts
        if cutlass.const_expr(self.driver.broadcast_suh):
            suh_rows = cutlass.Int64(1)
        gate_suh = cute.make_tensor(
            gate_suh_ptr,
            layout=cute.make_layout(
                (suh_rows * cutlass.Int64(self.hidden_size),), stride=(1,)
            ),
        )
        up_suh = cute.make_tensor(
            up_suh_ptr,
            layout=cute.make_layout(
                (suh_rows * cutlass.Int64(self.hidden_size),), stride=(1,)
            ),
        )
        trellis_lut = cute.make_tensor(
            trellis_lut_ptr,
            layout=cute.make_layout(
                (Int64(_SQG_XOR_CHEB_T12_LUT_ENTRIES),), stride=(1,)
            ),
        )
        rotation_input = cute.make_tensor(
            rotation_input_ptr,
            layout=cute.make_layout(
                (active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        topk_weights = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout(
                (active_m.to(cutlass.Int64) * cutlass.Int64(self.top_k),),
                stride=(1,),
            ),
        )
        trellis_lut = cute.make_tensor(
            trellis_lut_ptr,
            layout=cute.make_layout(
                (cutlass.Int64(_SQG_XOR_CHEB_T12_LUT_ENTRIES),), stride=(1,)
            ),
        )
        self.kernel(
            rotation_input,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            t1_w13,
            t1_w2,
            t1_w13_scales,
            t1_w2_scales,
            t1_w13_global,
            t1_w2_global,
            fc1,
            activated,
            fc2,
            packed_route_indices,
            raw_topk_ids,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            global_to_combined,
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            trellis_lut,
            tier0_num_experts,
            tier1_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            route_num_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @property
    def token_major_rotation(self) -> bool:
        """Packed-route layers rotate each token's input once for all of its
        routes instead of re-reading the token row per packed route."""
        return bool(
            self._token_major_rotation
            and not self.driver.fused_input_rotation
            and not self.driver.direct_topk_routes
            and self.paired_boundary is None
            and not self.driver.broadcast_suh
            and not self.driver.coupled_hadamard
            and self.driver.rotation_input_dtype == "bf16"
        )

    @cute.jit
    def _run_input_rotation_token_major(
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
        # One warp owns eight H128 blocks of one token: each BF16 block is read
        # once and rotated for every route of the token. Lane t owns
        # elements 4t..4t+3 of a block; rounding matches the per-route path
        # (x and x*suh to fp16, fp32 butterfly, fp16 store).
        lane = tid & Int32(31)
        warps_per_cta = Int32(self.cta_threads // 32)
        hidden = Int32(self.hidden_size)
        nblk = Int32(self.hidden_size // 128)
        groups = (nblk + Int32(_ROTATION_BLOCKS_PER_UNIT - 1)) // Int32(_ROTATION_BLOCKS_PER_UNIT)
        unit = cta * warps_per_cta + (tid >> Int32(5))
        unit_stride = grid_x * warps_per_cta
        while unit < active_m * groups:
            token = unit // groups
            first_blk = (unit - token * groups) * Int32(_ROTATION_BLOCKS_PER_UNIT)
            last_blk = first_blk + Int32(_ROTATION_BLOCKS_PER_UNIT)
            if last_blk > nblk:
                last_blk = nblk
            experts = []
            for j in cutlass.range_constexpr(self.top_k):
                global_expert = raw_topk_ids[token * Int32(self.top_k) + Int32(j)].to(Int32)
                route_expert = Int32(-1)
                if global_expert >= Int32(0) and global_expert < route_num_experts:
                    route_expert = global_to_combined[global_expert].to(Int32)
                if route_expert >= total_experts:
                    route_expert = Int32(-1)
                experts.append(route_expert)
            blk = first_blk
            while blk < last_blk:
                col = blk * Int32(128) + lane * Int32(4)
                xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, token * hidden + col))
                x0, x1 = bfloat2_to_float2_scaled(xw0, cutlass.Float32(1.0))
                x2, x3 = bfloat2_to_float2_scaled(xw1, cutlass.Float32(1.0))
                x0, x1 = half2_to_float2_scaled(pack_f32x2_to_f16x2(x0, x1), cutlass.Float32(1.0))
                x2, x3 = half2_to_float2_scaled(pack_f32x2_to_f16x2(x2, x3), cutlass.Float32(1.0))
                for j in cutlass.range_constexpr(self.top_k):
                    expert_j = experts[j]
                    if expert_j >= Int32(0):
                        s_off = expert_j * hidden + col
                        out_off = (token * Int32(self.top_k) + Int32(j)) * hidden + col
                        for proj in cutlass.range_constexpr(2):
                            suh = gate_suh
                            out = a_gate
                            if cutlass.const_expr(proj == 1):
                                suh = up_suh
                                out = a_up
                            sw0, sw1 = ld_global_nc_v2_u32(get_ptr_as_int64(suh, s_off))
                            s0, s1 = half2_to_float2_scaled(sw0, cutlass.Float32(1.0))
                            s2, s3 = half2_to_float2_scaled(sw1, cutlass.Float32(1.0))
                            v0, v1 = half2_to_float2_scaled(
                                pack_f32x2_to_f16x2(x0 * s0, x1 * s1), cutlass.Float32(1.0)
                            )
                            v2, v3 = half2_to_float2_scaled(
                                pack_f32x2_to_f16x2(x2 * s2, x3 * s3), cutlass.Float32(1.0)
                            )
                            h0, h1, h2, h3 = self.driver._had128_quad(v0, v1, v2, v3, lane)
                            st_global_v2_u32(
                                get_ptr_as_int64(out, out_off),
                                pack_f32x2_to_f16x2(h0, h1),
                                pack_f32x2_to_f16x2(h2, h3),
                            )
                blk += Int32(1)
            unit += unit_stride

    @cute.jit
    def _run_input_rotation_token_major_ws(
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
        lane = tid & Int32(31)
        warps_per_cta = Int32(self.cta_threads // 32)
        hidden = Int32(self.hidden_size)
        nblk = Int32(self.hidden_size // 128)
        blocks_per_unit = 2 if (self.hidden_size // 128) % 2 == 0 else _ROTATION_BLOCKS_PER_UNIT
        groups = (nblk + Int32(blocks_per_unit - 1)) // Int32(blocks_per_unit)
        unit = cta * warps_per_cta + (tid >> Int32(5))
        unit_stride = grid_x * warps_per_cta
        while unit < active_m * groups:
            token = unit // groups
            first_blk = (unit - token * groups) * Int32(blocks_per_unit)
            last_blk = first_blk + Int32(blocks_per_unit)
            if last_blk > nblk:
                last_blk = nblk
            if cutlass.const_expr(self.top_k <= 32):
                my_expert = Int32(-1)
                if lane < Int32(self.top_k):
                    global_expert = raw_topk_ids[token * Int32(self.top_k) + lane].to(Int32)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        route_expert = global_to_combined[global_expert].to(Int32)
                        if route_expert < total_experts:
                            my_expert = route_expert
                blk = first_blk
                while blk < last_blk:
                    col = blk * Int32(128) + lane * Int32(4)
                    xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, token * hidden + col))
                    x0, x1 = bfloat2_to_float2_scaled(xw0, cutlass.Float32(1.0))
                    x2, x3 = bfloat2_to_float2_scaled(xw1, cutlass.Float32(1.0))
                    x0, x1 = half2_to_float2_scaled(pack_f32x2_to_f16x2(x0, x1), cutlass.Float32(1.0))
                    x2, x3 = half2_to_float2_scaled(pack_f32x2_to_f16x2(x2, x3), cutlass.Float32(1.0))
                    j = Int32(0)
                    while j < Int32(self.top_k):
                        expert_j = cute.arch.shuffle_sync(my_expert, j)
                        if expert_j >= Int32(0):
                            s_off = expert_j * hidden + col
                            out_off = (token * Int32(self.top_k) + j) * hidden + col
                            gw0, gw1 = ld_global_nc_v2_u32(get_ptr_as_int64(gate_suh, s_off))
                            uw0, uw1 = ld_global_nc_v2_u32(get_ptr_as_int64(up_suh, s_off))
                            gs0, gs1 = half2_to_float2_scaled(gw0, cutlass.Float32(1.0))
                            gs2, gs3 = half2_to_float2_scaled(gw1, cutlass.Float32(1.0))
                            gv0, gv1 = half2_to_float2_scaled(
                                pack_f32x2_to_f16x2(x0 * gs0, x1 * gs1), cutlass.Float32(1.0)
                            )
                            gv2, gv3 = half2_to_float2_scaled(
                                pack_f32x2_to_f16x2(x2 * gs2, x3 * gs3), cutlass.Float32(1.0)
                            )
                            gh0, gh1, gh2, gh3 = self.driver._had128_quad(gv0, gv1, gv2, gv3, lane)
                            st_global_v2_u32(
                                get_ptr_as_int64(a_gate, out_off),
                                pack_f32x2_to_f16x2(gh0, gh1),
                                pack_f32x2_to_f16x2(gh2, gh3),
                            )
                            us0, us1 = half2_to_float2_scaled(uw0, cutlass.Float32(1.0))
                            us2, us3 = half2_to_float2_scaled(uw1, cutlass.Float32(1.0))
                            uv0, uv1 = half2_to_float2_scaled(
                                pack_f32x2_to_f16x2(x0 * us0, x1 * us1), cutlass.Float32(1.0)
                            )
                            uv2, uv3 = half2_to_float2_scaled(
                                pack_f32x2_to_f16x2(x2 * us2, x3 * us3), cutlass.Float32(1.0)
                            )
                            uh0, uh1, uh2, uh3 = self.driver._had128_quad(uv0, uv1, uv2, uv3, lane)
                            st_global_v2_u32(
                                get_ptr_as_int64(a_up, out_off),
                                pack_f32x2_to_f16x2(uh0, uh1),
                                pack_f32x2_to_f16x2(uh2, uh3),
                            )
                        j += Int32(1)
                    blk += Int32(1)
            else:
                experts = []
                for j in cutlass.range_constexpr(self.top_k):
                    global_expert = raw_topk_ids[token * Int32(self.top_k) + Int32(j)].to(Int32)
                    route_expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        route_expert = global_to_combined[global_expert].to(Int32)
                    if route_expert >= total_experts:
                        route_expert = Int32(-1)
                    experts.append(route_expert)
                blk = first_blk
                while blk < last_blk:
                    col = blk * Int32(128) + lane * Int32(4)
                    xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, token * hidden + col))
                    x0, x1 = bfloat2_to_float2_scaled(xw0, cutlass.Float32(1.0))
                    x2, x3 = bfloat2_to_float2_scaled(xw1, cutlass.Float32(1.0))
                    x0, x1 = half2_to_float2_scaled(pack_f32x2_to_f16x2(x0, x1), cutlass.Float32(1.0))
                    x2, x3 = half2_to_float2_scaled(pack_f32x2_to_f16x2(x2, x3), cutlass.Float32(1.0))
                    for j in cutlass.range_constexpr(self.top_k):
                        expert_j = experts[j]
                        if expert_j >= Int32(0):
                            s_off = expert_j * hidden + col
                            out_off = (token * Int32(self.top_k) + Int32(j)) * hidden + col
                            for proj in cutlass.range_constexpr(2):
                                suh = gate_suh
                                out = a_gate
                                if cutlass.const_expr(proj == 1):
                                    suh = up_suh
                                    out = a_up
                                sw0, sw1 = ld_global_nc_v2_u32(get_ptr_as_int64(suh, s_off))
                                s0, s1 = half2_to_float2_scaled(sw0, cutlass.Float32(1.0))
                                s2, s3 = half2_to_float2_scaled(sw1, cutlass.Float32(1.0))
                                v0, v1 = half2_to_float2_scaled(
                                    pack_f32x2_to_f16x2(x0 * s0, x1 * s1), cutlass.Float32(1.0)
                                )
                                v2, v3 = half2_to_float2_scaled(
                                    pack_f32x2_to_f16x2(x2 * s2, x3 * s3), cutlass.Float32(1.0)
                                )
                                h0, h1, h2, h3 = self.driver._had128_quad(v0, v1, v2, v3, lane)
                                st_global_v2_u32(
                                    get_ptr_as_int64(out, out_off),
                                    pack_f32x2_to_f16x2(h0, h1),
                                    pack_f32x2_to_f16x2(h2, h3),
                                )
                    blk += Int32(1)
            unit += unit_stride

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

        if cutlass.const_expr(self.driver.direct_topk_routes):
            map_index = cta * Int32(self.cta_threads) + tid
            map_stride = grid_x * Int32(self.cta_threads)
            live_routes = active_m * Int32(self.top_k)
            while map_index < live_routes:
                global_expert = raw_topk_ids[map_index].to(Int32)
                combined_expert = Int32(-1)
                if (
                    global_expert >= Int32(0)
                    and global_expert < route_num_experts
                ):
                    combined_expert = global_to_combined[global_expert].to(Int32)
                packed_route_indices[map_index] = combined_expert
                map_index += map_stride
            self.driver._grid_barrier(workspace, tid, grid_x)

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
        fc1_a = rotation_gate
        fc1_a_alt = rotation_up
        if cutlass.const_expr(self.driver.fused_input_rotation):
            # FC1 stages the BF16 token rows and rotates them itself.
            fc1_a = rotation_input
            fc1_a_alt = rotation_input
            self._fused_suh = (gate_suh, up_suh)
        fc1_emit = partial(
            self._emit_tier_tile,
            True,
            fc1_a,
            fc1_a_alt,
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
        # L2 weight schedule hooks (None unless the decode schedule asks).
        fc1_prefetch = None
        fc2_prefetch = None
        if cutlass.const_expr(self.driver.fc1.l2_prefetch_k_tiles > 0):
            fc1_prefetch = partial(
                self._prefetch_tier_tile,
                True,
                t0_w13,
                t1_w13,
                packed_route_indices,
                block_expert_ids,
                descriptor_map,
                tid,
                tier0_num_experts,
                tier1_num_experts,
                tier0_fc2_experts,
                tier1_fc2_experts,
                tier0_gate_experts,
                tier1_gate_experts,
                tier0_up_experts,
                tier1_up_experts,
            )
        if cutlass.const_expr(self.driver.fc2.l2_prefetch_k_tiles > 0):
            fc2_prefetch = partial(
                self._prefetch_tier_tile,
                False,
                t0_w2,
                t1_w2,
                packed_route_indices,
                block_expert_ids,
                descriptor_map,
                tid,
                tier0_num_experts,
                tier1_num_experts,
                tier0_fc2_experts,
                tier1_fc2_experts,
                tier0_gate_experts,
                tier1_gate_experts,
                tier0_up_experts,
                tier1_up_experts,
            )
        total_experts = tier0_num_experts + tier1_num_experts
        if cutlass.const_expr(self.token_major_rotation):
            if active_m >= Int32(_TOKEN_MAJOR_ROTATION_MIN_ROWS):
                self._run_input_rotation_token_major(
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
            else:
                # Few rows: spread (route, block) units over the whole grid.
                self.driver._run_input_rotation(
                    rotation_input,
                    rotation_gate,
                    rotation_up,
                    gate_suh,
                    up_suh,
                    packed_route_indices,
                    block_expert_ids,
                    packed_route_count,
                    descriptor_map,
                    total_experts,
                    total_experts,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
            self.driver._grid_barrier(workspace, tid, grid_x)
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
            input_rotated=self.token_major_rotation,
            fc1_prefetch_tile=fc1_prefetch,
            fc2_prefetch_tile=fc2_prefetch,
        )


_K4_GROUPWISE_FAMILIES = frozenset(
    {
        (4608, 384, 8),  # Inkling
        (4096, 352, 8),  # Step-5
        (6144, 256, 8),  # GLM 5.3
        (6144, 128, 4),  # ML4
        (4096, 256, 8),  # Muse
        (4096, 288, 8),  # GLM Flash K4
    }
)


@dsl_user_op
def _gw_atom_add_acq_rel_gpu_i32(
    addr: Int64, val: Int32, *, loc=None, ip=None
) -> Int32:
    """GPU-scope acquire-release global int32 atomic add returning old value."""
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
def _gw_ld_global_cg_v2_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> tuple[Uint32, Uint32]:
    """Coherent L2-cached 64-bit global load for intra-launch phase buffers."""
    result = llvm.inline_asm(
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
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _gw_cp_async4_shared_global_l2hint(
    smem_addr: Int32,
    gmem_addr: Int64,
    cache_policy: Uint64,
    *,
    loc=None,
    ip=None,
):
    """Unpredicated 16-byte ``cp.async.cg.shared.global`` with L2 cache hint."""
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


@cute.jit
def _gw_warp_scan_inc_i32(val: Int32, lane: Int32) -> Int32:
    """Inclusive intra-warp sum scan across 32 lanes."""
    for offset in cutlass.range_constexpr(5):
        step = 1 << offset
        other = cute.arch.shuffle_sync_up(
            val, offset=step, mask_and_clamp=0
        )
        if lane >= Int32(step):
            val = val + other
    return val


class W4A16MixedTrellisGroupwiseKernel(W4A16MixedTrellisKernel):
    """SM121 K4 small-M group-wise cooperative mixed-Trellis kernel (v2).

    At ``active_m <= 8`` (up to 64 routed token-expert pairs), partitions the
    cooperative launch's 128-thread ``M8 x K64 x N128`` sub-workers into an
    even count ``G in {2, 4, 6, 8}`` of independent groups inside one launch.
    Each physical SM runs two independent 128-thread sub-workers (``bar 1`` and
    ``bar 2``, 256 threads/CTA at 1 CTA/SM within the <=64 KiB L1-preserving
    SMEM carveout across both K4/K5 and K4/K6 tiers), antithetically pairing
    heavy group ``c`` with light group ``G - 1 - c`` and reversed worker index
    on each SM cluster. Groups run ``input rotation -> FC1 -> activation ->
    FC2`` asynchronously with phase-separated L2-sector-isolated group barriers.

    Both ``active_m <= 8`` and ``active_m > 8`` share a single unified 4-stage
    ``M8 x K64 x N128`` sub-worker GEMM engine with cross-tile persistent
    ``cp.async`` weight/activation lookahead, disjoint epilogue SMEM, 3-load
    32-bit K5/K6 stream alignment, and L2 ``evict_first`` weight streaming.
    """

    SUB_STAGES = 4
    SUB_A_STAGE_BYTES = 1024
    SUB_SH_FOLD_BYTES = 4096
    SUB_SH_OUT_BYTES = 2176
    GROUP_SMEM_BYTES = 1536

    OFF_HDR = 0
    OFF_GRP_RB_PREFIX = 64
    OFF_GRP_RT_PREFIX = 128
    OFF_SORTED_RB = 192
    OFF_SORTED_EXPERT = 448
    OFF_FC1_PREFIX = 704
    OFF_SORTED_HALF_TILES = 976
    OFF_LIVE_ROUTES = 1232

    TMP_KEY = 0
    TMP_EXPERT = 256
    TMP_FC1_TILES = 512
    TMP_HALF_TILES = 768
    TMP_ACT_NBLK = 1024
    TMP_LIVE_CNT = 1280
    TMP_SORTED_FC1 = 1536
    TMP_SORTED_CNT = 1792
    TMP_RT_PREFIX = 2048
    TMP_ROUTES = 2320

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        descriptor_bits: int = 9,
        compact_weights: bool = False,
        force_blocks_per_sm: int | None = None,
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
        self.tier0_bits = int(tier0.trellis_bits)
        self.tier1_bits = int(tier1.trellis_bits)
        self.max_bits = max(self.tier0_bits, self.tier1_bits)
        if (
            self.cta_threads not in (128, 256)
            or self.tier0_bits != 4
            or self.tier1_bits not in (5, 6)
            or self.driver.moe_block_size != 8
            or self.driver.fc2.moe_block_size != 8
            or self.driver.fc2.schedule_route_block_factor != 1
            or self.driver.fc1.tile_k != 64
            or self.driver.fc2.tile_k != 64
            or self.driver.direct_topk_routes
            or self.driver.fused_input_rotation
            or self.driver.broadcast_suh
            or self.driver.coupled_hadamard
            or self.driver.rotation_input_dtype != "bf16"
            or self.driver.sqg_xor_cheb_t12_smem
            or self.driver.fc1.l2_prefetch_k_tiles != 0
            or self.driver.fc2.l2_prefetch_k_tiles != 0
            or self.driver.phase_l2_prefetch
            or self.hidden_size % 256 != 0
            or self.intermediate_size % 128 != 0
        ):
            raise ValueError(
                "W4A16MixedTrellisGroupwiseKernel requires packed M8 MCG K4/(K5|K6) BF16-input geometry"
            )

        self.full_compact = bool(
            self.intermediate_size == int(self.driver.fc1.tile_n)
        )
        if force_blocks_per_sm == 2:
            if int(self.driver.fc1.tile_n) != 128:
                raise ValueError(
                    "two blocks/SM override requires 128-wide tile configuration"
                )
            self.dual_sub_workers = False
            self.cta_threads = 128
            self.blocks_per_sm = 2
        else:
            self.dual_sub_workers = True
            self.cta_threads = 256
            self.blocks_per_sm = 1

        self.sub_b_stage_bytes = 4 * 8 * 32 * self.max_bits
        self.sub_sh_b_off = 0
        self.sub_sh_a_off = self.SUB_STAGES * self.sub_b_stage_bytes
        self.disjoint_fold_out = bool(
            self.dual_sub_workers and self.max_bits <= 5
        )
        self.sub_sh_fold_off = (
            self.sub_sh_a_off + self.SUB_STAGES * self.SUB_A_STAGE_BYTES
        )
        if self.disjoint_fold_out:
            self.sub_sh_out_off = (
                self.sub_sh_fold_off + self.SUB_SH_FOLD_BYTES
            )
            self.sub_worker_smem_bytes = (
                self.sub_sh_out_off + self.SUB_SH_OUT_BYTES
            )
        else:
            self.sub_sh_out_off = self.sub_sh_fold_off
            self.sub_worker_smem_bytes = (
                self.sub_sh_fold_off + self.SUB_SH_OUT_BYTES
            )

        workers_per_cta = 2 if self.dual_sub_workers else 1
        self.group_smem_off = workers_per_cta * self.sub_worker_smem_bytes
        raw_total_bytes = self.group_smem_off + self.GROUP_SMEM_BYTES
        aligned_total_bytes = ((raw_total_bytes + 1023) // 1024) * 1024
        if self.dual_sub_workers and aligned_total_bytes + 1024 > 65536:
            raise ValueError(
                "dual sub-worker shared memory exceeds 64 KiB L1-preserving carveout"
            )
        self.shared_words = aligned_total_bytes // 4

        self.gemm_fc1_tile_n = 128
        self.gemm_fc2_tile_n = 128
        self.gemm_fc1_n_tiles = int(self.driver.fc1_cols) // 128
        self.gemm_fc1_half_tiles = self.gemm_fc1_n_tiles // 2
        self.gemm_fc2_n_tiles = int(self.hidden_size) // 128
        self.gemm_fc1_k_tiles = int(self.hidden_size) // 64
        self.gemm_fc2_k_tiles = int(self.intermediate_size) // 64
        self.worker_threads = 128
        self.worker_warps = 4

        avail_locks = int(self.driver.barrier_count_off)
        if avail_locks < 6:
            raise ValueError("insufficient workspace capacity for group barriers")
        self.max_groups = min(8, avail_locks // 6)
        if avail_locks >= 256:
            self.group_barrier_stride = 32
        elif avail_locks >= 128:
            self.group_barrier_stride = 16
        else:
            self.group_barrier_stride = max(
                6, avail_locks // max(self.max_groups, 1)
            )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return super().__cache_key__ + (
            (
                "k4_groupwise_v2",
                self.cta_threads,
                self.blocks_per_sm,
                self.dual_sub_workers,
                self.disjoint_fold_out,
                self.sub_worker_smem_bytes,
                self.group_smem_off,
                self.max_groups,
                self.group_barrier_stride,
            ),
        )

    @cute.jit
    def _worker_sync(self, sub_bar_id: Int32):
        if cutlass.const_expr(self.dual_sub_workers):
            cute.arch.barrier(
                barrier_id=sub_bar_id,
                number_of_threads=128,
            )
        else:
            cute.arch.sync_threads()

    @cute.jit
    def _group_barrier(
        self,
        workspace: cute.Tensor,
        sub_tid: Int32,
        sub_bar_id: Int32,
        group_id: Int32,
        workers_per_group: Int32,
        phase: cutlass.Constexpr[int],
    ):
        self._worker_sync(sub_bar_id)
        if workers_per_group > Int32(1):
            if sub_tid == Int32(0):
                base_off = group_id * Int32(self.group_barrier_stride) + Int32(
                    2 * int(phase)
                )
                count_addr = get_ptr_as_int64(workspace, base_off)
                sense_addr = get_ptr_as_int64(workspace, base_off + Int32(1))
                old_sense = ld_global_acquire_i32(sense_addr)
                old_count = _gw_atom_add_acq_rel_gpu_i32(count_addr, Int32(1))
                if old_count == workers_per_group - Int32(1):
                    st_global_i32(count_addr, Int32(0))
                    red_add_global_release_i32(sense_addr, Int32(1))
                else:
                    sense = old_sense
                    while sense == old_sense:
                        sense = ld_global_acquire_i32(sense_addr)
            self._worker_sync(sub_bar_id)

    @cute.jit
    def _select_group_count(
        self,
        d_valid: Int32,
        total_workers: Int32,
    ) -> Int32:
        """Select even ``G in {4, 6, 8}`` (or ``2``/``1`` fallback) minimizing
        physical SM-cluster K64xN128 wave work plus barrier cost."""
        g_sel = Int32(1)
        if d_valid >= Int32(2):
            best_g = Int32(0)
            best_cost = Int32(1 << 29)
            fc1_nt = Int32(self.gemm_fc1_n_tiles)
            fc1_kt = Int32(self.gemm_fc1_k_tiles)
            fc2_nt = Int32(self.gemm_fc2_n_tiles)
            fc2_kt = Int32(self.gemm_fc2_k_tiles)
            for cand in cutlass.range_constexpr(3):
                g_val = 4 if cand == 0 else (6 if cand == 1 else 8)
                if cutlass.const_expr(g_val <= self.max_groups):
                    if d_valid >= Int32(g_val) and (
                        total_workers % Int32(g_val)
                    ) == Int32(0):
                        h_val = Int32(g_val // 2)
                        wpg = total_workers // Int32(g_val)
                        e_pair = (d_valid + h_val - Int32(1)) // h_val
                        sm_fc1_waves = (
                            e_pair * fc1_nt + wpg - Int32(1)
                        ) // wpg
                        sm_fc2_waves = (
                            e_pair * fc2_nt + wpg - Int32(1)
                        ) // wpg
                        sm_k64_work = (
                            sm_fc1_waves * fc1_kt + sm_fc2_waves * fc2_kt
                        )
                        e0 = (e_pair + Int32(1)) >> Int32(1)
                        grp0_k64 = (
                            ((e0 * fc1_nt + wpg - Int32(1)) // wpg) * fc1_kt
                            + ((e0 * fc2_nt + wpg - Int32(1)) // wpg) * fc2_kt
                        )
                        cost = sm_k64_work * Int32(4) + grp0_k64 + wpg * Int32(2)
                        if cost <= best_cost:
                            best_cost = cost
                            best_g = Int32(g_val)
            if best_g > Int32(0):
                g_sel = best_g
            elif (
                cutlass.const_expr(2 <= self.max_groups)
                and (total_workers % Int32(2)) == Int32(0)
            ):
                g_sel = Int32(2)
        return g_sel

    @cute.jit
    def _build_group_schedule_smem(
        self,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        total_experts: Int32,
        live_routes: Int32,
        raw_blocks: Int32,
        total_workers: Int32,
        smem_base: Int32,
        tid: Int32,
    ) -> Int32:
        """Build exact byte-weighted snake group schedule in shared memory."""
        sched_smem = smem_base + Int32(self.group_smem_off)
        tmp_smem = smem_base

        if tid < Int32(64):
            rb = tid
            key = Int32(-1)
            expert = Int32(-1)
            fc1_tiles = Int32(0)
            half_tiles = Int32(0)
            act_nblk = Int32(self.intermediate_size // 128)
            live_cnt = Int32(0)

            if rb < raw_blocks:
                ce = block_expert_ids[rb].to(Int32)
                for k in cutlass.range_constexpr(8):
                    r_k = packed_route_indices[rb * Int32(8) + Int32(k)].to(Int32)
                    st_shared_i32(
                        tmp_smem
                        + Int32(self.TMP_ROUTES)
                        + (rb * Int32(8) + Int32(k)) * Int32(4),
                        r_k,
                    )
                    if r_k >= Int32(0) and r_k < live_routes:
                        live_cnt += Int32(1)
                if live_cnt > Int32(0) and ce >= Int32(0) and ce < total_experts:
                    ce_i64 = ce.to(Int64)
                    tot_i64 = total_experts.to(Int64)
                    d0 = descriptor_map[ce_i64].to(Int32)
                    d1 = descriptor_map[tot_i64 + ce_i64].to(Int32)
                    d2 = descriptor_map[Int64(2) * tot_i64 + ce_i64].to(Int32)
                    if d0 >= Int32(0) and d1 >= Int32(0) and d2 >= Int32(0):
                        bit_delta = Int32(self.tier1_bits - self.tier0_bits)
                        b0 = Int32(self.tier0_bits) + (
                            (d0 >> Int32(self.descriptor_bits)) & Int32(1)
                        ) * bit_delta
                        b1 = Int32(self.tier0_bits) + (
                            (d1 >> Int32(self.descriptor_bits)) & Int32(1)
                        ) * bit_delta
                        b2 = Int32(self.tier0_bits) + (
                            (d2 >> Int32(self.descriptor_bits)) & Int32(1)
                        ) * bit_delta
                        live_w = Int32(self.intermediate_size)
                        if cutlass.const_expr(
                            self.compact_weights and not self.full_compact
                        ):
                            live_w = descriptor_map[
                                Int64(3) * tot_i64 + ce_i64
                            ].to(Int32)
                        half_tiles = live_w // Int32(128)
                        fc1_tiles = half_tiles * Int32(2)
                        act_nblk = half_tiles
                        if half_tiles > Int32(0):
                            score = act_nblk * (b0 + b1 + b2)
                            expert = ce
                            key = (score << Int32(8)) | (Int32(63) - rb)

            st_shared_i32(tmp_smem + Int32(self.TMP_KEY) + rb * Int32(4), key)
            st_shared_i32(tmp_smem + Int32(self.TMP_EXPERT) + rb * Int32(4), expert)
            st_shared_i32(
                tmp_smem + Int32(self.TMP_FC1_TILES) + rb * Int32(4), fc1_tiles
            )
            st_shared_i32(
                tmp_smem + Int32(self.TMP_HALF_TILES) + rb * Int32(4), half_tiles
            )
            st_shared_i32(
                tmp_smem + Int32(self.TMP_ACT_NBLK) + rb * Int32(4), act_nblk
            )
            st_shared_i32(
                tmp_smem + Int32(self.TMP_LIVE_CNT) + rb * Int32(4), live_cnt
            )
            st_shared_i32(
                tmp_smem + Int32(self.TMP_SORTED_FC1) + rb * Int32(4), Int32(0)
            )
            st_shared_i32(
                tmp_smem + Int32(self.TMP_SORTED_CNT) + rb * Int32(4), Int32(0)
            )

        cute.arch.sync_threads()

        if tid < Int32(64):
            my_key = ld_shared_i32_relaxed(
                tmp_smem + Int32(self.TMP_KEY) + tid * Int32(4)
            )
            d_valid = Int32(0)
            rk = Int32(0)
            for j4 in cutlass.range_constexpr(16):
                k0, k1, k2, k3 = ld_shared_v4_u32(
                    tmp_smem + Int32(self.TMP_KEY + j4 * 16)
                )
                i0 = Int32(k0)
                i1 = Int32(k1)
                i2 = Int32(k2)
                i3 = Int32(k3)
                d_valid += (
                    (i0 >= Int32(0)).to(Int32)
                    + (i1 >= Int32(0)).to(Int32)
                    + (i2 >= Int32(0)).to(Int32)
                    + (i3 >= Int32(0)).to(Int32)
                )
                rk += (
                    (i0 > my_key).to(Int32)
                    + (i1 > my_key).to(Int32)
                    + (i2 > my_key).to(Int32)
                    + (i3 > my_key).to(Int32)
                )

            if d_valid > Int32(0):
                num_groups = self._select_group_count(d_valid, total_workers)
                workers_per_group = total_workers // num_groups
                q_grp = d_valid // num_groups
                r_grp = d_valid - q_grp * num_groups

                if tid <= num_groups:
                    extra = tid
                    if (q_grp & Int32(1)) == Int32(0):
                        if extra > r_grp:
                            extra = r_grp
                    else:
                        extra = tid - (num_groups - r_grp)
                        if extra < Int32(0):
                            extra = Int32(0)
                    st_shared_i32(
                        sched_smem
                        + Int32(self.OFF_GRP_RB_PREFIX)
                        + tid * Int32(4),
                        tid * q_grp + extra,
                    )

                if my_key >= Int32(0):
                    row_in_grp = rk // num_groups
                    rem = rk - row_in_grp * num_groups
                    grp = rem
                    if (row_in_grp & Int32(1)) != Int32(0):
                        grp = num_groups - Int32(1) - rem
                    extra_g = grp
                    if (q_grp & Int32(1)) == Int32(0):
                        if extra_g > r_grp:
                            extra_g = r_grp
                    else:
                        extra_g = grp - (num_groups - r_grp)
                        if extra_g < Int32(0):
                            extra_g = Int32(0)
                    pos = grp * q_grp + extra_g + row_in_grp
                    st_shared_i32(
                        sched_smem + Int32(self.OFF_SORTED_RB) + pos * Int32(4),
                        tid,
                    )
                    st_shared_i32(
                        sched_smem
                        + Int32(self.OFF_SORTED_EXPERT)
                        + pos * Int32(4),
                        ld_shared_i32_relaxed(
                            tmp_smem + Int32(self.TMP_EXPERT) + tid * Int32(4)
                        ),
                    )
                    st_shared_i32(
                        sched_smem
                        + Int32(self.OFF_SORTED_HALF_TILES)
                        + pos * Int32(4),
                        ld_shared_i32_relaxed(
                            tmp_smem
                            + Int32(self.TMP_HALF_TILES)
                            + tid * Int32(4)
                        ),
                    )
                    st_shared_i32(
                        tmp_smem + Int32(self.TMP_SORTED_FC1) + pos * Int32(4),
                        ld_shared_i32_relaxed(
                            tmp_smem
                            + Int32(self.TMP_FC1_TILES)
                            + tid * Int32(4)
                        ),
                    )
                    st_shared_i32(
                        tmp_smem + Int32(self.TMP_SORTED_CNT) + pos * Int32(4),
                        ld_shared_i32_relaxed(
                            tmp_smem + Int32(self.TMP_LIVE_CNT) + tid * Int32(4)
                        ),
                    )
                if tid == Int32(0):
                    st_shared_i32(sched_smem + Int32(self.OFF_HDR + 0), d_valid)
                    st_shared_i32(
                        sched_smem + Int32(self.OFF_HDR + 4), num_groups
                    )
                    st_shared_i32(
                        sched_smem + Int32(self.OFF_HDR + 8), workers_per_group
                    )
            else:
                if tid == Int32(0):
                    st_shared_i32(
                        sched_smem + Int32(self.OFF_HDR + 0), Int32(0)
                    )

        cute.arch.sync_threads()

        d_valid_cta = ld_shared_i32_relaxed(sched_smem + Int32(self.OFF_HDR + 0))
        if d_valid_cta > Int32(0):
            if tid < Int32(32):
                lane = tid
                num_groups_w = ld_shared_i32_relaxed(
                    sched_smem + Int32(self.OFF_HDR + 4)
                )
                rb0 = lane
                rb1 = lane + Int32(32)
                f0 = Int32(0)
                f1 = Int32(0)
                c0_rt = Int32(0)
                c1_rt = Int32(0)
                if rb0 < d_valid_cta:
                    f0 = ld_shared_i32_relaxed(
                        tmp_smem + Int32(self.TMP_SORTED_FC1) + rb0 * Int32(4)
                    )
                    c0_rt = ld_shared_i32_relaxed(
                        tmp_smem + Int32(self.TMP_SORTED_CNT) + rb0 * Int32(4)
                    )
                if rb1 < d_valid_cta:
                    f1 = ld_shared_i32_relaxed(
                        tmp_smem + Int32(self.TMP_SORTED_FC1) + rb1 * Int32(4)
                    )
                    c1_rt = ld_shared_i32_relaxed(
                        tmp_smem + Int32(self.TMP_SORTED_CNT) + rb1 * Int32(4)
                    )

                inc_f0 = _gw_warp_scan_inc_i32(f0, lane)
                exc_f0 = inc_f0 - f0
                tot_f0 = cute.arch.shuffle_sync(inc_f0, Int32(31))
                inc_f1 = _gw_warp_scan_inc_i32(f1, lane)
                exc_f1 = tot_f0 + inc_f1 - f1
                tot_f = tot_f0 + cute.arch.shuffle_sync(inc_f1, Int32(31))

                st_shared_i32(
                    sched_smem + Int32(self.OFF_FC1_PREFIX) + rb0 * Int32(4),
                    exc_f0,
                )
                st_shared_i32(
                    sched_smem + Int32(self.OFF_FC1_PREFIX) + rb1 * Int32(4),
                    exc_f1,
                )

                inc_c0 = _gw_warp_scan_inc_i32(c0_rt, lane)
                exc_c0 = inc_c0 - c0_rt
                tot_c0 = cute.arch.shuffle_sync(inc_c0, Int32(31))
                inc_c1 = _gw_warp_scan_inc_i32(c1_rt, lane)
                exc_c1 = tot_c0 + inc_c1 - c1_rt
                tot_c = tot_c0 + cute.arch.shuffle_sync(inc_c1, Int32(31))

                st_shared_i32(
                    tmp_smem + Int32(self.TMP_RT_PREFIX) + rb0 * Int32(4),
                    exc_c0,
                )
                st_shared_i32(
                    tmp_smem + Int32(self.TMP_RT_PREFIX) + rb1 * Int32(4),
                    exc_c1,
                )
                if lane == Int32(0):
                    st_shared_i32(
                        sched_smem + Int32(self.OFF_FC1_PREFIX) + Int32(64 * 4),
                        tot_f,
                    )
                    st_shared_i32(
                        tmp_smem + Int32(self.TMP_RT_PREFIX) + Int32(64 * 4),
                        tot_c,
                    )
                cute.arch.sync_warp()
                if lane <= num_groups_w:
                    rb_p = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_GRP_RB_PREFIX)
                        + lane * Int32(4)
                    )
                    rt_p = ld_shared_i32_relaxed(
                        tmp_smem + Int32(self.TMP_RT_PREFIX) + rb_p * Int32(4)
                    )
                    st_shared_i32(
                        sched_smem
                        + Int32(self.OFF_GRP_RT_PREFIX)
                        + lane * Int32(4),
                        rt_p,
                    )

            cute.arch.sync_threads()

            if tid < d_valid_cta:
                pos_out = tid
                orig_rb = ld_shared_i32_relaxed(
                    sched_smem + Int32(self.OFF_SORTED_RB) + pos_out * Int32(4)
                )
                ce_out = ld_shared_i32_relaxed(
                    sched_smem
                    + Int32(self.OFF_SORTED_EXPERT)
                    + pos_out * Int32(4)
                )
                nblk_out = ld_shared_i32_relaxed(
                    tmp_smem + Int32(self.TMP_ACT_NBLK) + orig_rb * Int32(4)
                )
                rt_out = ld_shared_i32_relaxed(
                    tmp_smem + Int32(self.TMP_RT_PREFIX) + pos_out * Int32(4)
                )
                packed_meta = (nblk_out << Int32(22)) | (ce_out << Int32(8))
                for k_out in cutlass.range_constexpr(8):
                    rk_out = ld_shared_i32_relaxed(
                        tmp_smem
                        + Int32(self.TMP_ROUTES)
                        + (orig_rb * Int32(8) + Int32(k_out)) * Int32(4)
                    )
                    if rk_out >= Int32(0) and rk_out < live_routes:
                        st_shared_i32(
                            sched_smem
                            + Int32(self.OFF_LIVE_ROUTES)
                            + rt_out * Int32(4),
                            packed_meta | rk_out,
                        )
                        rt_out += Int32(1)

            cute.arch.sync_threads()
        return d_valid_cta

    @cute.jit
    def _run_group_input_rotation(
        self,
        x_input: cute.Tensor,
        a_gate: cute.Tensor,
        a_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        sched_smem: Int32,
        sub_tid: Int32,
        group_id: Int32,
        sched_worker: Int32,
        workers_per_group: Int32,
    ):
        r_begin = ld_shared_i32_relaxed(
            sched_smem + Int32(self.OFF_GRP_RT_PREFIX) + group_id * Int32(4)
        )
        r_end = ld_shared_i32_relaxed(
            sched_smem
            + Int32(self.OFF_GRP_RT_PREFIX)
            + (group_id + Int32(1)) * Int32(4)
        )
        group_routes = r_end - r_begin
        nblk = Int32(self.hidden_size // 128)
        lane = sub_tid & Int32(31)
        warp_in_worker = sub_tid >> Int32(5)
        elem = lane * Int32(4)
        unit = sched_worker * Int32(self.worker_warps) + warp_in_worker
        unit_stride = workers_per_group * Int32(self.worker_warps)
        total_units = group_routes * nblk

        while unit < total_units:
            local_r = unit // nblk
            blk = unit - local_r * nblk
            packed_entry = ld_shared_i32_relaxed(
                sched_smem
                + Int32(self.OFF_LIVE_ROUTES)
                + (r_begin + local_r) * Int32(4)
            )
            route = packed_entry & Int32(0xFF)
            expert = (packed_entry >> Int32(8)) & Int32(0x3FFF)
            token = route // Int32(self.top_k)
            col0 = blk * Int32(128) + elem

            x_off = Int64(token) * Int64(self.hidden_size) + Int64(col0)
            s_off = Int64(expert) * Int64(self.hidden_size) + Int64(col0)
            out_off = Int64(route) * Int64(self.hidden_size) + Int64(col0)

            xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, x_off))
            gw0, gw1 = ld_global_nc_v2_u32(get_ptr_as_int64(gate_suh, s_off))
            uw0, uw1 = ld_global_nc_v2_u32(get_ptr_as_int64(up_suh, s_off))

            xh0 = cvt_bf16x2_to_f16x2_via_f32(xw0)
            xh1 = cvt_bf16x2_to_f16x2_via_f32(xw1)
            x0, x1 = f16x2_to_f32x2(xh0)
            x2, x3 = f16x2_to_f32x2(xh1)

            sg0, sg1 = f16x2_to_f32x2(gw0)
            sg2, sg3 = f16x2_to_f32x2(gw1)
            g0, g1 = f16x2_to_f32x2(pack_f32x2_to_f16x2(x0 * sg0, x1 * sg1))
            g2, g3 = f16x2_to_f32x2(pack_f32x2_to_f16x2(x2 * sg2, x3 * sg3))
            gh0, gh1, gh2, gh3 = self.driver._had128_quad(g0, g1, g2, g3, lane)
            st_global_v2_u32(
                get_ptr_as_int64(a_gate, out_off),
                pack_f32x2_to_f16x2(gh0, gh1),
                pack_f32x2_to_f16x2(gh2, gh3),
            )

            su0, su1 = f16x2_to_f32x2(uw0)
            su2, su3 = f16x2_to_f32x2(uw1)
            u0, u1 = f16x2_to_f32x2(pack_f32x2_to_f16x2(x0 * su0, x1 * su1))
            u2, u3 = f16x2_to_f32x2(pack_f32x2_to_f16x2(x2 * su2, x3 * su3))
            uh0, uh1, uh2, uh3 = self.driver._had128_quad(u0, u1, u2, u3, lane)
            st_global_v2_u32(
                get_ptr_as_int64(a_up, out_off),
                pack_f32x2_to_f16x2(uh0, uh1),
                pack_f32x2_to_f16x2(uh2, uh3),
            )
            unit += unit_stride

    @cute.jit
    def _run_whole_grid_input_rotation(
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
        lane = tid & Int32(31)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.hidden_size // 128)
        groups = (nblk + Int32(_ROTATION_BLOCKS_PER_UNIT - 1)) // Int32(
            _ROTATION_BLOCKS_PER_UNIT
        )
        elem = lane * Int32(4)
        unit = cta * warps_per_cta + (tid >> Int32(5))
        unit_stride = grid_x * warps_per_cta
        total_units = active_m * groups
        exp_regs = cute.make_rmem_tensor((self.top_k,), Int32)

        while unit < total_units:
            token = unit // groups
            first_blk = (unit - token * groups) * Int32(
                _ROTATION_BLOCKS_PER_UNIT
            )
            last_blk = first_blk + Int32(_ROTATION_BLOCKS_PER_UNIT)
            if last_blk > nblk:
                last_blk = nblk
            for j in cutlass.range_constexpr(self.top_k):
                global_expert = raw_topk_ids[
                    token * Int32(self.top_k) + Int32(j)
                ].to(Int32)
                route_expert = Int32(-1)
                if (
                    global_expert >= Int32(0)
                    and global_expert < route_num_experts
                ):
                    route_expert = global_to_combined[global_expert].to(Int32)
                if route_expert >= total_experts:
                    route_expert = Int32(-1)
                exp_regs[j] = route_expert

            blk = first_blk
            while blk < last_blk:
                col0 = blk * Int32(128) + elem
                x_off = token.to(Int64) * Int64(self.hidden_size) + col0.to(
                    Int64
                )
                xw0, xw1 = ld_global_nc_v2_u32(get_ptr_as_int64(x_input, x_off))
                xh0 = cvt_bf16x2_to_f16x2_via_f32(xw0)
                xh1 = cvt_bf16x2_to_f16x2_via_f32(xw1)
                x0, x1 = f16x2_to_f32x2(xh0)
                x2, x3 = f16x2_to_f32x2(xh1)
                for jr in cutlass.range_constexpr(self.top_k):
                    exp_j = exp_regs[jr]
                    if exp_j >= Int32(0):
                        route = token * Int32(self.top_k) + Int32(jr)
                        s_off = exp_j.to(Int64) * Int64(
                            self.hidden_size
                        ) + col0.to(Int64)
                        out_off = route.to(Int64) * Int64(
                            self.hidden_size
                        ) + col0.to(Int64)
                        gw0, gw1 = ld_global_nc_v2_u32(
                            get_ptr_as_int64(gate_suh, s_off)
                        )
                        uw0, uw1 = ld_global_nc_v2_u32(
                            get_ptr_as_int64(up_suh, s_off)
                        )
                        sg0, sg1 = f16x2_to_f32x2(gw0)
                        sg2, sg3 = f16x2_to_f32x2(gw1)
                        g0, g1 = f16x2_to_f32x2(
                            pack_f32x2_to_f16x2(x0 * sg0, x1 * sg1)
                        )
                        g2, g3 = f16x2_to_f32x2(
                            pack_f32x2_to_f16x2(x2 * sg2, x3 * sg3)
                        )
                        gh0, gh1, gh2, gh3 = self.driver._had128_quad(
                            g0, g1, g2, g3, lane
                        )
                        st_global_v2_u32(
                            get_ptr_as_int64(a_gate, out_off),
                            pack_f32x2_to_f16x2(gh0, gh1),
                            pack_f32x2_to_f16x2(gh2, gh3),
                        )

                        su0, su1 = f16x2_to_f32x2(uw0)
                        su2, su3 = f16x2_to_f32x2(uw1)
                        u0, u1 = f16x2_to_f32x2(
                            pack_f32x2_to_f16x2(x0 * su0, x1 * su1)
                        )
                        u2, u3 = f16x2_to_f32x2(
                            pack_f32x2_to_f16x2(x2 * su2, x3 * su3)
                        )
                        uh0, uh1, uh2, uh3 = self.driver._had128_quad(
                            u0, u1, u2, u3, lane
                        )
                        st_global_v2_u32(
                            get_ptr_as_int64(a_up, out_off),
                            pack_f32x2_to_f16x2(uh0, uh1),
                            pack_f32x2_to_f16x2(uh2, uh3),
                        )
                blk += Int32(1)
            unit += unit_stride

    @cute.jit
    def _activate_one_block(
        self,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        route: Int32,
        expert: Int32,
        blk: Int32,
        lane: Int32,
        elem: Int32,
    ):
        col0 = blk * Int32(128) + elem
        g_off = Int64(route) * Int64(self.driver.fc1_cols) + Int64(col0)
        u_off = g_off + Int64(self.intermediate_size)
        s_off = (
            Int64(expert) * Int64(3 * self.intermediate_size) + Int64(col0)
        )

        gp01, gp23 = _gw_ld_global_cg_v2_u32(get_ptr_as_int64(fc1, g_off))
        up01, up23 = _gw_ld_global_cg_v2_u32(get_ptr_as_int64(fc1, u_off))
        svg01, svg23 = ld_global_nc_v2_u32(
            get_ptr_as_int64(intermediate_rotations, s_off)
        )
        svu01, svu23 = ld_global_nc_v2_u32(
            get_ptr_as_int64(
                intermediate_rotations,
                s_off + Int64(self.intermediate_size),
            )
        )
        sd01, sd23 = ld_global_nc_v2_u32(
            get_ptr_as_int64(
                intermediate_rotations,
                s_off + Int64(2 * self.intermediate_size),
            )
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
                * cute.math.tanh(
                    iu0 / linear_beta, fastmath=self.driver.fast_math
                )
                * sd0
            )
            a1 = (
                beta
                * cute.math.tanh(ig1 / beta, fastmath=self.driver.fast_math)
                * self.driver._sigmoid_f32(ig1)
                * linear_beta
                * cute.math.tanh(
                    iu1 / linear_beta, fastmath=self.driver.fast_math
                )
                * sd1
            )
            a2 = (
                beta
                * cute.math.tanh(ig2 / beta, fastmath=self.driver.fast_math)
                * self.driver._sigmoid_f32(ig2)
                * linear_beta
                * cute.math.tanh(
                    iu2 / linear_beta, fastmath=self.driver.fast_math
                )
                * sd2
            )
            a3 = (
                beta
                * cute.math.tanh(ig3 / beta, fastmath=self.driver.fast_math)
                * self.driver._sigmoid_f32(ig3)
                * linear_beta
                * cute.math.tanh(
                    iu3 / linear_beta, fastmath=self.driver.fast_math
                )
                * sd3
            )
        else:
            a0 = self.driver._gated_silu(ig0) * iu0 * sd0
            a1 = self.driver._gated_silu(ig1) * iu1 * sd1
            a2 = self.driver._gated_silu(ig2) * iu2 * sd2
            a3 = self.driver._gated_silu(ig3) * iu3 * sd3

        o0, o1, o2, o3 = self.driver._had128_quad(a0, a1, a2, a3, lane)
        out_off = Int64(route) * Int64(self.intermediate_size) + Int64(col0)
        st_global_v2_u32(
            get_ptr_as_int64(activated, out_off),
            pack_f32x2_to_f16x2(o0, o1),
            pack_f32x2_to_f16x2(o2, o3),
        )

    @cute.jit
    def _run_group_activation(
        self,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        sched_smem: Int32,
        sub_tid: Int32,
        group_id: Int32,
        sched_worker: Int32,
        workers_per_group: Int32,
    ):
        r_begin = ld_shared_i32_relaxed(
            sched_smem + Int32(self.OFF_GRP_RT_PREFIX) + group_id * Int32(4)
        )
        r_end = ld_shared_i32_relaxed(
            sched_smem
            + Int32(self.OFF_GRP_RT_PREFIX)
            + (group_id + Int32(1)) * Int32(4)
        )
        group_routes = r_end - r_begin
        nblk = Int32(self.intermediate_size // 128)
        lane = sub_tid & Int32(31)
        warp_in_worker = sub_tid >> Int32(5)
        elem = lane * Int32(4)
        unit = sched_worker * Int32(self.worker_warps) + warp_in_worker
        unit_stride = workers_per_group * Int32(self.worker_warps)
        total_units = group_routes * nblk

        while unit < total_units:
            local_r = unit // nblk
            blk = unit - local_r * nblk
            packed_entry = ld_shared_i32_relaxed(
                sched_smem
                + Int32(self.OFF_LIVE_ROUTES)
                + (r_begin + local_r) * Int32(4)
            )
            live_nblk = (packed_entry >> Int32(22)) & Int32(0xFF)
            if blk < live_nblk:
                route = packed_entry & Int32(0xFF)
                expert = (packed_entry >> Int32(8)) & Int32(0x3FFF)
                self._activate_one_block(
                    fc1,
                    activated,
                    intermediate_rotations,
                    route,
                    expert,
                    blk,
                    lane,
                    elem,
                )
            unit += unit_stride

    @cute.jit
    def _run_whole_grid_activation(
        self,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        raw_topk_ids: cute.Tensor,
        global_to_combined: cute.Tensor,
        descriptor_map: cute.Tensor,
        route_num_experts: Int32,
        total_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        lane = tid & Int32(31)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.intermediate_size // 128)
        live_routes = active_m * Int32(self.top_k)
        elem = lane * Int32(4)
        unit = cta * warps_per_cta + (tid >> Int32(5))
        unit_stride = grid_x * warps_per_cta
        total_units = live_routes * nblk

        while unit < total_units:
            route = unit // nblk
            blk = unit - route * nblk
            global_expert = raw_topk_ids[route].to(Int32)
            if global_expert >= Int32(0) and global_expert < route_num_experts:
                expert = global_to_combined[global_expert].to(Int32)
                if expert >= Int32(0) and expert < total_experts:
                    live_nblk = nblk
                    if cutlass.const_expr(
                        self.compact_weights and not self.full_compact
                    ):
                        live_nblk = (
                            descriptor_map[
                                Int64(3) * total_experts.to(Int64)
                                + expert.to(Int64)
                            ].to(Int32)
                            // Int32(128)
                        )
                    if blk < live_nblk:
                        self._activate_one_block(
                            fc1,
                            activated,
                            intermediate_rotations,
                            route,
                            expert,
                            blk,
                            lane,
                            elem,
                        )
            unit += unit_stride

    @cute.jit
    def _compute_tier_b_addr_and_stride(
        self,
        is_fc1: cutlass.Constexpr[bool],
        bits: cutlass.Constexpr[int],
        b_flat: cute.Tensor,
        local_expert: Int32,
        output_n_tile: Int32,
        weight_offset: Int64,
        live_width: Int32,
        tier_gate_experts: Int32,
    ):
        tile_u32 = 8 * int(bits)
        fc1_half_tiles = Int32(self.gemm_fc1_half_tiles)
        b_base_u32 = Int64(0)
        b_stride_u32 = Int64(0)
        if cutlass.const_expr(self.compact_weights):
            if cutlass.const_expr(is_fc1):
                compact_n16 = Int64(self.intermediate_size // 16)
                if cutlass.const_expr(not self.full_compact):
                    compact_n16 = live_width.to(Int64) // Int64(16)
                local_n16 = (output_n_tile % fc1_half_tiles).to(Int64) * Int64(
                    8
                )
            else:
                compact_n16 = Int64(self.hidden_size // 16)
                local_n16 = output_n_tile.to(Int64) * Int64(8)
            b_base_u32 = weight_offset + local_n16 * Int64(tile_u32)
            b_stride_u32 = compact_n16 * Int64(tile_u32)
        elif cutlass.const_expr(is_fc1):
            half_n16 = self.intermediate_size // 16
            proj = (output_n_tile >= fc1_half_tiles).to(Int64)
            local_n16 = (
                output_n_tile - proj.to(Int32) * fc1_half_tiles
            ).to(Int64) * Int64(8)
            proj_expert_u32 = Int64(
                (self.hidden_size // 16) * half_n16 * tile_u32
            )
            plane_u32 = tier_gate_experts.to(Int64) * proj_expert_u32
            b_base_u32 = (
                proj * plane_u32
                + local_expert.to(Int64) * proj_expert_u32
                + local_n16 * Int64(tile_u32)
            )
            b_stride_u32 = Int64(half_n16 * tile_u32)
        else:
            n16 = self.hidden_size // 16
            expert_u32 = Int64(
                (self.intermediate_size // 16) * n16 * tile_u32
            )
            b_base_u32 = local_expert.to(Int64) * expert_u32 + output_n_tile.to(
                Int64
            ) * Int64(8 * tile_u32)
            b_stride_u32 = Int64(n16 * tile_u32)
        return (
            get_ptr_as_int64(b_flat, b_base_u32),
            b_stride_u32 * Int64(4),
        )

    @cute.jit
    def _find_next_valid_tile(
        self,
        is_fc1: cutlass.Constexpr[bool],
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        sched_smem: Int32,
        sub_tid: Int32,
        mode_groupwise: Int32,
        b_begin: Int32,
        b_end: Int32,
        base_fc1_t: Int32,
        start_task: Int32,
        task_step: Int32,
        task_limit: Int32,
        live_routes: Int32,
        total_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
    ):
        found = Int32(0)
        out_task = start_task
        out_tier = Int32(0)
        out_kt = Int32(
            self.gemm_fc1_k_tiles if is_fc1 else self.gemm_fc2_k_tiles
        )
        out_b_byte = Int64(0)
        out_b_stride = Int64(0)
        out_a_base = Int64(0)
        out_a_valid = Int32(0)
        out_epi_addr = Int64(0)
        out_epi_valid = Int32(0)
        out_scale = cutlass.Float32(0.0)

        fc1_half_tiles = Int32(self.gemm_fc1_half_tiles)
        n_tiles = Int32(
            self.gemm_fc1_n_tiles if is_fc1 else self.gemm_fc2_n_tiles
        )
        cand = start_task
        while cand < task_limit:
            next_cand = cand + task_step
            rb = Int32(0)
            ce = Int32(-1)
            output_n_tile = Int32(0)
            if mode_groupwise != Int32(0):
                if cutlass.const_expr(
                    is_fc1 and self.compact_weights and not self.full_compact
                ):
                    target = base_fc1_t + cand
                    lo = b_begin
                    hi = b_end
                    for _ in cutlass.range_constexpr(6):
                        mid = (lo + hi) >> Int32(1)
                        if mid > lo:
                            mid_val = ld_shared_i32_relaxed(
                                sched_smem
                                + Int32(self.OFF_FC1_PREFIX)
                                + mid * Int32(4)
                            )
                            if mid_val <= target:
                                lo = mid
                            else:
                                hi = mid
                    pos = lo
                    pos_base = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_FC1_PREFIX)
                        + pos * Int32(4)
                    )
                    local_t = target - pos_base
                    live_half = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_SORTED_HALF_TILES)
                        + pos * Int32(4)
                    )
                    output_n_tile = local_t
                    if local_t >= live_half:
                        output_n_tile = fc1_half_tiles + (local_t - live_half)
                    rb = ld_shared_i32_relaxed(
                        sched_smem + Int32(self.OFF_SORTED_RB) + pos * Int32(4)
                    )
                    ce = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_SORTED_EXPERT)
                        + pos * Int32(4)
                    )
                else:
                    local_b = cand // n_tiles
                    output_n_tile = cand - local_b * n_tiles
                    pos = b_begin + local_b
                    rb = ld_shared_i32_relaxed(
                        sched_smem + Int32(self.OFF_SORTED_RB) + pos * Int32(4)
                    )
                    ce = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_SORTED_EXPERT)
                        + pos * Int32(4)
                    )
            else:
                if cutlass.const_expr(self.size_m > 8):
                    rb = cand // n_tiles
                    output_n_tile = cand - rb * n_tiles
                    ce = block_expert_ids[rb].to(Int32)

            if ce >= Int32(0) and ce < total_experts:
                descriptor_row = Int32(2)
                if cutlass.const_expr(is_fc1):
                    descriptor_row = Int32(0)
                    if output_n_tile >= fc1_half_tiles:
                        descriptor_row = Int32(1)
                tot_i64 = total_experts.to(Int64)
                ce_i64 = ce.to(Int64)
                descriptor = descriptor_map[
                    descriptor_row.to(Int64) * tot_i64 + ce_i64
                ].to(Int32)
                live_width = Int32(self.intermediate_size)
                weight_offset = Int64(0)
                cand_kt = Int32(
                    self.gemm_fc1_k_tiles if is_fc1 else self.gemm_fc2_k_tiles
                )
                if cutlass.const_expr(self.compact_weights):
                    if cutlass.const_expr(not self.full_compact):
                        live_width = descriptor_map[
                            Int64(3) * tot_i64 + ce_i64
                        ].to(Int32)
                        if cutlass.const_expr(is_fc1):
                            if (output_n_tile % fc1_half_tiles) >= (
                                live_width // Int32(128)
                            ):
                                descriptor = Int32(-1)
                        else:
                            cand_kt = live_width // Int32(64)
                            if cand_kt < Int32(2):
                                descriptor = Int32(-1)
                    weight_offset = descriptor_map[
                        (descriptor_row.to(Int64) + Int64(4)) * tot_i64
                        + ce_i64
                    ].to(Int64)

                if descriptor >= Int32(0) and cand_kt >= Int32(2):
                    cand_tier = descriptor >> Int32(self.descriptor_bits)
                    local_expert = descriptor & Int32(
                        (1 << self.descriptor_bits) - 1
                    )
                    if cutlass.const_expr(is_fc1):
                        t0_lim = tier0_gate_experts
                        t1_lim = tier1_gate_experts
                        if output_n_tile >= fc1_half_tiles:
                            t0_lim = tier0_up_experts
                            t1_lim = tier1_up_experts
                    else:
                        t0_lim = tier0_fc2_experts
                        t1_lim = tier1_fc2_experts

                    tier_ok = Int32(0)
                    if cand_tier == Int32(0) and local_expert < t0_lim:
                        tier_ok = Int32(1)
                    if cand_tier == Int32(1) and local_expert < t1_lim:
                        tier_ok = Int32(1)
                    if tier_ok != Int32(0):
                        found = Int32(1)
                        next_cand = task_limit
                        out_task = cand
                        out_tier = cand_tier
                        out_kt = cand_kt
                        if cand_tier == Int32(0):
                            out_scale = t0_global_scale[local_expert].to(
                                cutlass.Float32
                            )
                            out_b_byte, out_b_stride = (
                                self._compute_tier_b_addr_and_stride(
                                    is_fc1,
                                    self.tier0_bits,
                                    t0_b_flat,
                                    local_expert,
                                    output_n_tile,
                                    weight_offset,
                                    live_width,
                                    tier0_gate_experts,
                                )
                            )
                        else:
                            out_scale = t1_global_scale[local_expert].to(
                                cutlass.Float32
                            )
                            out_b_byte, out_b_stride = (
                                self._compute_tier_b_addr_and_stride(
                                    is_fc1,
                                    self.tier1_bits,
                                    t1_b_flat,
                                    local_expert,
                                    output_n_tile,
                                    weight_offset,
                                    live_width,
                                    tier1_gate_experts,
                                )
                            )

                        a_row = (sub_tid >> Int32(3)) & Int32(7)
                        a_col_int4 = sub_tid & Int32(7)
                        a_route = packed_route_indices[
                            rb * Int32(8) + a_row
                        ].to(Int32)
                        if (
                            sub_tid < Int32(64)
                            and a_route >= Int32(0)
                            and a_route < live_routes
                        ):
                            out_a_valid = Int32(1)
                        safe_a_route = a_route
                        if out_a_valid == Int32(0):
                            safe_a_route = Int32(0)
                        if cutlass.const_expr(is_fc1):
                            a_off = safe_a_route.to(Int64) * Int64(
                                self.hidden_size
                            ) + Int64(a_col_int4 * 8)
                            out_a_base = get_ptr_as_int64(a_flat, a_off)
                            if output_n_tile >= fc1_half_tiles:
                                out_a_base = get_ptr_as_int64(a_alt_flat, a_off)
                        else:
                            a_off = safe_a_route.to(Int64) * Int64(
                                self.intermediate_size
                            ) + Int64(a_col_int4 * 8)
                            out_a_base = get_ptr_as_int64(a_flat, a_off)

                        epi_row = sub_tid >> Int32(4)
                        epi_col_int4 = sub_tid & Int32(15)
                        epi_route = packed_route_indices[
                            rb * Int32(8) + epi_row
                        ].to(Int32)
                        if epi_route >= Int32(0) and epi_route < live_routes:
                            out_epi_valid = Int32(1)
                        safe_epi_route = epi_route
                        if out_epi_valid == Int32(0):
                            safe_epi_route = Int32(0)
                        c_cols = (
                            self.driver.fc1_cols if is_fc1 else self.hidden_size
                        )
                        c_off = safe_epi_route.to(Int64) * Int64(c_cols) + (
                            output_n_tile.to(Int64) * Int64(16)
                            + epi_col_int4.to(Int64)
                        ) * Int64(8)
                        out_epi_addr = get_ptr_as_int64(c_flat, c_off)

            cand = next_cand

        return (
            found,
            out_task,
            out_tier,
            out_kt,
            out_b_byte,
            out_b_stride,
            out_a_base,
            out_a_valid,
            out_epi_addr,
            out_epi_valid,
            out_scale,
        )

    @cute.jit
    def _stage_sub_k_tile_bits(
        self,
        worker_smem_base: Int32,
        sub_tid: Int32,
        dst_pipe: Int32,
        k_step: Int32,
        b_byte_addr: Int64,
        b_k16_byte_stride: Int64,
        a_base_addr: Int64,
        a_valid: Int32,
        b_l2_policy: Uint64,
        bits: cutlass.Constexpr[int],
    ):
        a_row = sub_tid >> Int32(3)
        a_col = sub_tid & Int32(7)
        a_swz = (a_row << Int32(3)) + (a_col ^ a_row)
        a_dst = (
            worker_smem_base
            + Int32(self.sub_sh_a_off)
            + dst_pipe * Int32(self.SUB_A_STAGE_BYTES)
            + a_swz * Int32(16)
        )
        if sub_tid < Int32(64):
            cp_async4_shared_global_pred(
                a_dst,
                a_base_addr + k_step.to(Int64) * Int64(128),
                a_valid,
            )

        chunks_per_kt = 8 * (2 * int(bits))
        total_chunks = 4 * chunks_per_kt
        iters = (total_chunks + 127) // 128
        b_stage_dst = (
            worker_smem_base
            + Int32(self.sub_sh_b_off)
            + dst_pipe * Int32(self.sub_b_stage_bytes)
        )
        k16_base = k_step.to(Int64) * Int64(4)
        for i in cutlass.range_constexpr(iters):
            chunk = sub_tid + Int32(i * 128)
            if cutlass.const_expr(int(bits) == 4):
                kt_off = chunk >> Int32(6)
                in_kt = chunk & Int32(63)
            else:
                kt_off = chunk // Int32(chunks_per_kt)
                in_kt = chunk - kt_off * Int32(chunks_per_kt)
            b_dst = b_stage_dst + chunk * Int32(16)
            b_src = (
                b_byte_addr
                + (k16_base + kt_off.to(Int64)) * b_k16_byte_stride
                + in_kt.to(Int64) * Int64(16)
            )
            if cutlass.const_expr((i + 1) * 128 <= total_chunks):
                _gw_cp_async4_shared_global_l2hint(b_dst, b_src, b_l2_policy)
            else:
                cp_async4_shared_global_pred_l2hint(
                    b_dst,
                    b_src,
                    (chunk < Int32(total_chunks)).to(Int32),
                    b_l2_policy,
                )

    @cute.jit
    def _load_sub_a_regs(
        self,
        a_regs: cute.Tensor,
        worker_smem_base: Int32,
        a_rd_row: Int32,
        a_rd_col_base: Int32,
        pipe: Int32,
        kk: cutlass.Constexpr[int],
    ):
        col = a_rd_col_base + Int32(2 * int(kk))
        swz = (a_rd_row << Int32(3)) + (col ^ a_rd_row)
        a_addr = (
            worker_smem_base
            + Int32(self.sub_sh_a_off)
            + pipe * Int32(self.SUB_A_STAGE_BYTES)
            + swz * Int32(16)
        )
        a0, a1 = ldmatrix_m8n8x2_b16(a_addr)
        a_regs[0] = a0
        a_regs[1] = a1

    @cute.jit
    def _load_sub_b_regs(
        self,
        b_regs: cute.Tensor,
        worker_smem_base: Int32,
        lane: Int32,
        prev_lane: Int32,
        warp_row: Int32,
        w_n: Int32,
        i0: Int32,
        i1: Int32,
        i2: Int32,
        s2: Int32,
        delta: Int32,
        pipe: Int32,
        kk: cutlass.Constexpr[int],
        bits: cutlass.Constexpr[int],
    ):
        tile_u32 = 8 * int(bits)
        kt_local = (warp_row << Int32(1)) + Int32(int(kk))
        base_u32 = ((kt_local << Int32(3)) + (w_n << Int32(2))) * Int32(
            tile_u32
        )
        b_region = (
            worker_smem_base
            + Int32(self.sub_sh_b_off)
            + pipe * Int32(self.sub_b_stage_bytes)
        )
        if cutlass.const_expr(int(bits) == 4):
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                b = ld_shared_u32(b_region + (tbase + lane) * Int32(4))
                a = cute.arch.shuffle_sync(b, prev_lane)
                b_regs[0, jj] = b
                b_regs[1, jj] = (a << Uint32(16)) | (b >> Uint32(16))
        else:
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                z0 = ld_shared_u32(b_region + (tbase + i0) * Int32(4))
                z1 = ld_shared_u32(b_region + (tbase + i1) * Int32(4))
                z2 = ld_shared_u32(b_region + (tbase + i2) * Int32(4))
                wa, wb = trellis_align_stream_u32x2(z0, z1, z2, s2, delta)
                b_regs[0, jj] = wa
                b_regs[1, jj] = wb

    @cute.jit
    def _dequant_and_mma_sub_step(
        self,
        acc,
        b_cur: cute.Tensor,
        a_cur: cute.Tensor,
        bits: cutlass.Constexpr[int],
    ):
        for jj in cutlass.range_constexpr(4):
            if cutlass.const_expr(int(bits) == 4):
                o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                    b_cur[0, jj], b_cur[1, jj], 4
                )
            else:
                o0, o1, o2, o3 = packed_dequant_trellis_stream_to_half2x4(
                    b_cur[0, jj], b_cur[1, jj], int(bits)
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
                a_cur[0],
                a_cur[1],
            )
            acc[jj * 4 + 0][0] = d0
            acc[jj * 4 + 1][0] = d1
            acc[jj * 4 + 2][0] = d2
            acc[jj * 4 + 3][0] = d3

    @cute.jit
    def _run_sub_gemm_k_loop(
        self,
        acc,
        worker_smem_base: Int32,
        sub_tid: Int32,
        sub_bar_id: Int32,
        base_pipe: Int32,
        cur_kt: Int32,
        cur_b_byte: Int64,
        cur_b_stride: Int64,
        cur_a_base: Int64,
        cur_a_valid: Int32,
        next_valid: Int32,
        next_tier: Int32,
        next_kt: Int32,
        next_b_byte: Int64,
        next_b_stride: Int64,
        next_a_base: Int64,
        next_a_valid: Int32,
        b_l2_policy: Uint64,
        bits: cutlass.Constexpr[int],
        can_have_kt2: cutlass.Constexpr[bool],
    ):
        lane = sub_tid & Int32(31)
        prev_lane = (lane + Int32(31)) & Int32(31)
        warp_id = sub_tid >> Int32(5)
        warp_row = warp_id >> Int32(1)
        w_n = warp_id & Int32(1)
        l16 = lane & Int32(15)
        a_rd_row = l16 & Int32(7)
        a_rd_col_base = (l16 >> Int32(3)) + (warp_row << Int32(2))

        i0 = Int32(0)
        i1 = Int32(0)
        i2 = Int32(0)
        s2 = Int32(0)
        delta = Int32(0)
        if cutlass.const_expr(int(bits) != 4):
            tile_u32 = 8 * int(bits)
            i0, i2, s2, delta = trellis256_lane_geom_bits(
                lane, 0, 8, int(bits)
            )
            i1 = i0 + Int32(1)
            i1 = i1 - Int32(tile_u32) * (i1 >= Int32(tile_u32)).to(Int32)

        b_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_next = cute.make_rmem_tensor((2, 4), Uint32)
        a_cur = cute.make_rmem_tensor((2,), Uint32)
        a_next = cute.make_rmem_tensor((2,), Uint32)

        self._load_sub_b_regs(
            b_cur,
            worker_smem_base,
            lane,
            prev_lane,
            warp_row,
            w_n,
            i0,
            i1,
            i2,
            s2,
            delta,
            base_pipe,
            0,
            bits,
        )
        self._load_sub_a_regs(
            a_cur, worker_smem_base, a_rd_row, a_rd_col_base, base_pipe, 0
        )

        next_base = (base_pipe + cur_kt) & Int32(3)
        start_next = cur_kt - Int32(3)
        if cutlass.const_expr(can_have_kt2):
            if start_next < Int32(0):
                start_next = Int32(0)

        tile_idx = Int32(0)
        while tile_idx < cur_kt:
            cur_pipe = (base_pipe + tile_idx) & Int32(3)
            self._load_sub_b_regs(
                b_next,
                worker_smem_base,
                lane,
                prev_lane,
                warp_row,
                w_n,
                i0,
                i1,
                i2,
                s2,
                delta,
                cur_pipe,
                1,
                bits,
            )
            self._load_sub_a_regs(
                a_next, worker_smem_base, a_rd_row, a_rd_col_base, cur_pipe, 1
            )

            if tile_idx < start_next:
                fetch_k = tile_idx + Int32(3)
                dst_pipe = (base_pipe + fetch_k) & Int32(3)
                self._stage_sub_k_tile_bits(
                    worker_smem_base,
                    sub_tid,
                    dst_pipe,
                    fetch_k,
                    cur_b_byte,
                    cur_b_stride,
                    cur_a_base,
                    cur_a_valid,
                    b_l2_policy,
                    bits,
                )
            else:
                rem = tile_idx - start_next
                if next_valid != Int32(0) and rem < next_kt:
                    dst_pipe = (next_base + rem) & Int32(3)
                    if next_tier == Int32(0):
                        self._stage_sub_k_tile_bits(
                            worker_smem_base,
                            sub_tid,
                            dst_pipe,
                            rem,
                            next_b_byte,
                            next_b_stride,
                            next_a_base,
                            next_a_valid,
                            b_l2_policy,
                            self.tier0_bits,
                        )
                    else:
                        self._stage_sub_k_tile_bits(
                            worker_smem_base,
                            sub_tid,
                            dst_pipe,
                            rem,
                            next_b_byte,
                            next_b_stride,
                            next_a_base,
                            next_a_valid,
                            b_l2_policy,
                            self.tier1_bits,
                        )
            cute.arch.cp_async_commit_group()

            if tile_idx + Int32(1) < cur_kt:
                cute.arch.cp_async_wait_group(2)
                self._worker_sync(sub_bar_id)

            self._dequant_and_mma_sub_step(acc, b_cur, a_cur, bits)
            for jj in cutlass.range_constexpr(4):
                b_cur[0, jj] = b_next[0, jj]
                b_cur[1, jj] = b_next[1, jj]
            a_cur[0] = a_next[0]
            a_cur[1] = a_next[1]

            if tile_idx + Int32(1) < cur_kt:
                next_pipe = (cur_pipe + Int32(1)) & Int32(3)
                self._load_sub_b_regs(
                    b_next,
                    worker_smem_base,
                    lane,
                    prev_lane,
                    warp_row,
                    w_n,
                    i0,
                    i1,
                    i2,
                    s2,
                    delta,
                    next_pipe,
                    0,
                    bits,
                )
                self._load_sub_a_regs(
                    a_next,
                    worker_smem_base,
                    a_rd_row,
                    a_rd_col_base,
                    next_pipe,
                    0,
                )

            self._dequant_and_mma_sub_step(acc, b_cur, a_cur, bits)
            if tile_idx + Int32(1) < cur_kt:
                for jj2 in cutlass.range_constexpr(4):
                    b_cur[0, jj2] = b_next[0, jj2]
                    b_cur[1, jj2] = b_next[1, jj2]
                a_cur[0] = a_next[0]
                a_cur[1] = a_next[1]

            tile_idx += Int32(1)

        if cutlass.const_expr(can_have_kt2):
            if cur_kt == Int32(2):
                if next_valid != Int32(0) and Int32(2) < next_kt:
                    dst_pipe_kt2 = (next_base + Int32(2)) & Int32(3)
                    if next_tier == Int32(0):
                        self._stage_sub_k_tile_bits(
                            worker_smem_base,
                            sub_tid,
                            dst_pipe_kt2,
                            Int32(2),
                            next_b_byte,
                            next_b_stride,
                            next_a_base,
                            next_a_valid,
                            b_l2_policy,
                            self.tier0_bits,
                        )
                    else:
                        self._stage_sub_k_tile_bits(
                            worker_smem_base,
                            sub_tid,
                            dst_pipe_kt2,
                            Int32(2),
                            next_b_byte,
                            next_b_stride,
                            next_a_base,
                            next_a_valid,
                            b_l2_policy,
                            self.tier1_bits,
                        )
                cute.arch.cp_async_commit_group()

    @cute.jit
    def _epilogue_sub_tile(
        self,
        acc,
        worker_smem_base: Int32,
        sub_tid: Int32,
        sub_bar_id: Int32,
        epi_out_addr: Int64,
        epi_valid: Int32,
        scale_f32: cutlass.Float32,
    ):
        fold_base = worker_smem_base + Int32(self.sub_sh_fold_off)
        out_base = worker_smem_base + Int32(self.sub_sh_out_off)
        if cutlass.const_expr(self.disjoint_fold_out):
            if sub_tid >= Int32(64):
                wr_tid = sub_tid - Int32(64)
                for jjw in cutlass.range_constexpr(4):
                    st_shared_v4_f32(
                        fold_base + (Int32(jjw * 64) + wr_tid) * Int32(16),
                        acc[jjw * 4 + 0][0],
                        acc[jjw * 4 + 1][0],
                        acc[jjw * 4 + 2][0],
                        acc[jjw * 4 + 3][0],
                    )
            self._worker_sync(sub_bar_id)
            if sub_tid < Int32(64):
                for jjr in cutlass.range_constexpr(4):
                    r0, r1, r2, r3 = ld_shared_v4_f32(
                        fold_base + (Int32(jjr * 64) + sub_tid) * Int32(16)
                    )
                    acc[jjr * 4 + 0][0] = acc[jjr * 4 + 0][0] + r0
                    acc[jjr * 4 + 1][0] = acc[jjr * 4 + 1][0] + r1
                    acc[jjr * 4 + 2][0] = acc[jjr * 4 + 2][0] + r2
                    acc[jjr * 4 + 3][0] = acc[jjr * 4 + 3][0] + r3
        else:
            for half in cutlass.range_constexpr(2):
                if sub_tid >= Int32(64):
                    wr_tid = sub_tid - Int32(64)
                    for jlw in cutlass.range_constexpr(2):
                        jjhw = 2 * half + jlw
                        st_shared_v4_f32(
                            fold_base + (Int32(jlw * 64) + wr_tid) * Int32(16),
                            acc[jjhw * 4 + 0][0],
                            acc[jjhw * 4 + 1][0],
                            acc[jjhw * 4 + 2][0],
                            acc[jjhw * 4 + 3][0],
                        )
                self._worker_sync(sub_bar_id)
                if sub_tid < Int32(64):
                    for jlr in cutlass.range_constexpr(2):
                        jjhr = 2 * half + jlr
                        r0, r1, r2, r3 = ld_shared_v4_f32(
                            fold_base
                            + (Int32(jlr * 64) + sub_tid) * Int32(16)
                        )
                        acc[jjhr * 4 + 0][0] = acc[jjhr * 4 + 0][0] + r0
                        acc[jjhr * 4 + 1][0] = acc[jjhr * 4 + 1][0] + r1
                        acc[jjhr * 4 + 2][0] = acc[jjhr * 4 + 2][0] + r2
                        acc[jjhr * 4 + 3][0] = acc[jjhr * 4 + 3][0] + r3
                self._worker_sync(sub_bar_id)

        if sub_tid < Int32(64):
            lane = sub_tid & Int32(31)
            w_n = sub_tid >> Int32(5)
            c_sh_wr = (
                Int32(136) * ((lane & Int32(3)) << Int32(1))
                + (lane >> Int32(2))
                + (w_n << Int32(6))
            )
            for jjo in cutlass.range_constexpr(4):
                wr = c_sh_wr + Int32(16 * jjo)
                st_shared_f16_from_f32(
                    out_base + wr * Int32(2),
                    acc[jjo * 4 + 0][0] * scale_f32,
                )
                st_shared_f16_from_f32(
                    out_base + (wr + Int32(136)) * Int32(2),
                    acc[jjo * 4 + 1][0] * scale_f32,
                )
                st_shared_f16_from_f32(
                    out_base + (wr + Int32(8)) * Int32(2),
                    acc[jjo * 4 + 2][0] * scale_f32,
                )
                st_shared_f16_from_f32(
                    out_base + (wr + Int32(144)) * Int32(2),
                    acc[jjo * 4 + 3][0] * scale_f32,
                )

        cute.arch.cp_async_wait_group(2)
        self._worker_sync(sub_bar_id)

        if epi_valid != Int32(0):
            c_sh_rd_bytes = (
                (sub_tid >> Int32(4)) * Int32(17) + (sub_tid & Int32(15))
            ) * Int32(16)
            q0, q1, q2, q3 = ld_shared_v4_u32(out_base + c_sh_rd_bytes)
            st_global_v4_u32(epi_out_addr, q0, q1, q2, q3)

    @cute.jit
    def _run_sub_gemm_phase(
        self,
        is_fc1: cutlass.Constexpr[bool],
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        worker_smem_base: Int32,
        sched_smem: Int32,
        sub_tid: Int32,
        sub_bar_id: Int32,
        mode_groupwise: Int32,
        b_begin: Int32,
        b_end: Int32,
        base_fc1_t: Int32,
        task_start: Int32,
        task_step: Int32,
        task_limit: Int32,
        live_routes: Int32,
        total_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
    ):
        b_l2_policy = create_l2_evict_first_policy()
        can_have_kt2 = bool(
            (not is_fc1)
            and (
                (self.compact_weights and not self.full_compact)
                or self.gemm_fc2_k_tiles == 2
            )
        )
        (
            cur_valid,
            cur_task,
            cur_tier,
            cur_kt,
            cur_b_byte,
            cur_b_stride,
            cur_a_base,
            cur_a_valid,
            cur_epi_addr,
            cur_epi_valid,
            cur_scale,
        ) = self._find_next_valid_tile(
            is_fc1,
            a_flat,
            a_alt_flat,
            t0_b_flat,
            t0_global_scale,
            t1_b_flat,
            t1_global_scale,
            c_flat,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            sched_smem,
            sub_tid,
            mode_groupwise,
            b_begin,
            b_end,
            base_fc1_t,
            task_start,
            task_step,
            task_limit,
            live_routes,
            total_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )

        base_pipe = Int32(0)
        if cur_valid != Int32(0):
            for p in cutlass.range_constexpr(3):
                if Int32(p) < cur_kt:
                    if cur_tier == Int32(0):
                        self._stage_sub_k_tile_bits(
                            worker_smem_base,
                            sub_tid,
                            Int32(p),
                            Int32(p),
                            cur_b_byte,
                            cur_b_stride,
                            cur_a_base,
                            cur_a_valid,
                            b_l2_policy,
                            self.tier0_bits,
                        )
                    else:
                        self._stage_sub_k_tile_bits(
                            worker_smem_base,
                            sub_tid,
                            Int32(p),
                            Int32(p),
                            cur_b_byte,
                            cur_b_stride,
                            cur_a_base,
                            cur_a_valid,
                            b_l2_policy,
                            self.tier1_bits,
                        )
                cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(2)
            self._worker_sync(sub_bar_id)

        acc = [cute.make_rmem_tensor((1,), cutlass.Float32) for _ in range(16)]
        while cur_valid != Int32(0):
            for frag in cutlass.range_constexpr(16):
                acc[frag].fill(0.0)
            (
                next_valid,
                next_task,
                next_tier,
                next_kt,
                next_b_byte,
                next_b_stride,
                next_a_base,
                next_a_valid,
                next_epi_addr,
                next_epi_valid,
                next_scale,
            ) = self._find_next_valid_tile(
                is_fc1,
                a_flat,
                a_alt_flat,
                t0_b_flat,
                t0_global_scale,
                t1_b_flat,
                t1_global_scale,
                c_flat,
                packed_route_indices,
                block_expert_ids,
                descriptor_map,
                sched_smem,
                sub_tid,
                mode_groupwise,
                b_begin,
                b_end,
                base_fc1_t,
                cur_task + task_step,
                task_step,
                task_limit,
                live_routes,
                total_experts,
                tier0_fc2_experts,
                tier1_fc2_experts,
                tier0_gate_experts,
                tier1_gate_experts,
                tier0_up_experts,
                tier1_up_experts,
            )

            if cur_tier == Int32(0):
                self._run_sub_gemm_k_loop(
                    acc,
                    worker_smem_base,
                    sub_tid,
                    sub_bar_id,
                    base_pipe,
                    cur_kt,
                    cur_b_byte,
                    cur_b_stride,
                    cur_a_base,
                    cur_a_valid,
                    next_valid,
                    next_tier,
                    next_kt,
                    next_b_byte,
                    next_b_stride,
                    next_a_base,
                    next_a_valid,
                    b_l2_policy,
                    self.tier0_bits,
                    can_have_kt2,
                )
            else:
                self._run_sub_gemm_k_loop(
                    acc,
                    worker_smem_base,
                    sub_tid,
                    sub_bar_id,
                    base_pipe,
                    cur_kt,
                    cur_b_byte,
                    cur_b_stride,
                    cur_a_base,
                    cur_a_valid,
                    next_valid,
                    next_tier,
                    next_kt,
                    next_b_byte,
                    next_b_stride,
                    next_a_base,
                    next_a_valid,
                    b_l2_policy,
                    self.tier1_bits,
                    can_have_kt2,
                )

            self._epilogue_sub_tile(
                acc,
                worker_smem_base,
                sub_tid,
                sub_bar_id,
                cur_epi_addr,
                cur_epi_valid,
                cur_scale,
            )

            base_pipe = (base_pipe + cur_kt) & Int32(3)
            cur_valid = next_valid
            cur_task = next_task
            cur_tier = next_tier
            cur_kt = next_kt
            cur_b_byte = next_b_byte
            cur_b_stride = next_b_stride
            cur_a_base = next_a_base
            cur_a_valid = next_a_valid
            cur_epi_addr = next_epi_addr
            cur_epi_valid = next_epi_valid
            cur_scale = next_scale

        cute.arch.cp_async_wait_group(0)

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
        sched_smem = smem_base + Int32(self.group_smem_off)
        total_experts = tier0_num_experts + tier1_num_experts
        live_routes = active_m * Int32(self.top_k)
        raw_blocks = packed_route_count[Int32(0)].to(Int32) >> Int32(3)
        if raw_blocks < Int32(0):
            raw_blocks = Int32(0)

        if cutlass.const_expr(self.dual_sub_workers):
            sub_id = tid >> Int32(7)
            sub_tid = tid & Int32(127)
            sub_bar_id = sub_id + Int32(1)
            worker_smem_base = smem_base + sub_id * Int32(
                self.sub_worker_smem_bytes
            )
            total_workers = grid_x << Int32(1)
            wg_task_start = (cta << Int32(1)) + sub_id
        else:
            sub_id = Int32(0)
            sub_tid = tid
            sub_bar_id = Int32(0)
            worker_smem_base = smem_base
            total_workers = grid_x
            wg_task_start = cta

        mode_groupwise = Int32(0)
        if cutlass.const_expr(self.size_m <= 8):
            if raw_blocks <= Int32(64):
                mode_groupwise = Int32(1)
        else:
            if active_m <= Int32(8) and raw_blocks <= Int32(64):
                mode_groupwise = Int32(1)

        d_valid = Int32(0)
        group_id = Int32(0)
        sched_worker = wg_task_start
        workers_per_group = total_workers
        b_begin = Int32(0)
        b_end = Int32(0)
        base_fc1_t = Int32(0)
        fc1_task_start = wg_task_start
        fc1_task_step = total_workers
        fc1_task_limit = Int32(0)
        fc2_task_start = wg_task_start
        fc2_task_step = total_workers
        fc2_task_limit = Int32(0)

        if mode_groupwise != Int32(0):
            d_valid = self._build_group_schedule_smem(
                packed_route_indices,
                block_expert_ids,
                descriptor_map,
                total_experts,
                live_routes,
                raw_blocks,
                total_workers,
                smem_base,
                tid,
            )
            if d_valid > Int32(0):
                num_groups = ld_shared_i32_relaxed(
                    sched_smem + Int32(self.OFF_HDR + 4)
                )
                workers_per_group = ld_shared_i32_relaxed(
                    sched_smem + Int32(self.OFF_HDR + 8)
                )
                if num_groups == Int32(1):
                    group_id = Int32(0)
                    sched_worker = wg_task_start
                else:
                    if cutlass.const_expr(self.dual_sub_workers):
                        c_idx = cta // workers_per_group
                        s_idx = cta - c_idx * workers_per_group
                        if sub_id == Int32(0):
                            group_id = c_idx
                            sched_worker = s_idx
                        else:
                            group_id = num_groups - Int32(1) - c_idx
                            sched_worker = (
                                workers_per_group - Int32(1) - s_idx
                            )
                    else:
                        v_sub = cta & Int32(1)
                        v_sm = cta >> Int32(1)
                        c_idx = v_sm // workers_per_group
                        s_idx = v_sm - c_idx * workers_per_group
                        if v_sub == Int32(0):
                            group_id = c_idx
                            sched_worker = s_idx
                        else:
                            group_id = num_groups - Int32(1) - c_idx
                            sched_worker = (
                                workers_per_group - Int32(1) - s_idx
                            )

                b_begin = ld_shared_i32_relaxed(
                    sched_smem
                    + Int32(self.OFF_GRP_RB_PREFIX)
                    + group_id * Int32(4)
                )
                b_end = ld_shared_i32_relaxed(
                    sched_smem
                    + Int32(self.OFF_GRP_RB_PREFIX)
                    + (group_id + Int32(1)) * Int32(4)
                )
                fc1_task_start = sched_worker
                fc1_task_step = workers_per_group
                if cutlass.const_expr(
                    self.compact_weights and not self.full_compact
                ):
                    base_fc1_t = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_FC1_PREFIX)
                        + b_begin * Int32(4)
                    )
                    end_fc1_t = ld_shared_i32_relaxed(
                        sched_smem
                        + Int32(self.OFF_FC1_PREFIX)
                        + b_end * Int32(4)
                    )
                    fc1_task_limit = end_fc1_t - base_fc1_t
                else:
                    fc1_task_limit = (b_end - b_begin) * Int32(
                        self.gemm_fc1_n_tiles
                    )
                fc2_task_start = sched_worker
                fc2_task_step = workers_per_group
                fc2_task_limit = (b_end - b_begin) * Int32(
                    self.gemm_fc2_n_tiles
                )

                self._run_group_input_rotation(
                    rotation_input,
                    rotation_gate,
                    rotation_up,
                    gate_suh,
                    up_suh,
                    sched_smem,
                    sub_tid,
                    group_id,
                    sched_worker,
                    workers_per_group,
                )
                self._group_barrier(
                    workspace,
                    sub_tid,
                    sub_bar_id,
                    group_id,
                    workers_per_group,
                    0,
                )
        else:
            if cutlass.const_expr(self.size_m > 8):
                self._run_whole_grid_input_rotation(
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
                self.driver._grid_barrier(workspace, tid, grid_x)
                fc1_task_limit = raw_blocks * Int32(self.gemm_fc1_n_tiles)
                fc2_task_limit = raw_blocks * Int32(self.gemm_fc2_n_tiles)

        self._run_sub_gemm_phase(
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_global,
            t1_w13,
            t1_w13_global,
            fc1,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            worker_smem_base,
            sched_smem,
            sub_tid,
            sub_bar_id,
            mode_groupwise,
            b_begin,
            b_end,
            base_fc1_t,
            fc1_task_start,
            fc1_task_step,
            fc1_task_limit,
            live_routes,
            total_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )

        if mode_groupwise != Int32(0):
            if d_valid > Int32(0):
                self._group_barrier(
                    workspace,
                    sub_tid,
                    sub_bar_id,
                    group_id,
                    workers_per_group,
                    1,
                )
                self._run_group_activation(
                    fc1,
                    activated,
                    intermediate_rotations,
                    sched_smem,
                    sub_tid,
                    group_id,
                    sched_worker,
                    workers_per_group,
                )
                self._group_barrier(
                    workspace,
                    sub_tid,
                    sub_bar_id,
                    group_id,
                    workers_per_group,
                    2,
                )
        else:
            if cutlass.const_expr(self.size_m > 8):
                self.driver._grid_barrier(workspace, tid, grid_x)
                self._run_whole_grid_activation(
                    fc1,
                    activated,
                    intermediate_rotations,
                    raw_topk_ids,
                    global_to_combined,
                    descriptor_map,
                    route_num_experts,
                    total_experts,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
                self.driver._grid_barrier(workspace, tid, grid_x)

        self._run_sub_gemm_phase(
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_global,
            t1_w2,
            t1_w2_global,
            fc2,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            worker_smem_base,
            sched_smem,
            sub_tid,
            sub_bar_id,
            mode_groupwise,
            b_begin,
            b_end,
            Int32(0),
            fc2_task_start,
            fc2_task_step,
            fc2_task_limit,
            live_routes,
            total_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
        )


class W4A16MixedTrellis3Kernel(W4A16MixedTrellisKernel):
    """One cooperative grid over three native Trellis bitrates.

    The fixed three-tier ABI keeps K3/K4 execution unchanged and adds K5 only
    for projection-mixed layers whose descriptor map names a third tier.
    """

    # The three-tier ABI has an independent persistent compile namespace.
    ABI_VERSION = 7

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        tier2: W4A16FusedMoeKernel,
        descriptor_bits: int = 9,
    ):
        kernels = (driver, tier0, tier1, tier2)
        for name, moe in zip(
            ("driver", "tier0", "tier1", "tier2"), kernels, strict=True
        ):
            if not moe.full_rotation or not moe.intermediate_rotation:
                raise ValueError(f"mixed Trellis {name} requires full rotation")
            if moe.direct_topk_routes or moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} requires route packing")
            if moe.weight_layout != "trellis_t256":
                raise ValueError(f"mixed Trellis {name} requires native t256 weights")
            if moe.element_dtype != "fp16":
                raise ValueError(f"mixed Trellis {name} requires fp16 GEMM operands")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "rotation_input_dtype",
            "broadcast_suh",
            "trellis_codebook",
            "cta_threads",
            "sms",
        ):
            values = tuple(getattr(moe, attr) for moe in kernels)
            if values[1:] != values[:-1]:
                raise ValueError(f"mixed Trellis kernels disagree on {attr}: {values}")
        for phase in ("fc1", "fc2"):
            gemms = tuple(getattr(moe, phase) for moe in kernels)
            geometry = tuple(
                (
                    gemm.n_tiles,
                    gemm.k_tiles,
                    gemm.tile_n,
                    gemm.tile_k,
                    gemm.cta_threads,
                    gemm.moe_block_size,
                    gemm.schedule_route_block_factor,
                )
                for gemm in gemms
            )
            if geometry[1:] != geometry[:-1]:
                raise ValueError(
                    f"mixed Trellis kernels disagree on {phase} geometry: {geometry}"
                )
        fc2_factor = int(driver.fc2.schedule_route_block_factor)
        expected_factor = int(driver.moe_block_size // driver.fc2.moe_block_size)
        if fc2_factor < 1 or expected_factor % fc2_factor != 0:
            raise ValueError(
                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        tiers = (tier0, tier1, tier2)
        self.descriptor_bits = descriptor_bits
        if descriptor_bits not in (9, 10):
            raise ValueError("descriptor width must be 9 or 10 bits")
        if any(tier.num_experts > (1 << descriptor_bits) for tier in tiers):
            raise ValueError("tier-local expert ids exceed descriptor width")
        if driver.num_experts != sum(tier.num_experts for tier in tiers):
            raise ValueError("driver expert count must equal the sum of all tiers")
        self.driver = driver
        self.tier0 = tier0
        self.tier1 = tier1
        self.tier2 = tier2
        self.size_m = driver.size_m
        self.hidden_size = driver.hidden_size
        self.intermediate_size = driver.intermediate_size
        self.top_k = driver.top_k
        self.cta_threads = driver.cta_threads
        self.blocks_per_sm = min(tier.blocks_per_sm for tier in kernels)
        self.shared_words = max(tier.shared_words for tier in kernels)
        self.sqg_xor_cheb_t12_smem_off = max(
            tier.sqg_xor_cheb_t12_smem_off for tier in kernels
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis3",
            self.ABI_VERSION,
            self.driver.__cache_key__,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.tier2.__cache_key__,
            self.blocks_per_sm,
            self.shared_words,
        ) + (() if self.descriptor_bits == 9 else ("descriptor_bits", self.descriptor_bits))

    @cute.jit
    def _emit_tier_tile3(
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
        t2_b_flat: cute.Tensor,
        t2_scales_flat: cute.Tensor,
        t2_global_scale: cute.Tensor,
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
        tier2_num_experts: Int32,
        tier0_fc2_experts: Int32,
        tier1_fc2_experts: Int32,
        tier2_fc2_experts: Int32,
        tier0_gate_experts: Int32,
        tier1_gate_experts: Int32,
        tier2_gate_experts: Int32,
        tier0_up_experts: Int32,
        tier1_up_experts: Int32,
        tier2_up_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        metadata_block_idx = route_block_idx
        if cutlass.const_expr(not is_fc1):
            metadata_block_idx = route_block_idx // Int32(
                self.driver.moe_block_size
                // (
                    self.driver.fc2.moe_block_size
                    * self.driver.fc2.schedule_route_block_factor
                )
            )
        combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
        total_experts = tier0_num_experts + tier1_num_experts + tier2_num_experts
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
                    t0_in_bounds = local_expert < tier0_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t0_in_bounds = local_expert < tier0_up_experts
                else:
                    t0_in_bounds = local_expert < tier0_fc2_experts
                if tier == Int32(0) and t0_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier0.fc1
                    else:
                        gemm = self.tier0.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t0_b_flat,
                        c_flat,
                        t0_scales_flat,
                        t0_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

                if cutlass.const_expr(is_fc1):
                    t1_in_bounds = local_expert < tier1_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t1_in_bounds = local_expert < tier1_up_experts
                else:
                    t1_in_bounds = local_expert < tier1_fc2_experts
                if tier == Int32(1) and t1_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier1.fc1
                    else:
                        gemm = self.tier1.fc2
                    self._dispatch_tier_gemm(
                        gemm,
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
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

                if cutlass.const_expr(is_fc1):
                    t2_in_bounds = local_expert < tier2_gate_experts
                    if output_n_tile >= fc1_half_tiles:
                        t2_in_bounds = local_expert < tier2_up_experts
                else:
                    t2_in_bounds = local_expert < tier2_fc2_experts
                if tier == Int32(2) and t2_in_bounds:
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier2.fc1
                    else:
                        gemm = self.tier2.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t2_b_flat,
                        c_flat,
                        t2_scales_flat,
                        t2_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
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
        t2_w13_ptr: cute.Pointer,
        t2_w2_ptr: cute.Pointer,
        t2_w13_scales_ptr: cute.Pointer,
        t2_w2_scales_ptr: cute.Pointer,
        t2_w13_global_ptr: cute.Pointer,
        t2_w2_global_ptr: cute.Pointer,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map_ptr: cute.Pointer,
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
        tier2_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier2_fc2_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier2_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        tier2_up_experts: cutlass.Int32,
    ):
        tier0_experts = cutlass.Int64(tier0_num_experts)
        tier1_experts = cutlass.Int64(tier1_num_experts)
        tier2_experts = cutlass.Int64(tier2_num_experts)
        tier0_fc2 = cutlass.Int64(tier0_fc2_experts)
        tier1_fc2 = cutlass.Int64(tier1_fc2_experts)
        tier2_fc2 = cutlass.Int64(tier2_fc2_experts)
        tier0_gate = cutlass.Int64(tier0_gate_experts)
        tier1_gate = cutlass.Int64(tier1_gate_experts)
        tier2_gate = cutlass.Int64(tier2_gate_experts)
        total_experts = tier0_experts + tier1_experts + tier2_experts

        def weight_tensor(ptr, elements):
            return cute.make_tensor(
                ptr,
                layout=cute.make_layout((elements,), stride=(1,)),
            )

        t0_w13 = weight_tensor(
            t0_w13_ptr,
            tier0_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier0.trellis_bits),
        )
        t0_w2 = weight_tensor(
            t0_w2_ptr,
            tier0_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier0.trellis_bits),
        )
        t1_w13 = weight_tensor(
            t1_w13_ptr,
            tier1_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier1.trellis_bits),
        )
        t1_w2 = weight_tensor(
            t1_w2_ptr,
            tier1_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier1.trellis_bits),
        )
        t2_w13 = weight_tensor(
            t2_w13_ptr,
            tier2_gate
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(self.driver.fc1_cols // 16)
            * cutlass.Int64(8 * self.tier2.trellis_bits),
        )
        t2_w2 = weight_tensor(
            t2_w2_ptr,
            tier2_fc2
            * cutlass.Int64(self.intermediate_size // 16)
            * cutlass.Int64(self.hidden_size // 16)
            * cutlass.Int64(8 * self.tier2.trellis_bits),
        )

        t0_w13_scales = weight_tensor(
            t0_w13_scales_ptr,
            tier0_experts
            * cutlass.Int64(self.tier0.fc1.scale_k_groups)
            * cutlass.Int64(self.tier0.fc1.scale_size_n // 4),
        )
        t0_w2_scales = weight_tensor(
            t0_w2_scales_ptr,
            tier0_experts
            * cutlass.Int64(self.tier0.fc2.scale_k_groups)
            * cutlass.Int64(self.tier0.fc2.scale_size_n // 4),
        )
        t1_w13_scales = weight_tensor(
            t1_w13_scales_ptr,
            tier1_experts
            * cutlass.Int64(self.tier1.fc1.scale_k_groups)
            * cutlass.Int64(self.tier1.fc1.scale_size_n // 4),
        )
        t1_w2_scales = weight_tensor(
            t1_w2_scales_ptr,
            tier1_experts
            * cutlass.Int64(self.tier1.fc2.scale_k_groups)
            * cutlass.Int64(self.tier1.fc2.scale_size_n // 4),
        )
        t2_w13_scales = weight_tensor(
            t2_w13_scales_ptr,
            tier2_experts
            * cutlass.Int64(self.tier2.fc1.scale_k_groups)
            * cutlass.Int64(self.tier2.fc1.scale_size_n // 4),
        )
        t2_w2_scales = weight_tensor(
            t2_w2_scales_ptr,
            tier2_experts
            * cutlass.Int64(self.tier2.fc2.scale_k_groups)
            * cutlass.Int64(self.tier2.fc2.scale_size_n // 4),
        )
        t0_w13_global = weight_tensor(t0_w13_global_ptr, tier0_experts)
        t0_w2_global = weight_tensor(t0_w2_global_ptr, tier0_fc2)
        t1_w13_global = weight_tensor(t1_w13_global_ptr, tier1_experts)
        t1_w2_global = weight_tensor(t1_w2_global_ptr, tier1_fc2)
        t2_w13_global = weight_tensor(t2_w13_global_ptr, tier2_experts)
        t2_w2_global = weight_tensor(t2_w2_global_ptr, tier2_fc2)
        descriptor_map = weight_tensor(
            descriptor_map_ptr, cutlass.Int64(3) * total_experts
        )
        intermediate_rotations = weight_tensor(
            intermediate_rotations_ptr,
            total_experts * cutlass.Int64(3 * self.intermediate_size),
        )
        suh_rows = total_experts
        if cutlass.const_expr(self.driver.broadcast_suh):
            suh_rows = cutlass.Int64(1)
        gate_suh = weight_tensor(
            gate_suh_ptr, suh_rows * cutlass.Int64(self.hidden_size)
        )
        up_suh = weight_tensor(up_suh_ptr, suh_rows * cutlass.Int64(self.hidden_size))
        trellis_lut = weight_tensor(
            trellis_lut_ptr, Int64(_SQG_XOR_CHEB_T12_LUT_ENTRIES)
        )
        trellis_lut_addr = get_ptr_as_int64(trellis_lut, Int32(0))
        rotation_input = weight_tensor(
            rotation_input_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),
        )
        topk_weights = weight_tensor(
            topk_weights_ptr,
            active_m.to(cutlass.Int64) * cutlass.Int64(self.top_k),
        )
        self.kernel3(
            rotation_input,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            t1_w13,
            t1_w2,
            t1_w13_scales,
            t1_w2_scales,
            t1_w13_global,
            t1_w2_global,
            t2_w13,
            t2_w2,
            t2_w13_scales,
            t2_w2_scales,
            t2_w13_global,
            t2_w2_global,
            fc1,
            activated,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            trellis_lut_addr,
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier2_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier2_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            tier2_up_experts,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @cute.kernel
    def kernel3(
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
        t2_w13: cute.Tensor,
        t2_w2: cute.Tensor,
        t2_w13_scales: cute.Tensor,
        t2_w2_scales: cute.Tensor,
        t2_w13_global: cute.Tensor,
        t2_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        trellis_lut_addr: Int64,
        tier0_num_experts: cutlass.Int32,
        tier1_num_experts: cutlass.Int32,
        tier2_num_experts: cutlass.Int32,
        tier0_fc2_experts: cutlass.Int32,
        tier1_fc2_experts: cutlass.Int32,
        tier2_fc2_experts: cutlass.Int32,
        tier0_gate_experts: cutlass.Int32,
        tier1_gate_experts: cutlass.Int32,
        tier2_gate_experts: cutlass.Int32,
        tier0_up_experts: cutlass.Int32,
        tier1_up_experts: cutlass.Int32,
        tier2_up_experts: cutlass.Int32,
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
        phase_lut_addr = trellis_lut_addr
        if cutlass.const_expr(self.driver.sqg_xor_cheb_t12_smem):
            self.driver._sqg_smem_copy(
                trellis_lut_addr,
                smem_base + Int32(self.sqg_xor_cheb_t12_smem_off),
                _SQG_XOR_CHEB_T12_SMEM_REGION_BYTES,
                tid,
            )
            cute.arch.sync_threads()
            phase_lut_addr = Int64(smem_base + Int32(self.sqg_xor_cheb_t12_smem_off))
        common = (
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
        )
        counts = (
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
            tier0_fc2_experts,
            tier1_fc2_experts,
            tier2_fc2_experts,
            tier0_gate_experts,
            tier1_gate_experts,
            tier2_gate_experts,
            tier0_up_experts,
            tier1_up_experts,
            tier2_up_experts,
        )
        fc1_emit = partial(
            self._emit_tier_tile3,
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_scales,
            t0_w13_global,
            t1_w13,
            t1_w13_scales,
            t1_w13_global,
            t2_w13,
            t2_w13_scales,
            t2_w13_global,
            fc1,
            *common,
            fc1_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m,
            *counts,
        )
        fc2_emit = partial(
            self._emit_tier_tile3,
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_scales,
            t0_w2_global,
            t1_w2,
            t1_w2_scales,
            t1_w2_global,
            t2_w2,
            t2_w2_scales,
            t2_w2_global,
            fc2,
            *common,
            fc2_scratch,
            workspace,
            phase_lut_addr,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
            *counts,
        )
        total_experts = tier0_num_experts + tier1_num_experts + tier2_num_experts
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
            input_rotated=False,
            fc1_prefetch_tile=None,
            fc2_prefetch_tile=None,
        )


_CACHE: dict[tuple[object, ...], MixedTrellisCompileResult] = {}
_CACHE3: dict[tuple[object, ...], MixedTrellis3CompileResult] = {}
_ROUTE_PACK_WARMED: set[tuple[object, ...]] = set()


def _mixed_route_num_experts(
    expert_map: torch.Tensor,
    expected_route_num_experts: int,
) -> int:
    route_num_experts = int(expert_map.numel())
    if route_num_experts != int(expected_route_num_experts):
        raise ValueError(
            "mixed Trellis route map must match the compiled route namespace: "
            f"map={route_num_experts}, compiled={int(expected_route_num_experts)}"
        )
    return route_num_experts


def warmup_mixed_trellis_route_pack(
    launch: MixedTrellisCompileResult,
    buffers: MixedTrellisBuffers,
    *,
    expert_map: torch.Tensor,
) -> int:
    """Materialize every route-pack specialization reachable by ``launch``.

    Route packing buckets token capacity to powers of two. A profile pass at
    the maximum batch therefore does not cover smaller decode, speculative,
    or final-prefill-chunk buckets. Load those CUDA modules eagerly while the
    serving framework is still profiling persistent memory, so KV sizing sees
    their real driver footprint instead of discovering it under live traffic.
    """
    device = buffers.packed_route_indices.device
    if device.type != "cuda":
        raise RuntimeError("mixed Trellis route-pack warmup requires CUDA buffers")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("mixed Trellis route-pack warmup cannot run during capture")

    route_num_experts = _mixed_route_num_experts(
        expert_map, int(launch.topk_sum.route_num_experts)
    )
    warmed = 0
    pending_keys: list[tuple[object, ...]] = []
    with torch.cuda.device(device):
        device_index = int(torch.cuda.current_device())
        for token_count in route_pack_warmup_token_counts(launch.size_m):
            key = (
                device_index,
                str(launch.route_ids_dtype),
                int(token_count),
                int(launch.top_k),
                route_num_experts,
                int(launch.moe_block_size),
                True,
            )
            if key in _ROUTE_PACK_WARMED:
                continue
            dummy_topk_ids = torch.zeros(
                (token_count, launch.top_k),
                dtype=launch.route_ids_dtype,
                device=device,
            )
            pack_topk_routes_by_expert(
                dummy_topk_ids,
                launch.moe_block_size,
                route_num_experts,
                expert_map=expert_map,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
            )
            pending_keys.append(key)
            warmed += 1
        torch.cuda.current_stream(device).synchronize()
        _ROUTE_PACK_WARMED.update(pending_keys)
    return warmed


def _select_mixed_fc2_kernel(
    build_kernel,
    *,
    moe_block_size: int,
    max_shared_mem: int,
):
    """Prefer native large-M FC2 and fall back only when resources require it."""

    moe_block_size = int(moe_block_size)
    if moe_block_size not in (32, 64):
        return build_kernel(False)
    try:
        native = build_kernel(False)
    except ValueError as exc:
        resource_errors = (
            "missing W4A16 register count",
            "shared-memory footprint exceeds device opt-in limit",
        )
        if not any(marker in str(exc) for marker in resource_errors):
            raise
    else:
        if native.shared_words * 4 <= int(max_shared_mem):
            return native
    return build_kernel(True)


def _apply_mixed_residency(kernel, requested: int | None, max_shared_mem: int) -> None:
    """Apply an offline plan override without changing automatic launch policy.

    The conservative register bound assumes the full 256-register allocation
    quantum per thread. Shared storage includes one driver KiB per block and
    uses the supplied per-block opt-in limit as a conservative SM budget.
    Residency already participates in the kernel's immutable compile key.
    """
    if requested is None:
        return
    if type(requested) is not int or requested not in (1, 2):
        raise ValueError("mixed Trellis residency override must be one or two blocks/SM")
    if requested * kernel.cta_threads > 256:
        raise ValueError("mixed Trellis residency exceeds the conservative register budget")
    if requested * (kernel.shared_words * 4 + 1024) > int(max_shared_mem):
        raise ValueError("mixed Trellis residency exceeds the conservative shared-memory budget")
    kernel.blocks_per_sm = requested


@dataclass(frozen=True)
class MixedTrellisDecodeSchedule:
    """When, and with which L2 policy, the cooperative kernel fetches weight
    words. Every option is a memory schedule of the same loads, products and
    sums, so a schedule never changes a bit of the output:

    - ``l2=2``: weight words are staged with an L2 evict-first policy, so the
      one-pass weight stream gives way to the activation rows, the phase
      buffers and the prefetched tile starts that are read again;
    - ``pf1=N`` / ``pf2=N``: before each FC1 / FC2 whole tile runs, the CTA
      prefetches the first N weight K tiles (and route indices) of its next
      tile into L2, so that tile's pipeline does not start with a DRAM round
      trip;
    - ``pdl=2``: each GEMM's first tile is prefetched before the grid barrier
      that precedes it (FC1's during the input rotation, FC2's while a CTA
      waits for the slowest FC1 tile and runs the activation), the memory
      analogue of starting the down kernel while gate/up ends.
    """

    l2_evict_first_b: bool = False
    fc1_prefetch_k_tiles: int = 0
    fc2_prefetch_k_tiles: int = 0
    phase_prefetch: bool = False

    def canonical(self) -> str:
        return (
            f"l2={2 if self.l2_evict_first_b else 1},pf1={self.fc1_prefetch_k_tiles},"
            f"pf2={self.fc2_prefetch_k_tiles},pdl={2 if self.phase_prefetch else 1}"
        )

    def kernel_options(self) -> dict[str, object]:
        return {
            "l2_evict_first_b": self.l2_evict_first_b,
            "fc1_l2_prefetch_k_tiles": self.fc1_prefetch_k_tiles,
            "fc2_l2_prefetch_k_tiles": self.fc2_prefetch_k_tiles,
            "phase_l2_prefetch": self.phase_prefetch,
        }


# Named decode schedules. "gb10" is the DGX Spark (SM121, LPDDR5X) schedule
# for the GLM 5.3 Flash Spark decode packages (FR-G.7(b)): evict-first weight
# words only. Measured on GB10 (one TP4 rank slice, uniform top-8 routes,
# benchmarks/benchmark_glmf_decode_schedule.py), evict-first alone shortened a
# call by 1.6-7.3% at 1-80 rows; the next-tile and phase prefetches lost 4% on
# their own and added nothing to evict-first from 4 rows up, so they stay
# options for other devices and are off here.
DECODE_SCHEDULE_PRESETS = {
    "gb10": "l2=2,pf1=0,pf2=0,pdl=1",
}


def parse_decode_schedule(spec: str | None) -> MixedTrellisDecodeSchedule | None:
    """Parse a preset name or ``key=value`` list (l2, pf1, pf2, pdl); None,
    "" and "default" are the default schedule (no option)."""
    if spec is None or spec in ("", "default"):
        return None
    text = DECODE_SCHEDULE_PRESETS.get(str(spec), str(spec))
    values = {"l2": 1, "pf1": 0, "pf2": 0, "pdl": 1}
    seen = set()
    for item in text.split(","):
        key, sep, value = item.strip().partition("=")
        if not sep or key not in values or key in seen:
            raise ValueError(f"unknown or repeated decode schedule option {item!r} in {spec!r}")
        seen.add(key)
        try:
            values[key] = int(value)
        except ValueError:
            raise ValueError(f"decode schedule option {key} needs an integer: {item!r}") from None
    if values["l2"] not in (1, 2) or values["pdl"] not in (1, 2):
        raise ValueError(f"decode schedule l2 and pdl are 1 or 2: {spec!r}")
    if not 0 <= values["pf1"] <= 64 or not 0 <= values["pf2"] <= 64:
        raise ValueError(f"decode schedule prefetch depths are 0..64 K tiles: {spec!r}")
    if values["pdl"] == 2 and not (values["pf1"] and values["pf2"]):
        raise ValueError(f"decode schedule pdl=2 needs pf1 and pf2: {spec!r}")
    schedule = MixedTrellisDecodeSchedule(
        l2_evict_first_b=values["l2"] == 2,
        fc1_prefetch_k_tiles=values["pf1"],
        fc2_prefetch_k_tiles=values["pf2"],
        phase_prefetch=values["pdl"] == 2,
    )
    return None if schedule == MixedTrellisDecodeSchedule() else schedule


def effective_decode_schedule(driver: W4A16FusedMoeKernel) -> str | None:
    """The canonical decode schedule a kernel built from ``driver`` runs: each
    GEMM bounds a prefetch depth by its own K tiles (``W4A16GemmKernel``), so
    requests that differ only past that bound build one program, share one
    cache entry and report one schedule (None: the default schedule)."""
    schedule = MixedTrellisDecodeSchedule(
        l2_evict_first_b=bool(driver.fc1.l2_evict_first_b),
        fc1_prefetch_k_tiles=int(driver.fc1.l2_prefetch_k_tiles),
        fc2_prefetch_k_tiles=int(driver.fc2.l2_prefetch_k_tiles),
        phase_prefetch=bool(driver.phase_l2_prefetch),
    )
    return None if schedule == MixedTrellisDecodeSchedule() else schedule.canonical()


def compile_mixed_trellis(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    top_k: int,
    max_m_blocks: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    tier0_bits: int = 3,
    tier1_bits: int = 4,
    trellis_codebook: str = "mcg",
    swiglu_limit: float | None = None,
    activation: str = "silu",
    moe_block_size: int = 8,
    rotation_input_dtype: str = "bf16",
    full_rotation_output_dtype: str = "fp32",
    route_ids_dtype: torch.dtype = torch.int32,
    direct_topk_routes: bool = False,
    broadcast_suh: bool = False,
    broadcast_svh: bool = False,
    route_num_experts: int | None = None,
    force_blocks_per_sm: int | None = None,
    paired_boundary: str | None = None,
    compact_weights: bool = False,
    token_major_rotation: bool = False,
    fused_input_rotation: bool = False,
    warp_specialized: bool = False,
    input_format: str = "bf16",
    ws_input_stages: int | None = None,
    ws_dynamic_tiles: bool = False,
    activations: str = "a16",
    decode_schedule: str | None = None,
    workspace_sms: int | None = None,
    ws_prerotate: bool | None = None,
    row_addresses: bool = False,
    _kimi_tp6_override: bool | None = None,
    _k4_groupwise_override: bool | None = None,
) -> MixedTrellisCompileResult:
    if row_addresses and not 0 < int(size_m) <= 8192:
        raise ValueError("row-address WS capacity must be in 1..8192")
    if row_addresses and (not warp_specialized or activations != "a16"):
        raise ValueError("row addresses require WS A16 (BF16 or E4M3 K32 input)")
    if activations not in ("a16", "a8"):
        raise ValueError(f"unsupported EXL3 activations {activations!r}")
    if activations == "a8" and not warp_specialized:
        raise ValueError("EXL3 A8 requires the warp-specialized prefill kernel")
    schedule = parse_decode_schedule(decode_schedule)
    if schedule is not None and (
        warp_specialized or paired_boundary is not None
        or ((compact_weights or direct_topk_routes) and (
            schedule.fc1_prefetch_k_tiles or schedule.fc2_prefetch_k_tiles or schedule.phase_prefetch))
    ):
        raise ValueError(
            "the L2 decode schedule requires cooperative disjoint weights; "
            "compact and direct routes support evict-first only"
        )
    if input_format not in ("bf16", "e4m3_k32"):
        raise ValueError(f"unsupported mixed Trellis input format {input_format!r}")
    if input_format == "e4m3_k32" and (
        not warp_specialized or rotation_input_dtype != "bf16" or int(hidden_size) % 512
    ):
        raise ValueError(
            "E4M3 K32 wire input requires the warp-specialized kernel and "
            "16-byte wire rows (hidden % 512 == 0)"
        )
    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("mixed Trellis route IDs must be int32 or int64")
    if int(size_m) * int(top_k) > torch.iinfo(torch.int32).max:
        raise ValueError("mixed Trellis routed-row count must fit in int32")
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
        int(value) for value in force_tile_config
    )
    trellis_codebook, bits = _normalize_mixed_trellis_format(
        trellis_codebook,
        (tier0_bits, tier1_bits),
    )
    tier0_bits, tier1_bits = bits
    if activations == "a8" and trellis_codebook != "mcg":
        raise ValueError("EXL3 A8 INT8 weight conversion requires the MCG codebook")
    # 192-wide and unequal FC1/FC2 N tiles exist only in the warp-specialized
    # kernel; its cooperative sub-kernels (decode helpers, activation) keep 128.
    ws_tile_n = None
    if fc1_tile_n != fc2_tile_n or 192 in (fc1_tile_n, fc2_tile_n):
        if not warp_specialized:
            raise ValueError("192-wide or unequal N tiles require the warp-specialized kernel")
        ws_tile_n = (fc1_tile_n, fc2_tile_n)
        fc1_tile_n = fc2_tile_n = 128
    direct_topk_routes = bool(direct_topk_routes)
    # Mixed Trellis uses the whole-tile scheduler below: one CTA owns the
    # complete K reduction for an MN tile.  The retired split-K scheduler lost
    # cross-tier partials with K64, but keeping its K128 guard here also blocks
    # the production K64/N256 geometry after that hazard has been removed.
    counts = (int(tier0_num_experts), int(tier1_num_experts))
    if any(value <= 0 or value > _MAX_TIER_EXPERTS for value in counts):
        raise ValueError(
            "two-tier mixed Trellis requires each tier to contain 1..1024 slots"
        )
    total_experts = sum(counts)
    if route_num_experts is None:
        route_num_experts = total_experts
    route_num_experts = int(route_num_experts)
    descriptor_bits = _descriptor_bits(route_num_experts)
    def make_kernel(
        num_experts: int,
        bits: int,
        *,
        grouped_m8_fc2: bool,
    ) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            swiglu_limit=swiglu_limit,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            fc2_moe_block_size=(8 if grouped_m8_fc2 else moe_block_size),
            fc2_schedule_route_block_factor=(2 if grouped_m8_fc2 else 1),
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            trellis_codebook=trellis_codebook,
            intermediate_rotation=True,
            full_rotation=True,
            workspace_sms=workspace_sms,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=broadcast_suh,
            direct_topk_routes=direct_topk_routes,
            schedule_whole_tiles=True,
            fused_input_rotation=fused_input_rotation and not warp_specialized,
            activation_output_a8=activations == "a8",
            **({} if schedule is None else schedule.kernel_options()),
        )

    def build_kernel(grouped_m8_fc2: bool) -> W4A16MixedTrellisKernel:
        common = {"grouped_m8_fc2": grouped_m8_fc2}
        if warp_specialized:
            from .mixed_trellis_ws import W4A16MixedTrellisWSKernel

            if paired_boundary is not None or token_major_rotation:
                raise ValueError(
                    "warp-specialized mixed Trellis has its own input rotation "
                    "and no paired boundary"
                )
            return W4A16MixedTrellisWSKernel(
                driver=make_kernel(total_experts, tier0_bits, **common),
                tier0=make_kernel(int(tier0_num_experts), int(tier0_bits), **common),
                tier1=make_kernel(int(tier1_num_experts), int(tier1_bits), **common),
                descriptor_bits=descriptor_bits,
                max_shared_mem=int(max_shared_mem),
                input_format=input_format,
                ws_tile_n=ws_tile_n,
                input_stages=ws_input_stages,
                dynamic_tiles=ws_dynamic_tiles,
                activations=activations,
                compact_weights=compact_weights,
                prerotate=ws_prerotate,
                row_addresses=row_addresses,
            )
        use_kimi_tp6 = (
            _kimi_tp6_override
            if _kimi_tp6_override is not None
            else (
                (hidden_size, intermediate_size, top_k, route_num_experts)
                == (3584, 512, 16, 896)
                and str(activation).strip().lower() == "situ"
                and swiglu_limit is None
                and (tier0_bits, tier1_bits) == (2, 3)
                and trellis_codebook == "mcg"
                and rotation_input_dtype == "bf16"
                and full_rotation_output_dtype == "bf16"
                and int(size_m) in (1, 80)
                and int(moe_block_size) == 8
                and not grouped_m8_fc2
                and force_blocks_per_sm in (None, 1)
                and not direct_topk_routes
                and not broadcast_suh
                and not compact_weights
                and paired_boundary is None
                and not token_major_rotation
                and not fused_input_rotation
                and activations == "a16"
                and input_format == "bf16"
                and not row_addresses
                and (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n)
                == (64, 256, 64, 256)
                and schedule == MixedTrellisDecodeSchedule(l2_evict_first_b=True)
                and not _w4a16_small_m_splitk_enabled()
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability(torch.cuda.current_device())
                == (12, 1)
            )
        )
        if use_kimi_tp6:
            from .mixed_trellis_kimi import W4A16MixedTrellisKimiTP6

            return W4A16MixedTrellisKimiTP6(
                driver=make_kernel(total_experts, tier0_bits, **common),
                tier0=make_kernel(int(tier0_num_experts), int(tier0_bits), **common),
                tier1=make_kernel(int(tier1_num_experts), int(tier1_bits), **common),
                descriptor_bits=descriptor_bits,
            )
        use_k4_groupwise = (
            _k4_groupwise_override
            if _k4_groupwise_override is not None
            else (
                (int(tier0_bits), int(tier1_bits)) in {(4, 5), (4, 6)}
                and (int(hidden_size), int(route_num_experts), int(top_k))
                in _K4_GROUPWISE_FAMILIES
                and 1 <= int(size_m) <= 80
                and int(moe_block_size) == 8
                and not grouped_m8_fc2
                and force_blocks_per_sm in (None, 1, 2)
                and (force_blocks_per_sm != 2 or fc1_tile_n == 128)
                and trellis_codebook == "mcg"
                and rotation_input_dtype == "bf16"
                and not direct_topk_routes
                and not broadcast_suh
                and paired_boundary is None
                and not token_major_rotation
                and not fused_input_rotation
                and activations == "a16"
                and input_format == "bf16"
                and not row_addresses
                and (fc1_tile_k, fc2_tile_k) == (64, 64)
                and fc1_tile_n == fc2_tile_n
                and fc1_tile_n in (128, 256)
                and int(hidden_size) % 256 == 0
                and int(intermediate_size) % 128 == 0
                and (not compact_weights or int(intermediate_size) % int(fc1_tile_n) == 0)
                and schedule in (None, MixedTrellisDecodeSchedule(l2_evict_first_b=True))
                and not _w4a16_small_m_splitk_enabled()
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability(torch.cuda.current_device())
                == (12, 1)
            )
        )
        if use_k4_groupwise:
            return W4A16MixedTrellisGroupwiseKernel(
                driver=make_kernel(total_experts, tier0_bits, **common),
                tier0=make_kernel(int(tier0_num_experts), int(tier0_bits), **common),
                tier1=make_kernel(int(tier1_num_experts), int(tier1_bits), **common),
                descriptor_bits=descriptor_bits,
                compact_weights=compact_weights,
                force_blocks_per_sm=force_blocks_per_sm,
            )
        return W4A16MixedTrellisKernel(
            driver=make_kernel(total_experts, tier0_bits, **common),
            tier0=make_kernel(int(tier0_num_experts), int(tier0_bits), **common),
            tier1=make_kernel(int(tier1_num_experts), int(tier1_bits), **common),
            descriptor_bits=descriptor_bits,
            paired_boundary=paired_boundary,
            compact_weights=compact_weights,
            token_major_rotation=token_major_rotation,
        )

    if warp_specialized:
        if force_blocks_per_sm not in (None, 1):
            raise ValueError("warp-specialized mixed Trellis runs one CTA per SM")
        kernel = build_kernel(False)
    else:
        kernel = _select_mixed_fc2_kernel(
            build_kernel,
            moe_block_size=moe_block_size,
            max_shared_mem=max_shared_mem,
        )
        _apply_mixed_residency(kernel, force_blocks_per_sm, max_shared_mem)
    # shared_words is the complete dynamically allocated MemRange used by the
    # cooperative kernel. CUDA permits a launch exactly at the device's
    # opt-in shared-memory limit; rejecting an additional 512 bytes here
    # unnecessarily excludes the stock mixed-K tile geometry at block-64.
    if kernel.shared_words * 4 > int(max_shared_mem):
        raise ValueError(
            "mixed Trellis shared-memory requirement exceeds the device limit: "
            f"required={kernel.shared_words * 4} "
            f"limit={int(max_shared_mem)}"
        )
    device = int(torch.cuda.current_device())
    cache_key = (
        "mixed_trellis",
        device,
        kernel.__cache_key__,
        str(route_ids_dtype),
        int(size_m),
        int(max_m_blocks),
        direct_topk_routes,
    )
    topk_sum = compile_w4a16_topk_sum(
        m=size_m,
        topk=top_k,
        hidden_size=hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        full_rotation_output_dtype=full_rotation_output_dtype,
        num_experts=total_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=True,
        broadcast_svh=broadcast_svh,
        row_addresses=row_addresses,
    )
    # The schedule this program runs, as built: a prefetch depth past a GEMM's
    # K tiles builds the same program as that bound, and reports the bound.
    decode_schedule = None if schedule is None else effective_decode_schedule(kernel.driver)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        # Compilation is independent of an artifact's tier partition. Replace
        # plan metadata and the top-k launch on every cache lookup so one
        # artifact cannot leak its partition into another artifact.
        return replace(
            cached,
            topk_sum=topk_sum,
            tier0_num_experts=int(tier0_num_experts),
            tier1_num_experts=int(tier1_num_experts),
            sms=int(sms),
            broadcast_suh=bool(broadcast_suh),
            broadcast_svh=bool(broadcast_svh),
            decode_schedule=decode_schedule,
            workspace_sms=workspace_sms,
        )

    compile_m = _fake_m_for_specialization(size_m)
    compile_rows = compile_m * top_k
    fc1_cols = 2 * intermediate_size
    cutlass_dtype = cutlass.Float16
    rotation_dtype = (
        cutlass.Uint8
        if input_format == "e4m3_k32"
        else _cutlass_element_dtype(rotation_input_dtype)
    )

    if row_addresses:
        rotation_dtype = cutlass.Int64

    def tensor(dtype, elements: int, *, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    def tier_args():
        return (
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        )

    scratch_elements = max(
        fc1_cols * compile_rows,
        hidden_size * compile_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        make_ptr(rotation_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        *tier_args(),
        *tier_args(),
        tensor(cutlass_dtype, compile_rows * fc1_cols),
        tensor(cutlass_dtype, compile_rows * intermediate_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass.Int32, compile_rows if direct_topk_routes else moe_block_size),
        tensor(
            cutlass.Int64 if route_ids_dtype == torch.int64 else cutlass.Int32,
            compile_rows,
        ),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Int32, 1, align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + 2),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        # FC2 counts are independent artifact data; trace with the FC1 values.
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        1,
        1,
        current_cuda_stream(),
        # Gate-count trace placeholders (keyword-only, last in the signature);
        # real counts ride each launch.
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        # Up-count trace placeholders; real counts ride each launch too.
        Int32(tier0_num_experts),
        Int32(tier1_num_experts),
        Int32(route_num_experts),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mixed_trellis_ws" if warp_specialized else "moe.w4a16.mixed_trellis",
            (
                getattr(kernel, "WS_ABI_VERSION", 0) * 1000 + W4A16MixedTrellisKernel.ABI_VERSION
            ),
            cache_key,
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = MixedTrellisCompileResult(
        compiled=compiled,
        topk_sum=topk_sum,
        trellis_lut=_trellis256_execution_lut(
            torch.device("cuda", device), trellis_codebook
        ),
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        tier0_num_experts=int(tier0_num_experts),
        tier1_num_experts=int(tier1_num_experts),
        tier0_bits=int(tier0_bits),
        tier1_bits=int(tier1_bits),
        trellis_codebook=trellis_codebook,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=ws_tile_n[0] if ws_tile_n else fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=ws_tile_n[1] if ws_tile_n else fc2_tile_n,
        moe_block_size=int(moe_block_size),
        fc2_moe_block_size=int(kernel.driver.fc2.moe_block_size),
        fc2_schedule_route_block_factor=int(
            kernel.driver.fc2.schedule_route_block_factor
        ),
        max_m_blocks=int(max_m_blocks),
        blocks_per_sm=int(kernel.blocks_per_sm),
        sms=int(sms),
        shared_memory_bytes=int(kernel.shared_words * 4),
        rotation_input_dtype=str(rotation_input_dtype),
        route_ids_dtype=route_ids_dtype,
        direct_topk_routes=direct_topk_routes,
        broadcast_suh=bool(broadcast_suh),
        broadcast_svh=bool(broadcast_svh),
        paired_boundary=paired_boundary,
        compact_weights=bool(compact_weights),
        fused_input_rotation=bool(fused_input_rotation) and not warp_specialized,
        warp_specialized=bool(warp_specialized),
        ws_prerotate=bool(getattr(kernel, "ws_prerotate", False)),
        input_format=str(input_format),
        row_addresses=bool(row_addresses),
        ws_input_stages=ws_input_stages,
        activations=activations,
        decode_schedule=decode_schedule,
        workspace_sms=workspace_sms,
    )
    _CACHE[cache_key] = result
    return result


def compile_mixed_trellis3(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    tier2_num_experts: int,
    top_k: int,
    max_m_blocks: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    tier0_bits: int = 3,
    tier1_bits: int = 4,
    tier2_bits: int = 5,
    trellis_codebook: str = "mcg",
    swiglu_limit: float | None = None,
    activation: str = "silu",
    moe_block_size: int = 8,
    rotation_input_dtype: str = "bf16",
    full_rotation_output_dtype: str = "fp32",
    route_ids_dtype: torch.dtype = torch.int32,
    broadcast_suh: bool = False,
    broadcast_svh: bool = False,
    route_num_experts: int | None = None,
    workspace_sms: int | None = None,
    decode_schedule: str | None = None,
) -> MixedTrellis3CompileResult:
    """Compile the dedicated three-bitrate cooperative Trellis grid."""

    schedule = parse_decode_schedule(decode_schedule)
    if schedule is not None and (schedule.fc1_prefetch_k_tiles or schedule.fc2_prefetch_k_tiles or schedule.phase_prefetch):
        raise ValueError("multi-tier decode schedules support evict-first only")

    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("mixed Trellis route IDs must be int32 or int64")
    if int(size_m) * int(top_k) > torch.iinfo(torch.int32).max:
        raise ValueError("mixed Trellis routed-row count must fit in int32")
    trellis_codebook, bits = _normalize_mixed_trellis_format(
        trellis_codebook,
        (tier0_bits, tier1_bits, tier2_bits),
    )
    counts = tuple(
        int(value)
        for value in (
            tier0_num_experts,
            tier1_num_experts,
            tier2_num_experts,
        )
    )
    if any(value <= 0 or value > _MAX_TIER_EXPERTS for value in counts):
        raise ValueError(
            "three-tier mixed Trellis requires each tier to contain 1..1024 slots"
        )
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
        int(value) for value in force_tile_config
    )
    # The whole-tile scheduler gives one CTA the complete K reduction, so K64
    # does not use the retired cross-CTA partial-reduction path.
    total_experts = sum(counts)
    if route_num_experts is None:
        route_num_experts = total_experts
    route_num_experts = int(route_num_experts)
    descriptor_bits = _descriptor_bits(route_num_experts)
    def make_kernel(
        num_experts: int,
        trellis_bits: int,
        *,
        grouped_m8_fc2: bool,
    ) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            swiglu_limit=swiglu_limit,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            fc2_moe_block_size=(8 if grouped_m8_fc2 else moe_block_size),
            fc2_schedule_route_block_factor=(2 if grouped_m8_fc2 else 1),
            element_dtype="fp16",
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            intermediate_rotation=True,
            full_rotation=True,
            workspace_sms=workspace_sms,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=broadcast_suh,
            schedule_whole_tiles=True,
            **({} if schedule is None else schedule.kernel_options()),
        )

    def build_kernel(grouped_m8_fc2: bool) -> W4A16MixedTrellis3Kernel:
        common = {"grouped_m8_fc2": grouped_m8_fc2}
        return W4A16MixedTrellis3Kernel(
            driver=make_kernel(total_experts, bits[0], **common),
            tier0=make_kernel(counts[0], bits[0], **common),
            tier1=make_kernel(counts[1], bits[1], **common),
            tier2=make_kernel(counts[2], bits[2], **common),
            descriptor_bits=descriptor_bits,
        )

    kernel = _select_mixed_fc2_kernel(
        build_kernel,
        moe_block_size=moe_block_size,
        max_shared_mem=max_shared_mem,
    )
    if kernel.shared_words * 4 > int(max_shared_mem):
        raise ValueError(
            "mixed Trellis shared-memory requirement exceeds the device limit: "
            f"required={kernel.shared_words * 4} limit={int(max_shared_mem)}"
        )
    device = int(torch.cuda.current_device())
    cache_key = (
        "mixed_trellis3",
        device,
        kernel.__cache_key__,
        str(route_ids_dtype),
        int(size_m),
        int(max_m_blocks),
    )
    topk_sum = compile_w4a16_topk_sum(
        m=size_m,
        topk=top_k,
        hidden_size=hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        full_rotation_output_dtype=full_rotation_output_dtype,
        num_experts=total_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=True,
        broadcast_svh=broadcast_svh,
    )
    decode_schedule = None if schedule is None else schedule.canonical()
    cached = _CACHE3.get(cache_key)
    if cached is not None:
        return replace(
            cached,
            topk_sum=topk_sum,
            tier0_num_experts=counts[0],
            tier1_num_experts=counts[1],
            tier2_num_experts=counts[2],
            sms=int(sms),
            broadcast_suh=bool(broadcast_suh),
            broadcast_svh=bool(broadcast_svh),
            decode_schedule=decode_schedule,
        )

    compile_m = _fake_m_for_specialization(size_m)
    compile_rows = compile_m * top_k
    fc1_cols = 2 * intermediate_size
    cutlass_dtype = cutlass.Float16
    rotation_dtype = _cutlass_element_dtype(rotation_input_dtype)

    def tensor(dtype, elements: int, *, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    def tier_args():
        return (
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        )

    scratch_elements = max(
        fc1_cols * compile_rows,
        hidden_size * compile_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        make_ptr(rotation_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        *tier_args(),
        *tier_args(),
        *tier_args(),
        tensor(cutlass_dtype, compile_rows * fc1_cols),
        tensor(cutlass_dtype, compile_rows * intermediate_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass.Int32, moe_block_size),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Int32, 1, align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + 2),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        1,
        1,
        current_cuda_stream(),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
        Int32(counts[0]),
        Int32(counts[1]),
        Int32(counts[2]),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mixed_trellis3",
            W4A16MixedTrellis3Kernel.ABI_VERSION,
            cache_key,
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = MixedTrellis3CompileResult(
        compiled=compiled,
        topk_sum=topk_sum,
        trellis_lut=_trellis256_execution_lut(
            torch.device("cuda", device), trellis_codebook
        ),
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        tier0_num_experts=counts[0],
        tier1_num_experts=counts[1],
        tier2_num_experts=counts[2],
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        tier2_bits=bits[2],
        trellis_codebook=trellis_codebook,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        moe_block_size=int(moe_block_size),
        fc2_moe_block_size=int(kernel.driver.fc2.moe_block_size),
        fc2_schedule_route_block_factor=int(
            kernel.driver.fc2.schedule_route_block_factor
        ),
        max_m_blocks=int(max_m_blocks),
        blocks_per_sm=int(kernel.blocks_per_sm),
        sms=int(sms),
        shared_memory_bytes=int(kernel.shared_words * 4),
        rotation_input_dtype=str(rotation_input_dtype),
        route_ids_dtype=route_ids_dtype,
        direct_topk_routes=False,
        broadcast_suh=bool(broadcast_suh),
        broadcast_svh=bool(broadcast_svh),
        workspace_sms=workspace_sms,
        decode_schedule=decode_schedule,
    )
    _CACHE3[cache_key] = result
    return result


def _make_mixed_trellis_buffers(
    launch: MixedTrellisCompileResult | MixedTrellis3CompileResult,
    *,
    device: torch.device,
    sms: int,
    route_num_experts: int,
) -> MixedTrellisBuffers:
    if launch.workspace_sms is not None:
        if int(sms) > int(launch.workspace_sms):
            raise ValueError("mixed Trellis workspace SM capacity is smaller than the device")
        sms = int(launch.workspace_sms)
    capacity_rows = launch.size_m * launch.top_k
    route_slots = (
        capacity_rows
        if launch.direct_topk_routes
        else max_packed_route_slots(
            capacity_rows, launch.moe_block_size, route_num_experts
        )
    )
    route_blocks = (
        capacity_rows
        if launch.direct_topk_routes
        else (route_slots + launch.moe_block_size - 1) // launch.moe_block_size
    )
    if route_blocks > launch.max_m_blocks:
        raise ValueError(
            "mixed Trellis route buffers require "
            f"{route_blocks} blocks, but the launch was compiled for "
            f"{launch.max_m_blocks}"
        )
    fc1_cols = 2 * launch.intermediate_size
    # FC1 consumes rotation_gate before the grid-wide activation barrier. FC2
    # starts only after that barrier, so its output can safely reuse the same
    # storage without changing either phase's tensor layout.
    rotation_gate = torch.empty(
        (capacity_rows, launch.hidden_size), dtype=torch.float16, device=device
    )
    # A fused FC1 input rotation (and the warp-specialized kernels, which
    # rotate FC1 inputs in shared memory) never write or read the per-route
    # up copy; the gate buffer then only backs the FC2 output.
    rotation_up = rotation_gate
    if not (
        getattr(launch, "fused_input_rotation", False)
        or (getattr(launch, "warp_specialized", False)
            and not getattr(launch, "ws_prerotate", False))
    ):
        rotation_up = torch.empty(
            (capacity_rows, launch.hidden_size), dtype=torch.float16, device=device
        )
    return MixedTrellisBuffers(
        rotation_gate=rotation_gate,
        rotation_up=rotation_up,
        fc1=torch.empty((capacity_rows, fc1_cols), dtype=torch.float16, device=device),
        activated=torch.empty(
            (capacity_rows, launch.intermediate_size),
            dtype=torch.float16,
            device=device,
        ),
        fc2=rotation_gate,
        output=torch.empty(
            (launch.size_m,) if launch.row_addresses else (launch.size_m, launch.hidden_size),
            dtype=(
                torch.int64 if launch.row_addresses else (
                    torch.bfloat16 if launch.topk_sum.full_rotation_output_dtype == "bf16"
                    else torch.float32
                )
            ),
            device=device,
        ),
        packed_route_indices=torch.empty(route_slots, dtype=torch.int32, device=device),
        block_expert_ids=torch.empty(route_blocks, dtype=torch.int32, device=device),
        packed_route_count=torch.empty(1, dtype=torch.int32, device=device),
        expert_offsets=torch.empty(
            route_num_experts + 1, dtype=torch.int32, device=device
        ),
        expert_counts=torch.empty(route_num_experts, dtype=torch.int32, device=device),
        fc1_scratch=torch.empty(
            packed_gemm_scratch_elements(
                size_n=fc1_cols,
                route_slots=route_slots,
                moe_block_size=launch.moe_block_size,
                sms=sms,
            ),
            dtype=torch.float32,
            device=device,
        ),
        fc2_scratch=torch.empty(
            packed_gemm_scratch_elements(
                size_n=launch.hidden_size,
                route_slots=route_slots,
                moe_block_size=launch.moe_block_size,
                sms=sms,
            ),
            dtype=torch.float32,
            device=device,
        ),
        workspace=torch.zeros(
            max(sms * 4, launch.blocks_per_sm * sms) + 2,
            dtype=torch.int32,
            device=device,
        ),
    )


def make_mixed_trellis_segment_buffers(
    launch: MixedTrellisCompileResult, *, rows: int, device: torch.device, sms: int,
    scratch: torch.Tensor | None = None,
) -> MixedTrellisBuffers:
    """Bind a live-sized WS row-address arena; the caller retains its backing.

    Graph replay must retain this arena and both external address tables until
    queued GPU work drains. No allocation happens when ``scratch`` is supplied.
    """
    from .host import exl3_ws_segment_scratch_regions

    if not launch.row_addresses or not launch.warp_specialized or launch.ws_prerotate:
        raise ValueError("segment buffers require row-address WS with in-tile rotation")
    sms = int(launch.workspace_sms or sms)
    regions = exl3_ws_segment_scratch_regions(
        rows=rows, maximum=launch.size_m, hidden=launch.hidden_size,
        intermediate=launch.intermediate_size, topk=launch.top_k,
        experts=int(launch.topk_sum.route_num_experts), block=launch.moe_block_size,
        max_m_blocks=launch.max_m_blocks, sms=sms)
    total = sum((n + 255) & ~255 for n in regions.values())
    if scratch is None:
        scratch = torch.empty(total, dtype=torch.uint8, device=device)
    if (scratch.dtype != torch.uint8 or scratch.device != torch.device(device)
            or not scratch.is_contiguous() or scratch.numel() < total
            or scratch.data_ptr() % 256):
        raise ValueError("EXL3 segment scratch must be contiguous, 256B-aligned uint8 with sufficient bytes")
    tensors = {}
    offset = 0
    routes = rows * launch.top_k
    shapes = dict(rotation_gate=(routes, launch.hidden_size),
                  fc1=(routes, 2 * launch.intermediate_size),
                  activated=(routes, launch.intermediate_size))
    for name, size in regions.items():
        dtype = (torch.float16 if name in shapes else torch.int64 if name == "output"
                 else torch.float32 if name.endswith("scratch") else torch.int32)
        view = scratch[offset:offset + size].view(dtype)
        tensors[name] = view.reshape(shapes[name]) if name in shapes else view
        offset += (size + 255) & ~255
    tensors["workspace"].zero_()
    tensors["rotation_up"] = tensors["fc2"] = tensors["rotation_gate"]
    return MixedTrellisBuffers(**tensors)


def make_mixed_trellis_buffers(
    launch: MixedTrellisCompileResult,
    *,
    device: torch.device,
    sms: int,
) -> MixedTrellisBuffers:
    return _make_mixed_trellis_buffers(
        launch,
        device=device,
        sms=sms,
        route_num_experts=int(launch.topk_sum.route_num_experts),
    )


def make_mixed_trellis3_buffers(
    launch: MixedTrellis3CompileResult,
    *,
    device: torch.device,
    sms: int,
) -> MixedTrellisBuffers:
    return _make_mixed_trellis_buffers(
        launch,
        device=device,
        sms=sms,
        route_num_experts=int(launch.topk_sum.route_num_experts),
    )


def build_ordered_maps(
    tier0_num_experts: int,
    tier1_num_experts: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tier0_num_experts = int(tier0_num_experts)
    tier1_num_experts = int(tier1_num_experts)
    return build_tiered_maps(
        range(tier0_num_experts),
        range(tier0_num_experts, tier0_num_experts + tier1_num_experts),
        device=device,
    )


def build_tiered_maps(
    tier0_global_ids: Sequence[int],
    tier1_global_ids: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build immutable route and descriptor maps for one mixed expert layer."""

    tier0_ids = tuple(int(expert_id) for expert_id in tier0_global_ids)
    tier1_ids = tuple(int(expert_id) for expert_id in tier1_global_ids)
    if any(len(ids) > _MAX_TIER_EXPERTS for ids in (tier0_ids, tier1_ids)):
        raise ValueError(
            f"each mixed Trellis tier supports at most {_MAX_TIER_EXPERTS} experts"
        )
    total = len(tier0_ids) + len(tier1_ids)
    descriptor_bits = _descriptor_bits(total)
    if sorted((*tier0_ids, *tier1_ids)) != list(range(total)):
        raise ValueError(
            f"mixed Trellis tier ids must be a disjoint partition of [0, {total})"
        )
    global_to_combined_host = [-1] * total
    for local_id, global_id in enumerate(tier0_ids):
        global_to_combined_host[global_id] = local_id
    for local_id, global_id in enumerate(tier1_ids):
        global_to_combined_host[global_id] = len(tier0_ids) + local_id
    global_to_combined = torch.tensor(
        global_to_combined_host, dtype=torch.int32, device=device
    )
    descriptor_row = torch.tensor(
        [
            *range(len(tier0_ids)),
            *((1 << descriptor_bits) | i for i in range(len(tier1_ids))),
        ],
        dtype=torch.int32,
        device=device,
    )
    # The descriptor table carries one row per projection (gate, up, down).
    # Per-expert tiering is the degenerate case where all three rows are
    # identical, reproducing single-row behaviour bit-for-bit.
    descriptor = descriptor_row.repeat(3).contiguous()
    # Preserve the per-tier gate/up counts encoded by the descriptor so binding
    # can reject a mismatched storage contract without a device synchronization.
    # Per-expert tiering has gate == up == the tier partition.
    counts = (len(tier0_ids), len(tier1_ids))
    descriptor._mt_projection_counts = (counts, counts)
    return global_to_combined, descriptor


def build_projection_tiered_maps(
    gate_tiers: Sequence[int],
    up_tiers: Sequence[int],
    down_tiers: Sequence[int],
    *,
    tier_slots: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build route and projection descriptors for a mixed-bitrate MoE layer.

    Each projection argument contains one tier ID per global expert. A global
    expert may use different tiers for gate, up, and down projections, so the
    route map remains an identity map and the descriptor rows carry all tier
    and tier-local slot information.

    Returns ``(global_to_combined, descriptor_map)``. ``descriptor_map`` is a
    contiguous ``int32[3 * sum(tier_slots)]`` tensor laid out as gate, up, and
    down rows. A populated entry encodes
    ``(tier << bits) | tier_local_index`` (9 bits through E512, otherwise 10);
    unused padded slots contain ``-1``.
    """

    projections = (
        ("gate", tuple(int(t) for t in gate_tiers)),
        ("up", tuple(int(t) for t in up_tiers)),
        ("down", tuple(int(t) for t in down_tiers)),
    )
    slots = tuple(int(value) for value in tier_slots)
    if len(slots) not in (2, 3):
        raise ValueError(
            "mixed Trellis tier_slots must contain exactly two or three counts"
        )
    num_experts = len(projections[0][1])
    descriptor_bits = _descriptor_bits(num_experts)
    max_tier_experts = 1 << descriptor_bits
    if any(value < 0 or value > max_tier_experts for value in slots):
        raise ValueError(
            "mixed Trellis tier slots must be in "
            f"[0, {max_tier_experts}]"
        )
    num_experts = len(projections[0][1])
    rows: list[int] = []
    projection_counts: list[tuple[int, ...]] = []
    for name, tiers in projections:
        if len(tiers) != num_experts:
            raise ValueError(
                "mixed Trellis projection tier lists must agree on expert count: "
                f"{name} has {len(tiers)}, expected {num_experts}"
            )
        if any(t < 0 or t >= len(slots) for t in tiers):
            raise ValueError(
                f"mixed Trellis {name} tier ids must be in [0, {len(slots)})"
            )
        counters = [0] * len(slots)
        row = []
        for tier in tiers:
            local = counters[tier]
            counters[tier] += 1
            if local >= max_tier_experts:
                raise ValueError(
                    f"mixed Trellis {name} tier {tier} exceeds "
                    f"{_MAX_TIER_EXPERTS} experts"
                )
            row.append((tier << descriptor_bits) | local)
        projection_counts.append(tuple(counters))
        rows.extend(row)
    # The launch sizes the descriptor namespace as the sum of the tier slot
    # counts. A tier slot is max(gate_count, up_count), so with per-projection
    # tiering that sum can exceed the real expert count. Routing remains a
    # separate, exact-size namespace; only the descriptor rows are padded.
    required_fc1 = tuple(
        max(projection_counts[0][tier], projection_counts[1][tier])
        for tier in range(len(slots))
    )
    if any(slots[tier] < required_fc1[tier] for tier in range(len(slots))):
        raise ValueError(
            "mixed Trellis tier slots cannot address all gate/up locals: "
            f"slots={slots}, required={required_fc1}"
        )
    stride = sum(slots)
    if stride < num_experts:
        raise ValueError(
            f"mixed Trellis tier slots ({stride}) cannot address {num_experts} experts"
        )
    global_to_combined = torch.arange(num_experts, dtype=torch.int32, device=device)
    descriptor = torch.full((3 * stride,), -1, dtype=torch.int32, device=device)
    for row_index in range(3):
        base = row_index * num_experts
        descriptor[row_index * stride : row_index * stride + num_experts] = (
            torch.tensor(
                rows[base : base + num_experts], dtype=torch.int32, device=device
            )
        )
    # Preserve the gate/up counts encoded by the descriptor so binding can
    # reject a mismatched storage contract without a device synchronization.
    descriptor._mt_projection_counts = (
        projection_counts[0],
        projection_counts[1],
    )
    return global_to_combined, descriptor


def _check_descriptor_projection_counts(
    descriptor_map: torch.Tensor,
    total_experts: int,
    *,
    gate_counts: tuple[int, ...],
    up_counts: tuple[int, ...],
    descriptor_bits: int = 9,
) -> None:
    """Fail closed when launch counts disagree with the descriptor map.

    The device dispatch bounds gate/up locals by the launch counts, so a
    descriptor entry at or beyond its projection's count is silently skipped
    and the corresponding FC1 half stays zero. The in-tree builders publish
    the counts their descriptor encodes; descriptors from other producers pay
    one host copy here, memoized on the tensor, so steady-state launches
    never synchronize.
    """

    encoded = getattr(descriptor_map, "_mt_projection_counts", None)
    if encoded is None:
        rows = descriptor_map.detach().cpu().reshape(-1)[:3 * total_experts].view(3, total_experts)
        derived = []
        tier_count = len(gate_counts)
        for row in rows[:2]:
            live = row[row >= 0]
            encoded_tiers = live >> descriptor_bits
            if bool((encoded_tiers >= tier_count).any()):
                raise ValueError(
                    "mixed Trellis descriptor contains a tier outside the "
                    f"launch range [0, {tier_count})"
                )
            derived.append(
                tuple(int((encoded_tiers == tier).sum()) for tier in range(tier_count))
            )
        encoded = (tuple(derived[0]), tuple(derived[1]))
        descriptor_map._mt_projection_counts = encoded
    expected_gate = tuple(int(value) for value in encoded[0])
    expected_up = tuple(int(value) for value in encoded[1])
    got_gate = tuple(int(value) for value in gate_counts)
    got_up = tuple(int(value) for value in up_counts)
    if got_gate != expected_gate or got_up != expected_up:
        raise ValueError(
            "mixed Trellis projection counts disagree with the descriptor "
            f"map: gate {got_gate} vs encoded {expected_gate}, up {got_up} "
            f"vs encoded {expected_up}"
        )


def _require_capture_safe_descriptor_metadata(
    descriptor_map: torch.Tensor,
) -> None:
    if torch.cuda.is_current_stream_capturing() and getattr(
        descriptor_map, "_mt_projection_counts", None
    ) is None:
        raise RuntimeError(
            "mixed Trellis capture-time binding requires descriptor projection "
            "counts prepared on the host"
        )


def combine_trellis_rotations(
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    *additional_tiers: MixedTrellisTier,
) -> MixedTrellisRotations:
    """Materialize one tier-ordered table set once during model preparation."""
    tiers = (tier0, tier1, *additional_tiers)
    return MixedTrellisRotations(
        intermediate=torch.cat(
            tuple(tier.intermediate_rotations for tier in tiers), dim=0
        ).contiguous(),
        gate_suh=torch.cat(tuple(tier.gate_suh for tier in tiers), dim=0).contiguous(),
        up_suh=torch.cat(tuple(tier.up_suh for tier in tiers), dim=0).contiguous(),
        down_svh=torch.cat(tuple(tier.down_svh for tier in tiers), dim=0).contiguous(),
    )


def _compact_trellis_layout(rows, *, hidden_size, intermediate_size, bits, descriptor_bits=9):
    """Canonical compact ABI: three tier/local rows, live width, three U32 offsets."""
    if len(rows) != 4 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("compact descriptor requires four equally sized source rows")
    if hidden_size <= 0 or hidden_size % 128 or intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("compact descriptor geometry must use whole H128 blocks")
    if len(bits) != 2 or any(bit not in (2, 3, 4, 5, 6) for bit in bits):
        raise ValueError("compact descriptor requires two supported trellis tiers")
    offsets = [[-1] * len(rows[0]) for _ in range(3)]
    sizes = [[0, 0, 0] for _ in bits]
    counts = [[0, 0, 0] for _ in bits]
    for projection in range(3):
        for expert, descriptor in enumerate(rows[projection]):
            width = rows[3][expert]
            if all(row[expert] == -1 for row in rows[:3]) and width == -1:
                continue
            if not isinstance(width, int) or isinstance(width, bool) or not 128 <= width <= intermediate_size or width % 128:
                raise ValueError("compact live width must be positive H128 and within resident width")
            if descriptor < 0:
                continue
            tier, local = descriptor >> descriptor_bits, descriptor & ((1 << descriptor_bits) - 1)
            if tier >= len(bits) or local != counts[tier][projection]:
                raise ValueError("compact descriptors require canonical dense projection-local IDs")
            offsets[projection][expert] = sizes[tier][projection]
            sizes[tier][projection] += hidden_size * width * bits[tier] // 32
            counts[tier][projection] += 1
    for expert, descriptor in enumerate(rows[1]):
        if descriptor >= 0:
            offsets[1][expert] += sizes[descriptor >> descriptor_bits][0]
    if any(value > 0x7fffffff for row in offsets for value in row):
        raise ValueError("compact projection offsets exceed signed int32 ABI")
    payloads = tuple((size[0] + size[1], size[2]) for size in sizes)
    return [*rows, *offsets], payloads, tuple(tuple(row) for row in counts)


def build_compact_trellis_descriptor(descriptor_map, widths, *, hidden_size, intermediate_size,
                                      bits=(3, 4), descriptor_bits=9):
    """Prepare immutable true-width weight offsets before graph capture."""
    if descriptor_map.dtype != torch.int32 or descriptor_map.numel() % 3:
        raise ValueError("compact source descriptor must have three int32 rows")
    rows = descriptor_map.detach().cpu().reshape(3, -1).tolist()
    rows.append(list(widths))
    packed, _, _ = _compact_trellis_layout(rows, hidden_size=hidden_size,
        intermediate_size=intermediate_size, bits=bits, descriptor_bits=descriptor_bits)
    result = torch.tensor(packed, dtype=torch.int32, device=descriptor_map.device).flatten()
    if hasattr(descriptor_map, "_mt_projection_counts"):
        result._mt_projection_counts = descriptor_map._mt_projection_counts
    return result


def _validate_mixed_trellis_tier_storage(
    *,
    name: str,
    tier: MixedTrellisTier,
    expected_experts: int,
    bits: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    gate_experts: int | None = None,
    up_experts: int | None = None,
    compact_payload: tuple[int, int, int] | None = None,
) -> None:
    """Fail closed before binding expert-sized storage as raw CuTe pointers."""
    expected_experts = int(expected_experts)
    bits = int(bits)
    # Per-projection membership lets a tier hold a different number of FC2
    # (down) experts than FC1 (gate/up) slots, so the FC2 count cannot be
    # assumed equal to expected_experts. Derive it from the W2 payload itself,
    # which is the tensor that actually carries the data, then require the
    # global-scale vector to agree. Deriving it from the scale vector instead
    # would report a malformed scale as a confusing W2 extent error.
    w2_expert_stride = (
        (int(intermediate_size) // 16) * (int(hidden_size) // 16) * (8 * bits)
    )
    w2_elements = int(tier.w2.numel())
    # The FC2 expert count is independent of the FC1 gate/up slot counts.
    # The widest descriptor supports 1024 tier-local experts.
    if compact_payload is None and (
        tier.w2.dtype != torch.int32
        or w2_expert_stride <= 0
        or w2_elements % w2_expert_stride != 0
        or not 1 <= w2_elements // w2_expert_stride <= _MAX_TIER_EXPERTS
    ):
        raise ValueError(
            f"mixed Trellis {name}.w2 must be torch.int32 holding "
            f"1..{_MAX_TIER_EXPERTS} whole FC2 experts of "
            f"{w2_expert_stride} elements, got {w2_elements}"
        )
    fc2_experts = w2_elements // w2_expert_stride if compact_payload is None else compact_payload[2]
    if gate_experts is None and up_experts is None:
        gate_experts = expected_experts
        up_experts = expected_experts
    elif gate_experts is None or up_experts is None:
        raise ValueError(
            f"mixed Trellis {name} requires paired gate_experts/up_experts"
        )
    gate_experts = int(gate_experts)
    up_experts = int(up_experts)
    if not (
        0 <= gate_experts <= expected_experts and 0 <= up_experts <= expected_experts
    ):
        raise ValueError(
            f"mixed Trellis {name} projection counts must both be in "
            f"[0, {expected_experts}], got gate={gate_experts}, up={up_experts}"
        )
    # Callers without projection counts use one gate and one up plane per tier
    # slot. Projection-aware callers couple the physical payload exactly to
    # gate_experts + up_experts. One dummy plane is permitted only when both
    # projection counts are zero.
    _proj_stride = (
        (int(hidden_size) // 16) * (int(intermediate_size) // 16) * (8 * bits)
    )
    _w13_planes = int(tier.w13.numel()) // _proj_stride if _proj_stride else 0
    _expected_w13_planes = max(gate_experts + up_experts, 1)
    if (
        tier.w13.dtype != torch.int32
        or tier.w13.device != device
        or not tier.w13.is_contiguous()
        or _proj_stride <= 0
        or (compact_payload is None and (
            int(tier.w13.numel()) % _proj_stride != 0
            or _w13_planes != _expected_w13_planes))
        or (compact_payload is not None and int(tier.w13.numel()) != max(compact_payload[0], 4))
        or int(tier.w13.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            f"mixed Trellis {name}.w13 must be contiguous int32 on {device} "
            f"with exactly {_expected_w13_planes} projection planes, got "
            f"{int(tier.w13.numel())} elements"
        )
    expected = (
        (
            "w2",
            tier.w2,
            torch.int32,
            (fc2_experts
            * (int(intermediate_size) // 16)
            * (int(hidden_size) // 16)
            * (8 * bits)) if compact_payload is None else max(compact_payload[1], 4),
        ),
        # Native Trellis decodes its codebook tiles directly. These scale
        # pointers retain the prepared four-byte dummy ABI and are not the
        # per-expert K/32 scale grids used by packed weights.
        ("w13_scale", tier.w13_scale, torch.uint8, 4),
        ("w2_scale", tier.w2_scale, torch.uint8, 4),
        (
            "w13_global_scale",
            tier.w13_global_scale,
            torch.float32,
            expected_experts,
        ),
        (
            "w2_global_scale",
            tier.w2_global_scale,
            torch.float32,
            fc2_experts,
        ),
    )
    for field, tensor, expected_dtype, expected_elements in expected:
        if (
            tensor.dtype != expected_dtype
            or tensor.device != device
            or not tensor.is_contiguous()
            or int(tensor.numel()) != int(expected_elements)
            or int(tensor.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"mixed Trellis {name}.{field} must be contiguous "
                f"{expected_dtype} on {device} with {int(expected_elements)} "
                "elements and at least 16-byte alignment"
            )


def _mixed_trellis_projection_counts(
    name: str,
    values: tuple[int, int] | None,
    defaults: tuple[int, int],
) -> tuple[int, int]:
    if values is None:
        return defaults
    if (
        not isinstance(values, tuple)
        or len(values) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        )
    ):
        raise TypeError(f"mixed Trellis {name} must contain two integer counts")
    return tuple(int(value) for value in values)


def _mixed_trellis_tier_pointers(tier: MixedTrellisTier) -> tuple[object, ...]:
    return (
        make_ptr(
            cutlass.Int32,
            tier.w13.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            tier.w2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            tier.w13_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            tier.w2_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            tier.w13_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            tier.w2_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def bind_mixed_trellis(
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    global_to_combined: torch.Tensor,
    descriptor_map: torch.Tensor,
    rotations: MixedTrellisRotations,
    launch: MixedTrellisCompileResult,
    *,
    gate_experts: tuple[int, int] | None = None,
    up_experts: tuple[int, int] | None = None,
) -> MixedTrellisBinding:
    """Validate immutable two-tier artifacts and bind their launch pointers."""

    _require_capture_safe_descriptor_metadata(descriptor_map)
    if (gate_experts is None) != (up_experts is None):
        raise ValueError(
            "mixed Trellis projection-tight storage requires paired "
            "gate_experts/up_experts"
        )
    counts = (
        int(launch.tier0_num_experts),
        int(launch.tier1_num_experts),
    )
    gate_counts = _mixed_trellis_projection_counts("gate_experts", gate_experts, counts)
    up_counts = _mixed_trellis_projection_counts("up_experts", up_experts, counts)
    tiers = (tier0, tier1)
    bits = (launch.tier0_bits, launch.tier1_bits)
    device = tier0.w13.device
    compact_payloads = None
    compact_counts = None
    if launch.compact_weights:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compact descriptor validation must precede graph capture")
        if descriptor_map.dtype != torch.int32 or descriptor_map.numel() != 7 * sum(counts):
            raise ValueError("compact descriptor requires seven int32 rows")
        host_rows = descriptor_map.detach().cpu().reshape(7, -1).tolist()
        expected, compact_payloads, compact_counts = _compact_trellis_layout(host_rows[:4],
            hidden_size=launch.hidden_size, intermediate_size=launch.intermediate_size,
            bits=bits, descriptor_bits=_descriptor_bits(int(launch.topk_sum.route_num_experts)))
        if host_rows != expected:
            raise ValueError("compact descriptor projection offsets are not canonical")
        if not launch.warp_specialized and any(width > 0 and width % launch.fc1_tile_n for width in host_rows[3]):
            raise ValueError("compact live widths must align to the FC1 N tile")
    for tier_id, (tier, expected_experts, tier_bits) in enumerate(
        zip(tiers, counts, bits, strict=True)
    ):
        actual_experts = int(tier.num_experts)
        if actual_experts != expected_experts:
            raise ValueError(
                f"mixed Trellis tier{tier_id} has {actual_experts} experts, "
                f"expected the launch-plan count {expected_experts}"
            )
        tier_codebook = str(tier.trellis_codebook).lower()
        if tier_codebook != launch.trellis_codebook:
            raise ValueError(
                f"mixed Trellis tier{tier_id} uses codebook {tier_codebook!r}, "
                f"expected the launch-plan codebook {launch.trellis_codebook!r}"
            )
        _validate_mixed_trellis_tier_storage(
            name=f"tier{tier_id}",
            tier=tier,
            expected_experts=expected_experts,
            bits=tier_bits,
            hidden_size=launch.hidden_size,
            intermediate_size=launch.intermediate_size,
            device=device,
            gate_experts=gate_counts[tier_id],
            up_experts=up_counts[tier_id],
            compact_payload=None if compact_payloads is None else (
                *compact_payloads[tier_id], compact_counts[tier_id][2]),
        )

    total_experts = sum(counts)
    route_num_experts = _mixed_route_num_experts(
        global_to_combined, int(launch.topk_sum.route_num_experts)
    )
    for name, mapping, expected_entries in (
        ("global_to_combined", global_to_combined, route_num_experts),
        ("descriptor_map", descriptor_map, (7 if launch.compact_weights else (4 if launch.paired_boundary is not None else 3)) * total_experts),
    ):
        if (
            mapping.dtype != torch.int32
            or mapping.device != device
            or not mapping.is_contiguous()
            or int(mapping.numel()) != expected_entries
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous int32 on {device} "
                f"with {expected_entries} elements"
            )
    _check_descriptor_projection_counts(
        descriptor_map,
        total_experts,
        gate_counts=gate_counts,
        up_counts=up_counts,
        descriptor_bits=_descriptor_bits(route_num_experts),
    )
    for name, table, expected_elements in (
        (
            "intermediate rotations",
            rotations.intermediate,
            total_experts * 3 * launch.intermediate_size,
        ),
        (
            "gate SUH",
            rotations.gate_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "up SUH",
            rotations.up_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "down SVH",
            rotations.down_svh,
            (1 if launch.broadcast_svh else total_experts) * launch.hidden_size,
        ),
    ):
        if (
            table.dtype != torch.float16
            or table.device != device
            or not table.is_contiguous()
            or int(table.numel()) != expected_elements
            or int(table.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous fp16 on {device} "
                f"with {expected_elements} elements and at least 16-byte alignment"
            )
    return MixedTrellisBinding(
        launch=launch,
        tiers=tiers,
        global_to_combined=global_to_combined,
        descriptor_map=descriptor_map,
        rotations=rotations,
        gate_counts=gate_counts,
        up_counts=up_counts,
        fc2_counts=tuple(int(tier.w2_global_scale.numel()) for tier in tiers),
        total_experts=total_experts,
        route_num_experts=route_num_experts,
        device=device,
        tier_pointers=tuple(_mixed_trellis_tier_pointers(tier) for tier in tiers),
        descriptor_map_ptr=make_ptr(
            cutlass.Int32,
            descriptor_map.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        intermediate_rotations_ptr=make_ptr(
            cutlass.Float16,
            rotations.intermediate.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        gate_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.gate_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        up_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.up_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        global_to_combined_ptr=make_ptr(
            cutlass.Int32,
            global_to_combined.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        down_svh_ptr=make_ptr(
            cutlass.Float16,
            rotations.down_svh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def run_bound_mixed_trellis(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    binding: MixedTrellisBinding,
    buffers: MixedTrellisBuffers,
) -> torch.Tensor:
    """Run a graph-safe two-tier launch from a validated binding."""

    launch = binding.launch
    wire_input = getattr(launch, "input_format", "bf16") == "e4m3_k32"
    input_cols = int(launch.hidden_size) + (
        int(launch.hidden_size) // 32 if wire_input else 0
    )
    expected_shape = (int(x.shape[0]),) if launch.row_addresses and x.ndim else None
    valid_shape = tuple(x.shape) == expected_shape if launch.row_addresses else (x.ndim == 2 and int(x.shape[1]) == input_cols)
    if not valid_shape:
        raise ValueError(
            "mixed Trellis input must have shape "
            f"[M, {input_cols}], got {tuple(x.shape)}"
        )
    m = int(x.shape[0])
    if m <= 0:
        raise ValueError(f"mixed Trellis requires at least one active row, got {m}")
    if m > launch.size_m:
        raise ValueError(f"active rows {m} exceed launch capacity {launch.size_m}")
    expected_route_shape = (m, int(launch.top_k))
    for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
        if tuple(tensor.shape) != expected_route_shape:
            raise ValueError(
                f"mixed Trellis {name} must have shape {expected_route_shape}, "
                f"got {tuple(tensor.shape)}"
            )
    expected_input_dtype = (
        torch.uint8
        if wire_input
        else torch.bfloat16 if launch.rotation_input_dtype == "bf16" else torch.float16
    )
    if launch.row_addresses:
        expected_input_dtype = torch.int64
    for name, tensor, expected_dtype in (
        ("input", x, expected_input_dtype),
        ("topk_ids", topk_ids, launch.route_ids_dtype),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"mixed Trellis {name} must be {expected_dtype}, got {tensor.dtype}"
            )
        if tensor.device != binding.device:
            raise ValueError(
                f"mixed Trellis {name} must be on {binding.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"mixed Trellis {name} must be contiguous")
    expected_output_dtype = (
        torch.bfloat16
        if launch.topk_sum.full_rotation_output_dtype == "bf16"
        else torch.float32
    )
    expected_output_shapes = {
        (m, launch.hidden_size),
        (launch.size_m, launch.hidden_size),
    }
    if launch.row_addresses:
        expected_output_dtype = torch.int64
        expected_output_shapes = {(m,), (launch.size_m,)}
    if (
        tuple(buffers.output.shape) not in expected_output_shapes
        or buffers.output.dtype != expected_output_dtype
        or buffers.output.device != binding.device
        or not buffers.output.is_contiguous()
        or int(buffers.output.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            "mixed Trellis output buffer must be contiguous, 16-byte-aligned "
            f"{expected_output_dtype} with live [{m},{launch.hidden_size}] or "
            f"capacity [{launch.size_m},{launch.hidden_size}] shape on "
            f"{binding.device}"
        )
    if int(x.data_ptr()) % 16 != 0:
        raise ValueError("mixed Trellis input must have at least 16-byte alignment")

    live_routes = m * launch.top_k
    required_route_slots = (
        live_routes
        if launch.direct_topk_routes
        else max_packed_route_slots(
            live_routes, launch.moe_block_size, binding.route_num_experts
        )
    )
    required_route_blocks = (
        live_routes
        if launch.direct_topk_routes
        else (required_route_slots + launch.moe_block_size - 1)
        // launch.moe_block_size
    )
    if required_route_blocks > launch.max_m_blocks:
        raise RuntimeError(
            "mixed Trellis request requires "
            f"{required_route_blocks} route blocks, but the launch supports "
            f"{launch.max_m_blocks}"
        )
    if buffers.packed_route_indices.numel() < required_route_slots:
        raise RuntimeError(
            "mixed Trellis packed-route buffer is below request capacity"
        )
    if buffers.block_expert_ids.numel() < required_route_blocks:
        raise RuntimeError(
            "mixed Trellis block-expert buffer is below request capacity"
        )
    if launch.direct_topk_routes:
        # Map global expert ids inside the cooperative grid. Uniform Trellis
        # already follows this contract; saving one launch per MoE layer is
        # material for speculative decode's tiny direct-route batches.
        packed = buffers.packed_route_indices[:live_routes]
        raw_topk_ids = topk_ids.reshape(-1)
        block_experts = buffers.block_expert_ids
        packed_count = buffers.packed_route_count
    else:
        packed, block_experts, packed_count = pack_topk_routes_by_expert(
            topk_ids,
            launch.moe_block_size,
            binding.route_num_experts,
            expert_map=binding.global_to_combined,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
        )
        raw_topk_ids = topk_ids.reshape(-1)
    stream = current_cuda_stream()
    launch.compiled(
        make_ptr(
            cutlass.Int64 if launch.row_addresses else cutlass.Uint8 if wire_input else _cutlass_element_dtype(launch.rotation_input_dtype),
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        *binding.tier_pointers[0],
        *binding.tier_pointers[1],
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        raw_topk_ids,
        block_experts,
        packed_count,
        binding.descriptor_map_ptr,
        binding.global_to_combined_ptr,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        binding.intermediate_rotations_ptr,
        binding.gate_suh_ptr,
        binding.up_suh_ptr,
        make_ptr(
            cutlass.Uint8,
            launch.trellis_lut.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(launch.tier0_num_experts),
        Int32(launch.tier1_num_experts),
        Int32(binding.fc2_counts[0]),
        Int32(binding.fc2_counts[1]),
        m,
        max(int(launch.blocks_per_sm) * int(launch.sms), 1),
        stream,
        tier0_gate_experts=Int32(binding.gate_counts[0]),
        tier1_gate_experts=Int32(binding.gate_counts[1]),
        tier0_up_experts=Int32(binding.up_counts[0]),
        tier1_up_experts=Int32(binding.up_counts[1]),
        route_num_experts=Int32(binding.route_num_experts),
    )
    launch.topk_sum.compiled(
        make_ptr(
            cutlass.Float16,
            buffers.fc2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int64 if launch.row_addresses else (
                cutlass.BFloat16
                if launch.topk_sum.full_rotation_output_dtype == "bf16"
                else cutlass.Float32
            ),
            buffers.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32 if topk_ids.dtype == torch.int32 else cutlass.Int64,
            topk_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4 if topk_ids.dtype == torch.int32 else 8,
        ),
        binding.global_to_combined_ptr,
        binding.down_svh_ptr,
        Int32(launch.topk_sum.num_experts),
        Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


def _mixed_trellis3_projection_counts(
    name: str,
    values: tuple[int, int, int] | None,
    defaults: tuple[int, int, int],
) -> tuple[int, int, int]:
    if values is None:
        return defaults
    if (
        not isinstance(values, tuple)
        or len(values) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        )
    ):
        raise TypeError(
            f"three-tier mixed Trellis {name} must contain three integer counts"
        )
    return tuple(int(value) for value in values)


def bind_mixed_trellis3(
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    tier2: MixedTrellisTier,
    global_to_combined: torch.Tensor,
    descriptor_map: torch.Tensor,
    rotations: MixedTrellisRotations,
    launch: MixedTrellis3CompileResult,
    *,
    gate_experts: tuple[int, int, int] | None = None,
    up_experts: tuple[int, int, int] | None = None,
) -> MixedTrellis3Binding:
    """Validate immutable K3/K4/K5 artifacts and bind their launch pointers."""

    _require_capture_safe_descriptor_metadata(descriptor_map)
    if (gate_experts is None) != (up_experts is None):
        raise ValueError(
            "mixed Trellis projection-tight storage requires paired "
            "gate_experts/up_experts"
        )
    counts = (
        int(launch.tier0_num_experts),
        int(launch.tier1_num_experts),
        int(launch.tier2_num_experts),
    )
    gate_counts = _mixed_trellis3_projection_counts(
        "gate_experts", gate_experts, counts
    )
    up_counts = _mixed_trellis3_projection_counts("up_experts", up_experts, counts)
    tiers = (tier0, tier1, tier2)
    bits = (launch.tier0_bits, launch.tier1_bits, launch.tier2_bits)
    device = tier0.w13.device
    for tier_id, (tier, expected_experts, tier_bits) in enumerate(
        zip(tiers, counts, bits, strict=True)
    ):
        actual_experts = int(tier.num_experts)
        if actual_experts != expected_experts:
            raise ValueError(
                f"mixed Trellis tier{tier_id} has {actual_experts} experts, "
                f"expected the launch-plan count {expected_experts}"
            )
        tier_codebook = str(tier.trellis_codebook).lower()
        if tier_codebook != launch.trellis_codebook:
            raise ValueError(
                f"mixed Trellis tier{tier_id} uses codebook {tier_codebook!r}, "
                f"expected the launch-plan codebook {launch.trellis_codebook!r}"
            )
        _validate_mixed_trellis_tier_storage(
            name=f"tier{tier_id}",
            tier=tier,
            expected_experts=expected_experts,
            bits=tier_bits,
            hidden_size=launch.hidden_size,
            intermediate_size=launch.intermediate_size,
            device=device,
            gate_experts=gate_counts[tier_id],
            up_experts=up_counts[tier_id],
        )

    total_experts = sum(counts)
    route_num_experts = _mixed_route_num_experts(
        global_to_combined, int(launch.topk_sum.route_num_experts)
    )
    for name, mapping, expected_entries in (
        ("global_to_combined", global_to_combined, route_num_experts),
        ("descriptor_map", descriptor_map, 3 * total_experts),
    ):
        if (
            mapping.dtype != torch.int32
            or mapping.device != device
            or not mapping.is_contiguous()
            or int(mapping.numel()) != expected_entries
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous int32 on {device} "
                f"with {expected_entries} elements"
            )
    _check_descriptor_projection_counts(
        descriptor_map,
        total_experts,
        gate_counts=gate_counts,
        up_counts=up_counts,
        descriptor_bits=_descriptor_bits(route_num_experts),
    )
    for name, table, expected_elements in (
        (
            "intermediate rotations",
            rotations.intermediate,
            total_experts * 3 * launch.intermediate_size,
        ),
        (
            "gate SUH",
            rotations.gate_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "up SUH",
            rotations.up_suh,
            (1 if launch.broadcast_suh else total_experts) * launch.hidden_size,
        ),
        (
            "down SVH",
            rotations.down_svh,
            (1 if launch.broadcast_svh else total_experts) * launch.hidden_size,
        ),
    ):
        if (
            table.dtype != torch.float16
            or table.device != device
            or not table.is_contiguous()
            or int(table.numel()) != expected_elements
            or int(table.data_ptr()) % 16 != 0
        ):
            raise ValueError(
                f"mixed Trellis {name} must be contiguous fp16 on {device} "
                f"with {expected_elements} elements and at least 16-byte alignment"
            )

    return MixedTrellis3Binding(
        launch=launch,
        tiers=tiers,
        global_to_combined=global_to_combined,
        descriptor_map=descriptor_map,
        rotations=rotations,
        gate_counts=gate_counts,
        up_counts=up_counts,
        fc2_counts=tuple(int(tier.w2_global_scale.numel()) for tier in tiers),
        total_experts=total_experts,
        route_num_experts=route_num_experts,
        device=device,
        tier_pointers=tuple(_mixed_trellis_tier_pointers(tier) for tier in tiers),
        descriptor_map_ptr=make_ptr(
            cutlass.Int32,
            descriptor_map.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        intermediate_rotations_ptr=make_ptr(
            cutlass.Float16,
            rotations.intermediate.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        gate_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.gate_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        up_suh_ptr=make_ptr(
            cutlass.Float16,
            rotations.up_suh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        global_to_combined_ptr=make_ptr(
            cutlass.Int32,
            global_to_combined.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        down_svh_ptr=make_ptr(
            cutlass.Float16,
            rotations.down_svh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def run_bound_mixed_trellis3(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    binding: MixedTrellis3Binding,
    buffers: MixedTrellisBuffers,
) -> torch.Tensor:
    """Run a graph-safe K3/K4/K5 launch from a validated binding."""

    launch = binding.launch
    if x.ndim != 2 or int(x.shape[1]) != int(launch.hidden_size):
        raise ValueError(
            "mixed Trellis input must have shape "
            f"[M, {int(launch.hidden_size)}], got {tuple(x.shape)}"
        )
    m = int(x.shape[0])
    if m <= 0:
        raise ValueError(f"mixed Trellis requires at least one active row, got {m}")
    if m > launch.size_m:
        raise ValueError(f"active rows {m} exceed launch capacity {launch.size_m}")
    expected_route_shape = (m, int(launch.top_k))
    for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
        if tuple(tensor.shape) != expected_route_shape:
            raise ValueError(
                f"mixed Trellis {name} must have shape {expected_route_shape}, "
                f"got {tuple(tensor.shape)}"
            )
    expected_input_dtype = (
        torch.bfloat16 if launch.rotation_input_dtype == "bf16" else torch.float16
    )
    for name, tensor, expected_dtype in (
        ("input", x, expected_input_dtype),
        ("topk_ids", topk_ids, launch.route_ids_dtype),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"mixed Trellis {name} must be {expected_dtype}, got {tensor.dtype}"
            )
        if tensor.device != binding.device:
            raise ValueError(
                f"mixed Trellis {name} must be on {binding.device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"mixed Trellis {name} must be contiguous")
    expected_output_dtype = (
        torch.bfloat16
        if launch.topk_sum.full_rotation_output_dtype == "bf16"
        else torch.float32
    )
    expected_output_shapes = {
        (m, launch.hidden_size),
        (launch.size_m, launch.hidden_size),
    }
    if (
        tuple(buffers.output.shape) not in expected_output_shapes
        or buffers.output.dtype != expected_output_dtype
        or buffers.output.device != binding.device
        or not buffers.output.is_contiguous()
        or int(buffers.output.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            "mixed Trellis3 output buffer must be contiguous, 16-byte-aligned "
            f"{expected_output_dtype} with live [{m},{launch.hidden_size}] or "
            f"capacity [{launch.size_m},{launch.hidden_size}] shape on "
            f"{binding.device}"
        )
    if int(x.data_ptr()) % 16 != 0:
        raise ValueError("mixed Trellis input must have at least 16-byte alignment")

    required_route_slots = max_packed_route_slots(
        m * launch.top_k, launch.moe_block_size, binding.route_num_experts
    )
    required_route_blocks = (
        required_route_slots + launch.moe_block_size - 1
    ) // launch.moe_block_size
    if required_route_blocks > launch.max_m_blocks:
        raise RuntimeError(
            "mixed Trellis request requires "
            f"{required_route_blocks} route blocks, but the launch supports "
            f"{launch.max_m_blocks}"
        )
    if buffers.packed_route_indices.numel() < required_route_slots:
        raise RuntimeError(
            "mixed Trellis packed-route buffer is below request capacity"
        )
    if buffers.block_expert_ids.numel() < required_route_blocks:
        raise RuntimeError(
            "mixed Trellis block-expert buffer is below request capacity"
        )
    packed, block_experts, packed_count = pack_topk_routes_by_expert(
        topk_ids,
        launch.moe_block_size,
        binding.route_num_experts,
        expert_map=binding.global_to_combined,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
    )
    stream = current_cuda_stream()

    launch.compiled(
        make_ptr(
            _cutlass_element_dtype(launch.rotation_input_dtype),
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        *binding.tier_pointers[0],
        *binding.tier_pointers[1],
        *binding.tier_pointers[2],
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        block_experts,
        packed_count,
        binding.descriptor_map_ptr,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        binding.intermediate_rotations_ptr,
        binding.gate_suh_ptr,
        binding.up_suh_ptr,
        make_ptr(
            cutlass.Uint8,
            launch.trellis_lut.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(int(launch.tier0_num_experts)),
        Int32(int(launch.tier1_num_experts)),
        Int32(int(launch.tier2_num_experts)),
        Int32(binding.fc2_counts[0]),
        Int32(binding.fc2_counts[1]),
        Int32(binding.fc2_counts[2]),
        m,
        max(int(launch.blocks_per_sm) * int(launch.sms), 1),
        stream,
        tier0_gate_experts=Int32(binding.gate_counts[0]),
        tier1_gate_experts=Int32(binding.gate_counts[1]),
        tier2_gate_experts=Int32(binding.gate_counts[2]),
        tier0_up_experts=Int32(binding.up_counts[0]),
        tier1_up_experts=Int32(binding.up_counts[1]),
        tier2_up_experts=Int32(binding.up_counts[2]),
    )
    launch.topk_sum.compiled(
        make_ptr(
            cutlass.Float16,
            buffers.fc2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            (
                cutlass.BFloat16
                if launch.topk_sum.full_rotation_output_dtype == "bf16"
                else cutlass.Float32
            ),
            buffers.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32 if topk_ids.dtype == torch.int32 else cutlass.Int64,
            topk_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4 if topk_ids.dtype == torch.int32 else 8,
        ),
        binding.global_to_combined_ptr,
        binding.down_svh_ptr,
        Int32(launch.topk_sum.num_experts),
        Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


__all__ = [
    "MixedTrellisBuffers",
    "MixedTrellisBinding",
    "MixedTrellisCompileResult",
    "MixedTrellis3CompileResult",
    "MixedTrellis3Binding",
    "MixedTrellisRotations",
    "W4A16MixedTrellis3Kernel",
    "bind_mixed_trellis",
    "bind_mixed_trellis3",
    "build_ordered_maps",
    "build_projection_tiered_maps",
    "build_tiered_maps",
    "combine_trellis_rotations",
    "compile_mixed_trellis",
    "compile_mixed_trellis3",
    "make_mixed_trellis_buffers",
    "make_mixed_trellis3_buffers",
    "run_bound_mixed_trellis",
    "run_bound_mixed_trellis3",
    "warmup_mixed_trellis_route_pack",
]
