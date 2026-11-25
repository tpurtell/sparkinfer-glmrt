"""Hy4-flash (SigLIP2) vision tower specialized SM120/SM121 CuTe DSL kernels.

Only the 10 Duel B Hy4-flash vision tower programs dispatch into this module:
- 6 biased BF16 projections: (768->1152, 1152->1152, 1152->4304, 4304->1152,
  4608->4608, 4608->3072), using warp-specialized TMA + m16n8k16 BF16->FP32
  MMA, cross-stage pipelining, M-grouped serpentine L2 CTA swizzling,
  row-band tiling (64x96@4 / 64x128@4 / 128x96@3 / 128x128@3), and hoisted
  BF16x2 bias epilogue with single BF16 rounding;
- dynamic_attention_h16_d72 (heads=16, dim=72, segments=8): 8-compute-warp
  ping-pong 128x128 2-stage TMA attention overlapping HMMA (PV+QK) with
  SFU/ALU softmax across WG0/WG1, with Q resident in registers, sQ/sO aliasing
  sK (82 KiB SMEM), zero inner-tile masking, and NaN-safe V-tail zeroing;
- exact erf GELU (width=4304, 4608): ILP=2 128-bit vectorized loads/stores
  with branchless CUDA 13.0 libdevice __nv_erff polynomial;
- scaled bilinear position (channels=1152, 128x128, ATen FMA contraction):
  2D grid (width, height) eliminating integer divisions, warp-uniform
  coordinate math, and 128-bit vectorized loads/stores.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    pack_f32x2_to_bfloat2,
    st_global_u32,
    st_global_v4_u32,
)
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.cute_dsl_utils import assume_tensor_aligned
from b12x.attention._shared.contiguous.named_barrier import NamedBarrierFwd
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention._shared.cute import copy as cute_copy
from b12x.attention._shared.cute import ops as cute_ops
from b12x.attention._shared.cute import pipeline as cute_pipeline
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo

from .pointwise import _add, _fma, _mul, _sub
from .vision_exact_gelu import _ex2_ftz

# =============================================================================
# 1. Hy4 Biased BF16 Projection (TMA Warp-Specialized + L2 Swizzle)
# =============================================================================

HY4_BIASED_SHAPES = frozenset(
    (
        (768, 1152),
        (1152, 1152),
        (1152, 4304),
        (4304, 1152),
        (4608, 4608),
        (4608, 3072),
    )
)

# Per-(width, outputs) row-band tile plans: tuple of
# (max_rows_or_None, tile_m, tile_n, num_stages, group_m).
# All entries use 2x2 compute warps + 1 TMA producer warp (160 threads),
# tile_k=64, mbar_last=True, cross-stage pipelining, and identical ascending
# K16 m16n8k16 accumulation order (bitwise-invariant across row bands).
# Distinct tile configs within a shape are deduplicated so identical bands
# share one compiled kernel.
# - 64x96x64 @ 4 stages:  81,984 B SMEM -> 1 CTA/SM, 4 TMA stages, exact divisor of 1152/3072/4608.
# - 64x128x64 @ 4 stages: 98,368 B SMEM -> 1 CTA/SM, 4 TMA stages.
# - 128x96x64 @ 3 stages: 86,064 B SMEM -> 1 CTA/SM, 3 TMA stages, 0.37% N-pad on 4304.
# - 128x128x64 @ 3 stages: 98,352 B SMEM -> 1 CTA/SM, 3 TMA stages, peak AI (64 FLOP/B).
_HY4_BIASED_PLANS: dict[
    tuple[int, int], tuple[tuple[int | None, int, int, int, int], ...]
] = {
    (768, 1152): (
        (2560, 64, 128, 4, 8),
        (3584, 64, 96, 4, 8),
        (4608, 128, 128, 3, 8),
        (None, 64, 128, 4, 8),
    ),
    (1152, 1152): (
        (2560, 64, 128, 4, 8),
        (3584, 64, 96, 4, 8),
        (4608, 128, 128, 3, 8),
        (None, 64, 128, 4, 8),
    ),
    (4304, 1152): (
        (2560, 64, 128, 4, 8),
        (3584, 64, 96, 4, 8),
        (4608, 128, 128, 3, 8),
        (None, 64, 128, 4, 8),
    ),
    (1152, 4304): (
        (1536, 128, 96, 3, 8),
        (2560, 64, 128, 4, 8),
        (None, 128, 128, 3, 8),
    ),
    (4608, 4608): (
        (600, 64, 128, 4, 8),
        (900, 64, 96, 4, 8),
        (1280, 128, 128, 3, 8),
        (None, 64, 128, 4, 8),
    ),
    (4608, 3072): ((None, 64, 96, 4, 8),),
}


class Hy4BiasedGemm:
    """Warp-specialized TMA BF16 GEMM with fused BF16 bias and L2 tile swizzle.

    Computes ``out[m, n] = bf16(sum_k(f32(x[m, k]) * f32(w[n, k])) + f32(bias[n]))``
    with FP32 tensor-core accumulation and a single BF16 rounding store.
    Supports K and N that are multiples of 16 (including K=4304 and N=4304):
    TMA automatically zero-fills OOB K/N/M lanes in shared memory, and the
    epilogue predicates bias loads and output stores on live ``rows`` and ``N``.
    """

    tile_k = 64
    warp_layout = (2, 2)
    num_compute_warps = 4
    producer_warp = 4
    num_threads = 160
    buffer_align_bytes = 1024

    def __init__(
        self,
        n: int,
        k: int,
        *,
        tile_m: int,
        tile_n: int,
        num_stages: int,
        group_m: int = 8,
    ):
        self.n = int(n)
        self.k = int(k)
        self.tile_m = int(tile_m)
        self.tile_n = int(tile_n)
        self.num_stages = int(num_stages)
        self.group_m = int(group_m)
        if self.n <= 0 or self.k <= 0 or self.n % 16 or self.k % 16:
            raise ValueError("Hy4BiasedGemm requires positive N and K divisible by 16")
        self.k_tiles = (self.k + self.tile_k - 1) // self.tile_k
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.k_unroll = self.num_stages

    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.warp_layout[0], self.warp_layout[1], 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _get_smem_layouts(self) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.tile_k,
            ),
            BFloat16,
        )
        sA_layout = cute.tile_to_shape(
            layout_atom,
            (self.tile_m, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        sB_layout = cute.tile_to_shape(
            layout_atom,
            (self.tile_n, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        return sA_layout, sB_layout

    def _get_shared_storage_cls(
        self,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        # mbar_last avoids 1024-byte alignment padding before sA.
        SharedStorage.__annotations__ = {
            "sA": cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(sA_layout)],
                self.buffer_align_bytes,
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(sB_layout)],
                self.buffer_align_bytes,
            ],
            "mbar_ptr": cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0):
            x_t = cute.make_tensor(
                x, cute.make_layout((rows, self.k), stride=(self.k, 1))
            )
            w_t = cute.make_tensor(
                weight, cute.make_layout((self.n, self.k), stride=(self.k, 1))
            )
            sA_layout, sB_layout = self._get_smem_layouts()
            tiled_mma = self._get_tiled_mma()
            SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
            sA_tma_layout = cute.slice_(sA_layout, (None, None, 0))
            sB_tma_layout = cute.slice_(sB_layout, (None, None, 0))
            tma_atom_A, tma_tensor_A = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                x_t,
                sA_tma_layout,
                (self.tile_m, self.tile_k),
                num_multicast=1,
            )
            tma_atom_B, tma_tensor_B = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                w_t,
                sB_tma_layout,
                (self.tile_n, self.tile_k),
                num_multicast=1,
            )
            grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
            total_tiles = grid_m * Int32(self.n_tiles)
            self.kernel(
                tma_tensor_A,
                tma_tensor_B,
                bias,
                out,
                tma_atom_A,
                tma_atom_B,
                sA_layout,
                sB_layout,
                tiled_mma,
                SharedStorage,
                rows,
                grid_m,
            ).launch(
                grid=(total_tiles, 1, 1),
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Pointer,
        output: cute.Pointer,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        num_tokens: Int32,
        grid_m: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # M-grouped CTA rasterization with serpentine N order across adjacent
        # M-groups: keeps X strip resident across all N-tiles (eliminating L2
        # thrashing at 16,384 rows) and reuses tail W-tiles across groups.
        group_size = Int32(self.group_m)
        tiles_per_group = group_size * Int32(self.n_tiles)
        group_id = pid // tiles_per_group
        first_m = group_id * group_size
        rem_m = grid_m - first_m
        m_in_group = group_size if rem_m > group_size else rem_m
        rem = pid - group_id * tiles_per_group
        n_raw = rem // m_in_group
        m_tile = first_m + (rem - n_raw * m_in_group)
        n_tile = (
            Int32(self.n_tiles - 1) - n_raw
            if (group_id & Int32(1)) != Int32(0)
            else n_raw
        )

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_B)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (
            (self.tile_m + self.tile_n) * self.tile_k * BFloat16.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_compute_warps,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        gA = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None))
        gB = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None))
        cta_layout = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_A,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_B,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_stages,
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
            acc = cute.make_rmem_tensor(acc_shape, Float32)
            acc.fill(0.0)

            smem_copy_atom_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                BFloat16,
            )
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                BFloat16,
            )
            smem_thr_copy_A = cute.make_tiled_copy_A(
                smem_copy_atom_A,
                tiled_mma,
            ).get_slice(tidx)
            smem_thr_copy_B = cute.make_tiled_copy_B(
                smem_copy_atom_B,
                tiled_mma,
            ).get_slice(tidx)
            tSsA = smem_thr_copy_A.partition_S(sA)
            tSsB = smem_thr_copy_B.partition_S(sB)

            target_a = smem_thr_copy_A.retile(tCrA)
            target_b = smem_thr_copy_B.retile(tCrB)
            num_k_blocks = cute.size(tSsA.shape[2])

            # Cross-stage pipelined K mainloop: overlaps next-stage barrier
            # wait and k_block=0 ldmatrix with final MMA of current stage.
            peek = load_pipeline.consumer_try_wait(consumer_state)
            load_pipeline.consumer_wait(consumer_state, peek)
            cute.copy(
                smem_thr_copy_A,
                tSsA[None, None, 0, consumer_state.index],
                target_a[None, None, 0],
            )
            cute.copy(
                smem_thr_copy_B,
                tSsB[None, None, 0, consumer_state.index],
                target_b[None, None, 0],
            )
            if const_expr(self.k_tiles > 1):
                eff_unroll = min(self.k_unroll, self.k_tiles - 1)
                for _k_tile in cutlass.range(self.k_tiles - 1, unroll=eff_unroll):
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        k_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                        if k_block == num_k_blocks - 1:
                            cute.arch.fence_view_async_shared()
                            load_pipeline.consumer_release(consumer_state)
                            consumer_state.advance()
                            peek = load_pipeline.consumer_try_wait(consumer_state)
                            load_pipeline.consumer_wait(consumer_state, peek)
                        cute.copy(
                            smem_thr_copy_A,
                            tSsA[None, None, k_next, consumer_state.index],
                            target_a[None, None, k_next],
                        )
                        cute.copy(
                            smem_thr_copy_B,
                            tSsB[None, None, k_next, consumer_state.index],
                            target_b[None, None, k_next],
                        )
                        cute.gemm(
                            thr_mma,
                            acc,
                            tCrA[None, None, k_block],
                            tCrB[None, None, k_block],
                            acc,
                        )
            for k_block in cutlass.range_constexpr(num_k_blocks):
                if k_block < num_k_blocks - 1:
                    cute.copy(
                        smem_thr_copy_A,
                        tSsA[None, None, k_block + 1, consumer_state.index],
                        target_a[None, None, k_block + 1],
                    )
                    cute.copy(
                        smem_thr_copy_B,
                        tSsB[None, None, k_block + 1, consumer_state.index],
                        target_b[None, None, k_block + 1],
                    )
                else:
                    cute.arch.fence_view_async_shared()
                    load_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()
                cute.gemm(
                    thr_mma,
                    acc,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    acc,
                )

            # Epilogue: hoist BF16x2 bias load per N-slice (reused across all
            # M rows owned by this thread), add in FP32, single BF16x2 store.
            coordinates = thr_mma.partition_C(
                cute.make_identity_tensor((self.tile_m, self.tile_n))
            )
            acc_mn = layout_utils.reshape_acc_to_mn(acc)
            coord_mn = layout_utils.reshape_acc_to_mn(coordinates)
            out_base = Int64(output.toint())
            bias_base = Int64(bias.toint())
            num_m = cute.size(acc_mn.shape[0])
            num_n_pairs = cute.size(acc_mn.shape[1]) // 2
            m_base = m_tile * Int32(self.tile_m)
            n_base = n_tile * Int32(self.tile_n)
            full_m = m_base + Int32(self.tile_m) <= num_tokens
            if full_m:
                for np in cutlass.range_constexpr(num_n_pairs):
                    col = n_base + coord_mn[0, 2 * np][1]
                    if const_expr(self.n % self.tile_n == 0) or col < Int32(self.n):
                        b_word = ld_global_nc_u32(bias_base + Int64(col) * Int64(2))
                        b0 = _bf16_lo(b_word)
                        b1 = _bf16_hi(b_word)
                        for mr in cutlass.range_constexpr(num_m):
                            token = m_base + coord_mn[mr, 0][0]
                            a0 = acc_mn[mr, 2 * np] + b0
                            a1 = acc_mn[mr, 2 * np + 1] + b1
                            at = (
                                Int64(token) * Int64(self.n) + Int64(col)
                            ) * Int64(2)
                            st_global_u32(
                                out_base + at, pack_f32x2_to_bfloat2(a0, a1)
                            )
            else:
                for np in cutlass.range_constexpr(num_n_pairs):
                    col = n_base + coord_mn[0, 2 * np][1]
                    if const_expr(self.n % self.tile_n == 0) or col < Int32(self.n):
                        b_word = ld_global_nc_u32(bias_base + Int64(col) * Int64(2))
                        b0 = _bf16_lo(b_word)
                        b1 = _bf16_hi(b_word)
                        for mr in cutlass.range_constexpr(num_m):
                            token = m_base + coord_mn[mr, 0][0]
                            if token < num_tokens:
                                a0 = acc_mn[mr, 2 * np] + b0
                                a1 = acc_mn[mr, 2 * np + 1] + b1
                                at = (
                                    Int64(token) * Int64(self.n) + Int64(col)
                                ) * Int64(2)
                                st_global_u32(
                                    out_base + at, pack_f32x2_to_bfloat2(a0, a1)
                                )

        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_stages,
            )
            prod_unroll = min(self.k_unroll, self.k_tiles)
            for k_tile in cutlass.range(self.k_tiles, unroll=prod_unroll):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, m_tile, k_tile)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_B,
                    tBgB[(None, n_tile, k_tile)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class Hy4RoutedBiasedProjection:
    """Row-band router over ``Hy4BiasedGemm`` for the 6 Hy4 tower projections."""

    def __init__(self, width: int, outputs: int):
        self.width = int(width)
        self.outputs = int(outputs)
        plan = _HY4_BIASED_PLANS[(self.width, self.outputs)]
        self.limits = tuple(entry[0] for entry in plan if entry[0] is not None)
        self.floors = (0,) + self.limits[:-1]
        # Deduplicate identical tile configs so non-adjacent bands share one cubin.
        unique: dict[tuple[int, int, int, int], Hy4BiasedGemm] = {}
        self.kernels: list[Hy4BiasedGemm] = []
        for _, tm, tn, st, gm in plan:
            cfg = (tm, tn, st, gm)
            if cfg not in unique:
                unique[cfg] = Hy4BiasedGemm(
                    self.outputs,
                    self.width,
                    tile_m=tm,
                    tile_n=tn,
                    num_stages=st,
                    group_m=gm,
                )
            self.kernels.append(unique[cfg])

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(len(self.kernels) == 1):
            self.kernels[0](x, weight, bias, out, rows, stream)
        else:
            for i in cutlass.range_constexpr(len(self.limits)):
                if rows > Int32(self.floors[i]):
                    if rows <= Int32(self.limits[i]):
                        self.kernels[i](x, weight, bias, out, rows, stream)
            if rows > Int32(self.limits[-1]):
                self.kernels[-1](x, weight, bias, out, rows, stream)


# =============================================================================
# 2. Hy4 Dynamic Vision Attention (16x72, segments=8, 8-warp Ping-Pong 128x128)
# =============================================================================

_BAR_WG0 = 5
_BAR_WG1 = 6


def _make_hy4_attn_shared_storage_cls(
    k_numel: int,
    v_numel: int,
    num_stages: int,
    buffer_align_bytes: int = 1024,
):
    """Allocate sK and sV only: sQ (prologue) and sO (epilogue) alias sK[0]."""

    class SharedStorageKV:
        pass

    SharedStorageKV.__annotations__ = {
        "sK": cute.struct.Align[
            cute.struct.MemRange[BFloat16, int(k_numel)],
            buffer_align_bytes,
        ],
        "sV": cute.struct.Align[
            cute.struct.MemRange[BFloat16, int(v_numel)],
            buffer_align_bytes,
        ],
        "mbar_ptr": cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 16],
        "mbar_ptr_K": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, num_stages * 2], 16
        ],
        "mbar_ptr_V": cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, num_stages * 2], 16
        ],
    }
    return cute.struct(SharedStorageKV)


@cute.jit
def _warp_mma_gemm_qk_reg_a(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
):
    """Q-in-regs QK GEMM pipelining 2 ldmatrix.x4 (4 MMA N-tiles) at a time.

    Keeps peak tCrB live registers at 16 (vs 64 for full-k double buffering)
    while overlapping 2 ldmatrix.x4 with 4 m16n8k16 HMMAs, so the 8-compute-warp
    128x128 kernel compiles spill-free at REG:168 with zero setmaxnreg.
    """
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    nk = cute.size(tCsB.shape[2])
    n_copy = cute.size(tCsB.shape[1])
    tCrB_g = cute.logical_divide(tCrB, (None, 4, None))
    acc_g = cute.logical_divide(acc, (None, None, 4))
    cute.copy(smem_thr_copy_B, tCsB[None, 0, 0], tCrB_copy_view[None, 0, 0])
    cute.copy(smem_thr_copy_B, tCsB[None, 1, 0], tCrB_copy_view[None, 1, 0])
    for k in cutlass.range_constexpr(nk):
        for cg in cutlass.range_constexpr(n_copy // 2):
            if const_expr(cg + 1 < n_copy // 2):
                nc0 = 2 * (cg + 1)
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, nc0, k],
                    tCrB_copy_view[None, nc0, k],
                )
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, nc0 + 1, k],
                    tCrB_copy_view[None, nc0 + 1, k],
                )
            elif const_expr(k + 1 < nk):
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, 0, k + 1],
                    tCrB_copy_view[None, 0, k + 1],
                )
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, 1, k + 1],
                    tCrB_copy_view[None, 1, k + 1],
                )
            cute.gemm(
                tiled_mma,
                acc_g[None, None, (None, cg)],
                tCrA[None, None, k],
                tCrB_g[None, (None, cg), k],
                acc_g[None, None, (None, cg)],
            )


@cute.jit
def _warp_mma_gemm_pv(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    tCcB: Optional[cute.Tensor] = None,
    valid_k: Optional[Int32] = None,
):
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )
        if const_expr(tCcB is not None):
            fragment = tCrB[None, None, k]
            coordinates = tCcB[None, None, k]
            for element in cutlass.range_constexpr(cute.size(fragment)):
                if coordinates[element][1] >= valid_k:
                    fragment[element] = tCrB.element_type(0.0)
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def _pp_wait(wg: Int32, num_compute_threads: cutlass.Constexpr):
    if wg == Int32(0):
        cute.arch.barrier(barrier_id=_BAR_WG0, number_of_threads=num_compute_threads)
    else:
        cute.arch.barrier(barrier_id=_BAR_WG1, number_of_threads=num_compute_threads)


@cute.jit
def _pp_arrive_other(wg: Int32, num_compute_threads: cutlass.Constexpr):
    if wg == Int32(0):
        cute.arch.barrier_arrive(
            barrier_id=_BAR_WG1, number_of_threads=num_compute_threads
        )
    else:
        cute.arch.barrier_arrive(
            barrier_id=_BAR_WG0, number_of_threads=num_compute_threads
        )


class Hy4DynamicVisionAttention:
    """8-compute-warp ping-pong 128x128 2-stage TMA attention for Hy4 (16x72, 8 segs).

    Architecture:
    - 8 compute warps split into WG0 (warps 0..3) and WG1 (warps 4..7), plus
      1 TMA producer warp (warp 8) = 288 threads total (2 compute warps per SM
      subpartition).
    - Ping-pong scheduling overlaps WG0's [PV(step) + QK(step+1)] HMMA burst
      with WG1's [Softmax + rescale_O + BF16 cvt] SFU/ALU burst and vice versa,
      eliminating the SFU (MUFU.EX2) bottleneck of single-warpgroup attention.
    - tile_m=128, tile_n=128 halves softmax rescalings, warp reductions, and
      barriers per token relative to tile_n=64.
    - Q is loaded into registers once in prologue (20 regs/thread) from sQ
      aliased onto sK[0], so 2 full 128x80 stages of K and V fit in 82 KiB SMEM
      (<= 101 KiB on SM120/SM121). sO also aliases sK[0] in epilogue.
    - N-pipelined QK (2 ldmatrix.x4 / 4 HMMAs per step) keeps peak register
      pressure within the native 168-reg/thread SM120/SM121 9-warp ceiling with
      zero register spills (STACK:0, STL=0, LDL=0) and zero setmaxnreg.
    - K-tail masking and NaN-safe V-tail zeroing run only on n_block_max - 1;
      all interior KV blocks have zero masking and check_inf=False.
    """

    heads = 16
    dim = 72
    segments = 8
    tile_hdim = 80
    tile_m = 128
    tile_n = 128
    num_stages = 2
    num_compute_warps = 8
    producer_warp_idx = 8
    num_threads = 288
    num_mma_threads = 256
    num_epilogue_threads = 256
    buffer_align_bytes = 1024

    def __init__(self):
        pass

    def _get_smem_layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, BFloat16, self.tile_hdim
            ),
            BFloat16,
        )
        sQ_layout = cute.tile_to_shape(atom, (self.tile_m, self.tile_hdim), (0, 1))
        sK_layout = cute.tile_to_shape(
            atom, (self.tile_n, self.tile_hdim, self.num_stages), (0, 1, 2)
        )
        sV_layout = cute.tile_to_shape(
            atom, (self.tile_n, self.tile_hdim, self.num_stages), (0, 1, 2)
        )
        sO_layout = sQ_layout
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            BFloat16,
            num_bits_per_copy=128,
        )
        tV_shape_dim_1 = atom.outer.shape[1] // 8
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, cute.make_layout((1, 8))
        )
        return sQ_layout, sK_layout, sV_layout, sO_layout, gmem_tiled_copy_O

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        out: cute.Pointer,
        lse: cute.Pointer,
        lengths: cute.Pointer,
        rows: Int32,
        scale: Float32,
        l2_bytes: Int32,
        stream: cuda.CUstream,
    ):
        del l2_bytes
        if rows > Int32(0):
            shape = (rows, self.heads, self.dim)
            stride = (self.heads * self.dim, self.dim, 1)
            mQ, mK, mV, mO = [
                assume_tensor_aligned(
                    cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))
                )
                for ptr in (q, k, v, out)
            ]
            mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[0, 2, 1]))
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[0, 2, 1]))
            mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[0, 2, 1]))
            mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[0, 2, 1]))
            mLSE = cute.make_tensor(
                lse, cute.make_layout((rows, self.heads), stride=(1, rows))
            )
            mCu = cute.make_tensor(
                lengths, cute.make_layout((self.segments + 1,), stride=(1,))
            )

            sQ_layout, sK_layout, sV_layout, sO_layout, gmem_tiled_copy_O = (
                self._get_smem_layouts()
            )
            tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
            SharedStorage = _make_hy4_attn_shared_storage_cls(
                k_numel=cute.cosize(sK_layout),
                v_numel=cute.cosize(sV_layout),
                num_stages=self.num_stages,
                buffer_align_bytes=self.buffer_align_bytes,
            )

            sK_tma_layout = cute.select(sK_layout, mode=[0, 1])
            sV_tma_layout = cute.select(sV_layout, mode=[0, 1])
            tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mQ,
                sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mK,
                sK_tma_layout,
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mV,
                sV_tma_layout,
                (self.tile_n, self.tile_hdim),
                1,
            )
            softmax_scale_log2, softmax_scale = cute_ops.compute_softmax_scale_log2(scale)
            total_blocks_max = (
                rows + Int32(self.segments * (self.tile_m - 1))
            ) // Int32(self.tile_m)
            grid_dim = (total_blocks_max * Int32(self.heads), Int32(1), Int32(1))

            self.kernel(
                tma_tensor_Q,
                tma_tensor_K,
                tma_tensor_V,
                mO,
                mLSE,
                mCu,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                softmax_scale_log2,
                softmax_scale,
                sQ_layout,
                sK_layout,
                sV_layout,
                sO_layout,
                gmem_tiled_copy_O,
                tiled_mma_qk,
                tiled_mma_pv,
                SharedStorage,
            ).launch(
                grid=grid_dim,
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.jit
    def _resolve_tile(self, mCu: cute.Tensor, tile_idx: Int32):
        """Single-pass warp prefix-sum over the 8 segments -> (m_block, head_idx, batch_idx, valid)."""
        lane = cute.arch.lane_idx()
        cur_cu = Int32(0)
        if lane <= Int32(self.segments):
            cur_cu = mCu[lane]
        next_cu = cute.arch.shuffle_sync_down(cur_cu, offset=1)
        seqlen = next_cu - cur_cu
        num_m = (
            (seqlen + Int32(self.tile_m - 1)) // Int32(self.tile_m)
            if lane < Int32(self.segments)
            else Int32(0)
        )
        cum_m = cute_ops.warp_prefix_sum(num_m, lane)
        total_m = cute.arch.shuffle_sync(cum_m, 31)
        total_tiles = total_m * Int32(self.heads)
        is_valid = tile_idx < total_tiles
        m_block = Int32(0)
        head_idx = Int32(0)
        batch_idx = Int32(self.segments)
        if is_valid:
            batch_idx = cute.arch.popc(
                cute.arch.vote_ballot_sync(cum_m * Int32(self.heads) <= tile_idx)
            )
            prev_cum = (
                Int32(0)
                if batch_idx == Int32(0)
                else cute.arch.shuffle_sync(cum_m, batch_idx - Int32(1))
            )
            batch_m = cute.arch.shuffle_sync(num_m, batch_idx)
            mh_block = tile_idx - prev_cum * Int32(self.heads)
            head_idx = mh_block // batch_m
            m_block = mh_block - head_idx * batch_m
        return m_block, head_idx, batch_idx, is_valid

    @cute.jit
    def _mask_k_tail(
        self,
        acc_S: cute.Tensor,
        n_block: Int32,
        seqlen_k: Int32,
        thr_mma_qk: cute.TiledMma,
    ):
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma_qk.get_slice(0).partition_C(cS)
        )
        thr_col_offset = tScS_mn[0][1]
        col_limit = seqlen_k - n_block * Int32(self.tile_n)
        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
            col_in_tile = t0ScS_mn[0, c][1] + thr_col_offset
            if col_in_tile >= col_limit:
                for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                    acc_S_mn[r, c] = -Float32.inf

    @cute.jit
    def _epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        offset_q: Int32,
        seqlen_q: Int32,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
    ):
        rO = cute.make_fragment_like(acc_O, BFloat16)
        rO.store(acc_O.load().to(BFloat16))
        # Ensure all 8 compute warps finished reading sK/sV before aliasing sO onto sK[0].
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            BFloat16,
            num_bits_per_copy=2 * BFloat16.width,
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(
            tidx
        )
        cute.copy(
            smem_copy_atom_O,
            smem_thr_copy_O.retile(rO),
            smem_thr_copy_O.partition_D(sO),
        )

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        mLSE_cur = cute.domain_offset((offset_q,), mLSE[None, head_idx])
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        gLSE_expanded = cute.make_tensor(
            gLSE.iterator,
            cute.append(gLSE.layout, cute.make_layout((self.tile_hdim,), stride=(0,))),
        )
        thr_mma = tiled_mma.get_slice(tidx)
        taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
        if taccOcO[0][1] == 0:
            for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                if t0accOcO[m, 0][0] < seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
                    taccOgLSE[m, 0] = lse[m]

        mO_cur = cute.domain_offset((offset_q, 0), mO[None, None, head_idx])
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, BFloat16)
        cute.autovec_copy(tOsO, tOrO)
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = cute_ops.predicate_k(tOcO, limit=Int32(self.dim))
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mCu: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tile_idx, _, _ = cute.arch.block_idx()
        m_block, head_idx, batch_idx, is_valid = self._resolve_tile(mCu, tile_idx)
        if is_valid:
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)

            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            mbar_ptr_Q = storage.mbar_ptr.data_ptr()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(mbar_ptr_Q, 1)
            cute.arch.sync_threads()

            q_bytes = self.tile_m * self.tile_hdim * 2
            kv_bytes = self.tile_n * self.tile_hdim * 2
            pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, self.num_compute_warps
            )
            pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            )
            pipeline_k = cute_pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline_kv_producer_group,
                consumer_group=pipeline_kv_consumer_group,
                tx_count=kv_bytes,
                defer_sync=True,
            )
            pipeline_v = cute_pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline_kv_producer_group,
                consumer_group=pipeline_kv_consumer_group,
                tx_count=kv_bytes,
                defer_sync=False,
            )

            # sQ (prologue) and sO (epilogue) alias the first 20 KiB of sK.
            sQ = storage.sK.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            sVt = layout_utils.transpose_view(sV)
            sO = storage.sK.get_tensor(
                sO_layout.outer, swizzle=sO_layout.inner, dtype=BFloat16
            )

            offset_q = mCu[batch_idx]
            seqlen = mCu[batch_idx + Int32(1)] - offset_q
            n_block_max = (seqlen + Int32(self.tile_n - 1)) // Int32(self.tile_n)

            if warp_idx == Int32(self.producer_warp_idx):
                kv_producer_state = cute_pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Producer, self.num_stages
                )
                mQ_cur = cute.domain_offset((offset_q, 0, 0), mQ)[None, None, head_idx]
                gQ = cute.local_tile(
                    mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0)
                )
                load_Q, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_Q, q_bytes)
                load_Q(tma_bar_ptr=mbar_ptr_Q)

                # Wait until all 8 compute warps finish loading Q -> tSrQ
                # before overwriting sK[0] with K tiles.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_threads,
                )

                mK_cur = cute.domain_offset((offset_q, 0, 0), mK)[None, None, head_idx]
                mV_cur = cute.domain_offset((offset_q, 0, 0), mV)[None, None, head_idx]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdim), (None, 0))
                load_K, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK
                )
                load_K = cute_copy.tma_producer_copy_fn(load_K, pipeline_k)
                load_V, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV
                )
                load_V = cute_copy.tma_producer_copy_fn(load_V, pipeline_v)

                for n_tile in cutlass.range(n_block_max, unroll=1):
                    n_block = n_block_max - Int32(1) - n_tile
                    pipeline_k.producer_acquire(kv_producer_state)
                    load_K(src_idx=n_block, producer_state=kv_producer_state)
                    pipeline_v.producer_acquire(kv_producer_state)
                    load_V(src_idx=n_block, producer_state=kv_producer_state)
                    kv_producer_state.advance()

                pipeline_k.producer_tail(kv_producer_state.clone())
                pipeline_v.producer_tail(kv_producer_state.clone())

            elif warp_idx < Int32(self.num_compute_warps):
                tidx = cute.arch.thread_idx()[0]
                wg = warp_idx >> Int32(2)  # 0 = warps 0..3, 1 = warps 4..7

                thr_mma_qk = tiled_mma_qk.get_slice(tidx)
                thr_mma_pv = tiled_mma_pv.get_slice(tidx)
                tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
                tSrK = thr_mma_qk.make_fragment_B(
                    thr_mma_qk.partition_B(sK[None, None, 0])
                )
                tOrVt = thr_mma_pv.make_fragment_B(
                    thr_mma_pv.partition_B(sVt[None, None, 0])
                )
                acc_shape_O = thr_mma_pv.partition_shape_C(
                    (self.tile_m, self.tile_hdim)
                )
                acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
                acc_O.fill(0.0)
                acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))

                smem_copy_atom_QK = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    BFloat16,
                )
                smem_copy_atom_V = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                    BFloat16,
                )
                smem_thr_copy_Q = cute_ops.make_tiled_copy_A(
                    smem_copy_atom_QK, tiled_mma_qk
                ).get_slice(tidx)
                smem_thr_copy_K = cute_ops.make_tiled_copy_B(
                    smem_copy_atom_QK, tiled_mma_qk
                ).get_slice(tidx)
                smem_thr_copy_V = cute_ops.make_tiled_copy_B(
                    smem_copy_atom_V, tiled_mma_pv
                ).get_slice(tidx)
                tSsQ = smem_thr_copy_Q.partition_S(sQ)
                tSsK = smem_thr_copy_K.partition_S(sK)
                tOsVt = smem_thr_copy_V.partition_S(sVt)

                softmax_num_rows = acc_O.shape[0][0] * acc_O.shape[1]
                softmax = Softmax.create(
                    softmax_scale_log2,
                    num_rows=softmax_num_rows,
                    softmax_scale=softmax_scale,
                )
                softmax.reset()

                # Load Q from sQ into registers once, then release sK[0] to producer.
                cute.arch.mbarrier_wait(mbar_ptr_Q, phase=Int32(0))
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                for k_step in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    cute.copy(
                        smem_thr_copy_Q,
                        tSsQ[None, None, k_step],
                        tSrQ_copy_view[None, None, k_step],
                    )
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_threads,
                )

                k_state = cute_pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
                )
                v_state = cute_pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
                )
                rP = cute.make_rmem_tensor(acc_shape_S, BFloat16)
                tOrP = layout_utils.reshape_acc_to_frgA(rP)
                tCcV_tail = thr_mma_pv.partition_B(
                    cute.make_identity_tensor((self.tile_hdim, self.tile_n))
                )
                first_valid_k = seqlen - (n_block_max - Int32(1)) * Int32(self.tile_n)

                if n_block_max > Int32(0):
                    # Seed ping-pong: WG1 grants WG0 initial HMMA turn.
                    if wg == Int32(1):
                        cute.arch.barrier_arrive(
                            barrier_id=_BAR_WG0, number_of_threads=self.num_mma_threads
                        )

                    # --- Tile 0: QK[0] (HMMA) -> release other WG -> Mask + Softmax[0] (SFU) ---
                    _pp_wait(wg, self.num_mma_threads)
                    pipeline_k.consumer_wait(
                        k_state, pipeline_k.consumer_try_wait(k_state)
                    )
                    acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
                    acc_S.fill(0.0)
                    _warp_mma_gemm_qk_reg_a(
                        thr_mma_qk,
                        acc_S,
                        tSrQ,
                        tSrK,
                        tSsK[None, None, None, k_state.index],
                        smem_thr_copy_K,
                    )
                    cute.arch.fence_view_async_shared()
                    pipeline_k.consumer_release(k_state)
                    k_state.advance()
                    _pp_arrive_other(wg, self.num_mma_threads)

                    self._mask_k_tail(
                        acc_S, n_block_max - Int32(1), seqlen, thr_mma_qk
                    )
                    softmax.online_softmax(acc_S, is_first=True, check_inf=True)
                    rP.store(acc_S.load().to(BFloat16))

                    if n_block_max == Int32(1):
                        # Single KV block: run PV[0] with V-tail zeroing.
                        _pp_wait(wg, self.num_mma_threads)
                        pipeline_v.consumer_wait(
                            v_state, pipeline_v.consumer_try_wait(v_state)
                        )
                        _warp_mma_gemm_pv(
                            thr_mma_pv,
                            acc_O,
                            tOrP,
                            tOrVt,
                            tOsVt[None, None, None, v_state.index],
                            smem_thr_copy_V,
                            tCcB=tCcV_tail,
                            valid_k=first_valid_k,
                        )
                        cute.arch.fence_view_async_shared()
                        pipeline_v.consumer_release(v_state)
                        v_state.advance()
                        if wg == Int32(0):
                            cute.arch.barrier_arrive(
                                barrier_id=_BAR_WG1,
                                number_of_threads=self.num_mma_threads,
                            )
                    else:
                        # --- Peeled step 0: PV[0] (with V-tail zeroing) + QK[1] -> Softmax[1] ---
                        _pp_wait(wg, self.num_mma_threads)
                        pipeline_v.consumer_wait(
                            v_state, pipeline_v.consumer_try_wait(v_state)
                        )
                        _warp_mma_gemm_pv(
                            thr_mma_pv,
                            acc_O,
                            tOrP,
                            tOrVt,
                            tOsVt[None, None, None, v_state.index],
                            smem_thr_copy_V,
                            tCcB=tCcV_tail,
                            valid_k=first_valid_k,
                        )
                        cute.arch.fence_view_async_shared()
                        pipeline_v.consumer_release(v_state)
                        v_state.advance()

                        pipeline_k.consumer_wait(
                            k_state, pipeline_k.consumer_try_wait(k_state)
                        )
                        acc_S1 = cute.make_rmem_tensor(acc_shape_S, Float32)
                        acc_S1.fill(0.0)
                        _warp_mma_gemm_qk_reg_a(
                            thr_mma_qk,
                            acc_S1,
                            tSrQ,
                            tSrK,
                            tSsK[None, None, None, k_state.index],
                            smem_thr_copy_K,
                        )
                        cute.arch.fence_view_async_shared()
                        pipeline_k.consumer_release(k_state)
                        k_state.advance()
                        _pp_arrive_other(wg, self.num_mma_threads)

                        row_scale1 = softmax.online_softmax(
                            acc_S1, is_first=False, check_inf=False
                        )
                        softmax.rescale_O(acc_O, row_scale1)
                        rP.store(acc_S1.load().to(BFloat16))

                        # --- Steady-state loop: PV[s] + QK[s+1] || Softmax[s+1] ---
                        for _step in cutlass.range(n_block_max - Int32(2), unroll=1):
                            _pp_wait(wg, self.num_mma_threads)
                            pipeline_v.consumer_wait(
                                v_state, pipeline_v.consumer_try_wait(v_state)
                            )
                            _warp_mma_gemm_pv(
                                thr_mma_pv,
                                acc_O,
                                tOrP,
                                tOrVt,
                                tOsVt[None, None, None, v_state.index],
                                smem_thr_copy_V,
                            )
                            cute.arch.fence_view_async_shared()
                            pipeline_v.consumer_release(v_state)
                            v_state.advance()

                            pipeline_k.consumer_wait(
                                k_state, pipeline_k.consumer_try_wait(k_state)
                            )
                            acc_Si = cute.make_rmem_tensor(acc_shape_S, Float32)
                            acc_Si.fill(0.0)
                            _warp_mma_gemm_qk_reg_a(
                                thr_mma_qk,
                                acc_Si,
                                tSrQ,
                                tSrK,
                                tSsK[None, None, None, k_state.index],
                                smem_thr_copy_K,
                            )
                            cute.arch.fence_view_async_shared()
                            pipeline_k.consumer_release(k_state)
                            k_state.advance()
                            _pp_arrive_other(wg, self.num_mma_threads)

                            row_scale_i = softmax.online_softmax(
                                acc_Si, is_first=False, check_inf=False
                            )
                            softmax.rescale_O(acc_O, row_scale_i)
                            rP.store(acc_Si.load().to(BFloat16))

                        # --- Final V: PV[last] (interior block, no V-tail zeroing) ---
                        _pp_wait(wg, self.num_mma_threads)
                        pipeline_v.consumer_wait(
                            v_state, pipeline_v.consumer_try_wait(v_state)
                        )
                        _warp_mma_gemm_pv(
                            thr_mma_pv,
                            acc_O,
                            tOrP,
                            tOrVt,
                            tOsVt[None, None, None, v_state.index],
                            smem_thr_copy_V,
                        )
                        cute.arch.fence_view_async_shared()
                        pipeline_v.consumer_release(v_state)
                        v_state.advance()
                        if wg == Int32(0):
                            cute.arch.barrier_arrive(
                                barrier_id=_BAR_WG1,
                                number_of_threads=self.num_mma_threads,
                            )

                row_scale = softmax.finalize(sink_val=None)
                softmax.rescale_O(acc_O, row_scale)
                self._epilogue(
                    acc_O,
                    softmax.row_sum,
                    mO,
                    mLSE,
                    sO,
                    offset_q,
                    seqlen,
                    gmem_tiled_copy_O,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                )


# =============================================================================
# 3. Hy4 Vectorized Exact erf GELU (width=4304, 4608, ILP=2)
# =============================================================================

HY4_GELU_WIDTHS = frozenset((4304, 4608))


@cute.jit
def _erff_branchless(value: Float32) -> Float32:
    """Branchless CUDA 13.0 libdevice __nv_erff polynomial (0-ULP on all BF16)."""
    absolute = cute.math.abs(value)
    large = absolute >= Float32(1.002959966659546)
    argument = cutlass.select_(large, absolute, _mul(value, value))
    c0 = cutlass.select_(
        large, Float32(0.00011219871521461755), Float32(8.483494457323104e-05)
    )
    c1 = cutlass.select_(
        large, Float32(-0.0013275252422317863), Float32(-0.0008213091641664505)
    )
    c2 = cutlass.select_(
        large, Float32(0.00839653518050909), Float32(0.005213488824665546)
    )
    c3 = cutlass.select_(
        large, Float32(-0.04024658352136612), Float32(-0.026868773624300957)
    )
    c4 = cutlass.select_(
        large, Float32(0.15950430929660797), Float32(0.11284004896879196)
    )
    c5 = cutlass.select_(
        large, Float32(0.9129176735877991), Float32(-0.37612664699554443)
    )
    c6 = cutlass.select_(
        large, Float32(0.6290600299835205), Float32(0.12837915122509003)
    )
    poly = _fma(c0, argument, c1)
    poly = _fma(poly, argument, c2)
    poly = _fma(poly, argument, c3)
    poly = _fma(poly, argument, c4)
    poly = _fma(poly, argument, c5)
    poly = _fma(poly, argument, c6)
    factor = cutlass.select_(large, -absolute, value)
    poly_res = _fma(poly, factor, factor)
    large_res = cute.math.copysign(_sub(Float32(1.0), _ex2_ftz(poly_res)), value)
    return cutlass.select_(large, large_res, poly_res)


@cute.jit
def _gelu_f32(value: Float32) -> Float32:
    argument = _mul(value, Float32(0.7071067811865476))
    activation = _erff_branchless(argument)
    scaled = _mul(value, Float32(0.5))
    return _mul(scaled, _add(Float32(1.0), activation))


class Hy4ExactGelu:
    """ILP=2 128-bit vectorized branchless exact erf GELU for 8-aligned widths.

    Each 128-thread CTA handles 256 16-byte vectors (2 coalesced vec8 loads
    issued upfront per thread), maximizing memory-level parallelism without
    reducing CTA count on small merged row counts.
    """

    def __init__(self, width: int):
        self.width = int(width)
        if self.width <= 0 or self.width % 8:
            raise ValueError("Hy4ExactGelu requires positive width divisible by 8")
        self.vecs_per_row = self.width // 8

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        if rows > Int32(0):
            total_vecs = Int64(rows) * Int64(self.vecs_per_row)
            grid_x = (total_vecs + Int64(255)) // Int64(256)
            self.kernel(x, out, total_vecs).launch(
                grid=(grid_x, 1, 1), block=(128, 1, 1), stream=stream
            )

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, total_vecs: Int64):
        base_vec = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(
            cute.arch.thread_idx()[0]
        )
        vec1 = base_vec + Int64(128)
        x_addr = Int64(x.toint())
        out_addr = Int64(out.toint())
        if vec1 < total_vecs:
            off0 = base_vec * Int64(16)
            off1 = vec1 * Int64(16)
            a0, a1, a2, a3 = ld_global_nc_v4_u32(x_addr + off0)
            b0, b1, b2, b3 = ld_global_nc_v4_u32(x_addr + off1)
            oa0 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a0)), _gelu_f32(_bf16_hi(a0))
            )
            oa1 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a1)), _gelu_f32(_bf16_hi(a1))
            )
            oa2 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a2)), _gelu_f32(_bf16_hi(a2))
            )
            oa3 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a3)), _gelu_f32(_bf16_hi(a3))
            )
            st_global_v4_u32(out_addr + off0, oa0, oa1, oa2, oa3)
            ob0 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(b0)), _gelu_f32(_bf16_hi(b0))
            )
            ob1 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(b1)), _gelu_f32(_bf16_hi(b1))
            )
            ob2 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(b2)), _gelu_f32(_bf16_hi(b2))
            )
            ob3 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(b3)), _gelu_f32(_bf16_hi(b3))
            )
            st_global_v4_u32(out_addr + off1, ob0, ob1, ob2, ob3)
        elif base_vec < total_vecs:
            off0 = base_vec * Int64(16)
            a0, a1, a2, a3 = ld_global_nc_v4_u32(x_addr + off0)
            oa0 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a0)), _gelu_f32(_bf16_hi(a0))
            )
            oa1 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a1)), _gelu_f32(_bf16_hi(a1))
            )
            oa2 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a2)), _gelu_f32(_bf16_hi(a2))
            )
            oa3 = pack_f32x2_to_bfloat2(
                _gelu_f32(_bf16_lo(a3)), _gelu_f32(_bf16_hi(a3))
            )
            st_global_v4_u32(out_addr + off0, oa0, oa1, oa2, oa3)


# =============================================================================
# 4. Hy4 Vectorized Scaled Bilinear Position (1152 ch, 128x128, ATen FMA)
# =============================================================================


class Hy4ScaledBilinearPosition:
    """Bit-exact ATen bilinear position resize for 1152 channels, 128x128 table.

    Launches a 2D grid ``(width, height, 1)`` with 160 threads per CTA
    (144 active lanes x 8 BF16 = 1152 channels), eliminating all 64-bit integer
    divisions, sharing warp-uniform interpolation weights across all channels
    of a pixel, and issuing 128-bit coalesced loads and stores.
    """

    channels = 1152
    table_height = 128
    table_width = 128
    vecs_per_pixel = 144  # 1152 / 8
    block_threads = 160

    def __init__(self):
        pass

    @cute.jit
    def __call__(
        self,
        table: cute.Pointer,
        out: cute.Pointer,
        height: Int32,
        width: Int32,
        scale_h: Float32,
        scale_w: Float32,
        stream: cuda.CUstream,
    ):
        if height > Int32(0) and width > Int32(0):
            self.kernel(table, out, height, width, scale_h, scale_w).launch(
                grid=(width, height, 1),
                block=(self.block_threads, 1, 1),
                stream=stream,
            )

    @cute.jit
    def _source(self, rscale: Float32, dst: Int64, length: int):
        src = _fma(rscale, _add(Float32(dst), Float32(0.5)), Float32(-0.5))
        src = cute.math.max(src, Float32(0.0))
        low = Int64(src)
        step = cute.math.min(low + Int64(1), Int64(length - 1)) - low
        lam1 = _sub(src, Float32(low))
        lam0 = _sub(Float32(1.0), lam1)
        return low, step, lam0, lam1

    @cute.jit
    def _interp_pair(
        self,
        wa: Uint32,
        wb: Uint32,
        wc: Uint32,
        wd: Uint32,
        w0: Float32,
        w1: Float32,
        h0: Float32,
        h1: Float32,
    ) -> Uint32:
        top_lo = _fma(w0, _bf16_lo(wa), _mul(w1, _bf16_lo(wb)))
        bot_lo = _fma(w0, _bf16_lo(wc), _mul(w1, _bf16_lo(wd)))
        val_lo = _fma(h0, top_lo, _mul(h1, bot_lo))
        top_hi = _fma(w0, _bf16_hi(wa), _mul(w1, _bf16_hi(wb)))
        bot_hi = _fma(w0, _bf16_hi(wc), _mul(w1, _bf16_hi(wd)))
        val_hi = _fma(h0, top_hi, _mul(h1, bot_hi))
        return pack_f32x2_to_bfloat2(val_lo, val_hi)

    @cute.kernel
    def kernel(
        self,
        table: cute.Pointer,
        out: cute.Pointer,
        height: Int32,
        width: Int32,
        scale_h: Float32,
        scale_w: Float32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        if tid < Int32(self.vecs_per_pixel):
            x = Int64(cute.arch.block_idx()[0])
            y = Int64(cute.arch.block_idx()[1])
            chan_bytes = Int64(tid) * Int64(16)
            out_pixel_bytes = (y * Int64(width) + x) * Int64(self.channels * 2)
            out_addr = Int64(out.toint()) + out_pixel_bytes + chan_bytes
            table_base = Int64(table.toint())
            if height == Int32(self.table_height) and width == Int32(self.table_width):
                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                    table_base + out_pixel_bytes + chan_bytes
                )
                st_global_v4_u32(out_addr, v0, v1, v2, v3)
            else:
                y0, dy, h0, h1 = self._source(scale_h, y, self.table_height)
                x0, dx, w0, w1 = self._source(scale_w, x, self.table_width)
                row_bytes = Int64(self.table_width * self.channels * 2)
                pixel_bytes = Int64(self.channels * 2)
                base_addr = (
                    table_base
                    + (y0 * Int64(self.table_width) + x0) * pixel_bytes
                    + chan_bytes
                )
                dx_bytes = dx * pixel_bytes
                dy_bytes = dy * row_bytes
                a0, a1, a2, a3 = ld_global_nc_v4_u32(base_addr)
                b0, b1, b2, b3 = ld_global_nc_v4_u32(base_addr + dx_bytes)
                c0, c1, c2, c3 = ld_global_nc_v4_u32(base_addr + dy_bytes)
                d0, d1, d2, d3 = ld_global_nc_v4_u32(base_addr + dy_bytes + dx_bytes)
                o0 = self._interp_pair(a0, b0, c0, d0, w0, w1, h0, h1)
                o1 = self._interp_pair(a1, b1, c1, d1, w0, w1, h0, h1)
                o2 = self._interp_pair(a2, b2, c2, d2, w0, w1, h0, h1)
                o3 = self._interp_pair(a3, b3, c3, d3, w0, w1, h0, h1)
                st_global_v4_u32(out_addr, o0, o1, o2, o3)
