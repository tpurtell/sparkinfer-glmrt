"""Inkling 2 Mini dedicated small-row BF16 projection programs (M=1..8).

Dedicated Mini-only programs that accelerate attention-input and router
projections at small decode/verify widths (rows 1..8) while preserving stock
``TmaBf16Projection`` above 8 rows via an exact scalar route switch (zero host
sync, CUDA-graph capturable):

* ``compile_inkling2_mini_attn_in_proj_local_aot``: fuses all 6 local
  attention-input projections sharing ``(xq, xk, xv)`` — Q [3072 BF16],
  R [384 BF16], Gate [3072 BF16], Sinks [24 F32], K [1024 BF16], V [1024 BF16]
  (total N=8600, 39.63 MB) — into ONE TMA + ``m16n8k16`` launch. Preserves
  exact ordered-K16 MMA accumulation, so outputs are bit-for-bit identical to
  stock across all rows 1..4096.
* ``compile_inkling2_mini_attn_in_proj_global_aot``: fuses all 8 global
  attention-input projections (local + k_dyn [32 F32], v_dyn [32 F32]; total
  N=8664, 39.92 MB) into ONE TMA + ``m16n8k16`` launch (bit-exact to stock
  across all rows 1..4096).
* ``compile_inkling2_mini_router_proj_aot``: single-launch block-K BF16 GEMV
  for ``mlp.gate`` (2304 -> 128 F32, 0 scratch, 288 threads/CTA) for rows 1..8
  (bit-exact row-invariant across 1..8 vs M=1), switching to stock
  ``TmaBf16Projection`` above 8 rows.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import pack_f32x2_to_bfloat2, st_global_u32
from b12x.gemm.bf16_gemv._prefill import _assume_tma_source_aligned, _warp_gemm_direct
from b12x.gemm.bf16_gemv._skinny import (
    TmaBf16Projection,
    _bf16_hi,
    _bf16_lo,
    _ld_cached,
    _ld_stream,
)
from ._common import Operand, Scalar, compile_program

SMALL_ROW_MAX = 8
STOCK_TILES = ((16, 1, 4, 16, 64, 6),)
ATTN_IN_TMA_CFG = dict(warp_n=4, tile_n=32, num_stages=10, tile_k=64)
ROUTER_BLOCK_CFG = dict(cols=1, threads=288)


@cute.jit
def _warp_reduce_sum(value: Float32) -> Float32:
    v = value
    for shift in cutlass.range_constexpr(5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << shift)
    return v


@cute.jit
def _store_scalar(out_ptr: cute.Pointer, row: Int32, col: Int64, n_cols: Int64,
                  value: Float32, out_is_f32: cutlass.Constexpr):
    if const_expr(out_is_f32):
        addr = Int64(out_ptr.toint()) + (Int64(row) * n_cols + col) * Int64(4)
        ptr = cute.make_ptr(Float32, addr, cute.AddressSpace.gmem, assumed_align=4)
        ptr[0] = value
    else:
        addr = Int64(out_ptr.toint()) + (Int64(row) * n_cols + col) * Int64(2)
        ptr = cute.make_ptr(BFloat16, addr, cute.AddressSpace.gmem, assumed_align=2)
        ptr[0] = value.to(BFloat16)


class FusedAttnInTmaKernel:
    """Single-launch fused TMA + m16n8k16 attention-input projection for M <= 16.

    Executes the exact ordered-K16 uncompensated MMA accumulation of stock
    ``Bf16PrefillKernel`` across all 6 local (Q, Gate, K, V, R, Sinks) or
    8 global (+ k_dyn, v_dyn) outputs in one kernel launch -> 100% bit-exact
    to stock and flat latency across M = 1..16.
    """

    K = 2304
    N_Q = 3072
    N_GATE = 3072
    N_K = 1024
    N_V = 1024
    N_R = 384
    N_SINK = 24
    N_DYN = 32
    buffer_align_bytes = 1024

    def __init__(self, *, is_global: bool = False, warp_n: int = 4,
                 tile_n: int = 32, num_stages: int = 10, tile_k: int = 64):
        self.is_global = bool(is_global)
        self.tile_m = 16
        self.warp_m = 1
        self.warp_n = int(warp_n)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.num_stages = int(num_stages)
        self.num_compute_warps = self.warp_m * self.warp_n
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        if self.K % self.tile_k or self.tile_k % 16:
            raise ValueError(f"K={self.K} must divide by tile_k={self.tile_k}")
        self.k_tiles = self.K // self.tile_k

        def tiles(n: int) -> int:
            return (n + self.tile_n - 1) // self.tile_n

        self.tq = tiles(self.N_Q)
        self.tgate = self.tq + tiles(self.N_GATE)
        self.tk = self.tgate + tiles(self.N_K)
        self.tv = self.tk + tiles(self.N_V)
        self.tr = self.tv + tiles(self.N_R)
        self.tsink = self.tr + tiles(self.N_SINK)
        if self.is_global:
            self.tkdyn = self.tsink + tiles(self.N_DYN)
            self.tvdyn = self.tkdyn + tiles(self.N_DYN)
            self.total_tiles = self.tvdyn
        else:
            self.tkdyn = self.tsink
            self.tvdyn = self.tsink
            self.total_tiles = self.tsink

    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.warp_m, self.warp_n, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _get_smem_layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, self.tile_k),
            BFloat16,
        )
        sA_layout = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        sB_layout = cute.tile_to_shape(atom, (self.tile_n, self.tile_k, self.num_stages), order=(0, 1, 2))
        return sA_layout, sB_layout

    def _get_shared_storage_cls(self, sA_layout, sB_layout):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sA": cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sA_layout)], self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(sB_layout)], self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

    def _make_tma_a(self, ptr: cute.Pointer, rows: Int32, sA_tma_layout):
        t = _assume_tma_source_aligned(cute.make_tensor(ptr, cute.make_layout((rows, self.K), stride=(self.K, 1))))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), t, sA_tma_layout, (self.tile_m, self.tile_k), num_multicast=1,
        )

    def _make_tma_b(self, ptr: cute.Pointer, n: int, sB_tma_layout):
        t = _assume_tma_source_aligned(cute.make_tensor(ptr, cute.make_layout((n, self.K), stride=(self.K, 1))))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), t, sB_tma_layout, (self.tile_n, self.tile_k), num_multicast=1,
        )

    @cute.jit
    def launch_local(self, xq: cute.Pointer, xk: cute.Pointer, xv: cute.Pointer,
                     wq: cute.Pointer, wr: cute.Pointer, wgate: cute.Pointer,
                     wsink: cute.Pointer, wk: cute.Pointer, wv: cute.Pointer,
                     out_q: cute.Pointer, out_r: cute.Pointer, out_gate: cute.Pointer,
                     out_sink: cute.Pointer, out_k: cute.Pointer, out_v: cute.Pointer,
                     rows: Int32, stream: cuda.CUstream):
        sA_layout, sB_layout = self._get_smem_layouts()
        tiled_mma = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
        sA_tma = cute.slice_(sA_layout, (None, None, 0))
        sB_tma = cute.slice_(sB_layout, (None, None, 0))

        tma_xq, g_xq = self._make_tma_a(xq, rows, sA_tma)
        tma_xk, g_xk = self._make_tma_a(xk, rows, sA_tma)
        tma_xv, g_xv = self._make_tma_a(xv, rows, sA_tma)
        tma_wq, g_wq = self._make_tma_b(wq, self.N_Q, sB_tma)
        tma_wg, g_wg = self._make_tma_b(wgate, self.N_GATE, sB_tma)
        tma_wk, g_wk = self._make_tma_b(wk, self.N_K, sB_tma)
        tma_wv, g_wv = self._make_tma_b(wv, self.N_V, sB_tma)
        tma_wr, g_wr = self._make_tma_b(wr, self.N_R, sB_tma)
        tma_ws, g_ws = self._make_tma_b(wsink, self.N_SINK, sB_tma)

        self.kernel_local(
            g_xq, g_xk, g_xv, g_wq, g_wg, g_wk, g_wv, g_wr, g_ws,
            tma_xq, tma_xk, tma_xv, tma_wq, tma_wg, tma_wk, tma_wv, tma_wr, tma_ws,
            out_q, out_gate, out_k, out_v, out_r, out_sink,
            sA_layout, sB_layout, tiled_mma, SharedStorage, rows,
        ).launch(
            grid=(self.total_tiles, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def launch_global(self, xq: cute.Pointer, xk: cute.Pointer, xv: cute.Pointer,
                      wq: cute.Pointer, wr: cute.Pointer, wgate: cute.Pointer,
                      wsink: cute.Pointer, wk: cute.Pointer, wv: cute.Pointer,
                      wk_dyn: cute.Pointer, wv_dyn: cute.Pointer,
                      out_q: cute.Pointer, out_r: cute.Pointer, out_gate: cute.Pointer,
                      out_sink: cute.Pointer, out_k: cute.Pointer, out_v: cute.Pointer,
                      out_k_dyn: cute.Pointer, out_v_dyn: cute.Pointer,
                      rows: Int32, stream: cuda.CUstream):
        sA_layout, sB_layout = self._get_smem_layouts()
        tiled_mma = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
        sA_tma = cute.slice_(sA_layout, (None, None, 0))
        sB_tma = cute.slice_(sB_layout, (None, None, 0))

        tma_xq, g_xq = self._make_tma_a(xq, rows, sA_tma)
        tma_xk, g_xk = self._make_tma_a(xk, rows, sA_tma)
        tma_xv, g_xv = self._make_tma_a(xv, rows, sA_tma)
        tma_wq, g_wq = self._make_tma_b(wq, self.N_Q, sB_tma)
        tma_wg, g_wg = self._make_tma_b(wgate, self.N_GATE, sB_tma)
        tma_wk, g_wk = self._make_tma_b(wk, self.N_K, sB_tma)
        tma_wv, g_wv = self._make_tma_b(wv, self.N_V, sB_tma)
        tma_wr, g_wr = self._make_tma_b(wr, self.N_R, sB_tma)
        tma_ws, g_ws = self._make_tma_b(wsink, self.N_SINK, sB_tma)
        tma_wkd, g_wkd = self._make_tma_b(wk_dyn, self.N_DYN, sB_tma)
        tma_wvd, g_wvd = self._make_tma_b(wv_dyn, self.N_DYN, sB_tma)

        self.kernel_global(
            g_xq, g_xk, g_xv, g_wq, g_wg, g_wk, g_wv, g_wr, g_ws, g_wkd, g_wvd,
            tma_xq, tma_xk, tma_xv, tma_wq, tma_wg, tma_wk, tma_wv, tma_wr, tma_ws, tma_wkd, tma_wvd,
            out_q, out_gate, out_k, out_v, out_r, out_sink, out_k_dyn, out_v_dyn,
            sA_layout, sB_layout, tiled_mma, SharedStorage, rows,
        ).launch(
            grid=(self.total_tiles, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _compute_epilogue(self, sA, sB, load_pipeline, tiled_mma, tidx,
                          local_tile: Int32, n_lim: Int32, o_ptr: Int64, is_f32: Int32, num_tokens: Int32):
        consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
        acc.fill(0.0)
        copy_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16)
        smem_thr_copy_A = cute.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tidx)
        smem_thr_copy_B = cute.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tidx)
        tSsA = smem_thr_copy_A.partition_S(sA)
        tSsB = smem_thr_copy_B.partition_S(sB)

        for _k_tile in cutlass.range(self.k_tiles):
            load_pipeline.consumer_wait(consumer_state)
            _warp_gemm_direct(
                thr_mma, acc, tCrA, tCrB,
                tSsA[None, None, None, consumer_state.index],
                tSsB[None, None, None, consumer_state.index],
                smem_thr_copy_A, smem_thr_copy_B,
            )
            cute.arch.fence_view_async_shared()
            load_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

        coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
        if is_f32 == Int32(0):
            for pair in cutlass.range_constexpr(cute.size(acc) // 2):
                coord = coordinates[2 * pair]
                token = coord[0]
                column = local_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens and column < n_lim:
                    at = (Int64(token) * Int64(n_lim) + Int64(column)) * Int64(2)
                    st_global_u32(o_ptr + at, pack_f32x2_to_bfloat2(acc[2 * pair], acc[2 * pair + 1]))
        else:
            for index in cutlass.range_constexpr(cute.size(acc)):
                coord = coordinates[index]
                token = coord[0]
                column = local_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens and column < n_lim:
                    at = (Int64(token) * Int64(n_lim) + Int64(column)) * Int64(4)
                    p = cute.make_ptr(Float32, o_ptr + at, cute.AddressSpace.gmem, assumed_align=4)
                    p[0] = acc[index]

    @cute.jit
    def _tma_part_a(self, tma_atom, g_tensor, sA):
        gA = cute.local_tile(g_tensor, (self.tile_m, self.tile_k), (None, None))
        return cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
        )

    @cute.jit
    def _tma_part_b(self, tma_atom, g_tensor, sB):
        gB = cute.local_tile(g_tensor, (self.tile_n, self.tile_k), (None, None))
        return cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2),
        )

    @cute.jit
    def _produce_loop(self, load_pipeline, tma_a, tAgA, tAsA, tma_b, tBgB, tBsB, local_tile: Int32):
        producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        for k_tile in cutlass.range(self.k_tiles):
            load_pipeline.producer_acquire(producer_state)
            cute.copy(
                tma_a, tAgA[(None, Int32(0), k_tile)], tAsA[(None, producer_state.index)],
                tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
            )
            cute.copy(
                tma_b, tBgB[(None, local_tile, k_tile)], tBsB[(None, producer_state.index)],
                tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
            )
            load_pipeline.producer_commit(producer_state)
            producer_state.advance()
        load_pipeline.producer_tail(producer_state)

    @cute.kernel
    def kernel_local(self,
                     g_xq: cute.Tensor, g_xk: cute.Tensor, g_xv: cute.Tensor,
                     g_wq: cute.Tensor, g_wg: cute.Tensor, g_wk: cute.Tensor,
                     g_wv: cute.Tensor, g_wr: cute.Tensor, g_ws: cute.Tensor,
                     tma_xq: cute.CopyAtom, tma_xk: cute.CopyAtom, tma_xv: cute.CopyAtom,
                     tma_wq: cute.CopyAtom, tma_wg: cute.CopyAtom, tma_wk: cute.CopyAtom,
                     tma_wv: cute.CopyAtom, tma_wr: cute.CopyAtom, tma_ws: cute.CopyAtom,
                     out_q: cute.Pointer, out_gate: cute.Pointer, out_k: cute.Pointer,
                     out_v: cute.Pointer, out_r: cute.Pointer, out_sink: cute.Pointer,
                     sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
                     tiled_mma: cute.TiledMma, SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bid = Int32(cute.arch.block_idx()[0])
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (self.tile_m + self.tile_n) * self.tile_k * BFloat16.width // 8
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        local_tile = bid
        n_lim = Int32(self.N_Q)
        o_ptr = Int64(out_q.toint())
        is_f32 = Int32(0)
        if bid < Int32(self.tq):
            pass
        elif bid < Int32(self.tgate):
            local_tile = bid - Int32(self.tq)
            n_lim = Int32(self.N_GATE)
            o_ptr = Int64(out_gate.toint())
        elif bid < Int32(self.tk):
            local_tile = bid - Int32(self.tgate)
            n_lim = Int32(self.N_K)
            o_ptr = Int64(out_k.toint())
        elif bid < Int32(self.tv):
            local_tile = bid - Int32(self.tk)
            n_lim = Int32(self.N_V)
            o_ptr = Int64(out_v.toint())
        elif bid < Int32(self.tr):
            local_tile = bid - Int32(self.tv)
            n_lim = Int32(self.N_R)
            o_ptr = Int64(out_r.toint())
        else:
            local_tile = bid - Int32(self.tr)
            n_lim = Int32(self.N_SINK)
            o_ptr = Int64(out_sink.toint())
            is_f32 = Int32(1)

        if warp_idx < Int32(self.num_compute_warps):
            self._compute_epilogue(sA, sB, load_pipeline, tiled_mma, tidx, local_tile, n_lim, o_ptr, is_f32, num_tokens)
        elif warp_idx == Int32(self.producer_warp):
            if bid < Int32(self.tq):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_wq, g_wq, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_wq, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tgate):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_wg, g_wg, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_wg, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tk):
                tAsA, tAgA = self._tma_part_a(tma_xk, g_xk, sA)
                tBsB, tBgB = self._tma_part_b(tma_wk, g_wk, sB)
                self._produce_loop(load_pipeline, tma_xk, tAgA, tAsA, tma_wk, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tv):
                tAsA, tAgA = self._tma_part_a(tma_xv, g_xv, sA)
                tBsB, tBgB = self._tma_part_b(tma_wv, g_wv, sB)
                self._produce_loop(load_pipeline, tma_xv, tAgA, tAsA, tma_wv, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tr):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_wr, g_wr, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_wr, tBgB, tBsB, local_tile)
            else:
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_ws, g_ws, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_ws, tBgB, tBsB, local_tile)

    @cute.kernel
    def kernel_global(self,
                      g_xq: cute.Tensor, g_xk: cute.Tensor, g_xv: cute.Tensor,
                      g_wq: cute.Tensor, g_wg: cute.Tensor, g_wk: cute.Tensor,
                      g_wv: cute.Tensor, g_wr: cute.Tensor, g_ws: cute.Tensor,
                      g_wkd: cute.Tensor, g_wvd: cute.Tensor,
                      tma_xq: cute.CopyAtom, tma_xk: cute.CopyAtom, tma_xv: cute.CopyAtom,
                      tma_wq: cute.CopyAtom, tma_wg: cute.CopyAtom, tma_wk: cute.CopyAtom,
                      tma_wv: cute.CopyAtom, tma_wr: cute.CopyAtom, tma_ws: cute.CopyAtom,
                      tma_wkd: cute.CopyAtom, tma_wvd: cute.CopyAtom,
                      out_q: cute.Pointer, out_gate: cute.Pointer, out_k: cute.Pointer,
                      out_v: cute.Pointer, out_r: cute.Pointer, out_sink: cute.Pointer,
                      out_k_dyn: cute.Pointer, out_v_dyn: cute.Pointer,
                      sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
                      tiled_mma: cute.TiledMma, SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bid = Int32(cute.arch.block_idx()[0])
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (self.tile_m + self.tile_n) * self.tile_k * BFloat16.width // 8
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        local_tile = bid
        n_lim = Int32(self.N_Q)
        o_ptr = Int64(out_q.toint())
        is_f32 = Int32(0)
        if bid < Int32(self.tq):
            pass
        elif bid < Int32(self.tgate):
            local_tile = bid - Int32(self.tq)
            n_lim = Int32(self.N_GATE)
            o_ptr = Int64(out_gate.toint())
        elif bid < Int32(self.tk):
            local_tile = bid - Int32(self.tgate)
            n_lim = Int32(self.N_K)
            o_ptr = Int64(out_k.toint())
        elif bid < Int32(self.tv):
            local_tile = bid - Int32(self.tk)
            n_lim = Int32(self.N_V)
            o_ptr = Int64(out_v.toint())
        elif bid < Int32(self.tr):
            local_tile = bid - Int32(self.tv)
            n_lim = Int32(self.N_R)
            o_ptr = Int64(out_r.toint())
        elif bid < Int32(self.tsink):
            local_tile = bid - Int32(self.tr)
            n_lim = Int32(self.N_SINK)
            o_ptr = Int64(out_sink.toint())
            is_f32 = Int32(1)
        elif bid < Int32(self.tkdyn):
            local_tile = bid - Int32(self.tsink)
            n_lim = Int32(self.N_DYN)
            o_ptr = Int64(out_k_dyn.toint())
            is_f32 = Int32(1)
        else:
            local_tile = bid - Int32(self.tkdyn)
            n_lim = Int32(self.N_DYN)
            o_ptr = Int64(out_v_dyn.toint())
            is_f32 = Int32(1)

        if warp_idx < Int32(self.num_compute_warps):
            self._compute_epilogue(sA, sB, load_pipeline, tiled_mma, tidx, local_tile, n_lim, o_ptr, is_f32, num_tokens)
        elif warp_idx == Int32(self.producer_warp):
            if bid < Int32(self.tq):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_wq, g_wq, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_wq, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tgate):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_wg, g_wg, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_wg, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tk):
                tAsA, tAgA = self._tma_part_a(tma_xk, g_xk, sA)
                tBsB, tBgB = self._tma_part_b(tma_wk, g_wk, sB)
                self._produce_loop(load_pipeline, tma_xk, tAgA, tAsA, tma_wk, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tv):
                tAsA, tAgA = self._tma_part_a(tma_xv, g_xv, sA)
                tBsB, tBgB = self._tma_part_b(tma_wv, g_wv, sB)
                self._produce_loop(load_pipeline, tma_xv, tAgA, tAsA, tma_wv, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tr):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_wr, g_wr, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_wr, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tsink):
                tAsA, tAgA = self._tma_part_a(tma_xq, g_xq, sA)
                tBsB, tBgB = self._tma_part_b(tma_ws, g_ws, sB)
                self._produce_loop(load_pipeline, tma_xq, tAgA, tAsA, tma_ws, tBgB, tBsB, local_tile)
            elif bid < Int32(self.tkdyn):
                tAsA, tAgA = self._tma_part_a(tma_xk, g_xk, sA)
                tBsB, tBgB = self._tma_part_b(tma_wkd, g_wkd, sB)
                self._produce_loop(load_pipeline, tma_xk, tAgA, tAsA, tma_wkd, tBgB, tBsB, local_tile)
            else:
                tAsA, tAgA = self._tma_part_a(tma_xv, g_xv, sA)
                tBsB, tBgB = self._tma_part_b(tma_wvd, g_wvd, sB)
                self._produce_loop(load_pipeline, tma_xv, tAgA, tAsA, tma_wvd, tBgB, tBsB, local_tile)


class BlockKSmallRowKernel:
    """Single-launch block-K BF16 GEMV for narrow N (Router 2304 -> 128 F32)."""

    def __init__(self, n: int, k: int, *, out_dtype=Float32, cols: int = 1, threads: int = 288):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.out_is_f32 = out_dtype == Float32
        self.cols = int(cols)
        self.threads = int(threads)
        self.vectors = self.k // 8
        if self.k % 8 or self.vectors % self.threads or self.threads % 32:
            raise ValueError("K/8 must be divisible by threads (multiple of 32)")
        if self.n % self.cols:
            raise ValueError("N must be divisible by cols")
        self.steps = self.vectors // self.threads
        self.warps = self.threads // 32

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[
                cute.struct.MemRange[Float32, self.warps * SMALL_ROW_MAX * self.cols], 16
            ],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, w, out, rows).launch(
            grid=(self.n // self.cols, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp = tidx // Int32(32)
        lane = tidx % Int32(32)
        col0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(
            cute.make_layout((self.warps, SMALL_ROW_MAX * self.cols),
                             stride=(SMALL_ROW_MAX * self.cols, 1)))
        w_base = Int64(w.toint()) + col0 * Int64(self.k * 2) + Int64(tidx) * Int64(16)
        x_base = Int64(x.toint()) + Int64(tidx) * Int64(16)
        for r in cutlass.range_constexpr(1, SMALL_ROW_MAX + 1):
            if rows == Int32(r):
                self._run_fixed(w_base, x_base, partial, out, col0, tidx, warp, lane, r)

    @cute.jit
    def _run_fixed(self, w_base: Int64, x_base: Int64, partial: cute.Tensor, out: cute.Pointer,
                   col0: Int64, tidx: Int32, warp: Int32, lane: Int32,
                   live_rows: cutlass.Constexpr):
        C = self.cols
        M = live_rows
        acc = cute.make_rmem_tensor(cute.make_layout((M * C,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(M * C):
            acc[i] = Float32(0.0)
        for step in cutlass.range_constexpr(self.steps):
            off = Int64(step * self.threads * 16)
            wv = cute.make_rmem_tensor(cute.make_layout((C * 8,), stride=(1,)), Float32)
            for c in cutlass.range_constexpr(C):
                words = _ld_stream(w_base + Int64(c * self.k * 2) + off)
                for j in cutlass.range_constexpr(4):
                    wv[8 * c + 2 * j] = _bf16_lo(words[j])
                    wv[8 * c + 2 * j + 1] = _bf16_hi(words[j])
            for m in cutlass.range_constexpr(M):
                xw = _ld_cached(x_base + Int64(m * self.k * 2) + off)
                xs = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Float32)
                for j in cutlass.range_constexpr(4):
                    xs[2 * j] = _bf16_lo(xw[j])
                    xs[2 * j + 1] = _bf16_hi(xw[j])
                for c in cutlass.range_constexpr(C):
                    a = acc[m * C + c]
                    for j in cutlass.range_constexpr(8):
                        a = a + wv[8 * c + j] * xs[j]
                    acc[m * C + c] = a
        for i in cutlass.range_constexpr(M * C):
            v = _warp_reduce_sum(acc[i])
            if lane == Int32(0):
                partial[warp, i] = v
        cute.arch.sync_threads()
        if tidx < Int32(M * C):
            m = tidx // Int32(C)
            c = tidx % Int32(C)
            total = Float32(0.0)
            for w_idx in cutlass.range_constexpr(self.warps):
                total = total + partial[w_idx, tidx]
            _store_scalar(out, m, col0 + Int64(c), Int64(self.n), total, self.out_is_f32)


class _RoutedAttnInTmaLocal:
    def __init__(self, *, warp_n: int = 4, tile_n: int = 32, num_stages: int = 10, tile_k: int = 64):
        self.small = FusedAttnInTmaKernel(
            is_global=False, warp_n=warp_n, tile_n=tile_n, num_stages=num_stages, tile_k=tile_k,
        )
        self.stock_q = TmaBf16Projection(3072, 2304, out_dtype=BFloat16, tiles_by_rows=STOCK_TILES)
        self.stock_r = TmaBf16Projection(384, 2304, out_dtype=BFloat16, tiles_by_rows=STOCK_TILES)
        self.stock_kv = TmaBf16Projection(1024, 2304, out_dtype=BFloat16, tiles_by_rows=STOCK_TILES)
        self.stock_sink = TmaBf16Projection(24, 2304, out_dtype=Float32, tiles_by_rows=STOCK_TILES)

    @cute.jit
    def __call__(self, xq: cute.Pointer, xk: cute.Pointer, xv: cute.Pointer,
                 wq: cute.Pointer, wr: cute.Pointer, wgate: cute.Pointer,
                 wsink: cute.Pointer, wk: cute.Pointer, wv: cute.Pointer,
                 out_q: cute.Pointer, out_r: cute.Pointer, out_gate: cute.Pointer,
                 out_sink: cute.Pointer, out_k: cute.Pointer, out_v: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows <= Int32(SMALL_ROW_MAX):
            self.small.launch_local(
                xq, xk, xv, wq, wr, wgate, wsink, wk, wv,
                out_q, out_r, out_gate, out_sink, out_k, out_v, rows, stream,
            )
        else:
            self.stock_q(xq, wq, out_q, rows, stream)
            self.stock_r(xq, wr, out_r, rows, stream)
            self.stock_q(xq, wgate, out_gate, rows, stream)
            self.stock_sink(xq, wsink, out_sink, rows, stream)
            self.stock_kv(xk, wk, out_k, rows, stream)
            self.stock_kv(xv, wv, out_v, rows, stream)


class _RoutedAttnInTmaGlobal:
    def __init__(self, *, warp_n: int = 4, tile_n: int = 32, num_stages: int = 10, tile_k: int = 64):
        self.small = FusedAttnInTmaKernel(
            is_global=True, warp_n=warp_n, tile_n=tile_n, num_stages=num_stages, tile_k=tile_k,
        )
        self.stock_q = TmaBf16Projection(3072, 2304, out_dtype=BFloat16, tiles_by_rows=STOCK_TILES)
        self.stock_r = TmaBf16Projection(384, 2304, out_dtype=BFloat16, tiles_by_rows=STOCK_TILES)
        self.stock_kv = TmaBf16Projection(1024, 2304, out_dtype=BFloat16, tiles_by_rows=STOCK_TILES)
        self.stock_sink = TmaBf16Projection(24, 2304, out_dtype=Float32, tiles_by_rows=STOCK_TILES)
        self.stock_dyn = TmaBf16Projection(32, 2304, out_dtype=Float32, tiles_by_rows=STOCK_TILES)

    @cute.jit
    def __call__(self, xq: cute.Pointer, xk: cute.Pointer, xv: cute.Pointer,
                 wq: cute.Pointer, wr: cute.Pointer, wgate: cute.Pointer,
                 wsink: cute.Pointer, wk: cute.Pointer, wv: cute.Pointer,
                 wk_dyn: cute.Pointer, wv_dyn: cute.Pointer,
                 out_q: cute.Pointer, out_r: cute.Pointer, out_gate: cute.Pointer,
                 out_sink: cute.Pointer, out_k: cute.Pointer, out_v: cute.Pointer,
                 out_k_dyn: cute.Pointer, out_v_dyn: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows <= Int32(SMALL_ROW_MAX):
            self.small.launch_global(
                xq, xk, xv, wq, wr, wgate, wsink, wk, wv, wk_dyn, wv_dyn,
                out_q, out_r, out_gate, out_sink, out_k, out_v, out_k_dyn, out_v_dyn, rows, stream,
            )
        else:
            self.stock_q(xq, wq, out_q, rows, stream)
            self.stock_r(xq, wr, out_r, rows, stream)
            self.stock_q(xq, wgate, out_gate, rows, stream)
            self.stock_sink(xq, wsink, out_sink, rows, stream)
            self.stock_kv(xk, wk, out_k, rows, stream)
            self.stock_dyn(xk, wk_dyn, out_k_dyn, rows, stream)
            self.stock_kv(xv, wv, out_v, rows, stream)
            self.stock_dyn(xv, wv_dyn, out_v_dyn, rows, stream)


class _RoutedRouterProj:
    def __init__(self, *, cols: int = 1, threads: int = 288):
        self.small = BlockKSmallRowKernel(128, 2304, out_dtype=Float32, cols=cols, threads=threads)
        self.stock = TmaBf16Projection(128, 2304, out_dtype=Float32, tiles_by_rows=STOCK_TILES)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if rows <= Int32(SMALL_ROW_MAX):
            self.small(x, w, out, rows, stream)
        else:
            self.stock(x, w, out, rows, stream)


def _validate_max_rows(max_rows: int) -> int:
    if type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1:
        raise ValueError("invalid Inkling 2 Mini projection max_rows")
    return max_rows


def compile_inkling2_mini_attn_in_proj_local_aot(*, max_rows: int = 4096):
    """Fused 6-output local attention-input BF16/F32 projection for Inkling 2 Mini."""
    max_rows = _validate_max_rows(max_rows)
    launch = _RoutedAttnInTmaLocal(**ATTN_IN_TMA_CFG)
    return compile_program(
        launch,
        name="inkling2_mini_attn_in_proj_local",
        operands=(
            Operand("xq", torch.bfloat16, "[rows,2304]"),
            Operand("xk", torch.bfloat16, "[rows,2304]"),
            Operand("xv", torch.bfloat16, "[rows,2304]"),
            Operand("wq", torch.bfloat16, "[3072,2304]"),
            Operand("wr", torch.bfloat16, "[384,2304]"),
            Operand("wgate", torch.bfloat16, "[3072,2304]"),
            Operand("wsink", torch.bfloat16, "[24,2304]"),
            Operand("wk", torch.bfloat16, "[1024,2304]"),
            Operand("wv", torch.bfloat16, "[1024,2304]"),
            Operand("out_q", torch.bfloat16, "[rows,3072]", "out"),
            Operand("out_r", torch.bfloat16, "[rows,384]", "out"),
            Operand("out_gate", torch.bfloat16, "[rows,3072]", "out"),
            Operand("out_sink", torch.float32, "[rows,24]", "out"),
            Operand("out_k", torch.bfloat16, "[rows,1024]", "out"),
            Operand("out_v", torch.bfloat16, "[rows,1024]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=("attn_in_local", max_rows, tuple(sorted(ATTN_IN_TMA_CFG.items()))),
        geometry=dict(
            width=2304,
            small_row_max=SMALL_ROW_MAX,
            global_dynamic=False,
            outputs=[3072, 384, 3072, 24, 1024, 1024],
            reduction="ordered_k16",
            max_rows=max_rows,
        ),
        doc=__doc__,
    )


def compile_inkling2_mini_attn_in_proj_global_aot(*, max_rows: int = 4096):
    """Fused 8-output global attention-input BF16/F32 projection for Inkling 2 Mini."""
    max_rows = _validate_max_rows(max_rows)
    launch = _RoutedAttnInTmaGlobal(**ATTN_IN_TMA_CFG)
    return compile_program(
        launch,
        name="inkling2_mini_attn_in_proj_global",
        operands=(
            Operand("xq", torch.bfloat16, "[rows,2304]"),
            Operand("xk", torch.bfloat16, "[rows,2304]"),
            Operand("xv", torch.bfloat16, "[rows,2304]"),
            Operand("wq", torch.bfloat16, "[3072,2304]"),
            Operand("wr", torch.bfloat16, "[384,2304]"),
            Operand("wgate", torch.bfloat16, "[3072,2304]"),
            Operand("wsink", torch.bfloat16, "[24,2304]"),
            Operand("wk", torch.bfloat16, "[1024,2304]"),
            Operand("wv", torch.bfloat16, "[1024,2304]"),
            Operand("wk_dyn", torch.bfloat16, "[32,2304]"),
            Operand("wv_dyn", torch.bfloat16, "[32,2304]"),
            Operand("out_q", torch.bfloat16, "[rows,3072]", "out"),
            Operand("out_r", torch.bfloat16, "[rows,384]", "out"),
            Operand("out_gate", torch.bfloat16, "[rows,3072]", "out"),
            Operand("out_sink", torch.float32, "[rows,24]", "out"),
            Operand("out_k", torch.bfloat16, "[rows,1024]", "out"),
            Operand("out_v", torch.bfloat16, "[rows,1024]", "out"),
            Operand("out_k_dyn", torch.float32, "[rows,32]", "out"),
            Operand("out_v_dyn", torch.float32, "[rows,32]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=("attn_in_global", max_rows, tuple(sorted(ATTN_IN_TMA_CFG.items()))),
        geometry=dict(
            width=2304,
            small_row_max=SMALL_ROW_MAX,
            global_dynamic=True,
            outputs=[3072, 384, 3072, 24, 1024, 1024, 32, 32],
            reduction="ordered_k16",
            max_rows=max_rows,
        ),
        doc=__doc__,
    )


def compile_inkling2_mini_router_proj_aot(*, max_rows: int = 4096):
    """Small-row block-K (M<=8) + stock TMA (M>8) router projection 2304->128 F32."""
    max_rows = _validate_max_rows(max_rows)
    launch = _RoutedRouterProj(**ROUTER_BLOCK_CFG)
    return compile_program(
        launch,
        name="inkling2_mini_router_proj",
        operands=(
            Operand("x", torch.bfloat16, "[rows,2304]"),
            Operand("w", torch.bfloat16, "[128,2304]"),
            Operand("out", torch.float32, "[rows,128]", "out"),
        ),
        scalars=(Scalar("rows"),),
        key=("router_proj", max_rows, tuple(sorted(ROUTER_BLOCK_CFG.items()))),
        geometry=dict(
            width=2304,
            outputs=128,
            output_dtype="torch.float32",
            small_row_max=SMALL_ROW_MAX,
            reduction="block_k288_le8_ordered_k16_gt8",
            max_rows=max_rows,
        ),
        doc=__doc__,
    )
