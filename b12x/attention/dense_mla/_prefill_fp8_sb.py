"""K3 dense-MLA prefill over compact FP8 records, shared-operand tiles.

Record (656 bytes, the shared GLM/DeepSeek MLA latent ABI):
    [  0, 512)  E4M3 latent, natural dimension order, four 128-wide groups
    [512, 528)  four FP32 group scales (dequantized value = e4m3 * scale)
    [528, 656)  64 BF16 raw key tail

Arithmetic is BF16 tensor-core math with exact widening: E4M3 widens to BF16
without rounding, each group's QK partial is scaled in FP32 after its MMA, and
the PV side folds the group scale into the BF16 probability operand.

Two query rows per CTA, and every widened fragment is shared between them:

* Score phase.  A math warp computes *both* rows' sixteen heads as two m16
  tiles over one widened K fragment, so a key byte is widened once per warp per
  chunk instead of once per (warp, row).  Four score warps cover the chunk,
  sixteen candidates each.
* Value phase.  A value warp owns 64 latent dims for both rows, so one widened
  V fragment feeds two MMAs (one per row) and a value byte is widened once per
  warp per chunk.  All eight math warps take part, each holding 8 n-tiles x 2
  rows = 64 accumulator registers, the same per-thread accumulator load as the
  round-2 kernel.

The KV ring holds whole records exactly as stored; a chunk never crosses a
64-token page, so a full chunk is a single bulk copy.  ``splits`` cuts a chunk's
copy into evenly sized bulk copies with one mbarrier each, so a score warp
starts as soon as the records it owns have landed instead of waiting for the
whole chunk; the value phase still waits for every split.  A chunk that reaches
past ``cache_length`` falls back to per-record copies that substitute record
zero for the invalid tail (masked by the consumer, never NaN-poisoning PV).

Serves the single-request, non-windowed causal extend route only (one split).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    fmax_f32,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_shared_f32,
    ld_shared_v4_f32,
    ldmatrix_m16n16x1_trans_b8,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4_u32,
)

from ._math import exp2_approx, log2_approx

RECORD_BYTES = 656
LATENT_DIM = 512
SCALE_OFFSET = 512
TAIL_OFFSET = 528
QK_DIM = 576
VALUE_DIM = 512
GROUP = 128
GROUPS = LATENT_DIM // GROUP
HEADS_PER_TILE = 16
ROWS = 2
MATH_WARPS = 8
# Every math warp scores eight candidates of the chunk, so the score phase
# spans the whole chunk with all eight warps issuing tensor ops.
WARP_CANDS = 8
# Each value warp owns this many latent dims for both query rows (two m16 tiles
# fed by one widened V fragment).
VALUE_DIMS_PER_WARP = VALUE_DIM // MATH_WARPS
# BF16 query row in shared memory: 576 dims + 8 elements so successive head
# rows rotate across bank groups for ldmatrix (1168 = 9*128 + 16).
Q_ROW_BYTES = QK_DIM * 2 + 16
Q_TILE_BYTES = ROWS * HEADS_PER_TILE * Q_ROW_BYTES
_MASK = -1.0e30


def smem_bytes(*, chunk: int, stages: int, splits: int) -> int:
    kv = stages * chunk * RECORD_BYTES
    p = ROWS * GROUPS * HEADS_PER_TILE * (chunk + 8) * 2
    reduce = 2 * ROWS * MATH_WARPS * HEADS_PER_TILE * 4
    barriers = (stages * splits + stages) * 8
    return kv + Q_TILE_BYTES + p + reduce + barriers + 4 * 128


class DenseMlaFp8SharedBPrefillKernel:
    def __init__(
        self,
        *,
        num_heads: int,
        page_size: int = 64,
        chunk: int = 64,
        stages: int = 1,
        splits: int = 1,
        min_blocks: int = 1,
    ):
        if num_heads % HEADS_PER_TILE:
            raise ValueError("FP8 MLA prefill needs a multiple of sixteen heads")
        if chunk != 64 or page_size % chunk:
            raise ValueError("shared-B FP8 MLA prefill scores 64-token chunks")
        if stages not in (1, 2, 3):
            raise ValueError("unsupported FP8 MLA prefill ring depth")
        if splits not in (1, 2, 4) or chunk % splits:
            raise ValueError("splits must be 1, 2 or 4 and divide the chunk")
        if chunk // splits < 8:
            raise ValueError("a copy split must hold at least eight records")
        self.num_heads = int(num_heads)
        self.page_size = int(page_size)
        self.chunk = int(chunk)
        self.stages = int(stages)
        self.splits = int(splits)
        self.min_blocks = int(min_blocks)
        self.head_tiles = self.num_heads // HEADS_PER_TILE
        # Candidates per score warp: n8 tiles of eight records.
        self.warp_cands = self.chunk // MATH_WARPS
        self.ntiles = self.warp_cands // 8
        self.split_records = self.chunk // self.splits
        self.kv_stage_bytes = self.chunk * RECORD_BYTES
        self.split_bytes = self.split_records * RECORD_BYTES
        self.math_threads = 32 * MATH_WARPS
        self.block_threads = self.math_threads + 32
        self.p_stride = (self.chunk + 8) * 2
        self.p_group_bytes = HEADS_PER_TILE * self.p_stride
        self.p_row_bytes = GROUPS * self.p_group_bytes
        total = smem_bytes(chunk=chunk, stages=stages, splits=splits)
        if total > 101376:
            raise ValueError(f"FP8 MLA prefill tile needs {total} bytes of SMEM")

    def _storage(self):
        p_total = ROWS * GROUPS * self.p_group_bytes
        red_total = 2 * ROWS * MATH_WARPS * HEADS_PER_TILE

        class Storage:
            pass

        Storage.__annotations__ = {
            "kv": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.stages * self.kv_stage_bytes], 128
            ],
            "q": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, Q_TILE_BYTES], 128],
            "p": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, p_total], 128],
            "reduce": cute.struct.Align[cute.struct.MemRange[cutlass.Float32, red_total], 16],
            "mbar": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint64, self.stages * self.splits + self.stages], 16
            ],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        sm_scale_log2: Float32,
        q_stride_row_bytes: Int64,
        page_stride_bytes: Int64,
        page_table_stride: Int64,
        total_q: Int32,
        batch: Int32,
        stream: cuda.CUstream,
    ):
        query_tiles = (total_q + Int32(ROWS - 1)) // Int32(ROWS) + batch
        self.kernel(
            q_bytes, cache_bytes, page_table, cache_seqlens, cu_seqlens_q,
            output, final_lse, sm_scale_log2, q_stride_row_bytes,
            page_stride_bytes, page_table_stride, total_q, batch,
        ).launch(
            grid=(query_tiles, self.head_tiles, 1),
            block=[self.block_threads, 1, 1],
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    # ------------------------------------------------------------------ math
    @cute.jit
    def _stage_query(
        self, q_bytes: cute.Tensor, q_addr: Int32, query_start: Int32, head_base: Int32,
        total_q: Int32, query_end: Int32, math_tid: Int32, q_stride_row_bytes: Int64,
    ):
        """Stage both rows' sixteen BF16 query heads, permuting each 16-dim block.

        Within a latent block the score MMA's k-slots (2t, 2t+1, 2t+8, 2t+9)
        address physical dimensions (4t .. 4t+3), the byte order a
        non-transposed ldmatrix delivers for E4M3 keys.  As 32-bit words the
        permutation is [w0, w2, w4, w6, w1, w3, w5, w7].  The tail is natural.
        A row past the request end replicates row zero and is masked when it
        stores, so no lane sees uninitialised operands.
        """
        blocks = QK_DIM // 16
        units = ROWS * HEADS_PER_TILE * blocks
        unit = math_tid
        while unit < Int32(units):
            row_in_tile = unit // Int32(HEADS_PER_TILE * blocks)
            rem = unit - row_in_tile * Int32(HEADS_PER_TILE * blocks)
            head = rem // Int32(blocks)
            block = rem - head * Int32(blocks)
            row = query_start + row_in_tile
            safe_row = row
            if row >= total_q or row >= query_end:
                safe_row = Int32(0)
            src = get_ptr_as_int64(
                q_bytes,
                safe_row.to(Int64) * q_stride_row_bytes
                + (head_base + head).to(Int64) * Int64(QK_DIM * 2)
                + (block * Int32(32)).to(Int64),
            )
            w0, w1, w2, w3 = ld_global_nc_v4_u32(src)
            w4, w5, w6, w7 = ld_global_nc_v4_u32(src + Int64(16))
            dst = (
                q_addr
                + row_in_tile * Int32(HEADS_PER_TILE * Q_ROW_BYTES)
                + head * Int32(Q_ROW_BYTES)
                + block * Int32(32)
            )
            if block < Int32(LATENT_DIM // 16):
                st_shared_v4_u32(dst, w0, w2, w4, w6)
                st_shared_v4_u32(dst + Int32(16), w1, w3, w5, w7)
            else:
                st_shared_v4_u32(dst, w0, w1, w2, w3)
                st_shared_v4_u32(dst + Int32(16), w4, w5, w6, w7)
            unit += Int32(32 * MATH_WARPS)

    @cute.jit
    def _scores(self, q_addr: Int32, kv_addr: Int32, warp: Int32, lane: Int32):
        """Both rows' sixteen heads x this warp's candidates; 4*ntiles each.

        One widened K fragment feeds the two rows' m16 tiles, so each key byte
        is widened once per warp per chunk.
        """
        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col_bytes = (lane >> Int32(4)) * Int32(16)
        q0 = q_addr + a_row * Int32(Q_ROW_BYTES) + a_col_bytes
        q1 = q0 + Int32(HEADS_PER_TILE * Q_ROW_BYTES)
        b_row = lane & Int32(7)
        b_col = (lane >> Int32(3)) * Int32(16)
        first = warp * Int32(self.warp_cands)
        k_lane = kv_addr + (first + b_row) * Int32(RECORD_BYTES) + b_col
        tid = lane & Int32(3)
        qk0 = [Float32(0.0) for _ in range(4 * self.ntiles)]
        qk1 = [Float32(0.0) for _ in range(4 * self.ntiles)]
        for g in cutlass.range_constexpr(GROUPS):
            part0 = [Float32(0.0) for _ in range(4 * self.ntiles)]
            part1 = [Float32(0.0) for _ in range(4 * self.ntiles)]
            for slab in cutlass.range_constexpr(GROUP // 64):
                base = g * GROUP + slab * 64
                kregs = []
                for nt in cutlass.range_constexpr(self.ntiles):
                    k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(
                        k_lane + Int32(nt * 8 * RECORD_BYTES + base)
                    )
                    kregs.append((k0, k1, k2, k3))
                for blk in cutlass.range_constexpr(4):
                    a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(
                        q0 + Int32((base + blk * 16) * 2)
                    )
                    a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(
                        q1 + Int32((base + blk * 16) * 2)
                    )
                    for nt in cutlass.range_constexpr(self.ntiles):
                        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(kregs[nt][blk])
                        (
                            part0[nt * 4], part0[nt * 4 + 1],
                            part0[nt * 4 + 2], part0[nt * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            part0[nt * 4], part0[nt * 4 + 1],
                            part0[nt * 4 + 2], part0[nt * 4 + 3],
                            a00, a01, a02, a03, b0, b1,
                        )
                        (
                            part1[nt * 4], part1[nt * 4 + 1],
                            part1[nt * 4 + 2], part1[nt * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            part1[nt * 4], part1[nt * 4 + 1],
                            part1[nt * 4 + 2], part1[nt * 4 + 3],
                            a10, a11, a12, a13, b0, b1,
                        )
            for nt in cutlass.range_constexpr(self.ntiles):
                c0 = first + Int32(nt * 8) + tid * Int32(2)
                s0 = ld_shared_f32(
                    kv_addr + c0 * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET + 4 * g)
                )
                s1 = ld_shared_f32(
                    kv_addr + (c0 + Int32(1)) * Int32(RECORD_BYTES)
                    + Int32(SCALE_OFFSET + 4 * g)
                )
                qk0[nt * 4] = qk0[nt * 4] + part0[nt * 4] * s0
                qk0[nt * 4 + 1] = qk0[nt * 4 + 1] + part0[nt * 4 + 1] * s1
                qk0[nt * 4 + 2] = qk0[nt * 4 + 2] + part0[nt * 4 + 2] * s0
                qk0[nt * 4 + 3] = qk0[nt * 4 + 3] + part0[nt * 4 + 3] * s1
                qk1[nt * 4] = qk1[nt * 4] + part1[nt * 4] * s0
                qk1[nt * 4 + 1] = qk1[nt * 4 + 1] + part1[nt * 4 + 1] * s1
                qk1[nt * 4 + 2] = qk1[nt * 4 + 2] + part1[nt * 4 + 2] * s0
                qk1[nt * 4 + 3] = qk1[nt * 4 + 3] + part1[nt * 4 + 3] * s1
        # BF16 raw tail, natural order, unscaled.
        for slab in cutlass.range_constexpr(2):
            kregs = []
            for nt in cutlass.range_constexpr(self.ntiles):
                k0, k1, k2, k3 = ldmatrix_m8n8x4_b16(
                    k_lane + Int32(nt * 8 * RECORD_BYTES + TAIL_OFFSET + slab * 64)
                )
                kregs.append((k0, k1, k2, k3))
            for half in cutlass.range_constexpr(2):
                a00, a01, a02, a03 = ldmatrix_m8n8x4_b16(
                    q0 + Int32((LATENT_DIM + slab * 32 + half * 16) * 2)
                )
                a10, a11, a12, a13 = ldmatrix_m8n8x4_b16(
                    q1 + Int32((LATENT_DIM + slab * 32 + half * 16) * 2)
                )
                for nt in cutlass.range_constexpr(self.ntiles):
                    (
                        qk0[nt * 4], qk0[nt * 4 + 1], qk0[nt * 4 + 2], qk0[nt * 4 + 3],
                    ) = mma_m16n8k16_f32_bf16(
                        qk0[nt * 4], qk0[nt * 4 + 1], qk0[nt * 4 + 2], qk0[nt * 4 + 3],
                        a00, a01, a02, a03,
                        kregs[nt][half * 2], kregs[nt][half * 2 + 1],
                    )
                    (
                        qk1[nt * 4], qk1[nt * 4 + 1], qk1[nt * 4 + 2], qk1[nt * 4 + 3],
                    ) = mma_m16n8k16_f32_bf16(
                        qk1[nt * 4], qk1[nt * 4 + 1], qk1[nt * 4 + 2], qk1[nt * 4 + 3],
                        a10, a11, a12, a13,
                        kregs[nt][half * 2], kregs[nt][half * 2 + 1],
                    )
        return qk0 + qk1

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        sm_scale_log2: Float32,
        q_stride_row_bytes: Int64,
        page_stride_bytes: Int64,
        page_table_stride: Int64,
        total_q: Int32,
        batch: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)
        tile_index, head_tile, _ = cute.arch.block_idx()
        tile_index = Int32(tile_index)
        head_base = Int32(head_tile) * Int32(HEADS_PER_TILE)

        # Tile -> request, scanning the per-request tile counts (two rows each).
        request = Int32(0)
        tile_valid = Int32(1)
        remaining = tile_index
        tile_offset = Int32(0)
        scan = Int32(0)
        found = Int32(0)
        while scan < batch:
            if found == Int32(0):
                scan_begin = Int32(cu_seqlens_q[scan])
                scan_end = Int32(cu_seqlens_q[scan + Int32(1)])
                scan_tiles = (scan_end - scan_begin + Int32(ROWS - 1)) // Int32(ROWS)
                if remaining < scan_tiles:
                    request = scan
                    tile_offset = remaining
                    found = Int32(1)
                else:
                    remaining -= scan_tiles
            scan += Int32(1)
        tile_valid = found
        query_begin = Int32(cu_seqlens_q[request])
        query_end = Int32(cu_seqlens_q[request + Int32(1)])
        query_length = query_end - query_begin
        query_start = query_begin + tile_offset * Int32(ROWS)
        cache_length = Int32(cache_seqlens[request])
        visible_max = Int32(0)
        for r in cutlass.range_constexpr(ROWS):
            row = query_start + r
            if row < total_q and row < query_end:
                end = cache_length - query_length + (row - query_begin) + Int32(1)
                if end > cache_length:
                    end = cache_length
                if end > visible_max:
                    visible_max = end
        active_chunks = (visible_max + Int32(self.chunk - 1)) // Int32(self.chunk)
        if tile_valid == Int32(0):
            active_chunks = Int32(0)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        kv_addr = shared_ptr_to_u32(storage.kv.data_ptr())
        q_addr = shared_ptr_to_u32(storage.q.data_ptr())
        p_addr = shared_ptr_to_u32(storage.p.data_ptr())
        red_addr = shared_ptr_to_u32(storage.reduce.data_ptr())
        mbar_base = storage.mbar.data_ptr()
        if tid == Int32(0):
            for s in cutlass.range_constexpr(self.stages * self.splits):
                cute.arch.mbarrier_init(mbar_base + s, Int32(1))
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(
                    mbar_base + Int32(self.stages * self.splits + s), Int32(MATH_WARPS)
                )
        cute.arch.barrier()

        if warp_id == Int32(MATH_WARPS):
            producer_phase = Int32(1)
            stage = Int32(0)
            table_base = request.to(Int64) * page_table_stride
            for local in cutlass.range(active_chunks, unroll=1):
                token_begin = Int32(local) * Int32(self.chunk)
                cute.arch.mbarrier_wait(
                    mbar_base + Int32(self.stages * self.splits + stage), phase=producer_phase
                )
                dst = kv_addr + stage * Int32(self.kv_stage_bytes)
                full = mbar_base + stage * Int32(self.splits)
                if lane == Int32(0):
                    for sp in cutlass.range_constexpr(self.splits):
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + sp, Int32(self.split_bytes)
                        )
                if token_begin + Int32(self.chunk) <= cache_length:
                    if lane == Int32(0):
                        page = token_begin // Int32(self.page_size)
                        physical = Int32(page_table[table_base + page.to(Int64)])
                        head = get_ptr_as_int64(
                            cache_bytes,
                            physical.to(Int64) * page_stride_bytes
                            + (token_begin - page * Int32(self.page_size)).to(Int64)
                            * Int64(RECORD_BYTES),
                        )
                        for sp in cutlass.range_constexpr(self.splits):
                            cp_async_bulk_g2s_mbar(
                                dst + sp * Int32(self.split_bytes),
                                head + sp * Int64(self.split_bytes),
                                Int32(self.split_bytes),
                                shared_ptr_to_u32(full + sp),
                            )
                else:
                    # Per-record copies; records past the cache read record zero
                    # and the consumer masks their scores, so a NaN-poisoned tail
                    # never reaches PV.
                    entry = lane
                    while entry < Int32(self.chunk):
                        token = token_begin + entry
                        if token >= cache_length:
                            token = Int32(0)
                        page = token // Int32(self.page_size)
                        physical = Int32(page_table[table_base + page.to(Int64)])
                        src = get_ptr_as_int64(
                            cache_bytes,
                            physical.to(Int64) * page_stride_bytes
                            + (token - page * Int32(self.page_size)).to(Int64)
                            * Int64(RECORD_BYTES),
                        )
                        sp = entry // Int32(self.split_records)
                        cp_async_bulk_g2s_mbar(
                            dst + entry * Int32(RECORD_BYTES), src, Int32(RECORD_BYTES),
                            shared_ptr_to_u32(full + sp),
                        )
                        entry += Int32(32)
                stage += Int32(1)
                if stage == Int32(self.stages):
                    stage = Int32(0)
                    producer_phase ^= Int32(1)
        else:
            self._stage_query(
                q_bytes, q_addr, query_start, head_base, total_q, query_end, tid,
                q_stride_row_bytes,
            )
            cute.arch.barrier(barrier_id=1, number_of_threads=self.math_threads)
            warp = warp_id
            # Per-row view of the query tile and of the softmax state.
            red_max = red_addr
            red_sum = red_addr + Int32(ROWS * MATH_WARPS * HEADS_PER_TILE)
            chunk_phase = Int32(0)
            stage = Int32(0)
            acc0 = cute.make_rmem_tensor(32, Float32)
            acc1 = cute.make_rmem_tensor(32, Float32)
            for i in cutlass.range_constexpr(32):
                acc0[i] = Float32(0.0)
                acc1[i] = Float32(0.0)
            # Running softmax state: m and l per (row, head-of-lane).
            stat = cute.make_rmem_tensor(4, Float32)  # m0(gid), m1(gid), l0, l1
            stat2 = cute.make_rmem_tensor(4, Float32)  # the +8 head of each row
            stat[0] = Float32(_MASK)
            stat[1] = Float32(_MASK)
            stat[2] = Float32(0.0)
            stat[3] = Float32(0.0)
            stat2[0] = Float32(_MASK)
            stat2[1] = Float32(_MASK)
            stat2[2] = Float32(0.0)
            stat2[3] = Float32(0.0)
            # Value operands: lane l (mod 16) supplies record row pi(l) so that
            # the transposed byte ldmatrix lands in the m16n8k16 B layout.
            l16 = lane & Int32(15)
            v_row = (
                (l16 & Int32(1))
                | ((l16 & Int32(4)) >> Int32(1))
                | ((l16 & Int32(8)) >> Int32(1))
                | ((l16 & Int32(2)) << Int32(2))
            )
            p_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
            p_col = (lane >> Int32(4)) * Int32(16)
            v_group = warp >> Int32(1)
            v_base = warp * Int32(VALUE_DIMS_PER_WARP)
            p_lane0 = (
                p_addr + v_group * Int32(self.p_group_bytes) + p_row * Int32(self.p_stride) + p_col
            )
            p_lane1 = p_lane0 + Int32(self.p_row_bytes)
            tid4 = lane & Int32(3)
            gid = lane >> Int32(2)
            for local in cutlass.range(active_chunks, unroll=1):
                chunk = Int32(local)
                chunk_begin = chunk * Int32(self.chunk)
                kv_addr_s = kv_addr + stage * Int32(self.kv_stage_bytes)
                full = mbar_base + stage * Int32(self.splits)
                # A warp starts on the copy splits its own candidates live in;
                # the value phase reads every split and catches up before PV.
                split_first = (warp * Int32(self.warp_cands)) // Int32(self.split_records)
                split_last = (
                    warp * Int32(self.warp_cands) + Int32(self.warp_cands - 1)
                ) // Int32(self.split_records)
                for sp in cutlass.range(split_first, split_last + Int32(1), unroll=1):
                    cute.arch.mbarrier_wait(full + sp, phase=chunk_phase)
                qk = self._scores(q_addr, kv_addr_s, warp, lane)
                qk0 = qk[: 4 * self.ntiles]
                qk1 = qk[4 * self.ntiles :]
                mx0 = Float32(_MASK)
                mx1 = Float32(_MASK)
                mx0b = Float32(_MASK)
                mx1b = Float32(_MASK)
                for nt in cutlass.range_constexpr(self.ntiles):
                    c0 = (
                        chunk_begin + warp * Int32(self.warp_cands)
                        + Int32(nt * 8) + tid4 * Int32(2)
                    )
                    for r in cutlass.range_constexpr(4):
                        qk0[nt * 4 + r] = qk0[nt * 4 + r] * sm_scale_log2
                        qk1[nt * 4 + r] = qk1[nt * 4 + r] * sm_scale_log2
                    for row in cutlass.range_constexpr(ROWS):
                        end = cache_length - query_length + (
                            query_start + row - query_begin
                        ) + Int32(1)
                        if end > cache_length:
                            end = cache_length
                        if row == 0:
                            if c0 >= end:
                                qk0[nt * 4] = Float32(_MASK)
                                qk0[nt * 4 + 2] = Float32(_MASK)
                            if c0 + Int32(1) >= end:
                                qk0[nt * 4 + 1] = Float32(_MASK)
                                qk0[nt * 4 + 3] = Float32(_MASK)
                        else:
                            if c0 >= end:
                                qk1[nt * 4] = Float32(_MASK)
                                qk1[nt * 4 + 2] = Float32(_MASK)
                            if c0 + Int32(1) >= end:
                                qk1[nt * 4 + 1] = Float32(_MASK)
                                qk1[nt * 4 + 3] = Float32(_MASK)
                    mx0 = fmax_f32(mx0, fmax_f32(qk0[nt * 4], qk0[nt * 4 + 1]))
                    mx1 = fmax_f32(mx1, fmax_f32(qk1[nt * 4], qk1[nt * 4 + 1]))
                    mx0b = fmax_f32(mx0b, fmax_f32(qk0[nt * 4 + 2], qk0[nt * 4 + 3]))
                    mx1b = fmax_f32(mx1b, fmax_f32(qk1[nt * 4 + 2], qk1[nt * 4 + 3]))
                for off in (1, 2):
                    mx0 = fmax_f32(mx0, cute.arch.shuffle_sync_bfly(mx0, offset=off))
                    mx1 = fmax_f32(mx1, cute.arch.shuffle_sync_bfly(mx1, offset=off))
                    mx0b = fmax_f32(mx0b, cute.arch.shuffle_sync_bfly(mx0b, offset=off))
                    mx1b = fmax_f32(mx1b, cute.arch.shuffle_sync_bfly(mx1b, offset=off))
                if tid4 == Int32(0):
                    st_shared_f32(
                        red_max + (Int32(0 * MATH_WARPS + warp) * HEADS_PER_TILE + gid)
                        * Int32(4), mx0,
                    )
                    st_shared_f32(
                        red_max + (Int32(0 * MATH_WARPS + warp) * HEADS_PER_TILE
                                   + gid + Int32(8)) * Int32(4), mx0b,
                    )
                    st_shared_f32(
                        red_max + (Int32(1 * MATH_WARPS + warp) * HEADS_PER_TILE + gid)
                        * Int32(4), mx1,
                    )
                    st_shared_f32(
                        red_max + (Int32(1 * MATH_WARPS + warp) * HEADS_PER_TILE
                                   + gid + Int32(8)) * Int32(4), mx1b,
                    )
                cute.arch.barrier(barrier_id=2, number_of_threads=self.math_threads)
                # This chunk's maximum over every warp, folded into the row's
                # previous maximum.  Every warp derives the same four values
                # from the same shared slots, so all of them rescale their
                # accumulators by the same four alphas and keep the same
                # running state; no second publish is needed.
                bm0 = stat[0]
                bm1 = stat2[0]
                bm2 = stat[1]
                bm3 = stat2[1]
                for w in cutlass.range_constexpr(MATH_WARPS):
                    base = red_max + (Int32(0 * MATH_WARPS + w) * HEADS_PER_TILE) * Int32(4)
                    bm0 = fmax_f32(bm0, ld_shared_f32(base + gid * Int32(4)))
                    bm1 = fmax_f32(bm1, ld_shared_f32(base + (gid + Int32(8)) * Int32(4)))
                    base = red_max + (Int32(1 * MATH_WARPS + w) * HEADS_PER_TILE) * Int32(4)
                    bm2 = fmax_f32(bm2, ld_shared_f32(base + gid * Int32(4)))
                    bm3 = fmax_f32(bm3, ld_shared_f32(base + (gid + Int32(8)) * Int32(4)))
                al0 = exp2_approx(stat[0] - bm0)
                al1 = exp2_approx(stat2[0] - bm1)
                al2 = exp2_approx(stat[1] - bm2)
                al3 = exp2_approx(stat2[1] - bm3)
                stat[0] = bm0
                stat2[0] = bm1
                stat[1] = bm2
                stat2[1] = bm3
                for j in cutlass.range_constexpr(32):
                    if j % 4 < 2:
                        acc0[j] = acc0[j] * al0
                        acc1[j] = acc1[j] * al2
                    else:
                        acc0[j] = acc0[j] * al1
                        acc1[j] = acc1[j] * al3
                # Probabilities, scaled copies for each value group, row sums.
                # The probabilities overwrite the scores in place: a score is
                # dead once every row sum is folded in, and the staging loop
                # below re-reads them per value group.
                sum0 = Float32(0.0)
                sum1 = Float32(0.0)
                sum0b = Float32(0.0)
                sum1b = Float32(0.0)
                for nt in cutlass.range_constexpr(self.ntiles):
                    qk0[nt * 4] = exp2_approx(qk0[nt * 4] - bm0)
                    qk0[nt * 4 + 1] = exp2_approx(qk0[nt * 4 + 1] - bm0)
                    qk0[nt * 4 + 2] = exp2_approx(qk0[nt * 4 + 2] - bm1)
                    qk0[nt * 4 + 3] = exp2_approx(qk0[nt * 4 + 3] - bm1)
                    qk1[nt * 4] = exp2_approx(qk1[nt * 4] - bm2)
                    qk1[nt * 4 + 1] = exp2_approx(qk1[nt * 4 + 1] - bm2)
                    qk1[nt * 4 + 2] = exp2_approx(qk1[nt * 4 + 2] - bm3)
                    qk1[nt * 4 + 3] = exp2_approx(qk1[nt * 4 + 3] - bm3)
                    sum0 = sum0 + qk0[nt * 4] + qk0[nt * 4 + 1]
                    sum0b = sum0b + qk0[nt * 4 + 2] + qk0[nt * 4 + 3]
                    sum1 = sum1 + qk1[nt * 4] + qk1[nt * 4 + 1]
                    sum1b = sum1b + qk1[nt * 4 + 2] + qk1[nt * 4 + 3]
                stat[2] = stat[2] * al0 + sum0
                stat2[2] = stat2[2] * al1 + sum0b
                stat[3] = stat[3] * al2 + sum1
                stat2[3] = stat2[3] * al3 + sum1b
                # P_g[row][h][c] = p[row][h][c] * scale[c][g], one staging copy
                # per value group, read back by the value warps.
                for nt in cutlass.range_constexpr(self.ntiles):
                    lc = warp * Int32(self.warp_cands) + Int32(nt * 8) + tid4 * Int32(2)
                    sa0, sa1, sa2, sa3 = ld_shared_v4_f32(
                        kv_addr_s + lc * Int32(RECORD_BYTES) + Int32(SCALE_OFFSET)
                    )
                    sb0, sb1, sb2, sb3 = ld_shared_v4_f32(
                        kv_addr_s + (lc + Int32(1)) * Int32(RECORD_BYTES)
                        + Int32(SCALE_OFFSET)
                    )
                    sa = (sa0, sa1, sa2, sa3)
                    sb = (sb0, sb1, sb2, sb3)
                    for g in cutlass.range_constexpr(GROUPS):
                        base_row0 = (
                            p_addr + Int32(0 * self.p_row_bytes + g * self.p_group_bytes)
                            + gid * Int32(self.p_stride) + lc * Int32(2)
                        )
                        base_row1 = (
                            p_addr + Int32(1 * self.p_row_bytes + g * self.p_group_bytes)
                            + gid * Int32(self.p_stride) + lc * Int32(2)
                        )
                        st_shared_u32(
                            base_row0,
                            pack_f32x2_to_bfloat2(
                                qk0[nt * 4] * sa[g], qk0[nt * 4 + 1] * sb[g]
                            ),
                        )
                        st_shared_u32(
                            base_row0 + Int32(8 * self.p_stride),
                            pack_f32x2_to_bfloat2(
                                qk0[nt * 4 + 2] * sa[g], qk0[nt * 4 + 3] * sb[g]
                            ),
                        )
                        st_shared_u32(
                            base_row1,
                            pack_f32x2_to_bfloat2(
                                qk1[nt * 4] * sa[g], qk1[nt * 4 + 1] * sb[g]
                            ),
                        )
                        st_shared_u32(
                            base_row1 + Int32(8 * self.p_stride),
                            pack_f32x2_to_bfloat2(
                                qk1[nt * 4 + 2] * sa[g], qk1[nt * 4 + 3] * sb[g]
                            ),
                        )
                if self.splits > 1:
                    # The value phase reads every split; catch up on the ones
                    # this warp did not need for its own candidates.
                    for sp in cutlass.range_constexpr(self.splits):
                        if sp < split_first or sp > split_last:
                            cute.arch.mbarrier_wait(full + sp, phase=chunk_phase)
                cute.arch.barrier(barrier_id=2, number_of_threads=self.math_threads)
                # Value MMA: this warp owns 64 latent dims of both query rows.
                for ks in cutlass.range_constexpr(self.chunk // 16):
                    p0a0, p0a1, p0a2, p0a3 = ldmatrix_m8n8x4_b16(p_lane0 + Int32(ks * 32))
                    p1a0, p1a1, p1a2, p1a3 = ldmatrix_m8n8x4_b16(p_lane1 + Int32(ks * 32))
                    v_lane = (
                        kv_addr_s + (Int32(ks * 16) + v_row) * Int32(RECORD_BYTES) + v_base
                    )
                    for blk in cutlass.range_constexpr(VALUE_DIMS_PER_WARP // 16):
                        f0, f1 = ldmatrix_m16n16x1_trans_b8(v_lane + Int32(blk * 16))
                        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
                        b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
                        n0 = blk * 2
                        (
                            acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            acc0[n0 * 4], acc0[n0 * 4 + 1], acc0[n0 * 4 + 2], acc0[n0 * 4 + 3],
                            p0a0, p0a1, p0a2, p0a3, b0, b1,
                        )
                        n1 = blk * 2 + 1
                        (
                            acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            acc0[n1 * 4], acc0[n1 * 4 + 1], acc0[n1 * 4 + 2], acc0[n1 * 4 + 3],
                            p0a0, p0a1, p0a2, p0a3, b2, b3,
                        )
                        (
                            acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            acc1[n0 * 4], acc1[n0 * 4 + 1], acc1[n0 * 4 + 2], acc1[n0 * 4 + 3],
                            p1a0, p1a1, p1a2, p1a3, b0, b1,
                        )
                        (
                            acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                        ) = mma_m16n8k16_f32_bf16(
                            acc1[n1 * 4], acc1[n1 * 4 + 1], acc1[n1 * 4 + 2], acc1[n1 * 4 + 3],
                            p1a0, p1a1, p1a2, p1a3, b2, b3,
                        )
                if self.stages > 1:
                    # P is single-buffered: no warp may stage the next chunk's
                    # probabilities before every warp has read this chunk's,
                    # and with a deeper ring the release only frees the KV
                    # stage, so close that window here.
                    cute.arch.barrier(barrier_id=2, number_of_threads=self.math_threads)
                # Every warp releases the stage once its last KV read is done.
                cute.arch.sync_warp()
                if lane == Int32(0):
                    cute.arch.mbarrier_arrive(
                        mbar_base + Int32(self.stages * self.splits + stage)
                    )
                stage += Int32(1)
                if stage == Int32(self.stages):
                    stage = Int32(0)
                    chunk_phase ^= Int32(1)

            # Epilogue: score warps publish their per-candidate row sums, then
            # every warp normalises and stores its own value dims.  Each lane
            # accumulated only its own candidates, so the four lanes of a
            # group add up first.
            l0 = stat[2]
            l0b = stat2[2]
            l1 = stat[3]
            l1b = stat2[3]
            for off in (1, 2):
                l0 = l0 + cute.arch.shuffle_sync_bfly(l0, offset=off)
                l0b = l0b + cute.arch.shuffle_sync_bfly(l0b, offset=off)
                l1 = l1 + cute.arch.shuffle_sync_bfly(l1, offset=off)
                l1b = l1b + cute.arch.shuffle_sync_bfly(l1b, offset=off)
            if tid4 == Int32(0):
                st_shared_f32(
                    red_sum + (Int32(0 * MATH_WARPS + warp) * HEADS_PER_TILE + gid)
                    * Int32(4), l0,
                )
                st_shared_f32(
                    red_sum + (Int32(0 * MATH_WARPS + warp) * HEADS_PER_TILE
                               + gid + Int32(8)) * Int32(4), l0b,
                )
                st_shared_f32(
                    red_sum + (Int32(1 * MATH_WARPS + warp) * HEADS_PER_TILE + gid)
                    * Int32(4), l1,
                )
                st_shared_f32(
                    red_sum + (Int32(1 * MATH_WARPS + warp) * HEADS_PER_TILE
                               + gid + Int32(8)) * Int32(4), l1b,
                )
            cute.arch.barrier(barrier_id=2, number_of_threads=self.math_threads)
            t = cute.make_rmem_tensor(4, Float32)
            for r in cutlass.range_constexpr(ROWS):
                t[r * 2] = Float32(0.0)
                t[r * 2 + 1] = Float32(0.0)
            for w in cutlass.range_constexpr(MATH_WARPS):
                for r in cutlass.range_constexpr(ROWS):
                    base = red_sum + (Int32(r * MATH_WARPS + w) * HEADS_PER_TILE) * Int32(4)
                    t[r * 2] = t[r * 2] + ld_shared_f32(base + gid * Int32(4))
                    t[r * 2 + 1] = t[r * 2 + 1] + ld_shared_f32(
                        base + (gid + Int32(8)) * Int32(4)
                    )
            inv = cute.make_rmem_tensor(4, Float32)
            for i in cutlass.range_constexpr(4):
                inv[i] = Float32(0.0)
                if t[i] > Float32(0.0):
                    inv[i] = Float32(1.0) / t[i]
            for row in cutlass.range_constexpr(ROWS):
                row_id = query_start + row
                if row_id < total_q and row_id < query_end and tile_valid != Int32(0):
                    h0 = head_base + gid
                    h1 = h0 + Int32(8)
                    out0 = get_ptr_as_int64(
                        output,
                        (row_id.to(Int64) * Int64(self.num_heads) + h0.to(Int64))
                        * Int64(VALUE_DIM) + (v_base + tid4 * Int32(2)).to(Int64),
                    )
                    out1 = out0 + Int64(8 * VALUE_DIM * 2)
                    acc = acc0 if row == 0 else acc1
                    for j in cutlass.range_constexpr(8):
                        st_global_u32(
                            out0 + Int64(j * 16),
                            pack_f32x2_to_bfloat2(
                                acc[j * 4] * inv[row * 2], acc[j * 4 + 1] * inv[row * 2]
                            ),
                        )
                        st_global_u32(
                            out1 + Int64(j * 16),
                            pack_f32x2_to_bfloat2(
                                acc[j * 4 + 2] * inv[row * 2 + 1],
                                acc[j * 4 + 3] * inv[row * 2 + 1],
                            ),
                        )
            if warp == Int32(0) and tid4 == Int32(0):
                for row in cutlass.range_constexpr(ROWS):
                    row_id = query_start + row
                    if row_id < total_q and row_id < query_end and tile_valid != Int32(0):
                        lse0 = Float32(-Float32.inf)
                        if t[row * 2] > Float32(0.0):
                            lse0 = (log2_approx(t[row * 2]) + stat[row]) * Float32(
                                0.6931471805599453
                            )
                        lse1 = Float32(-Float32.inf)
                        if t[row * 2 + 1] > Float32(0.0):
                            lse1 = (log2_approx(t[row * 2 + 1]) + stat2[row]) * Float32(
                                0.6931471805599453
                            )
                        final_lse[row_id, head_base + gid] = lse0
                        final_lse[row_id, head_base + gid + Int32(8)] = lse1


__all__ = ["DenseMlaFp8SharedBPrefillKernel", "smem_bytes"]
