"""Narrow-width DeepSeek V4 Flash MXFP4 prefill kernels for GB10/SM121.

Statically isolated to the shipped V4 Flash m4096 prefill bucket on SM121:
K=4096, E=256, top-6, N in {256, 288, 320}, width=128, ordered FP32 route
output, non-coded UE8M0 scales, fc1_k_splits=1, fc2_n_splits=1.

Reuses V41SlicePipeline's exact caller-owned scratch and output ABI while
preserving bit-identical numerical semantics (ascending K128 FC1 accumulation,
exact BF16-rounded SwiGLU + 1e-4 MXFP8 quantization floor, and ascending
128-K slice-folded FP32 FC2 accumulation).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    create_l2_evict_first_policy,
    e2m1x8_to_qmma_e2m1x8,
    fmax_f32,
    fmin_f32,
    get_ptr_as_int64,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    max_abs_32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_global_u8,
    st_global_u16,
    st_global_u32,
    st_global_v4_u32,
    st_shared_u32,
)
from b12x.gemm.bf16_gemv._skinny import _bf16_hi, _bf16_lo
from b12x.moe._shared.kernels.w4a8_v41_slice import fadd_rn_f32

# Layout offsets inside the caller-owned metadata[routes, 19] Int32 buffer
# (total capacity 24576 * 19 = 466944 Int32 words = 1.87 MB):
#   [0]                : total active 128-row chunks (<= 448)
#   [1]                : invalid route count (0..24576)
#   [16 .. 16+448)     : tile_code[t] = expert | (chunk << 16)
#   [1024 .. 25600)    : pair_tok[pos] = p // topk
#   [26624 .. 51200)   : pair_route[pos] = p
#   [52224 .. 76800)   : invalid_routes[idx] = p
_META_TILES_OFF = 16
_META_TOK_OFF = 1024
_META_ROUTE_OFF = 26624
_META_INVALID_OFF = 52224
_TILE_M = 128


def dsv4f_narrow_prefill_qualified(
    *,
    capacity: int,
    width: int,
    atomic_tokens: bool,
    experts: int,
    topk: int,
    intermediate: int,
    hidden: int,
    fc2_n_splits: int,
    scale_codes: bool,
    fc1_k_splits: int,
    compute_capability=None,
) -> bool:
    """Qualification selector for narrow DeepSeek V4 Flash m4096 prefill on SM121."""
    if not (
        capacity == 4096
        and width == 128
        and not atomic_tokens
        and experts == 256
        and topk == 6
        and intermediate in (256, 288, 320)
        and hidden == 4096
        and fc2_n_splits == 1
        and not scale_codes
        and fc1_k_splits == 1
    ):
        return False
    if compute_capability is None:
        from b12x._lib.gating import get_compute_capability

        compute_capability = get_compute_capability()
    return compute_capability is not None and tuple(compute_capability) == (12, 1)


def dsv4f_act_row_bytes(intermediate: int) -> int:
    """16-byte-aligned compact MXFP8 row stride: N payload + 4-byte-aligned scales."""
    scale_words = (intermediate // 32 + 3) // 4
    return (intermediate + scale_words * 4 + 15) // 16 * 16


@dsl_user_op
def _atom_add_shared_i32(address: Int32, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(address).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cp_async4_l2hint(
    smem_addr: Int32, gmem_addr: Int64, cache_policy: Uint64, *, loc=None, ip=None
):
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
def _st_shared_v2_f32(
    smem_addr: Int32, v0: Float32, v1: Float32, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_l2hint_v4_u32(
    addr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    cache_policy: Uint64,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
            Uint64(cache_policy).ir_value(loc=loc, ip=ip),
        ],
        "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;",
        "l,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cvt_rn_bf16x2_f32(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """IEEE-754 RNE f32x2 -> bf16x2 (low 16b = x0, high 16b = x1), matching .to(BFloat16)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Dsv4fNarrowPrep:
    """Single-CTA 1024-thread route histogram, 128-row chunk planner, and scatter."""

    threads = 1024

    def __init__(self, *, capacity: int, experts: int, topk: int):
        self.capacity = int(capacity)
        self.experts = int(experts)
        self.topk = int(topk)
        self.routes = self.capacity * self.topk
        self.max_tiles = (self.routes + _TILE_M - 1) // _TILE_M + self.experts

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "counts": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "cursor": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "tiles": cute.struct.Align[cute.struct.MemRange[Int32, self.experts], 16],
            "inv_cnt": cute.struct.Align[cute.struct.MemRange[Int32, 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        routing: cute.Tensor,
        live: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            ids, routing, live, counts, prefixes, metadata, grouped, rows, self._storage()
        ).launch(grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        ids: cute.Tensor,
        routing: cute.Tensor,
        live: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        rows: Int32,
        Storage: cutlass.Constexpr,
    ):
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        s_counts = storage.counts.get_tensor(cute.make_layout((self.experts,)))
        s_cursor = storage.cursor.get_tensor(cute.make_layout((self.experts,)))
        s_tiles = storage.tiles.get_tensor(cute.make_layout((self.experts,)))
        s_inv = storage.inv_cnt.get_tensor(cute.make_layout((4,)))
        counts_addr = shared_ptr_to_u32(storage.counts.data_ptr())
        cursor_addr = shared_ptr_to_u32(storage.cursor.data_ptr())
        inv_addr = shared_ptr_to_u32(storage.inv_cnt.data_ptr())

        meta_flat = cute.make_tensor(
            metadata.iterator, cute.make_layout(self.routes * 19)
        )
        rows_clamped = max(Int32(0), min(rows, Int32(self.capacity)))
        pairs = rows_clamped * Int32(self.topk)
        e_count = Int32(self.experts)

        if tidx == Int32(0):
            live[0] = rows_clamped
            s_inv[0] = Int32(0)
        if tidx < e_count:
            s_counts[tidx] = Int32(0)
        cute.arch.sync_threads()

        # Pass 1: shared-memory histogram over live routes.
        p = tidx
        while p < pairs:
            e = Int32(ids[p])
            if (e >= Int32(0)) & (e < e_count):
                _atom_add_shared_i32(counts_addr + e * Int32(4), Int32(1))
            p = p + Int32(self.threads)
        cute.arch.sync_threads()

        # Exclusive prefix scan over the 256 experts (routes and 128-row chunks).
        n = Int32(0)
        tiles_e = Int32(0)
        if tidx < e_count:
            n = Int32(s_counts[tidx])
            tiles_e = (n + Int32(_TILE_M - 1)) // Int32(_TILE_M)
            s_cursor[tidx] = n
            s_tiles[tidx] = tiles_e
        cute.arch.sync_threads()

        step = 1
        while step < self.experts:
            a = Int32(0)
            b = Int32(0)
            if (tidx < e_count) & (tidx >= Int32(step)):
                a = Int32(s_cursor[tidx - Int32(step)])
                b = Int32(s_tiles[tidx - Int32(step)])
            cute.arch.sync_threads()
            if (tidx < e_count) & (tidx >= Int32(step)):
                s_cursor[tidx] = Int32(s_cursor[tidx]) + a
                s_tiles[tidx] = Int32(s_tiles[tidx]) + b
            cute.arch.sync_threads()
            step = step * 2

        if tidx < e_count:
            offset = Int32(s_cursor[tidx]) - n
            first_tile = Int32(s_tiles[tidx]) - tiles_e
            counts[tidx] = n
            prefixes[tidx, 0] = offset
            prefixes[tidx, 1] = first_tile
            s_cursor[tidx] = offset
            if n > Int32(0):
                for t in cutlass.range(tiles_e, unroll=1):
                    meta_flat[Int32(_META_TILES_OFF) + first_tile + Int32(t)] = (
                        tidx | (Int32(t) << Int32(16))
                    )
            if tidx == e_count - Int32(1):
                meta_flat[0] = first_tile + tiles_e
        cute.arch.sync_threads()

        # Pass 2: scatter grouped route metadata and collect invalid routes.
        p = tidx
        while p < pairs:
            e = Int32(ids[p])
            if (e >= Int32(0)) & (e < e_count):
                pos = _atom_add_shared_i32(cursor_addr + e * Int32(4), Int32(1))
                meta_flat[Int32(_META_TOK_OFF) + pos] = p // Int32(self.topk)
                meta_flat[Int32(_META_ROUTE_OFF) + pos] = p
                grouped[pos] = routing[p]
            else:
                inv_idx = _atom_add_shared_i32(inv_addr, Int32(1))
                meta_flat[Int32(_META_INVALID_OFF) + inv_idx] = p
            p = p + Int32(self.threads)
        cute.arch.sync_threads()

        if tidx == Int32(0):
            meta_flat[1] = Int32(s_inv[0])


class Dsv4fNarrowFc1:
    """Expert-stationary M128 x N64 FC1 + SwiGLU + MXFP8 quantize with 3-stage K64 ring.

    SMEM = 38.0 KiB -> fits 2 CTAs/SM on SM121.
    Uses L2::evict_first on one-pass W13/S13 weights so input wire rows stay L2-resident.
    """

    threads = 256
    tile_m = _TILE_M
    tile_n = 64
    k_step = 64
    ring = 3
    mid_stride = 144  # 64 bf16 (128 B) + 16 B padding -> 0 bank conflicts

    def __init__(
        self,
        *,
        capacity: int,
        topk: int,
        experts: int,
        hidden: int,
        intermediate: int,
        exact_storage: bool,
    ):
        self.capacity = int(capacity)
        self.topk = int(topk)
        self.routes = self.capacity * self.topk
        self.experts = int(experts)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.exact_storage = bool(exact_storage)
        self.kernel_intermediate = (
            self.intermediate if self.exact_storage else (self.intermediate + 127) // 128 * 128
        )
        self.n_blocks = (self.intermediate + self.tile_n - 1) // self.tile_n
        self.k_tiles = self.hidden // 128
        self.a_bytes = self.tile_m * self.k_step  # 8192 B
        self.w_bytes = 2 * self.tile_n * (self.k_step // 2)  # 4096 B (up + gate)
        self.xs_bytes = self.tile_m * 4  # 512 B per K128
        self.s13_bytes = 2 * self.tile_n * 4  # 512 B per K128 (up + gate)
        self.act_row_bytes = dsv4f_act_row_bytes(self.intermediate)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "a": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.ring * self.a_bytes], 1024],
            "w": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.ring * self.w_bytes], 1024],
            "xs": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.xs_bytes], 16],
            "s13": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * self.s13_bytes], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        partial: cute.Tensor,
        max_tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x, xs, w13, s13, counts, prefixes, metadata, grouped, partial, self._storage()
        ).launch(
            grid=(self.n_blocks, max_tiles, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _load_a(
        self,
        step: Int32,
        stage: Int32,
        sa: Int32,
        src: cute.Tensor,
        ch: Int32,
        r_base: Int32,
    ):
        a_stage = sa + stage * Int32(self.a_bytes)
        k_off = Int64(step) * Int64(self.k_step)
        for q in cutlass.range_constexpr(2):
            if src[q] >= Int64(0):
                row = r_base + Int32(64 * q)
                sw_ch = ch ^ (row & Int32(2))
                cp_async4_shared_global(
                    a_stage + row * Int32(self.k_step) + sw_ch * Int32(16),
                    src[q] + k_off,
                )

    @cute.jit
    def _load_xs(self, kt: Int32, s_xs: Int32, s_src: Int64, tidx: Int32):
        if s_src >= Int64(0):
            cp_async_u32_shared_global(
                s_xs + (kt & Int32(1)) * Int32(self.xs_bytes) + tidx * Int32(4),
                s_src + Int64(kt) * Int64(4),
            )

    @cute.jit
    def _load_w13(
        self,
        kt: Int32,
        k_half: cutlass.Constexpr,
        stage: Int32,
        sw: Int32,
        w13_base: Int64,
        expert: Int64,
        n0: Int32,
        tidx: Int32,
        w_policy: Uint64,
    ):
        proj = tidx // Int32(128)  # 0 = UP, 1 = GATE
        rem = tidx % Int32(128)
        kb_local = rem // Int32(64)
        chunk_local = (rem // Int32(32)) % Int32(2)
        lane_w = rem % Int32(32)
        kb = Int32(k_half * 2) + kb_local
        n = n0 + chunk_local * Int32(32)
        dst = sw + stage * Int32(self.w_bytes) + tidx * Int32(16)
        if const_expr(self.exact_storage):
            if n < Int32(self.intermediate):
                tile = n // Int32(128)
                tile_rows = Int32(128)
                if tile == Int32(self.intermediate // 128):
                    tile_rows = Int32(self.intermediate % 128)
                word = (
                    expert * Int64(self.kernel_intermediate * self.hidden // 4)
                    + Int64(proj) * Int64(self.intermediate * (self.hidden // 8))
                    + Int64(tile) * Int64(128 * (self.hidden // 8))
                    + Int64(kt * Int32(4) + kb) * Int64((tile_rows // Int32(32)) * Int32(128))
                    + Int64(((n % Int32(128)) // Int32(32)) * Int32(128) + lane_w * Int32(4))
                )
                _cp_async4_l2hint(dst, w13_base + word * Int64(4), w_policy)
        else:
            n_full = n + proj * Int32(self.kernel_intermediate)
            tile = (n_full // Int32(256)) * Int32(self.k_tiles) + kt
            word = (
                expert * Int64(self.kernel_intermediate * self.hidden // 4)
                + Int64(tile) * Int64(4096)
                + Int64(kb * Int32(1024) + ((n_full % Int32(256)) // Int32(32)) * Int32(128) + lane_w * Int32(4))
            )
            _cp_async4_l2hint(dst, w13_base + word * Int64(4), w_policy)

    @cute.jit
    def _load_s13(
        self,
        kt: Int32,
        s_s13: Int32,
        s13_base: Int64,
        expert: Int64,
        n0: Int32,
        tidx: Int32,
        w_policy: Uint64,
    ):
        if tidx < Int32(32):
            proj = tidx // Int32(16)  # 0 = UP, 1 = GATE
            idx = tidx % Int32(16)
            n = n0 + idx * Int32(4)
            dst = s_s13 + (kt & Int32(1)) * Int32(self.s13_bytes) + tidx * Int32(16)
            if const_expr(self.exact_storage):
                if n < Int32(self.intermediate):
                    tile = n // Int32(128)
                    tile_rows = Int32(128)
                    if tile == Int32(self.intermediate // 128):
                        tile_rows = Int32(self.intermediate % 128)
                    word = (
                        expert * Int64(self.kernel_intermediate * self.hidden // 64)
                        + Int64(proj) * Int64(self.intermediate * (self.hidden // 128))
                        + Int64(tile) * Int64(self.hidden)
                        + Int64(kt * tile_rows + (n % Int32(128)))
                    )
                    _cp_async4_l2hint(dst, s13_base + word * Int64(4), w_policy)
            else:
                n_full = n + proj * Int32(self.kernel_intermediate)
                tile = (n_full // Int32(256)) * Int32(self.k_tiles) + kt
                word = (
                    expert * Int64(self.kernel_intermediate * self.hidden // 64)
                    + Int64(tile) * Int64(256)
                    + Int64(n_full % Int32(256))
                )
                _cp_async4_l2hint(dst, s13_base + word * Int64(4), w_policy)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        partial: cute.Tensor,
        Storage: cutlass.Constexpr,
    ):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)
        meta_flat = cute.make_tensor(
            metadata.iterator, cute.make_layout(self.routes * 19)
        )
        if tile < Int32(meta_flat[0]):
            code = Int32(meta_flat[Int32(_META_TILES_OFF) + tile])
            e = code & Int32(0xFFFF)
            row0 = (code >> Int32(16)) * Int32(_TILE_M)
            live = Int32(counts[e]) - row0
            if live > Int32(_TILE_M):
                live = Int32(_TILE_M)
            first = Int32(prefixes[e, 0]) + row0
            expert = Int64(e)
            n0 = n_blk * Int32(self.tile_n)

            tidx = Int32(cute.arch.thread_idx()[0])
            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            g = lane // Int32(4)
            c = lane % Int32(4)
            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)

            sa = shared_ptr_to_u32(storage.a.data_ptr())
            sw = shared_ptr_to_u32(storage.w.data_ptr())
            s_xs = shared_ptr_to_u32(storage.xs.data_ptr())
            s_s13 = shared_ptr_to_u32(storage.s13.data_ptr())

            x_base = get_ptr_as_int64(x, Int64(0))
            xs_base = get_ptr_as_int64(xs, Int64(0))
            w13_base = get_ptr_as_int64(w13, Int64(0))
            s13_base = get_ptr_as_int64(s13, Int64(0))
            x_row_bytes = Int64(x.stride[0]) * Int64(4)
            xs_row_bytes = Int64(xs.stride[0])
            w_policy = create_l2_evict_first_policy()

            ch = tidx % Int32(4)
            r_base = tidx // Int32(4)
            src = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int64)
            for q in cutlass.range_constexpr(2):
                row = r_base + Int32(64 * q)
                src[q] = Int64(-1)
                if row < live:
                    tok = Int32(meta_flat[Int32(_META_TOK_OFF) + first + row])
                    src[q] = x_base + Int64(tok) * x_row_bytes + Int64(ch) * Int64(16)
            s_src = Int64(-1)
            if tidx < live:
                tok = Int32(meta_flat[Int32(_META_TOK_OFF) + first + tidx])
                s_src = xs_base + Int64(tok) * xs_row_bytes

            # Prologue: stage K64 steps 0 and 1 (plus K128 scales for kt=0).
            self._load_a(Int32(0), Int32(0), sa, src, ch, r_base)
            self._load_w13(Int32(0), 0, Int32(0), sw, w13_base, expert, n0, tidx, w_policy)
            self._load_xs(Int32(0), s_xs, s_src, tidx)
            self._load_s13(Int32(0), s_s13, s13_base, expert, n0, tidx, w_policy)
            cute.arch.cp_async_commit_group()

            self._load_a(Int32(1), Int32(1), sa, src, ch, r_base)
            self._load_w13(Int32(0), 1, Int32(1), sw, w13_base, expert, n0, tidx, w_policy)
            cute.arch.cp_async_commit_group()

            active_n = (n0 + warp_n * Int32(32)) < Int32(self.intermediate)
            active_mt0 = (warp_m * Int32(32) < live) & active_n
            active_mt1 = (warp_m * Int32(32) + Int32(16) < live) & active_n

            gate = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            up = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            for v in cutlass.range_constexpr(32):
                gate[v] = Float32(0.0)
                up[v] = Float32(0.0)

            asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
            wsc_u = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
            wsc_g = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
            for v in cutlass.range_constexpr(2):
                asc[v] = Uint32(0)
            for v in cutlass.range_constexpr(4):
                wsc_u[v] = Uint32(0)
                wsc_g[v] = Uint32(0)

            row_m0_lo = warp_m * Int32(32) + g
            row_m0_hi = row_m0_lo + Int32(8)
            row_m1_lo = row_m0_lo + Int32(16)
            row_m1_hi = row_m0_lo + Int32(24)
            srow_m0 = warp_m * Int32(32) + g + (c & Int32(1)) * Int32(8)
            srow_m1 = srow_m0 + Int32(16)

            for kt in cutlass.range(self.k_tiles, unroll=1):
                for k_half in cutlass.range_constexpr(2):
                    step = kt * Int32(2) + Int32(k_half)
                    stage = step % Int32(self.ring)
                    pf_stage = (step + Int32(2)) % Int32(self.ring)
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()

                    if kt + Int32(1) < Int32(self.k_tiles):
                        self._load_a(step + Int32(2), pf_stage, sa, src, ch, r_base)
                        self._load_w13(
                            kt + Int32(1), k_half, pf_stage, sw, w13_base, expert, n0, tidx, w_policy
                        )
                        if const_expr(k_half == 0):
                            self._load_xs(kt + Int32(1), s_xs, s_src, tidx)
                            self._load_s13(kt + Int32(1), s_s13, s13_base, expert, n0, tidx, w_policy)
                    cute.arch.cp_async_commit_group()

                    if active_mt0:
                        if const_expr(k_half == 0):
                            xs_slot = s_xs + (kt & Int32(1)) * Int32(self.xs_bytes)
                            s13_slot = s_s13 + (kt & Int32(1)) * Int32(self.s13_bytes)
                            asc[0] = ld_shared_u32(xs_slot + srow_m0 * Int32(4))
                            if active_mt1:
                                asc[1] = ld_shared_u32(xs_slot + srow_m1 * Int32(4))
                            for nf in cutlass.range_constexpr(4):
                                scol = warp_n * Int32(32) + Int32(nf * 8) + g
                                wsc_u[nf] = ld_shared_u32(s13_slot + scol * Int32(4))
                                wsc_g[nf] = ld_shared_u32(
                                    s13_slot + Int32(self.tile_n * 4) + scol * Int32(4)
                                )

                        a_stage = sa + stage * Int32(self.a_bytes)
                        w_stage = sw + stage * Int32(self.w_bytes)
                        for kb_local in cutlass.range_constexpr(2):
                            kb = k_half * 2 + kb_local
                            u_words = ld_shared_v4_u32(
                                w_stage + (Int32(kb_local * 2) + warp_n) * Int32(512) + lane * Int32(16)
                            )
                            g_words = ld_shared_v4_u32(
                                w_stage + Int32(2048) + (Int32(kb_local * 2) + warp_n) * Int32(512) + lane * Int32(16)
                            )
                            chunk_a = Int32(2 * kb_local) + (c >> Int32(1))
                            sub_a = (c & Int32(1)) * Int32(8)

                            # mt = 0
                            a0, a2 = ld_shared_v2_u32(
                                a_stage + row_m0_lo * Int32(self.k_step)
                                + (chunk_a ^ (row_m0_lo & Int32(2))) * Int32(16) + sub_a
                            )
                            a1, a3 = ld_shared_v2_u32(
                                a_stage + row_m0_hi * Int32(self.k_step)
                                + (chunk_a ^ (row_m0_hi & Int32(2))) * Int32(16) + sub_a
                            )
                            for nf in cutlass.range_constexpr(4):
                                u0, u1 = e2m1x8_to_qmma_e2m1x8(u_words[nf])
                                g0, g1 = e2m1x8_to_qmma_e2m1x8(g_words[nf])
                                f = 4 * nf
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                    gate[f], gate[f + 1], gate[f + 2], gate[f + 3],
                                    a0, a1, a2, a3, g0, g1, asc[0], wsc_g[nf],
                                    bid_a=kb, bid_b=kb,
                                )
                                gate[f] = d0
                                gate[f + 1] = d1
                                gate[f + 2] = d2
                                gate[f + 3] = d3
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                    up[f], up[f + 1], up[f + 2], up[f + 3],
                                    a0, a1, a2, a3, u0, u1, asc[0], wsc_u[nf],
                                    bid_a=kb, bid_b=kb,
                                )
                                up[f] = d0
                                up[f + 1] = d1
                                up[f + 2] = d2
                                up[f + 3] = d3

                            # mt = 1
                            if active_mt1:
                                a0, a2 = ld_shared_v2_u32(
                                    a_stage + row_m1_lo * Int32(self.k_step)
                                    + (chunk_a ^ (row_m1_lo & Int32(2))) * Int32(16) + sub_a
                                )
                                a1, a3 = ld_shared_v2_u32(
                                    a_stage + row_m1_hi * Int32(self.k_step)
                                    + (chunk_a ^ (row_m1_hi & Int32(2))) * Int32(16) + sub_a
                                )
                                for nf in cutlass.range_constexpr(4):
                                    u0, u1 = e2m1x8_to_qmma_e2m1x8(u_words[nf])
                                    g0, g1 = e2m1x8_to_qmma_e2m1x8(g_words[nf])
                                    f = 16 + 4 * nf
                                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                        gate[f], gate[f + 1], gate[f + 2], gate[f + 3],
                                        a0, a1, a2, a3, g0, g1, asc[1], wsc_g[nf],
                                        bid_a=kb, bid_b=kb,
                                    )
                                    gate[f] = d0
                                    gate[f + 1] = d1
                                    gate[f + 2] = d2
                                    gate[f + 3] = d3
                                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                        up[f], up[f + 1], up[f + 2], up[f + 3],
                                        a0, a1, a2, a3, u0, u1, asc[1], wsc_u[nf],
                                        bid_a=kb, bid_b=kb,
                                    )
                                    up[f] = d0
                                    up[f + 1] = d1
                                    up[f + 2] = d2
                                    up[f + 3] = d3

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Epilogue: exact BF16-rounded SwiGLU * routing weight into drained `a` SMEM.
            if active_mt0:
                for mt in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(16 * mt + 8 * half) + g
                        if row < live:
                            rw = Float32(grouped[Int64(first) + Int64(row)])
                            for nf in cutlass.range_constexpr(4):
                                f = 4 * (4 * mt + nf) + 2 * half
                                gv0 = gate[f].to(cutlass.BFloat16).to(Float32)
                                uv0 = up[f].to(cutlass.BFloat16).to(Float32)
                                gv0 = fmin_f32(gv0, Float32(10))
                                uv0 = fmax_f32(Float32(-10), fmin_f32(uv0, Float32(10)))
                                sig0 = cute.arch.rcp_approx(
                                    Float32(1) + cute.math.exp(-gv0, fastmath=True)
                                )
                                v0 = gv0 * sig0 * uv0 * rw

                                gv1 = gate[f + 1].to(cutlass.BFloat16).to(Float32)
                                uv1 = up[f + 1].to(cutlass.BFloat16).to(Float32)
                                gv1 = fmin_f32(gv1, Float32(10))
                                uv1 = fmax_f32(Float32(-10), fmin_f32(uv1, Float32(10)))
                                sig1 = cute.arch.rcp_approx(
                                    Float32(1) + cute.math.exp(-gv1, fastmath=True)
                                )
                                v1 = gv1 * sig1 * uv1 * rw

                                col_local = warp_n * Int32(32) + Int32(8 * nf) + Int32(2) * c
                                st_shared_u32(
                                    sa + row * Int32(self.mid_stride) + col_local * Int32(2),
                                    _cvt_rn_bf16x2_f32(v0, v1),
                                )
            cute.arch.sync_threads()

            # Row-local K32 MXFP8 quantization (1 K32 block per thread) -> compact unpadded `partial`.
            act_base = get_ptr_as_int64(partial, Int64(0))
            row = tidx // Int32(2)
            blk = tidx % Int32(2)
            col0 = n0 + blk * Int32(32)
            if (row < live) & (col0 < Int32(self.intermediate)):
                values = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
                for q in cutlass.range_constexpr(4):
                    words = ld_shared_v4_u32(
                        sa + row * Int32(self.mid_stride) + blk * Int32(64) + Int32(16 * q)
                    )
                    for w in cutlass.range_constexpr(4):
                        values[8 * q + 2 * w] = _bf16_lo(words[w])
                        values[8 * q + 2 * w + 1] = _bf16_hi(words[w])
                payload, scale = quantize_block_fp8_mx(
                    values, fmax_f32(max_abs_32(values), Float32(1e-4))
                )
                at = act_base + (Int64(first) + Int64(row)) * Int64(self.act_row_bytes)
                st_global_v4_u32(
                    at + Int64(col0), payload[0], payload[1], payload[2], payload[3]
                )
                st_global_v4_u32(
                    at + Int64(col0) + Int64(16), payload[4], payload[5], payload[6], payload[7]
                )
                if const_expr(self.intermediate == 288):
                    if col0 == Int32(256):
                        st_global_u32(at + Int64(288 + 8), scale)
                    else:
                        st_global_u8(
                            at + Int64(288) + Int64(col0 // Int32(32)), cutlass.Uint8(scale)
                        )
                elif const_expr(self.intermediate == 320):
                    st_global_u8(
                        at + Int64(320) + Int64(col0 // Int32(32)), cutlass.Uint8(scale)
                    )
                    if col0 == Int32(256):
                        st_global_u16(at + Int64(320 + 10), Uint32(0))
                else:
                    st_global_u8(
                        at + Int64(self.intermediate) + Int64(col0 // Int32(32)),
                        cutlass.Uint8(scale),
                    )


class Dsv4fNarrowFc2:
    """Expert-stationary M128 FC2 with SMEM-resident act, 3-stage W2 ring, and coalesced FP32 output.

    Each CTA loads one 128-row expert chunk's entire MXFP8 activation tile into SMEM once
    (32/36/40 KiB) and streams 4 consecutive N64 output tiles (256 hidden columns) through a
    continuous 3-stage W2/S2 ring (L2::evict_first), preserving exact ascending 128-K slice
    FP32 folding (`fadd_rn_f32`) and writing coalesced 256-byte streaming bursts per route row.
    """

    threads = 256
    tile_m = _TILE_M
    tile_n = 64
    tiles_per_cta = 4  # 256 hidden columns per CTA -> 16 CTAs per chunk
    w_ring = 3
    epi_stride = 288  # 64 f32 (256 B) + 32 B padding -> 0 bank conflicts on v2 store and v4 load

    def __init__(
        self,
        *,
        capacity: int,
        topk: int,
        experts: int,
        hidden: int,
        intermediate: int,
        exact_storage: bool,
    ):
        self.capacity = int(capacity)
        self.topk = int(topk)
        self.routes = self.capacity * self.topk
        self.experts = int(experts)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.exact_storage = bool(exact_storage)
        self.kernel_intermediate = (
            self.intermediate if self.exact_storage else (self.intermediate + 127) // 128 * 128
        )
        self.slices = (self.intermediate + 127) // 128
        self.n_ctas = self.hidden // (self.tile_n * self.tiles_per_cta)
        self.act_row_bytes = dsv4f_act_row_bytes(self.intermediate)
        self.act_bytes = self.tile_m * self.intermediate
        self.as_bytes = self.slices * self.tile_m * 4
        self.w_stage_bytes = 4 * self.tile_n * 16  # 4096 B (max K128 x N64 E2M1)
        self.s2_stage_bytes = self.tile_n * 4  # 256 B
        self.epi_bytes = self.tile_m * self.epi_stride  # 36864 B

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "act": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.act_bytes], 1024],
            "w2": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.w_ring * self.w_stage_bytes], 1024
            ],
            "epi": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.epi_bytes], 1024],
            "as_buf": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.as_bytes], 16],
            "s2": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.w_ring * self.s2_stage_bytes], 16
            ],
            "route": cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(
        self,
        w2: cute.Tensor,
        s2: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        max_tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            w2, s2, counts, prefixes, metadata, partial, output, self._storage()
        ).launch(
            grid=(self.n_ctas, max_tiles, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _load_w2_s2(
        self,
        w2_base: Int64,
        s2_base: Int64,
        sw: Int32,
        ss2: Int32,
        expert: Int64,
        ot64: Int32,
        slice_id: cutlass.Constexpr,
        slot: Int32,
        tidx: Int32,
        w_policy: Uint64,
    ):
        kb_count = min(4, self.intermediate // 32 - slice_id * 4)
        w_stage = sw + slot * Int32(self.w_stage_bytes)
        s_stage = ss2 + slot * Int32(self.s2_stage_bytes)

        # W2 weights: kb_count * 64 transfers of 16 bytes (all 256 threads for full slice).
        if tidx < Int32(kb_count * 64):
            lane_w = tidx % Int32(32)
            chunk_local = (tidx // Int32(32)) % Int32(2)
            kb = tidx // Int32(64)
            k = Int32(slice_id * 4) + kb
            if const_expr(self.exact_storage):
                n_128 = (ot64 // Int32(2)) * Int32(128)
                chunk_128 = (ot64 % Int32(2)) * Int32(2) + chunk_local
                word = (
                    expert * Int64(self.kernel_intermediate * self.hidden // 8)
                    + Int64(n_128) * Int64(self.kernel_intermediate // 8)
                    + Int64(k * Int32(512) + chunk_128 * Int32(128) + lane_w * Int32(4))
                )
            else:
                n = ot64 * Int32(64) + chunk_local * Int32(32)
                tile = (n // Int32(256)) * Int32(self.kernel_intermediate // 128) + Int32(slice_id)
                word = (
                    expert * Int64(self.kernel_intermediate * self.hidden // 8)
                    + Int64(tile) * Int64(4096)
                    + Int64(
                        kb * Int32(1024)
                        + ((n % Int32(256)) // Int32(32)) * Int32(128)
                        + lane_w * Int32(4)
                    )
                )
            _cp_async4_l2hint(w_stage + tidx * Int32(16), w2_base + word * Int64(4), w_policy)

        # S2 scales: 64 rows -> 16 x 16B cp.async for full slices, compact u8/u16 for tail.
        if const_expr(not self.exact_storage):
            if tidx < Int32(16):
                n0 = ot64 * Int32(64)
                tile = (n0 // Int32(256)) * Int32(self.kernel_intermediate // 128) + Int32(slice_id)
                word = (
                    expert * Int64(self.kernel_intermediate * self.hidden // 128)
                    + Int64(tile) * Int64(256)
                    + Int64(n0 % Int32(256))
                    + Int64(tidx * Int32(4))
                )
                _cp_async4_l2hint(s_stage + tidx * Int32(16), s2_base + word * Int64(4), w_policy)
        else:
            columns = self.intermediate // 32
            tile_base = (
                expert * Int64(self.kernel_intermediate * self.hidden // 32)
                + Int64(ot64 // Int32(2)) * Int64(128 * columns)
            )
            row_in_128 = (ot64 % Int32(2)) * Int32(64)
            if const_expr(kb_count == 4):
                if tidx < Int32(16):
                    byte_off = (
                        tile_base
                        + Int64(slice_id * 512)
                        + Int64(row_in_128 * Int32(4) + tidx * Int32(16))
                    )
                    _cp_async4_l2hint(s_stage + tidx * Int32(16), s2_base + byte_off, w_policy)
            elif const_expr(kb_count == 1):
                if tidx < Int32(64):
                    byte_off = tile_base + Int64(slice_id * 512) + Int64(row_in_128 + tidx)
                    st_shared_u32(s_stage + tidx * Int32(4), _ld_global_nc_u8(s2_base + byte_off))
            elif const_expr(kb_count == 2):
                if tidx < Int32(64):
                    byte_off = (
                        tile_base
                        + Int64(slice_id * 512)
                        + Int64((row_in_128 + tidx) * Int32(2))
                    )
                    st_shared_u32(s_stage + tidx * Int32(4), _ld_global_nc_u16(s2_base + byte_off))

    @cute.kernel
    def kernel(
        self,
        w2: cute.Tensor,
        s2: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        Storage: cutlass.Constexpr,
    ):
        n_blk = Int32(cute.arch.block_idx()[0])
        tile = Int32(cute.arch.block_idx()[1])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(Storage)

        meta_flat = cute.make_tensor(
            metadata.iterator, cute.make_layout(self.routes * 19)
        )
        total_tiles = Int32(meta_flat[0])
        invalid_count = Int32(meta_flat[1])
        out_base = get_ptr_as_int64(output, Int64(0))
        ot_base = n_blk * Int32(self.tiles_per_cta)

        if tile < total_tiles:
            code = Int32(meta_flat[Int32(_META_TILES_OFF) + tile])
            e = code & Int32(0xFFFF)
            row0 = (code >> Int32(16)) * Int32(_TILE_M)
            live = Int32(counts[e]) - row0
            if live > Int32(_TILE_M):
                live = Int32(_TILE_M)
            first = Int32(prefixes[e, 0]) + row0
            expert = Int64(e)

            warp_id = tidx // Int32(32)
            lane = tidx % Int32(32)
            g = lane // Int32(4)
            c = lane % Int32(4)
            warp_m = warp_id // Int32(2)
            warp_n = warp_id % Int32(2)

            s_act = shared_ptr_to_u32(storage.act.data_ptr())
            sw = shared_ptr_to_u32(storage.w2.data_ptr())
            s_epi = shared_ptr_to_u32(storage.epi.data_ptr())
            s_as = shared_ptr_to_u32(storage.as_buf.data_ptr())
            ss2 = shared_ptr_to_u32(storage.s2.data_ptr())
            s_route = storage.route.get_tensor(cute.make_layout((self.tile_m,)))

            act_base = get_ptr_as_int64(partial, Int64(0))
            w2_base = get_ptr_as_int64(w2, Int64(0))
            s2_base = get_ptr_as_int64(s2, Int64(0))
            stream_policy = create_l2_evict_first_policy()

            # Load route indices once into SMEM.
            if tidx < live:
                s_route[tidx] = Int32(meta_flat[Int32(_META_ROUTE_OFF) + first + tidx])

            # Prologue Group 0: entire chunk's act + act scales + W2/S2(ot=0, slice=0).
            row_a = tidx // Int32(2)
            half_a = tidx % Int32(2)
            if row_a < live:
                row_gmem = act_base + (Int64(first) + Int64(row_a)) * Int64(self.act_row_bytes)
                for k_blk in cutlass.range_constexpr(self.intermediate // 32):
                    cp_async4_shared_global(
                        s_act + Int32(k_blk * 4096) + row_a * Int32(32) + half_a * Int32(16),
                        row_gmem + Int64(k_blk * 32) + Int64(half_a) * Int64(16),
                    )
            if tidx < live:
                row_gmem = act_base + (Int64(first) + Int64(tidx)) * Int64(self.act_row_bytes)
                for s in cutlass.range_constexpr(self.slices):
                    cp_async_u32_shared_global(
                        s_as + Int32(s * 512) + tidx * Int32(4),
                        row_gmem + Int64(self.intermediate + s * 4),
                    )
            self._load_w2_s2(
                w2_base, s2_base, sw, ss2, expert, ot_base, 0, Int32(0), tidx, stream_policy
            )
            cute.arch.cp_async_commit_group()

            # Prologue Group 1: W2/S2(ot=0, slice=1).
            self._load_w2_s2(
                w2_base, s2_base, sw, ss2, expert, ot_base, 1, Int32(1), tidx, stream_policy
            )
            cute.arch.cp_async_commit_group()

            active_mt0 = warp_m * Int32(32) < live
            active_mt1 = warp_m * Int32(32) + Int32(16) < live
            row_m0_lo = warp_m * Int32(32) + g
            row_m0_hi = row_m0_lo + Int32(8)
            row_m1_lo = row_m0_lo + Int32(16)
            row_m1_hi = row_m0_lo + Int32(24)
            srow_m0 = warp_m * Int32(32) + g + (c & Int32(1)) * Int32(8)
            srow_m1 = srow_m0 + Int32(16)

            total = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            acc = cute.make_rmem_tensor(cute.make_layout((32,), stride=(1,)), Float32)
            asc = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
            wsc = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)

            for local_ot in cutlass.range(self.tiles_per_cta, unroll=1):
                for v in cutlass.range_constexpr(32):
                    total[v] = Float32(0.0)

                for slice_id in cutlass.range_constexpr(self.slices):
                    step = local_ot * Int32(self.slices) + Int32(slice_id)
                    slot = step % Int32(self.w_ring)
                    pf_slot = (step + Int32(2)) % Int32(self.w_ring)
                    pf_ot = local_ot + Int32((slice_id + 2) // self.slices)
                    pf_slice = (slice_id + 2) % self.slices

                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()

                    if pf_ot < Int32(self.tiles_per_cta):
                        self._load_w2_s2(
                            w2_base, s2_base, sw, ss2, expert, ot_base + pf_ot,
                            pf_slice, pf_slot, tidx, stream_policy,
                        )
                    cute.arch.cp_async_commit_group()

                    if active_mt0:
                        for v in cutlass.range_constexpr(32):
                            acc[v] = Float32(0.0)
                        as_slice = s_as + Int32(slice_id * 512)
                        s2_slot = ss2 + slot * Int32(self.s2_stage_bytes)
                        w2_slot = sw + slot * Int32(self.w_stage_bytes)
                        asc[0] = ld_shared_u32(as_slice + srow_m0 * Int32(4))
                        if active_mt1:
                            asc[1] = ld_shared_u32(as_slice + srow_m1 * Int32(4))
                        for nf in cutlass.range_constexpr(4):
                            wsc[nf] = ld_shared_u32(
                                s2_slot + (warp_n * Int32(32) + Int32(nf * 8) + g) * Int32(4)
                            )

                        kb_count = min(4, self.intermediate // 32 - slice_id * 4)
                        for kb in cutlass.range_constexpr(kb_count):
                            k_blk = slice_id * 4 + kb
                            a_blk = s_act + Int32(k_blk * 4096)
                            b_words = ld_shared_v4_u32(
                                w2_slot + (Int32(kb * 2) + warp_n) * Int32(512) + lane * Int32(16)
                            )
                            # mt = 0
                            a0, a2 = ld_shared_v2_u32(a_blk + row_m0_lo * Int32(32) + c * Int32(8))
                            a1, a3 = ld_shared_v2_u32(a_blk + row_m0_hi * Int32(32) + c * Int32(8))
                            for nf in cutlass.range_constexpr(4):
                                b0, b1 = e2m1x8_to_qmma_e2m1x8(b_words[nf])
                                f = 4 * nf
                                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                    a0, a1, a2, a3, b0, b1, asc[0], wsc[nf],
                                    bid_a=kb, bid_b=kb,
                                )
                                acc[f] = d0
                                acc[f + 1] = d1
                                acc[f + 2] = d2
                                acc[f + 3] = d3

                            # mt = 1
                            if active_mt1:
                                a0, a2 = ld_shared_v2_u32(
                                    a_blk + row_m1_lo * Int32(32) + c * Int32(8)
                                )
                                a1, a3 = ld_shared_v2_u32(
                                    a_blk + row_m1_hi * Int32(32) + c * Int32(8)
                                )
                                for nf in cutlass.range_constexpr(4):
                                    b0, b1 = e2m1x8_to_qmma_e2m1x8(b_words[nf])
                                    f = 16 + 4 * nf
                                    d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                        a0, a1, a2, a3, b0, b1, asc[1], wsc[nf],
                                        bid_a=kb, bid_b=kb,
                                    )
                                    acc[f] = d0
                                    acc[f + 1] = d1
                                    acc[f + 2] = d2
                                    acc[f + 3] = d3

                        # Exact ascending-slice IEEE-754 RNE FP32 fold.
                        for v in cutlass.range_constexpr(16):
                            total[v] = fadd_rn_f32(total[v], acc[v])
                        if active_mt1:
                            for v in cutlass.range_constexpr(16, 32):
                                total[v] = fadd_rn_f32(total[v], acc[v])

                # Coalesced FP32 epilogue through dedicated 36 KiB SMEM staging.
                if active_mt0:
                    for half in cutlass.range_constexpr(2):
                        row = warp_m * Int32(32) + Int32(8 * half) + g
                        for nf in cutlass.range_constexpr(4):
                            f = 4 * nf + 2 * half
                            col_bytes = (warp_n * Int32(32) + Int32(8 * nf) + Int32(2) * c) * Int32(4)
                            _st_shared_v2_f32(
                                s_epi + row * Int32(self.epi_stride) + col_bytes,
                                total[f],
                                total[f + 1],
                            )
                    if active_mt1:
                        for half in cutlass.range_constexpr(2):
                            row = warp_m * Int32(32) + Int32(16 + 8 * half) + g
                            for nf in cutlass.range_constexpr(4):
                                f = 16 + 4 * nf + 2 * half
                                col_bytes = (
                                    warp_n * Int32(32) + Int32(8 * nf) + Int32(2) * c
                                ) * Int32(4)
                                _st_shared_v2_f32(
                                    s_epi + row * Int32(self.epi_stride) + col_bytes,
                                    total[f],
                                    total[f + 1],
                                )
                cute.arch.sync_threads()

                n0 = (ot_base + local_ot) * Int32(self.tile_n)
                seg = tidx % Int32(16)
                r0 = tidx // Int32(16)
                for q in cutlass.range_constexpr(8):
                    row = r0 + Int32(16 * q)
                    if row < live:
                        route = Int64(s_route[row])
                        v0, v1, v2, v3 = ld_shared_v4_u32(
                            s_epi + row * Int32(self.epi_stride) + seg * Int32(16)
                        )
                        _st_global_l2hint_v4_u32(
                            out_base
                            + (route * Int64(self.hidden) + Int64(n0)) * Int64(4)
                            + Int64(seg) * Int64(16),
                            v0, v1, v2, v3, stream_policy,
                        )

            cute.arch.cp_async_wait_group(0)

        # Zero output planes for invalid routes (uniformly skipped when invalid_count == 0).
        if invalid_count > Int32(0):
            col_begin = n_blk * Int32(self.tile_n * self.tiles_per_cta)
            grid_y = Int32(cute.arch.grid_dim()[1])
            idx = tile
            while idx < invalid_count:
                route = Int64(meta_flat[Int32(_META_INVALID_OFF) + idx])
                if tidx < Int32(64):
                    st_global_v4_u32(
                        out_base
                        + (route * Int64(self.hidden) + Int64(col_begin)) * Int64(4)
                        + Int64(tidx) * Int64(16),
                        Uint32(0), Uint32(0), Uint32(0), Uint32(0),
                    )
                idx = idx + grid_y
