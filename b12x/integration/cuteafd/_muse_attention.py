"""Muse-specific score rounding on MiMo's allocation-free online GQA structure.

Both routes use one query token per CTA: a row's key tiles and reduction order
cannot depend on how prefill units were split. Shared MiMo code stays untouched.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import ld_shared_f32_offset, shared_ptr_to_u32
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import warp_mma_gemm
from b12x.attention._shared.contiguous.softmax import Softmax
from ._common import Operand, Scalar, compile_program
from ._fp8_weights import _e4m3x8_scaled_bf16
from ._glm_kernels import _bf16
from ._mimo_kernels import MimoGqaAttention, MimoRingCommit
from .pointwise import _mul


@dsl_user_op
def _cp_async_128b_zfill_off(smem_base: Int32, smem_off: int, gmem_addr: Int64, src_bytes: Int32,
                             *, loc=None, ip=None) -> None:
    dst = f"[$0+{int(smem_off)}]" if smem_off else "[$0]"
    llvm.inline_asm(
        None,
        [
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        f"cp.async.cg.shared.global.L2::128B {dst}, [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_32b_zfill(smem_addr: Int32, gmem_addr: Int64, src_bytes: Int32,
                        *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v2_u32_off(smem_base: Int32, byte_off: int, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    src = f"[$2+{int(byte_off)}]" if byte_off else "[$2]"
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_base).ir_value(loc=loc, ip=ip)],
        f"ld.shared.v2.u32 {{$0, $1}}, {src};",
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
def _st_shared_v4_u32_off(smem_base: Int32, byte_off: int, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32,
                          *, loc=None, ip=None) -> None:
    dst = f"[$0+{int(byte_off)}]" if byte_off else "[$0]"
    llvm.inline_asm(
        None,
        [
            Int32(smem_base).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        f"st.shared.v4.u32 {dst}, {{$1, $2, $3, $4}};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class MuseGqaAttention(MimoGqaAttention):
    def __init__(self, *, kv8: str | None = None, **kwargs):
        if kv8 not in (None, "e4m3"):
            raise ValueError("MuseGqaAttention supports kv8 None or 'e4m3'")
        # Enter MimoGqaAttention.kernel on its BF16 contract (self.kv_fp8 = False) so
        # kernel() allocates zero cross-tile raw/raw_s prefetch registers, keeps slot-0
        # addressing, and preserves the exact 22.6 KB BF16 shared-memory footprint.
        super().__init__(kv8=None, **kwargs)
        self.muse_fp8 = kv8 == "e4m3"
        if self.muse_fp8:
            if (self.head, self.v_head, self.kv_group, self.tile_n) != (64, 64, 64, 64):
                raise ValueError("Muse FP8 KV requires 64-wide heads/groups and 64-key tiles")
            self.kv8 = "e4m3"
            self.record_bytes = self.kv_heads * 136
            if self.record_bytes % 16:
                raise ValueError("Muse FP8 KV record_bytes must be 16-byte aligned")

    @cute.jit
    def _stage_and_widen_fp8_k(self, key_addr: cute.Tensor, buf: Int32, kv_head: Int32, cache: cute.Pointer,
                               s_k: cute.Tensor, s_v: cute.Tensor, thread: Int32):
        """Async-stage K (4 KB) + K/V FP32 scales (512 B) in group 1 and V (4 KB) in group 2
        into s_v[0..8192) and unused key_addr[1], widen K into swizzled s_k while V transfers,
        then drain V so QK/softmax and non-QK-warp V widening proceed concurrently."""
        raw_base = shared_ptr_to_u32(s_v.iterator)
        scale_base = shared_ptr_to_u32(key_addr.iterator) + Int32(self.tile_n * 8)
        k_vecs = (self.tile_n * 4 + self.threads - 1) // self.threads
        tok16_0 = thread // Int32(4)
        dim16 = (thread % Int32(4)) * Int32(16)
        raw_thr_base = raw_base + thread * Int32(16)
        k_head_off = Int64(kv_head) * Int64(64) + Int64(dim16)
        v_head_off = k_head_off + Int64(self.kv_heads * 64)
        fallback = Int64(cache.toint())
        bases = cute.make_rmem_tensor(cute.make_layout((k_vecs,)), Int64)
        for it in cutlass.range_constexpr(k_vecs):
            tok = tok16_0 + Int32(it * (self.threads // 4))
            if cutlass.const_expr((self.tile_n * 4) % self.threads == 0):
                b = Int64(key_addr[buf, tok])
                bases[it] = b
                valid = b >= Int64(0)
                _cp_async_128b_zfill_off(
                    raw_thr_base, it * self.threads * 16,
                    cutlass.select_(valid, b + k_head_off, fallback),
                    cutlass.select_(valid, Int32(16), Int32(0)),
                )
            else:
                b = Int64(-1)
                if thread + Int32(it * self.threads) < Int32(self.tile_n * 4):
                    b = Int64(key_addr[buf, tok])
                    valid = b >= Int64(0)
                    _cp_async_128b_zfill_off(
                        raw_thr_base, it * self.threads * 16,
                        cutlass.select_(valid, b + k_head_off, fallback),
                        cutlass.select_(valid, Int32(16), Int32(0)),
                    )
                bases[it] = b

        k_scale_off = Int64(self.kv_heads * 128) + Int64(kv_head) * Int64(4)
        v_scale_off = k_scale_off + Int64(self.kv_heads * 4)
        if cutlass.const_expr(self.threads == 2 * self.tile_n):
            is_v = thread >= Int32(self.tile_n)
            s_tok = cutlass.select_(is_v, thread - Int32(self.tile_n), thread)
            s_base = Int64(key_addr[buf, s_tok])
            s_valid = s_base >= Int64(0)
            s_off = cutlass.select_(is_v, v_scale_off, k_scale_off)
            _cp_async_32b_zfill(
                scale_base + thread * Int32(4),
                cutlass.select_(s_valid, s_base + s_off, fallback),
                cutlass.select_(s_valid, Int32(4), Int32(0)),
            )
        else:
            if thread < Int32(2 * self.tile_n):
                is_v = thread >= Int32(self.tile_n)
                s_tok = cutlass.select_(is_v, thread - Int32(self.tile_n), thread)
                s_base = Int64(key_addr[buf, s_tok])
                s_valid = s_base >= Int64(0)
                s_off = cutlass.select_(is_v, v_scale_off, k_scale_off)
                _cp_async_32b_zfill(
                    scale_base + thread * Int32(4),
                    cutlass.select_(s_valid, s_base + s_off, fallback),
                    cutlass.select_(s_valid, Int32(4), Int32(0)),
                )
        cute.arch.cp_async_commit_group()

        for it in cutlass.range_constexpr(k_vecs):
            if cutlass.const_expr((self.tile_n * 4) % self.threads == 0):
                b = Int64(bases[it])
                valid = b >= Int64(0)
                _cp_async_128b_zfill_off(
                    raw_thr_base, 4096 + it * self.threads * 16,
                    cutlass.select_(valid, b + v_head_off, fallback),
                    cutlass.select_(valid, Int32(16), Int32(0)),
                )
            else:
                if thread + Int32(it * self.threads) < Int32(self.tile_n * 4):
                    b = Int64(bases[it])
                    valid = b >= Int64(0)
                    _cp_async_128b_zfill_off(
                        raw_thr_base, 4096 + it * self.threads * 16,
                        cutlass.select_(valid, b + v_head_off, fallback),
                        cutlass.select_(valid, Int32(16), Int32(0)),
                    )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        k_base = shared_ptr_to_u32(s_k.iterator)
        k_steps = (self.tile_n * 8 + self.threads - 1) // self.threads
        k_tok0 = thread // Int32(8)
        k_col = thread % Int32(8)
        k_raw_thr = raw_base + thread * Int32(8)
        k_sc_thr = scale_base + k_tok0 * Int32(4)
        k_dst_thr = k_base + k_tok0 * Int32(128) + ((k_col ^ (k_tok0 % Int32(8))) * Int32(16))
        for step in cutlass.range_constexpr(k_steps):
            if cutlass.const_expr((self.tile_n * 8) % self.threads == 0):
                lo, hi = _ld_shared_v2_u32_off(k_raw_thr, step * self.threads * 8)
                ks = ld_shared_f32_offset(k_sc_thr, step * (self.threads // 8) * 4)
                w0, w1, w2, w3 = _e4m3x8_scaled_bf16(lo, hi, ks)
                _st_shared_v4_u32_off(k_dst_thr, step * (self.threads // 8) * 128, w0, w1, w2, w3)
            else:
                if thread + Int32(step * self.threads) < Int32(self.tile_n * 8):
                    lo, hi = _ld_shared_v2_u32_off(k_raw_thr, step * self.threads * 8)
                    ks = ld_shared_f32_offset(k_sc_thr, step * (self.threads // 8) * 4)
                    w0, w1, w2, w3 = _e4m3x8_scaled_bf16(lo, hi, ks)
                    _st_shared_v4_u32_off(k_dst_thr, step * (self.threads // 8) * 128, w0, w1, w2, w3)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    @cute.jit
    def _widen_fp8_v(self, key_addr: cute.Tensor, s_v: cute.Tensor, v_tid: Int32):
        """Widen raw V in s_v[4096..8192) into swizzled s_v[0..8192) on idle non-QK warps."""
        raw_base = shared_ptr_to_u32(s_v.iterator)
        scale_base = shared_ptr_to_u32(key_addr.iterator) + Int32(self.tile_n * 8)
        v_threads = 64 if self.qk_warps == 1 else 128
        v_half_steps = 256 // v_threads
        v_tok0 = v_tid // Int32(8)
        v_col = v_tid % Int32(8)
        v_raw_thr = raw_base + Int32(4096) + v_tid * Int32(8)
        v_sc_thr = scale_base + Int32(256) + v_tok0 * Int32(4)
        v_dst_thr = raw_base + v_tok0 * Int32(128) + ((v_col ^ (v_tok0 % Int32(8))) * Int32(16))
        for step in cutlass.range_constexpr(v_half_steps):
            lo, hi = _ld_shared_v2_u32_off(v_raw_thr, step * v_threads * 8)
            vs = ld_shared_f32_offset(v_sc_thr, step * (v_threads // 8) * 4)
            w0, w1, w2, w3 = _e4m3x8_scaled_bf16(lo, hi, vs)
            _st_shared_v4_u32_off(v_dst_thr, step * (v_threads // 8) * 128, w0, w1, w2, w3)
        v_lo = cute.make_rmem_tensor(cute.make_layout((v_half_steps,)), Uint32)
        v_hi = cute.make_rmem_tensor(cute.make_layout((v_half_steps,)), Uint32)
        for step in cutlass.range_constexpr(v_half_steps):
            lo, hi = _ld_shared_v2_u32_off(v_raw_thr, 2048 + step * v_threads * 8)
            v_lo[step] = lo
            v_hi[step] = hi
        cute.arch.barrier(barrier_id=1, number_of_threads=v_threads)
        for step in cutlass.range_constexpr(v_half_steps):
            vs = ld_shared_f32_offset(v_sc_thr, 128 + step * (v_threads // 8) * 4)
            w0, w1, w2, w3 = _e4m3x8_scaled_bf16(v_lo[step], v_hi[step], vs)
            _st_shared_v4_u32_off(v_dst_thr, 4096 + step * (v_threads // 8) * 128, w0, w1, w2, w3)

    @cute.jit
    def _tile(self, tile: Int32, next_tile: Int32, has_next, buf: Int32, lo: Int64, hi: Int64, row0: Int64,
              kv_head: Int32, cache: cute.Pointer,
              kv_step: cute.Pointer, page_table: cute.Pointer, table_base: Int64, seq_start: Int64,
              step_first: Int64, ring_base: Int64, token_pos: cute.Tensor, key_addr: cute.Tensor,
              key_pos: cute.Tensor, s_k: cute.Tensor, s_v: cute.Tensor, s_p: cute.Tensor, s_scale: cute.Tensor,
              thread: Int32, tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma, r_q: cute.Tensor,
              r_k: cute.Tensor, r_p: cute.Tensor, r_v: cute.Tensor, c_q: cute.TiledCopy, c_k: cute.TiledCopy,
              c_p: cute.TiledCopy, c_v: cute.TiledCopy, cs_q: cute.Tensor, cs_k: cute.Tensor, cs_p: cute.Tensor,
              cs_v: cute.Tensor, acc_o: cute.Tensor, softmax: Softmax, raw: cute.Tensor, raw_s: cute.Tensor,
              is_first: cutlass.Constexpr):
        if cutlass.const_expr(self.muse_fp8):
            if cutlass.const_expr(is_first):
                self._resolve(tile, buf, lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                              ring_base, key_addr, key_pos, thread)
                cute.arch.sync_threads()
            self._stage_and_widen_fp8_k(key_addr, buf, kv_head, cache, s_k, s_v, thread)
        else:
            self._resolve(tile, buf, lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                          ring_base, key_addr, key_pos, thread)
            cute.arch.sync_threads()
            self._load_bf16(key_addr, kv_head, cache, s_k, s_v)
            cute.arch.sync_threads()
        qk_threads = self.qk_warps * 32
        if thread < Int32(qk_threads):
            acc_s = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc_s.fill(0.0)
            warp_mma_gemm(tiled_mma_qk, acc_s, r_q, r_k, cs_q, cs_k, c_q, c_k, A_in_regs=not is_first)
            s_mn = layout_utils.reshape_acc_to_mn(acc_s)
            coords = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n))))
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                t_local = coords[m, 0][0] // Int32(self.group)
                p = Int64(-1)
                if t_local < Int32(self.tokens):
                    p = Int64(token_pos[t_local])
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    # Eager QK matmul and scaling each materialize a BF16 tensor.
                    s_mn[m, n] = _bf16(_mul(_bf16(s_mn[m, n]), Float32(0.125)))
                    kp = Int64(key_pos[buf, coords[m, n][1]])
                    masked = (p < Int64(0)) | (kp < Int64(0)) | (kp > p)
                    if cutlass.const_expr(self.window > 0):
                        masked = masked | (kp <= p - Int64(self.window))
                    if masked:
                        s_mn[m, n] = -Float32.inf
            row_scale = softmax.online_softmax(acc_s, is_first=is_first, check_inf=True)
            for m in cutlass.range_constexpr(cute.size(s_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(s_mn.shape[1])):
                    c = coords[m, n]
                    s_p[c[0], c[1]] = BFloat16(s_mn[m, n])
                if coords[m, 0][1] == Int32(0):
                    s_scale[coords[m, 0][0]] = row_scale[m]
        else:
            if cutlass.const_expr(self.muse_fp8):
                if cutlass.const_expr(self.qk_warps == 1):
                    if thread >= Int32(64):
                        self._widen_fp8_v(key_addr, s_v, thread - Int32(64))
                else:
                    if thread < Int32(qk_threads + 128):
                        self._widen_fp8_v(key_addr, s_v, thread - Int32(qk_threads))
        cute.arch.sync_threads()
        if cutlass.const_expr(self.muse_fp8):
            if has_next:
                self._resolve(next_tile, buf, lo, hi, cache, kv_step, page_table, table_base, seq_start,
                              step_first, ring_base, key_addr, key_pos, thread)
        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        if cutlass.const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
                scale = Float32(s_scale[o_coords[m, 0][0]])
                o_mn[m, None].store(o_mn[m, None].load() * scale)
        warp_mma_gemm(tiled_mma_pv, acc_o, r_p, r_v, cs_p, cs_v, c_p, c_v)
        cute.arch.sync_threads()


class FullAttention:
    def __init__(self, heads, kv_heads, kv_cache="bf16"):
        # Scaling is performed before the softmax, including its BF16 boundary.
        self.attn = MuseGqaAttention(heads=heads, kv_heads=kv_heads, tokens=1, window=0,
            paged=True, sink=False, direct=True, softmax_scale=1.0, head=64, v_head=64,
            kv8="e4m3" if kv_cache == "fp8" else None, kv_group=64)

    @cute.jit
    def run(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
            out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32, stream: cuda.CUstream):
        f32 = cute.make_ptr(Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.attn(q, kv_cache, kv_cache, positions, page_table, page_table, page_table, q, out,
                  f32, f32, rows, table_stride, Int32(1), stream)


class FullPrefill(FullAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32, stream: cuda.CUstream):
        self.run(q, kv_cache, positions, page_table, out, scratch, rows, table_stride, stream)


class FullDecode(FullAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, table_stride: Int32, splits: Int32,
                 stream: cuda.CUstream):
        # A single ordered scan deliberately ignores the split hint for row invariance.
        self.run(q, kv_cache, positions, page_table, out, scratch, rows, table_stride, stream)


class SwaAttention:
    def __init__(self, heads, kv_heads, ring_rows, window, kv_cache="bf16"):
        self.attn = MuseGqaAttention(heads=heads, kv_heads=kv_heads, tokens=1, window=window,
            paged=False, sink=False, direct=True, softmax_scale=1.0, ring_rows=ring_rows,
            head=64, v_head=64, kv8="e4m3" if kv_cache == "fp8" else None, kv_group=64)

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_step: cute.Pointer, ring: cute.Pointer, positions: cute.Pointer,
                 ring_slots: cute.Pointer, seq_first: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        f32 = cute.make_ptr(Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        self.attn(q, ring, kv_step, positions, seq_first, ring_slots, seq_first, q, out,
                  f32, f32, rows, Int32(0), Int32(1), stream)


def _check(heads, max_rows):
    if heads not in (64, 128) or max_rows <= 0:
        raise ValueError("Muse requires heads 64/128 and positive max_rows")
    return heads // 8


def attention_scratch_bytes(*args, **kwargs):
    return 1024


def compile_attention_aot(*, heads=128, kind, route, max_rows=4096, ring_rows=6144,
                          kv_heads=None, window=2048, kv_cache="bf16"):
    if kv_heads is None:
        kv_heads = _check(heads, max_rows)
    elif heads <= 0 or kv_heads <= 0 or heads % kv_heads or max_rows <= 0:
        raise ValueError("Muse attention fixture requires positive divisible head counts")
    if kind not in ("full", "swa") or route not in ("prefill", "decode"):
        raise ValueError("Muse attention requires full/swa and prefill/decode")
    if window <= 0 or ring_rows < window or ring_rows % 64:
        raise ValueError("Muse ring capacity must be page-aligned and at least the positive window")
    if kv_cache not in ("bf16", "fp8"):
        raise ValueError("Muse KV cache must be bf16 or fp8")
    dtype = torch.uint8 if kv_cache == "fp8" else torch.bfloat16
    q = Operand("q", torch.bfloat16, f"[rows,{heads},64]")
    positions = Operand("positions", torch.int64, "[rows]", align=8)
    out = Operand("out", torch.bfloat16, f"[rows,{heads},64]", "out")
    scratch = Operand("scratch", torch.uint8, "[1024]", "scratch")
    record = kv_heads * (136 if kv_cache == "fp8" else 128)
    if kind == "full":
        launch = (FullDecode if route == "decode" else FullPrefill)(heads, kv_heads, kv_cache)
        operands = (q, Operand("kv_cache", dtype, f"[pages*64,{record}]", "inout"),
                    positions, Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4), out, scratch)
        scalars = (Scalar("rows"), Scalar("table_stride"))
        if route == "decode":
            scalars += (Scalar("splits"),)
    else:
        launch = SwaAttention(heads, kv_heads, ring_rows, window, kv_cache)
        operands = (q, Operand("kv_step", dtype, f"[rows,{record}]"),
                    Operand("ring", dtype, f"[rings*{ring_rows},{record}]", "inout"), positions,
                    Operand("ring_slots", torch.int64, "[rows]", align=8),
                    Operand("seq_first", torch.int32, "[rows]", align=4), out, scratch)
        scalars = (Scalar("rows"),)
    return compile_program(launch, name=f"muse_attention_{kind}", operands=operands, scalars=scalars,
        key=(heads, kv_heads, kind, route, max_rows, ring_rows, window, kv_cache),
        geometry=dict(heads=heads, kv_heads=kv_heads, head_dim=64, kind=kind, route=route,
                      max_rows=max_rows, ring_rows=ring_rows, window=window if kind == "swa" else 0,
                      max_splits=1, rows_per_cta=1, kv_cache=kv_cache,
                      record_bytes=kv_heads * (136 if kv_cache == "fp8" else 256)),
        scratch={"scratch": lambda rows: 1024}, version=2 if kv_cache == "fp8" else 1)


def compile_swa_commit_aot(*, heads=128, max_rows=4096, ring_rows=6144, kv_cache="bf16"):
    kv_heads = _check(heads, max_rows)
    if kv_cache not in ("bf16", "fp8"):
        raise ValueError("Muse KV cache must be bf16 or fp8")
    record_bytes = kv_heads * (136 if kv_cache == "fp8" else 256)
    dtype = torch.uint8 if kv_cache == "fp8" else torch.bfloat16
    record = record_bytes if kv_cache == "fp8" else record_bytes // 2
    return compile_program(MimoRingCommit(record=kv_heads * 128, ring_rows=ring_rows,
                                         record_bytes=record_bytes),
        name="muse_swa_commit", operands=(
            Operand("kv_step", dtype, f"[rows,{record}]"),
            Operand("ring", dtype, f"[rings*{ring_rows},{record}]", "inout"),
            Operand("ring_slots", torch.int64, "[rows]", align=8),
            Operand("seq_first", torch.int32, "[rows]", align=4)),
        scalars=(Scalar("rows"),), key=(heads, ring_rows, max_rows, kv_cache),
        geometry=dict(heads=heads, kv_heads=kv_heads, ring_rows=ring_rows, max_rows=max_rows,
                      kv_cache=kv_cache, record_bytes=record_bytes))
