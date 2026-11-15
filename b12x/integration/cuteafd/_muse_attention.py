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

from b12x._lib.intrinsics import (
    ld_shared_f32,
    ld_shared_v2_u32,
    shared_ptr_to_u32,
    st_shared_f32,
    st_shared_v4_u32,
)
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import warp_mma_gemm
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention.paged._selected_forward import _cp_async_load_128b_zfill
from ._common import Operand, Scalar, compile_program
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from ._glm_kernels import _bf16
from ._mimo_kernels import MimoGqaAttention, MimoRingCommit
from .pointwise import _mul


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
    def _load_fp8(self, key_addr: cute.Tensor, buf: Int32, kv_head: Int32, cache: cute.Pointer,
                  s_k: cute.Tensor, s_v: cute.Tensor, thread: Int32):
        """Async-stage 4 KB raw V into s_k[0..4096) and 4 KB raw K into s_k[4096..8192) via
        16B cp.async ZFILL while staging all 128 per-token FP32 scales into unused key_addr[1],
        then widen V into disjoint s_v and widen K into s_k via cute.crd2idx."""
        raw_base = shared_ptr_to_u32(s_k.iterator)
        scale_base = shared_ptr_to_u32(key_addr.iterator) + Int32(self.tile_n * 8)
        vecs = (self.tile_n * 4 + self.threads - 1) // self.threads
        k_head_base = Int64(kv_head) * Int64(64)
        v_head_base = Int64(self.kv_heads * 64) + k_head_base
        fallback = Int64(cache.toint())
        for it in cutlass.range_constexpr(vecs):
            linear = thread + Int32(it * self.threads)
            if linear < Int32(self.tile_n * 4):
                tok = linear // Int32(4)
                dim = (linear % Int32(4)) * Int32(16)
                base = Int64(key_addr[buf, tok])
                valid = base >= Int64(0)
                sz = cutlass.select_(valid, Int32(16), Int32(0))
                v_src = cutlass.select_(valid, base + v_head_base + Int64(dim), fallback)
                k_src = cutlass.select_(valid, base + k_head_base + Int64(dim), fallback)
                _cp_async_load_128b_zfill(raw_base + linear * Int32(16), v_src, sz)
                _cp_async_load_128b_zfill(raw_base + Int32(4096) + linear * Int32(16), k_src, sz)
        k_scale_off = Int64(self.kv_heads * 128) + Int64(kv_head) * Int64(4)
        v_scale_off = k_scale_off + Int64(self.kv_heads * 4)
        s_iters = (2 * self.tile_n + self.threads - 1) // self.threads
        for it in cutlass.range_constexpr(s_iters):
            s_idx = thread + Int32(it * self.threads)
            if s_idx < Int32(2 * self.tile_n):
                is_v = s_idx >= Int32(self.tile_n)
                s_tok = cutlass.select_(is_v, s_idx - Int32(self.tile_n), s_idx)
                s_base = Int64(key_addr[buf, s_tok])
                s_valid = s_base >= Int64(0)
                s_off = cutlass.select_(is_v, v_scale_off, k_scale_off)
                s_val = cutlass.select_(s_valid, _ld_f32(cutlass.select_(s_valid, s_base + s_off, fallback)),
                                        Float32(0.0))
                st_shared_f32(scale_base + s_idx * Int32(4), s_val)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        steps = (self.tile_n * 8 + self.threads - 1) // self.threads
        for step in cutlass.range_constexpr(steps):
            linear = thread + Int32(step * self.threads)
            if linear < Int32(self.tile_n * 8):
                tok = linear // Int32(8)
                dim = (linear % Int32(8)) * Int32(8)
                lo, hi = ld_shared_v2_u32(raw_base + linear * Int32(8))
                vs = ld_shared_f32(scale_base + Int32(256) + tok * Int32(4))
                w0, w1, w2, w3 = _e4m3x8_scaled_bf16(lo, hi, vs)
                st_shared_v4_u32(shared_ptr_to_u32(s_v.iterator + cute.crd2idx((tok, dim), s_v.layout)),
                                 w0, w1, w2, w3)

        k_lo = cute.make_rmem_tensor(cute.make_layout((steps,)), Uint32)
        k_hi = cute.make_rmem_tensor(cute.make_layout((steps,)), Uint32)
        k_sc = cute.make_rmem_tensor(cute.make_layout((steps,)), Float32)
        for step in cutlass.range_constexpr(steps):
            linear = thread + Int32(step * self.threads)
            k_lo[step] = Uint32(0)
            k_hi[step] = Uint32(0)
            k_sc[step] = Float32(0.0)
            if linear < Int32(self.tile_n * 8):
                tok = linear // Int32(8)
                lo, hi = ld_shared_v2_u32(raw_base + Int32(4096) + linear * Int32(8))
                k_lo[step] = lo
                k_hi[step] = hi
                k_sc[step] = ld_shared_f32(scale_base + tok * Int32(4))
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr(steps):
            linear = thread + Int32(step * self.threads)
            if linear < Int32(self.tile_n * 8):
                tok = linear // Int32(8)
                dim = (linear % Int32(8)) * Int32(8)
                w0, w1, w2, w3 = _e4m3x8_scaled_bf16(k_lo[step], k_hi[step], k_sc[step])
                st_shared_v4_u32(shared_ptr_to_u32(s_k.iterator + cute.crd2idx((tok, dim), s_k.layout)),
                                 w0, w1, w2, w3)

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
        self._resolve(tile, buf, lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                      ring_base, key_addr, key_pos, thread)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.muse_fp8):
            self._load_fp8(key_addr, buf, kv_head, cache, s_k, s_v, thread)
        else:
            self._load_bf16(key_addr, kv_head, cache, s_k, s_v)
        cute.arch.sync_threads()
        if thread < Int32(self.qk_warps * 32):
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
        cute.arch.sync_threads()
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
        scratch={"scratch": lambda rows: 1024}, version=3 if kv_cache == "fp8" else 1)


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
