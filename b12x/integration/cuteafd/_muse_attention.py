"""Muse-specific score rounding on MiMo's allocation-free online GQA structure.

Both routes use one query token per CTA: a row's key tiles and reduction order
cannot depend on how prefill units were split. Shared MiMo code stays untouched.

The 8-bit KV route keeps the reference's per-element BF16 dequantization: the
MMAs consume ``bf16(kv * scale)``, exactly the bytes the BF16 route consumes
when it is handed the same-scale QDQ records, so the two routes are
bit-identical. Each tile's bytes are converted straight from global memory into
its shared BF16 tile at the QK/PV phase boundaries (see ``_stage``) instead of
being staged in registers at the previous tile's start, which keeps a tile's
raw words and scales out of the MMA register lifetime.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import ld_global_nc_v4_u32, shared_ptr_to_u32, st_shared_v4_u32
from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import warp_mma_gemm
from b12x.attention._shared.contiguous.softmax import Softmax
from ._common import Operand, Scalar, compile_program
from ._fp8_weights import _e4m3x8_scaled_bf16, _ld_f32
from ._glm_kernels import _bf16
from ._mimo_kernels import MimoGqaAttention, MimoRingCommit, _s8x8_scaled_bf16
from .pointwise import _mul


class MuseGqaAttention(MimoGqaAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # _stage converts whole 16-byte pieces of one head at a fixed offset; every supported
        # Muse geometry (head = v_head = 64, one 64-dim scale group) divides that way.
        if self.kv_fp8 and (self.chunk != 16 or self.head != self.v_head
                            or self.threads % (self.head // 16)):
            raise ValueError("Muse 8-bit attention stages whole 16-byte pieces")

    @cute.jit
    def _stage(self, key_addr: cute.Tensor, buf: Int32, kv_head: Int32, cache: cute.Pointer,
               value: cutlass.Constexpr, s_k: cute.Tensor, s_v: cute.Tensor, thread: Int32):
        """Convert one tile's 8-bit keys (``value`` false) or values of slot ``buf`` into their
        shared BF16 tile, one 16-byte piece at a time.

        Byte-for-byte what ``_load_raw`` + ``_widen`` produce -- ``bf16(q * scale)`` per
        element, missing keys zero-filled -- but each piece's words and scale are consumed as
        they are loaded, so a tile's staging registers never stay live into an MMA phase."""
        k_scale_off = self.kv_heads * (self.head + self.v_head)
        v_scale_off = k_scale_off + 4 * self.kv_heads * self.k_scales
        widen = _e4m3x8_scaled_bf16 if self.kv8 == "e4m3" else _s8x8_scaled_bf16
        if cutlass.const_expr(value):
            plane = Int64(self.kv_heads * self.head) + Int64(kv_head) * Int64(self.v_head)
            scale_base = Int64(v_scale_off) + Int64(4 * self.v_scales) * Int64(kv_head)
            per_scale = self.v_head if self.kv_group == 0 else self.kv_group
            vectors, tile = self.v_head // 16, s_v
        else:
            plane = Int64(kv_head) * Int64(self.head)
            scale_base = Int64(k_scale_off) + Int64(4 * self.k_scales) * Int64(kv_head)
            per_scale = self.head if self.kv_group == 0 else self.kv_group
            vectors, tile = self.head // 16, s_k
        fallback = Int64(cache.toint())
        # Every piece this thread owns sits at the same head offset (threads is a multiple of
        # `vectors`), so one loop-invariant piece and scale index serves them all.
        dim = (thread % Int32(vectors)) * Int32(16)
        scale_off = scale_base + Int64(4) * Int64(dim // Int32(per_scale))
        for it in cutlass.range_constexpr(self.tile_n * vectors // self.threads):
            token = (thread + Int32(it * self.threads)) // Int32(vectors)
            base = Int64(key_addr[buf, token])
            valid = base >= Int64(0)
            words = ld_global_nc_v4_u32(cutlass.select_(valid, base + plane + Int64(dim), fallback))
            scale = cutlass.select_(
                valid, _ld_f32(cutlass.select_(valid, base + scale_off, fallback)), Float32(0.0))
            words = (cutlass.select_(valid, words[0], Uint32(0)), cutlass.select_(valid, words[1], Uint32(0)),
                     cutlass.select_(valid, words[2], Uint32(0)), cutlass.select_(valid, words[3], Uint32(0)))
            for half in cutlass.range_constexpr(2):
                w0, w1, w2, w3 = widen(words[2 * half], words[2 * half + 1], scale)
                st_shared_v4_u32(shared_ptr_to_u32(tile.iterator + cute.crd2idx(
                    (token, dim + Int32(8 * half)), tile.layout)), w0, w1, w2, w3)

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
        if cutlass.const_expr(self.kv_fp8):
            # The next tile's addresses go to their own key_addr slot, so resolving them here
            # costs nothing and leaves its conversion to this tile's QK/PV boundaries.
            if has_next:
                self._resolve(next_tile, Int32(1) - buf, lo, hi, cache, kv_step, page_table, table_base,
                              seq_start, step_first, ring_base, key_addr, key_pos, thread)
            if cutlass.const_expr(is_first):
                # The prologue staged this tile's bytes in registers; every later tile arrives
                # already converted (the previous call's _stage at its QK/PV boundaries).
                self._widen(raw, raw_s, s_k, s_v, thread)
                cute.arch.sync_threads()
        else:
            self._resolve(tile, buf, lo, hi, cache, kv_step, page_table, table_base, seq_start, step_first,
                          ring_base, key_addr, key_pos, thread)
            cute.arch.sync_threads()
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
        if cutlass.const_expr(self.kv_fp8):
            if has_next:
                # Every warp has finished reading this tile's shared keys, so the next tile's
                # keys convert now, under the PV MMA.
                self._stage(key_addr, Int32(1) - buf, kv_head, cache, False, s_k, s_v, thread)
        o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        o_coords = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(cute.make_identity_tensor((self.tile_m, self.v_head))))
        if cutlass.const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(o_mn.shape[0])):
                scale = Float32(s_scale[o_coords[m, 0][0]])
                o_mn[m, None].store(o_mn[m, None].load() * scale)
        warp_mma_gemm(tiled_mma_pv, acc_o, r_p, r_v, cs_p, cs_v, c_p, c_v)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.kv_fp8):
            if has_next:
                # Values are free too; the next tile's QK and softmax hide these loads.
                self._stage(key_addr, Int32(1) - buf, kv_head, cache, True, s_k, s_v, thread)


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
                      record_bytes=kv_heads * (136 if kv_cache == "fp8" else 256)), scratch={"scratch": lambda rows: 1024})


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
