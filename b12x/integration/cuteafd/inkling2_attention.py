"""Inkling cache ABI v2: segmented biased-sink GQA with native global FP8.

Global operands are q, kv_cache, positions, page_table, table_indices,
seq_first, tile_rows, rel, sinks, out, scratch; runtime scalars are rows,
table_stride, tiles, and (global decode with max_splits > 1) the launcher's
runtime ``device_sm_count`` (``cudaDevAttrMultiProcessorCount``). Each row
selects the request page table at table_indices[row]*table_stride. Global
decode derives each row's split count on device from its own context length
and the device's static SM count:

    active_max_splits = device_max_splits(device_sm_count, max_splits)
    row_splits = clamp(ceil((pos + 1) / split_chunk), 1, active_max_splits)

so a row's reduction order is independent of batch composition and rows with
1 split write directly without touching scratch. FP8 records contain K[1024]
and V[1024] E4M3 payloads then K-scale[8] and V-scale[8] FP32 descales in
one 2112-byte physical record. BF16 records retain their 4096-byte layout.

Local operands are q, kv_step, ring, positions, ring_slots, seq_first,
tile_rows, rel, sinks, out, scratch; scalars are rows and tiles. Local
records remain BF16. ring_slots are absolute record indices into a strided
request bank, not a dense array of fixed-sized rings.

tile_rows[cta] gives the absolute packed query start; seq_first[row] is
the segment's absolute first row. Ten-row prefill tiles clamp at the next
segment boundary. Decode uses one row per CTA. KV packing completes before
attention; the local commit runs afterward. The relative bias and per-row
sink follow the checkpoint's native arithmetic in _inkling2_kernels.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import INKLING2_MINI, INKLING2_SMALL, Inkling2Geometry, Operand, Scalar, compile_program
from ._inkling2_kernels import DEFAULT_SPLIT_CHUNK, InklingGqaAttention, InklingSplitMerge
from ._inkling2_mini_prefill import InklingMiniGlobalFp8Prefill, InklingMiniLocalBf16Prefill
from ._mimo_kernels import MAX_SPLITS, MimoRingCommit

__all__ = [
    "DEFAULT_MAX_SPLITS",
    "DEFAULT_SPLIT_CHUNK",
    "attention_scratch_bytes",
    "compile_inkling2_attention_aot",
    "device_max_splits",
    "prefill_tokens",
    "record_bytes",
    "row_splits",
]

_ALIGN = 1024
DEFAULT_MAX_SPLITS = 32
KINDS = ("global", "local")


def _align(value: int) -> int:
    return (int(value) + _ALIGN - 1) // _ALIGN * _ALIGN


def _check(kind: str, route: str, max_rows: int) -> int:
    if kind not in KINDS:
        raise ValueError("kind is 'global' or 'local'")
    if route not in ("decode", "prefill"):
        raise ValueError("route is 'decode' or 'prefill'")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    return int(max_rows)


def record_bytes(g: Inkling2Geometry, kv_format: str = "bf16") -> int:
    """Bytes of one token's BF16 KV record."""
    return g.record_elems + g.kv_heads * 8 if kv_format == "fp8" else g.record_elems * 2


def prefill_tokens(g: Inkling2Geometry) -> int:
    """Rows per CTA of the prefill route: a 64-row M tile of (token, head) pairs."""
    return 64 // (g.heads // g.kv_heads)


def device_max_splits(sm_count: int, max_splits: int = DEFAULT_MAX_SPLITS) -> int:
    """Static per-device split cap derived from ``cudaDevAttrMultiProcessorCount``.

    - ``sm_count >= 128`` (RTX PRO 6000 188 SMs, RTX 5090 170 SMs): up to 32 splits
    - ``64 <= sm_count < 128`` (mid-SM SM120 cards): up to 16 splits
    - ``0 < sm_count < 64`` (GB10 48 SMs, off the serving path): up to 8 splits
    """
    cap = max(1, int(max_splits))
    sms = int(sm_count)
    if sms <= 0:
        return cap
    if sms < 64:
        limit = 8
    elif sms < 128:
        limit = 16
    else:
        limit = 32
    return max(1, min(cap, limit))


def row_splits(context_len: int, max_splits: int = DEFAULT_MAX_SPLITS,
               split_chunk: int = DEFAULT_SPLIT_CHUNK, sm_count: int | None = None) -> int:
    """Per-row split count chosen on device for a row with ``context_len`` keys (``pos + 1``)."""
    if int(split_chunk) <= 0:
        raise ValueError("split_chunk must be positive")
    cap = max(1, int(max_splits)) if sm_count is None else device_max_splits(sm_count, max_splits)
    n = (int(context_len) + int(split_chunk) - 1) // int(split_chunk)
    return max(1, min(cap, n))


def attention_scratch_bytes(g: Inkling2Geometry, kind: str, route: str, rows: int,
                            max_splits: int = DEFAULT_MAX_SPLITS) -> int:
    rows = max(int(rows), 1)
    if kind == "global" and route == "decode" and int(max_splits) > 1:
        return _align(rows * max_splits * g.heads * g.head_dim * 4) + _align(rows * max_splits * g.heads * 4)
    return _ALIGN


def _is_inkling2_mini_geometry(g: Inkling2Geometry) -> bool:
    return (
        g.heads == INKLING2_MINI.heads
        and g.kv_heads == INKLING2_MINI.kv_heads
        and g.head_dim == INKLING2_MINI.head_dim
        and g.window == INKLING2_MINI.window
        and g.rel_extent == INKLING2_MINI.rel_extent
        and g.ring_rows == INKLING2_MINI.ring_rows
        and g.page_rows == INKLING2_MINI.page_rows
    )


class _GlobalAttention:
    def __init__(self, g: Inkling2Geometry, route: str, max_splits: int, split_chunk: int, kv_format: str):
        self.g, self.route = g, route
        self.max_splits = int(max_splits)
        self.split_chunk = int(split_chunk)
        decode = route == "decode"
        per_row = decode and self.max_splits > 1
        self.runtime_sm_count = per_row
        if not decode and kv_format == "fp8" and _is_inkling2_mini_geometry(g):
            self.attn = InklingMiniGlobalFp8Prefill(softmax_scale=g.softmax_scale)
        else:
            self.attn = InklingGqaAttention(
                heads=g.heads, kv_heads=g.kv_heads, tokens=1 if decode else prefill_tokens(g), window=0, paged=True,
                extent=g.extent("global"), softmax_scale=g.softmax_scale, direct=not per_row, page_rows=g.page_rows,
                ring_rows=g.ring_rows, kv_format=kv_format, per_row_splits=per_row, split_chunk=self.split_chunk)
        self.merge = (
            InklingSplitMerge(heads=g.heads, v_head=g.head_dim, per_row_splits=True, split_chunk=self.split_chunk)
            if per_row else None
        )

    def key(self) -> tuple:
        return (self.attn.key(), None if self.merge is None else self.merge.key(),
                self.route, self.max_splits, self.split_chunk, self.runtime_sm_count)

    @cute.jit
    def run(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
            table_indices: cute.Pointer, seq_first: cute.Pointer, tile_rows: cute.Pointer,
            rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
            table_stride: Int32, tiles: Int32, device_sm_count: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        partials = cute.make_ptr(cutlass.Float32, base, cute.AddressSpace.gmem, assumed_align=16)
        lse_off = base + (Int64(rows) * Int64(self.max_splits * self.g.heads * self.g.head_dim * 4)
                          + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        lse = cute.make_ptr(cutlass.Float32, lse_off, cute.AddressSpace.gmem, assumed_align=16)
        # The ring operands (kv_step, ring_slots, seq_first) are unused on the paged route.
        if cutlass.const_expr(self.merge is not None):
            limit = Int32(self.max_splits)
            if device_sm_count > Int32(0):
                if device_sm_count < Int32(64):
                    limit = Int32(8)
                elif device_sm_count < Int32(128):
                    limit = Int32(16)
                else:
                    limit = Int32(32)
            active_splits = Int32(self.max_splits)
            if limit < active_splits:
                active_splits = limit
            if active_splits < Int32(1):
                active_splits = Int32(1)
            self.attn(q, kv_cache, kv_cache, positions, page_table, table_indices, tile_rows, seq_first, seq_first,
                      rel, sinks, out, partials, lse, rows, table_stride, tiles, active_splits, stream)
            self.merge(partials, lse, positions, sinks, out, rows, active_splits, stream)
        else:
            self.attn(q, kv_cache, kv_cache, positions, page_table, table_indices, tile_rows, seq_first, seq_first, rel,
                      sinks, out, partials, lse, rows, table_stride, tiles, Int32(1), stream)


class _GlobalAttentionDecode(_GlobalAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 table_indices: cute.Pointer, seq_first: cute.Pointer, tile_rows: cute.Pointer,
                 rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_stride: Int32, tiles: Int32, device_sm_count: Int32, stream: cuda.CUstream):
        self.run(q, kv_cache, positions, page_table, table_indices, seq_first, tile_rows, rel, sinks, out, scratch,
                 rows, table_stride, tiles, device_sm_count, stream)


class _GlobalAttentionDirect(_GlobalAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 table_indices: cute.Pointer, seq_first: cute.Pointer, tile_rows: cute.Pointer,
                 rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_stride: Int32, tiles: Int32, stream: cuda.CUstream):
        self.run(q, kv_cache, positions, page_table, table_indices, seq_first, tile_rows, rel, sinks, out, scratch,
                 rows, table_stride, tiles, Int32(0), stream)


class _LocalAttention:
    def __init__(self, g: Inkling2Geometry, route: str):
        self.g, self.route = g, route
        if route == "prefill" and _is_inkling2_mini_geometry(g):
            self.attn = InklingMiniLocalBf16Prefill(softmax_scale=g.softmax_scale)
        else:
            self.attn = InklingGqaAttention(
                heads=g.heads, kv_heads=g.kv_heads, tokens=1 if route == "decode" else prefill_tokens(g),
                window=g.window, paged=False, extent=g.extent("local"), softmax_scale=g.softmax_scale, direct=True,
                page_rows=g.page_rows, ring_rows=g.ring_rows)
        self.commit = MimoRingCommit(record=g.record_elems, ring_rows=g.ring_rows, record_bytes=record_bytes(g))

    def key(self) -> tuple:
        return (self.attn.key(), self.route)

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_step: cute.Pointer, ring: cute.Pointer, positions: cute.Pointer,
                 ring_slots: cute.Pointer, seq_first: cute.Pointer, tile_rows: cute.Pointer, rel: cute.Pointer, sinks: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, tiles: Int32, stream: cuda.CUstream):
        f32 = cute.make_ptr(cutlass.Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        i32 = cute.make_ptr(cutlass.Int32, Int64(seq_first.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.attn(q, ring, kv_step, positions, i32, i32, tile_rows, ring_slots, seq_first, rel, sinks, out, f32, f32, rows,
                  Int32(0), tiles, Int32(1), stream)
        self.commit(kv_step, ring, ring_slots, seq_first, rows, stream)


def compile_inkling2_attention_aot(g: Inkling2Geometry = INKLING2_SMALL, *, kind: str, route: str, max_rows: int,
                                   max_splits: int = DEFAULT_MAX_SPLITS, split_chunk: int = DEFAULT_SPLIT_CHUNK,
                                   kv_format: str = "bf16"):
    """Biased-sink GQA attention of ``kind`` for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check(kind, route, max_rows)
    if kv_format not in ("bf16", "fp8") or (kind == "local" and kv_format != "bf16"):
        raise ValueError("FP8 KV is global only")
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in 1..{MAX_SPLITS}")
    if int(split_chunk) <= 0:
        raise ValueError("split_chunk must be positive")
    n, d, r, extent = g.heads, g.head_dim, g.record_elems, g.extent(kind)
    q = Operand("q", torch.bfloat16, f"[rows,{n},{d}]")
    rel = Operand("rel", torch.bfloat16, f"[rows,{n},{extent}]", align=16,
                  note="additive bias per (row, head, distance back); zero beyond the extent")
    sinks = Operand("sinks", torch.float32, f"[rows,{n}]", align=4, note="unscaled sink logit per (row, head)")
    out = Operand("out", torch.bfloat16, f"[rows,{n},{d}]", "out")
    scratch = Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch")
    runtime_sm_count = kind == "global" and route == "decode" and int(max_splits) > 1
    if kind == "global":
        cls = _GlobalAttentionDecode if runtime_sm_count else _GlobalAttentionDirect
        launch = cls(g, route, max_splits, split_chunk, kv_format)
        operands = (q, Operand("kv_cache", torch.uint8 if kv_format == "fp8" else torch.bfloat16,
                                  f"[pages*{g.page_rows},{record_bytes(g, kv_format)} bytes]"),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("page_table", torch.int32, "[tables,table_stride]", align=4),
                    Operand("table_indices", torch.int32, "[rows]", align=4),
                    Operand("seq_first", torch.int32, "[rows]", align=4),
                    Operand("tile_rows", torch.int32, "[tiles]", align=4), rel, sinks, out, scratch)
        scalars = (Scalar("rows"), Scalar("table_stride"), Scalar("tiles"))
        if runtime_sm_count:
            scalars += (Scalar("device_sm_count"),)
    else:
        launch = _LocalAttention(g, route)
        operands = (q, Operand("kv_step", torch.bfloat16, f"[rows,{r}]"),
                    Operand("ring", torch.bfloat16, f"[rings*{g.ring_rows},{r}]", "inout"),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("ring_slots", torch.int64, "[rows]", align=8),
                    Operand("seq_first", torch.int32, "[rows]", align=4),
                    Operand("tile_rows", torch.int32, "[tiles]", align=4), rel, sinks, out, scratch)
        scalars = (Scalar("rows"), Scalar("tiles"))
    geometry = {"kind": kind, "route": route, "heads": n, "kv_heads": g.kv_heads, "head_dim": d,
                "record_elems": r, "record_bytes": record_bytes(g, kv_format),
                "cache_abi": 2, "kv_format": kv_format, "scale_offset": g.record_elems if kv_format == "fp8" else 0,
                "max_rows": max_rows, "window": g.window if kind == "local" else 0, "extent": extent,
                "ring_rows": g.ring_rows, "page_rows": g.page_rows, "sinks": "per_row_head",
                "max_splits": int(max_splits) if kind == "global" and route == "decode" else 1,
                "split_chunk": int(split_chunk) if runtime_sm_count else 0,
                "runtime_sm_count": runtime_sm_count,
                "rows_per_cta": launch.attn.tokens}
    return compile_program(
        launch, name=f"inkling2_{kind}_attention", operands=operands, scalars=scalars,
        key=("cache_v2", max_rows, kind, route, kv_format, launch.key()),
        geometry=geometry,
        scratch={"scratch": lambda rows: attention_scratch_bytes(g, kind, route, rows, max_splits)},
        doc=__doc__,
    )
