"""Inkling 2 Small attention AOT programs (``inkling2_*``) for cuteafd.

One biased-sink GQA tile (``_inkling2_kernels.InklingGqaAttention``) in four
exports, selected by layer kind and route; see that module for the arithmetic
and :class:`Inkling2Geometry` for the static shapes (48 / 8 heads of 128,
window 1024, bias extent 1024, BF16 records of 4096 bytes).

Inputs common to every program, per live row ``r`` (``rows <= max_rows``):

* ``q``      BF16 ``[rows, 48, 128]``: the L2-clipped (and on global layers
             log-scaled) query.
* ``rel``    BF16 ``[rows, 48, extent]``: ``rel[r, h, d]`` is the additive
             bias of head ``h`` for a key ``d`` positions back (``r_proj`` x
             L2 clip x ``rel_logits_proj.proj``, computed upstream).
* ``sinks``  F32 ``[rows, 48]``: the unscaled sink logit per (token, head).
* ``positions`` I64 ``[rows]``: the row's position in its sequence.
* ``out``    BF16 ``[rows, 48, 128]``: attention output before the gate.

``compile_inkling2_attention_aot(g, kind="global", route=..., max_rows=M)``::

    inkling2_global_attention(q, kv_cache, positions, page_table, rel, sinks, out, scratch,
                              rows, table_stride[, splits], stream)

Paged BF16 records ``kv_cache [pages*64, 2048]``; ``page_table`` is per row
(``[rows, table_stride]``, decode) or one row shared by the whole step
(``table_stride == 0``, prefill: one sequence per step). Decode (one row per
CTA) splits the key range over ``splits <= max_splits`` CTAs and merges the
FP32 partials in ``scratch`` with the sink once (``InklingSplitMerge``);
prefill (10 rows per CTA) attends directly.

``compile_inkling2_attention_aot(g, kind="local", route=..., max_rows=M)``::

    inkling2_local_attention(q, kv_step, ring, positions, ring_slots, seq_first, rel, sinks, out,
                             scratch, rows, stream)

Sliding window over this step's records ``kv_step [rows, 2048]`` and each
sequence's ring (``ring [rings*2048, 2048]``, slot ``ring_slots[row]``);
``seq_first[row]`` is the step's first row of the row's sequence. The step's
records are then committed into the rings (``MimoRingCommit``), so a row's
record must be in ``kv_step`` before the launch, not in the ring.

Scratch: ``attention_scratch_bytes(g, kind, route, rows)``; 1 KiB except for
the global decode partials (``rows * max_splits * 48 * (128 * 4 + 4)``).
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from ._common import INKLING2_SMALL, Inkling2Geometry, Operand, Scalar, compile_program
from ._inkling2_kernels import InklingGqaAttention, InklingSplitMerge
from ._mimo_kernels import MAX_SPLITS, MimoRingCommit

__all__ = [
    "DEFAULT_MAX_SPLITS",
    "attention_scratch_bytes",
    "compile_inkling2_attention_aot",
    "prefill_tokens",
    "record_bytes",
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


def record_bytes(g: Inkling2Geometry) -> int:
    """Bytes of one token's BF16 KV record."""
    return g.record_elems * 2


def prefill_tokens(g: Inkling2Geometry) -> int:
    """Rows per CTA of the prefill route: a 64-row M tile of (token, head) pairs."""
    return 64 // (g.heads // g.kv_heads)


def attention_scratch_bytes(g: Inkling2Geometry, kind: str, route: str, rows: int,
                            max_splits: int = DEFAULT_MAX_SPLITS) -> int:
    rows = max(int(rows), 1)
    if kind == "global" and route == "decode":
        return _align(rows * max_splits * g.heads * g.head_dim * 4) + _align(rows * max_splits * g.heads * 4)
    return _ALIGN


class _GlobalAttention:
    def __init__(self, g: Inkling2Geometry, route: str, max_splits: int):
        self.g, self.route = g, route
        decode = route == "decode"
        self.attn = InklingGqaAttention(
            heads=g.heads, kv_heads=g.kv_heads, tokens=1 if decode else prefill_tokens(g), window=0, paged=True,
            extent=g.extent("global"), softmax_scale=g.softmax_scale, direct=not decode, page_rows=g.page_rows,
            ring_rows=g.ring_rows)
        self.merge = InklingSplitMerge(heads=g.heads, v_head=g.head_dim) if decode else None
        self.max_splits = int(max_splits)

    def key(self) -> tuple:
        return (self.attn.key(), self.route, self.max_splits)

    @cute.jit
    def run(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
            rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
            table_stride: Int32, splits: Int32, stream: cuda.CUstream):
        base = Int64(scratch.toint())
        partials = cute.make_ptr(cutlass.Float32, base, cute.AddressSpace.gmem, assumed_align=16)
        lse_off = base + (Int64(rows) * Int64(self.max_splits * self.g.heads * self.g.head_dim * 4)
                          + Int64(_ALIGN - 1)) // Int64(_ALIGN) * Int64(_ALIGN)
        lse = cute.make_ptr(cutlass.Float32, lse_off, cute.AddressSpace.gmem, assumed_align=16)
        # The ring operands (kv_step, ring_slots, seq_first) are unused on the paged route.
        self.attn(q, kv_cache, kv_cache, positions, page_table, page_table, page_table, rel, sinks, out, partials,
                  lse, rows, table_stride, splits, stream)
        if cutlass.const_expr(self.merge is not None):
            self.merge(partials, lse, sinks, out, rows, splits, stream)


class _GlobalAttentionDecode(_GlobalAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_stride: Int32, splits: Int32, stream: cuda.CUstream):
        clamped = splits
        if clamped > Int32(self.max_splits):
            clamped = Int32(self.max_splits)
        if clamped < Int32(1):
            clamped = Int32(1)
        self.run(q, kv_cache, positions, page_table, rel, sinks, out, scratch, rows, table_stride, clamped, stream)


class _GlobalAttentionPrefill(_GlobalAttention):
    @cute.jit
    def __call__(self, q: cute.Pointer, kv_cache: cute.Pointer, positions: cute.Pointer, page_table: cute.Pointer,
                 rel: cute.Pointer, sinks: cute.Pointer, out: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 table_stride: Int32, stream: cuda.CUstream):
        self.run(q, kv_cache, positions, page_table, rel, sinks, out, scratch, rows, table_stride, Int32(1), stream)


class _LocalAttention:
    def __init__(self, g: Inkling2Geometry, route: str):
        self.g, self.route = g, route
        self.attn = InklingGqaAttention(
            heads=g.heads, kv_heads=g.kv_heads, tokens=1 if route == "decode" else prefill_tokens(g),
            window=g.window, paged=False, extent=g.extent("local"), softmax_scale=g.softmax_scale, direct=True,
            page_rows=g.page_rows, ring_rows=g.ring_rows)
        self.commit = MimoRingCommit(record=g.record_elems, ring_rows=g.ring_rows, record_bytes=record_bytes(g))

    def key(self) -> tuple:
        return (self.attn.key(), self.route)

    @cute.jit
    def __call__(self, q: cute.Pointer, kv_step: cute.Pointer, ring: cute.Pointer, positions: cute.Pointer,
                 ring_slots: cute.Pointer, seq_first: cute.Pointer, rel: cute.Pointer, sinks: cute.Pointer,
                 out: cute.Pointer, scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        f32 = cute.make_ptr(cutlass.Float32, Int64(scratch.toint()), cute.AddressSpace.gmem, assumed_align=16)
        i32 = cute.make_ptr(cutlass.Int32, Int64(seq_first.toint()), cute.AddressSpace.gmem, assumed_align=4)
        self.attn(q, ring, kv_step, positions, i32, ring_slots, seq_first, rel, sinks, out, f32, f32, rows,
                  Int32(0), Int32(1), stream)
        self.commit(kv_step, ring, ring_slots, seq_first, rows, stream)


def compile_inkling2_attention_aot(g: Inkling2Geometry = INKLING2_SMALL, *, kind: str, route: str, max_rows: int,
                                   max_splits: int = DEFAULT_MAX_SPLITS):
    """Biased-sink GQA attention of ``kind`` for ``rows <= max_rows``; see the module docstring."""
    max_rows = _check(kind, route, max_rows)
    if not 1 <= int(max_splits) <= MAX_SPLITS:
        raise ValueError(f"max_splits must be in 1..{MAX_SPLITS}")
    n, d, r, extent = g.heads, g.head_dim, g.record_elems, g.extent(kind)
    q = Operand("q", torch.bfloat16, f"[rows,{n},{d}]")
    rel = Operand("rel", torch.bfloat16, f"[rows,{n},{extent}]", align=2,
                  note="additive bias per (row, head, distance back); zero beyond the extent")
    sinks = Operand("sinks", torch.float32, f"[rows,{n}]", align=4, note="unscaled sink logit per (row, head)")
    out = Operand("out", torch.bfloat16, f"[rows,{n},{d}]", "out")
    scratch = Operand("scratch", torch.uint8, "[attention_scratch_bytes]", "scratch")
    if kind == "global":
        cls = _GlobalAttentionDecode if route == "decode" else _GlobalAttentionPrefill
        launch = cls(g, route, max_splits)
        operands = (q, Operand("kv_cache", torch.bfloat16, f"[pages*{g.page_rows},{r}]"),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("page_table", torch.int32, "[rows|1,table_stride]", align=4), rel, sinks, out, scratch)
        scalars = (Scalar("rows"), Scalar("table_stride"))
        if route == "decode":
            scalars += (Scalar("splits"),)
    else:
        launch = _LocalAttention(g, route)
        operands = (q, Operand("kv_step", torch.bfloat16, f"[rows,{r}]"),
                    Operand("ring", torch.bfloat16, f"[rings*{g.ring_rows},{r}]", "inout"),
                    Operand("positions", torch.int64, "[rows]", align=8),
                    Operand("ring_slots", torch.int64, "[rows]", align=8),
                    Operand("seq_first", torch.int32, "[rows]", align=4), rel, sinks, out, scratch)
        scalars = (Scalar("rows"),)
    return compile_program(
        launch, name=f"inkling2_{kind}_attention", operands=operands, scalars=scalars,
        key=(max_rows, kind, route, launch.key()),
        geometry={"kind": kind, "route": route, "heads": n, "kv_heads": g.kv_heads, "head_dim": d,
                  "record_elems": r, "record_bytes": record_bytes(g), "max_rows": max_rows,
                  "window": g.window if kind == "local" else 0, "extent": extent, "ring_rows": g.ring_rows,
                  "page_rows": g.page_rows, "sinks": "per_row_head",
                  "max_splits": int(max_splits) if kind == "global" and route == "decode" else 1,
                  "rows_per_cta": launch.attn.tokens},
        scratch={"scratch": lambda rows: attention_scratch_bytes(g, kind, route, rows, max_splits)},
        doc=__doc__,
    )
