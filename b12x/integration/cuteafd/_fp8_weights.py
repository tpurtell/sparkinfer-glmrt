"""FP8 E4M3 weights with FP32 128x128 block scales for the GLM programs.

GLM checkpoints store dense weights as E4M3 ``[N, K]`` with one FP32 scale
per 128x128 block (``[ceil(N/128), K/128]``). Re-quantizing those scales to
powers of two costs 0.025 nats, so these kernels apply them exactly:

``MmaFp8Gemv``   decode rows (one thin 16-row tile, then multi-tile rows to the
    program's capacity on the same ``warps``/``groups``): m16n8k16 BF16 MMAs fed
    straight from global memory, weights widened in registers to
    ``bf16(w * s)`` (exactly the BF16 program's dequantized weight), FP32
    accumulation; reads half the weight bytes of the BF16 GEMV.
``TmaFp8Gemm``    the BF16 TMA tensor-core GEMM with E4M3 weight tiles widened
    to ``bf16(w * s)`` in shared memory; bitwise equal to the BF16 kernel over
    dequantized weights (W8A16; block, per-row or K-block-major per-row scales).
``RoutedFp8Projection``  ``MmaFp8Gemv`` up to ``decode_rows`` live rows, the
    BF16 TMA GEMM over the BF16 copy of the weight above.
``Fp8Projection``  a weight held only as E4M3 + scales (no BF16 copy): decode
    programs run ``MmaFp8Gemv`` then ``TmaFp8Gemm``; prefill programs W8A8
    (``BlockFp8Projection``: E4M3 activations per row and 128-K block, the
    official FP8 releases' served numerics) or ``TmaFp8Gemm`` on a launch switch.

A decode projection is row-invariant: every width from 1 to its program's row
capacity is computed by ``MmaFp8Gemv`` tiles that walk K in one order -- a thin
``FP8_GEMV_ROWS``-row tile and a multi-tile GEMV to ``decode_rows`` that keeps
the thin tile's ``warps``/``groups``. ``m_tiles`` only decides which row tile an
accumulator set belongs to; it does not change the K-blocks, the k-slot order or
the warp reduction order, so a row's output bits do not move when the batch
crosses the 16-row tile boundary. A capacity beyond one CTA's share of tiles
(``gemv_cta_tiles``) is covered by the launch grid's M dimension, one CTA per
``m_tiles``-row band, and the fold still sums the same warps in the same order,
so splitting a wide batch across CTAs is not a reduction change either. Only
widths past the program's capacity (a prefill step, or ``decode_rows`` itself for
a program exported wider) take the TMA GEMM, which reduces K in a different
order.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    ld_shared_v2_u32,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v2_f32,
    st_shared_v4_u32,
)

from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection, _ld_cached, _ld_stream

__all__ = ["CandidateFp8Projection", "CandidateMmaFp8Gemv", "CandidateTmaFp8Gemm", "FP8_DECODE_ROWS",
           "FP8_GEMV_ROWS", "Fp8Projection", "candidate_gemv_cta_tiles", "candidate_gemv_groups", "decode_tile_n",
           "MmaFp8Gemv", "RoutedFp8Projection", "TmaFp8Gemm", "check_w8_mode", "fp8_only_operands", "fp8_operands",
           "fp8_scale_shape", "gemv_warps", "projection", "w8_scalars"]


# Most decode rows ``MmaFp8Gemv``'s thin tile computes (its M-tile ceiling): the
# register-resident GEMV every decode program runs at C1 and at narrow batches.
FP8_GEMV_ROWS = 16
# Row capacity every decode program reads the FP8 weights through one canonical
# GEMV reduction up to, unless the caller declares its own. The multi-tile GEMV
# tiles carry the thin tile's (warps, groups), so the switch at FP8_GEMV_ROWS is
# a reduction-order-preserving split of the same K walk.
FP8_DECODE_ROWS = 64


def fp8_scale_shape(n: int, k: int) -> tuple[int, int]:
    """FP32 block-scale grid of an ``[n, k]`` E4M3 weight."""
    return (int(n) + 127) // 128, (int(k) + 127) // 128


@dsl_user_op
def _ld_f32(address, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.f32 $0, [$1];", "=f,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


class RoutedFp8Projection:
    """``out = x @ w^T`` for decode rows: ``MmaFp8Gemv`` over the E4M3 weight
    and its FP32 block scales up to ``decode_rows`` live rows, the BF16 TMA
    GEMM over ``w_bf16`` (the same weight dequantized) above; the branch is
    on the ``rows`` scalar.

    Both GEMV tiles are built with one ``warps``/``groups``, so a row's bits do
    not move when the batch crosses the ``FP8_GEMV_ROWS`` tile boundary."""

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, decode_rows: int | None = None):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        self.decode_rows = FP8_DECODE_ROWS if decode_rows is None else int(decode_rows)
        if self.decode_rows < FP8_GEMV_ROWS:
            raise ValueError(f"an FP8 projection covers at least {FP8_GEMV_ROWS} rows")
        warps = gemv_warps(self.k)
        self.fp8 = MmaFp8Gemv(self.n, self.k, max_rows=FP8_GEMV_ROWS, warps=warps, out_dtype=out_dtype)
        self.fp8_wide = MmaFp8Gemv(self.n, self.k, max_rows=self.decode_rows, warps=warps,
                                   out_dtype=out_dtype) if self.decode_rows > FP8_GEMV_ROWS else None
        self.max_fp8_rows = self.decode_rows
        self.large = TmaBf16Projection(self.n, self.k, out_dtype=out_dtype, tiles_by_rows=())

    def key(self) -> tuple:
        return (None if self.fp8_wide is None else self.fp8_wide.key(), self.max_fp8_rows,
                tuple(sorted(self.large.config.items())))

    @cute.jit
    def __call__(self, x: cute.Pointer, w_bf16: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        if rows <= Int32(self.max_fp8_rows):
            if cutlass.const_expr(self.fp8_wide is None):
                self.fp8(x, w_fp8, scale, out, rows, stream)
            else:
                if rows <= Int32(FP8_GEMV_ROWS):
                    self.fp8(x, w_fp8, scale, out, rows, stream)
                else:
                    self.fp8_wide(x, w_fp8, scale, out, rows, stream)
        else:
            self.large(x, w_bf16, out, rows, stream)


class _Bf16Projection:
    """The BF16 projection behind the FP8 call signature (FP8 operands unused)."""

    def __init__(self, n: int, k: int):
        from ._glm_kernels import glm_projection

        self.inner = glm_projection(n, k)

    def key(self) -> tuple:
        return self.inner.key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w_bf16: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.inner(x, w_bf16, out, rows, stream)


def projection(n: int, k: int, fp8: bool, *, decode_rows: int | None = None):
    """A GLM projection with the ``(x, w, w_fp8, scale, out, rows, stream)`` call.

    ``decode_rows`` (FP8 only): the GEMV's row capacity, defaulting to
    ``FP8_DECODE_ROWS``. One ``projection`` serves both a decode program (whose
    rows are the decode stream, canonical to its capacity) and a prefill program
    (whose narrow steps are prefill tails, not the decode stream), so a caller
    that wants a narrower capacity passes it explicitly."""
    return RoutedFp8Projection(n, k, decode_rows=decode_rows) if fp8 else _Bf16Projection(n, k)


class Fp8Projection:
    """``out = x @ dequant(w)^T`` over a weight held only as E4M3 ``[N, K]`` plus FP32 scales
    (no BF16 copy). ``row_scales``: one scale per output row and 128-wide K block instead of
    the checkpoint's 128x128 grid (``[N, K/128]`` in decode programs, K-block major
    ``[K/128, N]`` in prefill programs).

    Decode programs (``prefill_rows`` None): ``MmaFp8Gemv`` for ``rows <= min(fp8_rows,
    max_gemv_rows)`` (the ``gemv_rows``-row tile to ``gemv_rows``, the multi-tile GEMV with the
    same ``warps``/``groups`` above it), ``TmaFp8Gemm`` above: W8A16, bitwise the BF16 programs
    over ``bf16(w * s)`` weights. Both GEMV tiles walk K in one order, so the decode range is
    row-invariant to ``max_gemv_rows``.

    Prefill programs: W8A8 when ``fp8_rows & prefill_mask`` is nonzero (``BlockFp8Projection``:
    E4M3 activations per row and 128-K block with FP32 ``amax / 448`` scales, the official FP8
    releases' served numerics), else ``TmaFp8Gemm`` (W8A16). ``qscratch`` holds the quantized
    rows (``prefill_scratch_bytes``)."""

    def __init__(self, n: int, k: int, *, prefill_rows: int | None = None, row_scales: bool = False,
                 kmajor: bool = False, gemv_rows: int = 16, wide_rows: int = 0, warps: int | None = None,
                 groups: int = 4, prefill_mask: int = 0xFF, out_dtype=cutlass.BFloat16,
                 wide_warps: int | None = None, wide_groups: int | None = None,
                 decode_rows: int | None = None):
        self.n, self.k = int(n), int(k)
        self.row_scales = bool(row_scales)
        # Per-row scales K-block major in decode programs too (one copy serves both).
        self.kmajor = bool(kmajor) or (self.row_scales and prefill_rows is not None)
        self.prefill_mask = int(prefill_mask)
        self.prefill = prefill_rows is not None
        self.gemv = self.gemv_wide = self.w8a8 = None
        if self.prefill:
            from ._glmf_fp8 import BlockFp8Projection

            if out_dtype != cutlass.BFloat16 and self.prefill_mask:
                raise ValueError("non-BF16 prefill output requires W8A16 (prefill_mask=0)")
            self.w8a8 = BlockFp8Projection(self.n, self.k, int(prefill_rows), row_scales=self.row_scales) \
                if self.prefill_mask else None
            # 128-row tiles: the E4M3 tile is widened once per 128 rows (SM120 at 325 W, 4096 rows,
            # 6144x16384: 4 warps 3650 us, 8 warps 2779 us; BF16 TMA 128x128 tiles 2244 us).
            self.w8a16 = TmaFp8Gemm(self.n, self.k, scales="row_kmajor" if self.row_scales else "block",
                                    compute_warps=8, num_stages=3, out_dtype=out_dtype)
            self.max_gemv_rows = 0
        else:
            warps = gemv_warps(self.k) if warps is None else int(warps)
            groups = int(groups)
            self.gemv = MmaFp8Gemv(self.n, self.k, max_rows=int(gemv_rows), warps=warps, groups=groups,
                                   row_scales=self.row_scales, kmajor=self.kmajor, out_dtype=out_dtype)
            # The decode capacity: FP8_DECODE_ROWS unless the caller exports its program wider;
            # wide_rows only raises it, so a narrow wide_rows cannot leave decode rows above it
            # on the TMA route (whose K order differs).
            cap = max(FP8_DECODE_ROWS if decode_rows is None else int(decode_rows), int(wide_rows),
                      int(gemv_rows))
            if (warps if wide_warps is None else int(wide_warps),
                    groups if wide_groups is None else int(wide_groups)) != (warps, groups):
                raise ValueError("the wide GEMV tile must keep the thin tile's warps and groups: "
                                 "one K order covers the decode range")
            self.gemv_wide = MmaFp8Gemv(self.n, self.k, max_rows=cap, warps=warps, groups=groups,
                                        row_scales=self.row_scales, kmajor=self.kmajor,
                                        out_dtype=out_dtype) if cap > int(gemv_rows) else None
            self.gemv_rows = int(gemv_rows)
            self.max_gemv_rows = cap if self.gemv_wide is not None else int(gemv_rows)
            tile_n = decode_tile_n(self.n)
            self.w8a16 = TmaFp8Gemm(self.n, self.k, scales=("row_kmajor" if self.kmajor else "row")
                                    if self.row_scales else "block", tile_n=tile_n,
                                    num_stages=4 if tile_n == 128 else 6, out_dtype=out_dtype)

    def key(self) -> tuple:
        return (self.n, self.k, self.row_scales, self.kmajor, self.prefill_mask, self.max_gemv_rows,
                None if self.gemv is None else self.gemv.key(),
                None if self.gemv_wide is None else self.gemv_wide.key(),
                None if self.w8a8 is None else self.w8a8.key(), self.w8a16.key())

    def prefill_scratch_bytes(self, rows: int) -> int:
        from ._glmf_fp8 import quant_scratch_bytes

        return quant_scratch_bytes(self.k, rows) if self.prefill else 0

    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, fp8_rows: Int32, qscratch: Int64, stream: cuda.CUstream):
        if cutlass.const_expr(self.prefill):
            if cutlass.const_expr(self.w8a8 is None):
                self.w8a16(x, w_fp8, scale, out, rows, stream)
            else:
                if (fp8_rows & Int32(self.prefill_mask)) != Int32(0):
                    self.w8a8(x, w_fp8, scale, out, qscratch, rows, stream)
                else:
                    self.w8a16(x, w_fp8, scale, out, rows, stream)
        else:
            limit = fp8_rows
            if limit > Int32(self.max_gemv_rows):
                limit = Int32(self.max_gemv_rows)
            if rows <= limit:
                if cutlass.const_expr(self.gemv_wide is None):
                    self.gemv(x, w_fp8, scale, out, rows, stream)
                else:
                    if rows <= Int32(self.gemv_rows):
                        self.gemv(x, w_fp8, scale, out, rows, stream)
                    else:
                        self.gemv_wide(x, w_fp8, scale, out, rows, stream)
            else:
                self.w8a16(x, w_fp8, scale, out, rows, stream)


def decode_tile_n(n: int) -> int:
    """``TmaFp8Gemm`` N tile for decode steps above the GEMV rows (17-64 rows: a single M tile,
    so the CTA count is the N tiles): 128 from 16384 outputs, 64 from 4096, else 32. SM120 at
    325 W, L2-cold, 17/64 rows, us (BF16 TMA -> FP8 128 / chosen): GLM 5.3 o_proj 6144x16384
    143/145 -> 188/191 / 125/131 (64), q_b 16384x2048 51/53 -> 37/35 (128), q_a|kv_a 2624x6144
    59/59 -> 74/76 / 37/43 (32), dense down 6144x12288 127/127 -> 145/145 / 94/100 (64)."""
    n = int(n)
    if n >= 16384:
        return 128
    return 64 if n >= 4096 and n % 64 == 0 else 32


def check_w8_mode(mode: str) -> str:
    """``fp8_only`` of the compile functions: ``"decode"`` or ``"prefill"``."""
    if mode not in ("decode", "prefill"):
        raise ValueError(f"fp8_only must be 'decode' or 'prefill', got {mode!r}")
    return mode


def w8_scalars(prefill: bool) -> tuple:
    """Launch scalars of FP8-only programs: ``rows``, plus the prefill W8A8 switch."""
    from ._common import Scalar

    if prefill:
        return (Scalar("rows"), Scalar("fp8_rows", note="prefill: nonzero runs W8A8 (E4M3 activations per "
                                                        "row and 128-K block), 0 W8A16"))
    return (Scalar("rows"),)


def fp8_only_operands(name: str, n: int, k: int, *, row_scales: bool = False, prefill: bool = False,
                      kmajor: bool = False) -> tuple:
    """The ``{name}_fp8`` E4M3 ``[n, k]`` and ``{name}_scale`` FP32 operands of an ``Fp8Projection``:
    the checkpoint's 128x128 grid ``[ceil(n/128), k/128]``, or with ``row_scales`` one scale per
    row and 128-K block (``[n, k/128]``; prefill programs, and ``kmajor`` decode programs, take
    them K-block major as ``{name}_kscale [k/128, n]``)."""
    from ._common import Operand
    import torch

    kb = -(-int(k) // 128)
    if row_scales and (prefill or kmajor):
        return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]", note="checkpoint E4M3 weight"),
                Operand(f"{name}_kscale", torch.float32, f"[{kb},{n}]", align=16,
                        note="per-row x 128-K FP32 scales, K-block major"))
    if row_scales:
        shape, note = f"[{n},{kb}]", "per-row x 128-K FP32 scales"
    else:
        shape, note = f"[{-(-int(n) // 128)},{kb}]", "checkpoint weight_scale_inv (FP32 128x128 block scales)"
    return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]", note="checkpoint E4M3 weight"),
            Operand(f"{name}_scale", torch.float32, shape, align=16 if prefill else 4, note=note))


def fp8_operands(name: str, n: int, k: int) -> tuple:
    """The ``{name}_fp8`` / ``{name}_scale`` pointer operands of an FP8 weight."""
    from ._common import Operand
    import torch

    rows, cols = fp8_scale_shape(n, k)
    return (Operand(f"{name}_fp8", torch.float8_e4m3fn, f"[{n},{k}]", note="checkpoint E4M3 weight"),
            Operand(f"{name}_scale", torch.float32, f"[{rows},{cols}]", align=4,
                    note="checkpoint weight_scale_inv (FP32 128x128 block scales)"))


def gemv_warps(k: int) -> int:
    """K-splitting warps per CTA (SM120, L2-cold weights, us at M=1/5/16):
    hidden-width inputs (K=6144) take 8 (q_a+kv_a 14/16/18 vs 18/18/23 with 4;
    dense gate/up 98/98/100 vs 104/104/106), longer and shorter K take 4
    (o_proj 68/70/76 vs 76/76/80; dense down 53/57/63 vs 57/59/59)."""
    return 8 if int(k) == 6144 else 4


def gemv_cta_tiles(n: int, k: int, *, groups: int, warps: int) -> int:
    """Most 16-row M-tiles one ``MmaFp8Gemv`` CTA holds (its fragment registers), leaving the
    rest to the launch grid's M dimension. Measured per shape at SM120/325 W, L2-cold, us
    at 17/32/64 rows, 1/2/4 tiles per CTA (see the note in ``MmaFp8Gemv``).

    Requalified per shape (engine #210): two tiles pay only where the doubled M grid fills
    the machine -- the k=2560 tall-N projections (16480x2560 and 13952x2560, 0.73-0.94 of
    four tiles at every width, 0.71-0.87 of the one-CTA route). Vocab-wide heads (n > 16480)
    keep four tiles so a 64-row decode step reads the 606 MiB head weights once instead of
    twice. k=2048 stays four: its two
    16384-wide consumers build the same kernel key yet split on the arm (0.85 with two
    tiles for the plain one against 1.06-1.12 at 48/64 rows for the routed one), and a
    thinner K loses outright (16384x1536 at 1.18/1.40 of four at 48/64 rows)."""
    k = int(k)
    return 2 if 2048 < k <= 2560 and int(n) <= 16480 else 4


# ---------------------------------------------------------------------------
# TMA tensor-core GEMM over FP8 weights (BF16 MMA, FP32 accumulation)
# ---------------------------------------------------------------------------


@dsl_user_op
def _e4m3x8_scaled_bf16(lo, hi, s, *, loc=None, ip=None):
    """bf16(e4m3 * s) for eight packed E4M3 values, as four bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip),
         Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e0, e1, e2, e3, h0, h1, h2, h3, h4, h5, h6, h7;
            .reg .b32 p0, p1, p2, p3;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            mov.b32 {e0, e1}, $4;
            mov.b32 {e2, e3}, $5;
            cvt.rn.f16x2.e4m3x2 p0, e0;
            cvt.rn.f16x2.e4m3x2 p1, e1;
            cvt.rn.f16x2.e4m3x2 p2, e2;
            cvt.rn.f16x2.e4m3x2 p3, e3;
            mov.b32 {h0, h1}, p0;
            mov.b32 {h2, h3}, p1;
            mov.b32 {h4, h5}, p2;
            mov.b32 {h6, h7}, p3;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.f32.f16 f4, h4;
            cvt.f32.f16 f5, h5;
            cvt.f32.f16 f6, h6;
            cvt.f32.f16 f7, h7;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
            cvt.rn.bf16x2.f32 $2, f5, f4;
            cvt.rn.bf16x2.f32 $3, f7, f6;
        }
        """,
        "=r,=r,=r,=r,r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


class TmaFp8Gemm:
    """``out = x @ (w * s)^T`` over E4M3 ``w [N, K]`` with FP32 scales per 128-wide K block.

    The Bf16PrefillKernel pipeline (TMA producer warp, warp MMA m16n8k16,
    FP32 accumulators) with the weight tile fetched as E4M3 (half the bytes).
    After each stage lands, the compute warps widen it to ``bf16(w * s)``
    into a swizzled BF16 tile and run the MMAs from it, so results equal the
    BF16 program over ``bf16(w * s)`` weights (W8A16, bitwise).

    ``scales``: ``"block"`` the checkpoint's 128x128 grid ``[ceil(N/128),
    K/128]`` (one scale per 128x64 tile); ``"row"`` one scale per output row
    and K block, ``[N, K/128]``; ``"row_kmajor"`` the same scales K-block
    major, ``[K/128, N]`` (the block-FP8 prefill GEMM's layout).
    """

    tile_k = 64
    buffer_align_bytes = 1024

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16, compute_warps: int = 4,
                 num_stages: int = 4, scales: str = "block", tile_n: int = 128):
        self.n, self.k = int(n), int(k)
        self.out_dtype = out_dtype
        # Narrower N tiles give narrow, long-K projections more CTAs (each widens fewer rows).
        if int(tile_n) not in (32, 64, 128):
            raise ValueError("FP8 TMA GEMM tile_n is 32, 64 or 128")
        self.tile_n = int(tile_n)
        if scales not in ("block", "row", "row_kmajor"):
            raise ValueError(f"scales must be block, row or row_kmajor, got {scales!r}")
        self.scales = scales
        self.num_compute_warps = int(compute_warps)
        self.compute_threads = 32 * self.num_compute_warps
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        self.tile_m = 16 * self.num_compute_warps
        self.num_stages = int(num_stages)
        if self.k % 128:
            raise ValueError("FP8 TMA GEMM needs K % 128 == 0")
        self.k_tiles = self.k // self.tile_k
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.k_blocks = self.k // 128
        self.chunks = self.tile_n * self.tile_k // 8 // self.compute_threads

    def key(self) -> tuple:
        return (self.n, self.k, str(self.out_dtype), self.num_compute_warps, self.num_stages, self.scales,
                self.tile_n)

    def _tiled_mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    @cute.jit
    def _scale(self, scale: cute.Pointer, n_tile: Int32, n: Int32, k_tile: Int32) -> Float32:
        """The scale of weight row ``n`` of ``n_tile`` at K tile ``k_tile`` (rows past N clamp to
        the last one: their E4M3 values are TMA zero fill)."""
        kb = k_tile // Int32(2)
        if cutlass.const_expr(self.scales == "block"):
            block = n_tile // Int32(128 // self.tile_n)
            return _ld_f32(Int64(scale.toint()) + (Int64(block) * Int64(self.k_blocks) + Int64(kb)) * Int64(4))
        row = n_tile * Int32(self.tile_n) + n
        if row > Int32(self.n - 1):
            row = Int32(self.n - 1)
        if cutlass.const_expr(self.scales == "row"):
            return _ld_f32(Int64(scale.toint()) + (Int64(row) * Int64(self.k_blocks) + Int64(kb)) * Int64(4))
        return _ld_f32(Int64(scale.toint()) + (Int64(kb) * Int64(self.n) + Int64(row)) * Int64(4))

    def _layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, self.tile_k),
            cutlass.BFloat16)
        s_a = cute.tile_to_shape(atom, (self.tile_m, self.tile_k, self.num_stages), order=(0, 1, 2))
        s_b = cute.tile_to_shape(atom, (self.tile_n, self.tile_k), order=(0, 1))
        s_w = cute.make_layout((self.tile_n, self.tile_k, self.num_stages),
                               stride=(self.tile_k, 1, self.tile_n * self.tile_k))
        return s_a, s_b, s_w

    def _storage(self, s_a, s_b, s_w):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sA": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_a)], self.buffer_align_bytes],
            "sB": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(s_b)], self.buffer_align_bytes],
            "sW": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, cute.cosize(s_w)], self.buffer_align_bytes],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        a_t = cute.make_tensor(x, cute.make_layout((rows, self.k), stride=(self.k, 1)))
        w8 = cute.make_ptr(cutlass.Uint8, Int64(w.toint()), cute.AddressSpace.gmem, assumed_align=16)
        w_t = cute.make_tensor(w8, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
        o_t = cute.make_tensor(out, cute.make_layout((rows, self.n), stride=(self.n, 1)))
        s_a, s_b, s_w = self._layouts()
        storage = self._storage(s_a, s_b, s_w)
        tma_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a_t, cute.slice_(s_a, (None, None, 0)),
            (self.tile_m, self.tile_k), num_multicast=1)
        tma_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), w_t, cute.slice_(s_w, (None, None, 0)),
            (self.tile_n, self.tile_k), num_multicast=1)
        grid_m = (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(tma_tensor_a, tma_tensor_w, o_t, scale, tma_a, tma_w, s_a, s_b, s_w,
                    self._tiled_mma(), storage, rows).launch(
            grid=(grid_m, self.n_tiles, 1), block=[self.num_threads, 1, 1], stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight: cute.Tensor, output: cute.Tensor, scale: cute.Pointer,
               tma_atom_a: cute.CopyAtom, tma_atom_w: cute.CopyAtom, s_a_layout: cute.ComposedLayout,
               s_b_layout: cute.ComposedLayout, s_w_layout: cute.Layout, tiled_mma: cute.TiledMma,
               SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_w)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        s_w = storage.sW.get_tensor(s_w_layout)
        tma_bytes = (self.tile_m * 2 + self.tile_n) * self.tile_k
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None))
        g_w = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2),
                                           cute.group_modes(g_a, 0, 2))
        t_ws, t_wg = cpasync.tma_partition(tma_atom_w, 0, cta_layout, cute.group_modes(s_w, 0, 2),
                                           cute.group_modes(g_w, 0, 2))
        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            thr_mma = tiled_mma.get_slice(tidx)
            t_csa = thr_mma.partition_A(s_a)
            t_csb = thr_mma.partition_B(s_b)
            t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
            t_crb = thr_mma.make_fragment_B(t_csb)
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc.fill(0.0)
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                tiled_mma).get_slice(tidx)
            t_ssa = copy_a.partition_S(s_a)
            t_ssb = copy_b.partition_S(s_b)
            w_base = shared_ptr_to_u32(storage.sW.data_ptr())
            b_base = shared_ptr_to_u32(storage.sB.data_ptr())
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                s = self._scale(scale, n_tile, Int32(0), k_tile)
                load_pipeline.consumer_wait(consumer_state)
                stage = w_base + Int32(consumer_state.index) * Int32(self.tile_n * self.tile_k)
                for i in cutlass.range_constexpr(self.chunks):
                    chunk = Int32(i * self.compute_threads) + Int32(tidx)
                    n = chunk // Int32(self.tile_k // 8)
                    c = chunk % Int32(self.tile_k // 8)
                    if cutlass.const_expr(self.scales != "block"):
                        s = self._scale(scale, n_tile, n, k_tile)
                    lo, hi = ld_shared_v2_u32(stage + n * Int32(self.tile_k) + c * Int32(8))
                    v0, v1, v2, v3 = _e4m3x8_scaled_bf16(lo, hi, s)
                    # 128-byte swizzle: 16-byte chunk c of row n sits at chunk c ^ (n % 8).
                    st_shared_v4_u32(b_base + n * Int32(128) + ((c ^ (n % Int32(8))) * Int32(16)), v0, v1, v2, v3)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
                shared_a = t_ssa[None, None, None, consumer_state.index]
                target_a = copy_a.retile(t_cra)
                target_b = copy_b.retile(t_crb)
                cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                cute.copy(copy_b, t_ssb[None, None, 0], target_b[None, None, 0])
                for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                    if kk < cute.size(shared_a.shape[2]) - 1:
                        cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                        cute.copy(copy_b, t_ssb[None, None, kk + 1], target_b[None, None, kk + 1])
                    cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
                cute.arch.fence_view_async_shared()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            for index in cutlass.range_constexpr(cute.size(acc)):
                coord = coordinates[index]
                token = m_tile * Int32(self.tile_m) + coord[0]
                column = n_tile * Int32(self.tile_n) + coord[1]
                if token < num_tokens and column < Int32(self.n):
                    output[Int64(token), Int64(column)] = acc[index].to(output.element_type)
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, t_ag[(None, m_tile, k_tile)], t_as[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_w, t_wg[(None, n_tile, k_tile)], t_ws[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


# ---------------------------------------------------------------------------
# Tensor-core GEMV over FP8 weights for decode rows (<= 64)
# ---------------------------------------------------------------------------


@dsl_user_op
def _e4m3x4_scaled_bf16x2x2(word, s, *, loc=None, ip=None):
    """bf16(e4m3 * s) for four packed E4M3 (byte 0 first) as two bf16x2 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip), Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23, h0, h1, h2, h3;
            .reg .b32 p01, p23;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.f16x2.e4m3x2 p01, e01;
            cvt.rn.f16x2.e4m3x2 p23, e23;
            mov.b32 {h0, h1}, p01;
            mov.b32 {h2, h3}, p23;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            mul.rn.f32 f0, f0, $3;
            mul.rn.f32 f1, f1, $3;
            mul.rn.f32 f2, f2, $3;
            mul.rn.f32 f3, f3, $3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class MmaFp8Gemv:
    """Decode-row GEMM on tensor cores straight from global memory.

    A CTA owns ``8 * groups`` output columns and ``m_tiles`` 16-row tiles of the
    ``max_rows``-row capacity; its ``warps`` split K. A row's tile is the launch
    grid's second dimension when the program's capacity needs more tiles than one
    CTA holds (``tiles_by_rows``/``total_m_tiles`` are the program's; ``m_tiles``,
    from ``gemv_cta_tiles``, is one CTA's share), so a wider batch adds CTAs over
    the same rows and the same K order instead of lengthening one CTA's work.
    Per 64-wide K chunk a warp runs four m16n8k16 BF16 MMAs per (16-row tile,
    8-column group). MMA k-slot ``s`` of step ``t`` holds logical ``k =
    16 * quad(s) + 4t + idx(s)`` (``idx`` of slots ``2j, 2j+1, 2j+8, 2j+9``
    is 0..3), so each lane reads one contiguous 16-byte weight run per group
    (row ``lane/4``) and two contiguous 32-byte activation runs per tile,
    reused by every group. Weights enter the MMA as ``bf16(w * s)`` (the BF16
    program's dequantized weight; one scale per 128-wide K block since a
    CTA's columns share a 128-row block), FP32 accumulation; warps reduce
    through shared memory (one tile at a time when a CTA holds more than two,
    ``sync_threads`` between tiles, and the same ascending warp order either
    way); output BF16 or FP32 ``[rows, N]``. Nothing here depends on the live
    ``rows``: a row's K blocks, k-slots and cross-warp fold are fixed by
    ``(n, k, warps, groups)``, so every width up to ``max_rows`` is bitwise one
    launch per row.

    ``row_scales``: the scale grid is ``[N, K/128]`` FP32, one scale per
    output row and 128-wide K block (per-channel or row-block quantization of
    a BF16 weight); each lane scales its own row's weights. ``kmajor``: those
    per-row scales stored K-block major, ``[K/128, N]``.
    """

    def __init__(self, n: int, k: int, *, max_rows: int = 64, warps: int = 4, groups: int = 4,
                 out_dtype=cutlass.BFloat16, row_scales: bool = False, kmajor: bool = False):
        self.row_scales = bool(row_scales)
        self.kmajor = bool(kmajor)
        if self.kmajor and not self.row_scales:
            raise ValueError("K-block-major scales are per-row scales")
        self.n, self.k = int(n), int(k)
        self.warps, self.groups = int(warps), int(groups)
        self.cols = 8 * self.groups
        self.total_m_tiles = (int(max_rows) + 15) // 16
        self.m_tiles = min(self.total_m_tiles, gemv_cta_tiles(self.n, self.k, groups=self.groups,
                                                             warps=self.warps))
        self.multi_m_grid = self.total_m_tiles > self.m_tiles
        self.out_dtype = out_dtype
        if self.n % self.cols or 128 % self.cols or self.k % (128 * self.warps):
            raise ValueError("MMA FP8 GEMV needs N % (8*groups) == 0, 8*groups dividing 128 and "
                             "K a multiple of 128 * warps")
        self.k_per_warp = self.k // self.warps
        self.blocks = self.k_per_warp // 128
        self.k_blocks = self.k // 128
        self.tile_frags = self.groups * 4
        self.frags = self.m_tiles * self.tile_frags
        # One row tile's worth of partials when a CTA holds more than two (a 4-tile
        # 8-warp CTA otherwise needs 32 KiB for a fold that reuses one tile anyway).
        self.smem_frags = self.frags if self.m_tiles <= 2 else self.tile_frags

    def key(self) -> tuple:
        return (self.n, self.k, self.warps, self.groups, self.total_m_tiles, self.m_tiles,
                str(self.out_dtype), self.row_scales, self.kmajor)

    def _storage(self):
        class Storage:
            pass

        Storage.__annotations__ = {
            "partial": cute.struct.Align[cute.struct.MemRange[Float32, self.warps * self.smem_frags * 32], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        if const_expr(self.multi_m_grid):
            cta_rows = 16 * self.m_tiles
            grid_m = (rows + Int32(cta_rows - 1)) // Int32(cta_rows)
            self.kernel(x, w, scale, out, rows).launch(
                grid=(self.n // self.cols, grid_m, 1), block=(32 * self.warps, 1, 1), stream=stream)
        else:
            self.kernel(x, w, scale, out, rows).launch(
                grid=(self.n // self.cols, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _load_block(self, dest: cute.Tensor, w_row: Int64, k_off: Int64):
        for chunk in cutlass.range_constexpr(2):
            for gi in cutlass.range_constexpr(self.groups):
                words = _ld_stream(w_row + Int64(gi * 8 * self.k) + k_off + Int64(chunk * 64))
                for t in cutlass.range_constexpr(4):
                    dest[(chunk * self.groups + gi) * 4 + t] = words[t]

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        g = lane // Int32(4)
        j = lane % Int32(4)
        n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        m0 = (Int32(cute.arch.block_idx()[1]) * Int32(16 * self.m_tiles)
              if const_expr(self.multi_m_grid) else Int32(0))
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        partial = storage.partial.get_tensor(cute.make_layout((self.warps * self.smem_frags * 32,)))
        acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.frags):
            acc[i] = Float32(0.0)
        k_begin = Int64(warp_id) * Int64(self.k_per_warp)
        w_row = Int64(w.toint()) + (n0 + Int64(g)) * Int64(self.k) + k_begin + Int64(16) * Int64(j)
        x_base = Int64(x.toint()) + (k_begin + Int64(16) * Int64(j)) * Int64(2)
        # Scale of (row group gi, K block): one per CTA column block, or per lane row.
        s_rows = self.groups if self.row_scales else 1
        s_first = (n0 + Int64(g)) if self.row_scales else (n0 // Int64(128))
        # Byte strides between a lane's scales of consecutive K blocks and row groups.
        s_kb = Int64(self.n * 4) if self.kmajor else Int64(4)
        s_gi = Int64(8 * 4) if self.kmajor else Int64(8 * self.k_blocks * 4)
        s_row = Int64(scale.toint()) + s_first * (Int64(4) if self.kmajor else Int64(self.k_blocks * 4)) \
            + (k_begin // Int64(128)) * s_kb
        live_tiles = ((rows - m0 + Int32(15)) // Int32(16)
                      if const_expr(self.multi_m_grid) else (rows + Int32(15)) // Int32(16))
        per_block = 2 * self.groups * 4
        # Register double buffer: block b+1's weights and scales load while b computes.
        cur = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
        nxt = cute.make_rmem_tensor(cute.make_layout((per_block,), stride=(1,)), Uint32)
        s_cur = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)
        s_nxt = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)
        self._load_block(cur, w_row, Int64(0))
        for si in cutlass.range_constexpr(s_rows):
            s_cur[si] = _ld_f32(s_row + Int64(si) * s_gi)
        for block in cutlass.range(self.blocks, unroll=1):
            for si in cutlass.range_constexpr(s_rows):
                s_nxt[si] = s_cur[si]
            if block + 1 < self.blocks:
                self._load_block(nxt, w_row, Int64(block + 1) * Int64(128))
                for si in cutlass.range_constexpr(s_rows):
                    s_nxt[si] = _ld_f32(s_row + Int64(si) * s_gi + Int64(block + 1) * s_kb)
            for chunk in cutlass.range_constexpr(2):
                k_off = Int64(block) * Int64(128) + Int64(chunk * 64)
                bw = cute.make_rmem_tensor(cute.make_layout((8 * self.groups,), stride=(1,)), Uint32)
                for gi in cutlass.range_constexpr(self.groups):
                    for t in cutlass.range_constexpr(4):
                        b0, b1 = _e4m3x4_scaled_bf16x2x2(cur[(chunk * self.groups + gi) * 4 + t],
                                                          s_cur[gi if self.row_scales else 0])
                        bw[8 * gi + 2 * t] = b0
                        bw[8 * gi + 2 * t + 1] = b1
                for mt in cutlass.range_constexpr(self.m_tiles):
                    if Int32(mt) < live_tiles:
                        r_lo = m0 + Int32(16 * mt) + g
                        r_hi = r_lo + Int32(8)
                        xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                        for i in cutlass.range_constexpr(16):
                            xa[i] = Uint32(0)
                        if r_lo < rows:
                            at = x_base + (Int64(r_lo) * Int64(self.k) + k_off) * Int64(2)
                            lo = _ld_cached(at)
                            hi = _ld_cached(at + Int64(16))
                            for i in cutlass.range_constexpr(4):
                                xa[i] = lo[i]
                                xa[4 + i] = hi[i]
                        if r_hi < rows:
                            at = x_base + (Int64(r_hi) * Int64(self.k) + k_off) * Int64(2)
                            lo = _ld_cached(at)
                            hi = _ld_cached(at + Int64(16))
                            for i in cutlass.range_constexpr(4):
                                xa[8 + i] = lo[i]
                                xa[12 + i] = hi[i]
                        for gi in cutlass.range_constexpr(self.groups):
                            f = 4 * (mt * self.groups + gi)
                            for t in cutlass.range_constexpr(4):
                                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                    acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                    xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                    bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                                acc[f] = d0
                                acc[f + 1] = d1
                                acc[f + 2] = d2
                                acc[f + 3] = d3
            for i in cutlass.range_constexpr(per_block):
                cur[i] = nxt[i]
            for si in cutlass.range_constexpr(s_rows):
                s_cur[si] = s_nxt[si]
        out_bytes = 2 if const_expr(self.out_dtype == cutlass.BFloat16) else 4
        if const_expr(self.m_tiles <= 2):
            for i in cutlass.range_constexpr(self.frags):
                partial[(warp_id * Int32(self.frags) + Int32(i)) * Int32(32) + lane] = acc[i]
            cute.arch.sync_threads()
            # Every warp stores a share of the fragments.
            for i in cutlass.range_constexpr(self.frags):
                if Int32(i % self.warps) == warp_id:
                    total = Float32(0.0)
                    for src in cutlass.range_constexpr(self.warps):
                        total = total + partial[(Int32(src * self.frags + i)) * Int32(32) + lane]
                    mt = i // (4 * self.groups)
                    gi = (i // 4) % self.groups
                    row = m0 + Int32(16 * mt) + g + Int32(8 * ((i % 4) // 2))
                    col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                    if row < rows:
                        address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(out_bytes)
                        target = cute.make_ptr(self.out_dtype, address, cute.AddressSpace.gmem,
                                               assumed_align=out_bytes)
                        target[0] = total.to(self.out_dtype)
        else:
            # One row tile's partials at a time through the tile-sized buffer, same order.
            for mt in cutlass.range_constexpr(self.m_tiles):
                if Int32(mt) < live_tiles:
                    if const_expr(mt > 0):
                        cute.arch.sync_threads()
                    for i in cutlass.range_constexpr(self.tile_frags):
                        partial[(warp_id * Int32(self.tile_frags) + Int32(i)) * Int32(32) + lane] = (
                            acc[mt * self.tile_frags + i])
                    cute.arch.sync_threads()
                    for i in cutlass.range_constexpr(self.tile_frags):
                        if Int32(i % self.warps) == warp_id:
                            total = Float32(0.0)
                            for src in cutlass.range_constexpr(self.warps):
                                total = total + partial[(Int32(src * self.tile_frags + i)) * Int32(32) + lane]
                            gi = i // 4
                            row = m0 + Int32(16 * mt) + g + Int32(8 * ((i % 4) // 2))
                            col = n0 + Int64(8 * gi) + Int64(2) * Int64(j) + Int64(i % 2)
                            if row < rows:
                                address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(out_bytes)
                                target = cute.make_ptr(self.out_dtype, address, cute.AddressSpace.gmem,
                                                       assumed_align=out_bytes)
                                target[0] = total.to(self.out_dtype)


# ---------------------------------------------------------------------------
# Candidate FP8 dense kernels (bit-exact scheduling & pipeline optimizations)
# ---------------------------------------------------------------------------

_DEVICE_CAPS_CACHE: dict[int, tuple[int, int]] = {}


def _query_device_caps() -> tuple[int, int]:
    """Query ``(sm_count, l2_bytes)`` of the active CUDA device at runtime (never bake 188 or L2)."""
    err, ctx_dev = cuda.cuCtxGetDevice()
    dev_id = int(ctx_dev) if err == cuda.CUresult.CUDA_SUCCESS else 0
    cached = _DEVICE_CAPS_CACHE.get(dev_id)
    if cached is not None:
        return cached
    err_sm, sm_count = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_id
    )
    err_l2, l2_bytes = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev_id
    )
    sm = int(sm_count) if err_sm == cuda.CUresult.CUDA_SUCCESS and int(sm_count) > 0 else 170
    l2 = int(l2_bytes) if err_l2 == cuda.CUresult.CUDA_SUCCESS and int(l2_bytes) > 0 else (64 << 20)
    if err == cuda.CUresult.CUDA_SUCCESS and err_sm == cuda.CUresult.CUDA_SUCCESS:
        _DEVICE_CAPS_CACHE[dev_id] = (sm, l2)
    return sm, l2


def candidate_gemv_groups(n: int, k: int, *, warps: int = 4, groups: int = 4, row_scales: bool = False) -> int:
    """Select column groups per CTA using runtime SM count while preserving exact K reduction."""
    n = int(n)
    sm_count, _ = _query_device_caps()
    # When groups=2 (16 cols/CTA) yields fewer than 2.5 CTAs/SM, drop to groups=1 (8 cols/CTA)
    # so every SM has multiple resident CTAs (e.g. N <= 6144 on 170-SM 5090 and 188-SM PRO 6000).
    if n // 16 < (5 * sm_count) // 2 and n % 8 == 0:
        return 1
    if n % 16 == 0:
        return 2
    return 1 if n % 8 == 0 else int(groups)


def candidate_gemv_cta_tiles(n: int, k: int, *, groups: int, warps: int) -> int:
    """Select 16-row M-tiles per CTA for wide decode GEMV using runtime L2 capacity."""
    n, k = int(n), int(k)
    _, l2_bytes = _query_device_caps()
    # When the weight matrix fits within half of L2 and N <= 16480, 2 M-tiles per CTA halves
    # accumulator register pressure at 32 rows (grid_m=1) and doubles CTA parallelism at 64 rows
    # while keeping weight reuse in L2. Larger weights keep 4 tiles so 64 rows streams DRAM once.
    if n * k <= l2_bytes // 2 and n <= 16480:
        return 2
    return 4


@dsl_user_op
def _ld_stream_v4(address, *, loc=None, ip=None):
    """16-byte streaming weight load with side effects to preserve software-pipelined schedule."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::evict_first.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_cached_v4(address, *, loc=None, ip=None):
    """16-byte L1-cached activation load with side effects to preserve prefetch schedule."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::evict_last.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_f32_se(address, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(address).ir_value(loc=loc, ip=ip)], "ld.global.nc.f32 $0, [$1];", "=f,l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _st_shared_v2_f32(smem_addr, v0, v1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip),
         Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)


@dsl_user_op
def _ld_shared_v2_f32(smem_addr, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)))


@dsl_user_op
def _pack_bf16x2_rn(x0, x1, *, loc=None, ip=None) -> Uint32:
    """Pack two FP32 values into bf16x2 with exact round-to-nearest-even (no satfinite)."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def _e4m3x8_scaled_bf16_fast(lo, hi, s, *, loc=None, ip=None):
    """Bit-identical bf16(e4m3 * s) for eight E4M3 bytes using SM120 cvt.rn.bf16x2.e4m3x2."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip),
         Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e0, e1, e2, e3;
            .reg .b32 p0, p1, p2, p3, b0, b1, b2, b3, b4, b5, b6, b7;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
            mov.b32 {e0, e1}, $4;
            mov.b32 {e2, e3}, $5;
            cvt.rn.bf16x2.e4m3x2 p0, e0;
            cvt.rn.bf16x2.e4m3x2 p1, e1;
            cvt.rn.bf16x2.e4m3x2 p2, e2;
            cvt.rn.bf16x2.e4m3x2 p3, e3;
            shl.b32 b0, p0, 16;
            and.b32 b1, p0, 0xffff0000;
            shl.b32 b2, p1, 16;
            and.b32 b3, p1, 0xffff0000;
            shl.b32 b4, p2, 16;
            and.b32 b5, p2, 0xffff0000;
            shl.b32 b6, p3, 16;
            and.b32 b7, p3, 0xffff0000;
            mov.b32 f0, b0;
            mov.b32 f1, b1;
            mov.b32 f2, b2;
            mov.b32 f3, b3;
            mov.b32 f4, b4;
            mov.b32 f5, b5;
            mov.b32 f6, b6;
            mov.b32 f7, b7;
            mul.rn.f32 f0, f0, $6;
            mul.rn.f32 f1, f1, $6;
            mul.rn.f32 f2, f2, $6;
            mul.rn.f32 f3, f3, $6;
            mul.rn.f32 f4, f4, $6;
            mul.rn.f32 f5, f5, $6;
            mul.rn.f32 f6, f6, $6;
            mul.rn.f32 f7, f7, $6;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
            cvt.rn.bf16x2.f32 $2, f5, f4;
            cvt.rn.bf16x2.f32 $3, f7, f6;
        }
        """,
        "=r,=r,=r,=r,r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _e4m3x4_scaled_bf16x2x2_fast(word, s, *, loc=None, ip=None):
    """Bit-identical bf16(e4m3 * s) for four E4M3 bytes using SM120 cvt.rn.bf16x2.e4m3x2."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip), Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23;
            .reg .b32 p01, p23, b0, b1, b2, b3;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.bf16x2.e4m3x2 p01, e01;
            cvt.rn.bf16x2.e4m3x2 p23, e23;
            shl.b32 b0, p01, 16;
            and.b32 b1, p01, 0xffff0000;
            shl.b32 b2, p23, 16;
            and.b32 b3, p23, 0xffff0000;
            mov.b32 f0, b0;
            mov.b32 f1, b1;
            mov.b32 f2, b2;
            mov.b32 f3, b3;
            mul.rn.f32 f0, f0, $3;
            mul.rn.f32 f1, f1, $3;
            mul.rn.f32 f2, f2, $3;
            mul.rn.f32 f3, f3, $3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)
    return (Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)))


class CandidateTmaFp8Gemm(TmaFp8Gemm):
    """Bit-identical ``TmaFp8Gemm`` with software-pipelined ``sW -> sB`` widening overlapped
    with Tensor Core MMAs, hoisted swizzle offsets, 128-K scale reuse, and packed bf16x2 stores."""

    def key(self) -> tuple:
        return ("candidate_tma_fp8_gemm", super().key())

    @cute.kernel
    def kernel(self, source: cute.Tensor, weight: cute.Tensor, output: cute.Tensor, scale: cute.Pointer,
               tma_atom_a: cute.CopyAtom, tma_atom_w: cute.CopyAtom, s_a_layout: cute.ComposedLayout,
               s_b_layout: cute.ComposedLayout, s_w_layout: cute.Layout, tiled_mma: cute.TiledMma,
               SharedStorage: cutlass.Constexpr, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_w)
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_a = storage.sA.get_tensor(s_a_layout.outer, swizzle=s_a_layout.inner)
        s_b = storage.sB.get_tensor(s_b_layout.outer, swizzle=s_b_layout.inner)
        s_w = storage.sW.get_tensor(s_w_layout)
        tma_bytes = (self.tile_m * 2 + self.tile_n) * self.tile_k
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_compute_warps),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()
        g_a = cute.local_tile(source, (self.tile_m, self.tile_k), (None, None))
        g_w = cute.local_tile(weight, (self.tile_n, self.tile_k), (None, None))
        cta_layout = cute.make_layout(1)
        t_as, t_ag = cpasync.tma_partition(tma_atom_a, 0, cta_layout, cute.group_modes(s_a, 0, 2),
                                           cute.group_modes(g_a, 0, 2))
        t_ws, t_wg = cpasync.tma_partition(tma_atom_w, 0, cta_layout, cute.group_modes(s_w, 0, 2),
                                           cute.group_modes(g_w, 0, 2))
        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            thr_mma = tiled_mma.get_slice(tidx)
            t_csa = thr_mma.partition_A(s_a)
            t_csb = thr_mma.partition_B(s_b)
            t_cra = thr_mma.make_fragment_A(t_csa[None, None, None, 0])
            t_crb = thr_mma.make_fragment_B(t_csb)
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.tile_m, self.tile_n)), Float32)
            acc.fill(0.0)
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
                tiled_mma).get_slice(tidx)
            t_ssa = copy_a.partition_S(s_a)
            t_ssb = copy_b.partition_S(s_b)
            w_base = shared_ptr_to_u32(storage.sW.data_ptr())
            b_base = shared_ptr_to_u32(storage.sB.data_ptr())
            # Hoist constant thread swizzle offsets outside the K-loop.
            n0_tidx = Int32(tidx) // Int32(8)
            c_tidx = Int32(tidx) % Int32(8)
            n_step = self.compute_threads // 8
            w_tidx = w_base + Int32(tidx) * Int32(8)
            b_tidx = b_base + n0_tidx * Int32(128) + ((c_tidx ^ (n0_tidx % Int32(8))) * Int32(16))
            s_arr = cute.make_rmem_tensor(cute.make_layout((self.chunks,), stride=(1,)), Float32)
            s_base = cute.make_rmem_tensor(cute.make_layout((self.chunks,), stride=(1,)), Int64)
            s_kb_stride = Int64(self.n * 4) if self.scales == "row_kmajor" else Int64(4)
            if cutlass.const_expr(self.scales == "block"):
                block_idx = n_tile // Int32(128 // self.tile_n)
                s_base[0] = Int64(scale.toint()) + Int64(block_idx) * Int64(self.k_blocks * 4)
                s0 = _ld_f32_se(s_base[0])
                for i in cutlass.range_constexpr(self.chunks):
                    s_arr[i] = s0
            else:
                for i in cutlass.range_constexpr(self.chunks):
                    row = n_tile * Int32(self.tile_n) + n0_tidx + Int32(i * n_step)
                    if row > Int32(self.n - 1):
                        row = Int32(self.n - 1)
                    if cutlass.const_expr(self.scales == "row"):
                        s_base[i] = Int64(scale.toint()) + Int64(row) * Int64(self.k_blocks * 4)
                    else:
                        s_base[i] = Int64(scale.toint()) + Int64(row) * Int64(4)
                    s_arr[i] = _ld_f32_se(s_base[i])
            # Prologue: wait for k_tile = 0 and widen sW[0] -> sB with batched shared loads.
            load_pipeline.consumer_wait(consumer_state)
            stage0 = w_tidx + Int32(consumer_state.index) * Int32(self.tile_n * self.tile_k)
            w_lo = cute.make_rmem_tensor(cute.make_layout((self.chunks,), stride=(1,)), Uint32)
            w_hi = cute.make_rmem_tensor(cute.make_layout((self.chunks,), stride=(1,)), Uint32)
            for i in cutlass.range_constexpr(self.chunks):
                w_lo[i], w_hi[i] = ld_shared_v2_u32(stage0 + Int32(i * self.compute_threads * 8))
            for i in cutlass.range_constexpr(self.chunks):
                v0, v1, v2, v3 = _e4m3x8_scaled_bf16_fast(w_lo[i], w_hi[i], s_arr[i])
                st_shared_v4_u32(b_tidx + Int32(i * n_step * 128), v0, v1, v2, v3)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
            # Mainloop over k_tile = 0 .. k_tiles - 2: overlap k_tile + 1 widening with kk = 2, 3 MMAs.
            for k_tile in cutlass.range(self.k_tiles - 1, unroll_full=False):
                shared_a = t_ssa[None, None, None, consumer_state.index]
                target_a = copy_a.retile(t_cra)
                target_b = copy_b.retile(t_crb)
                cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
                cute.copy(copy_b, t_ssb[None, None, 0], target_b[None, None, 0])
                # kk = 0
                cute.copy(copy_a, shared_a[None, None, 1], target_a[None, None, 1])
                cute.copy(copy_b, t_ssb[None, None, 1], target_b[None, None, 1])
                cute.gemm(thr_mma, acc, t_cra[None, None, 0], t_crb[None, None, 0], acc)
                # kk = 1
                cute.copy(copy_a, shared_a[None, None, 2], target_a[None, None, 2])
                cute.copy(copy_b, t_ssb[None, None, 2], target_b[None, None, 2])
                cute.gemm(thr_mma, acc, t_cra[None, None, 1], t_crb[None, None, 1], acc)
                # kk = 2: issue final ldmatrix for kk = 3, then immediately release stage & sync sB.
                cute.copy(copy_a, shared_a[None, None, 3], target_a[None, None, 3])
                cute.copy(copy_b, t_ssb[None, None, 3], target_b[None, None, 3])
                cute.arch.fence_view_async_shared()
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                next_k = k_tile + Int32(1)
                if (next_k & Int32(1)) == Int32(0):
                    kb_off = Int64(next_k // Int32(2)) * s_kb_stride
                    if cutlass.const_expr(self.scales == "block"):
                        s_next = _ld_f32_se(s_base[0] + kb_off)
                        for i in cutlass.range_constexpr(self.chunks):
                            s_arr[i] = s_next
                    else:
                        for i in cutlass.range_constexpr(self.chunks):
                            s_arr[i] = _ld_f32_se(s_base[i] + kb_off)
                load_pipeline.consumer_wait(consumer_state)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
                next_stage = w_tidx + Int32(consumer_state.index) * Int32(self.tile_n * self.tile_k)
                for i in cutlass.range_constexpr(self.chunks):
                    w_lo[i], w_hi[i] = ld_shared_v2_u32(next_stage + Int32(i * self.compute_threads * 8))
                cute.gemm(thr_mma, acc, t_cra[None, None, 2], t_crb[None, None, 2], acc)
                half_chunks = self.chunks // 2
                for i in cutlass.range_constexpr(half_chunks):
                    v0, v1, v2, v3 = _e4m3x8_scaled_bf16_fast(w_lo[i], w_hi[i], s_arr[i])
                    st_shared_v4_u32(b_tidx + Int32(i * n_step * 128), v0, v1, v2, v3)
                # kk = 3: run final MMA while finishing the remaining sW -> sB chunks.
                cute.gemm(thr_mma, acc, t_cra[None, None, 3], t_crb[None, None, 3], acc)
                for i in cutlass.range_constexpr(self.chunks - half_chunks):
                    idx = half_chunks + i
                    v0, v1, v2, v3 = _e4m3x8_scaled_bf16_fast(w_lo[idx], w_hi[idx], s_arr[idx])
                    st_shared_v4_u32(b_tidx + Int32(idx * n_step * 128), v0, v1, v2, v3)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.compute_threads)
            # Postlude: consume final k_tile = k_tiles - 1 (already widened in sB).
            shared_a = t_ssa[None, None, None, consumer_state.index]
            target_a = copy_a.retile(t_cra)
            target_b = copy_b.retile(t_crb)
            cute.copy(copy_a, shared_a[None, None, 0], target_a[None, None, 0])
            cute.copy(copy_b, t_ssb[None, None, 0], target_b[None, None, 0])
            for kk in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
                if kk < cute.size(shared_a.shape[2]) - 1:
                    cute.copy(copy_a, shared_a[None, None, kk + 1], target_a[None, None, kk + 1])
                    cute.copy(copy_b, t_ssb[None, None, kk + 1], target_b[None, None, kk + 1])
                cute.gemm(thr_mma, acc, t_cra[None, None, kk], t_crb[None, None, kk], acc)
            cute.arch.fence_view_async_shared()
            load_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
            # Epilogue: packed 32-bit bf16x2 stores for BF16 output.
            coordinates = thr_mma.partition_C(cute.make_identity_tensor((self.tile_m, self.tile_n)))
            if cutlass.const_expr(self.out_dtype == cutlass.BFloat16):
                out_base = Int64(output.iterator.toint())
                for pair_idx in cutlass.range_constexpr(cute.size(acc) // 2):
                    idx0 = 2 * pair_idx
                    coord = coordinates[idx0]
                    token = m_tile * Int32(self.tile_m) + coord[0]
                    column = n_tile * Int32(self.tile_n) + coord[1]
                    if token < num_tokens and column < Int32(self.n):
                        packed = _pack_bf16x2_rn(acc[idx0], acc[idx0 + 1])
                        byte_off = (Int64(token) * Int64(self.n) + Int64(column)) * Int64(2)
                        st_global_u32(out_base + byte_off, packed)
            else:
                for index in cutlass.range_constexpr(cute.size(acc)):
                    coord = coordinates[index]
                    token = m_tile * Int32(self.tile_m) + coord[0]
                    column = n_tile * Int32(self.tile_n) + coord[1]
                    if token < num_tokens and column < Int32(self.n):
                        output[Int64(token), Int64(column)] = acc[index].to(output.element_type)
        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            for k_tile in cutlass.range(self.k_tiles, unroll_full=False):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, t_ag[(None, m_tile, k_tile)], t_as[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_w, t_wg[(None, n_tile, k_tile)], t_ws[(None, producer_state.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state))
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class CandidateMmaFp8Gemv(MmaFp8Gemv):
    """Bit-identical ``MmaFp8Gemv`` with 4-chunk rotating ping-pong pipeline (zero MOV copies,
    3 chunks continuously in flight), immediate per-group weight reload, SM120 direct
    ``cvt.rn.bf16x2.e4m3x2`` dequant, clamped/prefetched activation loads, and vectorized
    ``v2.f32`` shared fold + packed ``bf16x2`` stores."""

    def __init__(self, n: int, k: int, *, max_rows: int = 64, warps: int = 4, groups: int = 4,
                 out_dtype=cutlass.BFloat16, row_scales: bool = False, kmajor: bool = False,
                 cta_tiles: int | None = None):
        super().__init__(n, k, max_rows=max_rows, warps=warps, groups=groups, out_dtype=out_dtype,
                         row_scales=row_scales, kmajor=kmajor)
        target_tiles = (candidate_gemv_cta_tiles(self.n, self.k, groups=self.groups, warps=self.warps)
                        if cta_tiles is None else int(cta_tiles))
        self.m_tiles = min(self.total_m_tiles, target_tiles)
        self.multi_m_grid = self.total_m_tiles > self.m_tiles
        self.frags = self.m_tiles * self.tile_frags
        # Use single-barrier fold whenever total partial SMEM is <= 16 KiB.
        self.smem_frags = self.frags if (self.warps * self.frags * 32 * 4 <= 16384) else self.tile_frags

    def key(self) -> tuple:
        return ("candidate_mma_fp8_gemv", super().key(), self.smem_frags)

    @cute.jit
    def _load_chunk_v4(self, dest: cute.Tensor, w_row: Int64, k_off: Int64):
        for gi in cutlass.range_constexpr(self.groups):
            w0, w1, w2, w3 = _ld_stream_v4(w_row + Int64(gi * 8 * self.k) + k_off)
            dest[gi * 4 + 0] = w0
            dest[gi * 4 + 1] = w1
            dest[gi * 4 + 2] = w2
            dest[gi * 4 + 3] = w3

    @cute.jit
    def _load_scale_se(self, s_dest: cute.Tensor, s_row: Int64, block_idx: Int64, s_gi: Int64, s_kb: Int64):
        s_rows = self.groups if self.row_scales else 1
        base = s_row + block_idx * s_kb
        for si in cutlass.range_constexpr(s_rows):
            s_dest[si] = _ld_f32_se(base + Int64(si) * s_gi)

    @cute.jit
    def _consume_chunk(self, acc: cute.Tensor, c_buf: cute.Tensor, s_buf: cute.Tensor,
                       x_lo_base: cute.Tensor, x_hi_base: cute.Tensor,
                       w_row: Int64, s_row: Int64, s_gi: Int64, s_kb: Int64,
                       m0: Int32, rows: Int32, live_tiles: Int32,
                       k_off: Int64, next_k_off: Int64, next_block_idx: Int64,
                       reload_w: cutlass.Constexpr, reload_s: cutlass.Constexpr):
        x_byte_off = k_off * Int64(2)
        if const_expr(self.m_tiles == 1):
            lo0, lo1, lo2, lo3 = _ld_cached_v4(x_lo_base[0] + x_byte_off)
            lo4, lo5, lo6, lo7 = _ld_cached_v4(x_lo_base[0] + x_byte_off + Int64(16))
            hi0, hi1, hi2, hi3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            hi4, hi5, hi6, hi7 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
            if m0 + Int32(8) < rows:
                hi0, hi1, hi2, hi3 = _ld_cached_v4(x_hi_base[0] + x_byte_off)
                hi4, hi5, hi6, hi7 = _ld_cached_v4(x_hi_base[0] + x_byte_off + Int64(16))
            xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
            xa[0], xa[1], xa[2], xa[3] = lo0, lo1, lo2, lo3
            xa[4], xa[5], xa[6], xa[7] = lo4, lo5, lo6, lo7
            xa[8], xa[9], xa[10], xa[11] = hi0, hi1, hi2, hi3
            xa[12], xa[13], xa[14], xa[15] = hi4, hi5, hi6, hi7
            for gi in cutlass.range_constexpr(self.groups):
                s_val = s_buf[gi if self.row_scales else 0]
                f = 4 * gi
                for t in cutlass.range_constexpr(4):
                    b0, b1 = _e4m3x4_scaled_bf16x2x2_fast(c_buf[gi * 4 + t], s_val)
                    d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                        acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                        xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                        b0, b1)
                    acc[f] = d0
                    acc[f + 1] = d1
                    acc[f + 2] = d2
                    acc[f + 3] = d3
                if const_expr(reload_w):
                    w0, w1, w2, w3 = _ld_stream_v4(w_row + Int64(gi * 8 * self.k) + next_k_off)
                    c_buf[gi * 4 + 0] = w0
                    c_buf[gi * 4 + 1] = w1
                    c_buf[gi * 4 + 2] = w2
                    c_buf[gi * 4 + 3] = w3
            if const_expr(reload_s):
                self._load_scale_se(s_buf, s_row, next_block_idx, s_gi, s_kb)
        else:
            bw = cute.make_rmem_tensor(cute.make_layout((8 * self.groups,), stride=(1,)), Uint32)
            for gi in cutlass.range_constexpr(self.groups):
                s_val = s_buf[gi if self.row_scales else 0]
                for t in cutlass.range_constexpr(4):
                    b0, b1 = _e4m3x4_scaled_bf16x2x2_fast(c_buf[gi * 4 + t], s_val)
                    bw[8 * gi + 2 * t] = b0
                    bw[8 * gi + 2 * t + 1] = b1
                if const_expr(reload_w):
                    w0, w1, w2, w3 = _ld_stream_v4(w_row + Int64(gi * 8 * self.k) + next_k_off)
                    c_buf[gi * 4 + 0] = w0
                    c_buf[gi * 4 + 1] = w1
                    c_buf[gi * 4 + 2] = w2
                    c_buf[gi * 4 + 3] = w3
            if const_expr(reload_s):
                self._load_scale_se(s_buf, s_row, next_block_idx, s_gi, s_kb)
            for mt in cutlass.range_constexpr(self.m_tiles):
                if Int32(mt) < live_tiles:
                    lo0, lo1, lo2, lo3 = _ld_cached_v4(x_lo_base[mt] + x_byte_off)
                    lo4, lo5, lo6, lo7 = _ld_cached_v4(x_lo_base[mt] + x_byte_off + Int64(16))
                    hi0, hi1, hi2, hi3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    hi4, hi5, hi6, hi7 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if m0 + Int32(16 * mt + 8) < rows:
                        hi0, hi1, hi2, hi3 = _ld_cached_v4(x_hi_base[mt] + x_byte_off)
                        hi4, hi5, hi6, hi7 = _ld_cached_v4(x_hi_base[mt] + x_byte_off + Int64(16))
                    xa = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                    xa[0], xa[1], xa[2], xa[3] = lo0, lo1, lo2, lo3
                    xa[4], xa[5], xa[6], xa[7] = lo4, lo5, lo6, lo7
                    xa[8], xa[9], xa[10], xa[11] = hi0, hi1, hi2, hi3
                    xa[12], xa[13], xa[14], xa[15] = hi4, hi5, hi6, hi7
                    for gi in cutlass.range_constexpr(self.groups):
                        f = 4 * (mt * self.groups + gi)
                        for t in cutlass.range_constexpr(4):
                            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                                acc[f], acc[f + 1], acc[f + 2], acc[f + 3],
                                xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1],
                                bw[8 * gi + 2 * t], bw[8 * gi + 2 * t + 1])
                            acc[f] = d0
                            acc[f + 1] = d1
                            acc[f + 2] = d2
                            acc[f + 3] = d3

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, scale: cute.Pointer, out: cute.Pointer, rows: Int32):
        tidx = Int32(cute.arch.thread_idx()[0])
        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        g = lane // Int32(4)
        j = lane % Int32(4)
        n0 = Int64(cute.arch.block_idx()[0]) * Int64(self.cols)
        m0 = (Int32(cute.arch.block_idx()[1]) * Int32(16 * self.m_tiles)
              if const_expr(self.multi_m_grid) else Int32(0))
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self._storage())
        smem_base = shared_ptr_to_u32(storage.partial.data_ptr())
        acc = cute.make_rmem_tensor(cute.make_layout((self.frags,), stride=(1,)), Float32)
        for i in cutlass.range_constexpr(self.frags):
            acc[i] = Float32(0.0)
        k_begin = Int64(warp_id) * Int64(self.k_per_warp)
        w_row = Int64(w.toint()) + (n0 + Int64(g)) * Int64(self.k) + k_begin + Int64(16) * Int64(j)
        x_base = Int64(x.toint()) + (k_begin + Int64(16) * Int64(j)) * Int64(2)
        s_rows = self.groups if self.row_scales else 1
        s_first = (n0 + Int64(g)) if self.row_scales else (n0 // Int64(128))
        s_kb = Int64(self.n * 4) if self.kmajor else Int64(4)
        s_gi = Int64(8 * 4) if self.kmajor else Int64(8 * self.k_blocks * 4)
        s_row = Int64(scale.toint()) + s_first * (Int64(4) if self.kmajor else Int64(self.k_blocks * 4)) \
            + (k_begin // Int64(128)) * s_kb
        live_tiles = ((rows - m0 + Int32(15)) // Int32(16)
                      if const_expr(self.multi_m_grid) else (rows + Int32(15)) // Int32(16))
        max_row = rows - Int32(1)
        if max_row < Int32(0):
            max_row = Int32(0)
        x_lo_base = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int64)
        x_hi_base = cute.make_rmem_tensor(cute.make_layout((self.m_tiles,), stride=(1,)), Int64)
        for mt in cutlass.range_constexpr(self.m_tiles):
            r_lo = m0 + Int32(16 * mt) + g
            if r_lo > max_row:
                r_lo = max_row
            r_hi = m0 + Int32(16 * mt) + g + Int32(8)
            if r_hi > max_row:
                r_hi = max_row
            x_lo_base[mt] = x_base + Int64(r_lo) * Int64(self.k * 2)
            x_hi_base[mt] = x_base + Int64(r_hi) * Int64(self.k * 2)

        chunk_words = 4 * self.groups
        b0_c0 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        b0_c1 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        b1_c0 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        b1_c1 = cute.make_rmem_tensor(cute.make_layout((chunk_words,), stride=(1,)), Uint32)
        s0 = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)
        s1 = cute.make_rmem_tensor(cute.make_layout((s_rows,), stride=(1,)), Float32)

        if const_expr(self.blocks == 1):
            self._load_chunk_v4(b0_c0, w_row, Int64(0))
            self._load_chunk_v4(b0_c1, w_row, Int64(64))
            self._load_scale_se(s0, s_row, Int64(0), s_gi, s_kb)
            self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                m0, rows, live_tiles, Int64(0), Int64(0), Int64(0), False, False)
            self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                m0, rows, live_tiles, Int64(64), Int64(0), Int64(0), False, False)
        else:
            self._load_chunk_v4(b0_c0, w_row, Int64(0))
            self._load_chunk_v4(b0_c1, w_row, Int64(64))
            self._load_scale_se(s0, s_row, Int64(0), s_gi, s_kb)
            self._load_chunk_v4(b1_c0, w_row, Int64(128))
            self._load_chunk_v4(b1_c1, w_row, Int64(192))
            self._load_scale_se(s1, s_row, Int64(1), s_gi, s_kb)
            tail_blocks = 2 if (self.blocks % 2 == 0) else 3
            main_pairs = (self.blocks - tail_blocks) // 2
            if const_expr(main_pairs > 0):
                for pair in cutlass.range(main_pairs, unroll=1):
                    b = Int64(pair) * Int64(2)
                    self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, b * Int64(128), (b + Int64(2)) * Int64(128),
                                        b + Int64(2), True, False)
                    self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, b * Int64(128) + Int64(64),
                                        (b + Int64(2)) * Int64(128) + Int64(64), b + Int64(2), True, True)
                    self._consume_chunk(acc, b1_c0, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, (b + Int64(1)) * Int64(128),
                                        (b + Int64(3)) * Int64(128), b + Int64(3), True, False)
                    self._consume_chunk(acc, b1_c1, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                        m0, rows, live_tiles, (b + Int64(1)) * Int64(128) + Int64(64),
                                        (b + Int64(3)) * Int64(128) + Int64(64), b + Int64(3), True, True)
            b_tail = Int64(2 * main_pairs)
            if const_expr(tail_blocks == 2):
                self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b1_c0, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b1_c1, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)
            else:
                self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128), (b_tail + Int64(2)) * Int64(128),
                                    b_tail + Int64(2), True, False)
                self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, b_tail * Int64(128) + Int64(64),
                                    (b_tail + Int64(2)) * Int64(128) + Int64(64), b_tail + Int64(2), True, True)
                self._consume_chunk(acc, b1_c0, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b1_c1, s1, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(1)) * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b0_c0, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(2)) * Int64(128), Int64(0), Int64(0), False, False)
                self._consume_chunk(acc, b0_c1, s0, x_lo_base, x_hi_base, w_row, s_row, s_gi, s_kb,
                                    m0, rows, live_tiles, (b_tail + Int64(2)) * Int64(128) + Int64(64), Int64(0), Int64(0), False, False)

        if const_expr(self.smem_frags == self.frags):
            pairs = self.frags // 2
            for p in cutlass.range_constexpr(pairs):
                addr = smem_base + ((warp_id * Int32(pairs) + Int32(p)) * Int32(32) + lane) * Int32(8)
                _st_shared_v2_f32(addr, acc[2 * p], acc[2 * p + 1])
            cute.arch.sync_threads()
            for p in cutlass.range_constexpr(pairs):
                if Int32(p % self.warps) == warp_id:
                    total0 = Float32(0.0)
                    total1 = Float32(0.0)
                    for src in cutlass.range_constexpr(self.warps):
                        addr = smem_base + (Int32(src * pairs + p) * Int32(32) + lane) * Int32(8)
                        v0, v1 = _ld_shared_v2_f32(addr)
                        total0 = total0 + v0
                        total1 = total1 + v1
                    mt = p // (2 * self.groups)
                    gi = (p // 2) % self.groups
                    row = m0 + Int32(16 * mt) + g + Int32(8 * (p % 2))
                    col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                    if row < rows:
                        if const_expr(self.out_dtype == cutlass.BFloat16):
                            address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(2)
                            st_global_u32(address, _pack_bf16x2_rn(total0, total1))
                        else:
                            address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(4)
                            st_global_v2_f32(address, total0, total1)
        else:
            tile_pairs = self.tile_frags // 2
            for mt in cutlass.range_constexpr(self.m_tiles):
                if Int32(mt) < live_tiles:
                    if const_expr(mt > 0):
                        cute.arch.sync_threads()
                    for p in cutlass.range_constexpr(tile_pairs):
                        addr = smem_base + ((warp_id * Int32(tile_pairs) + Int32(p)) * Int32(32) + lane) * Int32(8)
                        base_idx = mt * self.tile_frags + 2 * p
                        _st_shared_v2_f32(addr, acc[base_idx], acc[base_idx + 1])
                    cute.arch.sync_threads()
                    for p in cutlass.range_constexpr(tile_pairs):
                        if Int32(p % self.warps) == warp_id:
                            total0 = Float32(0.0)
                            total1 = Float32(0.0)
                            for src in cutlass.range_constexpr(self.warps):
                                addr = smem_base + (Int32(src * tile_pairs + p) * Int32(32) + lane) * Int32(8)
                                v0, v1 = _ld_shared_v2_f32(addr)
                                total0 = total0 + v0
                                total1 = total1 + v1
                            gi = p // 2
                            row = m0 + Int32(16 * mt) + g + Int32(8 * (p % 2))
                            col = n0 + Int64(8 * gi) + Int64(2) * Int64(j)
                            if row < rows:
                                if const_expr(self.out_dtype == cutlass.BFloat16):
                                    address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(2)
                                    st_global_u32(address, _pack_bf16x2_rn(total0, total1))
                                else:
                                    address = Int64(out.toint()) + (Int64(row) * Int64(self.n) + col) * Int64(4)
                                    st_global_v2_f32(address, total0, total1)


class CandidateFp8Projection(Fp8Projection):
    """Candidate ``Fp8Projection`` with distinct ``key()`` and bit-identical output to ``Fp8Projection``."""

    def __init__(self, n: int, k: int, *, prefill_rows: int | None = None, row_scales: bool = False,
                 kmajor: bool = False, gemv_rows: int = 16, wide_rows: int = 0, warps: int | None = None,
                 groups: int = 4, prefill_mask: int = 0xFF, out_dtype=cutlass.BFloat16,
                 wide_warps: int | None = None, wide_groups: int | None = None,
                 decode_rows: int | None = None, cand_groups: int | None = None,
                 cand_cta_tiles: int | None = None):
        self.n, self.k = int(n), int(k)
        self.row_scales = bool(row_scales)
        self.kmajor = bool(kmajor) or (self.row_scales and prefill_rows is not None)
        self.prefill_mask = int(prefill_mask)
        self.prefill = prefill_rows is not None
        self.gemv = self.gemv_wide = self.w8a8 = None
        if self.prefill:
            from ._glmf_fp8 import BlockFp8Projection

            if out_dtype != cutlass.BFloat16 and self.prefill_mask:
                raise ValueError("non-BF16 prefill output requires W8A16 (prefill_mask=0)")
            self.w8a8 = BlockFp8Projection(self.n, self.k, int(prefill_rows), row_scales=self.row_scales) \
                if self.prefill_mask else None
            self.w8a16 = CandidateTmaFp8Gemm(
                self.n, self.k, scales="row_kmajor" if self.row_scales else "block",
                compute_warps=8, num_stages=3, out_dtype=out_dtype)
            self.max_gemv_rows = 0
        else:
            warps = gemv_warps(self.k) if warps is None else int(warps)
            base_groups = int(groups)
            if (warps if wide_warps is None else int(wide_warps),
                    base_groups if wide_groups is None else int(wide_groups)) != (warps, base_groups):
                raise ValueError("the wide GEMV tile must keep the thin tile's warps and groups: "
                                 "one K order covers the decode range")
            chosen_groups = (candidate_gemv_groups(self.n, self.k, warps=warps, groups=base_groups,
                                                   row_scales=self.row_scales)
                             if cand_groups is None else int(cand_groups))
            self.gemv = CandidateMmaFp8Gemv(
                self.n, self.k, max_rows=int(gemv_rows), warps=warps, groups=chosen_groups,
                row_scales=self.row_scales, kmajor=self.kmajor, out_dtype=out_dtype,
                cta_tiles=cand_cta_tiles)
            cap = max(FP8_DECODE_ROWS if decode_rows is None else int(decode_rows), int(wide_rows),
                      int(gemv_rows))
            self.gemv_wide = CandidateMmaFp8Gemv(
                self.n, self.k, max_rows=cap, warps=warps, groups=chosen_groups,
                row_scales=self.row_scales, kmajor=self.kmajor, out_dtype=out_dtype,
                cta_tiles=cand_cta_tiles) if cap > int(gemv_rows) else None
            self.gemv_rows = int(gemv_rows)
            self.max_gemv_rows = cap if self.gemv_wide is not None else int(gemv_rows)
            tile_n = decode_tile_n(self.n)
            self.w8a16 = CandidateTmaFp8Gemm(
                self.n, self.k, scales=("row_kmajor" if self.kmajor else "row")
                if self.row_scales else "block", tile_n=tile_n,
                num_stages=4 if tile_n == 128 else 6, out_dtype=out_dtype)

    def key(self) -> tuple:
        return ("candidate_fp8_proj", super().key())

