"""TMA tiles of the Qwen 3.8 Flash Next BF16 attention projections by live rows.

The decode programs of the qwen4 GDN and full-attention layers run one serial
BF16 projection after the other at 1-16 live rows, where ``prefill_config``'s
64 x 128 tile leaves most of the GPU's SMs with no CTA (the o_proj's 2560-wide
output is 20 CTAs on 188 SMs). ``glm_projection(tiles_by_rows=...)`` entries
retile that same MMA accumulation, so every entry gives ``prefill_config``'s
bits (``RowTiledBf16Projection``; checked by
``tests/integration/test_cuteafd_qwen4_decode_tiles.py``).

RTX PRO 6000 (188 SMs), L2-cold medians of one launch, ``us`` at 1 / 4 / 16 /
32 / 64 live rows, ``prefill_config`` against the entry that serves them
(``benchmarks/bench_cuteafd_qwen4_decode_tiles.py``):

| weight shape [N, K] | prefill_config    | entry                              |
|---------------------|-------------------|------------------------------------|
| ``[16480, 2560]``   | 59 61 59 59 61    | 57 57 57 59 <= 32 rows, 59 <= 64   |
| ``[13952, 2560]``   | 51 53 51 51 51    | 49 51 51 53 <= 32 rows             |
| ``[2560, 6144]``    | 45 45 45 45 45    | 27 27 27 27 <= 32 rows, 29 <= 128  |

The ``[2560, 6144]`` shape is every o_proj of a cycle (36 GDN + 12 full
attention: the GDN value width and the q/gate width are both 6144), 48 of the
96 projection launches, and the starved one that pays (45 -> 27 us). Rows past
the last entry's limit keep ``prefill_config`` (the prefill arms' shapes
already saturate DRAM there, so no entry is defined past them). Every entry
splits along N (80 CTAs and up), so no SM count is baked in: a card with fewer
SMs packs more CTAs into each SM rather than losing them. The tables are
decode-only: ``compile_qwen4_gdn_aot``/``compile_qwen4_attn_producer_aot``/
``compile_qwen4_attn_o_aot`` take ``decode_tiles=True``.

``qwen4_projection`` is the BF16 projection behind ``_fp8_weights``' call
signature: without a table it is that module's audited ``projection`` unchanged,
with one it wraps ``glm_projection(tiles_by_rows=...)`` here, so the knob stays
out of the audited wrapper and only the qwen4 modules carry it.
"""

from __future__ import annotations

#: ``tiles_by_rows`` entries ``(rows, compute_warps, tile_n, num_stages)`` by weight shape ``[N, K]``.
QWEN4_DECODE_TILES = {
    (16480, 2560): ((32, 1, 32, 4), (64, 2, 32, 4)),
    (13952, 2560): ((32, 1, 32, 12),),
    (2560, 6144): ((32, 1, 32, 6), (128, 2, 32, 4)),
}


def qwen4_decode_tiles(shape: tuple[int, int]) -> tuple:
    """The decode row tiles of a ``[N, K]`` BF16 weight shape (none without a table)."""
    return QWEN4_DECODE_TILES.get(tuple(int(v) for v in shape), ())


import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32

from ._fp8_weights import projection


class _TiledBf16Projection:
    """``glm_projection(tiles_by_rows=...)`` behind the FP8 call signature (``_fp8_weights``)."""

    def __init__(self, n: int, k: int, tiles_by_rows: tuple):
        from ._glm_kernels import glm_projection

        self.tiles_by_rows = tuple(tuple(int(v) for v in entry) for entry in tiles_by_rows)
        self.inner = glm_projection(n, k, tiles_by_rows=self.tiles_by_rows)

    def key(self) -> tuple:
        return self.inner.key()

    @cute.jit
    def __call__(self, x: cute.Pointer, w_bf16: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.inner(x, w_bf16, out, rows, stream)


def qwen4_projection(n: int, k: int, fp8: bool, tiles_by_rows: tuple = ()):
    """``projection(n, k, fp8)``, with ``tiles_by_rows`` on the BF16 route (no table -> as shipped)."""
    if not tiles_by_rows:
        return projection(n, k, fp8)
    if fp8:
        raise ValueError("tiles_by_rows is the BF16 projection's (fp8=False)")
    return _TiledBf16Projection(n, k, tiles_by_rows)
