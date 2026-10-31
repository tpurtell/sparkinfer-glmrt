"""Per-head position-dependent query scaling (scalable softmax, SSMax) with
optional interleaved RoPE, in the reference's BF16 rounding order.

``q [rows, H, D]`` BF16 (``D = nope + rope``, RoPE dims last): when ``rope``,
the rope part ``q[..., nope:]`` is rotated as interleaved pairs
``(x[2i], x[2i+1])`` by frequency ``i`` and written de-interleaved
(``[even' | odd']``); then every one of the ``D`` dims is multiplied by the
per-(row, head) scale ``s_h * log(pos + 1) + b_h``. Every step rounds where
the reference (``modeling_mistral_large4``) materializes a BF16 tensor:

* cos/sin come from the FP32 ``cos_sin`` table rounded to BF16 (the
  reference's ``cos.to(x.dtype)``);
* RoPE: ``bf16(bf16(x0 * c) - bf16(x1 * s))`` and ``bf16(bf16(x1 * c) + bf16(x0 * s))``;
* scale: ``log(pos + 1)`` in FP32 then BF16; ``bf16(bf16(s * log) + b)``;
* output ``bf16(q * scale)``.

Outputs: the scaled ``q`` in place (the nope part feeds the latent
absorption) and, when ``out_rope`` is given, the scaled rotated rope part at
``out[row, h, out_offset : out_offset + rope]`` (the absorbed query's RoPE
tail, row stride ``out_row`` elements). ``rope`` 0 (NoPE layers) scales all
``D`` dims and writes no tail.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ._glm_kernels import _bf16

__all__ = ["QueryScale"]


class QueryScale:
    """One CTA per row; thread ``t`` owns head ``t // lanes`` and dims
    ``lane, lane + lanes, ...`` of it (``lanes`` = 32)."""

    lanes = 32

    def __init__(self, *, heads: int, head_dim: int, rope: int, out_row: int = 0, out_offset: int = 0,
                 token_row: int | None = None):
        self.heads, self.head, self.rope = int(heads), int(head_dim), int(rope)
        if self.rope % 2 or not 0 <= self.rope <= self.head or self.head % self.lanes:
            raise ValueError("query scale needs an even rope width inside a 32-multiple head")
        if self.rope and self.rope != 2 * self.lanes:
            raise ValueError("interleaved RoPE runs one frequency per lane (64 rope dims)")
        self.nope = self.head - self.rope
        self.out_row, self.out_offset = int(out_row), int(out_offset)
        if self.rope and (self.out_row < self.out_offset + self.rope or self.out_offset < 0):
            raise ValueError("the rope tail must fit its output row")
        # Elements between output tokens: heads x out_row, or a wider combined query
        # (a head-split share writing its heads into both GPUs' query rows).
        self.token_row = None if token_row is None else int(token_row)
        if self.token_row is not None and self.token_row < self.heads * max(self.out_row, 1):
            raise ValueError("the token row must hold every head's output row")
        # Heads per CTA pass: 8 warps of one head each.
        self.warps = 8

    def key(self) -> tuple:
        key = ("query_scale", self.heads, self.head, self.rope, self.out_row, self.out_offset)
        return key if self.token_row is None else key + (self.token_row,)

    @cute.jit
    def __call__(self, q: cute.Pointer, positions: cute.Pointer, cos_sin: cute.Pointer,
                 s: cute.Pointer, b: cute.Pointer, query: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        h, d = self.heads, self.head
        token = h * max(self.out_row, 1) if cutlass.const_expr(self.token_row is None) else self.token_row
        out_layout = cute.make_layout((m, h, max(self.out_row, 1)), stride=(token, max(self.out_row, 1), 1))
        self.kernel(
            cute.make_tensor(q, cute.make_layout((m, h, d), stride=(h * d, d, 1))),
            cute.make_tensor(positions, cute.make_layout((m,))),
            cos_sin,
            cute.make_tensor(s, cute.make_layout((h,))),
            cute.make_tensor(b, cute.make_layout((h,))),
            cute.make_tensor(query, out_layout),
        ).launch(grid=(rows, 1, 1), block=(self.warps * self.lanes, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, positions: cute.Tensor, cos_sin: cute.Pointer,
               s: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(self.lanes)
        warp = tidx // Int32(self.lanes)
        position = Int64(positions[token])
        log_pos = _bf16(cute.math.log(Float32(position + Int64(1)), fastmath=False))
        cos_v = Float32(0.0)
        sin_v = Float32(0.0)
        if cutlass.const_expr(self.rope > 0):
            cs = cute.make_ptr(Float32, Int64(cos_sin.toint()) + position * Int64(self.rope * 4),
                               cute.AddressSpace.gmem, assumed_align=4)
            cos_v = _bf16(Float32(cs[lane]))
            sin_v = _bf16(Float32(cs[Int32(self.lanes) + lane]))
        for pass_ in cutlass.range_constexpr((self.heads + self.warps - 1) // self.warps):
            head = Int64(warp) + Int64(pass_ * self.warps)
            if head < Int64(self.heads):
                scale = _bf16(_bf16(_bf16(Float32(s[head])) * log_pos) + _bf16(Float32(b[head])))
                for j in cutlass.range_constexpr(self.nope // self.lanes):
                    dim = Int64(j * self.lanes) + Int64(lane)
                    q[token, head, dim] = (_bf16(Float32(q[token, head, dim]) * scale)).to(BFloat16)
                if cutlass.const_expr(self.rope > 0):
                    base = Int64(self.nope) + Int64(2) * Int64(lane)
                    x0 = Float32(q[token, head, base])
                    x1 = Float32(q[token, head, base + Int64(1)])
                    even = _bf16(_bf16(x0 * cos_v) - _bf16(x1 * sin_v))
                    odd = _bf16(_bf16(x1 * cos_v) + _bf16(x0 * sin_v))
                    even = _bf16(even * scale)
                    odd = _bf16(odd * scale)
                    # All reads of this head's pairs precede the in-place writes
                    # within the warp (each lane owns its own pair).
                    cute.arch.sync_warp()
                    q[token, head, Int64(self.nope) + Int64(lane)] = even.to(BFloat16)
                    q[token, head, Int64(self.nope + self.lanes) + Int64(lane)] = odd.to(BFloat16)
                    if cutlass.const_expr(self.out_row > 0):
                        out[token, head, Int64(self.out_offset) + Int64(lane)] = even.to(BFloat16)
                        out[token, head, Int64(self.out_offset + self.lanes) + Int64(lane)] = odd.to(BFloat16)
