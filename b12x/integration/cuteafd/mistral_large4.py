"""Mistral Large 4 (``mistral_large4``) programs beside the GLM 5.x ones.

Mistral Large 4 is GLM 5.3's MLA + DSA layer without q-LoRA, plus scalable
softmax, NoPE every fourth layer and cross-layer latent sharing. The shared
GLM programs run it unchanged where the contracts agree:

* RoPE layers: the 656-byte latent record (``glm_stage_rank_pack`` with no q
  part), GLM 5.3 sparse MLA at ``softmax_scale = 128**-0.5``;
* NoPE layers: the 528-byte record and GLM 5.3 Flash's 512-wide sparse MLA;
* the DSA index epilogue (``GlmIndexPost``, NoPE layers with ``rope=0``) and
  the deterministic index top-k (lower key id on ties).

New here, general per-head operations:

``compile_query_scale_aot(geometry, rope=..., max_rows=R)`` (``QueryScale``):
the attention query after ``wq`` (``[rows, H, nope+rope]``), RoPE then the
per-head scalable-softmax factor ``s_h log(pos + 1) + b_h`` in place, the
scaled rotated tail also written into the absorbed query::

    q          bf16 [rows,H,D]     inout  wq(x) viewed per head; scaled in place
    positions  i64  [rows]         in
    cos_sin    f32  [P,64]         in     cos(32) | sin(32) of the LeRoPE frequencies
    s          bf16 [H]            in     ssmax_scaling.s
    b          bf16 [H]            in     ssmax_scaling.b
    query      bf16 [rows,H,576]   out    [:, :, 512:576] = scaled rope(q_rot) (RoPE layers)
    rows       int32

``compile_query_scale_aot(geometry, rope=0, heads=I, head_dim=128)`` is the
indexer query's SSMax (before ``GlmIndexPost`` applies its RoPE, the
reference's order); ``query`` is unused.

``compile_index_post_aot(geometry, rope=..., max_rows=R)``: the GLM DSA index
epilogue (``glm_stage_index_post`` ABI) with the query/key RoPE skipped on
NoPE layers; ``iq`` is the SSMax-scaled ``wq_b`` output.

``compile_rank_pack_aot(geometry, rope=..., max_rows=R)``: ``kv_a_norm`` and
the FP8 latent record of ``wkv_a(x)`` (no q-LoRA): RoPE layers write the
656-byte record (rotated key in its BF16 tail), NoPE layers the 528-byte one::

    kv         bf16 [rows,576|512]  in     wkv_a_with_mqa output
    kv_a_norm  bf16 [512]           in
    positions  i64  [rows]          in
    slots      i64  [rows]          in     page*64+row (<0 skips)
    cos_sin    f32  [P,64]          in     (RoPE layers)
    cache      u8   [pages,64*R]    inout
    rows       int32

``compile_kv_b_aot(geometry, stage="absorb"|"values", rope=..., max_rows=R)``:
the per-head BF16 ``wkv_b`` GEMMs (``wkv_b`` is BF16 in the checkpoint)::

    absorb:  a bf16 [rows,H,128] (q_nope leading)  w bf16 [H,512,nope]  out bf16 [rows,H,latent]
    values:  a bf16 [rows,H,512]                   w bf16 [H,128,512]   out bf16 [rows,H,128]

``latent`` 576 (RoPE layers; the tail is the query scale's) or 512 (NoPE).

``compile_layer_norm_aot(width=W, max_rows=R)``: LayerNorm with bias (the
indexer's ``q_input_norm``), F32 statistics, BF16 out::

    x          bf16 [rows,W]   in
    weight     bf16 [W]        in
    bias       bf16 [W]        in
    out        bf16 [rows,W]   out
    rows       int32
"""
from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ._common import Operand, Scalar, compile_program
from ._glm_kernels import (BatchedBf16Gemm, GlmIndexPost, GlmRankNormPackKV, _block_sum, _bf16, _reduction_storage,
                           _rsqrt)
from ._query_scale import QueryScale

__all__ = ["ML4", "Ml4Geometry", "LayerNormBias", "compile_index_post_aot", "compile_layer_norm_aot",
           "compile_index_q_fp8_aot", "compile_kv_b_aot", "compile_query_scale_aot", "compile_rank_pack_aot",
           "compile_sparse_mla_aot", "nope_attention", "rope_attention"]


@dataclass(frozen=True)
class Ml4Geometry:
    """Static Mistral Large 4 geometry baked into the ``ml4_*`` programs."""

    name: str = "mistral_large4"
    hidden: int = 6144
    heads: int = 64
    kv_lora_rank: int = 512
    qk_nope_dim: int = 64
    qk_rope_dim: int = 64
    v_head_dim: int = 128
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    dense_inter: int = 24576
    moe_inter: int = 6144
    routed_experts: int = 128
    top_k: int = 4
    norm_eps: float = 1.0e-5
    index_norm_eps: float = 1.0e-6
    page_rows: int = 64

    def __post_init__(self) -> None:
        if (self.kv_lora_rank, self.qk_rope_dim, self.index_head_dim) != (512, 64, 128):
            raise ValueError("Mistral Large 4 programs require the GLM 512 + 64 record and 128-dim index keys")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    def nope_dim(self, rope: bool) -> int:
        """NoPE layers spend the rope width on nope (the query stays 128 wide)."""
        return self.qk_nope_dim if rope else self.qk_head_dim


ML4 = Ml4Geometry()


def rope_attention(g: Ml4Geometry = ML4):
    """The GLM 5.x program geometry of a RoPE layer: no q-LoRA, nope 64 + rope 64,
    v 128, the 656-byte record, ``softmax_scale = 128**-0.5``."""
    from ._common import GLMGeometry
    return GLMGeometry(name="ml4", hidden=g.hidden, heads=g.heads, q_lora_rank=0, kv_lora_rank=g.kv_lora_rank,
                       qk_nope_dim=g.qk_nope_dim, qk_rope_dim=g.qk_rope_dim, v_head_dim=g.v_head_dim,
                       index_heads=g.index_heads, index_head_dim=g.index_head_dim, index_topk=g.index_topk,
                       dense_inter=g.dense_inter, moe_inter=g.moe_inter, routed_experts=g.routed_experts,
                       norm_eps=g.norm_eps, index_norm_eps=g.index_norm_eps, page_rows=g.page_rows)


def nope_attention(g: Ml4Geometry = ML4):
    """The GLM 5.3 Flash (NoPE MLA) program geometry of a NoPE layer: nope 128,
    v 128, the 528-byte record and 512-wide absorbed query, plain top-k
    (one-token index pools), ``softmax_scale = 128**-0.5``."""
    from ._common import GLMFGeometry
    return GLMFGeometry(name="ml4_nope", hidden=g.hidden, heads=g.heads, q_lora_rank=0,
                        kv_lora_rank=g.kv_lora_rank, qk_nope_dim=g.qk_head_dim, v_head_dim=g.v_head_dim,
                        index_heads=g.index_heads, index_head_dim=g.index_head_dim, index_topk=g.index_topk,
                        index_kpool=1, dense_inter=g.dense_inter, moe_inter=g.moe_inter,
                        routed_experts=g.routed_experts, top_k=g.top_k, swiglu_limit=0.0,
                        norm_eps=g.norm_eps, index_norm_eps=g.index_norm_eps, page_rows=g.page_rows)


def compile_sparse_mla_aot(g: Ml4Geometry = ML4, *, rope: bool, route: str, max_rows: int, **kwargs):
    """GLM 5.x sparse MLA at a Mistral Large 4 layer's record contract. The
    program name carries the family so its compile key never aliases a GLM
    program of another softmax scale."""
    from .glm_sparse_mla import compile_glm_sparse_mla_aot
    attention = rope_attention(g) if rope else nope_attention(g)
    assert abs(attention.softmax_scale - g.softmax_scale) < 1e-15
    return compile_glm_sparse_mla_aot(attention, route=route, max_rows=max_rows,
                                      name="ml4_sparse_mla" + ("" if rope else "_nope"), **kwargs)


def compile_query_scale_aot(g: Ml4Geometry = ML4, *, rope: bool, max_rows: int,
                            heads: int | None = None, head_dim: int | None = None):
    """Per-head SSMax with optional interleaved RoPE (see the module docstring)."""
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    heads = g.heads if heads is None else int(heads)
    head_dim = g.qk_head_dim if head_dim is None else int(head_dim)
    rope_dims = g.qk_rope_dim if rope else 0
    latent = g.kv_lora_rank + g.qk_rope_dim
    launch = QueryScale(heads=heads, head_dim=head_dim, rope=rope_dims,
                        out_row=latent if rope else 0, out_offset=g.kv_lora_rank if rope else 0)
    operands = (
        Operand("q", torch.bfloat16, f"[rows,{heads},{head_dim}]", "inout"),
        Operand("positions", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("s", torch.bfloat16, f"[{heads}]"),
        Operand("b", torch.bfloat16, f"[{heads}]"),
        Operand("query", torch.bfloat16, f"[rows,{heads},{latent}]" if rope else "unused", "out"),
    )
    return compile_program(
        launch, name="ml4_query_scale" + ("_rope" if rope else ""), operands=operands,
        scalars=(Scalar("rows"),), key=(launch.key(), int(max_rows)),
        geometry=dict(heads=heads, head_dim=head_dim, rope=rope_dims, max_rows=int(max_rows),
                      out_offset=launch.out_offset), doc=__doc__)


class LayerNormBias:
    """``torch.nn.functional.layer_norm`` with weight and bias over BF16 rows:
    F32 mean and biased variance, ``bf16((x - mean) * rstd * w + b)``."""

    threads = 256

    def __init__(self, *, width: int, eps: float):
        self.width, self.eps = int(width), float(eps)
        if self.width % self.threads:
            raise ValueError("LayerNorm width must divide by the CTA width")
        self.per = self.width // self.threads
        self.warps = self.threads // 32

    def key(self) -> tuple:
        return ("layer_norm_bias", self.width, self.eps)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, bias: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        m = Int64(rows)
        w = self.width
        self.kernel(cute.make_tensor(x, cute.make_layout((m, w), stride=(w, 1))),
                    cute.make_tensor(weight, cute.make_layout((w,))),
                    cute.make_tensor(bias, cute.make_layout((w,))),
                    cute.make_tensor(out, cute.make_layout((m, w), stride=(w, 1)))
                    ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, weight: cute.Tensor, bias: cute.Tensor, out: cute.Tensor):
        token = Int64(cute.arch.block_idx()[0])
        tidx = Int32(cute.arch.thread_idx()[0])
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_reduction_storage(2, self.warps))
        sums = storage.sums.get_tensor(cute.make_layout((2, self.warps), stride=(self.warps, 1)))
        values = cute.make_rmem_tensor(cute.make_layout((self.per,), stride=(1,)), Float32)
        total = Float32(0.0)
        for j in cutlass.range_constexpr(self.per):
            values[j] = Float32(x[token, Int64(j * self.threads) + Int64(tidx)])
            total = total + values[j]
        mean = _block_sum(total, sums, Int32(0), self.warps) / Float32(self.width)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(self.per):
            diff = values[j] - mean
            square = square + diff * diff
        rstd = _rsqrt(_block_sum(square, sums, Int32(1), self.warps) / Float32(self.width) + Float32(self.eps))
        for j in cutlass.range_constexpr(self.per):
            d = Int64(j * self.threads) + Int64(tidx)
            out[token, d] = _bf16((values[j] - mean) * rstd * Float32(weight[d]) + Float32(bias[d])).to(BFloat16)


def compile_layer_norm_aot(*, width: int, max_rows: int, eps: float = 1.0e-6):
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    launch = LayerNormBias(width=width, eps=eps)
    return compile_program(
        launch, name="ml4_layer_norm", scalars=(Scalar("rows"),), key=(launch.key(), int(max_rows)),
        operands=(Operand("x", torch.bfloat16, f"[rows,{width}]"),
                  Operand("weight", torch.bfloat16, f"[{width}]"),
                  Operand("bias", torch.bfloat16, f"[{width}]"),
                  Operand("out", torch.bfloat16, f"[rows,{width}]", "out")),
        geometry=dict(width=int(width), eps=float(eps), max_rows=int(max_rows)), doc=__doc__)


def compile_index_post_aot(g: Ml4Geometry = ML4, *, rope: bool, max_rows: int):
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    i = g.index_heads
    launch = GlmIndexPost(heads=i, eps=g.index_norm_eps, weight_scale=float(i) ** -0.5 * 128.0 ** -0.5,
                          page_rows=g.page_rows, rope=rope)
    operands = (Operand("iq", torch.bfloat16, f"[rows,{i},128]"),
                Operand("kw", torch.bfloat16, f"[rows,{128 + i}]"),
                Operand("positions", torch.int64, "[rows]", align=8),
                Operand("slots", torch.int64, "[rows]", align=8),
                Operand("cos_sin", torch.float32, "[P,64]", align=4),
                Operand("k_weight", torch.bfloat16, "[128]"),
                Operand("k_bias", torch.bfloat16, "[128]"),
                Operand("q_fp8", torch.float8_e4m3fn, f"[rows,{i},128]", "out"),
                Operand("head_weights", torch.float32, f"[rows,{i}]", "out"),
                Operand("cache", torch.uint8, f"[pages,{g.page_rows * (128 + 4)}]", "inout"))
    return compile_program(
        launch, name="ml4_index_post" + ("" if rope else "_nope"), operands=operands, scalars=(Scalar("rows"),),
        key=(g, bool(rope), int(max_rows)),
        geometry=dict(index_heads=i, rope=bool(rope), page_rows=g.page_rows, index_norm_eps=g.index_norm_eps,
                      max_rows=int(max_rows)), doc=__doc__)


class _RankPack:
    """``GlmRankNormPackKV`` without the q part (no q-LoRA, no q_resid)."""

    def __init__(self, rope: bool, eps: float, page_rows: int):
        self.pack = GlmRankNormPackKV(q_rank=0, eps=eps, page_rows=page_rows,
                                      record_bytes=656 if rope else 528, rope=64 if rope else 0)

    def key(self) -> tuple:
        return ("rank_pack", self.pack.rope, self.pack.eps, self.pack.page_rows)

    @cute.jit
    def __call__(self, kv: cute.Pointer, kv_a_norm: cute.Pointer, positions: cute.Pointer, slots: cute.Pointer,
                 cos_sin: cute.Pointer, cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        # q_norm / q_resid are never read or written at q_rank 0: bind live pointers.
        self.pack(kv, kv_a_norm, kv_a_norm, positions, slots, cos_sin, kv, cache, rows, stream)


def compile_rank_pack_aot(g: Ml4Geometry = ML4, *, rope: bool, max_rows: int):
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    launch = _RankPack(rope, g.norm_eps, g.page_rows)
    width = g.kv_lora_rank + (g.qk_rope_dim if rope else 0)
    record = 656 if rope else 528
    operands = (Operand("kv", torch.bfloat16, f"[rows,{width}]"),
                Operand("kv_a_norm", torch.bfloat16, f"[{g.kv_lora_rank}]"),
                Operand("positions", torch.int64, "[rows]", align=8),
                Operand("slots", torch.int64, "[rows]", align=8),
                Operand("cos_sin", torch.float32, "[P,64]" if rope else "unused", align=4),
                Operand("cache", torch.uint8, f"[pages,{g.page_rows * record}]", "inout"))
    return compile_program(
        launch, name="ml4_rank_pack" + ("" if rope else "_nope"), operands=operands, scalars=(Scalar("rows"),),
        key=(launch.key(), int(max_rows)),
        geometry=dict(rope=bool(rope), record_bytes=record, page_rows=g.page_rows, norm_eps=g.norm_eps,
                      max_rows=int(max_rows)), doc=__doc__)


def compile_kv_b_aot(g: Ml4Geometry = ML4, *, stage: str, rope: bool, max_rows: int):
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    n, c, head, nope = g.heads, g.kv_lora_rank, g.qk_head_dim, g.nope_dim(rope)
    latent = c + (g.qk_rope_dim if rope else 0)
    if stage == "absorb":
        launch = BatchedBf16Gemm(n=c, k=nope, batch=n, a_row=n * head, a_batch=head, o_row=n * latent, o_batch=latent)
        operands = (Operand("a", torch.bfloat16, f"[rows,{n},{head}]"),
                    Operand("w", torch.bfloat16, f"[{n},{c},{nope}]", note="kv_b k rows per head, transposed"),
                    Operand("out", torch.bfloat16, f"[rows,{n},{latent}]", "out"))
    elif stage == "values":
        v = g.v_head_dim
        launch = BatchedBf16Gemm(n=v, k=c, batch=n, a_row=n * c, a_batch=c, o_row=n * v, o_batch=v)
        operands = (Operand("a", torch.bfloat16, f"[rows,{n},{c}]"),
                    Operand("w", torch.bfloat16, f"[{n},{v},{c}]", note="kv_b v rows per head"),
                    Operand("out", torch.bfloat16, f"[rows,{n},{v}]", "out"))
    else:
        raise ValueError("kv_b stage must be absorb or values")
    return compile_program(
        launch, name=f"ml4_{stage}" + ("" if rope or stage == "values" else "_nope"), operands=operands,
        scalars=(Scalar("rows"),), key=(launch.key(), int(max_rows)),
        geometry=dict(stage=stage, rope=bool(rope), heads=n, nope=nope, latent=latent, max_rows=int(max_rows)),
        doc=__doc__)


class _IndexQueryFp8:
    """``out = LayerNorm(h) @ dequant(wq_b)^T`` over the checkpoint's own FP8 block-128
    ``wq_b`` (E4M3 ``[I*128, hidden]`` and its 128x128 FP32 scale grid): GLM's FP8-only
    projection (``Fp8Projection``) at the direct indexer query's K = hidden. Decode runs
    W8A16 at the decode GEMV capacity; prefill takes the W8A8/W8A16 switch scalar."""

    def __init__(self, g: Ml4Geometry, prefill_rows: int | None):
        from ._fp8_weights import Fp8Projection
        self.g, self.prefill = g, prefill_rows is not None
        self.proj = Fp8Projection(g.index_heads * g.index_head_dim, g.hidden, prefill_rows=prefill_rows)

    def key(self) -> tuple:
        return ("ml4_index_q_fp8", self.proj.key())

    def scratch_bytes(self, rows: int) -> int:
        return self.proj.prefill_scratch_bytes(rows)

    @cute.jit
    def body(self, x: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
             scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.proj(x, w_fp8, scale, out, rows, fp8_rows, Int64(scratch.toint()), stream)


class _IndexQueryFp8Decode(_IndexQueryFp8):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        from ._fp8_weights import FP8_DECODE_ROWS
        self.body(x, w_fp8, scale, out, scratch, rows, Int32(FP8_DECODE_ROWS), stream)


class _IndexQueryFp8Prefill(_IndexQueryFp8):
    @cute.jit
    def __call__(self, x: cute.Pointer, w_fp8: cute.Pointer, scale: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, fp8_rows: Int32, stream: cuda.CUstream):
        self.body(x, w_fp8, scale, out, scratch, rows, fp8_rows, stream)


def compile_index_q_fp8_aot(g: Ml4Geometry = ML4, *, mode: str, max_rows: int):
    """The indexer query projection over a checkpoint that keeps ``indexer.wq_b`` in its
    shipped FP8 block-128 form (``ml4_index_q``): ``x`` is the indexer's LayerNorm output,
    ``out`` BF16 ``[rows, I*128]`` (then the index query scale). ``mode`` ``decode`` or
    ``prefill`` as GLM's FP8-only programs."""
    from ._fp8_weights import check_w8_mode, fp8_only_operands, w8_scalars
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    prefill = check_w8_mode(mode) == "prefill"
    launch = (_IndexQueryFp8Prefill if prefill else _IndexQueryFp8Decode)(g, int(max_rows) if prefill else None)
    n, k = g.index_heads * g.index_head_dim, g.hidden
    operands = (
        Operand("x", torch.bfloat16, f"[rows,{k}]"),
        *fp8_only_operands("w_iq", n, k, prefill=prefill),
        Operand("out", torch.bfloat16, f"[rows,{n}]", "out"),
        Operand("scratch", torch.uint8, "[index_q_scratch_bytes]", "scratch"),
    )
    return compile_program(
        launch, name="ml4_index_q", operands=operands, scalars=w8_scalars(prefill),
        key=(int(max_rows), mode, launch.key()), scratch={"scratch": launch.scratch_bytes},
        geometry=dict(outputs=n, width=k, max_rows=int(max_rows), fp8_weights="only", mode=mode), doc=__doc__)
