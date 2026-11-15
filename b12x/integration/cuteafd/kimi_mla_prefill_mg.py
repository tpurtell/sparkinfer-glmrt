"""Kimi MLA MG prefill over native 656-byte FP8 records.

GLM_NSA has exactly Kimi's absorbed query and cache layout. Reuse its traits,
not its model softmax scale. BF16 attention (BF16 Q, exact K with its FP32
group scale applied to the FP32 partial, BF16 V and P, FP32 accumulate) is the
mode that meets Kimi's oracle bounds against the exact decoded record; FP8 QK
does not and stays a diagnostic arm. Kimi's split decode kernels still round
each dequantized key to BF16: a known decode-side difference. Dense causal extend builds one shared physical-slot
vector plus per-query visible lengths; sparse takes the selector's native slots.
No split partials or cache conversion. The public Kimi routes are unchanged.
"""
from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
import torch

from ._common import Operand, Scalar, compile_program


class _DenseMetadata:
    def __init__(self, capacity):
        self.capacity = capacity

    @cute.jit
    def __call__(self, table: cute.Pointer, length: cute.Pointer, slots: cute.Pointer,
                 visible: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(table, length, slots, visible, rows).launch(
            grid=(-(-self.capacity // 256), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, table: cute.Pointer, length: cute.Pointer, slots: cute.Pointer,
               visible: cute.Pointer, rows: Int32):
        i = Int64(cute.arch.block_idx()[0]) * Int64(256) + Int64(cute.arch.thread_idx()[0])
        if i < Int64(self.capacity):
            slot = Int32(-1)
            if i < Int64(length[0]):
                page = Int64(table[i // Int64(64)])
                if page >= Int64(0):
                    slot = Int32(page * Int64(64) + i % Int64(64))
            slots[i] = slot
        if i < Int64(rows):
            visible[i] = length[0] - rows + Int32(i) + Int32(1)


class _MG:
    def __init__(self, g, capacity, *, sparse, bf16_compute):
        from b12x.attention._shared.mla.prefill import _mg_head_partitions
        from b12x.attention._shared.mla.prefill_mg import UnifiedPrefillMGKernel
        from b12x.attention._shared.mla.smem_mg import make_smem_layout_mg, max_mg_head_groups
        from b12x.attention._shared.mla.traits import (
            ComputeMode, ModelType, ScaleFormat, make_unified_traits, with_bf16_attention,
        )

        self.g, self.capacity, self.sparse = g, capacity, sparse
        traits = make_unified_traits(ModelType.GLM_NSA, ComputeMode.FP8,
                                     ScaleFormat.ARBITRARY_FP32)
        if bf16_compute:
            traits = with_bf16_attention(traits)
        self.metadata = _DenseMetadata(capacity)
        self.kernels = []
        for groups, active, offset in _mg_head_partitions(
                g.heads, traits.hpb, max_mg_head_groups(traits)):
            layout = make_smem_layout_mg(traits, groups)
            self.kernels.append(UnifiedPrefillMGKernel(
                traits, layout, 1 if sparse else 64, -(-capacity // 64),
                replicate_h=active // layout.heads_per_cta, num_heads=g.heads,
                q_stride=(g.heads * 576, 576, 1),
                indices_stride0=capacity if sparse else 0,
                output_stride=(g.heads * 512, 512, 1),
                out_lse_stride=(g.heads, 1), has_sink=False, topk=capacity,
                head_offset=offset, valid_hpb=traits.hpb,
                zero_masked_v=not bf16_compute,
            ))

    @cute.jit
    def __call__(self, query: cute.Pointer, cache: cute.Pointer, table: cute.Pointer,
                 lengths: cute.Pointer, out: cute.Pointer, lse: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        n = self.g.heads
        m = Int64(rows)
        index_ptr, length_ptr = table, lengths
        if cutlass.const_expr(not self.sparse):
            index_ptr = cute.make_ptr(Int32, Int64(scratch.toint()), cute.AddressSpace.gmem,
                                      assumed_align=16)
            length_ptr = cute.make_ptr(Int32, Int64(scratch.toint()) + Int64(self.capacity * 4),
                                       cute.AddressSpace.gmem, assumed_align=4)
            self.metadata(table, lengths, index_ptr, length_ptr, rows, stream)
        q = cute.make_tensor(query, cute.make_layout((m, n, 576), stride=(n * 576, 576, 1)))
        kv = cute.make_tensor(cache, cute.make_layout((1,)))
        idx = cute.make_tensor(index_ptr, cute.make_layout((m, self.capacity),
            stride=(self.capacity if self.sparse else 0, 1)))
        lens = cute.make_tensor(length_ptr, cute.make_layout((m,)))
        sink = cute.make_tensor(cute.make_ptr(Float32, Int64(length_ptr.toint()),
            cute.AddressSpace.gmem, assumed_align=4), cute.make_layout((1,)))
        o = cute.make_tensor(out, cute.make_layout((m, n, 512), stride=(n * 512, 512, 1)))
        l = cute.make_tensor(lse, cute.make_layout((m, n), stride=(n, 1)))
        for k in cutlass.range_constexpr(len(self.kernels)):
            self.kernels[k](q, kv, idx, lens, sink, o, l,
                Float32(self.g.softmax_scale * math.log2(math.e)), Float32(1),
                Int64(656 if self.sparse else 64 * 656), rows, stream)


def compile_kimi_mg_prefill_aot(g, *, max_rows=4096, max_context=131072,
                                 sparse=False, bf16_compute=False):
    """MG candidate. LSE is base 2, unlike ordinary Kimi's natural LSE.

    Dense: one request; table is the ordinary 64-record page table and lengths
    contains total cache length (including query rows). Sparse: selector slots
    [rows,2048] with valid leading lengths [rows], including zero-length rows.
    ``bf16_compute`` selects BF16 QK and PV with exact keys over the same
    FP8 records.
    """
    if g.kv_cache != "fp8":
        raise ValueError("MG traits support the 656-byte FP8 record, not raw BF16 KV")
    if not 1 <= int(max_rows) <= 4096 or not int(max_rows) <= int(max_context) <= 1048576:
        raise ValueError("invalid planned row/context capacity")
    capacity = 2048 if sparse else int(max_context)
    launch = _MG(g, capacity, sparse=bool(sparse), bf16_compute=bool(bf16_compute))
    return compile_program(launch, name="kimi_mg_prefill_experiment",
        operands=(Operand("query", torch.bfloat16, f"[rows,{g.heads},576]"),
                  Operand("cache", torch.uint8, "[pool_tokens,656]"),
                  Operand("table", torch.int32, "[rows,2048] or [pages]", align=4),
                  Operand("lengths", torch.int32, "[rows] or [1]", align=4),
                  Operand("out", torch.bfloat16, f"[rows,{g.heads},512]", "out"),
                  Operand("lse", torch.float32, f"[rows,{g.heads}]", "out"),
                  Operand("scratch", torch.uint8, "[scratch_bytes]", "scratch")),
        scalars=(Scalar("rows"),),
        key=(g, int(max_rows), capacity, bool(sparse), bool(bf16_compute)),
        geometry={**vars(g), "max_rows": int(max_rows), "capacity": capacity,
                  "sparse": bool(sparse), "bf16_compute": bool(bf16_compute), "lse_base": 2},
        scratch={"scratch": lambda rows: 0 if sparse else 4 * (capacity + int(rows))})
