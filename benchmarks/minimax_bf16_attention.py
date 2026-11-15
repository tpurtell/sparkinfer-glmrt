"""Benchmark-only BF16 MiniMax adapter; no exported program or serving ABI."""
from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from b12x.attention.paged.forward_extend_generic import PagedForwardKernel
from b12x.integration.cuteafd.minimax_attention import _Attention


class Bf16Attention(_Attention):
    def __init__(self, max_rows, max_batch, mode, query_heads=64, kv_heads=4):
        super().__init__(max_rows, max_batch, mode, query_heads, kv_heads)
        self.record_bytes = kv_heads * 128
        traits = replace(self.forward.traits, kv_dtype=torch.bfloat16,
                         upcast_stride_k=16, upcast_stride_v=16)
        self.forward = PagedForwardKernel(
            cutlass.BFloat16, cutlass.BFloat16, cutlass.BFloat16, cutlass.BFloat16,
            traits=traits, msa_block_sparse=True, msa_union_tile=mode == "extend", page_size=64)

    @cute.jit
    def __call__(self, query: cute.Pointer, key: cute.Pointer, value: cute.Pointer,
                 table: cute.Pointer, lengths: cute.Pointer, offsets: cute.Pointer,
                 selected: cute.Pointer, output: cute.Pointer, lse: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, batch: Int32, stride: Int32,
                 pages: Int64, stream: cuda.CUstream):
        # The parent restores Uint8 pointers at its adapter boundary. Preserve
        # all metadata behavior but give the BF16 kernel typed cache pointers.
        self.run(query, key, value, table, lengths, offsets, selected,
                 output, lse, scratch, rows, batch, stride, pages, stream)

    @cute.jit
    def run(self, query, key, value, table, lengths, offsets, selected, output, lse,
            scratch, rows, batch, stride, pages, stream):
        work = rows if cutlass.const_expr(self.mode == "decode") else (rows + 7) // 8 + batch
        requests = self._scratch(scratch, self.offsets["requests"])
        tiles = self._scratch(scratch, self.offsets["tiles"])
        zeros = self._scratch(scratch, self.offsets["zeros"])
        valid = self._scratch(scratch, self.offsets["valid"])
        blocks = self._scratch(scratch, self.offsets["union_blocks"])
        masks = self._scratch(scratch, self.offsets["union_masks"])
        counts = self._scratch(scratch, self.offsets["union_counts"])
        self.metadata(offsets, requests, tiles, zeros, valid, work, batch).launch(
            grid=((work + batch + 128) // 128, 1, 1), block=(128, 1, 1), stream=stream)
        union_blocks, union_masks, union_counts = None, None, None
        if cutlass.const_expr(self.mode == "extend"):
            self.union(selected, lengths, offsets, requests, tiles, valid, blocks, masks, counts, work).launch(
                grid=(work, self.kv_heads, 1), block=(32, 1, 1), stream=stream)
            union_blocks = self._tensor(blocks, (work, self.kv_heads, 128), (self.kv_heads * 128, 128, 1))
            union_masks = self._tensor(masks, (work, self.kv_heads, 128), (self.kv_heads * 128, 128, 1))
            union_counts = self._tensor(counts, (work, self.kv_heads), (self.kv_heads, 1))
        cache_layout = cute.make_layout((Int32(pages), 64, self.kv_heads, 128),
                                       stride=(64 * self.kv_heads * 128, self.kv_heads * 128, 128, 1))
        dummy_float = self._tensor(cute.recast_ptr(zeros, dtype=cutlass.Float32), (1,), (1,))
        dummy_desc = self._tensor(cute.recast_ptr(zeros, dtype=Int64), (1,), (1,))
        self.forward(
            self._tensor(query, (rows, self.query_heads, 128), (self.query_heads * 128, 128, 1)),
            cute.make_tensor(key, cache_layout), cute.make_tensor(value, cache_layout),
            self._tensor(table, (batch, stride), (Int64(stride), 1)),
            self._tensor(lengths, (batch,), (1,)), self._tensor(offsets, (batch + 1,), (1,)),
            self._tensor(requests, (work,), (1,)), self._tensor(tiles, (work,), (1,)),
            self._tensor(zeros, (work,), (1,)), self._tensor(zeros, (batch + 1,), (1,)),
            self._tensor(zeros, (1,), (1,)), self._tensor(zeros, (batch,), (1,)),
            self._tensor(valid, (work,), (1,)),
            self._tensor(selected, (self.kv_heads, self.rows, 16), (self.rows * 16, 16, 1)),
            union_blocks, union_masks, union_counts, dummy_float, None,
            self._tensor(output, (rows, self.query_heads, 128), (self.query_heads * 128, 128, 1)),
            self._tensor(lse, (self.query_heads, rows), (1, self.query_heads)),
            None, None, dummy_desc, dummy_desc, stream)
