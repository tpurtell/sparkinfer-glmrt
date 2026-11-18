"""Inkling geometry-driven FP32 residual RMSNorm and BF16 K|V record packing.

No change to MiniMax's norm: its vectorized eight-elements/thread geometry
requires H multiple2048, whereas H4608 requires tail masks or a different
CTA shape. Reusing it would alter existing compile identity/resource layouts.
Here the norm covers weighted hidden4608 and unweighted V heads128, with
F32/BF16 output selected at compile time and no zero-centered weight.
KV packing scatters rows into one flat BF16 K|V cache, with I64 slot offsets;
negative slots are skipped. The same program fills local step records or
global paged pools without a host hop. Rows and slots are runtime quantities.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
import torch

from b12x._lib.intrinsics import cvt_f32x4_to_e4m3x4, div_rn_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _block_sum, _reduction_storage, _rsqrt, _fp8_scale, _warp_max


class _Norm:
    threads = 128

    def __init__(self, width, heads, weighted, eps):
        self.width, self.heads, self.weighted, self.eps = width, heads, weighted, float(eps)

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, weight, out).launch(grid=(rows, self.heads, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, weight: cute.Pointer, out: cute.Pointer):
        row, head, _ = cute.arch.block_idx()
        tid = Int64(cute.arch.thread_idx()[0])
        base = (Int64(row)*Int64(self.heads)+Int64(head))*Int64(self.width)
        storage = utils.SmemAllocator().allocate(_reduction_storage(1, self.threads//32))
        sums = storage.sums.get_tensor(cute.make_layout((1, self.threads//32)))
        values = cute.make_rmem_tensor((self.width//self.threads,), Float32)
        square = Float32(0)
        for j in cutlass.range_constexpr(self.width//self.threads):
            value = Float32(x[base+tid+Int64(j*self.threads)])
            values[j] = value
            square = square+value*value
        total = _block_sum(square, sums, Int32(0), self.threads//32)
        inv = _rsqrt(div_rn_f32(total, Float32(self.width))+Float32(self.eps))
        for j in cutlass.range_constexpr(self.width//self.threads):
            col = tid+Int64(j*self.threads)
            result = values[j]*inv
            if cutlass.const_expr(self.weighted):
                result = result*Float32(weight[Int64(head)*Int64(self.width)+col])
            out[base+col] = result


class _Pack:
    threads = 256

    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
                 cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(k, v, slots, cache).launch(grid=(rows, (self.width+self.threads-1)//self.threads, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer, cache: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1])*Int64(self.threads)+Int64(cute.arch.thread_idx()[0])
        slot = Int64(slots[row])
        if slot >= Int64(0) and col < Int64(self.width):
            cache[slot*Int64(2*self.width)+col] = Float32(k[row*Int64(self.width)+col])
            cache[slot*Int64(2*self.width)+Int64(self.width)+col] = Float32(v[row*Int64(self.width)+col])


class _PackFp8:
    """One warp per token/head/plane, with rank-local values and scales."""
    def __init__(self, heads):
        self.heads = heads
        self.scale_offset = 2 * heads * 128
        self.record_bytes = self.scale_offset + 2 * heads * 4

    @cute.jit
    def __call__(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer,
                 cache: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(k, v, slots, cache).launch(grid=(rows, self.heads, 2), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, k: cute.Pointer, v: cute.Pointer, slots: cute.Pointer, cache: cute.Pointer):
        row, head, plane = cute.arch.block_idx()
        lane = Int64(cute.arch.thread_idx()[0])
        source = k
        if plane == Int32(1):
            source = v
        values = cute.make_rmem_tensor((4,), Float32)
        maximum = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            values[j] = Float32(source[(Int64(row)*Int64(self.heads)+Int64(head))*Int64(128)+lane*Int64(4)+Int64(j)])
            maximum = cutlass.max(maximum, cute.math.abs(values[j]))
        scale = _fp8_scale(_warp_max(maximum))
        inverse = div_rn_f32(Float32(1.0), scale)
        packed = cvt_f32x4_to_e4m3x4(values[0]*inverse, values[1]*inverse,
                                    values[2]*inverse, values[3]*inverse)
        slot = Int64(slots[Int64(row)])
        if slot >= Int64(0):
            base = Int64(cache.toint())+slot*Int64(self.record_bytes)
            destination = cute.make_ptr(Uint32, base+(Int64(plane)*Int64(self.heads)+Int64(head))*Int64(128)
                                        +lane*Int64(4), cute.AddressSpace.gmem, assumed_align=4)
            destination[0] = packed
            if lane == Int64(0):
                scales = cute.make_ptr(Float32, base+Int64(self.scale_offset), cute.AddressSpace.gmem, assumed_align=4)
                scales[Int64(plane)*Int64(self.heads)+Int64(head)] = scale


class _PaddedTranspose:
    threads = 256

    def __init__(self, rows, cols, padded_cols, element_bytes):
        self.rows, self.cols, self.padded_cols = rows, cols, padded_cols
        self.element_bytes = element_bytes

    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, stream: cuda.CUstream):
        source = cute.recast_ptr(source, dtype=Uint8)
        output = cute.recast_ptr(output, dtype=Uint8)
        total = self.cols*self.padded_cols*self.element_bytes
        self.kernel(source, output).launch(grid=((total+self.threads-1)//self.threads, 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, output: cute.Pointer):
        index = Int64(cute.arch.block_idx()[0])*Int64(self.threads)+Int64(cute.arch.thread_idx()[0])
        if index < Int64(self.cols*self.padded_cols*self.element_bytes):
            byte = index%Int64(self.element_bytes)
            element = index//Int64(self.element_bytes)
            row, col = element//Int64(self.padded_cols), element%Int64(self.padded_cols)
            value = Uint8(0)
            if col < Int64(self.rows):
                value = Uint8(source[(col*Int64(self.cols)+row)*Int64(self.element_bytes)+byte])
            output[index] = value


def compile_padded_transpose_aot(*, dtype=torch.bfloat16, rows=16, cols=1024, padded_cols=64):
    """Bit-exact static tensor transpose+zero-pad in one device pass.

    Input [rows,cols] becomes output [cols,padded_cols]. No floating-point
    conversion occurs: signed zeros, NaN payloads and every storage bit survive.
    Geometry is immutable weight metadata, never a live request quantity.
    """
    if (dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.int8, torch.uint8,
                      torch.int32, torch.int64) or type(rows) is not int or type(cols) is not int
            or type(padded_cols) is not int or min(rows, cols, padded_cols) <= 0 or rows > padded_cols):
        raise ValueError("invalid padded transpose dtype/geometry")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    if cols*padded_cols*element_bytes > 2**63-1:
        raise ValueError("padded transpose extent exceeds signed 64-bit indexing")
    return compile_program(_PaddedTranspose(rows, cols, padded_cols, element_bytes), name="padded_transpose",
        operands=(Operand("source", dtype, f"[{rows},{cols}]"),
                  Operand("output", dtype, f"[{cols},{padded_cols}]", "out")),
        scalars=(), key=(dtype, rows, cols, padded_cols),
        geometry=dict(rows=rows, cols=cols, padded_cols=padded_cols, element_bytes=element_bytes,
                      dtype=str(dtype)), doc=__doc__)


def compile_inkling2_norm_aot(*, width=4608, heads=1, weighted=True, eps=1e-6,
                             output_dtype=torch.bfloat16, max_rows=4096):
    import math
    if (type(width) is not int or width <= 0 or width%128 or type(heads) is not int or heads <= 0
            or not math.isfinite(eps) or eps<=0 or output_dtype not in (torch.bfloat16, torch.float32)
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31-1):
        raise ValueError("invalid Inkling RMSNorm geometry")
    return compile_program(_Norm(width, heads, weighted, eps), name="inkling2_norm",
        operands=(Operand("x", torch.float32, f"[rows,{heads},{width}]"),
                  Operand("weight", torch.bfloat16, f"[{heads},{width}]"),
                  Operand("out", output_dtype, f"[rows,{heads},{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width, heads, weighted, eps, output_dtype, max_rows),
        geometry=dict(width=width, heads=heads, weighted=weighted, eps=eps,
                      output_dtype=str(output_dtype), max_rows=max_rows), doc=__doc__)


def compile_inkling2_kv_pack_aot(*, heads=8, dim=128, input_dtype=torch.float32, max_rows=4096, kv_format="bf16"):
    if (kv_format not in ("bf16", "fp8") or (kv_format == "fp8" and (heads, dim) not in ((8, 128), (4, 128)))
            or type(heads) is not int or heads <= 0 or type(dim) is not int or dim <= 0
            or input_dtype not in (torch.float32, torch.bfloat16) or type(max_rows) is not int
            or not 1 <= max_rows <= 2**31-1):
        raise ValueError("invalid Inkling KV pack geometry")
    width = heads*dim
    pack = _PackFp8(heads) if kv_format == "fp8" else _Pack(width)
    return compile_program(pack, name="inkling2_kv_pack",
        operands=(Operand("k", input_dtype, f"[rows,{heads},{dim}]"),
                  Operand("v", input_dtype, f"[rows,{heads},{dim}]"),
                  Operand("slots", torch.int64, "[rows]", align=8),
                  Operand("cache", torch.uint8 if kv_format == "fp8" else torch.bfloat16,
                          f"[slots,{pack.record_bytes}]" if kv_format == "fp8" else f"[slots,{2*width}]", "inout")),
        scalars=(Scalar("rows"),), key=("cache_v2", heads, dim, input_dtype, max_rows, kv_format),
        geometry=dict(heads=heads, dim=dim, input_dtype=str(input_dtype), max_rows=max_rows,
                      cache_abi=2, kv_format=kv_format, record_bytes=pack.record_bytes if kv_format == "fp8" else 4*width,
                      scale_offset=pack.scale_offset if kv_format == "fp8" else 0), doc=__doc__)
