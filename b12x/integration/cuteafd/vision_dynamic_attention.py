"""Runtime-row view of the existing contiguous varlen vision attention kernel.

Legacy static wrappers are deliberately unchanged. TMA tensor extents, LSE head
stride and launch upper bound all use the caller's admitted live row extent.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32

from ._common import Operand, Scalar, compile_program


class DynamicVisionAttention:
    def __init__(self, heads, dim, segments=1):
        from b12x.attention._shared.contiguous.forward import ContiguousAttentionForwardKernel
        self.heads, self.dim, self.segments = heads, dim, segments
        self.kernel = ContiguousAttentionForwardKernel(
            BFloat16, dim, head_dim_v=dim, qhead_per_kvhead=1,
            is_causal=False, is_local=False, pack_gqa=False, tile_m=128, tile_n=64)

    @cute.jit
    def __call__(self, q: cute.Pointer, k: cute.Pointer, v: cute.Pointer,
                 out: cute.Pointer, lse: cute.Pointer, lengths: cute.Pointer,
                 rows: Int32, scale: Float32, l2_bytes: Int32, stream: cuda.CUstream):
        shape = (rows, self.heads, self.dim)
        stride = (self.heads * self.dim, self.dim, 1)
        tensors = tuple(cute.make_tensor(pointer,cute.make_layout(shape,stride=stride))
                        for pointer in (q,k,v,out))
        lse_tensor = cute.make_tensor(lse,cute.make_layout((self.heads,rows),stride=(rows,1)))
        cumulative = cute.make_tensor(lengths,cute.make_layout((self.segments+1,),stride=(1,)))
        # With cumulative lengths both SeqlenInfo and the varlen scheduler read
        # the real sequence extent from device memory. Static non-varlen bounds
        # are unused; the dynamic Q extent controls the launch upper bound.
        # With several segments (images of one batch), each attends only to
        # itself; trailing segments padded with the last offset are empty.
        self.kernel(*tensors,lse_tensor,scale,mCuSeqlensQ=cumulative,mCuSeqlensK=cumulative,
            has_attention_sink_bias=False,logical_num_batch_static=1,
            logical_seqlen_q_static=0,logical_seqlen_k_static=0,
            l2_cache_size=l2_bytes,window_size_left=None,window_size_right=None,stream=stream)


def compile_dynamic_vision_attention_aot(*, heads, dim, segments=1):
    """Per-image full attention over runtime rows.

    ``lengths`` holds ``segments + 1`` cumulative row offsets; a batch of fewer
    images pads the tail with the last live offset (empty segments write
    nothing). ``segments`` is a planned capacity, never a live image count.
    One segment keeps the historical compile key and ABI byte-identical.
    """
    if (heads,dim) not in ((12,128),(16,72),(16,80),(16,128)):
        raise ValueError("dynamic vision attention requires12x128,16x72,16x80 or shared16x128")
    if not isinstance(segments,int) or segments<1 or segments>64:
        raise ValueError("dynamic vision attention segments must be a planned capacity in 1..64")
    key=(heads,dim) if segments==1 else (heads,dim,segments)
    geometry=dict(heads=heads,dim=dim,rows="runtime",causal=False)
    if segments>1:
        geometry["segments"]=segments
    from ._hy4_vision_kernels import Hy4DynamicVisionAttention
    launch = (Hy4DynamicVisionAttention() if (heads,dim,segments)==(16,72,8)
              else DynamicVisionAttention(heads,dim,segments))
    return compile_program(launch,name="dynamic_vision_attention",
        operands=(Operand("q",torch.bfloat16,f"[rows,{heads},{dim}]"),
                  Operand("k",torch.bfloat16,f"[rows,{heads},{dim}]"),
                  Operand("v",torch.bfloat16,f"[rows,{heads},{dim}]"),
                  Operand("out",torch.bfloat16,f"[rows,{heads},{dim}]","out"),
                  Operand("lse",torch.float32,f"[{heads},rows]","scratch"),
                  Operand("lengths",torch.int32,f"[{segments+1}]")),
        scalars=(Scalar("rows"),Scalar("scale","float32"),Scalar("l2_bytes","int32")),
        key=key,geometry=geometry)
