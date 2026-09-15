"""Native pointer ABI for direct DS41RT FP4 attention plus sink merge.

Internal AOT surface: host integration validates all pointers, capacities,
alignment, immutable leases and scratch disjointness. All row counts are live
launch arguments. No allocation, compilation or host synchronization in replay.
"""
from dataclasses import replace
import math

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Uint64

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from .kernel import UnifiedDecodeKernel
from .merge import SparseMLASplitDecodeSinkMergeKernel
from .smem import make_smem_layout
from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits


class NativeV41Attention:
    def __init__(self):
        traits = make_unified_traits(ModelType.DSV41, ComputeMode.FP8,
                                     ScaleFormat.NVFP4_E4M3, fp8_rope=False)
        traits = replace(traits, compute_mode=ComputeMode.FP8, fp8_internal=True,
                         q_nope_stride=528, kv_smem_stride=624,
                         nt_per_warp_xv=traits.nt_per_warp_xv*2)
        self.decode = UnifiedDecodeKernel(traits, make_smem_layout(traits), 64, 1,
            h_blocks=4, num_splits=10, num_heads=64, q_head_dim=512,
            topk=128, extra_topk=512, q_stride=(32768, 512, 1),
            swa_indices_stride0=512, extra_indices_stride0=512,
            mid_out_stride=(327680, 5120, 512, 1), mid_lse_stride=(640, 10, 1),
            has_extra=True, pbs_extra=256, valid_hpb=16, native_dsv41_fp8=True)
        self.merge = SparseMLASplitDecodeSinkMergeKernel(static_num_chunks=10)

    @cute.jit
    def __call__(self, query: cute.Pointer, descriptors: cute.Pointer,
                 metadata: cute.Pointer, selected: cute.Pointer, bounds: cute.Pointer,
                 sink: cute.Pointer, partials: cute.Pointer, lses: cute.Pointer,
                 output: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        q = cute.make_tensor(query, cute.make_layout((rows, 64, 512), stride=(32768, 512, 1)))
        desc = cute.make_tensor(descriptors, cute.make_layout((rows, 15), stride=(15, 1)))
        meta = cute.make_tensor(metadata, cute.make_layout((rows, 10), stride=(10, 1)))
        ids = cute.make_tensor(selected, cute.make_layout((rows, 512), stride=(512, 1)))
        begins = cute.make_tensor(bounds, cute.make_layout((rows,), stride=(1,)))
        p = cute.make_tensor(partials, cute.make_layout((rows, 64, 10, 512), stride=(327680, 5120, 512, 1)))
        lse = cute.make_tensor(lses, cute.make_layout((rows, 64, 10), stride=(640, 10, 1)))
        sinks = cute.make_tensor(sink, cute.make_layout((64,), stride=(1,)))
        out = cute.make_tensor(output, cute.make_layout((rows, 64, 512), stride=(32768, 512, 1)))
        self.decode.call_native_v41(q, desc, meta, ids, begins, p, lse,
                                    Float32(512**-.5 * math.log2(math.e)), rows, stream)
        # Chunk-count tensor is unused by the static 10-part merge.
        self.merge(p, lse, begins, sinks, out, stream)


def compile_native_v41_attention_aot():
    launch = NativeV41Attention()
    key = (64, 512, 128, 512, 10, torch.cuda.current_device())
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    types = (BFloat16, Uint64, Uint64, Int32, Uint64, Float32, BFloat16, Float32, BFloat16)
    alignments = (16, 8, 8, 4, 8, 4, 16, 4, 16)
    pointers = tuple(make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)
                     for dtype, align in zip(types, alignments))
    return compile_cute(launch, *pointers, Int32(1), current_cuda_stream(),
                        compile_spec=KernelCompileSpec.from_key("attention.native_v41", 1, key))
