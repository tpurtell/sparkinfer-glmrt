"""Native TP2 compact pipeline consuming the existing FP8 K32 wire rows.

The signature matches V41SlicePipeline so the native expert ABI is unchanged.
Only intermediate, projections and FP32 route output require scratch. Counts
is bound to the native initialized unit-scale vector by the AOT bridge.
"""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Uint8
from b12x.moe._shared.kernels.w4a8_compact_micro import _DirectW4A8CompactLaunch


class V41CompactPipeline:
    def __init__(self, capacity, sm_count):
        self.sm_count = int(sm_count)
        self.compact = _DirectW4A8CompactLaunch(
            max_tokens=capacity, num_topk=6, k=5120, n=1152, experts=384,
            input_scale_count=384, down_scale_count=384, swiglu_limit=10,
            fast_math=False, native_v41=True, wire_rows=True)

    @cute.jit
    def __call__(
        self, x: cute.Tensor, xs: cute.Tensor,
        w13: cute.Tensor, s13: cute.Tensor, w2: cute.Tensor, s2: cute.Tensor,
        ids: cute.Tensor, routing: cute.Tensor, live: cute.Tensor,
        packed: cute.Tensor, counts: cute.Tensor, prefixes: cute.Tensor,
        metadata: cute.Tensor, grouped: cute.Tensor, inverse: cute.Tensor,
        partial: cute.Tensor, output: cute.Tensor,
        rows: Int32, stream: cuda.CUstream,
    ):
        self.compact(
            cute.recast_ptr(x.iterator, dtype=Uint8), xs.iterator,
            w13.iterator, s13.iterator, packed.iterator, partial.iterator,
            routing.iterator, ids.iterator, counts.iterator, counts.iterator,
            w2.iterator, s2.iterator, output.iterator, counts.iterator,
            counts.iterator, rows * Int32(6), Int32(self.sm_count), stream)
