"""Exportable V4.1 Spark slice pipeline with caller-owned persistent scratch."""

import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32
from b12x.moe._shared.kernels.v41_route_plan import V41RoutePlan, V41SliceReduce
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel


class V41SlicePipeline:
    def __init__(self, capacity, width):
        self.capacity = capacity
        self.plan = V41RoutePlan(capacity)
        self.compute = V41FusedSliceKernel(width, grouped=True)
        self.reduce = V41SliceReduce(width, capacity)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        xs: cute.Tensor,
        w13: cute.Tensor,
        s13: cute.Tensor,
        w2: cute.Tensor,
        s2: cute.Tensor,
        ids: cute.Tensor,
        routing: cute.Tensor,
        live: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped: cute.Tensor,
        inverse: cute.Tensor,
        partial: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.publish(live, rows).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        self.plan(
            ids,
            routing,
            live,
            packed,
            counts,
            prefixes,
            metadata,
            grouped,
            inverse,
            stream,
        )
        self.compute(
            x,
            xs,
            w13,
            s13,
            w2,
            s2,
            grouped,
            partial,
            rows,
            stream,
            metadata,
            Int32(self.capacity * 6),
        )
        self.reduce(partial, output, inverse, live, stream)

    @cute.kernel
    def publish(self, live: cute.Tensor, rows: Int32):
        if cute.arch.thread_idx()[0] == 0:
            live[0] = rows
