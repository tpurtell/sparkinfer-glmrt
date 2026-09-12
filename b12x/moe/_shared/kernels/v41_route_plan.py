"""Allocation-free stable route grouping for the experimental V4.1 slices.

All buffers are caller-owned. Capacity and expert geometry are static; live
rows are read from a device scalar on every replay. Metadata capacity must be
capacity * topk (the worst case of one group per route). Invalid expert IDs
produce inverse=-1. Duplicate IDs are retained as separate routes.
"""

import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Uint32, Float32, range_constexpr


class V41RoutePlan:
    def __init__(self, capacity, experts=384, topk=6):
        assert capacity > 0 and experts > 0 and topk > 0
        self.capacity = capacity
        self.experts = experts
        self.topk = topk
        self.routes = capacity * topk

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        weights: cute.Tensor,
        live_rows: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped_weights: cute.Tensor,
        inverse: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.pack(ids, live_rows, packed, counts, metadata, inverse).launch(
            grid=(self.experts, 1, 1), block=(32, 1, 1), stream=stream
        )
        self.prefix(counts, prefixes).launch(
            grid=((self.experts + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )
        self.scatter(
            weights, packed, counts, prefixes, metadata, grouped_weights, inverse
        ).launch(grid=(self.experts, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def pack(
        self,
        ids: cute.Tensor,
        live_rows: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        metadata: cute.Tensor,
        inverse: cute.Tensor,
    ):
        expert = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        for i in range(expert * 32 + lane, self.routes, self.experts * 32):
            metadata[i, 1] = Int32(0)
            inverse[i] = Int32(-1)
        rows = max(Int32(0), min(live_rows[0], Int32(self.capacity)))
        total = Int32(0)
        for base in range(0, rows * self.topk, 32):
            pair = base + lane
            match = False
            if pair < rows * self.topk:
                match = ids[pair] == expert
            ballot = cute.arch.vote_ballot_sync(match)
            offset = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
            if match:
                packed[Int64(expert) * self.routes + total + offset] = pair
            total += cute.arch.popc(ballot)
        if lane == 0:
            counts[expert] = total

    @cute.kernel
    def prefix(self, counts: cute.Tensor, prefixes: cute.Tensor):
        expert = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if expert < self.experts:
            routes = Int32(0)
            groups = Int32(0)
            for e in range(expert):
                count = counts[e]
                routes += count
                groups += (count + 16 - 1) // 16
            prefixes[expert, 0] = routes
            prefixes[expert, 1] = groups

    @cute.kernel
    def scatter(
        self,
        weights: cute.Tensor,
        packed: cute.Tensor,
        counts: cute.Tensor,
        prefixes: cute.Tensor,
        metadata: cute.Tensor,
        grouped_weights: cute.Tensor,
        inverse: cute.Tensor,
    ):
        expert = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        count = counts[expert]
        base = prefixes[expert, 0]
        group_base = prefixes[expert, 1]
        for j in range(lane, ((count + 16 - 1) // 16) * 16, 32):
            group = group_base + j // 16
            local = j % 16
            if local == 0:
                metadata[group, 0] = expert
                metadata[group, 1] = min(Int32(16), count - j)
                metadata[group, 2] = base + j
            row = Int32(-1)
            if j < count:
                pair = packed[Int64(expert) * self.routes + j]
                row = pair // self.topk
                grouped_weights[base + j] = weights[pair]
                inverse[pair] = base + j
            metadata[group, 3 + local] = row


class V41SliceReduce:
    """Ordered FP32 slice sum into original route order; invalid routes are zero."""

    def __init__(self, width, capacity, topk=6, *, intermediate=576):
        assert width in (64, 128, 192) and capacity > 0 and topk > 0
        self.capacity = capacity
        assert intermediate > 0 and intermediate % 32 == 0
        self.slices = (intermediate + width - 1) // width
        self.routes = capacity * topk
        self.topk = topk

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        inverse: cute.Tensor,
        live_rows: cute.Tensor,
        stream: cuda.CUstream,
        rows: Int32 = -1,
    ):
        launch_rows = self.capacity if rows < 0 else min(rows, self.capacity)
        self.kernel(source, dest, inverse, live_rows).launch(
            grid=(max(1, (launch_rows * self.topk * 5120 + 255) // 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        inverse: cute.Tensor,
        live_rows: cute.Tensor,
    ):
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        route, col = index // 5120, index % 5120
        if route < min(live_rows[0] * self.topk, self.routes):
            grouped = inverse[route]
            value = Float32(0)
            if grouped >= 0:
                for plane in range_constexpr(self.slices):
                    value += source[plane, Int64(grouped), col]
            dest[route, col] = value
