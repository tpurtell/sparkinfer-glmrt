"""Exportable V4.1 Spark slice pipeline with caller-owned persistent scratch.

The ordered default writes FP32 route planes. Opt-in atomic_tokens clears and
writes a flat FP32 token output without materializing route/slice intermediates.
Both paths keep live rows out of compile keys and bound launches by live work.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64, Float32
from b12x._lib.utils import runtime_num_sm
from b12x.moe._shared.kernels.v41_route_plan import V41RoutePlan, V41SliceReduce
from b12x.moe._shared.kernels.w4a8_v41_slice import (
    V41FusedSliceKernel,
    V41SliceFc1Kernel,
    V41SliceFc2Kernel,
)


def fc1_partial_words(capacity, width, topk, intermediate, activation_precision="a8", cta_splits=1):
    """Caller-owned compact activation words followed by raw FP32 split planes."""
    if cta_splits not in (1, 2, 4, 8) or activation_precision not in ("a8", "a16"):
        raise ValueError("invalid native FC1 scratch contract")
    slices = (intermediate + width - 1) // width
    compact = width // 2 if activation_precision == "a16" else width // 4 + width // 32
    raw = 2 * width * cta_splits if cta_splits > 1 else 0
    return slices * capacity * topk * (compact + raw)


class V41SlicePipeline:
    def __init__(self, capacity, width, atomic_tokens=False, *, experts=384, topk=6, intermediate=576,
                 hidden=5120, exact_storage=False, fc2_n_splits=1, scale_codes=False,
                 activation_precision="a8", swiglu_limit=10.0,
                 silu_bf16_product=False, route_weight_before_fc2=True,
                 round_fc2_bf16=False, fc1_k_splits=1, fc1_cta_splits=1):
        self.capacity = capacity
        self.hidden = hidden
        self.atomic_tokens = atomic_tokens
        self.experts = experts
        self.topk = topk
        if activation_precision not in ("a8", "a16"):
            raise ValueError("native MXFP4 activation precision must be a8 or a16")
        if swiglu_limit < 0:
            raise ValueError("SwiGLU clamp must be nonnegative")
        if atomic_tokens and (activation_precision != "a8" or swiglu_limit != 10.0
                              or silu_bf16_product or not route_weight_before_fc2
                              or round_fc2_bf16 or fc1_k_splits != 1 or fc1_cta_splits != 1):
            raise ValueError("atomic output requires the default activation contract")
        self.plan = V41RoutePlan(capacity, experts, topk, route_metadata=not atomic_tokens)
        if atomic_tokens:
            self.compute = V41FusedSliceKernel(width, grouped=True, atomic_tokens=True,
                                               intermediate=intermediate, hidden=hidden, exact_storage=exact_storage,
                                               fc2_n_splits=fc2_n_splits, scale_codes=scale_codes)
            self.fc1 = None
            self.fc2 = None
        else:
            self.compute = None
            self.fc1 = V41SliceFc1Kernel(width, capacity, topk=topk, intermediate=intermediate,
                                         hidden=hidden, exact_storage=exact_storage, scale_codes=scale_codes,
                                         activation_precision=activation_precision, swiglu_limit=swiglu_limit,
                                         silu_bf16_product=silu_bf16_product,
                                         route_weight_before_fc2=route_weight_before_fc2,
                                         fc1_k_splits=fc1_k_splits, fc1_cta_splits=fc1_cta_splits)
            self.fc2 = V41SliceFc2Kernel(width, capacity, topk=topk, intermediate=intermediate,
                                         hidden=hidden, exact_storage=exact_storage, fc2_n_splits=fc2_n_splits,
                                         scale_codes=scale_codes, activation_precision=activation_precision,
                                         route_weight_before_fc2=route_weight_before_fc2,
                                         round_fc2_bf16=round_fc2_bf16)
        self.reduce = V41SliceReduce(width, capacity, topk, intermediate=intermediate, hidden=hidden)
        self.partial_words = (1 if atomic_tokens else fc1_partial_words(
            capacity, width, topk, intermediate, activation_precision, fc1_cta_splits))

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
        groups = max(1, min(rows * self.topk, self.experts + max(rows * self.topk - self.experts, 0) // 16))
        if cutlass.const_expr(self.atomic_tokens):
            self.clear_output(output, rows).launch(
                grid=(max(1, (rows * self.hidden + 255) // 256), 1, 1),
                block=(256, 1, 1), stream=stream,
            )
            self.compute(
                x,
                xs,
                w13,
                s13,
                w2,
                s2,
                grouped,
                output,
                rows,
                stream,
                metadata,
                groups,
            )
        else:
            self.fc1(x, xs, w13, s13, grouped, partial, metadata, groups, stream)
            self.fc2(w2, s2, partial, output, metadata, inverse, live, groups, stream, grouped)

    @cute.kernel
    def publish(self, live: cute.Tensor, rows: Int32):
        if cute.arch.thread_idx()[0] == 0:
            live[0] = rows

    @cute.kernel
    def clear_output(self, output: cute.Tensor, rows: Int32):
        i = Int64(cute.arch.block_idx()[0]) * 256 + Int64(cute.arch.thread_idx()[0])
        if i < Int64(rows) * self.hidden:
            output[i] = Float32(0)


class V41RouteCombine:
    """Ordered original-slot FP32 sum, rounded once into BF16 token output."""

    def __init__(self, hidden, topk):
        self.hidden, self.topk = hidden, topk

    @cute.jit
    def __call__(self, source: cute.Tensor, output: cute.Tensor, rows: Int32,
                 stream: cuda.CUstream):
        self.kernel(source, output, rows).launch(
            grid=(max(1, (rows * self.hidden + 255) // 256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor, rows: Int32):
        from b12x.moe._shared.kernels.w4a8_v41_slice import fadd_rn_f32
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        if index < Int64(rows) * self.hidden:
            row, col = index // self.hidden, index % self.hidden
            value = Float32(0)
            for slot in cutlass.range_constexpr(self.topk):
                value = fadd_rn_f32(value, source[row * self.topk + slot, col])
            output[row, col] = value.to(cutlass.BFloat16)


class V41DraftSlicePipeline:
    """Local dSpark BF16 input through FP8 quantization and ordered route output.

    All temporaries belong to the caller. The quantizer uses the same K32,
    subgroup-four, 1e-4-floor contract as the native input quantizer.
    """

    # Qualified model recipe; explicit widths remain available for comparison.
    DEFAULT_WIDTH = 192

    def __init__(self, capacity, width, *, scale_codes=False):
        from b12x._lib.quant.mxfp8_rows import _MXFP8RowsQuantLaunch
        self.quant = _MXFP8RowsQuantLaunch(
            5120, cutlass.BFloat16, 4, 256, 32, False, 1e-4, wire_rows=True
        )
        self.pipeline = V41SlicePipeline(
            capacity, width, experts=128, topk=3, intermediate=2304,
            scale_codes=scale_codes
        )

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
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
        # 160 K32 groups / row, 8 groups / warp, 8 warps / CTA.
        grid = min(max(1, (rows * 20 + 7) // 8), runtime_num_sm() * Int32(4))
        self.quant(source.iterator, x.iterator, xs.iterator, xs.iterator,
                   rows, Int32(5120), grid, stream)
        self.pipeline(x, xs, w13, s13, w2, s2, ids, routing, live, packed,
                      counts, prefixes, metadata, grouped, inverse, partial,
                      output, rows, stream)
