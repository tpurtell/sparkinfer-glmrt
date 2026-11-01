"""Benchmark-only V4 semantic adapter around public row-major STREAM GEMMs."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from b12x.integration.cuteafd.fp8_moe import _StreamRoute
from b12x.moe._shared.kernels.w4a8_v41_slice import fadd_rn_f32


class V4StreamContract(_StreamRoute):
    def __init__(self, geometry, capacity, *, native_resident=False, native_exact=False, v4=True,
                 scale_codes=False):
        super().__init__(geometry, capacity, True, mxfp4_down_a8=True)
        self.v4 = bool(v4)
        for kernel in (self.gate_up, self.down):
            kernel.native_resident, kernel.native_exact = bool(native_resident), bool(native_exact)
            kernel.scale_codes = bool(scale_codes)
        if scale_codes and not native_resident:
            raise ValueError("compact scale codes require native resident weights")
        self.gate_up.silu_bf16_product = not v4
        self.gate_up.route_weight_before_fc2 = v4
        self.down.round_output_bf16 = not v4

    @cute.jit
    def __call__(self, x, ids, routing, w1, s1, w3, s3, w2, s2, output,
                 meta, pair_row, pair_pos, act, down, grouped_routing, rows, stream):
        pairs = rows * self.g.top_k
        tiles = (pairs + 127) // 128 + min(pairs, self.g.experts)
        self.prep(ids, meta, pair_row, pair_pos, rows, pairs, stream)
        self.group_weights(routing, pair_pos, grouped_routing, pairs).launch(
            grid=(max(1, (pairs + 255) // 256), 1, 1), block=(256, 1, 1), stream=stream)
        self.gate_up(x, pair_row, meta, w1, s1, w3, s3, act, tiles, stream, grouped_routing)
        self.down(act, meta, w2, s2, down, tiles, stream)
        self.ordered_combine(down, pair_pos, routing, output, rows).launch(
            grid=(max(1, (rows * self.g.hidden + 255) // 256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def group_weights(self, routing: cute.Pointer, positions: cute.Pointer,
                      grouped: cute.Pointer, pairs: Int32):
        i = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if i < pairs:
            position = positions[i]
            if position >= 0:
                grouped[position] = routing[i]

    @cute.kernel
    def ordered_combine(self, down: cute.Pointer, positions: cute.Pointer, routing: cute.Pointer,
                        output: cute.Pointer, rows: Int32):
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        if index < Int64(rows) * self.g.hidden:
            row, col = index // self.g.hidden, index % self.g.hidden
            value = Float32(0)
            for slot in cutlass.range_constexpr(self.g.top_k):
                position = positions[row * self.g.top_k + slot]
                if position >= 0:
                    route = Float32(down[Int64(position) * self.g.hidden + col])
                    if cutlass.const_expr(not self.v4):
                        route *= routing[row * self.g.top_k + slot]
                    value = fadd_rn_f32(value, route)
            output[index] = value.to(cutlass.BFloat16)
