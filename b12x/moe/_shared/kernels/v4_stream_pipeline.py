"""Ordered native MXFP4 V4/V4.1 prefill over caller-owned scratch.

Weights and scales retain their decode residency. Original-slot FP32 route
planes use the native engine's existing ordered fold and BF16 compactor.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from b12x.integration.cuteafd._fp8_moe_kernels import MoePrep, meta_words
from b12x.integration.cuteafd._fp8_moe_stream import stream_max_tiles
from b12x.integration.cuteafd._mxfp4_down_a8_plan import mxfp8_down_row_bytes
from b12x.integration.cuteafd._mxfp4_moe_a8 import StreamMxfp4DownA8
from b12x.integration.cuteafd._mxfp4_moe_stream import StreamMxfp4GateUp
from b12x.moe._shared.kernels.w4a8_v41_slice import fadd_rn_f32


def stream_scratch_specs(capacity, hidden, intermediate, experts, topk, *, row_addresses=False):
    """Static tensor specs, shared by the exporter and scratch-size query."""
    if (capacity not in ((8192,) if row_addresses else (256, 1024, 4096)) or hidden <= 0 or hidden % 512
            or intermediate <= 0 or intermediate % 32 or experts <= 0 or topk <= 0):
        raise ValueError("invalid native V4 prefill geometry")
    pairs = capacity * topk
    tiles = stream_max_tiles(experts, pairs)
    return (("meta", "i32", (meta_words(experts, tiles),)),
            ("pair_row", "i32", (pairs,)), ("pair_pos", "i32", (pairs,)),
            ("act", "u8", (tiles * 128 * mxfp8_down_row_bytes(intermediate),)),
            ("grouped_routing", "f32", (pairs,)), ("grouped_route", "i32", (pairs,)),
            ("output", "f32", (pairs, hidden)))


def stream_scratch_layout(capacity, hidden, intermediate, experts, topk, *, row_addresses=False):
    offset, layout = 0, []
    for name, dtype, shape in stream_scratch_specs(capacity, hidden, intermediate, experts, topk, row_addresses=row_addresses):
        offset = (offset + 15) // 16 * 16
        size = 1 if dtype == "u8" else 4
        for extent in shape:
            size *= extent
        layout.append(dict(name=name, dtype=dtype, shape=shape, offset=offset, nbytes=size))
        offset += size
    return tuple(layout), offset


class NativeTokenCombine:
    """Benchmark equivalent of native compact_routes<Hidden, Routes>."""

    def __init__(self, hidden, topk, *, row_addresses=False):
        self.hidden, self.topk = hidden, topk
        self.row_addresses = bool(row_addresses)

    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, rows: Int32, stream):
        self.kernel(source, output, rows).launch(
            grid=(max(1, (rows * self.hidden + 255) // 256), 1, 1),
            block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, output: cute.Pointer, rows: Int32):
        index = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        if index < Int64(rows) * self.hidden:
            row, col = index // self.hidden, index % self.hidden
            value = Float32(source[row * self.topk * self.hidden + col])
            for slot in cutlass.range_constexpr(1, self.topk):
                value = fadd_rn_f32(value, Float32(source[(row * self.topk + slot) * self.hidden + col]))
            if cutlass.const_expr(self.row_addresses):
                address = cute.make_tensor(cute.recast_ptr(output, dtype=Int64), cute.make_layout(1))[row]
                target = cute.make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=16)
                cute.make_tensor(target, cute.make_layout(self.hidden))[col] = value.to(cutlass.BFloat16)
            else:
                output[index] = value.to(cutlass.BFloat16)


class V4StreamPipeline:
    """One ordered A8 prefill recipe; native slice decode remains separate."""

    def __init__(self, capacity, *, hidden, intermediate, experts, topk=6, scale_codes=False, row_addresses=False):
        self.layout, self.scratch_bytes = stream_scratch_layout(
            capacity, hidden, intermediate, experts, topk, row_addresses=row_addresses)
        self.row_addresses = bool(row_addresses)
        self.capacity, self.hidden, self.intermediate = capacity, hidden, intermediate
        self.experts, self.topk, self.scale_codes = experts, topk, bool(scale_codes)
        self.max_tiles = stream_max_tiles(experts, capacity * topk)
        self.prep = MoePrep(experts=experts, top_k=topk, pad=1, max_tiles=self.max_tiles,
                            tile_rows=128, chunked=True)
        self.gate_up = StreamMxfp4GateUp(hidden=hidden, inter=intermediate, experts=experts,
            limit=10., qmma=True, wire=True, act_mxfp8=True, silu_bf16_product=False,
            route_weight_before_fc2=True, native_resident=True,
            native_exact=intermediate % 128 != 0, scale_codes=scale_codes, row_addresses=row_addresses)
        self.down = StreamMxfp4DownA8(hidden=hidden, inter=intermediate, experts=experts,
            round_output_bf16=False, native_resident=True, original_routes=True,
            native_exact=intermediate % 128 != 0, scale_codes=scale_codes)

    def key(self):
        return ("v4_native_stream", 2, self.capacity, self.hidden, self.intermediate,
                self.experts, self.topk, self.scale_codes, self.gate_up.key(), self.down.key()) + (("row_addresses",) if self.row_addresses else ())

    @cute.jit
    def __call__(self, x: cute.Pointer, w13: cute.Pointer, s13: cute.Pointer,
                 w2: cute.Pointer, s2: cute.Pointer, ids: cute.Pointer, routing: cute.Pointer,
                 meta: cute.Pointer, pair_row: cute.Pointer, pair_pos: cute.Pointer,
                 act: cute.Pointer, grouped_routing: cute.Pointer, grouped_route: cute.Pointer,
                 output: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        pairs = rows * self.topk
        tiles = (pairs + 127) // 128 + min(pairs, self.experts)
        self.prep(ids, meta, pair_row, pair_pos, rows, pairs, stream)
        self.group_weights(routing, pair_pos, grouped_routing, grouped_route, output, pairs).launch(
            grid=(max(1, (pairs + 255) // 256), 1, 1), block=(256, 1, 1), stream=stream)
        self.gate_up(x, pair_row, meta, w13, s13, w13, s13, act, tiles, stream, grouped_routing)
        self.down(act, meta, w2, s2, output, tiles, stream, grouped_route)

    @cute.kernel
    def group_weights(self, routing: cute.Pointer, positions: cute.Pointer,
                      grouped: cute.Pointer, grouped_route: cute.Pointer,
                      output: cute.Pointer, pairs: Int32):
        index = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if index < pairs:
            position = positions[index]
            if position >= 0:
                grouped[position] = routing[index]
                grouped_route[position] = index
            else:
                # Invalid routes have no DownA8 CTA, so publish their zero plane here.
                for col in cutlass.range(self.hidden, unroll=1):
                    output[Int64(index) * self.hidden + col] = Float32(0)


class V4StreamSegments:
    """Native ordered expert pipeline with input gather and BF16 reply scatter."""

    def __init__(self, capacity, *, hidden, intermediate, experts, topk=6, scale_codes=False):
        self.pipeline = V4StreamPipeline(capacity, hidden=hidden, intermediate=intermediate,
            experts=experts, topk=topk, scale_codes=scale_codes, row_addresses=True)
        self.combine = NativeTokenCombine(hidden, topk, row_addresses=True)

    @cute.jit
    def __call__(self, x: cute.Pointer, w13: cute.Pointer, s13: cute.Pointer,
                 w2: cute.Pointer, s2: cute.Pointer, ids: cute.Pointer, routing: cute.Pointer,
                 meta: cute.Pointer, pair_row: cute.Pointer, pair_pos: cute.Pointer,
                 act: cute.Pointer, grouped_routing: cute.Pointer, grouped_route: cute.Pointer,
                 partial: cute.Pointer, output: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.pipeline(x, w13, s13, w2, s2, ids, routing, meta, pair_row, pair_pos,
                      act, grouped_routing, grouped_route, partial, rows, stream)
        self.combine(partial, output, rows, stream)


def compile_v4_stream_segments_aot(*, hidden=5120, intermediate=576, experts=384, topk=6,
                                   scale_codes=False, capacity=8192):
    """Compile beside ordinary native programs; live row counts never specialize."""
    import torch
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program

    # Weights are opaque native arena views; extents are validated by the loader.
    operands = (Operand("input_rows", torch.int64, "[rows]"),
        *(Operand(name, torch.uint32, "[native_weights]") for name in ("w13", "s13", "w2", "s2")),
        Operand("ids", torch.int32, f"[rows,{topk}]"),
        Operand("routing", torch.float32, f"[rows,{topk}]"))
    specs = stream_scratch_specs(capacity, hidden, intermediate, experts, topk, row_addresses=True)
    types = {"i32": torch.int32, "u8": torch.uint8, "f32": torch.float32}
    operands += tuple(Operand("partial" if name == "output" else name, types[dtype],
        "[" + ",".join(map(str, shape)) + "]", "scratch") for name, dtype, shape in specs)
    operands += (Operand("output_rows", torch.int64, "[rows]", "out"),)
    layout, _ = stream_scratch_layout(capacity, hidden, intermediate, experts, topk, row_addresses=True)
    sizes = {entry["name"]: entry["nbytes"] for entry in layout}
    sizes["partial"] = sizes.pop("output")
    return compile_program(V4StreamSegments(capacity, hidden=hidden, intermediate=intermediate,
        experts=experts, topk=topk, scale_codes=scale_codes), name="v4_native_stream_segments",
        operands=operands, scalars=(Scalar("rows"),),
        key=(capacity, hidden, intermediate, experts, topk, bool(scale_codes)),
        geometry=dict(max_rows=capacity, hidden=hidden, intermediate=intermediate, experts=experts,
            topk=topk, row_addresses=True, scale_codes=bool(scale_codes)),
        scratch={name: (lambda _rows, size=size: size) for name, size in sizes.items()})
