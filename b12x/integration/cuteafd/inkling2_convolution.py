"""Inkling causal residual convolutions, including per-token global KV taps.

Static weights are BF16 [width, taps], oldest-first. Dynamic weights are
BF16 static_w [taps, width] (lag-first) plus [rows, heads, taps]. The dynamic
coefficient applies to every channel in its head, not to the source token.
Arithmetic is FP32, with separate multiply/add and one output cast, matching
cuteafd's #1915 short_conv; normalization is 1/sqrt(taps+1) for static and
1/sqrt(2*taps+1) for dynamic.

Rows form contiguous sequence segments. first[row] is that segment's first
row; slots[row] selects its exclusive history [slots, taps-1, width], stored
newest-first. History must be zeroed for fresh sequences. The convolution
reads the entire old history before a second kernel commits the final
(taps-1) inputs. This ordering supports arbitrary short chunks, graph replay,
and packed sequences without host work. A slot occurs in only one segment.
Input/output/history and weights are disjoint. No allocation occurs at run.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ._common import Operand, Scalar, compile_program


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    # Keep torch's materialized multiplication separate from the following add.
    return Float32(llvm.inline_asm(
        T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


class _Convolution:
    threads = 256

    def __init__(self, width, taps, heads, dynamic):
        self.width, self.taps, self.heads = int(width), int(taps), int(heads)
        self.dynamic = bool(dynamic)
        self.factor = (self.taps * (2 if self.dynamic else 1) + 1) ** -0.5

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, dynamic: cute.Pointer,
                 history: cute.Pointer, first: cute.Pointer, slots: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.convolve(x, weight, dynamic, history, first, slots, out, rows).launch(
            grid=(rows, (self.width + self.threads - 1) // self.threads, 1),
            block=(self.threads, 1, 1), stream=stream)
        self.commit(x, history, first, slots, rows).launch(
            grid=(rows, (self.width + self.threads - 1) // self.threads, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def convolve(self, x: cute.Pointer, weight: cute.Pointer, dynamic: cute.Pointer,
                 history: cute.Pointer, first: cute.Pointer, slots: cute.Pointer,
                 out: cute.Pointer, rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if row < Int64(rows) and col < Int64(self.width):
            begin = Int64(first[row])
            slot = Int64(slots[row])
            acc = Float32(0)
            for lag in cutlass.range_constexpr(self.taps):
                value = Float32(0)
                if row - Int64(lag) >= begin:
                    value = Float32(x[(row - Int64(lag)) * Int64(self.width) + col])
                else:
                    prior = Int64(lag) - (row - begin) - Int64(1)
                    value = Float32(history[(slot * Int64(self.taps - 1) + prior) * Int64(self.width) + col])
                tap = Float32(0)
                if cutlass.const_expr(self.dynamic):
                    head = col // Int64(self.width // self.heads)
                    tap = Float32(weight[Int64(lag * self.width) + col]) + Float32(
                        dynamic[(row * Int64(self.heads) + head) * Int64(self.taps) + Int64(lag)])
                else:
                    tap = Float32(weight[col * Int64(self.taps) + Int64(self.taps - 1 - lag)])
                acc = acc + _mul_rn(value, tap)
            out[row * Int64(self.width) + col] = (acc + Float32(x[row * Int64(self.width) + col])) * Float32(self.factor)

    @cute.kernel
    def commit(self, x: cute.Pointer, history: cute.Pointer, first: cute.Pointer,
               slots: cute.Pointer, rows: Int32):
        row = Int64(cute.arch.block_idx()[0])
        col = Int64(cute.arch.block_idx()[1]) * Int64(self.threads) + Int64(cute.arch.thread_idx()[0])
        if row < Int64(rows) and col < Int64(self.width):
            last = row + Int64(1) == Int64(rows)
            if row + Int64(1) < Int64(rows):
                last = Int64(first[row + Int64(1)]) != Int64(first[row])
            if last:
                begin, slot = Int64(first[row]), Int64(slots[row])
                # Descending lags preserve old inputs when a chunk is shorter
                # than the history: each shifted read precedes its overwrite.
                for rev in cutlass.range_constexpr(self.taps - 1):
                    lag = self.taps - 2 - rev
                    value = Float32(0)
                    if row - Int64(lag) >= begin:
                        value = Float32(x[(row - Int64(lag)) * Int64(self.width) + col])
                    else:
                        prior = Int64(lag) - (row - begin) - Int64(1)
                        value = Float32(history[(slot * Int64(self.taps - 1) + prior) * Int64(self.width) + col])
                    history[(slot * Int64(self.taps - 1) + Int64(lag)) * Int64(self.width) + col] = value


def compile_inkling2_convolution_aot(*, width: int, taps: int = 4, heads: int = 1,
                                    dynamic: bool = False, max_rows: int = 4096,
                                    dtype=torch.bfloat16):
    """Compile static geometry once; live row counts and segment ids are runtime.

    The unused dynamic pointer on the static route still needs a valid aligned
    device address (the static weights may be passed a second time).
    """
    if (type(width) is not int or type(taps) is not int or type(heads) is not int
            or width <= 0 or not 2 <= taps <= 8 or heads <= 0 or width % heads
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31 - 1
            or dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("invalid Inkling convolution geometry/capacity/dtype")
    kind = "dynamic" if dynamic else "static"
    operands = (
        Operand("x", dtype, f"[rows,{width}]"),
        Operand("weight", torch.bfloat16, f"[{taps},{width}]" if dynamic else f"[{width},{taps}]"),
        Operand("dynamic", dtype, f"[rows,{heads},{taps}]"),
        Operand("history", dtype, f"[slots,{taps-1},{width}]", "inout"),
        Operand("first", torch.int32, "[rows]", align=4),
        Operand("slots", torch.int64, "[rows]", align=8),
        Operand("out", dtype, f"[rows,{width}]", "out"),
    )
    return compile_program(
        _Convolution(width, taps, heads, dynamic), name=f"inkling2_{kind}_convolution",
        operands=operands, scalars=(Scalar("rows"),),
        key=(width, taps, heads, kind, dtype, max_rows),
        geometry={"width": width, "taps": taps, "heads": heads, "dynamic": bool(dynamic),
                  "history": "newest_first", "max_rows": max_rows},
        doc=__doc__)
