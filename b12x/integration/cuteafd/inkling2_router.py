"""Inkling sigmoid+bias top-k with unbiased log-sigmoid softmax weights.

Selection is stable (smaller expert id wins ties), and selection bias never
enters the weights. Weights use log-sigmoid's stable -softplus(-logit), not a
softmax of raw logits. The expert kernels consume these weights BEFORE the
down activation QDQ. Also exports pure attention sigmoid-gate multiplication
and standalone interleaved SiTU with uncapped-gate sigmoid, useful as native
producer/oracle boundaries. No allocation, scratch or host work at launch.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64
import torch

from b12x._lib.intrinsics import div_rn_f32, fabs_f32, fmax_f32, fmin_f32
from ._common import Operand, Scalar, compile_program
from ._glm_kernels import _reduction_storage


class _Router:
    threads = 128

    def __init__(self, experts, topk, route_scale):
        self.experts, self.topk, self.route_scale = experts, topk, float(route_scale)

    @cute.jit
    def __call__(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer,
                 weights: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(logits, bias, ids, weights).launch(grid=(rows, 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Pointer, bias: cute.Pointer, ids: cute.Pointer, weights: cute.Pointer):
        row = Int64(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        slots = (self.experts + self.threads - 1) // self.threads
        scores = cute.make_rmem_tensor((slots,), Float32)
        raw = cute.make_rmem_tensor((slots,), Float32)
        selected = cute.make_rmem_tensor((self.topk,), Float32)
        storage = utils.SmemAllocator().allocate(_reduction_storage(2, 4))
        sm = storage.sums.get_tensor(cute.make_layout((2, 4), stride=(4, 1)))
        for j in cutlass.range_constexpr(slots):
            expert = tid + Int32(j*self.threads)
            score, value = Float32(-float("inf")), Float32(0)
            if expert < Int32(self.experts):
                value = Float32(logits[row*Int64(self.experts)+Int64(expert)])
                score = div_rn_f32(Float32(1), Float32(1)+cute.math.exp(-value, fastmath=False)) + Float32(bias[expert])
            scores[j], raw[j] = score, value
        maximum = Float32(-float("inf"))
        for choice in cutlass.range_constexpr(self.topk):
            best = Float32(-float("inf"))
            index = Int32(2147483647)
            for j in cutlass.range_constexpr(slots):
                candidate = tid + Int32(j*self.threads)
                if scores[j] > best or (scores[j] == best and candidate < index):
                    best, index = scores[j], candidate
            for shift in cutlass.range_constexpr(5):
                other = Float32(cute.arch.shuffle_sync_bfly(best, offset=1<<shift))
                other_id = Int32(cute.arch.shuffle_sync_bfly(index, offset=1<<shift))
                if other > best or (other == best and other_id < index):
                    best, index = other, other_id
            if tid % Int32(32) == Int32(0):
                sm[0, tid//Int32(32)], sm[1, tid//Int32(32)] = best, index.to(Float32)
            cute.arch.sync_threads()
            best, index = sm[0, 0], sm[1, 0].to(Int32)
            for w in cutlass.range_constexpr(1, 4):
                other, other_id = sm[0, w], sm[1, w].to(Int32)
                if other > best or (other == best and other_id < index):
                    best, index = other, other_id
            cute.arch.sync_threads()
            value = Float32(0)
            for j in cutlass.range_constexpr(slots):
                if tid + Int32(j*self.threads) == index:
                    value = raw[j]
                    scores[j] = Float32(-float("inf"))
            # The selected owner broadcasts its original unbiased logit.
            for shift in cutlass.range_constexpr(5):
                value = value + Float32(cute.arch.shuffle_sync_bfly(value, offset=1<<shift))
            if tid % Int32(32) == Int32(0):
                sm[0, tid//Int32(32)] = value
            cute.arch.sync_threads()
            value = sm[0, 0] + sm[0, 1] + sm[0, 2] + sm[0, 3]
            cute.arch.sync_threads()
            logsigmoid = fmin_f32(value, Float32(0)) - cute.math.log(
                Float32(1) + cute.math.exp(-fabs_f32(value), fastmath=False), fastmath=False)
            selected[choice] = logsigmoid
            maximum = fmax_f32(maximum, logsigmoid)
            if tid == Int32(0):
                ids[row*Int64(self.topk)+Int64(choice)] = index
        denominator = Float32(0)
        for choice in cutlass.range_constexpr(self.topk):
            selected[choice] = cute.math.exp(selected[choice]-maximum, fastmath=False)
            denominator = denominator + selected[choice]
        if tid == Int32(0):
            for choice in cutlass.range_constexpr(self.topk):
                weights[row*Int64(self.topk)+Int64(choice)] = div_rn_f32(selected[choice], denominator)*Float32(self.route_scale)


class _Gate:
    threads = 256

    def __init__(self, width, situ, gate_cap, up_cap):
        self.width, self.situ = width, situ
        self.gate_cap, self.up_cap = float(gate_cap), float(up_cap)

    @cute.jit
    def __call__(self, x: cute.Pointer, gate: cute.Pointer, gamma: cute.Pointer,
                 out: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.kernel(x, gate, gamma, out, rows).launch(
            grid=((rows*self.width+self.threads-1)//self.threads, 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, gate: cute.Pointer, gamma: cute.Pointer,
               out: cute.Pointer, rows: Int32):
        at = Int64(cute.arch.block_idx()[0])*Int64(self.threads)+Int64(cute.arch.thread_idx()[0])
        if at < Int64(rows)*Int64(self.width):
            if cutlass.const_expr(self.situ):
                g, u = Float32(x[at*Int64(2)]), Float32(x[at*Int64(2)+Int64(1)])
                sigmoid = div_rn_f32(Float32(1), Float32(1)+cute.math.exp(-g, fastmath=False))
                capped_gate = Float32(self.gate_cap)*cute.math.tanh(div_rn_f32(g, Float32(self.gate_cap)), fastmath=False)
                capped_up = Float32(self.up_cap)*cute.math.tanh(div_rn_f32(u, Float32(self.up_cap)), fastmath=False)
                out[at] = ((capped_gate*sigmoid)*capped_up)*Float32(gamma[at//Int64(self.width)])
            else:
                g = Float32(gate[at])
                out[at] = Float32(x[at])*div_rn_f32(Float32(1), Float32(1)+cute.math.exp(-g, fastmath=False))


def compile_inkling2_router_aot(*, experts=384, topk=8, route_scale=1., max_rows=4096):
    import math
    if (type(experts) is not int or not 1 <= experts <= 4096 or type(topk) is not int
            or not 1 <= topk <= min(experts, 16) or not math.isfinite(route_scale) or route_scale <= 0
            or type(max_rows) is not int or not 1 <= max_rows <= 2**31-1):
        raise ValueError("invalid Inkling routing geometry")
    return compile_program(_Router(experts, topk, route_scale), name="inkling2_router",
        operands=(Operand("logits", torch.float32, f"[rows,{experts}]"),
                  Operand("bias", torch.float32, f"[{experts}]"),
                  Operand("ids", torch.int32, f"[rows,{topk}]", "out"),
                  Operand("weights", torch.float32, f"[rows,{topk}]", "out")),
        scalars=(Scalar("rows"),), key=(experts, topk, route_scale, max_rows),
        geometry=dict(experts=experts, topk=topk, route_scale=route_scale, max_rows=max_rows), doc=__doc__)


def compile_inkling2_gate_aot(*, width, situ=False, gate_cap=4., up_cap=25., max_rows=4096,
                             dtype=torch.bfloat16):
    import math
    if (type(width) is not int or width <= 0 or not math.isfinite(gate_cap) or gate_cap <= 0
            or not math.isfinite(up_cap) or up_cap <= 0 or type(max_rows) is not int
            or not 1 <= max_rows <= (2**31-1)//width or dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("invalid Inkling gate geometry")
    return compile_program(_Gate(width, situ, gate_cap, up_cap), name="inkling2_situ" if situ else "inkling2_gate",
        operands=(Operand("x", dtype, f"[rows,{width*2 if situ else width}]"),
                  Operand("gate", dtype, f"[rows,{width}]"), Operand("gamma", torch.float32, "[rows]"),
                  Operand("out", dtype, f"[rows,{width}]", "out")),
        scalars=(Scalar("rows"),), key=(width, situ, gate_cap, up_cap, max_rows, dtype),
        geometry=dict(width=width, situ=situ, gate_cap=gate_cap, up_cap=up_cap, max_rows=max_rows), doc=__doc__)
