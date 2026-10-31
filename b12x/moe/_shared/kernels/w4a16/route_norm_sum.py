"""Top-k sum with a per-route output RMSNorm before the routing weight.

Muse Spark (transformers ``MuseSparkExperts``) normalizes every routed
expert's down-projection output with ``MuseSparkTextInputScaledRMSNorm``
before multiplying by its routing weight:

    y_e   = bf16(down_e(act(gate_up_e(x))))              # expert output, BF16
    s_e   = bf16(f32(y_e) * f32(bf16(1 + w)))            # scale first, round
                                                         # (1 + w is a BF16 add)
    n_e   = bf16(s_e * rsqrt(mean(s_e^2) + eps))         # then normalize, round
    out   = sum_e bf16(n_e * topk_weight_e)              # F32, router order

A norm over the whole expert output does not distribute over intermediate
slices, so the experts it serves are whole on each rank (expert parallel);
``expert_map`` maps a global id to the rank-local slot or -1 (skipped). A
rank's output is its owned routes' sum; the coordinator adds the ranks.
The weighted values round to BF16 as the reference's ``.to(bf16)`` before
its ``index_add_``; the sum itself accumulates in F32 in router order (the
eager reference accumulates in BF16 in expert-id order, a deterministic
difference within the fidelity bars, not a different model).

The fc2 rows are the mixed Trellis kernel's FP16 H128-rotated route outputs.
Each route is un-rotated exactly as ``W4A16TopKSumKernel`` full rotation
does (H128 then the expert's ``svh``), then rounded to BF16: the official
expert output dtype. Every multiply/add in the norm is explicit ``.rn`` PTX
(no FMA contraction), and the per-row sum of squares is reduced in a fixed
order (columns ascending per thread, butterfly per warp, warps ascending),
so ``reference`` reproduces it bit for bit on the host.

One CTA owns one token; its 256 threads hold the token's 4096 columns in
registers (16 each, four H128 slabs per warp), so no route values touch
shared memory except eight per-warp partial sums.
"""
from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import f16x2_to_f32x2, get_ptr_as_int64, ld_global_nc_v2_u32
from b12x._lib.program_cache import register_program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_THREADS = 256
_WARPS = _THREADS // 32


def _asm_f32(op: str, *args):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value() for a in args],
            op,
            "=f" + ",f" * len(args),
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return _asm_f32("mul.rn.f32 $0, $1, $2;", a, b)


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return _asm_f32("add.rn.f32 $0, $1, $2;", a, b)


@dsl_user_op
def _sub_rn(a, b, *, loc=None, ip=None):
    return _asm_f32("sub.rn.f32 $0, $1, $2;", a, b)


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    return _asm_f32("div.rn.f32 $0, $1, $2;", a, b)


@dsl_user_op
def _sqrt_rn(a, *, loc=None, ip=None):
    return _asm_f32("sqrt.rn.f32 $0, $1;", a)


@dsl_user_op
def _round_bf16(a, *, loc=None, ip=None):
    """F32 -> BF16 (RN) -> F32, as ``tensor.to(bfloat16).float()``."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value()],
            "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; mov.b32 $0, {0, h}; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class RouteNormTopKSumKernel:
    """Whole-expert top-k sum with per-route input-scaled RMSNorm (Muse Spark)."""

    def __init__(self, *, topk: int, hidden_size: int, eps: float, output_dtype: str = "fp32"):
        if hidden_size != 4096:
            # 256 threads x 16 columns; other widths need another register tiling.
            raise ValueError("route-norm top-k sum is tiled for hidden_size 4096")
        if not 1 <= topk <= 16:
            raise ValueError("route-norm top-k sum supports top-k 1..16")
        if output_dtype not in ("fp32", "bf16"):
            raise ValueError("route-norm top-k sum output must be fp32 or bf16")
        if not eps > 0:
            raise ValueError("route-norm epsilon must be positive")
        self.topk = int(topk)
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.output_bf16 = output_dtype == "bf16"

    @property
    def __cache_key__(self):
        return ("w4a16_route_norm_topk_sum", 1, self.topk, self.hidden_size, self.eps, self.output_bf16)

    @cute.jit
    def __call__(
        self,
        fc2_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        route_expert_ids_ptr: cute.Pointer,
        expert_map_ptr: cute.Pointer,
        svh_ptr: cute.Pointer,
        norm_weight_ptr: cute.Pointer,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        def flat(ptr, n):
            return cute.make_tensor(ptr, layout=cute.make_layout((n,), stride=(1,)))

        rows = active_m * Int32(self.topk)
        self.kernel(
            flat(fc2_ptr, rows * Int32(self.hidden_size)),
            flat(output_ptr, active_m * Int32(self.hidden_size)),
            flat(topk_weights_ptr, rows),
            flat(route_expert_ids_ptr, rows),
            flat(expert_map_ptr, Int64(route_num_experts) + Int64(1)),
            flat(svh_ptr, Int64(weight_num_experts) * Int64(self.hidden_size)),
            flat(norm_weight_ptr, Int32(self.hidden_size)),
            weight_num_experts,
            route_num_experts,
        ).launch(grid=(active_m, 1, 1), block=[_THREADS, 1, 1], stream=stream)

    @cute.jit
    def _had128_quad(self, v0, v1, v2, v3, lane):
        """``W4A16TopKSumKernel._had128_quad`` (same butterflies and scale)."""
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        rs = cutlass.Float32(0.088388347648)
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    @cute.kernel
    def kernel(
        self,
        fc2: cute.Tensor,
        output: cute.Tensor,
        topk_weights: cute.Tensor,
        route_expert_ids: cute.Tensor,
        expert_map: cute.Tensor,
        svh: cute.Tensor,
        norm_weight: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) & Int32(31)
        warp = Int32(tidx) >> Int32(5)
        token = Int32(token)
        partial_ptr = cute.arch.alloc_smem(cutlass.Float32, _WARPS)
        partials = cute.make_tensor(partial_ptr, cute.make_layout(_WARPS))
        # Warp w owns H128 slabs 4w..4w+3; lane l owns columns l*4..l*4+3 of each.
        acc = cute.make_rmem_tensor((16,), cutlass.Float32)
        scaled = cute.make_rmem_tensor((16,), cutlass.Float32)
        scale = cute.make_rmem_tensor((16,), cutlass.Float32)
        for slab in cutlass.range_constexpr(4):
            col = (warp * Int32(4) + Int32(slab)) * Int32(128) + lane * Int32(4)
            for j in cutlass.range_constexpr(4):
                acc[slab * 4 + j] = cutlass.Float32(0.0)
                # `1.0 + weight` on the BF16 parameter rounds to BF16.
                scale[slab * 4 + j] = _round_bf16(_add_rn(cutlass.Float32(1.0), norm_weight[col + Int32(j)].to(cutlass.Float32)))
        eps = cutlass.Float32(self.eps)
        hidden = cutlass.Float32(self.hidden_size)
        for route in cutlass.range_constexpr(self.topk):
            row = token * Int32(self.topk) + Int32(route)
            raw = route_expert_ids[row].to(Int32)
            expert = Int32(-1)
            if raw >= Int32(0) and raw < route_num_experts:
                expert = expert_map[raw].to(Int32)
            # Block-uniform branch: every thread reads the same route id, so
            # the barriers inside are reached by the whole CTA or by none.
            if expert >= Int32(0) and expert < weight_num_experts:
                square = cutlass.Float32(0.0)
                for slab in cutlass.range_constexpr(4):
                    col = (warp * Int32(4) + Int32(slab)) * Int32(128) + lane * Int32(4)
                    p01, p23 = ld_global_nc_v2_u32(get_ptr_as_int64(fc2, Int64(row) * Int64(self.hidden_size) + Int64(col)))
                    s01, s23 = ld_global_nc_v2_u32(get_ptr_as_int64(svh, Int64(expert) * Int64(self.hidden_size) + Int64(col)))
                    v0, v1 = f16x2_to_f32x2(p01)
                    v2, v3 = f16x2_to_f32x2(p23)
                    h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
                    c0, c1 = f16x2_to_f32x2(s01)
                    c2, c3 = f16x2_to_f32x2(s23)
                    for j, (h, c) in enumerate(((h0, c0), (h1, c1), (h2, c2), (h3, c3))):
                        y = _round_bf16(_mul_rn(h, c))
                        value = _round_bf16(_mul_rn(y, scale[slab * 4 + j]))
                        scaled[slab * 4 + j] = value
                        square = _add_rn(square, _mul_rn(value, value))
                for i in cutlass.range_constexpr(5):
                    square = _add_rn(square, cute.arch.shuffle_sync_bfly(square, offset=1 << i))
                if lane == Int32(0):
                    partials[warp] = square
                cute.arch.sync_threads()
                total = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(_WARPS):
                    total = _add_rn(total, partials[Int32(w)])
                cute.arch.sync_threads()
                inv = _div_rn(cutlass.Float32(1.0), _sqrt_rn(_add_rn(_div_rn(total, hidden), eps)))
                weight = topk_weights[row].to(cutlass.Float32)
                for k in cutlass.range_constexpr(16):
                    normed = _round_bf16(_mul_rn(scaled[k], inv))
                    acc[k] = _add_rn(acc[k], _round_bf16(_mul_rn(normed, weight)))
        for slab in cutlass.range_constexpr(4):
            col = (warp * Int32(4) + Int32(slab)) * Int32(128) + lane * Int32(4)
            base = token * Int32(self.hidden_size) + col
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.output_bf16):
                    output[base + Int32(j)] = cutlass.BFloat16(acc[slab * 4 + j])
                else:
                    output[base + Int32(j)] = acc[slab * 4 + j]


@dataclass(frozen=True)
class RouteNormTopKSumCompileResult:
    compiled: object
    topk: int
    hidden_size: int
    eps: float
    output_dtype: str
    route_ids_dtype: torch.dtype = torch.int32


_CACHE: dict[tuple, RouteNormTopKSumCompileResult] = {}
register_program_cache(_CACHE)


def compile_route_norm_topk_sum(*, topk: int, hidden_size: int, eps: float, output_dtype: str = "fp32",
                                route_ids_dtype: torch.dtype = torch.int32) -> RouteNormTopKSumCompileResult:
    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("top-k route expert ids must be int32 or int64")
    kernel = RouteNormTopKSumKernel(topk=topk, hidden_size=hidden_size, eps=eps, output_dtype=output_dtype)
    cache_key = (*kernel.__cache_key__, str(route_ids_dtype))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    route_cutlass = cutlass.Int32 if route_ids_dtype == torch.int32 else cutlass.Int64
    align = 4 if route_ids_dtype == torch.int32 else 8
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    compiled = b12x_compile(
        kernel,
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16 if output_dtype == "bf16" else cutlass.Float32, 16,
                 cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(route_cutlass, align, cute.AddressSpace.gmem, assumed_align=align),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1), Int32(1), 1, current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("moe.w4a16.route_norm_topk_sum", 1, cache_key),
    )
    result = RouteNormTopKSumCompileResult(compiled=compiled, topk=topk, hidden_size=hidden_size, eps=eps,
                                           output_dtype=output_dtype, route_ids_dtype=route_ids_dtype)
    attach_programs(result, compiled)
    _CACHE[cache_key] = result
    return result


def reference(fc2: torch.Tensor, topk_weights: torch.Tensor, route_ids: torch.Tensor, expert_map: torch.Tensor,
              svh: torch.Tensor, norm_weight: torch.Tensor, eps: float, output_dtype: str = "fp32") -> torch.Tensor:
    """Host model of the kernel's exact arithmetic (CPU float32, no FMA).

    fc2 [M*K, H] FP16 rotated route outputs; svh [E_local, H] FP16; norm_weight
    [H] BF16; expert_map [E_global] int32 (local slot or -1).
    """
    import numpy as np
    m_k, hidden = fc2.shape
    topk = topk_weights.shape[-1]
    m = m_k // topk
    def bf16(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()
    # H128 transform exactly as the warp butterflies: 4-point then 5 lane stages.
    def had128(block):
        v = block.reshape(32, 4).astype(np.float32)
        s0, d0, s1, d1 = v[:, 0] + v[:, 1], v[:, 0] - v[:, 1], v[:, 2] + v[:, 3], v[:, 2] - v[:, 3]
        h = np.stack([s0 + s1, d0 + d1, s0 - s1, d0 - d1], axis=1).astype(np.float32)
        lanes = np.arange(32)
        for i in range(5):
            st = 1 << i
            partner = h[lanes ^ st]
            upper = (lanes & st) != 0
            h = np.where(upper[:, None], partner - h, partner + h).astype(np.float32)
        return (h * np.float32(0.088388347648)).astype(np.float32).reshape(128)
    rotated = fc2.float().numpy()
    weights = topk_weights.float().reshape(m, topk).numpy()
    ids = route_ids.reshape(m, topk).cpu().numpy()
    emap = expert_map.cpu().numpy()
    svh_np = svh.float().numpy()
    scale = (1.0 + norm_weight).float().numpy().astype(np.float32)  # BF16 add, as the module
    out = np.zeros((m, hidden), dtype=np.float32)
    for t in range(m):
        acc = np.zeros(hidden, dtype=np.float32)
        for r in range(topk):
            raw = int(ids[t, r])
            expert = int(emap[raw]) if 0 <= raw < emap.shape[0] else -1
            if expert < 0 or expert >= svh_np.shape[0]:
                continue
            row = rotated[t * topk + r]
            y = np.concatenate([had128(row[b * 128:(b + 1) * 128]) for b in range(hidden // 128)])
            y = bf16(y * svh_np[expert])
            acc = (acc + route_norm(y, scale, eps, np.float32(weights[t, r]))).astype(np.float32)
        out[t] = acc
    result = torch.from_numpy(out)
    return result.to(torch.bfloat16) if output_dtype == "bf16" else result


def route_norm(y, scale, eps, weight):
    """One route's BF16 expert output -> bf16(norm(y) * weight), the kernel's order."""
    import numpy as np
    def bf16(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()
    hidden = y.shape[0]
    s = bf16(y * scale)
    # Per-thread sums (4 slabs x 4 cols, ascending), butterfly per warp, warps ascending.
    thread_sum = np.zeros((_WARPS, 32), dtype=np.float32)
    for w in range(_WARPS):
        for lane in range(32):
            total = np.float32(0.0)
            for slab in range(4):
                base = (w * 4 + slab) * 128 + lane * 4
                for j in range(4):
                    total = np.float32(total + np.float32(s[base + j] * s[base + j]))
            thread_sum[w, lane] = total
    for i in range(5):
        partner = thread_sum[:, np.arange(32) ^ (1 << i)]
        thread_sum = (thread_sum + partner).astype(np.float32)
    total = np.float32(0.0)
    for w in range(_WARPS):
        total = np.float32(total + thread_sum[w, 0])
    inv = np.float32(np.float32(1.0) / np.float32(np.sqrt(np.float32(np.float32(total / np.float32(hidden)) + np.float32(eps)))))
    n = bf16(s * inv)
    return bf16(n * weight)
