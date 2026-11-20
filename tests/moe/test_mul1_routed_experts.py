"""Bitwise reference oracles for MUL1 (0x83DCD12D) and unchanged MCG (0xCBAC1FED) routed experts.

Synthetic fixtures use GPTQModel ``exl3_lib``'s exact tensor contract:
  - ``.trellis``: ``int16[K // 16, N // 16, 16 * bits]``
  - ``.suh``: ``float16[K]``
  - ``.svh``: ``float16[N]``
  - ``.mul1``: ``int32`` scalar ``0x83DCD12D`` (or ``.mcg`` ``0xCBAC1FED``)

Weight reconstruction runs exclusively on CPU through the kernel-independent
``tests._reference.native_trellis.Reference`` MSB-first tail-biting decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32
import numpy as np
import pytest
import torch

from b12x._lib.intrinsics import (
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    pack_f32x2_to_f16x2,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _cutlass_element_dtype,
    compile_w4a16_fused_moe,
    compile_w4a16_gemm,
    compile_w4a16_topk_sum,
    pack_topk_routes_by_expert,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    MixedTrellisRotations,
    _descriptor_bits,
    _mixed_trellis_tier_pointers,
    bind_mixed_trellis,
    bind_mixed_trellis3,
    build_compact_trellis_descriptor,
    build_projection_tiered_maps,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    compile_mixed_trellis3,
    make_mixed_trellis3_buffers,
    make_mixed_trellis_buffers,
    make_mixed_trellis_segment_buffers,
    run_bound_mixed_trellis,
    run_bound_mixed_trellis3,
    warmup_mixed_trellis_route_pack,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis4 import compile_mixed_trellis4
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)
from b12x.moe._shared.kernels.w4a16.route_pack import (
    compile_w4a16_route_pack_launches,
)
import b12x.moe._shared.trellis_codebooks as _trellis_codebooks
from tests._reference.native_trellis import Reference

MCG_MULTIPLIER = _trellis_codebooks.MCG_MULTIPLIER
MUL1_MULTIPLIER = getattr(_trellis_codebooks, "MUL1_MULTIPLIER", 0x83DCD12D)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


@dataclass(frozen=True)
class SyntheticExl3Projection:
    """One projection tensor bundle packed identically to GPTQModel's exl3_lib."""

    trellis: torch.Tensor  # int16 [K // 16, N // 16, 16 * bits]
    suh: torch.Tensor  # float16 [K]
    svh: torch.Tensor  # float16 [N]
    marker_name: str  # "mul1" or "mcg"
    marker: torch.Tensor  # int32 scalar (0x83DCD12D or 0xCBAC1FED)


def make_gptqmodel_exl3_projection(
    in_features: int,
    out_features: int,
    bits: int,
    *,
    codebook: str,
    seed: int,
    device: torch.device,
) -> SyntheticExl3Projection:
    rng = np.random.default_rng(seed)
    trellis_np = rng.integers(
        -32768,
        32768,
        size=(in_features // 16, out_features // 16, 16 * bits),
        dtype=np.int16,
    )
    suh_np = (0.75 + 0.5 * rng.random(in_features)) * rng.choice(
        np.array([-1.0, 1.0]), size=in_features
    )
    svh_np = (0.75 + 0.5 * rng.random(out_features)) * rng.choice(
        np.array([-1.0, 1.0]), size=out_features
    )
    marker_val = MUL1_MULTIPLIER if codebook == "mul1" else MCG_MULTIPLIER
    return SyntheticExl3Projection(
        trellis=torch.from_numpy(trellis_np).to(device=device),
        suh=torch.from_numpy(suh_np.astype(np.float16)).to(device=device),
        svh=torch.from_numpy(svh_np.astype(np.float16)).to(device=device),
        marker_name=codebook,
        marker=torch.tensor(np.int32(np.uint32(marker_val)), dtype=torch.int32, device=device),
    )


def decode_exl3_projection_cpu(proj: SyntheticExl3Projection, bits: int) -> torch.Tensor:
    """Independent CPU decode of a ``[K/16, N/16, 16*bits]`` int16 ``.trellis`` tensor to FP16 ``[K, N]``."""
    raw = proj.trellis.detach().cpu().contiguous().numpy()
    k16, n16, width = raw.shape
    assert width == 16 * bits
    ref = Reference(proj.marker_name)
    tiles = ref.decode_tiles(raw.tobytes(), bits)  # float32 [k16 * n16, 16, 16]
    dense = (
        tiles.astype(np.float16)
        .reshape(k16, n16, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(k16 * 16, n16 * 16)
        .copy()
    )
    return torch.from_numpy(dense)


_REF_GEMM_CACHE: dict[tuple[int, int], object] = {}
_REF_HAD_CACHE: dict[int, object] = {}


@cute.jit
def _pack_half2_u32(v0_f16: cute.Tensor, idx0: Int32, idx1: Int32) -> Uint32:
    return pack_f32x2_to_f16x2(
        v0_f16[idx0].to(Float32),
        v0_f16[idx1].to(Float32),
    )


def _run_exact_fp16_mma_gemm_2chain(
    a_f16: torch.Tensor,
    b_f16: torch.Tensor,
    *,
    tile_k: int = 64,
) -> torch.Tensor:
    """Standalone dense FP16 reference GEMM using ``m16n8k16`` (no trellis code).

    Matches ``W4A16GemmKernel``'s ``m8`` 2-chain (``tile_k=64``) or 4-chain
    (``tile_k=128``) warp-row accumulation order bit-for-bit on dense FP16 inputs.
    """
    assert a_f16.dtype == torch.float16 and b_f16.dtype == torch.float16
    m, k = a_f16.shape
    k_b, n = b_f16.shape
    assert k == k_b and k % tile_k == 0 and n % 16 == 0
    chains = tile_k // 32
    assert chains in (2, 4)

    a_pad_rows = ((max(m, 1) + 7) // 8) * 8
    if a_pad_rows != m:
        a_work = torch.zeros((a_pad_rows, k), dtype=torch.float16, device=a_f16.device)
        a_work[:m].copy_(a_f16)
    else:
        a_work = a_f16.contiguous()
    b_work = b_f16.contiguous()
    c_work = torch.empty((a_pad_rows, n), dtype=torch.float16, device=a_f16.device)

    a_flat = a_work.reshape(-1)
    b_flat = b_work.reshape(-1)
    c_flat = c_work.reshape(-1)

    key = (k, n, chains, a_pad_rows)
    if key not in _REF_GEMM_CACHE:
        k_iters = k // tile_k

        @cute.kernel
        def _gemm_kernel(a_t: cute.Tensor, b_t: cute.Tensor, c_t: cute.Tensor):
            lane, _, _ = cute.arch.thread_idx()
            m_blk, n_tile, _ = cute.arch.block_idx()
            r = Int32(2) * (lane % Int32(4))
            c = lane // Int32(4)
            m_row = m_blk * Int32(8) + c
            n_col0 = n_tile * Int32(16) + c
            n_col1 = n_col0 + Int32(8)

            acc = cute.make_rmem_tensor((chains, 4), Float32)
            for ch in cutlass.range_constexpr(chains):
                for j in cutlass.range_constexpr(4):
                    acc[ch, j] = Float32(0.0)

            for kt in range(k_iters):
                for ch in cutlass.range_constexpr(chains):
                    for sub in cutlass.range_constexpr(2):
                        k_base = Int32((kt * tile_k) + (ch * 32) + (sub * 16))
                        k0 = k_base + r
                        k1 = k0 + Int32(1)
                        k2 = k0 + Int32(8)
                        k3 = k0 + Int32(9)
                        a0 = _pack_half2_u32(
                            a_t,
                            m_row * Int32(k) + k0,
                            m_row * Int32(k) + k1,
                        )
                        a1 = _pack_half2_u32(
                            a_t,
                            m_row * Int32(k) + k2,
                            m_row * Int32(k) + k3,
                        )
                        o0 = _pack_half2_u32(
                            b_t,
                            k0 * Int32(n) + n_col0,
                            k1 * Int32(n) + n_col0,
                        )
                        o1 = _pack_half2_u32(
                            b_t,
                            k2 * Int32(n) + n_col0,
                            k3 * Int32(n) + n_col0,
                        )
                        o2 = _pack_half2_u32(
                            b_t,
                            k0 * Int32(n) + n_col1,
                            k1 * Int32(n) + n_col1,
                        )
                        o3 = _pack_half2_u32(
                            b_t,
                            k2 * Int32(n) + n_col1,
                            k3 * Int32(n) + n_col1,
                        )
                        d0, d1, d2, d3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                            acc[ch, 0],
                            acc[ch, 1],
                            acc[ch, 2],
                            acc[ch, 3],
                            o0,
                            o2,
                            o1,
                            o3,
                            a0,
                            a1,
                        )
                        acc[ch, 0] = d0
                        acc[ch, 1] = d1
                        acc[ch, 2] = d2
                        acc[ch, 3] = d3

            if cutlass.const_expr(chains == 4):
                s0 = acc[0, 0] + ((acc[1, 0] + acc[3, 0]) + acc[2, 0])
                s1 = acc[0, 1] + ((acc[1, 1] + acc[3, 1]) + acc[2, 1])
                s2 = acc[0, 2] + ((acc[1, 2] + acc[3, 2]) + acc[2, 2])
                s3 = acc[0, 3] + ((acc[1, 3] + acc[3, 3]) + acc[2, 3])
            else:
                s0 = acc[0, 0] + acc[1, 0]
                s1 = acc[0, 1] + acc[1, 1]
                s2 = acc[0, 2] + acc[1, 2]
                s3 = acc[0, 3] + acc[1, 3]

            out_r0 = m_blk * Int32(8) + Int32(2) * (lane % Int32(4))
            out_r1 = out_r0 + Int32(1)
            c_t[out_r0 * Int32(n) + n_col0] = s0.to(cutlass.Float16)
            c_t[out_r1 * Int32(n) + n_col0] = s1.to(cutlass.Float16)
            c_t[out_r0 * Int32(n) + n_col1] = s2.to(cutlass.Float16)
            c_t[out_r1 * Int32(n) + n_col1] = s3.to(cutlass.Float16)

        @cute.jit
        def _launch_gemm(a_t: cute.Tensor, b_t: cute.Tensor, c_t: cute.Tensor, stream):
            _gemm_kernel(a_t, b_t, c_t).launch(
                grid=(a_pad_rows // 8, n // 16, 1),
                block=(32, 1, 1),
                stream=stream,
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_GEMM_CACHE[key] = cute.compile(
            _launch_gemm,
            from_dlpack(a_flat),
            from_dlpack(b_flat),
            from_dlpack(c_flat),
            stream,
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _REF_GEMM_CACHE[key](
        from_dlpack(a_flat),
        from_dlpack(b_flat),
        from_dlpack(c_flat),
        stream,
    )
    return c_work[:m]


def _run_exact_had128_f32(x_f32: torch.Tensor) -> torch.Tensor:
    """Exact warp-butterfly H128 matching ``had_hf_r_128_inner`` (``scale = 0.088388347648``)."""
    orig_shape = x_f32.shape
    assert orig_shape[-1] % 128 == 0
    blocks = x_f32.contiguous().reshape(-1, 128)
    num_blocks = int(blocks.shape[0])
    out = torch.empty_like(blocks)
    in_flat = blocks.reshape(-1)
    out_flat = out.reshape(-1)

    if num_blocks not in _REF_HAD_CACHE:
        @cute.kernel
        def _had_kernel(src: cute.Tensor, dst: cute.Tensor):
            lane, _, _ = cute.arch.thread_idx()
            blk, _, _ = cute.arch.block_idx()
            base = blk * Int32(128) + lane * Int32(4)
            v0 = src[base + Int32(0)].to(Float32)
            v1 = src[base + Int32(1)].to(Float32)
            v2 = src[base + Int32(2)].to(Float32)
            v3 = src[base + Int32(3)].to(Float32)
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
            rs = Float32(0.088388347648)
            dst[base + Int32(0)] = h0 * rs
            dst[base + Int32(1)] = h1 * rs
            dst[base + Int32(2)] = h2 * rs
            dst[base + Int32(3)] = h3 * rs

        @cute.jit
        def _launch_had(src: cute.Tensor, dst: cute.Tensor, stream):
            _had_kernel(src, dst).launch(
                grid=(num_blocks, 1, 1),
                block=(32, 1, 1),
                stream=stream,
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_HAD_CACHE[num_blocks] = cute.compile(
            _launch_had,
            from_dlpack(in_flat),
            from_dlpack(out_flat),
            stream,
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _REF_HAD_CACHE[num_blocks](from_dlpack(in_flat), from_dlpack(out_flat), stream)
    return out.reshape(orig_shape)


_REF_INTER_ACT_CACHE: dict[tuple[str, float | None, bool, int, int], object] = {}
_REF_FC2_ROT_CACHE: dict[tuple[int, int], object] = {}


@cute.jit
def _had128_quad_ref(
    v0: Float32, v1: Float32, v2: Float32, v3: Float32, lane: Int32
):
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
    rs = Float32(0.088388347648)
    return h0 * rs, h1 * rs, h2 * rs, h3 * rs


def _run_exact_inter_rot_act_f16(
    fc1_g_f16: torch.Tensor,
    fc1_u_f16: torch.Tensor,
    rot_row_f16: torch.Tensor,
    *,
    activation: str,
    swiglu_limit: float | None,
    fast_math: bool = True,
    output_f32: bool = False,
) -> torch.Tensor:
    """Single-kernel post-FC1 ``H128 -> * svg/svu -> gated_act -> * sd -> H128 -> fp16/fp32``."""
    rows, intermediate = fc1_g_f16.shape
    assert fc1_u_f16.shape == (rows, intermediate)
    assert rot_row_f16.shape == (3 * intermediate,)
    assert intermediate % 128 == 0
    nblk = intermediate // 128
    total_units = rows * nblk
    out_tensor = torch.empty(
        (rows, intermediate),
        dtype=torch.float32 if output_f32 else torch.float16,
        device=fc1_g_f16.device,
    )
    g_flat = fc1_g_f16.contiguous().reshape(-1)
    u_flat = fc1_u_f16.contiguous().reshape(-1)
    r_flat = rot_row_f16.contiguous().reshape(-1)
    o_flat = out_tensor.reshape(-1)

    key = (activation, swiglu_limit, bool(fast_math), bool(output_f32), rows, intermediate)
    if key not in _REF_INTER_ACT_CACHE:
        is_situ = activation == "situ"
        is_post_clamp = activation == "silu_post_clamp"
        has_limit = swiglu_limit is not None
        limit_val = float(swiglu_limit or 0.0)
        use_fast = bool(fast_math)

        @cute.jit
        def _eval_one(ig: Float32, iu: Float32, sd: Float32) -> Float32:
            if cutlass.const_expr(has_limit):
                lim = Float32(limit_val)
                neg_lim = Float32(-limit_val)
                if cutlass.const_expr(not is_post_clamp):
                    if ig > lim:
                        ig = lim
                if iu < neg_lim:
                    iu = neg_lim
                if iu > lim:
                    iu = lim
            if cutlass.const_expr(use_fast):
                e = cute.math.exp(-ig, fastmath=True)
            else:
                e = cute.math.exp(-ig, fastmath=False)
            sig = Float32(1.0) / (Float32(1.0) + e)
            if cutlass.const_expr(is_situ):
                beta = Float32(SITU_DEFAULT_BETA)
                lbeta = Float32(SITU_DEFAULT_LINEAR_BETA)
                return (
                    beta
                    * cute.math.tanh(ig / beta, fastmath=use_fast)
                    * sig
                    * lbeta
                    * cute.math.tanh(iu / lbeta, fastmath=use_fast)
                    * sd
                )
            silu = ig * sig
            if cutlass.const_expr(is_post_clamp and has_limit):
                lim = Float32(limit_val)
                if silu > lim:
                    silu = lim
            return silu * iu * sd

        @cute.kernel
        def _inter_act_kernel(
            g_t: cute.Tensor, u_t: cute.Tensor, r_t: cute.Tensor, o_t: cute.Tensor
        ):
            lane, _, _ = cute.arch.thread_idx()
            unit, _, _ = cute.arch.block_idx()
            row = unit // Int32(nblk)
            blk = unit - row * Int32(nblk)
            col0 = blk * Int32(128) + lane * Int32(4)
            base = row * Int32(intermediate) + col0
            g0 = g_t[base + Int32(0)].to(Float32)
            g1 = g_t[base + Int32(1)].to(Float32)
            g2 = g_t[base + Int32(2)].to(Float32)
            g3 = g_t[base + Int32(3)].to(Float32)
            u0 = u_t[base + Int32(0)].to(Float32)
            u1 = u_t[base + Int32(1)].to(Float32)
            u2 = u_t[base + Int32(2)].to(Float32)
            u3 = u_t[base + Int32(3)].to(Float32)
            gh0, gh1, gh2, gh3 = _had128_quad_ref(g0, g1, g2, g3, lane)
            uh0, uh1, uh2, uh3 = _had128_quad_ref(u0, u1, u2, u3, lane)
            isz = Int32(intermediate)
            down = isz + isz
            ig0 = gh0 * r_t[col0 + Int32(0)].to(Float32)
            ig1 = gh1 * r_t[col0 + Int32(1)].to(Float32)
            ig2 = gh2 * r_t[col0 + Int32(2)].to(Float32)
            ig3 = gh3 * r_t[col0 + Int32(3)].to(Float32)
            iu0 = uh0 * r_t[col0 + isz + Int32(0)].to(Float32)
            iu1 = uh1 * r_t[col0 + isz + Int32(1)].to(Float32)
            iu2 = uh2 * r_t[col0 + isz + Int32(2)].to(Float32)
            iu3 = uh3 * r_t[col0 + isz + Int32(3)].to(Float32)
            sd0 = r_t[col0 + down + Int32(0)].to(Float32)
            sd1 = r_t[col0 + down + Int32(1)].to(Float32)
            sd2 = r_t[col0 + down + Int32(2)].to(Float32)
            sd3 = r_t[col0 + down + Int32(3)].to(Float32)
            a0 = _eval_one(ig0, iu0, sd0)
            a1 = _eval_one(ig1, iu1, sd1)
            a2 = _eval_one(ig2, iu2, sd2)
            a3 = _eval_one(ig3, iu3, sd3)
            o0, o1, o2, o3 = _had128_quad_ref(a0, a1, a2, a3, lane)
            if cutlass.const_expr(output_f32):
                o_t[base + Int32(0)] = o0
                o_t[base + Int32(1)] = o1
                o_t[base + Int32(2)] = o2
                o_t[base + Int32(3)] = o3
            else:
                o_t[base + Int32(0)] = o0.to(cutlass.Float16)
                o_t[base + Int32(1)] = o1.to(cutlass.Float16)
                o_t[base + Int32(2)] = o2.to(cutlass.Float16)
                o_t[base + Int32(3)] = o3.to(cutlass.Float16)

        @cute.jit
        def _launch_inter_act(
            g_t: cute.Tensor, u_t: cute.Tensor, r_t: cute.Tensor, o_t: cute.Tensor, stream
        ):
            _inter_act_kernel(g_t, u_t, r_t, o_t).launch(
                grid=(total_units, 1, 1),
                block=(32, 1, 1),
                stream=stream,
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_INTER_ACT_CACHE[key] = cute.compile(
            _launch_inter_act,
            from_dlpack(g_flat),
            from_dlpack(u_flat),
            from_dlpack(r_flat),
            from_dlpack(o_flat),
            stream,
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _REF_INTER_ACT_CACHE[key](
        from_dlpack(g_flat),
        from_dlpack(u_flat),
        from_dlpack(r_flat),
        from_dlpack(o_flat),
        stream,
    )
    return out_tensor


def _run_exact_fc2_rot_f32(
    fc2_f16: torch.Tensor,
    svh_f16: torch.Tensor,
) -> torch.Tensor:
    """Single-kernel post-FC2 ``H128(fc2_f16) * svh_f16`` in Float32."""
    rows, hidden = fc2_f16.shape
    assert svh_f16.shape == (hidden,)
    assert hidden % 128 == 0
    nblk = hidden // 128
    total_units = rows * nblk
    out_f32 = torch.empty((rows, hidden), dtype=torch.float32, device=fc2_f16.device)
    in_flat = fc2_f16.contiguous().reshape(-1)
    s_flat = svh_f16.contiguous().reshape(-1)
    out_flat = out_f32.reshape(-1)
    key = (rows, hidden)
    if key not in _REF_FC2_ROT_CACHE:
        @cute.kernel
        def _fc2_rot_kernel(src: cute.Tensor, svh: cute.Tensor, dst: cute.Tensor):
            lane, _, _ = cute.arch.thread_idx()
            unit, _, _ = cute.arch.block_idx()
            row = unit // Int32(nblk)
            blk = unit - row * Int32(nblk)
            col0 = blk * Int32(128) + lane * Int32(4)
            base = row * Int32(hidden) + col0
            v0 = src[base + Int32(0)].to(Float32)
            v1 = src[base + Int32(1)].to(Float32)
            v2 = src[base + Int32(2)].to(Float32)
            v3 = src[base + Int32(3)].to(Float32)
            h0, h1, h2, h3 = _had128_quad_ref(v0, v1, v2, v3, lane)
            dst[base + Int32(0)] = h0 * svh[col0 + Int32(0)].to(Float32)
            dst[base + Int32(1)] = h1 * svh[col0 + Int32(1)].to(Float32)
            dst[base + Int32(2)] = h2 * svh[col0 + Int32(2)].to(Float32)
            dst[base + Int32(3)] = h3 * svh[col0 + Int32(3)].to(Float32)

        @cute.jit
        def _launch_fc2_rot(src: cute.Tensor, svh: cute.Tensor, dst: cute.Tensor, stream):
            _fc2_rot_kernel(src, svh, dst).launch(
                grid=(total_units, 1, 1),
                block=(32, 1, 1),
                stream=stream,
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_FC2_ROT_CACHE[key] = cute.compile(
            _launch_fc2_rot,
            from_dlpack(in_flat),
            from_dlpack(s_flat),
            from_dlpack(out_flat),
            stream,
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _REF_FC2_ROT_CACHE[key](
        from_dlpack(in_flat),
        from_dlpack(s_flat),
        from_dlpack(out_flat),
        stream,
    )
    return out_f32


def reference_exl3_routed_moe_bitwise(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_weights_f16: list[torch.Tensor],
    up_weights_f16: list[torch.Tensor],
    down_weights_f16: list[torch.Tensor],
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    output_dtype: torch.dtype = torch.float32,
    tile_k: int = 64,
) -> torch.Tensor:
    """Bitwise reference MoE over CPU-decoded ``Reference(codebook)`` FP16 expert weights."""
    device = x.device
    m, hidden = x.shape
    _, top_k = topk_ids.shape
    intermediate = int(gate_weights_f16[0].shape[1])

    route_outputs = torch.zeros((m, top_k, hidden), dtype=torch.float32, device=device)
    ids_cpu = topk_ids.detach().cpu().numpy()

    for expert in sorted({int(e) for e in ids_cpu.reshape(-1) if int(e) >= 0}):
        positions = np.argwhere(ids_cpu == expert)
        token_idx = torch.from_numpy(positions[:, 0]).to(device=device, dtype=torch.long)
        slot_idx = torch.from_numpy(positions[:, 1]).to(device=device, dtype=torch.long)

        x_e = x.index_select(0, token_idx).to(torch.float16).to(torch.float32)
        g_suh = gate_suh[0 if gate_suh.shape[0] == 1 else expert].to(torch.float32)
        u_suh = up_suh[0 if up_suh.shape[0] == 1 else expert].to(torch.float32)

        rot_g = _run_exact_had128_f32(
            (x_e * g_suh).to(torch.float16).to(torch.float32)
        ).to(torch.float16)
        rot_u = _run_exact_had128_f32(
            (x_e * u_suh).to(torch.float16).to(torch.float32)
        ).to(torch.float16)

        w_g = gate_weights_f16[expert].to(device=device)
        w_u = up_weights_f16[expert].to(device=device)
        w_d = down_weights_f16[expert].to(device=device)

        fc1_g = _run_exact_fp16_mma_gemm_2chain(rot_g, w_g, tile_k=tile_k)
        fc1_u = _run_exact_fp16_mma_gemm_2chain(rot_u, w_u, tile_k=tile_k)

        act_f16 = _run_exact_inter_rot_act_f16(
            fc1_g,
            fc1_u,
            intermediate_rotations[expert],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )

        fc2_f16 = _run_exact_fp16_mma_gemm_2chain(act_f16, w_d, tile_k=tile_k)
        d_svh_f16 = down_svh[0 if down_svh.shape[0] == 1 else expert]
        fc2_rot = _run_exact_fc2_rot_f32(fc2_f16, d_svh_f16)
        route_outputs[token_idx, slot_idx] = fc2_rot

    acc = torch.zeros((m, hidden), dtype=torch.float32, device=device)
    w_f32 = topk_weights.to(torch.float32)
    for slot in range(top_k):
        acc = torch.addcmul(acc, route_outputs[:, slot], w_f32[:, slot : slot + 1])
    return acc.to(output_dtype)


def assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, label: str = "") -> None:
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert bool(torch.isfinite(actual).all()), f"{label}: non-finite values in actual"
    assert int(actual.count_nonzero().item()) > 0, f"{label}: all-zero output"
    bit_dtype = torch.uint16 if actual.element_size() == 2 else torch.uint32
    a_bits = actual.contiguous().view(bit_dtype)
    e_bits = expected.contiguous().view(bit_dtype)
    if not torch.equal(a_bits, e_bits):
        mismatches = int((a_bits != e_bits).sum().item())
        max_abs = float((actual.float() - expected.float()).abs().max().item())
        raise AssertionError(
            f"{label}: {mismatches}/{actual.numel()} elements differ bitwise "
            f"(max_abs={max_abs:.6e})"
        )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_w4a16_gemm_trellis256_bitwise_matches_reference(codebook: str, bits: int) -> None:
    """Bitwise check of GPTQModel-packed ``.trellis`` through routed W4A16GemmKernel."""
    device = torch.device("cuda", torch.cuda.current_device())
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    m, k, n = 8, 256, 256
    proj = make_gptqmodel_exl3_projection(
        k, n, bits, codebook=codebook, seed=10_000 + bits * 17 + (1 if codebook == "mul1" else 0), device=device
    )
    decoded_b = decode_exl3_projection_cpu(proj, bits).to(device=device)

    rng = torch.Generator(device=device).manual_seed(20_000 + bits)
    x = (torch.randn((m, k), generator=rng, device=device) * 0.125).to(torch.float16)
    output = torch.empty((m, n), dtype=torch.float16, device=device)
    c_tmp = torch.empty(
        (
            packed_gemm_scratch_elements(
                size_n=n,
                route_slots=m,
                moe_block_size=8,
                sms=sms,
            ),
        ),
        dtype=torch.float32,
        device=device,
    )
    workspace = torch.zeros((max(sms * 4 * 64, n // 16),), dtype=torch.int32, device=device)
    dummy_scale = torch.zeros((4,), dtype=torch.int32, device=device)
    dummy_lut = torch.zeros((16,), dtype=torch.uint8, device=device)
    global_scale = torch.ones((4,), dtype=torch.float32, device=device)[:1]
    packed_routes = torch.arange(m, dtype=torch.int32, device=device)
    block_experts = torch.zeros((4,), dtype=torch.int32, device=device)[:1]
    route_count = torch.tensor([m], dtype=torch.int32, device=device)
    topk_weights = torch.ones((m,), dtype=torch.float32, device=device)

    compiled_gemm = compile_w4a16_gemm(
        size_m=m,
        size_n=n,
        size_k=k,
        num_experts=1,
        top_k=1,
        mul_topk_weights=False,
        tile_n=128,
        tile_k=64,
        moe_block_size=8,
        max_m_blocks=1,
        element_dtype="fp16",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="packed",
        trellis_bits=bits,
        trellis_codebook=codebook,
        dense_route_fast_path=False,
    )
    output.fill_(float("nan"))
    grid_x = min(sms * compiled_gemm.blocks_per_sm, max(n // 128, 1))
    compiled_gemm.compiled(
        make_ptr(cutlass.Float16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        proj.trellis.reshape(-1).view(torch.int32),
        make_ptr(cutlass.Float16, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        dummy_scale,
        global_scale,
        packed_routes,
        block_experts,
        route_count,
        topk_weights,
        c_tmp,
        workspace,
        dummy_lut,
        m,
        grid_x,
        current_cuda_stream(),
    )
    actual = output.clone()
    torch.cuda.synchronize(device)
    expected = _run_exact_fp16_mma_gemm_2chain(x, decoded_b, tile_k=64)
    assert_bitwise_equal(actual, expected, f"w4a16_gemm_{codebook}_k{bits}")


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    ("activation", "swiglu_limit", "out_dtype"),
    [
        ("silu", None, torch.bfloat16),
        ("silu", None, torch.float32),
        ("silu_post_clamp", 7.0, torch.bfloat16),
        ("situ", None, torch.bfloat16),
    ],
)
def test_w4a16_fused_moe_routed_experts_bitwise_matches_reference(
    codebook: str,
    bits: int,
    activation: str,
    swiglu_limit: float | None,
    out_dtype: torch.dtype,
) -> None:
    """Bitwise oracle for W4A16FusedMoeKernel across K2..K6, activations, and output dtypes."""
    if (activation, out_dtype) != ("silu", torch.bfloat16) and bits not in (2, 4, 6):
        pytest.skip("non-default activation/dtype spot-checked on K2, K4, K6")

    device = torch.device("cuda", torch.cuda.current_device())
    experts, top_k = 4, 2
    capacity_m = 8
    hidden, intermediate = 256, 256
    tile_config = (64, 128, 64, 128)

    gate_projs = [
        make_gptqmodel_exl3_projection(
            hidden, intermediate, bits, codebook=codebook, seed=30_000 + bits * 100 + e * 3, device=device
        )
        for e in range(experts)
    ]
    up_projs = [
        make_gptqmodel_exl3_projection(
            hidden, intermediate, bits, codebook=codebook, seed=30_000 + bits * 100 + e * 3 + 1, device=device
        )
        for e in range(experts)
    ]
    down_projs = [
        make_gptqmodel_exl3_projection(
            intermediate, hidden, bits, codebook=codebook, seed=30_000 + bits * 100 + e * 3 + 2, device=device
        )
        for e in range(experts)
    ]

    w13 = torch.stack(
        (
            torch.stack([p.trellis for p in gate_projs], dim=0),
            torch.stack([p.trellis for p in up_projs], dim=0),
        ),
        dim=0,
    ).contiguous()
    w2 = torch.stack([p.trellis for p in down_projs], dim=0).contiguous()
    gate_suh = torch.stack([p.suh for p in gate_projs], dim=0).contiguous()
    up_suh = torch.stack([p.suh for p in up_projs], dim=0).contiguous()
    intermediate_rotations = torch.cat(
        (
            torch.stack([p.svh for p in gate_projs], dim=0),
            torch.stack([p.svh for p in up_projs], dim=0),
            torch.stack([p.suh for p in down_projs], dim=0),
        ),
        dim=1,
    ).contiguous()
    down_svh = torch.stack([p.svh for p in down_projs], dim=0).contiguous()

    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation=activation,
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook=codebook,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        tile_config=tile_config,
    )
    assert prepared.trellis_codebook == codebook

    gate_w = [decode_exl3_projection_cpu(p, bits) for p in gate_projs]
    up_w = [decode_exl3_projection_cpu(p, bits) for p in up_projs]
    down_w = [decode_exl3_projection_cpu(p, bits) for p in down_projs]

    rng = torch.Generator(device="cpu").manual_seed(40_000 + bits * 13)
    x_full = (torch.randn((capacity_m, hidden), generator=rng) * 0.05).to(
        device=device, dtype=torch.bfloat16
    )
    ids_full = torch.stack(
        [torch.randperm(experts, generator=rng)[:top_k] for _ in range(capacity_m)]
    ).to(device=device, dtype=torch.int32)
    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    buffers = make_w4a16_packed_buffers(
        prepared,
        m=capacity_m,
        topk=top_k,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    out_buf = torch.empty((capacity_m, hidden), dtype=out_dtype, device=device)

    def run_live(live_m: int) -> torch.Tensor:
        return run_w4a16_moe(
            x_full[:live_m],
            prepared,
            weights_full[:live_m],
            ids_full[:live_m],
            activation=activation,
            swiglu_limit=swiglu_limit,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=out_buf[:live_m],
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            route_block_size_m=8,
            intermediate_rotation_scales=intermediate_rotations,
            full_rotation=True,
            suh_gate_table=gate_suh,
            suh_up_table=up_suh,
            svh_table=down_svh,
            rotation_a_gate=buffers.rotation_a_gate,
            rotation_a_up=buffers.rotation_a_up,
        )

    out_buf.fill_(float("nan"))
    eager_full = run_live(capacity_m).clone()
    torch.cuda.synchronize(device)

    expected_full = reference_exl3_routed_moe_bitwise(
        x_full,
        ids_full,
        weights_full,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
        activation=activation,
        swiglu_limit=swiglu_limit,
        output_dtype=out_dtype,
        tile_k=tile_config[0],
    )
    assert_bitwise_equal(
        eager_full,
        expected_full,
        f"fused_moe_{codebook}_k{bits}_{activation}_{out_dtype}",
    )

    # Precompiled capacity reuse across live M counts + CUDA graph capture/replay under frozen resolution.
    if bits == 4 and activation == "silu" and out_dtype == torch.bfloat16:
        props = torch.cuda.get_device_properties(device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(getattr(props, "shared_memory_per_block_optin", 101376))
        route_slots_cap = max_packed_route_slots(capacity_m * top_k, 8, experts)
        max_m_blocks = (route_slots_cap + 7) // 8
        fused_launch = compile_w4a16_fused_moe(
            size_m=capacity_m,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=experts,
            top_k=top_k,
            activation=activation,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=max_m_blocks,
            element_dtype="fp16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=bits,
            trellis_codebook=codebook,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
            force_tile_config=tile_config,
        )
        topk_sum_launch = compile_w4a16_topk_sum(
            m=capacity_m,
            topk=top_k,
            hidden_size=hidden,
            element_dtype="fp16",
            full_rotation=True,
            full_rotation_output_dtype="bf16",
            num_experts=experts,
        )
        route_pack_launches = compile_w4a16_route_pack_launches(
            tokens=capacity_m,
            topk=top_k,
            block_size=8,
            num_experts=experts,
            ordinal=int(device.index),
        )

        def run_preplanned(live_m: int) -> torch.Tensor:
            return run_w4a16_moe(
                x_full[:live_m],
                prepared,
                weights_full[:live_m],
                ids_full[:live_m],
                activation=activation,
                swiglu_limit=swiglu_limit,
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=buffers.intermediate_cache2,
                output=out_buf[:live_m],
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
                fused_launch=fused_launch,
                topk_sum_launch=topk_sum_launch,
                route_pack_launches=route_pack_launches,
                route_block_size_m=8,
                intermediate_rotation_scales=intermediate_rotations,
                full_rotation=True,
                suh_gate_table=gate_suh,
                suh_up_table=up_suh,
                svh_table=down_svh,
                rotation_a_gate=buffers.rotation_a_gate,
                rotation_a_up=buffers.rotation_a_up,
            )

        with kernel_resolution_guard("frozen W4A16 precompiled capacity"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_preplanned(capacity_m)
            for _ in range(2):
                out_buf.fill_(float("nan"))
                buffers.intermediate_cache13.fill_(float("nan"))
                buffers.intermediate_cache2.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert_bitwise_equal(out_buf, expected_full, f"graph_{codebook}_k4")

            x_saved = x_full.clone()
            w_saved = weights_full.clone()
            try:
                for live_m in (1, 3, capacity_m):
                    x_full.copy_(x_saved)
                    weights_full.copy_(w_saved)
                    if live_m < capacity_m:
                        x_full[live_m:].fill_(float("nan"))
                        weights_full[live_m:].fill_(float("nan"))
                    out_buf.fill_(float("nan"))
                    short = run_preplanned(live_m).clone()
                    torch.cuda.synchronize(device)
                    expected_live = reference_exl3_routed_moe_bitwise(
                        x_saved[:live_m],
                        ids_full[:live_m],
                        w_saved[:live_m],
                        gate_w,
                        up_w,
                        down_w,
                        gate_suh,
                        up_suh,
                        intermediate_rotations,
                        down_svh,
                        activation=activation,
                        swiglu_limit=swiglu_limit,
                        output_dtype=out_dtype,
                        tile_k=tile_config[0],
                    )
                    assert_bitwise_equal(short, expected_live, f"frozen_m{live_m}_{codebook}_k4")
            finally:
                x_full.copy_(x_saved)
                weights_full.copy_(w_saved)


def _build_projection_tiered_fixture(
    *,
    hidden: int,
    intermediate: int,
    tier_bits: tuple[int, ...],
    gate_tiers: tuple[int, ...],
    up_tiers: tuple[int, ...],
    down_tiers: tuple[int, ...],
    codebook: str,
    seed_base: int,
    tile_config: tuple[int, int, int, int],
    activation: str,
    device: torch.device,
):
    """Build projection-tiered mixed-Trellis prepared tiers + CPU-decoded per-expert FP16 weights."""
    num_experts = len(gate_tiers)
    num_tiers = len(tier_bits)
    slots_per_tier = num_experts // num_tiers
    assert num_experts == slots_per_tier * num_tiers

    gate_projs = [
        make_gptqmodel_exl3_projection(
            hidden,
            intermediate,
            tier_bits[gate_tiers[e]],
            codebook=codebook,
            seed=seed_base + e * 7 + 0,
            device=device,
        )
        for e in range(num_experts)
    ]
    up_projs = [
        make_gptqmodel_exl3_projection(
            hidden,
            intermediate,
            tier_bits[up_tiers[e]],
            codebook=codebook,
            seed=seed_base + e * 7 + 1,
            device=device,
        )
        for e in range(num_experts)
    ]
    down_projs = [
        make_gptqmodel_exl3_projection(
            intermediate,
            hidden,
            tier_bits[down_tiers[e]],
            codebook=codebook,
            seed=seed_base + e * 7 + 2,
            device=device,
        )
        for e in range(num_experts)
    ]

    gate_suh = torch.stack([p.suh for p in gate_projs], dim=0).contiguous()
    up_suh = torch.stack([p.suh for p in up_projs], dim=0).contiguous()
    inter_rot = torch.cat(
        (
            torch.stack([p.svh for p in gate_projs], dim=0),
            torch.stack([p.svh for p in up_projs], dim=0),
            torch.stack([p.suh for p in down_projs], dim=0),
        ),
        dim=1,
    ).contiguous()
    down_svh = torch.stack([p.svh for p in down_projs], dim=0).contiguous()

    prepared_tiers = []
    for t_idx, bits in enumerate(tier_bits):
        g_ids = [e for e in range(num_experts) if gate_tiers[e] == t_idx]
        u_ids = [e for e in range(num_experts) if up_tiers[e] == t_idx]
        d_ids = [e for e in range(num_experts) if down_tiers[e] == t_idx]
        assert len(g_ids) == slots_per_tier and len(u_ids) == slots_per_tier and len(d_ids) == slots_per_tier
        w13_t = torch.stack(
            (
                torch.stack([gate_projs[e].trellis for e in g_ids], dim=0),
                torch.stack([up_projs[e].trellis for e in u_ids], dim=0),
            ),
            dim=0,
        ).contiguous()
        w2_t = torch.stack([down_projs[e].trellis for e in d_ids], dim=0).contiguous()
        prepared_tiers.append(
            prepare_trellis256_moe_weights(
                w13_t,
                w2_t,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=slots_per_tier,
                activation=activation,
                fc1_tile_n=tile_config[1],
                fc2_tile_n=tile_config[3],
                params_dtype=torch.float16,
                w13_layout="trellis_t256_proj",
                trellis_bits=bits,
                codebook=codebook,
                gate_suh=gate_suh[:slots_per_tier].contiguous(),
                up_suh=up_suh[:slots_per_tier].contiguous(),
                intermediate_rotations=inter_rot[:slots_per_tier].contiguous(),
                down_svh=down_svh[:slots_per_tier].contiguous(),
                tile_config=tile_config,
            )
        )

    if num_tiers in (2, 3):
        mapping, descriptor = build_projection_tiered_maps(
            gate_tiers,
            up_tiers,
            down_tiers,
            tier_slots=(slots_per_tier,) * num_tiers,
            device=device,
        )
    else:
        desc_bits = _descriptor_bits(num_experts)
        rows: list[int] = []
        for proj_tiers in (gate_tiers, up_tiers, down_tiers):
            counters = [0] * num_tiers
            for tier_id in proj_tiers:
                local = counters[tier_id]
                counters[tier_id] += 1
                rows.append((tier_id << desc_bits) | local)
        mapping = torch.arange(num_experts, dtype=torch.int32, device=device)
        descriptor = torch.tensor(rows, dtype=torch.int32, device=device).contiguous()
    rotations = MixedTrellisRotations(
        intermediate=inter_rot,
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
    )
    gate_w = [decode_exl3_projection_cpu(gate_projs[e], tier_bits[gate_tiers[e]]) for e in range(num_experts)]
    up_w = [decode_exl3_projection_cpu(up_projs[e], tier_bits[up_tiers[e]]) for e in range(num_experts)]
    down_w = [decode_exl3_projection_cpu(down_projs[e], tier_bits[down_tiers[e]]) for e in range(num_experts)]
    return (
        tuple(prepared_tiers),
        mapping,
        descriptor,
        rotations,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
    )


def _build_disjoint_2tier_fixture(
    *,
    hidden: int,
    intermediate: int,
    bits: tuple[int, int],
    codebook: str,
    seed_base: int,
    tile_config: tuple[int, int, int, int],
    activation: str,
    device: torch.device,
):
    """Build disjoint 2-tier fixture with interleaved global IDs (tier0=(2,0), tier1=(3,1))."""
    tier_globals = ((2, 0), (3, 1))
    tiers = []
    gate_w_by_global: dict[int, torch.Tensor] = {}
    up_w_by_global: dict[int, torch.Tensor] = {}
    down_w_by_global: dict[int, torch.Tensor] = {}

    for t_idx, (t_bits, globals_t) in enumerate(zip(bits, tier_globals, strict=True)):
        g_projs = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, t_bits, codebook=codebook, seed=seed_base + t_idx * 50 + i * 3 + 0, device=device
            )
            for i in range(2)
        ]
        u_projs = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, t_bits, codebook=codebook, seed=seed_base + t_idx * 50 + i * 3 + 1, device=device
            )
            for i in range(2)
        ]
        d_projs = [
            make_gptqmodel_exl3_projection(
                intermediate, hidden, t_bits, codebook=codebook, seed=seed_base + t_idx * 50 + i * 3 + 2, device=device
            )
            for i in range(2)
        ]
        w13_t = torch.stack(
            (
                torch.stack([p.trellis for p in g_projs], dim=0),
                torch.stack([p.trellis for p in u_projs], dim=0),
            ),
            dim=0,
        ).contiguous()
        w2_t = torch.stack([p.trellis for p in d_projs], dim=0).contiguous()
        g_suh_t = torch.stack([p.suh for p in g_projs], dim=0).contiguous()
        u_suh_t = torch.stack([p.suh for p in u_projs], dim=0).contiguous()
        inter_t = torch.cat(
            (
                torch.stack([p.svh for p in g_projs], dim=0),
                torch.stack([p.svh for p in u_projs], dim=0),
                torch.stack([p.suh for p in d_projs], dim=0),
            ),
            dim=1,
        ).contiguous()
        d_svh_t = torch.stack([p.svh for p in d_projs], dim=0).contiguous()
        tiers.append(
            prepare_trellis256_moe_weights(
                w13_t,
                w2_t,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=2,
                activation=activation,
                fc1_tile_n=tile_config[1],
                fc2_tile_n=tile_config[3],
                params_dtype=torch.float16,
                w13_layout="trellis_t256_proj",
                trellis_bits=t_bits,
                codebook=codebook,
                gate_suh=g_suh_t,
                up_suh=u_suh_t,
                intermediate_rotations=inter_t,
                down_svh=d_svh_t,
                tile_config=tile_config,
            )
        )
        for local_i, g_exp in enumerate(globals_t):
            gate_w_by_global[g_exp] = decode_exl3_projection_cpu(g_projs[local_i], t_bits)
            up_w_by_global[g_exp] = decode_exl3_projection_cpu(u_projs[local_i], t_bits)
            down_w_by_global[g_exp] = decode_exl3_projection_cpu(d_projs[local_i], t_bits)

    mapping, descriptor = build_tiered_maps(tier_globals[0], tier_globals[1], device=device)
    rotations = combine_trellis_rotations(tiers[0], tiers[1])
    idx = mapping.to(dtype=torch.long)
    gate_suh_global = rotations.gate_suh.view(4, hidden).index_select(0, idx).contiguous()
    up_suh_global = rotations.up_suh.view(4, hidden).index_select(0, idx).contiguous()
    inter_global = rotations.intermediate.view(4, 3 * intermediate).index_select(0, idx).contiguous()
    down_svh_global = rotations.down_svh.view(4, hidden).index_select(0, idx).contiguous()
    gate_w = [gate_w_by_global[e] for e in range(4)]
    up_w = [up_w_by_global[e] for e in range(4)]
    down_w = [down_w_by_global[e] for e in range(4)]
    return (
        (tiers[0], tiers[1]),
        mapping,
        descriptor,
        rotations,
        gate_w,
        up_w,
        down_w,
        gate_suh_global,
        up_suh_global,
        inter_global,
        down_svh_global,
    )


def _encode_wire_e4m3_k32(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode BF16 [M, H] rows into E4M3+UE8M0 K32 wire bytes and exact widened BF16 rows."""
    rows, h = x_bf16.shape
    blocks = x_bf16.float().reshape(rows, h // 32, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    e4m3 = (blocks / torch.exp2(exponent)).to(torch.float8_e4m3fn)
    wire = torch.empty((rows, h + h // 32), dtype=torch.uint8, device=x_bf16.device)
    wire[:, :h] = e4m3.reshape(rows, h).view(torch.uint8)
    wire[:, h:] = (exponent.reshape(rows, h // 32) + 127).to(torch.uint8)
    decoded_bf16 = (e4m3.float() * torch.exp2(exponent)).reshape(rows, h).to(torch.bfloat16)
    return wire, decoded_bf16


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    ("bits", "variant"),
    [
        ((2, 3), "proj_packed_bf16"),
        ((3, 4), "proj_direct_fp32"),
        ((4, 5), "proj_packed_bf16_graph"),
        ((4, 6), "proj_post_clamp"),
        ((5, 6), "proj_situ"),
        ((3, 4), "disjoint_token_major"),
        ((4, 5), "disjoint_fused_rot"),
        ((3, 4), "compact_true_width"),
        ((3, 4), "paired_boundary_last"),
    ],
)
def test_mixed_trellis_2tier_cooperative_bitwise_matches_reference(
    codebook: str,
    bits: tuple[int, int],
    variant: str,
) -> None:
    """Bitwise oracle for 2-tier cooperative W4A16MixedTrellisKernel across K2..K6 and modes."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden, intermediate = 256, 256
    experts, top_k = 4, 2
    token_major = variant == "disjoint_token_major"
    fused_rot = variant == "disjoint_fused_rot"
    capacity_m = 64 if token_major else (16 if fused_rot else 8)
    block_m = 16 if fused_rot else 8
    tile_config = (64, 128, 64, 128)
    activation = "situ" if variant == "proj_situ" else ("silu_post_clamp" if variant == "proj_post_clamp" else "silu")
    swiglu_limit = 7.0 if variant == "proj_post_clamp" else None
    out_dtype_str = "fp32" if variant == "proj_direct_fp32" else "bf16"
    out_dtype = torch.float32 if out_dtype_str == "fp32" else torch.bfloat16
    direct = variant == "proj_direct_fp32"

    seed_base = 60_000 + bits[0] * 300 + bits[1] * 30 + (1 if codebook == "mul1" else 0)
    if variant.startswith("proj_"):
        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
        ) = _build_projection_tiered_fixture(
            hidden=hidden,
            intermediate=intermediate,
            tier_bits=bits,
            gate_tiers=(0, 1, 0, 1),
            up_tiers=(1, 0, 0, 1),
            down_tiers=(0, 1, 1, 0),
            codebook=codebook,
            seed_base=seed_base,
            tile_config=tile_config,
            activation=activation,
            device=device,
        )
        gate_counts: tuple[int, int] | None = (2, 2)
        up_counts: tuple[int, int] | None = (2, 2)
    else:
        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
        ) = _build_disjoint_2tier_fixture(
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            codebook=codebook,
            seed_base=seed_base,
            tile_config=tile_config,
            activation=activation,
            device=device,
        )
        gate_counts = None
        up_counts = None

    compact = variant == "compact_true_width"
    paired_boundary = "last" if variant == "paired_boundary_last" else None

    if compact:
        # Combined order is (2, 0, 3, 1); vary live widths across H128 multiples.
        combined_widths = [256, 128, 128, 256]
        descriptor = build_compact_trellis_descriptor(
            descriptor,
            combined_widths,
            hidden_size=hidden,
            intermediate_size=intermediate,
            bits=bits,
        )
        compact_tiers = []
        for t_idx, tier in enumerate(tiers):
            t_bits = bits[t_idx]
            packed = tier.w13.reshape(2, 2, hidden // 16, intermediate // 16, 8 * t_bits)
            down = tier.w2.reshape(2, intermediate // 16, hidden // 16, 8 * t_bits)
            local_widths = combined_widths[t_idx * 2 : t_idx * 2 + 2]
            w13_c = torch.cat(
                [packed[p, e, :, : w // 16].contiguous().flatten() for p in range(2) for e, w in enumerate(local_widths)]
            )
            w2_c = torch.cat([down[e, : w // 16].contiguous().flatten() for e, w in enumerate(local_widths)])
            compact_tiers.append(replace(tier, w13=w13_c, w2=w2_c))
        tiers = (compact_tiers[0], compact_tiers[1])
        # Zero the unowned upper 128 intermediate scales in the reference table for 128-wide experts.
        inter_rot = inter_rot.clone()
        for g_exp in range(4):
            c_idx = int(mapping[g_exp].item())
            live_w = combined_widths[c_idx]
            if live_w < intermediate:
                inter_rot[g_exp, live_w:intermediate] = 0
                inter_rot[g_exp, intermediate + live_w : 2 * intermediate] = 0
                inter_rot[g_exp, 2 * intermediate + live_w : 3 * intermediate] = 0
    elif paired_boundary is not None:
        ext = torch.cat((descriptor, torch.ones(4, dtype=torch.int32, device=device)))
        if hasattr(descriptor, "_mt_projection_counts"):
            ext._mt_projection_counts = descriptor._mt_projection_counts
        descriptor = ext

    route_slots = (
        capacity_m * top_k
        if direct
        else max_packed_route_slots(capacity_m * top_k, block_m, experts)
    )
    max_m_blocks = route_slots if direct else (route_slots + block_m - 1) // block_m
    launch = compile_mixed_trellis(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        route_num_experts=experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        trellis_codebook=codebook,
        swiglu_limit=swiglu_limit,
        activation=activation,
        moe_block_size=block_m,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype=out_dtype_str,
        direct_topk_routes=direct,
        paired_boundary=paired_boundary,
        compact_weights=compact,
        token_major_rotation=token_major,
        fused_input_rotation=fused_rot,
    )
    assert launch.trellis_codebook == codebook
    binding = bind_mixed_trellis(
        tiers[0],
        tiers[1],
        mapping,
        descriptor,
        rotations,
        launch,
        gate_experts=gate_counts,
        up_experts=up_counts,
    )
    buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)

    rng = torch.Generator(device="cpu").manual_seed(65_000 + bits[0] * 19 + bits[1])
    x_full = (torch.randn((capacity_m, hidden), generator=rng) * 0.05).to(
        device=device, dtype=torch.bfloat16
    )
    ids_cpu = torch.stack([torch.randperm(experts, generator=rng)[:top_k] for _ in range(capacity_m)])
    weights_cpu = torch.softmax(torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1)
    if direct:
        ids_cpu[1, 1] = -1
        weights_cpu[1, 1] = 0.0
    ids_full = ids_cpu.to(device=device, dtype=torch.int32)
    weights_full = weights_cpu.to(device=device)

    buffers.output.fill_(float("nan"))
    actual_full = run_bound_mixed_trellis(x_full, weights_full, ids_full, binding, buffers).clone()
    torch.cuda.synchronize(device)

    expected_full = reference_exl3_routed_moe_bitwise(
        x_full,
        ids_full,
        weights_full,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
        activation=activation,
        swiglu_limit=swiglu_limit,
        output_dtype=out_dtype,
        tile_k=tile_config[0],
    )
    assert_bitwise_equal(actual_full, expected_full, f"mixed2_{codebook}_{bits}_{variant}")

    if variant in ("proj_packed_bf16_graph", "compact_true_width"):
        if not direct:
            warmup_mixed_trellis_route_pack(launch, buffers, expert_map=mapping)
        with kernel_resolution_guard("frozen mixed_trellis 2-tier"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_bound_mixed_trellis(x_full, weights_full, ids_full, binding, buffers)
            for _ in range(2):
                buffers.output.fill_(float("nan"))
                buffers.fc1.fill_(float("nan"))
                buffers.activated.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert_bitwise_equal(
                    buffers.output[:capacity_m],
                    expected_full,
                    f"mixed2_graph_{codebook}_{bits}_{variant}",
                )

            x_saved = x_full.clone()
            w_saved = weights_full.clone()
            try:
                for live_m in (1, 3, capacity_m):
                    x_full.copy_(x_saved)
                    weights_full.copy_(w_saved)
                    if live_m < capacity_m:
                        x_full[live_m:].fill_(float("nan"))
                        weights_full[live_m:].fill_(float("nan"))
                    buffers.output.fill_(float("nan"))
                    short = run_bound_mixed_trellis(
                        x_full[:live_m], weights_full[:live_m], ids_full[:live_m], binding, buffers
                    ).clone()
                    torch.cuda.synchronize(device)
                    expected_live = reference_exl3_routed_moe_bitwise(
                        x_saved[:live_m],
                        ids_full[:live_m],
                        w_saved[:live_m],
                        gate_w,
                        up_w,
                        down_w,
                        gate_suh,
                        up_suh,
                        inter_rot,
                        down_svh,
                        activation=activation,
                        swiglu_limit=swiglu_limit,
                        output_dtype=out_dtype,
                        tile_k=tile_config[0],
                    )
                    assert_bitwise_equal(
                        short, expected_live, f"mixed2_frozen_m{live_m}_{codebook}_{bits}_{variant}"
                    )
            finally:
                x_full.copy_(x_saved)
                weights_full.copy_(w_saved)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize("bits", [(2, 3, 4), (3, 4, 5), (4, 5, 6)])
def test_mixed_trellis_3tier_cooperative_bitwise_matches_reference(
    codebook: str,
    bits: tuple[int, int, int],
) -> None:
    """Bitwise oracle for 3-tier cooperative W4A16MixedTrellis3Kernel across K2..K6."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden, intermediate = 256, 256
    experts, top_k = 6, 2
    capacity_m = 8
    tile_config = (64, 128, 64, 128)
    seed_base = 70_000 + bits[0] * 200 + bits[2] * 20 + (1 if codebook == "mul1" else 0)

    (
        tiers,
        mapping,
        descriptor,
        rotations,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
    ) = _build_projection_tiered_fixture(
        hidden=hidden,
        intermediate=intermediate,
        tier_bits=bits,
        gate_tiers=(0, 1, 2, 0, 1, 2),
        up_tiers=(1, 2, 0, 2, 0, 1),
        down_tiers=(2, 0, 1, 1, 2, 0),
        codebook=codebook,
        seed_base=seed_base,
        tile_config=tile_config,
        activation="silu",
        device=device,
    )

    route_slots = max_packed_route_slots(capacity_m * top_k, 8, experts)
    max_m_blocks = (route_slots + 7) // 8
    launch = compile_mixed_trellis3(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        tier2_num_experts=2,
        route_num_experts=experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        tier2_bits=bits[2],
        trellis_codebook=codebook,
        activation="silu",
        moe_block_size=8,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype="bf16",
    )
    assert launch.trellis_codebook == codebook
    binding = bind_mixed_trellis3(
        tiers[0],
        tiers[1],
        tiers[2],
        mapping,
        descriptor,
        rotations,
        launch,
        gate_experts=(2, 2, 2),
        up_experts=(2, 2, 2),
    )
    buffers = make_mixed_trellis3_buffers(launch, device=device, sms=sms)

    rng = torch.Generator(device="cpu").manual_seed(75_000 + bits[0] * 17)
    x_full = (torch.randn((capacity_m, hidden), generator=rng) * 0.05).to(
        device=device, dtype=torch.bfloat16
    )
    ids_full = torch.stack(
        [torch.randperm(experts, generator=rng)[:top_k] for _ in range(capacity_m)]
    ).to(device=device, dtype=torch.int32)
    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    buffers.output.fill_(float("nan"))
    actual_full = run_bound_mixed_trellis3(x_full, weights_full, ids_full, binding, buffers).clone()
    torch.cuda.synchronize(device)

    expected_full = reference_exl3_routed_moe_bitwise(
        x_full,
        ids_full,
        weights_full,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
        activation="silu",
        output_dtype=torch.bfloat16,
        tile_k=tile_config[0],
    )
    assert_bitwise_equal(actual_full, expected_full, f"mixed3_{codebook}_{bits}")

    if bits == (3, 4, 5):
        warmup_mixed_trellis_route_pack(launch, buffers, expert_map=mapping)
        with kernel_resolution_guard("frozen mixed_trellis 3-tier"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_bound_mixed_trellis3(x_full, weights_full, ids_full, binding, buffers)
            buffers.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert_bitwise_equal(buffers.output[:capacity_m], expected_full, f"mixed3_graph_{codebook}")

            x_saved = x_full.clone()
            w_saved = weights_full.clone()
            try:
                for live_m in (1, 3, capacity_m):
                    x_full.copy_(x_saved)
                    weights_full.copy_(w_saved)
                    if live_m < capacity_m:
                        x_full[live_m:].fill_(float("nan"))
                        weights_full[live_m:].fill_(float("nan"))
                    buffers.output.fill_(float("nan"))
                    short = run_bound_mixed_trellis3(
                        x_full[:live_m], weights_full[:live_m], ids_full[:live_m], binding, buffers
                    ).clone()
                    torch.cuda.synchronize(device)
                    expected_live = reference_exl3_routed_moe_bitwise(
                        x_saved[:live_m],
                        ids_full[:live_m],
                        w_saved[:live_m],
                        gate_w,
                        up_w,
                        down_w,
                        gate_suh,
                        up_suh,
                        inter_rot,
                        down_svh,
                        activation="silu",
                        output_dtype=torch.bfloat16,
                        tile_k=tile_config[0],
                    )
                    assert_bitwise_equal(short, expected_live, f"mixed3_frozen_m{live_m}_{codebook}")
            finally:
                x_full.copy_(x_saved)
                weights_full.copy_(w_saved)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    ("bits", "mode"),
    [
        ((3, 4), "bf16_intile"),
        ((4, 5), "bf16_prerotate"),
        ((3, 4), "bf16_compact"),
        ((4, 5), "wire_e4m3_k32"),
        ((4, 5), "row_addresses"),
    ],
)
def test_mixed_trellis_ws_a16_bitwise_matches_reference(
    codebook: str,
    bits: tuple[int, int],
    mode: str,
) -> None:
    """Bitwise oracle for warp-specialized A16 W4A16MixedTrellisWSKernel across BF16, wire, compact, and row_addresses."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden = 512 if mode == "wire_e4m3_k32" else 256
    intermediate = 256
    experts, top_k = 4, 2
    capacity_m = 32
    block_m = 16
    tile_config = (64, 128, 64, 128)
    seed_base = 80_000 + bits[0] * 100 + bits[1] * 10 + (1 if codebook == "mul1" else 0)

    (
        tiers,
        mapping,
        descriptor,
        rotations,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
    ) = _build_disjoint_2tier_fixture(
        hidden=hidden,
        intermediate=intermediate,
        bits=bits,
        codebook=codebook,
        seed_base=seed_base,
        tile_config=tile_config,
        activation="silu",
        device=device,
    )

    compact = mode == "bf16_compact"
    if compact:
        combined_widths = [256, 128, 128, 256]
        descriptor = build_compact_trellis_descriptor(
            descriptor,
            combined_widths,
            hidden_size=hidden,
            intermediate_size=intermediate,
            bits=bits,
        )
        compact_tiers = []
        for t_idx, tier in enumerate(tiers):
            t_bits = bits[t_idx]
            packed = tier.w13.reshape(2, 2, hidden // 16, intermediate // 16, 8 * t_bits)
            down = tier.w2.reshape(2, intermediate // 16, hidden // 16, 8 * t_bits)
            local_widths = combined_widths[t_idx * 2 : t_idx * 2 + 2]
            w13_c = torch.cat(
                [packed[p, e, :, : w // 16].contiguous().flatten() for p in range(2) for e, w in enumerate(local_widths)]
            )
            w2_c = torch.cat([down[e, : w // 16].contiguous().flatten() for e, w in enumerate(local_widths)])
            compact_tiers.append(replace(tier, w13=w13_c, w2=w2_c))
        tiers = (compact_tiers[0], compact_tiers[1])
        inter_rot = inter_rot.clone()
        for g_exp in range(4):
            c_idx = int(mapping[g_exp].item())
            live_w = combined_widths[c_idx]
            if live_w < intermediate:
                inter_rot[g_exp, live_w:intermediate] = 0
                inter_rot[g_exp, intermediate + live_w : 2 * intermediate] = 0
                inter_rot[g_exp, 2 * intermediate + live_w : 3 * intermediate] = 0

    wire = mode == "wire_e4m3_k32"
    indirect = mode == "row_addresses"
    prerotate = True if mode == "bf16_prerotate" else False
    route_slots = max_packed_route_slots(capacity_m * top_k, block_m, experts)
    max_m_blocks = (route_slots + block_m - 1) // block_m

    launch = compile_mixed_trellis(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        route_num_experts=experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        trellis_codebook=codebook,
        activation="silu",
        moe_block_size=block_m,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype="bf16",
        warp_specialized=True,
        input_format="e4m3_k32" if wire else "bf16",
        ws_dynamic_tiles=(mode == "bf16_intile"),
        ws_prerotate=prerotate,
        compact_weights=compact,
        row_addresses=indirect,
    )
    assert launch.trellis_codebook == codebook
    binding = bind_mixed_trellis(tiers[0], tiers[1], mapping, descriptor, rotations, launch)

    rng = torch.Generator(device="cpu").manual_seed(85_000 + bits[0] * 23)
    x_bf16 = (torch.randn((capacity_m, hidden), generator=rng) * 0.05).to(
        device=device, dtype=torch.bfloat16
    )
    ids_full = torch.stack(
        [torch.randperm(experts, generator=rng)[:top_k] for _ in range(capacity_m)]
    ).to(device=device, dtype=torch.int32)
    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    if wire:
        x_input, x_ref = _encode_wire_e4m3_k32(x_bf16)
    else:
        x_input, x_ref = x_bf16, x_bf16

    if indirect:
        # Reverse row-address permutation exercises true non-sequential indirect gather/scatter.
        perm = torch.arange(capacity_m - 1, -1, -1, device=device, dtype=torch.long)
        out_backing = torch.full((capacity_m, hidden), float("nan"), dtype=torch.bfloat16, device=device)
        x_addrs = (x_bf16.data_ptr() + perm * (hidden * 2)).to(dtype=torch.int64).contiguous()
        y_addrs = (out_backing.data_ptr() + perm * (hidden * 2)).to(dtype=torch.int64).contiguous()
        seg_buffers = make_mixed_trellis_segment_buffers(launch, rows=capacity_m, device=device, sms=sms)
        seg_buffers = replace(seg_buffers, output=y_addrs)
        pack_topk_routes_by_expert(
            ids_full[perm],
            block_m,
            experts,
            expert_map=mapping,
            packed_route_indices=seg_buffers.packed_route_indices,
            block_expert_ids=seg_buffers.block_expert_ids,
            packed_route_count=seg_buffers.packed_route_count,
            expert_offsets=seg_buffers.expert_offsets,
            expert_counts=seg_buffers.expert_counts,
        )
        with kernel_resolution_guard("frozen WS row_addresses"):
            run_bound_mixed_trellis(x_addrs, weights_full[perm], ids_full[perm], binding, seg_buffers)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_bound_mixed_trellis(x_addrs, weights_full[perm], ids_full[perm], binding, seg_buffers)
            out_backing.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize(device)
        expected_full = reference_exl3_routed_moe_bitwise(
            x_ref,
            ids_full,
            weights_full,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
            activation="silu",
            output_dtype=torch.bfloat16,
            tile_k=tile_config[0],
        )
        assert_bitwise_equal(out_backing, expected_full, f"ws_row_addrs_{codebook}")
        return

    buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
    warmup_mixed_trellis_route_pack(launch, buffers, expert_map=mapping)

    with kernel_resolution_guard(f"frozen WS {mode}"):
        for live_m in (3, 17, capacity_m):
            buffers.output.fill_(float("nan"))
            actual_live = run_bound_mixed_trellis(
                x_input[:live_m], weights_full[:live_m], ids_full[:live_m], binding, buffers
            ).clone()
            torch.cuda.synchronize(device)
            expected_live = reference_exl3_routed_moe_bitwise(
                x_ref[:live_m],
                ids_full[:live_m],
                weights_full[:live_m],
                gate_w,
                up_w,
                down_w,
                gate_suh,
                up_suh,
                inter_rot,
                down_svh,
                activation="silu",
                output_dtype=torch.bfloat16,
                tile_k=tile_config[0],
            )
            assert_bitwise_equal(actual_live, expected_live, f"ws_{mode}_m{live_m}_{codebook}")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_bound_mixed_trellis(x_input, weights_full, ids_full, binding, buffers)
        buffers.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        assert_bitwise_equal(buffers.output[:capacity_m], expected_live, f"ws_{mode}_graph_{codebook}")


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    ("bits", "k4_case"),
    [
        ((4, 5), "m1_persistent_4stage"),
        ((4, 5), "m80_persistent_and_medm_prefix"),
        ((4, 6), "m80_compact_post_clamp"),
        ((4, 3), "m16_proj_situ"),
        ((4, 2), "m16_smallm_fused_and_act_pdl"),
    ],
)
def test_mixed_trellis_k4_decode_bitwise_matches_reference(
    codebook: str,
    bits: tuple[int, int],
    k4_case: str,
) -> None:
    """Bitwise oracle for W4A16MixedTrellisK4Decode across all persistent and PDL sub-kernels."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    hw_sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden, intermediate = 256, 256
    experts, top_k = 4, 2
    tile_config = (64, 128, 64, 128)
    activation = (
        "situ"
        if k4_case == "m16_proj_situ"
        else ("silu_post_clamp" if k4_case == "m80_compact_post_clamp" else "silu")
    )
    swiglu_limit = 7.0 if k4_case == "m80_compact_post_clamp" else None
    compact = k4_case == "m80_compact_post_clamp"
    proj_tiered = k4_case == "m16_proj_situ"

    capacity_m = (
        1
        if k4_case == "m1_persistent_4stage"
        else (80 if k4_case.startswith("m80_") else 16)
    )
    seed_base = 90_000 + bits[1] * 100 + (1 if codebook == "mul1" else 0)

    if proj_tiered:
        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
        ) = _build_projection_tiered_fixture(
            hidden=hidden,
            intermediate=intermediate,
            tier_bits=bits,
            gate_tiers=(0, 1, 0, 1),
            up_tiers=(1, 0, 0, 1),
            down_tiers=(0, 1, 1, 0),
            codebook=codebook,
            seed_base=seed_base,
            tile_config=tile_config,
            activation=activation,
            device=device,
        )
        gate_counts: tuple[int, int] | None = (2, 2)
        up_counts: tuple[int, int] | None = (2, 2)
    else:
        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
        ) = _build_disjoint_2tier_fixture(
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            codebook=codebook,
            seed_base=seed_base,
            tile_config=tile_config,
            activation=activation,
            device=device,
        )
        gate_counts = None
        up_counts = None

    if compact:
        combined_widths = [256, 128, 128, 256]
        descriptor = build_compact_trellis_descriptor(
            descriptor,
            combined_widths,
            hidden_size=hidden,
            intermediate_size=intermediate,
            bits=bits,
        )
        compact_tiers = []
        for t_idx, tier in enumerate(tiers):
            t_bits = bits[t_idx]
            packed = tier.w13.reshape(2, 2, hidden // 16, intermediate // 16, 8 * t_bits)
            down = tier.w2.reshape(2, intermediate // 16, hidden // 16, 8 * t_bits)
            local_widths = combined_widths[t_idx * 2 : t_idx * 2 + 2]
            w13_c = torch.cat(
                [packed[p, e, :, : w // 16].contiguous().flatten() for p in range(2) for e, w in enumerate(local_widths)]
            )
            w2_c = torch.cat([down[e, : w // 16].contiguous().flatten() for e, w in enumerate(local_widths)])
            compact_tiers.append(replace(tier, w13=w13_c, w2=w2_c))
        tiers = (compact_tiers[0], compact_tiers[1])
        inter_rot = inter_rot.clone()
        for g_exp in range(4):
            c_idx = int(mapping[g_exp].item())
            live_w = combined_widths[c_idx]
            if live_w < intermediate:
                inter_rot[g_exp, live_w:intermediate] = 0
                inter_rot[g_exp, intermediate + live_w : 2 * intermediate] = 0
                inter_rot[g_exp, 2 * intermediate + live_w : 3 * intermediate] = 0

    route_slots = max_packed_route_slots(capacity_m * top_k, 8, experts)
    max_m_blocks = (route_slots + 7) // 8

    rng = torch.Generator(device="cpu").manual_seed(95_000 + bits[1] * 17)
    x_full = (torch.randn((capacity_m, hidden), generator=rng) * 0.05).to(
        device=device, dtype=torch.bfloat16
    )
    ids_full = torch.stack(
        [torch.randperm(experts, generator=rng)[:top_k] for _ in range(capacity_m)]
    ).to(device=device, dtype=torch.int32)
    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    if k4_case == "m16_smallm_fused_and_act_pdl":
        # Deterministically exercise both fc1_fused_smallm_kernel(do_fused_act=1) (sms=2 -> ws_cap=8,
        # live_m=4 -> max_live_blocks=4) and fc1_fused_smallm_kernel(do_fused_act=0) + act_kernel
        # (sms=1 -> ws_cap=4, live_m=6 -> max_live_blocks=5) on every GPU.
        for launch_sms, live_m in ((2, 4), (1, 6)):
            launch_sub = compile_mixed_trellis(
                size_m=capacity_m,
                hidden_size=hidden,
                intermediate_size=intermediate,
                tier0_num_experts=2,
                tier1_num_experts=2,
                route_num_experts=experts,
                top_k=top_k,
                max_m_blocks=max_m_blocks,
                sms=launch_sms,
                workspace_sms=hw_sms,
                max_shared_mem=max_shared_mem,
                force_tile_config=tile_config,
                tier0_bits=bits[0],
                tier1_bits=bits[1],
                trellis_codebook=codebook,
                activation=activation,
                moe_block_size=8,
                rotation_input_dtype="bf16",
                full_rotation_output_dtype="bf16",
                _k4_decode_override=True,
            )
            binding_sub = bind_mixed_trellis(
                tiers[0], tiers[1], mapping, descriptor, rotations, launch_sub
            )
            buffers_sub = make_mixed_trellis_buffers(launch_sub, device=device, sms=launch_sms)
            buffers_sub.output.fill_(float("nan"))
            actual_sub = run_bound_mixed_trellis(
                x_full[:live_m], weights_full[:live_m], ids_full[:live_m], binding_sub, buffers_sub
            ).clone()
            torch.cuda.synchronize(device)
            expected_sub = reference_exl3_routed_moe_bitwise(
                x_full[:live_m],
                ids_full[:live_m],
                weights_full[:live_m],
                gate_w,
                up_w,
                down_w,
                gate_suh,
                up_suh,
                inter_rot,
                down_svh,
                activation=activation,
                output_dtype=torch.bfloat16,
                tile_k=tile_config[0],
            )
            assert_bitwise_equal(
                actual_sub, expected_sub, f"k4_smallm_sms{launch_sms}_m{live_m}_{codebook}"
            )
        return

    launch = compile_mixed_trellis(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        route_num_experts=experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks,
        sms=hw_sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        trellis_codebook=codebook,
        swiglu_limit=swiglu_limit,
        activation=activation,
        moe_block_size=8,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype="bf16",
        compact_weights=compact,
        _k4_decode_override=True,
    )
    assert launch.trellis_codebook == codebook
    binding = bind_mixed_trellis(
        tiers[0],
        tiers[1],
        mapping,
        descriptor,
        rotations,
        launch,
        gate_experts=gate_counts,
        up_experts=up_counts,
    )
    buffers = make_mixed_trellis_buffers(launch, device=device, sms=hw_sms)
    warmup_mixed_trellis_route_pack(launch, buffers, expert_map=mapping)

    live_counts = (
        (1,)
        if capacity_m == 1
        else ((1, 8, 24) if k4_case == "m80_persistent_and_medm_prefix" else ((1, 8, 20) if compact else (1, 5, 16)))
    )

    m1_launch = None
    m1_binding = None
    m1_buffers = None
    if k4_case == "m80_persistent_and_medm_prefix":
        m1_slots = max_packed_route_slots(top_k, 8, experts)
        m1_launch = compile_mixed_trellis(
            size_m=1,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tier0_num_experts=2,
            tier1_num_experts=2,
            route_num_experts=experts,
            top_k=top_k,
            max_m_blocks=(m1_slots + 7) // 8,
            sms=hw_sms,
            max_shared_mem=max_shared_mem,
            force_tile_config=tile_config,
            tier0_bits=bits[0],
            tier1_bits=bits[1],
            trellis_codebook=codebook,
            activation=activation,
            moe_block_size=8,
            rotation_input_dtype="bf16",
            full_rotation_output_dtype="bf16",
            _k4_decode_override=True,
        )
        m1_binding = bind_mixed_trellis(tiers[0], tiers[1], mapping, descriptor, rotations, m1_launch)
        m1_buffers = make_mixed_trellis_buffers(m1_launch, device=device, sms=hw_sms)
        warmup_mixed_trellis_route_pack(m1_launch, m1_buffers, expert_map=mapping)

    with kernel_resolution_guard(f"frozen k4_decode {k4_case}"):
        x_saved = x_full.clone()
        w_saved = weights_full.clone()
        row0_outputs: list[torch.Tensor] = []
        if m1_binding is not None and m1_buffers is not None:
            m1_buffers.output.fill_(float("nan"))
            out_m1 = run_bound_mixed_trellis(
                x_saved[:1], w_saved[:1], ids_full[:1], m1_binding, m1_buffers
            ).clone()
            torch.cuda.synchronize(device)
            row0_outputs.append(out_m1)
        try:
            for live_m in live_counts:
                x_full.copy_(x_saved)
                weights_full.copy_(w_saved)
                if live_m < capacity_m:
                    x_full[live_m:].fill_(float("nan"))
                    weights_full[live_m:].fill_(float("nan"))
                buffers.output.fill_(float("nan"))
                actual = run_bound_mixed_trellis(
                    x_full[:live_m], weights_full[:live_m], ids_full[:live_m], binding, buffers
                ).clone()
                torch.cuda.synchronize(device)
                expected = reference_exl3_routed_moe_bitwise(
                    x_saved[:live_m],
                    ids_full[:live_m],
                    w_saved[:live_m],
                    gate_w,
                    up_w,
                    down_w,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    activation=activation,
                    swiglu_limit=swiglu_limit,
                    output_dtype=torch.bfloat16,
                    tile_k=tile_config[0],
                )
                assert_bitwise_equal(actual, expected, f"k4_{k4_case}_m{live_m}_{codebook}")
                row0_outputs.append(actual[:1].clone())
        finally:
            x_full.copy_(x_saved)
            weights_full.copy_(w_saved)

        for other in row0_outputs[1:]:
            assert_bitwise_equal(other, row0_outputs[0], f"k4_prefix_row0_{k4_case}_{codebook}")

        cap_test_m = live_counts[-1]
        expected_cap = reference_exl3_routed_moe_bitwise(
            x_full[:cap_test_m],
            ids_full[:cap_test_m],
            weights_full[:cap_test_m],
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
            activation=activation,
            swiglu_limit=swiglu_limit,
            output_dtype=torch.bfloat16,
            tile_k=tile_config[0],
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_bound_mixed_trellis(
                x_full[:cap_test_m], weights_full[:cap_test_m], ids_full[:cap_test_m], binding, buffers
            )
        for _ in range(2):
            buffers.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert_bitwise_equal(
                buffers.output[:cap_test_m], expected_cap, f"k4_graph_{k4_case}_{codebook}"
            )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(torch.cuda.current_device()) != (12, 1),
    reason="requires Spark SM121 for automatic use_sm121_k4_decode dispatch",
)
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
def test_mixed_trellis_k4_decode_sm121_auto_inkling2_shard_bitwise(codebook: str) -> None:
    """Verify automatic SM121 K4 decode selection on Inkling2/Hy4 Flash shard geometry (4608, 384, E=384, k=8, compact K4/K5)."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden, intermediate = 4608, 384
    route_experts, top_k = 384, 8
    capacity_m = 8
    bits = (4, 5)
    tile_config = (64, 128, 64, 128)
    chosen_globals = [0, 51, 102, 153, 204, 255, 306, 383]
    tier0_globals = (chosen_globals[2], chosen_globals[0], chosen_globals[6], chosen_globals[4])
    tier1_globals = (chosen_globals[3], chosen_globals[1], chosen_globals[7], chosen_globals[5])

    seed_base = 98_000 + (1 if codebook == "mul1" else 0)
    tiers = []
    gate_w_by_global: dict[int, torch.Tensor] = {}
    up_w_by_global: dict[int, torch.Tensor] = {}
    down_w_by_global: dict[int, torch.Tensor] = {}
    for t_idx, (t_bits, globals_t) in enumerate(zip(bits, (tier0_globals, tier1_globals), strict=True)):
        g_projs = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, t_bits, codebook=codebook, seed=seed_base + t_idx * 40 + i * 3 + 0, device=device
            )
            for i in range(4)
        ]
        u_projs = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, t_bits, codebook=codebook, seed=seed_base + t_idx * 40 + i * 3 + 1, device=device
            )
            for i in range(4)
        ]
        d_projs = [
            make_gptqmodel_exl3_projection(
                intermediate, hidden, t_bits, codebook=codebook, seed=seed_base + t_idx * 40 + i * 3 + 2, device=device
            )
            for i in range(4)
        ]
        w13_t = torch.stack(
            (
                torch.stack([p.trellis for p in g_projs], dim=0),
                torch.stack([p.trellis for p in u_projs], dim=0),
            ),
            dim=0,
        ).contiguous()
        w2_t = torch.stack([p.trellis for p in d_projs], dim=0).contiguous()
        g_suh_t = torch.stack([p.suh for p in g_projs], dim=0).contiguous()
        u_suh_t = torch.stack([p.suh for p in u_projs], dim=0).contiguous()
        inter_t = torch.cat(
            (
                torch.stack([p.svh for p in g_projs], dim=0),
                torch.stack([p.svh for p in u_projs], dim=0),
                torch.stack([p.suh for p in d_projs], dim=0),
            ),
            dim=1,
        ).contiguous()
        d_svh_t = torch.stack([p.svh for p in d_projs], dim=0).contiguous()
        tiers.append(
            prepare_trellis256_moe_weights(
                w13_t,
                w2_t,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=4,
                activation="silu",
                fc1_tile_n=tile_config[1],
                fc2_tile_n=tile_config[3],
                params_dtype=torch.float16,
                w13_layout="trellis_t256_proj",
                trellis_bits=t_bits,
                codebook=codebook,
                gate_suh=g_suh_t,
                up_suh=u_suh_t,
                intermediate_rotations=inter_t,
                down_svh=d_svh_t,
                tile_config=tile_config,
            )
        )
        for local_i, g_exp in enumerate(globals_t):
            gate_w_by_global[g_exp] = decode_exl3_projection_cpu(g_projs[local_i], t_bits)
            up_w_by_global[g_exp] = decode_exl3_projection_cpu(u_projs[local_i], t_bits)
            down_w_by_global[g_exp] = decode_exl3_projection_cpu(d_projs[local_i], t_bits)

    route_map = torch.full((route_experts,), -1, dtype=torch.int32, device=device)
    combined_order = list(tier0_globals) + list(tier1_globals)
    for c_idx, g_exp in enumerate(combined_order):
        route_map[g_exp] = c_idx
    desc_row = torch.tensor(
        [0, 1, 2, 3, (1 << 9) | 0, (1 << 9) | 1, (1 << 9) | 2, (1 << 9) | 3],
        dtype=torch.int32,
        device=device,
    )
    base_desc = desc_row.repeat(3).contiguous()
    base_desc._mt_projection_counts = ((4, 4), (4, 4))

    combined_widths = [384, 256, 128, 384, 256, 384, 128, 256]
    compact_desc = build_compact_trellis_descriptor(
        base_desc,
        combined_widths,
        hidden_size=hidden,
        intermediate_size=intermediate,
        bits=bits,
    )
    compact_tiers = []
    for t_idx, tier in enumerate(tiers):
        t_bits = bits[t_idx]
        packed = tier.w13.reshape(2, 4, hidden // 16, intermediate // 16, 8 * t_bits)
        down = tier.w2.reshape(4, intermediate // 16, hidden // 16, 8 * t_bits)
        local_widths = combined_widths[t_idx * 4 : t_idx * 4 + 4]
        w13_c = torch.cat(
            [packed[p, e, :, : w // 16].contiguous().flatten() for p in range(2) for e, w in enumerate(local_widths)]
        )
        w2_c = torch.cat([down[e, : w // 16].contiguous().flatten() for e, w in enumerate(local_widths)])
        compact_tiers.append(replace(tier, w13=w13_c, w2=w2_c))

    rotations = combine_trellis_rotations(tiers[0], tiers[1])
    route_slots = max_packed_route_slots(capacity_m * top_k, 8, route_experts)
    common_compile = dict(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=4,
        tier1_num_experts=4,
        route_num_experts=route_experts,
        top_k=top_k,
        max_m_blocks=(route_slots + 7) // 8,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=4,
        tier1_bits=5,
        trellis_codebook=codebook,
        activation="silu",
        moe_block_size=8,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype="bf16",
        compact_weights=True,
    )
    auto_launch = compile_mixed_trellis(**common_compile)
    explicit_k4 = compile_mixed_trellis(**common_compile, _k4_decode_override=True)
    coop_launch = compile_mixed_trellis(**common_compile, _k4_decode_override=False)
    # Automatic SM121 dispatch must select W4A16MixedTrellisK4Decode (not cooperative).
    assert auto_launch.compiled is explicit_k4.compiled
    assert auto_launch.compiled is not coop_launch.compiled

    rng = torch.Generator(device="cpu").manual_seed(99_123)
    x = (torch.randn((capacity_m, hidden), generator=rng) * 0.02).to(device=device, dtype=torch.bfloat16)
    chosen_t = torch.tensor(chosen_globals, dtype=torch.int32)
    ids = torch.stack([chosen_t[torch.randperm(8, generator=rng)] for _ in range(capacity_m)]).to(device=device)
    weights = torch.softmax(torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1).to(
        device=device
    )

    auto_binding = bind_mixed_trellis(
        compact_tiers[0], compact_tiers[1], route_map, compact_desc, rotations, auto_launch
    )
    coop_binding = bind_mixed_trellis(
        compact_tiers[0], compact_tiers[1], route_map, compact_desc, rotations, coop_launch
    )
    auto_buf = make_mixed_trellis_buffers(auto_launch, device=device, sms=sms)
    coop_buf = make_mixed_trellis_buffers(coop_launch, device=device, sms=sms)

    auto_out = run_bound_mixed_trellis(x, weights, ids, auto_binding, auto_buf).clone()
    coop_out = run_bound_mixed_trellis(x, weights, ids, coop_binding, coop_buf).clone()
    torch.cuda.synchronize(device)
    assert_bitwise_equal(auto_out, coop_out, f"sm121_auto_inkling2_vs_coop_{codebook}")

    # Also verify bit-for-bit against CPU Reference(codebook) over the 8 compact local experts.
    local_ids = torch.stack(
        [torch.tensor([combined_order.index(int(g)) for g in row.tolist()], dtype=torch.int32) for row in ids.cpu()]
    ).to(device=device)
    inter_ref = rotations.intermediate.view(8, 3 * intermediate).clone()
    for c_idx, live_w in enumerate(combined_widths):
        if live_w < intermediate:
            inter_ref[c_idx, live_w:intermediate] = 0
            inter_ref[c_idx, intermediate + live_w : 2 * intermediate] = 0
            inter_ref[c_idx, 2 * intermediate + live_w : 3 * intermediate] = 0
    expected_cpu = reference_exl3_routed_moe_bitwise(
        x,
        local_ids,
        weights,
        [gate_w_by_global[g] for g in combined_order],
        [up_w_by_global[g] for g in combined_order],
        [down_w_by_global[g] for g in combined_order],
        rotations.gate_suh.view(8, hidden),
        rotations.up_suh.view(8, hidden),
        inter_ref,
        rotations.down_svh.view(8, hidden),
        activation="silu",
        output_dtype=torch.bfloat16,
        tile_k=tile_config[0],
    )
    assert_bitwise_equal(auto_out, expected_cpu, f"sm121_auto_inkling2_vs_cpu_{codebook}")


def _run_mixed_trellis4_launch(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    launch,
    tiers: tuple,
    global_to_combined: torch.Tensor,
    descriptor_map: torch.Tensor,
    rotations: MixedTrellisRotations,
    buffers,
    gate_counts: tuple[int, int, int, int],
    up_counts: tuple[int, int, int, int],
    fc2_counts: tuple[int, int, int, int],
    route_num_experts: int,
) -> torch.Tensor:
    """Invoke compiled 4-tier cooperative kernel + topk_sum epilogue using only 0f1fed7f-compatible APIs."""
    m = int(x.shape[0])
    packed, block_experts, packed_count = pack_topk_routes_by_expert(
        topk_ids,
        launch.moe_block_size,
        route_num_experts,
        expert_map=global_to_combined,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
    )
    stream = current_cuda_stream()
    tier_ptrs = tuple(_mixed_trellis_tier_pointers(t) for t in tiers)
    descriptor_ptr = make_ptr(
        cutlass.Int32, descriptor_map.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    global_map_ptr = make_ptr(
        cutlass.Int32, global_to_combined.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    inter_rot_ptr = make_ptr(
        cutlass.Float16, rotations.intermediate.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    gate_suh_ptr = make_ptr(
        cutlass.Float16, rotations.gate_suh.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    up_suh_ptr = make_ptr(
        cutlass.Float16, rotations.up_suh.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    down_svh_ptr = make_ptr(
        cutlass.Float16, rotations.down_svh.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    launch.compiled(
        make_ptr(
            _cutlass_element_dtype(launch.rotation_input_dtype),
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        *tier_ptrs[0],
        *tier_ptrs[1],
        *tier_ptrs[2],
        *tier_ptrs[3],
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        block_experts,
        packed_count,
        descriptor_ptr,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        inter_rot_ptr,
        gate_suh_ptr,
        up_suh_ptr,
        make_ptr(
            cutlass.Uint8,
            launch.trellis_lut.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(int(launch.tier0_num_experts)),
        Int32(int(launch.tier1_num_experts)),
        Int32(int(launch.tier2_num_experts)),
        Int32(int(launch.tier3_num_experts)),
        Int32(fc2_counts[0]),
        Int32(fc2_counts[1]),
        Int32(fc2_counts[2]),
        Int32(fc2_counts[3]),
        m,
        max(int(launch.blocks_per_sm) * int(launch.sms), 1),
        stream,
        tier0_gate_experts=Int32(gate_counts[0]),
        tier1_gate_experts=Int32(gate_counts[1]),
        tier2_gate_experts=Int32(gate_counts[2]),
        tier3_gate_experts=Int32(gate_counts[3]),
        tier0_up_experts=Int32(up_counts[0]),
        tier1_up_experts=Int32(up_counts[1]),
        tier2_up_experts=Int32(up_counts[2]),
        tier3_up_experts=Int32(up_counts[3]),
    )
    launch.topk_sum.compiled(
        make_ptr(
            cutlass.Float16,
            buffers.fc2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.BFloat16
            if launch.topk_sum.full_rotation_output_dtype == "bf16"
            else cutlass.Float32,
            buffers.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32 if topk_ids.dtype == torch.int32 else cutlass.Int64,
            topk_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4 if topk_ids.dtype == torch.int32 else 8,
        ),
        global_map_ptr,
        down_svh_ptr,
        Int32(launch.topk_sum.num_experts),
        Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    ("bits", "case_mode"),
    [
        ((2, 3, 4, 5), "proj_tiered_m8_silu_bf16"),
        ((3, 4, 5, 6), "proj_tiered_m16_gb10_graph_frozen"),
        ((4, 5, 6, 2), "proj_tiered_m8_clamp_bf16"),
        ((2, 4, 5, 6), "padded_e600_desc10_situ_fp32"),
    ],
)
def test_mixed_trellis_4tier_cooperative_bitwise_matches_reference(
    codebook: str,
    bits: tuple[int, int, int, int],
    case_mode: str,
) -> None:
    """Bitwise oracle for 4-tier cooperative W4A16MixedTrellis4Kernel across K2..K6, schedules, E600 10-bit descriptors, graph replay, and frozen live M."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden, intermediate = 256, 256
    tile_config = (64, 128, 64, 128)
    top_k = 3
    seed_base = 110_000 + sum(b * (i + 1) * 17 for i, b in enumerate(bits)) + (1 if codebook == "mul1" else 0)

    if case_mode == "padded_e600_desc10_situ_fp32":
        # Exercise E>512 10-bit descriptors, padded projection slots (total_experts=9 > 8 live),
        # asymmetric gate_counts != up_counts, Situ activation, and fp32 output.
        num_live = 8
        route_num_experts = 600
        chosen_globals = [0, 73, 149, 251, 383, 511, 512, 599]
        gate_tiers = (0, 0, 0, 1, 1, 2, 2, 3)  # counts: (3, 2, 2, 1)
        up_tiers = (0, 1, 1, 2, 2, 3, 3, 0)    # counts: (2, 2, 2, 2)
        down_tiers = (3, 2, 1, 0, 3, 2, 1, 0)  # counts: (2, 2, 2, 2)
        tier_slots = (3, 2, 2, 2)              # sum = 9 > 8 live
        gate_counts = (3, 2, 2, 1)
        up_counts = (2, 2, 2, 2)
        fc2_counts = (2, 2, 2, 2)
        activation = "situ"
        swiglu_limit = None
        out_dtype_str = "fp32"
        out_dtype = torch.float32
        block_m = 8
        capacity_m = 8
        decode_schedule = None

        gate_projs = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, bits[gate_tiers[e]], codebook=codebook, seed=seed_base + e * 7 + 0, device=device
            )
            for e in range(num_live)
        ]
        up_projs = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, bits[up_tiers[e]], codebook=codebook, seed=seed_base + e * 7 + 1, device=device
            )
            for e in range(num_live)
        ]
        down_projs = [
            make_gptqmodel_exl3_projection(
                intermediate, hidden, bits[down_tiers[e]], codebook=codebook, seed=seed_base + e * 7 + 2, device=device
            )
            for e in range(num_live)
        ]

        gate_suh_live = torch.stack([p.suh for p in gate_projs], dim=0).contiguous()
        up_suh_live = torch.stack([p.suh for p in up_projs], dim=0).contiguous()
        inter_live = torch.cat(
            (
                torch.stack([p.svh for p in gate_projs], dim=0),
                torch.stack([p.svh for p in up_projs], dim=0),
                torch.stack([p.suh for p in down_projs], dim=0),
            ),
            dim=1,
        ).contiguous()
        down_svh_live = torch.stack([p.svh for p in down_projs], dim=0).contiguous()

        prepared_tiers = []
        proj_stride = (hidden // 16) * (intermediate // 16) * 8
        for t_idx, t_bits in enumerate(bits):
            slots_t = tier_slots[t_idx]
            g_ids = [e for e in range(num_live) if gate_tiers[e] == t_idx]
            u_ids = [e for e in range(num_live) if up_tiers[e] == t_idx]
            d_ids = [e for e in range(num_live) if down_tiers[e] == t_idx]
            g_padded = [gate_projs[g_ids[i % len(g_ids)]].trellis for i in range(slots_t)]
            u_padded = [up_projs[u_ids[i % len(u_ids)]].trellis for i in range(slots_t)]
            d_padded = [down_projs[d_ids[i % len(d_ids)]].trellis for i in range(slots_t)]
            w13_full = torch.stack(
                (torch.stack(g_padded, dim=0), torch.stack(u_padded, dim=0)), dim=0
            ).contiguous()
            w2_full = torch.stack(d_padded, dim=0).contiguous()
            prep = prepare_trellis256_moe_weights(
                w13_full,
                w2_full,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=slots_t,
                activation=activation,
                fc1_tile_n=tile_config[1],
                fc2_tile_n=tile_config[3],
                params_dtype=torch.float16,
                w13_layout="trellis_t256_proj",
                trellis_bits=t_bits,
                codebook=codebook,
                gate_suh=gate_suh_live[:slots_t].contiguous(),
                up_suh=up_suh_live[:slots_t].contiguous(),
                intermediate_rotations=inter_live[:slots_t].contiguous(),
                down_svh=down_svh_live[:slots_t].contiguous(),
                tile_config=tile_config,
            )
            plane_elems = proj_stride * t_bits
            w13_planes = prep.w13.reshape(2 * slots_t, plane_elems)
            w13_tight = torch.cat(
                (w13_planes[: len(g_ids)], w13_planes[slots_t : slots_t + len(u_ids)]), dim=0
            ).contiguous().view(-1)
            w2_planes = prep.w2.reshape(slots_t, plane_elems)
            w2_tight = w2_planes[: len(d_ids)].contiguous().view(-1)
            w2_gs_tight = prep.w2_global_scale[: len(d_ids)].contiguous()
            prepared_tiers.append(
                replace(prep, w13=w13_tight, w2=w2_tight, w2_global_scale=w2_gs_tight)
            )
        tiers = tuple(prepared_tiers)

        desc_bits = _descriptor_bits(route_num_experts)
        assert desc_bits == 10
        total_slots = sum(tier_slots)
        descriptor = torch.full((3 * total_slots,), -1, dtype=torch.int32, device=device)
        for row_idx, proj_tiers in enumerate((gate_tiers, up_tiers, down_tiers)):
            counters = [0, 0, 0, 0]
            row_vals = []
            for t_id in proj_tiers:
                loc = counters[t_id]
                counters[t_id] += 1
                row_vals.append((t_id << desc_bits) | loc)
            descriptor[row_idx * total_slots : row_idx * total_slots + num_live] = torch.tensor(
                row_vals, dtype=torch.int32, device=device
            )
        mapping = torch.full((route_num_experts,), -1, dtype=torch.int32, device=device)
        for c_idx, g_exp in enumerate(chosen_globals):
            mapping[g_exp] = c_idx

        pad_extra = total_slots - num_live
        rotations = MixedTrellisRotations(
            intermediate=torch.cat((inter_live, inter_live[:pad_extra]), dim=0).contiguous(),
            gate_suh=torch.cat((gate_suh_live, gate_suh_live[:pad_extra]), dim=0).contiguous(),
            up_suh=torch.cat((up_suh_live, up_suh_live[:pad_extra]), dim=0).contiguous(),
            down_svh=torch.cat((down_svh_live, down_svh_live[:pad_extra]), dim=0).contiguous(),
        )
        gate_w = [decode_exl3_projection_cpu(gate_projs[e], bits[gate_tiers[e]]) for e in range(num_live)]
        up_w = [decode_exl3_projection_cpu(up_projs[e], bits[up_tiers[e]]) for e in range(num_live)]
        down_w = [decode_exl3_projection_cpu(down_projs[e], bits[down_tiers[e]]) for e in range(num_live)]
        gate_suh_ref = gate_suh_live
        up_suh_ref = up_suh_live
        inter_ref = inter_live
        down_svh_ref = down_svh_live
    else:
        num_live = 8
        route_num_experts = 8
        chosen_globals = list(range(8))
        gate_tiers = (0, 1, 2, 3, 0, 1, 2, 3)
        up_tiers = (1, 2, 3, 0, 2, 3, 0, 1)
        down_tiers = (3, 0, 1, 2, 1, 0, 3, 2)
        tier_slots = (2, 2, 2, 2)
        gate_counts = (2, 2, 2, 2)
        up_counts = (2, 2, 2, 2)
        fc2_counts = (2, 2, 2, 2)
        activation = "silu_post_clamp" if case_mode == "proj_tiered_m8_clamp_bf16" else "silu"
        swiglu_limit = 7.0 if activation == "silu_post_clamp" else None
        out_dtype_str = "bf16"
        out_dtype = torch.bfloat16
        block_m = 16 if case_mode == "proj_tiered_m16_gb10_graph_frozen" else 8
        capacity_m = 16 if block_m == 16 else 8
        decode_schedule = "gb10" if case_mode == "proj_tiered_m16_gb10_graph_frozen" else None

        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh_ref,
            up_suh_ref,
            inter_ref,
            down_svh_ref,
        ) = _build_projection_tiered_fixture(
            hidden=hidden,
            intermediate=intermediate,
            tier_bits=bits,
            gate_tiers=gate_tiers,
            up_tiers=up_tiers,
            down_tiers=down_tiers,
            codebook=codebook,
            seed_base=seed_base,
            tile_config=tile_config,
            activation=activation,
            device=device,
        )

    route_slots = max_packed_route_slots(capacity_m * top_k, block_m, route_num_experts)
    max_m_blocks = (route_slots + block_m - 1) // block_m
    launch = compile_mixed_trellis4(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=tier_slots[0],
        tier1_num_experts=tier_slots[1],
        tier2_num_experts=tier_slots[2],
        tier3_num_experts=tier_slots[3],
        route_num_experts=route_num_experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        tier2_bits=bits[2],
        tier3_bits=bits[3],
        trellis_codebook=codebook,
        swiglu_limit=swiglu_limit,
        activation=activation,
        moe_block_size=block_m,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype=out_dtype_str,
        decode_schedule=decode_schedule,
    )
    assert launch.trellis_codebook == codebook
    if block_m == 16:
        assert launch.moe_block_size == 16 and launch.decode_schedule == "l2=2,pf1=0,pf2=0,pdl=1"
    buffers = make_mixed_trellis3_buffers(launch, device=device, sms=sms)

    rng = torch.Generator(device="cpu").manual_seed(115_000 + bits[0] * 19 + len(case_mode))
    x_full = (torch.randn((capacity_m, hidden), generator=rng) * 0.05).to(
        device=device, dtype=torch.bfloat16
    )
    chosen_t = torch.tensor(chosen_globals, dtype=torch.int32)
    local_ids_cpu = torch.stack(
        [torch.randperm(num_live, generator=rng)[:top_k] for _ in range(capacity_m)]
    )
    ids_full = chosen_t[local_ids_cpu].to(device=device, dtype=torch.int32).contiguous()
    local_ids_ref = local_ids_cpu.to(device=device, dtype=torch.int32).contiguous()
    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    def run_4tier(live_m: int) -> torch.Tensor:
        return _run_mixed_trellis4_launch(
            x_full[:live_m],
            weights_full[:live_m],
            ids_full[:live_m],
            launch=launch,
            tiers=tiers,
            global_to_combined=mapping,
            descriptor_map=descriptor,
            rotations=rotations,
            buffers=buffers,
            gate_counts=gate_counts,
            up_counts=up_counts,
            fc2_counts=fc2_counts,
            route_num_experts=route_num_experts,
        )

    buffers.output.fill_(float("nan"))
    actual_full = run_4tier(capacity_m).clone()
    torch.cuda.synchronize(device)

    expected_full = reference_exl3_routed_moe_bitwise(
        x_full,
        local_ids_ref,
        weights_full,
        gate_w,
        up_w,
        down_w,
        gate_suh_ref,
        up_suh_ref,
        inter_ref,
        down_svh_ref,
        activation=activation,
        swiglu_limit=swiglu_limit,
        output_dtype=out_dtype,
        tile_k=tile_config[0],
    )
    assert_bitwise_equal(actual_full, expected_full, f"mixed4_{case_mode}_{codebook}_{bits}")

    if case_mode == "proj_tiered_m16_gb10_graph_frozen":
        warmup_mixed_trellis_route_pack(launch, buffers, expert_map=mapping)
        with kernel_resolution_guard("frozen mixed_trellis 4-tier"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_4tier(capacity_m)
            for _ in range(2):
                buffers.output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert_bitwise_equal(
                    buffers.output[:capacity_m], expected_full, f"mixed4_graph_{codebook}_{bits}"
                )

            x_saved = x_full.clone()
            w_saved = weights_full.clone()
            try:
                for live_m in (1, 3, capacity_m):
                    x_full.copy_(x_saved)
                    weights_full.copy_(w_saved)
                    if live_m < capacity_m:
                        x_full[live_m:].fill_(float("nan"))
                        weights_full[live_m:].fill_(float("nan"))
                    buffers.output.fill_(float("nan"))
                    short = run_4tier(live_m).clone()
                    torch.cuda.synchronize(device)
                    expected_live = reference_exl3_routed_moe_bitwise(
                        x_saved[:live_m],
                        local_ids_ref[:live_m],
                        w_saved[:live_m],
                        gate_w,
                        up_w,
                        down_w,
                        gate_suh_ref,
                        up_suh_ref,
                        inter_ref,
                        down_svh_ref,
                        activation=activation,
                        swiglu_limit=swiglu_limit,
                        output_dtype=out_dtype,
                        tile_k=tile_config[0],
                    )
                    assert_bitwise_equal(short, expected_live, f"mixed4_frozen_m{live_m}_{codebook}")
            finally:
                x_full.copy_(x_saved)
                weights_full.copy_(w_saved)


def _build_kimi_tp6_fixture(
    *,
    tier1_bits: int,
    codebook: str,
    seed_base: int,
    projection_tiered: bool,
    device: torch.device,
):
    """Build Kimi K3.1 TP6 (3584x512, E896, k16, situ) physical tiers + CPU-decoded Reference(codebook) weights."""
    hidden, intermediate = 3584, 512
    route_num_experts = 896
    tile_config = (64, 256, 64, 256)

    # Decode a compact pool of distinct GPTQModel-packed .trellis projections on CPU,
    # then materialize 22 physical experts (with distinct per-expert SUH/SVH scales) across E896.
    k2_templates = [
        (
            make_gptqmodel_exl3_projection(hidden, intermediate, 2, codebook=codebook, seed=seed_base + i * 3 + 0, device=device),
            make_gptqmodel_exl3_projection(hidden, intermediate, 2, codebook=codebook, seed=seed_base + i * 3 + 1, device=device),
            make_gptqmodel_exl3_projection(intermediate, hidden, 2, codebook=codebook, seed=seed_base + i * 3 + 2, device=device),
        )
        for i in range(3)
    ]
    k_hi_templates = [
        (
            make_gptqmodel_exl3_projection(hidden, intermediate, tier1_bits, codebook=codebook, seed=seed_base + 50 + i * 3 + 0, device=device),
            make_gptqmodel_exl3_projection(hidden, intermediate, tier1_bits, codebook=codebook, seed=seed_base + 50 + i * 3 + 1, device=device),
            make_gptqmodel_exl3_projection(intermediate, hidden, tier1_bits, codebook=codebook, seed=seed_base + 50 + i * 3 + 2, device=device),
        )
        for i in range(2)
    ]
    k2_decoded = [
        (
            decode_exl3_projection_cpu(g, 2),
            decode_exl3_projection_cpu(u, 2),
            decode_exl3_projection_cpu(d, 2),
        )
        for g, u, d in k2_templates
    ]
    k_hi_decoded = [
        (
            decode_exl3_projection_cpu(g, tier1_bits),
            decode_exl3_projection_cpu(u, tier1_bits),
            decode_exl3_projection_cpu(d, tier1_bits),
        )
        for g, u, d in k_hi_templates
    ]

    chosen_globals = [
        0, 1, 2, 3, 63, 127, 255, 383, 447, 510, 511,
        512, 513, 576, 640, 704, 767, 800, 832, 864, 894, 895,
    ]
    num_live = len(chosen_globals)  # 22 = 16 tier0 + 6 tier1
    assert num_live == 22

    if projection_tiered:
        # Per-projection cross-bitrate memberships: 16 slots in tier0 (K2), 6 slots in tier1 (tier1_bits).
        gate_tiers = tuple([0] * 16 + [1] * 6)
        up_tiers = tuple([1, 0, 1, 0, 1, 0] + [0] * 13 + [1, 1, 1])  # 16 tier0, 6 tier1
        down_tiers = tuple([0, 1, 0, 1] + [0] * 14 + [1, 1, 1, 1])   # 16 tier0, 6 tier1
    else:
        gate_tiers = tuple([0] * 16 + [1] * 6)
        up_tiers = gate_tiers
        down_tiers = gate_tiers

    rng_scales = torch.Generator(device="cpu").manual_seed(seed_base + 777)

    def rand_signs(rows: int, cols: int) -> torch.Tensor:
        vals = torch.where(
            torch.rand((rows, cols), generator=rng_scales) < 0.5,
            torch.full((rows, cols), -1.0, dtype=torch.float32),
            torch.full((rows, cols), 1.0, dtype=torch.float32),
        )
        # Include smooth non-unit positive/negative FP16 scaling factors as well.
        mag = 0.75 + 0.5 * torch.rand((rows, cols), generator=rng_scales, dtype=torch.float32)
        return (vals * mag).to(device=device, dtype=torch.float16).contiguous()

    gate_suh = rand_signs(num_live, hidden)
    up_suh = rand_signs(num_live, hidden)
    inter_rot = rand_signs(num_live, 3 * intermediate)
    down_svh = rand_signs(num_live, hidden)

    gate_w: list[torch.Tensor] = []
    up_w: list[torch.Tensor] = []
    down_w: list[torch.Tensor] = []
    g_trellis_by_e: list[torch.Tensor] = []
    u_trellis_by_e: list[torch.Tensor] = []
    d_trellis_by_e: list[torch.Tensor] = []

    for e in range(num_live):
        gt, ut, dt = gate_tiers[e], up_tiers[e], down_tiers[e]
        if gt == 0:
            idx = e % len(k2_templates)
            g_trellis_by_e.append(k2_templates[idx][0].trellis)
            gate_w.append(k2_decoded[idx][0])
        else:
            idx = e % len(k_hi_templates)
            g_trellis_by_e.append(k_hi_templates[idx][0].trellis)
            gate_w.append(k_hi_decoded[idx][0])

        if ut == 0:
            idx = (e + 1) % len(k2_templates)
            u_trellis_by_e.append(k2_templates[idx][1].trellis)
            up_w.append(k2_decoded[idx][1])
        else:
            idx = (e + 1) % len(k_hi_templates)
            u_trellis_by_e.append(k_hi_templates[idx][1].trellis)
            up_w.append(k_hi_decoded[idx][1])

        if dt == 0:
            idx = (e + 2) % len(k2_templates)
            d_trellis_by_e.append(k2_templates[idx][2].trellis)
            down_w.append(k2_decoded[idx][2])
        else:
            idx = (e + 1) % len(k_hi_templates)
            d_trellis_by_e.append(k_hi_templates[idx][2].trellis)
            down_w.append(k_hi_decoded[idx][2])

    tier_slots = (16, 6)
    prepared_tiers = []
    for t_idx, (t_bits, slots_t) in enumerate(((2, 16), (tier1_bits, 6))):
        g_ids = [e for e in range(num_live) if gate_tiers[e] == t_idx]
        u_ids = [e for e in range(num_live) if up_tiers[e] == t_idx]
        d_ids = [e for e in range(num_live) if down_tiers[e] == t_idx]
        assert len(g_ids) == slots_t and len(u_ids) == slots_t and len(d_ids) == slots_t
        w13_t = torch.stack(
            (
                torch.stack([g_trellis_by_e[e] for e in g_ids], dim=0),
                torch.stack([u_trellis_by_e[e] for e in u_ids], dim=0),
            ),
            dim=0,
        ).contiguous()
        w2_t = torch.stack([d_trellis_by_e[e] for e in d_ids], dim=0).contiguous()
        prepared_tiers.append(
            prepare_trellis256_moe_weights(
                w13_t,
                w2_t,
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=slots_t,
                activation="situ",
                fc1_tile_n=tile_config[1],
                fc2_tile_n=tile_config[3],
                params_dtype=torch.float16,
                w13_layout="trellis_t256_proj",
                trellis_bits=t_bits,
                codebook=codebook,
                gate_suh=gate_suh[:slots_t].contiguous(),
                up_suh=up_suh[:slots_t].contiguous(),
                intermediate_rotations=inter_rot[:slots_t].contiguous(),
                down_svh=down_svh[:slots_t].contiguous(),
                tile_config=tile_config,
            )
        )

    desc_bits = _descriptor_bits(route_num_experts)
    assert desc_bits == 10
    rows: list[int] = []
    for proj_tiers in (gate_tiers, up_tiers, down_tiers):
        counters = [0, 0]
        for t_id in proj_tiers:
            loc = counters[t_id]
            counters[t_id] += 1
            rows.append((t_id << desc_bits) | loc)
    descriptor = torch.tensor(rows, dtype=torch.int32, device=device).contiguous()
    descriptor._mt_projection_counts = (tier_slots, tier_slots)

    route_map = torch.full((route_num_experts,), -1, dtype=torch.int32, device=device)
    for c_idx, g_exp in enumerate(chosen_globals):
        route_map[g_exp] = c_idx

    rotations = MixedTrellisRotations(
        intermediate=inter_rot,
        gate_suh=gate_suh,
        up_suh=up_suh,
        down_svh=down_svh,
    )
    return (
        tuple(prepared_tiers),
        route_map,
        descriptor,
        rotations,
        chosen_globals,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    "kimi_case",
    [
        "m1_and_m80_spans_gb10",
        "m80_proj_tiered_k2_k4_fp32",
    ],
)
def test_mixed_trellis_kimi_tp6_bitwise_matches_reference(
    codebook: str,
    kimi_case: str,
) -> None:
    """Bitwise oracle for W4A16MixedTrellisKimiTP6 across m1, m80 span kinds {1,2,3,4}, E896 10-bit descriptors, prefix invariance, graph replay, and cooperative parity."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    hidden, intermediate = 3584, 512
    route_num_experts, top_k = 896, 16
    tile_config = (64, 256, 64, 256)
    proj_tiered = kimi_case == "m80_proj_tiered_k2_k4_fp32"
    tier1_bits = 4 if proj_tiered else 3
    out_dtype_str = "fp32" if proj_tiered else "bf16"
    out_dtype = torch.float32 if proj_tiered else torch.bfloat16
    decode_schedule = None if proj_tiered else "gb10"
    seed_base = 130_000 + (500 if proj_tiered else 0) + (1 if codebook == "mul1" else 0)

    (
        tiers,
        route_map,
        descriptor,
        rotations,
        chosen_globals,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
    ) = _build_kimi_tp6_fixture(
        tier1_bits=tier1_bits,
        codebook=codebook,
        seed_base=seed_base,
        projection_tiered=proj_tiered,
        device=device,
    )

    capacity_m = 80
    route_slots_80 = max_packed_route_slots(capacity_m * top_k, 8, route_num_experts)
    max_m_blocks_80 = (route_slots_80 + 7) // 8

    common_80 = dict(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=16,
        tier1_num_experts=6,
        route_num_experts=route_num_experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks_80,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tile_config,
        tier0_bits=2,
        tier1_bits=tier1_bits,
        trellis_codebook=codebook,
        activation="situ",
        swiglu_limit=None,
        moe_block_size=8,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype=out_dtype_str,
        decode_schedule=decode_schedule,
    )
    kimi_80 = compile_mixed_trellis(**common_80, _kimi_tp6_override=True)
    coop_80 = compile_mixed_trellis(**common_80, _kimi_tp6_override=False)
    assert kimi_80.trellis_codebook == codebook
    assert kimi_80.compiled is not coop_80.compiled

    kimi_80_bind = bind_mixed_trellis(
        tiers[0], tiers[1], route_map, descriptor, rotations, kimi_80, gate_experts=(16, 6), up_experts=(16, 6)
    )
    coop_80_bind = bind_mixed_trellis(
        tiers[0], tiers[1], route_map, descriptor, rotations, coop_80, gate_experts=(16, 6), up_experts=(16, 6)
    )
    kimi_80_buf = make_mixed_trellis_buffers(kimi_80, device=device, sms=sms)
    coop_80_buf = make_mixed_trellis_buffers(coop_80, device=device, sms=sms)
    warmup_mixed_trellis_route_pack(kimi_80, kimi_80_buf, expert_map=route_map)
    warmup_mixed_trellis_route_pack(coop_80, coop_80_buf, expert_map=route_map)

    rng = torch.Generator(device="cpu").manual_seed(135_000 + len(kimi_case))
    test_m = 12 if proj_tiered else 28
    x_full = (torch.randn((capacity_m, hidden), generator=rng) * 0.03).to(
        device=device, dtype=torch.bfloat16
    )

    # Construct per-row top-16 local expert selections (0..21) so m=28 simultaneously hits
    # span_len in {1, 2, 3, 4} on tier0 (K2) and span_len in {1, 2, 3} on tier1 (K3/K4).
    chosen_t = torch.tensor(chosen_globals, dtype=torch.int32)
    pool_locals = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21]
    local_rows: list[torch.Tensor] = []
    for r in range(capacity_m):
        fixed: list[int] = [0]  # 28/28 rows at live_m=28 -> 4 M8 blocks (span_len=4)
        if r < 20:
            fixed.extend((1, 16))  # 20 rows -> 3 M8 blocks (span_len=3 on tier0 & tier1)
        if r < 12:
            fixed.extend((2, 17))  # 12 rows -> 2 M8 blocks (span_len=2 on tier0 & tier1)
        if r < 5:
            fixed.extend((3, 18))  # 5 rows -> 1 M8 block (span_len=1 on tier0 & tier1)
        need = top_k - len(fixed)
        perm = torch.randperm(len(pool_locals), generator=rng)[:need].tolist()
        chosen_row = fixed + [pool_locals[idx] for idx in perm]
        order = torch.randperm(top_k, generator=rng)
        local_rows.append(torch.tensor(chosen_row, dtype=torch.int32)[order])
    local_ids_cpu = torch.stack(local_rows, dim=0)
    ids_full = chosen_t[local_ids_cpu].to(device=device, dtype=torch.int32).contiguous()
    local_ids_ref = local_ids_cpu.to(device=device, dtype=torch.int32).contiguous()

    # Include unrouted -1 sentinels on row 2 slots 13..15 to exercise live-route masking.
    ids_full[2, 13:] = -1
    local_ids_ref[2, 13:] = -1

    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    if proj_tiered:
        kimi_80_buf.output.fill_(float("nan"))
        coop_80_buf.output.fill_(float("nan"))
        actual = run_bound_mixed_trellis(
            x_full[:test_m], weights_full[:test_m], ids_full[:test_m], kimi_80_bind, kimi_80_buf
        ).clone()
        coop_out = run_bound_mixed_trellis(
            x_full[:test_m], weights_full[:test_m], ids_full[:test_m], coop_80_bind, coop_80_buf
        ).clone()
        torch.cuda.synchronize(device)
        expected = reference_exl3_routed_moe_bitwise(
            x_full[:test_m],
            local_ids_ref[:test_m],
            weights_full[:test_m],
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
            activation="situ",
            output_dtype=out_dtype,
            tile_k=tile_config[0],
        )
        assert_bitwise_equal(actual, coop_out, f"kimi_proj_vs_coop_{codebook}")
        assert_bitwise_equal(actual, expected, f"kimi_proj_vs_cpu_{codebook}")
        return

    route_slots_1 = max_packed_route_slots(top_k, 8, route_num_experts)
    kimi_1 = compile_mixed_trellis(
        **{**common_80, "size_m": 1, "max_m_blocks": (route_slots_1 + 7) // 8},
        _kimi_tp6_override=True,
    )
    kimi_1_bind = bind_mixed_trellis(
        tiers[0], tiers[1], route_map, descriptor, rotations, kimi_1, gate_experts=(16, 6), up_experts=(16, 6)
    )
    kimi_1_buf = make_mixed_trellis_buffers(kimi_1, device=device, sms=sms)
    warmup_mixed_trellis_route_pack(kimi_1, kimi_1_buf, expert_map=route_map)

    with kernel_resolution_guard("frozen kimi_tp6 m1 + m80"):
        x_saved = x_full.clone()
        w_saved = weights_full.clone()

        kimi_1_buf.output.fill_(float("nan"))
        out_m1 = run_bound_mixed_trellis(
            x_saved[:1], w_saved[:1], ids_full[:1], kimi_1_bind, kimi_1_buf
        ).clone()
        torch.cuda.synchronize(device)

        row0_outputs = [out_m1]
        expected_28 = None
        try:
            for live_m in (1, 3, test_m):
                x_full.copy_(x_saved)
                weights_full.copy_(w_saved)
                if live_m < capacity_m:
                    x_full[live_m:].fill_(float("nan"))
                    weights_full[live_m:].fill_(float("nan"))
                kimi_80_buf.output.fill_(float("nan"))
                actual = run_bound_mixed_trellis(
                    x_full[:live_m], weights_full[:live_m], ids_full[:live_m], kimi_80_bind, kimi_80_buf
                ).clone()
                torch.cuda.synchronize(device)
                expected = reference_exl3_routed_moe_bitwise(
                    x_saved[:live_m],
                    local_ids_ref[:live_m],
                    w_saved[:live_m],
                    gate_w,
                    up_w,
                    down_w,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    activation="situ",
                    output_dtype=out_dtype,
                    tile_k=tile_config[0],
                )
                assert_bitwise_equal(actual, expected, f"kimi_m80_live{live_m}_{codebook}")
                row0_outputs.append(actual[:1].clone())
                if live_m == test_m:
                    expected_28 = expected
                    coop_80_buf.output.fill_(float("nan"))
                    coop_actual = run_bound_mixed_trellis(
                        x_saved[:live_m], w_saved[:live_m], ids_full[:live_m], coop_80_bind, coop_80_buf
                    ).clone()
                    torch.cuda.synchronize(device)
                    assert_bitwise_equal(actual, coop_actual, f"kimi_m80_vs_coop_{codebook}")
        finally:
            x_full.copy_(x_saved)
            weights_full.copy_(w_saved)

        for other in row0_outputs[1:]:
            assert_bitwise_equal(other, out_m1, f"kimi_row0_invariance_{codebook}")

        assert expected_28 is not None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_bound_mixed_trellis(
                x_full[:test_m], weights_full[:test_m], ids_full[:test_m], kimi_80_bind, kimi_80_buf
            )
        for _ in range(2):
            kimi_80_buf.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert_bitwise_equal(
                kimi_80_buf.output[:test_m], expected_28, f"kimi_m80_graph_{codebook}"
            )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(torch.cuda.current_device()) != (12, 1),
    reason="requires Spark SM121 for automatic use_kimi_tp6 dispatch",
)
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
def test_mixed_trellis_kimi_tp6_sm121_auto_selector_bitwise(codebook: str) -> None:
    """Verify automatic SM121 use_kimi_tp6 dispatch on Kimi K3.1 TP6 geometry (3584, 512, E=896, k=16, K2/K3, situ, gb10)."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    (
        tiers,
        route_map,
        descriptor,
        rotations,
        chosen_globals,
        gate_w,
        up_w,
        down_w,
        gate_suh,
        up_suh,
        inter_rot,
        down_svh,
    ) = _build_kimi_tp6_fixture(
        tier1_bits=3,
        codebook=codebook,
        seed_base=140_000 + (1 if codebook == "mul1" else 0),
        projection_tiered=False,
        device=device,
    )

    for cap_m, live_m in ((1, 1), (80, 12)):
        route_slots = max_packed_route_slots(cap_m * 16, 8, 896)
        common = dict(
            size_m=cap_m,
            hidden_size=3584,
            intermediate_size=512,
            tier0_num_experts=16,
            tier1_num_experts=6,
            route_num_experts=896,
            top_k=16,
            max_m_blocks=(route_slots + 7) // 8,
            sms=sms,
            max_shared_mem=max_shared_mem,
            force_tile_config=(64, 256, 64, 256),
            tier0_bits=2,
            tier1_bits=3,
            trellis_codebook=codebook,
            activation="situ",
            swiglu_limit=None,
            moe_block_size=8,
            rotation_input_dtype="bf16",
            full_rotation_output_dtype="bf16",
            decode_schedule="gb10",
        )
        auto_launch = compile_mixed_trellis(**common)
        explicit_kimi = compile_mixed_trellis(**common, _kimi_tp6_override=True)
        coop_launch = compile_mixed_trellis(**common, _kimi_tp6_override=False)
        assert auto_launch.compiled is explicit_kimi.compiled
        assert auto_launch.compiled is not coop_launch.compiled

        rng = torch.Generator(device="cpu").manual_seed(145_000 + cap_m)
        x = (torch.randn((live_m, 3584), generator=rng) * 0.03).to(device=device, dtype=torch.bfloat16)
        chosen_t = torch.tensor(chosen_globals, dtype=torch.int32)
        local_ids_cpu = torch.stack([torch.randperm(22, generator=rng)[:16] for _ in range(live_m)])
        ids = chosen_t[local_ids_cpu].to(device=device, dtype=torch.int32).contiguous()
        local_ids_ref = local_ids_cpu.to(device=device, dtype=torch.int32).contiguous()
        weights = torch.softmax(torch.randn((live_m, 16), generator=rng, dtype=torch.float32), dim=-1).to(
            device=device
        )

        auto_bind = bind_mixed_trellis(tiers[0], tiers[1], route_map, descriptor, rotations, auto_launch)
        coop_bind = bind_mixed_trellis(tiers[0], tiers[1], route_map, descriptor, rotations, coop_launch)
        auto_buf = make_mixed_trellis_buffers(auto_launch, device=device, sms=sms)
        coop_buf = make_mixed_trellis_buffers(coop_launch, device=device, sms=sms)

        auto_out = run_bound_mixed_trellis(x, weights, ids, auto_bind, auto_buf).clone()
        coop_out = run_bound_mixed_trellis(x, weights, ids, coop_bind, coop_buf).clone()
        torch.cuda.synchronize(device)
        expected = reference_exl3_routed_moe_bitwise(
            x,
            local_ids_ref,
            weights,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
            activation="situ",
            output_dtype=torch.bfloat16,
            tile_k=64,
        )
        assert_bitwise_equal(auto_out, coop_out, f"sm121_auto_kimi_m{cap_m}_vs_coop_{codebook}")
        assert_bitwise_equal(auto_out, expected, f"sm121_auto_kimi_m{cap_m}_vs_cpu_{codebook}")


_REF_A8_QUANT_CACHE: dict[int, object] = {}


def _quantize_a8_blocks_exact(x_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-independent per-H128-block INT8 activation quantizer (no ws_a8 imports)."""
    rows, cols = x_f32.shape
    assert cols % 128 == 0
    nblk = cols // 128
    total_blocks = rows * nblk
    q_i32 = torch.empty((total_blocks, 128), dtype=torch.int32, device=x_f32.device)
    scales_f32 = torch.empty((total_blocks,), dtype=torch.float32, device=x_f32.device)

    in_flat = x_f32.contiguous().reshape(-1)
    q_flat = q_i32.reshape(-1)
    s_flat = scales_f32.reshape(-1)

    if total_blocks not in _REF_A8_QUANT_CACHE:
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T, dsl_user_op

        @dsl_user_op
        def _fma_magic_low_s8_i32(val: Float32, inv: Float32, *, loc=None, ip=None) -> Int32:
            return Int32(
                llvm.inline_asm(
                    T.i32(),
                    [Float32(val).ir_value(loc=loc, ip=ip), Float32(inv).ir_value(loc=loc, ip=ip)],
                    "{ .reg .f32 f; .reg .b32 b;\n"
                    "fma.rn.f32 f, $1, $2, 0f4B400000;\n"
                    "mov.b32 b, f;\n"
                    "cvt.s32.s8 $0, b; }",
                    "=r,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc,
                    ip=ip,
                )
            )

        @cute.kernel
        def _a8_quant_kernel(src: cute.Tensor, q_dst: cute.Tensor, s_dst: cute.Tensor):
            lane, _, _ = cute.arch.thread_idx()
            blk, _, _ = cute.arch.block_idx()
            base = blk * Int32(128) + lane * Int32(4)
            v0 = src[base + Int32(0)].to(Float32)
            v1 = src[base + Int32(1)].to(Float32)
            v2 = src[base + Int32(2)].to(Float32)
            v3 = src[base + Int32(3)].to(Float32)
            a0 = -v0 if v0 < Float32(0.0) else v0
            a1 = -v1 if v1 < Float32(0.0) else v1
            a2 = -v2 if v2 < Float32(0.0) else v2
            a3 = -v3 if v3 < Float32(0.0) else v3
            amax = a0 if a0 >= a1 else a1
            if a2 > amax:
                amax = a2
            if a3 > amax:
                amax = a3
            for i in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_bfly(amax, offset=1 << i)
                if other > amax:
                    amax = other
            inv = Float32(0.0)
            if amax > Float32(0.0):
                inv = Float32(127.0) / amax
            q_dst[base + Int32(0)] = _fma_magic_low_s8_i32(v0, inv)
            q_dst[base + Int32(1)] = _fma_magic_low_s8_i32(v1, inv)
            q_dst[base + Int32(2)] = _fma_magic_low_s8_i32(v2, inv)
            q_dst[base + Int32(3)] = _fma_magic_low_s8_i32(v3, inv)
            if lane == Int32(0):
                s_dst[blk] = amax * Float32(1.0 / 127.0)

        @cute.jit
        def _launch_a8_quant(src: cute.Tensor, q_dst: cute.Tensor, s_dst: cute.Tensor, stream):
            _a8_quant_kernel(src, q_dst, s_dst).launch(
                grid=(total_blocks, 1, 1),
                block=(32, 1, 1),
                stream=stream,
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _REF_A8_QUANT_CACHE[total_blocks] = cute.compile(
            _launch_a8_quant,
            from_dlpack(in_flat),
            from_dlpack(q_flat),
            from_dlpack(s_flat),
            stream,
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _REF_A8_QUANT_CACHE[total_blocks](
        from_dlpack(in_flat), from_dlpack(q_flat), from_dlpack(s_flat), stream
    )
    return q_i32.reshape(rows, cols), scales_f32.reshape(rows, nblk)


def _run_exact_a8_gemm_f16(
    q_a_i32: torch.Tensor,
    s_a_f32: torch.Tensor,
    w_f16: torch.Tensor,
) -> torch.Tensor:
    """Exact per-H128-block INT8 dot product + magic-bit float rescale + /32 FP16 store."""
    rows, k = q_a_i32.shape
    wk, n = w_f16.shape
    assert wk == k and k % 128 == 0
    # round(32 * w) via exact IEEE FP16 FMA(w, 32.0, 1536.0) low mantissa byte.
    w_fma_i16 = (w_f16.to(torch.float64) * 32.0 + 1536.0).to(torch.float16).view(torch.int16)
    w_s8_i32 = (w_fma_i16.to(torch.int32) & 0xFF).to(torch.uint8).view(torch.int8).to(torch.int32)

    facc = torch.zeros((rows, n), dtype=torch.float32, device=q_a_i32.device)
    for b in range(k // 128):
        qa_b = q_a_i32[:, 128 * b : 128 * (b + 1)].to(torch.float64)
        wb_b = w_s8_i32[128 * b : 128 * (b + 1), :].to(torch.float64)
        # |dot| <= 128 * 127 * 127 = 2,064,512 << 2^53, so float64 matmul is integer-exact.
        dot_i32 = torch.matmul(qa_b, wb_b).to(torch.int64).to(torch.int32)
        bits_i32 = (dot_i32 + 0x4B400000).to(torch.int32)
        t_f32 = bits_i32.view(torch.float32) - 12582912.0
        facc = torch.addcmul(facc, t_f32, s_a_f32[:, b : b + 1])
    return ((facc + 0.0) * 0.03125).to(torch.float16)


def _decode_wire_e4m3_k32_to_f16(wire: torch.Tensor, hidden: int) -> torch.Tensor:
    """Decode [M, H + H/32] E4M3 + UE8M0 wire rows to the exact FP16 values widened by WS."""
    rows = int(wire.shape[0])
    e4m3 = wire[:, :hidden].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    s_u8 = wire[:, hidden:].to(torch.int32)
    s_bits = s_u8 << 23
    s_bits = torch.where(s_u8 == 0, torch.full_like(s_bits, 0x00400000), s_bits)
    s_bits = torch.where(s_u8 == 255, torch.full_like(s_bits, 0x7FFFFFFF), s_bits)
    scales = s_bits.view(torch.float32)
    return (e4m3.reshape(rows, hidden // 32, 32) * scales.unsqueeze(-1)).to(torch.float16).reshape(rows, hidden)


def reference_exl3_ws_a8_routed_moe_bitwise(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_weights_f16: list[torch.Tensor],
    up_weights_f16: list[torch.Tensor],
    down_weights_f16: list[torch.Tensor],
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
    *,
    input_format: str = "bf16",
    activation: str = "silu",
    swiglu_limit: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Bitwise reference for warp-specialized EXL3 A8 over CPU-decoded Reference(codebook) FP16 weights."""
    device = x.device
    m = int(x.shape[0])
    hidden = int(gate_weights_f16[0].shape[0])
    _, top_k = topk_ids.shape

    if input_format == "e4m3_k32":
        x_base_f16 = _decode_wire_e4m3_k32_to_f16(x, hidden)
    else:
        x_base_f16 = x.to(torch.float16)

    route_outputs = torch.zeros((m, top_k, hidden), dtype=torch.float32, device=device)
    ids_cpu = topk_ids.detach().cpu().numpy()

    for expert in sorted({int(e) for e in ids_cpu.reshape(-1) if int(e) >= 0}):
        positions = np.argwhere(ids_cpu == expert)
        token_idx = torch.from_numpy(positions[:, 0]).to(device=device, dtype=torch.long)
        slot_idx = torch.from_numpy(positions[:, 1]).to(device=device, dtype=torch.long)

        x_e = x_base_f16.index_select(0, token_idx)
        g_suh = gate_suh[0 if gate_suh.shape[0] == 1 else expert]
        u_suh = up_suh[0 if up_suh.shape[0] == 1 else expert]

        rot_g_f32 = _run_exact_had128_f32((x_e * g_suh).to(torch.float16).to(torch.float32))
        rot_u_f32 = _run_exact_had128_f32((x_e * u_suh).to(torch.float16).to(torch.float32))
        qg, sg = _quantize_a8_blocks_exact(rot_g_f32)
        qu, su = _quantize_a8_blocks_exact(rot_u_f32)

        w_g = gate_weights_f16[expert].to(device=device)
        w_u = up_weights_f16[expert].to(device=device)
        w_d = down_weights_f16[expert].to(device=device)

        fc1_g = _run_exact_a8_gemm_f16(qg, sg, w_g)
        fc1_u = _run_exact_a8_gemm_f16(qu, su, w_u)

        act_f32 = _run_exact_inter_rot_act_f16(
            fc1_g,
            fc1_u,
            intermediate_rotations[expert],
            activation=activation,
            swiglu_limit=swiglu_limit,
            output_f32=True,
        )
        qa, sa = _quantize_a8_blocks_exact(act_f32)
        fc2_f16 = _run_exact_a8_gemm_f16(qa, sa, w_d)

        d_svh_f16 = down_svh[0 if down_svh.shape[0] == 1 else expert]
        fc2_rot = _run_exact_fc2_rot_f32(fc2_f16, d_svh_f16)
        route_outputs[token_idx, slot_idx] = fc2_rot

    acc = torch.zeros((m, hidden), dtype=torch.float32, device=device)
    w_f32 = topk_weights.to(torch.float32)
    for slot in range(top_k):
        acc = torch.addcmul(acc, route_outputs[:, slot], w_f32[:, slot : slot + 1])
    return acc.to(output_dtype)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    "a8_case",
    [
        "bf16_k2_k3_silu_m16",
        "bf16_k3_k4_graph_and_frozen_live_rows",
        "wire_e4m3_k4_k5_tiles192x256_clamp",
        "bf16_k5_k6_proj_tiered_situ_fp32_m32",
    ],
)
def test_mixed_trellis_ws_a8_bitwise_matches_reference(
    codebook: str,
    a8_case: str,
) -> None:
    """Bitwise oracle for warp-specialized A8 W4A16MixedTrellisWSKernel across K2..K6, BF16 & E4M3 wire inputs, 192x256 tiles, dynamic claims, graph replay, and frozen live rows."""
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)

    if a8_case == "bf16_k2_k3_silu_m16":
        bits = (2, 3)
        input_format = "bf16"
        hidden, intermediate = 512, 256
        tiles = (64, 128, 64, 128)
        block_m = 16
        capacity_m = 32
        activation = "silu"
        swiglu_limit = None
        out_dtype_str = "bf16"
        out_dtype = torch.bfloat16
        dynamic_tiles = False
        proj_tiered = False
    elif a8_case == "bf16_k3_k4_graph_and_frozen_live_rows":
        bits = (3, 4)
        input_format = "bf16"
        hidden, intermediate = 512, 256
        tiles = (64, 128, 64, 128)
        block_m = 16
        capacity_m = 64
        activation = "silu"
        swiglu_limit = None
        out_dtype_str = "bf16"
        out_dtype = torch.bfloat16
        dynamic_tiles = True
        proj_tiered = False
    elif a8_case == "wire_e4m3_k4_k5_tiles192x256_clamp":
        bits = (4, 5)
        input_format = "e4m3_k32"
        hidden, intermediate = 512, 384
        tiles = (64, 192, 64, 256)
        block_m = 16
        capacity_m = 48
        activation = "silu_post_clamp"
        swiglu_limit = 7.0
        out_dtype_str = "bf16"
        out_dtype = torch.bfloat16
        dynamic_tiles = False
        proj_tiered = False
    else:
        assert a8_case == "bf16_k5_k6_proj_tiered_situ_fp32_m32"
        bits = (5, 6)
        input_format = "bf16"
        hidden, intermediate = 512, 256
        tiles = (64, 128, 64, 128)
        block_m = 32
        capacity_m = 32
        activation = "situ"
        swiglu_limit = None
        out_dtype_str = "fp32"
        out_dtype = torch.float32
        dynamic_tiles = False
        proj_tiered = True

    experts, top_k = 4, 2
    prep_tiles = (64, 128, 64, 128)
    seed_base = 150_000 + bits[0] * 100 + bits[1] * 10 + (1 if codebook == "mul1" else 0)

    if proj_tiered:
        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
        ) = _build_projection_tiered_fixture(
            hidden=hidden,
            intermediate=intermediate,
            tier_bits=bits,
            gate_tiers=(0, 1, 0, 1),
            up_tiers=(1, 0, 0, 1),
            down_tiers=(0, 0, 1, 1),
            codebook=codebook,
            seed_base=seed_base,
            tile_config=prep_tiles,
            activation=activation,
            device=device,
        )
    else:
        (
            tiers,
            mapping,
            descriptor,
            rotations,
            gate_w,
            up_w,
            down_w,
            gate_suh,
            up_suh,
            inter_rot,
            down_svh,
        ) = _build_disjoint_2tier_fixture(
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            codebook=codebook,
            seed_base=seed_base,
            tile_config=prep_tiles,
            activation=activation,
            device=device,
        )

    route_slots = max_packed_route_slots(capacity_m * top_k, block_m, experts)
    max_m_blocks = (route_slots + block_m - 1) // block_m
    common_compile = dict(
        size_m=capacity_m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        route_num_experts=experts,
        top_k=top_k,
        max_m_blocks=max_m_blocks,
        sms=sms,
        max_shared_mem=max_shared_mem,
        force_tile_config=tiles,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        trellis_codebook=codebook,
        swiglu_limit=swiglu_limit,
        activation=activation,
        moe_block_size=block_m,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype=out_dtype_str,
        warp_specialized=True,
        input_format=input_format,
        ws_dynamic_tiles=dynamic_tiles,
    )
    a8_launch = compile_mixed_trellis(**common_compile, activations="a8")
    a16_launch = compile_mixed_trellis(**common_compile, activations="a16")
    assert a8_launch.trellis_codebook == codebook
    assert a8_launch.activations == "a8"
    assert a8_launch.compiled is not a16_launch.compiled

    a8_bind = bind_mixed_trellis(tiers[0], tiers[1], mapping, descriptor, rotations, a8_launch)
    a16_bind = bind_mixed_trellis(tiers[0], tiers[1], mapping, descriptor, rotations, a16_launch)
    a8_buf = make_mixed_trellis_buffers(a8_launch, device=device, sms=sms)
    a16_buf = make_mixed_trellis_buffers(a16_launch, device=device, sms=sms)
    warmup_mixed_trellis_route_pack(a8_launch, a8_buf, expert_map=mapping)
    warmup_mixed_trellis_route_pack(a16_launch, a16_buf, expert_map=mapping)

    rng = torch.Generator(device="cpu").manual_seed(155_000 + bits[0] * 19 + len(a8_case))
    x_bf16 = (torch.randn((capacity_m, hidden), generator=rng) * 0.04).to(
        device=device, dtype=torch.bfloat16
    )
    ids_full = torch.stack(
        [torch.randperm(experts, generator=rng)[:top_k] for _ in range(capacity_m)]
    ).to(device=device, dtype=torch.int32)
    weights_full = torch.softmax(
        torch.randn((capacity_m, top_k), generator=rng, dtype=torch.float32), dim=-1
    ).to(device=device)

    def make_operand(src_bf16: torch.Tensor) -> torch.Tensor:
        if input_format == "bf16":
            return src_bf16.contiguous()
        rows = int(src_bf16.shape[0])
        groups = src_bf16.float().reshape(rows, hidden // 32, 32)
        exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(2.0**-100) / 448.0)).clamp(-127, 127)
        vals = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
        return torch.cat(
            (vals.view(torch.uint8).reshape(rows, hidden), (exponent + 127).to(torch.uint8)),
            dim=1,
        ).contiguous()

    live_counts = (
        (1, 17, capacity_m)
        if a8_case in ("bf16_k3_k4_graph_and_frozen_live_rows", "wire_e4m3_k4_k5_tiles192x256_clamp")
        else (capacity_m,)
    )
    operands = {m_live: make_operand(x_bf16[:m_live]) for m_live in live_counts}

    with kernel_resolution_guard(f"frozen ws_a8 {a8_case}"):
        w_saved = weights_full.clone()
        try:
            for live_m in live_counts:
                weights_full.copy_(w_saved)
                if live_m < capacity_m:
                    weights_full[live_m:].fill_(float("nan"))
                a8_buf.output.fill_(float("nan"))
                a16_buf.output.fill_(float("nan"))
                actual_a8 = run_bound_mixed_trellis(
                    operands[live_m], weights_full[:live_m], ids_full[:live_m], a8_bind, a8_buf
                ).clone()
                actual_a16 = run_bound_mixed_trellis(
                    operands[live_m], weights_full[:live_m], ids_full[:live_m], a16_bind, a16_buf
                ).clone()
                torch.cuda.synchronize(device)

                expected_a8 = reference_exl3_ws_a8_routed_moe_bitwise(
                    operands[live_m],
                    ids_full[:live_m],
                    w_saved[:live_m],
                    gate_w,
                    up_w,
                    down_w,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    input_format=input_format,
                    activation=activation,
                    swiglu_limit=swiglu_limit,
                    output_dtype=out_dtype,
                )
                assert_bitwise_equal(actual_a8, expected_a8, f"ws_a8_{a8_case}_m{live_m}_{codebook}")

                rel = (actual_a8.float() - actual_a16.float()).norm() / actual_a16.float().norm().clamp_min(1e-12)
                cos = torch.nn.functional.cosine_similarity(
                    actual_a8.float().flatten(), actual_a16.float().flatten(), dim=0
                )
                assert float(rel.item()) < 0.045, f"ws_a8 vs a16 rel={rel.item()}"
                assert float(cos.item()) > 0.999, f"ws_a8 vs a16 cos={cos.item()}"
        finally:
            weights_full.copy_(w_saved)

        if 17 in operands:
            expected_17 = reference_exl3_ws_a8_routed_moe_bitwise(
                operands[17],
                ids_full[:17],
                weights_full[:17],
                gate_w,
                up_w,
                down_w,
                gate_suh,
                up_suh,
                inter_rot,
                down_svh,
                input_format=input_format,
                activation=activation,
                swiglu_limit=swiglu_limit,
                output_dtype=out_dtype,
            )
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    run_bound_mixed_trellis(
                        operands[17], weights_full[:17], ids_full[:17], a8_bind, a8_buf
                    )
                for _ in range(2):
                    a8_buf.output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert_bitwise_equal(
                        a8_buf.output[:17], expected_17, f"ws_a8_graph_{a8_case}_{codebook}"
                    )
            finally:
                graph.reset()


def _make_balanced_routes(
    m: int,
    top_k: int,
    num_experts: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate distinct per-token ``int32`` route IDs and normalized ``float32`` weights."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.empty((m, top_k), dtype=torch.int32)
    for row in range(m):
        ids[row] = torch.randperm(num_experts, generator=gen)[:top_k].to(torch.int32)
    weights = torch.softmax(torch.randn((m, top_k), generator=gen, dtype=torch.float32), dim=-1)
    return ids.to(device=device).contiguous(), weights.to(device=device).contiguous()


def _decode_native_trellis_tensor_cpu(
    trellis_i16: torch.Tensor, bits: int, codebook: str
) -> torch.Tensor:
    """Decode native ``[K/16, N/16, 16*bits]`` int16 tiles to FP16 ``[K, N]`` on CPU."""
    raw = trellis_i16.detach().cpu().contiguous().numpy()
    k16, n16, width = raw.shape
    assert width == 16 * bits
    tiles = Reference(codebook).decode_tiles(raw.tobytes(), bits)
    return torch.from_numpy(
        tiles.astype(np.float16)
        .reshape(k16, n16, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(k16 * 16, n16 * 16)
        .copy()
    )


def _decode_btx_payloads_to_dense(
    payloads,
    *,
    num_experts: int,
    hidden: int,
    intermediate: int,
    codebook: str,
    uniform_bits: int | None,
    rates_fc1: torch.Tensor | None = None,
    rates_fc2: torch.Tensor | None = None,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Decode BTX plane words to ``(gate_w, up_w, down_w)`` FP16 lists on ``device``."""
    from b12x.moe._shared.btx_schema import rate_code_bits

    slots = intermediate // 32
    hidden_tiles = hidden // 16
    gate_list: list[torch.Tensor] = []
    up_list: list[torch.Tensor] = []
    down_list: list[torch.Tensor] = []
    for expert in range(num_experts):
        if uniform_bits is not None:
            for matrix, out_list in ((0, gate_list), (1, up_list)):
                tiles = torch.empty(
                    (hidden_tiles, 2 * slots, 16 * uniform_bits), dtype=torch.int16
                )
                for slot in range(slots):
                    low, high = payloads.planes[(expert, slot, matrix)]
                    tiles[:, 2 * slot] = low
                    tiles[:, 2 * slot + 1] = high
                out_list.append(
                    _decode_native_trellis_tensor_cpu(tiles, uniform_bits, codebook).to(device)
                )
            d_tiles = torch.empty(
                (2 * slots, hidden_tiles, 16 * uniform_bits), dtype=torch.int16
            )
            for slot in range(slots):
                low, high = payloads.planes[(expert, slot, 2)]
                d_tiles[2 * slot] = low
                d_tiles[2 * slot + 1] = high
            down_list.append(
                _decode_native_trellis_tensor_cpu(d_tiles, uniform_bits, codebook).to(device)
            )
        else:
            assert rates_fc1 is not None and rates_fc2 is not None
            assert slots == 8
            fc1_lo, fc1_hi = rate_code_bits(int(rates_fc1[0, expert]))
            fc2_lo, fc2_hi = rate_code_bits(int(rates_fc2[0, expert]))
            for matrix, out_list in ((0, gate_list), (1, up_list)):
                low_k8 = torch.stack(
                    [payloads.planes[(expert, s, matrix)][0] for s in range(8)], dim=1
                )
                high_k8 = torch.stack(
                    [payloads.planes[(expert, s, matrix)][1] for s in range(8)], dim=1
                )
                dec_lo = _decode_native_trellis_tensor_cpu(low_k8, fc1_lo, codebook)
                dec_hi = _decode_native_trellis_tensor_cpu(high_k8, fc1_hi, codebook)
                out_list.append(torch.cat((dec_lo, dec_hi), dim=1).to(device))
            low_8n = torch.stack(
                [payloads.planes[(expert, s, 2)][0] for s in range(8)], dim=0
            )
            high_8n = torch.stack(
                [payloads.planes[(expert, s, 2)][1] for s in range(8)], dim=0
            )
            dec_d_lo = _decode_native_trellis_tensor_cpu(low_8n, fc2_lo, codebook)
            dec_d_hi = _decode_native_trellis_tensor_cpu(high_8n, fc2_hi, codebook)
            down_list.append(torch.cat((dec_d_lo, dec_d_hi), dim=0).to(device))
    return gate_list, up_list, down_list


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    "btx_case",
    [
        "btx_uniform_k4_and_tp_moe",
        "btx_per_expert_pair_pdynamic",
        "btx_per_expert_pair_p33_p43",
    ],
)
def test_prep_btx_container_routed_moe_bitwise_matches_reference(
    tmp_path,
    codebook: str,
    btx_case: str,
) -> None:
    """Bitwise gate for BTX synth/read/prepare (uniform + per-expert-pair) on MUL1 and MCG."""
    from b12x.moe._shared.btx_schema import rate_code
    from b12x.moe._shared.kernels.w4a16.btx import (
        prepare_btx_moe_weights,
        read_btx_layer,
        read_btx_manifest,
    )
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig,
        synth_layer_payloads,
        write_btx_checkpoint,
    )
    from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
    from b12x.moe.fused_moe._impl import (
        TPMoEScratchCaps,
        plan_b12x_fp4_moe_weights,
        plan_tp_moe_scratch,
        prepare_b12x_fp4_moe_weights,
    )
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    device = torch.device("cuda", torch.cuda.current_device())
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    hidden, intermediate, num_experts, top_k = 256, 256, 4, 2
    slots = intermediate // 32
    seed_offset = 1 if codebook == "mul1" else 0

    if btx_case == "btx_uniform_k4_and_tp_moe":
        cfg = BtxSynthConfig(
            codebook=codebook,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            moe_layer_indices=(0,),
            bits=4,
            per_expert_input_rotations=True,
            extent_alignment_slots=4,
            seed=701 + seed_offset,
        )
        uniform_bits: int | None = 4
        rates_fc1 = rates_fc2 = None
    elif btx_case == "btx_per_expert_pair_pdynamic":
        fc1 = torch.tensor(
            [[rate_code(3, 3), rate_code(2, 4), rate_code(3, 3), rate_code(2, 4)]],
            dtype=torch.uint8,
        )
        fc2 = torch.tensor(
            [[rate_code(2, 4), rate_code(3, 3), rate_code(3, 3), rate_code(2, 4)]],
            dtype=torch.uint8,
        )
        cfg = BtxSynthConfig(
            codebook=codebook,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            moe_layer_indices=(0,),
            bits=None,
            rate_tables={0: (fc1, fc2)},
            per_expert_input_rotations=True,
            extent_alignment_slots=8,
            seed=711 + seed_offset,
        )
        uniform_bits = None
        rates_fc1, rates_fc2 = fc1, fc2
    else:
        fc1 = torch.tensor(
            [[rate_code(3, 3), rate_code(4, 3), rate_code(3, 3), rate_code(4, 3)]],
            dtype=torch.uint8,
        )
        fc2 = torch.tensor(
            [[rate_code(4, 3), rate_code(3, 3), rate_code(3, 3), rate_code(4, 3)]],
            dtype=torch.uint8,
        )
        cfg = BtxSynthConfig(
            codebook=codebook,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            moe_layer_indices=(0,),
            bits=None,
            rate_tables={0: (fc1, fc2)},
            per_expert_input_rotations=True,
            extent_alignment_slots=8,
            seed=721 + seed_offset,
        )
        uniform_bits = None
        rates_fc1, rates_fc2 = fc1, fc2

    ckpt_dir = tmp_path / f"{btx_case}_{codebook}"
    written = write_btx_checkpoint(ckpt_dir, cfg)
    manifest = read_btx_manifest(ckpt_dir)
    assert manifest == written
    assert manifest.codebook == codebook
    assert manifest.codebook_seed == (
        MUL1_MULTIPLIER if codebook == "mul1" else MCG_MULTIPLIER
    )
    layer = read_btx_layer(
        ckpt_dir, manifest, 0, first_slot=0, slot_count=slots, verify_sha=True
    )
    btx_prepared = prepare_btx_moe_weights(
        layer, activation="silu", device=device
    )
    assert btx_prepared.trellis_codebook == codebook

    payloads = synth_layer_payloads(cfg, 0)
    gate_w, up_w, down_w = _decode_btx_payloads_to_dense(
        payloads,
        num_experts=num_experts,
        hidden=hidden,
        intermediate=intermediate,
        codebook=codebook,
        uniform_bits=uniform_bits,
        rates_fc1=rates_fc1,
        rates_fc2=rates_fc2,
        device=device,
    )

    rng = torch.Generator(device=device).manual_seed(160_000 + seed_offset)
    m_cap = 16
    x_bf16 = (torch.randn((m_cap, hidden), generator=rng, device=device) * 0.25).to(
        torch.bfloat16
    )
    ids_full, weights_full = _make_balanced_routes(
        m_cap, top_k, num_experts, seed=160_100 + seed_offset, device=device
    )
    packed_buf = make_w4a16_packed_buffers(
        btx_prepared,
        m=m_cap,
        topk=top_k,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    out_direct = torch.empty((m_cap, hidden), dtype=torch.bfloat16, device=device)
    out_direct.fill_(float("nan"))
    actual_direct = run_w4a16_moe(
        x_bf16,
        btx_prepared,
        weights_full,
        ids_full,
        activation="silu",
        intermediate_cache13=packed_buf.intermediate_cache13,
        intermediate_cache2=packed_buf.intermediate_cache2,
        output=out_direct,
        fc1_c_tmp=packed_buf.fc1_c_tmp,
        fc2_c_tmp=packed_buf.fc2_c_tmp,
        packed_route_indices=packed_buf.packed_route_indices,
        block_expert_ids=packed_buf.block_expert_ids,
        packed_route_count=packed_buf.packed_route_count,
        expert_offsets=packed_buf.expert_offsets,
        expert_counts=packed_buf.expert_counts,
        route_block_size_m=8,
        intermediate_rotation_scales=btx_prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=btx_prepared.gate_suh,
        suh_up_table=btx_prepared.up_suh,
        svh_table=btx_prepared.down_svh,
        rotation_a_gate=packed_buf.rotation_a_gate,
        rotation_a_up=packed_buf.rotation_a_up,
    ).clone()
    torch.cuda.synchronize(device)
    expected_full = reference_exl3_routed_moe_bitwise(
        x_bf16,
        ids_full,
        weights_full,
        gate_w,
        up_w,
        down_w,
        btx_prepared.gate_suh,
        btx_prepared.up_suh,
        btx_prepared.intermediate_rotations,
        btx_prepared.down_svh,
        activation="silu",
        swiglu_limit=None,
        output_dtype=torch.bfloat16,
    )
    assert_bitwise_equal(actual_direct, expected_full, f"{btx_case}_direct_{codebook}")

    if btx_case == "btx_uniform_k4_and_tp_moe":
        w_plan = plan_b12x_fp4_moe_weights(
            quant_modes="w4a16",
            source_format="btx",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            w13_layout="w13",
            trellis_bits=4,
            trellis_codebook=codebook,
            trellis_tile_config=(64, 256, 64, 256),
        )
        experts = prepare_b12x_fp4_moe_weights(
            plan=w_plan,
            params_dtype=torch.bfloat16,
            btx_layer=layer,
            btx_device=device,
        )
        decode_cfg = MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        )
        scratch_plan = plan_tp_moe_scratch(
            TPMoEScratchCaps(
                decode_config=decode_cfg,
                max_tokens=m_cap,
                num_topk=top_k,
                route_num_experts=num_experts,
                device=device,
                weight_plan=w_plan,
                quant_mode="w4a16",
                w4a16_block_size_m=8,
                full_rotation_output_dtype=torch.bfloat16,
            ),
            prewarm_launches=True,
        )
        scratch_tensors = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in scratch_plan.scratch_specs()
        )
        out_tp = torch.empty_like(x_bf16)
        with kernel_resolution_guard(f"frozen tp_moe {btx_case}_{codebook}"):
            for live_m in (1, 5, m_cap):
                out_tp.fill_(float("nan"))
                binding = scratch_plan.bind(
                    scratch=scratch_tensors,
                    a=x_bf16[:live_m],
                    experts=experts,
                    topk_weights=weights_full[:live_m],
                    topk_ids=ids_full[:live_m],
                    output=out_tp[:live_m],
                )
                actual_tp = binding.run().clone()
                torch.cuda.synchronize(device)
                expected_live = reference_exl3_routed_moe_bitwise(
                    x_bf16[:live_m],
                    ids_full[:live_m],
                    weights_full[:live_m],
                    gate_w,
                    up_w,
                    down_w,
                    btx_prepared.gate_suh,
                    btx_prepared.up_suh,
                    btx_prepared.intermediate_rotations,
                    btx_prepared.down_svh,
                    activation="silu",
                    swiglu_limit=None,
                    output_dtype=torch.bfloat16,
                )
                assert_bitwise_equal(
                    actual_tp, expected_live, f"{btx_case}_tp_m{live_m}_{codebook}"
                )

            graph = torch.cuda.CUDAGraph()
            try:
                bind_g = scratch_plan.bind(
                    scratch=scratch_tensors,
                    a=x_bf16[:5],
                    experts=experts,
                    topk_weights=weights_full[:5],
                    topk_ids=ids_full[:5],
                    output=out_tp[:5],
                )
                with torch.cuda.graph(graph):
                    bind_g.run()
                expected_5 = reference_exl3_routed_moe_bitwise(
                    x_bf16[:5],
                    ids_full[:5],
                    weights_full[:5],
                    gate_w,
                    up_w,
                    down_w,
                    btx_prepared.gate_suh,
                    btx_prepared.up_suh,
                    btx_prepared.intermediate_rotations,
                    btx_prepared.down_svh,
                    activation="silu",
                    swiglu_limit=None,
                    output_dtype=torch.bfloat16,
                )
                for _ in range(2):
                    out_tp.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert_bitwise_equal(
                        out_tp[:5], expected_5, f"{btx_case}_graph_{codebook}"
                    )
            finally:
                graph.reset()


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize(
    "exec_case",
    [
        "exl3_uniform_canonical_route_and_moe",
        "exl3_projection_native_2tier_k3_k4",
        "b12x_trellis_v2_atoms_3tier_k3_k4_k5",
    ],
)
def test_prep_btx_fused_moe_routing_execution_bitwise_matches_reference(
    codebook: str,
    exec_case: str,
) -> None:
    """Bitwise gate for high-level fused_moe preparation, routing, and execution on MUL1 and MCG."""
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._impl import (
        TPMoEScratchCaps,
        plan_b12x_fp4_moe_weights,
        plan_tp_moe_scratch,
        prepare_b12x_projection_native_trellis_weights,
    )
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights
    from b12x.preparation import PreparationSession, PreparedCall

    device = torch.device("cuda", torch.cuda.current_device())
    seed_offset = 1 if codebook == "mul1" else 0
    decode_cfg = MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode="packed",
    )

    if exec_case == "exl3_uniform_canonical_route_and_moe":
        num_experts, hidden, intermediate, bits, top_k, m_cap = 4, 256, 256, 4, 2, 16
        gate_p = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, bits, codebook=codebook,
                seed=170_000 + e * 11 + seed_offset, device=device,
            )
            for e in range(num_experts)
        ]
        up_p = [
            make_gptqmodel_exl3_projection(
                hidden, intermediate, bits, codebook=codebook,
                seed=170_100 + e * 11 + seed_offset, device=device,
            )
            for e in range(num_experts)
        ]
        down_p = [
            make_gptqmodel_exl3_projection(
                intermediate, hidden, bits, codebook=codebook,
                seed=170_200 + e * 11 + seed_offset, device=device,
            )
            for e in range(num_experts)
        ]
        w13 = torch.stack(
            (
                torch.stack([p.trellis for p in gate_p], dim=0),
                torch.stack([p.trellis for p in up_p], dim=0),
            ),
            dim=0,
        ).contiguous()
        w2 = torch.stack([p.trellis for p in down_p], dim=0).contiguous()
        gate_suh = torch.stack([p.suh for p in gate_p], dim=0).contiguous()
        up_suh = torch.stack([p.suh for p in up_p], dim=0).contiguous()
        inter_rot = torch.cat(
            (
                torch.stack([p.svh for p in gate_p], dim=0),
                torch.stack([p.svh for p in up_p], dim=0),
                torch.stack([p.suh for p in down_p], dim=0),
            ),
            dim=1,
        ).contiguous()
        down_svh = torch.stack([p.svh for p in down_p], dim=0).contiguous()

        gate_w = [decode_exl3_projection_cpu(p, bits).to(device) for p in gate_p]
        up_w = [decode_exl3_projection_cpu(p, bits).to(device) for p in up_p]
        down_w = [decode_exl3_projection_cpu(p, bits).to(device) for p in down_p]

        src_kwargs = {"codebook": "mul1"} if codebook == "mul1" else {}
        marker_kwargs = (
            {"mul1": gate_p[0].marker}
            if codebook == "mul1"
            else {"mcg": gate_p[0].marker}
        )
        weight_plan = fused_moe.plan_weights(
            source=fused_moe.Exl3TrellisSource(
                bits=bits, tile_config=(64, 256, 64, 256), **src_kwargs
            ),
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=num_experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
            ),
        )
        experts = fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.Exl3TrellisWeights(
                w13=w13,
                w2=w2,
                gate_suh=gate_suh,
                up_suh=up_suh,
                intermediate_rotations=inter_rot,
                down_svh=down_svh,
                **marker_kwargs,
            ),
        )

        from b12x.moe._shared.routing import route_topk as _native_route_topk

        rng = torch.Generator(device=device).manual_seed(170_500 + seed_offset)
        x_bf16 = (torch.randn((m_cap, hidden), generator=rng, device=device) * 0.25).to(
            torch.bfloat16
        )
        router_logits = torch.randn(
            (m_cap, num_experts), generator=rng, device=device, dtype=torch.float32
        )
        topk_logits = torch.empty((m_cap, top_k), dtype=torch.float32, device=device)
        topk_ids = torch.empty((m_cap, top_k), dtype=torch.int32, device=device)
        topk_weights = torch.empty((m_cap, top_k), dtype=torch.float32, device=device)
        _native_route_topk(
            router_logits,
            topk_logits,
            topk_ids,
            topk_weights,
            renormalize=True,
            score_func="softmax",
        )

        exec_plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=m_cap, top_k=top_k, warmup_token_counts=(1, 7)
            ),
            invocation={"w4a16_block_size_m": 8},
            override=decode_cfg,
        )

        def _prep_moe_call(m_rows: int):
            def _call(state):
                scratch = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device)
                    for s in state.scratch.scratch_specs()
                )
                out = torch.empty((m_rows, hidden), dtype=torch.float32, device=device)
                ids_tmp, w_tmp = _make_balanced_routes(
                    m_rows, top_k, num_experts, seed=999, device=device
                )
                b = state.bind(
                    a=x_bf16[:m_rows],
                    topk_ids=ids_tmp,
                    topk_weights=w_tmp,
                    output=out,
                    scratch=scratch,
                )
                return PreparedCall(
                    run=lambda: state.run(b), output=out, owners=(scratch, b)
                )

            return _call

        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            session.prepare(
                (
                    exec_plan.request(
                        name=f"moe-{codebook}",
                        prepare_calls={
                            cnt: _prep_moe_call(cnt) for cnt in exec_plan.token_counts
                        },
                    ),
                )
            )
            session.freeze()
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in exec_plan.scratch_specs()
            )
            output = torch.empty((m_cap, hidden), dtype=torch.float32, device=device)
            for live_m in (1, 7, m_cap):
                output.fill_(float("nan"))
                binding = fused_moe.bind(
                    exec_plan,
                    a=x_bf16[:live_m],
                    experts=experts,
                    topk_ids=topk_ids[:live_m],
                    topk_weights=topk_weights[:live_m],
                    output=output[:live_m],
                    scratch=scratch,
                )
                actual = fused_moe.run(binding=binding).clone()
                torch.cuda.synchronize(device)
                expected = reference_exl3_routed_moe_bitwise(
                    x_bf16[:live_m],
                    topk_ids[:live_m],
                    topk_weights[:live_m],
                    gate_w,
                    up_w,
                    down_w,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    activation="silu",
                    swiglu_limit=None,
                    output_dtype=torch.float32,
                )
                assert_bitwise_equal(
                    actual, expected, f"{exec_case}_m{live_m}_{codebook}"
                )
            graph = torch.cuda.CUDAGraph()
            try:
                bind_g = fused_moe.bind(
                    exec_plan,
                    a=x_bf16[:7],
                    experts=experts,
                    topk_ids=topk_ids[:7],
                    topk_weights=topk_weights[:7],
                    output=output[:7],
                    scratch=scratch,
                )
                with session.capture(), torch.cuda.graph(graph):
                    fused_moe.run(binding=bind_g)
                expected_7 = reference_exl3_routed_moe_bitwise(
                    x_bf16[:7],
                    topk_ids[:7],
                    topk_weights[:7],
                    gate_w,
                    up_w,
                    down_w,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    activation="silu",
                    swiglu_limit=None,
                    output_dtype=torch.float32,
                )
                for _ in range(2):
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert_bitwise_equal(
                        output[:7], expected_7, f"{exec_case}_graph_{codebook}"
                    )
            finally:
                graph.reset()
        return

    if exec_case == "exl3_projection_native_2tier_k3_k4":
        num_experts, hidden, intermediate, top_k, m_cap = 6, 256, 256, 2, 16
        # Per-projection tier assignment across (3, 4) so individual experts mix K3 and K4.
        gate_members = ((0, 2, 4), (1, 3, 5))
        up_members = ((1, 2, 5), (0, 3, 4))
        down_members = ((0, 1, 3), (2, 4, 5))
        tier_bits = (3, 4)

        gate_dense = [None] * num_experts
        up_dense = [None] * num_experts
        down_dense = [None] * num_experts
        gate_suh_list = [None] * num_experts
        up_suh_list = [None] * num_experts
        gate_svh_list = [None] * num_experts
        up_svh_list = [None] * num_experts
        down_suh_list = [None] * num_experts
        down_svh_list = [None] * num_experts
        sample_marker = None

        native_tiers_list: list[ProjectionTrellisTierWeights] = []
        for t_idx, b in enumerate(tier_bits):
            g_projs = []
            for e in gate_members[t_idx]:
                p = make_gptqmodel_exl3_projection(
                    hidden, intermediate, b, codebook=codebook,
                    seed=171_000 + e * 13 + seed_offset, device=device,
                )
                sample_marker = p.marker
                g_projs.append(p.trellis)
                gate_dense[e] = decode_exl3_projection_cpu(p, b).to(device)
                gate_suh_list[e] = p.suh
                gate_svh_list[e] = p.svh
            u_projs = []
            for e in up_members[t_idx]:
                p = make_gptqmodel_exl3_projection(
                    hidden, intermediate, b, codebook=codebook,
                    seed=171_200 + e * 13 + seed_offset, device=device,
                )
                u_projs.append(p.trellis)
                up_dense[e] = decode_exl3_projection_cpu(p, b).to(device)
                up_suh_list[e] = p.suh
                up_svh_list[e] = p.svh
            d_projs = []
            for e in down_members[t_idx]:
                p = make_gptqmodel_exl3_projection(
                    intermediate, hidden, b, codebook=codebook,
                    seed=171_400 + e * 13 + seed_offset, device=device,
                )
                d_projs.append(p.trellis)
                down_dense[e] = decode_exl3_projection_cpu(p, b).to(device)
                down_suh_list[e] = p.suh
                down_svh_list[e] = p.svh

            native_tiers_list.append(
                ProjectionTrellisTierWeights(
                    bits=b,
                    w13=torch.cat(
                        (torch.stack(g_projs, dim=0), torch.stack(u_projs, dim=0)), dim=0
                    ).contiguous(),
                    w2=torch.stack(d_projs, dim=0).contiguous(),
                    gate_experts=gate_members[t_idx],
                    up_experts=up_members[t_idx],
                    down_experts=down_members[t_idx],
                )
            )

        gate_suh = torch.stack(gate_suh_list, dim=0).contiguous()  # type: ignore[arg-type]
        up_suh = torch.stack(up_suh_list, dim=0).contiguous()  # type: ignore[arg-type]
        inter_rot = torch.cat(
            (
                torch.stack(gate_svh_list, dim=0),  # type: ignore[arg-type]
                torch.stack(up_svh_list, dim=0),  # type: ignore[arg-type]
                torch.stack(down_suh_list, dim=0),  # type: ignore[arg-type]
            ),
            dim=1,
        ).contiguous()
        down_svh = torch.stack(down_svh_list, dim=0).contiguous()  # type: ignore[arg-type]
        gate_w = [t for t in gate_dense if t is not None]
        up_w = [t for t in up_dense if t is not None]
        down_w = [t for t in down_dense if t is not None]
        assert len(gate_w) == len(up_w) == len(down_w) == num_experts

        plan = plan_b12x_fp4_moe_weights(
            quant_modes="w4a16",
            source_format="exl3_trellis_mcg",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            w13_layout="trellis_t256_proj",
            w4a16_layout="trellis_native",
            trellis_bits=3,
            trellis_codebook=codebook,
            trellis_rate_granularity="per_expert_projection",
        )
        marker_kw = {"mul1": sample_marker} if codebook == "mul1" else {}
        experts = prepare_b12x_projection_native_trellis_weights(
            plan=plan,
            native_tiers=tuple(native_tiers_list),
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=inter_rot,
            down_svh=down_svh,
            **marker_kw,
        )
        scratch_plan = plan_tp_moe_scratch(
            TPMoEScratchCaps(
                decode_config=decode_cfg,
                max_tokens=m_cap,
                num_topk=top_k,
                route_num_experts=num_experts,
                device=device,
                weight_plan=plan,
                quant_mode="w4a16",
                full_rotation_output_dtype=torch.bfloat16,
                mixed_trellis_route_id_dtypes=(torch.int32,),
                mixed_trellis_broadcast_suh=(False,),
                mixed_trellis_broadcast_svh=(False,),
            ),
            prewarm_launches=True,
        )
    else:
        num_experts, hidden, intermediate, top_k, m_cap = 6, 256, 256, 2, 12
        slots = intermediate // 32
        hidden_tiles = hidden // 16
        rate_vals = torch.tensor(
            [
                [0x33, 0x44, 0x33],
                [0x44, 0x33, 0x55],
                [0x55, 0x55, 0x44],
                [0x33, 0x55, 0x44],
                [0x44, 0x44, 0x33],
                [0x55, 0x33, 0x55],
            ],
            dtype=torch.uint8,
            device=device,
        )
        bits_host = (rate_vals.detach().cpu().to(torch.int64) & 0x0F).tolist()
        rng_np = np.random.default_rng(172_000 + seed_offset)

        gate_dense = []
        up_dense = []
        down_dense = []
        expert_bundles_by_slot: list[list[torch.Tensor]] = [[] for _ in range(slots)]
        for e in range(num_experts):
            for proj_idx, out_list in ((0, gate_dense), (1, up_dense), (2, down_dense)):
                b = int(bits_host[e][proj_idx])
                planes_np = rng_np.integers(
                    -32768, 32768, size=(slots, 2, hidden_tiles, 16 * b), dtype=np.int16
                )
                planes_t = torch.from_numpy(planes_np)
                for s in range(slots):
                    expert_bundles_by_slot[s].append(
                        planes_t[s].contiguous().view(torch.uint8).reshape(-1)
                    )
                if proj_idx < 2:
                    native_t = planes_t.permute(2, 0, 1, 3).reshape(
                        hidden_tiles, 2 * slots, 16 * b
                    )
                else:
                    native_t = planes_t.reshape(2 * slots, hidden_tiles, 16 * b)
                out_list.append(
                    _decode_native_trellis_tensor_cpu(native_t, b, codebook).to(device)
                )

        atoms = torch.stack(
            [torch.cat(slot_parts, dim=0) for slot_parts in expert_bundles_by_slot],
            dim=0,
        ).contiguous().to(device=device)

        rng_rot = torch.Generator(device=device).manual_seed(172_300 + seed_offset)
        in_vecs = (0.75 + 0.5 * torch.rand((num_experts, 2, hidden), generator=rng_rot, device=device)).to(
            torch.float16
        )
        inter_vecs = (0.75 + 0.5 * torch.rand((num_experts, 3, intermediate), generator=rng_rot, device=device)).to(
            torch.float16
        )
        out_vecs = (0.75 + 0.5 * torch.rand((num_experts, hidden), generator=rng_rot, device=device)).to(
            torch.float16
        )
        gate_suh = in_vecs[:, 0].contiguous()
        up_suh = in_vecs[:, 1].contiguous()
        inter_rot = inter_vecs.reshape(num_experts, 3 * intermediate).contiguous()
        down_svh = out_vecs.contiguous()
        gate_w = gate_dense
        up_w = up_dense
        down_w = down_dense

        v2_cfg = fused_moe.TrellisConfig.from_dict(
            {
                "version": 2,
                "codebook": codebook,
                "rate": {"granularity": "per_expert_projection"},
                "scale": {
                    "input_scales": {"vectors": "per_expert", "gains": "none"},
                    "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                    "output_scales": {"vectors": "per_expert", "gains": "none"},
                },
                "transform": {
                    "projection": {"kind": "scaled_hadamard", "block_size": 128},
                    "expert": {"kind": "none"},
                },
            }
        )
        w_plan = fused_moe.plan_weights(
            source=v2_cfg,
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=num_experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
            ),
        )
        prepared_public = fused_moe.prepare_weights(
            plan=w_plan,
            weights=fused_moe.TrellisWeights(
                atoms=atoms,
                rate=rate_vals,
                input_scales=fused_moe.ScaleFactors(vectors=in_vecs),
                intermediate_scales=fused_moe.ScaleFactors(vectors=inter_vecs),
                output_scales=fused_moe.ScaleFactors(vectors=out_vecs),
            ),
        )
        experts = prepared_public._impl
        scratch_plan = plan_tp_moe_scratch(
            TPMoEScratchCaps(
                decode_config=decode_cfg,
                max_tokens=m_cap,
                num_topk=top_k,
                route_num_experts=num_experts,
                device=device,
                weight_plan=w_plan._impl,
                quant_mode="w4a16",
                full_rotation_output_dtype=torch.bfloat16,
                mixed_trellis_route_id_dtypes=(torch.int32,),
                mixed_trellis_broadcast_suh=(False,),
                mixed_trellis_broadcast_svh=(False,),
            ),
            prewarm_launches=True,
        )

    scratch_tensors = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in scratch_plan.scratch_specs()
    )
    rng_x = torch.Generator(device=device).manual_seed(173_000 + seed_offset)
    x_bf16 = (torch.randn((m_cap, hidden), generator=rng_x, device=device) * 0.25).to(
        torch.bfloat16
    )
    ids_full, weights_full = _make_balanced_routes(
        m_cap, top_k, num_experts, seed=173_100 + seed_offset, device=device
    )
    out_buf = torch.empty_like(x_bf16)

    live_rows = (1, 5, m_cap) if m_cap == 16 else (1, 6, m_cap)
    with kernel_resolution_guard(f"frozen {exec_case}_{codebook}"):
        for live_m in live_rows:
            out_buf.fill_(float("nan"))
            binding = scratch_plan.bind(
                scratch=scratch_tensors,
                a=x_bf16[:live_m],
                experts=experts,
                topk_weights=weights_full[:live_m],
                topk_ids=ids_full[:live_m],
                output=out_buf[:live_m],
            )
            actual = binding.run().clone()
            torch.cuda.synchronize(device)
            live_tile_k = (
                int(binding.mixed_trellis_binding.launch.fc1_tile_k)
                if binding.mixed_trellis_binding is not None
                else 64
            )
            expected = reference_exl3_routed_moe_bitwise(
                x_bf16[:live_m],
                ids_full[:live_m],
                weights_full[:live_m],
                gate_w,
                up_w,
                down_w,
                gate_suh,
                up_suh,
                inter_rot,
                down_svh,
                activation="silu",
                swiglu_limit=None,
                output_dtype=torch.bfloat16,
                tile_k=live_tile_k,
            )
            assert_bitwise_equal(
                actual, expected, f"{exec_case}_m{live_m}_{codebook}"
            )

        mid_m = live_rows[1]
        graph = torch.cuda.CUDAGraph()
        try:
            bind_g = scratch_plan.bind(
                scratch=scratch_tensors,
                a=x_bf16[:mid_m],
                experts=experts,
                topk_weights=weights_full[:mid_m],
                topk_ids=ids_full[:mid_m],
                output=out_buf[:mid_m],
            )
            with torch.cuda.graph(graph):
                bind_g.run()
            mid_tile_k = (
                int(bind_g.mixed_trellis_binding.launch.fc1_tile_k)
                if bind_g.mixed_trellis_binding is not None
                else 64
            )
            expected_mid = reference_exl3_routed_moe_bitwise(
                x_bf16[:mid_m],
                ids_full[:mid_m],
                weights_full[:mid_m],
                gate_w,
                up_w,
                down_w,
                gate_suh,
                up_suh,
                inter_rot,
                down_svh,
                activation="silu",
                swiglu_limit=None,
                output_dtype=torch.bfloat16,
                tile_k=mid_tile_k,
            )
            for _ in range(2):
                out_buf.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                assert_bitwise_equal(
                    out_buf[:mid_m], expected_mid, f"{exec_case}_graph_{codebook}"
                )
        finally:
            graph.reset()


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
def test_prep_btx_mul1_compat_lifts_and_marker_guards_gpu(tmp_path) -> None:
    """Bitwise gate for MUL1 btx_compat v1/v2 lifts + prepare.py checkpoint marker guards on GPU."""
    from b12x.moe._shared.btx_schema import matrix_atom_bytes, rate_code
    from b12x.moe._shared.kernels.w4a16.btx import (
        prepare_btx_moe_weights,
        read_btx_layer,
    )
    from b12x.moe._shared.kernels.w4a16.btx_compat import (
        lift_qsrt_atoms_v1_extent,
        lift_qsrt_atoms_v2_extent,
        lift_qsrt_atoms_v3_extent,
    )
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig,
        synth_layer_payloads,
        write_btx_checkpoint,
    )
    from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
    from b12x.moe.fused_moe._impl import (
        TPMoEScratchCaps,
        plan_b12x_fp4_moe_weights,
        plan_tp_moe_scratch,
        prepare_b12x_fp4_moe_weights,
    )
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    device = torch.device("cuda", torch.cuda.current_device())
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)

    # 1. Direct prepare_trellis256_moe_weights with `.mul1` checkpoint marker (no explicit codebook=)
    hidden, intermediate, num_experts, bits, top_k = 256, 256, 3, 4, 2
    p_gate = [
        make_gptqmodel_exl3_projection(
            hidden, intermediate, bits, codebook="mul1", seed=175_000 + e, device=device
        )
        for e in range(num_experts)
    ]
    p_up = [
        make_gptqmodel_exl3_projection(
            hidden, intermediate, bits, codebook="mul1", seed=175_100 + e, device=device
        )
        for e in range(num_experts)
    ]
    p_down = [
        make_gptqmodel_exl3_projection(
            intermediate, hidden, bits, codebook="mul1", seed=175_200 + e, device=device
        )
        for e in range(num_experts)
    ]
    w13 = torch.stack(
        (
            torch.stack([p.trellis for p in p_gate], dim=0),
            torch.stack([p.trellis for p in p_up], dim=0),
        ),
        dim=0,
    ).contiguous()
    w2 = torch.stack([p.trellis for p in p_down], dim=0).contiguous()
    gate_suh = torch.stack([p.suh for p in p_gate], dim=0).contiguous()
    up_suh = torch.stack([p.suh for p in p_up], dim=0).contiguous()
    inter_rot = torch.cat(
        (
            torch.stack([p.svh for p in p_gate], dim=0),
            torch.stack([p.svh for p in p_up], dim=0),
            torch.stack([p.suh for p in p_down], dim=0),
        ),
        dim=1,
    ).contiguous()
    down_svh = torch.stack([p.svh for p in p_down], dim=0).contiguous()

    prep_mul1 = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        activation="silu",
        fc1_tile_n=256,
        fc2_tile_n=256,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        mul1=p_gate[0].marker,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=inter_rot,
        down_svh=down_svh,
    )
    assert prep_mul1.trellis_codebook == "mul1"
    assert prep_mul1.w13.data_ptr() == w13.data_ptr()

    with pytest.raises(ValueError, match="multiple EXL3 codebook markers"):
        prepare_trellis256_moe_weights(
            w13,
            w2,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            activation="silu",
            fc1_tile_n=256,
            fc2_tile_n=256,
            w13_layout="trellis_t256_proj",
            mcg=0xCBAC1FED,
            mul1=p_gate[0].marker,
        )
    with pytest.raises(ValueError, match="disagrees"):
        prepare_trellis256_moe_weights(
            w13,
            w2,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            activation="silu",
            fc1_tile_n=256,
            fc2_tile_n=256,
            w13_layout="trellis_t256_proj",
            codebook="mcg",
            mul1=p_gate[0].marker,
        )

    # Bind-time fail-closed guard when prepared codebook disagrees with scratch plan
    mcg_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        trellis_bits=bits,
        trellis_codebook="mcg",
    )
    mul1_plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        trellis_bits=bits,
        trellis_codebook="mul1",
    )
    mul1_experts = prepare_b12x_fp4_moe_weights(
        plan=mul1_plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=inter_rot,
        down_svh=down_svh,
        trellis_mul1=p_gate[0].marker,
    )
    mcg_scratch_plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            decode_config=MoeDecodeConfig(
                backend="w4a16",
                route_planner="internal",
                max_active_clusters=None,
                w4a16_route_mode="packed",
            ),
            max_tokens=4,
            num_topk=top_k,
            route_num_experts=num_experts,
            device=device,
            weight_plan=mcg_plan,
            quant_mode="w4a16",
            w4a16_block_size_m=8,
            full_rotation_output_dtype=torch.bfloat16,
        ),
        prewarm_launches=True,
    )
    mcg_scratch = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=device)
        for s in mcg_scratch_plan.scratch_specs()
    )
    x_small = torch.ones((2, hidden), dtype=torch.bfloat16, device=device)
    ids_small, w_small = _make_balanced_routes(
        2, top_k, num_experts, seed=175_300, device=device
    )
    with pytest.raises(
        ValueError, match="experts do not match the plan used to size TP MoE scratch"
    ):
        mcg_scratch_plan.bind(
            scratch=mcg_scratch,
            a=x_small,
            experts=mul1_experts,
            topk_weights=w_small,
            topk_ids=ids_small,
        )
    with pytest.raises(ValueError, match="codebook does not match the plan"):
        mcg_scratch_plan.bind(
            scratch=mcg_scratch,
            a=x_small,
            experts=replace(mul1_experts, plan=mcg_plan),
            topk_weights=w_small,
            topk_ids=ids_small,
        )

    # 2. btx_compat v1 (PDYNAMIC) and v2 (P33_P43) lifts with codebook="mul1"
    expert_ids = torch.tensor([3, 7, 11], dtype=torch.int32)
    format_codes = torch.tensor([0x21, 0x02, 0x11], dtype=torch.uint8)
    rotation = (5 * expert_ids.to(torch.int64) + 1) % 12
    logical_pair = (0 - rotation) % 12
    fc1_codes = torch.where(
        logical_pair < (format_codes.to(torch.int64) >> 4),
        rate_code(2, 4),
        rate_code(3, 3),
    ).to(torch.uint8)
    fc2_codes = torch.where(
        logical_pair < (format_codes.to(torch.int64) & 0xF),
        rate_code(2, 4),
        rate_code(3, 3),
    ).to(torch.uint8)

    v1_synth_cfg = BtxSynthConfig(
        codebook="mul1",
        num_experts=3,
        hidden_size=hidden,
        intermediate_size=256,
        moe_layer_indices=(1,),
        bits=None,
        rate_tables={
            1: (fc1_codes.reshape(1, -1).clone(), fc2_codes.reshape(1, -1).clone())
        },
        extent_alignment_slots=8,
        seed=175_400,
    )
    v1_manifest = write_btx_checkpoint(tmp_path / "v1-mul1", v1_synth_cfg)
    v1_synth_layer = read_btx_layer(
        tmp_path / "v1-mul1", v1_manifest, 1, first_slot=0, slot_count=8
    )
    from_synth_v1 = prepare_btx_moe_weights(
        v1_synth_layer, activation="silu", device=device
    )
    v1_payloads = synth_layer_payloads(v1_synth_cfg, 1)
    mbytes = matrix_atom_bytes(hidden, 3, 3)
    v1_bundle = 3 * mbytes + 3 * 64
    v1_raw = torch.zeros((8, 3, v1_bundle), dtype=torch.uint8)
    for s in range(8):
        for e in range(3):
            cur = 0
            for mat in range(3):
                lo_p, hi_p = v1_payloads.planes[(e, s, mat)]
                for pl in (lo_p, hi_p):
                    raw_b = pl.contiguous().view(torch.uint8).reshape(-1)
                    v1_raw[s, e, cur : cur + raw_b.numel()] = raw_b
                    cur += raw_b.numel()
            for mat in range(3):
                v1_raw[s, e, cur : cur + 64] = (
                    v1_payloads.rotations[s, e, mat].contiguous().view(torch.uint8)
                )
                cur += 64

    lifted_v1_layer = lift_qsrt_atoms_v1_extent(
        v1_raw,
        first_atom_slot=0,
        layer_index=1,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        global_intermediate_size=256,
        gate_suh=v1_payloads.gate_suh,
        up_suh=v1_payloads.up_suh,
        down_svh=v1_payloads.down_svh,
        codebook="mul1",
    )
    assert lifted_v1_layer.manifest.codebook == "mul1"
    assert lifted_v1_layer.manifest.codebook_seed == MUL1_MULTIPLIER
    lifted_v1 = prepare_btx_moe_weights(
        lifted_v1_layer, activation="silu", device=device
    )
    assert torch.equal(lifted_v1.w13, from_synth_v1.w13)
    assert torch.equal(lifted_v1.w2, from_synth_v1.w2)
    assert torch.equal(
        lifted_v1.intermediate_rotations, from_synth_v1.intermediate_rotations
    )

    gate_v1, up_v1, down_v1 = _decode_btx_payloads_to_dense(
        v1_payloads,
        num_experts=3,
        hidden=hidden,
        intermediate=256,
        codebook="mul1",
        uniform_bits=None,
        rates_fc1=fc1_codes.reshape(1, -1),
        rates_fc2=fc2_codes.reshape(1, -1),
        device=device,
    )
    x_v1 = (
        torch.randn(
            (8, hidden),
            generator=torch.Generator(device=device).manual_seed(175_500),
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    ids_v1, w_v1 = _make_balanced_routes(8, 2, 3, seed=175_600, device=device)
    buf_v1 = make_w4a16_packed_buffers(
        lifted_v1,
        m=8,
        topk=2,
        dtype=torch.float16,
        device=device,
        full_rotation=True,
        block_size_m=8,
    )
    out_v1 = torch.empty_like(x_v1)
    actual_v1 = run_w4a16_moe(
        x_v1,
        lifted_v1,
        w_v1,
        ids_v1,
        activation="silu",
        intermediate_cache13=buf_v1.intermediate_cache13,
        intermediate_cache2=buf_v1.intermediate_cache2,
        output=out_v1,
        fc1_c_tmp=buf_v1.fc1_c_tmp,
        fc2_c_tmp=buf_v1.fc2_c_tmp,
        packed_route_indices=buf_v1.packed_route_indices,
        block_expert_ids=buf_v1.block_expert_ids,
        packed_route_count=buf_v1.packed_route_count,
        expert_offsets=buf_v1.expert_offsets,
        expert_counts=buf_v1.expert_counts,
        route_block_size_m=8,
        intermediate_rotation_scales=lifted_v1.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=lifted_v1.gate_suh,
        suh_up_table=lifted_v1.up_suh,
        svh_table=lifted_v1.down_svh,
        rotation_a_gate=buf_v1.rotation_a_gate,
        rotation_a_up=buf_v1.rotation_a_up,
    ).clone()
    torch.cuda.synchronize(device)
    expected_v1 = reference_exl3_routed_moe_bitwise(
        x_v1,
        ids_v1,
        w_v1,
        gate_v1,
        up_v1,
        down_v1,
        lifted_v1.gate_suh,
        lifted_v1.up_suh,
        lifted_v1.intermediate_rotations,
        lifted_v1.down_svh,
        activation="silu",
        swiglu_limit=None,
        output_dtype=torch.bfloat16,
    )
    assert_bitwise_equal(actual_v1, expected_v1, "btx_compat_v1_lift_mul1")

    # 3. btx_compat v2 and v3 lifts with codebook="mul1"
    v2_cfg = BtxSynthConfig(
        codebook="mul1",
        num_experts=2,
        hidden_size=512,
        intermediate_size=512,
        moe_layer_indices=(1,),
        bits=3,
        coupled=True,
        pre_block=512,
        post_block=128,
        per_expert_input_rotations=True,
        extent_alignment_slots=4,
        extent_barriers=(8,),
        seed=175_700,
    )
    v2_manifest = write_btx_checkpoint(tmp_path / "v2-mul1", v2_cfg)
    v2_synth_layer = read_btx_layer(
        tmp_path / "v2-mul1", v2_manifest, 1, first_slot=0, slot_count=8
    )
    from_synth_v2 = prepare_btx_moe_weights(
        v2_synth_layer,
        activation="silu",
        device=device,
        tile_config=(128, 128, 128, 128),
    )
    v2_payloads = synth_layer_payloads(v2_cfg, 1)
    v2_sec = matrix_atom_bytes(512, 3, 3)
    v2_bundle = 3 * v2_sec + 3 * 64
    v2_raw = torch.zeros((8, 2 * v2_bundle), dtype=torch.uint8)
    for s in range(8):
        cur = 0
        for e in range(2):
            for mat in range(3):
                lo_p, hi_p = v2_payloads.planes[(e, s, mat)]
                for pl in (lo_p, hi_p):
                    raw_b = pl.contiguous().view(torch.uint8).reshape(-1)
                    v2_raw[s, cur : cur + raw_b.numel()] = raw_b
                    cur += raw_b.numel()
            for mat in range(3):
                v2_raw[s, cur : cur + 64] = (
                    v2_payloads.rotations[s, e, mat].contiguous().view(torch.uint8)
                )
                cur += 64
    lifted_v2_layer = lift_qsrt_atoms_v2_extent(
        v2_raw,
        profile="k3_coupled_h512_h128",
        first_atom_slot=0,
        layer_index=1,
        hidden_size=512,
        global_intermediate_size=512,
        num_experts=2,
        gate_suh=v2_payloads.gate_suh,
        up_suh=v2_payloads.up_suh,
        down_svh=v2_payloads.down_svh,
        rotation_draws=v2_payloads.rotation_draws,
        codebook="mul1",
    )
    assert lifted_v2_layer.manifest.codebook == "mul1"
    assert lifted_v2_layer.manifest.codebook_seed == MUL1_MULTIPLIER
    lifted_v2 = prepare_btx_moe_weights(
        lifted_v2_layer,
        activation="silu",
        device=device,
        tile_config=(128, 128, 128, 128),
    )
    assert lifted_v2.trellis_codebook == "mul1"
    assert torch.equal(lifted_v2.w13, from_synth_v2.w13)
    assert torch.equal(lifted_v2.w2, from_synth_v2.w2)
    assert torch.equal(
        lifted_v2.intermediate_rotations, from_synth_v2.intermediate_rotations
    )

    lifted_v3_layer = lift_qsrt_atoms_v3_extent(
        torch.zeros((4, 2 * 3 * v2_sec), dtype=torch.uint8),
        first_atom_slot=0,
        layer_index=1,
        hidden_size=512,
        global_intermediate_size=512,
        num_experts=2,
        gate_suh_signs=torch.ones(512, dtype=torch.float16),
        up_suh_signs=torch.ones(512, dtype=torch.float16),
        down_svh_signs=torch.ones(512, dtype=torch.float16),
        gate_svh_signs=torch.ones(512, dtype=torch.float16),
        up_svh_signs=torch.ones(512, dtype=torch.float16),
        down_suh_signs=torch.ones(512, dtype=torch.float16),
        layer_scale_magnitudes=torch.ones(3, dtype=torch.float16),
        expert_scale_magnitudes=torch.ones((3, 2), dtype=torch.float16),
        rotation_draws=torch.zeros(2, dtype=torch.uint8),
        codebook="mul1",
    )
    assert lifted_v3_layer.manifest.codebook == "mul1"
    assert lifted_v3_layer.manifest.codebook_seed == MUL1_MULTIPLIER
    lifted_v3 = prepare_btx_moe_weights(
        lifted_v3_layer,
        activation="silu",
        device=device,
        tile_config=(128, 128, 128, 128),
    )
    assert lifted_v3.trellis_codebook == "mul1"


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120 or SM121 GPU")
@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
def test_typed_v2_trellis_config_uniform_k4_end_to_end_matches_reference(
    codebook: str,
) -> None:
    """End-to-end typed v2 TrellisConfig(uniform, mul1/mcg) with rate 0x44 (true uniform K4)."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._impl import TPMoEScratchCaps, plan_tp_moe_scratch
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    from b12x.preparation import PreparationSession, PreparedCall

    device = torch.device("cuda", torch.cuda.current_device())
    num_experts, hidden, intermediate, top_k, m_cap = 4, 256, 256, 2, 16
    bits = 4
    slots = intermediate // 32
    hidden_tiles = hidden // 16
    seed_offset = 100 if codebook == "mul1" else 200

    rng_np = np.random.default_rng(184_000 + seed_offset)
    gate_dense = []
    up_dense = []
    down_dense = []
    expert_bundles_by_slot: list[list[torch.Tensor]] = [[] for _ in range(slots)]
    for _e in range(num_experts):
        for proj_idx, out_list in ((0, gate_dense), (1, up_dense), (2, down_dense)):
            planes_np = rng_np.integers(
                -32768, 32768, size=(slots, 2, hidden_tiles, 16 * bits), dtype=np.int16
            )
            planes_t = torch.from_numpy(planes_np)
            for s in range(slots):
                expert_bundles_by_slot[s].append(
                    planes_t[s].contiguous().view(torch.uint8).reshape(-1)
                )
            if proj_idx < 2:
                native_t = planes_t.permute(2, 0, 1, 3).reshape(
                    hidden_tiles, 2 * slots, 16 * bits
                )
            else:
                native_t = planes_t.reshape(2 * slots, hidden_tiles, 16 * bits)
            out_list.append(
                _decode_native_trellis_tensor_cpu(native_t, bits, codebook).to(device)
            )

    atoms = torch.stack(
        [torch.cat(slot_parts, dim=0) for slot_parts in expert_bundles_by_slot],
        dim=0,
    ).contiguous().to(device=device)
    rate_0x44 = torch.tensor([0x44], dtype=torch.uint8, device=device)

    rng_rot = torch.Generator(device=device).manual_seed(184_300 + seed_offset)
    in_vecs = (0.75 + 0.5 * torch.rand((num_experts, 2, hidden), generator=rng_rot, device=device)).to(
        torch.float16
    )
    inter_vecs = (0.75 + 0.5 * torch.rand((num_experts, 3, intermediate), generator=rng_rot, device=device)).to(
        torch.float16
    )
    out_vecs = (0.75 + 0.5 * torch.rand((num_experts, hidden), generator=rng_rot, device=device)).to(
        torch.float16
    )
    gate_suh = in_vecs[:, 0].contiguous()
    up_suh = in_vecs[:, 1].contiguous()
    inter_rot = inter_vecs.reshape(num_experts, 3 * intermediate).contiguous()
    down_svh = out_vecs.contiguous()

    v2_uniform_cfg = fused_moe.TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": codebook,
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {"vectors": "per_expert", "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": "per_expert", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {"kind": "none"},
            },
        }
    )
    w_plan = fused_moe.plan_weights(
        source=v2_uniform_cfg,
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=w_plan,
        weights=fused_moe.TrellisWeights(
            atoms=atoms,
            rate=rate_0x44,
            input_scales=fused_moe.ScaleFactors(vectors=in_vecs),
            intermediate_scales=fused_moe.ScaleFactors(vectors=inter_vecs),
            output_scales=fused_moe.ScaleFactors(vectors=out_vecs),
        ),
    )
    assert experts.plan._impl.trellis_bits == 4
    assert experts._impl.plan.trellis_bits == 4
    assert experts.plan._impl.trellis_codebook == codebook

    decode_cfg = MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode="packed",
    )

    # Sizing scratch from the pre-weight placeholder w_plan (trellis_bits=3) fails closed on bind.
    stale_scratch = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            decode_config=decode_cfg,
            max_tokens=m_cap,
            num_topk=top_k,
            route_num_experts=num_experts,
            device=device,
            weight_plan=w_plan._impl,
            quant_mode="w4a16",
        ),
        prewarm_launches=False,
    )
    stale_bufs = tuple(
        torch.empty(s.shape, dtype=s.dtype, device=device)
        for s in stale_scratch.scratch_specs()
    )
    rng_x = torch.Generator(device=device).manual_seed(184_500 + seed_offset)
    x_bf16 = (torch.randn((m_cap, hidden), generator=rng_x, device=device) * 0.25).to(
        torch.bfloat16
    )
    ids_full, weights_full = _make_balanced_routes(
        m_cap, top_k, num_experts, seed=184_600 + seed_offset, device=device
    )
    with pytest.raises(ValueError, match="experts do not match the plan used to size TP MoE scratch"):
        stale_scratch.bind(
            scratch=stale_bufs,
            a=x_bf16,
            experts=experts._impl,
            topk_weights=weights_full,
            topk_ids=ids_full,
        )

    # Canonical end-to-end typed v2 plan_execution + PreparationSession + frozen replay + CUDA graph.
    exec_plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(
            max_tokens=m_cap, top_k=top_k, warmup_token_counts=(1, 5)
        ),
        invocation={"w4a16_block_size_m": 8},
        override=decode_cfg,
    )

    def _prep_call(m_rows: int):
        def _call(state):
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in state.scratch.scratch_specs()
            )
            out = torch.empty((m_rows, hidden), dtype=torch.float32, device=device)
            ids_tmp, w_tmp = _make_balanced_routes(
                m_rows, top_k, num_experts, seed=991, device=device
            )
            b = state.bind(
                a=x_bf16[:m_rows],
                topk_ids=ids_tmp,
                topk_weights=w_tmp,
                output=out,
                scratch=scratch,
            )
            return PreparedCall(run=lambda: state.run(b), output=out, owners=(scratch, b))

        return _call

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(
            (
                exec_plan.request(
                    name=f"v2-uniform-k4-{codebook}",
                    prepare_calls={cnt: _prep_call(cnt) for cnt in exec_plan.token_counts},
                ),
            )
        )
        session.freeze()
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in exec_plan.scratch_specs()
        )
        output = torch.empty((m_cap, hidden), dtype=torch.float32, device=device)
        with kernel_resolution_guard(f"frozen v2_uniform_k4_{codebook}"):
            for live_m in (1, 5, m_cap):
                output.fill_(float("nan"))
                binding = fused_moe.bind(
                    exec_plan,
                    a=x_bf16[:live_m],
                    topk_ids=ids_full[:live_m],
                    topk_weights=weights_full[:live_m],
                    output=output[:live_m],
                    scratch=scratch,
                )
                fused_moe.run(exec_plan, binding)
                torch.cuda.synchronize(device)
                expected_f32 = reference_exl3_routed_moe_bitwise(
                    x_bf16[:live_m],
                    ids_full[:live_m],
                    weights_full[:live_m],
                    gate_dense,
                    up_dense,
                    down_dense,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    activation="silu",
                    swiglu_limit=None,
                    output_dtype=torch.float32,
                    tile_k=64,
                )
                assert_bitwise_equal(
                    output[:live_m].clone(),
                    expected_f32,
                    f"v2_uniform_k4_m{live_m}_{codebook}",
                )

            graph = torch.cuda.CUDAGraph()
            try:
                bind_g = fused_moe.bind(
                    exec_plan,
                    a=x_bf16[:5],
                    topk_ids=ids_full[:5],
                    topk_weights=weights_full[:5],
                    output=output[:5],
                    scratch=scratch,
                )
                with torch.cuda.graph(graph):
                    fused_moe.run(exec_plan, bind_g)
                expected_g = reference_exl3_routed_moe_bitwise(
                    x_bf16[:5],
                    ids_full[:5],
                    weights_full[:5],
                    gate_dense,
                    up_dense,
                    down_dense,
                    gate_suh,
                    up_suh,
                    inter_rot,
                    down_svh,
                    activation="silu",
                    swiglu_limit=None,
                    output_dtype=torch.float32,
                    tile_k=64,
                )
                for _ in range(2):
                    output[:5].fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert_bitwise_equal(
                        output[:5].clone(),
                        expected_g,
                        f"v2_uniform_k4_graph_{codebook}",
                    )
            finally:
                graph.reset()





