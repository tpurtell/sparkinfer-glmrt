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
    _W4A16GemmLaunch,
    compile_w4a16_fused_moe,
    compile_w4a16_gemm,
    compile_w4a16_topk_sum,
    pack_topk_routes_by_expert,
    run_trellis256_dense,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    MixedTrellisRotations,
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
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_dense_weight,
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

            s0 = acc[0, 0] + acc[1, 0]
            s1 = acc[0, 1] + acc[1, 1]
            s2 = acc[0, 2] + acc[1, 2]
            s3 = acc[0, 3] + acc[1, 3]
            if cutlass.const_expr(chains == 4):
                s0 = (s0 + acc[2, 0]) + acc[3, 0]
                s1 = (s1 + acc[2, 1]) + acc[3, 1]
                s2 = (s2 + acc[2, 2]) + acc[3, 2]
                s3 = (s3 + acc[2, 3]) + acc[3, 3]

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
) -> torch.Tensor:
    """Single-kernel post-FC1 ``H128 -> * svg/svu -> gated_act -> * sd -> H128 -> fp16``."""
    rows, intermediate = fc1_g_f16.shape
    assert fc1_u_f16.shape == (rows, intermediate)
    assert rot_row_f16.shape == (3 * intermediate,)
    assert intermediate % 128 == 0
    nblk = intermediate // 128
    total_units = rows * nblk
    out_f16 = torch.empty((rows, intermediate), dtype=torch.float16, device=fc1_g_f16.device)
    g_flat = fc1_g_f16.contiguous().reshape(-1)
    u_flat = fc1_u_f16.contiguous().reshape(-1)
    r_flat = rot_row_f16.contiguous().reshape(-1)
    o_flat = out_f16.reshape(-1)

    key = (activation, swiglu_limit, bool(fast_math), rows, intermediate)
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
    return out_f16


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
    """Bitwise check of GPTQModel-packed ``.trellis`` through W4A16GemmKernel (no trellis_linear edits)."""
    device = torch.device("cuda", torch.cuda.current_device())
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    m, k, n = 8, 256, 256
    proj = make_gptqmodel_exl3_projection(
        k, n, bits, codebook=codebook, seed=10_000 + bits * 17 + (1 if codebook == "mul1" else 0), device=device
    )
    decoded_b = decode_exl3_projection_cpu(proj, bits).to(device=device)

    unit_k = torch.ones((k,), dtype=torch.float16, device=device)
    unit_n = torch.ones((n,), dtype=torch.float16, device=device)
    weight = prepare_trellis256_dense_weight(
        proj.trellis,
        unit_k,
        unit_n,
        codebook=codebook,
        params_dtype=torch.float16,
    )
    assert weight.trellis_codebook == codebook
    assert weight.trellis.data_ptr() == proj.trellis.data_ptr()

    rng = torch.Generator(device=device).manual_seed(20_000 + bits)
    x = (torch.randn((m, k), generator=rng, device=device) * 0.125).to(torch.float16)
    output = torch.empty((m, n), dtype=torch.float16, device=device)
    gemm_output = torch.empty((m, n), dtype=torch.float16, device=device)
    rotated_f16 = torch.empty((m, k), dtype=torch.float16, device=device)
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

    def identity_hadamard(src: torch.Tensor, dst: torch.Tensor, _ls, _rs, _sc: float) -> None:
        dst.copy_(src)

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
        dense_route_fast_path=True,
    )
    launch = _W4A16GemmLaunch(kernel=compiled_gemm, c_tmp=c_tmp)
    output.fill_(float("nan"))
    actual = run_trellis256_dense(
        x,
        weight,
        launch=launch,
        execution_lut=None,
        grid_cap=sms * compiled_gemm.blocks_per_sm,
        output=output,
        gemm_output=gemm_output,
        rotated_f16=rotated_f16,
        hadamard_128=identity_hadamard,
    ).clone()
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

    mapping, descriptor = build_projection_tiered_maps(
        gate_tiers,
        up_tiers,
        down_tiers,
        tier_slots=(slots_per_tier,) * num_tiers,
        device=device,
    )
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
