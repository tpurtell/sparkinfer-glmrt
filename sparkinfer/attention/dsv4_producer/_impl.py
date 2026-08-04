from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ...gemm._shared.block_fp8 import (
    BlockFP8LinearBinding,
    BlockFP8LinearScratchCaps,
    BlockFP8LinearScratchPlan,
    BlockFP8LinearWeight,
    block_fp8_linear_mxfp8,
    pack_block_fp8_linear_weight_mxfp8,
    plan_block_fp8_linear_scratch,
)


DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_KV_PAGE_SIZE = 256
DSV4_KV_PAYLOAD_BYTES = 576
DSV4_KV_SCALE_BYTES = 8
DSV4_KV_PAGE_BYTES = 149_760
DSV4_FP8_MAX = 448.0
_SCRATCH_ALIGNMENT = 1024


def _align_up(value: int, alignment: int = _SCRATCH_ALIGNMENT) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


@dataclass(frozen=True)
class DSV4ProducerWeights:
    qkv_rank: BlockFP8LinearWeight
    q: BlockFP8LinearWeight
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    hidden: int
    q_lora_rank: int
    heads: int
    head_dim: int = DSV4_HEAD_DIM


@dataclass(frozen=True, kw_only=True)
class DSV4ProducerCaps:
    device: torch.device | str
    max_tokens: int
    hidden: int
    q_lora_rank: int
    heads: int
    head_dim: int = DSV4_HEAD_DIM
    nope_dim: int = DSV4_NOPE_DIM
    rope_dim: int = DSV4_ROPE_DIM
    page_size: int = DSV4_KV_PAGE_SIZE
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "hidden", int(self.hidden))
        object.__setattr__(self, "q_lora_rank", int(self.q_lora_rank))
        object.__setattr__(self, "heads", int(self.heads))
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.hidden not in (4096, 7168):
            raise ValueError(f"DSV4 producer hidden must be 4096 or 7168, got {self.hidden}")
        if self.q_lora_rank not in (1024, 1536):
            raise ValueError(
                f"DSV4 producer q_lora_rank must be 1024 or 1536, got {self.q_lora_rank}"
            )
        if self.heads not in (64, 128):
            raise ValueError(f"DSV4 producer heads must be 64 or 128, got {self.heads}")
        if (self.head_dim, self.nope_dim, self.rope_dim) != (512, 448, 64):
            raise ValueError(
                "DSV4 producer requires head/nope/rope 512/448/64, got "
                f"{self.head_dim}/{self.nope_dim}/{self.rope_dim}"
            )
        if self.page_size != DSV4_KV_PAGE_SIZE:
            raise ValueError(
                f"DSV4 producer requires {DSV4_KV_PAGE_SIZE}-token pages, got {self.page_size}"
            )
        if self.dtype != torch.bfloat16:
            raise ValueError(f"DSV4 producer requires BF16 activations, got {self.dtype}")


@dataclass(frozen=True)
class _DSV4ProducerScratchLayout:
    nbytes: int
    qkv_linear_offset: int
    qkv_linear_bytes: int
    q_linear_offset: int
    q_linear_bytes: int
    qkv_output_offset: int
    q_rank_offset: int


@dataclass(frozen=True, kw_only=True)
class DSV4ProducerBinding:
    hidden_states: torch.Tensor
    positions: torch.Tensor
    main_slots: torch.Tensor
    cos_sin_cache: torch.Tensor
    main_kv_cache: torch.Tensor
    query: torch.Tensor
    weights: DSV4ProducerWeights
    qkv_linear: BlockFP8LinearBinding
    q_linear: BlockFP8LinearBinding
    qkv_output: torch.Tensor
    q_rank: torch.Tensor
    eps: float

    def run(self) -> torch.Tensor:
        return run_dsv4_producer(binding=self)


@dataclass(frozen=True)
class DSV4ProducerPlan:
    caps: DSV4ProducerCaps
    layout: _DSV4ProducerScratchLayout
    qkv_linear_plan: BlockFP8LinearScratchPlan
    q_linear_plan: BlockFP8LinearScratchPlan
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        main_slots: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        main_kv_cache: torch.Tensor,
        query: torch.Tensor,
        weights: DSV4ProducerWeights,
        eps: float = 1.0e-6,
        expected_m: int | None = None,
    ) -> DSV4ProducerBinding:
        tokens = _validate_runtime_tensors(
            self.caps,
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin_cache,
            main_kv_cache=main_kv_cache,
            query=query,
            eps=eps,
        )
        _validate_weights(self.caps, weights)
        arena = scratch_tensor(scratch, self._scratch_specs, owner="DSV4 producer")

        qkv_scratch = arena.narrow(
            0, self.layout.qkv_linear_offset, self.layout.qkv_linear_bytes
        )
        q_scratch = arena.narrow(0, self.layout.q_linear_offset, self.layout.q_linear_bytes)
        qkv_output = (
            arena.narrow(
                0,
                self.layout.qkv_output_offset,
                self.caps.max_tokens
                * (self.caps.q_lora_rank + self.caps.head_dim)
                * torch.empty((), dtype=torch.bfloat16).element_size(),
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.q_lora_rank + self.caps.head_dim, 1)
            .narrow(0, 0, tokens)
        )
        q_rank = (
            arena.narrow(
                0,
                self.layout.q_rank_offset,
                self.caps.max_tokens
                * self.caps.q_lora_rank
                * torch.empty((), dtype=torch.bfloat16).element_size(),
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.q_lora_rank)
            .narrow(0, 0, tokens)
        )
        qkv_linear = self.qkv_linear_plan.bind(
            scratch=qkv_scratch,
            source=hidden_states,
            packed_weight=weights.qkv_rank,
            output=qkv_output,
            expected_m=expected_m,
            activation_block_size=128,
        )
        q_linear = self.q_linear_plan.bind(
            scratch=q_scratch,
            source=q_rank,
            packed_weight=weights.q,
            output=query.view(tokens, self.caps.heads * self.caps.head_dim, 1),
            expected_m=expected_m,
            activation_block_size=128,
        )
        return DSV4ProducerBinding(
            hidden_states=hidden_states,
            positions=positions,
            main_slots=main_slots,
            cos_sin_cache=cos_sin_cache,
            main_kv_cache=main_kv_cache,
            query=query,
            weights=weights,
            qkv_linear=qkv_linear,
            q_linear=q_linear,
            qkv_output=qkv_output,
            q_rank=q_rank,
            eps=float(eps),
        )


def pack_dsv4_producer_weights(
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
) -> DSV4ProducerWeights:
    """Pack checkpoint tensors and concatenate the shared-input projections."""

    for name, tensor in (
        ("wq_a", wq_a),
        ("wq_a_scale", wq_a_scale),
        ("wq_b", wq_b),
        ("wq_b_scale", wq_b_scale),
        ("wkv", wkv),
        ("wkv_scale", wkv_scale),
        ("q_norm", q_norm),
        ("kv_norm", kv_norm),
    ):
        _check_cuda_tensor(name, tensor)
    if wq_a.ndim != 2 or wq_b.ndim != 2 or wkv.ndim != 2:
        raise ValueError("DSV4 producer projection weights must be rank-2")
    q_lora_rank, hidden = map(int, wq_a.shape)
    head_dim, kv_hidden = map(int, wkv.shape)
    q_width, q_input = map(int, wq_b.shape)
    if hidden != kv_hidden or q_input != q_lora_rank:
        raise ValueError(
            "DSV4 producer projection dimensions disagree: "
            f"wq_a={tuple(wq_a.shape)} wq_b={tuple(wq_b.shape)} wkv={tuple(wkv.shape)}"
        )
    if head_dim != DSV4_HEAD_DIM or q_width % head_dim:
        raise ValueError(
            f"DSV4 producer requires wkv N=512 and wq_b N divisible by 512, got {head_dim}/{q_width}"
        )
    heads = q_width // head_dim
    caps = DSV4ProducerCaps(
        device=wq_a.device,
        max_tokens=1,
        hidden=hidden,
        q_lora_rank=q_lora_rank,
        heads=heads,
    )
    del caps
    if q_norm.shape != (q_lora_rank,) or kv_norm.shape != (head_dim,):
        raise ValueError(
            f"DSV4 norm shapes must be {(q_lora_rank,)} and {(head_dim,)}, got "
            f"{tuple(q_norm.shape)} and {tuple(kv_norm.shape)}"
        )
    if q_norm.dtype != torch.bfloat16 or kv_norm.dtype != torch.bfloat16:
        raise ValueError("DSV4 producer norm weights must be BF16")
    if not q_norm.is_contiguous() or not kv_norm.is_contiguous():
        raise ValueError("DSV4 producer norm weights must be contiguous")
    if any(tensor.device != wq_a.device for tensor in (wq_b, wkv, q_norm, kv_norm)):
        raise ValueError("DSV4 producer weights must share one CUDA device")

    # Both projections consume the identical checkpoint activation-quantized
    # hidden row. Concatenating their N dimension is exact because q_lora_rank
    # and head_dim are independently aligned to the checkpoint's N128 blocks.
    qkv_weight = torch.cat((wq_a, wkv), dim=0)
    qkv_scale = torch.cat((wq_a_scale, wkv_scale), dim=0)
    return DSV4ProducerWeights(
        qkv_rank=pack_block_fp8_linear_weight_mxfp8(qkv_weight, qkv_scale),
        q=pack_block_fp8_linear_weight_mxfp8(wq_b, wq_b_scale),
        q_norm=q_norm.detach(),
        kv_norm=kv_norm.detach(),
        hidden=hidden,
        q_lora_rank=q_lora_rank,
        heads=heads,
        head_dim=head_dim,
    )


def plan_dsv4_producer(caps: DSV4ProducerCaps) -> DSV4ProducerPlan:
    qkv_linear_plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=caps.device,
            max_tokens=caps.max_tokens,
            in_features=caps.hidden,
            out_features=caps.q_lora_rank + caps.head_dim,
            output_dtype=caps.dtype,
        )
    )
    q_linear_plan = plan_block_fp8_linear_scratch(
        BlockFP8LinearScratchCaps(
            device=caps.device,
            max_tokens=caps.max_tokens,
            in_features=caps.q_lora_rank,
            out_features=caps.heads * caps.head_dim,
            output_dtype=caps.dtype,
        )
    )
    cursor = 0
    qkv_linear_offset = _align_up(cursor)
    qkv_linear_bytes = qkv_linear_plan.scratch_specs()[0].nbytes
    cursor = qkv_linear_offset + qkv_linear_bytes
    q_linear_offset = _align_up(cursor)
    q_linear_bytes = q_linear_plan.scratch_specs()[0].nbytes
    cursor = q_linear_offset + q_linear_bytes
    qkv_output_offset = _align_up(cursor)
    cursor = qkv_output_offset + caps.max_tokens * (caps.q_lora_rank + caps.head_dim) * 2
    q_rank_offset = _align_up(cursor)
    cursor = q_rank_offset + caps.max_tokens * caps.q_lora_rank * 2
    layout = _DSV4ProducerScratchLayout(
        nbytes=_align_up(cursor),
        qkv_linear_offset=qkv_linear_offset,
        qkv_linear_bytes=qkv_linear_bytes,
        q_linear_offset=q_linear_offset,
        q_linear_bytes=q_linear_bytes,
        qkv_output_offset=qkv_output_offset,
        q_rank_offset=q_rank_offset,
    )
    return DSV4ProducerPlan(
        caps=caps,
        layout=layout,
        qkv_linear_plan=qkv_linear_plan,
        q_linear_plan=q_linear_plan,
        _scratch_specs=(
            scratch_buffer_spec(
                "dsv4_producer.scratch", nbytes=layout.nbytes, device=caps.device
            ),
        ),
    )


def _validate_weights(caps: DSV4ProducerCaps, weights: DSV4ProducerWeights) -> None:
    expected = (caps.hidden, caps.q_lora_rank, caps.heads, caps.head_dim)
    actual = (weights.hidden, weights.q_lora_rank, weights.heads, weights.head_dim)
    if actual != expected:
        raise ValueError(f"DSV4 producer weights {actual} do not match caps {expected}")
    if weights.qkv_rank.in_features != caps.hidden or weights.qkv_rank.out_features != (
        caps.q_lora_rank + caps.head_dim
    ):
        raise ValueError("DSV4 joint Q-rank/KV packed weight has drifted geometry")
    if weights.q.in_features != caps.q_lora_rank or weights.q.out_features != (
        caps.heads * caps.head_dim
    ):
        raise ValueError("DSV4 Q-B packed weight has drifted geometry")


def _validate_runtime_tensors(
    caps: DSV4ProducerCaps,
    *,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    main_slots: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    main_kv_cache: torch.Tensor,
    query: torch.Tensor,
    eps: float,
) -> int:
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("positions", positions),
        ("main_slots", main_slots),
        ("cos_sin_cache", cos_sin_cache),
        ("main_kv_cache", main_kv_cache),
        ("query", query),
    ):
        _check_cuda_tensor(name, tensor)
        if tensor.device != caps.device:
            raise ValueError(f"{name} device {tensor.device} does not match {caps.device}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != caps.hidden:
        raise ValueError(
            f"hidden_states must have shape [tokens,{caps.hidden}], got {tuple(hidden_states.shape)}"
        )
    tokens = int(hidden_states.shape[0])
    if tokens <= 0 or tokens > caps.max_tokens:
        raise ValueError(f"tokens must be in [1,{caps.max_tokens}], got {tokens}")
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    if positions.shape != (tokens,) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"positions must be int32/int64 [{tokens}]")
    if main_slots.shape != (tokens,) or main_slots.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"main_slots must be int32/int64 [{tokens}]")
    if not positions.is_contiguous() or not main_slots.is_contiguous():
        raise ValueError("positions and main_slots must be contiguous")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != caps.rope_dim:
        raise ValueError(
            f"cos_sin_cache must have shape [positions,{caps.rope_dim}], got {tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.dtype != torch.float32 or not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous FP32")
    if (
        main_kv_cache.ndim != 2
        or main_kv_cache.shape[1] != DSV4_KV_PAGE_BYTES
        or main_kv_cache.dtype != torch.uint8
        or not main_kv_cache.is_contiguous()
    ):
        raise ValueError(
            f"main_kv_cache must be contiguous uint8 [pages,{DSV4_KV_PAGE_BYTES}]"
        )
    if int(main_kv_cache.shape[0]) <= 0:
        raise ValueError("main_kv_cache must contain at least one physical page")
    if query.shape != (tokens, caps.heads, caps.head_dim):
        raise ValueError(
            f"query must have shape {(tokens, caps.heads, caps.head_dim)}, got {tuple(query.shape)}"
        )
    if query.dtype != torch.bfloat16 or not query.is_contiguous():
        raise ValueError("query must be contiguous BF16")
    if not math.isfinite(float(eps)) or not float(eps) > 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    return tokens


@triton.jit
def _normalize_rank_pack_kv_kernel(
    qkv,
    q_norm,
    kv_norm,
    positions,
    slots,
    cos_sin,
    q_rank_out,
    cache_fp8,
    cache_bf16,
    cache_u8,
    qkv_stride_t,
    q_rank_stride_t,
    cos_sin_stride_pos,
    eps,
    Q_RANK: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token = tl.program_id(0)

    qd = tl.arange(0, Q_BLOCK)
    qmask = qd < Q_RANK
    qv = tl.load(qkv + token * qkv_stride_t + qd, mask=qmask, other=0.0).to(tl.float32)
    q_inv = tl.rsqrt(tl.sum(qv * qv, axis=0) / Q_RANK + eps)
    qw = tl.load(q_norm + qd, mask=qmask, other=0.0).to(tl.float32)
    tl.store(
        q_rank_out + token * q_rank_stride_t + qd,
        (qv * q_inv * qw).to(tl.bfloat16),
        mask=qmask,
    )

    kd = tl.arange(0, HEAD_DIM)
    kv = tl.load(qkv + token * qkv_stride_t + Q_RANK + kd).to(tl.float32)
    k_inv = tl.rsqrt(tl.sum(kv * kv, axis=0) / HEAD_DIM + eps)

    slot = tl.load(slots + token).to(tl.int64)
    page = slot // PAGE_SIZE
    row = slot - page * PAGE_SIZE
    page_base = page * PAGE_BYTES
    data_base = page_base + row * PAYLOAD_BYTES
    scale_base = (
        page_base
        + PAGE_SIZE * PAYLOAD_BYTES
        + row * SCALE_BYTES
    )

    # Store seven independently scaled 64-value NoPE groups. The integer
    # exponent path implements pow2-ceil without an approximate log2 boundary.
    gd = tl.arange(0, 64)
    for group in range(NOPE_DIM // 64):
        dim = group * 64 + gd
        values = tl.load(qkv + token * qkv_stride_t + Q_RANK + dim).to(tl.float32)
        weights = tl.load(kv_norm + dim).to(tl.float32)
        values = (values * k_inv * weights).to(tl.bfloat16).to(tl.float32)
        max_abs = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-4)
        raw_scale = max_abs / FP8_MAX
        bits = raw_scale.to(tl.uint32, bitcast=True)
        mantissa = bits & 0x007FFFFF
        rounded = (bits + 0x00800000) & 0x7F800000
        scale_bits = tl.where(mantissa != 0, rounded, bits & 0x7F800000)
        scale = scale_bits.to(tl.float32, bitcast=True)
        quant = tl.maximum(tl.minimum(values / scale, FP8_MAX), -FP8_MAX)
        tl.store(cache_fp8 + data_base + dim, quant.to(tl.float8e4nv))
        tl.store(cache_u8 + scale_base + group, (scale_bits >> 23).to(tl.uint8))

    # The RoPE cache is [cos(32), sin(32)] and the model uses adjacent complex
    # pairs. Store the rotated BF16 lane directly behind the NoPE payload.
    rd = tl.arange(0, ROPE_DIM)
    rope_raw = tl.load(qkv + token * qkv_stride_t + Q_RANK + NOPE_DIM + rd).to(
        tl.float32
    )
    rope_weight = tl.load(kv_norm + NOPE_DIM + rd).to(tl.float32)
    rope = (rope_raw * k_inv * rope_weight).to(tl.bfloat16).to(tl.float32)
    partner_d = rd ^ 1
    partner_raw = tl.load(
        qkv + token * qkv_stride_t + Q_RANK + NOPE_DIM + partner_d
    ).to(tl.float32)
    partner_weight = tl.load(kv_norm + NOPE_DIM + partner_d).to(tl.float32)
    partner = (partner_raw * k_inv * partner_weight).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(positions + token)
    pair = rd >> 1
    cs = cos_sin + pos * cos_sin_stride_pos
    cos_v = tl.load(cs + pair)
    sin_v = tl.load(cs + ROPE_DIM // 2 + pair)
    rotated = tl.where(
        (rd & 1) == 0,
        rope * cos_v - partner * sin_v,
        rope * cos_v + partner * sin_v,
    )
    rope_bf16_offset = (data_base + NOPE_DIM) // 2
    tl.store(cache_bf16 + rope_bf16_offset + rd, rotated.to(tl.bfloat16))
    tl.store(cache_u8 + scale_base + 7, 0)


@triton.jit
def _normalize_query_rope_kernel(
    query,
    positions,
    cos_sin,
    query_stride_t,
    query_stride_h,
    cos_sin_stride_pos,
    eps,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    base = query + token * query_stride_t + head * query_stride_h
    values = tl.load(base + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, axis=0) / HEAD_DIM + eps)
    normalized = (values * inv).to(tl.bfloat16).to(tl.float32)
    rd = d - NOPE_DIM
    rope_mask = d >= NOPE_DIM
    partner_d = NOPE_DIM + (rd ^ 1)
    partner = tl.load(base + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
    partner = (partner * inv).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(positions + token)
    pair = tl.maximum(rd >> 1, 0)
    cs = cos_sin + pos * cos_sin_stride_pos
    cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
    sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
    rotated = tl.where(
        (rd & 1) == 0,
        normalized * cos_v - partner * sin_v,
        normalized * cos_v + partner * sin_v,
    )
    output = tl.where(rope_mask, rotated, normalized)
    tl.store(base + d, output.to(tl.bfloat16))


def _run_normalize_rank_pack_kv(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_rank_out: torch.Tensor,
    main_kv_cache: torch.Tensor,
    *,
    eps: float,
) -> None:
    tokens = int(qkv.shape[0])
    q_rank = int(q_rank_out.shape[1])
    _normalize_rank_pack_kv_kernel[(tokens,)](
        qkv,
        q_norm,
        kv_norm,
        positions,
        slots,
        cos_sin_cache,
        q_rank_out,
        main_kv_cache.view(torch.float8_e4m3fn),
        main_kv_cache.view(torch.bfloat16),
        main_kv_cache,
        qkv.stride(0),
        q_rank_out.stride(0),
        cos_sin_cache.stride(0),
        float(eps),
        Q_RANK=q_rank,
        Q_BLOCK=triton.next_power_of_2(q_rank),
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_SIZE=DSV4_KV_PAGE_SIZE,
        PAYLOAD_BYTES=DSV4_KV_PAYLOAD_BYTES,
        SCALE_BYTES=DSV4_KV_SCALE_BYTES,
        PAGE_BYTES=DSV4_KV_PAGE_BYTES,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=8,
    )


def _run_normalize_query_rope(
    query: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float,
) -> None:
    tokens, heads, _ = map(int, query.shape)
    _normalize_query_rope_kernel[(tokens, heads)](
        query,
        positions,
        cos_sin_cache,
        query.stride(0),
        query.stride(1),
        cos_sin_cache.stride(0),
        float(eps),
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        num_warps=8,
    )


def run_dsv4_producer(*, binding: DSV4ProducerBinding) -> torch.Tensor:
    """Run the allocation-free DSV4 query/main-KV producer on the current stream."""

    if not isinstance(binding, DSV4ProducerBinding):
        raise TypeError("run_dsv4_producer requires a DSV4ProducerBinding")
    block_fp8_linear_mxfp8(binding=binding.qkv_linear)
    _run_normalize_rank_pack_kv(
        binding.qkv_output[:, :, 0],
        binding.weights.q_norm,
        binding.weights.kv_norm,
        binding.positions,
        binding.main_slots,
        binding.cos_sin_cache,
        binding.q_rank,
        binding.main_kv_cache,
        eps=binding.eps,
    )
    block_fp8_linear_mxfp8(binding=binding.q_linear)
    _run_normalize_query_rope(
        binding.query,
        binding.positions,
        binding.cos_sin_cache,
        eps=binding.eps,
    )
    return binding.query


__all__ = [
    "DSV4ProducerBinding",
    "DSV4ProducerCaps",
    "DSV4ProducerPlan",
    "DSV4ProducerWeights",
    "pack_dsv4_producer_weights",
    "plan_dsv4_producer",
    "run_dsv4_producer",
]
