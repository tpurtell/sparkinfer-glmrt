from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor


DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_SOURCE_PAGE_SIZE = 256
DSV4_KV_PAYLOAD_BYTES = 576
DSV4_KV_SCALE_BYTES = 8
DSV4_INDEX_HEAD_DIM = 128
DSV4_INDEX_PAGE_SIZE = 64
DSV4_INDEX_PAGE_BYTES = 8_448
DSV4_FP8_MAX = 448.0
_SCRATCH_ALIGNMENT = 1_024


def _align_up(value: int, alignment: int = _SCRATCH_ALIGNMENT) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _compressed_main_page_bytes(ratio: int) -> int:
    rows = DSV4_SOURCE_PAGE_SIZE // int(ratio)
    return _align_up(
        rows * (DSV4_KV_PAYLOAD_BYTES + DSV4_KV_SCALE_BYTES),
        DSV4_KV_PAYLOAD_BYTES,
    )


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


@dataclass(frozen=True)
class DSV4CompressorWeights:
    joint_projection: torch.Tensor
    joint_projection_t: torch.Tensor
    main_ape: torch.Tensor
    main_norm: torch.Tensor
    index_ape: torch.Tensor | None
    index_norm: torch.Tensor | None
    hidden: int
    compress_ratio: int
    with_indexer: bool


@dataclass(frozen=True, kw_only=True)
class DSV4CompressorCaps:
    device: torch.device | str
    max_tokens: int
    hidden: int
    compress_ratio: int
    with_indexer: bool
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "hidden", int(self.hidden))
        object.__setattr__(self, "compress_ratio", int(self.compress_ratio))
        object.__setattr__(self, "with_indexer", bool(self.with_indexer))
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.hidden not in (4_096, 7_168):
            raise ValueError(
                f"DSV4 compressor hidden must be 4096 or 7168, got {self.hidden}"
            )
        if self.compress_ratio not in (4, 128):
            raise ValueError(
                f"DSV4 compressor ratio must be 4 or 128, got {self.compress_ratio}"
            )
        if self.with_indexer != (self.compress_ratio == 4):
            raise ValueError(
                "DSV4 C=4 layers require the learned index compressor and "
                "C=128 layers must not carry one"
            )
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"DSV4 compressor activations must be BF16, got {self.dtype}"
            )

    @property
    def overlap(self) -> bool:
        return self.compress_ratio == 4

    @property
    def coefficient(self) -> int:
        return 2 if self.overlap else 1

    @property
    def main_projected_width(self) -> int:
        return self.coefficient * DSV4_HEAD_DIM

    @property
    def index_projected_width(self) -> int:
        return self.coefficient * DSV4_INDEX_HEAD_DIM if self.with_indexer else 0

    @property
    def joint_projection_width(self) -> int:
        return 2 * (self.main_projected_width + self.index_projected_width)

    @property
    def state_rows(self) -> int:
        return self.coefficient * self.compress_ratio

    @property
    def main_page_rows(self) -> int:
        return DSV4_SOURCE_PAGE_SIZE // self.compress_ratio

    @property
    def main_page_bytes(self) -> int:
        return _compressed_main_page_bytes(self.compress_ratio)


@dataclass(frozen=True)
class _DSV4CompressorScratchLayout:
    nbytes: int
    projection_offset: int
    projection_bytes: int


@dataclass(frozen=True, kw_only=True)
class DSV4CompressorBinding:
    hidden_states: torch.Tensor
    positions: torch.Tensor
    sequence_ids: torch.Tensor
    compressed_slots: torch.Tensor
    compressed_cos_sin_cache: torch.Tensor
    compressed_main_cache: torch.Tensor
    main_kv_state: torch.Tensor
    main_score_state: torch.Tensor
    index_cache: torch.Tensor | None
    index_kv_state: torch.Tensor | None
    index_score_state: torch.Tensor | None
    weights: DSV4CompressorWeights
    projection: torch.Tensor
    compress_ratio: int
    eps: float

    def run_decode(self) -> None:
        run_dsv4_compressor_decode(binding=self)


@dataclass(frozen=True, kw_only=True)
class DSV4CompressorPrefillBinding:
    hidden_states: torch.Tensor
    active_groups: torch.Tensor
    group_source_starts: torch.Tensor
    group_rope_positions: torch.Tensor
    compressed_slots: torch.Tensor
    active_sequences: torch.Tensor
    sequence_offsets: torch.Tensor
    state_sequence_ids: torch.Tensor
    compressed_cos_sin_cache: torch.Tensor
    compressed_main_cache: torch.Tensor
    main_kv_state: torch.Tensor
    main_score_state: torch.Tensor
    index_cache: torch.Tensor | None
    index_kv_state: torch.Tensor | None
    index_score_state: torch.Tensor | None
    weights: DSV4CompressorWeights
    projection: torch.Tensor
    compress_ratio: int
    eps: float

    @property
    def group_capacity(self) -> int:
        return int(self.group_source_starts.shape[0])

    @property
    def sequence_capacity(self) -> int:
        return int(self.state_sequence_ids.shape[0])

    def run_prefill(self) -> None:
        run_dsv4_compressor_prefill(binding=self)


@dataclass(frozen=True, kw_only=True)
class DSV4CompressorContinuationBinding:
    hidden_states: torch.Tensor
    active_groups: torch.Tensor
    group_sequence_slots: torch.Tensor
    group_source_positions: torch.Tensor
    group_rope_positions: torch.Tensor
    compressed_slots: torch.Tensor
    active_sequences: torch.Tensor
    sequence_offsets: torch.Tensor
    sequence_start_positions: torch.Tensor
    state_sequence_ids: torch.Tensor
    compressed_cos_sin_cache: torch.Tensor
    compressed_main_cache: torch.Tensor
    main_kv_state: torch.Tensor
    main_score_state: torch.Tensor
    index_cache: torch.Tensor | None
    index_kv_state: torch.Tensor | None
    index_score_state: torch.Tensor | None
    weights: DSV4CompressorWeights
    projection: torch.Tensor
    compress_ratio: int
    eps: float

    @property
    def group_capacity(self) -> int:
        return int(self.group_sequence_slots.shape[0])

    @property
    def sequence_capacity(self) -> int:
        return int(self.state_sequence_ids.shape[0])

    def run_continuation(self) -> None:
        run_dsv4_compressor_continuation(binding=self)


@dataclass(frozen=True)
class DSV4CompressorPlan:
    caps: DSV4CompressorCaps
    layout: _DSV4CompressorScratchLayout
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind_decode(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        sequence_ids: torch.Tensor,
        compressed_slots: torch.Tensor,
        compressed_cos_sin_cache: torch.Tensor,
        compressed_main_cache: torch.Tensor,
        main_kv_state: torch.Tensor,
        main_score_state: torch.Tensor,
        weights: DSV4CompressorWeights,
        index_cache: torch.Tensor | None = None,
        index_kv_state: torch.Tensor | None = None,
        index_score_state: torch.Tensor | None = None,
        eps: float = 1.0e-6,
        expected_m: int | None = None,
        rows_are_sequence_unique: bool = False,
    ) -> DSV4CompressorBinding:
        if not rows_are_sequence_unique:
            raise ValueError(
                "decode rows must be acknowledged as sequence-unique; ordered "
                "multi-token sequence updates require the future prefill binding"
            )
        tokens = _validate_runtime_tensors(
            self.caps,
            hidden_states=hidden_states,
            positions=positions,
            sequence_ids=sequence_ids,
            compressed_slots=compressed_slots,
            compressed_cos_sin_cache=compressed_cos_sin_cache,
            compressed_main_cache=compressed_main_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            eps=eps,
        )
        if expected_m is not None and tokens != int(expected_m):
            raise ValueError(f"expected_m={expected_m} does not match tokens={tokens}")
        _validate_weights(self.caps, weights)
        arena = scratch_tensor(scratch, self._scratch_specs, owner="DSV4 compressor")
        projection = (
            arena.narrow(
                0,
                self.layout.projection_offset,
                self.layout.projection_bytes,
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.joint_projection_width)
            .narrow(0, 0, tokens)
        )
        return DSV4CompressorBinding(
            hidden_states=hidden_states,
            positions=positions,
            sequence_ids=sequence_ids,
            compressed_slots=compressed_slots,
            compressed_cos_sin_cache=compressed_cos_sin_cache,
            compressed_main_cache=compressed_main_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            weights=weights,
            projection=projection,
            compress_ratio=self.caps.compress_ratio,
            eps=float(eps),
        )

    def bind_prefill(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        hidden_states: torch.Tensor,
        active_groups: torch.Tensor,
        group_source_starts: torch.Tensor,
        group_rope_positions: torch.Tensor,
        compressed_slots: torch.Tensor,
        active_sequences: torch.Tensor,
        sequence_offsets: torch.Tensor,
        state_sequence_ids: torch.Tensor,
        compressed_cos_sin_cache: torch.Tensor,
        compressed_main_cache: torch.Tensor,
        main_kv_state: torch.Tensor,
        main_score_state: torch.Tensor,
        weights: DSV4CompressorWeights,
        index_cache: torch.Tensor | None = None,
        index_kv_state: torch.Tensor | None = None,
        index_score_state: torch.Tensor | None = None,
        eps: float = 1.0e-6,
        expected_m: int | None = None,
        initial_prefill: bool = False,
    ) -> DSV4CompressorPrefillBinding:
        if not initial_prefill:
            raise ValueError(
                "prefill must be acknowledged as starting at logical position zero; "
                "ordered continuation chunks require a future binding"
            )
        tokens = _validate_runtime_tensors(
            self.caps,
            hidden_states=hidden_states,
            positions=group_source_starts,
            sequence_ids=group_rope_positions,
            compressed_slots=compressed_slots,
            compressed_cos_sin_cache=compressed_cos_sin_cache,
            compressed_main_cache=compressed_main_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            eps=eps,
            metadata_tokens=int(group_source_starts.shape[0]),
        )
        _validate_prefill_metadata(
            self.caps,
            active_groups=active_groups,
            group_source_starts=group_source_starts,
            group_rope_positions=group_rope_positions,
            compressed_slots=compressed_slots,
            active_sequences=active_sequences,
            sequence_offsets=sequence_offsets,
            state_sequence_ids=state_sequence_ids,
            state_sequences=int(main_kv_state.shape[0]),
        )
        if expected_m is not None and tokens != int(expected_m):
            raise ValueError(f"expected_m={expected_m} does not match tokens={tokens}")
        _validate_weights(self.caps, weights)
        arena = scratch_tensor(scratch, self._scratch_specs, owner="DSV4 compressor")
        projection = (
            arena.narrow(
                0,
                self.layout.projection_offset,
                self.layout.projection_bytes,
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.joint_projection_width)
            .narrow(0, 0, tokens)
        )
        return DSV4CompressorPrefillBinding(
            hidden_states=hidden_states,
            active_groups=active_groups,
            group_source_starts=group_source_starts,
            group_rope_positions=group_rope_positions,
            compressed_slots=compressed_slots,
            active_sequences=active_sequences,
            sequence_offsets=sequence_offsets,
            state_sequence_ids=state_sequence_ids,
            compressed_cos_sin_cache=compressed_cos_sin_cache,
            compressed_main_cache=compressed_main_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            weights=weights,
            projection=projection,
            compress_ratio=self.caps.compress_ratio,
            eps=float(eps),
        )

    def bind_continuation(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        hidden_states: torch.Tensor,
        active_groups: torch.Tensor,
        group_sequence_slots: torch.Tensor,
        group_source_positions: torch.Tensor,
        group_rope_positions: torch.Tensor,
        compressed_slots: torch.Tensor,
        active_sequences: torch.Tensor,
        sequence_offsets: torch.Tensor,
        sequence_start_positions: torch.Tensor,
        state_sequence_ids: torch.Tensor,
        compressed_cos_sin_cache: torch.Tensor,
        compressed_main_cache: torch.Tensor,
        main_kv_state: torch.Tensor,
        main_score_state: torch.Tensor,
        weights: DSV4CompressorWeights,
        index_cache: torch.Tensor | None = None,
        index_kv_state: torch.Tensor | None = None,
        index_score_state: torch.Tensor | None = None,
        eps: float = 1.0e-6,
        expected_m: int | None = None,
        ordered_continuation: bool = False,
    ) -> DSV4CompressorContinuationBinding:
        if not ordered_continuation:
            raise ValueError(
                "continuation must be acknowledged as one ordered chunk per "
                "sequence with persistent state from every preceding token"
            )
        tokens = _validate_runtime_tensors(
            self.caps,
            hidden_states=hidden_states,
            positions=group_sequence_slots,
            sequence_ids=group_rope_positions,
            compressed_slots=compressed_slots,
            compressed_cos_sin_cache=compressed_cos_sin_cache,
            compressed_main_cache=compressed_main_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            eps=eps,
            metadata_tokens=int(group_sequence_slots.shape[0]),
        )
        _validate_continuation_metadata(
            self.caps,
            active_groups=active_groups,
            group_sequence_slots=group_sequence_slots,
            group_source_positions=group_source_positions,
            group_rope_positions=group_rope_positions,
            compressed_slots=compressed_slots,
            active_sequences=active_sequences,
            sequence_offsets=sequence_offsets,
            sequence_start_positions=sequence_start_positions,
            state_sequence_ids=state_sequence_ids,
            state_sequences=int(main_kv_state.shape[0]),
        )
        if expected_m is not None and tokens != int(expected_m):
            raise ValueError(f"expected_m={expected_m} does not match tokens={tokens}")
        _validate_weights(self.caps, weights)
        arena = scratch_tensor(scratch, self._scratch_specs, owner="DSV4 compressor")
        projection = (
            arena.narrow(
                0,
                self.layout.projection_offset,
                self.layout.projection_bytes,
            )
            .view(torch.bfloat16)
            .view(self.caps.max_tokens, self.caps.joint_projection_width)
            .narrow(0, 0, tokens)
        )
        return DSV4CompressorContinuationBinding(
            hidden_states=hidden_states,
            active_groups=active_groups,
            group_sequence_slots=group_sequence_slots,
            group_source_positions=group_source_positions,
            group_rope_positions=group_rope_positions,
            compressed_slots=compressed_slots,
            active_sequences=active_sequences,
            sequence_offsets=sequence_offsets,
            sequence_start_positions=sequence_start_positions,
            state_sequence_ids=state_sequence_ids,
            compressed_cos_sin_cache=compressed_cos_sin_cache,
            compressed_main_cache=compressed_main_cache,
            main_kv_state=main_kv_state,
            main_score_state=main_score_state,
            index_cache=index_cache,
            index_kv_state=index_kv_state,
            index_score_state=index_score_state,
            weights=weights,
            projection=projection,
            compress_ratio=self.caps.compress_ratio,
            eps=float(eps),
        )


def pack_dsv4_compressor_weights(
    main_wkv: torch.Tensor,
    main_wgate: torch.Tensor,
    main_ape: torch.Tensor,
    main_norm: torch.Tensor,
    *,
    index_wkv: torch.Tensor | None = None,
    index_wgate: torch.Tensor | None = None,
    index_ape: torch.Tensor | None = None,
    index_norm: torch.Tensor | None = None,
) -> DSV4CompressorWeights:
    """Concatenate checkpoint BF16 projections once at load time."""

    for name, tensor in (
        ("main_wkv", main_wkv),
        ("main_wgate", main_wgate),
        ("main_ape", main_ape),
        ("main_norm", main_norm),
    ):
        _check_cuda_tensor(name, tensor)
    if main_wkv.ndim != 2 or main_wgate.ndim != 2 or main_wkv.shape != main_wgate.shape:
        raise ValueError("main_wkv and main_wgate must have one identical rank-2 shape")
    projected_width, hidden = map(int, main_wkv.shape)
    if projected_width == 2 * DSV4_HEAD_DIM:
        ratio = 4
    elif projected_width == DSV4_HEAD_DIM:
        ratio = 128
    else:
        raise ValueError(
            f"main compressor projected width must be 1024 or 512, got {projected_width}"
        )
    with_indexer = ratio == 4
    caps = DSV4CompressorCaps(
        device=main_wkv.device,
        max_tokens=1,
        hidden=hidden,
        compress_ratio=ratio,
        with_indexer=with_indexer,
    )
    if main_ape.shape != (ratio, projected_width):
        raise ValueError(
            f"main_ape must have shape {(ratio, projected_width)}, got {tuple(main_ape.shape)}"
        )
    if main_norm.shape != (DSV4_HEAD_DIM,):
        raise ValueError(f"main_norm must have shape {(DSV4_HEAD_DIM,)}")
    if main_wkv.dtype != torch.bfloat16 or main_wgate.dtype != torch.bfloat16:
        raise ValueError(
            "compressor projection weights must be BF16 checkpoint tensors"
        )
    if main_ape.dtype != torch.float32 or main_norm.dtype != torch.bfloat16:
        raise ValueError("compressor APE must be FP32 and norm weight must be BF16")

    optional = (index_wkv, index_wgate, index_ape, index_norm)
    if with_indexer and any(tensor is None for tensor in optional):
        raise ValueError("C=4 compressor weights require the complete index compressor")
    if not with_indexer and any(tensor is not None for tensor in optional):
        raise ValueError("C=128 compressor weights cannot carry an index compressor")
    tensors = [main_wkv, main_wgate, main_ape, main_norm]
    projections = [main_wkv, main_wgate]
    if with_indexer:
        assert index_wkv is not None
        assert index_wgate is not None
        assert index_ape is not None
        assert index_norm is not None
        if index_wkv.shape != (256, hidden) or index_wgate.shape != (256, hidden):
            raise ValueError("C=4 index wkv/wgate must each have shape [256, hidden]")
        if index_ape.shape != (4, 256) or index_norm.shape != (128,):
            raise ValueError("C=4 index APE/norm must have shapes [4,256] and [128]")
        if index_wkv.dtype != torch.bfloat16 or index_wgate.dtype != torch.bfloat16:
            raise ValueError("index compressor projection weights must be BF16")
        if index_ape.dtype != torch.float32 or index_norm.dtype != torch.bfloat16:
            raise ValueError("index compressor APE must be FP32 and norm must be BF16")
        tensors.extend(optional)  # type: ignore[arg-type]
        projections.extend((index_wkv, index_wgate))
    if any(tensor.device != main_wkv.device for tensor in tensors):
        raise ValueError("DSV4 compressor weights must share one CUDA device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("DSV4 compressor checkpoint tensors must be contiguous")

    joint_projection = torch.cat(projections, dim=0).contiguous()
    if tuple(joint_projection.shape) != (caps.joint_projection_width, hidden):
        raise AssertionError(
            "internal DSV4 compressor joint projection geometry drifted"
        )
    return DSV4CompressorWeights(
        joint_projection=joint_projection,
        joint_projection_t=joint_projection.t(),
        main_ape=main_ape.detach(),
        main_norm=main_norm.detach(),
        index_ape=index_ape.detach() if index_ape is not None else None,
        index_norm=index_norm.detach() if index_norm is not None else None,
        hidden=hidden,
        compress_ratio=ratio,
        with_indexer=with_indexer,
    )


def plan_dsv4_compressor(caps: DSV4CompressorCaps) -> DSV4CompressorPlan:
    projection_offset = 0
    projection_bytes = caps.max_tokens * caps.joint_projection_width * 2
    layout = _DSV4CompressorScratchLayout(
        nbytes=_align_up(projection_bytes),
        projection_offset=projection_offset,
        projection_bytes=projection_bytes,
    )
    return DSV4CompressorPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "dsv4_compressor.scratch", nbytes=layout.nbytes, device=caps.device
            ),
        ),
    )


def _validate_weights(caps: DSV4CompressorCaps, weights: DSV4CompressorWeights) -> None:
    expected = (caps.hidden, caps.compress_ratio, caps.with_indexer)
    actual = (weights.hidden, weights.compress_ratio, weights.with_indexer)
    if actual != expected:
        raise ValueError(
            f"DSV4 compressor weights {actual} do not match caps {expected}"
        )
    if weights.joint_projection.shape != (caps.joint_projection_width, caps.hidden):
        raise ValueError("DSV4 compressor joint projection geometry drifted")
    if weights.joint_projection_t.shape != (caps.hidden, caps.joint_projection_width):
        raise ValueError("DSV4 compressor transposed projection view geometry drifted")


def _validate_runtime_tensors(
    caps: DSV4CompressorCaps,
    *,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    sequence_ids: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_cos_sin_cache: torch.Tensor,
    compressed_main_cache: torch.Tensor,
    main_kv_state: torch.Tensor,
    main_score_state: torch.Tensor,
    index_cache: torch.Tensor | None,
    index_kv_state: torch.Tensor | None,
    index_score_state: torch.Tensor | None,
    eps: float,
    metadata_tokens: int | None = None,
) -> int:
    required = (
        ("hidden_states", hidden_states),
        ("positions", positions),
        ("sequence_ids", sequence_ids),
        ("compressed_slots", compressed_slots),
        ("compressed_cos_sin_cache", compressed_cos_sin_cache),
        ("compressed_main_cache", compressed_main_cache),
        ("main_kv_state", main_kv_state),
        ("main_score_state", main_score_state),
    )
    for name, tensor in required:
        _check_cuda_tensor(name, tensor)
        if tensor.device != caps.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {caps.device}"
            )
    if hidden_states.ndim != 2 or hidden_states.shape[1] != caps.hidden:
        raise ValueError(f"hidden_states must have shape [tokens,{caps.hidden}]")
    tokens = int(hidden_states.shape[0])
    if not 1 <= tokens <= caps.max_tokens:
        raise ValueError(f"tokens must be in [1,{caps.max_tokens}], got {tokens}")
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    metadata_rows = tokens if metadata_tokens is None else int(metadata_tokens)
    for name, tensor in (
        ("positions", positions),
        ("sequence_ids", sequence_ids),
        ("compressed_slots", compressed_slots),
    ):
        if (
            tensor.shape != (metadata_rows,)
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32 [{metadata_rows}]")
    if (
        compressed_cos_sin_cache.ndim != 2
        or compressed_cos_sin_cache.shape[1] != DSV4_ROPE_DIM
        or compressed_cos_sin_cache.dtype != torch.float32
        or not compressed_cos_sin_cache.is_contiguous()
    ):
        raise ValueError(
            "compressed_cos_sin_cache must be contiguous FP32 [positions,64]"
        )
    if (
        compressed_main_cache.ndim != 2
        or compressed_main_cache.shape[1] != caps.main_page_bytes
        or compressed_main_cache.dtype != torch.uint8
        or not compressed_main_cache.is_contiguous()
        or compressed_main_cache.shape[0] <= 0
    ):
        raise ValueError(
            f"compressed_main_cache must be contiguous uint8 [pages,{caps.main_page_bytes}]"
        )
    if main_kv_state.shape != main_score_state.shape:
        raise ValueError("main KV and score state shapes must match")
    expected_tail = (caps.state_rows, caps.main_projected_width)
    if main_kv_state.ndim != 3 or tuple(main_kv_state.shape[1:]) != expected_tail:
        raise ValueError(
            f"main compressor states must have shape [sequences,{expected_tail}]"
        )
    if int(main_kv_state.shape[0]) <= 0:
        raise ValueError("main compressor state must contain at least one sequence")
    for name, state in (
        ("main_kv_state", main_kv_state),
        ("main_score_state", main_score_state),
    ):
        if state.dtype != torch.float32 or not state.is_contiguous():
            raise ValueError(f"{name} must be contiguous FP32")

    index_tensors = (index_cache, index_kv_state, index_score_state)
    if caps.with_indexer:
        if any(tensor is None for tensor in index_tensors):
            raise ValueError("C=4 runtime requires index cache and paired index states")
        assert index_cache is not None
        assert index_kv_state is not None
        assert index_score_state is not None
        for name, tensor in (
            ("index_cache", index_cache),
            ("index_kv_state", index_kv_state),
            ("index_score_state", index_score_state),
        ):
            _check_cuda_tensor(name, tensor)
            if tensor.device != caps.device:
                raise ValueError(
                    f"{name} device {tensor.device} does not match {caps.device}"
                )
        if (
            index_cache.ndim != 2
            or index_cache.shape[1] != DSV4_INDEX_PAGE_BYTES
            or index_cache.dtype != torch.uint8
            or not index_cache.is_contiguous()
            or index_cache.shape[0] != compressed_main_cache.shape[0]
        ):
            raise ValueError("index_cache must be contiguous uint8 [same_pages,8448]")
        expected_index = (int(main_kv_state.shape[0]), caps.state_rows, 256)
        if (
            index_kv_state.shape != expected_index
            or index_score_state.shape != expected_index
        ):
            raise ValueError(
                f"index compressor states must have shape {expected_index}"
            )
        if any(
            state.dtype != torch.float32 or not state.is_contiguous()
            for state in (index_kv_state, index_score_state)
        ):
            raise ValueError("index compressor states must be contiguous FP32")
    elif any(tensor is not None for tensor in index_tensors):
        raise ValueError("C=128 runtime cannot carry index cache or state")
    if not math.isfinite(float(eps)) or not float(eps) > 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    return tokens


def _validate_prefill_metadata(
    caps: DSV4CompressorCaps,
    *,
    active_groups: torch.Tensor,
    group_source_starts: torch.Tensor,
    group_rope_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    active_sequences: torch.Tensor,
    sequence_offsets: torch.Tensor,
    state_sequence_ids: torch.Tensor,
    state_sequences: int,
) -> None:
    for name, tensor in (
        ("active_groups", active_groups),
        ("group_source_starts", group_source_starts),
        ("group_rope_positions", group_rope_positions),
        ("compressed_slots", compressed_slots),
        ("active_sequences", active_sequences),
        ("sequence_offsets", sequence_offsets),
        ("state_sequence_ids", state_sequence_ids),
    ):
        _check_cuda_tensor(name, tensor)
        if tensor.device != caps.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {caps.device}"
            )
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32")
    if active_groups.numel() != 1 or active_sequences.numel() != 1:
        raise ValueError(
            "active_groups and active_sequences must each contain one int32"
        )
    group_capacity = int(group_source_starts.shape[0])
    if group_source_starts.ndim != 1:
        raise ValueError("group_source_starts must be rank-1")
    if group_rope_positions.shape != (group_capacity,) or compressed_slots.shape != (
        group_capacity,
    ):
        raise ValueError("prefill group metadata arrays must have one shared capacity")
    if group_capacity > caps.max_tokens // caps.compress_ratio:
        raise ValueError(
            f"group capacity {group_capacity} exceeds max initial-prefill groups "
            f"{caps.max_tokens // caps.compress_ratio}"
        )
    if state_sequence_ids.ndim != 1 or state_sequence_ids.numel() <= 0:
        raise ValueError("state_sequence_ids must be a non-empty rank-1 tensor")
    sequence_capacity = int(state_sequence_ids.shape[0])
    if sequence_offsets.shape != (sequence_capacity + 1,):
        raise ValueError(f"sequence_offsets must have shape [{sequence_capacity + 1}]")
    if sequence_capacity > state_sequences:
        raise ValueError(
            f"sequence capacity {sequence_capacity} exceeds state capacity {state_sequences}"
        )


def _validate_continuation_metadata(
    caps: DSV4CompressorCaps,
    *,
    active_groups: torch.Tensor,
    group_sequence_slots: torch.Tensor,
    group_source_positions: torch.Tensor,
    group_rope_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    active_sequences: torch.Tensor,
    sequence_offsets: torch.Tensor,
    sequence_start_positions: torch.Tensor,
    state_sequence_ids: torch.Tensor,
    state_sequences: int,
) -> None:
    for name, tensor in (
        ("active_groups", active_groups),
        ("group_sequence_slots", group_sequence_slots),
        ("group_source_positions", group_source_positions),
        ("group_rope_positions", group_rope_positions),
        ("compressed_slots", compressed_slots),
        ("active_sequences", active_sequences),
        ("sequence_offsets", sequence_offsets),
        ("sequence_start_positions", sequence_start_positions),
        ("state_sequence_ids", state_sequence_ids),
    ):
        _check_cuda_tensor(name, tensor)
        if tensor.device != caps.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {caps.device}"
            )
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32")
    if active_groups.numel() != 1 or active_sequences.numel() != 1:
        raise ValueError(
            "active_groups and active_sequences must each contain one int32"
        )
    group_capacity = int(group_sequence_slots.shape[0])
    if group_sequence_slots.ndim != 1:
        raise ValueError("group_sequence_slots must be rank-1")
    if (
        group_source_positions.shape != (group_capacity,)
        or group_rope_positions.shape != (group_capacity,)
        or compressed_slots.shape != (group_capacity,)
    ):
        raise ValueError("continuation group metadata arrays must share one capacity")
    if group_capacity > caps.max_tokens:
        raise ValueError(
            f"continuation group capacity {group_capacity} exceeds max tokens "
            f"{caps.max_tokens}"
        )
    if state_sequence_ids.ndim != 1 or state_sequence_ids.numel() <= 0:
        raise ValueError("state_sequence_ids must be a non-empty rank-1 tensor")
    sequence_capacity = int(state_sequence_ids.shape[0])
    if sequence_offsets.shape != (sequence_capacity + 1,):
        raise ValueError(f"sequence_offsets must have shape [{sequence_capacity + 1}]")
    if sequence_start_positions.shape != (sequence_capacity,):
        raise ValueError(
            f"sequence_start_positions must have shape [{sequence_capacity}]"
        )
    if sequence_capacity > min(state_sequences, caps.max_tokens):
        raise ValueError(
            f"sequence capacity {sequence_capacity} exceeds state/token capacity "
            f"{min(state_sequences, caps.max_tokens)}"
        )


@triton.jit
def _update_pool_pack_main_kernel(
    projection,
    positions,
    sequence_ids,
    compressed_slots,
    cos_sin,
    kv_state,
    score_state,
    ape,
    norm,
    cache_fp8,
    cache_bf16,
    cache_u8,
    projection_stride_t,
    cos_sin_stride_pos,
    state_stride_seq,
    state_stride_row,
    eps,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    PROJECTED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(positions + token)
    sequence = tl.load(sequence_ids + token).to(tl.int64)
    lane = position % RATIO
    state_row = lane + RATIO if OVERLAP else lane
    state_base = sequence * state_stride_seq + state_row * state_stride_row

    pd = tl.arange(0, PROJECTED_BLOCK)
    pmask = pd < PROJECTED_WIDTH
    projected = tl.load(
        projection + token * projection_stride_t + PROJECTION_OFFSET + pd,
        mask=pmask,
        other=0.0,
    ).to(tl.float32)
    scores = tl.load(
        projection
        + token * projection_stride_t
        + PROJECTION_OFFSET
        + PROJECTED_WIDTH
        + pd,
        mask=pmask,
        other=0.0,
    ).to(tl.float32)
    positional = tl.load(ape + lane * PROJECTED_WIDTH + pd, mask=pmask, other=0.0)
    tl.store(kv_state + state_base + pd, projected, mask=pmask)
    tl.store(score_state + state_base + pd, scores + positional, mask=pmask)

    emits = lane + 1 == RATIO
    if emits:
        d = tl.arange(0, HEAD_DIM)
        row_max = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
        for row in range(RATIO):
            if OVERLAP:
                previous_base = sequence * state_stride_seq + row * state_stride_row
                current_base = (
                    sequence * state_stride_seq + (RATIO + row) * state_stride_row
                )
                previous_score = tl.load(score_state + previous_base + d)
                current_score = tl.load(score_state + current_base + HEAD_DIM + d)
                row_max = tl.maximum(row_max, previous_score)
                row_max = tl.maximum(row_max, current_score)
            else:
                row_base = sequence * state_stride_seq + row * state_stride_row
                row_max = tl.maximum(row_max, tl.load(score_state + row_base + d))

        numerator = tl.zeros((HEAD_DIM,), tl.float32)
        denominator = tl.zeros((HEAD_DIM,), tl.float32)
        for row in range(RATIO):
            if OVERLAP:
                previous_base = sequence * state_stride_seq + row * state_stride_row
                current_base = (
                    sequence * state_stride_seq + (RATIO + row) * state_stride_row
                )
                previous_score = tl.load(score_state + previous_base + d)
                current_score = tl.load(score_state + current_base + HEAD_DIM + d)
                previous_weight = tl.exp(previous_score - row_max)
                current_weight = tl.exp(current_score - row_max)
                numerator += previous_weight * tl.load(kv_state + previous_base + d)
                numerator += current_weight * tl.load(
                    kv_state + current_base + HEAD_DIM + d
                )
                denominator += previous_weight + current_weight
            else:
                row_base = sequence * state_stride_seq + row * state_stride_row
                row_score = tl.load(score_state + row_base + d)
                row_weight = tl.exp(row_score - row_max)
                numerator += row_weight * tl.load(kv_state + row_base + d)
                denominator += row_weight
        pooled = (numerator / denominator).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
        normalized = (
            (pooled * inv * tl.load(norm + d).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        rope_mask = d >= NOPE_DIM
        partner_d = NOPE_DIM + ((d - NOPE_DIM) ^ 1)
        partner_max = tl.where(
            rope_mask,
            tl.full((HEAD_DIM,), float("-inf"), tl.float32),
            tl.zeros((HEAD_DIM,), tl.float32),
        )
        for row in range(RATIO):
            if OVERLAP:
                previous_base = sequence * state_stride_seq + row * state_stride_row
                current_base = (
                    sequence * state_stride_seq + (RATIO + row) * state_stride_row
                )
                previous_score = tl.load(
                    score_state + previous_base + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                )
                current_score = tl.load(
                    score_state + current_base + HEAD_DIM + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                )
                partner_max = tl.maximum(partner_max, previous_score)
                partner_max = tl.maximum(partner_max, current_score)
            else:
                row_base = sequence * state_stride_seq + row * state_stride_row
                partner_max = tl.maximum(
                    partner_max,
                    tl.load(
                        score_state + row_base + partner_d,
                        mask=rope_mask,
                        other=float("-inf"),
                    ),
                )
        partner_num = tl.zeros((HEAD_DIM,), tl.float32)
        partner_den = tl.zeros((HEAD_DIM,), tl.float32)
        for row in range(RATIO):
            if OVERLAP:
                previous_base = sequence * state_stride_seq + row * state_stride_row
                current_base = (
                    sequence * state_stride_seq + (RATIO + row) * state_stride_row
                )
                previous_score = tl.load(
                    score_state + previous_base + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                )
                current_score = tl.load(
                    score_state + current_base + HEAD_DIM + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                )
                previous_weight = tl.exp(previous_score - partner_max)
                current_weight = tl.exp(current_score - partner_max)
                partner_num += previous_weight * tl.load(
                    kv_state + previous_base + partner_d,
                    mask=rope_mask,
                    other=0.0,
                )
                partner_num += current_weight * tl.load(
                    kv_state + current_base + HEAD_DIM + partner_d,
                    mask=rope_mask,
                    other=0.0,
                )
                partner_den += previous_weight + current_weight
            else:
                row_base = sequence * state_stride_seq + row * state_stride_row
                partner_score = tl.load(
                    score_state + row_base + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                )
                partner_weight = tl.exp(partner_score - partner_max)
                partner_num += partner_weight * tl.load(
                    kv_state + row_base + partner_d,
                    mask=rope_mask,
                    other=0.0,
                )
                partner_den += partner_weight
        partner_pooled = (
            (partner_num / tl.maximum(partner_den, 1.0)).to(tl.bfloat16).to(tl.float32)
        )
        partner_normalized = (
            (
                partner_pooled
                * inv
                * tl.load(norm + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        rope_position = position + 1 - RATIO
        pair = tl.maximum((d - NOPE_DIM) >> 1, 0)
        cs = cos_sin + rope_position * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            ((d - NOPE_DIM) & 1) == 0,
            normalized * cos_v - partner_normalized * sin_v,
            normalized * cos_v + partner_normalized * sin_v,
        )
        output = tl.where(rope_mask, rotated, normalized)

        slot = tl.load(compressed_slots + token).to(tl.int64)
        page = slot // PAGE_ROWS
        page_row = slot - page * PAGE_ROWS
        page_base = page * PAGE_BYTES
        data_base = page_base + page_row * PAYLOAD_BYTES
        scale_base = page_base + PAGE_ROWS * PAYLOAD_BYTES + page_row * SCALE_BYTES
        for group in range(NOPE_DIM // 64):
            group_mask = (d >= group * 64) & (d < (group + 1) * 64)
            max_abs = tl.maximum(
                tl.max(tl.where(group_mask, tl.abs(output), 0.0), axis=0),
                1.0e-4,
            )
            raw_scale = max_abs / FP8_MAX
            bits = raw_scale.to(tl.uint32, bitcast=True)
            mantissa = bits & 0x007FFFFF
            rounded_bits = (bits + 0x00800000) & 0x7F800000
            scale_bits = tl.where(mantissa != 0, rounded_bits, bits & 0x7F800000)
            scale = scale_bits.to(tl.float32, bitcast=True)
            quant = tl.maximum(tl.minimum(output / scale, FP8_MAX), -FP8_MAX)
            tl.store(
                cache_fp8 + data_base + d,
                quant.to(tl.float8e4nv),
                mask=group_mask,
            )
            tl.store(cache_u8 + scale_base + group, (scale_bits >> 23).to(tl.uint8))
        tl.store(
            cache_bf16 + data_base // 2 + d - NOPE_DIM // 2,
            output.to(tl.bfloat16),
            mask=rope_mask,
        )
        tl.store(cache_u8 + scale_base + 7, 0)

        if OVERLAP:
            for row in range(RATIO):
                previous_base = sequence * state_stride_seq + row * state_stride_row
                current_base = (
                    sequence * state_stride_seq + (RATIO + row) * state_stride_row
                )
                tl.store(
                    kv_state + previous_base + pd,
                    tl.load(kv_state + current_base + pd, mask=pmask, other=0.0),
                    mask=pmask,
                )
                tl.store(
                    score_state + previous_base + pd,
                    tl.load(score_state + current_base + pd, mask=pmask, other=0.0),
                    mask=pmask,
                )


@triton.jit
def _fwht_stage(values, d, WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr):
    shaped = tl.reshape(values, (HEAD_DIM // (2 * WIDTH), 2, WIDTH))
    partner = tl.reshape(tl.flip(shaped, 1), (HEAD_DIM,))
    return tl.where((d & WIDTH) == 0, values + partner, partner - values)


@triton.jit
def _update_pool_pack_index_kernel(
    projection,
    positions,
    sequence_ids,
    compressed_slots,
    cos_sin,
    kv_state,
    score_state,
    ape,
    norm,
    cache_fp8,
    cache_fp32,
    projection_stride_t,
    cos_sin_stride_pos,
    state_stride_seq,
    state_stride_row,
    eps,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token = tl.program_id(0)
    position = tl.load(positions + token)
    sequence = tl.load(sequence_ids + token).to(tl.int64)
    lane = position % RATIO
    state_row = RATIO + lane
    state_base = sequence * state_stride_seq + state_row * state_stride_row
    pd = tl.arange(0, PROJECTED_WIDTH)
    projected = tl.load(
        projection + token * projection_stride_t + PROJECTION_OFFSET + pd
    ).to(tl.float32)
    scores = tl.load(
        projection
        + token * projection_stride_t
        + PROJECTION_OFFSET
        + PROJECTED_WIDTH
        + pd
    ).to(tl.float32)
    positional = tl.load(ape + lane * PROJECTED_WIDTH + pd)
    tl.store(kv_state + state_base + pd, projected)
    tl.store(score_state + state_base + pd, scores + positional)

    emits = lane + 1 == RATIO
    if emits:
        d = tl.arange(0, HEAD_DIM)
        row_max = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
        for row in range(RATIO):
            previous_base = sequence * state_stride_seq + row * state_stride_row
            current_base = (
                sequence * state_stride_seq + (RATIO + row) * state_stride_row
            )
            row_max = tl.maximum(row_max, tl.load(score_state + previous_base + d))
            row_max = tl.maximum(
                row_max, tl.load(score_state + current_base + HEAD_DIM + d)
            )
        numerator = tl.zeros((HEAD_DIM,), tl.float32)
        denominator = tl.zeros((HEAD_DIM,), tl.float32)
        for row in range(RATIO):
            previous_base = sequence * state_stride_seq + row * state_stride_row
            current_base = (
                sequence * state_stride_seq + (RATIO + row) * state_stride_row
            )
            previous_score = tl.load(score_state + previous_base + d)
            current_score = tl.load(score_state + current_base + HEAD_DIM + d)
            previous_weight = tl.exp(previous_score - row_max)
            current_weight = tl.exp(current_score - row_max)
            numerator += previous_weight * tl.load(kv_state + previous_base + d)
            numerator += current_weight * tl.load(
                kv_state + current_base + HEAD_DIM + d
            )
            denominator += previous_weight + current_weight
        pooled = (numerator / denominator).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
        normalized = (
            (pooled * inv * tl.load(norm + d).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        rope_mask = d >= NOPE_DIM
        partner_d = NOPE_DIM + ((d - NOPE_DIM) ^ 1)
        partner_max = tl.where(
            rope_mask,
            tl.full((HEAD_DIM,), float("-inf"), tl.float32),
            tl.zeros((HEAD_DIM,), tl.float32),
        )
        for row in range(RATIO):
            previous_base = sequence * state_stride_seq + row * state_stride_row
            current_base = (
                sequence * state_stride_seq + (RATIO + row) * state_stride_row
            )
            partner_max = tl.maximum(
                partner_max,
                tl.load(
                    score_state + previous_base + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                ),
            )
            partner_max = tl.maximum(
                partner_max,
                tl.load(
                    score_state + current_base + HEAD_DIM + partner_d,
                    mask=rope_mask,
                    other=float("-inf"),
                ),
            )
        partner_num = tl.zeros((HEAD_DIM,), tl.float32)
        partner_den = tl.zeros((HEAD_DIM,), tl.float32)
        for row in range(RATIO):
            previous_base = sequence * state_stride_seq + row * state_stride_row
            current_base = (
                sequence * state_stride_seq + (RATIO + row) * state_stride_row
            )
            previous_score = tl.load(
                score_state + previous_base + partner_d,
                mask=rope_mask,
                other=float("-inf"),
            )
            current_score = tl.load(
                score_state + current_base + HEAD_DIM + partner_d,
                mask=rope_mask,
                other=float("-inf"),
            )
            previous_weight = tl.exp(previous_score - partner_max)
            current_weight = tl.exp(current_score - partner_max)
            partner_num += previous_weight * tl.load(
                kv_state + previous_base + partner_d, mask=rope_mask, other=0.0
            )
            partner_num += current_weight * tl.load(
                kv_state + current_base + HEAD_DIM + partner_d,
                mask=rope_mask,
                other=0.0,
            )
            partner_den += previous_weight + current_weight
        partner_pooled = (
            (partner_num / tl.maximum(partner_den, 1.0)).to(tl.bfloat16).to(tl.float32)
        )
        partner_normalized = (
            (
                partner_pooled
                * inv
                * tl.load(norm + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rope_position = position + 1 - RATIO
        pair = tl.maximum((d - NOPE_DIM) >> 1, 0)
        cs = cos_sin + rope_position * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            ((d - NOPE_DIM) & 1) == 0,
            normalized * cos_v - partner_normalized * sin_v,
            normalized * cos_v + partner_normalized * sin_v,
        )
        hadamard = tl.where(rope_mask, rotated, normalized)

        # Sylvester FWHT in register space. Each stage pairs lane d with
        # d^width; the second half computes partner-current to preserve the
        # canonical fast_hadamard_transform sign order.
        hadamard = _fwht_stage(hadamard, d, WIDTH=1, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=2, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=4, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=8, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=16, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=32, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=64, HEAD_DIM=HEAD_DIM)
        hadamard = (hadamard * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)

        # Checkpoint QAT applies E2M1 block-32 quantize/dequantize in place
        # before the runtime FP8 index-cache encoding.
        blocks = tl.reshape(tl.abs(hadamard), (HEAD_DIM // 32, 32))
        block_max = tl.max(blocks, axis=1)
        raw_fp4_scale = tl.maximum(block_max, 6.0 * 1.1754943508222875e-38) / 6.0
        scale_bits = raw_fp4_scale.to(tl.uint32, bitcast=True)
        mantissa = scale_bits & 0x007FFFFF
        rounded_bits = (scale_bits + 0x00800000) & 0x7F800000
        fp4_scale = tl.where(mantissa != 0, rounded_bits, scale_bits & 0x7F800000).to(
            tl.float32, bitcast=True
        )
        fp4_scale = tl.reshape(
            tl.broadcast_to(tl.expand_dims(fp4_scale, 1), (HEAD_DIM // 32, 32)),
            (HEAD_DIM,),
        )
        magnitude = tl.minimum(tl.abs(hadamard) / fp4_scale, 6.0)
        fp4 = tl.where(
            magnitude < 0.25,
            0.0,
            tl.where(
                magnitude < 0.75,
                0.5,
                tl.where(
                    magnitude < 1.25,
                    1.0,
                    tl.where(
                        magnitude < 1.75,
                        1.5,
                        tl.where(
                            magnitude < 2.5,
                            2.0,
                            tl.where(
                                magnitude < 3.5,
                                3.0,
                                tl.where(magnitude < 5.0, 4.0, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        fp4 = tl.where(hadamard < 0.0, -fp4, fp4) * fp4_scale
        fp4 = fp4.to(tl.bfloat16).to(tl.float32)

        cache_scale = tl.max(tl.abs(fp4), axis=0) / FP8_MAX
        cache_scale = tl.where(cache_scale > 0.0, cache_scale, 1.0)
        quant = tl.maximum(tl.minimum(fp4 / cache_scale, FP8_MAX), -FP8_MAX)
        slot = tl.load(compressed_slots + token).to(tl.int64)
        page = slot // PAGE_ROWS
        page_row = slot - page * PAGE_ROWS
        page_base = page * PAGE_BYTES
        tl.store(
            cache_fp8 + page_base + page_row * HEAD_DIM + d, quant.to(tl.float8e4nv)
        )
        tl.store(
            cache_fp32 + (page_base + PAGE_ROWS * HEAD_DIM) // 4 + page_row,
            cache_scale,
        )

        for row in range(RATIO):
            previous_base = sequence * state_stride_seq + row * state_stride_row
            current_base = (
                sequence * state_stride_seq + (RATIO + row) * state_stride_row
            )
            tl.store(
                kv_state + previous_base + pd, tl.load(kv_state + current_base + pd)
            )
            tl.store(
                score_state + previous_base + pd,
                tl.load(score_state + current_base + pd),
            )


@triton.jit
def _prefill_pool_dimension(
    projection,
    ape,
    source_start,
    rope_position,
    d,
    valid_mask,
    projection_stride_t,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    current_d = HEAD_DIM + d if OVERLAP else d
    row_max = tl.where(
        valid_mask,
        tl.full((HEAD_DIM,), float("-inf"), tl.float32),
        tl.zeros((HEAD_DIM,), tl.float32),
    )
    for row in range(RATIO):
        current_row = source_start + row
        current_score = tl.load(
            projection
            + current_row * projection_stride_t
            + PROJECTION_OFFSET
            + PROJECTED_WIDTH
            + current_d,
            mask=valid_mask,
            other=float("-inf"),
        ).to(tl.float32)
        current_score += tl.load(
            ape + row * PROJECTED_WIDTH + current_d,
            mask=valid_mask,
            other=0.0,
        )
        row_max = tl.maximum(row_max, current_score)
        if OVERLAP:
            previous_mask = valid_mask & (rope_position > 0)
            previous_row = source_start - RATIO + row
            previous_score = tl.load(
                projection
                + previous_row * projection_stride_t
                + PROJECTION_OFFSET
                + PROJECTED_WIDTH
                + d,
                mask=previous_mask,
                other=float("-inf"),
            ).to(tl.float32)
            previous_score += tl.load(
                ape + row * PROJECTED_WIDTH + d,
                mask=previous_mask,
                other=0.0,
            )
            row_max = tl.maximum(row_max, previous_score)

    numerator = tl.zeros((HEAD_DIM,), tl.float32)
    denominator = tl.zeros((HEAD_DIM,), tl.float32)
    for row in range(RATIO):
        current_row = source_start + row
        current_score = tl.load(
            projection
            + current_row * projection_stride_t
            + PROJECTION_OFFSET
            + PROJECTED_WIDTH
            + current_d,
            mask=valid_mask,
            other=float("-inf"),
        ).to(tl.float32)
        current_score += tl.load(
            ape + row * PROJECTED_WIDTH + current_d,
            mask=valid_mask,
            other=0.0,
        )
        current_weight = tl.exp(current_score - row_max)
        current_value = tl.load(
            projection
            + current_row * projection_stride_t
            + PROJECTION_OFFSET
            + current_d,
            mask=valid_mask,
            other=0.0,
        ).to(tl.float32)
        numerator += current_weight * current_value
        denominator += current_weight
        if OVERLAP:
            previous_mask = valid_mask & (rope_position > 0)
            previous_row = source_start - RATIO + row
            previous_score = tl.load(
                projection
                + previous_row * projection_stride_t
                + PROJECTION_OFFSET
                + PROJECTED_WIDTH
                + d,
                mask=previous_mask,
                other=float("-inf"),
            ).to(tl.float32)
            previous_score += tl.load(
                ape + row * PROJECTED_WIDTH + d,
                mask=previous_mask,
                other=0.0,
            )
            previous_weight = tl.exp(previous_score - row_max)
            previous_value = tl.load(
                projection + previous_row * projection_stride_t + PROJECTION_OFFSET + d,
                mask=previous_mask,
                other=0.0,
            ).to(tl.float32)
            numerator += previous_weight * previous_value
            denominator += previous_weight
    return numerator / tl.maximum(denominator, 1.0)


@triton.jit
def _prefill_pool_pack_main_kernel(
    projection,
    active_groups,
    group_source_starts,
    group_rope_positions,
    compressed_slots,
    cos_sin,
    ape,
    norm,
    cache_fp8,
    cache_bf16,
    cache_u8,
    projection_stride_t,
    cos_sin_stride_pos,
    eps,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    group = tl.program_id(0)
    if group < tl.load(active_groups):
        source_start = tl.load(group_source_starts + group)
        rope_position = tl.load(group_rope_positions + group)
        d = tl.arange(0, HEAD_DIM)
        valid = d < HEAD_DIM
        pooled = (
            _prefill_pool_dimension(
                projection,
                ape,
                source_start,
                rope_position,
                d,
                valid,
                projection_stride_t,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        inv = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
        normalized = (
            (pooled * inv * tl.load(norm + d).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        rope_mask = d >= NOPE_DIM
        partner_d = NOPE_DIM + ((d - NOPE_DIM) ^ 1)
        partner_pooled = (
            _prefill_pool_dimension(
                projection,
                ape,
                source_start,
                rope_position,
                partner_d,
                rope_mask,
                projection_stride_t,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        partner_normalized = (
            (
                partner_pooled
                * inv
                * tl.load(norm + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        pair = tl.maximum((d - NOPE_DIM) >> 1, 0)
        cs = cos_sin + rope_position * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            ((d - NOPE_DIM) & 1) == 0,
            normalized * cos_v - partner_normalized * sin_v,
            normalized * cos_v + partner_normalized * sin_v,
        )
        output = tl.where(rope_mask, rotated, normalized)

        slot = tl.load(compressed_slots + group).to(tl.int64)
        page = slot // PAGE_ROWS
        page_row = slot - page * PAGE_ROWS
        page_base = page * PAGE_BYTES
        data_base = page_base + page_row * PAYLOAD_BYTES
        scale_base = page_base + PAGE_ROWS * PAYLOAD_BYTES + page_row * SCALE_BYTES
        for scale_group in range(NOPE_DIM // 64):
            scale_mask = (d >= scale_group * 64) & (d < (scale_group + 1) * 64)
            max_abs = tl.maximum(
                tl.max(tl.where(scale_mask, tl.abs(output), 0.0), axis=0),
                1.0e-4,
            )
            raw_scale = max_abs / FP8_MAX
            bits = raw_scale.to(tl.uint32, bitcast=True)
            mantissa = bits & 0x007FFFFF
            rounded_bits = (bits + 0x00800000) & 0x7F800000
            scale_bits = tl.where(mantissa != 0, rounded_bits, bits & 0x7F800000)
            scale = scale_bits.to(tl.float32, bitcast=True)
            quant = tl.maximum(tl.minimum(output / scale, FP8_MAX), -FP8_MAX)
            tl.store(
                cache_fp8 + data_base + d,
                quant.to(tl.float8e4nv),
                mask=scale_mask,
            )
            tl.store(
                cache_u8 + scale_base + scale_group,
                (scale_bits >> 23).to(tl.uint8),
            )
        tl.store(
            cache_bf16 + data_base // 2 + d - NOPE_DIM // 2,
            output.to(tl.bfloat16),
            mask=rope_mask,
        )
        tl.store(cache_u8 + scale_base + 7, 0)


@triton.jit
def _prefill_pool_pack_index_kernel(
    projection,
    active_groups,
    group_source_starts,
    group_rope_positions,
    compressed_slots,
    cos_sin,
    ape,
    norm,
    cache_fp8,
    cache_fp32,
    projection_stride_t,
    cos_sin_stride_pos,
    eps,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    group = tl.program_id(0)
    if group < tl.load(active_groups):
        source_start = tl.load(group_source_starts + group)
        rope_position = tl.load(group_rope_positions + group)
        d = tl.arange(0, HEAD_DIM)
        valid = d < HEAD_DIM
        pooled = (
            _prefill_pool_dimension(
                projection,
                ape,
                source_start,
                rope_position,
                d,
                valid,
                projection_stride_t,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=True,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        inv = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
        normalized = (
            (pooled * inv * tl.load(norm + d).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rope_mask = d >= NOPE_DIM
        partner_d = NOPE_DIM + ((d - NOPE_DIM) ^ 1)
        partner_pooled = (
            _prefill_pool_dimension(
                projection,
                ape,
                source_start,
                rope_position,
                partner_d,
                rope_mask,
                projection_stride_t,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=True,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        partner_normalized = (
            (
                partner_pooled
                * inv
                * tl.load(norm + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        pair = tl.maximum((d - NOPE_DIM) >> 1, 0)
        cs = cos_sin + rope_position * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            ((d - NOPE_DIM) & 1) == 0,
            normalized * cos_v - partner_normalized * sin_v,
            normalized * cos_v + partner_normalized * sin_v,
        )
        hadamard = tl.where(rope_mask, rotated, normalized)
        hadamard = _fwht_stage(hadamard, d, WIDTH=1, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=2, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=4, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=8, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=16, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=32, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=64, HEAD_DIM=HEAD_DIM)
        hadamard = (hadamard * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)

        blocks = tl.reshape(tl.abs(hadamard), (HEAD_DIM // 32, 32))
        block_max = tl.max(blocks, axis=1)
        raw_fp4_scale = tl.maximum(block_max, 6.0 * 1.1754943508222875e-38) / 6.0
        scale_bits = raw_fp4_scale.to(tl.uint32, bitcast=True)
        mantissa = scale_bits & 0x007FFFFF
        rounded_bits = (scale_bits + 0x00800000) & 0x7F800000
        fp4_scale = tl.where(mantissa != 0, rounded_bits, scale_bits & 0x7F800000).to(
            tl.float32, bitcast=True
        )
        fp4_scale = tl.reshape(
            tl.broadcast_to(tl.expand_dims(fp4_scale, 1), (HEAD_DIM // 32, 32)),
            (HEAD_DIM,),
        )
        magnitude = tl.minimum(tl.abs(hadamard) / fp4_scale, 6.0)
        fp4 = tl.where(
            magnitude < 0.25,
            0.0,
            tl.where(
                magnitude < 0.75,
                0.5,
                tl.where(
                    magnitude < 1.25,
                    1.0,
                    tl.where(
                        magnitude < 1.75,
                        1.5,
                        tl.where(
                            magnitude < 2.5,
                            2.0,
                            tl.where(
                                magnitude < 3.5,
                                3.0,
                                tl.where(magnitude < 5.0, 4.0, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        fp4 = tl.where(hadamard < 0.0, -fp4, fp4) * fp4_scale
        fp4 = fp4.to(tl.bfloat16).to(tl.float32)
        cache_scale = tl.max(tl.abs(fp4), axis=0) / FP8_MAX
        cache_scale = tl.where(cache_scale > 0.0, cache_scale, 1.0)
        quant = tl.maximum(tl.minimum(fp4 / cache_scale, FP8_MAX), -FP8_MAX)
        slot = tl.load(compressed_slots + group).to(tl.int64)
        page = slot // PAGE_ROWS
        page_row = slot - page * PAGE_ROWS
        page_base = page * PAGE_BYTES
        tl.store(
            cache_fp8 + page_base + page_row * HEAD_DIM + d,
            quant.to(tl.float8e4nv),
        )
        tl.store(
            cache_fp32 + (page_base + PAGE_ROWS * HEAD_DIM) // 4 + page_row,
            cache_scale,
        )


@triton.jit
def _prefill_finalize_state_kernel(
    projection,
    active_sequences,
    sequence_offsets,
    state_sequence_ids,
    kv_state,
    score_state,
    ape,
    projection_stride_t,
    state_stride_seq,
    state_stride_row,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    PROJECTED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    sequence_slot = tl.program_id(0)
    state_row = tl.program_id(1)
    if sequence_slot < tl.load(active_sequences):
        state_sequence = tl.load(state_sequence_ids + sequence_slot).to(tl.int64)
        source_start = tl.load(sequence_offsets + sequence_slot)
        source_end = tl.load(sequence_offsets + sequence_slot + 1)
        source_tokens = source_end - source_start
        cutoff = (source_tokens // RATIO) * RATIO
        remainder = source_tokens - cutoff
        if OVERLAP:
            previous = state_row < RATIO
            lane = tl.where(previous, state_row, state_row - RATIO)
            fill = tl.where(previous, cutoff >= RATIO, lane < remainder)
            source_row = tl.where(
                previous,
                source_start + cutoff - RATIO + lane,
                source_start + cutoff + lane,
            )
        else:
            lane = state_row
            fill = lane < remainder
            source_row = source_start + cutoff + lane
        pd = tl.arange(0, PROJECTED_BLOCK)
        pmask = pd < PROJECTED_WIDTH
        state_base = state_sequence * state_stride_seq + state_row * state_stride_row
        if fill:
            values = tl.load(
                projection + source_row * projection_stride_t + PROJECTION_OFFSET + pd,
                mask=pmask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.load(
                projection
                + source_row * projection_stride_t
                + PROJECTION_OFFSET
                + PROJECTED_WIDTH
                + pd,
                mask=pmask,
                other=0.0,
            ).to(tl.float32)
            scores += tl.load(
                ape + lane * PROJECTED_WIDTH + pd,
                mask=pmask,
                other=0.0,
            )
        else:
            values = tl.zeros((PROJECTED_BLOCK,), tl.float32)
            scores = tl.full((PROJECTED_BLOCK,), float("-inf"), tl.float32)
        tl.store(kv_state + state_base + pd, values, mask=pmask)
        tl.store(score_state + state_base + pd, scores, mask=pmask)


@triton.jit
def _continuation_load_dimension(
    projection,
    state,
    ape,
    sequence_chunk_offset,
    chunk_logical_start,
    state_sequence,
    source_position,
    d,
    valid_mask,
    projection_stride_t,
    state_stride_seq,
    state_stride_row,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    SCORE: tl.constexpr,
):
    source_valid = source_position >= 0
    from_projection = source_position >= chunk_logical_start
    projection_row = sequence_chunk_offset + source_position - chunk_logical_start
    projection_d = d + PROJECTED_WIDTH if SCORE else d
    masked_value = float("-inf") if SCORE else 0.0
    projected = tl.load(
        projection
        + projection_row * projection_stride_t
        + PROJECTION_OFFSET
        + projection_d,
        mask=valid_mask & source_valid & from_projection,
        other=masked_value,
    ).to(tl.float32)
    lane = source_position % RATIO
    if SCORE:
        projected += tl.load(
            ape + lane * PROJECTED_WIDTH + d,
            mask=valid_mask & source_valid & from_projection,
            other=0.0,
        )

    initial_group_start = chunk_logical_start - chunk_logical_start % RATIO
    if OVERLAP:
        state_row = tl.where(
            source_position >= initial_group_start,
            RATIO + lane,
            lane,
        )
    else:
        state_row = lane
    state_base = state_sequence * state_stride_seq + state_row * state_stride_row
    carried = tl.load(
        state + state_base + d,
        mask=valid_mask & source_valid & ~from_projection,
        other=masked_value,
    ).to(tl.float32)
    return tl.where(from_projection, projected, carried)


@triton.jit
def _continuation_pool_dimension(
    projection,
    kv_state,
    score_state,
    ape,
    sequence_chunk_offset,
    chunk_logical_start,
    state_sequence,
    group_start,
    d,
    valid_mask,
    projection_stride_t,
    state_stride_seq,
    state_stride_row,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    current_d = HEAD_DIM + d if OVERLAP else d
    row_max = tl.where(
        valid_mask,
        tl.full((HEAD_DIM,), float("-inf"), tl.float32),
        tl.zeros((HEAD_DIM,), tl.float32),
    )
    for row in range(RATIO):
        current_position = group_start + row
        current_score = _continuation_load_dimension(
            projection,
            score_state,
            ape,
            sequence_chunk_offset,
            chunk_logical_start,
            state_sequence,
            current_position,
            current_d,
            valid_mask,
            projection_stride_t,
            state_stride_seq,
            state_stride_row,
            PROJECTION_OFFSET=PROJECTION_OFFSET,
            PROJECTED_WIDTH=PROJECTED_WIDTH,
            RATIO=RATIO,
            OVERLAP=OVERLAP,
            SCORE=True,
        )
        row_max = tl.maximum(row_max, current_score)
        if OVERLAP:
            previous_mask = valid_mask & (group_start > 0)
            previous_position = group_start - RATIO + row
            previous_score = _continuation_load_dimension(
                projection,
                score_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                previous_position,
                d,
                previous_mask,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                SCORE=True,
            )
            row_max = tl.maximum(row_max, previous_score)

    numerator = tl.zeros((HEAD_DIM,), tl.float32)
    denominator = tl.zeros((HEAD_DIM,), tl.float32)
    for row in range(RATIO):
        current_position = group_start + row
        current_score = _continuation_load_dimension(
            projection,
            score_state,
            ape,
            sequence_chunk_offset,
            chunk_logical_start,
            state_sequence,
            current_position,
            current_d,
            valid_mask,
            projection_stride_t,
            state_stride_seq,
            state_stride_row,
            PROJECTION_OFFSET=PROJECTION_OFFSET,
            PROJECTED_WIDTH=PROJECTED_WIDTH,
            RATIO=RATIO,
            OVERLAP=OVERLAP,
            SCORE=True,
        )
        current_value = _continuation_load_dimension(
            projection,
            kv_state,
            ape,
            sequence_chunk_offset,
            chunk_logical_start,
            state_sequence,
            current_position,
            current_d,
            valid_mask,
            projection_stride_t,
            state_stride_seq,
            state_stride_row,
            PROJECTION_OFFSET=PROJECTION_OFFSET,
            PROJECTED_WIDTH=PROJECTED_WIDTH,
            RATIO=RATIO,
            OVERLAP=OVERLAP,
            SCORE=False,
        )
        current_weight = tl.exp(current_score - row_max)
        numerator += current_weight * current_value
        denominator += current_weight
        if OVERLAP:
            previous_mask = valid_mask & (group_start > 0)
            previous_position = group_start - RATIO + row
            previous_score = _continuation_load_dimension(
                projection,
                score_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                previous_position,
                d,
                previous_mask,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                SCORE=True,
            )
            previous_value = _continuation_load_dimension(
                projection,
                kv_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                previous_position,
                d,
                previous_mask,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                SCORE=False,
            )
            previous_weight = tl.exp(previous_score - row_max)
            numerator += previous_weight * previous_value
            denominator += previous_weight
    return numerator / tl.maximum(denominator, 1.0)


@triton.jit
def _continuation_pool_pack_main_kernel(
    projection,
    active_groups,
    group_sequence_slots,
    group_source_positions,
    group_rope_positions,
    compressed_slots,
    sequence_offsets,
    sequence_start_positions,
    state_sequence_ids,
    cos_sin,
    kv_state,
    score_state,
    ape,
    norm,
    cache_fp8,
    cache_bf16,
    cache_u8,
    projection_stride_t,
    cos_sin_stride_pos,
    state_stride_seq,
    state_stride_row,
    eps,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    group = tl.program_id(0)
    if group < tl.load(active_groups):
        sequence_slot = tl.load(group_sequence_slots + group)
        sequence_chunk_offset = tl.load(sequence_offsets + sequence_slot)
        chunk_logical_start = tl.load(sequence_start_positions + sequence_slot)
        state_sequence = tl.load(state_sequence_ids + sequence_slot).to(tl.int64)
        group_start = tl.load(group_source_positions + group)
        rope_position = tl.load(group_rope_positions + group)
        d = tl.arange(0, HEAD_DIM)
        valid = d < HEAD_DIM
        pooled = (
            _continuation_pool_dimension(
                projection,
                kv_state,
                score_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                group_start,
                d,
                valid,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        inv = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
        normalized = (
            (pooled * inv * tl.load(norm + d).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        rope_mask = d >= NOPE_DIM
        partner_d = NOPE_DIM + ((d - NOPE_DIM) ^ 1)
        partner_pooled = (
            _continuation_pool_dimension(
                projection,
                kv_state,
                score_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                group_start,
                partner_d,
                rope_mask,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=OVERLAP,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        partner_normalized = (
            (
                partner_pooled
                * inv
                * tl.load(norm + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        pair = tl.maximum((d - NOPE_DIM) >> 1, 0)
        cs = cos_sin + rope_position * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            ((d - NOPE_DIM) & 1) == 0,
            normalized * cos_v - partner_normalized * sin_v,
            normalized * cos_v + partner_normalized * sin_v,
        )
        output = tl.where(rope_mask, rotated, normalized)

        slot = tl.load(compressed_slots + group).to(tl.int64)
        page = slot // PAGE_ROWS
        page_row = slot - page * PAGE_ROWS
        page_base = page * PAGE_BYTES
        data_base = page_base + page_row * PAYLOAD_BYTES
        scale_base = page_base + PAGE_ROWS * PAYLOAD_BYTES + page_row * SCALE_BYTES
        for scale_group in range(NOPE_DIM // 64):
            scale_mask = (d >= scale_group * 64) & (d < (scale_group + 1) * 64)
            max_abs = tl.maximum(
                tl.max(tl.where(scale_mask, tl.abs(output), 0.0), axis=0),
                1.0e-4,
            )
            raw_scale = max_abs / FP8_MAX
            bits = raw_scale.to(tl.uint32, bitcast=True)
            mantissa = bits & 0x007FFFFF
            rounded_bits = (bits + 0x00800000) & 0x7F800000
            scale_bits = tl.where(mantissa != 0, rounded_bits, bits & 0x7F800000)
            scale = scale_bits.to(tl.float32, bitcast=True)
            quant = tl.maximum(tl.minimum(output / scale, FP8_MAX), -FP8_MAX)
            tl.store(
                cache_fp8 + data_base + d,
                quant.to(tl.float8e4nv),
                mask=scale_mask,
            )
            tl.store(
                cache_u8 + scale_base + scale_group,
                (scale_bits >> 23).to(tl.uint8),
            )
        tl.store(
            cache_bf16 + data_base // 2 + d - NOPE_DIM // 2,
            output.to(tl.bfloat16),
            mask=rope_mask,
        )
        tl.store(cache_u8 + scale_base + 7, 0)


@triton.jit
def _continuation_pool_pack_index_kernel(
    projection,
    active_groups,
    group_sequence_slots,
    group_source_positions,
    group_rope_positions,
    compressed_slots,
    sequence_offsets,
    sequence_start_positions,
    state_sequence_ids,
    cos_sin,
    kv_state,
    score_state,
    ape,
    norm,
    cache_fp8,
    cache_fp32,
    projection_stride_t,
    cos_sin_stride_pos,
    state_stride_seq,
    state_stride_row,
    eps,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    group = tl.program_id(0)
    if group < tl.load(active_groups):
        sequence_slot = tl.load(group_sequence_slots + group)
        sequence_chunk_offset = tl.load(sequence_offsets + sequence_slot)
        chunk_logical_start = tl.load(sequence_start_positions + sequence_slot)
        state_sequence = tl.load(state_sequence_ids + sequence_slot).to(tl.int64)
        group_start = tl.load(group_source_positions + group)
        rope_position = tl.load(group_rope_positions + group)
        d = tl.arange(0, HEAD_DIM)
        valid = d < HEAD_DIM
        pooled = (
            _continuation_pool_dimension(
                projection,
                kv_state,
                score_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                group_start,
                d,
                valid,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=True,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        inv = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
        normalized = (
            (pooled * inv * tl.load(norm + d).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rope_mask = d >= NOPE_DIM
        partner_d = NOPE_DIM + ((d - NOPE_DIM) ^ 1)
        partner_pooled = (
            _continuation_pool_dimension(
                projection,
                kv_state,
                score_state,
                ape,
                sequence_chunk_offset,
                chunk_logical_start,
                state_sequence,
                group_start,
                partner_d,
                rope_mask,
                projection_stride_t,
                state_stride_seq,
                state_stride_row,
                PROJECTION_OFFSET=PROJECTION_OFFSET,
                PROJECTED_WIDTH=PROJECTED_WIDTH,
                RATIO=RATIO,
                OVERLAP=True,
                HEAD_DIM=HEAD_DIM,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        partner_normalized = (
            (
                partner_pooled
                * inv
                * tl.load(norm + partner_d, mask=rope_mask, other=0.0).to(tl.float32)
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        pair = tl.maximum((d - NOPE_DIM) >> 1, 0)
        cs = cos_sin + rope_position * cos_sin_stride_pos
        cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
        sin_v = tl.load(cs + ROPE_DIM // 2 + pair, mask=rope_mask, other=0.0)
        rotated = tl.where(
            ((d - NOPE_DIM) & 1) == 0,
            normalized * cos_v - partner_normalized * sin_v,
            normalized * cos_v + partner_normalized * sin_v,
        )
        hadamard = tl.where(rope_mask, rotated, normalized)
        hadamard = _fwht_stage(hadamard, d, WIDTH=1, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=2, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=4, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=8, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=16, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=32, HEAD_DIM=HEAD_DIM)
        hadamard = _fwht_stage(hadamard, d, WIDTH=64, HEAD_DIM=HEAD_DIM)
        hadamard = (hadamard * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)

        blocks = tl.reshape(tl.abs(hadamard), (HEAD_DIM // 32, 32))
        block_max = tl.max(blocks, axis=1)
        raw_fp4_scale = tl.maximum(block_max, 6.0 * 1.1754943508222875e-38) / 6.0
        scale_bits = raw_fp4_scale.to(tl.uint32, bitcast=True)
        mantissa = scale_bits & 0x007FFFFF
        rounded_bits = (scale_bits + 0x00800000) & 0x7F800000
        fp4_scale = tl.where(mantissa != 0, rounded_bits, scale_bits & 0x7F800000).to(
            tl.float32, bitcast=True
        )
        fp4_scale = tl.reshape(
            tl.broadcast_to(tl.expand_dims(fp4_scale, 1), (HEAD_DIM // 32, 32)),
            (HEAD_DIM,),
        )
        magnitude = tl.minimum(tl.abs(hadamard) / fp4_scale, 6.0)
        fp4 = tl.where(
            magnitude < 0.25,
            0.0,
            tl.where(
                magnitude < 0.75,
                0.5,
                tl.where(
                    magnitude < 1.25,
                    1.0,
                    tl.where(
                        magnitude < 1.75,
                        1.5,
                        tl.where(
                            magnitude < 2.5,
                            2.0,
                            tl.where(
                                magnitude < 3.5,
                                3.0,
                                tl.where(magnitude < 5.0, 4.0, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        fp4 = tl.where(hadamard < 0.0, -fp4, fp4) * fp4_scale
        fp4 = fp4.to(tl.bfloat16).to(tl.float32)
        cache_scale = tl.max(tl.abs(fp4), axis=0) / FP8_MAX
        cache_scale = tl.where(cache_scale > 0.0, cache_scale, 1.0)
        quant = tl.maximum(tl.minimum(fp4 / cache_scale, FP8_MAX), -FP8_MAX)
        slot = tl.load(compressed_slots + group).to(tl.int64)
        page = slot // PAGE_ROWS
        page_row = slot - page * PAGE_ROWS
        page_base = page * PAGE_BYTES
        tl.store(
            cache_fp8 + page_base + page_row * HEAD_DIM + d,
            quant.to(tl.float8e4nv),
        )
        tl.store(
            cache_fp32 + (page_base + PAGE_ROWS * HEAD_DIM) // 4 + page_row,
            cache_scale,
        )


@triton.jit
def _continuation_finalize_state_kernel(
    projection,
    active_sequences,
    sequence_offsets,
    sequence_start_positions,
    state_sequence_ids,
    kv_state,
    score_state,
    ape,
    projection_stride_t,
    state_stride_seq,
    state_stride_row,
    PROJECTION_OFFSET: tl.constexpr,
    PROJECTED_WIDTH: tl.constexpr,
    PROJECTED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    STATE_ROW_START: tl.constexpr,
):
    sequence_slot = tl.program_id(0)
    state_row = STATE_ROW_START + tl.program_id(1)
    if sequence_slot < tl.load(active_sequences):
        sequence_chunk_offset = tl.load(sequence_offsets + sequence_slot)
        sequence_chunk_end = tl.load(sequence_offsets + sequence_slot + 1)
        chunk_logical_start = tl.load(sequence_start_positions + sequence_slot)
        state_sequence = tl.load(state_sequence_ids + sequence_slot).to(tl.int64)
        logical_end = chunk_logical_start + sequence_chunk_end - sequence_chunk_offset
        cutoff = logical_end // RATIO * RATIO
        remainder = logical_end - cutoff
        if OVERLAP:
            previous = state_row < RATIO
            lane = tl.where(previous, state_row, state_row - RATIO)
            fill = tl.where(previous, cutoff >= RATIO, lane < remainder)
            source_position = tl.where(
                previous,
                cutoff - RATIO + lane,
                cutoff + lane,
            )
        else:
            lane = state_row
            fill = lane < remainder
            source_position = cutoff + lane
        pd = tl.arange(0, PROJECTED_BLOCK)
        pmask = pd < PROJECTED_WIDTH
        values = _continuation_load_dimension(
            projection,
            kv_state,
            ape,
            sequence_chunk_offset,
            chunk_logical_start,
            state_sequence,
            source_position,
            pd,
            pmask & fill,
            projection_stride_t,
            state_stride_seq,
            state_stride_row,
            PROJECTION_OFFSET=PROJECTION_OFFSET,
            PROJECTED_WIDTH=PROJECTED_WIDTH,
            RATIO=RATIO,
            OVERLAP=OVERLAP,
            SCORE=False,
        )
        scores = _continuation_load_dimension(
            projection,
            score_state,
            ape,
            sequence_chunk_offset,
            chunk_logical_start,
            state_sequence,
            source_position,
            pd,
            pmask & fill,
            projection_stride_t,
            state_stride_seq,
            state_stride_row,
            PROJECTION_OFFSET=PROJECTION_OFFSET,
            PROJECTED_WIDTH=PROJECTED_WIDTH,
            RATIO=RATIO,
            OVERLAP=OVERLAP,
            SCORE=True,
        )
        values = tl.where(fill, values, 0.0)
        scores = tl.where(fill, scores, float("-inf"))
        state_base = state_sequence * state_stride_seq + state_row * state_stride_row
        tl.store(kv_state + state_base + pd, values, mask=pmask)
        tl.store(score_state + state_base + pd, scores, mask=pmask)


def _run_main(binding: DSV4CompressorBinding) -> None:
    ratio = binding.compress_ratio
    projected_width = DSV4_HEAD_DIM * (2 if ratio == 4 else 1)
    page_rows = DSV4_SOURCE_PAGE_SIZE // ratio
    page_bytes = _compressed_main_page_bytes(ratio)
    _update_pool_pack_main_kernel[(int(binding.hidden_states.shape[0]),)](
        binding.projection,
        binding.positions,
        binding.sequence_ids,
        binding.compressed_slots,
        binding.compressed_cos_sin_cache,
        binding.main_kv_state,
        binding.main_score_state,
        binding.weights.main_ape,
        binding.weights.main_norm,
        binding.compressed_main_cache.view(torch.float8_e4m3fn),
        binding.compressed_main_cache.view(torch.bfloat16),
        binding.compressed_main_cache,
        binding.projection.stride(0),
        binding.compressed_cos_sin_cache.stride(0),
        binding.main_kv_state.stride(0),
        binding.main_kv_state.stride(1),
        binding.eps,
        PROJECTION_OFFSET=0,
        PROJECTED_WIDTH=projected_width,
        PROJECTED_BLOCK=triton.next_power_of_2(projected_width),
        RATIO=ratio,
        OVERLAP=ratio == 4,
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_ROWS=page_rows,
        PAYLOAD_BYTES=DSV4_KV_PAYLOAD_BYTES,
        SCALE_BYTES=DSV4_KV_SCALE_BYTES,
        PAGE_BYTES=page_bytes,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=8,
    )


def _run_index(binding: DSV4CompressorBinding) -> None:
    assert binding.index_cache is not None
    assert binding.index_kv_state is not None
    assert binding.index_score_state is not None
    assert binding.weights.index_ape is not None
    assert binding.weights.index_norm is not None
    _update_pool_pack_index_kernel[(int(binding.hidden_states.shape[0]),)](
        binding.projection,
        binding.positions,
        binding.sequence_ids,
        binding.compressed_slots,
        binding.compressed_cos_sin_cache,
        binding.index_kv_state,
        binding.index_score_state,
        binding.weights.index_ape,
        binding.weights.index_norm,
        binding.index_cache.view(torch.float8_e4m3fn),
        binding.index_cache.view(torch.float32),
        binding.projection.stride(0),
        binding.compressed_cos_sin_cache.stride(0),
        binding.index_kv_state.stride(0),
        binding.index_kv_state.stride(1),
        binding.eps,
        PROJECTION_OFFSET=2 * 1_024,
        PROJECTED_WIDTH=256,
        RATIO=4,
        HEAD_DIM=DSV4_INDEX_HEAD_DIM,
        NOPE_DIM=64,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_ROWS=DSV4_INDEX_PAGE_SIZE,
        PAGE_BYTES=DSV4_INDEX_PAGE_BYTES,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=4,
    )


def _run_prefill_main(binding: DSV4CompressorPrefillBinding) -> None:
    if binding.group_capacity == 0:
        return
    ratio = binding.compress_ratio
    projected_width = DSV4_HEAD_DIM * (2 if ratio == 4 else 1)
    page_rows = DSV4_SOURCE_PAGE_SIZE // ratio
    page_bytes = _compressed_main_page_bytes(ratio)
    _prefill_pool_pack_main_kernel[(binding.group_capacity,)](
        binding.projection,
        binding.active_groups,
        binding.group_source_starts,
        binding.group_rope_positions,
        binding.compressed_slots,
        binding.compressed_cos_sin_cache,
        binding.weights.main_ape,
        binding.weights.main_norm,
        binding.compressed_main_cache.view(torch.float8_e4m3fn),
        binding.compressed_main_cache.view(torch.bfloat16),
        binding.compressed_main_cache,
        binding.projection.stride(0),
        binding.compressed_cos_sin_cache.stride(0),
        binding.eps,
        PROJECTION_OFFSET=0,
        PROJECTED_WIDTH=projected_width,
        RATIO=ratio,
        OVERLAP=ratio == 4,
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_ROWS=page_rows,
        PAYLOAD_BYTES=DSV4_KV_PAYLOAD_BYTES,
        SCALE_BYTES=DSV4_KV_SCALE_BYTES,
        PAGE_BYTES=page_bytes,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=8,
    )


def _run_prefill_index(binding: DSV4CompressorPrefillBinding) -> None:
    if binding.group_capacity == 0:
        return
    assert binding.index_cache is not None
    assert binding.weights.index_ape is not None
    assert binding.weights.index_norm is not None
    _prefill_pool_pack_index_kernel[(binding.group_capacity,)](
        binding.projection,
        binding.active_groups,
        binding.group_source_starts,
        binding.group_rope_positions,
        binding.compressed_slots,
        binding.compressed_cos_sin_cache,
        binding.weights.index_ape,
        binding.weights.index_norm,
        binding.index_cache.view(torch.float8_e4m3fn),
        binding.index_cache.view(torch.float32),
        binding.projection.stride(0),
        binding.compressed_cos_sin_cache.stride(0),
        binding.eps,
        PROJECTION_OFFSET=2 * 1_024,
        PROJECTED_WIDTH=256,
        RATIO=4,
        HEAD_DIM=DSV4_INDEX_HEAD_DIM,
        NOPE_DIM=64,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_ROWS=DSV4_INDEX_PAGE_SIZE,
        PAGE_BYTES=DSV4_INDEX_PAGE_BYTES,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=4,
    )


def _finalize_prefill_state(
    binding: DSV4CompressorPrefillBinding,
    *,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    projection_offset: int,
    projected_width: int,
) -> None:
    _prefill_finalize_state_kernel[(binding.sequence_capacity, int(kv_state.shape[1]))](
        binding.projection,
        binding.active_sequences,
        binding.sequence_offsets,
        binding.state_sequence_ids,
        kv_state,
        score_state,
        ape,
        binding.projection.stride(0),
        kv_state.stride(0),
        kv_state.stride(1),
        PROJECTION_OFFSET=projection_offset,
        PROJECTED_WIDTH=projected_width,
        PROJECTED_BLOCK=triton.next_power_of_2(projected_width),
        RATIO=binding.compress_ratio,
        OVERLAP=binding.compress_ratio == 4,
        num_warps=4,
    )


def run_dsv4_compressor_prefill(*, binding: DSV4CompressorPrefillBinding) -> None:
    """Run allocation-free initial prefill over one fixed graph bucket."""

    if not isinstance(binding, DSV4CompressorPrefillBinding):
        raise TypeError(
            "run_dsv4_compressor_prefill requires a DSV4CompressorPrefillBinding"
        )
    torch.mm(
        binding.hidden_states,
        binding.weights.joint_projection_t,
        out=binding.projection,
    )
    _run_prefill_main(binding)
    _finalize_prefill_state(
        binding,
        kv_state=binding.main_kv_state,
        score_state=binding.main_score_state,
        ape=binding.weights.main_ape,
        projection_offset=0,
        projected_width=DSV4_HEAD_DIM * (2 if binding.compress_ratio == 4 else 1),
    )
    if binding.compress_ratio == 4:
        assert binding.index_kv_state is not None
        assert binding.index_score_state is not None
        assert binding.weights.index_ape is not None
        _run_prefill_index(binding)
        _finalize_prefill_state(
            binding,
            kv_state=binding.index_kv_state,
            score_state=binding.index_score_state,
            ape=binding.weights.index_ape,
            projection_offset=2 * 1_024,
            projected_width=256,
        )


def _run_continuation_main(binding: DSV4CompressorContinuationBinding) -> None:
    if binding.group_capacity == 0:
        return
    ratio = binding.compress_ratio
    projected_width = DSV4_HEAD_DIM * (2 if ratio == 4 else 1)
    page_rows = DSV4_SOURCE_PAGE_SIZE // ratio
    page_bytes = _compressed_main_page_bytes(ratio)
    _continuation_pool_pack_main_kernel[(binding.group_capacity,)](
        binding.projection,
        binding.active_groups,
        binding.group_sequence_slots,
        binding.group_source_positions,
        binding.group_rope_positions,
        binding.compressed_slots,
        binding.sequence_offsets,
        binding.sequence_start_positions,
        binding.state_sequence_ids,
        binding.compressed_cos_sin_cache,
        binding.main_kv_state,
        binding.main_score_state,
        binding.weights.main_ape,
        binding.weights.main_norm,
        binding.compressed_main_cache.view(torch.float8_e4m3fn),
        binding.compressed_main_cache.view(torch.bfloat16),
        binding.compressed_main_cache,
        binding.projection.stride(0),
        binding.compressed_cos_sin_cache.stride(0),
        binding.main_kv_state.stride(0),
        binding.main_kv_state.stride(1),
        binding.eps,
        PROJECTION_OFFSET=0,
        PROJECTED_WIDTH=projected_width,
        RATIO=ratio,
        OVERLAP=ratio == 4,
        HEAD_DIM=DSV4_HEAD_DIM,
        NOPE_DIM=DSV4_NOPE_DIM,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_ROWS=page_rows,
        PAYLOAD_BYTES=DSV4_KV_PAYLOAD_BYTES,
        SCALE_BYTES=DSV4_KV_SCALE_BYTES,
        PAGE_BYTES=page_bytes,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=8,
    )


def _run_continuation_index(binding: DSV4CompressorContinuationBinding) -> None:
    if binding.group_capacity == 0:
        return
    assert binding.index_cache is not None
    assert binding.index_kv_state is not None
    assert binding.index_score_state is not None
    assert binding.weights.index_ape is not None
    assert binding.weights.index_norm is not None
    _continuation_pool_pack_index_kernel[(binding.group_capacity,)](
        binding.projection,
        binding.active_groups,
        binding.group_sequence_slots,
        binding.group_source_positions,
        binding.group_rope_positions,
        binding.compressed_slots,
        binding.sequence_offsets,
        binding.sequence_start_positions,
        binding.state_sequence_ids,
        binding.compressed_cos_sin_cache,
        binding.index_kv_state,
        binding.index_score_state,
        binding.weights.index_ape,
        binding.weights.index_norm,
        binding.index_cache.view(torch.float8_e4m3fn),
        binding.index_cache.view(torch.float32),
        binding.projection.stride(0),
        binding.compressed_cos_sin_cache.stride(0),
        binding.index_kv_state.stride(0),
        binding.index_kv_state.stride(1),
        binding.eps,
        PROJECTION_OFFSET=2 * 1_024,
        PROJECTED_WIDTH=256,
        RATIO=4,
        HEAD_DIM=DSV4_INDEX_HEAD_DIM,
        NOPE_DIM=64,
        ROPE_DIM=DSV4_ROPE_DIM,
        PAGE_ROWS=DSV4_INDEX_PAGE_SIZE,
        PAGE_BYTES=DSV4_INDEX_PAGE_BYTES,
        FP8_MAX=DSV4_FP8_MAX,
        num_warps=4,
    )


def _launch_continuation_finalize_phase(
    binding: DSV4CompressorContinuationBinding,
    *,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    projection_offset: int,
    projected_width: int,
    state_row_start: int,
    state_rows: int,
) -> None:
    ratio = binding.compress_ratio
    _continuation_finalize_state_kernel[(binding.sequence_capacity, state_rows)](
        binding.projection,
        binding.active_sequences,
        binding.sequence_offsets,
        binding.sequence_start_positions,
        binding.state_sequence_ids,
        kv_state,
        score_state,
        ape,
        binding.projection.stride(0),
        kv_state.stride(0),
        kv_state.stride(1),
        PROJECTION_OFFSET=projection_offset,
        PROJECTED_WIDTH=projected_width,
        PROJECTED_BLOCK=triton.next_power_of_2(projected_width),
        RATIO=ratio,
        OVERLAP=ratio == 4,
        STATE_ROW_START=state_row_start,
        num_warps=4,
    )


def _finalize_continuation_state(
    binding: DSV4CompressorContinuationBinding,
    *,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    projection_offset: int,
    projected_width: int,
) -> None:
    ratio = binding.compress_ratio
    _launch_continuation_finalize_phase(
        binding,
        kv_state=kv_state,
        score_state=score_state,
        ape=ape,
        projection_offset=projection_offset,
        projected_width=projected_width,
        state_row_start=0,
        state_rows=ratio,
    )
    if ratio == 4:
        # Preserve old current rows until the previous window has consumed
        # them; the two launches are ordered on the binding's CUDA stream.
        _launch_continuation_finalize_phase(
            binding,
            kv_state=kv_state,
            score_state=score_state,
            ape=ape,
            projection_offset=projection_offset,
            projected_width=projected_width,
            state_row_start=ratio,
            state_rows=ratio,
        )


def run_dsv4_compressor_continuation(
    *, binding: DSV4CompressorContinuationBinding
) -> None:
    """Run one ordered continuation chunk per active sequence without allocation."""

    if not isinstance(binding, DSV4CompressorContinuationBinding):
        raise TypeError(
            "run_dsv4_compressor_continuation requires a "
            "DSV4CompressorContinuationBinding"
        )
    torch.mm(
        binding.hidden_states,
        binding.weights.joint_projection_t,
        out=binding.projection,
    )
    _run_continuation_main(binding)
    if binding.compress_ratio == 4:
        _run_continuation_index(binding)
    _finalize_continuation_state(
        binding,
        kv_state=binding.main_kv_state,
        score_state=binding.main_score_state,
        ape=binding.weights.main_ape,
        projection_offset=0,
        projected_width=DSV4_HEAD_DIM * (2 if binding.compress_ratio == 4 else 1),
    )
    if binding.compress_ratio == 4:
        assert binding.index_kv_state is not None
        assert binding.index_score_state is not None
        assert binding.weights.index_ape is not None
        _finalize_continuation_state(
            binding,
            kv_state=binding.index_kv_state,
            score_state=binding.index_score_state,
            ape=binding.weights.index_ape,
            projection_offset=2 * 1_024,
            projected_width=256,
        )


def run_dsv4_compressor_decode(*, binding: DSV4CompressorBinding) -> None:
    """Run one sequence-unique decode row per active sequence without allocation."""

    if not isinstance(binding, DSV4CompressorBinding):
        raise TypeError("run_dsv4_compressor_decode requires a DSV4CompressorBinding")
    torch.mm(
        binding.hidden_states,
        binding.weights.joint_projection_t,
        out=binding.projection,
    )
    _run_main(binding)
    if binding.compress_ratio == 4:
        _run_index(binding)


__all__ = [
    "DSV4CompressorBinding",
    "DSV4CompressorCaps",
    "DSV4CompressorContinuationBinding",
    "DSV4CompressorPlan",
    "DSV4CompressorPrefillBinding",
    "DSV4CompressorWeights",
    "pack_dsv4_compressor_weights",
    "plan_dsv4_compressor",
    "run_dsv4_compressor_continuation",
    "run_dsv4_compressor_decode",
    "run_dsv4_compressor_prefill",
]
