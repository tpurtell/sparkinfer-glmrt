"""Native TP4 route reduction at the official V4.1 expert output boundary."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_tp4_kernel(
    p0,
    p1,
    p2,
    p3,
    shared,
    output,
    M: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // H, offset % H
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(TOPK):
        index = (row * TOPK + slot) * H + col
        value = tl.load(p0 + index, row < M, other=0)
        value += tl.load(p1 + index, row < M, other=0)
        value += tl.load(p2 + index, row < M, other=0)
        value += tl.load(p3 + index, row < M, other=0)
        # FC2's BF16 output boundary belongs after TP, separately per expert.
        total += value.to(tl.bfloat16).to(tl.float32)
    if HAS_SHARED:
        total += tl.load(shared + offset, row < M, other=0).to(tl.float32)
    tl.store(output + offset, total, row < M)


@torch.library.custom_op("b12x::v41_reduce_tp4", mutates_args={"output"})
def _launch(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    shared: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    m, topk, h = p0.shape
    _reduce_tp4_kernel[(triton.cdiv(m * h, 256),)](
        p0,
        p1,
        p2,
        p3,
        output if shared is None else shared,
        output,
        m,
        topk,
        h,
        shared is not None,
        256,
    )


@_launch.register_fake
def _fake(p0, p1, p2, p3, shared, output):
    return None


def reduce_v41_tp4_routes(partials, *, output, shared=None):
    """Sum four FP32 [M, topk, H] route planes into caller-owned BF16 output.

    Every plane must use the same token and route order, and the transport must
    finish writing all four planes before this launch on the current stream.
    Shared-expert output, if supplied, is added after routed-expert rounding.
    """
    if len(partials) != 4:
        raise ValueError("V4.1 backbone reduction requires exactly four TP ranks")
    first = partials[0]
    if first.ndim != 3 or any(size <= 0 for size in first.shape):
        raise ValueError("route partials must have positive [M, topk, H] shape")
    m, _, h = first.shape
    if (
        output.shape != (m, h)
        or output.dtype != torch.bfloat16
        or output.device != first.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous BF16 [M, H] on the route device")
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    for value in partials:
        if (
            value.shape != first.shape
            or value.dtype != torch.float32
            or value.device != first.device
            or not value.is_cuda
            or not value.is_contiguous()
        ):
            raise ValueError(
                "TP route planes must be matching contiguous CUDA FP32 tensors"
            )
        if max(output_start, value.data_ptr()) < min(
            output_end, value.data_ptr() + value.numel() * value.element_size()
        ):
            raise ValueError("output must not overlap TP route planes")
    if shared is not None and (
        shared.shape != output.shape
        or shared.dtype != output.dtype
        or shared.device != output.device
        or not shared.is_contiguous()
    ):
        raise ValueError("shared expert must match the BF16 output contract")
    _launch(*partials, shared, output)
    return output
