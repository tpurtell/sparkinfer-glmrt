"""Fused top-k-sum + compressed reply launch, for the Spark reply duel.

``compile_fused_sum_reply`` compiles the full-rotation top-k sum with the reply
quantization fused into its epilogue and returns a plain launch closure. The
closure takes Torch buffers (fc2 route values, wire output, router weights,
router ids, expert map, svh table), the live row count and a CUstream; every
compile and allocation happens here, in the factory.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .kernel import compile_w4a16_topk_sum


_ELEMENT_DTYPES = {"bf16": cutlass.BFloat16, "fp16": cutlass.Float16}


def compile_fused_sum_reply(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str = "fp16",
    full_rotation: bool = True,
    full_rotation_output_dtype: str = "bf16",
    num_experts: int = 0,
    route_num_experts: int = 0,
    route_ids_dtype: torch.dtype = torch.int32,
    use_expert_map: bool = True,
    broadcast_svh: bool = False,
    reply_format: int,
):
    """Compile the fused kernel and return ``launch(fc2, out, weights, ids, map, svh, rows, stream)``."""
    reply_format = int(reply_format)
    if not full_rotation:
        raise ValueError("the fused reply epilogue requires full rotation")
    launcher = compile_w4a16_topk_sum(
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        full_rotation=True,
        full_rotation_output_dtype=full_rotation_output_dtype,
        num_experts=num_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=use_expert_map,
        broadcast_svh=broadcast_svh,
        reply_format=reply_format,
    )
    element = _ELEMENT_DTYPES[element_dtype]
    route_dtype = cutlass.Int32 if route_ids_dtype == torch.int32 else cutlass.Int64
    route_align = 4 if route_ids_dtype == torch.int32 else 8

    def launch(fc2, out, weights, ids, expert_map, svh, rows, stream):
        launcher.compiled(
            make_ptr(element, fc2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, weights.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(route_dtype, ids.data_ptr(), cute.AddressSpace.gmem, assumed_align=route_align),
            make_ptr(cutlass.Int32, expert_map.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(cutlass.Float16, svh.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            cutlass.Int32(num_experts),
            cutlass.Int32(route_num_experts),
            rows,
            stream,
        )

    return launch
