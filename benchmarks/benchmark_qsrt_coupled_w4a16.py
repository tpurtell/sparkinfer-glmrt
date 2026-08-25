"""Measure ordinary and coupled-Hadamard QSRT W4A16 decode.

The default geometry is Kimi-K3 at TP8: one token, top-16 routing, hidden
width 3,584, and local intermediate width 384.  Both transform policies use
the same fixed-rate payload, scale tensors, route schedule, scratch capacity,
and CUDA graph replay protocol.  The measured difference therefore isolates
the activation-boundary coupled transform and its coupled output reduction.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import statistics

import torch

from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--intermediate-size", type=int, default=384)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--transform",
        choices=("ordinary", "coupled", "both"),
        default="both",
    )
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="print per-kernel CUDA time for one measured replay",
    )
    return parser.parse_args()


def _time_graph(
    graph: torch.cuda.CUDAGraph, *, iterations: int, rounds: int
) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return samples


def main() -> None:
    args = _parse_args()
    if args.experts < args.topk:
        raise ValueError("--experts must be at least --topk")
    if args.hidden_size % 512 or args.intermediate_size % 128:
        raise ValueError(
            "coupled Hadamard requires hidden_size % 512 == 0 and "
            "intermediate_size % 128 == 0"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    if torch.cuda.get_device_capability(device)[0] != 12:
        raise RuntimeError("this benchmark requires an SM12x GPU")

    e = args.experts
    h = args.hidden_size
    i = args.intermediate_size
    bits = args.bits
    w13 = torch.randint(
        -32768,
        32767,
        (2, e, h // 16, i // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (e, i // 16, h // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    h_scale = torch.ones((1, h), dtype=torch.float16, device=device)
    ordinary_i = torch.ones((e, 3 * i), dtype=torch.float16, device=device)
    coupled_signs = torch.randint(
        0, 2, (e, 3 * i), dtype=torch.int32, device=device
    ).mul_(2).sub_(1).to(torch.float16)
    tile_config = (128, 128, 128, 128)
    ordinary = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=h,
        intermediate_size=i,
        num_experts=e,
        activation="situ",
        params_dtype=torch.float16,
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="sqg_e4m3",
        gate_suh=h_scale,
        up_suh=h_scale,
        intermediate_rotations=ordinary_i,
        down_svh=h_scale,
        tile_config=tile_config,
    )
    assert ordinary.trellis is not None
    coupled = replace(
        ordinary,
        trellis=replace(
            ordinary.trellis,
            intermediate_rotations=torch.cat(
                (ordinary_i, coupled_signs), dim=1
            ).contiguous(),
            coupled_hadamard=True,
        ),
    )
    source = (
        torch.randn((1, h), dtype=torch.float32, device=device) * 1.0e-3
    ).to(torch.bfloat16)
    topk_ids = torch.arange(
        args.topk, dtype=torch.int32, device=device
    ).reshape(1, args.topk)
    topk_weights = torch.rand(
        (1, args.topk), dtype=torch.float32, device=device
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"shape=M1/H{h}/I{i}/E{e}/topk{args.topk}/K{bits}"
    )
    print("transform       median_us     min_us     max_us")
    transforms = (("ordinary", ordinary), ("coupled", coupled))
    if args.transform != "both":
        transforms = tuple(
            pair for pair in transforms if pair[0] == args.transform
        )
    for name, prepared in transforms:
        buffers = make_w4a16_packed_buffers(
            prepared,
            m=1,
            topk=args.topk,
            dtype=torch.float16,
            device=device,
            full_rotation=True,
            block_size_m=8,
        )

        def run() -> torch.Tensor:
            return run_w4a16_moe(
                source,
                prepared,
                topk_weights,
                topk_ids,
                activation="situ",
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=buffers.intermediate_cache2,
                output=buffers.output,
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
                route_block_size_m=8,
                intermediate_rotation_scales=prepared.intermediate_rotations,
                full_rotation=True,
                suh_gate_table=prepared.gate_suh,
                suh_up_table=prepared.up_suh,
                svh_table=prepared.down_svh,
                rotation_a_gate=buffers.rotation_a_gate,
                rotation_a_up=buffers.rotation_a_up,
            )

        eager = run().clone()
        torch.cuda.synchronize(device)
        if not torch.isfinite(eager).all():
            raise RuntimeError(f"{name} produced non-finite output")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        if not torch.equal(captured, eager):
            raise RuntimeError(f"{name} graph replay differs from eager")
        samples = _time_graph(
            graph, iterations=args.iterations, rounds=args.rounds
        )
        print(
            f"{name:<15} {statistics.median(samples):9.3f} "
            f"{min(samples):10.3f} {max(samples):10.3f}"
        )
        if args.profile_kernels:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                graph.replay()
                torch.cuda.synchronize(device)
            print(
                profile.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
            )
            for event in sorted(
                profile.key_averages(),
                key=lambda item: item.self_device_time_total,
                reverse=True,
            ):
                if event.self_device_time_total:
                    print(
                        f"{event.self_device_time_total:9.3f} us  "
                        f"{event.key}"
                    )


if __name__ == "__main__":
    main()
