"""Real lossless PCIe DMA versus PyNCCL, with dtype-bound eager replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_PROTO", "LL,LL128,Simple")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce
from benchmarks.common import nvidia_smi_gpu_mode_snapshot
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _repository(root):
    return {
        "worktree": str(root),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "-C", str(root), "status", "--short"], text=True
        ).strip(),
    }


def _worker(rank, args, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.world_size,
    )
    group = dist.group.WORLD
    nccl = PyNcclCommunicator(group, device)
    if not nccl.available or nccl.disabled:
        raise RuntimeError("PyNCCL reference communicator unavailable")
    capacity = args.max_rows * args.hidden_size
    raw = PCIeDmaAllReduce(
        exchange_group=group, device=device, max_bytes=capacity * 4, fp8="0"
    )
    replay = PCIeDmaAllReduce(
        exchange_group=group, device=device, max_bytes=capacity * 4, fp8="0"
    )
    raw.min_bytes = replay.min_bytes = args.min_bytes
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    payload = None
    if rank == 0:
        root = Path(__file__).resolve().parents[1]
        vllm_root = Path(importlib.util.find_spec("vllm").origin).parents[1]
        payload = {
            "command": [sys.executable, *sys.argv],
            "arguments": {**vars(args), "output": str(args.output)},
            "repositories": {"b12x": _repository(root), "vllm": _repository(vllm_root)},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("NCCL_", "CUDA_", "B12X_", "CUTE_", "OMP_"))
            },
            "toolchain": {
                name: importlib.metadata.version(name)
                for name in ("torch", "nvidia-cutlass-dsl", "vllm")
            },
            "scope": "CUDA events around repeated PUBLIC out-of-place eager all_reduce calls. Includes launch gaps and output allocation. Same input, alternating arm order; report all-rank samples and rank-max medians. No full-model speedup inference.",
            "ratio_direction": "PyNCCL median / DMA median; >1 means DMA faster",
            "gpu_before": nvidia_smi_gpu_mode_snapshot(),
            "results": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for dtype_name in args.dtypes:
            replay.prepare_eager_replay(dtypes[dtype_name], max_elements=capacity)
        for dtype_name in args.dtypes:
            dtype = dtypes[dtype_name]
            for rows in args.rows:
                base = (
                    torch.arange(
                        rows * args.hidden_size, device=device, dtype=torch.int32
                    )
                    % 17
                    - 8
                ).float().view(rows, args.hidden_size) / 16
                source = (base + rank / 8).to(dtype)
                expected = (
                    base * args.world_size
                    + args.world_size * (args.world_size - 1) / 16
                ).to(dtype)
                if not raw.should_allreduce(source):
                    raise ValueError(
                        f"Unqualified DMA shape: {dtype_name}, rows={rows}"
                    )
                arms = {
                    "nccl": lambda: nccl.all_reduce(source),
                    "dma_raw": lambda: raw.all_reduce(source),
                    "dma_replay": lambda: replay.all_reduce(source),
                }
                last = {}
                for name, run in arms.items():
                    last[name] = run()
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(last[name], expected, rtol=0, atol=0)
                    for _ in range(args.warmup):
                        last[name] = run()
                torch.cuda.synchronize(device)
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                timings = {name: [] for name in arms}
                rank_samples = {name: [] for name in arms}
                for sample in range(args.samples):
                    order = list(arms) if sample % 2 == 0 else list(reversed(arms))
                    for name in order:
                        dist.barrier()
                        start.record()
                        for _ in range(args.iters):
                            last[name] = arms[name]()
                        end.record()
                        end.synchronize()
                        local_us = start.elapsed_time(end) * 1000 / args.iters
                        all_ranks = [None] * args.world_size
                        dist.all_gather_object(all_ranks, local_us)
                        timings[name].append(max(all_ranks))
                        rank_samples[name].append(all_ranks)
                for name in arms:
                    torch.testing.assert_close(last[name], expected, rtol=0, atol=0)
                # Replays must read new input, not amplify stale in-place data.
                source.neg_()
                for name, run in arms.items():
                    torch.testing.assert_close(run(), -expected, rtol=0, atol=0)
                result = {
                    "dtype": dtype_name,
                    "rows": rows,
                    "hidden_size": args.hidden_size,
                    "bytes": source.nbytes,
                    "replay_elements": capacity,
                    "samples_rank_max_us": timings,
                    "samples_per_rank_us": rank_samples,
                    "median_us": {
                        name: median(values) for name, values in timings.items()
                    },
                    "nccl_over_dma_raw": median(timings["nccl"])
                    / median(timings["dma_raw"]),
                    "nccl_over_dma_replay": median(timings["nccl"])
                    / median(timings["dma_replay"]),
                    "correctness": "Exact dyadic rank-sum before/after timing and after input mutation; outputs out-of-place and finite.",
                }
                if rank == 0:
                    payload["results"].append(result)
                    payload["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
                    args.output.write_text(json.dumps(payload, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                key: result[key]
                                for key in (
                                    "dtype",
                                    "rows",
                                    "bytes",
                                    "median_us",
                                    "nccl_over_dma_replay",
                                )
                            }
                        ),
                        flush=True,
                    )
    finally:
        torch.cuda.synchronize(device)
        replay.close()
        raw.close()
        nccl.destroy()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--rows", default="1024,2048,4096")
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--dtypes", default="bf16,fp32")
    parser.add_argument("--min-bytes", type=int, default=6 << 20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.rows = [int(value) for value in args.rows.split(",")]
    args.dtypes = args.dtypes.split(",")
    if not args.rows or any(row <= 0 or row > args.max_rows for row in args.rows):
        parser.error("rows must be positive and within max-rows")
    if any(dtype not in ("bf16", "fp32") for dtype in args.dtypes):
        parser.error("dtypes must be bf16/fp32")
    if (
        min(args.world_size, args.hidden_size, args.max_rows, args.iters, args.samples)
        <= 0
        or args.warmup < 0
    ):
        parser.error("invalid benchmark capacity or repetition count")
    if torch.cuda.device_count() < args.world_size:
        parser.error("not enough visible CUDA devices")
    mp.spawn(_worker, args=(args, _free_port()), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
