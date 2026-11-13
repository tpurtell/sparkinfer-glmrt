"""Time the actual intrinsic NVFP4 quantizers, one thread per 16-value block.

Rows are 6144-wide by default (GLM/ML4/MiniMax input width). This isolates
quantizer arithmetic and memory traffic, not a complete routed expert.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32, Uint64
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import max_abs_16, quantize_block_fp4, quantize_block_fp4_fast
from benchmarks.bench_cuteafd_glm_programs import warmed_samples


class QuantizeBlocks:
    def __init__(self, fast):
        self.fast = fast

    @cute.jit
    def __call__(self, values: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor,
                 blocks: Int32, stream: cuda.CUstream):
        self.kernel(values, packed, scales, blocks).launch(
            grid=((blocks + 127) // 128, 1, 1), block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, values: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor, blocks: Int32):
        block = cute.arch.block_idx()[0] * Int32(128) + cute.arch.thread_idx()[0]
        if block < blocks:
            row = cute.make_rmem_tensor((16,), Float32)
            for i in cutlass.range_constexpr(16):
                row[i] = Float32(values[block, i])
            amax = max_abs_16(row)
            if cutlass.const_expr(self.fast):
                payload, scale = quantize_block_fp4_fast(row, amax, Float32(1.0))
            else:
                payload, scale = quantize_block_fp4(row, amax, Float32(1.0))
            packed[block, 0] = Uint32(payload & Uint64(0xFFFFFFFF))
            packed[block, 1] = Uint32(payload >> Uint64(32))
            scales[block] = Int32(scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="64,2752,8192")
    parser.add_argument("--width", type=int, default=6144)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=16)
    parser.add_argument("--l2-cold", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = [int(r) for r in args.rows.split(",")]
    if args.width % 16 or min(counts) < 1:
        parser.error("positive rows and width divisible by 16 required")
    capacity = max(counts) * args.width // 16
    gen = torch.Generator(device="cuda").manual_seed(289)
    values = torch.randn(capacity, 16, generator=gen, device="cuda")
    # Include subnormal-scale blocks without changing the instruction sequence.
    values[::2].mul_(0.01)
    packed = torch.empty(capacity, 2, dtype=torch.uint32, device="cuda")
    scales = torch.empty(capacity, dtype=torch.int32, device="cuda")
    operands = tuple(from_dlpack(t, assumed_align=16) for t in (values, packed, scales))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    report = dict(device=torch.cuda.get_device_name(), width=args.width, results=[])
    directory = args.output.parent / (args.output.stem + "-objects")
    directory.mkdir(parents=True, exist_ok=True)
    for fast in (False, True):
        name = "quantize_block_fp4" + ("_fast" if fast else "")
        compiled = cute.compile(QuantizeBlocks(fast), *operands, Int32(capacity), stream)
        compiled.export_to_c(str(directory), name, name)
        for rows in counts:
            blocks = rows * args.width // 16
            def launch():
                compiled(*operands, Int32(blocks), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            samples = warmed_samples(launch, args)
            entry = dict(program=name, rows=rows, median_us=median(s["us"] for s in samples), samples=samples)
            report["results"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in entry.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
