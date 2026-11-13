#!/usr/bin/env python3
"""K6 dense trellis: F64 oracle for BF16 vs F16 MMA arms, plus quantization error.

Copies k8duel-gem/eval_r2_oracle.py. Weights come from the official EXL3
encoder (GPTQModel quantize_tiles/pack_trellis) on Gaussian tiles, so for the
same tile we have: W_src (encoder input, F32), W_q (decoded F16 weights, checked
against tests/_reference/native_trellis.py) and the packed payload. Errors:
  quant  : x @ W_q   vs x @ W_src         (quantization error at this width)
  kernel : arm output vs x_arm @ W_q (F64) (kernel-added error, same rounded input)
  total  : arm output vs x_full @ W_src   (end-to-end vs full-precision)
Rotated activations come from the real H128 ingress formula (F32 x*suh, H128).
Usage: eval_k6_oracle.py ROOT_BF16 ROOT_F16 OUT.json K:N:SIDE ...
"""
import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/cuteafd/third_party/sparkinfer")
from tests._reference.native_trellis import Reference

import os
BITS = int(os.environ.get("BITS", "6"))
SUH = float(os.environ.get("SUH", "1.0"))
XRMS = float(os.environ.get("XRMS", "1.0"))  # raw hidden-row rms  # real checkpoints: rms 0.015-0.03
ROWS = [1, 8, 16, 64, 2048]
MAX_ROWS = int(os.environ.get("MAX_ROWS", "2048"))
ENCODER = "/opt/cuteafd-target/enc/trellis_bench_encoder.so"


def load_lib(path: Path):
    lib = C.CDLL(str(path.resolve()))
    lib.load.argtypes, lib.load.restype = [], None
    lib.run.argtypes = [C.c_void_p] * 5 + [C.c_int32] * 3 + [C.c_void_p]
    lib.run.restype = C.c_int32
    lib.load()
    return lib


def fwht_rows(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    h = x.reshape(-1, 128).clone()
    n = 1
    while n < 128:
        h = h.reshape(-1, 128 // (2 * n), 2, n)
        a, b = h[:, :, 0, :], h[:, :, 1, :]
        h = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        n *= 2
    return (h * 128.0 ** -0.5).reshape(shape)


def rel(out, ref):
    d = out.double() - ref
    return float(torch.linalg.norm(d) / torch.linalg.norm(ref))


PERM = []
for lane in range(32):
    r, c = 2 * (lane % 4), lane // 4
    PERM.extend((r*16+c, (r+1)*16+c, (r+8)*16+c, (r+9)*16+c,
                 r*16+c+8, (r+1)*16+c+8, (r+8)*16+c+8, (r+9)*16+c+8))


def encode(k, n, side, costs, edges):
    """Official encoder in K128 chunks; returns payload (u8), W_src [K,N] f32, W_q [K,N] f32."""
    inverse = torch.argsort(torch.tensor(PERM, device="cuda"))
    packed = torch.empty((k // 16, n // 16, 16 * BITS), dtype=torch.int16, device="cuda")
    w_src = torch.empty((k, n), dtype=torch.float32, device="cuda")
    w_q = torch.empty((k, n), dtype=torch.float32, device="cuda")
    oracle = Reference("mcg")
    for begin in range(0, k, 128):
        tiles = torch.randn((128 * n // 256, 256), device="cuda", dtype=torch.float32)
        quant = torch.empty_like(tiles)
        idx = torch.empty_like(tiles, dtype=torch.int16)
        torch.ops.trellis_bench_encoder.quantize_tiles(tiles, quant, idx, costs, edges, BITS, True, False)
        enc = torch.empty((8, n // 16, 16 * BITS), device="cuda", dtype=torch.int16)
        torch.ops.trellis_bench_encoder.pack_trellis(enc, idx.view(8, n // 16, 256), BITS)
        if begin in (0, k - 128):
            got = oracle.decode_tiles(enc.view(-1, 16 * BITS)[:8].cpu().numpy().tobytes(), BITS).reshape(8, 256)[:, PERM]
            if not np.array_equal(got.view(np.uint32), quant[:8].cpu().numpy().view(np.uint32)):
                raise RuntimeError("encoder reconstruction differs from independent decoder")
        packed[begin // 16:(begin + 128) // 16] = enc
        to_kn = lambda t: t[:, inverse].view(8, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(128, n)
        w_src[begin:begin + 128] = to_kn(tiles)
        w_q[begin:begin + 128] = to_kn(quant)
    if side == 1:
        payload = packed.view(k // 16, n // 128, 8, 16 * BITS).permute(1, 0, 2, 3).contiguous()
    else:
        payload = packed.view(k // 128, 8, n // 128, 8, 16 * BITS).permute(0, 2, 1, 3, 4).contiguous()
    return payload.view(torch.uint8).flatten(), w_src, w_q


def main():
    root_b, root_f, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    shapes = [tuple(map(int, s.split(":"))) for s in sys.argv[4:]]
    torch.ops.load_library(ENCODER)
    props = torch.cuda.get_device_properties(0)
    cta_limit = props.multi_processor_count * 4
    stream = torch.cuda.current_stream().cuda_stream
    costs = torch.empty((256, 2, 65536 >> BITS), device="cuda", dtype=torch.float16)
    edges = torch.empty((256, 256, 65536 >> BITS), device="cuda", dtype=torch.int16)
    report = dict(bits=BITS, device=props.name, rows=ROWS, shapes=[])
    for k, n, side in shapes:
        torch.manual_seed(273)
        splits = (k + 1023) // 1024
        sub = f"k{k}_n{n}_s{side}_b{BITS}_mcg_{{}}_mr{MAX_ROWS}"
        libs = {"bf16": load_lib(root_b / sub.format("bf16") / "candidate.so"),
                "f16": load_lib(root_f / sub.format("f16") / "candidate.so")}
        payload, w_src, w_q = encode(k, n, side, costs, edges)
        cap = max(ROWS)
        # Real ingress: raw BF16 hidden row, F32 x*suh, H128, then the arm's dtype.
        raw = (torch.randn(cap, k, device="cuda") * XRMS).to(torch.bfloat16)
        suh = ((torch.rand(k, device="cuda") * 1.75 + 0.125) * SUH).to(torch.float16) * \
            torch.where(torch.rand(k, device="cuda") < 0.5, -1.0, 1.0).to(torch.float16)
        rot = fwht_rows(raw.float() * suh.float())
        xs = {"bf16": rot.to(torch.bfloat16).contiguous(), "f16": rot.to(torch.float16).contiguous()}
        scales = torch.ones(k // 128, device="cuda", dtype=torch.float16)
        w_q64, w_src64 = w_q.double(), w_src.double()
        rot64 = rot.double()
        f16r = rot.to(torch.float16)
        tiny = 2.0 ** -14
        rec = dict(shape=f"{k}:{n}:{side}", suh_scale=SUH, x_rms=XRMS, min_abs_rotated=float(rot.abs().min()),
                   f16_subnormal_fraction=float(((f16r != 0) & (f16r.abs() < tiny)).float().mean()),
                   f16_underflow_zero_fraction=float(((f16r == 0) & (rot != 0)).float().mean()),
                   rms_rotated=float(rot.pow(2).mean().sqrt()), max_abs_raw=float(raw.float().abs().max()),
                   max_abs_rotated=float(rot.abs().max()), max_abs_weight=float(w_q.abs().max()),
                   max_abs_weight_src=float(w_src.abs().max()), records=[])
        for rows in ROWS:
            ref_full = rot64[:rows] @ w_src64            # unquantized, unrounded
            ref_q_full = rot64[:rows] @ w_q64            # quantized weights, unrounded x
            r = dict(rows=rows, quant=rel(ref_q_full, ref_full))
            for arm, lib in libs.items():
                ptrs = torch.tensor([xs[arm].data_ptr()], device="cuda", dtype=torch.int64)
                part = torch.full((129 * splits * n,), float("nan"), device="cuda")
                out = torch.full((cap * n,), float("nan"), device="cuda")
                rc = lib.run(ptrs.data_ptr(), payload.data_ptr(), scales.data_ptr(), part.data_ptr(),
                             out.data_ptr(), rows, 0, cta_limit, stream)
                torch.cuda.synchronize()
                if rc:
                    raise RuntimeError(f"{arm} run failed {rc}")
                o = out[:rows * n].view(rows, n)
                if not torch.isfinite(o).all():
                    raise RuntimeError(f"{arm} output non-finite")
                same_x = xs[arm][:rows].double() @ w_q64
                r[arm] = dict(kernel=rel(o, same_x), vs_q_full=rel(o, ref_q_full), total=rel(o, ref_full),
                              input_rounding=rel(same_x, ref_q_full))
            r["kernel_reduction_share_of_total_bf16"] = (r["bf16"]["vs_q_full"] - r["f16"]["vs_q_full"]) / r["bf16"]["total"]
            rec["records"].append(r)
            print(json.dumps({"shape": rec["shape"], **r}), flush=True)
        report["shapes"].append(rec)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != "records"}), flush=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        del payload, w_src, w_q, w_q64, w_src64, rot64
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
