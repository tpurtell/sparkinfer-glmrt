"""Independent NumPy trellis decode + Torch MXFP8/coupled single-expert oracle.

Run directly with a pinned sidecar. This is correctness evidence, not a benchmark.
The tile layout and MCG constants follow test_trellis_linear.py; no native
weight-decoder or MMA implementation is used by the reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE
from b12x.moe._shared.trellismx.p8_coupled_scales import hadamard_blocks


def decode_physical(trellis, scales):
    """Return [N,K] FP32 weights from independent sliding L16 ring windows."""
    native = trellis.contiguous().numpy()
    kt, nt, extent = native.shape
    bits = extent // 16
    packed = native.view(np.uint16).reshape(kt, nt, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (packed[..., 1].astype(np.uint32) << 16)
    block = np.empty((kt, nt, 16, 16), dtype=np.float16)
    for lane in range(32):
        rows = ((lane % 4) * 2, (lane % 4) * 2 + 1,
                (lane % 4) * 2 + 8, (lane % 4) * 2 + 9)
        for weight in range(8):
            end = (lane * 8 + weight + 257) * bits
            first, last = (end - 16) // 32, (end - 1) // 32
            merged = (words[..., first % (8 * bits)].astype(np.uint64) << 32) | words[..., last % (8 * bits)]
            window = ((merged >> ((last + 1) * 32 - end)) & 65535).astype(np.uint64)
            value = ((window * np.uint64(0xCBAC1FED)) & np.uint64(0xffffffff)).astype(np.uint32)
            value = (value & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
            lo = (value & 65535).astype(np.uint16).view(np.float16)
            hi = (value >> 16).astype(np.uint16).view(np.float16)
            # Alpha2 compander, with both arithmetic operations rounded to FP16.
            decoded = ((lo + hi).astype(np.float16) * np.float16(2)).astype(np.float16)
            col = 2 * (lane // 8 + (4 if weight >= 4 else 0)) + ((lane >> 2) & 1)
            block[..., rows[weight % 4], col] = decoded
    logical = torch.from_numpy(block.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16).copy())
    logical = logical.to(torch.float8_e4m3fn).float().T.contiguous()
    factor = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    return logical * factor


def quant_rows(value):
    blocks = value.float().reshape(value.shape[0], -1, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    ratio = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    scale = torch.exp2(torch.ceil(torch.log2(ratio)).clamp(-127, 127))
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(value)


def reference(x, gate, up, down, component, expert):
    gate_svh, up_svh, down_suh = component.split_intermediate()
    pre_signs, post_signs = component.split_signs()
    source = hadamard_blocks(x.half().float(), 512)
    source = quant_rows(hadamard_blocks(source * component.gate_up_suh.float(), 128))
    g = (source @ gate.T).half()
    u = (source @ up.T).half()
    m, width = g.shape
    raw = torch.stack((g.reshape(m, -1, 32), u.reshape(m, -1, 32)), dim=2).reshape(m, 2 * width)
    raw_scale = torch.stack((gate_svh[expert].reshape(-1, 32), up_svh[expert].reshape(-1, 32)), dim=1).reshape(1, 2 * width)
    mixed = hadamard_blocks(hadamard_blocks(raw.float(), 128) * raw_scale.float(), 128) * pre_signs.float()
    g, u = mixed[:, 0::2].clamp(max=10), mixed[:, 1::2].clamp(-10, 10)
    middle = hadamard_blocks(g * torch.sigmoid(g) * u * post_signs.float(), 128)
    middle = quant_rows(hadamard_blocks(middle * down_suh[expert].float(), 128))
    result = (middle @ down.T).half().float()
    return hadamard_blocks(hadamard_blocks(result, 128) * component.down_svh.float(), 512)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--layer', type=int, default=4)
    parser.add_argument('--rank', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = json.loads((args.checkpoint / 'trellismx-manifest.json').read_text())
    record = next(r for r in manifest['files'] if (r['layer'], r['rank']) == (args.layer, args.rank))
    path = args.checkpoint / record['path']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    runtime = P8NativeTPMoE(path, device=torch.device('cuda'), tp_rank=args.rank,
        world_size=4, layer=args.layer, intermediate=512,
        expected_design_sha256=record['source_design_sha256'],
        expected_transform_sha256=hashlib.sha256((args.checkpoint / 'design/transform.json').read_bytes()).hexdigest(),
        small_m_scheduler=True, fc1_tile_n=128, fuse_scratch_zero=True,
        grid_policy=True, fc1_warp_quant=False, fc1_broadcast_a=True)
    print(json.dumps({'sidecar_sha256': record['sha256'], 'gpu': torch.cuda.get_device_name(),
                      'scope': 'independently decoded single-expert numerical comparison'}), flush=True)
    torch.manual_seed(20260910)
    with safe_open(path, framework='pt', device='cpu') as sf, torch.inference_mode():
        for expert in (0, 137, 287):
            # Trellis slots are gate/up; scale planes are up/gate.
            w13 = sf.get_tensor('w13_trellis')[:, expert]
            s13 = sf.get_tensor('w13_scale_ue8m0')[expert]
            gate = decode_physical(w13[0], s13[512:])
            up = decode_physical(w13[1], s13[:512])
            down = decode_physical(sf.get_tensor('w2_trellis')[expert], sf.get_tensor('w2_scale_ue8m0')[expert])
            for m, amplitude in ((1, 0.1), (8, 1.0), (32, 0.1)):
                x = (torch.randn(m, 4096) * amplitude).bfloat16()
                ids = torch.tensor([(expert + i) % 288 for i in range(8)], dtype=torch.int32).expand(m, -1).contiguous().cuda()
                weights = torch.zeros(m, 8, device='cuda'); weights[:, 0] = 1
                actual = runtime(x.cuda(), weights, ids).float().cpu()
                expected = reference(x, gate, up, down, runtime.scale_component, expert)
                cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
                relative_l2 = ((actual - expected).norm() / expected.norm()).item()
                print(json.dumps({'expert': expert, 'tokens': m, 'amplitude': amplitude,
                                  'cosine': cosine, 'relative_l2': relative_l2}), flush=True)
                assert torch.isfinite(actual).all() and cosine >= 0.999 and relative_l2 <= 0.02
    print(json.dumps({'status': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
