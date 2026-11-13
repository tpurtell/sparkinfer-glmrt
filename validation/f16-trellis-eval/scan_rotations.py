#!/usr/bin/env python3
"""Read suh/svh (F16) magnitudes and trellis bit widths from shipped EXL3 checkpoints.

Reads only safetensors headers plus the small F16 rotation tensors of a few
layers; never the trellis payloads. Usage: scan_rotations.py SNAPSHOT [LAYERS...]
"""
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def headers(snapshot):
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with f.open("rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            yield f, 8 + n, json.loads(h.read(n))


def main():
    snap = sys.argv[1]
    layers = set(sys.argv[2:]) or None
    bits = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(list)
    for f, base, hdr in headers(snap):
        with f.open("rb") as h:
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                m = re.search(r"layers\.(\d+)\.(.*)\.(trellis|suh|svh)$", name)
                if not m:
                    continue
                layer, mod, kind = m.groups()
                role = "expert" if ".experts." in mod else "core"
                proj = re.sub(r"experts\.\d+\.", "experts.*.", mod)
                if kind == "trellis":
                    bits[role][meta["shape"][-1] // 16] += 1
                    continue
                if layers is not None and layer not in layers:
                    continue
                a, b = meta["data_offsets"]
                h.seek(base + a)
                v = np.frombuffer(h.read(b - a), dtype=np.float16).astype(np.float32)
                stats[(role, proj.split(".")[-1], kind)].append((float(np.abs(v).max()), float(np.sqrt((v * v).mean()))))
    out = {"bits": {r: dict(v) for r, v in bits.items()}, "rotations": {}}
    for (role, proj, kind), vals in sorted(stats.items()):
        mx = [x for x, _ in vals]
        rms = [y for _, y in vals]
        out["rotations"][f"{role}.{proj}.{kind}"] = dict(n=len(vals), max=max(mx), rms_mean=float(np.mean(rms)),
                                                         rms_max=max(rms))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
