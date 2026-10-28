#!/usr/bin/env python3
"""Frozen decode harness for the Kimi K3 EXL3 K2 cooperative Trellis kernel.

Target (#819): `W4A16MixedTrellisKernel` in
`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`, the cooperative K2 expert
kernel that carried 96.79% of Spark kernel busy time in the saved Kimi K3
decode trace (plus its 0.15% m1 specialization: 96.94% together).

What this harness freezes
-------------------------
* The oracle is the kernel source at the cuteafd pin (see FROZEN_BLOBS),
  archived by `freeze-oracle` as a git tree plus per-file SHA256 and staged on
  both duel seeds. Every candidate output must be byte-identical to the
  oracle's on every fixture.
* Weights are real: two fixed layers (13 early, 92 late) of the shipped
  `wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6` checkpoint, rank-0 of the TP6
  width-512 Spark slice, all 896 routed experts at K2 (`situ`, unclamped),
  extracted by `stage` into one safetensors file per layer.
* The launch is the engine's, not a re-derivation: the AOT export contract at
  the pin (hidden 3584, rank intermediate 512, 896 experts, top-k 16, tile
  64/256/64/256, 8-row route blocks, L2 evict-first decode schedule, tiers
  K2/K3 with 896/896 compile counts and runtime tier-1 counts 0), precompiled
  route-pack launches per capacity, then the cooperative kernel and TopKSum.
* Fixtures are trace-calibrated. The saved trace carries per-launch cost but no
  expert IDs, so the number of distinct experts touched per call is the
  uniform-bandwidth lower bound implied by each row count's measured launch
  time (weighted isotonic fit over the trace, see `derive_distinct_curve`).
  Every fixture is built independently at its row count with exactly that many
  distinct experts.

Protocol
--------
`fixtures` writes the fixture manifest (hashes of every generated tensor plus
the trace fit). `stage` extracts the two layers from the checkpoint. `run`
loads one arm (oracle or candidate), checks byte exactness on rows 1..80 for
both layers, then times the full graph on the shared timing set with
duplicate-control graphs and writes JSON. `session` keeps several arms
resident in one process per arm and interleaves one timing round per arm in
rotating order, so every entrant and the oracle are measured on the same seed,
in the same session, against the same frozen digests; it then applies the ship
bar (share-weighted speedup >= 1.10x on both layers, no fixture regressing
beyond duplicate-control noise, byte-exact first).

The timing set is weighted by the trace's own share of kernel busy time:
rows 72, 41, 33, 31/32 (~70%), plus 5 and 6 for C1 verify and m1 at row 1.
Every candidate must also declare its installed decode capacities; adding a
specialization requires extra exports and planner workspace and is timed at
the capacity the row policy would actually pick.

All estimates in this file are nominal: the roofline divides unique weight
bytes by 273 GB/s and is not a measured ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Frozen provenance

PIN_COMMIT = "26a1e5072fb707864845b88db65ddf1e80c5c872"

# `git show <PIN>:<path> | sha256sum` for every file on the serving launch path.
FROZEN_BLOBS = {
    "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py":
        "27281fe21837a68caf46c41b2cd3df8f4c008f33781d4c68ae20489cff161fbb",
    "b12x/moe/_shared/kernels/w4a16/kernel.py":
        "7063dcd7e28e837d3fb91493bb809ec5ea6f82d5656a8eb0038e9dcff0214a7c",
    "b12x/moe/_shared/kernels/w4a16/host.py":
        "4f1881e1db9245775f14aead47c5b911d1e262774c507f9863b1a73bb833266f",
    "b12x/moe/_shared/kernels/w4a16/route_pack.py":
        "ef50f695bfc4a5960a5300bbae9301cb86da9ddcb64a904d540d22f8907e4bb6",
    "b12x/moe/_shared/kernels/w4a16/prepare.py":
        "554f3d9a8e8843fde1e2b086c667fa8778bd51acde635c7b831ea9157ac2ecce",
    "b12x/moe/fused_moe/trellis.py":
        "20e4499a3c574df9f4b4ba92dfaef71e91ba0593a0b718c4992c1d1bc19b9be0",
    "b12x/moe/fused_moe/_impl.py":
        "5511938eee0612aaa578bbdc4fdd68363416797f97e6479948d5d7429ea252cd",
}

# ---------------------------------------------------------------------------
# Frozen engine geometry (python/tools/aot/package_exl3_aot.py 'kimik3'
# tp6-width512, bits (2, 3), capacities 1/80; trace grid 48x1x1, 256 threads,
# 35,840 B dynamic shared memory)

HIDDEN = 3584                 # checkpoint hidden size
INTERMEDIATE = 512            # per-rank intermediate width (3072 / TP6)
FULL_INTERMEDIATE = 3072      # checkpoint intermediate size, sliced per rank
EXPERTS = 896                 # routed experts per layer
TOPK = 16
TIER_BITS = (2, 3)            # routed experts K2; empty K3 tier mirrors the AOT export
ROUTE_BLOCK = 8
TILE = (64, 256, 64, 256)
SMS = 48                      # GB10 SM121 SM count; AOT export and trace agree
BLOCKS_PER_SM = 1
WORKSPACE_SMS = SMS           # EXL3_AOT_MAX_SMS[(12, 1)]; the exporter's sms/workspace_sms
DECODE_SCHEDULE = "l2=2,pf1=0,pf2=0,pdl=1"   # PURE_EVICT_FIRST
ACTIVATION = "situ"
SWIGLU_LIMIT = None
LAYERS = (13, 92)
CAPACITY_ROWS = 80
CAPACITY_M1 = 1
TAIL_ROWS = 8

EXPERT_BYTES = 3 * HIDDEN * INTERMEDIATE * 2 // 8   # 1,376,256 B at K2
NOMINAL_BANDWIDTH_GB_S = 273.0

# ---------------------------------------------------------------------------
# Trace evidence (coordinator nsys 2026.2.1; saved trace
# kimi-k6-profile-r3-nsys/analysis/ostrich-rank0.sqlite, session epoch
# 1792681706125078052, 36.7 s, 104,400 launches, 17,095.4 ms kernel busy).

TRACE_META = {
    "sqlite": "~/.cache/cuteafd/builds/kimi-k6-profile-r3-nsys/analysis/ostrich-rank0.sqlite",
    "expert_timing": "~/.cache/cuteafd/builds/kimi-k6-profile-r3-nsys/"
                     "kimi-k3-max-prof-r1/profile/expert-timing-ostrich-rank0.jsonl",
    "session_start_epoch_ns": 1792681706125078052,
    "window_s": 36.7,
    "all_kernel_launches": 104400,
    "all_kernel_busy_ms": 17095.4,
    "m80_busy_ms": 16547.0,
    "m80_share_pct": 96.792,
    "m1_busy_ms": 25.0,
    "m1_share_pct": 0.146,
    "combined_share_pct": 96.938,
    "note": "saved traces carry no expert IDs; distinct-expert counts below are "
            "the uniform-bandwidth lower bound implied by measured launch time",
}

# (rows, launches, ms per launch) for every row count observed in the trace.
TRACE_ROWS = (
    (1, 160, 0.156),
    (2, 147, 0.224),
    (4, 880, 0.333),
    (5, 1760, 0.387),
    (6, 1280, 0.439),
    (8, 80, 0.531),
    (10, 240, 0.415),
    (13, 320, 0.496),
    (14, 320, 0.822),
    (17, 80, 0.822),
    (19, 160, 0.841),
    (21, 80, 1.126),
    (23, 80, 1.284),
    (25, 160, 0.839),
    (26, 160, 0.949),
    (28, 80, 1.227),
    (30, 240, 1.234),
    (31, 800, 1.359),
    (32, 1040, 1.355),
    (33, 1680, 1.391),
    (34, 320, 1.453),
    (35, 160, 1.713),
    (36, 160, 1.400),
    (37, 80, 1.470),
    (38, 240, 1.477),
    (40, 80, 1.460),
    (41, 2480, 1.275),
    (42, 240, 1.366),
    (47, 80, 1.524),
    (72, 1360, 2.457),
)

# Share of all kernel busy time (%) for the timed rows.
TIMING_SHARES_PCT = {
    1: 0.146,
    5: 3.984,
    6: 3.286,
    31: 6.359,
    32: 8.243,
    33: 13.669,
    41: 18.497,
    72: 19.544,
}

TIMING_ROWS = (1, 5, 6, 31, 32, 33, 41, 72)
EXACTNESS_ROWS = tuple(range(1, CAPACITY_ROWS + 1))
SHIP_BAR = 1.10
NOISE_FLOOR = 0.01

DEFAULT_CAPACITIES = ({"capacity": CAPACITY_M1}, {"capacity": CAPACITY_ROWS})


# ---------------------------------------------------------------------------
# Pure geometry shared by the harness and its CPU tests


def per_expert_ms(expert_bytes: int = EXPERT_BYTES,
                  bw_gb_s: float = NOMINAL_BANDWIDTH_GB_S) -> float:
    """Nominal time to stream one expert's K2 weights once (ms)."""
    return expert_bytes / (bw_gb_s * 1e6)


def roofline_ms(distinct_experts: int,
                expert_bytes: int = EXPERT_BYTES,
                bw_gb_s: float = NOMINAL_BANDWIDTH_GB_S) -> float:
    """Nominal optimistic floor: unique weight bytes / nominal bandwidth."""
    return distinct_experts * expert_bytes / (bw_gb_s * 1e6)


def _block_average(block):
    return sum(v * w for _, v, w in block) / sum(w for _, _, w in block)


def _isotonic(points):
    """Weighted pool-adjacent-violators; points are (row, value, weight)."""
    stack = []
    for point in points:
        stack.append([point])
        while len(stack) > 1 and _block_average(stack[-2]) > _block_average(stack[-1]):
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right)
    return {row: _block_average(block) for block in stack for row, _, _ in block}


def derive_distinct_curve(trace_rows=TRACE_ROWS,
                          expert_bytes: int = EXPERT_BYTES,
                          bw_gb_s: float = NOMINAL_BANDWIDTH_GB_S):
    """Distinct experts per call implied by each trace row count.

    Raw value = min(trace ms / per-expert nominal ms, 16*rows, 896), floored
    at the 16 experts that one row's top-16 routes must touch; weighted
    isotonic in launch count removes per-layer placement noise without
    inventing non-monotone steps. Widths the trace never launched are linearly
    interpolated between fitted rows and extrapolated past the last one.
    """
    per_ms = per_expert_ms(expert_bytes, bw_gb_s)
    observed = sorted(int(row) for row, _, _ in trace_rows)
    table = {int(row): (int(n), float(ms)) for row, n, ms in trace_rows}
    raw = []
    for row in observed:
        launches, ms = table[row]
        value = min(float(ms) / per_ms, float(min(16 * row, EXPERTS)))
        raw.append((row, max(16.0, value), float(launches)))
    iso = _isotonic(raw)

    def interpolate(row):
        if row in iso:
            return iso[row]
        below = [r for r in observed if r < row]
        above = [r for r in observed if r > row]
        if not below:
            return iso[above[0]]
        if not above:
            a, b = observed[-2], observed[-1]
            return iso[b] + (row - b) * (iso[b] - iso[a]) / (b - a)
        a, b = below[-1], above[0]
        return iso[a] + (row - a) * (iso[b] - iso[a]) / (b - a)

    curve = {}
    for row in range(1, CAPACITY_ROWS + 1):
        value = interpolate(row)
        curve[row] = max(16, min(int(round(value)), min(16 * row, EXPERTS)))
    if any(curve[r] > curve[r + 1] for r in range(1, CAPACITY_ROWS)):
        raise AssertionError("distinct-expert curve is not monotone")
    return curve


def route_slots_for_capacity(capacity: int, topk: int = TOPK,
                             block_size: int = ROUTE_BLOCK,
                             num_experts: int = EXPERTS):
    """Canonical b12x route packing geometry for one capacity.

    Mirrors host.route_pack_capacity(capacity * topk, block, experts, topk):
    token capacity rounds up to a power of two, packed slots reserve one
    block per expert for uneven groups, and below one expert per slot the
    smaller of numel * block and the uneven reservation wins.
    """
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError(f"capacity must be positive, got {capacity}")
    tokens = (capacity * topk + topk - 1) // topk
    token_capacity = 1 << (max(tokens, 1) - 1).bit_length()
    numel_capacity = token_capacity * topk
    slots = numel_capacity + num_experts * (block_size - 1)
    if numel_capacity < num_experts:
        slots = min(numel_capacity * block_size, slots)
    slots = max(slots, 1)
    blocks = (slots + block_size - 1) // block_size
    return numel_capacity, slots, blocks


def planner_tile(capacity: int, hidden: int = HIDDEN,
                 intermediate: int = INTERMEDIATE, configured=None):
    """b12x `_projection_mixed_tile_config` with no configured override."""
    if configured is not None:
        return tuple(int(value) for value in configured)
    if 10 <= int(capacity) <= 32:
        return (128, 128, 128, 128)
    tile_n = 256 if intermediate % 256 == 0 and hidden % 256 == 0 else 128
    return (64, tile_n, 64, tile_n)


def empty_tier_plane_elements(bits: int, hidden: int = HIDDEN,
                              intermediate: int = INTERMEDIATE) -> int:
    """int32 elements of one unreachable tier plane (bind's non-null ABI)."""
    return (intermediate // 16) * (hidden // 16) * (8 * int(bits))


def checkpoint_source_shapes(bits: int, hidden: int = HIDDEN,
                             full_intermediate: int = FULL_INTERMEDIATE):
    """Unsliced per-expert checkpoint trellis shapes for one tier."""
    return {
        "w1": (hidden // 16, full_intermediate // 16, 16 * bits),
        "w3": (hidden // 16, full_intermediate // 16, 16 * bits),
        "w2": (full_intermediate // 16, hidden // 16, 16 * bits),
    }


def checkpoint_rank_shapes(bits: int, hidden: int = HIDDEN,
                           intermediate: int = INTERMEDIATE):
    """Rank-slice per-expert trellis shapes (w1/w3 N axis, w2 K axis)."""
    return {
        "w1": (hidden // 16, intermediate // 16, 16 * bits),
        "w3": (hidden // 16, intermediate // 16, 16 * bits),
        "w2": (intermediate // 16, hidden // 16, 16 * bits),
    }


def output_check_shapes(row: int, capacity: int, tail: int = TAIL_ROWS):
    """Output views exercised per fixture: capacity-shaped and live+tail."""
    if not 1 <= int(row) <= int(capacity):
        raise ValueError(f"row {row} outside capacity {capacity}")
    return [("capacity", (int(capacity), HIDDEN)),
            ("live-parent", (int(row) + int(tail), HIDDEN))]


def normalize_capacities(entries=None):
    """Validate one arm's installed decode capacities.

    The engine's row policy picks the smallest installed capacity >= rows.
    Candidates may add capacities but must keep both engine exports (m1 for
    row 1, m80 for the 2..80 tail) so the comparison still describes the
    engine's decode path; each added capacity is disclosed as an extra export
    and priced by its workspace. The shipped decode schedule is fixed (the
    engine's kimi export uses the pure evict-first schedule at both
    capacities), so all arms are timed on the same schedule.
    """
    if entries is None:
        entries = DEFAULT_CAPACITIES
    normalized = []
    seen = set()
    for entry in entries:
        if isinstance(entry, int):
            entry = {"capacity": entry}
        capacity = int(entry["capacity"])
        if not 1 <= capacity <= CAPACITY_ROWS:
            raise ValueError(f"decode capacity must be 1..{CAPACITY_ROWS}, got {capacity}")
        if capacity in seen:
            raise ValueError(f"duplicate decode capacity {capacity}")
        seen.add(capacity)
        if entry.get("tile") is not None:
            tile = tuple(int(value) for value in entry["tile"])
            if len(tile) != 4 or any(value <= 0 for value in tile):
                raise ValueError(f"capacity {capacity}: tile must be four positive ints")
            tile_source = "explicit"
        else:
            tile = planner_tile(capacity)
            tile_source = "planner"
        blocks_per_sm = int(entry.get("blocks_per_sm", BLOCKS_PER_SM))
        if blocks_per_sm < 1:
            raise ValueError(f"capacity {capacity}: blocks_per_sm must be positive")
        force_blocks_per_sm = entry.get("force_blocks_per_sm")
        if force_blocks_per_sm is not None:
            force_blocks_per_sm = int(force_blocks_per_sm)
            if force_blocks_per_sm != blocks_per_sm:
                raise ValueError(
                    f"capacity {capacity}: force_blocks_per_sm must agree with "
                    "blocks_per_sm")
        schedule = entry.get("decode_schedule", DECODE_SCHEDULE)
        if schedule in ("gb10", DECODE_SCHEDULE):
            schedule = DECODE_SCHEDULE
        else:
            raise ValueError(
                f"capacity {capacity}: the engine's kimi decode export runs "
                f"{DECODE_SCHEDULE!r}; other schedules are not comparable")
        normalized.append({
            "capacity": capacity,
            "tile": tile,
            "tile_source": tile_source,
            "blocks_per_sm": blocks_per_sm,
            "force_blocks_per_sm": force_blocks_per_sm,
            "decode_schedule": schedule,
            "package_policy": (
                "excluded: package_exl3_aot rejects kimik3 m16"
                if capacity == 16 else "allowed"),
        })
    normalized.sort(key=lambda item: item["capacity"])
    if CAPACITY_M1 not in seen:
        raise ValueError("installed capacities must keep the m1 export for row 1")
    if CAPACITY_ROWS not in seen:
        raise ValueError("installed capacities must keep the m80 export for rows 2..80")
    return normalized


def capacity_for_rows(rows: int, capacities) -> int:
    """The engine row policy: smallest installed capacity that fits."""
    rows = int(rows)
    if rows < 1:
        raise ValueError(f"decode rows must be positive, got {rows}")
    eligible = sorted(
        int(entry["capacity"]) if isinstance(entry, dict) else int(entry)
        for entry in capacities)
    for capacity in eligible:
        if capacity >= rows:
            return capacity
    raise ValueError(f"no installed capacity fits {rows} rows")


def capacity_disclosure(capacities):
    """Extra exports a capacity list adds over the engine's m1/m80 pair."""
    installed = [int(entry["capacity"]) if isinstance(entry, dict) else int(entry)
                 for entry in capacities]
    extra = [capacity for capacity in sorted(installed)
             if capacity not in (CAPACITY_M1, CAPACITY_ROWS)]
    return {"installed": sorted(installed),
            "extra_exports": len(extra),
            "extra_capacities": extra,
            "planner_workspace": "each installed capacity owns its own workspace; "
                                 "the planner pays the sum"}


def capacity_entry(capacities, capacity):
    for entry in capacities:
        if isinstance(entry, dict) and int(entry["capacity"]) == int(capacity):
            return entry
    if isinstance(capacities, (list, tuple)) and capacities and isinstance(capacities[0], int):
        return {"capacity": int(capacity)}
    raise KeyError(f"capacity {capacity} not installed")


def engine_scalars(capacity: int, entry=None):
    """Runtime scalars the engine's AOT executor passes at this capacity.

    tier1 keeps its 896-expert compile count but zero runtime projections: the
    shipped checkpoint has no K3 routed experts, and the engine passes the
    checkpoint's own membership (tier0 896/896/896, tier1 0/0/0).
    """
    blocks_per_sm = int((entry or {}).get("blocks_per_sm", BLOCKS_PER_SM))
    return {
        "tier0_num_experts": EXPERTS,
        "tier1_num_experts": EXPERTS,
        "tier0_fc2": EXPERTS,
        "tier1_fc2": 0,
        "tier0_gate": EXPERTS,
        "tier1_gate": 0,
        "tier0_up": EXPERTS,
        "tier1_up": 0,
        "route_num_experts": EXPERTS,
        "grid_x": SMS * blocks_per_sm,
    }


def compile_options(capacity: int, entry=None, sms: int = SMS,
                    max_shared_mem=None):
    """compile_mixed_trellis kwargs matching the pin's kimi AOT export.

    The exporter compiles with ``sms=EXL3_AOT_MAX_SMS[(12, 1)]`` (=48) and the
    same ``workspace_sms``, leaving residency to the cooperative planner
    (``force_blocks_per_sm=None``); the harness reproduces all three.
    """
    entry = entry or {}
    _, slots, blocks = route_slots_for_capacity(capacity)
    if entry.get("tile") is not None:
        tile = tuple(int(value) for value in entry["tile"])
    else:
        tile = planner_tile(capacity)
    return {
        "size_m": int(capacity),
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "tier0_num_experts": EXPERTS,
        "tier1_num_experts": EXPERTS,
        "top_k": TOPK,
        "route_num_experts": EXPERTS,
        "max_m_blocks": blocks,
        "sms": int(sms),
        "workspace_sms": int(WORKSPACE_SMS),
        "max_shared_mem": max_shared_mem,
        "force_tile_config": tile,
        "tier0_bits": TIER_BITS[0],
        "tier1_bits": TIER_BITS[1],
        "trellis_codebook": "mcg",
        "swiglu_limit": SWIGLU_LIMIT,
        "activation": ACTIVATION,
        "moe_block_size": ROUTE_BLOCK,
        "rotation_input_dtype": "bf16",
        "full_rotation_output_dtype": "bf16",
        "route_ids_dtype": "torch.int32",
        "direct_topk_routes": False,
        "force_blocks_per_sm": entry.get("force_blocks_per_sm"),
        "decode_schedule": entry.get("decode_schedule", DECODE_SCHEDULE),
        "route_slots": slots,
    }


def timing_weights(rows=TIMING_ROWS, shares=TIMING_SHARES_PCT):
    total = sum(float(shares[row]) for row in rows)
    return {int(row): float(shares[row]) / total for row in rows}


def unique_storage_bytes(tensors) -> int:
    """Bytes of distinct device storage, merging aliased ranges.

    fc2 aliases rotation_gate by design (FC2 writes the buffer FC1's rotation
    output vacates after the activation barrier), so summing fields would
    double-count it.
    """
    ranges = []
    for tensor in tensors:
        span = int(tensor.numel()) * int(tensor.element_size())
        if span:
            ranges.append((int(tensor.data_ptr()), int(tensor.data_ptr()) + span))
    total, cursor = 0, None
    for begin, end in sorted(ranges):
        if cursor is None or begin > cursor:
            total += end - begin
            cursor = end
        elif end > cursor:
            total += end - cursor
            cursor = end
    return total


# ---------------------------------------------------------------------------
# Fixtures


def fixture_layer_seed(layer: int) -> int:
    return 0x4B330000 + int(layer) * 0x100


def _normalized_weights(weights):
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def build_layer_fixtures(layer: int, curve=None, rows=EXACTNESS_ROWS):
    """Deterministic per-row routes and inputs for one fixed layer.

    Each row r is an independent fixture: exactly curve[r] distinct experts,
    16 distinct experts per row (a router never repeats an expert), and a
    shared input tensor whose prefix x[:r] is the row-r input.
    """
    curve = curve or derive_distinct_curve()
    rows = [int(row) for row in rows]
    gen_x = torch.Generator(device="cpu").manual_seed(fixture_layer_seed(layer) ^ 0x58)
    x = (torch.randn((max(rows), HIDDEN), generator=gen_x, dtype=torch.float32) * 0.25
         ).to(torch.bfloat16)
    gen_perm = torch.Generator(device="cpu").manual_seed(fixture_layer_seed(layer) ^ 0xA1)
    perm = torch.randperm(EXPERTS, generator=gen_perm)
    gen_w = torch.Generator(device="cpu").manual_seed(fixture_layer_seed(layer) ^ 0x77)
    fixtures = {}
    for row in rows:
        distinct = int(curve[row])
        if distinct < TOPK or distinct > min(row * TOPK, EXPERTS):
            raise ValueError(f"row {row}: {distinct} distinct experts is not routable")
        active = perm[:distinct]
        sequence = torch.arange(row * TOPK) % distinct
        ids = active[sequence].view(row, TOPK).to(torch.int32)
        weights = _normalized_weights(
            torch.rand((row, TOPK), generator=gen_w, dtype=torch.float32) * 0.8 + 0.2)
        order = torch.argsort(weights, dim=1, descending=True, stable=True)
        ids = torch.gather(ids, 1, order).contiguous()
        weights = torch.gather(weights, 1, order).contiguous()
        fixtures[row] = {"ids": ids, "weights": weights,
                         "distinct": distinct, "active": active.clone()}
    return {"x": x, "rows": fixtures}


def _hash_tensor(digest, name, tensor):
    tensor = tensor.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    digest.update(b"\0")


def fixtures_sha256(layer_data, curve, exactness_rows, timing_rows):
    """Content hash over every generated fixture and the trace fit."""
    digest = hashlib.sha256()
    meta = {
        "schema": "cuteafd.kimi-k3-trellis-fixtures.v1",
        "geometry": {
            "hidden": HIDDEN, "intermediate": INTERMEDIATE, "experts": EXPERTS,
            "top_k": TOPK, "tier_bits": list(TIER_BITS),
        },
        "exactness_rows": [int(row) for row in exactness_rows],
        "timing_rows": [int(row) for row in timing_rows],
        "timing_shares_pct": {str(k): v for k, v in sorted(TIMING_SHARES_PCT.items())},
        "distinct": {str(k): int(v) for k, v in sorted(curve.items())},
    }
    digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    for layer in sorted(layer_data):
        data = layer_data[layer]
        _hash_tensor(digest, f"l{layer}/x", data["x"])
        for row in sorted(data["rows"]):
            _hash_tensor(digest, f"l{layer}/r{row}/ids", data["rows"][row]["ids"])
            _hash_tensor(digest, f"l{layer}/r{row}/weights", data["rows"][row]["weights"])
    return digest.hexdigest()


def build_fixture_manifest(layers=LAYERS, curve=None):
    curve = curve or derive_distinct_curve()
    layer_data = {int(layer): build_layer_fixtures(layer, curve) for layer in layers}
    return {
        "schema": "cuteafd.kimi-k3-trellis-fixtures.v1",
        "pin": {"fork_commit": PIN_COMMIT, "blobs": dict(FROZEN_BLOBS)},
        "geometry": {
            "hidden": HIDDEN, "intermediate": INTERMEDIATE,
            "full_intermediate": FULL_INTERMEDIATE, "experts": EXPERTS,
            "top_k": TOPK, "tier_bits": list(TIER_BITS),
            "route_block": ROUTE_BLOCK, "tile": list(TILE), "sms": SMS,
            "activation": ACTIVATION, "swiglu_limit": SWIGLU_LIMIT,
            "decode_schedule": DECODE_SCHEDULE,
            "expert_bytes": EXPERT_BYTES,
            "nominal_bandwidth_gb_s": NOMINAL_BANDWIDTH_GB_S,
        },
        "trace": dict(TRACE_META, rows=[
            {"rows": row, "launches": n, "ms_per_launch": ms}
            for row, n, ms in TRACE_ROWS]),
        "distinct": {
            "model": "uniform-bandwidth lower bound, weighted isotonic over the "
                     "trace's measured ms/launch",
            "per_expert_ms": per_expert_ms(),
            "values": {str(k): int(v) for k, v in sorted(curve.items())},
        },
        "layers": [int(layer) for layer in layers],
        "exactness_rows": list(EXACTNESS_ROWS),
        "timing_rows": list(TIMING_ROWS),
        "timing_shares_pct": {str(k): v for k, v in sorted(TIMING_SHARES_PCT.items())},
        "fixtures_sha256": fixtures_sha256(
            layer_data, curve, EXACTNESS_ROWS, TIMING_ROWS),
        "_layer_data": layer_data,
    }


def verify_fixture_manifest(manifest):
    """Recompute every fixture from the frozen trace table; refuse drift."""
    if manifest.get("schema") != "cuteafd.kimi-k3-trellis-fixtures.v1":
        raise ValueError("not a Kimi K3 Trellis fixture manifest")
    if manifest.get("pin", {}).get("fork_commit") != PIN_COMMIT:
        raise ValueError("fixture manifest was built for a different fork pin")
    if manifest["pin"]["blobs"] != FROZEN_BLOBS:
        raise ValueError("fixture manifest kernel hashes differ from this harness")
    rows = [(entry["rows"], entry["launches"], entry["ms_per_launch"])
            for entry in manifest["trace"]["rows"]]
    if tuple(rows) != TRACE_ROWS:
        raise ValueError("fixture manifest trace table differs from the saved trace")
    curve = derive_distinct_curve()
    recorded = {int(k): int(v) for k, v in manifest["distinct"]["values"].items()}
    if recorded != curve:
        raise ValueError("fixture manifest distinct-expert curve was edited")
    layers = tuple(int(layer) for layer in manifest["layers"])
    rebuilt = {layer: build_layer_fixtures(layer, curve) for layer in layers}
    digest = fixtures_sha256(rebuilt, curve, EXACTNESS_ROWS, TIMING_ROWS)
    if digest != manifest["fixtures_sha256"]:
        raise ValueError("fixtures do not match the manifest hash")
    return rebuilt, curve


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hashes(source: Path) -> dict:
    result = {}
    for relative, expected in FROZEN_BLOBS.items():
        path = Path(source) / relative
        result[relative] = sha256_file(path) if path.is_file() else None
    return result


def git_provenance(source: Path):
    source = Path(source)
    try:
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return {"commit": head.stdout.strip(), "dirty_entries": len(dirty)}


# ---------------------------------------------------------------------------
# Oracle freeze and checkpoint staging


def freeze_oracle(fork: Path, out: Path, harness: Path, test: Path):
    """Archive the pin's tree plus the frozen harness; record every hash."""
    out = Path(out)
    tree = out / "oracle-tree"
    if tree.exists():
        shutil.rmtree(tree)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "oracle-tree.tar"
    completed = subprocess.run(
        ["git", "-C", str(fork), "archive", "--format=tar", PIN_COMMIT],
        stdout=open(archive, "wb"), stderr=subprocess.PIPE, timeout=600)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
    tree.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(tree)
    for name in FROZEN_BLOBS:
        if not (tree / name).is_file():
            raise FileNotFoundError(f"pin archive lacks {name}")
    hashes = file_hashes(tree)
    if hashes != FROZEN_BLOBS:
        raise ValueError(f"pin archive kernel hashes differ: {hashes}")
    (tree / "benchmarks").mkdir(exist_ok=True)
    shutil.copy2(harness, tree / "benchmarks" / Path(harness).name)
    (tree / "tests").mkdir(exist_ok=True)
    shutil.copy2(test, tree / "tests" / Path(test).name)
    record = {
        "schema": "cuteafd.kimi-k3-trellis-oracle.v1",
        "fork_commit": PIN_COMMIT,
        "blobs": hashes,
        "archive_sha256": sha256_file(archive),
        "harness": {"file": f"benchmarks/{Path(harness).name}",
                    "sha256": sha256_file(harness)},
        "tests": {"file": f"tests/{Path(test).name}", "sha256": sha256_file(test)},
        "frozen_unix_ns": time.time_ns(),
    }
    write_json(out / "ORACLE.json", record)
    return record


def stage_checkpoint(checkpoint: Path, out: Path, layers=LAYERS,
                     width: int = INTERMEDIATE, start: int = 0,
                     full_intermediate: int = FULL_INTERMEDIATE,
                     repo_id: str = "wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6",
                     revision: str | None = None):
    """Extract real rank-slice expert weights into one file per layer."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"not an HF checkpoint: {index_path}")
    revision = revision or checkpoint.name
    index = load_json(index_path)["weight_map"]
    if not 1 <= width <= full_intermediate or width % 128 or start % 128:
        raise ValueError("rank width must be a positive multiple of 128 with 128-aligned start")
    if start + width > full_intermediate:
        raise ValueError("rank slice exceeds the checkpoint intermediate size")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "cuteafd.kimi-k3-trellis-stage.v1",
        "checkpoint": {"repo_id": repo_id, "revision": revision,
                       "path": str(checkpoint.resolve()),
                       "index_sha256": sha256_file(index_path)},
        "slice": {"rank_width": int(width), "start": int(start),
                  "full_intermediate": int(full_intermediate),
                  "hidden": HIDDEN},
        "experts": EXPERTS,
        "layers": {},
    }
    expected_full = checkpoint_source_shapes(TIER_BITS[0], HIDDEN, full_intermediate)
    expected_rank = checkpoint_rank_shapes(TIER_BITS[0], HIDDEN, width)
    for layer in layers:
        layer = int(layer)
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
        keys = {expert: {} for expert in range(EXPERTS)}
        shards = set()
        for expert in range(EXPERTS):
            for projection in ("w1", "w3", "w2"):
                for suffix in ("trellis", "suh", "svh", "mcg"):
                    name = f"{prefix}.{expert}.{projection}.{suffix}"
                    if name not in index:
                        raise KeyError(f"missing checkpoint tensor {name}")
                    keys[expert][(projection, suffix)] = name
                    shards.add(index[name])
        handles = {}
        try:
            for shard in sorted(shards):
                handles[shard] = safe_open(str(checkpoint / shard), framework="pt",
                                           device="cpu")
            w13 = torch.empty((2 * EXPERTS, *expected_rank["w1"]), dtype=torch.int16)
            w2 = torch.empty((EXPERTS, *expected_rank["w2"]), dtype=torch.int16)
            gate_suh = torch.empty((EXPERTS, HIDDEN), dtype=torch.float16)
            up_suh = torch.empty((EXPERTS, HIDDEN), dtype=torch.float16)
            intermediate = torch.empty((EXPERTS, 3 * width), dtype=torch.float16)
            down_svh = torch.empty((EXPERTS, HIDDEN), dtype=torch.float16)
            bits_seen = set()
            mcg_checked = 0
            for expert in range(EXPERTS):
                for projection in ("w1", "w3", "w2"):
                    names = keys[expert]

                    def tensor(suffix, projection=projection):
                        name = names[(projection, suffix)]
                        return handles[index[name]].get_tensor(name)

                    trellis = tensor("trellis")
                    bits = int(trellis.shape[-1]) // 16
                    bits_seen.add(bits)
                    if tuple(trellis.shape) != expected_full[projection]:
                        raise ValueError(
                            f"layer {layer} expert {expert} {projection}: shape "
                            f"{tuple(trellis.shape)} != {expected_full[projection]}")
                    if projection == "w2":
                        rank = trellis[start // 16:(start + width) // 16].contiguous()
                        if tuple(rank.shape) != expected_rank["w2"]:
                            raise ValueError("w2 rank slice has the wrong shape")
                        if tuple(tensor("suh").shape) != (full_intermediate,):
                            raise ValueError("w2.suh has the wrong shape")
                        if tuple(tensor("svh").shape) != (HIDDEN,):
                            raise ValueError("w2.svh has the wrong shape")
                        w2[expert].copy_(rank)
                        intermediate[expert, 2 * width:].copy_(
                            tensor("suh")[start:start + width].to(torch.float16))
                        down_svh[expert].copy_(tensor("svh").to(torch.float16))
                    else:
                        rank = trellis[:, start // 16:(start + width) // 16].contiguous()
                        if tuple(rank.shape) != expected_rank[projection]:
                            raise ValueError(f"{projection} rank slice has the wrong shape")
                        if tuple(tensor("suh").shape) != (HIDDEN,):
                            raise ValueError(f"{projection}.suh has the wrong shape")
                        if tuple(tensor("svh").shape) != (full_intermediate,):
                            raise ValueError(f"{projection}.svh has the wrong shape")
                        target = w13[expert] if projection == "w1" else w13[EXPERTS + expert]
                        target.copy_(rank)
                        (gate_suh if projection == "w1" else up_suh)[expert].copy_(
                            tensor("suh").to(torch.float16))
                        offset = 0 if projection == "w1" else width
                        intermediate[expert, offset:offset + width].copy_(
                            tensor("svh")[start:start + width].to(torch.float16))
                    magic = int(tensor("mcg").item()) & 0xFFFFFFFF
                    if magic != 0xCBAC1FED:
                        raise ValueError(
                            f"layer {layer} expert {expert} {projection}: not MCG "
                            f"(magic {magic:#x})")
                    mcg_checked += 1
        finally:
            handles.clear()
        if bits_seen != {TIER_BITS[0]}:
            raise ValueError(
                f"layer {layer}: routed experts must all be K{TIER_BITS[0]}, "
                f"got tiers {sorted(bits_seen)}")
        name = f"layer{layer}.safetensors"
        path = out / name
        header = {
            "layer": str(layer), "rank_width": str(width), "rank_start": str(start),
            "checkpoint_repo_id": repo_id, "checkpoint_revision": revision,
            "expert_bits": str(TIER_BITS[0]), "experts": str(EXPERTS),
        }
        save_file({"w13": w13, "w2": w2, "gate_suh": gate_suh, "up_suh": up_suh,
                   "intermediate_rotations": intermediate, "down_svh": down_svh},
                  str(path), metadata=header)
        manifest["layers"][str(layer)] = {
            "file": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "bits": TIER_BITS[0],
            "experts": EXPERTS,
            "projections": ["w1", "w3", "w2"],
            "mcg_checked": mcg_checked,
            "tensors": {
                "w13": {"shape": list(w13.shape), "dtype": "torch.int16"},
                "w2": {"shape": list(w2.shape), "dtype": "torch.int16"},
                "gate_suh": {"shape": list(gate_suh.shape), "dtype": "torch.float16"},
                "up_suh": {"shape": list(up_suh.shape), "dtype": "torch.float16"},
                "intermediate_rotations": {
                    "shape": list(intermediate.shape), "dtype": "torch.float16"},
                "down_svh": {"shape": list(down_svh.shape), "dtype": "torch.float16"},
            },
        }
        del w13, w2, gate_suh, up_suh, intermediate, down_svh
    write_json(out / "manifest.json", manifest)
    return manifest


def verify_stage_manifest(stage: Path, manifest) -> dict:
    """Verify staged files before allocating device memory."""
    stage = Path(stage)
    if manifest.get("schema") != "cuteafd.kimi-k3-trellis-stage.v1":
        raise ValueError("not a Kimi K3 Trellis stage manifest")
    if int(manifest["experts"]) != EXPERTS:
        raise ValueError("stage manifest expert count differs from the harness")
    for layer, info in manifest["layers"].items():
        path = stage / info["file"]
        if not path.is_file():
            raise FileNotFoundError(f"staged weights missing: {path}")
        if path.stat().st_size != int(info["bytes"]):
            raise ValueError(f"staged weights size changed: {path}")
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"staged weights hash changed: {path}")
    return manifest


# ---------------------------------------------------------------------------
# Device-side arm


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_layer(stage: Path, manifest, layer: int, device: torch.device):
    """Load one staged layer and run the engine's native Trellis preparation."""
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import (
        ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights)

    info = manifest["layers"][str(int(layer))]
    with safe_open(str(Path(stage) / info["file"]), framework="pt",
                   device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).to(device, non_blocking=False)
                   for name in ("w13", "w2", "gate_suh", "up_suh",
                                "intermediate_rotations", "down_svh")}
    _require(tuple(tensors["w13"].shape) ==
             (2 * EXPERTS, HIDDEN // 16, INTERMEDIATE // 16, 16 * TIER_BITS[0]),
             "staged w13 shape is not the engine's K2 rank layout")
    _require(tuple(tensors["w2"].shape) ==
             (EXPERTS, INTERMEDIATE // 16, HIDDEN // 16, 16 * TIER_BITS[0]),
             "staged w2 shape is not the engine's K2 rank layout")
    members = tuple(range(EXPERTS))
    tier0 = ProjectionTrellisTierWeights(
        TIER_BITS[0], tensors["w13"], tensors["w2"], members, members, members)
    empty_bits = TIER_BITS[1]
    empty_w13 = torch.empty((0, HIDDEN // 16, INTERMEDIATE // 16,
                             16 * empty_bits), dtype=torch.int16, device=device)
    empty_w2 = torch.empty((0, INTERMEDIATE // 16, HIDDEN // 16,
                            16 * empty_bits), dtype=torch.int16, device=device)
    tier1 = ProjectionTrellisTierWeights(
        empty_bits, empty_w13, empty_w2, (), (), ())
    prepared = prepare_projection_native_trellis_weights(
        (tier0, tier1),
        gate_suh=tensors["gate_suh"],
        up_suh=tensors["up_suh"],
        intermediate_rotations=tensors["intermediate_rotations"],
        down_svh=tensors["down_svh"],
        activation=ACTIVATION,
        params_dtype=torch.bfloat16,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )
    # An empty projection membership is legal; the native binding only needs
    # one unreachable physical plane for its non-null storage ABI (the same
    # padding cuteafd's own V4.1 loader applies).
    padded = []
    for bits, tier in zip(TIER_BITS, prepared.tiers):
        updates = {}
        if tier.w13.numel() == 0:
            updates["w13"] = torch.zeros(
                empty_tier_plane_elements(bits), dtype=torch.int32, device=device)
        if tier.w2.numel() == 0:
            updates["w2"] = torch.zeros(
                empty_tier_plane_elements(bits), dtype=torch.int32, device=device)
            updates["w2_global_scale"] = torch.ones(
                1, dtype=torch.float32, device=device)
        padded.append(replace(tier, **updates))
    prepared = replace(prepared, tiers=tuple(padded))
    del tensors
    torch.cuda.empty_cache()
    return prepared


def compile_capacity(mt, capacity: int, entry, device: torch.device):
    props = torch.cuda.get_device_properties(device)
    if int(props.multi_processor_count) != SMS:
        raise ValueError(
            f"device reports {props.multi_processor_count} SMs; the frozen "
            f"engine export and trace were built for {SMS}")
    options = compile_options(
        capacity, entry, sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin))
    kwargs = dict(options)
    slots = kwargs.pop("route_slots")
    kwargs["route_ids_dtype"] = torch.int32
    started = time.time()
    launch = mt.compile_mixed_trellis(**kwargs)
    elapsed = time.time() - started
    meta = {
        "capacity": int(capacity),
        "compile_seconds": elapsed,
        "options": {key: (list(value) if isinstance(value, tuple) else value)
                    for key, value in options.items()},
        "route_slots": int(slots),
        "blocks_per_sm": int(launch.blocks_per_sm),
        "sms": int(launch.sms),
        "shared_memory_bytes": int(launch.shared_memory_bytes),
        "tile": [int(launch.fc1_tile_k), int(launch.fc1_tile_n),
                 int(launch.fc2_tile_k), int(launch.fc2_tile_n)],
        "decode_schedule": getattr(launch, "decode_schedule", None),
    }
    expected = compile_options(capacity, entry, sms=SMS)
    if int(launch.size_m) != capacity:
        raise ValueError("compiled launch capacity differs from the request")
    if meta["tile"] != list(expected["force_tile_config"]):
        raise ValueError(f"compiled tile {meta['tile']} != declared "
                         f"{expected['force_tile_config']}")
    if int(launch.max_m_blocks) != int(expected["max_m_blocks"]):
        raise ValueError("compiled route-block capacity differs from the row policy")
    if bool(launch.direct_topk_routes):
        raise ValueError("the engine's kimi decode export uses packed routes")
    if int(launch.blocks_per_sm) != int(entry["blocks_per_sm"]):
        raise ValueError(
            f"compiled residency {launch.blocks_per_sm} differs from the "
            f"declared blocks_per_sm {entry['blocks_per_sm']}")
    if getattr(launch, "decode_schedule", None) != entry["decode_schedule"]:
        raise ValueError(
            f"compiled schedule {getattr(launch, 'decode_schedule', None)!r} != "
            f"declared {entry['decode_schedule']!r}")
    return launch, meta


def make_capacity_buffers(mt, launch, device: torch.device, sms: int):
    """Engine buffers, with the canonical packer's route metadata sizes.

    make_mixed_trellis_buffers sizes route metadata from the exact live
    capacity, while the AOT packer rounds token capacity to a power-of-two
    bucket; the engine manifest publishes the larger canonical allocations.
    """
    buffers = mt.make_mixed_trellis_buffers(launch, device=device, sms=sms)
    _, slots, blocks = route_slots_for_capacity(int(launch.size_m))
    buffers = replace(
        buffers,
        packed_route_indices=torch.empty(slots, dtype=torch.int32, device=device),
        block_expert_ids=torch.empty(blocks, dtype=torch.int32, device=device),
    )
    return buffers


def _pointer_dtype(mt, topk_ids):
    return mt.cutlass.Int32 if topk_ids.dtype == torch.int32 else mt.cutlass.Int64


def launch_engine_sequence(mt, x, topk_weights, topk_ids, binding, buffers,
                           scalars, route_launches):
    """The engine's exact launch sequence, with its own runtime scalars.

    This is run_bound_mixed_trellis with two deliberate differences that match
    the AOT executor rather than the Python convenience binder: the route
    packer uses the capacity's precompiled launch plan (so every row at m80
    packs at the canonical 2048-route bucket), and the tier-1 FC2 count is the
    engine's 0 rather than the binder's dummy-plane count of 1.
    """
    from b12x.moe._shared.kernels.w4a16.route_pack import pack_topk_routes_by_expert

    launch = binding.launch
    m = int(x.shape[0])
    _require(x.ndim == 2 and int(x.shape[1]) == HIDDEN,
             f"input must be [M, {HIDDEN}]")
    _require(1 <= m <= int(launch.size_m),
             f"active rows {m} outside launch capacity {launch.size_m}")
    _require(tuple(topk_ids.shape) == (m, TOPK), "topk_ids shape mismatch")
    _require(tuple(topk_weights.shape) == (m, TOPK), "topk_weights shape mismatch")
    _require(x.dtype == torch.bfloat16 and x.is_contiguous(), "input must be contiguous bf16")
    _require(topk_ids.dtype == torch.int32 and topk_ids.is_contiguous(),
             "topk_ids must be contiguous int32")
    _require(topk_weights.dtype == torch.float32 and topk_weights.is_contiguous(),
             "topk_weights must be contiguous fp32")
    _require(x.device == binding.device and topk_ids.device == binding.device
             and topk_weights.device == binding.device,
             "inputs must live on the binding device")
    _require(tuple(buffers.output.shape) in ((m, HIDDEN), (launch.size_m, HIDDEN)),
             f"output must be [m, {HIDDEN}] or [{launch.size_m}, {HIDDEN}]")
    _require(buffers.output.dtype == torch.bfloat16 and buffers.output.is_contiguous()
             and buffers.output.device == binding.device,
             "output must be contiguous bf16 on the binding device")
    required_slots = m * TOPK + EXPERTS * (ROUTE_BLOCK - 1)
    if m * TOPK < EXPERTS:
        required_slots = min(m * TOPK * ROUTE_BLOCK, required_slots)
    required_blocks = (required_slots + ROUTE_BLOCK - 1) // ROUTE_BLOCK
    _require(required_blocks <= int(launch.max_m_blocks),
             f"request needs {required_blocks} route blocks; launch supports "
             f"{launch.max_m_blocks}")
    _require(buffers.packed_route_indices.numel() >= required_slots
             and buffers.block_expert_ids.numel() >= required_blocks,
             "route workspace is below request capacity")
    packed, block_experts, packed_count = pack_topk_routes_by_expert(
        topk_ids,
        launch.moe_block_size,
        binding.route_num_experts,
        expert_map=binding.global_to_combined,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        launches=route_launches,
    )
    raw_topk_ids = topk_ids.reshape(-1)
    stream = mt.current_cuda_stream()
    launch.compiled(
        mt.make_ptr(mt.cutlass.BFloat16, x.data_ptr(), mt.cute.AddressSpace.gmem,
                    assumed_align=16),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        *binding.tier_pointers[0],
        *binding.tier_pointers[1],
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        raw_topk_ids,
        block_experts,
        packed_count,
        binding.descriptor_map_ptr,
        binding.global_to_combined_ptr,
        mt.make_ptr(mt.cutlass.Float32, topk_weights.data_ptr(),
                    mt.cute.AddressSpace.gmem, assumed_align=4),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        binding.intermediate_rotations_ptr,
        binding.gate_suh_ptr,
        binding.up_suh_ptr,
        mt.make_ptr(mt.cutlass.Uint8, launch.trellis_lut.data_ptr(),
                    mt.cute.AddressSpace.gmem, assumed_align=16),
        mt.Int32(scalars["tier0_num_experts"]),
        mt.Int32(scalars["tier1_num_experts"]),
        mt.Int32(scalars["tier0_fc2"]),
        mt.Int32(scalars["tier1_fc2"]),
        m,
        int(scalars["grid_x"]),
        stream,
        tier0_gate_experts=mt.Int32(scalars["tier0_gate"]),
        tier1_gate_experts=mt.Int32(scalars["tier1_gate"]),
        tier0_up_experts=mt.Int32(scalars["tier0_up"]),
        tier1_up_experts=mt.Int32(scalars["tier1_up"]),
        route_num_experts=mt.Int32(scalars["route_num_experts"]),
    )
    launch.topk_sum.compiled(
        mt.make_ptr(mt.cutlass.Float16, buffers.fc2.data_ptr(),
                    mt.cute.AddressSpace.gmem, assumed_align=16),
        mt.make_ptr(
            mt.cutlass.BFloat16
            if launch.topk_sum.full_rotation_output_dtype == "bf16"
            else mt.cutlass.Float32,
            buffers.output.data_ptr(), mt.cute.AddressSpace.gmem, assumed_align=16),
        mt.make_ptr(mt.cutlass.Float32, topk_weights.data_ptr(),
                    mt.cute.AddressSpace.gmem, assumed_align=4),
        mt.make_ptr(_pointer_dtype(mt, topk_ids), topk_ids.data_ptr(),
                    mt.cute.AddressSpace.gmem,
                    assumed_align=4 if topk_ids.dtype == torch.int32 else 8),
        binding.global_to_combined_ptr,
        binding.down_svh_ptr,
        mt.Int32(launch.topk_sum.num_experts),
        mt.Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


def _digest_live(tensor) -> str:
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def exactness_pass(mt, layer, capacity, fixture, binding, buffers, scalars,
                   route_launches, device):
    """Byte exactness, nonzero output and poisoned tails at one fixture."""
    row = len(fixture["ids"])
    x = fixture["x"].to(device, non_blocking=False)
    ids = fixture["ids"].to(device, non_blocking=False)
    weights = fixture["weights"].to(device, non_blocking=False)
    results = {}

    def run(mode):
        if mode == "capacity":
            output = torch.full((capacity, HIDDEN), float("nan"),
                                dtype=torch.bfloat16, device=device)
        else:
            parent = torch.full((row + TAIL_ROWS, HIDDEN), float("nan"),
                                dtype=torch.bfloat16, device=device)
            output = parent[:row]
        buffers.output = output
        launch_engine_sequence(mt, x, weights, ids, binding, buffers,
                               scalars, route_launches)
        torch.cuda.synchronize(device)
        live = output[:row]
        tail = (output[row:] if mode == "capacity" else parent[row:])
        return {
            "digest": _digest_live(live),
            "tail_ok": bool(torch.isnan(tail).all()) if tail.numel() else True,
            "finite": bool(torch.isfinite(live).all()),
            "nonzero": bool((live != 0).any()),
        }

    clean = run("capacity")
    live_parent = run("live-parent")
    repeat = run("capacity")
    # Poison every temporary the launch writes before reusing it; stale-state
    # leaks show up as a different (or non-finite) live digest.
    for name in ("rotation_gate", "rotation_up", "fc1", "activated"):
        getattr(buffers, name).fill_(float("nan"))
    scratch = run("capacity")
    buffers.output = torch.empty((capacity, HIDDEN), dtype=torch.bfloat16,
                                 device=device)
    results.update({
        "layer": int(layer), "row": int(row), "capacity": int(capacity),
        "distinct": int(fixture["distinct"]),
        "digest": clean["digest"],
        "digest_live_parent": live_parent["digest"],
        "digest_repeat": repeat["digest"],
        "digest_scratch_poison": scratch["digest"],
        "scratch_sensitive": scratch["digest"] != clean["digest"],
        "tail_capacity_ok": clean["tail_ok"] and repeat["tail_ok"],
        "tail_live_ok": live_parent["tail_ok"],
        "live_finite": clean["finite"],
        "nonzero": clean["nonzero"],
    })
    results["ok"] = bool(
        clean["digest"] == live_parent["digest"] == repeat["digest"]
        and results["tail_capacity_ok"] and results["tail_live_ok"]
        and clean["finite"] and clean["nonzero"])
    results["scratch_finite"] = scratch["finite"]
    return results


def _timer_events(device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def _timed_replay(run, flush, start, end, replays, out):
    for _ in range(int(replays)):
        flush.zero_()
        start.record()
        run()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end))


class ArmRuntime:
    """One loaded arm: weights, launches, route plans, graphs, exactness."""

    def __init__(self, args):
        self.args = args
        self.source = Path(args.source)
        self.torch_device = torch.device(args.device)
        torch.cuda.set_device(self.torch_device)
        props = torch.cuda.get_device_properties(self.torch_device)
        self.props = props
        self.manifest = load_json(Path(args.fixtures))
        self.layer_data, self.curve = verify_fixture_manifest(self.manifest)
        self.stage_manifest = verify_stage_manifest(
            Path(args.stage), load_json(Path(args.stage) / "manifest.json"))
        self.capacities = normalize_capacities(
            json.loads(args.capacities) if args.capacities else None)
        self.mt = _import_kernel(self.source)
        self.route_launches = {}
        for entry in self.capacities:
            capacity = int(entry["capacity"])
            started = time.time()
            self.route_launches[capacity] = _compile_route_plan(capacity)
            entry["route_plan_seconds"] = time.time() - started
        self.launches = {}
        self.launch_meta = {}
        for entry in self.capacities:
            launch, meta = compile_capacity(
                self.mt, int(entry["capacity"]), entry, self.torch_device)
            self.launches[int(entry["capacity"])] = launch
            self.launch_meta[int(entry["capacity"])] = meta
        self.bindings = {}
        self.prepared = {}
        for layer in self.manifest["layers"]:
            layer = int(layer)
            prepared = prepare_layer(Path(args.stage), self.stage_manifest,
                                     layer, self.torch_device)
            self.prepared[layer] = prepared
            for entry in self.capacities:
                capacity = int(entry["capacity"])
                launch = self.launches[capacity]
                self.bindings[(layer, capacity)] = self.mt.bind_mixed_trellis(
                    *prepared.tiers,
                    prepared.global_to_combined,
                    prepared.descriptor_map,
                    prepared.rotations,
                    launch,
                    gate_experts=prepared.gate_counts,
                    up_experts=prepared.up_counts,
                )
        self.buffer_sets = {}
        for entry in self.capacities:
            capacity = int(entry["capacity"])
            launch = self.launches[capacity]
            for variant in ("a", "b"):
                self.buffer_sets[(capacity, variant)] = make_capacity_buffers(
                    self.mt, launch, self.torch_device, int(props.multi_processor_count))
        self.capacity_meta = self._capacity_meta()
        self.exactness = {}
        self.exactness_failures = []
        self._run_exactness()
        self.digests = {"schema": "cuteafd.kimi-k3-trellis-digests.v1",
                        "fixtures_sha256": self.manifest["fixtures_sha256"],
                        "oracle_commit": PIN_COMMIT,
                        "digests": {f"l{r['layer']}/r{r['row']}": r["digest"]
                                    for r in self.exactness},
                        "capacities": {str(e["capacity"]): e["capacity"]
                                       for e in self.capacities}}
        self.digest_mismatches = self._compare_digests(args.digests)
        self.graphs = {}
        self.graph_notes = []
        self._capture_graphs()
        self.flush = torch.empty(
            max(int(getattr(props, "L2_cache_size", 0)) * 2, 64 << 20),
            dtype=torch.int8, device=self.torch_device)
        self.start_event, self.end_event = _timer_events(self.torch_device)

    # -- startup steps ------------------------------------------------------

    def _capacity_meta(self):
        meta = {}
        for entry in self.capacities:
            capacity = int(entry["capacity"])
            buffers = self.buffer_sets[(capacity, "a")]
            _, slots, blocks = route_slots_for_capacity(capacity)
            names = [field.name for field in fields(buffers)]
            storage = unique_storage_bytes(
                [getattr(buffers, name) for name in names])
            meta[str(capacity)] = {
                "capacity": capacity,
                "tile": list(entry["tile"]),
                "tile_source": entry["tile_source"],
                "blocks_per_sm": int(entry["blocks_per_sm"]),
                "decode_schedule": entry["decode_schedule"],
                "package_policy": entry["package_policy"],
                "route_slots": int(slots),
                "route_blocks": int(blocks),
                "workspace_bytes": int(storage),
                "buffer_bytes": {
                    name: int(getattr(buffers, name).numel()
                              * getattr(buffers, name).element_size())
                    for name in names},
                "route_plan_seconds": entry.get("route_plan_seconds"),
                **self.launch_meta[capacity],
            }
        disclosure = capacity_disclosure(self.capacities)
        total = sum(item["workspace_bytes"] for item in meta.values())
        return {
            "installed": [int(e["capacity"]) for e in self.capacities],
            "entries": meta,
            "total_workspace_bytes": int(total),
            "disclosure": disclosure,
        }

    def _fixture_inputs(self, layer, row):
        data = self.layer_data[int(layer)]
        return (data["x"][:row].to(self.torch_device, non_blocking=False),
                data["rows"][row]["ids"].to(self.torch_device, non_blocking=False),
                data["rows"][row]["weights"].to(self.torch_device, non_blocking=False))

    def _run_exactness(self):
        for layer in self.manifest["layers"]:
            layer = int(layer)
            for row in EXACTNESS_ROWS:
                capacity = capacity_for_rows(row, self.capacities)
                x, ids, weights = self._fixture_inputs(layer, row)
                fixture = {"x": x, "ids": ids, "weights": weights,
                           "distinct": self.curve[row]}
                self.exactness.append(exactness_pass(
                    self.mt, layer, capacity, fixture,
                    self.bindings[(layer, capacity)],
                    self.buffer_sets[(capacity, "a")],
                    engine_scalars(capacity, capacity_entry(self.capacities, capacity)),
                    self.route_launches[capacity], self.torch_device))
                if not self.exactness[-1]["ok"]:
                    self.exactness_failures.append(
                        f"l{layer}/r{row}")

    def _compare_digests(self, path):
        if not path:
            return []
        expected = load_json(Path(path))
        if expected.get("fixtures_sha256") != self.manifest["fixtures_sha256"]:
            raise ValueError("frozen digest file was produced for other fixtures")
        mismatches = []
        for key, digest in expected.get("digests", {}).items():
            actual = self.digests["digests"].get(key)
            if actual != digest:
                mismatches.append({"fixture": key, "expected": digest,
                                   "actual": actual})
        return mismatches

    def _capture_graphs(self):
        for layer in self.manifest["layers"]:
            layer = int(layer)
            for row in TIMING_ROWS:
                capacity = capacity_for_rows(row, self.capacities)
                entry = capacity_entry(self.capacities, capacity)
                binding = self.bindings[(layer, capacity)]
                scalars = engine_scalars(capacity, entry)
                x, ids, weights = self._fixture_inputs(layer, row)
                for variant in ("a", "b"):
                    buffers = self.buffer_sets[(capacity, variant)]
                    buffers.output = torch.empty(
                        (capacity, HIDDEN), dtype=torch.bfloat16,
                        device=self.torch_device)
                    run = lambda: launch_engine_sequence(
                        self.mt, x, weights, ids, binding, buffers, scalars,
                        self.route_launches[capacity])
                    for _ in range(2):
                        run()
                    torch.cuda.synchronize(self.torch_device)
                    graph = None
                    mode = "graph"
                    note = None
                    try:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                    except RuntimeError as error:
                        graph = None
                        mode = "eager"
                        note = str(error)
                    self.graphs[(layer, row, variant)] = {
                        "graph": graph, "mode": mode, "error": note,
                        "capacity": capacity, "x": x, "ids": ids,
                        "weights": weights, "binding": binding,
                        "scalars": scalars,
                    }
                    if mode == "eager":
                        self.graph_notes.append(
                            f"l{layer}/r{row}/{variant}: capture failed: {note}")
        # A captured graph must reproduce the exactness digest, and its
        # capacity-shaped output tail must stay poisoned.
        self.graph_digest_failures = []
        for layer in self.manifest["layers"]:
            layer = int(layer)
            for row in TIMING_ROWS:
                capacity = capacity_for_rows(row, self.capacities)
                fixture = self._digest_expectation(layer, row)
                for variant in ("a", "b"):
                    item = self.graphs[(layer, row, variant)]
                    buffers = self.buffer_sets[(capacity, variant)]
                    buffers.output.fill_(float("nan"))
                    if item["graph"] is not None:
                        item["graph"].replay()
                    else:
                        launch_engine_sequence(
                            self.mt, item["x"], item["weights"], item["ids"],
                            item["binding"], buffers, item["scalars"],
                            self.route_launches[capacity])
                    torch.cuda.synchronize(self.torch_device)
                    live = buffers.output[:row]
                    tail = buffers.output[row:]
                    ok = (_digest_live(live) == fixture
                          and (not tail.numel() or bool(torch.isnan(tail).all())))
                    item["graph_digest_ok"] = ok
                    if not ok:
                        self.graph_digest_failures.append(
                            f"l{layer}/r{row}/{variant}")

    def _digest_expectation(self, layer, row):
        for record in self.exactness:
            if record["layer"] == int(layer) and record["row"] == int(row):
                return record["digest"]
        raise KeyError((layer, row))

    # -- timing -------------------------------------------------------------

    def time_round(self, round_index, replays):
        order = list(TIMING_ROWS)
        shift = round_index % len(order)
        order = order[shift:] + order[:shift]
        fixtures = []
        for layer in self.manifest["layers"]:
            layer = int(layer)
            for row in order:
                capacity = capacity_for_rows(row, self.capacities)
                item_a = self.graphs[(layer, row, "a")]
                item_b = self.graphs[(layer, row, "b")]
                samples_a, samples_b = [], []
                for variant, item, sink in (
                        ("a", item_a, samples_a), ("b", item_b, samples_b)):
                    buffers = self.buffer_sets[(capacity, variant)]
                    capacity_entry_ = capacity_entry(self.capacities, capacity)
                    if item["graph"] is not None:
                        run = item["graph"].replay
                    else:
                        run = (lambda item=item, buffers=buffers: launch_engine_sequence(
                            self.mt, item["x"], item["weights"], item["ids"],
                            item["binding"], buffers, item["scalars"],
                            self.route_launches[capacity]))
                    _timed_replay(run, self.flush, self.start_event,
                                  self.end_event, replays, sink)
                fixtures.append({
                    "id": f"l{layer}/r{row}", "layer": layer, "row": row,
                    "capacity": capacity, "a": samples_a, "b": samples_b,
                    "mode": item_a["mode"],
                })
        return {"round": int(round_index), "replays": int(replays),
                "fixtures": fixtures}

    def provenance(self):
        harness = Path(__file__).resolve()
        return {
            "harness_sha256": sha256_file(harness),
            "source": str(self.source),
            "git": git_provenance(self.source),
            "file_hashes": file_hashes(self.source),
            "frozen_pin_match": file_hashes(self.source) == FROZEN_BLOBS,
            "fixtures": str(Path(self.args.fixtures).resolve()),
            "fixtures_sha256": self.manifest["fixtures_sha256"],
            "stage": str(Path(self.args.stage).resolve()),
            "stage_manifest_sha256": sha256_file(Path(self.args.stage) / "manifest.json"),
            "digests_file": (str(Path(self.args.digests).resolve())
                             if self.args.digests else None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": {"name": self.props.name,
                       "capability": [self.props.major, self.props.minor],
                       "sms": int(self.props.multi_processor_count),
                       "shared_memory_per_block_optin":
                           int(self.props.shared_memory_per_block_optin)},
        }

    def ready_payload(self):
        return {
            "event": "ready",
            "provenance": self.provenance(),
            "capacities": self.capacity_meta,
            "exactness": self.exactness,
            "exactness_ok": not self.exactness_failures,
            "exactness_failures": self.exactness_failures,
            "digest_mismatches": self.digest_mismatches,
            "graphs": {f"l{layer}/r{row}/{variant}": {
                "mode": item["mode"], "error": item["error"],
                "graph_digest_ok": item["graph_digest_ok"],
            } for (layer, row, variant), item in self.graphs.items()},
            "graph_digest_failures": self.graph_digest_failures,
            "graph_notes": self.graph_notes,
        }


def _import_kernel(source: Path):
    source = str(Path(source).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis
    return mixed_trellis


def _compile_route_plan(capacity: int):
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches)
    return compile_w4a16_route_pack_launches(
        tokens=int(capacity), topk=TOPK, block_size=ROUTE_BLOCK,
        num_experts=EXPERTS, ordinal=torch.cuda.current_device())


# ---------------------------------------------------------------------------
# Single-arm run (baseline)


def collect_samples(rounds, arm):
    samples = {}
    for entry in rounds:
        for fixture in entry["arms"][arm]["fixtures"]:
            item = samples.setdefault(
                fixture["id"], {"a": [], "b": [], "capacity": fixture["capacity"],
                                "layer": fixture["layer"], "row": fixture["row"]})
            item["a"].extend(fixture["a"])
            item["b"].extend(fixture["b"])
    return samples


def fixture_sample_stats(samples):
    stats = {}
    for fixture, data in samples.items():
        a = statistics.median(data["a"])
        b = statistics.median(data["b"])
        stats[fixture] = {
            "a_ms": a, "b_ms": b, "median_ms": (a + b) / 2.0,
            "dup_spread": abs(a - b) / min(a, b) if min(a, b) > 0 else float("inf"),
            "samples": len(data["a"]) + len(data["b"]),
            "capacity": data.get("capacity"),
            "layer": data.get("layer"), "row": data.get("row"),
        }
    return stats


def compare_arm(oracle_stats, candidate_stats, rows=TIMING_ROWS,
                shares=None, layers=LAYERS, bar=SHIP_BAR, floor=NOISE_FLOOR):
    """Per-layer share-weighted speedup, per-fixture ratios and regressions."""
    shares = shares or TIMING_SHARES_PCT
    per_layer = {}
    for layer in layers:
        entries = []
        weight_sum = 0.0
        weighted = 0.0
        regressions = []
        for row in rows:
            fixture = f"l{int(layer)}/r{int(row)}"
            oracle = oracle_stats[fixture]
            candidate = candidate_stats[fixture]
            ratio = oracle["median_ms"] / candidate["median_ms"]
            noise = max(oracle["dup_spread"], candidate["dup_spread"], floor)
            regressed = candidate["median_ms"] > oracle["median_ms"] * (1.0 + noise)
            weight = float(shares[row])
            weight_sum += weight
            weighted += weight * ratio
            entries.append({
                "fixture": fixture, "row": int(row),
                "oracle_ms": oracle["median_ms"], "candidate_ms": candidate["median_ms"],
                "ratio": ratio, "noise": noise, "regressed": regressed,
                "oracle_dup_spread": oracle["dup_spread"],
                "candidate_dup_spread": candidate["dup_spread"],
                "candidate_capacity": candidate.get("capacity"),
            })
            if regressed:
                regressions.append(fixture)
        weighted_speedup = weighted / weight_sum if weight_sum else 0.0
        per_layer[str(int(layer))] = {
            "weighted_speedup": weighted_speedup,
            "bar": bar,
            "bar_met": bool(weighted_speedup >= bar and not regressions),
            "regressions": regressions,
            "fixtures": entries,
        }
    combined_weight = 0.0
    combined_ratio = 0.0
    for layer in per_layer.values():
        for entry in layer["fixtures"]:
            weight = float(shares[entry["row"]])
            combined_weight += weight
            combined_ratio += weight * entry["ratio"]
    return {
        "layers": per_layer,
        "combined_weighted_speedup": (combined_ratio / combined_weight
                                      if combined_weight else 0.0),
        "ship_bar_met": all(layer["bar_met"] for layer in per_layer.values()),
        "bar": bar,
        "noise_floor": floor,
    }


def baseline_summary(runtime, rounds):
    samples = collect_samples(rounds, "oracle")
    stats = fixture_sample_stats(samples)
    for fixture, item in stats.items():
        item["roofline_ms"] = roofline_ms(runtime.curve[int(item["row"])])
        item["nominal_weight_bytes"] = (
            int(runtime.curve[int(item["row"])]) * EXPERT_BYTES)
    return {"stats": stats,
            "weights": {str(k): v for k, v in timing_weights().items()},
            "roofline_note": "nominal optimistic floor: distinct experts x "
                             f"{EXPERT_BYTES} B / {NOMINAL_BANDWIDTH_GB_S} GB/s"}


# ---------------------------------------------------------------------------
# Worker process protocol


class WorkerHandle:
    def __init__(self, name, command, log_path, env):
        self.name = name
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "wb")
        started = time.time()
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, env=env, bufsize=0)
        self.started = started

    def send(self, payload):
        line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def receive(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(
                f"{self.name}: no response within {timeout:.0f}s "
                f"(see {self.log_path})")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"{self.name}: worker exited rc={self.proc.poll()} "
                f"(see {self.log_path})")
        return json.loads(line.decode("utf-8"))

    def close(self, timeout=60.0):
        try:
            if self.proc.poll() is None:
                self.send({"cmd": "exit"})
                self.proc.wait(timeout=timeout)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self.kill()
        finally:
            self.log.close()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def _rotate(items, offset):
    items = list(items)
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def validate_session_spec(spec):
    if "arms" not in spec or not spec["arms"]:
        raise ValueError("session spec has no arms")
    names = [arm["name"] for arm in spec["arms"]]
    if len(set(names)) != len(names):
        raise ValueError("session arm names must be unique")
    if "oracle" not in names:
        raise ValueError("session spec must include the 'oracle' arm")
    for arm in spec["arms"]:
        if not arm.get("source"):
            raise ValueError(f"arm {arm['name']}: source is required")
        if not Path(arm["source"]).is_dir():
            raise FileNotFoundError(f"arm {arm['name']}: {arm['source']} is not a directory")
    if not spec.get("digests"):
        raise ValueError("session spec requires the frozen digests file")
    spec.setdefault("rounds", 3)
    spec.setdefault("replays", 5)
    spec.setdefault("startup_timeout", 1800)
    spec.setdefault("round_timeout", 900)
    return spec


def run_session(args):
    spec = validate_session_spec(load_json(Path(args.spec)))
    fixtures = load_json(Path(spec["fixtures"]))
    if fixtures.get("fixtures_sha256") is None:
        raise ValueError("fixtures file has no hash")
    driver_harness = Path(__file__).resolve()
    driver_hash = sha256_file(driver_harness)
    cache_root = Path(args.cache_root or (Path(spec["fixtures"]).parent / "session-cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    common_env = dict(os.environ)
    common_env["HOME"] = str(cache_root / "home")
    common_env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    common_env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    common_env["B12X_COMPILE_DISK_CACHE"] = "0"
    Path(common_env["HOME"]).mkdir(parents=True, exist_ok=True)
    workers = {}
    log_dir = Path(args.log_dir or (cache_root / "logs"))
    rounds = []
    result = {
        "schema": "cuteafd.kimi-k3-trellis-session.v1",
        "seed": args.seed,
        "spec": str(Path(args.spec).resolve()),
        "driver_harness_sha256": driver_hash,
        "fixtures_sha256": fixtures["fixtures_sha256"],
        "digests": str(Path(spec["digests"]).resolve()),
        "rounds_requested": int(spec["rounds"]),
        "replays": int(spec["replays"]),
        "arms": {},
        "rounds": rounds,
    }
    try:
        for arm in spec["arms"]:
            harness = Path(arm["source"]) / "benchmarks" / Path(driver_harness).name
            if not harness.is_file():
                raise FileNotFoundError(f"{arm['name']}: missing {harness}")
            if sha256_file(harness) != driver_hash:
                raise ValueError(
                    f"{arm['name']}: harness differs from the frozen session harness")
            env = dict(common_env)
            command = [
                sys.executable, str(harness), "serve",
                "--source", str(Path(arm["source"]).resolve()),
                "--stage", spec["stage"],
                "--fixtures", spec["fixtures"],
                "--digests", spec["digests"],
                "--capacities", json.dumps(arm.get("capacities")
                                           if arm.get("capacities")
                                           else list(DEFAULT_CAPACITIES)),
                "--replays", str(int(spec["replays"])),
                "--rounds", str(int(spec["rounds"])),
                "--tag", arm["name"],
            ]
            workers[arm["name"]] = WorkerHandle(
                arm["name"], command, log_dir / f"{arm['name']}.log", env)
        deadline = time.time() + float(spec["startup_timeout"])
        for arm in spec["arms"]:
            name = arm["name"]
            remaining = max(1.0, deadline - time.time())
            payload = workers[name].receive(remaining)
            if payload.get("event") != "ready":
                raise RuntimeError(f"{name}: unexpected startup event {payload}")
            result["arms"][name] = {
                "source": str(Path(arm["source"]).resolve()),
                "capacities_requested": arm.get("capacities") or list(DEFAULT_CAPACITIES),
                "ready": payload,
            }
        oracle_hashes = result["arms"]["oracle"]["ready"]["provenance"]["file_hashes"]
        if oracle_hashes != FROZEN_BLOBS:
            raise ValueError(
                "the oracle arm is not the frozen pin tree: "
                f"{ {k: v for k, v in oracle_hashes.items() if v != FROZEN_BLOBS.get(k)} }")
        for name, entry in result["arms"].items():
            if entry["ready"]["provenance"]["fixtures_sha256"] != fixtures["fixtures_sha256"]:
                raise ValueError(f"{name}: worker loaded different fixtures")
        for round_index in range(int(spec["rounds"])):
            order = _rotate([arm["name"] for arm in spec["arms"]], round_index)
            entry = {"round": round_index, "order": order, "arms": {}}
            for name in order:
                workers[name].send({"cmd": "round", "round": round_index})
                payload = workers[name].receive(float(spec["round_timeout"]))
                if payload.get("event") != "round":
                    raise RuntimeError(f"{name}: unexpected round event {payload}")
                entry["arms"][name] = payload
            rounds.append(entry)
        for name in workers:
            workers[name].close()
        result["aggregate"] = aggregate_session(result, fixtures, spec)
        write_json(Path(args.out), result)
        return result
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        try:
            write_json(Path(args.out), result)
        except OSError:
            pass
        raise
    finally:
        for worker in workers.values():
            worker.kill()


def aggregate_session(result, fixtures, spec):
    """Apply exactness and the share-weighted ship bar to one session."""
    digest_file = load_json(Path(spec["digests"]))
    frozen = digest_file.get("digests", {})
    samples = {name: collect_samples(result["rounds"], name)
               for name in result["arms"]}
    stats = {name: fixture_sample_stats(value) for name, value in samples.items()}
    oracle = stats["oracle"]
    aggregate = {"arms": {}, "protocol": {
        "rounds": int(spec["rounds"]),
        "replays_per_graph_per_round": int(spec["replays"]),
        "order": "rotating per round; every arm plus the oracle in one session",
        "duplicate_control": "two captured graphs per fixture; A/B medians "
                             "measure same-session noise",
        "fixtures": str(Path(spec["fixtures"]).resolve()),
    }}
    for name, entry in result["arms"].items():
        ready = entry["ready"]
        exact = bool(ready["exactness_ok"] and not ready["digest_mismatches"]
                     and not ready["graph_digest_failures"])
        if name == "oracle":
            compare = {"identical": True}
            ship = exact
        else:
            compare = compare_arm(oracle, stats[name])
            ship = bool(exact and compare["ship_bar_met"])
        aggregate["arms"][name] = {
            "exact": exact,
            "exactness_failures": ready["exactness_failures"],
            "digest_mismatches": ready["digest_mismatches"],
            "graph_digest_failures": ready["graph_digest_failures"],
            "frozen_pin_match": ready["provenance"]["frozen_pin_match"],
            "capacities": ready["capacities"],
            "baseline_ms": {fixture: item["median_ms"]
                            for fixture, item in stats[name].items()},
            "duplicate_spread": {fixture: item["dup_spread"]
                                 for fixture, item in stats[name].items()},
            "compare": compare,
            "shippable": ship,
        }
    # Every arm must have matched the frozen digests; the aggregate says so.
    aggregate["frozen_digests_checked"] = bool(frozen)
    return aggregate


# ---------------------------------------------------------------------------
# CLI


def cmd_fixtures(args):
    manifest = build_fixture_manifest()
    layer_data = manifest.pop("_layer_data")
    out = Path(args.out)
    if out.suffix == ".json":
        write_json(out, manifest)
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "fixtures.json", manifest)
    print(json.dumps({"fixtures": str(out), "sha256": manifest["fixtures_sha256"],
                      "layers": manifest["layers"],
                      "timing_rows": manifest["timing_rows"]}))
    del layer_data
    return manifest


def cmd_stage(args):
    manifest = stage_checkpoint(
        Path(args.checkpoint), Path(args.out),
        layers=[int(value) for value in args.layers.split(",")],
        width=args.width, start=args.rank_start,
        full_intermediate=args.full_intermediate,
        repo_id=args.repo_id, revision=args.revision)
    print(json.dumps({"stage": str(args.out),
                      "layers": {layer: info["sha256"]
                                 for layer, info in manifest["layers"].items()}}))


def cmd_freeze_oracle(args):
    record = freeze_oracle(Path(args.fork), Path(args.out),
                           Path(args.harness), Path(args.tests))
    print(json.dumps({"oracle": str(args.out), "archive_sha256":
                      record["archive_sha256"]}))


def _arm_args(args, runtime_args):
    namespace = argparse.Namespace(**runtime_args)
    runtime = ArmRuntime(namespace)
    return runtime


def _rounds_local(runtime, rounds, replays):
    collected = []
    for index in range(int(rounds)):
        payload = runtime.time_round(index, int(replays))
        collected.append({"round": index, "order": ["oracle"],
                          "arms": {"oracle": payload}})
    return collected


def _write_digests(path, runtime, tag):
    payload = dict(runtime.digests)
    payload["arms"] = [{"name": tag,
                        "capacities": [int(e["capacity"]) for e in runtime.capacities]}]
    write_json(Path(path), payload)


def cmd_run(args):
    runtime = _arm_args(args, {
        "source": args.source, "stage": args.stage, "fixtures": args.fixtures,
        "capacities": args.capacities, "device": args.device,
        "digests": args.digests, "tag": args.tag})
    ready = runtime.ready_payload()
    if ready["exactness_ok"] is False or ready["digest_mismatches"]:
        raise SystemExit(
            "exactness failed; refusing to publish a timing baseline: "
            + json.dumps({"failures": ready["exactness_failures"],
                          "mismatches": ready["digest_mismatches"]}))
    if ready["graph_digest_failures"]:
        raise SystemExit("captured graphs do not reproduce exactness digests")
    rounds = _rounds_local(runtime, args.rounds, args.replays)
    payload = {
        "schema": "cuteafd.kimi-k3-trellis-run.v1",
        "tag": args.tag,
        "seed": args.seed,
        **ready,
        "rounds": rounds,
        "baseline": baseline_summary(runtime, rounds),
    }
    write_json(Path(args.out), payload)
    if args.write_digests:
        _write_digests(args.write_digests, runtime, args.tag)
    print(json.dumps({
        "out": str(args.out),
        "exactness_ok": ready["exactness_ok"],
        "graph_digest_failures": ready["graph_digest_failures"],
        "graph_notes": ready["graph_notes"],
        "median_ms": {fixture: item["median_ms"]
                      for fixture, item in payload["baseline"]["stats"].items()},
        "duplicate_spread": {fixture: item["dup_spread"]
                             for fixture, item in payload["baseline"]["stats"].items()},
        "write_digests": args.write_digests,
    }))


def cmd_serve(args):
    runtime = _arm_args(args, {
        "source": args.source, "stage": args.stage, "fixtures": args.fixtures,
        "capacities": args.capacities, "device": args.device,
        "digests": args.digests, "tag": args.tag})

    def emit(payload):
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        sys.stdout.flush()

    emit(runtime.ready_payload())
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        command = json.loads(raw)
        if command["cmd"] == "round":
            try:
                emit({"event": "round",
                      **runtime.time_round(int(command["round"]),
                                           int(args.replays))})
            except Exception as error:  # noqa: BLE001 - report to the driver
                emit({"event": "error", "round": command.get("round"),
                      "error": f"{type(error).__name__}: {error}"})
                raise
        elif command["cmd"] == "exit":
            emit({"event": "bye"})
            return
        else:
            emit({"event": "error", "error": f"unknown command {command['cmd']}"})
            raise SystemExit(2)
    return


def cmd_session(args):
    result = run_session(args)
    aggregate = result["aggregate"]
    print(json.dumps({
        "out": str(args.out),
        "seed": args.seed,
        "arms": {name: {"exact": entry["exact"],
                        "shippable": entry["shippable"],
                        "weighted": {layer: value["weighted_speedup"]
                                     for layer, value in
                                     entry["compare"].get("layers", {}).items()}}
                 for name, entry in aggregate["arms"].items()},
    }, sort_keys=True))


def _common_arm_parser(parser):
    parser.add_argument("--source", required=True,
                        help="fork tree used for b12x (oracle-tree for the oracle)")
    parser.add_argument("--stage", required=True, help="staged weight directory")
    parser.add_argument("--fixtures", required=True, help="fixtures.json")
    parser.add_argument("--capacities", default=None,
                        help="JSON list of installed decode capacities")
    parser.add_argument("--digests", default=None,
                        help="frozen oracle digests to verify against")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="arm")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fixtures = sub.add_parser("fixtures", help="write the deterministic fixture manifest")
    fixtures.add_argument("--out", required=True)
    fixtures.set_defaults(func=cmd_fixtures)

    stage = sub.add_parser("stage", help="extract real expert weights for fixed layers")
    stage.add_argument("--checkpoint", required=True)
    stage.add_argument("--out", required=True)
    stage.add_argument("--layers", default="13,92")
    stage.add_argument("--width", type=int, default=INTERMEDIATE)
    stage.add_argument("--rank-start", type=int, default=0)
    stage.add_argument("--full-intermediate", type=int, default=FULL_INTERMEDIATE)
    stage.add_argument("--repo-id", default="wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6")
    stage.add_argument("--revision", default=None)
    stage.set_defaults(func=cmd_stage)

    freeze = sub.add_parser("freeze-oracle", help="archive the pin tree and hashes")
    freeze.add_argument("--fork", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--harness", required=True)
    freeze.add_argument("--tests", required=True)
    freeze.set_defaults(func=cmd_freeze_oracle)

    run = sub.add_parser("run", help="single-arm exactness + warm timing baseline")
    _common_arm_parser(run)
    run.add_argument("--out", required=True)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--replays", type=int, default=5)
    run.add_argument("--write-digests", default=None)
    run.add_argument("--seed", default="unknown")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="worker process for an interleaved session")
    _common_arm_parser(serve)
    serve.add_argument("--rounds", type=int, default=3)
    serve.add_argument("--replays", type=int, default=5)
    serve.set_defaults(func=cmd_serve)

    session = sub.add_parser("session", help="interleaved multi-arm timing session")
    session.add_argument("--spec", required=True)
    session.add_argument("--out", required=True)
    session.add_argument("--seed", default="unknown")
    session.add_argument("--log-dir", default=None)
    session.add_argument("--cache-root", default=None)
    session.set_defaults(func=cmd_session)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
