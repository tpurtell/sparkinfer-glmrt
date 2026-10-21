"""Bitwise K1 contract for expert-major EXLR8 prefill."""
import os
from pathlib import Path

import pytest
import torch

from b12x.integration.cuteafd.exlr8 import compile_exlr8_a16_aot
from benchmarks.benchmark_exlr8_a16 import compact_unit_descriptors, compact_program
from b12x.integration.cuteafd.exlr8_glue import compile_exlr8_group_aot
from b12x._lib.runtime_control import kernel_resolution_guard
from .exlr8_oracle import Case, make_wire


def buffers(rows, hidden, units, topk, capacity, group_rows):
    return dict(group_expert=torch.empty(capacity, dtype=torch.int32, device="cuda"),
                route_slots=torch.empty((capacity, group_rows), dtype=torch.int32, device="cuda"),
                inverse=torch.empty((rows, topk), dtype=torch.int32, device="cuda"),
                count=torch.empty(1, dtype=torch.int32, device="cuda"),
                activation=torch.empty((rows, hidden), dtype=torch.float16, device="cuda"),
                fc1=torch.empty((capacity, units, 2, (hidden+1023)//1024, group_rows, 128), device="cuda"),
                intermediate=torch.empty((capacity, units, group_rows, 128), dtype=torch.float16, device="cuda"),
                fc2=torch.empty((capacity, group_rows, hidden), device="cuda"),
                output=torch.empty((rows+1, hidden), dtype=torch.bfloat16, device="cuda"))


def programs(hidden, units, topk, capacity, group_rows, grouped):
    candidate = compile_exlr8_a16_aot(hidden=hidden, units=units, top_k=topk,
                                      max_groups=capacity, device_count=True,
                                      grouped=grouped, group_rows=group_rows)
    frozen = os.environ.get("EXLR8_FROZEN_GROUPED_ARM")
    if grouped and frozen:
        from benchmarks.benchmark_exlr8_a16 import compile_grouped_arm
        kernel, group, _ = compile_grouped_arm(Path(frozen), candidate)
        return group, kernel
    return (compile_exlr8_group_aot(experts=4, top_k=topk, max_groups=capacity,
                                   hidden=hidden, max_rows=4096 if grouped else 64,
                                   group_rows=group_rows), candidate)


def compute(k1, state, weights, pointers, bits, signs, rows, limit):
    extra = state["compact"][:2] if compact_program(k1) else ()
    scalars = (rows, limit, state["compact"][2]) if extra else (rows, limit)
    k1.launch(state["activation"], pointers, bits, state["group_expert"], state["route_slots"],
              state["inverse"], weights, state["fc1"], state["intermediate"], state["fc2"],
              signs[1], state["output"], state["count"], *extra, scalars=scalars)


def run(program, state, wire, ids, weights, pointers, bits, signs, rows, limit):
    group, k1 = program
    group.launch(ids, wire, signs[0], state["group_expert"], state["route_slots"],
                 state["inverse"], state["count"], state["activation"], scalars=(rows, limit))
    compute(k1, state, weights, pointers, bits, signs, rows, limit)


def reverse_packing(state, rows):
    count = int(state["count"].item())
    state["group_expert"][:count].copy_(state["group_expert"][:count].flip(0))
    state["route_slots"][:count].copy_(state["route_slots"][:count].flip((0, 1)))
    slots = state["route_slots"][:count].flatten().long()
    packed = torch.arange(slots.numel(), device=slots.device, dtype=torch.int32)
    valid = slots >= 0
    state["inverse"][:rows].flatten()[slots[valid]] = packed[valid]


# (group_rows, units): the units=8 case exceeds the shared-memory budget for
# resident intermediates, so it exercises the streamed FC2 ring.
@pytest.mark.parametrize("group_rows, units", [(16, 3), (32, 3), (64, 3), (64, 8)])
@pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"), reason="checkpoint fixture required")
def test_checkpoint_grouped_matches_chunks(group_rows, units):
    torch.cuda.set_per_process_memory_fraction(0.25)
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=40, hidden=1152, units=units)
    rows, hidden, topk = 129, case.h, case.topk
    gen = torch.Generator().manual_seed(408)
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen).cuda() * 0.02)
    ids = (torch.arange(rows * topk, device="cuda").reshape(rows, topk) % 4).int()
    weights = torch.rand((rows, topk), generator=gen).cuda() / topk
    signs = (case.signs_in, case.signs_out)
    base = buffers(64, hidden, units, topk, 128, 16)
    dec = programs(hidden, units, topk, 128, 16, False)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        live = min(rows-first, 64)
        run(dec, base, wire[first:], ids[first:], weights[first:], case.pointers, case.bits, signs, live, 48)
        expected[first:first+live].copy_(base["output"][:live])
    state = buffers(rows, hidden, units, topk, 128, group_rows)
    grouped = programs(hidden, units, topk, 128, group_rows, True)
    state["compact"] = compact_unit_descriptors(case.bits)
    with kernel_resolution_guard("EXLR8 grouped runtime rows and packing"):
        for live in (1, 8, 16, 17, 32, 63, 64, 65, 127, 128, 129):
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16))
                reverse_packing(state, live)
                compute(grouped[1], state, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16))
                assert torch.isnan(state["output"][live:]).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, rows, 48)
        for _ in range(2):
            weights.mul_(0.5)
            wire.copy_(wire.roll(1, 0))
            ids.copy_(ids.roll(1, 0))
            graph.replay()
            actual = state["output"][:rows].clone()
            for first in range(0, rows, 64):
                live = min(rows-first, 64)
                run(dec, base, wire[first:], ids[first:], weights[first:], case.pointers, case.bits, signs, live, 48)
                assert torch.equal(actual[first:first+live].view(torch.int16), base["output"][:live].view(torch.int16))


@pytest.mark.skipif(not os.environ.get("CUTEAFD_EXLR8_SNAPSHOT"), reason="checkpoint fixture required")
def test_checkpoint_every_live_row_count_matches_chunks():
    import json
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    units, group_rows, rows = 3, 64, 4096
    case = Case(os.environ["CUTEAFD_EXLR8_SNAPSHOT"], layer=40, hidden=1152, units=units)
    hidden, topk = case.h, case.topk
    capacity = (rows * topk + group_rows - 1) // group_rows + 4
    gen = torch.Generator().manual_seed(408)
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen).cuda() * 0.02)
    ids = (torch.arange(rows * topk, device="cuda").reshape(rows, topk) % 4).int()
    weights = torch.rand((rows, topk), generator=gen).cuda() / topk
    signs = (case.signs_in, case.signs_out)
    base = buffers(64, hidden, units, topk, 128, 16)
    dec = programs(hidden, units, topk, 128, 16, False)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        live = min(rows - first, 64)
        run(dec, base, wire[first:], ids[first:], weights[first:], case.pointers, case.bits, signs, live, 48)
        expected[first:first+live].copy_(base["output"][:live])
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    state = buffers(rows, hidden, units, topk, capacity, group_rows)
    grouped = programs(hidden, units, topk, capacity, group_rows, True)
    state["compact"] = compact_unit_descriptors(case.bits)
    live_counts = range(1, rows + 1) if os.environ.get("EXLR8_EXHAUSTIVE") else (1, 16, 64, 65, 512, 1024, 4096)
    with kernel_resolution_guard("EXLR8 grouped checkpoint exhaustive row invariance"):
        for live in live_counts:
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped, state, wire, ids, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16)), (live, limit)
                reverse_packing(state, live)
                compute(grouped[1], state, weights, case.pointers, case.bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16)), (live, limit, "reverse")
                assert torch.isnan(state["output"][live:]).all()
    cgroup_cur = Path("/sys/fs/cgroup/memory.current")
    cgroup_peak = Path("/sys/fs/cgroup/memory.peak")
    cgroup_max = Path("/sys/fs/cgroup/memory.max")
    limit_16gb = 16 * (1024 ** 3)
    cg_cur = int(cgroup_cur.read_text().strip()) if cgroup_cur.exists() else None
    cg_peak = int(cgroup_peak.read_text().strip()) if cgroup_peak.exists() else None
    cg_max_raw = cgroup_max.read_text().strip() if cgroup_max.exists() else None
    cg_limit = int(cg_max_raw) if (cg_max_raw and cg_max_raw.isdigit()) else limit_16gb
    effective_limit = min(limit_16gb, cg_limit)
    cuda_alloc = int(torch.cuda.memory_allocated())
    cuda_max_alloc = int(torch.cuda.max_memory_allocated())
    cuda_res = int(torch.cuda.memory_reserved())
    cuda_max_res = int(torch.cuda.max_memory_reserved())
    observed_cgroup = cg_peak if cg_peak is not None else cg_cur
    fits_16gb = bool(
        cuda_max_alloc < effective_limit
        and cuda_max_res < effective_limit
        and (observed_cgroup is None or observed_cgroup < effective_limit)
    )
    mem_report = {
        "hidden": hidden,
        "blocks": hidden // 128,
        "rows": rows,
        "topk": topk,
        "units": units,
        "capacity": capacity,
        "group_rows": group_rows,
        "frozen_arm": os.environ.get("EXLR8_FROZEN_GROUPED_ARM"),
        "cuda_allocated_bytes": cuda_alloc,
        "cuda_max_allocated_bytes": cuda_max_alloc,
        "cuda_reserved_bytes": cuda_res,
        "cuda_max_reserved_bytes": cuda_max_res,
        "cgroup_memory_current_bytes": cg_cur,
        "cgroup_memory_peak_bytes": cg_peak,
        "cgroup_memory_max": cg_max_raw,
        "effective_limit_bytes": effective_limit,
        "fits_16gb_container": fits_16gb,
    }
    out_dir = Path("/workspace/cuteafd/results")
    if out_dir.exists():
        arm_tag = Path(os.environ.get("EXLR8_FROZEN_GROUPED_ARM", "head")).name[:8]
        (out_dir / f"argon4-s2.mem1152-{arm_tag}.json").write_text(json.dumps(mem_report, indent=2) + "\n")
    assert fits_16gb, f"Exhaustive hidden=1152 test exceeded 16 GiB container limit: {mem_report}"


@pytest.mark.parametrize("units", [1, 16])
def test_every_live_row_count_matches_decode(units):
    torch.cuda.set_per_process_memory_fraction(0.25)
    # Two K128 blocks: block 1's pinned FMA uses its own scale load and the
    # partial plane is written after both blocks accumulate. Cross-split
    # read+add is covered by the hidden-1152 checkpoint cases (9 blocks,
    # 2 splits).
    hidden, topk, rows = 256, 2, 4096
    gen = torch.Generator(device="cuda").manual_seed(408)
    owners = []
    pointers = torch.zeros((4, units, 3, 3), dtype=torch.int64)
    bits = torch.zeros((4, units, 3), dtype=torch.int32)
    for e in range(4):
        # Include an empty expert and rotated, unequal noncontiguous unit sets.
        owned = range(units) if units == 1 else (() if e == 3 else
                (u for u in range(units) if (units * e + u + 3) % 6 == 0))
        for u in owned:
            for p in range(3):
                width = 2 + (e+u+p) % 3
                ts = (torch.randint(256, (hidden*16*width,), generator=gen, dtype=torch.uint8, device="cuda"),
                      torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                      torch.full((hidden//128,), 0.1, dtype=torch.float16, device="cuda"))
                owners.extend(ts)
                pointers[e, u, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[e, u, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(rows*topk, device="cuda") * 3 + 1) % 4).reshape(rows, topk).int()
    weights = torch.rand((rows, topk), generator=gen, device="cuda") / topk
    base, state = buffers(64, hidden, units, topk, 128, 16), buffers(rows, hidden, units, topk, 132, 64)
    dec, grouped = programs(hidden, units, topk, 128, 16, False), programs(hidden, units, topk, 132, 64, True)
    state["compact"] = compact_unit_descriptors(bits)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        run(dec, base, wire[first:], ids[first:], weights[first:], pointers, bits, signs, 64, 48)
        expected[first:first+64].copy_(base["output"][:64])
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    live_counts = range(1, rows+1) if os.environ.get("EXLR8_EXHAUSTIVE") else (1, 16, 64, 65, 512, 1024, 4096)
    with kernel_resolution_guard("EXLR8 grouped exhaustive row invariance"):
        for live in live_counts:
            for limit in (1, 48, 188):
                state["output"].fill_(float("nan"))
                run(grouped, state, wire, ids, weights, pointers, bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16)), (live, limit)
                reverse_packing(state, live)
                compute(grouped[1], state, weights, pointers, bits, signs, live, limit)
                assert torch.equal(state["output"][:live].view(torch.int16), expected[:live].view(torch.int16)), (live, limit, "reverse")
                assert torch.isnan(state["output"][live:]).all()


def test_schedule_dispatch_and_env_override(monkeypatch):
    if os.environ.get("EXLR8_FROZEN_GROUPED_ARM"):
        return
    from b12x.moe._shared.kernels.w4a16.exlr8_grouped import Exlr8GroupedKernel
    monkeypatch.delenv("EXLR8_GROUPED_SCHEDULE", raising=False)
    monkeypatch.delenv("EXLR8_GROUPED_BASE_THRESHOLD", raising=False)
    assert Exlr8GroupedKernel.resolve_base_row_threshold() == 4096
    monkeypatch.setenv("EXLR8_GROUPED_SCHEDULE", "base")
    assert Exlr8GroupedKernel.resolve_base_row_threshold() == 1
    monkeypatch.setenv("EXLR8_GROUPED_SCHEDULE", "v28")
    assert Exlr8GroupedKernel.resolve_base_row_threshold() == 4097
    monkeypatch.setenv("EXLR8_GROUPED_SCHEDULE", "auto")
    monkeypatch.setenv("EXLR8_GROUPED_BASE_THRESHOLD", "2049")
    assert Exlr8GroupedKernel.resolve_base_row_threshold() == 2049
    monkeypatch.delenv("EXLR8_GROUPED_BASE_THRESHOLD", raising=False)

    torch.cuda.set_per_process_memory_fraction(0.25)
    hidden, units, topk, rows = 6144, 1, 2, 4096
    gen = torch.Generator(device="cuda").manual_seed(408)
    owners = []
    pointers = torch.zeros((4, units, 3, 3), dtype=torch.int64)
    bits = torch.zeros((4, units, 3), dtype=torch.int32)
    for e in range(4):
        for u in range(units):
            for p in range(3):
                width = 2 + (e + u + p) % 3
                ts = (
                    torch.randint(256, (hidden * 16 * width,), generator=gen, dtype=torch.uint8, device="cuda"),
                    torch.full((128,), 0.1, dtype=torch.float16, device="cuda"),
                    torch.full((hidden // 128,), 0.1, dtype=torch.float16, device="cuda"),
                )
                owners.extend(ts)
                pointers[e, u, p] = torch.tensor([t.data_ptr() for t in ts])
                bits[e, u, p] = width
    pointers, bits = pointers.cuda(), bits.cuda()
    signs = [torch.ones(hidden, dtype=torch.int8, device="cuda") for _ in range(2)]
    wire, _ = make_wire(torch.randn((rows, hidden), generator=gen, device="cuda") * 0.02)
    ids = ((torch.arange(rows * topk, device="cuda") * 3 + 1) % 4).reshape(rows, topk).int()
    weights = torch.rand((rows, topk), generator=gen, device="cuda") / topk
    base, state = buffers(64, hidden, units, topk, 128, 16), buffers(rows, hidden, units, topk, 132, 64)
    dec = programs(hidden, units, topk, 128, 16, False)
    state["compact"] = compact_unit_descriptors(bits)
    expected = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    for first in range(0, rows, 64):
        run(dec, base, wire[first:], ids[first:], weights[first:], pointers, bits, signs, 64, 48)
        expected[first:first + 64].copy_(base["output"][:64])
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)

    for schedule in ("auto", "base", "v28"):
        monkeypatch.setenv("EXLR8_GROUPED_SCHEDULE", schedule)
        grouped = programs(hidden, units, topk, 132, 64, True)
        with kernel_resolution_guard(f"EXLR8 grouped schedule={schedule}"):
            for live in (1, 384, 512, 768, 1024, 1408, 2048, 4096):
                for limit in (96, 188):
                    state["output"].fill_(float("nan"))
                    run(grouped, state, wire, ids, weights, pointers, bits, signs, live, limit)
                    assert torch.equal(
                        state["output"][:live].view(torch.int16),
                        expected[:live].view(torch.int16),
                    ), (schedule, live, limit)
                    assert torch.isnan(state["output"][live:]).all()
    monkeypatch.delenv("EXLR8_GROUPED_SCHEDULE", raising=False)


def test_frozen_candidate_arms():
    if os.environ.get("EXLR8_FROZEN_GROUPED_ARM") or os.environ.get("EXLR8_RUN_AUTO_ARMS") != "1":
        return
    arms_root = Path(os.environ.get("EXLR8_GROUPED_ARMS_DIR", "/workspace/cuteafd/results/arms"))
    if not arms_root.is_dir():
        return
    prefixes = os.environ.get("EXLR8_CANDIDATE_ARM_PREFIXES", "154a80cb,6093aada,7b45d52f,46a3477c").split(",")
    import subprocess
    import sys
    for prefix in (p.strip() for p in prefixes if p.strip()):
        matches = [d for d in sorted(arms_root.glob(prefix + "*")) if d.is_dir()]
        if not matches:
            continue
        arm_dir = matches[0]
        env = dict(os.environ)
        env["EXLR8_FROZEN_GROUPED_ARM"] = str(arm_dir)
        env["EXLR8_SKIP_AUTO_ARMS"] = "1"
        cmd = [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            str(Path(__file__).resolve()),
            f"--basetemp=/workspace/cuteafd/results/pytest-arm-{prefix}",
        ]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log_path = Path(f"/workspace/cuteafd/results/argon4-s2.arm-{prefix}.tests.log")
        if log_path.parent.exists():
            log_path.write_text(f"ARM={arm_dir}\nCMD={' '.join(cmd)}\nRC={res.returncode}\n{res.stdout}\n{res.stderr}\n")
        assert res.returncode == 0, f"Frozen arm {arm_dir} failed pytest:\n{res.stdout}\n{res.stderr}"
