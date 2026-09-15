"""Preview the rank-zero preparation dashboard without a GPU.

Replays a synthetic 28-request preparation (cached, fixed, and raced requests,
a rank wait, compile bursts, and finishing) through ``PreparationDisplay``.

  python scripts/preview_preparation_display.py --live [--speed 4]
      drive the real display on this terminal, S times faster than the timeline
  python scripts/preview_preparation_display.py --frames DIR [--width 140 --height 40]
      export SVG (and PNG when rsvg-convert exists) stills at each phase
"""
import argparse
import io
import os
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from b12x.preparation.types import PreparationProgress  # noqa: E402
from b12x.preparation._progress import PreparationDisplay  # noqa: E402

FAMILIES = ("attention", "gemm", "moe", "norm", "sequence", "quantization", "comm")
NAMES = {
    "attention": ("dense_mla", "paged", "dsa_indexer", "sparse_mla", "qsa"),
    "gemm": ("mxfp8_linear", "blockscaled", "mm", "wo_projection", "bf16_gemv"),
    "moe": ("fused_moe", "ep_moe"), "norm": ("mhc", "hyperconnection"),
    "sequence": ("gdn_decode", "ple", "gdn_prefill"), "quantization": ("nvfp4", "mxfp8"),
    "comm": ("pcie",),
}


class _Requirement:
    """Stand-in for a CollectiveRequirement; the display only reads ``key``."""

    key = ("allreduce", "world", 8)


class Timeline:
    """Snapshots with the same field semantics as PreparationJob._progress."""

    def __init__(self, seed=7, requests=28, rounds=7):
        self.rng = random.Random(seed)
        self.total = requests
        self.rounds = rounds
        self.completed = 0
        self.cache_hits = 0
        self.measured = 0
        self.compilations = 0
        self.active = 0
        self.events = []

    def snap(self, delay, **fields):
        base = dict(
            running=False, pending_compilation=False, ready_collectives=(), done=False,
            phase="planning", component_id="", request_name="", completed_requests=self.completed,
            total_requests=self.total, candidate_count=0, candidates_prepared=0,
            measured_candidates=self.measured, completed_rounds=0, total_rounds=0,
            latest_round_us=(), cache_hits=self.cache_hits, compilations=self.compilations,
            active_compilations=self.active, elapsed_seconds=0.0, tuning_stopped=False,
        )
        base.update(fields)
        self.events.append((delay, PreparationProgress(**base)))

    def build(self):
        rng = self.rng
        self.snap(0.0, phase="planning", total_requests=0)
        self.snap(0.6, phase="planning")
        self.cache_hits = 9
        self.snap(0.9, phase="planning")
        kinds = ["cached"] * 9 + ["fixed"] * 8 + ["raced"] * 11
        rng.shuffle(kinds)
        kinds[0] = "raced"
        kinds[3] = "cached"
        for index, kind in enumerate(kinds):
            family = rng.choice(FAMILIES)
            name = rng.choice(NAMES[family])
            common = dict(
                component_id=f"{family}.{name}",
                request_name=f"{name}_q{rng.choice((1, 4, 16, 64))}_k{rng.choice((1024, 4096, 16384))}",
            )
            if index == 12:
                self.snap(0.2, phase="waiting for ranks", ready_collectives=(_Requirement(),), **common)
                self.snap(2.4, phase="waiting for ranks", ready_collectives=(_Requirement(),), **common)
            if kind == "cached":
                self.snap(0.05, phase="selecting", **common)
                self.snap(0.15, phase="priming", running=True, **common)
                self.completed += 1
                self.snap(0.25, phase="priming", **common)
            elif kind == "fixed":
                self.snap(0.05, phase="selecting", candidate_count=1, **common)
                programs = rng.randint(2, 6)
                for step in range(programs):
                    self.active = min(8, programs - step)
                    self.snap(0.35, phase="compiling", pending_compilation=True, candidate_count=1, **common)
                    self.compilations += 1
                self.active = 0
                self.snap(0.3, phase="priming", running=True, candidate_count=1, **common)
                self.completed += 1
                self.snap(0.2, phase="priming", candidate_count=1, **common)
            else:
                self._race(index, rng.choice((4, 6, 8, 12, 16)), common)
        self.snap(0.3, phase="finishing")
        self.snap(1.2, phase="finishing")
        self.snap(0.6, phase="ready", done=True)
        return self.events

    def _race(self, index, count, common):
        rng = self.rng
        self.snap(0.1, phase="selecting", candidate_count=count, **common)
        for staged in range(count):
            self.active = min(8, count - staged)
            self.compilations += 1
            self.snap(0.22, phase="compiling", pending_compilation=True, candidate_count=count, **common)
        self.active = 0
        for staged in range(count + 1):
            self.snap(0.12, phase="preparing candidates", running=True, candidate_count=count,
                      candidates_prepared=staged, **common)
        self.snap(0.9, phase="calibrating", running=True, candidate_count=count, candidates_prepared=count, **common)
        base = [rng.uniform(25, 140) for _ in range(count)]
        latest = ()
        for turn in range(1, self.rounds + 1):
            latest = tuple(value * rng.uniform(0.93, 1.07) for value in base)
            self.active = 2 if turn in (2, 3) and index % 3 == 0 else 0
            self.snap(0.55, phase="autotuning", running=True, candidate_count=count, candidates_prepared=count,
                      completed_rounds=turn, total_rounds=self.rounds, latest_round_us=latest, **common)
        self.measured += count
        race = dict(candidate_count=count, candidates_prepared=count, completed_rounds=self.rounds,
                    total_rounds=self.rounds, latest_round_us=latest, **common)
        self.snap(0.3, phase="compiling", pending_compilation=True, **race)
        self.snap(0.3, phase="priming", running=True, **race)
        self.completed += 1
        self.snap(0.15, phase="priming", **race)


def live(speed):
    with PreparationDisplay(global_rank=0) as display:
        for delay, progress in Timeline().build():
            time.sleep(delay / speed)
            display.update(progress)
            if progress.phase == "autotuning" and progress.completed_rounds in (3, 5):
                print(f"[log] background output interleaves above the panel during {progress.component_id}")


def _moment(progress, seen):
    checks = (
        ("01_plan", progress.phase == "planning"),
        ("02_forge", progress.phase == "compiling" and progress.active_compilations >= 4 and progress.candidate_count == 1),
        ("03_staging", progress.phase == "preparing candidates" and progress.candidates_prepared == 3),
        ("04_calibrating", progress.phase == "calibrating"),
        ("05_race_round1", progress.phase == "autotuning" and progress.completed_rounds == 1 and progress.candidate_count >= 8),
        ("06_race_round6", progress.phase == "autotuning" and progress.completed_rounds == 6 and progress.candidate_count >= 8),
        ("07_wait_ranks", progress.phase == "waiting for ranks"),
        ("08_priming_ledger", progress.phase == "priming" and progress.completed_requests >= 15),
        ("09_finishing", progress.phase == "finishing"),
        ("10_ready", progress.done),
    )
    for key, hit in checks:
        if hit and key not in seen:
            seen.add(key)
            return key
    return None


def _save(console, renderable, directory, key, width, height):
    console.print(renderable)
    path = os.path.join(directory, f"{key}_{width}x{height}.svg")
    with open(path, "w") as handle:
        svg = console.export_svg(title=f"b12x {key}", clear=True)
        # Preserve terminal spacing in SVG rasterizers as well as browsers.
        svg = svg.replace('<svg ', '<svg xml:space="preserve" ', 1)
        svg = svg.replace('font-family: Fira Code, monospace;', 'font-family: Source Code Pro, monospace;')
        handle.write(svg)
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-o", path[:-4] + ".png", path], check=True)


def frames(directory, width, height):
    from rich.console import Console

    display = PreparationDisplay(global_rank=0, stream=io.StringIO())
    display._started = time.monotonic()
    display._request_started = display._started
    console = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor",
                      width=width, height=height, record=True)
    display._console = console
    os.makedirs(directory, exist_ok=True)
    seen = set()
    for _, progress in Timeline().build():
        display.update(progress)
        key = _moment(progress, seen)
        if key is None:
            continue
        _save(console, display._render(), directory, key, width, height)
    display.close(failed=True)
    _save(console, display._render(), directory, "11_failed", width, height)
    print("frames written to", directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--frames")
    parser.add_argument("--width", type=int, default=140)
    parser.add_argument("--height", type=int, default=40)
    args = parser.parse_args()
    if args.live:
        live(args.speed)
    elif args.frames:
        frames(args.frames, args.width, args.height)
    else:
        parser.print_help()
