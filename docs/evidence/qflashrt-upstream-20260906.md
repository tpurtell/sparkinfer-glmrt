# QFlashRT upstream integration, 2026-09-06

Merged upstream `79e228ec6d7882a303868f63390ca75d9a35f5de` into fork
`fe054789069579e19ae5ec21f880b397bcf6575b`. This preserves recent projection-mixed
EXL3 routing and speculative-width kernels that were absent from older GLMRT pins.

The upstream changes bring bounded PLE, paged CuTe QSA prefill, chunked KDA
prefill, shared NVFP4 A4/A16 storage, and GB10 direct I/O/host storage support.
Merge resolutions preserve GLM_NEXT no-RoPE NVFP4 traits, local prefill capacities,
DSA supertile rebalancing, and the measured mixed-Trellis register declaration.
Unplanned decode normalizes flattened page widths before resolving traits;
prefill consumes per-record widths directly. Test fixtures now use supported
QSA geometry and the actual Qwen Flash Next expert dimensions (2560/640,
512 experts, top-10).

KDA had no catalog registration upstream. It now has a typed policy provider
that races public plans with `v_split` 64/32 at 16/48 heads and 128/1024 tokens,
with and without checkpoint export. All 16 candidates passed on each device.
Qualification compares outputs and written states to the FP32 recurrent oracle,
checks untouched states, and verifies graph replay with poisoned output/scratch,
stable addresses and zero replay allocation. Profiles cover exact measured
queries only. Max-Q KDA has an explicit empty rule set and remains heuristic;
no Max-Q measurements or performance claims are made here.

## Environments and results

| Target | Environment | Selected regression suites |
| --- | --- | --- |
| RTX PRO 6000 Blackwell Workstation, SM120, local GPU 1 | `qflashrt-quant-wip`, image `6e33f47a674d`, Python 3.12, Torch `2.12.0a0+5aff3928d8.nv26.05`, DSL 4.6.2, TVM FFI 0.1.12 | Attention/PLE/mixed EXL3: 356 passed, 1 skipped, 34 deselected. Final KDA/NVFP4/indexer/policy: 218 passed. |
| NVIDIA GB10, SM121, `ostrich` | `qflashrt-kernel-wip`, image `94711456c8e9`, Python 3.12, Torch 2.13.0+cu130, DSL 4.6.2 | Loader/PLE: 162 passed. Attention/mixed EXL3: 192 passed, 4 skipped, 34 deselected. Final KDA/NVFP4/indexer/policy: 218 passed. |

Tests run with `PYTHONPATH` pointing to this checkout and pytest `-o addopts=''`.
The selected mixed-EXL3 suite excludes the larger stress/performance cases with:

```text
not two_tier_matches and not one_grid_large and not glm52_large
and not runtime_partition and not shared_h_matches
```

The final 218-test suite is:

```bash
python -m pytest -o addopts='' -q --timeout=180 \
  tests/policy/test_policy_context.py tests/policy/test_component_catalog.py \
  tests/sequence/test_kda_prefill.py tests/moe/test_nvfp4_auto.py \
  tests/attention/test_paged_indexer_integration.py
```

Generate the KDA profile on each physical device with:

```bash
python scripts/generate_gpu_profile.py --components sequence.kda_prefill \
  --merge-from b12x/policy/_profiles/data/DEVICE.json.gz \
  --output /path/to/kda-profile.json --work-dir /path/to/kda-checkpoints \
  --groups 3 --repetitions 5
```

Raw reports and resumable candidate measurements remain in QFlashRT's ignored
`.qflashrt-cache/` and the `ostrich` workspace. Source transfer used
`rdmasync --rdma=required` over two rails. This is targeted kernel correctness
evidence, not whole-repository qualification or full-model speed/quality evidence.
