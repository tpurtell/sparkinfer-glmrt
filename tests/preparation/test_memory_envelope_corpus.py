"""Every declared block-scaled configuration fits the caller's workspace cap.

The corpus under ``tests/preparation/corpora`` is written by vLLM's driver when
``VLLM_B12X_DUMP_QUERIES`` names a file: one JSON line per scalar declaration
with its stage, component, encoded query and invocation. This test replays the
block-scaled GEMM records on the host with a synthetic SM120 identity and
checks the memory formulas of every legal configuration; it also checks that
pool-dependent families were declared in the state stage and weight-only
families in the weights stage. A built-in record reproduces the vision-tower
shape that exhausted device memory in a full-model start.
"""
import json
import pathlib

import pytest

from b12x.preparation import DetectedDevice, DeviceIdentity

CORPORA = pathlib.Path(__file__).parent / "corpora"
IDENTITY = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
DEVICE = DetectedDevice(ordinal=0, identity=IDENTITY)
WORKSPACE_CAP = 2_000_000_000
STATE_STAGE_COMPONENTS = {
    "sequence.gdn_prefill", "sequence.kda_prefill", "sequence.ple",
    "attention.gdn", "attention.gqa", "attention.mla", "attention.qsa",
    "attention.sparse_mla", "attention.compressed_sparse_mla",
    "attention.dsa_indexer", "attention.mla_compress",
}
BUILTIN = [{
    "stage": "weights", "unit": "MXFP8", "request": "builtin.vision.fc",
    "autotune": False, "component": "gemm.blockscaled_precision",
    "query": {
        "recipe": "mxfp8", "num_tokens": 65_536, "in_features": 4_352,
        "padded_in_features": 4_352, "out_features": 3_456, "activation_mode": "a16",
        "source_contiguous": True, "source_aligned": True,
        "workspace_form": "provided", "workspace_nbytes": WORKSPACE_CAP,
    },
    "invocation": {},
}]


def _records():
    records = list(BUILTIN)
    for path in sorted(CORPORA.glob("*.jsonl")) if CORPORA.exists() else ():
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def _blockscaled_plan(record):
    import torch

    from b12x.gemm.blockscaled import _tuning
    from b12x.gemm.blockscaled._preparation import plan

    fields = {name: value for name, value in record["query"].items() if name != "codegen"}
    query = _tuning.BlockscaledQuery(**fields)
    return plan(query), query


@pytest.mark.parametrize("record", [r for r in _records() if r["component"] == "gemm.blockscaled_precision"],
                         ids=lambda record: record["request"])
def test_blockscaled_candidates_fit_the_workspace_cap(record):
    from b12x.gemm.blockscaled._preparation import _owned_bytes, _workspace_bytes
    from b12x.gemm.blockscaled._tuning import TUNING, effective_a16_config
    from b12x.preparation.types import _plan_scope

    declaration, query = _blockscaled_plan(record)
    configuration = TUNING.configure(query, device=IDENTITY, override=None)
    configs = [configuration.default, *(config for _, config in TUNING.iterate(configuration))]
    cap = query.workspace_nbytes
    for config in configs:
        needed = _workspace_bytes(query, config)
        assert cap is None or needed <= cap, (config, needed)
        with _plan_scope(declaration):
            requirements = declaration._memory_requirements(config, DEVICE)
        if query.workspace_form == "provided":
            assert requirements.scratch_nbytes <= (cap or requirements.scratch_nbytes) + 255
        else:
            assert _owned_bytes(query, config) <= (cap or 0) + 2 * query.num_tokens * query.padded_in_features + 8 << 20
    if record["request"] == "builtin.vision.fc":
        splits = {effective_a16_config(query, config)[2] for config in configs if config.mode == "a16"}
        assert splits == {1, 2}
        assert max(_workspace_bytes(query, config) for config in configs) == 1_811_939_328


def test_pool_dependent_declarations_belong_to_the_state_stage():
    for record in _records():
        expected = "state" if record["component"] in STATE_STAGE_COMPONENTS else "weights"
        assert record["stage"] == expected, (record["request"], record["component"], record["stage"])
