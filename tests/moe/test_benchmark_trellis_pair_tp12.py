from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file
import torch

from benchmarks.benchmark_trellis_pair_moe_tp12 import (
    _CODEBOOK_SOURCE_FORMATS,
    _FIXTURE_KIND,
    _FIXTURE_SCHEMA_VERSION,
    _RouteReplay,
    _load_route_fixture,
    _mode_tables,
    _paired_ratio_bootstrap,
    _write_new_result,
)


def test_qsrt_benchmark_uses_sqg_xor_cheb_t12() -> None:
    assert _CODEBOOK_SOURCE_FORMATS["sqg_xor_cheb_t12"] == "qsrt_sqg_e4m3"


def _write_route_fixture(
    path,
    *,
    manifest_override: object | None = None,
    tensor_overrides: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    experts = 32
    rows = 5
    ids = torch.stack(
        [
            (torch.arange(16, dtype=torch.int32) + row).remainder(experts)
            for row in range(rows)
        ]
    )
    modes = torch.zeros(experts, dtype=torch.uint8)
    modes[[1, 7, 18]] = 1
    tensors = {
        "fc1_pair_modes": modes,
        "fc2_pair_modes": modes.clone(),
        "topk_ids": ids,
        "topk_weights": torch.full((rows, 16), 1.0 / 16, dtype=torch.float32),
        "observation_ids": torch.arange(100, 100 + rows, dtype=torch.int64),
        "source_row_indices": torch.arange(0, 2 * rows, 2, dtype=torch.int64),
    }
    if tensor_overrides:
        tensors.update(tensor_overrides)
    routed_p24 = tensors["fc1_pair_modes"][tensors["topk_ids"].long()]
    manifest: object = {
        "kind": _FIXTURE_KIND,
        "schema_version": _FIXTURE_SCHEMA_VERSION,
        "tp_size": 12,
        "top_k": 16,
        "hidden_size": 3584,
        "local_intermediate_size": 256,
        "num_experts": experts,
        "fixture_rows": rows,
        "p24_route_executions": int(routed_p24.sum()),
    }
    if manifest_override is not None:
        manifest = manifest_override
    save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})
    return tensors


def test_paired_ratio_bootstrap_closes_exact_scale() -> None:
    baseline = [10.0, 11.0, 12.0, 13.0, 14.0]

    identity = _paired_ratio_bootstrap(
        baseline, baseline, replicates=200, seed=7
    )
    doubled = _paired_ratio_bootstrap(
        [2.0 * value for value in baseline],
        baseline,
        replicates=200,
        seed=7,
    )

    assert identity["point_estimate"] == pytest.approx(1.0)
    assert identity["bootstrap_ci95_low"] == pytest.approx(1.0)
    assert identity["bootstrap_ci95_high"] == pytest.approx(1.0)
    assert doubled["point_estimate"] == pytest.approx(2.0)
    assert doubled["bootstrap_ci95_low"] == pytest.approx(2.0)
    assert doubled["bootstrap_ci95_high"] == pytest.approx(2.0)


def test_paired_ratio_bootstrap_rejects_unpaired_samples() -> None:
    with pytest.raises(ValueError, match="same nonzero length"):
        _paired_ratio_bootstrap([1.0], [1.0, 2.0], replicates=20, seed=7)


def test_benchmark_result_is_atomic_and_never_overwritten(tmp_path) -> None:
    path = tmp_path / "result.json"

    _write_new_result(path, '{"complete": true}\n')

    assert path.read_text() == '{"complete": true}\n'
    assert not list(tmp_path.glob(".result.json.*.tmp"))
    with pytest.raises(FileExistsError):
        _write_new_result(path, '{"complete": false}\n')
    assert path.read_text() == '{"complete": true}\n'


def test_route_fixture_roundtrip_drives_exact_sparse_mode_tables(tmp_path) -> None:
    path = tmp_path / "routes.safetensors"
    tensors = _write_route_fixture(path)

    fixture = _load_route_fixture(path)
    fc1, fc2 = _mode_tables(
        "sparse",
        fixture.num_experts,
        device=torch.device("cpu"),
        sparse_p24_experts=1,
        route_fixture=fixture,
    )

    assert fixture.path == path.resolve()
    assert len(fixture.sha256) == 64
    assert fixture.num_experts == 32
    assert torch.equal(fc1, tensors["fc1_pair_modes"].to(torch.int32))
    assert torch.equal(fc2, tensors["fc2_pair_modes"].to(torch.int32))
    assert fixture.manifest["loaded_fc1_p24_route_fraction"] == pytest.approx(
        float(fc1[fixture.topk_ids.long()].float().mean())
    )


def test_route_replay_cycles_fixture_windows_without_reallocating_targets() -> None:
    source_ids = torch.arange(5 * 16, dtype=torch.int32).reshape(5, 16)
    source_weights = source_ids.to(torch.float32) / 100
    target_ids = torch.empty((2, 16), dtype=torch.int32)
    target_weights = torch.empty((2, 16), dtype=torch.float32)
    replay = _RouteReplay(
        source_ids=source_ids,
        source_weights=source_weights,
        target_ids=target_ids,
        target_weights=target_weights,
    )
    ids_pointer = target_ids.data_ptr()
    weights_pointer = target_weights.data_ptr()

    replay.load(1)
    assert torch.equal(target_ids, source_ids[2:4])
    assert torch.equal(target_weights, source_weights[2:4])
    replay.load(2)
    assert torch.equal(target_ids, source_ids[0:2])
    assert torch.equal(target_weights, source_weights[0:2])
    assert target_ids.data_ptr() == ids_pointer
    assert target_weights.data_ptr() == weights_pointer


@pytest.mark.parametrize(
    ("manifest_override", "tensor_overrides", "message"),
    [
        ([], None, "JSON object"),
        (
            None,
            {"fc1_pair_modes": torch.zeros(32, dtype=torch.int16)},
            "fc1 modes",
        ),
        (
            None,
            {
                "topk_ids": torch.zeros((5, 16), dtype=torch.int32),
            },
            "duplicate experts",
        ),
        (
            None,
            {"source_row_indices": torch.tensor([0, 2, 2, 6, 8])},
            "strictly increasing",
        ),
    ],
)
def test_route_fixture_rejects_malformed_input(
    tmp_path, manifest_override, tensor_overrides, message
) -> None:
    path = tmp_path / "bad.safetensors"
    _write_route_fixture(
        path,
        manifest_override=manifest_override,
        tensor_overrides=tensor_overrides,
    )

    with pytest.raises(ValueError, match=message):
        _load_route_fixture(path)
