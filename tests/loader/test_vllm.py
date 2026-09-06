"""The adapter preserves vLLM's indexed source selection and owned inputs."""

import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("vllm")

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from b12x.integration.vllm.loader import B12xModelLoader
from b12x.loader._checkpoint import DirectWeightSession
from b12x.loader._pool import owns_tensor, shared_pool, weight_pool


@pytest.mark.parametrize("show_progress", [True, False])
def test_draft_iterator_uses_index_and_retained_tensors_keep_their_bytes(
    tmp_path, capsys, show_progress
):
    """Draft loading must not open unrelated target shards or reuse buffers."""
    save_file({"mtp.weight": torch.arange(16)}, tmp_path / "draft.safetensors")
    save_file({"mtp.bias": torch.arange(4) + 100}, tmp_path / "bias.safetensors")
    (tmp_path / "target.safetensors").write_bytes(b"must not be opened")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.weight": "draft.safetensors",
                    "mtp.bias": "bias.safetensors",
                    "model.weight": "target.safetensors",
                }
            }
        )
    )
    config = LoadConfig(
        load_format="b12x",
        use_tqdm_on_load=show_progress,
    )
    loader = B12xModelLoader(config)
    source = DefaultModelLoader.Source(
        str(tmp_path), revision=None, prefix="draft.", weight_name_prefixes=("mtp.",)
    )
    with shared_pool(allocation="pinned_wc"), DirectWeightSession() as session:
        loader._session = session
        retained = dict(loader._get_weights_iterator(source))
        assert set(retained) == {"draft.mtp.weight", "draft.mtp.bias"}
        values = {}
        for name, descriptor in retained.items():
            values[name] = torch.empty_like(descriptor, device="cuda")
            assert session(values[name], descriptor)
        loader._session = None
    torch.testing.assert_close(values["draft.mtp.weight"].cpu(), torch.arange(16))
    torch.testing.assert_close(values["draft.mtp.bias"].cpu(), torch.arange(4) + 100)
    assert config.load_format == "b12x"
    assert config.model_loader_extra_config == {}
    progress = capsys.readouterr().err
    if show_progress:
        assert "Loading safetensors checkpoint shards (b12x)" in progress
        assert "100% Completed | 2/2" in progress
    else:
        assert "Completed" not in progress


def test_gdn_convolution_shards_read_into_final_parameter_slices(tmp_path):
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )
    from vllm.model_executor.weight_transfer import weight_transfer

    path = tmp_path / "conv.safetensors"
    checkpoint = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    save_file({"conv.weight": checkpoint}, path)
    loader = mamba_v2_sharded_weight_loader(
        [(8, 0, 0), (4, 2, 1)], tp_size=2, tp_rank=1
    )
    with shared_pool(), DirectWeightSession() as session, weight_transfer(session):
        source = dict(session.weights([path]))["conv.weight"]
        target = torch.full((6, 2), -1.0, device="cuda")
        loader(target, source)
    torch.testing.assert_close(target.cpu(), checkpoint[4:])


def test_loader_policy_keeps_hyperconnection_workspaces_out_of_shared_weights(tmp_path):
    """Constructor workspaces stay GPU-owned while weights receive direct reads."""
    from vllm.model_executor.weight_transfer import copy_weight, weight_transfer
    from vllm.models.qwen3_8_flash_next.hyperconnection import (
        GroupedGemmaRMSNorm,
        HyperConnectionConfig,
        HyperConnectionWorkspace,
    )

    expected = torch.arange(128, dtype=torch.bfloat16)
    path = tmp_path / "norm.safetensors"
    save_file({"weight": expected}, path)
    with (
        weight_pool(allocation="pinned_wc") as allocator,
        DirectWeightSession() as session,
        weight_transfer(session, allocator=allocator),
        torch.device("cuda"),
    ):
        norm = GroupedGemmaRMSNorm(128, eps=1e-6, group_size=32, dtype=expected.dtype)
        workspace = HyperConnectionWorkspace(
            HyperConnectionConfig(4, 32, expected.dtype, 16, 1e-6), 8
        )
        source = dict(session.weights([path]))["weight"]
        copy_weight(norm.weight, source)
        assert owns_tensor(norm.weight)
        assert all(not owns_tensor(buffer) for buffer in workspace.buffers())
    torch.testing.assert_close(norm.weight.cpu(), expected)
    for buffer in workspace.buffers():
        buffer.fill_(7)
    torch.cuda.synchronize()
    torch.testing.assert_close(norm.weight.cpu(), expected)
