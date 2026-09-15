from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.deepseek_attention_profiles import load_attention_profile


def _config(root: Path, *, v41: bool = True) -> dict:
    text = {
        "model_type": "deepseek_v41_text" if v41 else "deepseek_v4",
        "num_attention_heads": 64,
        "index_n_heads": 32 if v41 else 64,
        "head_dim": 512,
        "index_head_dim": 128,
        "index_topk": 512,
        "sliding_window": 128,
        "num_hidden_layers": 8,
        "num_nextn_predict_layers": 3 if v41 else 1,
        "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1]
        if v41
        else [0, 0, 4, 128, 4, 128, 4, 128],
        "index_source_layer_ids": [2, 4, 6],
        "candidate_source_layer_id": 4,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
    }
    raw = {"model_type": "deepseek_v41", "text_config": text} if v41 else text
    (root / "config.json").write_text(json.dumps(raw))
    return raw


def test_tp_shards_attention_but_not_indexer_heads(tmp_path: Path) -> None:
    _config(tmp_path)
    tp2 = load_attention_profile("deepseek-v4.1-flash", tmp_path, tp_size=2)
    tp4 = load_attention_profile("deepseek-v4.1-flash", tmp_path, tp_size=4)
    assert (tp2.attention_heads, tp4.attention_heads) == (32, 16)
    assert {form.heads for profile in (tp2, tp4) for form in profile.indexer_forms} == {
        32
    }


def test_ratio_one_remains_indexed_and_reuse_layers_do_not_rescore(
    tmp_path: Path,
) -> None:
    _config(tmp_path)
    profile = load_attention_profile("deepseek-v4.1-flash", tmp_path)
    index = {form.name: form for form in profile.indexer_forms}
    assert index["c2-dense"].layers == (2,)
    assert index["c1-source"].layers == (4,)
    assert index["c1-reindex"].layers == (6,)
    assert (
        index["c1-source"].candidate_topk_blocks * 8
        == index["c1-reindex"].max_candidates
    )
    attention = {form.name: form for form in profile.mla_forms}
    assert attention["c1"].layers == (4, 5, 6, 7)
    assert attention["c1"].indexed_width == attention["c2"].indexed_width == 512
    assert attention["c1"].indexed_page_size == 2 * attention["c2"].indexed_page_size
    assert attention["draft-swa"].indexed_width == 0
    assert attention["draft-swa"].swa_width == 192


def test_profile_rejects_incompatible_model_identity_and_source_topology(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path)
    raw["model_type"] = "deepseek_v4"
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_attention_profile("deepseek-v4.1-flash", tmp_path)
    raw["model_type"] = "deepseek_v41"
    # A later source using ratio two cannot be described as a C1 reindex layer.
    raw["text_config"]["compress_ratios"][6] = 2
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_attention_profile("deepseek-v4.1-flash", tmp_path)


def test_v4_profile_cannot_silently_misdescribe_fixed_index_pages(
    tmp_path: Path,
) -> None:
    _config(tmp_path, v41=False)
    with pytest.raises(ValueError):
        load_attention_profile("deepseek-v4-flash", tmp_path, block_size=512)


def test_candidate_oracle_accepts_ties_but_rejects_inferior_blocks() -> None:
    from benchmarks.benchmark_dsa_indexer_profiles import _assert_published_blocks

    scores = torch.ones((1, 32768), dtype=torch.bfloat16)
    # Any 2047 equal-valued older blocks plus the mandatory newest block is
    # valid. Index stability at a BF16 boundary tie is not the model contract.
    blocks = torch.cat((torch.arange(2047) * 2, torch.tensor([4095])))
    candidates = (blocks[:, None] * 8 + torch.arange(8)).reshape(1, -1).int()
    lengths = torch.tensor([16384], dtype=torch.int32)
    _assert_published_blocks(scores, candidates, lengths, 32768)
    scores[0, :8] = 0
    with pytest.raises(AssertionError):
        _assert_published_blocks(scores, candidates, lengths, 32768)
