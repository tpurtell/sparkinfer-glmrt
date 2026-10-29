"""CPU-only frozen duel geometry and provenance contracts."""
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/bench_kimi_ws_prefill.py"
spec = importlib.util.spec_from_file_location("duel", PATH)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


@pytest.mark.parametrize("capacity", D.CAPACITIES)
def test_prefix_probes_cover_rows_and_boundaries(capacity):
    rows = D.probe_rows(capacity)
    assert set(range(1, 129)) <= set(rows)
    assert capacity - 1 in rows
    assert all(1 <= row < capacity for row in rows)


@pytest.mark.parametrize("profile,width", D.PROFILES.items())
def test_geometry(profile, width):
    assert width in (1536, 3072)
    assert profile.startswith("rtx-")
    assert width % 256 == 0


def manifest():
    # The pinned package exports up to its widest shipped cell; merged wave
    # extents reuse that cell instead of a new export.
    return dict(sparkinfer_revision=D.PIN, compute=[12, 0], variants=[
        dict(directory=f"{profile}/m{capacity}", capacity=capacity, intermediate=width,
             experts=896, top_k=16, bits=[2, 3], blocks_per_sm=1,
             tile=[64, 128 if capacity == 4096 else 256, 64, 256], route_block=64,
             token_major_rotation=capacity != 4096, warp_specialized=capacity == 4096,
             ws_dynamic_tiles=capacity == 4096)
        for profile, width in D.PROFILES.items()
        for capacity in D.CAPACITIES if capacity <= D.SHIPPED_CAPACITY])


def test_manifest_preserves_shipped_recipe():
    cells = D.validate_manifest(manifest())
    assert len(cells) == 12
    for name, options in cells.items():
        ws = int(name.rsplit("m", 1)[1]) > 2752
        assert options["mode"] == ("ws" if ws else "cooperative")
        assert options["rotation"] == ("route" if ws else "token")


def test_merged_extents_reuse_the_shipped_ws_cell():
    cells = D.validate_manifest(manifest())
    for profile in D.PROFILES:
        assert cells[f"{profile}/m5504"] == cells[f"{profile}/m4096"]
        assert cells[f"{profile}/m8256"] == cells[f"{profile}/m4096"]


def test_merged_extents_need_the_shipped_ws_cell():
    data = manifest()
    for variant in data["variants"]:
        if variant["capacity"] == 4096:
            variant["warp_specialized"] = False
    with pytest.raises(ValueError):
        D.validate_manifest(data)


@pytest.mark.parametrize("field,value", [("sparkinfer_revision", "bad"), ("compute", [12, 1])])
def test_wrong_oracle_rejected(field, value):
    data = manifest()
    data[field] = value
    with pytest.raises(ValueError):
        D.validate_manifest(data)


def test_wrong_shape_rejected():
    data = manifest()
    data["variants"][0]["experts"] = 256
    with pytest.raises(ValueError):
        D.validate_manifest(data)


@pytest.mark.parametrize("category,name", [
    ("harness", "benchmarks/harness.py"),
    ("data", "fixtures/fixture.json"),
    ("data", "baselines/rtx/digests.json"),
    ("kernel_sources", "b12x/kernel.py"),
])
def test_frozen_hash_detects_mutation(tmp_path, category, name):
    duel = tmp_path / "duel"
    source = duel / "oracle/oracle-tree"
    root = duel if category == "data" else source
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    record = dict(harness={}, data={}, kernel_sources={})
    record[category][name] = D.sha(path)
    D.save(duel / "oracle/ORACLE.json", record)
    assert D.check_frozen(source, duel) == record
    path.write_bytes(b"modified")
    with pytest.raises(ValueError):
        D.check_frozen(source, duel)


@pytest.mark.parametrize("sealed,config", [(True, None), (False, "candidate.json")])
def test_baseline_cannot_replace_seal_or_use_config(sealed, config):
    data = {"baselines/rtx/digests.json": "sha"} if sealed else {}
    with pytest.raises(ValueError):
        D.validate_baseline_write({"data": data}, True, config)


def test_cell_keys_are_unique():
    keys = [D.cell_key(p, c, l, d) for p in D.PROFILES for c in D.CAPACITIES
            for l in D.LAYERS for d in ("random", "realistic")]
    assert len(keys) == len(set(keys)) == 48


@pytest.mark.parametrize("arm", [
    {"name": "oracle", "source": "/duel/oracle/oracle-tree", "config": "candidate.json"},
    {"name": "oracle", "source": "/other/tree"},
])
def test_session_rejects_fake_oracle(arm):
    with pytest.raises(ValueError):
        D.validate_session({"duel": "/duel", "arms": [arm]})


def test_session_accepts_frozen_oracle():
    assert D.validate_session({"duel": "/duel", "arms": [
        {"name": "oracle", "source": "/duel/oracle/oracle-tree"},
        {"name": "candidate", "source": "/entrant", "config": "candidate.json"},
    ]}) == ["oracle", "candidate"]
