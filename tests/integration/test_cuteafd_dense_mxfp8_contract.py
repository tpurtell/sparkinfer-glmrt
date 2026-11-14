"""CPU-only shared-record dense ABI contracts."""
import pytest
import torch
from b12x.integration.cuteafd import dense_mxfp8 as dense


def test_quant_records_abi(monkeypatch):
    monkeypatch.setattr(dense, "compile_program", lambda launch, **kw: (launch, kw))
    core, info = dense.compile_dense_mxfp8_quant_aot(input_features=6144, input_dtype="bf16")
    assert core.bf16
    assert [o.name for o in info["operands"]] == ["rotated_inputs", "records"]
    assert [s.name for s in info["scalars"]] == ["rows", "rotation_id", "cta_limit"]
    assert dense.record_bytes(1, 6144) == dense.record_bytes(16, 6144) == 192*528
    assert dense.record_bytes(17, 6144) == 2*192*528


def test_native_mxfp8_bf16_one_entry_table_contract(monkeypatch):
    monkeypatch.setattr(dense, "compile_program", lambda launch, **kw: (launch, kw))
    for k, n, side in [(6144, 512, 1), (3072, 6144, 2)]:
        quant, qinfo = dense.compile_dense_mxfp8_quant_aot(
            input_features=k, input_dtype="bf16", rotation_count=1)
        gemm, info = dense.compile_dense_mxfp8_aot(input_features=k, output_features=n,
            encoding="mxfp8", input_dtype="bf16", rotation_count=1, sharded_side=side)
        assert quant.bf16
        assert qinfo["operands"][0].shape == info["operands"][0].shape == "[1]"
        assert qinfo["geometry"]["rotation_count"] == info["geometry"]["rotation_count"] == 1
        assert info["operands"][2].dtype == torch.uint8
        assert gemm.scratch_bytes == dense.record_bytes(4096, k)


@pytest.mark.parametrize("encoding,n", [("trellis",8192),("mxfp8",288),("mxfp8",64)])
@pytest.mark.parametrize("side", [1,2])
def test_gemm_abi(monkeypatch, encoding, n, side):
    monkeypatch.setattr(dense, "compile_program", lambda launch, **kw: (launch, kw))
    core, info = dense.compile_dense_mxfp8_aot(input_features=2048,output_features=n,
                                            encoding=encoding,sharded_side=side)
    assert (core.n_cta,core.mpw,core.m_cta) == (64,2,256)
    assert [o.name for o in info["operands"]] == ["rotated_inputs","payload","scales","records","output"]
    assert info["operands"][3].dtype == torch.uint8
    assert core.scratch_bytes == dense.record_bytes(4096,2048)


def test_native_decode_cutovers_preserve_abi_and_capacity_only_keys(monkeypatch):
    monkeypatch.setattr(dense, "compile_program", lambda launch, **kw: (launch, kw))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    expected = {(6144, 8192): 32, (6144, 512): 64, (6144, 128): 64,
        (8192, 6144): 32, (6144, 3072): 64, (3072, 6144): 32,
        (6144, 12288): 32, (12288, 6144): 32, (6144, 4096): 64,
        (6144, 256): 64, (4096, 6144): 64, (6144, 1536): 64,
        (1536, 6144): 32, (6144, 6144): 32}
    assert dense.NATIVE_DECODE_ROWS == expected
    for (k, n), threshold in expected.items():
        args = dict(input_features=k, output_features=n, encoding="mxfp8",
            input_dtype="bf16", rotation_count=1)
        prefill, old = dense.compile_dense_mxfp8_aot(**args)
        routed, new = dense.compile_dense_mxfp8_aot(**args, decode="auto")
        assert routed.decode_rows == threshold
        assert routed.scratch_bytes == prefill.scratch_bytes
        for field in ("name", "operands", "scalars", "geometry"):
            assert new[field] == old[field]
        assert new["key"] != old["key"]
    routed, _ = dense.compile_dense_mxfp8_aot(input_features=128, output_features=32,
        encoding="mxfp8", decode="auto")
    assert routed.decode_rows == 0
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    routed, _ = dense.compile_dense_mxfp8_aot(input_features=6144, output_features=8192,
        encoding="mxfp8", decode="auto")
    assert routed.decode_rows == 0
    for mode in (True, "auto"):
        with pytest.raises(ValueError, match="group32 weights"):
            dense.compile_dense_mxfp8_aot(input_features=128, output_features=128, decode=mode)


@pytest.mark.parametrize("kwargs", [dict(input_features=127),dict(input_features=128,max_rows=0),
    dict(input_features=128,output_features=33),dict(input_features=128,encoding="fp8_row"),dict(input_features=128,encoding="int4_g32"),
    dict(input_features=128,output_features=64),dict(input_features=128,input_dtype="f32")])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        dense.geometry(**kwargs)
