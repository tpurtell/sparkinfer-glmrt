"""Wide-core parity against frozen native arithmetic and independent F64 gates."""
import functools
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
def test_export(codebook, side, tmp_path):
    import multiprocessing
    import torch
    from b12x._lib.compile_pool import _initialize_worker
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    _initialize_worker(0, (12, 0), "trellis-wide", "trellis-wide",
        int(os.environ.get("TRELLIS_EXPORT_SMS", "188")), 101376, 102400,
        multiprocessing.Array("i", [0, 0]))
    directory = Path(os.environ.get("TRELLIS_EXPORT_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"wide_{codebook}_s{side}"
    with exportable_compilation():
        program = compile_trellis_linear_aot(input_features=7168, output_features=6144,
            bits=6, codebook=codebook, sharded_side=side, wide=True)
    program.export_to_c(str(directory), stem, stem)
    assert validate_exported_header(program, directory / f"{stem}.h", stem)["argument_count"] == 9
    assert not torch.cuda.is_initialized()


@pytest.fixture
def wide(monkeypatch):
    from b12x.integration.cuteafd import trellis_linear
    original = trellis_linear.compile_trellis_linear_aot
    monkeypatch.setattr(trellis_linear, "compile_trellis_linear_aot", functools.partial(original, wide=True))
    return original


@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
def test_accuracy_and_invariance(wide, codebook, side):
    from tests.gemm.test_native_trellis_linear import test_accuracy_rows_and_cta_invariance
    test_accuracy_rows_and_cta_invariance(side, 6, codebook)


@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
def test_graph(wide, codebook):
    from tests.gemm.test_native_trellis_linear import test_high_rate_graph_replay_and_frozen_resolution
    test_high_rate_graph_replay_and_frozen_resolution(6, codebook)


@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
def test_register_core_exact(wide, codebook, side):
    import torch
    from tests.gemm.test_native_trellis_linear import Case
    c = Case(k=1152, n=256, side=side, bits=6, rows=257, codebook=codebook)
    c.launch(257)
    expected = c.output.clone()
    c.program = wide(input_features=1152, output_features=256, sharded_side=side,
        bits=6, codebook=codebook, rotation_count=2, max_rows=257)
    c.launch(257)
    assert torch.equal(expected.view(torch.int32), c.output.view(torch.int32))


from tests.gemm.test_native_trellis_linear import KIMI_SHAPES


@pytest.mark.skipif(os.environ.get("NATIVE_TRELLIS_FULL_SHAPES") != "1", reason="opt-in full shapes")
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("k,n", KIMI_SHAPES)
def test_full_shapes(wide, codebook, side, k, n):
    from tests.gemm.test_native_trellis_linear import test_kimi_shapes
    test_kimi_shapes(k, n, side, 6, codebook)
