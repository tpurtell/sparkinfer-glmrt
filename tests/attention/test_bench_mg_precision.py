"""Contracts for the MG precision diagnostic."""
from pathlib import Path

import pytest

from benchmarks.bench_mg_precision import SOURCE, aot_cell_seed, arm_traits, context_cpu_reference, critical_path_samples, decode_arm_traits, parse_args
from b12x.attention._shared.mla.traits import ComputeMode, ModelType


def test_default_geometry_and_controls(tmp_path):
    args = parse_args(["--format", "dsv4", "--output", str(tmp_path / "out.json")])
    assert (args.heads, args.indexed_width) == (64, 512)
    assert args.rows == [1, 64, 2048, 2752, 4096]
    assert args.scores == ["spread", "peaked"]


@pytest.mark.parametrize("option,value", [("--rows", "0"), ("--scores", "other"), ("--heads", "17"), ("--samples", "0")])
def test_invalid_controls(option, value, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--format", "dsv41", "--output", str(tmp_path / "o.json"), option, value])


def test_584b_qk_modes_preserve_cache_format():
    fp8 = arm_traits("dsv4", "fp8-qk")
    bf16 = arm_traits("dsv4", "bf16-qk")
    assert fp8.model_type == bf16.model_type == ModelType.DSV4
    assert fp8.kv_gmem_stride == bf16.kv_gmem_stride == 584
    assert fp8.compute_mode == ComputeMode.FP8
    assert bf16.compute_mode == ComputeMode.BF16
    assert not fp8.fp8_internal and not bf16.fp8_internal


def test_native_internal_fp8_does_not_select_fp8_qk():
    bf16 = arm_traits("dsv41", "bf16-pv")
    fp8 = arm_traits("dsv41", "fp8-pv")
    assert bf16.model_type == fp8.model_type == ModelType.DSV41
    assert bf16.compute_mode == fp8.compute_mode == ComputeMode.BF16
    assert not bf16.fp8_internal and fp8.fp8_internal


def test_decode_traits_match_native_aot():
    fp8 = decode_arm_traits("fp8-decode")
    assert fp8.compute_mode == ComputeMode.FP8 and fp8.fp8_internal
    assert (fp8.q_nope_stride, fp8.kv_smem_stride, fp8.nt_per_warp_xv) == (528, 624, 2)
    assert decode_arm_traits("fp8-f32-partials") == fp8
    bf16 = decode_arm_traits("bf16-decode")
    assert bf16.compute_mode == ComputeMode.BF16 and not bf16.fp8_internal


def test_decode_contract_and_accuracy_only(tmp_path):
    args = parse_args(["--format", "dsv41", "--mode", "decode", "--rows", "1,2,4,8,16,64",
        "--accuracy-only", "--output", str(tmp_path / "out.json")])
    assert args.timing_rows == []
    assert args.rows == [1, 2, 4, 8, 16, 64]
    with pytest.raises(SystemExit):
        parse_args(["--format", "dsv4", "--mode", "decode", "--output", str(tmp_path / "o.json")])


def test_wrong_format_arm_rejected():
    with pytest.raises(ValueError):
        arm_traits("dsv41", "fp8-qk")


def test_decode_aot_has_fixed_production_cells_and_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("B12X_MLA_SM120_NUM_SPLITS", raising=False)
    argv = ["--mode", "decode-aot", "--output", str(tmp_path / "decode.json")]
    assert parse_args(argv).cells is None
    assert parse_args(argv + ["--cells", "glm-h64,v4p-h128"]).cells == ["glm-h64", "v4p-h128"]
    for flags in (["--rows", "1"], ["--cells", "v4p-h16"], ["--samples", "1"]):
        with pytest.raises(SystemExit):
            parse_args(argv + flags)
    monkeypatch.setenv("B12X_MLA_SM120_NUM_SPLITS", "2")
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_aot_cell_seeds_do_not_depend_on_selection_or_iteration_order():
    cells = (("glmf-h64", "sparse"), ("v4p-h64", "c128"), ("v4p-h64", "c4"))
    forward = {cell: aot_cell_seed(2146, *cell) for cell in cells}
    assert forward == {cell: aot_cell_seed(2146, *cell) for cell in reversed(cells)}
    assert len(set(forward.values())) == len(cells)
    assert aot_cell_seed(2147, *cells[0]) != forward[cells[0]]


def test_c128_1m_confirmation_only_accepts_v4_cells(tmp_path):
    argv = ["--mode", "decode-aot", "--confirm-c128-1m", "--output", str(tmp_path / "decode.json")]
    assert parse_args(argv + ["--cells", "v4p-h128"]).confirm_c128_1m
    with pytest.raises(SystemExit):
        parse_args(argv + ["--cells", "glm-h64"])


def test_context_critical_path_pairs_alternating_blocks(tmp_path):
    args = parse_args(["--mode", "context-aot", "--cells", "v4p,ml4nope", "--output", str(tmp_path / "out.json")])
    assert args.cells == ["v4p", "ml4nope"]
    assert critical_path_samples([[1.0, 4.0, 1.0], [3.0, 2.0, 3.0]]) == [3.0, 4.0, 3.0]


@pytest.mark.parametrize("family,context", [("glm", False), ("glmf", False), ("v4f", False),
    ("v4p", False), ("glm", True), ("glmf", True), ("ml4", True)])
def test_aot_first_launch_inside_capture(family, context):
    """A new process must not silently choose a different cold graph route."""
    import os
    import subprocess
    import sys
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    child = r"""
import os
import sys
sys.path.insert(0, os.environ["CUTEAFD_SPARKINFER_SOURCE_DIR"])
import torch
from cuda.bindings import driver as cuda
from b12x.integration.cuteafd import FLASH, PRO, GLM53, GLM53_FLASH, context_split, exportable_compilation, mistral_large4
from b12x.integration.cuteafd.dsv4_sparse_mla import compile_dsv4_sparse_mla_aot
from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot
from importlib.util import module_from_spec, spec_from_file_location
spec = spec_from_file_location("duel_bench", os.environ["CUTEAFD_DUEL_HARNESS"])
harness = module_from_spec(spec)
spec.loader.exec_module(harness)
make_records = harness.make_records

family, context = sys.argv[1], sys.argv[2] == "context"
v4 = family.startswith("v4")
g = {"glm": GLM53, "glmf": GLM53_FLASH, "v4f": FLASH, "v4p": PRO}.get(family)
if g is None:
    g = mistral_large4.rope_attention()
rows = 64
device = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=device).manual_seed(2146)
qdim = 512 if v4 else g.latent_dim
width = 128 if v4 else getattr(g, "sparse_topk", g.index_topk)
cache, indices, lengths, *_ = make_records(width, 256 if v4 else g.page_rows, qdim, v4, gen, rows, device)
query = torch.randn((rows, g.heads, qdim), device=device, generator=gen).bfloat16()
args = (query, cache, indices, lengths)
with exportable_compilation():
    if context:
        # One chunk per split is the device-dependent dispatch arm.
        heads = g.heads // 2
        program = context_split.compile_sparse_mla_partial_aot(
            g, max_rows=rows, head_begin=heads, head_count=heads,
            num_splits=(width + 63) // 64, precision="bf16")
        lse = torch.full((rows, heads), float("nan"), device=device)
        args += (None, None, None)
    elif v4:
        extra, ei, el, *_ = make_records(8192, 2, qdim, True, gen, rows, device)
        args += (extra, ei, el, torch.zeros(g.heads, device=device))
        program = compile_dsv4_sparse_mla_aot(g, route="decode", max_rows=rows,
                   indexed_width=8192, indexed_page_rows=2, precision="bf16")
    else:
        program = compile_glm_sparse_mla_aot(g, route="decode", max_rows=rows,
                   precision="bf16", fp32_partials=family == "glmf")
out = torch.full((rows, heads if context else g.heads, 512), float("nan"), device=device, dtype=torch.bfloat16)
scratch = torch.empty(program.scratch_bytes(rows)["scratch"], device=device, dtype=torch.uint8)
stream = torch.cuda.Stream()
handle = cuda.CUstream(stream.cuda_stream)
scalars = (rows,) if v4 or context else (rows, 8)
outputs = (out, lse) if context else (out,)
def launch():
    program.launch(*args, *outputs, scratch, scalars=scalars, stream=handle)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
# First invocation of this compiled program, with no dispatch cache warm-up.
with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
    launch()
graph.replay()
torch.cuda.synchronize()
assert all(x.isfinite().all() for x in outputs), "cold graph output is not finite"
cold = [x.clone() for x in outputs]
for x in outputs:
    x.fill_(float("nan"))
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    launch()
stream.synchronize()
assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
           for a, b in zip(cold, outputs, strict=True)), "cold graph differs from warm launch"
"""
    env = {**os.environ, "CUTEAFD_SPARKINFER_SOURCE_DIR": str(SOURCE),
           "CUTEAFD_DUEL_HARNESS": str(Path(__file__).resolve().parents[2] / "benchmarks/bench_mg_precision.py")}
    result = subprocess.run([sys.executable, "-c", child, family, "context" if context else "full"], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_context_reference_respects_ordered_indices_and_sink():
    import torch
    q = torch.tensor([[[0.]], [[1.]]])
    k, v = torch.tensor([[0.], [1.]]), torch.tensor([[2.], [5.]])
    ids = torch.tensor([[0, 1], [1, 1]])
    output, lse = context_cpu_reference(q, (k, v, ids), (), torch.zeros(1), 1.)
    score = torch.tensor([[0., 0., 0.], [1., 1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(lse[:, 0], score.logsumexp(-1))
    torch.testing.assert_close(output[:, 0, 0], (score.softmax(-1) * torch.tensor([[2., 5., 0.], [5., 5., 0.]])).sum(-1))
