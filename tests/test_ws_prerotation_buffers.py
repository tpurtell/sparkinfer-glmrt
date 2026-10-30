"""CPU contracts for WS token-major rotation storage and launch ordering."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "b12x/moe/_shared/kernels/w4a16/mixed_trellis.py"
WS = SOURCE.with_name("mixed_trellis_ws.py")


@pytest.mark.parametrize("rows", [1024, 2048, 2752, 4096])
@pytest.mark.parametrize("width", [1536, 3072])
@pytest.mark.parametrize("prerotate", [False, True])
def test_rotation_workspace_has_exact_capacity_and_retains_fc2_alias(rows, width, prerotate):
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_make_mixed_trellis_buffers")
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    def allocate(shape, **kwargs):
        return SimpleNamespace(shape=shape, dtype=kwargs["dtype"])
    torch = SimpleNamespace(empty=allocate, zeros=allocate, float16="f16",
                            bfloat16="bf16", float32="f32", int32="i32")
    scope = dict(torch=torch, MixedTrellisBuffers=SimpleNamespace,
                 max_packed_route_slots=lambda rows, block, experts: rows + experts * block,
                 packed_gemm_scratch_elements=lambda **kw: 1)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    launch = SimpleNamespace(workspace_sms=None, size_m=rows, top_k=16,
                             direct_topk_routes=False, moe_block_size=64,
                             max_m_blocks=100000, intermediate_size=width,
                             hidden_size=3584, fused_input_rotation=False,
                             warp_specialized=True, ws_prerotate=prerotate,
                             topk_sum=SimpleNamespace(full_rotation_output_dtype="fp32"),
                             blocks_per_sm=1)
    buffers = scope[node.name](launch, device="cpu", sms=170, route_num_experts=896)
    assert buffers.fc2 is buffers.rotation_gate
    assert (buffers.rotation_up is not buffers.rotation_gate) == prerotate
    assert buffers.rotation_up.shape == (rows * 16, 3584)


def test_rotation_is_stream_ordered_before_ws_fc1_without_cooperative_launch():
    tree = ast.parse(WS.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    entry = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    launches = sorted((n for n in ast.walk(entry) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == "launch"),
                      key=lambda n: n.lineno)
    assert [n.func.value.func.attr for n in launches] == [
        "ws_rotation_kernel", "ws_fc1_kernel", "ws_act_kernel", "ws_fc2_kernel"]
    for call in launches:
        assert any(k.arg == "stream" and isinstance(k.value, ast.Name)
                   and k.value.id == "stream" for k in call.keywords)
        assert not any(k.arg == "cooperative" for k in call.keywords)


def test_prerotate_selector_defaults_to_the_bf16_a16_predicate():
    tree = ast.parse(WS.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert any(a.arg == "prerotate" for a in init.args.kwonlyargs)
    source = ast.unparse(init)
    assert "if prerotate is None" in source.replace("\n", " ") or "prerotate is None" in source
    compile_src = ast.unparse(next(n for n in ast.parse(SOURCE.read_text()).body
                                   if isinstance(n, ast.FunctionDef) and n.name == "compile_mixed_trellis"))
    assert "prerotate=ws_prerotate" in compile_src
