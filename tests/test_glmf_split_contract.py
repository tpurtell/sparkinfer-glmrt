"""CPU-only ABI contracts for Flash's projection-free primitive exports."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Operand:
    name: str
    dtype: str
    shape: str
    role: str = "in"
    align: int = 16


class Geometry(SimpleNamespace):
    pass


class Primitive:
    def __init__(self, *args, **kwargs):
        self.pack = self.absorb = self.uv = self


G = Geometry(hidden=4096, heads=64, q_lora_rank=1536, kv_lora_rank=512,
    qk_head_dim=256, qk_nope_dim=256, v_head_dim=256, latent_dim=512,
    qkv_a_width=2048, kv_page_bytes=33792)


def compiler():
    source = ROOT / "b12x/integration/cuteafd/glmf_split.py"
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compile_glmf_stage_aot"]
    ns = dict(GLM53_FLASH=G, GLMFGeometry=Geometry, Operand=Operand,
        Scalar=lambda x: x, _RankNorm=Primitive, _MlaProducer=Primitive,
        _GlmfOutput=Primitive, _SwiGLU=Primitive, _check_rows=int,
        _align=lambda n: (n + 1023)//1024*1024,
        torch=SimpleNamespace(bfloat16="bf16", uint8="u8", int64="i64"),
        compile_program=lambda launch, **kw: dict(launch=launch, **kw))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns["compile_glmf_stage_aot"]


@pytest.mark.parametrize("stage,names", [
    ("rank_norm", ["q_a", "kv_a", "q_a_norm", "kv_a_norm", "kv_slots", "q_resid", "kv_cache", "scratch"]),
    ("absorb", ["a", "w", "out"]), ("values", ["a", "w", "out"]),
    ("swiglu", ["gate", "up", "hidden", "scratch"]),
])
def test_stage_abi(stage, names):
    p = compiler()(stage=stage, max_rows=4096)
    assert [o.name for o in p["operands"]] == names
    assert p["scalars"] == ("rows",)
    if stage == "rank_norm":
        assert p["operands"][1].shape == "[rows,512]"
        assert p["operands"][6].shape == "[pages,33792]"
        assert p["scratch"]["scratch"](17) == 17*2048*2
    if stage == "swiglu":
        assert p["scratch"]["scratch"](17) == 17*2048*4


def test_kda_reuses_original_geometry_and_clamped_activation():
    split = (ROOT / "b12x/integration/cuteafd/glmf_split.py").read_text()
    shared = (ROOT / "b12x/integration/cuteafd/kimi_split.py").read_text()
    assert "compile_kda_core_aot(g," in split
    assert "_GlmfFfn(g, inter, False, max_rows).swiglu" in split
    assert "self.core = _Kda(g," in shared
    assert 'self.full_rank_gate = kda_layout(g)["full_rank_gate"]' in shared
    assert "self.core.in_proj = _NoProjection()" in shared
