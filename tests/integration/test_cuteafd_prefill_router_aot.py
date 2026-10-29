"""One arithmetic route for every family's prefill router, including MiMo hi/lo."""
from types import SimpleNamespace

import pytest
import torch

from ..conftest import require_b12x
from ._cuteafd import assert_bitwise


@pytest.fixture(scope="module", params=["glm", "glmf", "mimo", "mimof", "mimop", "kimi", "qwen4"])
def router(request):
    require_b12x()
    from b12x.integration.cuteafd import exportable_compilation
    from b12x.integration.cuteafd import glm_ffn, glmf, mimo_ffn, qwen4
    from b12x.integration.cuteafd._common import (
        GLM53, GLM53_FLASH, MIMO_V2_FLASH, MIMO_V26_FLASH, MIMO_V26_PRO, QWEN38_FLASH_NEXT,
    )

    family = request.param
    geometry, compile_router = {
        "glm": (GLM53, glm_ffn.compile_glm_prefill_router_scores_aot),
        "glmf": (GLM53_FLASH, glmf.compile_glmf_prefill_router_scores_aot),
        "mimo": (MIMO_V2_FLASH, mimo_ffn.compile_mimo_prefill_router_scores_aot),
        "mimof": (MIMO_V26_FLASH, mimo_ffn.compile_mimo_prefill_router_scores_aot),
        "mimop": (MIMO_V26_PRO, mimo_ffn.compile_mimo_prefill_router_scores_aot),
        "kimi": (SimpleNamespace(hidden=7168, routed_experts=896, top_k=16),
                 glmf.compile_glmf_prefill_router_scores_aot),
        "qwen4": (QWEN38_FLASH_NEXT, qwen4.compile_qwen4_prefill_router_scores_aot),
    }[family]
    with exportable_compilation():
        program = compile_router(geometry)
    assert program.geometry["route"] == "mma_prefill"
    assert program.geometry["exact_row_cap"] == 2**31 - 1
    return family, geometry, program


@pytest.mark.parametrize("mode", ["random", "cancellation", "dynamic"])
def test_prefill_router_row_invariance(router, mode):
    from b12x.integration.cuteafd.mimo_ffn import router_scratch_bytes, split_router_weight

    device = require_b12x()
    family, g, program = router
    e, h = g.routed_experts, g.hidden
    gen = torch.Generator().manual_seed(e + h)
    gate = torch.randn((e, h), generator=gen).to(device) / 64
    x = torch.randn((257, h), generator=gen).bfloat16().to(device)
    if mode == "cancellation":
        gate[:, 1::2] = -gate[:, ::2]
        x[:, 1::2] = x[:, ::2]
        x[:, ::257] *= 1.0078125
    elif mode == "dynamic":
        scale = torch.pow(2., (torch.arange(h, device=device) % 41 - 20).float())
        x = (x.float() * scale).bfloat16()
        gate /= scale
    hilo = getattr(g, "router_fp32", False)
    weight = split_router_weight(gate) if hilo else gate.bfloat16()
    is_mimo = family.startswith("mimo")

    def run(value):
        logits = torch.full((len(value), e), float("nan"), device=device)
        operands = [value.contiguous(), weight, logits]
        if is_mimo:
            operands.append(torch.empty(router_scratch_bytes(g, len(value)), dtype=torch.uint8, device=device))
        program.launch(*operands, scalars=(len(value),))
        torch.cuda.synchronize()
        assert torch.isfinite(logits).all()
        return logits

    single = torch.cat([run(x[row:row + 1]) for row in range(len(x))])
    for rows in range(1, 258):
        assert_bitwise(run(x[:rows]), single[:rows], f"{family} {mode} rows={rows}")
    full = run(x)
    for start, end in ((1, 65), (29, 60), (63, 128), (64, 129), (127, 257)):
        assert_bitwise(run(x[start:end]), full[start:end], f"{family} {mode} packed offset={start}")
    reference = x.float() @ weight[:e].float().t()
    if hilo:
        reference += x.float() @ weight[e:].float().t()
    torch.testing.assert_close(full, reference, rtol=1e-4, atol=1e-4)
