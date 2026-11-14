"""#1915 q/k/relative clip producer oracle and replay invariants.

FP32 normalized RMS < 2e-6; BF16 <= one output ulp (norm reduction ordering).
Includes inactive clip, zero, large positions, nonuniform scale and mutation.
"""
import functools

import pytest
import torch

from ..conftest import require_b12x


@functools.lru_cache(None)
def program(heads, dim, scaled, log, dtype):
    from b12x.integration.cuteafd.inkling2_clip import compile_inkling2_clip_aot
    return compile_inkling2_clip_aot(heads=heads, dim=dim, threshold=8 if dim == 16 else 128**.5*2,
                                    scaled=scaled, log_floor=4 if log else 0, log_alpha=.25 if log else 0,
                                    dtype=dtype)


def reference(x, scale, positions, dim, log):
    # Verbatim #1915 l2_clip plus the attention_forward global tau calculation.
    value = x.float() if scale is None else x.float() * scale.float()
    threshold = 8 if dim == 16 else 128**.5*2
    value = value * (threshold / (value.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.)
    if log:
        tau = 1. + .25 * torch.log(((positions + 1).float() / 4).clamp_min(1.))
        value = value * tau[:, None, None]
    return value


@pytest.mark.parametrize("heads,dim,scaled", [(48, 128, True), (48, 16, True), (8, 128, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("log", [False, True])
def test_clip_oracle_and_replay(heads, dim, scaled, dtype, log):
    require_b12x()
    p = program(heads, dim, scaled, log, dtype)
    for count in (1, 17, 2048, 2752, 4096):
        x = (torch.randn(count, heads, dim, device="cuda") * 5).to(dtype)
        x[0] = 0
        if count > 1:
            x[1] *= .001
        scale = torch.randn(heads, dim, device="cuda").bfloat16() if scaled else None
        positions = torch.arange(count, device="cuda", dtype=torch.int64) + 65530
        out = torch.full_like(x, float("nan"))
        def run():
            p.launch(x, scale, positions, out, scalars=(count,))
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.mul_(.7)
        positions.add_(100)
        if scaled:
            scale.mul_(1.1)
        graph.replay()
        want = reference(x, scale, positions, dim, log)
        assert torch.isfinite(out).all()
        if dtype == torch.float32:
            torch.testing.assert_close(out, want, atol=2e-6, rtol=2e-6)
        else:
            torch.testing.assert_close(out.float(), want, atol=2**-7, rtol=2**-7)
        assert torch.count_nonzero(out[0]) == 0
