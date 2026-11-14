"""Independent #1915 normalized-key AND normalized-value mixture.

F32 2e-5 absolute/relative oracle bar; input arena must remain byte-exact.
"""
import functools

import pytest
import torch

from ..conftest import require_b12x


@functools.lru_cache(None)
def program(queries, addend):
    from b12x.integration.cuteafd.inkling2_residual import compile_inkling2_residual_aot
    return compile_inkling2_residual_aot(queries=queries, addend=addend)


def reference(arena, query, addend=None):
    # Equivalent to the frozen #1915 attn_res; each candidate has its own RMS.
    values = arena.float().clone()
    if addend is not None:
        values[-1] += addend
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    queries = query.float() * values.shape[-1] ** -.5
    output = []
    for q in queries:
        scores = (values * q).sum(-1).transpose(0, 1).softmax(-1)
        output.append((values * scores.transpose(0, 1)[..., None]).sum(0))
    return torch.stack(output)


@pytest.mark.parametrize("queries", [1, 3])
@pytest.mark.parametrize("has_addend", [False, True])
def test_residual_oracle(queries, has_addend):
    require_b12x()
    p = program(queries, has_addend)
    for count in (1, 17, 63):
        capacity = count + 3
        arena = torch.randn(9, capacity, 4608, device="cuda")
        # Greatly different magnitudes detect accidental raw-value mixing.
        arena *= torch.tensor([.001, .01, .1, 1, 3, 10, 20, 30, 40], device="cuda")[:, None, None]
        arena[:, count:] = float("nan")
        query = torch.randn(queries, 4608, device="cuda") * 3
        addend = torch.randn(count, 4608, device="cuda") if has_addend else None
        original = arena[:, :count].clone()
        out = torch.empty(queries, count, 4608, device="cuda")
        for candidates in (1, 2, 6, 8, 9):
            p.launch(arena, addend, query, out, scalars=(count, capacity, candidates))
            want = reference(arena[:candidates, :count], query, addend)
            torch.testing.assert_close(out, want, atol=2e-5, rtol=2e-5)
            assert torch.equal(arena[:, :count], original)
            assert torch.isnan(arena[:, count:]).all()


@pytest.mark.parametrize("queries", [1, 3])
def test_replay_changes_query_and_prefix(queries):
    require_b12x()
    p = program(queries, True)
    count = 7
    arena = torch.randn(9, count, 4608, device="cuda")
    query = torch.randn(queries, 4608, device="cuda")
    addend = torch.randn(count, 4608, device="cuda")
    out = torch.empty(queries, count, 4608, device="cuda")
    def run():
        p.launch(arena, addend, query, out, scalars=(count, count, 9))
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    query.mul_(2)
    arena[0].zero_()
    addend.mul_(3)
    graph.replay()
    torch.testing.assert_close(out, reference(arena, query, addend), atol=2e-5, rtol=2e-5)
