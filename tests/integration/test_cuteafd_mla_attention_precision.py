"""Precision must change real QK/PV arithmetic, not only program metadata."""
from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x


@pytest.mark.parametrize("family", ["glm", "glmf", "v4"])
@pytest.mark.parametrize("route", ["decode", "prefill"])
def test_bf16_attention_beats_fp8_on_identical_records(family, route):
    require_b12x()
    from b12x.integration.cuteafd import GLM53, GLM53_FLASH, FLASH
    from b12x.integration.cuteafd.glm_sparse_mla import compile_glm_sparse_mla_aot
    from b12x.integration.cuteafd.dsv4_sparse_mla import compile_dsv4_sparse_mla_aot
    from b12x.attention._shared.mla.reference import (
        pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference,
    )
    from b12x.attention._shared.mla.compressed_reference import (
        pack_compressed_sparse_mla_kv_cache_reference,
        unpack_compressed_sparse_mla_kv_cache_reference,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2146)
    rows, heads, context = 8, 16, 256
    if family == "v4":
        raw = torch.randn(context, 512, device="cuda").bfloat16()
        cache = pack_compressed_sparse_mla_kv_cache_reference(raw[:, :448], raw[:, 448:], page_size=256)
        nope, rope = unpack_compressed_sparse_mla_kv_cache_reference(cache, page_size=256)
        keys = torch.cat((nope, rope), -1).float()
        qdim, topk, scale = 512, 128, 512 ** -.5
        geometry = replace(FLASH, heads=heads)
    else:
        geometry = replace(GLM53_FLASH if family == "glmf" else GLM53, heads=heads)
        qdim = geometry.latent_dim
        raw = torch.randn(context, qdim, device="cuda").bfloat16()
        records = pack_mla_kv_cache_reference(raw[:, :512], raw[:, 512:] if qdim == 576 else None)
        keys = unpack_mla_kv_cache_reference(records).squeeze(1).float()
        cache = records.reshape(context // 64, -1).contiguous()
        topk, scale = getattr(geometry, "sparse_topk", geometry.index_topk), geometry.softmax_scale
    query = (torch.randn(rows, heads, qdim, device="cuda") * 2.5).bfloat16()
    count = 128
    selected = (1 + (torch.arange(rows, device="cuda")[:, None] * 7
                     + torch.arange(count, device="cuda")[None]) % (context - 1)).int()
    indices = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    indices[:, :count] = selected
    lengths = torch.full((rows,), count, device="cuda", dtype=torch.int32)
    k = keys[selected.long()]
    scores = torch.bmm(query.float(), k.transpose(1, 2)) * scale
    if family == "v4":
        sink = torch.zeros(heads, device="cuda")
        denominator = torch.logsumexp(torch.cat((scores, sink[None, :, None].expand(rows, -1, -1)), -1), -1)
        expected = torch.bmm(torch.exp(scores - denominator[:, :, None]), k)
    else:
        expected = torch.bmm(scores.softmax(-1), k[:, :, :512])
    errors = {}
    for precision in ("fp8", "bf16"):
        if family == "v4":
            program = compile_dsv4_sparse_mla_aot(geometry, route=route, max_rows=rows, precision=precision)
        else:
            program = compile_glm_sparse_mla_aot(geometry, route=route, max_rows=rows, precision=precision)
        output = torch.empty((rows, heads, 512), device="cuda", dtype=torch.bfloat16)
        scratch = torch.empty(program.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
        if family == "v4":
            program.launch(query, cache, indices, lengths, cache, indices, lengths,
                           sink, output, scratch, scalars=(rows,))
        else:
            program.launch(query, cache, indices, lengths, output, scratch,
                           scalars=(rows, rows) if route == "decode" else (rows,))
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        errors[precision] = float((output.float() - expected).square().mean().sqrt()
                                  / expected.square().mean().sqrt())
    print(f"{family}/{route}: F32 relative RMS {errors}")
    assert errors["bf16"] < .01, errors
    assert errors["bf16"] < errors["fp8"] * .6, errors
