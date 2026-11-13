"""Inkling 2 attention oracle: the cuteafd reference's ``relative_attention``.

Copied verbatim (plus ``ROWS``, which the reference imports from its
``shape_invariant`` module) from cuteafd ``python/reference/families/inkling2/
model.py`` at branch ``work/inkling2-fidelity`` commit ``02cf57ff8``
(file introduced by ``270ad4322`` "Add independent Inkling 2 MXFP8 text
reference", issue #1915; sha256 of the file
``1e0b4f50d6a6c807f60c205fcc62209d853323a2e9f50a36045a26b8bd488cf0``). That
reference produced the family's fidelity golden (``inkling2-small-v1_20261113
_own-text-mxfp8-f32``) and matches vLLM ``d664cd1e``
``tests/models/inkling2/test_attention.py::_ref_attention``:

    scores = (q . k + bias / scale) * scale,  bias = rel[q, head, q_pos - k_pos] for 0 <= d < extent
    softmax over keys plus one unscaled sink logit per (token, head); out = p @ v, then sigmoid(gate).

Run it in F32 (``q``, ``k``, ``v``, ``relative``, ``gate``, ``sinks`` F32) for
the oracle; the kernel test passes ``gate = 0`` and doubles the result, so the
gate factor is exactly 0.5 and the comparison is to the bare attention.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ROWS = 128


def relative_attention(q, k, v, relative, gate, sinks, *, window=None):
    """Relative bias is unscaled; a zero-valued sink adds exp(sink) to Z.

    Fixed query/key geometry per causal M128 tile preserves prefix bytes even
    when a longer suffix changes the last tile or expert-token populations.
    """
    if (q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or not k.shape[1]
            or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]
            or q.shape[1] % k.shape[1] or gate.shape != q.shape
            or sinks.shape != q.shape[:2] or relative.ndim != 3
            or relative.shape[:2] != q.shape[:2] or not relative.shape[-1]
            or (window is not None and (type(window) is not int or window <= 0))):
        raise ValueError("invalid Inkling attention geometry")
    length, heads, dim = q.shape
    output = torch.empty_like(q)
    for start in range(0, length, ROWS):
        count, end = min(ROWS, length - start), start + ROWS
        first = 0 if window is None else max(0, start - window + 1)
        qpos = torch.arange(start, end, device=q.device)[:, None]
        kpos = torch.arange(first, end, device=q.device)[None, :]
        distance = qpos - kpos
        valid = distance >= 0
        if window is not None:
            valid &= distance < window
        extent = relative.shape[-1]
        for head in range(heads):
            kv_head = head // (heads // k.shape[1])
            query = F.pad(q[start:start + count, head], (0, 0, 0, ROWS - count))
            keys = F.pad(k[first:min(end, length), kv_head], (0, 0, 0, max(0, end - length)))
            values = F.pad(v[first:min(end, length), kv_head], (0, 0, 0, max(0, end - length)))
            rel = F.pad(relative[start:start + count, head], (0, 0, 0, ROWS - count))
            bias = rel.gather(-1, distance.clamp(0, extent - 1))
            bias = bias.masked_fill((distance < 0) | (distance >= extent), 0.)
            scores = ((query @ keys.T) + bias / dim ** -.5) * dim ** -.5
            scores.masked_fill_(~valid, -float("inf"))
            sink = F.pad(sinks[start:start + count, head], (0, ROWS - count))
            probabilities = torch.cat((scores, sink[:, None]), -1).softmax(-1)[:, :-1]
            attended = probabilities @ values
            output[start:start + count, head] = (attended[:count]
                * gate[start:start + count, head].sigmoid())
    return output
