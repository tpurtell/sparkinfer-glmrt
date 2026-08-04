"""Checkpoint-native DeepSeek V4 learned KV compressor producer.

The op preserves the model's FP32 gated-pooling state while keeping every
serving buffer caller-owned.  Checkpoint BF16 ``wkv``/``wgate`` projections
are concatenated once at load time.  ``run_decode`` projects the hidden rows
into the planned arena, updates sequence-local C=4 overlap or C=128 state,
and writes completed groups directly into the compressed MLA page.  C=4 also
produces the randomized-Hadamard index cache in its planar FP8+FP32-scale ABI.

The decode binding requires at most one row for each sequence.  Multi-token
prefill/MTP state evolution will use a separate ordered binding; accepting it
here would race the sequence-local state and silently change model semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="dsv4_compressor",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Weights",
        "plan",
        "bind_decode",
        "pack_weights",
        "run_decode",
        "is_supported",
    ),
    dtypes=("bf16", "fp32", "fp8_e4m3"),
    recipes=("dsv4",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/sparkinfer",
        commit="native",
        paths=("sparkinfer/attention/dsv4_compressor/_impl.py",),
    ),
    test_path="tests/attention/test_dsv4_compressor.py",
    since="0.8.0",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Weights,
        bind_decode,
        is_supported,
        pack_weights,
        plan,
        run_decode,
    )

install_lazy_api(globals(), META)
