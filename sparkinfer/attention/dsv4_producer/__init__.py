"""Checkpoint-native DeepSeek V4 query and main-KV producer.

The producer preserves the serialized block-FP8 projection contract while
keeping serving storage caller-owned.  ``pack_weights`` concatenates ``wq_a``
and ``wkv`` once at load time, so one hidden-row quantization and GEMM emits
both low-rank Q and raw KV.  ``run`` then:

1. normalizes low-rank Q and normalizes/RoPE-packs KV directly into the DSV4
   256-source-token physical page;
2. projects ``wq_b`` directly into the caller's final query buffer; and
3. applies per-head RMS normalization and partial RoPE in place.

There is no BF16 KV staging allocation and no serving-time tensor allocation.
The planned lifecycle is ``pack_weights`` (one time) -> ``plan`` -> ``bind``
(views only) -> ``run`` (CUDA-graph-capture safe after prewarm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="dsv4_producer",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Weights",
        "plan",
        "bind",
        "pack_weights",
        "run",
        "is_supported",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/sparkinfer",
        commit="native",
        paths=(
            "sparkinfer/attention/dsv4_producer/_impl.py",
            "sparkinfer/gemm/_shared/block_fp8.py",
        ),
    ),
    test_path="tests/attention/test_dsv4_producer.py",
    since="0.8.0",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Weights,
        bind,
        is_supported,
        pack_weights,
        plan,
        run,
    )

install_lazy_api(globals(), META)
