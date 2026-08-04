"""Checkpoint-native DeepSeek V4 query and main-KV producer.

The producer preserves the serialized block-FP8 projection contract while
keeping serving storage caller-owned.  ``pack_weights`` concatenates ``wq_a``
and ``wkv`` once at load time, so one hidden-row quantization and GEMM emits
both low-rank Q and raw KV.  ``run`` then:

1. normalizes low-rank Q and normalizes/RoPE-packs KV directly into the DSV4
   256-source-token physical page;
2. projects ``wq_b`` directly into the caller's final query buffer; and
3. applies per-head RMS normalization and partial RoPE in place.

The learned C=4 index producer consumes the same normalized Q-rank while it is
live, projects its separate 64x128 query, applies partial RoPE, randomized
Hadamard, and E2M1 QAT, and emits FP8 scorer queries plus FP32 learned head
weights into caller-owned buffers.  Selection remains owned by
``attention.nsa_indexer`` rather than duplicating its physical-slot top-k path.

There is no BF16 KV staging allocation and no serving-time tensor allocation.
The planned lifecycle is ``pack_weights`` (one time) -> ``plan`` -> ``bind``
(views only) -> ``run`` (CUDA-graph-capture safe after prewarm).

Integrated dSpark prompt priming uses ``plan_kv``/``bind_kv``/``run_kv`` to
project and pack only the target-main KV rows.  It shares the packed weights
and exact cache format above without paying for unused query projections.
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
        "IndexerCaps",
        "IndexerPlan",
        "IndexerBinding",
        "IndexerWeights",
        "KVPlan",
        "KVBinding",
        "Weights",
        "plan",
        "plan_indexer",
        "plan_kv",
        "bind",
        "bind_indexer",
        "bind_kv",
        "pack_weights",
        "pack_indexer_weights",
        "run",
        "run_indexer",
        "run_kv",
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
        IndexerBinding,
        IndexerCaps,
        IndexerPlan,
        IndexerWeights,
        KVBinding,
        KVPlan,
        Plan,
        Weights,
        bind,
        bind_indexer,
        bind_kv,
        is_supported,
        pack_indexer_weights,
        pack_weights,
        plan,
        plan_indexer,
        plan_kv,
        run,
        run_indexer,
        run_kv,
    )

install_lazy_api(globals(), META)
