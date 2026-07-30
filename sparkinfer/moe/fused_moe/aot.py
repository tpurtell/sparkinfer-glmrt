"""Stable AOT surface for Sparkinfer fused-MoE artifacts."""

from __future__ import annotations

from .._shared.kernels.w4a4_w4a16.composed import (
    CAPACITY_BUCKETS as W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS,
)
from .._shared.kernels.w4a4_w4a16.composed import (
    BoundW4A4FC1W4A16FC2SparkWorkspace,
    W4A4FC1W4A16FC2SparkAOTArtifact,
    W4A4FC1W4A16FC2SparkSpec,
    bind_w4a4_fc1_w4a16_fc2_spark_workspace,
    compile_w4a4_fc1_w4a16_fc2_spark_aot,
    initialize_w4a4_fc1_w4a16_fc2_spark_routes,
)
from .._shared.kernels.w4a4_w4a16.prepare import (
    BoundW4A4FC1W4A16FC2Expert,
    W4A4FC1W4A16FC2Weights,
    bind_w4a4_fc1_w4a16_fc2_expert,
    prepare_w4a4_fc1_w4a16_fc2_weights,
)

__all__ = [
    "BoundW4A4FC1W4A16FC2Expert",
    "BoundW4A4FC1W4A16FC2SparkWorkspace",
    "W4A4_FC1_W4A16_FC2_SPARK_CAPACITY_BUCKETS",
    "W4A4FC1W4A16FC2SparkAOTArtifact",
    "W4A4FC1W4A16FC2SparkSpec",
    "W4A4FC1W4A16FC2Weights",
    "bind_w4a4_fc1_w4a16_fc2_expert",
    "bind_w4a4_fc1_w4a16_fc2_spark_workspace",
    "compile_w4a4_fc1_w4a16_fc2_spark_aot",
    "initialize_w4a4_fc1_w4a16_fc2_spark_routes",
    "prepare_w4a4_fc1_w4a16_fc2_weights",
]
