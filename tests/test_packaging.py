from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_PACKAGE = "sparkinfer.comm.pcie"
RUNTIME_CUDA_SOURCES = {
    "pcie_dcp_a2a.cu",
    "pcie_dcp_topk.cu",
    "pcie_dma.cu",
    "pcie_oneshot.cu",
    "pcie_twoshot.cu",
}
FORK_RUNTIME_REQUIREMENTS = {
    "torch>=2.12.0a0",
    "nvidia-cutlass-dsl==4.6.1",
    "nvidia-cutlass-dsl-libs-base==4.6.1",
    "nvidia-cutlass-dsl-libs-core==4.6.1",
    "nvidia-cutlass-dsl-libs-cu12==4.6.1",
    "nvidia-cutlass-dsl-libs-cu13==4.6.1",
}


def test_runtime_cuda_sources_are_in_package_data() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]

    assert package_data[PCIE_PACKAGE] == ["*.cu"]
    assert {
        path.name for path in (ROOT / "sparkinfer" / "comm" / "pcie").glob("*.cu")
    } == RUNTIME_CUDA_SOURCES


def test_fork_runtime_requirements_match_ngc_and_hybrid_toolchain() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = set(config["project"]["dependencies"])

    # NGC 26.05 ships a 2.12.0a0+ vendor build, which a >=2.12.0 stable floor
    # rejects. Hybrid artifacts are compiled and run with the exact 4.6.1 map.
    assert dependencies >= FORK_RUNTIME_REQUIREMENTS
