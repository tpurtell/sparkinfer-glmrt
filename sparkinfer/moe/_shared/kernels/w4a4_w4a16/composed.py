"""AOT composition for source-W13 W4A4 FC1 and packed-W2 W4A16 FC2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import torch

import sparkinfer._lib.dense_gemm as dense
from sparkinfer._lib.utils import make_ptr
from sparkinfer.moe._shared.kernels.w4a16.host import (
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    select_route_block_size_m,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    _cutlass_element_dtype,
    _select_tile_config,
    compile_w4a16_activation,
    compile_w4a16_gemm,
    cute,
)

from .prepare import BoundW4A4FC1W4A16FC2Expert


CAPACITY_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
SPARK_AOT_ABI_VERSION = 1
LOCK_ELEMENTS = 1024


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


@dataclass(frozen=True, kw_only=True)
class SparkWorkspaceRegion:
    name: str
    offset_bytes: int
    size_bytes: int
    dtype: str
    shape: tuple[int, ...]
    zero_before_launch: bool
    purpose: str

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes


@dataclass(frozen=True, kw_only=True)
class SparkWorkspaceLayout:
    alignment: int
    size_bytes: int
    regions: tuple[SparkWorkspaceRegion, ...]

    def region(self, name: str) -> SparkWorkspaceRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


@dataclass(frozen=True, kw_only=True)
class W4A4FC1W4A16FC2SparkSpec:
    """One Spark top-1 capacity specialization.

    ``m`` is compile/workspace capacity. A launch may use any
    ``1 <= active_rows <= m``; CUDA graph replay freezes the launch scalar, so
    an integration either captures per-active-row graphs or executes a padded
    capacity graph.
    """

    m: int
    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    w13_layout: str = "w13"
    planning_sms: int = 48
    tuned_grid_x: int = 48
    target_arch: str = "sm_121"
    fast_math: bool = True

    def __post_init__(self) -> None:
        for name in (
            "m",
            "hidden_size",
            "intermediate_size",
            "planning_sms",
            "tuned_grid_x",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "activation", str(self.activation).lower())
        object.__setattr__(self, "w13_layout", str(self.w13_layout).lower())
        object.__setattr__(self, "target_arch", str(self.target_arch).lower())
        if self.m not in CAPACITY_BUCKETS:
            raise ValueError(f"m must be one of {CAPACITY_BUCKETS}")
        if self.hidden_size <= 0 or self.hidden_size % 128:
            raise ValueError("hidden_size must be a positive multiple of 128")
        if self.intermediate_size <= 0 or self.intermediate_size % 64:
            raise ValueError("intermediate_size must be a positive multiple of 64")
        if self.activation != "silu":
            raise ValueError("the Spark mixed recipe currently requires silu")
        if self.w13_layout != "w13":
            raise ValueError(
                "the Spark mixed recipe requires source [up; gate] W13 order"
            )
        if self.planning_sms <= 0 or self.tuned_grid_x <= 0:
            raise ValueError("planning_sms and tuned_grid_x must be positive")
        if self.target_arch not in {"sm_120", "sm_121"}:
            raise ValueError("target_arch must be 'sm_120' or 'sm_121'")

    @property
    def fc1_cols(self) -> int:
        return 2 * self.intermediate_size

    @property
    def route_block_size(self) -> int:
        return select_route_block_size_m(self.m, 1, 1)

    @property
    def route_slots(self) -> int:
        return max_packed_route_slots(self.m, self.route_block_size, 1)

    @property
    def route_blocks(self) -> int:
        return (self.route_slots + self.route_block_size - 1) // self.route_block_size

    @property
    def fc2_scratch_elements(self) -> int:
        return packed_gemm_scratch_elements(
            size_n=self.hidden_size,
            route_slots=self.route_slots,
            moe_block_size=self.route_block_size,
            sms=self.planning_sms,
        )

    @property
    def symbol(self) -> str:
        return (
            "sparkinfer_w4a4_fc1_w4a16_fc2_spark_"
            f"m{self.m}_h{self.hidden_size}_i{self.intermediate_size}_"
            f"{self.w13_layout}_{self.target_arch}"
        )

    def workspace_layout(self) -> SparkWorkspaceLayout:
        alignment = 256
        regions: list[SparkWorkspaceRegion] = []
        offset = 0

        def add(
            name: str,
            size_bytes: int,
            dtype: str,
            shape: tuple[int, ...],
            *,
            zero_before_launch: bool,
            purpose: str,
        ) -> None:
            nonlocal offset
            offset = _align_up(offset, alignment)
            region = SparkWorkspaceRegion(
                name=name,
                offset_bytes=offset,
                size_bytes=int(size_bytes),
                dtype=dtype,
                shape=shape,
                zero_before_launch=zero_before_launch,
                purpose=purpose,
            )
            regions.append(region)
            offset = region.end_bytes

        add(
            "fc1_bf16",
            self.m * self.fc1_cols * 2,
            "bfloat16",
            (self.m, self.fc1_cols, 1),
            zero_before_launch=False,
            purpose="source-order W4A4 FC1 output",
        )
        add(
            "activated_bf16",
            self.m * self.intermediate_size * 2,
            "bfloat16",
            (self.m, self.intermediate_size),
            zero_before_launch=False,
            purpose="BF16 SiLU/product input to packed W4A16 FC2",
        )
        add(
            "packed_route_indices",
            self.route_slots * 4,
            "int32",
            (self.route_slots,),
            zero_before_launch=False,
            purpose="preinitialized top-1 dense routes",
        )
        add(
            "block_expert_ids",
            self.route_blocks * 4,
            "int32",
            (self.route_blocks,),
            zero_before_launch=False,
            purpose="one expert id per packed route block",
        )
        add(
            "packed_route_count",
            4,
            "int32",
            (1,),
            zero_before_launch=False,
            purpose="padded route count for the captured active row count",
        )
        add(
            "topk_weights",
            self.m * 4,
            "float32",
            (self.m,),
            zero_before_launch=False,
            purpose="ABI top-k weights; FC2 does not multiply them",
        )
        add(
            "fc2_scratch",
            self.fc2_scratch_elements * 4,
            "float32",
            (self.fc2_scratch_elements,),
            zero_before_launch=False,
            purpose="packed W4A16 FC2 accumulation scratch",
        )
        add(
            "fc2_locks",
            LOCK_ELEMENTS * 4,
            "int32",
            (LOCK_ELEMENTS,),
            zero_before_launch=True,
            purpose="packed W4A16 FC2 split-work locks",
        )
        return SparkWorkspaceLayout(
            alignment=alignment,
            size_bytes=_align_up(offset, alignment),
            regions=tuple(regions),
        )


@dataclass(frozen=True, kw_only=True)
class BoundW4A4FC1W4A16FC2SparkWorkspace:
    spec: W4A4FC1W4A16FC2SparkSpec
    arena: torch.Tensor
    fc1_bf16: torch.Tensor
    activated_bf16: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    topk_weights: torch.Tensor
    fc2_scratch: torch.Tensor
    fc2_locks: torch.Tensor


def bind_w4a4_fc1_w4a16_fc2_spark_workspace(
    arena: torch.Tensor,
    spec: W4A4FC1W4A16FC2SparkSpec,
) -> BoundW4A4FC1W4A16FC2SparkWorkspace:
    """Bind typed fixed-offset views without a device allocation."""

    if arena.dtype != torch.uint8:
        raise TypeError(f"workspace arena must be uint8, got {arena.dtype}")
    if not arena.is_contiguous():
        raise ValueError("workspace arena must be contiguous")
    layout = spec.workspace_layout()
    if arena.numel() < layout.size_bytes:
        raise ValueError(
            f"workspace arena has {arena.numel()} bytes; "
            f"{layout.size_bytes} bytes are required"
        )
    if arena.data_ptr() % layout.alignment:
        raise ValueError(f"workspace base must be {layout.alignment}-byte aligned")

    def view(name: str, dtype: torch.dtype) -> torch.Tensor:
        region = layout.region(name)
        return (
            arena.narrow(0, region.offset_bytes, region.size_bytes)
            .view(dtype)
            .reshape(region.shape)
        )

    return BoundW4A4FC1W4A16FC2SparkWorkspace(
        spec=spec,
        arena=arena,
        fc1_bf16=view("fc1_bf16", torch.bfloat16),
        activated_bf16=view("activated_bf16", torch.bfloat16),
        packed_route_indices=view("packed_route_indices", torch.int32),
        block_expert_ids=view("block_expert_ids", torch.int32),
        packed_route_count=view("packed_route_count", torch.int32),
        topk_weights=view("topk_weights", torch.float32),
        fc2_scratch=view("fc2_scratch", torch.float32),
        fc2_locks=view("fc2_locks", torch.int32),
    )


def initialize_w4a4_fc1_w4a16_fc2_spark_routes(
    workspace: BoundW4A4FC1W4A16FC2SparkWorkspace,
    spec: W4A4FC1W4A16FC2SparkSpec,
    *,
    active_rows: int,
) -> None:
    """Initialize fixed top-1 route metadata before CUDA graph capture."""

    if workspace.spec != spec:
        raise ValueError("workspace was bound for a different Spark spec")
    active_rows = int(active_rows)
    if not 1 <= active_rows <= spec.m:
        raise ValueError(f"active_rows must be in [1, {spec.m}]")
    workspace.packed_route_indices.fill_(active_rows)
    workspace.packed_route_indices[:active_rows].copy_(
        torch.arange(
            active_rows,
            dtype=torch.int32,
            device=workspace.packed_route_indices.device,
        )
    )
    workspace.block_expert_ids.zero_()
    workspace.packed_route_count.fill_(_align_up(active_rows, spec.route_block_size))
    workspace.topk_weights.fill_(1.0)
    workspace.fc2_locks.zero_()


@dataclass(frozen=True, kw_only=True)
class W4A4FC1W4A16FC2SparkAOTArtifact:
    """Three exact CuTe objects behind one graph-safe launch/export API."""

    spec: W4A4FC1W4A16FC2SparkSpec
    fc1_compiled: object
    activation_compiled: object
    fc2_compiled: object
    _fc1_launch: object
    _activation_result: object
    _fc2_result: object
    fc1_plan: tuple[int, int]
    fc2_plan: tuple[int, int]

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "recipe": "w4a4_fc1_w4a16_fc2_spark",
            "abi_version": SPARK_AOT_ABI_VERSION,
            "symbol": self.spec.symbol,
            "target_arch": self.spec.target_arch,
            "capacity_m": self.spec.m,
            "runtime_active_rows": f"1..{self.spec.m}",
            "w13_layout": self.spec.w13_layout,
            "phases": (
                "source_w13_w4a4_fc1",
                "source_order_bf16_silu_product",
                "packed_w2_w4a16_fc2",
            ),
            "scalar_contract": {
                "input": (
                    "prequantized E2M1 payload with complete per-K16 E4M3 "
                    "scales; no checkpoint input-global multiplier"
                ),
                "w13_runtime_alpha": "raw checkpoint W13 weight_scale_2",
                "w2_w4a16_alpha": (
                    "BF16 W4A16 compensated W2 weight_scale_2 from packed prep"
                ),
            },
            "weights": {
                "storage_policy": "source_w13_packed_w2_in_place",
                "w13": "one source-layout allocation",
                "w2": "one packed-layout allocation aliasing source staging",
                "packed_w13": False,
                "source_w2": False,
            },
            "workspace": asdict(self.spec.workspace_layout()),
            "launch": {
                "planning_sms": self.spec.planning_sms,
                "tuned_grid_x": self.spec.tuned_grid_x,
                "hardware_sm_count": "reported separately by the integration",
                "cooperative_grid_barrier": False,
                "fc1_tile_mn": self.fc1_plan,
                "fc2_tile_kn": self.fc2_plan,
                "materialized_intermediate_bytes": (
                    self.spec.m * (self.spec.fc1_cols + self.spec.intermediate_size) * 2
                ),
                "reorder_buffer_bytes": 0,
            },
        }

    def export_to_c(
        self,
        output_dir: str | Path,
        *,
        file_name: str | None = None,
        symbol_prefix: str = "",
        symbol_base: str | None = None,
    ) -> Path:
        """Export all phase objects and one composition manifest."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        base = self.spec.symbol if file_name is None else str(file_name)
        exported_base = (
            f"{symbol_prefix}{self.spec.symbol}"
            if symbol_base is None
            else str(symbol_base)
        )
        self.fc1_compiled.export_to_c(
            str(output),
            f"{base}_fc1",
            f"{exported_base}_fc1",
        )
        self.activation_compiled.export_to_c(
            str(output),
            f"{base}_activation",
            f"{exported_base}_activation",
        )
        self.fc2_compiled.export_to_c(
            str(output),
            f"{base}_fc2",
            f"{exported_base}_fc2",
        )
        manifest = output / f"{base}.json"
        manifest_payload = dict(self.metadata)
        manifest_payload["export"] = {
            "file_base": base,
            "symbol_base": exported_base,
            "phases": {
                phase: {
                    "header": f"{base}_{phase}.h",
                    "object": f"{base}_{phase}.o",
                    "symbol": f"{exported_base}_{phase}",
                }
                for phase in ("fc1", "activation", "fc2")
            },
        }
        manifest.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return manifest

    def launch(
        self,
        *,
        input_packed: torch.Tensor,
        input_scale: torch.Tensor,
        weights: BoundW4A4FC1W4A16FC2Expert,
        workspace: BoundW4A4FC1W4A16FC2SparkWorkspace,
        output: torch.Tensor,
        active_rows: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """Enqueue the fixed three-stage sequence without allocating."""

        self.launch_fc1(
            input_packed=input_packed,
            input_scale=input_scale,
            weights=weights,
            workspace=workspace,
            active_rows=active_rows,
            stream=stream,
        )
        self.launch_activation(
            workspace=workspace,
            active_rows=active_rows,
            stream=stream,
        )
        self.launch_fc2(
            weights=weights,
            workspace=workspace,
            output=output,
            active_rows=active_rows,
            stream=stream,
        )

    def _validate_workspace(
        self,
        workspace: BoundW4A4FC1W4A16FC2SparkWorkspace,
    ) -> None:
        if workspace.spec != self.spec:
            raise ValueError("workspace was bound for a different Spark spec")

    def _validate_active_rows(self, active_rows: int) -> int:
        active_rows = int(active_rows)
        if not 1 <= active_rows <= self.spec.m:
            raise ValueError(f"active_rows must be in [1, {self.spec.m}]")
        return active_rows

    def launch_fc1(
        self,
        *,
        input_packed: torch.Tensor,
        input_scale: torch.Tensor,
        weights: BoundW4A4FC1W4A16FC2Expert,
        workspace: BoundW4A4FC1W4A16FC2SparkWorkspace,
        active_rows: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """Launch the exact source-layout W4A4 FC1 phase."""

        self._validate_workspace(workspace)
        active_rows = self._validate_active_rows(active_rows)
        expected_input_shape = (active_rows, self.spec.hidden_size // 2, 1)
        if tuple(input_packed.shape) != expected_input_shape:
            raise ValueError(
                f"input_packed must have shape {expected_input_shape}, "
                f"got {tuple(input_packed.shape)}"
            )
        fc1 = workspace.fc1_bf16[:active_rows]
        self._fc1_launch(
            a_tensor_gpu=input_packed,
            b_tensor_gpu=weights.w13_weight_source,
            sfa_tensor_gpu=input_scale,
            sfb_tensor_gpu=weights.w13_scale_source,
            c_tensor_gpu=fc1,
            alpha_tensor_gpu=weights.w13_runtime_alpha,
            stream_int=int(stream.cuda_stream),
        )

    def launch_activation(
        self,
        *,
        workspace: BoundW4A4FC1W4A16FC2SparkWorkspace,
        active_rows: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """Launch source-order BF16 SiLU/product without a reorder buffer."""

        self._validate_workspace(workspace)
        active_rows = self._validate_active_rows(active_rows)
        fc1 = workspace.fc1_bf16[:active_rows]
        activated = workspace.activated_bf16[:active_rows]
        self._activation_result.compiled(
            fc1.view(-1),
            activated.view(-1),
            active_rows,
            cuda.CUstream(int(stream.cuda_stream)),
        )

    def launch_fc2(
        self,
        *,
        weights: BoundW4A4FC1W4A16FC2Expert,
        workspace: BoundW4A4FC1W4A16FC2SparkWorkspace,
        output: torch.Tensor,
        active_rows: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """Launch packed-layout BF16 W4A16 FC2."""

        self._validate_workspace(workspace)
        active_rows = self._validate_active_rows(active_rows)
        if output.dtype != torch.bfloat16 or output.numel() < (
            active_rows * self.spec.hidden_size
        ):
            raise ValueError("output must be a sufficiently large BF16 tensor")
        activated = workspace.activated_bf16[:active_rows]
        workspace.fc2_locks.zero_()
        element = _cutlass_element_dtype("bf16")
        a_pointer = make_ptr(
            element,
            activated.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        self._fc2_result.compiled(
            a_pointer,
            a_pointer,
            weights.w2_weight_packed.view(torch.int32).view(-1),
            make_ptr(
                element,
                output.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            weights.w2_scale_packed.view(torch.uint8).view(torch.int32).view(-1),
            weights.w2_w4a16_alpha,
            workspace.packed_route_indices,
            workspace.block_expert_ids,
            workspace.packed_route_count,
            make_ptr(
                cutlass.Float32,
                workspace.topk_weights.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            workspace.fc2_scratch,
            workspace.fc2_locks,
            active_rows,
            self.spec.tuned_grid_x,
            cuda.CUstream(int(stream.cuda_stream)),
        )


_SPARK_COMPILE_CACHE: dict[
    tuple[object, ...],
    W4A4FC1W4A16FC2SparkAOTArtifact,
] = {}


def _compile_source_w4a4_fc1(
    spec: W4A4FC1W4A16FC2SparkSpec,
) -> tuple[object, object, tuple[int, int]]:
    plan = dense._select_default_dense_gemm_plan(
        spec.m,
        spec.fc1_cols,
        spec.hidden_size,
        spec.planning_sms,
        is_mxfp8=False,
        is_mxfp6=False,
        expected_m=spec.m,
    )
    policy = dense._dense_gemm_policy_for(
        m=spec.m,
        n=spec.fc1_cols,
        k=spec.hidden_size,
        l=1,
        ab_dtype=cutlass.Float4E2M1FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=spec.planning_sms,
        expected_m=spec.m,
    )
    if policy.split_k_slices != 1:
        raise RuntimeError("mixed Spark FC1 requires a single-slice dense plan")
    launch = dense._get_compiled_dense_gemm(
        n=spec.fc1_cols,
        k=spec.hidden_size,
        l=1,
        c_l=1,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=16,
        mma_k=64,
        tile_k=128,
        mma_tiler_mn=plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=spec.planning_sms,
        sm_version=spec.target_arch,
        load_path=plan.load_path,
        swap_ab=plan.swap_ab,
        sfb_k_reuse=False,
        b_tile_major=False,
        alpha_is_one=False,
        direct_sfa_live16=False,
    )
    compiled = getattr(launch, "_sparkinfer_compiled_kernel", None)
    if compiled is None:
        raise RuntimeError("dense FC1 compile did not expose its AOT object")
    return launch, compiled, plan.mma_tiler_mn


def compile_w4a4_fc1_w4a16_fc2_spark_aot(
    spec: W4A4FC1W4A16FC2SparkSpec,
    *,
    device: torch.device | str | None = None,
) -> W4A4FC1W4A16FC2SparkAOTArtifact:
    """Compile the exact source-W13/packed-W2 Spark composition."""

    if not isinstance(spec, W4A4FC1W4A16FC2SparkSpec):
        raise TypeError("spec must be W4A4FC1W4A16FC2SparkSpec")
    device_obj = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if device_obj.type != "cuda":
        raise ValueError("mixed Spark AOT compilation requires a CUDA device")
    device_index = int(
        torch.cuda.current_device() if device_obj.index is None else device_obj.index
    )
    capability = torch.cuda.get_device_capability(device_index)
    device_arch = f"sm_{capability[0]}{capability[1]}"
    if spec.target_arch != device_arch:
        raise ValueError(
            f"spec targets {spec.target_arch}, but compile device is {device_arch}; "
            "export release objects on the exact target architecture"
        )
    cache_key = (device_index, spec)
    cached = _SPARK_COMPILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    properties = torch.cuda.get_device_properties(device_index)
    fc1_launch, fc1_compiled, fc1_plan = _compile_source_w4a4_fc1(spec)
    activation = compile_w4a16_activation(
        rows=spec.m,
        intermediate_size=spec.intermediate_size,
        activation=spec.activation,
        element_dtype="bf16",
        fast_math=spec.fast_math,
        w13_layout=spec.w13_layout,
    )
    tile_k, tile_n, _, _ = _select_tile_config(
        problem_m=spec.m,
        problem_n=spec.hidden_size,
        problem_k=spec.intermediate_size,
        top_k=1,
        moe_block_size=spec.route_block_size,
        sms=spec.planning_sms,
        max_shared_mem=int(properties.shared_memory_per_block_optin),
        scale_format="e4m3_k16",
    )
    fc2 = compile_w4a16_gemm(
        size_m=spec.m,
        size_n=spec.hidden_size,
        size_k=spec.intermediate_size,
        num_experts=1,
        top_k=1,
        mul_topk_weights=False,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=spec.route_block_size,
        max_m_blocks=spec.route_blocks,
        element_dtype="bf16",
        scale_format="e4m3_k16",
    )
    artifact = W4A4FC1W4A16FC2SparkAOTArtifact(
        spec=spec,
        fc1_compiled=fc1_compiled,
        activation_compiled=activation.compiled,
        fc2_compiled=fc2.compiled,
        _fc1_launch=fc1_launch,
        _activation_result=activation,
        _fc2_result=fc2,
        fc1_plan=fc1_plan,
        fc2_plan=(tile_k, tile_n),
    )
    _SPARK_COMPILE_CACHE[cache_key] = artifact
    return artifact


def clear_w4a4_fc1_w4a16_fc2_spark_cache() -> None:
    _SPARK_COMPILE_CACHE.clear()


__all__ = [
    "CAPACITY_BUCKETS",
    "SPARK_AOT_ABI_VERSION",
    "BoundW4A4FC1W4A16FC2SparkWorkspace",
    "SparkWorkspaceLayout",
    "SparkWorkspaceRegion",
    "W4A4FC1W4A16FC2SparkAOTArtifact",
    "W4A4FC1W4A16FC2SparkSpec",
    "bind_w4a4_fc1_w4a16_fc2_spark_workspace",
    "clear_w4a4_fc1_w4a16_fc2_spark_cache",
    "compile_w4a4_fc1_w4a16_fc2_spark_aot",
    "initialize_w4a4_fc1_w4a16_fc2_spark_routes",
]
