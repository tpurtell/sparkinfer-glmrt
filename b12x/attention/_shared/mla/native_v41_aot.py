"""Native pointer ABI for direct DS41RT FP4 attention plus sink merge.

Internal AOT surface: host integration validates all pointers, capacities,
alignment, immutable leases and scratch disjointness. All row counts are live
launch arguments. No allocation, compilation or host synchronization in replay.
"""
from dataclasses import dataclass
import math
import os

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Uint64

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from .kernel import UnifiedDecodeKernel
from .merge import SparseMLASplitDecodeSinkMergeKernel
from .smem import make_native_v41_compact_smem_layout, make_smem_layout
from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits

_UNSET = object()


@dataclass(frozen=True)
class SmOccupancyLimits:
    num_sms: int
    max_smem_per_mp: int
    max_smem_per_block: int
    max_smem_per_block_optin: int
    reserved_smem_per_block: int
    max_regs_per_mp: int
    max_regs_per_block: int
    max_threads_per_mp: int
    max_blocks_per_mp: int
    warp_size: int


def query_sm_occupancy_limits(device: int | None = None) -> SmOccupancyLimits | None:
    """Query runtime CUDA device attributes without hardcoded SM/SMEM/reg limits."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return None
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = int(torch.cuda.current_device())
    try:
        init_err, = cuda.cuInit(0)
        if init_err != cuda.CUresult.CUDA_SUCCESS:
            return None
        err, cu_dev = cuda.cuDeviceGet(int(device))
        if err != cuda.CUresult.CUDA_SUCCESS:
            return None

        def _attr(name: str) -> int:
            res, val = cuda.cuDeviceGetAttribute(
                getattr(cuda.CUdevice_attribute, name), cu_dev
            )
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuDeviceGetAttribute({name}) failed with {res!r}")
            return int(val)

        num_sms = _attr("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT")
        if num_sms <= 0:
            return None
        return SmOccupancyLimits(
            num_sms=num_sms,
            max_smem_per_mp=_attr("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"),
            max_smem_per_block=_attr("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK"),
            max_smem_per_block_optin=_attr("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"),
            reserved_smem_per_block=_attr("CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK"),
            max_regs_per_mp=_attr("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR"),
            max_regs_per_block=_attr("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK"),
            max_threads_per_mp=_attr("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR"),
            max_blocks_per_mp=_attr("CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR"),
            warp_size=_attr("CU_DEVICE_ATTRIBUTE_WARP_SIZE"),
        )
    except (RuntimeError, OSError):
        return None


def compute_cubin_occupancy_from_attrs(
    *,
    regs_per_thread: int,
    static_smem_bytes: int,
    dynamic_smem_bytes: int,
    block_threads: int,
    limits: SmOccupancyLimits,
    reg_alloc_granularity: int = 8,
) -> dict[str, int]:
    """Derive achieved CTAs/SM and limiting factors purely from cubin + device attrs."""
    if dynamic_smem_bytes > limits.max_smem_per_block_optin:
        return {
            "active_ctas_per_sm": 0,
            "thread_ctas": 0,
            "smem_ctas": 0,
            "reg_ctas": 0,
            "rounded_regs_per_thread": 0,
            "cta_smem_footprint": dynamic_smem_bytes + max(static_smem_bytes, limits.reserved_smem_per_block),
        }
    warps_per_cta = (block_threads + limits.warp_size - 1) // limits.warp_size
    rounded_regs = (
        (regs_per_thread + reg_alloc_granularity - 1)
        // reg_alloc_granularity
        * reg_alloc_granularity
    )
    regs_per_cta = rounded_regs * warps_per_cta * limits.warp_size
    reg_ctas = (
        limits.max_regs_per_mp // regs_per_cta
        if regs_per_cta <= limits.max_regs_per_block and regs_per_cta > 0
        else 0
    )
    cta_smem = dynamic_smem_bytes + max(static_smem_bytes, limits.reserved_smem_per_block)
    smem_ctas = limits.max_smem_per_mp // cta_smem if cta_smem > 0 else limits.max_blocks_per_mp
    thread_ctas = limits.max_threads_per_mp // block_threads
    active_ctas = min(limits.max_blocks_per_mp, thread_ctas, smem_ctas, reg_ctas)
    return {
        "active_ctas_per_sm": active_ctas,
        "thread_ctas": thread_ctas,
        "smem_ctas": smem_ctas,
        "reg_ctas": reg_ctas,
        "rounded_regs_per_thread": rounded_regs,
        "cta_smem_footprint": cta_smem,
    }


def resolve_native_v41_cta_placement(
    *,
    block_threads: int,
    compact_smem_bytes: int,
    full_smem_bytes: int,
    limits: SmOccupancyLimits | None,
) -> tuple[bool, int]:
    """Select (native_v41_2cta, min_blocks_per_mp) from runtime device attributes.

    When device attributes are unavailable, select R1 full-tile (False, 1).
    """
    if limits is None:
        return False, 1
    warps_per_cta = (block_threads + limits.warp_size - 1) // limits.warp_size
    cta_smem_2 = compact_smem_bytes + limits.reserved_smem_per_block
    smem_fits_2 = (
        compact_smem_bytes <= limits.max_smem_per_block_optin
        and 2 * cta_smem_2 <= limits.max_smem_per_mp
    )
    threads_fit_2 = 2 * block_threads <= limits.max_threads_per_mp
    blocks_fit_2 = limits.max_blocks_per_mp >= 2
    reg_budget_2 = ((limits.max_regs_per_mp // (2 * warps_per_cta)) // limits.warp_size) & ~7
    regs_fit_2 = reg_budget_2 > 0 and (reg_budget_2 * warps_per_cta * limits.warp_size) <= limits.max_regs_per_block
    if smem_fits_2 and threads_fit_2 and blocks_fit_2 and regs_fit_2:
        return True, 2
    if full_smem_bytes <= limits.max_smem_per_block_optin:
        return False, 1
    return True, 1


def compute_native_v41_compact_threshold_rows(
    *,
    h_blocks: int,
    num_splits: int,
    limits: SmOccupancyLimits | None,
    block_threads: int | None = None,
    compact_smem_bytes: int | None = None,
    full_smem_bytes: int | None = None,
) -> int | None:
    """Compute live-row crossover ceil(num_sms / (h_blocks * num_splits)) from device attrs.

    Returns None when runtime attributes are unavailable or 2 CTAs/SM cannot fit,
    selecting the R1 full-tile path across all live-row buckets.
    """
    if h_blocks <= 0 or num_splits <= 0:
        raise ValueError("h_blocks and num_splits must be positive")
    if limits is None or limits.num_sms <= 0:
        return None
    if block_threads is None or compact_smem_bytes is None or full_smem_bytes is None:
        traits = make_unified_traits(
            ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3, fp8_rope=False
        )
        if block_threads is None:
            block_threads = traits.block_threads
        if compact_smem_bytes is None:
            compact_smem_bytes = make_native_v41_compact_smem_layout(traits).total_bytes
        if full_smem_bytes is None:
            full_smem_bytes = make_smem_layout(traits).total_bytes
    use_compact_2cta, min_blocks_per_mp = resolve_native_v41_cta_placement(
        block_threads=block_threads,
        compact_smem_bytes=compact_smem_bytes,
        full_smem_bytes=full_smem_bytes,
        limits=limits,
    )
    if not (use_compact_2cta and min_blocks_per_mp >= 2):
        return None
    ctas_per_row = h_blocks * num_splits
    return (limits.num_sms + ctas_per_row - 1) // ctas_per_row


def select_native_v41_variant_for_rows(
    bucket_rows: int,
    threshold_rows: int | None,
) -> str:
    """Bind a concrete live-row bucket to 'full' (R1) or 'compact' (R2)."""
    rows_int = int(bucket_rows)
    if rows_int < 1:
        raise ValueError(f"bucket_rows must be >= 1, got {bucket_rows}")
    if threshold_rows is not None and rows_int >= threshold_rows:
        return "compact"
    return "full"


class NativeV41Attention:
    def __init__(
        self,
        heads: int = 64,
        *,
        device: int | None = None,
        variant: str | None = None,
        bucket_rows: int | None = None,
        limits: SmOccupancyLimits | None | object = _UNSET,
    ):
        if heads not in (32, 64):
            raise ValueError("native V4.1 attention requires 32 or 64 local heads")
        if variant not in (None, "full", "compact"):
            raise ValueError(f"variant must be None, 'full', or 'compact', got {variant!r}")
        if variant is not None and bucket_rows is not None:
            raise ValueError("specify at most one of variant or bucket_rows")
        self.heads = heads
        # Keep checkpoint-native scales and BF16 QK/PV for decode/verify,
        # matching the prefill arithmetic without canonical FP8 requantization.
        traits = make_unified_traits(ModelType.DSV41, ComputeMode.BF16,
                                     ScaleFormat.NVFP4_E4M3, fp8_rope=False)
        compact_layout = make_native_v41_compact_smem_layout(traits)
        full_layout = make_smem_layout(traits)
        resolved_limits = (
            query_sm_occupancy_limits(device) if limits is _UNSET else limits
        )
        self.threshold_rows = compute_native_v41_compact_threshold_rows(
            h_blocks=heads // 16,
            num_splits=10,
            limits=resolved_limits,
            block_threads=traits.block_threads,
            compact_smem_bytes=compact_layout.total_bytes,
            full_smem_bytes=full_layout.total_bytes,
        )
        if variant is not None:
            chosen = variant
        elif bucket_rows is not None:
            chosen = select_native_v41_variant_for_rows(bucket_rows, self.threshold_rows)
        else:
            chosen = "full"

        if chosen == "compact":
            if resolved_limits is None:
                use_compact_2cta, min_blocks_per_mp = True, 2
            else:
                cand_compact, cand_min_blocks = resolve_native_v41_cta_placement(
                    block_threads=traits.block_threads,
                    compact_smem_bytes=compact_layout.total_bytes,
                    full_smem_bytes=full_layout.total_bytes,
                    limits=resolved_limits,
                )
                use_compact_2cta = True
                min_blocks_per_mp = 2 if (cand_compact and cand_min_blocks >= 2) else 1
        else:
            if (
                resolved_limits is not None
                and full_layout.total_bytes > resolved_limits.max_smem_per_block_optin
            ):
                use_compact_2cta, min_blocks_per_mp = True, 1
            else:
                use_compact_2cta, min_blocks_per_mp = False, 1

        self.variant = "compact" if use_compact_2cta else "full"
        layout = compact_layout if use_compact_2cta else full_layout
        carveout = (
            int(cuda.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_SHARED)
            if resolved_limits is None and min_blocks_per_mp > 1
            else None
        )
        self.decode = UnifiedDecodeKernel(traits, layout, 64, 1,
            h_blocks=heads//16, num_splits=10, num_heads=heads, q_head_dim=512,
            topk=128, extra_topk=512, q_stride=(heads*512, 512, 1),
            swa_indices_stride0=512, extra_indices_stride0=512,
            mid_out_stride=(heads*5120, 5120, 512, 1), mid_lse_stride=(heads*10, 10, 1),
            has_extra=True, pbs_extra=256, valid_hpb=16, vector_q=True,
            native_v41_2cta=use_compact_2cta, min_blocks_per_mp=min_blocks_per_mp,
            preferred_smem_carveout=carveout)
        self.merge = SparseMLASplitDecodeSinkMergeKernel(static_num_chunks=10)

    @classmethod
    def for_rows(
        cls,
        heads: int,
        bucket_rows: int,
        *,
        device: int | None = None,
        limits: SmOccupancyLimits | None | object = _UNSET,
    ) -> "NativeV41Attention":
        return cls(heads, device=device, bucket_rows=bucket_rows, limits=limits)

    @cute.jit
    def __call__(self, query: cute.Pointer, descriptors: cute.Pointer,
                 metadata: cute.Pointer, selected: cute.Pointer, bounds: cute.Pointer,
                 sink: cute.Pointer, partials: cute.Pointer, lses: cute.Pointer,
                 output: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        q = cute.make_tensor(query, cute.make_layout((rows, self.heads, 512), stride=(self.heads*512, 512, 1)))
        desc = cute.make_tensor(descriptors, cute.make_layout((rows, 15), stride=(15, 1)))
        meta = cute.make_tensor(metadata, cute.make_layout((rows, 10), stride=(10, 1)))
        ids = cute.make_tensor(selected, cute.make_layout((rows, 512), stride=(512, 1)))
        begins = cute.make_tensor(bounds, cute.make_layout((rows,), stride=(1,)))
        p = cute.make_tensor(partials, cute.make_layout((rows, self.heads, 10, 512), stride=(self.heads*5120, 5120, 512, 1)))
        lse = cute.make_tensor(lses, cute.make_layout((rows, self.heads, 10), stride=(self.heads*10, 10, 1)))
        sinks = cute.make_tensor(sink, cute.make_layout((self.heads,), stride=(1,)))
        out = cute.make_tensor(output, cute.make_layout((rows, self.heads, 512), stride=(self.heads*512, 512, 1)))
        self.decode.call_native_v41(q, desc, meta, ids, begins, p, lse,
                                    Float32(512**-.5 * math.log2(math.e)), rows, stream)
        # Chunk-count tensor is unused by the static 10-part merge.
        self.merge(p, lse, begins, sinks, out, stream)


class NativeV41BucketDispatcher:
    """Host-side live-row bucket dispatcher between precompiled R1 and R2 variants.

    Crossover ``threshold_rows`` is computed once at startup from runtime device
    attributes. Selection runs on the CPU host when building/capturing the launch
    for a concrete row bucket, baking exactly one precompiled path into each
    bucket's CUDA graph without any @cute.jit branch or device conditional.
    """

    def __init__(
        self,
        *,
        heads: int,
        threshold_rows: int | None,
        full,
        compact,
        full_key: tuple[object, ...],
        compact_key: tuple[object, ...] | None,
    ):
        self.heads = heads
        self.threshold_rows = threshold_rows
        self.full = full
        self.compact = compact
        self.full_key = full_key
        self.compact_key = compact_key
        carriers = (full,) if compact is None else (full, compact)
        attach_programs(self, *carriers)

    def variant_for_rows(self, bucket_rows: int) -> str:
        if self.compact is None:
            if int(bucket_rows) < 1:
                raise ValueError(f"bucket_rows must be >= 1, got {bucket_rows}")
            return "full"
        return select_native_v41_variant_for_rows(bucket_rows, self.threshold_rows)

    def for_rows(self, bucket_rows: int):
        """Return the single precompiled variant bound to ``bucket_rows``."""
        return (
            self.compact
            if self.variant_for_rows(bucket_rows) == "compact"
            else self.full
        )

    def __call__(
        self,
        query,
        descriptors,
        metadata,
        selected,
        bounds,
        sink,
        partials,
        lses,
        output,
        rows,
        stream,
    ):
        try:
            bucket_rows = int(rows)
        except Exception as exc:
            raise TypeError(
                "NativeV41BucketDispatcher requires a concrete host row count at "
                "bucket launch-build/capture time; do not trace it inside @cute.jit"
            ) from exc
        target = self.for_rows(bucket_rows)
        return target(
            query,
            descriptors,
            metadata,
            selected,
            bounds,
            sink,
            partials,
            lses,
            output,
            rows,
            stream,
        )

    def export_to_c(self, *args, **kwargs):
        """Export single-stem R1 full-tile fallback (never slower than R1)."""
        return self.full.export_to_c(*args, **kwargs)


def _compile_native_v41_variant(
    launch: NativeV41Attention,
    *,
    device: int | None,
):
    key = (
        launch.heads,
        512,
        128,
        512,
        10,
        device,
        launch.decode.native_v41_2cta,
        launch.decode.min_blocks_per_mp,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    types = (BFloat16, Uint64, Uint64, Int32, Uint64, Float32, BFloat16, Float32, BFloat16)
    alignments = (16, 8, 8, 4, 8, 4, 16, 4, 16)
    pointers = tuple(
        make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)
        for dtype, align in zip(types, alignments)
    )
    compiled = compile_cute(
        launch,
        *pointers,
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("attention.native_v41", 4, key),
    )
    return compiled, key


def compile_native_v41_attention_aot(
    heads: int = 64,
    *,
    variant: str | None = None,
    bucket_rows: int | None = None,
):
    """Compile static local-head variants ahead of time; live row counts never enter keys."""
    if variant not in (None, "full", "compact"):
        raise ValueError(f"variant must be None, 'full', or 'compact', got {variant!r}")
    if variant is not None and bucket_rows is not None:
        raise ValueError("specify at most one of variant or bucket_rows")
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    limits = query_sm_occupancy_limits(device)
    if variant is not None:
        launch = NativeV41Attention(heads, device=device, variant=variant, limits=limits)
        compiled, _ = _compile_native_v41_variant(launch, device=device)
        return compiled

    full_launch = NativeV41Attention(heads, device=device, variant="full", limits=limits)
    full_compiled, full_key = _compile_native_v41_variant(full_launch, device=device)
    threshold_rows = full_launch.threshold_rows
    if threshold_rows is not None:
        compact_launch = NativeV41Attention(
            heads, device=device, variant="compact", limits=limits
        )
        compact_compiled, compact_key = _compile_native_v41_variant(
            compact_launch, device=device
        )
    else:
        compact_compiled, compact_key = None, None

    dispatcher = NativeV41BucketDispatcher(
        heads=heads,
        threshold_rows=threshold_rows,
        full=full_compiled,
        compact=compact_compiled,
        full_key=full_key,
        compact_key=compact_key,
    )
    if bucket_rows is not None:
        return dispatcher.for_rows(bucket_rows)
    return dispatcher
