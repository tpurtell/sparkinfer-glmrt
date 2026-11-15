"""Native pointer ABI for direct DS41RT FP4 attention plus sink merge.

Internal AOT surface: host integration validates all pointers, capacities,
alignment, immutable leases and scratch disjointness. All row counts are live
launch arguments. No allocation, compilation or host synchronization in replay.
"""
from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from .kernel import UnifiedDecodeKernel
from .merge import SparseMLASplitDecodeSinkMergeKernel
from .smem import make_native_v41_compact_smem_layout, make_smem_layout
from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits

_UNSET = object()
_FALLBACK_THRESHOLD_ROWS = 2147483647


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
    """Query runtime CUDA device attributes without hardcoded SM/SMEM/reg limits.

    Strictly non-throwing across CPU/offline containers without libcuda.so.1:
    returns None on any raised Exception or non-CUDA_SUCCESS driver status.
    """
    try:
        if device is None:
            if not torch.cuda.is_available():
                return None
            device = int(torch.cuda.current_device())
        init_res = cuda.cuInit(0)
        init_err = init_res[0] if isinstance(init_res, tuple) else init_res
        if init_err != cuda.CUresult.CUDA_SUCCESS:
            return None
        dev_res = cuda.cuDeviceGet(int(device))
        if not isinstance(dev_res, tuple) or len(dev_res) < 2:
            return None
        err, cu_dev = dev_res[0], dev_res[1]
        if err != cuda.CUresult.CUDA_SUCCESS:
            return None

        def _attr(name: str) -> int | None:
            attr_enum = getattr(cuda.CUdevice_attribute, name)
            attr_res = cuda.cuDeviceGetAttribute(attr_enum, cu_dev)
            if not isinstance(attr_res, tuple) or len(attr_res) < 2:
                return None
            res, val = attr_res[0], attr_res[1]
            if res != cuda.CUresult.CUDA_SUCCESS:
                return None
            return int(val)

        num_sms = _attr("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT")
        max_smem_per_mp = _attr("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR")
        max_smem_per_block = _attr("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK")
        max_smem_per_block_optin = _attr("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
        reserved_smem_per_block = _attr("CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK")
        max_regs_per_mp = _attr("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR")
        max_regs_per_block = _attr("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK")
        max_threads_per_mp = _attr("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR")
        max_blocks_per_mp = _attr("CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR")
        warp_size = _attr("CU_DEVICE_ATTRIBUTE_WARP_SIZE")
        values = (
            num_sms,
            max_smem_per_mp,
            max_smem_per_block,
            max_smem_per_block_optin,
            reserved_smem_per_block,
            max_regs_per_mp,
            max_regs_per_block,
            max_threads_per_mp,
            max_blocks_per_mp,
            warp_size,
        )
        if any(v is None for v in values) or num_sms <= 0 or warp_size <= 0:
            return None
        return SmOccupancyLimits(
            num_sms=num_sms,
            max_smem_per_mp=max_smem_per_mp,
            max_smem_per_block=max_smem_per_block,
            max_smem_per_block_optin=max_smem_per_block_optin,
            reserved_smem_per_block=reserved_smem_per_block,
            max_regs_per_mp=max_regs_per_mp,
            max_regs_per_block=max_regs_per_block,
            max_threads_per_mp=max_threads_per_mp,
            max_blocks_per_mp=max_blocks_per_mp,
            warp_size=warp_size,
        )
    except Exception:
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


def _safe_current_cuda_device() -> int | None:
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.current_device())
    except Exception:
        return None


def _safe_compile_cuda_stream() -> cuda.CUstream:
    try:
        return current_cuda_stream()
    except Exception:
        return cuda.CUstream(0)


def _can_query_cutlass_max_smem_per_mp() -> bool:
    """Return True iff CuTe's compile-time max_smem_per_mp query succeeds."""
    try:
        from cutlass.cutlass_dsl import cuda_helpers

        val = cuda_helpers.get_device_attribute(
            cuda_helpers.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )
        return int(val) > 0
    except Exception:
        return False


def _find_builtin_module():
    mod = ir.InsertionPoint.current.block.owner
    while mod.operation.name != "builtin.module":
        mod = mod.operation.parent
    return mod


def _ensure_host_threshold_decls(mod, global_name: str | None = None, *, loc=None):
    declarations = {
        "_cudaGetDevice": "!llvm.func<i32 (ptr)>",
        "_cudaDeviceGetAttribute": "!llvm.func<i32 (ptr, i32, i32)>",
    }
    symbols = ir.SymbolTable(mod)
    arr_ty = ir.Type.parse("!llvm.array<32 x i32>")
    with ir.InsertionPoint(mod.regions[0].blocks[0]):
        for name, sig in declarations.items():
            if name not in symbols:
                llvm.func(name, ir.TypeAttr.get(ir.Type.parse(sig)), loc=loc)
        if global_name is not None and global_name not in symbols:
            zero_attr = ir.DenseElementsAttr.get_splat(
                ir.RankedTensorType.get([32], T.i32()),
                ir.IntegerAttr.get(T.i32(), 0),
            )
            linkage = ir.Attribute.parse("#llvm.linkage<internal>")
            llvm.GlobalOp(
                arr_ty,
                global_name,
                linkage,
                value=zero_attr,
                alignment=4,
                loc=loc,
            )


@dsl_user_op
def _host_load_cached_threshold(global_name: str, *, loc=None, ip=None):
    mod = _find_builtin_module()
    _ensure_host_threshold_decls(mod, global_name, loc=loc)
    ptr_ty = ir.Type.parse("!llvm.ptr")
    i32 = T.i32()
    one = Int32(1).ir_value(loc=loc, ip=ip)
    zero = Int32(0).ir_value(loc=loc, ip=ip)
    fallback = Int32(_FALLBACK_THRESHOLD_ROWS).ir_value(loc=loc, ip=ip)
    thirty_two = Int32(32).ir_value(loc=loc, ip=ip)
    neg_one = Int32(-1).ir_value(loc=loc, ip=ip)

    dev_ptr = llvm.alloca(ptr_ty, one, i32, alignment=4, loc=loc, ip=ip)
    llvm.store(neg_one, dev_ptr, loc=loc, ip=ip)
    err = llvm.call(i32, [dev_ptr], [], [], callee="_cudaGetDevice", loc=loc, ip=ip)
    dev = llvm.load(i32, dev_ptr, loc=loc, ip=ip)

    err_ok = arith.cmpi(arith.CmpIPredicate.eq, err, zero, loc=loc, ip=ip)
    dev_ge0 = arith.cmpi(arith.CmpIPredicate.sge, dev, zero, loc=loc, ip=ip)
    dev_lt32 = arith.cmpi(arith.CmpIPredicate.slt, dev, thirty_two, loc=loc, ip=ip)
    valid = arith.andi(err_ok, arith.andi(dev_ge0, dev_lt32, loc=loc, ip=ip), loc=loc, ip=ip)

    safe_idx = arith.select(valid, dev, zero, loc=loc, ip=ip)
    base_ptr = llvm.AddressOfOp(ptr_ty, global_name, loc=loc, ip=ip).result
    slot_ptr = llvm.getelementptr(
        ptr_ty,
        base_ptr,
        [safe_idx],
        [-2147483648],
        i32,
        llvm.GEPNoWrapFlags.inbounds,
        loc=loc,
        ip=ip,
    )
    loaded = llvm.load(
        i32,
        slot_ptr,
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        loc=loc,
        ip=ip,
    )
    out_thr = arith.select(valid, loaded, fallback, loc=loc, ip=ip)
    out_dev = arith.select(valid, dev, neg_one, loc=loc, ip=ip)
    return Int32(out_dev), Int32(out_thr)


@dsl_user_op
def _host_store_cached_threshold(
    global_name: str, dev: Int32, threshold: Int32, *, loc=None, ip=None
):
    mod = _find_builtin_module()
    _ensure_host_threshold_decls(mod, global_name, loc=loc)
    ptr_ty = ir.Type.parse("!llvm.ptr")
    i32 = T.i32()
    dev_val = dev.ir_value(loc=loc, ip=ip)
    thr_val = threshold.ir_value(loc=loc, ip=ip)
    base_ptr = llvm.AddressOfOp(ptr_ty, global_name, loc=loc, ip=ip).result
    slot_ptr = llvm.getelementptr(
        ptr_ty,
        base_ptr,
        [dev_val],
        [-2147483648],
        i32,
        llvm.GEPNoWrapFlags.inbounds,
        loc=loc,
        ip=ip,
    )
    llvm.store(
        thr_val,
        slot_ptr,
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _host_query_device_attrs(dev: Int32, *, loc=None, ip=None):
    """Query CUDA runtime device attributes via CuTe runtime's _cudaDeviceGetAttribute.

    Attribute IDs follow stable CUDA runtime cudaDeviceAttr ABI:
      16: MultiProcessorCount
      81: MaxSharedMemoryPerMultiprocessor
      97: MaxSharedMemoryPerBlockOptin
      111: ReservedSharedMemoryPerBlock
      82: MaxRegistersPerMultiprocessor
      12: MaxRegistersPerBlock
      39: MaxThreadsPerMultiProcessor
      106: MaxBlocksPerMultiprocessor
      10: WarpSize
    """
    mod = _find_builtin_module()
    _ensure_host_threshold_decls(mod, None, loc=loc)
    ptr_ty = ir.Type.parse("!llvm.ptr")
    i32 = T.i32()
    one = Int32(1).ir_value(loc=loc, ip=ip)
    zero = Int32(0).ir_value(loc=loc, ip=ip)
    dev_val = dev.ir_value(loc=loc, ip=ip)
    val_ptr = llvm.alloca(ptr_ty, one, i32, alignment=4, loc=loc, ip=ip)

    attr_ids = (16, 81, 97, 111, 82, 12, 39, 106, 10)
    err_acc = zero
    vals = []
    for attr_id in attr_ids:
        llvm.store(zero, val_ptr, loc=loc, ip=ip)
        attr_cst = Int32(attr_id).ir_value(loc=loc, ip=ip)
        err = llvm.call(
            i32,
            [val_ptr, attr_cst, dev_val],
            [],
            [],
            callee="_cudaDeviceGetAttribute",
            loc=loc,
            ip=ip,
        )
        err_acc = arith.ori(err_acc, err, loc=loc, ip=ip)
        vals.append(Int32(llvm.load(i32, val_ptr, loc=loc, ip=ip)))
    ok = arith.select(
        arith.cmpi(arith.CmpIPredicate.eq, err_acc, zero, loc=loc, ip=ip),
        one,
        zero,
        loc=loc,
        ip=ip,
    )
    return (Int32(ok), *vals)


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
        traits = make_unified_traits(
            ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3, fp8_rope=False
        )
        compact_layout = make_native_v41_compact_smem_layout(traits)
        full_layout = make_smem_layout(traits)
        self.block_threads = traits.block_threads
        self.compact_smem_bytes = compact_layout.total_bytes
        self.ctas_per_row = (heads // 16) * 10
        self.cache_symbol = f"_native_v41_thr_h{heads}"
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
        common_kwargs = dict(
            h_blocks=heads // 16,
            num_splits=10,
            num_heads=heads,
            q_head_dim=512,
            topk=128,
            extra_topk=512,
            q_stride=(heads * 512, 512, 1),
            swa_indices_stride0=512,
            extra_indices_stride0=512,
            mid_out_stride=(heads * 5120, 5120, 512, 1),
            mid_lse_stride=(heads * 10, 10, 1),
            has_extra=True,
            pbs_extra=256,
            valid_hpb=16,
            vector_q=True,
        )

        def _make_full_kernel():
            if (
                resolved_limits is not None
                and full_layout.total_bytes > resolved_limits.max_smem_per_block_optin
            ):
                return UnifiedDecodeKernel(
                    traits,
                    compact_layout,
                    64,
                    1,
                    **common_kwargs,
                    native_v41_2cta=True,
                    min_blocks_per_mp=1,
                    preferred_smem_carveout=None,
                )
            return UnifiedDecodeKernel(
                traits,
                full_layout,
                64,
                1,
                **common_kwargs,
                native_v41_2cta=False,
                min_blocks_per_mp=1,
                preferred_smem_carveout=None,
            )

        def _make_compact_kernel():
            if resolved_limits is None:
                min_blocks_per_mp = 2
            else:
                cand_compact, cand_min_blocks = resolve_native_v41_cta_placement(
                    block_threads=traits.block_threads,
                    compact_smem_bytes=compact_layout.total_bytes,
                    full_smem_bytes=full_layout.total_bytes,
                    limits=resolved_limits,
                )
                min_blocks_per_mp = 2 if (cand_compact and cand_min_blocks >= 2) else 1
            carveout = (
                None
                if min_blocks_per_mp <= 1 or _can_query_cutlass_max_smem_per_mp()
                else int(cuda.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_SHARED)
            )
            return UnifiedDecodeKernel(
                traits,
                compact_layout,
                64,
                1,
                **common_kwargs,
                native_v41_2cta=True,
                min_blocks_per_mp=min_blocks_per_mp,
                preferred_smem_carveout=carveout,
            )

        if variant is None and bucket_rows is None:
            self.dual_dispatch = True
            self.decode_full = _make_full_kernel()
            self.decode_compact = _make_compact_kernel()
            self.variant = "dual"
            self.decode = (
                self.decode_compact
                if self.threshold_rows is not None
                else self.decode_full
            )
        else:
            self.dual_dispatch = False
            chosen = (
                variant
                if variant is not None
                else select_native_v41_variant_for_rows(bucket_rows, self.threshold_rows)
            )
            if chosen == "compact":
                self.decode = _make_compact_kernel()
                self.variant = "compact"
            else:
                self.decode = _make_full_kernel()
                self.variant = "compact" if self.decode.native_v41_2cta else "full"
            self.decode_full = None
            self.decode_compact = None
        self._default_decode = self.decode
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
        scale = Float32(512**-.5 * math.log2(math.e))
        if cutlass.const_expr(self.decode is not self._default_decode or not self.dual_dispatch):
            self.decode.call_native_v41(q, desc, meta, ids, begins, p, lse, scale, rows, stream)
        else:
            dev, threshold = _host_load_cached_threshold(self.cache_symbol)
            if threshold == Int32(0) and dev >= Int32(0):
                (
                    ok,
                    num_sms,
                    smem_mp,
                    smem_optin,
                    rsv_smem,
                    regs_mp,
                    regs_blk,
                    thr_mp,
                    blk_mp,
                    warp_sz,
                ) = _host_query_device_attrs(dev)
                threshold = Int32(_FALLBACK_THRESHOLD_ROWS)
                if ok != Int32(0) and num_sms > Int32(0) and warp_sz > Int32(0):
                    warps = (Int32(self.block_threads) + warp_sz - Int32(1)) // warp_sz
                    cta_smem_2 = Int32(self.compact_smem_bytes) + rsv_smem
                    reg_budget_2 = ((regs_mp // (Int32(2) * warps)) // warp_sz) & Int32(-8)
                    if (
                        Int32(self.compact_smem_bytes) <= smem_optin
                        and Int32(2) * cta_smem_2 <= smem_mp
                        and Int32(2 * self.block_threads) <= thr_mp
                        and blk_mp >= Int32(2)
                        and reg_budget_2 > Int32(0)
                        and (reg_budget_2 * warps * warp_sz) <= regs_blk
                    ):
                        cpr = Int32(self.ctas_per_row)
                        threshold = (num_sms + cpr - Int32(1)) // cpr
                _host_store_cached_threshold(self.cache_symbol, dev, threshold)
            if rows >= threshold:
                self.decode_compact.call_native_v41(q, desc, meta, ids, begins, p, lse, scale, rows, stream)
            else:
                self.decode_full.call_native_v41(q, desc, meta, ids, begins, p, lse, scale, rows, stream)
        # Chunk-count tensor is unused by the static 10-part merge.
        self.merge(p, lse, begins, sinks, out, stream)


def _compile_native_v41_launch(
    launch: NativeV41Attention,
    *,
    device: int | None,
):
    if launch.dual_dispatch and launch.decode is launch._default_decode:
        key = (
            launch.heads,
            512,
            128,
            512,
            10,
            device,
            "dual_r1_r2",
            launch.decode_full.native_v41_2cta,
            launch.decode_full.min_blocks_per_mp,
            launch.decode_compact.native_v41_2cta,
            launch.decode_compact.min_blocks_per_mp,
        )
    else:
        key = (
            launch.heads,
            512,
            128,
            512,
            10,
            device,
            launch.variant,
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
    compile_kwargs = {"no_jit_engine": True} if device is None else {}
    compiled = compile_cute(
        launch,
        *pointers,
        Int32(1),
        _safe_compile_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("attention.native_v41", 5, key),
        **compile_kwargs,
    )
    return compiled


def compile_native_v41_attention_aot(
    heads: int = 64,
    *,
    variant: str | None = None,
    bucket_rows: int | None = None,
):
    """Compile native V4.1 attention ahead of time; live row counts never enter keys."""
    if variant not in (None, "full", "compact"):
        raise ValueError(f"variant must be None, 'full', or 'compact', got {variant!r}")
    if variant is not None and bucket_rows is not None:
        raise ValueError("specify at most one of variant or bucket_rows")
    device = _safe_current_cuda_device()
    limits = query_sm_occupancy_limits(device)
    launch = NativeV41Attention(
        heads,
        device=device,
        variant=variant,
        bucket_rows=bucket_rows,
        limits=limits,
    )
    return _compile_native_v41_launch(launch, device=device)
