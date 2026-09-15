"""Direct DS41RT four-source address resolution for a future native producer.

The descriptor is the native 120-byte ``ds41rt_v41_sparse_kv_t`` viewed as
15 uint64 words. Callers validate pointer spans/capacities before launch, retain
all storage, and order metadata updates on the consuming lane's stream. This
helper does not load or repack KV payloads and is not yet wired into attention.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64

from b12x._lib.intrinsics import ld_global_nc_u32, ld_global_nc_v2_u32
from b12x._lib.intrinsics import cp_async_bulk_g2s_mbar, shared_ptr_to_u32, st_shared_u32
from .io import stage_dsv41_fp8_scales


@cute.jit
def _load_u64(address: Int64):
    lo, hi = ld_global_nc_v2_u32(address)
    return Uint64(lo) | (Uint64(hi) << Uint64(32))


@cute.jit
def native_v41_metadata_valid(descriptor: cute.Tensor, metadata: cute.Tensor, window_begin: Uint64):
    m0, offset, count, position = metadata[0], metadata[1], metadata[2], metadata[3]
    committed, private_count = metadata[6], metadata[7]
    private_offset, private_stride = metadata[8], metadata[9]
    window_capacity = descriptor[11]
    source_capacity, private_capacity = descriptor[12], descriptor[13]
    page_stride = descriptor[14] & Uint64(0xFFFFFFFF)
    compressed = descriptor[14] >> Uint64(32)
    valid = (window_begin <= m0) & (m0 <= Uint64(1048576)) & (count > Uint64(0))
    valid = valid & (m0 == _load_u64(Int64(descriptor[8])))
    valid = valid & (count <= Uint64(1048576) - m0) & (offset <= window_capacity)
    valid = valid & (count <= window_capacity - offset)
    valid = valid & (position >= m0) & (position - m0 < count)
    if valid and compressed != Uint64(0):
        source_end = _load_u64(Int64(descriptor[10]))
        boundary_ok = committed == source_end
        if private_count == Uint64(0):
            boundary_ok = committed <= source_end
        valid = (metadata[4] == Uint64(0)) & boundary_ok & (committed <= Uint64(1048576))
        valid = valid & (private_count <= Uint64(1048576) - committed)
        valid = valid & (metadata[5] <= committed + private_count)
        valid = valid & ((private_stride == Uint64(1)) | (private_stride == Uint64(2)))
        valid = valid & (private_offset <= private_capacity)
        if private_count != Uint64(0):
            # Division is reached only for a valid nonzero stride.
            if valid:
                valid = private_offset < private_capacity
                if valid:
                    valid = private_count - Uint64(1) <= (private_capacity - Uint64(1) - private_offset) // private_stride
    return valid


@cute.jit
def resolve_native_v41_record(
    descriptor: cute.Tensor, metadata: cute.Tensor, selected: cute.Tensor,
    key: Int32, window_begin: Uint64,
):
    """Return (value address, scale address); (0, 0) means a masked key.

    Keys 0..127 are the causal window, 128..639 are selected source IDs.
    This fixed boundary pads short windows, independently of the native
    implementation's optional compact physical tile schedule.
    """
    m0, offset, count, position = metadata[0], metadata[1], metadata[2], metadata[3]
    committed, private_count = metadata[6], metadata[7]
    private_offset, private_stride = metadata[8], metadata[9]
    window_capacity = descriptor[11]
    source_capacity, private_capacity = descriptor[12], descriptor[13]
    page_stride = descriptor[14] & Uint64(0xFFFFFFFF)
    compressed = descriptor[14] >> Uint64(32)
    valid = native_v41_metadata_valid(descriptor, metadata, window_begin)
    value_address, scale_address = Uint64(0), Uint64(0)
    source, row = Int32(0), Uint64(0)
    if valid:
        if key >= Int32(0) and key < Int32(128):
            first = Uint64(0)
            if position >= Uint64(128):
                first = position + Uint64(1) - Uint64(128)
            logical = first + Uint64(key)
            valid = (logical >= window_begin) & (logical <= position)
            if logical < m0:
                row = logical % Uint64(128)
            else:
                source = Int32(1)
                row = offset + logical - m0
        elif key >= Int32(128) and key < Int32(640) and compressed != Uint64(0):
            logical_id = Int32(selected[key - Int32(128)])
            valid = (logical_id >= Int32(0)) & (Uint64(logical_id) < metadata[5])
            if valid:
                logical = Uint64(logical_id)
                if logical >= committed:
                    source = Int32(3)
                    valid = logical - committed < private_count
                    row = private_offset + (logical - committed) * private_stride
                else:
                    source = Int32(2)
                    valid = logical < page_stride * Uint64(256)
                    if valid:
                        page = ld_global_nc_u32(Int64(descriptor[9] + (logical // Uint64(256)) * Uint64(4)))
                        row = Uint64(page) * Uint64(256) + logical % Uint64(256)
                        valid = row < source_capacity
        else:
            valid = False
        if valid:
            value_bytes, scale_bytes = Uint64(512), Uint64(16)
            if source >= Int32(2) and compressed == Uint64(2):
                value_bytes, scale_bytes = Uint64(256), Uint64(32)
            value_address = descriptor[source] + row * value_bytes
            scale_address = descriptor[source + Int32(4)] + row * scale_bytes
    return value_address, scale_address


@cute.jit
def issue_native_v41_gather(
    descriptor: cute.Tensor, metadata: cute.Tensor, selected: cute.Tensor,
    window_begin: Uint64, kv_addr: Int32, scale_addr: Int32,
    token_indices: cute.Tensor, full_mbar: cute.Pointer,
    start: Int32, io_lane: Int32, *, swa: cutlass.Constexpr,
    kv_stride: cutlass.Constexpr, io_threads: cutlass.Constexpr,
):
    """Stage native FP8-window/FP4-source records into upstream FP8 decode SMEM.

    Payload planes must be 16-byte aligned. Uses the existing 64-key producer
    stage and one arrival per IO warp. Native descriptor format must be FP4.
    Invalid keys read only the allocated ring's fallback row; their tag and
    canonical metadata mask that payload. No full-pool or selected-pool repack.
    """
    payload_bytes = 512 if swa else 256
    for part in cutlass.range_constexpr((64 + io_threads - 1) // io_threads):
        entry = io_lane + Int32(part * io_threads)
        if entry < Int32(64):
            key = start + entry + Int32(0 if swa else 128)
            value, scale = resolve_native_v41_record(descriptor, metadata, selected, key, window_begin)
            valid = value != Uint64(0)
            index, tag = key, Uint32(1 if swa else 0)
            if not valid:
                index, tag = Int32(-1), Uint32(2)
                value = Uint64(descriptor[0])
            token_indices[entry] = index
            row_addr = kv_addr + entry * Int32(kv_stride)
            st_shared_u32(row_addr + Int32(528), tag)
            stage_dsv41_fp8_scales(
                descriptor, Int64(0), row_addr + Int32(544),
                scale_addr + entry * Int32(8), valid,
                swa=swa, scale_address=Int64(scale),
            )
            cp_async_bulk_g2s_mbar(row_addr, Int64(value), Int32(payload_bytes), shared_ptr_to_u32(full_mbar))
    cute.arch.fence_acq_rel_cta()
    if (io_lane & Int32(31)) == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar, Int32(64 * payload_bytes // (io_threads // 32)))
