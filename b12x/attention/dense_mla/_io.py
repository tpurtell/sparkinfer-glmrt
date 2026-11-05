"""Dense page-table traversal and BF16/compact-MLA record staging."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    cvt_e4m3x4_to_f32x4,
    get_ptr_as_int64,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_u32,
)


@cute.jit
def issue_dense_page_gather(
    cache_bytes: cute.Tensor,
    page_table: cute.Tensor,
    kv_dst_addr: Int32,
    full_mbar_ptr,
    request: Int32,
    token_begin: Int32,
    token_end: Int32,
    io_lane: Int32,
    page_stride_bytes: Int64,
    record_stride_bytes_global: Int64,
    page_table_stride: Int64,
    *,
    page_size: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
    candidates_per_chunk: cutlass.Constexpr,
    io_threads: cutlass.Constexpr,
):
    """Stage one dense window with one bulk transaction per record.

    Every pool-scaled product is promoted before multiplication. Invalid tail
    rows copy physical record zero and are masked by the consumer; issuing a
    fixed byte count keeps the full-barrier protocol independent of live
    sequence length and therefore graph-replay stable.

    Lane zero arms the barrier with the chunk's exact byte count and each lane
    issues ``candidates_per_chunk // io_threads`` bulk copies, so the whole
    window moves with one instruction per record and the barrier completes on
    the arriving bytes rather than on a per-lane wait.  The caller therefore
    initializes the full barrier with a single arrival.
    """
    full_mbar_addr = shared_ptr_to_u32(full_mbar_ptr)
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(
            full_mbar_ptr,
            Int32(candidates_per_chunk * record_bytes),
        )

    entry = io_lane
    for _ in cutlass.range_constexpr(candidates_per_chunk // io_threads):
        logical_token = token_begin + entry
        safe_token = logical_token
        if logical_token >= token_end:
            safe_token = Int32(0)
        logical_page = safe_token // Int32(page_size)
        in_page = safe_token - logical_page * Int32(page_size)
        physical_page = Int32(
            page_table[request.to(Int64) * page_table_stride + logical_page.to(Int64)]
        )

        # Load-bearing Int64 conversions. Neither multiplication is allowed to
        # occur in Int32, even when benchmark page ids happen to be small.
        source_offset = (
            physical_page.to(Int64) * page_stride_bytes
            + in_page.to(Int64) * record_stride_bytes_global
        )
        cp_async_bulk_g2s_mbar(
            kv_dst_addr + entry * Int32(record_stride_bytes),
            get_ptr_as_int64(cache_bytes, source_offset),
            Int32(record_bytes),
            full_mbar_addr,
        )
        entry += Int32(io_threads)


@cute.jit
def issue_ds_mla_page_gather(
    cache_bytes: cute.Tensor, page_table: cute.Tensor, kv_dst_addr: Int32,
    full_mbar_ptr, request: Int32, token_begin: Int32, token_end: Int32,
    io_lane: Int32, page_stride_bytes: Int64, record_stride_bytes_global: Int64,
    page_table_stride: Int64, *, page_size: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr, candidates_per_chunk: cutlass.Constexpr,
    record_base: Int32, record_count: cutlass.Constexpr,
):
    """Read GLM's 656-byte record directly into the existing BF16 math tile.

    Only the current shared-memory window is widened; no global BF16 cache or
    query conversion is needed. Every producer lane publishes its own stores.
    ``record_base``/``record_count`` are one gathering warp's slice of the
    chunk; each warp of a multi-warp gather stages its own records.
    """
    request_base = request.to(Int64) * page_table_stride
    if cutlass.const_expr(candidates_per_chunk == page_size):
        # The producer begins every chunk at ``chunk * candidates_per_chunk``,
        # and a chunk is one page wide, so the whole chunk resolves through a
        # single page-table entry instead of one dependent global page lookup
        # per record. Candidates at or past ``token_end`` keep reading page
        # zero's first record, exactly as the general path below does.
        page = token_begin // Int32(page_size)
        chunk_physical = Int32(page_table[request_base + page.to(Int64)])
        zero_physical = Int32(page_table[request_base])
        batch = 8
        raw = cute.make_rmem_tensor((batch, 5), Uint32)
        scale_v = cute.make_rmem_tensor((batch, 4), Float32)
        for base in cutlass.range(record_count // batch, unroll=1):
            # Issue the whole batch's global loads before its first shared
            # store: the gather is latency bound, so eight independent round
            # trips in flight cost about one.
            for e in cutlass.range_constexpr(batch):
                entry = record_base + base * Int32(batch) + Int32(e)
                physical = chunk_physical
                # Chunk-local record index; token_begin is page aligned, so the
                # in-page offset is the entry itself.
                offset = entry
                if token_begin + entry >= token_end:
                    physical = zero_physical
                    offset = Int32(0)
                source = get_ptr_as_int64(
                    cache_bytes,
                    physical.to(Int64) * page_stride_bytes
                    + offset.to(Int64) * record_stride_bytes_global,
                )
                words = cute.make_ptr(Uint32, source, cute.AddressSpace.gmem, assumed_align=4)
                scales = cute.make_ptr(Float32, source + Int64(512), cute.AddressSpace.gmem, assumed_align=4)
                for j in cutlass.range_constexpr(4):
                    raw[e, j] = words[Int32(j) * Int32(32) + io_lane]
                    scale_v[e, j] = Float32(scales[Int32(j)])
                raw[e, 4] = words[Int32(132) + io_lane]
            for e in cutlass.range_constexpr(batch):
                destination = kv_dst_addr + (
                    record_base + base * Int32(batch) + Int32(e)
                ) * Int32(record_stride_bytes)
                for j in cutlass.range_constexpr(4):
                    a, b, c, d = cvt_e4m3x4_to_f32x4(raw[e, j])
                    scale = scale_v[e, j]
                    store = destination + (Int32(j) * Int32(128) + io_lane * Int32(4)) * Int32(2)
                    st_shared_u32(store, pack_f32x2_to_bfloat2(a * scale, b * scale))
                    st_shared_u32(store + Int32(4), pack_f32x2_to_bfloat2(c * scale, d * scale))
                st_shared_u32(destination + Int32(1024) + io_lane * Int32(4), raw[e, 4])
    else:
        for local in cutlass.range(record_count, unroll=1):
            entry = record_base + local
            token = token_begin + Int32(entry)
            if token >= token_end:
                token = Int32(0)
            page = token // Int32(page_size)
            physical = Int32(page_table[request_base + page.to(Int64)])
            source = get_ptr_as_int64(cache_bytes, physical.to(Int64) * page_stride_bytes
                + (token - page * Int32(page_size)).to(Int64) * record_stride_bytes_global)
            words = cute.make_ptr(Uint32, source, cute.AddressSpace.gmem, assumed_align=4)
            scales = cute.make_ptr(Float32, source + Int64(512), cute.AddressSpace.gmem, assumed_align=4)
            destination = kv_dst_addr + Int32(entry) * Int32(record_stride_bytes)
            for j in cutlass.range_constexpr(4):
                group = Int32(j)
                packed = words[group * Int32(32) + io_lane]
                a, b, c, d = cvt_e4m3x4_to_f32x4(packed)
                scale = Float32(scales[group])
                offset = destination + (group * Int32(128) + io_lane * Int32(4)) * Int32(2)
                st_shared_u32(offset, pack_f32x2_to_bfloat2(a * scale, b * scale))
                st_shared_u32(offset + Int32(4), pack_f32x2_to_bfloat2(c * scale, d * scale))
            st_shared_u32(destination + Int32(1024) + io_lane * Int32(4), words[Int32(132) + io_lane])
    cute.arch.mbarrier_arrive(full_mbar_ptr)


__all__ = ["issue_dense_page_gather", "issue_ds_mla_page_gather"]
