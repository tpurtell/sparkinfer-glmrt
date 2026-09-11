"""Exact-width views of the N256/K128 lane-major W4A8 resident layout.

These staging primitives do not change the resident representation. Callers
provide valid, 32-row-aligned slices and synchronize cp.async before reading
shared memory. Slice width is static geometry; position and strides are runtime.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from b12x._lib.intrinsics import cp_async4_shared_global, get_ptr_as_int64


@cute.jit
def stage_repacked_b_slice(
    source: cute.Tensor,
    shared_base: Int32,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_tile: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
):
    """Copy N64/N128/N192 weights into compact lane-major shared storage.

    The compact destination has shape [4,width/32,32,4] in u32 words.
    A slice may cross an N256 boundary, including independently padded W13
    projection halves. All source offsets and products use 64-bit arithmetic.
    """
    assert width in (64, 128, 192)
    chunks = width // 32
    transfers = 4 * chunks * 32
    for iteration in cutlass.range_constexpr((transfers + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(transfers):
            lane = index % Int32(32)
            chunk = (index // Int32(32)) % Int32(chunks)
            kb = index // Int32(32 * chunks)
            n = Int64(n_start) + Int64(chunk) * Int64(32)
            tile = (n // Int64(256)) * Int64(k_tiles) + Int64(k_tile)
            word = (expert_word_base + tile * Int64(4096)
                    + Int64(kb) * Int64(1024)
                    + ((n % Int64(256)) // Int64(32)) * Int64(128)
                    + Int64(lane) * Int64(4))
            cp_async4_shared_global(shared_base + index * Int32(16),
                                    get_ptr_as_int64(source, word))


@cute.jit
def stage_repacked_sfb_slice(
    source: cute.Tensor,
    shared_base: Int32,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_tile: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    width: cutlass.Constexpr,
):
    """Copy the corresponding N rows of four packed K32 UE8M0 scales."""
    assert width in (64, 128, 192)
    transfers = width // 4
    for iteration in cutlass.range_constexpr((transfers + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(transfers):
            n = Int64(n_start) + Int64(index) * Int64(4)
            tile = (n // Int64(256)) * Int64(k_tiles) + Int64(k_tile)
            word = expert_word_base + tile * Int64(256) + n % Int64(256)
            cp_async4_shared_global(shared_base + index * Int32(16),
                                    get_ptr_as_int64(source, word))


@cute.jit
def stage_repacked_b_k_slice(
    source: cute.Tensor,
    shared_base: Int32,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_start: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    depth: cutlass.Constexpr,
):
    """Copy N128 by K64/K128/K192 into [depth/32,4,32,4] u32.

    Both starts must be aligned to 32. Handles crossings of both packed tile
    axes; the caller owns bounds and completion just as for N-slice staging.
    """
    assert depth in (64, 128, 192)
    transfers = (depth // 32) * 4 * 32
    for iteration in cutlass.range_constexpr((transfers + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(transfers):
            lane = index % Int32(32)
            chunk = (index // Int32(32)) % Int32(4)
            kb = index // Int32(128)
            n = Int64(n_start) + Int64(chunk) * Int64(32)
            k = Int64(k_start) // Int64(32) + Int64(kb)
            tile = (n // Int64(256)) * Int64(k_tiles) + k // Int64(4)
            word = (expert_word_base + tile * Int64(4096)
                    + (k % Int64(4)) * Int64(1024)
                    + ((n % Int64(256)) // Int64(32)) * Int64(128)
                    + Int64(lane) * Int64(4))
            cp_async4_shared_global(shared_base + index * Int32(16),
                                    get_ptr_as_int64(source, word))


@cute.jit
def stage_repacked_sfb_k_slice(
    source: cute.Tensor,
    destination: cute.Tensor,
    expert_word_base: Int64,
    k_tiles: Int32,
    k_start: Int32,
    n_start: Int32,
    thread: Int32,
    threads: cutlass.Constexpr,
    depth: cutlass.Constexpr,
):
    """Gather N128 scales as [ceil(depth/128),128] packed u32 words.

    Only the final scale word is zero-padded to four bytes for MMA byte-id
    selection. Weight payload has no padding. Stores are synchronous; callers
    synchronize threads as well as completing the weight cp.async group.
    """
    assert depth in (64, 128, 192)
    groups = (depth + 127) // 128
    for iteration in cutlass.range_constexpr((128 * groups + threads - 1) // threads):
        index = thread + Int32(iteration * threads)
        if index < Int32(128 * groups):
            row = index % Int32(128)
            group = index // Int32(128)
            n = Int64(n_start) + Int64(row)
            packed = cutlass.Uint32(0)
            for byte in cutlass.range_constexpr(4):
                block = group * Int32(4) + Int32(byte)
                if block < Int32(depth // 32):
                    k = Int64(k_start) // Int64(32) + Int64(block)
                    tile = (n // Int64(256)) * Int64(k_tiles) + k // Int64(4)
                    word = source[expert_word_base + tile * Int64(256) + n % Int64(256)]
                    value = (cutlass.Uint32(word) >> cutlass.Uint32((k % Int64(4)) * Int64(8))) & cutlass.Uint32(255)
                    packed |= value << cutlass.Uint32(byte * 8)
            destination[index] = packed.to(destination.element_type)
