"""CuTe DSL implementations of the PCIe oneshot collectives.

The launch ABI uses stable device-resident ``uint64`` pointer tables.  This is
the only intentional ABI difference from the former CUDA extension (which
passed signal pointers by value and loaded data pointers through ``RankData``).
The payload loop, 16-byte packs, rank rotation, system-scope flag protocol,
and fused row barrier retain the CUDA implementation's ordering.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    ld_global_v4_u32,
    st_global_v4_f32,
    st_global_v4_u32,
    u32_as_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    atomic_add_global_u32,
    graph_epoch_arrive,
    ld_global_f32,
    ld_relaxed_gpu_u32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    pcie_arrive_and_wait,
    rsqrt_approx_f32,
    spin_until_changed_acquire_gpu,
    st_global_f32,
    st_global_u32,
    threadfence_gpu,
    unpack_bf16x2,
    unpack_f16x2,
)


_MAX_BLOCKS = 36
_MAX_RANKS = 16
_FLAG_STRIDE = 32
_SELF_COUNTER_BYTES = _MAX_BLOCKS * _MAX_RANKS * 4
_PEER_SLOT_BYTES = _MAX_BLOCKS * _MAX_RANKS * _FLAG_STRIDE * 4
_PEER_COUNTER_BYTES = 2 * _PEER_SLOT_BYTES
# Words one and two of the first padded peer-flag record are unused by the
# native barrier layout.  Graph-only slot state lives there without changing
# the signal allocation or any eager address.
_GRAPH_EPOCH_OFFSET = _SELF_COUNTER_BYTES + 4
_GRAPH_ARRIVED_OFFSET = _SELF_COUNTER_BYTES + 8
_RMS_ARRIVE_OFFSET = _SELF_COUNTER_BYTES + _PEER_COUNTER_BYTES
_RMS_GEN_OFFSET = 150_016
_RMS_PARTIAL_OFFSET = 150_272
_REG_PACKS = 3

_DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}

# CUDA graph capture must never be the first call that compiles or loads a
# specialization.  These process-local keys record successful launcher
# construction; host runtimes query them before entering the capture path.
_PREPARED_ONESHOT_LAUNCHERS: set[tuple[object, ...]] = set()
_PREPARED_FUSED_ONESHOT_LAUNCHERS: set[tuple[object, ...]] = set()


class _PackedMath:
    def __init__(self, dtype_name: str) -> None:
        if dtype_name not in _DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported oneshot dtype {dtype_name!r}")
        self._dtype_name = dtype_name
        self._pack_elems = _DTYPE_PACK_ELEMS[dtype_name]

    @cute.jit
    def _load_accumulate(
        self,
        accumulator: cute.Tensor,
        address: Int64,
        initialize: cutlass.Constexpr[bool],
    ) -> None:
        words = ld_global_v4_u32(address)
        if cutlass.const_expr(self._dtype_name == "float32"):
            for word in cutlass.range_constexpr(4):
                value = u32_as_f32(words[word])
                if cutlass.const_expr(initialize):
                    accumulator[word] = value
                else:
                    accumulator[word] = accumulator[word] + value
        else:
            for word in cutlass.range_constexpr(4):
                if cutlass.const_expr(self._dtype_name == "float16"):
                    lo, hi = unpack_f16x2(words[word])
                else:
                    lo, hi = unpack_bf16x2(words[word])
                lane = word * 2
                if cutlass.const_expr(initialize):
                    accumulator[lane] = lo
                    accumulator[lane + 1] = hi
                else:
                    accumulator[lane] = accumulator[lane] + lo
                    accumulator[lane + 1] = accumulator[lane + 1] + hi

    @cute.jit
    def _store_accumulator(
        self, address: Int64, accumulator: cute.Tensor
    ) -> None:
        if cutlass.const_expr(self._dtype_name == "float32"):
            st_global_v4_f32(
                address,
                accumulator[0],
                accumulator[1],
                accumulator[2],
                accumulator[3],
            )
        else:
            packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for word in cutlass.range_constexpr(4):
                lane = word * 2
                if cutlass.const_expr(self._dtype_name == "float16"):
                    packed[word] = pack_f32x2_to_f16x2(
                        accumulator[lane], accumulator[lane + 1]
                    )
                else:
                    packed[word] = pack_f32x2_to_bf16x2(
                        accumulator[lane], accumulator[lane + 1]
                    )
            st_global_v4_u32(
                address, packed[0], packed[1], packed[2], packed[3]
            )

    @cute.jit
    def _copy_pack(self, source: Int64, destination: Int64) -> None:
        words = ld_global_v4_u32(source)
        st_global_v4_u32(
            destination, words[0], words[1], words[2], words[3]
        )

    @cute.jit
    def _warp_reduce_down(self, value: Float32) -> Float32:
        for offset in (16, 8, 4, 2, 1):
            value = value + cute.arch.shuffle_sync_down(value, offset=offset)
        return value


class _OneshotLaunch(_PackedMath):
    def __init__(
        self,
        dtype_name: str,
        world_size: int,
        rank: int,
        stage_input: bool,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
    ) -> None:
        super().__init__(dtype_name)
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._stage_input = bool(stage_input)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = int(threads)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            input_ptr,
            output_ptr,
            size_packs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _multi_gpu_barrier(self, signal_ptrs: cute.Pointer) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self._stage_input):
            cute.arch.sync_threads()
        if tidx < Int32(self._world_size):
            self_signal = Int64(signal_ptrs[self._rank])
            peer_signal = Int64(signal_ptrs[tidx])
            self_counter = self_signal + (
                Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
            ) * Int64(4)
            peer_slot0 = peer_signal + Int64(_SELF_COUNTER_BYTES) + (
                Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(self._rank * _FLAG_STRIDE)
            ) * Int64(4)
            wait_slot0 = self_signal + Int64(_SELF_COUNTER_BYTES) + (
                Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(tidx) * Int64(_FLAG_STRIDE)
            ) * Int64(4)
            pcie_arrive_and_wait(
                self_counter,
                peer_slot0,
                wait_slot0,
                Int64(_PEER_SLOT_BYTES),
            )
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(
                Int64(signal_ptrs[self._rank]) + Int64(_GRAPH_EPOCH_OFFSET)
            )
            # The existing block/peer barrier below proves every thread has
            # consumed this load before any CTA can post its tail arrival.
            slot = (generation + Uint32(self._slot_bias)) % Uint32(2)
            peer_ptrs = peer_ptrs + Int64(slot) * Int64(_MAX_RANKS)

        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        if cutlass.const_expr(self._stage_input):
            staging_base = Int64(peer_ptrs[self._rank])
            stage_index = index
            while stage_index < size_packs:
                self._copy_pack(
                    input_base + Int64(stage_index) * Int64(16),
                    staging_base + Int64(stage_index) * Int64(16),
                )
                stage_index += stride

        self._multi_gpu_barrier(signal_ptrs)

        while index < size_packs:
            accumulator = cute.make_rmem_tensor(
                (self._pack_elems,), cutlass.Float32
            )
            for peer_index in cutlass.range_constexpr(self._world_size):
                peer_rank = (self._rank + peer_index) % self._world_size
                peer_base = Int64(peer_ptrs[peer_rank])
                self._load_accumulate(
                    accumulator,
                    peer_base + Int64(index) * Int64(16),
                    peer_index == 0,
                )
            self._store_accumulator(
                output_base + Int64(index) * Int64(16), accumulator
            )
            index += stride

        if cutlass.const_expr(self._device_slot_selection):
            if Int32(tidx) == Int32(0):
                self_signal = Int64(signal_ptrs[self._rank])
                graph_epoch_arrive(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_ARRIVED_OFFSET),
                    Uint32(gdim),
                )


class _FusedOneshotLaunch(_PackedMath):
    def __init__(
        self,
        dtype_name: str,
        world_size: int,
        rank: int,
        mode: str,
        single_cta: bool,
        register_normalize: bool,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
    ) -> None:
        super().__init__(dtype_name)
        if mode not in ("registered", "stage_pull", "stage_push"):
            raise ValueError(f"invalid fused oneshot mode {mode!r}")
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._mode = mode
        self._single_cta = bool(single_cta)
        self._register_normalize = bool(register_normalize)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = int(threads)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        residual_output_ptr: cute.Pointer,
        hidden_packs: Int32,
        rows: Int32,
        ctas_per_row: Int32,
        shard_packs: Int64,
        epsilon: Float32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            input_ptr,
            residual_ptr,
            weight_ptr,
            output_ptr,
            residual_output_ptr,
            hidden_packs,
            rows,
            ctas_per_row,
            shard_packs,
            epsilon,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _multi_gpu_barrier(self, signal_ptrs: cute.Pointer) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self._mode != "registered"):
            cute.arch.sync_threads()
        if tidx < Int32(self._world_size):
            self_signal = Int64(signal_ptrs[self._rank])
            peer_signal = Int64(signal_ptrs[tidx])
            self_counter = self_signal + (
                Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
            ) * Int64(4)
            peer_slot0 = peer_signal + Int64(_SELF_COUNTER_BYTES) + (
                Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(self._rank * _FLAG_STRIDE)
            ) * Int64(4)
            wait_slot0 = self_signal + Int64(_SELF_COUNTER_BYTES) + (
                Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(tidx) * Int64(_FLAG_STRIDE)
            ) * Int64(4)
            pcie_arrive_and_wait(
                self_counter,
                peer_slot0,
                wait_slot0,
                Int64(_PEER_SLOT_BYTES),
            )
        cute.arch.sync_threads()

    @cute.jit
    def _stage_pack(
        self,
        peer_ptrs: cute.Pointer,
        input_base: Int64,
        index: Int64,
        shard_packs: Int64,
    ) -> None:
        if cutlass.const_expr(self._mode == "stage_pull"):
            self._copy_pack(
                input_base + index * Int64(16),
                Int64(peer_ptrs[self._rank]) + index * Int64(16),
            )
        elif cutlass.const_expr(self._mode == "stage_push"):
            source = input_base + index * Int64(16)
            words = ld_global_v4_u32(source)
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                destination_rank = (self._rank + peer_index) % self._world_size
                destination = Int64(peer_ptrs[destination_rank]) + (
                    Int64(self._rank) * shard_packs + index
                ) * Int64(16)
                st_global_v4_u32(
                    destination, words[0], words[1], words[2], words[3]
                )

    @cute.jit
    def _reduce_pack(
        self,
        peer_ptrs: cute.Pointer,
        input_base: Int64,
        residual_base: Int64,
        index: Int64,
        shard_packs: Int64,
        accumulator: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(self._mode == "stage_push"):
            self._load_accumulate(
                accumulator,
                input_base + index * Int64(16),
                True,
            )
            local_stage = Int64(peer_ptrs[self._rank])
            initialized = True
            for source_rank in cutlass.range_constexpr(self._world_size):
                if cutlass.const_expr(source_rank != self._rank):
                    self._load_accumulate(
                        accumulator,
                        local_stage
                        + (
                            Int64(source_rank) * shard_packs + index
                        )
                        * Int64(16),
                        not initialized,
                    )
                    initialized = True
        else:
            for peer_index in cutlass.range_constexpr(self._world_size):
                peer_rank = (self._rank + peer_index) % self._world_size
                self._load_accumulate(
                    accumulator,
                    Int64(peer_ptrs[peer_rank]) + index * Int64(16),
                    peer_index == 0,
                )
        self._load_accumulate(
            accumulator,
            residual_base + index * Int64(16),
            False,
        )

    @cute.jit
    def _block_reduce(
        self,
        value: Float32,
        warp_sums: cute.Tensor,
    ) -> Float32:
        tidx, _, _ = cute.arch.thread_idx()
        lane = Int32(tidx) % Int32(32)
        warp = Int32(tidx) // Int32(32)
        value = self._warp_reduce_down(value)
        if lane == Int32(0):
            warp_sums[warp] = value
        cute.arch.sync_threads()
        block_value = Float32(0.0)
        if Int32(tidx) < Int32(self._threads // 32):
            block_value = warp_sums[lane]
        if warp == Int32(0):
            block_value = self._warp_reduce_down(block_value)
        return block_value

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        residual_output_ptr: cute.Pointer,
        hidden_packs: Int32,
        rows: Int32,
        ctas_per_row: Int32,
        shard_packs: Int64,
        epsilon: Float32,
    ) -> None:
        del rows
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self._single_cta):
            row = Int32(bidx)
            row_cta = Int32(0)
            col_stride = Int32(self._threads)
        else:
            row = Int32(bidx) // ctas_per_row
            row_cta = Int32(bidx) - row * ctas_per_row
            col_stride = ctas_per_row * Int32(self._threads)
        col0 = row_cta * Int32(self._threads) + Int32(tidx)
        row_offset = Int64(row) * Int64(hidden_packs)

        input_base = Int64(input_ptr.toint())
        residual_base = Int64(residual_ptr.toint())
        weight_base = Int64(weight_ptr.toint())
        output_base = Int64(output_ptr.toint())
        residual_output_base = Int64(residual_output_ptr.toint())
        self_signal = Int64(signal_ptrs[self._rank])
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(
                self_signal + Int64(_GRAPH_EPOCH_OFFSET)
            )
            slot = (generation + Uint32(self._slot_bias)) % Uint32(2)
            peer_ptrs = peer_ptrs + Int64(slot) * Int64(_MAX_RANKS)

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
        )
        inv_rms_smem = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )

        row_generation = Uint32(0)
        if cutlass.const_expr(not self._single_cta):
            if Int32(tidx) == Int32(0):
                row_generation = ld_relaxed_gpu_u32(
                    self_signal + Int64(_RMS_GEN_OFFSET) + Int64(row) * Int64(4)
                )

        if cutlass.const_expr(self._register_normalize):
            for pack_number in cutlass.range_constexpr(_REG_PACKS):
                col = col0 + Int32(pack_number) * col_stride
                if col < hidden_packs:
                    self._stage_pack(
                        peer_ptrs,
                        input_base,
                        row_offset + Int64(col),
                        shard_packs,
                    )
        elif cutlass.const_expr(self._mode != "registered"):
            col = col0
            while col < hidden_packs:
                self._stage_pack(
                    peer_ptrs,
                    input_base,
                    row_offset + Int64(col),
                    shard_packs,
                )
                col += col_stride

        self._multi_gpu_barrier(signal_ptrs)

        square_sum = Float32(0.0)
        values = cute.make_rmem_tensor(
            (_REG_PACKS, self._pack_elems), cutlass.Float32
        )
        if cutlass.const_expr(self._register_normalize):
            for pack_number in cutlass.range_constexpr(_REG_PACKS):
                col = col0 + Int32(pack_number) * col_stride
                if col < hidden_packs:
                    index = row_offset + Int64(col)
                    accumulator = cute.make_rmem_tensor(
                        (self._pack_elems,), cutlass.Float32
                    )
                    self._reduce_pack(
                        peer_ptrs,
                        input_base,
                        residual_base,
                        index,
                        shard_packs,
                        accumulator,
                    )
                    self._store_accumulator(
                        residual_output_base + index * Int64(16), accumulator
                    )
                    for lane in cutlass.range_constexpr(self._pack_elems):
                        value = accumulator[lane]
                        square_sum = square_sum + value * value
                        values[pack_number, lane] = value
        else:
            col = col0
            while col < hidden_packs:
                index = row_offset + Int64(col)
                accumulator = cute.make_rmem_tensor(
                    (self._pack_elems,), cutlass.Float32
                )
                self._reduce_pack(
                    peer_ptrs,
                    input_base,
                    residual_base,
                    index,
                    shard_packs,
                    accumulator,
                )
                self._store_accumulator(
                    residual_output_base + index * Int64(16), accumulator
                )
                for lane in cutlass.range_constexpr(self._pack_elems):
                    value = accumulator[lane]
                    square_sum = square_sum + value * value
                col += col_stride

        square_sum = self._block_reduce(square_sum, warp_sums)
        hidden_size_f = Float32(hidden_packs * Int32(self._pack_elems))
        if cutlass.const_expr(self._single_cta):
            if Int32(tidx) == Int32(0):
                inv_rms_smem[0] = rsqrt_approx_f32(
                    square_sum / hidden_size_f + epsilon
                )
        else:
            if Int32(tidx) == Int32(0):
                st_global_f32(
                    self_signal
                    + Int64(_RMS_PARTIAL_OFFSET)
                    + Int64(bidx) * Int64(4),
                    square_sum,
                )
                threadfence_gpu()
                arrive_address = (
                    self_signal
                    + Int64(_RMS_ARRIVE_OFFSET)
                    + Int64(row) * Int64(4)
                )
                prior = atomic_add_global_u32(arrive_address, Uint32(1))
                if prior == Uint32(ctas_per_row - Int32(1)):
                    st_global_u32(arrive_address, Uint32(0))
                    threadfence_gpu()
                    atomic_add_global_u32(
                        self_signal
                        + Int64(_RMS_GEN_OFFSET)
                        + Int64(row) * Int64(4),
                        Uint32(1),
                    )
                else:
                    spin_until_changed_acquire_gpu(
                        self_signal
                        + Int64(_RMS_GEN_OFFSET)
                        + Int64(row) * Int64(4),
                        row_generation,
                    )
            cute.arch.sync_threads()
            if Int32(tidx) < Int32(32):
                partial = Float32(0.0)
                partial_index = Int32(tidx)
                while partial_index < ctas_per_row:
                    partial = partial + ld_global_f32(
                        self_signal
                        + Int64(_RMS_PARTIAL_OFFSET)
                        + (
                            Int64(row) * Int64(ctas_per_row)
                            + Int64(partial_index)
                        )
                        * Int64(4)
                    )
                    partial_index += Int32(32)
                partial = self._warp_reduce_down(partial)
                if Int32(tidx) == Int32(0):
                    inv_rms_smem[0] = rsqrt_approx_f32(
                        partial / hidden_size_f + epsilon
                    )
        cute.arch.sync_threads()
        inv_rms = inv_rms_smem[0]

        if cutlass.const_expr(self._register_normalize):
            for pack_number in cutlass.range_constexpr(_REG_PACKS):
                col = col0 + Int32(pack_number) * col_stride
                if col < hidden_packs:
                    scale = cute.make_rmem_tensor(
                        (self._pack_elems,), cutlass.Float32
                    )
                    self._load_accumulate(
                        scale, weight_base + Int64(col) * Int64(16), True
                    )
                    normalized = cute.make_rmem_tensor(
                        (self._pack_elems,), cutlass.Float32
                    )
                    for lane in cutlass.range_constexpr(self._pack_elems):
                        normalized[lane] = (
                            values[pack_number, lane]
                            * (inv_rms * scale[lane])
                        )
                    self._store_accumulator(
                        output_base
                        + (row_offset + Int64(col)) * Int64(16),
                        normalized,
                    )
        else:
            col = col0
            while col < hidden_packs:
                index = row_offset + Int64(col)
                value = cute.make_rmem_tensor(
                    (self._pack_elems,), cutlass.Float32
                )
                scale = cute.make_rmem_tensor(
                    (self._pack_elems,), cutlass.Float32
                )
                self._load_accumulate(
                    value,
                    residual_output_base + index * Int64(16),
                    True,
                )
                self._load_accumulate(
                    scale, weight_base + Int64(col) * Int64(16), True
                )
                for lane in cutlass.range_constexpr(self._pack_elems):
                    value[lane] = value[lane] * (inv_rms * scale[lane])
                self._store_accumulator(
                    output_base + index * Int64(16), value
                )
                col += col_stride

        if cutlass.const_expr(self._device_slot_selection):
            if Int32(tidx) == Int32(0):
                graph_epoch_arrive(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_ARRIVED_OFFSET),
                    Uint32(gdim),
                )


def _dummy(dtype, alignment: int):
    return make_ptr(
        dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment
    )


def _oneshot_process_key(
    dtype_name: str,
    world_size: int,
    rank: int,
    stage_input: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        str(dtype_name),
        int(world_size),
        int(rank),
        bool(stage_input),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        int(threads),
        int(device_index),
    )


def is_oneshot_launcher_prepared(
    dtype_name: str,
    world_size: int,
    rank: int,
    stage_input: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> bool:
    """Return whether this exact process-local launcher is already loaded."""

    return _oneshot_process_key(
        dtype_name,
        world_size,
        rank,
        stage_input,
        device_slot_selection,
        slot_bias,
        threads,
        device_index,
    ) in _PREPARED_ONESHOT_LAUNCHERS


@functools.cache
def get_oneshot_launcher(
    dtype_name: str,
    world_size: int,
    rank: int,
    stage_input: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> Callable[..., None]:
    process_key = _oneshot_process_key(
        dtype_name,
        world_size,
        rank,
        stage_input,
        device_slot_selection,
        slot_bias,
        threads,
        device_index,
    )
    del device_index  # retained in the functools and preparation keys
    slot_bias = int(slot_bias) & 1 if device_slot_selection else 0
    launch = _OneshotLaunch(
        dtype_name,
        world_size,
        rank,
        stage_input,
        device_slot_selection,
        slot_bias,
        threads,
    )
    cache_key = (
        dtype_name,
        int(world_size),
        int(rank),
        bool(stage_input),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.oneshot", 1, cache_key
        ),
    )

    def run(
        peer_table_address: int,
        signal_table_address: int,
        input_address: int,
        output_address: int,
        size_packs: int,
        grid_x: int,
    ) -> None:
        raw(
            make_ptr(
                cutlass.Uint64,
                peer_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint64,
                signal_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint32,
                input_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(size_packs),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_ONESHOT_LAUNCHERS.add(process_key)
    return run


def _fused_oneshot_process_key(
    dtype_name: str,
    world_size: int,
    rank: int,
    mode: str,
    single_cta: bool,
    register_normalize: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        str(dtype_name),
        int(world_size),
        int(rank),
        str(mode),
        bool(single_cta),
        bool(register_normalize),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        int(threads),
        int(device_index),
    )


def is_fused_oneshot_launcher_prepared(
    dtype_name: str,
    world_size: int,
    rank: int,
    mode: str,
    single_cta: bool,
    register_normalize: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> bool:
    """Return whether this exact fused graph launcher is already loaded."""

    return _fused_oneshot_process_key(
        dtype_name,
        world_size,
        rank,
        mode,
        single_cta,
        register_normalize,
        device_slot_selection,
        slot_bias,
        threads,
        device_index,
    ) in _PREPARED_FUSED_ONESHOT_LAUNCHERS


@functools.cache
def get_fused_oneshot_launcher(
    dtype_name: str,
    world_size: int,
    rank: int,
    mode: str,
    single_cta: bool,
    register_normalize: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> Callable[..., None]:
    process_key = _fused_oneshot_process_key(
        dtype_name,
        world_size,
        rank,
        mode,
        single_cta,
        register_normalize,
        device_slot_selection,
        slot_bias,
        threads,
        device_index,
    )
    del device_index
    slot_bias = int(slot_bias) & 1 if device_slot_selection else 0
    launch = _FusedOneshotLaunch(
        dtype_name,
        world_size,
        rank,
        mode,
        single_cta,
        register_normalize,
        device_slot_selection,
        slot_bias,
        threads,
    )
    cache_key = (
        dtype_name,
        int(world_size),
        int(rank),
        mode,
        bool(single_cta),
        bool(register_normalize),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        1,
        1,
        1,
        0.0,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.oneshot_fused_rmsnorm", 1, cache_key
        ),
    )

    def run(
        peer_table_address: int,
        signal_table_address: int,
        input_address: int,
        residual_address: int,
        weight_address: int,
        output_address: int,
        residual_output_address: int,
        hidden_packs: int,
        rows: int,
        ctas_per_row: int,
        shard_packs: int,
        epsilon: float,
        grid_x: int,
    ) -> None:
        raw(
            make_ptr(cutlass.Uint64, peer_table_address, cute.AddressSpace.gmem, assumed_align=8),
            make_ptr(cutlass.Uint64, signal_table_address, cute.AddressSpace.gmem, assumed_align=8),
            make_ptr(cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, residual_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, weight_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, residual_output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(hidden_packs),
            int(rows),
            int(ctas_per_row),
            int(shard_packs),
            float(epsilon),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_FUSED_ONESHOT_LAUNCHERS.add(process_key)
    return run


__all__ = [
    "get_fused_oneshot_launcher",
    "get_oneshot_launcher",
    "is_fused_oneshot_launcher_prepared",
    "is_oneshot_launcher_prepared",
]
