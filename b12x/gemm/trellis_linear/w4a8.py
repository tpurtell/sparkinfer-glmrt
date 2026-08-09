"""Direct E4M3 trellis decode into SM120 MXFP8 MMA.

This module is the dense correctness/performance anchor for the routed port.
It consumes the exact compact P24/P33 payload, reconstructs E4M3 weights in
registers, and feeds those bytes directly to ``m16n8k32`` block-scaled MMA.
No FP16/BF16 weight tile is materialized.  SQG-Cheb uses the universal
2 KiB rank-to-E4M3 descriptor read through the read-only global cache; it is
runtime state, not checkpoint metadata.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint16, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    mxfp8_mma_m16n8k32_f32_e4m3,
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_lut,
)
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    empty_mxfp8_rows_for_dense_gemm,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _resolve_exl3_hadamard_128,
    _trellis_dense_buffer,
)


_PAIR_KINDS = {"P24", "P33"}
_RATE_AXES = {"k", "n"}
_W4A8_CODEBOOKS = {"sqg_xor_cheb_t12"}


class _TrellisW4A8DenseLaunch:
    """One-warp M16xN8 kernel with a direct native-tile B operand."""

    threads = 32

    def __init__(
        self,
        *,
        size_k: int,
        size_n: int,
        pair_kind: str,
        rate_axis: str,
        trellis_codebook: str,
    ) -> None:
        self.size_k = int(size_k)
        self.size_n = int(size_n)
        self.pair_kind = str(pair_kind).upper()
        self.rate_axis = str(rate_axis).lower()
        self.trellis_codebook = str(trellis_codebook).lower()
        if self.trellis_codebook not in _W4A8_CODEBOOKS:
            raise ValueError(
                "W4A8 trellis codebook must be one of "
                f"{sorted(_W4A8_CODEBOOKS)}, got {self.trellis_codebook!r}"
            )
        self.low_bits, self.high_bits = (
            (2, 4) if self.pair_kind == "P24" else (3, 3)
        )
        self.trellis_words = self.size_k * self.size_n * 3 // 32

    @cute.jit
    def __call__(
        self,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        trellis_ptr: cute.Pointer,
        rank_lut_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values = cute.make_tensor(
            values_ptr,
            cute.make_ordered_layout((m, self.size_k // 4), order=(1, 0)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_ordered_layout((m, self.size_k // 32), order=(1, 0)),
        )
        trellis = cute.make_tensor(
            trellis_ptr,
            cute.make_layout((self.trellis_words,)),
        )
        rank_lut = cute.make_tensor(
            rank_lut_ptr,
            cute.make_layout((4096,)),
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_ordered_layout((m, self.size_n), order=(1, 0)),
        )
        self.kernel(values, scale_rows, trellis, rank_lut, output, m).launch(
            grid=(cute.ceil_div(m, 16), self.size_n // 8, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _lane_geom(self, lane: Int32, bits: cutlass.Constexpr[int]):
        bits_i32 = Int32(int(bits))
        ring_u32 = Int32(8 * int(bits))
        t_offset = Int32(8) * lane
        b1 = (t_offset + Int32(257)) * bits_i32
        b0 = b1 - Int32(16)
        b2 = b1 + Int32(7 * int(bits))
        i0 = b0 >> Int32(5)
        i2 = (b2 - Int32(1)) >> Int32(5)
        ia = i0 - ring_u32 * (i0 >= ring_u32).to(Int32)
        ib = i2 - ring_u32 * (i2 >= ring_u32).to(Int32)
        s2 = (i2 + Int32(1)) * Int32(32) - b2
        return ia, ib, s2

    @cute.jit
    def _tile_base(
        self,
        k16: Int32,
        n16: Int32,
        record: Int32,
        bits: cutlass.Constexpr[int],
    ) -> Int64:
        if cutlass.const_expr(self.rate_axis == "n"):
            pair_u32_per_k16 = 8 * 8 * (self.low_bits + self.high_bits)
            high_base_u32 = 8 * 8 * self.low_bits
            return (
                Int64(k16) * Int64(pair_u32_per_k16)
                + Int64(record) * Int64(high_base_u32)
                + Int64(n16) * Int64(8 * int(bits))
            )

        n16_count = self.size_n // 16
        low_record_u32 = 8 * n16_count * 8 * self.low_bits
        local_k16 = k16 - record * Int32(8)
        return (
            Int64(record) * Int64(low_record_u32)
            + Int64(local_k16) * Int64(n16_count * 8 * int(bits))
            + Int64(n16) * Int64(8 * int(bits))
        )

    @cute.jit
    def _decode_tile(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        k16: Int32,
        n16: Int32,
        record: Int32,
        n_high: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_addr: Int64,
    ) -> Uint32:
        ia, ib, s2 = self._lane_geom(lane, bits)
        base = self._tile_base(k16, n16, record, bits)
        a = Uint32(trellis[base + Int64(ia)])
        b = Uint32(trellis[base + Int64(ib)])
        merged = (Int64(a) << Int64(32)) | Int64(b)
        win_a = Uint32(merged >> Int64(s2))
        win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
        lo, hi = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            win_a,
            win_b,
            rank_lut_addr,
            int(bits),
        )
        value = lo
        if n_high != Int32(0):
            value = hi
        return value

    @cute.jit
    def _decode_k32_bits(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        k32: Int32,
        n16: Int32,
        record: Int32,
        n_high: Int32,
        bits: cutlass.Constexpr[int],
        rank_lut_addr: Int64,
    ):
        e0 = self._decode_tile(
            trellis,
            lane,
            k32 * Int32(2),
            n16,
            record,
            n_high,
            bits,
            rank_lut_addr,
        )
        e1 = self._decode_tile(
            trellis,
            lane,
            k32 * Int32(2) + Int32(1),
            n16,
            record,
            n_high,
            bits,
            rank_lut_addr,
        )
        c = lane & Int32(3)
        own = e0
        send = e1
        if c >= Int32(2):
            own = e1
            send = e0
        # See tests/moe/test_w4a8_fragment_probe.py: the opposite register is
        # selected before xor-2 so every receiver gets the same K16 tile.
        peer = Uint32(cute.arch.shuffle_sync_bfly(send, offset=2))
        return own, peer

    @cute.jit
    def _decode_k32(
        self,
        trellis: cute.Tensor,
        lane: Int32,
        k32: Int32,
        n_base: Int32,
        rank_lut_addr: Int64,
    ):
        n_record = n_base // Int32(128)
        n_local = n_base - n_record * Int32(128)
        n16 = n_local // Int32(16)
        if cutlass.const_expr(self.rate_axis == "k"):
            # K-axis containers concatenate two complete K128 records, so N
            # remains the ordinary full-width native-tile coordinate.
            n16 = n_base // Int32(16)
        n_high = (n_base & Int32(15)) >= Int32(8)
        if cutlass.const_expr(self.pair_kind == "P33"):
            record = n_record if cutlass.const_expr(self.rate_axis == "n") else (
                k32 // Int32(4)
            )
            return self._decode_k32_bits(
                trellis,
                lane,
                k32,
                n16,
                record,
                n_high.to(Int32),
                3,
                rank_lut_addr,
            )

        b0 = Uint32(0)
        b1 = Uint32(0)
        if cutlass.const_expr(self.rate_axis == "n"):
            if n_record == Int32(0):
                b0, b1 = self._decode_k32_bits(
                    trellis,
                    lane,
                    k32,
                    n16,
                    Int32(0),
                    n_high.to(Int32),
                    2,
                    rank_lut_addr,
                )
            else:
                b0, b1 = self._decode_k32_bits(
                    trellis,
                    lane,
                    k32,
                    n16,
                    Int32(1),
                    n_high.to(Int32),
                    4,
                    rank_lut_addr,
                )
        else:
            if k32 < Int32(4):
                b0, b1 = self._decode_k32_bits(
                    trellis,
                    lane,
                    k32,
                    n16,
                    Int32(0),
                    n_high.to(Int32),
                    2,
                    rank_lut_addr,
                )
            else:
                b0, b1 = self._decode_k32_bits(
                    trellis,
                    lane,
                    k32,
                    n16,
                    Int32(1),
                    n_high.to(Int32),
                    4,
                    rank_lut_addr,
                )
        return b0, b1

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        scale_rows: cute.Tensor,
        trellis: cute.Tensor,
        rank_lut: cute.Tensor,
        output: cute.Tensor,
        m: Int32,
    ) -> None:
        lane = cute.arch.lane_idx()
        rank_lut_addr = get_ptr_as_int64(rank_lut, Int32(0))
        bmx, bny, _ = cute.arch.block_idx()
        c = lane & Int32(3)
        g = lane >> Int32(2)
        m_base = Int32(bmx) * Int32(16)
        n_base = Int32(bny) * Int32(8)
        m_lo = m_base + g
        m_hi = m_lo + Int32(8)

        d0 = Float32(0.0)
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        d3 = Float32(0.0)
        k32 = Int32(0)
        while k32 < Int32(self.size_k // 32):
            b0, b1 = self._decode_k32(
                trellis,
                lane,
                k32,
                n_base,
                rank_lut_addr,
            )
            word0 = k32 * Int32(8) + c * Int32(2)
            a0 = Uint32(0)
            a1 = Uint32(0)
            a2 = Uint32(0)
            a3 = Uint32(0)
            if m_lo < m:
                a0 = Uint32(values[m_lo, word0])
                a2 = Uint32(values[m_lo, word0 + Int32(1)])
            if m_hi < m:
                a1 = Uint32(values[m_hi, word0])
                a3 = Uint32(values[m_hi, word0 + Int32(1)])

            sf_row = m_base + g + ((lane & Int32(1)) << Int32(3))
            sf = Uint32(127)
            if sf_row < m:
                sf = Uint32(scale_rows[sf_row, k32])
            sfa = sf * Uint32(0x01010101)
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                d0,
                d1,
                d2,
                d3,
                a0,
                a1,
                a2,
                a3,
                b0,
                b1,
                sfa,
                Uint32(0x7F7F7F7F),
            )
            k32 += Int32(1)

        col = n_base + c * Int32(2)
        if m_lo < m:
            output[m_lo, col] = cutlass.Float16(d0)
            output[m_lo, col + Int32(1)] = cutlass.Float16(d1)
        if m_hi < m:
            output[m_hi, col] = cutlass.Float16(d2)
            output[m_hi, col + Int32(1)] = cutlass.Float16(d3)


@functools.cache
def _compile_dense(
    size_k: int,
    size_n: int,
    pair_kind: str,
    rate_axis: str,
    trellis_codebook: str,
    device_index: int,
):
    launch = _TrellisW4A8DenseLaunch(
        size_k=size_k,
        size_n=size_n,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        trellis_codebook=trellis_codebook,
    )
    key = (
        int(size_k),
        int(size_n),
        pair_kind,
        rate_axis,
        trellis_codebook,
        int(device_index),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.trellis_w4a8_dense",
            2,
            key,
        ),
    )


def _validate_quantized(
    quantized: MXFP8Rows,
    *,
    m: int,
    k: int,
    device: torch.device,
) -> None:
    if tuple(quantized.values.shape) != (m, k):
        raise ValueError(
            "quantized values must have shape "
            f"{(m, k)}, got {tuple(quantized.values.shape)}"
        )
    if tuple(quantized.scale_rows.shape) not in {
        (m, k // 32),
        (1, m, k // 32),
    }:
        raise ValueError(
            "quantized scale_rows must have one-group shape "
            f"{(m, k // 32)} or {(1, m, k // 32)}, got "
            f"{tuple(quantized.scale_rows.shape)}"
        )
    for name, tensor in (
        ("values", quantized.values),
        ("scale_rows", quantized.scale_rows),
    ):
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"quantized {name} must be contiguous on {device}")
    # scale_mma is intentionally a permuted view over contiguous physical
    # storage.  The quantizer writes that storage through its raw base pointer;
    # this dense kernel consumes scale_rows and never dereferences the view.
    if quantized.scale_mma.device != device:
        raise ValueError(f"quantized scale_mma must be on {device}")


def run_trellis256_dense_w4a8(
    x: torch.Tensor,
    prepared_dense,
    *,
    output: torch.Tensor | None = None,
    input_f16: torch.Tensor | None = None,
    rotated_f16: torch.Tensor | None = None,
    quantized: MXFP8Rows | None = None,
    gemm_output_f16: torch.Tensor | None = None,
    output_f16: torch.Tensor | None = None,
    hadamard_128=None,
) -> torch.Tensor:
    """Execute one compact E4M3 P24/P33 linear through native W4A8 MMA."""
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.ndim != 2 or int(x.shape[0]) <= 0 or not x.is_contiguous():
        raise ValueError("x must be a non-empty contiguous rank-2 tensor")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x must be fp16 or bf16, got {x.dtype}")
    trellis_codebook = str(
        getattr(prepared_dense, "trellis_codebook", "")
    ).lower()
    if trellis_codebook not in _W4A8_CODEBOOKS:
        raise NotImplementedError(
            "W4A8 trellis decode requires one of "
            f"{sorted(_W4A8_CODEBOOKS)}, got {trellis_codebook!r}"
        )
    pair_kind = str(getattr(prepared_dense, "trellis_pair_kind", "")).upper()
    rate_axis = str(getattr(prepared_dense, "trellis_rate_axis", "")).lower()
    if pair_kind not in _PAIR_KINDS or rate_axis not in _RATE_AXES:
        raise ValueError("W4A8 dense trellis requires a prepared P24/P33 pair")
    if int(getattr(prepared_dense, "num_experts", 0)) != 1:
        raise ValueError("W4A8 dense trellis requires E=1 prepared weights")
    if prepared_dense.trellis.device != x.device:
        raise ValueError("x and prepared weights must share one CUDA device")

    m, size_k = (int(v) for v in x.shape)
    size_n = int(prepared_dense.out_features)
    if size_k != int(prepared_dense.in_features):
        raise ValueError(
            f"x has K={size_k}, prepared weight expects {prepared_dense.in_features}"
        )
    device = x.device
    output = _trellis_dense_buffer(
        "w4a8 output", output, shape=(m, size_n), dtype=x.dtype, device=device
    )
    gemm_output_f16 = _trellis_dense_buffer(
        "w4a8 gemm_output_f16",
        gemm_output_f16,
        shape=(m, size_n),
        dtype=torch.float16,
        device=device,
    )
    hadamard_128 = _resolve_exl3_hadamard_128(hadamard_128)
    if x.dtype == torch.float16:
        x_f16 = x
    else:
        input_f16 = _trellis_dense_buffer(
            "w4a8 input_f16",
            input_f16,
            shape=(m, size_k),
            dtype=torch.float16,
            device=device,
        )
        input_f16.copy_(x)
        x_f16 = input_f16
    rotated_f16 = _trellis_dense_buffer(
        "w4a8 rotated_f16",
        rotated_f16,
        shape=(m, size_k),
        dtype=torch.float16,
        device=device,
    )
    hadamard_128(x_f16, rotated_f16, prepared_dense.suh, None, 1.0)

    if quantized is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A8 trellis quantization storage must be supplied during "
                "graph capture"
            )
        quantized = empty_mxfp8_rows_for_dense_gemm(
            m, size_k, device=device
        )
    _validate_quantized(quantized, m=m, k=size_k, device=device)
    quantize_mxfp8_rows_cute(
        rotated_f16,
        quantized.values,
        quantized.scale_rows,
        quantized.scale_mma,
        value_order="trellis_native_mma",
    )

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compiled = _compile_dense(
        size_k,
        size_n,
        pair_kind,
        rate_axis,
        trellis_codebook,
        int(device_index),
    )
    rank_lut = sqg_xor_cheb_t12_lut(device)
    compiled(
        make_ptr(
            cutlass.Uint32,
            quantized.values.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            quantized.scale_rows.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint32,
            prepared_dense.trellis.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            rank_lut.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float16,
            gemm_output_f16.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        m,
        current_cuda_stream(),
    )

    if output.dtype == torch.float16:
        hadamard_128(gemm_output_f16, output, None, prepared_dense.svh, 1.0)
    else:
        output_f16 = _trellis_dense_buffer(
            "w4a8 output_f16",
            output_f16,
            shape=(m, size_n),
            dtype=torch.float16,
            device=device,
        )
        hadamard_128(gemm_output_f16, output_f16, None, prepared_dense.svh, 1.0)
        output.copy_(output_f16)
    return output


__all__ = ["run_trellis256_dense_w4a8"]
