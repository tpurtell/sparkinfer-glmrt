"""Bit-equality of the direct state-table decode against the T12 decode.

The rate-indexed direct table precomposes the frozen XOR-Cheb rank map with
the modal T12 staircase (``direct[state] = t12[rank(state) >> 4]``), so the
two decode intrinsics must produce identical bytes for every window. The
probe drives both intrinsics on the same random ring windows and compares
the packed results.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
    packed_decode_trellis_sqg_direct_lut_to_e4m3x8,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_direct_lut_cpu,
    sqg_xor_cheb_t12_lut_cpu,
)
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x

require_b12x()


class _DecodeProbe:
    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(wins, t12, direct, out).launch(
            grid=(1, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = Int32(tidx)
        t12_addr = t12.iterator.toint()
        dir_addr = direct.iterator.toint()
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        lo_t, hi_t = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            wa, wb, t12_addr, self.bits, t12_in_shared=False
        )
        lo_d, hi_d = packed_decode_trellis_sqg_direct_lut_to_e4m3x8(
            wa, wb, dir_addr, self.bits, rate_indexed=True
        )
        out[4 * lane] = Int32(lo_t)
        out[4 * lane + 1] = Int32(hi_t)
        out[4 * lane + 2] = Int32(lo_d)
        out[4 * lane + 3] = Int32(hi_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_direct_lut_decode_bit_equals_t12(bits: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260812 + bits)
    wins = torch.zeros(64, dtype=torch.int32, device=device)
    t12 = sqg_xor_cheb_t12_lut_cpu().to(device)
    direct = sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    out = torch.zeros(128, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(t12, assumed_align=16),
            from_dlpack(direct, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_DecodeProbe(bits), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(
                -(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device
            )
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 4)
        mismatched += int((o[:, 0] != o[:, 2]).sum())
        mismatched += int((o[:, 1] != o[:, 3]).sum())
    assert mismatched == 0, mismatched
