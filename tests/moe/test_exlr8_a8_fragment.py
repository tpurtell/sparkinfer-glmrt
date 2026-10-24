"""One-warp native MCG decode into MXFP8 FC1 fragments, before the full port."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    mxfp8_mma_m16n8k32_f32_e4m3,
    packed_dequant_trellis_to_half2x4,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a8_trellis_decode import _W4A8_TRELLIS_A_K32_PERM


@dsl_user_op
def _f16x2_to_e4m3x2(value: Uint32, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [value.ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 e; cvt.rn.satfinite.e4m3x2.f16x2 e, $1; cvt.u32.u16 $0, e; }",
        "=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


class _Fragment:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(self, a: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor,
                 out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(a, packed, scales, out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor, out: cute.Tensor):
        lane = cute.arch.lane_idx()
        c, r = lane & Int32(3), lane >> Int32(2)
        ia, ib, shift, _ = trellis256_lane_geom_bits(lane, 0, 8, self.width)
        frags = []
        for kt in cutlass.range_constexpr(2):
            wa, wb = Uint32(0), Uint32(0)
            if cutlass.const_expr(self.width == 4):
                b = Uint32(packed[kt, lane])
                prev = cute.arch.shuffle_sync(b, (lane + Int32(31)) & Int32(31))
                wa, wb = b, (prev << Uint32(16)) | (b >> Uint32(16))
            else:
                x, y = Uint32(packed[kt, ia]), Uint32(packed[kt, ib])
                window = (x.to(Int64) << Int64(32)) | y.to(Int64)
                wa = (window >> shift.to(Int64)).to(Uint32)
                wb = (window >> (shift + Int32(4 * self.width)).to(Int64)).to(Uint32)
            o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(wa, wb, self.width)
            frags.append((_f16x2_to_e4m3x2(o0) | (_f16x2_to_e4m3x2(o1) << Uint32(16)),
                          _f16x2_to_e4m3x2(o2) | (_f16x2_to_e4m3x2(o3) << Uint32(16))))
        sfa = Uint32(scales[r + (lane & Int32(1)) * Int32(8)])
        for half in cutlass.range_constexpr(2):
            own, send = frags[0][half], frags[1][half]
            if c >= Int32(2):
                own, send = frags[1][half], frags[0][half]
            peer = Uint32(cute.arch.shuffle_sync_bfly(send, offset=2))
            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                Float32(0), Float32(0), Float32(0), Float32(0),
                Uint32(a[r, 2 * c]), Uint32(a[r + Int32(8), 2 * c]),
                Uint32(a[r, 2 * c + Int32(1)]), Uint32(a[r + Int32(8), 2 * c + Int32(1)]),
                own, peer, sfa, Uint32(0x7F))
            out[r, 2 * c + Int32(8 * half)] = d0
            out[r, 2 * c + Int32(8 * half + 1)] = d1
            out[r + Int32(8), 2 * c + Int32(8 * half)] = d2
            out[r + Int32(8), 2 * c + Int32(8 * half + 1)] = d3


def _reference(packed, width):
    """Independent scalar ring extraction and the MCG half-add codebook."""
    words = packed.cpu().to(torch.int64).tolist()
    b = torch.empty(16, 32)
    for kt in range(2):
        for lane in range(32):
            if width == 4:
                wa = words[kt][lane]
                wb = ((words[kt][(lane + 31) & 31] << 16) | (wa >> 16)) & 0xFFFFFFFF
            else:
                end = (8 * lane + 264) * width
                i0 = ((8 * lane + 257) * width - 16) // 32
                i2 = (end - 1) // 32
                window = (words[kt][i0 % (8 * width)] << 32) | words[kt][i2 % (8 * width)]
                shift = (i2 + 1) * 32 - end
                wa, wb = (window >> shift) & 0xFFFFFFFF, (window >> (shift + 4 * width)) & 0xFFFFFFFF
            states = [(word >> (j * width)) & 0xFFFF for word in (wb, wa) for j in (3, 2, 1, 0)]
            hbits = [(((s * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60) for s in states]
            halves = torch.tensor([[h & 0xFFFF, h >> 16] for h in hbits], dtype=torch.uint16).view(torch.float16)
            vals = halves.sum(1).to(torch.float8_e4m3fn).float()
            c, n = lane & 3, lane >> 2
            for j, k in enumerate((2 * c, 2 * c + 1, 2 * c + 8, 2 * c + 9)):
                b[n, kt * 16 + k] = vals[j]
                b[n + 8, kt * 16 + k] = vals[j + 4]
    return b.cuda()


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("scaled", [False, True], ids=["unit", "e8m0"])
def test_exlr8_a8_fragment(width, scaled):
    gen = torch.Generator().manual_seed(408 + width)
    packed = torch.randint(0, 2**32, (2, 8 * width), generator=gen, dtype=torch.int64).to(torch.uint32).cuda()
    a = (torch.randint(-8, 9, (16, 32), generator=gen).float() / 4).cuda()
    scale = torch.ones(16, device="cuda")
    if scaled:
        a = (torch.randn((16, 32), generator=gen) * torch.exp2(torch.arange(16)[:, None].float() / 2 - 8)).half().cuda().float()
        scale = torch.exp2(torch.ceil(torch.log2(a.abs().amax(1) / 448)))
    aq = (a / scale[:, None]).to(torch.float8_e4m3fn)
    perm = torch.tensor(_W4A8_TRELLIS_A_K32_PERM, device="cuda")
    ap = aq.view(torch.uint8)[:, perm].contiguous().view(torch.int32)
    codes = (torch.log2(scale).int() + 127).to(torch.int32)
    out = torch.empty((16, 16), device="cuda")
    tensors = [from_dlpack(t, assumed_align=16) for t in (ap, packed, codes, out)]
    tensors[0].element_type = Uint32
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_Fragment(width), *tensors, stream)
    compiled(*tensors, stream)
    torch.cuda.synchronize()
    # FP64 accumulation followed by FP32 avoids TF32 and tests the exact dot.
    ref = ((aq.float() * scale[:, None]).double() @ _reference(packed, width).double().T).float()
    assert torch.isfinite(out).all() and out.abs().max() > 0
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-6 if scaled else 0)
