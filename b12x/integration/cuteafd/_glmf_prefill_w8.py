"""W8A16 prefill with one graph-owned BF16 weight expansion per launch."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.intrinsics import ld_global_nc_v2_u32, st_global_v4_u32
from b12x.gemm.bf16_gemv._skinny import TmaBf16Projection, WIDE_PREFILL_CONFIG
from ._fp8_weights import _e4m3x4_bf16x2x2, _e4m3x8_scaled_bf16


class ExpandRow128Weight:
    """Expand E4M3; channel scales stay separate until the MMA epilogue."""

    threads = 256
    elements = 2048

    def __init__(self, n: int, k: int, *, scale_granularity: str = "row128"):
        self.n, self.k = int(n), int(k)
        if self.k % 128:
            raise ValueError("row128 expansion requires K % 128 == 0")
        if scale_granularity not in ("row128", "channel"):
            raise ValueError("scale_granularity must be row128 or channel")
        self.channel = scale_granularity == "channel"

    @cute.jit
    def __call__(self, weight: cute.Pointer, scales: cute.Pointer, out: cute.Pointer, stream: cuda.CUstream):
        self.kernel(weight, scales, out).launch(
            grid=((self.n * self.k + self.elements - 1) // self.elements, 1, 1),
            block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, weight: cute.Pointer, scales: cute.Pointer, out: cute.Pointer):
        at = Int64(cute.arch.block_idx()[0]) * Int64(self.elements) + Int64(cute.arch.thread_idx()[0]) * Int64(8)
        if at < Int64(self.n * self.k):
            lo, hi = ld_global_nc_v2_u32(Int64(weight.toint()) + at)
            if cutlass.const_expr(self.channel):
                a, b = _e4m3x4_bf16x2x2(lo)
                c, d = _e4m3x4_bf16x2x2(hi)
                st_global_v4_u32(Int64(out.toint()) + at * Int64(2), a, b, c, d)
            else:
                # Interim Row128 fallback: full-fragment native scaling regressed
                # both architectures; channel-only epilogue scaling is not valid here.
                row, kb = at // Int64(self.k), (at % Int64(self.k)) // Int64(128)
                scale = scales[kb * Int64(self.n) + row]
                words = _e4m3x8_scaled_bf16(lo, hi, scale)
                st_global_v4_u32(Int64(out.toint()) + at * Int64(2), *words)


class ExpandedW8Prefill:
    """Reuse one transient weight arena; explicit channel scales apply once in F32.

    Row128 keeps the marked scaled-BF16 fallback (cuteafd#2068). Channel must
    come from checkpoint/conversion metadata, never inferred from scale layout.
    """

    def __init__(self, n: int, k: int, *, out_dtype=cutlass.BFloat16,
                 scale_granularity: str = "row128"):
        self.n, self.k = int(n), int(k)
        self.expand = ExpandRow128Weight(n, k, scale_granularity=scale_granularity)
        self.scale_granularity = scale_granularity
        self.native_scale = self.expand.channel
        self.projection = TmaBf16Projection(n, k, out_dtype=out_dtype, **WIDE_PREFILL_CONFIG)

    def key(self):
        return (self.n, self.k, self.scale_granularity, str(self.projection.out_dtype),
                tuple(sorted(self.projection.config.items())))

    @cute.jit
    def __call__(self, x: cute.Pointer, weight: cute.Pointer, scales: cute.Pointer, out: cute.Pointer,
                 scratch: cute.Pointer, rows: Int32, stream: cuda.CUstream):
        self.expand(weight, scales, scratch, stream)
        if cutlass.const_expr(self.native_scale):
            xt = cute.make_tensor(x, cute.make_layout((rows, self.k), stride=(self.k, 1)))
            wt = cute.make_tensor(scratch, cute.make_layout((self.n, self.k), stride=(self.k, 1)))
            ot = cute.make_tensor(out, cute.make_layout((rows, self.n), stride=(self.n, 1)))
            self.projection.kernel(xt, wt, ot, rows, stream, scales)
        else:
            self.projection(x, scratch, out, rows, stream)
