"""Native V4.1 mHC FP32 projection partials for bounded small-row serving.

The caller supplies FP32 scratch [rows,24,split_k,2] (dot, square).
Contiguous K partitions change FP32 reduction order; they do not quantize
weights or activations. All pool-scaled addressing is Int64. Live rows
control only the launch grid and never compilation identity.
"""

from __future__ import annotations
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, BFloat16, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import make_ptr, current_cuda_stream


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class _V41Project:
    def __init__(self, split):
        self.split = split

    @cute.jit
    def __call__(
        self,
        r: cute.Pointer,
        w: cute.Pointer,
        p: cute.Pointer,
        m: Int32,
        s: cuda.CUstream,
    ):
        self.kernel(r, w, p).launch(
            grid=(m, 24, self.split), block=(32, 1, 1), stream=s
        )

    @cute.kernel
    def kernel(self, r: cute.Pointer, w: cute.Pointer, p: cute.Pointer):
        row, proj, part = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        dot = Float32(0)
        square = Float32(0)
        for j in cutlass.range(20480 // self.split // 32, unroll=8):
            col = (
                Int64(part) * Int64(20480 // self.split)
                + Int64(j) * Int64(32)
                + Int64(lane)
            )
            x = Float32(r[Int64(row) * Int64(20480) + col])
            v = Float32(w[Int64(proj) * Int64(20480) + col])
            dot = _fma_rn(x, v, dot)
            square = _fma_rn(x, x, square)
        for shift in cutlass.range_constexpr(5):
            dot += cute.arch.shuffle_sync_down(dot, offset=16 >> shift)
            square += cute.arch.shuffle_sync_down(square, offset=16 >> shift)
        if lane == 0:
            at = ((Int64(row) * 24 + Int64(proj)) * self.split + Int64(part)) * 2
            p[at] = dot
            p[at + 1] = square


def compile_v41_mhc_project_aot(*, split_k: int = 8):
    """Export one runtime-row launch; scratch is rows * 24 * split_k * 8 bytes."""
    if split_k not in (1, 2, 4, 8):
        raise ValueError("V4.1 mHC split_k must be 1, 2, 4, or 8")
    launch = _V41Project(split_k)
    key = (20480, 24, split_k, torch.cuda.current_device())
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    return compile_cute(
        launch,
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("norm.v41_mhc_project", 1, key),
    )
