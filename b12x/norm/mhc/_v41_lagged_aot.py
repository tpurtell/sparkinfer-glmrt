"""Native DS4.1 lagged pre: upstream partial/finalize with runtime row count.

Scratch is rows * 80 * 25 FP32 bytes, reusable collapsed storage. Residual,
incoming mix and weights are read-only. All outputs and scratch are disjoint.
The caller preinitializes the compiled program before graph capture.
"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, BFloat16, Float32
from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.utils import make_ptr, current_cuda_stream
from ._kernels import MHCPostPrePartialKernel, MHCFinalizeGramKernel


class _V41Lagged:
    def __init__(self):
        self.partial = MHCPostPrePartialKernel(hidden_size=5120, split_k=80,
            pre_only=True, materialize_pre=False, lagged_mix=True, partials_per_cta=4)
        self.finalize = MHCFinalizeGramKernel(hidden_size=5120, split_k=80,
            rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-20,
            fuse_norm=True, lagged_mix=True, lagged_prepared=True)

    @cute.jit
    def __call__(self, residual: cute.Pointer, fn: cute.Pointer, scale: cute.Pointer,
                 bias: cute.Pointer, incoming: cute.Pointer, norm: cute.Pointer,
                 predicted: cute.Pointer, post: cute.Pointer, comb: cute.Pointer,
                 normalized: cute.Pointer, scratch: cute.Pointer, rows: Int32,
                 stream: cuda.CUstream):
        m = Int64(rows)
        r = cute.make_tensor(residual, cute.make_layout((m,4,5120), stride=(20480,5120,1)))
        w = cute.make_tensor(fn, cute.make_layout((24,20480), stride=(20480,1)))
        p = cute.make_tensor(scratch, cute.make_layout((m,80,25), stride=(2000,25,1)))
        i = cute.make_tensor(incoming, cute.make_layout((m,4), stride=(4,1)))
        y = cute.make_tensor(normalized, cute.make_layout((m,5120), stride=(5120,1)))
        self.partial(r,r,p,p,w,p,r,i,y,rows,stream)
        self.finalize(r,p,
            cute.make_tensor(scale,cute.make_layout(3)),
            cute.make_tensor(bias,cute.make_layout(24)), y,
            cute.make_tensor(post,cute.make_layout((m,4),stride=(4,1))),
            cute.make_tensor(comb,cute.make_layout((m,4,4),stride=(16,4,1))),
            cute.make_tensor(norm,cute.make_layout(5120)),i,
            cute.make_tensor(predicted,cute.make_layout((m,4),stride=(4,1))),rows,stream)


def compile_v41_lagged_aot():
    types = (BFloat16,Float32,Float32,Float32,Float32,BFloat16,Float32,Float32,Float32,BFloat16,Float32)
    return compile_cute(_V41Lagged(),
        *(make_ptr(dtype,16,cute.AddressSpace.gmem,assumed_align=16) for dtype in types),
        Int32(1),current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key('norm.v41_lagged_native',1,(5120,80,25,4)))
