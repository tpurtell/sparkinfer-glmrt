"""ATen's four-warp vectorized BF16 RMS boundary, separate from affine defaults."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16,Float32,Int32,Int64
from b12x._lib.intrinsics import div_rn_f32,rsqrt_approx_ftz_f32
from ._common import Operand,Scalar,compile_program
from .pointwise import _add,_mul,_fma,_view


class VectorizedRms:
    def __init__(self,width):self.width=width
    @cute.jit
    def __call__(self,x:cute.Pointer,gamma:cute.Pointer,out:cute.Pointer,rows:Int32,
                 epsilon:Float32,stream:cuda.CUstream):
        self.kernel(x,gamma,out,epsilon).launch(grid=(rows,1,1),block=(128,1,1),stream=stream)
    @cute.kernel
    def kernel(self,x:cute.Pointer,gamma:cute.Pointer,out:cute.Pointer,epsilon:Float32):
        x,gamma,out=_view(x),_view(gamma),_view(out)
        row=Int64(cute.arch.block_idx()[0]);tid=Int32(cute.arch.thread_idx()[0])
        lane,warp=tid%32,tid//32;base=row*Int64(self.width)
        squares=Float32(0.0)
        # ATen vec_size4: each thread consumes4 adjacent values, then steps
        #128 vectors. FFMA per square, SHFL.DOWN16..1, shared warp2->0/3->1
        # then1->0. Its BF16 output is gamma*(rsqrt*x), rounded exactly once.
        for block in cutlass.range_constexpr((self.width+511)//512):
            for element in cutlass.range_constexpr(4):
                col=tid*4+Int32(block*512+element)
                if col<Int32(self.width):
                    value=Float32(x[base+Int64(col)])
                    squares=_fma(value,value,squares)
        for step in cutlass.range_constexpr(5):
            squares=_add(squares,Float32(cute.arch.shuffle_sync_down(squares,offset=16>>step)))
        partial=cutlass.utils.SmemAllocator().allocate_tensor(Float32,cute.make_layout((4,)))
        if lane==0:partial[warp]=squares
        cute.arch.sync_threads()
        if tid==0:
            combined=_add(_add(Float32(partial[0]),Float32(partial[2])),
                          _add(Float32(partial[1]),Float32(partial[3])))
            variance=div_rn_f32(combined,Float32(self.width))
            partial[0]=rsqrt_approx_ftz_f32(_add(variance,epsilon))
        cute.arch.sync_threads()
        inv=Float32(partial[0])
        for block in cutlass.range_constexpr((self.width+511)//512):
            for element in cutlass.range_constexpr(4):
                col=tid*4+Int32(block*512+element)
                if col<Int32(self.width):
                    out[base+Int64(col)]=_mul(Float32(gamma[col]),_mul(inv,Float32(x[base+Int64(col)]))).to(BFloat16)


def compile_vectorized_vision_rms_aot(*,width):
    if width<=0 or width%4:raise ValueError("vectorized vision RMS width must be positive and divisible by4")
    return compile_program(VectorizedRms(width),name="vision_vectorized_rms",
        operands=(Operand("x",torch.bfloat16,f"[rows,{width}]"),Operand("gamma",torch.bfloat16,f"[{width}]"),
                  Operand("out",torch.bfloat16,f"[rows,{width}]","out")),
        scalars=(Scalar("rows"),Scalar("epsilon","float32")),key=(width,),
        geometry=dict(width=width,reduction="aten_4warp_vec4_fma_down_shared",epsilon="runtime_fp32"))
