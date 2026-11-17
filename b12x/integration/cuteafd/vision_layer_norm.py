"""ATen BF16 vectorized Welford LayerNorm; existing affine defaults unchanged."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16,Float32,Int32,Int64
from b12x._lib.intrinsics import div_rn_f32,rsqrt_approx_ftz_f32
from ._common import Operand,Scalar,compile_program
from .pointwise import _add,_sub,_mul,_fma,_view


class VectorizedLayerNorm:
    def __init__(self,width):self.width=width
    @cute.jit
    def __call__(self,x:cute.Pointer,gamma:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,
                 rows:Int32,epsilon:Float32,stream:cuda.CUstream):
        self.kernel(x,gamma,bias,out,epsilon).launch(grid=(rows,1,1),block=(128,1,1),stream=stream)
    @cute.jit
    def combine(self,mean:Float32,square:Float32,count:Float32,
                other_mean:Float32,other_square:Float32,other_count:Float32):
        total=_add(count,other_count)
        new_mean,new_square=Float32(0.0),Float32(0.0)
        if total>Float32(0.0):
            delta=_sub(mean,other_mean)
            reciprocal=div_rn_f32(Float32(1.0),total)
            a=_mul(other_count,reciprocal)
            b=_mul(count,reciprocal)
            new_mean=_fma(other_mean,a,_mul(b,mean))
            correction=_mul(_mul(delta,delta),other_count)
            new_square=_fma(b,correction,_add(other_square,square))
        return new_mean,new_square,total
    @cute.kernel
    def kernel(self,x:cute.Pointer,gamma:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,epsilon:Float32):
        x,gamma,bias,out=_view(x),_view(gamma),_view(bias),_view(out)
        row=Int64(cute.arch.block_idx()[0]);tid=Int32(cute.arch.thread_idx()[0])
        lane,warp=tid%32,tid//32;base=row*Int64(self.width)
        mean,square,count=Float32(0.0),Float32(0.0),Float32(0.0)
        for block in cutlass.range_constexpr((self.width+511)//512):
            for element in cutlass.range_constexpr(4):
                col=tid*4+Int32(block*512+element)
                if col<Int32(self.width):
                    value=Float32(x[base+Int64(col)])
                    delta=_sub(value,mean)
                    count=_add(count,Float32(1.0))
                    mean=_fma(delta,div_rn_f32(Float32(1.0),count),mean)
                    square=_fma(delta,_sub(value,mean),square)
        for step in cutlass.range_constexpr(5):
            om=Float32(cute.arch.shuffle_sync_down(mean,offset=16>>step))
            os=Float32(cute.arch.shuffle_sync_down(square,offset=16>>step))
            oc=Float32(cute.arch.shuffle_sync_down(count,offset=16>>step))
            mean,square,count=self.combine(mean,square,count,om,os,oc)
        partial=cutlass.utils.SmemAllocator().allocate_tensor(Float32,cute.make_layout((12,)))
        if lane==0:
            partial[warp*3]=mean;partial[warp*3+1]=square;partial[warp*3+2]=count
        cute.arch.sync_threads()
        if tid==0:
            m0,s0,c0=self.combine(Float32(partial[0]),Float32(partial[1]),Float32(partial[2]),
                                 Float32(partial[6]),Float32(partial[7]),Float32(partial[8]))
            m1,s1,c1=self.combine(Float32(partial[3]),Float32(partial[4]),Float32(partial[5]),
                                 Float32(partial[9]),Float32(partial[10]),Float32(partial[11]))
            final_mean,final_square,_=self.combine(m0,s0,c0,m1,s1,c1)
            partial[0]=final_mean
            partial[1]=rsqrt_approx_ftz_f32(_add(div_rn_f32(final_square,Float32(self.width)),epsilon))
        cute.arch.sync_threads()
        mean,inv=Float32(partial[0]),Float32(partial[1])
        for block in cutlass.range_constexpr((self.width+511)//512):
            for element in cutlass.range_constexpr(4):
                col=tid*4+Int32(block*512+element)
                if col<Int32(self.width):
                    normalized=_mul(inv,_sub(Float32(x[base+Int64(col)]),mean))
                    out[base+Int64(col)]=_fma(Float32(gamma[col]),normalized,Float32(bias[col])).to(BFloat16)


def compile_vectorized_vision_layer_norm_aot(*,width):
    if width<=0 or width%4:raise ValueError("vectorized vision LayerNorm width must be positive and divisible by4")
    return compile_program(VectorizedLayerNorm(width),name="vision_vectorized_layer_norm",
        operands=(Operand("x",torch.bfloat16,f"[rows,{width}]"),Operand("gamma",torch.bfloat16,f"[{width}]"),
                  Operand("bias",torch.bfloat16,f"[{width}]"),Operand("out",torch.bfloat16,f"[rows,{width}]","out")),
        scalars=(Scalar("rows"),Scalar("epsilon","float32")),key=(width,),
        geometry=dict(width=width,reduction="aten_4warp_vec4_welford_down_shared",epsilon="runtime_fp32"))
