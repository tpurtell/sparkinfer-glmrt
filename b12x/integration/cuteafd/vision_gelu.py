"""BF16 tanh GELU boundary matching the official MoonViT PyTorch activation."""
import math
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16,Float32,Int32,Int64
from ._common import Operand,Scalar,compile_program
from .pointwise import _add,_mul,_fma,_view


class TanhGelu:
    def __init__(self,width):self.width=width
    @cute.jit
    def __call__(self,x:cute.Pointer,out:cute.Pointer,rows:Int32,stream:cuda.CUstream):
        count=Int64(rows)*Int64(self.width)
        self.kernel(x,out,count).launch(grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)
    @cute.kernel
    def kernel(self,x:cute.Pointer,out:cute.Pointer,count:Int64):
        x,out=_view(x),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            value=Float32(x[at])
            cube=_mul(_mul(value,value),value)
            inner=_mul(Float32(math.sqrt(2.0/math.pi)),_fma(Float32(0.044715),cube,value))
            activation=cute.math.tanh(inner,fastmath=False)
            out[at]=_mul(_mul(Float32(0.5),value),_add(Float32(1.0),activation)).to(BFloat16)


def compile_tanh_gelu_aot(*,width):
    if width<=0:raise ValueError("tanh GELU width must be positive")
    return compile_program(TanhGelu(width),name="vision_tanh_gelu",
        operands=(Operand("x",torch.bfloat16,f"[rows,{width}]"),
                  Operand("out",torch.bfloat16,f"[rows,{width}]","out")),
        scalars=(Scalar("rows"),),key=(width,),geometry=dict(width=width,activation="gelu_pytorch_tanh"))
