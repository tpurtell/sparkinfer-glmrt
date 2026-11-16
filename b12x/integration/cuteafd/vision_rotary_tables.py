"""Source-matched FP32 rotary tables, computed once per still-image grid."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32,Int32,Int64
from ._common import Operand,Scalar,compile_program
from .pointwise import _mul,_view


class RotaryTables:
    def __init__(self,axes,adjacent,group_major):
        self.axes,self.adjacent,self.group_major=axes,adjacent,group_major
        self.frequencies=sum(axes)//2

    @cute.jit
    def __call__(self,t:cute.Pointer,h:cute.Pointer,w:cute.Pointer,cos:cute.Pointer,sin:cute.Pointer,
                 height:Int32,width:Int32,stream:cuda.CUstream):
        count=Int64(height)*Int64(width)*Int64(self.frequencies)
        self.kernel(t,h,w,cos,sin,width,count).launch(grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,t:cute.Pointer,h:cute.Pointer,w:cute.Pointer,cos:cute.Pointer,sin:cute.Pointer,
               width:Int32,count:Int64):
        t,h,w,cos,sin=_view(t),_view(h),_view(w),_view(cos),_view(sin)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            row,frequency=at//Int64(self.frequencies),at%Int64(self.frequencies)
            y,x=row//Int64(width),row%Int64(width)
            if cutlass.const_expr(self.group_major):
                unit,local=row//4,row%4
                y=unit//Int64(width//2)*2+local//2
                x=unit%Int64(width//2)*2+local%2
            angle=Float32(0.0)
            if cutlass.const_expr(self.adjacent):
                # MoonViT interleaves x,y complex frequencies, not split halves.
                pair=frequency//2
                if frequency%2==0:angle=_mul(Float32(x),Float32(w[pair]))
                else:angle=_mul(Float32(y),Float32(h[pair]))
            else:
                if frequency<Int64(self.axes[0]//2):
                    angle=Float32(0.0)  # One temporal patch for a still image.
                elif frequency<Int64((self.axes[0]+self.axes[1])//2):
                    angle=_mul(Float32(y),Float32(h[frequency-Int64(self.axes[0]//2)]))
                else:
                    angle=_mul(Float32(x),Float32(w[frequency-Int64((self.axes[0]+self.axes[1])//2)]))
            cos[at]=cute.math.cos(angle,fastmath=False)
            sin[at]=cute.math.sin(angle,fastmath=False)


def compile_vision_rotary_tables_aot(*,axes,adjacent=False,group_major=False):
    axes=tuple(axes)
    if (axes,bool(adjacent),bool(group_major)) not in [((26,26,26),False,True),((0,64,64),True,False)]:
        raise ValueError("rotary table layout requires MiniMax3D or MoonViT2D still geometry")
    return compile_program(RotaryTables(axes,bool(adjacent),bool(group_major)),name="vision_rotary_tables",
        operands=(Operand("t",torch.float32,f"[{axes[0]//2}]"),
                  Operand("h",torch.float32,f"[{axes[1]//2}]"),Operand("w",torch.float32,f"[{axes[2]//2}]"),
                  Operand("cos",torch.float32,f"[height*width,{sum(axes)//2}]","out"),
                  Operand("sin",torch.float32,f"[height*width,{sum(axes)//2}]","out")),
        scalars=(Scalar("height"),Scalar("width")),key=(axes,bool(adjacent),bool(group_major)),
        geometry=dict(axes=axes,adjacent=bool(adjacent),group_major=bool(group_major),temporal=1))
