"""Still-image RGB8 patch layout and spatial grouping, without host byte shuttles."""
from __future__ import annotations
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16,Float32,Int32,Int64,Uint32
from ._common import Operand,Scalar,compile_program
from .pointwise import _view


class StillPatches:
    def __init__(self,patch,temporal,group_major):
        self.patch,self.temporal,self.group_major=patch,temporal,group_major

    @cute.jit
    def __call__(self,rgb:cute.Pointer,lut:cute.Pointer,out:cute.Pointer,
                 height:Int32,width:Int32,stream:cuda.CUstream):
        count=Int64(height)*Int64(width)*Int64(3*self.temporal)
        self.kernel(rgb,lut,out,width,count).launch(grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,rgb:cute.Pointer,lut:cute.Pointer,out:cute.Pointer,width:Int32,count:Int64):
        rgb,lut,out=_view(rgb),_view(lut),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            patch_area=Int64(self.patch*self.patch)
            row,feature=at//Int64(3*self.temporal*self.patch*self.patch),at%Int64(3*self.temporal*self.patch*self.patch)
            channel=feature//Int64(self.temporal*self.patch*self.patch)
            pixel=feature%patch_area;gw=Int64(width//self.patch)
            py,px=row//gw,row%gw
            if cutlass.const_expr(self.group_major):
                unit,local=row//Int64(4),row%Int64(4)
                py=unit//(gw//Int64(2))*Int64(2)+local//Int64(2)
                px=unit%(gw//Int64(2))*Int64(2)+local%Int64(2)
            y=py*Int64(self.patch)+pixel//Int64(self.patch)
            x=px*Int64(self.patch)+pixel%Int64(self.patch)
            value=Uint32(rgb[(y*Int64(width)+x)*Int64(3)+channel])
            out[at]=Float32(lut[channel*Int64(256)+Int64(value)]).to(BFloat16)


def compile_still_patches_aot(*,patch=14,temporal=1,group_major=False):
    if not ((patch==14 and temporal in (1,2) and group_major==(temporal==2))
            or (patch==16 and temporal==1 and not group_major)):
        raise ValueError("still patch layout requires raster14/t1, group-major14/t2 or raster16/t1")
    return compile_program(StillPatches(patch,temporal,bool(group_major)),name="vision_still_patches",
        operands=(Operand("rgb",torch.uint8,"[height,width,3]"),Operand("lut",torch.float32,"[3,256]"),
                  Operand("out",torch.bfloat16,f"[height*width/{patch*patch},{3*temporal*patch*patch}]","out")),
        scalars=(Scalar("height"),Scalar("width")),key=(patch,temporal,bool(group_major)),
        geometry=dict(patch=patch,temporal=temporal,group_major=bool(group_major),normalization="reference_lut"))


class SpatialGroup:
    def __init__(self,channels,group_major,channel_major=False):
        self.channels,self.group_major,self.channel_major=channels,group_major,channel_major

    @cute.jit
    def __call__(self,x:cute.Pointer,out:cute.Pointer,height:Int32,width:Int32,stream:cuda.CUstream):
        count=Int64(height)*Int64(width)*Int64(self.channels)
        self.kernel(x,out,width,count).launch(grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,x:cute.Pointer,out:cute.Pointer,width:Int32,count:Int64):
        x,out=_view(x),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            if cutlass.const_expr(self.group_major):
                out[at]=x[at]
            elif cutlass.const_expr(self.channel_major):
                # Conv2d(kernel=2,stride=2) weight order [O,C,ky,kx]: feature c*4+ky*2+kx.
                unit,feature=at//Int64(4*self.channels),at%Int64(4*self.channels)
                channel,local=feature//Int64(4),feature%Int64(4)
                y=unit//Int64(width//2)*Int64(2)+local//Int64(2)
                xx=unit%Int64(width//2)*Int64(2)+local%Int64(2)
                out[at]=x[(y*Int64(width)+xx)*Int64(self.channels)+channel]
            else:
                unit,feature=at//Int64(4*self.channels),at%Int64(4*self.channels)
                local,channel=feature//Int64(self.channels),feature%Int64(self.channels)
                y=unit//Int64(width//2)*Int64(2)+local//Int64(2)
                xx=unit%Int64(width//2)*Int64(2)+local%Int64(2)
                out[at]=x[(y*Int64(width)+xx)*Int64(self.channels)+channel]


def compile_spatial_group_aot(*,channels,group_major=False,channel_major=False):
    """Raster patch rows to merged 2x2 units.

    Default feature order is (ky,kx) major then channel. ``channel_major`` gives
    channel then (ky,kx), so a 2x2 stride-2 Conv2d weight [O,C,2,2] flattens to
    a plain [O,4C] projection. Existing keys are unchanged.
    """
    if channels<=0:raise ValueError("spatial grouping channels must be positive")
    if channel_major and group_major:raise ValueError("channel-major grouping needs raster input")
    key=(channels,bool(group_major)) if not channel_major else (channels,False,"channel_major")
    geometry=dict(channels=channels,merge=2,input_group_major=bool(group_major))
    if channel_major:geometry["feature_order"]="channel_then_ky_kx"
    return compile_program(SpatialGroup(channels,bool(group_major),bool(channel_major)),name="vision_spatial_group",
        operands=(Operand("x",torch.bfloat16,f"[height*width,{channels}]"),
                  Operand("out",torch.bfloat16,f"[height*width/4,{4*channels}]","out")),
        scalars=(Scalar("height"),Scalar("width")),key=key,geometry=geometry)
