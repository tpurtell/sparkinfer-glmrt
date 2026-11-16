"""Sentinel-safe Inkling PE pixels from RGB8 canvas plus a content rectangle."""
from __future__ import annotations
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16,Float32,Int32,Int64,Uint32
from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand,Scalar,compile_program
from .pointwise import _add,_sub,_mul,_view


class PixelPatches:
    def __init__(self,patch):self.patch=patch

    @cute.jit
    def __call__(self,rgb:cute.Pointer,mean:cute.Pointer,processor_std:cute.Pointer,std:cute.Pointer,
                 out:cute.Pointer,height:Int32,width:Int32,content_x:Int32,content_y:Int32,
                 content_width:Int32,content_height:Int32,stream:cuda.CUstream):
        count=Int64(height)*Int64(width)*3
        if height>0 and width>0 and height%self.patch==0 and width%self.patch==0:
            self.kernel(rgb,mean,processor_std,std,out,width,count,content_x,content_y,content_width,content_height).launch(
                grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,rgb:cute.Pointer,mean:cute.Pointer,processor_std:cute.Pointer,std:cute.Pointer,
               out:cute.Pointer,width:Int32,count:Int64,cx:Int32,cy:Int32,cw:Int32,ch:Int32):
        rgb,mean,processor_std,std,out=_view(rgb),_view(mean),_view(processor_std),_view(std),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            row,feature=at//Int64(3*self.patch*self.patch),at%Int64(3*self.patch*self.patch)
            channel=feature%3;pixel=feature//3;gw=Int64(width//self.patch)
            y=row//gw*self.patch+pixel//self.patch;x=row%gw*self.patch+pixel%self.patch
            value=Float32(-1.0)
            if x>=Int64(cx) and x<Int64(cx)+Int64(cw) and y>=Int64(cy) and y<Int64(cy)+Int64(ch):
                value=Float32(Uint32(rgb[(y*Int64(width)+x)*3+channel]))
            value=div_rn_f32(_sub(div_rn_f32(value,Float32(255.0)),Float32(mean[channel])),Float32(processor_std[channel]))
            value=Float32(value.to(BFloat16))
            value=_sub(_mul(Float32(2.0),_add(_mul(value,Float32(std[channel])),Float32(mean[channel]))),Float32(1.0))
            out[at]=value.to(BFloat16)


class CheckedPixels:
    def __init__(self,program,patch):self.program,self.patch=program,patch
    def __getattr__(self,name):return getattr(self.program,name)
    def launch(self,*tensors,scalars,stream=None):
        if len(scalars)!=6 or len(tensors)!=5:raise ValueError("PE pixels need canvas/content extents and five operands")
        h,w,x,y,cw,ch=scalars
        if min(h,w,cw,ch)<=0 or min(x,y)<0 or h%self.patch or w%self.patch or x+cw>w or y+ch>h:
            raise ValueError("PE pixel content must be a nonempty rectangle inside a patch-aligned canvas")
        if h*w//(self.patch*self.patch)>16384:raise ValueError("PE pixel patch capacity exceeded")
        if any(t.numel()<n for t,n in zip(tensors,[h*w*3,3,3,3,h*w*3])):
            raise ValueError("PE pixel storage below admitted extent")
        return self.program.launch(*tensors,scalars=scalars,stream=stream)


def compile_pixel_patches_aot(*,patch=20):
    if patch<=0 or patch>64:raise ValueError("PE patch size must be in1..64")
    return CheckedPixels(compile_program(PixelPatches(patch),name="perception_rgb_content_patches",operands=(
        Operand("rgb",torch.uint8,"[height,width,3]"),Operand("mean",torch.float32,"[3]",align=4),
        Operand("processor_std",torch.float32,"[3]",align=4),Operand("std",torch.float32,"[3]",align=4),
        Operand("out",torch.bfloat16,f"[height*width/{patch*patch},{patch*patch*3}]","out")),
        scalars=tuple(Scalar(n) for n in ["height","width","content_x","content_y","content_width","content_height"]),
        key=(patch,),geometry=dict(patch=patch,layout="HWC",padding=-1,bf16_clip_remap=True)),patch)
