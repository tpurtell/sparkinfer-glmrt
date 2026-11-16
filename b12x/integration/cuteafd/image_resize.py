"""Owner-side normalized RGB8 -> unclamped FP32 separable tensor resize.

Follows ATen CUDA upsample_antialias weights and horizontal-then-vertical
accumulation. Caller owns all storage; image sizes are runtime launch values.
"""
from __future__ import annotations
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32,Int32,Int64,Uint32
from b12x._lib.intrinsics import div_rn_f32
from ._common import Operand,Scalar,compile_program
from .pointwise import _add,_fma,_mul,_sub,_view


class NormalizeRgb:
    @cute.jit
    def __call__(self,rgb:cute.Pointer,mean:cute.Pointer,std:cute.Pointer,out:cute.Pointer,
                 height:Int32,width:Int32,stream:cuda.CUstream):
        count=Int64(height)*Int64(width)*Int64(3)
        self.kernel(rgb,mean,std,out,count).launch(grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,rgb:cute.Pointer,mean:cute.Pointer,std:cute.Pointer,out:cute.Pointer,count:Int64):
        rgb,mean,std,out=_view(rgb),_view(mean),_view(std),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            channel=at%3
            value=_mul(Float32(Uint32(rgb[at])),Float32(1.0/255.0))
            out[at]=div_rn_f32(_sub(value,Float32(mean[channel])),Float32(std[channel]))


class ResizeCoefficients:
    def __init__(self,filter,antialias):
        self.filter,self.antialias=filter,antialias

    @cute.jit
    def __call__(self,indices:cute.Pointer,sizes:cute.Pointer,weights:cute.Pointer,
                 input_size:Int32,output_size:Int32,taps:Int32,stream:cuda.CUstream):
        self.kernel(indices,sizes,weights,input_size,output_size,taps).launch(
            grid=((output_size+127)//128,1,1),block=(128,1,1),stream=stream)

    @cute.jit
    def cubic(self,x:Float32):
        if x<Float32(0.0):
            x=-x
        a=Float32(-.5)
        if cutlass.const_expr(not self.antialias):
            a=Float32(-.75)
        value=Float32(0.0)
        if x<Float32(1.0):
            value=_fma(_mul(_fma(_add(a,Float32(2.0)),x,-_add(a,Float32(3.0))),x),x,Float32(1.0))
        elif x<Float32(2.0):
            value=_mul(_fma(_fma(_sub(x,Float32(5.0)),x,Float32(8.0)),x,Float32(-4.0)),a)
        return value

    @cute.kernel
    def kernel(self,indices:cute.Pointer,sizes:cute.Pointer,weights:cute.Pointer,
               input_size:Int32,output_size:Int32,taps:Int32):
        indices,sizes,weights=_view(indices),_view(sizes),_view(weights)
        row=Int32(cute.arch.block_idx()[0])*128+Int32(cute.arch.thread_idx()[0])
        if row<output_size:
            scale=div_rn_f32(Float32(input_size),Float32(output_size))
            factor=Float32(1.0)
            if cutlass.const_expr(self.antialias):
                if scale>=Float32(1.0):
                    factor=scale
            support=Float32(2.0)
            if cutlass.const_expr(self.filter=="bilinear"):
                support=Float32(1.0)
            support=_mul(support,factor)
            center=_mul(scale,_add(Float32(row),Float32(.5)))
            first=Int32(_add(_sub(center,support),Float32(.5)))
            if first<Int32(0):
                first=Int32(0)
            end=Int32(_add(_add(center,support),Float32(.5)))
            if end>input_size:
                end=input_size
            size=end-first
            indices[row]=first;sizes[row]=size
            invscale=div_rn_f32(Float32(1.0),factor)
            total=Float32(0.0)
            first_minus_center=_fma(-scale,_add(Float32(row),Float32(.5)),Float32(first))
            for tap in cutlass.range(size):
                offset=_mul(_add(_add(Float32(tap),first_minus_center),Float32(.5)),invscale)
                value=Float32(0.0)
                if cutlass.const_expr(self.filter=="bicubic"):
                    value=self.cubic(offset)
                else:
                    if offset<Float32(0.0):
                        offset=-offset
                    if offset<Float32(1.0):
                        value=_sub(Float32(1.0),offset)
                weights[Int64(row)*Int64(taps)+Int64(tap)]=value
                total=_add(total,value)
            for tap in cutlass.range(taps):
                at=Int64(row)*Int64(taps)+Int64(tap)
                value=Float32(0.0)
                if tap<size:
                    value=Float32(weights[at])
                    if total!=Float32(0.0):
                        value=div_rn_f32(value,total)
                weights[at]=value


class ResizePass:
    def __init__(self,vertical):
        self.vertical=vertical

    @cute.jit
    def __call__(self,x:cute.Pointer,indices:cute.Pointer,sizes:cute.Pointer,weights:cute.Pointer,
                 out:cute.Pointer,input_height:Int32,input_width:Int32,output_size:Int32,taps:Int32,
                 stream:cuda.CUstream):
        count=Int64(input_height)*Int64(output_size)*3
        if cutlass.const_expr(self.vertical):
            count=Int64(output_size)*Int64(input_width)*3
        self.kernel(x,indices,sizes,weights,out,input_width,output_size,taps,count).launch(
            grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,x:cute.Pointer,indices:cute.Pointer,sizes:cute.Pointer,weights:cute.Pointer,
               out:cute.Pointer,input_width:Int32,output_size:Int32,taps:Int32,count:Int64):
        x,indices,sizes,weights,out=_view(x),_view(indices),_view(sizes),_view(weights),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            channel=at%3;pixel=at//3
            axis=pixel%Int64(output_size)
            base=(pixel//Int64(output_size))*Int64(input_width)*3+channel
            stride=Int64(3)
            if cutlass.const_expr(self.vertical):
                axis=pixel//Int64(input_width)
                base=(pixel%Int64(input_width))*3+channel
                stride=Int64(input_width)*3
            first=Int64(indices[axis]);size=Int32(sizes[axis])
            value=_mul(Float32(x[base+first*stride]),Float32(weights[axis*Int64(taps)]))
            for tap in cutlass.range(1,size):
                value=_fma(Float32(x[base+(first+Int64(tap))*stride]),
                    Float32(weights[axis*Int64(taps)+Int64(tap)]),value)
            out[at]=value


def compile_rgb_normalize_aot():
    return compile_program(NormalizeRgb(),name="image_rgb_normalize",operands=(
        Operand("rgb",torch.uint8,"[height,width,3]"),Operand("mean",torch.float32,"[3]",align=4),
        Operand("std",torch.float32,"[3]",align=4),Operand("out",torch.float32,"[height,width,3]","out")),
        scalars=(Scalar("height"),Scalar("width")),key=(),geometry=dict(channels=3),version=3)


def compile_resize_coefficients_aot(*,filter="bicubic",antialias=True):
    if filter not in ("bicubic","bilinear"):
        raise ValueError("tensor resize filter must be bicubic or bilinear")
    if not antialias:
        raise ValueError("non-antialiased tensor resize needs its distinct boundary rule")
    return compile_program(ResizeCoefficients(filter,bool(antialias)),name="image_resize_coefficients",operands=(
        Operand("indices",torch.int32,"[output_size]","out"),Operand("sizes",torch.int32,"[output_size]","out"),
        Operand("weights",torch.float32,"[output_size,taps]","out")),
        scalars=(Scalar("input_size"),Scalar("output_size"),Scalar("taps")),key=(filter,bool(antialias)),
        geometry=dict(filter=filter,antialias=bool(antialias),coefficient=-.5 if filter=="bicubic" else None),version=3)


def compile_resize_pass_aot(*,vertical):
    return compile_program(ResizePass(bool(vertical)),name="image_resize_pass",operands=(
        Operand("x",torch.float32,"[input_height,input_width,3]"),Operand("indices",torch.int32,"[output_size]"),
        Operand("sizes",torch.int32,"[output_size]"),Operand("weights",torch.float32,"[output_size,taps]"),
        Operand("out",torch.float32,"[output_height,output_width,3]","out")),
        scalars=(Scalar("input_height"),Scalar("input_width"),Scalar("output_size"),Scalar("taps")),
        key=(bool(vertical),),geometry=dict(channels=3,vertical=bool(vertical),accumulator="float32",clamp=False))
