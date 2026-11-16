"""PE centered LayerNorm, native patch rows and learned position/CLS boundaries."""
from __future__ import annotations
import math
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16,Float32,Int32,Int64
from b12x._lib.intrinsics import div_rn_f32,rsqrt_approx_ftz_f32
from ._common import Operand,Scalar,compile_program
from ._glm_kernels import _warp_sum
from .pointwise import _add,_sub,_mul,_fma,_sqrt_rn,_view


class CheckedForward:
    """Host validation for pointer ABI tests; native owners admit the same extents."""
    def __init__(self,program,kind,**geometry):
        self.program,self.kind,self.geometry=program,kind,geometry

    def __getattr__(self,name):
        return getattr(self.program,name)

    def launch(self,*tensors,scalars,stream=None):
        if self.kind=="norm":
            if len(scalars)!=2 or scalars[0]<=0 or not math.isfinite(scalars[1]) or scalars[1]<=0:
                raise ValueError("PE norm requires positive rows and finite positive epsilon")
            rows,epsilon=scalars;hidden=self.geometry["hidden"]
            expected=[rows*hidden,hidden,hidden,rows*hidden]
        else:
            if len(scalars)!=2 or not all(isinstance(value,int) and value>0 for value in scalars):
                raise ValueError("PE image extents must be positive integers")
            height,width=scalars
            if height*width>16384:
                raise ValueError("PE image grid exceeds admitted capacity")
            if self.kind=="patch":
                patch=self.geometry["patch"]
                if height%patch or width%patch:
                    raise ValueError("PE normalized image extents must divide the patch size")
                expected=[height*width*3,height*width*3]
            else:
                hidden,grid=self.geometry["hidden"],self.geometry["grid"]
                expected=[height*width*hidden,grid*grid*hidden,hidden,hidden,
                    (height*width+int(self.geometry["class_token"]))*hidden]
        if len(tensors)!=len(expected) or any(tensor is None or tensor.numel()<size for tensor,size in zip(tensors,expected)):
            raise ValueError("PE forward operand storage below admitted extent")
        return self.program.launch(*tensors,scalars=scalars,stream=stream)


class CenteredLayerNorm:
    def __init__(self,width,vendor_rsqrt):
        self.width,self.vendor_rsqrt=width,vendor_rsqrt

    @cute.jit
    def __call__(self,x:cute.Pointer,weight:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,
                 rows:Int32,epsilon:Float32,stream:cuda.CUstream):
        self.kernel(x,weight,bias,out,epsilon).launch(grid=(rows,1,1),block=(32,1,1),stream=stream)

    @cute.kernel
    def kernel(self,x:cute.Pointer,weight:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,epsilon:Float32):
        x,weight,bias,out=_view(x),_view(weight),_view(bias),_view(out)
        row=Int64(cute.arch.block_idx()[0]);lane=Int32(cute.arch.thread_idx()[0])
        base=row*Int64(self.width)
        total=Float32(0.0)
        for item in cutlass.range_constexpr((self.width+31)//32):
            col=lane+Int32(item*32)
            if col<Int32(self.width):
                total=_add(total,Float32(x[base+Int64(col)]))
        mean=div_rn_f32(_warp_sum(total),Float32(self.width))
        squares=Float32(0.0)
        for item in cutlass.range_constexpr((self.width+31)//32):
            col=lane+Int32(item*32)
            if col<Int32(self.width):
                difference=_sub(Float32(x[base+Int64(col)]),mean)
                squares=_add(squares,_mul(difference,difference))
        variance=_add(div_rn_f32(_warp_sum(squares),Float32(self.width)),epsilon)
        inv=Float32(0.0)
        if cutlass.const_expr(self.vendor_rsqrt):
            # Inkling's pinned Triton _layer_norm explicitly uses tl.math.rsqrt.
            inv=rsqrt_approx_ftz_f32(variance)
        else:
            inv=div_rn_f32(Float32(1.0),_sqrt_rn(variance))
        for item in cutlass.range_constexpr((self.width+31)//32):
            col=lane+Int32(item*32)
            if col<Int32(self.width):
                value=_mul(_sub(Float32(x[base+Int64(col)]),mean),inv)
                out[base+Int64(col)]=_fma(Float32(weight[col]),value,Float32(bias[col])).to(BFloat16)


class InklingLayerNorm:
    """Pinned Triton layout: four warps, eight adjacent values per thread;
    local order 1,0,2..7, descending XOR16..1, then cross-warp XOR2,1.
    Squared differences use the vendor PTX's explicit FMA accumulation.
    """
    @cute.jit
    def __call__(self,x:cute.Pointer,weight:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,
                 rows:Int32,epsilon:Float32,stream:cuda.CUstream):
        self.kernel(x,weight,bias,out,epsilon).launch(grid=(rows,1,1),block=(128,1,1),stream=stream)

    @cute.jit
    def reduce(self,value:Float32,partials:cute.Tensor,lane:Int32,warp:Int32):
        for shift in cutlass.range_constexpr(5):
            value=_add(value,Float32(cute.arch.shuffle_sync_bfly(value,offset=16>>shift)))
        if lane==Int32(0):partials[warp]=value
        cute.arch.sync_threads()
        total=Float32(partials[lane%Int32(4)])
        total=_add(total,Float32(cute.arch.shuffle_sync_bfly(total,offset=2)))
        total=_add(total,Float32(cute.arch.shuffle_sync_bfly(total,offset=1)))
        cute.arch.sync_threads()
        return total

    @cute.kernel
    def kernel(self,x:cute.Pointer,weight:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,epsilon:Float32):
        x,weight,bias,out=_view(x),_view(weight),_view(bias),_view(out)
        row=Int64(cute.arch.block_idx()[0]);tid=Int32(cute.arch.thread_idx()[0])
        lane,warp=tid%32,tid//32;base=row*1024+Int64(tid)*8
        values=[]
        for item in cutlass.range_constexpr(8):values.append(Float32(x[base+item]))
        partials=cutlass.utils.SmemAllocator().allocate_tensor(Float32,cute.make_layout((4,)))
        total=_add(values[0],values[1])
        for item in cutlass.range_constexpr(2,8):total=_add(total,values[item])
        mean=div_rn_f32(self.reduce(total,partials,lane,warp),Float32(1024.0))
        differences=[]
        for item in cutlass.range_constexpr(8):differences.append(_sub(values[item],mean))
        squares=_mul(differences[1],differences[1])
        squares=_fma(differences[0],differences[0],squares)
        for item in cutlass.range_constexpr(2,8):squares=_fma(differences[item],differences[item],squares)
        variance=_add(div_rn_f32(self.reduce(squares,partials,lane,warp),Float32(1024.0)),epsilon)
        inv=rsqrt_approx_ftz_f32(variance)
        for item in cutlass.range_constexpr(8):
            col=tid*8+item
            out[base+item]=_fma(Float32(weight[col]),_mul(differences[item],inv),Float32(bias[col])).to(BFloat16)


class NormalizedPatches:
    def __init__(self,patch,hwc):
        self.patch,self.hwc=patch,hwc

    @cute.jit
    def __call__(self,pixels:cute.Pointer,out:cute.Pointer,height:Int32,width:Int32,stream:cuda.CUstream):
        count=Int64(height)*Int64(width)*Int64(3)
        if height%Int32(self.patch)!=Int32(0) or width%Int32(self.patch)!=Int32(0):
            count=Int64(0)
        self.kernel(pixels,out,width,count).launch(grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def kernel(self,pixels:cute.Pointer,out:cute.Pointer,width:Int32,count:Int64):
        pixels,out=_view(pixels),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            features=Int64(self.patch*self.patch*3)
            row,local=at//features,at%features
            channel=local//Int64(self.patch*self.patch)
            pixel=local%Int64(self.patch*self.patch)
            if cutlass.const_expr(self.hwc):
                channel=local%3;pixel=local//3
            gw=Int64(width//Int32(self.patch))
            y=(row//gw)*Int64(self.patch)+pixel//Int64(self.patch)
            x=(row%gw)*Int64(self.patch)+pixel%Int64(self.patch)
            out[at]=Float32(pixels[(y*Int64(width)+x)*3+channel]).to(BFloat16)


class LearnedPosition:
    def __init__(self,hidden,grid,class_token):
        self.hidden,self.grid,self.class_token=hidden,grid,class_token

    @cute.jit
    def __call__(self,patch:cute.Pointer,table:cute.Pointer,cls:cute.Pointer,cls_position:cute.Pointer,
                 out:cute.Pointer,height:Int32,width:Int32,stream:cuda.CUstream):
        count=(Int64(height)*Int64(width)+Int64(self.class_token))*Int64(self.hidden)
        self.kernel(patch,table,cls,cls_position,out,height,width,count).launch(
            grid=((count+255)//256,1,1),block=(256,1,1),stream=stream)

    @cute.jit
    def lerp(self,a:Float32,b:Float32,wa:Float32,wb:Float32):
        value=_fma(a,wa,_mul(b,wb))
        if wa<wb:
            value=_fma(b,wb,_mul(a,wa))
        return value

    @cute.kernel
    def kernel(self,patch:cute.Pointer,table:cute.Pointer,cls:cute.Pointer,cls_position:cute.Pointer,
               out:cute.Pointer,height:Int32,width:Int32,count:Int64):
        patch,table,cls,cls_position,out=_view(patch),_view(table),_view(cls),_view(cls_position),_view(out)
        at=Int64(cute.arch.block_idx()[0])*256+Int64(cute.arch.thread_idx()[0])
        if at<count:
            row,col=at//Int64(self.hidden),at%Int64(self.hidden)
            p=row-Int64(self.class_token)
            if cutlass.const_expr(self.class_token):
                if row==Int64(0):
                    out[at]=_add(Float32(cls[col]),Float32(cls_position[col])).to(BFloat16)
                else:
                    self.patch_value(p,col,patch,table,out,at,height,width)
            else:
                self.patch_value(p,col,patch,table,out,at,height,width)

    @cute.jit
    def patch_value(self,p:Int64,col:Int64,patch:cute.Tensor,table:cute.Tensor,out:cute.Tensor,
                    at:Int64,height:Int32,width:Int32):
        y=div_rn_f32(_mul(Float32(self.grid),_add(Float32(p//Int64(width)),Float32(.5))),Float32(height))
        x=div_rn_f32(_mul(Float32(self.grid),_add(Float32(p%Int64(width)),Float32(.5))),Float32(width))
        y=_sub(y,Float32(.5));x=_sub(x,Float32(.5))
        if y<Float32(0.0):y=Float32(0.0)
        if x<Float32(0.0):x=Float32(0.0)
        if y>Float32(self.grid-1):y=Float32(self.grid-1)
        if x>Float32(self.grid-1):x=Float32(self.grid-1)
        y0,x0=Int64(cute.math.floor(y)),Int64(cute.math.floor(x))
        y1,x1=y0+1,x0+1
        if y1>=Int64(self.grid):y1=Int64(self.grid-1)
        if x1>=Int64(self.grid):x1=Int64(self.grid-1)
        wy,wx=_sub(y,Float32(y0)),_sub(x,Float32(x0))
        top=self.lerp(Float32(table[(y0*self.grid+x0)*self.hidden+col]),
            Float32(table[(y0*self.grid+x1)*self.hidden+col]),_sub(Float32(1.0),wx),wx)
        bottom=self.lerp(Float32(table[(y1*self.grid+x0)*self.hidden+col]),
            Float32(table[(y1*self.grid+x1)*self.hidden+col]),_sub(Float32(1.0),wx),wx)
        position=self.lerp(top,bottom,_sub(Float32(1.0),wy),wy).to(BFloat16)
        out[at]=_add(Float32(patch[p*Int64(self.hidden)+col]),Float32(position)).to(BFloat16)


def compile_layer_norm_aot(*,hidden,vendor_rsqrt=False):
    if hidden<=0 or hidden>16384:
        raise ValueError("PE norm width must be in1..16384")
    if vendor_rsqrt and hidden!=1024:
        raise ValueError("pinned Inkling LayerNorm requires width1024")
    launch=InklingLayerNorm() if vendor_rsqrt else CenteredLayerNorm(hidden,False)
    return CheckedForward(compile_program(launch,name="perception_centered_norm",operands=(
        Operand("x",torch.bfloat16,f"[rows,{hidden}]"),Operand("weight",torch.bfloat16,f"[{hidden}]"),
        Operand("bias",torch.bfloat16,f"[{hidden}]"),Operand("out",torch.bfloat16,f"[rows,{hidden}]","out")),
        scalars=(Scalar("rows"),Scalar("epsilon","float32")),key=(hidden,bool(vendor_rsqrt)),
        geometry=dict(hidden=hidden,centered_variance=True,vendor_rsqrt=bool(vendor_rsqrt)),version=2),"norm",hidden=hidden)


def compile_normalized_patches_aot(*,patch,hwc=False):
    if patch<=0 or patch>64:
        raise ValueError("patch size must be in1..64")
    return CheckedForward(compile_program(NormalizedPatches(patch,bool(hwc)),name="perception_normalized_patches",operands=(
        Operand("pixels",torch.float32,"[height,width,3]"),Operand("out",torch.bfloat16,f"[height*width/{patch*patch},{3*patch*patch}]","out")),
        scalars=(Scalar("height"),Scalar("width")),key=(patch,bool(hwc)),geometry=dict(patch=patch,hwc=bool(hwc)),version=2),"patch",patch=patch)


def compile_learned_position_aot(*,hidden,grid,class_token=False):
    if min(hidden,grid)<=0:
        raise ValueError("PE position geometry must be positive")
    return CheckedForward(compile_program(LearnedPosition(hidden,grid,bool(class_token)),name="perception_learned_position",operands=(
        Operand("patch",torch.bfloat16,f"[height*width,{hidden}]"),Operand("table",torch.bfloat16,f"[{grid*grid},{hidden}]"),
        Operand("cls",torch.bfloat16,f"[{hidden}]"),Operand("cls_position",torch.bfloat16,f"[{hidden}]"),
        Operand("out",torch.bfloat16,f"[height*width+{int(class_token)},{hidden}]","out")),
        scalars=(Scalar("height"),Scalar("width")),key=(hidden,grid,bool(class_token)),
        geometry=dict(hidden=hidden,grid=grid,class_token=bool(class_token),mode="bilinear")),"position",hidden=hidden,grid=grid,class_token=bool(class_token))
