"""Private V4.1 tensor-core score kernel with native append-only overlays.

Pointer-only launch ABI; all pool arithmetic is Int64, live geometry is runtime.
A warp scores sixteen candidates against 32 index heads in four MMA column tiles.
"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from ..._lib.intrinsics import (bf16_mma_m16n8k16_f32, fp4_decode_2,
    f16x2_to_f32x2, u32_as_f32, cvt_f32_to_bf16_bits)
from .mxfp4 import _flat


@cute.jit
def _pair(data, scales, base, sfbase, column):
    lo, hi = f16x2_to_f32x2(fp4_decode_2(Uint32(data[base + column // Int64(2)])))
    sf = Uint32(scales[sfbase + column // Int64(32)])
    scale = u32_as_f32(sf << Uint32(23))
    if sf == Uint32(0):
        scale = Float32(2.0**-127)
    return cvt_f32_to_bf16_bits(lo * scale) | (cvt_f32_to_bf16_bits(hi * scale) << Uint32(16))


class V41OverlayScore:
    @cute.jit
    def __call__(self, q:cute.Pointer, qs:cute.Pointer, weights:cute.Pointer,
                 keys:cute.Pointer, ks:cute.Pointer, pages:cute.Pointer,
                 lengths:cute.Pointer, metadata:cute.Pointer, positions:cute.Pointer,
                 output:cute.Pointer, proposals:cute.Pointer, ps:cute.Pointer,
                 rows:Int32, width:Int32, slots:Int32, stride:Int32,
                 capacity:Int64, proposal_capacity:Int64, stream:cuda.CUstream):
        self.kernel(_flat(q),_flat(qs),_flat(weights),_flat(keys),_flat(ks),
                    _flat(pages),_flat(lengths),_flat(metadata),_flat(positions),
                    _flat(output),_flat(proposals),_flat(ps),width,slots,stride,
                    capacity,proposal_capacity).launch(
                        grid=((width+63)//64,rows,1),block=(128,1,1),stream=stream)

    @cute.kernel
    def kernel(self,q,qs,weights,keys,ks,pages,lengths,metadata,positions,output,
               proposals,ps,width,slots,stride,capacity,proposal_capacity):
        tx,_,_=cute.arch.thread_idx()
        bx,by,_=cute.arch.block_idx()
        row=Int64(by)
        lane=Int32(tx)%32
        group=lane//4
        pair=lane%4
        start_col=Int32(bx)*64+(Int32(tx)//32)*16
        valid=cute.make_rmem_tensor((2,),Int32)
        physical=cute.make_rmem_tensor((2,),Int64)
        proposed=cute.make_rmem_tensor((2,),Int32)
        slot=metadata[row*6]
        causal=metadata[row*6+1]
        start=metadata[row*6+2]
        count=metadata[row*6+3]
        offset=metadata[row*6+4]
        step=metadata[row*6+5]
        for part in cutlass.range_constexpr(2):
            col=start_col+group+part*8
            valid[part]=Int32(0)
            physical[part]=Int64(0)
            proposed[part]=Int32(0)
            if col<width and slot>=0 and slot<Int64(slots):
                pos=positions[row*Int64(width)+Int64(col)]
                committed=lengths[slot]
                descriptor=(start==committed and start>=0 and start<=1048576 and
                    count>=0 and count<=1048576-start and (step==1 or step==2) and
                    offset>=0 and offset<=proposal_capacity)
                if count>0:
                    descriptor=descriptor and offset<proposal_capacity
                    if offset<proposal_capacity and step>0:
                        descriptor=descriptor and count-1<=(proposal_capacity-1-offset)//step
                if descriptor and pos>=0 and pos<causal:
                    if pos<committed:
                        if pos<Int64(stride)*256:
                            p=Int64(pages[slot*Int64(stride)+pos//256])*256+pos%256
                            if p>=0 and p<capacity:
                                physical[part]=p
                                valid[part]=Int32(1)
                    elif pos-committed<count:
                        physical[part]=offset+(pos-committed)*step
                        proposed[part]=Int32(1)
                        valid[part]=Int32(1)
        live=cute.arch.vote_ballot_sync((valid[0]!=0) | (valid[1]!=0))
        if live!=0:
            acc=cute.make_rmem_tensor((4,4),Float32)
            acc.fill(Float32(0))
            for tile in cutlass.range_constexpr(8):
                a=cute.make_rmem_tensor((4,),Uint32)
                a.fill(Uint32(0))
                column=Int64(tile*16+pair*2)
                for part in cutlass.range_constexpr(2):
                    if valid[part]!=0:
                        p=physical[part]
                        if proposed[part]!=0:
                            a[part]=_pair(proposals,ps,p*64,p*4,column)
                            a[part+2]=_pair(proposals,ps,p*64,p*4,column+8)
                        else:
                            a[part]=_pair(keys,ks,p*64,p*4,column)
                            a[part+2]=_pair(keys,ks,p*64,p*4,column+8)
                for chunk in cutlass.range_constexpr(4):
                    head=Int64(chunk*8+group)
                    qb=(row*32+head)*64
                    sb=(row*32+head)*4
                    b0=_pair(q,qs,qb,sb,column)
                    b1=_pair(q,qs,qb,sb,column+8)
                    d0,d1,d2,d3=bf16_mma_m16n8k16_f32(
                        acc[chunk,0],acc[chunk,1],acc[chunk,2],acc[chunk,3],
                        a[0],a[1],a[2],a[3],b0,b1)
                    acc[chunk,0]=d0
                    acc[chunk,1]=d1
                    acc[chunk,2]=d2
                    acc[chunk,3]=d3
            for part in cutlass.range_constexpr(2):
                vals=cute.make_rmem_tensor((4,2),Float32)
                for chunk in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(2):
                        dot=Float32(BFloat16(acc[chunk,part*2+h]))
                        weight=Float32(weights[row*32+Int64(chunk*8+pair*2+h)])
                        vals[chunk,h]=Float32(BFloat16(cutlass.max(dot,Float32(0))*weight))
                # Same head-sum tree as the native 32-lane downward reduction.
                x0=(vals[0,0]+vals[2,0])+(vals[1,0]+vals[3,0])
                x1=(vals[0,1]+vals[2,1])+(vals[1,1]+vals[3,1])
                x0=x0+cute.arch.shuffle_sync_bfly(x0,offset=2)
                x1=x1+cute.arch.shuffle_sync_bfly(x1,offset=2)
                x0=x0+cute.arch.shuffle_sync_bfly(x0,offset=1)
                x1=x1+cute.arch.shuffle_sync_bfly(x1,offset=1)
                result=Float32(BFloat16(x0+x1))
                col=start_col+group+part*8
                if pair==0 and col<width:
                    if valid[part]==0:
                        result=Float32(-float('inf'))
                    output[row*Int64(width)+Int64(col)]=result
        else:
            for part in cutlass.range_constexpr(2):
                col=start_col+group+part*8
                if pair==0 and col<width:
                    output[row*Int64(width)+Int64(col)]=Float32(-float('inf'))
