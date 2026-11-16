"""Hash-bound ATen CUDA coefficient oracle from the installed Torch revision."""
from __future__ import annotations
import functools
import hashlib
import os
from pathlib import Path

REVISION="5aff3928d8"
SHA256="0d91b9e2b3f76a18c2028aa98232c20e45fc7759d40acb60c4385156bf338e1e"


@functools.cache
def coefficients():
    from torch.utils.cpp_extension import load_inline
    path=Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"UpSample-pinned.cuh"
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA256:
        raise ValueError("installed ATen CUDA resize source differs")
    source=raw.decode()
    start=source.index("struct BicubicFilterFunctor")
    end=source.index("template <typename scalar_t, typename accscalar_t>\n__device__ __forceinline__ accscalar_t interpolate_aa_single_dim",start)
    definitions=source[start:end]
    cuda="""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
"""+definitions+"""
__global__ void vendor_weights(int32_t* indices,int32_t* sizes,float* weights,int input,int output,int taps) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=output)return;
 float scale=__fdiv_rn(float(input),float(output));
 float support=scale>=1.f?2.f*scale:2.f;
 int first,count;float center;
 _compute_weights_span(row,input,scale,support,first,count,center);
 indices[row]=first;sizes[row]=count;
 _compute_weights<float,float>(weights+int64_t(row)*taps,scale,taps,BicubicFilterFunctor{},first-center,count);
}
std::vector<torch::Tensor> resize_weights(int64_t input,int64_t output,int64_t taps) {
 auto indices=torch::empty({output},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
 auto sizes=torch::empty_like(indices);
 auto weights=torch::empty({output,taps},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
 vendor_weights<<<(output+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(indices.data_ptr<int32_t>(),sizes.data_ptr<int32_t>(),weights.data_ptr<float>(),input,output,taps);
 return {indices,sizes,weights};
}
"""
    os.environ.setdefault("MAX_JOBS","2")
    module=load_inline(name="pe_aten_weights_"+SHA256[:12],
        cpp_sources="std::vector<torch::Tensor> resize_weights(int64_t input,int64_t output,int64_t taps);",
        cuda_sources=cuda,functions=["resize_weights"],extra_cuda_cflags=["-O2"],verbose=False)
    return module.resize_weights
