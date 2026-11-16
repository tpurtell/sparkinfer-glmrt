// Exact codec oracle copied from cuteafd native/shared/cuda/draft_kv.cuh.
// The containing test checks byte parity, including scales and high offsets.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

template<int heads>
__global__ void codec(const __nv_bfloat16* k, const __nv_bfloat16* v,
                      const int64_t* slots, uint8_t* kp, uint8_t* vp) {
  const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  const int64_t slot = slots[row];
  if (slot < 0) return;
  constexpr uint64_t bytes = (heads * 132 + 15) / 16 * 16;
  for (int plane = 0; plane < 2; ++plane) {
    const auto* source = plane ? v : k;
    auto* record = (plane ? vp : kp) + uint64_t(slot) * bytes;
    float values[4], amax = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      values[j] = __bfloat162float(source[(uint64_t(row) * heads + head) * 128 + lane + j * 32]);
      amax = fmaxf(amax, fabsf(values[j]));
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    const float scale = amax > 0 ? __fdiv_rn(amax, 448.f) : 1.f;
    const float inverse = __fdiv_rn(1.f, scale);
    if (!lane) reinterpret_cast<float*>(record + heads * 128)[head] = scale;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = values[j] * inverse;
      const float x1 = __shfl_xor_sync(0xffffffffu, x, 1);
      const float x2 = __shfl_xor_sync(0xffffffffu, x, 2);
      const float x3 = __shfl_xor_sync(0xffffffffu, x1, 2);
      if (lane % 4 == 0) {
        const auto lo = __nv_cvt_float2_to_fp8x2(make_float2(x, x1), __NV_SATFINITE, __NV_E4M3);
        const auto hi = __nv_cvt_float2_to_fp8x2(make_float2(x2, x3), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t*>(record + head * 128 + lane + j * 32) = uint32_t(lo) | (uint32_t(hi) << 16);
      }
    }
    if (!head && lane < bytes - heads * 132) record[heads * 132 + lane] = 0;
  }
}
extern "C" int launch_codec(void* k, void* v, void* slots, void* kp, void* vp,
                             int rows, int heads, void* stream) {
  if (heads == 8) codec<8><<<dim3(rows,8),32,0,(cudaStream_t)stream>>>((__nv_bfloat16*)k,(__nv_bfloat16*)v,(int64_t*)slots,(uint8_t*)kp,(uint8_t*)vp);
  else codec<4><<<dim3(rows,4),32,0,(cudaStream_t)stream>>>((__nv_bfloat16*)k,(__nv_bfloat16*)v,(int64_t*)slots,(uint8_t*)kp,(uint8_t*)vp);
  return cudaGetLastError();
}
