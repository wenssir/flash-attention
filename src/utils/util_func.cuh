#pragma once

template <typename T>
__device__ __forceinline__ float4 load_float4(const T* __restrict__ ptr, int cur) {
    return *reinterpret_cast<const float4*>(ptr + cur);
}