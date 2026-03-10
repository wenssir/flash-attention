#pragma once

#include "../config/macros.cuh"

template <typename T>
__device__ __forceinline__ float4 load_float4(const T* __restrict__ ptr, int cur) {
    return *reinterpret_cast<const float4*>(ptr + cur);
}

namespace forward {

template <typename KernelConfig>
HOST_DEVICE constexpr int forward_smem_bytes() {
    constexpr int q_bytes = KernelConfig::BlockM * KernelConfig::HeadDim *
                            static_cast<int>(sizeof(typename KernelConfig::Element));
    constexpr int kv_bytes = 2 * KernelConfig::BlockN * KernelConfig::HeadDim *
                             static_cast<int>(sizeof(typename KernelConfig::Element));
    constexpr int softmax_state_bytes = 2 * KernelConfig::NWarps * KernelConfig::MmaM *
                                        static_cast<int>(sizeof(float));
    return q_bytes + kv_bytes + softmax_state_bytes;
}

} // namespace forward
