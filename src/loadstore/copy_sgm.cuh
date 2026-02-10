#pragma once

#include "../config/macros.cuh"

namespace loadstore {

template <typename TensorS, typename TensorG>
DEVICE void copy_s2g(TensorS const& smem, TensorG& gmem) {
    int tid = threadIdx.x;
    int n = static_cast<int>(gmem.size());
    auto const* s_ptr = smem.data_ptr();
    auto* g_ptr = gmem.data_ptr();

    for (int i = tid; i < n; i += blockDim.x) {
        g_ptr[i] = s_ptr[i];
    }
}

struct CopyS2GOp {
    template <typename TensorG, typename TensorS>
    DEVICE void operator()(TensorG& gmem, TensorS const& smem) const {
        copy_s2g(smem, gmem);
    }
};

} // namespace loadstore
