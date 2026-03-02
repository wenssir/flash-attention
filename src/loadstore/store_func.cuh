#pragma once

#include "../config/macros.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace loadstore {
template <typename T>
DEVICE void store_sm_to_gm(const T* __restrict__ smem, T* __restrict__ gmem) {
    *gmem = *smem;
}

template <typename T, int Vec=4>
DEVICE void copy_s2g_vec(T const* s_ptr, T* g_ptr, bool guard) {

}

template <typename Element, typename index_t, int FragCols>
DEVICE void store_row_fragment(Element* __restrict__ O,
                               index_t o_base,
                               int global_row,
                               int stride_seq_o,
                               int col_base,
                               const float (&acc)[FragCols],
                               float inv_l) {
    Element* out_ptr = O + o_base + static_cast<index_t>(global_row) * stride_seq_o + col_base;

    if constexpr (std::is_same_v<Element, __half>) {
        constexpr int kVec = 8;
        static_assert((FragCols % kVec) == 0, "FragCols must be multiple of 8 for half vector store");

        #pragma unroll
        for (int d0 = 0; d0 < FragCols; d0 += kVec) {
            __align__(16) __half pack[kVec];
            #pragma unroll
            for (int i = 0; i < kVec; ++i) {
                pack[i] = __float2half(acc[d0 + i] * inv_l);
            }
            auto* dst = out_ptr + d0;
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(pack);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < FragCols; ++d) {
            out_ptr[d] = static_cast<Element>(acc[d] * inv_l);
        }
    }
}

} // namespace loadstore
