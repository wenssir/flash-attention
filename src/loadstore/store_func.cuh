#pragma once

#include "../config/macros.cuh"

namespace ldst {
template <typename T>
DEVICE void store_sm_to_gm(const T* __restrict__ smem, T* __restrict__ gmem) {
    *gmem = *smem;
}

template <typename T, int Vec=4>
DEVICE void copy_s2g_vec(T const* s_ptr, T* g_ptr, bool guard) {

}


}

