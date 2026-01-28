#pragma once

#include "../utils/util_func.cuh"

template<typename T, const int BLOCK_SIZE, const int BLOCK_M, const int HEAD_DIM>
__device__ void load_tile_sync(const T* __restrict__ gmem, T* smem, int stride) {
    int d4 = HEAD_DIM >> 2;
    const int total_q_elems = BLOCK_M * d4;
    #pragma unroll
    for (int idx = threadIdx.x; idx < total_q_elems; idx += BLOCK_SIZE) {
        int row = idx / d4;
        int col = idx % d4;
        reinterpret_cast<float4*>(smem)[idx] = load_float4(gmem, row * stride + col * 4);
    }
}

template<typename T>
void load_sync(const T* gmem, const T* smem) {
    *smem = *gmem;
}

template<typename T>
void load_async(const T* gmem, const T* smem) {
    asm volatile("cp.async.cg.shared.global.L2.128B [%1], [%2], 16"
                :
                : "r"(gmem), "r"(smem)
                : "memory");
}

template<typename T>
void load_tensor_core(const T* gmem, const T* smem) {

}

template<typename T>
void load_tma(const T* gmem, const T* smem) {
    
}