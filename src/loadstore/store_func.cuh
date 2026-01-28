#pragma once

#include "../utils/util_func.cuh"

namespace ldst {
template <typename T>
__device__ void store_sm_to_gm(const T* __restrict__ smem, T* __restrict__ gmem) {
    *gmem = *smem;
}

template <typename T>
__device__ void store_sm_to_gm_async(const T* __restrict__ smem, T* __restrict__ gmem) {

}


template<typename T, const int Br, const int d>
__device__ void save_acc_o_to_gm(const T* __restrict__ acc_o, T* __restrict__ gmem, T* __restrict__ l, const int stride, const int row_id_in_warp, const int warp_start_row
                                ,const int thread_start_in_row, const int elem_num_per_thread_in_row, const int block_row_count) {
    int local_row_idx = 0;
    #pragma unroll
    for (int current_row = row_id_in_warp + warp_start_row; current_row < Br; current_row += block_row_count) {
        if (current_row < Br) {
            float inv_l = 1.0f / l[local_row_idx];
            #pragma unroll
            for (int j = 0; j < elem_num_per_thread_in_row; j += 4) {
                const int offset = thread_start_in_row + j;
                if (offset < d) {
                    float4 res;
                    res.x = acc_o[local_row_idx * elem_num_per_thread_in_row + j] * inv_l;
                    res.y = acc_o[local_row_idx * elem_num_per_thread_in_row + j + 1] * inv_l;
                    res.z = acc_o[local_row_idx * elem_num_per_thread_in_row + j + 2] * inv_l;
                    res.w = acc_o[local_row_idx * elem_num_per_thread_in_row + j + 3] * inv_l;
                    
                    *reinterpret_cast<float4*>(&((gmem + current_row * stride)[offset])) = res;
                }
            }
        }
        local_row_idx++;
    }
}


}


