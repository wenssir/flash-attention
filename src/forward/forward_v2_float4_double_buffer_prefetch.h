#pragma once

#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "../loadstore/load_func.cuh"
#include "../loadstore/store_func.cuh"


using ValueType = float;
using IndexType = int;


template<const IndexType Br, const IndexType Bc, const IndexType d, const IndexType blockSize, const IndexType warpSize>
__global__ void flash_attention_forward_v2_tile_and_prefetch(
    const ValueType* __restrict__ Q,
    const ValueType* __restrict__ K,
    const ValueType* __restrict__ V,
    ValueType* __restrict__ O,
    const IndexType N,
    const IndexType Batch,
    const IndexType Head,
    const IndexType q_stride_b, const IndexType q_stride_h, const IndexType q_stride_n,
    const IndexType k_stride_b, const IndexType k_stride_h, const IndexType k_stride_n,
    const IndexType v_stride_b, const IndexType v_stride_h, const IndexType v_stride_n,
    const IndexType o_stride_b, const IndexType o_stride_h, const IndexType o_stride_n
) {
    // IndexType bid = blockIdx.z;
    // IndexType hid = blockIdx.y;
    // IndexType tid = blockIdx.x; // tile ID

    constexpr int thread_per_row = (d == 64) ? 8 : (d == 128 ? 16 : 32);
    constexpr int rows_per_warp = warpSize / thread_per_row;
    const float softmax_scale = rsqrtf(static_cast<float>(d));

    const float* q_ptr = Q + q_stride_b * blockIdx.z + q_stride_h * blockIdx.y + Br * q_stride_n * blockIdx.x;
    const float* k_ptr = K + k_stride_b * blockIdx.z + k_stride_h * blockIdx.y;
    const float* v_ptr = V + v_stride_b * blockIdx.z + v_stride_h * blockIdx.y;

    __shared__ float qmem[Br * d];
    __shared__ float kmem[2][Bc * d]; // for prefetch
    __shared__ float vmem[2][Bc * d];

    constexpr int warp_count = blockSize / warpSize;
    constexpr int count_per_warp = (Br + warp_count - 1) / warp_count;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const int warp_start_row = warp_id * rows_per_warp;
    const int row_id_in_warp = lane_id / thread_per_row;

    constexpr int block_row_count = rows_per_warp * warp_count;

    constexpr int elem_num_per_thread_in_row = (d + thread_per_row - 1) / thread_per_row;

    const int thread_start_in_row = elem_num_per_thread_in_row * (lane_id % thread_per_row);

    // for register tiling
    float kreg[8];
    float vreg[8];

    float m[count_per_warp];
    float l[count_per_warp];

    float acc_o[count_per_warp][elem_num_per_thread_in_row];

    for (int i = 0; i < count_per_warp; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        for (int j = 0; j < elem_num_per_thread_in_row; ++j) {
            acc_o[i][j] = 0.0f;
        }
    }

    int stage = 0;
    load_tile_sync<float, blockSize, Br, d>(q_ptr, qmem, q_stride_n);
    load_tile_sync<float, blockSize, Bc, d>(k_ptr, kmem[0], k_stride_n);
    load_tile_sync<float, blockSize, Bc, d>(v_ptr, vmem[0], v_stride_n);
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        // prefetch next k/v shared memory tile
        if (j + Bc < N) {
            const float* k_next_ptr = k_ptr + (j + Bc) * k_stride_n;
            const float* v_next_ptr = v_ptr + (j + Bc) * v_stride_n;
            int write_stage = (stage + 1) % 2;
            load_tile_sync<float, blockSize, Bc, d>(k_next_ptr, kmem[write_stage], k_stride_n);
            load_tile_sync<float, blockSize, Bc, d>(v_next_ptr, vmem[write_stage], v_stride_n);
            __syncthreads();
        }

        int local_row_idx = 0;
        for (int current_row = row_id_in_warp + warp_start_row; current_row < Br; current_row += block_row_count) {
            if (current_row < Br) {
                float qreg[elem_num_per_thread_in_row];
                #pragma unroll
                for (int index = 0; index < elem_num_per_thread_in_row; index += 4) {
                    *reinterpret_cast<float4*>(&qreg[index]) = load_float4(qmem, current_row * d + thread_start_in_row + index);
                }
                for (int k = 0; k < Bc; ++k) {
                    float Sij = 0.0;
                    int stride = k * d;
                    *reinterpret_cast<float4*>(&kreg[0]) = load_float4(kmem[stage % 2], k * d + thread_start_in_row);
                    #pragma unroll
                    for (int index = 0; index < elem_num_per_thread_in_row; index += 4) {
                        if (thread_start_in_row + index < d) {
                            if (index < elem_num_per_thread_in_row - 4) {
                                *reinterpret_cast<float4*>(&kreg[(index + 4) % 8]) = load_float4(kmem[stage % 2], stride + thread_start_in_row + index + 4);
                            }
                            Sij += qreg[index] * kreg[index % 8];
                            Sij += qreg[index + 1] * kreg[index % 8 + 1];
                            Sij += qreg[index + 2] * kreg[index % 8 + 2];
                            Sij += qreg[index + 3] * kreg[index % 8 + 3];
                        }
                    }

                    #pragma unroll
                    for (int t = thread_per_row >> 1; t >= 1; t = t >> 1) {
                        Sij += __shfl_xor_sync(0xffffffff, Sij, t);
                    }
                    Sij *= softmax_scale;

                    ValueType m_prev = m[local_row_idx];
                    ValueType l_prev = l[local_row_idx];
                    
                    ValueType m_new = max(m_prev, Sij);

                    ValueType p_val = expf(Sij - m_new);
                    ValueType alpha = expf(m_prev - m_new);

                    l[local_row_idx] = l_prev * alpha + p_val;

                    m[local_row_idx] = m_new;


                    *reinterpret_cast<float4*>(&vreg[0]) = load_float4(vmem[stage % 2], stride + thread_start_in_row);
                    #pragma unroll
                    for (int index = 0; index < elem_num_per_thread_in_row; index += 4) {
                        int col_idx = thread_start_in_row + index;
                        if (col_idx < d) {
                            if (index < elem_num_per_thread_in_row - 4) {
                                *reinterpret_cast<float4*>(&vreg[(index + 4) % 8]) = load_float4(vmem[stage % 2], stride + thread_start_in_row + index + 4);
                            }
                            acc_o[local_row_idx][index] = acc_o[local_row_idx][index] * alpha + p_val * vreg[index % 8];
                            acc_o[local_row_idx][index + 1] = acc_o[local_row_idx][index + 1] * alpha + p_val * vreg[index % 8 + 1];
                            acc_o[local_row_idx][index + 2] = acc_o[local_row_idx][index + 2] * alpha + p_val * vreg[index % 8 + 2];
                            acc_o[local_row_idx][index + 3] = acc_o[local_row_idx][index + 3] * alpha + p_val * vreg[index % 8 + 3];
                        }
                    }
                }
            }
            local_row_idx++;
        }
        __syncthreads();

        stage = (stage + 1) % 2;
    }

    ValueType* o_row_ptr = O + o_stride_b * blockIdx.z + o_stride_h * blockIdx.y + (blockIdx.x * Br * q_stride_n);
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
                    res.x = acc_o[local_row_idx][j] * inv_l;
                    res.y = acc_o[local_row_idx][j + 1] * inv_l;
                    res.z = acc_o[local_row_idx][j + 2] * inv_l;
                    res.w = acc_o[local_row_idx][j + 3] * inv_l;

                    *reinterpret_cast<float4*>(&((o_row_ptr + current_row * q_stride_n)[offset])) = res;
                }
            }
        }
        local_row_idx++;
    }
}

