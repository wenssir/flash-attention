#pragma once

#include <cuda_runtime.h>
#include <cuda/pipeline>

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

    constexpr int d4 = d / 4;

    constexpr int thread_per_row = (d == 64) ? 8 : (d == 128 ? 16 : 32);
    constexpr int rows_per_warp = warpSize / thread_per_row;

    const float4* q_ptr = (float4*)(Q + q_stride_b * blockIdx.z + q_stride_h * blockIdx.y + Br * q_stride_n * blockIdx.x);
    const float4* k_ptr = (float4*)(K + k_stride_b * blockIdx.z + k_stride_h * blockIdx.y);
    const float4* v_ptr = (float4*)(V + v_stride_b * blockIdx.z + v_stride_h * blockIdx.y);

    // float4* o_ptr = (float4*)(O + o_stride_b * blockIdx.z + o_stride_h * blockIdx.y + Br * q_stride_n * blockIdx.x);

    __shared__ float4 qmem[Br * d4];
    __shared__ float4 kmem[2][Bc * d4]; // for prefetch
    __shared__ float4 vmem[2][Bc * d4];

    constexpr int warp_count = blockSize / warpSize;
    constexpr int count_per_warp = (Br + warp_count - 1) / warp_count;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const int warp_start_row = warp_id * rows_per_warp;
    const int row_id_in_warp = lane_id / thread_per_row;
    const int thread_id_in_row = lane_id % thread_per_row;


    constexpr int elem_num_per_thread_in_row = (d4 + thread_per_row - 1) / thread_per_row;

    // for register tiling
    float4 kreg[2];
    float4 vreg[2];

    float m[count_per_warp];
    float l[count_per_warp];

    float acc_o[count_per_warp][elem_num_per_thread_in_row * 4];

    for (int i = 0; i < count_per_warp; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        for (int j = 0; j < elem_num_per_thread_in_row * 4; ++j) {
            acc_o[i][j] = 0.0f;
        }
    }

    int stage = 0;
    const int total_q_elems = Br * d4;
    #pragma unroll
    for (int idx = threadIdx.x; idx < total_q_elems; idx += blockSize) {
        int row = idx / d4;
        int col = idx % d4;
        qmem[idx] = *(q_ptr + row * (q_stride_n / 4) + col);
    }

    // prefetch first k/v shared memory tile
    const int total_kv_elems = Bc * d4;
    #pragma unroll
    for (int idx = threadIdx.x; idx < total_kv_elems; idx += blockSize) {
        int row = idx / d4;
        int col = idx % d4;
        kmem[0][idx] = *(k_ptr + row * (k_stride_n / 4) + col);
        vmem[0][idx] = *(v_ptr + row * (v_stride_n / 4) + col);
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        // prefetch next k/v shared memory tile
        if (j < N - Bc) {
            const float4* k_next_ptr = k_ptr + (j + Bc) * (k_stride_n / 4);
            const float4* v_next_ptr = v_ptr + (j + Bc) * (v_stride_n / 4);
            int write_stage = (stage + 1) % 2;
            #pragma unroll
            for (int idx = threadIdx.x; idx < total_kv_elems; idx += blockSize) {
                int row = idx / d4;
                int col = idx % d4;
                kmem[write_stage][idx] = *(k_next_ptr + row * (k_stride_n / 4) + col);
                vmem[write_stage][idx] = *(v_next_ptr + row * (v_stride_n / 4) + col);
            }
            __syncthreads();
        }

        int local_row_idx = 0;
        for (int i = warp_start_row; i < Br; i += rows_per_warp * warp_count) {
            int current_row = row_id_in_warp + i;
            if (current_row < Br) {
                float4 qreg[elem_num_per_thread_in_row];
                #pragma unroll
                for (int j = 0; j < elem_num_per_thread_in_row; ++j) {
                    qreg[j] = qmem[current_row * d4 + thread_id_in_row * elem_num_per_thread_in_row + j];
                }
                for (int k = 0; k < Bc; ++k) {
                    float Sij = 0.0;
                    kreg[0] = kmem[stage % 2][k * d4 + thread_id_in_row * elem_num_per_thread_in_row];
                    #pragma unroll
                    for (int j = 0; j < elem_num_per_thread_in_row; ++j) {
                        if (thread_id_in_row * elem_num_per_thread_in_row + j < d4) {
                            if (j < elem_num_per_thread_in_row - 1) {
                                kreg[(j + 1) % 2] = kmem[stage % 2][k * d4 + thread_id_in_row * elem_num_per_thread_in_row + j + 1];
                            }
                            Sij += qreg[j].x * kreg[j % 2].x;
                            Sij += qreg[j].y * kreg[j % 2].y;
                            Sij += qreg[j].z * kreg[j % 2].z;
                            Sij += qreg[j].w * kreg[j % 2].w;
                        }
                    }

                    Sij += __shfl_xor_sync(0xffffffff, Sij, 4);
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 2);
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 1);

                    ValueType m_prev = m[local_row_idx];
                    ValueType l_prev = l[local_row_idx];
                    
                    ValueType m_new = max(m_prev, Sij);

                    ValueType p_val = expf(Sij - m_new);
                    ValueType alpha = expf(m_prev - m_new);

                    l[local_row_idx] = l_prev * alpha + p_val;

                    m[local_row_idx] = m_new;


                    vreg[0] = vmem[stage % 2][k * d4 + thread_id_in_row * elem_num_per_thread_in_row];
                    #pragma unroll
                    for (int index = 0; index < elem_num_per_thread_in_row; ++index) {
                        int col_idx = thread_id_in_row * elem_num_per_thread_in_row + index;
                        if (col_idx < d4) {
                            if (index < elem_num_per_thread_in_row - 1) {
                                vreg[(index + 1) % 2] = vmem[stage % 2][k * d4 + thread_id_in_row * elem_num_per_thread_in_row + index + 1];
                            }
                            acc_o[local_row_idx][index * 4] = acc_o[local_row_idx][index * 4] * alpha + p_val * vreg[index % 2].x;
                            acc_o[local_row_idx][index * 4 + 1] = acc_o[local_row_idx][index * 4 + 1] * alpha + p_val * vreg[index % 2].y;
                            acc_o[local_row_idx][index * 4 + 2] = acc_o[local_row_idx][index * 4 + 2] * alpha + p_val * vreg[index % 2].z;
                            acc_o[local_row_idx][index * 4 + 3] = acc_o[local_row_idx][index * 4 + 3] * alpha + p_val * vreg[index % 2].w;
                        }
                    }

                }
            }
            local_row_idx++;
        }
        __syncthreads();

        stage = (stage + 1) % 2;
    }


    int local_row_idx = 0;
    int stride = rows_per_warp * warp_count;
    #pragma unroll
    for (int i = warp_start_row; i < Br; i += stride) {
        int current_row = row_id_in_warp + i;
        ValueType* o_row_ptr = (float*)O + o_stride_b * blockIdx.z + o_stride_h * blockIdx.y + (blockIdx.x * Br * q_stride_n) + (current_row * q_stride_n);

        if (current_row < Br) {
            float inv_l = 1.0f / l[local_row_idx];
            #pragma unroll
            for (int j = 0; j < elem_num_per_thread_in_row; ++j) {
                const int offset = thread_id_in_row * elem_num_per_thread_in_row + j;
                if (offset < d4) {
                    float4 res;
                    res.x = acc_o[local_row_idx][j * 4] * inv_l;
                    res.y = acc_o[local_row_idx][j * 4 + 1] * inv_l;
                    res.z = acc_o[local_row_idx][j * 4 + 2] * inv_l;
                    res.w = acc_o[local_row_idx][j * 4 + 3] * inv_l;
                    
                    ((float4*)o_row_ptr)[offset] = res;
                }
            }
        }
        local_row_idx++;
    }
}


