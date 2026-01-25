#pragma once

#include <cuda_runtime.h>

using ValueType = float;
using IndexType = int;

template<const IndexType Br, const IndexType Bc, const IndexType d, const IndexType blockSize, const IndexType warpSize>
__global__ void flash_attention_v2_naive(
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
    IndexType bid = blockIdx.z;
    IndexType hid = blockIdx.y;
    IndexType tid = blockIdx.x; // tile ID

    const IndexType q_tile_stride = Br * q_stride_n;

    IndexType q_offset = q_stride_b * bid + q_stride_h * hid + q_tile_stride * tid;
    IndexType k_offset = k_stride_b * bid + k_stride_h * hid;
    IndexType v_offset = v_stride_b * bid + v_stride_h * hid;
    IndexType o_offset = o_stride_b * bid + o_stride_h * hid;

    const ValueType* q_ptr = Q + q_offset;
    const ValueType* k_ptr = K + k_offset;
    const ValueType* v_ptr = V + v_offset;

    ValueType* o_ptr = O + o_offset + Br * q_stride_n * tid;

    __shared__ ValueType qmem[Br * d];
    __shared__ ValueType kmem[Bc * d];
    __shared__ ValueType vmem[Bc * d];

    IndexType per_thread_row_count = (d + blockSize - 1) / blockSize;

    const IndexType warp_count = blockSize / warpSize;
    constexpr int rows_per_warp = (Br + warp_count - 1) / warp_count;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    constexpr int per_thread_row_count_warp = (d + warpSize - 1) / warpSize;

    float m[rows_per_warp];
    float l[rows_per_warp];

    float acc_o[rows_per_warp][per_thread_row_count_warp];

    for (int i = 0; i < rows_per_warp; ++i) {
        m[i] = -INFINITY; // 初始化为负无穷
        l[i] = 0.0f;
        for (int j = 0; j < per_thread_row_count_warp; ++j) {
            acc_o[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < Br; ++i) {
        const ValueType* tmp_ptr = q_ptr + i * q_stride_n + threadIdx.x * per_thread_row_count;
        for (int j = 0; j < per_thread_row_count; ++j) {
            qmem[i * d + threadIdx.x * per_thread_row_count + j] = *(tmp_ptr + j);
        }
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        const ValueType* k_tile_ptr = k_ptr + k_stride_n * j;
        const ValueType* v_tile_ptr = v_ptr + v_stride_n * j;

        for (int i = 0; i < Bc; ++i) {
            const IndexType row_offset_k = i * k_stride_n + threadIdx.x * per_thread_row_count;
            const IndexType row_offset_v = i * v_stride_n + threadIdx.x * per_thread_row_count;
            for (int index = 0; index < per_thread_row_count; ++index) {
                if (threadIdx.x * per_thread_row_count + index < d) {
                    kmem[i * d + threadIdx.x * per_thread_row_count + index] = *(k_tile_ptr + row_offset_k + index);
                    vmem[i * d + threadIdx.x * per_thread_row_count + index] = *(v_tile_ptr + row_offset_v + index);
                }
            }
        }
        __syncthreads();

        for (int i = 0; i < Bc; ++i) {
            for (int j = 0; j <  rows_per_warp; ++j) { // j 代表第几行
                int t = warp_id * rows_per_warp + j;
                ValueType Sij = 0;
                if (t < Br) {
                    for (int index = lane_id * per_thread_row_count_warp; index < (lane_id + 1) * per_thread_row_count_warp; ++index) {
                        if (index < d) {
                            Sij += qmem[t * d + index] * kmem[i * d + index];
                        }
                    }

                    Sij += __shfl_xor_sync(0xffffffff, Sij, 16); // 0-15 和 16-31 交换相加
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 8);  // 0-7 和 8-15 ...
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 4);
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 2);
                    Sij += __shfl_xor_sync(0xffffffff, Sij, 1);
                    
                    ValueType m_prev = m[j];
                    ValueType l_prev = l[j];
                    
                    ValueType m_new = max(m_prev, Sij);

                    ValueType p_val = expf(Sij - m_new);
                    ValueType alpha = expf(m_prev - m_new);

                    l[j] = l_prev * alpha + p_val;

                    m[j] = m_new;

                    for (int index = 0; index < per_thread_row_count_warp; ++index) {
                        int col_idx = lane_id * per_thread_row_count_warp + index;
                        if (col_idx < d) {
                            float v_val = vmem[i * d + col_idx];
                            acc_o[j][index] = acc_o[j][index] * alpha + p_val * v_val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < rows_per_warp; ++i) {
        int t = warp_id * rows_per_warp + i;
        if (t < Br) {
            const int row_offset = t * o_stride_n; // 相对于 o_ptr 的行偏移
            for (int j = 0; j < per_thread_row_count_warp; ++j) {
                const int offset = lane_id * per_thread_row_count_warp + j;
                if (offset < d) {
                    // 最终结果 = 累加值 / 分母 l
                    *(o_ptr + row_offset + offset) = acc_o[i][j] / l[i];
                }
            }
        }
    }
}


