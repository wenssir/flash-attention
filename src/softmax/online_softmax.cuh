#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "../config/macros.cuh"
#include "../tensor_core/tensor.cuh"
#include "../tensor_core/ldmatrix_traits.cuh"
#include "../loadstore/copy_rsm.cuh"

namespace softmax {

DEVICE float compute_p(float s, float m_new, float softmax_scale) {
    return exp2f(s * softmax_scale - m_new * softmax_scale);
}

DEVICE float renorm_coeff(float m_old, float m_new, float softmax_scale) {
    return exp2f((m_old - m_new) * softmax_scale);
}

DEVICE float reduce_row_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
    return x;
}

DEVICE float reduce_row_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    return x;
}

DEVICE void lane_rows(int lane, int& row0, int& row1) {
    row0 = lane >> 2;
    row1 = row0 + 8;
}

template <typename KernelConfig>
DEVICE void calc_row_max(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    float& m_next0,
    float& m_next1,
    float m_cur0,
    float m_cur1,
    int warp,
    int lane,
    int q_tile,
    int kv_tile,
    int seq_len,
    bool causal
) {
    int row0 = lane >> 2;
    int row1 = row0 + 8;

    int warp_row_base = q_tile * KernelConfig::BlockM + warp * KernelConfig::MmaM;
    int row0_global = warp_row_base + row0;
    int row1_global = warp_row_base + row1;
    bool row0_valid = row0_global < seq_len;
    bool row1_valid = row1_global < seq_len;

    m_next0 = m_cur0;
    m_next1 = m_cur1;

    #pragma unroll
    for (int n_tile = 0; n_tile < KernelConfig::ScoreTiles; ++n_tile) {
        int col0 = kv_tile * KernelConfig::BlockN + n_tile * KernelConfig::MmaN + ((lane & 0x3) << 1);
        int col1 = col0 + 1;

        bool row0_col0_valid = row0_valid && col0 < seq_len && (!causal || col0 <= row0_global);
        bool row0_col1_valid = row0_valid && col1 < seq_len && (!causal || col1 <= row0_global);
        bool row1_col0_valid = row1_valid && col0 < seq_len && (!causal || col0 <= row1_global);
        bool row1_col1_valid = row1_valid && col1 < seq_len && (!causal || col1 <= row1_global);

        scores[n_tile](0) = row0_col0_valid ? scores[n_tile](0) : -INFINITY;
        scores[n_tile](1) = row0_col1_valid ? scores[n_tile](1) : -INFINITY;
        scores[n_tile](2) = row1_col0_valid ? scores[n_tile](2) : -INFINITY;
        scores[n_tile](3) = row1_col1_valid ? scores[n_tile](3) : -INFINITY;

        m_next0 = fmaxf(m_next0, scores[n_tile](0));
        m_next0 = fmaxf(m_next0, scores[n_tile](1));
        m_next1 = fmaxf(m_next1, scores[n_tile](2));
        m_next1 = fmaxf(m_next1, scores[n_tile](3));
    }

    m_next0 = reduce_row_max(m_next0);
    m_next1 = reduce_row_max(m_next1);
}

template <typename KernelConfig>
DEVICE void scale_l_o(
    typename KernelConfig::FragAcc* acc_o,
    float& softmax_m0,
    float& softmax_l0,
    float& softmax_m1,
    float& softmax_l1,
    float m_next0,
    float m_next1,
    float score_scale_log2e,
    int kv_tile
) {
    float alpha0 = kv_tile == 0 ? 0.0f : renorm_coeff(softmax_m0, m_next0, score_scale_log2e);
    float alpha1 = kv_tile == 0 ? 0.0f : renorm_coeff(softmax_m1, m_next1, score_scale_log2e);

    softmax_m0 = m_next0;
    softmax_m1 = m_next1;
    softmax_l0 = kv_tile == 0 ? 0.0f : softmax_l0 * alpha0;
    softmax_l1 = kv_tile == 0 ? 0.0f : softmax_l1 * alpha1;

    constexpr int OutputTiles = KernelConfig::HeadDim / KernelConfig::MmaN;
    #pragma unroll
    for (int d_tile = 0; d_tile < OutputTiles; ++d_tile) {
        acc_o[d_tile](0) *= alpha0;
        acc_o[d_tile](1) *= alpha0;
        acc_o[d_tile](2) *= alpha1;
        acc_o[d_tile](3) *= alpha1;
    }
}

template <typename KernelConfig>
DEVICE void exponentiate_scores(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    float m0,
    float m1,
    float score_scale_log2e
) {
    #pragma unroll
    for (int n_tile = 0; n_tile < KernelConfig::ScoreTiles; ++n_tile) {
        scores[n_tile](0) = compute_p(scores[n_tile](0), m0, score_scale_log2e);
        scores[n_tile](1) = compute_p(scores[n_tile](1), m0, score_scale_log2e);
        scores[n_tile](2) = compute_p(scores[n_tile](2), m1, score_scale_log2e);
        scores[n_tile](3) = compute_p(scores[n_tile](3), m1, score_scale_log2e);
    }
}

template <typename KernelConfig>
DEVICE void update_row_exp_sum(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    float& softmax_l0,
    float& softmax_l1
) {
    float row0_sum = 0.0f;
    float row1_sum = 0.0f;

    #pragma unroll
    for (int n_tile = 0; n_tile < KernelConfig::ScoreTiles; ++n_tile) {
        row0_sum += scores[n_tile](0) + scores[n_tile](1);
        row1_sum += scores[n_tile](2) + scores[n_tile](3);
    }

    softmax_l0 = fmaf(1.0f, row0_sum, softmax_l0);
    softmax_l1 = fmaf(1.0f, row1_sum, softmax_l1);
}

template <typename KernelConfig>
DEVICE void online_softmax_update_scores(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    typename KernelConfig::FragAcc* acc_o,
    float row0_max,
    float row1_max,
    float& softmax_m0,
    float& softmax_l0,
    float& softmax_m1,
    float& softmax_l1,
    float score_scale_log2e,
    int kv_tile
) {
    scale_l_o<KernelConfig>(
        acc_o, softmax_m0, softmax_l0, softmax_m1, softmax_l1,
        row0_max, row1_max, score_scale_log2e, kv_tile);
    exponentiate_scores<KernelConfig>(scores, softmax_m0, softmax_m1, score_scale_log2e);
    update_row_exp_sum<KernelConfig>(scores, softmax_l0, softmax_l1);
}

template <typename KernelConfig>
DEVICE void normalize_output(
    typename KernelConfig::FragAcc* acc_o,
    float softmax_l0,
    float softmax_l1
) {
    softmax_l0 = reduce_row_sum(softmax_l0);
    softmax_l1 = reduce_row_sum(softmax_l1);
    float inv_l0 = 1.0f / softmax_l0;
    float inv_l1 = 1.0f / softmax_l1;

    constexpr int OutputTiles = KernelConfig::HeadDim / KernelConfig::MmaN;
    #pragma unroll
    for (int d_tile = 0; d_tile < OutputTiles; ++d_tile) {
        acc_o[d_tile](0) *= inv_l0;
        acc_o[d_tile](1) *= inv_l0;
        acc_o[d_tile](2) *= inv_l1;
        acc_o[d_tile](3) *= inv_l1;
    }
}

template <typename KernelConfig>
DEVICE int output_smem_offset(int warp, int d_tile, int row, int col) {
    return (((warp * KernelConfig::OutputTiles + d_tile) * KernelConfig::MmaM + row) *
            KernelConfig::MmaN + col);
}

template <typename KernelConfig>
DEVICE void finalize_output(
    typename KernelConfig::FragAcc* acc_o,
    float softmax_l0,
    float softmax_l1,
    float* smem_o,
    typename KernelConfig::Element* O,
    typename KernelConfig::index_t o_base,
    int stride_seq_o,
    int seq_len,
    int q_block_idx,
    int warp,
    int lane
) {
    int row0 = lane >> 2;
    int row1 = row0 + 8;

    int row0_global = q_block_idx * KernelConfig::BlockM + warp * KernelConfig::MmaM + row0;
    int row1_global = q_block_idx * KernelConfig::BlockM + warp * KernelConfig::MmaM + row1;

    float safe_l0 = row0_global < seq_len ? softmax_l0 : 1.0f;
    float safe_l1 = row1_global < seq_len ? softmax_l1 : 1.0f;
    normalize_output<KernelConfig>(acc_o, safe_l0, safe_l1);

    #pragma unroll
    for (int d_tile = 0; d_tile < KernelConfig::OutputTiles; ++d_tile) {
        #pragma unroll
        for (int reg_idx = 0; reg_idx < tensor::RF2SmemLayoutTraits<tensor::MmaShapeM16N8K16>::regs_per_lane; ++reg_idx) {
            int m = 0;
            int n = 0;
            tensor::RF2SmemLayoutTraits<tensor::MmaShapeM16N8K16>::lane_reg_to_mn(lane, reg_idx, m, n);
            smem_o[output_smem_offset<KernelConfig>(warp, d_tile, m, n)] = acc_o[d_tile](reg_idx);
        }
    }
    __syncthreads();

    int tid = threadIdx.x;
    constexpr int BlockElems = KernelConfig::BlockM * KernelConfig::HeadDim;
    for (int elem = tid; elem < BlockElems; elem += blockDim.x) {
        int row = elem / KernelConfig::HeadDim;
        int col_idx = elem - row * KernelConfig::HeadDim;
        int row_global = q_block_idx * KernelConfig::BlockM + row;
        if (row_global < seq_len) {
            int warp_idx = row / KernelConfig::MmaM;
            int row_in_warp = row - warp_idx * KernelConfig::MmaM;
            int d_tile = col_idx / KernelConfig::MmaN;
            int col_in_tile = col_idx - d_tile * KernelConfig::MmaN;
            O[o_base + static_cast<typename KernelConfig::index_t>(row_global) * stride_seq_o + col_idx] =
                __float2half(smem_o[output_smem_offset<KernelConfig>(warp_idx, d_tile, row_in_warp, col_in_tile)]);
        }
    }
}

} // namespace softmax
