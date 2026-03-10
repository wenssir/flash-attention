#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "../config/macros.cuh"
#include "../tensor_core/tensor.cuh"
#include "../tensor_core/accum_layout_traits.cuh"
#include "../loadstore/copy_rsm.cuh"

namespace softmax {

DEVICE float compute_p(float s, float m_new, float softmax_scale) {
    return exp2f(s * softmax_scale - m_new * softmax_scale);
}

DEVICE float renorm_coeff(float m_old, float m_new, float softmax_scale) {
    return exp2f((m_old - m_new) * softmax_scale);
}

DEVICE void lane_rows(int lane, int& row0, int& row1) {
    int col = 0;
    tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>::lane_reg_to_mn(lane, 0, row0, col);
    tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>::lane_reg_to_mn(lane, 2, row1, col);
}

DEVICE float reduce_row_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
    return x;
}

DEVICE float reduce_row_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    return x;
}

DEVICE bool row_owner_lane(int lane) {
    return (lane & 0x3) == 0;
}

template <typename KernelConfig>
DEVICE void mask_and_reduce_scores(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    float& row0_max,
    float& row1_max,
    int warp,
    int lane,
    int q_tile,
    int kv_tile,
    int seq_len,
    bool causal
) {
    int row0 = 0;
    int row1 = 0;
    lane_rows(lane, row0, row1);

    int warp_row_base = q_tile * KernelConfig::BlockM + warp * KernelConfig::MmaM;
    int row0_global = warp_row_base + row0;
    int row1_global = warp_row_base + row1;
    bool row0_valid = row0_global < seq_len;
    bool row1_valid = row1_global < seq_len;

    row0_max = -INFINITY;
    row1_max = -INFINITY;

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

        row0_max = fmaxf(row0_max, scores[n_tile](0));
        row0_max = fmaxf(row0_max, scores[n_tile](1));
        row1_max = fmaxf(row1_max, scores[n_tile](2));
        row1_max = fmaxf(row1_max, scores[n_tile](3));
    }

    row0_max = reduce_row_max(row0_max);
    row1_max = reduce_row_max(row1_max);
}

template <typename KernelConfig>
DEVICE void online_softmax_update_scores(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    typename KernelConfig::FragAcc* acc_o,
    float row0_max,
    float row1_max,
    float* softmax_m,
    float* softmax_l,
    float score_scale_log2e,
    int warp,
    int lane,
    int kv_tile
) {
    int row0 = 0;
    int row1 = 0;
    lane_rows(lane, row0, row1);

    float old_m0 = softmax_m[warp * KernelConfig::MmaM + row0];
    float old_l0 = softmax_l[warp * KernelConfig::MmaM + row0];
    float old_m1 = softmax_m[warp * KernelConfig::MmaM + row1];
    float old_l1 = softmax_l[warp * KernelConfig::MmaM + row1];

    float new_m0 = fmaxf(old_m0, row0_max);
    float new_m1 = fmaxf(old_m1, row1_max);
    float alpha0 = kv_tile == 0 ? 0.0f : renorm_coeff(old_m0, new_m0, score_scale_log2e);
    float alpha1 = kv_tile == 0 ? 0.0f : renorm_coeff(old_m1, new_m1, score_scale_log2e);

    float row0_sum = 0.0f;
    float row1_sum = 0.0f;

    #pragma unroll
    for (int n_tile = 0; n_tile < KernelConfig::ScoreTiles; ++n_tile) {
        float p00 = compute_p(scores[n_tile](0), new_m0, score_scale_log2e);
        float p01 = compute_p(scores[n_tile](1), new_m0, score_scale_log2e);
        float p10 = compute_p(scores[n_tile](2), new_m1, score_scale_log2e);
        float p11 = compute_p(scores[n_tile](3), new_m1, score_scale_log2e);

        scores[n_tile](0) = p00;
        scores[n_tile](1) = p01;
        scores[n_tile](2) = p10;
        scores[n_tile](3) = p11;

        row0_sum += p00 + p01;
        row1_sum += p10 + p11;
    }

    row0_sum = reduce_row_sum(row0_sum);
    row1_sum = reduce_row_sum(row1_sum);

    if (row_owner_lane(lane)) {
        softmax_m[warp * KernelConfig::MmaM + row0] = new_m0;
        softmax_l[warp * KernelConfig::MmaM + row0] = kv_tile == 0 ? row0_sum : (old_l0 * alpha0 + row0_sum);
        softmax_m[warp * KernelConfig::MmaM + row1] = new_m1;
        softmax_l[warp * KernelConfig::MmaM + row1] = kv_tile == 0 ? row1_sum : (old_l1 * alpha1 + row1_sum);
    }

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
DEVICE void normalize_output(
    typename KernelConfig::FragAcc* acc_o,
    float* softmax_l,
    int seq_len,
    int q_block_idx,
    int warp,
    int lane
) {
    int row0 = 0;
    int row1 = 0;
    int col = 0;
    tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>::lane_reg_to_mn(lane, 0, row0, col);
    tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>::lane_reg_to_mn(lane, 2, row1, col);

    int row0_global = q_block_idx * KernelConfig::BlockM + warp * KernelConfig::MmaM + row0;
    int row1_global = q_block_idx * KernelConfig::BlockM + warp * KernelConfig::MmaM + row1;

    float inv_l0 = row0_global < seq_len ? 1.0f / softmax_l[warp * KernelConfig::MmaM + row0] : 0.0f;
    float inv_l1 = row1_global < seq_len ? 1.0f / softmax_l[warp * KernelConfig::MmaM + row1] : 0.0f;

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
DEVICE void store_output_r2s(
    typename KernelConfig::FragAcc* acc_o,
    float* smem_o,
    int warp,
    int d_tile
) {
    auto o_tile = tensor::make_tensor(
        smem_o + warp * KernelConfig::MmaM * KernelConfig::MmaN,
        KernelConfig::o_tile_layout);
    loadstore::copy_r2s(o_tile, acc_o[d_tile]);
}

template <typename KernelConfig>
DEVICE void store_output_s2g(
    float* smem_o,
    typename KernelConfig::Element* O,
    typename KernelConfig::index_t o_base,
    int stride_seq_o,
    int seq_len,
    int q_block_idx,
    int warp,
    int lane,
    int d_tile
) {
    auto o_tile = tensor::make_tensor(
        smem_o + warp * KernelConfig::MmaM * KernelConfig::MmaN,
        KernelConfig::o_tile_layout);
    if (lane < KernelConfig::MmaM) {
        int row = lane;
        int row_global = q_block_idx * KernelConfig::BlockM + warp * KernelConfig::MmaM + row;
        if (row_global < seq_len) {
            #pragma unroll
            for (int col = 0; col < KernelConfig::MmaN; ++col) {
                int col_global = d_tile * KernelConfig::MmaN + col;
                float out = o_tile(row, col);
                O[o_base + static_cast<typename KernelConfig::index_t>(row_global) * stride_seq_o + col_global] =
                    __float2half(out);
            }
        }
    }
}

template <typename KernelConfig>
DEVICE void finalize_output(
    typename KernelConfig::FragAcc* acc_o,
    float* softmax_l,
    float* smem_o,
    typename KernelConfig::Element* O,
    typename KernelConfig::index_t o_base,
    int stride_seq_o,
    int seq_len,
    int q_block_idx,
    int warp,
    int lane
) {
    normalize_output<KernelConfig>(acc_o, softmax_l, seq_len, q_block_idx, warp, lane);
    #pragma unroll
    for (int d_tile = 0; d_tile < KernelConfig::OutputTiles; ++d_tile) {
        store_output_r2s<KernelConfig>(acc_o, smem_o, warp, d_tile);
        __syncwarp();
        store_output_s2g<KernelConfig>(smem_o, O, o_base, stride_seq_o, seq_len, q_block_idx, warp, lane, d_tile);
        __syncwarp();
    }
}

} // namespace softmax
