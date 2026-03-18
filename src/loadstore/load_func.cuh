#pragma once

#include "../config/macros.cuh"
#include "../numeric/Int.cuh"
#include "../tensor_core/tensor.cuh"
#include "../layout/layout.h"
#include "../ptx/copy.cuh"
#include "./copy_gsm.cuh"

namespace loadstore {

DEVICE void commit_group() {
    ptx::cp_async_fence();
}

template <int N>
DEVICE void wait_group() {
    ptx::cp_async_wait(numeric::Int<N>{});
}

template <typename KernelConfig, typename TensorQ>
DEVICE void load_q(
    TensorQ& sQ,
    typename KernelConfig::Element* Q,
    typename KernelConfig::index_t q_base,
    int q_block_idx,
    int seq_len,
    int stride_seq_q
) {
    auto* gQTile = Q + q_base + static_cast<typename KernelConfig::index_t>(q_block_idx * KernelConfig::BlockM) * stride_seq_q;
    int valid_q_rows = max(0, min(KernelConfig::BlockM, seq_len - q_block_idx * KernelConfig::BlockM));
    if (valid_q_rows == KernelConfig::BlockM) {
        auto gQTensor = tensor::make_tensor(gQTile, layout::make_layout(layout::make_shape(KernelConfig::BlockM, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_q, 1)));
        copy_g2s_q<KernelConfig>(gQTensor, sQ);
    } else {
        auto gQTensor = tensor::make_tensor(gQTile, layout::make_layout(layout::make_shape(KernelConfig::BlockM, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_q, 1)));
        copy_g2s_q_predicated<KernelConfig>(gQTensor, sQ, valid_q_rows);
    }
}

template <typename KernelConfig, typename TensorQ>
DEVICE void load_q_async(
    TensorQ& sQ,
    typename KernelConfig::Element* Q,
    typename KernelConfig::index_t q_base,
    int q_block_idx,
    int seq_len,
    int stride_seq_q
) {
    auto* gQTile = Q + q_base + static_cast<typename KernelConfig::index_t>(q_block_idx * KernelConfig::BlockM) * stride_seq_q;
    int valid_q_rows = max(0, min(KernelConfig::BlockM, seq_len - q_block_idx * KernelConfig::BlockM));
    if (valid_q_rows == KernelConfig::BlockM) {
        auto gQTensor = tensor::make_tensor(gQTile, layout::make_layout(layout::make_shape(KernelConfig::BlockM, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_q, 1)));
        copy_g2s_q_async<KernelConfig>(gQTensor, sQ);
    } else {
        auto gQTensor = tensor::make_tensor(gQTile, layout::make_layout(layout::make_shape(KernelConfig::BlockM, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_q, 1)));
        copy_g2s_q_predicated<KernelConfig>(gQTensor, sQ, valid_q_rows);
    }
}

template <typename KernelConfig, typename TensorK>
DEVICE void load_k(
    TensorK& sK,
    typename KernelConfig::Element* K,
    typename KernelConfig::index_t k_base,
    int kv_block_idx,
    int seq_len,
    int stride_seq_k
) {
    using index_t = typename KernelConfig::index_t;
    index_t k_offset = static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * stride_seq_k;
    auto* gKTile = K + k_base + k_offset;
    int valid_k_rows = max(0, min(KernelConfig::BlockN, seq_len - kv_block_idx * KernelConfig::BlockN));
    if (valid_k_rows == KernelConfig::BlockN) {
        auto gKTensor = tensor::make_tensor(gKTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_k, 1)));
        copy_g2s_kv<KernelConfig>(gKTensor, sK);
    } else {
        auto gKTensor = tensor::make_tensor(gKTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_k, 1)));
        copy_g2s_kv_predicated<KernelConfig>(gKTensor, sK, valid_k_rows);
    }
}

template <typename KernelConfig, typename TensorK>
DEVICE void load_k_async(
    TensorK& sK,
    typename KernelConfig::Element* K,
    typename KernelConfig::index_t k_base,
    int kv_block_idx,
    int seq_len,
    int stride_seq_k
) {
    using index_t = typename KernelConfig::index_t;
    index_t k_offset = static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * stride_seq_k;
    auto* gKTile = K + k_base + k_offset;
    int valid_k_rows = max(0, min(KernelConfig::BlockN, seq_len - kv_block_idx * KernelConfig::BlockN));
    if (valid_k_rows == KernelConfig::BlockN) {
        auto gKTensor = tensor::make_tensor(gKTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_k, 1)));
        copy_g2s_kv_async<KernelConfig>(gKTensor, sK);
    } else {
        auto gKTensor = tensor::make_tensor(gKTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_k, 1)));
        copy_g2s_kv_predicated<KernelConfig>(gKTensor, sK, valid_k_rows);
    }
}

template <typename KernelConfig, typename TensorV>
DEVICE void load_v(
    TensorV& sV,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t v_base,
    int kv_block_idx,
    int seq_len,
    int stride_seq_v
) {
    using index_t = typename KernelConfig::index_t;
    index_t v_offset = static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * stride_seq_v;
    auto* gVTile = V + v_base + v_offset;
    int valid_v_rows = max(0, min(KernelConfig::BlockN, seq_len - kv_block_idx * KernelConfig::BlockN));
    if (valid_v_rows == KernelConfig::BlockN) {
        auto gVTensor = tensor::make_tensor(gVTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_v, 1)));
        copy_g2s_kv<KernelConfig>(gVTensor, sV);
    } else {
        auto gVTensor = tensor::make_tensor(gVTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_v, 1)));
        copy_g2s_kv_predicated<KernelConfig>(gVTensor, sV, valid_v_rows);
    }
}

template <typename KernelConfig, typename TensorV>
DEVICE void load_v_async(
    TensorV& sV,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t v_base,
    int kv_block_idx,
    int seq_len,
    int stride_seq_v
) {
    using index_t = typename KernelConfig::index_t;
    index_t v_offset = static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * stride_seq_v;
    auto* gVTile = V + v_base + v_offset;
    int valid_v_rows = max(0, min(KernelConfig::BlockN, seq_len - kv_block_idx * KernelConfig::BlockN));
    if (valid_v_rows == KernelConfig::BlockN) {
        auto gVTensor = tensor::make_tensor(gVTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_v, 1)));
        copy_g2s_kv_async<KernelConfig>(gVTensor, sV);
    } else {
        auto gVTensor = tensor::make_tensor(gVTile, layout::make_layout(layout::make_shape(KernelConfig::BlockN, KernelConfig::HeadDim),
                                                                         layout::make_stride(stride_seq_v, 1)));
        copy_g2s_kv_predicated<KernelConfig>(gVTensor, sV, valid_v_rows);
    }
}

} // namespace loadstore
