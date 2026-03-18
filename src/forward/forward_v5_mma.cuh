#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "../config/config.cuh"
#include "../layout/layout.h"
#include "../layout/coordinate.h"
#include "../tensor_core/tensor.cuh"
#include "../tensor_core/fragment.cuh"
#include "../loadstore/copy_gsm.cuh"
#include "../loadstore/load_func.cuh"
#include "../loadstore/copy_srm.cuh"
#include "../loadstore/copy_rsm.cuh"
#include "../softmax/online_softmax.cuh"
#include "../ptx/mma_atom.cuh"

namespace forward {

DEVICE uint32_t pack_half2(float x, float y) {
    union {
        __half2 h2;
        uint32_t u32;
    } packed{};
    packed.h2 = __float22half2_rn(make_float2(x, y));
    return packed.u32;
}

template <typename KernelConfig>
DEVICE void convert_scores_pair_to_p_frag(
    typename KernelConfig::FragP& p_frag,
    typename KernelConfig::FragAcc const& score_left,
    typename KernelConfig::FragAcc const& score_right
) {
    p_frag(0) = pack_half2(score_left(0), score_left(1));
    p_frag(1) = pack_half2(score_left(2), score_left(3));
    p_frag(2) = pack_half2(score_right(0), score_right(1));
    p_frag(3) = pack_half2(score_right(2), score_right(3));
}

template <typename KernelConfig, typename TensorQ, typename TensorK>
DEVICE void gemm_qk(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    TensorQ& sQ,
    TensorK& sK,
    int warp
) {
    typename KernelConfig::FragQ q_frag{};
    typename KernelConfig::FragK k_frag{};
    #pragma unroll
    for (int out_tile = 0; out_tile < KernelConfig::ScoreTiles; ++out_tile) {
        scores[out_tile].clear();
        #pragma unroll
        for (int k_tile = 0; k_tile < KernelConfig::QkKSteps; ++k_tile) {
            auto q_frag_view = loadstore::partition_fragment_A<KernelConfig>(sQ, warp, k_tile);
            auto k_frag_view = loadstore::partition_fragment_B<KernelConfig>(sK, out_tile, k_tile);
            loadstore::load_fragment<KernelConfig>(q_frag, q_frag_view);
            loadstore::load_fragment<KernelConfig>(k_frag, k_frag_view);
            KernelConfig::Atom::fma(q_frag, k_frag, scores[out_tile]);
        }
    }
}

template <typename KernelConfig, typename TensorV>
DEVICE void gemm_pv(
    typename KernelConfig::FragAcc* acc_o,
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    TensorV& sV
) {
    typename KernelConfig::FragP p_frag{};
    typename KernelConfig::FragV v_frag{};
    #pragma unroll
    for (int out_tile = 0; out_tile < KernelConfig::OutputTiles; ++out_tile) {
        #pragma unroll
        for (int k_tile = 0; k_tile < KernelConfig::PvKTiles; ++k_tile) {
            auto v_frag_view = loadstore::partition_fragment_V<KernelConfig>(sV, k_tile, out_tile);
            convert_scores_pair_to_p_frag<KernelConfig>(
                p_frag,
                scores[2 * k_tile + 0],
                scores[2 * k_tile + 1]);
            loadstore::load_fragment<KernelConfig>(v_frag, v_frag_view);
            KernelConfig::Atom::fma(p_frag, v_frag, acc_o[out_tile]);
        }
    }
}

template <bool IsFirst, typename KernelConfig, typename TensorQ, typename TensorK, typename TensorV>
DEVICE void process_kv_block(
    TensorQ& sQ,
    TensorK& sK,
    TensorV& sV,
    typename KernelConfig::Element* K,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t k_base,
    typename KernelConfig::index_t v_base,
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    typename KernelConfig::FragAcc* acc_o,
    float& softmax_m0,
    float& softmax_l0,
    float& softmax_m1,
    float& softmax_l1,
    float softmax_scale_log2e,
    int warp,
    int lane,
    int q_block_idx,
    int kv_block_idx,
    int n_kv_tiles,
    int seq_len,
    bool causal,
    int stride_seq_k,
    int stride_seq_v
) {
    if constexpr (IsFirst) {
        loadstore::wait_group<0>();
        __syncthreads();
    }

    loadstore::load_v_async<KernelConfig>(sV, V, v_base, kv_block_idx, seq_len, stride_seq_v);
    loadstore::commit_group();

    gemm_qk<KernelConfig>(scores, sQ, sK, warp);

    loadstore::wait_group<0>();
    __syncthreads();

    float row0_max = -INFINITY;
    float row1_max = -INFINITY;
    softmax::calc_row_max<KernelConfig>(
        scores, row0_max, row1_max, softmax_m0, softmax_m1,
        warp, lane, q_block_idx, kv_block_idx, seq_len, causal);
    softmax::online_softmax_update_scores<KernelConfig>(
        scores, acc_o, row0_max, row1_max, softmax_m0, softmax_l0, softmax_m1, softmax_l1,
        softmax_scale_log2e, kv_block_idx);

    if (kv_block_idx + 1 < n_kv_tiles) {
        loadstore::load_k_async<KernelConfig>(sK, K, k_base, kv_block_idx + 1, seq_len, stride_seq_k);
        loadstore::commit_group();
    }
    gemm_pv<KernelConfig>(acc_o, scores, sV);
}

template <typename KernelConfig>
__global__ void flash_attention_forward_v5_mma(const config::ForwardKernelArgs args) {
    using Element = typename KernelConfig::Element;
    using index_t = typename KernelConfig::index_t;

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int warp = threadIdx.x >> 5;

    if (args.head_dim != KernelConfig::HeadDim) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Element* smem_ptr = reinterpret_cast<Element*>(smem_raw);

    auto sQ = tensor::make_tensor(smem_ptr, typename KernelConfig::SmemLayoutQ{});
    auto sK = tensor::make_tensor(sQ.data_ptr() + sQ.size(), typename KernelConfig::SmemLayoutKV{});
    auto sV = tensor::make_tensor(sK.data_ptr() + sK.size(), typename KernelConfig::SmemLayoutKV{});

    auto* Q = static_cast<typename KernelConfig::Element*>(args.Q);
    auto* K = static_cast<typename KernelConfig::Element*>(args.K);
    auto* V = static_cast<typename KernelConfig::Element*>(args.V);
    auto* O = static_cast<typename KernelConfig::Element*>(args.O);

    index_t q_base = static_cast<index_t>(batch_idx) * args.stride_batch_q + static_cast<index_t>(head_idx) * args.stride_head_q;
    index_t k_base = static_cast<index_t>(batch_idx) * args.stride_batch_k + static_cast<index_t>(head_idx) * args.stride_head_k;
    index_t v_base = static_cast<index_t>(batch_idx) * args.stride_batch_v + static_cast<index_t>(head_idx) * args.stride_head_v;
    index_t o_base = static_cast<index_t>(batch_idx) * args.stride_batch_o + static_cast<index_t>(head_idx) * args.stride_head_o;

    loadstore::load_q_async<KernelConfig>(sQ, Q, q_base, q_block_idx, args.seq_len, args.stride_seq_q);

    typename KernelConfig::FragAcc acc_o[KernelConfig::OutputTiles];
    #pragma unroll
    for (int d_tile = 0; d_tile < KernelConfig::OutputTiles; ++d_tile) {
        acc_o[d_tile].clear();
    }

    int lane = threadIdx.x & 31;
    int row0 = 0;
    int row1 = 0;
    softmax::lane_rows(lane, row0, row1);
    float softmax_m0 = -INFINITY;
    float softmax_l0 = 0.0f;
    float softmax_m1 = -INFINITY;
    float softmax_l1 = 0.0f;

    const float softmax_scale_log2e = args.softmax_scale * KernelConfig::Log2e;
    int n_kv_tiles = (args.seq_len + KernelConfig::BlockN - 1) / KernelConfig::BlockN;

    typename KernelConfig::FragAcc scores[KernelConfig::ScoreTiles];

    if (n_kv_tiles > 0) {
        loadstore::load_k_async<KernelConfig>(sK, K, k_base, 0, args.seq_len, args.stride_seq_k);
        loadstore::commit_group();
        process_kv_block<true, KernelConfig>(
            sQ, sK, sV, K, V, k_base, v_base, scores, acc_o,
            softmax_m0, softmax_l0, softmax_m1, softmax_l1, softmax_scale_log2e, warp, lane,
            q_block_idx, 0, n_kv_tiles, args.seq_len, args.causal != 0,
            args.stride_seq_k, args.stride_seq_v);
    }

    for (int kv_block_idx = 1; kv_block_idx < n_kv_tiles; ++kv_block_idx) {
        process_kv_block<false, KernelConfig>(
            sQ, sK, sV, K, V, k_base, v_base, scores, acc_o,
            softmax_m0, softmax_l0, softmax_m1, softmax_l1, softmax_scale_log2e, warp, lane,
            q_block_idx, kv_block_idx, n_kv_tiles, args.seq_len, args.causal != 0,
            args.stride_seq_k, args.stride_seq_v);
    }

    float* smem_o = reinterpret_cast<float*>(sK.data_ptr());
    softmax::finalize_output<KernelConfig>(
        acc_o, softmax_l0, softmax_l1, smem_o, O, o_base, args.stride_seq_o, args.seq_len, q_block_idx, warp, lane);
}

} // namespace forward
