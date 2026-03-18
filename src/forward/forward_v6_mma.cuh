#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "../config/config.cuh"
#include "../tensor_core/tensor.cuh"
#include "../loadstore/copy_gsm.cuh"
#include "../loadstore/copy_gsm_direct.cuh"
#include "../loadstore/load_func.cuh"
#include "../loadstore/copy_srm.cuh"
#include "../loadstore/copy_srm_direct.cuh"
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

template <typename KernelConfig>
DEVICE void preload_q_rf(
    typename KernelConfig::FragQ (&q_frags)[KernelConfig::QkKSteps],
    auto& sQ,
    int warp
) {
    #pragma unroll
    for (int k_tile = 0; k_tile < KernelConfig::QkKSteps; ++k_tile) {
        loadstore::load_fragment_q_direct<KernelConfig>(q_frags[k_tile], sQ.data_ptr(), warp, k_tile);
    }
}

template <typename KernelConfig>
DEVICE void gemm_qk(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    typename KernelConfig::FragQ const (&q_frags)[KernelConfig::QkKSteps],
    auto& sK
) {
    constexpr int KStageTiles = 2;
    typename KernelConfig::FragK k_stage[KStageTiles];
    #pragma unroll
    for (int out_tile = 0; out_tile < KernelConfig::ScoreTiles; ++out_tile) {
        scores[out_tile].clear();
        #pragma unroll
        for (int k_tile_base = 0; k_tile_base < KernelConfig::QkKSteps; k_tile_base += KStageTiles) {
            #pragma unroll
            for (int stage = 0; stage < KStageTiles; ++stage) {
                int k_tile = k_tile_base + stage;
                if (k_tile < KernelConfig::QkKSteps) {
                    loadstore::load_fragment_k_direct<KernelConfig>(
                        k_stage[stage], sK.data_ptr(), out_tile, k_tile);
                } else {
                    k_stage[stage].clear();
                }
            }
            #pragma unroll
            for (int stage = 0; stage < KStageTiles; ++stage) {
                int k_tile = k_tile_base + stage;
                if (k_tile < KernelConfig::QkKSteps) {
                    KernelConfig::Atom::fma(q_frags[k_tile], k_stage[stage], scores[out_tile]);
                }
            }
        }
    }
}

template <typename KernelConfig>
DEVICE void materialize_p_rf(
    typename KernelConfig::FragP (&p_frags)[KernelConfig::PvKTiles],
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles]
) {
    #pragma unroll
    for (int k_tile = 0; k_tile < KernelConfig::PvKTiles; ++k_tile) {
        convert_scores_pair_to_p_frag<KernelConfig>(
            p_frags[k_tile],
            scores[2 * k_tile + 0],
            scores[2 * k_tile + 1]);
    }
}

template <typename KernelConfig>
DEVICE void gemm_pv(
    typename KernelConfig::FragAcc* acc_o,
    typename KernelConfig::FragP const (&p_frags)[KernelConfig::PvKTiles],
    auto& sV
) {
    constexpr int VStageTiles = 2;
    typename KernelConfig::FragV v_stage[VStageTiles];
    #pragma unroll
    for (int out_tile = 0; out_tile < KernelConfig::OutputTiles; ++out_tile) {
        #pragma unroll
        for (int k_tile_base = 0; k_tile_base < KernelConfig::PvKTiles; k_tile_base += VStageTiles) {
            #pragma unroll
            for (int stage = 0; stage < VStageTiles; ++stage) {
                int k_tile = k_tile_base + stage;
                if (k_tile < KernelConfig::PvKTiles) {
                    loadstore::load_fragment_v_direct<KernelConfig>(
                        v_stage[stage], sV.data_ptr(), k_tile, out_tile);
                } else {
                    v_stage[stage].clear();
                }
            }
            #pragma unroll
            for (int stage = 0; stage < VStageTiles; ++stage) {
                int k_tile = k_tile_base + stage;
                if (k_tile < KernelConfig::PvKTiles) {
                    KernelConfig::Atom::fma(p_frags[k_tile], v_stage[stage], acc_o[out_tile]);
                }
            }
        }
    }
}

template <typename KernelConfig, typename TensorQ>
DEVICE void load_q_async(
    TensorQ& sQ,
    typename KernelConfig::Element* Q,
    typename KernelConfig::index_t q_base,
    int q_block_idx
) {
    using index_t = typename KernelConfig::index_t;
    auto* gQTile = Q + q_base + static_cast<index_t>(q_block_idx * KernelConfig::BlockM) * KernelConfig::HeadDim;
    loadstore::copy_g2s_q_async_direct<KernelConfig>(gQTile, KernelConfig::HeadDim, sQ);
}

template <typename KernelConfig, typename TensorKV>
DEVICE void load_k_async(
    TensorKV& sK,
    typename KernelConfig::Element* K,
    typename KernelConfig::index_t k_base,
    int kv_block_idx
) {
    using index_t = typename KernelConfig::index_t;
    auto* gKTile = K + k_base + static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * KernelConfig::HeadDim;
    loadstore::copy_g2s_kv_async_direct<KernelConfig>(gKTile, KernelConfig::HeadDim, sK);
}

template <typename KernelConfig, typename TensorKV>
DEVICE void load_v_async(
    TensorKV& sV,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t v_base,
    int kv_block_idx
) {
    using index_t = typename KernelConfig::index_t;
    auto* gVTile = V + v_base + static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * KernelConfig::HeadDim;
    loadstore::copy_g2s_kv_async_direct<KernelConfig>(gVTile, KernelConfig::HeadDim, sV);
}

template <typename KernelConfig>
DEVICE void calc_row_max(
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    float& m_next0,
    float& m_next1,
    float m_cur0,
    float m_cur1
) {
    m_next0 = m_cur0;
    m_next1 = m_cur1;

    #pragma unroll
    for (int n_tile = 0; n_tile < KernelConfig::ScoreTiles; ++n_tile) {
        m_next0 = fmaxf(m_next0, scores[n_tile](0));
        m_next0 = fmaxf(m_next0, scores[n_tile](1));
        m_next1 = fmaxf(m_next1, scores[n_tile](2));
        m_next1 = fmaxf(m_next1, scores[n_tile](3));
    }

    m_next0 = softmax::reduce_row_max(m_next0);
    m_next1 = softmax::reduce_row_max(m_next1);
}

template <typename KernelConfig>
DEVICE void finalize_output(
    typename KernelConfig::FragAcc* acc_o,
    float softmax_l0,
    float softmax_l1,
    float* smem_o,
    typename KernelConfig::Element* O,
    typename KernelConfig::index_t o_base,
    int q_block_idx,
    int warp
) {
    int lane = threadIdx.x & 31;
    softmax_l0 = softmax::reduce_row_sum(softmax_l0);
    softmax_l1 = softmax::reduce_row_sum(softmax_l1);
    float inv_l0 = 1.0f / softmax_l0;
    float inv_l1 = 1.0f / softmax_l1;

    constexpr int OutputTiles = KernelConfig::HeadDim / KernelConfig::MmaN;
    #pragma unroll
    for (int d_tile = 0; d_tile < OutputTiles; ++d_tile) {
        acc_o[d_tile](0) *= inv_l0;
        acc_o[d_tile](1) *= inv_l0;
        acc_o[d_tile](2) *= inv_l1;
        acc_o[d_tile](3) *= inv_l1;
        #pragma unroll
        for (int reg_idx = 0; reg_idx < tensor::RF2SmemLayoutTraits<tensor::MmaShapeM16N8K16>::regs_per_lane; ++reg_idx) {
            int m = 0;
            int n = 0;
            tensor::RF2SmemLayoutTraits<tensor::MmaShapeM16N8K16>::lane_reg_to_mn(lane, reg_idx, m, n);
            smem_o[softmax::output_smem_offset<KernelConfig>(warp, d_tile, m, n)] = acc_o[d_tile](reg_idx);
        }
    }
    __syncwarp();

    constexpr int WarpElems = KernelConfig::MmaM * KernelConfig::HeadDim;
    for (int elem = lane; elem < WarpElems; elem += 32) {
        int row_in_warp = elem / KernelConfig::HeadDim;
        int col = elem - row_in_warp * KernelConfig::HeadDim;
        int row_global = q_block_idx * KernelConfig::BlockM + warp * KernelConfig::MmaM + row_in_warp;
        int d_tile = col / KernelConfig::MmaN;
        int col_in_tile = col - d_tile * KernelConfig::MmaN;
        O[o_base + static_cast<typename KernelConfig::index_t>(row_global) * KernelConfig::HeadDim + col] =
            __float2half(smem_o[softmax::output_smem_offset<KernelConfig>(warp, d_tile, row_in_warp, col_in_tile)]);
    }
}

template <bool IsFirst, typename KernelConfig>
DEVICE void process_kv_block(
    auto& sK,
    auto& sV,
    typename KernelConfig::Element* K,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t k_base,
    typename KernelConfig::index_t v_base,
    typename KernelConfig::FragQ const (&q_frags)[KernelConfig::QkKSteps],
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    typename KernelConfig::FragAcc* acc_o,
    float& softmax_m0,
    float& softmax_l0,
    float& softmax_m1,
    float& softmax_l1,
    float softmax_scale_log2e,
    int kv_block_idx
) {
    loadstore::wait_group<0>();
    __syncthreads();

    load_v_async<KernelConfig>(sV, V, v_base, kv_block_idx);
    loadstore::commit_group();

    gemm_qk<KernelConfig>(scores, q_frags, sK);

    float row0_max = -INFINITY;
    float row1_max = -INFINITY;
    calc_row_max<KernelConfig>(scores, row0_max, row1_max, softmax_m0, softmax_m1);
    softmax::online_softmax_update_scores<KernelConfig>(
        scores, acc_o, row0_max, row1_max, softmax_m0, softmax_l0, softmax_m1, softmax_l1,
        softmax_scale_log2e, kv_block_idx);

    typename KernelConfig::FragP p_frags[KernelConfig::PvKTiles];
    materialize_p_rf<KernelConfig>(p_frags, scores);

    loadstore::wait_group<0>();
    __syncthreads();

    load_k_async<KernelConfig>(sK, K, k_base, kv_block_idx + 1);
    loadstore::commit_group();

    gemm_pv<KernelConfig>(acc_o, p_frags, sV);
}

template <bool IsFirst, typename KernelConfig>
DEVICE void process_kv_block_last(
    auto& sK,
    auto& sV,
    typename KernelConfig::Element* K,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t k_base,
    typename KernelConfig::index_t v_base,
    typename KernelConfig::FragQ const (&q_frags)[KernelConfig::QkKSteps],
    typename KernelConfig::FragAcc (&scores)[KernelConfig::ScoreTiles],
    typename KernelConfig::FragAcc* acc_o,
    float& softmax_m0,
    float& softmax_l0,
    float& softmax_m1,
    float& softmax_l1,
    float softmax_scale_log2e,
    int kv_block_idx
) {
    loadstore::wait_group<0>();
    __syncthreads();

    load_v_async<KernelConfig>(sV, V, v_base, kv_block_idx);
    loadstore::commit_group();

    gemm_qk<KernelConfig>(scores, q_frags, sK);

    float row0_max = -INFINITY;
    float row1_max = -INFINITY;
    calc_row_max<KernelConfig>(scores, row0_max, row1_max, softmax_m0, softmax_m1);
    softmax::online_softmax_update_scores<KernelConfig>(
        scores, acc_o, row0_max, row1_max, softmax_m0, softmax_l0, softmax_m1, softmax_l1,
        softmax_scale_log2e, kv_block_idx);

    typename KernelConfig::FragP p_frags[KernelConfig::PvKTiles];
    materialize_p_rf<KernelConfig>(p_frags, scores);

    loadstore::wait_group<0>();
    __syncthreads();

    gemm_pv<KernelConfig>(acc_o, p_frags, sV);
}

template <typename KernelConfig>
__global__ void flash_attention_forward_v6_mma(const config::ForwardKernelArgs args) {
    using Element = typename KernelConfig::Element;
    using index_t = typename KernelConfig::index_t;

    if (args.head_dim != KernelConfig::HeadDim || args.causal != 0) {
        return;
    }
    if ((args.seq_len % KernelConfig::BlockM) != 0 || (args.seq_len % KernelConfig::BlockN) != 0) {
        return;
    }
    if (args.stride_seq_q != KernelConfig::HeadDim || args.stride_seq_k != KernelConfig::HeadDim ||
        args.stride_seq_v != KernelConfig::HeadDim || args.stride_seq_o != KernelConfig::HeadDim) {
        return;
    }

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int warp = threadIdx.x >> 5;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Element* smem_ptr = reinterpret_cast<Element*>(smem_raw);

    auto sQ = tensor::make_tensor(smem_ptr, typename KernelConfig::SmemLayoutQ{});
    auto sK = tensor::make_tensor(sQ.data_ptr() + sQ.size(), typename KernelConfig::SmemLayoutKV{});
    auto sV = tensor::make_tensor(sK.data_ptr() + sK.size(), typename KernelConfig::SmemLayoutKV{});

    auto* Q = static_cast<Element*>(args.Q);
    auto* K = static_cast<Element*>(args.K);
    auto* V = static_cast<Element*>(args.V);
    auto* O = static_cast<Element*>(args.O);

    index_t head_offset = static_cast<index_t>(args.seq_len) * KernelConfig::HeadDim;
    index_t batch_offset = static_cast<index_t>(args.heads) * head_offset;
    index_t q_base = static_cast<index_t>(batch_idx) * batch_offset + static_cast<index_t>(head_idx) * head_offset;
    index_t k_base = q_base;
    index_t v_base = q_base;
    index_t o_base = q_base;

    typename KernelConfig::FragAcc acc_o[KernelConfig::OutputTiles];
    #pragma unroll
    for (int d_tile = 0; d_tile < KernelConfig::OutputTiles; ++d_tile) {
        acc_o[d_tile].clear();
    }

    float softmax_m0 = -INFINITY;
    float softmax_l0 = 0.0f;
    float softmax_m1 = -INFINITY;
    float softmax_l1 = 0.0f;

    const float softmax_scale_log2e = args.softmax_scale * KernelConfig::Log2e;
    const int n_kv_tiles = args.seq_len / KernelConfig::BlockN;

    typename KernelConfig::FragQ q_frags[KernelConfig::QkKSteps];
    typename KernelConfig::FragAcc scores[KernelConfig::ScoreTiles];

    load_q_async<KernelConfig>(sQ, Q, q_base, q_block_idx);
    loadstore::commit_group();
    loadstore::wait_group<0>();
    __syncwarp();
    preload_q_rf<KernelConfig>(q_frags, sQ, warp);

    if (n_kv_tiles > 0) {
        load_k_async<KernelConfig>(sK, K, k_base, 0);
        loadstore::commit_group();
        process_kv_block<true, KernelConfig>(
            sK, sV, K, V, k_base, v_base, q_frags, scores, acc_o,
            softmax_m0, softmax_l0, softmax_m1, softmax_l1,
            softmax_scale_log2e, 0);
    }

    for (int kv_block_idx = 1; kv_block_idx < n_kv_tiles - 1; ++kv_block_idx) {
        process_kv_block<false, KernelConfig>(
            sK, sV, K, V, k_base, v_base, q_frags, scores, acc_o,
            softmax_m0, softmax_l0, softmax_m1, softmax_l1,
            softmax_scale_log2e, kv_block_idx);
    }

    if (n_kv_tiles > 1) {
        process_kv_block_last<false, KernelConfig>(
            sK, sV, K, V, k_base, v_base, q_frags, scores, acc_o,
            softmax_m0, softmax_l0, softmax_m1, softmax_l1,
            softmax_scale_log2e, n_kv_tiles - 1);
    }

    float* smem_o = reinterpret_cast<float*>(sK.data_ptr());
    finalize_output<KernelConfig>(acc_o, softmax_l0, softmax_l1, smem_o, O, o_base, q_block_idx, warp);
}

} // namespace forward
