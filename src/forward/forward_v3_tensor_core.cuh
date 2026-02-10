#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#include "../common/common.h"
#include "../config/config.cuh"
#include "../layout/layout.h"
#include "../loadstore/copy_atom.cuh"
#include "../softmax/online_softmax.cuh"
#include "../tensor_core/tensor.cuh"

namespace forward {

template <typename T>
DEVICE float to_float(T v) {
    return static_cast<float>(v);
}

template <>
DEVICE float to_float<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
DEVICE T from_float(float v) {
    return static_cast<T>(v);
}

template <>
DEVICE __half from_float<__half>(float v) {
    return __float2half(v);
}

template <typename KernelConfig>
HOST_DEVICE constexpr int forward_v3_smem_bytes() {
    return (KernelConfig::BlockM * KernelConfig::HeadDim + 2 * KernelConfig::BlockN * KernelConfig::HeadDim) *
           static_cast<int>(sizeof(typename KernelConfig::Element));
}

template <typename KernelConfig>
__global__ void flash_attention_forward_v3_tensor_core(const config::ForwardKernelArgs args) {
    using Element = typename KernelConfig::Element;
    using index_t = typename KernelConfig::index_t;

    static_assert(KernelConfig::HeadDim == constants::V3_HEAD_DIM, "v3 baseline expects head_dim=128");
    static_assert(KernelConfig::BlockM == constants::V3_BLOCK_M, "v3 baseline expects BlockM=64");
    static_assert(KernelConfig::BlockN == constants::V3_BLOCK_N, "v3 baseline expects BlockN=64");
    static_assert(KernelConfig::NWarps == constants::V3_WARPS, "v3 baseline expects 4 warps");

    int sample = blockIdx.z;
    int head = blockIdx.y;
    int q_tile = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    Element* smem_ptr = reinterpret_cast<Element*>(smem_raw);

    auto smem_q_layout = layout::make_layout(
        layout::make_shape(numeric::Int<KernelConfig::BlockM>{}, numeric::Int<KernelConfig::HeadDim>{}),
        layout::make_stride(numeric::Int<KernelConfig::HeadDim>{}, numeric::Int<1>{}));
    auto smem_kv_layout = layout::make_layout(
        layout::make_shape(numeric::Int<KernelConfig::BlockN>{}, numeric::Int<KernelConfig::HeadDim>{}),
        layout::make_stride(numeric::Int<KernelConfig::HeadDim>{}, numeric::Int<1>{}));

    auto sQ = tensor::make_tensor(smem_ptr, smem_q_layout);
    auto sK = tensor::make_tensor(sQ.data_ptr() + sQ.size(), smem_kv_layout);
    auto sV = tensor::make_tensor(sK.data_ptr() + sK.size(), smem_kv_layout);

    Element* Q = static_cast<Element*>(args.Q);
    Element* K = static_cast<Element*>(args.K);
    Element* V = static_cast<Element*>(args.V);
    Element* O = static_cast<Element*>(args.O);

    index_t q_base = static_cast<index_t>(sample) * args.stride_batch_q + static_cast<index_t>(head) * args.stride_head_q;
    index_t k_base = static_cast<index_t>(sample) * args.stride_batch_k + static_cast<index_t>(head) * args.stride_head_k;
    index_t v_base = static_cast<index_t>(sample) * args.stride_batch_v + static_cast<index_t>(head) * args.stride_head_v;
    index_t o_base = static_cast<index_t>(sample) * args.stride_batch_o + static_cast<index_t>(head) * args.stride_head_o;

    auto gQ = tensor::make_tensor(Q + q_base,
        layout::make_layout(layout::make_shape(args.seq_len, args.head_dim), layout::make_stride(args.stride_seq_q, 1)));
    auto gK = tensor::make_tensor(K + k_base,
        layout::make_layout(layout::make_shape(args.seq_len, args.head_dim), layout::make_stride(args.stride_seq_k, 1)));
    auto gV = tensor::make_tensor(V + v_base,
        layout::make_layout(layout::make_shape(args.seq_len, args.head_dim), layout::make_stride(args.stride_seq_v, 1)));

    auto q_tile_layout = layout::make_layout(
        layout::make_shape(numeric::Int<KernelConfig::BlockM>{}, numeric::Int<KernelConfig::HeadDim>{}),
        gQ.layout().stride());
    auto kv_tile_layout = layout::make_layout(
        layout::make_shape(numeric::Int<KernelConfig::BlockN>{}, numeric::Int<KernelConfig::HeadDim>{}),
        gK.layout().stride());

    auto gQTile = tensor::local_tile(gQ, q_tile_layout, layout::make_coordinate(q_tile, 0));
    loadstore::copy_tile<KernelConfig>(sQ, gQTile, loadstore::CopyG2SOp{});
    __syncthreads();

    int local_row = tid;
    int global_row = q_tile * KernelConfig::BlockM + local_row;

    float acc[KernelConfig::HeadDim];
    #pragma unroll
    for (int d = 0; d < KernelConfig::HeadDim; ++d) {
        acc[d] = 0.0f;
    }
    softmax::OnlineSoftmaxState<1> sm_state;

    for (int kv_tile = 0; kv_tile < args.seq_len / KernelConfig::BlockN; ++kv_tile) {
        auto gKTile = tensor::local_tile(gK, kv_tile_layout, layout::make_coordinate(kv_tile, 0));
        auto gVTile = tensor::local_tile(gV, kv_tile_layout, layout::make_coordinate(kv_tile, 0));

        loadstore::copy_tile<KernelConfig>(sK, gKTile, loadstore::CopyG2SOp{});
        loadstore::copy_tile<KernelConfig>(sV, gVTile, loadstore::CopyG2SOp{});
        __syncthreads();

        if (local_row < KernelConfig::BlockM && global_row < args.seq_len) {
            if (kv_tile == 0) {
                softmax::online_softmax_init(sm_state);
            }

            #pragma unroll
            for (int j = 0; j < KernelConfig::BlockN; ++j) {
                int global_col = kv_tile * KernelConfig::BlockN + j;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < KernelConfig::HeadDim; ++d) {
                    float qv = to_float(sQ(local_row, d));
                    float kv = to_float(sK(j, d));
                    score += qv * kv;
                }

                score *= args.softmax_scale;
                if (args.causal && global_col > global_row) {
                    score = -INFINITY;
                }

                float score_frag[1][1] = {{score}};
                softmax::online_softmax_step<1, 1>(
                    score_frag, sm_state, 1.0f, (kv_tile == 0 && j == 0),
                    [&](int, float alpha) {
                        #pragma unroll
                        for (int d = 0; d < KernelConfig::HeadDim; ++d) {
                            acc[d] *= alpha;
                        }
                    });
                float p = score_frag[0][0];

                #pragma unroll
                for (int d = 0; d < KernelConfig::HeadDim; ++d) {
                    float vv = to_float(sV(j, d));
                    acc[d] += p * vv;
                }
            }
        }

        __syncthreads();
    }

    if (local_row < KernelConfig::BlockM && global_row < args.seq_len) {
        softmax::online_softmax_finalize<1>(sm_state, [&](int, float inv_l) {
            #pragma unroll
            for (int d = 0; d < KernelConfig::HeadDim; ++d) {
                O[o_base + static_cast<index_t>(global_row) * args.stride_seq_o + d] =
                    from_float<Element>(acc[d] * inv_l);
            }
        });
    }
}

} // namespace forward
