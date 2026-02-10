#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "../common/common.h"
#include "../layout/shape.h"
#include "../layout/stride.cuh"
#include "../layout/layout.h"
#include "../layout/composedLayout.h"
#include "macros.cuh"
#include "../tensor_core/swizzle.cuh"

namespace config {

struct ForwardKernelArgs {
    void* __restrict__ Q;
    void* __restrict__ K;
    void* __restrict__ V;
    void* __restrict__ O;

    int stride_batch_q;
    int stride_seq_q;
    int stride_head_q;
    int stride_batch_k;
    int stride_seq_k;
    int stride_head_k;
    int stride_batch_v;
    int stride_seq_v;
    int stride_head_v;
    int stride_batch_o;
    int stride_seq_o;
    int stride_head_o;

    int seq_len;
    int heads;
    int head_dim;

    float softmax_scale;
    int causal; // 0: disabled, 1: causal mask enabled
};

template <typename ELEMENT>
struct KernelConfig {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using Element = ELEMENT;
    static constexpr bool Has_cp_async = true;
#else
    using Element = ELEMENT;
    static constexpr bool Has_cp_async = false;
#endif
    using ElementAccum = float;
    using index_t = int64_t;
};

template <int HEADDIM, int BLOCKM, int BLOCKN, int NWARPS, typename ELEMENT,
          typename Base = KernelConfig<ELEMENT>>
struct FlashFwdKernelConfig : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;

    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static constexpr int BlockM = BLOCKM;
    static constexpr int BlockN = BLOCKN;
    static constexpr int HeadDim = HEADDIM;
    static constexpr int NWarps = NWARPS;
    static constexpr int NThreads = NWarps * 32;

    // For warp g2s copy of (BLOCKM/NWARPS, HEADDIM) = (16, 128):
    // 32 lanes arranged as (4,8), each lane copies 8 contiguous elems.
    static constexpr int thread_m = 4;
    static constexpr int thread_k = 8;
    static constexpr int copy_k = 8;
    static constexpr bool UseSmemSwizzle = false;

    static constexpr auto warp_tile_shape =
        layout::make_shape(numeric::Int<16>{}, numeric::Int<128>{});
    static constexpr auto thread_shape_in_warp =
        layout::make_shape(numeric::Int<1>{}, numeric::Int<copy_k>{});

    // Swizzled layouts (kept for future optimization, disabled now).
    // using SmemLayoutQSwizzle = decltype(layout::composition(
    //     tensor::Swizzle<3, 3, 3>{},
    //     layout::make_layout(
    //         layout::make_shape(numeric::Int<BLOCKM>{}, numeric::Int<HEADDIM>{}),
    //         layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{}))));
    // using SmemLayoutKVSwizzle = decltype(layout::composition(
    //     tensor::Swizzle<3, 3, 3>{},
    //     layout::make_layout(
    //         layout::make_shape(numeric::Int<BLOCKN>{}, numeric::Int<HEADDIM>{}),
    //         layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{}))));

    using SmemLayoutQNoSwizzle = decltype(layout::composition(
        tensor::NoSwizzle{},
        layout::make_layout(
            layout::make_shape(numeric::Int<BLOCKM>{}, numeric::Int<HEADDIM>{}),
            layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{}))));

    using SmemLayoutKVNoSwizzle = decltype(layout::composition(
        tensor::NoSwizzle{},
        layout::make_layout(
            layout::make_shape(numeric::Int<BLOCKN>{}, numeric::Int<HEADDIM>{}),
            layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{}))));

    using SmemLayoutQ = SmemLayoutQNoSwizzle;
    using SmemLayoutKV = SmemLayoutKVNoSwizzle;
};

using ForwardV3Config = FlashFwdKernelConfig<
    constants::V3_HEAD_DIM,
    constants::V3_BLOCK_M,
    constants::V3_BLOCK_N,
    constants::V3_WARPS,
    __half>;

} // namespace config
