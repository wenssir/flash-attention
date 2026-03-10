#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

#include "../common/common.h"
#include "../layout/shape.h"
#include "../layout/stride.cuh"
#include "../layout/layout.h"
#include "../layout/composedLayout.h"
#include "macros.cuh"
#include "../tensor_core/fragment.cuh"
#include "../tensor_core/swizzle.cuh"
#include "../loadstore/copy_srm.cuh"
#include "../ptx/mma_atom.cuh"

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
    using Atom = ptx::Sm80MmaF16F16F32M16N8K16;
    using FragAcc = tensor::AccumFragment<float, 4>;
    using FragQ = tensor::Fragment<Element, loadstore::LdmatrixHelperQ<Element>, 16, 16>;
    using FragP = FragQ;
    using FragK = tensor::Fragment<
        Element,
        loadstore::LdmatrixHelper<ptx::LDMATRIX_X2_TRANS<Element>, 8, 16, true>,
        8,
        16>;
    using FragV = tensor::Fragment<Element, loadstore::LdmatrixHelperK<Element>, 16, 8>;

    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static constexpr int BlockM = BLOCKM;
    static constexpr int BlockN = BLOCKN;
    static constexpr int HeadDim = HEADDIM;
    static constexpr int NWarps = NWARPS;
    static constexpr int WarpSize = 32;
    static constexpr int NThreads = NWarps * WarpSize;
    static constexpr int RowsPerWarp = BlockM / NWarps;
    static constexpr int HalfHeadDim = HeadDim / 2;
    static constexpr int VecBytes = 16;
    static constexpr float Log2e = 1.4426950408889634f;
    static constexpr int MmaM = 16;
    static constexpr int MmaN = 8;
    static constexpr int MmaK = 16;

    // For warp g2s copy of (BLOCKM/NWARPS, HEADDIM) = (16, 128):
    // 32 lanes arranged as (4,8), each lane copies 8 contiguous elems.
    static constexpr int thread_m = 4;
    static constexpr int thread_k = 8;
    static constexpr int copy_k = 8;
    static constexpr bool UseSmemSwizzle = true;
    static constexpr int QkKSteps = HeadDim / MmaK;
    static constexpr int ScoreTiles = BlockN / MmaN;
    static constexpr int PvKTiles = BlockN / MmaK;
    static constexpr int OutputTiles = HeadDim / MmaN;

    static_assert(std::is_same_v<Element, __half>, "v4 skeleton only supports fp16");
    static_assert(BlockM == NWarps * MmaM, "v4 requires BlockM == NWarps * 16");
    static_assert((HeadDim % MmaK) == 0, "HeadDim must be divisible by 16");
    static_assert((BlockN % MmaN) == 0, "BlockN must be divisible by 8");
    static_assert((BlockN % MmaK) == 0, "v4 PV path requires BlockN divisible by 16");

    static constexpr auto warp_tile_shape =
        layout::make_shape(numeric::Int<16>{}, numeric::Int<128>{});
    static constexpr auto thread_shape_in_warp =
        layout::make_shape(numeric::Int<1>{}, numeric::Int<copy_k>{});

    using SmemLayoutQLinear = decltype(layout::make_layout(
        layout::make_shape(numeric::Int<BLOCKM>{}, numeric::Int<HEADDIM>{}),
        layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{})));
    using SmemLayoutKVLinear = decltype(layout::make_layout(
        layout::make_shape(numeric::Int<BLOCKN>{}, numeric::Int<HEADDIM>{}),
        layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{})));

    using SmemLayoutQSwizzle = decltype(layout::composition(
        tensor::Swizzle<3, 3, 3>{},
        layout::make_layout(
            layout::make_shape(numeric::Int<BLOCKM>{}, numeric::Int<HEADDIM>{}),
            layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{}))));
    using SmemLayoutKVSwizzle = decltype(layout::composition(
        tensor::Swizzle<3, 3, 3>{},
        layout::make_layout(
            layout::make_shape(numeric::Int<BLOCKN>{}, numeric::Int<HEADDIM>{}),
            layout::make_stride(numeric::Int<HEADDIM>{}, numeric::Int<1>{}))));

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

    using SmemLayoutQ = std::conditional_t<UseSmemSwizzle, SmemLayoutQSwizzle, SmemLayoutQNoSwizzle>;
    using SmemLayoutKV = std::conditional_t<UseSmemSwizzle, SmemLayoutKVSwizzle, SmemLayoutKVNoSwizzle>;
    using QSubLayout = decltype(layout::make_layout(
        layout::make_shape(numeric::Int<MmaM>{}, numeric::Int<MmaK>{}),
        layout::make_stride(numeric::Int<HeadDim>{}, numeric::Int<1>{})));
    using KSubLayout = decltype(layout::make_layout(
        layout::make_shape(numeric::Int<MmaN>{}, numeric::Int<MmaK>{}),
        layout::make_stride(numeric::Int<HeadDim>{}, numeric::Int<1>{})));
    using VSubLayout = decltype(layout::make_layout(
        layout::make_shape(numeric::Int<MmaK>{}, numeric::Int<MmaN>{}),
        layout::make_stride(numeric::Int<HeadDim>{}, numeric::Int<1>{})));
    using OTileLayout = decltype(layout::make_layout(
        layout::make_shape(numeric::Int<MmaM>{}, numeric::Int<MmaN>{}),
        layout::make_stride(numeric::Int<MmaN>{}, numeric::Int<1>{})));

    static constexpr QSubLayout q_sub_layout{};
    static constexpr KSubLayout k_sub_layout{};
    static constexpr VSubLayout v_sub_layout{};
    static constexpr OTileLayout o_tile_layout{};
};

using ForwardConfig = FlashFwdKernelConfig<
    constants::HEAD_DIM,
    constants::BLOCK_M,
    constants::BLOCK_N,
    constants::WARPS,
    __half>;

} // namespace config
