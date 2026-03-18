#pragma once

#include <cstdint>

#include "../tensor_core/tensor.cuh"
#include "../config/macros.cuh"
#include "../layout/coordinate.h"
#include "../layout/layout.h"
#include "../ptx/copy.cuh"
#include "./thread_map.cuh"

namespace loadstore {

template <typename Element>
struct GM2SMVec {
    DEVICE static void copy(Element const* gmem, Element* smem) {
        *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<uint4 const*>(gmem);
    }
};

template <typename Element>
struct GM2SMAsyncVec {
    DEVICE static void copy(Element const* gmem, Element* smem) {
        using Vec = uint4;
        ptx::CP_ASYNC_CACHE_GLOBAL<Vec, Vec>::copy(
            *reinterpret_cast<Vec const*>(gmem),
            *reinterpret_cast<Vec*>(smem));
    }
};

template <typename KernelConfig>
DEVICE int smem_offset_g2s(int row, int col) {
    int inner_row = row % KernelConfig::SmemAtomRows;
    int tile_row = row / KernelConfig::SmemAtomRows;
    int inner_col = col % KernelConfig::SmemAtomCols;
    int tile_col = col / KernelConfig::SmemAtomCols;

    constexpr int kAtomElems = KernelConfig::SmemAtomRows * KernelConfig::SmemAtomCols;
    constexpr int kTileRowStride = KernelConfig::HeadDim * KernelConfig::SmemAtomRows;

    int atom_offset = inner_row * KernelConfig::SmemAtomCols + inner_col;
    if constexpr (KernelConfig::UseSmemSwizzle) {
        atom_offset = tensor::Swizzle<3, 3, 3>{}(atom_offset);
    }

    return tile_row * kTileRowStride + tile_col * kAtomElems + atom_offset;
}

template <typename Map, typename CopyOp, typename TensorG, typename TensorS>
DEVICE void copy_g2s(TensorG const& gmem, TensorS& smem) {
    using Element = typename Map::Element;
    static_assert((Map::kVecElems * static_cast<int>(sizeof(Element))) == 16,
                  "copy_g2s expects 16-byte copy per lane");
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));
    auto* gmem_ptr = gmem.data_ptr();
    auto gmem_layout = gmem.layout();
    Element* smem_ptr = smem.data_ptr();

    #pragma unroll
    for (int ir = 0; ir < Map::IterRows; ++ir) {
        int row = base_row + Map::row_advance(ir);
        #pragma unroll
        for (int ic = 0; ic < Map::IterCols; ++ic) {
            int col = base_col + Map::col_advance(ic);
            auto gmem_offset = gmem_layout(layout::make_coordinate(row, col));
            Element* dst = smem_ptr + smem_offset_g2s<typename Map::Config>(row, col);
            CopyOp::copy(gmem_ptr + gmem_offset, dst);
        }
    }
}

template <typename Map, typename CopyOp, typename TensorG, typename TensorS>
DEVICE void copy_g2s_predicated(TensorG const& gmem, TensorS& smem, int valid_rows) {
    using Element = typename Map::Element;
    static_assert((Map::kVecElems * static_cast<int>(sizeof(Element))) == 16,
                  "copy_g2s_predicated expects 16-byte copy per lane");
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));
    auto* gmem_ptr = gmem.data_ptr();
    auto gmem_layout = gmem.layout();
    Element* smem_ptr = smem.data_ptr();

    #pragma unroll
    for (int ir = 0; ir < Map::IterRows; ++ir) {
        int row = base_row + Map::row_advance(ir);
        #pragma unroll
        for (int ic = 0; ic < Map::IterCols; ++ic) {
            int col = base_col + Map::col_advance(ic);
            Element* dst = smem_ptr + smem_offset_g2s<typename Map::Config>(row, col);
            if (row < valid_rows) {
                auto gmem_offset = gmem_layout(layout::make_coordinate(row, col));
                CopyOp::copy(gmem_ptr + gmem_offset, dst);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_q(TensorG const& gmem, TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, smem);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_q_async(TensorG const& gmem, TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s<Map, GM2SMAsyncVec<typename KernelConfig::Element>>(gmem, smem);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_q_predicated(TensorG const& gmem, TensorS& smem, int valid_rows) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, smem, valid_rows);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_kv(TensorG const& gmem, TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, smem);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_kv_async(TensorG const& gmem, TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s<Map, GM2SMAsyncVec<typename KernelConfig::Element>>(gmem, smem);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_kv_predicated(TensorG const& gmem, TensorS& smem, int valid_rows) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, smem, valid_rows);
}

} // namespace loadstore
