#pragma once

#include "./copy_gsm.cuh"

namespace loadstore {

template <typename Map, typename CopyOp, typename TensorS, typename Index>
DEVICE void copy_g2s_direct(typename Map::Element const* gmem,
                            Index gmem_stride,
                            TensorS& smem) {
    using Element = typename Map::Element;
    static_assert((Map::kVecElems * static_cast<int>(sizeof(Element))) == 16,
                  "copy_g2s_direct expects 16-byte copy per lane");

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));
    Element* smem_ptr = smem.data_ptr();

    #pragma unroll
    for (int ir = 0; ir < Map::IterRows; ++ir) {
        int row = base_row + Map::row_advance(ir);
        #pragma unroll
        for (int ic = 0; ic < Map::IterCols; ++ic) {
            int col = base_col + Map::col_advance(ic);
            Element* dst = smem_ptr + smem_offset_g2s<typename Map::Config>(row, col);
            CopyOp::copy(gmem + static_cast<Index>(row) * gmem_stride + col, dst);
        }
    }
}

template <typename Map, typename CopyOp, typename TensorS, typename Index>
DEVICE void copy_g2s_predicated_direct(typename Map::Element const* gmem,
                                       Index gmem_stride,
                                       TensorS& smem,
                                       int valid_rows) {
    using Element = typename Map::Element;
    static_assert((Map::kVecElems * static_cast<int>(sizeof(Element))) == 16,
                  "copy_g2s_predicated_direct expects 16-byte copy per lane");

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));
    Element* smem_ptr = smem.data_ptr();

    #pragma unroll
    for (int ir = 0; ir < Map::IterRows; ++ir) {
        int row = base_row + Map::row_advance(ir);
        #pragma unroll
        for (int ic = 0; ic < Map::IterCols; ++ic) {
            int col = base_col + Map::col_advance(ic);
            Element* dst = smem_ptr + smem_offset_g2s<typename Map::Config>(row, col);
            if (row < valid_rows) {
                CopyOp::copy(gmem + static_cast<Index>(row) * gmem_stride + col, dst);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_q_direct(typename KernelConfig::Element const* gmem,
                              Index gmem_stride,
                              TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_direct<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_q_async_direct(typename KernelConfig::Element const* gmem,
                                    Index gmem_stride,
                                    TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_direct<Map, GM2SMAsyncVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_q_predicated_direct(typename KernelConfig::Element const* gmem,
                                         Index gmem_stride,
                                         TensorS& smem,
                                         int valid_rows) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated_direct<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem, valid_rows);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_q_async_predicated_direct(typename KernelConfig::Element const* gmem,
                                               Index gmem_stride,
                                               TensorS& smem,
                                               int valid_rows) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated_direct<Map, GM2SMAsyncVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem, valid_rows);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_kv_direct(typename KernelConfig::Element const* gmem,
                               Index gmem_stride,
                               TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_direct<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_kv_async_direct(typename KernelConfig::Element const* gmem,
                                     Index gmem_stride,
                                     TensorS& smem) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_direct<Map, GM2SMAsyncVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_kv_predicated_direct(typename KernelConfig::Element const* gmem,
                                          Index gmem_stride,
                                          TensorS& smem,
                                          int valid_rows) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated_direct<Map, GM2SMVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem, valid_rows);
}

template <typename KernelConfig, typename TensorS, typename Index>
DEVICE void copy_g2s_kv_async_predicated_direct(typename KernelConfig::Element const* gmem,
                                                Index gmem_stride,
                                                TensorS& smem,
                                                int valid_rows) {
    using Map = G2SLayoutThreadMap<KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated_direct<Map, GM2SMAsyncVec<typename KernelConfig::Element>>(gmem, gmem_stride, smem, valid_rows);
}

} // namespace loadstore
