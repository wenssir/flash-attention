#pragma once

#include <cstdint>

#include "../tensor_core/tensor.cuh"
#include "../config/macros.cuh"
#include "../layout/coordinate.h"
#include "../layout/layout.h"
#include "./thread_map.cuh"

namespace loadstore {

template <typename Element>
struct GM2SMVec {
    DEVICE static void copy(Element const* gmem, Element* smem) {
        *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<uint4 const*>(gmem);
    }
};

template <typename Map, typename TensorG, typename TensorS>
DEVICE void copy_g2s(TensorG const& gmem, TensorS& smem) {
    using Element = typename Map::Element;
    static_assert((Map::kVecElems * static_cast<int>(sizeof(Element))) == 16,
                  "copy_g2s expects 16-byte copy per lane");

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));

    #pragma unroll
    for (int ir = 0; ir < Map::IterRows; ++ir) {
        int row_delta = Map::row_advance(ir);
        #pragma unroll
        for (int ic = 0; ic < Map::IterCols; ++ic) {
            int col_delta = Map::col_advance(ic);
            auto coord = layout::make_coordinate(base_row + row_delta, base_col + col_delta);
            GM2SMVec<Element>::copy(
                &gmem(coord),
                &smem(coord));
        }
    }
}

template <typename Map, typename TensorG, typename TensorS>
DEVICE void copy_g2s_predicated(TensorG const& gmem, TensorS& smem, int valid_rows) {
    using Element = typename Map::Element;
    static_assert((Map::kVecElems * static_cast<int>(sizeof(Element))) == 16,
                  "copy_g2s_predicated expects 16-byte copy per lane");
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));

    #pragma unroll
    for (int ir = 0; ir < Map::IterRows; ++ir) {
        int row_delta = Map::row_advance(ir);
        int row = base_row + row_delta;
        #pragma unroll
        for (int ic = 0; ic < Map::IterCols; ++ic) {
            int col_delta = Map::col_advance(ic);
            auto coord = layout::make_coordinate(row, base_col + col_delta);
            Element* s_ptr = &smem(coord);
            if (row < valid_rows) {
                GM2SMVec<Element>::copy(
                    &gmem(coord),
                    s_ptr);
            } else {
                *reinterpret_cast<uint4*>(s_ptr) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_q(TensorG const& gmem, TensorS& smem) {
    using Map = G2SLayoutThreadMap<
        KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s<Map>(gmem, smem);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_q_predicated(TensorG const& gmem, TensorS& smem, int valid_rows) {
    using Map = G2SLayoutThreadMap<
        KernelConfig, KernelConfig::BlockM, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated<Map>(gmem, smem, valid_rows);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_kv(TensorG const& gmem, TensorS& smem) {
    using Map = G2SLayoutThreadMap<
        KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s<Map>(gmem, smem);
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_g2s_kv_predicated(TensorG const& gmem, TensorS& smem, int valid_rows) {
    using Map = G2SLayoutThreadMap<
        KernelConfig, KernelConfig::BlockN, KernelConfig::HeadDim, KernelConfig::copy_k>;
    copy_g2s_predicated<Map>(gmem, smem, valid_rows);
}

struct CopyG2SOp {
  template <typename KernelConfig, typename TensorS, typename TensorG>
  DEVICE void operator()(TensorS& smem, TensorG const& gmem) const {
    copy_g2s_q<KernelConfig>(gmem, smem);
  }
};

struct CopyG2SKVOp {
  template <typename KernelConfig, typename TensorS, typename TensorG>
  DEVICE void operator()(TensorS& smem, TensorG const& gmem) const {
    copy_g2s_kv<KernelConfig>(gmem, smem);
  }
};

template <typename KernelConfig, typename TensorQ, typename TensorGQ>
DEVICE void load_q(TensorQ& sQ, TensorGQ const& gQTile, int valid_q_rows) {
    if (valid_q_rows == KernelConfig::BlockM) {
        copy_g2s_q<KernelConfig>(gQTile, sQ);
    } else {
        copy_g2s_q_predicated<KernelConfig>(gQTile, sQ, valid_q_rows);
    }
}

template <typename KernelConfig, typename TensorK, typename TensorGKLayout>
DEVICE void load_k(
    TensorK& sK,
    typename KernelConfig::Element* K,
    typename KernelConfig::index_t k_base,
    TensorGKLayout const& gk_layout,
    int kv_block_idx,
    int seq_len,
    int stride_seq_k
) {
    using index_t = typename KernelConfig::index_t;
    index_t k_offset = static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * stride_seq_k;
    auto gKTile = tensor::make_tensor(K + k_base + k_offset, gk_layout);
    int valid_k_rows = max(0, min(KernelConfig::BlockN, seq_len - kv_block_idx * KernelConfig::BlockN));
    if (valid_k_rows == KernelConfig::BlockN) {
        copy_g2s_kv<KernelConfig>(gKTile, sK);
    } else {
        copy_g2s_kv_predicated<KernelConfig>(gKTile, sK, valid_k_rows);
    }
}

template <typename KernelConfig, typename TensorV, typename TensorGVLayout>
DEVICE void load_v(
    TensorV& sV,
    typename KernelConfig::Element* V,
    typename KernelConfig::index_t v_base,
    TensorGVLayout const& gv_layout,
    int kv_block_idx,
    int seq_len,
    int stride_seq_v
) {
    using index_t = typename KernelConfig::index_t;
    index_t v_offset = static_cast<index_t>(kv_block_idx * KernelConfig::BlockN) * stride_seq_v;
    auto gVTile = tensor::make_tensor(V + v_base + v_offset, gv_layout);
    int valid_v_rows = max(0, min(KernelConfig::BlockN, seq_len - kv_block_idx * KernelConfig::BlockN));
    if (valid_v_rows == KernelConfig::BlockN) {
        copy_g2s_kv<KernelConfig>(gVTile, sV);
    } else {
        copy_g2s_kv_predicated<KernelConfig>(gVTile, sV, valid_v_rows);
    }
}

} // namespace loadstore
