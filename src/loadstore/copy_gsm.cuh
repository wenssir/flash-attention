#pragma once

#include "../config/macros.cuh"
#include "../layout/layout.h"
#include "../layout/coordinate.h"
#include "../tensor_core/tensor.cuh"
#include "../utils/print.h"

namespace loadstore {

template <typename KernelConfig>
struct ThreadMap {
    static constexpr int kThreadM = KernelConfig::thread_m;
    static constexpr int kThreadK = KernelConfig::thread_k;
    static constexpr int kCopyK = KernelConfig::copy_k;

    DEVICE static constexpr auto warp_coord(int warp_id) {
        return layout::make_coordinate(warp_id, 0);
    }

    DEVICE static constexpr auto thread_coord(int lane_id) {
        int tr = lane_id / kThreadK;
        int tc = lane_id % kThreadK;
        return layout::make_coordinate(tr, tc);
    }

    DEVICE static constexpr auto iter_coord(int ik) {
        return layout::make_coordinate(0, ik);
    }
};

template <typename KernelConfig>
DEVICE constexpr auto make_iter_tile_shape() {
    return layout::make_shape(
        numeric::Int<KernelConfig::thread_m>{},
        numeric::Int<KernelConfig::thread_k * KernelConfig::copy_k>{}
    );
}

template <typename KernelConfig, typename TensorG, typename TensorS>
DEVICE void copy_with_map_and_slice(TensorG const& gmem, TensorS& smem) {
    using Map = ThreadMap<KernelConfig>;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    auto g_shape = gmem.layout().shape();
    int tile_rows = static_cast<int>(layout::get_shape_size(cxx::get<0>(g_shape)));
    int tile_cols = static_cast<int>(layout::get_shape_size(cxx::get<1>(g_shape)));
    int warp_rows = tile_rows / KernelConfig::NWarps;

    auto warp_layout = layout::make_layout(
        layout::make_shape(warp_rows, tile_cols),
        gmem.layout().stride());
    auto g_warp = tensor::local_tile(gmem, warp_layout, Map::warp_coord(warp_id));
    auto s_warp = tensor::local_tile(smem, warp_layout, Map::warp_coord(warp_id));

    auto iter_tile_shape = make_iter_tile_shape<KernelConfig>();
    auto divided = layout::zipped_divide(warp_layout, iter_tile_shape);
    auto outer_shape = divided.outer.shape();
    int iter_m = static_cast<int>(layout::get_shape_size(cxx::get<0>(outer_shape)));
    int iter_k = static_cast<int>(layout::get_shape_size(cxx::get<1>(outer_shape)));

    auto lane_coord = Map::thread_coord(lane_id);
    int lane_row = cxx::get<0>(lane_coord);
    int lane_col = cxx::get<1>(lane_coord);
    int lane_col_base = lane_col * KernelConfig::copy_k;

    #pragma unroll
    for (int im = 0; im < iter_m; ++im) {
        #pragma unroll
        for (int ik = 0; ik < iter_k; ++ik) {
            auto iter_coord = layout::make_coordinate(im, ik);
            auto g_iter = tensor::local_tile(g_warp, divided.inner, iter_coord);
            auto s_iter = tensor::local_tile(s_warp, divided.inner, iter_coord);
            auto* g_ptr = &g_iter(lane_row, lane_col_base);
            auto* s_ptr = &s_iter(lane_row, lane_col_base);

            *reinterpret_cast<uint4*>(s_ptr) = *reinterpret_cast<const uint4*>(g_ptr);
        }
    }
}

struct CopyG2SOp {
  template <typename KernelConfig, typename TensorS, typename TensorG>
  DEVICE void operator()(TensorS& smem, TensorG const& gmem) const {
    copy_with_map_and_slice<KernelConfig>(gmem, smem);
  }
};

} // namespace loadstore
