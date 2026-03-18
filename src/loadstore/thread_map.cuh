#pragma once

#include "../config/macros.cuh"
#include "../layout/coordinate.h"

namespace loadstore {

template <typename KernelConfig, int BlockRows, int BlockCols, int VecElems = KernelConfig::copy_k>
struct G2SLayoutThreadMap {
    using Config = KernelConfig;
    using Element = typename KernelConfig::Element;
    static constexpr int kBlockRows = BlockRows;
    static constexpr int kBlockCols = BlockCols;
    static constexpr int kVecElems = VecElems;
    static constexpr int kNWarps = KernelConfig::NWarps;
    static constexpr int kAtomRows = KernelConfig::SmemAtomRows;
    static constexpr int kAtomCols = KernelConfig::SmemAtomCols;
    static constexpr int kWarpSize = 32;
    static constexpr int kThreadsPerRow = kAtomCols / kVecElems;
    static constexpr int kRowsPerIter = kWarpSize / kThreadsPerRow;

    static_assert((BlockRows % kNWarps) == 0,
                  "BlockRows must be divisible by NWarps");
    static_assert((kAtomCols % kVecElems) == 0,
                  "Smem atom width must be divisible by vec elems");
    static_assert((kWarpSize % kThreadsPerRow) == 0,
                  "Threads per row must divide warp size");
    static_assert((BlockCols % kAtomCols) == 0,
                  "BlockCols must be divisible by smem atom width");
    static_assert(((BlockRows / kNWarps) % kRowsPerIter) == 0,
                  "Rows per warp must be divisible by rows per iter");

    static constexpr int WarpRows = BlockRows / kNWarps;
    static constexpr int IterRows = WarpRows / kRowsPerIter;
    static constexpr int IterCols = BlockCols / kAtomCols;

    DEVICE static constexpr auto lane_coord(int lane_id) {
        int row = lane_id / kThreadsPerRow;
        int col = (lane_id % kThreadsPerRow) * kVecElems;
        return layout::make_coordinate(row, col);
    }

    DEVICE static constexpr int warp_row_base(int warp_id) {
        return warp_id * WarpRows;
    }

    DEVICE static constexpr int row_advance(int iter_row) {
        return iter_row * kRowsPerIter;
    }

    DEVICE static constexpr int col_advance(int iter_col) {
        return iter_col * kAtomCols;
    }
};

} // namespace loadstore
