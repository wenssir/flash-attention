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
    static constexpr int kThreadM = KernelConfig::thread_m;
    static constexpr int kThreadK = KernelConfig::thread_k;
    static constexpr int kVecElems = VecElems;
    static constexpr int kNWarps = KernelConfig::NWarps;

    static_assert((BlockRows % kNWarps) == 0,
                  "BlockRows must be divisible by NWarps");
    static_assert((BlockCols % (kThreadK * kVecElems)) == 0,
                  "BlockCols must be divisible by thread_k * vec_elems");
    static_assert(((BlockRows / kNWarps) % kThreadM) == 0,
                  "rows per warp must be divisible by thread_m");

    static constexpr int WarpRows = BlockRows / kNWarps;
    static constexpr int OpRows = kThreadM;
    static constexpr int OpCols = kThreadK * kVecElems;
    static constexpr int IterRows = WarpRows / OpRows;
    static constexpr int IterCols = BlockCols / OpCols;

    DEVICE static constexpr auto lane_coord(int lane_id) {
        int row = lane_id / kThreadK;
        int col = (lane_id % kThreadK) * kVecElems;
        return layout::make_coordinate(row, col);
    }

    DEVICE static constexpr int warp_row_base(int warp_id) {
        return warp_id * WarpRows;
    }

    DEVICE static constexpr int row_advance(int iter_row) {
        return iter_row * OpRows;
    }

    DEVICE static constexpr int col_advance(int iter_col) {
        return iter_col * OpCols;
    }
};

template <typename KernelConfig>
struct LdmatrixThreadMap {
    static constexpr int kWarpSize = 32;

    DEVICE static constexpr int lane_id(int tid) {
        return tid & (kWarpSize - 1);
    }

    DEVICE static constexpr int warp_id(int tid) {
        return tid / kWarpSize;
    }

    DEVICE static constexpr auto warp_coord(int wid) {
        return layout::make_coordinate(wid, 0);
    }
};

template <typename KernelConfig>
struct S2RQThreadMap {
    static constexpr int kWarpSize = 32;

    DEVICE static constexpr int lane_id(int tid) {
        return tid & (kWarpSize - 1);
    }
};

template <typename KernelConfig>
struct S2RKVThreadMap {
    static constexpr int kWarpSize = 32;

    DEVICE static constexpr int lane_id(int tid) {
        return tid & (kWarpSize - 1);
    }
};

template <typename KernelConfig>
struct AccumThreadMap {
    static constexpr int kWarpSize = 32;

    DEVICE static constexpr int lane_id(int tid) {
        return tid & (kWarpSize - 1);
    }

    DEVICE static constexpr int warp_id(int tid) {
        return tid / kWarpSize;
    }

    DEVICE static constexpr auto warp_coord(int wid) {
        return layout::make_coordinate(wid, 0);
    }
};

template <typename LayoutTraits>
struct R2SThreadMap {
    static constexpr int kWarpSize = 32;
    static constexpr int regs_per_lane = LayoutTraits::regs_per_lane;

    DEVICE static constexpr int lane_id(int tid) {
        return tid & (kWarpSize - 1);
    }

    DEVICE static void lane_reg_to_mn(int lane, int reg_idx, int& m, int& n) {
        LayoutTraits::lane_reg_to_mn(lane, reg_idx, m, n);
    }
};

} // namespace loadstore
