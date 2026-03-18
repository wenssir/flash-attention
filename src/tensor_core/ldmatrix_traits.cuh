#pragma once

#include "../config/macros.cuh"
#include "../layout/coordinate.h"
#include "../ptx/ptx.cuh"

namespace tensor {

template <int M_, int N_, int K_>
struct MmaShape {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
};

using MmaShapeM16N8K16 = MmaShape<16, 8, 16>;

template <typename T>
struct LdmatrixTraitsQ {
    using Ldmatrix = ptx::LDMATRIX_X4<T>;

    DEVICE static auto coord(int lane_id) {
        if (lane_id < 16) {
            return layout::make_coordinate(lane_id, 0);
        }
        return layout::make_coordinate(lane_id - 16, 8);
    }
};

template <typename T>
struct LdmatrixTraitsK {
    using Ldmatrix = ptx::LDMATRIX_X2_TRANS<T>;

    DEVICE static auto coord(int lane_id) {
        return layout::make_coordinate(lane_id % 16, 0);
    }
};

template <typename T>
struct LdmatrixTraits8x16Trans {
    using Ldmatrix = ptx::LDMATRIX_X2_TRANS<T>;

    DEVICE static auto coord(int lane_id) {
        return layout::make_coordinate(lane_id % 8, 0);
    }
};

template <typename T>
using LdmatrixHelperQ = LdmatrixTraitsQ<T>;

template <typename T>
using LdmatrixHelperK = LdmatrixTraitsK<T>;

template <typename MmaShapeT>
struct RF2SmemLayoutTraits;

template <>
struct RF2SmemLayoutTraits<MmaShapeM16N8K16> {
    static constexpr int regs_per_lane = 4;

    DEVICE static void lane_reg_to_mn(int lane, int reg_idx, int& m, int& n) {
        int row_group = lane >> 4;
        int col_group = (lane >> 2) & 0x3;
        int row_pair = lane & 0x3;
        int row_offset = reg_idx >= 2 ? 8 : 0;
        int col_offset = reg_idx & 0x1;

        m = row_offset + row_group * 4 + col_group;
        n = row_pair * 2 + col_offset;
    }
};

} // namespace tensor
