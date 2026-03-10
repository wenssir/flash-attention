#pragma once

#include "../config/macros.cuh"

namespace tensor {

template <int M_, int N_, int K_>
struct MmaShape {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
};

using MmaShapeM16N8K16 = MmaShape<16, 8, 16>;

template <typename MmaShapeT>
struct AccumLayoutTraits;

template <>
struct AccumLayoutTraits<MmaShapeM16N8K16> {
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
