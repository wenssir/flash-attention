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

    DEVICE static void lane_reg_to_raw(int lane, int reg_idx, int& rr, int& rc) {
        int quad = lane & 0x3;
        int col_pair = (lane >> 2) & 0x3;
        int row_group = lane >> 4;

        int row0 = row_group * 4 + quad;
        int row1 = row0 + 8;
        int col0 = col_pair * 2;
        int col1 = col0 + 1;

        if (reg_idx == 0) {
            rr = row0;
            rc = col0;
        } else if (reg_idx == 1) {
            rr = row0;
            rc = col1;
        } else if (reg_idx == 2) {
            rr = row1;
            rc = col0;
        } else {
            rr = row1;
            rc = col1;
        }
    }

    DEVICE static void raw_to_logical(int rr, int rc, int& m, int& n) {
        n = ((rr & 0x3) << 1) | (rc & 0x1);
        m = ((rr >> 2) << 2) | (rc >> 1);
    }
};

} // namespace tensor

