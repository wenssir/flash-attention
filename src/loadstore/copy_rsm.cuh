#pragma once

#include "../config/macros.cuh"
#include "../tensor_core/accum_layout_traits.cuh"

namespace loadstore {

template <typename Regs>
DEVICE auto reg_at(Regs const& regs, int idx) {
    if constexpr (requires { regs(idx); }) {
        return regs(idx);
    } else {
        return regs[idx];
    }
}

template <bool StoreLogical = true, typename LayoutTraits = tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>>
struct CopyR2SOp {
    template <typename TensorS, typename Regs>
    DEVICE void operator()(TensorS& smem, Regs const& regs) const {
        int lane = threadIdx.x % 32;
        #pragma unroll
        for (int i = 0; i < LayoutTraits::regs_per_lane; ++i) {
            int rr = 0;
            int rc = 0;
            LayoutTraits::lane_reg_to_raw(lane, i, rr, rc);
            store_one(smem, rr, rc, reg_at(regs, i));
        }
    }

private:
    template <typename TensorS, typename T>
    DEVICE static void store_one(TensorS& smem, int raw_r, int raw_c, T const& value) {
        if constexpr (StoreLogical) {
            int m, n;
            LayoutTraits::raw_to_logical(raw_r, raw_c, m, n);
            smem(m, n) = value;
        } else {
            smem(raw_r, raw_c) = value;
        }
    }
};

template <typename TensorS, typename Regs>
DEVICE void copy_r2s(TensorS& smem, Regs const& regs) {
    CopyR2SOp<true> op{};
    op(smem, regs);
}

} // namespace loadstore
