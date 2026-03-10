#pragma once

#include "../config/macros.cuh"
#include "../tensor_core/accum_layout_traits.cuh"
#include "../tensor_core/fragment.cuh"

namespace loadstore {

template <typename T, int N>
DEVICE auto reg_at(tensor::AccumFragment<T, N> const& regs, int idx) {
    return regs(idx);
}

template <typename T, typename Helper, int Rows, int Cols>
DEVICE auto reg_at(tensor::Fragment<T, Helper, Rows, Cols> const& regs, int idx) {
    return regs(idx);
}

template <typename Regs>
DEVICE auto reg_at(Regs const& regs, int idx) {
    return regs[idx];
}

template <
    bool StoreLogical = true,
    typename LayoutTraits = tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>,
    typename TensorS,
    typename Regs>
DEVICE void copy_r2s(TensorS& smem, Regs const& regs) {
    int lane = threadIdx.x & 31;

    static_assert(StoreLogical, "Only logical accumulator store is supported");

    #pragma unroll
    for (int i = 0; i < LayoutTraits::regs_per_lane; ++i) {
        int m = 0;
        int n = 0;
        LayoutTraits::lane_reg_to_mn(lane, i, m, n);
        smem(m, n) = reg_at(regs, i);
    }
}

template <bool StoreLogical = true, typename LayoutTraits = tensor::AccumLayoutTraits<tensor::MmaShapeM16N8K16>>
struct CopyR2SOp {
    template <typename TensorS, typename Regs>
    DEVICE void operator()(TensorS& smem, Regs const& regs) const {
        copy_r2s<StoreLogical, LayoutTraits>(smem, regs);
    }
};

struct CopyR2SRawOp {
    template <typename TensorS, typename Regs>
    DEVICE void operator()(TensorS& smem, Regs const& regs) const {
        copy_r2s<true>(smem, regs);
    }
};

} // namespace loadstore
