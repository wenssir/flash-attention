#pragma once

#include "./copy_srm.cuh"

namespace loadstore {

template <typename Frag, typename Ldmatrix, typename Element>
DEVICE void ldmatrix_load(Frag& frag, Element* ptr) {
    if constexpr (Frag::REGS_PER_LANE == 4) {
        Ldmatrix::load(ptr, frag(0), frag(1), frag(2), frag(3));
    } else if constexpr (Frag::REGS_PER_LANE == 2) {
        Ldmatrix::load(ptr, frag(0), frag(1));
    } else if constexpr (Frag::REGS_PER_LANE == 1) {
        Ldmatrix::load(ptr, frag(0));
    }
}

template <typename KernelConfig>
DEVICE void load_fragment_q_direct(typename KernelConfig::FragQ& frag,
                                   typename KernelConfig::Element* smem_ptr,
                                   int warp,
                                   int k_tile) {
    using Helper = tensor::LdmatrixHelperQ<typename KernelConfig::Element>;
    int lane_id = threadIdx.x & 31;
    auto coord = Helper::coord(lane_id);
    int row = warp * KernelConfig::MmaM + static_cast<int>(cxx::get<0>(coord));
    int col = k_tile * KernelConfig::MmaK + static_cast<int>(cxx::get<1>(coord));
    ldmatrix_load<typename KernelConfig::FragQ, typename Helper::Ldmatrix>(
        frag, smem_ptr + smem_offset<KernelConfig>(row, col));
}

template <typename KernelConfig>
DEVICE void load_fragment_k_direct(typename KernelConfig::FragK& frag,
                                   typename KernelConfig::Element* smem_ptr,
                                   int out_tile,
                                   int k_tile) {
    using Helper = tensor::LdmatrixHelperK<typename KernelConfig::Element>;
    int lane_id = threadIdx.x & 31;
    auto coord = Helper::coord(lane_id);
    int row = out_tile * KernelConfig::MmaN + static_cast<int>(cxx::get<0>(coord));
    int col = k_tile * KernelConfig::MmaK + static_cast<int>(cxx::get<1>(coord));
    ldmatrix_load<typename KernelConfig::FragK, typename Helper::Ldmatrix>(
        frag, smem_ptr + smem_offset<KernelConfig>(row, col));
}

template <typename KernelConfig>
DEVICE void load_fragment_v_direct(typename KernelConfig::FragV& frag,
                                   typename KernelConfig::Element* smem_ptr,
                                   int k_tile,
                                   int out_tile) {
    using Helper = tensor::LdmatrixHelperK<typename KernelConfig::Element>;
    int lane_id = threadIdx.x & 31;
    auto coord = Helper::coord(lane_id);
    int row = k_tile * KernelConfig::MmaK + static_cast<int>(cxx::get<0>(coord));
    int col = out_tile * KernelConfig::MmaN + static_cast<int>(cxx::get<1>(coord));
    ldmatrix_load<typename KernelConfig::FragV, typename Helper::Ldmatrix>(
        frag, smem_ptr + smem_offset<KernelConfig>(row, col));
}

} // namespace loadstore
