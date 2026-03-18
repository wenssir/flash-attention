#pragma once

#include "../config/macros.cuh"
#include "../layout/coordinate.h"
#include "../numeric/Int.cuh"
#include "../ptx/ptx.cuh"
#include "../tensor_core/ldmatrix_traits.cuh"
#include "../tensor_core/tensor.cuh"
#include "../tensor_core/swizzle.cuh"
#include "./thread_map.cuh"

namespace loadstore {

template <typename KernelConfig>
DEVICE int smem_offset(int row, int col) {
    int inner_row = row % KernelConfig::SmemAtomRows;
    int tile_row = row / KernelConfig::SmemAtomRows;
    int inner_col = col % KernelConfig::SmemAtomCols;
    int tile_col = col / KernelConfig::SmemAtomCols;

    constexpr int kAtomElems = KernelConfig::SmemAtomRows * KernelConfig::SmemAtomCols;
    constexpr int kTileRowStride = KernelConfig::HeadDim * KernelConfig::SmemAtomRows;

    int atom_offset = inner_row * KernelConfig::SmemAtomCols + inner_col;
    if constexpr (KernelConfig::UseSmemSwizzle) {
        atom_offset = tensor::Swizzle<3, 3, 3>{}(atom_offset);
    }

    return tile_row * kTileRowStride + tile_col * kAtomElems + atom_offset;
}

template <typename KernelConfig, typename TensorS>
struct FragmentTensorView {
    TensorS& tensor;
    int base_row;
    int base_col;

    DEVICE auto data_ptr() const {
        return tensor.data_ptr();
    }

    DEVICE int offset(int row, int col) const {
        return smem_offset<KernelConfig>(base_row + row, base_col + col);
    }
};

template <typename KernelConfig, typename TensorS>
DEVICE auto partition_fragment_A(TensorS& s_tensor, int warp, int k_tile) {
    return FragmentTensorView<KernelConfig, TensorS>{
        s_tensor,
        warp * KernelConfig::MmaM,
        k_tile * KernelConfig::MmaK};
}

template <typename KernelConfig, typename TensorS>
DEVICE auto partition_fragment_B(TensorS& s_tensor, int out_tile, int k_tile) {
    return FragmentTensorView<KernelConfig, TensorS>{
        s_tensor,
        out_tile * KernelConfig::MmaN,
        k_tile * KernelConfig::MmaK};
}

template <typename KernelConfig, typename TensorS>
DEVICE auto partition_fragment_V(TensorS& s_tensor, int k_tile, int out_tile) {
    return FragmentTensorView<KernelConfig, TensorS>{
        s_tensor,
        k_tile * KernelConfig::MmaK,
        out_tile * KernelConfig::MmaN};
}

template <typename KernelConfig, typename Frag, typename TensorS>
DEVICE void load_fragment(Frag& frag, TensorS& s_tensor) {
    int lane_id = threadIdx.x & 31;
    frag.load(s_tensor, lane_id);
}

} // namespace loadstore
