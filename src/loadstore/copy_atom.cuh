#pragma once

#include "../config/macros.cuh"
#include "./copy_gsm.cuh"
#include "./copy_rsm.cuh"
#include "./copy_sgm.cuh"
#include "./copy_srm.cuh"

namespace loadstore {

template <typename DstTensor, typename SrcTensor, typename CopyOp>
DEVICE void copy_tile(DstTensor& dst, SrcTensor const& src, CopyOp const& op) {
    op(dst, src);
}

template <typename KernelConfig, typename DstTensor, typename SrcTensor, typename CopyOp>
DEVICE void copy_tile(DstTensor& dst, SrcTensor const& src, CopyOp const& op) {
    if constexpr (requires { op.template operator()<KernelConfig>(dst, src); }) {
        op.template operator()<KernelConfig>(dst, src);
    } else {
        op(dst, src);
    }
}

}
