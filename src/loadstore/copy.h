#pragma once

#include "../config/macros.cuh"

namespace loadstore {

template <typename CopyOp, typename SrcView, typename DstView>
DEVICE void copy(CopyOp const&, SrcView const& src, DstView& dst, int base_row, int base_col) {
    #pragma unroll
    for (int ir = 0; ir < SrcView::IterRows; ++ir) {
        #pragma unroll
        for (int ic = 0; ic < SrcView::IterCols; ++ic) {
            CopyOp::copy(src.ptr(ir, ic, base_row, base_col), dst.ptr(ir, ic, base_row, base_col));
        }
    }
}

template <typename CopyOp, typename SrcView, typename DstView, typename Pred>
DEVICE void copy_if(CopyOp const&, SrcView const& src, DstView& dst, int base_row, int base_col, Pred&& pred) {
    #pragma unroll
    for (int ir = 0; ir < SrcView::IterRows; ++ir) {
        #pragma unroll
        for (int ic = 0; ic < SrcView::IterCols; ++ic) {
            if (pred(ir, ic)) {
                CopyOp::copy(src.ptr(ir, ic, base_row, base_col), dst.ptr(ir, ic, base_row, base_col));
            }
        }
    }
}

} // namespace loadstore
