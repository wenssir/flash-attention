#pragma once

#include "../config/macros.cuh"
#include "./ptx.cuh"

namespace ptx {

// c is updated in-place: c = a * b + c.
template <typename FragA, typename FragB, typename FragC>
DEVICE void mma_atom(FragA const& a, FragB const& b, FragC& c) {
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    ptx::MMA_M16N8K16_F32<__half> mma;
    mma(a(0), a(1), a(2), a(3),
        b(0), b(1),
        c(0), c(1), c(2), c(3),
        d0, d1, d2, d3);

    c(0) = d0;
    c(1) = d1;
    c(2) = d2;
    c(3) = d3;
}

}
