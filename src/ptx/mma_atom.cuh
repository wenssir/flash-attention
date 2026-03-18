#pragma once

#include "../config/macros.cuh"
#include "./ptx.cuh"

namespace ptx {

struct Sm80MmaF16F16F32M16N8K16 {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;
    static constexpr int regs_a = 4;
    static constexpr int regs_b = 2;
    static constexpr int regs_c = 4;

    template <typename FragA, typename FragB, typename FragC>
    DEVICE static void fma(FragA const& a, FragB const& b, FragC& c) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c(0)), "+f"(c(1)), "+f"(c(2)), "+f"(c(3))
            : "r"(a(0)), "r"(a(1)), "r"(a(2)), "r"(a(3)),
              "r"(b(0)), "r"(b(1))
        );
    }
};

template <typename FragA, typename FragB, typename FragC>
DEVICE void mma_atom(FragA const& a, FragB const& b, FragC& c) {
    Sm80MmaF16F16F32M16N8K16::fma(a, b, c);
}

}
