#pragma once

#include <cassert>

#include "../config/macros.cuh"
#include "../numeric/Int.cuh"

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#  define CUDA_ARCH_SM80
#endif

namespace ptx {

template <class TS, class TD = TS>
struct CP_ASYNC_CACHE_ALWAYS {
    using SRegisters = TS[1];
    using DRegisters = TD[1];

    HOST_DEVICE static void copy(TS const& src, TD& dst) {
#if defined(CUDA_ARCH_SM80)
        TS const* gmem_ptr    = &src;
        uint32_t smem_int_ptr = __cvta_generic_to_shared(&dst);
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2; "
            :: "r"(smem_int_ptr),
               "l"(gmem_ptr),
               "n"(sizeof(TS))
        );
#else
    printf("Cp.async is not enabled");
    assert(0);
#endif
    }
};

template <class TS, class TD = TS>
struct CP_ASYNC_CACHE_GLOBAL {
    using SRegisters = TS[1];
    using DRegisters = TD[1];

    HOST_DEVICE static void copy(TS const& src, TD& dst) {
#if defined(CUDA_ARCH_SM80)
        TS const* gmem_ptr    = &src;
        uint32_t smem_int_ptr = __cvta_generic_to_shared(&dst);
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2; "
            :: "r"(smem_int_ptr),
               "l"(gmem_ptr),
               "n"(sizeof(TS))
        );
#else
    printf("Cp.async is not enabled");
    assert(0);
#endif
    }
};


HOST_DEVICE void cp_async_fence() {
#if defined(CUDA_ARCH_SM80)
    asm volatile(
        "cp.async.commit_group;\n" ::
    );
#endif
}

template <int N>
HOST_DEVICE
void
cp_async_wait(numeric::Int<N>)
{
#if defined(CUDA_ARCH_SM80)
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
  }
#endif
}

};
