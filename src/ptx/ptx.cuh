#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "../config/macros.cuh"

namespace ptx {

template<typename T>
struct MMA_M16N8K16_F32 {
    DEVICE void operator()(
        const uint32_t& a0, const uint32_t& a1, const uint32_t& a2, const uint32_t& a3,
        const uint32_t& b0, const uint32_t& b1,
        const float& c0, const float& c1, const float& c2, const float& c3,
        float& d0, float& d1, float& d2, float& d3
    ) const {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3)
        );
    }
};

// for fp8
template<typename T>
struct MMA_M16N8K16_F16 {
    DEVICE void operator()(
        const uint32_t& a0, const uint32_t& a1, const uint32_t& a2, const uint32_t& a3,
        const uint32_t& b0, const uint32_t& b1,
        const float& c0, const float& c1, const float& c2, const float& c3,
        float& d0, float& d1, float& d2, float& d3
    ) const {
        // asm volatile(
        //     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        //     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        //     : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        //     : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        //       "r"(b0), "r"(b1),
        //       "f"(c0), "f"(c1), "f"(c2), "f"(c3)
        // );
    }
};

template<typename T>
struct LDMATRIX_X4 {
    DEVICE void operator()(T* smem, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) const {
        static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "ldmatrix only support fp16/bf16");

        uint32_t smem_int_ptr = __cvta_generic_to_shared(smem);

        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_int_ptr)
        );
    }
};

template<typename T>
struct LDMATRIX_X4_TRANS {
    DEVICE void operator()(T* smem, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) const {
        static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "ldmatrix only support fp16/bf16");

        uint32_t smem_int_ptr = __cvta_generic_to_shared(smem);

        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(smem_int_ptr)
        );
    }
};

template<typename T>
struct LDMATRIX_X2 {
    DEVICE void operator()(T* smem, uint32_t& r0, uint32_t& r1) const {
        static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "ldmatrix only support fp16/bf16");

        uint32_t smem_int_ptr = __cvta_generic_to_shared(smem);

        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
            "{%0, %1}, [%2];"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_int_ptr)
        );
    }
};

template<typename T>
struct LDMATRIX_X1 {
    DEVICE void operator()(T* smem, uint32_t& r0) const {
        static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "ldmatrix only support fp16/bf16");

        uint32_t smem_int_ptr = __cvta_generic_to_shared(smem);

        asm volatile(
            "ldmatrix.sync.aligned.x1.m8n8.shared.b16 "
            "{%0}, [%1];"
            : "=r"(r0)
            : "r"(smem_int_ptr)
        );
    }
};

template<typename T>
struct LDMATRIX_X2_TRANS {
    DEVICE void operator()(T* smem, uint32_t& r0, uint32_t& r1) const {
        static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "ldmatrix only support fp16/bf16");

        uint32_t smem_int_ptr = __cvta_generic_to_shared(smem);

        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
            "{%0, %1}, [%2];"
            : "=r"(r0), "=r"(r1)
            : "r"(smem_int_ptr)
        );
    }
};

}