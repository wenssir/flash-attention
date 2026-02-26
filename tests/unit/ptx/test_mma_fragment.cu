#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>

#include "src/ptx/ptx.cuh"
#include "src/tensor_core/fragment.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1); \
        } \
    } while (0)

template <typename T, int LD>
struct RowMajorView {
    T* ptr;

    template <typename Coord>
    __device__ T& operator()(Coord const& coord) {
        return ptr[cxx::get<0>(coord) * LD + cxx::get<1>(coord)];
    }
};

using FragA = tensor::Fragment<__half, loadstore::LdmatrixHelperQ<__half>, 16, 16>;
using FragB = tensor::Fragment<__half, loadstore::LdmatrixHelperK<__half>, 16, 8>;

__global__ void mma_fragment_kernel(float* out) {
    __shared__ __align__(16) __half sA[16 * 16];
    __shared__ __align__(16) __half sB[16 * 8];

    constexpr float a_val = 0.5f;
    constexpr float b_val = 0.25f;

    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
        sA[i] = __float2half(a_val);
    }
    for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) {
        sB[i] = __float2half(b_val);
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    int lane_id = threadIdx.x;
    RowMajorView<__half, 16> a_view{sA};
    RowMajorView<__half, 8> b_view{sB};

    FragA frag_a;
    FragB frag_b;
    frag_a.load(a_view, lane_id);
    frag_b.load(b_view, lane_id);

    constexpr float c_init = 1.5f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    ptx::MMA_M16N8K16_F32<__half> mma;
    mma(frag_a(0), frag_a(1), frag_a(2), frag_a(3),
        frag_b(0), frag_b(1),
        c_init, c_init, c_init, c_init,
        d0, d1, d2, d3);

    int base = lane_id * 4;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

__global__ void mma_fragment_kernel_non_integer(float* out) {
    __shared__ __align__(16) __half sA[16 * 16];
    __shared__ __align__(16) __half sB[16 * 8];

    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
        int m = idx / 16;
        int k = idx % 16;
        float a_val = 0.10f * static_cast<float>(m + 1) + 0.01f * static_cast<float>(k + 1) + 0.003f;
        sA[idx] = __float2half(a_val);
    }
    for (int idx = threadIdx.x; idx < 16 * 8; idx += blockDim.x) {
        int k = idx / 8;
        int n = idx % 8;
        float b_val = 0.07f * static_cast<float>(k + 1) - 0.015f * static_cast<float>(n + 1) + 0.004f;
        sB[idx] = __float2half(b_val);
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    int lane_id = threadIdx.x;
    RowMajorView<__half, 16> a_view{sA};
    RowMajorView<__half, 8> b_view{sB};

    FragA frag_a;
    FragB frag_b;
    frag_a.load(a_view, lane_id);
    frag_b.load(b_view, lane_id);

    constexpr float c_init = 0.123f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    ptx::MMA_M16N8K16_F32<__half> mma;
    mma(frag_a(0), frag_a(1), frag_a(2), frag_a(3),
        frag_b(0), frag_b(1),
        c_init, c_init, c_init, c_init,
        d0, d1, d2, d3);

    int base = lane_id * 4;
    out[base + 0] = d0;
    out[base + 1] = d1;
    out[base + 2] = d2;
    out[base + 3] = d3;
}

void accum_lane_to_matrix_16x8(const float* lane_accum, float* matrix) {
    float raw[16 * 8];
    for (int i = 0; i < 16 * 8; ++i) {
        raw[i] = 0.0f;
    }

    // Raw accumulator layout directly mapped from lane fragments.
    for (int lane = 0; lane < 32; ++lane) {
        int quad = lane % 4;
        int col_pair = (lane / 4) % 4;
        int row_group = lane / 16;

        int row0 = row_group * 4 + quad;
        int row1 = row0 + 8;
        int col0 = col_pair * 2;
        int col1 = col0 + 1;

        int base = lane * 4;
        raw[row0 * 8 + col0] = lane_accum[base + 0];
        raw[row0 * 8 + col1] = lane_accum[base + 1];
        raw[row1 * 8 + col0] = lane_accum[base + 2];
        raw[row1 * 8 + col1] = lane_accum[base + 3];
    }

    // Convert raw accumulator layout to logical C(m,n) layout.
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            int raw_r = n / 2 + 4 * (m / 4);
            int raw_c = 2 * (m % 4) + (n % 2);
            matrix[m * 8 + n] = raw[raw_r * 8 + raw_c];
        }
    }
}

void print_matrix_16x8(const float* matrix) {
    printf("Output matrix (16x8):\n");
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            printf("%9.5f ", matrix[m * 8 + n]);
        }
        printf("\n");
    }
}

void make_expected_matrix_non_integer(float* expected) {
    constexpr float c_init = 0.123f;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = c_init;
            for (int k = 0; k < 16; ++k) {
                float a_val = 0.10f * static_cast<float>(m + 1) + 0.01f * static_cast<float>(k + 1) + 0.003f;
                float b_val = 0.07f * static_cast<float>(k + 1) - 0.015f * static_cast<float>(n + 1) + 0.004f;
                float a_q = __half2float(__float2half(a_val));
                float b_q = __half2float(__float2half(b_val));
                acc += a_q * b_q;
            }
            expected[m * 8 + n] = acc;
        }
    }
}

int main() {
    constexpr int kLanes = 32;
    constexpr int kAccPerLane = 4;
    constexpr int kCount = kLanes * kAccPerLane;

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, kCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, kCount * sizeof(float)));

    mma_fragment_kernel<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_out[kCount];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, kCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    float a_q = __half2float(__float2half(0.5f));
    float b_q = __half2float(__float2half(0.25f));
    float expected = 1.5f + 16.0f * (a_q * b_q);

    for (int i = 0; i < kCount; ++i) {
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            printf("FAIL idx=%d got=%.6f expected=%.6f\n", i, h_out[i], expected);
            return 1;
        }
    }

    printf("PASS: fragment+mma A*B+C, lane0=[%.6f %.6f %.6f %.6f], expected=%.6f\n",
           h_out[0], h_out[1], h_out[2], h_out[3], expected);

    CUDA_CHECK(cudaMalloc(&d_out, kCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, kCount * sizeof(float)));

    mma_fragment_kernel_non_integer<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, kCount * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    bool all_same = true;
    for (int i = 1; i < kCount; ++i) {
        if (fabsf(h_out[i] - h_out[0]) > 1e-4f) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
        printf("FAIL: non-integer test output is unexpectedly uniform.\n");
        return 1;
    }

    float matrix[16 * 8];
    accum_lane_to_matrix_16x8(h_out, matrix);
    print_matrix_16x8(matrix);

    float expected_matrix[16 * 8];
    make_expected_matrix_non_integer(expected_matrix);
    printf("Expected matrix (16x8):\n");
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            printf("%9.5f ", expected_matrix[m * 8 + n]);
        }
        printf("\n");
    }

    constexpr float tol = 1e-2f;
    int mismatch_count = 0;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float got = matrix[m * 8 + n];
            float exp = expected_matrix[m * 8 + n];
            if (fabsf(got - exp) > tol) {
                if (mismatch_count < 16) {
                    printf("Mismatch (%d,%d): got=%.6f expected=%.6f\n", m, n, got, exp);
                }
                mismatch_count++;
            }
        }
    }
    if (mismatch_count > 0) {
        printf("FAIL: non-integer matrix mismatch count=%d (tol=%.1e)\n", mismatch_count, tol);
        return 1;
    }

    printf("PASS: non-integer fragment+mma test\n");
    return 0;
}
