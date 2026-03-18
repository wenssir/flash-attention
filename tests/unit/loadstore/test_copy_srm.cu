#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

#include "tensor_core/fragment.cuh"
#include "loadstore/copy_srm.cuh"
#include "loadstore/copy_gsm.cuh"
#include "layout/layout.h"
#include "tensor_core/tensor.cuh"
#include "numeric/Int.cuh"
#include "ptx/ptx.cuh"
#include "config/config.cuh"

using namespace tensor;
using namespace loadstore;
using namespace layout;

namespace {

void accum_lane_to_matrix_16x8_test(const float* lane_accum, float* matrix) {
    float raw[16 * 8];
    for (int i = 0; i < 16 * 8; ++i) {
        raw[i] = 0.0f;
    }

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

    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            int raw_r = n / 2 + 4 * (m / 4);
            int raw_c = 2 * (m % 4) + (n % 2);
            matrix[m * 8 + n] = raw[raw_r * 8 + raw_c];
        }
    }
}

// ============================================================
// Simple test: verify input matrix pointer positions for 16x16
// ============================================================

template <typename T, int Rows, int Cols, typename LdmatrixHelper>
__global__ void test_input_pointer_positions() {
    using namespace layout;

    int lane_id = threadIdx.x % 32;

    // Declare shared memory (16x16 = 256 elements)
    __shared__ T smem_buffer[256];

    // Create tensor for shared memory
    auto smem_layout = make_layout(
        make_shape(numeric::Int<Rows>{}, numeric::Int<Cols>{}),
        make_stride(numeric::Int<Cols>{}, numeric::Int<1>{})
    );
    auto smem_tensor = make_tensor(smem_buffer, smem_layout);

    // Get the coordinate for this lane
    auto coord = LdmatrixHelper::coord(lane_id);

    // Calculate the pointer position
    T* ptr = &smem_tensor(coord);

    // Print: lane_id -> (row, col) -> pointer_offset
    if (lane_id < 32) {
        int row = cxx::get<0>(coord);
        int col = cxx::get<1>(coord);
        int offset = static_cast<int>(ptr - smem_buffer);

        printf("Lane %2d: coord=(%2d, %2d), offset=%3d\n",
               lane_id, row, col, offset);
    }
}

TEST(LdmatrixTest, InputPointerPositions_16x16) {
    printf("\n=== Testing 16x16 LDMATRIX_X4 Input Pointer Positions ===\n\n");

    test_input_pointer_positions<__half, 16, 16, tensor::LdmatrixHelperQ<__half>><<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n=== Expected Pattern ===\n");
    printf("Each 8-thread group should point at one 8x8 tile\n");
    printf("For 16x16: lanes 0-7 -> top-left, 8-15 -> bottom-left,\n");
    printf("            16-23 -> top-right, 24-31 -> bottom-right\n");
    printf("\n");
}

__global__ void test_helper_q_coords_kernel(int* out) {
    if (threadIdx.x != 0) {
        return;
    }
    auto c0 = tensor::LdmatrixHelperQ<__half>::coord(0);
    auto c7 = tensor::LdmatrixHelperQ<__half>::coord(7);
    auto c8 = tensor::LdmatrixHelperQ<__half>::coord(8);
    auto c15 = tensor::LdmatrixHelperQ<__half>::coord(15);
    auto c16 = tensor::LdmatrixHelperQ<__half>::coord(16);
    auto c23 = tensor::LdmatrixHelperQ<__half>::coord(23);
    auto c24 = tensor::LdmatrixHelperQ<__half>::coord(24);
    auto c31 = tensor::LdmatrixHelperQ<__half>::coord(31);

    int vals[16] = {
        static_cast<int>(cxx::get<0>(c0)), static_cast<int>(cxx::get<1>(c0)),
        static_cast<int>(cxx::get<0>(c7)), static_cast<int>(cxx::get<1>(c7)),
        static_cast<int>(cxx::get<0>(c8)), static_cast<int>(cxx::get<1>(c8)),
        static_cast<int>(cxx::get<0>(c15)), static_cast<int>(cxx::get<1>(c15)),
        static_cast<int>(cxx::get<0>(c16)), static_cast<int>(cxx::get<1>(c16)),
        static_cast<int>(cxx::get<0>(c23)), static_cast<int>(cxx::get<1>(c23)),
        static_cast<int>(cxx::get<0>(c24)), static_cast<int>(cxx::get<1>(c24)),
        static_cast<int>(cxx::get<0>(c31)), static_cast<int>(cxx::get<1>(c31)),
    };
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = vals[i];
    }
}

TEST(LdmatrixTest, HelperQUsesRowStartsFor16x16) {
    int *d_out = nullptr;
    int h_out[16] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_helper_q_coords_kernel<<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 0);  EXPECT_EQ(h_out[1], 0);
    EXPECT_EQ(h_out[2], 7);  EXPECT_EQ(h_out[3], 0);
    EXPECT_EQ(h_out[4], 8);  EXPECT_EQ(h_out[5], 0);
    EXPECT_EQ(h_out[6], 15); EXPECT_EQ(h_out[7], 0);
    EXPECT_EQ(h_out[8], 0);  EXPECT_EQ(h_out[9], 8);
    EXPECT_EQ(h_out[10], 7); EXPECT_EQ(h_out[11], 8);
    EXPECT_EQ(h_out[12], 8); EXPECT_EQ(h_out[13], 8);
    EXPECT_EQ(h_out[14], 15); EXPECT_EQ(h_out[15], 8);
}

template <typename KernelConfig>
__global__ void test_q_identity_retile_starts_kernel(int* out) {
    if (threadIdx.x != 0) {
        return;
    }

    using namespace numeric;
    using Element = typename KernelConfig::Element;
    float* storage = nullptr;

    auto atom_swizzle = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sQ = make_tensor(storage,
        tile_to_shape(atom_swizzle, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));

    auto q_view = retile_D<tensor::LdmatrixHelperQ<Element>>(partition_fragment_A<KernelConfig>(sQ, /*warp=*/1, /*k_tile=*/2));

    int lanes[6] = {0, 7, 15, 16, 23, 31};
    for (int i = 0; i < 6; ++i) {
        int lane = lanes[i];
        auto coord = tensor::LdmatrixHelperQ<Element>::coord(lane);
        int local_row = static_cast<int>(cxx::get<0>(coord));
        int local_col = static_cast<int>(cxx::get<1>(coord));
        int global_row = 16 + local_row;
        int global_col = 32 + local_col;
        int base = i * 2;
        out[base + 0] = global_row;
        out[base + 1] = global_col;
        (void)q_view.offset(coord);
    }
}

TEST(CopySrmTest, QIdentityRetileStartsMatchMmaImage) {
    using KernelConfig = config::ForwardConfig;
    int *d_out = nullptr;
    int h_out[12] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_q_identity_retile_starts_kernel<KernelConfig><<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 16); EXPECT_EQ(h_out[1], 32); // lane 0
    EXPECT_EQ(h_out[2], 23); EXPECT_EQ(h_out[3], 32); // lane 7
    EXPECT_EQ(h_out[4], 31); EXPECT_EQ(h_out[5], 32); // lane 15
    EXPECT_EQ(h_out[6], 16); EXPECT_EQ(h_out[7], 40); // lane 16
    EXPECT_EQ(h_out[8], 23); EXPECT_EQ(h_out[9], 40); // lane 23
    EXPECT_EQ(h_out[10], 31); EXPECT_EQ(h_out[11], 40); // lane 31
}

template <typename KernelConfig>
__global__ void test_partition_fragment_q_kernel(int* out_linear, int* out_swizzle) {
    if (threadIdx.x != 0) {
        return;
    }

    using namespace numeric;
    float* storage = nullptr;

    auto atom_linear = composition(
        tensor::NoSwizzle{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto atom_swizzle = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));

    auto s_linear = make_tensor(storage,
        tile_to_shape(atom_linear, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));
    auto s_swizzle = make_tensor(storage,
        tile_to_shape(atom_swizzle, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));

    auto q_linear = partition_fragment_A<KernelConfig>(s_linear, /*warp=*/1, /*k_tile=*/2);
    auto q_swizzle = partition_fragment_A<KernelConfig>(s_swizzle, /*warp=*/1, /*k_tile=*/2);

    out_linear[0] = static_cast<int>(q_linear.offset(make_coordinate(0, 0)));
    out_linear[1] = static_cast<int>(q_linear.offset(make_coordinate(0, 8)));
    out_linear[2] = static_cast<int>(q_linear.offset(make_coordinate(9, 0)));
    out_linear[3] = static_cast<int>(q_linear.offset(make_coordinate(15, 15)));

    out_swizzle[0] = static_cast<int>(q_swizzle.offset(make_coordinate(0, 0)));
    out_swizzle[1] = static_cast<int>(q_swizzle.offset(make_coordinate(0, 8)));
    out_swizzle[2] = static_cast<int>(q_swizzle.offset(make_coordinate(9, 0)));
    out_swizzle[3] = static_cast<int>(q_swizzle.offset(make_coordinate(15, 15)));
}

template <typename KernelConfig>
__global__ void test_partition_fragment_k_kernel(int* out_linear, int* out_swizzle) {
    if (threadIdx.x != 0) {
        return;
    }

    using namespace numeric;
    float* storage = nullptr;

    auto atom_linear = composition(
        tensor::NoSwizzle{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto atom_swizzle = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));

    auto s_linear = make_tensor(storage,
        tile_to_shape(atom_linear, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));
    auto s_swizzle = make_tensor(storage,
        tile_to_shape(atom_swizzle, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));

    auto k_linear = partition_fragment_B<KernelConfig>(s_linear, /*out_tile=*/3, /*k_tile=*/2);
    auto k_swizzle = partition_fragment_B<KernelConfig>(s_swizzle, /*out_tile=*/3, /*k_tile=*/2);

    out_linear[0] = static_cast<int>(k_linear.offset(make_coordinate(0, 0)));
    out_linear[1] = static_cast<int>(k_linear.offset(make_coordinate(0, 7)));
    out_linear[2] = static_cast<int>(k_linear.offset(make_coordinate(8, 0)));
    out_linear[3] = static_cast<int>(k_linear.offset(make_coordinate(15, 7)));

    out_swizzle[0] = static_cast<int>(k_swizzle.offset(make_coordinate(0, 0)));
    out_swizzle[1] = static_cast<int>(k_swizzle.offset(make_coordinate(0, 7)));
    out_swizzle[2] = static_cast<int>(k_swizzle.offset(make_coordinate(8, 0)));
    out_swizzle[3] = static_cast<int>(k_swizzle.offset(make_coordinate(15, 7)));
}

template <typename KernelConfig>
__global__ void test_partition_fragment_v_kernel(int* out_linear, int* out_swizzle) {
    if (threadIdx.x != 0) {
        return;
    }

    using namespace numeric;
    float* storage = nullptr;

    auto atom_linear = composition(
        tensor::NoSwizzle{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto atom_swizzle = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));

    auto s_linear = make_tensor(storage,
        tile_to_shape(atom_linear, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));
    auto s_swizzle = make_tensor(storage,
        tile_to_shape(atom_swizzle, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));

    auto v_linear = partition_fragment_V<KernelConfig>(s_linear, /*k_tile=*/2, /*out_tile=*/3);
    auto v_swizzle = partition_fragment_V<KernelConfig>(s_swizzle, /*k_tile=*/2, /*out_tile=*/3);

    out_linear[0] = static_cast<int>(v_linear.offset(make_coordinate(0, 0)));
    out_linear[1] = static_cast<int>(v_linear.offset(make_coordinate(0, 7)));
    out_linear[2] = static_cast<int>(v_linear.offset(make_coordinate(8, 0)));
    out_linear[3] = static_cast<int>(v_linear.offset(make_coordinate(15, 7)));

    out_swizzle[0] = static_cast<int>(v_swizzle.offset(make_coordinate(0, 0)));
    out_swizzle[1] = static_cast<int>(v_swizzle.offset(make_coordinate(0, 7)));
    out_swizzle[2] = static_cast<int>(v_swizzle.offset(make_coordinate(8, 0)));
    out_swizzle[3] = static_cast<int>(v_swizzle.offset(make_coordinate(15, 7)));
}

TEST(CopySrmTest, PartitionFragmentQOffsets) {
    using KernelConfig = config::ForwardConfig;

    int *d_linear = nullptr, *d_swizzle = nullptr;
    int h_linear[4] = {}, h_swizzle[4] = {};
    ASSERT_EQ(cudaMalloc(&d_linear, sizeof(h_linear)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_swizzle, sizeof(h_swizzle)), cudaSuccess);

    test_partition_fragment_q_kernel<KernelConfig><<<1, 1>>>(d_linear, d_swizzle);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_linear, d_linear, sizeof(h_linear), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_swizzle, d_swizzle, sizeof(h_swizzle), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_linear), cudaSuccess);
    ASSERT_EQ(cudaFree(d_swizzle), cudaSuccess);

    EXPECT_EQ(h_linear[0], 16 * 128 + 32);
    EXPECT_EQ(h_linear[1], 16 * 128 + 40);
    EXPECT_EQ(h_linear[2], (25 / 8 * 2 + 0) * 512 + (25 % 8) * 64 + 32);
    EXPECT_EQ(h_linear[3], (31 / 8 * 2 + 0) * 512 + (31 % 8) * 64 + 47);

    EXPECT_EQ(h_swizzle[0], h_linear[0]);
    EXPECT_EQ(h_swizzle[1], h_linear[1]);
    EXPECT_NE(h_swizzle[2], h_linear[2]);
    EXPECT_NE(h_swizzle[3], h_linear[3]);
}

TEST(CopySrmTest, PartitionFragmentKOffsets) {
    using KernelConfig = config::ForwardConfig;

    int *d_linear = nullptr, *d_swizzle = nullptr;
    int h_linear[4] = {}, h_swizzle[4] = {};
    ASSERT_EQ(cudaMalloc(&d_linear, sizeof(h_linear)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_swizzle, sizeof(h_swizzle)), cudaSuccess);

    test_partition_fragment_k_kernel<KernelConfig><<<1, 1>>>(d_linear, d_swizzle);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_linear, d_linear, sizeof(h_linear), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_swizzle, d_swizzle, sizeof(h_swizzle), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_linear), cudaSuccess);
    ASSERT_EQ(cudaFree(d_swizzle), cudaSuccess);

    EXPECT_EQ(h_linear[0], 24 * 128 + 32);
    EXPECT_EQ(h_linear[1], 24 * 128 + 39);
    EXPECT_EQ(h_linear[2], (32 / 8 * 2 + 0) * 512 + (32 % 8) * 64 + 32);
    EXPECT_EQ(h_linear[3], (39 / 8 * 2 + 0) * 512 + (39 % 8) * 64 + 39);

    EXPECT_EQ(h_swizzle[0], h_linear[0]);
    EXPECT_EQ(h_swizzle[1], h_linear[1]);
    EXPECT_EQ(h_swizzle[2], h_linear[2]);
    EXPECT_NE(h_swizzle[3], h_linear[3]);
}

TEST(CopySrmTest, PartitionFragmentVOffsets) {
    using KernelConfig = config::ForwardConfig;

    int *d_linear = nullptr, *d_swizzle = nullptr;
    int h_linear[4] = {}, h_swizzle[4] = {};
    ASSERT_EQ(cudaMalloc(&d_linear, sizeof(h_linear)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_swizzle, sizeof(h_swizzle)), cudaSuccess);

    test_partition_fragment_v_kernel<KernelConfig><<<1, 1>>>(d_linear, d_swizzle);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_linear, d_linear, sizeof(h_linear), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_swizzle, d_swizzle, sizeof(h_swizzle), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_linear), cudaSuccess);
    ASSERT_EQ(cudaFree(d_swizzle), cudaSuccess);

    EXPECT_EQ(h_linear[0], 32 * 128 + 24);
    EXPECT_EQ(h_linear[1], 32 * 128 + 31);
    EXPECT_EQ(h_linear[2], 40 * 128 + 24);
    EXPECT_EQ(h_linear[3], (47 / 8 * 2 + 0) * 512 + (47 % 8) * 64 + 31);

    EXPECT_EQ(h_swizzle[0], h_linear[0]);
    EXPECT_EQ(h_swizzle[1], h_linear[1]);
    EXPECT_EQ(h_swizzle[2], h_linear[2]);
    EXPECT_NE(h_swizzle[3], h_linear[3]);
}

template <typename KernelConfig>
__global__ void test_retile_q_load_kernel(uint32_t* out_regs) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragQ = typename KernelConfig::FragQ;

    __shared__ Element smem[KernelConfig::BlockM * KernelConfig::HeadDim];

    for (int idx = threadIdx.x; idx < KernelConfig::BlockM * KernelConfig::HeadDim; idx += blockDim.x) {
        reinterpret_cast<unsigned short*>(smem)[idx] = static_cast<unsigned short>(idx);
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto s_tensor = make_tensor(
        smem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));

    auto frag_view = partition_fragment_A<KernelConfig>(s_tensor, /*warp=*/1, /*k_tile=*/2);
    auto ld_view = retile_D<tensor::LdmatrixHelperQ<Element>>(frag_view);

    FragQ frag;
    frag.load(ld_view, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragQ::REGS_PER_LANE; ++i) {
        out_regs[lane * FragQ::REGS_PER_LANE + i] = frag(i);
    }
}

template <typename KernelConfig>
__global__ void test_retile_k_load_kernel(uint32_t* out_regs) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragK = typename KernelConfig::FragK;

    __shared__ Element smem[KernelConfig::BlockN * KernelConfig::HeadDim];

    for (int idx = threadIdx.x; idx < KernelConfig::BlockN * KernelConfig::HeadDim; idx += blockDim.x) {
        reinterpret_cast<unsigned short*>(smem)[idx] = static_cast<unsigned short>(idx);
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto s_tensor = make_tensor(
        smem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));

    auto frag_view = partition_fragment_B<KernelConfig>(s_tensor, /*out_tile=*/3, /*k_tile=*/2);
    auto ld_view = retile_D<tensor::LdmatrixHelperK<Element>>(frag_view);

    FragK frag;
    frag.load(ld_view, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragK::REGS_PER_LANE; ++i) {
        out_regs[lane * FragK::REGS_PER_LANE + i] = frag(i);
    }
}

template <typename KernelConfig>
__global__ void test_g2s_s2r_q_kernel(const typename KernelConfig::Element* gmem, uint32_t* out_regs, bool baseline_fill) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragQ = typename KernelConfig::FragQ;

    __shared__ Element smem[KernelConfig::BlockM * KernelConfig::HeadDim];

    auto g_layout = make_layout(
        make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto g_tensor = make_tensor(gmem, g_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto s_tensor = make_tensor(
        smem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));

    if (!baseline_fill) {
        copy_g2s_q<KernelConfig>(g_tensor, s_tensor);
    } else {
        for (int idx = threadIdx.x; idx < KernelConfig::BlockM * KernelConfig::HeadDim; idx += blockDim.x) {
            int row = idx / KernelConfig::HeadDim;
            int col = idx % KernelConfig::HeadDim;
            s_tensor(loadstore::smem_coord<KernelConfig>(row, col)) = g_tensor(make_coordinate(row, col));
        }
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto frag_view = partition_fragment_A<KernelConfig>(s_tensor, /*warp=*/1, /*k_tile=*/2);
    auto ld_view = retile_D<tensor::LdmatrixHelperQ<Element>>(frag_view);

    FragQ frag;
    frag.load(ld_view, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragQ::REGS_PER_LANE; ++i) {
        out_regs[lane * FragQ::REGS_PER_LANE + i] = frag(i);
    }
}

template <typename KernelConfig>
__global__ void test_g2s_s2r_k_kernel(const typename KernelConfig::Element* gmem, uint32_t* out_regs, bool baseline_fill) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragK = typename KernelConfig::FragK;

    __shared__ Element smem[KernelConfig::BlockN * KernelConfig::HeadDim];

    auto g_layout = make_layout(
        make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto g_tensor = make_tensor(gmem, g_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto s_tensor = make_tensor(
        smem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));

    if (!baseline_fill) {
        copy_g2s_kv<KernelConfig>(g_tensor, s_tensor);
    } else {
        for (int idx = threadIdx.x; idx < KernelConfig::BlockN * KernelConfig::HeadDim; idx += blockDim.x) {
            int row = idx / KernelConfig::HeadDim;
            int col = idx % KernelConfig::HeadDim;
            s_tensor(loadstore::smem_coord<KernelConfig>(row, col)) = g_tensor(make_coordinate(row, col));
        }
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto frag_view = partition_fragment_B<KernelConfig>(s_tensor, /*out_tile=*/3, /*k_tile=*/2);
    auto ld_view = retile_D<tensor::LdmatrixHelperK<Element>>(frag_view);

    FragK frag;
    frag.load(ld_view, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragK::REGS_PER_LANE; ++i) {
        out_regs[lane * FragK::REGS_PER_LANE + i] = frag(i);
    }
}

template <typename KernelConfig>
__global__ void test_g2s_s2r_v_kernel(const typename KernelConfig::Element* gmem, uint32_t* out_regs, bool baseline_fill) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragV = typename KernelConfig::FragV;

    __shared__ Element smem[KernelConfig::BlockN * KernelConfig::HeadDim];

    auto g_layout = make_layout(
        make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto g_tensor = make_tensor(gmem, g_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto s_tensor = make_tensor(
        smem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));

    if (!baseline_fill) {
        copy_g2s_kv<KernelConfig>(g_tensor, s_tensor);
    } else {
        for (int idx = threadIdx.x; idx < KernelConfig::BlockN * KernelConfig::HeadDim; idx += blockDim.x) {
            int row = idx / KernelConfig::HeadDim;
            int col = idx % KernelConfig::HeadDim;
            s_tensor(loadstore::smem_coord<KernelConfig>(row, col)) = g_tensor(make_coordinate(row, col));
        }
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto frag_view = partition_fragment_V<KernelConfig>(s_tensor, /*k_tile=*/2, /*out_tile=*/3);
    auto ld_view = retile_D<tensor::LdmatrixHelperK<Element>>(frag_view);

    FragV frag;
    frag.load(ld_view, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragV::REGS_PER_LANE; ++i) {
        out_regs[lane * FragV::REGS_PER_LANE + i] = frag(i);
    }
}

template <typename KernelConfig>
__global__ void test_q_frag_load_matches_dense_kernel(
    const typename KernelConfig::Element* q_gmem,
    uint32_t* out_pipe,
    uint32_t* out_dense) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragQ = typename KernelConfig::FragQ;

    __shared__ Element sQMem[KernelConfig::BlockM * KernelConfig::HeadDim];
    __shared__ Element denseMem[16 * 16];

    auto gq_layout = make_layout(
        make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto gQ = make_tensor(q_gmem, gq_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sQ = make_tensor(
        sQMem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));
    auto dense = make_tensor(
        denseMem,
        make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));

    copy_g2s_q<KernelConfig>(gQ, sQ);
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int col = idx % 16;
        dense(make_coordinate(row, col)) = gQ(make_coordinate(16 + row, 32 + col));
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto q_view = retile_D<tensor::LdmatrixHelperQ<Element>>(partition_fragment_A<KernelConfig>(sQ, /*warp=*/1, /*k_tile=*/2));
    FragQ frag_pipe{};
    FragQ frag_dense{};
    frag_pipe.load(q_view, threadIdx.x);
    frag_dense.load(dense, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragQ::REGS_PER_LANE; ++i) {
        out_pipe[lane * FragQ::REGS_PER_LANE + i] = frag_pipe(i);
        out_dense[lane * FragQ::REGS_PER_LANE + i] = frag_dense(i);
    }
}

template <typename KernelConfig>
__global__ void test_q_frag_load_matches_dense_origin_kernel(
    const typename KernelConfig::Element* q_gmem,
    uint32_t* out_pipe,
    uint32_t* out_dense) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragQ = typename KernelConfig::FragQ;

    __shared__ Element sQMem[KernelConfig::BlockM * KernelConfig::HeadDim];
    __shared__ Element denseMem[16 * 16];

    auto gq_layout = make_layout(
        make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto gQ = make_tensor(q_gmem, gq_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sQ = make_tensor(
        sQMem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));
    auto dense = make_tensor(
        denseMem,
        make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));

    copy_g2s_q<KernelConfig>(gQ, sQ);
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int col = idx % 16;
        dense(make_coordinate(row, col)) = gQ(make_coordinate(row, col));
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto q_view = retile_D<tensor::LdmatrixHelperQ<Element>>(partition_fragment_A<KernelConfig>(sQ, /*warp=*/0, /*k_tile=*/0));
    FragQ frag_pipe{};
    FragQ frag_dense{};
    frag_pipe.load(q_view, threadIdx.x);
    frag_dense.load(dense, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragQ::REGS_PER_LANE; ++i) {
        out_pipe[lane * FragQ::REGS_PER_LANE + i] = frag_pipe(i);
        out_dense[lane * FragQ::REGS_PER_LANE + i] = frag_dense(i);
    }
}

template <typename KernelConfig>
__global__ void test_k_frag_load_matches_dense_kernel(
    const typename KernelConfig::Element* k_gmem,
    uint32_t* out_pipe,
    uint32_t* out_dense) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragK = typename KernelConfig::FragK;

    __shared__ Element sKMem[KernelConfig::BlockN * KernelConfig::HeadDim];
    __shared__ Element denseMem[16 * 8];

    auto gk_layout = make_layout(
        make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto gK = make_tensor(k_gmem, gk_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sK = make_tensor(
        sKMem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));
    auto dense = make_tensor(
        denseMem,
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));

    copy_g2s_kv<KernelConfig>(gK, sK);
    for (int idx = threadIdx.x; idx < 16 * 8; idx += blockDim.x) {
        int row = idx / 8;
        int col = idx % 8;
        dense(make_coordinate(row, col)) = gK(make_coordinate(24 + row, 32 + col));
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto k_view = retile_D<tensor::LdmatrixHelperK<Element>>(partition_fragment_B<KernelConfig>(sK, /*out_tile=*/3, /*k_tile=*/2));
    FragK frag_pipe{};
    FragK frag_dense{};
    frag_pipe.load(k_view, threadIdx.x);
    frag_dense.load(dense, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragK::REGS_PER_LANE; ++i) {
        out_pipe[lane * FragK::REGS_PER_LANE + i] = frag_pipe(i);
        out_dense[lane * FragK::REGS_PER_LANE + i] = frag_dense(i);
    }
}

template <typename KernelConfig>
__global__ void test_v_frag_load_matches_dense_kernel(
    const typename KernelConfig::Element* v_gmem,
    uint32_t* out_pipe,
    uint32_t* out_dense) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragV = typename KernelConfig::FragV;

    __shared__ Element sVMem[KernelConfig::BlockN * KernelConfig::HeadDim];
    __shared__ Element denseMem[16 * 8];

    auto gv_layout = make_layout(
        make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto gV = make_tensor(v_gmem, gv_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sV = make_tensor(
        sVMem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));
    auto dense = make_tensor(
        denseMem,
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));

    copy_g2s_kv<KernelConfig>(gV, sV);
    for (int idx = threadIdx.x; idx < 16 * 8; idx += blockDim.x) {
        int row = idx / 8;
        int col = idx % 8;
        dense(make_coordinate(row, col)) = gV(make_coordinate(32 + row, 24 + col));
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto v_view = retile_D<tensor::LdmatrixHelperK<Element>>(partition_fragment_V<KernelConfig>(sV, /*k_tile=*/2, /*out_tile=*/3));
    FragV frag_pipe{};
    FragV frag_dense{};
    frag_pipe.load(v_view, threadIdx.x);
    frag_dense.load(dense, threadIdx.x);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < FragV::REGS_PER_LANE; ++i) {
        out_pipe[lane * FragV::REGS_PER_LANE + i] = frag_pipe(i);
        out_dense[lane * FragV::REGS_PER_LANE + i] = frag_dense(i);
    }
}

template <typename KernelConfig>
__global__ void test_pv_mma_pipeline_v_kernel(
    const typename KernelConfig::Element* v_gmem,
    float* out_lane_accum) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using FragP = typename KernelConfig::FragP;
    using FragV = typename KernelConfig::FragV;
    using FragAcc = typename KernelConfig::FragAcc;

    __shared__ Element sVMem[KernelConfig::BlockN * KernelConfig::HeadDim];
    __shared__ Element densePMem[16 * 16];

    auto gv_layout = make_layout(
        make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{}),
        make_stride(Int<KernelConfig::HeadDim>{}, Int<1>{}));
    auto gV = make_tensor(v_gmem, gv_layout);

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sV = make_tensor(
        sVMem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockN>{}, Int<KernelConfig::HeadDim>{})));
    auto denseP = make_tensor(
        densePMem,
        make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));

    copy_g2s_kv<KernelConfig>(gV, sV);
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int col = idx % 16;
        float p = 0.02f * static_cast<float>(row + 1) - 0.01f * static_cast<float>(col + 1) + 0.003f;
        denseP(make_coordinate(row, col)) = __float2half(p);
    }
    __syncthreads();

    if (threadIdx.x >= 32) {
        return;
    }

    auto v_view = retile_D<tensor::LdmatrixHelperK<Element>>(partition_fragment_V<KernelConfig>(sV, /*k_tile=*/2, /*out_tile=*/3));

    FragP p_frag{};
    FragV v_frag{};
    FragAcc acc{};
    acc.clear();
    p_frag.load(denseP, threadIdx.x);
    v_frag.load(v_view, threadIdx.x);
    KernelConfig::Atom::fma(p_frag, v_frag, acc);

    int lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out_lane_accum[lane * 4 + i] = acc(i);
    }
}

template <typename KernelConfig>
__global__ void test_q_lane_mapping_kernel(int* out) {
    using namespace numeric;
    using Element = typename KernelConfig::Element;
    using Helper = tensor::LdmatrixHelperQ<Element>;

    __shared__ Element sQMem[KernelConfig::BlockM * KernelConfig::HeadDim];

    auto atom = composition(
        tensor::Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<KernelConfig::SmemAtomRows>{}, Int<KernelConfig::SmemAtomCols>{}),
                    make_stride(Int<KernelConfig::SmemAtomCols>{}, Int<1>{})));
    auto sQ = make_tensor(
        sQMem,
        tile_to_shape(atom, make_shape(Int<KernelConfig::BlockM>{}, Int<KernelConfig::HeadDim>{})));

    auto q_view = retile_D<Helper>(partition_fragment_A<KernelConfig>(sQ, /*warp=*/1, /*k_tile=*/2));

    int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

    auto coord = Helper::coord(lane);
    int local_row = static_cast<int>(cxx::get<0>(coord));
    int local_col = static_cast<int>(cxx::get<1>(coord));
    int global_row = 16 + local_row;
    int global_col = 32 + local_col;

    auto hier = loadstore::smem_coord<KernelConfig>(global_row, global_col);
    int inner_row = static_cast<int>(cxx::get<0>(cxx::get<0>(hier)));
    int tile_row  = static_cast<int>(cxx::get<1>(cxx::get<0>(hier)));
    int inner_col = static_cast<int>(cxx::get<0>(cxx::get<1>(hier)));
    int tile_col  = static_cast<int>(cxx::get<1>(cxx::get<1>(hier)));
    int offset = static_cast<int>(q_view.offset(coord));

    int base = lane * 7;
    out[base + 0] = global_row;
    out[base + 1] = global_col;
    out[base + 2] = tile_row;
    out[base + 3] = inner_row;
    out[base + 4] = tile_col;
    out[base + 5] = inner_col;
    out[base + 6] = offset;
}

TEST(CopySrmTest, RetileQLoadsNonZeroRegisters) {
    using KernelConfig = config::ForwardConfig;
    using FragQ = typename KernelConfig::FragQ;

    constexpr int kCount = 32 * FragQ::REGS_PER_LANE;
    uint32_t *d_regs = nullptr;
    uint32_t h_regs[kCount] = {};
    ASSERT_EQ(cudaMalloc(&d_regs, sizeof(h_regs)), cudaSuccess);

    test_retile_q_load_kernel<KernelConfig><<<1, 32>>>(d_regs);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_regs, d_regs, sizeof(h_regs), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_regs), cudaSuccess);

    bool any_nonzero = false;
    for (uint32_t v : h_regs) {
        any_nonzero |= (v != 0);
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(CopySrmTest, RetileKLoadsNonZeroRegisters) {
    using KernelConfig = config::ForwardConfig;
    using FragK = typename KernelConfig::FragK;

    constexpr int kCount = 32 * FragK::REGS_PER_LANE;
    uint32_t *d_regs = nullptr;
    uint32_t h_regs[kCount] = {};
    ASSERT_EQ(cudaMalloc(&d_regs, sizeof(h_regs)), cudaSuccess);

    test_retile_k_load_kernel<KernelConfig><<<1, 32>>>(d_regs);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_regs, d_regs, sizeof(h_regs), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_regs), cudaSuccess);

    bool any_nonzero = false;
    for (uint32_t v : h_regs) {
        any_nonzero |= (v != 0);
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(CopySrmTest, G2SThenS2R_QMatchesBaselineFill) {
    using KernelConfig = config::ForwardConfig;
    using FragQ = typename KernelConfig::FragQ;
    using Element = typename KernelConfig::Element;

    constexpr int kElems = KernelConfig::BlockM * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragQ::REGS_PER_LANE;
    Element* d_gmem = nullptr;
    uint32_t *d_pipe = nullptr, *d_base = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_base[kRegs] = {};
    std::vector<unsigned short> h_init(kElems);
    for (int i = 0; i < kElems; ++i) {
        h_init[i] = static_cast<unsigned short>(i);
    }

    ASSERT_EQ(cudaMalloc(&d_gmem, kElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_base, sizeof(h_base)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gmem, h_init.data(), kElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_g2s_s2r_q_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_gmem, d_pipe, false);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    test_g2s_s2r_q_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_gmem, d_base, true);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_base, d_base, sizeof(h_base), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_gmem), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_base), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_base[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, G2SThenS2R_KMatchesBaselineFill) {
    using KernelConfig = config::ForwardConfig;
    using FragK = typename KernelConfig::FragK;
    using Element = typename KernelConfig::Element;

    constexpr int kElems = KernelConfig::BlockN * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragK::REGS_PER_LANE;
    Element* d_gmem = nullptr;
    uint32_t *d_pipe = nullptr, *d_base = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_base[kRegs] = {};
    std::vector<unsigned short> h_init(kElems);
    for (int i = 0; i < kElems; ++i) {
        h_init[i] = static_cast<unsigned short>(i);
    }

    ASSERT_EQ(cudaMalloc(&d_gmem, kElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_base, sizeof(h_base)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gmem, h_init.data(), kElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_g2s_s2r_k_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_gmem, d_pipe, false);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    test_g2s_s2r_k_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_gmem, d_base, true);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_base, d_base, sizeof(h_base), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_gmem), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_base), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_base[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, G2SThenS2R_VMatchesBaselineFill) {
    using KernelConfig = config::ForwardConfig;
    using FragV = typename KernelConfig::FragV;
    using Element = typename KernelConfig::Element;

    constexpr int kElems = KernelConfig::BlockN * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragV::REGS_PER_LANE;
    Element* d_gmem = nullptr;
    uint32_t *d_pipe = nullptr, *d_base = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_base[kRegs] = {};
    std::vector<unsigned short> h_init(kElems);
    for (int i = 0; i < kElems; ++i) {
        h_init[i] = static_cast<unsigned short>(i);
    }

    ASSERT_EQ(cudaMalloc(&d_gmem, kElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_base, sizeof(h_base)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gmem, h_init.data(), kElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_g2s_s2r_v_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_gmem, d_pipe, false);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    test_g2s_s2r_v_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_gmem, d_base, true);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_base, d_base, sizeof(h_base), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_gmem), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_base), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_base[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, QFragLoadMatchesDenseBaseline) {
    using KernelConfig = config::ForwardConfig;
    using Element = typename KernelConfig::Element;
    using FragQ = typename KernelConfig::FragQ;
    constexpr int kQElems = KernelConfig::BlockM * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragQ::REGS_PER_LANE;

    std::vector<unsigned short> h_q(kQElems);
    for (int idx = 0; idx < kQElems; ++idx) {
        int row = idx / KernelConfig::HeadDim;
        int col = idx % KernelConfig::HeadDim;
        h_q[idx] = static_cast<unsigned short>((row << 8) | col);
    }

    Element* d_q = nullptr;
    uint32_t *d_pipe = nullptr, *d_dense = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_dense[kRegs] = {};
    ASSERT_EQ(cudaMalloc(&d_q, kQElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dense, sizeof(h_dense)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_q, h_q.data(), kQElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_q_frag_load_matches_dense_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_q, d_pipe, d_dense);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dense, d_dense, sizeof(h_dense), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_q), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dense), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_dense[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, QFragLoadMatchesDenseBaseline_OriginTile) {
    using KernelConfig = config::ForwardConfig;
    using Element = typename KernelConfig::Element;
    using FragQ = typename KernelConfig::FragQ;
    constexpr int kQElems = KernelConfig::BlockM * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragQ::REGS_PER_LANE;

    std::vector<unsigned short> h_q(kQElems);
    for (int idx = 0; idx < kQElems; ++idx) {
        int row = idx / KernelConfig::HeadDim;
        int col = idx % KernelConfig::HeadDim;
        h_q[idx] = static_cast<unsigned short>((row << 8) | col);
    }

    Element* d_q = nullptr;
    uint32_t *d_pipe = nullptr, *d_dense = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_dense[kRegs] = {};
    ASSERT_EQ(cudaMalloc(&d_q, kQElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dense, sizeof(h_dense)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_q, h_q.data(), kQElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_q_frag_load_matches_dense_origin_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_q, d_pipe, d_dense);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dense, d_dense, sizeof(h_dense), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_q), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dense), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_dense[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, KFragLoadMatchesDenseBaseline) {
    using KernelConfig = config::ForwardConfig;
    using Element = typename KernelConfig::Element;
    using FragK = typename KernelConfig::FragK;
    constexpr int kKElems = KernelConfig::BlockN * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragK::REGS_PER_LANE;

    std::vector<unsigned short> h_k(kKElems);
    for (int idx = 0; idx < kKElems; ++idx) {
        int row = idx / KernelConfig::HeadDim;
        int col = idx % KernelConfig::HeadDim;
        h_k[idx] = static_cast<unsigned short>((row << 8) | col);
    }

    Element* d_k = nullptr;
    uint32_t *d_pipe = nullptr, *d_dense = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_dense[kRegs] = {};
    ASSERT_EQ(cudaMalloc(&d_k, kKElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dense, sizeof(h_dense)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_k, h_k.data(), kKElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_k_frag_load_matches_dense_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_k, d_pipe, d_dense);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dense, d_dense, sizeof(h_dense), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_k), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dense), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_dense[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, VFragLoadMatchesDenseBaseline) {
    using KernelConfig = config::ForwardConfig;
    using Element = typename KernelConfig::Element;
    using FragV = typename KernelConfig::FragV;
    constexpr int kVElems = KernelConfig::BlockN * KernelConfig::HeadDim;
    constexpr int kRegs = 32 * FragV::REGS_PER_LANE;

    std::vector<unsigned short> h_v(kVElems);
    for (int idx = 0; idx < kVElems; ++idx) {
        int row = idx / KernelConfig::HeadDim;
        int col = idx % KernelConfig::HeadDim;
        h_v[idx] = static_cast<unsigned short>((row << 8) | col);
    }

    Element* d_v = nullptr;
    uint32_t *d_pipe = nullptr, *d_dense = nullptr;
    uint32_t h_pipe[kRegs] = {}, h_dense[kRegs] = {};
    ASSERT_EQ(cudaMalloc(&d_v, kVElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pipe, sizeof(h_pipe)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dense, sizeof(h_dense)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_v, h_v.data(), kVElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_v_frag_load_matches_dense_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_v, d_pipe, d_dense);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_pipe, d_pipe, sizeof(h_pipe), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dense, d_dense, sizeof(h_dense), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_v), cudaSuccess);
    ASSERT_EQ(cudaFree(d_pipe), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dense), cudaSuccess);

    for (int i = 0; i < kRegs; ++i) {
        EXPECT_EQ(h_pipe[i], h_dense[i]) << "mismatch at reg " << i;
    }
}

TEST(CopySrmTest, PVMmaWithPipelineVMatchesReference) {
    using KernelConfig = config::ForwardConfig;
    using Element = typename KernelConfig::Element;
    using FragAcc = typename KernelConfig::FragAcc;
    constexpr int kVElems = KernelConfig::BlockN * KernelConfig::HeadDim;
    constexpr int kAcc = 32 * 4;

    std::vector<Element> h_v(kVElems);
    for (int idx = 0; idx < kVElems; ++idx) {
        int row = idx / KernelConfig::HeadDim;
        int col = idx % KernelConfig::HeadDim;
        float v = 0.05f * static_cast<float>(row + 1) + 0.01f * static_cast<float>(col + 1) - 0.002f;
        h_v[idx] = __float2half_rn(v);
    }

    Element* d_v = nullptr;
    float* d_acc = nullptr;
    float h_acc[kAcc] = {};
    float matrix[16 * 8] = {};
    float ref[16 * 8] = {};
    ASSERT_EQ(cudaMalloc(&d_v, kVElems * sizeof(Element)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_acc, sizeof(h_acc)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_v, h_v.data(), kVElems * sizeof(Element), cudaMemcpyHostToDevice), cudaSuccess);

    test_pv_mma_pipeline_v_kernel<KernelConfig><<<1, KernelConfig::NThreads>>>(d_v, d_acc);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_acc, d_acc, sizeof(h_acc), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_v), cudaSuccess);
    ASSERT_EQ(cudaFree(d_acc), cudaSuccess);

    accum_lane_to_matrix_16x8_test(h_acc, matrix);

    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) {
                float p = 0.02f * static_cast<float>(m + 1) - 0.01f * static_cast<float>(k + 1) + 0.003f;
                float v = 0.05f * static_cast<float>(32 + k + 1) + 0.01f * static_cast<float>(24 + n + 1) - 0.002f;
                float p_q = __half2float(__float2half_rn(p));
                float v_q = __half2float(__float2half_rn(v));
                acc += p_q * v_q;
            }
            ref[m * 8 + n] = acc;
        }
    }

    constexpr float tol = 1e-2f;
    for (int i = 0; i < 16 * 8; ++i) {
        EXPECT_NEAR(matrix[i], ref[i], tol) << "mismatch at matrix idx " << i;
    }
}

TEST(CopySrmTest, DiagnoseQLaneMappingAcrossTwo8RowAtoms) {
    using KernelConfig = config::ForwardConfig;

    int* d_out = nullptr;
    int h_out[32 * 7] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);

    test_q_lane_mapping_kernel<KernelConfig><<<1, 32>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    auto check_lane = [&](int lane, int global_row, int global_col, int tile_row, int inner_row, int tile_col, int inner_col) {
        int base = lane * 7;
        EXPECT_EQ(h_out[base + 0], global_row) << "lane " << lane;
        EXPECT_EQ(h_out[base + 1], global_col) << "lane " << lane;
        EXPECT_EQ(h_out[base + 2], tile_row) << "lane " << lane;
        EXPECT_EQ(h_out[base + 3], inner_row) << "lane " << lane;
        EXPECT_EQ(h_out[base + 4], tile_col) << "lane " << lane;
        EXPECT_EQ(h_out[base + 5], inner_col) << "lane " << lane;
    };

    check_lane(0, 16, 32, 2, 0, 0, 32);
    check_lane(7, 23, 32, 2, 7, 0, 32);
    check_lane(15, 31, 32, 3, 7, 0, 32);
    check_lane(16, 16, 40, 2, 0, 0, 40);
    check_lane(23, 23, 40, 2, 7, 0, 40);
    check_lane(31, 31, 40, 3, 7, 0, 40);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
