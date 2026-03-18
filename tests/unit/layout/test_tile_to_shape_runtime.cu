#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/layout/layout.h"
#include "src/layout/composedLayout.h"
#include "src/tensor_core/tensor.cuh"
#include "src/tensor_core/swizzle.cuh"
#include "src/loadstore/thread_map.cuh"

namespace {

template <int BlockM_, int BlockN_, int CopyK, int NWarps_>
struct TestKernelConfig {
    using Element = float;
    static constexpr int BlockM = BlockM_;
    static constexpr int BlockN = BlockN_;
    static constexpr int HeadDim = BlockN_;
    static constexpr int copy_k = CopyK;
    static constexpr int NWarps = NWarps_;
    static constexpr int SmemAtomRows = 8;
    static constexpr int SmemAtomCols = 64;
};

__global__ void test_plain_layout_kernel(int* out_layout, int* out_dot) {
    if (threadIdx.x != 0) {
        return;
    }

    auto coord00 = layout::make_coordinate(0, 0);
    auto coord064 = layout::make_coordinate(0, 64);
    auto coord40 = layout::make_coordinate(4, 0);
    auto coord464 = layout::make_coordinate(4, 64);
    auto stride = layout::make_stride(numeric::Int<128>{}, numeric::Int<1>{});
    auto g_layout = layout::make_layout(
        layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{}),
        stride);

    out_layout[0] = static_cast<int>(g_layout(coord00));
    out_layout[1] = static_cast<int>(g_layout(coord064));
    out_layout[2] = static_cast<int>(g_layout(coord40));
    out_layout[3] = static_cast<int>(g_layout(coord464));

    out_dot[0] = static_cast<int>(container::dot_product(coord00, stride));
    out_dot[1] = static_cast<int>(container::dot_product(coord064, stride));
    out_dot[2] = static_cast<int>(container::dot_product(coord40, stride));
    out_dot[3] = static_cast<int>(container::dot_product(coord464, stride));
}

__global__ void test_blocked_product_runtime_kernel(int* out) {
    if (threadIdx.x != 0) {
        return;
    }

    auto atom = layout::make_layout(
        layout::make_shape(numeric::Int<2>{}, numeric::Int<3>{}),
        layout::make_stride(numeric::Int<3>{}, numeric::Int<1>{}));
    auto tiler = layout::make_ordered_layout(layout::make_shape(numeric::Int<2>{}, numeric::Int<2>{}));
    auto result = layout::blocked_product(atom, tiler);

    auto c0 = layout::make_coordinate(layout::make_coordinate(0, 0), layout::make_coordinate(0, 0));
    auto c1 = layout::make_coordinate(layout::make_coordinate(0, 1), layout::make_coordinate(0, 0));
    auto c2 = layout::make_coordinate(layout::make_coordinate(1, 0), layout::make_coordinate(2, 0));
    auto c3 = layout::make_coordinate(layout::make_coordinate(1, 1), layout::make_coordinate(2, 1));

    out[0] = static_cast<int>(result(c0));
    out[1] = static_cast<int>(result(c1));
    out[2] = static_cast<int>(result(c2));
    out[3] = static_cast<int>(result(c3));
}

} // namespace

TEST(TileToShapeRuntimeTest, PlainLayoutAndDotProductMatchExpectedOffsets) {
    int *d_layout = nullptr, *d_dot = nullptr;
    int h_layout[4] = {}, h_dot[4] = {};

    ASSERT_EQ(cudaMalloc(&d_layout, sizeof(h_layout)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dot, sizeof(h_dot)), cudaSuccess);

    test_plain_layout_kernel<<<1, 1>>>(d_layout, d_dot);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_layout, d_layout, sizeof(h_layout), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dot, d_dot, sizeof(h_dot), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_layout), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dot), cudaSuccess);

    EXPECT_EQ(h_layout[0], 0);
    EXPECT_EQ(h_layout[1], 64);
    EXPECT_EQ(h_layout[2], 4 * 128);
    EXPECT_EQ(h_layout[3], 4 * 128 + 64);

    EXPECT_EQ(h_dot[0], 0);
    EXPECT_EQ(h_dot[1], 64);
    EXPECT_EQ(h_dot[2], 4 * 128);
    EXPECT_EQ(h_dot[3], 4 * 128 + 64);
}

TEST(TileToShapeRuntimeTest, BlockedProductRuntime2DPlainAtom) {
    int* d_out = nullptr;
    int h_out[4] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_blocked_product_runtime_kernel<<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 0);
    EXPECT_EQ(h_out[1], 6);
    EXPECT_EQ(h_out[2], 5);
    EXPECT_EQ(h_out[3], 23);
}

TEST(TileToShapeRuntimeTest, TiledNoSwizzleLayoutMatchesExpectedOffsets) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;

    int* d_out = nullptr;
    int h_out[4] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_tiled_no_swizzle_kernel<Cfg><<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 0);
    EXPECT_EQ(h_out[1], 512);
    EXPECT_EQ(h_out[2], 4 * 64);
    EXPECT_EQ(h_out[3], 4 * 64 + 512);
}

TEST(TileToShapeRuntimeTest, TiledPlainLayoutMatchesExpectedOffsets) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;

    int* d_out = nullptr;
    int h_out[4] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_tiled_plain_layout_kernel<Cfg><<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 0);
    EXPECT_EQ(h_out[1], 512);
    EXPECT_EQ(h_out[2], 4 * 64);
    EXPECT_EQ(h_out[3], 4 * 64 + 512);
}

TEST(TileToShapeRuntimeTest, TiledSwizzleLayoutMatchesExpectedOffsets) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;

    int* d_out = nullptr;
    int h_out[4] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_tiled_swizzle_kernel<Cfg><<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 0);
    EXPECT_EQ(h_out[1], 512);
    EXPECT_EQ(h_out[2], 288);
    EXPECT_EQ(h_out[3], 288 + 512);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
