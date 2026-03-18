#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/loadstore/copy_gsm.cuh"
#include "src/layout/layout.h"
#include "src/layout/composedLayout.h"
#include "src/tensor_core/tensor.cuh"
#include "src/tensor_core/swizzle.cuh"

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

} // namespace

TEST(ThreadSliceShapeTest, QMapUses8x64PagingGeometry) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;
    using Map = loadstore::G2SLayoutThreadMap<Cfg, Cfg::BlockM, Cfg::HeadDim, Cfg::copy_k>;

    EXPECT_EQ(Map::IterRows, 4);
    EXPECT_EQ(Map::IterCols, 2);
}

template <typename Cfg>
__global__ void test_qmap_kernel(int* out) {
    using Map = loadstore::G2SLayoutThreadMap<Cfg, Cfg::BlockM, Cfg::HeadDim, Cfg::copy_k>;
    if (threadIdx.x != 0) {
        return;
    }
    auto lane0 = Map::lane_coord(0);
    auto lane7 = Map::lane_coord(7);
    auto lane8 = Map::lane_coord(8);
    out[0] = static_cast<int>(cxx::get<0>(lane0));
    out[1] = static_cast<int>(cxx::get<1>(lane0));
    out[2] = static_cast<int>(cxx::get<0>(lane7));
    out[3] = static_cast<int>(cxx::get<1>(lane7));
    out[4] = static_cast<int>(cxx::get<0>(lane8));
    out[5] = static_cast<int>(cxx::get<1>(lane8));
    out[6] = Map::warp_row_base(0);
    out[7] = Map::warp_row_base(1);
    out[8] = Map::row_advance(0);
    out[9] = Map::row_advance(1);
    out[10] = Map::col_advance(0);
    out[11] = Map::col_advance(1);
}

TEST(ThreadSliceShapeTest, QMapKernelValuesMatchExpectedGeometry) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;

    int* d_out = nullptr;
    int h_out[12] = {};
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    test_qmap_kernel<Cfg><<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);

    EXPECT_EQ(h_out[0], 0);
    EXPECT_EQ(h_out[1], 0);
    EXPECT_EQ(h_out[2], 0);
    EXPECT_EQ(h_out[3], 56);
    EXPECT_EQ(h_out[4], 1);
    EXPECT_EQ(h_out[5], 0);
    EXPECT_EQ(h_out[6], 0);
    EXPECT_EQ(h_out[7], 16);
    EXPECT_EQ(h_out[8], 0);
    EXPECT_EQ(h_out[9], 4);
    EXPECT_EQ(h_out[10], 0);
    EXPECT_EQ(h_out[11], 64);
}

template <typename Cfg>
__global__ void test_partition_q_kernel(int* out_src, int* out_dst_linear, int* out_dst_swizzle) {
    using Map = loadstore::G2SLayoutThreadMap<Cfg, Cfg::BlockM, Cfg::HeadDim, Cfg::copy_k>;
    if (threadIdx.x != 0) {
        return;
    }

    float* storage = nullptr;
    auto g_layout = layout::make_layout(
        layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{}),
        layout::make_stride(numeric::Int<128>{}, numeric::Int<1>{}));
    auto g_tensor = tensor::make_tensor(storage, g_layout);

    auto atom_linear = layout::composition(
        tensor::NoSwizzle{},
        layout::make_layout(
            layout::make_shape(numeric::Int<8>{}, numeric::Int<64>{}),
            layout::make_stride(numeric::Int<64>{}, numeric::Int<1>{})));
    auto atom_swizzle = layout::composition(
        tensor::Swizzle<3, 3, 3>{},
        layout::make_layout(
            layout::make_shape(numeric::Int<8>{}, numeric::Int<64>{}),
            layout::make_stride(numeric::Int<64>{}, numeric::Int<1>{})));

    auto smem_linear = tensor::make_tensor(
        storage,
        layout::tile_to_shape(atom_linear, layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{})));
    auto smem_swizzle = tensor::make_tensor(
        storage,
        layout::tile_to_shape(atom_swizzle, layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{})));

    int lane = 0;
    int warp = 0;
    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));

    out_src[0] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(0), base_col + Map::col_advance(0))));
    out_src[1] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(0), base_col + Map::col_advance(1))));
    out_src[2] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(1), base_col + Map::col_advance(0))));
    out_src[3] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(1), base_col + Map::col_advance(1))));

    out_dst_linear[0] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(0));
    out_dst_linear[1] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(1));
    out_dst_linear[2] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(0));
    out_dst_linear[3] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(1));

    out_dst_swizzle[0] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(0));
    out_dst_swizzle[1] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(1));
    out_dst_swizzle[2] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(0));
    out_dst_swizzle[3] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(1));
}

template <typename Cfg>
__global__ void test_partition_k_kernel(int* out_src, int* out_dst_linear, int* out_dst_swizzle) {
    using Map = loadstore::G2SLayoutThreadMap<Cfg, Cfg::BlockN, Cfg::HeadDim, Cfg::copy_k>;
    if (threadIdx.x != 0) {
        return;
    }

    float* storage = nullptr;
    auto g_layout = layout::make_layout(
        layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{}),
        layout::make_stride(numeric::Int<128>{}, numeric::Int<1>{}));
    auto g_tensor = tensor::make_tensor(storage, g_layout);

    auto atom_linear = layout::composition(
        tensor::NoSwizzle{},
        layout::make_layout(
            layout::make_shape(numeric::Int<8>{}, numeric::Int<64>{}),
            layout::make_stride(numeric::Int<64>{}, numeric::Int<1>{})));
    auto atom_swizzle = layout::composition(
        tensor::Swizzle<3, 3, 3>{},
        layout::make_layout(
            layout::make_shape(numeric::Int<8>{}, numeric::Int<64>{}),
            layout::make_stride(numeric::Int<64>{}, numeric::Int<1>{})));

    auto smem_linear = tensor::make_tensor(
        storage,
        layout::tile_to_shape(atom_linear, layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{})));
    auto smem_swizzle = tensor::make_tensor(
        storage,
        layout::tile_to_shape(atom_swizzle, layout::make_shape(numeric::Int<64>{}, numeric::Int<128>{})));

    int lane = 0;
    int warp = 0;
    auto lane_coord = Map::lane_coord(lane);
    int base_row = Map::warp_row_base(warp) + static_cast<int>(cxx::get<0>(lane_coord));
    int base_col = static_cast<int>(cxx::get<1>(lane_coord));

    out_src[0] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(0), base_col + Map::col_advance(0))));
    out_src[1] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(0), base_col + Map::col_advance(1))));
    out_src[2] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(1), base_col + Map::col_advance(0))));
    out_src[3] = static_cast<int>(g_tensor.layout()(layout::make_coordinate(base_row + Map::row_advance(1), base_col + Map::col_advance(1))));

    out_dst_linear[0] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(0));
    out_dst_linear[1] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(1));
    out_dst_linear[2] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(0));
    out_dst_linear[3] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(1));

    out_dst_swizzle[0] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(0));
    out_dst_swizzle[1] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(0), base_col + Map::col_advance(1));
    out_dst_swizzle[2] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(0));
    out_dst_swizzle[3] = loadstore::smem_offset_g2s<Cfg>(base_row + Map::row_advance(1), base_col + Map::col_advance(1));
}

TEST(ThreadSliceShapeTest, PartitionQViewsMatchExpectedOffsets) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;

    int *d_src = nullptr, *d_dst_linear = nullptr, *d_dst_swizzle = nullptr;
    int h_src[4] = {}, h_dst_linear[4] = {}, h_dst_swizzle[4] = {};

    ASSERT_EQ(cudaMalloc(&d_src, sizeof(h_src)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst_linear, sizeof(h_dst_linear)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst_swizzle, sizeof(h_dst_swizzle)), cudaSuccess);

    test_partition_q_kernel<Cfg><<<1, 32>>>(d_src, d_dst_linear, d_dst_swizzle);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_src, d_src, sizeof(h_src), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dst_linear, d_dst_linear, sizeof(h_dst_linear), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dst_swizzle, d_dst_swizzle, sizeof(h_dst_swizzle), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_src[0], 0);
    EXPECT_EQ(h_src[1], 64);
    EXPECT_EQ(h_src[2], 4 * 128);
    EXPECT_EQ(h_src[3], 4 * 128 + 64);

    EXPECT_EQ(h_dst_linear[0], 0);
    EXPECT_EQ(h_dst_linear[1], 512);
    EXPECT_EQ(h_dst_linear[2], 4 * 64);
    EXPECT_EQ(h_dst_linear[3], 4 * 64 + 512);

    EXPECT_EQ(h_dst_swizzle[0], 0);
    EXPECT_EQ(h_dst_swizzle[1], 512);
    EXPECT_EQ(h_dst_swizzle[2], 288);
    EXPECT_EQ(h_dst_swizzle[3], 288 + 512);

    ASSERT_EQ(cudaFree(d_src), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dst_linear), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dst_swizzle), cudaSuccess);
}

TEST(ThreadSliceShapeTest, PartitionKViewsMatchExpectedOffsets) {
    using Cfg = TestKernelConfig<64, 128, 8, 4>;

    int *d_src = nullptr, *d_dst_linear = nullptr, *d_dst_swizzle = nullptr;
    int h_src[4] = {}, h_dst_linear[4] = {}, h_dst_swizzle[4] = {};

    ASSERT_EQ(cudaMalloc(&d_src, sizeof(h_src)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst_linear, sizeof(h_dst_linear)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst_swizzle, sizeof(h_dst_swizzle)), cudaSuccess);

    test_partition_k_kernel<Cfg><<<1, 32>>>(d_src, d_dst_linear, d_dst_swizzle);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_src, d_src, sizeof(h_src), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dst_linear, d_dst_linear, sizeof(h_dst_linear), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_dst_swizzle, d_dst_swizzle, sizeof(h_dst_swizzle), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_src[0], 0);
    EXPECT_EQ(h_src[1], 64);
    EXPECT_EQ(h_src[2], 4 * 128);
    EXPECT_EQ(h_src[3], 4 * 128 + 64);

    EXPECT_EQ(h_dst_linear[0], 0);
    EXPECT_EQ(h_dst_linear[1], 512);
    EXPECT_EQ(h_dst_linear[2], 4 * 64);
    EXPECT_EQ(h_dst_linear[3], 4 * 64 + 512);

    EXPECT_EQ(h_dst_swizzle[0], 0);
    EXPECT_EQ(h_dst_swizzle[1], 512);
    EXPECT_EQ(h_dst_swizzle[2], 288);
    EXPECT_EQ(h_dst_swizzle[3], 288 + 512);

    ASSERT_EQ(cudaFree(d_src), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dst_linear), cudaSuccess);
    ASSERT_EQ(cudaFree(d_dst_swizzle), cudaSuccess);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
