#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/loadstore/copy_gsm.cuh"

namespace {

template <int WarpM, int WarpK, int ThreadM, int ThreadK, int CopyK>
struct TestKernelConfig {
    static constexpr int thread_m = ThreadM;
    static constexpr int thread_k = ThreadK;
    static constexpr int copy_k = CopyK;
    static constexpr auto warp_tile_shape = layout::make_shape(numeric::Int<WarpM>{}, numeric::Int<WarpK>{});
    static constexpr auto thread_shape_in_warp = layout::make_shape(numeric::Int<1>{}, numeric::Int<CopyK>{});
};

template <typename KernelConfig>
__global__ void query_outer_shape_kernel(int* out_m, int* out_k) {
    constexpr auto kWarpK = cxx::get<1>(KernelConfig::warp_tile_shape);
    auto warp_layout = layout::make_layout(
        KernelConfig::warp_tile_shape,
        layout::make_stride(kWarpK, numeric::Int<1>{})
    );
    auto iter_tile_shape = loadstore::make_iter_tile_shape<KernelConfig>();
    auto divided = layout::zipped_divide(warp_layout, iter_tile_shape);
    auto outer_shape = divided.outer.shape();
    *out_m = static_cast<int>(layout::get_shape_size(cxx::get<0>(outer_shape)));
    *out_k = static_cast<int>(layout::get_shape_size(cxx::get<1>(outer_shape)));
}

} // namespace

TEST(ThreadSliceShapeTest, OuterShapeIs4x2For16x128) {
    using Cfg = TestKernelConfig<16, 128, 4, 8, 8>;
    int* d_m = nullptr;
    int* d_k = nullptr;
    ASSERT_EQ(cudaMalloc(&d_m, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_k, sizeof(int)), cudaSuccess);

    query_outer_shape_kernel<Cfg><<<1, 1>>>(d_m, d_k);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_m = -1;
    int h_k = -1;
    ASSERT_EQ(cudaMemcpy(&h_m, d_m, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_k, d_k, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_m, 4);
    EXPECT_EQ(h_k, 2);

    ASSERT_EQ(cudaFree(d_m), cudaSuccess);
    ASSERT_EQ(cudaFree(d_k), cudaSuccess);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
