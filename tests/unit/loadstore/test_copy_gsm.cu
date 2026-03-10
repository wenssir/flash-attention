#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/loadstore/copy_gsm.cuh"

namespace {

template <int WarpM, int WarpK, int ThreadM, int ThreadK, int CopyK>
struct TestKernelConfig {
    using Element = float;
    static constexpr int NWarps = 1;
    static constexpr int thread_m = ThreadM;
    static constexpr int thread_k = ThreadK;
    static constexpr int copy_k = CopyK;
    static constexpr auto warp_tile_shape = layout::make_shape(numeric::Int<WarpM>{}, numeric::Int<WarpK>{});
    static constexpr auto thread_shape_in_warp = layout::make_shape(numeric::Int<1>{}, numeric::Int<CopyK>{});
};

template <typename KernelConfig>
__global__ void copy_gsm_kernel(float const* gmem, float* smem_like) {
    constexpr auto kWarpK = cxx::get<1>(KernelConfig::warp_tile_shape);
    auto layout_2d = layout::make_layout(
        KernelConfig::warp_tile_shape,
        layout::make_stride(kWarpK, numeric::Int<1>{})
    );

    auto g_tensor = tensor::make_tensor(gmem, layout_2d);
    auto s_tensor = tensor::make_tensor(smem_like, layout_2d);
    using Map = loadstore::G2SLayoutThreadMap<
        KernelConfig,
        static_cast<int>(cxx::get<0>(KernelConfig::warp_tile_shape)),
        static_cast<int>(cxx::get<1>(KernelConfig::warp_tile_shape)),
        KernelConfig::copy_k>;
    loadstore::copy_g2s<Map>(g_tensor, s_tensor);
}

template <typename KernelConfig, int M, int K>
void run_copy_test() {
    constexpr int kCount = M * K;
    float h_in[kCount];
    float h_out[kCount];
    for (int i = 0; i < kCount; ++i) {
        h_in[i] = static_cast<float>(i + 1);
        h_out[i] = 0.0f;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_in, sizeof(h_in)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_out, 0, sizeof(h_out)), cudaSuccess);

    copy_gsm_kernel<KernelConfig><<<1, 32>>>(d_in, d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < kCount; ++i) {
        EXPECT_FLOAT_EQ(h_out[i], h_in[i]) << "Mismatch at index " << i;
    }

    ASSERT_EQ(cudaFree(d_in), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);
}

} // namespace

TEST(CopyGsmTest, Warp16x128Thread4x8Copy8) {
    using Cfg = TestKernelConfig<16, 128, 4, 8, 8>;
    run_copy_test<Cfg, 16, 128>();
}

TEST(CopyGsmTest, Warp8x64Thread4x8Copy4) {
    using Cfg = TestKernelConfig<8, 64, 4, 8, 4>;
    run_copy_test<Cfg, 8, 64>();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
