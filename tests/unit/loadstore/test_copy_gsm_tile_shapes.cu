#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/config/config.cuh"
#include "src/layout/layout.h"
#include "src/loadstore/copy_gsm.cuh"
#include "src/tensor_core/tensor.cuh"

namespace {

struct CopyCfg16x128 : public config::KernelConfig<float> {
    using Element = float;
    static constexpr int BlockM = 16;
    static constexpr int BlockN = 64;
    static constexpr int HeadDim = 128;
    static constexpr int NWarps = 1;
    static constexpr int NThreads = 32;
    static constexpr int thread_m = 4;
    static constexpr int thread_k = 8;
    static constexpr int copy_k = 8;

    static constexpr auto warp_tile_shape =
        layout::make_shape(numeric::Int<16>{}, numeric::Int<128>{});
};

template <typename Cfg>
__global__ void copy_gsm_tile_kernel(float const* gmem, float* out) {
    auto tile_layout = layout::make_layout(
        layout::make_shape(numeric::Int<Cfg::BlockM>{}, numeric::Int<Cfg::HeadDim>{}),
        layout::make_stride(numeric::Int<Cfg::HeadDim>{}, numeric::Int<1>{})
    );
    auto g_tensor = tensor::make_tensor(gmem, tile_layout);
    auto out_tensor = tensor::make_tensor(out, tile_layout);
    using Map = loadstore::G2SLayoutThreadMap<Cfg, Cfg::BlockM, Cfg::HeadDim, Cfg::copy_k>;
    loadstore::copy_g2s<Map>(g_tensor, out_tensor);
}

template <typename Cfg>
void run_copy_tile_case() {
    constexpr int M = Cfg::BlockM;
    constexpr int K = Cfg::HeadDim;
    constexpr int Count = M * K;

    float h_in[Count];
    float h_out[Count];
    for (int i = 0; i < Count; ++i) {
        h_in[i] = static_cast<float>((i % 173) - 86) * 0.125f;
        h_out[i] = 0.0f;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_in, sizeof(h_in)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_out, 0, sizeof(h_out)), cudaSuccess);

    copy_gsm_tile_kernel<Cfg><<<1, Cfg::NThreads>>>(d_in, d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < Count; ++i) {
        ASSERT_FLOAT_EQ(h_in[i], h_out[i]) << "Mismatch at idx=" << i;
    }

    ASSERT_EQ(cudaFree(d_in), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);
}

} // namespace

TEST(CopyGsmTileShapesTest, Copy16x128_OneWarp) {
    run_copy_tile_case<CopyCfg16x128>();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
