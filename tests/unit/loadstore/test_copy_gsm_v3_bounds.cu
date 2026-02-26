#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/config/config.cuh"
#include "src/loadstore/copy_gsm.cuh"
#include "src/layout/layout.h"
#include "src/tensor_core/tensor.cuh"

namespace {

template <typename Cfg>
__global__ void copy_gsm_v3_kernel(float const* gmem, float* smem_like) {
    auto l = layout::make_layout(
        layout::make_shape(numeric::Int<Cfg::BlockM>{}, numeric::Int<Cfg::HeadDim>{}),
        layout::make_stride(numeric::Int<Cfg::HeadDim>{}, numeric::Int<1>{})
    );
    auto g_tensor = tensor::make_tensor(gmem, l);
    auto s_tensor = tensor::make_tensor(smem_like, l);
    loadstore::copy_with_map_and_slice<Cfg>(g_tensor, s_tensor);
}

} // namespace

TEST(CopyGsmV3Test, BoundsAndCorrectness_64x128_4Warps) {
    using Cfg = config::ForwardV3Config;
    constexpr int M = Cfg::BlockM;
    constexpr int K = Cfg::HeadDim;
    constexpr int Count = M * K;

    float h_in[Count];
    float h_out[Count];
    for (int i = 0; i < Count; ++i) {
        h_in[i] = static_cast<float>((i % 97) - 48) * 0.25f;
        h_out[i] = 0.0f;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_in, sizeof(h_in)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(h_out)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_out, 0, sizeof(h_out)), cudaSuccess);

    copy_gsm_v3_kernel<Cfg><<<1, Cfg::NThreads>>>(d_in, d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < Count; ++i) {
        ASSERT_FLOAT_EQ(h_in[i], h_out[i]) << "Mismatch at idx=" << i;
    }

    ASSERT_EQ(cudaFree(d_in), cudaSuccess);
    ASSERT_EQ(cudaFree(d_out), cudaSuccess);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
