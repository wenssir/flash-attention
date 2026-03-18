#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "src/loadstore/copy_atom.cuh"
#include "src/layout/layout.h"
#include "src/tensor_core/tensor.cuh"
#include "src/tensor_core/fragment.cuh"

namespace {

template <int M, int K, int ThreadM = 1, int ThreadK = 1, int CopyK = 1>
struct TestCfg {
    using Element = float;
    static constexpr int NWarps = 1;
    static constexpr int thread_m = ThreadM;
    static constexpr int thread_k = ThreadK;
    static constexpr int copy_k = CopyK;
    static constexpr auto warp_tile_shape = layout::make_shape(numeric::Int<M>{}, numeric::Int<K>{});
};

template <typename KernelConfig>
__global__ void copy_g2s_kernel(float* smem_like, float const* gmem) {
    auto l = layout::make_layout(
        KernelConfig::warp_tile_shape,
        layout::make_stride(cxx::get<1>(KernelConfig::warp_tile_shape), numeric::Int<1>{})
    );
    auto s_tensor = tensor::make_tensor(smem_like, l);
    auto g_tensor = tensor::make_tensor(gmem, l);
    using Map = loadstore::G2SLayoutThreadMap<
        KernelConfig,
        static_cast<int>(cxx::get<0>(KernelConfig::warp_tile_shape)),
        static_cast<int>(cxx::get<1>(KernelConfig::warp_tile_shape)),
        KernelConfig::copy_k>;
    loadstore::copy_g2s<Map>(g_tensor, s_tensor);
}

template <typename KernelConfig>
__global__ void copy_s2g_kernel(float* gmem, float const* smem_like) {
    auto l = layout::make_layout(
        KernelConfig::warp_tile_shape,
        layout::make_stride(cxx::get<1>(KernelConfig::warp_tile_shape), numeric::Int<1>{})
    );
    auto g_tensor = tensor::make_tensor(gmem, l);
    auto s_tensor = tensor::make_tensor(smem_like, l);
    loadstore::CopyS2GOp op{};
    op(g_tensor, s_tensor);
}

using FragA = tensor::Fragment<__half, tensor::LdmatrixHelperQ<__half>, 16, 16>;

__global__ void copy_s2r_kernel(uint32_t* out_regs, __half* smem_src) {
    auto l = layout::make_layout(
        layout::make_shape(numeric::Int<16>{}, numeric::Int<16>{}),
        layout::make_stride(numeric::Int<16>{}, numeric::Int<1>{})
    );
    auto s_tensor = tensor::make_tensor(smem_src, l);
    FragA frag;
    loadstore::CopyS2ROp op{};
    op(frag, s_tensor);
    for (int i = 0; i < FragA::REGS_PER_LANE; ++i) {
        out_regs[i] = frag(i);
    }
}

__global__ void copy_r2s_kernel(float* out_matrix) {
    auto l = layout::make_layout(
        layout::make_shape(numeric::Int<16>{}, numeric::Int<8>{}),
        layout::make_stride(numeric::Int<8>{}, numeric::Int<1>{})
    );
    auto s_tensor = tensor::make_tensor(out_matrix, l);

    float regs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    loadstore::CopyR2SOp<true> op{};
    op(s_tensor, regs);
}

} // namespace

TEST(CopyAtomTest, CopyG2SOp) {
    using Cfg = TestCfg<1, 1>;
    float* d_g = nullptr;
    float* d_s = nullptr;
    ASSERT_EQ(cudaMalloc(&d_g, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, sizeof(float)), cudaSuccess);

    float h = 7.5f;
    ASSERT_EQ(cudaMemcpy(d_g, &h, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_s, 0, sizeof(float)), cudaSuccess);

    copy_g2s_kernel<Cfg><<<1, 1>>>(d_s, d_g);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float out = 0.0f;
    ASSERT_EQ(cudaMemcpy(&out, d_s, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(out, h);

    ASSERT_EQ(cudaFree(d_g), cudaSuccess);
    ASSERT_EQ(cudaFree(d_s), cudaSuccess);
}

TEST(CopyAtomTest, CopyS2ROp) {
    uint32_t* d_regs = nullptr;
    __half* d_s = nullptr;
    ASSERT_EQ(cudaMalloc(&d_regs, FragA::REGS_PER_LANE * sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, 16 * 16 * sizeof(__half)), cudaSuccess);

    __half h_in[16 * 16];
    for (int i = 0; i < 16 * 16; ++i) {
        h_in[i] = __float2half(1.0f);
    }
    ASSERT_EQ(cudaMemcpy(d_s, h_in, sizeof(h_in), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_regs, 0, FragA::REGS_PER_LANE * sizeof(uint32_t)), cudaSuccess);

    copy_s2r_kernel<<<1, 1>>>(d_regs, d_s);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t h_regs[FragA::REGS_PER_LANE] = {};
    ASSERT_EQ(cudaMemcpy(h_regs, d_regs, sizeof(h_regs), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < FragA::REGS_PER_LANE; ++i) {
        EXPECT_NE(h_regs[i], 0u);
    }

    ASSERT_EQ(cudaFree(d_regs), cudaSuccess);
    ASSERT_EQ(cudaFree(d_s), cudaSuccess);
}

TEST(CopyAtomTest, CopyR2SOp) {
    float* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, 16 * 8 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_out, 0, 16 * 8 * sizeof(float)), cudaSuccess);

    copy_r2s_kernel<<<1, 1>>>(d_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float h_out[16 * 8] = {};
    ASSERT_EQ(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_FLOAT_EQ(h_out[0 * 8 + 0], 1.0f);
    EXPECT_FLOAT_EQ(h_out[0 * 8 + 1], 2.0f);
    EXPECT_FLOAT_EQ(h_out[8 * 8 + 0], 3.0f);
    EXPECT_FLOAT_EQ(h_out[8 * 8 + 1], 4.0f);

    ASSERT_EQ(cudaFree(d_out), cudaSuccess);
}

TEST(CopyAtomTest, CopyS2GOp) {
    using Cfg = TestCfg<1, 1>;
    float* d_g = nullptr;
    float* d_s = nullptr;
    ASSERT_EQ(cudaMalloc(&d_g, sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, sizeof(float)), cudaSuccess);

    float h = 9.25f;
    ASSERT_EQ(cudaMemcpy(d_s, &h, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_g, 0, sizeof(float)), cudaSuccess);

    copy_s2g_kernel<Cfg><<<1, 1>>>(d_g, d_s);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float out = 0.0f;
    ASSERT_EQ(cudaMemcpy(&out, d_g, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_FLOAT_EQ(out, h);

    ASSERT_EQ(cudaFree(d_g), cudaSuccess);
    ASSERT_EQ(cudaFree(d_s), cudaSuccess);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
