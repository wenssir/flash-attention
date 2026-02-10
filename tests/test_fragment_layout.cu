#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "../src/layout/layout.h"
#include "../src/layout/composedLayout.h"
#include "../src/layout/offset.cuh"
#include "../src/tensor_core/fragment.cuh"
#include "../src/tensor_core/swizzle.cuh"
#include "../src/numeric/Int.cuh"

using namespace layout;
using namespace numeric;
using namespace tensor;

// =============================================================================
// Test: Fragment 编译和常量验证
// =============================================================================

TEST(FragmentLayoutTest, Fragment16x16_Constants) {
    // 验证编译时常量
    using Frag16x16 = Fragment<__half, 16, 16, Layout<int, int>>;
    EXPECT_EQ(Frag16x16::REGS_PER_LANE, 4);
}

TEST(FragmentLayoutTest, Fragment16x8_Constants) {
    using Frag16x8 = Fragment<__half, 16, 8, Layout<int, int>>;
    EXPECT_EQ(Frag16x8::REGS_PER_LANE, 2);
}

// =============================================================================
// Test: Kernel 测试（实际 GPU 执行）
// =============================================================================

__global__ void fragment_test_kernel(__half* smem, uint32_t* output) {
    int lane_id = threadIdx.x % 32;

    // 在 kernel 内直接构造简单的 Layout（不调用 make_layout）
    Layout<int, int> layout(32, 1);  // shape=32, stride=1

    // 创建 Fragment
    Fragment<__half, 16, 16, Layout<int, int>> frag(layout);

    // 加载
    frag.load(smem, lane_id);

    // 输出结果
    int base_idx = threadIdx.x * 4;
    output[base_idx + 0] = frag(0);
    output[base_idx + 1] = frag(1);
    output[base_idx + 2] = frag(2);
    output[base_idx + 3] = frag(3);
}

TEST(FragmentLayoutTest, KernelExecution) {
    // 测试 kernel 能够执行
    const int smem_size = 16 * 16;
    const int output_size = 32 * 4;  // 32 线程，每个 4 个寄存器

    __half* d_smem;
    uint32_t* d_output;
    cudaMalloc(&d_smem, smem_size * sizeof(__half));
    cudaMalloc(&d_output, output_size * sizeof(uint32_t));

    // 初始化 SMEM 数据
    __half* h_smem = new __half[smem_size];
    for (int i = 0; i < smem_size; ++i) {
        h_smem[i] = __float2half(i * 0.1f);
    }
    cudaMemcpy(d_smem, h_smem, smem_size * sizeof(__half), cudaMemcpyHostToDevice);

    // 运行 kernel
    fragment_test_kernel<<<1, 32>>>(d_smem, d_output);
    cudaDeviceSynchronize();

    // 验证没有错误
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess);

    // 清理
    delete[] h_smem;
    cudaFree(d_smem);
    cudaFree(d_output);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
