#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#include "tensor_core/fragment.cuh"
#include "loadstore/copy_srm.cuh"
#include "layout/layout.h"
#include "tensor_core/tensor.cuh"
#include "numeric/Int.cuh"
#include "ptx/ptx.cuh"

using namespace tensor;
using namespace loadstore;
using namespace layout;

namespace {

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
    LdmatrixHelper helper;
    auto coord = helper(lane_id);

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

    test_input_pointer_positions<__half, 16, 16, LdmatrixHelperQ<__half>><<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n=== Expected Pattern ===\n");
    printf("Each thread should read 8 consecutive elements in a row\n");
    printf("For 16x16: threads 0-7 read rows 0-15, cols 0-7\n");
    printf("             threads 8-15 read rows 0-15, cols 8-15 (but we only have 32 threads)\n");
    printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
