#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "src/tensor_core/tensor.cuh"

using namespace tensor;

// Helper kernel to copy tile data to contiguous memory
template <typename TileType, typename OutputType>
__global__ void copy_tile_by_coords(TileType tile, OutputType* output) {
    auto shape = tile.shape();
    constexpr int rank = cxx::tuple_size_v<decltype(shape)>;

    if constexpr (rank == 2) {
        int rows = cxx::get<0>(shape);
        int cols = cxx::get<1>(shape);
        int idx = threadIdx.x;

        if (idx < rows * cols) {
            int r = idx / cols;
            int c = idx % cols;
            output[idx] = tile(r, c);
        }
    }
}

namespace {

// Test basic local_tile functionality
TEST(LocalTileTest, Basic2DTile) {
    // Setup: 8x8 matrix
    constexpr int rows = 8;
    constexpr int cols = 8;
    float h_data[rows * cols];

    // Fill with row-major data (C/C++ convention)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            h_data[r * cols + c] = static_cast<float>(r * 10 + c);
        }
    }

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, sizeof(h_data)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice), cudaSuccess);

    // Create tensor with row-major layout: stride = (cols, 1)
    // For shape=(rows, cols): coord(r,c) → r*cols + c
    auto shape = layout::make_shape(numeric::Int<rows>{}, numeric::Int<cols>{});
    auto stride = layout::make_stride(numeric::Int<cols>{}, numeric::Int<1>{});
    auto layout = layout::make_layout(shape, stride);
    auto tensor = make_tensor(d_data, layout);

    // Test 1: Extract 2x2 tile at (0, 0)
    {
        auto tile_shape = layout::make_shape(numeric::Int<2>{}, numeric::Int<2>{});
        auto tile_layout = layout::make_layout(tile_shape, stride);
        auto coord = layout::make_coordinate(numeric::Int<0>{}, numeric::Int<0>{});
        auto tile = local_tile(tensor, tile_layout, coord);

        // Copy using tile's operator() which respects the layout
        float* d_tile_contiguous = nullptr;
        ASSERT_EQ(cudaMalloc(&d_tile_contiguous, 4 * sizeof(float)), cudaSuccess);
        copy_tile_by_coords<<<1, 4>>>(tile, d_tile_contiguous);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        float h_result[4];
        ASSERT_EQ(cudaMemcpy(h_result, d_tile_contiguous, sizeof(h_result), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaFree(d_tile_contiguous), cudaSuccess);

        // Row-major layout: data[r][c] is at r*cols + c
        // tile 索引: [0] [1] [2] [3]
        // 对应位置: (0,0) (0,1) (1,0) (1,1)
        EXPECT_FLOAT_EQ(h_result[0], h_data[0]);  // (0,0) -> h_data[0*8+0] = 0
        EXPECT_FLOAT_EQ(h_result[1], h_data[1]);  // (0,1) -> h_data[0*8+1] = 1
        EXPECT_FLOAT_EQ(h_result[2], h_data[8]);  // (1,0) -> h_data[1*8+0] = 10
        EXPECT_FLOAT_EQ(h_result[3], h_data[9]);  // (1,1) -> h_data[1*8+1] = 11
    }

    // Test 2: Extract 2x2 tile at (2, 3)
    {
        auto tile_shape = layout::make_shape(numeric::Int<2>{}, numeric::Int<2>{});
        auto tile_layout = layout::make_layout(tile_shape, stride);
        auto coord = layout::make_coordinate(numeric::Int<2>{}, numeric::Int<3>{});
        auto tile = local_tile(tensor, tile_layout, coord);

        float* d_tile_contiguous = nullptr;
        ASSERT_EQ(cudaMalloc(&d_tile_contiguous, 4 * sizeof(float)), cudaSuccess);
        copy_tile_by_coords<<<1, 4>>>(tile, d_tile_contiguous);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        float h_result[4];
        ASSERT_EQ(cudaMemcpy(h_result, d_tile_contiguous, sizeof(h_result), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaFree(d_tile_contiguous), cudaSuccess);

        // coord = (2, 3) means the 2nd tile in row dim, 3rd tile in col dim
        // tile_shape = (2, 2), so top-left corner is at (2*2, 3*2) = (4, 6)
        EXPECT_FLOAT_EQ(h_result[0], h_data[4 * 8 + 6]);   // tile(0,0) -> tensor(4,6) -> 46
        EXPECT_FLOAT_EQ(h_result[1], h_data[4 * 8 + 7]);   // tile(0,1) -> tensor(4,7) -> 47
        EXPECT_FLOAT_EQ(h_result[2], h_data[5 * 8 + 6]);   // tile(1,0) -> tensor(5,6) -> 56
        EXPECT_FLOAT_EQ(h_result[3], h_data[5 * 8 + 7]);   // tile(1,1) -> tensor(5,7) -> 57
    }

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test edge case: single element tile
TEST(LocalTileTest, SingleElementTile) {
    constexpr int rows = 4;
    constexpr int cols = 4;
    float h_data[rows * cols] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, sizeof(h_data)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice), cudaSuccess);

    auto shape = layout::make_shape(numeric::Int<rows>{}, numeric::Int<cols>{});
    auto stride = layout::make_stride(numeric::Int<rows>{}, numeric::Int<1>{});
    auto layout = layout::make_layout(shape, stride);
    auto tensor = make_tensor(d_data, layout);

    // Extract 1x1 tile at each corner
    struct TestCase {
        int r, c;
        float expected;
    };

    TestCase tests[] = {
        {0, 0, 1.0f},
        {0, 3, 4.0f},
        {3, 0, 13.0f},
        {3, 3, 16.0f},
        {1, 2, 7.0f}
    };

    for (const auto& test : tests) {
        auto tile_shape = layout::make_shape(numeric::Int<1>{}, numeric::Int<1>{});
        auto tile_layout = layout::make_layout(tile_shape, stride);
        auto coord = layout::make_coordinate(test.r, test.c);
        auto tile = local_tile(tensor, tile_layout, coord);

        float* d_result = nullptr;
        ASSERT_EQ(cudaMalloc(&d_result, sizeof(float)), cudaSuccess);
        copy_tile_by_coords<<<1, 1>>>(tile, d_result);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        float result;
        ASSERT_EQ(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaFree(d_result), cudaSuccess);
        EXPECT_FLOAT_EQ(result, test.expected)
            << "Failed at coord (" << test.r << ", " << test.c << ")";
    }

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test edge case: tile covering entire tensor
TEST(LocalTileTest, EntireTensorAsTile) {
    constexpr int rows = 4;
    constexpr int cols = 4;
    float h_data[rows * cols];

    for (int i = 0; i < rows * cols; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, sizeof(h_data)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice), cudaSuccess);

    auto shape = layout::make_shape(numeric::Int<rows>{}, numeric::Int<cols>{});
    auto stride = layout::make_stride(numeric::Int<rows>{}, numeric::Int<1>{});
    auto layout = layout::make_layout(shape, stride);
    auto tensor = make_tensor(d_data, layout);

    // Extract entire tensor as a single tile
    auto tile_shape = layout::make_shape(numeric::Int<rows>{}, numeric::Int<cols>{});
    auto tile_layout = layout::make_layout(tile_shape, stride);
    auto coord = layout::make_coordinate(numeric::Int<0>{}, numeric::Int<0>{});
    auto tile = local_tile(tensor, tile_layout, coord);

    // Tile should point to same data
    EXPECT_EQ(tile.data_ptr(), d_data);

    // Read through tile and verify
    float* d_result_contiguous = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result_contiguous, rows * cols * sizeof(float)), cudaSuccess);
    copy_tile_by_coords<<<1, rows * cols>>>(tile, d_result_contiguous);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float h_result[rows * cols];
    ASSERT_EQ(cudaMemcpy(h_result, d_result_contiguous, sizeof(h_result), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_result_contiguous), cudaSuccess);

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]) << "Mismatch at index " << i;
    }

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test edge case: non-uniform stride
TEST(LocalTileTest, NonUniformStride) {
    constexpr int rows = 4;
    constexpr int cols = 3;
    constexpr int stride_rows = 8;  // Non-contiguous (padding)
    constexpr int stride_cols = 1;

    float h_data[stride_rows * cols];
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < stride_rows; ++r) {
            h_data[c * stride_rows + r] = (r < rows) ? static_cast<float>(c * 10 + r) : -1.0f;
        }
    }

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, sizeof(h_data)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice), cudaSuccess);

    auto shape = layout::make_shape(numeric::Int<rows>{}, numeric::Int<cols>{});
    auto stride = layout::make_stride(numeric::Int<stride_rows>{}, numeric::Int<stride_cols>{});
    auto layout = layout::make_layout(shape, stride);
    auto tensor = make_tensor(d_data, layout);

    // Extract 2x2 tile at tile index (0, 0)
    // tile_shape = (2, 2), so actual top-left corner is at (0*2, 0*2) = (0, 0)
    auto tile_shape = layout::make_shape(numeric::Int<2>{}, numeric::Int<2>{});
    auto tile_layout = layout::make_layout(tile_shape, stride);
    auto coord = layout::make_coordinate(numeric::Int<0>{}, numeric::Int<0>{});
    auto tile = local_tile(tensor, tile_layout, coord);

    // Verify accesses respect non-uniform stride
    float* d_tile_contiguous = nullptr;
    ASSERT_EQ(cudaMalloc(&d_tile_contiguous, 4 * sizeof(float)), cudaSuccess);
    copy_tile_by_coords<<<1, 4>>>(tile, d_tile_contiguous);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float h_result[4];
    ASSERT_EQ(cudaMemcpy(h_result, d_tile_contiguous, sizeof(h_result), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_tile_contiguous), cudaSuccess);

    // coord = (0, 0), tile_shape = (2, 2) -> top-left corner at (0, 0)
    // Tensor shape = (4, 3), stride = (8, 1)
    // tensor(r, c) is at h_data[r * stride_rows + c]
    // Data layout (transposed):
    //   h_data[0..7]   = {0, 1, 2, 3, -1, -1, -1, -1}     <- column 0 in data, row 0 in tensor
    //   h_data[8..15]  = {10, 11, 12, 13, -1, -1, -1, -1}  <- column 1 in data, row 1 in tensor
    //   h_data[16..23] = {20, 21, 22, 23, -1, -1, -1, -1}  <- column 2 in data, row 2 in tensor
    EXPECT_FLOAT_EQ(h_result[0], h_data[0 * stride_rows + 0]);  // tile(0,0) -> tensor(0,0) -> h_data[0] = 0
    EXPECT_FLOAT_EQ(h_result[1], h_data[0 * stride_rows + 1]);  // tile(0,1) -> tensor(0,1) -> h_data[1] = 1
    EXPECT_FLOAT_EQ(h_result[2], h_data[1 * stride_rows + 0]);  // tile(1,0) -> tensor(1,0) -> h_data[8] = 10
    EXPECT_FLOAT_EQ(h_result[3], h_data[1 * stride_rows + 1]);  // tile(1,1) -> tensor(1,1) -> h_data[9] = 11

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test coordinate computation on device
// This test verifies that local_tile works correctly when called on GPU device
__global__ void local_tile_kernel(float* data, float* results, int* checks) {
    auto shape = layout::make_shape(numeric::Int<8>{}, numeric::Int<8>{});
    auto stride = layout::make_stride(numeric::Int<8>{}, numeric::Int<1>{});
    auto layout = layout::make_layout(shape, stride);
    auto tensor = make_tensor(data, layout);

    // Test different tiles with coord = (tile_row_idx, tile_col_idx)
    auto tile_shape = layout::make_shape(numeric::Int<2>{}, numeric::Int<2>{});
    auto tile_layout = layout::make_layout(tile_shape, stride);

    int idx = 0;
    for (int tile_r = 0; tile_r < 4; tile_r += 2) {
        for (int tile_c = 0; tile_c < 4; tile_c += 2) {
            // coord represents "which tile" (tile index), not direct coordinates
            auto coord = layout::make_coordinate(tile_r, tile_c);
            auto tile = local_tile(tensor, tile_layout, coord);

            // Read first element of tile (tile(0,0))
            results[idx] = tile(0, 0);

            // Verify coordinate calculation:
            // coord = (tile_r, tile_c) means actual top-left corner is at:
            //   (tile_r * tile_rows, tile_c * tile_cols)
            // For tile_shape=(2,2): actual_coord = (tile_r*2, tile_c*2)
            int actual_r = tile_r * 2;
            int actual_c = tile_c * 2;
            int expected_offset = actual_r * 8 + actual_c;  // row-major: r*cols + c

            // Get offset from tile's ComposedLayout
            auto tile_layout = tile.layout();
            auto tile_offset = tile_layout.offset();
            int actual_offset = tile_offset();  // Call operator() to get the offset value

            checks[idx] = (expected_offset == actual_offset) ? 1 : 0;

            idx++;
        }
    }
}

TEST(LocalTileTest, DeviceSideCoordinateComputation) {
    constexpr int data_size = 8 * 8;
    float h_data[data_size];
    for (int i = 0; i < data_size; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    float* d_data = nullptr;
    float* d_results = nullptr;
    int* d_checks = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, sizeof(h_data)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_results, 4 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_checks, 4 * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice), cudaSuccess);

    local_tile_kernel<<<1, 1>>>(d_data, d_results, d_checks);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float h_results[4];
    int h_checks[4];
    ASSERT_EQ(cudaMemcpy(h_results, d_results, sizeof(h_results), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_checks, d_checks, sizeof(h_checks), cudaMemcpyDeviceToHost), cudaSuccess);

    // Test 4 tiles with coord = (tile_r, tile_c):
    // (0,0): actual_coord=(0,0), offset=0*8+0=0,   value=h_data[0]=0
    // (0,2): actual_coord=(0,4), offset=0*8+4=4,   value=h_data[4]=4
    // (2,0): actual_coord=(4,0), offset=4*8+0=32,  value=h_data[32]=32
    // (2,2): actual_coord=(4,4), offset=4*8+4=36,  value=h_data[36]=36

    // Verify offsets are correct
    EXPECT_EQ(h_checks[0], 1);  // (0,0) -> offset 0
    EXPECT_EQ(h_checks[1], 1);  // (0,2) -> offset 4
    EXPECT_EQ(h_checks[2], 1);  // (2,0) -> offset 32
    EXPECT_EQ(h_checks[3], 1);  // (2,2) -> offset 36

    // Verify values read through tile(0,0) are correct
    EXPECT_FLOAT_EQ(h_results[0], h_data[0]);   // tile(0,0) from coord(0,0) -> h_data[0]
    EXPECT_FLOAT_EQ(h_results[1], h_data[4]);   // tile(0,0) from coord(0,2) -> h_data[4]
    EXPECT_FLOAT_EQ(h_results[2], h_data[32]);  // tile(0,0) from coord(2,0) -> h_data[32]
    EXPECT_FLOAT_EQ(h_results[3], h_data[36]);  // tile(0,0) from coord(2,2) -> h_data[36]

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
    ASSERT_EQ(cudaFree(d_results), cudaSuccess);
    ASSERT_EQ(cudaFree(d_checks), cudaSuccess);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
