#include <gtest/gtest.h>

#include "../../src/layout/layout.h"
#include "../../src/layout/shape.h"
#include "../../src/layout/coordinate.h"
#include "../../src/numeric/Int.cuh"

using namespace layout;
using namespace numeric;

// =============================================================================
// Test: zipped_divide - 1D Layout
// =============================================================================

TEST(ZippedDivideTest, 1D_Basic) {
    // Input:  shape = 8, stride = 1
    // Tile:   tile = 2
    // Expected:
    //   Outer: shape = 4 (number of tiles), stride = 2 (skip one tile)
    //   Inner: shape = 2 (tile size), stride = 1 (within tile stride)
    auto shape = Int<8>{};
    auto stride = Int<1>{};
    Layout layout(shape, stride);

    auto tile = Int<2>{};
    auto result = zipped_divide(layout, tile);

    // Verify outer layout
    EXPECT_EQ(result.outer.size(), 4);  // 8 / 2 = 4 tiles
    EXPECT_EQ(result.outer(Int<1>{}), 2);  // tile 1 starts at offset 2

    // Verify inner layout
    EXPECT_EQ(result.inner.size(), 2);  // tile size = 2
    EXPECT_EQ(result.inner(Int<1>{}), 1);  // offset 1 within tile
}

TEST(ZippedDivideTest, 1D_SizeConsistency) {
    // Verify: total size is preserved
    // outer.size() * inner.size() == original.size()
    auto shape = Int<16>{};
    auto stride = Int<1>{};
    Layout layout(shape, stride);

    auto tile = Int<4>{};
    auto result = zipped_divide(layout, tile);

    EXPECT_EQ(result.outer.size() * result.inner.size(), 16);
}

TEST(ZippedDivideTest, 1D_CombinedAccess) {
    // Simulate accessing element using outer + inner coordinates
    // Element at global position 5:
    //   - outer position: 5 / 4 = 1 (second tile)
    //   - inner position: 5 % 4 = 1 (second element in tile)
    auto shape = Int<16>{};
    auto stride = Int<1>{};
    Layout layout(shape, stride);

    auto tile = Int<4>{};
    auto result = zipped_divide(layout, tile);

    // Test element at global position 5:
    // - outer position 1, inner position 1
    int outer_base = result.outer(Int<1>{});
    int inner_offset = result.inner(Int<1>{});

    // outer_base = 1 * 4 = 4 (start of second tile)
    // inner_offset = 1 (offset within tile)
    // total = 4 + 1 = 5
    EXPECT_EQ(outer_base + inner_offset, 5);
}

// =============================================================================
// Test: zipped_divide - 2D Layout
// =============================================================================

TEST(ZippedDivideTest, 2D_Basic) {
    // Input:  shape = (8, 16), stride = (1, 8) [column major]
    // Tile:   tile = (2, 4)
    // Expected:
    //   Outer: shape = (4, 4), stride = (2, 32)
    //          - 8/2=4 tiles in dim 0, stride 2 (skip tile)
    //          - 16/4=4 tiles in dim 1, stride 8*4=32 (skip tile row)
    //   Inner: shape = (2, 4), stride = (1, 8)
    //          - Original stride, unchanged within tile
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<8>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // Verify outer layout
    EXPECT_EQ(result.outer.size(), 4 * 4);  // 4x4 = 16 tiles total

    // Outer position (1, 2) means: 2nd tile in dim 0, 3rd tile in dim 1
    // Should skip 1 full tile in dim 0 (2 rows) and 2 full tile-rows in dim 1
    auto outer_coord = make_coordinate(Int<1>{}, Int<2>{});
    int outer_offset = result.outer(outer_coord);
    EXPECT_EQ(outer_offset, 1 * 2 + 2 * 32);  // = 66

    // Verify inner layout (should have original stride)
    EXPECT_EQ(result.inner.size(), 2 * 4);  // tile size = 2x4 = 8

    auto inner_coord = make_coordinate(Int<1>{}, Int<2>{});
    int inner_offset = result.inner(inner_coord);
    EXPECT_EQ(inner_offset, 1 + 2 * 8);  // = 17
}

TEST(ZippedDivideTest, 2D_SizeConsistency) {
    // Verify: total size is preserved
    auto shape = make_shape(Int<12>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<12>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<3>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // outer: 4x4 = 16 tiles, inner: 3x4 = 12 elements per tile
    EXPECT_EQ(result.outer.size() * result.inner.size(), 12 * 16);
}

TEST(ZippedDivideTest, 2D_CombinedAccess) {
    // Simulate accessing element at global position (row=7, col=13)
    // Tile shape: (3, 4)
    // Tile indices: (7/3=2, 13/4=3)
    // Inner indices: (7%3=1, 13%4=1)
    auto shape = make_shape(Int<12>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<12>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<3>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // Calculate expected offset directly for position (7, 13)
    int expected_offset = layout(make_coordinate(Int<7>{}, Int<13>{}));

    // Calculate using outer + inner
    // Outer: tile position (2, 3)
    // Inner: position within tile (1, 1)
    int outer_base = result.outer(make_coordinate(Int<2>{}, Int<3>{}));
    int inner_offset = result.inner(make_coordinate(Int<1>{}, Int<1>{}));

    EXPECT_EQ(outer_base + inner_offset, expected_offset);
}

// =============================================================================
// Test: zipped_divide - 3D Layout
// =============================================================================

TEST(ZippedDivideTest, 3D_Basic) {
    // Input:  shape = (4, 8, 16), stride = (1, 4, 32) [column major]
    // Tile:   tile = (2, 4, 4)
    // Expected:
    //   Outer: shape = (2, 2, 4), stride = (2, 16, 128)
    //   Inner: shape = (2, 4, 4), stride = (1, 4, 32)
    auto shape = make_shape(Int<4>{}, Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<4>{}, Int<32>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // Verify sizes
    EXPECT_EQ(result.outer.size(), 2 * 2 * 4);  // 16 tiles
    EXPECT_EQ(result.inner.size(), 2 * 4 * 4);   // 32 elements per tile
    EXPECT_EQ(result.outer.size() * result.inner.size(), 4 * 8 * 16);
}

TEST(ZippedDivideTest, 3D_CombinedAccess) {
    // Access element at global position (d0=3, d1=5, d2=11)
    // Tile: (2, 4, 4)
    // Outer: (3/2=1, 5/4=1, 11/4=2)
    // Inner: (3%2=1, 5%4=1, 11%4=3)
    auto shape = make_shape(Int<4>{}, Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<4>{}, Int<32>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // Expected offset for position (3, 5, 11)
    int expected = layout(make_coordinate(Int<3>{}, Int<5>{}, Int<11>{}));

    // Using outer + inner
    int outer_base = result.outer(make_coordinate(Int<1>{}, Int<1>{}, Int<2>{}));
    int inner_offset = result.inner(make_coordinate(Int<1>{}, Int<1>{}, Int<3>{}));

    EXPECT_EQ(outer_base + inner_offset, expected);
}

// =============================================================================
// Test: Compare logical_divide vs zipped_divide
// =============================================================================

TEST(DivideComparison, SameSizeDifferentRepresentation) {
    // Both should preserve total size
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<8>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});

    auto logical_result = logical_divide(layout, tile);
    auto zipped_result = zipped_divide(layout, tile);

    // logical_divide: shape = ((2, 4), (4, 4)), size = 2*4*4*4 = 128
    // zipped_divide: outer.size() * inner.size() = 16 * 8 = 128
    EXPECT_EQ(layout.size(), logical_result.size());
    EXPECT_EQ(layout.size(), zipped_result.outer.size() * zipped_result.inner.size());
}

// =============================================================================
// Test: Row Major vs Column Major - logical_divide
// =============================================================================

TEST(LogicalDivideTest, 2D_RowMajor) {
    // Input:  shape = (8, 16), stride = (16, 1) [row major]
    // Tile:   tile = (2, 4)
    // Expected:
    //   Output: shape = ((2, 4), (4, 4)), stride = ((16, 64), (1, 2))
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<16>{}, Int<1>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = logical_divide(layout, tile);

    result.print();
    EXPECT_EQ(result.size(), 8 * 16);
}

TEST(LogicalDivideTest, 2D_ColumnMajor) {
    // Input:  shape = (8, 16), stride = (1, 8) [column major]
    // Tile:   tile = (2, 4)
    // Expected:
    //   Output: shape = ((2, 4), (4, 4)), stride = ((1, 2), (8, 32))
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<8>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = logical_divide(layout, tile);

    result.print();
    EXPECT_EQ(result.size(), 8 * 16);
}

TEST(LogicalDivideTest, 2D_RowVsColumn_SameShape) {
    // Verify both preserve size, but produce different strides
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto tile = make_shape(Int<2>{}, Int<4>{});

    // Row major
    auto stride_row = make_stride(Int<16>{}, Int<1>{});
    Layout layout_row(shape, stride_row);
    auto result_row = logical_divide(layout_row, tile);

    // Column major
    auto stride_col = make_stride(Int<1>{}, Int<8>{});
    Layout layout_col(shape, stride_col);
    auto result_col = logical_divide(layout_col, tile);

    // Both should preserve total size
    EXPECT_EQ(result_row.size(), result_col.size());
    EXPECT_EQ(result_row.size(), 8 * 16);
}

// =============================================================================
// Test: Row Major vs Column Major - zipped_divide
// =============================================================================

TEST(ZippedDivideTest, 2D_RowMajor_Basic) {
    // Input:  shape = (8, 16), stride = (16, 1) [row major]
    // Tile:   tile = (2, 4)
    // Expected:
    //   Outer: shape = (4, 4), stride = (32, 4)
    //          - 8/2=4 tiles in dim 0, stride 16*2=32 (skip tile)
    //          - 16/4=4 tiles in dim 1, stride 1*4=4 (skip tile column)
    //   Inner: shape = (2, 4), stride = (16, 1)
    //          - Original stride, unchanged within tile
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<16>{}, Int<1>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // Verify outer layout
    EXPECT_EQ(result.outer.size(), 4 * 4);  // 4x4 = 16 tiles total

    // Outer position (1, 2) in row major:
    // - dim 0: 1 * 32 = 32
    // - dim 1: 2 * 4 = 8
    // Total: 32 + 8 = 40
    auto outer_coord = make_coordinate(Int<1>{}, Int<2>{});
    int outer_offset = result.outer(outer_coord);
    EXPECT_EQ(outer_offset, 40);

    // Verify inner layout (should have original stride)
    EXPECT_EQ(result.inner.size(), 2 * 4);  // tile size = 2x4 = 8

    auto inner_coord = make_coordinate(Int<1>{}, Int<2>{});
    int inner_offset = result.inner(inner_coord);
    EXPECT_EQ(inner_offset, 1 * 16 + 2 * 1);  // = 18
}

TEST(ZippedDivideTest, 2D_ColumnMajor_Basic) {
    // Input:  shape = (8, 16), stride = (1, 8) [column major]
    // Tile:   tile = (2, 4)
    // Expected:
    //   Outer: shape = (4, 4), stride = (2, 32)
    //   Inner: shape = (2, 4), stride = (1, 8)
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<8>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = zipped_divide(layout, tile);

    // Verify outer layout
    EXPECT_EQ(result.outer.size(), 4 * 4);  // 4x4 = 16 tiles total

    // Outer position (1, 2) in column major:
    // - dim 0: 1 * 2 = 2
    // - dim 1: 2 * 32 = 64
    // Total: 2 + 64 = 66
    auto outer_coord = make_coordinate(Int<1>{}, Int<2>{});
    int outer_offset = result.outer(outer_coord);
    EXPECT_EQ(outer_offset, 66);

    // Verify inner layout (should have original stride)
    EXPECT_EQ(result.inner.size(), 2 * 4);  // tile size = 2x4 = 8

    auto inner_coord = make_coordinate(Int<1>{}, Int<2>{});
    int inner_offset = result.inner(inner_coord);
    EXPECT_EQ(inner_offset, 1 * 1 + 2 * 8);  // = 17
}

TEST(ZippedDivideTest, 2D_RowVsColumn_CombinedAccess) {
    // Test that row major and column major produce correct combined offsets
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto tile = make_shape(Int<2>{}, Int<4>{});

    // Test element at global position (5, 11)
    auto global_coord = make_coordinate(Int<5>{}, Int<11>{});

    // Row major version
    auto stride_row = make_stride(Int<16>{}, Int<1>{});
    Layout layout_row(shape, stride_row);
    auto result_row = zipped_divide(layout_row, tile);

    int expected_row = layout_row(global_coord);
    int outer_row = result_row.outer(make_coordinate(Int<2>{}, Int<2>{}));  // 5/2=2, 11/4=2
    int inner_row = result_row.inner(make_coordinate(Int<1>{}, Int<3>{}));  // 5%2=1, 11%4=3
    EXPECT_EQ(outer_row + inner_row, expected_row);

    // Column major version
    auto stride_col = make_stride(Int<1>{}, Int<8>{});
    Layout layout_col(shape, stride_col);
    auto result_col = zipped_divide(layout_col, tile);

    int expected_col = layout_col(global_coord);
    int outer_col = result_col.outer(make_coordinate(Int<2>{}, Int<2>{}));  // 5/2=2, 11/4=2
    int inner_col = result_col.inner(make_coordinate(Int<1>{}, Int<3>{}));  // 5%2=1, 11%4=3
    EXPECT_EQ(outer_col + inner_col, expected_col);

    // The actual offsets should differ between row/column major
    EXPECT_NE(expected_row, expected_col);
}

TEST(ZippedDivideTest, 2D_RowVsColumn_SizeConsistency) {
    // Both should preserve total size regardless of memory layout
    auto shape = make_shape(Int<12>{}, Int<16>{});
    auto tile = make_shape(Int<3>{}, Int<4>{});

    // Row major
    auto stride_row = make_stride(Int<16>{}, Int<1>{});
    Layout layout_row(shape, stride_row);
    auto result_row = zipped_divide(layout_row, tile);

    // Column major
    auto stride_col = make_stride(Int<1>{}, Int<12>{});
    Layout layout_col(shape, stride_col);
    auto result_col = zipped_divide(layout_col, tile);

    // All should have same total size
    EXPECT_EQ(result_row.outer.size() * result_row.inner.size(), 12 * 16);
    EXPECT_EQ(result_col.outer.size() * result_col.inner.size(), 12 * 16);
    EXPECT_EQ(result_row.outer.size() * result_row.inner.size(),
              result_col.outer.size() * result_col.inner.size());
}

TEST(ZippedDivideTest, 2D_SquareMatrix_RowVsColumn) {
    // Test with square matrix where row/col major have interesting properties
    auto shape = make_shape(Int<16>{}, Int<16>{});
    auto tile = make_shape(Int<4>{}, Int<4>{});

    // Row major
    auto stride_row = make_stride(Int<16>{}, Int<1>{});
    Layout layout_row(shape, stride_row);
    auto result_row = zipped_divide(layout_row, tile);

    // Column major
    auto stride_col = make_stride(Int<1>{}, Int<16>{});
    Layout layout_col(shape, stride_col);
    auto result_col = zipped_divide(layout_col, tile);

    // Both should produce 4x4 = 16 tiles
    EXPECT_EQ(result_row.outer.size(), 16);
    EXPECT_EQ(result_col.outer.size(), 16);

    // Each tile should be 4x4 = 16 elements
    EXPECT_EQ(result_row.inner.size(), 16);
    EXPECT_EQ(result_col.inner.size(), 16);

    // Test diagonal element (5, 5)
    auto coord = make_coordinate(Int<5>{}, Int<5>{});
    int row_offset = layout_row(coord);
    int col_offset = layout_col(coord);

    // For square matrices, diagonal elements have same offset in both layouts
    EXPECT_EQ(row_offset, col_offset);

    // Test using zipped_divide
    int outer_row = result_row.outer(make_coordinate(Int<1>{}, Int<1>{}));  // 5/4=1, 5/4=1
    int inner_row = result_row.inner(make_coordinate(Int<1>{}, Int<1>{}));  // 5%4=1, 5%4=1
    EXPECT_EQ(outer_row + inner_row, row_offset);

    int outer_col = result_col.outer(make_coordinate(Int<1>{}, Int<1>{}));
    int inner_col = result_col.inner(make_coordinate(Int<1>{}, Int<1>{}));
    EXPECT_EQ(outer_col + inner_col, col_offset);
}

// =============================================================================
// Test: Edge Cases
// =============================================================================

TEST(ZippedDivideTest, EdgeCase_TileSize1) {
    // Tile size = 1 means each element is its own tile
    auto shape = Int<8>{};
    auto stride = Int<1>{};
    Layout layout(shape, stride);

    auto tile = Int<1>{};
    auto result = zipped_divide(layout, tile);

    EXPECT_EQ(result.outer.size(), 8);  // 8 tiles
    EXPECT_EQ(result.inner.size(), 1);  // 1 element per tile
}

TEST(ZippedDivideTest, EdgeCase_TileEqualsSize) {
    // Tile size = total size means single tile containing everything
    auto shape = Int<8>{};
    auto stride = Int<1>{};
    Layout layout(shape, stride);

    auto tile = Int<8>{};
    auto result = zipped_divide(layout, tile);

    EXPECT_EQ(result.outer.size(), 1);  // 1 tile
    EXPECT_EQ(result.inner.size(), 8);  // 8 elements in this tile
}

TEST(ZippedDivideTest, EdgeCase_PowerOfTwo) {
    // Test with power of 2 sizes (common in GPU)
    auto shape = make_shape(Int<128>{}, Int<64>{});
    auto stride = make_stride(Int<1>{}, Int<128>{});
    Layout layout(shape, stride);

    auto tile = make_shape(Int<32>{}, Int<16>{});
    auto result = zipped_divide(layout, tile);

    EXPECT_EQ(result.outer.size(), 4 * 4);   // (128/32) * (64/16) = 4 * 4
    EXPECT_EQ(result.inner.size(), 32 * 16); // tile size
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
