#include <gtest/gtest.h>

#include "../../../src/layout/layout.h"
#include "../../../src/layout/shape.h"
#include "../../../src/layout/coordinate.h"
#include "../../../src/numeric/Int.cuh"

using namespace layout;
using namespace numeric;

// =============================================================================
// Test: Layout size()
// =============================================================================

TEST(LayoutTest, Size_1D) {
    // 1D layout: shape = 8
    auto shape = make_shape(Int<8>{});
    auto stride = make_stride(Int<1>{});
    Layout l(shape, stride);

    EXPECT_EQ(l.size(), 8);
}

TEST(LayoutTest, Size_2D) {
    // 2D layout: shape = (3, 4), size = 12
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<1>{}, Int<3>{});
    Layout l(shape, stride);

    EXPECT_EQ(l.size(), 12);
}

TEST(LayoutTest, Size_3D) {
    // 3D layout: shape = (2, 3, 4), size = 24
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<1>{}, Int<2>{}, Int<6>{});
    Layout l(shape, stride);

    EXPECT_EQ(l.size(), 24);
}

TEST(LayoutTest, Size_NestedShape) {
    // Nested shape: ((2, 3), (4, 5)), size = 2 * 3 * 4 * 5 = 120
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), make_shape(Int<4>{}, Int<5>{}));
    auto stride = make_stride(make_stride(Int<1>{}, Int<2>{}), make_stride(Int<6>{}, Int<30>{}));
    Layout l(shape, stride);

    EXPECT_EQ(l.size(), 120);
}

// =============================================================================
// Test: Layout operator()
// =============================================================================

TEST(LayoutTest, Operator_1D) {
    // 1D layout: shape = 8, stride = 1
    auto shape = make_shape(Int<8>{});
    auto stride = make_stride(Int<1>{});
    Layout l(shape, stride);

    // coord_to_idx for 1D is simply coord * stride
    auto coord = make_coordinate(Int<5>{});
    EXPECT_EQ(l(coord), 5);
}

TEST(LayoutTest, Operator_2D_ColumnMajor) {
    // 2D layout: shape = (3, 4), column major stride = (1, 3)
    // Matrix layout:
    //   (0,0)=0  (0,1)=3  (0,2)=6  (0,3)=9
    //   (1,0)=1  (1,1)=4  (1,2)=7  (1,3)=10
    //   (2,0)=2  (2,1)=5  (2,2)=8  (2,3)=11
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<1>{}, Int<3>{});
    Layout l(shape, stride);

    // Test various coordinates: idx = row * 1 + col * 3
    auto coord0 = make_coordinate(Int<0>{}, Int<0>{});
    EXPECT_EQ(l(coord0), 0);

    auto coord1 = make_coordinate(Int<1>{}, Int<0>{});
    EXPECT_EQ(l(coord1), 1);

    auto coord2 = make_coordinate(Int<2>{}, Int<0>{});
    EXPECT_EQ(l(coord2), 2);

    auto coord3 = make_coordinate(Int<0>{}, Int<1>{});
    EXPECT_EQ(l(coord3), 3);

    auto coord4 = make_coordinate(Int<1>{}, Int<1>{});
    EXPECT_EQ(l(coord4), 4);

    auto coord5 = make_coordinate(Int<2>{}, Int<2>{});
    EXPECT_EQ(l(coord5), 8);

    auto coord6 = make_coordinate(Int<2>{}, Int<3>{});
    EXPECT_EQ(l(coord6), 11);
}

TEST(LayoutTest, Operator_3D_ColumnMajor) {
    // 3D layout: shape = (2, 3, 4), column major stride = (1, 2, 6)
    // idx = i * 1 + j * 2 + k * 6
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<1>{}, Int<2>{}, Int<6>{});
    Layout l(shape, stride);

    auto coord0 = make_coordinate(Int<0>{}, Int<0>{}, Int<0>{});
    EXPECT_EQ(l(coord0), 0);

    auto coord1 = make_coordinate(Int<1>{}, Int<0>{}, Int<0>{});
    EXPECT_EQ(l(coord1), 1);

    auto coord2 = make_coordinate(Int<0>{}, Int<1>{}, Int<0>{});
    EXPECT_EQ(l(coord2), 2);

    auto coord3 = make_coordinate(Int<1>{}, Int<2>{}, Int<1>{});
    EXPECT_EQ(l(coord3), 1 + 2 * 2 + 1 * 6);  // 11
}

TEST(LayoutTest, Operator_4D_ColumnMajor) {
    // 4D layout: shape = (2, 2, 3, 4), column major stride = (1, 2, 4, 12)
    // idx = d0 * 1 + d1 * 2 + d2 * 4 + d3 * 12
    auto shape = make_shape(Int<2>{}, Int<2>{}, Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<1>{}, Int<2>{}, Int<4>{}, Int<12>{});
    Layout l(shape, stride);

    auto coord = make_coordinate(Int<1>{}, Int<1>{}, Int<2>{}, Int<1>{});
    EXPECT_EQ(l(coord), 1 * 1 + 1 * 2 + 2 * 4 + 1 * 12);  // 21
}

TEST(LayoutTest, Operator_NestedCoord) {
    // Nested coordinate with nested shape
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), make_shape(Int<4>{}, Int<5>{}));
    auto stride = make_stride(make_stride(Int<1>{}, Int<2>{}), make_stride(Int<6>{}, Int<30>{}));
    Layout l(shape, stride);

    auto coord = make_coordinate(make_shape(Int<1>{}, Int<2>{}), make_shape(Int<3>{}, Int<4>{}));
    // idx = (1 * 1 + 2 * 2) + (3 * 6 + 4 * 30) = 5 + 138 = 143
    EXPECT_EQ(l(coord), 143);
}

// =============================================================================
// Test: Row Major vs Column Major Layout
// =============================================================================

TEST(LayoutTest, Operator_2D_RowMajor) {
    // 2D layout: shape = (3, 4), row major stride = (4, 1)
    // Matrix layout in memory (row by row):
    //   (0,0)=0  (0,1)=1  (0,2)=2  (0,3)=3
    //   (1,0)=4  (1,1)=5  (1,2)=6  (1,3)=7
    //   (2,0)=8  (2,1)=9  (2,2)=10 (2,3)=11
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<4>{}, Int<1>{});
    Layout l(shape, stride);

    // Test various coordinates: idx = row * 4 + col
    auto coord0 = make_coordinate(Int<0>{}, Int<0>{});
    EXPECT_EQ(l(coord0), 0);

    auto coord1 = make_coordinate(Int<0>{}, Int<1>{});
    EXPECT_EQ(l(coord1), 1);

    auto coord2 = make_coordinate(Int<0>{}, Int<3>{});
    EXPECT_EQ(l(coord2), 3);

    auto coord3 = make_coordinate(Int<1>{}, Int<0>{});
    EXPECT_EQ(l(coord3), 4);

    auto coord4 = make_coordinate(Int<1>{}, Int<1>{});
    EXPECT_EQ(l(coord4), 5);

    auto coord5 = make_coordinate(Int<2>{}, Int<2>{});
    EXPECT_EQ(l(coord5), 10);

    auto coord6 = make_coordinate(Int<2>{}, Int<3>{});
    EXPECT_EQ(l(coord6), 11);
}

TEST(LayoutTest, Operator_2D_RowMajorVsColumnMajor) {
    // Compare row major and column major for the same matrix
    auto shape = make_shape(Int<3>{}, Int<4>{});

    // Row major: stride = (4, 1), idx = row * 4 + col
    auto stride_row = make_stride(Int<4>{}, Int<1>{});
    Layout l_row(shape, stride_row);

    // Column major: stride = (1, 3), idx = row + col * 3
    auto stride_col = make_stride(Int<1>{}, Int<3>{});
    Layout l_col(shape, stride_col);

    // Test coordinate (1, 2)
    auto coord = make_coordinate(Int<1>{}, Int<2>{});
    EXPECT_EQ(l_row(coord), 1 * 4 + 2);  // 6
    EXPECT_EQ(l_col(coord), 1 + 2 * 3);  // 7

    // Test coordinate (2, 1)
    auto coord2 = make_coordinate(Int<2>{}, Int<1>{});
    EXPECT_EQ(l_row(coord2), 2 * 4 + 1);  // 9
    EXPECT_EQ(l_col(coord2), 2 + 1 * 3);  // 5
}

TEST(LayoutTest, Operator_2D_RowMajor_CornerCases) {
    // Test edge cases for row major layout
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<4>{}, Int<1>{});
    Layout l(shape, stride);

    // First element
    auto first = make_coordinate(Int<0>{}, Int<0>{});
    EXPECT_EQ(l(first), 0);

    // Last element
    auto last = make_coordinate(Int<2>{}, Int<3>{});
    EXPECT_EQ(l(last), 11);

    // Entire first row
    EXPECT_EQ(l(make_coordinate(Int<0>{}, Int<0>{})), 0);
    EXPECT_EQ(l(make_coordinate(Int<0>{}, Int<1>{})), 1);
    EXPECT_EQ(l(make_coordinate(Int<0>{}, Int<2>{})), 2);
    EXPECT_EQ(l(make_coordinate(Int<0>{}, Int<3>{})), 3);

    // Entire last row
    EXPECT_EQ(l(make_coordinate(Int<2>{}, Int<0>{})), 8);
    EXPECT_EQ(l(make_coordinate(Int<2>{}, Int<1>{})), 9);
    EXPECT_EQ(l(make_coordinate(Int<2>{}, Int<2>{})), 10);
    EXPECT_EQ(l(make_coordinate(Int<2>{}, Int<3>{})), 11);
}

TEST(LayoutTest, Operator_2D_ColumnMajor_CornerCases) {
    // Test edge cases for column major layout
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_stride(Int<1>{}, Int<3>{});
    Layout l(shape, stride);

    // First element
    auto first = make_coordinate(Int<0>{}, Int<0>{});
    EXPECT_EQ(l(first), 0);

    // Last element
    auto last = make_coordinate(Int<2>{}, Int<3>{});
    EXPECT_EQ(l(last), 11);

    // Entire first column
    EXPECT_EQ(l(make_coordinate(Int<0>{}, Int<0>{})), 0);
    EXPECT_EQ(l(make_coordinate(Int<1>{}, Int<0>{})), 1);
    EXPECT_EQ(l(make_coordinate(Int<2>{}, Int<0>{})), 2);

    // Entire last column
    EXPECT_EQ(l(make_coordinate(Int<0>{}, Int<3>{})), 9);
    EXPECT_EQ(l(make_coordinate(Int<1>{}, Int<3>{})), 10);
    EXPECT_EQ(l(make_coordinate(Int<2>{}, Int<3>{})), 11);
}

TEST(LayoutTest, Operator_2D_SquareMatrix_RowVsCol) {
    // Test square matrix (special case where row/col major might coincide for some elements)
    auto shape = make_shape(Int<4>{}, Int<4>{});

    // Row major: stride = (4, 1)
    auto stride_row = make_stride(Int<4>{}, Int<1>{});
    Layout l_row(shape, stride_row);

    // Column major: stride = (1, 4)
    auto stride_col = make_stride(Int<1>{}, Int<4>{});
    Layout l_col(shape, stride_col);

    // Diagonal elements should be the same
    auto coord0 = make_coordinate(Int<0>{}, Int<0>{});
    EXPECT_EQ(l_row(coord0), l_col(coord0));

    auto coord1 = make_coordinate(Int<1>{}, Int<1>{});
    EXPECT_EQ(l_row(coord1), l_col(coord1));

    auto coord2 = make_coordinate(Int<2>{}, Int<2>{});
    EXPECT_EQ(l_row(coord2), l_col(coord2));

    auto coord3 = make_coordinate(Int<3>{}, Int<3>{});
    EXPECT_EQ(l_row(coord3), l_col(coord3));

    // Off-diagonal elements should differ
    auto coord = make_coordinate(Int<1>{}, Int<2>{});
    EXPECT_NE(l_row(coord), l_col(coord));
}

// =============================================================================
// Test: make_compact_stride with Layout
// =============================================================================

TEST(LayoutTest, MakeCompactStride_2D) {
    // Test compact stride creation for 2D (column major)
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_compact_stride(shape);
    Layout l(shape, stride);

    // Column major: stride = (1, 3)
    EXPECT_EQ(std::get<0>(stride), 1);
    EXPECT_EQ(std::get<1>(stride), 3);

    auto coord = make_coordinate(Int<2>{}, Int<3>{});
    EXPECT_EQ(l(coord), 2 + 3 * 3);  // 11
}

TEST(LayoutTest, MakeCompactStride_3D) {
    // Test compact stride creation for 3D (column major)
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto stride = make_compact_stride(shape);
    Layout l(shape, stride);

    // Column major: stride = (1, 2, 6)
    EXPECT_EQ(std::get<0>(stride), 1);
    EXPECT_EQ(std::get<1>(stride), 2);
    EXPECT_EQ(std::get<2>(stride), 6);
}

// =============================================================================
// Test: logical_divide
// =============================================================================

TEST(LayoutTest, LogicalDivide_1D) {
    // Input:  shape = 8, stride = 1
    // Tile:   tile = 2
    // Output: shape = (2, 4), stride = (1, 2)
    // Size:   2 * 4 = 8 (same as input)
    auto shape = Int<8>{};
    auto stride = Int<1>{};
    Layout l(shape, stride);

    auto tile = Int<2>{};
    auto result = logical_divide(l, tile);

    result.print(); // Expected: Shape: (_2,_4) Stride: (_1,_2)

    EXPECT_EQ(result.size(), 8);
}

TEST(LayoutTest, LogicalDivide_2D) {
    // Input:  shape = (8, 16), stride = (1, 8)
    // Tile:   tile = (2, 4)
    // Output: shape = ((2, 4), (4, 4)), stride = ((1, 2), (8, 32))
    // Size:   2 * 4 * 4 * 4 = 128 (same as input 8 * 16)
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<8>{});
    Layout l(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = logical_divide(l, tile);

    result.print(); // Expected: Shape: ((_2,_4),(_4,_4)) Stride: ((_1,_2),(_8,_32))

    EXPECT_EQ(result.size(), 128);
}

TEST(LayoutTest, LogicalDivide_3D) {
    // Input:  shape = (4, 8, 16), stride = (1, 4, 32)
    // Tile:   tile = (2, 4, 4)
    // Output: shape = ((2, 2), (4, 2), (4, 4)), stride = ((1, 2), (4, 16), (32, 128))
    // Size:   2 * 2 * 4 * 2 * 4 * 4 = 512 (same as input 4 * 8 * 16)
    auto shape = make_shape(Int<4>{}, Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<4>{}, Int<32>{});
    Layout l(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{}, Int<4>{});
    auto result = logical_divide(l, tile);

    result.print(); // Expected: Shape: ((_2,_2),(_4,_2),(_4,_4)) Stride: ((_1,_2),(_4,_16),(_32,_128))

    EXPECT_EQ(result.size(), 512);
}

TEST(LayoutTest, LogicalDivide_SizeConsistency) {
    // Verify: size(layout) == size(logical_divide(layout, tile))
    // logical_divide does not change the total size, only reorganizes the layout
    auto shape = make_shape(Int<8>{}, Int<16>{});
    auto stride = make_stride(Int<1>{}, Int<8>{});
    Layout l(shape, stride);

    auto tile = make_shape(Int<2>{}, Int<4>{});
    auto result = logical_divide(l, tile);

    l.print();     // Expected: Shape: (_8,_16) Stride: (_1,_8)
    result.print(); // Expected: Shape: ((_2,_4),(_4,_4)) Stride: ((_1,_2),(_8,_32))

    EXPECT_EQ(l.size(), result.size());
}

// =============================================================================
// Test: Combined Layout Operations
// =============================================================================

TEST(LayoutTest, Layout_CompleteWorkflow) {
    // Input:  shape = (4, 6), stride = (1, 4) [column major]
    // Tile:   tile = (2, 3)
    // Output: shape = ((2, 2), (3, 2)), stride = ((1, 2), (4, 12))
    // Size:   2 * 2 * 3 * 2 = 24 (same as input 4 * 6)
    auto shape = make_shape(Int<4>{}, Int<6>{});
    auto stride = make_compact_stride(shape);
    Layout l(shape, stride);

    l.print(); // Expected: Shape: (_4,_6) Stride: (_1,_4)

    EXPECT_EQ(l.size(), 24);

    auto coord = make_coordinate(Int<3>{}, Int<5>{});
    EXPECT_EQ(l(coord), 3 + 5 * 4);  // 23 (last element)

    auto tile = make_shape(Int<2>{}, Int<3>{});
    auto divided = logical_divide(l, tile);

    divided.print(); // Expected: Shape: ((_2,_2),(_3,_2)) Stride: ((_1,_2),(_4,_12))

    EXPECT_EQ(divided.size(), 24);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
