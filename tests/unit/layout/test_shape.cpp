#include <gtest/gtest.h>

#include "../../../src/layout/shape.h"
#include "../../../src/numeric/Int.cuh"

using namespace layout;
using namespace numeric;

// =============================================================================
// Test: get_shape_size
// =============================================================================

TEST(ShapeTest, GetShapeSize_SingleInt) {
    // Test single int value
    auto shape = 5;
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 5);
}

TEST(ShapeTest, GetShapeSize_SingleIntConstant) {
    // Test single Int<> constant
    auto shape = Int<5>{};
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 5);
}

TEST(ShapeTest, GetShapeSize_FlatTuple) {
    // Test flat tuple: (2, 3, 4) -> size = 24
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 24);
}

TEST(ShapeTest, GetShapeSize_NestedTuple) {
    // Test nested tuple: ((2, 3), (4, 5)) -> size = 2 * 3 * 4 * 5 = 120
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), make_shape(Int<4>{}, Int<5>{}));
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 120);
}

TEST(ShapeTest, GetShapeSize_DeeplyNestedTuple) {
    // Test deeply nested tuple: (((2, 3)), 4) -> size = 2 * 3 * 4 = 24
    auto shape = make_shape(make_shape(make_shape(Int<2>{}, Int<3>{})), Int<4>{});
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 24);
}

TEST(ShapeTest, GetShapeSize_MixedIntAndIntConstant) {
    // Test mixed int and Int<>
    auto shape = make_shape(2, Int<3>{}, 4);
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 24);
}

TEST(ShapeTest, GetShapeSize_SingleElementTuple) {
    // Test single element tuple
    auto shape = make_shape(Int<7>{});
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 7);
}

TEST(ShapeTest, GetShapeSize_TupleWithOnes) {
    // Test tuple containing ones: (1, 5, 1) -> size = 5
    auto shape = make_shape(Int<1>{}, Int<5>{}, Int<1>{});
    auto size = get_shape_size(shape);
    EXPECT_EQ(size, 5);
}

// =============================================================================
// Test: prefix_product_size
// =============================================================================

TEST(ShapeTest, PrefixProductSize_Index0) {
    // prefix_product_size<0> should always be 1
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto result = prefix_product_size<0>(shape);
    EXPECT_EQ(result, 1);
}

TEST(ShapeTest, PrefixProductSize_Index1) {
    // prefix_product_size<1> = shape[0] = 2
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto result = prefix_product_size<1>(shape);
    EXPECT_EQ(result, 2);
}

TEST(ShapeTest, PrefixProductSize_Index2) {
    // prefix_product_size<2> = shape[0] * shape[1] = 2 * 3 = 6
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto result = prefix_product_size<2>(shape);
    EXPECT_EQ(result, 6);
}

TEST(ShapeTest, PrefixProductSize_Index3) {
    // prefix_product_size<3> = shape[0] * shape[1] * shape[2] = 2 * 3 * 4 = 24
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto result = prefix_product_size<3>(shape);
    EXPECT_EQ(result, 24);
}

TEST(ShapeTest, PrefixProductSize_NestedShape) {
    // Test with nested shapes: ((2, 3), 4)
    // prefix_product_size<0> = 1
    // prefix_product_size<1> = 1
    // prefix_product_size<2> = 2 * 3 = 6
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), Int<4>{});
    auto result0 = prefix_product_size<0>(shape);
    auto result1 = prefix_product_size<1>(shape);
    auto result2 = prefix_product_size<2>(shape);

    EXPECT_EQ(result0, 1);
    EXPECT_EQ(result1, 6);
    EXPECT_EQ(result2, 24);
}

// =============================================================================
// Test: make_compact_stride
// =============================================================================

TEST(ShapeTest, MakeCompactStride_SingleElement) {
    // Single element shape should have stride = 1
    auto shape = make_shape(Int<5>{});
    auto stride = make_compact_stride(shape);
    EXPECT_EQ(std::get<0>(stride), 1);
}

TEST(ShapeTest, MakeCompactStride_1D) {
    // 1D shape: (8) -> stride: (1)
    auto shape = make_shape(Int<8>{});
    auto stride = make_compact_stride(shape);
    EXPECT_EQ(std::get<0>(stride), 1);
}

TEST(ShapeTest, MakeCompactStride_2D) {
    auto shape = make_shape(Int<3>{}, Int<4>{});
    auto stride = make_compact_stride(shape);
    EXPECT_EQ(std::get<0>(stride), 1);
    EXPECT_EQ(std::get<1>(stride), 3);
}

TEST(ShapeTest, MakeCompactStride_3D) {
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto stride = make_compact_stride(shape);
    EXPECT_EQ(std::get<0>(stride), 1);
    EXPECT_EQ(std::get<1>(stride), 2);
    EXPECT_EQ(std::get<2>(stride), 6);
}

TEST(ShapeTest, MakeCompactStride_4D) {
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{}, Int<5>{});
    auto stride = make_compact_stride(shape);
    EXPECT_EQ(std::get<0>(stride), 1);
    EXPECT_EQ(std::get<1>(stride), 2);
    EXPECT_EQ(std::get<2>(stride), 6);
    EXPECT_EQ(std::get<3>(stride), 24);
}

TEST(ShapeTest, MakeCompactStride_NestedShape) {
    auto shape = make_shape(make_shape(Int<2>{}, Int<3>{}), make_shape(Int<4>{}, Int<5>{}));
    auto stride = make_compact_stride(shape); // (1 , 2), (6, 24)

    auto inner_stride_0 = std::get<0>(stride);
    auto inner_stride_1 = std::get<1>(stride);

    EXPECT_EQ(std::get<0>(inner_stride_0), 1);
    EXPECT_EQ(std::get<1>(inner_stride_0), 2);
    EXPECT_EQ(std::get<0>(inner_stride_1), 6);
    EXPECT_EQ(std::get<1>(inner_stride_1), 24);
}

TEST(ShapeTest, MakeCompactStride_ShapeWithOnes) {
    // Shape with ones: (1, 5, 1) -> stride: (1, 1, 5) col major
    auto shape = make_shape(Int<1>{}, Int<5>{}, Int<1>{});
    auto stride = make_compact_stride(shape);
    EXPECT_EQ(std::get<0>(stride), 1);
    EXPECT_EQ(std::get<1>(stride), 1);
    EXPECT_EQ(std::get<2>(stride), 5);
}

TEST(ShapeTest, MakeCompactStride) {
    auto shape = make_shape(make_shape(Int<4>{}, Int<5>{}), Int<4>{}, Int<5>{}, Int<1>{});
    auto stride = make_compact_stride(shape); // ((1, 4), 20, 80, 400)
    EXPECT_EQ(std::get<1>(std::get<0>(stride)), 4);
    EXPECT_EQ(std::get<1>(stride), 20);
    EXPECT_EQ(std::get<3>(stride), 400);
}


// =============================================================================
// Test: Combined Size and Stride
// =============================================================================

TEST(ShapeTest, SizeAndStrided_Consistency) {
    // Verify that total size matches stride computation
    auto shape = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto size = get_shape_size(shape);
    auto stride = make_compact_stride(shape); // (1, 2, 6)

    // Total elements should be 24
    EXPECT_EQ(size, 24);

    // First stride times first dimension should give total size
    EXPECT_EQ(std::get<2>(stride), 6);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
