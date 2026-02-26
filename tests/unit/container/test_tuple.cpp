#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>

// Minimal DEVICE macro for CPU testing
#define DEVICE inline

// Define numeric namespace inline to avoid config.cuh dependency
namespace numeric {

template <int N>
using Int = std::integral_constant<int, N>;

template <typename A, typename B>
DEVICE constexpr auto mixed_mul(A a, B b) {
    if constexpr (std::is_same_v<A, int> || std::is_same_v<B, int>) {
        return (int)a * (int)b;
    } else {
        return Int<A::value * B::value>{};
    }
}

} // namespace numeric

// Define container functions inline to avoid config.cuh dependency

namespace container {

template <typename T>
struct is_tuple : std::false_type {};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

template <typename Tuple, typename Func, size_t... Is>
DEVICE constexpr void tuple_for_each_impl(Tuple&& t, Func&& f, std::index_sequence<Is...>) {
    (f(std::get<Is>(t)), ...);
}

template <typename Tuple, typename Func>
DEVICE constexpr void tuple_for_each(Tuple&& t, Func&& f) {
    constexpr auto size = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    tuple_for_each_impl(std::forward<Tuple>(t), std::forward<Func>(f), std::make_index_sequence<size>{});
}

template <typename... Ts>
using Tuple = std::tuple<Ts...>;

template <typename A, typename B>
DEVICE constexpr auto dot_product(A a, B b) {
    if constexpr (!is_tuple_v<A>) {
        return numeric::mixed_mul(a, b);
    } else {
        static_assert(is_tuple_v<A> && is_tuple_v<B>, "dot_product only accepts tuples");
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>, "dot_product only accepts tuples of the same size");
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return (numeric::Int<0>{} + ... + dot_product(std::get<Is>(a), std::get<Is>(b)));
        }(std::make_index_sequence<std::tuple_size_v<A>>{});
    }
}

template <typename A, typename B>
DEVICE constexpr auto tuple_add(A a, B b) {
    if constexpr (!is_tuple_v<A> && !is_tuple_v<B>) {
        return a + b;
    } else if constexpr (!is_tuple_v<A>) {
        static_assert(std::tuple_size_v<B> == 1, "Scalar-tuple addition requires tuple size 1");
        return std::make_tuple(a + std::get<0>(b));
    } else if constexpr (!is_tuple_v<B>) {
        static_assert(std::tuple_size_v<A> == 1, "Tuple-scalar addition requires tuple size 1");
        return std::make_tuple(std::get<0>(a) + b);
    } else {
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>, "Tuple addition requires same size");
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(tuple_add(std::get<Is>(a), std::get<Is>(b))...);
        }(std::make_index_sequence<std::tuple_size_v<A>>{});
    }
}

template <typename A, typename B>
DEVICE constexpr auto tuple_scale(A a, B b) {
    if constexpr (!is_tuple_v<A> && !is_tuple_v<B>) {
        return a * b;
    } else if constexpr (!is_tuple_v<A>) {
        static_assert(std::tuple_size_v<B> == 1, "Scalar-tuple scale requires tuple size 1");
        return std::make_tuple(a * std::get<0>(b));
    } else if constexpr (!is_tuple_v<B>) {
        static_assert(std::tuple_size_v<A> == 1, "Tuple-scalar scale requires tuple size 1");
        return std::make_tuple(std::get<0>(a) * b);
    } else {
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>, "Tuple scale requires same size");
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(tuple_scale(std::get<Is>(a), std::get<Is>(b))...);
        }(std::make_index_sequence<std::tuple_size_v<A>>{});
    }
}

} // namespace container

using namespace container;
using namespace numeric;

// =============================================================================
// Test: tuple_scale
// =============================================================================

TEST(TupleScaleTest, ScalarScalar) {
    // Test scalar * scalar
    auto result = tuple_scale(3, 4);
    EXPECT_EQ(result, 12);
}

TEST(TupleScaleTest, IntConstantScalar) {
    // Test Int<> * scalar
    auto result = tuple_scale(Int<3>{}, 4);
    EXPECT_EQ(result, 12);
}

TEST(TupleScaleTest, ScalarIntConstant) {
    // Test scalar * Int<>
    auto result = tuple_scale(3, Int<4>{});
    EXPECT_EQ(result, 12);
}

TEST(TupleScaleTest, IntConstantIntConstant) {
    // Test Int<> * Int<>
    auto result = tuple_scale(Int<3>{}, Int<4>{});
    EXPECT_EQ(result, 12);
}

TEST(TupleScaleTest, ScalarTupleSingle) {
    // Test scalar * single-element tuple
    auto t = std::make_tuple(5);
    auto result = tuple_scale(3, t);
    EXPECT_EQ(std::get<0>(result), 15);
}

TEST(TupleScaleTest, TupleSingleScalar) {
    // Test single-element tuple * scalar
    auto t = std::make_tuple(5);
    auto result = tuple_scale(t, 3);
    EXPECT_EQ(std::get<0>(result), 15);
}

TEST(TupleScaleTest, Tuple2D) {
    // Test 2D tuple * 2D tuple
    auto a = std::make_tuple(2, 3);
    auto b = std::make_tuple(4, 5);
    auto result = tuple_scale(a, b);
    EXPECT_EQ(std::get<0>(result), 8);
    EXPECT_EQ(std::get<1>(result), 15);
}

TEST(TupleScaleTest, Tuple3D) {
    // Test 3D tuple * 3D tuple
    auto a = std::make_tuple(2, 3, 4);
    auto b = std::make_tuple(5, 6, 7);
    auto result = tuple_scale(a, b);
    EXPECT_EQ(std::get<0>(result), 10);
    EXPECT_EQ(std::get<1>(result), 18);
    EXPECT_EQ(std::get<2>(result), 28);
}

TEST(TupleScaleTest, NestedTuple) {
    // Test nested tuple * nested tuple: ((2, 3), (4, 5)) * ((6, 7), (8, 9))
    auto a = std::make_tuple(std::make_tuple(2, 3), std::make_tuple(4, 5));
    auto b = std::make_tuple(std::make_tuple(6, 7), std::make_tuple(8, 9));
    auto result = tuple_scale(a, b);

    auto inner0 = std::get<0>(result);
    auto inner1 = std::get<1>(result);

    EXPECT_EQ(std::get<0>(inner0), 12);
    EXPECT_EQ(std::get<1>(inner0), 21);
    EXPECT_EQ(std::get<0>(inner1), 32);
    EXPECT_EQ(std::get<1>(inner1), 45);
}

TEST(TupleScaleTest, NestedTupleWithIntConstants) {
    // Test nested tuple with Int<> constants
    auto a = std::make_tuple(Int<2>{}, Int<3>{});
    auto b = std::make_tuple(Int<4>{}, Int<5>{});
    auto result = tuple_scale(a, b);
    EXPECT_EQ(std::get<0>(result), 8);
    EXPECT_EQ(std::get<1>(result), 15);
}

TEST(TupleScaleTest, MixedIntAndIntConstant) {
    // Test mixed int and Int<> in tuple
    auto a = std::make_tuple(2, Int<3>{});
    auto b = std::make_tuple(Int<4>{}, 5);
    auto result = tuple_scale(a, b);
    EXPECT_EQ(std::get<0>(result), 8);
    EXPECT_EQ(std::get<1>(result), 15);
}

// =============================================================================
// Test: tuple_add
// =============================================================================

TEST(TupleAddTest, ScalarScalar) {
    auto result = tuple_add(3, 4);
    EXPECT_EQ(result, 7);
}

TEST(TupleAddTest, IntConstantIntConstant) {
    auto result = tuple_add(Int<3>{}, Int<4>{});
    EXPECT_EQ(result, 7);
}

TEST(TupleAddTest, ScalarTupleSingle) {
    auto t = std::make_tuple(5);
    auto result = tuple_add(3, t);
    EXPECT_EQ(std::get<0>(result), 8);
}

TEST(TupleAddTest, TupleSingleScalar) {
    auto t = std::make_tuple(5);
    auto result = tuple_add(t, 3);
    EXPECT_EQ(std::get<0>(result), 8);
}

TEST(TupleAddTest, Tuple2D) {
    auto a = std::make_tuple(2, 3);
    auto b = std::make_tuple(4, 5);
    auto result = tuple_add(a, b);
    EXPECT_EQ(std::get<0>(result), 6);
    EXPECT_EQ(std::get<1>(result), 8);
}

TEST(TupleAddTest, Tuple3D) {
    auto a = std::make_tuple(2, 3, 4);
    auto b = std::make_tuple(5, 6, 7);
    auto result = tuple_add(a, b);
    EXPECT_EQ(std::get<0>(result), 7);
    EXPECT_EQ(std::get<1>(result), 9);
    EXPECT_EQ(std::get<2>(result), 11);
}

TEST(TupleAddTest, NestedTuple) {
    auto a = std::make_tuple(std::make_tuple(2, 3), std::make_tuple(4, 5));
    auto b = std::make_tuple(std::make_tuple(6, 7), std::make_tuple(8, 9));
    auto result = tuple_add(a, b);

    auto inner0 = std::get<0>(result);
    auto inner1 = std::get<1>(result);

    EXPECT_EQ(std::get<0>(inner0), 8);
    EXPECT_EQ(std::get<1>(inner0), 10);
    EXPECT_EQ(std::get<0>(inner1), 12);
    EXPECT_EQ(std::get<1>(inner1), 14);
}

// =============================================================================
// Test: dot_product
// =============================================================================

TEST(DotProductTest, ScalarScalar) {
    auto result = dot_product(3, 4);
    EXPECT_EQ(result, 12);
}

TEST(DotProductTest, IntConstantIntConstant) {
    auto result = dot_product(Int<3>{}, Int<4>{});
    EXPECT_EQ(result, 12);
}

TEST(DotProductTest, Tuple2D) {
    auto a = std::make_tuple(2, 3);
    auto b = std::make_tuple(4, 5);
    auto result = dot_product(a, b);
    EXPECT_EQ(result, 2 * 4 + 3 * 5);  // 8 + 15 = 23
}

TEST(DotProductTest, Tuple3D) {
    auto a = std::make_tuple(2, 3, 4);
    auto b = std::make_tuple(5, 6, 7);
    auto result = dot_product(a, b);
    EXPECT_EQ(result, 2 * 5 + 3 * 6 + 4 * 7);  // 10 + 18 + 28 = 56
}

TEST(DotProductTest, NestedTuple) {
    // Nested tuple dot product: ((2, 3), (4, 5)) . ((6, 7), (8, 9))
    // = 2*6 + 3*7 + 4*8 + 5*9 = 12 + 21 + 32 + 45 = 110
    auto a = std::make_tuple(std::make_tuple(2, 3), std::make_tuple(4, 5));
    auto b = std::make_tuple(std::make_tuple(6, 7), std::make_tuple(8, 9));
    auto result = dot_product(a, b);
    EXPECT_EQ(result, 110);
}

TEST(DotProductTest, NestedTupleWithIntConstants) {
    auto a = std::make_tuple(Int<2>{}, Int<3>{}, Int<4>{});
    auto b = std::make_tuple(Int<5>{}, Int<6>{}, Int<7>{});
    auto result = dot_product(a, b);
    EXPECT_EQ(result, 56);
}

// =============================================================================
// Test: tuple_for_each
// =============================================================================

TEST(TupleForEachTest, BasicTuple) {
    auto t = std::make_tuple(1, 2, 3, 4);
    int sum = 0;
    tuple_for_each(t, [&sum](auto x) { sum += x; });
    EXPECT_EQ(sum, 10);
}

TEST(TupleForEachTest, NestedTuple) {
    auto t = std::make_tuple(std::make_tuple(1, 2), std::make_tuple(3, 4));
    int sum = 0;
    tuple_for_each(t, [&sum](auto inner) {
        tuple_for_each(inner, [&sum](auto x) { sum += x; });
    });
    EXPECT_EQ(sum, 10);
}

TEST(TupleForEachTest, TupleWithIntConstants) {
    auto t = std::make_tuple(Int<1>{}, Int<2>{}, Int<3>{});
    int sum = 0;
    tuple_for_each(t, [&sum](auto x) { sum += x; });
    EXPECT_EQ(sum, 6);
}

// =============================================================================
// Test: is_tuple trait
// =============================================================================

TEST(IsTupleTest, IntIsNotTuple) {
    EXPECT_FALSE(is_tuple_v<int>);
}

TEST(IsTupleTest, IntConstantIsNotTuple) {
    EXPECT_FALSE(is_tuple_v<Int<5>>);
}

TEST(IsTupleTest, StdTupleIsTuple) {
    using TupleType = std::tuple<int, int>;
    EXPECT_TRUE(is_tuple_v<TupleType>);
    using SingleTuple = std::tuple<int>;
    EXPECT_TRUE(is_tuple_v<SingleTuple>);
}

TEST(IsTupleTest, NestedTupleIsTuple) {
    using NestedTuple = std::tuple<std::tuple<int, int>, int>;
    EXPECT_TRUE(is_tuple_v<NestedTuple>);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
