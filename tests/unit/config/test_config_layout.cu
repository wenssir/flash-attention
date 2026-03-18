#include <type_traits>

#include <gtest/gtest.h>

#include "src/config/config.cuh"
#include "src/tensor_core/tensor.cuh"

namespace {

template <typename T>
struct has_runtime_int_leaf : std::false_type {};

template <>
struct has_runtime_int_leaf<int> : std::true_type {};

template <typename T>
struct has_runtime_int_leaf<T const> : has_runtime_int_leaf<T> {};

template <typename T>
struct has_runtime_int_leaf<T&> : has_runtime_int_leaf<T> {};

template <typename T>
struct has_runtime_int_leaf<T&&> : has_runtime_int_leaf<T> {};

template <typename... Ts>
struct has_runtime_int_leaf<cxx::tuple<Ts...>>
    : cxx::bool_constant<(has_runtime_int_leaf<Ts>::value || ...)> {};

using Cfg = config::ForwardConfig;
using QLayout = Cfg::SmemLayoutQ;
using KVLayout = Cfg::SmemLayoutKV;

using QSizeType = std::remove_cvref_t<decltype(QLayout{}.size())>;
using KVSizeType = std::remove_cvref_t<decltype(KVLayout{}.size())>;
static_assert(!std::is_same_v<QSizeType, int>, "SmemLayoutQ::size() should not degrade to runtime int");
static_assert(!std::is_same_v<KVSizeType, int>, "SmemLayoutKV::size() should not degrade to runtime int");

using QShapeType = std::remove_cvref_t<decltype(QLayout{}.shape())>;
using QStrideType = std::remove_cvref_t<decltype(QLayout{}.stride())>;
using KVShapeType = std::remove_cvref_t<decltype(KVLayout{}.shape())>;
using KVStrideType = std::remove_cvref_t<decltype(KVLayout{}.stride())>;

static_assert(!has_runtime_int_leaf<QShapeType>::value, "SmemLayoutQ shape should not contain runtime int leaves");
static_assert(!has_runtime_int_leaf<QStrideType>::value, "SmemLayoutQ stride should not contain runtime int leaves");
static_assert(!has_runtime_int_leaf<KVShapeType>::value, "SmemLayoutKV shape should not contain runtime int leaves");
static_assert(!has_runtime_int_leaf<KVStrideType>::value, "SmemLayoutKV stride should not contain runtime int leaves");

TEST(ConfigLayoutTest, SmemLayoutsHaveExpectedCompileTimeSpan) {
    EXPECT_TRUE(std::is_default_constructible_v<QLayout>);
    EXPECT_TRUE(std::is_default_constructible_v<KVLayout>);
    EXPECT_FALSE(std::is_trivially_default_constructible_v<QLayout>);
    EXPECT_FALSE(std::is_trivially_default_constructible_v<KVLayout>);

    constexpr auto q_size = QLayout{}.size();
    constexpr auto kv_size = KVLayout{}.size();
    EXPECT_EQ(static_cast<int>(q_size), Cfg::BlockM * Cfg::HeadDim);
    EXPECT_EQ(static_cast<int>(kv_size), Cfg::BlockN * Cfg::HeadDim);
}

TEST(ConfigLayoutTest, TensorWithCompileTimeSmemLayoutStaysLightweight) {
    using QTensor = tensor::Tensor<__half*, QLayout>;
    using KVTensor = tensor::Tensor<__half*, KVLayout>;

    EXPECT_LT(sizeof(QLayout), 64u);
    EXPECT_LT(sizeof(KVLayout), 64u);
    EXPECT_LE(sizeof(QTensor), sizeof(__half*) + 16u);
    EXPECT_LE(sizeof(KVTensor), sizeof(__half*) + 16u);
}

}  // namespace
