#pragma once

#include "../config/macros.cuh"
#include "../container/tuple.cuh"

namespace layout {

struct Underscore {};

inline constexpr Underscore _;

template <typename T>
struct is_underscore : cxx::false_type {};

template <>
struct is_underscore<Underscore> : cxx::true_type {};

template <typename T>
constexpr bool is_underscore_v = is_underscore<cxx::remove_cv_t<cxx::remove_reference_t<T>>>::value;

template <typename T>
struct has_underscore : is_underscore<cxx::remove_cv_t<cxx::remove_reference_t<T>>> {};

template <typename... Ts>
struct has_underscore<cxx::tuple<Ts...>>
    : cxx::bool_constant<(has_underscore<Ts>::value || ...)> {};

template <typename... Ts>
using Coordinate = container::Tuple<Ts...>;

template <typename... Ts>
HOST_DEVICE constexpr auto make_coordinate(Ts... ts) {
    return cxx::make_tuple(ts...);
}

}
