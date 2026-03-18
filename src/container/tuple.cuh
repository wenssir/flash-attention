#pragma once

#include <iostream>

#include "../config/macros.cuh"
#include "../numeric/Int.cuh"

namespace container {

template <typename T>
struct is_tuple : cxx::false_type {};

template <typename... Ts>
struct is_tuple<cxx::tuple<Ts...>> : cxx::true_type {};

template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

template <typename Tuple, typename Func, size_t... Is>
DEVICE constexpr void tuple_for_each_impl(Tuple&& t, Func&& f, cxx::index_sequence<Is...>) {
    (f(cxx::get<Is>(t)), ...);
}

template <typename Tuple, typename Func>
DEVICE constexpr void tuple_for_each(Tuple&& t, Func&& f) {
    constexpr auto size = cxx::tuple_size_v<cxx::remove_reference_t<Tuple>>;
    tuple_for_each_impl(cxx::forward<Tuple>(t), cxx::forward<Func>(f), cxx::make_index_sequence<size>{});
}

template <typename... Ts>
using Tuple = cxx::tuple<Ts...>;

template <typename A, typename B>
HOST_DEVICE constexpr auto dot_product(A a, B b);

template <size_t I, typename A, typename B>
HOST_DEVICE constexpr auto dot_product_tuple(A a, B b) {
    if constexpr (I == cxx::tuple_size_v<A>) {
        return numeric::Int<0>{};
    } else {
        return numeric::mixed_add(
            dot_product(cxx::get<I>(a), cxx::get<I>(b)),
            dot_product_tuple<I + 1>(a, b));
    }
}

template <typename A, typename B>
HOST_DEVICE constexpr auto dot_product(A a, B b) {
    if constexpr (!is_tuple_v<A>) {
        return numeric::mixed_mul(a, b);
    } else {
        static_assert(is_tuple_v<A> && is_tuple_v<B>, "dot_product only accepts tuples");
        static_assert(cxx::tuple_size_v<A> == cxx::tuple_size_v<B>, "dot_product only accepts tuples of the same size");
        return dot_product_tuple<0>(a, b);
    }
}

template <typename A, typename B>
HOST_DEVICE constexpr auto tuple_add(A a, B b) {
    if constexpr (!is_tuple_v<A> && !is_tuple_v<B>) {
        return a + b;
    } else if constexpr (!is_tuple_v<A>) {
        static_assert(cxx::tuple_size_v<B> == 1, "Scalar-tuple addition requires tuple size 1");
        return cxx::make_tuple(a + cxx::get<0>(b));
    } else if constexpr (!is_tuple_v<B>) {
        static_assert(cxx::tuple_size_v<A> == 1, "Tuple-scalar addition requires tuple size 1");
        return cxx::make_tuple(cxx::get<0>(a) + b);
    } else {
        static_assert(cxx::tuple_size_v<A> == cxx::tuple_size_v<B>, "Tuple addition requires same size");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return cxx::make_tuple(tuple_add(cxx::get<Is>(a), cxx::get<Is>(b))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<A>>{});
    }
}

template <typename A, typename B>
HOST_DEVICE constexpr auto tuple_scale(A a, B b) {
    if constexpr (!is_tuple_v<A> && !is_tuple_v<B>) {
        return numeric::mixed_mul(a, b);
    } else if constexpr (!is_tuple_v<A>) {
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return cxx::make_tuple(tuple_scale(a, cxx::get<Is>(b))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<B>>{});
    } else if constexpr (!is_tuple_v<B>) {
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return cxx::make_tuple(tuple_scale(cxx::get<Is>(a), b)...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<A>>{});
    } else {
        static_assert(cxx::tuple_size_v<A> == cxx::tuple_size_v<B>, "Tuple scale requires same size");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return cxx::make_tuple(tuple_scale(cxx::get<Is>(a), cxx::get<Is>(b))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<A>>{});
    }
}

} // namespace container
