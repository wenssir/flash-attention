#pragma once

#include <cstdio>

#include "../config/macros.cuh"
#include "../numeric/Int.cuh"


HOST_DEVICE void print(int i) {
    printf("%d", i);
}

template <int N>
HOST_DEVICE void print(numeric::Int<N>) {
    printf("_%d", N);
}

HOST_DEVICE void print(const char* s) {
    printf("%s", s);
}

template <typename Tuple, size_t... Is>
HOST_DEVICE void print_tuple_impl(Tuple const& t, cxx::index_sequence<Is...>) {
    printf("(");

    (((Is == 0 ? void() : print(",")), print(cxx::get<Is>(t))), ...);

    printf(")");
}

template <typename... Ts>
HOST_DEVICE void print(cxx::tuple<Ts...> const& t) {
    print_tuple_impl(t, cxx::make_index_sequence<sizeof...(Ts)>{});
}

template <typename T>
HOST_DEVICE void print_shape(T const& shape) {
    printf("Shape: ");
    print(shape); // 自动匹配重载
    printf("\n");
}

template <typename T>
HOST_DEVICE void print_stride(T const& stride) {
    printf("Stride: ");
    print(stride); // 自动匹配重载
    printf("\n");
}
