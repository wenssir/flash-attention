#pragma once

#include <type_traits>

#include "../config/macros.cuh"

namespace numeric{

template <int N>
using Int = cxx::integral_constant<int, N>;

template <typename A, typename B>
HOST_DEVICE constexpr auto mixed_mul(A a, B b) {
    if constexpr (std::is_same_v<A, int> || std::is_same_v<B, int>) {
        return (int)a * (int)b;
    } else {
        return Int<A::value * B::value>{};
    }
}


template <typename A, typename B>
HOST_DEVICE constexpr auto mixed_add(A a, B b) {
    if constexpr (std::is_same_v<A, int> || std::is_same_v<B, int>) {
        return (int)a + (int)b;
    } else {
        return Int<A::value + B::value>{};
    }
}

template <int N, int M>
HOST_DEVICE constexpr auto operator+(Int<N>, Int<M>) {
    return Int<N + M>{};
}

template <int N>
HOST_DEVICE constexpr int operator+(Int<N>, int m) { return N + m; }

template <int M>
HOST_DEVICE constexpr int operator+(int n, Int<M>) { return n + M; }

template <int N, int M>
HOST_DEVICE constexpr auto operator*(Int<N>, Int<M>) {
    return Int<N * M>{};
}

template <int N>
HOST_DEVICE constexpr int operator*(Int<N>, int m) { return N * m; }

template <int M>
HOST_DEVICE constexpr int operator*(int n, Int<M>) { return n * M; }

template <int N, int M>
HOST_DEVICE constexpr auto operator/(Int<N>, Int<M>) {
    static_assert(M != 0, "Division by zero");
    static_assert(N % M == 0, "Static division must be exact for logical_divide");
    return Int<N / M>{};
}

template <int N>
HOST_DEVICE constexpr int operator/(Int<N>, int m) { return N / m;}

template <int M>
HOST_DEVICE constexpr int operator/(int n, Int<M>) { return n / M; }


}
