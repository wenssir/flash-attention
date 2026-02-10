#pragma once

#include "../config/macros.cuh"

namespace numeric{
template<typename A, typename B>
constexpr
void add(A a, B b) {
    return a + b;
}

template<typename A, typename B>
constexpr
void mul(A a, B b) {
    return a * b;
}

}
