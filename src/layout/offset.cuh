#pragma once

#include "../config/macros.cuh"
#include "./shape.h"
#include "./stride.cuh"

namespace layout {

template <typename Shape, typename Stride>
struct Offset {
    [[no_unique_address]] Shape _shape;
    [[no_unique_address]] Stride _stride;

    HOST_DEVICE constexpr Offset(Shape s, Stride d) : _shape(s), _stride(d) {}

    HOST_DEVICE constexpr auto operator()() const {
        return coord_to_idx(_shape, _stride, _shape);
    }

    HOST_DEVICE constexpr auto shape() const {
        return _shape;
    }

    HOST_DEVICE constexpr auto stride() const {
        return _stride;
    }

};

template <typename Shape, typename Stride>
HOST_DEVICE constexpr auto make_offset(Shape s, Stride d) {
    return Offset<Shape, Stride>(s, d);
}

template <typename ShapeA, typename StrideA, typename ShapeB, typename StrideB, typename Stride>
HOST_DEVICE constexpr auto add_offset(Offset<ShapeA, StrideA> const& a,
                                      Offset<ShapeB, StrideB> const& b,
                                      Stride const& stride) {
    return make_offset(container::tuple_add(a.shape(), b.shape()), stride);
}

template <typename ShapeA, typename StrideA, typename Stride>
HOST_DEVICE constexpr auto add_offset(Offset<ShapeA, StrideA> const& a,
                                      numeric::Int<0> const&,
                                      Stride const& stride) {
    return make_offset(a.shape(), stride);
}

template <typename ShapeB, typename StrideB, typename Stride>
HOST_DEVICE constexpr auto add_offset(numeric::Int<0> const&,
                                      Offset<ShapeB, StrideB> const& b,
                                      Stride const& stride) {
    return make_offset(b.shape(), stride);
}

template <typename Stride>
HOST_DEVICE constexpr auto add_offset(numeric::Int<0> const&,
                                      numeric::Int<0> const&,
                                      Stride const&) {
    return numeric::Int<0>{};
}

}
