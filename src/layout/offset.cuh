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

}
