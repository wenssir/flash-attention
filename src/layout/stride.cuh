#pragma once

#include "../config/macros.cuh"
#include "../container/tuple.cuh"
#include "../numeric/Int.cuh"

namespace layout {

template <typename... Ts>
using Stride = container::Tuple<Ts...>;

template <typename... Ts>
HOST_DEVICE constexpr auto make_stride(Ts... ts) {
    return cxx::make_tuple(ts...);
}

template <class Shape, class Stride, class Coord>
HOST_DEVICE constexpr auto coord_to_idx(Shape const& shape, Stride const& stride, Coord const& coord) {
    if constexpr (container::is_tuple_v<Stride> && container::is_tuple_v<Coord>) {
        return container::dot_product(coord, stride);
    } else {
        return stride * coord;
    }
}

}
