#pragma once

#include "../config/macros.cuh"
#include "../container/tuple.cuh"
#include "./stride.cuh"

namespace layout {

template <typename... Ts>
using Shape = container::Tuple<Ts...>;

template <typename... Ts>
HOST_DEVICE constexpr auto make_shape(Ts... ts) {
    return cxx::make_tuple(ts...);
}

template <class Shape>
HOST_DEVICE constexpr auto get_shape_size(Shape const& shape) {
    if constexpr (!container::is_tuple_v<Shape>) {
        return shape;
    } else {
        auto result = numeric::Int<1>{};
        [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            ((result = numeric::mixed_mul(result, get_shape_size(cxx::get<Is>(shape)))), ...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<Shape>>{});
        return result;
    }
}

template <size_t I, class Shape>
HOST_DEVICE constexpr auto prefix_product_size(Shape const& shape) {
    static_assert(container::is_tuple_v<Shape>, "prefix_product_size only accepts tuples");

    if constexpr (I == 0) {
        return numeric::Int<1>{};
    } else {
        return numeric::mixed_mul(prefix_product_size<I - 1>(shape), get_shape_size(cxx::get<I - 1>(shape)));
    }

}

template <class Shape, typename CurrentStride = numeric::Int<1>>
HOST_DEVICE constexpr auto make_compact_stride(Shape const& shape, CurrentStride current_stride = numeric::Int<1>{}) {
    if constexpr (!container::is_tuple_v<Shape>) {
        return current_stride;
    } else {
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return layout::make_stride(
                make_compact_stride(cxx::get<Is>(shape),
                numeric::mixed_mul(current_stride, prefix_product_size<Is>(shape)))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<Shape>>{});
    }
}

}
