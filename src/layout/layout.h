#pragma once

#include "../config/macros.cuh"
#include "./stride.cuh"
#include "./shape.h"
#include "../utils/print.h"

namespace layout {

template <typename Shape, typename Stride>
struct Layout {
    using ShapeType = Shape;
    using StrideType = Stride;

    [[no_unique_address]] Shape _shape;
    [[no_unique_address]] Stride _stride;

    HOST_DEVICE constexpr Layout() : _shape(), _stride() {}

    HOST_DEVICE constexpr Layout(Shape s, Stride d) : _shape(s), _stride(d) {}

    template <typename Coord>
    HOST_DEVICE constexpr auto operator()(Coord const& coord) const {
        return coord_to_idx(_shape, _stride, coord);
    }

    template <typename... Ints>
    HOST_DEVICE constexpr auto operator()(Ints... coords) const {
        return coord_to_idx(_shape, _stride, cxx::make_tuple(coords...));
    }

    HOST_DEVICE constexpr auto size() const {
       return get_shape_size(_shape);
    }

    HOST_DEVICE constexpr auto shape() const {
        return _shape;
    }

    HOST_DEVICE constexpr auto stride() const {
        return _stride;
    }

    HOST_DEVICE void print() {
        print_shape(_shape);
        print_stride(_stride);
    }

};

template <typename Shape, typename Stride>
HOST_DEVICE constexpr auto make_layout(Shape s, Stride d) {
    return Layout<Shape, Stride>(s, d);
}

template <typename Shape, typename Stride, typename Tile>
HOST_DEVICE constexpr auto divide_element(Shape const& s, Stride const& d, Tile const& t) {
    auto new_shape = s / t;
    auto new_stride = d * t;
    
    return make_layout(make_shape(t, new_shape), make_stride(d, new_stride));
}

template <typename Layout, typename Tile>
HOST_DEVICE constexpr auto logical_divide(Layout const& layout, Tile const& t) {
    auto shape = layout.shape();
    auto stride = layout.stride();

    if constexpr (!container::is_tuple_v<decltype(shape)>) {
        return divide_element(shape, stride, t);
    } else {
        static_assert(cxx::tuple_size_v<decltype(shape)> == cxx::tuple_size_v<Tile>, 
                      "Rank mismatch in logical_divide");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            auto sub_res = cxx::make_tuple(
                logical_divide(make_layout(cxx::get<Is>(shape), 
                cxx::get<Is>(stride)), cxx::get<Is>(t))...
            );

            return make_layout(
                make_shape(cxx::get<Is>(sub_res).shape()...),
                make_stride(cxx::get<Is>(sub_res).stride()...)
            );
        }(cxx::make_index_sequence<cxx::tuple_size_v<decltype(shape)>>{});
    }
}

template <typename OuterLayout, typename InnerLayout>
struct ZippedResult {
    [[no_unique_address]] OuterLayout outer;
    [[no_unique_address]] InnerLayout inner;
};

template <typename Shape, typename Stride, typename Tile>
HOST_DEVICE constexpr auto divide_element_zipped(Shape const& s, Stride const& d, Tile const& t) {
    auto in_shape = t;
    auto in_stride = d;

    auto out_shape = s / t;
    auto out_stride = d * t;

    auto inner = make_layout(in_shape, in_stride);
    auto outer = make_layout(out_shape, out_stride);

    return ZippedResult<decltype(outer), decltype(inner)>{outer, inner};
}


template <typename Layout, typename Tile>
HOST_DEVICE constexpr auto zipped_divide(Layout const& layout, Tile const& t) {
    auto shape = layout.shape();
    auto stride = layout.stride();

    if constexpr (!container::is_tuple_v<decltype(shape)>) {
        return divide_element_zipped(shape, stride, t);
    } 
    else {
        static_assert(cxx::tuple_size_v<decltype(shape)> == cxx::tuple_size_v<Tile>, 
                      "Rank mismatch in zipped_divide");

        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            auto sub_results = cxx::make_tuple(
                divide_element_zipped(
                    cxx::get<Is>(shape), 
                    cxx::get<Is>(stride), 
                    cxx::get<Is>(t)
                )...
            );

            auto outer = make_layout(
                make_shape(cxx::get<Is>(sub_results).outer.shape()...),
                make_stride(cxx::get<Is>(sub_results).outer.stride()...)
            );

            auto inner = make_layout(
                make_shape(cxx::get<Is>(sub_results).inner.shape()...),
                make_stride(cxx::get<Is>(sub_results).inner.stride()...)
            );

            return ZippedResult<decltype(outer), decltype(inner)>{outer, inner};

        }(cxx::make_index_sequence<cxx::tuple_size_v<decltype(shape)>>{});
    }
}

}
