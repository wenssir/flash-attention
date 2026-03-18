#pragma once

#include "../config/macros.cuh"
#include "./stride.cuh"
#include "./shape.h"
#include "./coordinate.h"
#include "./offset.cuh"
#include "./composedLayout.h"
#include "../tensor_core/swizzle.cuh"
#include "../utils/print.h"

namespace layout {

namespace detail {

template <typename Shape>
HOST_DEVICE constexpr auto max_coord(Shape const& shape) {
    if constexpr (!container::is_tuple_v<Shape>) {
        if constexpr (!std::is_same_v<Shape, int>) {
            static_assert(Shape::value > 0, "Shape extent must be positive");
            return numeric::Int<Shape::value - 1>{};
        } else {
            return shape - 1;
        }
    } else {
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return make_coordinate(max_coord(cxx::get<Is>(shape))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<Shape>>{});
    }
}

template <typename A, typename B>
HOST_DEVICE constexpr auto zip_elements(A const& a, B const& b) {
    if constexpr (!container::is_tuple_v<A> && !container::is_tuple_v<B>) {
        return make_shape(a, b);
    } else {
        static_assert(container::is_tuple_v<A> && container::is_tuple_v<B>, "zip_elements requires matching tuple structure");
        static_assert(cxx::tuple_size_v<A> == cxx::tuple_size_v<B>, "zip_elements requires tuples of same rank");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return make_shape(zip_elements(cxx::get<Is>(a), cxx::get<Is>(b))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<A>>{});
    }
}

template <typename A, typename B>
HOST_DEVICE constexpr auto shape_div(A const& a, B const& b) {
    if constexpr (!container::is_tuple_v<A>) {
        if constexpr (!std::is_same_v<A, int> && !std::is_same_v<B, int>) {
            static_assert(B::value != 0, "Division by zero");
            static_assert((A::value % B::value) == 0, "Static shape_div must divide exactly");
            return numeric::Int<A::value / B::value>{};
        } else if constexpr (!std::is_same_v<A, int> && std::is_same_v<B, int>) {
            return A::value / b;
        } else if constexpr (std::is_same_v<A, int> && !std::is_same_v<B, int>) {
            return a / B::value;
        } else {
            return a / b;
        }
    } else {
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return make_shape(shape_div(cxx::get<Is>(a), cxx::get<Is>(b))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<A>>{});
    }
}

template <size_t I, typename Shape>
HOST_DEVICE constexpr auto tile_stride_element(Shape const& shape) {
    if constexpr (I + 1 >= cxx::tuple_size_v<Shape>) {
        return numeric::Int<1>{};
    } else {
        return numeric::mixed_mul(get_shape_size(cxx::get<I + 1>(shape)), tile_stride_element<I + 1>(shape));
    }
}

template <typename Shape>
HOST_DEVICE constexpr auto make_tile_stride(Shape const& shape) {
    return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
        return make_stride(tile_stride_element<Is>(shape)...);
    }(cxx::make_index_sequence<cxx::tuple_size_v<Shape>>{});
}

template <typename CoordElem, typename ShapeElem, typename StrideElem>
HOST_DEVICE constexpr auto slice_offset_element(CoordElem const& coord, ShapeElem const&, StrideElem const& stride) {
    if constexpr (container::is_tuple_v<CoordElem>) {
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return (numeric::Int<0>{} + ... + slice_offset_element(cxx::get<Is>(coord), numeric::Int<0>{}, cxx::get<Is>(stride)));
        }(cxx::make_index_sequence<cxx::tuple_size_v<CoordElem>>{});
    } else if constexpr (is_underscore_v<CoordElem>) {
        return numeric::Int<0>{};
    } else {
        return numeric::mixed_mul(coord, stride);
    }
}

template <typename CoordElem, typename ShapeElem, typename StrideElem>
HOST_DEVICE constexpr auto slice_shape_element(CoordElem const&, ShapeElem const& shape, StrideElem const&) {
    return shape;
}

template <typename CoordElem, typename ShapeElem, typename StrideElem>
HOST_DEVICE constexpr auto slice_stride_element(CoordElem const&, ShapeElem const&, StrideElem const& stride) {
    return stride;
}

template <typename Coord, typename Shape, typename Stride>
HOST_DEVICE constexpr auto slice_offset(Coord const& coord, Shape const& shape, Stride const& stride) {
    if constexpr (!container::is_tuple_v<Coord>) {
        return slice_offset_element(coord, shape, stride);
    } else {
        static_assert(cxx::tuple_size_v<Coord> == cxx::tuple_size_v<Shape>, "Rank mismatch in slice_offset");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return (numeric::Int<0>{} + ... + slice_offset_element(cxx::get<Is>(coord), cxx::get<Is>(shape), cxx::get<Is>(stride)));
        }(cxx::make_index_sequence<cxx::tuple_size_v<Coord>>{});
    }
}

template <typename Coord, typename Shape, typename Stride>
HOST_DEVICE constexpr auto slice_shape(Coord const& coord, Shape const& shape, Stride const& stride) {
    if constexpr (!container::is_tuple_v<Coord>) {
        static_assert(is_underscore_v<Coord>, "slice_shape requires underscore on scalar coordinate");
        return slice_shape_element(coord, shape, stride);
    } else {
        static_assert(cxx::tuple_size_v<Coord> == cxx::tuple_size_v<Shape>, "Rank mismatch in slice_shape");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return make_shape(slice_shape_element(cxx::get<Is>(coord), cxx::get<Is>(shape), cxx::get<Is>(stride))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<Coord>>{});
    }
}

template <typename Coord, typename Shape, typename Stride>
HOST_DEVICE constexpr auto slice_stride(Coord const& coord, Shape const& shape, Stride const& stride) {
    if constexpr (!container::is_tuple_v<Coord>) {
        static_assert(is_underscore_v<Coord>, "slice_stride requires underscore on scalar coordinate");
        return slice_stride_element(coord, shape, stride);
    } else {
        static_assert(cxx::tuple_size_v<Coord> == cxx::tuple_size_v<Shape>, "Rank mismatch in slice_stride");
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return make_stride(slice_stride_element(cxx::get<Is>(coord), cxx::get<Is>(shape), cxx::get<Is>(stride))...);
        }(cxx::make_index_sequence<cxx::tuple_size_v<Coord>>{});
    }
}

template <typename Coord, typename Shape, typename Stride>
HOST_DEVICE constexpr auto compact_slice_shape(Coord const& coord, Shape const& shape, Stride const& stride) {
    if constexpr (!container::is_tuple_v<Coord>) {
        static_assert(is_underscore_v<Coord>, "compact_slice_shape requires underscore on scalar coordinate");
        return shape;
    } else {
        constexpr size_t N = cxx::tuple_size_v<Coord>;
        auto append_shape = []<typename CoordElem, typename ShapeElem, typename StrideElem>(CoordElem const& coord_elem,
                                                                                             ShapeElem const& shape_elem,
                                                                                             StrideElem const& stride_elem) {
            if constexpr (is_underscore_v<CoordElem>) {
                return cxx::make_tuple(slice_shape_element(coord_elem, shape_elem, stride_elem));
            } else {
                return cxx::tuple<>{};
            }
        };
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return cxx::tuple_cat(
                append_shape(cxx::get<Is>(coord), cxx::get<Is>(shape), cxx::get<Is>(stride))...
            );
        }(cxx::make_index_sequence<N>{});
    }
}

template <typename Coord, typename Shape, typename Stride>
HOST_DEVICE constexpr auto compact_slice_stride(Coord const& coord, Shape const& shape, Stride const& stride) {
    if constexpr (!container::is_tuple_v<Coord>) {
        static_assert(is_underscore_v<Coord>, "compact_slice_stride requires underscore on scalar coordinate");
        return stride;
    } else {
        constexpr size_t N = cxx::tuple_size_v<Coord>;
        auto append_stride = []<typename CoordElem, typename ShapeElem, typename StrideElem>(CoordElem const& coord_elem,
                                                                                              ShapeElem const& shape_elem,
                                                                                              StrideElem const& stride_elem) {
            if constexpr (is_underscore_v<CoordElem>) {
                return cxx::make_tuple(slice_stride_element(coord_elem, shape_elem, stride_elem));
            } else {
                return cxx::tuple<>{};
            }
        };
        return [&]<size_t... Is>(cxx::index_sequence<Is...>) {
            return cxx::tuple_cat(
                append_stride(cxx::get<Is>(coord), cxx::get<Is>(shape), cxx::get<Is>(stride))...
            );
        }(cxx::make_index_sequence<N>{});
    }
}

} // namespace detail

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
        if constexpr (has_underscore<Coord>::value) {
            return slice(coord, *this);
        } else {
            return coord_to_idx(_shape, _stride, coord);
        }
    }

    template <typename... Ints>
    HOST_DEVICE constexpr auto operator()(Ints... coords) const {
        return coord_to_idx(_shape, _stride, cxx::make_tuple(coords...));
    }

    HOST_DEVICE constexpr auto size() const {
        return numeric::mixed_add(coord_to_idx(_shape, _stride, detail::max_coord(_shape)),
                                  numeric::Int<1>{});
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

template <typename Shape>
HOST_DEVICE constexpr auto make_ordered_layout(Shape const& shape) {
    return make_layout(shape, make_compact_stride(shape));
}

template <typename Shape, typename Order>
HOST_DEVICE constexpr auto make_ordered_layout(Shape const& shape, Order const&) {
    return make_ordered_layout(shape);
}

template <typename Coord, typename Shape, typename Stride>
HOST_DEVICE constexpr auto slice(Coord const& coord, Layout<Shape, Stride> const& layout) {
    auto base_offset = detail::slice_offset(coord, layout.shape(), layout.stride());
    auto sub_shape = detail::compact_slice_shape(coord, layout.shape(), layout.stride());
    auto sub_stride = detail::compact_slice_stride(coord, layout.shape(), layout.stride());
    return composition(tensor::NoSwizzle{}, base_offset, make_layout(sub_shape, sub_stride));
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

template <typename LayoutA, typename LayoutB>
HOST_DEVICE constexpr auto blocked_product(LayoutA const& block, LayoutB const& tiler) {
    auto block_shape = block.shape();
    auto tiler_shape = tiler.shape();
    auto block_stride = block.stride();
    auto tiler_stride = container::tuple_scale(tiler.stride(), block.size());

    if constexpr (container::is_tuple_v<decltype(block_shape)>) {
        static_assert(cxx::tuple_size_v<decltype(block_shape)> == cxx::tuple_size_v<decltype(tiler_shape)>,
                      "blocked_product currently requires block and tiler to have the same rank");
    }

    return make_layout(
        detail::zip_elements(block_shape, tiler_shape),
        detail::zip_elements(block_stride, tiler_stride)
    );
}

template <typename Layout, typename TrgShape>
HOST_DEVICE constexpr auto tile_to_shape(Layout const& block, TrgShape const& trg_shape) {
    auto product_shape = detail::shape_div(trg_shape, block.shape());
    return blocked_product(block, make_layout(product_shape, detail::make_tile_stride(product_shape)));
}

template <typename Layout, typename TrgShape, typename Order>
HOST_DEVICE constexpr auto tile_to_shape(Layout const& block, TrgShape const& trg_shape, Order const& order) {
    auto product_shape = detail::shape_div(trg_shape, block.shape());
    return blocked_product(block, make_ordered_layout(product_shape, order));
}

}
