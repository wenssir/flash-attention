#pragma once

#include "../config/macros.cuh"
#include "../layout/shape.h"
#include "../layout/layout.h"
#include "../layout/offset.cuh"
#include "../layout/coordinate.h"
#include "../layout/composedLayout.h"
#include "../tensor_core/swizzle.cuh"
#include "../utils/print.h"
#include "../tensor_core/tensor.cuh"


namespace tensor {

template<typename Engine, typename Layout>
struct Tensor{
    Engine _data;
    Layout _layout;

    HOST_DEVICE constexpr Tensor() : _data(nullptr), _layout() {}

    HOST_DEVICE constexpr Tensor(Engine d, Layout l) : _data(d), _layout(l) {}

    template <typename... Coords>
    DEVICE constexpr decltype(auto) operator()(Coords... c) {
        auto result = _layout(c...);
        if constexpr (layout::has_underscore<decltype(cxx::make_tuple(c...))>::value) {
            return Tensor<Engine, decltype(result)>(_data, result);
        } else {
            return _data[result];
        }
    }

    template <typename... Coords>
    DEVICE constexpr decltype(auto) operator()(Coords... c) const {
        auto result = _layout(c...);
        if constexpr (layout::has_underscore<decltype(cxx::make_tuple(c...))>::value) {
            return Tensor<Engine, decltype(result)>(_data, result);
        } else {
            return _data[result];
        }
    }

    template <typename... Coords>
    HOST_DEVICE constexpr auto offset(Coords... c) const {
        return _layout(c...);
    }

    HOST_DEVICE constexpr auto shape() const { return _layout.shape(); }
    HOST_DEVICE constexpr auto stride() const { return _layout.stride(); }
    HOST_DEVICE constexpr auto size() const { return _layout.size(); }
    HOST_DEVICE constexpr auto layout() const { return _layout; }

    HOST_DEVICE constexpr auto data_ptr() const { return _data; }

    // Print function for debugging
    HOST void print() const {
        printf("Tensor @ %p:\n", _data);
        printf("  Shape: ");
        print_shape(_layout.shape());
        printf("  Stride: ");
        print_stride(_layout.stride());
        printf("  Size: %zu\n", static_cast<size_t>(_layout.size()));
    }

};

template <typename Ptr, typename Layout>
HOST_DEVICE constexpr auto make_tensor(Ptr ptr, Layout layout, int offset = 0) {
    return Tensor<Ptr, Layout>(ptr + offset, layout);
}

template <typename Ptr, typename Shape, typename Stride>
HOST_DEVICE constexpr auto make_tensor(Ptr ptr, Shape s, Stride d, int offset = 0) {
    return Tensor<Ptr, decltype(Layout(s, d))>(ptr[offset], s, d);
}

template <typename TensorType, typename LayoutType, typename Coord>
HOST_DEVICE constexpr auto local_tile(TensorType && tensor, LayoutType sub_layout, Coord coord) {
    auto sub_shape = sub_layout.shape();
    auto layout = tensor.layout();
    auto base_coord = container::tuple_scale(coord, sub_shape);
    if constexpr (layout::is_composed_layout_v<decltype(layout)>) {
        auto base_offset = layout.inner()(base_coord);
        auto new_composed_layout = layout::ComposedLayout(
            layout.outer(),
            layout::make_offset(base_offset),
            sub_layout);
        return tensor::Tensor(tensor.data_ptr(), new_composed_layout);
    } else {
        auto base_offset = layout(base_coord);
        auto new_composed_layout = layout::ComposedLayout(
            NoSwizzle(),
            layout::make_offset(base_offset),
            sub_layout);
        return tensor::Tensor(tensor.data_ptr(), new_composed_layout);
    }
}

}
