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
    DEVICE constexpr auto& operator()(Coords... c) {
        auto offset = _layout(c...);
        return _data[offset];
    }

    template <typename... Coords>
    DEVICE constexpr auto const& operator()(Coords... c) const {
        auto offset = _layout(c...);
        return _data[offset];
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

    // Print first N elements
    HOST void print_elements(int max_elements = 10) const {
        print();
        printf("  Data (first %d elements): ", max_elements);
        int n = static_cast<int>(_layout.size());
        if (n > max_elements) n = max_elements;
        for (int i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<std::remove_pointer_t<Engine>, float>) {
                printf("%.2f ", static_cast<float>(_data[i]));
            } else if constexpr (std::is_same_v<std::remove_pointer_t<Engine>, int>) {
                printf("%d ", static_cast<int>(_data[i]));
            } else {
                printf("%p ", static_cast<void*>(_data[i]));
            }
        }
        if (static_cast<int>(_layout.size()) > max_elements) {
            printf("... (%zu more)", static_cast<size_t>(_layout.size() - max_elements));
        }
        printf("\n");
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
    auto tensor_stride = layout.stride();
    auto offset = layout::make_offset(container::tuple_scale(coord, sub_shape), tensor_stride);
    if constexpr (layout::is_composed_layout_v<decltype(layout)>) {
        auto tensor_offset = layout.offset();
        offset = layout::add_offset(offset, tensor_offset, tensor_stride);
        auto new_composed_layout = layout::ComposedLayout(layout.outer(), offset, layout::make_layout(sub_shape, tensor_stride));
        return tensor::Tensor(tensor.data_ptr(), new_composed_layout);
    } else {
        auto new_composed_layout = layout::ComposedLayout(NoSwizzle(), offset, layout::make_layout(sub_shape, tensor_stride));
        return tensor::Tensor(tensor.data_ptr(), new_composed_layout);
    }
}

}
