#pragma once

#include <type_traits>
#include "./layout.h"
#include "../utils/print.h"

namespace layout {

template <typename OuterLayout, typename Offset, typename InnerLayout>
struct ComposedLayout;
    
// Trait to detect ComposedLayout
template <typename T>
struct is_composed_layout : cxx::false_type {};

template <typename OuterLayout, typename Offset, typename InnerLayout>
struct is_composed_layout<ComposedLayout<OuterLayout, Offset, InnerLayout>> : cxx::true_type {};

template <typename T>
constexpr bool is_composed_layout_v = is_composed_layout<T>::value;

template <typename OuterLayout, typename Offset, typename InnerLayout>
struct ComposedLayout {
    [[no_unique_address]] OuterLayout _outer;
    [[no_unique_address]] Offset      _offset;
    [[no_unique_address]] InnerLayout _inner;

    HOST_DEVICE constexpr ComposedLayout()
    requires(cxx::is_default_constructible_v<OuterLayout> &&
           cxx::is_default_constructible_v<Offset> &&
           cxx::is_default_constructible_v<InnerLayout>) = default;

    HOST_DEVICE constexpr ComposedLayout(OuterLayout const& o, Offset const& off, InnerLayout const& i)
        : _outer(o), _offset(off), _inner(i) {}

    template <typename Coord>
    HOST_DEVICE constexpr auto operator()(Coord const& c) const {
        return _outer(_offset() + _inner(c));
    }

    template <typename... Ints>
    HOST_DEVICE constexpr auto operator()(Ints... coords) const {
        return operator()(cxx::make_tuple(coords...));
    }

    HOST_DEVICE constexpr auto outer() const { return _outer; }

    HOST_DEVICE constexpr auto size() const { return _inner.size(); }
    HOST_DEVICE constexpr auto shape() const { return _inner.shape(); }
    HOST_DEVICE constexpr auto stride() const { return _inner.stride(); }
    HOST_DEVICE constexpr auto offset() const { return _offset; }

    // Print function for debugging
    HOST_DEVICE void print() const {
        printf("ComposedLayout:\n");
        printf("  Offset: ");
        print_shape(_offset.shape());
        printf("  Inner: ");
        print_shape(_inner.shape());
        printf("  Inner stride: ");
        print_stride(_inner.stride());
        printf("  Size: %zu\n", static_cast<size_t>(size()));
    }
};


template <typename Outer, typename Offset, typename Inner>
HOST_DEVICE constexpr auto composition(Outer const& o, Offset const& off, Inner const& i) {
    return ComposedLayout<Outer, Offset, Inner>(o, off, i);
}

template <typename Outer, typename Inner>
HOST_DEVICE constexpr auto composition(Outer const& o, Inner const& i) {
    return ComposedLayout<Outer, numeric::Int<0>, Inner>(o, numeric::Int<0>{}, i);
}

}
