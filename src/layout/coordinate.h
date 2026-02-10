#pragma once

#include "../config/macros.cuh"
#include "../container/tuple.cuh"

namespace layout {

template <typename... Ts>
using Coordinate = container::Tuple<Ts...>;

template <typename... Ts>
HOST_DEVICE constexpr auto make_coordinate(Ts... ts) {
    return cxx::make_tuple(ts...);
}

}
