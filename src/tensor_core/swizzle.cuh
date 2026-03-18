#pragma once

namespace tensor {

template <int BBits = 3, int MBase = 0, int SShift = 3>
struct Swizzle {
    static constexpr int mbase = MBase;
    static constexpr int mask_bits = BBits;
    static constexpr int mask_shift = SShift;

    static constexpr int bit_mask = (1 << mask_bits) - 1;
    static constexpr int yy_mask = bit_mask << (mbase + mask_shift);
    static constexpr int yy_mask_lowest_bit = yy_mask & -yy_mask;

    HOST_DEVICE auto operator()(int const &offset) const {
        const int row_shifted = (offset & yy_mask) >> mask_shift;
        return offset ^ row_shifted;
    }

    template <typename Coord>
    HOST_DEVICE constexpr auto operator()(Coord const& c) const {
        if constexpr (container::is_tuple_v<Coord>) {
            return operator()(cxx::get<0>(c));
        } else {
            return operator()(c);
        }
    }
};

struct NoSwizzle {
    template <typename T>
    HOST_DEVICE constexpr auto operator()(T const& offset) const { return offset; }
};

}
