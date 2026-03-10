#pragma once

#include "../config/macros.cuh"
#include "../layout/coordinate.h"
#include "../numeric/Int.cuh"
#include "../ptx/ptx.cuh"
#include "./thread_map.cuh"

namespace loadstore {

template <typename LdmatrixT, int ROWS, int COLS, bool Trans = false>
struct LdmatrixHelper {
    using Ldmatrix = LdmatrixT;

    static_assert(ROWS == 16 || ROWS == 8, "ROWS must be 16 or 8");
    static_assert(COLS == 16 || COLS == 8, "COLS must be 16 or 8");

    DEVICE static auto coord(int lane_id) {
        if constexpr (!Trans) {
            return coord_non_trans(lane_id);
        } else {
            return coord_trans(lane_id);
        }
    }

private:
    DEVICE static auto coord_non_trans(int lane_id) {
        using namespace layout;
        if constexpr (ROWS == 16 && COLS == 16) {
            if (lane_id < 16) {
                return make_coordinate(lane_id, 0);
            }
            return make_coordinate(lane_id - 16, 8);
        } else if constexpr (ROWS == 16 && COLS == 8) {
            return make_coordinate(lane_id % 16, 0);
        } else {
            return make_coordinate(lane_id % 8, 0);
        }
    }

    DEVICE static auto coord_trans(int lane_id) {
        using namespace layout;
        if constexpr (ROWS == 16 && COLS == 16) {
            return make_coordinate(lane_id % 16, 0);
        } else if constexpr (ROWS == 16 && COLS == 8) {
            return make_coordinate(lane_id % 16, 0);
        } else {
            return make_coordinate(lane_id % 8, 0);
        }
    }
};

template <typename T>
using LdmatrixHelperQ = LdmatrixHelper<ptx::LDMATRIX_X4<T>, 16, 16>;

template <typename T>
using LdmatrixHelperK = LdmatrixHelper<ptx::LDMATRIX_X2_TRANS<T>, 16, 8>;

template <typename Map, typename Frag, typename TensorS>
DEVICE void copy_s2r(Frag& frag, TensorS& s_tensor) {
    int lane_id = Map::lane_id(threadIdx.x);
    frag.load(s_tensor, lane_id);
}

template <typename KernelConfig, typename Frag, typename TensorS>
DEVICE void load_fragment_a(Frag& frag, TensorS& s_tensor) {
    using Map = S2RQThreadMap<KernelConfig>;
    copy_s2r<Map>(frag, s_tensor);
}

template <typename KernelConfig, typename Frag, typename TensorS>
DEVICE void load_fragment_b(Frag& frag, TensorS& s_tensor) {
    using Map = S2RKVThreadMap<KernelConfig>;
    copy_s2r<Map>(frag, s_tensor);
}

template <typename KernelConfig, typename Frag, typename TensorS>
DEVICE void copy_s2r_q(Frag& frag, TensorS& s_tensor) {
    load_fragment_a<KernelConfig>(frag, s_tensor);
}

template <typename KernelConfig, typename Frag, typename TensorS>
DEVICE void copy_s2r_kv(Frag& frag, TensorS& s_tensor) {
    load_fragment_b<KernelConfig>(frag, s_tensor);
}

struct CopyS2ROp {
    template <typename KernelConfig, typename Frag, typename TensorS>
    DEVICE void operator()(Frag& frag, TensorS& s_tensor) const {
        load_fragment_a<KernelConfig>(frag, s_tensor);
    }

    template <typename Frag, typename TensorS>
    DEVICE void operator()(Frag& frag, TensorS& s_tensor) const {
        copy_s2r<LdmatrixThreadMap<void>>(frag, s_tensor);
    }
};

struct CopyS2RKVOp {
    template <typename KernelConfig, typename Frag, typename TensorS>
    DEVICE void operator()(Frag& frag, TensorS& s_tensor) const {
        load_fragment_b<KernelConfig>(frag, s_tensor);
    }

    template <typename Frag, typename TensorS>
    DEVICE void operator()(Frag& frag, TensorS& s_tensor) const {
        copy_s2r<LdmatrixThreadMap<void>>(frag, s_tensor);
    }
};

} // namespace loadstore
