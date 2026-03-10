#pragma once

#include <cuda_fp16.h>
#include "../ptx/ptx.cuh"
#include "../loadstore/copy_srm.cuh"

namespace tensor{

template<typename T, typename Helper, int ROWS, int COLS>
struct Fragment{
    static_assert(ROWS % 8 == 0 && COLS % 8 == 0, "Fragment size must be divisible by 8");
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "Fragment only supports fp16/bf16");

    static constexpr int REGS_PER_LANE = (ROWS * COLS) / 32 / 2;
    static constexpr int kRows = ROWS;
    static constexpr int kCols = COLS;

    uint32_t regs[REGS_PER_LANE];

    DEVICE constexpr Fragment() {
        #pragma unroll
        for (int i = 0; i < REGS_PER_LANE; ++i) {
            regs[i] = 0;
        }
    }

    DEVICE uint32_t& operator()(int idx) {
        return regs[idx];
    }

    DEVICE const uint32_t& operator()(int idx) const {
        return regs[idx];
    }

    template <typename Tensor>
    DEVICE void load(Tensor& tensor, int lane_id) {
        auto coord = Helper::coord(lane_id);

        T* ptr = &tensor(coord);

        using Ldmatrix = typename Helper::Ldmatrix;
        if constexpr (REGS_PER_LANE == 4) {
            Ldmatrix::load(ptr, regs[0], regs[1], regs[2], regs[3]);
        } else if constexpr (REGS_PER_LANE == 2) {
            Ldmatrix::load(ptr, regs[0], regs[1]);
        } else if constexpr (REGS_PER_LANE == 1) {
            Ldmatrix::load(ptr, regs[0]);
        }
    }

    DEVICE void clear() {
        #pragma unroll
        for (int i = 0; i < REGS_PER_LANE; ++i) {
            regs[i] = 0;
        }
    }

    DEVICE constexpr int size() const {
        return REGS_PER_LANE;
    }

    DEVICE uint32_t* data_ptr() {
        return regs;
    }

    DEVICE uint32_t const* data_ptr() const {
        return regs;
    }
};

template <typename T, int N>
struct AccumFragment {
    static_assert(std::is_same_v<T, float>, "AccumFragment currently supports float accumulator only");
    static_assert(N > 0, "AccumFragment size must be positive");

    static constexpr int kRegs = N;

    T regs[N];

    DEVICE constexpr AccumFragment() {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            regs[i] = T(0);
        }
    }

    DEVICE T& operator()(int idx) {
        return regs[idx];
    }

    DEVICE const T& operator()(int idx) const {
        return regs[idx];
    }

    DEVICE void clear() {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            regs[i] = T(0);
        }
    }

    DEVICE constexpr int size() const {
        return N;
    }

    DEVICE T* data_ptr() {
        return regs;
    }

    DEVICE const T* data_ptr() const {
        return regs;
    }
};

template <typename T>
struct is_fragment : cxx::false_type {};

template <typename T, typename Helper, int ROWS, int COLS>
struct is_fragment<Fragment<T, Helper, ROWS, COLS>> : cxx::true_type {};

template <typename T, int N>
struct is_fragment<AccumFragment<T, N>> : cxx::true_type {};

template <typename T>
constexpr bool is_fragment_v = is_fragment<cxx::remove_cv_t<cxx::remove_reference_t<T>>>::value;

template <typename Frag>
struct FragmentView {
    Frag& frag;

    DEVICE constexpr explicit FragmentView(Frag& frag_) : frag(frag_) {}

    DEVICE auto& operator()(int i) {
        return frag(i);
    }

    DEVICE auto const& operator()(int i) const {
        return frag(i);
    }

    DEVICE constexpr int size() const {
        return frag.size();
    }
};

template <typename Frag>
DEVICE constexpr auto make_fragment_view(Frag& frag) {
    return FragmentView<Frag>{frag};
}

template<typename T, typename Helper, int ROWS, int COLS, typename Tensor>
__device__ void load_warp_block(Tensor& tensor) {
    int lane_id = threadIdx.x % 32;
    Fragment<T, Helper, ROWS, COLS> frag;

    frag.load(tensor, lane_id);
}

}
