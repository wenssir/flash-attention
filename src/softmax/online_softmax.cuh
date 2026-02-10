#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "../config/macros.cuh"

namespace softmax {

template <int Rows>
struct OnlineSoftmaxState {
    float m[Rows];
    float l[Rows];
};

DEVICE float compute_p(float s, float m_new, float softmax_scale) {
    return exp2f(s * softmax_scale - m_new * softmax_scale);
}

DEVICE float renorm_coeff(float m_old, float m_new, float softmax_scale) {
    return exp2f((m_old - m_new) * softmax_scale);
}


template <int Rows>
DEVICE void online_softmax_init(OnlineSoftmaxState<Rows>& st) {
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        st.m[r] = -INFINITY;
        st.l[r] = 0.0f;
    }
}

template <int Rows, int Cols, typename RenormAccFn>
DEVICE void online_softmax_step(float (&scores)[Rows][Cols],
                                OnlineSoftmaxState<Rows>& st,
                                float softmax_scale,
                                bool is_first,
                                RenormAccFn&& renorm_acc) {
    float m_new[Rows];

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float v = scores[r][0];
        #pragma unroll
        for (int c = 1; c < Cols; ++c) {
            v = fmaxf(v, scores[r][c]);
        }
        m_new[r] = is_first ? v : fmaxf(st.m[r], v);
    }

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float alpha = renorm_coeff(st.m[r], m_new[r], softmax_scale);
        if (!is_first) {
            renorm_acc(r, alpha);
        } else {
            alpha = 0.0f;
        }

        float l_new = is_first ? 0.0f : st.l[r] * alpha;
        #pragma unroll
        for (int c = 0; c < Cols; ++c) {
            float p = compute_p(scores[r][c], m_new[r], softmax_scale);
            scores[r][c] = p;
            l_new += p;
        }
        st.m[r] = m_new[r];
        st.l[r] = l_new;
    }
}

template <int Rows, typename NormalizeAccFn>
DEVICE void online_softmax_finalize(OnlineSoftmaxState<Rows>& st,
                                    NormalizeAccFn&& normalize_acc) {
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float inv_l = 1.0f / st.l[r];
        normalize_acc(r, inv_l);
    }
}

} // namespace softmax
