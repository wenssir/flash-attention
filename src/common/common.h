#pragma once

namespace constants {
    constexpr int WARP_SIZE = 32;
    constexpr int BYTES_PER_FP16 = 2;
    constexpr int BYTES_PER_FP32 = 4;

    constexpr int HEAD_DIM = 128;
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int WARPS = 4;

    constexpr int THREADS_PER_BLOCK = WARPS * WARP_SIZE;
    constexpr int WARPS_PER_BLOCK = WARPS;
};
