#pragma once

namespace constants {
    constexpr int WARP_SIZE = 32;
    constexpr int BYTES_PER_FP16 = 2;
    constexpr int BYTES_PER_FP32 = 4;

    constexpr int BLOCK_M = 128; // for q
    constexpr int BLOCK_N = 64; // for k/v
    constexpr int HEAD_DIM = 64;

    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
};