#pragma once

namespace constants {
    constexpr int WARP_SIZE = 32;
    constexpr int BYTES_PER_FP16 = 2;
    constexpr int BYTES_PER_FP32 = 4;

    constexpr int V3_HEAD_DIM = 128;
    constexpr int V3_BLOCK_M = 64;
    constexpr int V3_BLOCK_N = 64;
    constexpr int V3_WARPS = 4;

    constexpr int THREADS_PER_BLOCK = V3_WARPS * WARP_SIZE;
    constexpr int WARPS_PER_BLOCK = V3_WARPS;
};
