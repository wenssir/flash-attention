#pragma once 

#include <cuda_runtime.h>

namespace config {

struct KernelConfig {
    int block_m;
    int block_n;
    int head_dim;
    int threads_per_block;
    float scale;

    __host__ KernelConfig(int bm, int bn, int hd, int tpb)
        : block_m(bm), block_n(bn), head_dim(hd),
          threads_per_block(tpb), scale(1.0f / sqrtf(hd)) {}
};

}