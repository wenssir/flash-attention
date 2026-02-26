#pragma once

#include <vector>
#include <string>
#include <tuple>

struct TestConfig {
    std::string name;
    int B, H, N, d;
};

inline std::vector<TestConfig> get_default_configs() {
    return {
        {"tiny", 2, 4, 512, 64},
        {"small", 4, 8, 1024, 128},
        {"medium", 8, 16, 2048, 128},
        {"large", 8, 16, 4096, 128}
    };
}
