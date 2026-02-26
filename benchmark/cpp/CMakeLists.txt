cmake_minimum_required(VERSION 3.18)
project(FlashAttentionBenchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)  # A100

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")

find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../src
    ${CMAKE_SOURCE_DIR}/include
)

# Benchmark executable
add_executable(benchmark
    benchmark.cu
)

target_compile_features(benchmark PUBLIC cxx_std_17)
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Correctness executable
add_executable(correctness
    correctness.cu
)

target_compile_features(correctness PUBLIC cxx_std_17)
set_target_properties(correctness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
