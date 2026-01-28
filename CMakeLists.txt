cmake_minimum_required(VERSION 3.18)
project(FlashAttentionV2
    VERSION 2.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Flash Attention V2 Implementation with float4/double-buffer/prefetch optimizations"
)

# ============================================================
# Build Type
# ============================================================
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 17)

# ============================================================
# Options
# ============================================================
option(FA_BUILD_TESTS "Build test executables" ON)
option(FA_BUILD_UTILS "Build utility executables" ON)
option(FA_BUILD_EXAMPLES "Build example programs" ON)

# ============================================================
# CUDA Architectures
# ============================================================
# Default architectures (adjust based on your GPU)
#   sm_75: RTX 2080/Tesla T4
#   sm_80: A100
#   sm_86: RTX 3090/A40
#   sm_89: RTX 4090/L4
#   sm_90: H100/H800
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")

# Allow override via environment variable: CUDA_ARCHITECTURES="90" cmake ..
if(DEFINED ENV{CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES})
    message(STATUS "Custom CUDA architectures from env: $ENV{CUDA_ARCHITECTURES}")
endif()

# ============================================================
# Compile Flags
# ============================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

# Release flags
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# ============================================================
# Find Dependencies
# ============================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================
# Include Directories
# ============================================================
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

# ============================================================
# Source Files
# ============================================================
set(FA_HEADERS
    # Common
    src/common/common.h

    # Config
    src/config/config.cuh

    # Utils
    src/utils/util_func.cuh

    # Load/Store
    src/loadstore/load_func.cuh
    src/loadstore/store_func.cuh
    src/loadstore/ptx_func.cuh

    # Forward
    src/forward/forward_v1_naive.h
    src/forward/forward_v2_float4_double_buffer_prefetch.h
)

# ============================================================
# Library Target (header-only, but useful for IDE projects)
# ============================================================
add_library(FlashAttentionV2 INTERFACE)
target_link_libraries(FlashAttentionV2 INTERFACE CUDA::cudart)
target_include_directories(FlashAttentionV2 INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

# Alias for shorter name
add_library(fa::fa ALIAS FlashAttentionV2)

# ============================================================
# Test Executables
# ============================================================
if(FA_BUILD_TESTS)
    message(STATUS "Building tests: ON")

    # test_flash_attention
    add_executable(test_flash_attention tests/test_flash_attention.cu)
    target_link_libraries(test_flash_attention PRIVATE fa::fa)
    set_target_properties(test_flash_attention PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # profile_kernel
    add_executable(profile_kernel tests/profile_kernel.cu)
    target_link_libraries(profile_kernel PRIVATE fa::fa)
    set_target_properties(profile_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    message(STATUS "Building tests: OFF")
endif()

# ============================================================
# Utility Executables
# ============================================================
if(FA_BUILD_UTILS)
    message(STATUS "Building utils: ON")

    # get_gpu_hardware_param
    add_executable(get_gpu_hardware_param utils/get_gpu_hardware_param.cu)
    target_link_libraries(get_gpu_hardware_param PRIVATE CUDA::cudart)
else()
    message(STATUS "Building utils: OFF")
endif()

# ============================================================
# Example Programs
# ============================================================
if(FA_BUILD_EXAMPLES)
    message(STATUS "Building examples: ON")

    # vector_add (basic CUDA example)
    add_executable(vector_add vector_add.cu)
    target_link_libraries(vector_add PRIVATE CUDA::cudart)
else()
    message(STATUS "Building examples: OFF")
endif()

# ============================================================
# Print Configuration Summary
# ============================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Flash Attention V2 Configuration")
message(STATUS "========================================")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard:     ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA version:      ${CUDA_VERSION}")
message(STATUS "CUDA archs:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build tests:       ${FA_BUILD_TESTS}")
message(STATUS "Build utils:       ${FA_BUILD_UTILS}")
message(STATUS "Build examples:    ${FA_BUILD_EXAMPLES}")
message(STATUS "Build directory:   ${CMAKE_BINARY_DIR}")
message(STATUS "========================================")
message(STATUS "")

# ============================================================
# Installation (optional)
# ============================================================
include(GNUInstallDirs)

install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flash_attn_v2
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

install(TARGETS FlashAttentionV2
    EXPORT FlashAttentionV2Targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT FlashAttentionV2Targets
    FILE FlashAttentionV2Targets.cmake
    NAMESPACE fa::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlashAttentionV2
)

# ============================================================
# CPack (for creating packages)
# ============================================================
set(CPACK_PACKAGE_NAME "FlashAttentionV2")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "FlashAttnV2")
set(CPACK_GENERATOR "TGZ;DEB")

include(CPack)
