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
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Fetch GoogleTest (gtest) for unit testing
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

    # Container
    src/container/tuple.cuh

    # Numeric
    src/numeric/Int.cuh
    src/numeric/real.h

    # Layout
    src/layout/stride.cuh
    src/layout/shape.h
    src/layout/coordinate.h
    src/layout/offset.cuh
    src/layout/layout.h
    src/layout/composedLayout.h

    # PTX
    src/ptx/ptx.cuh
    src/ptx/copy.cuh
    src/ptx/mma_atom.cuh

    # Tensor Core
    src/tensor_core/fragment.cuh
    src/tensor_core/accum_layout_traits.cuh
    src/tensor_core/swizzle.cuh
    src/tensor_core/tensor.cuh

    # Softmax
    src/softmax/online_softmax.cuh

    # Load/Store
    src/loadstore/load_func.cuh
    src/loadstore/store_func.cuh
    src/loadstore/copy_gsm.cuh
    src/loadstore/copy_srm.cuh
    src/loadstore/ldmatrix_coord.cuh

    # Utils
    src/utils/util_func.cuh
    src/utils/print.h

    # Forward
    src/forward/forward_v1_naive.h
    src/forward/forward_v2_float4_double_buffer_prefetch.h
    src/forward/forward_v3_tensor_core.cuh
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
    include(GoogleTest)

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

    # test_shape (gtest unit tests)
    add_executable(test_shape tests/layout/test_shape.cpp)
    target_link_libraries(test_shape PRIVATE gtest_main)
    gtest_discover_tests(test_shape)

    # test_layout (gtest unit tests)
    add_executable(test_layout tests/layout/test_layout.cpp)
    target_link_libraries(test_layout PRIVATE gtest_main)
    gtest_discover_tests(test_layout)

    # test_divide (gtest unit tests)
    add_executable(test_divide tests/layout/test_divide.cpp)
    target_link_libraries(test_divide PRIVATE gtest_main)
    gtest_discover_tests(test_divide)

    # test_ComposedLayout (gtest unit tests)
    add_executable(test_ComposedLayout tests/layout/test_ComposedLayout.cpp)
    target_link_libraries(test_ComposedLayout PRIVATE gtest_main)
    gtest_discover_tests(test_ComposedLayout)

    # test_tuple (gtest unit tests)
    add_executable(test_tuple tests/container/test_tuple.cpp)
    target_link_libraries(test_tuple PRIVATE gtest_main)
    gtest_discover_tests(test_tuple)

    # test_fragment_layout (gtest unit tests)
    add_executable(test_fragment_layout tests/test_fragment_layout.cu)
    target_link_libraries(test_fragment_layout PRIVATE fa::fa gtest_main)
    set_target_properties(test_fragment_layout PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_fragment_layout)

    # test_mma_fragment (standalone CUDA smoke test)
    add_executable(test_mma_fragment tests/ptx/test_mma_fragment.cu)
    target_link_libraries(test_mma_fragment PRIVATE fa::fa)
    set_target_properties(test_mma_fragment PROPERTIES
        CUDA_ARCHITECTURES "80;86;89;90"
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME test_mma_fragment COMMAND test_mma_fragment)

    # test_forward_v3_correctness (standalone CUDA correctness test)
    add_executable(test_forward_v3_correctness tests/test_forward_v3_correctness.cu)
    target_link_libraries(test_forward_v3_correctness PRIVATE fa::fa)
    set_target_properties(test_forward_v3_correctness PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME test_forward_v3_correctness COMMAND test_forward_v3_correctness)

    # test_copy_gsm (gtest unit tests)
    add_executable(test_copy_gsm tests/loadstore/test_copy_gsm.cu)
    target_link_libraries(test_copy_gsm PRIVATE fa::fa gtest_main)
    set_target_properties(test_copy_gsm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_copy_gsm)

    # test_copy_gsm_v3_bounds (gtest unit tests)
    add_executable(test_copy_gsm_v3_bounds tests/loadstore/test_copy_gsm_v3_bounds.cu)
    target_link_libraries(test_copy_gsm_v3_bounds PRIVATE fa::fa gtest_main)
    set_target_properties(test_copy_gsm_v3_bounds PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_copy_gsm_v3_bounds)

    # test_copy_gsm_tile_shapes (gtest unit tests)
    add_executable(test_copy_gsm_tile_shapes tests/loadstore/test_copy_gsm_tile_shapes.cu)
    target_link_libraries(test_copy_gsm_tile_shapes PRIVATE fa::fa gtest_main)
    set_target_properties(test_copy_gsm_tile_shapes PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_copy_gsm_tile_shapes)

    # test_thread_slice (gtest unit tests)
    add_executable(test_thread_slice tests/loadstore/test_thread_slice.cu)
    target_link_libraries(test_thread_slice PRIVATE fa::fa gtest_main)
    set_target_properties(test_thread_slice PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_thread_slice)

    # test_tensor (gtest unit tests)
    add_executable(test_tensor tests/tensor/test_tensor.cu)
    target_link_libraries(test_tensor PRIVATE fa::fa gtest_main)
    set_target_properties(test_tensor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_tensor)

    # test_copy_srm (gtest unit tests)
    add_executable(test_copy_srm tests/loadstore/test_copy_srm.cu)
    target_link_libraries(test_copy_srm PRIVATE fa::fa gtest_main)
    set_target_properties(test_copy_srm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_copy_srm)

    # test_copy_atom (gtest unit tests)
    add_executable(test_copy_atom tests/loadstore/test_copy_atom.cu)
    target_link_libraries(test_copy_atom PRIVATE fa::fa gtest_main)
    set_target_properties(test_copy_atom PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    gtest_discover_tests(test_copy_atom)

    # Direct run targets for v3 debug/correctness tests
    add_custom_target(run_test_copy_gsm_v3_bounds
        COMMAND $<TARGET_FILE:test_copy_gsm_v3_bounds>
        DEPENDS test_copy_gsm_v3_bounds
        USES_TERMINAL
    )

    add_custom_target(run_test_copy_gsm_tile_shapes
        COMMAND $<TARGET_FILE:test_copy_gsm_tile_shapes>
        DEPENDS test_copy_gsm_tile_shapes
        USES_TERMINAL
    )

    add_custom_target(run_test_forward_v3_correctness
        COMMAND $<TARGET_FILE:test_forward_v3_correctness>
        DEPENDS test_forward_v3_correctness
        USES_TERMINAL
    )

    add_custom_target(run_test_forward_v3_all
        DEPENDS
            run_test_copy_gsm_v3_bounds
            run_test_copy_gsm_tile_shapes
            run_test_forward_v3_correctness
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

    # vector_add (basic CUDA example) - only if file exists
    if(EXISTS "${PROJECT_SOURCE_DIR}/vector_add.cu")
        add_executable(vector_add vector_add.cu)
        target_link_libraries(vector_add PRIVATE CUDA::cudart)
    endif()
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
