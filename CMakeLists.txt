cmake_minimum_required(VERSION 3.18)

project(FlashAttentionV2
    VERSION 2.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Flash Attention V2 Implementation with float4/double-buffer/prefetch optimizations"
)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard" FORCE)
set(CMAKE_CUDA_STANDARD 20 CACHE STRING "CUDA standard" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(FA_BUILD_TESTS "Build test executables" ON)
option(FA_BUILD_UTILS "Build utility executables" ON)
option(FA_BUILD_EXAMPLES "Build example programs" ON)
set(FA_TEST_PROFILE "all" CACHE STRING "all;local_correctness;remote_ampere;remote_hopper")
set_property(CACHE FA_TEST_PROFILE PROPERTY STRINGS all local_correctness remote_ampere remote_hopper)

set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120" CACHE STRING "CUDA architectures")
if(DEFINED ENV{CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES})
elseif(FA_TEST_PROFILE STREQUAL "remote_ampere")
    set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures" FORCE)
elseif(FA_TEST_PROFILE STREQUAL "remote_hopper")
    set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

set(FA_HEADERS
    src/common/common.h
    src/config/config.cuh
    src/container/tuple.cuh
    src/numeric/Int.cuh
    src/numeric/real.h
    src/layout/stride.cuh
    src/layout/shape.h
    src/layout/coordinate.h
    src/layout/offset.cuh
    src/layout/layout.h
    src/layout/composedLayout.h
    src/ptx/ptx.cuh
    src/ptx/copy.cuh
    src/ptx/mma_atom.cuh
    src/tensor_core/fragment.cuh
    src/tensor_core/accum_layout_traits.cuh
    src/tensor_core/swizzle.cuh
    src/tensor_core/tensor.cuh
    src/softmax/online_softmax.cuh
    src/loadstore/load_func.cuh
    src/loadstore/store_func.cuh
    src/loadstore/thread_map.cuh
    src/loadstore/copy_gsm.cuh
    src/loadstore/copy_srm.cuh
    src/loadstore/ldmatrix_coord.cuh
    src/utils/util_func.cuh
    src/utils/print.h
    src/forward/forward_v1_naive.h
    src/forward/forward_v2_float4_double_buffer_prefetch.h
    src/forward/forward_v3_layout.cuh
    src/forward/forward_v4_mma.cuh
)

add_library(FlashAttentionV2 INTERFACE)
target_link_libraries(FlashAttentionV2 INTERFACE CUDA::cudart)
target_include_directories(FlashAttentionV2 INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
add_library(fa::fa ALIAS FlashAttentionV2)

if(FA_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)

    function(fa_add_cpp_gtest target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE gtest_main)
        gtest_discover_tests(${target})
    endfunction()

    function(fa_add_cuda_gtest target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE fa::fa gtest_main)
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        gtest_discover_tests(${target})
    endfunction()

    function(fa_add_cuda_smoke target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE fa::fa)
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    set(FA_BUILD_GPU_RUNTIME_TESTS ON)
    if(FA_TEST_PROFILE STREQUAL "local_correctness")
        set(FA_BUILD_GPU_RUNTIME_TESTS OFF)
    endif()

    fa_add_cpp_gtest(test_shape tests/unit/layout/test_shape.cpp)
    fa_add_cpp_gtest(test_layout tests/unit/layout/test_layout.cpp)
    fa_add_cpp_gtest(test_divide tests/unit/layout/test_divide.cpp)
    fa_add_cpp_gtest(test_ComposedLayout tests/unit/layout/test_ComposedLayout.cpp)
    fa_add_cpp_gtest(test_tuple tests/unit/container/test_tuple.cpp)
    fa_add_cuda_gtest(test_config_layout tests/unit/config/test_config_layout.cu)

    if(FA_BUILD_GPU_RUNTIME_TESTS)
        add_executable(test_flash_attention tests/integration/test_flash_attention.cu)
        target_link_libraries(test_flash_attention PRIVATE fa::fa)
        set_target_properties(test_flash_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME test_flash_attention COMMAND test_flash_attention)

        fa_add_cuda_gtest(test_fragment_layout tests/unit/layout/test_fragment_layout.cu)
        fa_add_cuda_gtest(test_tile_to_shape_runtime tests/unit/layout/test_tile_to_shape_runtime.cu)
        fa_add_cuda_smoke(test_mma_fragment tests/unit/ptx/test_mma_fragment.cu)

        if(EXISTS "${PROJECT_SOURCE_DIR}/tests/unit/ptx/test_ptx.cu")
            fa_add_cuda_smoke(test_ptx tests/unit/ptx/test_ptx.cu)
        endif()

        fa_add_cuda_gtest(test_tensor tests/unit/tensor/test_tensor.cu)
        fa_add_cuda_gtest(test_copy_gsm tests/unit/loadstore/test_copy_gsm.cu)
        fa_add_cuda_gtest(test_copy_gsm_v3_bounds tests/unit/loadstore/test_copy_gsm_v3_bounds.cu)
        fa_add_cuda_gtest(test_copy_gsm_tile_shapes tests/unit/loadstore/test_copy_gsm_tile_shapes.cu)
        fa_add_cuda_gtest(test_thread_slice tests/unit/loadstore/test_thread_slice.cu)
        fa_add_cuda_gtest(test_copy_srm tests/unit/loadstore/test_copy_srm.cu)
        fa_add_cuda_gtest(test_copy_atom tests/unit/loadstore/test_copy_atom.cu)

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

        add_custom_target(run_test_flash_attention
            COMMAND $<TARGET_FILE:test_flash_attention>
            DEPENDS test_flash_attention
            USES_TERMINAL
        )
    endif()
endif()

if(FA_BUILD_UTILS)
    add_executable(get_gpu_hardware_param utils/get_gpu_hardware_param.cu)
    target_link_libraries(get_gpu_hardware_param PRIVATE CUDA::cudart)
    add_executable(profile_kernel benchmark/cpp/benchmark.cu)
    target_link_libraries(profile_kernel PRIVATE fa::fa CUDA::cudart_static)
    set_target_properties(profile_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_executable(correctness_v4 benchmark/cpp/correctness_v4.cu)
    target_link_libraries(correctness_v4 PRIVATE fa::fa CUDA::cudart_static)
    set_target_properties(correctness_v4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_executable(correctness_v5 benchmark/cpp/correctness_v5.cu)
    target_link_libraries(correctness_v5 PRIVATE fa::fa CUDA::cudart_static)
    set_target_properties(correctness_v5 PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    add_executable(correctness_v6 benchmark/cpp/correctness_v6.cu)
    target_link_libraries(correctness_v6 PRIVATE fa::fa CUDA::cudart_static)
    set_target_properties(correctness_v6 PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(FA_BUILD_EXAMPLES)
    if(EXISTS "${PROJECT_SOURCE_DIR}/vector_add.cu")
        add_executable(vector_add vector_add.cu)
        target_link_libraries(vector_add PRIVATE CUDA::cudart)
    endif()
endif()

message(STATUS "")
message(STATUS "FlashAttentionV2")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build tests: ${FA_BUILD_TESTS}")
message(STATUS "Test profile: ${FA_TEST_PROFILE}")
message(STATUS "Build utils: ${FA_BUILD_UTILS}")
message(STATUS "Build examples: ${FA_BUILD_EXAMPLES}")
message(STATUS "")

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

set(CPACK_PACKAGE_NAME "FlashAttentionV2")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "FlashAttnV2")
set(CPACK_GENERATOR "TGZ;DEB")
include(CPack)
