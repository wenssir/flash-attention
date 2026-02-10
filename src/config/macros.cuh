#pragma once

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  include <cuda/std/tuple>
#  include <cuda/std/type_traits>
#  include <cuda/std/utility>
namespace cxx = cuda::std;
#else
#  include <tuple>
#  include <type_traits>
#  include <utility>
namespace cxx = std;
#endif

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  define HOST_DEVICE __forceinline__ __host__ __device__
#  define DEVICE      __forceinline__          __device__
#  define HOST        __forceinline__ __host__
#else
#  define HOST_DEVICE inline
#  define DEVICE      inline
#  define HOST        inline
#endif
