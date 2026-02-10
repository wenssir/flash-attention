#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceId = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    std::cout << "--- GPU Hardware Properties for " << prop.name << " ---" << std::endl;
    
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;

    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << std::endl;

    std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;

    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;

    return 0;
}