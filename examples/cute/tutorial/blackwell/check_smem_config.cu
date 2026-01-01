#include <cuda_runtime.h>
#include <iostream>

__global__ void dummy_kernel() {}

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    std::cout << "Shared Memory Per Block (static): " << props.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared Memory Per Block (opt-in): " << props.sharedMemPerBlockOptin << " bytes" << std::endl;
    std::cout << "Shared Memory Per Multiprocessor: " << props.sharedMemPerMultiprocessor << " bytes" << std::endl;
    
    // Try to set larger shared memory
    size_t smemSize = 50000;  // 50KB
    cudaError_t err = cudaFuncSetAttribute(
        dummy_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smemSize
    );
    
    if (err != cudaSuccess) {
        std::cerr << "cudaFuncSetAttribute failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Successfully set dynamic SMEM to " << smemSize << " bytes" << std::endl;
    }
    
    return 0;
}
