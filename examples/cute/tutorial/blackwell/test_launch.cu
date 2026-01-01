#include <cuda_runtime.h>
#include <iostream>

__global__ void simple_kernel() {
    printf("Kernel launched successfully!\n");
}

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Max shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    
    simple_kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Success!" << std::endl;
    return 0;
}
