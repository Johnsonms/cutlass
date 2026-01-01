#include <iostream>

int main() {
    // A: 128×64 FP16 elements
    size_t A_size = 128 * 64 * 2;  // 2 bytes per FP16
    
    // B: 256×64 FP16 elements  
    size_t B_size = 256 * 64 * 2;
    
    // Barriers and other data
    size_t barriers = 8 + 8 + 4;  // mma_barrier + tma_barrier + tmem_base_ptr
    
    size_t total = A_size + B_size + barriers;
    
    std::cout << "A size: " << A_size << " bytes (" << A_size/1024.0 << " KB)" << std::endl;
    std::cout << "B size: " << B_size << " bytes (" << B_size/1024.0 << " KB)" << std::endl;
    std::cout << "Barriers: " << barriers << " bytes" << std::endl;
    std::cout << "Total SMEM: " << total << " bytes (" << total/1024.0 << " KB)" << std::endl;
    std::cout << "Max available: 49152 bytes (48 KB)" << std::endl;
    
    if (total > 49152) {
        std::cout << "ERROR: SMEM requirement exceeds hardware limit!" << std::endl;
    }
    
    return 0;
}
