/***************************************************************************************************
 * Analysis of warp A layout for m16n8k16 instruction
 **************************************************************************************************/

#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
  std::cout << "=== Warp A Layout Analysis for m16n8k16 ===" << std::endl << std::endl;

  // User's proposed layout
  auto warp_A_layout = Layout<
    Shape <Shape<Int<4>, Int<16>>, Shape<Int<8>, Int<16>>>,
    Stride<Stride<Int<2048>, Int<128>>, Stride<Int<16>, Int<1>>>
  >{};

  std::cout << "Proposed warp A layout:" << std::endl;
  std::cout << warp_A_layout << std::endl << std::endl;

  std::cout << "Total size: " << size(warp_A_layout) << " elements" << std::endl;
  std::cout << std::endl;

  // Analyze the structure
  std::cout << "=== Layout Structure Analysis ===" << std::endl;
  std::cout << "Shape: ((4, 16), (8, 16))" << std::endl;
  std::cout << "  - Mode 0: (4, 16)  → 4 * 16 = 64" << std::endl;
  std::cout << "  - Mode 1: (8, 16)  → 8 * 16 = 128" << std::endl;
  std::cout << "  - Total: 64 * 128 = 8192 elements" << std::endl;
  std::cout << std::endl;

  std::cout << "Stride: ((2048, 128), (16, 1))" << std::endl;
  std::cout << "  - Mode 0 strides: (2048, 128)" << std::endl;
  std::cout << "  - Mode 1 strides: (16, 1)" << std::endl;
  std::cout << std::endl;

  // Compare with m16n8k16 requirements
  std::cout << "=== m16n8k16 Instruction Requirements ===" << std::endl;
  std::cout << "For A matrix:" << std::endl;
  std::cout << "  - Single MMA tile: 16×16 (M×K) = 256 elements" << std::endl;
  std::cout << "  - Warp size: 32 threads" << std::endl;
  std::cout << "  - Elements per thread: 256/32 = 8 elements" << std::endl;
  std::cout << std::endl;

  std::cout << "This layout has: 8192 elements" << std::endl;
  std::cout << "Number of MMA tiles: 8192/256 = " << (8192/256) << std::endl;
  std::cout << std::endl;

  // Interpret the layout dimensions
  std::cout << "=== Possible Interpretation ===" << std::endl;
  std::cout << "If this represents a shared memory tile:" << std::endl;
  std::cout << std::endl;

  // Try to infer the tile shape from strides
  std::cout << "From stride analysis:" << std::endl;
  std::cout << "  - Stride 2048: suggests M dimension increment" << std::endl;
  std::cout << "  - Stride 128: suggests one row in a K=128 tile" << std::endl;
  std::cout << "  - Stride 16: suggests K dimension = 16 (one MMA K)" << std::endl;
  std::cout << "  - Stride 1: consecutive K elements" << std::endl;
  std::cout << std::endl;

  std::cout << "Likely tile dimensions:" << std::endl;
  std::cout << "  - K dimension: 128 (from stride 128 = one row)" << std::endl;
  std::cout << "  - M dimension: 8192/128 = 64 rows" << std::endl;
  std::cout << "  - Tile shape: 64×128" << std::endl;
  std::cout << std::endl;

  // Check if this makes sense for multiple MMAs
  std::cout << "=== MMA Tiling ===" << std::endl;
  std::cout << "If processing 64×128 tile:" << std::endl;
  std::cout << "  - M direction: 64/16 = 4 MMA tiles" << std::endl;
  std::cout << "  - K direction: 128/16 = 8 K iterations" << std::endl;
  std::cout << "  - Total: 4 MMA tiles × 8 K-iterations" << std::endl;
  std::cout << std::endl;

  // Show first few elements
  std::cout << "=== First 8 Element Offsets ===" << std::endl;
  for (int i = 0; i < 8; i++) {
    std::cout << "  Element [" << i << "] → offset " << warp_A_layout(i) << std::endl;
  }
  std::cout << std::endl;

  // Compare with single-thread fragment layout
  std::cout << "=== Comparison with Single-Thread Fragment ===" << std::endl;
  auto thread_frag = Layout<
    Shape<Int<2>, Int<2>, Int<2>>,
    Stride<Int<1>, Int<128>, Int<8>>
  >{};
  std::cout << "Single thread fragment (from kernel_1.cu analysis):" << std::endl;
  std::cout << "  " << thread_frag << std::endl;
  std::cout << "  Pattern: [0, 1, 128, 129, 8, 9, 136, 137]" << std::endl;
  std::cout << std::endl;

  std::cout << "=== Recommendation ===" << std::endl;
  std::cout << "The proposed layout seems to describe a TILE (64×128), not a warp fragment." << std::endl;
  std::cout << "For a single warp processing m16n8k16:" << std::endl;
  std::cout << "  - If describing warp's data: should be 256 elements (16×16)" << std::endl;
  std::cout << "  - If describing per-thread: should be 8 elements" << std::endl;
  std::cout << "  - If describing shared memory tile: 8192 elements is valid" << std::endl;
  std::cout << std::endl;

  std::cout << "Consider what this layout represents:" << std::endl;
  std::cout << "  1. Shared memory tile → then it's appropriate" << std::endl;
  std::cout << "  2. Warp fragment → should be smaller (256 elements)" << std::endl;
  std::cout << "  3. Thread fragment → should be much smaller (8 elements)" << std::endl;

  return 0;
}
