/***************************************************************************************************
 * Analysis of thread-to-tile A layout from mma_matmul_1_0
 *
 * From kernel_1.cu lines 60-67:
 * aReg[0] = As[mWarp + groupID    ][groupLaneID*2    ];  // (g, 2l)
 * aReg[1] = As[mWarp + groupID    ][groupLaneID*2 + 1];  // (g, 2l+1)
 * aReg[2] = As[mWarp + groupID + 8][groupLaneID*2    ];  // (g+8, 2l)
 * aReg[3] = As[mWarp + groupID + 8][groupLaneID*2 + 1];  // (g+8, 2l+1)
 * aReg[4] = As[mWarp + groupID    ][groupLaneID*2 + 8];  // (g, 2l+8)
 * aReg[5] = As[mWarp + groupID    ][groupLaneID*2 + 9];  // (g, 2l+9)
 * aReg[6] = As[mWarp + groupID + 8][groupLaneID*2 + 8];  // (g+8, 2l+8)
 * aReg[7] = As[mWarp + groupID + 8][groupLaneID*2 + 9];  // (g+8, 2l+9)
 *
 * Where:
 * - groupID = laneID / 4     (ranges 0-7)
 * - groupLaneID = laneID % 4 (ranges 0-3)
 * - Tile size: 16x16 (M x K)
 * - 32 threads per warp, each thread holds 8 elements (aReg[0-7])
 **************************************************************************************************/

#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
  std::cout << "=== Thread-to-Tile A Layout Analysis ===" << std::endl << std::endl;

  // The access pattern for a single thread (groupID=g, groupLaneID=l):
  // Elements accessed in 16x16 tile (M x K coordinates):
  // (g, 2l), (g, 2l+1), (g+8, 2l), (g+8, 2l+1), (g, 2l+8), (g, 2l+9), (g+8, 2l+8), (g+8, 2l+9)

  // Linear offsets (assuming row-major As[m][k] → m*16 + k):
  // 16g + 2l: [0, 1, 128, 129, 8, 9, 136, 137]
  //           └─┬─┘ └──┬──┘  └─┬─┘ └──┬──┘
  //           base   +128    +8   +128+8

  // CuTe Layout representation for a SINGLE thread's fragment:
  // Shape:  (2, 2, 2)
  // Stride: (1, 128, 8)
  //
  // This describes how the 8 register elements (aReg[0-7]) map to
  // the 16x16 tile, assuming row-major layout As[m][k] → m*16 + k
  //
  // In CuTe, the first mode varies fastest (column-major indexing).
  // Breakdown:
  // - First mode (size 2, stride 1): consecutive K elements
  //     aReg[even] vs aReg[odd] → k and k+1
  // - Second mode (size 2, stride 128): M dimension
  //     toggles between base M and M+8 → (aReg[0-1,4-5] vs aReg[2-3,6-7])
  // - Third mode (size 2, stride 8): K chunks
  //     toggles between K chunks [0-1] and [8-9] → (aReg[0-3] vs aReg[4-7])
  //
  // Pattern: [0, 1, 128, 129, 8, 9, 136, 137]

  auto A_frag_layout =
    Layout<
      Shape<Int<2>, Int<2>, Int<2>>,
      Stride<Int<1>, Int<128>, Int<8>>
    >{};

  std::cout << "Thread fragment A layout:" << std::endl;
  std::cout << A_frag_layout << std::endl << std::endl;

  std::cout << "Total size: " << size(A_frag_layout) << " elements" << std::endl;
  std::cout << std::endl;

  // Verify with specific examples
  std::cout << "=== Verification ===" << std::endl;
  std::cout << "For thread with groupID=3, groupLaneID=2:" << std::endl;
  std::cout << "  Expected As indices: [3][4], [3][5], [11][4], [11][5], [3][12], [3][13], [11][12], [11][13]" << std::endl;
  std::cout << "  Linear offsets (m*16+k): 52, 53, 180, 181, 60, 61, 188, 189" << std::endl;
  std::cout << std::endl;

  // Compute base offset for this thread
  int groupID = 3;
  int groupLaneID = 2;
  int base_m = groupID;        // 3
  int base_k = 2 * groupLaneID; // 4
  int base_offset = base_m * 16 + base_k; // 52

  std::cout << "  Base offset (groupID=3, groupLaneID=2): " << base_offset << std::endl;
  std::cout << "  Layout computed offsets:" << std::endl;

  // Map through the layout
  for (int i = 0; i < 8; i++) {
    int offset = A_frag_layout(i);
    std::cout << "    aReg[" << i << "] → offset " << offset;
    if (i == 0) std::cout << " (base)";
    if (i == 1) std::cout << " (+1)";
    if (i == 2) std::cout << " (+128, next M)";
    if (i == 3) std::cout << " (+129)";
    if (i == 4) std::cout << " (+8, next K chunk)";
    if (i == 5) std::cout << " (+9)";
    if (i == 6) std::cout << " (+136)";
    if (i == 7) std::cout << " (+137)";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // Show 2D collapsed version for visualization
  std::cout << "=== 2D Collapsed Layout for Visualization ===" << std::endl;
  auto A_frag_2d = make_layout(make_shape(Int<4>{}, Int<2>{}),
                                make_stride(Int<2>{}, Int<1>{}));
  std::cout << "Simplified 2D view (4x2): " << A_frag_2d << std::endl;
  std::cout << "Note: This doesn't capture the full 3D structure with stride 128" << std::endl;
  std::cout << std::endl;

  return 0;
}
