/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe Tutorial: TiledMMA, ThrMMA 和线程坐标概念
// 本示例演示 MMA 相关的核心线程概念

#include <iostream>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

using namespace cute;

// 简化的 kernel 来演示概念
__global__ void mma_thread_demo_kernel(int total_blocks_m, int total_blocks_n)
{
  // ═══════════════════════════════════════════════════════
  // 1. 线程和 Block 的基本信息
  // ═══════════════════════════════════════════════════════
  int thread_id = threadIdx.x;
  int block_id_x = blockIdx.x;
  int block_id_y = blockIdx.y;
  int block_dim = blockDim.x;

  // 只让一个线程打印，避免输出混乱
  bool is_first_thread = (thread_id == 0);
  bool is_first_block = (block_id_x == 0 && block_id_y == 0);

  if (is_first_thread && is_first_block) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("1. 基本线程信息\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Grid 配置: gridDim.x=%d, gridDim.y=%d\n", gridDim.x, gridDim.y);
    printf("Block 配置: blockDim.x=%d\n", block_dim);
    printf("总共 Block 数: %d (= %d × %d)\n", gridDim.x * gridDim.y, gridDim.x, gridDim.y);
    printf("每个 Block 线程数: %d\n", block_dim);
    printf("总线程数: %d\n", gridDim.x * gridDim.y * block_dim);
    printf("\n");
  }
  __syncthreads();

  // ═══════════════════════════════════════════════════════
  // 2. Cluster Layout (VMNK 坐标系统)
  // ═══════════════════════════════════════════════════════
  if (is_first_thread && is_first_block) {
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("2. VMNK 坐标系统\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("V (Variant/Peer): CTA 在 Cluster 中的编号\n");
    printf("M: MMA 在 M 维度的位置\n");
    printf("N: MMA 在 N 维度的位置\n");
    printf("K: MMA 在 K 维度的位置 (通常用 _ 表示所有)\n");
    printf("\n");
  }
  __syncthreads();

  // 假设 Cluster 配置: (2, 1, 1) - 2 个 CTA 在 V 维度
  int cluster_size_v = 2;

  // 计算 VMNK 坐标
  int mma_v = block_id_x % cluster_size_v;           // V: Peer CTA 编号
  int mma_m = block_id_x / cluster_size_v;           // M: M 维度位置
  int mma_n = block_id_y;                            // N: N 维度位置

  // 打印前几个 Block 的坐标
  if (is_first_thread && block_id_x < 4 && block_id_y < 2) {
    printf("Block[%d,%d] 的 VMNK 坐标: V=%d, M=%d, N=%d, K=_\n",
           block_id_x, block_id_y, mma_v, mma_m, mma_n);
  }
  __syncthreads();

  // ═══════════════════════════════════════════════════════
  // 3. TiledMMA 概念（主机端创建，这里只是说明）
  // ═══════════════════════════════════════════════════════
  if (is_first_thread && is_first_block) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("3. TiledMMA 概念\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("TiledMMA = Cluster 级别的 MMA 描述\n");
    printf("  - 定义了整个 Cluster 的 MMA 模式\n");
    printf("  - 包含 MMA 指令的形状: (M, N, K)\n");
    printf("  - 例如: SM100_MMA (128, 256, 16)\n");
    printf("  - 表示单个 MMA 指令处理 128×256×16 的数据\n");
    printf("\n");
    printf("在代码中创建:\n");
    printf("  TiledMMA tiled_mma = make_tiled_mma(\n");
    printf("      SM100_MMA_F16BF16_SS<...>{128, 256, ...}\n");
    printf("  );\n");
    printf("\n");
  }
  __syncthreads();

  // ═══════════════════════════════════════════════════════
  // 4. ThrMMA 概念（从 TiledMMA 获取）
  // ═══════════════════════════════════════════════════════
  if (is_first_thread && is_first_block) {
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("4. ThrMMA 概念\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("ThrMMA = 单个 CTA 的 MMA 操作符\n");
    printf("  - 从 TiledMMA 中提取特定 CTA 的视图\n");
    printf("  - 根据 V 坐标 (mma_v) 来区分不同的 CTA\n");
    printf("\n");
    printf("在代码中获取:\n");
    printf("  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);\n");
    printf("  //                                   ^^^^^^\n");
    printf("  //                            V 坐标 (0, 1, 2, ...)\n");
    printf("\n");
  }
  __syncthreads();

  // ═══════════════════════════════════════════════════════
  // 5. 数据流示例
  // ═══════════════════════════════════════════════════════
  if (is_first_thread && block_id_x < 2 && block_id_y == 0) {
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("5. Block[%d,0] (V=%d, M=%d) 的数据流\n", block_id_x, mma_v, mma_m);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");
    printf("步骤 1: 提取 CTA 的数据 (local_tile)\n");
    printf("  gA = local_tile(mA, mma_tiler, mma_coord)\n");
    printf("  // 根据 (V=%d, M=%d, N=0) 坐标提取数据\n", mma_v, mma_m);
    printf("\n");
    printf("步骤 2: 获取 CTA 的 MMA 操作符 (get_slice)\n");
    printf("  cta_mma = tiled_mma.get_slice(%d)\n", mma_v);
    printf("  // V=%d 表示这是 Cluster 中的第 %d 个 CTA\n", mma_v, mma_v);
    printf("\n");
    printf("步骤 3: 分区数据 (partition_A)\n");
    printf("  tCgA = cta_mma.partition_A(gA)\n");
    printf("  // 按 MMA 需求重新组织数据\n");
    printf("\n");
    printf("步骤 4: 执行 MMA\n");
    printf("  gemm(tiled_mma, tCrA, tCrB, tCtAcc)\n");
    printf("\n");
  }
  __syncthreads();

  // ═══════════════════════════════════════════════════════
  // 6. 完整的坐标映射示例
  // ═══════════════════════════════════════════════════════
  if (is_first_thread && is_first_block) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("6. 完整坐标映射示例\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");
    printf("假设: GEMM 512×1024, MMA Tile 128×256\n");
    printf("Grid: %d × %d blocks\n", total_blocks_m, total_blocks_n);
    printf("Cluster: 2 CTAs (V=0,1)\n");
    printf("\n");
    printf("Block 坐标 → VMNK 坐标 → 处理的数据范围\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (int bx = 0; bx < 4 && bx < total_blocks_m; bx++) {
      for (int by = 0; by < 2 && by < total_blocks_n; by++) {
        int v = bx % 2;
        int m = bx / 2;
        int n = by;
        int m_start = m * 128;
        int n_start = n * 256;
        printf("Block[%d,%d] → V=%d,M=%d,N=%d → C[%d:%d, %d:%d]\n",
               bx, by, v, m, n,
               m_start, m_start+128, n_start, n_start+256);
      }
    }
    printf("...\n");
    printf("\n");
  }
  __syncthreads();
}

// 主机端代码：演示如何创建和理解 TiledMMA
void host_tiled_mma_demo()
{
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "主机端: TiledMMA 创建和理解" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

  // 创建一个简化的 TiledMMA (用于 SM100)
  using TypeA = cute::half_t;
  using TypeB = cute::half_t;
  using TypeC = float;

  std::cout << "1. 创建 TiledMMA:" << std::endl;
  std::cout << "   TiledMMA tiled_mma = make_tiled_mma(" << std::endl;
  std::cout << "       SM100_MMA_F16BF16_SS<half, half, float," << std::endl;
  std::cout << "                            128, 256,  // M, N" << std::endl;
  std::cout << "                            K, K>{}    // A, B layout" << std::endl;
  std::cout << "   );" << std::endl;
  std::cout << std::endl;

  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC,
                           128, 256,
                           UMMA::Major::K, UMMA::Major::K>{}
  );

  std::cout << "2. TiledMMA 的结构:" << std::endl;
  std::cout << "   "; print(tiled_mma); std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "3. 关键信息:" << std::endl;
  std::cout << "   MMA Shape (M, N, K): "
            << tile_size<0>(tiled_mma) << " × "
            << tile_size<1>(tiled_mma) << " × "
            << tile_size<2>(tiled_mma) << std::endl;
  std::cout << "   这是单个 MMA 指令的大小" << std::endl;
  std::cout << std::endl;

  std::cout << "4. 使用流程:" << std::endl;
  std::cout << "   a) 在主机创建 TiledMMA" << std::endl;
  std::cout << "   b) 传递给 kernel" << std::endl;
  std::cout << "   c) 在 kernel 中用 get_slice(mma_v) 获取 ThrMMA" << std::endl;
  std::cout << "   d) 用 ThrMMA 进行 partition 和 MMA 操作" << std::endl;
  std::cout << std::endl;
}

int main()
{
  std::cout << "═══════════════════════════════════════════════════════" << std::endl;
  std::cout << "   TiledMMA, ThrMMA 和线程坐标概念教程" << std::endl;
  std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;

  // 主机端演示
  host_tiled_mma_demo();

  // 模拟 GEMM 问题: 512×1024
  // MMA Tile: 128×256
  // 需要 (512/128) × (1024/256) = 4×4 = 16 个 blocks
  int gemm_m = 512;
  int gemm_n = 1024;
  int mma_tile_m = 128;
  int mma_tile_n = 256;

  int num_blocks_m = gemm_m / mma_tile_m;  // 4
  int num_blocks_n = gemm_n / mma_tile_n;  // 4

  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "Kernel 参数:" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
  std::cout << "GEMM 大小: " << gemm_m << "×" << gemm_n << std::endl;
  std::cout << "MMA Tile 大小: " << mma_tile_m << "×" << mma_tile_n << std::endl;
  std::cout << "需要 Blocks: " << num_blocks_m << "×" << num_blocks_n
            << " = " << num_blocks_m * num_blocks_n << std::endl;
  std::cout << "每个 Block 线程数: 128" << std::endl;
  std::cout << std::endl;

  // 启动 kernel
  dim3 gridDim(num_blocks_m, num_blocks_n, 1);
  dim3 blockDim(128, 1, 1);

  std::cout << "启动 Kernel..." << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

  mma_thread_demo_kernel<<<gridDim, blockDim>>>(num_blocks_m, num_blocks_n);
  cudaDeviceSynchronize();

  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "总结" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

  std::cout << "关键概念:" << std::endl;
  std::cout << "1. TiledMMA: Cluster 级别的 MMA 描述（整体）" << std::endl;
  std::cout << "2. ThrMMA: 单个 CTA 的 MMA 操作符（局部）" << std::endl;
  std::cout << "3. VMNK 坐标: V=Peer, M=行位置, N=列位置, K=深度" << std::endl;
  std::cout << "4. get_slice(mma_v): 从 TiledMMA 提取 ThrMMA" << std::endl;
  std::cout << "5. partition_A/B/C: 用 ThrMMA 分区数据" << std::endl;
  std::cout << std::endl;

  std::cout << "数据流:" << std::endl;
  std::cout << "  TiledMMA (整体)" << std::endl;
  std::cout << "      ↓ get_slice(mma_v)" << std::endl;
  std::cout << "  ThrMMA (CTA 视图)" << std::endl;
  std::cout << "      ↓ partition_A(gA)" << std::endl;
  std::cout << "  tCgA (重组后的数据)" << std::endl;
  std::cout << "      ↓ gemm(...)" << std::endl;
  std::cout << "  执行 MMA 计算" << std::endl;
  std::cout << std::endl;

  std::cout << "下一步: 在 VSCode 中调试 01_mma_sm100.cu" << std::endl;
  std::cout << "  - 在第 204 行设置断点 (get_slice)" << std::endl;
  std::cout << "  - 在第 205 行设置断点 (partition_A)" << std::endl;
  std::cout << "  - 观察 mma_v, cta_mma, tCgA 的值" << std::endl;
  std::cout << std::endl;

  return 0;
}
