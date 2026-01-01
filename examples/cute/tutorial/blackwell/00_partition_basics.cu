/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe Tutorial: Partition Basics - 理解数据如何分配给线程
// 本示例演示 partition 的核心概念

#include <iostream>
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

// 简单的 GPU kernel 来演示 partition
__global__ void partition_demo_kernel()
{
  // 每个线程的 ID
  int tid = threadIdx.x;

  // 创建一个 8×8 的共享内存数据
  __shared__ int shared_data[64];

  // 初始化：每个元素的值 = 它的索引
  if (tid < 64) {
    shared_data[tid] = tid;
  }
  __syncthreads();

  // 创建一个 tensor 表示这 8×8 的数据
  auto smem_layout = make_layout(make_shape(Int<8>{}, Int<8>{}),
                                 make_stride(Int<8>{}, Int<1>{}));
  Tensor sTensor = make_tensor(make_smem_ptr(shared_data), smem_layout);

  // 情况1：没有 partition - 所有线程看到完整的数据
  if (tid == 0) {
    printf("\n=== 情况1: 完整的 Tensor (8x8) ===\n");
    printf("所有线程都能看到全部 64 个元素\n");
    printf("sTensor shape: ");
    print(shape(sTensor));
    printf("\n");
  }
  __syncthreads();

  // 情况2：使用 local_tile 提取 2×4 的块
  if (tid == 0) {
    printf("\n=== 情况2: local_tile - 提取一个 2x4 的块 ===\n");

    // 提取位置 (2, 1) 的 2×4 块
    auto tile_shape = make_shape(Int<2>{}, Int<4>{});
    auto coord = make_coord(2, 1);  // 从第2行第1列开始

    Tensor sTile = local_tile(sTensor, tile_shape, coord);
    printf("sTile shape: ");
    print(shape(sTile));
    printf("\n");

    printf("sTile 的内容（行主序）:\n");
    for (int i = 0; i < 2; ++i) {
      printf("  [");
      for (int j = 0; j < 4; ++j) {
        printf("%2d ", sTile(i, j));
      }
      printf("]\n");
    }
  }
  __syncthreads();

  // 情况3：简单的线程级 partition
  if (tid == 0) {
    printf("\n=== 情况3: 手动 Partition - 16个线程分配64个元素 ===\n");
    printf("每个线程处理 4 个连续元素\n\n");
    printf("Thread  0 处理元素: [ 0  1  2  3]\n");
    printf("Thread  1 处理元素: [ 4  5  6  7]\n");
    printf("Thread  2 处理元素: [ 8  9 10 11]\n");
    printf("Thread  3 处理元素: [12 13 14 15]\n");
    printf("... (其余12个线程处理剩余48个元素)\n");
  }
  __syncthreads();

  // 情况4：2D partition - 更接近 MMA 的方式
  if (tid == 0) {
    printf("\n=== 情况4: 2D Partition - 4x4线程网格处理8x8数据 ===\n");
    printf("每个线程处理一个 2×2 的小块\n\n");
    printf("Thread[0,0] 处理块 [0:1, 0:1]:\n");
    printf("  [ 0  1]\n");
    printf("  [ 8  9]\n\n");
    printf("Thread[0,1] 处理块 [0:1, 2:3]:\n");
    printf("  [ 2  3]\n");
    printf("  [10 11]\n\n");
    printf("... (其余14个线程处理剩余的块)\n");
  }
  __syncthreads();

  // 情况5：使用 CuTe 的 logical_divide 来创建分层结构
  if (tid == 0) {
    printf("\n=== 情况5: logical_divide - 创建分层 Tile 结构 ===\n");

    // 将 8×8 按 2×2 分块
    auto tiled = logical_divide(smem_layout, make_shape(Int<2>{}, Int<2>{}));
    printf("原始 layout: ");
    print(smem_layout);
    printf("\n");
    printf("分块后: ");
    print(tiled);
    printf("\n");
    printf("解释: 分成 (4×4) 个 blocks，每个 block 是 (2×2)\n");
    printf("      第一组 (2,2) 表示每个 tile 内部的大小\n");
    printf("      第二组 (4,4) 表示有 4×4=16 个 tiles\n");
  }
  __syncthreads();
}

// 主机端示例：更详细的 partition 演示
void host_partition_demo()
{
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "主机端 Partition 演示" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

  // 创建一个 16×16 的数据
  int data[256];
  for (int i = 0; i < 256; ++i) data[i] = i;
  int* data_ptr = &data[0];  // 显式获取指针

  auto layout = make_layout(make_shape(Int<16>{}, Int<16>{}),
                            make_stride(Int<16>{}, Int<1>{}));
  auto tensor = make_tensor(data_ptr, layout);

  std::cout << "1. 原始 Tensor (16×16):" << std::endl;
  std::cout << "   shape: " << shape(tensor) << std::endl;
  std::cout << "   stride: " << stride(tensor) << std::endl;
  std::cout << std::endl;

  // 使用 logical_divide 分块
  auto tile_shape = make_shape(Int<4>{}, Int<4>{});
  auto tiled = logical_divide(layout, tile_shape);

  std::cout << "2. 使用 logical_divide 按 4×4 分块:" << std::endl;
  std::cout << "   分块后 layout: " << tiled << std::endl;
  std::cout << "   解释:" << std::endl;
  std::cout << "   - ((4,4),(4,4)) : 第一个 (4,4) 是每个 tile 的大小" << std::endl;
  std::cout << "                      第二个 (4,4) 是有 4×4=16 个 tiles" << std::endl;
  std::cout << std::endl;

  // 提取特定的 tile
  std::cout << "3. 提取第 (1,2) 个 tile（第1行第2列的4×4块）:" << std::endl;
  auto specific_tile = local_tile(tensor, tile_shape, make_coord(1, 2));
  std::cout << "   tile shape: " << shape(specific_tile) << std::endl;
  std::cout << "   tile 内容 (前两行):" << std::endl;
  std::cout << "   [" << specific_tile(0,0) << " " << specific_tile(0,1)
            << " " << specific_tile(0,2) << " " << specific_tile(0,3) << "]" << std::endl;
  std::cout << "   [" << specific_tile(1,0) << " " << specific_tile(1,1)
            << " " << specific_tile(1,2) << " " << specific_tile(1,3) << "]" << std::endl;
  std::cout << "   ..." << std::endl;
  std::cout << std::endl;

  // 演示如何模拟线程级 partition
  std::cout << "4. 模拟 4×4 线程网格的 Partition:" << std::endl;
  std::cout << "   假设有 16 个线程（4×4网格），每个线程处理一个 4×4 tile" << std::endl;
  std::cout << std::endl;

  for (int thread_m = 0; thread_m < 4; ++thread_m) {
    for (int thread_n = 0; thread_n < 4; ++thread_n) {
      int thread_id = thread_m * 4 + thread_n;
      auto thread_tile = local_tile(tensor, tile_shape, make_coord(thread_m, thread_n));

      // 只打印前 4 个线程的信息
      if (thread_id < 4) {
        std::cout << "   Thread[" << thread_m << "," << thread_n << "] (tid="
                  << thread_id << ") 处理 tile 起始位置: ("
                  << thread_m * 4 << ", " << thread_n * 4 << ")" << std::endl;
        std::cout << "   首元素值: " << thread_tile(0, 0) << std::endl;
        std::cout << std::endl;
      }
    }
  }

  std::cout << "5. 关键理解:" << std::endl;
  std::cout << "   ✓ Tile: 将大数据分成小块（空间划分）" << std::endl;
  std::cout << "   ✓ Partition: 将这些块分配给不同的处理单元（工作分配）" << std::endl;
  std::cout << "   ✓ logical_divide: 在 layout 层面描述分块结构" << std::endl;
  std::cout << "   ✓ local_tile: 提取特定坐标的 tile 数据" << std::endl;
  std::cout << std::endl;
}

// 演示 MMA-like partition（简化版）
void mma_like_partition_demo()
{
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "MMA-like Partition 演示（简化版）" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

  // 模拟一个 GEMM 问题：C[8,8] = A[8,4] × B[8,4]
  std::cout << "问题设定: C[8,8] = A[8,4] × B[8,4]^T" << std::endl;
  std::cout << std::endl;

  int A_data[32], B_data[32];
  for (int i = 0; i < 32; ++i) {
    A_data[i] = i;
    B_data[i] = i + 100;
  }
  int* A_ptr = &A_data[0];  // 显式获取指针
  int* B_ptr = &B_data[0];  // 显式获取指针

  auto layout_A = make_layout(make_shape(Int<8>{}, Int<4>{}),
                              make_stride(Int<4>{}, Int<1>{}));
  auto layout_B = make_layout(make_shape(Int<8>{}, Int<4>{}),
                              make_stride(Int<4>{}, Int<1>{}));

  auto A = make_tensor(A_ptr, layout_A);
  auto B = make_tensor(B_ptr, layout_B);

  std::cout << "1. 原始矩阵:" << std::endl;
  std::cout << "   A: " << shape(A) << std::endl;
  std::cout << "   B: " << shape(B) << std::endl;
  std::cout << std::endl;

  // 定义 MMA tile 大小
  auto mma_shape_MN = make_shape(Int<4>{}, Int<4>{});  // 每个 MMA 处理 4×4 的输出
  auto mma_shape_K = Int<2>{};  // 每次 MMA 累加 K=2

  std::cout << "2. MMA Tile 设定:" << std::endl;
  std::cout << "   MMA 输出: " << mma_shape_MN << " (4×4)" << std::endl;
  std::cout << "   MMA K: " << mma_shape_K << " (每次累加 K=2)" << std::endl;
  std::cout << std::endl;

  std::cout << "3. Partition 策略:" << std::endl;
  std::cout << "   - 输出 C[8,8] 分成 (2×2) 个 MMA tiles" << std::endl;
  std::cout << "   - 每个 MMA tile 是 4×4" << std::endl;
  std::cout << "   - K=4 需要累加 2 次（4/2=2）" << std::endl;
  std::cout << std::endl;

  std::cout << "4. CTA 0 负责的 MMA Tile (0,0):" << std::endl;

  // 提取 CTA 0 的 tile
  auto A_tile = local_tile(A, make_shape(Int<4>{}, Int<2>{}), make_coord(0, 0));
  auto B_tile = local_tile(B, make_shape(Int<4>{}, Int<2>{}), make_coord(0, 0));

  std::cout << "   A_tile[4,2]: 提取 A 的 [0:4, 0:2] 区域" << std::endl;
  std::cout << "   前两行: [" << A_tile(0,0) << " " << A_tile(0,1) << "]" << std::endl;
  std::cout << "           [" << A_tile(1,0) << " " << A_tile(1,1) << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "   B_tile[4,2]: 提取 B 的 [0:4, 0:2] 区域" << std::endl;
  std::cout << "   前两行: [" << B_tile(0,0) << " " << B_tile(0,1) << "]" << std::endl;
  std::cout << "           [" << B_tile(1,0) << " " << B_tile(1,1) << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "5. 执行流程（伪代码）:" << std::endl;
  std::cout << "   for k_tile = 0 to 1:  // K=4, 每次处理 K=2, 需要 2 次" << std::endl;
  std::cout << "       A_slice = A_tile[:, k_tile*2:(k_tile+1)*2]  // [4,2]" << std::endl;
  std::cout << "       B_slice = B_tile[:, k_tile*2:(k_tile+1)*2]  // [4,2]" << std::endl;
  std::cout << "       C_partial[4,4] += MMA(A_slice[4,2], B_slice[4,2]^T)" << std::endl;
  std::cout << std::endl;

  std::cout << "6. 关键点:" << std::endl;
  std::cout << "   ✓ Tile 定义了每个 CTA 处理的数据范围" << std::endl;
  std::cout << "   ✓ Partition 定义了 CTA 内部如何分配给线程/MMA单元" << std::endl;
  std::cout << "   ✓ K 维度的循环实现累加操作" << std::endl;
  std::cout << "   ✓ 在 01_mma_sm100.cu 中，这些概念完全一样，只是规模更大" << std::endl;
  std::cout << std::endl;
}

int main()
{
  std::cout << "═══════════════════════════════════════════════════════" << std::endl;
  std::cout << "   CuTe Partition 基础教程" << std::endl;
  std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;

  // 主机端演示
  host_partition_demo();

  // MMA-like 演示
  mma_like_partition_demo();

  // GPU kernel 演示
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
  std::cout << "GPU Kernel Partition 演示" << std::endl;
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;

  partition_demo_kernel<<<1, 256>>>();
  cudaDeviceSynchronize();

  std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
  std::cout << "教程完成！" << std::endl;
  std::cout << "═══════════════════════════════════════════════════════" << std::endl;

  std::cout << "\n关键要点总结:" << std::endl;
  std::cout << "1. Tile = 空间划分，将大数据分成小块" << std::endl;
  std::cout << "2. Partition = 工作分配，将小块分给处理单元" << std::endl;
  std::cout << "3. logical_divide = 在 layout 层面描述分块" << std::endl;
  std::cout << "4. local_tile = 提取特定位置的 tile" << std::endl;
  std::cout << "5. 在 GEMM 中：Grid级Tile → CTA级Tile → MMA级Partition" << std::endl;
  std::cout << "\n现在你已经准备好学习 01_mma_sm100.cu 了！" << std::endl;

  return 0;
}
