/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe Tutorial 00: Tile Basics
// 本示例演示 CuTe Tile（分块）的基本概念

#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
  std::cout << "=== CuTe Tile 基础教程 ===" << std::endl << std::endl;

  // ========================================
  // 1. 为什么需要 Tile (分块)? - 查询实际 GPU 信息
  // ========================================
  std::cout << "1. 为什么需要 Tile (分块)?" << std::endl;
  std::cout << "   大问题 -> 分解为小块 -> 并行处理" << std::endl;
  std::cout << std::endl;

  // 查询 GPU 设备属性
  cudaDeviceProp prop;
  int device = 0;
  cudaGetDeviceProperties(&prop, device);

  std::cout << "   你的 GPU 信息: " << prop.name << std::endl;
  std::cout << "   - SM 数量: " << prop.multiProcessorCount << std::endl;
  std::cout << "   - 每个 SM 的最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "   - 每个 Block 的最大线程数: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "   - 每个 SM 的最大 Block 数: " << prop.maxBlocksPerMultiProcessor << std::endl;
  std::cout << "   - Warp 大小: " << prop.warpSize << " 个线程" << std::endl;
  std::cout << "   - 每个 SM 的最大 Warp 数: "
            << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
  std::cout << std::endl;

  std::cout << "   ⚠️  重要理解：" << std::endl;
  std::cout << "   - 每个 Block 最多 " << prop.maxThreadsPerBlock << " 个线程（单个 Block 的限制）" << std::endl;
  std::cout << "   - 每个 SM 最多 " << prop.maxThreadsPerMultiProcessor << " 个线程（SM 的总容量）" << std::endl;
  std::cout << "   - 每个 SM 最多 " << prop.maxBlocksPerMultiProcessor << " 个 Block（并发 Block 限制）" << std::endl;
  std::cout << std::endl;

  std::cout << "   实际例子（假设只考虑线程数限制）：" << std::endl;
  std::cout << "   - 如果每个 Block 有 256 线程 → 一个 SM 可运行 "
            << std::min(prop.maxThreadsPerMultiProcessor / 256, prop.maxBlocksPerMultiProcessor)
            << " 个 Block" << std::endl;
  std::cout << "   - 如果每个 Block 有 512 线程 → 一个 SM 可运行 "
            << std::min(prop.maxThreadsPerMultiProcessor / 512, prop.maxBlocksPerMultiProcessor)
            << " 个 Block" << std::endl;
  std::cout << "   - 如果每个 Block 有 1024 线程 → 一个 SM 可运行 "
            << std::min(prop.maxThreadsPerMultiProcessor / 1024, prop.maxBlocksPerMultiProcessor)
            << " 个 Block" << std::endl;
  std::cout << std::endl;

  std::cout << "   GPU 层次结构:" << std::endl;
  std::cout << "   Grid (整个问题)" << std::endl;
  std::cout << "     ↓ 包含多个 Block (CTA)" << std::endl;
  std::cout << "   Block/CTA - 运行在一个 SM 上" << std::endl;
  std::cout << "     ↓ 包含多个 Warp" << std::endl;
  std::cout << "   Warp - " << prop.warpSize << " 个线程同步执行 (SIMT)" << std::endl;
  std::cout << "     ↓ 包含 " << prop.warpSize << " 个 Thread" << std::endl;
  std::cout << "   Thread - 最小执行单元" << std::endl;
  std::cout << std::endl;

  std::cout << "   Tile 策略让我们在不同层次分配工作：" << std::endl;
  std::cout << "   - Grid 层: 将问题分成多个 Tile，每个 Block 处理一个 Tile" << std::endl;
  std::cout << "   - Block 层: 将 Tile 分给多个 Warp" << std::endl;
  std::cout << "   - Warp 层: 将数据分给 " << prop.warpSize << " 个 Thread 并行处理" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 2. 基本 Tile 操作: logical_divide
  // ========================================
  std::cout << "2. logical_divide: 逻辑分块" << std::endl;

  // 创建一个 8×12 的 layout
  auto big_layout = make_layout(make_shape(Int<8>{}, Int<12>{}),
                                make_stride(Int<12>{}, Int<1>{}));
  std::cout << "   原始 Layout: " << big_layout << std::endl;

  // 定义 tile 大小: 2×3
  auto tile_shape = make_shape(Int<2>{}, Int<3>{});
  std::cout << "   Tile 大小: " << tile_shape << std::endl;

  // 执行逻辑分块
  auto tiled = logical_divide(big_layout, tile_shape);
  std::cout << "   分块后: " << tiled << std::endl;
  std::cout << "   解释: ((2,3),(4,4)):((12,1),(2,3))" << std::endl;
  std::cout << "   - 第一组 (2,3): 每个 tile 内部是 2×3" << std::endl;
  std::cout << "   - 第二组 (4,4): 有 4×4 = 16 个 tiles" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 3. 可视化 Tile
  // ========================================
  std::cout << "3. Tile 可视化" << std::endl;
  std::cout << "   原始 8×12 矩阵分成 (2×3) 的块:" << std::endl;
  std::cout << std::endl;
  std::cout << "   ┌─────┬─────┬─────┬─────┐" << std::endl;
  std::cout << "   │ T00 │ T01 │ T02 │ T03 │  <- 每个 T 是 2×3" << std::endl;
  std::cout << "   ├─────┼─────┼─────┼─────┤" << std::endl;
  std::cout << "   │ T10 │ T11 │ T12 │ T13 │" << std::endl;
  std::cout << "   ├─────┼─────┼─────┼─────┤" << std::endl;
  std::cout << "   │ T20 │ T21 │ T22 │ T23 │" << std::endl;
  std::cout << "   ├─────┼─────┼─────┼─────┤" << std::endl;
  std::cout << "   │ T30 │ T31 │ T32 │ T33 │" << std::endl;
  std::cout << "   └─────┴─────┴─────┴─────┘" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 4. local_tile: 提取特定的 Tile
  // ========================================
  std::cout << "4. local_tile: 提取特定 Tile" << std::endl;

  // 创建一个简单的 tensor
  int data_arr[96];  // 8×12 = 96
  for (int i = 0; i < 96; ++i) data_arr[i] = i;
  int* data = &data_arr[0];

  auto tensor = make_tensor(data, big_layout);
  std::cout << "   原始 Tensor (8×12): " << tensor << std::endl;

  // 演示概念：Tile是将大矩阵分成小块
  std::cout << "   如果用 2×3 的块来分，会有 " << (8/2) << "×" << (12/3) << " = " << (8/2)*(12/3) << " 个块" << std::endl;

  std::cout << "   例如 Tile(1,2) 位于第1行第2列的位置" << std::endl;
  std::cout << "   起始坐标: (" << 1*2 << ", " << 2*3 << ")" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 5. Step 投影: 选择性 Tile
  // ========================================
  std::cout << "5. Step 投影: 只在部分维度分块" << std::endl;

  // 创建 3D tensor: 4×6×8
  auto layout_3d = make_layout(make_shape(Int<4>{}, Int<6>{}, Int<8>{}));
  std::cout << "   3D Layout: " << layout_3d << std::endl;

  auto tiler_3d = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
  auto coord_3d = make_coord(0, 1, 0);

  // 只在前两个维度 tile，第三维度保持完整
  // Step<_1, _1, X>: 选择 dim0 和 dim1，忽略 dim2
  std::cout << "   使用 Step<_1, _1, X> (只 tile 前两维):" << std::endl;
  std::cout << "   _1 表示选择这个维度进行 tile" << std::endl;
  std::cout << "   X 表示保留这个维度完整" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 6. GEMM 中的分层 Tile 策略
  // ========================================
  std::cout << "6. GEMM 中的分层 Tile 策略" << std::endl;
  std::cout << "   (参考 01_mma_sm100.cu 第 387-391 行)" << std::endl << std::endl;

  // 模拟 GEMM 问题规模
  int Gemm_M = 512, Gemm_N = 1024, Gemm_K = 256;
  std::cout << "   GEMM 问题: " << Gemm_M << "×" << Gemm_N << "×" << Gemm_K << std::endl;
  std::cout << std::endl;

  // MMA 指令级别 (硬件 tcgen05.mma)
  int Mma_M = 128, Mma_N = 256, Mma_K = 16;
  std::cout << "   第3层 - MMA 指令:" << std::endl;
  std::cout << "   - 尺寸: " << Mma_M << "×" << Mma_N << "×" << Mma_K << std::endl;
  std::cout << "   - 这是单个硬件指令处理的大小" << std::endl;
  std::cout << "   - 在 Blackwell SM100 上是 tcgen05.mma" << std::endl;
  std::cout << std::endl;

  // MMA Tile 级别 (CTA 处理)
  int MmaTile_M = 128, MmaTile_N = 256, MmaTile_K = 64;  // K 重复 4 次
  std::cout << "   第2层 - MMA Tile (CTA 级别):" << std::endl;
  std::cout << "   - 尺寸: " << MmaTile_M << "×" << MmaTile_N << "×" << MmaTile_K << std::endl;
  std::cout << "   - K 维度: " << Mma_K << " × 4 = " << MmaTile_K << " (重复4次MMA)" << std::endl;
  std::cout << "   - 一个 CTA (线程块) 处理一个 MMA Tile" << std::endl;
  std::cout << std::endl;

  // Grid 级别 (整个问题)
  int Tiles_M = Gemm_M / MmaTile_M;  // 512/128 = 4
  int Tiles_N = Gemm_N / MmaTile_N;  // 1024/256 = 4
  int Tiles_K = Gemm_K / MmaTile_K;  // 256/64 = 4
  std::cout << "   第1层 - Grid 级别:" << std::endl;
  std::cout << "   - Tiles: " << Tiles_M << "×" << Tiles_N << "×" << Tiles_K << std::endl;
  std::cout << "   - 总共需要 " << Tiles_M * Tiles_N << " 个 CTA" << std::endl;
  std::cout << "   - 每个 CTA 执行 " << Tiles_K << " 次 MMA Tile" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 7. 完整的 Tile 层次结构
  // ========================================
  std::cout << "7. 完整的 Tile 层次结构总结" << std::endl;
  std::cout << std::endl;
  std::cout << "   整个 GEMM (" << Gemm_M << "×" << Gemm_N << "×" << Gemm_K << ")" << std::endl;
  std::cout << "   ↓ 用 mma_tiler 分成 " << Tiles_M*Tiles_N << " 个块" << std::endl;
  std::cout << "   MMA Tile (" << MmaTile_M << "×" << MmaTile_N << "×" << MmaTile_K << ") ← 一个 CTA 处理" << std::endl;
  std::cout << "   ↓ 用 tiled_mma 分成 " << MmaTile_K/Mma_K << " 次 MMA" << std::endl;
  std::cout << "   MMA 指令 (" << Mma_M << "×" << Mma_N << "×" << Mma_K << ") ← 单个硬件指令" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 8. partition 和 Tile 的关系
  // ========================================
  std::cout << "8. partition: 将 Tile 分配给线程" << std::endl;
  std::cout << "   Tile 定义数据块的大小和位置" << std::endl;
  std::cout << "   partition 定义如何将这个块分配给不同线程" << std::endl;
  std::cout << std::endl;
  std::cout << "   在 01_mma_sm100.cu 中:" << std::endl;
  std::cout << "   - gA 是完整的 MMA Tile" << std::endl;
  std::cout << "   - tCgA = cta_mma.partition_A(gA)" << std::endl;
  std::cout << "   - partition_A 将 gA 按 MMA 需求分配" << std::endl;
  std::cout << "   - 不同的 CTA 获得不同的 partition" << std::endl;
  std::cout << std::endl;

  std::cout << "=== 教程完成 ===" << std::endl;
  std::cout << "关键要点:" << std::endl;
  std::cout << "1. Tile = 将大问题分解为小块，便于并行" << std::endl;
  std::cout << "2. 分层 Tile: Grid -> CTA -> Warp -> Thread" << std::endl;
  std::cout << "3. local_tile 提取特定的块" << std::endl;
  std::cout << "4. Step 投影控制在哪些维度分块" << std::endl;
  std::cout << "5. GEMM 使用 3 层 Tile 结构优化性能" << std::endl;

  return 0;
}
