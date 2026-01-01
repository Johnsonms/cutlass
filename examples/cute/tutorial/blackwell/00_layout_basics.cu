/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe Tutorial 00: Layout Basics
// 本示例演示 CuTe Layout 的基本概念

#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
  std::cout << "=== CuTe Layout 基础教程 ===" << std::endl << std::endl;

  auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                      Stride<_1,Stride<_6,_2>>>{};
  std::cout << "***********"<<std::endl;
  print_latex(layout);
  std::cout << "***********"<<std::endl;
  auto result = coalesce(layout);    // _12:_1
  print_latex(result);

  // ========================================
  // 1. 基本 Layout：Shape + Stride
  // ========================================
  std::cout << "1. 基本 Layout 概念" << std::endl;
  std::cout << "   Layout = (Shape, Stride)" << std::endl;
  std::cout << "   Shape 定义逻辑维度大小，Stride 定义内存跳跃步长" << std::endl << std::endl;

  // 创建一个 4x3 的行主序 layout
  auto row_major = make_layout(make_shape(Int<4>{}, Int<3>{}),     // Shape: (4, 3)
                               make_stride(Int<3>{}, Int<1>{}));   // Stride: (3, 1)
  std::cout << "   行主序 4×3: " << row_major << std::endl;
  std::cout << "   解释: (4,3):(3,1) 表示 4 行 3 列，行间隔 3，列间隔 1" << std::endl;

  // 计算坐标到线性地址的映射
  std::cout << "   坐标(0,0) -> 地址: " << row_major(0, 0) << std::endl;  // 0*3 + 0*1 = 0
  std::cout << "   坐标(1,0) -> 地址: " << row_major(1, 0) << std::endl;  // 1*3 + 0*1 = 3
  std::cout << "   坐标(1,2) -> 地址: " << row_major(1, 2) << std::endl;  // 1*3 + 2*1 = 5
  std::cout << std::endl;

  // 创建列主序 layout
  auto col_major = make_layout(make_shape(Int<4>{}, Int<3>{}),     // Shape: (4, 3)
                               make_stride(Int<1>{}, Int<4>{}));   // Stride: (1, 4)
  std::cout << "   列主序 4×3: " << col_major << std::endl;
  std::cout << "   解释: (4,3):(1,4) 表示 4 行 3 列，行间隔 1，列间隔 4" << std::endl;
  std::cout << "   坐标(1,2) -> 地址: " << col_major(1, 2) << std::endl;  // 1*1 + 2*4 = 9
  std::cout << std::endl;

  // ========================================
  // 2. 内存布局可视化
  // ========================================
  std::cout << "2. 内存布局可视化" << std::endl;
  std::cout << "   行主序内存排列: [行0的元素...][行1的元素...][行2的元素...]" << std::endl;
  std::cout << "   对于 4×3 矩阵:" << std::endl;
  std::cout << "   索引: ";
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::cout << "(" << i << "," << j << "):" << row_major(i, j) << " ";
    }
  }
  std::cout << std::endl << std::endl;

  // ========================================
  // 3. K-major Layout (常用于 GEMM)
  // ========================================
  std::cout << "3. K-major Layout (GEMM 中常见)" << std::endl;

  int M = 128, K = 64;
  // K-major 意味着 K 维度是连续的 (stride=1)
  auto A_layout = make_layout(make_shape(Int<128>{}, Int<64>{}),
                              make_stride(Int<64>{}, Int<1>{}));
  std::cout << "   A 矩阵 " << M << "×" << K << " K-major: " << A_layout << std::endl;
  std::cout << "   这意味着每行的 K 个元素在内存中是连续的" << std::endl;
  std::cout << "   坐标(0,0) -> " << A_layout(0, 0) << std::endl;
  std::cout << "   坐标(0,1) -> " << A_layout(0, 1) << " (连续)" << std::endl;
  std::cout << "   坐标(1,0) -> " << A_layout(1, 0) << " (跳过 64 个元素)" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 4. 嵌套 Layout
  // ========================================
  std::cout << "4. 嵌套 Layout (分层结构)" << std::endl;

  // 创建一个 (2,2) 的块，每块 3 个元素
  auto nested = make_layout(make_shape(make_shape(Int<2>{}, Int<2>{}), Int<3>{}),
                            make_stride(make_stride(Int<1>{}, Int<2>{}), Int<4>{}));
  std::cout << "   嵌套 Layout: " << nested << std::endl;
  std::cout << "   ((2,2),3):((1,2),4) 表示:" << std::endl;
  std::cout << "   - 外层: 3 个组，每组间隔 4" << std::endl;
  std::cout << "   - 内层: 2×2 块，第一维间隔 1，第二维间隔 2" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 5. Layout 的实用函数
  // ========================================
  std::cout << "5. Layout 实用函数" << std::endl;

  std::cout << "   size(layout): " << size(row_major) << " (总元素数)" << std::endl;
  std::cout << "   size<0>(layout): " << size<0>(row_major) << " (第0维大小)" << std::endl;
  std::cout << "   size<1>(layout): " << size<1>(row_major) << " (第1维大小)" << std::endl;
  std::cout << "   rank(layout): " << rank_v<decltype(row_major)> << " (维度数)" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 6. 实际应用：理解 GEMM 中的 Layout
  // ========================================
  std::cout << "6. GEMM 中的 Layout (参考 01_mma_sm100.cu)" << std::endl;

  int Gemm_M = 512, Gemm_N = 1024, Gemm_K = 256;

  // A: M×K, K-major (行主序，每行 K 个连续元素)
  auto layout_A = make_layout(make_shape(Gemm_M, Gemm_K),
                              make_stride(Gemm_K, Int<1>{}));
  std::cout << "   A 矩阵 (" << Gemm_M << "×" << Gemm_K << "): " << layout_A << std::endl;
  std::cout << "   -> K-major, 适合按行读取" << std::endl;

  // B: N×K, K-major (列主序，每列 K 个连续元素)
  auto layout_B = make_layout(make_shape(Gemm_N, Gemm_K),
                              make_stride(Gemm_K, Int<1>{}));
  std::cout << "   B 矩阵 (" << Gemm_N << "×" << Gemm_K << "): " << layout_B << std::endl;
  std::cout << "   -> K-major, 适合按列读取" << std::endl;

  // C/D: M×N, N-major (行主序)
  auto layout_C = make_layout(make_shape(Gemm_M, Gemm_N),
                              make_stride(Gemm_N, Int<1>{}));
  std::cout << "   C/D 矩阵 (" << Gemm_M << "×" << Gemm_N << "): " << layout_C << std::endl;
  std::cout << "   -> N-major, 行主序存储" << std::endl;
  std::cout << std::endl;

  std::cout << "=== 教程完成 ===" << std::endl;
  std::cout << "提示: 使用 cuda-gdb 可以单步调试并查看 layout 的值" << std::endl;

  return 0;
}
