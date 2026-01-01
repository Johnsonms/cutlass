/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe Tutorial 00: Tensor Basics
// 本示例演示 CuTe Tensor 的基本概念

#include <iostream>
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

// 简单的 GPU kernel 来演示 Tensor 操作
__global__ void tensor_demo_kernel(int* data, int M, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // 在 GPU 上创建 tensor
  auto layout = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  auto tensor = make_tensor(data, layout);

  if (idx < M * N) {
    int m = idx / N;
    int n = idx % N;
    // 使用 tensor 的逻辑坐标访问
    tensor(m, n) = m * 100 + n;
  }
}

int main()
{
  std::cout << "=== CuTe Tensor 基础教程 ===" << std::endl << std::endl;

  // ========================================
  // 1. Tensor = Pointer + Layout
  // ========================================
  std::cout << "1. Tensor 基本概念" << std::endl;
  std::cout << "   Tensor = 数据指针 + Layout" << std::endl;
  std::cout << "   提供统一的多维数据访问接口" << std::endl << std::endl;

  // 主机端数据
  int host_data_arr[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int* host_data = &host_data_arr[0];  // 明确获取指针

  // 创建 layout
  auto layout = make_layout(make_shape(Int<4>{}, Int<3>{}),     // 4×3 矩阵
                            make_stride(Int<3>{}, Int<1>{}));   // 行主序

  // 创建 tensor
  auto tensor = make_tensor(host_data, layout);

  std::cout << "   Tensor: " << tensor << std::endl;
  std::cout << "   访问 tensor(1,2): " << tensor(1, 2) << std::endl;
  std::cout << "   对应 host_data[" << layout(1, 2) << "]: " << host_data[layout(1, 2)] << std::endl;
  std::cout << std::endl;

  // ========================================
  // 2. 不同内存类型的 Tensor（概念）
  // ========================================
  std::cout << "2. 不同内存类型的 Tensor" << std::endl << std::endl;

  std::cout << "   CuTe 支持多种内存类型的 Tensor：" << std::endl;
  std::cout << "   - GMEM (全局内存): make_tensor(make_gmem_ptr(ptr), layout)" << std::endl;
  std::cout << "     -> 最慢，但容量最大" << std::endl;
  std::cout << "   - SMEM (共享内存): make_tensor(make_smem_ptr(ptr), layout)" << std::endl;
  std::cout << "     -> 快，线程块内共享" << std::endl;
  std::cout << "   - TMEM (张量内存): SM100 专用" << std::endl;
  std::cout << "     -> 存储MMA累加器" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 3. Tensor 访问
  // ========================================
  std::cout << "3. Tensor 访问操作" << std::endl;

  // 访问单个元素
  std::cout << "   tensor(0,0) = " << tensor(0, 0) << std::endl;
  std::cout << "   tensor(2,1) = " << tensor(2, 1) << std::endl;

  // 修改元素
  tensor(1, 1) = 99;
  std::cout << "   修改后 tensor(1,1) = " << tensor(1, 1) << std::endl;
  std::cout << "   对应 host_data_arr[" << layout(1, 1) << "] = " << host_data_arr[layout(1, 1)] << std::endl;
  std::cout << std::endl;

  // ========================================
  // 4. Tensor 命名约定 (参考 01_mma_sm100.cu)
  // ========================================
  std::cout << "4. Tensor 命名约定" << std::endl;
  std::cout << "   在 CUTLASS/CuTe 代码中，tensor 名称有特定含义:" << std::endl;
  std::cout << std::endl;
  std::cout << "   前缀含义:" << std::endl;
  std::cout << "   - g: GMEM (全局内存，如 gA, gB)" << std::endl;
  std::cout << "   - s: SMEM (共享内存，如 tCsA, tCsB)" << std::endl;
  std::cout << "   - t: TMEM (张量内存 SM100，如 tCtAcc)" << std::endl;
  std::cout << "   - r: RMEM (寄存器，如 tDrAcc, tDrC)" << std::endl;
  std::cout << std::endl;
  std::cout << "   复合命名: tXgY" << std::endl;
  std::cout << "   - t: tensor" << std::endl;
  std::cout << "   - X: 分区模式 (C=CTA的MMA, D=Copy的线程)" << std::endl;
  std::cout << "   - g/s/r: 内存类型" << std::endl;
  std::cout << "   - Y: 矩阵名 (A/B/C/D或Acc)" << std::endl;
  std::cout << std::endl;
  std::cout << "   例如:" << std::endl;
  std::cout << "   - tCgA: 用CTA的MMA模式分区全局内存的A矩阵" << std::endl;
  std::cout << "   - tCsB: 用CTA的MMA模式分区共享内存的B矩阵" << std::endl;
  std::cout << "   - tDrAcc: 用Copy线程模式分区寄存器的累加器" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 5. GPU 上的 Tensor 操作（简化版本）
  // ========================================
  std::cout << "5. GPU Kernel 中使用 Tensor（概念说明）" << std::endl;
  std::cout << "   在 GPU kernel 中，可以用同样的方式创建和使用 tensor：" << std::endl;
  std::cout << "   - 在 device 函数中：make_tensor(device_ptr, layout)" << std::endl;
  std::cout << "   - 使用逻辑坐标访问：tensor(m, n)" << std::endl;
  std::cout << "   - CuTe 自动处理内存访问" << std::endl;
  std::cout << std::endl;

  // ========================================
  // 6. Tensor 实用函数
  // ========================================
  std::cout << "6. Tensor 实用函数" << std::endl;

  std::cout << "   size(tensor): " << size(tensor) << " (总元素数)" << std::endl;
  std::cout << "   size<0>(tensor): " << size<0>(tensor) << " (第0维大小)" << std::endl;
  std::cout << "   size<1>(tensor): " << size<1>(tensor) << " (第1维大小)" << std::endl;
  std::cout << "   shape(tensor): " << shape(tensor) << std::endl;
  std::cout << "   data pointer: " << tensor.data() << std::endl;
  std::cout << std::endl;

  std::cout << "=== 教程完成 ===" << std::endl;
  std::cout << "关键要点:" << std::endl;
  std::cout << "1. Tensor = 指针 + Layout，提供多维数组抽象" << std::endl;
  std::cout << "2. 不同的指针类型对应不同的内存层次" << std::endl;
  std::cout << "3. 使用逻辑坐标访问，CuTe 自动计算物理地址" << std::endl;
  std::cout << "4. 命名约定帮助理解数据在哪个内存以及如何分区" << std::endl;

  return 0;
}
