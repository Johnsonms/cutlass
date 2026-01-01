# CuTe Atom 和 TiledMMA 学习指南

## 📚 概述

本教程通过传统 CUDA 和 CuTe 的对比示例，帮助你深入理解 CuTe 的核心概念：
- **Atom**：最基本的计算单元
- **TiledMMA**：Atom 在多线程中的组织方式
- **ThrMMA**：单个线程的视角

## 📂 文件说明

### 1. `atom_concept_simple.cu` - 概念讲解（推荐先看）
**难度：⭐ 入门级**

专门用于理解概念的教学示例，包含：
- Atom 的定义和例子（FMA、Tensor Core、Copy）
- TiledMMA 的创建和工作原理
- ThrMMA 的使用方法
- 4×4 矩阵乘法的完整小例子

**特点：**
- 代码简单，注释详细
- 有大量 `printf` 输出，清楚展示每个步骤
- 无需 GPU 也能理解概念（虽然运行需要 GPU）

### 2. `cuda_vs_cute_comparison.cu` - 完整对比（深入理解）
**难度：⭐⭐⭐ 中级**

并排对比传统 CUDA 和 CuTe 的实现，包含：
- 传统 CUDA GEMM 实现（手动索引计算）
- CuTe GEMM 实现（使用 Atom 和 TiledMMA）
- 详细的概念注释和对比说明
- 结果验证和性能对比

**特点：**
- 真实的 GEMM 实现（64×64×64 矩阵）
- 清楚展示传统方法的痛点
- 深入理解 CuTe 的优势

## 🚀 快速开始

### 步骤 1：编译示例

```bash
# 进入教程目录
cd /app/tensorrt_llm/cutlass/examples/cute/tutorial

# 编译概念讲解示例
nvcc -std=c++17 -arch=sm_70 \
     -I../../../include \
     atom_concept_simple.cu \
     -o atom_concept_simple

# 编译对比示例
nvcc -std=c++17 -arch=sm_70 \
     -I../../../include \
     cuda_vs_cute_comparison.cu \
     -o cuda_vs_cute_comparison
```

**注意：** 根据你的 GPU 架构调整 `-arch` 参数：
- SM 7.0 (Volta): `-arch=sm_70`
- SM 8.0 (Ampere): `-arch=sm_80`
- SM 9.0 (Hopper): `-arch=sm_90`

### 步骤 2：运行示例

```bash
# 运行概念讲解
./atom_concept_simple

# 运行对比示例
./cuda_vs_cute_comparison
```

## 📖 学习路径

### 第一阶段：理解基础概念（1-2 小时）

1. **运行并阅读输出：`atom_concept_simple`**
   ```bash
   ./atom_concept_simple | less
   ```

   重点关注：
   - Atom 的三种例子（FMA、Tensor Core、Copy）
   - TiledMMA 的创建过程
   - 线程布局的含义
   - ThrMMA 的作用

2. **阅读源代码：`atom_concept_simple.cu`**

   重点关注：
   - `UniversalFMA<float>` 的定义（Atom）
   - `make_tiled_mma()` 的调用（TiledMMA）
   - `tiled_mma.get_slice()` 的使用（ThrMMA）
   - `partition_A/B/C()` 的作用（数据分割）

3. **实验和修改**

   尝试修改代码：
   ```cpp
   // 原来：4×4 线程布局
   Layout<Shape<_4, _4, _1>>{}

   // 修改为：8×2 线程布局
   Layout<Shape<_8, _2, _1>>{}
   ```
   观察输出的变化。

### 第二阶段：对比传统 CUDA（2-3 小时）

1. **运行对比示例**
   ```bash
   ./cuda_vs_cute_comparison
   ```

   观察：
   - 两个 kernel 的结果是否一致
   - 性能差异（如果有）

2. **并排阅读两种实现**

   打开 `cuda_vs_cute_comparison.cu`，对比：

   | 步骤 | 传统 CUDA | CuTe |
   |------|-----------|------|
   | 索引计算 | `tid / N`, `tid % N` | `Layout` |
   | 内存访问 | `A[row*K + col]` | `Tensor(row, col)` |
   | 线程分配 | 手动循环和条件 | `partition_A/B/C` |
   | 计算 | 显式 `+=` 循环 | `gemm()` |

3. **添加调试输出**

   在两个 kernel 中添加 `printf`，对比中间结果：
   ```cpp
   // 传统 CUDA
   if (tid == 0) {
       printf("Thread 0 processes row=%d, col=%d\n", thread_row, thread_col);
   }

   // CuTe
   if (threadIdx.x == 0) {
       printf("Thread 0 tensor shape: ");
       print(tCsA.shape());
       printf("\n");
   }
   ```

### 第三阶段：深入理解（3-5 小时）

1. **研究 TiledMMA 的内部结构**

   在 `atom_concept_simple.cu` 中添加：
   ```cpp
   auto tiled_mma = make_tiled_mma(...);

   if (threadIdx.x == 0) {
       printf("TiledMMA info:\n");
       print("  Shape: "); print(tiled_mma.shape()); printf("\n");
       print("  Size: %d threads\n", size(tiled_mma));
   }
   ```

2. **实验不同的 Atom**

   尝试使用不同类型的 Atom：
   ```cpp
   // 标量 FMA
   using Atom1 = UniversalFMA<float>;

   // 双精度 FMA
   using Atom2 = UniversalFMA<double>;

   // Tensor Core (需要 SM80+)
   using Atom3 = SM80_16x8x8_F16F16F16F16_TN;
   ```

3. **阅读现有的复杂示例**

   现在你可以理解更复杂的示例了：
   - `sgemm_1.cu` - 基础 GEMM
   - `sgemm_2.cu` - 使用 TiledCopy 和 TiledMMA
   - `sgemm_sm80.cu` - 高级流水线优化

## 🔍 调试技巧

### 1. 使用 CUTE_STATIC_ASSERT 检查形状

CuTe 提供编译时检查：
```cpp
CUTE_STATIC_ASSERT_V(size<0>(tCsA) == Int<16>{});  // 检查 M 维度
```

如果形状不匹配，编译器会报错，帮助你发现问题。

### 2. 使用 print() 函数查看结构

CuTe 提供了 `print()` 函数来打印 Layout 和 Tensor 信息：
```cpp
if (threadIdx.x == 0) {
    printf("Tensor A shape: "); print(tA.shape()); printf("\n");
    printf("Tensor A stride: "); print(tA.stride()); printf("\n");
    printf("Tensor A layout: "); print(tA.layout()); printf("\n");
}
```

### 3. 使用 cuda-gdb 单步调试

```bash
# 编译时添加调试符号
nvcc -g -G -std=c++17 -arch=sm_70 \
     -I../../../include \
     atom_concept_simple.cu \
     -o atom_concept_simple_debug

# 使用 cuda-gdb 调试
cuda-gdb ./atom_concept_simple_debug

# 在 cuda-gdb 中
(cuda-gdb) break tiny_gemm_example  # 设置断点
(cuda-gdb) run                       # 运行
(cuda-gdb) cuda thread (0,0,0)       # 切换到线程 0
(cuda-gdb) print tCsA                # 查看变量
```

### 4. 使用 Nsight Compute 分析性能

```bash
# 分析对比示例的两个 kernel
ncu --set full \
    --kernel-name gemm_traditional_cuda \
    --kernel-name gemm_cute \
    ./cuda_vs_cute_comparison

# 对比内存访问模式
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    --kernel-name-base regex \
    --kernel-name "gemm.*" \
    ./cuda_vs_cute_comparison
```

## 💡 核心概念速查表

### Atom（原子操作）

| 类型 | 用途 | 形状 | 硬件 |
|------|------|------|------|
| `UniversalFMA<float>` | 标量乘加 | (1,1,1) | CUDA Core |
| `SM80_16x8x16_F16` | Tensor Core MMA | (16,8,16) | Tensor Core |
| `UniversalCopy<uint128_t>` | 向量化复制 | 16 bytes | Mem Unit |

### TiledMMA（平铺的 MMA）

```cpp
auto tiled_mma = make_tiled_mma(
    Atom{},                      // 计算单元
    ThreadLayout{}               // 线程布局
);
```

**作用：**
- 定义使用多少线程
- 定义线程如何排列
- 定义每个线程的工作

### ThrMMA（线程的 MMA）

```cpp
auto thr_mma = tiled_mma.get_slice(threadIdx.x);
Tensor tA = thr_mma.partition_A(src_A);  // 分割 A
Tensor tB = thr_mma.partition_B(src_B);  // 分割 B
Tensor tC = thr_mma.partition_C(src_C);  // 分割 C
```

**作用：**
- 提取单个线程的信息
- 自动分割数据给这个线程
- 提供数据访问接口

## 🎯 常见问题

### Q1: Atom 和指令的关系是什么？

**A:** Atom 是对硬件指令的抽象封装：
- **底层**：PTX 指令（如 `fma.f32`、`mma.sync.m16n8k16`）
- **Atom**：C++ 类型，描述指令的行为和形状
- **好处**：统一接口，容易替换和优化

### Q2: 为什么需要 TiledMMA？直接用 Atom 不行吗？

**A:** 因为 GPU 编程是多线程的：
- **Atom**：描述单个操作（如一个 FMA）
- **TiledMMA**：描述多个线程如何协作使用 Atom
- **类比**：Atom 是"工具"，TiledMMA 是"工作流程"

### Q3: ThrMMA 和 TiledMMA 的区别？

**A:** 视角不同：
- **TiledMMA**：整体视角，描述所有线程
- **ThrMMA**：单个线程视角，描述"我"的工作
- **关系**：`ThrMMA = TiledMMA.get_slice(thread_id)`

### Q4: partition 是什么意思？

**A:** "分割"或"分配"：
- 将整体数据按照 ThrMMA 的规则分配给当前线程
- 自动计算这个线程需要读取/写入的数据位置
- 返回一个 Tensor，包含这个线程的数据子集

### Q5: 如何选择线程布局？

**A:** 考虑几个因素：
1. **硬件限制**：warp 大小（32）、block 大小（≤1024）
2. **Atom 形状**：如果 Atom 是 16×8，布局应该是其倍数
3. **内存访问**：合并访问需要连续的线程访问连续地址
4. **寄存器使用**：每个线程的数据量 = 总数据量 / 线程数

## 📚 进阶资源

### CuTe 官方文档

位置：`/app/tensorrt_llm/cutlass/media/docs/cpp/cute/`

推荐阅读顺序：
1. `00_quickstart.md` - 快速入门
2. `01_layout.md` - Layout 详解
3. `03_tensor.md` - Tensor 详解
4. `0t_mma_atom.md` - **MMA Atom 详解（重点）**
5. `0x_gemm_tutorial.md` - **GEMM 教程（重点）**

### 现有示例代码

| 文件 | 难度 | 重点内容 |
|------|------|----------|
| `sgemm_1.cu` | ⭐⭐ | 基础 GEMM，传统线程布局 |
| `sgemm_2.cu` | ⭐⭐⭐ | **TiledCopy + TiledMMA** |
| `sgemm_sm80.cu` | ⭐⭐⭐⭐ | 双层流水线，高级优化 |
| `tiled_copy.cu` | ⭐⭐ | TiledCopy 详解 |
| `blackwell/00_*.cu` | ⭐ | 基础概念教程（中文注释）|

### 推荐学习路径

```
1. atom_concept_simple.cu           (本教程)
   ↓
2. cuda_vs_cute_comparison.cu       (本教程)
   ↓
3. blackwell/00_layout_basics.cu    (Layout 基础)
   ↓
4. blackwell/00_tensor_basics.cu    (Tensor 基础)
   ↓
5. sgemm_1.cu                       (基础 GEMM)
   ↓
6. sgemm_2.cu                       (TiledMMA 应用)
   ↓
7. 0x_gemm_tutorial.md              (理论深化)
   ↓
8. sgemm_sm80.cu                    (高级优化)
```

## 🛠️ 实践项目建议

完成教程后，尝试这些项目来巩固理解：

### 项目 1：实现简单的 GEMV（矩阵-向量乘法）
**难度：⭐⭐**
- 使用 UniversalFMA Atom
- 体会 1D vs 2D 的线程布局
- 对比传统实现

### 项目 2：添加不同的数据类型支持
**难度：⭐⭐**
- 修改示例支持 double、half
- 理解 Atom 的类型参数
- 测试性能差异

### 项目 3：优化内存访问
**难度：⭐⭐⭐**
- 添加 TiledCopy 优化数据加载
- 使用 shared memory 降低延迟
- 使用 Nsight Compute 验证优化效果

### 项目 4：实现简单的 Convolution
**难度：⭐⭐⭐⭐**
- 使用 CuTe 实现 2D 卷积
- 理解 im2col 的 CuTe 表示
- 复用 GEMM 的 TiledMMA

## 📧 反馈和问题

如果你在学习过程中遇到问题或有改进建议，可以：
1. 检查 CuTe 官方文档
2. 查看 CUTLASS GitHub Issues
3. 参考现有的示例代码
4. 使用调试工具逐步分析

祝学习顺利！🎉

---

**最后更新：** 2025-12-21
**作者：** Claude (Anthropic)
**许可：** BSD-3-Clause (与 CUTLASS 相同)
