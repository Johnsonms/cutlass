# CuTe 基础教程

本目录包含 CuTe 的基础概念教程，帮助你理解 Layout、Tensor 和 Tile。

## 教程列表

### 基础概念（适用于所有 GPU）

1. **00_layout_basics.cu** - Layout 基础
   - Layout = (Shape, Stride)
   - 行主序 vs 列主序
   - K-major Layout
   - 嵌套 Layout
   - GEMM 中的 Layout 应用

2. **00_tensor_basics.cu** - Tensor 基础
   - Tensor = Pointer + Layout
   - 不同内存类型的 Tensor (GMEM, SMEM, TMEM, RMEM)
   - Tensor 切片和访问
   - Tensor 命名约定
   - GPU Kernel 中使用 Tensor

3. **00_tile_basics.cu** - Tile 基础
   - 为什么需要 Tile
   - logical_divide 逻辑分块
   - local_tile 提取特定块
   - Step 投影
   - GEMM 中的分层 Tile 策略

### SM100 专用教程（需要 Blackwell GPU）

4. **01_mma_sm100.cu** - SM100 MMA 指令
5. **02_mma_tma_sm100.cu** - TMA 内存访问
6. **03_mma_tma_multicast_sm100.cu** - 多播 TMA
7. **04_mma_tma_2sm_sm100.cu** - 2SM MMA
8. **05_mma_tma_epi_sm100.cu** - TMA Epilogue

## 编译和运行

### 1. 配置构建（在 cutlass 根目录）

```bash
cd /app/tensorrt_llm/cutlass
mkdir -p build && cd build

# 如果只想运行基础教程（00_*），任何 GPU 架构都可以
cmake .. -DCUTLASS_NVCC_ARCHS=80  # Ampere
# 或
cmake .. -DCUTLASS_NVCC_ARCHS=90a  # Hopper
# 或
cmake .. -DCUTLASS_NVCC_ARCHS=100a  # Blackwell (SM100 教程需要)
```

### 2. 编译基础教程

```bash
# 编译所有基础教程
make cute_tutorial_00_layout_basics -j12
make cute_tutorial_00_tensor_basics -j12
make cute_tutorial_00_tile_basics -j12

# 或者一次性编译所有 cute 教程
make -j12 | grep cute_tutorial
```

### 3. 运行示例

```bash
# 在 build 目录下
./examples/cute/tutorial/blackwell/cute_tutorial_00_layout_basics
./examples/cute/tutorial/blackwell/cute_tutorial_00_tensor_basics
./examples/cute/tutorial/blackwell/cute_tutorial_00_tile_basics
```

### 4. 使用 cuda-gdb 调试

```bash
# 单步调试以深入理解
cuda-gdb ./examples/cute/tutorial/blackwell/cute_tutorial_00_layout_basics

# 在 gdb 中
(gdb) break main
(gdb) run
(gdb) next    # 单步执行
(gdb) print layout  # 查看变量
```

### 5. 编译 SM100 教程（需要 Blackwell GPU）

```bash
# 必须使用 100a 架构
cmake .. -DCUTLASS_NVCC_ARCHS=100a
make cute_tutorial_01_mma_sm100 -j12

# 运行（可指定矩阵大小）
./examples/cute/tutorial/blackwell/cute_tutorial_01_mma_sm100 [M] [N] [K]
./examples/cute/tutorial/blackwell/cute_tutorial_01_mma_sm100 512 1024 256
```

## 学习路径建议

### 第一步：理解基础概念
1. 先运行 `00_layout_basics`，理解内存布局
2. 然后运行 `00_tensor_basics`，理解 Tensor 抽象
3. 最后运行 `00_tile_basics`，理解分块策略

### 第二步：结合 01_mma_sm100.cu 学习
在理解基础概念后，回到 `01_mma_sm100.cu`：
- 查看第 535-545 行：Layout 定义
- 查看第 351-354 行：Tensor 创建
- 查看第 387-391 行：Tile 策略
- 查看第 170-173 行：local_tile 使用

### 第三步：使用调试器深入学习
```bash
cuda-gdb ./examples/cute/tutorial/blackwell/cute_tutorial_01_mma_sm100

(gdb) break 01_mma_sm100.cu:375  # tiled_mma 定义处
(gdb) run 512 1024 256
(gdb) print tiled_mma
(gdb) next
```

## 关键概念速查

| 概念 | 含义 | 示例 |
|-----|------|------|
| **Layout** | Shape + Stride，定义坐标到地址映射 | `(4,3):(3,1)` |
| **Tensor** | Pointer + Layout，多维数据抽象 | `make_tensor(ptr, layout)` |
| **Tile** | 将大问题分块，便于并行 | `local_tile(tensor, tiler, coord)` |
| **TiledMMA** | 平铺的 MMA 操作 | `make_tiled_mma(...)` |
| **partition** | 将数据分配给线程 | `cta_mma.partition_A(gA)` |

## 命名约定

### Tensor 前缀
- `g`: GMEM (全局内存) - 最慢，容量最大
- `s`: SMEM (共享内存) - 快，线程块共享
- `t`: TMEM (张量内存) - SM100 专用
- `r`: RMEM (寄存器) - 最快，容量最小

### 复合命名：`tXgY`
- `t`: tensor
- `X`: 分区模式 (C=CTA, D=Copy, ...)
- `g/s/r/t`: 内存类型
- `Y`: 矩阵名 (A/B/C/D/Acc)

示例：
- `tCgA`: 用 CTA 的 MMA 模式分区全局内存的 A
- `tDrAcc`: 用 Copy 模式分区寄存器的累加器

## 问题排查

### 编译错误
```bash
# 确保在正确的目录
cd /app/tensorrt_llm/cutlass/build

# 重新配置
rm -rf *
cmake .. -DCUTLASS_NVCC_ARCHS=80  # 或其他架构

# 查看详细编译输出
make cute_tutorial_00_layout_basics VERBOSE=1
```

### 运行时错误
```bash
# 检查 CUDA 可用性
nvidia-smi

# 检查可执行文件位置
find . -name "cute_tutorial_00*"
```

## 更多资源

- [CuTe 官方文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CUTLASS 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [原始教程](./01_mma_sm100.cu) - 完整的 SM100 GEMM 实现

## 反馈

如果发现问题或有改进建议，欢迎提 issue！
