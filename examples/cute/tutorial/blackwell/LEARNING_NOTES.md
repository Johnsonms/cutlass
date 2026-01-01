# CuTe 学习笔记 - 2025-12-21

## 学习进度

✅ 完成的教程：
- [x] Layout Basics - 理解 (Shape, Stride)
- [x] Tensor Basics - 理解 Pointer + Layout
- [x] Tile Basics - 理解数据分块和 GPU 层次
- [x] Partition Basics - 理解工作分配
- [x] MMA Thread Concepts - 理解 TiledMMA 和 ThrMMA

⏳ 下一步：
- [ ] 调试 01_mma_sm100.cu
- [ ] 理解完整的 GEMM 实现

## 关键概念速查

### 基础概念

**Layout = (Shape, Stride)**
```cpp
auto layout = make_layout(make_shape(4, 3), make_stride(3, 1));
// (4,3):(3,1) - 4×3 矩阵，行主序
```

**Tensor = Pointer + Layout**
```cpp
auto tensor = make_tensor(data_ptr, layout);
// 提供多维访问: tensor(i, j)
```

**Tile = 数据分块**
```cpp
auto tile = local_tile(tensor, tile_shape, coord);
// 提取特定位置的数据块
```

**Partition = 工作分配**
```cpp
auto tCgA = cta_mma.partition_A(gA);
// 将数据按 MMA 需求分配
```

### 线程层次

```
Grid (整个问题)
  ↓
Cluster (Blackwell 新增)
  ↓
CTA / Block (线程块)
  ↓
Warp (32 个线程)
  ↓
Thread (单个线程)
```

### TiledMMA 和 ThrMMA

**TiledMMA**: Cluster 级别的 MMA 描述
```cpp
TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<...>{128, 256, ...});
```

**ThrMMA**: 单个 CTA 的 MMA 操作符
```cpp
ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
// mma_v 是 CTA 在 Cluster 中的编号
```

### VMNK 坐标系统

```cpp
auto mma_coord_vmnk = make_coord(
    blockIdx.x % cluster_size_v,  // V: Peer CTA 编号
    blockIdx.x / cluster_size_v,  // M: M 维度位置
    blockIdx.y,                    // N: N 维度位置
    _                              // K: 所有 K
);
```

## 重要的 Q&A

### Q1: CTA 和 Thread Block 是一个东西吗？

**A:** 是的！完全相同。
- CTA = Cooperative Thread Array (硬件术语)
- Thread Block = CUDA 编程术语
- 它们指的是同一个概念

### Q2: local_tile 和 logical_divide 的关系？

**A:** 不同用途：
- `logical_divide`: 描述分块结构（看的角度）
- `local_tile`: 提取特定数据块（拿数据）
- 通常直接用 `local_tile`，不需要先 `logical_divide`

### Q3: partition_A 做了什么？

**A:** 将 CTA 的数据按 MMA 指令需求重新组织：
```
输入: gA (128, 64, 4)  - 逻辑视图
输出: tCgA ((_128,_16), _1, _4, 4)  - MMA 格式
```

### Q4: get_slice(mma_v) 的作用？

**A:** 从 TiledMMA 提取当前 CTA 的 MMA 操作符：
- `mma_v` 是 CTA 在 Cluster 中的编号（V 维度）
- 返回的 `ThrMMA` 用于后续的 partition 操作

### Q5: 为什么需要 V 维度？

**A:** Blackwell (SM100) 的 MMA 是 Cluster-scoped 的：
- 多个 CTA 可以协同工作
- V 维度标识 CTA 在 Cluster 中的位置
- 不同 V 的 CTA 可能处理相同位置但扮演不同角色

## 数据流总结

### 完整的 GEMM 数据流

```
1. 问题空间
   mA (完整的 A 矩阵) [512, 256]
       ↓ local_tile(mma_tiler, coord)

2. CTA 数据块
   gA (当前 CTA 的 A 块) [128, 64, 4]
       ↓ cta_mma.partition_A(gA)

3. 重组数据
   tCgA (MMA 格式) [(_128,_16), _1, _4, 4]
       ↓ copy to SMEM

4. 共享内存
   tCsA (SMEM 中的 A)
       ↓ make_fragment_A

5. 片段/描述符
   tCrA (SMEM 描述符)
       ↓ gemm(tiled_mma, tCrA, tCrB, tCtAcc)

6. 累加器
   tCtAcc (TMEM 累加器) - 最终结果
```

## 命名约定

```
tCgA 的含义:
  t: tensor
  C: CTA 的分区模式
  g: global memory
  A: A 矩阵

类似的:
  tCsB: tensor, CTA partition, shared memory, B matrix
  tDrAcc: tensor, Copy thread partition, register, accumulator
  tCtAcc: tensor, CTA partition, tensor memory, accumulator
```

## GPU 硬件信息（NVIDIA B200）

```
- SM 数量: 148
- 每个 SM 的最大线程数: 2048
- 每个 Block 的最大线程数: 1024
- 每个 SM 的最大 Block 数: 32
- Warp 大小: 32 个线程

实例：
- 256 threads/block → 一个 SM 可运行 8 blocks
- 512 threads/block → 一个 SM 可运行 4 blocks
- 1024 threads/block → 一个 SM 可运行 2 blocks
```

## 调试断点位置

### 01_mma_sm100.cu 关键行

```
第 157-160 行: mma_coord_vmnk 计算
  观察: blockIdx, cluster_layout

第 170 行: gA = local_tile(...)
  观察: gA 的 shape

第 203 行: mma_v = get<0>(mma_coord_vmnk)
  观察: mma_v 的值

第 204 行: cta_mma = tiled_mma.get_slice(mma_v)
  观察: cta_mma 的类型和内容

第 205 行: tCgA = cta_mma.partition_A(gA)
  观察: tCgA 的 shape 变化

第 266 行: for (int k_tile = 0; k_tile < ...)
  外层 K 循环

第 289 行: gemm(tiled_mma, tCrA, tCrB, tCtAcc)
  实际 MMA 执行
```

## 快速命令

```bash
# 编译所有教程
./b.sh

# 编译单个
./b.sh layout    # Layout 教程
./b.sh tensor    # Tensor 教程
./b.sh tile      # Tile 教程
./b.sh partition # Partition 教程
./b.sh thread    # MMA Thread 概念

# 运行
./layout_basics
./tensor_basics
./tile_basics
./partition_basics
./mma_thread_concepts

# VSCode 调试
# 按 F5 → 选择配置 → 开始调试
```

## 下次继续的步骤

1. **复习**: 运行 `./mma_thread_concepts`
2. **阅读**: 查看 `DEBUG_GUIDE.md`
3. **调试**: 在 VSCode 中调试 `01_mma_sm100.cu`
   - 断点: 第 204, 205, 266, 289 行
4. **理解**: 完整的 GEMM 实现流程
5. **实验**: 修改参数，观察行为变化

## 遇到的问题和解决方法

### 问题 1: 编译错误 - arch mismatch
**解决**: 更新 `b.sh` 使用 `-gencode arch=compute_100,code=sm_100`

### 问题 2: 数组指针问题
**解决**: 显式获取指针 `int* ptr = &array[0];`

### 问题 3: Tile 坐标理解错误
**解决**: tile(1,2) 从位置 (row*tile_h, col*tile_w) 开始

## 参考资料

- CuTe 文档: `/app/tensorrt_llm/cutlass/media/docs/cute/`
- 示例代码: `/app/tensorrt_llm/cutlass/examples/cute/`
- 调试指南: `./DEBUG_GUIDE.md`
- 本笔记: `./LEARNING_NOTES.md`

---

**最后更新**: 2025-12-21
**下次计划**: 调试 01_mma_sm100.cu，理解完整 GEMM 实现
