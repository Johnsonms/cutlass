# TiledMMA 和 ThrMMA 调试指南

## 关键概念速查

```
TiledMMA:  Cluster 级别的 MMA 描述（整体规划）
ThrMMA:    单个 CTA 的 MMA 操作符（局部执行）
mma_v:     CTA 在 Cluster 中的编号 (V 维度)
mma_m:     MMA 在 M 维度的位置
mma_n:     MMA 在 N 维度的位置
```

## VSCode 调试步骤

### 1. 运行基础示例熟悉概念

```bash
cd /app/tensorrt_llm/cutlass/examples/cute/tutorial/blackwell
./mma_thread_concepts
```

**观察输出中的关键信息：**
- TiledMMA 的结构：`Shape_MNK: (_128,_256,_16)`
- VMNK 坐标映射：`Block[x,y] → V=?, M=?, N=?`
- 数据流：`TiledMMA → ThrMMA → partition → gemm`

### 2. 在 VSCode 中调试 01_mma_sm100.cu

打开文件并设置断点：

```
关键断点位置：

第 157 行: auto mma_coord_vmnk = make_coord(...)
  └─ 观察: blockIdx.x, blockIdx.y
  └─ 计算: V, M, N 坐标

第 203 行: auto mma_v = get<0>(mma_coord_vmnk);
  └─ 观察: mma_v 的值（0, 1, 2, ...）

第 204 行: ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  └─ 关键！从 TiledMMA 获取 ThrMMA
  └─ 观察: cta_mma 的类型和内容

第 205 行: Tensor tCgA = cta_mma.partition_A(gA);
  └─ 观察: gA 的 shape 和 tCgA 的 shape
  └─ 对比变化
```

### 3. 使用 cuda-gdb 调试

```bash
cd /app/tensorrt_llm/cutlass/build
cuda-gdb ./examples/cute/tutorial/blackwell/cute_tutorial_01_mma_sm100

# 在 gdb 中
(cuda-gdb) break 01_mma_sm100.cu:204
(cuda-gdb) run 512 1024 256

# 当断点命中时
(cuda-gdb) print mma_v
(cuda-gdb) print blockIdx.x
(cuda-gdb) print blockIdx.y
```

## 关键变量解读

### mma_coord_vmnk 的计算

```cpp
// 第 157-160 行
auto mma_coord_vmnk = make_coord(
    blockIdx.x % size<0>(cluster_layout_vmnk),  // V: Peer CTA
    blockIdx.x / size<0>(cluster_layout_vmnk),  // M: M 维度位置
    blockIdx.y,                                  // N: N 维度位置
    _                                           // K: 所有 K
);

// 例如：如果 cluster_size_v = 2, blockIdx.x = 5, blockIdx.y = 2
// V = 5 % 2 = 1
// M = 5 / 2 = 2
// N = 2
// 结果: (V=1, M=2, N=2, K=_)
```

### TiledMMA 的结构

```
TiledMMA
  ThrLayoutVMNK:  (_1,_1,_1,_1):(_0,_0,_0,_0)
    └─ 定义了 V, M, N, K 四个维度的线程布局

  MMA_Atom
    Shape_MNK:  (_128,_256,_16)
      └─ 单个 MMA 指令的大小: 128×256×16

    LayoutA_TV: (_1,(_128,_16)):(_0,(_1,_128))
      └─ A 矩阵的线程到值映射
```

### partition_A 的形状变换

```
输入 gA:   (128, 64, 4)
           │   │   └─ K tiles (256/64=4)
           │   └───── K per tile
           └───────── M dimension

输出 tCgA: ((_128,_16), _1, _4, 4)
           │  │    │   │   │   └─ K tiles (与 gA 相同)
           │  │    │   │   └───── K blocks (64/16=4)
           │  │    │   └───────── M MMA count (128/128=1)
           │  │    └───────────── K per MMA
           │  └────────────────── M per MMA
           └───────────────────── MMA 指令格式
```

## 坐标到数据的映射

### 完整流程示例

假设 GEMM: C[512,1024] = A[512,256] × B[1024,256]

```
1. Block[3,2] 的坐标计算:
   cluster_size_v = 2
   V = 3 % 2 = 1  (Peer 1)
   M = 3 / 2 = 1  (第1个 M tile)
   N = 2          (第2个 N tile)

2. 处理的数据范围:
   M 范围: [1*128, 2*128) = [128, 256)
   N 范围: [2*256, 3*256) = [512, 768)

3. 输出位置:
   C[128:256, 512:768]
```

## 常见问题

### Q1: V 坐标的作用？

**A:** V 坐标标识 CTA 在 Cluster 中的位置。在 Blackwell (SM100) 中，MMA 是 Cluster-scoped 的，多个 CTA 可以协同工作。V 坐标用于区分不同的 CTA，确保它们正确协作。

### Q2: TiledMMA 和 ThrMMA 的区别？

**A:**
- **TiledMMA**: 整个 Cluster 的 MMA 描述，包含所有 CTA 的信息
- **ThrMMA**: 单个 CTA 的 MMA 操作符，通过 `get_slice(mma_v)` 从 TiledMMA 提取

类比：总谱 vs 分谱

### Q3: partition_A 为什么改变形状？

**A:** partition_A 将逻辑上连续的矩阵块重新组织成 MMA 指令需要的格式。这样硬件 MMA 单元可以直接使用，无需额外的数据重排。

### Q4: 为什么 Block[0,0] 和 Block[1,0] 的 V 不同但处理相同位置？

**A:** 这是 Cluster-scoped MMA 的特点。V=0 和 V=1 的两个 CTA 可能处理相同的输出位置，但它们在 Cluster 内扮演不同角色（比如处理不同的 K 切片或协同累加）。

## 调试技巧

### 1. 打印关键坐标

在 kernel 中添加：

```cpp
if (thread0()) {
    printf("Block[%d,%d]: V=%d, M=%d, N=%d\n",
           blockIdx.x, blockIdx.y, mma_v, mma_m, mma_n);
}
```

### 2. 检查形状变换

```cpp
if (thread0()) {
    print("gA shape: "); print(shape(gA)); print("\n");
    print("tCgA shape: "); print(shape(tCgA)); print("\n");
}
```

### 3. 验证数据范围

```cpp
if (thread0()) {
    printf("Processing C[%d:%d, %d:%d]\n",
           mma_m * 128, (mma_m+1) * 128,
           mma_n * 256, (mma_n+1) * 256);
}
```

## 下一步

1. **运行** `mma_thread_concepts` 熟悉输出
2. **阅读** 输出中的坐标映射关系
3. **设置断点** 在 `01_mma_sm100.cu` 的关键位置
4. **单步调试** 观察变量变化
5. **对照** 本文档理解每个步骤

## 推荐学习路径

```
1. 运行 mma_thread_concepts
   └─ 理解 TiledMMA, ThrMMA, VMNK 概念

2. 在 01_mma_sm100.cu 设置断点
   └─ 第 204 行 (get_slice)
   └─ 第 205 行 (partition_A)

3. 启动调试，观察
   └─ mma_v 的值
   └─ cta_mma 的结构
   └─ tCgA 的形状变化

4. 单步执行主循环
   └─ 第 266 行 (k_tile 循环)
   └─ 第 289 行 (gemm 执行)

5. 理解完整数据流
   └─ GMEM → SMEM → TMEM
   └─ Tile → Partition → MMA
```

祝调试愉快！🎯
