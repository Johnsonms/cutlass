# CuTe TiledMMA 和 ThrMMA 详解

## 目录
1. [前置知识：TV Layout 概念](#1-前置知识tv-layout-概念)
2. [TiledMMA 类详解](#2-tiledmma-类详解)
3. [ThrMMA 类详解](#3-thrmma-类详解)
4. [Atom 与 MMA 指令的关系](#4-atom-与-mma-指令的关系)
5. [ThrLayoutVMNK 的 Stride 解析](#5-thrlayoutvmnk-的-stride-解析)
6. [完整示例](#6-完整示例)
7. [总结](#7-总结)

---

## 1. 前置知识：TV Layout 概念

### 1.1 对话背景

有开发者询问 Eric Auld 关于"计算资源分块"的问题：

> "我理解了数据维度的分块（M-tiles, N-tiles, K-tiles），但不清楚线程分块层次和 Tensor Core（计算资源）是如何映射的？"

### 1.2 Eric 的解答：TV Layout

**TV Layout** 是一个通用策略，用于描述"线程-数据"映射：

```cpp
Layout<Shape<T, V>, Stride<...>>
```

- **T 模式（Thread）**：线程维度
- **V 模式（Value）**：每个线程处理的值/元素维度
- 每个模式都可以是**层次化的**（包含子模式）

### 1.3 使用方式

#### 查询特定线程的工作范围

```cpp
// 将线程 ID=17 代入 T 模式，保持 V 模式开放
layout(17, _)  // 返回线程 17 处理的所有元素的坐标
```

#### 向量化操作（Vectorized Access）

当线程需要**同步行进**（lockstep）访问内存时：

```cpp
// 依次将值代入 V 模式，查看每一步所有线程访问的地址
layout(_, 0)  // 第 0 步：所有线程访问的地址
layout(_, 1)  // 第 1 步：所有线程访问的地址
// ...
```

### 1.4 与代码的关联

`MMAThrLayout` 就是这种 **TV Layout 的具体实例**，用于描述 MMA 指令中线程如何映射到数据。

---

## 2. TiledMMA 类详解

### 2.1 定义

**文件位置**：`/app/tensorrt_llm/cutlass/include/cute/atom/mma_atom.hpp:208-457`

```cpp
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom
{
  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_;

  // ...
};
```

### 2.2 核心概念

**TiledMMA 描述的是"所有线程如何协同完成一个大的 MMA 操作"**

#### 模板参数

| 参数 | 含义 | 示例 |
|------|------|------|
| **MMA_Atom** | 单个 MMA 原子操作 | `SM80_16x8x16_F32F16F16F32_TN` |
| **AtomLayoutMNK** | Atom 在 M/N/K 的平铺方式 | `Layout<Shape<_2,_4,_1>>` |
| **PermutationMNK** | 应用到 M/N/K 的排列 | `Tile<Underscore,Underscore,Underscore>` |

#### 核心成员：ThrLayoutVMNK

```cpp
// 通过 tiled_product 计算得到
ThrLayoutVMNK = tiled_product(AtomThrID, AtomLayoutMNK)
```

**四维结构**：
```
Shape<ThrV, ThrM, ThrN, ThrK>
```

- **ThrV**: Atom 内部的线程 ID（来自 AtomThrID）
- **ThrM/N/K**: 跨 Atom 平铺的线程坐标

**示例**：
```cpp
// 假设：
// - 单个 Atom 需要 32 个线程（一个 warp）
// - AtomLayoutMNK = Layout<Shape<_2,_4,_1>>
//
// 则：
ThrLayoutVMNK = Shape<_32, _2, _4, _1>
// 总线程数 = 32 × 2 × 4 × 1 = 256 个线程（8 个 warp）
```

### 2.3 关键方法

#### thrfrg_C/A/B - 张量分块

**功能**：将全局张量转换为 TV Layout 形式

**以 thrfrg_C 为例**（`mma_atom.hpp:252-274`）：

```cpp
// 输入：(M, N, ...) 形状的张量
// 输出：((ThrV,(ThrM,ThrN)), (FrgV,(RestM,RestN,...)))
//       ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
//       线程维度 (T)          值/元素维度 (V)
```

**转换流程**：

```
(M,N)                                    // 原始形状
→ (PermM,PermN)                          // 应用排列
→ ((AtomM,AtomN),(RestM,RestN))         // 按 Atom 形状切分
→ ((ThrV,FrgV),(RestM,RestN))           // 映射到 TV 布局
→ ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN))) // 按线程平铺切分
```

**这就是 TV Layout 的实现！**

#### get_slice(thread_idx) - 获取单线程视角

```cpp
template <class ThrIdx>
CUTE_HOST_DEVICE constexpr
auto get_slice(ThrIdx const& thr_idx) const
{
  auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
  return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
}
```

**用法**：
```cpp
auto thr_mma = tiled_mma.get_slice(42);  // 获取线程 42 的视角
// 返回 ThrMMA 对象
```

### 2.4 可视化示例

```cpp
// 配置
TiledMMA<MMA_Atom<SM80_16x8x16_...>,
         Layout<Shape<_2, _4, _1>>>

// 形状：
// - 单个 Atom: 16×8 (M×N)
// - 平铺: 2×4 个 Atom
// - 总形状: 32×32 (M×N)
```

**Atom 网格布局**：
```
         N 维度 (32 = 4 × 8)
     ┌────┬────┬────┬────┐
     │ A0 │ A1 │ A2 │ A3 │  M 维度
M    ├────┼────┼────┼────┤  (32 = 2 × 16)
     │ A4 │ A5 │ A6 │ A7 │
     └────┴────┴────┴────┘

每个格子 (A0-A7) 是一个 Atom (16×8)
总共 8 个 Atom = 2(M) × 4(N) × 1(K)
```

---

## 3. ThrMMA 类详解

### 3.1 定义

**文件位置**：`/app/tensorrt_llm/cutlass/include/cute/atom/mma_atom.hpp:459-520`

```cpp
template <class TiledMMA, class ThrVMNK>
struct ThrMMA : TiledMMA
{
  ThrVMNK thr_vmnk_;  // 该线程在 VMNK 空间中的坐标

  // partition_C/A/B 方法
  // partition_fragment_C/A/B 方法
};
```

### 3.2 核心概念

**ThrMMA 描述的是"单个线程看到的数据分区"**

#### 核心成员：thr_vmnk_

```cpp
ThrVMNK thr_vmnk_;  // 例如：(v=5, m=1, n=2, k=0)
```

这个坐标标识了**当前线程在 TiledMMA 中的位置**。

### 3.3 关键方法

#### partition_C/A/B - 提取线程数据

**以 partition_C 为例**（`mma_atom.hpp:467-473`）：

```cpp
template <class CTensor>
CUTE_HOST_DEVICE constexpr
auto partition_C(CTensor&& ctensor) const
{
  // 1. 将全局张量按 TiledMMA 的 TV 布局重新组织
  auto thr_tensor = make_tensor(
    static_cast<CTensor&&>(ctensor).data(),
    this->thrfrg_C(ctensor.layout())
  );
  // 形状：((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN)))

  // 2. 使用 thr_vmnk_ 索引线程维度，保持值维度开放
  auto thr_vmn = make_coord(
    get<0>(thr_vmnk_),
    make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_))
  );
  return thr_tensor(thr_vmn, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  //                 ^^^^^^              ^
  //                 线程坐标            保持所有值维度
}
```

**这正是 Eric 对话中描述的操作**：
> "plug 17 into the first mode and leave the second mode open"

```cpp
// 等价概念：
layout(thr_id, _)  // 代入线程 ID，保持值维度开放
```

#### partition_fragment_A/B/C - 创建寄存器片段

```cpp
template <class CTensor>
CUTE_HOST_DEVICE constexpr
auto partition_fragment_C(CTensor&& ctensor) const {
  return TiledMMA::make_fragment_C(partition_C(ctensor));
}
// 先分区，再创建适合该线程的寄存器布局
```

---

## 4. Atom 与 MMA 指令的关系

### 4.1 核心结论

✅ **Atom = 单条 MMA 硬件指令**

### 4.2 具体示例

一个 `MMA_Atom` 封装了**一条硬件 MMA 指令**：

```cpp
// Tensor Core 指令（SM80/Ampere）
mma.sync.aligned.m16n8k16.f32.f16.f16.f32

// 对应的 CuTe Atom：
using Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
```

**这条指令的特性**：
- **Shape_MNK**: `(16, 8, 16)`
  - 计算 16×8 的输出矩阵
  - 消耗 K 维度的 16 个元素
- **ThrID**: `32` 个线程（一个 warp）
- **TV Layout**: 定义 32 个线程如何映射到 16×8×16 的数据

### 4.3 平铺 Atom 的含义

#### 示例：`Layout<Shape<_2, _4, _1>>`

```cpp
TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
         Layout<Shape<_2, _4, _1>>>  // AtomLayoutMNK
```

**含义**：
```
在 M 维度平铺 2 个 Atom → 2 × 16 = 32 (M 方向)
在 N 维度平铺 4 个 Atom → 4 × 8  = 32 (N 方向)
在 K 维度平铺 1 个 Atom → 1 × 16 = 16 (K 方向)
```

#### 可视化

```
         N 维度 (32 = 4 × 8)
     ┌────┬────┬────┬────┐
     │ A0 │ A1 │ A2 │ A3 │  M 维度
M    ├────┼────┼────┼────┤  (32 = 2 × 16)
     │ A4 │ A5 │ A6 │ A7 │
     └────┴────┴────┴────┘

每个格子 (A0-A7) 是一个 Atom (16×8)
总共 8 个 Atom = 2(M) × 4(N) × 1(K)
```

### 4.4 线程资源分配

#### 单个 Atom 的线程需求

```cpp
// 单个 Atom 需要 32 个线程（一个 warp）
AtomThrID = 32
```

#### TiledMMA 的总线程数

```cpp
// AtomLayoutMNK = Layout<Shape<_2, _4, _1>>
ThrLayoutVMNK = tiled_product(AtomThrID, AtomLayoutMNK)
              = Shape<32, 2, 4, 1>

总线程数 = 32 × 2 × 4 × 1 = 256 个线程
         = 8 个 warp
```

**解释**：
- **32 (ThrV)**: 每个 Atom 内部的线程
- **2 (ThrM)**: M 维度平铺需要 2 组线程
- **4 (ThrN)**: N 维度平铺需要 4 组线程
- **1 (ThrK)**: K 维度不需要额外线程

### 4.5 实际硬件行为

```cpp
// 8 个 warp 同时执行（如果 SM 资源允许）：
Warp 0 (threads  0-31 ): Atom[0,0,0] 的 m16n8k16 指令
Warp 1 (threads 32-63 ): Atom[0,1,0] 的 m16n8k16 指令
Warp 2 (threads 64-95 ): Atom[0,2,0] 的 m16n8k16 指令
Warp 3 (threads 96-127): Atom[0,3,0] 的 m16n8k16 指令
Warp 4 (threads 128-159): Atom[1,0,0] 的 m16n8k16 指令
Warp 5 (threads 160-191): Atom[1,1,0] 的 m16n8k16 指令
Warp 6 (threads 192-223): Atom[1,2,0] 的 m16n8k16 指令
Warp 7 (threads 224-255): Atom[1,3,0] 的 m16n8k16 指令
```

**每个 warp 执行的是同一条硬件指令**，但操作不同的数据地址。

### 4.6 关键要点

| 概念 | 含义 |
|------|------|
| **Atom** | 单条 MMA 硬件指令（如 `m16n8k16`） |
| **平铺 Atom** | 在 M/N/K 维度上复制该指令 |
| **不是创造新指令** | 而是在不同数据区域执行相同指令 |
| **AtomLayoutMNK** | 决定需要多少个 Atom、多少个线程、每个线程负责哪块数据 |

---

## 5. ThrLayoutVMNK 的 Stride 解析

### 5.1 构造方式

```cpp
using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
```

**`tiled_product`** 将两个布局合并：
- **AtomThrID**: 单个 Atom 内部的线程布局
- **AtomLayoutMNK**: Atom 在 M/N/K 维度的平铺布局

### 5.2 具体示例

#### 假设配置

```cpp
// 单个 Atom 的线程布局（通常是简单的线性布局）
AtomThrID = Layout<Shape<_32>, Stride<_1>>
//          32 个线程，连续编号（stride=1）

// Atom 的平铺方式
AtomLayoutMNK = Layout<Shape <_2, _4, _1>,
                       Stride<_4, _1, _8>>
//              M: 2 个 Atom, stride=4
//              N: 4 个 Atom, stride=1
//              K: 1 个 Atom, stride=8
```

#### tiled_product 的结果

```cpp
ThrLayoutVMNK = Layout<Shape <_32, _2, _4, _1>,
                       Stride<_1,  _32, _64, _256>>
//                      ^    ^    ^    ^
//                      V    M    N    K
```

### 5.3 Stride 的含义

#### Stride 向量：`<1, 32, 64, 256>`

| 维度 | Shape | Stride | 含义 |
|------|-------|--------|------|
| **V** | 32 | **1** | Atom 内每个线程 ID +1，thread_idx +1 |
| **M** | 2  | **32** | M 维度每移动 1 个 Atom，thread_idx +32 |
| **N** | 4  | **64** | N 维度每移动 1 个 Atom，thread_idx +64 |
| **K** | 1  | **256** | K 维度每移动 1 个 Atom，thread_idx +256 |

#### 计算公式

**Stride 告诉我们：在 VMNK 逻辑坐标空间中移动时，物理线程 ID 如何变化**

```cpp
// 逻辑坐标 (v, m, n, k) → 物理线程 ID
thread_id = v * 1 + m * 32 + n * 64 + k * 256
```

### 5.4 实际映射示例

#### 示例 1：同一个 Atom 内的线程

```cpp
// Atom[0,0,0] 的线程
(v=0,  m=0, n=0, k=0) → thread_id = 0
(v=1,  m=0, n=0, k=0) → thread_id = 1
(v=31, m=0, n=0, k=0) → thread_id = 31

// 结论：Atom 内的 32 个线程是连续的 (stride=1)
```

#### 示例 2：跨 M 维度的 Atom

```cpp
// 从 Atom[0,0,0] 到 Atom[1,0,0]（M 维度 +1）
(v=0, m=0, n=0, k=0) → thread_id = 0
(v=0, m=1, n=0, k=0) → thread_id = 32   // +32

// 结论：M 维度每跨越一个 Atom，thread_id 跳跃 32
```

#### 示例 3：跨 N 维度的 Atom

```cpp
// 从 Atom[0,0,0] 到 Atom[0,1,0]（N 维度 +1）
(v=0, m=0, n=0, k=0) → thread_id = 0
(v=0, m=0, n=1, k=0) → thread_id = 64   // +64
(v=0, m=0, n=2, k=0) → thread_id = 128  // +128
(v=0, m=0, n=3, k=0) → thread_id = 192  // +192

// 结论：N 维度每跨越一个 Atom，thread_id 跳跃 64
```

#### 示例 4：完整的 Atom 网格

```
N 维度 →
┌────────┬────────┬────────┬────────┐
│ 0-31   │ 64-95  │ 128-159│ 192-223│  ← M=0
├────────┼────────┼────────┼────────┤
│ 32-63  │ 96-127 │ 160-191│ 224-255│  ← M=1
└────────┴────────┴────────┴────────┘
  N=0      N=1      N=2      N=3

每个格子内是 32 个连续的线程 ID
```

### 5.5 逆向计算示例

#### 使用 get_flat_coord

```cpp
auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(100);
// 计算 thread_id=100 的逻辑坐标

// 逆向计算（根据 stride）：
// 100 = v*1 + m*32 + n*64 + k*256
//
// k = 100 / 256 = 0
// 剩余 100
// n = 100 / 64 = 1
// 剩余 36
// m = 36 / 32 = 1
// 剩余 4
// v = 4 / 1 = 4
//
// 结果：(v=4, m=1, n=1, k=0)
```

**验证**：
```cpp
thread_id = 4*1 + 1*32 + 1*64 + 0*256 = 100 ✓
```

这个线程属于 **Atom[m=1, n=1, k=0]**，是该 Atom 内的第 **4 号**线程。

### 5.6 Stride 计算规则

#### tiled_product 的原理

```cpp
// 伪代码逻辑
Stride_V = Stride_AtomThrID = 1
Stride_M = size(AtomThrID) * Stride_AtomLayoutMNK[M]
         = 32 * 1 = 32
Stride_N = size(AtomThrID) * Stride_AtomLayoutMNK[N]
         = 32 * 2 = 64
Stride_K = size(AtomThrID) * Stride_AtomLayoutMNK[K]
         = 32 * 8 = 256
```

**关键规则**：
- **V 维度**：保持 Atom 内部的原始 stride（通常是 1）
- **M/N/K 维度**：原始 stride × Atom 的线程数

### 5.7 不同配置的 Stride 示例

#### 配置 A：行优先平铺

```cpp
AtomLayoutMNK = Layout<Shape <_2, _4, _1>,
                       Stride<_1, _2, _8>>  // M 优先
ThrLayoutVMNK = Layout<Shape <_32, _2,  _4,  _1>,
                       Stride<_1,  _32, _64, _256>>
```

**布局**（线程 ID 分布）：
```
M 维度
↓
┌─────┬─────┬─────┬─────┐
│ 0   │ 64  │ 128 │ 192 │ M=0, N=0-3
├─────┼─────┼─────┼─────┤
│ 32  │ 96  │ 160 │ 224 │ M=1, N=0-3
└─────┴─────┴─────┴─────┘
```

#### 配置 B：列优先平铺

```cpp
AtomLayoutMNK = Layout<Shape <_2, _4, _1>,
                       Stride<_4, _1, _8>>  // N 优先
ThrLayoutVMNK = Layout<Shape <_32, _2,   _4,  _1>,
                       Stride<_1,  _128, _32, _256>>
```

**布局**（线程 ID 分布）：
```
M 维度
↓
┌─────┬─────┬─────┬─────┐
│ 0   │ 32  │ 64  │ 96  │ M=0, N=0-3
├─────┼─────┼─────┼─────┤
│ 128 │ 160 │ 192 │ 224 │ M=1, N=0-3
└─────┴─────┴─────┴─────┘
```

### 5.8 Stride 的实际作用

#### 在 partition_C 中的使用

```cpp
auto partition_C(CTensor&& ctensor) const
{
  auto thr_tensor = make_tensor(ctensor.data(), this->thrfrg_C(ctensor.layout()));

  // 使用 thr_vmnk_ 索引
  auto thr_vmn = make_coord(get<0>(thr_vmnk_),
                            make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
  return thr_tensor(thr_vmn, ...);
  //                ^^^^^^^^
  //                根据 stride 计算实际内存偏移
}
```

**Stride 决定了数据访问模式**：
- 不同的 stride → 不同的内存访问顺序
- 影响合并访问（coalescing）性能
- 影响 shared memory bank conflict

---

## 6. 完整示例

### 6.1 典型使用流程

```cpp
// 1. 定义一个 TiledMMA（在编译时）
using Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMMAType = TiledMMA<Atom,
                              Layout<Shape<_2,_4,_1>>,
                              Permutations>;
TiledMMAType tiled_mma;

// 2. 在运行时，每个线程获取自己的视角
int thread_id = threadIdx.x;
auto thr_mma = tiled_mma.get_slice(thread_id);

// 3. 使用 ThrMMA 分区全局数据
Tensor gC = make_tensor(...);  // 全局 C 矩阵
auto tCgC = thr_mma.partition_C(gC);  // 该线程负责的部分

// 4. 创建寄存器片段
auto tCrC = thr_mma.partition_fragment_C(gC);
auto tCrA = thr_mma.partition_fragment_A(gA);
auto tCrB = thr_mma.partition_fragment_B(gB);

// 5. 执行 MMA 操作
tiled_mma.call(tCrC, tCrA, tCrB);
```

### 6.2 线程到数据的完整映射

```cpp
// 假设 thread_id = 100
auto thr_vmnk = thr_layout_vmnk.get_flat_coord(100);
// 结果：(v=4, m=1, n=1, k=0)

// 解释：
// - 该线程属于 Atom[m=1, n=1, k=0]
// - 在该 Atom 内是第 4 号线程
// - 该 Atom 负责矩阵的 [16:32, 8:16] 区域（M×N）
// - 第 4 号线程负责该区域内特定的元素（由 Atom 的 TV Layout 定义）
```

---

## 7. 总结

### 7.1 概念层次关系

```
MMA_Atom (单个原子操作)
    ↓
TiledMMA (多个 Atom 的平铺，全局视角)
    ↓
ThrMMA (单个线程的视角)
```

### 7.2 核心类对比

| 类 | 视角 | 关键数据 | 主要功能 |
|----|------|---------|---------|
| **MMA_Atom** | 硬件指令 | `Shape_MNK`, `ThrID`, `LayoutC_TV` | 封装单个 MMA 指令 |
| **TiledMMA** | 全局 | `ThrLayoutVMNK` | 描述所有线程如何平铺 Atom |
| **ThrMMA** | 单线程 | `thr_vmnk_`（线程坐标） | 提取该线程的数据分区 |

### 7.3 TV Layout 与实现的对应

| Eric 的概念 | CuTe 实现 |
|------------|----------|
| **TV Layout** | `thrfrg_C/A/B` 返回的布局 |
| **T 模式（Thread）** | 第一个 mode：`(ThrV,(ThrM,ThrN))` |
| **V 模式（Value）** | 第二个 mode：`(FrgV,(RestM,RestN))` |
| **`layout(17, _)`** | `ThrMMA::partition_C` 的操作 |
| **Vectorized lockstep** | `partition_fragment` 后的内存访问模式 |

### 7.4 计算资源映射层次

```
物理资源（Tensor Core 指令）
    ↓
MMA_Atom (m16n8k16)
    ├─ ThrID: 32 threads (warp)
    └─ LayoutC_TV: 如何映射 (Thread, Value)
        ↓
TiledMMA
    ├─ AtomLayoutMNK: 如何平铺多个 Atom (2×4×1)
    └─ ThrLayoutVMNK: 所有线程的 VMNK 坐标 (32, 2, 4, 1)
        ├─ Shape: <32, 2, 4, 1>
        └─ Stride: <1, 32, 64, 256>
            ↓
ThrMMA (thread_id=100)
    └─ thr_vmnk_: 该线程的坐标 (v=4, m=1, n=1, k=0)
        ↓
partition_C(gC) → 返回该线程负责的元素
```

### 7.5 关键要点

1. **Atom = 单条 MMA 硬件指令**
   - 定义了基本的计算单元
   - 包含 shape、线程数、TV layout

2. **TiledMMA = Atom 的平铺**
   - 通过 `AtomLayoutMNK` 定义平铺方式
   - 生成 `ThrLayoutVMNK` 描述所有线程
   - 提供 `thrfrg_C/A/B` 实现 TV Layout

3. **ThrMMA = 单线程视角**
   - 通过 `get_slice(thread_id)` 获取
   - 使用 `partition_C/A/B` 提取该线程的数据
   - 实现了 "plug thread_id, leave values open" 的模式

4. **ThrLayoutVMNK 的 Stride 是核心**
   - 定义了线程 ID 到逻辑坐标的映射
   - 决定了数据访问模式和性能
   - 影响内存合并访问和 bank conflict

5. **TV Layout 是统一抽象**
   - T 模式（Thread）：描述哪些线程参与
   - V 模式（Value）：描述每个线程处理的数据
   - 贯穿整个 CuTe 框架的核心概念

---

## 参考

- **源文件**：`/app/tensorrt_llm/cutlass/include/cute/atom/mma_atom.hpp`
- **关键行号**：
  - MMA_Atom: 44-196
  - TiledMMA: 208-457
  - ThrMMA: 459-520
- **NVIDIA CUTLASS 文档**：https://github.com/NVIDIA/cutlass
- **CuTe 教程**：https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md

---

*文档生成时间：2025-12-21*
