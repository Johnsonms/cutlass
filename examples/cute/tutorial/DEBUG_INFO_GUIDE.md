# tiled_copy.cu Debug Information Guide

本文档说明在 `tiled_copy.cu` 中添加的所有调试打印信息。

## 编译选项

### 基础调试版本
```bash
./build_tiled_copy_debug.sh
./tiled_copy_debug
```

### 详细调试版本（包含所有打印）
```bash
./build_tiled_copy_verbose.sh
./tiled_copy_verbose
```

## 调试宏说明

### CUTE_DEBUG_COPY_ATOM
- **位置**: `/app/tensorrt_llm/tensorrt_llm/deep_gemm/include/cute/atom/copy_atom.hpp`
- **函数**: `make_tiled_copy()` (Line 508-514)
- **打印内容**:
  - `thr_layout`: 线程布局
  - `val_layout`: 值布局
  - `layout_mn`: raked_product 结果
  - `layout_tv`: TV Layout (right_inverse 结果)
  - `tiler`: 计算出的 Tiler_MN

### CUTE_DEBUG_PARTITION
- **位置**: `/app/tensorrt_llm/tensorrt_llm/deep_gemm/include/cute/atom/copy_atom.hpp`
- **函数**: `ThrCopy::partition_S()`, `ThrCopy::partition_D()`, 和 `tile2thrfrg()`
- **打印内容**:
  - 输入 tensor 的 layout
  - `tidfrg_S/D` 转换后的 TV layout
  - `thr_tensor` 的结构
  - 线程索引操作
  - **tile2thrfrg 详细步骤** (仅 thread 1, block (0,0)):
    - 输入 tensor layout
    - `atom_layout_TV`: zipped_divide 结果
    - `trg_layout_TV`: compose 变换结果
    - `thrval2mn`: coalesce 后的坐标映射
    - `tv_tensor`: 最终 TV 形式的 tensor
    - 索引后的最终结果

## 打印信息分层

### 1. Host-side Setup (main 函数中)

```
=== Host-side Setup ===
tensor_shape: (256, 512)
tensor_S: gmem_ptr[32b](0x...) o (256,512):(512,_1)
tensor_D: gmem_ptr[32b](0x...) o (256,512):(512,_1)
```

**解释**:
- 原始张量的形状和内存布局
- 数据指针和 stride 信息

### 2. Tiling Configuration

```
--- Tiling Configuration ---
block_shape: (_128,_64)
tiled_tensor_S: gmem_ptr[32b](0x...) o ((_128,_64),(2,8)):(...)
tiled_tensor_D: gmem_ptr[32b](0x...) o ((_128,_64),(2,8)):(...)
Number of tiles: (2, 8)
```

**解释**:
- Block shape: 128×64
- Tiled tensor shape: ((128,64), (2,8))
  - 第一个模式: Tile 内的形状
  - 第二个模式: Tile 的数量 (m', n')

### 3. TiledCopy Configuration

```
--- TiledCopy Configuration ---
thr_layout: (_32,_8):(_1,_32)
  Total threads: 256
val_layout: (_4,_1):(_1,_0)
  Values per thread: 4

Computed Tiler (thr * val):
  Tiler_M = 32 × 4 = 128
  Tiler_N = 8 × 1 = 8

Memory access pattern:
  Vector width: 16 bytes
  Elements per access: 4
```

**解释**:
- 线程布局: 32×8 = 256 个线程
- 值布局: 每线程 4 个元素
- **Tiler 计算**: 这是关键！
  - M 维度: 32 threads × 4 values = 128
  - N 维度: 8 threads × 1 value = 8

### 4. make_tiled_copy 内部 (CUTE_DEBUG_COPY_ATOM)

```
thr_layout: (_32,_8):(_1,_32)
val_layout: (_4,_1):(_1,_0)
layout_mn : ((_4,_32),(_1,_8)):((_256,_1),(_0,_32))
layout_tv : (_256,_4):(_4,_1)
tiler     : (_128,_8)
```

**解释**:
- `layout_mn`: (M,N) → (thr,val) 映射
  - Shape: ((4,32),(1,8)) - 嵌套的线程和值
  - Stride: ((256,1),(0,32))
- `layout_tv`: (thr,val) → (M,N) 映射（反向）
  - Shape: (256, 4) - 256 个线程，每个 4 个值
  - Stride: (4, 1)
- `tiler`: (128, 8) - 验证了我们的计算！

### 5. Kernel Launch Configuration

```
--- Kernel Launch Configuration ---
gridDim:  (2, 8, 1)
blockDim: (256, 1, 1)
Total threads: 4096
```

**解释**:
- Grid: 2×8 = 16 个 blocks
- Block: 256 个线程
- 总线程数: 16 × 256 = 4096

### 6. Device-side Info (copy_kernel)

```
=== copy_kernel Debug Info ===
Block (0, 0), Grid dim (2, 8)
tile_S: gmem_ptr[32b](0x...) o (_128,_64):(...)
tile_D: gmem_ptr[32b](0x...) o (_128,_64):(...)

--- Thread 0 ---
thr_tile_S: gmem_ptr[32b](0x...) o (_4,_1):(...)
thr_tile_D: gmem_ptr[32b](0x...) o (_4,_1):(...)

--- Thread 1 ---
thr_tile_S: gmem_ptr[32b](0x...) o (_4,_1):(...)
...
```

**解释**:
- 每个 block 处理的 tile: 128×64
- 每个线程处理的部分: 4×1 = 4 个元素

### 7. Device-side Info (copy_kernel_vectorized)

```
=== copy_kernel_vectorized Debug Info ===
Block (0, 0), Grid dim (2, 8)
tiled_copy: Copy_Atom...
tile_S: gmem_ptr[32b](0x...) o (_128,_8):(...)

--- Thread 1 (vectorized) ---
thr_copy: ThrCopy...

=== partition_S (thread 1) ===
stensor.layout(): (_128,_8):(...)
tidfrg_S result (TV layout): ((_256,_4),(_1,_8)):(...)
thr_tensor: gmem_ptr[32b](0x...) o ((_256,_4),(_1,_8)):(...)
Indexing with thr_idx_=1, keeping all values

thr_tile_S: gmem_ptr[32b](0x...) o ((_4,_1),(_1,_8)):(...)
thr_tile_D: gmem_ptr[32b](0x...) o ((_4,_1),(_1,_8)):(...)
thr_tile_S size: 32 elements
```

**解释**:
- 展示了 `tidfrg_S` 的 TV Layout 转换
- `thr_tensor` 的结构: ((256,4), (1,8))
  - (256, 4): Thread-Value 维度
  - (1, 8): Rest 维度
- 线程 1 索引后: ((4,1), (1,8))
  - 每个线程处理 4×1×1×8 = 32 个元素

### 8. tile2thrfrg 内部转换细节 (新增)

```
=== tile2thrfrg (thread 1, block (0,0)) ===
Input tensor.layout(): ((_128,_64),(_2,_8)):(...)
atom_layout_TV (zipped_divide result): ((_32,_4),(_8,_1)):(...)
trg_layout_TV (compose result): ((_32,_4),(_8,_1)):(...)
thrval2mn (coalesce result): ((_256,_4),(_1,_8)):(...)
tv_tensor (tensor.compose result): gmem_ptr[32b](0x...) o ((_256,_4),(_1,_8)):(...)
Final result (tv_tensor indexed): gmem_ptr[32b](0x...) o ((_256,_4),(_1,_8)):(...)
```

**解释**:
这是 TV Layout 转换的最核心步骤，展示了 `tile2thrfrg` 函数内部的完整变换流程：

1. **Input tensor.layout()**:
   - 输入是 `zipped_divide` 的结果: `((TileM, TileN), (RestM, RestN))`
   - 例如: `((128, 64), (2, 8))` - 表示 128×64 的 tile，有 2×8 个这样的 tiles

2. **atom_layout_TV (zipped_divide 结果)**:
   - `zipped_divide(TiledLayout_TV, make_shape(AtomNumThr, AtomNumVal))`
   - 将 TiledLayout_TV (256, 4):(4, 1) 按照 (Thr=32, Val=4) 分组
   - 结果: `((32, 4), (8, 1))` - 嵌套结构
     - 第一组 (32, 4): atom 内的 thread 和 value
     - 第二组 (8, 1): 剩余的 thread 和 value

3. **trg_layout_TV (compose 结果)**:
   - `atom_layout_TV.compose(ref2trg, _)`
   - ref2trg 是 `right_inverse(AtomLayoutRef).compose(AtomLayoutSrc)`
   - 对于简单情况（源和目标相同），这一步可能是恒等变换
   - 结果形状与 atom_layout_TV 相同: `((32, 4), (8, 1))`

4. **thrval2mn (coalesce 结果)**:
   - `coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1,_1>>{})`
   - zip 操作重新组织 thread 和 value 维度
   - coalesce 将嵌套结构合并：
     - Thread 维度: 32 × 8 = 256
     - Value 维度: 4 × 1 = 4
   - 结果: `((256, 4), (1, 8))` - 标准 TV Layout 形式！
     - (256, 4): Thread-Value 扁平化
     - (1, 8): Rest 维度（tile 数量）

5. **tv_tensor (tensor.compose 结果)**:
   - `tensor.compose(thrval2mn, _)`
   - 将 thrval2mn 的坐标映射应用到输入 tensor
   - 结果是带有 TV Layout 的 tensor: `((256, 4), (1, 8))`

6. **Final result (索引后)**:
   - `tv_tensor(make_coord(_, _), _)`
   - 索引操作 `(_, _)` 选择第一个模式，保持后续维度
   - 最终形状仍然是: `((256, 4), (1, 8))`
   - 这个结果会在 `partition_S` 中被进一步索引为单个线程的数据

**关键洞察**:
- **zipped_divide**: 创建嵌套的 atom/rest 结构
- **compose**: 应用坐标变换（源到目标的映射）
- **coalesce**: 将嵌套的 ((32,4),(8,1)) 扁平化为 ((256,4),(1,8))
- **最终的 TV Layout**: (Thread=256, Value=4, Rest=...)

这个函数是理解 CuTe 的关键，它完成了从 tile 结构到 thread-value 结构的完整转换！

## 关键概念对应

| 概念 | Host 计算 | Device 观察 |
|------|----------|------------|
| **Tiler** | 32×4 = 128 (M)<br>8×1 = 8 (N) | 每个 tile: 128×8 |
| **Thread partition** | layout_tv: (256, 4) | 每线程: 4 个元素 |
| **Grid dimension** | (2, 8) blocks | 16 个 blocks |
| **Total coverage** | 128×8×2×8 = 131072 | 256×512 = 131072 ✓ |

## 调试流程建议

1. **先运行 verbose 版本**:
   ```bash
   ./tiled_copy_verbose > output.txt 2>&1
   ```

2. **分析 Host-side 配置**:
   - 确认 thr_layout 和 val_layout
   - 验证 Tiler 计算
   - 检查 layout_tv 的 stride

3. **查看 Device-side 分区**:
   - 检查线程 0 和线程 1 的数据
   - 验证 tidfrg_S 的 TV Layout
   - 确认每个线程处理的元素数

4. **使用 cuda-gdb 深入调试**:
   ```bash
   cuda-gdb ./tiled_copy_verbose
   (cuda-gdb) break tiled_copy.cu:147
   (cuda-gdb) run
   (cuda-gdb) cuda thread (0,0,0) (0,0,1)
   (cuda-gdb) print thr_tile_S
   ```

## 恢复原始文件

```bash
# 恢复 copy_atom.hpp
cp /app/tensorrt_llm/tensorrt_llm/deep_gemm/include/cute/atom/copy_atom.hpp.original \
   /app/tensorrt_llm/tensorrt_llm/deep_gemm/include/cute/atom/copy_atom.hpp

# 移除调试打印（可选）
git checkout tiled_copy.cu
```

## 总结

所有调试打印从三个层次展示了 CuTe 的工作机制：

1. **Host-side**: 配置和计算 TiledCopy 参数
2. **Conversion**: TV Layout 转换（raked_product, right_inverse）
3. **Device-side**: 实际的线程分区和数据访问

通过这些打印，你可以完整理解从 `thr_layout` 和 `val_layout` 到最终线程数据分配的整个过程！
