# B200 编译和调试指南

## 🚀 快速开始

### 1. 编译所有示例（发布模式）

```bash
./compile_b200.sh
```

### 2. 编译调试版本（带调试符号）

```bash
./compile_b200.sh -d
```

### 3. 只编译特定文件

```bash
# 只编译概念讲解
./compile_b200.sh simple

# 只编译对比示例（调试模式）
./compile_b200.sh -d compare
```

## 📋 完整使用说明

### 命令格式

```bash
./compile_b200.sh [选项] [目标]
```

### 选项

| 选项 | 说明 |
|------|------|
| `-d, --debug` | 编译调试版本，包含调试符号 (`-g -G -lineinfo`) |
| `-r, --release` | 编译发布版本（默认，优化性能） |
| `-h, --help` | 显示帮助信息 |

### 目标

| 目标 | 说明 |
|------|------|
| `simple` | 只编译 `atom_concept_simple.cu` |
| `compare` | 只编译 `cuda_vs_cute_comparison.cu` |
| `all` | 编译所有文件（默认） |

## 💡 常用场景

### 场景 1：首次学习，快速运行

```bash
# 编译发布版本（最快）
./compile_b200.sh

# 运行概念讲解
./atom_concept_simple

# 运行对比示例
./cuda_vs_cute_comparison
```

### 场景 2：调试代码，查看中间结果

```bash
# 编译调试版本
./compile_b200.sh -d

# 使用 cuda-gdb 调试
cuda-gdb ./atom_concept_simple_debug
```

在 cuda-gdb 中：
```gdb
(cuda-gdb) break demo_atom_concept    # 设置断点
(cuda-gdb) run                         # 运行程序
(cuda-gdb) cuda thread (0,0,0)         # 切换到线程 0
(cuda-gdb) print thr_mma               # 查看变量
(cuda-gdb) continue                    # 继续执行
```

### 场景 3：修改代码后快速测试

```bash
# 只重新编译修改的文件
./compile_b200.sh simple

# 立即运行测试
./atom_concept_simple
```

### 场景 4：性能分析

```bash
# 编译发布版本（无调试符号，性能最佳）
./compile_b200.sh -r compare

# 使用 Nsight Compute 分析
ncu --set full ./cuda_vs_cute_comparison

# 或者对比两个 kernel
ncu --kernel-name "regex:gemm.*" ./cuda_vs_cute_comparison
```

## 🐛 调试技巧

### 1. 添加调试输出

在代码中添加打印语句：

```cpp
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Debug: tiled_mma size = %d\n", size(tiled_mma));
    printf("Debug: thr_mma shape = ");
    print(thr_mma.shape());
    printf("\n");
}
```

重新编译：
```bash
./compile_b200.sh -d simple
./atom_concept_simple_debug
```

### 2. 使用 cuda-gdb 单步调试

```bash
# 编译调试版本
./compile_b200.sh -d compare

# 启动 cuda-gdb
cuda-gdb ./cuda_vs_cute_comparison_debug
```

常用命令：
```gdb
# 设置断点
(cuda-gdb) break gemm_cute              # 函数断点
(cuda-gdb) break file.cu:123            # 行号断点

# 运行和控制
(cuda-gdb) run                           # 开始运行
(cuda-gdb) step                          # 单步进入
(cuda-gdb) next                          # 单步跳过
(cuda-gdb) continue                      # 继续执行

# 查看变量
(cuda-gdb) print variable                # 打印变量
(cuda-gdb) info locals                   # 查看所有局部变量

# GPU 相关
(cuda-gdb) cuda thread                   # 查看当前 CUDA 线程
(cuda-gdb) cuda thread (0,0,0)           # 切换到 block(0,0), thread(0)
(cuda-gdb) cuda kernel                   # 查看当前 kernel
(cuda-gdb) info cuda threads             # 列出所有 CUDA 线程
```

### 3. 使用 compute-sanitizer 检查错误

```bash
# 编译调试版本
./compile_b200.sh -d

# 运行内存检查
compute-sanitizer --tool memcheck ./atom_concept_simple_debug

# 运行竞态条件检查
compute-sanitizer --tool racecheck ./atom_concept_simple_debug

# 运行初始化检查
compute-sanitizer --tool initcheck ./atom_concept_simple_debug
```

### 4. 使用 Nsight Compute 分析性能

```bash
# 编译发布版本（调试版本性能不准确）
./compile_b200.sh -r compare

# 基础性能分析
ncu ./cuda_vs_cute_comparison

# 详细分析特定 kernel
ncu --set full \
    --kernel-name "gemm_cute" \
    --launch-skip 0 \
    --launch-count 1 \
    ./cuda_vs_cute_comparison

# 对比两个 kernel
ncu --kernel-name "regex:gemm.*" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./cuda_vs_cute_comparison
```

## 📝 编译标志说明

### 发布模式（默认）

```bash
nvcc -std=c++17 -arch=sm_100 -I../../../include file.cu -o output
```

- **优点**：性能最佳，文件较小
- **缺点**：无法调试，错误信息较少
- **适用**：性能测试、最终运行

### 调试模式 (`-d`)

```bash
nvcc -std=c++17 -arch=sm_100 -g -G -lineinfo -I../../../include file.cu -o output_debug
```

- `-g`：生成主机端调试信息
- `-G`：生成设备端调试信息（**重要！**）
- `-lineinfo`：生成行号映射信息
- **优点**：可以使用 cuda-gdb 调试，错误信息详细
- **缺点**：性能较差（可能慢 10-100 倍），文件较大
- **适用**：开发、调试、学习

## ⚠️ 常见问题

### Q1: 编译失败："error: identifier ... is undefined"

**原因**：CuTe 头文件路径不正确

**解决**：
```bash
# 检查 include 目录是否存在
ls ../../../include/cute/

# 如果路径不对，修改脚本中的 INCLUDE_DIR 变量
```

### Q2: 运行时错误："no kernel image is available for execution"

**原因**：编译的架构与 GPU 不匹配

**解决**：
```bash
# 确认你的 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# B200 应该是 10.0 (sm_100)
# 如果不是，修改脚本中的 ARCH 变量
```

### Q3: 调试版本运行很慢

**原因**：`-G` 标志会禁用所有优化

**解决**：
- 调试时使用小数据集
- 只在需要调试时编译调试版本
- 性能测试必须使用发布版本

### Q4: cuda-gdb 无法显示变量值

**原因**：编译器优化导致变量被优化掉

**解决**：
- 确保使用调试版本 (`-d`)
- 在变量定义后立即设置断点
- 使用 `volatile` 关键字防止优化（不推荐）

## 🔧 自定义编译

如果需要更多控制，可以手动编译：

```bash
# 基础编译
nvcc -std=c++17 -arch=sm_100 \
     -I../../../include \
     atom_concept_simple.cu \
     -o atom_concept_simple

# 调试编译
nvcc -std=c++17 -arch=sm_100 \
     -g -G -lineinfo \
     -I../../../include \
     atom_concept_simple.cu \
     -o atom_concept_simple_debug

# 添加额外标志
nvcc -std=c++17 -arch=sm_100 \
     -O3 \                              # 最高优化级别
     --use_fast_math \                  # 使用快速数学库
     -Xcompiler -Wall \                 # 启用所有警告
     -I../../../include \
     atom_concept_simple.cu \
     -o atom_concept_simple_optimized
```

## 📚 进阶资源

### CUDA 调试工具文档

- [cuda-gdb 用户指南](https://docs.nvidia.com/cuda/cuda-gdb/)
- [compute-sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/)
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)

### CuTe 相关

- CuTe 文档：`/app/tensorrt_llm/cutlass/media/docs/cpp/cute/`
- CUTLASS 示例：`/app/tensorrt_llm/cutlass/examples/`

## 💻 完整工作流示例

```bash
# 1. 编译调试版本
./compile_b200.sh -d simple

# 2. 添加 printf 调试
vim atom_concept_simple.cu
# 添加你的调试代码

# 3. 重新编译
./compile_b200.sh -d simple

# 4. 运行查看输出
./atom_concept_simple_debug

# 5. 如果需要更深入调试，使用 cuda-gdb
cuda-gdb ./atom_concept_simple_debug

# 6. 确认无误后，编译发布版本测试性能
./compile_b200.sh -r simple
./atom_concept_simple

# 7. 性能分析
ncu --set full ./atom_concept_simple
```

---

**提示**：调试时推荐使用 VSCode + Nsight VSCode Extension，可以提供图形化的调试界面。
