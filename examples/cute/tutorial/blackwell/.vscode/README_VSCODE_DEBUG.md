# VSCode 调试 CuTe 教程

## 快速开始

### 1. 在 VSCode 中打开目录

```bash
# 打开这个目录
code /app/tensorrt_llm/cutlass/examples/cute/tutorial/blackwell
```

### 2. 安装推荐的扩展

VSCode 会提示安装推荐扩展：
- **C/C++** (ms-vscode.cpptools) - C++ 支持
- **CUDA** (nvidia.nsight-vscode-edition) - CUDA 调试支持

点击"安装"即可。

### 3. 开始调试

#### 方法 1：使用调试面板

1. 按 `F5` 或点击左侧"运行和调试"图标
2. 选择调试配置：
   - `Debug 00_layout_basics` - Layout 基础
   - `Debug 00_tensor_basics` - Tensor 基础
   - `Debug 00_tile_basics` - Tile 基础
   - `Debug 01_mma_sm100` - SM100 MMA
3. 点击绿色播放按钮开始调试

#### 方法 2：设置断点后调试

1. 打开源文件（如 `00_layout_basics.cu`）
2. 在想要停下的代码行左侧点击，设置**红色断点**
3. 按 `F5` 开始调试
4. 程序会在断点处暂停

### 4. 调试操作

| 操作 | 快捷键 | 说明 |
|-----|--------|------|
| **继续** | `F5` | 运行到下一个断点 |
| **单步跳过** | `F10` | 执行当前行，不进入函数 |
| **单步进入** | `F11` | 进入函数内部 |
| **单步跳出** | `Shift+F11` | 跳出当前函数 |
| **重启** | `Ctrl+Shift+F5` | 重新开始调试 |
| **停止** | `Shift+F5` | 停止调试 |

### 5. 查看变量

调试时可以：
- **悬停查看**：鼠标悬停在变量上查看值
- **变量面板**：左侧"变量"面板显示所有局部变量
- **监视**：在"监视"面板添加表达式
- **调试控制台**：输入表达式查看值

## 推荐的调试流程

### 学习 Layout（00_layout_basics.cu）

1. 打开 `00_layout_basics.cu`
2. 在这些关键行设置断点：
   ```
   第 19 行：auto row_major = make_layout(...)
   第 30 行：auto col_major = make_layout(...)
   第 44 行：auto A_layout = make_layout(...)
   ```
3. 按 `F5` 选择 "Debug 00_layout_basics"
4. 程序在断点停下时，查看变量值：
   - 鼠标悬停在 `row_major` 上
   - 在调试控制台输入 `p row_major`
   - 查看 shape 和 stride

### 学习 Tensor（00_tensor_basics.cu）

1. 打开 `00_tensor_basics.cu`
2. 关键断点位置：
   ```
   第 52 行：auto tensor = make_tensor(...)
   第 69 行：auto gmem_tensor = make_tensor(...)
   ```
3. 调试时观察：
   - `tensor` 的结构
   - 数据指针和 layout 的关系
   - 不同内存类型的区别

### 学习 01_mma_sm100.cu

1. 打开 `01_mma_sm100.cu`
2. 重要断点：
   ```
   第 371 行：TiledMMA tiled_mma = make_tiled_mma(...)
   第 388 行：auto bM = tile_size<0>(tiled_mma)
   第 535 行：Layout layout_A = make_layout(...)
   ```
3. 使用 "Debug 01_mma_sm100 (512x1024x256)" 配置
4. 单步执行，理解：
   - TiledMMA 的创建
   - MMA tile 的计算
   - Layout 和问题规模的关系

## 构建任务

按 `Ctrl+Shift+B` 或 `Cmd+Shift+B` 可以快速构建：

- **Build All CuTe Tutorials** (默认) - 构建所有教程
- **Build 00_layout_basics** - 只构建 Layout 示例
- **Build 00_tensor_basics** - 只构建 Tensor 示例
- **Build 00_tile_basics** - 只构建 Tile 示例
- **Build 01_mma_sm100** - 只构建 MMA 示例

## 调试技巧

### 1. 条件断点

右键断点 → "编辑断点" → 添加条件：
```
i == 5          // 当 i 等于 5 时停下
size > 100      // 当 size 大于 100 时停下
```

### 2. 日志点（Logpoint）

右键行号 → "添加日志点"，输入：
```
layout: {layout}
tensor size: {size(tensor)}
```

程序运行时会打印，但不会停下来。

### 3. 查看 CuTe 对象

在调试控制台（Debug Console）中：
```
p layout                    # 打印 layout
p size(tensor)              # 打印 tensor 大小
p shape(layout)             # 打印 shape
p stride(layout)            # 打印 stride
```

### 4. 自定义输入参数

使用 "Debug 01_mma_sm100 (custom size)" 配置，可以自定义 M、N、K 维度。

## 文件结构

```
.vscode/
├── launch.json          # 调试配置
├── tasks.json           # 构建任务
├── settings.json        # 编辑器设置
├── extensions.json      # 推荐扩展
└── README_VSCODE_DEBUG.md  # 本文档
```

## 常见问题

### Q: 找不到可执行文件？

**A:** 先构建项目：
```bash
cd /app/tensorrt_llm/cutlass/build
make cute_tutorial_00_layout_basics -j12
```

或在 VSCode 中按 `Ctrl+Shift+B` 选择构建任务。

### Q: 无法设置断点？

**A:** 确保：
1. 文件已保存
2. 已经成功构建（`-g` 调试符号）
3. 使用的是对应的调试配置

### Q: 调试器启动失败？

**A:** 检查：
1. cuda-gdb 是否安装：`which cuda-gdb`
2. 可执行文件路径是否正确
3. GPU 是否可用：`nvidia-smi`

### Q: 变量显示为 `<optimized out>`？

**A:** 这是因为编译器优化。可以：
1. 修改 CMakeLists.txt 降低优化级别
2. 使用日志点代替断点
3. 在关键位置添加 `volatile` 关键字

## 高级：修改编译选项

如果需要更好的调试体验，可以修改编译选项：

```bash
cd /app/tensorrt_llm/cutlass/build
cmake .. -DCUTLASS_NVCC_ARCHS=100a -DCMAKE_BUILD_TYPE=Debug
make cute_tutorial_00_layout_basics -j12
```

这会禁用优化，但编译速度会变慢。

## 资源

- [VSCode C++ 调试文档](https://code.visualstudio.com/docs/cpp/cpp-debug)
- [CUDA GDB 文档](https://docs.nvidia.com/cuda/cuda-gdb/)
- [CuTe 文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)

祝调试愉快！🎯
