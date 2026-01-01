/***************************************************************************************************
 * Atom 概念的最简化示例
 *
 * 本示例用最简单的代码展示 Atom 和 TiledMMA 的核心概念
 **************************************************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

// ═══════════════════════════════════════════════════════════════════════════════
// 示例 1：理解 Atom - 最基本的计算单元
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * 什么是 Atom？
 *
 * 想象你在工厂工作：
 * - 传统方式：你需要知道如何操作每台机器，如何协调工人，如何分配任务
 * - Atom 方式：Atom 是一台"标准机器"的说明书，告诉你这台机器能做什么
 *
 * Atom 的特点：
 * 1. 描述单个硬件操作（如一个 FMA 指令，或一个 Tensor Core 指令）
 * 2. 定义操作的"形状"（处理多少数据）
 * 3. 不关心有多少工人（线程）使用它
 */

__global__ void demo_atom_concept() {
    if (threadIdx.x == 0) {
        printf("\n╔════════════════════════════════════════════════════════╗\n");
        printf("║          Atom 概念演示                                 ║\n");
        printf("╚════════════════════════════════════════════════════════╝\n\n");

        // ─────────────────────────────────────────────────────────────────
        // Atom 例子 1：标量 FMA
        // ─────────────────────────────────────────────────────────────────
        printf("【Atom 例子 1】：标量 FMA (Fused Multiply-Add)\n");
        printf("─────────────────────────────────────────────────────────────\n");

        using FMA_Atom = UniversalFMA<float, float, float, float>;

        printf("类型：UniversalFMA<float>\n");
        printf("描述：执行一个浮点 FMA 操作\n");
        printf("形状：(1, 1, 1) - 处理单个标量\n");
        printf("PTX 指令：fma.f32 d, a, b, c  // d = a*b + c\n");
        printf("硬件：任何 CUDA 核心\n");
        printf("\n");
        printf("传统 CUDA 等价代码：\n");
        printf("    float c = 0.0f;\n");
        printf("    c += a * b;  // ← 这就是一个 Atom！\n");
        printf("\n");

        // ─────────────────────────────────────────────────────────────────
        // Atom 例子 2：Tensor Core MMA
        // ─────────────────────────────────────────────────────────────────
        printf("【Atom 例子 2】：Tensor Core MMA (仅概念说明)\n");
        printf("─────────────────────────────────────────────────────────────\n");
        printf("类型：SM80_16x8x16_F16F16F16F16_TN\n");
        printf("描述：一个 Tensor Core 矩阵乘法指令\n");
        printf("形状：(16, 8, 16) - M=16, N=8, K=16\n");
        printf("     C[16,8] += A[16,16] * B[8,16]\n");
        printf("PTX 指令：mma.sync.aligned.m16n8k16...\n");
        printf("硬件：Tensor Core (SM80+)\n");
        printf("\n");
        printf("一个 Tensor Core Atom 相当于：\n");
        printf("    for (m : 16) for (n : 8) for (k : 16)\n");
        printf("        C[m][n] += A[m][k] * B[n][k];\n");
        printf("但在硬件中并行执行！\n");
        printf("\n");

        // ─────────────────────────────────────────────────────────────────
        // Atom 例子 3：Copy Atom
        // ─────────────────────────────────────────────────────────────────
        printf("【Atom 例子 3】：Copy Atom\n");
        printf("─────────────────────────────────────────────────────────────\n");
        printf("类型：UniversalCopy<uint128_t>\n");
        printf("描述：一次复制 128 位 (16 字节)\n");
        printf("形状：处理 16 字节的数据\n");
        printf("PTX 指令：ld.global.v4.f32 或类似\n");
        printf("硬件：内存访问单元\n");
        printf("\n");
        printf("传统 CUDA 等价代码：\n");
        printf("    *((uint128_t*)dst) = *((uint128_t*)src);\n");
        printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 示例 2：理解 TiledMMA - Atom 的组织方式
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * 什么是 TiledMMA？
 *
 * 继续工厂的比喻：
 * - Atom：一台机器的说明书
 * - TiledMMA：工厂的组织方案
 *   - 有多少台这样的机器？
 *   - 如何排列这些机器？
 *   - 每台机器负责处理什么？
 *
 * TiledMMA 的作用：
 * 1. 决定使用多少个线程
 * 2. 决定每个线程如何使用 Atom
 * 3. 自动计算数据分配
 */

__global__ void demo_tiledmma_concept() {
    if (threadIdx.x == 0) {
        printf("\n╔════════════════════════════════════════════════════════╗\n");
        printf("║        TiledMMA 概念演示                               ║\n");
        printf("╚════════════════════════════════════════════════════════╝\n\n");

        printf("【概念】：从 Atom 到 TiledMMA\n");
        printf("─────────────────────────────────────────────────────────────\n");
        printf("\n");
        printf("第 1 步：选择一个 Atom\n");
        printf("    using MyAtom = UniversalFMA<float>;\n");
        printf("    这是我们的基本计算单元\n");
        printf("\n");

        printf("第 2 步：定义线程布局\n");
        printf("    Layout<Shape<_16, _8, _1>>{}\n");
        printf("    意思是：\n");
        printf("      - 16 个线程负责 M 维度\n");
        printf("      - 8 个线程负责 N 维度\n");
        printf("      - 1 表示 K 维度（每次处理 1 个元素）\n");
        printf("    总线程数：16 × 8 × 1 = 128 线程\n");
        printf("\n");

        printf("第 3 步：创建 TiledMMA\n");
        printf("    auto tiled_mma = make_tiled_mma(\n");
        printf("        MyAtom{},                      // Atom\n");
        printf("        Layout<Shape<_16, _8, _1>>{}   // 线程布局\n");
        printf("    );\n");
        printf("\n");

        printf("【结果】：TiledMMA 的工作原理\n");
        printf("─────────────────────────────────────────────────────────────\n");
        printf("\n");
        printf("现在 TiledMMA 知道：\n");
        printf("  1. 使用 128 个线程\n");
        printf("  2. 每个线程执行 1 个 FMA Atom\n");
        printf("  3. 线程排列成 16×8 的网格\n");
        printf("  4. 整体处理的矩阵形状：\n");
        printf("     - M 维度：16 个线程 × 1 (Atom 的 M) = 16\n");
        printf("     - N 维度：8 个线程 × 1 (Atom 的 N) = 8\n");
        printf("     - K 维度：1 (每次迭代)\n");
        printf("\n");

        printf("【对比】：传统 CUDA vs TiledMMA\n");
        printf("─────────────────────────────────────────────────────────────\n");
        printf("\n");
        printf("传统 CUDA - 需要手动编写：\n");
        printf("    int tid = threadIdx.x;\n");
        printf("    int row = tid / 8;        // 手动计算\n");
        printf("    int col = tid %% 8;        // 手动计算\n");
        printf("    if (row < 16 && col < 8) {\n");
        printf("        // 手动计算全局索引\n");
        printf("        int idx = ...; \n");
        printf("        C[idx] += A[...] * B[...];\n");
        printf("    }\n");
        printf("\n");

        printf("TiledMMA - 自动处理：\n");
        printf("    auto thr_mma = tiled_mma.get_slice(threadIdx.x);\n");
        printf("    Tensor tCsA = thr_mma.partition_A(sA);  // 自动分割\n");
        printf("    Tensor tCsB = thr_mma.partition_B(sB);  // 自动分割\n");
        printf("    Tensor tCrC = thr_mma.partition_C(gC);  // 自动分割\n");
        printf("    gemm(tiled_mma, tCsA, tCsB, tCrC);      // 自动计算\n");
        printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 示例 3：ThrMMA - 单个线程的视角
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void demo_thrmma_concept() {
    // 创建一个简单的 TiledMMA
    using MyAtom = UniversalFMA<float, float, float, float>;
     auto tiled_mma = make_tiled_mma(
        MyAtom{},
        Layout<Shape<_4, _4, _1>>{}  // 4×4 = 16 线程
    );

    // 每个线程获取自己的 ThrMMA
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    if (threadIdx.x < 16) {  // 只使用前 16 个线程
        printf("线程 %2d: ", threadIdx.x);
        printf("我负责的位置 (M, N) 可以通过 partition 自动计算\n");

        // 前几个线程展示详细信息
        if (threadIdx.x < 4) {
            int m_coord = threadIdx.x / 4;  // 简化示例
            int n_coord = threadIdx.x % 4;
            printf("         → 示例：可能负责 M=%d, N=%d 的数据\n", m_coord, n_coord);
        }
    }

    if (threadIdx.x == 0) {
        printf("\n【关键点】：\n");
        printf("- ThrMMA 封装了 \"这个线程\" 的所有信息\n");
        printf("- 使用 partition_A/B/C 自动计算数据分割\n");
        printf("- 不需要手动计算索引！\n");
        printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 示例 4：简化的小例子（手动实现，不使用 TiledMMA）
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void tiny_gemm_example() {
    // 假设有 4×4 的矩阵在共享内存
    __shared__ float sA[4][4];
    __shared__ float sB[4][4];
    __shared__ float sC[4][4];

    // 初始化（简化）
    if (threadIdx.x < 16) {
        int row = threadIdx.x / 4;
        int col = threadIdx.x % 4;
        sA[row][col] = row + 1.0f;  // 简单的值便于验证
        sB[row][col] = col + 1.0f;
        sC[row][col] = 0.0f;
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // 手动 GEMM（演示 Atom 的概念）
    // ═══════════════════════════════════════════════════════════════════

    if (threadIdx.x < 16) {
        int row = threadIdx.x / 4;
        int col = threadIdx.x % 4;

        // 每个线程计算 C 的一个元素
        float acc = 0.0f;
        for (int k = 0; k < 4; ++k) {
            // 这里的每个乘加操作就是一个 "Atom"！
            // 相当于 UniversalFMA<float>
            acc += sA[row][k] * sB[k][col];
            //     ↑ 这就是 Atom 在做的事情
        }
        sC[row][col] = acc;
    }

    __syncthreads();

    // 打印结果（仅第一个线程）
    if (threadIdx.x == 0) {
        printf("\n【4×4 矩阵乘法结果】：\n");
        printf("C = A * B (手动实现，每个 FMA 就是一个 Atom)\n\n");
        printf("A 矩阵:\n");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                printf("%6.1f ", sA[i][j]);
            }
            printf("\n");
        }
        printf("\nB 矩阵:\n");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                printf("%6.1f ", sB[i][j]);
            }
            printf("\n");
        }
        printf("\nC 矩阵 = A * B:\n");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                printf("%6.1f ", sC[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("        CuTe Atom 和 TiledMMA 概念讲解\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    // 示例 1：Atom 概念
    demo_atom_concept<<<1, 32>>>();
    cudaDeviceSynchronize();

    // 示例 2：TiledMMA 概念
    demo_tiledmma_concept<<<1, 32>>>();
    cudaDeviceSynchronize();

    // 示例 3：ThrMMA 概念
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║        ThrMMA 概念演示                                 ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    demo_thrmma_concept<<<1, 32>>>();
    cudaDeviceSynchronize();

    // 示例 4：完整小例子
    tiny_gemm_example<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("                    总结\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    printf("1. Atom：最基本的计算单元（一个硬件指令）\n");
    printf("   例子：UniversalFMA = 一个标量乘加操作\n\n");
    printf("2. TiledMMA：Atom 在多线程中的组织方式\n");
    printf("   作用：决定多少线程、如何排列、每个线程做什么\n\n");
    printf("3. ThrMMA：单个线程从 TiledMMA 获取的信息\n");
    printf("   作用：知道自己负责哪些数据，如何计算\n\n");
    printf("4. partition：自动将数据分配给线程\n");
    printf("   好处：不需要手动计算索引！\n\n");
    printf("5. gemm()：执行实际计算，内部调用 Atom\n");
    printf("   好处：自动优化，支持不同硬件\n\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}

/*
 * 编译命令：
 *
 *   nvcc -std=c++17 -arch=sm_70 \
 *        -I../../../../include \
 *        atom_concept_simple.cu \
 *        -o atom_concept_simple
 *
 * 运行：
 *
 *   ./atom_concept_simple
 *
 *
 * 学习建议：
 *
 * 1. 先运行这个程序，理解概念输出
 * 2. 然后阅读代码，看 CuTe API 的使用
 * 3. 尝试修改参数（如线程布局）观察变化
 * 4. 最后看完整的 cuda_vs_cute_comparison.cu 对比示例
 */
