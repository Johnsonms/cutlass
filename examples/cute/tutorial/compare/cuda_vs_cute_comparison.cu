/***************************************************************************************************
 * CUDA vs CuTe 对比示例
 * 目标：通过简单的矩阵乘法 C = A * B，理解 Atom、TiledMMA 等核心概念
 *
 * 本示例包含两个实现：
 * 1. 传统 CUDA 实现（手动索引计算）
 * 2. CuTe 实现（使用 Atom 和 TiledMMA）
 **************************************************************************************************/

#include <iostream>
#include <cuda_runtime.h>

// CuTe headers
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

// ═══════════════════════════════════════════════════════════════════════════════
// 方法 1：传统 CUDA 实现
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * 传统 CUDA 的问题：
 * 1. 手动计算线程索引 (tid, row, col)
 * 2. 手动计算全局内存地址 (A[row*K + k])
 * 3. 手动管理共享内存的排列
 * 4. 代码冗长，难以修改和优化
 * 5. 不同架构需要重写代码
 */

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_traditional_cuda(
    const float* A,  // [M, K]
    const float* B,  // [K, N]
    float* C,        // [M, N]
    int M, int N, int K)
{
    // ─────────────────────────────────────────────────────────────────
    // 步骤 1：手动计算线程和 Block 索引
    // ─────────────────────────────────────────────────────────────────
    int tid = threadIdx.x;
    int bx = blockIdx.x;  // Block 在 M 维度的位置
    int by = blockIdx.y;  // Block 在 N 维度的位置

    // 计算这个 Block 负责的输出 C 的起始位置
    int block_row_start = bx * BLOCK_M;
    int block_col_start = by * BLOCK_N;

    // ─────────────────────────────────────────────────────────────────
    // 步骤 2：手动分配共享内存
    // ─────────────────────────────────────────────────────────────────
    __shared__ float smem_A[BLOCK_M][BLOCK_K];
    __shared__ float smem_B[BLOCK_K][BLOCK_N];

    // ─────────────────────────────────────────────────────────────────
    // 步骤 3：手动计算每个线程负责的元素
    // ─────────────────────────────────────────────────────────────────
    // 假设每个线程负责输出 C 的一个元素
    int threads_per_block = blockDim.x;
    int total_elements_M = BLOCK_M;
    int total_elements_N = BLOCK_N;

    // 这里简化：假设 threads_per_block >= BLOCK_M * BLOCK_N
    // 实际代码需要更复杂的逻辑
    int thread_row = tid / BLOCK_N;
    int thread_col = tid % BLOCK_N;

    if (thread_row >= BLOCK_M) return;

    // ─────────────────────────────────────────────────────────────────
    // 步骤 4：手动计算全局内存索引并加载到共享内存
    // ─────────────────────────────────────────────────────────────────
    float accumulator = 0.0f;

    // 循环遍历 K 维度
    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; ++k_tile) {
        // 手动加载 A 的 tile 到共享内存
        // 每个线程需要负责加载多个元素（这里简化）
        int k_start = k_tile * BLOCK_K;

        // 加载 A: smem_A[BLOCK_M][BLOCK_K]
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += threads_per_block) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = block_row_start + row;
            int global_col = k_start + col;

            if (global_row < M && global_col < K) {
                smem_A[row][col] = A[global_row * K + global_col];  // 手动计算索引！
            } else {
                smem_A[row][col] = 0.0f;
            }
        }

        // 加载 B: smem_B[BLOCK_K][BLOCK_N]
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += threads_per_block) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_row = k_start + row;
            int global_col = block_col_start + col;

            if (global_row < K && global_col < N) {
                smem_B[row][col] = B[global_row * N + global_col];  // 手动计算索引！
            } else {
                smem_B[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // ─────────────────────────────────────────────────────────────────
        // 步骤 5：手动编写 FMA 循环（这就是 "Atom" 的概念）
        // ─────────────────────────────────────────────────────────────────
        // FMA = Fused Multiply-Add，即 a = a + b * c
        // 在传统 CUDA 中，这是一个基本的算术操作

        for (int k = 0; k < BLOCK_K; ++k) {
            // 这里的每个 FMA 操作就是一个 "Atom"（原子操作）
            accumulator += smem_A[thread_row][k] * smem_B[k][thread_col];
            //              ↑ 这就是最基础的计算单元 - Atom！
        }

        __syncthreads();
    }

    // ─────────────────────────────────────────────────────────────────
    // 步骤 6：手动写回结果到全局内存
    // ─────────────────────────────────────────────────────────────────
    int global_row = block_row_start + thread_row;
    int global_col = block_col_start + thread_col;

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = accumulator;  // 手动计算索引！
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// 方法 2：CuTe 实现
// ═══════════════════════════════════════════════════════════════════════════════

/*
 * CuTe 的优势：
 * 1. Layout 自动管理内存排列和索引
 * 2. Tensor 提供类型安全的多维数组访问
 * 3. Atom 封装基本计算单元（FMA、Tensor Core 等）
 * 4. TiledMMA 自动处理线程间的工作分配
 * 5. 代码简洁，容易修改和移植
 */

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_cute(
    const float* A,  // [M, K]
    const float* B,  // [K, N]
    float* C,        // [M, N]
    int M, int N, int K)
{
    using namespace cute;

    // ═══════════════════════════════════════════════════════════════════
    // 概念对比图：
    //
    //   传统 CUDA                    CuTe
    //   ───────────────────────────────────────────────────────────
    //   手动索引计算          →     Layout (描述内存排列)
    //   float* A               →     Tensor (指针 + Layout)
    //   tid, bx, by            →     自动通过 Tensor 分割处理
    //   单个 FMA 指令          →     Atom (FMA, Tensor Core 等)
    //   手动分配工作给线程     →     TiledMMA (自动线程分配)
    // ═══════════════════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────────────────
    // 步骤 1：定义 Layout（代替手动索引计算）
    // ─────────────────────────────────────────────────────────────────

    // 定义全局矩阵的 Layout
    // Layout<Shape, Stride> 描述了数据在内存中的排列方式
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));  // 行主序
    auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));  // 行主序（转置后）
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));  // 行主序

    // ─────────────────────────────────────────────────────────────────
    // 步骤 2：创建 Tensor（代替 float* 指针）
    // ─────────────────────────────────────────────────────────────────
    // Tensor = 指针 + Layout
    // 提供类型安全的多维访问，无需手动计算索引！

    Tensor mA = make_tensor(make_gmem_ptr(A), layout_A);  // 全局内存 A
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_B);  // 全局内存 B
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_C);  // 全局内存 C

    // ─────────────────────────────────────────────────────────────────
    // 步骤 3：定义 Block Tiler（自动处理 Block 分割）
    // ─────────────────────────────────────────────────────────────────
    auto cta_tiler = make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}, Int<BLOCK_K>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    // 使用 local_tile 自动切分，无需手动计算！
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLOCK_M, BLOCK_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLOCK_N, BLOCK_K, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLOCK_M, BLOCK_N)

    // ─────────────────────────────────────────────────────────────────
    // 步骤 4：定义共享内存 Tensor
    // ─────────────────────────────────────────────────────────────────
    __shared__ float smem_A[BLOCK_M * BLOCK_K];
    __shared__ float smem_B[BLOCK_N * BLOCK_K];

    auto smem_layout_A = make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{}));
    auto smem_layout_B = make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_K>{}));

    Tensor sA = make_tensor(make_smem_ptr(smem_A), smem_layout_A);  // (BLOCK_M, BLOCK_K)
    Tensor sB = make_tensor(make_smem_ptr(smem_B), smem_layout_B);  // (BLOCK_N, BLOCK_K)

    // ═══════════════════════════════════════════════════════════════════
    // 【关键概念 1】：Atom - 最基本的计算单元
    // ═══════════════════════════════════════════════════════════════════
    //
    // Atom 是 CuTe 中最基础的抽象，代表一个单独的硬件操作：
    //
    // 1. UniversalFMA<float> - 标量 FMA 指令
    //    对应 PTX: fma.f32 d, a, b, c  (d = a*b + c)
    //    这是最基本的 Atom，每个线程执行一个 FMA
    //
    // 2. SM80_16x8x16_F16F16F16F16_TN (Tensor Core Atom)
    //    对应一个 Tensor Core 的 MMA 指令
    //    一次处理 16x8x16 的矩阵块
    //
    // 3. Copy_Atom - 内存复制的基本单元
    //    如 UniversalCopy<uint128_t> - 一次复制 128 位
    //
    // Atom 的特点：
    // - 描述单个硬件指令的行为
    // - 定义输入/输出的形状（Shape）
    // - 不涉及多线程分配
    // ═══════════════════════════════════════════════════════════════════

    // 定义 MMA Atom：使用标量 FMA
    using MMA_Atom = UniversalFMA<float, float, float, float>;
    // 这个 Atom 表示：每次执行一个 float FMA 操作
    // 形状是 (1, 1, 1) - 一个线程处理一个元素

    // ═══════════════════════════════════════════════════════════════════
    // 【关键概念 2】：TiledMMA - Atom 的平铺
    // ═══════════════════════════════════════════════════════════════════
    //
    // TiledMMA = Atom 在多线程间的组织方式
    //
    // 回忆传统 CUDA：
    //   - 需要手动决定：哪个线程处理哪个元素
    //   - 需要手动编写：tid -> (row, col) 的映射
    //
    // TiledMMA 自动完成这些：
    //   - 输入：Atom（基本计算单元）
    //   - 输入：线程布局（多少线程，如何排列）
    //   - 输出：每个线程的工作分配
    //
    // 例子：
    //   make_tiled_mma(
    //       UniversalFMA<float>{},        // Atom: 单个 FMA
    //       Layout<Shape<_16, _8, _1>>{}  // 16x8 个线程，每个做 1 个 FMA
    //   )
    //
    //   结果：128 个线程，每个线程负责输出 C 的一个元素
    //         自动计算每个线程需要从 A、B 读取哪些数据
    // ═══════════════════════════════════════════════════════════════════

    // 创建 TiledMMA
    auto tiled_mma = make_tiled_mma(
        MMA_Atom{},                                    // 使用 FMA Atom
        Layout<Shape<_32, _8, _1>>{}                   // 线程布局：32x8 线程
    );
    //
    // 解释：
    // - MMA_Atom{} 是单个 FMA 操作
    // - Layout<Shape<_32, _8, _1>>{} 表示：
    //     * 32 个线程负责 M 维度 (对齐 M128，每线程处理 128/32=4 个元素)
    //     * 8 个线程负责 N 维度 (对齐 N256，每线程处理 256/8=32 个元素)
    //     * 1 表示 K 维度（FMA 是标量操作）
    // - 总共 32 * 8 = 256 个线程

    // ═══════════════════════════════════════════════════════════════════
    // 【关键概念 3】：ThrMMA - 单个线程的视角
    // ═══════════════════════════════════════════════════════════════════
    //
    // ThrMMA = 从 TiledMMA 中提取单个线程的信息
    //
    // TiledMMA：描述所有线程的整体行为
    // ThrMMA：描述 threadIdx.x 这个线程的具体工作
    //
    // thr_mma 包含：
    // - 这个线程在 MMA 中的位置（坐标）
    // - 这个线程需要处理的数据形状
    // - 如何从全局/共享内存分割数据
    // ═══════════════════════════════════════════════════════════════════

    // 获取当前线程的 MMA 信息
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    // thr_mma 现在包含了 threadIdx.x 这个线程的所有信息

    // ─────────────────────────────────────────────────────────────────
    // 步骤 5：分割数据给线程（自动！）
    // ─────────────────────────────────────────────────────────────────
    // partition_A/B/C：根据 ThrMMA 自动计算这个线程负责的数据
    // 无需手动计算 tid -> (row, col) 的映射！

    Tensor tCsA = thr_mma.partition_A(sA);  // 这个线程负责的 A 数据
    Tensor tCsB = thr_mma.partition_B(sB);  // 这个线程负责的 B 数据
    Tensor tCgC = thr_mma.partition_C(gC);  // 这个线程负责的 C 数据

    // 分配寄存器（累加器）
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // 寄存器中的累加器
    clear(tCrC);  // 初始化为 0

    // ─────────────────────────────────────────────────────────────────
    // 步骤 6：主循环 - 自动处理数据移动和计算
    // ─────────────────────────────────────────────────────────────────
    int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // 加载数据到共享内存（简化版，实际需要 TiledCopy）
        // 这里为了对比清晰，用简单的 copy
        copy(gA(_, _, k_tile), sA);
        copy(gB(_, _, k_tile), sB);

        __syncthreads();

        // ═════════════════════════════════════════════════════════════
        // 【核心计算】：gemm() 函数自动调用 Atom
        // ═════════════════════════════════════════════════════════════
        //
        // gemm(tiled_mma, tCsA, tCsB, tCrC) 会：
        // 1. 遍历 K 维度的所有元素
        // 2. 对每个元素调用 MMA_Atom 的 call() 方法
        // 3. MMA_Atom::call() 内部执行实际的 FMA 指令
        //
        // 相当于传统 CUDA 中的：
        //   for (int k = 0; k < BLOCK_K; ++k) {
        //       accumulator += A[...][k] * B[k][...];  // ← 这是 Atom！
        //   }
        //
        // 但 CuTe 自动处理了：
        // - 索引计算
        // - 边界检查
        // - 向量化（如果可能）
        // ═════════════════════════════════════════════════════════════

        gemm(tiled_mma, tCsA, tCsB, tCrC);
        //   ↑          ↑     ↑     ↑
        //   │          │     │     └─ 累加器（输入/输出）
        //   │          │     └─────── B 矩阵数据
        //   │          └───────────── A 矩阵数据
        //   └──────────────────────── 包含 Atom 信息的 TiledMMA

        __syncthreads();
    }

    // ─────────────────────────────────────────────────────────────────
    // 步骤 7：写回结果（自动处理索引）
    // ─────────────────────────────────────────────────────────────────
    copy(tCrC, tCgC);  // 从寄存器复制到全局内存，索引自动计算！
}


// ═══════════════════════════════════════════════════════════════════════════════
// 【概念总结】：Atom vs TiledMMA
// ═══════════════════════════════════════════════════════════════════════════════
/*
 *
 * 1. Atom（原子操作）
 *    ├─ 定义：最小的、不可分割的计算单元
 *    ├─ 例子：
 *    │   ├─ UniversalFMA<float>：一个标量 FMA 指令
 *    │   ├─ SM80_16x8x16_F16：一个 Tensor Core MMA 指令
 *    │   └─ UniversalCopy<uint128_t>：一次 128 位复制
 *    ├─ 特点：
 *    │   ├─ 描述单个硬件指令
 *    │   ├─ 定义操作的形状（如 16x8x16）
 *    │   └─ 不涉及线程分配
 *    └─ 对应传统 CUDA：
 *        └─ 单个算术运算：c += a * b
 *
 * 2. TiledMMA（平铺的 MMA）
 *    ├─ 定义：Atom 在多个线程间的组织和复制
 *    ├─ 创建：make_tiled_mma(Atom, ThreadLayout)
 *    ├─ 作用：
 *    │   ├─ 决定多少个线程参与计算
 *    │   ├─ 决定每个线程负责哪些数据
 *    │   └─ 生成高效的内存访问模式
 *    ├─ 对应传统 CUDA：
 *    │   └─ 手动编写的线程分配逻辑：
 *    │       int row = tid / N;
 *    │       int col = tid % N;
 *    └─ 优势：
 *        ├─ 自动处理线程分配
 *        ├─ 自动处理向量化
 *        ├─ 容易修改和调优
 *        └─ 跨架构可移植
 *
 * 3. ThrMMA（线程的 MMA）
 *    ├─ 定义：从 TiledMMA 提取单个线程的信息
 *    ├─ 获取：tiled_mma.get_slice(threadIdx.x)
 *    ├─ 包含：
 *    │   ├─ 线程在 MMA 中的坐标
 *    │   ├─ 线程负责的数据形状
 *    │   └─ partition_A/B/C 方法
 *    └─ 对应传统 CUDA：
 *        └─ 单个线程的视角：
 *            float my_data = A[my_row][my_col];
 *
 * 4. 层次关系
 *
 *    Hardware Instruction (硬件指令)
 *           ↓ 封装
 *    Atom (原子操作) ──────────────────┐
 *           ↓ 复制和平铺                │ 描述单个操作
 *    TiledMMA (整体视角) ───────────────┤
 *           ↓ 提取                      │ 描述多线程组织
 *    ThrMMA (单线程视角) ───────────────┘
 *           ↓ 使用
 *    partition_A/B/C (数据分割)
 *           ↓
 *    gemm() (实际计算)
 *
 * 5. 类比理解
 *
 *    传统 CUDA                    CuTe
 *    ─────────────────────────────────────────────────
 *    一个 CPU 核心               Atom
 *    整个 CPU (多核)             TiledMMA
 *    单个核心的工作              ThrMMA
 *    手动分配任务                partition_A/B/C
 *    执行任务                    gemm()
 *
 * 6. 进阶：Tensor Core 的例子
 *
 *    // Tensor Core Atom（一个 Tensor Core 指令）
 *    using TC_Atom = SM80_16x8x16_F16F16F16F16_TN;
 *    // 形状：M=16, N=8, K=16
 *    // 一次处理 16x8x16 的矩阵乘法
 *
 *    // TiledMMA：在 warp 中平铺 Tensor Core
 *    auto tiled_mma = make_tiled_mma(
 *        TC_Atom{},
 *        Layout<Shape<_2, _2, _1>>{}  // 2x2 个 Tensor Core
 *    );
 *    // 结果：4 个 Tensor Core，每个处理 16x8x16
 *    //       总共处理 (2*16) x (2*8) x 16 = 32x16x16
 *
 */

// ═══════════════════════════════════════════════════════════════════════════════
// 辅助函数：验证结果
// ═══════════════════════════════════════════════════════════════════════════════

void verify_results(const float* C_cuda, const float* C_cute, int M, int N, float tolerance = 1e-4f) {
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(C_cuda[i] - C_cute[i]);
        max_diff = std::max(max_diff, diff);

        if (diff > tolerance) {
            if (errors < 10) {  // 只打印前 10 个错误
                int row = i / N;
                int col = i % N;
                printf("Error at [%d,%d]: CUDA=%.6f, CuTe=%.6f, diff=%.6f\n",
                       row, col, C_cuda[i], C_cute[i], diff);
            }
            errors++;
        }
    }

    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("验证结果\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("总元素数: %d\n", M * N);
    printf("错误数: %d\n", errors);
    printf("最大差异: %.6f\n", max_diff);

    if (errors == 0) {
        printf("✓ 结果匹配！传统 CUDA 和 CuTe 实现结果一致。\n");
    } else {
        printf("✗ 结果不匹配！\n");
    }
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 主函数：运行两种实现并对比
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    // 问题规模 - 对齐 SM M128N256K16 指令
    const int M = 256;   // 2x M128
    const int N = 512;   // 2x N256
    const int K = 64;    // 4x K16

    const int BLOCK_M = 128;  // 对齐 M128
    const int BLOCK_N = 256;  // 对齐 N256
    const int BLOCK_K = 16;   // 对齐 K16

    printf("\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("CUDA vs CuTe 对比示例\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("矩阵规模: C[%d,%d] = A[%d,%d] * B[%d,%d]\n", M, N, M, K, K, N);
    printf("Block 大小: [%d, %d, %d]\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("════════════════════════════════════════════════════════════\n\n");

    // 分配主机内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_cuda = (float*)malloc(size_C);
    float* h_C_cute = (float*)malloc(size_C);

    // 初始化输入数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C_cuda, *d_C_cute;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_cuda, size_C);
    cudaMalloc(&d_C_cute, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 配置 kernel 启动参数
    dim3 grid(M / BLOCK_M, N / BLOCK_N);
    dim3 block(256);  // 256 线程 (32x8)，对齐 M128N256K16

    printf("运行传统 CUDA kernel...\n");
    gemm_traditional_cuda<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block>>>(
        d_A, d_B, d_C_cuda, M, N, K
    );
    cudaDeviceSynchronize();
    printf("✓ 传统 CUDA kernel 完成\n\n");

    printf("运行 CuTe kernel...\n");
    gemm_cute<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block>>>(
        d_A, d_B, d_C_cute, M, N, K
    );
    cudaDeviceSynchronize();
    printf("✓ CuTe kernel 完成\n\n");

    // 复制结果回主机
    cudaMemcpy(h_C_cuda, d_C_cuda, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cute, d_C_cute, size_C, cudaMemcpyDeviceToHost);

    // 验证结果
    verify_results(h_C_cuda, h_C_cute, M, N);

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C_cuda);
    free(h_C_cute);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_cuda);
    cudaFree(d_C_cute);

    printf("\n示例完成！\n\n");

    return 0;
}


// ═══════════════════════════════════════════════════════════════════════════════
// 编译和运行说明
// ═══════════════════════════════════════════════════════════════════════════════
/*
 *
 * 编译命令：
 *
 *   nvcc -std=c++17 -arch=sm_80 \
 *        -I../../../../include \
 *        cuda_vs_cute_comparison.cu \
 *        -o cuda_vs_cute_comparison
 *
 * 运行：
 *
 *   ./cuda_vs_cute_comparison
 *
 *
 * 调试建议：
 *
 * 1. 在两个 kernel 中添加 printf 语句，对比：
 *    - 线程索引的计算方式
 *    - 内存访问的索引
 *    - 中间计算结果
 *
 * 2. 使用 cuda-gdb 单步调试：
 *    - 观察传统 CUDA 中手动计算的索引
 *    - 观察 CuTe 中 Tensor 的自动索引
 *
 * 3. 使用 Nsight Compute 分析：
 *    - 对比两个 kernel 的性能
 *    - 观察内存访问模式
 *    - 观察指令吞吐量
 *
 * 4. 修改实验：
 *    - 尝试不同的 BLOCK_M, BLOCK_N, BLOCK_K
 *    - 尝试使用 Tensor Core Atom (需要 sm_80+)
 *    - 尝试添加 TiledCopy 优化数据加载
 *
 */
