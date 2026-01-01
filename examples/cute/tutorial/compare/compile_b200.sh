#!/bin/bash

#######################################################################
# B200 (Blackwell/SM100) 编译脚本
# 用于编译 CuTe Atom 教程示例
#######################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
ARCH="sm_100"  # B200 架构
CXX_STD="c++17"

# 自动检测 CUTLASS include 路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 尝试多个可能的路径
if [ -d "${SCRIPT_DIR}/../../../../include/cute" ]; then
    INCLUDE_DIR="../../../../include"  # 在 compare 子目录
elif [ -d "${SCRIPT_DIR}/../../../include/cute" ]; then
    INCLUDE_DIR="../../../include"      # 在 tutorial 目录
elif [ -d "/app/tensorrt_llm/cutlass/include/cute" ]; then
    INCLUDE_DIR="/app/tensorrt_llm/cutlass/include"  # 绝对路径
else
    echo -e "${RED}错误: 找不到 CuTe 头文件${NC}"
    echo "请检查 CUTLASS 安装路径"
    exit 1
fi

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  B200 (Blackwell/SM100) CuTe 教程编译脚本${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

#######################################################################
# 函数：编译单个文件
#######################################################################
compile_file() {
    local source_file=$1
    local output_name=$2
    local debug_mode=$3

    local debug_flags=""
    if [ "$debug_mode" == "debug" ]; then
        debug_flags="-g -G -lineinfo"
        output_name="${output_name}_debug"
    fi

    echo -e "${YELLOW}正在编译: ${source_file}${NC}"
    echo -e "  输出: ${output_name}"
    echo -e "  架构: ${ARCH}"
    if [ ! -z "$debug_flags" ]; then
        echo -e "  模式: 调试模式 (debug)"
    else
        echo -e "  模式: 发布模式 (release)"
    fi

    nvcc -std=${CXX_STD} \
         -arch=${ARCH} \
         ${debug_flags} \
         -I${INCLUDE_DIR} \
         ${source_file} \
         -o ${output_name}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 编译成功: ${output_name}${NC}"
    else
        echo -e "${RED}✗ 编译失败: ${source_file}${NC}"
        exit 1
    fi
    echo ""
}

#######################################################################
# 主程序
#######################################################################

# 检查 CUDA 环境
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}错误: 找不到 nvcc 编译器${NC}"
    echo "请确保 CUDA 已安装并且在 PATH 中"
    exit 1
fi

echo "CUDA 编译器: $(which nvcc)"
echo "CUDA 版本: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo ""

# 解析命令行参数
MODE="release"
TARGET="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            MODE="debug"
            shift
            ;;
        -r|--release)
            MODE="release"
            shift
            ;;
        simple|concept)
            TARGET="simple"
            shift
            ;;
        compare|comparison)
            TARGET="compare"
            shift
            ;;
        all)
            TARGET="all"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项] [目标]"
            echo ""
            echo "选项:"
            echo "  -d, --debug     编译调试版本 (包含 -g -G 标志)"
            echo "  -r, --release   编译发布版本 (默认)"
            echo ""
            echo "目标:"
            echo "  simple          只编译 atom_concept_simple.cu"
            echo "  compare         只编译 cuda_vs_cute_comparison.cu"
            echo "  all             编译所有文件 (默认)"
            echo ""
            echo "示例:"
            echo "  $0                    # 编译所有文件 (发布模式)"
            echo "  $0 -d                 # 编译所有文件 (调试模式)"
            echo "  $0 -d simple          # 只编译 simple (调试模式)"
            echo "  $0 compare            # 只编译 compare (发布模式)"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "编译模式: ${MODE}"
echo "编译目标: ${TARGET}"
echo ""

#######################################################################
# 开始编译
#######################################################################

# 编译 atom_concept_simple.cu
if [ "$TARGET" == "all" ] || [ "$TARGET" == "simple" ]; then
    if [ -f "atom_concept_simple.cu" ]; then
        compile_file "atom_concept_simple.cu" "atom_concept_simple" "$MODE"
    else
        echo -e "${RED}错误: 找不到文件 atom_concept_simple.cu${NC}"
    fi
fi

# 编译 cuda_vs_cute_comparison.cu
if [ "$TARGET" == "all" ] || [ "$TARGET" == "compare" ]; then
    if [ -f "cuda_vs_cute_comparison.cu" ]; then
        compile_file "cuda_vs_cute_comparison.cu" "cuda_vs_cute_comparison" "$MODE"
    else
        echo -e "${RED}错误: 找不到文件 cuda_vs_cute_comparison.cu${NC}"
    fi
fi

#######################################################################
# 完成
#######################################################################

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}编译完成！${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# 列出生成的可执行文件
echo "生成的文件:"
if [ "$MODE" == "debug" ]; then
    ls -lh *_debug 2>/dev/null || echo "  (无)"
else
    ls -lh atom_concept_simple cuda_vs_cute_comparison 2>/dev/null || echo "  (无)"
fi

echo ""
echo "运行示例:"
if [ "$MODE" == "debug" ]; then
    echo "  ./atom_concept_simple_debug"
    echo "  ./cuda_vs_cute_comparison_debug"
    echo ""
    echo "使用 cuda-gdb 调试:"
    echo "  cuda-gdb ./atom_concept_simple_debug"
    echo "  cuda-gdb ./cuda_vs_cute_comparison_debug"
else
    echo "  ./atom_concept_simple"
    echo "  ./cuda_vs_cute_comparison"
fi
echo ""
