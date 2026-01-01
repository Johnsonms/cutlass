#!/bin/bash
# CuTe Tutorial 简易构建脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# CUTLASS 根目录
CUTLASS_ROOT="/app/tensorrt_llm/cutlass"

# 编译选项
NVCC_FLAGS="-g -G -std=c++17 --expt-relaxed-constexpr"
INCLUDE_FLAGS="-I${CUTLASS_ROOT}/include -I${CUTLASS_ROOT}/tools/util/include"
# ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"  # Ampere
# ARCH_FLAGS="-gencode arch=compute_90,code=sm_90"  # Hopper
ARCH_FLAGS="-gencode arch=compute_100,code=sm_100"  # Blackwell (B200)

echo -e "${GREEN}=== CuTe Tutorial 构建脚本 ===${NC}"
echo ""

# 函数：编译单个文件
build_file() {
    local source=$1
    local output=$2

    echo -e "${YELLOW}编译: ${source} -> ${output}${NC}"
    nvcc ${NVCC_FLAGS} ${INCLUDE_FLAGS} ${ARCH_FLAGS} ${source} -o ${output}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 成功: ${output}${NC}"
        ls -lh ${output}
        echo ""
        return 0
    else
        echo -e "${RED}✗ 失败: ${source}${NC}"
        echo ""
        return 1
    fi
}

# 如果指定了参数，只编译指定的文件
if [ $# -gt 0 ]; then
    case $1 in
        layout|00)
            build_file "00_layout_basics.cu" "layout_basics"
            ;;
        tensor|01)
            build_file "00_tensor_basics.cu" "tensor_basics"
            ;;
        tile|02)
            build_file "00_tile_basics.cu" "tile_basics"
            ;;
        partition|03)
            build_file "00_partition_basics.cu" "partition_basics"
            ;;
        thread|mma-thread|04)
            build_file "00_mma_thread_concepts.cu" "mma_thread_concepts"
            ;;
        mma|05)
            build_file "01_mma_sm100.cu" "mma_sm100"
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "用法: ./b.sh [layout|tensor|tile|partition|thread|mma|all]"
            echo "  或: ./b.sh          (编译所有)"
            exit 1
            ;;
    esac
else
    # 编译所有基础教程
    echo "编译所有基础教程..."
    echo ""

    build_file "00_layout_basics.cu" "layout_basics"
    build_file "00_tensor_basics.cu" "tensor_basics"
    build_file "00_tile_basics.cu" "tile_basics"
    build_file "00_partition_basics.cu" "partition_basics"
    build_file "00_mma_thread_concepts.cu" "mma_thread_concepts"

    echo -e "${GREEN}=== 构建完成 ===${NC}"
    echo ""
    echo "运行示例："
    echo "  ./layout_basics"
    echo "  ./tensor_basics"
    echo "  ./tile_basics"
    echo "  ./partition_basics"
    echo "  ./mma_thread_concepts"
fi
