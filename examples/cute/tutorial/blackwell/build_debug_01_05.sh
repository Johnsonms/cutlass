#!/bin/bash
# Debug Build Script for 01_mma_sm100.cu through 05_mma_tma_epi_sm100.cu
#
# IMPORTANT: This script addresses SM100 Blackwell compilation and runtime issues:
# 1. Uses sm_100a architecture (not sm_100) for tcgen05 instruction support
# 2. Uses -O2 by default to avoid stack overflow with CuTe templates in debug mode
# 3. Provides 3 build modes: debug-opt (default), release, full-debug
#
# Usage:
#   ./build_debug_01_05.sh 01              # Build example 01 with default mode
#   BUILD_MODE=release ./build_debug_01_05.sh all  # Build all in release mode

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# CUTLASS root directory
CUTLASS_ROOT="/app/tensorrt_llm/cutlass"

# Build mode selection (default: debug-opt)
# Note: SM100 feature macros are automatically enabled when using sm_100a
BUILD_MODE=${BUILD_MODE:-debug-opt}

case $BUILD_MODE in
    full-debug)
        echo -e "${YELLOW}Building in Full Debug Mode (may overflow stack)${NC}"
        # Full debug: -G -O0 (WARNING: May cause stack overflow with CuTe templates!)
        NVCC_FLAGS="-g -G -std=c++17 --expt-relaxed-constexpr -O0"
        STACK_FLAGS="--maxrregcount=64"
        MODE_DESC="Full Debug (may overflow stack)"
        ;;
    debug-opt)
        # Debug-optimized: -g -O2 (Recommended: debuggable without stack overflow)
        NVCC_FLAGS="-g -std=c++17 --expt-relaxed-constexpr -O2"
        STACK_FLAGS=""
        MODE_DESC="Debug-Optimized (recommended)"
        ;;
    release)
        # Release: -O3 (Maximum performance)
        NVCC_FLAGS="-std=c++17 --expt-relaxed-constexpr -O3"
        STACK_FLAGS=""
        MODE_DESC="Release"
        ;;
    *)
        echo -e "${RED}Unknown BUILD_MODE: $BUILD_MODE${NC}"
        exit 1
        ;;
esac

INCLUDE_FLAGS="-I${CUTLASS_ROOT}/include -I${CUTLASS_ROOT}/tools/util/include"
# CRITICAL: Must use sm_100a (not sm_100) for tcgen05 instruction support!
ARCH_FLAGS="--generate-code=arch=compute_100a,code=sm_100a"  # Blackwell (B200)
CUTLASS_FLAGS="-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CuTe Debug Build (01-05 MMA Examples)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Build Mode: ${MODE_DESC}${NC}"
echo -e "${YELLOW}Architecture: sm_100a (Blackwell with tcgen05 support)${NC}"
echo -e "${YELLOW}Compiler flags: ${NVCC_FLAGS}${NC}"
echo ""

# Function to build a single file
build_file() {
    local source=$1
    local output=$2
    local number=$3

    echo -e "${YELLOW}[$number] Building: ${source}${NC}"
    echo -e "    Output: ${output}"

    nvcc ${NVCC_FLAGS} ${CUTLASS_FLAGS} ${STACK_FLAGS} ${INCLUDE_FLAGS} ${ARCH_FLAGS} ${source} -o ${output}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}    ✓ Success${NC}"
        ls -lh ${output} | awk '{print "    Size: " $5}'
        echo ""
        return 0
    else
        echo -e "${RED}    ✗ Failed${NC}"
        echo ""
        return 1
    fi
}

# Track build results
SUCCESS_COUNT=0
FAIL_COUNT=0

# If argument provided, build specific file
if [ $# -gt 0 ]; then
    case $1 in
        01|1)
            build_file "01_mma_sm100.cu" "01_mma_sm100_debug" "01"
            ;;
        02|2)
            build_file "02_mma_tma_sm100.cu" "02_mma_tma_sm100_debug" "02"
            ;;
        03|3)
            build_file "03_mma_tma_multicast_sm100.cu" "03_mma_tma_multicast_sm100_debug" "03"
            ;;
        04|4)
            build_file "04_mma_tma_2sm_sm100.cu" "04_mma_tma_2sm_sm100_debug" "04"
            ;;
        05|5)
            build_file "05_mma_tma_epi_sm100.cu" "05_mma_tma_epi_sm100_debug" "05"
            ;;
        all)
            build_file "01_mma_sm100.cu" "01_mma_sm100_debug" "01"
            [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

            build_file "02_mma_tma_sm100.cu" "02_mma_tma_sm100_debug" "02"
            [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

            build_file "03_mma_tma_multicast_sm100.cu" "03_mma_tma_multicast_sm100_debug" "03"
            [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

            build_file "04_mma_tma_2sm_sm100.cu" "04_mma_tma_2sm_sm100_debug" "04"
            [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

            build_file "05_mma_tma_epi_sm100.cu" "05_mma_tma_epi_sm100_debug" "05"
            [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

            echo -e "${BLUE}========================================${NC}"
            echo -e "${GREEN}  Success: ${SUCCESS_COUNT}/5${NC}"
            [ $FAIL_COUNT -gt 0 ] && echo -e "${RED}  Failed:  ${FAIL_COUNT}/5${NC}"
            echo -e "${BLUE}========================================${NC}"
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage: $0 [01|02|03|04|05|all]"
            echo ""
            echo "Examples:"
            echo "  $0 01                    # Build only 01_mma_sm100.cu"
            echo "  $0 all                   # Build all 01-05"
            echo "  $0                       # Build all (default)"
            echo ""
            echo "Build Modes (set BUILD_MODE environment variable):"
            echo "  BUILD_MODE=debug-opt $0  # Debug with -O2 (default, recommended)"
            echo "  BUILD_MODE=release $0    # Release with -O3"
            echo "  BUILD_MODE=full-debug $0 # Full debug -G -O0 (may overflow stack!)"
            exit 1
            ;;
    esac
else
    # Build all by default
    echo -e "${BLUE}Building all examples (01-05)...${NC}"
    echo ""

    build_file "01_mma_sm100.cu" "01_mma_sm100_debug" "01"
    [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

    build_file "02_mma_tma_sm100.cu" "02_mma_tma_sm100_debug" "02"
    [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

    build_file "03_mma_tma_multicast_sm100.cu" "03_mma_tma_multicast_sm100_debug" "03"
    [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

    build_file "04_mma_tma_2sm_sm100.cu" "04_mma_tma_2sm_sm100_debug" "04"
    [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

    build_file "05_mma_tma_epi_sm100.cu" "05_mma_tma_epi_sm100_debug" "05"
    [ $? -eq 0 ] && ((SUCCESS_COUNT++)) || ((FAIL_COUNT++))

    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}  Build Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}  Success: ${SUCCESS_COUNT}/5${NC}"
    [ $FAIL_COUNT -gt 0 ] && echo -e "${RED}  Failed:  ${FAIL_COUNT}/5${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    if [ $SUCCESS_COUNT -gt 0 ]; then
        echo -e "${GREEN}Run examples:${NC}"
        [ -f "01_mma_sm100_debug" ] && echo "  ./01_mma_sm100_debug"
        [ -f "02_mma_tma_sm100_debug" ] && echo "  ./02_mma_tma_sm100_debug"
        [ -f "03_mma_tma_multicast_sm100_debug" ] && echo "  ./03_mma_tma_multicast_sm100_debug"
        [ -f "04_mma_tma_2sm_sm100_debug" ] && echo "  ./04_mma_tma_2sm_sm100_debug"
        [ -f "05_mma_tma_epi_sm100_debug" ] && echo "  ./05_mma_tma_epi_sm100_debug"
        echo ""
        echo -e "${YELLOW}Debug with cuda-gdb:${NC}"
        echo "  cuda-gdb ./01_mma_sm100_debug"
    fi
fi
