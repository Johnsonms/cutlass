#!/bin/bash
# Script to build Blackwell tutorials with debug flags using existing Makefile

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Directories
CUTLASS_ROOT="/app/tensorrt_llm/cutlass"
BUILD_DIR="${CUTLASS_ROOT}/build"
BLACKWELL_BUILD="${BUILD_DIR}/examples/cute/tutorial/blackwell"
SOURCE_DIR="${CUTLASS_ROOT}/examples/cute/tutorial/blackwell"

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Blackwell CuTe Tutorial Builder (B200 SM100) ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${RED}Error: Build directory not found: ${BUILD_DIR}${NC}"
    echo -e "${YELLOW}Please run CMake configuration first or use cmake_build_debug.sh${NC}"
    exit 1
fi

# Parse arguments
BUILD_MODE="Release"
BUILD_TARGET=""
RECONFIGURE=0
VERBOSE=0
COPY_LOCAL=0
JOBS=$(nproc)

show_usage() {
    echo "Usage: $0 [options] [target]"
    echo ""
    echo "Build Modes:"
    echo "  --debug, -d       Build with debug flags (-g -G -O0)"
    echo "  --release, -r     Build with release flags (default: -O3)"
    echo "  --reldbg          Build with release + debug info (-O2 -g)"
    echo ""
    echo "Targets:"
    echo "  01, 02, 03, 04, 05   Build specific tutorial"
    echo "  all                   Build all SM100 tutorials (01-05)"
    echo "  clean                 Clean build artifacts"
    echo ""
    echo "Options:"
    echo "  --local           Copy binaries to source folder after build"
    echo "  --reconfig        Reconfigure CMake before building"
    echo "  -v, --verbose     Verbose build output"
    echo "  -j N              Use N parallel jobs (default: $(nproc))"
    echo ""
    echo "Examples:"
    echo "  $0 --debug 01           # Debug build of tutorial 01"
    echo "  $0 -d all               # Debug build all tutorials"
    echo "  $0 --debug --local 01   # Build and copy to source folder"
    echo "  $0 --release 02         # Release build of tutorial 02"
    echo "  $0 --reconfig --debug   # Reconfigure and rebuild all in debug"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            ;;
        -d|--debug)
            BUILD_MODE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_MODE="Release"
            shift
            ;;
        --reldbg)
            BUILD_MODE="RelWithDebInfo"
            shift
            ;;
        --local)
            COPY_LOCAL=1
            shift
            ;;
        --reconfig)
            RECONFIGURE=1
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        01|1)
            BUILD_TARGET="cute_tutorial_01_mma_sm100"
            shift
            ;;
        02|2)
            BUILD_TARGET="cute_tutorial_02_mma_tma_sm100"
            shift
            ;;
        03|3)
            BUILD_TARGET="cute_tutorial_03_mma_tma_multicast_sm100"
            shift
            ;;
        04|4)
            BUILD_TARGET="cute_tutorial_04_mma_tma_2sm_sm100"
            shift
            ;;
        05|5)
            BUILD_TARGET="cute_tutorial_05_mma_tma_epi_sm100"
            shift
            ;;
        all)
            BUILD_TARGET="all"
            shift
            ;;
        clean)
            echo -e "${YELLOW}Cleaning Blackwell tutorial build artifacts...${NC}"
            cd "${BLACKWELL_BUILD}"
            make clean
            echo -e "${GREEN}✓ Clean complete${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            show_usage
            ;;
    esac
done

# Show build configuration
echo -e "${CYAN}Build Configuration:${NC}"
echo -e "  Mode: ${YELLOW}${BUILD_MODE}${NC}"
case $BUILD_MODE in
    Debug)
        echo -e "  Flags: ${YELLOW}-g -G -O0${NC} (full debug, no optimization)"
        ;;
    Release)
        echo -e "  Flags: ${YELLOW}-O3 -DNDEBUG${NC} (optimized, no debug)"
        ;;
    RelWithDebInfo)
        echo -e "  Flags: ${YELLOW}-O2 -g -DNDEBUG${NC} (optimized + debug info)"
        ;;
esac
echo -e "  Build dir: ${BUILD_DIR}"
echo -e "  Parallel jobs: ${JOBS}"
echo ""

# Check current build mode
CURRENT_MODE=$(grep "CMAKE_BUILD_TYPE:STRING=" "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2)
echo -e "${CYAN}Current build mode: ${YELLOW}${CURRENT_MODE}${NC}"

# Reconfigure if needed
if [ "${CURRENT_MODE}" != "${BUILD_MODE}" ] || [ ${RECONFIGURE} -eq 1 ]; then
    echo -e "${YELLOW}Reconfiguring CMake to ${BUILD_MODE} mode...${NC}"
    cd "${BUILD_DIR}"

    # Set CUDA debug flags based on mode
    if [ "${BUILD_MODE}" = "Debug" ]; then
        CUDA_DEBUG_FLAGS="-g -G -O0"
    else
        CUDA_DEBUG_FLAGS=""
    fi

    cmake "${CUTLASS_ROOT}" \
        -DCMAKE_BUILD_TYPE="${BUILD_MODE}" \
        -DCMAKE_CUDA_FLAGS="${CUDA_DEBUG_FLAGS}" \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Reconfigured to ${BUILD_MODE} mode${NC}"
    else
        echo -e "${RED}✗ Reconfiguration failed${NC}"
        exit 1
    fi
    echo ""
fi

# Determine targets to build
if [ -z "$BUILD_TARGET" ]; then
    BUILD_TARGET="cute_tutorial_01_mma_sm100 cute_tutorial_02_mma_tma_sm100 cute_tutorial_03_mma_tma_multicast_sm100 cute_tutorial_04_mma_tma_2sm_sm100 cute_tutorial_05_mma_tma_epi_sm100"
    echo -e "${YELLOW}Building all SM100 tutorials (01-05)${NC}"
elif [ "$BUILD_TARGET" = "all" ]; then
    BUILD_TARGET="cute_tutorial_01_mma_sm100 cute_tutorial_02_mma_tma_sm100 cute_tutorial_03_mma_tma_multicast_sm100 cute_tutorial_04_mma_tma_2sm_sm100 cute_tutorial_05_mma_tma_epi_sm100"
fi

echo ""

# Build
cd "${BLACKWELL_BUILD}"

SUCCESS_COUNT=0
FAIL_COUNT=0

MAKE_FLAGS="-j${JOBS}"
[ ${VERBOSE} -eq 1 ] && MAKE_FLAGS="${MAKE_FLAGS} VERBOSE=1"

for target in ${BUILD_TARGET}; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Building: ${CYAN}${target}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    START_TIME=$(date +%s)

    if [ ${VERBOSE} -eq 1 ]; then
        make ${MAKE_FLAGS} ${target}
    else
        make ${MAKE_FLAGS} ${target} 2>&1 | grep -E "(Building|Linking|error|warning|✓|✗)"
    fi

    BUILD_STATUS=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [ ${BUILD_STATUS} -eq 0 ]; then
        echo -e "${GREEN}✓ ${target} built successfully${NC} (${ELAPSED}s)"

        # Show file info
        if [ -f "${target}" ]; then
            SIZE=$(ls -lh "${target}" | awk '{print $5}')
            echo -e "  ${CYAN}→${NC} Size: ${SIZE}"
            echo -e "  ${CYAN}→${NC} Location: ${BLACKWELL_BUILD}/${target}"

            # Copy to source directory if --local flag is set
            if [ ${COPY_LOCAL} -eq 1 ]; then
                cp "${target}" "${SOURCE_DIR}/"
                echo -e "  ${CYAN}→${NC} Copied to: ${SOURCE_DIR}/${target}"
            fi
        fi
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ ${target} build failed${NC}"
        ((FAIL_COUNT++))
    fi
    echo ""
done

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Build Summary                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}  Success: ${SUCCESS_COUNT}${NC}"
[ ${FAIL_COUNT} -gt 0 ] && echo -e "${RED}  Failed:  ${FAIL_COUNT}${NC}"
echo -e "${BLUE}──────────────────────────────────────────────────${NC}"
echo -e "  Mode: ${YELLOW}${BUILD_MODE}${NC}"
if [ ${COPY_LOCAL} -eq 1 ]; then
    echo -e "  Output: ${SOURCE_DIR}/ ${CYAN}(local copy)${NC}"
else
    echo -e "  Output: ${BLACKWELL_BUILD}/"
fi
echo ""

if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo -e "${GREEN}To run:${NC}"
    if [ ${COPY_LOCAL} -eq 1 ]; then
        echo -e "  cd ${SOURCE_DIR}"
        [ -f "${SOURCE_DIR}/cute_tutorial_01_mma_sm100" ] && echo -e "  ./cute_tutorial_01_mma_sm100"
    else
        echo -e "  cd ${BLACKWELL_BUILD}"
        [ -f "${BLACKWELL_BUILD}/cute_tutorial_01_mma_sm100" ] && echo -e "  ./cute_tutorial_01_mma_sm100"
    fi
    echo ""

    if [ "${BUILD_MODE}" = "Debug" ]; then
        echo -e "${YELLOW}To debug with cuda-gdb:${NC}"
        if [ ${COPY_LOCAL} -eq 1 ]; then
            echo -e "  cd ${SOURCE_DIR}"
            echo -e "  cuda-gdb ./cute_tutorial_01_mma_sm100"
        else
            echo -e "  cd ${BLACKWELL_BUILD}"
            echo -e "  cuda-gdb ./cute_tutorial_01_mma_sm100"
        fi
    fi
fi
