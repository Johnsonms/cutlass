#!/bin/bash
# CMake Debug Build Script for Blackwell CuTe Tutorials

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
CUTLASS_ROOT="/app/tensorrt_llm/cutlass"
BUILD_DIR="${CUTLASS_ROOT}/build_debug"
SOURCE_DIR="${CUTLASS_ROOT}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CMake Debug Build for CuTe Tutorials${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
BUILD_TARGET=""
CLEAN_BUILD=0
CONFIGURE_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
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
            BUILD_TARGET="all_cute_blackwell"
            shift
            ;;
        clean)
            CLEAN_BUILD=1
            shift
            ;;
        configure)
            CONFIGURE_ONLY=1
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage: $0 [options] [target]"
            echo ""
            echo "Targets:"
            echo "  01, 02, 03, 04, 05  - Build specific tutorial"
            echo "  all                  - Build all SM100 tutorials"
            echo ""
            echo "Options:"
            echo "  clean      - Clean build directory before building"
            echo "  configure  - Only configure, don't build"
            echo ""
            echo "Examples:"
            echo "  $0 01              # Build tutorial 01"
            echo "  $0 clean all       # Clean build all"
            echo "  $0 configure       # Just configure CMake"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "${BUILD_DIR}"
    echo -e "${GREEN}✓ Clean complete${NC}"
    echo ""
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure CMake with debug flags
echo -e "${YELLOW}Configuring CMake...${NC}"
echo -e "  Build dir: ${BUILD_DIR}"
echo -e "  Build type: Debug"
echo -e "  CUDA arch: SM100 (compute_100)"
echo ""

cmake "${SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCUTLASS_NVCC_ARCHS="100a" \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUDA_ARCHS="100" \
    -DCMAKE_CUDA_FLAGS="-g -G -O0 --expt-relaxed-constexpr" \
    -DCMAKE_CUDA_FLAGS_DEBUG="-g -G -O0" \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configured successfully${NC}"
echo ""

# Exit if configure-only
if [ $CONFIGURE_ONLY -eq 1 ]; then
    echo -e "${BLUE}Configuration complete. To build, run:${NC}"
    echo -e "  cd ${BUILD_DIR}"
    echo -e "  make cute_tutorial_01_mma_sm100 -j\$(nproc)"
    exit 0
fi

# Determine what to build
if [ -z "$BUILD_TARGET" ]; then
    # Default: build all SM100 tutorials
    BUILD_TARGET="cute_tutorial_01_mma_sm100 cute_tutorial_02_mma_tma_sm100 cute_tutorial_03_mma_tma_multicast_sm100 cute_tutorial_04_mma_tma_2sm_sm100 cute_tutorial_05_mma_tma_epi_sm100"
    echo -e "${YELLOW}Building all SM100 tutorials (default)...${NC}"
else
    echo -e "${YELLOW}Building target: ${BUILD_TARGET}...${NC}"
fi

echo ""

# Build the targets
for target in $BUILD_TARGET; do
    echo -e "${BLUE}Building ${target}...${NC}"
    make $target -j$(nproc)

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${target} built successfully${NC}"

        # Find the executable
        EXE=$(find "${BUILD_DIR}" -type f -executable -name "${target}")
        if [ -n "$EXE" ]; then
            ls -lh "$EXE" | awk '{print "  Size: " $5 " at " $9}'
        fi
        echo ""
    else
        echo -e "${RED}✗ ${target} build failed${NC}"
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Executables location:${NC}"
echo -e "  ${BUILD_DIR}/examples/cute/tutorial/blackwell/"
echo ""
echo -e "${GREEN}To run:${NC}"
echo -e "  cd ${BUILD_DIR}/examples/cute/tutorial/blackwell"
echo -e "  ./cute_tutorial_01_mma_sm100"
echo ""
echo -e "${YELLOW}To debug with cuda-gdb:${NC}"
echo -e "  cd ${BUILD_DIR}/examples/cute/tutorial/blackwell"
echo -e "  cuda-gdb ./cute_tutorial_01_mma_sm100"
