#!/bin/bash
# Build tiled_copy.cu with verbose debug output

set -e

CUTLASS_ROOT="/app/tensorrt_llm/cutlass"
TENSORRT_ROOT="/app/tensorrt_llm/tensorrt_llm/deep_gemm"
TUTORIAL_DIR="${CUTLASS_ROOT}/examples/cute/tutorial"
OUTPUT="tiled_copy_verbose"
HEADER_FILE="${TENSORRT_ROOT}/include/cute/atom/copy_atom.hpp"

echo "================================================"
echo "Building tiled_copy with debug prints enabled"
echo "================================================"

# Step 1: Backup and patch the header file
if [ ! -f "${HEADER_FILE}.original" ]; then
    echo "Backing up ${HEADER_FILE}..."
    cp "${HEADER_FILE}" "${HEADER_FILE}.original"
fi

echo "Patching header to enable debug prints..."
sed -i '508s/#if 0/#ifdef CUTE_DEBUG_COPY_ATOM/' "${HEADER_FILE}"

# Step 2: Compile with debug macro defined
echo "Compiling with -DCUTE_DEBUG_COPY_ATOM..."

nvcc -std=c++17 \
     -O0 -g -G \
     -DCUTE_DEBUG_COPY_ATOM \
     -DCUTE_DEBUG_PARTITION \
     -I${TENSORRT_ROOT}/include \
     -I${CUTLASS_ROOT}/include \
     -I${CUTLASS_ROOT}/tools/util/include \
     -I${TUTORIAL_DIR} \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -gencode=arch=compute_100,code=sm_100 \
     ${TUTORIAL_DIR}/tiled_copy.cu \
     -o ${OUTPUT}

echo ""
echo "✓ Build complete: ${OUTPUT}"
echo ""
echo "Run with:"
echo "  ./${OUTPUT}"
echo ""
echo "To restore original header:"
echo "  cp ${HEADER_FILE}.original ${HEADER_FILE}"
