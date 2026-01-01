#!/bin/bash
# Quick debug build script for tiled_copy.cu

set -e

CUTLASS_ROOT="/app/tensorrt_llm/cutlass"
TUTORIAL_DIR="${CUTLASS_ROOT}/examples/cute/tutorial"
OUTPUT="tiled_copy_debug"

echo "Building tiled_copy.cu with debug symbols..."

nvcc -std=c++17 \
     -O0 -g -G \
     -I${CUTLASS_ROOT}/include \
     -I${CUTLASS_ROOT}/tools/util/include \
     -I${TUTORIAL_DIR} \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -gencode=arch=compute_100,code=sm_100 \
     ${TUTORIAL_DIR}/tiled_copy.cu \
     -o ${OUTPUT}

echo "✓ Build complete: ${OUTPUT}"
echo "Run with: ./${OUTPUT}"
echo "Debug with: cuda-gdb ./${OUTPUT}"
