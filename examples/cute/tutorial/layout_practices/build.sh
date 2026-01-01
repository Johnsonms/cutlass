#!/bin/bash

# Standalone build script for CuTe Hello World with debug info
# This builds ONLY the hello_world example, independent of other CUTLASS examples

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUTLASS_ROOT="/app/tensorrt_llm/cutlass"

echo "Building CuTe Hello World (standalone) with debug info..."

cd "$SCRIPT_DIR"

# Direct nvcc compilation - completely independent
nvcc hello_world.cu -o hello_world \
  -I"${CUTLASS_ROOT}/include" \
  -std=c++17 \
  --expt-relaxed-constexpr \
  -g -G -O0 \
  -arch=sm_100

if [ $? -eq 0 ]; then
  echo ""
  echo "✓ Build complete with debug info!"
  echo "  Executable: ${SCRIPT_DIR}/hello_world"
  echo ""
  echo "Run with:"
  echo "  ./hello_world"
  echo ""
  echo "Debug with:"
  echo "  cuda-gdb ./hello_world"
else
  echo "✗ Build failed!"
  exit 1
fi
