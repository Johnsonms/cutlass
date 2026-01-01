#!/bin/bash
# Build and run with tensor debug prints enabled

echo "Building with DEBUG_PRINT_TENSORS enabled..."

nvcc -g -G -std=c++17 --expt-relaxed-constexpr -O0 \
  -DDEBUG_PRINT_TENSORS \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  -I/app/tensorrt_llm/cutlass/include \
  -I/app/tensorrt_llm/cutlass/tools/util/include \
  --generate-code=arch=compute_100a,code=sm_100a \
  --maxrregcount=64 \
  01_mma_sm100.cu -o 01_mma_sm100_debug_prints

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Running with tensor debug prints..."
    echo "========================================"
    ./01_mma_sm100_debug_prints
else
    echo "Build failed!"
    exit 1
fi
