# CuTe Hello World

A **standalone** introduction to CUTLASS CuTe layouts - completely independent from other CUTLASS examples.

## What is CuTe?

CuTe (CUTLASS Unified Template Extensions) is a template library for defining and manipulating data layouts in GPU memory.

## Build Instructions (Standalone)

This example builds independently and doesn't require building the entire CUTLASS project:

```bash
cd /app/tensorrt_llm/cutlass/examples/cute/tutorial/layout_practices
chmod +x build.sh
./build.sh
```

Then run:
```bash
./hello_world
```

Or debug:
```bash
cuda-gdb ./hello_world
```

### Manual Compilation

```bash
cd /app/tensorrt_llm/cutlass/examples/cute/tutorial/layout_practices
nvcc hello_world.cu -o hello_world \
  -I/app/tensorrt_llm/cutlass/include \
  -std=c++17 \
  --expt-relaxed-constexpr \
  -g -G -O0 \
  -arch=sm_100
```

## What You'll Learn

- Creating 1D and 2D layouts
- Understanding row-major vs column-major layouts
- Coordinate-to-index mapping
- Basic CuTe API usage

## Next Steps

After running this example, explore:
- `../blackwell/00_layout_basics.cu` - More detailed layout concepts
- `../blackwell/00_tensor_basics.cu` - Working with tensors
- CUTLASS documentation at https://github.com/NVIDIA/cutlass
