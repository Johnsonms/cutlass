# Debugging CUDA Tensors in VSCode

This guide shows how to examine tensor data while debugging CUDA kernels in VSCode.

## Setup

1. **Build with debug symbols** (already done):
   ```bash
   BUILD_MODE=full-debug ./build_debug_01_05.sh 01
   ```

2. **Launch configuration** is already added to `.vscode/launch.json`:
   - Configuration name: "Debug Local 01_mma_sm100_debug"
   - Uses `cuda-gdb` as the debugger
   - Sets `CUDA_LAUNCH_BLOCKING=1` for synchronous execution

## Method 1: Examining Host Tensors (Easiest)

### In VSCode Debug View:
1. Press `F5` or select "Debug Local 01_mma_sm100_debug" from the debug dropdown
2. Set breakpoint at line 584 (after device-to-host copy):
   ```cpp
   thrust::host_vector<TypeD> host_D = device_D;  // <- Set breakpoint here
   ```
3. When breakpoint hits, examine in **Variables** panel:
   - `host_A` - Expand to see vector elements
   - `host_tensor_A` - CuTe tensor wrapper (shows layout info)
   - `host_D` - Result data

### In Debug Console (gdb commands):
```gdb
# Print first 10 elements of host_A
p host_A[0]@10

# Print specific element
p host_A[5]

# Print tensor shape information
p host_tensor_A.layout_

# Print raw pointer and first elements
p *host_A.data()@5
```

## Method 2: Examining Device Tensors

Device memory requires copying to host first. Add helper code:

```cpp
// In kernel or after kernel launch
__device__ void print_tensor_slice(Tensor const& tensor, int max_elements = 10) {
    if (thread0()) {
        for (int i = 0; i < min(max_elements, size(tensor)); ++i) {
            printf("tensor[%d] = %f\n", i, float(tensor(i)));
        }
    }
}
```

### Using cuda-gdb to copy device memory:

1. Break in kernel:
   ```cpp
   if (thread0()) {
       printf("Debug point\n");  // <- Set breakpoint here in kernel
   }
   ```

2. In Debug Console:
   ```gdb
   # Switch to CUDA thread
   cuda thread (0,0,0)

   # Print device variable (only works for small data)
   p tDrC

   # Copy device memory to host and examine
   cuda memcheck
   ```

## Method 3: Using printf in Kernels (Most Practical)

Add print statements in the kernel to examine specific values:

```cpp
// In gemm_device kernel at line ~310 (after loading accumulators)
if (thread0()) {
    // Print layout info
    print("tDrAcc shape: "); print(shape(tDrAcc)); print("\n");

    // Print first few values
    printf("First accumulator values:\n");
    for (int i = 0; i < 4; ++i) {
        printf("  tDrAcc(%d) = %f\n", i, float(tDrAcc(i)));
    }
}
```

Rebuild and run to see output.

## Method 4: Using Conditional Breakpoints

In VSCode, right-click on a breakpoint → Edit Breakpoint → Condition:

```cpp
// Break only for specific thread
threadIdx.x == 0 && blockIdx.x == 0

// Break only when value meets condition
host_A[0] > 100.0f
```

## Method 5: Examining CuTe Tensor Layouts

CuTe tensors have complex layouts. To understand them:

```gdb
# In Debug Console when stopped at breakpoint
p tCgA          # Print entire tensor structure
p shape(tCgA)   # Print shape
p stride(tCgA)  # Print strides
p size(tCgA)    # Print total size

# For nested structures, navigate:
p tCgA.layout_.shape_
p tCgA.layout_.stride_
p tCgA.engine_
```

## Method 6: Visualizing Tensor Data

### Export to file for analysis:
```cpp
// Add this after line 584 in main():
{
    std::ofstream out("tensor_dump.txt");
    for (int m = 0; m < 16; ++m) {  // First 16 rows
        for (int n = 0; n < 16; ++n) {  // First 16 cols
            out << host_tensor_D(m, n) << " ";
        }
        out << "\n";
    }
}
```

Then visualize with Python:
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('tensor_dump.txt')
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title('Tensor D (first 16x16)')
plt.show()
```

## Method 7: Using Watch Expressions

In VSCode Debug view, add to **Watch** panel:

```cpp
// Watch specific elements
host_A[0]
host_tensor_A.layout_.shape_
size(host_tensor_A)

// Watch computations
host_D[0] - host_reference_D[0]  // Error at position 0
```

## Practical Example: Debug Flow

1. **Set breakpoint** at line 275 (after loading to SMEM):
   ```cpp
   cooperative_copy<128>(threadIdx.x, tCgB(_,_,_,k_tile), tCsB);
   __syncthreads();  // <- Breakpoint here
   ```

2. **Check SMEM layout**:
   ```gdb
   p tCsA  # Shows SMEM tensor info
   p tCsB
   ```

3. **Set breakpoint** at line 316 (after TMEM load):
   ```cpp
   copy(tiled_t2r_copy, tDtAcc, tDrAcc);  // <- Breakpoint here
   ```

4. **Examine registers**:
   ```gdb
   p tDrAcc  # Accumulator in registers
   # Note: cuda-gdb can show register values for current thread
   ```

5. **Set breakpoint** at line 321 (after final computation):
   ```cpp
   copy(tDrC, tDgD);  // <- Breakpoint here
   ```

6. **Compare with reference**:
   ```gdb
   # After line 596
   p relative_error
   p host_tensor_D(0,0)
   p host_reference_tensor_D(0,0)
   ```

## Troubleshooting

### "Cannot access memory" errors
- Device memory isn't directly accessible in host debugger
- Use `printf` in kernel or copy to host first

### Breakpoints not hitting in kernel
- Ensure `CUDA_LAUNCH_BLOCKING=1` is set
- Check breakpoint is in device code path
- Try breaking at `thread0()` condition

### Complex tensor structures
- CuTe tensors are template-heavy
- Use `print(tensor)` in device code instead of gdb examination
- Focus on `layout_`, `engine_`, and data pointer

## Quick Reference Card

| Task | Method |
|------|--------|
| Print in kernel | `print(tensor); print("\n");` |
| Print values | `printf("val=%f\n", float(tensor(i)));` |
| Break in kernel | Add `if (thread0()) { }` with breakpoint |
| Examine host | Use Variables panel or `p var@count` |
| Check layout | `p tensor.layout_` in gdb |
| Export data | Write to file, analyze externally |
| Visual inspection | Use printf or export to Python/matplotlib |
