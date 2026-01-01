# Practical CUDA Tensor Debugging Example

This is a step-by-step walkthrough of debugging tensor values in the 01_mma_sm100.cu example.

## Quick Start: Three Ways to Debug

### Method 1: Printf Debugging (Fastest)
```bash
# Build and run with debug prints
./debug_with_prints.sh
```

This will show tensor values during kernel execution.

### Method 2: VSCode Interactive Debugging
1. Open `01_mma_sm100.cu` in VSCode
2. Press `F5` and select "Debug Local 01_mma_sm100_debug"
3. Set breakpoints as shown below

### Method 3: Command-line cuda-gdb
```bash
cuda-gdb ./01_mma_sm100_debug
(cuda-gdb) break 01_mma_sm100.cu:584
(cuda-gdb) run
```

---

## Scenario 1: Examining Input Tensors (Host Memory)

### Goal: Check if input data is correctly initialized

**Location**: Line 584 in `main()`
```cpp
thrust::host_vector<TypeD> host_D = device_D;  // <- Set breakpoint here
```

### Steps in VSCode:
1. Set breakpoint at line 584
2. Press `F5` to start debugging
3. When breakpoint hits, open **VARIABLES** panel
4. Expand:
   - `host_A` → `_M_impl` → `_M_start` to see raw array
   - `host_tensor_A` → to see CuTe layout info
   - `host_tensor_A.layout_` → `shape_` to see dimensions

### In Debug Console (gdb):
```gdb
# Print first 10 elements of matrix A
p host_A._M_impl._M_start[0]@10

# Print specific element at row=5, col=10
# For layout (M,K) with stride (K,1), index = row*K + col
p host_A._M_impl._M_start[5*256 + 10]

# Print tensor metadata
p host_tensor_A.layout_.shape_
p host_tensor_A.layout_.stride_

# Calculate and print size
p size(host_tensor_A)
```

### Expected Output:
```
$1 = {1.2, 45.7, 23.4, ...}  # Random values between -1 and 1
```

---

## Scenario 2: Examining Device Tensors Before Kernel

### Goal: Verify data was copied to device correctly

**Location**: Line 578 (just before kernel call)

### Add temporary host copy:
```cpp
// At line 577, before gemm_host_f16xf16_f32_f32_tnt call
{
  thrust::host_vector<TypeA> verify_A = device_A;
  printf("Device A first 5 elements: ");
  for (int i = 0; i < 5; ++i) {
    printf("%.3f ", float(verify_A[i]));
  }
  printf("\n");
}
```

Rebuild and run to see device data before kernel.

---

## Scenario 3: Examining Tensors Inside Kernel

### Goal: Check intermediate computation values

**Location**: Lines 319-332 (accumulator values in kernel)

### Using Debug Prints (Recommended):
```bash
# Build with DEBUG_PRINT_TENSORS flag
./debug_with_prints.sh
```

Output will show:
```
=== DEBUG: Accumulator Values (Thread 0) ===
tDrAcc shape: (_64,_4,_4)
First 4 accumulator values:
  tDrAcc[0] = 12345.678900
  tDrAcc[1] = 23456.789012
  ...
```

### Using cuda-gdb (Advanced):
```bash
cuda-gdb ./01_mma_sm100_debug
```

```gdb
# Set breakpoint in kernel
(cuda-gdb) break 01_mma_sm100.cu:325

# Run program
(cuda-gdb) run

# When kernel hits breakpoint, switch to CUDA thread (0,0,0)
(cuda-gdb) cuda thread (0,0,0)

# Try to print device variables (limited support)
(cuda-gdb) info locals
(cuda-gdb) print k_tile

# Note: Complex types like tensors may not print correctly
# That's why printf is more reliable
```

---

## Scenario 4: Examining Shared Memory

### Goal: Check if data was loaded to SMEM correctly

**Location**: Line 280 (after cooperative_copy to SMEM)

### Add debug code:
```cpp
// At line 281, after __syncthreads()
#ifdef DEBUG_PRINT_TENSORS
if (thread0()) {
    printf("\n=== SMEM Data (k_tile=%d) ===\n", k_tile);
    printf("tCsA shape: "); print(shape(tCsA)); printf("\n");
    // Can't directly print SMEM contents, but can verify layout
    printf("tCsA layout verified\n");
}
#endif
```

---

## Scenario 5: Comparing Results with Reference

### Goal: Find where computation differs from expected

**Location**: Line 604 (after comparing with reference)

### In Debug Console:
```gdb
# Break after comparison
(cuda-gdb) break 01_mma_sm100.cu:604

# Examine error values
(cuda-gdb) p relative_error

# If error > 0, find specific mismatches
(cuda-gdb) p host_tensor_D(0,0)
(cuda-gdb) p host_reference_tensor_D(0,0)
(cuda-gdb) p host_tensor_D(0,0) - host_reference_tensor_D(0,0)

# Loop through to find max error location
(cuda-gdb) call print_max_error(host_tensor_D, host_reference_tensor_D)
```

### Add helper function for detailed error analysis:
```cpp
// Add before main()
void print_first_mismatch(auto const& result, auto const& reference, float threshold = 1e-5f) {
    for (int m = 0; m < size<0>(result); ++m) {
        for (int n = 0; n < size<1>(result); ++n) {
            float diff = abs(result(m,n) - reference(m,n));
            if (diff > threshold) {
                printf("First mismatch at (%d,%d): result=%.6f, reference=%.6f, diff=%.6f\n",
                       m, n, float(result(m,n)), float(reference(m,n)), diff);
                return;
            }
        }
    }
    printf("No mismatches found above threshold %.6f\n", threshold);
}

// Call at line 604:
// print_first_mismatch(host_tensor_D, host_reference_tensor_D);
```

---

## Scenario 6: Watching Tensor Values Change

### Goal: Track how a specific element changes through computation

### Using VSCode Watch Panel:

1. Set breakpoint at line 309 (after loading C)
2. Add to **WATCH** panel:
   ```
   host_tensor_C(0,0)
   ```

3. Step through code (F10) and watch value

4. Add more watches:
   ```
   host_tensor_D(0,0)
   host_reference_tensor_D(0,0)
   host_tensor_D(0,0) - host_reference_tensor_D(0,0)
   ```

### Using Conditional Breakpoint:

Right-click breakpoint → "Edit Breakpoint" → Add condition:
```cpp
host_tensor_D(0,0) != host_reference_tensor_D(0,0)
```

This breaks only when values differ.

---

## Scenario 7: Memory Layout Visualization

### Goal: Understand tensor memory layout

**Location**: Lines 550-558 (tensor creation)

### In Debug Console:
```gdb
# Break after tensor creation
(cuda-gdb) break 01_mma_sm100.cu:558

# Examine layout
(cuda-gdb) p layout_A
(cuda-gdb) p shape(layout_A)    # Should show (512, 256)
(cuda-gdb) p stride(layout_A)   # Should show (256, 1) for K-major

# Calculate physical address of element (m,k)
# address = base + m*stride[0] + k*stride[1]
# For element (5, 10):
(cuda-gdb) p &host_A._M_impl._M_start[5*256 + 10*1]
```

### Export layout for visualization:
```cpp
// Add at line 560
printf("Layout A: shape=(%d,%d) stride=(%d,%d)\n",
       int(size<0>(layout_A)), int(size<1>(layout_A)),
       int(stride<0>(layout_A)), int(stride<1>(layout_A)));
```

---

## Common Debugging Patterns

### Pattern 1: Sanity Check Values
```cpp
if (thread0()) {
    // Check for NaN or Inf
    for (int i = 0; i < min(10, size(tensor)); ++i) {
        float val = float(tensor(i));
        if (isnan(val) || isinf(val)) {
            printf("ERROR: Invalid value at %d: %f\n", i, val);
        }
    }
}
```

### Pattern 2: Range Check
```cpp
if (thread0()) {
    float min_val = 1e9, max_val = -1e9;
    for (int i = 0; i < size(tensor); ++i) {
        float val = float(tensor(i));
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    printf("Tensor range: [%.6f, %.6f]\n", min_val, max_val);
}
```

### Pattern 3: Statistical Summary
```cpp
if (thread0()) {
    float sum = 0, sum_sq = 0;
    int count = size(tensor);
    for (int i = 0; i < count; ++i) {
        float val = float(tensor(i));
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / count;
    float variance = sum_sq / count - mean * mean;
    printf("Tensor stats: mean=%.6f, variance=%.6f\n", mean, variance);
}
```

---

## Tips and Tricks

### 1. Use thread0() for Prints
Always wrap kernel printf in `if (thread0())` to avoid duplicate output from multiple threads.

### 2. Reduce Problem Size
When debugging, use smaller tensors:
```bash
./01_mma_sm100_debug 128 256 64  # Instead of 512 1024 256
```

### 3. Isolate the Problem
Comment out parts of the kernel to isolate issues:
```cpp
// Temporarily disable MMA to test just data loading
// gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
```

### 4. Save Intermediate Results
```cpp
// Save SMEM state to global memory for inspection
static __device__ float debug_buffer[1024];
if (thread0()) {
    for (int i = 0; i < min(1024, size(tCsA)); ++i) {
        debug_buffer[i] = float(tCsA(i));
    }
}
// Copy debug_buffer to host after kernel
```

### 5. Use Color in Output
```cpp
#define RED "\033[31m"
#define GREEN "\033[32m"
#define RESET "\033[0m"

if (error > threshold) {
    printf(RED "ERROR: Large error detected\n" RESET);
} else {
    printf(GREEN "PASS\n" RESET);
}
```

---

## Debugging Checklist

When tensor values are wrong, check:

- [ ] Input data initialized correctly?
- [ ] Device memory allocated and copied?
- [ ] Tensor layouts match expected (shape, stride)?
- [ ] Kernel launched with correct grid/block dimensions?
- [ ] Shared memory size sufficient?
- [ ] Stack size adequate (especially in debug mode)?
- [ ] Synchronization points correct?
- [ ] Index calculations correct?
- [ ] Data types compatible (half_t vs float)?
- [ ] Alpha/beta scaling applied correctly?

---

## Summary

| Method | Best For | Complexity |
|--------|----------|------------|
| Printf in kernel | Quick checks, actual values | Easy |
| VSCode breakpoints | Host-side inspection | Easy |
| cuda-gdb | Complex device state | Hard |
| Export to file | Visualization, analysis | Medium |
| Watch expressions | Tracking changes | Easy |

**Recommendation**: Start with printf debugging, then use VSCode for host-side inspection if needed.
