## [CUTLASS] SM90 cp.async + GMMA: add async-proxy fence before wgmma reads cp.async-filled smem

Fixes #3721.

### Problem
The SM90 `cp.async` + GMMA warp-specialized mainloops (`sm90_mma_multistage_gmma_ss_warpspecialized.hpp`, and the B operand in `sm90_mma_multistage_gmma_rs_warpspecialized.hpp`) fill shared memory with non-bulk `cp.async`, plus `clear()` for residue K-tiles. Both write through the generic proxy. `wgmma.mma_async` then reads that memory through the async proxy. PTX (§9.7.17.7.1) requires an async-proxy fence between the two, and nothing on this path provides one.

### Change
Add `cutlass::arch::fence_view_async_shared()` (`fence.proxy.async.shared::cta`) in the consumer, right after each wait that hands a stage to the MMA warps and before the stage is read by wgmma. That's 3 places in the SS collective and 3 in the RS collective. In the RS collective the fence also runs when `try_wait` already succeeded, because that `try_wait` is the acquire.

**Why the consumer side?** The alternative is for each producer thread to wait for its own `cp.async` to finish (`cp.async.wait_all`), fence, and then do a plain arrive. That gives up the overlap that `cp.async.mbarrier.arrive` provides. With the consumer-side fence, the barrier wait synchronizes with the producer, and the proxy fence then orders the reading thread's later async-proxy reads. **Maintainers: please confirm this placement is sufficient under your memory model, or say if you prefer the producer-side version.** Happy to switch.

### Testing (8x H100 80GB, CUDA 13.3)
- The missing fence does **not** cause wrong results on H100: 0 mismatches over thousands of runs without the fence. This change makes the code correct per the PTX spec rather than fixing an observed failure.
- After the change, the SASS has `FENCE.VIEW.ASYNC.S` on these paths; before, the only such fences were in the epilogue.
- Exact-match correctness against a reference GEMM over 4 shapes, including K-residue cases (64x64x320, 1000x1032x1000, 4096³, 128x256x72): fp16 SS with WS, cooperative and ping-pong; tf32 RS with WS and cooperative; tf32 RS with transposed B. 0 mismatches.
- Unit tests: all 295 tests in 8 SM90 GEMM unit-test binaries pass on H100 (Release, sm_90a): `tensorop_sm90` (96), `cluster_multicast_sm90` (92), `alignx_sm90` (77), `gmma_rs_warpspecialized_sm90` (16), `sm90_ptr_array` (4), `sm90_group_gemm` (4), `sm90_group_gemm_pingpong` (3), `sm90_blockwise` (3). These ran on a build containing both this change and the companion fix for #3722. `alignx_sm90` covers the `KernelCpAsyncWarpSpecialized{,Cooperative,Pingpong}` schedules in fp16, bf16, tf32 and s8.

Performance, release builds, best of 3 interleaved runs (TFLOPS):

| config | shape | before | after | Δ |
|---|---|---:|---:|---:|
| fp16 SS, WS | 2048x2048x512 | 150.8 | 147.6 | −2.2% |
| fp16 SS, WS | 4096³ | 370.3 | 353.0 | −4.7% |
| fp16 SS, WS | 8192³ | 391.8 | 384.3 | −1.9% |
| fp16 SS, cooperative | 4096³ | 394.0 | 389.8 | −1.1% |
| fp16 SS, ping-pong | 4096³ | 430.0 | 425.3 | −1.1% |
| tf32 RS, WS | 4096³ | 192.0 | 189.6 | −1.3% |
| tf32 RS, cooperative | 4096³ | 188.6 | 186.2 | −1.3% |
| tf32 RS (transposed B), WS | 4096³ | 87.4 | 87.7 | +0.3% |

The 8192³ results for the cooperative and ping-pong kernels were noisy (±5%) and are left out of the table.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
