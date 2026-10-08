## [CUTLASS] SM90 TMA GEMM: make the pipeline leader the TMA-issuing lane

Fixes #3722.

### Problem
In the SM90 TMA warp-specialized GEMM kernels, `PipelineTmaAsync`'s leader (`is_leader`) is set to `warp_group_thread_idx == 0`. However, the mainloop collectives call `producer_acquire` and issue the TMA loads only from the lane elected by `cute::elect_one_sync()`. The leader's `arrive_and_expect_tx` is the stage's only producer arrival, so this works only if `elect.sync` picks lane 0. PTX promises that the election is deterministic for a given mask, but not which lane it picks. If it picked any other lane, no lane would arrive with the expected bytes and the kernel would hang.

### Change
Derive the leader from the same election that picks the TMA issuer, as the SM100, SM120 and Hopper FMHA kernels already do:

- `sm90_gemm_tma_warpspecialized{,_cooperative,_pingpong}.hpp` and `sm90_gemm_array_tma_warpspecialized_{cooperative,pingpong}.hpp`: `is_leader = Producer && <mainloop warp> && lane_predicate`. Only the mainloop warp is eligible, so the `MainloopAux` warp, which also has the Producer role, never adds a second `arrive_and_expect_tx`.
- `sm90_mma_tma_gmma_ss.hpp` (non-warp-specialized `KernelTma`): `is_leader = warp_idx == 0 && elect_one_sync()`.
- `sm90_pipeline.hpp`: the debug-only `brkpt` in `producer_acquire` assumed the leader is lane 0. It now fires only if more than one lane in a warp is a leader, which is what its comment says it checks.
- Examples 63 (weight prefetch) and 88 (Hopper FMHA, `fmha_kernel_tma.hpp`) had the same pattern and get the same change.
- Removed the `warp_group_thread_idx` variables this change leaves unused.

All `elect_one_sync()` calls involved use the full mask, so PTX guarantees they elect the same lane. I checked that no SM90 TMA collective hard-codes lane 0 to issue its TMA loads. The collectives that call `producer_acquire` from all 32 lanes (mixed-input, FP8 blockwise) still work, because exactly one lane is still the leader.

### Testing (8x H100 80GB, CUDA 13.3)
- On current hardware `elect.sync` always picks lane 0, so this is not a behavior change today. A micro-test with 8,448 warps never elected a different lane.
- Release builds (`-DNDEBUG`): the mainloop machine code for WS, cooperative, ping-pong, `KernelTma` and the RS variant has the same instruction count before and after; the diffs are only register renaming.
- Exact-match correctness against a reference GEMM over 4 shapes (64x64x320, 1000x1032x1000, 4096³, 128x256x72), in debug and release: WS, cooperative, ping-pong, cooperative with a 2x1 cluster, `KernelTma`, and tf32 RS WS. 0 mismatches.
- Performance in release builds, before vs. after: within ±0.6% for every config and shape.
- Unit tests: all 295 tests in 8 SM90 GEMM unit-test binaries pass on H100 (Release, sm_90a): `tensorop_sm90` (96), `cluster_multicast_sm90` (92), `alignx_sm90` (77), `gmma_rs_warpspecialized_sm90` (16), `sm90_ptr_array` (4), `sm90_group_gemm` (4), `sm90_group_gemm_pingpong` (3), `sm90_blockwise` (3). These ran on a build containing both this change and the companion fix for #3721.
- Examples: `56_hopper_ptr_array_batched_gemm`, `57_hopper_grouped_gemm` and `63_hopper_gemm_with_weight_prefetch` report `Disposition: Passed`. `88_hopper_fmha --verify` passes for head dims 32 and 64, which run the non-warp-specialized `FmhaKernelTma`, and for the WS cooperative and ping-pong kernels, with no, causal and residual masks.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
