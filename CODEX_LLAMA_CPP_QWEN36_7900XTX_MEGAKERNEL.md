# Codex Mission: llama.cpp fork extreme optimization for Qwen3.6 on AMD Radeon RX 7900 XTX

## Role
You are Codex acting as a senior GPU inference engineer and llama.cpp/ggml contributor. You are working inside a fork of `llama.cpp` and your mission is to implement the fastest possible inference path for Qwen3.6 models on a single AMD Radeon RX 7900 XTX, with special focus on decode throughput, low latency, memory movement elimination, and hardware-specific ROCm/HIP optimization.

This is not a research note. You must modify the repository, implement code, benchmark it, and leave the project in a buildable, runnable state.

## Hard operating rules

1. Do not stop after analysis only. Implement code.
2. Do not sleep, wait, pause, or run commands whose purpose is merely delaying work.
3. Keep working continuously until the requested deadline: **06 May 2026 at 08:00 Europe/Rome**, unless the implementation is complete earlier and all acceptance tests pass.
4. Use internet research when useful to identify current best practices, upstream changes, Qwen3.6 architecture details, llama.cpp ROCm/HIP backend changes, AMD RDNA3 optimization techniques, and relevant kernels.
5. Prefer concrete code changes over long explanations.
6. Avoid unnecessary architectural complexity. Every change must serve throughput, latency, correctness, or maintainability.
7. Do not introduce large external dependencies unless they are clearly justified and optional.
8. Do not break existing llama.cpp functionality for other models/backends.
9. If a requested throughput target is physically impossible on the available hardware/model/quantization/context, still implement the maximum feasible optimization path and produce a clear benchmark report explaining the hard bottleneck with evidence.
10. Always leave a final implementation report in the repository root as `QWEN36_7900XTX_OPTIMIZATION_REPORT.md`.

## Target hardware

Primary target:

- GPU: AMD Radeon RX 7900 XTX
- Architecture: RDNA3 / gfx1100
- VRAM: 24 GB
- Backend: ROCm/HIP preferred
- Fallback backend: Vulkan only if it is clearly faster for a specific kernel path, but do not make Vulkan the main path unless proven by benchmarks
- CPU: unknown; do not depend on high-end CPU performance for decode hot path

Assume the real value is achieved by optimizing GPU decode and reducing CPU/GPU synchronization, memory copies, kernel launch overhead, and non-fused operations.

## Target models

Implement and optimize for both:

1. **Qwen3.6-35B-A3B**
   - MoE model
   - Approximately 35B total parameters, approximately 3B active parameters per token
   - Includes routed experts and shared expert behavior
   - May appear in FP8 or GGUF quantized forms

2. **Qwen3.6-27B**
   - Dense 27B model
   - Must support the same optimized inference pipeline where applicable

Important: inspect model metadata/config/GGUF tensors. Do not hard-code fragile assumptions if the repo exposes metadata-driven architecture support.

## Primary performance targets

Current reported baseline: approximately **80 token/s decode**.

Mandatory stretch targets:

- Single-user decode: **1200 token/s or higher** on one RX 7900 XTX.
- Multi-user aggregate decode: **3000 token/s or higher** on one RX 7900 XTX.

These are aggressive targets. Treat them as non-negotiable engineering goals, but verify with real benchmarks. If hardware/model math proves the exact targets cannot be reached, implement all feasible optimizations and document the measured ceiling, limiting factor, profiler data, model size, quantization, batch size, context, and command lines.

## Main objective

Create a Qwen3.6-specific high-throughput inference path inside llama.cpp/ggml for RX 7900 XTX by implementing a **megakernel or minimal-kernel fused path** for decode, with separate handling for:

- Qwen3.6-35B-A3B MoE decode
- Qwen3.6-27B dense decode
- single-user low-latency decode
- multi-user continuous batching decode

The goal is to remove bottlenecks from the decode loop by fusing operations, reducing memory movement, minimizing kernel launches, avoiding CPU synchronization, improving quantized matmul throughput, and optimizing KV/cache/state access.

## Non-goals

Do not spend time on:

- generic UI work
- simulator features
- enterprise compliance
- unrelated model families unless required to preserve existing tests
- speculative large rewrites that do not land working code
- theoretical design documents without implementation

## Required investigation before coding

Perform a fast but concrete inspection of the repository:

1. Identify current llama.cpp/ggml architecture path for Qwen/Qwen3/Qwen3MoE/Qwen3.6.
2. Identify whether Qwen3.6 uses dedicated graph ops, generic transformer ops, hybrid attention, Gated DeltaNet, MoE routing, FP8, quantized GGUF, or custom metadata.
3. Identify backend files for ROCm/HIP, especially:
   - `ggml/src/ggml-hip/*`
   - `ggml/src/ggml-cuda/*` if HIP reuses CUDA-like source
   - `ggml/src/ggml-vulkan/*` for comparison only
   - quantized matmul kernels
   - MoE kernels
   - flash attention or attention-like kernels
   - graph scheduling and backend buffer code
4. Identify exact hot path during decode by running a baseline benchmark.
5. Record baseline benchmark commands and results in the report.

## Benchmark methodology

Add or reuse benchmark scripts so results are reproducible.

Create a script:

`bench/qwen36_7900xtx_bench.sh`

It should support at least:

- model path argument
- model type label: `qwen36-35b-a3b` or `qwen36-27b`
- backend selection
- number of users / parallel sequences
- prompt length
- generated tokens
- context length
- batch/ubatch settings
- GPU layer settings
- output JSON or markdown log

Benchmark scenarios:

### Single-user decode

- 1 sequence
- warmup run
- fixed prompt length
- fixed generated token count
- report prompt processing tok/s and decode tok/s separately

### Multi-user aggregate decode

- multiple concurrent sequences or continuous batching through `llama-server` if available
- report aggregate decode tok/s
- report per-user latency where possible
- avoid measuring only prompt processing throughput

### Correctness sanity check

- same prompt before/after optimization
- verify no NaNs
- verify token output is plausible
- verify logits are close enough when running deterministic mode, or document expected quantization/kernel variance

## Profiling methodology

Use available tools:

- `rocprof`, `rocprofv2`, or current ROCm profiler if installed
- `hipprof` if available
- llama.cpp built-in timing logs
- backend graph timing
- kernel launch count
- memory bandwidth estimates
- VRAM allocation logging

Profile and report:

1. top kernels by time
2. kernel launch count per generated token
3. CPU/GPU synchronization points
4. host-to-device and device-to-host copies during decode
5. matmul throughput estimate
6. memory bandwidth estimate
7. attention/KV cost
8. MoE routing/expert dispatch cost

## Implementation plan

### Phase 1: Build and baseline

- Confirm the project builds with ROCm/HIP on the current machine.
- Use correct AMD target for RX 7900 XTX, usually `gfx1100`.
- Prefer CMake options aligned with the current llama.cpp tree.
- Run a baseline benchmark for both target models if available locally.
- If models are not present, add scripts and clear instructions but still implement the code changes.

### Phase 2: Eliminate obvious bottlenecks

Find and remove:

- unnecessary host/device copies during decode
- repeated tensor reshaping or layout conversion
- CPU-side routing loops for MoE when GPU routing is possible
- per-token allocations
- per-token graph rebuilds if avoidable
- repeated metadata lookups in hot path
- avoidable synchronization after every small op
- small kernels that can be fused
- backend fallback to CPU for any Qwen3.6 hot-path op

Add logging/assertions to detect CPU fallback for target models.

### Phase 3: Qwen3.6 dense optimized path

For Qwen3.6-27B dense:

- optimize decode path first, not only prompt processing
- fuse normalization + projection where feasible
- fuse activation + elementwise ops
- optimize quantized matrix-vector / matrix-small-batch path for RDNA3
- reduce kernel launches per layer
- optimize attention or hybrid attention path
- ensure KV/cache layout is coalesced and GPU-resident
- tune workgroup sizes/wavefront usage for RDNA3
- prefer vectorized loads/stores
- minimize LDS bank conflicts
- maximize occupancy without excessive register pressure

### Phase 4: Qwen3.6-35B-A3B MoE optimized path

For Qwen3.6-35B-A3B:

- implement or improve GPU-side router/top-k dispatch
- avoid CPU routing
- avoid scattering tiny expert jobs in a way that creates many small kernels
- group tokens by expert efficiently for multi-user batching
- implement fused expert execution where feasible
- exploit active-parameter sparsity
- optimize shared expert path
- fuse gating weights into expert output combine
- minimize intermediate writes
- implement specialized kernels for the typical decode batch sizes

If a true single monolithic megakernel is not maintainable, implement a minimal set of fused kernels and document why this is faster and safer.

### Phase 5: Megakernel / fused decode path

Implement a specialized Qwen3.6 decode execution path that fuses as many of these as practical:

- RMSNorm or equivalent normalization
- Q/K/V projection or relevant projections
- RoPE / partial RoPE if present
- attention or hybrid attention update
- residual operations
- MLP / SwiGLU or Qwen-specific activation
- MoE route + expert matmul + combine for A3B
- quant dequant where possible fused into matmul
- logits projection optimizations if applicable

Requirements:

- compile only when ROCm/HIP backend is enabled
- guard with runtime feature checks and model architecture checks
- provide fallback to existing llama.cpp path
- expose an enable/disable flag such as `--qwen36-fast-path` or backend-specific environment variable
- include debug mode to compare against normal path

### Phase 6: Quantization and FP8 handling

Investigate current support for:

- GGUF quantized Qwen3.6 files
- FP8 Qwen3.6-35B-A3B if present
- Q4_K, Q5_K, Q6_K, Q8_0, IQ/ik-style quants if the fork supports them

Optimize the most relevant format available locally. Priority:

1. format that fits fully in 24 GB VRAM
2. fastest decode format with acceptable quality
3. format commonly used by local users

Do not add a new quantization format unless absolutely necessary. Prefer optimizing existing quantized matmul paths.

### Phase 7: Continuous batching / multi-user serving

For aggregate multi-user throughput:

- optimize llama-server or equivalent serving path
- enable continuous batching where available
- reduce per-request overhead
- batch decode tokens from multiple users efficiently
- keep KV caches GPU-resident
- avoid lock contention in scheduler
- report aggregate tok/s and per-user latency

Add a benchmark mode that simulates concurrent requests without introducing a large external load-testing dependency.

### Phase 8: Build flags and hardware tuning

Add documented build profile for RX 7900 XTX:

- ROCm/HIP enabled
- AMD GPU target `gfx1100`
- release build
- link-time optimization if safe
- native CPU flags where safe
- optional fast math only if correctness is acceptable

Example expected documentation location:

`docs/qwen36_7900xtx_rocm.md`

Include exact commands for:

- clean build
- benchmark
- profiler run
- enabling/disabling fast path

## Code quality requirements

- Keep changes isolated and understandable.
- Prefer backend-specific files over polluting generic model code.
- Add comments only where the optimization is non-obvious.
- Do not hard-code absolute local paths.
- Do not delete existing features.
- Use feature flags for experimental fast paths.
- Ensure compile errors are fixed before finishing.
- Run formatting if the repository has a standard formatter.

## Correctness requirements

Before declaring success:

1. Build must pass.
2. Existing relevant tests must pass, or failures must be documented if unrelated.
3. Target model must run without crash.
4. Decode output must be plausible.
5. No CPU fallback in hot path unless explicitly documented and unavoidable.
6. No avoidable H2D/D2H copies during decode.
7. Fast path must be disableable.
8. Benchmark report must include baseline and optimized numbers.

## Files to create or update

Required new files:

- `QWEN36_7900XTX_OPTIMIZATION_REPORT.md`
- `docs/qwen36_7900xtx_rocm.md`
- `bench/qwen36_7900xtx_bench.sh`

Likely files to inspect/update:

- `ggml/src/ggml-hip/*`
- `ggml/src/ggml-cuda/*` if HIP shares CUDA source
- `ggml/src/ggml-backend*`
- `ggml/src/ggml.c` / `ggml/src/ggml.cpp` depending tree layout
- `src/llama.cpp`
- model architecture files for Qwen/Qwen3/Qwen3MoE/Qwen3.6
- server batching/scheduler code if optimizing multi-user throughput
- CMake build files

## Final report format

At the end, write `QWEN36_7900XTX_OPTIMIZATION_REPORT.md` with:

1. summary of implemented changes
2. exact git diff summary
3. build commands used
4. benchmark commands used
5. baseline performance table
6. optimized performance table
7. single-user decode tok/s
8. multi-user aggregate decode tok/s
9. prompt processing tok/s
10. model file/quantization/context/batch settings
11. profiler summary
12. remaining bottlenecks
13. whether 1200 tok/s single-user was reached
14. whether 3000 tok/s multi-user was reached
15. if not reached, precise reason with evidence
16. next highest-impact optimization

## Strong preference

The best solution is not a giant unmaintainable hack. The best solution is a hardware-specialized fast path that is:

- extremely fast on RX 7900 XTX
- safe to disable
- correct enough for production inference
- benchmarked honestly
- isolated from generic llama.cpp paths
- ready for iterative improvement

## Start now

Begin by inspecting the repository, building the current ROCm/HIP backend, and creating the benchmark script. Then profile decode, implement the highest-impact bottleneck removals first, and continue until the deadline or until all targets are met.

You can't run code here cause you are on a MAC. The code will be tested on another machine at the end of the work
