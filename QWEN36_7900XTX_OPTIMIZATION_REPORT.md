# Qwen3.6 RX 7900 XTX Optimization Report

Date: 2026-05-06  
Target: AMD Radeon RX 7900 XTX, RDNA3/gfx1100, ROCm/HIP

## 1. Summary

Implemented in this worktree:

- Added `--qwen36-fast-path` / `--no-qwen36-fast-path` common CLI/server flag. It maps to the existing RDNA3 Qwen3.6 backend gates for MMVQ, TopK, Q8 activation cache, and TQKV GQA decode.
- Added required benchmark wrapper: `bench/qwen36_7900xtx_bench.sh`.
- Added ROCm/RX 7900 XTX operating guide: `docs/qwen36_7900xtx_rocm.md`.
- Preserved the existing experimental superlayer work already present in the dirty tree, including L0 recurrent/post-attention/router/gate-up replacement scaffolding.
- Added physical L0 MoE down projection plus router-weighted reduction behind `GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_DOWN=1`.
- Updated the aggregate `GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0=1` path so it enables every implemented L0 replacement stage.
- Wired `GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL=1` to the composed final runtime: physical L0 superlayer replacement plus all-layer one-layer QKV/projection mega stages where supported.

The current machine is macOS/Apple Silicon without ROCm or the target GPU. Target ROCm build, profiling, and model benchmarks must be run on the RX 7900 XTX host.

AI assistance was used for this local fork work. Do not submit these changes upstream without human review, ownership, and disclosure consistent with this repository policy.

## 2. Git Diff Summary

Tracked diff observed before adding this report:

```text
common/arg.cpp                                  |   28 +
docs/rdna3-qwen36-mega-decode.md               |   15 +-
ggml/src/ggml-cuda/ggml-cuda.cu                |  254 ++-
ggml/src/ggml-cuda/rdna3-qwen36-superlayer.cu  | 2580 +++++++++++++++++++++---
ggml/src/ggml-cuda/rdna3-qwen36-superlayer.cuh |    5 +
```

New required files:

```text
bench/qwen36_7900xtx_bench.sh
docs/qwen36_7900xtx_rocm.md
QWEN36_7900XTX_OPTIMIZATION_REPORT.md
```

Pre-existing untracked file left untouched:

```text
CODEX_LLAMA_CPP_QWEN36_7900XTX_MEGAKERNEL.md
```

## 3. Build Commands Used

Host-only syntax and portable build checks run on macOS:

```bash
bash -n bench/qwen36_7900xtx_bench.sh
bash -n scripts/build-rocm-gfx1100.sh
bash -n scripts/run-qwen36-35b-tqkv.sh

cmake -S . -B build-codex-verify \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-codex-verify --target llama-cli llama-server llama-bench -j 8
cmake --build build-codex-verify -j 2
```

Result: passed. The second full host-only build also completed after the L0 MoE down changes.

Target ROCm build command to run on the 7900 XTX host:

```bash
./scripts/build-rocm-gfx1100.sh
```

## 4. Benchmark Commands

Baseline single-user:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset baseline \
  --scenario single \
  --prompt-len 8192 \
  --gen-tokens 256
```

Optimized single-user:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset fast \
  --scenario single \
  --prompt-len 8192 \
  --gen-tokens 256
```

Optimized multi-user:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset fast \
  --scenario multi \
  --users 8 \
  --prompt-len 4096 \
  --gen-tokens 256
```

Final preset:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset final \
  --scenario single \
  --prompt-len 8192 \
  --gen-tokens 256
```

Physical L0 replacement isolation:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset superlayer-l0 \
  --scenario single
```

## 5. Baseline Performance

Not measured on this Mac. User-reported baseline is approximately 80 token/s decode.

| model | quant | context | batch/ubatch | users | prompt tok/s | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | pending target run | 128000 | 4096/1024 | 1 | pending | reported ~80 |
| Qwen3.6-27B | pending target run | 128000 | 4096/1024 | 1 | pending | pending |

## 6. Optimized Performance

Not measured on target hardware yet.

| model | preset | users | prompt tok/s | decode tok/s | aggregate decode tok/s |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | fast | 1 | pending | pending | pending |
| Qwen3.6-35B-A3B | fast | 8 | pending | pending | pending |
| Qwen3.6-27B | fast | 1 | pending | pending | pending |

## 7. Model And Graph Findings

- Qwen3.6-35B-A3B GGUFs currently appear through the Qwen3.5 MoE converter path (`Qwen3_5MoeForConditionalGeneration` -> `qwen35moe`).
- Relevant local architecture paths include `LLM_ARCH_QWEN35`, `LLM_ARCH_QWEN35MOE`, and `LLM_ARCH_QWEN3NEXT`.
- Generic MoE graph path builds `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_topk`, `ffn_moe_gate_up`, `ffn_moe_swiglu`, `ffn_moe_down`, weighted combine, and `ffn_moe_out`.
- HIP builds reuse `ggml/src/ggml-cuda/*.cu` through `ggml/src/ggml-hip/CMakeLists.txt`.
- Existing fork hot-path controls cover MMVQ, fused MoE gate/up and down paths, TopK MoE, Gated Delta Net, TQKV attention, backend sampling, host-output suppression, and graph-contract validation.

## 8. Profiler Summary

Target profiler data not available on this host. Required target collection:

```bash
GGML_CUDA_RDNA3_OP_PROFILE=1 \
GGML_CUDA_RDNA3_OP_PROFILE_MAX_EVALS=8 \
GGML_CUDA_RDNA3_OP_PROFILE_SUMMARY_ROWS=40 \
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --preset fast \
  --scenario single \
  --repetitions 1
```

If available:

```bash
rocprof --hip-trace --stats \
  -o bench-results/qwen36-7900xtx/rocprof-fast.csv \
  bench/qwen36_7900xtx_bench.sh \
    --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --preset fast \
    --scenario single \
    --repetitions 1
```

## 9. Targets

| target | status |
|---|---|
| 1200 tok/s single-user decode | not verified |
| 3000 tok/s multi-user aggregate decode | not verified |
| Build passes | host-only passed; ROCm target pending |
| No CPU fallback in hot path | target validation pending |
| Fast path disableable | yes, `--no-qwen36-fast-path` and env off |

The reason the throughput targets are not claimed is evidence-based: the only available execution host here is a Mac without the RX 7900 XTX, ROCm runtime, or local Qwen3.6 models.

## 10. Remaining Bottlenecks

Highest-impact next work on target hardware:

1. Run baseline and `--qwen36-fast-path` benchmarks, then compare kernel launch count and D2H copies.
2. Validate whether `MUL_MAT_ID` or router/top-k still triggers any host synchronization in one-token decode.
3. Validate the complete physical L0 replacement path on the RX 7900 XTX ROCm host.
4. Move next-token and position state fully device-side for continuous decode.
5. Expand backend sampling beyond greedy/top-k without full logits readback.

## 11. References Used

- Qwen3.6-35B-A3B announcement: https://qwen.ai/blog?id=qwen3.6-35b-a3b
- GGUF metadata example for Qwen3.6-35B-A3B: https://huggingface.co/batiai/Qwen3.6-35B-A3B-GGUF
- llama.cpp HIP build docs in this repo: `docs/build.md`
- Local RDNA3 Qwen3.6 notes: `docs/rdna3-qwen36-mega-decode.md`
