# Qwen3.6 on RX 7900 XTX with ROCm/HIP

This note is for this fork's experimental Qwen3.6 RDNA3 path. It targets one AMD Radeon RX 7900 XTX (`gfx1100`, 24 GB VRAM) and keeps the optimized path opt-in.

The public Qwen3.6-35B-A3B GGUFs are usually exposed to llama.cpp through the Qwen3.5 MoE architecture path. Reported metadata for current GGUFs: 40 layers, hidden size 2048, 256 routed experts, top-8 routed experts plus shared expert, hybrid Gated DeltaNet and attention layers, and 262K native context. Dense Qwen3.6-27B uses the same dense matvec, attention, KV, and sampler offload improvements where the graph matches existing ggml ops.

## Build

Use a ROCm machine with the RX 7900 XTX visible.

```bash
git status --short
./scripts/build-rocm-gfx1100.sh
```

Equivalent explicit CMake command:

```bash
ROCM_PATH=/opt/rocm \
CC=/opt/rocm/llvm/bin/amdclang \
CXX=/opt/rocm/llvm/bin/amdclang++ \
cmake -S . -B build-rocm-gfx1100 -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-rocm-gfx1100 -j "$(nproc)"
```

If `GGML_HIP_ROCWMMA_FATTN=ON` fails because rocWMMA headers are not installed, rebuild without that option and record the change in the benchmark report.

## Fast Path Control

The conservative fast path is exposed as both CLI/server arg and env preset:

```bash
--qwen36-fast-path
--no-qwen36-fast-path
LLAMA_ARG_QWEN36_FAST_PATH=1
```

It enables these existing backend gates:

```bash
GGML_CUDA_RDNA3_QWEN36_FASTPATH=1
GGML_CUDA_RDNA3_QWEN36_LINEAR_MMVQ_FAST=1
GGML_CUDA_RDNA3_QWEN36_TOPK_FASTPATH=1
GGML_CUDA_RDNA3_MMVQ_Q8_CACHE=1
GGML_CUDA_RDNA3_TQKV_FATTN_GQA_DECODE=1
```

Final composed path:

```bash
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL=1
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_WEIGHTS=1
```

This preset enables the RDNA3 Qwen3.6 fast path, the physical L0 superlayer replacement, and the all-layer one-layer QKV/projection mega stages where the graph and weight types match.

For the current physical L0 replacement path:

```bash
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER=1
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DISPATCH=1
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0=1
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_WEIGHTS=1
```

`REPLACE_L0=1` enables all implemented L0 stages, including RMS/QKV/projection, recurrent post-processing, attention output, post-attention norm, MoE router, MoE gate/up, and MoE down plus weighted reduction.

## Single-User Benchmark

Baseline:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset baseline \
  --scenario single \
  --prompt-len 8192 \
  --gen-tokens 256 \
  --ctx-size 128000 \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --output-format json
```

Optimized:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset final \
  --scenario single \
  --prompt-len 8192 \
  --gen-tokens 256 \
  --ctx-size 128000 \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --output-format json
```

Dense model:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-27B-Q4_K_M.gguf \
  --model-type qwen36-27b \
  --backend hip \
  --preset fast \
  --scenario single \
  --prompt-len 8192 \
  --gen-tokens 256
```

## Multi-User Benchmark

This starts `llama-server`, sends concurrent OpenAI-compatible requests, and writes aggregate decode throughput plus per-user latency.

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset fast \
  --scenario multi \
  --users 8 \
  --prompt-len 4096 \
  --gen-tokens 256 \
  --ctx-size 128000 \
  --batch-size 4096 \
  --ubatch-size 1024
```

For strict graph-contract validation:

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-type qwen36-35b-a3b \
  --backend hip \
  --preset mega-contract \
  --scenario multi \
  --users 8
```

`mega-contract` should abort if the one-token decode graph falls back to the legacy path.

## Correctness Sanity

```bash
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --preset fast \
  --scenario correctness
```

The script runs deterministic baseline and optimized samples, rejects obvious NaN/Inf text, and saves both outputs. For a real kernel validation pass, compare logits or selected intermediate tensors with the existing per-stage `REPLACE_L0_*` flags.

## Profiling

Built-in backend profiling:

```bash
GGML_CUDA_RDNA3_OP_PROFILE=1 \
GGML_CUDA_RDNA3_OP_PROFILE_MAX_EVALS=8 \
GGML_CUDA_RDNA3_OP_PROFILE_SUMMARY_ROWS=40 \
bench/qwen36_7900xtx_bench.sh \
  --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --preset fast \
  --scenario single
```

ROCm profiler, if installed:

```bash
rocprof --hip-trace --stats \
  -o bench-results/qwen36-7900xtx/rocprof-fast.csv \
  bench/qwen36_7900xtx_bench.sh \
    --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --preset fast \
    --scenario single \
    --repetitions 1
```

Collect at least:

- top kernels by time
- kernel launch count per generated token
- host-to-device and device-to-host copies during decode
- `MUL_MAT_ID` host-sync fallback status
- prompt tokens/s and decode tokens/s separately
- VRAM allocation and cache type

## Current Optimization Surface

Implemented or present in this fork:

- RDNA3 Qwen3.6 fast-path env gates for MMVQ, linear projection, TopK MoE, TQKV GQA decode, and Q8 activation cache.
- Backend sampling support to avoid full logits readback when sampler chain is compatible.
- Strict graph-contract and no-host-output guards for one-token decode validation.
- Experimental physical superlayer scaffold with isolated L0 replacement stages.
- Physical L0 MoE down projection plus weighted reduction behind `GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_DOWN=1`.
- Reproducible benchmark wrapper for single-user, multi-user, and correctness sanity runs.

Remaining high-impact work:

- Replace the scaffold superlayer with a complete numeric 40-layer decode runtime.
- Generalize the physical MoE tail beyond L0 and validate the full L0 replacement on the RX 7900 XTX ROCm host.
- Keep next-token state fully device-side so decode does not depend on host token/position updates.
- Add a backend-compatible distribution sampler chain beyond greedy/top-k.
