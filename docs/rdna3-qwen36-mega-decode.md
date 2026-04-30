# RDNA3 Qwen3.6 Mega Decode

This fork has an experimental contract mode for a device-resident Qwen3.6-35B-A3B decode path on RDNA3/gfx1100.

The target path is not the generic ggml executor. The target hot loop is:

1. keep token id, positions, KV, recurrent state, activations, router ids, router weights, logits, sampler state, and next token on the GPU;
2. run the fixed Qwen3.6 layer topology as one persistent mega-decode runtime or as a small fixed sequence of cooperative launches;
3. return tokens to the CPU only as final output for streaming, never as an input dependency for the next decode token.

## Physical superlayer path

The stricter target is a load-time generated physical superlayer for the exact Qwen3.6/gfx1100 topology. This is separate from the older one-layer QKV/projection experiment.

```bash
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER=1
```

When enabled on a matching one-token Qwen3.6 decode graph, the backend computes a topology fingerprint and materializes a cache artifact:

```text
$XDG_CACHE_HOME/llama.cpp/rdna3-qwen36-superlayer/<fingerprint>/
  manifest.json
  layout.generated.h
  fused_model.hip
  weightpack.plan
  weightpack.generated.h
  weightpack.layout.json
  runtime.layout.json
  runtime-bindings.layout.json
```

This artifact is intentionally not a generic layer manifest. The generated source has 40 static fused blocks in the exact graph order, with no runtime layer loop. The current runtime also allocates one device-side weightpack buffer, one persistent runtime scratch buffer, uploads a 40-entry layer descriptor table, uploads runtime tensor bindings for graph inputs/cache/state/output, and copies the packed tensors into deterministic gfx1100 offsets. The dispatch kernel can execute the first real fused numeric steps, L0 RMSNorm, L0 `linear_attn_qkv_mixed`, and the L0 `z`/`beta`/`alpha` projection bundle inside that single physical launch by reading activation input plus packed projection weights from the fused device weightpack. With `RUN_L0=1` it writes those outputs both into persistent scratch and into the real ggml tensors while the normal graph still runs. With `REPLACE_L0=1` the runner also skips the replaced L0 graph nodes after a successful superlayer dispatch. The remaining decode math beyond L0 is still scaffold code and does not replace the normal ggml decode yet.

Optional strict/scaffold controls:

```bash
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REQUIRED=1
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DISPATCH=1
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CONTRACT=0
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CACHE=/path/to/cache
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_BLOCKS=96
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_THREADS=256
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_RUN_L0=0
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0=0
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_RMS=0
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_QKV=0
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ=0
GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FULL_WEIGHTPACK=0
```

`GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SMOKE=1` is kept as an alias for `DISPATCH=1`. `DISPATCH=1` alone no longer materializes or launches per-token work; it only enables the channel. Set `CONTRACT=1` to materialize the runtime pack and launch the scaffold cooperative contract kernel without L0 math. Set `RUN_L0=1` to execute the L0 RMSNorm/QKV/projection math without skipping the normal graph. `GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0` also defaults to `0`; setting it to `1` implies L0 math and skips all replaced L0 graph nodes. The narrower `REPLACE_L0_RMS`, `REPLACE_L0_QKV`, and `REPLACE_L0_PROJ` flags replace only that L0 substage and are intended for correctness isolation. The artifact still records the full 40-layer pack layout, but the runtime device pack defaults to the implemented L0 tensor subset to avoid duplicating the full model in VRAM. `GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FULL_WEIGHTPACK=1` restores the full duplicate device pack and is expected to exceed 24 GB on the target model. The dispatch path launches one compiled cooperative physical contract kernel that calls 40 static layer blocks, reads through `device_pack + layer_descs`, reads runtime bindings, and optionally runs L0 math row-parallel in-block. It is still mostly scaffold, but it is a single physical dispatch over the packed model data, not a ggml layer loop.

The weightpack files define the persistent layout and the runtime log reports the process-local device buffer pointer as `device_pack=...`. The next implementation step is replacing the first recurrent attention state update with direct reads from the RMSNorm/QKV/projection scratch outputs and packed weights.

The current `GGML_CUDA_RDNA3_QWEN36_MEGA_DECODE=1` implementation is a hard contract gate. It validates that the one-token graph has the Qwen3.6-35B-A3B signature:

- 10 full-attention layers;
- 30 recurrent/Gated Delta Net layers;
- 40 MoE router nodes;
- 40 MoE down projections;
- fused gate/up projection nodes for every layer;
- no `MUL_MAT_ID` path that would need the legacy Device->Host sort fallback.

The server helper exposes this as:

```bash
RDNA3_KERNEL_PRESET=mega-contract scripts/run-qwen36-35b-tqkv.sh
```

If the default port `8002` is already busy and `TQKV_PORT` was not set explicitly, the helper picks the next free port and prints the effective URL. Set `TQKV_PORT=8003` to force a specific port, or `TQKV_PORT_AUTO=0` to fail instead of auto-selecting.

The `mega-contract` preset now forces backend sampling and blocks raw logits readback:

```bash
BACKEND_SAMPLING=1
GGML_CUDA_RDNA3_QWEN36_MEGA_REQUIRE_GRAPH=1
GGML_CUDA_RDNA3_QWEN36_MEGA_SAMPLE_TOKEN_ONLY=1
GGML_CUDA_RDNA3_QWEN36_MEGA_NO_RAW_LOGITS=1
GGML_CUDA_RDNA3_QWEN36_MEGA_ASYNC_INPUTS=1
```

This removes the legacy per-token `result_output` Device->Host copy from the normal benchmark path, requires the matching one-token decode graph to launch through CUDA/HIP graphs after a short warmup window, and only copies the sampled token from backend sampling. Raw logits are not reserved in the host output buffer while `NO_RAW_LOGITS` is active. Auxiliary sampler tensors (`logits`, `probs`, and candidate ids) are kept inside the graph unless the sampler chain fails to produce a backend sampled token, and the host output buffer does not reserve storage for them in token-only mode. Input tensor updates use backend async copies when the target input buffer is device-side. If backend sampling is disabled or made unavailable by an incompatible sampler, the run fails instead of silently measuring the old CPU-sampling path.

On ROCm builds without CUB, `mega-contract` uses a small-k large-vocabulary backend `TOP_K` path and sets the default server sampler chain to `top_k;temperature`. `top_p` remains disabled for the server defaults because it still requires a full-vocabulary `ARGSORT` backend op. The final distribution sampler is appended by the common sampler chain.

The helper also keeps the Qwen3.6 single-token kernels on latency-oriented occupancy defaults: dynamic top-k MoE rows/block, MoE gate/up rows/block = 4, MoE down rows/block = 8, and Gated Delta Net autoregressive columns/block = 8. Wider row/block settings reduce launch count but underfill the 1-token decode kernels on gfx1100.

When `GGML_CUDA_RDNA3_QWEN36_MEGA_REQUIRED=1`, failing this contract aborts immediately for one-token decode graphs instead of silently running the old decode path. Prompt processing and warmup graphs are reported but are not forced through this contract. The required mode also blocks the `MUL_MAT_ID` host-sync fallback inside matching one-token decode graphs and rejects per-op profiling there, because per-op profiling injects events and disables graph-level execution.

For the matching one-token Qwen3.6 decode graph, the CUDA/HIP graph cache key is stable per device instead of using `cgraph->nodes[0]`. This is deliberate: a fixed decode topology should not lose graph reuse because the generic ggml graph object or first-node pointer changes between tokens.

The graph launch guard allows a small number of direct executions for capture warmup. Tune it with:

```bash
GGML_CUDA_RDNA3_QWEN36_MEGA_GRAPH_GRACE_EVALS=8
```

If the graph never stabilizes, the run aborts with the direct-execution reason instead of producing a misleading tokens/sec result.

Optional strict output mode:

```bash
GGML_CUDA_RDNA3_QWEN36_MEGA_NO_HOST_OUTPUT=1
```

This aborts on CUDA/HIP backend tensor reads from device to host. It is intentionally strict and is meant to expose remaining CPU roundtrips while building the true device-resident token loop.

Full host-I/O strict mode:

```bash
GGML_CUDA_RDNA3_QWEN36_MEGA_NO_HOST_IO=1
```

This also aborts on one-token host token/position input. In the current ggml path this is expected to trip immediately during decode; that is the point of the guard. The final runtime must replace those host inputs with device-side next-token and position state.

Expected next implementation steps:

1. replace host `set_inputs()` for one-token decode with device-side token/position state;
2. move top-k/top-p/temp/dist sampling into the Qwen3.6 decode runtime and keep the sampled token as a device scalar;
3. compile a static per-layer device manifest from the loaded Qwen3.6 tensors;
4. collapse the per-layer ggml schedule into a persistent RDNA3 runtime with fixed activation scratch, fixed expert buffers, and no runtime allocation;
5. keep the existing ggml path only for prompt processing, unsupported models, and diagnostics.
