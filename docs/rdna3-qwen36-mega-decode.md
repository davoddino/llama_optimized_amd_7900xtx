# RDNA3 Qwen3.6 Mega Decode

This fork has an experimental contract mode for a device-resident Qwen3.6-35B-A3B decode path on RDNA3/gfx1100.

The target path is not the generic ggml executor. The target hot loop is:

1. keep token id, positions, KV, recurrent state, activations, router ids, router weights, logits, sampler state, and next token on the GPU;
2. run the fixed Qwen3.6 layer topology as one persistent mega-decode runtime or as a small fixed sequence of cooperative launches;
3. return tokens to the CPU only as final output for streaming, never as an input dependency for the next decode token.

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

When `GGML_CUDA_RDNA3_QWEN36_MEGA_REQUIRED=1`, failing this contract aborts immediately for one-token decode graphs instead of silently running the old decode path. Prompt processing and warmup graphs are reported but are not forced through this contract. The required mode also blocks the `MUL_MAT_ID` host-sync fallback inside matching one-token decode graphs and rejects per-op profiling there, because per-op profiling injects events and disables graph-level execution.

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
