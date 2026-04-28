#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RDNA3 / RX 7900 XTX decode-oriented profile for Qwen3.6-35B-A3B.
# The binary must be rebuilt after changing ggml/src/ggml-cuda/mmvq.cu.
export TQKV_PROFILE="${TQKV_PROFILE:-fast}"
export TQKV_PORT="${TQKV_PORT:-8002}"
export CTX_SIZE="${CTX_SIZE:-32768}"
export BATCH_SIZE="${BATCH_SIZE:-4096}"
export UBATCH_SIZE="${UBATCH_SIZE:-512}"
export PROMPT_CACHE_MB="${PROMPT_CACHE_MB:-0}"

exec "$SCRIPT_DIR/run-qwen36-35b-tqkv.sh"
