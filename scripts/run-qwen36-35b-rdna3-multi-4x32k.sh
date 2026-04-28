#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RDNA3 / RX 7900 XTX throughput-first profile:
# 4 server slots, each with a 32k context budget.
export TQKV_PROFILE="${TQKV_PROFILE:-fast}"
export TQKV_PORT="${TQKV_PORT:-8004}"
export CTX_SIZE="${CTX_SIZE:-131072}"
export PARALLEL="${PARALLEL:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export UBATCH_SIZE="${UBATCH_SIZE:-512}"
export PROMPT_CACHE_MB="${PROMPT_CACHE_MB:-0}"
export KV_UNIFIED="${KV_UNIFIED:-0}"
export CACHE_IDLE_SLOTS="${CACHE_IDLE_SLOTS:-0}"
export RDNA3_PROFILE_LOG="${RDNA3_PROFILE_LOG:-0}"
export RDNA3_FAIL_ON_HOST_SYNC="${RDNA3_FAIL_ON_HOST_SYNC:-0}"

exec "$SCRIPT_DIR/run-qwen36-35b-tqkv.sh"
