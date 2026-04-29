#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BENCH_PORT="${BENCH_PORT:-8002}"
export PROMPT_TOKENS="${PROMPT_TOKENS:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-512}"
export MIN_PREDICTED_TOKENS="${MIN_PREDICTED_TOKENS:-$MAX_TOKENS}"
export RUNS="${RUNS:-3}"
export OUT_DIR="${OUT_DIR:-bench-results/rdna3-single}"
export EXACT_DECODE="${EXACT_DECODE:-1}"

exec "$SCRIPT_DIR/bench-qwen36-server.sh"
