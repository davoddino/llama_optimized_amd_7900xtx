#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BENCH_PORT="${BENCH_PORT:-8002}"
export PROMPT_TOKENS="${PROMPT_TOKENS:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-1024}"
export MIN_PREDICTED_TOKENS="${MIN_PREDICTED_TOKENS:-$MAX_TOKENS}"
export RUNS="${RUNS:-5}"
export OUT_DIR="${OUT_DIR:-bench-results/qwen36-exact-decode}"
export EXACT_DECODE="${EXACT_DECODE:-1}"
export CACHE_BUST="${CACHE_BUST:-1}"
export REQUIRE_STOP_LIMIT="${REQUIRE_STOP_LIMIT:-1}"
export ALLOW_CACHE="${ALLOW_CACHE:-0}"
export ALLOW_DRAFT="${ALLOW_DRAFT:-0}"
export ALLOW_BACKEND_SAMPLING="${ALLOW_BACKEND_SAMPLING:-0}"

exec "$SCRIPT_DIR/bench-qwen36-server.sh"
