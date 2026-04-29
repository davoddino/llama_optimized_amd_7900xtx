#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export PORT="${PORT:-${BENCH_PORT:-8002}}"
export BENCH_PORT="$PORT"
export HOST="${HOST:-0.0.0.0}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"

export RDNA3_KERNEL_PRESET="${RDNA3_KERNEL_PRESET:-decode-max}"
export RDNA3_OP_PROFILE="${RDNA3_OP_PROFILE:-0}"
export RDNA3_MMVQ_Q8_CACHE="${RDNA3_MMVQ_Q8_CACHE:-1}"
export RDNA3_MMVQ_Q8_CACHE_LOG="${RDNA3_MMVQ_Q8_CACHE_LOG:-1}"

SERVER_LOG="${SERVER_LOG:-bench-results/qwen36-exact-decode/server.log}"
mkdir -p "$(dirname "$SERVER_LOG")"

server_pid=""

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

(
    cd "$REPO_DIR"
    exec "$SCRIPT_DIR/run-qwen36-35b-tqkv.sh"
) > "$SERVER_LOG" 2>&1 &
server_pid="$!"

for _ in $(seq 1 240); do
    if curl -fsS "http://$BENCH_HOST:$BENCH_PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server exited before becoming healthy; log: $SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://$BENCH_HOST:$BENCH_PORT/health" >/dev/null 2>&1; then
    echo "server did not become healthy within timeout; log: $SERVER_LOG" >&2
    exit 1
fi

"$SCRIPT_DIR/bench-qwen36-exact-decode.sh"
