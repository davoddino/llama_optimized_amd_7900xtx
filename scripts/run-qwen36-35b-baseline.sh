#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${LLAMA_DIR:-$REPO_DIR}"
MODEL_DIR="${MODEL_DIR:-/home/kilolab/big-storage/llms/models/unsloth}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
MODEL="${MODEL:-$MODEL_DIR/$MODEL_FILE}"
ALIAS="${ALIAS:-qwen3.6-35b}"
DOWNLOAD_URL="${DOWNLOAD_URL:-https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-GGUF/resolve/main/$MODEL_FILE}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CTX_SIZE="${CTX_SIZE:-128000}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"

if [[ ! -f "$MODEL" ]]; then
    if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
        mkdir -p "$MODEL_DIR"
        echo "Model not found. Downloading with wget:"
        echo "  $DOWNLOAD_URL"
        wget -c "$DOWNLOAD_URL" -O "$MODEL"
    else
        echo "Model not found: $MODEL" >&2
        echo "Set DOWNLOAD_MODEL=1 to download it, or set MODEL=/path/to/model.gguf." >&2
        exit 1
    fi
fi

if [[ -n "${SERVER_BIN:-}" ]]; then
    if [[ ! -x "$SERVER_BIN" ]]; then
        echo "SERVER_BIN is set but not executable: $SERVER_BIN" >&2
        exit 1
    fi
else
    SERVER_CANDIDATES=(
        "$LLAMA_DIR/build-rocm-gfx1100/bin/llama-server"
        "$LLAMA_DIR/build/bin/llama-server"
        "$LLAMA_DIR/bin/llama-server"
        "$LLAMA_DIR/llama-server"
    )

    SERVER_BIN=""
    for candidate in "${SERVER_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            SERVER_BIN="$candidate"
            break
        fi
    done

    if [[ -z "$SERVER_BIN" ]]; then
        echo "llama-server not found. Build first, then retry." >&2
        echo "Expected one of:" >&2
        printf '  %s\n' "${SERVER_CANDIDATES[@]}" >&2
        echo "Or set SERVER_BIN=/full/path/to/llama-server." >&2
        exit 1
    fi
fi

echo "Starting baseline server"
echo "  model: $MODEL"
echo "  cache: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "  bin:   $SERVER_BIN"
echo "  url:   http://$HOST:$PORT"

exec "$SERVER_BIN" \
    -m "$MODEL" \
    -a "$ALIAS" \
    -ngl all \
    -fa on \
    -np 1 \
    -cb \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --jinja \
    --reasoning off \
    --reasoning-format none \
    --reasoning-budget 0 \
    --temp 0 \
    --top-k 1 \
    --top-p 1 \
    --ctx-size "$CTX_SIZE" \
    --metrics \
    --host "$HOST" \
    --port "$PORT"
