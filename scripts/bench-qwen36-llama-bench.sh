#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="${LLAMA_DIR:-$REPO_DIR}"
MODEL_DIR="${MODEL_DIR:-/home/kilolab/big-storage/llms/models/unsloth}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
MODEL="${MODEL:-$MODEL_DIR/$MODEL_FILE}"

CTX_SIZE="${CTX_SIZE:-128000}"
PROMPT_SIZES="${PROMPT_SIZES:-8192,32768,65536}"
GEN_TOKENS="${GEN_TOKENS:-256}"
REPETITIONS="${REPETITIONS:-3}"
TQKV_PROFILE="${TQKV_PROFILE:-fast}"
OUT_FORMAT="${OUT_FORMAT:-md}"

case "$TQKV_PROFILE" in
    fast)
        CACHE_TYPE_K="tqkv_fast"
        CACHE_TYPE_V="tqkv_fast"
        ;;
    compact)
        CACHE_TYPE_K="tqkv_compact"
        CACHE_TYPE_V="tqkv_compact"
        ;;
    quality)
        CACHE_TYPE_K="tqkv_quality"
        CACHE_TYPE_V="tqkv_quality"
        ;;
    custom)
        CACHE_TYPE_K="${CACHE_TYPE_K:-tqkv_fast}"
        CACHE_TYPE_V="${CACHE_TYPE_V:-tqkv_fast}"
        ;;
    *)
        echo "Unsupported TQKV_PROFILE=$TQKV_PROFILE" >&2
        echo "Allowed values: fast, compact, quality, custom" >&2
        exit 1
        ;;
esac

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    exit 1
fi

if [[ -n "${BENCH_BIN:-}" ]]; then
    if [[ ! -x "$BENCH_BIN" ]]; then
        echo "BENCH_BIN is set but not executable: $BENCH_BIN" >&2
        exit 1
    fi
else
    BENCH_CANDIDATES=(
        "$LLAMA_DIR/build-rocm-gfx1100/bin/llama-bench"
        "$LLAMA_DIR/build/bin/llama-bench"
        "$LLAMA_DIR/bin/llama-bench"
        "$LLAMA_DIR/llama-bench"
    )

    BENCH_BIN=""
    for candidate in "${BENCH_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            BENCH_BIN="$candidate"
            break
        fi
    done

    if [[ -z "$BENCH_BIN" ]]; then
        echo "llama-bench not found. Build first, then retry." >&2
        echo "Expected one of:" >&2
        printf '  %s\n' "${BENCH_CANDIDATES[@]}" >&2
        echo "Or set BENCH_BIN=/full/path/to/llama-bench." >&2
        exit 1
    fi
fi

echo "llama-bench benchmark"
echo "  model:        $MODEL"
echo "  bin:          $BENCH_BIN"
echo "  ctx size:     $CTX_SIZE"
echo "  prompt sizes: $PROMPT_SIZES"
echo "  gen tokens:   $GEN_TOKENS"
echo "  repetitions:  $REPETITIONS"
echo "  profile:      $TQKV_PROFILE"
echo "  cache:        K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo

exec "$BENCH_BIN" \
    -m "$MODEL" \
    -ngl all \
    -fa 1 \
    -c "$CTX_SIZE" \
    -p "$PROMPT_SIZES" \
    -n "$GEN_TOKENS" \
    -r "$REPETITIONS" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    -o "$OUT_FORMAT"
