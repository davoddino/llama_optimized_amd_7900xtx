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
PORT="${TQKV_PORT:-8002}"
CTX_SIZE="${CTX_SIZE:-128000}"
PARALLEL="${PARALLEL:-${N_PARALLEL:-1}}"
TQKV_PROFILE="${TQKV_PROFILE:-fast}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-1024}"
PROMPT_CACHE_MB="${PROMPT_CACHE_MB:-0}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-0}"
CHECKPOINT_EVERY_NT="${CHECKPOINT_EVERY_NT:--1}"
BACKEND_SAMPLING="${BACKEND_SAMPLING:-0}"
SAMPLING_TEMP="${SAMPLING_TEMP:-1}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-20}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.95}"
REASONING_MODE="${REASONING_MODE:-auto}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:--1}"
KV_UNIFIED="${KV_UNIFIED:-}"
CACHE_IDLE_SLOTS="${CACHE_IDLE_SLOTS:-}"
RDNA3_PROFILE_LOG="${RDNA3_PROFILE_LOG:-0}"
RDNA3_OP_PROFILE="${RDNA3_OP_PROFILE:-0}"
RDNA3_OP_PROFILE_MAX_TOKENS="${RDNA3_OP_PROFILE_MAX_TOKENS:-}"
RDNA3_OP_PROFILE_MAX_EVALS="${RDNA3_OP_PROFILE_MAX_EVALS:-}"
RDNA3_OP_PROFILE_MAX_ROWS="${RDNA3_OP_PROFILE_MAX_ROWS:-}"
RDNA3_OP_PROFILE_SUMMARY_ROWS="${RDNA3_OP_PROFILE_SUMMARY_ROWS:-}"
RDNA3_GRAPH_LOG="${RDNA3_GRAPH_LOG:-0}"
RDNA3_FAIL_ON_HOST_SYNC="${RDNA3_FAIL_ON_HOST_SYNC:-0}"
RDNA3_KERNEL_PRESET="${RDNA3_KERNEL_PRESET:-auto}"
RDNA3_DISABLE_QWEN_TOPK="${RDNA3_DISABLE_QWEN_TOPK:-${RDNA3_DISABLE_QWEN35_TOPK:-0}}"
RDNA3_DISABLE_QWEN35_TOPK="${RDNA3_DISABLE_QWEN35_TOPK:-$RDNA3_DISABLE_QWEN_TOPK}"
RDNA3_DISABLE_MOE_GATE_UP_FUSED="${RDNA3_DISABLE_MOE_GATE_UP_FUSED:-0}"
RDNA3_DISABLE_MOE_DOWN_FUSED="${RDNA3_DISABLE_MOE_DOWN_FUSED:-0}"
RDNA3_DISABLE_MOE_COMBINE="${RDNA3_DISABLE_MOE_COMBINE:-0}"
RDNA3_DISABLE_SET_ROWS_PAIR_FUSED="${RDNA3_DISABLE_SET_ROWS_PAIR_FUSED:-0}"
RDNA3_SET_ROWS_PAIR_LOG="${RDNA3_SET_ROWS_PAIR_LOG:-0}"
RDNA3_DISABLE_GDN_AR_TILED="${RDNA3_DISABLE_GDN_AR_TILED:-0}"
RDNA3_QWEN36_FASTPATH="${RDNA3_QWEN36_FASTPATH:-1}"
RDNA3_QWEN36_LINEAR_MMVQ_FAST="${RDNA3_QWEN36_LINEAR_MMVQ_FAST:-1}"
RDNA3_MMVQ_Q8_CACHE="${RDNA3_MMVQ_Q8_CACHE:-1}"
RDNA3_DISABLE_MMVQ_Q8_CACHE="${RDNA3_DISABLE_MMVQ_Q8_CACHE:-0}"
RDNA3_QWEN36_TOPK_RPB="${RDNA3_QWEN36_TOPK_RPB:-}"
RDNA3_TQKV_FATTN_TILE="${RDNA3_TQKV_FATTN_TILE:-0}"
RDNA3_TQKV_FATTN_TILE_MIN_CTX="${RDNA3_TQKV_FATTN_TILE_MIN_CTX:-}"
RDNA3_TQKV_FATTN_KQ_LANES="${RDNA3_TQKV_FATTN_KQ_LANES:-}"
RDNA3_TQKV_FATTN_GQA_DECODE="${RDNA3_TQKV_FATTN_GQA_DECODE:-1}"
RDNA3_TQKV_FATTN_GQA_HEADS="${RDNA3_TQKV_FATTN_GQA_HEADS:-4}"
RDNA3_MOE_MMVQ_RPB="${RDNA3_MOE_MMVQ_RPB:-}"
RDNA3_MOE_DOWN_RPB="${RDNA3_MOE_DOWN_RPB:-}"
RDNA3_MOE_GATE_UP_RPB="${RDNA3_MOE_GATE_UP_RPB:-}"
RDNA3_GDN_WARPS="${RDNA3_GDN_WARPS:-}"
RDNA3_GDN_AR_COLS="${RDNA3_GDN_AR_COLS:-}"
RDNA3_GDN_STATE_INPLACE="${RDNA3_GDN_STATE_INPLACE:-1}"
RDNA3_SSM_CONV_STATE_INPLACE="${RDNA3_SSM_CONV_STATE_INPLACE:-1}"

case "$RDNA3_KERNEL_PRESET" in
    auto)
        ;;
    attn-kq4)
        RDNA3_TQKV_FATTN_KQ_LANES="${RDNA3_TQKV_FATTN_KQ_LANES:-4}"
        ;;
    attn-kq8)
        RDNA3_TQKV_FATTN_KQ_LANES="${RDNA3_TQKV_FATTN_KQ_LANES:-8}"
        ;;
    decode-wide)
        RDNA3_TQKV_FATTN_KQ_LANES="${RDNA3_TQKV_FATTN_KQ_LANES:-4}"
        RDNA3_GDN_AR_COLS="${RDNA3_GDN_AR_COLS:-16}"
        RDNA3_MOE_MMVQ_RPB="${RDNA3_MOE_MMVQ_RPB:-16}"
        RDNA3_MOE_DOWN_RPB="${RDNA3_MOE_DOWN_RPB:-8}"
        RDNA3_MOE_GATE_UP_RPB="${RDNA3_MOE_GATE_UP_RPB:-8}"
        ;;
    *)
        echo "Unsupported RDNA3_KERNEL_PRESET=$RDNA3_KERNEL_PRESET" >&2
        echo "Allowed values: auto, attn-kq4, attn-kq8, decode-wide" >&2
        exit 1
        ;;
esac

case "$TQKV_PROFILE" in
    fast)
        CACHE_TYPE_K="tqkv_4_0"
        CACHE_TYPE_V="tqkv_4_0"
        PROFILE_NOTE="4-bit TQKV path, best speed/quality starting point"
        ;;
    compact)
        CACHE_TYPE_K="tqkv_2_0"
        CACHE_TYPE_V="tqkv_2_0"
        PROFILE_NOTE="2-bit KV, lowest VRAM, quality-risk experiment"
        ;;
    quality)
        CACHE_TYPE_K="tqkv_3_5_ip"
        CACHE_TYPE_V="tqkv_3_5_ip"
        PROFILE_NOTE="inner-product residual KV, slow diagnostic profile"
        ;;
    custom)
        CACHE_TYPE_K="${CACHE_TYPE_K:-tqkv_4_0}"
        CACHE_TYPE_V="${CACHE_TYPE_V:-tqkv_4_0}"
        PROFILE_NOTE="custom cache types from CACHE_TYPE_K/CACHE_TYPE_V"
        ;;
    *)
        echo "Unsupported TQKV_PROFILE=$TQKV_PROFILE" >&2
        echo "Allowed values: fast, compact, quality, custom" >&2
        exit 1
        ;;
esac

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

SERVER_EXTRA_ARGS=()
if [[ "$KV_UNIFIED" == "1" ]]; then
    SERVER_EXTRA_ARGS+=(--kv-unified)
elif [[ "$KV_UNIFIED" == "0" ]]; then
    SERVER_EXTRA_ARGS+=(--no-kv-unified)
fi

if [[ "$CACHE_IDLE_SLOTS" == "1" ]]; then
    SERVER_EXTRA_ARGS+=(--cache-idle-slots)
elif [[ "$CACHE_IDLE_SLOTS" == "0" ]]; then
    SERVER_EXTRA_ARGS+=(--no-cache-idle-slots)
fi

if [[ "$BACKEND_SAMPLING" == "1" ]]; then
    SERVER_EXTRA_ARGS+=(--backend-sampling)
fi

if [[ "$RDNA3_PROFILE_LOG" == "1" ]]; then
    export GGML_CUDA_RDNA3_PROFILE_LOG=1
fi

if [[ "$RDNA3_OP_PROFILE" == "1" ]]; then
    export GGML_CUDA_RDNA3_OP_PROFILE=1
    RDNA3_OP_PROFILE_MAX_EVALS="${RDNA3_OP_PROFILE_MAX_EVALS:-8}"
fi

if [[ -n "$RDNA3_OP_PROFILE_MAX_TOKENS" ]]; then
    export GGML_CUDA_RDNA3_OP_PROFILE_MAX_TOKENS="$RDNA3_OP_PROFILE_MAX_TOKENS"
fi

if [[ -n "$RDNA3_OP_PROFILE_MAX_EVALS" ]]; then
    export GGML_CUDA_RDNA3_OP_PROFILE_MAX_EVALS="$RDNA3_OP_PROFILE_MAX_EVALS"
fi

if [[ -n "$RDNA3_OP_PROFILE_MAX_ROWS" ]]; then
    export GGML_CUDA_RDNA3_OP_PROFILE_MAX_ROWS="$RDNA3_OP_PROFILE_MAX_ROWS"
fi

if [[ -n "$RDNA3_OP_PROFILE_SUMMARY_ROWS" ]]; then
    export GGML_CUDA_RDNA3_OP_PROFILE_SUMMARY_ROWS="$RDNA3_OP_PROFILE_SUMMARY_ROWS"
fi

if [[ "$RDNA3_GRAPH_LOG" == "1" ]]; then
    export GGML_CUDA_RDNA3_GRAPH_LOG=1
fi

if [[ "$RDNA3_FAIL_ON_HOST_SYNC" == "1" ]]; then
    export GGML_CUDA_RDNA3_FAIL_ON_HOST_SYNC=1
fi

if [[ "$RDNA3_DISABLE_QWEN_TOPK" == "1" || "$RDNA3_DISABLE_QWEN35_TOPK" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_QWEN_TOPK=1
    export GGML_CUDA_RDNA3_DISABLE_QWEN35_TOPK=1
fi

if [[ "$RDNA3_DISABLE_MOE_GATE_UP_FUSED" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_MOE_GATE_UP_FUSED=1
fi

if [[ "$RDNA3_DISABLE_MOE_DOWN_FUSED" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_MOE_DOWN_FUSED=1
fi

if [[ "$RDNA3_DISABLE_MOE_COMBINE" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_MOE_COMBINE=1
fi

if [[ "$RDNA3_DISABLE_SET_ROWS_PAIR_FUSED" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_SET_ROWS_PAIR_FUSED=1
fi

if [[ "$RDNA3_SET_ROWS_PAIR_LOG" == "1" ]]; then
    export GGML_CUDA_RDNA3_SET_ROWS_PAIR_LOG=1
fi

if [[ "$RDNA3_DISABLE_GDN_AR_TILED" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_GDN_AR_TILED=1
fi

if [[ "$RDNA3_QWEN36_FASTPATH" == "1" ]]; then
    export GGML_CUDA_RDNA3_QWEN36_FASTPATH=1
fi

if [[ "$RDNA3_QWEN36_LINEAR_MMVQ_FAST" == "1" ]]; then
    export GGML_CUDA_RDNA3_QWEN36_LINEAR_MMVQ_FAST=1
fi

if [[ "$RDNA3_MMVQ_Q8_CACHE" == "1" ]]; then
    export GGML_CUDA_RDNA3_MMVQ_Q8_CACHE=1
fi

if [[ "$RDNA3_DISABLE_MMVQ_Q8_CACHE" == "1" ]]; then
    export GGML_CUDA_RDNA3_DISABLE_MMVQ_Q8_CACHE=1
fi

if [[ -n "$RDNA3_QWEN36_TOPK_RPB" ]]; then
    export GGML_CUDA_RDNA3_QWEN36_TOPK_RPB="$RDNA3_QWEN36_TOPK_RPB"
fi

if [[ "$RDNA3_TQKV_FATTN_TILE" == "1" ]]; then
    export GGML_CUDA_RDNA3_TQKV_FATTN_TILE=1
fi

if [[ -n "$RDNA3_TQKV_FATTN_TILE_MIN_CTX" ]]; then
    export GGML_CUDA_RDNA3_TQKV_FATTN_TILE_MIN_CTX="$RDNA3_TQKV_FATTN_TILE_MIN_CTX"
fi

if [[ -n "$RDNA3_TQKV_FATTN_KQ_LANES" ]]; then
    export GGML_CUDA_RDNA3_TQKV_FATTN_KQ_LANES="$RDNA3_TQKV_FATTN_KQ_LANES"
fi

export GGML_CUDA_RDNA3_TQKV_FATTN_GQA_DECODE="$RDNA3_TQKV_FATTN_GQA_DECODE"
export GGML_CUDA_RDNA3_TQKV_FATTN_GQA_HEADS="$RDNA3_TQKV_FATTN_GQA_HEADS"

if [[ -n "$RDNA3_MOE_MMVQ_RPB" ]]; then
    export GGML_CUDA_RDNA3_MOE_MMVQ_RPB="$RDNA3_MOE_MMVQ_RPB"
fi

if [[ -n "$RDNA3_MOE_DOWN_RPB" ]]; then
    export GGML_CUDA_RDNA3_MOE_DOWN_RPB="$RDNA3_MOE_DOWN_RPB"
fi

if [[ -n "$RDNA3_MOE_GATE_UP_RPB" ]]; then
    export GGML_CUDA_RDNA3_MOE_GATE_UP_RPB="$RDNA3_MOE_GATE_UP_RPB"
fi

if [[ -n "$RDNA3_GDN_WARPS" ]]; then
    export GGML_CUDA_RDNA3_GDN_WARPS="$RDNA3_GDN_WARPS"
fi

if [[ -n "$RDNA3_GDN_AR_COLS" ]]; then
    export GGML_CUDA_RDNA3_GDN_AR_COLS="$RDNA3_GDN_AR_COLS"
fi

if [[ "$RDNA3_GDN_STATE_INPLACE" == "1" ]]; then
    export GGML_CUDA_RDNA3_GDN_STATE_INPLACE=1
fi

if [[ "$RDNA3_SSM_CONV_STATE_INPLACE" == "1" ]]; then
    export GGML_CUDA_RDNA3_SSM_CONV_STATE_INPLACE=1
fi

echo "Starting TQKV server"
echo "  model: $MODEL"
echo "  profile: $TQKV_PROFILE ($PROFILE_NOTE)"
echo "  cache: K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "  ctx:   $CTX_SIZE"
echo "  parallel slots: $PARALLEL"
echo "  batch: $BATCH_SIZE / ubatch $UBATCH_SIZE"
echo "  prompt cache RAM: ${PROMPT_CACHE_MB} MiB"
echo "  ctx checkpoints: $CTX_CHECKPOINTS"
echo "  checkpoint every n tokens: $CHECKPOINT_EVERY_NT"
echo "  backend sampling: $BACKEND_SAMPLING"
echo "  default sampling: temp=$SAMPLING_TEMP top-k=$SAMPLING_TOP_K top-p=$SAMPLING_TOP_P"
echo "  reasoning: mode=$REASONING_MODE format=$REASONING_FORMAT budget=$REASONING_BUDGET"
echo "  rdna3 profile log: $RDNA3_PROFILE_LOG"
echo "  rdna3 op profile: $RDNA3_OP_PROFILE"
echo "  rdna3 op profile max tokens: ${RDNA3_OP_PROFILE_MAX_TOKENS:-all}"
echo "  rdna3 op profile max evals: ${RDNA3_OP_PROFILE_MAX_EVALS:-all}"
echo "  rdna3 op profile max rows: ${RDNA3_OP_PROFILE_MAX_ROWS:-64}"
echo "  rdna3 op profile summary rows: ${RDNA3_OP_PROFILE_SUMMARY_ROWS:-16}"
echo "  rdna3 graph log: $RDNA3_GRAPH_LOG"
echo "  rdna3 fail on host sync: $RDNA3_FAIL_ON_HOST_SYNC"
echo "  rdna3 kernel preset: $RDNA3_KERNEL_PRESET"
echo "  rdna3 disable qwen topk: $RDNA3_DISABLE_QWEN_TOPK"
echo "  rdna3 disable moe gate_up fused: $RDNA3_DISABLE_MOE_GATE_UP_FUSED"
echo "  rdna3 disable moe down fused: $RDNA3_DISABLE_MOE_DOWN_FUSED"
echo "  rdna3 disable moe combine: $RDNA3_DISABLE_MOE_COMBINE"
echo "  rdna3 disable set rows pair fused: $RDNA3_DISABLE_SET_ROWS_PAIR_FUSED"
echo "  rdna3 set rows pair log: $RDNA3_SET_ROWS_PAIR_LOG"
echo "  rdna3 disable gdn ar tiled: $RDNA3_DISABLE_GDN_AR_TILED"
echo "  rdna3 qwen36 fastpath: $RDNA3_QWEN36_FASTPATH"
echo "  rdna3 qwen36 linear mmvq fast: $RDNA3_QWEN36_LINEAR_MMVQ_FAST"
echo "  rdna3 mmvq q8 cache: $RDNA3_MMVQ_Q8_CACHE"
echo "  rdna3 disable mmvq q8 cache: $RDNA3_DISABLE_MMVQ_Q8_CACHE"
echo "  rdna3 qwen36 topk rows/block: ${RDNA3_QWEN36_TOPK_RPB:-auto}"
echo "  rdna3 tqkv fattn tile: $RDNA3_TQKV_FATTN_TILE"
echo "  rdna3 tqkv fattn tile min ctx: ${RDNA3_TQKV_FATTN_TILE_MIN_CTX:-off}"
echo "  rdna3 tqkv fattn KQ lanes: ${RDNA3_TQKV_FATTN_KQ_LANES:-auto}"
echo "  rdna3 tqkv fattn GQA decode: $RDNA3_TQKV_FATTN_GQA_DECODE"
echo "  rdna3 tqkv fattn GQA heads/block: $RDNA3_TQKV_FATTN_GQA_HEADS"
echo "  rdna3 moe mmvq rows/block: ${RDNA3_MOE_MMVQ_RPB:-auto}"
echo "  rdna3 moe down rows/block: ${RDNA3_MOE_DOWN_RPB:-${RDNA3_MOE_MMVQ_RPB:-auto}}"
echo "  rdna3 moe gate_up rows/block: ${RDNA3_MOE_GATE_UP_RPB:-auto}"
echo "  rdna3 gated delta net warps: ${RDNA3_GDN_WARPS:-auto}"
echo "  rdna3 gated delta net ar cols/block: ${RDNA3_GDN_AR_COLS:-auto}"
echo "  rdna3 gated delta net state inplace: $RDNA3_GDN_STATE_INPLACE"
echo "  rdna3 ssm conv state inplace: $RDNA3_SSM_CONV_STATE_INPLACE"
echo "  bin:   $SERVER_BIN"
echo "  url:   http://$HOST:$PORT"

exec "$SERVER_BIN" \
    -m "$MODEL" \
    -a "$ALIAS" \
    -ngl all \
    -fa on \
    -np "$PARALLEL" \
    -cb \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --jinja \
    --reasoning "$REASONING_MODE" \
    --reasoning-format "$REASONING_FORMAT" \
    --reasoning-budget "$REASONING_BUDGET" \
    --temp "$SAMPLING_TEMP" \
    --top-k "$SAMPLING_TOP_K" \
    --top-p "$SAMPLING_TOP_P" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --cache-ram "$PROMPT_CACHE_MB" \
    --ctx-checkpoints "$CTX_CHECKPOINTS" \
    --checkpoint-every-n-tokens "$CHECKPOINT_EVERY_NT" \
    --metrics \
    --host "$HOST" \
    --port "$PORT" \
    "${SERVER_EXTRA_ARGS[@]}"
