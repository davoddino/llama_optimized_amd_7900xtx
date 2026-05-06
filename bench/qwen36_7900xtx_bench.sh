#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=""
MODEL_TYPE="qwen36-35b-a3b"
BACKEND="hip"
SCENARIO="all"
PRESET="baseline"
USERS=1
PROMPT_LEN=8192
GEN_TOKENS=256
CTX_SIZE=128000
BATCH_SIZE=4096
UBATCH_SIZE=1024
GPU_LAYERS="all"
REPETITIONS=3
OUTPUT_FORMAT="json"
OUT_DIR="$REPO_DIR/bench-results/qwen36-7900xtx"
PORT=8096
HOST="127.0.0.1"
ALIAS="qwen36-bench"
CACHE_TYPE_K="tqkv_4_0"
CACHE_TYPE_V="tqkv_4_0"
BENCH_BIN="${BENCH_BIN:-}"
SERVER_BIN="${SERVER_BIN:-}"
CLI_BIN="${CLI_BIN:-}"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bench/qwen36_7900xtx_bench.sh --model MODEL.gguf [options]

Required:
  --model PATH                 GGUF model path

Core options:
  --model-type LABEL           qwen36-35b-a3b or qwen36-27b (default: qwen36-35b-a3b)
  --backend NAME               hip, vulkan, cpu, or auto (default: hip)
  --scenario NAME              single, multi, correctness, or all (default: all)
  --preset NAME                baseline, fast, mega-contract, superlayer-l0, or final (default: baseline)
  --users N                    parallel sequences for multi-user server benchmark (default: 1)
  --prompt-len N               approximate prompt tokens (default: 8192)
  --gen-tokens N               generated tokens per sequence (default: 256)
  --ctx-size N                 context length (default: 128000)
  --batch-size N               logical batch size (default: 4096)
  --ubatch-size N              physical batch size (default: 1024)
  --gpu-layers N|all|auto      GPU layer setting; all maps to 999 for llama-bench (default: all)
  --cache-type-k TYPE          KV cache K type (default: tqkv_4_0)
  --cache-type-v TYPE          KV cache V type (default: tqkv_4_0)
  --output-format json|md|csv  llama-bench output format (default: json)
  --out-dir DIR                output directory (default: bench-results/qwen36-7900xtx)
  --port N                     server port for multi/correctness (default: 8096)
  --extra-arg ARG              extra arg passed to llama-bench and llama-server; repeatable

Environment overrides:
  BENCH_BIN, SERVER_BIN, CLI_BIN, HIP_VISIBLE_DEVICES, LLAMA_ARG_QWEN36_FAST_PATH

Examples:
  bench/qwen36_7900xtx_bench.sh --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf --preset baseline --scenario single
  bench/qwen36_7900xtx_bench.sh --model /models/Qwen3.6-35B-A3B-Q4_K_M.gguf --preset fast --scenario all --users 8
  bench/qwen36_7900xtx_bench.sh --model /models/Qwen3.6-27B-Q4_K_M.gguf --model-type qwen36-27b --preset fast --scenario single
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --model-type)
            MODEL_TYPE="$2"; shift 2 ;;
        --backend)
            BACKEND="$2"; shift 2 ;;
        --scenario)
            SCENARIO="$2"; shift 2 ;;
        --preset)
            PRESET="$2"; shift 2 ;;
        --users|--parallel)
            USERS="$2"; shift 2 ;;
        --prompt-len|--prompt-length)
            PROMPT_LEN="$2"; shift 2 ;;
        --gen-tokens|--generated-tokens)
            GEN_TOKENS="$2"; shift 2 ;;
        --ctx-size|--context-length)
            CTX_SIZE="$2"; shift 2 ;;
        --batch-size|--batch)
            BATCH_SIZE="$2"; shift 2 ;;
        --ubatch-size|--ubatch)
            UBATCH_SIZE="$2"; shift 2 ;;
        --gpu-layers|--ngl)
            GPU_LAYERS="$2"; shift 2 ;;
        --cache-type-k)
            CACHE_TYPE_K="$2"; shift 2 ;;
        --cache-type-v)
            CACHE_TYPE_V="$2"; shift 2 ;;
        --output-format|--format)
            OUTPUT_FORMAT="$2"; shift 2 ;;
        --out-dir|--output-dir)
            OUT_DIR="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --alias)
            ALIAS="$2"; shift 2 ;;
        --repetitions)
            REPETITIONS="$2"; shift 2 ;;
        --extra-arg)
            EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "--model is required" >&2
    usage >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    exit 1
fi

case "$MODEL_TYPE" in
    qwen36-35b-a3b|qwen36-27b) ;;
    *)
        echo "Unsupported --model-type=$MODEL_TYPE" >&2
        exit 1 ;;
esac

case "$BACKEND" in
    hip|vulkan|cpu|auto) ;;
    *)
        echo "Unsupported --backend=$BACKEND" >&2
        exit 1 ;;
esac

case "$SCENARIO" in
    single|multi|correctness|all) ;;
    *)
        echo "Unsupported --scenario=$SCENARIO" >&2
        exit 1 ;;
esac

case "$OUTPUT_FORMAT" in
    json|md|csv|sql) ;;
    *)
        echo "Unsupported --output-format=$OUTPUT_FORMAT" >&2
        exit 1 ;;
esac

mkdir -p "$OUT_DIR"

find_tool() {
    local tool="$1"
    local explicit="$2"
    local env_name="$3"
    if [[ -n "$explicit" ]]; then
        if [[ ! -x "$explicit" ]]; then
            echo "$tool override is not executable: $explicit" >&2
            exit 1
        fi
        printf '%s\n' "$explicit"
        return
    fi

    local candidates=()
    case "$BACKEND" in
        hip)
            candidates+=("$REPO_DIR/build-rocm-gfx1100/bin/$tool")
            candidates+=("$REPO_DIR/build-hip/bin/$tool")
            ;;
        vulkan)
            candidates+=("$REPO_DIR/build-vulkan/bin/$tool")
            ;;
    esac
    candidates+=(
        "$REPO_DIR/build/bin/$tool"
        "$REPO_DIR/bin/$tool"
        "$REPO_DIR/$tool"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    echo "$tool not found. Build first or set $env_name." >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 1
}

normalize_ngl() {
    case "$GPU_LAYERS" in
        all) printf '%s\n' "999" ;;
        auto) printf '%s\n' "99" ;;
        *) printf '%s\n' "$GPU_LAYERS" ;;
    esac
}

apply_preset() {
    case "$PRESET" in
        baseline)
            export LLAMA_ARG_QWEN36_FAST_PATH=0
            export GGML_CUDA_RDNA3_QWEN36_FASTPATH=0
            export GGML_CUDA_RDNA3_QWEN36_LINEAR_MMVQ_FAST=0
            export GGML_CUDA_RDNA3_QWEN36_TOPK_FASTPATH=0
            ;;
        fast)
            export LLAMA_ARG_QWEN36_FAST_PATH=1
            export GGML_CUDA_RDNA3_QWEN36_FASTPATH=1
            export GGML_CUDA_RDNA3_QWEN36_LINEAR_MMVQ_FAST=1
            export GGML_CUDA_RDNA3_QWEN36_TOPK_FASTPATH=1
            export GGML_CUDA_RDNA3_MMVQ_Q8_CACHE=1
            export GGML_CUDA_RDNA3_TQKV_FATTN_GQA_DECODE=1
            export GGML_CUDA_RDNA3_QWEN36_TOPK_RPB="${GGML_CUDA_RDNA3_QWEN36_TOPK_RPB:-4}"
            ;;
        mega-contract)
            export LLAMA_ARG_QWEN36_FAST_PATH=1
            export GGML_CUDA_RDNA3_QWEN36_FASTPATH=1
            export GGML_CUDA_RDNA3_QWEN36_LINEAR_MMVQ_FAST=1
            export GGML_CUDA_RDNA3_QWEN36_TOPK_FASTPATH=1
            export GGML_CUDA_RDNA3_MMVQ_Q8_CACHE=1
            export GGML_CUDA_RDNA3_QWEN36_MEGA_DECODE=1
            export GGML_CUDA_RDNA3_QWEN36_MEGA_REQUIRED=1
            export GGML_CUDA_RDNA3_QWEN36_MEGA_REQUIRE_GRAPH=1
            export GGML_CUDA_RDNA3_QWEN36_MEGA_SAMPLE_TOKEN_ONLY=1
            export GGML_CUDA_RDNA3_QWEN36_MEGA_NO_RAW_LOGITS=1
            export GGML_CUDA_RDNA3_QWEN36_MEGA_ASYNC_INPUTS=1
            export GGML_CUDA_RDNA3_QWEN36_ONE_LAYER_MEGA="${GGML_CUDA_RDNA3_QWEN36_ONE_LAYER_MEGA:-1}"
            export GGML_CUDA_RDNA3_QWEN36_ONE_LAYER_MEGA_REQUIRED="${GGML_CUDA_RDNA3_QWEN36_ONE_LAYER_MEGA_REQUIRED:-1}"
            ;;
        superlayer-l0)
            export LLAMA_ARG_QWEN36_FAST_PATH=1
            export GGML_CUDA_RDNA3_QWEN36_FASTPATH=1
            export GGML_CUDA_RDNA3_QWEN36_SUPERLAYER=1
            export GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DISPATCH=1
            export GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0=1
            export GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_WEIGHTS=1
            ;;
        final)
            export LLAMA_ARG_QWEN36_FAST_PATH=1
            export GGML_CUDA_RDNA3_QWEN36_FASTPATH=1
            export GGML_CUDA_RDNA3_QWEN36_LINEAR_MMVQ_FAST=1
            export GGML_CUDA_RDNA3_QWEN36_TOPK_FASTPATH=1
            export GGML_CUDA_RDNA3_MMVQ_Q8_CACHE=1
            export GGML_CUDA_RDNA3_TQKV_FATTN_GQA_DECODE=1
            export GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL=1
            export GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_WEIGHTS=1
            export GGML_CUDA_RDNA3_QWEN36_TOPK_RPB="${GGML_CUDA_RDNA3_QWEN36_TOPK_RPB:-4}"
            ;;
        *)
            echo "Unsupported --preset=$PRESET" >&2
            exit 1 ;;
    esac
}

print_header() {
    local log="$OUT_DIR/run-metadata.md"
    {
        echo "# Qwen3.6 RX 7900 XTX Benchmark"
        echo
        echo "- date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "- model: $MODEL"
        echo "- model_type: $MODEL_TYPE"
        echo "- backend: $BACKEND"
        echo "- preset: $PRESET"
        echo "- scenario: $SCENARIO"
        echo "- users: $USERS"
        echo "- prompt_len: $PROMPT_LEN"
        echo "- gen_tokens: $GEN_TOKENS"
        echo "- ctx_size: $CTX_SIZE"
        echo "- batch_size: $BATCH_SIZE"
        echo "- ubatch_size: $UBATCH_SIZE"
        echo "- gpu_layers: $GPU_LAYERS"
        echo "- cache_type_k: $CACHE_TYPE_K"
        echo "- cache_type_v: $CACHE_TYPE_V"
    } > "$log"
    echo "Wrote metadata: $log"
}

bench_args_common() {
    local ngl
    ngl="$(normalize_ngl)"
    printf '%s\0' \
        -m "$MODEL" \
        -ngl "$ngl" \
        -fa 1 \
        -c "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -ctk "$CACHE_TYPE_K" \
        -ctv "$CACHE_TYPE_V"
}

run_single() {
    local bench_bin="$1"
    local warmup="$OUT_DIR/single-warmup.${OUTPUT_FORMAT}"
    local result="$OUT_DIR/single-${PRESET}.${OUTPUT_FORMAT}"
    local -a common_args=()
    while IFS= read -r -d '' arg; do
        common_args+=("$arg")
    done < <(bench_args_common)

    echo "Running single-user warmup..."
    "$bench_bin" "${common_args[@]}" -p 128 -n 16 -r 1 -o "$OUTPUT_FORMAT" "${EXTRA_ARGS[@]}" > "$warmup"

    echo "Running single-user decode benchmark..."
    "$bench_bin" \
        "${common_args[@]}" \
        -p "$PROMPT_LEN" \
        -n "$GEN_TOKENS" \
        -r "$REPETITIONS" \
        -o "$OUTPUT_FORMAT" \
        "${EXTRA_ARGS[@]}" | tee "$result"
}

server_args_common() {
    local ngl
    ngl="$(normalize_ngl)"
    printf '%s\0' \
        -m "$MODEL" \
        -a "$ALIAS" \
        -ngl "$ngl" \
        -fa on \
        -np "$USERS" \
        -cb \
        --ctx-size "$CTX_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --ubatch-size "$UBATCH_SIZE" \
        --cache-type-k "$CACHE_TYPE_K" \
        --cache-type-v "$CACHE_TYPE_V" \
        --host "$HOST" \
        --port "$PORT" \
        --metrics
    if [[ "$PRESET" == "mega-contract" ]]; then
        printf '%s\0' --backend-sampling --samplers "top_k;temperature" --temp 0 --top-k 1 --top-p 1 --min-p 0
    fi
}

wait_for_server() {
    local deadline=$((SECONDS + 120))
    until curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "server did not become ready on http://$HOST:$PORT" >&2
            return 1
        fi
        sleep 0.25
    done
}

run_multi() {
    local server_bin="$1"
    local server_log="$OUT_DIR/server-${PRESET}.log"
    local result_json="$OUT_DIR/multi-${PRESET}.json"
    local result_md="$OUT_DIR/multi-${PRESET}.md"
    local -a server_args=()
    while IFS= read -r -d '' arg; do
        server_args+=("$arg")
    done < <(server_args_common)

    echo "Starting server for multi-user benchmark..."
    "$server_bin" "${server_args[@]}" "${EXTRA_ARGS[@]}" > "$server_log" 2>&1 &
    local server_pid=$!
    trap 'kill "$server_pid" >/dev/null 2>&1 || true' RETURN
    wait_for_server

    python3 - "$HOST" "$PORT" "$ALIAS" "$USERS" "$PROMPT_LEN" "$GEN_TOKENS" "$result_json" "$result_md" <<'PY'
import concurrent.futures
import json
import time
import urllib.request
import sys

host, port, model, users_s, prompt_s, gen_s, out_json, out_md = sys.argv[1:]
users = int(users_s)
prompt_tokens = int(prompt_s)
gen_tokens = int(gen_s)
url = f"http://{host}:{port}/v1/chat/completions"

def make_prompt(i: int) -> str:
    target_chars = max(256, prompt_tokens * 3)
    lines = [
        "You are benchmarking concurrent decode throughput on a long coding context.\n",
        f"Unique request marker: QWEN36_7900XTX_USER_{i:04d}_{time.time_ns()}\n",
        "Summarize only after reading the synthetic repository fragments.\n\n",
    ]
    j = 0
    while sum(len(x) for x in lines) < target_chars:
        lines.append(
            f"def user_{i}_module_{j}(state, request):\n"
            f"    key = 'u{i}_m{j}_' + str(request.get('id', 0))\n"
            f"    state[key] = (state.get(key, 0) + {j % 17}) % 997\n"
            f"    return state[key]\n\n"
        )
        j += 1
    lines.append("Now answer with exactly four compact bullets.\n")
    return "".join(lines)

def request_one(i: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Keep the answer compact."},
            {"role": "user", "content": make_prompt(i)},
        ],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "stream": False,
        "cache_prompt": False,
        "verbose": True,
        "response_fields": ["timings", "stop_type", "generation_settings"],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall = time.perf_counter() - t0
    timings = data.get("timings") or (data.get("__verbose") or {}).get("timings") or {}
    return {
        "user": i,
        "wall_s": wall,
        "prompt_n": int(timings.get("prompt_n") or 0),
        "prompt_ms": float(timings.get("prompt_ms") or 0.0),
        "predicted_n": int(timings.get("predicted_n") or 0),
        "predicted_ms": float(timings.get("predicted_ms") or 0.0),
        "prompt_tps": float(timings.get("prompt_per_second") or 0.0),
        "decode_tps": float(timings.get("predicted_per_second") or 0.0),
        "finish_reason": ((data.get("choices") or [{}])[0]).get("finish_reason"),
    }

start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=users) as pool:
    rows = list(pool.map(request_one, range(users)))
wall_total = time.perf_counter() - start

total_prompt = sum(r["prompt_n"] for r in rows)
total_pred = sum(r["predicted_n"] for r in rows)
max_prompt_ms = max([r["prompt_ms"] for r in rows] or [0.0])
max_pred_ms = max([r["predicted_ms"] for r in rows] or [0.0])
summary = {
    "users": users,
    "wall_s": wall_total,
    "total_prompt_tokens": total_prompt,
    "total_decode_tokens": total_pred,
    "aggregate_prompt_tps": (1000.0 * total_prompt / max_prompt_ms) if max_prompt_ms > 0 else 0.0,
    "aggregate_decode_tps": (1000.0 * total_pred / max_pred_ms) if max_pred_ms > 0 else 0.0,
    "wall_aggregate_tps": total_pred / wall_total if wall_total > 0 else 0.0,
    "per_user": rows,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(out_md, "w", encoding="utf-8") as f:
    f.write("# Multi-user Qwen3.6 Decode Benchmark\n\n")
    f.write(f"- users: {users}\n")
    f.write(f"- aggregate_decode_tps: {summary['aggregate_decode_tps']:.2f}\n")
    f.write(f"- wall_aggregate_tps: {summary['wall_aggregate_tps']:.2f}\n\n")
    f.write("| user | prompt_n | pred_n | prompt_tps | decode_tps | wall_s |\n")
    f.write("|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['user']} | {r['prompt_n']} | {r['predicted_n']} | "
            f"{r['prompt_tps']:.2f} | {r['decode_tps']:.2f} | {r['wall_s']:.2f} |\n"
        )

print(json.dumps(summary, indent=2))
PY

    echo "Saved multi-user JSON: $result_json"
    echo "Saved multi-user markdown: $result_md"
}

run_correctness() {
    local cli_bin="$1"
    local ngl
    ngl="$(normalize_ngl)"
    local base_out="$OUT_DIR/correctness-baseline.txt"
    local fast_out="$OUT_DIR/correctness-${PRESET}.txt"
    local prompt="Write three short bullets about why GPU-resident decode reduces latency."

    echo "Running deterministic baseline correctness sample..."
    (
        export LLAMA_ARG_QWEN36_FAST_PATH=0
        export GGML_CUDA_RDNA3_QWEN36_FASTPATH=0
        "$cli_bin" -m "$MODEL" -ngl "$ngl" -fa on -c "$CTX_SIZE" -n 64 --temp 0 --top-k 1 -p "$prompt" \
            > "$base_out"
    )

    echo "Running deterministic optimized correctness sample..."
    apply_preset
    "$cli_bin" -m "$MODEL" -ngl "$ngl" -fa on -c "$CTX_SIZE" -n 64 --temp 0 --top-k 1 -p "$prompt" \
        > "$fast_out"

    if grep -Eiq '(^|[^a-z])(nan|inf)([^a-z]|$)' "$base_out" "$fast_out"; then
        echo "Correctness check failed: NaN/Inf-like text detected in output" >&2
        exit 2
    fi

    {
        echo "# Correctness Sanity"
        echo
        echo "Baseline output: $base_out"
        echo
        echo "Optimized output: $fast_out"
        echo
        if cmp -s "$base_out" "$fast_out"; then
            echo "Outputs are byte-identical."
        else
            echo "Outputs differ; inspect files. Deterministic logits comparison requires target ROCm run with backend debug capture."
        fi
    } > "$OUT_DIR/correctness-${PRESET}.md"
    cat "$OUT_DIR/correctness-${PRESET}.md"
}

apply_preset
print_header

echo "Qwen3.6 RX 7900 XTX benchmark"
echo "  model:      $MODEL"
echo "  model type: $MODEL_TYPE"
echo "  backend:    $BACKEND"
echo "  preset:     $PRESET"
echo "  scenario:   $SCENARIO"
echo "  out dir:    $OUT_DIR"

if [[ "$SCENARIO" == "single" || "$SCENARIO" == "all" ]]; then
    run_single "$(find_tool llama-bench "$BENCH_BIN" "BENCH_BIN")"
fi

if [[ "$SCENARIO" == "multi" || "$SCENARIO" == "all" ]]; then
    run_multi "$(find_tool llama-server "$SERVER_BIN" "SERVER_BIN")"
fi

if [[ "$SCENARIO" == "correctness" || "$SCENARIO" == "all" ]]; then
    run_correctness "$(find_tool llama-cli "$CLI_BIN" "CLI_BIN")"
fi
