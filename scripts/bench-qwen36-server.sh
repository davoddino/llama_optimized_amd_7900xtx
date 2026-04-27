#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
MODEL="${MODEL:-qwen3.6-35b}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUNS="${RUNS:-3}"
OUT_DIR="${OUT_DIR:-bench-results/server}"
CACHE_BUST="${CACHE_BUST:-1}"

command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

mkdir -p "$OUT_DIR"

echo "Server benchmark"
echo "  url:           $BASE_URL"
echo "  model:         $MODEL"
echo "  prompt approx: $PROMPT_TOKENS tokens"
echo "  max tokens:    $MAX_TOKENS"
echo "  runs:          $RUNS"
echo "  output dir:    $OUT_DIR"
echo
printf "%-5s %-10s %-10s %-10s %-12s %-12s %-10s\n" \
    "run" "prompt_n" "cache_n" "pred_n" "prompt_t/s" "decode_t/s" "wall_s"

for run in $(seq 1 "$RUNS"); do
    body="$(mktemp)"
    response="$OUT_DIR/server-run-${run}-prompt${PROMPT_TOKENS}-gen${MAX_TOKENS}.json"

    python3 - "$body" "$MODEL" "$PROMPT_TOKENS" "$MAX_TOKENS" "$run" "$CACHE_BUST" <<'PY'
import json
import sys

path, model, prompt_tokens_s, max_tokens_s, run_s, cache_bust_s = sys.argv[1:]
prompt_tokens = int(prompt_tokens_s)
max_tokens = int(max_tokens_s)
run = int(run_s)
cache_bust = cache_bust_s != "0"

# Rough token budget. Code-like text is intentionally used because the target
# workload is coding-agent context, not natural-language filler.
target_chars = max(256, prompt_tokens * 4)
unique = f"BENCH_RUN_{run:04d}" if cache_bust else "BENCH_REUSED_PREFIX"

line = (
    "def transform_{i}(state, request, cache):\n"
    "    key = f'{unique}_module_{{i}}_{{request.user_id}}'\n"
    "    if key not in cache:\n"
    "        cache[key] = {{'items': [], 'errors': [], 'latency_ms': i % 37}}\n"
    "    return cache[key], state.get('feature_flags', {{}})\n\n"
)

chunks = [
    "You are benchmarking a long-context coding-agent session.\n",
    f"Unique marker near the beginning to avoid KV prefix reuse: {unique}\n",
    "Analyze the following synthetic repository dump. At the end, answer with exactly five short bullets about likely performance bottlenecks.\n\n",
]

i = 0
while sum(len(c) for c in chunks) < target_chars:
    chunks.append(line.format(i=i, unique=unique))
    i += 1

chunks.append("\nNow answer with exactly five short bullets. Do not repeat the code.\n")
prompt = "".join(chunks)

payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": "You are a precise coding assistant. Keep the final answer compact.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    "max_tokens": max_tokens,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "stream": False,
    "timings_per_token": False,
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

    wall_s="$(
        curl -sS --fail \
            -H 'Content-Type: application/json' \
            -o "$response" \
            -w '%{time_total}' \
            "$BASE_URL/v1/chat/completions" \
            -d @"$body"
    )" || {
        status=$?
        echo "Request failed for run $run. Response, if any, is in: $response" >&2
        rm -f "$body"
        exit "$status"
    }

    rm -f "$body"

    python3 - "$response" "$run" "$wall_s" <<'PY'
import json
import sys

path, run_s, wall_s = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

timings = data.get("timings") or {}
prompt_n = timings.get("prompt_n", 0)
cache_n = timings.get("cache_n", 0)
pred_n = timings.get("predicted_n", 0)
prompt_tps = timings.get("prompt_per_second", 0.0)
decode_tps = timings.get("predicted_per_second", 0.0)

print(f"{int(run_s):<5d} {prompt_n:<10} {cache_n:<10} {pred_n:<10} {prompt_tps:<12.2f} {decode_tps:<12.2f} {float(wall_s):<10.2f}")
PY
done

echo
echo "Saved raw responses in: $OUT_DIR"
