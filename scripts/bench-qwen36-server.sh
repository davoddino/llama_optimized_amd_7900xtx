#!/usr/bin/env bash
set -euo pipefail

BENCH_PORT="${BENCH_PORT:-${TQKV_PORT:-8002}}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$BENCH_PORT}"
MODEL="${MODEL:-qwen3.6-35b}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUNS="${RUNS:-3}"
OUT_DIR="${OUT_DIR:-bench-results/server}"
CACHE_BUST="${CACHE_BUST:-1}"
PROMPT_CHARS_PER_TOKEN="${PROMPT_CHARS_PER_TOKEN:-3}"
EXACT_DECODE="${EXACT_DECODE:-1}"
MIN_PREDICTED_TOKENS="${MIN_PREDICTED_TOKENS:-$MAX_TOKENS}"
REQUIRE_STOP_LIMIT="${REQUIRE_STOP_LIMIT:-1}"
ALLOW_CACHE="${ALLOW_CACHE:-0}"
ALLOW_DRAFT="${ALLOW_DRAFT:-0}"
ALLOW_BACKEND_SAMPLING="${ALLOW_BACKEND_SAMPLING:-0}"
ALLOW_OP_PROFILE="${ALLOW_OP_PROFILE:-0}"

command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

if [[ "$EXACT_DECODE" != "0" && "$ALLOW_OP_PROFILE" == "0" ]]; then
    if [[ "${RDNA3_OP_PROFILE:-0}" != "0" || "${GGML_CUDA_RDNA3_OP_PROFILE:-0}" != "0" ]]; then
        echo "Exact decode benchmark refuses op profiling overhead; set ALLOW_OP_PROFILE=1 to override." >&2
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"

echo "Server benchmark"
echo "  url:           $BASE_URL"
echo "  model:         $MODEL"
echo "  prompt approx: $PROMPT_TOKENS tokens"
echo "  chars/token:   $PROMPT_CHARS_PER_TOKEN"
echo "  max tokens:    $MAX_TOKENS"
echo "  runs:          $RUNS"
echo "  output dir:    $OUT_DIR"
echo "  exact gate:    $EXACT_DECODE"
echo "  min pred:      $MIN_PREDICTED_TOKENS"
echo
printf "%-5s %-10s %-10s %-10s %-12s %-12s %-12s %-10s\n" \
    "run" "prompt_n" "cache_n" "pred_n" "prompt_t/s" "decode_t/s" "decode_c/s" "wall_s"

for run in $(seq 1 "$RUNS"); do
    body="$(mktemp)"
    response="$OUT_DIR/server-run-${run}-prompt${PROMPT_TOKENS}-gen${MAX_TOKENS}.json"

    python3 - "$body" "$MODEL" "$PROMPT_TOKENS" "$MAX_TOKENS" "$run" "$CACHE_BUST" "$PROMPT_CHARS_PER_TOKEN" "$EXACT_DECODE" <<'PY'
import json
import sys

path, model, prompt_tokens_s, max_tokens_s, run_s, cache_bust_s, chars_per_token_s, exact_decode_s = sys.argv[1:]
prompt_tokens = int(prompt_tokens_s)
max_tokens = int(max_tokens_s)
run = int(run_s)
cache_bust = cache_bust_s != "0"
chars_per_token = float(chars_per_token_s)
exact_decode = exact_decode_s != "0"

# Rough token budget. Code-like text is intentionally used because the target
# workload is coding-agent context, not natural-language filler.
target_chars = max(256, int(prompt_tokens * chars_per_token))
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
    "verbose": exact_decode,
}

if exact_decode:
    payload["cache_prompt"] = False
    payload["response_fields"] = [
        "stop_type",
        "tokens_cached",
        "timings",
        "generation_settings",
    ]

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

    python3 - "$response" "$run" "$wall_s" "$EXACT_DECODE" "$MIN_PREDICTED_TOKENS" "$REQUIRE_STOP_LIMIT" "$ALLOW_CACHE" "$ALLOW_DRAFT" "$ALLOW_BACKEND_SAMPLING" <<'PY'
import json
import sys

(
    path,
    run_s,
    wall_s,
    exact_decode_s,
    min_predicted_tokens_s,
    require_stop_limit_s,
    allow_cache_s,
    allow_draft_s,
    allow_backend_sampling_s,
) = sys.argv[1:]

exact_decode = exact_decode_s != "0"
min_predicted_tokens = int(min_predicted_tokens_s)
require_stop_limit = require_stop_limit_s != "0"
allow_cache = allow_cache_s != "0"
allow_draft = allow_draft_s != "0"
allow_backend_sampling = allow_backend_sampling_s != "0"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

timings = data.get("timings") or {}
verbose = data.get("__verbose") or {}
verbose_timings = verbose.get("timings") or {}
generation_settings = verbose.get("generation_settings") or {}

if not timings and verbose_timings:
    timings = verbose_timings

prompt_n = timings.get("prompt_n", 0)
cache_n = timings.get("cache_n", 0)
pred_n = timings.get("predicted_n", 0)
pred_ms = timings.get("predicted_ms", 0.0)
prompt_tps = timings.get("prompt_per_second", 0.0)
decode_tps = timings.get("predicted_per_second", 0.0)
decode_cons_n = timings.get("predicted_conservative_n")
decode_cons_tps = timings.get("predicted_conservative_per_second")

try:
    pred_n_i = int(pred_n or 0)
except (TypeError, ValueError):
    pred_n_i = 0

try:
    pred_ms_f = float(pred_ms or 0.0)
except (TypeError, ValueError):
    pred_ms_f = 0.0

if decode_cons_n is None:
    decode_cons_n = max(0, pred_n_i - 1)
if decode_cons_tps is None:
    decode_cons_tps = 0.0
    if pred_ms_f > 0.0 and int(decode_cons_n or 0) > 0:
        decode_cons_tps = 1000.0 * int(decode_cons_n) / pred_ms_f

usage = data.get("usage") or {}
prompt_details = usage.get("prompt_tokens_details") or {}
cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)

choices = data.get("choices") or []
finish_reason = ""
if choices:
    finish_reason = choices[0].get("finish_reason") or ""

stop_type = verbose.get("stop_type")
tokens_cached = int(verbose.get("tokens_cached", 0) or 0)
draft_n = int(timings.get("draft_n", 0) or verbose_timings.get("draft_n", 0) or 0)
spec_type = generation_settings.get("speculative.type")
backend_sampling = bool(generation_settings.get("backend_sampling", False))

errors = []
if exact_decode:
    if pred_n_i < min_predicted_tokens:
        errors.append(f"predicted_n={pred_n_i} < MIN_PREDICTED_TOKENS={min_predicted_tokens}")
    if not allow_cache and (int(cache_n or 0) != 0 or cached_tokens != 0 or tokens_cached != 0):
        errors.append(
            "cache reuse detected "
            f"(timings.cache_n={cache_n}, usage.cached_tokens={cached_tokens}, verbose.tokens_cached={tokens_cached})"
        )
    if not allow_draft and draft_n != 0:
        errors.append(f"speculative draft tokens detected (draft_n={draft_n})")
    if not allow_draft and spec_type not in (None, "none"):
        errors.append(f"speculative mode is active (speculative.type={spec_type})")
    if require_stop_limit:
        if stop_type is not None and stop_type != "limit":
            errors.append(f"stop_type={stop_type!r}, expected 'limit'")
        elif stop_type is None and finish_reason != "length":
            errors.append(f"finish_reason={finish_reason!r}, expected 'length'")
    if not allow_backend_sampling and backend_sampling:
        errors.append("backend sampling is active")

if errors:
    print(f"Run {run_s} failed exact-decode validation:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    print(f"Raw response: {path}", file=sys.stderr)
    sys.exit(2)

print(
    f"{int(run_s):<5d} {prompt_n:<10} {cache_n:<10} {pred_n:<10} "
    f"{float(prompt_tps or 0.0):<12.2f} {float(decode_tps or 0.0):<12.2f} "
    f"{float(decode_cons_tps or 0.0):<12.2f} {float(wall_s):<10.2f}"
)
PY
done

echo
echo "Saved raw responses in: $OUT_DIR"
