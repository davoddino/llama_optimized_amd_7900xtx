#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:${BENCH_PORT:-8004}}"
MODEL="${MODEL:-qwen3.6-35b}"
STREAMS="${STREAMS:-4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32768}"
MAX_TOKENS="${MAX_TOKENS:-512}"
OUT_DIR="${OUT_DIR:-bench-results/rdna3-multi-4x32k}"
PROMPT_CHARS_PER_TOKEN="${PROMPT_CHARS_PER_TOKEN:-3}"

command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

mkdir -p "$OUT_DIR"

python3 - "$BASE_URL" "$MODEL" "$STREAMS" "$PROMPT_TOKENS" "$MAX_TOKENS" "$OUT_DIR" "$PROMPT_CHARS_PER_TOKEN" <<'PY'
import concurrent.futures
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

base_url, model, streams_s, prompt_tokens_s, max_tokens_s, out_dir_s, chars_per_token_s = sys.argv[1:]
streams = int(streams_s)
prompt_tokens = int(prompt_tokens_s)
max_tokens = int(max_tokens_s)
chars_per_token = float(chars_per_token_s)
out_dir = pathlib.Path(out_dir_s)
out_dir.mkdir(parents=True, exist_ok=True)

def make_prompt(stream_id: int) -> str:
    target_chars = max(256, int(prompt_tokens * chars_per_token))
    unique = f"RDNA3_MULTI_STREAM_{stream_id:02d}_{time.time_ns()}"
    line = (
        "def route_{stream_id}_{i}(repo, request, cache):\n"
        "    key = f'{unique}_{{request.user_id}}_{{i}}'\n"
        "    if key not in cache:\n"
        "        cache[key] = {{'files': [], 'warnings': [], 'score': i % 97}}\n"
        "    return cache[key], repo.get('symbols', {{}})\n\n"
    )
    chunks = [
        "You are benchmarking a concurrent long-context coding-agent session.\n",
        f"Unique stream marker: {unique}\n",
        "Analyze this synthetic repository dump. At the end, answer with exactly five short bullets.\n\n",
    ]
    i = 0
    while sum(len(c) for c in chunks) < target_chars:
        chunks.append(line.format(stream_id=stream_id, i=i, unique=unique))
        i += 1
    chunks.append("\nNow answer with exactly five short bullets. Do not repeat the code.\n")
    return "".join(chunks)

def run_one(stream_id: int):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise coding assistant. Keep the final answer compact."},
            {"role": "user", "content": make_prompt(stream_id)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "stream": False,
        "timings_per_token": False,
    }

    req_path = out_dir / f"multi-stream-{stream_id:02d}-request.json"
    res_path = out_dir / f"multi-stream-{stream_id:02d}-response.json"
    req_path.write_text(json.dumps(payload), encoding="utf-8")

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as err:
        body = err.read()
        status = err.code
    t1 = time.perf_counter()

    res_path.write_bytes(body)
    if status >= 400:
        raise RuntimeError(f"stream {stream_id} HTTP {status}; response saved to {res_path}")

    parsed = json.loads(body.decode("utf-8"))
    timings = parsed.get("timings") or {}
    return {
        "stream": stream_id,
        "wall_s": t1 - t0,
        "prompt_n": int(timings.get("prompt_n", 0) or 0),
        "cache_n": int(timings.get("cache_n", 0) or 0),
        "pred_n": int(timings.get("predicted_n", 0) or 0),
        "prompt_tps": float(timings.get("prompt_per_second", 0.0) or 0.0),
        "decode_tps": float(timings.get("predicted_per_second", 0.0) or 0.0),
        "decode_ms": float(timings.get("predicted_ms", 0.0) or 0.0),
    }

print("RDNA3 multi-stream benchmark")
print(f"  url:           {base_url}")
print(f"  model:         {model}")
print(f"  streams:       {streams}")
print(f"  prompt approx: {prompt_tokens} tokens each")
print(f"  max tokens:    {max_tokens} each")
print(f"  output dir:    {out_dir}")
print()

start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as executor:
    futures = [executor.submit(run_one, i) for i in range(streams)]
    rows = [future.result() for future in concurrent.futures.as_completed(futures)]
end = time.perf_counter()

rows.sort(key=lambda row: row["stream"])
print(f"{'stream':<7} {'prompt_n':<10} {'cache_n':<10} {'pred_n':<8} {'prompt_t/s':<12} {'decode_t/s':<12} {'wall_s':<10}")
for row in rows:
    print(
        f"{row['stream']:<7d} {row['prompt_n']:<10d} {row['cache_n']:<10d} {row['pred_n']:<8d} "
        f"{row['prompt_tps']:<12.2f} {row['decode_tps']:<12.2f} {row['wall_s']:<10.2f}"
    )

total_pred = sum(row["pred_n"] for row in rows)
total_wall = end - start
max_decode_s = max((row["decode_ms"] for row in rows), default=0.0) / 1000.0
wall_tps = total_pred / total_wall if total_wall > 0 else 0.0
decode_tps_est = total_pred / max_decode_s if max_decode_s > 0 else 0.0

summary = {
    "streams": streams,
    "prompt_tokens_requested": prompt_tokens,
    "max_tokens_requested": max_tokens,
    "total_predicted_tokens": total_pred,
    "wall_seconds": total_wall,
    "aggregate_wall_tokens_per_second": wall_tps,
    "aggregate_decode_tokens_per_second_est": decode_tps_est,
    "rows": rows,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print()
print(f"aggregate wall tok/s:       {wall_tps:.2f}")
print(f"aggregate decode tok/s est: {decode_tps_est:.2f}")
print(f"summary: {out_dir / 'summary.json'}")
PY
