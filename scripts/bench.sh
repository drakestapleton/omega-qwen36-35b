#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Reproducible DGX Spark GB10 benchmark suite for Qwen 3.6 35B-A3B + Full-Chip APU.
# Must be run on DGX Spark inside an exclusive quiet spark-window:
#   WID=$(spark-window open omega-qwen36-35b-bench --lease 15 --gb10 exclusive --quiet exclusive --wait | tail -n 1)
#   trap 'spark-window close "$WID" >/dev/null 2>&1 || true' EXIT
#   spark-window run spark-receipt omega-qwen36-35b-bench -- sh scripts/bench.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN="${BIN_DIR:-$ROOT/build/q36}"
MODEL="${Q36_MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0}"
PORT="${Q36_BENCH_PORT:-18192}"
OUT="${BENCH_OUT:-$ROOT/build/bench-output.txt}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

log() { echo "$1" | tee -a "$OUT"; }

log "=== ENVIRONMENT ==="
{
  date -u +%FT%TZ
  uname -a
  nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,temperature.gpu,power.draw --format=csv
} | tee -a "$OUT"

log "=== 1. FULL-CHIP GB10 APU (20 CPU + 48 SM121 SMs) ==="
"$BIN/apu-fullchip" 2>&1 | tee -a "$OUT"
sleep 5

log "=== 2. FULL-W4 DECODE + PARITY (24 tok) ==="
"$BIN/q36-decode" "$MODEL" 24 w4 2>&1 | tee -a "$OUT"
sleep 8

log "=== 3. FULL-W4 + NATIVE MTP M=4 DECODE (64 tok) ==="
"$BIN/q36-decode" "$MODEL" 64 mtp 2>&1 | tee -a "$OUT"
sleep 8

log "=== 4. FULL-W4 CONTINUOUS BATCH-4 DECODE (64 tok) ==="
"$BIN/q36-decode" "$MODEL" 64 batch4 2>&1 | tee -a "$OUT"
sleep 8

log "=== 5. FULL-W4 CONTINUOUS BATCH-8 DECODE (64 tok) ==="
"$BIN/q36-decode" "$MODEL" 64 batch8 2>&1 | tee -a "$OUT"
sleep 8

log "=== 6. PREFILL 1024 TOKENS (HMMA FA2 + LIVE ARBORIS) ==="
"$BIN/q36-decode" "$MODEL" 1024 p 2>&1 | tee -a "$OUT"
sleep 8

log "=== 7. PREFILL 4096 TOKENS (HMMA FA2 + LIVE ARBORIS) ==="
"$BIN/q36-decode" "$MODEL" 4096 p 2>&1 | tee -a "$OUT"
sleep 8

log "=== 8. PREFILL 16384 TOKENS (HMMA FA2 + LIVE ARBORIS) ==="
"$BIN/q36-decode" "$MODEL" 16384 p 2>&1 | tee -a "$OUT"
sleep 8

log "=== 9. EXACT FP8 DECODE + PARITY (24 tok) ==="
"$BIN/q36-decode" "$MODEL" 24 2>&1 | tee -a "$OUT"
sleep 8

log "=== 10. LIVE HTTP PROSE DECODE & NEEDLE RECALL (q36-serve Full-W4 + Live Arboris :$PORT) ==="
SRV_LOG="$ROOT/build/serve-bench.log"
"$BIN/q36-serve" --model-dir "$MODEL" --port "$PORT" > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup_srv() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
trap cleanup_srv EXIT

ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  log "ERROR: q36-serve failed to become ready"
  cat "$SRV_LOG" | tee -a "$OUT"
  exit 1
fi

log "--- /health ---"
curl -s "http://127.0.0.1:$PORT/health" | tee -a "$OUT"; echo "" | tee -a "$OUT"

log "--- HTTP Short Factual Prose (max_tokens=32) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b-fp8","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}' | tee -a "$OUT"; echo "" | tee -a "$OUT"

log "--- HTTP Long-Form Prose Decode (max_tokens=256) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b-fp8","messages":[{"role":"user","content":"Write a clear, detailed technical essay explaining how unified-memory superchips couple CPU cores and GPU streaming multiprocessors over coherent interconnects, covering memory bandwidth ceilings, L2 cache locality, and autoregressive prose decoding latency."}],"max_tokens":256,"temperature":0}' | tee -a "$OUT"; echo "" | tee -a "$OUT"

log "--- HTTP 1600-tok Needle-in-a-Haystack Recall ---"
python3 - "$PORT" << "PYEOF" | tee -a "$OUT"
import json, sys, urllib.request
port = sys.argv[1]
filler = ("The archive records routine telemetry from orbital relay stations across sectors one through nine, noting nominal thermal margins, stable optical downlink framing, and periodic solar array calibration sweeps without anomaly. " * 42)
prompt = filler + "CRITICAL RECORD: Captain Elena Vance authorized deployment code Aurora-742 at station Omega-9. " + filler + "Question: Who authorized deployment code Aurora-742, and at which station? Reply concisely."
body = json.dumps({
    "model": "qwen3.6-35b-a3b-fp8",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 48,
    "temperature": 0
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode())
PYEOF

log "--- /health after HTTP runs ---"
curl -s "http://127.0.0.1:$PORT/health" | tee -a "$OUT"; echo "" | tee -a "$OUT"

cleanup_srv
trap - EXIT
sleep 8

log "=== 11. 128K CONTEXT BOUNDARY + SELF-TEST (q36-serve --self-test) ==="
"$BIN/q36-serve" --model-dir "$MODEL" --self-test 2>&1 | tee -a "$OUT"

log "=== DONE ==="
