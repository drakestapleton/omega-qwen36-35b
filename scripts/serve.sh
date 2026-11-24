#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Launch the pure seeded-Omega Zero-CUDA Qwen 3.6 35B-A3B OpenAI-compatible HTTP server.
# Defaults: Full-W4 + Live Arboris + 128K context window (pass --fp8 for Exact FP8, --mtp for Native MTP).
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN="${Q36_SERVE_BIN:-$ROOT/build/q36/q36-serve}"
MODEL_DIR="${Q36_MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0}"
PORT="${Q36_PORT:-18091}"
FLAGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --fp8) FLAGS="$FLAGS --fp8"; shift ;;
    --w4) shift ;;
    --mtp) FLAGS="$FLAGS --mtp"; shift ;;
    --batch) FLAGS="$FLAGS --batch"; shift ;;
    --spec) FLAGS="$FLAGS --spec"; shift ;;
    --self-test)
      # shellcheck disable=SC2086
      exec "$BIN" --model-dir "$MODEL_DIR" $FLAGS --self-test
      ;;
    -h|--help)
      echo "usage: sh scripts/serve.sh [--model-dir PATH] [--port PORT] [--w4|--fp8] [--mtp] [--batch] [--spec] [--self-test]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x "$BIN" ]; then
  echo "error: $BIN not found; run 'sh scripts/build.sh' first" >&2
  exit 1
fi

# shellcheck disable=SC2086
exec "$BIN" --model-dir "$MODEL_DIR" --port "$PORT" $FLAGS
