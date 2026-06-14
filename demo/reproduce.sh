#!/usr/bin/env bash
# reproduce.sh - stand up a vLLM server, put it under a realistic open-loop
# load, and run `scan` against it. Reproduces the demo end to end on a
# single Linux GPU box (a cheap datacenter GPU is plenty - L4 / A10 / A100).
#
# Prereqs: an NVIDIA GPU + driver, Python 3.10+, and the `scan` binary on PATH
# (build: `go build -o scan ./cmd/scan` from this repo, or `go install`).
# scan reads live GPU utilization from nvidia-smi over the sampling window; if the
# driver cannot report it, scan still prints the MFU gap and just omits the util line.
#
# vLLM is pinned for a reproducible take: `pip install vllm==0.10.2` in a venv
# (the known-good CUDA 12.8 recipe). Verify `--disable-log-requests` exists on your pin.
# If the server returns 500 on every request, pin the web stack vLLM 0.10.2 expects:
#   pip install fastapi==0.116.1 starlette==0.47.2 prometheus-fastapi-instrumentator==7.0.0
# (newer fastapi/starlette break the prometheus middleware vLLM mounts on /metrics.)
#
# Default load is QPS=6, the point where a 7B on one A100 stays visibly busy (~90%
# reported util) AND meets an 8s p95 SLO while MFU sits at ~4% - healthy-server waste,
# not an overloaded box. The run prints its own SLO verdict next to the scan output.
#
# Usage: ./reproduce.sh [MODEL] [PORT]
set -uo pipefail
MODEL="${1:-Qwen/Qwen2-7B-Instruct}"
NAME="${MODEL##*/}"
PORT="${2:-8001}"
EP="http://localhost:${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[demo] starting vLLM ($MODEL) on :$PORT ..."
vllm serve "$MODEL" --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --disable-log-requests > /tmp/vllm-demo.log 2>&1 &
VLLM_PID=$!
trap 'kill -9 $VLLM_PID $LOAD_PID 2>/dev/null' EXIT

echo "[demo] waiting for the model to load ..."
for _ in $(seq 1 150); do
  curl -fsS -m 4 "$EP/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -fsS -m 4 "$EP/v1/models" >/dev/null 2>&1 || { echo "[demo] FATAL: vLLM did not come up; see /tmp/vllm-demo.log"; exit 1; }

echo "[demo] applying open-loop load (Poisson, realistic length mix) ..."
SUMMARY=/tmp/loadgen-summary.json
rm -f "$SUMMARY"
python "$HERE/loadgen.py" --endpoints "$EP" --model "$MODEL" \
  --qps "${QPS:-6}" --duration "${DURATION:-90}" --slo-p95-ms 8000 --slo-p99-ms 15000 \
  --summary "$SUMMARY" > /tmp/loadgen-demo.log 2>&1 &
LOAD_PID=$!
sleep 20  # let the batch settle into steady state before measuring

echo "[demo] === scan ==="
# qwen2-7b is the built-in model-table token; pin it rather than deriving the name.
"${SCAN:-./scan}" --endpoint "$EP/metrics" --model qwen2-7b --rate "${RATE:-1.10}"

# Let the load run finish so its SLO verdict lands on disk, then report it next to
# the gap. A met SLO is the point: the gap above is waste on a HEALTHY server, not a
# server failing latency. (Persisting the summary is why --summary exists - a killed
# background driver loses its verdict otherwise.)
echo "[demo] waiting for the load run to finish so the SLO verdict is recorded ..."
wait "$LOAD_PID" 2>/dev/null
if [ -f "$SUMMARY" ]; then
  python - "$SUMMARY" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s, l = d["slo"], d["latency_s"]
print(f"[demo] SLO: p95 {l['p95']:.2f}s (target {s['p95_ms']/1000:.0f}s), "
      f"p99 {l['p99']:.2f}s (target {s['p99_ms']/1000:.0f}s), "
      f"{d['errors']} errors / {d['sent']} reqs -> met={s['met']}")
PY
else
  echo "[demo] (no load summary written; see /tmp/loadgen-demo.log)"
fi
