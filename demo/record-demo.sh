#!/usr/bin/env bash
# record-demo.sh - the tight, asciinema-friendly money shot. Assumes vLLM is
# ALREADY serving under load (run reproduce.sh first, or its server+loadgen
# steps). Shows the contrast that is the whole point: the GPU reports ~full
# utilization while real MFU is a fraction of it.
#
# Record ONLY via:  asciinema rec --title scan --command ./record-demo.sh scan-demo.cast
# (--command records just this script - no interactive prompt, hostname, or history leaks.)
set -uo pipefail
EP="${EP:-http://localhost:8001}"
NAME="${NAME:-Qwen2-7B-Instruct}"
RATE="${RATE:-1.10}"
GPU="${GPU:-NVIDIA A100-SXM4-40GB}"   # bake explicit so the take never depends on auto-detect
GPUCOUNT="${GPUCOUNT:-1}"
say(){ printf '\n\033[1;36m# %s\033[0m\n' "$*"; sleep 1.5; }

say "The GPU looks busy - nvidia-smi reports near-100% utilization:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
sleep 2

say "But utilization is not useful work. scan measures the MFU gap:"
sleep 1
"${SCAN:-./scan}" --endpoint "$EP/metrics" --model qwen2-7b --gpu "$GPU" --gpu-count "$GPUCOUNT" --rate "$RATE"
sleep 3
