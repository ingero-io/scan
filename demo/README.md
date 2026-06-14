# scan demo: the MFU gap on a real load

`scan` reads a serving engine's `/metrics` plus the GPU model and reports **MFU**
(model FLOPs utilization) - the work the GPU is actually getting out, versus what
"utilization" claims. This demo puts a real vLLM server under realistic open-loop
traffic and runs `scan` against it, so the number is measured on a live load, not
an idle box.

## Run it

```
./reproduce.sh                      # Qwen2-7B on :8001, default load (QPS=6)
QPS=6 RATE=1.10 ./reproduce.sh      # tune the load / your GPU $-rate
```

One Linux GPU box is enough - a cheap datacenter GPU (L4 / A10 / A100). You need
an NVIDIA driver, Python 3.10+, and the `scan` binary (build: `go build -o scan ./cmd/scan`). `loadgen.py` is
pure stdlib (no pip installs on the host).

The run prints the scan output **and the load's own SLO verdict**, e.g.
`SLO: p95 7.33s (target 8s), 11.64s p99 (target 15s), 0 errors / 1536 reqs -> met=True`.
That is the point: a met SLO means the MFU gap is waste on a healthy server, not a
box already failing latency. The recorded take is QPS=6 on a single A100 - ~90%
reported util, SLO met, ~4% MFU.

## What you'll see

```
$ ./scan --endpoint http://localhost:8001/metrics --model qwen2-7b --rate 1.10
GPU MFU scan (ESTIMATE - modeled, not measured)
  workload : qwen2-7b on 1x NVIDIA A100-SXM4-40GB
  output   : ... tokens/sec  ->  ... of ... TFLOP/s used
  util     : ~90%   (live nvidia-smi GPU utilization - what the dashboard shows)
  MFU      : ~X%   (the real work behind that utilization)
  headroom : ~Nx below well-batched serving (~35-50% MFU, public benchmark)
  cost     : ~$.../mo at $1.10/GPU/hr (--rate, upper bound)
  envelope : up to ~$..-$../mo of consolidation headroom (CEILING, not a promise)
  ...
```

The headroom **envelope is a ceiling** - the gross gap between this workload and a
well-batched replica, anchored to published serving benchmarks (~35-50% MFU), not
a savings promise. How much is *safely* recoverable without breaking your SLO,
which cause is responsible, and a signed before/after receipt: that is the Ingero
agent, not this meter. `scan` shows the gap; the agent recovers it.

## Honest by construction

Every number `scan` prints is an **estimate**: MFU is modeled from a published-spec
peak and a 2N-FLOPs-per-token approximation, not measured from the kernel. It is
meant to show the gap is real and large enough to care about. `scan` never claims a
cause or a recoverable figure - those need measurement the meter does not do.
