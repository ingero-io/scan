#!/usr/bin/env python3
"""loadgen.py - realistic open-loop load generator for an OpenAI-compatible
LLM serving endpoint (vLLM / SGLang). Used by the scan demo to put a server
under steady, production-shaped traffic so the MFU gap is measured on a real
load rather than an idle one.

OPEN-LOOP Poisson arrivals + a realistic lognormal prompt/output length mix, so
the engine's effective batch EMERGES from the traffic rather than being forced
with a --max-num-seqs knob.

Design notes:
- Arrivals: Poisson (exponential inter-arrival at --qps), open-loop: a slow
  response does NOT delay the next arrival (closed-loop drivers self-throttle
  and hide queueing pain).
- Request mix: lognormal prompt and output lengths (chat-ish defaults), each
  request samples its own lengths. No ignore_eos with a fixed short prompt.
- Routing: client-side round-robin across --endpoints. Equivalent to a v1
  router for this experiment; the simplification is disclosed in the summary.
- SLO: declared up front (--slo-p95-ms / --slo-p99-ms) and judged in the
  summary; a declared SLO that the load breaks is reported as a miss.
- Output: one CSV row per request (latency log) + one JSON summary with
  p50/p95/p99, achieved QPS, token counts, SLO verdict.

The driver only needs the OpenAI-compatible /v1/completions endpoint vLLM and
SGLang both expose. Pure stdlib (asyncio + urllib in threads) so the GPU host
needs no extra pip installs.
"""

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

WORDS = (
    "system latency throughput batch decode prefill tensor kernel memory "
    "schedule replica cluster token model serve route queue stall measure "
    "window report metric trace probe sample budget margin signal phase"
).split()


def build_prompt(n_tokens: int) -> str:
    # ~1 word ~= 1.3 tokens for common tokenizers; aim under so prompts do not
    # blow past the model context. Exactness is irrelevant - the MIX is what
    # matters, and prompt_tokens_total reports the real number.
    n_words = max(1, int(n_tokens / 1.3))
    return " ".join(random.choices(WORDS, k=n_words))


def sample_lognormal(median: float, sigma: float, lo: int, hi: int) -> int:
    v = int(random.lognormvariate(math.log(median), sigma))
    return max(lo, min(hi, v))


async def one_request(endpoint: str, model: str, prompt_tokens: int,
                      output_tokens: int, timeout: float):
    body = json.dumps({
        "model": model,
        "prompt": build_prompt(prompt_tokens),
        "max_tokens": output_tokens,
        # No ignore_eos: the output-length distribution is shaped by max_tokens
        # but real EOS behavior stays in play (spec: realistic mix, not forced).
        "temperature": 0.7,
    }).encode()

    def blocking():
        req = urllib.request.Request(
            endpoint + "/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    t0 = time.monotonic()
    try:
        data = await asyncio.to_thread(blocking)
        latency = time.monotonic() - t0
        usage = data.get("usage", {})
        return {
            "ok": True, "latency_s": latency, "endpoint": endpoint,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ConnectionError, json.JSONDecodeError, OSError) as e:
        return {"ok": False, "latency_s": time.monotonic() - t0,
                "endpoint": endpoint, "error": type(e).__name__,
                "prompt_tokens": 0, "completion_tokens": 0}


async def run(args) -> int:
    endpoints = [e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]
    if not endpoints:
        print("no endpoints", file=sys.stderr)
        return 2
    random.seed(args.seed)

    # TRUE open loop: each request blocks a worker thread (urllib is sync), and
    # asyncio.to_thread uses the loop's default ThreadPoolExecutor, which defaults
    # to only ~min(32, cpu+4) workers. That silently caps in-flight requests at
    # ~32 and makes a high --qps run behave CLOSED-loop at 32 concurrency (the
    # server never saturates, the queue never builds). Raise the pool well above
    # the peak in-flight we expect (qps x latency) so the load is genuinely open.
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=args.max_inflight))

    results = []
    tasks = []
    rr = 0
    t_start = time.monotonic()
    deadline = t_start + args.duration

    async def fire(ep):
        p = sample_lognormal(args.prompt_median, args.prompt_sigma, 8, args.prompt_max)
        o = sample_lognormal(args.output_median, args.output_sigma, 4, args.output_max)
        results.append(await one_request(ep, args.model, p, o, args.timeout))

    # Open loop: sleep exponential inter-arrival, fire-and-forget each request.
    while time.monotonic() < deadline:
        await asyncio.sleep(random.expovariate(args.qps))
        if time.monotonic() >= deadline:
            break
        tasks.append(asyncio.create_task(fire(endpoints[rr % len(endpoints)])))
        rr += 1

    if tasks:
        await asyncio.gather(*tasks)
    wall = time.monotonic() - t_start

    ok = [r for r in results if r["ok"]]
    lat = sorted(r["latency_s"] for r in ok)

    def pct(p):
        # Nearest-rank percentile: index ceil(p/100*N)-1. The naive
        # int(p/100*N) overstates by one rank (p99 of 100 samples would read
        # the max, i.e. p100) - and these numbers drive the SLO verdict.
        if not lat:
            return 0.0
        idx = max(0, math.ceil(p / 100 * len(lat)) - 1)
        return lat[min(len(lat) - 1, idx)]

    p95, p99 = pct(95), pct(99)
    summary = {
        "regime": args.regime,
        "endpoints": endpoints,
        "router": "client-side round-robin (v1 simplification, disclosed)",
        "arrivals": f"poisson qps={args.qps} open-loop seed={args.seed}",
        "length_mix": {
            "prompt": f"lognormal(median={args.prompt_median}, sigma={args.prompt_sigma}, max={args.prompt_max})",
            "output": f"lognormal(median={args.output_median}, sigma={args.output_sigma}, max={args.output_max})",
        },
        "wall_s": round(wall, 1),
        "sent": len(results), "ok": len(ok), "errors": len(results) - len(ok),
        "achieved_qps": round(len(results) / wall, 3) if wall > 0 else 0,
        "latency_s": {
            "p50": round(pct(50), 3), "p95": round(p95, 3), "p99": round(p99, 3),
            "mean": round(statistics.fmean(lat), 3) if lat else 0,
        },
        "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "completion_tokens": sum(r["completion_tokens"] for r in ok),
        "slo": {
            "p95_ms": args.slo_p95_ms, "p99_ms": args.slo_p99_ms,
            "met": (p95 * 1000 <= args.slo_p95_ms) and (p99 * 1000 <= args.slo_p99_ms),
        },
    }

    if args.latency_csv:
        with open(args.latency_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ok", "latency_s", "endpoint", "prompt_tokens",
                "completion_tokens", "error"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})

    out = json.dumps(summary, indent=2)
    if args.summary:
        with open(args.summary, "w") as f:
            f.write(out + "\n")
    print(out)
    # Exit code carries the SLO verdict so the orchestrator can branch on it.
    return 0 if summary["slo"]["met"] else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", required=True,
                    help="comma-separated replica base URLs (http://host:port)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--qps", type=float, required=True,
                    help="TOTAL Poisson arrival rate across all endpoints")
    ap.add_argument("--duration", type=float, default=300, help="seconds")
    ap.add_argument("--regime", default="", help="label: over-provisioned|well-loaded|kv-bound")
    ap.add_argument("--prompt-median", type=int, default=180)
    ap.add_argument("--prompt-sigma", type=float, default=0.8)
    ap.add_argument("--prompt-max", type=int, default=2048)
    ap.add_argument("--output-median", type=int, default=140)
    ap.add_argument("--output-sigma", type=float, default=0.7)
    ap.add_argument("--output-max", type=int, default=1024)
    ap.add_argument("--slo-p95-ms", type=float, default=8000)
    ap.add_argument("--slo-p99-ms", type=float, default=15000)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-inflight", type=int, default=4096,
                    help="open-loop concurrency ceiling (thread pool size); must exceed peak qps x latency or the load silently becomes closed-loop")
    ap.add_argument("--latency-csv", default="")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
