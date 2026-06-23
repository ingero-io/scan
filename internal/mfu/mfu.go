// Package mfu computes the MFU gap, the difference between the work a
// GPU could do at peak and the work an inference server is actually
// getting out of it, from numbers an operator already exposes (serving
// throughput and the GPU model). No eBPF, no root.
//
// IMPORTANT: every number this package produces is an ESTIMATE. MFU
// here is MODELED from a published-spec peak and a 2N-per-token FLOPs
// approximation, not measured from the kernel. It is meant to show the
// gap exists and is large enough to care about. Treat the output as
// "you are in the ballpark of X% MFU," not a certified figure. Known
// sources of error are documented on the tables below.
package mfu

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// flopsPerParamPerToken is the standard approximation for a
// transformer forward pass: ~2 FLOPs per parameter per token (Kaplan
// et al. 2020; Chinchilla). Training is ~6N; inference decode is ~2N.
// This ignores the attention term (small for typical context vs model
// size) and treats the model as dense, so it UNDERSTATES FLOPs for
// very long contexts and OVERSTATES achieved work for MoE models
// unless active (not total) params are used. Both are captured as
// caveats in Estimate.Caveats.
const flopsPerParamPerToken = 2.0

// hoursPerMonth is the 730-hour convention (365*24/12) used for
// monthly cost.
const hoursPerMonth = 730.0

// gpuPeakTFLOPS is dense (no 2:4 sparsity) BF16/FP16 tensor-core
// throughput in TFLOP/s per GPU, keyed by a distinctive lowercase
// token found in the GPU model string (e.g. `nvidia-smi
// --query-gpu=name`). Values are vendor dense-tensor spec, rounded.
// ESTIMATE: real sustained peak is lower than spec; FP8/INT8 workloads
// have higher peak than these BF16 numbers, so MFU is conservative
// (the gap is at least this large) for FP16/BF16 serving and
// OVERSTATED for FP8 serving. Match is longest-token-first.
var gpuPeakTFLOPS = []struct {
	token  string
	tflops float64
}{
	{"b200", 2250}, // Blackwell dense BF16 (approx)
	{"h200", 989},  // same compute as H100
	{"h100", 989},  // SXM5 dense BF16/FP16 (~989.5); PCIe is ~756, treated as upper bound
	{"gh200", 989}, // Grace-Hopper: H100 GPU
	{"l40s", 362},  // dense BF16
	{"a100", 312},  // SXM/PCIe dense BF16 (80GB and 40GB identical compute)
	{"a10g", 70},   // AWS A10G: GA102 variant, lower tensor throughput than the datacenter A10 (125)
	{"a10", 125},
	{"l4", 121},   // Ada L4 dense BF16
	{"v100", 112}, // FP16 tensor (no BF16)
	{"t4", 65},    // Turing FP16 tensor
}

// init sorts the GPU matching to longest-token-first so a substring
// collision (gh200 contains h200, l40s contains l4, a10g contains
// a10) resolves to the specific model, not the shorter token that
// happens to be a substring.
func init() {
	sort.SliceStable(gpuPeakTFLOPS, func(i, j int) bool {
		return len(gpuPeakTFLOPS[i].token) > len(gpuPeakTFLOPS[j].token)
	})
}

// modelParamsActive is ACTIVE non-embedding parameters per forward
// token, keyed by a distinctive lowercase token in the model name. For
// dense models this is total params; for MoE (mixtral, etc.) it is
// ACTIVE params per token, which is what determines FLOPs/token.
// ESTIMATE: approximate published sizes. Unknown models must pass an
// explicit --params override. Match is longest-token-first so
// "llama-3.1-405b" wins over "llama".
var modelParamsActive = []struct {
	token  string
	params float64
}{
	{"llama-3.1-405b", 405e9},
	{"llama-3-70b", 70e9},
	{"llama3-70b", 70e9},
	{"llama-3.1-8b", 8e9},
	{"llama-3-8b", 8e9},
	{"llama3-8b", 8e9},
	{"qwen2-72b", 72e9},
	{"qwen2.5-72b", 72e9},
	{"qwen2-7b", 7e9},
	{"mixtral-8x22b", 39e9}, // MoE: ~39B active of ~141B total
	{"mixtral-8x7b", 13e9},  // MoE: ~13B active of ~47B total
	{"mistral-7b", 7e9},
	{"gemma-2-27b", 27e9},
	{"gemma-2-9b", 9e9},
	{"phi-3-mini", 3.8e9},
	{"falcon-40b", 40e9},
	{"deepseek-v2", 21e9}, // MoE: ~21B active
}

// Input is the operator-supplied context for one estimate. TokensPerSec
// is the achieved OUTPUT (generation) token throughput aggregated
// across GPUCount GPUs. ParamsOverride, when > 0, wins over the model
// table (required for an unknown model). HourlyUSDPerGPU drives the
// cost lines; 0 leaves them zero.
type Input struct {
	Model           string
	ParamsOverride  float64
	GPU             string
	GPUCount        int
	TokensPerSec    float64
	HourlyUSDPerGPU float64
	// GPUUtilPct is the live nvidia-smi GPU utilization (0..100) observed
	// over the sampling window - the dashboard number scan contrasts MFU
	// against. Nil means it could not be read, in which case no utilization
	// is claimed rather than a fake one printed.
	GPUUtilPct *float64
}

// Estimate is the computed MFU gap and its dollar framing. All fields
// are estimates; Caveats lists the modeling assumptions that most
// affect accuracy for this input.
type Estimate struct {
	Model          string
	GPU            string
	GPUCount       int
	TokensPerSec   float64
	ParamsActive   float64
	AchievedTFLOPS float64
	PeakTFLOPS     float64  // aggregate across GPUCount
	MFU            float64  // achieved / peak, 0..1 in the normal case
	Plausible      bool     // false if MFU > 1 (impossible: bad params/throughput/dtype)
	GPUUtilPct     *float64 // live nvidia-smi utilization 0..100; nil if unread

	HourlyUSDPerGPU float64
	MonthlyUSD      float64 // GPUCount * hourly * 730

	// Consolidation-headroom ENVELOPE - a CEILING, never a recovery promise.
	// The share of this workload's spend that could be reclaimed IF it matched a
	// well-batched replica's efficiency, anchored to a PUBLISHED healthy-serving
	// MFU band (healthyServingMFULow..High), NOT to 100% MFU (unreachable). It is
	// GROSS headroom (idle + stalled). How much is SAFELY recoverable without
	// breaking the SLO, and a signed before/after receipt, is the Ingero agent's
	// job - this meter only shows the gap.
	HeadroomMultiple       float64 // healthy-mid MFU / MFU: "~Nx below well-batched serving" (0 if MFU<=0)
	HeadroomFracLow        float64 // 1 - MFU/healthyServingMFULow,  clamped [0,1]
	HeadroomFracHigh       float64 // 1 - MFU/healthyServingMFUHigh, clamped [0,1]
	MonthlyHeadroomLowUSD  float64 // HeadroomFracLow  * MonthlyUSD (0 when no rate)
	MonthlyHeadroomHighUSD float64 // HeadroomFracHigh * MonthlyUSD (0 when no rate)

	Caveats []string
}

// Published healthy-serving MFU band for well-batched LLM inference. PUBLIC
// reference (independent serving benchmarks report well-run serving in roughly
// this range), used only to anchor the headroom envelope to something
// defensible instead of to an unreachable 100% MFU. NOT a calibration and
// encodes none of any SLO-aware sizing method.
//
// Exported so callers (the CLI report, the Prometheus exporter) can publish the
// band itself. These are the same public 0.35 / 0.50 reference; the unexported
// aliases below keep the existing internal call sites terse.
const (
	HealthyServingMFULow  = 0.35
	HealthyServingMFUHigh = 0.50

	healthyServingMFULow  = HealthyServingMFULow
	healthyServingMFUHigh = HealthyServingMFUHigh
)

// ResolvePeakTFLOPS returns the per-GPU dense BF16/FP16 peak for a GPU
// model string, matching the longest distinctive token it contains.
func ResolvePeakTFLOPS(gpu string) (float64, error) {
	g := strings.ToLower(strings.TrimSpace(gpu))
	if g == "" {
		return 0, fmt.Errorf("gpu model is empty")
	}
	for _, e := range gpuPeakTFLOPS {
		if strings.Contains(g, e.token) {
			return e.tflops, nil
		}
	}
	return 0, fmt.Errorf("unknown GPU model %q: pass a known model or extend the peak table", gpu)
}

// ResolveParams returns active params/token for a model, preferring an
// explicit override (> 0) over the model table.
func ResolveParams(model string, override float64) (float64, error) {
	if override > 0 {
		return override, nil
	}
	m := strings.ToLower(strings.TrimSpace(model))
	for _, e := range modelParamsActive {
		if strings.Contains(m, e.token) {
			return e.params, nil
		}
	}
	return 0, fmt.Errorf("unknown model %q: pass --params <active-params> (e.g. 8e9)", model)
}

// Compute estimates MFU and the dollar gap for one input. It validates
// the numeric inputs and resolves the model + GPU against the bundled
// tables (or the params override). The result is an estimate; see the
// package doc.
func Compute(in Input) (Estimate, error) {
	if in.TokensPerSec <= 0 {
		return Estimate{}, fmt.Errorf("tokens/sec must be > 0, got %g", in.TokensPerSec)
	}
	if in.GPUCount <= 0 {
		return Estimate{}, fmt.Errorf("gpu count must be > 0, got %d", in.GPUCount)
	}
	params, err := ResolveParams(in.Model, in.ParamsOverride)
	if err != nil {
		return Estimate{}, err
	}
	peakPer, err := ResolvePeakTFLOPS(in.GPU)
	if err != nil {
		return Estimate{}, err
	}

	achieved := in.TokensPerSec * flopsPerParamPerToken * params / 1e12 // TFLOP/s
	peak := peakPer * float64(in.GPUCount)
	mfu := achieved / peak

	est := Estimate{
		Model:           in.Model,
		GPU:             in.GPU,
		GPUCount:        in.GPUCount,
		TokensPerSec:    in.TokensPerSec,
		ParamsActive:    params,
		AchievedTFLOPS:  achieved,
		PeakTFLOPS:      peak,
		MFU:             mfu,
		Plausible:       mfu <= 1.0,
		GPUUtilPct:      in.GPUUtilPct,
		HourlyUSDPerGPU: in.HourlyUSDPerGPU,
	}
	// Headroom envelope (MFU-only; independent of rate), anchored to the healthy
	// band and clamped to [0,1]. fracLow uses the LOWER healthy ref (smaller gap
	// -> smaller, more conservative headroom); fracHigh uses the upper ref.
	if mfu > 0 {
		est.HeadroomMultiple = ((healthyServingMFULow + healthyServingMFUHigh) / 2) / mfu
	}
	est.HeadroomFracLow = math.Max(0, math.Min(1, 1-mfu/healthyServingMFULow))
	est.HeadroomFracHigh = math.Max(0, math.Min(1, 1-mfu/healthyServingMFUHigh))
	if in.HourlyUSDPerGPU > 0 {
		est.MonthlyUSD = in.HourlyUSDPerGPU * float64(in.GPUCount) * hoursPerMonth
		est.MonthlyHeadroomLowUSD = est.HeadroomFracLow * est.MonthlyUSD
		est.MonthlyHeadroomHighUSD = est.HeadroomFracHigh * est.MonthlyUSD
	}

	if in.ParamsOverride > 0 {
		est.Caveats = append(est.Caveats, "params supplied explicitly; FLOPs/token = 2 x params (dense forward approximation)")
	}
	if strings.Contains(strings.ToLower(in.Model), "mixtral") ||
		strings.Contains(strings.ToLower(in.Model), "moe") ||
		strings.Contains(strings.ToLower(in.Model), "deepseek-v2") {
		est.Caveats = append(est.Caveats, "MoE model: estimate uses ACTIVE params/token; verify the active-param count for your variant")
	}
	if !est.Plausible {
		est.Caveats = append(est.Caveats, "MFU > 100%: inputs are inconsistent (FP8/INT8 serving, wrong params, or wrong GPU) - check dtype and model size")
	}
	return est, nil
}
