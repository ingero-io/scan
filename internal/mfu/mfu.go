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

// gpuPeak is one SKU's DENSE tensor peak, per GPU, at each compute
// precision the part supports. Every figure is dense (no 2:4 structured
// sparsity) because the numerator this divides is a dense 2N-per-token
// FLOPs count; mixing a sparsity-inflated peak into that ratio would
// halve every reported MFU.
//
// Sparsity headlines are the standing trap in vendor datasheets. NVIDIA
// prints "dense | sparse*" pairs with the sparse figure marked by an
// asterisk, and AMD quotes an FP4 number for CDNA4 that is four times
// its own FP8 dense figure because it assumes 2:4 sparsity. Only the
// left-hand, unasterisked, dense value belongs here. A second trap sits
// on H100 specifically: its BF16 SPARSE figure and its FP8 DENSE figure
// are both 1979, so a lookup that reached 1979 by way of sparsity would
// look correct on H100 and be wrong on every other part.
//
// A zero means the part has NO tensor hardware for that precision, not
// that the figure is unknown. Callers must treat zero as "unsupported"
// and fall back rather than divide by it.
type gpuPeak struct {
	// token is the distinctive lowercase substring matched against the
	// GPU model string (`nvidia-smi --query-gpu=name`, or the market
	// name amd-smi reports). Match is longest-first.
	token string
	// family is the coarse model name used for display and as the join
	// key against the hourly-rate tables. Form-factor variants of one
	// model share a family so splitting a row by form factor does not
	// fragment that key.
	family string
	// formFactor states the physical variant the figures describe, so no
	// row's convention has to be inferred from its neighbours.
	formFactor string

	bf16 float64 // dense BF16/FP16 tensor TFLOP/s
	fp8  float64 // dense FP8 tensor TFLOP/s; 0 = no FP8 tensor hardware
	fp4  float64 // dense FP4 tensor TFLOP/s; 0 = no FP4 tensor hardware
	int8 float64 // dense INT8 tensor TOPS (operations, not FLOPs); 0 = unsupported
}

// gpuPeakTFLOPS carries vendor dense-tensor spec, rounded. ESTIMATE:
// real sustained peak is below spec on every part.
//
// NVIDIA figures are the dense column of each part's datasheet
// specifications table (the value left of the "|", with the asterisked
// sparse value discarded): A10 125 | 250*, A10G 70 | 140*, L40S 362.05
// | 733*, L4 121 dense against its 242 sparse headline. Blackwell B200
// is 2250 dense BF16, 4500 dense FP8 and 9000 dense FP4, each half of
// the sparse figure NVIDIA leads with. Ampere A100 and earlier have no
// FP8 tensor hardware at all, and Turing T4 and Volta V100 have neither
// FP8 nor BF16, so those cells are zero rather than extrapolated.
//
// AMD figures are AMD's own published dense numbers. CDNA3 (MI300X, and
// MI325X which is the same compute with more memory) is 1307.4 dense
// FP16/BF16 and 2614.9 dense FP8, with no FP4 hardware. CDNA4 doubles
// FP8 against FP16 and doubles FP4 again against FP8, so MI355X is 2500
// / 5000 / 10000 and MI350X, the same die air-cooled at a lower clock,
// is 2300 / 4600 / 9200. MI350X and MI355X must not be collapsed. AMD's
// widely quoted 20.1 PFLOPS FP4 headline for MI355X is that part's
// dense FP4 with 2:4 sparsity applied and is deliberately not used here.
var gpuPeakTFLOPS = []gpuPeak{
	{token: "b200", family: "b200", formFactor: "SXM", bf16: 2250, fp8: 4500, fp4: 9000, int8: 4500},
	{token: "h200", family: "h200", formFactor: "SXM5", bf16: 989, fp8: 1979, int8: 1979},
	{token: "h100 pcie", family: "h100", formFactor: "PCIe", bf16: 756, fp8: 1513, int8: 1513},
	{token: "h100", family: "h100", formFactor: "SXM5", bf16: 989, fp8: 1979, int8: 1979},
	{token: "gh200", family: "gh200", formFactor: "Grace-Hopper superchip", bf16: 989, fp8: 1979, int8: 1979},
	{token: "l40s", family: "l40s", formFactor: "PCIe", bf16: 362.05, fp8: 733, int8: 733},
	{token: "a100", family: "a100", formFactor: "SXM4 and PCIe (80GB and 40GB identical compute)", bf16: 312, int8: 624},
	{token: "a10g", family: "a10g", formFactor: "PCIe (AWS variant, lower tensor throughput than the datacenter A10)", bf16: 70, int8: 140},
	{token: "a10", family: "a10", formFactor: "PCIe", bf16: 125, int8: 250},
	{token: "l4", family: "l4", formFactor: "PCIe", bf16: 121, fp8: 242, int8: 242},

	// V100 predates BF16 and FP8; the figure is FP16 tensor. The two form
	// factors differ by 10.4% and nvidia-smi always names the variant
	// ("Tesla V100-SXM2-32GB", "Tesla V100-PCIE-16GB"), so both resolve
	// exactly and the bare token is only reached when an operator passes
	// "v100" by hand.
	{token: "v100-sxm2", family: "v100", formFactor: "SXM2", bf16: 125},
	{token: "v100-pcie", family: "v100", formFactor: "PCIe", bf16: 112},
	{token: "v100", family: "v100", formFactor: "PCIe assumed (form factor not stated in the model string)", bf16: 112},

	{token: "t4", family: "t4", formFactor: "PCIe", bf16: 65, int8: 130},

	{token: "mi355x", family: "mi355x", formFactor: "OAM, liquid-cooled 1400W", bf16: 2500, fp8: 5000, fp4: 10000, int8: 5000},
	{token: "mi350x", family: "mi350x", formFactor: "OAM, air-cooled 1000W", bf16: 2300, fp8: 4600, fp4: 9200, int8: 4600},
	{token: "mi325x", family: "mi325x", formFactor: "OAM, 256GB HBM3E", bf16: 1307, fp8: 2615, int8: 2615},
	{token: "mi300x", family: "mi300x", formFactor: "OAM, 192GB HBM3", bf16: 1307, fp8: 2615, int8: 2615},
}

// peakFor returns the dense peak for one precision, and whether the part
// has hardware for it at all.
func (g gpuPeak) peakFor(p Precision) (float64, bool) {
	var v float64
	switch p {
	case PrecisionFP8:
		v = g.fp8
	case PrecisionFP4:
		v = g.fp4
	case PrecisionINT8:
		v = g.int8
	default:
		v = g.bf16
	}
	return v, v > 0
}

// init sorts the GPU matching to longest-token-first so a substring
// collision (gh200 contains h200, l40s contains l4, a10g contains
// a10, v100-sxm2 contains v100) resolves to the specific model, not
// the shorter token that happens to be a substring.
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
	// GPUUtilPct is the live vendor-reported GPU utilization (0..100)
	// observed over the sampling window - the dashboard number scan contrasts
	// MFU against. Nil means it could not be read, in which case no
	// utilization is claimed rather than a fake one printed.
	GPUUtilPct *float64
	// UtilTool names the tool that produced GPUUtilPct (nvidia-smi on
	// NVIDIA, amd-smi on ROCm) so the report attributes the number to the
	// operator's own instrument instead of naming the wrong vendor's tool on
	// their host. Empty falls back to a vendor-neutral phrase.
	UtilTool string

	// Precision is the COMPUTE precision the engine serves in, which selects
	// the peak the denominator divides by. PrecisionUnknown falls back to the
	// BF16 peak and forces an explicit "assumed" label onto the output,
	// because an unlabelled fallback reports an FP8 workload at twice its
	// real MFU.
	Precision Precision

	// DeviceAttribution says where GPUCount came from. The denominator
	// multiplies the per-GPU peak by GPUCount, which silently asserts that
	// this engine owns every device counted. A host-wide device enumeration
	// does not support that assertion: a tensor-parallel-1 replica on an
	// 8-GPU box is otherwise measured against eight GPUs' worth of hardware
	// it cannot touch.
	DeviceAttribution DeviceAttribution

	// MIGEnabled marks a host whose GPUs are partitioned into MIG instances.
	// A MIG-confined engine holds a fraction of a board's SMs while the host
	// still enumerates one row per PHYSICAL GPU, so the whole-board peak is
	// simply the wrong denominator.
	MIGEnabled bool
}

// DeviceAttribution records how confidently the GPU count in the
// denominator can be attributed to the engine being measured.
type DeviceAttribution string

const (
	// DevicesUnattributed is a host-wide device enumeration that was NOT
	// confirmed to describe this engine's allocation. It is the honest
	// default: counting every GPU on the box overstates the denominator,
	// which understates MFU, which overstates reclaimable headroom, which
	// recommends consolidating capacity that was never idle.
	DevicesUnattributed DeviceAttribution = ""
	// DevicesEngineDeclared came from the engine's own parallelism
	// configuration (tensor and pipeline parallel width, shard count), so it
	// describes the devices this replica actually spans.
	DevicesEngineDeclared DeviceAttribution = "engine-declared"
	// DevicesOperatorSet came from an explicit operator override.
	DevicesOperatorSet DeviceAttribution = "operator-set"
)

// attributed reports whether the device count can be tied to the engine.
func (d DeviceAttribution) attributed() bool {
	return d == DevicesEngineDeclared || d == DevicesOperatorSet
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
	GPUUtilPct     *float64 // live vendor-reported utilization 0..100; nil if unread
	UtilTool       string   // tool GPUUtilPct came from (nvidia-smi, amd-smi)

	// Precision is the compute precision whose dense peak was actually used
	// as the denominator. PrecisionAssumed marks it as a fallback rather than
	// a detection, so a reader can tell a measured divisor from a guessed one.
	Precision        Precision
	PrecisionAssumed bool

	// DeviceAttribution records whether PeakTFLOPS rests on a device count
	// tied to this engine or on a host-wide enumeration.
	DeviceAttribution DeviceAttribution

	HourlyUSDPerGPU float64
	MonthlyUSD      float64 // GPUCount * hourly * 730

	// DollarEnvelopeSuppressed is set when the headroom dollar figures are
	// withheld because the denominator rests on an unverified device
	// attribution. A dollar amount an operator is meant to act on implies a
	// precision the input does not have, so it is omitted.
	DollarEnvelopeSuppressed bool

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

// ResolvePeakTFLOPS returns the per-GPU DENSE peak for a GPU model
// string at a given compute precision, matching the longest distinctive
// token the model string contains.
//
// When the part has no tensor hardware for the requested precision (FP8
// on an A100, FP4 on anything before Blackwell or CDNA4) it returns the
// BF16 peak and reports supported=false, so the caller can say the
// precision and the hardware disagree instead of dividing by zero or by
// a peak the silicon cannot reach.
func ResolvePeakTFLOPS(gpu string, p Precision) (peak float64, supported bool, err error) {
	g := strings.ToLower(strings.TrimSpace(gpu))
	if g == "" {
		return 0, false, fmt.Errorf("gpu model is empty")
	}
	for _, e := range gpuPeakTFLOPS {
		if strings.Contains(g, e.token) {
			if v, ok := e.peakFor(p); ok {
				return v, true, nil
			}
			return e.bf16, false, nil
		}
	}
	return 0, false, fmt.Errorf("unknown GPU model %q: pass a known model or extend the peak table", gpu)
}

// ResolvePeakBF16TFLOPS returns the per-GPU dense BF16/FP16 peak. It is
// the precision-independent entry point for callers doing BF16-anchored
// hardware arithmetic rather than dividing a serving workload's achieved
// FLOPs.
func ResolvePeakBF16TFLOPS(gpu string) (float64, error) {
	peak, _, err := ResolvePeakTFLOPS(gpu, PrecisionBF16)
	return peak, err
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
	// A MIG-partitioned host enumerates one row per PHYSICAL GPU while the
	// engine is confined to a slice of one board's SMs. No scaling of the
	// whole-board peak is defensible without knowing which profile the
	// engine holds, so the estimate is refused rather than reported against
	// hardware the engine does not have. An operator who knows the slice can
	// still measure it by passing an explicit device count.
	if in.MIGEnabled && !in.DeviceAttribution.attributed() {
		return Estimate{}, fmt.Errorf("MIG is enabled on this host: the per-GPU peak describes a whole board, not the slice this engine holds; pass an explicit GPU count attributed to the engine")
	}
	params, err := ResolveParams(in.Model, in.ParamsOverride)
	if err != nil {
		return Estimate{}, err
	}
	// Resolve the denominator at the precision the engine actually computes
	// in. An undetected precision falls back to BF16 and is labelled; it is
	// never silently treated as a detection.
	precision := in.Precision
	precisionAssumed := precision == PrecisionUnknown
	if precisionAssumed {
		precision = PrecisionBF16
	}
	peakPer, precisionSupported, err := ResolvePeakTFLOPS(in.GPU, precision)
	if err != nil {
		return Estimate{}, err
	}

	achieved := in.TokensPerSec * flopsPerParamPerToken * params / 1e12 // TFLOP/s
	peak := peakPer * float64(in.GPUCount)
	mfu := achieved / peak

	est := Estimate{
		Model:             in.Model,
		GPU:               in.GPU,
		GPUCount:          in.GPUCount,
		TokensPerSec:      in.TokensPerSec,
		ParamsActive:      params,
		AchievedTFLOPS:    achieved,
		PeakTFLOPS:        peak,
		MFU:               mfu,
		Plausible:         mfu <= 1.0,
		GPUUtilPct:        in.GPUUtilPct,
		UtilTool:          in.UtilTool,
		HourlyUSDPerGPU:   in.HourlyUSDPerGPU,
		Precision:         precision,
		PrecisionAssumed:  precisionAssumed,
		DeviceAttribution: in.DeviceAttribution,
	}
	// Headroom envelope (MFU-only; independent of rate), anchored to the healthy
	// band and clamped to [0,1]. fracLow uses the LOWER healthy ref (smaller gap
	// -> smaller, more conservative headroom); fracHigh uses the upper ref.
	if mfu > 0 {
		est.HeadroomMultiple = ((healthyServingMFULow + healthyServingMFUHigh) / 2) / mfu
	}
	est.HeadroomFracLow = math.Max(0, math.Min(1, 1-mfu/healthyServingMFULow))
	est.HeadroomFracHigh = math.Max(0, math.Min(1, 1-mfu/healthyServingMFUHigh))
	// A device count that was never attributed to this engine makes the
	// headroom dollars unsafe to print: over-counting devices inflates the
	// denominator, which deflates MFU, which inflates the headroom fraction,
	// which recommends reclaiming capacity that is not idle. The spend line
	// stays (it is what the counted GPUs cost either way); the actionable
	// reclaim figure is withheld. A single counted GPU needs no attribution,
	// since one device is the whole host.
	unattributedMultiGPU := in.GPUCount > 1 && !in.DeviceAttribution.attributed()
	est.DollarEnvelopeSuppressed = unattributedMultiGPU
	if in.HourlyUSDPerGPU > 0 {
		est.MonthlyUSD = in.HourlyUSDPerGPU * float64(in.GPUCount) * hoursPerMonth
		if !est.DollarEnvelopeSuppressed {
			est.MonthlyHeadroomLowUSD = est.HeadroomFracLow * est.MonthlyUSD
			est.MonthlyHeadroomHighUSD = est.HeadroomFracHigh * est.MonthlyUSD
		}
	}

	// Label the divisor before anything else, because it is the assumption
	// most likely to be wrong and the one a reader is least likely to check.
	if precisionAssumed {
		est.Caveats = append(est.Caveats, "precision=assumed-bf16 (not detected): serving precision could not be read from the engine, so the denominator is the dense BF16 peak; FP8 serving would halve this MFU and FP4 would quarter it")
	} else {
		est.Caveats = append(est.Caveats, fmt.Sprintf("precision=%s (detected): denominator is the dense %s peak", precision, precision))
	}
	if !precisionSupported {
		est.Caveats = append(est.Caveats, fmt.Sprintf("%s serving was detected but %s has no %s tensor hardware; the denominator fell back to the dense BF16 peak and the inputs disagree", precision, in.GPU, precision))
	}
	if precision == PrecisionINT8 {
		est.Caveats = append(est.Caveats, "INT8 peak is quoted in TOPS, not TFLOPS: this ratio is model OPERATIONS against peak operations, not floating-point work")
	}
	if unattributedMultiGPU {
		est.Caveats = append(est.Caveats, fmt.Sprintf("device count %d is a host-wide enumeration not attributed to this engine: a replica narrower than the host is measured against GPUs it cannot use, so MFU is a LOWER bound and the headroom dollar envelope is withheld", in.GPUCount))
	}
	if in.ParamsOverride > 0 {
		est.Caveats = append(est.Caveats, "params supplied explicitly; FLOPs/token = 2 x params (dense forward approximation)")
	}
	if strings.Contains(strings.ToLower(in.Model), "mixtral") ||
		strings.Contains(strings.ToLower(in.Model), "moe") ||
		strings.Contains(strings.ToLower(in.Model), "deepseek-v2") {
		est.Caveats = append(est.Caveats, "MoE model: estimate uses ACTIVE params/token; verify the active-param count for your variant")
	}
	// Retained as a backstop only. This check cannot be the primary guard on
	// the denominator: a workload whose divisor is wrong by 2x usually lands
	// at a plausible-looking figure well under 100%, which passes here and is
	// read as actionable. The precision and device-attribution labels above
	// are what make a wrong divisor visible.
	if !est.Plausible {
		est.Caveats = append(est.Caveats, "MFU > 100%: inputs are inconsistent (wrong params, wrong GPU, or a serving precision below the resolved denominator) - check model size and the detected precision")
	}
	return est, nil
}
